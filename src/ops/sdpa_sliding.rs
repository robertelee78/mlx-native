//! Sliding-window scaled dot-product attention host dispatch.
//!
//! Same as [`sdpa`](super::sdpa) but applies a sliding window mask: for each
//! query position `q_pos`, keys at positions `k_pos < q_pos - window_size` are
//! masked to negative infinity before softmax.  Combined with the causal mask,
//! the effective attention window is `[max(0, q_pos - window_size), q_pos]`.
//!
//! Used by Gemma 4 sliding-window layers (5 of every 6 layers, window=1024).
//!
//! # Status (2026-04-25): broken at prefill shapes — repair-or-remove TBD
//!
//! TODO(2026-04-25): remove or repair — see audit
//! `cfa-20260425-fix-audit-findings`.
//!
//! At prefill seq_len `S < window_size` this kernel should be mathematically
//! identical to plain [`sdpa`](super::sdpa) (every position is within the
//! window, so the window mask is a no-op). It is not — measured against the
//! plain-`sdpa` reference, this kernel emits the pad token at the first
//! decode step on a Gemma-4 sliding layer with `seq_len=576`,
//! `window_size=1024`. See `/opt/hf2q/docs/spike-gate-a-prefill.md:118-130`
//! for the reproducer, and `Finding 2 — Blocker: sdpa_sliding broken at
//! prefill shapes` for the gate-A blocker context.
//!
//! Two ADRs in hf2q reference this op while the repair decision is open:
//!   * `ADR-011-flash-attn-prefill.md` — the in-flight flash-attn-prefill
//!     port is the planned long-term replacement (one mask-driven kernel
//!     covering both global and sliding layers); this op is the
//!     byte-identical correctness baseline for that work.
//!   * `ADR-011-phase2-port-swa-mask.md` — explicitly plans to keep this op
//!     present for the decode path during the cutover.
//!
//! Live consumers in this crate: `tests/test_sdpa.rs` and
//! `tests/bench_sdpa_tq.rs::bench_sdpa_sliding_layer`. There are no live
//! call sites in /opt/hf2q (audited 2026-04-25); the public dispatch fn is
//! marked `#[deprecated]` so any new caller surfaces a warning until the
//! prefill bug is either fixed or this op is formally retired in favour of
//! `flash_attn_prefill`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CapturedOpKind, CommandEncoder};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

/// MSL source for the sliding-window SDPA kernel (embedded at compile time).
pub static SDPA_SLIDING_SHADER_SOURCE: &str = include_str!("../shaders/sdpa_sliding.metal");

/// Register sliding-window SDPA shader source with the given kernel registry.
///
/// This must be called before dispatching any sliding-window SDPA operations.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("sdpa_sliding", SDPA_SLIDING_SHADER_SOURCE);
}

/// Parameters for the sliding-window SDPA kernel.
#[derive(Debug, Clone, Copy)]
pub struct SdpaSlidingParams {
    /// Number of query attention heads.
    pub n_heads: u32,
    /// Number of key/value attention heads (GQA: may be < n_heads).
    pub n_kv_heads: u32,
    /// Dimension of each attention head.
    pub head_dim: u32,
    /// Query sequence length.
    pub seq_len: u32,
    /// Key/value sequence length.
    pub kv_seq_len: u32,
    /// Sliding window size.  Attention is restricted to the last `window_size`
    /// key positions relative to each query position.
    pub window_size: u32,
    /// Attention score scaling factor. Typically `1.0 / sqrt(head_dim)`, but
    /// models like Gemma 4 (which use QK norms) require `scale = 1.0`.
    pub scale: f32,
    /// KV cache capacity — stride (in positions) between KV heads in the
    /// cache buffer.  When KV buffers use pre-allocated capacity > kv_seq_len,
    /// set this to the capacity.  0 means "use kv_seq_len" for backwards compat.
    pub kv_capacity: u32,
}

/// GPU-side parameter struct layout.  Must match the MSL `SdpaSlidingParams`
/// struct exactly (7 × u32 + 1 × f32 = 32 bytes, no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SdpaSlidingParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_seq_len: u32,
    window_size: u32,
    scale: f32,
    kv_capacity: u32,
}

/// Tile size for query positions per threadgroup.  Must match `TILE_Q` in the
/// MSL shader.
const TILE_Q: u32 = 32;

/// Validate sliding-window SDPA parameters.
fn validate_sliding_params(params: &SdpaSlidingParams) -> Result<()> {
    if params.head_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "head_dim must be > 0".into(),
        ));
    }
    if params.n_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "n_heads must be > 0".into(),
        ));
    }
    if params.n_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "n_kv_heads must be > 0".into(),
        ));
    }
    if params.n_heads % params.n_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "n_heads ({}) must be divisible by n_kv_heads ({})",
            params.n_heads, params.n_kv_heads
        )));
    }
    if params.seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "seq_len must be > 0".into(),
        ));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "kv_seq_len must be > 0".into(),
        ));
    }
    if params.window_size == 0 {
        return Err(MlxError::InvalidArgument(
            "window_size must be > 0".into(),
        ));
    }
    Ok(())
}

/// Validate that a buffer has the expected byte length for the given shape.
fn validate_buffer(buf: &MlxBuffer, name: &str, expected_elements: usize) -> Result<()> {
    let expected_bytes = expected_elements * buf.dtype().size_of();
    if buf.byte_len() < expected_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "{name} buffer too small: expected at least {expected_bytes} bytes, got {}",
            buf.byte_len()
        )));
    }
    Ok(())
}

/// Dispatch sliding-window scaled dot-product attention on the GPU.
///
/// Encodes a compute command into the provided `CommandEncoder` without
/// committing.  The caller controls when to call `encoder.commit_and_wait()`.
///
/// # Arguments
///
/// * `encoder`    — Command encoder to record the dispatch into.
/// * `registry`   — Kernel registry for pipeline lookup/compilation.
/// * `device`     — Metal device (for compilation and buffer allocation).
/// * `q`          — Query buffer `[batch, n_heads, seq_len, head_dim]`, F32.
/// * `k`          — Key buffer `[batch, n_kv_heads, kv_seq_len, head_dim]`, F32.
/// * `v`          — Value buffer `[batch, n_kv_heads, kv_seq_len, head_dim]`, F32.
/// * `output`     — Output buffer, same shape as Q, pre-allocated.
/// * `params`     — Attention parameters including `window_size`.
/// * `batch_size` — Number of independent sequences in the batch.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` for invalid parameters or mismatched
/// buffer sizes.
#[deprecated(
    note = "broken at prefill shapes — see /opt/hf2q/docs/spike-gate-a-prefill.md:118-130. \
            Use sdpa_decode for the seq_len=1 path or the in-flight flash_attn_prefill \
            kernel (ADR-011) for the prefill path. Repair-or-remove decision is open; \
            this attribute will lift once the kernel is fixed or the op is formally \
            retired in favour of flash_attn_prefill."
)]
pub fn sdpa_sliding(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    output: &MlxBuffer,
    params: &SdpaSlidingParams,
    batch_size: u32,
) -> Result<()> {
    validate_sliding_params(params)?;

    // Resolve kv_capacity: 0 means "same as kv_seq_len" for backwards compat.
    let kv_cap = if params.kv_capacity == 0 { params.kv_seq_len } else { params.kv_capacity };

    // Validate buffer sizes.
    let q_elements = batch_size as usize
        * params.n_heads as usize
        * params.seq_len as usize
        * params.head_dim as usize;
    // KV buffers are strided by kv_capacity, not kv_seq_len.
    let kv_elements = batch_size as usize
        * params.n_kv_heads as usize
        * kv_cap as usize
        * params.head_dim as usize;

    validate_buffer(q, "Q", q_elements)?;
    validate_buffer(k, "K", kv_elements)?;
    validate_buffer(v, "V", kv_elements)?;
    validate_buffer(output, "output", q_elements)?;

    // Allocate params buffer.
    let params_gpu = SdpaSlidingParamsGpu {
        n_heads: params.n_heads,
        n_kv_heads: params.n_kv_heads,
        head_dim: params.head_dim,
        seq_len: params.seq_len,
        kv_seq_len: params.kv_seq_len,
        window_size: params.window_size,
        scale: params.scale,
        kv_capacity: kv_cap,
    };
    let params_bytes = bytemuck::bytes_of(&params_gpu);
    let mut params_buf = device.alloc_buffer(
        params_bytes.len(),
        DType::U8,
        vec![params_bytes.len()],
    )?;
    {
        let dst: &mut [u8] = params_buf.as_mut_slice()?;
        dst[..params_bytes.len()].copy_from_slice(params_bytes);
    }

    // Get the compiled pipeline.
    // Select kernel based on buffer dtype.
    let kernel_name = if q.dtype() == DType::BF16 { "sdpa_sliding_bf16" } else { "sdpa_sliding" };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    // Calculate dispatch grid.
    let n_tiles = (params.seq_len + TILE_Q - 1) / TILE_Q;
    let threadgroups = MTLSize::new(
        batch_size as u64,
        params.n_heads as u64,
        n_tiles as u64,
    );
    let threadgroup_size = MTLSize::new(TILE_Q as u64, 1, 1);

    // Tag for the reorder pass (Phase 4e.3): SDPA is NOT reorderable.
    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // Encode the dispatch.
    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, q),
            (1, k),
            (2, v),
            (3, output),
            (4, &params_buf),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_sliding_params_ok() {
        let p = SdpaSlidingParams {
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 256,
            seq_len: 2048,
            kv_seq_len: 2048,
            window_size: 1024,
            scale: 1.0 / (256.0_f32).sqrt(),
            kv_capacity: 2048,
        };
        assert!(validate_sliding_params(&p).is_ok());
    }

    #[test]
    fn test_validate_sliding_params_zero_window() {
        let p = SdpaSlidingParams {
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 256,
            seq_len: 128,
            kv_seq_len: 128,
            window_size: 0,
            scale: 1.0,
            kv_capacity: 128,
        };
        assert!(matches!(
            validate_sliding_params(&p),
            Err(MlxError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_gpu_sliding_params_layout() {
        // Ensure SdpaSlidingParamsGpu is exactly 32 bytes (7 x u32 + 1 x f32 = 8 fields x 4).
        assert_eq!(std::mem::size_of::<SdpaSlidingParamsGpu>(), 32);
    }
}
