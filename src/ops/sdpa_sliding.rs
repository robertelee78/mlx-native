//! Sliding-window scaled dot-product attention host dispatch.
//!
//! Same as [`sdpa`](super::sdpa) but applies a sliding window mask: for each
//! query position `q_pos`, keys at positions `k_pos < q_pos - window_size` are
//! masked to negative infinity before softmax.  Combined with the causal mask,
//! the effective attention window is `[max(0, q_pos - window_size), q_pos]`.
//!
//! Used by Gemma 4 sliding-window layers (5 of every 6 layers, window=1024).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::CommandEncoder;
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
}

/// GPU-side parameter struct layout.  Must match the MSL `SdpaSlidingParams`
/// struct exactly (6 consecutive `uint32` values, no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SdpaSlidingParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_seq_len: u32,
    window_size: u32,
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

    // Validate buffer sizes.
    let q_elements = batch_size as usize
        * params.n_heads as usize
        * params.seq_len as usize
        * params.head_dim as usize;
    let kv_elements = batch_size as usize
        * params.n_kv_heads as usize
        * params.kv_seq_len as usize
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
        };
        assert!(matches!(
            validate_sliding_params(&p),
            Err(MlxError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_gpu_sliding_params_layout() {
        // Ensure SdpaSlidingParamsGpu is exactly 24 bytes (6 x u32).
        assert_eq!(std::mem::size_of::<SdpaSlidingParamsGpu>(), 24);
    }
}
