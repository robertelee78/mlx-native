//! Scaled dot-product attention (SDPA) host dispatch.
//!
//! Computes `softmax(Q * K^T / sqrt(head_dim)) * V` on the GPU using a fused
//! Metal compute kernel with causal masking.
//!
//! Supports grouped-query attention (GQA) where `n_heads > n_kv_heads`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

/// MSL source for the SDPA kernel (embedded at compile time).
pub static SDPA_SHADER_SOURCE: &str = include_str!("../shaders/sdpa.metal");

/// Register SDPA shader source with the given kernel registry.
///
/// This must be called before dispatching any SDPA operations.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("sdpa", SDPA_SHADER_SOURCE);
}

/// Parameters for the SDPA kernel.
///
/// These describe the tensor shapes and head configuration for the attention
/// computation.
#[derive(Debug, Clone, Copy)]
pub struct SdpaParams {
    /// Number of query attention heads (e.g. 16 for Gemma 4).
    pub n_heads: u32,
    /// Number of key/value attention heads (may be less than `n_heads` for GQA).
    pub n_kv_heads: u32,
    /// Dimension of each attention head.
    pub head_dim: u32,
    /// Query sequence length.
    pub seq_len: u32,
    /// Key/value sequence length (may differ from `seq_len` in decode mode).
    pub kv_seq_len: u32,
    /// Attention score scaling factor. Typically `1.0 / sqrt(head_dim)`, but
    /// models like Gemma 4 (which use QK norms) require `scale = 1.0`.
    pub scale: f32,
}

/// GPU-side parameter struct layout.  Must match the MSL `SdpaParams` struct
/// exactly (5 × u32 + 1 × f32 = 24 bytes, no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SdpaParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_seq_len: u32,
    scale: f32,
}

/// Tile size for query positions per threadgroup.  Must match `TILE_Q` in the
/// MSL shader.
const TILE_Q: u32 = 32;

/// Validate SDPA parameters and return a descriptive error if invalid.
fn validate_params(params: &SdpaParams) -> Result<()> {
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

/// Dispatch scaled dot-product attention on the GPU.
///
/// Encodes a compute command into the provided `CommandEncoder` without
/// committing.  The caller controls when to call `encoder.commit_and_wait()`.
///
/// # Arguments
///
/// * `encoder`  — Command encoder to record the dispatch into.
/// * `registry` — Kernel registry for pipeline lookup/compilation.
/// * `device`   — Metal device (needed for pipeline compilation and buffer allocation).
/// * `q`        — Query buffer, shape `[batch, n_heads, seq_len, head_dim]`, dtype F32.
/// * `k`        — Key buffer, shape `[batch, n_kv_heads, kv_seq_len, head_dim]`, dtype F32.
/// * `v`        — Value buffer, shape `[batch, n_kv_heads, kv_seq_len, head_dim]`, dtype F32.
/// * `output`   — Output buffer, same shape as Q, pre-allocated by caller.
/// * `params`   — Attention parameters (head counts, dimensions, sequence lengths).
/// * `batch_size` — Number of independent sequences in the batch.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` for invalid parameters or mismatched
/// buffer sizes.
pub fn sdpa(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    output: &MlxBuffer,
    params: &SdpaParams,
    batch_size: u32,
) -> Result<()> {
    validate_params(params)?;

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

    // Allocate a small buffer for the GPU-side params struct.
    let params_gpu = SdpaParamsGpu {
        n_heads: params.n_heads,
        n_kv_heads: params.n_kv_heads,
        head_dim: params.head_dim,
        seq_len: params.seq_len,
        kv_seq_len: params.kv_seq_len,
        scale: params.scale,
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
    let kernel_name = if q.dtype() == DType::BF16 { "sdpa_bf16" } else { "sdpa" };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    // Calculate dispatch grid.
    // Threadgroups: (batch, n_heads, ceil(seq_len / TILE_Q))
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
    fn test_validate_params_ok() {
        let p = SdpaParams {
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 256,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 1.0 / (256.0_f32).sqrt(),
        };
        assert!(validate_params(&p).is_ok());
    }

    #[test]
    fn test_validate_params_zero_head_dim() {
        let p = SdpaParams {
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 0,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 1.0,
        };
        assert!(matches!(
            validate_params(&p),
            Err(MlxError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_validate_params_bad_ratio() {
        let p = SdpaParams {
            n_heads: 16,
            n_kv_heads: 7,
            head_dim: 256,
            seq_len: 128,
            kv_seq_len: 128,
            scale: 1.0,
        };
        assert!(matches!(
            validate_params(&p),
            Err(MlxError::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_gpu_params_layout() {
        // Ensure SdpaParamsGpu is exactly 24 bytes (5 x u32 + 1 x f32).
        assert_eq!(std::mem::size_of::<SdpaParamsGpu>(), 24);
    }
}
