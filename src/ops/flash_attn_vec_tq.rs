//! Flash attention vector kernel dispatch for TurboQuant-compressed KV cache.
//!
//! Fork of `flash_attn_vec` that reads K and V from nibble-packed indices
//! + per-position norms + pre-rotated centroid table, instead of F16/F32 buffers.
//!
//! Key differences from `flash_attn_vec`:
//! - NWG=1: no reduce kernel needed (TQ's 4× smaller KV reads mean one
//!   workgroup per head is sufficient)
//! - Q is FWHT-rotated inside the kernel (shared memory)
//! - Accumulated output is inverse-FWHT-rotated before writing to dst
//! - Output goes directly to the destination buffer (no tmp buffer)

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the TQ flash attention vector kernel (embedded at compile time).
pub static FLASH_ATTN_VEC_TQ_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec_tq.metal");

/// Register TQ flash attention vector shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("flash_attn_vec_tq_dk256", FLASH_ATTN_VEC_TQ_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_tq_dk512", FLASH_ATTN_VEC_TQ_SHADER_SOURCE);
}

/// Parameters for the TQ flash attention vector kernel.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnVecTqParams {
    /// Number of query attention heads.
    pub num_heads: u32,
    /// Number of key/value attention heads (GQA: may be < num_heads).
    pub num_kv_heads: u32,
    /// Dimension of each attention head (256 or 512).
    pub head_dim: u32,
    /// Current KV sequence length (number of valid positions).
    pub kv_seq_len: u32,
    /// KV cache capacity (stride between KV heads in positions).
    pub kv_capacity: u32,
    /// Attention score scaling factor (e.g. 1/sqrt(head_dim) or 1.0).
    pub scale: f32,
    /// Mask type: 0=none, 1=causal, 2=sliding_window.
    pub mask_type: u32,
    /// Sliding window size (only used when mask_type == 2).
    pub sliding_window: u32,
    /// Logit softcapping (0 = disabled).
    pub softcap: f32,
}

/// GPU-side parameter struct. Must match the MSL `FlashAttnVecTqParams` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecTqParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    softcap: f32,
    nwg: u32,
}

/// Validate TQ flash attention parameters.
fn validate_params(params: &FlashAttnVecTqParams) -> Result<()> {
    if params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq: head_dim must be 256 or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_tq: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq: num_heads ({}) must be divisible by num_kv_heads ({})",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_tq: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq: kv_capacity ({}) must be >= kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    Ok(())
}

/// Dispatch TQ flash attention vector kernel on the GPU.
///
/// This dispatches a single Metal compute pass (NWG=1, no reduce kernel).
/// The kernel applies FWHT to Q, gathers K/V from centroid tables, and
/// applies inverse FWHT to the output.
///
/// # Arguments
///
/// * `encoder`      — Command encoder to record dispatches into.
/// * `registry`     — Kernel registry for pipeline lookup/compilation.
/// * `device`       — Metal device.
/// * `q`            — Query buffer `[num_heads, 1, head_dim]`, F32.
/// * `k_packed`     — Nibble-packed K indices `[num_kv_heads, kv_capacity, head_dim/2]`, U8.
/// * `k_norms`      — Per-position K norms `[num_kv_heads, kv_capacity]`, F32.
/// * `k_centroids`  — Pre-rotated K centroid table `[16, head_dim]`, F32.
/// * `v_packed`     — Nibble-packed V indices `[num_kv_heads, kv_capacity, head_dim/2]`, U8.
/// * `v_norms`      — Per-position V norms `[num_kv_heads, kv_capacity]`, F32.
/// * `v_centroids`  — Pre-rotated V centroid table `[16, head_dim]`, F32.
/// * `output`       — Output buffer `[num_heads, 1, head_dim]`, F32, pre-allocated.
/// * `params`       — TQ flash attention parameters.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_tq(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_packed: &MlxBuffer,
    k_norms: &MlxBuffer,
    k_centroids: &MlxBuffer,
    v_packed: &MlxBuffer,
    v_norms: &MlxBuffer,
    v_centroids: &MlxBuffer,
    output: &MlxBuffer,
    params: &FlashAttnVecTqParams,
) -> Result<()> {
    validate_params(params)?;

    let head_dim = params.head_dim;

    // NWG=1: single workgroup per head. No reduce kernel needed.
    // TQ's 4× bandwidth reduction means one workgroup is sufficient.
    let nwg: u32 = 1;

    let gpu_params = FlashAttnVecTqParamsGpu {
        n_heads: params.num_heads,
        n_kv_heads: params.num_kv_heads,
        head_dim: params.head_dim,
        kv_seq_len: params.kv_seq_len,
        kv_capacity: params.kv_capacity,
        scale: params.scale,
        mask_type: params.mask_type,
        sliding_window: params.sliding_window,
        softcap: params.softcap,
        nwg,
    };

    let kernel_name = match head_dim {
        256 => "flash_attn_vec_tq_dk256",
        512 => "flash_attn_vec_tq_dk512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq: unsupported head_dim {head_dim}"
        ))),
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    // Shared memory size — same layout as flash_attn_vec.
    // PK halfs (Q half4) + SH halfs (scratch) + 2*PV halfs (output float4)
    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32; // 4 * C = 128 halfs
    let shmem_halfs = pk + sh + 2 * pv;
    let shmem_bytes = shmem_halfs * 2; // 2 bytes per half

    // Tag for the reorder pass: SDPA is NOT reorderable.
    encoder.set_op_kind(CapturedOpKind::Sdpa);

    // Dispatch: (1 query, num_heads, 1 workgroup).
    // NWG=1: output goes directly to dst, no reduce kernel.
    let threadgroups = MTLSize::new(1, params.num_heads as u64, nwg as u64);
    let threadgroup_size = MTLSize::new(32, 1, 1); // 1 simdgroup of 32 threads

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_packed)),
            (3, KernelArg::Buffer(k_norms)),
            (4, KernelArg::Buffer(k_centroids)),
            (5, KernelArg::Buffer(v_packed)),
            (6, KernelArg::Buffer(v_norms)),
            (7, KernelArg::Buffer(v_centroids)),
            (8, KernelArg::Buffer(output)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // No reduce kernel: NWG=1 writes directly to output with FWHT applied.

    Ok(())
}

/// Pad x up to next multiple of n (n must be power of 2).
fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_params_ok() {
        let p = FlashAttnVecTqParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 1,
            sliding_window: 0,
            softcap: 0.0,
        };
        assert!(validate_params(&p).is_ok());
    }

    #[test]
    fn test_validate_params_bad_head_dim() {
        let p = FlashAttnVecTqParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 128,
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
        };
        assert!(validate_params(&p).is_err());
    }

    #[test]
    fn test_gpu_params_layout() {
        assert_eq!(
            std::mem::size_of::<FlashAttnVecTqParamsGpu>(),
            40, // 10 x u32/f32 = 40 bytes
        );
    }
}
