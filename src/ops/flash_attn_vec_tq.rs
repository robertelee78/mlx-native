//! Flash attention vector kernel dispatch for TurboQuant-compressed KV cache.
//!
//! Fork of `flash_attn_vec` that reads K and V from nibble-packed indices
//! + per-position norms, with inline scalar dequant from a register-resident
//! 16-element codebook. No centroid table buffer needed.
//!
//! Key differences from `flash_attn_vec`:
//! - Adaptive NWG (1-32) based on kv_seq_len. At short context NWG=1
//!   avoids the reduce kernel. At long context NWG scales up for parallelism.
//! - Caller handles FWHT: pre-rotates Q, post-rotates output (1× per head).
//! - Dequant is inline: codebook[nibble] * inv_sqrt(head_dim) * norm
//! - Zero scattered memory access — codebook fits in registers

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

/// GPU-side reduce params. Must match `FlashAttnVecReduceParams` in the MSL.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
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

/// Compute NWG for TQ SDPA.
///
/// NWG=16 is optimal across both short and long context on M5 Max
/// (measured: outperforms both NWG=1 and NWG=32 at all tested lengths).
/// NWG=32 adds reduce kernel overhead that outweighs its parallelism gain.
/// NWG<16 starves the GPU at long context.
///
/// Override: set HF2Q_TQ_NWG=N to force a specific value (for benchmarking).
fn compute_nwg(_kv_seq_len: u32) -> u32 {
    if let Ok(v) = std::env::var("HF2Q_TQ_NWG") {
        if let Ok(n) = v.parse::<u32>() {
            if n >= 1 && n <= 32 {
                return n;
            }
        }
    }
    16
}

/// Dispatch TQ flash attention vector kernel on the GPU.
///
/// Dispatches NWG=32 workgroups per head, then a reduce kernel.
///
/// **FWHT is NOT done inside this kernel.** The caller must:
/// 1. Pre-rotate Q via `dispatch_fwht_f32` before calling this function
/// 2. Apply inverse FWHT to the output after this function returns
///
/// With NWG=32, doing FWHT per-workgroup would repeat it 32× per head.
/// Keeping FWHT outside means it's done once per head regardless of NWG.
///
/// # Arguments
///
/// * `q`            — Query buffer `[num_heads, 1, head_dim]`, F32, **pre-rotated via FWHT**.
/// * `output`       — Output buffer `[num_heads, 1, head_dim]`, F32, **in rotated domain**.
/// * `tmp`          — Temporary buffer for NWG partial results.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_tq(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_packed: &MlxBuffer,
    k_norms: &MlxBuffer,
    v_packed: &MlxBuffer,
    v_norms: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    params: &FlashAttnVecTqParams,
) -> Result<()> {
    validate_params(params)?;

    let head_dim = params.head_dim;
    let nwg = compute_nwg(params.kv_seq_len);

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

    // Dispatch main kernel: (1 query, num_heads, NWG workgroups).
    let threadgroups = MTLSize::new(1, params.num_heads as u64, nwg as u64);
    let threadgroup_size = MTLSize::new(32, 1, 1); // 1 simdgroup of 32 threads

    // NWG=1: write directly to output (no reduce needed).
    // NWG>1: write to tmp, then reduce into output.
    let dst_buf = if nwg == 1 { output } else { tmp };

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_packed)),
            (3, KernelArg::Buffer(k_norms)),
            (4, KernelArg::Buffer(v_packed)),
            (5, KernelArg::Buffer(v_norms)),
            (6, KernelArg::Buffer(dst_buf)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // --- Reduce kernel (NWG > 1 only) ---
    if nwg > 1 {
        encoder.memory_barrier();

        let reduce_params = FlashAttnVecReduceParamsGpu { nrows: params.num_heads };

        let reduce_kernel = match head_dim {
            256 => "flash_attn_vec_reduce_dk256",
            512 => "flash_attn_vec_reduce_dk512",
            _ => unreachable!(),
        };
        let reduce_pipeline = registry.get_pipeline(reduce_kernel, device.metal_device())?;

        let reduce_tg = MTLSize::new(params.num_heads as u64, 1, 1);
        let reduce_tg_size = MTLSize::new(32 * nwg as u64, 1, 1);

        encoder.encode_threadgroups_with_args(
            reduce_pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&reduce_params))),
                (1, KernelArg::Buffer(tmp)),
                (2, KernelArg::Buffer(output)),
                (3, KernelArg::Bytes(as_bytes(&nwg))),
            ],
            reduce_tg,
            reduce_tg_size,
        );
    }

    Ok(())
}

/// Compute the size in bytes of the temporary buffer needed for TQ SDPA.
///
/// Sized for max NWG=32 regardless of actual adaptive NWG — the buffer is
/// allocated once at model load time and reused for all context lengths.
pub fn tmp_buffer_bytes(num_heads: u32, head_dim: u32) -> usize {
    let nrows = num_heads as usize;
    let max_nwg = 32usize;
    let dv = head_dim as usize;
    (nrows * max_nwg * (dv + 2)) * std::mem::size_of::<f32>()
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
