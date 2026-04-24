//! Flash attention vector kernel dispatch for higher-bit TurboQuant KV cache.
//!
//! Variant of `flash_attn_vec_tq` that reads K/V from byte-packed (1 byte/element)
//! higher-bit codebook indices. Supports 5-bit (32 centroids), 6-bit (64 centroids),
//! and 8-bit (256 centroids) Lloyd-Max codebooks for N(0,1).
//!
//! Bit-width is controlled at runtime via `FlashAttnVecTqHbParams::codebook_bits`.
//!
//! ADR-007 iter-24: measure Gate A/B/C at 5/6/8-bit to find smallest shippable bit-width.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the HB TQ flash attention vector kernel.
pub static FLASH_ATTN_VEC_TQ_HB_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec_tq_hb.metal");

/// Register HB TQ flash attention vector shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("flash_attn_vec_tq_hb_dk256", FLASH_ATTN_VEC_TQ_HB_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_tq_hb_dk512", FLASH_ATTN_VEC_TQ_HB_SHADER_SOURCE);
}

/// Parameters for the HB TQ flash attention vector kernel.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttnVecTqHbParams {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,
    pub scale: f32,
    pub mask_type: u32,
    pub sliding_window: u32,
    pub softcap: f32,
    /// Ring buffer start slot (same semantics as FlashAttnVecTqParams::ring_start).
    pub ring_start: u32,
    /// Scale divisor for D=512 per-block norms (matches hadamard_quantize_kv_hb convention).
    pub scale_factor_d512: f32,
    /// Codebook bit-width: 5, 6, or 8.
    pub codebook_bits: u32,
}

/// GPU-side parameter struct. Must match `FlashAttnVecTqHbParams` in the MSL exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecTqHbParamsGpu {
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
    ring_start: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,
}

/// GPU-side reduce params. Reuses the same reduce kernel as flash_attn_vec_tq.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
}

fn validate_params(params: &FlashAttnVecTqHbParams) -> Result<()> {
    if params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: head_dim must be 256 or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_tq_hb: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: num_heads ({}) % num_kv_heads ({}) != 0",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_tq_hb: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: kv_capacity ({}) < kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    if !matches!(params.codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: codebook_bits must be 5, 6, or 8, got {}",
            params.codebook_bits
        )));
    }
    Ok(())
}

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

/// Dispatch HB TQ flash attention vector kernel (5/6/8-bit byte-packed K/V).
///
/// Same calling convention as `flash_attn_vec_tq` except K/V are byte-packed
/// (1 byte/element) from the higher-bit encode path.
///
/// FWHT of Q must be applied by the caller before this call; inverse FWHT of
/// output must be applied by the caller after. Same as the 4-bit TQ path.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_tq_hb(
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
    params: &FlashAttnVecTqHbParams,
) -> Result<()> {
    validate_params(params)?;

    let head_dim = params.head_dim;
    let nwg = compute_nwg(params.kv_seq_len);

    let gpu_params = FlashAttnVecTqHbParamsGpu {
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
        ring_start: params.ring_start,
        scale_factor_d512: params.scale_factor_d512,
        codebook_bits: params.codebook_bits,
    };

    let kernel_name = match head_dim {
        256 => "flash_attn_vec_tq_hb_dk256",
        512 => "flash_attn_vec_tq_hb_dk512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_tq_hb: unsupported head_dim {head_dim}"
        ))),
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32;
    let shmem_halfs = pk + sh + 2 * pv;
    let shmem_bytes = shmem_halfs * 2;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    let threadgroups = MTLSize::new(1, params.num_heads as u64, nwg as u64);
    let threadgroup_size = MTLSize::new(32, 1, 1);

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

    // Reduce kernel (NWG > 1)
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

/// Size in bytes of the temporary buffer needed for HB TQ SDPA.
pub fn tmp_buffer_bytes(num_heads: u32, head_dim: u32) -> usize {
    let nrows = num_heads as usize;
    let max_nwg = 32usize;
    let dv = head_dim as usize;
    (nrows * max_nwg * (dv + 2)) * std::mem::size_of::<f32>()
}

fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_params_size() {
        // 13 fields × 4 bytes = 52 bytes
        assert_eq!(std::mem::size_of::<FlashAttnVecTqHbParamsGpu>(), 52);
    }

    #[test]
    fn test_validate_bad_bits() {
        let p = FlashAttnVecTqHbParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 4,  // invalid
        };
        assert!(validate_params(&p).is_err());
    }

    #[test]
    fn test_validate_ok_8bit() {
        let p = FlashAttnVecTqHbParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 256,
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };
        assert!(validate_params(&p).is_ok());
    }
}
