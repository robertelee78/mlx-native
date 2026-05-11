//! Flash attention vector kernel dispatch for hybrid F16-K + TQ-HB-V KV cache.
//!
//! ADR-028 Phase 10d (iter-349): closes the structural 1.81× per-dispatch
//! K-side gap measured iter-326..342 by reading K as F16 dense (peer-equivalent
//! layout, no codebook lookup) while V stays byte-packed TQ-HB.
//!
//! Memory cost: 158 MB at gemma4 32K context (vs 128 MB pure TQ-HB) — 3.19×
//! savings vs raw F32, preserving 81% of the TQ-HB advantage. See
//! `ADR-028 §iter-346` for the full memory math.
//!
//! ABI: re-uses `FlashAttnVecTqHbParams` (V-side fields are identical;
//! K-side `K_norms` simply absent from the buffer list, K codebook unused).
//! V codebook width controlled by the same `CBITS_FC` function constant
//! (5/6/8) at index 50 — identical to flash_attn_vec_tq_hb.
//!
//! Buffer slots (compacted vs TQ-HB which has 7 buffers):
//!   0 = params, 1 = Q, 2 = K_f16, 3 = V_packed, 4 = V_norms, 5 = dst.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

// Re-export the shared params struct + tmp helpers from the TQ-HB module —
// the hybrid kernel uses byte-identical layout (only K-side buffer differs).
pub use super::flash_attn_vec_tq_hb::{
    compute_nsg, tmp_buffer_bytes, FlashAttnVecTqHbParams,
};

/// MSL source for the hybrid SDPA kernel.
pub static FLASH_ATTN_VEC_HYBRID_SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec_hybrid.metal");

/// Register hybrid SDPA shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("flash_attn_vec_hybrid_dk256", FLASH_ATTN_VEC_HYBRID_SHADER_SOURCE);
    registry.register_source("flash_attn_vec_hybrid_dk512", FLASH_ATTN_VEC_HYBRID_SHADER_SOURCE);
}

/// GPU-side parameter struct (byte-identical layout to `FlashAttnVecTqHbParamsGpu`
/// in the TQ-HB module — re-defined locally to avoid cross-module pub-vis).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecHybridParamsGpu {
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
    fuse_fwht_pre: u32,
    nsg: u32,
}

/// GPU-side reduce params (re-uses flash_attn_vec_reduce_dk{256,512} kernel).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
}

fn validate_params(params: &FlashAttnVecTqHbParams) -> Result<()> {
    if params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: head_dim must be 256 or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_hybrid: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: num_heads ({}) % num_kv_heads ({}) != 0",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_hybrid: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: kv_capacity ({}) < kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    // V codebook bits — same allowed set as TQ-HB (5/6/8). Hybrid path's K is F16
    // and ignores this; only V-side honors it.
    if !matches!(params.codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: V codebook_bits must be 5, 6, or 8, got {}",
            params.codebook_bits
        )));
    }
    if params.nsg == 0 || (params.nsg & (params.nsg - 1)) != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: nsg must be a power of 2 (1, 2, 4, ...), got {}",
            params.nsg
        )));
    }
    if params.nsg > 4 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: nsg must be ≤ 4 (kernel reduce cap), got {}",
            params.nsg
        )));
    }
    Ok(())
}

/// Mirror of `flash_attn_vec_tq_hb::compute_nwg`. Hybrid uses the same kL
/// adaptive policy because the workgroup-axis split is independent of K dtype.
fn compute_nwg(kv_seq_len: u32) -> u32 {
    if let Ok(v) = std::env::var("HF2Q_HYBRID_NWG") {
        if let Ok(n) = v.parse::<u32>() {
            if (1..=32).contains(&n) {
                return n;
            }
        }
    }
    if kv_seq_len > 512 { 32 } else { 16 }
}

/// Dispatch the hybrid F16-K + TQ-HB-V flash attention vector kernel.
///
/// Caller responsibilities (same as `flash_attn_vec_tq_hb`):
///   * Q is F32 `[n_heads, head_dim]` (or `[n_heads, 1, head_dim]` for decode).
///   * Q must be FWHT-pre-rotated UNLESS `params.fuse_fwht_pre == 1` (kernel
///     applies the rotation internally).
///   * K_f16 is `[num_kv_heads, kv_capacity, head_dim]` half (F16, 2 bytes/elem).
///   * V_packed is `[num_kv_heads, kv_capacity, head_dim]` u8 (byte-packed).
///   * V_norms layout matches `flash_attn_vec_tq_hb` (D=256: 1/pos, D=512: 2/pos).
///   * Caller applies inverse-FWHT to `output` if Q was pre-rotated (mirrors
///     existing TQ-HB caller pattern at hf2q `forward_mlx.rs:3744+`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_hybrid(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k_f16: &MlxBuffer,
    v_packed: &MlxBuffer,
    v_norms: &MlxBuffer,
    output: &MlxBuffer,
    tmp: &MlxBuffer,
    params: &FlashAttnVecTqHbParams,
) -> Result<()> {
    validate_params(params)?;

    // K_f16 dtype check — hybrid path requires F16; F32 K would silently
    // misalign the kernel's `device const half *K_f16` reads (2 bytes/elem
    // vs 4) → 2× overrun on every K row → memory corruption. Catch loud.
    if k_f16.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: k_f16 must be DType::F16, got {:?}",
            k_f16.dtype()
        )));
    }

    let head_dim = params.head_dim;
    let nwg = compute_nwg(params.kv_seq_len);

    let gpu_params = FlashAttnVecHybridParamsGpu {
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
        fuse_fwht_pre: params.fuse_fwht_pre,
        nsg: params.nsg,
    };

    let kernel_name = match head_dim {
        256 => "flash_attn_vec_hybrid_dk256",
        512 => "flash_attn_vec_hybrid_dk512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "flash_attn_vec_hybrid: unsupported head_dim {head_dim}"
        ))),
    };
    // V codebook function constant (same as TQ-HB at index 50). Hybrid path's
    // K-side ignores this; only the V dequant inner loop honors it.
    let cbits_const = (params.codebook_bits as i32, 50usize);
    // ADR-029 iter-20 H27: V-dtype function constant (slot 51).  When V buffer
    // is F16-typed (caller has allocated full F16 KV cache via HF2Q_FULL_F16_KV),
    // the kernel takes the F16-V direct-read branch.  When U8-typed (legacy
    // TQ-HB byte-packed V), the kernel takes the dequant_hb_float4 branch.
    let v_is_f16: i32 = match v_packed.dtype() {
        crate::DType::F16 => 1,
        _ => 0,
    };
    let pipeline = registry
        .get_pipeline_with_constants(
            kernel_name,
            device.metal_device(),
            &[],
            &[(cbits_const.1, cbits_const.0), (51usize, v_is_f16)],
        )?;

    let pk = pad2(head_dim as usize, 128);
    let pv = pad2(head_dim as usize, 128);
    let sh = 4 * 32;
    // Same NSG-aware shmem layout as TQ-HB.
    let nsg = params.nsg as usize;
    let shmem_halfs = pk + nsg * (sh + 2 * pv);
    let shmem_bytes = shmem_halfs * 2;

    encoder.set_op_kind(CapturedOpKind::Sdpa);

    let threadgroups = MTLSize::new(1, params.num_heads as u64, nwg as u64);
    let threadgroup_size = MTLSize::new(32, params.nsg as u64, 1);

    let dst_buf = if nwg == 1 { output } else { tmp };

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k_f16)),     // F16 dense (vs k_packed + k_norms in TQ-HB)
            (3, KernelArg::Buffer(v_packed)),
            (4, KernelArg::Buffer(v_norms)),
            (5, KernelArg::Buffer(dst_buf)),
        ],
        &[(0, shmem_bytes as u64)],
        threadgroups,
        threadgroup_size,
    );

    // Reduce kernel (NWG > 1) — same kernel as TQ-HB.
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

fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_params_size() {
        // 15 u32-sized fields × 4 bytes = 60 bytes — must match TQ-HB layout
        // exactly so the kernel's `constant FlashAttnVecTqHbParams &params`
        // binding sees the same bytes.
        assert_eq!(std::mem::size_of::<FlashAttnVecHybridParamsGpu>(), 60);
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
            codebook_bits: 4,  // invalid for hybrid V-side (TQ-HB only takes 5/6/8)
            fuse_fwht_pre: 0,
            nsg: 1,
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
            fuse_fwht_pre: 0,
            nsg: 1,
        };
        assert!(validate_params(&p).is_ok());
    }

    #[test]
    fn test_validate_bad_head_dim() {
        let p = FlashAttnVecTqHbParams {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 128,  // unsupported
            kv_seq_len: 64,
            kv_capacity: 1024,
            scale: 1.0,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
            fuse_fwht_pre: 0,
            nsg: 1,
        };
        assert!(validate_params(&p).is_err());
    }
}
