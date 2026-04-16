//! Flash attention vector kernel for affine INT4-compressed KV cache.
//!
//! Same structure as flash_attn_vec_tq but decodes with symmetric affine math:
//!   value = scale_norm * (float(nibble) - 7.5)
//! instead of:
//!   value = CODEBOOK_4BIT[nibble] * scale_norm
//!
//! Buffer layout is identical to TQ — same packed nibbles + per-position float.
//! The "norms" buffer stores absmax * sqrt(hd) / 7.5 instead of L2 norm.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use metal::MTLSize;

/// MSL source for the affine4 SDPA kernel.
pub static SHADER_SOURCE: &str =
    include_str!("../shaders/flash_attn_vec_affine4.metal");

/// Register the affine4 SDPA shader source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("flash_attn_vec_affine4_dk256", SHADER_SOURCE);
    registry.register_source("flash_attn_vec_affine4_dk512", SHADER_SOURCE);
}

/// Reuse the same params struct as TQ — the shader expects identical layout.
pub use super::flash_attn_vec_tq::FlashAttnVecTqParams as FlashAttnVecAffine4Params;

/// GPU-side parameter struct — must match MSL `FlashAttnVecTqParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecAffine4ParamsGpu {
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

/// GPU-side reduce params.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttnVecReduceParamsGpu {
    nrows: u32,
}

/// Compute NWG — same logic as TQ (16 is optimal on M5 Max).
fn compute_nwg(_kv_seq_len: u32) -> u32 {
    if let Ok(v) = std::env::var("HF2Q_AFFINE4_NWG") {
        if let Ok(n) = v.parse::<u32>() {
            if n >= 1 && n <= 32 {
                return n;
            }
        }
    }
    16
}

/// Dispatch affine4 SDPA kernel.
///
/// Same interface as `flash_attn_vec_tq::flash_attn_vec_tq`.
/// Caller must pre-rotate Q and post-rotate output via FWHT.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_vec_affine4(
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
    params: &FlashAttnVecAffine4Params,
) -> Result<()> {
    let head_dim = params.head_dim;
    let nwg = compute_nwg(params.kv_seq_len);

    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "flash_attn_vec_affine4: kv_seq_len must be > 0".into(),
        ));
    }

    let gpu_params = FlashAttnVecAffine4ParamsGpu {
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
        256 => "flash_attn_vec_affine4_dk256",
        512 => "flash_attn_vec_affine4_dk512",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "flash_attn_vec_affine4: unsupported head_dim {head_dim}"
            )));
        }
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

    // Reduce kernel (NWG > 1). Reuses the same reduce kernel as F16/TQ.
    if nwg > 1 {
        encoder.memory_barrier();

        let reduce_params = FlashAttnVecReduceParamsGpu {
            nrows: params.num_heads,
        };

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

/// Compute tmp buffer bytes — same as TQ (sized for max NWG=32).
pub fn tmp_buffer_bytes(num_heads: u32, head_dim: u32) -> usize {
    let nrows = num_heads as usize;
    let max_nwg = 32usize;
    let dv = head_dim as usize;
    (nrows * max_nwg * (dv + 2)) * std::mem::size_of::<f32>()
}

fn pad2(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}
