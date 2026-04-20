//! Fused scale-mask-softmax for non-flash-attention prefill.
//!
//! Replaces three sequential dispatches (scale by 1/sqrt(hd), add bf16
//! mask, row-softmax) with one kernel.  Intended for hf2q's HF2Q_NO_FA
//! path; not used elsewhere.
//!
//! See `src/shaders/scale_mask_softmax.metal` for the kernel contract.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Host-side parameters for `scale_mask_softmax_f32`.
#[derive(Debug, Clone, Copy)]
pub struct ScaleMaskSoftmaxParams {
    /// Number of rows = `n_heads * seq_q` (one threadgroup per row).
    pub rows: u32,
    /// Length of the reduction axis = `seq_k`.  Must match mask row
    /// length.
    pub cols: u32,
    /// Number of query rows per head (= `seq_q`).  Lets the kernel
    /// derive `q = row_idx % seq_q` for the shared mask index.
    pub seq_q: u32,
    /// Pre-softmax multiplicative scale (e.g. `1.0 / sqrt(head_dim)`).
    pub scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleMaskSoftmaxGpuParams {
    cols: u32,
    seq_q: u32,
    scale: f32,
    _pad: u32,
}

/// Threadgroup size for the softmax reduction.  Matches
/// `softmax::dispatch_softmax_f32`'s convention — 256 threads per
/// threadgroup yields enough parallelism at cols = seq_k = 2455 without
/// overallocating shmem.
const THREADS_PER_TG: u64 = 256;

/// Dispatches `scale_mask_softmax_f32`.
///
/// # Errors
///
/// `MlxError::InvalidArgument` on buffer-size or shape inconsistencies.
pub fn dispatch_scale_mask_softmax_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    output: &MlxBuffer,
    mask_bf16: &MlxBuffer,
    params: &ScaleMaskSoftmaxParams,
) -> Result<()> {
    if params.rows == 0 || params.cols == 0 || params.seq_q == 0 {
        return Err(MlxError::InvalidArgument(
            "scale_mask_softmax_f32: rows/cols/seq_q must be > 0".into(),
        ));
    }
    if params.rows % params.seq_q != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "scale_mask_softmax_f32: rows ({}) must be a multiple of seq_q ({})",
            params.rows, params.seq_q
        )));
    }

    let f32_sz = DType::F32.size_of();
    let bf16_sz = DType::BF16.size_of();

    let expected_input_bytes = (params.rows as usize) * (params.cols as usize) * f32_sz;
    if input.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scale_mask_softmax_f32: input too small: expected {} bytes, got {}",
            expected_input_bytes, input.byte_len()
        )));
    }
    if output.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scale_mask_softmax_f32: output too small: expected {} bytes, got {}",
            expected_input_bytes, output.byte_len()
        )));
    }
    let expected_mask_bytes = (params.seq_q as usize) * (params.cols as usize) * bf16_sz;
    if mask_bf16.byte_len() < expected_mask_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scale_mask_softmax_f32: mask too small: expected {} bytes for [{}x{}] bf16, got {}",
            expected_mask_bytes, params.seq_q, params.cols, mask_bf16.byte_len()
        )));
    }

    let pipeline = registry
        .get_pipeline("scale_mask_softmax_f32", device.metal_device())?;

    let gpu_params = ScaleMaskSoftmaxGpuParams {
        cols: params.cols,
        seq_q: params.seq_q,
        scale: params.scale,
        _pad: 0,
    };

    let threadgroups = metal::MTLSize::new(params.rows as u64, 1, 1);
    let tg_size = std::cmp::min(THREADS_PER_TG, params.cols as u64);
    let threads_per_tg = metal::MTLSize::new(tg_size, 1, 1);
    let shmem_bytes = tg_size * (f32_sz as u64);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(mask_bf16)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shmem_bytes)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}
