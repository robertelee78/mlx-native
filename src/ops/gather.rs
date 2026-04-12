//! GPU-accelerated gather / index_select along dim=0.
//!
//! Gathers rows from a 2D source tensor using an index array:
//! `output[i, :] = src[indices[i], :]`
//!
//! Used for MoE scale factor gathering by expert index.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL source for the gather kernel (embedded at compile time).
pub static GATHER_SHADER_SOURCE: &str = include_str!("../shaders/gather.metal");

/// Register gather shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("gather_f32", GATHER_SHADER_SOURCE);
}

/// MSL-compatible params struct for gather.
///
/// Must match `GatherParams` in `gather.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuGatherParams {
    row_width: u32,
    n_indices: u32,
    src_rows: u32,
}

/// Dispatch a gather / index_select operation on the GPU.
///
/// Gathers rows from `src` using `indices`:
///   `output[i, j] = src[indices[i], j]`
///
/// # Arguments
///
/// * `encoder`   - Command encoder to record the dispatch into.
/// * `registry`  - Kernel registry (must have `gather_f32` registered).
/// * `device`    - Metal device for pipeline compilation.
/// * `src`       - Source buffer of shape `[src_rows, row_width]` (f32).
/// * `indices`   - Index buffer of shape `[n_indices]` (u32).
/// * `output`    - Output buffer of shape `[n_indices, row_width]` (f32).
/// * `src_rows`  - Number of rows in the source tensor.
/// * `row_width` - Number of columns (elements per row).
/// * `n_indices` - Number of indices to gather.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if any dimension is 0 or buffers are
/// too small.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gather_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    indices: &MlxBuffer,
    output: &MlxBuffer,
    src_rows: u32,
    row_width: u32,
    n_indices: u32,
) -> Result<()> {
    if src_rows == 0 || row_width == 0 || n_indices == 0 {
        return Err(MlxError::InvalidArgument(
            "gather_f32: all dimensions must be > 0".into(),
        ));
    }

    let src_bytes = src_rows as usize * row_width as usize * 4;
    if src.byte_len() < src_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "gather_f32: src buffer too small: need {} bytes, have {}",
            src_bytes,
            src.byte_len()
        )));
    }
    let idx_bytes = n_indices as usize * 4;
    if indices.byte_len() < idx_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "gather_f32: indices buffer too small: need {} bytes, have {}",
            idx_bytes,
            indices.byte_len()
        )));
    }
    let out_bytes = n_indices as usize * row_width as usize * 4;
    if output.byte_len() < out_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "gather_f32: output buffer too small: need {} bytes, have {}",
            out_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("gather_f32", device)?;

    let gpu_params = GpuGatherParams {
        row_width,
        n_indices,
        src_rows,
    };

    let grid = MTLSize::new(row_width as u64, n_indices as u64, 1);
    let tg = MTLSize::new(std::cmp::min(256, row_width as u64), 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(indices)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}
