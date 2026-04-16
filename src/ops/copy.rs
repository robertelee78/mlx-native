//! GPU-accelerated strided copy for making tensors contiguous.
//!
//! Copies a 2D strided tensor to a contiguous layout:
//!   `dst[row * cols + col] = src[row * stride_row + col * stride_col]`
//!
//! Used after transpose/permute operations to produce contiguous memory.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL source for the strided copy kernel (embedded at compile time).
pub static COPY_SHADER_SOURCE: &str = include_str!("../shaders/copy.metal");

/// Register strided copy shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("strided_copy_f32", COPY_SHADER_SOURCE);
    registry.register_source("offset_copy_f32", COPY_SHADER_SOURCE);
}

/// MSL-compatible params struct for strided copy.
///
/// Must match `StridedCopyParams` in `copy.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuStridedCopyParams {
    rows: u32,
    cols: u32,
    stride_row: u32,
    stride_col: u32,
}

/// Parameters for a strided copy operation.
pub struct StridedCopyParams {
    /// Number of rows in the output.
    pub rows: u32,
    /// Number of columns in the output.
    pub cols: u32,
    /// Stride (in elements) between rows in the source.
    pub stride_row: u32,
    /// Stride (in elements) between columns in the source.
    pub stride_col: u32,
}

/// Dispatch a strided copy operation on the GPU.
///
/// Copies a 2D strided tensor to contiguous layout:
///   `dst[row * cols + col] = src[row * stride_row + col * stride_col]`
///
/// # Arguments
///
/// * `encoder`  - Command encoder to record the dispatch into.
/// * `registry` - Kernel registry (must have `strided_copy_f32` registered).
/// * `device`   - Metal device for pipeline compilation.
/// * `src`      - Source buffer (f32, strided layout).
/// * `dst`      - Destination buffer (f32, contiguous output).
/// * `params`   - Copy parameters (rows, cols, strides).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if dimensions are 0 or buffers are
/// too small.
pub fn dispatch_strided_copy_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    params: &StridedCopyParams,
) -> Result<()> {
    if params.rows == 0 || params.cols == 0 {
        return Err(MlxError::InvalidArgument(
            "strided_copy_f32: rows and cols must be > 0".into(),
        ));
    }

    // Check destination buffer size (contiguous output).
    let dst_bytes = params.rows as usize * params.cols as usize * 4;
    if dst.byte_len() < dst_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "strided_copy_f32: dst buffer too small: need {} bytes, have {}",
            dst_bytes,
            dst.byte_len()
        )));
    }

    // Source buffer must be large enough for the maximum strided access.
    // Max index = (rows-1)*stride_row + (cols-1)*stride_col
    let max_src_idx = (params.rows as usize - 1) * params.stride_row as usize
        + (params.cols as usize - 1) * params.stride_col as usize;
    let src_min_bytes = (max_src_idx + 1) * 4;
    if src.byte_len() < src_min_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "strided_copy_f32: src buffer too small: need at least {} bytes for stride access, have {}",
            src_min_bytes,
            src.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("strided_copy_f32", device)?;

    let gpu_params = GpuStridedCopyParams {
        rows: params.rows,
        cols: params.cols,
        stride_row: params.stride_row,
        stride_col: params.stride_col,
    };

    let grid = MTLSize::new(params.cols as u64, params.rows as u64, 1);
    let tg = MTLSize::new(std::cmp::min(256, params.cols as u64), 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}

/// GPU-side params for offset copy.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOffsetCopyParams {
    src_offset: u32,
    dst_offset: u32,
    count: u32,
}

/// Copy `count` f32 elements from `src[src_offset..]` to `dst[dst_offset..]`.
///
/// Used during prefill to scatter/gather rows between large prefill buffers
/// and single-token activation buffers.
pub fn dispatch_copy_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    src_offset: usize,
    dst_offset: usize,
    count: usize,
) -> Result<()> {
    if count == 0 {
        return Ok(()); // no-op
    }
    let src_end_bytes = (src_offset + count) * 4;
    let dst_end_bytes = (dst_offset + count) * 4;
    if src.byte_len() < src_end_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "offset_copy_f32: src too small: need {} bytes (offset {} + count {}), have {}",
            src_end_bytes, src_offset, count, src.byte_len()
        )));
    }
    if dst.byte_len() < dst_end_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "offset_copy_f32: dst too small: need {} bytes (offset {} + count {}), have {}",
            dst_end_bytes, dst_offset, count, dst.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("offset_copy_f32", device)?;

    let gpu_params = GpuOffsetCopyParams {
        src_offset: src_offset as u32,
        dst_offset: dst_offset as u32,
        count: count as u32,
    };

    let grid = MTLSize::new(count as u64, 1, 1);
    let tg = MTLSize::new(std::cmp::min(256, count as u64), 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}
