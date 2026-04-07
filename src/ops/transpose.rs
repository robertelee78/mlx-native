//! GPU-accelerated 2D matrix transpose.
//!
//! Transposes a 2D matrix `[rows, cols]` to `[cols, rows]`.
//! Supports F32 and F16 dtypes.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL-compatible params struct for 2D transpose.
///
/// Must match `TransposeParams` in `elementwise.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTransposeParams {
    rows: u32,
    cols: u32,
}

/// Encode a 2D matrix transpose: `output[col, row] = input[row, col]`.
///
/// # Buffer expectations
///
/// * `input`  — `[rows, cols]` in the given dtype
/// * `output` — `[cols, rows]` in the given dtype (must be pre-allocated)
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// * `rows` or `cols` is zero
/// * `dtype` is not F32 or F16
/// * Buffers are too small
#[allow(clippy::too_many_arguments)]
pub fn transpose_2d(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<()> {
    if rows == 0 {
        return Err(MlxError::InvalidArgument(
            "transpose_2d: rows must be > 0".into(),
        ));
    }
    if cols == 0 {
        return Err(MlxError::InvalidArgument(
            "transpose_2d: cols must be > 0".into(),
        ));
    }

    let kernel_name = match dtype {
        DType::F32 => "transpose_2d_f32",
        DType::F16 => "transpose_2d_f16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "transpose_2d: unsupported dtype {dtype}"
            )));
        }
    };

    let elem_bytes = rows * cols * dtype.size_of();
    if input.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "transpose_2d: input buffer too small: need {} bytes, have {}",
            elem_bytes,
            input.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "transpose_2d: output buffer too small: need {} bytes, have {}",
            elem_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let gpu_params = GpuTransposeParams {
        rows: rows as u32,
        cols: cols as u32,
    };

    // 2D grid: (cols, rows)
    let grid = MTLSize::new(cols as u64, rows as u64, 1);
    let tg = MTLSize::new(
        std::cmp::min(16, cols as u64),
        std::cmp::min(16, rows as u64),
        1,
    );

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}
