//! Per-row sum reduction along the last dimension of a 2-D tensor +
//! its broadcast-along-cols backward.
//!
//! Used by reverse-mode autograd in downstream crates (hf2q ADR-020
//! Track 1: KL-divergence loss composition needs Σ_j p · (log_p − log_q)
//! per row).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static ROW_SUM_SHADER_SOURCE: &str = include_str!("../shaders/row_sum.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("row_sum_f32", ROW_SUM_SHADER_SOURCE);
    registry.register_source("row_sum_backward_f32", ROW_SUM_SHADER_SOURCE);
}

/// Encode `output[b] = Σ_j input[b, j]` for a 2-D `[rows, cols]` f32 input.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_row_sum_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    cols: u32,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(MlxError::InvalidArgument(
            "row_sum_f32: rows and cols must be > 0".into(),
        ));
    }
    let in_expected = (rows as usize) * (cols as usize);
    if input.element_count() != in_expected {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_f32: input element count {} != rows({}) * cols({})",
            input.element_count(),
            rows,
            cols
        )));
    }
    if output.element_count() != rows as usize {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_f32: output element count {} != rows({})",
            output.element_count(),
            rows
        )));
    }
    if input.dtype() != DType::F32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_f32: only f32 supported; got input={} output={}",
            input.dtype(),
            output.dtype()
        )));
    }
    if params_buf.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_f32: params_buf too small (need 8 bytes, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("row_sum_f32", device)?;
    let tg_size = std::cmp::min(256, cols.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, input), (1, output), (2, params_buf)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Encode `dx[b, i] = d_out[b]` (broadcast along the cols dim).  This
/// is the backward of [`dispatch_row_sum_f32`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_row_sum_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    d_out: &MlxBuffer,
    dx: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    cols: u32,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(MlxError::InvalidArgument(
            "row_sum_backward_f32: rows and cols must be > 0".into(),
        ));
    }
    if d_out.element_count() != rows as usize {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_backward_f32: d_out element count {} != rows({})",
            d_out.element_count(),
            rows
        )));
    }
    let dx_expected = (rows as usize) * (cols as usize);
    if dx.element_count() != dx_expected {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_backward_f32: dx element count {} != rows({}) * cols({})",
            dx.element_count(),
            rows,
            cols
        )));
    }
    if d_out.dtype() != DType::F32 || dx.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_backward_f32: only f32; d_out={} dx={}",
            d_out.dtype(),
            dx.dtype()
        )));
    }
    if params_buf.byte_len() < 8 {
        return Err(MlxError::InvalidArgument(format!(
            "row_sum_backward_f32: params_buf too small (need 8 bytes, got {})",
            params_buf.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("row_sum_backward_f32", device)?;
    let tg_size = std::cmp::min(256, cols.next_power_of_two()) as u64;

    encoder.encode_threadgroups(
        pipeline,
        &[(0, d_out), (1, dx), (2, params_buf)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
