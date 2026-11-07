//! Row-exact F32 mat-vec over a few input rows (`dense_matvec_f32_rows`).
//!
//! `output[r, n] = sum_k input[r, k] * weight[n, k]` for `r < rows`, with F32
//! input, weight and output. One dispatch reads each weight row once for all
//! `rows` inputs. The accumulation order of output `(r, n)` does not depend on
//! `rows` or on the other rows of the dispatch, so a row's result is bitwise
//! the same whether it runs alone (`rows = 1`) or in a batch; see the shader
//! header for the technique.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Metal shader source.
pub static DENSE_MATVEC_F32_ROWS_SHADER_SOURCE: &str =
    include_str!("../shaders/dense_matvec_f32_rows.metal");

/// Largest row count one dispatch accepts.
pub const DENSE_MATVEC_F32_ROWS_MAX_ROWS: u32 = 8;

const NSG: u64 = 4;
const NR0: u64 = 2;
const NW: u64 = 32;

/// Register the `dense_matvec_f32_rows` pipeline source.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("dense_matvec_f32_rows", DENSE_MATVEC_F32_ROWS_SHADER_SOURCE);
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseMatvecF32RowsGpuParams {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Encode `output[rows, n] = input[rows, k] @ weight[n, k]^T` (all F32).
///
/// # Errors
///
/// `MlxError::InvalidArgument` when `rows` is not in `1..=8`, `n` or `k` is
/// zero, `k % 4 != 0`, a buffer is not F32, too small, or the input/weight
/// view does not start on a 16-byte boundary.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dense_matvec_f32_rows(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    rows: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if !(1..=DENSE_MATVEC_F32_ROWS_MAX_ROWS).contains(&rows) {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matvec_f32_rows: rows must be in 1..={DENSE_MATVEC_F32_ROWS_MAX_ROWS}, got {rows}"
        )));
    }
    if n == 0 || k == 0 || k % 4 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matvec_f32_rows: n and k must be nonzero and k a multiple of 4, got n={n} k={k}"
        )));
    }
    for (buffer, label) in [(input, "input"), (weight, "weight"), (output, "output")] {
        if buffer.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matvec_f32_rows: {label} must be F32, got {:?}",
                buffer.dtype()
            )));
        }
    }
    for (buffer, label) in [(input, "input"), (weight, "weight")] {
        if buffer.byte_offset() % 16 != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matvec_f32_rows: {label} view offset {} is not 16-byte aligned",
                buffer.byte_offset()
            )));
        }
    }
    let f32_size = DType::F32.size_of();
    for (buffer, elements, label) in [
        (input, rows as usize * k as usize, "input"),
        (weight, n as usize * k as usize, "weight"),
        (output, rows as usize * n as usize, "output"),
    ] {
        if buffer.data_byte_len() < elements * f32_size {
            return Err(MlxError::InvalidArgument(format!(
                "dense_matvec_f32_rows: {label} too small (expected {} bytes, got {})",
                elements * f32_size,
                buffer.data_byte_len()
            )));
        }
    }

    let pipeline = registry.get_pipeline_with_constants(
        "dense_matvec_f32_rows",
        device.metal_device(),
        &[],
        &[(0, rows as i32)],
    )?;
    let params = DenseMatvecF32RowsGpuParams { n, k, _pad0: 0, _pad1: 0 };
    let shared_bytes = NR0 * u64::from(rows) * NSG * NW * f32_size as u64;
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        &[(0, shared_bytes)],
        MTLSize::new(u64::from(n).div_ceil(NR0), 1, 1),
        MTLSize::new(NW, NSG, 1),
    );
    Ok(())
}
