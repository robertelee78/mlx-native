//! Numerically stable softmax GPU dispatch.
//!
//! Computes softmax along the last dimension of a 2D tensor using the
//! subtract-max trick for numerical stability.  All accumulations use f32
//! even when inputs are f16 to prevent overflow.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the softmax kernels (embedded at compile time).
pub static SOFTMAX_SHADER_SOURCE: &str = include_str!("../shaders/softmax.metal");

/// Register softmax shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("softmax_f32", SOFTMAX_SHADER_SOURCE);
    registry.register_source("softmax_f16", SOFTMAX_SHADER_SOURCE);
}

/// Dispatch a softmax operation on the GPU.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have softmax sources registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, cols]` (f32 or f16).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[cols, 0]` as f32.
/// * `rows`       - Number of rows.
/// * `cols`       - Number of columns (softmax dimension).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - Input dtype is not f32 or f16.
/// - Input element count does not match rows * cols.
pub fn dispatch_softmax(
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
            "Softmax rows and cols must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (cols as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "Softmax input element count {} != rows({}) * cols({})",
            input.element_count(),
            rows,
            cols
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "Softmax output element count {} != rows({}) * cols({})",
            output.element_count(),
            rows,
            cols
        )));
    }

    let kernel_name = match input.dtype() {
        DType::F32 => "softmax_f32",
        DType::F16 => "softmax_f16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "Softmax unsupported dtype: {}",
                input.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // One threadgroup per row.  Threadgroup size must be a power of 2
    // for the tree reduction to work correctly.
    let tg_size = std::cmp::min(256, cols.next_power_of_two()) as u64;

    // Threadgroup shared memory: tg_size floats for the reduction.
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, input), (1, output), (2, params_buf)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
