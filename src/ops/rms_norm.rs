//! RMS Normalization GPU dispatch.
//!
//! Computes: `x * rsqrt(mean(x^2) + eps) * weight`
//!
//! The mean is computed over the last dimension.  eps=1e-6 is the standard
//! value for Gemma 4.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the RMS norm kernels (embedded at compile time).
pub static RMS_NORM_SHADER_SOURCE: &str = include_str!("../shaders/rms_norm.metal");

/// Register RMS norm shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("rms_norm_f32", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_f16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_bf16", RMS_NORM_SHADER_SOURCE);
}

/// Dispatch an RMS normalization operation on the GPU.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have rms_norm sources registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, dim]` (f32, f16, or bf16).
/// * `weight`     - Weight buffer of shape `[dim]` (same dtype as input; f32 for f32/f16 kernels, bf16 for bf16).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[eps, dim]` as f32.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - Input dtype is not f32, f16, or bf16.
/// - Input element count does not match rows * dim.
pub fn dispatch_rms_norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "RMS norm rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }

    let kernel_name = match input.dtype() {
        DType::F32 => "rms_norm_f32",
        DType::F16 => "rms_norm_f16",
        DType::BF16 => "rms_norm_bf16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "RMS norm unsupported dtype: {}",
                input.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // One threadgroup per row.  Threadgroup size must be a power of 2
    // for the tree reduction to work correctly.
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;

    // Threadgroup shared memory: tg_size floats for the reduction.
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, weight),
            (2, output),
            (3, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
