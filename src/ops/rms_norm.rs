//! RMS Normalization GPU dispatch.
//!
//! Computes: `x * rsqrt(mean(x^2) + eps) * weight`
//!
//! The mean is computed over the last dimension.  eps=1e-6 is the standard
//! value for Gemma 4.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::{CapturedOpKind, CommandEncoder};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the RMS norm kernels (embedded at compile time).
pub static RMS_NORM_SHADER_SOURCE: &str = include_str!("../shaders/rms_norm.metal");

/// Register RMS norm shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("rms_norm_f32", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_f16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_bf16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_no_scale_bf16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_no_scale_f32", RMS_NORM_SHADER_SOURCE);
    // Fused RMS norm + elementwise multiply (Phase 4e.2)
    registry.register_source("rms_norm_mul_f32", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_mul_f16", RMS_NORM_SHADER_SOURCE);
    registry.register_source("rms_norm_mul_bf16", RMS_NORM_SHADER_SOURCE);
}

/// Select the fused RMS norm + multiply kernel name based on input dtype.
fn fused_rms_norm_mul_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("rms_norm_mul_f32"),
        DType::F16 => Ok("rms_norm_mul_f16"),
        DType::BF16 => Ok("rms_norm_mul_bf16"),
        _ => Err(MlxError::InvalidArgument(format!(
            "Fused RMS norm+mul unsupported dtype: {}",
            dtype
        ))),
    }
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

    // Tag for the fusion pass (Phase 4e.2): RMS norm can fuse with a
    // subsequent elementwise multiply.
    encoder.set_op_kind(CapturedOpKind::RmsNorm);

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

/// Dispatch an RMS normalization without learned scale (bf16 only).
///
/// Computes: `output = x * rsqrt(mean(x^2) + eps)` — no weight multiplication.
/// Used for per-head V normalization in Gemma 4.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have rms_norm_no_scale_bf16 registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, dim]` (bf16).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[eps, dim]` as f32.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
pub fn dispatch_rms_norm_no_scale_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "RMS norm no_scale: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale: output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }

    let pipeline = registry.get_pipeline("rms_norm_no_scale_bf16", device)?;

    // One threadgroup per row.  Threadgroup size must be a power of 2
    // for the tree reduction to work correctly.
    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;

    // Threadgroup shared memory: tg_size floats for the reduction.
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, output),
            (2, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch an RMS normalization without learned scale (f32).
///
/// Computes: `output = x * rsqrt(mean(x^2) + eps)` -- no weight multiplication.
/// Used for per-head V normalization in Gemma 4 when activations are f32.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have rms_norm_no_scale_f32 registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, dim]` (f32).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[eps, dim]` as f32.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
pub fn dispatch_rms_norm_no_scale_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "RMS norm no_scale f32: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale f32: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "RMS norm no_scale f32: output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }

    let pipeline = registry.get_pipeline("rms_norm_no_scale_f32", device)?;

    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, output),
            (2, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch a fused RMS normalization + elementwise multiply.
///
/// Computes: `output = (input * rsqrt(mean(input^2) + eps) * weight) * scale`
///
/// This replaces the two-dispatch pattern:
///   1. `rms_norm(input, weight) -> tmp`
///   2. `elementwise_mul(tmp, scale) -> output`
///
/// with a single kernel pass, eliminating one barrier and one global memory
/// round-trip for the intermediate `tmp` buffer.
///
/// # Arguments
///
/// * `encoder`      - Command encoder to record the dispatch into.
/// * `registry`     - Kernel registry.
/// * `device`       - Metal device for pipeline compilation.
/// * `input`        - Input buffer of shape `[rows, dim]`.
/// * `norm_weight`  - Norm weight buffer of shape `[dim]`.
/// * `scale_weight` - Scale (MUL operand) buffer of shape `[rows, dim]`.
/// * `output`       - Output buffer of shape `[rows, dim]`.
/// * `params_buf`   - Params buffer containing `[eps, dim]` as f32.
/// * `rows`         - Number of rows.
/// * `dim`          - Dimension of the last axis.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rms_norm_mul(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    norm_weight: &MlxBuffer,
    scale_weight: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "Fused RMS norm+mul: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "Fused RMS norm+mul: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }

    let kernel_name = fused_rms_norm_mul_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, input),
            (1, norm_weight),
            (2, scale_weight),
            (3, output),
            (4, params_buf),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
