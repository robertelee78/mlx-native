//! L2 Normalization GPU dispatch.
//!
//! Computes: `x / sqrt(sum(x^2) + eps)` over the last dimension.
//!
//! Used by Gated DeltaNet to normalize Q and K after the conv1d state update
//! (ADR-013 Decision 3; spec derived from the mathematical definition of
//! L2 norm, not from llama.cpp source).
//!
//! Reduction is always performed in f32 for numerical stability regardless
//! of input dtype.
//!
//! # Invariants
//!
//! * Input and output share the same shape `[rows, dim]` and dtype.
//! * `params_buf` must hold exactly `[eps, dim as f32]` as two contiguous f32.
//! * `rows > 0`, `dim > 0`, `input.elements() == rows * dim`.
//!
//! # Threadgroup shape
//!
//! One threadgroup per row; threadgroup size = `min(256, next_power_of_two(dim))`.
//! Shared memory of `tg_size` floats is used for the tree reduction.
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the L2 norm kernels (embedded at compile time).
pub static L2_NORM_SHADER_SOURCE: &str = include_str!("../shaders/l2_norm.metal");

/// Register L2 norm shader sources with the given kernel registry.
///
/// Currently registered via `KernelRegistry::new()`'s static table;
/// this free function exists for symmetry with other ops' registration
/// helpers and may be used by tests that construct an empty registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("l2_norm_f32", L2_NORM_SHADER_SOURCE);
    registry.register_source("l2_norm_f16", L2_NORM_SHADER_SOURCE);
    registry.register_source("l2_norm_bf16", L2_NORM_SHADER_SOURCE);
}

/// Dispatch an L2 normalization operation on the GPU.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have l2_norm sources registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer of shape `[rows, dim]` (f32, f16, or bf16).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[eps, dim]` as two f32 values.
/// * `rows`       - Number of rows to normalize.
/// * `dim`        - Dimension of the last axis.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - `rows == 0` or `dim == 0`.
/// - `input.element_count() != rows * dim`.
/// - `output.element_count() != rows * dim`.
/// - Input and output dtypes differ.
/// - Input dtype is not f32, f16, or bf16.
pub fn dispatch_l2_norm(
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
            "L2 norm rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "L2 norm input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "L2 norm output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }
    if input.dtype() != output.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "L2 norm input/output dtype mismatch: {} vs {}",
            input.dtype(),
            output.dtype()
        )));
    }

    let kernel_name = match input.dtype() {
        DType::F32 => "l2_norm_f32",
        DType::F16 => "l2_norm_f16",
        DType::BF16 => "l2_norm_bf16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "L2 norm unsupported dtype: {}",
                input.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let tg_size = std::cmp::min(256, dim.next_power_of_two()) as u64;
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
