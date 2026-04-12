//! GELU activation (pytorch_tanh variant) GPU dispatch.
//!
//! Computes: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
//!
//! This is the exact variant used by Gemma 4. It is **not** the erf-based
//! GELU approximation.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the GELU kernels (embedded at compile time).
pub static GELU_SHADER_SOURCE: &str = include_str!("../shaders/gelu.metal");

/// Register GELU shader sources with the given kernel registry.
///
/// This must be called before dispatching any GELU operations.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("gelu_f32", GELU_SHADER_SOURCE);
    registry.register_source("gelu_f16", GELU_SHADER_SOURCE);
    registry.register_source("gelu_bf16", GELU_SHADER_SOURCE);
}

/// Dispatch a GELU activation on the GPU.
///
/// # Arguments
///
/// * `encoder`  - Command encoder to record the dispatch into.
/// * `registry` - Kernel registry (must have GELU sources registered).
/// * `device`   - Metal device for pipeline compilation.
/// * `input`    - Input buffer (f32, f16, or bf16).
/// * `output`   - Output buffer (same dtype and shape as input).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - Input dtype is not f32, f16, or bf16.
/// - Input and output element counts do not match.
pub fn dispatch_gelu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
) -> Result<()> {
    let n = input.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "GELU input must have at least one element".into(),
        ));
    }
    if output.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "GELU output element count {} != input element count {}",
            output.element_count(),
            n
        )));
    }

    let kernel_name = match input.dtype() {
        DType::F32 => "gelu_f32",
        DType::F16 => "gelu_f16",
        DType::BF16 => "gelu_bf16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "GELU unsupported dtype: {}",
                input.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let thread_count = n as u64;
    let threadgroup_size = std::cmp::min(256, thread_count);

    encoder.encode(
        pipeline,
        &[(0, input), (1, output)],
        MTLSize::new(thread_count, 1, 1),
        MTLSize::new(threadgroup_size, 1, 1),
    );

    Ok(())
}
