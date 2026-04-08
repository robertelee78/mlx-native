//! Softcap (tanh-based logit capping) GPU dispatch.
//!
//! Computes: `tanh(logits / cap) * cap`
//!
//! This bounds output logits to the range `(-cap, +cap)`.  Gemma 4 uses
//! `cap = 30.0` for final logit capping.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the softcap kernels (embedded at compile time).
pub static SOFTCAP_SHADER_SOURCE: &str = include_str!("../shaders/softcap.metal");

/// Register softcap shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("softcap_f32", SOFTCAP_SHADER_SOURCE);
    registry.register_source("softcap_f16", SOFTCAP_SHADER_SOURCE);
}

/// Dispatch a softcap operation on the GPU.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (must have softcap sources registered).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer (f32 or f16).
/// * `output`     - Output buffer (same dtype and shape as input).
/// * `params_buf` - Params buffer containing `[cap, n_elements_as_f32_bits]` as two f32 values.
/// * `cap`        - The capping value (e.g. 30.0).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - Input dtype is not f32 or f16.
/// - Input and output element counts do not match.
/// - Cap is not positive.
pub fn dispatch_softcap(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    cap: f32,
) -> Result<()> {
    if cap <= 0.0 {
        return Err(MlxError::InvalidArgument(format!(
            "Softcap cap must be positive, got {}",
            cap
        )));
    }

    let n = input.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "Softcap input must have at least one element".into(),
        ));
    }
    if output.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "Softcap output element count {} != input element count {}",
            output.element_count(),
            n
        )));
    }

    let _ = cap; // cap value is passed via params_buf

    let kernel_name = match input.dtype() {
        DType::F32 => "softcap_f32",
        DType::F16 => "softcap_f16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "Softcap unsupported dtype: {}",
                input.dtype()
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let threadgroup_size: u64 = std::cmp::min(256, n as u64);
    let threadgroup_count = (n as u64 + threadgroup_size - 1) / threadgroup_size;

    encoder.encode_threadgroups(
        pipeline,
        &[(0, input), (1, output), (2, params_buf)],
        MTLSize::new(threadgroup_count, 1, 1),
        MTLSize::new(threadgroup_size, 1, 1),
    );

    Ok(())
}
