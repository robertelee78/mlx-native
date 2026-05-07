//! Elementwise natural log forward + backward.
//!
//! Used by reverse-mode autograd in downstream crates (hf2q ADR-020
//! Track 1: log_softmax + KL-div composition).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static LOG_SHADER_SOURCE: &str = include_str!("../shaders/log_elementwise.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("log_f32", LOG_SHADER_SOURCE);
    registry.register_source("log_backward_f32", LOG_SHADER_SOURCE);
}

/// Encode `output[i] = log(input[i])` for f32 input.
///
/// Caller must ensure `input` is strictly positive — the kernel does
/// not check; `log(x ≤ 0)` produces NaN or `-inf` per IEEE 754.
pub fn dispatch_log_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
) -> Result<()> {
    let n = input.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "log_f32: input must have at least one element".into(),
        ));
    }
    if output.element_count() != n {
        return Err(MlxError::InvalidArgument(format!(
            "log_f32: output element count {} != input element count {}",
            output.element_count(),
            n
        )));
    }
    if input.dtype() != DType::F32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "log_f32: only f32 supported; got input={} output={}",
            input.dtype(),
            output.dtype()
        )));
    }

    let pipeline = registry.get_pipeline("log_f32", device)?;
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

/// Encode `dx[i] = dy[i] / x[i]` (the backward pass for elementwise
/// log).  `x` is the FORWARD INPUT, not the forward output.
pub fn dispatch_log_backward_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    dy: &MlxBuffer,
    dx: &MlxBuffer,
) -> Result<()> {
    let n = x.element_count();
    if n == 0 {
        return Err(MlxError::InvalidArgument(
            "log_backward_f32: x must have at least one element".into(),
        ));
    }
    for (label, buf) in [("dy", dy), ("dx", dx)] {
        if buf.element_count() != n {
            return Err(MlxError::InvalidArgument(format!(
                "log_backward_f32: {label} element count {} != x element count {}",
                buf.element_count(),
                n
            )));
        }
    }
    for (label, buf) in [("x", x), ("dy", dy), ("dx", dx)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "log_backward_f32: {label} dtype {} not f32",
                buf.dtype()
            )));
        }
    }

    let pipeline = registry.get_pipeline("log_backward_f32", device)?;
    let thread_count = n as u64;
    let threadgroup_size = std::cmp::min(256, thread_count);

    encoder.encode(
        pipeline,
        &[(0, x), (1, dy), (2, dx)],
        MTLSize::new(thread_count, 1, 1),
        MTLSize::new(threadgroup_size, 1, 1),
    );

    Ok(())
}
