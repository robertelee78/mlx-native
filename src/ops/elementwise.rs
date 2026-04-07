//! GPU-accelerated elementwise operations: add, multiply, and dtype cast.
//!
//! These kernels are used for residual connections (add), scaling (multiply),
//! and dtype conversion (cast) in the inference pipeline.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL-compatible params struct for elementwise kernels.
///
/// Must match `ElementwiseParams` in `elementwise.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuElementwiseParams {
    n_elements: u32,
}

/// Threadgroup size for elementwise dispatches.
const ELEMENTWISE_TG_SIZE: u64 = 256;

/// Select the correct kernel name for a dtype-specific elementwise op.
fn elementwise_kernel_name(op: &str, dtype: DType) -> Result<&'static str> {
    match (op, dtype) {
        ("add", DType::F32) => Ok("elementwise_add_f32"),
        ("add", DType::F16) => Ok("elementwise_add_f16"),
        ("mul", DType::F32) => Ok("elementwise_mul_f32"),
        ("mul", DType::F16) => Ok("elementwise_mul_f16"),
        _ => Err(MlxError::InvalidArgument(format!(
            "elementwise_{op}: unsupported dtype {dtype}"
        ))),
    }
}

/// Encode an elementwise binary operation (add or multiply).
///
/// Both inputs and the output must have the same dtype and at least
/// `n_elements * dtype.size_of()` bytes.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// * `n_elements` is zero
/// * dtype is not F32 or F16
/// * Buffers are too small
#[allow(clippy::too_many_arguments)]
fn elementwise_binary(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    a: &MlxBuffer,
    b: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
    op: &str,
    dtype: DType,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "elementwise_{op}: n_elements must be > 0"
        )));
    }

    let elem_bytes = n_elements * dtype.size_of();
    if a.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "elementwise_{op}: input 'a' buffer too small: need {} bytes, have {}",
            elem_bytes,
            a.byte_len()
        )));
    }
    if b.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "elementwise_{op}: input 'b' buffer too small: need {} bytes, have {}",
            elem_bytes,
            b.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "elementwise_{op}: output buffer too small: need {} bytes, have {}",
            elem_bytes,
            output.byte_len()
        )));
    }

    let kernel_name = elementwise_kernel_name(op, dtype)?;
    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let gpu_params = GpuElementwiseParams {
        n_elements: n_elements as u32,
    };

    let grid = MTLSize::new(n_elements as u64, 1, 1);
    let tg = MTLSize::new(std::cmp::min(ELEMENTWISE_TG_SIZE, n_elements as u64), 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(a)),
            (1, KernelArg::Buffer(b)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}

/// Encode elementwise addition: `output = a + b`.
///
/// Both inputs and output must have the same dtype (F32 or F16).
#[allow(clippy::too_many_arguments)]
pub fn elementwise_add(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    a: &MlxBuffer,
    b: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
    dtype: DType,
) -> Result<()> {
    elementwise_binary(encoder, registry, device, a, b, output, n_elements, "add", dtype)
}

/// Encode elementwise multiplication: `output = a * b`.
///
/// Both inputs and output must have the same dtype (F32 or F16).
#[allow(clippy::too_many_arguments)]
pub fn elementwise_mul(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    a: &MlxBuffer,
    b: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
    dtype: DType,
) -> Result<()> {
    elementwise_binary(encoder, registry, device, a, b, output, n_elements, "mul", dtype)
}

/// Cast direction for dtype conversion.
pub enum CastDirection {
    /// f16 -> f32
    F16ToF32,
    /// f32 -> f16
    F32ToF16,
    /// bf16 -> f32
    BF16ToF32,
    /// f32 -> bf16
    F32ToBF16,
}

impl CastDirection {
    fn kernel_name(&self) -> &'static str {
        match self {
            CastDirection::F16ToF32 => "cast_f16_to_f32",
            CastDirection::F32ToF16 => "cast_f32_to_f16",
            CastDirection::BF16ToF32 => "cast_bf16_to_f32",
            CastDirection::F32ToBF16 => "cast_f32_to_bf16",
        }
    }

    fn input_elem_size(&self) -> usize {
        match self {
            CastDirection::F16ToF32 | CastDirection::BF16ToF32 => 2,
            CastDirection::F32ToF16 | CastDirection::F32ToBF16 => 4,
        }
    }

    fn output_elem_size(&self) -> usize {
        match self {
            CastDirection::F16ToF32 | CastDirection::BF16ToF32 => 4,
            CastDirection::F32ToF16 | CastDirection::F32ToBF16 => 2,
        }
    }
}

/// Encode a dtype cast operation.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `n_elements` is zero or buffers
/// are too small.
pub fn cast(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
    direction: CastDirection,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "cast: n_elements must be > 0".into(),
        ));
    }

    let input_bytes = n_elements * direction.input_elem_size();
    if input.byte_len() < input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "cast: input buffer too small: need {} bytes, have {}",
            input_bytes,
            input.byte_len()
        )));
    }

    let output_bytes = n_elements * direction.output_elem_size();
    if output.byte_len() < output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "cast: output buffer too small: need {} bytes, have {}",
            output_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(direction.kernel_name(), device)?;

    let gpu_params = GpuElementwiseParams {
        n_elements: n_elements as u32,
    };

    let grid = MTLSize::new(n_elements as u64, 1, 1);
    let tg = MTLSize::new(std::cmp::min(ELEMENTWISE_TG_SIZE, n_elements as u64), 1, 1);

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
