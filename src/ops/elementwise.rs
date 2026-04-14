//! GPU-accelerated elementwise operations: add, multiply, and dtype cast.
//!
//! These kernels are used for residual connections (add), scaling (multiply),
//! and dtype conversion (cast) in the inference pipeline.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::{CapturedOpKind, CommandEncoder};
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
        ("add", DType::BF16) => Ok("elementwise_add_bf16"),
        ("mul", DType::F32) => Ok("elementwise_mul_f32"),
        ("mul", DType::F16) => Ok("elementwise_mul_f16"),
        ("mul", DType::BF16) => Ok("elementwise_mul_bf16"),
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

    // Tag for the fusion pass (Phase 4e.2): elementwise mul/add can fuse
    // with a preceding RMS norm.
    let op_tag = match op {
        "mul" => CapturedOpKind::ElemMul,
        "add" => CapturedOpKind::ElemAdd,
        _ => CapturedOpKind::Other,
    };
    encoder.set_op_kind(op_tag);

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

/// MSL-compatible params struct for scalar multiplication.
///
/// Must match `ScalarMulParams` in `elementwise.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuScalarMulParams {
    scalar: f32,
    count: u32,
}

/// Encode scalar multiplication: `output[i] = input[i] * scalar` (bf16).
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry.
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer (bf16).
/// * `output`     - Output buffer (bf16, same size as input).
/// * `n_elements` - Number of elements to process.
/// * `scalar`     - The f32 scalar to multiply by.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `n_elements` is zero or buffers are too small.
pub fn scalar_mul_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
    scalar: f32,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "scalar_mul_bf16: n_elements must be > 0".into(),
        ));
    }

    let elem_bytes = n_elements * DType::BF16.size_of();
    if input.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scalar_mul_bf16: input buffer too small: need {} bytes, have {}",
            elem_bytes,
            input.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scalar_mul_bf16: output buffer too small: need {} bytes, have {}",
            elem_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("scalar_mul_bf16", device)?;

    let gpu_params = GpuScalarMulParams {
        scalar,
        count: n_elements as u32,
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

/// Encode scalar multiplication: `output[i] = input[i] * scalar` (f32).
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry.
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer (f32).
/// * `output`     - Output buffer (f32, same size as input).
/// * `n_elements` - Number of elements to process.
/// * `scalar`     - The f32 scalar to multiply by.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `n_elements` is zero or buffers are too small.
pub fn scalar_mul_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
    scalar: f32,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "scalar_mul_f32: n_elements must be > 0".into(),
        ));
    }

    let elem_bytes = n_elements * DType::F32.size_of();
    if input.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scalar_mul_f32: input buffer too small: need {} bytes, have {}",
            elem_bytes,
            input.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scalar_mul_f32: output buffer too small: need {} bytes, have {}",
            elem_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("scalar_mul_f32", device)?;

    let gpu_params = GpuScalarMulParams {
        scalar,
        count: n_elements as u32,
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

/// MSL-compatible params struct for embedding_gather_scale_f32 kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEmbedGatherScaleParams {
    scale: f32,
    hidden_size: u32,
    token_id: u32,
}

/// Encode an embedding gather + scale: `output[i] = embed[token_id * hs + i] * scale`.
///
/// # Arguments
///
/// * `encoder`     - Command encoder.
/// * `registry`    - Kernel registry.
/// * `device`      - Metal device.
/// * `embed_table` - f32 `[vocab_size * hidden_size]`.
/// * `output`      - f32 `[hidden_size]`.
/// * `token_id`    - Token index into the embedding table.
/// * `hidden_size` - Embedding dimension.
/// * `scale`       - Scale factor (e.g. sqrt(hidden_size)).
pub fn embedding_gather_scale_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    embed_table: &MlxBuffer,
    output: &MlxBuffer,
    token_id: u32,
    hidden_size: usize,
    scale: f32,
) -> Result<()> {
    if hidden_size == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_gather_scale_f32: hidden_size must be > 0".into(),
        ));
    }
    let out_bytes = hidden_size * std::mem::size_of::<f32>();
    if output.byte_len() < out_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_gather_scale_f32: output too small: need {} bytes, have {}",
            out_bytes, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("embedding_gather_scale_f32", device)?;

    let gpu_params = GpuEmbedGatherScaleParams {
        scale,
        hidden_size: hidden_size as u32,
        token_id,
    };

    let grid = MTLSize::new(hidden_size as u64, 1, 1);
    let tg = MTLSize::new(std::cmp::min(ELEMENTWISE_TG_SIZE, hidden_size as u64), 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(embed_table)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}

/// Cast f32 to bf16 using an externally-provided encoder (no commit).
///
/// Encodes the `cast_f32_to_bf16` kernel into the given encoder without
/// committing or waiting.  Use this to chain the cast into a mega-encoder
/// alongside other GPU work, avoiding CPU round-trips.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (mutable for lazy pipeline compilation).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer (f32).
/// * `output`     - Output buffer (bf16, pre-allocated with `n_elements * 2` bytes).
/// * `n_elements` - Number of elements to cast.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `n_elements` is zero or buffers are
/// too small.
pub fn dispatch_cast_f32_to_bf16_with_encoder(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: u32,
) -> Result<()> {
    cast(
        encoder,
        registry,
        device,
        input,
        output,
        n_elements as usize,
        CastDirection::F32ToBF16,
    )
}

/// Cast bf16 to f32 using an externally-provided encoder (no commit).
///
/// Encodes the `cast_bf16_to_f32` kernel into the given encoder without
/// committing or waiting.  Use this to chain the cast into a mega-encoder
/// alongside other GPU work, avoiding CPU round-trips.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (mutable for lazy pipeline compilation).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer (bf16).
/// * `output`     - Output buffer (f32, pre-allocated with `n_elements * 4` bytes).
/// * `n_elements` - Number of elements to cast.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `n_elements` is zero or buffers are
/// too small.
pub fn dispatch_cast_bf16_to_f32_with_encoder(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: u32,
) -> Result<()> {
    cast(
        encoder,
        registry,
        device,
        input,
        output,
        n_elements as usize,
        CastDirection::BF16ToF32,
    )
}

/// Scale bf16 values by a scalar using an externally-provided encoder (no commit).
///
/// Encodes `output[i] = input[i] * scalar` (bf16) into the given encoder
/// without committing or waiting.  Use this to chain the scale into a
/// mega-encoder alongside other GPU work, avoiding CPU round-trips.
///
/// # Arguments
///
/// * `encoder`    - Command encoder to record the dispatch into.
/// * `registry`   - Kernel registry (mutable for lazy pipeline compilation).
/// * `device`     - Metal device for pipeline compilation.
/// * `input`      - Input buffer (bf16).
/// * `output`     - Output buffer (bf16, same size as input).
/// * `n_elements` - Number of elements to process.
/// * `scalar`     - The f32 scalar to multiply by (e.g. `sqrt(hidden_size)`).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `n_elements` is zero or buffers are
/// too small.
pub fn dispatch_scalar_mul_bf16_with_encoder(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: u32,
    scalar: f32,
) -> Result<()> {
    scalar_mul_bf16(
        encoder,
        registry,
        device,
        input,
        output,
        n_elements as usize,
        scalar,
    )
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
