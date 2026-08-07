//! DeepSeek-V4 0731 expert activation and routed-output reduction.
//!
//! The activation applies the official asymmetric clamp in F32: up is clamped
//! to `[-10, 10]`, gate only has an upper clamp, and `silu(gate) * up` may be
//! scaled by one selected routing weight per row. The reduction follows the
//! official ascending expert-loop order and adds the ungated shared expert last.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::deepseek_moe_routing::DEEPSEEK_MOE_TOP_K;

pub const DEEPSEEK_MOE_INTER_DIM: usize = 2048;
pub const DEEPSEEK_MOE_HIDDEN_DIM: usize = 4096;
pub const DEEPSEEK_MOE_SWIGLU_LIMIT: f32 = 10.0;
pub const DEEPSEEK_MOE_SWIGLU_KERNEL: &str = "deepseek_moe_swiglu_f32";
pub const DEEPSEEK_MOE_WEIGHTED_REDUCE_KERNEL: &str = "deepseek_moe_weighted_reduce_f32";
const THREADS: u64 = 256;

pub static DEEPSEEK_MOE_ACTIVATION_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_moe_activation.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_MOE_SWIGLU_KERNEL,
        DEEPSEEK_MOE_ACTIVATION_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_MOE_WEIGHTED_REDUCE_KERNEL,
        DEEPSEEK_MOE_ACTIVATION_SHADER_SOURCE,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepSeekMoeActivationParams {
    count: u32,
    use_weights: u32,
}

fn checked_shape(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "deepseek_moe_activation: shape product overflows: {dims:?}"
            ))
        })
    })
}

fn validate_buffer(buf: &MlxBuffer, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let elements = checked_shape(shape)?;
    if buf.dtype() != dtype {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_moe_activation: {name} must be {dtype}, got {}",
            buf.dtype()
        )));
    }
    if buf.shape() != shape || buf.byte_len() < elements * dtype.size_of() {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_moe_activation: {name} shape must be {shape:?}, got {:?}",
            buf.shape()
        )));
    }
    Ok(())
}

fn validate_count(count: usize, name: &str) -> Result<()> {
    if count == 0 || count > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_moe_activation: {name} must be in 1..=u32::MAX"
        )));
    }
    Ok(())
}

/// Encode the asymmetric clamped SwiGLU activation.
///
/// `gate`, `up`, and `output` are F32 `[rows, 2048]`. If present,
/// `selected_weights` is F32 `[rows]` and is multiplied before expert-down.
/// A nonfinite input fails its entire row closed to zero.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_moe_swiglu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate: &MlxBuffer,
    up: &MlxBuffer,
    selected_weights: Option<&MlxBuffer>,
    output: &MlxBuffer,
    rows: usize,
) -> Result<()> {
    validate_count(rows, "rows")?;
    let activation_shape = [rows, DEEPSEEK_MOE_INTER_DIM];
    validate_buffer(gate, "gate", DType::F32, &activation_shape)?;
    validate_buffer(up, "up", DType::F32, &activation_shape)?;
    validate_buffer(output, "output", DType::F32, &activation_shape)?;
    if let Some(weights) = selected_weights {
        validate_buffer(weights, "selected_weights", DType::F32, &[rows])?;
    }
    let params = DeepSeekMoeActivationParams {
        count: rows as u32,
        use_weights: selected_weights.is_some() as u32,
    };
    let weights_or_dummy = selected_weights.unwrap_or(gate);
    let pipeline = registry.get_pipeline(DEEPSEEK_MOE_SWIGLU_KERNEL, device.metal_device())?;
    if encoder.is_capturing() {
        let range = |buffer: &MlxBuffer| {
            let start = buffer.contents_ptr() as usize;
            (start, start + buffer.byte_len())
        };
        let mut reads = vec![range(gate), range(up)];
        if let Some(weights) = selected_weights {
            reads.push(range(weights));
        }
        encoder.set_pending_buffer_ranges(reads, vec![range(output)]);
    }
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(gate)),
            (2, KernelArg::Buffer(up)),
            (3, KernelArg::Buffer(weights_or_dummy)),
            (4, KernelArg::Buffer(output)),
        ],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}

/// Encode weighted top-six routed reduction plus the shared-expert add.
///
/// Layouts are indices/weights `[tokens, 6]`, routed outputs
/// `[tokens, 6, 4096]`, and shared/output `[tokens, 4096]`, all F32 except
/// I32 indices. Contributions are accumulated by ascending expert ID (stable
/// by slot for duplicate IDs), matching the official expert loop. Invalid IDs
/// or any nonfinite dynamic value fail the entire token closed to zero.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_moe_weighted_reduce(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    indices: &MlxBuffer,
    weights: &MlxBuffer,
    routed: &MlxBuffer,
    shared: &MlxBuffer,
    output: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    validate_count(n_tokens, "n_tokens")?;
    validate_buffer(
        indices,
        "indices",
        DType::I32,
        &[n_tokens, DEEPSEEK_MOE_TOP_K],
    )?;
    validate_buffer(
        weights,
        "weights",
        DType::F32,
        &[n_tokens, DEEPSEEK_MOE_TOP_K],
    )?;
    validate_buffer(
        routed,
        "routed",
        DType::F32,
        &[n_tokens, DEEPSEEK_MOE_TOP_K, DEEPSEEK_MOE_HIDDEN_DIM],
    )?;
    let hidden_shape = [n_tokens, DEEPSEEK_MOE_HIDDEN_DIM];
    validate_buffer(shared, "shared", DType::F32, &hidden_shape)?;
    validate_buffer(output, "output", DType::F32, &hidden_shape)?;
    let params = DeepSeekMoeActivationParams {
        count: n_tokens as u32,
        use_weights: 1,
    };
    let pipeline =
        registry.get_pipeline(DEEPSEEK_MOE_WEIGHTED_REDUCE_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(indices)),
            (2, KernelArg::Buffer(weights)),
            (3, KernelArg::Buffer(routed)),
            (4, KernelArg::Buffer(shared)),
            (5, KernelArg::Buffer(output)),
        ],
        MTLSize::new(n_tokens as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}
