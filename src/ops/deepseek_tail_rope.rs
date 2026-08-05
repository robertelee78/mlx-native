//! DeepSeek-V4 interleaved RoPE over the tail of each attention head.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

const F32_TO_BF16_KERNEL: &str = "deepseek_tail_rope_f32_to_bf16";
const BF16_KERNEL: &str = "deepseek_tail_rope_bf16";

pub static DEEPSEEK_TAIL_ROPE_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_tail_rope.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(F32_TO_BF16_KERNEL, DEEPSEEK_TAIL_ROPE_SHADER_SOURCE);
    registry.register_source(BF16_KERNEL, DEEPSEEK_TAIL_ROPE_SHADER_SOURCE);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeepSeekTailRopeParams {
    pub batch: u32,
    pub seq_len: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub inverse: u32,
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "deepseek_tail_rope: shape product overflows: {dims:?}"
            ))
        })
    })
}

fn validate(
    input: &MlxBuffer,
    output: &MlxBuffer,
    positions: &MlxBuffer,
    frequencies: &MlxBuffer,
    params: &DeepSeekTailRopeParams,
    input_dtype: DType,
) -> Result<(usize, usize)> {
    if params.batch == 0 || params.seq_len == 0 || params.heads == 0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_tail_rope: batch, seq_len, and heads must be nonzero".into(),
        ));
    }
    if params.rope_dim == 0
        || params.rope_dim % 2 != 0
        || params.rope_dim > params.head_dim
        || params.inverse > 1
    {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_tail_rope: invalid head/rope/inverse parameters: {}/{}/{}",
            params.head_dim, params.rope_dim, params.inverse
        )));
    }
    let shape = [
        params.batch as usize,
        params.seq_len as usize,
        params.heads as usize,
        params.head_dim as usize,
    ];
    let elements = checked_product(&shape)?;
    if input.dtype() != input_dtype || input.shape() != shape {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_tail_rope: input must be {input_dtype} {shape:?}, got {} {:?}",
            input.dtype(),
            input.shape()
        )));
    }
    if output.dtype() != DType::BF16 || output.shape() != shape {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_tail_rope: output must be bf16 {shape:?}, got {} {:?}",
            output.dtype(),
            output.shape()
        )));
    }
    if input.byte_len() < elements * input_dtype.size_of()
        || output.byte_len() < elements * DType::BF16.size_of()
    {
        return Err(MlxError::InvalidArgument(
            "deepseek_tail_rope: input or output buffer is too short".into(),
        ));
    }
    if positions.dtype() != DType::U32
        || positions.shape() != [params.seq_len as usize]
        || frequencies.dtype() != DType::F32
        || frequencies.shape() != [params.rope_dim as usize / 2]
    {
        return Err(MlxError::InvalidArgument(
            "deepseek_tail_rope: position/frequency buffer contract mismatch".into(),
        ));
    }
    if frequencies
        .as_slice::<f32>()?
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(MlxError::InvalidArgument(
            "deepseek_tail_rope: frequencies must be finite and positive".into(),
        ));
    }
    let vectors = checked_product(&shape[..3])?;
    let work_width =
        params.head_dim as usize - params.rope_dim as usize + params.rope_dim as usize / 2;
    Ok((vectors, work_width))
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    positions: &MlxBuffer,
    frequencies: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekTailRopeParams,
    input_dtype: DType,
    kernel: &str,
) -> Result<()> {
    let (vectors, work_width) =
        validate(input, output, positions, frequencies, params, input_dtype)?;
    let pipeline = registry.get_pipeline(kernel, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(positions)),
            (2, KernelArg::Buffer(frequencies)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(as_bytes(params))),
        ],
        MTLSize::new(work_width as u64, vectors as u64, 1),
        MTLSize::new(work_width.min(256) as u64, 1, 1),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_tail_rope_f32_to_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    positions: &MlxBuffer,
    frequencies: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekTailRopeParams,
) -> Result<()> {
    dispatch(
        encoder,
        registry,
        device,
        input,
        positions,
        frequencies,
        output,
        params,
        DType::F32,
        F32_TO_BF16_KERNEL,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_tail_rope_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    positions: &MlxBuffer,
    frequencies: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekTailRopeParams,
) -> Result<()> {
    dispatch(
        encoder,
        registry,
        device,
        input,
        positions,
        frequencies,
        output,
        params,
        DType::BF16,
        BF16_KERNEL,
    )
}
