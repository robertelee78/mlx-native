//! Direct dense GGUF embedding lookup preserving F32, F16, or BF16 storage.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

const KERNEL_BF16: &str = "embedding_gather_bf16_f32";
const KERNEL_F16: &str = "embedding_gather_f16_f32";
const KERNEL_F32: &str = "embedding_gather_f32_f32";

pub static EMBEDDING_DENSE_SHADER_SOURCE: &str = include_str!("../shaders/embedding_dense.metal");

pub fn register(registry: &mut KernelRegistry) {
    for kernel in [KERNEL_BF16, KERNEL_F16, KERNEL_F32] {
        registry.register_source(kernel, EMBEDDING_DENSE_SHADER_SOURCE);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingDenseParams {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_tokens: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEmbeddingDenseParams {
    vocab_size: u32,
    embed_dim: u32,
    n_tokens: u32,
    reserved: u32,
}

/// Gather dense GGUF rows directly on Metal and convert only the selected
/// values to the F32 activation dtype.
pub fn embedding_gather_dense(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &EmbeddingDenseParams,
) -> Result<()> {
    if params.vocab_size == 0 || params.embed_dim == 0 || params.n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_dense: all dimensions must be greater than zero".into(),
        ));
    }
    let kernel = match weight.dtype() {
        DType::BF16 => KERNEL_BF16,
        DType::F16 => KERNEL_F16,
        DType::F32 => KERNEL_F32,
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "embedding_dense: weight must be BF16, F16, or F32, got {other:?}"
            )))
        }
    };
    if token_ids.dtype() != DType::U32 || output.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_dense: expected token/output U32/F32, got {:?}/{:?}",
            token_ids.dtype(),
            output.dtype()
        )));
    }

    let weight_bytes = params
        .vocab_size
        .checked_mul(params.embed_dim)
        .and_then(|elements| elements.checked_mul(weight.dtype().size_of()))
        .ok_or_else(|| MlxError::InvalidArgument("embedding_dense: weight size overflow".into()))?;
    let token_bytes = params
        .n_tokens
        .checked_mul(DType::U32.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("embedding_dense: token size overflow".into()))?;
    let output_bytes = params
        .n_tokens
        .checked_mul(params.embed_dim)
        .and_then(|elements| elements.checked_mul(DType::F32.size_of()))
        .ok_or_else(|| MlxError::InvalidArgument("embedding_dense: output size overflow".into()))?;
    for (name, actual, required) in [
        ("weight", weight.data_byte_len(), weight_bytes),
        ("token_ids", token_ids.data_byte_len(), token_bytes),
        ("output", output.data_byte_len(), output_bytes),
    ] {
        if actual < required {
            return Err(MlxError::InvalidArgument(format!(
                "embedding_dense: {name} buffer needs {required} bytes, got {actual}"
            )));
        }
    }
    let ids = token_ids.as_slice::<u32>()?;
    if let Some((position, id)) = ids
        .iter()
        .take(params.n_tokens)
        .enumerate()
        .find(|(_, id)| **id as usize >= params.vocab_size)
    {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_dense: token_ids[{position}]={id} exceeds vocabulary {}",
            params.vocab_size
        )));
    }

    let gpu_params = GpuEmbeddingDenseParams {
        vocab_size: u32::try_from(params.vocab_size)
            .map_err(|_| MlxError::InvalidArgument("embedding_dense: vocab exceeds u32".into()))?,
        embed_dim: u32::try_from(params.embed_dim).map_err(|_| {
            MlxError::InvalidArgument("embedding_dense: embed_dim exceeds u32".into())
        })?,
        n_tokens: u32::try_from(params.n_tokens).map_err(|_| {
            MlxError::InvalidArgument("embedding_dense: n_tokens exceeds u32".into())
        })?,
        reserved: 0,
    };
    let pipeline = registry.get_pipeline(kernel, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(token_ids)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        MTLSize::new(params.embed_dim as u64, params.n_tokens as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}
