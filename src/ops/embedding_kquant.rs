//! Direct GGML Q5_K and Q6_K embedding lookup.

use metal::foreign_types::ForeignType;
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

const QK_K: usize = 256;
const VALUES_PER_THREAD: usize = 16;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q5_K_KERNEL: &str = "embedding_gather_q5_k_f32";
const Q6_K_KERNEL: &str = "embedding_gather_q6_k_f32";

pub static EMBEDDING_KQUANT_SHADER_SOURCE: &str = include_str!("../shaders/embedding_kquant.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(Q5_K_KERNEL, EMBEDDING_KQUANT_SHADER_SOURCE);
    registry.register_source(Q6_K_KERNEL, EMBEDDING_KQUANT_SHADER_SOURCE);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingQ5KParams {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingQ6KParams {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_tokens: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEmbeddingKQuantParams {
    vocab_size: u32,
    embed_dim: u32,
    blocks_per_row: u32,
    n_tokens: u32,
}

#[derive(Clone, Copy)]
struct LogicalRange {
    buffer_id: usize,
    start: u64,
    end: u64,
}

impl LogicalRange {
    fn new(buffer: &MlxBuffer, operation: &str) -> Result<Self> {
        let logical_bytes = u64::try_from(buffer.data_byte_len()).map_err(|_| {
            MlxError::InvalidArgument(format!("{operation}: logical byte length exceeds u64"))
        })?;
        let end = buffer
            .byte_offset()
            .checked_add(logical_bytes)
            .ok_or_else(|| {
                MlxError::InvalidArgument(format!("{operation}: logical range overflows u64"))
            })?;
        Ok(Self {
            buffer_id: buffer.metal_buffer().as_ptr() as usize,
            start: buffer.byte_offset(),
            end,
        })
    }

    fn overlaps(self, other: Self) -> bool {
        self.buffer_id == other.buffer_id && self.start < other.end && other.start < self.end
    }
}

#[allow(clippy::too_many_arguments)]
fn embedding_gather_kquant(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    vocab_size: usize,
    embed_dim: usize,
    n_tokens: usize,
    block_bytes: usize,
    kernel: &'static str,
    operation: &'static str,
) -> Result<()> {
    if vocab_size == 0 || embed_dim == 0 || n_tokens == 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: all dimensions must be greater than zero"
        )));
    }
    if embed_dim % QK_K != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: embed_dim {embed_dim} must be divisible by {QK_K}"
        )));
    }
    if weight.dtype() != DType::U8
        || token_ids.dtype() != DType::U32
        || output.dtype() != DType::F32
    {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: expected U8/U32/F32 buffers, got {:?}/{:?}/{:?}",
            weight.dtype(),
            token_ids.dtype(),
            output.dtype()
        )));
    }

    let blocks_per_row = embed_dim / QK_K;
    let weight_blocks = vocab_size.checked_mul(blocks_per_row).ok_or_else(|| {
        MlxError::InvalidArgument(format!("{operation}: weight block count overflow"))
    })?;
    if weight_blocks > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: weight block count exceeds u32 shader indexing"
        )));
    }
    let weight_bytes = weight_blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| MlxError::InvalidArgument(format!("{operation}: weight size overflow")))?;
    let token_bytes = n_tokens
        .checked_mul(DType::U32.size_of())
        .ok_or_else(|| MlxError::InvalidArgument(format!("{operation}: token size overflow")))?;
    let output_elements = n_tokens.checked_mul(embed_dim).ok_or_else(|| {
        MlxError::InvalidArgument(format!("{operation}: output element count overflow"))
    })?;
    if output_elements > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: output element count exceeds u32 shader indexing"
        )));
    }
    let output_bytes = output_elements
        .checked_mul(DType::F32.size_of())
        .ok_or_else(|| MlxError::InvalidArgument(format!("{operation}: output size overflow")))?;
    for (name, actual, required) in [
        ("weight", weight.data_byte_len(), weight_bytes),
        ("token_ids", token_ids.data_byte_len(), token_bytes),
        ("output", output.data_byte_len(), output_bytes),
    ] {
        if actual != required {
            return Err(MlxError::InvalidArgument(format!(
                "{operation}: {name} buffer must contain exactly {required} logical bytes, got {actual}"
            )));
        }
    }
    if !output.is_cpu_writable() {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: output buffer must be writable"
        )));
    }

    let output_range = LogicalRange::new(output, operation)?;
    if output_range.overlaps(LogicalRange::new(weight, operation)?)
        || output_range.overlaps(LogicalRange::new(token_ids, operation)?)
    {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: output logical range must not overlap weight or token_ids"
        )));
    }

    let ids = token_ids.as_slice::<u32>()?;
    if ids.len() != n_tokens {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: token_ids logical view has {} ids, expected {n_tokens}",
            ids.len()
        )));
    }
    if let Some((position, id)) = ids
        .iter()
        .enumerate()
        .find(|(_, id)| **id as usize >= vocab_size)
    {
        return Err(MlxError::InvalidArgument(format!(
            "{operation}: token_ids[{position}]={id} exceeds vocabulary {vocab_size}"
        )));
    }

    let gpu_params = GpuEmbeddingKQuantParams {
        vocab_size: u32::try_from(vocab_size).map_err(|_| {
            MlxError::InvalidArgument(format!("{operation}: vocab_size exceeds u32"))
        })?,
        embed_dim: u32::try_from(embed_dim).map_err(|_| {
            MlxError::InvalidArgument(format!("{operation}: embed_dim exceeds u32"))
        })?,
        blocks_per_row: u32::try_from(blocks_per_row).map_err(|_| {
            MlxError::InvalidArgument(format!("{operation}: row block count exceeds u32"))
        })?,
        n_tokens: u32::try_from(n_tokens)
            .map_err(|_| MlxError::InvalidArgument(format!("{operation}: n_tokens exceeds u32")))?,
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
        MTLSize::new((embed_dim / VALUES_PER_THREAD) as u64, n_tokens as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}

/// Gather and dequantize exact GGML Q5_K embedding rows on Metal.
#[allow(clippy::too_many_arguments)]
pub fn embedding_gather_q5_k(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &EmbeddingQ5KParams,
) -> Result<()> {
    embedding_gather_kquant(
        encoder,
        registry,
        device,
        weight,
        token_ids,
        output,
        params.vocab_size,
        params.embed_dim,
        params.n_tokens,
        Q5_K_BLOCK_BYTES,
        Q5_K_KERNEL,
        "embedding_q5_k",
    )
}

/// Gather and dequantize exact GGML Q6_K embedding rows on Metal.
#[allow(clippy::too_many_arguments)]
pub fn embedding_gather_q6_k(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &EmbeddingQ6KParams,
) -> Result<()> {
    embedding_gather_kquant(
        encoder,
        registry,
        device,
        weight,
        token_ids,
        output,
        params.vocab_size,
        params.embed_dim,
        params.n_tokens,
        Q6_K_BLOCK_BYTES,
        Q6_K_KERNEL,
        "embedding_q6_k",
    )
}
