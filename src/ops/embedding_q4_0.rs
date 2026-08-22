//! Direct GGML Q4_0 embedding lookup.

use metal::foreign_types::ForeignType;
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

const QK4_0: usize = 32;
const BLOCK_BYTES: usize = 18;
const VALUES_PER_THREAD: usize = 16;
const KERNEL: &str = "embedding_gather_q4_0_f32";

pub static EMBEDDING_Q4_0_SHADER_SOURCE: &str = include_str!("../shaders/embedding_q4_0.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(KERNEL, EMBEDDING_Q4_0_SHADER_SOURCE);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingQ4_0Params {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_tokens: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEmbeddingQ4_0Params {
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
    fn new(buffer: &MlxBuffer) -> Result<Self> {
        let logical_bytes = u64::try_from(buffer.data_byte_len()).map_err(|_| {
            MlxError::InvalidArgument("embedding_q4_0: logical byte length exceeds u64".into())
        })?;
        let end = buffer
            .byte_offset()
            .checked_add(logical_bytes)
            .ok_or_else(|| {
                MlxError::InvalidArgument("embedding_q4_0: logical range overflows u64".into())
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

/// Gather and dequantize Q4_0 embedding rows directly on Metal.
///
/// Every logical buffer byte and token ID is checked before pipeline lookup
/// or command encoding. The kernel repeats the vocabulary check as defense in
/// depth, so malformed IDs cannot produce an out-of-bounds weight read.
#[allow(clippy::too_many_arguments)]
pub fn embedding_gather_q4_0(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &EmbeddingQ4_0Params,
) -> Result<()> {
    if params.vocab_size == 0 || params.embed_dim == 0 || params.n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_q4_0: all dimensions must be greater than zero".into(),
        ));
    }
    if params.embed_dim % QK4_0 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_q4_0: embed_dim {} must be divisible by {QK4_0}",
            params.embed_dim
        )));
    }
    if weight.dtype() != DType::U8
        || token_ids.dtype() != DType::U32
        || output.dtype() != DType::F32
    {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_q4_0: expected U8/U32/F32 buffers, got {:?}/{:?}/{:?}",
            weight.dtype(),
            token_ids.dtype(),
            output.dtype()
        )));
    }

    let blocks_per_row = params.embed_dim / QK4_0;
    let weight_blocks = params
        .vocab_size
        .checked_mul(blocks_per_row)
        .ok_or_else(|| {
            MlxError::InvalidArgument("embedding_q4_0: weight block count overflow".into())
        })?;
    if weight_blocks > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "embedding_q4_0: weight block count exceeds u32 shader indexing".into(),
        ));
    }
    let weight_bytes = weight_blocks
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| MlxError::InvalidArgument("embedding_q4_0: weight size overflow".into()))?;
    let token_bytes = params
        .n_tokens
        .checked_mul(DType::U32.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("embedding_q4_0: token size overflow".into()))?;
    let output_elements = params
        .n_tokens
        .checked_mul(params.embed_dim)
        .ok_or_else(|| {
            MlxError::InvalidArgument("embedding_q4_0: output element count overflow".into())
        })?;
    if output_elements > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "embedding_q4_0: output element count exceeds u32 shader indexing".into(),
        ));
    }
    let output_bytes = output_elements
        .checked_mul(DType::F32.size_of())
        .ok_or_else(|| MlxError::InvalidArgument("embedding_q4_0: output size overflow".into()))?;
    for (name, actual, required) in [
        ("weight", weight.data_byte_len(), weight_bytes),
        ("token_ids", token_ids.data_byte_len(), token_bytes),
        ("output", output.data_byte_len(), output_bytes),
    ] {
        if actual != required {
            return Err(MlxError::InvalidArgument(format!(
                "embedding_q4_0: {name} buffer must contain exactly {required} logical bytes, got {actual}"
            )));
        }
    }
    if !output.is_cpu_writable() {
        return Err(MlxError::InvalidArgument(
            "embedding_q4_0: output buffer must be writable".into(),
        ));
    }

    let output_range = LogicalRange::new(output)?;
    if output_range.overlaps(LogicalRange::new(weight)?)
        || output_range.overlaps(LogicalRange::new(token_ids)?)
    {
        return Err(MlxError::InvalidArgument(
            "embedding_q4_0: output logical range must not overlap weight or token_ids".into(),
        ));
    }

    let ids = token_ids.as_slice::<u32>()?;
    if ids.len() != params.n_tokens {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_q4_0: token_ids logical view has {} ids, expected {}",
            ids.len(),
            params.n_tokens
        )));
    }
    if let Some((position, id)) = ids
        .iter()
        .enumerate()
        .find(|(_, id)| **id as usize >= params.vocab_size)
    {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_q4_0: token_ids[{position}]={id} exceeds vocabulary {}",
            params.vocab_size
        )));
    }

    let gpu_params = GpuEmbeddingQ4_0Params {
        vocab_size: u32::try_from(params.vocab_size).map_err(|_| {
            MlxError::InvalidArgument("embedding_q4_0: vocab_size exceeds u32".into())
        })?,
        embed_dim: u32::try_from(params.embed_dim).map_err(|_| {
            MlxError::InvalidArgument("embedding_q4_0: embed_dim exceeds u32".into())
        })?,
        blocks_per_row: u32::try_from(blocks_per_row).map_err(|_| {
            MlxError::InvalidArgument("embedding_q4_0: row block count exceeds u32".into())
        })?,
        n_tokens: u32::try_from(params.n_tokens).map_err(|_| {
            MlxError::InvalidArgument("embedding_q4_0: n_tokens exceeds u32".into())
        })?,
    };
    let pipeline = registry.get_pipeline(KERNEL, device.metal_device())?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(token_ids)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        MTLSize::new(
            (params.embed_dim / VALUES_PER_THREAD) as u64,
            params.n_tokens as u64,
            1,
        ),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}
