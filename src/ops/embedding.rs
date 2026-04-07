//! GPU-accelerated quantized embedding table lookup.
//!
//! Supports 4-bit and 6-bit quantized embedding tables, performing
//! on-the-fly dequantization during gather.  The dequantization formula
//! is `float_val = uint_val * scale + bias` with bf16 scales and biases.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// Parameters for quantized embedding gather.
pub struct EmbeddingGatherParams {
    /// Embedding dimension (number of float values per token).
    pub embed_dim: usize,
    /// Number of elements per quantization group (typically 64).
    pub group_size: usize,
    /// Quantization bit width: 4 or 6.
    pub bits: u8,
    /// Number of tokens to gather.
    pub n_tokens: usize,
}

/// MSL-compatible parameter struct for the embedding kernel.
///
/// Must match the `EmbeddingParams` struct in `embedding.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEmbeddingParams {
    embed_dim: u32,
    group_size: u32,
    packed_row_stride: u32,
    n_groups_per_row: u32,
}

/// Encode a quantized embedding gather operation into the command buffer.
///
/// Looks up `n_tokens` rows from a quantized embedding table, dequantizing
/// each row on-the-fly on the GPU.
///
/// # Buffer expectations
///
/// * `weight_packed` — Packed quantized embedding table.
///   - 4-bit: `[vocab_size, embed_dim / 8]` uint32 values (8 values per uint32).
///   - 6-bit: `[vocab_size, embed_dim * 3 / 4]` uint8 bytes (4 values per 3 bytes).
/// * `scales` — bf16 scales, `[vocab_size, n_groups_per_row]`.
/// * `biases` — bf16 biases, `[vocab_size, n_groups_per_row]`.
/// * `token_ids` — uint32 token IDs, `[n_tokens]`.
/// * `output` — f32 output buffer, `[n_tokens, embed_dim]`.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// * `bits` is not 4 or 6
/// * `embed_dim` is zero
/// * `group_size` is zero
/// * `embed_dim` is not divisible by `group_size`
/// * `n_tokens` is zero
/// * Output buffer is too small
#[allow(clippy::too_many_arguments)]
pub fn embedding_gather(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    weight_packed: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    token_ids: &MlxBuffer,
    output: &MlxBuffer,
    params: &EmbeddingGatherParams,
) -> Result<()> {
    // --- Validation ---
    if params.bits != 4 && params.bits != 6 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_gather: bits must be 4 or 6, got {}",
            params.bits
        )));
    }
    if params.embed_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_gather: embed_dim must be > 0".into(),
        ));
    }
    if params.group_size == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_gather: group_size must be > 0".into(),
        ));
    }
    if params.embed_dim % params.group_size != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_gather: embed_dim ({}) must be divisible by group_size ({})",
            params.embed_dim, params.group_size
        )));
    }
    if params.n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "embedding_gather: n_tokens must be > 0".into(),
        ));
    }

    let expected_output_bytes = params.n_tokens * params.embed_dim * std::mem::size_of::<f32>();
    if output.byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "embedding_gather: output buffer too small: need {} bytes, have {}",
            expected_output_bytes,
            output.byte_len()
        )));
    }

    // --- Compute layout parameters ---
    let n_groups_per_row = params.embed_dim / params.group_size;

    let packed_row_stride: u32 = match params.bits {
        4 => {
            // 8 values per uint32; stride in uint32 count
            (params.embed_dim / 8) as u32
        }
        6 => {
            // 4 values per 3 bytes; stride in bytes
            (params.embed_dim * 3 / 4) as u32
        }
        _ => unreachable!(), // validated above
    };

    let gpu_params = GpuEmbeddingParams {
        embed_dim: params.embed_dim as u32,
        group_size: params.group_size as u32,
        packed_row_stride,
        n_groups_per_row: n_groups_per_row as u32,
    };

    // --- Select kernel ---
    let kernel_name = match params.bits {
        4 => "embedding_gather_4bit",
        6 => "embedding_gather_6bit",
        _ => unreachable!(),
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // --- Encode dispatch ---
    let grid = MTLSize::new(params.embed_dim as u64, params.n_tokens as u64, 1);
    let tg_size = MTLSize::new(
        std::cmp::min(256, params.embed_dim as u64),
        1,
        1,
    );

    let params_bytes = as_bytes(&gpu_params);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(weight_packed)),
            (1, KernelArg::Buffer(scales)),
            (2, KernelArg::Buffer(biases)),
            (3, KernelArg::Buffer(token_ids)),
            (4, KernelArg::Buffer(output)),
            (5, KernelArg::Bytes(params_bytes)),
        ],
        grid,
        tg_size,
    );

    Ok(())
}
