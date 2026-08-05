//! Owned DeepSeek-V4 0731 lightning indexer and causal top-512 selection.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub const DEEPSEEK_INDEXER_HEADS: usize = 64;
pub const DEEPSEEK_INDEXER_HEAD_DIM: usize = 128;
pub const DEEPSEEK_INDEXER_TOP_K: usize = 512;
pub const DEEPSEEK_INDEXER_RATIO: usize = 4;
pub const DEEPSEEK_INDEXER_SCORE_KERNEL: &str = "deepseek_indexer_score_bf16";
pub const DEEPSEEK_INDEXER_TOPK_KERNEL: &str = "deepseek_indexer_topk_i32";
const THREADS: u64 = 256;

pub static DEEPSEEK_INDEXER_SHADER_SOURCE: &str = include_str!("../shaders/deepseek_indexer.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_INDEXER_SCORE_KERNEL,
        DEEPSEEK_INDEXER_SHADER_SOURCE,
    );
    registry.register_source(DEEPSEEK_INDEXER_TOPK_KERNEL, DEEPSEEK_INDEXER_SHADER_SOURCE);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeepSeekIndexerParams {
    pub batch: u32,
    pub query_len: u32,
    pub kv_len: u32,
    pub start_pos: u32,
    pub ratio: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub offset: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepSeekIndexerOutputLayout {
    row_stride: u32,
    column_offset: u32,
}

fn checked_shape(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "deepseek_indexer: shape product overflows: {dims:?}"
            ))
        })
    })
}

fn validate_buffer(buf: &MlxBuffer, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let elements = checked_shape(shape)?;
    if buf.dtype() != dtype {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_indexer: {name} must be {dtype}, got {}",
            buf.dtype()
        )));
    }
    if buf.shape() != shape || buf.byte_len() < elements * dtype.size_of() {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_indexer: {name} shape must be {shape:?}, got {:?}",
            buf.shape()
        )));
    }
    Ok(())
}

fn validate_params(p: &DeepSeekIndexerParams) -> Result<(usize, usize, usize)> {
    if p.batch == 0 || p.query_len == 0 || p.kv_len == 0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_indexer: batch, query_len, and kv_len must be nonzero".into(),
        ));
    }
    if p.start_pos != 0 && p.query_len != 1 {
        return Err(MlxError::InvalidArgument(
            "deepseek_indexer: incremental calls require query_len=1".into(),
        ));
    }
    if p.start_pos.checked_add(p.query_len).is_none() {
        return Err(MlxError::InvalidArgument(
            "deepseek_indexer: start_pos + query_len overflows".into(),
        ));
    }
    if p.ratio as usize != DEEPSEEK_INDEXER_RATIO
        || p.heads as usize != DEEPSEEK_INDEXER_HEADS
        || p.head_dim as usize != DEEPSEEK_INDEXER_HEAD_DIM
        || p.top_k as usize != DEEPSEEK_INDEXER_TOP_K
    {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_indexer: production shape is ratio=4, heads=64, head_dim=128, top_k=512; got {}, {}, {}, {}",
            p.ratio, p.heads, p.head_dim, p.top_k
        )));
    }
    if p.offset < 0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_indexer: offset must be nonnegative; -1 is reserved for the sentinel".into(),
        ));
    }
    let valid = (p.start_pos + p.query_len) as usize / DEEPSEEK_INDEXER_RATIO;
    if valid > p.kv_len as usize {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_indexer: kv_len {} is shorter than {valid} causal compressed entries",
            p.kv_len
        )));
    }
    if valid > 0 && (p.offset as i64 + valid as i64 - 1) > i32::MAX as i64 {
        return Err(MlxError::InvalidArgument(
            "deepseek_indexer: offset plus compressed index exceeds i32".into(),
        ));
    }
    let groups = checked_shape(&[p.batch as usize, p.query_len as usize, p.kv_len as usize])?;
    if groups > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "deepseek_indexer: dispatch grid exceeds Metal uint indexing".into(),
        ));
    }
    Ok((p.batch as usize, p.query_len as usize, p.kv_len as usize))
}

/// Score and select compressed KV entries for prefill or one-token decode.
///
/// Computes `sum_heads(relu(dot(q, kv)) * weights)`. Prefill/decode causal
/// limits are derived from absolute query position and ratio four. `scratch`
/// is overwritten during top-k selection. Output is always
/// `[batch, query_len, 512]`; unavailable/invalid entries are `-1`, while
/// selected entries include `offset`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_indexer(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    weights: &MlxBuffer,
    scratch: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekIndexerParams,
) -> Result<()> {
    dispatch_deepseek_indexer_into(
        encoder,
        registry,
        device,
        q,
        kv,
        weights,
        scratch,
        output,
        DEEPSEEK_INDEXER_TOP_K,
        0,
        params,
    )
}

/// Score and select compressed KV entries into a strided output row.
///
/// This is the batched-prefill sibling of [`dispatch_deepseek_indexer`]. It
/// writes the 512 selected indices at `column_offset` inside each
/// `row_stride`-wide `[batch, query]` row, leaving every other column
/// untouched. The contiguous entry point above remains source-compatible.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_indexer_into(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    weights: &MlxBuffer,
    scratch: &MlxBuffer,
    output: &MlxBuffer,
    output_row_stride: usize,
    output_column_offset: usize,
    params: &DeepSeekIndexerParams,
) -> Result<()> {
    let (batch, queries, kv_len) = validate_params(params)?;
    let tail_end = output_column_offset
        .checked_add(DEEPSEEK_INDEXER_TOP_K)
        .ok_or_else(|| {
            MlxError::InvalidArgument(
                "deepseek_indexer: output column offset plus top-k overflows".into(),
            )
        })?;
    if tail_end > output_row_stride {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_indexer: output tail {output_column_offset}..{tail_end} exceeds row stride {output_row_stride}"
        )));
    }
    let output_layout = DeepSeekIndexerOutputLayout {
        row_stride: u32::try_from(output_row_stride).map_err(|_| {
            MlxError::InvalidArgument(
                "deepseek_indexer: output row stride exceeds Metal uint indexing".into(),
            )
        })?,
        column_offset: u32::try_from(output_column_offset).map_err(|_| {
            MlxError::InvalidArgument(
                "deepseek_indexer: output column offset exceeds Metal uint indexing".into(),
            )
        })?,
    };
    validate_buffer(
        q,
        "q",
        DType::BF16,
        &[
            batch,
            queries,
            DEEPSEEK_INDEXER_HEADS,
            DEEPSEEK_INDEXER_HEAD_DIM,
        ],
    )?;
    validate_buffer(
        kv,
        "kv",
        DType::BF16,
        &[batch, kv_len, DEEPSEEK_INDEXER_HEAD_DIM],
    )?;
    validate_buffer(
        weights,
        "weights",
        DType::F32,
        &[batch, queries, DEEPSEEK_INDEXER_HEADS],
    )?;
    validate_buffer(scratch, "scratch", DType::F32, &[batch, queries, kv_len])?;
    validate_buffer(
        output,
        "output",
        DType::I32,
        &[batch, queries, output_row_stride],
    )?;

    let score_pipeline =
        registry.get_pipeline(DEEPSEEK_INDEXER_SCORE_KERNEL, device.metal_device())?;
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_threadgroups_with_args(
        score_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(kv)),
            (3, KernelArg::Buffer(weights)),
            (4, KernelArg::Buffer(scratch)),
        ],
        MTLSize::new((batch * queries * kv_len) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    // The command encoder uses concurrent dispatch. Top-k consumes scores
    // written above, so this dependency must be explicit.
    encoder.memory_barrier();
    let topk_pipeline =
        registry.get_pipeline(DEEPSEEK_INDEXER_TOPK_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        topk_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(scratch)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&output_layout))),
        ],
        MTLSize::new((batch * queries) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}
