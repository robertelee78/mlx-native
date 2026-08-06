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
pub const DEEPSEEK_INDEXER_SCORE_MMA_KERNEL: &str = "deepseek_indexer_score_mma_bf16";
pub const DEEPSEEK_INDEXER_TOPK_BLOCK_KERNEL: &str = "deepseek_indexer_topk_block_i32";
pub const DEEPSEEK_INDEXER_TOPK_MERGE_KERNEL: &str = "deepseek_indexer_topk_merge_i32";
pub const DEEPSEEK_INDEXER_TOPK_FINALIZE_KERNEL: &str = "deepseek_indexer_topk_finalize_i32";
const SCORE_THREADS: u64 = 256;
const SCORE_KEYS_PER_GROUP: usize = 64;
const SCORE_QUERIES_PER_GROUP: usize = 8;
const TOPK_BLOCK_THREADS: usize = 1024;
const TOPK_MERGE_THREADS: u64 = 256;

pub static DEEPSEEK_INDEXER_SHADER_SOURCE: &str = include_str!("../shaders/deepseek_indexer.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_INDEXER_SCORE_KERNEL,
        DEEPSEEK_INDEXER_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_INDEXER_SCORE_MMA_KERNEL,
        DEEPSEEK_INDEXER_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_INDEXER_TOPK_BLOCK_KERNEL,
        DEEPSEEK_INDEXER_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_INDEXER_TOPK_MERGE_KERNEL,
        DEEPSEEK_INDEXER_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_INDEXER_TOPK_FINALIZE_KERNEL,
        DEEPSEEK_INDEXER_SHADER_SOURCE,
    );
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

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepSeekIndexerTopKPlan {
    block_threads: u32,
    block_count: u32,
    list_count: u32,
    scratch_row_stride: u32,
}

/// I32 elements required by the optimized top-k ping-pong workspace.
pub fn deepseek_indexer_topk_scratch_elements(
    batch: usize,
    queries: usize,
    kv_len: usize,
) -> Result<usize> {
    let block_count = kv_len.div_ceil(TOPK_BLOCK_THREADS);
    checked_shape(&[batch, queries, 2, block_count, DEEPSEEK_INDEXER_TOP_K])
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
    let (batch, queries, kv_len) = validate_params(params)?;
    let topk_elements = deepseek_indexer_topk_scratch_elements(batch, queries, kv_len)?;
    let topk_scratch = device.alloc_buffer(
        topk_elements * DType::I32.size_of(),
        DType::I32,
        vec![batch, queries, topk_elements / (batch * queries)],
    )?;
    dispatch_deepseek_indexer_into(
        encoder,
        registry,
        device,
        q,
        kv,
        weights,
        scratch,
        &topk_scratch,
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
    topk_scratch: &MlxBuffer,
    output: &MlxBuffer,
    output_row_stride: usize,
    output_column_offset: usize,
    params: &DeepSeekIndexerParams,
) -> Result<()> {
    dispatch_deepseek_indexer_into_with_score_kernel(
        encoder,
        registry,
        device,
        q,
        kv,
        weights,
        scratch,
        topk_scratch,
        output,
        output_row_stride,
        output_column_offset,
        params,
        DEEPSEEK_INDEXER_SCORE_KERNEL,
    )
}

/// Encode the indexer with the same half-staged simdgroup-MMA arithmetic
/// used by llama.cpp's Metal lightning-indexer kernel.
///
/// The ordinary [`dispatch_deepseek_indexer_into`] entry point remains the
/// BF16-to-F32 reference path. Model runtimes that need production throughput
/// can opt into this explicitly without consulting process-global state.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_indexer_into_mma(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    weights: &MlxBuffer,
    scratch: &MlxBuffer,
    topk_scratch: &MlxBuffer,
    output: &MlxBuffer,
    output_row_stride: usize,
    output_column_offset: usize,
    params: &DeepSeekIndexerParams,
) -> Result<()> {
    dispatch_deepseek_indexer_into_with_score_kernel(
        encoder,
        registry,
        device,
        q,
        kv,
        weights,
        scratch,
        topk_scratch,
        output,
        output_row_stride,
        output_column_offset,
        params,
        DEEPSEEK_INDEXER_SCORE_MMA_KERNEL,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_deepseek_indexer_into_with_score_kernel(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    weights: &MlxBuffer,
    scratch: &MlxBuffer,
    topk_scratch: &MlxBuffer,
    output: &MlxBuffer,
    output_row_stride: usize,
    output_column_offset: usize,
    params: &DeepSeekIndexerParams,
    score_kernel: &str,
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
    let block_count = kv_len.div_ceil(TOPK_BLOCK_THREADS);
    let scratch_row_stride = block_count
        .checked_mul(DEEPSEEK_INDEXER_TOP_K)
        .ok_or_else(|| {
            MlxError::InvalidArgument("deepseek_indexer: top-k row stride overflows".into())
        })?;
    validate_buffer(
        topk_scratch,
        "topk_scratch",
        DType::I32,
        &[batch, queries, 2 * scratch_row_stride],
    )?;
    validate_buffer(
        output,
        "output",
        DType::I32,
        &[batch, queries, output_row_stride],
    )?;

    let score_pipeline = registry.get_pipeline(score_kernel, device.metal_device())?;
    encoder.set_op_kind(CapturedOpKind::Other);
    let score_params = DeepSeekIndexerParams {
        batch: 1,
        ..*params
    };
    for batch_index in 0..batch {
        let q_view = q
            .slice_view(
                (batch_index
                    * queries
                    * DEEPSEEK_INDEXER_HEADS
                    * DEEPSEEK_INDEXER_HEAD_DIM
                    * DType::BF16.size_of()) as u64,
                queries * DEEPSEEK_INDEXER_HEADS * DEEPSEEK_INDEXER_HEAD_DIM,
            )
            .with_shape(vec![
                1,
                queries,
                DEEPSEEK_INDEXER_HEADS,
                DEEPSEEK_INDEXER_HEAD_DIM,
            ])?;
        let kv_view = kv
            .slice_view(
                (batch_index * kv_len * DEEPSEEK_INDEXER_HEAD_DIM * DType::BF16.size_of()) as u64,
                kv_len * DEEPSEEK_INDEXER_HEAD_DIM,
            )
            .with_shape(vec![1, kv_len, DEEPSEEK_INDEXER_HEAD_DIM])?;
        let weights_view = weights
            .slice_view(
                (batch_index * queries * DEEPSEEK_INDEXER_HEADS * DType::F32.size_of()) as u64,
                queries * DEEPSEEK_INDEXER_HEADS,
            )
            .with_shape(vec![1, queries, DEEPSEEK_INDEXER_HEADS])?;
        let scratch_view = scratch
            .slice_view(
                (batch_index * queries * kv_len * DType::F32.size_of()) as u64,
                queries * kv_len,
            )
            .with_shape(vec![1, queries, kv_len])?;
        encoder.encode_threadgroups_with_args(
            score_pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&score_params))),
                (1, KernelArg::Buffer(&q_view)),
                (2, KernelArg::Buffer(&kv_view)),
                (3, KernelArg::Buffer(&weights_view)),
                (4, KernelArg::Buffer(&scratch_view)),
            ],
            MTLSize::new(
                kv_len.div_ceil(SCORE_KEYS_PER_GROUP) as u64,
                queries.div_ceil(SCORE_QUERIES_PER_GROUP) as u64,
                1,
            ),
            MTLSize::new(SCORE_THREADS, 1, 1),
        );
    }
    encoder.memory_barrier();

    let plan = DeepSeekIndexerTopKPlan {
        block_threads: TOPK_BLOCK_THREADS as u32,
        block_count: block_count as u32,
        list_count: block_count as u32,
        scratch_row_stride: scratch_row_stride as u32,
    };
    let bank_elements = batch * queries * scratch_row_stride;
    let bank_a = topk_scratch
        .slice_view(0, bank_elements)
        .with_shape(vec![bank_elements])?;
    let bank_b = topk_scratch
        .slice_view((bank_elements * DType::I32.size_of()) as u64, bank_elements)
        .with_shape(vec![bank_elements])?;
    let block_pipeline =
        registry.get_pipeline(DEEPSEEK_INDEXER_TOPK_BLOCK_KERNEL, device.metal_device())?;
    if block_pipeline.max_total_threads_per_threadgroup() < TOPK_BLOCK_THREADS as u64 {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_indexer: Metal pipeline supports only {} top-k threads; {TOPK_BLOCK_THREADS} required",
            block_pipeline.max_total_threads_per_threadgroup()
        )));
    }
    encoder.encode_threadgroups_with_args_and_shared(
        block_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Bytes(as_bytes(&plan))),
            (2, KernelArg::Buffer(scratch)),
            (3, KernelArg::Buffer(&bank_a)),
        ],
        &[(0, (TOPK_BLOCK_THREADS * DType::I32.size_of()) as u64)],
        MTLSize::new((batch * queries * block_count) as u64, 1, 1),
        MTLSize::new(TOPK_BLOCK_THREADS as u64, 1, 1),
    );
    encoder.memory_barrier();

    let merge_pipeline =
        registry.get_pipeline(DEEPSEEK_INDEXER_TOPK_MERGE_KERNEL, device.metal_device())?;
    let mut list_count = block_count;
    let mut input = &bank_a;
    let mut merge_output = &bank_b;
    while list_count > 1 {
        let merge_plan = DeepSeekIndexerTopKPlan {
            list_count: list_count as u32,
            ..plan
        };
        let merged_lists = list_count.div_ceil(2);
        encoder.encode_threadgroups_with_args(
            merge_pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(params))),
                (1, KernelArg::Bytes(as_bytes(&merge_plan))),
                (2, KernelArg::Buffer(scratch)),
                (3, KernelArg::Buffer(input)),
                (4, KernelArg::Buffer(merge_output)),
            ],
            MTLSize::new((batch * queries * merged_lists) as u64, 1, 1),
            MTLSize::new(TOPK_MERGE_THREADS, 1, 1),
        );
        encoder.memory_barrier();
        std::mem::swap(&mut input, &mut merge_output);
        list_count = merged_lists;
    }

    let final_plan = DeepSeekIndexerTopKPlan {
        list_count: 1,
        ..plan
    };
    let finalize_pipeline =
        registry.get_pipeline(DEEPSEEK_INDEXER_TOPK_FINALIZE_KERNEL, device.metal_device())?;
    encoder.encode_with_args(
        finalize_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Bytes(as_bytes(&final_plan))),
            (2, KernelArg::Bytes(as_bytes(&output_layout))),
            (3, KernelArg::Buffer(input)),
            (4, KernelArg::Buffer(output)),
        ],
        MTLSize::new(DEEPSEEK_INDEXER_TOP_K as u64, (batch * queries) as u64, 1),
        MTLSize::new(256, 1, 1),
    );
    Ok(())
}
