//! DeepSeek-V4 sparse attention with a shared KV stream and learned sinks.
//!
//! This implements the official 0731 inference boundary directly in Metal.
//! Selected KV entries act as both keys and values. The per-head sink logit
//! participates in softmax normalization but contributes no value vector.
//! Forward/inverse RoPE remain outside this primitive.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::flash_attn_prefill::FlashAttnPrefillParams;
use crate::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_bf16_d512_heads_as_rows_with_sinks;

pub const DEEPSEEK_SPARSE_HEADS: usize = 64;
pub const DEEPSEEK_SPARSE_HEAD_DIM: usize = 512;
pub const DEEPSEEK_INDEX_TOP_K: usize = 512;
pub const DEEPSEEK_SPARSE_ATTENTION_SCORE_KERNEL: &str = "deepseek_sparse_attention_score_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_REDUCE_KERNEL: &str = "deepseek_sparse_attention_reduce_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_VALIDATE_Q_KERNEL: &str =
    "deepseek_sparse_attention_validate_q_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_GATHER_KERNEL: &str = "deepseek_sparse_attention_gather_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_SANITIZE_KERNEL: &str =
    "deepseek_sparse_attention_sanitize_bf16";
const THREADS: u64 = 256;
const SLOTS_PER_GROUP: usize = 32;

pub static DEEPSEEK_SPARSE_ATTENTION_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_sparse_attention.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_SPARSE_ATTENTION_SCORE_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_SPARSE_ATTENTION_REDUCE_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_SPARSE_ATTENTION_VALIDATE_Q_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_SPARSE_ATTENTION_GATHER_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_SPARSE_ATTENTION_SANITIZE_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_SHADER_SOURCE,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeepSeekSparseAttentionParams {
    pub batch: u32,
    pub query_len: u32,
    pub kv_len: u32,
    pub top_k: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub scale: f32,
}

fn checked_shape(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "deepseek_sparse_attention: shape product overflows: {dims:?}"
            ))
        })
    })
}

fn validate_buffer(buf: &MlxBuffer, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let elements = checked_shape(shape)?;
    if buf.dtype() != dtype {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_sparse_attention: {name} must be {dtype}, got {}",
            buf.dtype()
        )));
    }
    if buf.shape() != shape || buf.byte_len() < elements * dtype.size_of() {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_sparse_attention: {name} shape must be {shape:?}, got {:?}",
            buf.shape()
        )));
    }
    Ok(())
}

fn validate_params(params: &DeepSeekSparseAttentionParams) -> Result<(usize, usize, usize)> {
    if params.batch == 0 || params.query_len == 0 || params.kv_len == 0 || params.top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_sparse_attention: batch, query_len, kv_len, and top_k must be nonzero".into(),
        ));
    }
    if params.heads as usize != DEEPSEEK_SPARSE_HEADS
        || params.head_dim as usize != DEEPSEEK_SPARSE_HEAD_DIM
    {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_sparse_attention: production shape is heads={}, head_dim={}, got {}, {}",
            DEEPSEEK_SPARSE_HEADS, DEEPSEEK_SPARSE_HEAD_DIM, params.heads, params.head_dim
        )));
    }
    if !params.scale.is_finite() || params.scale <= 0.0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_sparse_attention: scale must be finite and positive".into(),
        ));
    }
    let batch = params.batch as usize;
    let queries = params.query_len as usize;
    let kv = params.kv_len as usize;
    let groups = checked_shape(&[batch, queries, DEEPSEEK_SPARSE_HEADS])?;
    if groups > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "deepseek_sparse_attention: dispatch grid exceeds Metal uint indexing".into(),
        ));
    }
    Ok((batch, queries, kv))
}

/// F32 elements required by the selected-logit workspace.
pub fn deepseek_sparse_attention_scratch_elements(
    batch: usize,
    queries: usize,
    top_k: usize,
) -> Result<usize> {
    checked_shape(&[batch, queries, DEEPSEEK_SPARSE_HEADS, top_k])
}

/// Encode sparse attention for prefill or one-token decode.
///
/// Layouts are Q/output `[batch, query, 64, 512]`, shared KV
/// `[batch, kv, 512]`, sinks `[64]`, and indices `[batch, query, top_k]`.
/// An index of `-1` is skipped; any other out-of-range index fails the entire
/// affected output head closed to zero. Duplicate valid indices retain their
/// official multiplicity.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_sparse_attention(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekSparseAttentionParams,
) -> Result<()> {
    let (batch, queries, _) = validate_params(params)?;
    let top_k = params.top_k as usize;
    let scratch_elements = deepseek_sparse_attention_scratch_elements(batch, queries, top_k)?;
    let scratch = device.alloc_buffer(
        scratch_elements * DType::F32.size_of(),
        DType::F32,
        vec![batch, queries, DEEPSEEK_SPARSE_HEADS, top_k],
    )?;
    dispatch_deepseek_sparse_attention_with_scratch(
        encoder, registry, device, q, kv, sinks, indices, &scratch, output, params,
    )
}

/// Encode sparse attention using a caller-owned F32 selected-logit workspace.
/// Decode callers should reuse this allocation across steady-state tokens.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_sparse_attention_with_scratch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    scratch: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekSparseAttentionParams,
) -> Result<()> {
    let (batch, queries, kv_len) = validate_params(params)?;
    validate_buffer(
        q,
        "q",
        DType::BF16,
        &[
            batch,
            queries,
            DEEPSEEK_SPARSE_HEADS,
            DEEPSEEK_SPARSE_HEAD_DIM,
        ],
    )?;
    validate_buffer(
        kv,
        "kv",
        DType::BF16,
        &[batch, kv_len, DEEPSEEK_SPARSE_HEAD_DIM],
    )?;
    validate_buffer(sinks, "sinks", DType::F32, &[DEEPSEEK_SPARSE_HEADS])?;
    validate_buffer(
        indices,
        "indices",
        DType::I32,
        &[batch, queries, params.top_k as usize],
    )?;
    validate_buffer(
        scratch,
        "scratch",
        DType::F32,
        &[batch, queries, DEEPSEEK_SPARSE_HEADS, params.top_k as usize],
    )?;
    validate_buffer(
        output,
        "output",
        DType::BF16,
        &[
            batch,
            queries,
            DEEPSEEK_SPARSE_HEADS,
            DEEPSEEK_SPARSE_HEAD_DIM,
        ],
    )?;

    let score_pipeline = registry.get_pipeline(
        DEEPSEEK_SPARSE_ATTENTION_SCORE_KERNEL,
        device.metal_device(),
    )?;
    let head_groups = batch * queries * DEEPSEEK_SPARSE_HEADS;
    let blocks_per_head = (params.top_k as usize).div_ceil(SLOTS_PER_GROUP);
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args(
        score_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(kv)),
            (3, KernelArg::Buffer(indices)),
            (4, KernelArg::Buffer(scratch)),
        ],
        MTLSize::new((head_groups * blocks_per_head) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    encoder.memory_barrier();
    let reduce_pipeline = registry.get_pipeline(
        DEEPSEEK_SPARSE_ATTENTION_REDUCE_KERNEL,
        device.metal_device(),
    )?;
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args(
        reduce_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(kv)),
            (2, KernelArg::Buffer(sinks)),
            (3, KernelArg::Buffer(indices)),
            (4, KernelArg::Buffer(scratch)),
            (5, KernelArg::Buffer(output)),
        ],
        MTLSize::new(head_groups as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}

/// Gather selected shared-KV rows and run the llama.cpp-derived D=512 Flash
/// Attention kernel with each sparse query presented as an independent flash
/// batch. The selection is shared by all 64 query heads, so gathering once
/// lets the tiled kernel reuse each selected KV row across heads.
///
/// This is the scalable sparse-prefill path: work after selection is
/// `O(query_len * top_k)` rather than scanning the full compressed history.
/// `invalid_global` (`[B,Q]`) and `invalid_heads` (`[B,Q,64]`) must be U32
/// buffers zeroed by the caller before encoding. Invalid indices or non-finite
/// KV values zero every output head for that query; a non-finite query/sink
/// zeros only the affected head.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_sparse_attention_flash_prefill(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    gathered_kv: &MlxBuffer,
    mask: &MlxBuffer,
    invalid_global: &MlxBuffer,
    invalid_heads: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekSparseAttentionParams,
) -> Result<()> {
    let (batch, queries, kv_len) = validate_params(params)?;
    let top_k = params.top_k as usize;
    validate_buffer(
        q,
        "q",
        DType::BF16,
        &[
            batch,
            queries,
            DEEPSEEK_SPARSE_HEADS,
            DEEPSEEK_SPARSE_HEAD_DIM,
        ],
    )?;
    validate_buffer(
        kv,
        "kv",
        DType::BF16,
        &[batch, kv_len, DEEPSEEK_SPARSE_HEAD_DIM],
    )?;
    validate_buffer(sinks, "sinks", DType::F32, &[DEEPSEEK_SPARSE_HEADS])?;
    validate_buffer(indices, "indices", DType::I32, &[batch, queries, top_k])?;
    validate_buffer(
        gathered_kv,
        "gathered_kv",
        DType::BF16,
        &[batch, queries, top_k, DEEPSEEK_SPARSE_HEAD_DIM],
    )?;
    let flash_batches = batch.checked_mul(queries).ok_or_else(|| {
        MlxError::InvalidArgument("deepseek sparse flash batch count overflows usize".into())
    })?;
    validate_buffer(mask, "mask", DType::BF16, &[flash_batches, 1, top_k])?;
    validate_buffer(
        invalid_global,
        "invalid_global",
        DType::U32,
        &[batch, queries],
    )?;
    validate_buffer(
        invalid_heads,
        "invalid_heads",
        DType::U32,
        &[batch, queries, DEEPSEEK_SPARSE_HEADS],
    )?;
    validate_buffer(
        output,
        "output",
        DType::BF16,
        &[
            batch,
            queries,
            DEEPSEEK_SPARSE_HEADS,
            DEEPSEEK_SPARSE_HEAD_DIM,
        ],
    )?;

    let validate_pipeline = registry.get_pipeline(
        DEEPSEEK_SPARSE_ATTENTION_VALIDATE_Q_KERNEL,
        device.metal_device(),
    )?;
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args(
        validate_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(sinks)),
            (3, KernelArg::Buffer(invalid_heads)),
        ],
        MTLSize::new((flash_batches * DEEPSEEK_SPARSE_HEADS) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );

    let gather_pipeline = registry.get_pipeline(
        DEEPSEEK_SPARSE_ATTENTION_GATHER_KERNEL,
        device.metal_device(),
    )?;
    let gather_elements = flash_batches * top_k * DEEPSEEK_SPARSE_HEAD_DIM;
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args(
        gather_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(kv)),
            (2, KernelArg::Buffer(indices)),
            (3, KernelArg::Buffer(gathered_kv)),
            (4, KernelArg::Buffer(mask)),
            (5, KernelArg::Buffer(invalid_global)),
        ],
        MTLSize::new(gather_elements.div_ceil(THREADS as usize) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    encoder.memory_barrier();

    dispatch_flash_attn_prefill_bf16_d512_heads_as_rows_with_sinks(
        encoder,
        device,
        registry,
        q,
        gathered_kv,
        gathered_kv,
        mask,
        sinks,
        output,
        &FlashAttnPrefillParams {
            n_heads: 8,
            n_kv_heads: 1,
            head_dim: DEEPSEEK_SPARSE_HEAD_DIM as u32,
            seq_len_q: 8,
            seq_len_k: params.top_k,
            batch: u32::try_from(flash_batches).map_err(|_| {
                MlxError::InvalidArgument("deepseek sparse flash batch count exceeds u32".into())
            })?,
            scale: params.scale,
            do_causal: false,
        },
    )?;
    encoder.memory_barrier();

    let sanitize_pipeline = registry.get_pipeline(
        DEEPSEEK_SPARSE_ATTENTION_SANITIZE_KERNEL,
        device.metal_device(),
    )?;
    let output_elements = flash_batches * DEEPSEEK_SPARSE_HEADS * DEEPSEEK_SPARSE_HEAD_DIM;
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args(
        sanitize_pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(invalid_global)),
            (2, KernelArg::Buffer(invalid_heads)),
            (3, KernelArg::Buffer(output)),
        ],
        MTLSize::new(output_elements.div_ceil(THREADS as usize) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    encoder.memory_barrier();
    Ok(())
}

/// One-token compatibility wrapper over the scalable sparse-flash path.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_sparse_attention_flash_decode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    gathered_kv: &MlxBuffer,
    mask: &MlxBuffer,
    invalid_global: &MlxBuffer,
    invalid_heads: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekSparseAttentionParams,
) -> Result<()> {
    if params.batch != 1 || params.query_len != 1 {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_sparse_attention_flash_decode: requires batch=1 and query_len=1, got {}, {}",
            params.batch, params.query_len,
        )));
    }
    let top_k = params.top_k as usize;
    let gathered = gathered_kv.with_shape(vec![1, 1, top_k, DEEPSEEK_SPARSE_HEAD_DIM])?;
    let flash_mask = mask.with_shape(vec![1, 1, top_k])?;
    let global = invalid_global.with_shape(vec![1, 1])?;
    let heads = invalid_heads.with_shape(vec![1, 1, DEEPSEEK_SPARSE_HEADS])?;
    dispatch_deepseek_sparse_attention_flash_prefill(
        encoder,
        registry,
        device,
        q,
        kv,
        sinks,
        indices,
        &gathered,
        &flash_mask,
        &global,
        &heads,
        output,
        params,
    )
}
