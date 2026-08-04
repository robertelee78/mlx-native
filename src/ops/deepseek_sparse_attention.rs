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

pub const DEEPSEEK_SPARSE_HEADS: usize = 64;
pub const DEEPSEEK_SPARSE_HEAD_DIM: usize = 512;
pub const DEEPSEEK_INDEX_TOP_K: usize = 512;
pub const DEEPSEEK_SPARSE_ATTENTION_KERNEL: &str = "deepseek_sparse_attention_bf16";
const THREADS: u64 = 256;

pub static DEEPSEEK_SPARSE_ATTENTION_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_sparse_attention.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_SPARSE_ATTENTION_KERNEL,
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

    let pipeline =
        registry.get_pipeline(DEEPSEEK_SPARSE_ATTENTION_KERNEL, device.metal_device())?;
    let groups = batch * queries * DEEPSEEK_SPARSE_HEADS;
    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(kv)),
            (3, KernelArg::Buffer(sinks)),
            (4, KernelArg::Buffer(indices)),
            (5, KernelArg::Buffer(output)),
        ],
        MTLSize::new(groups as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}
