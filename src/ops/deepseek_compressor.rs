//! Stateful learned KV compression from the official DeepSeek-V4 0731 path.
//!
//! The learned projections are supplied as FP32 `kv` and `score`; this owned
//! primitive applies APE, per-feature gated softmax, incremental state writes,
//! the official BF16-before-RMSNorm boundary, and compressed-cache writes.
//! Cache/output values are normalized but not yet RoPE-transformed.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub const DEEPSEEK_COMPRESS_RATIO_OVERLAP: usize = 4;
pub const DEEPSEEK_COMPRESS_RATIO_LONG: usize = 128;
pub const DEEPSEEK_COMPRESSOR_KERNEL: &str = "deepseek_compressor_bf16";
const THREADS: u64 = 256;

pub static DEEPSEEK_COMPRESSOR_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_compressor.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_COMPRESSOR_KERNEL,
        DEEPSEEK_COMPRESSOR_SHADER_SOURCE,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeepSeekCompressorParams {
    pub batch: u32,
    pub seq_len: u32,
    pub start_pos: u32,
    pub ratio: u32,
    pub head_dim: u32,
    pub cache_len: u32,
    pub epsilon: f32,
}

impl DeepSeekCompressorParams {
    pub fn output_count(&self) -> usize {
        if self.start_pos == 0 {
            (self.seq_len / self.ratio.max(1)) as usize
        } else {
            usize::from(self.start_pos.saturating_add(1) % self.ratio.max(1) == 0)
        }
    }

    pub fn output_slots(&self) -> usize {
        self.output_count().max(1)
    }
}

fn checked_shape(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "deepseek_compressor: shape product overflows: {dims:?}"
            ))
        })
    })
}

fn validate_buffer(buf: &MlxBuffer, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let elements = checked_shape(shape)?;
    if buf.dtype() != dtype {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_compressor: {name} must be {dtype}, got {}",
            buf.dtype()
        )));
    }
    if buf.shape() != shape || buf.byte_len() < elements * dtype.size_of() {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_compressor: {name} shape must be {shape:?}, got {:?}",
            buf.shape()
        )));
    }
    Ok(())
}

fn validate_params(p: &DeepSeekCompressorParams) -> Result<(usize, usize, usize, usize)> {
    if p.batch == 0 || p.seq_len == 0 || p.cache_len == 0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_compressor: batch, seq_len, and cache_len must be nonzero".into(),
        ));
    }
    if p.start_pos != 0 && p.seq_len != 1 {
        return Err(MlxError::InvalidArgument(
            "deepseek_compressor: incremental calls require seq_len=1".into(),
        ));
    }
    if p.start_pos == u32::MAX {
        return Err(MlxError::InvalidArgument(
            "deepseek_compressor: start_pos cannot be u32::MAX".into(),
        ));
    }
    let ratio = p.ratio as usize;
    let dim = p.head_dim as usize;
    let supported = (ratio == DEEPSEEK_COMPRESS_RATIO_OVERLAP && (dim == 128 || dim == 512))
        || (ratio == DEEPSEEK_COMPRESS_RATIO_LONG && dim == 512);
    if !supported {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_compressor: supported (ratio, head_dim) are (4,128), (4,512), and (128,512), got ({ratio},{dim})"
        )));
    }
    if !p.epsilon.is_finite() || p.epsilon <= 0.0 {
        return Err(MlxError::InvalidArgument(
            "deepseek_compressor: epsilon must be finite and positive".into(),
        ));
    }
    let count = p.output_count();
    let last_cache = if p.start_pos == 0 {
        count
    } else if count == 1 {
        p.start_pos as usize / ratio + 1
    } else {
        0
    };
    if last_cache > p.cache_len as usize {
        return Err(MlxError::InvalidArgument(
            "deepseek_compressor: compressed cache is too short for the requested position".into(),
        ));
    }
    Ok((p.batch as usize, p.seq_len as usize, ratio, dim))
}

/// Encode prefill (`start_pos=0`) or one-token incremental compression.
///
/// `kv`/`score` are `[batch, seq_len, coff*head_dim]`, APE is
/// `[ratio, coff*head_dim]`, state is
/// `[batch, coff*ratio, coff*head_dim]`, normalized output is
/// `[batch, max(output_count,1), head_dim]`, and cache is
/// `[batch, cache_len, head_dim]`. `coff=2` only for ratio 4.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_compressor(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    kv: &MlxBuffer,
    score: &MlxBuffer,
    ape: &MlxBuffer,
    norm: &MlxBuffer,
    kv_state: &MlxBuffer,
    score_state: &MlxBuffer,
    output: &MlxBuffer,
    cache: &MlxBuffer,
    params: &DeepSeekCompressorParams,
) -> Result<()> {
    let (batch, seq_len, ratio, dim) = validate_params(params)?;
    let coff = if ratio == DEEPSEEK_COMPRESS_RATIO_OVERLAP {
        2
    } else {
        1
    };
    let projected = coff * dim;
    validate_buffer(kv, "kv", DType::F32, &[batch, seq_len, projected])?;
    validate_buffer(score, "score", DType::F32, &[batch, seq_len, projected])?;
    validate_buffer(ape, "ape", DType::F32, &[ratio, projected])?;
    validate_buffer(norm, "norm", DType::F32, &[dim])?;
    validate_buffer(
        kv_state,
        "kv_state",
        DType::F32,
        &[batch, coff * ratio, projected],
    )?;
    validate_buffer(
        score_state,
        "score_state",
        DType::F32,
        &[batch, coff * ratio, projected],
    )?;
    validate_buffer(
        output,
        "output",
        DType::BF16,
        &[batch, params.output_slots(), dim],
    )?;
    validate_buffer(
        cache,
        "cache",
        DType::BF16,
        &[batch, params.cache_len as usize, dim],
    )?;

    let groups_per_batch = params.output_slots();
    let groups = checked_shape(&[batch, groups_per_batch])?;
    if groups > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "deepseek_compressor: dispatch grid exceeds Metal uint indexing".into(),
        ));
    }
    let pipeline = registry.get_pipeline(DEEPSEEK_COMPRESSOR_KERNEL, device.metal_device())?;
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(params))),
            (1, KernelArg::Buffer(kv)),
            (2, KernelArg::Buffer(score)),
            (3, KernelArg::Buffer(ape)),
            (4, KernelArg::Buffer(norm)),
            (5, KernelArg::Buffer(kv_state)),
            (6, KernelArg::Buffer(score_state)),
            (7, KernelArg::Buffer(output)),
            (8, KernelArg::Buffer(cache)),
        ],
        MTLSize::new(groups as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}
