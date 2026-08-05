//! Flash-prefill selection-mask construction for DeepSeek-V4 sparse attention.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

const FILL_KERNEL: &str = "deepseek_sparse_prefill_mask_fill_bf16";
const SCATTER_KERNEL: &str = "deepseek_sparse_prefill_mask_scatter_bf16";
const FILL_F16_KERNEL: &str = "deepseek_sparse_prefill_mask_fill_f16";
const SCATTER_F16_KERNEL: &str = "deepseek_sparse_prefill_mask_scatter_f16";
const THREADS: u64 = 256;

pub static SHADER_SOURCE: &str = include_str!("../shaders/deepseek_sparse_prefill_mask.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(FILL_KERNEL, SHADER_SOURCE);
    registry.register_source(SCATTER_KERNEL, SHADER_SOURCE);
    registry.register_source(FILL_F16_KERNEL, SHADER_SOURCE);
    registry.register_source(SCATTER_F16_KERNEL, SHADER_SOURCE);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeepSeekSparsePrefillMaskParams {
    pub batch: u32,
    pub query_len: u32,
    /// Number of real K/V rows.
    pub kv_len: u32,
    pub top_k: u32,
    pub heads: u32,
}

fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "deepseek_sparse_prefill_mask: shape product overflows: {dims:?}"
            ))
        })
    })
}

/// Build a BF16 additive flash-attention mask.
///
/// Selected sparse positions receive `0`; all other K/V positions receive
/// `-inf`. Learned sinks are handled natively by the flash reducer.
pub fn dispatch_deepseek_sparse_prefill_mask(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    indices: &MlxBuffer,
    mask: &MlxBuffer,
    params: &DeepSeekSparsePrefillMaskParams,
) -> Result<()> {
    dispatch_typed(
        encoder,
        registry,
        device,
        indices,
        mask,
        params,
        DType::BF16,
        FILL_KERNEL,
        SCATTER_KERNEL,
    )
}

/// F16 sibling used by llama.cpp-compatible D=512 flash prefill.
pub fn dispatch_deepseek_sparse_prefill_mask_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    indices: &MlxBuffer,
    mask: &MlxBuffer,
    params: &DeepSeekSparsePrefillMaskParams,
) -> Result<()> {
    dispatch_typed(
        encoder,
        registry,
        device,
        indices,
        mask,
        params,
        DType::F16,
        FILL_F16_KERNEL,
        SCATTER_F16_KERNEL,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_typed(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    indices: &MlxBuffer,
    mask: &MlxBuffer,
    params: &DeepSeekSparsePrefillMaskParams,
    mask_dtype: DType,
    fill_kernel: &str,
    scatter_kernel: &str,
) -> Result<()> {
    if params.batch == 0
        || params.query_len == 0
        || params.kv_len == 0
        || params.top_k == 0
        || params.heads == 0
    {
        return Err(MlxError::InvalidArgument(
            "deepseek_sparse_prefill_mask: dimensions must be nonzero".into(),
        ));
    }
    let batch = params.batch as usize;
    let queries = params.query_len as usize;
    let kv_len = params.kv_len as usize;
    let top_k = params.top_k as usize;
    let heads = params.heads as usize;
    let index_elements = checked_product(&[batch, queries, top_k])?;
    let broadcast = mask.shape() == [queries, kv_len];
    if broadcast && batch != 1 {
        return Err(MlxError::InvalidArgument(
            "deepseek_sparse_prefill_mask: rank-2 broadcast masks require batch=1".into(),
        ));
    }
    let mask_heads = if broadcast { 1 } else { heads };
    let mask_elements = checked_product(&[batch, mask_heads, queries, kv_len])?;
    if indices.dtype() != DType::I32
        || indices.shape() != [batch, queries, top_k]
        || indices.byte_len() < index_elements * DType::I32.size_of()
    {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_sparse_prefill_mask: indices must be I32 [{batch}, {queries}, {top_k}], got {} {:?}",
            indices.dtype(),
            indices.shape()
        )));
    }
    let expected_mask = if broadcast {
        vec![queries, kv_len]
    } else {
        vec![batch, heads, queries, kv_len]
    };
    if mask.dtype() != mask_dtype
        || mask.shape() != expected_mask
        || mask.byte_len() < mask_elements * mask_dtype.size_of()
    {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_sparse_prefill_mask: mask must be {mask_dtype} {:?}, got {} {:?}",
            expected_mask,
            mask.dtype(),
            mask.shape()
        )));
    }

    let shader_params = DeepSeekSparsePrefillMaskParams {
        batch: params.batch,
        query_len: params.query_len,
        kv_len: params.kv_len,
        top_k: params.top_k,
        heads: mask_heads as u32,
    };

    let fill = registry.get_pipeline(fill_kernel, device.metal_device())?;
    encoder.set_op_kind(CapturedOpKind::Other);
    encoder.encode_with_args(
        fill,
        &[
            (0, KernelArg::Bytes(as_bytes(&shader_params))),
            (1, KernelArg::Buffer(mask)),
        ],
        MTLSize::new(mask_elements as u64, 1, 1),
        MTLSize::new(THREADS.min(mask_elements as u64), 1, 1),
    );
    encoder.memory_barrier();
    let scatter_elements = checked_product(&[batch, queries, top_k, mask_heads])?;
    let scatter = registry.get_pipeline(scatter_kernel, device.metal_device())?;
    encoder.encode_with_args(
        scatter,
        &[
            (0, KernelArg::Bytes(as_bytes(&shader_params))),
            (1, KernelArg::Buffer(indices)),
            (2, KernelArg::Buffer(mask)),
        ],
        MTLSize::new(scatter_elements as u64, 1, 1),
        MTLSize::new(THREADS.min(scatter_elements as u64), 1, 1),
    );
    Ok(())
}
