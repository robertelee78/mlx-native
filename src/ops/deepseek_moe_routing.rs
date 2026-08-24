//! DeepSeek-V4 0731 score- and hash-based MoE routing.
//!
//! Routing consumes the F32 gate projection directly. Score routing uses
//! `sqrt(softplus(logit))`, adds the learned bias only while selecting the
//! deterministic top six, then gathers and normalizes the unbiased scores.
//! Hash routing preserves each checkpoint `tid2eid` row verbatim.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub const DEEPSEEK_MOE_EXPERTS: usize = 256;
pub const DEEPSEEK_MOE_TOP_K: usize = 6;
pub const DEEPSEEK_MOE_ROUTE_SCALE: f32 = 1.5;
pub const DEEPSEEK_MOE_SCORE_ROUTE_KERNEL: &str = "deepseek_moe_score_route_f32";
pub const DEEPSEEK_MOE_HASH_ROUTE_KERNEL: &str = "deepseek_moe_hash_route_f32";
pub const DEEPSEEK_MOE_SANITIZE_INDICES_KERNEL: &str = "deepseek_moe_sanitize_indices";
const SCORE_THREADS: u64 = DEEPSEEK_MOE_EXPERTS as u64;
const HASH_THREADS: u64 = 256;

pub static DEEPSEEK_MOE_ROUTING_SHADER_SOURCE: &str =
    include_str!("../shaders/deepseek_moe_routing.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        DEEPSEEK_MOE_SCORE_ROUTE_KERNEL,
        DEEPSEEK_MOE_ROUTING_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_MOE_HASH_ROUTE_KERNEL,
        DEEPSEEK_MOE_ROUTING_SHADER_SOURCE,
    );
    registry.register_source(
        DEEPSEEK_MOE_SANITIZE_INDICES_KERNEL,
        DEEPSEEK_MOE_ROUTING_SHADER_SOURCE,
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeepSeekMoeRoutingParams {
    n_tokens: u32,
    vocab_size: u32,
}

fn checked_shape(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "deepseek_moe_routing: shape product overflows: {dims:?}"
            ))
        })
    })
}

fn validate_buffer(buf: &MlxBuffer, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let elements = checked_shape(shape)?;
    if buf.dtype() != dtype {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_moe_routing: {name} must be {dtype}, got {}",
            buf.dtype()
        )));
    }
    if buf.shape() != shape || buf.byte_len() < elements * dtype.size_of() {
        return Err(MlxError::InvalidArgument(format!(
            "deepseek_moe_routing: {name} shape must be {shape:?}, got {:?}",
            buf.shape()
        )));
    }
    Ok(())
}

fn ranges_overlap(left: &MlxBuffer, right: &MlxBuffer) -> bool {
    let left_start = left.contents_ptr() as usize;
    let right_start = right.contents_ptr() as usize;
    let left_end = left_start.saturating_add(left.byte_len());
    let right_end = right_start.saturating_add(right.byte_len());
    left_start < right_end && right_start < left_end
}

fn validate_common(
    logits: &MlxBuffer,
    out_indices: &MlxBuffer,
    out_weights: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    if n_tokens == 0 || n_tokens > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "deepseek_moe_routing: n_tokens must be in 1..=u32::MAX".into(),
        ));
    }
    validate_buffer(
        logits,
        "logits",
        DType::F32,
        &[n_tokens, DEEPSEEK_MOE_EXPERTS],
    )?;
    validate_buffer(
        out_indices,
        "out_indices",
        DType::I32,
        &[n_tokens, DEEPSEEK_MOE_TOP_K],
    )?;
    validate_buffer(
        out_weights,
        "out_weights",
        DType::F32,
        &[n_tokens, DEEPSEEK_MOE_TOP_K],
    )
}

/// Encode score routing for non-hash layers.
///
/// Layouts are logits `[tokens, 256]`, bias `[256]`, indices `[tokens, 6]`,
/// and weights `[tokens, 6]`. Equal selection scores choose the lower expert
/// ID first. Any nonfinite input affecting a token produces six `-1` indices
/// and zero weights for that token.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_moe_score_route(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    bias: &MlxBuffer,
    out_indices: &MlxBuffer,
    out_weights: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    validate_common(logits, out_indices, out_weights, n_tokens)?;
    validate_buffer(bias, "bias", DType::F32, &[DEEPSEEK_MOE_EXPERTS])?;
    let params = DeepSeekMoeRoutingParams {
        n_tokens: n_tokens as u32,
        vocab_size: 0,
    };
    let pipeline = registry.get_pipeline(DEEPSEEK_MOE_SCORE_ROUTE_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(logits)),
            (2, KernelArg::Buffer(bias)),
            (3, KernelArg::Buffer(out_indices)),
            (4, KernelArg::Buffer(out_weights)),
        ],
        MTLSize::new(n_tokens as u64, 1, 1),
        MTLSize::new(SCORE_THREADS, 1, 1),
    );
    Ok(())
}

/// Encode checkpoint-order hash routing for the first three 0731 layers.
///
/// `token_ids` is `[tokens]` and `tid2eid` is `[vocab_size, 6]`, both I32.
/// Invalid token/expert IDs or nonfinite selected logits fail the token closed.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_moe_hash_route(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    token_ids: &MlxBuffer,
    tid2eid: &MlxBuffer,
    out_indices: &MlxBuffer,
    out_weights: &MlxBuffer,
    n_tokens: usize,
    vocab_size: usize,
) -> Result<()> {
    validate_common(logits, out_indices, out_weights, n_tokens)?;
    if vocab_size == 0 || vocab_size > i32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "deepseek_moe_routing: vocab_size must be in 1..=i32::MAX".into(),
        ));
    }
    validate_buffer(token_ids, "token_ids", DType::I32, &[n_tokens])?;
    validate_buffer(
        tid2eid,
        "tid2eid",
        DType::I32,
        &[vocab_size, DEEPSEEK_MOE_TOP_K],
    )?;
    let params = DeepSeekMoeRoutingParams {
        n_tokens: n_tokens as u32,
        vocab_size: vocab_size as u32,
    };
    let pipeline = registry.get_pipeline(DEEPSEEK_MOE_HASH_ROUTE_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(logits)),
            (2, KernelArg::Buffer(token_ids)),
            (3, KernelArg::Buffer(tid2eid)),
            (4, KernelArg::Buffer(out_indices)),
            (5, KernelArg::Buffer(out_weights)),
        ],
        MTLSize::new((n_tokens as u64).div_ceil(HASH_THREADS), 1, 1),
        MTLSize::new(HASH_THREADS, 1, 1),
    );
    Ok(())
}

/// Convert signed route indices to the unsigned expert-matmul contract.
///
/// Invalid sentinels become expert zero solely to keep downstream pointer
/// arithmetic in range, while `invalid_status` is atomically made nonzero.
/// The caller owns that sticky one-word status for the complete inference
/// transaction and must reject it after the transaction's existing wait.
pub fn dispatch_deepseek_moe_sanitize_indices(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    indices: &MlxBuffer,
    safe_indices: &MlxBuffer,
    invalid_status: &MlxBuffer,
    n_tokens: usize,
) -> Result<()> {
    if n_tokens == 0 || n_tokens > u32::MAX as usize {
        return Err(MlxError::InvalidArgument(
            "deepseek_moe_routing: n_tokens must be in 1..=u32::MAX".into(),
        ));
    }
    let shape = [n_tokens, DEEPSEEK_MOE_TOP_K];
    validate_buffer(indices, "indices", DType::I32, &shape)?;
    validate_buffer(safe_indices, "safe_indices", DType::U32, &shape)?;
    validate_buffer(invalid_status, "invalid_status", DType::U32, &[1])?;
    if !invalid_status.is_cpu_writable() {
        return Err(MlxError::InvalidArgument(
            "deepseek_moe_routing: invalid_status must be writable".into(),
        ));
    }
    if ranges_overlap(invalid_status, indices) || ranges_overlap(invalid_status, safe_indices) {
        return Err(MlxError::InvalidArgument(
            "deepseek_moe_routing: invalid_status must not overlap route buffers".into(),
        ));
    }
    let params = DeepSeekMoeRoutingParams {
        n_tokens: n_tokens as u32,
        vocab_size: 0,
    };
    let pipeline =
        registry.get_pipeline(DEEPSEEK_MOE_SANITIZE_INDICES_KERNEL, device.metal_device())?;
    if encoder.is_capturing() {
        let range = |buffer: &MlxBuffer| {
            let start = buffer.contents_ptr() as usize;
            (start, start + buffer.byte_len())
        };
        encoder.set_pending_buffer_ranges(
            vec![range(indices), range(invalid_status)],
            vec![range(safe_indices), range(invalid_status)],
        );
    }
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(indices)),
            (2, KernelArg::Buffer(safe_indices)),
            (3, KernelArg::Buffer(invalid_status)),
        ],
        MTLSize::new(n_tokens as u64, 1, 1),
        MTLSize::new(DEEPSEEK_MOE_TOP_K as u64, 1, 1),
    );
    Ok(())
}
