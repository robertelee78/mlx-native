//! GPU fused softmax + top-K + renorm for MoE routing.
//!
//! Replaces `softmax_topk_renorm_cpu()` in `build_moe_ffn_layer_gpu_q`.
//! Outputs `ids [n_tokens * top_k]` u32 and `weights [n_tokens * top_k]` f32
//! entirely on the GPU, eliminating the CPU download of router logits and the
//! associated `commit_and_wait()` barrier between the router projection and the
//! expert matmuls.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Metal shader source (compiled lazily on first call).
pub static MOE_SOFTMAX_TOPK_SHADER_SOURCE: &str =
    include_str!("../shaders/moe_softmax_topk.metal");

/// Register the `moe_softmax_topk_f32` pipeline.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("moe_softmax_topk_f32", MOE_SOFTMAX_TOPK_SHADER_SOURCE);
}

/// GPU-side params struct; matches `MoeSoftmaxTopkParams` in the shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MoeSoftmaxTopkGpuParams {
    n_tokens:  u32,
    n_experts: u32,
    top_k:     u32,
    _pad:      f32,
}

/// Dispatch a fused softmax + top-K + renorm MoE routing kernel.
///
/// # Arguments
///
/// * `logits`       — F32 `[n_tokens, n_experts]` — raw router logits.
/// * `out_ids`      — U32 `[n_tokens * top_k]`   — expert IDs per (token, slot).
/// * `out_weights`  — F32 `[n_tokens * top_k]`   — routing weights (renormalized).
/// * `n_tokens`     — Number of tokens.
/// * `n_experts`    — Total number of experts.
/// * `top_k`        — Number of experts per token. Must be <= 64.
///
/// # Errors
///
/// `MlxError::InvalidArgument` for out-of-range parameters or buffer sizes.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_moe_softmax_topk(
    encoder:     &mut CommandEncoder,
    registry:    &mut KernelRegistry,
    device:      &MlxDevice,
    logits:      &MlxBuffer,
    out_ids:     &MlxBuffer,
    out_weights: &MlxBuffer,
    n_tokens:    u32,
    n_experts:   u32,
    top_k:       u32,
) -> Result<()> {
    if n_tokens == 0 || n_experts == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_softmax_topk: n_tokens, n_experts, and top_k must be > 0".into(),
        ));
    }
    if top_k > 64 {
        return Err(MlxError::InvalidArgument(format!(
            "moe_softmax_topk: top_k ({top_k}) > 64 (kernel supports up to 64)"
        )));
    }
    if top_k > n_experts {
        return Err(MlxError::InvalidArgument(format!(
            "moe_softmax_topk: top_k ({top_k}) > n_experts ({n_experts})"
        )));
    }

    let expected_logits  = (n_tokens  as usize) * (n_experts as usize) * DType::F32.size_of();
    let expected_ids     = (n_tokens  as usize) * (top_k     as usize) * DType::U32.size_of();
    let expected_weights = (n_tokens  as usize) * (top_k     as usize) * DType::F32.size_of();

    if logits.byte_len() < expected_logits {
        return Err(MlxError::InvalidArgument(format!(
            "moe_softmax_topk: logits too small (expected {expected_logits}, got {})",
            logits.byte_len()
        )));
    }
    if out_ids.byte_len() < expected_ids {
        return Err(MlxError::InvalidArgument(format!(
            "moe_softmax_topk: out_ids too small (expected {expected_ids}, got {})",
            out_ids.byte_len()
        )));
    }
    if out_weights.byte_len() < expected_weights {
        return Err(MlxError::InvalidArgument(format!(
            "moe_softmax_topk: out_weights too small (expected {expected_weights}, got {})",
            out_weights.byte_len()
        )));
    }

    let gpu_params = MoeSoftmaxTopkGpuParams {
        n_tokens,
        n_experts,
        top_k,
        _pad: 0.0,
    };

    let pipeline = registry.get_pipeline("moe_softmax_topk_f32", device.metal_device())?;

    // Threadgroup size = min(n_experts, 128), capped to avoid exceeding limits.
    let tg_size = (n_experts as u64).min(128);
    // Shared memory: 2 * tg_size floats (max + sum) + n_experts floats (probs).
    // = (2 * 128 + n_experts) * 4 bytes.  For n_experts=64: (256+64)*4 = 1280 bytes.
    let shmem_bytes = (2 * tg_size + n_experts as u64) * 4;

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(logits)),
            (2, KernelArg::Buffer(out_ids)),
            (3, KernelArg::Buffer(out_weights)),
        ],
        &[(0, shmem_bytes)],
        MTLSize::new(n_tokens as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
