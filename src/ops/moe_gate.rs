//! GPU-accelerated MoE gating: parallel top-K expert selection with softmax
//! routing.
//!
//! One threadgroup per token (grid = seq_len × 1 × 1), 128 threads per group.
//! Supports bf16 hidden state input, f32 router weights, and per-expert scale.
//!
//! Designed for Gemma 4: 128 experts, top-8 routing, hidden_dim=2816.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Parameters for MoE gate routing.
pub struct MoeGateParams {
    /// Hidden state dimension (e.g. 2816 for Gemma 4).
    pub hidden_dim: usize,
    /// Total number of experts (e.g. 128 for Gemma 4).
    pub n_experts: usize,
    /// Number of experts to select (e.g. 8 for Gemma 4).
    pub top_k: usize,
    /// Number of tokens in the sequence (seq_len >= 1).
    pub seq_len: usize,
    /// RMS norm epsilon (e.g. 1e-6).
    pub rms_eps: f32,
}

/// Encode a parallel MoE gate operation.
///
/// Launches one threadgroup per token; each threadgroup runs 128 threads that
/// cooperate on:
///   1. RMS-Norm of the token's hidden state.
///   2. Router matmul (each thread handles ⌈n_experts/128⌉ experts).
///   3. Top-K insertion sort + softmax + per_expert_scale (single thread).
///
/// # Buffer expectations
///
/// * `hidden_state`      — bf16, `[seq_len, hidden_dim]`
/// * `router_weights`    — f32,  `[n_experts, hidden_dim]` (row-major, pre-cached on GPU)
/// * `norm_weight`       — f32,  `[hidden_dim]` (RMS norm learned weight)
/// * `per_expert_scale`  — f32,  `[n_experts]`
/// * `out_expert_ids`    — u32,  `[seq_len, top_k]`  (output)
/// * `out_weights`       — f32,  `[seq_len, top_k]`  (output)
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if any parameter or buffer is invalid.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    hidden_state: &MlxBuffer,
    router_weights: &MlxBuffer,
    norm_weight: &MlxBuffer,
    per_expert_scale: &MlxBuffer,
    out_expert_ids: &MlxBuffer,
    out_weights: &MlxBuffer,
    params: &MoeGateParams,
) -> Result<()> {
    // --- Validation ---
    if params.hidden_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_gate: hidden_dim must be > 0".into(),
        ));
    }
    if params.n_experts == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_gate: n_experts must be > 0".into(),
        ));
    }
    if params.top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_gate: top_k must be > 0".into(),
        ));
    }
    if params.seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_gate: seq_len must be > 0".into(),
        ));
    }
    if params.top_k > params.n_experts {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: top_k ({}) must be <= n_experts ({})",
            params.top_k, params.n_experts
        )));
    }
    if params.n_experts > 128 {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: n_experts ({}) exceeds max 128 (shader fixed-size array limit)",
            params.n_experts
        )));
    }

    // bf16 elements are 2 bytes each
    let bf16_size = 2usize;
    let f32_size = std::mem::size_of::<f32>();
    let u32_size = std::mem::size_of::<u32>();

    let expected_hidden_bytes = params.seq_len * params.hidden_dim * bf16_size;
    if hidden_state.byte_len() < expected_hidden_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: hidden_state buffer too small: need {} bytes, have {}",
            expected_hidden_bytes,
            hidden_state.byte_len()
        )));
    }

    let expected_router_bytes = params.n_experts * params.hidden_dim * f32_size;
    if router_weights.byte_len() < expected_router_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: router_weights buffer too small: need {} bytes, have {}",
            expected_router_bytes,
            router_weights.byte_len()
        )));
    }

    let expected_norm_bytes = params.hidden_dim * f32_size;
    if norm_weight.byte_len() < expected_norm_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: norm_weight buffer too small: need {} bytes, have {}",
            expected_norm_bytes,
            norm_weight.byte_len()
        )));
    }

    let expected_scale_bytes = params.n_experts * f32_size;
    if per_expert_scale.byte_len() < expected_scale_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: per_expert_scale buffer too small: need {} bytes, have {}",
            expected_scale_bytes,
            per_expert_scale.byte_len()
        )));
    }

    let expected_ids_bytes = params.seq_len * params.top_k * u32_size;
    if out_expert_ids.byte_len() < expected_ids_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: out_expert_ids buffer too small: need {} bytes, have {}",
            expected_ids_bytes,
            out_expert_ids.byte_len()
        )));
    }

    let expected_weights_bytes = params.seq_len * params.top_k * f32_size;
    if out_weights.byte_len() < expected_weights_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: out_weights buffer too small: need {} bytes, have {}",
            expected_weights_bytes,
            out_weights.byte_len()
        )));
    }

    // --- Kernel dispatch ---
    let pipeline = registry.get_pipeline("moe_gate", device)?;

    // 128 threads per threadgroup — one per expert for the matmul phase.
    // Must be a power of 2 for the tree-reduction in RMS norm.
    let tg_threads: u64 = 128;

    // One threadgroup per token.
    let threadgroups = MTLSize::new(params.seq_len as u64, 1, 1);
    let threadgroup_size = MTLSize::new(tg_threads, 1, 1);

    // Threadgroup shared memory layout (see moe_gate.metal):
    //   [0 .. hidden_dim)                   — f32 normed hidden state
    //   [hidden_dim .. hidden_dim+n_experts) — f32 router logits / reduction scratch
    //
    // Both regions are large enough for the RMS reduction (uses logit region as
    // scratch with tg_size=128 slots, which fits within n_experts=128 floats).
    let shared_bytes =
        ((params.hidden_dim + params.n_experts) * std::mem::size_of::<f32>()) as u64;

    // Scalar constants passed as inline bytes via set_bytes.
    let hidden_dim_u32 = params.hidden_dim as u32;
    let n_experts_u32  = params.n_experts  as u32;
    let top_k_u32      = params.top_k      as u32;
    let rms_eps_f32    = params.rms_eps;

    use crate::encoder::{KernelArg, as_bytes};

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(hidden_state)),
            (1, KernelArg::Buffer(router_weights)),
            (2, KernelArg::Buffer(norm_weight)),
            (3, KernelArg::Buffer(per_expert_scale)),
            (4, KernelArg::Buffer(out_expert_ids)),
            (5, KernelArg::Buffer(out_weights)),
            (6, KernelArg::Bytes(as_bytes(&hidden_dim_u32))),
            (7, KernelArg::Bytes(as_bytes(&n_experts_u32))),
            (8, KernelArg::Bytes(as_bytes(&top_k_u32))),
            (9, KernelArg::Bytes(as_bytes(&rms_eps_f32))),
        ],
        &[(0, shared_bytes)],
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}
