//! GPU-accelerated MoE gating: top-K expert selection with softmax routing.
//!
//! Given a hidden state and a router weight matrix, computes router logits
//! for all experts, selects the top-K, and returns softmax-normalized
//! routing weights.
//!
//! Designed for Gemma 4: 128 experts, top-8 routing, hidden_dim=2816.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// Parameters for MoE gate routing.
pub struct MoeGateParams {
    /// Hidden state dimension (e.g. 2816 for Gemma 4).
    pub hidden_dim: usize,
    /// Total number of experts (e.g. 128 for Gemma 4).
    pub n_experts: usize,
    /// Number of experts to select (e.g. 8 for Gemma 4).
    pub top_k: usize,
}

/// MSL-compatible parameter struct for the moe_gate kernel.
///
/// Must match `MoeGateParams` struct in `moe_gate.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMoeGateParams {
    hidden_dim: u32,
    n_experts: u32,
    top_k: u32,
}

/// Encode a MoE gate operation: compute router logits, select top-K experts,
/// and softmax-normalize their routing weights.
///
/// # Buffer expectations
///
/// * `hidden_state`   — f32, `[hidden_dim]` (single token's hidden state)
/// * `router_weight`  — f32, `[n_experts, hidden_dim]` (row-major)
/// * `out_expert_ids` — u32, `[top_k]` (output: selected expert indices)
/// * `out_weights`    — f32, `[top_k]` (output: softmax routing weights)
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// * `hidden_dim`, `n_experts`, or `top_k` is zero
/// * `top_k > n_experts`
/// * `n_experts > 128` (shader uses fixed-size arrays)
/// * Buffer sizes are inconsistent
#[allow(clippy::too_many_arguments)]
pub fn moe_gate(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    hidden_state: &MlxBuffer,
    router_weight: &MlxBuffer,
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

    let expected_hidden_bytes = params.hidden_dim * std::mem::size_of::<f32>();
    if hidden_state.byte_len() < expected_hidden_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: hidden_state buffer too small: need {} bytes, have {}",
            expected_hidden_bytes,
            hidden_state.byte_len()
        )));
    }

    let expected_router_bytes =
        params.n_experts * params.hidden_dim * std::mem::size_of::<f32>();
    if router_weight.byte_len() < expected_router_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: router_weight buffer too small: need {} bytes, have {}",
            expected_router_bytes,
            router_weight.byte_len()
        )));
    }

    let expected_ids_bytes = params.top_k * std::mem::size_of::<u32>();
    if out_expert_ids.byte_len() < expected_ids_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: out_expert_ids buffer too small: need {} bytes, have {}",
            expected_ids_bytes,
            out_expert_ids.byte_len()
        )));
    }

    let expected_weights_bytes = params.top_k * std::mem::size_of::<f32>();
    if out_weights.byte_len() < expected_weights_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gate: out_weights buffer too small: need {} bytes, have {}",
            expected_weights_bytes,
            out_weights.byte_len()
        )));
    }

    // --- Build GPU params ---
    let gpu_params = GpuMoeGateParams {
        hidden_dim: params.hidden_dim as u32,
        n_experts: params.n_experts as u32,
        top_k: params.top_k as u32,
    };

    let pipeline = registry.get_pipeline("moe_gate", device)?;

    // Single-thread dispatch — the kernel does all work in one thread.
    // This is correct for Stage 1; Epic 6 can parallelize.
    let grid = MTLSize::new(1, 1, 1);
    let tg_size = MTLSize::new(1, 1, 1);

    let params_bytes = as_bytes(&gpu_params);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(hidden_state)),
            (1, KernelArg::Buffer(router_weight)),
            (2, KernelArg::Buffer(out_expert_ids)),
            (3, KernelArg::Buffer(out_weights)),
            (5, KernelArg::Bytes(params_bytes)),
        ],
        grid,
        tg_size,
    );

    Ok(())
}
