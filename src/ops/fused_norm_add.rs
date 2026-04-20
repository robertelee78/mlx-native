//! Fused RMS normalization + residual addition GPU dispatch (bf16).
//!
//! Implements Gemma4's post-attention and post-FFN ordering:
//!   `normed = rms_norm(input, weight, eps)`
//!   `output = residual + normed`
//!
//! Replaces two separate dispatches — `rms_norm_bf16` followed by
//! `elementwise_add_bf16` — with a single kernel launch per sub-layer.
//! Across Gemma4's 30 layers with 2-4 such operations each, this saves
//! 60-120 kernel dispatches per forward pass.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_threadgroups_with_args_and_shared, KernelArg};

/// MSL source embedded at compile time.
pub static FUSED_NORM_ADD_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_norm_add_bf16.metal");

/// Register both fused norm-add shader variants with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("fused_norm_add_bf16", FUSED_NORM_ADD_SHADER_SOURCE);
    registry.register_source(
        "fused_norm_add_no_weight_bf16",
        FUSED_NORM_ADD_SHADER_SOURCE,
    );
}

/// Compute the threadgroup size for a given dimension: smallest power-of-two
/// >= dim, capped at 256.
#[inline]
fn tg_size_for_dim(dim: u32) -> u64 {
    std::cmp::min(256, dim.next_power_of_two()) as u64
}

/// Dispatch a fused RMS normalization + residual addition operation.
///
/// Computes:
/// ```text
/// normed[i] = rms_norm(input[i], weight[i], eps)
/// output[i] = residual[i] + normed[i]
/// ```
///
/// This is the correct ordering for Gemma4's post-attention and post-FFN paths:
///   `residual1 = hidden_state + post_attn_norm(attn_output)`
///   `residual2 = residual1   + post_ffn_norm(ffn_output)`
///
/// # Arguments
///
/// * `encoder`  - Command encoder to record the dispatch into.
/// * `registry` - Kernel registry (must have `fused_norm_add_bf16` registered).
/// * `device`   - Metal device for pipeline compilation.
/// * `residual` - bf16 buffer of shape `[rows, dim]` — the residual stream.
/// * `input`    - bf16 buffer of shape `[rows, dim]` — sublayer output to normalize.
/// * `weight`   - bf16 buffer of shape `[dim]`       — RMS norm learned scale.
/// * `output`   - bf16 output buffer of shape `[rows, dim]` — `residual + normed`.
/// * `dim`      - Hidden dimension (last axis size).
/// * `rows`     - Number of rows (tokens) in the batch.
/// * `eps`      - RMS normalization epsilon (1e-6 for Gemma4).
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] if any buffer has the wrong element count
/// or if `rows`/`dim` are zero.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_norm_add_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    residual: &MlxBuffer,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    dim: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_norm_add: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);

    if residual.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_norm_add: residual element count {} != rows({}) * dim({})",
            residual.element_count(),
            rows,
            dim,
        )));
    }
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_norm_add: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim,
        )));
    }
    if weight.element_count() != dim as usize {
        return Err(MlxError::InvalidArgument(format!(
            "fused_norm_add: weight element count {} != dim({})",
            weight.element_count(),
            dim,
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_norm_add: output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim,
        )));
    }

    let pipeline = registry.get_pipeline("fused_norm_add_bf16", device)?;

    let tg_size = tg_size_for_dim(dim);
    // Shared memory: tg_size f32 slots for the sum-of-squares reduction.
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(residual)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(weight)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(as_bytes(&dim))),
            (5, KernelArg::Bytes(as_bytes(&rows))),
            (6, KernelArg::Bytes(as_bytes(&eps))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch a fused RMS normalization (no learned weight) + residual addition.
///
/// Computes:
/// ```text
/// normed[i] = input[i] * rsqrt(mean(input^2) + eps)   // no weight scale
/// output[i] = residual[i] + normed[i]
/// ```
///
/// Used for V-head norms in Gemma4 that have no learned scale parameter.
///
/// # Arguments
///
/// * `encoder`  - Command encoder to record the dispatch into.
/// * `registry` - Kernel registry (must have `fused_norm_add_no_weight_bf16` registered).
/// * `device`   - Metal device for pipeline compilation.
/// * `residual` - bf16 buffer of shape `[rows, dim]` — the residual stream.
/// * `input`    - bf16 buffer of shape `[rows, dim]` — sublayer output to normalize.
/// * `output`   - bf16 output buffer of shape `[rows, dim]` — `residual + normed`.
/// * `dim`      - Hidden dimension (last axis size).
/// * `rows`     - Number of rows (tokens) in the batch.
/// * `eps`      - RMS normalization epsilon.
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] if any buffer has the wrong element count
/// or if `rows`/`dim` are zero.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_norm_add_no_weight_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    residual: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    dim: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_norm_add_no_weight: rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);

    if residual.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_norm_add_no_weight: residual element count {} != rows({}) * dim({})",
            residual.element_count(),
            rows,
            dim,
        )));
    }
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_norm_add_no_weight: input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim,
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "fused_norm_add_no_weight: output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim,
        )));
    }

    let pipeline = registry.get_pipeline("fused_norm_add_no_weight_bf16", device)?;

    let tg_size = tg_size_for_dim(dim);
    let shared_mem_bytes = tg_size * 4;

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(residual)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&dim))),
            (4, KernelArg::Bytes(as_bytes(&rows))),
            (5, KernelArg::Bytes(as_bytes(&eps))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

// =========================================================================
// F32 variants
// =========================================================================

/// Dispatch a fused RMS normalization + residual addition (f32).
///
/// Computes:
/// ```text
/// normed[i] = rms_norm(input[i], weight[i], eps)
/// output[i] = residual[i] + normed[i]
/// ```
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_norm_add_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    residual: &MlxBuffer,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    dim: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_norm_add_f32: rows and dim must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_norm_add_f32", device)?;

    let tg_size = tg_size_for_dim(dim);
    let shared_mem_bytes = tg_size * 4;

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(residual)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(weight)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(as_bytes(&dim))),
            (5, KernelArg::Bytes(as_bytes(&rows))),
            (6, KernelArg::Bytes(as_bytes(&eps))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch fused MoE-weighted-sum + residual + RMS norm (f32).  Wave P4.13.
///
/// Replaces two dispatches (moe_weighted_sum_seq + fused_norm_add_f32) with
/// one kernel that:
///   1. Computes weighted_sum[i] = sum_k expert_outputs[i, k] * weights[k]
///   2. Computes RMS over the weighted sum
///   3. Writes output[i] = residual[i] + weighted_sum[i] * rms_inv * norm_weight[i]
///
/// Eliminates one global write+read of the [rows * dim] intermediate sum
/// buffer (~5 MB at pp2455 × 30 = 150 MB of read+write traffic).
///
/// # Arguments
///
/// * `expert_outputs` - MoE down outputs `[rows * top_k * dim]` (f32).
/// * `weights`        - Routing weights `[rows * top_k]` (f32).
/// * `residual`       - Residual stream `[rows * dim]` (f32).
/// * `norm_weight`    - RMS norm scale `[dim]` (f32).
/// * `output`         - Output buffer `[rows * dim]` (f32).
/// * `dim`            - Hidden dim.
/// * `top_k`          - Number of experts per token.
/// * `rows`           - Number of tokens.
/// * `eps`            - RMS norm epsilon.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_moe_wsum_norm_add_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    expert_outputs: &MlxBuffer,
    weights: &MlxBuffer,
    residual: &MlxBuffer,
    norm_weight: &MlxBuffer,
    output: &MlxBuffer,
    dim: u32,
    top_k: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || dim == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_moe_wsum_norm_add_f32: rows, dim, top_k must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_moe_wsum_norm_add_f32", device)?;

    // Threadgroup memory: tg_size floats for sum-of-squares reduction
    // scratch + dim floats for the per-row weighted_sum result buffer.
    let tg_size = tg_size_for_dim(dim);
    let shared_mem_bytes = ((tg_size as u64) + (dim as u64)) * 4;

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(expert_outputs)),
            (1, KernelArg::Buffer(weights)),
            (2, KernelArg::Buffer(residual)),
            (3, KernelArg::Buffer(norm_weight)),
            (4, KernelArg::Buffer(output)),
            (5, KernelArg::Bytes(as_bytes(&dim))),
            (6, KernelArg::Bytes(as_bytes(&top_k))),
            (7, KernelArg::Bytes(as_bytes(&rows))),
            (8, KernelArg::Bytes(as_bytes(&eps))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Dispatch fused MoE-weighted-sum + DOUBLE RMS norm + add (f32).  Wave P4.14.
///
/// Replaces a three-dispatch sequence (norm of pf_mlp_down +
/// moe_weighted_sum + fused_norm_add of weighted into normed mlp_down) with
/// one kernel.  See shader comment for the algorithm; saves 2 dispatches
/// per layer (60/prefill on Gemma 4) and eliminates two intermediate
/// buffer write+read cycles (~10 MB at pp2455 × 30 = 300 MB of memory
/// traffic).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_moe_wsum_dnorm_add_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    expert_outputs: &MlxBuffer,
    weights: &MlxBuffer,
    residual: &MlxBuffer,
    res_norm_weight: &MlxBuffer,
    moe_norm_weight: &MlxBuffer,
    output: &MlxBuffer,
    dim: u32,
    top_k: u32,
    rows: u32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || dim == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_moe_wsum_dnorm_add_f32: rows, dim, top_k must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_moe_wsum_dnorm_add_f32", device)?;

    // Threadgroup memory: 2 * tg_size (two reduction scratch arrays) + dim
    // (per-row sum_buf for the weighted-sum carryover).
    let tg_size = tg_size_for_dim(dim);
    let shared_mem_bytes = ((2u64 * tg_size as u64) + (dim as u64)) * 4;

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(expert_outputs)),
            (1, KernelArg::Buffer(weights)),
            (2, KernelArg::Buffer(residual)),
            (3, KernelArg::Buffer(res_norm_weight)),
            (4, KernelArg::Buffer(moe_norm_weight)),
            (5, KernelArg::Buffer(output)),
            (6, KernelArg::Bytes(as_bytes(&dim))),
            (7, KernelArg::Bytes(as_bytes(&top_k))),
            (8, KernelArg::Bytes(as_bytes(&rows))),
            (9, KernelArg::Bytes(as_bytes(&eps))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// GPU params struct for f32 variant — must match `FusedResidualNormF32Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedResidualNormF32Params {
    dim:       u32,
    rows:      u32,
    eps:       f32,
    write_sum: u32,
}

/// Dispatch fused residual add + RMS norm (f32).
///
/// Computes `normed = rms_norm(residual + input, weight, eps)` in one pass.
/// Optionally writes the un-normalized sum to `sum_output`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_residual_norm_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    residual: &MlxBuffer,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    normed_output: &MlxBuffer,
    sum_output: Option<&MlxBuffer>,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_residual_norm_f32: rows and dim must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_residual_norm_f32", device)?;

    let tg_size = tg_size_for_dim(dim);
    let shared_slots = std::cmp::max(tg_size as u32, dim);
    let shared_mem_bytes = (shared_slots as u64) * 4;

    let write_sum = sum_output.is_some();
    let gpu_params = GpuFusedResidualNormF32Params {
        dim,
        rows,
        eps,
        write_sum: u32::from(write_sum),
    };

    let sum_buf = sum_output.unwrap_or(normed_output);

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(residual)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(weight)),
            (3, KernelArg::Buffer(normed_output)),
            (4, KernelArg::Buffer(sum_buf)),
            (5, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// GPU params struct — must match `FusedResidualNormScalarF32Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedResidualNormScalarF32Params {
    dim:              u32,
    rows:             u32,
    eps:              f32,
    scalar_is_vector: u32,
}

/// Dispatch fused residual add + RMS norm + scalar multiply (f32).
///
/// Computes: `output = rms_norm(residual + input, weight, eps) * scalar`
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_residual_norm_scalar_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    residual: &MlxBuffer,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    scalar: &MlxBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    scalar_is_vector: bool,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_residual_norm_scalar_f32: rows and dim must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_residual_norm_scalar_f32", device)?;

    let tg_size = tg_size_for_dim(dim);
    let shared_slots = std::cmp::max(tg_size as u32, dim);
    let shared_mem_bytes = (shared_slots as u64) * 4;

    let gpu_params = GpuFusedResidualNormScalarF32Params {
        dim,
        rows,
        eps,
        scalar_is_vector: u32::from(scalar_is_vector),
    };

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(residual)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(weight)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Buffer(scalar)),
            (5, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// GPU params struct — must match `FusedMoeRoutingParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedMoeRoutingParams {
    num_experts: u32,
    top_k:       u32,
}

/// Dispatch fused MoE routing: softmax + argsort + gather top-K weights (f32).
///
/// Replaces 3 separate dispatches (softmax + argsort + gather_topk) with one.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_moe_routing_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    logits: &MlxBuffer,
    expert_ids: &MlxBuffer,
    routing_weights: &MlxBuffer,
    per_expert_scale: &MlxBuffer,
    num_experts: u32,
    top_k: u32,
) -> Result<()> {
    if num_experts == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_moe_routing_f32: num_experts and top_k must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_moe_routing_f32", device)?;

    let gpu_params = GpuFusedMoeRoutingParams {
        num_experts,
        top_k,
    };

    let tg_size = std::cmp::min(64, num_experts.next_power_of_two()) as u64;
    let shared_slots = 2 * num_experts + tg_size as u32;
    let shared_mem_bytes = (shared_slots as u64) * 4;

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(logits)),
            (1, KernelArg::Buffer(expert_ids)),
            (2, KernelArg::Buffer(routing_weights)),
            (3, KernelArg::Buffer(per_expert_scale)),
            (4, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(1, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Batched fused MoE routing for prefill.
///
/// Processes `n_tokens` routings in one dispatch. Same semantics as
/// `dispatch_fused_moe_routing_f32` but with batched buffers.
///
/// Buffer shapes:
/// - `logits`          `[n_tokens, num_experts]`
/// - `expert_ids`      `[n_tokens, top_k]`
/// - `routing_weights` `[n_tokens, top_k]`
/// - `per_expert_scale` `[num_experts]`
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_moe_routing_batch_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    logits: &MlxBuffer,
    expert_ids: &MlxBuffer,
    routing_weights: &MlxBuffer,
    per_expert_scale: &MlxBuffer,
    num_experts: u32,
    top_k: u32,
    n_tokens: u32,
) -> Result<()> {
    if num_experts == 0 || top_k == 0 || n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_moe_routing_batch_f32: num_experts, top_k, n_tokens must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_moe_routing_batch_f32", device)?;

    let gpu_params = GpuFusedMoeRoutingParams {
        num_experts,
        top_k,
    };

    let tg_size = std::cmp::min(64, num_experts.next_power_of_two()) as u64;
    let shared_slots = 2 * num_experts + tg_size as u32;
    let shared_mem_bytes = (shared_slots as u64) * 4;

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(logits)),
            (1, KernelArg::Buffer(expert_ids)),
            (2, KernelArg::Buffer(routing_weights)),
            (3, KernelArg::Buffer(per_expert_scale)),
            (4, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(n_tokens as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// F32 — fused norm(input) + add(residual) + scalar multiply
// Computes: output = (residual + rms_norm(input, weight)) * scalar
// This is the CORRECT end-of-layer operation (norm on input alone, not on sum).
// ---------------------------------------------------------------------------

/// GPU params struct — must match `FusedNormAddScalarF32Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedNormAddScalarF32Params {
    dim:              u32,
    rows:             u32,
    eps:              f32,
    scalar_is_vector: u32,
}

/// Dispatch fused norm + add + scalar multiply (f32).
///
/// Computes: `output = (residual + rms_norm(input, weight, eps)) * scalar`
///
/// The norm is applied to `input` ALONE, then the normed result is added to
/// `residual`, then scaled.  This matches Gemma 4's end-of-layer pattern.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_norm_add_scalar_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    residual: &MlxBuffer,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    scalar: &MlxBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    scalar_is_vector: bool,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_norm_add_scalar_f32: rows and dim must be > 0".into(),
        ));
    }

    let pipeline = registry.get_pipeline("fused_norm_add_scalar_f32", device)?;

    let tg_size = tg_size_for_dim(dim);
    let shared_mem_bytes = tg_size * 4;

    let gpu_params = GpuFusedNormAddScalarF32Params {
        dim,
        rows,
        eps,
        scalar_is_vector: u32::from(scalar_is_vector),
    };

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(residual)),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(weight)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Buffer(scalar)),
            (5, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
