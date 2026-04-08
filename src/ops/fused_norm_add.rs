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
