//! Fused per-head RMS normalization + NeoX RoPE GPU dispatch (bf16).
//!
//! Replaces two separate dispatches — `rms_norm_bf16` (per head) followed by
//! `rope_neox_bf16` — with a single kernel launch per Q or K projection.
//! Saves approximately 102 kernel dispatches per Gemma 4 forward pass.
//!
//! The Metal kernel reads each head's slice once, computes the RMS norm,
//! applies the optional weight scale, performs the NeoX rotation in-register,
//! and writes the result — no intermediate buffer required.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_threadgroups_with_args_and_shared, KernelArg};

/// MSL source embedded at compile time.
pub static FUSED_HEAD_NORM_ROPE_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_head_norm_rope_bf16.metal");

/// Register the fused head-norm + RoPE shader with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "fused_head_norm_rope_bf16",
        FUSED_HEAD_NORM_ROPE_SHADER_SOURCE,
    );
}

/// GPU params struct — must match `FusedHeadNormRopeParams` in the MSL shader exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedHeadNormRopeParams {
    head_dim:      u32,
    n_heads:       u32,
    half_rope_dim: u32,
    eps:           f32,
    has_weight:    u32, // 0 = no weight (V-norm variant), 1 = apply weight
}

/// Dispatch a fused per-head RMS normalization + NeoX RoPE operation.
///
/// Processes a single token's Q or K projection: normalizes each head's slice
/// with RMS norm, optionally applies a per-dimension weight, then applies the
/// NeoX-convention rotary embedding using precomputed cos/sin caches.
///
/// # Arguments
///
/// * `encoder`       - Command encoder to record the dispatch into.
/// * `registry`      - Kernel registry (must have fused_head_norm_rope_bf16 registered).
/// * `device`        - Metal device for pipeline compilation.
/// * `input`         - bf16 buffer of shape `[n_heads * head_dim]` (one token's Q or K).
/// * `output`        - bf16 output buffer, same shape as `input`.
/// * `norm_weight`   - bf16 weight buffer of shape `[head_dim]`.  Pass `None` for the
///                     no-scale variant (e.g. V-head normalization in Gemma 4).
/// * `cos_cache`     - f32 buffer of shape `[half_rope_dim]` with precomputed cosines.
/// * `sin_cache`     - f32 buffer of shape `[half_rope_dim]` with precomputed sines.
/// * `n_heads`       - Number of attention heads.
/// * `head_dim`      - Dimension of each head (e.g. 256 or 512 for Gemma 4).
/// * `half_rope_dim` - Half the rotary dimension (may be < head_dim/2 for partial rotary).
/// * `eps`           - RMS normalization epsilon (1e-6 for Gemma 4).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_head_norm_rope_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    norm_weight: Option<&MlxBuffer>,
    cos_cache: &MlxBuffer,
    sin_cache: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    half_rope_dim: u32,
    eps: f32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_head_norm_rope: n_heads and head_dim must be > 0".into(),
        ));
    }
    if half_rope_dim > head_dim / 2 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope: half_rope_dim ({}) must be <= head_dim/2 ({})",
            half_rope_dim,
            head_dim / 2,
        )));
    }

    let expected_elements = (n_heads as usize) * (head_dim as usize);
    if input.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope: input element count {} != n_heads({}) * head_dim({})",
            input.element_count(),
            n_heads,
            head_dim,
        )));
    }
    if output.element_count() != expected_elements {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope: output element count {} != n_heads({}) * head_dim({})",
            output.element_count(),
            n_heads,
            head_dim,
        )));
    }
    if cos_cache.element_count() < half_rope_dim as usize {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope: cos_cache element count {} < half_rope_dim ({})",
            cos_cache.element_count(),
            half_rope_dim,
        )));
    }
    if sin_cache.element_count() < half_rope_dim as usize {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope: sin_cache element count {} < half_rope_dim ({})",
            sin_cache.element_count(),
            half_rope_dim,
        )));
    }

    let pipeline = registry.get_pipeline("fused_head_norm_rope_bf16", device)?;

    // One threadgroup per head; size is the smallest power-of-two >= head_dim,
    // capped at 256 (Metal maximum for shared-memory reductions in this pattern).
    let tg_size = std::cmp::min(256, head_dim.next_power_of_two()) as u64;

    // Shared memory: tg_size f32 values for the parallel reduction AND for
    // caching normalized element values during the RoPE phase.
    let shared_mem_bytes = tg_size * 4; // sizeof(float) = 4

    let has_weight = norm_weight.is_some();
    let gpu_params = GpuFusedHeadNormRopeParams {
        head_dim,
        n_heads,
        half_rope_dim,
        eps,
        has_weight: u32::from(has_weight),
    };

    // When no weight buffer is provided, we still need to bind something at
    // buffer index 2 because Metal requires all declared buffers to be bound.
    // We reuse the input buffer as a harmless dummy — the shader checks
    // has_weight before reading norm_weight.
    let weight_buf = norm_weight.unwrap_or(input);

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(weight_buf)),
            (3, KernelArg::Buffer(cos_cache)),
            (4, KernelArg::Buffer(sin_cache)),
            (5, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(n_heads as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
