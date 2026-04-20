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

/// MSL source embedded at compile time (bf16).
pub static FUSED_HEAD_NORM_ROPE_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_head_norm_rope_bf16.metal");

/// MSL source embedded at compile time (f32).
pub static FUSED_HEAD_NORM_ROPE_F32_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_head_norm_rope_f32.metal");

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
///   no-scale variant (e.g. V-head normalization in Gemma 4).
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

// ---------------------------------------------------------------------------
// F32 variant — fused per-head RMS norm + NeoX RoPE with freq_factors
// ---------------------------------------------------------------------------

/// GPU params struct for f32 variant — must match `FusedHeadNormRopeF32Params`
/// in the MSL shader exactly.
///
/// `has_bf16_output` (ADR-011 Phase 3 Wave P3b.4) — when set, the kernel
/// co-writes every output element to an additional bf16 buffer at
/// buffer(6).  Used by batched prefill to fuse the f32→bf16 cast that
/// would otherwise run as a separate dispatch.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedHeadNormRopeF32Params {
    head_dim:         u32,
    n_heads:          u32,
    half_rope_dim:    u32,
    eps:              f32,
    has_weight:       u32,
    theta:            f32,
    has_freq_factors: u32,
    has_bf16_output:  u32,
    bf16_permuted:    u32, // P4.15: write bf16 at permuted layout
    seq_len:          u32, // P4.15: needed for permuted-index calc
    has_f32_perm_output: u32, // D.1: co-write f32 at permuted layout
}

/// Dispatch a fused per-head RMS norm + NeoX RoPE operation (f32).
///
/// Replaces two separate dispatches (rms_norm_f32 per-head + rope_neox_f32)
/// with a single kernel launch for Q or K.  Supports optional freq_factors
/// for Gemma 4's global attention layers.
///
/// # Arguments
///
/// * `encoder`       - Command encoder to record the dispatch into.
/// * `registry`      - Kernel registry (must have fused_head_norm_rope_f32 registered).
/// * `device`        - Metal device for pipeline compilation.
/// * `input`         - f32 buffer of shape `[n_heads * head_dim]` (one token's Q or K).
/// * `output`        - f32 output buffer, same shape as `input`.
/// * `norm_weight`   - f32 weight buffer of shape `[head_dim]`.  Pass `None` for the
///   no-scale variant (e.g. V-head unit normalization).
/// * `positions_buf` - u32 buffer of shape `[seq_len]` with position values.
/// * `freq_factors`  - Optional f32 buffer of shape `[half_rope_dim]` with per-pair
///   frequency divisors.  `None` for standard RoPE.
/// * `n_heads`       - Number of attention heads.
/// * `head_dim`      - Dimension of each head.
/// * `half_rope_dim` - Half the rotary dimension (may be < head_dim/2 for partial rotary).
/// * `eps`           - RMS normalization epsilon (1e-6 for Gemma 4).
/// * `theta`         - RoPE base frequency (e.g. 10000.0 or 1000000.0).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_head_norm_rope_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    norm_weight: Option<&MlxBuffer>,
    positions_buf: &MlxBuffer,
    freq_factors: Option<&MlxBuffer>,
    n_heads: u32,
    head_dim: u32,
    half_rope_dim: u32,
    eps: f32,
    theta: f32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_head_norm_rope_f32: n_heads and head_dim must be > 0".into(),
        ));
    }
    if half_rope_dim > head_dim / 2 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope_f32: half_rope_dim ({}) must be <= head_dim/2 ({})",
            half_rope_dim,
            head_dim / 2,
        )));
    }

    let pipeline = registry.get_pipeline("fused_head_norm_rope_f32", device)?;

    // Threadgroup must accommodate head_dim elements for shared memory caching.
    // Use next_power_of_two of head_dim, capped at 256.
    let tg_size = std::cmp::min(256, head_dim.next_power_of_two()) as u64;

    // Shared memory: head_dim floats for caching normalized values (and reduction).
    // head_dim may be > tg_size when head_dim > 256, but the kernel loops
    // with stride=tg_size so shared[i] is accessed for i < head_dim.
    // Allocate max(tg_size, head_dim) floats.
    let shared_slots = std::cmp::max(tg_size as u32, head_dim);
    let shared_mem_bytes = (shared_slots as u64) * 4;

    let has_weight = norm_weight.is_some();
    let has_ff = freq_factors.is_some();
    let gpu_params = GpuFusedHeadNormRopeF32Params {
        head_dim,
        n_heads,
        half_rope_dim,
        eps,
        has_weight: u32::from(has_weight),
        theta,
        has_freq_factors: u32::from(has_ff),
        has_bf16_output: 0,
        bf16_permuted: 0,
        seq_len: 1, // single-token decode variant
        has_f32_perm_output: 0,
    };

    // Bind dummies for optional buffers
    let weight_buf = norm_weight.unwrap_or(input);
    let ff_buf = freq_factors.unwrap_or(input);

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(weight_buf)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
            (4, KernelArg::Buffer(positions_buf)),
            (5, KernelArg::Buffer(ff_buf)),
            // buffer(6) unused when has_bf16_output=0 — bind `input` as harmless dummy.
            (6, KernelArg::Buffer(input)),
            // buffer(7) unused when has_f32_perm_output=0 — same dummy pattern.
            (7, KernelArg::Buffer(input)),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new(n_heads as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Batched fused head norm + RoPE for prefill (bf16).
///
/// Processes `seq_len` tokens at once. Input/output buffers have shape
/// `[seq_len, n_heads, head_dim]` (token-major, bf16). Launches
/// `seq_len * n_heads` threadgroups.
///
/// The kernel computes RoPE sin/cos on-the-fly from `positions_buf` and
/// `theta` (same as the f32 batch variant), so no precomputed cache buffers
/// are required.  Optionally accepts `freq_factors` for Gemma 4 global
/// attention layers.
///
/// * `input`/`output`: bf16 buffers of shape `[seq_len * n_heads * head_dim]`.
/// * `positions_buf`: u32 buffer of shape `[seq_len]`.
/// * `freq_factors`: optional f32 buffer of shape `[half_rope_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_head_norm_rope_batch_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    norm_weight: Option<&MlxBuffer>,
    positions_buf: &MlxBuffer,
    freq_factors: Option<&MlxBuffer>,
    n_heads: u32,
    head_dim: u32,
    half_rope_dim: u32,
    seq_len: u32,
    eps: f32,
    theta: f32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_head_norm_rope_batch_bf16: n_heads, head_dim, seq_len must be > 0".into(),
        ));
    }
    if half_rope_dim > head_dim / 2 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope_batch_bf16: half_rope_dim ({}) must be <= head_dim/2 ({})",
            half_rope_dim,
            head_dim / 2,
        )));
    }

    let pipeline = registry.get_pipeline("fused_head_norm_rope_batch_bf16", device)?;

    let tg_size = std::cmp::min(256, head_dim.next_power_of_two()) as u64;
    // Shared memory: max(tg_size, head_dim) f32 values for reduction + normalization.
    let shared_slots = std::cmp::max(tg_size as u32, head_dim);
    let shared_mem_bytes = (shared_slots as u64) * 4;

    let has_weight = norm_weight.is_some();
    let has_ff = freq_factors.is_some();

    // GPU params struct mirrors FusedHeadNormRopeBatchBf16Params in the shader.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuBatchBf16Params {
        head_dim:         u32,
        n_heads:          u32,
        half_rope_dim:    u32,
        eps:              f32,
        has_weight:       u32,
        theta:            f32,
        has_freq_factors: u32,
        _pad:             u32,
    }

    let gpu_params = GpuBatchBf16Params {
        head_dim,
        n_heads,
        half_rope_dim,
        eps,
        has_weight: u32::from(has_weight),
        theta,
        has_freq_factors: u32::from(has_ff),
        _pad: 0,
    };

    let weight_buf = norm_weight.unwrap_or(input);
    let ff_buf = freq_factors.unwrap_or(input);

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(weight_buf)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
            (4, KernelArg::Buffer(positions_buf)),
            (5, KernelArg::Buffer(ff_buf)),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new((n_heads as u64) * (seq_len as u64), 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Batched fused head norm + RoPE for prefill (f32).
///
/// Processes `seq_len` tokens at once. Input/output buffers have shape
/// `[seq_len, n_heads, head_dim]` (token-major). Launches
/// `seq_len * n_heads` threadgroups; the kernel's `seq_idx = head_id / n_heads`
/// formula picks the correct position from `positions_buf[seq_idx]`.
///
/// * `input`/`output`: f32 buffers of shape `[seq_len * n_heads * head_dim]`.
/// * `positions_buf`: u32 buffer of shape `[seq_len]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_head_norm_rope_batch_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    norm_weight: Option<&MlxBuffer>,
    positions_buf: &MlxBuffer,
    freq_factors: Option<&MlxBuffer>,
    n_heads: u32,
    head_dim: u32,
    half_rope_dim: u32,
    seq_len: u32,
    eps: f32,
    theta: f32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_head_norm_rope_batch_f32: n_heads, head_dim, seq_len must be > 0".into(),
        ));
    }
    if half_rope_dim > head_dim / 2 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope_batch_f32: half_rope_dim ({}) must be <= head_dim/2 ({})",
            half_rope_dim,
            head_dim / 2,
        )));
    }

    let pipeline = registry.get_pipeline("fused_head_norm_rope_f32", device)?;

    let tg_size = std::cmp::min(256, head_dim.next_power_of_two()) as u64;
    let shared_slots = std::cmp::max(tg_size as u32, head_dim);
    let shared_mem_bytes = (shared_slots as u64) * 4;

    let has_weight = norm_weight.is_some();
    let has_ff = freq_factors.is_some();
    let gpu_params = GpuFusedHeadNormRopeF32Params {
        head_dim,
        n_heads,
        half_rope_dim,
        eps,
        has_weight: u32::from(has_weight),
        theta,
        has_freq_factors: u32::from(has_ff),
        has_bf16_output: 0,
        bf16_permuted: 0,
        seq_len,
        has_f32_perm_output: 0,
    };

    let weight_buf = norm_weight.unwrap_or(input);
    let ff_buf = freq_factors.unwrap_or(input);

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(weight_buf)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
            (4, KernelArg::Buffer(positions_buf)),
            (5, KernelArg::Buffer(ff_buf)),
            // buffer(6) unused when has_bf16_output=0 — bind `input` as harmless dummy.
            (6, KernelArg::Buffer(input)),
            (7, KernelArg::Buffer(input)),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new((n_heads as u64) * (seq_len as u64), 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}

/// Batched fused head norm + RoPE for prefill (f32) with optional bf16
/// co-write — ADR-011 Phase 3 Wave P3b.4.
///
/// Same contract as `dispatch_fused_head_norm_rope_batch_f32`, plus an
/// optional `output_bf16` buffer.  When `Some(buf)`, the kernel writes
/// every output element also to `buf` at the same logical index, in
/// bf16.  This fuses the otherwise-separate `elementwise::cast_f32_to_bf16`
/// dispatch that turns pf_q_normed / pf_k_normed into their bf16
/// counterparts fed into the permute→SDPA stage.
///
/// * `output_bf16`: bf16 buffer of shape `[seq_len * n_heads * head_dim]`
///   (or `None` to match the original f32-only contract).
/// * `bf16_permuted`: P4.15 — when true, the bf16 output is written at
///   permuted layout `[n_heads, seq_len, head_dim]` (head-major) instead of
///   the natural `[seq_len, n_heads, head_dim]` (token-major).  Used to
///   absorb the post-norm `permute_021_bf16` dispatch when feeding FA's
///   head-major contract.  `output_bf16` must be sized `[seq_len * n_heads
///   * head_dim]` either way (same total element count).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_head_norm_rope_batch_f32_with_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    output_bf16: Option<&MlxBuffer>,
    norm_weight: Option<&MlxBuffer>,
    positions_buf: &MlxBuffer,
    freq_factors: Option<&MlxBuffer>,
    n_heads: u32,
    head_dim: u32,
    half_rope_dim: u32,
    seq_len: u32,
    eps: f32,
    theta: f32,
    bf16_permuted: bool,
) -> Result<()> {
    dispatch_fused_head_norm_rope_batch_f32_with_bf16_f32_perm(
        encoder, registry, device, input, output, output_bf16, None,
        norm_weight, positions_buf, freq_factors,
        n_heads, head_dim, half_rope_dim, seq_len, eps, theta,
        bf16_permuted,
    )
}

/// Extended variant that additionally accepts an optional `output_f32_perm`
/// buffer.  When `Some(buf)`, the kernel writes every f32 result element
/// also to `buf` at the permuted `[n_heads, seq_len, head_dim]` layout
/// (same index math as `bf16_permuted=true`).  Used by hf2q's HF2Q_NO_FA
/// path to skip the post-norm bf16→f32 Q cast that would otherwise run
/// as a separate `permute_021_bf16_to_f32([1,N,1])` dispatch.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_head_norm_rope_batch_f32_with_bf16_f32_perm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    output_bf16: Option<&MlxBuffer>,
    output_f32_perm: Option<&MlxBuffer>,
    norm_weight: Option<&MlxBuffer>,
    positions_buf: &MlxBuffer,
    freq_factors: Option<&MlxBuffer>,
    n_heads: u32,
    head_dim: u32,
    half_rope_dim: u32,
    seq_len: u32,
    eps: f32,
    theta: f32,
    bf16_permuted: bool,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_head_norm_rope_batch_f32_with_bf16: n_heads, head_dim, seq_len must be > 0".into(),
        ));
    }
    if half_rope_dim > head_dim / 2 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_head_norm_rope_batch_f32_with_bf16: half_rope_dim ({}) must be <= head_dim/2 ({})",
            half_rope_dim,
            head_dim / 2,
        )));
    }
    if let Some(buf) = output_bf16 {
        let expected = (seq_len as usize) * (n_heads as usize) * (head_dim as usize);
        if buf.element_count() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_head_norm_rope_batch_f32_with_bf16: output_bf16 element count {} < expected {}",
                buf.element_count(), expected
            )));
        }
    }
    if let Some(buf) = output_f32_perm {
        let expected = (seq_len as usize) * (n_heads as usize) * (head_dim as usize);
        if buf.element_count() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_head_norm_rope_batch_f32_with_bf16_f32_perm: output_f32_perm element count {} < expected {}",
                buf.element_count(), expected
            )));
        }
    }

    let pipeline = registry.get_pipeline("fused_head_norm_rope_f32", device)?;

    let tg_size = std::cmp::min(256, head_dim.next_power_of_two()) as u64;
    let shared_slots = std::cmp::max(tg_size as u32, head_dim);
    let shared_mem_bytes = (shared_slots as u64) * 4;

    let has_weight = norm_weight.is_some();
    let has_ff = freq_factors.is_some();
    let has_bf16 = output_bf16.is_some();
    let has_f32_perm = output_f32_perm.is_some();
    let gpu_params = GpuFusedHeadNormRopeF32Params {
        head_dim,
        n_heads,
        half_rope_dim,
        eps,
        has_weight: u32::from(has_weight),
        theta,
        has_freq_factors: u32::from(has_ff),
        has_bf16_output: u32::from(has_bf16),
        bf16_permuted: u32::from(bf16_permuted && (has_bf16 || has_f32_perm)),
        seq_len,
        has_f32_perm_output: u32::from(has_f32_perm),
    };

    let weight_buf = norm_weight.unwrap_or(input);
    let ff_buf = freq_factors.unwrap_or(input);
    let bf16_buf = output_bf16.unwrap_or(input);
    let f32_perm_buf = output_f32_perm.unwrap_or(input);

    encode_threadgroups_with_args_and_shared(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Buffer(weight_buf)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
            (4, KernelArg::Buffer(positions_buf)),
            (5, KernelArg::Buffer(ff_buf)),
            (6, KernelArg::Buffer(bf16_buf)),
            (7, KernelArg::Buffer(f32_perm_buf)),
        ],
        &[(0, shared_mem_bytes)],
        MTLSize::new((n_heads as u64) * (seq_len as u64), 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(())
}
