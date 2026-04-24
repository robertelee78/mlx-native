//! Hadamard-quantize KV cache kernel dispatch (ADR-007 Phase 1.1).
//!
//! Replaces `kv_cache_copy_batch_f32_to_f16` with a fused kernel that
//! applies a Fast Walsh-Hadamard Transform, extracts the L2 norm, and
//! quantizes each coordinate using the 4-bit Lloyd-Max codebook before
//! packing the indices as nibbles into the output buffer.
//!
//! Output format per head per token:
//! - `packed`: `[num_kv_heads, cache_capacity, head_dim/2]` u8 — nibble-packed 4-bit indices
//! - `norms`:
//!   - D=256: `[num_kv_heads, cache_capacity]` f32 — 1 norm per position (NORMS_PER_POS=1)
//!   - D=512: `[num_kv_heads, cache_capacity, 2]` f32 — 2 per-block norms per position
//!     (NORMS_PER_POS=2), per AmesianX cpy-utils.cuh:241-269 (ADR-007 iter-15 per-block norm).
//!
//! `norms_per_pos(head_dim)` = `head_dim / 256`. Callers must allocate norms buffers
//! with `num_kv_heads * cache_capacity * norms_per_pos(head_dim)` f32 elements.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{encode_threadgroups_with_args_and_shared, KernelArg};

/// MSL source for the `hadamard_quantize_kv` kernel (embedded at compile time).
pub static HADAMARD_QUANTIZE_KV_SHADER_SOURCE: &str =
    include_str!("../shaders/hadamard_quantize_kv.metal");

/// Register the `hadamard_quantize_kv` shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("hadamard_quantize_kv", HADAMARD_QUANTIZE_KV_SHADER_SOURCE);
}

/// Parameters struct matching the `HadamardQuantizeParams` in the Metal shader.
///
/// `repr(C)` + `bytemuck::Pod` ensures the struct can be passed directly via
/// `set_bytes` without any marshalling.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HadamardQuantizeParams {
    head_dim: u32,
    num_kv_heads: u32,
    write_pos: u32,
    cache_capacity: u32,
    is_sliding: u32,
    /// iter-18 S2B: D=512 per-block scale factor (ablation via HF2Q_SCALE_FORMULA).
    /// bare=1.0 (control), sqrt256=16.0, sqrt512≈22.627. D=256 path ignores this.
    scale_factor_d512: f32,
    /// iter-18 S2A: post-scale RMS probe flag (1=enabled, 0=disabled).
    rms_probe_enabled: u32,
}

/// Dispatch the fused Hadamard-quantize KV kernel on the GPU.
///
/// For each KV head vector (length `head_dim`) in the source:
/// 1. Applies in-place normalized FWHT (butterfly, in shared memory).
/// 2. Extracts the L2 norm of the rotated vector.
/// 3. Normalizes to unit sphere, then scales to N(0,1) domain.
/// 4. Finds the nearest 4-bit Lloyd-Max centroid for every coordinate.
/// 5. Packs pairs of 4-bit indices as nibbles into `packed`.
/// 6. Writes the L2 norm scalar to `norms`.
///
/// # Arguments
///
/// * `encoder`          — Command encoder to record the dispatch into.
/// * `registry`         — Kernel registry (must have `hadamard_quantize_kv` registered).
/// * `device`           — Metal device for pipeline compilation.
/// * `src`              — F32 buffer of shape `[num_kv_heads, head_dim]` (one token, all heads).
/// * `packed`           — u8 buffer of shape `[num_kv_heads, cache_capacity, head_dim/2]`.
/// * `norms`            — F32 buffer of shape `[num_kv_heads, cache_capacity]`.
/// * `num_kv_heads`     — Number of KV heads (threadgroups dispatched).
/// * `head_dim`         — Elements per head.  Must be a power of two in `[4, 4096]`.
/// * `cache_capacity`   — Cache capacity (ring buffer size for sliding, max_seq_len for global).
/// * `write_pos`        — Write position in cache (the kernel applies modulo for sliding window).
/// * `is_sliding`       — If `true`, `write_pos` is wrapped modulo `cache_capacity`.
/// * `scale_factor_d512`— iter-18 S2B: D=512 per-block scale factor (1.0=bare, 16.0=sqrt256,
///                        22.627=sqrt512). Pass `None` to use 1.0 (bare, iter-16 control).
/// * `rms_scratch`      — iter-18 S2A: optional scratch buffer for post-scale RMS probe.
///                        Layout: `[num_kv_heads, norms_per_pos, 16]` f32.  Pass `None` to disable.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - `head_dim` is not a power of two.
/// - `head_dim` is larger than 4096 (would exceed Metal 32 KB threadgroup limit at 2× float).
/// - `head_dim` is odd (nibble packing requires even count).
/// - Source buffer is smaller than `num_kv_heads * head_dim` f32 elements.
/// - `packed` buffer is smaller than `num_kv_heads * cache_capacity * head_dim/2` bytes.
/// - `norms` buffer is smaller than `num_kv_heads * cache_capacity` f32 elements.
/// - For global (non-sliding) caches: `write_pos >= cache_capacity`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hadamard_quantize_kv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    packed: &MlxBuffer,
    norms: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos: u32,
    is_sliding: bool,
    scale_factor_d512: Option<f32>,
    rms_scratch: Option<&MlxBuffer>,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    // head_dim must be a power of two for the butterfly pattern.
    if !head_dim.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv: head_dim must be a power of two, got {}",
            head_dim
        )));
    }

    // Shared memory: 2 * head_dim floats (data region + norm reduction scratch).
    // 2 * head_dim * 4 bytes <= 32768  =>  head_dim <= 4096.
    if head_dim > 4096 {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv: head_dim {} exceeds Metal 32 KB threadgroup limit \
             (max 4096 for 2x f32 shared memory)",
            head_dim
        )));
    }

    // Nibble packing requires an even head_dim (always true for powers of two >= 2).
    if head_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv: head_dim must be even for nibble packing, got {}",
            head_dim
        )));
    }

    // For global (non-sliding) cache, write_pos must be within bounds.
    if !is_sliding && write_pos >= cache_capacity {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv: global cache write_pos({}) >= cache_capacity({})",
            write_pos, cache_capacity
        )));
    }

    // Validate source buffer size.
    let required_src = (num_kv_heads as u64) * (head_dim as u64);
    if (src.element_count() as u64) < required_src {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv: src has {} elements but need {} \
             (num_kv_heads={} * head_dim={})",
            src.element_count(),
            required_src,
            num_kv_heads,
            head_dim,
        )));
    }

    // Validate packed buffer size (in bytes).
    let required_packed_bytes =
        (num_kv_heads as u64) * (cache_capacity as u64) * (head_dim as u64 / 2);
    if (packed.byte_len() as u64) < required_packed_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv: packed buffer has {} bytes but need {} \
             (num_kv_heads={} * cache_capacity={} * head_dim/2={})",
            packed.byte_len(),
            required_packed_bytes,
            num_kv_heads,
            cache_capacity,
            head_dim / 2,
        )));
    }

    // Validate norms buffer size.
    // D=256: 1 norm per position (NORMS_PER_POS=1).
    // D=512: 2 norms per position (NORMS_PER_POS=2), per AmesianX cpy-utils.cuh:241-269.
    let norms_per_pos = (head_dim / 256).max(1) as u64;
    let required_norms = (num_kv_heads as u64) * (cache_capacity as u64) * norms_per_pos;
    if (norms.element_count() as u64) < required_norms {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv: norms buffer has {} elements but need {} \
             (num_kv_heads={} * cache_capacity={} * norms_per_pos={})",
            norms.element_count(),
            required_norms,
            num_kv_heads,
            cache_capacity,
            norms_per_pos,
        )));
    }

    // Use the fast SIMD-shuffle kernel (zero threadgroup barriers).
    let kernel_name = match head_dim {
        256 => "hadamard_quantize_kv_fast_d256",
        512 => "hadamard_quantize_kv_fast_d512",
        _ => "hadamard_quantize_kv", // fallback to shared-memory version
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let effective_scale = scale_factor_d512.unwrap_or(1.0_f32);
    let probe_enabled = rms_scratch.is_some() as u32;
    let params = HadamardQuantizeParams {
        head_dim,
        num_kv_heads,
        write_pos,
        cache_capacity,
        is_sliding: if is_sliding { 1 } else { 0 },
        scale_factor_d512: effective_scale,
        rms_probe_enabled: probe_enabled,
    };
    let params_bytes = bytemuck::bytes_of(&params);

    if kernel_name.starts_with("hadamard_quantize_kv_fast") {
        // Fast kernel: 1 simdgroup (32 threads) per head, no shared memory.
        use super::encode_helpers::{encode_threadgroups_with_args, KernelArg as KA};
        // Scratch buffer at slot 4: bind real buffer if probe enabled, otherwise a dummy
        // (Metal requires a bound buffer even if the kernel won't write it).
        // We use the norms buffer as the dummy — the kernel only writes scratch when
        // rms_probe_enabled!=0, so the dummy binding is never written.
        let scratch_binding = rms_scratch.unwrap_or(norms);
        encode_threadgroups_with_args(
            encoder,
            pipeline,
            &[
                (0, KA::Buffer(src)),
                (1, KA::Buffer(packed)),
                (2, KA::Buffer(norms)),
                (3, KA::Bytes(params_bytes)),
                (4, KA::Buffer(scratch_binding)),
            ],
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(32, 1, 1), // 1 simdgroup
        );
    } else {
        // Fallback: shared-memory version for non-256/512 head_dim.
        let shared_mem_bytes = 2u64 * (head_dim as u64) * 4;
        encode_threadgroups_with_args_and_shared(
            encoder,
            pipeline,
            &[
                (0, KernelArg::Buffer(src)),
                (1, KernelArg::Buffer(packed)),
                (2, KernelArg::Buffer(norms)),
                (3, KernelArg::Bytes(params_bytes)),
            ],
            &[(0, shared_mem_bytes)],
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
    }

    Ok(())
}

/// Dispatch the Hadamard-quantize KV kernel over a sequence of tokens.
///
/// Wraps the single-token [`dispatch_hadamard_quantize_kv`] to populate
/// the TQ-packed cache for `n_tokens` consecutive positions from a batched
/// source buffer. The source buffer is laid out as
/// `[total_src_tokens, num_kv_heads, head_dim]` F32; this function iterates
/// the leading dimension starting at `src_tok_offset` and re-dispatches
/// the single-token kernel with a buffer byte offset, so the cleared kernel
/// source is untouched.
///
/// Cache positions written: `[write_pos_start, write_pos_start + n_tokens)`
/// (wrapped modulo `cache_capacity` when `is_sliding` is true).
///
/// # Arguments
///
/// * `src`             — F32 buffer `[total_src_tokens, num_kv_heads, head_dim]`.
/// * `packed`          — Output packed buffer (same layout as single-token).
/// * `norms`           — Output norms buffer (same layout as single-token).
/// * `write_pos_start` — First cache position to write.
/// * `n_tokens`        — How many consecutive positions to write.
/// * `src_tok_offset`  — Starting token index in `src` (matches the
///   batched dense-copy semantics; use `seq_len - n_tokens` when
///   sliding and the prefill has already exceeded the window).
///
/// # Performance notes
///
/// Correctness-first implementation: at pp2455 with 30 layers and the
/// Gemma-4 sliding/global layer split this issues on the order of
/// 147k kernel launches per prefill. If that is ever measured to be
/// the bottleneck, promote to a dedicated bulk shader with a 2-D
/// dispatch grid — this wrapper intentionally does not modify the
/// cleared single-token kernel source, so both variants remain
/// byte-identical in their math.
///
/// # Errors
///
/// Propagates any [`dispatch_hadamard_quantize_kv`] error encountered
/// on the per-position dispatches and adds one extra validation:
/// `src` must have at least `n_tokens * num_kv_heads * head_dim`
/// F32 elements.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hadamard_quantize_kv_seq(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    packed: &MlxBuffer,
    norms: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
    is_sliding: bool,
    scale_factor_d512: Option<f32>,
) -> Result<()> {
    if n_tokens == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    // Src must cover [src_tok_offset, src_tok_offset + n_tokens) slices.
    let required_src =
        (src_tok_offset as u64 + n_tokens as u64) * (num_kv_heads as u64) * (head_dim as u64);
    if (src.element_count() as u64) < required_src {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_seq: src has {} elements but need {} \
             (src_tok_offset={} + n_tokens={} * num_kv_heads={} * head_dim={})",
            src.element_count(),
            required_src,
            src_tok_offset,
            n_tokens,
            num_kv_heads,
            head_dim,
        )));
    }

    // Pre-shared setup for the per-position dispatches. The kernel name
    // and pipeline only depend on `head_dim`, so resolve once.
    if !head_dim.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_seq: head_dim must be a power of two, got {}",
            head_dim
        )));
    }
    if head_dim > 4096 {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_seq: head_dim {} exceeds Metal 32 KB threadgroup limit",
            head_dim
        )));
    }
    if head_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_seq: head_dim must be even for nibble packing, got {}",
            head_dim
        )));
    }

    let kernel_name = match head_dim {
        256 => "hadamard_quantize_kv_fast_d256",
        512 => "hadamard_quantize_kv_fast_d512",
        _ => "hadamard_quantize_kv",
    };
    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let bytes_per_token = (num_kv_heads as u64) * (head_dim as u64) * 4; // f32

    for i in 0..n_tokens {
        let write_pos = write_pos_start + i;

        if !is_sliding && write_pos >= cache_capacity {
            return Err(MlxError::InvalidArgument(format!(
                "hadamard_quantize_kv_seq: global cache write_pos({}) >= cache_capacity({}) at seq idx {}",
                write_pos, cache_capacity, i
            )));
        }

        let effective_scale = scale_factor_d512.unwrap_or(1.0_f32);
        let params = HadamardQuantizeParams {
            head_dim,
            num_kv_heads,
            write_pos,
            cache_capacity,
            is_sliding: if is_sliding { 1 } else { 0 },
            scale_factor_d512: effective_scale,
            rms_probe_enabled: 0, // probe not supported in bulk seq dispatch
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let src_offset = ((src_tok_offset + i) as u64) * bytes_per_token;

        if kernel_name.starts_with("hadamard_quantize_kv_fast") {
            use super::encode_helpers::encode_threadgroups_with_args;
            encode_threadgroups_with_args(
                encoder,
                pipeline,
                &[
                    (0, KernelArg::BufferWithOffset(src, src_offset)),
                    (1, KernelArg::Buffer(packed)),
                    (2, KernelArg::Buffer(norms)),
                    (3, KernelArg::Bytes(params_bytes)),
                    (4, KernelArg::Buffer(norms)), // dummy slot 4 (probe disabled)
                ],
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
        } else {
            let shared_mem_bytes = 2u64 * (head_dim as u64) * 4;
            encode_threadgroups_with_args_and_shared(
                encoder,
                pipeline,
                &[
                    (0, KernelArg::BufferWithOffset(src, src_offset)),
                    (1, KernelArg::Buffer(packed)),
                    (2, KernelArg::Buffer(norms)),
                    (3, KernelArg::Bytes(params_bytes)),
                ],
                &[(0, shared_mem_bytes)],
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(head_dim as u64, 1, 1),
            );
        }
    }

    Ok(())
}

// ============================================================================
// Track B (iter-21): higher-bit dispatch (5-bit or 6-bit, byte-packed).
// ============================================================================

/// GPU-side params for the higher-bit quantize kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HadamardQuantizeHbParams {
    head_dim: u32,
    num_kv_heads: u32,
    write_pos: u32,
    cache_capacity: u32,
    is_sliding: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,  // 5 or 6
}

/// Dispatch the higher-bit Hadamard-quantize KV kernel.
///
/// Same pipeline as 4-bit (FWHT + norm) but writes 1 byte per element
/// (byte-packed) using 5-bit (32 centroids), 6-bit (64 centroids), or
/// 8-bit (256 centroids) codebook. (iter-24 adds 8-bit support)
///
/// * `packed` must be `[num_kv_heads, cache_capacity, head_dim]` u8 (byte-packed).
/// * `norms` layout is identical to 4-bit path.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hadamard_quantize_kv_hb(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    packed: &MlxBuffer,      // byte-packed: [nkv, capacity, head_dim] u8
    norms: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos: u32,
    is_sliding: bool,
    scale_factor_d512: f32,
    codebook_bits: u32,      // 5 or 6
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 { return Ok(()); }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_hadamard_quantize_kv_hb: codebook_bits must be 5, 6, or 8, got {}", codebook_bits)));
    }

    let kernel_name = match head_dim {
        256 => "hadamard_quantize_kv_hb_d256",
        512 => "hadamard_quantize_kv_hb_d512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_hb: head_dim {} not supported (need 256 or 512)", head_dim))),
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let params = HadamardQuantizeHbParams {
        head_dim,
        num_kv_heads,
        write_pos,
        cache_capacity,
        is_sliding: if is_sliding { 1 } else { 0 },
        scale_factor_d512,
        codebook_bits,
    };
    let params_bytes = bytemuck::bytes_of(&params);

    use super::encode_helpers::{encode_threadgroups_with_args, KernelArg as KA};
    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KA::Buffer(src)),
            (1, KA::Buffer(packed)),
            (2, KA::Buffer(norms)),
            (3, KA::Bytes(params_bytes)),
        ],
        MTLSize::new(num_kv_heads as u64, 1, 1),
        MTLSize::new(32, 1, 1), // 1 simdgroup (32 threads)
    );

    Ok(())
}
