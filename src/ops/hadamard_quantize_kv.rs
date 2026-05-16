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
//!     (NORMS_PER_POS=2), per AmesianX cpy-utils.cuh:241-269 (ADR-007 per-block norm).
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
    /// D=512 per-block scale factor (ablation via HF2Q_SCALE_FORMULA).
    /// bare=1.0 (control), sqrt256=16.0, sqrt512≈22.627. D=256 path ignores this.
    scale_factor_d512: f32,
    /// Post-scale RMS probe flag (1=enabled, 0=disabled).
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
/// * `scale_factor_d512`— D=512 per-block scale factor (1.0=bare, 16.0=sqrt256,
///                        22.627=sqrt512). Pass `None` to use 1.0 (bare control).
/// * `rms_scratch`      — Optional scratch buffer for post-scale RMS probe.
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
// ADR-028 (Phase 7d / H4): fused K+V single-position 4-bit dispatch.
// ============================================================================

/// Dispatch the fused 4-bit Hadamard-quantize KV kernel.
///
/// Combines TWO consecutive `dispatch_hadamard_quantize_kv` calls (K then V
/// into the F32 shadow TQ-packed cache) into a single Metal dispatch via the
/// Z-dim split (`tgpig.z=0` → K stream, `tgpig.z=1` → V stream).
///
/// Saves one Apple Metal kernel-launch floor (~14 µs) per layer per decode
/// token. At gemma4 30 layers this drops 60→30 KV-write dispatches/decode-
/// token (~0.4 ms/token, ~3% theoretical). Result is byte-identical to the
/// 2-dispatch sequence at identical params — verified by
/// `test_hadamard_quantize_kv_fast_dual_byte_identity_d256`.
///
/// The RMS scratch probe path (HF2Q_DEBUG_TQ_RMS) is NOT supported by the
/// fused variant; it routes through the unmodified single-stream kernel.
///
/// * `src_k`, `src_v` — F32 `[num_kv_heads, head_dim]` per stream.
/// * `packed_k`, `packed_v` — u8 nibble-packed `[num_kv_heads, cache_capacity, head_dim/2]`.
/// * `norms_k`, `norms_v` — F32 `[num_kv_heads, cache_capacity (* norms_per_pos for d=512)]`.
/// * Other params mirror `dispatch_hadamard_quantize_kv` exactly.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hadamard_quantize_kv_fast_dual(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src_k: &MlxBuffer,
    src_v: &MlxBuffer,
    packed_k: &MlxBuffer,
    packed_v: &MlxBuffer,
    norms_k: &MlxBuffer,
    norms_v: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos: u32,
    is_sliding: bool,
    scale_factor_d512: Option<f32>,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    let kernel_name = match head_dim {
        256 => "hadamard_quantize_kv_fast_dual_d256",
        512 => "hadamard_quantize_kv_fast_dual_d512",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "hadamard_quantize_kv_fast_dual: head_dim {} not supported (need 256 or 512)",
                head_dim
            )));
        }
    };

    if !is_sliding && write_pos >= cache_capacity {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_fast_dual: global cache write_pos({}) >= cache_capacity({})",
            write_pos, cache_capacity
        )));
    }

    let required_src = (num_kv_heads as u64) * (head_dim as u64);
    if (src_k.element_count() as u64) < required_src {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_fast_dual: src_k has {} elements but need {}",
            src_k.element_count(), required_src
        )));
    }
    if (src_v.element_count() as u64) < required_src {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_fast_dual: src_v has {} elements but need {}",
            src_v.element_count(), required_src
        )));
    }

    let required_packed_bytes =
        (num_kv_heads as u64) * (cache_capacity as u64) * (head_dim as u64 / 2);
    if (packed_k.byte_len() as u64) < required_packed_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_fast_dual: packed_k has {} bytes but need {}",
            packed_k.byte_len(), required_packed_bytes
        )));
    }
    if (packed_v.byte_len() as u64) < required_packed_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_fast_dual: packed_v has {} bytes but need {}",
            packed_v.byte_len(), required_packed_bytes
        )));
    }

    let norms_per_pos = (head_dim / 256).max(1) as u64;
    let required_norms = (num_kv_heads as u64) * (cache_capacity as u64) * norms_per_pos;
    if (norms_k.element_count() as u64) < required_norms {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_fast_dual: norms_k has {} elements but need {}",
            norms_k.element_count(), required_norms
        )));
    }
    if (norms_v.element_count() as u64) < required_norms {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_fast_dual: norms_v has {} elements but need {}",
            norms_v.element_count(), required_norms
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let params = HadamardQuantizeParams {
        head_dim,
        num_kv_heads,
        write_pos,
        cache_capacity,
        is_sliding: if is_sliding { 1 } else { 0 },
        scale_factor_d512: scale_factor_d512.unwrap_or(1.0_f32),
        rms_probe_enabled: 0, // probe not supported in fused variant
    };
    let params_bytes = bytemuck::bytes_of(&params);

    use super::encode_helpers::{encode_threadgroups_with_args, KernelArg as KA};
    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KA::Buffer(src_k)),
            (1, KA::Buffer(src_v)),
            (2, KA::Buffer(packed_k)),
            (3, KA::Buffer(packed_v)),
            (4, KA::Buffer(norms_k)),
            (5, KA::Buffer(norms_v)),
            (6, KA::Bytes(params_bytes)),
        ],
        MTLSize::new(num_kv_heads as u64, 1, 2), // x=heads, z=K|V stream
        MTLSize::new(32, 1, 1),                  // 1 simdgroup
    );

    Ok(())
}

// ============================================================================
// Track B: higher-bit dispatch (5-bit or 6-bit, byte-packed).
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
/// 8-bit (256 centroids) codebook.
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

/// ADR-028 Phase 10e.5: no-FWHT V quantize for the hybrid path.
///
/// Same byte-packed Lloyd-Max output (5/6/8-bit) and same norm storage layout
/// as `dispatch_hadamard_quantize_kv_hb`, but skips the Hadamard rotation so
/// the SDPA dequant recovers raw V values (not FWHT-rotated).  Combined with
/// hybrid F16-K, this lets the SDPA dispatcher in hf2q drop BOTH the
/// `fwht_sign_premult` (Q) and `fwht_sign_undo` (output) dispatches per layer
/// — saves 60 dispatches/decode-token at gemma4 30L on top of the K-side
/// codebook elimination.
///
/// V-only by design — the hybrid path stores K as F16 dense, only V needs
/// quantization.  K-side encoder is `kv_cache_copy_batch_f32_to_f16` (already
/// in mlx-native).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_quantize_v_no_fwht(
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
    codebook_bits: u32,      // 5, 6, or 8
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 { return Ok(()); }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_kv_quantize_v_no_fwht: codebook_bits must be 5, 6, or 8, got {}",
            codebook_bits)));
    }

    let kernel_name = match head_dim {
        256 => "kv_quantize_v_no_fwht_d256",
        512 => "kv_quantize_v_no_fwht_d512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "kv_quantize_v_no_fwht: head_dim {} not supported (need 256 or 512)", head_dim))),
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

/// ADR-028 Phase 10c.5: fused F16-K-copy + V-no-FWHT-encode for the
/// hybrid path.
///
/// Combines the two hf2q hybrid-path decode dispatches into a single dispatch
/// via grid Z-dim:
///   * z=0 K stream: F32 src_k → F16 cache (mirrors `dispatch_kv_cache_copy_batch_f32_to_f16`)
///   * z=1 V stream: F32 src_v → byte-packed Lloyd-Max + L2 norm
///                    (mirrors `dispatch_kv_quantize_v_no_fwht`)
///
/// Result is byte-identical to the two stand-alone calls at identical params;
/// each stream takes the SAME math path as its stand-alone counterpart.
///
/// Saves one Apple Metal kernel-launch floor (~14 µs) per layer per decode
/// token.  At gemma4 30L: drops 60 → 30 KV-write dispatches/decode-token,
/// expected ~+1% decode (measured dispatch-floor savings).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_copy_kf16_quantize_v_no_fwht(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src_k: &MlxBuffer,
    src_v: &MlxBuffer,
    cache_k: &MlxBuffer,    // F16 cache
    packed_v: &MlxBuffer,   // U8 byte-packed
    norms_v: &MlxBuffer,    // F32 norms
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos: u32,
    is_sliding: bool,
    scale_factor_d512: f32,
    codebook_bits: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 { return Ok(()); }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_kv_copy_kf16_quantize_v_no_fwht: codebook_bits must be 5, 6, or 8, got {}",
            codebook_bits)));
    }
    if cache_k.dtype() != crate::DType::F16 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_kv_copy_kf16_quantize_v_no_fwht: cache_k must be DType::F16, got {:?}",
            cache_k.dtype())));
    }

    let kernel_name = match head_dim {
        256 => "kv_copy_kf16_quantize_v_no_fwht_d256",
        512 => "kv_copy_kf16_quantize_v_no_fwht_d512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "kv_copy_kf16_quantize_v_no_fwht: head_dim {} not supported (need 256 or 512)",
            head_dim))),
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
            (0, KA::Buffer(src_k)),
            (1, KA::Buffer(src_v)),
            (2, KA::Buffer(cache_k)),
            (3, KA::Buffer(packed_v)),
            (4, KA::Buffer(norms_v)),
            (5, KA::Bytes(params_bytes)),
        ],
        // Grid: (num_kv_heads, 1, 2) — Z=2 for K + V streams.
        MTLSize::new(num_kv_heads as u64, 1, 2),
        // Threadgroup: (32, 1, 1) — single simdgroup per stream.
        MTLSize::new(32, 1, 1),
    );

    Ok(())
}

/// ADR-028: fused K+V single-position Hadamard-quantize KV HB encoder.
///
/// Combines two `dispatch_hadamard_quantize_kv_hb` calls (one for K, one for V)
/// into a single dispatch via grid Z-dim. Saves one Apple Metal kernel-launch
/// floor (~14 µs) per layer per decode token. At gemma4 30 layers, drops
/// 60→30 HB-encode dispatches/decode-token, saving ~0.4 ms/token (~3% decode).
///
/// Result is byte-identical to two `dispatch_hadamard_quantize_kv_hb` calls
/// at identical params (verified by mlx-native unit test).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_hadamard_quantize_kv_hb_dual(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src_k: &MlxBuffer,
    src_v: &MlxBuffer,
    packed_k: &MlxBuffer,
    packed_v: &MlxBuffer,
    norms_k: &MlxBuffer,
    norms_v: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    write_pos: u32,
    is_sliding: bool,
    scale_factor_d512: f32,
    codebook_bits: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 { return Ok(()); }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_hadamard_quantize_kv_hb_dual: codebook_bits must be 5, 6, or 8, got {}", codebook_bits)));
    }

    let kernel_name = match head_dim {
        256 => "hadamard_quantize_kv_hb_dual_d256",
        512 => "hadamard_quantize_kv_hb_dual_d512",
        _ => return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_hb_dual: head_dim {} not supported (need 256 or 512)", head_dim))),
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
            (0, KA::Buffer(src_k)),
            (1, KA::Buffer(src_v)),
            (2, KA::Buffer(packed_k)),
            (3, KA::Buffer(packed_v)),
            (4, KA::Buffer(norms_k)),
            (5, KA::Buffer(norms_v)),
            (6, KA::Bytes(params_bytes)),
        ],
        MTLSize::new(num_kv_heads as u64, 1, 2), // x=heads, z=K|V stream
        MTLSize::new(32, 1, 1),                  // 1 simdgroup (32 threads)
    );

    Ok(())
}

/// ADR-027 Phase B — multi-token Hadamard-quantize KV (HB) dispatch.
///
/// Mirrors `dispatch_hadamard_quantize_kv_seq` (4-bit) but writes 1 byte
/// per element via the 5/6/8-bit codebook used by `flash_attn_vec_tq_hb`.
/// Walks `n_tokens` positions starting at `write_pos_start`, dispatching
/// the per-position HB encode kernel with successive `src_offset`
/// values (via `KernelArg::BufferWithOffset`).
///
/// Used by qwen35's prefill TQ encode loop in `gpu_full_attn::full_attn_
/// layer_gpu` (ADR-027 Phase B) to populate `FullAttnKvSlot.tq`
/// at all prefill positions before the first decode SDPA reads them.
///
/// # Arguments
///
/// - `src`: F32 buffer holding ≥ `(src_tok_offset + n_tokens) ×
///   num_kv_heads × head_dim` elements (multi-token K or V, seq-major
///   layout `[seq_len, num_kv_heads, head_dim]`).
/// - `packed`: U8 destination, `[num_kv_heads, cache_capacity, head_dim]`.
/// - `norms`: F32 destination, `[num_kv_heads, cache_capacity,
///   norms_per_pos]`.
/// - `n_tokens`: number of source tokens to encode.
/// - `src_tok_offset`: index in `src` at which to begin reading
///   (allows callers to encode a sub-range).
/// - `write_pos_start`: cache slot index of the first encoded token.
///
/// # Errors
///
/// - `n_tokens == 0` → no-op (returns `Ok(())`; mirrors the 4-bit `_seq`).
/// - `src` too small to cover `[src_tok_offset, src_tok_offset + n_tokens)`.
/// - `head_dim` not in {256, 512}.
/// - `codebook_bits` not in {5, 6, 8}.
/// - For non-sliding caches: `write_pos_start + n_tokens > cache_capacity`
///   detected per-position (matches the 4-bit `_seq` semantics).
///
/// # Performance notes
///
/// Correctness-first: dispatches one kernel launch per token. At pp2455
/// with 30 layers this is on the order of 147k launches per prefill
/// (mirrors the 4-bit `_seq` rationale). Promote to a 2-D dispatch
/// shader if measured to be the bottleneck.
#[allow(clippy::too_many_arguments)]
/// ADR-028 Phase 10e.5: no-FWHT V seq variant for batched prefill.
///
/// Dispatches `kv_quantize_v_no_fwht_d{256,512}` once per token in `[write_pos_start
/// .. write_pos_start+n_tokens)` from `src + src_tok_offset` rows.  Mirrors
/// `dispatch_hadamard_quantize_kv_hb_seq` exactly except the underlying kernel
/// is the no-FWHT variant.
///
/// Required so the batched-prefill V-encode and decode V-encode produce
/// CONSISTENT byte layout — without this, prefill stores FWHT-rotated V and
/// decode stores raw V, the SDPA dequant reads mixed-domain bytes, and output
/// is garbage.  Phase 10c established the V-encode site routing; Phase 10e.5
/// makes both sides use the no-FWHT path.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_quantize_v_no_fwht_seq(
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
    scale_factor_d512: f32,
    codebook_bits: u32,
) -> Result<()> {
    if n_tokens == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_kv_quantize_v_no_fwht_seq: codebook_bits must be \
             5, 6, or 8, got {}",
            codebook_bits
        )));
    }
    let kernel_name = match head_dim {
        256 => "kv_quantize_v_no_fwht_d256",
        512 => "kv_quantize_v_no_fwht_d512",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "kv_quantize_v_no_fwht_seq: head_dim {} not supported \
                 (need 256 or 512)",
                head_dim
            )))
        }
    };

    let required_src = (src_tok_offset as u64 + n_tokens as u64)
        * (num_kv_heads as u64)
        * (head_dim as u64);
    if (src.element_count() as u64) < required_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_quantize_v_no_fwht_seq: src has {} elements but need {} \
             (src_tok_offset={} + n_tokens={} * num_kv_heads={} * head_dim={})",
            src.element_count(), required_src,
            src_tok_offset, n_tokens, num_kv_heads, head_dim,
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let bytes_per_token = (num_kv_heads as u64) * (head_dim as u64) * 4;

    use super::encode_helpers::{encode_threadgroups_with_args, KernelArg as KA};
    for i in 0..n_tokens {
        let write_pos = write_pos_start + i;
        if !is_sliding && write_pos >= cache_capacity {
            return Err(MlxError::InvalidArgument(format!(
                "kv_quantize_v_no_fwht_seq: global cache write_pos({}) >= \
                 cache_capacity({}) at seq idx {}",
                write_pos, cache_capacity, i
            )));
        }
        let params = HadamardQuantizeHbParams {
            head_dim, num_kv_heads, write_pos, cache_capacity,
            is_sliding: if is_sliding { 1 } else { 0 },
            scale_factor_d512, codebook_bits,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let src_offset = ((src_tok_offset + i) as u64) * bytes_per_token;

        encode_threadgroups_with_args(
            encoder, pipeline,
            &[
                (0, KA::BufferWithOffset(src, src_offset)),
                (1, KA::Buffer(packed)),
                (2, KA::Buffer(norms)),
                (3, KA::Bytes(params_bytes)),
            ],
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    Ok(())
}

pub fn dispatch_hadamard_quantize_kv_hb_seq(
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
    scale_factor_d512: f32,
    codebook_bits: u32,
) -> Result<()> {
    if n_tokens == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    if !matches!(codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_hadamard_quantize_kv_hb_seq: codebook_bits must be \
             5, 6, or 8, got {}",
            codebook_bits
        )));
    }
    let kernel_name = match head_dim {
        256 => "hadamard_quantize_kv_hb_d256",
        512 => "hadamard_quantize_kv_hb_d512",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "hadamard_quantize_kv_hb_seq: head_dim {} not supported \
                 (need 256 or 512)",
                head_dim
            )))
        }
    };

    // Validate src has enough bytes to cover the requested slice.
    let required_src = (src_tok_offset as u64 + n_tokens as u64)
        * (num_kv_heads as u64)
        * (head_dim as u64);
    if (src.element_count() as u64) < required_src {
        return Err(MlxError::InvalidArgument(format!(
            "hadamard_quantize_kv_hb_seq: src has {} elements but need {} \
             (src_tok_offset={} + n_tokens={} * num_kv_heads={} * head_dim={})",
            src.element_count(),
            required_src,
            src_tok_offset,
            n_tokens,
            num_kv_heads,
            head_dim,
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, device)?;
    let bytes_per_token = (num_kv_heads as u64) * (head_dim as u64) * 4; // f32

    use super::encode_helpers::{encode_threadgroups_with_args, KernelArg as KA};
    for i in 0..n_tokens {
        let write_pos = write_pos_start + i;
        if !is_sliding && write_pos >= cache_capacity {
            return Err(MlxError::InvalidArgument(format!(
                "hadamard_quantize_kv_hb_seq: global cache write_pos({}) >= \
                 cache_capacity({}) at seq idx {}",
                write_pos, cache_capacity, i
            )));
        }
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
        let src_offset = ((src_tok_offset + i) as u64) * bytes_per_token;

        encode_threadgroups_with_args(
            encoder,
            pipeline,
            &[
                (0, KA::BufferWithOffset(src, src_offset)),
                (1, KA::Buffer(packed)),
                (2, KA::Buffer(norms)),
                (3, KA::Bytes(params_bytes)),
            ],
            MTLSize::new(num_kv_heads as u64, 1, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    Ok(())
}
