//! Decode-only fused Gated DeltaNet kernel — `simd_sum`-based variant.
//!
//! ADR-015 iter56. Mirrors llama.cpp's `kernel_gated_delta_net_f32_<NSG>`
//! threading model (32-lane simdgroup reductions, NSG state cells per
//! thread, no shared memory, no threadgroup barriers) — drop-in replacement
//! for [`super::gated_delta_net::dispatch_gated_delta_net`] when n_tokens is
//! small (decode regime, n_tokens=1).
//!
//! # When to use this vs `dispatch_gated_delta_net`
//!
//! * **Decode (n_tokens=1)** — use this. The existing
//!   `gated_delta_net_f32` uses 128-thread threadgroups + threadgroup memory
//!   + barriers, which on Apple GPU spills the 128-element private state row
//!   to threadgroup memory and bottlenecks on barrier stalls between every
//!   token. With NSG=4 simdgroup reductions the state row sits in registers
//!   (NSG=4 cells × 4 bytes = 16 bytes/thread) and reductions are
//!   single-cycle warp shuffles.
//! * **Prefill (n_tokens > 32 or so)** — keep using `dispatch_gated_delta_net`.
//!   The existing kernel's `output[i]` per-thread amortizes better when the
//!   state-update arithmetic dominates barrier overhead.
//!
//! Both kernels produce **bit-equivalent** output (modulo floating-point
//! associativity in the cross-lane reductions); the
//! `gated_delta_net_decode_parity` test in `tests/` enforces this on
//! representative shapes.
//!
//! # Buffer / params contract
//!
//! Identical to [`super::gated_delta_net::dispatch_gated_delta_net`]:
//! * `q` — pre-scaled by `1/sqrt(D_k)` (caller's responsibility), UNLESS
//!   `params[8]` (q_scale_bits) is non-zero, in which case `q` is passed
//!   unscaled and the kernel folds the scale in at writeback.
//! * `k`, `v`, `g`, `beta`, `state_in`, `output`, `state_out` — same shapes
//! * `params_buf` — 9-u32 layout (ADR-033 §Pi iter 25; extended from 8),
//!   built via [`super::gated_delta_net::build_gated_delta_net_params`] or
//!   [`super::gated_delta_net::build_gated_delta_net_params_with_q_scale`].
//!   Index 8 = `q_scale_bits` (0 = legacy / no fold-in). The dispatchers
//!   reject any buffer with fewer than 9 u32 to prevent an OOB read of
//!   `params[8]` in the kernel.
//!
//! # NSG selection
//!
//! `NSG = D_k / 32`. Currently supports `NSG ∈ {1, 2, 4}` covering
//! `D_k ∈ {32, 64, 128}` — the production Qwen3.5/3.6 head dim is 128 → NSG=4.
//! `D_k` MUST be a multiple of 32 and `<= 128`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::gated_delta_net::GatedDeltaNetParams;

pub static GATED_DELTA_NET_DECODE_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net_decode.metal");

/// ADR-034 task #90 (2026-05-21) — per-position state capture variant
/// for K≥2 speculative decoding on hybrid Qwen 3.5/3.6 (the MTPLX-style
/// rollback mechanism documented at
/// [[project_adr034_mtplx_gdn_capture_2026_05_21]]).
pub static GATED_DELTA_NET_DECODE_CAPTURE_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net_decode_capture.metal");

/// Hard cap: D_k must be ≤ 32 * MAX_NSG. NSG=4 → 128.
pub const MAX_NSG: u32 = 4;

/// Register the decode kernel under all three NSG-templated names.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "gated_delta_net_decode_f32_1",
        GATED_DELTA_NET_DECODE_SHADER_SOURCE,
    );
    registry.register_source(
        "gated_delta_net_decode_f32_2",
        GATED_DELTA_NET_DECODE_SHADER_SOURCE,
    );
    registry.register_source(
        "gated_delta_net_decode_f32_4",
        GATED_DELTA_NET_DECODE_SHADER_SOURCE,
    );
    // ADR-034 task #90 — per-position state-capture variants. Same
    // math + threading as decode_f32_<NSG>; ALSO writes
    // state_capture[D_k, D_v, n_v_heads, n_tokens, n_seqs] after each
    // token's update so caller can roll back to any accepted prefix
    // on partial-reject in K=N spec-decode.
    registry.register_source(
        "gated_delta_net_decode_capture_f32_1",
        GATED_DELTA_NET_DECODE_CAPTURE_SHADER_SOURCE,
    );
    registry.register_source(
        "gated_delta_net_decode_capture_f32_2",
        GATED_DELTA_NET_DECODE_CAPTURE_SHADER_SOURCE,
    );
    registry.register_source(
        "gated_delta_net_decode_capture_f32_4",
        GATED_DELTA_NET_DECODE_CAPTURE_SHADER_SOURCE,
    );
}

fn validate(
    p: &GatedDeltaNetParams,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g: &MlxBuffer,
    beta: &MlxBuffer,
    state_in: &MlxBuffer,
    output: &MlxBuffer,
    state_out: &MlxBuffer,
) -> Result<()> {
    if p.d_k == 0 || p.d_v == 0 || p.n_k_heads == 0 || p.n_v_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_decode: dims must all be > 0".into(),
        ));
    }
    if p.n_tokens == 0 || p.n_seqs == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_decode: n_tokens and n_seqs must be > 0".into(),
        ));
    }
    if p.n_v_heads % p.n_k_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: n_v_heads ({}) must be a multiple of n_k_heads ({})",
            p.n_v_heads, p.n_k_heads
        )));
    }
    if p.d_k % 32 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: D_k ({}) must be a multiple of 32 (simdgroup width)",
            p.d_k
        )));
    }
    if p.d_k / 32 > MAX_NSG {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: D_k ({}) implies NSG > {} — extend the .metal kernel",
            p.d_k, MAX_NSG
        )));
    }
    if p.d_v == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_decode: D_v must be > 0".into(),
        ));
    }
    let nsg = p.d_k / 32;
    if p.d_v % nsg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: D_v ({}) must be a multiple of NSG ({}=D_k/32)",
            p.d_v, nsg
        )));
    }

    let qk_elems = (p.d_k as usize)
        * (p.n_k_heads as usize)
        * (p.n_tokens as usize)
        * (p.n_seqs as usize);
    let v_elems = (p.d_v as usize)
        * (p.n_v_heads as usize)
        * (p.n_tokens as usize)
        * (p.n_seqs as usize);
    let scalar_elems = (p.n_v_heads as usize) * (p.n_tokens as usize) * (p.n_seqs as usize);
    let state_elems =
        (p.d_k as usize) * (p.d_v as usize) * (p.n_v_heads as usize) * (p.n_seqs as usize);

    for (name, buf, exp) in [
        ("q", q, qk_elems),
        ("k", k, qk_elems),
        ("v", v, v_elems),
        ("output", output, v_elems),
        ("g", g, scalar_elems),
        ("beta", beta, scalar_elems),
        ("state_in", state_in, state_elems),
        ("state_out", state_out, state_elems),
    ] {
        if buf.element_count() != exp {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode: {} element count {} != expected {}",
                name,
                buf.element_count(),
                exp
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode: {} must be f32 (got {})",
                name,
                buf.dtype()
            )));
        }
    }

    Ok(())
}

/// ADR-033 §Pi iter 25: the decode kernels read `params[8]` (q_scale_bits)
/// unconditionally, so the params buffer MUST hold at least 9 u32. Reject
/// shorter buffers here rather than triggering an OOB read on the GPU.
fn validate_params_buf(params_buf: &MlxBuffer) -> Result<()> {
    if params_buf.element_count() < 9 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode: params_buf has {} u32, expected >= 9 \
             (index 8 = q_scale_bits); build via build_gated_delta_net_params[_with_q_scale]",
            params_buf.element_count()
        )));
    }
    Ok(())
}

/// Dispatch the decode-only fused Gated DeltaNet kernel.
///
/// Equivalent math to [`super::gated_delta_net::dispatch_gated_delta_net`];
/// uses `simd_sum` reductions instead of threadgroup_barrier+shared memory.
/// Threadgroup `(32, NSG, 1)`; grid `(D_v / NSG, n_v_heads, n_seqs)`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net_decode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g: &MlxBuffer,
    beta: &MlxBuffer,
    state_in: &MlxBuffer,
    output: &MlxBuffer,
    state_out: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: GatedDeltaNetParams,
) -> Result<()> {
    validate(&p, q, k, v, g, beta, state_in, output, state_out)?;
    validate_params_buf(params_buf)?;

    let nsg: u32 = p.d_k / 32;
    let kernel_name = match nsg {
        1 => "gated_delta_net_decode_f32_1",
        2 => "gated_delta_net_decode_f32_2",
        4 => "gated_delta_net_decode_f32_4",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode: unsupported NSG={} (D_k={})",
                other, p.d_k
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // Threadgroup: (32, NSG, 1) = 32*NSG threads, NSG simdgroups.
    let tg = MTLSize::new(32, nsg as u64, 1);
    // Grid (in threadgroups): (D_v / NSG, n_v_heads, n_seqs).
    let grid_tgs = MTLSize::new((p.d_v / nsg) as u64, p.n_v_heads as u64, p.n_seqs as u64);

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, q),
            (1, k),
            (2, v),
            (3, g),
            (4, beta),
            (5, state_in),
            (6, output),
            (7, state_out),
            (8, params_buf),
        ],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// ADR-034 task #90 (2026-05-21) — Dispatch the Gated DeltaNet decode
/// kernel with **per-position state capture**, the MTPLX-style rollback
/// mechanism that unlocks K≥2 speculative decoding on hybrid Qwen 3.5/3.6.
///
/// Identical math + identical state_out semantics to
/// [`dispatch_gated_delta_net_decode`]; additionally writes the recurrent
/// state AFTER each token's update to a caller-owned `state_capture`
/// buffer of shape `[D_k, D_v, n_v_heads, n_tokens, n_seqs]` (innermost
/// `D_k`, same as `state_in`/`state_out` extended with a `n_tokens` axis).
///
/// On partial-reject in K=N spec-decode, the caller reads
/// `state_capture[..., accepted_idx, ...]` as the new active state for
/// the `LinearAttnStateSlot` — solving the GDN rollback gap documented
/// at task #86.
///
/// **Determinism contract**: `state_capture[..., n_tokens-1, ...]` is
/// guaranteed equal to `state_out` (byte-identical; same compute, same
/// thread writes both at the end of the token loop).
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] when:
/// - `state_capture.element_count() != n_tokens * state_elems`
/// - `state_capture.dtype() != F32`
/// - any of the other validations from [`validate`] fail.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net_decode_with_capture(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g: &MlxBuffer,
    beta: &MlxBuffer,
    state_in: &MlxBuffer,
    output: &MlxBuffer,
    state_out: &MlxBuffer,
    params_buf: &MlxBuffer,
    state_capture: &MlxBuffer,
    p: GatedDeltaNetParams,
) -> Result<()> {
    validate(&p, q, k, v, g, beta, state_in, output, state_out)?;
    validate_params_buf(params_buf)?;

    // Validate state_capture shape. The kernel uses `params.n_tokens` as
    // the per-sequence token stride (capture_seq_stride = n_tokens *
    // n_v_heads * D_v * D_k), so for n_seqs > 1 the buffer MUST be sized
    // exactly to that layout — otherwise sequence offsets interleave
    // incorrectly. For n_seqs == 1 (current Qwen 3.5/3.6 production),
    // seq_stride is unused and the buffer may be larger than required
    // (e.g. pre-allocated at max-spec-depth + 1 in
    // HybridKvCache::ensure_la_capture). Codex audit 2026-05-21
    // (task #90 Step 2) flagged the n_seqs>1 case — guard accordingly.
    let state_elems =
        (p.d_k as usize) * (p.d_v as usize) * (p.n_v_heads as usize) * (p.n_seqs as usize);
    let capture_required = state_elems * (p.n_tokens as usize);
    if p.n_seqs > 1 {
        if state_capture.element_count() != capture_required {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode_with_capture: state_capture element count {} != \
                 required {} (n_tokens={} × state_elems={}); strict EQUALITY required \
                 when n_seqs > 1 (got n_seqs={}) because the per-sequence token stride \
                 depends on params.n_tokens",
                state_capture.element_count(), capture_required, p.n_tokens,
                state_elems, p.n_seqs,
            )));
        }
    } else if state_capture.element_count() < capture_required {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode_with_capture: state_capture element count {} < \
             required {} (n_tokens={} × state_elems={})",
            state_capture.element_count(), capture_required, p.n_tokens, state_elems,
        )));
    }
    if state_capture.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_decode_with_capture: state_capture must be f32 (got {})",
            state_capture.dtype(),
        )));
    }

    let nsg: u32 = p.d_k / 32;
    let kernel_name = match nsg {
        1 => "gated_delta_net_decode_capture_f32_1",
        2 => "gated_delta_net_decode_capture_f32_2",
        4 => "gated_delta_net_decode_capture_f32_4",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_decode_with_capture: unsupported NSG={} (D_k={})",
                other, p.d_k
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let tg = MTLSize::new(32, nsg as u64, 1);
    let grid_tgs = MTLSize::new((p.d_v / nsg) as u64, p.n_v_heads as u64, p.n_seqs as u64);

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, q),
            (1, k),
            (2, v),
            (3, g),
            (4, beta),
            (5, state_in),
            (6, output),
            (7, state_out),
            (8, params_buf),
            (9, state_capture),
        ],
        grid_tgs,
        tg,
    );

    Ok(())
}
