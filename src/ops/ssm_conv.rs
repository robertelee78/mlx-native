//! SSM depthwise causal 1D conv + SiLU GPU dispatch.
//!
//! Used by Qwen3.5 Gated DeltaNet linear-attention layers to apply a
//! 4-kernel-wide causal conv1d across the QKV projection's output
//! (ADR-013 Decision 7).
//!
//! # Operation
//!
//! ```text
//! ssm_conv(x, kernel_w, state) -> (y, new_state)
//!   x:        [channels, n_tokens, n_seqs]
//!   kernel_w: [K, channels]              (K = 4 for Qwen3.5)
//!   state:    [K-1, channels, n_seqs]    (previous (K-1) conv inputs per seq)
//!
//! extended(c, t_ext, s) = state(t_ext, c, s)            if t_ext < K - 1
//!                         x(c, t_ext - (K-1), s)        otherwise
//! y(c, t, s) = silu( sum_{k=0..K} kernel_w(k, c) * extended(c, t + k, s) )
//! new_state(i, c, s) = extended(c, n_tokens + i, s)  for i in 0..K-1
//! ```
//!
//! # Memory layout (column-major, innermost-first)
//!
//! * `x[c, t, s]`        at offset `s * n_tokens * channels + t * channels + c`
//! * `y[c, t, s]`        same shape and layout as `x`
//! * `state[i, c, s]`    at offset `s * (K-1) * channels + c * (K-1) + i`
//! * `kernel_w[k, c]`    at offset `c * K + k`
//!
//! The per-(c, s) state row of K-1 values is contiguous in memory, matching
//! the expected ring-buffer slice that callers view as `state[:, c, s]`.
//!
//! # Two-pass design
//!
//! The forward and state-update kernels are separate dispatches because:
//! 1. When `n_tokens + i < K - 1` the state-update reads from the old state;
//!    this would alias the output if written in place.
//! 2. The state update is a small O(K × channels × n_seqs) pass whose
//!    arithmetic is different from the main conv; fusing them would waste
//!    threads.
//!
//! Callers must provide separate `old_state` and `new_state` buffers. The
//! `dispatch_ssm_conv` helper below accepts both in a single call and encodes
//! both kernels back-to-back.
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static SSM_CONV_SHADER_SOURCE: &str = include_str!("../shaders/ssm_conv.metal");

/// ADR-034 task #90 Step 4b (2026-05-21) — per-position conv-state capture
/// kernel for K=N speculative decoding rollback on hybrid Qwen 3.5/3.6.
/// Source: `../shaders/ssm_conv_capture.metal` — port of MTPLX
/// `gdn_capture.py:61-125`.
pub static SSM_CONV_CAPTURE_SHADER_SOURCE: &str =
    include_str!("../shaders/ssm_conv_capture.metal");

/// Register SSM conv shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("ssm_conv_forward_f32", SSM_CONV_SHADER_SOURCE);
    registry.register_source("ssm_conv_forward_bf16", SSM_CONV_SHADER_SOURCE);
    registry.register_source("ssm_conv_state_update_f32", SSM_CONV_SHADER_SOURCE);
    registry.register_source("ssm_conv_state_update_bf16", SSM_CONV_SHADER_SOURCE);
    // ADR-034 task #90 Step 4b — capture variant. Fused forward + per-position
    // state capture in one dispatch (no separate state-update kernel needed
    // because each per-position state is computed inline).
    registry.register_source(
        "ssm_conv_capture_forward_f32",
        SSM_CONV_CAPTURE_SHADER_SOURCE,
    );
}

/// Shape parameters for an ssm_conv dispatch.
#[derive(Debug, Clone, Copy)]
pub struct SsmConvParams {
    pub channels: u32,
    pub n_tokens: u32,
    pub n_seqs: u32,
    pub k_width: u32, // typically 4; ADR-013 forbids K <= 1
}

fn validate(
    params: &SsmConvParams,
    x: &MlxBuffer,
    kernel_w: &MlxBuffer,
    old_state: &MlxBuffer,
    new_state: &MlxBuffer,
    y: &MlxBuffer,
) -> Result<()> {
    if params.channels == 0 || params.n_tokens == 0 || params.n_seqs == 0 {
        return Err(MlxError::InvalidArgument(
            "ssm_conv: channels, n_tokens, n_seqs must all be > 0".into(),
        ));
    }
    if params.k_width < 2 {
        return Err(MlxError::InvalidArgument(
            "ssm_conv: k_width must be >= 2 (K=1 has empty state)".into(),
        ));
    }
    let x_elems = (params.channels as usize)
        .checked_mul(params.n_tokens as usize)
        .and_then(|v| v.checked_mul(params.n_seqs as usize))
        .ok_or_else(|| MlxError::InvalidArgument("ssm_conv: shape overflow".into()))?;
    let w_elems = (params.k_width as usize) * (params.channels as usize);
    let s_elems = ((params.k_width - 1) as usize)
        * (params.channels as usize)
        * (params.n_seqs as usize);

    if x.element_count() != x_elems {
        return Err(MlxError::InvalidArgument(format!(
            "ssm_conv: x element count {} != channels({}) * n_tokens({}) * n_seqs({})",
            x.element_count(),
            params.channels,
            params.n_tokens,
            params.n_seqs
        )));
    }
    if y.element_count() != x_elems {
        return Err(MlxError::InvalidArgument(format!(
            "ssm_conv: y element count {} != expected {}",
            y.element_count(),
            x_elems
        )));
    }
    if kernel_w.element_count() != w_elems {
        return Err(MlxError::InvalidArgument(format!(
            "ssm_conv: kernel_w element count {} != K({}) * channels({})",
            kernel_w.element_count(),
            params.k_width,
            params.channels
        )));
    }
    if old_state.element_count() != s_elems || new_state.element_count() != s_elems {
        return Err(MlxError::InvalidArgument(format!(
            "ssm_conv: state element count mismatch; old={} new={} expected {}",
            old_state.element_count(),
            new_state.element_count(),
            s_elems
        )));
    }

    let dt = x.dtype();
    for (name, buf) in [
        ("kernel_w", kernel_w),
        ("old_state", old_state),
        ("new_state", new_state),
        ("y", y),
    ] {
        if buf.dtype() != dt {
            return Err(MlxError::InvalidArgument(format!(
                "ssm_conv: dtype mismatch — x is {}, {} is {}",
                dt,
                name,
                buf.dtype()
            )));
        }
    }
    Ok(())
}

/// Dispatch a fused depthwise causal 1D conv + SiLU plus state update.
///
/// Two kernels are encoded back-to-back: the forward conv produces `y`, and a
/// small state update writes the last K-1 tokens of the extended stream into
/// `new_state`. Callers may point `old_state` and `new_state` at the same
/// backing buffer if-and-only-if `n_tokens >= k_width - 1` (the state-update
/// never reads from `old_state` in that regime, so aliasing is safe). For
/// decode with `n_tokens < K - 1` a separate buffer is mandatory.
///
/// # Arguments
///
/// * `params_buf` - buffer of 4 u32 `[channels, n_tokens, n_seqs, k_width]`.
///
/// # Errors
///
/// See [`validate`] for the full list.
pub fn dispatch_ssm_conv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    kernel_w: &MlxBuffer,
    old_state: &MlxBuffer,
    new_state: &MlxBuffer,
    y: &MlxBuffer,
    params_buf: &MlxBuffer,
    params: SsmConvParams,
) -> Result<()> {
    validate(&params, x, kernel_w, old_state, new_state, y)?;

    let (fwd_name, state_name) = match x.dtype() {
        DType::F32 => ("ssm_conv_forward_f32", "ssm_conv_state_update_f32"),
        DType::BF16 => ("ssm_conv_forward_bf16", "ssm_conv_state_update_bf16"),
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "ssm_conv: unsupported dtype {}",
                other
            )))
        }
    };

    // Forward: one thread per (c, t, s).
    let fwd_pipeline = registry.get_pipeline(fwd_name, device)?;
    let fwd_grid = MTLSize::new(
        params.channels as u64,
        params.n_tokens as u64,
        params.n_seqs as u64,
    );
    // Threadgroup: keep total <= 256, prefer packing along the channels axis.
    let tg_c = std::cmp::min(params.channels, 256).max(1);
    let remain = 256u32 / tg_c;
    let tg_t = std::cmp::min(params.n_tokens, remain).max(1);
    let remain2 = (256u32 / (tg_c * tg_t)).max(1);
    let tg_s = std::cmp::min(params.n_seqs, remain2).max(1);
    let fwd_tg = MTLSize::new(tg_c as u64, tg_t as u64, tg_s as u64);

    encoder.encode(
        fwd_pipeline,
        &[
            (0, x),
            (1, kernel_w),
            (2, old_state),
            (3, y),
            (4, params_buf),
        ],
        fwd_grid,
        fwd_tg,
    );

    // State update: one thread per (i, c, s), i in 0..K-1.
    let state_pipeline = registry.get_pipeline(state_name, device)?;
    let state_grid = MTLSize::new(
        (params.k_width - 1) as u64,
        params.channels as u64,
        params.n_seqs as u64,
    );
    let su_tg_i = (params.k_width - 1).max(1);
    let su_remain = (256u32 / su_tg_i).max(1);
    let su_tg_c_raw = std::cmp::min(params.channels, su_remain).max(1);
    // ADR-029 iter-175 Step 1r: ensure su_tg_i * su_tg_c is a multiple of
    // 32 (Apple's threadExecutionWidth).  Pre-fix this site computed
    // (3, 85, 1) → 255 threads per TG when k_width=4 and channels≥85,
    // which is NOT a multiple of 32 and causes UB under
    // HF2Q_PIPELINE_TG_MULT_HINT=1.  Coherence_smoke
    // apex-q5km/{the-quick-brown-fox, what-is-22, hello-my-name-is}
    // caught this via the Step 1q safety assertion.
    //
    // Strategy: shrink su_tg_c to the largest multiple of (32 / gcd(su_tg_i, 32))
    // that still respects the channel/256 limit.  gcd(3, 32) = 1 → step = 32,
    // so su_tg_c rounds DOWN to a multiple of 32 (85 → 64).  gcd(2, 32) = 2 →
    // step = 16; gcd(4, 32) = 4 → step = 8; gcd(8, 32) = 8 → step = 4; etc.
    // When su_tg_i is itself a multiple of 32 (rare for SSM conv), step = 1
    // and any su_tg_c works — leaves the original value.
    fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
        while b != 0 { let t = b; b = a % b; a = t; }
        a
    }
    let step = 32u32 / gcd_u32(su_tg_i, 32);
    let su_tg_c = if step <= 1 {
        su_tg_c_raw
    } else if su_tg_c_raw >= step {
        (su_tg_c_raw / step) * step  // round DOWN to multiple of step
    } else {
        // su_tg_c_raw < step: can't satisfy multiple-of-32 with this su_tg_i.
        // Leave su_tg_c_raw alone; safety check will fire under
        // HF2Q_PIPELINE_TG_MULT_HINT=1 (correct behavior — this case
        // means the kernel inherently can't satisfy the constraint).
        su_tg_c_raw
    };
    let su_remain2 = (256u32 / (su_tg_i * su_tg_c)).max(1);
    let su_tg_s = std::cmp::min(params.n_seqs, su_remain2).max(1);
    let state_tg = MTLSize::new(su_tg_i as u64, su_tg_c as u64, su_tg_s as u64);

    encoder.encode(
        state_pipeline,
        &[
            (0, x),
            (1, old_state),
            (2, new_state),
            (3, params_buf),
        ],
        state_grid,
        state_tg,
    );

    Ok(())
}

/// ADR-034 task #90 Step 4b (2026-05-21) — Dispatch the SSM conv kernel
/// with **per-position state capture**, used for K=N speculative decoding
/// rollback on hybrid Qwen 3.5/3.6. Fuses forward output (identical to
/// `ssm_conv_forward_f32`) with per-position state capture in one kernel
/// dispatch (no separate state-update kernel needed; the per-position
/// state is computed inline at each token).
///
/// Output buffer `conv_capture` shape `[n_seqs, n_tokens, K-1, channels]`
/// F32. The caller selects `conv_capture[s, accepted_idx, :, :]` on
/// partial-reject as the new active conv state.
///
/// Determinism contract: `conv_capture[..., n_tokens-1, ...]` is
/// byte-identical to the `new_state` produced by
/// `ssm_conv_state_update_f32` for the same inputs.
///
/// Only F32 supported in this iteration (matches the production hf2q
/// Qwen 3.5/3.6 path). BF16 variant can be added if a future arch needs it.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_ssm_conv_with_capture(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    x: &MlxBuffer,
    kernel_w: &MlxBuffer,
    old_state: &MlxBuffer,
    y: &MlxBuffer,
    conv_capture: &MlxBuffer,
    params_buf: &MlxBuffer,
    params: SsmConvParams,
) -> Result<()> {
    // Validate forward inputs (channels/n_tokens/n_seqs/k_width/x/y/old_state/kernel_w).
    // Construct a dummy new_state for the validate() signature (a real
    // new_state isn't needed because we don't dispatch the state-update
    // kernel — we capture inline).
    if params.channels == 0 || params.n_tokens == 0 || params.n_seqs == 0 {
        return Err(MlxError::InvalidArgument(
            "ssm_conv_with_capture: channels, n_tokens, n_seqs must all be > 0".into(),
        ));
    }
    if params.k_width < 2 {
        return Err(MlxError::InvalidArgument(
            "ssm_conv_with_capture: k_width must be >= 2".into(),
        ));
    }
    if x.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "ssm_conv_with_capture: only F32 supported in this iter; got x dtype {}",
            x.dtype()
        )));
    }

    let x_elems = (params.channels as usize)
        .checked_mul(params.n_tokens as usize)
        .and_then(|v| v.checked_mul(params.n_seqs as usize))
        .ok_or_else(|| MlxError::InvalidArgument("ssm_conv_with_capture: shape overflow".into()))?;
    let s_elems = ((params.k_width - 1) as usize)
        * (params.channels as usize)
        * (params.n_seqs as usize);
    let capture_elems = (params.n_seqs as usize)
        * (params.n_tokens as usize)
        * ((params.k_width - 1) as usize)
        * (params.channels as usize);
    let w_elems = (params.k_width as usize) * (params.channels as usize);

    for (name, buf, exp) in [
        ("x", x, x_elems),
        ("y", y, x_elems),
        ("kernel_w", kernel_w, w_elems),
        ("old_state", old_state, s_elems),
        ("conv_capture", conv_capture, capture_elems),
    ] {
        if buf.element_count() != exp {
            return Err(MlxError::InvalidArgument(format!(
                "ssm_conv_with_capture: {} element count {} != expected {}",
                name, buf.element_count(), exp
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "ssm_conv_with_capture: {} must be F32 (got {})",
                name, buf.dtype()
            )));
        }
    }

    let pipeline = registry.get_pipeline("ssm_conv_capture_forward_f32", device)?;

    // One thread per (c, t, s) — same dispatch grid as ssm_conv_forward.
    let grid = MTLSize::new(
        params.channels as u64,
        params.n_tokens as u64,
        params.n_seqs as u64,
    );
    // Same threadgroup packing strategy as ssm_conv_forward.
    let tg_c = std::cmp::min(params.channels, 256).max(1);
    let remain = 256u32 / tg_c;
    let tg_t = std::cmp::min(params.n_tokens, remain).max(1);
    let remain2 = (256u32 / (tg_c * tg_t)).max(1);
    let tg_s = std::cmp::min(params.n_seqs, remain2).max(1);
    let tg = MTLSize::new(tg_c as u64, tg_t as u64, tg_s as u64);

    encoder.encode(
        pipeline,
        &[
            (0, x),
            (1, kernel_w),
            (2, old_state),
            (3, y),
            (4, conv_capture),
            (5, params_buf),
        ],
        grid,
        tg,
    );

    Ok(())
}
