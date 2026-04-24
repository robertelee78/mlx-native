//! Gated DeltaNet fused GPU dispatch — the centerpiece of Qwen3.5
//! linear-attention layers.
//!
//! Implements the recurrence (per token `t` within a sequence):
//!
//! ```text
//! alpha       = exp(-g[t])
//! state_dec   = alpha * state                               // decay FIRST
//! delta       = v[t] - state_dec @ k[t]                    // use decayed state
//! state'      = state_dec + beta[t] * outer(delta, k[t])
//! output[t]   = state' @ q[t]
//! ```
//!
//! IMPORTANT: alpha is applied before computing delta. This matches
//! llama.cpp build_delta_net_autoregressive which does: s=s*exp(gate); sk=sum(s*k).
//!
//! Spec source: ADR-013 Decision 6. Derived from the mathematical spec;
//! no llama.cpp code copied.
//!
//! # GQA broadcast
//!
//! Qwen3.5 has `num_v_heads >= num_k_heads` (dense: 48 vs 16; MoE: 32 vs 16).
//! Each v_head picks its K/Q from `k_head = v_head / group_ratio`. The
//! kernel resolves this internally; the caller passes both counts.
//!
//! # Dtype / precision
//!
//! Forward accumulation is f32 throughout (`mamba_ssm_dtype: float32`).
//! Inputs and outputs are f32 in the current impl; bf16 variants are a
//! future add.
//!
//! # Memory layouts (innermost-first / column-major)
//!
//! | Buffer      | Shape                                 | Notes                    |
//! |-------------|---------------------------------------|--------------------------|
//! | `q`, `k`    | `[D_k, n_k_heads, n_tokens, n_seqs]`  | `d_k` fastest            |
//! | `v`         | `[D_v, n_v_heads, n_tokens, n_seqs]`  | `d_v` fastest            |
//! | `g`, `beta` | `[n_v_heads, n_tokens, n_seqs]`       | `v_head` fastest         |
//! | `state_*`   | `[D_k, D_v, n_v_heads, n_seqs]`       | `d_k` fastest (for per-thread contig reads) |
//! | `output`    | `[D_v, n_v_heads, n_tokens, n_seqs]`  | same as v                |
//!
//! # Threading
//!
//! One threadgroup per `(v_head, seq)`; threadgroup size = `D_v` threads.
//! Thread `i` owns `state[:, i, v_head, seq]` (D_k scalars) in private
//! memory, loaded once at start and written once at end. This keeps state
//! traffic `O(D_k × D_v)` per head per seq, independent of `n_tokens`.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static GATED_DELTA_NET_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net.metal");

/// Hard cap on per-thread state row length (must equal `MAX_STATE_D` in the
/// Metal kernel). Growing this on the Rust side without matching the shader
/// will silently produce wrong output — keep them in sync.
pub const MAX_STATE_D: u32 = 128;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("gated_delta_net_f32", GATED_DELTA_NET_SHADER_SOURCE);
}

#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetParams {
    pub d_k: u32,
    pub d_v: u32,
    pub n_k_heads: u32,
    pub n_v_heads: u32,
    pub n_tokens: u32,
    pub n_seqs: u32,
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
            "gated_delta_net: dims must all be > 0".into(),
        ));
    }
    if p.n_tokens == 0 || p.n_seqs == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net: n_tokens and n_seqs must be > 0".into(),
        ));
    }
    if p.n_v_heads % p.n_k_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net: n_v_heads ({}) must be a multiple of n_k_heads ({})",
            p.n_v_heads, p.n_k_heads
        )));
    }
    if p.d_k > MAX_STATE_D || p.d_v > MAX_STATE_D {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net: d_k ({}) and d_v ({}) must be <= MAX_STATE_D ({})",
            p.d_k, p.d_v, MAX_STATE_D
        )));
    }

    // Expected element counts.
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
                "gated_delta_net: {} element count {} != expected {}",
                name,
                buf.element_count(),
                exp
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net: {} must be f32 (got {})",
                name,
                buf.dtype()
            )));
        }
    }

    Ok(())
}

/// Dispatch the fused Gated DeltaNet kernel.
///
/// `params_buf` must hold the 8 u32 `[d_k, d_v, n_k_heads, n_v_heads,
/// n_tokens, n_seqs, 0, 0]` — use [`build_gated_delta_net_params`] to
/// allocate + populate it.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net(
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

    let pipeline = registry.get_pipeline("gated_delta_net_f32", device)?;

    // Threadgroup: D_v threads. One tg per (v_head, seq).
    let tg = MTLSize::new(p.d_v as u64, 1, 1);
    let grid_tgs = MTLSize::new(p.n_v_heads as u64, p.n_seqs as u64, 1);

    // Shared memory: 2*D_k + 2*D_v floats (k, q, v, delta).
    let shared_floats = 2 * (p.d_k as u64) + 2 * (p.d_v as u64);
    let shared_bytes = shared_floats * 4;

    encoder.encode_threadgroups_with_shared(
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
        &[(0, shared_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// Build the 8-u32 params buffer.
pub fn build_gated_delta_net_params(
    device: &crate::MlxDevice,
    p: GatedDeltaNetParams,
) -> Result<MlxBuffer> {
    let mut buf = device.alloc_buffer(8 * 4, DType::U32, vec![8])?;
    {
        let s = buf.as_mut_slice::<u32>()?;
        s[0] = p.d_k;
        s[1] = p.d_v;
        s[2] = p.n_k_heads;
        s[3] = p.n_v_heads;
        s[4] = p.n_tokens;
        s[5] = p.n_seqs;
        s[6] = 0;
        s[7] = 0;
    }
    Ok(buf)
}

// ==============================================================
// Scalar CPU reference (ADR-013 Decision 6 — "internal test oracle")
// ==============================================================

/// Pure-Rust scalar reference implementation of the spec.
/// Output: (output vec of length D_v * n_v_heads * n_tokens * n_seqs,
///          state_out vec of length D_k * D_v * n_v_heads * n_seqs).
///
/// Public so the hf2q side (when P4 lands) can reuse it as the
/// parity-test oracle per ADR-013 Decision 8.
pub fn cpu_reference_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    p: GatedDeltaNetParams,
) -> (Vec<f32>, Vec<f32>) {
    let d_k = p.d_k as usize;
    let d_v = p.d_v as usize;
    let nh_k = p.n_k_heads as usize;
    let nh_v = p.n_v_heads as usize;
    let n_t = p.n_tokens as usize;
    let n_s = p.n_seqs as usize;
    // GQA: tiled mapping k_head = v_head % n_k_heads (matches llama.cpp fused kernel).
    // Not block-style (v_head / group_ratio) which gives wrong ordering.
    let kq_token_stride = nh_k * d_k;
    let kq_seq_stride = n_t * kq_token_stride;
    let v_token_stride = nh_v * d_v;
    let v_seq_stride = n_t * v_token_stride;
    let scalar_seq_stride = n_t * nh_v;
    let state_head_stride = d_v * d_k;
    let state_seq_stride = nh_v * state_head_stride;

    let mut output = vec![0.0f32; n_s * v_seq_stride];
    // State is mutable; start from state_in.
    let mut state = state_in.to_vec();

    for s in 0..n_s {
        for vh in 0..nh_v {
            let kh = vh % nh_k;  // tiled GQA: matches llama.cpp k_head = v_head % n_k_heads

            for t in 0..n_t {
                let kq_base = s * kq_seq_stride + t * kq_token_stride + kh * d_k;
                let v_base = s * v_seq_stride + t * v_token_stride + vh * d_v;
                let sc_idx = s * scalar_seq_stride + t * nh_v + vh;

                let beta_val = beta[sc_idx];
                let g_val = g[sc_idx];
                let alpha = (-g_val).exp();

                let state_base = s * state_seq_stride + vh * state_head_stride;

                // Step 1: decay state — apply alpha BEFORE computing delta.
                // Matches llama.cpp: s = s * exp(gate); sk = sum(s * k).
                for i in 0..d_v {
                    for j in 0..d_k {
                        state[state_base + i * d_k + j] *= alpha;
                    }
                }

                // Step 2: Compute delta[i] = v[i] - sum_j (alpha*state)[j, i] * k[j]
                let mut delta = vec![0.0f32; d_v];
                for i in 0..d_v {
                    let mut sk = 0.0f32;
                    for j in 0..d_k {
                        sk += state[state_base + i * d_k + j] * k[kq_base + j];
                    }
                    delta[i] = v[v_base + i] - sk;
                }

                // Step 3: Update state: state[j, i] += beta * delta[i] * k[j]
                // (state is already alpha-decayed from step 1)
                for i in 0..d_v {
                    let beta_delta = beta_val * delta[i];
                    for j in 0..d_k {
                        let idx = state_base + i * d_k + j;
                        state[idx] += beta_delta * k[kq_base + j];
                    }
                }

                // output[i, vh, t, s] = sum_j state'[j, i] * q[j]
                for i in 0..d_v {
                    let mut acc = 0.0f32;
                    for j in 0..d_k {
                        acc += state[state_base + i * d_k + j] * q[kq_base + j];
                    }
                    output[v_base + i] = acc;
                }
            }
        }
    }

    (output, state)
}
