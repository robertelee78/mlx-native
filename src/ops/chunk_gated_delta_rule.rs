//! Wave 5b.1 iter 4 — chunk_gated_delta_rule_fwd orchestrator.
//!
//! Composes the iter-1..iter-3 chunk-pipeline kernels into a single
//! end-to-end forward pass. Mirrors FLA's
//!
//!   /opt/vllm/vllm/model_executor/layers/fla/ops/chunk.py:23-84
//!
//! call chain, plus the `use_qk_l2norm_in_kernel` pre-step at chunk.py:106-108.
//!
//! # Pipeline (per dispatch call)
//!
//! 1. (optional) `q = l2_norm(q)` and `k = l2_norm(k)` if `use_qk_l2norm`.
//! 2. `g_cumsum = chunk_local_cumsum(g_log_decay, BT)`             — per-chunk prefix sum on log-decay.
//! 3. `A_strict = chunk_scaled_dot_kkt(k_normed, beta, g_cumsum)`   — iter-2 kkt.
//! 4. `A_inv = chunk_tri_solve_invert(A_strict)`                    — `(I + A_strict)^-1` per chunk-block.
//! 5. `(w, u) = recompute_w_u(k_normed, v, beta, A_inv, g_cumsum)`  — iter-2.
//! 6. `(h, v_new, final_state) = chunk_inter_state(k_normed, w, u, g_cumsum, h0)` — iter-1.
//! 7. `o = chunk_fwd_o(q_normed, k_normed, v_new, h, g_cumsum, scale)` — iter-3.
//!
//! Returns `o [B, T, H, V] bf16` and `final_state [B, H, V, K] f32` via the
//! caller-provided output buffers.
//!
//! # Numerical bar
//!
//! End-to-end vs the FLA Python reference: `max_o_err < 1e-2`,
//! `max_final_state_err < 1e-2`. Looser than per-kernel fixtures (5e-3)
//! because 6 stages of bf16 round-off compound. The independent O(T^2)
//! oracle (in `tests/fixtures/chunk_gated_delta_rule_fwd_oracle.py`)
//! cross-validates the FLA reference itself at this same chunk-pipeline
//! shape, confirming the math is sound.
//!
//! # Allocations
//!
//! For each call, the orchestrator allocates these intermediate buffers:
//!   g_cumsum  [B, T, H]              f32  = B·T·H · 4 bytes
//!   A_strict  [B, T, H, BT]          f32  = B·T·H·BT · 4 bytes
//!   A_inv     [B, T, H, BT]          f32  = B·T·H·BT · 4 bytes
//!   w         [B, T, H, K]           bf16 = B·T·H·K · 2 bytes
//!   u         [B, T, H, V]           bf16 = B·T·H·V · 2 bytes
//!   h         [B, NT, H, V, K]       bf16 = B·NT·H·V·K · 2 bytes
//!   v_new     [B, T, H, V]           bf16 = B·T·H·V · 2 bytes
//!   q_normed  [B, T, Hg, K]          bf16 = B·T·Hg·K · 2 bytes (only if use_qk_l2norm)
//!   k_normed  [B, T, Hg, K]          bf16 = B·T·Hg·K · 2 bytes (only if use_qk_l2norm)
//!
//! Plus seven small u32 params buffers (≤ 11 u32 each).
//!
//! At the iter-4 test shape (B=1, T=128, Hg=2, H=4, K=128, V=128, BT=64,
//! NT=2, l2norm=on):
//!
//!   g_cumsum  : 1·128·4·4              = 2 048 B
//!   A_strict  : 1·128·4·64·4           = 131 072 B
//!   A_inv     : 1·128·4·64·4           = 131 072 B
//!   w         : 1·128·4·128·2          = 131 072 B
//!   u         : 1·128·4·128·2          = 131 072 B
//!   h         : 1·2·4·128·128·2        = 262 144 B
//!   v_new     : 1·128·4·128·2          = 131 072 B
//!   q_normed  : 1·128·2·128·2          = 65 536 B
//!   k_normed  : 1·128·2·128·2          = 65 536 B
//!   ───
//!   total     : 1 050 624 B (~1.00 MiB)
//!
//! Iter 5 fusion can bring this down (e.g., A_strict can be reused as A_inv
//! storage; v_new can alias u after the inter-state kernel) but iter-4
//! correctness comes first.
//!
//! # Spec sources
//!
//! - FLA orchestrator         : chunk.py:23-84
//! - L2-norm pre-step         : chunk.py:106-108 (Function.forward)
//! - chunk_local_cumsum       : cumsum.py:160-195 (scalar variant)
//! - chunk_scaled_dot_kkt_fwd : chunk_scaled_dot_kkt.py:36-99
//! - solve_tril               : solve_tril.py:506-530 (returns `(I+A)^-1`)
//! - recompute_w_u_fwd        : wy_fast.py:29-117
//! - chunk_gated_delta_rule_fwd_h : chunk_delta_h.py:43-298
//! - chunk_fwd_o              : chunk_o.py:42-138

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::MlxDevice;

use crate::ops::chunk_gated_delta_rule_tri_solve_invert::{
    build_chunk_tri_solve_invert_params, dispatch_chunk_tri_solve_invert,
    ChunkTriSolveInvertParams,
};
use crate::ops::gated_delta_net_chunk::{
    build_gated_delta_net_chunk_params, dispatch_gated_delta_net_chunk_inter_state,
    GatedDeltaNetChunkParams,
};
use crate::ops::gated_delta_net_chunk_o::{
    build_gated_delta_net_chunk_o_params, dispatch_gated_delta_net_chunk_o,
    GatedDeltaNetChunkOParams,
};
use crate::ops::gated_delta_net_kkt::{
    build_gated_delta_net_kkt_params, dispatch_gated_delta_net_kkt,
    GatedDeltaNetKktParams,
};
use crate::ops::gated_delta_net_recompute_wu::{
    build_gated_delta_net_recompute_wu_params, dispatch_gated_delta_net_recompute_wu,
    GatedDeltaNetRecomputeWuParams,
};
use crate::ops::l2_norm::dispatch_l2_norm;

pub static CHUNK_LOCAL_CUMSUM_G_SHADER_SOURCE: &str =
    include_str!("../shaders/chunk_local_cumsum_g.metal");

/// Required K — must equal 128 exactly. Sub-kernels inter_state and chunk_o
/// have compile-time-fixed 16 K-tiles in their simdgroup_matrix MMA loops
/// (Wave 5b.2 iter 1+2 + iter 1.5/2.5 narrowing). Codex iter-7 audit
/// caught the K<128 silent-corruption hole.
pub const MAX_K: u32 = 128;
/// Hard cap on per-tile head-dim V — matches sub-kernel caps.
pub const MAX_V: u32 = 256;
/// Iter-4 fixed BT (matches all sub-kernels).
pub const FIXED_BT: u32 = 64;
/// L2-norm epsilon — FLA `l2norm_fwd` default.
pub const L2_NORM_EPS: f32 = 1.0e-6;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "chunk_local_cumsum_g_f32",
        CHUNK_LOCAL_CUMSUM_G_SHADER_SOURCE,
    );
}

/// Shape parameters for the chunk-parallel gated delta-rule forward pass.
#[derive(Debug, Clone, Copy)]
pub struct ChunkGatedDeltaRuleParams {
    /// Batch size.
    pub b: u32,
    /// Sequence length (must be a multiple of `bt`).
    pub t: u32,
    /// K-head count (Hg).
    pub hg: u32,
    /// V-head count (H). `H % Hg == 0` is required for GQA.
    pub h: u32,
    /// Per-head K dimension.
    pub k: u32,
    /// Per-head V dimension.
    pub v: u32,
    /// Chunk size (BT). Iter 4: must be 64.
    pub bt: u32,
    /// Output scale (typically `K^-0.5`).
    pub scale: f32,
    /// Apply pre-l2-norm to q + k (Qwen3.6 default = true).
    pub use_qk_l2norm: bool,
}

impl ChunkGatedDeltaRuleParams {
    pub fn num_chunks(&self) -> u32 {
        self.t.div_ceil(self.bt)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate(
    p: &ChunkGatedDeltaRuleParams,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g_log_decay: &MlxBuffer,
    beta: &MlxBuffer,
    h0: &MlxBuffer,
    o: &MlxBuffer,
    final_state: &MlxBuffer,
) -> Result<()> {
    if p.b == 0 || p.t == 0 || p.hg == 0 || p.h == 0 || p.k == 0 || p.v == 0 || p.bt == 0 {
        return Err(MlxError::InvalidArgument(
            "chunk_gated_delta_rule_fwd: all dims must be > 0".into(),
        ));
    }
    if p.h % p.hg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_gated_delta_rule_fwd: H ({}) must be a multiple of Hg ({})",
            p.h, p.hg
        )));
    }
    // Orchestrator dispatches inter_state and chunk_o, both of which require
    // K==128 exactly (their simdgroup_matrix MMA K-tile loops are compile-time
    // hard-coded at 16 = K=128/8). Tighten orchestrator validation to match,
    // so misuse surfaces here with a clear error rather than as a sub-kernel
    // OOB or rejection cascade. Codex iter-7 audit (2026-04-27).
    if p.k != MAX_K {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_gated_delta_rule_fwd: K ({}) must equal MAX_K = {} exactly. \
             Sub-kernels inter_state and chunk_o have compile-time-fixed 16 \
             K-tiles in their simdgroup_matrix MMA loops; runtime K bounds \
             defeat MMA scheduling (3.15× regression measured). To support \
             other K values, port FLA's b_h1..b_h4 bank-split.",
            p.k, MAX_K
        )));
    }
    if p.v > MAX_V {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_gated_delta_rule_fwd: V ({}) exceeds chunk-pipeline cap \
             (MAX_V = {})",
            p.v, MAX_V
        )));
    }
    if p.bt != FIXED_BT {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_gated_delta_rule_fwd (iter 4): bt must be {} (got {})",
            FIXED_BT, p.bt
        )));
    }
    if p.t % p.bt != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_gated_delta_rule_fwd (iter 4): t ({}) must be a multiple of bt ({})",
            p.t, p.bt
        )));
    }
    if !p.scale.is_finite() {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_gated_delta_rule_fwd: scale must be finite (got {})",
            p.scale
        )));
    }

    // Buffer-shape + dtype checks (mirror sub-kernel checks but at the
    // orchestrator boundary — fast-fails before any GPU work).
    let q_elems = (p.b * p.t * p.hg * p.k) as usize;
    let k_elems = (p.b * p.t * p.hg * p.k) as usize;
    let v_elems = (p.b * p.t * p.h * p.v) as usize;
    let g_elems = (p.b * p.t * p.h) as usize;
    let beta_elems = (p.b * p.t * p.h) as usize;
    let h0_elems = (p.b * p.h * p.v * p.k) as usize;
    let o_elems = (p.b * p.t * p.h * p.v) as usize;
    let final_state_elems = (p.b * p.h * p.v * p.k) as usize;

    let bf16_inputs: [(&str, &MlxBuffer, usize); 4] = [
        ("q", q, q_elems),
        ("k", k, k_elems),
        ("v", v, v_elems),
        ("o", o, o_elems),
    ];
    for (name, buf, exp) in bf16_inputs {
        if buf.element_count() != exp {
            return Err(MlxError::InvalidArgument(format!(
                "chunk_gated_delta_rule_fwd: {} element count {} != expected {}",
                name,
                buf.element_count(),
                exp
            )));
        }
        if buf.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "chunk_gated_delta_rule_fwd: {} must be bf16 (got {})",
                name,
                buf.dtype()
            )));
        }
    }

    let f32_inputs: [(&str, &MlxBuffer, usize); 4] = [
        ("g_log_decay", g_log_decay, g_elems),
        ("beta", beta, beta_elems),
        ("h0", h0, h0_elems),
        ("final_state", final_state, final_state_elems),
    ];
    for (name, buf, exp) in f32_inputs {
        if buf.element_count() != exp {
            return Err(MlxError::InvalidArgument(format!(
                "chunk_gated_delta_rule_fwd: {} element count {} != expected {}",
                name,
                buf.element_count(),
                exp
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "chunk_gated_delta_rule_fwd: {} must be f32 (got {})",
                name,
                buf.dtype()
            )));
        }
    }

    Ok(())
}

/// Build the 5-u32 params buffer for `chunk_local_cumsum_g_f32`:
/// `[B, T, H, BT, NT]`.
fn build_chunk_local_cumsum_g_params(
    device: &MlxDevice,
    p: &ChunkGatedDeltaRuleParams,
) -> Result<MlxBuffer> {
    let mut buf = device.alloc_buffer(5 * 4, DType::U32, vec![5])?;
    {
        let s = buf.as_mut_slice::<u32>()?;
        s[0] = p.b;
        s[1] = p.t;
        s[2] = p.h;
        s[3] = p.bt;
        s[4] = p.num_chunks();
    }
    Ok(buf)
}

/// Build the 2-f32 params buffer for `dispatch_l2_norm`: `[eps, dim]`.
fn build_l2_norm_params(device: &MlxDevice, eps: f32, dim: u32) -> Result<MlxBuffer> {
    let mut buf = device.alloc_buffer(2 * 4, DType::F32, vec![2])?;
    {
        let s = buf.as_mut_slice::<f32>()?;
        s[0] = eps;
        s[1] = dim as f32;
    }
    Ok(buf)
}

/// Dispatch the chunk-local cumsum on g (B*NT*H independent chunks).
fn dispatch_chunk_local_cumsum_g(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    g_in: &MlxBuffer,
    g_out: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: &ChunkGatedDeltaRuleParams,
) -> Result<()> {
    let pipeline = registry.get_pipeline("chunk_local_cumsum_g_f32", device)?;
    let nt = p.num_chunks() as u64;
    // Wave 5b.1 iter 4.5 (M1): one threadgroup per (i_t, h, b) block.
    // The shader reads tg_pos.{y,z} for the (h, b*NT) coords and ignores
    // tg_pos.x — only thread 0 within the threadgroup walks the BT-long
    // serial scan (see chunk_local_cumsum_g.metal:39-68). Previously we
    // launched grid_x = BT = 64, which produced 64 redundant threadgroups
    // per block all writing the same outputs.
    let grid_tgs = MTLSize::new(1, p.h as u64, (p.b as u64) * nt);
    // Threadgroup: BT lanes wide; the kernel has only thread 0 do the
    // serial scan. Iter 5 perf can lift to a Hillis-Steele scan.
    let tg = MTLSize::new(p.bt as u64, 1, 1);
    encoder.encode_threadgroups(
        pipeline,
        &[(0, g_in), (1, g_out), (2, params_buf)],
        grid_tgs,
        tg,
    );
    Ok(())
}

/// Dispatch the end-to-end chunk-parallel gated delta-rule forward pass.
///
/// All intermediate buffers are allocated by this function; only the
/// caller's input + output buffers cross the call boundary. Use this
/// orchestrator as the user-facing API for the chunk-parallel path.
///
/// Inputs:
/// - `q`           : `[B, T, Hg, K]`  bf16  — query (raw, pre-l2norm if `use_qk_l2norm`).
/// - `k`           : `[B, T, Hg, K]`  bf16  — key   (same).
/// - `v`           : `[B, T, H,  V]`  bf16  — value.
/// - `g_log_decay` : `[B, T, H]`      f32   — RAW per-token log-decay (NOT cumsumed).
/// - `beta`        : `[B, T, H]`      f32   — per-token write-strength.
/// - `h0`          : `[B, H, V, K]`   f32   — initial state.
///
/// Outputs (caller-allocated):
/// - `o`           : `[B, T, H, V]`   bf16  — per-token output.
/// - `final_state` : `[B, H, V, K]`   f32   — state after the final chunk.
///
/// On commit, both output buffers are populated.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_chunk_gated_delta_rule_fwd(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g_log_decay: &MlxBuffer,
    beta: &MlxBuffer,
    h0: &MlxBuffer,
    o: &MlxBuffer,
    final_state: &MlxBuffer,
    p: ChunkGatedDeltaRuleParams,
) -> Result<()> {
    validate(&p, q, k, v, g_log_decay, beta, h0, o, final_state)?;

    let metal_device = device.metal_device();
    let nt = p.num_chunks();

    // -----------------------------------------------------------------
    // Stage 1: optional l2_norm on q + k (Qwen3.6 default = true).
    //
    // FLA chunk.py:106-108 normalizes BOTH q and k along the last axis (K).
    // The normalized tensors are then used for kkt + recompute_w_u + chunk_o,
    // AND for the inter-state kernel's GQA broadcast. We must use the
    // normed copies for ALL downstream stages.
    // -----------------------------------------------------------------
    let q_qk_elems = (p.b * p.t * p.hg * p.k) as usize;
    let q_normed_buf;
    let k_normed_buf;
    let q_for_pipeline: &MlxBuffer;
    let k_for_pipeline: &MlxBuffer;

    if p.use_qk_l2norm {
        q_normed_buf =
            device.alloc_buffer(q_qk_elems * 2, DType::BF16, vec![q_qk_elems])?;
        k_normed_buf =
            device.alloc_buffer(q_qk_elems * 2, DType::BF16, vec![q_qk_elems])?;

        let l2_params = build_l2_norm_params(device, L2_NORM_EPS, p.k)?;
        // Each (b, t, hg) row is a length-K vector to normalize.
        let rows = p.b * p.t * p.hg;
        dispatch_l2_norm(
            encoder, registry, metal_device, q, &q_normed_buf, &l2_params, rows, p.k,
        )?;
        encoder.memory_barrier();
        dispatch_l2_norm(
            encoder, registry, metal_device, k, &k_normed_buf, &l2_params, rows, p.k,
        )?;
        encoder.memory_barrier();

        q_for_pipeline = &q_normed_buf;
        k_for_pipeline = &k_normed_buf;
    } else {
        q_for_pipeline = q;
        k_for_pipeline = k;
    }

    // -----------------------------------------------------------------
    // Stage 2: chunk_local_cumsum on g_log_decay -> g_cumsum.
    // -----------------------------------------------------------------
    let g_elems = (p.b * p.t * p.h) as usize;
    let g_cumsum_buf =
        device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems])?;
    let cumsum_params = build_chunk_local_cumsum_g_params(device, &p)?;
    dispatch_chunk_local_cumsum_g(
        encoder,
        registry,
        metal_device,
        g_log_decay,
        &g_cumsum_buf,
        &cumsum_params,
        &p,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 3: chunk_scaled_dot_kkt -> A_strict [B, T, H, BT].
    // -----------------------------------------------------------------
    let a_elems = (p.b * p.t * p.h * p.bt) as usize;
    let a_strict_buf =
        device.alloc_buffer(a_elems * 4, DType::F32, vec![a_elems])?;
    let kkt_params_value = GatedDeltaNetKktParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        bt: p.bt,
    };
    let kkt_params = build_gated_delta_net_kkt_params(device, kkt_params_value)?;
    dispatch_gated_delta_net_kkt(
        encoder,
        registry,
        metal_device,
        k_for_pipeline,
        beta,
        &g_cumsum_buf,
        &a_strict_buf,
        &kkt_params,
        kkt_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 4: chunk_tri_solve_invert -> A_inv [B, T, H, BT].
    //
    // FLA solve_tril returns `(I + A_strict)^-1`. The dedicated kernel
    // here handles FLA's row-stride-H*BT layout directly (see
    // chunk_gated_delta_rule_tri_solve_invert.rs for the layout-vs-
    // batched-tri_solve trade-off rationale).
    // -----------------------------------------------------------------
    let a_inv_buf =
        device.alloc_buffer(a_elems * 4, DType::F32, vec![a_elems])?;
    let invert_params_value = ChunkTriSolveInvertParams {
        b: p.b,
        t: p.t,
        h: p.h,
        bt: p.bt,
    };
    let invert_params = build_chunk_tri_solve_invert_params(device, invert_params_value)?;
    dispatch_chunk_tri_solve_invert(
        encoder,
        registry,
        metal_device,
        &a_strict_buf,
        &a_inv_buf,
        &invert_params,
        invert_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 5: recompute_w_u -> w [B, T, H, K] bf16, u [B, T, H, V] bf16.
    //
    // PITFALL guard: v passed to recompute_w_u is the ORIGINAL v (the
    // input to the orchestrator), NOT v_new — v_new comes from stage 6.
    // -----------------------------------------------------------------
    let w_elems = (p.b * p.t * p.h * p.k) as usize;
    let u_elems = (p.b * p.t * p.h * p.v) as usize;
    let w_buf = device.alloc_buffer(w_elems * 2, DType::BF16, vec![w_elems])?;
    let u_buf = device.alloc_buffer(u_elems * 2, DType::BF16, vec![u_elems])?;
    let recompute_wu_params_value = GatedDeltaNetRecomputeWuParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        v: p.v,
        bt: p.bt,
    };
    let recompute_wu_params =
        build_gated_delta_net_recompute_wu_params(device, recompute_wu_params_value)?;
    dispatch_gated_delta_net_recompute_wu(
        encoder,
        registry,
        metal_device,
        k_for_pipeline,
        v,
        beta,
        &g_cumsum_buf,
        &a_inv_buf,
        &w_buf,
        &u_buf,
        &recompute_wu_params,
        recompute_wu_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 6: chunk_inter_state -> h [B, NT, H, V, K] bf16,
    //                              v_new [B, T, H, V] bf16,
    //                              final_state [B, H, V, K] f32.
    // -----------------------------------------------------------------
    let h_elems = (p.b * nt * p.h * p.v * p.k) as usize;
    let v_new_elems = (p.b * p.t * p.h * p.v) as usize;
    let h_buf = device.alloc_buffer(h_elems * 2, DType::BF16, vec![h_elems])?;
    let v_new_buf =
        device.alloc_buffer(v_new_elems * 2, DType::BF16, vec![v_new_elems])?;
    let chunk_params_value = GatedDeltaNetChunkParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        v: p.v,
        bt: p.bt,
    };
    let chunk_params = build_gated_delta_net_chunk_params(device, chunk_params_value)?;
    dispatch_gated_delta_net_chunk_inter_state(
        encoder,
        registry,
        metal_device,
        k_for_pipeline,
        &w_buf,
        &u_buf,
        &g_cumsum_buf,
        h0,
        &h_buf,
        &v_new_buf,
        final_state,
        &chunk_params,
        chunk_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 7: chunk_fwd_o -> o [B, T, H, V] bf16.
    //
    // PITFALL guard: g passed here is g_cumsum (NOT raw log-decay), and
    // v passed here is v_new (NOT the original v). Both are produced by
    // earlier stages.
    // -----------------------------------------------------------------
    let chunk_o_params_value = GatedDeltaNetChunkOParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        v: p.v,
        bt: p.bt,
        scale: p.scale,
    };
    let chunk_o_params =
        build_gated_delta_net_chunk_o_params(device, chunk_o_params_value)?;
    dispatch_gated_delta_net_chunk_o(
        encoder,
        registry,
        metal_device,
        q_for_pipeline,
        k_for_pipeline,
        &v_new_buf,
        &h_buf,
        &g_cumsum_buf,
        o,
        &chunk_o_params,
        chunk_o_params_value,
    )?;

    Ok(())
}

// =====================================================================
// ADR-015 iter83 — ChunkInternalArena (caller-owned scratch arena for the
// internal allocations of `dispatch_chunk_gated_delta_rule_fwd`).
//
// Mirrors the iter78 hf2q-side ChunkAllocsArena pattern at the next
// nesting level: lifts the orchestrator's per-call internal scratches
// (g_cumsum, A_strict, A_inv, w, u, h, v_new, plus the four small u32
// param buffers) to caller scope so they are allocated ONCE per prefill
// and reused across all DN-layer dispatches.
//
// Per the iter81 audit (`/tmp/cfa-iter81/research/MLX-NATIVE-AUDIT.md`):
// > chunk_gated_delta_rule.rs:400-543 has 10 alloc_buffer sites per call
// > (~33MB memset/layer at pp4096); could be a ChunkInternalArena win
// > for chunk-engaged workloads but fails predicate #3 on default
// > (pp4123 % 64 != 0).
//
// Workload-conditional WIN by design: NEUTRAL on default pp4123 (chunk
// path doesn't fire), expected wall improvement on chunk-engaged
// 4096-token workload mirroring iter78 (-50 to -100 ms).
//
// # Production scope
//
// hf2q always passes `use_qk_l2norm = false` to
// `dispatch_chunk_gated_delta_rule_fwd` (the wrapper pre-applies l2-norm
// using its own `q_normed`/`k_normed` slots). The l2-norm branch
// (q_normed_buf, k_normed_buf, l2_params) therefore never fires in
// production and is NOT part of this arena. Operator-level callers that
// pass `use_qk_l2norm = true` (e.g., the `_oracle` test path) MUST
// continue using the non-arena variant `dispatch_chunk_gated_delta_rule_fwd`.
//
// # iter58b residency contract
//
// All arena slots are caller-owned; they outlive every per-layer encoder
// commit inside the prefill loop. No `MlxBuffer::Drop` fires until the
// arena is dropped at the end of the prefill, AFTER the final
// `commit_and_wait_labeled` at the output head — same residency-
// rescission protection as DnPrefillArena / FaPrefillArena /
// ChunkAllocsArena.
// =====================================================================

/// Caller-owned scratch arena for the internal allocations of
/// [`dispatch_chunk_gated_delta_rule_fwd`].
///
/// Allocate once per prefill, validate against per-layer call shape via
/// [`ChunkInternalArena::validate_fits`], pass into
/// [`dispatch_chunk_gated_delta_rule_fwd_with_arena`].
///
/// # Memory footprint at apex pp4096 (B=1, T=4096, Hg=H=32, K=V=128, BT=64)
///
/// | Slot | DType | Bytes |
/// |---|---|---:|
/// | `g_cumsum_buf` | F32 | B*T*H*4 = 524 KB |
/// | `a_strict_buf` | F32 | B*T*H*BT*4 = 33.6 MB |
/// | `a_inv_buf` | F32 | B*T*H*BT*4 = 33.6 MB |
/// | `w_buf` | BF16 | B*T*H*K*2 = 33.6 MB |
/// | `u_buf` | BF16 | B*T*H*V*2 = 33.6 MB |
/// | `h_buf` | BF16 | B*NT*H*V*K*2 = 67.1 MB |
/// | `v_new_buf` | BF16 | B*T*H*V*2 = 33.6 MB |
/// | param bufs (×5) | U32/F32 | ~120 B total |
/// | **Total** | | **~235 MB** |
///
/// Allocated ONCE per prefill, reused across all DN layers (30 in
/// Qwen3.6 35B-A3B). Replaces ~30 × 235 MB = 7.0 GB of per-layer alloc
/// + memset on M5 Max unified memory.
pub struct ChunkInternalArena {
    /// `[B, T, H]` F32 — chunk-local cumsum on g_log_decay.
    pub g_cumsum_buf: MlxBuffer,
    /// `[B, T, H, BT]` F32 — strict lower-triangular A.
    pub a_strict_buf: MlxBuffer,
    /// `[B, T, H, BT]` F32 — `(I + A_strict)^-1`.
    pub a_inv_buf: MlxBuffer,
    /// `[B, T, H, K]` BF16 — `w` from `recompute_w_u`.
    pub w_buf: MlxBuffer,
    /// `[B, T, H, V]` BF16 — `u` from `recompute_w_u`.
    pub u_buf: MlxBuffer,
    /// `[B, NT, H, V, K]` BF16 — per-chunk hidden states `h`.
    pub h_buf: MlxBuffer,
    /// `[B, T, H, V]` BF16 — `v_new` from `chunk_inter_state`.
    pub v_new_buf: MlxBuffer,

    // ---- Small param buffers (caller-owned to avoid per-call alloc) ----
    /// 5-u32 cumsum params: `[B, T, H, BT, NT]`.
    pub cumsum_params_buf: MlxBuffer,
    /// 6-u32 kkt params (size determined by sub-kernel; see
    /// `build_gated_delta_net_kkt_params`).
    pub kkt_params_buf: MlxBuffer,
    /// 4-u32 invert params: `[B, T, H, BT]`.
    pub invert_params_buf: MlxBuffer,
    /// 7-u32 recompute_wu params.
    pub recompute_wu_params_buf: MlxBuffer,
    /// 7-u32 chunk params (chunk_inter_state).
    pub chunk_params_buf: MlxBuffer,
    /// 8-u32+f32 chunk_o params (with scale).
    pub chunk_o_params_buf: MlxBuffer,

    // ---- Capacity bookkeeping ----
    /// `B` value the arena was allocated for.
    pub b_capacity: u32,
    /// `T` (seq_len) value the arena was allocated for.
    pub t_capacity: u32,
    /// `Hg` value the arena was allocated for.
    pub hg_capacity: u32,
    /// `H` value the arena was allocated for.
    pub h_capacity: u32,
    /// `K` value the arena was allocated for.
    pub k_capacity: u32,
    /// `V` value the arena was allocated for.
    pub v_capacity: u32,
    /// `BT` value the arena was allocated for.
    pub bt_capacity: u32,
}

impl ChunkInternalArena {
    /// Allocate all internal scratches sized for a single prefill pass.
    ///
    /// # Arguments
    ///
    /// Sized per [`ChunkGatedDeltaRuleParams`] dims. Caller MUST pass
    /// the actual per-call dims; the arena enforces equality on each
    /// dispatch via [`Self::validate_fits`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if any dimension is zero or any
    /// [`MlxDevice::alloc_buffer`] call fails. Also returns `Err` for
    /// the same K/V/BT range as
    /// [`ChunkGatedDeltaRuleParams`] (K==MAX_K, V<=MAX_V, BT==FIXED_BT).
    pub fn new(
        device: &MlxDevice,
        b: u32,
        t: u32,
        hg: u32,
        h: u32,
        k: u32,
        v: u32,
        bt: u32,
    ) -> Result<Self> {
        if b == 0 || t == 0 || hg == 0 || h == 0 || k == 0 || v == 0 || bt == 0 {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: zero dim b={} t={} hg={} h={} k={} v={} bt={}",
                b, t, hg, h, k, v, bt,
            )));
        }
        if h % hg != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: H ({}) must be a multiple of Hg ({})",
                h, hg
            )));
        }
        if k != MAX_K {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: K ({}) must equal MAX_K = {} \
                 (sub-kernel hard constraint)",
                k, MAX_K
            )));
        }
        if v > MAX_V {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: V ({}) > MAX_V ({})",
                v, MAX_V
            )));
        }
        if bt != FIXED_BT {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: bt ({}) must equal FIXED_BT ({})",
                bt, FIXED_BT
            )));
        }
        if t % bt != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: t ({}) must be a multiple of bt ({})",
                t, bt
            )));
        }

        let nt = t.div_ceil(bt);
        let g_elems = (b * t * h) as usize;
        let a_elems = (b * t * h * bt) as usize;
        let w_elems = (b * t * h * k) as usize;
        let u_elems = (b * t * h * v) as usize;
        let h_elems = (b * nt * h * v * k) as usize;
        let v_new_elems = (b * t * h * v) as usize;

        let g_cumsum_buf = device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems])?;
        let a_strict_buf = device.alloc_buffer(a_elems * 4, DType::F32, vec![a_elems])?;
        let a_inv_buf = device.alloc_buffer(a_elems * 4, DType::F32, vec![a_elems])?;
        let w_buf = device.alloc_buffer(w_elems * 2, DType::BF16, vec![w_elems])?;
        let u_buf = device.alloc_buffer(u_elems * 2, DType::BF16, vec![u_elems])?;
        let h_buf = device.alloc_buffer(h_elems * 2, DType::BF16, vec![h_elems])?;
        let v_new_buf = device.alloc_buffer(v_new_elems * 2, DType::BF16, vec![v_new_elems])?;

        // Pre-allocate the small param buffers. They are filled per-call
        // in `dispatch_chunk_gated_delta_rule_fwd_with_arena` since
        // sub-kernel param structs encode per-call shape data identically
        // across layers in a single prefill, but defensive re-fill on
        // each call costs ~50 ns and avoids any risk of shape drift.
        let cumsum_params_buf = device.alloc_buffer(5 * 4, DType::U32, vec![5])?;
        let kkt_params_buf =
            build_gated_delta_net_kkt_params(device, GatedDeltaNetKktParams {
                b,
                t,
                hg,
                h,
                k,
                bt,
            })?;
        let invert_params_buf =
            build_chunk_tri_solve_invert_params(device, ChunkTriSolveInvertParams {
                b,
                t,
                h,
                bt,
            })?;
        let recompute_wu_params_buf = build_gated_delta_net_recompute_wu_params(
            device,
            GatedDeltaNetRecomputeWuParams {
                b,
                t,
                hg,
                h,
                k,
                v,
                bt,
            },
        )?;
        let chunk_params_buf = build_gated_delta_net_chunk_params(
            device,
            GatedDeltaNetChunkParams {
                b,
                t,
                hg,
                h,
                k,
                v,
                bt,
            },
        )?;
        // chunk_o needs a scale param; we pre-build with scale=1.0 and
        // OVERWRITE per-call inside the _with_arena dispatch since the
        // caller's `p.scale` is the only field that varies (and is
        // passed in via params). For Qwen3.6 35B-A3B prefill, scale is
        // constant 1.0 across layers but we keep the per-call rebuild
        // safety pattern of the non-arena variant.
        let chunk_o_params_buf = build_gated_delta_net_chunk_o_params(
            device,
            GatedDeltaNetChunkOParams {
                b,
                t,
                hg,
                h,
                k,
                v,
                bt,
                scale: 1.0,
            },
        )?;

        // Pre-fill the cumsum params (the only one we don't fill via a
        // sub-kernel build_*_params helper).
        let mut cumsum_params_buf = cumsum_params_buf;
        {
            let s = cumsum_params_buf.as_mut_slice::<u32>()?;
            s[0] = b;
            s[1] = t;
            s[2] = h;
            s[3] = bt;
            s[4] = nt;
        }

        Ok(Self {
            g_cumsum_buf,
            a_strict_buf,
            a_inv_buf,
            w_buf,
            u_buf,
            h_buf,
            v_new_buf,
            cumsum_params_buf,
            kkt_params_buf,
            invert_params_buf,
            recompute_wu_params_buf,
            chunk_params_buf,
            chunk_o_params_buf,
            b_capacity: b,
            t_capacity: t,
            hg_capacity: hg,
            h_capacity: h,
            k_capacity: k,
            v_capacity: v,
            bt_capacity: bt,
        })
    }

    /// Validate that a per-call shape matches the arena's allocated capacity.
    ///
    /// Equality is required on every dim — the arena slots are sized
    /// exactly for the prefill shape, and Qwen3.6 35B-A3B is uniform
    /// across DN layers (same n_v_heads / d_k / d_v throughout).
    ///
    /// # Errors
    ///
    /// Returns `Err` if any dim differs from the capacity recorded at
    /// [`Self::new`] time.
    pub fn validate_fits(
        &self,
        b: u32,
        t: u32,
        hg: u32,
        h: u32,
        k: u32,
        v: u32,
        bt: u32,
    ) -> Result<()> {
        if b != self.b_capacity
            || t != self.t_capacity
            || hg != self.hg_capacity
            || h != self.h_capacity
            || k != self.k_capacity
            || v != self.v_capacity
            || bt != self.bt_capacity
        {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::validate_fits: shape mismatch — \
                 capacity (b={}, t={}, hg={}, h={}, k={}, v={}, bt={}) \
                 vs call (b={}, t={}, hg={}, h={}, k={}, v={}, bt={})",
                self.b_capacity,
                self.t_capacity,
                self.hg_capacity,
                self.h_capacity,
                self.k_capacity,
                self.v_capacity,
                self.bt_capacity,
                b,
                t,
                hg,
                h,
                k,
                v,
                bt,
            )));
        }
        Ok(())
    }
}

/// Arena-aware variant of [`dispatch_chunk_gated_delta_rule_fwd`].
///
/// Threads through caller-owned [`ChunkInternalArena`] slots instead of
/// allocating fresh `MlxBuffer`s for each of the 7 large internal
/// scratches + 5 small param buffers. Otherwise byte-identical:
/// same encoder choreography, same memory_barrier placement, same
/// `commit_and_wait` discipline (driven by the caller's encoder).
///
/// # Constraint
///
/// Reserved for `p.use_qk_l2norm == false`. Production hf2q chunk path
/// pre-applies l2-norm in the wrapper and passes false; the
/// l2-norm branch's `q_normed` / `k_normed` allocations are NOT part of
/// the arena (they sit OUTSIDE the iter83 scope by design — see module
/// docs). For `use_qk_l2norm = true` callers must use the non-arena
/// [`dispatch_chunk_gated_delta_rule_fwd`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_chunk_gated_delta_rule_fwd_with_arena(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    g_log_decay: &MlxBuffer,
    beta: &MlxBuffer,
    h0: &MlxBuffer,
    o: &MlxBuffer,
    final_state: &MlxBuffer,
    arena: &mut ChunkInternalArena,
    p: ChunkGatedDeltaRuleParams,
) -> Result<()> {
    if p.use_qk_l2norm {
        return Err(MlxError::InvalidArgument(
            "dispatch_chunk_gated_delta_rule_fwd_with_arena: use_qk_l2norm=true is \
             reserved for the non-arena variant. Pre-apply l2-norm in the wrapper \
             and pass use_qk_l2norm=false."
                .into(),
        ));
    }
    validate(&p, q, k, v, g_log_decay, beta, h0, o, final_state)?;
    arena.validate_fits(p.b, p.t, p.hg, p.h, p.k, p.v, p.bt)?;

    let metal_device = device.metal_device();

    // Stage 1 SKIPPED — l2-norm is caller-side (use_qk_l2norm=false).
    let q_for_pipeline: &MlxBuffer = q;
    let k_for_pipeline: &MlxBuffer = k;

    // -----------------------------------------------------------------
    // Stage 2: chunk_local_cumsum on g_log_decay -> g_cumsum (arena slot).
    // The cumsum_params buffer was pre-filled in `ChunkInternalArena::new`
    // (no per-call rebuild needed; b/t/h/bt/nt are arena-fixed).
    // -----------------------------------------------------------------
    dispatch_chunk_local_cumsum_g(
        encoder,
        registry,
        metal_device,
        g_log_decay,
        &arena.g_cumsum_buf,
        &arena.cumsum_params_buf,
        &p,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 3: chunk_scaled_dot_kkt -> A_strict (arena slot).
    // -----------------------------------------------------------------
    let kkt_params_value = GatedDeltaNetKktParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        bt: p.bt,
    };
    dispatch_gated_delta_net_kkt(
        encoder,
        registry,
        metal_device,
        k_for_pipeline,
        beta,
        &arena.g_cumsum_buf,
        &arena.a_strict_buf,
        &arena.kkt_params_buf,
        kkt_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 4: chunk_tri_solve_invert -> A_inv (arena slot).
    // -----------------------------------------------------------------
    let invert_params_value = ChunkTriSolveInvertParams {
        b: p.b,
        t: p.t,
        h: p.h,
        bt: p.bt,
    };
    dispatch_chunk_tri_solve_invert(
        encoder,
        registry,
        metal_device,
        &arena.a_strict_buf,
        &arena.a_inv_buf,
        &arena.invert_params_buf,
        invert_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 5: recompute_w_u -> w + u (arena slots).
    // -----------------------------------------------------------------
    let recompute_wu_params_value = GatedDeltaNetRecomputeWuParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        v: p.v,
        bt: p.bt,
    };
    dispatch_gated_delta_net_recompute_wu(
        encoder,
        registry,
        metal_device,
        k_for_pipeline,
        v,
        beta,
        &arena.g_cumsum_buf,
        &arena.a_inv_buf,
        &arena.w_buf,
        &arena.u_buf,
        &arena.recompute_wu_params_buf,
        recompute_wu_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 6: chunk_inter_state -> h + v_new (arena slots) + final_state (caller).
    // -----------------------------------------------------------------
    let chunk_params_value = GatedDeltaNetChunkParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        v: p.v,
        bt: p.bt,
    };
    dispatch_gated_delta_net_chunk_inter_state(
        encoder,
        registry,
        metal_device,
        k_for_pipeline,
        &arena.w_buf,
        &arena.u_buf,
        &arena.g_cumsum_buf,
        h0,
        &arena.h_buf,
        &arena.v_new_buf,
        final_state,
        &arena.chunk_params_buf,
        chunk_params_value,
    )?;
    encoder.memory_barrier();

    // -----------------------------------------------------------------
    // Stage 7: chunk_fwd_o -> o (caller-provided).
    //
    // Rebuild chunk_o params to capture caller-provided p.scale (the
    // only field that can vary across calls; everything else is arena-
    // fixed). Reuse the arena slot — overwrite the contents in-place
    // via build_*_params write semantics.
    // -----------------------------------------------------------------
    let chunk_o_params_value = GatedDeltaNetChunkOParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k,
        v: p.v,
        bt: p.bt,
        scale: p.scale,
    };
    // Re-emit chunk_o params into the arena-owned slot so changes in
    // p.scale across calls are honored. The slot was sized correctly
    // at arena allocation time (44 bytes / 11 u32 — matches
    // build_gated_delta_net_chunk_o_params layout); we overwrite its
    // contents in-place to preserve the caller-owned arena lifetime
    // contract (no MlxBuffer Drop until the prefill's terminal
    // commit_and_wait). Layout is byte-identical to
    // gated_delta_net_chunk_o.rs:360-378.
    debug_assert!(
        arena.chunk_o_params_buf.byte_len() >= 11 * 4,
        "chunk_o_params_buf too small: {}",
        arena.chunk_o_params_buf.byte_len()
    );
    {
        let s = arena.chunk_o_params_buf.as_mut_slice::<u32>()?;
        s[0] = chunk_o_params_value.b;
        s[1] = chunk_o_params_value.t;
        s[2] = chunk_o_params_value.hg;
        s[3] = chunk_o_params_value.h;
        s[4] = chunk_o_params_value.k;
        s[5] = chunk_o_params_value.v;
        s[6] = chunk_o_params_value.bt;
        s[7] = chunk_o_params_value.num_chunks();
        s[8] = crate::ops::gated_delta_net_chunk_o::DEFAULT_BK;
        s[9] = crate::ops::gated_delta_net_chunk_o::DEFAULT_BV;
        // s[10] is the f32 scale; reinterpret as u32 bit pattern.
        s[10] = chunk_o_params_value.scale.to_bits();
    }
    dispatch_gated_delta_net_chunk_o(
        encoder,
        registry,
        metal_device,
        q_for_pipeline,
        k_for_pipeline,
        &arena.v_new_buf,
        &arena.h_buf,
        &arena.g_cumsum_buf,
        o,
        &arena.chunk_o_params_buf,
        chunk_o_params_value,
    )?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    //! Wave 5b.1 iter 4 — K=256 clamp unit test for the orchestrator.
    //!
    //! Mirrors the iter-2.5 + iter-3 K-cap discipline for the four
    //! sub-kernels (chunk_inter_state, kkt, recompute_w_u, chunk_fwd_o);
    //! closes the validation unit-test gap proactively for the new
    //! orchestrator entry point.
    use super::*;
    use crate::MlxDevice;

    /// Allocate a 1-element dummy buffer of the given dtype. The K-cap check
    /// in the orchestrator's `validate` fires before any buffer-size check,
    /// so these placeholder buffers exercise the error path without
    /// allocating GBs.
    fn dummy_buf(device: &MlxDevice, dtype: DType) -> MlxBuffer {
        device.alloc_buffer(2, dtype, vec![1]).expect("alloc dummy")
    }

    #[test]
    fn validate_rejects_k_above_max() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        let q_buf = dummy_buf(&device, DType::BF16);
        let k_buf = dummy_buf(&device, DType::BF16);
        let v_buf = dummy_buf(&device, DType::BF16);
        let g_buf = dummy_buf(&device, DType::F32);
        let beta_buf = dummy_buf(&device, DType::F32);
        let h0_buf = dummy_buf(&device, DType::F32);
        let o_buf = dummy_buf(&device, DType::BF16);
        let final_state_buf = dummy_buf(&device, DType::F32);

        let p = ChunkGatedDeltaRuleParams {
            b: 1,
            t: 128,
            hg: 2,
            h: 4,
            k: 256, // > MAX_K (192) — must reject.
            v: 128,
            bt: 64,
            scale: (128f32).powf(-0.5),
            use_qk_l2norm: true,
        };

        let err = validate(
            &p,
            &q_buf,
            &k_buf,
            &v_buf,
            &g_buf,
            &beta_buf,
            &h0_buf,
            &o_buf,
            &final_state_buf,
        )
        .expect_err("validate must reject K=256");
        let msg = err.to_string();
        assert!(
            msg.contains("256"),
            "expected K=256 in error message, got: {msg}"
        );
        assert!(
            msg.contains("MAX_K = 128") || msg.contains("MAX_K=128"),
            "expected explicit MAX_K=128 in error (orchestrator inherits sub-kernel \
             K==128-exact constraint per Wave 5b.2 iter 2.5), got: {msg}"
        );
        assert!(
            msg.contains("must equal") || msg.contains("hard-coded"),
            "expected exact-equality wording in error, got: {msg}"
        );
    }

    // -------------------------------------------------------------------
    // ADR-015 iter83 — ChunkInternalArena unit tests.
    // -------------------------------------------------------------------

    #[test]
    fn arena_new_apex_pp4096_shape_succeeds() {
        // Apex pp4096 dims: B=1, T=4096, Hg=H=32 (Qwen3.6 35B-A3B is
        // GQA-tile-expanded by hf2q wrapper to Hg==H), K=V=128, BT=64.
        let device = MlxDevice::new().expect("MlxDevice::new");
        let arena = ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 128, 64)
            .expect("arena alloc apex pp4096");
        assert_eq!(arena.b_capacity, 1);
        assert_eq!(arena.t_capacity, 4096);
        assert_eq!(arena.hg_capacity, 32);
        assert_eq!(arena.h_capacity, 32);
        assert_eq!(arena.k_capacity, 128);
        assert_eq!(arena.v_capacity, 128);
        assert_eq!(arena.bt_capacity, 64);
        // Sanity-check sizes for the largest slot (h_buf).
        // NT = 4096/64 = 64; h = 1 * 64 * 32 * 128 * 128 * 2 = 67_108_864 bytes.
        assert_eq!(arena.h_buf.byte_len(), 1 * 64 * 32 * 128 * 128 * 2);
    }

    #[test]
    fn arena_new_rejects_zero_dim() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        assert!(ChunkInternalArena::new(&device, 0, 4096, 32, 32, 128, 128, 64).is_err());
        assert!(ChunkInternalArena::new(&device, 1, 0, 32, 32, 128, 128, 64).is_err());
        assert!(ChunkInternalArena::new(&device, 1, 4096, 0, 32, 128, 128, 64).is_err());
        assert!(ChunkInternalArena::new(&device, 1, 4096, 32, 0, 128, 128, 64).is_err());
        assert!(ChunkInternalArena::new(&device, 1, 4096, 32, 32, 0, 128, 64).is_err());
        assert!(ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 0, 64).is_err());
        assert!(ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 128, 0).is_err());
    }

    #[test]
    fn arena_new_enforces_k_eq_max() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        // K != 128 must reject (hard sub-kernel constraint).
        let err = match ChunkInternalArena::new(&device, 1, 4096, 32, 32, 64, 128, 64) {
            Err(e) => e,
            Ok(_) => panic!("arena must reject K!=128"),
        };
        let msg = err.to_string();
        assert!(msg.contains("MAX_K"), "got: {msg}");
    }

    #[test]
    fn arena_new_enforces_bt_eq_fixed() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        // BT != 64 must reject.
        let err = match ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 128, 32) {
            Err(e) => e,
            Ok(_) => panic!("arena must reject bt!=FIXED_BT"),
        };
        let msg = err.to_string();
        assert!(msg.contains("FIXED_BT"), "got: {msg}");
    }

    #[test]
    fn arena_new_enforces_t_multiple_of_bt() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        // T not multiple of bt must reject.
        let err = match ChunkInternalArena::new(&device, 1, 4123, 32, 32, 128, 128, 64) {
            Err(e) => e,
            Ok(_) => panic!("arena must reject t%bt!=0"),
        };
        let msg = err.to_string();
        assert!(msg.contains("multiple"), "got: {msg}");
    }

    #[test]
    fn arena_new_enforces_h_multiple_of_hg() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        // H % Hg != 0 must reject.
        let err = match ChunkInternalArena::new(&device, 1, 4096, 3, 32, 128, 128, 64) {
            Err(e) => e,
            Ok(_) => panic!("arena must reject H%Hg!=0"),
        };
        let msg = err.to_string();
        assert!(msg.contains("multiple"), "got: {msg}");
    }

    #[test]
    fn arena_validate_fits_accepts_exact_match() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        let arena = ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 128, 64)
            .expect("arena alloc");
        arena.validate_fits(1, 4096, 32, 32, 128, 128, 64).expect("exact match accepts");
    }

    #[test]
    fn arena_validate_fits_rejects_drift() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        let arena = ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 128, 64)
            .expect("arena alloc");
        // Any drift on any dim must reject (Qwen3.6 35B-A3B is shape-uniform).
        assert!(arena.validate_fits(2, 4096, 32, 32, 128, 128, 64).is_err());
        assert!(arena.validate_fits(1, 2048, 32, 32, 128, 128, 64).is_err());
        assert!(arena.validate_fits(1, 4096, 16, 32, 128, 128, 64).is_err());
        assert!(arena.validate_fits(1, 4096, 32, 16, 128, 128, 64).is_err());
        assert!(arena.validate_fits(1, 4096, 32, 32, 64, 128, 64).is_err());
        assert!(arena.validate_fits(1, 4096, 32, 32, 128, 64, 64).is_err());
        assert!(arena.validate_fits(1, 4096, 32, 32, 128, 128, 32).is_err());
    }

    #[test]
    fn arena_dispatch_rejects_use_qk_l2norm_true() {
        // The _with_arena variant explicitly rejects use_qk_l2norm=true
        // (l2-norm branch is out-of-scope per iter83 design).
        let device = MlxDevice::new().expect("MlxDevice::new");
        let mut arena = ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 128, 64)
            .expect("arena alloc");
        let mut registry = crate::KernelRegistry::default();
        let mut enc = device.command_encoder().expect("encoder");

        let q = dummy_buf(&device, DType::BF16);
        let k = dummy_buf(&device, DType::BF16);
        let v = dummy_buf(&device, DType::BF16);
        let g = dummy_buf(&device, DType::F32);
        let beta = dummy_buf(&device, DType::F32);
        let h0 = dummy_buf(&device, DType::F32);
        let o = dummy_buf(&device, DType::BF16);
        let final_state = dummy_buf(&device, DType::F32);

        let p = ChunkGatedDeltaRuleParams {
            b: 1,
            t: 4096,
            hg: 32,
            h: 32,
            k: 128,
            v: 128,
            bt: 64,
            scale: 1.0,
            use_qk_l2norm: true, // must reject
        };

        let err = dispatch_chunk_gated_delta_rule_fwd_with_arena(
            &mut enc,
            &mut registry,
            &device,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &h0,
            &o,
            &final_state,
            &mut arena,
            p,
        )
        .expect_err("must reject use_qk_l2norm=true");
        let msg = err.to_string();
        assert!(msg.contains("use_qk_l2norm"), "got: {msg}");
        assert!(msg.contains("non-arena"), "got: {msg}");
    }

    #[test]
    fn arena_chunk_o_params_layout_matches_helper() {
        // Verify the in-place chunk_o param rewrite uses the same byte
        // layout as build_gated_delta_net_chunk_o_params (11 u32 = 44 B).
        let device = MlxDevice::new().expect("MlxDevice::new");
        let arena = ChunkInternalArena::new(&device, 1, 4096, 32, 32, 128, 128, 64)
            .expect("arena alloc");
        // 11 u32 fields × 4 bytes = 44 bytes minimum.
        assert!(
            arena.chunk_o_params_buf.byte_len() >= 11 * 4,
            "chunk_o_params_buf byte_len = {}, expected >= 44",
            arena.chunk_o_params_buf.byte_len()
        );
    }
}
