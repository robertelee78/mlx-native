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

/// Hard cap on per-tile head-dim K — matches all four sub-kernel caps
/// (chunk_inter_state, kkt, recompute_w_u, chunk_fwd_o all use 192).
/// Iter 5+ will autotune past this.
pub const MAX_K: u32 = 192;
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
    if p.k > MAX_K {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_gated_delta_rule_fwd: K ({}) exceeds chunk-pipeline 32 KB \
             threadgroup memory budget; iter-5+ will autotune (MAX_K = {})",
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
    let grid_tgs = MTLSize::new(p.bt as u64, p.h as u64, (p.b as u64) * nt);
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
            msg.contains("32 KB") || msg.contains("threadgroup"),
            "expected threadgroup-memory-budget context in error, got: {msg}"
        );
        assert!(
            msg.contains("MAX_K = 192") || msg.contains("MAX_K=192"),
            "expected explicit MAX_K cap in error, got: {msg}"
        );
    }
}
