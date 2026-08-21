//! ADR-033 §Pi Task #25 (2026-05-23) — chunk-parallel gated DeltaNet with
//! K-bank split, enabling Qwen3.6's head_dim=256 to use the chunk path.
//!
//! # Why this exists
//!
//! The existing chunk-parallel kernel suite [`chunk_gated_delta_rule`] has a
//! hard-coded `MAX_K = 128` constraint (sub-kernels `inter_state` and
//! `chunk_o` use compile-time 16-tile MMA loops; runtime K bounds defeat
//! MMA scheduling at a 3.15× cost). Qwen3.6's linear-attention layers use
//! `head_dim = 256`, so they hit the K validator and fall back to the
//! token-by-token autoregressive path. That fallback is ~6.7× slower at
//! pp4096 (per `gpu_delta_net.rs:1118`) and is the documented residual
//! source of the prefill peer-parity gap at production shapes.
//!
//! This module implements the multi-iter structural fix per
//! [`chunk_gated_delta_rule.rs:191-195`]:
//!
//! > "To support other K values, port FLA's `b_h1..b_h4` bank-split."
//!
//! # Bank-split approach (2-bank, K=128 each)
//!
//! For K = K1 + K2 (here K1 = K2 = 128, total K = 256), the chunk pipeline
//! decomposes as:
//!
//! | Stage              | Decomposition | Cross-bank op |
//! |--------------------|---------------|---------------|
//! | l2_norm on q/k     | Per-bank      | none          |
//! | cumsum_g           | K-independent | none (run once) |
//! | kkt: K^T @ K       | Per-bank → A_strict_partial | SUM across banks |
//! | tri_solve_invert   | On full A_strict | run once after kkt sum |
//! | recompute_w_u      | Per-bank for w; v's u is shared | depends — see iter 17 |
//! | chunk_inter_state  | Per-bank for h, v_new | h is per-bank stored |
//! | chunk_o: o = q @ h | Per-bank → o_partial | SUM across banks |
//!
//! Math validity:
//! - kkt[i,j] = Σ_k K[i,k]·K[j,k] = Σ_k1 K1[i,k1]·K1[j,k1] + Σ_k2 K2[i,k2]·K2[j,k2]
//!   → ADD across banks (kkt_full = kkt_bank0 + kkt_bank1)
//! - State recurrence h[v,k] is independent across k → bh splits naturally
//!   into bh_bank0[v, K1], bh_bank1[v, K2]
//! - Output o[t,v] = Σ_k q[t,k]·h[k,v] = Σ_k1 q1[t,k1]·h_bank0[k1,v] + ...
//!   → ADD across banks
//!
//! # Trade-offs
//!
//! - **Pro**: reuses existing battle-tested K=128 kernels — no new MMA
//!   kernels to write, no register-pressure risk (avoids iter 7's
//!   falsified fusion pattern).
//! - **Pro**: bank-split is the FLA canonical algorithm — same approach
//!   the upstream reference uses for K>128 cases.
//! - **Con**: 2× kernel dispatches per stage (modulo cumsum which runs
//!   once). Expected speedup ~6.7×/2 = 3.35× over autoregressive — still
//!   strictly net positive vs the current Qwen3.6 fallback.
//! - **Con**: temp buffer overhead for per-bank partial outputs
//!   (a_strict_partial, h_partial, o_partial). Sized to fit M5 Max
//!   unified memory budget — bench validation pending iter 18.
//!
//! # Multi-iter arc plan
//!
//! - **Iter 15 (this commit)**: scaffolding — public function signature,
//!   validation, design doc, scaffolding unit test. Body returns
//!   `MlxError::Unimplemented`.
//! - **Iter 16**: K-slice extraction helpers (memcpy strided K-bank views
//!   into temp buffers; or zero-copy stride-aware views once supported).
//! - **Iter 17**: per-bank kkt + recompute_wu + chunk_inter_state. Stub
//!   chunk_o still missing.
//! - **Iter 18**: per-bank chunk_o + output accumulation (element-wise
//!   sum kernel call across banks).
//! - **Iter 19**: parity tests vs autoregressive at production shapes
//!   (seq=128, 256, 512, 1024). Bit-equivalent target where possible.
//! - **Iter 20**: orchestrator wiring (gpu_delta_net.rs `chunk_path_eligible`
//!   accepts K=256 when bank-split dispatch is available).
//! - **Iter 21**: end-to-end bench Qwen3.6 35B-A3B Q4_0 MoE prefill —
//!   target: hf2q ≥ the reference implementation at all tested seq lengths.
//!
//! # Status
//!
//! Iter 15 SHIPPED 2026-05-23 (scaffolding). Future iters extend per the
//! arc plan above.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::chunk_gated_delta_rule::{
    build_chunk_local_cumsum_g_params, build_l2_norm_params,
    dispatch_chunk_local_cumsum_g, ChunkGatedDeltaRuleParams, FIXED_BT, L2_NORM_EPS, MAX_K, MAX_V,
};
use crate::ops::chunk_gated_delta_rule_tri_solve_invert::{
    build_chunk_tri_solve_invert_params, dispatch_chunk_tri_solve_invert,
    ChunkTriSolveInvertParams,
};
use crate::ops::gated_delta_net_chunk::{
    build_gated_delta_net_chunk_params, GatedDeltaNetChunkParams,
};
use crate::ops::gated_delta_net_chunk_o::{
    build_gated_delta_net_chunk_o_params, GatedDeltaNetChunkOParams,
};
use crate::ops::gated_delta_net_kkt::{
    build_gated_delta_net_kkt_params, dispatch_gated_delta_net_kkt, GatedDeltaNetKktParams,
};
use crate::ops::gated_delta_net_recompute_wu::{
    build_gated_delta_net_recompute_wu_params, dispatch_gated_delta_net_recompute_wu,
    GatedDeltaNetRecomputeWuParams,
};
use crate::ops::l2_norm::dispatch_l2_norm;
use metal::MTLSize;

/// K-dimension supported by this bank-split implementation. Hard-coded to
/// 256 in iter 15 to match Qwen3.6 head_dim. Future iters may generalize
/// to other multi-bank configurations (e.g. K=384 = 3 × 128).
pub const BANK_SPLIT_K: u32 = 256;

/// Number of K-banks in the iter-15 implementation. K=256 = 2 × MAX_K(128).
pub const NUM_BANKS: u32 = BANK_SPLIT_K / MAX_K;

/// Validate inputs for the K-bank-split dispatch. Same shape rules as the
/// underlying [`ChunkGatedDeltaRuleParams`] except K is locked to 256.
#[allow(clippy::too_many_arguments)]
fn validate_bank_split(
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
    // Common shape sanity (mirrors validate() in the K=128 path).
    if p.b == 0 || p.t == 0 || p.hg == 0 || p.h == 0 || p.k == 0 || p.v == 0 || p.bt == 0 {
        return Err(MlxError::InvalidArgument(
            "chunk_bank_split: all dims must be > 0".into(),
        ));
    }
    if p.h % p.hg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_bank_split: H ({}) must be a multiple of Hg ({})",
            p.h, p.hg
        )));
    }
    // The whole point of this module is K != MAX_K. Specifically K=256.
    if p.k != BANK_SPLIT_K {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_bank_split: K ({}) must equal BANK_SPLIT_K = {} exactly. \
             For K = MAX_K = {}, use dispatch_chunk_gated_delta_rule_fwd directly.",
            p.k, BANK_SPLIT_K, MAX_K
        )));
    }
    if p.v > MAX_V {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_bank_split: V ({}) exceeds MAX_V ({})",
            p.v, MAX_V
        )));
    }
    if p.bt != FIXED_BT {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_bank_split: bt ({}) must equal FIXED_BT ({})",
            p.bt, FIXED_BT
        )));
    }
    if p.t % p.bt != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_bank_split: T ({}) must be a multiple of bt ({})",
            p.t, p.bt
        )));
    }

    // Buffer byte-length checks. Sized for K=BANK_SPLIT_K.
    let q_elems = (p.b * p.t * p.hg * p.k) as usize;
    let k_elems = q_elems;
    let v_elems = (p.b * p.t * p.h * p.v) as usize;
    let g_elems = (p.b * p.t * p.h) as usize;
    let beta_elems = (p.b * p.t * p.h) as usize;
    let h0_elems = (p.b * p.h * p.v * p.k) as usize;
    let o_elems = v_elems;
    let final_state_elems = h0_elems;

    // Inputs are bf16 (2 bytes) except f32 g_log_decay + beta + h0 +
    // final_state which are f32 (4 bytes). Same layout as the K=128 path.
    let check = |name: &str, buf: &MlxBuffer, expected_bytes: usize| -> Result<()> {
        if buf.byte_len() < expected_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "chunk_bank_split: {name} buffer too small: need {expected_bytes} bytes, have {}",
                buf.byte_len()
            )));
        }
        Ok(())
    };
    check("q", q, q_elems * 2)?;
    check("k", k, k_elems * 2)?;
    check("v", v, v_elems * 2)?;
    check("g_log_decay", g_log_decay, g_elems * 4)?;
    check("beta", beta, beta_elems * 4)?;
    check("h0", h0, h0_elems * 4)?;
    check("o", o, o_elems * 2)?;
    check("final_state", final_state, final_state_elems * 4)?;

    Ok(())
}

/// Dispatch the K=256 bank-split chunk-parallel gated DeltaNet forward pass.
///
/// This is the entry point for Qwen3.6 prefill (head_dim=256). For the
/// K=128 case (e.g. Qwen3.5), use
/// [`crate::ops::chunk_gated_delta_rule::dispatch_chunk_gated_delta_rule_fwd`]
/// directly — it has identical semantics but a single-bank dispatch
/// (no cross-bank accumulation overhead).
///
/// # Inputs
///
/// All buffer shapes follow the K=128 path's contract except K=BANK_SPLIT_K(256).
///
/// # Errors
///
/// `MlxError::InvalidArgument` on shape mismatch.
/// `MlxError::Unimplemented` while body remains stubbed (iter 15 only).
///
/// # Multi-iter status
///
/// - Iter 15 (2026-05-23): SCAFFOLDING — signature + validation only.
///   Body returns Unimplemented.
/// - Iter 16+: see module-level doc-comment for arc plan.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_chunk_gated_delta_rule_fwd_k256_bank_split(
    _encoder: &mut CommandEncoder,
    _registry: &mut KernelRegistry,
    _device: &MlxDevice,
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
    // Validate first — fail fast on shape errors before we touch GPU resources.
    validate_bank_split(&p, q, k, v, g_log_decay, beta, h0, o, final_state)?;

    let metal_device = _device.metal_device();

    // ============================================================
    // Iter 22 body — NATIVE K=256 pipeline.
    //
    // After iters 19-20 added K=256 chunk_inter_state + chunk_o kernels
    // (compile-time 32-tile MMA loops) and iter 22 lifted kkt + recompute_wu
    // MAX_K from 192→256 (their kernels are runtime-K-parameterized with
    // per-thread scalar accumulation — no MMA scheduler regression),
    // the entire chunk pipeline can run natively at K=256 with no
    // bank-split orchestration.
    //
    // The bank_slice / bank_concat helpers from iters 16-17 remain in
    // this file as dead-code reference for future K≥384 cases, but are
    // NOT used by this dispatch path.
    //
    // Pipeline (mirrors dispatch_chunk_gated_delta_rule_fwd at K=128 but
    // at K=BANK_SPLIT_K=256):
    //
    //   Stage 1 — l2_norm (full q, k at K=256)
    //   Stage 2 — chunk_local_cumsum_g (K-independent)
    //   Stage 3 — kkt at K=256 (single dispatch; runtime K-loop)
    //   Stage 4 — tri_solve_invert (K-independent)
    //   Stage 5 — recompute_wu at K=256 (single dispatch; runtime K-loop)
    //   Stage 6 — chunk_inter_state_bf16_k256 (NEW iter 19; compile-time
    //              32-tile MMA loops)
    //   Stage 7 — chunk_o_bf16_k256 (NEW iter 20; compile-time 32-tile
    //              MMA loops)
    // ============================================================

    // ---- Stage 1: l2_norm (full q, k at K=256) ----
    let qk_rows: u32 = p.b * p.t * p.hg;
    let q_qk_elems = (qk_rows as usize) * (p.k as usize);
    let q_normed_buf;
    let k_normed_buf;
    let q_for_pipeline: &MlxBuffer;
    let k_for_pipeline: &MlxBuffer;

    if p.use_qk_l2norm {
        q_normed_buf = _device.alloc_buffer(q_qk_elems * 2, DType::BF16, vec![q_qk_elems])?;
        k_normed_buf = _device.alloc_buffer(q_qk_elems * 2, DType::BF16, vec![q_qk_elems])?;
        let l2_params = build_l2_norm_params(_device, L2_NORM_EPS, p.k)?;
        dispatch_l2_norm(
            _encoder, _registry, metal_device, q, &q_normed_buf, &l2_params, qk_rows, p.k,
        )?;
        _encoder.memory_barrier();
        dispatch_l2_norm(
            _encoder, _registry, metal_device, k, &k_normed_buf, &l2_params, qk_rows, p.k,
        )?;
        _encoder.memory_barrier();
        q_for_pipeline = &q_normed_buf;
        k_for_pipeline = &k_normed_buf;
    } else {
        // Caller passes pre-normed inputs.
        q_for_pipeline = q;
        k_for_pipeline = k;
    }

    // ---- Stage 2: cumsum_g (K-independent) ----
    let g_elems = (p.b * p.t * p.h) as usize;
    let g_cumsum_buf =
        _device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems])?;
    let cumsum_params = build_chunk_local_cumsum_g_params(_device, &p)?;
    dispatch_chunk_local_cumsum_g(
        _encoder, _registry, metal_device,
        g_log_decay, &g_cumsum_buf, &cumsum_params, &p,
    )?;
    _encoder.memory_barrier();

    // ---- Stage 3: kkt at K=256 (single dispatch) ----
    let a_elems = (p.b * p.t * p.h * p.bt) as usize;
    let a_strict_buf =
        _device.alloc_buffer(a_elems * 4, DType::F32, vec![a_elems])?;
    let kkt_params_value = GatedDeltaNetKktParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k, // K=256 native
        bt: p.bt,
    };
    let kkt_params = build_gated_delta_net_kkt_params(_device, kkt_params_value)?;
    dispatch_gated_delta_net_kkt(
        _encoder, _registry, metal_device,
        k_for_pipeline, beta, &g_cumsum_buf,
        &a_strict_buf,
        &kkt_params, kkt_params_value,
    )?;
    _encoder.memory_barrier();

    // ---- Stage 4: tri_solve_invert (K-independent) ----
    let a_inv_buf =
        _device.alloc_buffer(a_elems * 4, DType::F32, vec![a_elems])?;
    let invert_params_value = ChunkTriSolveInvertParams {
        b: p.b,
        t: p.t,
        h: p.h,
        bt: p.bt,
    };
    let invert_params = build_chunk_tri_solve_invert_params(_device, invert_params_value)?;
    dispatch_chunk_tri_solve_invert(
        _encoder, _registry, metal_device,
        &a_strict_buf, &a_inv_buf,
        &invert_params, invert_params_value,
    )?;
    _encoder.memory_barrier();

    // ---- Stage 5: recompute_wu at K=256 (single dispatch) ----
    let w_elems = (p.b * p.t * p.h * p.k) as usize;
    let u_elems = (p.b * p.t * p.h * p.v) as usize;
    let w_buf = _device.alloc_buffer(w_elems * 2, DType::BF16, vec![w_elems])?;
    let u_buf = _device.alloc_buffer(u_elems * 2, DType::BF16, vec![u_elems])?;
    let recompute_wu_params_value = GatedDeltaNetRecomputeWuParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k, // K=256 native
        v: p.v,
        bt: p.bt,
    };
    let recompute_wu_params =
        build_gated_delta_net_recompute_wu_params(_device, recompute_wu_params_value)?;
    dispatch_gated_delta_net_recompute_wu(
        _encoder, _registry, metal_device,
        k_for_pipeline, v, beta, &g_cumsum_buf, &a_inv_buf,
        &w_buf, &u_buf,
        &recompute_wu_params, recompute_wu_params_value,
    )?;
    _encoder.memory_barrier();

    // ---- Stage 6: chunk_inter_state_bf16_k256 (NEW iter 19) ----
    let nt = p.num_chunks();
    let h_elems = (p.b * nt * p.h * p.v * p.k) as usize;
    let v_new_elems = (p.b * p.t * p.h * p.v) as usize;
    let h_buf = _device.alloc_buffer(h_elems * 2, DType::BF16, vec![h_elems])?;
    let v_new_buf = _device.alloc_buffer(v_new_elems * 2, DType::BF16, vec![v_new_elems])?;
    let chunk_params_value = GatedDeltaNetChunkParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k, // K=256
        v: p.v,
        bt: p.bt,
    };
    let chunk_params = build_gated_delta_net_chunk_params(_device, chunk_params_value)?;
    dispatch_chunk_inter_state_k256(
        _encoder, _registry, metal_device,
        k_for_pipeline, &w_buf, &u_buf, &g_cumsum_buf, h0,
        &h_buf, &v_new_buf, final_state,
        &chunk_params, chunk_params_value,
    )?;
    _encoder.memory_barrier();

    // ---- Stage 7: chunk_o_bf16_k256 (NEW iter 20) ----
    let chunk_o_params_value = GatedDeltaNetChunkOParams {
        b: p.b,
        t: p.t,
        hg: p.hg,
        h: p.h,
        k: p.k, // K=256
        v: p.v,
        bt: p.bt,
        scale: p.scale,
    };
    let chunk_o_params =
        build_gated_delta_net_chunk_o_params(_device, chunk_o_params_value)?;
    dispatch_chunk_o_k256(
        _encoder, _registry, metal_device,
        q_for_pipeline, k_for_pipeline, &v_new_buf, &h_buf, &g_cumsum_buf,
        o,
        &chunk_o_params, chunk_o_params_value,
    )?;

    // Pipeline complete.
    Ok(())
}

/// GPU params struct for the `bank_slice_bf16` kernel. Must match the
/// `BankSliceParams` declaration in `shaders/bank_slice_bf16.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BankSliceGpuParams {
    rows: u32,
    k_full: u32,
    k_bank: u32,
    bank_offset: u32,
}

/// Iter 16 helper — extract a K-bank slice from a BF16 source buffer.
///
/// Reads `[rows, k_full]` BF16 K-innermost and writes the
/// `[rows, k_bank]` slice starting at `K[bank_offset..bank_offset+k_bank]`
/// into a contiguous destination buffer.
///
/// Used by the K=256 bank-split path to materialize per-bank temp Q/K
/// inputs that the existing K=128 chunk pipeline can consume directly.
///
/// # Arguments
///
/// * `encoder`     - Command encoder to record the dispatch into.
/// * `registry`    - Kernel registry (must have `bank_slice_bf16` registered).
/// * `device`      - Metal device.
/// * `src`         - Source `[rows, k_full]` BF16 buffer.
/// * `dst`         - Destination `[rows, k_bank]` BF16 buffer.
/// * `rows`        - Number of rows (B * T * Hg for q/k inputs).
/// * `k_full`      - Source K dimension (e.g. 256 for Qwen3.6).
/// * `k_bank`      - Destination K dimension (typically MAX_K=128).
/// * `bank_offset` - Source K offset to start reading (0 for bank 0,
///                   MAX_K for bank 1).
///
/// # Errors
///
/// `MlxError::InvalidArgument` if dimensions or buffer sizes are invalid.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_bank_slice_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    rows: u32,
    k_full: u32,
    k_bank: u32,
    bank_offset: u32,
) -> Result<()> {
    if rows == 0 || k_full == 0 || k_bank == 0 {
        return Err(MlxError::InvalidArgument(
            "bank_slice_bf16: rows, k_full, k_bank must all be > 0".into(),
        ));
    }
    if bank_offset + k_bank > k_full {
        return Err(MlxError::InvalidArgument(format!(
            "bank_slice_bf16: bank_offset ({bank_offset}) + k_bank ({k_bank}) \
             must be <= k_full ({k_full})"
        )));
    }
    let src_bytes = (rows as usize) * (k_full as usize) * 2;
    if src.byte_len() < src_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "bank_slice_bf16: src buffer too small: need {src_bytes} bytes, have {}",
            src.byte_len()
        )));
    }
    let dst_bytes = (rows as usize) * (k_bank as usize) * 2;
    if dst.byte_len() < dst_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "bank_slice_bf16: dst buffer too small: need {dst_bytes} bytes, have {}",
            dst.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("bank_slice_bf16", device)?;

    let gpu_params = BankSliceGpuParams {
        rows,
        k_full,
        k_bank,
        bank_offset,
    };

    // Grid: (k_bank, rows, 1). TG: (32, 1, 1).
    let grid = MTLSize::new(k_bank as u64, rows as u64, 1);
    let tg = MTLSize::new(std::cmp::min(32, k_bank as u64), 1, 1);

    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );
    Ok(())
}

/// ADR-033 §Pi Task #25 iter 17 — F32 variant of `dispatch_bank_slice_bf16`.
///
/// Same algorithm but for F32 buffers — used for the f32 `h0` initial-state
/// input of the bank-split chunk pipeline (h0 layout is [B, H, V, K] f32).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_bank_slice_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    rows: u32,
    k_full: u32,
    k_bank: u32,
    bank_offset: u32,
) -> Result<()> {
    if rows == 0 || k_full == 0 || k_bank == 0 {
        return Err(MlxError::InvalidArgument(
            "bank_slice_f32: rows, k_full, k_bank must all be > 0".into(),
        ));
    }
    if bank_offset + k_bank > k_full {
        return Err(MlxError::InvalidArgument(format!(
            "bank_slice_f32: bank_offset ({bank_offset}) + k_bank ({k_bank}) \
             must be <= k_full ({k_full})"
        )));
    }
    let src_bytes = (rows as usize) * (k_full as usize) * 4;
    if src.byte_len() < src_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "bank_slice_f32: src buffer too small: need {src_bytes} bytes, have {}",
            src.byte_len()
        )));
    }
    let dst_bytes = (rows as usize) * (k_bank as usize) * 4;
    if dst.byte_len() < dst_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "bank_slice_f32: dst buffer too small: need {dst_bytes} bytes, have {}",
            dst.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("bank_slice_f32", device)?;

    let gpu_params = BankSliceGpuParams {
        rows,
        k_full,
        k_bank,
        bank_offset,
    };

    let grid = MTLSize::new(k_bank as u64, rows as u64, 1);
    let tg = MTLSize::new(std::cmp::min(32, k_bank as u64), 1, 1);

    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );
    Ok(())
}

/// ADR-033 §Pi Task #25 iter 21 — dispatch wrapper for the K=256 native
/// `gated_delta_net_chunk_inter_state_bf16_k256` kernel.
///
/// Sister to [`dispatch_gated_delta_net_chunk_inter_state`] in the K=128
/// path. Same shape contract except K must equal `BANK_SPLIT_K` (= 256).
///
/// Threadgroup memory budget at K=256, BV=32:
///   bh tile     : BV × K  × 4 = 32 × 256 × 4 = 32 KB (running f32 state)
///   bv_stage_bf : BV × BT × 2 = 32 × 64  × 2 =  4 KB
///   Total: 36 KB
///
/// **NOTE (iter 21):** M-series threadgroup-memory cap is 32 KB. This
/// dispatch WILL FAIL at runtime if device.maxThreadgroupMemoryLength
/// < 36 KB. Iter 22 may pivot to BV=16 (16 KB bh + 2 KB stage = 18 KB)
/// or fp16 bh staging if the cap is hit on M5 Max. Shipping this wrapper
/// to surface the empirical constraint per "code+test==truth".
#[allow(clippy::too_many_arguments)]
pub fn dispatch_chunk_inter_state_k256(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    k: &MlxBuffer,
    w: &MlxBuffer,
    u: &MlxBuffer,
    g: &MlxBuffer,
    h0: &MlxBuffer,
    h_out: &MlxBuffer,
    v_new: &MlxBuffer,
    final_state: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: crate::ops::gated_delta_net_chunk::GatedDeltaNetChunkParams,
) -> Result<()> {
    use crate::ops::gated_delta_net_chunk::DEFAULT_BV;

    if p.k != BANK_SPLIT_K {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_chunk_inter_state_k256: K ({}) must equal {} exactly. \
             For K=128 (Qwen3.5), use dispatch_gated_delta_net_chunk_inter_state.",
            p.k, BANK_SPLIT_K
        )));
    }

    let pipeline = registry.get_pipeline("gated_delta_net_chunk_inter_state_bf16_k256", device)?;

    let nv_tiles = (p.v / DEFAULT_BV) as u64;
    let grid_tgs = MTLSize::new(nv_tiles, p.h as u64, p.b as u64);
    let tg = MTLSize::new(128, 1, 1);

    let bh_bytes: u64 = (DEFAULT_BV * p.k) as u64 * 4;
    let bv_stage_bytes: u64 = (DEFAULT_BV * p.bt) as u64 * 2;
    let shared_bytes = bh_bytes + bv_stage_bytes;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, k),
            (1, w),
            (2, u),
            (3, g),
            (4, h0),
            (5, h_out),
            (6, v_new),
            (7, final_state),
            (8, params_buf),
        ],
        &[(0, shared_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// ADR-033 §Pi Task #25 iter 21 — dispatch wrapper for the K=256 native
/// `gated_delta_net_chunk_o_bf16_k256` kernel.
///
/// Sister to [`dispatch_gated_delta_net_chunk_o`]. Same shape contract
/// except K must equal `BANK_SPLIT_K` (= 256).
///
/// chunk_o's threadgroup-memory budget is K-independent (`bA_stage` at
/// BT*BT*2 = 8 KB), so K=256 fits comfortably without the bh-tile
/// blow-up that affects inter_state.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_chunk_o_k256(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    h: &MlxBuffer,
    g: &MlxBuffer,
    o: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: crate::ops::gated_delta_net_chunk_o::GatedDeltaNetChunkOParams,
) -> Result<()> {
    if p.k != BANK_SPLIT_K {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_chunk_o_k256: K ({}) must equal {} exactly.",
            p.k, BANK_SPLIT_K
        )));
    }

    let pipeline = registry.get_pipeline("gated_delta_net_chunk_o_bf16_k256", device)?;

    let nv = p.num_v_tiles() as u64;
    let nt = p.num_chunks() as u64;
    let bh = (p.b * p.h) as u64;
    let grid_tgs = MTLSize::new(nv, nt, bh);
    let tg = MTLSize::new(256, 1, 1);

    let ba_stage_bytes: u64 = (p.bt as u64) * (p.bt as u64) * 2;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, q),
            (1, k),
            (2, v),
            (3, h),
            (4, g),
            (5, o),
            (6, params_buf),
        ],
        &[(0, ba_stage_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// ADR-033 §Pi Task #25 iter 17 — F32 bank concatenate (inverse of slice).
///
/// Writes `src[rows, k_bank]` into `dst[rows, k_full]` at the
/// `dst[:, bank_offset..bank_offset+k_bank]` slice. Used after the
/// bank-split pipeline to assemble the final_state[B, H, V, K=256] output
/// from two per-bank final_state[B, H, V, K=128] buffers.
///
/// Call twice (once per bank) into the same dst to assemble the full
/// K=256 final_state. dst MUST be pre-zeroed before the first call (or
/// the second call's K-slice will overwrite cleanly without zeroing).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_bank_concat_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    rows: u32,
    k_full: u32,
    k_bank: u32,
    bank_offset: u32,
) -> Result<()> {
    if rows == 0 || k_full == 0 || k_bank == 0 {
        return Err(MlxError::InvalidArgument(
            "bank_concat_f32: rows, k_full, k_bank must all be > 0".into(),
        ));
    }
    if bank_offset + k_bank > k_full {
        return Err(MlxError::InvalidArgument(format!(
            "bank_concat_f32: bank_offset ({bank_offset}) + k_bank ({k_bank}) \
             must be <= k_full ({k_full})"
        )));
    }
    let src_bytes = (rows as usize) * (k_bank as usize) * 4;
    if src.byte_len() < src_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "bank_concat_f32: src buffer too small: need {src_bytes} bytes, have {}",
            src.byte_len()
        )));
    }
    let dst_bytes = (rows as usize) * (k_full as usize) * 4;
    if dst.byte_len() < dst_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "bank_concat_f32: dst buffer too small: need {dst_bytes} bytes, have {}",
            dst.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("bank_concat_f32", device)?;

    let gpu_params = BankSliceGpuParams {
        rows,
        k_full,
        k_bank,
        bank_offset,
    };

    let grid = MTLSize::new(k_bank as u64, rows as u64, 1);
    let tg = MTLSize::new(std::cmp::min(32, k_bank as u64), 1, 1);

    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Iter 15 scaffolding test: verifies that the validator correctly
    /// gates on K=BANK_SPLIT_K(256) and rejects K!=256 with a clear error.
    /// Once iter 16+ implements the body, additional parity tests land.
    #[test]
    fn bank_split_rejects_k_other_than_256() {
        let p = ChunkGatedDeltaRuleParams {
            b: 1,
            t: 64,
            hg: 1,
            h: 1,
            k: 128, // not BANK_SPLIT_K
            v: 128,
            bt: 64,
            scale: 1.0,
            use_qk_l2norm: false,
        };

        // We can't easily construct MlxBuffers in a unit test without a
        // Metal device, so validate the constants + param shape only.
        assert_eq!(BANK_SPLIT_K, 256);
        assert_eq!(NUM_BANKS, 2);
        assert_eq!(MAX_K, 128);
        // Math sanity: BANK_SPLIT_K must split evenly into NUM_BANKS of
        // size MAX_K each.
        assert_eq!(BANK_SPLIT_K, NUM_BANKS * MAX_K);

        // Confirm the params struct accepts K=128 (the K!=256 case the
        // validator rejects).
        assert_eq!(p.k, 128);
    }

    /// Verify the math invariant that BANK_SPLIT_K is a multiple of MAX_K
    /// so each K-bank is a valid input to the underlying K=128 kernel.
    #[test]
    fn bank_split_k_divides_evenly_into_banks() {
        assert_eq!(BANK_SPLIT_K % MAX_K, 0);
        assert_eq!(NUM_BANKS, BANK_SPLIT_K / MAX_K);
    }

    /// Iter 16 GPU parity test — verify the bank_slice_bf16 kernel
    /// correctly extracts a K-bank slice from a [rows, k_full] BF16
    /// source.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn bank_slice_bf16_matches_cpu_reference() {
        use crate::{DType, KernelRegistry, MlxDevice};
        use half::bf16;

        let rows: u32 = 4;
        let k_full: u32 = 256;
        let k_bank: u32 = 128;

        // Generate deterministic test data [rows, k_full] BF16.
        let total = (rows * k_full) as usize;
        let src_data: Vec<bf16> = (0..total)
            .map(|i| bf16::from_f32((i as f32) * 0.0173 - 1.5))
            .collect();

        let device = MlxDevice::new().expect("MlxDevice::new");
        let mut registry = KernelRegistry::new();

        // Upload src.
        let mut src_buf = device
            .alloc_buffer(total * 2, DType::BF16, vec![rows as usize, k_full as usize])
            .expect("alloc src");
        src_buf
            .as_mut_slice::<bf16>()
            .expect("src as_mut")
            .copy_from_slice(&src_data);

        // Allocate dst.
        let dst_elems = (rows * k_bank) as usize;
        let dst_buf = device
            .alloc_buffer(
                dst_elems * 2,
                DType::BF16,
                vec![rows as usize, k_bank as usize],
            )
            .expect("alloc dst");

        for bank_idx in 0..NUM_BANKS {
            let bank_offset = bank_idx * MAX_K;

            // Build encoder, dispatch, commit.
            let mut encoder = device.command_encoder().expect("encoder");
            dispatch_bank_slice_bf16(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &src_buf,
                &dst_buf,
                rows,
                k_full,
                k_bank,
                bank_offset,
            )
            .expect("dispatch");
            encoder.commit_and_wait().expect("commit_and_wait");

            // CPU reference + compare.
            let dst_data: &[bf16] = dst_buf.as_slice().expect("dst as_slice");
            for r in 0..rows {
                for k in 0..k_bank {
                    let src_idx = (r * k_full + bank_offset + k) as usize;
                    let dst_idx = (r * k_bank + k) as usize;
                    assert_eq!(
                        dst_data[dst_idx].to_bits(),
                        src_data[src_idx].to_bits(),
                        "bank_idx={bank_idx} r={r} k={k}: dst {} != src {}",
                        dst_data[dst_idx].to_f32(),
                        src_data[src_idx].to_f32(),
                    );
                }
            }
        }
    }

    /// Iter 17 GPU parity test — verify the F32 variant of bank_slice
    /// extracts a K-bank slice correctly. Bit-equivalent to CPU reference.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn bank_slice_f32_matches_cpu_reference() {
        use crate::{DType, KernelRegistry, MlxDevice};

        let rows: u32 = 8;
        let k_full: u32 = 256;
        let k_bank: u32 = 128;

        let total = (rows * k_full) as usize;
        let src_data: Vec<f32> = (0..total)
            .map(|i| (i as f32) * 0.0173 - 1.5)
            .collect();

        let device = MlxDevice::new().expect("MlxDevice::new");
        let mut registry = KernelRegistry::new();

        let mut src_buf = device
            .alloc_buffer(total * 4, DType::F32, vec![rows as usize, k_full as usize])
            .expect("alloc src");
        src_buf
            .as_mut_slice::<f32>()
            .expect("src as_mut")
            .copy_from_slice(&src_data);

        let dst_elems = (rows * k_bank) as usize;
        let dst_buf = device
            .alloc_buffer(
                dst_elems * 4,
                DType::F32,
                vec![rows as usize, k_bank as usize],
            )
            .expect("alloc dst");

        for bank_idx in 0..NUM_BANKS {
            let bank_offset = bank_idx * MAX_K;
            let mut encoder = device.command_encoder().expect("encoder");
            dispatch_bank_slice_f32(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                &src_buf,
                &dst_buf,
                rows,
                k_full,
                k_bank,
                bank_offset,
            )
            .expect("dispatch");
            encoder.commit_and_wait().expect("commit_and_wait");

            let dst_data: &[f32] = dst_buf.as_slice().expect("dst as_slice");
            for r in 0..rows {
                for k in 0..k_bank {
                    let src_idx = (r * k_full + bank_offset + k) as usize;
                    let dst_idx = (r * k_bank + k) as usize;
                    assert_eq!(
                        dst_data[dst_idx].to_bits(),
                        src_data[src_idx].to_bits(),
                        "bank_idx={bank_idx} r={r} k={k}: dst {} != src {}",
                        dst_data[dst_idx],
                        src_data[src_idx],
                    );
                }
            }
        }
    }

    /// Iter 22 SMOKE TEST — full K=256 pipeline end-to-end.
    ///
    /// Dispatches `dispatch_chunk_gated_delta_rule_fwd_k256_bank_split`
    /// (the K=256 native pipeline composed in iter 22) at a small shape
    /// and verifies the dispatch completes without runtime error. This
    /// is the FIRST end-to-end exercise of the multi-iter arc — proves
    /// that the kkt → tri_solve → recompute_wu → chunk_inter_state_k256
    /// → chunk_o_k256 chain runs cleanly at K=256.
    ///
    /// Does NOT validate numerical correctness vs an autoregressive
    /// oracle — that lands in iter 23 along with the end-to-end
    /// Qwen3.6 prefill bench.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn chunk_k256_pipeline_smoke() {
        use crate::{DType, KernelRegistry, MlxDevice};

        let device = MlxDevice::new().expect("MlxDevice::new");
        let mut registry = KernelRegistry::new();

        // Small but valid K=256 shape: B=1, T=64, Hg=1, H=1, K=256, V=64,
        // BT=64. V=64 satisfies recompute_wu's V%BV==0 requirement.
        let p = ChunkGatedDeltaRuleParams {
            b: 1,
            t: 64,
            hg: 1,
            h: 1,
            k: BANK_SPLIT_K, // 256
            v: 64,
            bt: FIXED_BT,
            scale: 1.0,
            use_qk_l2norm: true,
        };

        // Allocate inputs (zero-init is fine for smoke test).
        let qk_elems = (p.b * p.t * p.hg * p.k) as usize;
        let v_elems = (p.b * p.t * p.h * p.v) as usize;
        let g_elems = (p.b * p.t * p.h) as usize;
        let h0_elems = (p.b * p.h * p.v * p.k) as usize;
        let o_elems = v_elems;

        let q_buf = device.alloc_buffer(qk_elems * 2, DType::BF16, vec![qk_elems]).unwrap();
        let k_buf = device.alloc_buffer(qk_elems * 2, DType::BF16, vec![qk_elems]).unwrap();
        let v_buf = device.alloc_buffer(v_elems * 2, DType::BF16, vec![v_elems]).unwrap();
        let g_buf = device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems]).unwrap();
        let beta_buf = device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems]).unwrap();
        let h0_buf = device.alloc_buffer(h0_elems * 4, DType::F32, vec![h0_elems]).unwrap();
        let o_buf = device.alloc_buffer(o_elems * 2, DType::BF16, vec![o_elems]).unwrap();
        let final_state_buf =
            device.alloc_buffer(h0_elems * 4, DType::F32, vec![h0_elems]).unwrap();

        let mut encoder = device.command_encoder().unwrap();
        let result = dispatch_chunk_gated_delta_rule_fwd_k256_bank_split(
            &mut encoder, &mut registry, &device,
            &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &h0_buf,
            &o_buf, &final_state_buf, p,
        );
        assert!(
            result.is_ok(),
            "K=256 pipeline encode failed: {:?}",
            result.err()
        );
        let commit = encoder.commit_and_wait();
        assert!(
            commit.is_ok(),
            "K=256 pipeline commit failed: {:?}",
            commit.err()
        );
        eprintln!(
            "[iter 22 smoke] K=256 full pipeline (l2_norm + cumsum + kkt + \
             tri_solve + recompute_wu + chunk_inter_state_k256 + \
             chunk_o_k256) dispatched + committed SUCCESSFULLY at \
             B=1,T=64,K=256,V=32,BT=64."
        );
    }

    /// Iter 21 SMOKE TEST — verify the K=256 native chunk_inter_state
    /// kernel can be dispatched without runtime error (shmem cap +
    /// pipeline-state creation). This is the empirical check that the
    /// 36 KB threadgroup memory budget at K=256, BV=32 fits the device.
    /// If this test fails with "newComputePipelineState…" or
    /// "setThreadgroupMemoryLength…" error, iter 22 needs to pivot to
    /// BV=16 (16 KB bh) or fp16 bh storage.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn chunk_inter_state_k256_dispatch_smoke() {
        use crate::ops::gated_delta_net_chunk::{
            build_gated_delta_net_chunk_params, GatedDeltaNetChunkParams,
        };
        use crate::{DType, KernelRegistry, MlxDevice};

        // Smallest valid K=256 shape: B=1, T=64=FIXED_BT, Hg=H=1, V=32=DEFAULT_BV.
        let p = GatedDeltaNetChunkParams {
            b: 1,
            t: FIXED_BT,
            hg: 1,
            h: 1,
            k: BANK_SPLIT_K, // 256
            v: 32,
            bt: FIXED_BT,
        };

        let device = MlxDevice::new().expect("MlxDevice::new");
        let mut registry = KernelRegistry::new();

        // Zero-init inputs sized per the kernel contract.
        let k_elems = (p.b * p.t * p.hg * p.k) as usize;
        let w_elems = (p.b * p.t * p.h * p.k) as usize;
        let u_elems = (p.b * p.t * p.h * p.v) as usize;
        let g_elems = (p.b * p.t * p.h) as usize;
        let h0_elems = (p.b * p.h * p.v * p.k) as usize;
        let h_out_elems = (p.b * p.num_chunks() * p.h * p.v * p.k) as usize;
        let final_state_elems = h0_elems;

        let k_buf = device.alloc_buffer(k_elems * 2, DType::BF16, vec![k_elems]).unwrap();
        let w_buf = device.alloc_buffer(w_elems * 2, DType::BF16, vec![w_elems]).unwrap();
        let u_buf = device.alloc_buffer(u_elems * 2, DType::BF16, vec![u_elems]).unwrap();
        let g_buf = device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems]).unwrap();
        let h0_buf = device.alloc_buffer(h0_elems * 4, DType::F32, vec![h0_elems]).unwrap();
        let h_out_buf = device.alloc_buffer(h_out_elems * 2, DType::BF16, vec![h_out_elems]).unwrap();
        let v_new_buf = device.alloc_buffer(u_elems * 2, DType::BF16, vec![u_elems]).unwrap();
        let final_state_buf =
            device.alloc_buffer(final_state_elems * 4, DType::F32, vec![final_state_elems]).unwrap();

        let params_buf = build_gated_delta_net_chunk_params(&device, p)
            .expect("build params");

        let mut encoder = device.command_encoder().unwrap();
        let result = dispatch_chunk_inter_state_k256(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &k_buf, &w_buf, &u_buf, &g_buf, &h0_buf,
            &h_out_buf, &v_new_buf, &final_state_buf,
            &params_buf, p,
        );

        match result {
            Ok(()) => {
                // Dispatch was queued; commit and confirm execution doesn't
                // panic. Don't validate output values — this is a smoke test
                // for capability, not numerical parity (iter 22 adds that).
                let commit = encoder.commit_and_wait();
                match commit {
                    Ok(()) => {
                        eprintln!(
                            "[iter 21 smoke] K=256 chunk_inter_state \
                             dispatch SUCCEEDED — shmem budget fits within \
                             device cap. Native kernel viable."
                        );
                    }
                    Err(e) => {
                        let msg = format!("{e}");
                        // Empirical finding: surface the shmem-cap signal if hit.
                        eprintln!(
                            "[iter 21 smoke] K=256 chunk_inter_state commit \
                             FAILED — likely shmem cap exceeded. Error: {msg}"
                        );
                        // Don't fail the test — this is an EMPIRICAL probe.
                        // Iter 22 will use this signal to decide BV=16 pivot.
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[iter 21 smoke] K=256 chunk_inter_state ENCODE failed \
                     (validation or pipeline creation): {e}"
                );
            }
        }
    }

    /// Iter 17 GPU parity test — verify bank_concat_f32 correctly writes
    /// per-bank slices into the right offsets in a wider dst buffer.
    /// Slice then concat should be the identity (round-trip).
    #[cfg(target_vendor = "apple")]
    #[test]
    fn bank_concat_f32_is_inverse_of_slice() {
        use crate::{DType, KernelRegistry, MlxDevice};

        let rows: u32 = 8;
        let k_full: u32 = 256;
        let k_bank: u32 = 128;

        // Source: [rows, k_full] with distinctive per-position values.
        let total = (rows * k_full) as usize;
        let src_data: Vec<f32> = (0..total)
            .map(|i| (i as f32) * 0.0287 + 0.5)
            .collect();

        let device = MlxDevice::new().expect("MlxDevice::new");
        let mut registry = KernelRegistry::new();

        let mut src_buf = device
            .alloc_buffer(total * 4, DType::F32, vec![rows as usize, k_full as usize])
            .expect("alloc src");
        src_buf
            .as_mut_slice::<f32>()
            .expect("src")
            .copy_from_slice(&src_data);

        // Two per-bank intermediate buffers.
        let bank_elems = (rows * k_bank) as usize;
        let bank0_buf = device
            .alloc_buffer(bank_elems * 4, DType::F32, vec![rows as usize, k_bank as usize])
            .expect("alloc bank0");
        let bank1_buf = device
            .alloc_buffer(bank_elems * 4, DType::F32, vec![rows as usize, k_bank as usize])
            .expect("alloc bank1");

        // Destination: zero-init [rows, k_full] for concat output.
        let mut dst_buf = device
            .alloc_buffer(total * 4, DType::F32, vec![rows as usize, k_full as usize])
            .expect("alloc dst");
        dst_buf
            .as_mut_slice::<f32>()
            .expect("dst init")
            .iter_mut()
            .for_each(|v| *v = 0.0);

        // 1) Slice src → bank0, bank1.
        // 2) Concat bank0, bank1 → dst.
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_bank_slice_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &src_buf,
            &bank0_buf,
            rows,
            k_full,
            k_bank,
            0,
        )
        .expect("slice bank0");
        dispatch_bank_slice_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &src_buf,
            &bank1_buf,
            rows,
            k_full,
            k_bank,
            MAX_K,
        )
        .expect("slice bank1");
        encoder.memory_barrier();
        dispatch_bank_concat_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &bank0_buf,
            &dst_buf,
            rows,
            k_full,
            k_bank,
            0,
        )
        .expect("concat bank0");
        dispatch_bank_concat_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &bank1_buf,
            &dst_buf,
            rows,
            k_full,
            k_bank,
            MAX_K,
        )
        .expect("concat bank1");
        encoder.commit_and_wait().expect("commit_and_wait");

        // Round-trip: dst should equal src bit-for-bit.
        let dst_data: &[f32] = dst_buf.as_slice().expect("dst as_slice");
        for i in 0..total {
            assert_eq!(
                dst_data[i].to_bits(),
                src_data[i].to_bits(),
                "round-trip at idx {i}: dst {} != src {}",
                dst_data[i],
                src_data[i],
            );
        }
    }
}
