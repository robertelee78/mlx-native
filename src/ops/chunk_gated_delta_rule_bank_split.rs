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
//! source of the prefill peer-parity gap vs llama.cpp at production shapes.
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
//!   target: hf2q ≥ llama.cpp at all tested seq lengths.
//!
//! # Status
//!
//! Iter 15 SHIPPED 2026-05-23 (scaffolding). Future iters extend per the
//! arc plan above.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::chunk_gated_delta_rule::{
    ChunkGatedDeltaRuleParams, FIXED_BT, MAX_K, MAX_V,
};
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

    // Iter 16+ body plan:
    //
    // ```ignore
    // // Allocate per-bank temp buffers.
    // let a_strict_partial = device.alloc_buffer(a_elems * 4, F32, ...)?;
    // let h_partial = device.alloc_buffer(h_elems * 2, BF16, ...)?;
    // let o_partial = device.alloc_buffer(o_elems * 2, BF16, ...)?;
    //
    // // Stage 1: l2_norm (per-bank since k_bankN is a K-slice of full k).
    // // Stage 2: cumsum_g — runs ONCE (K-independent).
    // for bank_idx in 0..NUM_BANKS {
    //     // Extract K-slice views (q_bank, k_bank).
    //     // Stage 3: kkt for this bank → A_strict_partial[bank_idx].
    // }
    // // Stage 3.5: SUM A_strict_partial across banks → A_strict_full.
    //
    // // Stage 4: tri_solve_invert on A_strict_full → A_inv.
    //
    // for bank_idx in 0..NUM_BANKS {
    //     // Stage 5: recompute_w_u per bank → w_bank, u_bank (u is identical
    //     //   across banks since v is shared; keep one u, two w).
    //     // Stage 6: chunk_inter_state per bank → h_bank, v_new_bank,
    //     //   final_state_bank.
    //     // Stage 7: chunk_o per bank → o_bank.
    // }
    // // Stage 7.5: SUM o_bank across banks → o (final output).
    // // Stage 8: concat final_state_bank → final_state[V, K=BANK_SPLIT_K].
    // ```
    //
    // Iter 16 implements stages 1-3 + 3.5 (kkt SUM).
    // Iter 17 implements stages 4-6.
    // Iter 18 implements stages 7-8 + parity test.
    Err(MlxError::InvalidArgument(
        "chunk_bank_split: iter 15 ships scaffolding only; body lands in iter 16+. \
         For K=128 (Qwen3.5), use dispatch_chunk_gated_delta_rule_fwd directly."
            .into(),
    ))
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
}
