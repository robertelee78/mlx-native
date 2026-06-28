//! ADR-040 §22 — F32-shadow falsification spike for the QuaRot-style o_proj FWHT fold.
//!
//! The hybrid TQ-HB-V path stores V rotated by H = FWHT_norm·D1 (sign applied
//! BEFORE the normalized butterfly), so the attention output is rotated:
//!     sdpa_out = softmax(QK)·(H·V) = H·(softmax(QK)·V) = H·true_sdpa
//! and a runtime `fwht_sign_undo` dispatch recovers `true_sdpa = H⁻¹·sdpa_out`
//! before o_proj.
//!
//! The fold ELIMINATES that runtime undo by baking H⁻¹ into the o_proj weight
//! offline:  W_o' = W_o · H⁻¹  (block-diagonal over heads). Then
//!     W_o' · sdpa_out = (W_o·H⁻¹)·(H·true_sdpa) = W_o·true_sdpa
//! i.e. the folded weight un-rotates as a free matmul side-effect.
//!
//! Because H⁻¹ = D1·FWHT_norm and the fold multiplies o_proj's INPUT columns,
//! the per-row, per-head-block fold is IDENTICAL to applying the V-encode
//! rotation (`sign_premult_fwht`) to each row block of W_o:
//!     W_o'[r, block] = FWHT_norm(D1 · W_o[r, block]) = encode_rot(W_o[r, block]).
//!
//! This pure-F32 spike (NO quantization) asserts:  old path  (rotated → undo →
//! W_o)  ==  new path  (rotated → W_o', no undo)  for one sliding (head_dim=256)
//! and one global (head_dim=512) layer shape. If this fails, the fold math is
//! wrong and we stop before touching conversion/runtime. Per codex's
//! APPROVE-WITH-CHANGES: quantization is added only AFTER this F32 equality holds.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::turboquant::{
    apply_d1_sign_mask_inplace, fwht_inplace, TBQ_SIGNS_256, TBQ_SIGNS_512,
};

fn pseudo_random(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Encode rotation H·x = FWHT_norm(D1·x): sign FIRST, then normalized butterfly.
/// Mirrors hadamard_quantize_kv_fast.metal (sign before WHT in encode).
fn encode_rot(x: &mut [f32], signs: &[u8]) {
    apply_d1_sign_mask_inplace(x, signs);
    fwht_inplace(x).expect("fwht");
}

/// Undo H⁻¹·x = D1·FWHT_norm(x): butterfly FIRST, then sign.
/// Mirrors fwht_sign_undo (FWHT → normalize → sign·elem).
fn undo_rot(x: &mut [f32], signs: &[u8]) {
    fwht_inplace(x).expect("fwht");
    apply_d1_sign_mask_inplace(x, signs);
}

/// out[r] = Σ_c W[r,c]·v[c]   (W is [hidden, in_dim] row-major).
fn matvec(w: &[f32], v: &[f32], hidden: usize, in_dim: usize) -> Vec<f32> {
    (0..hidden)
        .map(|r| {
            let row = &w[r * in_dim..(r + 1) * in_dim];
            row.iter().zip(v).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn run_case(head_dim: usize, n_heads: usize, hidden: usize, signs: &[u8], seed: u64) {
    let in_dim = n_heads * head_dim;

    // The TRUE attention output (pre-rotation), per head.
    let true_sdpa = pseudo_random(seed ^ 0x11, in_dim);
    // The o_proj weight [hidden, in_dim].
    let w_o = pseudo_random(seed ^ 0x22, hidden * in_dim);

    // What the kernel actually produces: rotated per-head sdpa_out = H·true.
    let mut rotated = true_sdpa.clone();
    for h in 0..n_heads {
        encode_rot(&mut rotated[h * head_dim..(h + 1) * head_dim], signs);
    }

    // --- OLD path: runtime undo recovers true_sdpa, then original W_o. ---
    let mut undone = rotated.clone();
    for h in 0..n_heads {
        undo_rot(&mut undone[h * head_dim..(h + 1) * head_dim], signs);
    }
    // Sanity: undo(rotated) must reproduce true_sdpa (H⁻¹H = I).
    let undo_err = undone
        .iter()
        .zip(&true_sdpa)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        undo_err < 1e-4,
        "head_dim={head_dim}: undo(rotated) != true_sdpa (max_abs={undo_err:.2e}) — \
         encode/undo are not inverses"
    );
    let out_old = matvec(&w_o, &undone, hidden, in_dim);

    // --- NEW path: fold H⁻¹ into W_o columns (per head block = encode_rot of
    //     each ROW block), feed the RAW rotated sdpa_out, no runtime undo. ---
    let mut w_folded = w_o.clone();
    for r in 0..hidden {
        let row = &mut w_folded[r * in_dim..(r + 1) * in_dim];
        for h in 0..n_heads {
            encode_rot(&mut row[h * head_dim..(h + 1) * head_dim], signs);
        }
    }
    let out_new = matvec(&w_folded, &rotated, hidden, in_dim);

    // --- The claim: folded·rotated == original·undone (== original·true). ---
    let max_abs = out_old
        .iter()
        .zip(&out_new)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let denom = out_old.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
    let max_rel = max_abs / denom;
    eprintln!(
        "[tq-fold-shadow] head_dim={head_dim} n_heads={n_heads} hidden={hidden}: \
         max_abs_diff={max_abs:.3e} max_rel={max_rel:.3e} (undo_err={undo_err:.2e})"
    );
    // F32 FWHT (log2 d butterfly stages) accumulates rounding; require tight rel.
    assert!(
        max_rel < 1e-4,
        "head_dim={head_dim}: folded o_proj diverges from undo+o_proj \
         (max_abs={max_abs:.3e}, max_rel={max_rel:.3e}) — FOLD MATH IS WRONG"
    );
}

#[test]
fn tq_fold_oproj_f32_shadow_sliding_256() {
    // Sliding layer: head_dim=256, 16 q-heads, TBQ_SIGNS_256.
    run_case(256, 16, 64, &TBQ_SIGNS_256, 0x5117_0256);
}

#[test]
fn tq_fold_oproj_f32_shadow_global_512() {
    // Global/full-attn layer: head_dim=512, 16 q-heads, TBQ_SIGNS_512.
    run_case(512, 16, 64, &TBQ_SIGNS_512, 0x610b_0512);
}
