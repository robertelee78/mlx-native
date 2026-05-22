//! ADR-037 Phase E1.1 parity test (2026-05-22).
//!
//! Acceptance gate: with qL=1 and an explicit mask buffer filled with
//! values that mimic flash_attn_vec's implicit causal mask, the
//! tree_attention kernel must produce **byte-identical** output to
//! flash_attn_vec for the same Q/K/V/params.
//!
//! This is the smallest unit that derisks the rest of the EAGLE-3
//! port (ADR-037 §8). If tree-attention can't pass tree=1 parity,
//! the entire stack rests on broken foundation.
//!
//! Test cases:
//! 1. dk256 dense (n_heads=4, kv_seq_len=32, full causal)
//! 2. dk256 GQA (n_heads=16, n_kv_heads=8, kv_seq_len=48)
//! 3. dk512 dense (n_heads=4, kv_seq_len=32)
//! 4. dk256 long (n_heads=4, kv_seq_len=512, exercises NWG > 1 reduce)
//! 5. dk256 kv_seq_len not a multiple of C=32 (exercises trailing
//!    partial chunk + out-of-range mask read path)

#![allow(non_snake_case)] // qL camel-case preserved for kernel-naming
                          // parity (kernel uses qL throughout — see
                          // ADR-034 task #89 + flash_attn_vec_tq).

use mlx_native::ops::flash_attn_vec::{self, FlashAttnVecParams};
use mlx_native::ops::tree_attention::{
    self, TreeAttentionParams, TREE_MASK_ATTENDED, TREE_MASK_MASKED,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn pseudo_random(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((x >> 33) as u32) & 0x7FFFFF;
    (bits as f32 / 0x7FFFFF as f32) * 2.0 - 1.0
}

fn fill_random(buf: &mut [f32], seed: u64) {
    for (i, val) in buf.iter_mut().enumerate() {
        *val = pseudo_random(seed + i as u64);
    }
}

/// Run flash_attn_vec + tree_attention on identical Q/K/V and assert
/// byte-identical output.
fn assert_tree1_byte_identical(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    seed: u64,
    label: &str,
) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let q_elems = (num_heads as usize) * (head_dim as usize);
    let kv_elems = (num_kv_heads as usize) * (kv_capacity as usize) * (head_dim as usize);

    let mut q_data = vec![0.0f32; q_elems];
    let mut k_data = vec![0.0f32; kv_elems];
    let mut v_data = vec![0.0f32; kv_elems];
    fill_random(&mut q_data, seed);
    fill_random(&mut k_data, seed + 10_000);
    fill_random(&mut v_data, seed + 20_000);

    // Build mask that mirrors flash_attn_vec's implicit causal at qL=1:
    //   - abs_pos = kv_seq_len - 1
    //   - causal_max_k = kv_seq_len
    //   - cells [0, kv_seq_len)         → TREE_MASK_ATTENDED (0.0)
    //   - cells [kv_seq_len, kv_seq_len) → empty (mask_stride == kv_seq_len)
    let mask_stride = kv_seq_len;
    let mask_elems = (1_usize) * (mask_stride as usize);
    let mut mask_data = vec![TREE_MASK_MASKED; mask_elems];
    for k_pos in 0..(kv_seq_len as usize) {
        mask_data[k_pos] = TREE_MASK_ATTENDED;
    }

    // GPU buffers — separate copies for flash_attn_vec and tree_attention
    // (so we can compare outputs side by side).
    let q_bytes = q_elems * 4;
    let kv_bytes = kv_elems * 4;
    let out_bytes = q_elems * 4;
    let mask_bytes = mask_elems * 4;

    let mut q_buf = device
        .alloc_buffer(q_bytes, DType::F32, vec![q_elems])
        .expect("alloc Q");
    let mut k_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elems])
        .expect("alloc K");
    let mut v_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elems])
        .expect("alloc V");
    let mut mask_buf = device
        .alloc_buffer(mask_bytes, DType::F32, vec![mask_elems])
        .expect("alloc mask");
    let fa_out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elems])
        .expect("alloc fa output");
    let tree_out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elems])
        .expect("alloc tree output");

    q_buf
        .as_mut_slice::<f32>()
        .expect("q slice")
        .copy_from_slice(&q_data);
    k_buf
        .as_mut_slice::<f32>()
        .expect("k slice")
        .copy_from_slice(&k_data);
    v_buf
        .as_mut_slice::<f32>()
        .expect("v slice")
        .copy_from_slice(&v_data);
    mask_buf
        .as_mut_slice::<f32>()
        .expect("mask slice")
        .copy_from_slice(&mask_data);

    // tmp buffers (same size — same NWG, same output layout).
    let fa_tmp_bytes = flash_attn_vec::tmp_buffer_bytes(num_heads, head_dim);
    let fa_tmp_buf = device
        .alloc_buffer(fa_tmp_bytes, DType::F32, vec![fa_tmp_bytes / 4])
        .expect("alloc fa tmp");
    let tree_tmp_bytes = tree_attention::tmp_buffer_bytes(num_heads, head_dim, 1);
    let tree_tmp_buf = device
        .alloc_buffer(tree_tmp_bytes, DType::F32, vec![tree_tmp_bytes / 4])
        .expect("alloc tree tmp");
    assert_eq!(
        fa_tmp_bytes, tree_tmp_bytes,
        "{label}: tmp buffer size mismatch (both kernels share layout)"
    );

    // --- Run flash_attn_vec ---
    let fa_params = FlashAttnVecParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type: 1, // causal
        sliding_window: 0,
        softcap: 0.0,
        q_seq_len: FlashAttnVecParams::DEFAULT_Q_SEQ_LEN,
    };
    {
        let mut enc = device.command_encoder().expect("encoder fa");
        flash_attn_vec::flash_attn_vec(
            &mut enc,
            &mut registry,
            &device,
            &q_buf,
            &k_buf,
            &v_buf,
            &fa_out_buf,
            &fa_tmp_buf,
            &fa_params,
        )
        .expect("flash_attn_vec dispatch");
        enc.commit_and_wait().expect("commit fa");
    }

    // --- Run tree_attention with mask mimicking causal ---
    let tree_params = TreeAttentionParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        q_seq_len: 1,
        mask_stride,
    };
    {
        let mut enc = device.command_encoder().expect("encoder tree");
        tree_attention::tree_attention(
            &mut enc,
            &mut registry,
            &device,
            &q_buf,
            &k_buf,
            &v_buf,
            &mask_buf,
            &tree_out_buf,
            &tree_tmp_buf,
            &tree_params,
        )
        .expect("tree_attention dispatch");
        enc.commit_and_wait().expect("commit tree");
    }

    // --- Byte-identity assertion ---
    let fa_out: &[f32] = fa_out_buf.as_slice::<f32>().expect("fa output slice");
    let tree_out: &[f32] = tree_out_buf.as_slice::<f32>().expect("tree output slice");
    assert_eq!(
        fa_out.len(),
        tree_out.len(),
        "{label}: output length mismatch"
    );

    // Compare as raw bytes (bit-equal). Both paths take identical FMA
    // ordering, identical online-softmax sequence, identical reduce
    // kernel — so the outputs must be bit-equal, not just close.
    let mut first_mismatch: Option<(usize, f32, f32)> = None;
    for (i, (fa, tree)) in fa_out.iter().zip(tree_out.iter()).enumerate() {
        if fa.to_bits() != tree.to_bits() {
            first_mismatch = Some((i, *fa, *tree));
            break;
        }
    }
    if let Some((i, fa, tree)) = first_mismatch {
        panic!(
            "{label}: byte-identity violated at index {i}: fa={fa:.9e} (bits {:#010x}), tree={tree:.9e} (bits {:#010x})",
            fa.to_bits(),
            tree.to_bits()
        );
    }
    eprintln!("{label}: byte-identical across {} F32 outputs", fa_out.len());
}

// --------------------------------------------------------------------------
// Test matrix
// --------------------------------------------------------------------------

#[test]
fn adr_037_e1_1_tree1_parity_dk256_basic_2026_05_22() {
    assert_tree1_byte_identical(
        4,                          // num_heads
        4,                          // num_kv_heads
        256,                        // head_dim
        32,                         // kv_seq_len (exactly C boundary)
        64,                         // kv_capacity
        1.0 / (256.0_f32).sqrt(),   // scale
        42,                         // seed
        "dk256 basic tree=1 parity",
    );
}

#[test]
fn adr_037_e1_1_tree1_parity_dk256_gqa_2026_05_22() {
    assert_tree1_byte_identical(
        16,
        8,
        256,
        48,
        64,
        1.0,
        100,
        "dk256 GQA tree=1 parity",
    );
}

#[test]
fn adr_037_e1_1_tree1_parity_dk512_basic_2026_05_22() {
    assert_tree1_byte_identical(
        4,
        4,
        512,
        32,
        64,
        1.0 / (512.0_f32).sqrt(),
        1111,
        "dk512 basic tree=1 parity",
    );
}

#[test]
fn adr_037_e1_1_tree1_parity_dk256_long_2026_05_22() {
    // kv_seq_len = 512 exercises NWG > 1 reduce kernel path:
    // 512 / C=32 = 16 chunks; with NWG=32, each workgroup handles
    // at most one chunk → reduce pass must combine 32 partials.
    assert_tree1_byte_identical(
        4,
        4,
        256,
        512,
        1024,
        1.0 / (256.0_f32).sqrt(),
        7777,
        "dk256 long tree=1 parity",
    );
}

/// ADR-037 Phase E1.2 (2026-05-22) — chain parity contract.
///
/// Extends the E1.1 byte-identity contract to qL > 1. flash_attn_vec
/// at qL > 1 (added by task #89) uses the implicit causal formula
/// `abs_pos = kv_seq_len - qL + iq1`, attended cells `[0, abs_pos + 1)`
/// per query row iq1. tree_attention with a per-row mask matching
/// that pattern must produce bit-equal output.
///
/// "Chain" topology: each tree-node attends ALL prior positions
/// (no holes within the mask row). Logically equivalent to running
/// qL successive single-token decodes at positions
/// [kv_seq_len - qL, kv_seq_len).
fn assert_chain_byte_identical(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    q_seq_len: u32,
    scale: f32,
    seed: u64,
    label: &str,
) {
    assert!(q_seq_len > 1, "{label}: chain test requires qL > 1");
    assert!(
        q_seq_len <= kv_seq_len,
        "{label}: qL must be <= kv_seq_len (flash_attn_vec invariant)"
    );

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let q_elems = (num_heads as usize) * (q_seq_len as usize) * (head_dim as usize);
    let kv_elems = (num_kv_heads as usize) * (kv_capacity as usize) * (head_dim as usize);

    let mut q_data = vec![0.0f32; q_elems];
    let mut k_data = vec![0.0f32; kv_elems];
    let mut v_data = vec![0.0f32; kv_elems];
    fill_random(&mut q_data, seed);
    fill_random(&mut k_data, seed + 10_000);
    fill_random(&mut v_data, seed + 20_000);

    // Build per-row mask mirroring flash_attn_vec's implicit causal:
    //   row iq1: cells [0, abs_pos+1) attended, rest masked
    //   abs_pos = kv_seq_len - q_seq_len + iq1
    let mask_stride = kv_seq_len;
    let mask_elems = (q_seq_len as usize) * (mask_stride as usize);
    let mut mask_data = vec![TREE_MASK_MASKED; mask_elems];
    for iq1 in 0..(q_seq_len as usize) {
        let abs_pos = (kv_seq_len as usize) - (q_seq_len as usize) + iq1;
        let causal_max_k = (abs_pos + 1).min(kv_seq_len as usize);
        let row_base = iq1 * (mask_stride as usize);
        for k_pos in 0..causal_max_k {
            mask_data[row_base + k_pos] = TREE_MASK_ATTENDED;
        }
    }

    let q_bytes = q_elems * 4;
    let kv_bytes = kv_elems * 4;
    let out_bytes = q_elems * 4;
    let mask_bytes = mask_elems * 4;

    let mut q_buf = device
        .alloc_buffer(q_bytes, DType::F32, vec![q_elems])
        .expect("alloc Q");
    let mut k_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elems])
        .expect("alloc K");
    let mut v_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elems])
        .expect("alloc V");
    let mut mask_buf = device
        .alloc_buffer(mask_bytes, DType::F32, vec![mask_elems])
        .expect("alloc mask");
    let fa_out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elems])
        .expect("alloc fa output");
    let tree_out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elems])
        .expect("alloc tree output");

    q_buf
        .as_mut_slice::<f32>()
        .expect("q slice")
        .copy_from_slice(&q_data);
    k_buf
        .as_mut_slice::<f32>()
        .expect("k slice")
        .copy_from_slice(&k_data);
    v_buf
        .as_mut_slice::<f32>()
        .expect("v slice")
        .copy_from_slice(&v_data);
    mask_buf
        .as_mut_slice::<f32>()
        .expect("mask slice")
        .copy_from_slice(&mask_data);

    // tmp buffer must include qL factor for both paths.
    let fa_tmp_bytes =
        flash_attn_vec::tmp_buffer_bytes_with_qL(num_heads, head_dim, q_seq_len);
    let fa_tmp_buf = device
        .alloc_buffer(fa_tmp_bytes, DType::F32, vec![fa_tmp_bytes / 4])
        .expect("alloc fa tmp");
    let tree_tmp_bytes = tree_attention::tmp_buffer_bytes(num_heads, head_dim, q_seq_len);
    let tree_tmp_buf = device
        .alloc_buffer(tree_tmp_bytes, DType::F32, vec![tree_tmp_bytes / 4])
        .expect("alloc tree tmp");
    assert_eq!(
        fa_tmp_bytes, tree_tmp_bytes,
        "{label}: tmp buffer size mismatch"
    );

    // --- flash_attn_vec at qL > 1 ---
    let fa_params = FlashAttnVecParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type: 1,
        sliding_window: 0,
        softcap: 0.0,
        q_seq_len,
    };
    {
        let mut enc = device.command_encoder().expect("encoder fa");
        flash_attn_vec::flash_attn_vec(
            &mut enc,
            &mut registry,
            &device,
            &q_buf,
            &k_buf,
            &v_buf,
            &fa_out_buf,
            &fa_tmp_buf,
            &fa_params,
        )
        .expect("flash_attn_vec dispatch (qL > 1)");
        enc.commit_and_wait().expect("commit fa");
    }

    // --- tree_attention at same qL with per-row causal mask ---
    let tree_params = TreeAttentionParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        q_seq_len,
        mask_stride,
    };
    {
        let mut enc = device.command_encoder().expect("encoder tree");
        tree_attention::tree_attention(
            &mut enc,
            &mut registry,
            &device,
            &q_buf,
            &k_buf,
            &v_buf,
            &mask_buf,
            &tree_out_buf,
            &tree_tmp_buf,
            &tree_params,
        )
        .expect("tree_attention dispatch (qL > 1)");
        enc.commit_and_wait().expect("commit tree");
    }

    let fa_out: &[f32] = fa_out_buf.as_slice::<f32>().expect("fa output slice");
    let tree_out: &[f32] = tree_out_buf.as_slice::<f32>().expect("tree output slice");
    assert_eq!(
        fa_out.len(),
        tree_out.len(),
        "{label}: output length mismatch"
    );

    let mut first_mismatch: Option<(usize, f32, f32)> = None;
    for (i, (fa, tree)) in fa_out.iter().zip(tree_out.iter()).enumerate() {
        if fa.to_bits() != tree.to_bits() {
            first_mismatch = Some((i, *fa, *tree));
            break;
        }
    }
    if let Some((i, fa, tree)) = first_mismatch {
        panic!(
            "{label}: byte-identity violated at index {i}: fa={fa:.9e} (bits {:#010x}), tree={tree:.9e} (bits {:#010x})",
            fa.to_bits(),
            tree.to_bits()
        );
    }
    eprintln!(
        "{label}: byte-identical across {} F32 outputs (qL={q_seq_len})",
        fa_out.len()
    );
}

// --------------------------------------------------------------------------
// Phase E1.2 — chain parity (qL > 1 byte-identity)
// --------------------------------------------------------------------------

#[test]
fn adr_037_e1_2_chain_parity_dk256_qL2_2026_05_22() {
    // Smallest non-trivial chain: qL=2 over kv=32. Validates that
    // task #89's per-query causal indexing (abs_pos formula) lands
    // in the same shader code path under the tree_attention mask.
    assert_chain_byte_identical(
        4,
        4,
        256,
        32,
        64,
        2,
        1.0 / (256.0_f32).sqrt(),
        4242,
        "dk256 chain qL=2 parity",
    );
}

#[test]
fn adr_037_e1_2_chain_parity_dk256_qL4_2026_05_22() {
    // qL=4 = production spec-decode batched-verify shape (per task #89
    // Step 2: kernel parity-verified at qL in {1,2,4,8}).
    assert_chain_byte_identical(
        4,
        4,
        256,
        32,
        64,
        4,
        1.0 / (256.0_f32).sqrt(),
        5555,
        "dk256 chain qL=4 parity",
    );
}

#[test]
fn adr_037_e1_2_chain_parity_dk256_qL8_gqa_2026_05_22() {
    // qL=8 (top of task #89 parity range) + GQA: validates that
    // per-row mask + GQA head-mapping interact correctly.
    assert_chain_byte_identical(
        16, // num_heads
        8,  // num_kv_heads (GQA 2:1)
        256,
        48, // kv_seq_len
        64,
        8, // qL
        1.0,
        6666,
        "dk256 GQA chain qL=8 parity",
    );
}

#[test]
fn adr_037_e1_2_chain_parity_dk512_qL4_2026_05_22() {
    assert_chain_byte_identical(
        4,
        4,
        512,
        32,
        64,
        4,
        1.0 / (512.0_f32).sqrt(),
        7777,
        "dk512 chain qL=4 parity",
    );
}

#[test]
fn adr_037_e1_2_chain_parity_dk256_long_qL4_2026_05_22() {
    // Long-context (kv=512) chain: validates that the NWG>1 reduce
    // path correctly combines partial S/M across qL=4 query rows
    // simultaneously. This is the closest test to production
    // spec-decode verify at long context.
    assert_chain_byte_identical(
        4,
        4,
        256,
        512,
        1024,
        4,
        1.0 / (256.0_f32).sqrt(),
        8888,
        "dk256 long chain qL=4 parity",
    );
}

// --------------------------------------------------------------------------
// Phase E1.1 tree=1 parity tests follow.
// --------------------------------------------------------------------------

#[test]
fn adr_037_e1_1_tree1_parity_dk256_unaligned_2026_05_22() {
    // kv_seq_len=50 is NOT a multiple of C=32. The trailing chunk
    // [32, 64) has cells at k_pos=50..64 which lie beyond
    // kv_seq_len. flash_attn_vec masks those via causal_max_k;
    // tree_attention masks them via the in-shader bounds check
    // (`k_pos < kv_seq_len`). Output must still be byte-identical.
    assert_tree1_byte_identical(
        4,
        4,
        256,
        50,
        128,
        1.0 / (256.0_f32).sqrt(),
        9999,
        "dk256 unaligned-kv_seq_len tree=1 parity",
    );
}

// --------------------------------------------------------------------------
// Phase E1.3 — fixed-square tree (non-causal within tree segment) vs CPU
// --------------------------------------------------------------------------
//
// E1.1 and E1.2 verify byte-identity against flash_attn_vec for masks that
// reduce to implicit causal. E1.3+ uses non-causal masks that have no
// flash_attn_vec equivalent — these need a CPU reference SDPA that accepts
// an explicit mask buffer. CPU reference uses f64 internally; comparison
// is within an absolute-error tolerance, not bit-equality.

/// Generic CPU reference SDPA with explicit mask buffer.
///
/// Q: [num_heads, q_seq_len, head_dim]   ← MATCHES kernel input layout
///                                         (kernel reads Q at offset
///                                         `(iq2 * q_l + iq1) * DK`)
/// K: [num_kv_heads, kv_capacity, head_dim]   (first kv_seq_len valid)
/// V: [num_kv_heads, kv_capacity, head_dim]
/// mask: [q_seq_len, mask_stride]  cell (i, j) ∈ {TREE_MASK_ATTENDED,
///                                              TREE_MASK_MASKED}
///
/// Returns output in **`[q_seq_len, num_heads, head_dim]` layout** —
/// matches the GPU kernel's output where row index `rid = iq2 +
/// iq1 * n_heads`. Reading `output[iq1 * n_heads * dim + h * dim + d]`
/// is `(query, head, dim)` order.
///
/// Mirrors the math the GPU kernel performs: softmax over masked dot
/// products with `scale`. Uses f64 accumulator for the dot product +
/// softmax-renormalize chain to match the per-thread-precision the
/// Metal kernel achieves via on-chip half/float interleaving.
#[allow(clippy::too_many_arguments)]
fn cpu_tree_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_seq_len: usize,
    kv_seq_len: usize,
    kv_capacity: usize,
    mask_stride: usize,
    scale: f32,
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads;
    let mut output = vec![0.0f32; num_heads * q_seq_len * head_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let k_head_base = kv_h * kv_capacity * head_dim;
        let v_head_base = kv_h * kv_capacity * head_dim;

        for iq1 in 0..q_seq_len {
            let q_offset = h * q_seq_len * head_dim + iq1 * head_dim;
            let mask_row_base = iq1 * mask_stride;

            // Gather attended positions and their dot products.
            let mut scores = Vec::<(usize, f32)>::with_capacity(kv_seq_len);
            for k_pos in 0..kv_seq_len {
                let cell = mask[mask_row_base + k_pos];
                if cell == TREE_MASK_MASKED {
                    continue;
                }
                debug_assert_eq!(
                    cell, TREE_MASK_ATTENDED,
                    "mask cells must be one of the sentinel values"
                );

                let k_off = k_head_base + k_pos * head_dim;
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[q_offset + d] as f64 * k[k_off + d] as f64;
                }
                scores.push((k_pos, dot as f32 * scale));
            }

            if scores.is_empty() {
                continue; // all-masked row → zero output (kernel does the same)
            }

            // Softmax (max-renorm for stability).
            let max_score = scores
                .iter()
                .map(|(_, s)| *s)
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> =
                scores.iter().map(|(_, s)| (*s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let inv_sum = if sum_exp == 0.0 { 0.0 } else { 1.0 / sum_exp };

            // Weighted V sum. Output layout is [query, head, dim] —
            // matches kernel rid = iq2 + iq1 * n_heads writes.
            let o_offset = iq1 * num_heads * head_dim + h * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for ((k_pos, _), &exp_s) in scores.iter().zip(exp_scores.iter()) {
                    let weight = exp_s * inv_sum;
                    acc += weight * v[v_head_base + k_pos * head_dim + d];
                }
                output[o_offset + d] = acc;
            }
        }
    }

    output
}

/// Run tree_attention on GPU with the given mask, return outputs.
#[allow(clippy::too_many_arguments)]
fn run_tree_attention_gpu(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    mask_data: &[f32],
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    q_seq_len: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    mask_stride: u32,
    scale: f32,
) -> Vec<f32> {
    let q_elems = (num_heads as usize) * (q_seq_len as usize) * (head_dim as usize);
    let kv_elems = (num_kv_heads as usize) * (kv_capacity as usize) * (head_dim as usize);
    let mask_elems = (q_seq_len as usize) * (mask_stride as usize);
    let out_bytes = q_elems * 4;

    let mut q_buf = device
        .alloc_buffer(q_elems * 4, DType::F32, vec![q_elems])
        .expect("alloc Q");
    let mut k_buf = device
        .alloc_buffer(kv_elems * 4, DType::F32, vec![kv_elems])
        .expect("alloc K");
    let mut v_buf = device
        .alloc_buffer(kv_elems * 4, DType::F32, vec![kv_elems])
        .expect("alloc V");
    let mut mask_buf = device
        .alloc_buffer(mask_elems * 4, DType::F32, vec![mask_elems])
        .expect("alloc mask");
    let out_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elems])
        .expect("alloc output");

    q_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(q_data);
    k_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(k_data);
    v_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(v_data);
    mask_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(mask_data);

    let tmp_bytes = tree_attention::tmp_buffer_bytes(num_heads, head_dim, q_seq_len);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("alloc tmp");

    let params = TreeAttentionParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        q_seq_len,
        mask_stride,
    };
    let mut enc = device.command_encoder().expect("encoder");
    tree_attention::tree_attention(
        &mut enc,
        registry,
        device,
        &q_buf,
        &k_buf,
        &v_buf,
        &mask_buf,
        &out_buf,
        &tmp_buf,
        &params,
    )
    .expect("tree_attention dispatch");
    enc.commit_and_wait().expect("commit");

    out_buf.as_slice::<f32>().unwrap().to_vec()
}

/// Compare GPU output to CPU reference within an absolute-error
/// tolerance (CPU uses f64, GPU uses interleaved half/float — exact
/// bit-equality is not expected).
fn assert_close(gpu: &[f32], cpu: &[f32], epsilon: f32, label: &str) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: output length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    if max_diff > epsilon {
        panic!(
            "{label}: max abs diff {max_diff:.6e} at idx {max_idx} exceeds tolerance {epsilon:.6e} (gpu={:.6}, cpu={:.6})",
            gpu[max_idx], cpu[max_idx]
        );
    }
    eprintln!(
        "{label}: max abs diff {max_diff:.6e} (within {epsilon:.6e}); compared {} values",
        gpu.len()
    );
}

/// Build a "fixed-square" tree mask: root + N leaves over a prefix.
///
/// Tree layout (qL = 1 + n_leaves total tree-nodes):
///   - tree-node 0 (root):     attends prefix [0, prefix_len) + self
///   - tree-node 1..=n_leaves: attends prefix [0, prefix_len) + root + self
///
/// Tree-nodes occupy absolute positions [prefix_len, prefix_len + qL).
/// `kv_seq_len = prefix_len + qL`.
fn build_fixed_square_mask(
    prefix_len: usize,
    n_leaves: usize,
    mask_stride: usize,
) -> Vec<f32> {
    let q_seq_len = 1 + n_leaves;
    let kv_seq_len = prefix_len + q_seq_len;
    assert!(mask_stride >= kv_seq_len, "mask_stride must be >= kv_seq_len");

    let mut mask = vec![TREE_MASK_MASKED; q_seq_len * mask_stride];

    // Root: attends [0, prefix_len) + self at (prefix_len).
    let root_base = 0 * mask_stride;
    for k_pos in 0..prefix_len {
        mask[root_base + k_pos] = TREE_MASK_ATTENDED;
    }
    mask[root_base + prefix_len] = TREE_MASK_ATTENDED;

    // Leaves: each attends [0, prefix_len) + root + self.
    for leaf in 0..n_leaves {
        let iq1 = 1 + leaf; // tree-node index
        let row_base = iq1 * mask_stride;
        for k_pos in 0..prefix_len {
            mask[row_base + k_pos] = TREE_MASK_ATTENDED;
        }
        // Root is at absolute position `prefix_len`.
        mask[row_base + prefix_len] = TREE_MASK_ATTENDED;
        // Self is at absolute position `prefix_len + iq1`.
        mask[row_base + prefix_len + iq1] = TREE_MASK_ATTENDED;
    }

    mask
}

#[allow(clippy::too_many_arguments)]
fn assert_fixed_square_matches_cpu(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    prefix_len: usize,
    n_leaves: usize,
    kv_capacity: u32,
    scale: f32,
    seed: u64,
    epsilon: f32,
    label: &str,
) {
    let q_seq_len = (1 + n_leaves) as u32;
    let kv_seq_len = (prefix_len + q_seq_len as usize) as u32;
    let mask_stride = kv_seq_len;

    assert!(kv_capacity >= kv_seq_len, "{label}: kv_capacity too small");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let q_elems = (num_heads as usize) * (q_seq_len as usize) * (head_dim as usize);
    let kv_elems = (num_kv_heads as usize) * (kv_capacity as usize) * (head_dim as usize);

    let mut q_data = vec![0.0f32; q_elems];
    let mut k_data = vec![0.0f32; kv_elems];
    let mut v_data = vec![0.0f32; kv_elems];
    fill_random(&mut q_data, seed);
    fill_random(&mut k_data, seed + 10_000);
    fill_random(&mut v_data, seed + 20_000);

    let mask_data = build_fixed_square_mask(prefix_len, n_leaves, mask_stride as usize);

    // --- GPU ---
    let gpu_out = run_tree_attention_gpu(
        &device,
        &mut registry,
        &q_data,
        &k_data,
        &v_data,
        &mask_data,
        num_heads,
        num_kv_heads,
        head_dim,
        q_seq_len,
        kv_seq_len,
        kv_capacity,
        mask_stride,
        scale,
    );

    // --- CPU ---
    let cpu_out = cpu_tree_sdpa(
        &q_data,
        &k_data,
        &v_data,
        &mask_data,
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
        q_seq_len as usize,
        kv_seq_len as usize,
        kv_capacity as usize,
        mask_stride as usize,
        scale,
    );

    assert_close(&gpu_out, &cpu_out, epsilon, label);
}

#[test]
fn adr_037_e1_3_fixed_square_dk256_root4leaves_2026_05_22() {
    // Smallest non-trivial tree: 1 root + 4 leaves, depth=2, fanout=4.
    // Prefix of 27 tokens → kv_seq_len = 27 + 5 = 32 (exact C boundary).
    // Each leaf attends prefix + root + self (no sibling-to-sibling
    // attention — strict tree semantics).
    assert_fixed_square_matches_cpu(
        4,                          // num_heads
        4,                          // num_kv_heads
        256,                        // head_dim
        27,                         // prefix_len
        4,                          // n_leaves
        64,                         // kv_capacity
        1.0 / (256.0_f32).sqrt(),   // scale
        12321,                      // seed
        1e-2,                       // tolerance (matches existing crate convention)
        "dk256 fixed-square 4-leaf tree (prefix=27, qL=5)",
    );
}

#[test]
fn adr_037_e1_3_fixed_square_dk256_gqa_root4leaves_2026_05_22() {
    // GQA variant — 16 query heads sharing 8 KV heads. Validates that
    // the mask is keyed by query position (not KV head) and survives
    // the heads_per_kv mapping in the shader.
    assert_fixed_square_matches_cpu(
        16, 8, 256, 27, 4, 64,
        1.0,
        45654,
        1e-2,
        "dk256 GQA fixed-square 4-leaf tree",
    );
}

#[test]
fn adr_037_e1_3_fixed_square_dk512_root4leaves_2026_05_22() {
    assert_fixed_square_matches_cpu(
        4, 4, 512, 27, 4, 64,
        1.0 / (512.0_f32).sqrt(),
        78787,
        1e-2,
        "dk512 fixed-square 4-leaf tree",
    );
}

#[test]
fn adr_037_e1_3_fixed_square_dk256_long_prefix_2026_05_22() {
    // Long prefix (kv_seq_len = 507 + 5 = 512) exercises NWG > 1
    // reduce kernel path simultaneously with non-causal tree mask.
    // This is the closest synthetic test to production EAGLE-3 long
    // context with a small tree on top.
    assert_fixed_square_matches_cpu(
        4, 4, 256, 507, 4, 1024,
        1.0 / (256.0_f32).sqrt(),
        91919,
        1e-2,
        "dk256 fixed-square 4-leaf tree (long prefix, kv=512)",
    );
}

// --------------------------------------------------------------------------
// Phase E1.4 — dynamic asymmetric tree vs CPU reference
// --------------------------------------------------------------------------
//
// Validates that the mask buffer correctly encodes ARBITRARY tree
// topologies, not just regular ones. Mirrors the kind of tree EAGLE-2
// dynamic expansion produces (top-K branches by confidence, with
// uneven branching factor per depth).

/// Build a tree mask from a parents array.
///
/// `parents[i]` is `Some(parent_idx)` for non-root nodes, `None` for
/// the root. Tree-node `i` lives at absolute KV position `prefix_len + i`.
/// Each tree-node attends the full prefix + itself + all ancestors.
fn build_tree_mask_from_parents(
    prefix_len: usize,
    parents: &[Option<usize>],
    mask_stride: usize,
) -> Vec<f32> {
    let q_seq_len = parents.len();
    let kv_seq_len = prefix_len + q_seq_len;
    assert!(mask_stride >= kv_seq_len);
    // Exactly one root.
    assert_eq!(
        parents.iter().filter(|p| p.is_none()).count(),
        1,
        "tree must have exactly one root"
    );

    let mut mask = vec![TREE_MASK_MASKED; q_seq_len * mask_stride];

    for iq1 in 0..q_seq_len {
        let row_base = iq1 * mask_stride;
        // Prefix: always attended.
        for k_pos in 0..prefix_len {
            mask[row_base + k_pos] = TREE_MASK_ATTENDED;
        }
        // Self + all ancestors.
        let mut cur = Some(iq1);
        while let Some(node) = cur {
            mask[row_base + prefix_len + node] = TREE_MASK_ATTENDED;
            cur = parents[node];
        }
    }

    mask
}

#[allow(clippy::too_many_arguments)]
fn assert_tree_matches_cpu(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    prefix_len: usize,
    parents: &[Option<usize>],
    kv_capacity: u32,
    scale: f32,
    seed: u64,
    epsilon: f32,
    label: &str,
) {
    let q_seq_len = parents.len() as u32;
    let kv_seq_len = (prefix_len + q_seq_len as usize) as u32;
    let mask_stride = kv_seq_len;

    assert!(kv_capacity >= kv_seq_len, "{label}: kv_capacity too small");

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let q_elems = (num_heads as usize) * (q_seq_len as usize) * (head_dim as usize);
    let kv_elems = (num_kv_heads as usize) * (kv_capacity as usize) * (head_dim as usize);

    let mut q_data = vec![0.0f32; q_elems];
    let mut k_data = vec![0.0f32; kv_elems];
    let mut v_data = vec![0.0f32; kv_elems];
    fill_random(&mut q_data, seed);
    fill_random(&mut k_data, seed + 10_000);
    fill_random(&mut v_data, seed + 20_000);

    let mask_data = build_tree_mask_from_parents(prefix_len, parents, mask_stride as usize);

    let gpu_out = run_tree_attention_gpu(
        &device,
        &mut registry,
        &q_data,
        &k_data,
        &v_data,
        &mask_data,
        num_heads,
        num_kv_heads,
        head_dim,
        q_seq_len,
        kv_seq_len,
        kv_capacity,
        mask_stride,
        scale,
    );

    let cpu_out = cpu_tree_sdpa(
        &q_data,
        &k_data,
        &v_data,
        &mask_data,
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
        q_seq_len as usize,
        kv_seq_len as usize,
        kv_capacity as usize,
        mask_stride as usize,
        scale,
    );

    assert_close(&gpu_out, &cpu_out, epsilon, label);
}

/// Canonical asymmetric tree for E1.4.
///
/// Topology (8 nodes, max depth 4):
///   0 (root)
///   ├── 1 ── 4 ── 7
///   │   └── 5
///   ├── 2 ── 6
///   └── 3
///
/// Branching factor by depth: 3, varies (2, 1, 0), 1, 0.
fn asymmetric_tree_parents() -> Vec<Option<usize>> {
    vec![
        None,    // 0 = root
        Some(0), // 1 ← 0
        Some(0), // 2 ← 0
        Some(0), // 3 ← 0
        Some(1), // 4 ← 1
        Some(1), // 5 ← 1
        Some(2), // 6 ← 2
        Some(4), // 7 ← 4
    ]
}

#[test]
fn adr_037_e1_4_dynamic_asymmetric_dk256_2026_05_22() {
    let parents = asymmetric_tree_parents();
    assert_tree_matches_cpu(
        4, 4, 256,
        24,                         // prefix_len → kv = 24 + 8 = 32
        &parents,
        64,
        1.0 / (256.0_f32).sqrt(),
        24681,
        1e-2,
        "dk256 dynamic asymmetric tree (8 nodes, varying branching)",
    );
}

#[test]
fn adr_037_e1_4_dynamic_asymmetric_dk256_gqa_2026_05_22() {
    let parents = asymmetric_tree_parents();
    assert_tree_matches_cpu(
        16, 8, 256,
        24,
        &parents,
        64,
        1.0,
        13579,
        1e-2,
        "dk256 GQA dynamic asymmetric tree",
    );
}

#[test]
fn adr_037_e1_4_dynamic_asymmetric_dk512_2026_05_22() {
    let parents = asymmetric_tree_parents();
    assert_tree_matches_cpu(
        4, 4, 512,
        24,
        &parents,
        64,
        1.0 / (512.0_f32).sqrt(),
        86420,
        1e-2,
        "dk512 dynamic asymmetric tree",
    );
}

#[test]
fn adr_037_e1_4_dynamic_asymmetric_chain_root_2026_05_22() {
    // Degenerate tree: linear chain [0 → 1 → 2 → 3 → 4]. Should
    // reduce to causal qL=5 at kv_seq_len=32 — but goes through
    // the explicit-mask path. Equivalent topology to the test in
    // E1.2 chain_parity at qL=5, validated via CPU reference here
    // (not byte-identity since CPU uses f64).
    let parents: Vec<Option<usize>> = vec![None, Some(0), Some(1), Some(2), Some(3)];
    assert_tree_matches_cpu(
        4, 4, 256,
        27,
        &parents,
        64,
        1.0 / (256.0_f32).sqrt(),
        97531,
        1e-2,
        "dk256 chain-as-degenerate-tree (5-node line, CPU ref)",
    );
}

// --------------------------------------------------------------------------
// Phase E1.5 — prefix+tree combined (long prefix + small tree on top)
// --------------------------------------------------------------------------
//
// Production-shaped: long context (kv ~512+) with small tree budget
// on top. Stress-tests interaction of NWG>1 reduce path, multi-query
// kernel, and arbitrary tree mask all at once.

#[test]
fn adr_037_e1_5_prefix_plus_tree_dk256_long_2026_05_22() {
    // 504-token natural prefix + 8-node asymmetric tree = kv 512.
    // Mirrors production EAGLE-3 at moderately long context.
    let parents = asymmetric_tree_parents();
    assert_tree_matches_cpu(
        4, 4, 256,
        504,                         // long prefix → kv = 512
        &parents,
        1024,
        1.0 / (256.0_f32).sqrt(),
        11_223_344,
        1e-2,
        "dk256 prefix+tree combined (prefix=504, tree=8, kv=512)",
    );
}

#[test]
fn adr_037_e1_5_prefix_plus_tree_dk256_gqa_long_2026_05_22() {
    // Same as above but with production GQA shape (e.g. Qwen 3.6 27B
    // dense uses 64 q-heads / 8 kv-heads in many configs).
    let parents = asymmetric_tree_parents();
    assert_tree_matches_cpu(
        16, 8, 256,
        504,
        &parents,
        1024,
        1.0,
        55_667_788,
        1e-2,
        "dk256 GQA prefix+tree combined (prefix=504, tree=8, kv=512)",
    );
}

#[test]
fn adr_037_e1_5_prefix_plus_tree_dk512_long_2026_05_22() {
    // dk512 variant for full kernel-template coverage.
    let parents = asymmetric_tree_parents();
    assert_tree_matches_cpu(
        4, 4, 512,
        504,
        &parents,
        1024,
        1.0 / (512.0_f32).sqrt(),
        99_887_766,
        1e-2,
        "dk512 prefix+tree combined (prefix=504, tree=8, kv=512)",
    );
}
