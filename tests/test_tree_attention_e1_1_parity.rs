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
