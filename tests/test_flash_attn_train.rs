//! Integration tests for the flash-attention training forward kernel.
//!
//! Tests `dispatch_flash_attn_train_fwd_bf16_d64` and
//! `dispatch_flash_attn_train_fwd_bf16_d256` against a CPU oracle that
//! computes BOTH the attention output `O` AND the per-row natural-log
//! logsumexp `L`.
//!
//! # CPU oracle design
//!
//! `sdpa_reference_with_logsumexp` mirrors the GPU kernel step-for-step:
//!
//! - Q pre-scaled by `scale * log2(e)` before QK^T.
//! - Online softmax via `f32::exp2(x - max)` (base-2, llama.cpp sentinel).
//! - L[b, h, i] computed from the final row max + log(sum_exp):
//!   `L = max_base2 * ln(2) + ln(sum_exp)` — FA-2 Algorithm 1 convention.
//! - O normalised by `sum_exp`; guard: fully-masked row → O = 0.
//!
//! # Tolerance
//!
//! GPU bf16 O vs CPU f32 O: `atol=5e-3` (bf16 input rounding, f32 MMA).
//! GPU f32 L vs CPU f32 L: `atol=3e-3` — L is written in f32 by the GPU.
//! The dot-product scores differ by ~1e-3 between GPU simdgroup MMA (with
//! non-deterministic reduction order across 64 or 256 bf16 elements) and
//! the CPU scalar loop (fixed left-to-right f32 accumulation).  That score
//! delta propagates into max_score and sum_score, giving L errors up to
//! ~2.2e-3 for D=256 at kL=128.  O errors are hidden by the final bf16
//! store; L errors are fully visible because L is stored in f32.

// ─── macOS guard ─────────────────────────────────────────────────────────────
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
mod flash_attn_train_tests {

use half::bf16;
use mlx_native::ops::flash_attn_train::{
    self as flash_attn_train,
    FlashAttnTrainParams,
    dispatch_flash_attn_train_fwd_bf16_d64,
    dispatch_flash_attn_train_fwd_bf16_d256,
    dispatch_flash_attn_train_bwd_bf16_d64,
    dispatch_flash_attn_train_bwd_bf16_d256,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ─────────────────────────────────────────────────────────────────────────────
// CPU oracle — forward O + logsumexp L
// ─────────────────────────────────────────────────────────────────────────────

/// CPU reference that mirrors the GPU kernel math exactly, and also
/// returns the per-row natural-log logsumexp L.
///
/// Returns `(O, L)` where:
/// - `O` has shape `[batch * n_heads * ql * head_dim]` (f32)
/// - `L` has shape `[batch * n_heads * ql]` (f32)
///
/// L[b, h, i] = max_base2 * ln(2) + ln(sum_exp_base2)
///
/// This is the FA-2 Algorithm 1 logsumexp in natural-log domain:
///   L_i = m_i + log(sum_j exp(s_ij - m_i))
/// where m_i = max_base2 * ln(2) is the base-2 max converted to nat-log.
///
/// Fully-masked rows: sum_exp = 0 → L = -inf (correct; backward skips them).
fn sdpa_reference_with_logsumexp(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    const LOG2E: f32 = std::f32::consts::LOG2_E; // 1 / ln(2) ≈ 1.44269504
    const LN2: f32 = std::f32::consts::LN_2;      // ln(2) ≈ 0.693147181

    let q_scale = scale * LOG2E; // Q pre-scaled by scale * log2(e)
    let heads_per_kv = n_heads / n_kv_heads;

    let mut out = vec![0.0f32; batch * n_heads * ql * head_dim];
    let mut lse = vec![0.0f32; batch * n_heads * ql];

    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            for q_pos in 0..ql {
                let q_base = b * n_heads * ql * head_dim + h * ql * head_dim + q_pos * head_dim;
                let kv_base = b * n_kv_heads * kl * head_dim + kv_h * kl * head_dim;

                // QK^T in base-2 scale
                let mut scores = vec![0.0f32; kl];
                for k_pos in 0..kl {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[kv_base + k_pos * head_dim + d];
                    }
                    scores[k_pos] = dot * q_scale;
                }

                // Additive mask (natural-log scale → ×log2(e) for base-2 space)
                if let Some(m) = mask {
                    let m_base = b * n_heads * ql * kl + h * ql * kl + q_pos * kl;
                    for k_pos in 0..kl {
                        scores[k_pos] += LOG2E * m[m_base + k_pos];
                    }
                }

                // Causal mask
                if do_causal {
                    let q_abs = kl.saturating_sub(ql) + q_pos;
                    for (k_pos, score) in scores.iter_mut().enumerate() {
                        if k_pos > q_abs {
                            *score = f32::NEG_INFINITY;
                        }
                    }
                }

                // Online softmax (base-2, finite-M sentinel = -FLT_MAX/2)
                let mut max_b2 = f32::MIN / 2.0; // = -FLT_MAX/2 (llama.cpp convention)
                for &s in &scores {
                    if s > max_b2 { max_b2 = s; }
                }

                let exp_scores: Vec<f32> = scores.iter()
                    .map(|&s| f32::exp2(s - max_b2))
                    .collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                // L_i = max_b2 * ln(2) + ln(sum_exp)
                // When sum_exp == 0 (fully masked): ln(0) = -inf (correct).
                let lse_val = max_b2 * LN2 + sum_exp.ln();
                lse[b * n_heads * ql + h * ql + q_pos] = lse_val;

                // O = weighted sum of V (single-pass, not multi-tile)
                let safe_sum = if sum_exp == 0.0 { 1.0 } else { sum_exp };
                let o_base = b * n_heads * ql * head_dim + h * ql * head_dim + q_pos * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for k_pos in 0..kl {
                        acc += (exp_scores[k_pos] / safe_sum)
                            * v[kv_base + k_pos * head_dim + d];
                    }
                    out[o_base + d] = acc;
                }
            }
        }
    }

    (out, lse)
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Seed for all tests — 0x545241494E303031 = "TRAIN001" in ASCII.
const SEED_VAL: u64 = 0x5452_4149_4E30_3031;

/// LCG pseudo-random f32 in `[-0.5, 0.5]`.
fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn f32_to_bf16(xs: &[f32]) -> Vec<bf16> {
    xs.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn bf16_to_f32(xs: &[bf16]) -> Vec<f32> {
    xs.iter().map(|&x| x.to_f32()).collect()
}

fn alloc_bf16(device: &MlxDevice, elems: usize, name: &str) -> mlx_native::MlxBuffer {
    device
        .alloc_buffer(elems * 2, DType::BF16, vec![elems])
        .unwrap_or_else(|e| panic!("alloc_bf16({name}, {elems}): {e:?}"))
}

fn alloc_f32(device: &MlxDevice, elems: usize, name: &str) -> mlx_native::MlxBuffer {
    device
        .alloc_buffer(elems * 4, DType::F32, vec![elems])
        .unwrap_or_else(|e| panic!("alloc_f32({name}, {elems}): {e:?}"))
}

fn fill_bf16_buf(buf: &mlx_native::MlxBuffer, data: &[bf16]) {
    let ptr = buf.contents_ptr() as *mut bf16;
    assert!(!ptr.is_null(), "contents_ptr is null");
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()); }
}

fn read_bf16_buf(buf: &mlx_native::MlxBuffer, elems: usize) -> Vec<bf16> {
    let ptr = buf.contents_ptr() as *const bf16;
    assert!(!ptr.is_null(), "contents_ptr is null");
    unsafe { std::slice::from_raw_parts(ptr, elems).to_vec() }
}

fn read_f32_buf(buf: &mlx_native::MlxBuffer, elems: usize) -> Vec<f32> {
    let ptr = buf.contents_ptr() as *const f32;
    assert!(!ptr.is_null(), "contents_ptr is null");
    unsafe { std::slice::from_raw_parts(ptr, elems).to_vec() }
}

/// Assert elementwise closeness (absolute tolerance only).
fn assert_close(actual: &[f32], expected: &[f32], atol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut worst_idx = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff { max_diff = diff; worst_idx = i; }
    }
    assert!(
        max_diff <= atol,
        "{label}: max_abs_error={max_diff:.4e} at index {worst_idx} \
         (actual={:.6}, expected={:.6}) exceeds atol={atol:.4e}",
        actual[worst_idx], expected[worst_idx]
    );
    eprintln!("{label}: PASS  max_abs_err={max_diff:.4e}  (atol={atol:.4e})");
}

/// Run the training-forward kernel and return (O_f32, L_f32).
///
/// The CPU oracle is run on bf16-rounded inputs so the reference has the same
/// input precision as the GPU.
#[allow(clippy::too_many_arguments)]
fn run_train_fwd(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_bf: &[bf16],
    k_bf: &[bf16],
    v_bf: &[bf16],
    mask_bf: Option<&[bf16]>,
    batch: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let q_elems = batch * n_q_heads * ql * head_dim;
    let kv_elems = batch * n_kv_heads * kl * head_dim;
    let l_elems = batch * n_q_heads * ql;

    let q_buf = alloc_bf16(device, q_elems, "Q");
    let k_buf = alloc_bf16(device, kv_elems, "K");
    let v_buf = alloc_bf16(device, kv_elems, "V");
    let mut o_buf = alloc_bf16(device, q_elems, "O");
    let mut l_buf = alloc_f32(device, l_elems, "L");

    fill_bf16_buf(&q_buf, q_bf);
    fill_bf16_buf(&k_buf, k_bf);
    fill_bf16_buf(&v_buf, v_bf);

    let mask_buf = mask_bf.map(|m| {
        let mask_elems = batch * n_q_heads * ql * kl;
        assert_eq!(m.len(), mask_elems, "mask length mismatch");
        let buf = alloc_bf16(device, mask_elems, "mask");
        fill_bf16_buf(&buf, m);
        buf
    });

    let params = FlashAttnTrainParams {
        batch: batch as u32,
        n_q_heads: n_q_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        q_seq_len: ql as u32,
        k_seq_len: kl as u32,
        scale,
        causal,
    };

    let mut encoder = device.command_encoder().expect("encoder");

    if head_dim == 64 {
        dispatch_flash_attn_train_fwd_bf16_d64(
            &mut encoder, device, registry,
            &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut o_buf, &mut l_buf,
            &params,
        ).expect("dispatch d64");
    } else {
        dispatch_flash_attn_train_fwd_bf16_d256(
            &mut encoder, device, registry,
            &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut o_buf, &mut l_buf,
            &params,
        ).expect("dispatch d256");
    }

    encoder.commit_and_wait().expect("commit_and_wait");

    let o_bf16 = read_bf16_buf(&o_buf, q_elems);
    let o_f32 = bf16_to_f32(&o_bf16);
    let l_f32 = read_f32_buf(&l_buf, l_elems);

    (o_f32, l_f32)
}

/// Compute CPU oracle on bf16-rounded inputs.
///
/// Widens bf16→f32 so the reference has the same precision loss as GPU inputs.
#[allow(clippy::too_many_arguments)]
fn oracle_for_bf16(
    q_bf: &[bf16],
    k_bf: &[bf16],
    v_bf: &[bf16],
    mask_bf: Option<&[bf16]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let q = bf16_to_f32(q_bf);
    let k = bf16_to_f32(k_bf);
    let v = bf16_to_f32(v_bf);
    let mask = mask_bf.map(bf16_to_f32);

    let (o_f32, l_f32) = sdpa_reference_with_logsumexp(
        &q, &k, &v, mask.as_deref(),
        batch, n_heads, n_kv_heads, ql, kl, head_dim, scale, causal,
    );

    // Round-trip O through bf16 to simulate the GPU's final store.
    let o_bf16 = f32_to_bf16(&o_f32);
    let o_rt = bf16_to_f32(&o_bf16);

    // L is stored as f32 by the GPU — no round-trip needed.
    (o_rt, l_f32)
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    flash_attn_train::register(&mut registry);
    (device, registry)
}

// ─────────────────────────────────────────────────────────────────────────────
// § 1  REGISTRATION / COMPILATION
// ─────────────────────────────────────────────────────────────────────────────

/// All 4 kernel names are registered and the D=64 / D=256 pipelines compile.
#[test]
fn test_kernel_names_and_library_compiles() {
    let (device, mut registry) = setup();

    // Check all names are in the registry by requesting their pipelines.
    for &name in flash_attn_train::all_kernel_names_for_test() {
        let result = registry.get_pipeline_with_bool_constants(
            name,
            device.metal_device(),
            &[(200, true), (201, true), (300, false), (301, false)],
        );
        match result {
            Ok(_) => eprintln!("test_kernel_names_and_library_compiles: {name} OK"),
            Err(e) => panic!("Pipeline compilation failed for {name}: {e:?}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  O PARITY — D=64
// ─────────────────────────────────────────────────────────────────────────────

/// GPU O matches CPU oracle O for D=64 unmasked, non-causal.
///
/// Shape: B=1, H=1, qL=32, kL=32, D=64.
/// Tolerance: atol=5e-3 (bf16 I/O rounding).
#[test]
fn test_forward_o_parity_d64_no_mask() {
    let (device, mut registry) = setup();

    let batch = 1; let h = 1; let kv_h = 1; let ql = 32; let kl = 32; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED_VAL, batch * h * ql * d);
    let k = pseudo_random_f32(SEED_VAL + 1, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED_VAL + 2, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let (gpu_o, _) = run_train_fwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let (ref_o, _) = oracle_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_o, &ref_o, 5e-3, "forward_o_parity_d64_no_mask");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  L PARITY — D=64 (the new feature)
// ─────────────────────────────────────────────────────────────────────────────

/// GPU L matches CPU oracle L for D=64 unmasked, non-causal.
///
/// Shape: B=1, H=1, qL=32, kL=32, D=64.
/// Tolerance: atol=3e-3 — L is f32 on GPU; MMA vs scalar dot-product order
/// difference contributes.
#[test]
fn test_forward_l_parity_d64_no_mask() {
    let (device, mut registry) = setup();

    let batch = 1; let h = 1; let kv_h = 1; let ql = 32; let kl = 32; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED_VAL + 10, batch * h * ql * d);
    let k = pseudo_random_f32(SEED_VAL + 11, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED_VAL + 12, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let (_, gpu_l) = run_train_fwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let (_, ref_l) = oracle_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_l, &ref_l, 3e-3, "forward_l_parity_d64_no_mask");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  O AND L PARITY — D=256
// ─────────────────────────────────────────────────────────────────────────────

/// GPU O and L match CPU oracle for D=256 unmasked, non-causal.
///
/// Shape: B=1, H=4, qL=128, kL=128, D=256.
/// Production Qwen3.6-35B-A3B head dimension.
#[test]
fn test_forward_o_l_parity_d256_no_mask() {
    let (device, mut registry) = setup();

    let batch = 1; let h = 4; let kv_h = 4; let ql = 128; let kl = 128; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED_VAL + 20, batch * h * ql * d);
    let k = pseudo_random_f32(SEED_VAL + 21, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED_VAL + 22, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let (gpu_o, gpu_l) = run_train_fwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let (ref_o, ref_l) = oracle_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_o, &ref_o, 5e-3, "d256_no_mask_O");
    assert_close(&gpu_l, &ref_l, 3e-3, "d256_no_mask_L");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  CAUSAL MASK — O AND L PARITY
// ─────────────────────────────────────────────────────────────────────────────

/// GPU O and L match CPU oracle with causal masking, D=64.
///
/// Shape: B=1, H=2, qL=64, kL=64, D=64, causal=true.
#[test]
fn test_forward_causal_mask_parity() {
    let (device, mut registry) = setup();

    let batch = 1; let h = 2; let kv_h = 2; let ql = 64; let kl = 64; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED_VAL + 30, batch * h * ql * d);
    let k = pseudo_random_f32(SEED_VAL + 31, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED_VAL + 32, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let (gpu_o, gpu_l) = run_train_fwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, true,
    );
    let (ref_o, ref_l) = oracle_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, true,
    );

    assert_close(&gpu_o, &ref_o, 5e-3, "causal_mask_O");
    assert_close(&gpu_l, &ref_l, 3e-3, "causal_mask_L");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  CAUSAL MASK STRICT — lower-triangular independence
// ─────────────────────────────────────────────────────────────────────────────

/// O[i] depends only on K[0..=i] and V[0..=i] under causal masking.
///
/// Construction: K rows j > q_pos_target are set to a large sentinel value
/// (128.0) that would dominate the softmax if they were attended.  With
/// correct causal masking, those rows contribute nothing.
///
/// The test runs the GPU kernel and verifies O[0..=q_pos_target] matches the
/// oracle computed on the unperturbed K (same Q, K rows 0..=q_pos_target,
/// V unperturbed).  If the GPU O diverges, the causal mask has an off-by-one.
#[test]
fn test_forward_causal_mask_strict_lower_triangular() {
    let (device, mut registry) = setup();

    let batch = 1; let h = 1; let kv_h = 1; let ql = 32; let kl = 32; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();
    let q_pos_target = 15usize; // the query row we specifically inspect

    let q_f32 = pseudo_random_f32(SEED_VAL + 40, batch * h * ql * d);
    let k_clean_f32 = pseudo_random_f32(SEED_VAL + 41, batch * kv_h * kl * d);
    let v_f32 = pseudo_random_f32(SEED_VAL + 42, batch * kv_h * kl * d);

    // Perturb K: set rows j > q_pos_target to sentinel 128.0 (in f32 space).
    let mut k_perturbed_f32 = k_clean_f32.clone();
    for j in (q_pos_target + 1)..kl {
        for dd in 0..d {
            k_perturbed_f32[j * d + dd] = 128.0_f32;
        }
    }

    let q_bf = f32_to_bf16(&q_f32);
    let k_clean_bf = f32_to_bf16(&k_clean_f32);
    let k_perturbed_bf = f32_to_bf16(&k_perturbed_f32);
    let v_bf = f32_to_bf16(&v_f32);

    // Run GPU with perturbed K (causal mask should exclude the sentinel rows).
    let (gpu_o_perturbed, _) = run_train_fwd(
        &device, &mut registry, &q_bf, &k_perturbed_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, true,
    );

    // CPU oracle with CLEAN K (causal mask masks the same future rows).
    let (ref_o_clean, _) = oracle_for_bf16(
        &q_bf, &k_clean_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, true,
    );

    // Compare O[0..=q_pos_target]: GPU with perturbed K should equal CPU with clean K.
    // Rows beyond q_pos_target may differ (they attend to different K rows anyway).
    let target_elems = (q_pos_target + 1) * d;
    let gpu_prefix = &gpu_o_perturbed[..target_elems];
    let ref_prefix = &ref_o_clean[..target_elems];

    assert_close(gpu_prefix, ref_prefix, 5e-3,
        "causal_strict_lower_triangular (rows 0..=q_pos_target)");
    eprintln!(
        "test_forward_causal_mask_strict_lower_triangular: PASS — \
         O[0..={q_pos_target}] independent of K rows [{}..]",
        q_pos_target + 1
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  SLIDING-WINDOW MASK (SWA) PARITY
// ─────────────────────────────────────────────────────────────────────────────

/// GPU O and L match oracle with sliding-window attention mask.
///
/// SWA is implemented via the additive mask buffer (has_mask=true) rather than
/// a separate function constant — the mask encodes -inf for positions
/// outside the window `[i - window, i]`.  This matches how hf2q's SWA layers
/// build their mask for inference, so the same path is exercised here.
///
/// Shape: B=1, H=2, qL=64, kL=64, D=64, window=16.
/// Mask: additive bf16, 0.0 for j in [i-16..=i], -inf otherwise.
#[test]
fn test_forward_sliding_window_parity() {
    let (device, mut registry) = setup();

    let batch = 1; let h = 2; let kv_h = 2; let ql = 64; let kl = 64; let d = 64;
    let window = 16usize;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED_VAL + 50, batch * h * ql * d);
    let k = pseudo_random_f32(SEED_VAL + 51, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED_VAL + 52, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Build SWA additive mask: 0.0 if j <= i && j >= i - window, else -inf.
    // No causal component — the mask itself encodes both past-limit and future.
    let mut mask_f32 = vec![f32::NEG_INFINITY; batch * h * ql * kl];
    for b in 0..batch {
        for hh in 0..h {
            for i in 0..ql {
                for j in 0..kl {
                    let in_window = (j <= i) && (j + window >= i);
                    let idx = b * h * ql * kl + hh * ql * kl + i * kl + j;
                    mask_f32[idx] = if in_window { 0.0_f32 } else { f32::NEG_INFINITY };
                }
            }
        }
    }
    let mask_bf = f32_to_bf16(&mask_f32);

    let (gpu_o, gpu_l) = run_train_fwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let (ref_o, ref_l) = oracle_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_o, &ref_o, 5e-3, "swa_O");
    assert_close(&gpu_l, &ref_l, 3e-3, "swa_L");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  GQA PARITY
// ─────────────────────────────────────────────────────────────────────────────

/// GPU O and L match oracle with grouped-query attention (gqa_factor=4).
///
/// Shape: B=1, H_q=8, H_kv=2, qL=64, kL=64, D=64.
/// Each pair of Q heads shares one KV head.
#[test]
fn test_forward_gqa_parity() {
    let (device, mut registry) = setup();

    let batch = 1; let h = 8; let kv_h = 2; let ql = 64; let kl = 64; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED_VAL + 60, batch * h * ql * d);
    let k = pseudo_random_f32(SEED_VAL + 61, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED_VAL + 62, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let (gpu_o, gpu_l) = run_train_fwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let (ref_o, ref_l) = oracle_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_o, &ref_o, 5e-3, "gqa_4x_O");
    assert_close(&gpu_l, &ref_l, 3e-3, "gqa_4x_L");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  LIBRARY COMPILATION GUARD
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that adding L_out (buffer 8) does not cause threadgroup-memory
/// overflow — the D=256 bf16 kernel must still compile on Apple Silicon.
///
/// If the shader's threadgroup memory grew beyond 32 KB this test fails with
/// `ShaderCompilationError` mentioning threadgroup memory.
#[test]
fn test_d256_library_compiles_with_l_out() {
    let (device, mut registry) = setup();

    let result = registry.get_pipeline_with_bool_constants(
        "flash_attn_train_fwd_bf16_d256",
        device.metal_device(),
        &[(200, true), (201, true), (300, false), (301, false)],
    );
    match result {
        Ok(_) => eprintln!("test_d256_library_compiles_with_l_out: OK"),
        Err(mlx_native::MlxError::ShaderCompilationError { name, message }) => {
            panic!(
                "D=256 bf16 train fwd failed to compile — if this mentions \
                 threadgroup memory, the L_out buffer likely pushed the tile over \
                 32 KB (it shouldn't: L_out is device memory, not threadgroup). \
                 name={name}, message={message}"
            );
        }
        Err(e) => panic!("Unexpected error: {e:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — backward kernel helpers
// ─────────────────────────────────────────────────────────────────────────────

/// CPU oracle for the FA-2 backward pass.
///
/// Given Q/K/V/O (bf16-rounded inputs, f32 values), the forward logsumexp L,
/// and the upstream gradient dO, computes (dQ, dK, dV) in f32.
///
/// Implements FA-2 Algorithm 4 equations:
///   D[i]      = rowsum(O[i] * dO[i])
///   S[i,j]    = scale * Q[i] · K[j]^T
///   P[i,j]    = exp(S[i,j] - L[i])
///   dV[j]    += P[i,j] * dO[i]
///   dP[i,j]   = dO[i] · V[j]^T
///   dS[i,j]   = P[i,j] * (dP[i,j] - D[i])
///   dS[i,j]   = 0  if causal and j > i  (already 0 through P)
///   dQ[i]    += scale * dS[i,j] * K[j]
///   dK[j]    += scale * dS[i,j] * Q[i]
///
/// All index conventions: [b * n_heads * seq * d] row-major, GQA via heads_per_kv.
#[allow(clippy::too_many_arguments)]
fn sdpa_backward_reference_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    l_nat: &[f32],
    do_: &[f32],
    mask: Option<&[f32]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let heads_per_kv = n_heads / n_kv_heads;
    let mut dq = vec![0.0f32; batch * n_heads * ql * head_dim];
    let mut dk = vec![0.0f32; batch * n_kv_heads * kl * head_dim];
    let mut dv = vec![0.0f32; batch * n_kv_heads * kl * head_dim];

    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            for q_i in 0..ql {
                let q_base = b * n_heads * ql * head_dim + h * ql * head_dim + q_i * head_dim;
                let l_i = l_nat[b * n_heads * ql + h * ql + q_i];

                // D[i] = sum_d O[i,d] * dO[i,d]  (O derived from forward; use dO·O trick)
                // Actually the kernel receives dO (upstream) and O (forward output).
                // For the oracle we compute D directly from the re-derived P:
                // D[i] = sum_j P[i,j] * dP[i,j]  — but that's equivalent.
                // Simpler: just store and reuse P below.

                // Compute raw scores S[i, 0..kl]
                let mut s = vec![0.0f32; kl];
                for k_j in 0..kl {
                    let kv_base = b * n_kv_heads * kl * head_dim
                        + kv_h * kl * head_dim
                        + k_j * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[kv_base + d];
                    }
                    s[k_j] = scale * dot;

                    // additive mask
                    if let Some(m) = mask {
                        let m_idx = b * n_heads * ql * kl + h * ql * kl + q_i * kl + k_j;
                        if m[m_idx] == f32::NEG_INFINITY {
                            s[k_j] = f32::NEG_INFINITY;
                        } else {
                            s[k_j] += m[m_idx];
                        }
                    }
                    // causal mask
                    if do_causal && k_j > q_i {
                        s[k_j] = f32::NEG_INFINITY;
                    }
                }

                // P[i, j] = exp(S[i,j] - L[i])
                let mut p = vec![0.0f32; kl];
                for k_j in 0..kl {
                    p[k_j] = if s[k_j] == f32::NEG_INFINITY {
                        0.0f32
                    } else {
                        (s[k_j] - l_i).exp()
                    };
                }

                // D[i] = sum_d dO[i,d] * O[i,d]
                // Since O[i,d] = sum_j P[i,j] * V[j,d], D[i] = sum_j P[i,j] * dP[i,j]
                // where dP[i,j] = dO[i] · V[j].  Compute D via sum_j P * dP:
                let mut d_i = 0.0f32;
                for k_j in 0..kl {
                    let kv_base = b * n_kv_heads * kl * head_dim
                        + kv_h * kl * head_dim
                        + k_j * head_dim;
                    let mut dp_j = 0.0f32;
                    for d in 0..head_dim {
                        dp_j += do_[q_base + d] * v[kv_base + d];
                    }
                    d_i += p[k_j] * dp_j;
                }

                for k_j in 0..kl {
                    let kv_base = b * n_kv_heads * kl * head_dim
                        + kv_h * kl * head_dim
                        + k_j * head_dim;
                    let dk_base = b * n_kv_heads * kl * head_dim
                        + kv_h * kl * head_dim
                        + k_j * head_dim;
                    let dv_base = dk_base;

                    // dP[i,j] = sum_d dO[i,d] * V[j,d]
                    let mut dp_j = 0.0f32;
                    for d in 0..head_dim {
                        dp_j += do_[q_base + d] * v[kv_base + d];
                    }

                    // dS[i,j] = P[i,j] * (dP[i,j] - D[i])
                    let ds_j = p[k_j] * (dp_j - d_i);

                    // dV[j] += P[i,j] * dO[i]
                    for d in 0..head_dim {
                        dv[dv_base + d] += p[k_j] * do_[q_base + d];
                    }

                    // dQ[i] += scale * dS[i,j] * K[j]
                    // dK[j] += scale * dS[i,j] * Q[i]
                    for d in 0..head_dim {
                        dq[q_base + d] += scale * ds_j * k[kv_base + d];
                        dk[dk_base + d] += scale * ds_j * q[q_base + d];
                    }
                }
            }
        }
    }

    (dq, dk, dv)
}

/// Run the full backward GPU chain and return (dQ_f32, dK_f32, dV_f32).
///
/// Runs forward first to get O and L, then runs backward.
/// All tensors are bf16-quantized before dispatch to match GPU precision.
#[allow(clippy::too_many_arguments)]
fn run_train_bwd(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_bf: &[bf16],
    k_bf: &[bf16],
    v_bf: &[bf16],
    mask_bf: Option<&[bf16]>,
    do_bf: &[bf16],
    batch: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_elems = batch * n_q_heads * ql * head_dim;
    let kv_elems = batch * n_kv_heads * kl * head_dim;
    let l_elems = batch * n_q_heads * ql;

    // Allocate and fill forward inputs.
    let q_buf = alloc_bf16(device, q_elems, "Q");
    let k_buf = alloc_bf16(device, kv_elems, "K");
    let v_buf = alloc_bf16(device, kv_elems, "V");
    let mut o_buf = alloc_bf16(device, q_elems, "O");
    let mut l_buf = alloc_f32(device, l_elems, "L");
    fill_bf16_buf(&q_buf, q_bf);
    fill_bf16_buf(&k_buf, k_bf);
    fill_bf16_buf(&v_buf, v_bf);

    let mask_buf = mask_bf.map(|m| {
        let mask_elems = batch * n_q_heads * ql * kl;
        assert_eq!(m.len(), mask_elems);
        let buf = alloc_bf16(device, mask_elems, "mask");
        fill_bf16_buf(&buf, m);
        buf
    });

    let do_buf = alloc_bf16(device, q_elems, "dO");
    fill_bf16_buf(&do_buf, do_bf);

    let mut dq_buf = alloc_bf16(device, q_elems, "dQ");
    let mut dk_buf = alloc_bf16(device, kv_elems, "dK");
    let mut dv_buf = alloc_bf16(device, kv_elems, "dV");

    let params = FlashAttnTrainParams {
        batch: batch as u32,
        n_q_heads: n_q_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        q_seq_len: ql as u32,
        k_seq_len: kl as u32,
        scale,
        causal,
    };

    let mut encoder = device.command_encoder().expect("command_encoder");

    // Forward pass.
    if head_dim == 64 {
        dispatch_flash_attn_train_fwd_bf16_d64(
            &mut encoder, device, registry,
            &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut o_buf, &mut l_buf,
            &params,
        ).expect("fwd d64");
    } else {
        dispatch_flash_attn_train_fwd_bf16_d256(
            &mut encoder, device, registry,
            &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut o_buf, &mut l_buf,
            &params,
        ).expect("fwd d256");
    }
    encoder.memory_barrier();

    // Backward pass.
    if head_dim == 64 {
        dispatch_flash_attn_train_bwd_bf16_d64(
            &mut encoder, device, registry,
            &q_buf, &k_buf, &v_buf, &o_buf, &l_buf, &do_buf,
            mask_buf.as_ref(), &mut dq_buf, &mut dk_buf, &mut dv_buf,
            &params,
        ).expect("bwd d64");
    } else {
        dispatch_flash_attn_train_bwd_bf16_d256(
            &mut encoder, device, registry,
            &q_buf, &k_buf, &v_buf, &o_buf, &l_buf, &do_buf,
            mask_buf.as_ref(), &mut dq_buf, &mut dk_buf, &mut dv_buf,
            &params,
        ).expect("bwd d256");
    }

    encoder.commit_and_wait().expect("commit_and_wait");

    let dq_f32 = bf16_to_f32(&read_bf16_buf(&dq_buf, q_elems));
    let dk_f32 = bf16_to_f32(&read_bf16_buf(&dk_buf, kv_elems));
    let dv_f32 = bf16_to_f32(&read_bf16_buf(&dv_buf, kv_elems));
    (dq_f32, dk_f32, dv_f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// § 10  BACKWARD — registration and compilation
// ─────────────────────────────────────────────────────────────────────────────

/// All 4 backward kernels compile on Metal (both D=64 and D=256).
#[test]
fn test_backward_kernel_names_and_library_compiles() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    flash_attn_train::register(&mut registry);
    flash_attn_train::register_bwd(&mut registry);

    for &name in flash_attn_train::all_bwd_kernel_names_for_test() {
        // The two main backward kernels need function constants; others are plain.
        let result = if name == "flash_attn_train_bwd_bf16_d64"
            || name == "flash_attn_train_bwd_bf16_d256"
        {
            registry.get_pipeline_with_bool_constants(
                name,
                device.metal_device(),
                &[(200, true), (201, true), (300, false), (301, false)],
            )
        } else {
            registry.get_pipeline(name, device.metal_device())
        };
        match result {
            Ok(pipeline) => {
                if name == "flash_attn_train_bwd_bf16_d256" {
                    assert!(
                        pipeline.static_threadgroup_memory_length() <= 24 * 1024,
                        "D256 backward must retain M1 headroom: static threadgroup memory = {} bytes",
                        pipeline.static_threadgroup_memory_length()
                    );
                }
                eprintln!("test_backward_kernel_names_and_library_compiles: {name} OK");
            }
            Err(e) => panic!("BWD pipeline compilation failed for {name}: {e:?}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 11  BACKWARD PARITY — D=64, no mask
// ─────────────────────────────────────────────────────────────────────────────

/// GPU dQ, dK, dV match CPU oracle for D=64 non-causal.
///
/// Shape: B=1, H=1, qL=32, kL=32, D=64.
/// Tolerance: atol=5e-3 (bf16 I/O, f32 atomic accumulation, non-deterministic
/// reduction order across threadgroups for dK/dV).
#[test]
fn test_backward_no_mask_d64_parity() {
    let (device, mut registry) = setup();
    flash_attn_train::register_bwd(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1; let ql = 32; let kl = 32; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q_f32 = pseudo_random_f32(SEED_VAL + 100, batch * h * ql * d);
    let k_f32 = pseudo_random_f32(SEED_VAL + 101, batch * kv_h * kl * d);
    let v_f32 = pseudo_random_f32(SEED_VAL + 102, batch * kv_h * kl * d);
    let do_f32 = pseudo_random_f32(SEED_VAL + 103, batch * h * ql * d);

    let q_bf = f32_to_bf16(&q_f32);
    let k_bf = f32_to_bf16(&k_f32);
    let v_bf = f32_to_bf16(&v_f32);
    let do_bf = f32_to_bf16(&do_f32);

    let (gpu_dq, gpu_dk, gpu_dv) = run_train_bwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None, &do_bf,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    // CPU oracle on bf16-rounded inputs.
    let q_rt = bf16_to_f32(&q_bf);
    let k_rt = bf16_to_f32(&k_bf);
    let v_rt = bf16_to_f32(&v_bf);
    let do_rt = bf16_to_f32(&do_bf);

    // Use the GPU L (from the forward pass) by running oracle_for_bf16 to get L.
    let (_, ref_l) = oracle_for_bf16(&q_bf, &k_bf, &v_bf, None, batch, h, kv_h, ql, kl, d, scale, false);

    let (ref_dq, ref_dk, ref_dv) = sdpa_backward_reference_f32(
        &q_rt, &k_rt, &v_rt, &ref_l, &do_rt, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_dq, &ref_dq, 5e-3, "bwd_d64_no_mask_dQ");
    assert_close(&gpu_dk, &ref_dk, 5e-3, "bwd_d64_no_mask_dK");
    assert_close(&gpu_dv, &ref_dv, 5e-3, "bwd_d64_no_mask_dV");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 12  BACKWARD PARITY — D=256, no mask
// ─────────────────────────────────────────────────────────────────────────────

/// GPU dQ, dK, dV match CPU oracle for D=256 non-causal.
///
/// Shape: B=1, H=4, qL=128, kL=128, D=256.
/// D=256 is the Qwen3.6-35B-A3B head dimension.
#[test]
fn test_backward_no_mask_d256_parity() {
    let (device, mut registry) = setup();
    flash_attn_train::register_bwd(&mut registry);

    let batch = 1; let h = 4; let kv_h = 4; let ql = 128; let kl = 128; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q_f32 = pseudo_random_f32(SEED_VAL + 110, batch * h * ql * d);
    let k_f32 = pseudo_random_f32(SEED_VAL + 111, batch * kv_h * kl * d);
    let v_f32 = pseudo_random_f32(SEED_VAL + 112, batch * kv_h * kl * d);
    let do_f32 = pseudo_random_f32(SEED_VAL + 113, batch * h * ql * d);

    let q_bf = f32_to_bf16(&q_f32);
    let k_bf = f32_to_bf16(&k_f32);
    let v_bf = f32_to_bf16(&v_f32);
    let do_bf = f32_to_bf16(&do_f32);

    let (gpu_dq, gpu_dk, gpu_dv) = run_train_bwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None, &do_bf,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    let q_rt = bf16_to_f32(&q_bf);
    let k_rt = bf16_to_f32(&k_bf);
    let v_rt = bf16_to_f32(&v_bf);
    let do_rt = bf16_to_f32(&do_bf);
    let (_, ref_l) = oracle_for_bf16(&q_bf, &k_bf, &v_bf, None, batch, h, kv_h, ql, kl, d, scale, false);

    let (ref_dq, ref_dk, ref_dv) = sdpa_backward_reference_f32(
        &q_rt, &k_rt, &v_rt, &ref_l, &do_rt, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_dq, &ref_dq, 5e-3, "bwd_d256_no_mask_dQ");
    assert_close(&gpu_dk, &ref_dk, 5e-3, "bwd_d256_no_mask_dK");
    assert_close(&gpu_dv, &ref_dv, 5e-3, "bwd_d256_no_mask_dV");
}

/// The M1-compatible D=256 backward tile is BQ=16. Exercise a one-row tail so
/// the host alignment constants and the shader's safe Q-tile load stay bound
/// to that geometry rather than the forward path's BQ=32.
#[test]
fn test_backward_no_mask_d256_unaligned_q_parity() {
    let (device, mut registry) = setup();
    flash_attn_train::register_bwd(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1; let ql = 17; let kl = 19; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q_bf = f32_to_bf16(&pseudo_random_f32(SEED_VAL + 114, batch * h * ql * d));
    let k_bf = f32_to_bf16(&pseudo_random_f32(SEED_VAL + 115, batch * kv_h * kl * d));
    let v_bf = f32_to_bf16(&pseudo_random_f32(SEED_VAL + 116, batch * kv_h * kl * d));
    let do_bf = f32_to_bf16(&pseudo_random_f32(SEED_VAL + 117, batch * h * ql * d));

    let (gpu_dq, gpu_dk, gpu_dv) = run_train_bwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None, &do_bf,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    let q_rt = bf16_to_f32(&q_bf);
    let k_rt = bf16_to_f32(&k_bf);
    let v_rt = bf16_to_f32(&v_bf);
    let do_rt = bf16_to_f32(&do_bf);
    let (_, ref_l) = oracle_for_bf16(
        &q_bf, &k_bf, &v_bf, None, batch, h, kv_h, ql, kl, d, scale, false,
    );
    let (ref_dq, ref_dk, ref_dv) = sdpa_backward_reference_f32(
        &q_rt, &k_rt, &v_rt, &ref_l, &do_rt, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_dq, &ref_dq, 5e-3, "bwd_d256_unaligned_q_dQ");
    assert_close(&gpu_dk, &ref_dk, 5e-3, "bwd_d256_unaligned_q_dK");
    assert_close(&gpu_dv, &ref_dv, 5e-3, "bwd_d256_unaligned_q_dV");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 13  BACKWARD — finite-difference falsifier
// ─────────────────────────────────────────────────────────────────────────────

/// Finite-difference check: GPU backward gradients match numerical gradients.
///
/// For a scalar loss L = sum(O * dO) (i.e. dO is the upstream gradient),
/// the analytical gradient w.r.t. Q[0,0,0,0] must match (L(Q+h) - L(Q-h)) / 2h.
///
/// Shape: B=1, H=1, qL=8, kL=8, D=64 (small, for fast evaluation).
/// Perturbation step h=1e-2 (larger than bf16 ulp at typical magnitudes).
/// Tolerance: atol=5e-2 (finite-diff + bf16 rounding combined).
///
/// Probes: Q[0,0,0,0], K[0,0,0,0], V[0,0,0,0].
#[test]
fn test_backward_finite_diff_falsifier() {
    let (device, mut registry) = setup();
    flash_attn_train::register_bwd(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1; let ql = 8; let kl = 8; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();
    let h_step = 1e-2_f32;

    let q_f32 = pseudo_random_f32(SEED_VAL + 200, batch * h * ql * d);
    let k_f32 = pseudo_random_f32(SEED_VAL + 201, batch * kv_h * kl * d);
    let v_f32 = pseudo_random_f32(SEED_VAL + 202, batch * kv_h * kl * d);
    // dO is the upstream gradient (acts as the "weight" for the scalar loss).
    let do_f32 = pseudo_random_f32(SEED_VAL + 203, batch * h * ql * d);

    /// Compute scalar loss L = sum(O * dO) using the CPU forward oracle.
    fn scalar_loss(
        q: &[f32], k: &[f32], v: &[f32], do_: &[f32],
        batch: usize, h: usize, kv_h: usize, ql: usize, kl: usize, d: usize, scale: f32,
    ) -> f32 {
        let (o, _) = sdpa_reference_with_logsumexp(
            q, k, v, None, batch, h, kv_h, ql, kl, d, scale, false,
        );
        o.iter().zip(do_.iter()).map(|(o, g)| o * g).sum()
    }

    // GPU backward gradients at the base point.
    let q_bf = f32_to_bf16(&q_f32);
    let k_bf = f32_to_bf16(&k_f32);
    let v_bf = f32_to_bf16(&v_f32);
    let do_bf = f32_to_bf16(&do_f32);
    let (gpu_dq, gpu_dk, gpu_dv) = run_train_bwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None, &do_bf,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    // Probe index 0 for Q.
    {
        let mut q_plus = q_f32.clone();  q_plus[0]  += h_step;
        let mut q_minus = q_f32.clone(); q_minus[0] -= h_step;
        let fd = (scalar_loss(&q_plus, &k_f32, &v_f32, &do_f32, batch, h, kv_h, ql, kl, d, scale)
                - scalar_loss(&q_minus, &k_f32, &v_f32, &do_f32, batch, h, kv_h, ql, kl, d, scale))
            / (2.0 * h_step);
        let analytical = gpu_dq[0];
        let diff = (fd - analytical).abs();
        assert!(
            diff <= 5e-2_f32,
            "finite_diff_falsifier: dQ[0] fd={fd:.5} analytical={analytical:.5} diff={diff:.5e}"
        );
        eprintln!("finite_diff_falsifier: dQ[0] fd={fd:.5} analytical={analytical:.5} diff={diff:.5e} PASS");
    }
    // Probe index 0 for K.
    {
        let mut k_plus = k_f32.clone();  k_plus[0]  += h_step;
        let mut k_minus = k_f32.clone(); k_minus[0] -= h_step;
        let fd = (scalar_loss(&q_f32, &k_plus, &v_f32, &do_f32, batch, h, kv_h, ql, kl, d, scale)
                - scalar_loss(&q_f32, &k_minus, &v_f32, &do_f32, batch, h, kv_h, ql, kl, d, scale))
            / (2.0 * h_step);
        let analytical = gpu_dk[0];
        let diff = (fd - analytical).abs();
        assert!(
            diff <= 5e-2_f32,
            "finite_diff_falsifier: dK[0] fd={fd:.5} analytical={analytical:.5} diff={diff:.5e}"
        );
        eprintln!("finite_diff_falsifier: dK[0] fd={fd:.5} analytical={analytical:.5} diff={diff:.5e} PASS");
    }
    // Probe index 0 for V.
    {
        let mut v_plus = v_f32.clone();  v_plus[0]  += h_step;
        let mut v_minus = v_f32.clone(); v_minus[0] -= h_step;
        let fd = (scalar_loss(&q_f32, &k_f32, &v_plus, &do_f32, batch, h, kv_h, ql, kl, d, scale)
                - scalar_loss(&q_f32, &k_f32, &v_minus, &do_f32, batch, h, kv_h, ql, kl, d, scale))
            / (2.0 * h_step);
        let analytical = gpu_dv[0];
        let diff = (fd - analytical).abs();
        assert!(
            diff <= 5e-2_f32,
            "finite_diff_falsifier: dV[0] fd={fd:.5} analytical={analytical:.5} diff={diff:.5e}"
        );
        eprintln!("finite_diff_falsifier: dV[0] fd={fd:.5} analytical={analytical:.5} diff={diff:.5e} PASS");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 14  BACKWARD — causal mask zeroes dK for future positions
// ─────────────────────────────────────────────────────────────────────────────

/// With causal masking, K[j] contributes to dK[j] only for Q rows i >= j.
///
/// Construct two K tensors that differ only at K rows j > q_pos_target.
/// With causal masking, those K rows should not influence dK for rows <= q_pos_target
/// relative to dQ.  More directly: dK[j > ql-1] should be ~0 when kl=ql and causal.
///
/// Simpler test: run causal backward; verify dK matches the CPU oracle.
/// The oracle implements causal via P[i,j]=0 for j>i, so dS[i,j]=0 → dK[j] gets
/// no contribution from Q rows that cannot attend to K[j].
#[test]
fn test_backward_causal_mask_parity() {
    let (device, mut registry) = setup();
    flash_attn_train::register_bwd(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1; let ql = 64; let kl = 64; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q_f32 = pseudo_random_f32(SEED_VAL + 300, batch * h * ql * d);
    let k_f32 = pseudo_random_f32(SEED_VAL + 301, batch * kv_h * kl * d);
    let v_f32 = pseudo_random_f32(SEED_VAL + 302, batch * kv_h * kl * d);
    let do_f32 = pseudo_random_f32(SEED_VAL + 303, batch * h * ql * d);

    let q_bf = f32_to_bf16(&q_f32);
    let k_bf = f32_to_bf16(&k_f32);
    let v_bf = f32_to_bf16(&v_f32);
    let do_bf = f32_to_bf16(&do_f32);

    let (gpu_dq, gpu_dk, gpu_dv) = run_train_bwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None, &do_bf,
        batch, h, kv_h, ql, kl, d, scale, true,  // causal=true
    );

    let q_rt = bf16_to_f32(&q_bf);
    let k_rt = bf16_to_f32(&k_bf);
    let v_rt = bf16_to_f32(&v_bf);
    let do_rt = bf16_to_f32(&do_bf);
    let (_, ref_l) = oracle_for_bf16(&q_bf, &k_bf, &v_bf, None, batch, h, kv_h, ql, kl, d, scale, true);

    let (ref_dq, ref_dk, ref_dv) = sdpa_backward_reference_f32(
        &q_rt, &k_rt, &v_rt, &ref_l, &do_rt, None,
        batch, h, kv_h, ql, kl, d, scale, true,
    );

    assert_close(&gpu_dq, &ref_dq, 5e-3, "causal_bwd_dQ");
    assert_close(&gpu_dk, &ref_dk, 5e-3, "causal_bwd_dK");
    assert_close(&gpu_dv, &ref_dv, 5e-3, "causal_bwd_dV");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 15  BACKWARD — sliding-window mask parity
// ─────────────────────────────────────────────────────────────────────────────

/// GPU dQ/dK/dV match CPU oracle with an additive SWA mask.
///
/// Shape: B=1, H=1, qL=32, kL=32, D=64, window=8.
#[test]
fn test_backward_sliding_window_mask_parity() {
    let (device, mut registry) = setup();
    flash_attn_train::register_bwd(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1; let ql = 32; let kl = 32; let d = 64;
    let window = 8usize;
    let scale = 1.0 / (d as f32).sqrt();

    let q_f32 = pseudo_random_f32(SEED_VAL + 400, batch * h * ql * d);
    let k_f32 = pseudo_random_f32(SEED_VAL + 401, batch * kv_h * kl * d);
    let v_f32 = pseudo_random_f32(SEED_VAL + 402, batch * kv_h * kl * d);
    let do_f32 = pseudo_random_f32(SEED_VAL + 403, batch * h * ql * d);

    // SWA mask: 0.0 if j <= i && j+window >= i, else -inf.
    let mut mask_f32 = vec![f32::NEG_INFINITY; batch * h * ql * kl];
    for b in 0..batch { for hh in 0..h { for i in 0..ql { for j in 0..kl {
        let in_window = (j <= i) && (j + window >= i);
        let idx = b * h * ql * kl + hh * ql * kl + i * kl + j;
        if in_window { mask_f32[idx] = 0.0_f32; }
    } } } }
    let mask_bf = f32_to_bf16(&mask_f32);

    let q_bf = f32_to_bf16(&q_f32);
    let k_bf = f32_to_bf16(&k_f32);
    let v_bf = f32_to_bf16(&v_f32);
    let do_bf = f32_to_bf16(&do_f32);

    let (gpu_dq, gpu_dk, gpu_dv) = run_train_bwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, Some(&mask_bf), &do_bf,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    let q_rt = bf16_to_f32(&q_bf);
    let k_rt = bf16_to_f32(&k_bf);
    let v_rt = bf16_to_f32(&v_bf);
    let do_rt = bf16_to_f32(&do_bf);
    let mask_rt = bf16_to_f32(&mask_bf);
    let (_, ref_l) = oracle_for_bf16(&q_bf, &k_bf, &v_bf, Some(&mask_bf), batch, h, kv_h, ql, kl, d, scale, false);

    let (ref_dq, ref_dk, ref_dv) = sdpa_backward_reference_f32(
        &q_rt, &k_rt, &v_rt, &ref_l, &do_rt, Some(&mask_rt),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close(&gpu_dq, &ref_dq, 5e-3, "swa_bwd_dQ");
    assert_close(&gpu_dk, &ref_dk, 5e-3, "swa_bwd_dK");
    assert_close(&gpu_dv, &ref_dv, 5e-3, "swa_bwd_dV");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 16  BACKWARD — GQA accumulation
// ─────────────────────────────────────────────────────────────────────────────

/// GPU dK/dV accumulate correctly across all Q-heads that share a KV head.
///
/// Shape: B=1, H_q=8, H_kv=2, gqa_factor=4, qL=32, kL=32, D=64.
/// With gqa_factor=4: Q heads {0,1,2,3} share KV head 0; {4,5,6,7} share KV head 1.
/// dK[0] must accumulate contributions from all 4 Q heads.
#[test]
fn test_backward_gqa_accumulation() {
    let (device, mut registry) = setup();
    flash_attn_train::register_bwd(&mut registry);

    let batch = 1; let h = 8; let kv_h = 2; let ql = 32; let kl = 32; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q_f32 = pseudo_random_f32(SEED_VAL + 500, batch * h * ql * d);
    let k_f32 = pseudo_random_f32(SEED_VAL + 501, batch * kv_h * kl * d);
    let v_f32 = pseudo_random_f32(SEED_VAL + 502, batch * kv_h * kl * d);
    let do_f32 = pseudo_random_f32(SEED_VAL + 503, batch * h * ql * d);

    let q_bf = f32_to_bf16(&q_f32);
    let k_bf = f32_to_bf16(&k_f32);
    let v_bf = f32_to_bf16(&v_f32);
    let do_bf = f32_to_bf16(&do_f32);

    let (gpu_dq, gpu_dk, gpu_dv) = run_train_bwd(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None, &do_bf,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    let q_rt = bf16_to_f32(&q_bf);
    let k_rt = bf16_to_f32(&k_bf);
    let v_rt = bf16_to_f32(&v_bf);
    let do_rt = bf16_to_f32(&do_bf);
    let (_, ref_l) = oracle_for_bf16(&q_bf, &k_bf, &v_bf, None, batch, h, kv_h, ql, kl, d, scale, false);

    let (ref_dq, ref_dk, ref_dv) = sdpa_backward_reference_f32(
        &q_rt, &k_rt, &v_rt, &ref_l, &do_rt, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    // GQA: dK has shape [B, H_kv, kL, D]; dQ has shape [B, H_q, qL, D].
    assert_close(&gpu_dq, &ref_dq, 5e-3, "gqa_bwd_dQ");
    assert_close(&gpu_dk, &ref_dk, 5e-3, "gqa_bwd_dK");
    assert_close(&gpu_dv, &ref_dv, 5e-3, "gqa_bwd_dV");
}

} // mod flash_attn_train_tests
