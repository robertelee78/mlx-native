//! ADR-028 iter-309 — parity test for `kernel_mul_mv_q6_K_f32_nr2`.
//!
//! The new kernel ports llama.cpp's `kernel_mul_mv_q6_K_f32_impl` with
//! `N_R0_Q6_K=2` (peer pattern) — 4 rows/TG (2 SGs × 2 rows) with
//! cached `yl[16]` shared across both rows.  The original
//! `kernel_mul_mv_q6_K_f32` processes 2 rows/TG (1 row/SG) with no Y
//! cache.
//!
//! Both kernels compute the same math (Q6_K dequant × F32 input,
//! simdgroup sum reduction).  We assert byte-equivalence within tight
//! f32 accumulation tolerance (1e-4 absolute; reduction ordering is
//! identical so the only delta is FMA-vs-mul-add fusion that the Metal
//! compiler may apply).
//!
//! Falsifier for ADR-028 iter-309 H1: this test must PASS for the
//! perf claim to be load-bearing.  If parity fails, the kernel is
//! buggy and any speed win is meaningless.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry, MlxDevice};

// PRNG matching test_quantized_matmul_mm.rs
fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

// Q6_K pack identical to test_quantized_matmul_mm.rs — duplicated to
// keep this test self-contained.
fn pack_q6_k(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 256 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(256) {
        let mut sub_scales = [0.0f32; 16];
        let mut sub_scale_int = [0i8; 16];
        let mut max_scale: f32 = 0.0;

        for (s, sub) in block.chunks(16).enumerate() {
            let amax = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            sub_scales[s] = amax;
            if amax > max_scale {
                max_scale = amax;
            }
        }

        let d = max_scale / (32.0 * 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for s in 0..16 {
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
        }

        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
                q6[s * 16 + i] = q;
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = l0_base + l;
                let v0 = q6[l0_base + l];
                let v2 = q6[l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                let v1 = q6[l0_base + l + 32];
                let v3 = q6[l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);

                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[ql_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }

        for l0_base in (0..32usize).step_by(4) {
            for l in 0..4usize {
                let ql_idx = 64 + l0_base + l;
                let qh_idx = 32 + l0_base + l;
                let v0 = q6[128 + l0_base + l];
                let v2 = q6[128 + l0_base + l + 64];
                ql[ql_idx] = (v0 & 0x0F) | ((v2 & 0x0F) << 4);

                let v1 = q6[128 + l0_base + l + 32];
                let v3 = q6[128 + l0_base + l + 96];
                ql[ql_idx + 32] = (v1 & 0x0F) | ((v3 & 0x0F) << 4);

                let h0 = (v0 >> 4) & 0x03;
                let h1 = (v1 >> 4) & 0x03;
                let h2 = (v2 >> 4) & 0x03;
                let h3 = (v3 >> 4) & 0x03;
                qh[qh_idx] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
            }
        }

        buf.extend_from_slice(&ql);
        buf.extend_from_slice(&qh);
        buf.extend_from_slice(
            &sub_scale_int.iter().map(|&s| s as u8).collect::<Vec<_>>(),
        );
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
    }
    buf
}

fn run_mv(use_nr2: bool, m: usize, n: usize, k: usize, weight_bytes: &[u8], input: &[f32]) -> Vec<f32> {
    // SAFETY: env vars are process-global; this test is single-threaded.
    if use_nr2 {
        std::env::set_var("HF2Q_Q6K_MV_NR2", "1");
    } else {
        std::env::remove_var("HF2Q_Q6K_MV_NR2");
    }

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let input_bytes = m * k * 4;
    let mut input_buf = device
        .alloc_buffer(input_bytes, DType::F32, vec![m, k])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input mut slice")
        .copy_from_slice(input);

    let mut weight_buf = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight mut slice")
        .copy_from_slice(weight_bytes);

    let output_bytes = m * n * 4;
    let mut output_buf = device
        .alloc_buffer(output_bytes, DType::F32, vec![m, n])
        .expect("alloc output");
    for v in output_buf
        .as_mut_slice::<f32>()
        .expect("output mut slice")
        .iter_mut()
    {
        *v = 0.0;
    }

    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q6_K,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::quantized_matmul_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &mut output_buf,
        &params,
    )
    .expect("mv dispatch");
    encoder.commit_and_wait().expect("GPU execution");

    let out = output_buf
        .as_slice::<f32>()
        .expect("read output")
        .to_vec();

    // Always clear env after the call so we don't leak state into other tests.
    std::env::remove_var("HF2Q_Q6K_MV_NR2");
    out
}

fn check_parity(label: &str, m: usize, n: usize, k: usize, tol: f32) {
    let weight_f32 = pseudo_random_f32(0xc0ffee, n * k);
    let weight_bytes = pack_q6_k(&weight_f32);
    let input = pseudo_random_f32(0xbeef, m * k);

    let baseline = run_mv(false, m, n, k, &weight_bytes, &input);
    let nr2 = run_mv(true, m, n, k, &weight_bytes, &input);

    assert_eq!(baseline.len(), nr2.len(), "{label}: output len mismatch");

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (&b, &n2)) in baseline.iter().zip(nr2.iter()).enumerate() {
        let diff = (b - n2).abs();
        let rel = if b.abs() > 1e-9 { diff / b.abs() } else { 0.0 };
        if diff > max_abs {
            max_abs = diff;
        }
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            diff <= tol,
            "{label}: row[{i}] baseline={b} nr2={n2} diff={diff} > tol={tol}"
        );
    }
    eprintln!(
        "[parity-q6k-nr2] {label} M={m} N={n} K={k}: max_abs={:.3e} max_rel={:.3e}",
        max_abs, max_rel
    );
}

#[test]
fn q6k_mv_nr2_parity_n4_k256() {
    // smallest case: N=4 (one full 4-row TG), K=256 (one block per row).
    check_parity("N=4 K=256", 1, 4, 256, 1e-4);
}

#[test]
fn q6k_mv_nr2_parity_n8_k512() {
    // 2 TGs in N, 2 blocks/row.
    check_parity("N=8 K=512", 1, 8, 512, 1e-4);
}

#[test]
fn q6k_mv_nr2_parity_gemma4_lm_head_shape() {
    // gemma4 LM-head: token_embd is N=262144 K=2816 Q6_K (per ADR-028
    // iter-187/188).  We can't test the full shape on every run, but
    // N=256 K=2816 exercises the per-row dequant + multi-block sum
    // path identically.  Use 256 to ensure exact div-by-4 grid.
    check_parity("gemma4-lmhead-mini N=256 K=2816", 1, 256, 2816, 5e-4);
}

#[test]
fn q6k_mv_nr2_parity_n_odd_boundary() {
    // N=6 is NOT divisible by 4 — exercises the boundary check
    // `if (out_row < p.ne01)` inside the kernel write-out.
    // Note: align=4 → 2 TGs cover rows 0-3 and 4-7; rows 6-7 don't
    // exist and the kernel must skip the write for those.
    check_parity("boundary N=6 K=256", 1, 6, 256, 1e-4);
}
