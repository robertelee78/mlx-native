//! ADR-028 iter-331 — parity test for `fused_norm_add_f32_v2`.
//!
//! Tests the peer-pattern variant (float4 vector loads + simd_sum
//! reduction) against the baseline scalar + tree-reduction kernel.
//! Both must produce identical output to within f32 accumulation
//! tolerance.  Reduction ordering differs (simdgroup partial sums vs
//! threadgroup tree), so we allow a slightly larger tolerance than
//! strict bit-equality (1e-4 abs, 1e-4 rel) — same tolerance as
//! iter-310's rms_norm_v2 parity test.
//!
//! Falsifier for ADR-028 iter-331 default-flip: this test MUST pass
//! for the perf claim to be load-bearing.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, KernelRegistry, MlxDevice};

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

fn run_fused_norm_add_f32(
    use_v2: bool,
    rows: u32,
    dim: u32,
    residual: &[f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    if use_v2 {
        std::env::set_var("HF2Q_FUSED_NORM_ADD_V2", "1");
    } else {
        std::env::set_var("HF2Q_FUSED_NORM_ADD_V2", "0");
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::fused_norm_add::register(&mut registry);

    let n = (rows as usize) * (dim as usize);
    let byte_len = n * 4;
    let weight_byte_len = (dim as usize) * 4;

    let mut residual_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc residual");
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc input");
    let mut weight_buf = device
        .alloc_buffer(weight_byte_len, DType::F32, vec![dim as usize])
        .expect("alloc weight");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc output");

    residual_buf
        .as_mut_slice::<f32>()
        .expect("residual")
        .copy_from_slice(residual);
    input_buf
        .as_mut_slice::<f32>()
        .expect("input")
        .copy_from_slice(input);
    weight_buf
        .as_mut_slice::<f32>()
        .expect("weight")
        .copy_from_slice(weight);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &residual_buf,
        &input_buf,
        &weight_buf,
        &output_buf,
        dim,
        rows,
        eps,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out = output_buf.as_slice::<f32>().expect("read").to_vec();
    std::env::remove_var("HF2Q_FUSED_NORM_ADD_V2");
    out
}

fn check_parity(label: &str, rows: u32, dim: u32, tol_abs: f32, tol_rel: f32) {
    let n = (rows as usize) * (dim as usize);
    let residual = pseudo_random_f32(0xfeedface, n);
    let input = pseudo_random_f32(0xc0ffee, n);
    let weight = pseudo_random_f32(0xbeef, dim as usize);
    let eps = 1e-6_f32;

    let out_baseline = run_fused_norm_add_f32(false, rows, dim, &residual, &input, &weight, eps);
    let out_v2 = run_fused_norm_add_f32(true, rows, dim, &residual, &input, &weight, eps);

    assert_eq!(
        out_baseline.len(),
        out_v2.len(),
        "[{}] output length mismatch",
        label
    );

    let mut max_abs_diff = 0.0_f32;
    let mut max_rel_diff = 0.0_f32;
    for (i, (&a, &b)) in out_baseline.iter().zip(out_v2.iter()).enumerate() {
        let abs_diff = (a - b).abs();
        let rel_diff = if a.abs() > 1e-8 {
            abs_diff / a.abs()
        } else {
            abs_diff
        };
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }
        assert!(
            abs_diff <= tol_abs || rel_diff <= tol_rel,
            "[{}] output diff at index {}: baseline={} v2={} abs_diff={} rel_diff={} (tol_abs={} tol_rel={})",
            label,
            i,
            a,
            b,
            abs_diff,
            rel_diff,
            tol_abs,
            tol_rel
        );
    }

    eprintln!(
        "[{}] PASS rows={} dim={} max_abs_diff={:.2e} max_rel_diff={:.2e}",
        label, rows, dim, max_abs_diff, max_rel_diff
    );
}

#[test]
fn fused_norm_add_v2_parity_gemma4_hidden() {
    // gemma4 hidden=2816
    check_parity("gemma4-hidden-1row", 1, 2816, 1e-4, 1e-4);
    check_parity("gemma4-hidden-4rows", 4, 2816, 1e-4, 1e-4);
}

#[test]
fn fused_norm_add_v2_parity_qwen35_hidden() {
    // qwen3.6 hidden=2048
    check_parity("qwen35-hidden-1row", 1, 2048, 1e-4, 1e-4);
    check_parity("qwen35-hidden-8rows", 8, 2048, 1e-4, 1e-4);
}

#[test]
fn fused_norm_add_v2_parity_head_dim() {
    // head_dim=256 — smaller threadgroup case
    check_parity("head-dim-1row", 1, 256, 1e-4, 1e-4);
    check_parity("head-dim-16rows", 16, 256, 1e-4, 1e-4);
}

#[test]
fn fused_norm_add_v2_parity_tiny() {
    // tiny dim — boundary case for n_sg=1 path
    check_parity("tiny-1row", 1, 64, 1e-4, 1e-4);
    check_parity("tiny-2rows", 2, 64, 1e-4, 1e-4);
}
