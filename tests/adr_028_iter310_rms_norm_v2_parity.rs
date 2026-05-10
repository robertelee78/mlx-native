//! ADR-028 iter-310 — parity test for `rms_norm_f32_v2` /
//! `rms_norm_no_scale_f32_v2`.
//!
//! Tests the peer-pattern variants (float4 vector loads + simd_sum
//! reduction) against the baseline scalar + tree-reduction kernels.
//! Both must produce identical output to within f32 accumulation
//! tolerance.  Reduction ordering differs (simdgroup partial sums vs
//! threadgroup tree), so we allow a slightly larger tolerance than
//! strict bit-equality (1e-4 abs, 1e-4 rel).
//!
//! Falsifier for ADR-028 iter-310 H2: this test MUST pass for the
//! perf claim to be load-bearing.

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

fn run_rms_norm_f32(use_v2: bool, rows: u32, dim: u32, input: &[f32], weight: &[f32]) -> Vec<f32> {
    if use_v2 {
        std::env::set_var("HF2Q_RMS_NORM_V2", "1");
    } else {
        std::env::remove_var("HF2Q_RMS_NORM_V2");
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::rms_norm::register(&mut registry);

    let n = (rows as usize) * (dim as usize);
    let byte_len = n * 4;
    let weight_byte_len = (dim as usize) * 4;

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc input");
    let mut weight_buf = device
        .alloc_buffer(weight_byte_len, DType::F32, vec![dim as usize])
        .expect("alloc weight");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .expect("alloc params");

    input_buf.as_mut_slice::<f32>().expect("input").copy_from_slice(input);
    weight_buf.as_mut_slice::<f32>().expect("weight").copy_from_slice(weight);
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("params");
        s[0] = 1e-6;
        s[1] = dim as f32;
    }

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rms_norm::dispatch_rms_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &weight_buf,
        &output_buf,
        &params_buf,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out = output_buf.as_slice::<f32>().expect("read").to_vec();
    std::env::remove_var("HF2Q_RMS_NORM_V2");
    out
}

fn run_rms_norm_no_scale_f32(use_v2: bool, rows: u32, dim: u32, input: &[f32]) -> Vec<f32> {
    if use_v2 {
        std::env::set_var("HF2Q_RMS_NORM_V2", "1");
    } else {
        std::env::remove_var("HF2Q_RMS_NORM_V2");
    }

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::rms_norm::register(&mut registry);

    let n = (rows as usize) * (dim as usize);
    let byte_len = n * 4;

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .expect("alloc params");

    input_buf.as_mut_slice::<f32>().expect("input").copy_from_slice(input);
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("params");
        s[0] = 1e-6;
        s[1] = dim as f32;
    }

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rms_norm::dispatch_rms_norm_no_scale_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        rows,
        dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out = output_buf.as_slice::<f32>().expect("read").to_vec();
    std::env::remove_var("HF2Q_RMS_NORM_V2");
    out
}

fn check_parity_with_weight(label: &str, rows: u32, dim: u32, tol_abs: f32) {
    let n = (rows as usize) * (dim as usize);
    let input = pseudo_random_f32(0xc0ffee, n);
    let weight = pseudo_random_f32(0xbeef, dim as usize);

    let baseline = run_rms_norm_f32(false, rows, dim, &input, &weight);
    let v2 = run_rms_norm_f32(true, rows, dim, &input, &weight);

    assert_eq!(baseline.len(), v2.len());
    let mut max_abs = 0f32;
    for (i, (&b, &v)) in baseline.iter().zip(v2.iter()).enumerate() {
        let diff = (b - v).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        assert!(
            diff <= tol_abs,
            "{label}: idx[{i}] baseline={b} v2={v} diff={diff} > tol={tol_abs}"
        );
    }
    eprintln!(
        "[parity-rms-v2] {label} rows={rows} dim={dim}: max_abs={:.3e}",
        max_abs
    );
}

fn check_parity_no_scale(label: &str, rows: u32, dim: u32, tol_abs: f32) {
    let n = (rows as usize) * (dim as usize);
    let input = pseudo_random_f32(0xc0ffee, n);

    let baseline = run_rms_norm_no_scale_f32(false, rows, dim, &input);
    let v2 = run_rms_norm_no_scale_f32(true, rows, dim, &input);

    assert_eq!(baseline.len(), v2.len());
    let mut max_abs = 0f32;
    for (i, (&b, &v)) in baseline.iter().zip(v2.iter()).enumerate() {
        let diff = (b - v).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        assert!(
            diff <= tol_abs,
            "{label}: idx[{i}] baseline={b} v2={v} diff={diff} > tol={tol_abs}"
        );
    }
    eprintln!(
        "[parity-rms-v2-noscale] {label} rows={rows} dim={dim}: max_abs={:.3e}",
        max_abs
    );
}

#[test]
fn rms_norm_v2_parity_tiny() {
    check_parity_with_weight("dim=8 (smallest float4 case, 2 lanes)", 4, 8, 1e-5);
}

#[test]
fn rms_norm_v2_parity_gemma4_hidden() {
    // gemma4 hidden_size = 3584.  This is the production hot path.
    check_parity_with_weight("gemma4 dim=3584", 1, 3584, 5e-5);
}

#[test]
fn rms_norm_v2_parity_qwen35_hidden() {
    // qwen3.6 35B-A3B-APEX hidden_size = 2048.
    check_parity_with_weight("qwen3.6 dim=2048", 1, 2048, 5e-5);
}

#[test]
fn rms_norm_v2_parity_head_dim() {
    // Q/K/V per-head norm in gemma4: head_dim=256.
    check_parity_with_weight("head dim=256", 30, 256, 1e-5);
}

#[test]
fn rms_norm_v2_no_scale_parity_gemma4() {
    check_parity_no_scale("no_scale gemma4 dim=3584", 1, 3584, 5e-5);
}

#[test]
fn rms_norm_v2_no_scale_parity_head_dim() {
    // gemma4 V-rmsnorm (per-head, no scale weight): head_dim=256.
    check_parity_no_scale("no_scale head dim=256", 30, 256, 1e-5);
}
