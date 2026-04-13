//! Tests for the RMS Normalization without learned scale (f32) GPU kernel.
//!
//! Verifies that the `rms_norm_no_scale_f32` kernel correctly normalizes
//! each row (head) independently without weight multiplication — used for
//! per-head V normalization in Gemma 4.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// CPU reference: x * rsqrt(mean(x^2) + eps) with no weight.
fn rms_norm_no_scale_ref(input: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = input.len() / dim;
    let mut output = vec![0.0f32; input.len()];
    for r in 0..rows {
        let base = r * dim;
        let row = &input[base..base + dim];
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / dim as f32;
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..dim {
            output[base + i] = row[i] * rms_inv;
        }
    }
    output
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::rms_norm::register(&mut registry);
    (device, registry)
}

/// 4 heads x 64 dims: verify each head independently normalized, no weight.
#[test]
fn test_rms_norm_no_scale_f32_per_head() {
    let (device, mut registry) = setup();
    let eps = 1e-6_f32;
    let rows: u32 = 4;   // num_heads
    let dim: u32 = 64;   // head_dim
    let n = (rows as usize) * (dim as usize);

    // Deterministic input: each head has distinct values
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 6.4).collect();

    let byte_len = n * std::mem::size_of::<f32>();
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc output");

    let params_byte_len = 2 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("write input");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("write params");
        s[0] = eps;
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

    let expected = rms_norm_no_scale_ref(&input_data, dim as usize, eps);
    let output: &[f32] = output_buf.as_slice().expect("read output");

    let mut max_diff = 0.0f32;
    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("test_rms_norm_no_scale_f32_per_head: max|delta| = {max_diff}");
    assert!(
        max_diff <= 1e-5,
        "Max diff {} exceeds 1e-5 tolerance",
        max_diff
    );

    // Verify each head is independently normalized: the L2 norm of each
    // head's output should be approximately sqrt(dim).
    for h in 0..rows as usize {
        let base = h * dim as usize;
        let head = &output[base..base + dim as usize];
        let sq_sum: f32 = head.iter().map(|&x| x * x).sum();
        let l2 = sq_sum.sqrt();
        let expected_l2 = (dim as f32).sqrt();
        let rel_err = ((l2 - expected_l2) / expected_l2).abs();
        assert!(
            rel_err < 0.01,
            "Head {} L2 norm {:.4} != expected {:.4} (rel_err={:.6})",
            h,
            l2,
            expected_l2,
            rel_err
        );
    }
}

/// Single row, small dim: basic correctness.
#[test]
fn test_rms_norm_no_scale_f32_single_row() {
    let (device, mut registry) = setup();
    let eps = 1e-6_f32;
    let rows: u32 = 1;
    let dim: u32 = 4;
    let n = (rows as usize) * (dim as usize);

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let byte_len = n * std::mem::size_of::<f32>();
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1, 4])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1, 4])
        .expect("alloc output");

    let params_byte_len = 2 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("write");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("write");
        s[0] = eps;
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

    let expected = rms_norm_no_scale_ref(&input_data, dim as usize, eps);
    let output: &[f32] = output_buf.as_slice().expect("read output");

    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "Index {}: expected={}, got={}, diff={}",
            i,
            expected[i],
            output[i],
            diff
        );
    }
}

/// Validation: rows=0 should error.
#[test]
fn test_rms_norm_no_scale_f32_zero_rows_error() {
    let (device, mut registry) = setup();

    let buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("buf");
    let params = device
        .alloc_buffer(8, DType::F32, vec![2])
        .expect("params");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = mlx_native::ops::rms_norm::dispatch_rms_norm_no_scale_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        &params,
        0, // rows = 0
        1,
    );
    assert!(result.is_err(), "Should error on rows=0");
}
