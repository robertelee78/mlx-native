//! Tests for the RMS Normalization GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// Reference RMS norm implementation in pure Rust.
fn rms_norm_ref(input: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = input.len() / dim;
    let mut output = vec![0.0f32; input.len()];

    for r in 0..rows {
        let base = r * dim;
        let row = &input[base..base + dim];

        // mean(x^2)
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / dim as f32;
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();

        for i in 0..dim {
            output[base + i] = row[i] * rms_inv * weight[i];
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

#[test]
fn test_rms_norm_f32_basic() {
    let (device, mut registry) = setup();
    let eps = 1e-6_f32;
    let rows: u32 = 4;
    let dim: u32 = 8;
    let n = (rows as usize) * (dim as usize);

    // Deterministic input
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.6).collect();
    let weight_data: Vec<f32> = (0..dim as usize).map(|i| 1.0 + (i as f32) * 0.05).collect();

    let byte_len = n * std::mem::size_of::<f32>();
    let weight_byte_len = (dim as usize) * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc input");
    let mut weight_buf = device
        .alloc_buffer(weight_byte_len, DType::F32, vec![dim as usize])
        .expect("alloc weight");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc output");

    // Params: [eps, dim]
    let params_byte_len = 2 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = weight_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&weight_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = eps;
        s[1] = dim as f32;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
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
    .expect("dispatch_rms_norm");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rms_norm_ref(&input_data, &weight_data, dim as usize, eps);
    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "RMS norm f32 mismatch at index {}: expected={}, got={}, diff={}",
            i, expected[i], output[i], diff
        );
    }
}

#[test]
fn test_rms_norm_f32_unit_weight() {
    // With all weights = 1.0, output = x * rsqrt(mean(x^2) + eps)
    let (device, mut registry) = setup();
    let eps = 1e-6_f32;
    let rows: u32 = 2;
    let dim: u32 = 4;
    let n = (rows as usize) * (dim as usize);

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let weight_data: Vec<f32> = vec![1.0; dim as usize];

    let byte_len = n * std::mem::size_of::<f32>();
    let weight_byte_len = (dim as usize) * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc input");
    let mut weight_buf = device
        .alloc_buffer(weight_byte_len, DType::F32, vec![dim as usize])
        .expect("alloc weight");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, dim as usize])
        .expect("alloc output");

    let params_byte_len = 2 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = weight_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&weight_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = eps;
        s[1] = dim as f32;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
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
    .expect("dispatch_rms_norm");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rms_norm_ref(&input_data, &weight_data, dim as usize, eps);
    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "RMS norm f32 unit weight mismatch at index {}: expected={}, got={}, diff={}",
            i, expected[i], output[i], diff
        );
    }

    // With unit weights and RMS norm, the output should have RMS = 1
    // (approximately, modulo eps)
    for r in 0..rows as usize {
        let base = r * (dim as usize);
        let row = &output[base..base + dim as usize];
        let rms: f32 = (row.iter().map(|&x| x * x).sum::<f32>() / dim as f32).sqrt();
        let diff = (rms - 1.0).abs();
        assert!(
            diff <= 1e-3,
            "RMS of normalized row {} should be ~1.0, got {}",
            r, rms
        );
    }
}

#[test]
fn test_rms_norm_f32_single_element() {
    let (device, mut registry) = setup();
    let eps = 1e-6_f32;

    // Single element: x * rsqrt(x^2 + eps) * w = sign(x) * w (approximately)
    let input_data: Vec<f32> = vec![3.0];
    let weight_data: Vec<f32> = vec![2.0];

    let mut input_buf = device
        .alloc_buffer(4, DType::F32, vec![1, 1])
        .expect("alloc input");
    let mut weight_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc weight");
    let output_buf = device
        .alloc_buffer(4, DType::F32, vec![1, 1])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = input_data[0];
    }
    {
        let s: &mut [f32] = weight_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = weight_data[0];
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = eps;
        s[1] = 1.0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::rms_norm::dispatch_rms_norm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &weight_buf,
        &output_buf,
        &params_buf,
        1,
        1,
    )
    .expect("dispatch_rms_norm");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rms_norm_ref(&input_data, &weight_data, 1, eps);
    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    let diff = (output[0] - expected[0]).abs();
    assert!(
        diff <= 1e-5,
        "RMS norm single element: expected={}, got={}",
        expected[0], output[0]
    );
}
