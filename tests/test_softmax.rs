//! Tests for the numerically stable softmax GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// Reference softmax implementation in pure Rust.
fn softmax_ref(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];

    for r in 0..rows {
        let base = r * cols;
        let row = &input[base..base + cols];

        // Find max
        let row_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max)
        let exps: Vec<f32> = row.iter().map(|&x| (x - row_max).exp()).collect();

        // Sum
        let sum: f32 = exps.iter().sum();

        // Normalize
        for i in 0..cols {
            output[base + i] = exps[i] / sum;
        }
    }
    output
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::softmax::register(&mut registry);
    (device, registry)
}

#[test]
fn test_softmax_f32_basic() {
    let (device, mut registry) = setup();
    let rows: u32 = 4;
    let cols: u32 = 8;
    let n = (rows as usize) * (cols as usize);

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 4.8).collect();

    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc output");

    // Params: [cols, 0]
    let params_byte_len = 2 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = cols as f32;
        s[1] = 0.0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::softmax::dispatch_softmax(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        rows,
        cols,
    )
    .expect("dispatch_softmax");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = softmax_ref(&input_data, rows as usize, cols as usize);
    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "Softmax f32 mismatch at index {}: expected={}, got={}, diff={}",
            i, expected[i], output[i], diff
        );
    }
}

#[test]
fn test_softmax_f32_sums_to_one() {
    let (device, mut registry) = setup();
    let rows: u32 = 3;
    let cols: u32 = 16;
    let n = (rows as usize) * (cols as usize);

    let input_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).sin() * 5.0).collect();

    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = cols as f32;
        s[1] = 0.0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::softmax::dispatch_softmax(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        rows,
        cols,
    )
    .expect("dispatch_softmax");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    // Check each row sums to 1.0
    for r in 0..rows as usize {
        let base = r * (cols as usize);
        let row_sum: f32 = output[base..base + cols as usize].iter().sum();
        let diff = (row_sum - 1.0).abs();
        assert!(
            diff <= 1e-5,
            "Softmax row {} sum should be 1.0, got {}",
            r, row_sum
        );
    }

    // Check all values are in [0, 1]
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Softmax output at {} should be in [0,1], got {}",
            i, val
        );
    }
}

#[test]
fn test_softmax_f32_large_magnitudes() {
    // Test numerical stability with very large values.
    let (device, mut registry) = setup();
    let rows: u32 = 2;
    let cols: u32 = 4;
    let n = (rows as usize) * (cols as usize);

    // Include very large values that would overflow exp() without max subtraction.
    let input_data: Vec<f32> = vec![
        1000.0, 999.0, 998.0, 997.0, // large positive
        -1000.0, -999.0, -998.0, -997.0, // large negative
    ];

    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![rows as usize, cols as usize])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = cols as f32;
        s[1] = 0.0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::softmax::dispatch_softmax(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        rows,
        cols,
    )
    .expect("dispatch_softmax");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    // No NaN or Inf
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Softmax output at {} is not finite: {}",
            i, val
        );
    }

    // Rows should sum to 1.0
    for r in 0..rows as usize {
        let base = r * (cols as usize);
        let row_sum: f32 = output[base..base + cols as usize].iter().sum();
        let diff = (row_sum - 1.0).abs();
        assert!(
            diff <= 1e-5,
            "Softmax row {} sum should be 1.0 with large magnitudes, got {}",
            r, row_sum
        );
    }

    // For the first row [1000, 999, 998, 997], the max element should get ~0.64
    let expected = softmax_ref(&input_data, rows as usize, cols as usize);
    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "Softmax large magnitude mismatch at {}: expected={}, got={}",
            i, expected[i], output[i]
        );
    }
}

#[test]
fn test_softmax_f32_uniform() {
    // All same values -> uniform distribution (1/cols each)
    let (device, mut registry) = setup();
    let rows: u32 = 1;
    let cols: u32 = 8;
    let n = cols as usize;

    let input_data: Vec<f32> = vec![5.0; n];

    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1, n])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1, n])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .expect("alloc params");

    {
        let s: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        s.copy_from_slice(&input_data);
    }
    {
        let s: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        s[0] = cols as f32;
        s[1] = 0.0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::softmax::dispatch_softmax(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        rows,
        cols,
    )
    .expect("dispatch_softmax");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    let expected_val = 1.0 / (cols as f32);

    for (i, &val) in output.iter().enumerate() {
        let diff = (val - expected_val).abs();
        assert!(
            diff <= 1e-5,
            "Softmax uniform at {}: expected={}, got={}",
            i, expected_val, val
        );
    }
}
