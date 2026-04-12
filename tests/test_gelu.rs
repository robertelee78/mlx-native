//! Tests for the GELU (pytorch_tanh variant) GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// Reference GELU pytorch_tanh implementation in pure Rust.
fn gelu_ref(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let coeff: f32 = 0.044715;
    let x_cubed = x * x * x;
    let inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    0.5 * x * (1.0 + inner.tanh())
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::gelu::register(&mut registry);
    (device, registry)
}

#[test]
fn test_gelu_f32_basic() {
    let (device, mut registry) = setup();

    let input_data: Vec<f32> = vec![
        -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, -4.0, 0.1,
    ];
    let n = input_data.len();
    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc output");

    // Write input data
    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }

    // Dispatch
    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::gelu::dispatch_gelu(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
    )
    .expect("dispatch_gelu");
    encoder.commit_and_wait().expect("commit_and_wait");

    // Verify
    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    for (i, &x) in input_data.iter().enumerate() {
        let expected = gelu_ref(x);
        let actual = output[i];
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1e-5,
            "GELU f32 mismatch at index {}: input={}, expected={}, got={}, diff={}",
            i, x, expected, actual, diff
        );
    }
}

#[test]
fn test_gelu_f32_zero() {
    let (device, mut registry) = setup();

    let byte_len = std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1])
        .expect("alloc output");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = 0.0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::gelu::dispatch_gelu(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
    )
    .expect("dispatch_gelu");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    assert!(
        output[0].abs() <= 1e-7,
        "GELU(0) should be 0, got {}",
        output[0]
    );
}

#[test]
fn test_gelu_f32_large_positive() {
    let (device, mut registry) = setup();

    let input_data: Vec<f32> = vec![10.0, 20.0, 50.0, 100.0];
    let n = input_data.len();
    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc output");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::gelu::dispatch_gelu(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
    )
    .expect("dispatch_gelu");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    // For large positive x, GELU(x) ~ x
    for (i, &x) in input_data.iter().enumerate() {
        let expected = gelu_ref(x);
        let diff = (output[i] - expected).abs();
        assert!(
            diff <= 1e-5,
            "GELU f32 large positive mismatch at {}: expected={}, got={}",
            i, expected, output[i]
        );
    }
}

#[test]
fn test_gelu_f32_large_negative() {
    let (device, mut registry) = setup();

    let input_data: Vec<f32> = vec![-10.0, -20.0, -50.0];
    let n = input_data.len();
    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc output");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::gelu::dispatch_gelu(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
    )
    .expect("dispatch_gelu");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    // For large negative x, GELU(x) ~ 0
    for (i, _) in input_data.iter().enumerate() {
        let expected = gelu_ref(input_data[i]);
        let diff = (output[i] - expected).abs();
        assert!(
            diff <= 1e-5,
            "GELU f32 large negative mismatch at {}: expected={}, got={}",
            i, expected, output[i]
        );
    }
}

/// Convert f32 to bf16 raw bits (u16), using round-to-nearest-even.
fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// Convert bf16 raw bits (u16) back to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[test]
fn test_gelu_bf16_basic() {
    let (device, mut registry) = setup();

    let input_f32: Vec<f32> = vec![
        -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, -4.0, 0.1,
    ];
    let n = input_f32.len();
    let byte_len = n * 2; // bf16 = 2 bytes per element

    // Encode input as bf16 (u16 words)
    let input_bf16: Vec<u16> = input_f32.iter().copied().map(f32_to_bf16).collect();
    // Decode back to f32 to know the exact bf16-quantized reference values
    let input_as_f32: Vec<f32> = input_bf16.iter().copied().map(bf16_to_f32).collect();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![n])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![n])
        .expect("alloc output");

    {
        let slice: &mut [u16] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_bf16);
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::gelu::dispatch_gelu(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
    )
    .expect("dispatch_gelu bf16");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[u16] = output_buf.as_slice().expect("as_slice");
    for (i, &x) in input_as_f32.iter().enumerate() {
        let expected = gelu_ref(x);
        let actual = bf16_to_f32(output[i]);
        let diff = (actual - expected).abs();
        // bf16 has ~7.2 bits of mantissa; allow 5e-3 for output quantization error
        assert!(
            diff <= 5e-3,
            "GELU bf16 mismatch at index {}: input={}, expected={}, got={}, diff={}",
            i, x, expected, actual, diff
        );
    }
}

#[test]
fn test_gelu_bf16_zero() {
    let (device, mut registry) = setup();

    let byte_len = 2; // one bf16 element
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![1])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![1])
        .expect("alloc output");

    {
        let slice: &mut [u16] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = f32_to_bf16(0.0);
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::gelu::dispatch_gelu(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
    )
    .expect("dispatch_gelu bf16 zero");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[u16] = output_buf.as_slice().expect("as_slice");
    let result = bf16_to_f32(output[0]);
    assert!(
        result.abs() <= 1e-3,
        "GELU bf16(0) should be 0, got {}",
        result
    );
}

#[test]
fn test_gelu_invalid_dtype() {
    let (device, mut registry) = setup();

    let input_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("command_encoder");
    let result = mlx_native::ops::gelu::dispatch_gelu(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
    );
    assert!(result.is_err(), "Should error on unsupported dtype");
}
