//! Tests for the softcap (tanh-based logit capping) GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// Reference softcap implementation.
fn softcap_ref(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::softcap::register(&mut registry);
    (device, registry)
}

#[test]
fn test_softcap_f32_basic() {
    let (device, mut registry) = setup();
    let cap = 30.0_f32;

    let input_data: Vec<f32> = vec![
        -100.0, -50.0, -30.0, -10.0, -1.0, 0.0, 1.0, 10.0, 30.0, 50.0, 100.0, 0.5,
    ];
    let n = input_data.len();
    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc output");

    // Params buffer: [cap]
    let params_byte_len = std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![1])
        .expect("alloc params");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }
    {
        let slice: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = cap;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::softcap::dispatch_softcap(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        cap,
    )
    .expect("dispatch_softcap");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    for (i, &x) in input_data.iter().enumerate() {
        let expected = softcap_ref(x, cap);
        let actual = output[i];
        let diff = (actual - expected).abs();
        assert!(
            diff <= 1e-5,
            "Softcap f32 mismatch at index {}: input={}, expected={}, got={}, diff={}",
            i, x, expected, actual, diff
        );
    }
}

#[test]
fn test_softcap_f32_output_bounded() {
    let (device, mut registry) = setup();
    let cap = 30.0_f32;

    let input_data: Vec<f32> = vec![-1000.0, -500.0, 0.0, 500.0, 1000.0];
    let n = input_data.len();
    let byte_len = n * std::mem::size_of::<f32>();

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(std::mem::size_of::<f32>(), DType::F32, vec![1])
        .expect("alloc params");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }
    {
        let slice: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = cap;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::softcap::dispatch_softcap(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        cap,
    )
    .expect("dispatch_softcap");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    for (i, &val) in output.iter().enumerate() {
        // Allow a tiny epsilon for floating-point imprecision in tanh * cap
        assert!(
            val.abs() <= cap + 1e-4,
            "Softcap output at {} should be bounded near [-{}, +{}], got {}",
            i, cap, cap, val
        );
    }
}

#[test]
fn test_softcap_f32_zero_input() {
    let (device, mut registry) = setup();
    let cap = 30.0_f32;

    let mut input_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc output");
    let mut params_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc params");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = 0.0;
    }
    {
        let slice: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = cap;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::softcap::dispatch_softcap(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        cap,
    )
    .expect("dispatch_softcap");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    assert!(
        output[0].abs() <= 1e-7,
        "Softcap(0) should be 0, got {}",
        output[0]
    );
}

#[test]
fn test_softcap_invalid_cap() {
    let (device, mut registry) = setup();

    let input_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc output");
    let params_buf = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc params");

    let mut encoder = device.command_encoder().expect("command_encoder");
    let result = mlx_native::ops::softcap::dispatch_softcap(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        -1.0, // invalid cap
    );
    assert!(result.is_err(), "Should error on negative cap");
}
