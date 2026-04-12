//! Tests for the argmax GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::argmax;
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

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    argmax::register(&mut registry);
    (device, registry)
}

/// CPU reference argmax: returns (index, value) of the maximum element.
fn cpu_argmax(data: &[f32]) -> (u32, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    (max_idx as u32, max_val)
}

#[test]
fn test_argmax_262144() {
    let (device, mut registry) = setup();
    let n: u32 = 262144; // Production vocab size

    let mut data = pseudo_random_f32(42, n as usize);
    // Plant a known max at a specific position to verify correctness
    let max_pos = 131072;
    data[max_pos] = 100.0;

    let (expected_idx, expected_val) = cpu_argmax(&data);
    assert_eq!(expected_idx, max_pos as u32);

    let byte_len = n as usize * 4;
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&data);

    let out_index = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc out_index");
    let out_value = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc out_value");

    let mut params_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc params");
    params_buf.as_mut_slice::<u32>().expect("write params")[0] = n;

    let mut encoder = device.command_encoder().expect("encoder");
    argmax::dispatch_argmax_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &out_index,
        &out_value,
        &params_buf,
        n,
    )
    .expect("dispatch_argmax_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual_idx = out_index.as_slice::<u32>().expect("read index")[0];
    let actual_val = out_value.as_slice::<f32>().expect("read value")[0];

    assert_eq!(
        actual_idx, expected_idx,
        "argmax index mismatch: GPU={}, CPU={} (expected value={})",
        actual_idx, expected_idx, expected_val
    );
    assert!(
        (actual_val - expected_val).abs() < 1e-6,
        "argmax value mismatch: GPU={}, CPU={}",
        actual_val,
        expected_val
    );
}

#[test]
fn test_argmax_random_no_planted_max() {
    let (device, mut registry) = setup();
    let n: u32 = 262144;

    let data = pseudo_random_f32(999, n as usize);
    let (expected_idx, expected_val) = cpu_argmax(&data);

    let byte_len = n as usize * 4;
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&data);

    let out_index = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc out_index");
    let out_value = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc out_value");

    let mut params_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc params");
    params_buf.as_mut_slice::<u32>().expect("write params")[0] = n;

    let mut encoder = device.command_encoder().expect("encoder");
    argmax::dispatch_argmax_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &out_index,
        &out_value,
        &params_buf,
        n,
    )
    .expect("dispatch_argmax_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual_idx = out_index.as_slice::<u32>().expect("read index")[0];
    let actual_val = out_value.as_slice::<f32>().expect("read value")[0];

    assert_eq!(
        actual_idx, expected_idx,
        "argmax random: GPU index={}, CPU index={} (value: GPU={}, CPU={})",
        actual_idx, expected_idx, actual_val, expected_val
    );
    assert!(
        (actual_val - expected_val).abs() < 1e-6,
        "argmax random: value mismatch GPU={}, CPU={}",
        actual_val,
        expected_val
    );
}

#[test]
fn test_argmax_small() {
    let (device, mut registry) = setup();
    let data: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let n = data.len() as u32;

    let byte_len = n as usize * 4;
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&data);

    let out_index = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc out_index");
    let out_value = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("alloc out_value");

    let mut params_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc params");
    params_buf.as_mut_slice::<u32>().expect("write params")[0] = n;

    let mut encoder = device.command_encoder().expect("encoder");
    argmax::dispatch_argmax_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &out_index,
        &out_value,
        &params_buf,
        n,
    )
    .expect("dispatch_argmax_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual_idx = out_index.as_slice::<u32>().expect("read index")[0];
    let actual_val = out_value.as_slice::<f32>().expect("read value")[0];

    assert_eq!(actual_idx, 1, "argmax small: expected index 1, got {}", actual_idx);
    assert!(
        (actual_val - 5.0).abs() < 1e-6,
        "argmax small: expected value 5.0, got {}",
        actual_val
    );
}

#[test]
fn test_argmax_zero_elements_error() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("buf");
    let out_idx = device.alloc_buffer(4, DType::U32, vec![1]).expect("out_idx");
    let out_val = device.alloc_buffer(4, DType::F32, vec![1]).expect("out_val");
    let params = device.alloc_buffer(4, DType::U32, vec![1]).expect("params");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = argmax::dispatch_argmax_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &out_idx,
        &out_val,
        &params,
        0,
    );

    assert!(result.is_err(), "Should error on n_elements=0");
}
