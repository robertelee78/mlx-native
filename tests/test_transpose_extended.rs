//! Tests for transpose GPU kernels: permute_021_bf16 on [A, B, C] tensors.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::transpose;
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

/// Convert f32 to bf16 raw bytes (2 bytes LE).
fn f32_to_bf16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let bf16_bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    bf16_bits.to_le_bytes()
}

/// Convert bf16 raw bytes to f32.
fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bf16_bits = u16::from_le_bytes(bytes);
    f32::from_bits((bf16_bits as u32) << 16)
}

/// CPU reference: permute [A, B, C] -> [B, A, C] in bf16.
///
/// Input layout:  input[a][b][c]  = input_flat[a * B * C + b * C + c]
/// Output layout: output[b][a][c] = output_flat[b * A * C + a * C + c]
fn cpu_permute_021(input: &[f32], dim_a: usize, dim_b: usize, dim_c: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; dim_a * dim_b * dim_c];
    for a in 0..dim_a {
        for b in 0..dim_b {
            for c in 0..dim_c {
                let in_idx = a * dim_b * dim_c + b * dim_c + c;
                let out_idx = b * dim_a * dim_c + a * dim_c + c;
                output[out_idx] = input[in_idx];
            }
        }
    }
    output
}

#[test]
fn test_permute_021_bf16_4_32_128() {
    let (device, mut registry) = setup();

    let dim_a = 4;
    let dim_b = 32;
    let dim_c = 128;
    let total = dim_a * dim_b * dim_c;

    // Generate test data as f32, then convert to bf16
    let input_f32: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 80.0).collect();

    // Convert to bf16 bytes
    let mut input_bytes = vec![0u8; total * 2];
    for (i, &v) in input_f32.iter().enumerate() {
        let bytes = f32_to_bf16_bytes(v);
        input_bytes[i * 2] = bytes[0];
        input_bytes[i * 2 + 1] = bytes[1];
    }

    // Read back as bf16-rounded f32 for the CPU reference
    let input_bf16_f32: Vec<f32> = (0..total)
        .map(|i| bf16_bytes_to_f32([input_bytes[i * 2], input_bytes[i * 2 + 1]]))
        .collect();

    let expected = cpu_permute_021(&input_bf16_f32, dim_a, dim_b, dim_c);

    // Allocate GPU buffers
    let byte_len = total * 2; // bf16

    let mut input_buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![total])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<u8>()
        .expect("write input")
        .copy_from_slice(&input_bytes);

    let output_buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![total])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    transpose::permute_021_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        dim_a,
        dim_b,
        dim_c,
    )
    .expect("permute_021_bf16 dispatch");

    encoder.commit_and_wait().expect("commit_and_wait");

    // Read back and compare
    let output_bytes: Vec<u8> = output_buf.as_slice::<u8>().expect("read output").to_vec();

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..total {
        let actual = bf16_bytes_to_f32([output_bytes[i * 2], output_bytes[i * 2 + 1]]);
        let diff = (actual - expected[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    let actual_at_max = bf16_bytes_to_f32([
        output_bytes[max_diff_idx * 2],
        output_bytes[max_diff_idx * 2 + 1],
    ]);

    println!(
        "permute_021 [4,32,128]: max|delta| = {} at index {} (actual={}, expected={})",
        max_diff, max_diff_idx, actual_at_max, expected[max_diff_idx]
    );

    // bf16 copy should be bitwise exact
    assert!(
        max_diff < 1e-10,
        "permute_021: max|delta| = {} at index {} (actual={}, expected={}) -- should be bitwise exact for a copy/permute",
        max_diff,
        max_diff_idx,
        actual_at_max,
        expected[max_diff_idx]
    );
}

#[test]
fn test_permute_021_bf16_1_1_1() {
    let (device, mut registry) = setup();

    let dim_a = 1;
    let dim_b = 1;
    let dim_c = 1;
    let _total = 1;

    let val: f32 = 3.14;
    let bf16 = f32_to_bf16_bytes(val);

    let mut input_buf = device
        .alloc_buffer(2, DType::BF16, vec![1])
        .expect("alloc input");
    input_buf.as_mut_slice::<u8>().expect("write")[0] = bf16[0];
    input_buf.as_mut_slice::<u8>().expect("write")[1] = bf16[1];

    let output_buf = device
        .alloc_buffer(2, DType::BF16, vec![1])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    transpose::permute_021_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        dim_a,
        dim_b,
        dim_c,
    )
    .expect("permute_021_bf16");

    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bytes: Vec<u8> = output_buf.as_slice::<u8>().expect("read").to_vec();
    let result = bf16_bytes_to_f32([out_bytes[0], out_bytes[1]]);
    let expected = bf16_bytes_to_f32(bf16);

    assert!(
        (result - expected).abs() < 1e-10,
        "1x1x1 permute: expected {}, got {}",
        expected,
        result
    );
}

#[test]
fn test_permute_021_zero_dim_error() {
    let (device, mut registry) = setup();

    let buf = device
        .alloc_buffer(2, DType::BF16, vec![1])
        .expect("buf");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = transpose::permute_021_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        0,
        1,
        1,
    );

    assert!(result.is_err(), "Should error on dim_a=0");
}
