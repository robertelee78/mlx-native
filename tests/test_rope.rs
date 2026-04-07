//! Tests for the Rotary Position Embedding (RoPE) GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// Reference RoPE implementation in pure Rust.
///
/// Processes a flat `[seq_len, head_dim]` array with given positions and theta.
fn rope_ref(input: &[f32], positions: &[u32], head_dim: usize, theta: f32) -> Vec<f32> {
    let seq_len = positions.len();
    let half_dim = head_dim / 2;
    let mut output = vec![0.0f32; seq_len * head_dim];

    for s in 0..seq_len {
        let pos = positions[s] as f32;
        for p in 0..half_dim {
            let dim_ratio = (2 * p) as f32 / head_dim as f32;
            let freq = 1.0 / theta.powf(dim_ratio);
            let angle = pos * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            let base = s * head_dim + 2 * p;
            let x0 = input[base];
            let x1 = input[base + 1];

            output[base] = x0 * cos_a - x1 * sin_a;
            output[base + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
    output
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::rope::register(&mut registry);
    (device, registry)
}

#[test]
fn test_rope_f32_theta_10000() {
    let (device, mut registry) = setup();
    let theta = 10000.0_f32;
    let seq_len: u32 = 4;
    let head_dim: u32 = 8;
    let n = (seq_len as usize) * (head_dim as usize);

    // Generate deterministic input data
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.6).collect();
    let positions: Vec<u32> = (0..seq_len).collect();

    let byte_len = n * std::mem::size_of::<f32>();
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![seq_len as usize, head_dim as usize])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![seq_len as usize, head_dim as usize])
        .expect("alloc output");

    // Params: [theta, head_dim, 0, 0]
    let params_byte_len = 4 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![4])
        .expect("alloc params");

    // Positions buffer
    let pos_byte_len = (seq_len as usize) * std::mem::size_of::<u32>();
    let mut positions_buf = device
        .alloc_buffer(pos_byte_len, DType::U32, vec![seq_len as usize])
        .expect("alloc positions");

    // Write data
    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }
    {
        let slice: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = theta;
        slice[1] = head_dim as f32;
        slice[2] = 0.0;
        slice[3] = 0.0;
    }
    {
        let slice: &mut [u32] = positions_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&positions);
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::rope::dispatch_rope(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        seq_len,
        head_dim,
    )
    .expect("dispatch_rope");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rope_ref(&input_data, &positions, head_dim as usize, theta);
    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "RoPE f32 theta=10000 mismatch at index {}: expected={}, got={}, diff={}",
            i, expected[i], output[i], diff
        );
    }
}

#[test]
fn test_rope_f32_theta_1000000() {
    let (device, mut registry) = setup();
    let theta = 1000000.0_f32;
    let seq_len: u32 = 4;
    let head_dim: u32 = 16;
    let n = (seq_len as usize) * (head_dim as usize);

    let input_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).sin()).collect();
    let positions: Vec<u32> = vec![0, 100, 500, 2048];

    let byte_len = n * std::mem::size_of::<f32>();
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![seq_len as usize, head_dim as usize])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![seq_len as usize, head_dim as usize])
        .expect("alloc output");

    let params_byte_len = 4 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![4])
        .expect("alloc params");

    let pos_byte_len = (seq_len as usize) * std::mem::size_of::<u32>();
    let mut positions_buf = device
        .alloc_buffer(pos_byte_len, DType::U32, vec![seq_len as usize])
        .expect("alloc positions");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }
    {
        let slice: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = theta;
        slice[1] = head_dim as f32;
        slice[2] = 0.0;
        slice[3] = 0.0;
    }
    {
        let slice: &mut [u32] = positions_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&positions);
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::rope::dispatch_rope(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        seq_len,
        head_dim,
    )
    .expect("dispatch_rope");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rope_ref(&input_data, &positions, head_dim as usize, theta);
    let output: &[f32] = output_buf.as_slice().expect("as_slice");

    for i in 0..n {
        let diff = (output[i] - expected[i]).abs();
        assert!(
            diff <= 1e-5,
            "RoPE f32 theta=1000000 mismatch at index {}: expected={}, got={}, diff={}",
            i, expected[i], output[i], diff
        );
    }
}

#[test]
fn test_rope_f32_position_zero() {
    // At position 0, cos(0)=1, sin(0)=0, so output should equal input.
    let (device, mut registry) = setup();
    let theta = 10000.0_f32;
    let seq_len: u32 = 1;
    let head_dim: u32 = 4;
    let n = (seq_len as usize) * (head_dim as usize);

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let byte_len = n * std::mem::size_of::<f32>();
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1, 4])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1, 4])
        .expect("alloc output");

    let params_byte_len = 4 * std::mem::size_of::<f32>();
    let mut params_buf = device
        .alloc_buffer(params_byte_len, DType::F32, vec![4])
        .expect("alloc params");

    let pos_byte_len = std::mem::size_of::<u32>();
    let mut positions_buf = device
        .alloc_buffer(pos_byte_len, DType::U32, vec![1])
        .expect("alloc positions");

    {
        let slice: &mut [f32] = input_buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(&input_data);
    }
    {
        let slice: &mut [f32] = params_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = theta;
        slice[1] = head_dim as f32;
        slice[2] = 0.0;
        slice[3] = 0.0;
    }
    {
        let slice: &mut [u32] = positions_buf.as_mut_slice().expect("as_mut_slice");
        slice[0] = 0;
    }

    let mut encoder = device.command_encoder().expect("command_encoder");
    mlx_native::ops::rope::dispatch_rope(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        seq_len,
        head_dim,
    )
    .expect("dispatch_rope");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("as_slice");
    for i in 0..n {
        let diff = (output[i] - input_data[i]).abs();
        assert!(
            diff <= 1e-5,
            "RoPE at position 0 should equal input: index {}, expected={}, got={}",
            i, input_data[i], output[i]
        );
    }
}

#[test]
fn test_rope_invalid_odd_head_dim() {
    let (device, mut registry) = setup();

    let input_buf = device
        .alloc_buffer(12, DType::F32, vec![1, 3])
        .expect("alloc input");
    let output_buf = device
        .alloc_buffer(12, DType::F32, vec![1, 3])
        .expect("alloc output");
    let params_buf = device
        .alloc_buffer(16, DType::F32, vec![4])
        .expect("alloc params");
    let positions_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("alloc positions");

    let mut encoder = device.command_encoder().expect("command_encoder");
    let result = mlx_native::ops::rope::dispatch_rope(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        1,
        3, // odd head_dim
    );
    assert!(result.is_err(), "Should error on odd head_dim");
}
