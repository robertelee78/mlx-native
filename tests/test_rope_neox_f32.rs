//! Tests for the NeoX-convention RoPE (f32) with optional freq_factors.
//!
//! Verifies:
//!   1. Standard NeoX RoPE (no freq_factors) matches CPU reference.
//!   2. With freq_factors all 1.0: identical to no freq_factors.
//!   3. With freq_factors containing 1e30: those pairs are identity (no rotation).
//!   4. Partial rotary: non-rotated dimensions pass through.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, KernelRegistry, MlxDevice};

/// CPU reference: NeoX RoPE with optional freq_factors, matching forward_mlx.rs.
fn rope_neox_ref(
    input: &[f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    positions: &[u32],
    theta: f32,
    freq_factors: Option<&[f32]>,
) -> Vec<f32> {
    let seq_len = positions.len();
    let half_dim = head_dim / 2;
    let half_rope = rope_dim / 2;
    let n_rows = seq_len * n_heads;
    let mut output = vec![0.0f32; n_rows * head_dim];

    for row in 0..n_rows {
        let seq_idx = row / n_heads;
        let pos = positions[seq_idx] as f32;
        let base = row * head_dim;

        for i in 0..half_rope {
            let dim_ratio = (2 * i) as f32 / head_dim as f32;
            let mut freq = pos / theta.powf(dim_ratio);
            if let Some(ff) = freq_factors {
                freq /= ff[i];
            }
            let cos_a = freq.cos();
            let sin_a = freq.sin();

            let x0 = input[base + i];
            let x1 = input[base + i + half_dim];
            output[base + i] = x0 * cos_a - x1 * sin_a;
            output[base + i + half_dim] = x1 * cos_a + x0 * sin_a;
        }

        // Pass-through for non-rotated dimensions
        for i in half_rope..half_dim {
            output[base + i] = input[base + i];
            output[base + i + half_dim] = input[base + i + half_dim];
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

fn alloc_f32(device: &MlxDevice, data: &[f32]) -> mlx_native::MlxBuffer {
    let n = data.len();
    let byte_len = n * std::mem::size_of::<f32>();
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc f32");
    let s: &mut [f32] = buf.as_mut_slice().expect("write f32");
    s.copy_from_slice(data);
    buf
}

fn alloc_u32(device: &MlxDevice, data: &[u32]) -> mlx_native::MlxBuffer {
    let n = data.len();
    let byte_len = n * std::mem::size_of::<u32>();
    let mut buf = device
        .alloc_buffer(byte_len, DType::U32, vec![n])
        .expect("alloc u32");
    let s: &mut [u32] = buf.as_mut_slice().expect("write u32");
    s.copy_from_slice(data);
    buf
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    println!(
        "{name}: max|delta| = {max_diff} at index {max_idx} (actual={}, expected={})",
        actual[max_idx], expected[max_idx]
    );
    assert!(
        max_diff <= tol,
        "{name}: max|delta| = {max_diff} at index {max_idx} exceeds tol {tol}"
    );
}

/// Standard NeoX RoPE (no freq_factors), 4 heads x 8 dims.
#[test]
fn test_rope_neox_f32_no_freq_factors() {
    let (device, mut registry) = setup();
    let theta = 10000.0_f32;
    let seq_len: u32 = 1;
    let n_heads: u32 = 4;
    let head_dim: u32 = 8;
    let rope_dim: u32 = 8; // full rotary
    let n_rows = (seq_len as usize) * (n_heads as usize);
    let n = n_rows * (head_dim as usize);

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.6).collect();
    let positions: Vec<u32> = vec![42]; // position 42

    let input_buf = alloc_f32(&device, &input_data);
    let output_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc output");
    let params_buf = alloc_f32(&device, &[theta, head_dim as f32, rope_dim as f32, 0.0]);
    let positions_buf = alloc_u32(&device, &positions);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rope::dispatch_rope_neox_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        None, // no freq_factors
        seq_len,
        n_heads,
        head_dim,
        rope_dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rope_neox_ref(
        &input_data,
        n_heads as usize,
        head_dim as usize,
        rope_dim as usize,
        &positions,
        theta,
        None,
    );
    let output: &[f32] = output_buf.as_slice().expect("read output");
    assert_close(output, &expected, 1e-5, "neox_f32_no_ff");
}

/// With freq_factors all 1.0: should be identical to no freq_factors.
#[test]
fn test_rope_neox_f32_freq_factors_all_ones() {
    let (device, mut registry) = setup();
    let theta = 1000000.0_f32;
    let seq_len: u32 = 1;
    let n_heads: u32 = 2;
    let head_dim: u32 = 16;
    let rope_dim: u32 = 16;
    let half_rope = rope_dim / 2;
    let n_rows = (seq_len as usize) * (n_heads as usize);
    let n = n_rows * (head_dim as usize);

    let input_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).sin()).collect();
    let positions: Vec<u32> = vec![100];
    let ff_data: Vec<f32> = vec![1.0; half_rope as usize];

    let input_buf = alloc_f32(&device, &input_data);
    let output_with_ff = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc out1");
    let output_without_ff = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc out2");
    let params_buf = alloc_f32(&device, &[theta, head_dim as f32, rope_dim as f32, 0.0]);
    let positions_buf = alloc_u32(&device, &positions);
    let ff_buf = alloc_f32(&device, &ff_data);

    // Dispatch with freq_factors=1.0
    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rope::dispatch_rope_neox_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_with_ff,
        &params_buf,
        &positions_buf,
        Some(&ff_buf),
        seq_len,
        n_heads,
        head_dim,
        rope_dim,
    )
    .expect("dispatch with ff");
    encoder.commit_and_wait().expect("commit_and_wait");

    // Dispatch without freq_factors
    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rope::dispatch_rope_neox_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_without_ff,
        &params_buf,
        &positions_buf,
        None,
        seq_len,
        n_heads,
        head_dim,
        rope_dim,
    )
    .expect("dispatch without ff");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_ff: &[f32] = output_with_ff.as_slice().expect("read");
    let out_no: &[f32] = output_without_ff.as_slice().expect("read");
    assert_close(out_ff, out_no, 1e-5, "ff_ones_vs_no_ff");
}

/// With freq_factors containing 1e30: those pairs should be identity.
/// This is how Gemma 4's global layers mask out rotation for partial rotary.
#[test]
fn test_rope_neox_f32_freq_factors_identity_mask() {
    let (device, mut registry) = setup();
    let theta = 1000000.0_f32;
    let seq_len: u32 = 1;
    let n_heads: u32 = 2;
    let head_dim: u32 = 16;
    let rope_dim: u32 = 16;
    let half_rope = (rope_dim / 2) as usize;
    let half_dim = (head_dim / 2) as usize;
    let n_rows = (seq_len as usize) * (n_heads as usize);
    let n = n_rows * (head_dim as usize);

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 + 0.1).collect();
    let positions: Vec<u32> = vec![500];

    // First 4 pairs rotate normally, last 4 pairs have freq_factor=1e30 (identity)
    let mut ff_data = vec![1.0f32; half_rope];
    for i in (half_rope / 2)..half_rope {
        ff_data[i] = 1e30;
    }

    let input_buf = alloc_f32(&device, &input_data);
    let output_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc output");
    let params_buf = alloc_f32(&device, &[theta, head_dim as f32, rope_dim as f32, 0.0]);
    let positions_buf = alloc_u32(&device, &positions);
    let ff_buf = alloc_f32(&device, &ff_data);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rope::dispatch_rope_neox_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        Some(&ff_buf),
        seq_len,
        n_heads,
        head_dim,
        rope_dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rope_neox_ref(
        &input_data,
        n_heads as usize,
        head_dim as usize,
        rope_dim as usize,
        &positions,
        theta,
        Some(&ff_data),
    );
    let output: &[f32] = output_buf.as_slice().expect("read output");

    // Overall comparison vs CPU reference
    assert_close(output, &expected, 1e-5, "ff_identity_mask_vs_ref");

    // Verify identity pairs specifically: for pairs with ff=1e30,
    // output should approximately equal input (cos~1, sin~0).
    for row in 0..n_rows {
        let base = row * head_dim as usize;
        for i in (half_rope / 2)..half_rope {
            let x0_in = input_data[base + i];
            let x1_in = input_data[base + i + half_dim];
            let x0_out = output[base + i];
            let x1_out = output[base + i + half_dim];
            let diff0 = (x0_out - x0_in).abs();
            let diff1 = (x1_out - x1_in).abs();
            assert!(
                diff0 < 1e-4,
                "Row {row}, pair {i}: x0 rotated when ff=1e30: in={x0_in}, out={x0_out}, diff={diff0}"
            );
            assert!(
                diff1 < 1e-4,
                "Row {row}, pair {i}: x1 rotated when ff=1e30: in={x1_in}, out={x1_out}, diff={diff1}"
            );
        }
    }
}

/// Partial rotary: rope_dim < head_dim, non-rotated dims pass through.
#[test]
fn test_rope_neox_f32_partial_rotary() {
    let (device, mut registry) = setup();
    let theta = 10000.0_f32;
    let seq_len: u32 = 1;
    let n_heads: u32 = 2;
    let head_dim: u32 = 16;
    let rope_dim: u32 = 8; // only half the dims are rotated
    let half_dim = (head_dim / 2) as usize;
    let half_rope = (rope_dim / 2) as usize;
    let n_rows = (seq_len as usize) * (n_heads as usize);
    let n = n_rows * (head_dim as usize);

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.15 - 2.4).collect();
    let positions: Vec<u32> = vec![10];

    let input_buf = alloc_f32(&device, &input_data);
    let output_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc output");
    let params_buf = alloc_f32(&device, &[theta, head_dim as f32, rope_dim as f32, 0.0]);
    let positions_buf = alloc_u32(&device, &positions);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rope::dispatch_rope_neox_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        None,
        seq_len,
        n_heads,
        head_dim,
        rope_dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let expected = rope_neox_ref(
        &input_data,
        n_heads as usize,
        head_dim as usize,
        rope_dim as usize,
        &positions,
        theta,
        None,
    );
    let output: &[f32] = output_buf.as_slice().expect("read output");

    // Overall comparison vs CPU reference
    assert_close(output, &expected, 1e-5, "partial_rotary_vs_ref");

    // Verify non-rotated dimensions are exact pass-through
    for row in 0..n_rows {
        let base = row * head_dim as usize;
        for i in half_rope..half_dim {
            assert_eq!(
                output[base + i],
                input_data[base + i],
                "Row {row}, first-half passthrough index {i}"
            );
            assert_eq!(
                output[base + i + half_dim],
                input_data[base + i + half_dim],
                "Row {row}, second-half passthrough index {}",
                i + half_dim
            );
        }
    }
}

/// Position 0: cos(0)=1, sin(0)=0, so output should equal input.
#[test]
fn test_rope_neox_f32_position_zero() {
    let (device, mut registry) = setup();
    let theta = 10000.0_f32;
    let seq_len: u32 = 1;
    let n_heads: u32 = 3;
    let head_dim: u32 = 8;
    let rope_dim: u32 = 8;
    let n_rows = (seq_len as usize) * (n_heads as usize);
    let n = n_rows * (head_dim as usize);

    let input_data: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let positions: Vec<u32> = vec![0];

    let input_buf = alloc_f32(&device, &input_data);
    let output_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc output");
    let params_buf = alloc_f32(&device, &[theta, head_dim as f32, rope_dim as f32, 0.0]);
    let positions_buf = alloc_u32(&device, &positions);

    let mut encoder = device.command_encoder().expect("encoder");
    mlx_native::ops::rope::dispatch_rope_neox_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        &params_buf,
        &positions_buf,
        None,
        seq_len,
        n_heads,
        head_dim,
        rope_dim,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let output: &[f32] = output_buf.as_slice().expect("read output");
    for i in 0..n {
        let diff = (output[i] - input_data[i]).abs();
        assert!(
            diff <= 1e-5,
            "Position 0 should equal input: index {}, expected={}, got={}, diff={}",
            i,
            input_data[i],
            output[i],
            diff
        );
    }
}

/// Validation: odd rope_dim should error.
#[test]
fn test_rope_neox_f32_odd_rope_dim_error() {
    let (device, mut registry) = setup();

    let buf = device
        .alloc_buffer(12, DType::F32, vec![3])
        .expect("buf");
    let params = device
        .alloc_buffer(16, DType::F32, vec![4])
        .expect("params");
    let pos = device
        .alloc_buffer(4, DType::U32, vec![1])
        .expect("pos");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = mlx_native::ops::rope::dispatch_rope_neox_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        &params,
        &pos,
        None,
        1,
        1,
        3, // head_dim
        3, // odd rope_dim
    );
    assert!(result.is_err(), "Should error on odd rope_dim");
}
