//! Tests for fused GPU kernels:
//!   - fused_residual_norm_bf16  (residual + input, then rms_norm)
//!   - fused_norm_add_bf16       (rms_norm(input), then residual + normed)
//!   - fused_head_norm_rope_bf16 (per-head rms_norm + neox RoPE)
//!
//! Each fused kernel is compared against a CPU reference that performs the
//! equivalent sequential operations at bf16 precision.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_head_norm_rope;
use mlx_native::ops::fused_norm_add;
use mlx_native::ops::fused_residual_norm;
use mlx_native::{DType, KernelRegistry, MlxDevice};

// --------------------------------------------------------------------------
// bf16 conversion helpers
// --------------------------------------------------------------------------

fn f32_to_bf16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let bf16_bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    bf16_bits.to_le_bytes()
}

fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bf16_bits = u16::from_le_bytes(bytes);
    f32::from_bits((bf16_bits as u32) << 16)
}

fn f32_to_bf16_f32(val: f32) -> f32 {
    bf16_bytes_to_f32(f32_to_bf16_bytes(val))
}

fn write_bf16(buf: &mut [u8], values: &[f32]) {
    for (i, &v) in values.iter().enumerate() {
        let b = f32_to_bf16_bytes(v);
        buf[i * 2] = b[0];
        buf[i * 2 + 1] = b[1];
    }
}

fn read_bf16(buf: &[u8], count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| bf16_bytes_to_f32([buf[i * 2], buf[i * 2 + 1]]))
        .collect()
}

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

// --------------------------------------------------------------------------
// CPU references
// --------------------------------------------------------------------------

/// CPU rms_norm: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
fn cpu_rms_norm(input: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let rows = input.len() / dim;
    let mut output = vec![0.0f32; input.len()];
    for r in 0..rows {
        let row = &input[r * dim..(r + 1) * dim];
        let sq_sum: f32 = row.iter().map(|x| x * x).sum();
        let rms = (sq_sum / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for d in 0..dim {
            output[r * dim + d] = row[d] * inv_rms * weight[d];
        }
    }
    output
}

// CPU rms_norm without weight scale (available for fused_norm_add_no_weight tests)
// fn cpu_rms_norm_no_weight(input: &[f32], dim: usize, eps: f32) -> Vec<f32> {
//     let rows = input.len() / dim;
//     let mut output = vec![0.0f32; input.len()];
//     for r in 0..rows {
//         let row = &input[r * dim..(r + 1) * dim];
//         let sq_sum: f32 = row.iter().map(|x| x * x).sum();
//         let rms = (sq_sum / dim as f32 + eps).sqrt();
//         let inv_rms = 1.0 / rms;
//         for d in 0..dim {
//             output[r * dim + d] = row[d] * inv_rms;
//         }
//     }
//     output
// }

/// CPU elementwise add
fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// CPU reference for fused_residual_norm: sum = residual + input, then rms_norm(sum, weight, eps)
fn cpu_fused_residual_norm(
    residual: &[f32],
    input: &[f32],
    weight: &[f32],
    dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let sum = cpu_add(residual, input);
    let normed = cpu_rms_norm(&sum, weight, dim, eps);
    (normed, sum)
}

/// CPU reference for fused_norm_add: normed = rms_norm(input, weight, eps), output = residual + normed
fn cpu_fused_norm_add(
    residual: &[f32],
    input: &[f32],
    weight: &[f32],
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let normed = cpu_rms_norm(input, weight, dim, eps);
    cpu_add(residual, &normed)
}

/// CPU reference for fused_head_norm_rope: per-head rms_norm + NeoX RoPE
fn cpu_fused_head_norm_rope(
    input: &[f32],     // [n_heads * head_dim]
    weight: &[f32],    // [head_dim]
    cos_cache: &[f32], // [half_rope_dim]
    sin_cache: &[f32], // [half_rope_dim]
    n_heads: usize,
    head_dim: usize,
    half_rope_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let head_start = h * head_dim;
        let head_slice = &input[head_start..head_start + head_dim];

        // RMS norm with weight
        let sq_sum: f32 = head_slice.iter().map(|x| x * x).sum();
        let rms = (sq_sum / head_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        let mut normed = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            normed[d] = head_slice[d] * inv_rms * weight[d];
        }

        // NeoX RoPE: first half and second half are paired
        // NeoX convention: x[i] paired with x[i + half_rope_dim]
        for p in 0..half_rope_dim {
            let x0 = normed[p];
            let x1 = normed[p + half_rope_dim];
            let cos_val = cos_cache[p];
            let sin_val = sin_cache[p];
            output[head_start + p] = x0 * cos_val - x1 * sin_val;
            output[head_start + p + half_rope_dim] = x0 * sin_val + x1 * cos_val;
        }

        // Copy dimensions beyond 2*half_rope_dim (pass-through)
        for d in (2 * half_rope_dim)..head_dim {
            output[head_start + d] = normed[d];
        }
    }

    output
}

// --------------------------------------------------------------------------
// Helpers for buffer creation
// --------------------------------------------------------------------------

fn alloc_bf16(device: &MlxDevice, data: &[f32]) -> mlx_native::MlxBuffer {
    let n = data.len();
    let byte_len = n * 2;
    let mut buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![n])
        .expect("alloc bf16");
    let bytes = buf.as_mut_slice::<u8>().expect("write bf16");
    write_bf16(bytes, data);
    buf
}

fn read_bf16_buf(buf: &mlx_native::MlxBuffer) -> Vec<f32> {
    let bytes: &[u8] = buf.as_slice().expect("read bf16");
    let count = buf.element_count();
    read_bf16(bytes, count)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, test_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{test_name}: length mismatch"
    );
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    println!(
        "{test_name}: max|delta| = {max_diff} at index {max_diff_idx} \
         (actual={}, expected={})",
        actual[max_diff_idx], expected[max_diff_idx]
    );
    assert!(
        max_diff <= tol,
        "{test_name}: max|delta| = {max_diff} at index {max_diff_idx} exceeds tolerance {tol} \
         (actual={}, expected={})",
        actual[max_diff_idx],
        expected[max_diff_idx]
    );
}

// --------------------------------------------------------------------------
// Tests: fused_residual_norm_bf16
// --------------------------------------------------------------------------

#[test]
fn test_fused_residual_norm_bf16() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    fused_residual_norm::register(&mut registry);

    let rows: u32 = 4;
    let dim: u32 = 128;
    let eps: f32 = 1e-6;
    let n = rows as usize * dim as usize;

    let residual_data = pseudo_random_f32(42, n);
    let input_data = pseudo_random_f32(137, n);
    let weight_data = pseudo_random_f32(999, dim as usize);

    // Round to bf16 for CPU reference
    let residual_bf16: Vec<f32> = residual_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();
    let input_bf16: Vec<f32> = input_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();
    let weight_bf16: Vec<f32> = weight_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();

    let (expected_normed, expected_sum) =
        cpu_fused_residual_norm(&residual_bf16, &input_bf16, &weight_bf16, dim as usize, eps);

    // GPU buffers
    let residual_buf = alloc_bf16(&device, &residual_data);
    let input_buf = alloc_bf16(&device, &input_data);
    let weight_buf = alloc_bf16(&device, &weight_data);
    let normed_buf = device
        .alloc_buffer(n * 2, DType::BF16, vec![n])
        .expect("alloc normed");
    let sum_buf = device
        .alloc_buffer(n * 2, DType::BF16, vec![n])
        .expect("alloc sum");

    let mut encoder = device.command_encoder().expect("encoder");
    fused_residual_norm::dispatch_fused_residual_norm_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &residual_buf,
        &input_buf,
        &weight_buf,
        &normed_buf,
        Some(&sum_buf),
        rows,
        dim,
        eps,
    )
    .expect("dispatch_fused_residual_norm_bf16");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual_normed = read_bf16_buf(&normed_buf);
    let actual_sum = read_bf16_buf(&sum_buf);

    // bf16 precision: allow slightly larger tolerance due to fused vs sequential differences
    assert_close(&actual_normed, &expected_normed, 5e-2, "fused_residual_norm normed");
    // bf16 addition can lose precision; use same tolerance as normed output
    assert_close(&actual_sum, &expected_sum, 5e-2, "fused_residual_norm sum");
}

// --------------------------------------------------------------------------
// Tests: fused_norm_add_bf16
// --------------------------------------------------------------------------

#[test]
fn test_fused_norm_add_bf16() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    fused_norm_add::register(&mut registry);

    let rows: u32 = 4;
    let dim: u32 = 128;
    let eps: f32 = 1e-6;
    let n = rows as usize * dim as usize;

    let residual_data = pseudo_random_f32(42, n);
    let input_data = pseudo_random_f32(137, n);
    let weight_data = pseudo_random_f32(999, dim as usize);

    let residual_bf16: Vec<f32> = residual_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();
    let input_bf16: Vec<f32> = input_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();
    let weight_bf16: Vec<f32> = weight_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();

    let expected = cpu_fused_norm_add(&residual_bf16, &input_bf16, &weight_bf16, dim as usize, eps);

    let residual_buf = alloc_bf16(&device, &residual_data);
    let input_buf = alloc_bf16(&device, &input_data);
    let weight_buf = alloc_bf16(&device, &weight_data);
    let output_buf = device
        .alloc_buffer(n * 2, DType::BF16, vec![n])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    fused_norm_add::dispatch_fused_norm_add_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &residual_buf,
        &input_buf,
        &weight_buf,
        &output_buf,
        dim,
        rows,
        eps,
    )
    .expect("dispatch_fused_norm_add_bf16");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual = read_bf16_buf(&output_buf);
    assert_close(&actual, &expected, 5e-2, "fused_norm_add");
}

// --------------------------------------------------------------------------
// Tests: fused_head_norm_rope_bf16
// --------------------------------------------------------------------------

#[test]
fn test_fused_head_norm_rope_bf16() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    fused_head_norm_rope::register(&mut registry);

    let n_heads: u32 = 8;
    let head_dim: u32 = 128;
    let half_rope_dim: u32 = 64; // head_dim / 2
    let eps: f32 = 1e-6;
    let total_elements = n_heads as usize * head_dim as usize;

    let input_data = pseudo_random_f32(42, total_elements);
    let weight_data = pseudo_random_f32(137, head_dim as usize);

    // Precompute cos/sin cache for position 0 with theta=10000
    let theta = 10000.0f32;
    let cos_cache: Vec<f32> = (0..half_rope_dim as usize)
        .map(|p| {
            let dim_ratio = (2 * p) as f32 / head_dim as f32;
            let freq = 1.0 / theta.powf(dim_ratio);
            // Position 0: cos(0) = 1.0 for all frequencies
            // Use position 5 for more interesting values
            let angle = 5.0 * freq;
            angle.cos()
        })
        .collect();
    let sin_cache: Vec<f32> = (0..half_rope_dim as usize)
        .map(|p| {
            let dim_ratio = (2 * p) as f32 / head_dim as f32;
            let freq = 1.0 / theta.powf(dim_ratio);
            let angle = 5.0 * freq;
            angle.sin()
        })
        .collect();

    // Round inputs to bf16 for CPU reference
    let input_bf16: Vec<f32> = input_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();
    let weight_bf16: Vec<f32> = weight_data.iter().map(|&v| f32_to_bf16_f32(v)).collect();

    let expected = cpu_fused_head_norm_rope(
        &input_bf16,
        &weight_bf16,
        &cos_cache,
        &sin_cache,
        n_heads as usize,
        head_dim as usize,
        half_rope_dim as usize,
        eps,
    );

    // GPU buffers
    let input_buf = alloc_bf16(&device, &input_data);
    let output_buf = device
        .alloc_buffer(total_elements * 2, DType::BF16, vec![total_elements])
        .expect("alloc output");
    let weight_buf = alloc_bf16(&device, &weight_data);

    // cos/sin caches are f32
    let cos_bytes = half_rope_dim as usize * 4;
    let mut cos_buf = device
        .alloc_buffer(cos_bytes, DType::F32, vec![half_rope_dim as usize])
        .expect("alloc cos");
    cos_buf
        .as_mut_slice::<f32>()
        .expect("write cos")
        .copy_from_slice(&cos_cache);

    let mut sin_buf = device
        .alloc_buffer(cos_bytes, DType::F32, vec![half_rope_dim as usize])
        .expect("alloc sin");
    sin_buf
        .as_mut_slice::<f32>()
        .expect("write sin")
        .copy_from_slice(&sin_cache);

    let mut encoder = device.command_encoder().expect("encoder");
    fused_head_norm_rope::dispatch_fused_head_norm_rope_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        Some(&weight_buf),
        &cos_buf,
        &sin_buf,
        n_heads,
        head_dim,
        half_rope_dim,
        eps,
    )
    .expect("dispatch_fused_head_norm_rope_bf16");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual = read_bf16_buf(&output_buf);
    // bf16 precision with multiple fused ops: use 5e-2 tolerance
    assert_close(&actual, &expected, 5e-2, "fused_head_norm_rope");
}

// --------------------------------------------------------------------------
// Validation error tests
// --------------------------------------------------------------------------

#[test]
fn test_fused_residual_norm_zero_dim_error() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    fused_residual_norm::register(&mut registry);

    let buf = device.alloc_buffer(2, DType::BF16, vec![1]).expect("buf");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = fused_residual_norm::dispatch_fused_residual_norm_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        &buf,
        &buf,
        None,
        0, // rows = 0
        1, // dim
        1e-6,
    );
    assert!(result.is_err(), "Should error on rows=0");
}

#[test]
fn test_fused_norm_add_zero_dim_error() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    fused_norm_add::register(&mut registry);

    let buf = device.alloc_buffer(2, DType::BF16, vec![1]).expect("buf");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = fused_norm_add::dispatch_fused_norm_add_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        &buf,
        &buf,
        0, // dim = 0
        1, // rows
        1e-6,
    );
    assert!(result.is_err(), "Should error on dim=0");
}

#[test]
fn test_fused_head_norm_rope_zero_heads_error() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    fused_head_norm_rope::register(&mut registry);

    let buf = device.alloc_buffer(2, DType::BF16, vec![1]).expect("buf");
    let f32_buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("f32buf");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = fused_head_norm_rope::dispatch_fused_head_norm_rope_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &buf,
        &buf,
        None,
        &f32_buf,
        &f32_buf,
        0,   // n_heads = 0
        128, // head_dim
        64,  // half_rope_dim
        1e-6,
    );
    assert!(result.is_err(), "Should error on n_heads=0");
}
