//! Unit tests for quantized_matmul GPU kernel — Q4_0, Q6_K, Q8_0.
//!
//! Tests all supported quant types at hf2q's production shapes:
//!   - M=1, N=4096, K=4096 (decode, attention projection)
//!   - M=1, N=14336, K=4096 (decode, MLP projection)
//!   - M=8, N=4096, K=4096 (prefill)
//!
//! CPU reference: dequantize weights to f32 (matching bf16 intermediate precision
//! of the Metal shader), then matmul.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, KernelRegistry, MlxDevice, QuantizedMatmulParams};

// --------------------------------------------------------------------------
// Pseudo-random number generator (matches other test files' pattern)
// --------------------------------------------------------------------------

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
            frac
        })
        .collect()
}

// --------------------------------------------------------------------------
// bf16 helpers (matching test_embedding.rs pattern)
// --------------------------------------------------------------------------

fn f32_to_bf16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let bf16_bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    bf16_bits.to_le_bytes()
}

fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bf16_bits = u16::from_le_bytes(bytes);
    let f32_bits = (bf16_bits as u32) << 16;
    f32::from_bits(f32_bits)
}

fn f32_to_bf16_f32(val: f32) -> f32 {
    bf16_bytes_to_f32(f32_to_bf16_bytes(val))
}

// --------------------------------------------------------------------------
// Quantization helpers (4-bit, 6-bit, 8-bit)
// --------------------------------------------------------------------------

fn quantize_4bit_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n_bins = 15.0_f32;
    let range = max_val - min_val;
    let scale = if range.abs() < 1e-10 { 1.0 } else { range / n_bins };
    let bias = min_val;

    let mut packed = Vec::new();
    for chunk in values.chunks(8) {
        let mut word: u32 = 0;
        for (k, &v) in chunk.iter().enumerate() {
            let uint_val = ((v - bias) / scale).round().clamp(0.0, n_bins) as u32;
            word |= (uint_val & 0xF) << (k * 4);
        }
        packed.extend_from_slice(&word.to_le_bytes());
    }

    (packed, scale, bias)
}

fn quantize_6bit_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n_bins = 63.0_f32;
    let range = max_val - min_val;
    let scale = if range.abs() < 1e-10 { 1.0 } else { range / n_bins };
    let bias = min_val;

    let mut packed = Vec::new();
    for chunk in values.chunks(4) {
        let mut pack: u32 = 0;
        for (k, &v) in chunk.iter().enumerate() {
            let uint_val = ((v - bias) / scale).round().clamp(0.0, n_bins) as u32;
            pack |= (uint_val & 0x3F) << (k * 6);
        }
        packed.push((pack & 0xFF) as u8);
        packed.push(((pack >> 8) & 0xFF) as u8);
        packed.push(((pack >> 16) & 0xFF) as u8);
    }

    (packed, scale, bias)
}

fn quantize_8bit_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n_bins = 255.0_f32;
    let range = max_val - min_val;
    let scale = if range.abs() < 1e-10 { 1.0 } else { range / n_bins };
    let bias = min_val;

    let mut packed = Vec::new();
    for chunk in values.chunks(4) {
        let mut word: u32 = 0;
        for (k, &v) in chunk.iter().enumerate() {
            let uint_val = ((v - bias) / scale).round().clamp(0.0, n_bins) as u32;
            word |= (uint_val & 0xFF) << (k * 8);
        }
        packed.extend_from_slice(&word.to_le_bytes());
    }

    (packed, scale, bias)
}

// --------------------------------------------------------------------------
// Dequantization helpers (matching the Metal shader's bf16 precision)
// --------------------------------------------------------------------------

fn dequant_4bit_value(packed_word: u32, idx: usize, scale_bf16: f32, bias_bf16: f32) -> f32 {
    let val = (packed_word >> (4 * idx)) & 0xF;
    // Match shader: bfloat(val) * scale + bias, then cast to f32
    let w_bf16 = f32_to_bf16_f32(val as f32) * scale_bf16 + bias_bf16;
    w_bf16
}

fn dequant_6bit_value(packed_triplet: u32, idx: usize, scale_bf16: f32, bias_bf16: f32) -> f32 {
    let val = (packed_triplet >> (6 * idx)) & 0x3F;
    let w_bf16 = f32_to_bf16_f32(val as f32) * scale_bf16 + bias_bf16;
    w_bf16
}

fn dequant_8bit_value(packed_word: u32, idx: usize, scale_bf16: f32, bias_bf16: f32) -> f32 {
    let val = (packed_word >> (8 * idx)) & 0xFF;
    let w_bf16 = f32_to_bf16_f32(val as f32) * scale_bf16 + bias_bf16;
    w_bf16
}

// --------------------------------------------------------------------------
// CPU reference matmul with bf16-precision dequant
// --------------------------------------------------------------------------

/// Dequantize entire weight matrix to f32 (via bf16), then compute matmul.
///
/// Weight layout: [N, K] packed, scales [N, num_groups], biases [N, num_groups]
/// Input layout: [M, K] f32
/// Output: [M, N] f32
fn cpu_quantized_matmul(
    input: &[f32],
    weight_packed: &[u8],
    scales_raw: &[u8],
    biases_raw: &[u8],
    m: usize,
    k: usize,
    n: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    let num_groups = (k + group_size - 1) / group_size;
    let mut output = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let sb_base = col * num_groups;
            let mut acc = 0.0f32;

            for ki in 0..k {
                let g = ki / group_size;
                let scale_off = (sb_base + g) * 2;
                let scale_bf16 = bf16_bytes_to_f32([scales_raw[scale_off], scales_raw[scale_off + 1]]);
                let bias_bf16 = bf16_bytes_to_f32([biases_raw[scale_off], biases_raw[scale_off + 1]]);

                let w = match bits {
                    4 => {
                        let values_per_pack = 8;
                        let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
                        let w_base = col * packs_per_row;
                        let pack_idx = ki / values_per_pack;
                        let in_pack = ki % values_per_pack;
                        let byte_off = (w_base + pack_idx) * 4;
                        let packed = u32::from_le_bytes([
                            weight_packed[byte_off],
                            weight_packed[byte_off + 1],
                            weight_packed[byte_off + 2],
                            weight_packed[byte_off + 3],
                        ]);
                        dequant_4bit_value(packed, in_pack, scale_bf16, bias_bf16)
                    }
                    6 => {
                        let triplets_per_row = (k + 3) / 4;
                        let row_bytes = triplets_per_row * 3;
                        let w_row_base = col * row_bytes;
                        let triplet_idx = ki / 4;
                        let in_triplet = ki % 4;
                        let byte_off = w_row_base + triplet_idx * 3;
                        let packed = (weight_packed[byte_off] as u32)
                            | ((weight_packed[byte_off + 1] as u32) << 8)
                            | ((weight_packed[byte_off + 2] as u32) << 16);
                        dequant_6bit_value(packed, in_triplet, scale_bf16, bias_bf16)
                    }
                    8 => {
                        let values_per_pack = 4;
                        let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
                        let w_base = col * packs_per_row;
                        let pack_idx = ki / values_per_pack;
                        let in_pack = ki % values_per_pack;
                        let byte_off = (w_base + pack_idx) * 4;
                        let packed = u32::from_le_bytes([
                            weight_packed[byte_off],
                            weight_packed[byte_off + 1],
                            weight_packed[byte_off + 2],
                            weight_packed[byte_off + 3],
                        ]);
                        dequant_8bit_value(packed, in_pack, scale_bf16, bias_bf16)
                    }
                    _ => panic!("unsupported bits {}", bits),
                };

                // Match shader: bfloat(input[row * K + k]) cast to float, then multiply
                let x_bf16 = f32_to_bf16_f32(input[row * k + ki]);
                acc += w * x_bf16;
            }

            output[row * n + col] = acc;
        }
    }

    output
}

// --------------------------------------------------------------------------
// Build weight/scales/biases buffers from f32 weight matrix
// --------------------------------------------------------------------------

struct QuantizedWeightData {
    weight_packed: Vec<u8>,
    scales_raw: Vec<u8>,
    biases_raw: Vec<u8>,
}

fn quantize_weight_matrix(
    float_weights: &[f32], // [N, K] row-major (N rows of K elements each)
    n: usize,
    k: usize,
    group_size: usize,
    bits: usize,
) -> QuantizedWeightData {
    let num_groups = (k + group_size - 1) / group_size;
    let mut all_packed = Vec::new();
    let mut all_scales = Vec::new();
    let mut all_biases = Vec::new();

    for col in 0..n {
        let row_data = &float_weights[col * k..(col + 1) * k];
        for g in 0..num_groups {
            let start = g * group_size;
            let end = std::cmp::min(start + group_size, k);
            let group_vals = &row_data[start..end];

            // Pad to full group_size if needed
            let mut padded = group_vals.to_vec();
            while padded.len() < group_size {
                padded.push(0.0);
            }

            let (packed, scale, bias) = match bits {
                4 => quantize_4bit_group(&padded),
                6 => quantize_6bit_group(&padded),
                8 => quantize_8bit_group(&padded),
                _ => panic!("unsupported bits"),
            };

            all_packed.extend_from_slice(&packed);
            all_scales.extend_from_slice(&f32_to_bf16_bytes(scale));
            all_biases.extend_from_slice(&f32_to_bf16_bytes(bias));
        }
    }

    QuantizedWeightData {
        weight_packed: all_packed,
        scales_raw: all_scales,
        biases_raw: all_biases,
    }
}

// --------------------------------------------------------------------------
// Test harness
// --------------------------------------------------------------------------

fn run_quantized_matmul_test(m: u32, n: u32, k: u32, bits: u32, group_size: u32, test_name: &str) {
    let device = MlxDevice::new().expect("MlxDevice::new should succeed");
    let mut registry = KernelRegistry::new();

    // Generate random weights [N, K] and input [M, K]
    let weight_floats = pseudo_random_f32(42, n as usize * k as usize);
    let input_data = pseudo_random_f32(137, m as usize * k as usize);

    // Quantize weights
    let qdata = quantize_weight_matrix(
        &weight_floats,
        n as usize,
        k as usize,
        group_size as usize,
        bits as usize,
    );

    // CPU reference
    let expected = cpu_quantized_matmul(
        &input_data,
        &qdata.weight_packed,
        &qdata.scales_raw,
        &qdata.biases_raw,
        m as usize,
        k as usize,
        n as usize,
        group_size as usize,
        bits as usize,
    );

    // Allocate GPU buffers
    let input_bytes = m as usize * k as usize * 4;
    let mut input_buf = device
        .alloc_buffer(input_bytes, DType::F32, vec![m as usize, k as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(qdata.weight_packed.len(), DType::U8, vec![qdata.weight_packed.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("write weight")
        .copy_from_slice(&qdata.weight_packed);

    let mut scales_buf = device
        .alloc_buffer(qdata.scales_raw.len(), DType::U8, vec![qdata.scales_raw.len()])
        .expect("alloc scales");
    scales_buf
        .as_mut_slice::<u8>()
        .expect("write scales")
        .copy_from_slice(&qdata.scales_raw);

    let mut biases_buf = device
        .alloc_buffer(qdata.biases_raw.len(), DType::U8, vec![qdata.biases_raw.len()])
        .expect("alloc biases");
    biases_buf
        .as_mut_slice::<u8>()
        .expect("write biases")
        .copy_from_slice(&qdata.biases_raw);

    let params = QuantizedMatmulParams {
        m,
        k,
        n,
        group_size,
        bits,
    };

    // Dispatch
    let mut encoder = device.command_encoder().expect("encoder");
    let output_buf = mlx_native::quantized_matmul(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &scales_buf,
        &biases_buf,
        &params,
    )
    .expect("quantized_matmul dispatch");

    encoder.commit_and_wait().expect("commit_and_wait");

    // Compare
    let actual: Vec<f32> = output_buf.as_slice::<f32>().expect("read output").to_vec();
    assert_eq!(
        actual.len(),
        expected.len(),
        "{test_name}: output length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
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

    // Tolerance: the GPU accumulates bf16-dequantized products in f32 over K
    // elements.  With K=4096 and bf16's ~0.5% relative error, the accumulated
    // absolute error can reach ~0.1.  Use a tolerance proportional to K.
    let tol = 0.1 * (k as f32 / 4096.0).max(1.0);
    assert!(
        max_diff <= tol,
        "{test_name}: max|delta| = {max_diff} at index {max_diff_idx} exceeds tolerance {tol} \
         (actual={}, expected={})",
        actual[max_diff_idx],
        expected[max_diff_idx]
    );
}

// --------------------------------------------------------------------------
// 4-bit tests
// --------------------------------------------------------------------------

#[test]
fn test_qmatmul_q4_m1_n4096_k4096() {
    run_quantized_matmul_test(1, 4096, 4096, 4, 32, "Q4 M=1 N=4096 K=4096");
}

#[test]
fn test_qmatmul_q4_m1_n14336_k4096() {
    run_quantized_matmul_test(1, 14336, 4096, 4, 32, "Q4 M=1 N=14336 K=4096");
}

#[test]
fn test_qmatmul_q4_m8_n4096_k4096() {
    run_quantized_matmul_test(8, 4096, 4096, 4, 32, "Q4 M=8 N=4096 K=4096");
}

// --------------------------------------------------------------------------
// 6-bit tests
// --------------------------------------------------------------------------

#[test]
fn test_qmatmul_q6_m1_n4096_k4096() {
    run_quantized_matmul_test(1, 4096, 4096, 6, 32, "Q6 M=1 N=4096 K=4096");
}

#[test]
fn test_qmatmul_q6_m1_n14336_k4096() {
    run_quantized_matmul_test(1, 14336, 4096, 6, 32, "Q6 M=1 N=14336 K=4096");
}

#[test]
fn test_qmatmul_q6_m8_n4096_k4096() {
    run_quantized_matmul_test(8, 4096, 4096, 6, 32, "Q6 M=8 N=4096 K=4096");
}

// --------------------------------------------------------------------------
// 8-bit tests
// --------------------------------------------------------------------------

#[test]
fn test_qmatmul_q8_m1_n4096_k4096() {
    run_quantized_matmul_test(1, 4096, 4096, 8, 32, "Q8 M=1 N=4096 K=4096");
}

#[test]
fn test_qmatmul_q8_m1_n14336_k4096() {
    run_quantized_matmul_test(1, 14336, 4096, 8, 32, "Q8 M=1 N=14336 K=4096");
}

#[test]
fn test_qmatmul_q8_m8_n4096_k4096() {
    run_quantized_matmul_test(8, 4096, 4096, 8, 32, "Q8 M=8 N=4096 K=4096");
}

// --------------------------------------------------------------------------
// SIMD path tests (quantized_matmul_simd)
// --------------------------------------------------------------------------

fn run_quantized_matmul_simd_test(m: u32, n: u32, k: u32, bits: u32, group_size: u32, test_name: &str) {
    let device = MlxDevice::new().expect("MlxDevice::new should succeed");
    let mut registry = KernelRegistry::new();

    let weight_floats = pseudo_random_f32(42, n as usize * k as usize);
    let input_data = pseudo_random_f32(137, m as usize * k as usize);

    let qdata = quantize_weight_matrix(
        &weight_floats,
        n as usize,
        k as usize,
        group_size as usize,
        bits as usize,
    );

    let expected = cpu_quantized_matmul(
        &input_data,
        &qdata.weight_packed,
        &qdata.scales_raw,
        &qdata.biases_raw,
        m as usize,
        k as usize,
        n as usize,
        group_size as usize,
        bits as usize,
    );

    let input_bytes = m as usize * k as usize * 4;
    let mut input_buf = device
        .alloc_buffer(input_bytes, DType::F32, vec![m as usize, k as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(qdata.weight_packed.len(), DType::U8, vec![qdata.weight_packed.len()])
        .expect("alloc weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("write weight")
        .copy_from_slice(&qdata.weight_packed);

    let mut scales_buf = device
        .alloc_buffer(qdata.scales_raw.len(), DType::U8, vec![qdata.scales_raw.len()])
        .expect("alloc scales");
    scales_buf
        .as_mut_slice::<u8>()
        .expect("write scales")
        .copy_from_slice(&qdata.scales_raw);

    let mut biases_buf = device
        .alloc_buffer(qdata.biases_raw.len(), DType::U8, vec![qdata.biases_raw.len()])
        .expect("alloc biases");
    biases_buf
        .as_mut_slice::<u8>()
        .expect("write biases")
        .copy_from_slice(&qdata.biases_raw);

    let params = QuantizedMatmulParams {
        m,
        k,
        n,
        group_size,
        bits,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    let output_buf = mlx_native::quantized_matmul_simd(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &scales_buf,
        &biases_buf,
        &params,
    )
    .expect("quantized_matmul_simd dispatch");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual: Vec<f32> = output_buf.as_slice::<f32>().expect("read output").to_vec();
    assert_eq!(actual.len(), expected.len());

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
        "{test_name} (SIMD): max|delta| = {max_diff} at index {max_diff_idx} \
         (actual={}, expected={})",
        actual[max_diff_idx], expected[max_diff_idx]
    );

    // SIMD path may have slightly different accumulation order; use 5e-3 tolerance
    assert!(
        max_diff <= 5e-3,
        "{test_name} (SIMD): max|delta| = {max_diff} at index {max_diff_idx} exceeds tolerance 5e-3 \
         (actual={}, expected={})",
        actual[max_diff_idx],
        expected[max_diff_idx]
    );
}

#[test]
fn test_qmatmul_simd_q4_m1_n4096_k4096() {
    run_quantized_matmul_simd_test(1, 4096, 4096, 4, 32, "Q4 M=1 N=4096 K=4096");
}

#[test]
fn test_qmatmul_simd_q8_m1_n4096_k4096() {
    run_quantized_matmul_simd_test(1, 4096, 4096, 8, 32, "Q8 M=1 N=4096 K=4096");
}

// --------------------------------------------------------------------------
// Edge cases / validation tests
// --------------------------------------------------------------------------

#[test]
fn test_qmatmul_unsupported_bits() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("buf");

    let params = QuantizedMatmulParams {
        m: 1,
        k: 1,
        n: 1,
        group_size: 1,
        bits: 3, // unsupported
    };

    let mut encoder = device.command_encoder().expect("encoder");
    let result = mlx_native::quantized_matmul(
        &mut encoder,
        &mut registry,
        &device,
        &buf,
        &buf,
        &buf,
        &buf,
        &params,
    );

    assert!(result.is_err(), "Should error on unsupported bits=3");
}

#[test]
fn test_qmatmul_zero_dimensions() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("buf");

    let params = QuantizedMatmulParams {
        m: 0,
        k: 4,
        n: 4,
        group_size: 4,
        bits: 4,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    let result = mlx_native::quantized_matmul(
        &mut encoder,
        &mut registry,
        &device,
        &buf,
        &buf,
        &buf,
        &buf,
        &params,
    );

    assert!(result.is_err(), "Should error on M=0");
}
