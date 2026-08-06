//! Tests for the expert-routed (MoE) quantized matmul GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::quantized_matmul_id::{quantized_matmul_id, QuantizedMatmulIdParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

/// Pack f32 values into 4-bit quantized format with bf16 scales and biases.
///
/// Returns (weight_bytes, scales_u16, biases_u16).
///
/// Each group of `group_size` values is affine-quantized to 4-bit [0, 15]:
///   quant_val = round((float_val - bias) / scale)
///   scale = (max - min) / 15
///   bias  = min
fn quantize_4bit(
    data: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
    let num_groups_per_row = (k + group_size - 1) / group_size;
    let packs_per_row = (k + 7) / 8; // 8 values per uint32

    let mut weight_bytes = vec![0u8; n * packs_per_row * 4];
    let mut scales = vec![0u16; n * num_groups_per_row];
    let mut biases = vec![0u16; n * num_groups_per_row];

    for row in 0..n {
        for g in 0..num_groups_per_row {
            let start = g * group_size;
            let end = (start + group_size).min(k);

            // Find min/max in this group.
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for i in start..end {
                let v = data[row * k + i];
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }

            let range = max_val - min_val;
            let scale = if range < 1e-10 { 1.0f32 } else { range / 15.0 };
            let bias = min_val;

            // Store scale and bias as bf16.
            scales[row * num_groups_per_row + g] = f32_to_bf16(scale);
            biases[row * num_groups_per_row + g] = f32_to_bf16(bias);

            // Quantize values.
            for i in start..end {
                let v = data[row * k + i];
                let qval = ((v - bias) / scale).round().clamp(0.0, 15.0) as u32;

                let pack_idx = i / 8;
                let in_pack = i % 8;
                let byte_offset = row * packs_per_row * 4 + pack_idx * 4;

                // Read existing packed uint32.
                let existing = u32::from_le_bytes([
                    weight_bytes[byte_offset],
                    weight_bytes[byte_offset + 1],
                    weight_bytes[byte_offset + 2],
                    weight_bytes[byte_offset + 3],
                ]);
                let updated = existing | (qval << (4 * in_pack as u32));
                let bytes = updated.to_le_bytes();
                weight_bytes[byte_offset] = bytes[0];
                weight_bytes[byte_offset + 1] = bytes[1];
                weight_bytes[byte_offset + 2] = bytes[2];
                weight_bytes[byte_offset + 3] = bytes[3];
            }
        }
    }

    (weight_bytes, scales, biases)
}

/// CPU reference: dequantize 4-bit weights and compute matmul for one expert.
fn dequant_matmul_ref(
    input_row: &[f32],
    weight_bytes: &[u8],
    scales: &[u16],
    biases: &[u16],
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<f32> {
    let num_groups_per_row = (k + group_size - 1) / group_size;
    let packs_per_row = (k + 7) / 8;
    let mut output = vec![0.0f32; n];

    for col in 0..n {
        let sb_base = col * num_groups_per_row;
        let mut acc = 0.0f64; // use f64 for reference precision

        for ki in 0..k {
            let g = ki / group_size;
            let scale = bf16_to_f32(scales[sb_base + g]);
            let bias = bf16_to_f32(biases[sb_base + g]);

            let pack_idx = ki / 8;
            let in_pack = ki % 8;
            let byte_offset = col * packs_per_row * 4 + pack_idx * 4;
            let packed = u32::from_le_bytes([
                weight_bytes[byte_offset],
                weight_bytes[byte_offset + 1],
                weight_bytes[byte_offset + 2],
                weight_bytes[byte_offset + 3],
            ]);
            let qval = (packed >> (4 * in_pack as u32)) & 0xF;

            // Dequantize using bf16 arithmetic to match the GPU kernel.
            let w_bf16 = f32_to_bf16(qval as f32);
            let w_dequant = bf16_to_f32(
                bf16_mul_add(w_bf16, f32_to_bf16(scale), f32_to_bf16(bias)),
            );

            let x_bf16 = bf16_to_f32(f32_to_bf16(input_row[ki]));
            acc += (w_dequant as f64) * (x_bf16 as f64);
        }
        output[col] = acc as f32;
    }
    output
}

// bf16 conversion helpers.
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round to nearest even.
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits.wrapping_add(round)) >> 16) as u16
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn bf16_mul_add(a: u16, b: u16, c: u16) -> u16 {
    let fa = bf16_to_f32(a);
    let fb = bf16_to_f32(b);
    let fc = bf16_to_f32(c);
    f32_to_bf16(fa * fb + fc)
}

/// Deterministic pseudo-random f32 values.
fn make_f32_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            ((z as i64 as f64) * (1.0 / (i64::MAX as f64)) * 0.5) as f32
        })
        .collect()
}

#[test]
fn test_quantized_matmul_id_4bit_4tokens_8experts_top2() {
    let (device, mut registry) = setup();

    let num_experts: usize = 8;
    let n: usize = 32; // output dim per expert (small for test)
    let k: usize = 64; // inner dim (must be multiple of 8 for 4-bit)
    let group_size: usize = 32;
    let n_tokens: usize = 4;
    let n_expert_used: usize = 2;
    let bits: u32 = 4;

    // Generate random expert weights and quantize each expert.
    let mut all_weight_bytes: Vec<u8> = Vec::new();
    let mut all_scales: Vec<u16> = Vec::new();
    let mut all_biases: Vec<u16> = Vec::new();

    let mut expert_wb: Vec<Vec<u8>> = Vec::new();
    let mut expert_sc: Vec<Vec<u16>> = Vec::new();
    let mut expert_bi: Vec<Vec<u16>> = Vec::new();

    for e in 0..num_experts {
        let float_data = make_f32_vec(n * k, 0xCAFE + (e as u64) * 0x1234);
        let (wb, sc, bi) = quantize_4bit(&float_data, n, k, group_size);
        all_weight_bytes.extend_from_slice(&wb);
        all_scales.extend_from_slice(&sc);
        all_biases.extend_from_slice(&bi);
        expert_wb.push(wb);
        expert_sc.push(sc);
        expert_bi.push(bi);
    }

    // Input tokens.
    let input_data = make_f32_vec(n_tokens * k, 0xBEEF);

    // Expert ids: deterministic routing.
    // token 0 -> experts [0, 3]
    // token 1 -> experts [1, 5]
    // token 2 -> experts [2, 7]
    // token 3 -> experts [4, 6]
    let ids_data: Vec<u32> = vec![0, 3, 1, 5, 2, 7, 4, 6];

    // --- CPU reference ---
    let mut ref_output = vec![0.0f32; n_tokens * n_expert_used * n];
    for t in 0..n_tokens {
        for s in 0..n_expert_used {
            let expert_id = ids_data[t * n_expert_used + s] as usize;
            let input_row = &input_data[t * k..(t + 1) * k];
            let expert_out = dequant_matmul_ref(
                input_row,
                &expert_wb[expert_id],
                &expert_sc[expert_id],
                &expert_bi[expert_id],
                n,
                k,
                group_size,
            );
            let base = (t * n_expert_used + s) * n;
            ref_output[base..base + n].copy_from_slice(&expert_out);
        }
    }

    // --- GPU dispatch ---
    let mut input_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![n_tokens, k])
        .expect("input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("w")
        .copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(all_weight_bytes.len(), DType::U8, vec![all_weight_bytes.len()])
        .expect("weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("w")
        .copy_from_slice(&all_weight_bytes);

    let mut scales_buf = device
        .alloc_buffer(all_scales.len() * 2, DType::U16, vec![all_scales.len()])
        .expect("scales");
    scales_buf
        .as_mut_slice::<u16>()
        .expect("w")
        .copy_from_slice(&all_scales);

    let mut biases_buf = device
        .alloc_buffer(all_biases.len() * 2, DType::U16, vec![all_biases.len()])
        .expect("biases");
    biases_buf
        .as_mut_slice::<u16>()
        .expect("w")
        .copy_from_slice(&all_biases);

    let mut ids_buf = device
        .alloc_buffer(ids_data.len() * 4, DType::U32, vec![n_tokens, n_expert_used])
        .expect("ids");
    ids_buf
        .as_mut_slice::<u32>()
        .expect("w")
        .copy_from_slice(&ids_data);

    let mut encoder = device.command_encoder().expect("encoder");
    let output_buf = quantized_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &scales_buf,
        &biases_buf,
        &ids_buf,
        &QuantizedMatmulIdParams {
            m: n_tokens as u32,
            k: k as u32,
            n: n as u32,
            group_size: group_size as u32,
            bits,
            n_expert_used: n_expert_used as u32,
            num_experts: num_experts as u32,
        },
    )
    .expect("quantized_matmul_id");
    encoder.commit_and_wait().expect("commit");

    let output: &[f32] = output_buf.as_slice().expect("read");
    assert_eq!(output.len(), ref_output.len());

    let mut max_diff: f32 = 0.0;
    let mut max_idx: usize = 0;
    for i in 0..output.len() {
        let diff = (output[i] - ref_output[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    eprintln!(
        "quantized_matmul_id Q4: max|delta|={:.3e} at idx {} (gpu={}, ref={})",
        max_diff, max_idx, output[max_idx], ref_output[max_idx]
    );
    assert!(
        max_diff <= 1e-4,
        "quantized_matmul_id Q4 exceeds tolerance 1e-4: max|delta|={}",
        max_diff
    );
}

// --- Validation tests ---

#[test]
fn test_quantized_matmul_id_zero_experts_error() {
    let (device, mut registry) = setup();

    let buf = device
        .alloc_buffer(256, DType::F32, vec![64])
        .expect("buf");
    let mut encoder = device.command_encoder().expect("enc");

    let result = quantized_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &buf,
        &buf,
        &buf,
        &buf,
        &buf,
        &QuantizedMatmulIdParams {
            m: 4,
            k: 64,
            n: 32,
            group_size: 32,
            bits: 4,
            n_expert_used: 2,
            num_experts: 0,
        },
    );
    assert!(result.is_err(), "zero num_experts should error");
}

#[test]
fn test_quantized_matmul_id_unsupported_bits_error() {
    let (device, mut registry) = setup();

    let buf = device
        .alloc_buffer(256, DType::F32, vec![64])
        .expect("buf");
    let mut encoder = device.command_encoder().expect("enc");

    let result = quantized_matmul_id(
        &mut encoder,
        &mut registry,
        &device,
        &buf,
        &buf,
        &buf,
        &buf,
        &buf,
        &QuantizedMatmulIdParams {
            m: 1,
            k: 64,
            n: 32,
            group_size: 32,
            bits: 3,
            n_expert_used: 1,
            num_experts: 1,
        },
    );
    assert!(result.is_err(), "bits=3 should error");
}
