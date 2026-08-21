//! Focused Q3_K runtime parity tests.
//!
//! The fixture packer is the inverse of the canonical
//! `dequantize_row_q3_K` formula. It deliberately exercises every scale
//! field, hmask plane, and packed-quant shift without depending on an
//! external converter at test time.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    DType, GgmlIdMmDispatchParams, GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams,
    GgmlType, KernelRegistry, MlxDevice,
};

const QK: usize = 256;
const BLOCK_BYTES: usize = 110;

fn make_q3_block(seed: usize) -> ([u8; BLOCK_BYTES], [f32; QK]) {
    let mut packed = [0u8; BLOCK_BYTES];
    let mut expected = [0.0f32; QK];
    let d = half::f16::from_f32(0.03125);
    packed[108..110].copy_from_slice(&d.to_le_bytes());
    let d = d.to_f32();

    for group in 0..16 {
        // Covers negative, zero, and positive centered 6-bit scales.
        let encoded_scale = ((group * 11 + seed * 7) % 64) as u8;
        let low = encoded_scale & 0x0f;
        let high = encoded_scale >> 4;
        if group < 8 {
            packed[96 + group] |= low;
        } else {
            packed[96 + group - 8] |= low << 4;
        }
        packed[104 + group % 4] |= high << (2 * (group / 4));

        let half = group / 8;
        let group_in_half = group % 8;
        let q_offset = 32 + half * 32 + (group_in_half % 2) * 16;
        let shift = 2 * (group_in_half / 2);
        let hmask = 1u8 << (group / 2);
        let scale = encoded_scale as i32 - 32;

        for lane in 0..16 {
            let quant = ((group * 5 + lane * 3 + seed) % 8) as i8 - 4;
            let low2 = if quant < 0 { quant + 4 } else { quant } as u8;
            packed[q_offset + lane] |= low2 << shift;
            if quant >= 0 {
                packed[(group_in_half % 2) * 16 + lane] |= hmask;
            }
            expected[group * 16 + lane] = d * scale as f32 * quant as f32;
        }
    }

    (packed, expected)
}

fn make_weights(rows: usize, k: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % QK, 0);
    let mut packed = Vec::with_capacity(rows * (k / QK) * BLOCK_BYTES);
    let mut dequant = Vec::with_capacity(rows * k);
    for row in 0..rows {
        for block in 0..k / QK {
            let (bytes, values) = make_q3_block(seed + row * 17 + block * 29);
            packed.extend_from_slice(&bytes);
            dequant.extend_from_slice(&values);
        }
    }
    (packed, dequant)
}

fn input_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 13) % 257) as f32 - 128.0) / 257.0)
        .collect()
}

fn cpu_matmul(input: &[f32], weights: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for im in 0..m {
        for row in 0..n {
            let mut sum = 0.0f32;
            for col in 0..k {
                sum += input[im * k + col] * weights[row * k + col];
            }
            output[im * n + row] = sum;
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32], abs: f32, rel: f32, label: &str) {
    assert_eq!(actual.len(), expected.len());
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let err = (got - want).abs();
        if err > worst.3 {
            worst = (i, got, want, err);
        }
        let limit = abs + rel * want.abs();
        assert!(
            err <= limit,
            "{label}: index {i}, got {got}, want {want}, err {err}, limit {limit}"
        );
    }
    eprintln!(
        "{label}: max_abs={} at {} (got={}, expected={})",
        worst.3, worst.0, worst.1, worst.2
    );
}

fn upload_u8(device: &MlxDevice, data: &[u8]) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(data.len(), DType::U8, vec![data.len()])
        .unwrap();
    buffer.as_mut_slice::<u8>().unwrap().copy_from_slice(data);
    buffer
}

fn upload_f32(device: &MlxDevice, data: &[f32], shape: Vec<usize>) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(std::mem::size_of_val(data), DType::F32, shape)
        .unwrap();
    buffer.as_mut_slice::<f32>().unwrap().copy_from_slice(data);
    buffer
}

#[test]
fn q3_k_type_size_and_cpu_dequant_match_reference_formula() {
    assert_eq!(GgmlType::Q3_K.block_values(), 256);
    assert_eq!(GgmlType::Q3_K.block_bytes(), 110);
    assert_eq!(
        mlx_native::gguf::test_only_ggml_type_from_u32(11).unwrap(),
        GgmlType::Q3_K
    );
    assert_eq!(
        mlx_native::gguf::test_only_compute_byte_len(&[3, 512], GgmlType::Q3_K).unwrap(),
        660
    );

    let (packed0, expected0) = make_q3_block(3);
    let (packed1, expected1) = make_q3_block(41);
    let mut packed = packed0.to_vec();
    packed.extend_from_slice(&packed1);
    let mut expected = expected0.to_vec();
    expected.extend_from_slice(&expected1);
    let mut actual = vec![0.0f32; 512];
    mlx_native::gguf::test_only_dequantize(&packed, GgmlType::Q3_K, &mut actual).unwrap();
    assert_eq!(actual, expected);

    let malformed = &packed[..packed.len() - 1];
    assert!(
        mlx_native::gguf::test_only_dequantize(malformed, GgmlType::Q3_K, &mut actual).is_err()
    );
}

#[test]
fn q3_k_dense_mv_simd_mm_and_tensor_mm_match_cpu() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (n, k, m) = (67usize, 512usize, 17usize);
    let (packed, weights) = make_weights(n, k, 11);
    let input = input_values(m * k, 23);
    let expected = cpu_matmul(&input, &weights, m, n, k);
    let weight_buf = upload_u8(&device, &packed);

    // Dedicated packed-dot MV, including a non-multiple-of-four N tail.
    let input_mv = upload_f32(&device, &input[..k], vec![1, k]);
    let output_mv = upload_f32(&device, &vec![0.0; n], vec![1, n]);
    let mv_params = GgmlQuantizedMatmulParams {
        m: 1,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q3_K,
    };
    let mut encoder = device.command_encoder().unwrap();
    mlx_native::quantized_matmul_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input_mv,
        &weight_buf,
        &output_mv,
        &mv_params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(
        output_mv.as_slice::<f32>().unwrap(),
        &expected[..n],
        2.0e-4,
        2.0e-5,
        "Q3_K dense MV",
    );

    let input_mm = upload_f32(&device, &input, vec![m, k]);
    let mm_params = GgmlQuantizedMatmulParams {
        m: m as u32,
        ..mv_params
    };

    // Explicit non-tensor simdgroup MMA fallback.
    let output_simd = upload_f32(&device, &vec![0.0; m * n], vec![m, n]);
    let mut encoder = device.command_encoder().unwrap();
    mlx_native::ops::quantized_matmul_ggml::dispatch_mm_simd_for_test(
        &mut encoder,
        &mut registry,
        &device,
        &input_mm,
        &weight_buf,
        &output_simd,
        &mm_params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(
        output_simd.as_slice::<f32>().unwrap(),
        &expected,
        3.0e-3,
        2.0e-4,
        "Q3_K dense simd MM",
    );

    // Production MM selection: tensor V2 on supported Apple GPUs, with the
    // existing capability fallback to tensor V1 or simdgroup MMA.
    let output_auto = upload_f32(&device, &vec![0.0; m * n], vec![m, n]);
    let mut encoder = device.command_encoder().unwrap();
    mlx_native::dispatch_mm_for_test(
        &mut encoder,
        &mut registry,
        &device,
        &input_mm,
        &weight_buf,
        &output_auto,
        &mm_params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(
        output_auto.as_slice::<f32>().unwrap(),
        &expected,
        3.0e-2,
        2.0e-3,
        "Q3_K dense tensor/auto MM",
    );
}

#[test]
fn q3_k_moe_mv_id_and_mm_id_match_cpu() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (n_experts, n, k) = (8usize, 67usize, 512usize);
    let mut packed = Vec::new();
    let mut weights = Vec::new();
    let mut expert_bytes = 0usize;
    for expert in 0..n_experts {
        let (p, w) = make_weights(n, k, 101 + expert * 43);
        expert_bytes = p.len();
        packed.extend_from_slice(&p);
        weights.extend_from_slice(&w);
    }
    let weight_buf = upload_u8(&device, &packed);

    let run = |n_tokens: usize, top_k: usize, force_mm: bool, registry: &mut KernelRegistry| {
        let input = input_values(n_tokens * k, 71 + n_tokens);
        let ids: Vec<u32> = (0..n_tokens)
            .flat_map(|token| {
                (0..top_k).map(move |slot| ((token * 2 + slot + 1) % n_experts) as u32)
            })
            .collect();
        let mut expected = vec![0.0f32; n_tokens * top_k * n];
        for (route, &expert) in ids.iter().enumerate() {
            let token = route / top_k;
            let got = cpu_matmul(
                &input[token * k..(token + 1) * k],
                &weights[expert as usize * n * k..(expert as usize + 1) * n * k],
                1,
                n,
                k,
            );
            expected[route * n..(route + 1) * n].copy_from_slice(&got);
        }

        let input_buf = upload_f32(&device, &input, vec![n_tokens, k]);
        let mut ids_buf = device
            .alloc_buffer(ids.len() * 4, DType::U32, vec![ids.len()])
            .unwrap();
        ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);
        let output = upload_f32(
            &device,
            &vec![0.0; n_tokens * top_k * n],
            vec![n_tokens, top_k, n],
        );

        if force_mm {
            let params = GgmlIdMmDispatchParams {
                n_tokens: n_tokens as u32,
                top_k: top_k as u32,
                n: n as u32,
                k: k as u32,
                n_experts: n_experts as u32,
                expert_stride: expert_bytes as u64,
                ggml_type: GgmlType::Q3_K,
            };
            let htpe = device
                .alloc_buffer(params.htpe_bytes(), DType::U32, vec![n_experts])
                .unwrap();
            let hids = device
                .alloc_buffer(params.hids_bytes(), DType::U32, vec![n_experts, n_tokens])
                .unwrap();
            let mut encoder = device.command_encoder().unwrap();
            mlx_native::dispatch_id_mm_for_test(
                &mut encoder,
                registry,
                &device,
                &input_buf,
                &weight_buf,
                &ids_buf,
                &htpe,
                &hids,
                &output,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            assert_close(
                output.as_slice::<f32>().unwrap(),
                &expected,
                3.0e-2,
                2.0e-3,
                "Q3_K MoE MM_ID",
            );
        } else {
            let params = GgmlQuantizedMatmulIdParams {
                n_tokens: n_tokens as u32,
                top_k: top_k as u32,
                n: n as u32,
                k: k as u32,
                n_experts: n_experts as u32,
                expert_stride: expert_bytes as u64,
                ggml_type: GgmlType::Q3_K,
            };
            let mut encoder = device.command_encoder().unwrap();
            mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
                &mut encoder,
                registry,
                &device,
                &input_buf,
                &weight_buf,
                &ids_buf,
                &output,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            assert_close(
                output.as_slice::<f32>().unwrap(),
                &expected,
                2.0e-4,
                2.0e-5,
                "Q3_K MoE MV_ID",
            );
        }
    };

    run(4, 1, false, &mut registry);
    run(4, 6, false, &mut registry);
    run(16, 1, true, &mut registry);
    run(16, 6, true, &mut registry);
}
