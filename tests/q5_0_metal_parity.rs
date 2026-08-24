#![cfg(target_vendor = "apple")]

use half::f16;
use mlx_native::{
    embedding_gather_q5_0, mul_mv_ext_dispatch, quantized_matmul_ggml,
    quantized_matmul_ggml_batched_mm, quantized_matmul_ggml_batched_mm_strided_input,
    quantized_matmul_ggml_batched_mv, quantized_matmul_ggml_with_policy, quantized_matmul_id_ggml,
    quantized_matmul_id_ggml_mv_with_policy, quantized_matmul_id_ggml_pooled,
    quantized_matmul_id_ggml_pooled_pair, quantized_matmul_id_ggml_pooled_pair_with_policy,
    quantized_matmul_id_ggml_pooled_slotted, quantized_matmul_id_ggml_with_policy,
    quantized_matmul_id_swiglu_q4_0, quantized_matmul_mm_tensor_perm021, DType,
    EmbeddingQ5_0Params, GgmlBatchedQuantizedMatmulInputStrides, GgmlBatchedQuantizedMatmulParams,
    GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlQuantizedMatmulPerm021Params,
    GgmlRoutingPolicy, GgmlTensorMmPreference, GgmlType, GgufFile, IdMmScratch, KernelRegistry,
    MlxBuffer, MlxDevice, MulMvExtParams,
};

const QK5_0: usize = 32;
const BLOCK_BYTES: usize = 22;
const K: usize = 64;
const N: usize = 17;
const N_EXPERTS: usize = 8;
const TOP_K: usize = 6;
const EXPERT_PADDING: usize = 10;

fn block(seed: usize) -> [u8; BLOCK_BYTES] {
    let mut bytes = [0u8; BLOCK_BYTES];
    let scale = [0.125, -0.25, 0.5, -1.0][seed % 4];
    bytes[..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    let qh = (0..32).fold(0u32, |mask, bit| {
        mask | ((((seed * 13 + bit * 7 + 3) >> 2) as u32 & 1) << bit)
    });
    bytes[2..6].copy_from_slice(&qh.to_le_bytes());
    for lane in 0..16 {
        let low = ((seed * 5 + lane * 11 + 1) & 0x0f) as u8;
        let high = ((seed * 17 + lane * 3 + 9) & 0x0f) as u8;
        bytes[6 + lane] = low | (high << 4);
    }
    bytes
}

fn matrix_bytes(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(N * K / QK5_0 * BLOCK_BYTES);
    for row in 0..N {
        for block_index in 0..K / QK5_0 {
            bytes.extend_from_slice(&block(seed + row * 19 + block_index));
        }
    }
    bytes
}

fn embedding_row_bytes(seed: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(K / QK5_0 * BLOCK_BYTES);
    for block_index in 0..K / QK5_0 {
        bytes.extend_from_slice(&block(seed + block_index));
    }
    bytes
}

fn stacked_expert_bytes() -> Vec<u8> {
    let per_expert = N * K / QK5_0 * BLOCK_BYTES;
    let stride = per_expert + EXPERT_PADDING;
    let mut bytes = vec![0xA5; (N_EXPERTS - 1) * stride + per_expert];
    for expert in 0..N_EXPERTS {
        let start = expert * stride;
        bytes[start..start + per_expert].copy_from_slice(&matrix_bytes(101 + expert * 997));
    }
    bytes
}

fn input_rows(rows: usize, salt: usize) -> Vec<f32> {
    (0..rows * K)
        .map(|index| {
            let value = ((index * 29 + salt * 37 + index / K * 11) % 97) as f32;
            (value - 48.0) / 23.0
        })
        .collect()
}

fn cpu_matrix(weight: &[u8]) -> Vec<f32> {
    let row_bytes = K / QK5_0 * BLOCK_BYTES;
    let mut decoded = vec![0.0; N * K];
    for row in 0..N {
        mlx_native::gguf::test_only_dequantize(
            &weight[row * row_bytes..(row + 1) * row_bytes],
            GgmlType::Q5_0,
            &mut decoded[row * K..(row + 1) * K],
        )
        .unwrap();
    }
    decoded
}

fn cpu_dense(input: &[f32], weight: &[u8], rows: usize) -> Vec<f32> {
    let decoded = cpu_matrix(weight);
    let mut output = vec![0.0; rows * N];
    for m in 0..rows {
        for n in 0..N {
            let mut sum = 0.0;
            for k in 0..K {
                sum += input[m * K + k] * decoded[n * K + k];
            }
            output[m * N + n] = sum;
        }
    }
    output
}

fn u8_buffer(device: &MlxDevice, bytes: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .unwrap();
    buffer.as_mut_slice::<u8>().unwrap().copy_from_slice(bytes);
    buffer
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .unwrap();
    buffer
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(values);
    buffer
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .unwrap();
    buffer
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(values);
    buffer
}

fn empty_buffer(device: &MlxDevice, byte_len: usize, dtype: DType) -> MlxBuffer {
    device
        .alloc_buffer(byte_len, dtype, vec![byte_len / dtype.size_of()])
        .unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "{context} value {index}: got {got}, expected {want}, tolerance {tolerance}"
        );
    }
}

fn assert_all_nan(values: &[f32], context: &str) {
    for (index, value) in values.iter().enumerate() {
        assert!(value.is_nan(), "{context} value {index} was {value}");
    }
}

#[test]
fn dense_q5_0_executes_native_bytes_at_every_scheduler_width() {
    let device = MlxDevice::new().unwrap();
    let weight_bytes = matrix_bytes(7);
    let weight = u8_buffer(&device, &weight_bytes);
    let mut registry = KernelRegistry::new();
    for m in [1usize, 2, 8, 9, 33, 129] {
        let input_values = input_rows(m, m);
        let input = f32_buffer(&device, &input_values);
        let output = device
            .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
            .unwrap();
        let mut encoder = device.command_encoder().unwrap();
        quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weight,
            &output,
            &GgmlQuantizedMatmulParams {
                m: m as u32,
                n: N as u32,
                k: K as u32,
                ggml_type: GgmlType::Q5_0,
            },
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_close(
            output.as_slice::<f32>().unwrap(),
            &cpu_dense(&input_values, &weight_bytes, m),
            &format!("dense M={m}"),
        );
    }
}

#[test]
fn q5_0_dispatchers_reject_representation_substitution_before_encoding() {
    let device = MlxDevice::new().unwrap();
    let input = f32_buffer(&device, &input_rows(1, 13));
    let output = empty_buffer(&device, N * 4, DType::F32);
    let native_weight_bytes = matrix_bytes(17).len();
    let native_weight = u8_buffer(&device, &matrix_bytes(17));
    let f16_labeled_weight = empty_buffer(&device, native_weight_bytes, DType::F16);
    let params = GgmlQuantizedMatmulParams {
        m: 1,
        n: N as u32,
        k: K as u32,
        ggml_type: GgmlType::Q5_0,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let error = quantized_matmul_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &f16_labeled_weight,
        &output,
        &params,
    )
    .expect_err("Q5_0 weights must remain native U8 GGUF blocks");
    assert!(
        error.to_string().contains("native U8 GGUF blocks"),
        "{error}"
    );
    assert!(encoder.take_capture().unwrap().is_empty());

    for (case, wrong_input, wrong_output) in [
        (
            "dense input",
            empty_buffer(&device, K * 4, DType::F16),
            empty_buffer(&device, N * 4, DType::F32),
        ),
        (
            "dense output",
            f32_buffer(&device, &input_rows(1, 29)),
            empty_buffer(&device, N * 4, DType::F16),
        ),
    ] {
        let mut encoder = device.command_encoder().unwrap();
        encoder.start_capture();
        let error = quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &wrong_input,
            &native_weight,
            &wrong_output,
            &params,
        )
        .expect_err(case);
        assert!(error.to_string().contains("F32 input"), "{case}: {error}");
        assert!(encoder.take_capture().unwrap().is_empty());
    }

    let expert_weight = u8_buffer(&device, &stacked_expert_bytes());
    let wrong_ids = empty_buffer(&device, TOP_K * 4, DType::F32);
    let expert_output = empty_buffer(&device, TOP_K * N * 4, DType::F32);
    let expert_params = GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: TOP_K as u32,
        n: N as u32,
        k: K as u32,
        n_experts: N_EXPERTS as u32,
        expert_stride: (N * K / QK5_0 * BLOCK_BYTES + EXPERT_PADDING) as u64,
        ggml_type: GgmlType::Q5_0,
    };
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let error = quantized_matmul_id_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &expert_weight,
        &wrong_ids,
        &expert_output,
        &expert_params,
    )
    .expect_err("expert IDs must be U32");
    assert!(error.to_string().contains("U32 expert IDs"), "{error}");
    assert!(encoder.take_capture().unwrap().is_empty());

    let valid_mv_ids = u32_buffer(&device, &[0, 1, 2, 3, 4, 5]);
    let wrong_expert_output = empty_buffer(&device, TOP_K * N * 4, DType::F16);
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let error = quantized_matmul_id_ggml(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &expert_weight,
        &valid_mv_ids,
        &wrong_expert_output,
        &expert_params,
    )
    .expect_err("expert output must be F32");
    assert!(error.to_string().contains("F32 output"), "{error}");
    assert!(encoder.take_capture().unwrap().is_empty());

    let mut scratch = IdMmScratch::alloc(&device, N_EXPERTS as u32, 1).unwrap();
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let error = quantized_matmul_id_ggml_pooled(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &expert_weight,
        &wrong_ids,
        &expert_output,
        &mut scratch,
        &expert_params,
    )
    .expect_err("pooled expert IDs must be U32");
    assert!(error.to_string().contains("U32 expert IDs"), "{error}");
    assert!(encoder.take_capture().unwrap().is_empty());

    let routed_input = f32_buffer(&device, &input_rows(TOP_K, 19));
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let error = quantized_matmul_id_swiglu_q4_0(
        &mut encoder,
        &mut registry,
        &device,
        &routed_input,
        &routed_input,
        &expert_weight,
        &wrong_ids,
        &expert_output,
        &expert_params,
    )
    .expect_err("fused expert IDs must be U32");
    assert!(error.to_string().contains("U32 expert IDs"), "{error}");
    assert!(encoder.take_capture().unwrap().is_empty());

    let valid_ids = u32_buffer(&device, &[0, 1, 2, 3, 4, 5]);
    let mut pair_scratch = IdMmScratch::alloc(&device, N_EXPERTS as u32, 33).unwrap();
    let pair_params = GgmlQuantizedMatmulIdParams {
        n_tokens: 33,
        ..expert_params
    };
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let error = quantized_matmul_id_ggml_pooled_pair(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &expert_weight,
        &f16_labeled_weight,
        &valid_ids,
        &expert_output,
        &expert_output,
        &mut pair_scratch,
        &pair_params,
    )
    .expect_err("both paired expert weights must be native U8 blocks");
    assert!(
        error.to_string().contains("native U8 GGUF blocks"),
        "{error}"
    );
    assert!(encoder.take_capture().unwrap().is_empty());
}

#[test]
fn q5_0_width_amortized_route_preserves_results() {
    let device = MlxDevice::new().unwrap();
    let weight_bytes = matrix_bytes(31);
    let weight = u8_buffer(&device, &weight_bytes);
    for m in [2usize, 8] {
        let input_values = input_rows(m, 91 + m);
        let input = f32_buffer(&device, &input_values);
        let output = device
            .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
            .unwrap();
        let mut routing = GgmlRoutingPolicy::default();
        routing.dense_decode_mvn = false;
        routing.dense_decode_mv_ext = true;
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().unwrap();
        quantized_matmul_ggml_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weight,
            &output,
            &GgmlQuantizedMatmulParams {
                m: m as u32,
                n: N as u32,
                k: K as u32,
                ggml_type: GgmlType::Q5_0,
            },
            &routing,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_close(
            output.as_slice::<f32>().unwrap(),
            &cpu_dense(&input_values, &weight_bytes, m),
            &format!("mv_ext M={m}"),
        );
    }
}

#[test]
fn q5_0_width_amortized_batch_broadcasts_one_native_weight() {
    let device = MlxDevice::new().unwrap();
    let weight_bytes = matrix_bytes(53);
    let weight = u8_buffer(&device, &weight_bytes);
    let batch = 3usize;
    let m = 2usize;
    let mut input_values = Vec::new();
    let mut expected = Vec::new();
    for batch_index in 0..batch {
        let rows = input_rows(m, 131 + batch_index);
        expected.extend(cpu_dense(&rows, &weight_bytes, m));
        input_values.extend(rows);
    }
    let input = f32_buffer(&device, &input_values);
    let output = empty_buffer(&device, batch * m * N * 4, DType::F32);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    mul_mv_ext_dispatch(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &input,
        &output,
        &MulMvExtParams {
            m: m as u32,
            n: N as u32,
            k: K as u32,
            batch: batch as u32,
            ggml_type: GgmlType::Q5_0,
        },
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(
        output.as_slice::<f32>().unwrap(),
        &expected,
        "mv_ext broadcast batch",
    );
}

#[test]
fn q5_0_simd_mm_fallback_matches_cpu() {
    let device = MlxDevice::new().unwrap();
    let weight_bytes = matrix_bytes(211);
    let weight = u8_buffer(&device, &weight_bytes);

    let m = 33usize;
    let input_values = input_rows(m, 223);
    let input = f32_buffer(&device, &input_values);
    let output = device
        .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
        .unwrap();
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &GgmlQuantizedMatmulParams {
            m: m as u32,
            n: N as u32,
            k: K as u32,
            ggml_type: GgmlType::Q5_0,
        },
        &GgmlRoutingPolicy {
            dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
            allow_dense_large_tile_mm: false,
            ..GgmlRoutingPolicy::default()
        },
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(
        output.as_slice::<f32>().unwrap(),
        &cpu_dense(&input_values, &weight_bytes, m),
        "forced simd MM",
    );
}

#[cfg(mlx_native_has_metal_tensor_artifact)]
#[test]
fn q5_0_bf16_perm021_matches_cpu() {
    let device = MlxDevice::new().unwrap();
    let weight_bytes = matrix_bytes(211);
    let weight = u8_buffer(&device, &weight_bytes);
    let m = 9usize;
    let n_heads = 2usize;
    let head_dim = K / n_heads;
    let physical_f32 = input_rows(m, 227);
    let mut physical_bf16 = device
        .alloc_buffer(m * K * 2, DType::BF16, vec![n_heads, m, head_dim])
        .unwrap();
    let mut logical_f32 = vec![0.0; m * K];
    {
        let stored = physical_bf16.as_mut_slice::<u16>().unwrap();
        for head in 0..n_heads {
            for token in 0..m {
                for column in 0..head_dim {
                    let physical_index = head * m * head_dim + token * head_dim + column;
                    let source_index = token * K + head * head_dim + column;
                    let bits = half::bf16::from_f32(physical_f32[source_index]).to_bits();
                    stored[physical_index] = bits;
                    logical_f32[source_index] = half::bf16::from_bits(bits).to_f32();
                }
            }
        }
    }
    let permuted_output = device
        .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
        .unwrap();
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    quantized_matmul_mm_tensor_perm021(
        &mut encoder,
        &mut registry,
        &device,
        &physical_bf16,
        &weight,
        &permuted_output,
        &GgmlQuantizedMatmulPerm021Params {
            m: m as u32,
            n: N as u32,
            k: K as u32,
            head_dim: head_dim as u32,
            ggml_type: GgmlType::Q5_0,
        },
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(
        permuted_output.as_slice::<f32>().unwrap(),
        &cpu_dense(&logical_f32, &weight_bytes, m),
        "BF16 perm021",
    );
}

#[test]
fn q5_0_independent_batched_mv_mm_and_strided_mm_match_cpu() {
    let device = MlxDevice::new().unwrap();
    let batch = 3usize;
    let weights: Vec<Vec<u8>> = (0..batch).map(|b| matrix_bytes(401 + b * 101)).collect();
    let weight_bytes: Vec<u8> = weights.iter().flatten().copied().collect();
    let weight = u8_buffer(&device, &weight_bytes);

    for m in [2usize, 9] {
        let mut contiguous = Vec::new();
        let mut expected = Vec::new();
        for b in 0..batch {
            let rows = input_rows(m, 701 + b * 17 + m);
            expected.extend(cpu_dense(&rows, &weights[b], m));
            contiguous.extend(rows);
        }
        let input = f32_buffer(&device, &contiguous);
        let output = device
            .alloc_buffer(batch * m * N * 4, DType::F32, vec![batch, m, N])
            .unwrap();
        let params = GgmlBatchedQuantizedMatmulParams {
            batch: batch as u32,
            m: m as u32,
            n: N as u32,
            k: K as u32,
            ggml_type: GgmlType::Q5_0,
        };
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().unwrap();
        if m <= 8 {
            quantized_matmul_ggml_batched_mv(
                &mut encoder,
                &mut registry,
                &device,
                &input,
                &weight,
                &output,
                &params,
            )
            .unwrap();
        } else {
            quantized_matmul_ggml_batched_mm(
                &mut encoder,
                &mut registry,
                &device,
                &input,
                &weight,
                &output,
                &params,
            )
            .unwrap();
        }
        encoder.commit_and_wait().unwrap();
        assert_close(
            output.as_slice::<f32>().unwrap(),
            &expected,
            &format!("batched M={m}"),
        );
    }

    let m = 9usize;
    let row_stride = K + 8;
    let batch_stride = m * row_stride + 8;
    let mut strided = vec![f32::NAN; batch * batch_stride];
    let mut expected = Vec::new();
    for b in 0..batch {
        let rows = input_rows(m, 1_001 + b * 43);
        expected.extend(cpu_dense(&rows, &weights[b], m));
        for row in 0..m {
            let dst = b * batch_stride + row * row_stride;
            strided[dst..dst + K].copy_from_slice(&rows[row * K..(row + 1) * K]);
        }
    }
    let input = f32_buffer(&device, &strided);
    let output = device
        .alloc_buffer(batch * m * N * 4, DType::F32, vec![batch, m, N])
        .unwrap();
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    quantized_matmul_ggml_batched_mm_strided_input(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &GgmlBatchedQuantizedMatmulParams {
            batch: batch as u32,
            m: m as u32,
            n: N as u32,
            k: K as u32,
            ggml_type: GgmlType::Q5_0,
        },
        &GgmlBatchedQuantizedMatmulInputStrides {
            row_bytes: (row_stride * 4) as u64,
            batch_bytes: (batch_stride * 4) as u64,
        },
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(
        output.as_slice::<f32>().unwrap(),
        &expected,
        "strided batched MM",
    );
}

#[test]
fn q5_0_embedding_crosses_nibble_and_high_bit_boundaries() {
    let device = MlxDevice::new().unwrap();
    let vocab = 3usize;
    let weights: Vec<Vec<u8>> = (0..vocab)
        .map(|row| embedding_row_bytes(2_001 + row * 313))
        .collect();
    let weight_bytes: Vec<u8> = weights.iter().flatten().copied().collect();
    let weight = u8_buffer(&device, &weight_bytes);
    let ids_host = [2u32, 0, 1, 2];
    let mut ids = u32_buffer(&device, &ids_host);
    let output = device
        .alloc_buffer(ids_host.len() * K * 4, DType::F32, vec![ids_host.len(), K])
        .unwrap();
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    embedding_gather_q5_0(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &EmbeddingQ5_0Params {
            vocab_size: vocab,
            embed_dim: K,
            n_tokens: ids_host.len(),
        },
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let actual = output.as_slice::<f32>().unwrap();
    for (token_index, &id) in ids_host.iter().enumerate() {
        let mut expected = vec![0.0; K];
        mlx_native::gguf::test_only_dequantize(
            &weights[id as usize],
            GgmlType::Q5_0,
            &mut expected,
        )
        .unwrap();
        assert_close(
            &actual[token_index * K..(token_index + 1) * K],
            &expected,
            &format!("embedding token {token_index}"),
        );
    }

    ids.as_mut_slice::<u32>().unwrap()[0] = vocab as u32;
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    let error = embedding_gather_q5_0(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &EmbeddingQ5_0Params {
            vocab_size: vocab,
            embed_dim: K,
            n_tokens: ids_host.len(),
        },
    )
    .expect_err("invalid Q5_0 embedding ID must fail before encoding");
    assert!(error.to_string().contains("token_ids[0]=3"), "{error}");
    assert!(encoder.take_capture().unwrap().is_empty());
}

fn q5_0_embedding_fixture(weights: &[u8], vocab: usize) -> Vec<u8> {
    let name = "token_embd.weight";
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&1u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&(name.len() as u64).to_le_bytes());
    file.extend_from_slice(name.as_bytes());
    file.extend_from_slice(&2u32.to_le_bytes());
    file.extend_from_slice(&(K as u64).to_le_bytes());
    file.extend_from_slice(&(vocab as u64).to_le_bytes());
    file.extend_from_slice(&6u32.to_le_bytes());
    file.extend_from_slice(&32u64.to_le_bytes());
    while file.len() % 32 != 0 {
        file.push(0);
    }
    file.extend_from_slice(&[0u8; 32]);
    file.extend_from_slice(weights);
    file
}

#[test]
fn q5_0_mapped_embedding_offset_matches_owned_native_bytes() {
    let vocab = 3usize;
    let weights: Vec<u8> = (0..vocab)
        .flat_map(|row| embedding_row_bytes(2_501 + row * 113))
        .collect();
    let path = std::env::temp_dir().join(format!(
        "mlx_q5_0_embedding_{}_{}.gguf",
        std::process::id(),
        weights.len()
    ));
    std::fs::write(&path, q5_0_embedding_fixture(&weights, vocab)).unwrap();

    let device = MlxDevice::new().unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    let mapped = gguf
        .load_tensor_mapped("token_embd.weight", &device)
        .unwrap();
    let owned = gguf.load_tensor("token_embd.weight", &device).unwrap();
    assert!(mapped.is_file_backed());
    assert!(!owned.is_file_backed());
    assert_ne!(mapped.byte_offset(), 0);
    assert_eq!(mapped.dtype(), DType::U8);
    assert_eq!(mapped.data_byte_len(), weights.len());
    assert_eq!(owned.data_byte_len(), weights.len());

    let ids_host = [2u32, 0];
    let ids = u32_buffer(&device, &ids_host);
    let mapped_output = device
        .alloc_buffer(ids_host.len() * K * 4, DType::F32, vec![ids_host.len(), K])
        .unwrap();
    let owned_output = device
        .alloc_buffer(ids_host.len() * K * 4, DType::F32, vec![ids_host.len(), K])
        .unwrap();
    let params = EmbeddingQ5_0Params {
        vocab_size: vocab,
        embed_dim: K,
        n_tokens: ids_host.len(),
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    embedding_gather_q5_0(
        &mut encoder,
        &mut registry,
        &device,
        &mapped,
        &ids,
        &mapped_output,
        &params,
    )
    .unwrap();
    embedding_gather_q5_0(
        &mut encoder,
        &mut registry,
        &device,
        &owned,
        &ids,
        &owned_output,
        &params,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    for (index, (&mapped_value, &owned_value)) in mapped_output
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .zip(owned_output.as_slice::<f32>().unwrap())
        .enumerate()
    {
        assert_eq!(
            mapped_value.to_bits(),
            owned_value.to_bits(),
            "mapped/owned Q5_0 value {index}"
        );
    }

    std::fs::remove_file(path).unwrap();
}

fn expert_expected(
    input: &[f32],
    weights: &[u8],
    ids: &[u32],
    n_tokens: usize,
    slotted: bool,
) -> Vec<f32> {
    let expert_bytes = N * K / QK5_0 * BLOCK_BYTES;
    let expert_stride = expert_bytes + EXPERT_PADDING;
    let decoded: Vec<Vec<f32>> = (0..N_EXPERTS)
        .map(|expert| {
            let start = expert * expert_stride;
            cpu_matrix(&weights[start..start + expert_bytes])
        })
        .collect();
    let mut output = vec![0.0; n_tokens * TOP_K * N];
    for token in 0..n_tokens {
        for slot in 0..TOP_K {
            let row = token * TOP_K + slot;
            let input_row = if slotted { row } else { token };
            let expert = ids[row] as usize;
            for n in 0..N {
                let mut sum = 0.0;
                for k in 0..K {
                    sum += input[input_row * K + k] * decoded[expert][n * K + k];
                }
                output[row * N + n] = sum;
            }
        }
    }
    output
}

#[test]
fn q5_0_expert_mv_mm_shared_slotted_and_pair_use_native_blocks() {
    let device = MlxDevice::new().unwrap();
    let weight_bytes = stacked_expert_bytes();
    let weight = u8_buffer(&device, &weight_bytes);
    let expert_stride = (N * K / QK5_0 * BLOCK_BYTES + EXPERT_PADDING) as u64;

    for n_tokens in [1usize, 33] {
        let ids_host: Vec<u32> = (0..n_tokens)
            .flat_map(|token| (0..TOP_K).map(move |slot| ((token + slot) % N_EXPERTS) as u32))
            .collect();
        let ids = u32_buffer(&device, &ids_host);
        let input_values = input_rows(n_tokens, 3_001 + n_tokens);
        let input = f32_buffer(&device, &input_values);
        let output = device
            .alloc_buffer(
                n_tokens * TOP_K * N * 4,
                DType::F32,
                vec![n_tokens, TOP_K, N],
            )
            .unwrap();
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: n_tokens as u32,
            top_k: TOP_K as u32,
            n: N as u32,
            k: K as u32,
            n_experts: N_EXPERTS as u32,
            expert_stride,
            ggml_type: GgmlType::Q5_0,
        };
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().unwrap();
        if n_tokens == 1 {
            quantized_matmul_id_ggml_mv_with_policy(
                &mut encoder,
                &mut registry,
                &device,
                &input,
                &weight,
                &ids,
                &output,
                &params,
                &GgmlRoutingPolicy::default(),
            )
            .unwrap();
        } else {
            quantized_matmul_id_ggml(
                &mut encoder,
                &mut registry,
                &device,
                &input,
                &weight,
                &ids,
                &output,
                &params,
            )
            .unwrap();
        }
        encoder.commit_and_wait().unwrap();
        assert_close(
            output.as_slice::<f32>().unwrap(),
            &expert_expected(&input_values, &weight_bytes, &ids_host, n_tokens, false),
            &format!("expert shared M={n_tokens}"),
        );

        if n_tokens == 33 {
            let slotted_values = input_rows(n_tokens * TOP_K, 4_001);
            let slotted_input = f32_buffer(&device, &slotted_values);
            let slotted_output = device
                .alloc_buffer(
                    n_tokens * TOP_K * N * 4,
                    DType::F32,
                    vec![n_tokens, TOP_K, N],
                )
                .unwrap();
            let shared_output = device
                .alloc_buffer(
                    n_tokens * TOP_K * N * 4,
                    DType::F32,
                    vec![n_tokens, TOP_K, N],
                )
                .unwrap();
            let pair_first = device
                .alloc_buffer(
                    n_tokens * TOP_K * N * 4,
                    DType::F32,
                    vec![n_tokens, TOP_K, N],
                )
                .unwrap();
            let pair_second = device
                .alloc_buffer(
                    n_tokens * TOP_K * N * 4,
                    DType::F32,
                    vec![n_tokens, TOP_K, N],
                )
                .unwrap();
            let mut registry = KernelRegistry::new();

            let mut shared_scratch =
                IdMmScratch::alloc(&device, N_EXPERTS as u32, n_tokens as u32).unwrap();
            let mut encoder = device.command_encoder().unwrap();
            quantized_matmul_id_ggml_pooled(
                &mut encoder,
                &mut registry,
                &device,
                &input,
                &weight,
                &ids,
                &shared_output,
                &mut shared_scratch,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();

            let mut slotted_scratch =
                IdMmScratch::alloc(&device, N_EXPERTS as u32, n_tokens as u32).unwrap();
            let mut encoder = device.command_encoder().unwrap();
            quantized_matmul_id_ggml_pooled_slotted(
                &mut encoder,
                &mut registry,
                &device,
                &slotted_input,
                &weight,
                &ids,
                &slotted_output,
                &mut slotted_scratch,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();

            let mut pair_scratch =
                IdMmScratch::alloc(&device, N_EXPERTS as u32, n_tokens as u32).unwrap();
            let mut encoder = device.command_encoder().unwrap();
            quantized_matmul_id_ggml_pooled_pair(
                &mut encoder,
                &mut registry,
                &device,
                &input,
                &weight,
                &weight,
                &ids,
                &pair_first,
                &pair_second,
                &mut pair_scratch,
                &params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            assert_close(
                slotted_output.as_slice::<f32>().unwrap(),
                &expert_expected(&slotted_values, &weight_bytes, &ids_host, n_tokens, true),
                "expert slotted",
            );
            let shared_expected =
                expert_expected(&input_values, &weight_bytes, &ids_host, n_tokens, false);
            assert_close(
                shared_output.as_slice::<f32>().unwrap(),
                &shared_expected,
                "expert pooled shared",
            );
            assert_close(
                pair_first.as_slice::<f32>().unwrap(),
                &shared_expected,
                "expert pair first",
            );
            assert_close(
                pair_second.as_slice::<f32>().unwrap(),
                &shared_expected,
                "expert pair second",
            );
        }
    }
}

#[test]
fn q5_0_expert_ids_fail_closed_on_device_without_weight_oob() {
    let device = MlxDevice::new().unwrap();
    let weight = u8_buffer(&device, &stacked_expert_bytes());
    let expert_stride = (N * K / QK5_0 * BLOCK_BYTES + EXPERT_PADDING) as u64;

    // The matvec route has no multiplicity restriction, but every ID is
    // bounds-checked before the expert stride is applied. Only the invalid
    // row is poisoned because every other row remains independently valid.
    let mv_ids_host = [0, 1, 2, 3, 4, u32::MAX];
    let mv_ids = u32_buffer(&device, &mv_ids_host);
    let mv_input = f32_buffer(&device, &input_rows(1, 5_001));
    let mv_output = empty_buffer(&device, TOP_K * N * 4, DType::F32);
    let mv_params = GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: TOP_K as u32,
        n: N as u32,
        k: K as u32,
        n_experts: N_EXPERTS as u32,
        expert_stride,
        ggml_type: GgmlType::Q5_0,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    quantized_matmul_id_ggml_mv_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &mv_input,
        &weight,
        &mv_ids,
        &mv_output,
        &mv_params,
        &GgmlRoutingPolicy::default(),
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let mv_values = mv_output.as_slice::<f32>().unwrap();
    assert!(mv_values[..(TOP_K - 1) * N]
        .iter()
        .all(|value| value.is_finite()));
    assert_all_nan(&mv_values[(TOP_K - 1) * N..], "invalid expert MV row");

    let n_tokens = 33usize;
    let valid_ids: Vec<u32> = (0..n_tokens)
        .flat_map(|token| (0..TOP_K).map(move |slot| ((token + slot) % N_EXPERTS) as u32))
        .collect();
    let mm_input = f32_buffer(&device, &input_rows(n_tokens, 5_101));
    let mm_params = GgmlQuantizedMatmulIdParams {
        n_tokens: n_tokens as u32,
        ..mv_params
    };

    for (case, mutate, tensor_preference) in [
        (
            "out-of-range SIMD",
            0usize,
            GgmlTensorMmPreference::ForceSimd,
        ),
        (
            "duplicate-per-token SIMD",
            1usize,
            GgmlTensorMmPreference::ForceSimd,
        ),
        (
            "duplicate-per-token device-selected",
            1usize,
            GgmlTensorMmPreference::AutoProbe,
        ),
    ] {
        let mut invalid_ids = valid_ids.clone();
        if mutate == 0 {
            invalid_ids[TOP_K + 2] = u32::MAX;
        } else {
            invalid_ids[TOP_K + 1] = invalid_ids[TOP_K];
        }
        let ids = u32_buffer(&device, &invalid_ids);
        let output = empty_buffer(&device, n_tokens * TOP_K * N * 4, DType::F32);
        let mut encoder = device.command_encoder().unwrap();
        let routing = GgmlRoutingPolicy {
            expert_mm_threshold: 32,
            expert_tensor_mm: tensor_preference,
            ..GgmlRoutingPolicy::default()
        };
        quantized_matmul_id_ggml_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &mm_input,
            &weight,
            &ids,
            &output,
            &mm_params,
            &routing,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
        assert_all_nan(output.as_slice::<f32>().unwrap(), case);

        if mutate == 1 && tensor_preference == GgmlTensorMmPreference::AutoProbe {
            let first = empty_buffer(&device, n_tokens * TOP_K * N * 4, DType::F32);
            let second = empty_buffer(&device, n_tokens * TOP_K * N * 4, DType::F32);
            let mut scratch =
                IdMmScratch::alloc(&device, N_EXPERTS as u32, n_tokens as u32).unwrap();
            let mut encoder = device.command_encoder().unwrap();
            quantized_matmul_id_ggml_pooled_pair_with_policy(
                &mut encoder,
                &mut registry,
                &device,
                &mm_input,
                &weight,
                &weight,
                &ids,
                &first,
                &second,
                &mut scratch,
                &mm_params,
                &routing,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            assert_all_nan(first.as_slice::<f32>().unwrap(), "invalid pair first");
            assert_all_nan(second.as_slice::<f32>().unwrap(), "invalid pair reuse");
        }
    }
}
