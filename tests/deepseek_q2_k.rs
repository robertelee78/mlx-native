//! GGML Q2_K loading and Metal execution proofs for DeepSeek-V4 Q2_K_S.

use mlx_native::{
    gguf::{test_only_dequantize, GgufFile},
    ops::{
        embedding_q2_k::{embedding_gather_q2_k, EmbeddingQ2KParams},
        quantized_matmul_ggml::{
            dispatch_mm_for_test, dispatch_mm_simd_for_test, quantized_matmul_q2_k_batched_mv,
        },
    },
    quantized_matmul_ggml, quantized_matmul_id_ggml, quantized_matmul_id_ggml_pooled,
    quantized_matmul_id_ggml_pooled_pair, quantized_matmul_id_ggml_pooled_slotted, DType,
    GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlType, IdMmScratch, KernelRegistry,
    MlxDevice,
};

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 84;

fn block(seed: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(BLOCK_BYTES);
    for group in 0..16u8 {
        let scale = group.wrapping_add(seed) % 15 + 1;
        let min = 15 - ((group.wrapping_mul(3).wrapping_add(seed)) % 16);
        bytes.push((min << 4) | scale);
    }
    for index in 0..64u8 {
        bytes.push(index.wrapping_mul(73).wrapping_add(seed.wrapping_mul(19)));
    }
    bytes.extend_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
    bytes.extend_from_slice(&half::f16::from_f32(0.0625).to_bits().to_le_bytes());
    assert_eq!(bytes.len(), BLOCK_BYTES);
    bytes
}

/// Canonical loop order from GGML's `dequantize_row_q2_K`.
fn reference_dequant(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len(), BLOCK_BYTES);
    let scales = &bytes[..16];
    let qs = &bytes[16..80];
    let d = half::f16::from_le_bytes([bytes[80], bytes[81]]).to_f32();
    let dmin = half::f16::from_le_bytes([bytes[82], bytes[83]]).to_f32();
    let mut output = Vec::with_capacity(QK_K);
    let mut scale_index = 0usize;

    for half in 0..2 {
        for shift in [0, 2, 4, 6] {
            for segment in 0..2 {
                let scale = scales[scale_index];
                scale_index += 1;
                let dl = d * (scale & 0x0f) as f32;
                let ml = dmin * (scale >> 4) as f32;
                let q_offset = half * 32 + segment * 16;
                for lane in 0..16 {
                    let q = (qs[q_offset + lane] >> shift) & 0x03;
                    output.push(dl * q as f32 - ml);
                }
            }
        }
    }
    output
}

fn input(rows: usize) -> Vec<f32> {
    (0..rows * QK_K)
        .map(|index| ((index * 17 + 11) % 97) as f32 / 48.0 - 1.0)
        .collect()
}

fn cpu_dot(weight: &[f32], input: &[f32]) -> f32 {
    weight.iter().zip(input).map(|(w, x)| w * x).sum()
}

#[test]
fn q2_k_host_decode_matches_all_256_canonical_values() {
    let bytes = block(5);
    let expected = reference_dequant(&bytes);
    let mut actual = vec![0.0f32; QK_K];
    test_only_dequantize(&bytes, GgmlType::Q2_K, &mut actual).expect("Q2_K decode");
    assert_eq!(actual, expected);
    assert!(actual.iter().any(|value| *value < 0.0));
    assert!(actual.iter().any(|value| *value > 0.0));
}

#[test]
fn q2_k_host_decode_rejects_malformed_buffers() {
    let bytes = block(1);
    let mut output = vec![0.0f32; QK_K];
    assert!(test_only_dequantize(&bytes[..BLOCK_BYTES - 1], GgmlType::Q2_K, &mut output).is_err());
    assert!(test_only_dequantize(&bytes, GgmlType::Q2_K, &mut output[..QK_K - 1]).is_err());
}

#[test]
fn q2_k_embedding_gather_matches_canonical_rows_and_rejects_bad_ids() {
    let vocab = 3usize;
    let mut weights = Vec::with_capacity(vocab * BLOCK_BYTES);
    let mut decoded = Vec::with_capacity(vocab);
    for row in 0..vocab {
        let bytes = block(row as u8 + 2);
        decoded.push(reference_dequant(&bytes));
        weights.extend(bytes);
    }
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::embedding_q2_k::register(&mut registry);
    let mut weight = device
        .alloc_buffer(weights.len(), DType::U8, vec![vocab, QK_K])
        .expect("weight");
    weight
        .as_mut_slice::<u8>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut ids = device
        .alloc_buffer(2 * 4, DType::U32, vec![2])
        .expect("ids");
    ids.as_mut_slice::<u32>()
        .expect("id slice")
        .copy_from_slice(&[2, 0]);
    let output = device
        .alloc_buffer(2 * QK_K * 4, DType::F32, vec![2, QK_K])
        .expect("output");
    let params = EmbeddingQ2KParams {
        vocab_size: vocab,
        embed_dim: QK_K,
        n_tokens: 2,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    embedding_gather_q2_k(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &params,
    )
    .expect("Q2_K embedding gather");
    encoder.commit_and_wait().expect("GPU completion");
    let actual = output.as_slice::<f32>().expect("output slice");
    for (actual_row, expected_row) in actual.chunks_exact(QK_K).zip([&decoded[2], &decoded[0]]) {
        for (&got, &want) in actual_row.iter().zip(expected_row) {
            assert!((got - want).abs() <= 1e-6, "{got} != {want}");
        }
    }

    ids.as_mut_slice::<u32>().expect("id slice")[1] = vocab as u32;
    let mut encoder = device.command_encoder().expect("encoder");
    assert!(embedding_gather_q2_k(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &params,
    )
    .is_err());
}

#[test]
fn q2_k_gguf_type_10_parses_with_exact_block_size() {
    let name = "q2.weight";
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&1u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&(name.len() as u64).to_le_bytes());
    file.extend_from_slice(name.as_bytes());
    file.extend_from_slice(&1u32.to_le_bytes());
    file.extend_from_slice(&(QK_K as u64).to_le_bytes());
    file.extend_from_slice(&10u32.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    while file.len() % 32 != 0 {
        file.push(0);
    }
    file.extend_from_slice(&block(3));

    let path = std::env::temp_dir().join(format!("mlx_q2k_{}.gguf", std::process::id()));
    std::fs::write(&path, file).expect("write fixture");
    let gguf = GgufFile::open(&path).expect("parse Q2_K GGUF");
    let info = gguf.tensor_info(name).expect("tensor info");
    assert_eq!(info.ggml_type, GgmlType::Q2_K);
    assert_eq!(info.byte_len, BLOCK_BYTES);
    let _ = std::fs::remove_file(path);
}

fn run_dense(rows: usize, force_mm: bool, force_simd: bool) {
    let n = 8usize;
    let mut weights = Vec::with_capacity(n * BLOCK_BYTES);
    let mut decoded = Vec::with_capacity(n);
    for row in 0..n {
        let bytes = block(row as u8 + 1);
        decoded.push(reference_dequant(&bytes));
        weights.extend(bytes);
    }
    let inputs = input(rows);
    let expected: Vec<f32> = (0..rows)
        .flat_map(|m| {
            (0..n).map({
                let inputs = &inputs;
                let decoded = &decoded;
                move |row| cpu_dot(&decoded[row], &inputs[m * QK_K..(m + 1) * QK_K])
            })
        })
        .collect();

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut weight_buf = device
        .alloc_buffer(weights.len(), DType::U8, vec![weights.len()])
        .expect("weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut input_buf = device
        .alloc_buffer(inputs.len() * 4, DType::F32, vec![inputs.len()])
        .expect("input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&inputs);
    let output_buf = device
        .alloc_buffer(expected.len() * 4, DType::F32, vec![expected.len()])
        .expect("output");
    let params = GgmlQuantizedMatmulParams {
        m: rows as u32,
        n: n as u32,
        k: QK_K as u32,
        ggml_type: GgmlType::Q2_K,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    if force_simd {
        dispatch_mm_simd_for_test(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &output_buf,
            &params,
        )
        .expect("Q2_K forced simdgroup Metal MM");
    } else if force_mm {
        dispatch_mm_for_test(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &output_buf,
            &params,
        )
        .expect("Q2_K forced Metal MM");
    } else {
        quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &output_buf,
            &params,
        )
        .expect("Q2_K Metal matmul");
    }
    encoder.commit_and_wait().expect("GPU completion");
    let actual = output_buf.as_slice::<f32>().expect("output slice");
    let tolerance = if force_mm || force_simd || rows > 8 {
        1e-2
    } else {
        2e-4
    };
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "dense mismatch {index}: {got} != {want}"
        );
    }
}

#[test]
fn q2_k_dense_metal_decode_matches_reference() {
    run_dense(1, false, false);
}

#[test]
fn q2_k_batched_mv_is_byte_identical_to_independent_dispatches() {
    let batches = 3usize;
    let n = 8usize;
    let mut weights = Vec::with_capacity(batches * n * BLOCK_BYTES);
    for batch in 0..batches {
        for row in 0..n {
            weights.extend(block((batch * n + row + 1) as u8));
        }
    }
    let inputs = input(batches);
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut weight = device
        .alloc_buffer(weights.len(), DType::U8, vec![batches, n, QK_K])
        .expect("batched weights");
    weight
        .as_mut_slice::<u8>()
        .expect("weight bytes")
        .copy_from_slice(&weights);
    let mut input = device
        .alloc_buffer(
            inputs.len() * DType::F32.size_of(),
            DType::F32,
            vec![batches, 1, QK_K],
        )
        .expect("batched inputs");
    input
        .as_mut_slice::<f32>()
        .expect("input values")
        .copy_from_slice(&inputs);
    let batched_output = device
        .alloc_buffer(
            batches * n * DType::F32.size_of(),
            DType::F32,
            vec![batches, 1, n],
        )
        .expect("batched output");
    let serial_output = device
        .alloc_buffer(
            batches * n * DType::F32.size_of(),
            DType::F32,
            vec![batches, 1, n],
        )
        .expect("serial output");

    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_q2_k_batched_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &batched_output,
        batches as u32,
        1,
        n as u32,
        QK_K as u32,
    )
    .expect("batched Q2_K matvec");
    for batch in 0..batches {
        let input_view = input.slice_view((batch * QK_K * DType::F32.size_of()) as u64, QK_K);
        let weight_view = weight.slice_view((batch * n * BLOCK_BYTES) as u64, n * BLOCK_BYTES);
        let output_view = serial_output.slice_view((batch * n * DType::F32.size_of()) as u64, n);
        quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input_view,
            &weight_view,
            &output_view,
            &GgmlQuantizedMatmulParams {
                m: 1,
                n: n as u32,
                k: QK_K as u32,
                ggml_type: GgmlType::Q2_K,
            },
        )
        .expect("serial Q2_K matvec");
    }
    encoder.commit_and_wait().expect("GPU completion");
    assert_eq!(
        batched_output.as_slice::<f32>().expect("batched values"),
        serial_output.as_slice::<f32>().expect("serial values")
    );
}

#[test]
fn q2_k_dense_metal_prefill_matches_reference() {
    run_dense(9, false, false);
}

#[test]
fn q2_k_dense_forced_mm_matches_reference() {
    run_dense(9, true, false);
}

#[test]
fn q2_k_dense_forced_simd_mm_matches_reference() {
    run_dense(9, false, true);
}

fn run_expert(tokens: usize, top_k: usize, experts: usize, force_mm: bool) {
    let n = 8usize;
    let mut weights = Vec::new();
    let mut decoded = vec![vec![vec![0.0f32; QK_K]; n]; experts];
    for (expert, expert_rows) in decoded.iter_mut().enumerate() {
        for (row, decoded_row) in expert_rows.iter_mut().enumerate() {
            let bytes = block((expert * n + row + 1) as u8);
            *decoded_row = reference_dequant(&bytes);
            weights.extend(bytes);
        }
    }
    let inputs = input(tokens);
    let ids: Vec<u32> = (0..tokens)
        .flat_map(|token| (0..top_k).map(move |slot| ((token * 3 + slot) % experts) as u32))
        .collect();
    let mut expected = Vec::new();
    for (route, &expert) in ids.iter().enumerate() {
        let token = route / top_k;
        for row in 0..n {
            expected.push(cpu_dot(
                &decoded[expert as usize][row],
                &inputs[token * QK_K..(token + 1) * QK_K],
            ));
        }
    }

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut weight_buf = device
        .alloc_buffer(weights.len(), DType::U8, vec![weights.len()])
        .expect("weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut input_buf = device
        .alloc_buffer(inputs.len() * 4, DType::F32, vec![inputs.len()])
        .expect("input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&inputs);
    let mut ids_buf = device
        .alloc_buffer(ids.len() * 4, DType::U32, vec![ids.len()])
        .expect("ids");
    ids_buf
        .as_mut_slice::<u32>()
        .expect("ids slice")
        .copy_from_slice(&ids);
    let output_buf = device
        .alloc_buffer(expected.len() * 4, DType::F32, vec![expected.len()])
        .expect("output");
    let prepared_output_buf = device
        .alloc_buffer(expected.len() * 4, DType::F32, vec![expected.len()])
        .expect("prepared output");
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: QK_K as u32,
        n_experts: experts as u32,
        expert_stride: (n * BLOCK_BYTES) as u64,
        ggml_type: GgmlType::Q2_K,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    if force_mm {
        let mut scratch =
            IdMmScratch::alloc(&device, experts as u32, tokens as u32).expect("pair scratch");
        quantized_matmul_id_ggml_pooled_pair(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &weight_buf,
            &ids_buf,
            &output_buf,
            &prepared_output_buf,
            &mut scratch,
            &params,
        )
        .expect("Q2_K paired expert Metal MM_ID");
    } else {
        quantized_matmul_id_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &ids_buf,
            &output_buf,
            &params,
        )
        .expect("Q2_K expert Metal matmul");
    }
    encoder.commit_and_wait().expect("GPU completion");
    let actual = output_buf.as_slice::<f32>().expect("output slice");
    if force_mm {
        assert_eq!(
            prepared_output_buf
                .as_slice::<f32>()
                .expect("prepared output slice"),
            actual,
            "Q2_K prepared schedule must preserve ordinary mm_id output bit-for-bit"
        );
    }
    let tolerance = if force_mm || (tokens > 32 && matches!(top_k, 1 | 6 | 8)) {
        1e-2
    } else {
        2e-4
    };
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "expert mismatch {index}: {got} != {want}"
        );
    }
}

#[test]
fn q2_k_expert_routed_metal_matches_reference() {
    run_expert(2, 2, 3, false);
}

#[test]
fn q2_k_expert_mm_id_top_k_6_matches_reference() {
    run_expert(33, 6, 8, true);
}

#[test]
fn q2_k_public_prefill_top_k_6_matches_reference() {
    run_expert(33, 6, 8, false);
}

#[test]
fn q2_k_slotted_down_mm_is_byte_identical_to_flattened_rows() {
    let tokens = 33usize;
    let top_k = 6usize;
    let experts = 8usize;
    let n = 8usize;
    let routed_rows = tokens * top_k;

    let mut weights = Vec::new();
    for expert in 0..experts {
        for row in 0..n {
            weights.extend(block((expert * n + row + 1) as u8));
        }
    }
    let routed_input = input(routed_rows);
    let ids: Vec<u32> = (0..tokens)
        .flat_map(|token| (0..top_k).map(move |slot| ((token * 3 + slot) % experts) as u32))
        .collect();

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut weight_buf = device
        .alloc_buffer(weights.len(), DType::U8, vec![weights.len()])
        .expect("weight");
    weight_buf
        .as_mut_slice::<u8>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut input_buf = device
        .alloc_buffer(
            routed_input.len() * 4,
            DType::F32,
            vec![tokens, top_k, QK_K],
        )
        .expect("slotted input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&routed_input);
    let mut ids_buf = device
        .alloc_buffer(ids.len() * 4, DType::U32, vec![tokens, top_k])
        .expect("ids");
    ids_buf
        .as_mut_slice::<u32>()
        .expect("ids slice")
        .copy_from_slice(&ids);
    let flat_output = device
        .alloc_buffer(routed_rows * n * 4, DType::F32, vec![routed_rows, n])
        .expect("flat output");
    let slotted_output = device
        .alloc_buffer(routed_rows * n * 4, DType::F32, vec![tokens, top_k, n])
        .expect("slotted output");

    let flat_params = GgmlQuantizedMatmulIdParams {
        n_tokens: routed_rows as u32,
        top_k: 1,
        n: n as u32,
        k: QK_K as u32,
        n_experts: experts as u32,
        expert_stride: (n * BLOCK_BYTES) as u64,
        ggml_type: GgmlType::Q2_K,
    };
    let slotted_params = GgmlQuantizedMatmulIdParams {
        n_tokens: tokens as u32,
        top_k: top_k as u32,
        ..flat_params
    };
    let mut flat_scratch =
        IdMmScratch::alloc(&device, experts as u32, routed_rows as u32).expect("flat scratch");
    let mut slotted_scratch =
        IdMmScratch::alloc(&device, experts as u32, tokens as u32).expect("slotted scratch");

    let mut flat_encoder = device.command_encoder().expect("flat encoder");
    quantized_matmul_id_ggml_pooled(
        &mut flat_encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &ids_buf,
        &flat_output,
        &mut flat_scratch,
        &flat_params,
    )
    .expect("flattened Q2_K down mm_id");
    flat_encoder.commit_and_wait().expect("flat GPU completion");

    let mut slotted_encoder = device.command_encoder().expect("slotted encoder");
    quantized_matmul_id_ggml_pooled_slotted(
        &mut slotted_encoder,
        &mut registry,
        &device,
        &input_buf,
        &weight_buf,
        &ids_buf,
        &slotted_output,
        &mut slotted_scratch,
        &slotted_params,
    )
    .expect("slotted Q2_K down mm_id");
    slotted_encoder
        .commit_and_wait()
        .expect("slotted GPU completion");

    assert_eq!(
        slotted_output
            .as_slice::<f32>()
            .expect("slotted output values"),
        flat_output.as_slice::<f32>().expect("flat output values")
    );
}
