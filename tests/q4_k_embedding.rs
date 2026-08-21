//! Exact GGML Q4_K embedding-gather parity, routing, and validation gates.

#![cfg(target_vendor = "apple")]

use half::f16;
use mlx_native::{
    embedding_gather_q4_k, CapturedNode, DType, DispatchKind, EmbeddingQ4KParams, GgmlType,
    GgufFile, GraphExecutor, KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 144;
const QWEN38_HIDDEN: usize = 5_120;
const VOCAB: usize = 3;

fn packed_scales(scales: [u8; 8], minimums: [u8; 8]) -> [u8; 12] {
    let mut packed = [0u8; 12];
    for group in 0..4 {
        packed[group] = scales[group] & 0x3f;
        packed[group + 4] = minimums[group] & 0x3f;
    }
    for group in 4..8 {
        packed[group + 4] = (scales[group] & 0x0f) | ((minimums[group] & 0x0f) << 4);
        packed[group - 4] |= (scales[group] >> 4) << 6;
        packed[group] |= (minimums[group] >> 4) << 6;
    }
    packed
}

fn block(row: usize, block_index: usize) -> [u8; BLOCK_BYTES] {
    let mut bytes = [0u8; BLOCK_BYTES];
    bytes[..2].copy_from_slice(&f16::from_f32(1.0 / 256.0).to_le_bytes());
    bytes[2..4].copy_from_slice(&f16::from_f32(1.0 / 512.0).to_le_bytes());

    let mut scales = [0u8; 8];
    let mut minimums = [0u8; 8];
    for group in 0..8 {
        scales[group] = ((row * 17 + block_index * 11 + group * 7) % 63 + 1) as u8;
        minimums[group] = ((row * 13 + block_index * 5 + group * 9) % 64) as u8;
    }
    bytes[4..16].copy_from_slice(&packed_scales(scales, minimums));

    for pair in 0..4 {
        for lane in 0..32 {
            let low = ((row * 3 + block_index * 5 + pair * 7 + lane) & 0x0f) as u8;
            let high = ((row * 11 + block_index * 3 + pair * 5 + lane * 7) & 0x0f) as u8;
            bytes[16 + pair * 32 + lane] = low | (high << 4);
        }
    }
    bytes
}

fn qwen38_weights() -> Vec<u8> {
    let blocks_per_row = QWEN38_HIDDEN / QK_K;
    let mut weights = Vec::with_capacity(VOCAB * blocks_per_row * BLOCK_BYTES);
    for row in 0..VOCAB {
        for block_index in 0..blocks_per_row {
            weights.extend_from_slice(&block(row, block_index));
        }
    }
    weights
}

fn cpu_row(weights: &[u8], row: usize) -> Vec<f32> {
    let row_bytes = QWEN38_HIDDEN / QK_K * BLOCK_BYTES;
    let mut output = vec![0.0f32; QWEN38_HIDDEN];
    mlx_native::gguf::test_only_dequantize(
        &weights[row * row_bytes..(row + 1) * row_bytes],
        GgmlType::Q4_K,
        &mut output,
    )
    .expect("CPU Q4_K dequantization");
    output
}

fn owned_weight(device: &MlxDevice, weights: &[u8]) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(weights.len(), DType::U8, vec![VOCAB, QWEN38_HIDDEN])
        .expect("Q4_K weight buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("Q4_K weight bytes")
        .copy_from_slice(weights);
    buffer
}

fn token_buffer(device: &MlxDevice, ids: &[u32]) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(ids.len() * 4, DType::U32, vec![ids.len()])
        .expect("token IDs");
    buffer
        .as_mut_slice::<u32>()
        .expect("token ID bytes")
        .copy_from_slice(ids);
    buffer
}

#[test]
fn qwen38_width_graph_gather_is_bit_exact() {
    let weights = qwen38_weights();
    let ids_host = [2u32, 0, 1, 2];
    let device = MlxDevice::new().expect("Metal device");
    let weight = owned_weight(&device, &weights);
    let ids = token_buffer(&device, &ids_host);
    let output = device
        .alloc_buffer(
            ids_host.len() * QWEN38_HIDDEN * 4,
            DType::F32,
            vec![ids_host.len(), QWEN38_HIDDEN],
        )
        .expect("embedding output");
    let executor = GraphExecutor::new(device.clone());
    let mut registry = KernelRegistry::new();
    let params = EmbeddingQ4KParams {
        vocab_size: VOCAB,
        embed_dim: QWEN38_HIDDEN,
        n_tokens: ids_host.len(),
    };
    let mut session = executor.begin().expect("graph session");
    session
        .embedding_gather_q4_k(&mut registry, &device, &weight, &ids, &output, &params)
        .expect("Q4_K graph embedding gather");
    session.finish().expect("GPU completion");

    let expected: Vec<Vec<f32>> = (0..VOCAB).map(|row| cpu_row(&weights, row)).collect();
    for (token_index, (actual, &row)) in output
        .as_slice::<f32>()
        .expect("embedding output values")
        .chunks_exact(QWEN38_HIDDEN)
        .zip(ids_host.iter())
        .enumerate()
    {
        for (column, (&got, &want)) in actual.iter().zip(&expected[row as usize]).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "token {token_index}, row {row}, column {column}"
            );
        }
    }
}

#[test]
fn dispatch_route_and_invalid_ids_fail_before_encoding() {
    let weights = qwen38_weights();
    let device = MlxDevice::new().expect("Metal device");
    let weight = owned_weight(&device, &weights);
    let mut ids = token_buffer(&device, &[0]);
    let output = device
        .alloc_buffer(QWEN38_HIDDEN * 4, DType::F32, vec![1, QWEN38_HIDDEN])
        .expect("embedding output");
    let params = EmbeddingQ4KParams {
        vocab_size: VOCAB,
        embed_dim: QWEN38_HIDDEN,
        n_tokens: 1,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("command encoder");
    encoder.start_capture();
    embedding_gather_q4_k(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &params,
    )
    .expect("capture Q4_K embedding dispatch");
    let captured = encoder.take_capture().expect("captured graph");
    assert_eq!(captured.len(), 1);
    let CapturedNode::Dispatch {
        pipeline,
        threads_per_grid,
        threads_per_threadgroup,
        dispatch_kind,
        ..
    } = &captured[0]
    else {
        panic!("expected one captured dispatch");
    };
    assert_eq!(pipeline.label(), "embedding_gather_q4_k_f32");
    assert!(matches!(dispatch_kind, DispatchKind::Threads));
    assert_eq!(threads_per_grid.width, (QWEN38_HIDDEN / 16) as u64);
    assert_eq!(threads_per_grid.height, 1);
    assert_eq!(threads_per_threadgroup.width, 256);

    ids.as_mut_slice::<u32>().expect("token ID bytes")[0] = VOCAB as u32;
    let mut encoder = device.command_encoder().expect("invalid-ID encoder");
    encoder.start_capture();
    let error = embedding_gather_q4_k(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &params,
    )
    .expect_err("out-of-range token ID must fail");
    assert!(error.to_string().contains("token_ids[0]=3"), "{error}");
    assert!(
        encoder
            .take_capture()
            .expect("invalid-ID capture")
            .is_empty(),
        "invalid IDs must be rejected before command encoding"
    );
}

#[test]
fn strict_preflight_rejects_malformed_contracts() {
    let weights = qwen38_weights();
    let device = MlxDevice::new().expect("Metal device");
    let weight = owned_weight(&device, &weights);
    let ids = token_buffer(&device, &[0]);
    let output = device
        .alloc_buffer(QWEN38_HIDDEN * 4, DType::F32, vec![1, QWEN38_HIDDEN])
        .expect("embedding output");
    let mut registry = KernelRegistry::new();

    let mut reject = |weight: &mlx_native::MlxBuffer,
                      ids: &mlx_native::MlxBuffer,
                      output: &mlx_native::MlxBuffer,
                      params: EmbeddingQ4KParams| {
        let mut encoder = device.command_encoder().expect("validation encoder");
        embedding_gather_q4_k(
            &mut encoder,
            &mut registry,
            &device,
            weight,
            ids,
            output,
            &params,
        )
        .expect_err("malformed Q4_K embedding contract must fail")
    };

    let exact = EmbeddingQ4KParams {
        vocab_size: VOCAB,
        embed_dim: QWEN38_HIDDEN,
        n_tokens: 1,
    };
    assert!(reject(
        &weight,
        &ids,
        &output,
        EmbeddingQ4KParams {
            embed_dim: QWEN38_HIDDEN - 1,
            ..exact
        }
    )
    .to_string()
    .contains("divisible by 256"));

    assert!(reject(
        &weight,
        &ids,
        &output,
        EmbeddingQ4KParams {
            vocab_size: u32::MAX as usize,
            ..exact
        }
    )
    .to_string()
    .contains("weight block count exceeds u32 shader indexing"));
    assert!(reject(
        &weight,
        &ids,
        &output,
        EmbeddingQ4KParams {
            n_tokens: u32::MAX as usize,
            ..exact
        }
    )
    .to_string()
    .contains("output element count exceeds u32 shader indexing"));

    let oversized_weight = device
        .alloc_buffer(weights.len() + 1, DType::U8, vec![weights.len() + 1])
        .expect("oversized weight");
    assert!(reject(&oversized_weight, &ids, &output, exact)
        .to_string()
        .contains("must contain exactly"));

    let short_output = output.slice_view(0, QWEN38_HIDDEN - 1);
    assert!(reject(&weight, &ids, &short_output, exact)
        .to_string()
        .contains("must contain exactly"));

    let wrong_ids = device
        .alloc_buffer(4, DType::F32, vec![1])
        .expect("wrong ID dtype");
    assert!(reject(&weight, &wrong_ids, &output, exact)
        .to_string()
        .contains("expected U8/U32/F32"));

    let alias_params = EmbeddingQ4KParams {
        vocab_size: 64,
        embed_dim: QK_K,
        n_tokens: 9,
    };
    let alias_weight = device
        .alloc_buffer(64 * BLOCK_BYTES, DType::U8, vec![64, QK_K])
        .expect("aliased weight storage");
    let alias_output = mlx_native::MlxBuffer::from_raw(
        alias_weight.metal_buffer().clone(),
        DType::F32,
        vec![9, QK_K],
    );
    let alias_ids = token_buffer(&device, &[0; 9]);
    assert!(
        reject(&alias_weight, &alias_ids, &alias_output, alias_params)
            .to_string()
            .contains("must not overlap")
    );

    assert!(reject(
        &weight,
        &ids,
        &output,
        EmbeddingQ4KParams {
            n_tokens: 0,
            ..exact
        }
    )
    .to_string()
    .contains("greater than zero"));
}

fn q4_k_fixture(weights: &[u8]) -> Vec<u8> {
    let name = "token_embd.weight";
    let mut file = Vec::new();
    file.extend_from_slice(b"GGUF");
    file.extend_from_slice(&3u32.to_le_bytes());
    file.extend_from_slice(&1u64.to_le_bytes());
    file.extend_from_slice(&0u64.to_le_bytes());
    file.extend_from_slice(&(name.len() as u64).to_le_bytes());
    file.extend_from_slice(name.as_bytes());
    file.extend_from_slice(&2u32.to_le_bytes());
    file.extend_from_slice(&(QWEN38_HIDDEN as u64).to_le_bytes());
    file.extend_from_slice(&(VOCAB as u64).to_le_bytes());
    file.extend_from_slice(&12u32.to_le_bytes()); // GGML_TYPE_Q4_K
    file.extend_from_slice(&32u64.to_le_bytes());
    while file.len() % 32 != 0 {
        file.push(0);
    }
    file.extend_from_slice(&[0u8; 32]);
    file.extend_from_slice(weights);
    file
}

#[test]
fn mapped_and_owned_qwen38_rows_are_bit_identical() {
    let weights = qwen38_weights();
    let path = std::env::temp_dir().join(format!("mlx_q4k_embedding_{}.gguf", std::process::id()));
    std::fs::write(&path, q4_k_fixture(&weights)).expect("write Q4_K GGUF fixture");

    let device = MlxDevice::new().expect("Metal device");
    let gguf = GgufFile::open(&path).expect("open Q4_K GGUF fixture");
    let mapped = gguf
        .load_tensor_mapped("token_embd.weight", &device)
        .expect("map Q4_K embedding");
    let owned = gguf
        .load_tensor("token_embd.weight", &device)
        .expect("copy Q4_K embedding");
    assert!(mapped.is_file_backed());
    assert!(!owned.is_file_backed());
    assert_ne!(mapped.byte_offset(), 0, "mapped route must bind an offset");
    assert_eq!(mapped.shape(), &[VOCAB, QWEN38_HIDDEN]);
    assert_eq!(mapped.data_byte_len(), owned.data_byte_len());

    let ids_host = [2u32, 0];
    let ids = token_buffer(&device, &ids_host);
    let output_bytes = ids_host.len() * QWEN38_HIDDEN * 4;
    let mapped_output = device
        .alloc_buffer(
            output_bytes,
            DType::F32,
            vec![ids_host.len(), QWEN38_HIDDEN],
        )
        .expect("mapped output");
    let owned_output = device
        .alloc_buffer(
            output_bytes,
            DType::F32,
            vec![ids_host.len(), QWEN38_HIDDEN],
        )
        .expect("owned output");
    let params = EmbeddingQ4KParams {
        vocab_size: VOCAB,
        embed_dim: QWEN38_HIDDEN,
        n_tokens: ids_host.len(),
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("command encoder");
    embedding_gather_q4_k(
        &mut encoder,
        &mut registry,
        &device,
        &mapped,
        &ids,
        &mapped_output,
        &params,
    )
    .expect("mapped Q4_K gather");
    embedding_gather_q4_k(
        &mut encoder,
        &mut registry,
        &device,
        &owned,
        &ids,
        &owned_output,
        &params,
    )
    .expect("owned Q4_K gather");
    encoder.commit_and_wait().expect("GPU completion");

    for (index, (&mapped_value, &owned_value)) in mapped_output
        .as_slice::<f32>()
        .expect("mapped output values")
        .iter()
        .zip(owned_output.as_slice::<f32>().expect("owned output values"))
        .enumerate()
    {
        assert_eq!(
            mapped_value.to_bits(),
            owned_value.to_bits(),
            "mapped/owned value {index}"
        );
    }

    std::fs::remove_file(path).expect("remove Q4_K GGUF fixture");
}
