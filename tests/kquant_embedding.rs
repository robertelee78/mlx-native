//! Exact GGML Q5_K/Q6_K embedding-gather parity and validation gates.

#![cfg(target_vendor = "apple")]

use half::f16;
use mlx_native::{
    embedding_gather_q5_k, embedding_gather_q6_k, DType, EmbeddingQ5KParams, EmbeddingQ6KParams,
    GgmlType, KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;
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

fn q5_k_block(row: usize, block_index: usize) -> [u8; Q5_K_BLOCK_BYTES] {
    let mut bytes = [0u8; Q5_K_BLOCK_BYTES];
    bytes[..2].copy_from_slice(&f16::from_f32(1.0 / 256.0).to_le_bytes());
    bytes[2..4].copy_from_slice(&f16::from_f32(1.0 / 512.0).to_le_bytes());
    let mut scales = [0u8; 8];
    let mut minimums = [0u8; 8];
    for group in 0..8 {
        scales[group] = ((row * 17 + block_index * 11 + group * 7) % 63 + 1) as u8;
        minimums[group] = ((row * 13 + block_index * 5 + group * 9) % 64) as u8;
    }
    bytes[4..16].copy_from_slice(&packed_scales(scales, minimums));
    for lane in 0..32 {
        bytes[16 + lane] = (row * 31 + block_index * 7 + lane * 13) as u8;
    }
    for pair in 0..4 {
        for lane in 0..32 {
            let low = ((row * 3 + block_index * 5 + pair * 7 + lane) & 0x0f) as u8;
            let high = ((row * 11 + block_index * 3 + pair * 5 + lane * 7) & 0x0f) as u8;
            bytes[48 + pair * 32 + lane] = low | (high << 4);
        }
    }
    bytes
}

fn q6_k_block(row: usize, block_index: usize) -> [u8; Q6_K_BLOCK_BYTES] {
    let mut bytes = [0u8; Q6_K_BLOCK_BYTES];
    for (index, byte) in bytes[..128].iter_mut().enumerate() {
        *byte = (row * 19 + block_index * 23 + index * 29) as u8;
    }
    for (index, byte) in bytes[128..192].iter_mut().enumerate() {
        *byte = (row * 37 + block_index * 17 + index * 11) as u8;
    }
    for (index, byte) in bytes[192..208].iter_mut().enumerate() {
        let scale = ((row * 7 + block_index * 5 + index * 9) % 63) as i8 - 31;
        *byte = scale as u8;
    }
    bytes[208..210].copy_from_slice(&f16::from_f32(1.0 / 1024.0).to_le_bytes());
    bytes
}

fn weights(block_bytes: usize, make_block: impl Fn(usize, usize) -> Vec<u8>) -> Vec<u8> {
    let blocks_per_row = QWEN38_HIDDEN / QK_K;
    let mut weights = Vec::with_capacity(VOCAB * blocks_per_row * block_bytes);
    for row in 0..VOCAB {
        for block_index in 0..blocks_per_row {
            weights.extend_from_slice(&make_block(row, block_index));
        }
    }
    weights
}

fn q5_weights() -> Vec<u8> {
    weights(Q5_K_BLOCK_BYTES, |row, block| {
        q5_k_block(row, block).to_vec()
    })
}

fn q6_weights() -> Vec<u8> {
    weights(Q6_K_BLOCK_BYTES, |row, block| {
        q6_k_block(row, block).to_vec()
    })
}

fn owned_buffer(device: &MlxDevice, bytes: &[u8]) -> mlx_native::MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![VOCAB, QWEN38_HIDDEN])
        .expect("packed embedding buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("packed embedding bytes")
        .copy_from_slice(bytes);
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

fn assert_exact_rows(kind: GgmlType, packed: &[u8], block_bytes: usize) {
    let ids_host = [2u32, 0, 1, 2];
    let device = MlxDevice::new().expect("Metal device");
    let weight = owned_buffer(&device, packed);
    let ids = token_buffer(&device, &ids_host);
    let output = device
        .alloc_buffer(
            ids_host.len() * QWEN38_HIDDEN * 4,
            DType::F32,
            vec![ids_host.len(), QWEN38_HIDDEN],
        )
        .expect("embedding output");
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("embedding encoder");
    match kind {
        GgmlType::Q5_K => embedding_gather_q5_k(
            &mut encoder,
            &mut registry,
            &device,
            &weight,
            &ids,
            &output,
            &EmbeddingQ5KParams {
                vocab_size: VOCAB,
                embed_dim: QWEN38_HIDDEN,
                n_tokens: ids_host.len(),
            },
        ),
        GgmlType::Q6_K => embedding_gather_q6_k(
            &mut encoder,
            &mut registry,
            &device,
            &weight,
            &ids,
            &output,
            &EmbeddingQ6KParams {
                vocab_size: VOCAB,
                embed_dim: QWEN38_HIDDEN,
                n_tokens: ids_host.len(),
            },
        ),
        _ => unreachable!(),
    }
    .expect("encode native embedding gather");
    encoder
        .commit_and_wait_labeled("test.embedding.kquant")
        .expect("GPU completion");

    let row_bytes = QWEN38_HIDDEN / QK_K * block_bytes;
    let expected: Vec<Vec<f32>> = (0..VOCAB)
        .map(|row| {
            let mut values = vec![0.0; QWEN38_HIDDEN];
            mlx_native::gguf::test_only_dequantize(
                &packed[row * row_bytes..(row + 1) * row_bytes],
                kind,
                &mut values,
            )
            .expect("CPU dequantization");
            values
        })
        .collect();
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
                "{kind:?} token {token_index}, row {row}, column {column}"
            );
        }
    }
}

#[test]
fn qwen38_width_q5_k_gather_is_bit_exact() {
    assert_exact_rows(GgmlType::Q5_K, &q5_weights(), Q5_K_BLOCK_BYTES);
}

#[test]
fn qwen38_width_q6_k_gather_is_bit_exact() {
    assert_exact_rows(GgmlType::Q6_K, &q6_weights(), Q6_K_BLOCK_BYTES);
}

#[test]
fn invalid_token_fails_before_q5_k_encoding() {
    let packed = q5_weights();
    let device = MlxDevice::new().expect("Metal device");
    let weight = owned_buffer(&device, &packed);
    let ids = token_buffer(&device, &[VOCAB as u32]);
    let output = device
        .alloc_buffer(QWEN38_HIDDEN * 4, DType::F32, vec![1, QWEN38_HIDDEN])
        .expect("embedding output");
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("embedding encoder");
    encoder.start_capture();
    let error = embedding_gather_q5_k(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &EmbeddingQ5KParams {
            vocab_size: VOCAB,
            embed_dim: QWEN38_HIDDEN,
            n_tokens: 1,
        },
    )
    .expect_err("invalid token ID must fail");
    assert!(error.to_string().contains("token_ids[0]=3"), "{error}");
    assert!(
        encoder
            .take_capture()
            .expect("invalid-token capture")
            .is_empty(),
        "validation failure must encode no work"
    );
}
