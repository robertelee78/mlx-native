//! Tests for the quantized embedding gather GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};
use mlx_native::ops::embedding::{embedding_gather, EmbeddingGatherParams};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

/// Helper: quantize f32 values to 4-bit with MLX affine format.
/// Returns (packed_u32_bytes, scale_bf16, bias_bf16) for a single group.
fn quantize_4bit_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n_bins = 15.0_f32; // 2^4 - 1
    let range = max_val - min_val;
    let scale = if range.abs() < 1e-10 { 1.0 } else { range / n_bins };
    let bias = min_val;

    let mut packed = Vec::new();
    // 8 values per uint32
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

/// Helper: quantize f32 values to 6-bit with MLX 3-byte triplet format.
fn quantize_6bit_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n_bins = 63.0_f32; // 2^6 - 1
    let range = max_val - min_val;
    let scale = if range.abs() < 1e-10 { 1.0 } else { range / n_bins };
    let bias = min_val;

    let mut packed = Vec::new();
    // 4 values per 3 bytes
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

/// Helper: convert f32 to bf16 bytes (2 bytes, little-endian).
fn f32_to_bf16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    // bf16 is the upper 16 bits of f32 with round-to-nearest-even
    let bf16_bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
    bf16_bits.to_le_bytes()
}

#[test]
fn test_embedding_gather_4bit_basic() {
    let (device, mut registry) = setup();

    // Create a small embedding table: 4 tokens, embed_dim=8, group_size=8
    let vocab_size = 4;
    let embed_dim = 8;
    let group_size = 8;

    // Generate known float values for the embedding table
    let mut float_table = vec![vec![0.0f32; embed_dim]; vocab_size];
    for i in 0..vocab_size {
        for j in 0..embed_dim {
            float_table[i][j] = (i as f32 * 10.0 + j as f32) * 0.1;
        }
    }

    // Quantize each row
    let mut all_packed = Vec::new();
    let mut all_scales = Vec::new();
    let mut all_biases = Vec::new();
    let mut expected_dequant = vec![vec![0.0f32; embed_dim]; vocab_size];

    for i in 0..vocab_size {
        let (packed, scale, bias) = quantize_4bit_group(&float_table[i]);
        all_packed.extend_from_slice(&packed);
        let bf16_scale = f32_to_bf16_bytes(scale);
        let bf16_bias = f32_to_bf16_bytes(bias);
        all_scales.extend_from_slice(&bf16_scale);
        all_biases.extend_from_slice(&bf16_bias);

        // Compute expected dequantized values
        for j in 0..embed_dim {
            let uint_val = ((float_table[i][j] - bias) / scale).round().clamp(0.0, 15.0);
            expected_dequant[i][j] = uint_val * scale + bias;
        }
    }

    // Create GPU buffers
    let packed_row_stride = embed_dim / 8; // uint32s per row
    let packed_bytes = all_packed.len();
    let mut weight_buf = device
        .alloc_buffer(packed_bytes, DType::U32, vec![vocab_size, packed_row_stride])
        .expect("weight");
    weight_buf.as_mut_slice::<u8>().expect("write weight").copy_from_slice(&all_packed);

    let n_groups = 1; // embed_dim / group_size = 1
    let scales_bytes = all_scales.len();
    let mut scales_buf = device
        .alloc_buffer(scales_bytes, DType::U16, vec![vocab_size, n_groups])
        .expect("scales");
    scales_buf.as_mut_slice::<u8>().expect("write scales").copy_from_slice(&all_scales);

    let biases_bytes = all_biases.len();
    let mut biases_buf = device
        .alloc_buffer(biases_bytes, DType::U16, vec![vocab_size, n_groups])
        .expect("biases");
    biases_buf.as_mut_slice::<u8>().expect("write biases").copy_from_slice(&all_biases);

    // Token IDs: look up tokens 2, 0, 3
    let token_ids: Vec<u32> = vec![2, 0, 3];
    let n_tokens = token_ids.len();
    let mut token_buf = device
        .alloc_buffer(n_tokens * 4, DType::U32, vec![n_tokens])
        .expect("tokens");
    token_buf.as_mut_slice::<u32>().expect("write tokens").copy_from_slice(&token_ids);

    let output_bytes = n_tokens * embed_dim * 4;
    let output_buf = device
        .alloc_buffer(output_bytes, DType::F32, vec![n_tokens, embed_dim])
        .expect("output");

    // Dispatch
    let mut encoder = device.command_encoder().expect("encoder");
    embedding_gather(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &weight_buf,
        &scales_buf,
        &biases_buf,
        &token_buf,
        &output_buf,
        &EmbeddingGatherParams {
            embed_dim,
            group_size,
            bits: 4,
            n_tokens,
        },
    )
    .expect("embedding_gather");
    encoder.commit_and_wait().expect("commit");

    // Verify
    let output: &[f32] = output_buf.as_slice().expect("read output");
    for (tok_idx, &token_id) in token_ids.iter().enumerate() {
        for j in 0..embed_dim {
            let actual = output[tok_idx * embed_dim + j];
            let expected = expected_dequant[token_id as usize][j];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-3,
                "4bit embedding mismatch: token_id={}, dim={}, expected={}, got={}, diff={}",
                token_id, j, expected, actual, diff
            );
        }
    }
}

#[test]
fn test_embedding_gather_6bit_basic() {
    let (device, mut registry) = setup();

    // 4 tokens, embed_dim=8, group_size=8
    let vocab_size = 4;
    let embed_dim = 8;
    let group_size = 8;

    let mut float_table = vec![vec![0.0f32; embed_dim]; vocab_size];
    for i in 0..vocab_size {
        for j in 0..embed_dim {
            float_table[i][j] = ((i as f32) - 2.0) * 0.3 + (j as f32) * 0.15;
        }
    }

    let mut all_packed = Vec::new();
    let mut all_scales = Vec::new();
    let mut all_biases = Vec::new();
    let mut expected_dequant = vec![vec![0.0f32; embed_dim]; vocab_size];

    for i in 0..vocab_size {
        let (packed, scale, bias) = quantize_6bit_group(&float_table[i]);
        all_packed.extend_from_slice(&packed);
        all_scales.extend_from_slice(&f32_to_bf16_bytes(scale));
        all_biases.extend_from_slice(&f32_to_bf16_bytes(bias));

        for j in 0..embed_dim {
            let uint_val = ((float_table[i][j] - bias) / scale).round().clamp(0.0, 63.0);
            expected_dequant[i][j] = uint_val * scale + bias;
        }
    }

    let packed_row_stride = embed_dim * 3 / 4; // bytes per row for 6-bit
    let mut weight_buf = device
        .alloc_buffer(all_packed.len(), DType::U8, vec![vocab_size, packed_row_stride])
        .expect("weight");
    weight_buf.as_mut_slice::<u8>().expect("w").copy_from_slice(&all_packed);

    let n_groups = 1;
    let mut scales_buf = device
        .alloc_buffer(all_scales.len(), DType::U16, vec![vocab_size, n_groups])
        .expect("scales");
    scales_buf.as_mut_slice::<u8>().expect("s").copy_from_slice(&all_scales);

    let mut biases_buf = device
        .alloc_buffer(all_biases.len(), DType::U16, vec![vocab_size, n_groups])
        .expect("biases");
    biases_buf.as_mut_slice::<u8>().expect("b").copy_from_slice(&all_biases);

    let token_ids: Vec<u32> = vec![1, 3, 0, 2];
    let n_tokens = token_ids.len();
    let mut token_buf = device
        .alloc_buffer(n_tokens * 4, DType::U32, vec![n_tokens])
        .expect("tokens");
    token_buf.as_mut_slice::<u32>().expect("t").copy_from_slice(&token_ids);

    let output_buf = device
        .alloc_buffer(n_tokens * embed_dim * 4, DType::F32, vec![n_tokens, embed_dim])
        .expect("output");

    let mut encoder = device.command_encoder().expect("encoder");
    embedding_gather(
        &mut encoder, &mut registry, device.metal_device(),
        &weight_buf, &scales_buf, &biases_buf, &token_buf, &output_buf,
        &EmbeddingGatherParams {
            embed_dim, group_size, bits: 6, n_tokens,
        },
    ).expect("embedding_gather");
    encoder.commit_and_wait().expect("commit");

    let output: &[f32] = output_buf.as_slice().expect("read");
    for (tok_idx, &token_id) in token_ids.iter().enumerate() {
        for j in 0..embed_dim {
            let actual = output[tok_idx * embed_dim + j];
            let expected = expected_dequant[token_id as usize][j];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-2,
                "6bit embedding mismatch: token_id={}, dim={}, expected={}, got={}, diff={}",
                token_id, j, expected, actual, diff
            );
        }
    }
}

// ---- Validation tests ----

#[test]
fn test_embedding_gather_invalid_bits() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(64, DType::U32, vec![16]).expect("buf");
    let out = device.alloc_buffer(64, DType::F32, vec![16]).expect("out");
    let mut encoder = device.command_encoder().expect("enc");

    let result = embedding_gather(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, &buf, &buf, &out,
        &EmbeddingGatherParams {
            embed_dim: 8, group_size: 8, bits: 5, n_tokens: 1,
        },
    );
    assert!(result.is_err(), "bits=5 should error");
}

#[test]
fn test_embedding_gather_zero_embed_dim() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(64, DType::U32, vec![16]).expect("buf");
    let out = device.alloc_buffer(64, DType::F32, vec![16]).expect("out");
    let mut encoder = device.command_encoder().expect("enc");

    let result = embedding_gather(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, &buf, &buf, &out,
        &EmbeddingGatherParams {
            embed_dim: 0, group_size: 8, bits: 4, n_tokens: 1,
        },
    );
    assert!(result.is_err(), "zero embed_dim should error");
}

#[test]
fn test_embedding_gather_group_not_divisible() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(64, DType::U32, vec![16]).expect("buf");
    let out = device.alloc_buffer(64, DType::F32, vec![16]).expect("out");
    let mut encoder = device.command_encoder().expect("enc");

    let result = embedding_gather(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, &buf, &buf, &out,
        &EmbeddingGatherParams {
            embed_dim: 10, group_size: 8, bits: 4, n_tokens: 1,
        },
    );
    assert!(result.is_err(), "embed_dim not divisible by group_size should error");
}
