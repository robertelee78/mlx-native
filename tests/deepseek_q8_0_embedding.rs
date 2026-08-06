use half::f16;
use mlx_native::{
    embedding_gather_q8_0, DType, EmbeddingQ8_0Params, KernelRegistry, MlxDevice,
};

const QK8_0: usize = 32;
const BLOCK_BYTES: usize = 34;

fn block(scale: f32, seed: i8) -> ([u8; BLOCK_BYTES], [f32; QK8_0]) {
    let mut bytes = [0u8; BLOCK_BYTES];
    bytes[..2].copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
    let mut expected = [0.0f32; QK8_0];
    let actual_scale = f16::from_f32(scale).to_f32();
    for (column, (byte, value)) in bytes[2..].iter_mut().zip(expected.iter_mut()).enumerate() {
        let quant = seed.wrapping_add(column as i8).wrapping_sub(16);
        *byte = quant as u8;
        *value = actual_scale * quant as f32;
    }
    (bytes, expected)
}

#[test]
fn q8_0_embedding_gather_matches_rows_and_rejects_bad_ids() {
    let vocab = 3usize;
    let mut weights = Vec::with_capacity(vocab * BLOCK_BYTES);
    let mut decoded = Vec::with_capacity(vocab);
    for row in 0..vocab {
        let (bytes, expected) = block(0.125 * (row + 1) as f32, row as i8 * 7);
        weights.extend_from_slice(&bytes);
        decoded.push(expected);
    }

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::embedding_q8_0::register(&mut registry);
    let mut weight = device
        .alloc_buffer(weights.len(), DType::U8, vec![vocab, QK8_0])
        .expect("weight");
    weight
        .as_mut_slice::<u8>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut ids = device
        .alloc_buffer(2 * DType::U32.size_of(), DType::U32, vec![2])
        .expect("ids");
    ids.as_mut_slice::<u32>()
        .expect("id slice")
        .copy_from_slice(&[2, 0]);
    let output = device
        .alloc_buffer(2 * QK8_0 * DType::F32.size_of(), DType::F32, vec![2, QK8_0])
        .expect("output");
    let params = EmbeddingQ8_0Params {
        vocab_size: vocab,
        embed_dim: QK8_0,
        n_tokens: 2,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    embedding_gather_q8_0(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &params,
    )
    .expect("Q8_0 embedding gather");
    encoder.commit_and_wait().expect("GPU completion");
    let actual = output.as_slice::<f32>().expect("output slice");
    for (actual_row, expected_row) in actual.chunks_exact(QK8_0).zip([&decoded[2], &decoded[0]]) {
        for (&got, &want) in actual_row.iter().zip(expected_row) {
            assert!((got - want).abs() <= 1e-6, "{got} != {want}");
        }
    }

    ids.as_mut_slice::<u32>().expect("id slice")[1] = vocab as u32;
    let mut encoder = device.command_encoder().expect("encoder");
    assert!(embedding_gather_q8_0(
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
fn q8_0_embedding_gather_rejects_malformed_layouts() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::embedding_q8_0::register(&mut registry);
    let weight = device
        .alloc_buffer(BLOCK_BYTES, DType::U8, vec![1, QK8_0])
        .expect("weight");
    let mut ids = device
        .alloc_buffer(DType::U32.size_of(), DType::U32, vec![1])
        .expect("ids");
    ids.as_mut_slice::<u32>().expect("ids slice")[0] = 0;
    let output = device
        .alloc_buffer(QK8_0 * DType::F32.size_of(), DType::F32, vec![1, QK8_0])
        .expect("output");
    let mut encoder = device.command_encoder().expect("encoder");
    assert!(embedding_gather_q8_0(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &EmbeddingQ8_0Params {
            vocab_size: 1,
            embed_dim: QK8_0 - 1,
            n_tokens: 1,
        },
    )
    .is_err());
}
