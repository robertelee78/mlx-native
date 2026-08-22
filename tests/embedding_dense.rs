use half::{bf16, f16};
use mlx_native::{embedding_gather_dense, DType, EmbeddingDenseParams, KernelRegistry, MlxDevice};

fn run_dense(dtype: DType) {
    const VOCAB: usize = 4;
    const HIDDEN: usize = 7;
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::embedding_dense::register(&mut registry);
    let values: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|index| index as f32 * 0.25 - 2.0)
        .collect();
    let bytes: Vec<u8> = match dtype {
        DType::BF16 => values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_le_bytes())
            .collect(),
        DType::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect(),
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        other => panic!("unsupported fixture dtype {other:?}"),
    };
    let mut weight = device
        .alloc_buffer(bytes.len(), dtype, vec![VOCAB, HIDDEN])
        .unwrap();
    weight.as_mut_slice::<u8>().unwrap().copy_from_slice(&bytes);
    let ids_values = [3_u32, 1, 0];
    let mut ids = device
        .alloc_buffer(ids_values.len() * 4, DType::U32, vec![ids_values.len()])
        .unwrap();
    ids.as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids_values);
    let output = device
        .alloc_buffer(
            ids_values.len() * HIDDEN * 4,
            DType::F32,
            vec![ids_values.len(), HIDDEN],
        )
        .unwrap();
    let mut encoder = device.command_encoder().unwrap();
    embedding_gather_dense(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &EmbeddingDenseParams {
            vocab_size: VOCAB,
            embed_dim: HIDDEN,
            n_tokens: ids_values.len(),
        },
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();

    let actual = output.as_slice::<f32>().unwrap();
    for (row, token) in ids_values.into_iter().enumerate() {
        for column in 0..HIDDEN {
            let source = values[token as usize * HIDDEN + column];
            let expected = match dtype {
                DType::BF16 => bf16::from_f32(source).to_f32(),
                DType::F16 => f16::from_f32(source).to_f32(),
                DType::F32 => source,
                _ => unreachable!(),
            };
            assert_eq!(actual[row * HIDDEN + column], expected);
        }
    }
}

#[test]
fn dense_embedding_preserves_bf16_rows() {
    run_dense(DType::BF16);
}

#[test]
fn dense_embedding_preserves_f16_rows() {
    run_dense(DType::F16);
}

#[test]
fn dense_embedding_preserves_f32_rows() {
    run_dense(DType::F32);
}

#[test]
fn dense_embedding_rejects_out_of_range_token_before_encoding() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let weight = device.alloc_buffer(8, DType::BF16, vec![2, 2]).unwrap();
    let mut ids = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
    ids.as_mut_slice::<u32>().unwrap()[0] = 2;
    let output = device.alloc_buffer(8, DType::F32, vec![1, 2]).unwrap();
    let mut encoder = device.command_encoder().unwrap();
    let error = embedding_gather_dense(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &ids,
        &output,
        &EmbeddingDenseParams {
            vocab_size: 2,
            embed_dim: 2,
            n_tokens: 1,
        },
    )
    .expect_err("out-of-range token must fail closed");
    assert!(error.to_string().contains("exceeds vocabulary"));
}
