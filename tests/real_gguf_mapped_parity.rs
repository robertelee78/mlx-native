//! Real-artifact parity checks for file-backed Metal weights.

#![cfg(target_vendor = "apple")]

use std::path::Path;

use mlx_native::ops::embedding_q8_0::{embedding_gather_q8_0, EmbeddingQ8_0Params};
use mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml;
use mlx_native::{DType, GgmlQuantizedMatmulParams, GgmlType, GgufFile, KernelRegistry, MlxDevice};

#[test]
#[ignore = "requires HF2Q_DEEPSEEK4_GGUF pointing to the local DeepSeek-V4 artifact"]
fn real_q8_embedding_mapped_matches_owned_buffer() {
    let path = std::env::var("HF2Q_DEEPSEEK4_GGUF").expect("set HF2Q_DEEPSEEK4_GGUF");
    let gguf = GgufFile::open(Path::new(&path)).expect("open real DeepSeek GGUF");
    let info = gguf
        .tensor_info("token_embd.weight")
        .expect("token embedding info");
    assert_eq!(info.ggml_type, GgmlType::Q8_0);
    assert_eq!(info.shape.len(), 2);

    let vocab = info.shape[0];
    let embed_dim = info.shape[1];
    let device = MlxDevice::new().expect("Metal device");
    let mapped_set = gguf
        .map_tensor_data(&device)
        .expect("map shared GGUF tensor segments");
    let mapped = mapped_set
        .load_tensor("token_embd.weight")
        .expect("map token embedding");
    let owned = gguf
        .load_tensor("token_embd.weight", &device)
        .expect("copy token embedding");
    assert!(mapped.is_file_backed());
    assert!(!owned.is_file_backed());
    assert_eq!(mapped.data_byte_len(), owned.data_byte_len());

    let ids_host = [0u32, 1, 42, (vocab - 1) as u32];
    let mut ids = device
        .alloc_buffer(ids_host.len() * 4, DType::U32, vec![ids_host.len()])
        .expect("allocate token ids");
    ids.as_mut_slice::<u32>()
        .expect("token id slice")
        .copy_from_slice(&ids_host);
    let mapped_out = device
        .alloc_buffer(
            ids_host.len() * embed_dim * 4,
            DType::F32,
            vec![ids_host.len(), embed_dim],
        )
        .expect("allocate mapped output");
    let owned_out = device
        .alloc_buffer(
            ids_host.len() * embed_dim * 4,
            DType::F32,
            vec![ids_host.len(), embed_dim],
        )
        .expect("allocate owned output");
    let params = EmbeddingQ8_0Params {
        vocab_size: vocab,
        embed_dim,
        n_tokens: ids_host.len(),
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("command encoder");
    embedding_gather_q8_0(
        &mut encoder,
        &mut registry,
        &device,
        &mapped,
        &ids,
        &mapped_out,
        &params,
    )
    .expect("mapped embedding dispatch");
    embedding_gather_q8_0(
        &mut encoder,
        &mut registry,
        &device,
        &owned,
        &ids,
        &owned_out,
        &params,
    )
    .expect("owned embedding dispatch");
    encoder.commit_and_wait().expect("embedding completion");

    let mapped_values = mapped_out.as_slice::<f32>().expect("mapped output slice");
    let owned_values = owned_out.as_slice::<f32>().expect("owned output slice");
    assert_eq!(mapped_values.len(), owned_values.len());
    for (index, (&actual, &expected)) in mapped_values.iter().zip(owned_values).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "embedding value {index}"
        );
    }
}

#[test]
#[ignore = "requires HF2Q_DEEPSEEK4_GGUF pointing to the local DeepSeek-V4 artifact"]
fn real_layer0_attention_matrices_match_owned_buffers() {
    let path = std::env::var("HF2Q_DEEPSEEK4_GGUF").expect("set HF2Q_DEEPSEEK4_GGUF");
    let gguf = GgufFile::open(Path::new(&path)).expect("open real DeepSeek GGUF");
    let device = MlxDevice::new().expect("Metal device");
    eprintln!(
        "Metal max buffer length: {}",
        device.metal_device().max_buffer_length()
    );
    let mapped_set = gguf
        .map_tensor_data(&device)
        .expect("map shared GGUF tensor segments");
    eprintln!("GGUF mapped segments: {}", mapped_set.segment_count());
    let mut registry = KernelRegistry::new();
    let names = [
        "blk.0.hc_attn_fn.weight",
        "blk.0.attn_q_a.weight",
        "blk.0.attn_q_b.weight",
        "blk.0.attn_kv.weight",
        "blk.0.attn_output_a.weight",
        "blk.0.attn_output_b.weight",
    ];

    for name in names {
        let info = gguf.tensor_info(name).expect("layer-0 tensor info");
        let [n, k]: [usize; 2] = info
            .shape
            .as_slice()
            .try_into()
            .expect("layer-0 matrix shape");
        assert!(
            matches!(
                info.ggml_type,
                GgmlType::Q2_K
                    | GgmlType::Q3_K
                    | GgmlType::Q4_K
                    | GgmlType::Q5_K
                    | GgmlType::Q6_K
                    | GgmlType::Q8_0
                    | GgmlType::Q4_0
            ),
            "unsupported test type {:?} for {name}",
            info.ggml_type
        );
        let mapped = mapped_set
            .load_tensor(name)
            .unwrap_or_else(|error| panic!("map {name}: {error}"));
        let owned = gguf
            .load_tensor(name, &device)
            .unwrap_or_else(|error| panic!("copy {name}: {error}"));
        let rows = 5usize;
        let input_values: Vec<f32> = (0..rows * k)
            .map(|index| ((index * 17 + 3) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let mut input = device
            .alloc_buffer(input_values.len() * 4, DType::F32, vec![rows, k])
            .expect("allocate input");
        input
            .as_mut_slice::<f32>()
            .expect("input slice")
            .copy_from_slice(&input_values);
        let mapped_out = device
            .alloc_buffer(rows * n * 4, DType::F32, vec![rows, n])
            .expect("allocate mapped output");
        let owned_out = device
            .alloc_buffer(rows * n * 4, DType::F32, vec![rows, n])
            .expect("allocate owned output");
        let params = GgmlQuantizedMatmulParams {
            m: rows as u32,
            n: n as u32,
            k: k as u32,
            ggml_type: info.ggml_type,
        };
        let mut encoder = device.command_encoder().expect("command encoder");
        quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &mapped,
            &mapped_out,
            &params,
        )
        .unwrap_or_else(|error| panic!("mapped matmul {name}: {error}"));
        quantized_matmul_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &owned,
            &owned_out,
            &params,
        )
        .unwrap_or_else(|error| panic!("owned matmul {name}: {error}"));
        encoder.commit_and_wait().expect("matmul completion");

        let mapped_values = mapped_out.as_slice::<f32>().expect("mapped output slice");
        let owned_values = owned_out.as_slice::<f32>().expect("owned output slice");
        for (index, (&actual, &expected)) in mapped_values.iter().zip(owned_values).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{name} output {index}; type {:?}, shape {:?}",
                info.ggml_type,
                info.shape
            );
        }
        eprintln!(
            "mapped parity: {name} {:?} {:?}",
            info.ggml_type, info.shape
        );
    }
}
