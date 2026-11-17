use super::*;

pub(super) fn run(device: &MlxDevice, registry: &mut KernelRegistry) -> Value {
    let base = GgmlQuantizedMatmulIdParams {
        n_tokens: 1, top_k: 1, n: 64, k: 256, n_experts: 2,
        expert_stride: 64 * 144 + 1, ggml_type: GgmlType::Q4_K,
    };
    let matrix_bytes = 64 * 144;
    let mut source = alloc(device, base.expert_stride as usize + matrix_bytes, DType::U8);
    for expert in 0..2 {
        let offset = expert * base.expert_stride as usize;
        let bytes = source.as_mut_slice::<u8>().expect("odd-stride source");
        for (index, block) in bytes[offset..offset + matrix_bytes].chunks_exact_mut(144).enumerate() {
            for (column, value) in block.iter_mut().enumerate() {
                *value = (index * 37 + column * 19 + expert * 11) as u8;
            }
            block[..2].copy_from_slice(&0x1c01u16.to_le_bytes());
            block[2..4].copy_from_slice(&0x1401u16.to_le_bytes());
        }
    }
    let raw = bind_quantized_matmul_id_ggml_grouped_m8_raw(&source, &base).expect("odd raw stride");
    let mut encoder = device.command_encoder().expect("odd-stride preparation");
    let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(
        &mut encoder, registry, device, &source, &base).expect("odd-stride comparator");
    encoder.commit_and_wait().expect("checked preparation");
    let mut cases = Vec::new();
    for rows in [2, 3, 9] {
        let p = GgmlQuantizedMatmulIdParams { n_tokens: rows, ..base };
        let mut input = alloc(device, rows as usize * 256, DType::F32);
        for (index, value) in input.as_mut_slice::<f32>().expect("odd-stride input").iter_mut().enumerate() {
            *value = ((index % 97) as f32 - 48.0) * 0.003;
        }
        let mut ids = alloc(device, rows as usize, DType::U32);
        ids.as_mut_slice::<u32>().expect("odd-stride routes").fill(1);
        let gathered = alloc(device, grouped_m8_gathered_rows(rows, 1, 2).expect("guarded rows") * 256, DType::BF16);
        let mut outputs = [alloc(device, rows as usize * 64, DType::F32), alloc(device, rows as usize * 64, DType::F32)];
        for output in &mut outputs {
            output.as_mut_slice::<f32>().expect("odd-stride output poison").fill(f32::from_bits(F32_POISON));
        }
        let mut scratch = IdMmScratch::alloc(device, 2, rows).expect("odd-stride schedule");
        let mut encoder = device.command_encoder().expect("odd-stride projection");
        for (arm, weight) in [&prepared, &raw].into_iter().enumerate() {
            quantized_matmul_id_ggml_grouped_m8_epilogue(
                &mut encoder, registry, device, &input, &gathered, weight, &ids, &outputs[arm],
                None, &mut scratch, &p, false, arm == 1, arm == 1, GroupedM8Input::F32,
                GroupedM8Epilogue::F32).expect("odd-stride projection dispatch");
        }
        encoder.commit_and_wait().expect("checked odd-stride projection");
        let a = outputs[0].as_slice::<f32>().expect("prepared odd-stride output");
        let b = outputs[1].as_slice::<f32>().expect("raw odd-stride output");
        assert!(a.iter().any(|value| *value != 0.0), "trivial odd-stride fixture");
        assert!(a.iter().chain(b).all(|value| value.is_finite()));
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()), "odd-stride bit mismatch");
        cases.push(json!({"rows":rows,"compared_elements":a.len(),"mismatch_elements":0,
            "source_expert_stride_bytes":base.expert_stride,"routed_expert":1}));
    }
    json!({"case_count":cases.len(),"cases":cases})
}
