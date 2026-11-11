use half::bf16;
use mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml_mv;
use mlx_native::{
    ggml_expert_bytes, prepare_quantized_matmul_id_ggml_grouped_m8,
    quantized_matmul_id_ggml_grouped_m8, quantized_matmul_id_ggml_grouped_m8_register,
    quantized_matmul_id_ggml_grouped_mv,
    quantized_matmul_id_ggml_grouped_native_m8, DType, GgmlQuantizedMatmulIdParams, GgmlType,
    GroupedM8Epilogue, GroupedM8Input, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn value(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f32_buffer(device: &MlxDevice, len: usize, seed: u64) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("f32");
    let mut rng = Rng(seed);
    for value in buffer.as_mut_slice::<f32>().expect("f32 slice") {
        *value = rng.value();
    }
    buffer
}

fn ids(device: &MlxDevice, tokens: usize, experts: usize) -> MlxBuffer {
    let values: Vec<u32> = (0..tokens)
        .flat_map(|token| (0..experts).map(move |slot| ((slot + token) % experts) as u32))
        .collect();
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("ids");
    buffer
        .as_mut_slice::<u32>()
        .expect("ids slice")
        .copy_from_slice(&values);
    buffer
}

fn weights(
    device: &MlxDevice,
    ggml_type: GgmlType,
    experts: u32,
    n: u32,
    k: u32,
    seed: u64,
) -> (MlxBuffer, u64) {
    let stride =
        u64::from(n) * u64::from(k / ggml_type.block_values()) * u64::from(ggml_type.block_bytes());
    let bytes = ggml_expert_bytes(ggml_type, experts, n, k, stride).expect("weight bytes") as usize;
    let mut buffer = device
        .alloc_buffer(bytes, DType::U8, vec![bytes])
        .expect("weights");
    let data = buffer.as_mut_slice::<u8>().expect("weight slice");
    let mut rng = Rng(seed);
    for chunk in data.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next().to_le_bytes()[..chunk.len()]);
    }
    let scale_offsets: &[usize] = match ggml_type {
        GgmlType::Q4_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        _ => unreachable!(),
    };
    for block in data.chunks_exact_mut(ggml_type.block_bytes() as usize) {
        for (index, offset) in scale_offsets.iter().copied().enumerate() {
            let sign = if index == 0 && rng.next() & 1 != 0 {
                0x8000
            } else {
                0
            };
            let bits = sign | (7 << 10) | ((rng.next() >> 8) as u16 & 0x03ff);
            block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    (buffer, stride)
}

fn assert_bitwise(grouped: &MlxBuffer, reference: &MlxBuffer, elements: usize, label: &str) {
    let grouped = &grouped.as_slice::<f32>().expect("grouped output")[..elements];
    let reference = &reference.as_slice::<f32>().expect("reference output")[..elements];
    for (index, (&actual, &expected)) in grouped.iter().zip(reference).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} element {index}: {actual} != {expected}"
        );
    }
}

fn assert_tensor_close(candidate: &MlxBuffer, reference: &MlxBuffer, elements: usize, label: &str) {
    let candidate = &candidate.as_slice::<f32>().expect("candidate output")[..elements];
    let reference = &reference.as_slice::<f32>().expect("reference output")[..elements];
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    let mut maximum_error = 0.0f32;
    let mut reference_maximum = 0.0f32;
    for (&actual, &expected) in candidate.iter().zip(reference) {
        let error = (actual - expected).abs();
        maximum_error = maximum_error.max(error);
        reference_maximum = reference_maximum.max(expected.abs());
        squared_error += f64::from(error) * f64::from(error);
        squared_reference += f64::from(expected) * f64::from(expected);
    }
    let relative_rmse = (squared_error / squared_reference).sqrt();
    let normalized_maximum = maximum_error / reference_maximum;
    assert!(
        relative_rmse < 3.0e-3 && normalized_maximum < 3.0e-3,
        "{label}: relative RMSE {relative_rmse:e}, normalized maximum {normalized_maximum:e}"
    );
}

fn grouped_matches_mv(ggml_type: GgmlType) {
    const TOKENS: u32 = 11;
    const EXPERTS: u32 = 8;
    const TOP_K: u32 = 8;
    const K: u32 = 256;
    const N: u32 = 64;
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let ids = ids(&device, TOKENS as usize, EXPERTS as usize);
    let input = f32_buffer(&device, (TOKENS * K) as usize, 11);
    let slotted = f32_buffer(&device, (TOKENS * TOP_K * K) as usize, 12);
    let (weight, stride) = weights(&device, ggml_type, EXPERTS, N, K, 13);
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS,
        top_k: TOP_K,
        n: N,
        k: K,
        n_experts: EXPERTS,
        expert_stride: stride,
        ggml_type,
    };
    let elements = (TOKENS * TOP_K * N) as usize;
    let mut grouped = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("grouped");
    let mut reference = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("reference");
    let mut scratch = IdMmScratch::alloc(&device, EXPERTS, TOKENS).expect("scratch");
    let table_rows = EXPERTS * 8 + TOKENS * TOP_K + 7;
    let gathered_elements = (table_rows * K) as usize;
    let gathered = device
        .alloc_buffer(gathered_elements * 2, DType::BF16, vec![gathered_elements])
        .expect("gathered input");
    let gathered_sum_elements = (table_rows * K * 3 / 32) as usize;
    let gathered_sums = device
        .alloc_buffer(
            gathered_sum_elements * 4,
            DType::F32,
            vec![gathered_sum_elements],
        )
        .expect("Table16 sums");
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &grouped,
        &mut scratch,
        &params,
        false,
        false,
    )
    .expect("grouped shared-input projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &reference,
        &params,
    )
    .expect("reference shared-input projection");
    encoder
        .commit_and_wait_labeled("test.grouped_mv.shared")
        .expect("commit shared");
    assert_bitwise(&grouped, &reference, elements, "shared input");

    grouped
        .as_mut_slice::<u32>()
        .expect("grouped words")
        .fill(0x7fc0_0011);
    reference
        .as_mut_slice::<u32>()
        .expect("reference words")
        .fill(0x7fc0_0012);
    let slotted_params = params;
    let flat_params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS * TOP_K,
        top_k: 1,
        ..params
    };
    let prep_output = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("schedule preparation output");
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &prep_output,
        &mut scratch,
        &params,
        false,
        false,
    )
    .expect("prepare grouped schedule");
    quantized_matmul_id_ggml_grouped_mv(
        &mut encoder,
        &mut registry,
        &device,
        &slotted,
        &weight,
        &ids,
        &grouped,
        &mut scratch,
        &slotted_params,
        true,
        true,
    )
    .expect("grouped slotted projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &slotted,
        &weight,
        &ids,
        &reference,
        &flat_params,
    )
    .expect("reference slotted projection");
    encoder
        .commit_and_wait_labeled("test.grouped_mv.slotted")
        .expect("commit slotted");
    assert_bitwise(&grouped, &reference, elements, "slotted input");

    grouped
        .as_mut_slice::<u32>()
        .expect("grouped words")
        .fill(0x7fc0_0011);
    reference
        .as_mut_slice::<u32>()
        .expect("reference words")
        .fill(0x7fc0_0012);
    let mut encoder = device.command_encoder().expect("encoder");
    let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &weight,
        &params,
    )
    .expect("prepare grouped M8 experts");
    quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &prepared,
        &ids,
        &grouped,
        &mut scratch,
        &params,
        false,
        false,
        false,
    )
    .expect("grouped M8 shared-input projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &reference,
        &params,
    )
    .expect("reference M8 shared-input projection");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.shared")
        .expect("commit shared M8");
    assert_tensor_close(&grouped, &reference, elements, "M8 shared input");
    grouped
        .as_mut_slice::<u32>()
        .expect("grouped words")
        .fill(0x7fc0_0011);
    reference
        .as_mut_slice::<u32>()
        .expect("reference words")
        .fill(0x7fc0_0012);
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_m8_register(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &gathered_sums,
        &prepared,
        &ids,
        &grouped,
        None,
        &mut scratch,
        &params,
        false,
        false,
        false,
        GroupedM8Input::F32,
        GroupedM8Epilogue::F32,
    )
    .expect("Table16 register shared-input projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &reference,
        &params,
    )
    .expect("reference Table16 shared-input projection");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.register.shared")
        .expect("commit Table16 shared M8");
    assert_tensor_close(
        &grouped,
        &reference,
        elements,
        "Table16 register shared input",
    );

    grouped
        .as_mut_slice::<u32>()
        .expect("grouped words")
        .fill(0x7fc0_0011);
    reference
        .as_mut_slice::<u32>()
        .expect("reference words")
        .fill(0x7fc0_0012);
    let prep_output = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("M8 schedule preparation output");
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &prepared,
        &ids,
        &prep_output,
        &mut scratch,
        &params,
        false,
        false,
        false,
    )
    .expect("prepare grouped M8 schedule");
    quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &slotted,
        &gathered,
        &prepared,
        &ids,
        &grouped,
        &mut scratch,
        &slotted_params,
        true,
        true,
        false,
    )
    .expect("grouped M8 slotted projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &slotted,
        &weight,
        &ids,
        &reference,
        &flat_params,
    )
    .expect("reference M8 slotted projection");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.slotted")
        .expect("commit slotted M8");
    assert_tensor_close(&grouped, &reference, elements, "M8 slotted input");

    grouped
        .as_mut_slice::<u32>()
        .expect("grouped words")
        .fill(0x7fc0_0011);
    reference
        .as_mut_slice::<u32>()
        .expect("reference words")
        .fill(0x7fc0_0012);
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_native_m8(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &grouped,
        &mut scratch,
        &params,
        false,
        false,
    )
    .expect("native M8 shared-input projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &reference,
        &params,
    )
    .expect("reference native M8 shared-input projection");
    encoder
        .commit_and_wait_labeled("test.grouped_native_m8.shared")
        .expect("commit native shared M8");
    assert_tensor_close(&grouped, &reference, elements, "native M8 shared input");

    grouped
        .as_mut_slice::<u32>()
        .expect("grouped words")
        .fill(0x7fc0_0011);
    reference
        .as_mut_slice::<u32>()
        .expect("reference words")
        .fill(0x7fc0_0012);
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_native_m8(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &ids,
        &prep_output,
        &mut scratch,
        &params,
        false,
        false,
    )
    .expect("prepare native M8 schedule");
    quantized_matmul_id_ggml_grouped_native_m8(
        &mut encoder,
        &mut registry,
        &device,
        &slotted,
        &weight,
        &ids,
        &grouped,
        &mut scratch,
        &slotted_params,
        true,
        true,
    )
    .expect("native M8 slotted projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &slotted,
        &weight,
        &ids,
        &reference,
        &flat_params,
    )
    .expect("reference native M8 slotted projection");
    encoder
        .commit_and_wait_labeled("test.grouped_native_m8.slotted")
        .expect("commit native slotted M8");
    assert_tensor_close(&grouped, &reference, elements, "native M8 slotted input");

    let gate_bf16 = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("BF16 gate");
    let fused_bf16 = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("BF16 fused output");
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_m8_register(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &gathered_sums,
        &prepared,
        &ids,
        &gate_bf16,
        None,
        &mut scratch,
        &params,
        false,
        false,
        false,
        GroupedM8Input::F32,
        GroupedM8Epilogue::Bf16,
    )
    .expect("BF16 gate projection");
    quantized_matmul_id_ggml_grouped_m8_register(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &gathered_sums,
        &prepared,
        &ids,
        &fused_bf16,
        Some(&gate_bf16),
        &mut scratch,
        &params,
        false,
        true,
        true,
        GroupedM8Input::F32,
        GroupedM8Epilogue::SiluMulBf16,
    )
    .expect("fused BF16 up projection");
    quantized_matmul_id_ggml_grouped_m8_register(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &gathered_sums,
        &prepared,
        &ids,
        &reference,
        None,
        &mut scratch,
        &params,
        false,
        true,
        true,
        GroupedM8Input::F32,
        GroupedM8Epilogue::F32,
    )
    .expect("F32 up reference");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.fused_silu")
        .expect("commit fused BF16 projection");
    let gates = gate_bf16.as_slice::<u16>().expect("BF16 gates");
    let fused = fused_bf16.as_slice::<u16>().expect("BF16 fused output");
    let up = reference.as_slice::<f32>().expect("F32 up output");
    for index in 0..elements {
        let gate = bf16::from_bits(gates[index]).to_f32();
        let expected = bf16::from_f32((gate / (1.0 + (-gate).exp())) * up[index]).to_f32();
        let actual = bf16::from_bits(fused[index]).to_f32();
        let tolerance = 1.0e-5f32.max(expected.abs() * 1.0e-2);
        assert!(
            (actual - expected).abs() <= tolerance,
            "fused BF16 SiLU element {index}: {actual} != {expected} (tol {tolerance})"
        );
    }
    let mut slotted_bf16 = device
        .alloc_buffer(
            (TOKENS * TOP_K * K * 2) as usize,
            DType::BF16,
            vec![(TOKENS * TOP_K * K) as usize],
        )
        .expect("BF16 slotted input");
    let mut slotted_bf16_f32 = device
        .alloc_buffer(
            (TOKENS * TOP_K * K * 4) as usize,
            DType::F32,
            vec![(TOKENS * TOP_K * K) as usize],
        )
        .expect("widened BF16 slotted input");
    let source = slotted.as_slice::<f32>().expect("slotted source");
    let bf16_words = slotted_bf16
        .as_mut_slice::<u16>()
        .expect("BF16 slotted words");
    let widened = slotted_bf16_f32
        .as_mut_slice::<f32>()
        .expect("widened BF16 slotted values");
    for ((word, wide), value) in bf16_words.iter_mut().zip(widened).zip(source) {
        let rounded = bf16::from_f32(*value);
        *word = rounded.to_bits();
        *wide = rounded.to_f32();
    }
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_m8_register(
        &mut encoder,
        &mut registry,
        &device,
        &slotted_bf16,
        &gathered,
        &gathered_sums,
        &prepared,
        &ids,
        &grouped,
        None,
        &mut scratch,
        &params,
        true,
        false,
        false,
        GroupedM8Input::Bf16,
        GroupedM8Epilogue::F32,
    )
    .expect("BF16-input down projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &slotted_bf16_f32,
        &weight,
        &ids,
        &reference,
        &flat_params,
    )
    .expect("widened BF16 down reference");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.bf16_input")
        .expect("commit BF16-input projection");
    assert_tensor_close(&grouped, &reference, elements, "BF16-input down");
}

#[test]
#[ignore = "single-layer Qwen3.6 performance probe"]
fn benchmark_qwen36_gate_projection() {
    use std::time::Instant;

    const TOKENS: u32 = 18;
    const EXPERTS: u32 = 128;
    const TOP_K: u32 = 8;
    const K: u32 = 5120;
    const N: u32 = 1536;
    const REPEATS: usize = 100;

    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let ids = {
        let values: Vec<u32> = (0..TOKENS)
            .flat_map(|token| (0..TOP_K).map(move |slot| (token * 13 + slot * 17) % EXPERTS))
            .collect();
        let mut buffer = device
            .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
            .expect("ids");
        buffer
            .as_mut_slice::<u32>()
            .expect("ids slice")
            .copy_from_slice(&values);
        buffer
    };
    let input = f32_buffer(&device, (TOKENS * K) as usize, 101);
    let (weight, stride) = weights(&device, GgmlType::Q4_K, EXPERTS, N, K, 102);
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS,
        top_k: TOP_K,
        n: N,
        k: K,
        n_experts: EXPERTS,
        expert_stride: stride,
        ggml_type: GgmlType::Q4_K,
    };
    let elements = (TOKENS * TOP_K * N) as usize;
    let output = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("output");
    let table_rows = EXPERTS * 8 + TOKENS * TOP_K + 7;
    let gathered_elements = (table_rows * K) as usize;
    let gathered = device
        .alloc_buffer(gathered_elements * 2, DType::BF16, vec![gathered_elements])
        .expect("gathered input");
    let gathered_sum_elements = (table_rows * K * 3 / 32) as usize;
    let gathered_sums = device
        .alloc_buffer(
            gathered_sum_elements * 4,
            DType::F32,
            vec![gathered_sum_elements],
        )
        .expect("Table16 sums");

    let prepared = {
        let mut encoder = device.command_encoder().expect("encoder");
        let prepared = prepare_quantized_matmul_id_ggml_grouped_m8(
            &mut encoder,
            &mut registry,
            &device,
            &weight,
            &params,
        )
        .expect("prepare");
        encoder
            .commit_and_wait_labeled("bench.prepare")
            .expect("commit prepare");
        prepared
    };

    for route in ["scalar", "native_m8", "prepared_m8", "register_m8"] {
        let mut scratch = IdMmScratch::alloc(&device, EXPERTS, TOKENS).expect("scratch");
        let mut encoder = device.command_encoder().expect("encoder");
        for repeat in 0..REPEATS {
            match route {
                "scalar" => quantized_matmul_id_ggml_grouped_mv(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &input,
                    &weight,
                    &ids,
                    &output,
                    &mut scratch,
                    &params,
                    false,
                    repeat != 0,
                ),
                "native_m8" => quantized_matmul_id_ggml_grouped_native_m8(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &input,
                    &weight,
                    &ids,
                    &output,
                    &mut scratch,
                    &params,
                    false,
                    repeat != 0,
                ),
                "prepared_m8" => quantized_matmul_id_ggml_grouped_m8(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &input,
                    &gathered,
                    &prepared,
                    &ids,
                    &output,
                    &mut scratch,
                    &params,
                    false,
                    repeat != 0,
                    repeat != 0,
                ),
                "register_m8" => quantized_matmul_id_ggml_grouped_m8_register(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &input,
                    &gathered,
                    &gathered_sums,
                    &prepared,
                    &ids,
                    &output,
                    None,
                    &mut scratch,
                    &params,
                    false,
                    repeat != 0,
                    repeat != 0,
                    GroupedM8Input::F32,
                    GroupedM8Epilogue::F32,
                ),
                _ => unreachable!(),
            }
            .unwrap_or_else(|error| panic!("{route} repeat {repeat}: {error}"));
        }
        let started = Instant::now();
        encoder
            .commit_and_wait_labeled("bench.grouped_expert")
            .expect("commit benchmark");
        let elapsed = started.elapsed();
        eprintln!(
            "{route}: {:.3} ms/dispatch ({REPEATS} dispatches)",
            elapsed.as_secs_f64() * 1000.0 / REPEATS as f64
        );
    }
}
#[test]
fn grouped_q4_k_matches_mv() {
    grouped_matches_mv(GgmlType::Q4_K);
}

#[test]
fn grouped_q6_k_matches_mv() {
    grouped_matches_mv(GgmlType::Q6_K);
}
