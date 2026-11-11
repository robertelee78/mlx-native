use half::bf16;
use mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml_mv;
use mlx_native::{
    ggml_expert_bytes, prepare_quantized_matmul_id_ggml_grouped_m8,
    prepare_quantized_matmul_id_ggml_grouped_m8_with_tail,
    quantized_matmul_id_ggml_grouped_m8_epilogue, quantized_matmul_id_ggml_grouped_m8_gate_up,
    DType, GgmlQuantizedMatmulIdParams, GgmlType, GroupedM8Epilogue, GroupedM8Input, IdMmScratch,
    KernelRegistry, MlxBuffer, MlxDevice,
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
        .expect("f32 buffer");
    let mut rng = Rng(seed);
    for value in buffer.as_mut_slice::<f32>().expect("f32 slice") {
        *value = rng.value();
    }
    buffer
}

fn ids(device: &MlxDevice, tokens: usize, top_k: usize, experts: usize) -> MlxBuffer {
    let values: Vec<u32> = (0..tokens)
        .flat_map(|token| (0..top_k).map(move |slot| ((slot * 3 + token * 5) % experts) as u32))
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
    const WIDTH: u32 = 256;

    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let ids = ids(&device, TOKENS as usize, TOP_K as usize, EXPERTS as usize);
    let input = f32_buffer(&device, (TOKENS * WIDTH) as usize, 11);
    let (gate_weight, stride) = weights(&device, ggml_type, EXPERTS, WIDTH, WIDTH, 13);
    let (up_weight, _) = weights(&device, ggml_type, EXPERTS, WIDTH, WIDTH, 14);
    let (down_weight, _) = weights(&device, ggml_type, EXPERTS, WIDTH, WIDTH, 15);
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS,
        top_k: TOP_K,
        n: WIDTH,
        k: WIDTH,
        n_experts: EXPERTS,
        expert_stride: stride,
        ggml_type,
    };
    let flat_params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS * TOP_K,
        top_k: 1,
        ..params
    };
    let elements = (TOKENS * TOP_K * WIDTH) as usize;
    let gathered_elements = ((TOKENS * TOP_K + EXPERTS) * WIDTH) as usize;
    let gathered = device
        .alloc_buffer(gathered_elements * 2, DType::BF16, vec![gathered_elements])
        .expect("gathered input");
    let gate_bf16 = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("gate BF16");
    let up_f32 = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("up F32");
    let up_reference = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("up reference");
    let one_pass_bf16 = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("one-pass gate/up");
    let down_f32 = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("down F32");
    let down_reference = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("down reference");
    let mut widened = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("widened intermediate");
    let mut scratch = IdMmScratch::alloc(&device, EXPERTS, TOKENS).expect("scratch");

    let mut encoder = device.command_encoder().expect("prepare encoder");
    let gate = prepare_quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &gate_weight,
        &params,
    )
    .expect("prepare gate");
    let up = prepare_quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &up_weight,
        &params,
    )
    .expect("prepare up");
    let down = prepare_quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &down_weight,
        &params,
    )
    .expect("prepare down");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.prepare")
        .expect("commit preparation");

    let undersized_gather = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("undersized gather");
    let mut validation_encoder = device.command_encoder().expect("validation encoder");
    let error = quantized_matmul_id_ggml_grouped_m8_epilogue(
        &mut validation_encoder,
        &mut registry,
        &device,
        &input,
        &undersized_gather,
        &gate,
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
    .expect_err("compact schedule guard rows must be reserved");
    assert!(
        error
            .to_string()
            .contains("gathered input buffer too small"),
        "unexpected guard-row validation error: {error}"
    );

    let mut encoder = device.command_encoder().expect("projection encoder");
    quantized_matmul_id_ggml_grouped_m8_epilogue(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &gate,
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
    .expect("staged gate");
    quantized_matmul_id_ggml_grouped_m8_epilogue(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &up,
        &ids,
        &up_f32,
        None,
        &mut scratch,
        &params,
        false,
        true,
        true,
        GroupedM8Input::F32,
        GroupedM8Epilogue::F32,
    )
    .expect("staged up");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &up_weight,
        &ids,
        &up_reference,
        &params,
    )
    .expect("up reference");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.projection")
        .expect("commit projections");
    assert_tensor_close(&up_f32, &up_reference, elements, "prepared projection");

    let mut scratch = IdMmScratch::alloc(&device, EXPERTS, TOKENS).expect("fused scratch");
    let mut encoder = device.command_encoder().expect("fused encoder");
    quantized_matmul_id_ggml_grouped_m8_gate_up(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &gate,
        &up,
        &ids,
        &one_pass_bf16,
        &mut scratch,
        &params,
        GroupedM8Input::F32,
    )
    .expect("one-pass gate/up");
    quantized_matmul_id_ggml_grouped_m8_epilogue(
        &mut encoder,
        &mut registry,
        &device,
        &one_pass_bf16,
        &gathered,
        &down,
        &ids,
        &down_f32,
        None,
        &mut scratch,
        &params,
        true,
        true,
        false,
        GroupedM8Input::Bf16,
        GroupedM8Epilogue::F32,
    )
    .expect("BF16-input down");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.gate_up_down")
        .expect("commit fused path");

    let gate_values = gate_bf16.as_slice::<u16>().expect("gate values");
    let up_values = up_f32.as_slice::<f32>().expect("up values");
    let fused_values = one_pass_bf16.as_slice::<u16>().expect("fused values");
    for index in 0..elements {
        let gate = bf16::from_bits(gate_values[index]).to_f32();
        let up = bf16::from_f32(up_values[index]).to_f32();
        let expected = bf16::from_f32((gate / (1.0 + (-gate).exp())) * up).to_f32();
        let actual = bf16::from_bits(fused_values[index]).to_f32();
        let tolerance = 1.0e-5f32.max(expected.abs() * 1.0e-2);
        assert!(
            (actual - expected).abs() <= tolerance,
            "one-pass gate/up element {index}: {actual} != {expected} (tol {tolerance})"
        );
    }

    for (wide, &word) in widened
        .as_mut_slice::<f32>()
        .expect("widened values")
        .iter_mut()
        .zip(fused_values)
    {
        *wide = bf16::from_bits(word).to_f32();
    }
    let mut encoder = device.command_encoder().expect("reference encoder");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &widened,
        &down_weight,
        &ids,
        &down_reference,
        &flat_params,
    )
    .expect("down reference");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.down_reference")
        .expect("commit down reference");
    assert_tensor_close(&down_f32, &down_reference, elements, "BF16-input down");
}

#[test]
fn grouped_q4_k_matches_mv() {
    grouped_matches_mv(GgmlType::Q4_K);
}

#[test]
fn grouped_q6_k_matches_mv() {
    grouped_matches_mv(GgmlType::Q6_K);
}

fn grouped_tail_matches_combined_mv(ggml_type: GgmlType) {
    const TOKENS: u32 = 5;
    const PRIMARY_EXPERTS: u32 = 8;
    const TAIL_EXPERTS: u32 = 1;
    const EXPERTS: u32 = PRIMARY_EXPERTS + TAIL_EXPERTS;
    const TOP_K: u32 = 9;
    const WIDTH: u32 = 256;

    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let input = f32_buffer(&device, (TOKENS * WIDTH) as usize, 201);
    let (primary_weight, stride) = weights(&device, ggml_type, PRIMARY_EXPERTS, WIDTH, WIDTH, 202);
    let (tail_weight, tail_stride) = weights(&device, ggml_type, TAIL_EXPERTS, WIDTH, WIDTH, 203);
    assert_eq!(stride, tail_stride);

    let primary_params = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS,
        top_k: TOP_K,
        n: WIDTH,
        k: WIDTH,
        n_experts: PRIMARY_EXPERTS,
        expert_stride: stride,
        ggml_type,
    };
    let tail_params = GgmlQuantizedMatmulIdParams {
        n_experts: TAIL_EXPERTS,
        expert_stride: tail_stride,
        ..primary_params
    };
    let combined_params = GgmlQuantizedMatmulIdParams {
        n_experts: EXPERTS,
        ..primary_params
    };

    let id_values: Vec<u32> = (0..TOKENS)
        .flat_map(|token| (0..TOP_K).map(move |slot| (slot + token) % EXPERTS))
        .collect();
    let mut expert_ids = device
        .alloc_buffer(
            id_values.len() * DType::U32.size_of(),
            DType::U32,
            vec![id_values.len()],
        )
        .expect("combined ids");
    expert_ids
        .as_mut_slice::<u32>()
        .expect("combined id slice")
        .copy_from_slice(&id_values);

    let combined_bytes = (u64::from(EXPERTS) * stride) as usize;
    let mut combined_weight = device
        .alloc_buffer(combined_bytes, DType::U8, vec![combined_bytes])
        .expect("combined weight");
    let primary_bytes = primary_weight.as_slice::<u8>().expect("primary bytes");
    let tail_bytes = tail_weight.as_slice::<u8>().expect("tail bytes");
    let combined = combined_weight
        .as_mut_slice::<u8>()
        .expect("combined bytes");
    combined[..primary_bytes.len()].copy_from_slice(primary_bytes);
    combined[primary_bytes.len()..primary_bytes.len() + tail_bytes.len()]
        .copy_from_slice(tail_bytes);

    let elements = (TOKENS * TOP_K * WIDTH) as usize;
    let gathered_elements = ((TOKENS * TOP_K + EXPERTS) * WIDTH) as usize;
    let gathered = device
        .alloc_buffer(
            gathered_elements * DType::BF16.size_of(),
            DType::BF16,
            vec![gathered_elements],
        )
        .expect("gathered input");
    let candidate = device
        .alloc_buffer(elements * DType::F32.size_of(), DType::F32, vec![elements])
        .expect("candidate");
    let reference = device
        .alloc_buffer(elements * DType::F32.size_of(), DType::F32, vec![elements])
        .expect("reference");
    let mut scratch =
        IdMmScratch::alloc_with_top_k(&device, EXPERTS, TOKENS, TOP_K).expect("scratch");

    let mut encoder = device.command_encoder().expect("encoder");
    let prepared = prepare_quantized_matmul_id_ggml_grouped_m8_with_tail(
        &mut encoder,
        &mut registry,
        &device,
        &primary_weight,
        &primary_params,
        &tail_weight,
        &tail_params,
    )
    .expect("prepare routed and shared experts");
    quantized_matmul_id_ggml_grouped_m8_epilogue(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &gathered,
        &prepared,
        &expert_ids,
        &candidate,
        None,
        &mut scratch,
        &combined_params,
        false,
        false,
        false,
        GroupedM8Input::F32,
        GroupedM8Epilogue::F32,
    )
    .expect("combined grouped projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &combined_weight,
        &expert_ids,
        &reference,
        &combined_params,
    )
    .expect("combined MV reference");
    encoder
        .commit_and_wait_labeled("test.grouped_m8.tail")
        .expect("commit tail parity");
    assert_tensor_close(&candidate, &reference, elements, "prepared tail projection");
}

#[test]
fn grouped_q4_k_tail_matches_combined_mv() {
    grouped_tail_matches_combined_mv(GgmlType::Q4_K);
}

#[test]
fn grouped_q6_k_tail_matches_combined_mv() {
    grouped_tail_matches_combined_mv(GgmlType::Q6_K);
}

#[test]
#[ignore = "single-layer Qwen3.6 performance probe"]
fn benchmark_qwen36_gate_projection() {
    use std::time::Instant;

    const TOKENS: u32 = 18;
    const EXPERTS: u32 = 256;
    const TOP_K: u32 = 8;
    const K: u32 = 2048;
    const N: u32 = 512;
    const REPEATS: usize = 100;

    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let ids = ids(&device, TOKENS as usize, TOP_K as usize, EXPERTS as usize);
    let input = f32_buffer(&device, (TOKENS * K) as usize, 101);
    let (gate_weight, stride) = weights(&device, GgmlType::Q4_K, EXPERTS, N, K, 102);
    let (up_weight, _) = weights(&device, GgmlType::Q4_K, EXPERTS, N, K, 103);
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
    let gathered_elements = ((TOKENS * TOP_K + EXPERTS) * K) as usize;
    let gathered = device
        .alloc_buffer(gathered_elements * 2, DType::BF16, vec![gathered_elements])
        .expect("gathered input");
    let output = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("output");
    let fused = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .expect("fused output");

    let mut encoder = device.command_encoder().expect("prepare encoder");
    let gate = prepare_quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &gate_weight,
        &params,
    )
    .expect("prepare gate");
    let up = prepare_quantized_matmul_id_ggml_grouped_m8(
        &mut encoder,
        &mut registry,
        &device,
        &up_weight,
        &params,
    )
    .expect("prepare up");
    encoder
        .commit_and_wait_labeled("bench.grouped_m8.prepare")
        .expect("commit preparation");

    let mut scratch = IdMmScratch::alloc(&device, EXPERTS, TOKENS).expect("scratch");
    let mut encoder = device.command_encoder().expect("projection encoder");
    for repeat in 0..REPEATS {
        quantized_matmul_id_ggml_grouped_m8_epilogue(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &gathered,
            &gate,
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
        )
        .unwrap_or_else(|error| panic!("prepared projection repeat {repeat}: {error}"));
    }
    let started = Instant::now();
    encoder
        .commit_and_wait_labeled("bench.grouped_m8.projection")
        .expect("commit projection benchmark");
    eprintln!(
        "prepared_m8: {:.3} ms/projection ({REPEATS} projections)",
        started.elapsed().as_secs_f64() * 1000.0 / REPEATS as f64
    );

    let mut scratch = IdMmScratch::alloc(&device, EXPERTS, TOKENS).expect("fused scratch");
    let mut encoder = device.command_encoder().expect("fused encoder");
    for repeat in 0..REPEATS {
        quantized_matmul_id_ggml_grouped_m8_gate_up(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &gathered,
            &gate,
            &up,
            &ids,
            &fused,
            &mut scratch,
            &params,
            GroupedM8Input::F32,
        )
        .unwrap_or_else(|error| panic!("one-pass gate/up repeat {repeat}: {error}"));
    }
    let started = Instant::now();
    encoder
        .commit_and_wait_labeled("bench.grouped_m8.gate_up")
        .expect("commit gate/up benchmark");
    eprintln!(
        "one_pass_gate_up: {:.3} ms/pair ({REPEATS} pairs)",
        started.elapsed().as_secs_f64() * 1000.0 / REPEATS as f64
    );
}
