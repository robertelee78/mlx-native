use mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml_mv;
use mlx_native::{
    ggml_expert_bytes, quantized_matmul_id_ggml_grouped_mv, DType,
    GgmlQuantizedMatmulIdParams, GgmlType, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice,
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
    let mut buffer = device.alloc_buffer(len * 4, DType::F32, vec![len]).expect("f32");
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
    buffer.as_mut_slice::<u32>().expect("ids slice").copy_from_slice(&values);
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
    let stride = u64::from(n)
        * u64::from(k / ggml_type.block_values())
        * u64::from(ggml_type.block_bytes());
    let bytes = ggml_expert_bytes(ggml_type, experts, n, k, stride).expect("weight bytes") as usize;
    let mut buffer = device.alloc_buffer(bytes, DType::U8, vec![bytes]).expect("weights");
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
            let sign = if index == 0 && rng.next() & 1 != 0 { 0x8000 } else { 0 };
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

fn grouped_matches_mv(ggml_type: GgmlType) {
    const TOKENS: u32 = 11;
    const EXPERTS: u32 = 8;
    const TOP_K: u32 = 8;
    const K: u32 = 256;
    const N: u32 = 32;
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
    let mut grouped = device.alloc_buffer(elements * 4, DType::F32, vec![elements]).expect("grouped");
    let mut reference = device.alloc_buffer(elements * 4, DType::F32, vec![elements]).expect("reference");
    let mut scratch = IdMmScratch::alloc(&device, EXPERTS, TOKENS).expect("scratch");
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_grouped_mv(
        &mut encoder, &mut registry, &device, &input, &weight, &ids, &grouped,
        &mut scratch, &params, false, false,
    )
    .expect("grouped shared-input projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder, &mut registry, &device, &input, &weight, &ids, &reference, &params,
    )
    .expect("reference shared-input projection");
    encoder.commit_and_wait_labeled("test.grouped_mv.shared").expect("commit shared");
    assert_bitwise(&grouped, &reference, elements, "shared input");

    grouped.as_mut_slice::<u32>().expect("grouped words").fill(0x7fc0_0011);
    reference.as_mut_slice::<u32>().expect("reference words").fill(0x7fc0_0012);
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
        &mut encoder, &mut registry, &device, &input, &weight, &ids, &prep_output,
        &mut scratch, &params, false, false,
    )
    .expect("prepare grouped schedule");
    quantized_matmul_id_ggml_grouped_mv(
        &mut encoder, &mut registry, &device, &slotted, &weight, &ids, &grouped,
        &mut scratch, &slotted_params, true, true,
    )
    .expect("grouped slotted projection");
    quantized_matmul_id_ggml_mv(
        &mut encoder, &mut registry, &device, &slotted, &weight, &ids, &reference, &flat_params,
    )
    .expect("reference slotted projection");
    encoder.commit_and_wait_labeled("test.grouped_mv.slotted").expect("commit slotted");
    assert_bitwise(&grouped, &reference, elements, "slotted input");
}

#[test]
fn grouped_q4_k_matches_mv() {
    grouped_matches_mv(GgmlType::Q4_K);
}

#[test]
fn grouped_q6_k_matches_mv() {
    grouped_matches_mv(GgmlType::Q6_K);
}
