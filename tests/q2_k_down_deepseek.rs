//! ADR-066 N-IQ2XXS §3.5/§3.7: the reused Q2_K expert kernels at ds4's
//! DeepSeek-V4-Flash down shape (256 experts, top-6, K = 2048 -> N = 4096).
//!
//! The down input is per routed row. Joint decode issues it as one forced
//! `mv_id` call with `n_tokens = 6 * m`, `top_k = 1`; prefill uses the
//! slotted `mm_id` entry. Bounds (fixed before any Metal run) match
//! `tests/iq2_xxs_deepseek.rs`: `mv_id` `2.5e-4 * sum|w x|`, `mm_id`
//! `2e-3 * sum|w x|`.

#[path = "support/deepseek_expert.rs"]
mod support;

use mlx_native::{
    quantized_matmul_id_ggml_mv_with_policy, quantized_matmul_id_ggml_pooled_slotted_with_policy,
    quantized_matmul_id_ggml_pooled_with_policy, GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy,
    GgmlType, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice,
};
use support::*;

const DOWN: ExpertShape = ExpertShape {
    ggml_type: GgmlType::Q2_K,
    n: 4096,
    k: 2048,
};
const MV_BOUND: f64 = 2.5e-4;
const MM_BOUND: f64 = 2e-3;
const WEIGHT_SEED: u64 = 0x2b2b_0002;

fn params(tokens: usize, top_k: usize) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens: tokens as u32,
        top_k: top_k as u32,
        n: DOWN.n as u32,
        k: DOWN.k as u32,
        n_experts: EXPERTS as u32,
        expert_stride: DOWN.expert_bytes() as u64,
        ggml_type: GgmlType::Q2_K,
    }
}

/// Forced `mv_id` over `routed` rows, each with its own input row
/// (`n_tokens = routed`, `top_k = 1`).
fn run_mv(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MlxBuffer,
    input: &[f32],
    ids: &[u32],
) -> Vec<f32> {
    let routed = ids.len();
    let input_buf = f32_buffer(device, input);
    let ids_buf = u32_buffer(device, ids);
    let output = zero_f32_buffer(device, routed * DOWN.n);
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_mv_with_policy(
        &mut encoder,
        registry,
        device,
        &input_buf,
        weights,
        &ids_buf,
        &output,
        &params(routed, 1),
        &GgmlRoutingPolicy::default(),
    )
    .expect("Q2_K down mv_id");
    encoder.commit_and_wait().expect("GPU completion");
    output.as_slice::<f32>().expect("output").to_vec()
}

/// Slotted `mm_id` (`[tokens, top_k, K]` input) and, when requested, the
/// flattened `top_k = 1` form of the same rows.
fn run_mm(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MlxBuffer,
    input: &[f32],
    ids: &[u32],
    tokens: usize,
    policy: &GgmlRoutingPolicy,
    with_flattened: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let routed = tokens * TOP_K;
    let input_buf = f32_buffer(device, input);
    let ids_buf = u32_buffer(device, ids);
    let slotted = zero_f32_buffer(device, routed * DOWN.n);
    let mut scratch = IdMmScratch::alloc(device, EXPERTS as u32, tokens as u32).expect("scratch");
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_pooled_slotted_with_policy(
        &mut encoder,
        registry,
        device,
        &input_buf,
        weights,
        &ids_buf,
        &slotted,
        &mut scratch,
        &params(tokens, TOP_K),
        policy,
    )
    .expect("slotted Q2_K down mm_id");
    encoder.commit_and_wait().expect("slotted completion");
    let slotted_values = slotted.as_slice::<f32>().expect("slotted").to_vec();
    drop(slotted);
    if !with_flattened {
        return (slotted_values, None);
    }
    let flat = zero_f32_buffer(device, routed * DOWN.n);
    let mut flat_scratch =
        IdMmScratch::alloc(device, EXPERTS as u32, routed as u32).expect("flat scratch");
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_pooled_with_policy(
        &mut encoder,
        registry,
        device,
        &input_buf,
        weights,
        &ids_buf,
        &flat,
        &mut flat_scratch,
        &params(routed, 1),
        policy,
    )
    .expect("flattened Q2_K down mm_id");
    encoder.commit_and_wait().expect("flat completion");
    let flat_values = flat.as_slice::<f32>().expect("flat").to_vec();
    (slotted_values, Some(flat_values))
}

#[test]
fn q2_k_down_mv_id_deepseek() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let weights = expert_weights(&device, DOWN, WEIGHT_SEED);
    let weight_bytes = weights.as_slice::<u8>().expect("weights");
    // Whole decode rows route top-6; other counts route each row to one
    // seeded expert. Either way every routed row has its own input row.
    let mut cases: Vec<(String, Vec<u32>)> = [1usize, 2, 4, 6, 8, 24, 32]
        .into_iter()
        .map(|routed| {
            let seed = 0xd0_0000 + routed as u64;
            let ids = if routed % TOP_K == 0 {
                routes(routed / TOP_K, TOP_K, seed)
            } else {
                routes(routed, 1, seed)
            };
            (format!("Q2_K down mv_id {routed} routed rows"), ids)
        })
        .collect();
    cases.push((
        "Q2_K down mv_id shared-expert route".into(),
        shared_expert_routes(4, TOP_K, 33, 0xd2_0000),
    ));
    for (label, ids) in cases {
        let input = inputs(ids.len(), DOWN.k, 0xd3_0000 + ids.len() as u64);
        let got = run_mv(&device, &mut registry, &weights, &input, &ids);
        assert_oracle(
            &label,
            weight_bytes,
            DOWN,
            &input,
            &ids,
            |routed| routed,
            &got,
            &sample_rows(ids.len(), usize::MAX, 0),
            &sample_columns(DOWN.n, 256),
            MV_BOUND,
        );
    }
}

/// Joint decode's down call (`n_tokens = 6 * m`, `top_k = 1`, forced
/// `mv_id`; at m = 8 that is 48 routed rows, above the 32-row `mm_id`
/// threshold) must equal m per-row calls of 6 routed rows bit for bit.
#[test]
fn q2_k_down_mv_id_row_bitwise() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let weights = expert_weights(&device, DOWN, WEIGHT_SEED);
    for m in 1..=8usize {
        let routed = TOP_K * m;
        let input = inputs(routed, DOWN.k, 0xe0e0 + m as u64);
        let ids = routes(m, TOP_K, 0xe1e1 + m as u64);
        let joint = run_mv(&device, &mut registry, &weights, &input, &ids);
        let mut serial = Vec::with_capacity(joint.len());
        for row in 0..m {
            serial.extend(run_mv(
                &device,
                &mut registry,
                &weights,
                &input[row * TOP_K * DOWN.k..(row + 1) * TOP_K * DOWN.k],
                &ids[row * TOP_K..(row + 1) * TOP_K],
            ));
        }
        assert_bitwise(&format!("Q2_K down mv_id joint m={m}"), &joint, &serial);
    }
}

fn mm_cases(
    label: &str,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    policy: GgmlRoutingPolicy,
) {
    let weights = expert_weights(device, DOWN, WEIGHT_SEED);
    let weight_bytes = weights.as_slice::<u8>().expect("weights");
    let mut cases: Vec<(usize, Vec<u32>, &str)> = [33usize, 64, 512, 4096]
        .into_iter()
        .map(|tokens| {
            (
                tokens,
                routes(tokens, TOP_K, 0xf0_0000 + tokens as u64),
                "seeded",
            )
        })
        .collect();
    cases.push((
        512,
        shared_expert_routes(512, TOP_K, 97, 0xf1_0000),
        "shared-expert",
    ));
    for (tokens, ids, kind) in cases {
        let routed = tokens * TOP_K;
        let input = inputs(routed, DOWN.k, 0xf2_0000 + tokens as u64);
        let (slotted, flat) = run_mm(
            device,
            registry,
            &weights,
            &input,
            &ids,
            tokens,
            &policy,
            tokens <= 512,
        );
        if let Some(flat) = flat {
            assert_bitwise(
                &format!("{label} {kind} {tokens}: slotted vs flattened"),
                &slotted,
                &flat,
            );
        }
        assert_oracle(
            &format!("{label} {kind} {tokens}x{TOP_K}"),
            weight_bytes,
            DOWN,
            &input,
            &ids,
            |routed| routed,
            &slotted,
            &sample_rows(routed, 48, 0xf3_0000 + tokens as u64),
            &sample_columns(DOWN.n, 64),
            MM_BOUND,
        );
    }
}

#[test]
fn q2_k_down_mm_id_deepseek() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mm_cases(
        "Q2_K down simdgroup mm_id",
        &device,
        &mut registry,
        simd_policy(),
    );
}

#[test]
fn q2_k_down_mm_id_tensor_deepseek() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    if !tensor_mm_id_available(&mut registry, &device, "kernel_mul_mm_id_q2_K_tensor_f32") {
        eprintln!("tensor-API mm_id is unavailable on this device; simdgroup parity covers it");
        return;
    }
    mm_cases(
        "Q2_K down tensor mm_id",
        &device,
        &mut registry,
        auto_policy(),
    );
}
