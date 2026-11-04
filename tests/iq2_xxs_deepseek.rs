//! ADR-066 N-IQ2XXS: IQ2_XXS expert kernels at the DeepSeek-V4-Flash
//! gate/up shape (256 experts, top-6, K = 4096 -> N = 2048).
//!
//! Oracle: CPU dequant (bit-exact against the reference, see
//! `tests/iq2_xxs_dequant.rs`) and an F64 dot. Bounds were fixed in the
//! design before any Metal run:
//! - `mv_id`: `|got - want| <= 2.5e-4 * sum|w x|` (worst-case F32
//!   summation, K * 2^-24 at K = 4096);
//! - `mm_id` simdgroup and tensor: `<= 2e-3 * sum|w x|` (half staging of
//!   the weights, and of the input in the tensor path), valid on the value
//!   ranges in `support/deepseek_expert.rs`.

#[path = "support/deepseek_expert.rs"]
mod support;

use mlx_native::{
    quantized_matmul_id_ggml_mv_with_policy, quantized_matmul_id_ggml_pooled_pair_with_policy,
    quantized_matmul_id_ggml_pooled_with_policy, quantized_matmul_id_ggml_with_policy,
    GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy, GgmlType, IdMmScratch, KernelRegistry,
    MlxBuffer, MlxDevice,
};
use support::*;

const GATE_UP: ExpertShape = ExpertShape {
    ggml_type: GgmlType::IQ2_XXS,
    n: 2048,
    k: 4096,
};
const MV_BOUND: f64 = 2.5e-4;
const MM_BOUND: f64 = 2e-3;
const WEIGHT_SEED: u64 = 0x1a2b_0001;

fn params(tokens: usize, top_k: usize) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens: tokens as u32,
        top_k: top_k as u32,
        n: GATE_UP.n as u32,
        k: GATE_UP.k as u32,
        n_experts: EXPERTS as u32,
        expert_stride: GATE_UP.expert_bytes() as u64,
        ggml_type: GgmlType::IQ2_XXS,
    }
}

#[derive(Clone, Copy)]
enum Route {
    ForcedMv,
    Auto(fn() -> GgmlRoutingPolicy),
}

#[allow(clippy::too_many_arguments)]
fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MlxBuffer,
    input: &[f32],
    ids: &[u32],
    tokens: usize,
    top_k: usize,
    route: Route,
) -> Vec<f32> {
    let input_buf = f32_buffer(device, input);
    let ids_buf = u32_buffer(device, ids);
    let output = zero_f32_buffer(device, tokens * top_k * GATE_UP.n);
    let p = params(tokens, top_k);
    let mut encoder = device.command_encoder().expect("encoder");
    match route {
        Route::ForcedMv => quantized_matmul_id_ggml_mv_with_policy(
            &mut encoder,
            registry,
            device,
            &input_buf,
            weights,
            &ids_buf,
            &output,
            &p,
            &GgmlRoutingPolicy::default(),
        ),
        Route::Auto(policy) => quantized_matmul_id_ggml_with_policy(
            &mut encoder,
            registry,
            device,
            &input_buf,
            weights,
            &ids_buf,
            &output,
            &p,
            &policy(),
        ),
    }
    .expect("IQ2_XXS expert dispatch");
    encoder.commit_and_wait().expect("GPU completion");
    output.as_slice::<f32>().expect("output").to_vec()
}

#[allow(clippy::too_many_arguments)]
fn check_case(
    label: &str,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MlxBuffer,
    tokens: usize,
    top_k: usize,
    ids: &[u32],
    route: Route,
    bound: f64,
    row_cap: usize,
    column_count: usize,
) {
    let input = inputs(tokens, GATE_UP.k, 0x5eed_0000 + tokens as u64);
    let got = run(device, registry, weights, &input, ids, tokens, top_k, route);
    let rows = sample_rows(tokens * top_k, row_cap, 0x0bad_0000 + tokens as u64);
    let columns = sample_columns(GATE_UP.n, column_count);
    assert_oracle(
        label,
        weights.as_slice::<u8>().expect("weights"),
        GATE_UP,
        &input,
        ids,
        |routed| routed / top_k,
        &got,
        &rows,
        &columns,
        bound,
    );
}

#[test]
fn iq2_xxs_mv_id_deepseek() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let weights = expert_weights(&device, GATE_UP, WEIGHT_SEED);
    // (n_tokens, top_k): 1, 2, 4, 6, 8, 24 and 32 routed rows.
    for (tokens, top_k) in [
        (1, 1),
        (2, 1),
        (4, 1),
        (1, TOP_K),
        (8, 1),
        (4, TOP_K),
        (32, 1),
    ] {
        let ids = routes(tokens, top_k, 0x1d5_0000 + (tokens * 8 + top_k) as u64);
        check_case(
            &format!("IQ2_XXS mv_id {tokens}x{top_k}"),
            &device,
            &mut registry,
            &weights,
            tokens,
            top_k,
            &ids,
            Route::ForcedMv,
            MV_BOUND,
            usize::MAX,
            256,
        );
    }
    let ids = shared_expert_routes(4, TOP_K, 17, 0xadd0);
    check_case(
        "IQ2_XXS mv_id shared-expert route",
        &device,
        &mut registry,
        &weights,
        4,
        TOP_K,
        &ids,
        Route::ForcedMv,
        MV_BOUND,
        usize::MAX,
        256,
    );
}

/// Joint decode issues gate/up as one forced `mv_id` call with
/// `n_tokens = m`, `top_k = 6`; it must equal m one-row calls bit for bit.
#[test]
fn iq2_xxs_mv_id_row_bitwise() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let weights = expert_weights(&device, GATE_UP, WEIGHT_SEED);
    for m in 1..=8usize {
        let input = inputs(m, GATE_UP.k, 0x7070 + m as u64);
        let ids = routes(m, TOP_K, 0x7171 + m as u64);
        let joint = run(
            &device,
            &mut registry,
            &weights,
            &input,
            &ids,
            m,
            TOP_K,
            Route::ForcedMv,
        );
        let mut serial = Vec::with_capacity(joint.len());
        for row in 0..m {
            serial.extend(run(
                &device,
                &mut registry,
                &weights,
                &input[row * GATE_UP.k..(row + 1) * GATE_UP.k],
                &ids[row * TOP_K..(row + 1) * TOP_K],
                1,
                TOP_K,
                Route::ForcedMv,
            ));
        }
        assert_bitwise(&format!("IQ2_XXS mv_id joint m={m}"), &joint, &serial);
    }
}

fn mm_cases(
    label: &str,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    policy: fn() -> GgmlRoutingPolicy,
) {
    let weights = expert_weights(device, GATE_UP, WEIGHT_SEED);
    for tokens in [33usize, 64, 512, 4096] {
        let ids = routes(tokens, TOP_K, 0x3300 + tokens as u64);
        check_case(
            &format!("{label} {tokens}x{TOP_K}"),
            device,
            registry,
            &weights,
            tokens,
            TOP_K,
            &ids,
            Route::Auto(policy),
            MM_BOUND,
            48,
            64,
        );
    }
    let ids = shared_expert_routes(512, TOP_K, 201, 0x5151);
    check_case(
        &format!("{label} shared-expert route 512x{TOP_K}"),
        device,
        registry,
        &weights,
        512,
        TOP_K,
        &ids,
        Route::Auto(policy),
        MM_BOUND,
        48,
        64,
    );
}

#[test]
fn iq2_xxs_mm_id_deepseek() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mm_cases(
        "IQ2_XXS simdgroup mm_id",
        &device,
        &mut registry,
        simd_policy,
    );
}

#[test]
fn iq2_xxs_mm_id_tensor_deepseek() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    if !tensor_mm_id_available(
        &mut registry,
        &device,
        "kernel_mul_mm_id_iq2_xxs_tensor_f32",
    ) {
        eprintln!("tensor-API mm_id is unavailable on this device; simdgroup parity covers it");
        return;
    }
    mm_cases("IQ2_XXS tensor mm_id", &device, &mut registry, auto_policy);
}

/// ds4's gate and up are both IQ2_XXS: the paired entry runs map0 once and
/// each output must equal the unpaired `mm_id` output bit for bit.
#[test]
fn iq2_xxs_pair_deepseek() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let gate = expert_weights(&device, GATE_UP, WEIGHT_SEED);
    let up = expert_weights(&device, GATE_UP, WEIGHT_SEED ^ 0xffff);
    for policy in [simd_policy as fn() -> GgmlRoutingPolicy, auto_policy] {
        for tokens in [33usize, 512] {
            let input = inputs(tokens, GATE_UP.k, 0x9a17 + tokens as u64);
            let ids = routes(tokens, TOP_K, 0x9a18 + tokens as u64);
            let input_buf = f32_buffer(&device, &input);
            let ids_buf = u32_buffer(&device, &ids);
            let len = tokens * TOP_K * GATE_UP.n;
            let pair_gate = zero_f32_buffer(&device, len);
            let pair_up = zero_f32_buffer(&device, len);
            let solo_gate = zero_f32_buffer(&device, len);
            let solo_up = zero_f32_buffer(&device, len);
            let p = params(tokens, TOP_K);
            let routing = policy();
            let mut scratch =
                IdMmScratch::alloc(&device, EXPERTS as u32, tokens as u32).expect("scratch");
            let mut encoder = device.command_encoder().expect("encoder");
            quantized_matmul_id_ggml_pooled_pair_with_policy(
                &mut encoder,
                &mut registry,
                &device,
                &input_buf,
                &gate,
                &up,
                &ids_buf,
                &pair_gate,
                &pair_up,
                &mut scratch,
                &p,
                &routing,
            )
            .expect("paired IQ2_XXS mm_id");
            encoder.commit_and_wait().expect("pair completion");
            for (weights, output) in [(&gate, &solo_gate), (&up, &solo_up)] {
                let mut encoder = device.command_encoder().expect("encoder");
                quantized_matmul_id_ggml_pooled_with_policy(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &input_buf,
                    weights,
                    &ids_buf,
                    output,
                    &mut scratch,
                    &p,
                    &routing,
                )
                .expect("unpaired IQ2_XXS mm_id");
                encoder.commit_and_wait().expect("solo completion");
            }
            let read = |buffer: &MlxBuffer| buffer.as_slice::<f32>().expect("output").to_vec();
            assert_bitwise(
                &format!("IQ2_XXS pair gate {tokens}"),
                &read(&pair_gate),
                &read(&solo_gate),
            );
            assert_bitwise(
                &format!("IQ2_XXS pair up {tokens}"),
                &read(&pair_up),
                &read(&solo_up),
            );
            assert_ne!(
                read(&pair_gate),
                read(&pair_up),
                "gate and up weights differ"
            );
        }
    }
}
