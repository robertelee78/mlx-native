//! GPU byte-parity gate for the Q5_K column-amortizing small-batch matvec.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::{
    quantized_matmul_ggml_with_policy, quantized_matmul_ggml_with_policy_and_trace, DType,
    GgmlQuantizedMatmulParams, GgmlResolvedKernelRoute, GgmlRoutingPolicy, GgmlType,
    GgmlWorkloadClass, KernelPipelineOrigin, KernelRegistry, MlxDevice,
};

const QK_K: usize = 256;
const BLOCK_Q5_K_BYTES: usize = 176;
const N: usize = 513;
const K: usize = 5_120;

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn random_f32(state: &mut u64) -> f32 {
    let unit = (next_u64(state) >> 40) as f32 / (1u32 << 24) as f32;
    unit - 0.5
}

fn q5k_weight_bytes(state: &mut u64) -> Vec<u8> {
    let n_blocks = N * (K / QK_K);
    let mut bytes = Vec::with_capacity(n_blocks * BLOCK_Q5_K_BYTES);
    for block in 0..n_blocks {
        let d = half::f16::from_f32(0.001 + (block % 31) as f32 * 0.000_031_25);
        let dmin = half::f16::from_f32(0.000_5 + (block % 17) as f32 * 0.000_015_625);
        bytes.extend_from_slice(&d.to_bits().to_le_bytes());
        bytes.extend_from_slice(&dmin.to_bits().to_le_bytes());
        for _ in 0..12 + 32 + 128 {
            bytes.push(next_u64(state) as u8);
        }
    }
    assert_eq!(bytes.len(), n_blocks * BLOCK_Q5_K_BYTES);
    bytes
}

fn assert_q5k_mn_matches_serial(m: usize, seed: u64) {
    let mut state = seed;
    let weight_bytes = q5k_weight_bytes(&mut state);
    let input: Vec<f32> = (0..m * K).map(|_| random_f32(&mut state)).collect();

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut weight = device
        .alloc_buffer(weight_bytes.len(), DType::U8, vec![weight_bytes.len()])
        .expect("weight buffer");
    weight
        .as_mut_slice::<u8>()
        .expect("weight slice")
        .copy_from_slice(&weight_bytes);

    let serial_policy = GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: false,
        dense_decode_mvn: false,
        dense_decode_mv_ext: false,
        ..GgmlRoutingPolicy::default()
    };
    let serial_params = GgmlQuantizedMatmulParams {
        m: 1,
        n: N as u32,
        k: K as u32,
        ggml_type: GgmlType::Q5_K,
    };
    let mut expected = Vec::with_capacity(m * N);
    for column in 0..m {
        let mut column_input = device
            .alloc_buffer(K * 4, DType::F32, vec![1, K])
            .expect("scalar input");
        column_input
            .as_mut_slice::<f32>()
            .expect("scalar input slice")
            .copy_from_slice(&input[column * K..(column + 1) * K]);
        let output = device
            .alloc_buffer(N * 4, DType::F32, vec![1, N])
            .expect("scalar output");
        let mut encoder = device.command_encoder().expect("scalar encoder");
        quantized_matmul_ggml_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &column_input,
            &weight,
            &output,
            &serial_params,
            &serial_policy,
        )
        .expect("scalar dispatch");
        encoder.commit_and_wait().expect("scalar GPU execution");
        expected.extend_from_slice(output.as_slice::<f32>().expect("scalar result"));
    }

    let mut batched_input = device
        .alloc_buffer(m * K * 4, DType::F32, vec![m, K])
        .expect("mN input");
    batched_input
        .as_mut_slice::<f32>()
        .expect("mN input slice")
        .copy_from_slice(&input);
    let mut output = device
        .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
        .expect("mN output");
    const UNWRITTEN: u32 = 0x7fc1_2345;
    output
        .as_mut_slice::<f32>()
        .expect("mN output slice")
        .fill(f32::from_bits(UNWRITTEN));
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: N as u32,
        k: K as u32,
        ggml_type: GgmlType::Q5_K,
    };
    let exact_width_policy = GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: false,
        dense_decode_mvn: true,
        dense_decode_mv_ext: false,
        ..GgmlRoutingPolicy::default()
    };
    let mut encoder = device.command_encoder().expect("mN encoder");
    let trace = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &batched_input,
        &weight,
        &output,
        &params,
        &exact_width_policy,
        GgmlWorkloadClass::ContinuousWidth,
    )
    .expect("traced mN dispatch");
    assert_eq!(
        trace.resolved_route,
        GgmlResolvedKernelRoute::DenseQ5kWidthMn
    );
    assert_eq!(trace.dispatches.len(), if m <= 5 { 1 } else { 2 });
    let precompiled_disabled = |name| {
        matches!(
            std::env::var(name).as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        )
    };
    let expected_origin = if precompiled_disabled("MLX_PRECOMPILED_METALLIB")
        || precompiled_disabled("MLX_PRECOMPILED_METALLIB_FCV")
    {
        KernelPipelineOrigin::RuntimeSource
    } else {
        KernelPipelineOrigin::PrecompiledMetallib
    };
    assert!(
        trace
            .dispatches
            .iter()
            .all(|dispatch| dispatch.pipeline.origin == expected_origin),
        "Q5_K mN delivery path mismatch: expected {expected_origin:?}, got {:?}",
        trace
            .dispatches
            .iter()
            .map(|dispatch| dispatch.pipeline.origin)
            .collect::<Vec<_>>()
    );
    encoder.commit_and_wait().expect("mN GPU execution");

    let actual = output.as_slice::<f32>().expect("mN result");
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        assert_ne!(
            got.to_bits(),
            UNWRITTEN,
            "Q5_K mN row was not executed: m={m}, index={index}"
        );
        assert!(
            got.is_finite(),
            "Q5_K mN produced non-finite output: m={m}, index={index}, value={got:?}"
        );
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "Q5_K mN byte mismatch: m={m}, column={}, row={}, scalar={want:?}, mN={got:?}",
            index / N,
            index % N,
        );
    }
}

#[test]
fn q5k_adaptive_mn_is_byte_identical_to_serial_for_every_production_width() {
    for m in 2..=8 {
        assert_q5k_mn_matches_serial(m, 0x5135_4b4d_0000_0000 + m as u64);
    }
}
