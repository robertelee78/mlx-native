//! Exact-mN regression for nonzero-base logical input/output views.
//!
//! `KernelArg::BufferWithOffset` takes an absolute allocation offset. These
//! cases force the physical width-4 and tiled width-7 paths to compose their
//! column offsets with an `MlxBuffer::byte_offset()`, while parent canaries
//! prove no prefix or suffix was touched.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::{
    quantized_matmul_ggml_with_policy, quantized_matmul_ggml_with_policy_and_trace, DType,
    GgmlQuantizedMatmulParams, GgmlResolvedKernelRoute, GgmlRoutingPolicy, GgmlType,
    GgmlWorkloadClass, KernelPipelineOrigin, KernelRegistry, MlxDevice,
};

const N: usize = 513;
const K: usize = 512;
const INPUT_PREFIX: usize = 19;
const INPUT_SUFFIX: usize = 23;
const OUTPUT_PREFIX: usize = 17;
const OUTPUT_SUFFIX: usize = 29;
const INPUT_CANARY: u32 = 0x4b17_cafe;
const OUTPUT_CANARY: u32 = 0x4b17_dead;

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

fn packed_weights(kind: GgmlType, state: &mut u64) -> Vec<u8> {
    let block_bytes = kind.block_bytes() as usize;
    let blocks = N * (K / kind.block_values() as usize);
    let mut bytes = vec![0u8; blocks * block_bytes];
    for (block_index, block) in bytes.chunks_exact_mut(block_bytes).enumerate() {
        for byte in block.iter_mut() {
            *byte = next_u64(state) as u8;
        }
        let d = half::f16::from_f32(0.001 + (block_index % 31) as f32 * 0.000_031_25);
        match kind {
            GgmlType::Q4_K | GgmlType::Q5_K => {
                let dmin = half::f16::from_f32(0.000_5 + (block_index % 17) as f32 * 0.000_015_625);
                block[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
            }
            GgmlType::Q6_K => {
                block[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
            }
            other => panic!("unsupported test codec {other:?}"),
        }
    }
    bytes
}

fn expected_route(kind: GgmlType) -> GgmlResolvedKernelRoute {
    match kind {
        GgmlType::Q4_K => GgmlResolvedKernelRoute::DenseQ4kWidthMn,
        GgmlType::Q5_K => GgmlResolvedKernelRoute::DenseQ5kWidthMn,
        GgmlType::Q6_K => GgmlResolvedKernelRoute::DenseQ6kWidthMn,
        other => panic!("unsupported test codec {other:?}"),
    }
}

fn assert_parent_canaries(parent: &[f32], prefix: usize, payload: usize, marker: u32, label: &str) {
    for (index, value) in parent[..prefix].iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            marker,
            "{label} prefix canary changed at {index}"
        );
    }
    for (index, value) in parent[prefix + payload..].iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            marker,
            "{label} suffix canary changed at {index}"
        );
    }
}

fn assert_logical_view_case(kind: GgmlType, m: usize) {
    let mut state = 0x4b56_0000_0000_0000 ^ ((kind as u32 as u64) << 16) ^ m as u64;
    let weights = packed_weights(kind, &mut state);
    let inputs: Vec<f32> = (0..m * K).map(|_| random_f32(&mut state)).collect();
    assert_ne!(
        inputs[..K].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        inputs[K..2 * K]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "input rows must be distinct"
    );

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut weight = device
        .alloc_buffer(weights.len(), DType::U8, vec![weights.len()])
        .expect("weight parent");
    weight
        .as_mut_slice::<u8>()
        .expect("weight bytes")
        .copy_from_slice(&weights);

    let input_parent_len = INPUT_PREFIX + m * K + INPUT_SUFFIX;
    let mut input_parent = device
        .alloc_buffer(input_parent_len * 4, DType::F32, vec![input_parent_len])
        .expect("input parent");
    input_parent
        .as_mut_slice::<f32>()
        .expect("input parent slice")
        .fill(f32::from_bits(INPUT_CANARY));
    let mut input_view = input_parent.slice_view((INPUT_PREFIX * 4) as u64, m * K);
    input_view
        .as_logical_mut_slice::<f32>()
        .expect("input logical view")
        .copy_from_slice(&inputs);
    let input_snapshot: Vec<u32> = input_parent
        .as_slice::<f32>()
        .expect("input parent snapshot")
        .iter()
        .map(|value| value.to_bits())
        .collect();

    let scalar_policy = GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: false,
        dense_decode_mvn: false,
        dense_decode_mv_ext: false,
        ..GgmlRoutingPolicy::default()
    };
    let scalar_params = GgmlQuantizedMatmulParams {
        m: 1,
        n: N as u32,
        k: K as u32,
        ggml_type: kind,
    };
    let mut expected = Vec::with_capacity(m * N);
    for column in 0..m {
        let column_input = input_view.slice_view((column * K * 4) as u64, K);
        let output = device
            .alloc_buffer(N * 4, DType::F32, vec![N])
            .expect("scalar output");
        let mut encoder = device.command_encoder().expect("scalar encoder");
        quantized_matmul_ggml_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &column_input,
            &weight,
            &output,
            &scalar_params,
            &scalar_policy,
        )
        .expect("scalar authority dispatch");
        encoder
            .commit_and_wait()
            .expect("scalar authority execution");
        expected.extend_from_slice(output.as_slice::<f32>().expect("scalar result"));
    }

    let output_parent_len = OUTPUT_PREFIX + m * N + OUTPUT_SUFFIX;
    let mut output_parent = device
        .alloc_buffer(output_parent_len * 4, DType::F32, vec![output_parent_len])
        .expect("output parent");
    output_parent
        .as_mut_slice::<f32>()
        .expect("output parent slice")
        .fill(f32::from_bits(OUTPUT_CANARY));
    let output_view = output_parent.slice_view((OUTPUT_PREFIX * 4) as u64, m * N);

    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: N as u32,
        k: K as u32,
        ggml_type: kind,
    };
    let policy = GgmlRoutingPolicy {
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
        &input_view,
        &weight,
        &output_view,
        &params,
        &policy,
        GgmlWorkloadClass::ContinuousWidth,
    )
    .expect("traced mN dispatch");
    assert_eq!(trace.resolved_route, expected_route(kind));
    assert_eq!(trace.dispatches.len(), if m == 7 { 2 } else { 1 });
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
        "{kind:?} width {m} delivery path mismatch: expected {expected_origin:?}, got {:?}",
        trace
            .dispatches
            .iter()
            .map(|dispatch| dispatch.pipeline.origin)
            .collect::<Vec<_>>()
    );
    encoder.commit_and_wait().expect("mN execution");

    let actual = output_view
        .as_logical_slice::<f32>()
        .expect("mN logical result");
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        assert!(
            got.is_finite(),
            "{kind:?} width {m} produced non-finite output at {index}: {got:?}"
        );
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{kind:?} logical-view mismatch at width={m}, column={}, row={}",
            index / N,
            index % N,
        );
    }

    let input_after: Vec<u32> = input_parent
        .as_slice::<f32>()
        .expect("input parent after")
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(
        input_after, input_snapshot,
        "{kind:?} width {m} mutated input parent"
    );
    assert_parent_canaries(
        output_parent
            .as_slice::<f32>()
            .expect("output parent after"),
        OUTPUT_PREFIX,
        m * N,
        OUTPUT_CANARY,
        &format!("{kind:?} width {m} output"),
    );
}

#[test]
fn exact_kquant_mn_honors_nonzero_base_views_for_physical_and_tiled_widths() {
    for kind in [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K] {
        for m in [4, 7] {
            assert_logical_view_case(kind, m);
        }
    }
}
