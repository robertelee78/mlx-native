//! Q5_K q4x4 arithmetic must be byte-identical across decode widths.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::{
    quantized_matmul_ggml_with_policy, quantized_matmul_ggml_with_policy_and_trace, CapturedNode,
    DType, GgmlQuantizedMatmulParams, GgmlResolvedKernelRoute, GgmlRoutingPolicy, GgmlType,
    GgmlWorkloadClass, KernelPipelineOrigin, KernelRegistry, MlxBuffer, MlxDevice,
};

const N: usize = 513;
const BLOCK_Q5_K_BYTES: usize = 176;
const UNWRITTEN: u32 = 0x7fc1_2345;
const INPUT_CANARY: u32 = 0x5135_cafe;
const OUTPUT_CANARY: u32 = 0x5135_dead;
const INPUT_PREFIX: usize = 19;
const INPUT_SUFFIX: usize = 23;
const OUTPUT_PREFIX: usize = 17;
const OUTPUT_SUFFIX: usize = 29;

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn fill_q5k(buffer: &mut MlxBuffer, state: &mut u64) {
    for (block, bytes) in buffer
        .as_mut_slice::<u8>()
        .expect("Q5_K buffer")
        .chunks_exact_mut(BLOCK_Q5_K_BYTES)
        .enumerate()
    {
        let d = half::f16::from_f32(0.001 + (block % 31) as f32 * 0.000_031_25);
        let dmin = half::f16::from_f32(0.000_5 + (block % 17) as f32 * 0.000_015_625);
        bytes[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        bytes[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        for byte in &mut bytes[4..] {
            *byte = next_u64(state) as u8;
        }
    }
}

fn fill_f32(buffer: &mut MlxBuffer, state: &mut u64) {
    for value in buffer.as_mut_slice::<f32>().expect("F32 buffer") {
        let unit = (next_u64(state) >> 40) as f32 / (1u32 << 24) as f32;
        *value = unit - 0.5;
    }
}

fn assert_widths_match_r1(k: usize) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut state = 0x5135_4558_5457_4944 ^ k as u64;
    let blocks = N * (k / 256);
    let mut weight = device
        .alloc_buffer(
            blocks * BLOCK_Q5_K_BYTES,
            DType::U8,
            vec![blocks * BLOCK_Q5_K_BYTES],
        )
        .expect("weight");
    fill_q5k(&mut weight, &mut state);
    let policy = GgmlRoutingPolicy {
        dense_q5k_canonical_q4x4: true,
        ..GgmlRoutingPolicy::default()
    };
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

    for m in 1..=8usize {
        let input_parent_len = INPUT_PREFIX + m * k + INPUT_SUFFIX;
        let mut input_parent = device
            .alloc_buffer(input_parent_len * 4, DType::F32, vec![input_parent_len])
            .expect("input parent");
        input_parent
            .as_mut_slice::<f32>()
            .expect("input parent slice")
            .fill(f32::from_bits(INPUT_CANARY));
        let mut input = input_parent.slice_view((INPUT_PREFIX * 4) as u64, m * k);
        fill_f32(&mut input, &mut state);
        let input_snapshot: Vec<u32> = input_parent
            .as_slice::<f32>()
            .expect("input snapshot")
            .iter()
            .map(|value| value.to_bits())
            .collect();

        let output_parent_len = OUTPUT_PREFIX + m * N + OUTPUT_SUFFIX;
        let mut output_parent = device
            .alloc_buffer(output_parent_len * 4, DType::F32, vec![output_parent_len])
            .expect("output parent");
        output_parent
            .as_mut_slice::<f32>()
            .expect("output parent slice")
            .fill(f32::from_bits(OUTPUT_CANARY));
        let mut batched_output = output_parent.slice_view((OUTPUT_PREFIX * 4) as u64, m * N);
        batched_output
            .as_logical_mut_slice::<f32>()
            .expect("batched output logical slice")
            .fill(f32::from_bits(UNWRITTEN));

        let mut encoder = device.command_encoder().expect("batched encoder");
        let trace = quantized_matmul_ggml_with_policy_and_trace(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            &weight,
            &batched_output,
            &GgmlQuantizedMatmulParams {
                m: m as u32,
                n: N as u32,
                k: k as u32,
                ggml_type: GgmlType::Q5_K,
            },
            &policy,
            if m == 1 {
                GgmlWorkloadClass::DecodeSingle
            } else {
                GgmlWorkloadClass::ContinuousWidth
            },
        )
        .expect("batched Q5_K route");
        assert_eq!(
            trace.resolved_route,
            GgmlResolvedKernelRoute::DenseQ5kCanonicalQ4x4
        );
        assert_eq!(trace.dispatches.len(), 1);
        assert_eq!(trace.dispatches[0].pipeline.origin, expected_origin);
        encoder.commit_and_wait().expect("batched execution");

        let batched_bits: Vec<u32> = batched_output
            .as_logical_slice::<f32>()
            .expect("batched output logical slice")
            .iter()
            .map(|value| value.to_bits())
            .collect();
        for column in 0..m {
            let single_input = input.slice_view((column * k * 4) as u64, k);
            let mut single_output = device
                .alloc_buffer(N * 4, DType::F32, vec![1, N])
                .expect("single output");
            single_output
                .as_mut_slice::<f32>()
                .expect("single output slice")
                .fill(f32::from_bits(UNWRITTEN));
            let mut single_encoder = device.command_encoder().expect("single encoder");
            let single_trace = quantized_matmul_ggml_with_policy_and_trace(
                &mut single_encoder,
                &mut registry,
                &device,
                &single_input,
                &weight,
                &single_output,
                &GgmlQuantizedMatmulParams {
                    m: 1,
                    n: N as u32,
                    k: k as u32,
                    ggml_type: GgmlType::Q5_K,
                },
                &policy,
                GgmlWorkloadClass::DecodeSingle,
            )
            .expect("single Q5_K route");
            assert_eq!(
                single_trace.resolved_route,
                GgmlResolvedKernelRoute::DenseQ5kCanonicalQ4x4
            );
            assert_eq!(single_trace.dispatches[0].pipeline.origin, expected_origin);
            single_encoder.commit_and_wait().expect("single execution");
            for (row, (want, &got)) in single_output
                .as_slice::<f32>()
                .expect("single output slice")
                .iter()
                .map(|value| value.to_bits())
                .zip(&batched_bits[column * N..(column + 1) * N])
                .enumerate()
            {
                assert_ne!(got, UNWRITTEN, "m={m} column={column} row={row}");
                assert_eq!(
                    got, want,
                    "Q5_K width drift at K={k} m={m} column={column} row={row}: r1={want:#010x}, batched={got:#010x}"
                );
            }
        }

        assert_eq!(
            input_parent
                .as_slice::<f32>()
                .expect("input parent after")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            input_snapshot,
            "K={k} m={m} mutated input parent"
        );
        for (label, values, prefix, payload, marker) in [
            (
                "input",
                input_parent.as_slice::<f32>().expect("input parent"),
                INPUT_PREFIX,
                m * k,
                INPUT_CANARY,
            ),
            (
                "output",
                output_parent.as_slice::<f32>().expect("output parent"),
                OUTPUT_PREFIX,
                m * N,
                OUTPUT_CANARY,
            ),
        ] {
            assert!(
                values[..prefix]
                    .iter()
                    .all(|value| value.to_bits() == marker),
                "K={k} m={m} changed {label} prefix canary"
            );
            assert!(
                values[prefix + payload..]
                    .iter()
                    .all(|value| value.to_bits() == marker),
                "K={k} m={m} changed {label} suffix canary"
            );
        }
    }
}

#[test]
fn q5k_q4x4_is_byte_identical_across_decode_widths() {
    assert_widths_match_r1(512);
    assert_widths_match_r1(5_120);

    if std::env::var_os("MLX_NATIVE_Q5K_RUNTIME_SOURCE_CHILD").is_none() {
        let output = std::process::Command::new(std::env::current_exe().expect("current test exe"))
            .arg("--exact")
            .arg("q5k_q4x4_is_byte_identical_across_decode_widths")
            .arg("--nocapture")
            .env("MLX_NATIVE_Q5K_RUNTIME_SOURCE_CHILD", "1")
            .env("MLX_PRECOMPILED_METALLIB", "0")
            .env("MLX_PRECOMPILED_METALLIB_FCV", "0")
            .output()
            .expect("run runtime-source Q5_K width gate");
        assert!(
            output.status.success(),
            "runtime-source Q5_K width gate failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
            "runtime-source Q5_K width gate executed no exact test: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

#[test]
fn q5k_canonical_route_captures_real_dependencies() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let params = GgmlQuantizedMatmulParams {
        m: 4,
        n: 9,
        k: 256,
        ggml_type: GgmlType::Q5_K,
    };
    let input = device
        .alloc_buffer(4 * 256 * 4, DType::F32, vec![4, 256])
        .expect("input");
    let weight = device
        .alloc_buffer(9 * BLOCK_Q5_K_BYTES, DType::U8, vec![9 * BLOCK_Q5_K_BYTES])
        .expect("weight");
    let output = device
        .alloc_buffer(4 * 9 * 4, DType::F32, vec![4, 9])
        .expect("output");
    let mut encoder = device.command_encoder().expect("capture encoder");
    encoder.start_capture();
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input,
        &weight,
        &output,
        &params,
        &GgmlRoutingPolicy {
            dense_q5k_canonical_q4x4: true,
            ..GgmlRoutingPolicy::default()
        },
    )
    .expect("capture canonical Q5_K route");
    let nodes = encoder.take_capture().expect("captured graph");
    assert_eq!(nodes.len(), 1);
    match &nodes[0] {
        CapturedNode::Dispatch { reads, writes, .. } => {
            assert_eq!(reads.len(), 2, "weight and input must be tracked reads");
            assert_eq!(writes.len(), 1, "output must be a tracked write");
        }
        CapturedNode::Barrier => panic!("expected one tracked Q5_K dispatch"),
    }
}
