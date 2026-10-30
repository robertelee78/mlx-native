//! GPU byte-parity gate for the Q8_0 NR2 multi-input-row matvec.
//!
//! `kernel_mul_mv_q8_0_f32_nr2_r{R1}` reads each Q8_0 weight block once per
//! threadgroup and applies it to up to R1 input rows. Every output must be
//! bit-identical to an independent single-row `kernel_mul_mv_q8_0_f32_nr2`
//! call on that input row, for dense, batched, partial-tile, and opt-out
//! routes.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::{
    dispatch_mv_q8_0_nr2_rows_for_test, quantized_matmul_ggml_batched_mv_with_policy,
    quantized_matmul_ggml_with_policy, quantized_matmul_ggml_with_policy_and_trace, CapturedNode,
    CommandEncoder, DType, GgmlBatchedQuantizedMatmulParams, GgmlQuantizedMatmulParams,
    GgmlResolvedKernelRoute, GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass, KernelRegistry,
    MlxBuffer, MlxDevice,
};

const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 34;
const UNWRITTEN: u32 = 0x7fc1_2345;

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

/// Uniform in [-1, 1).
fn signed_unit(state: &mut u64) -> f32 {
    let unit = (next_u64(state) >> 40) as f32 / (1u32 << 24) as f32;
    unit * 2.0 - 1.0
}

/// Mixed-magnitude activations: mostly unit-range, with large, tiny, and
/// strictly negative outliers so accumulation order matters.
fn random_input(state: &mut u64) -> f32 {
    let value = signed_unit(state);
    match next_u64(state) % 8 {
        0 => value * 1.0e4,
        1 => value * 1.0e-3,
        2 => -value.abs() * 300.0 - 1.0,
        _ => value,
    }
}

/// Raw Q8_0 blocks with full-range int8 quants and signed, mixed-magnitude
/// half scales.
fn random_q8_0(state: &mut u64, rows: usize, k: usize) -> Vec<u8> {
    let blocks = rows * (k / QK8_0);
    let mut bytes = Vec::with_capacity(blocks * BLOCK_Q8_0_BYTES);
    for _ in 0..blocks {
        let magnitude = match next_u64(state) % 4 {
            0 => 4.0,
            1 => 1.0e-3,
            _ => 0.05,
        };
        let d = half::f16::from_f32(signed_unit(state) * magnitude);
        bytes.extend_from_slice(&d.to_bits().to_le_bytes());
        for _ in 0..QK8_0 {
            bytes.push(next_u64(state) as u8);
        }
    }
    bytes
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32 slice")
        .copy_from_slice(values);
    buffer
}

fn u8_buffer(device: &MlxDevice, bytes: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("u8 buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("u8 slice")
        .copy_from_slice(bytes);
    buffer
}

fn sentinel_buffer(device: &MlxDevice, elements: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("output buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("output slice")
        .fill(f32::from_bits(UNWRITTEN));
    buffer
}

/// Encode one independent single-row NR2 call per input row and return the
/// output buffers (read after the caller commits).
#[allow(clippy::too_many_arguments)]
fn encode_single_row_reference(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<MlxBuffer> {
    let params = GgmlQuantizedMatmulParams {
        m: 1,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q8_0,
    };
    (0..m)
        .map(|row| {
            let row_input = f32_buffer(device, &input[row * k..(row + 1) * k]);
            let output = sentinel_buffer(device, n);
            let trace = quantized_matmul_ggml_with_policy_and_trace(
                encoder,
                registry,
                device,
                &row_input,
                weight,
                &output,
                &params,
                &GgmlRoutingPolicy::default(),
                GgmlWorkloadClass::DecodeSingle,
            )
            .expect("single-row NR2 reference");
            assert_eq!(trace.resolved_route, GgmlResolvedKernelRoute::DenseMvNr2);
            output
        })
        .collect()
}

fn assert_rows_match(label: &str, reference: &[MlxBuffer], actual: &[f32], n: usize) {
    for (row, expected) in reference.iter().enumerate() {
        let expected = expected.as_slice::<f32>().expect("reference result");
        for col in 0..n {
            let want = expected[col];
            let got = actual[row * n + col];
            assert_ne!(
                want.to_bits(),
                UNWRITTEN,
                "{label}: reference row {row} col {col} unwritten"
            );
            assert_ne!(
                got.to_bits(),
                UNWRITTEN,
                "{label}: multi-row output row {row} col {col} unwritten"
            );
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}: byte mismatch at row {row} col {col}: single-row={want:?} multi-row={got:?}"
            );
        }
    }
}

#[test]
fn q8_0_multi_row_is_byte_identical_to_single_row_nr2() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    for n in [1usize, 2, 63, 4096] {
        for k in [32usize, 256, 4096, 4128] {
            let mut state = 0x5138_0000_0000_0001 ^ ((n as u64) << 20) ^ k as u64;
            let weight = u8_buffer(&device, &random_q8_0(&mut state, n, k));
            for m in [2usize, 3, 4, 5, 8] {
                let input: Vec<f32> = (0..m * k).map(|_| random_input(&mut state)).collect();
                let mut encoder = device.command_encoder().expect("encoder");
                let reference = encode_single_row_reference(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &weight,
                    &input,
                    m,
                    n,
                    k,
                );

                let batched_input = f32_buffer(&device, &input);
                let output = sentinel_buffer(&device, m * n);
                let params = GgmlQuantizedMatmulParams {
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                    ggml_type: GgmlType::Q8_0,
                };
                let trace = quantized_matmul_ggml_with_policy_and_trace(
                    &mut encoder,
                    &mut registry,
                    &device,
                    &batched_input,
                    &weight,
                    &output,
                    &params,
                    &GgmlRoutingPolicy::default(),
                    GgmlWorkloadClass::ContinuousWidth,
                )
                .expect("multi-row trace");
                assert_eq!(
                    trace.resolved_route,
                    GgmlResolvedKernelRoute::DenseQ8_0MvNr2Rows
                );
                assert_eq!(trace.dispatches.len(), 1);
                assert_eq!(
                    trace.dispatches[0].pipeline.kernel_name,
                    format!("kernel_mul_mv_q8_0_f32_nr2_r{m}")
                );
                encoder.commit_and_wait().expect("GPU execution");

                let actual = output.as_slice::<f32>().expect("multi-row result");
                assert_rows_match(&format!("n={n} k={k} m={m}"), &reference, actual, n);
            }
        }
    }
}

#[test]
fn q8_0_multi_row_partial_final_tile_is_guarded_and_exact() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (n, k, m) = (63usize, 4128usize, 5usize);
    let mut state = 0x5138_7a11_0000_0005;
    let weight = u8_buffer(&device, &random_q8_0(&mut state, n, k));
    let input: Vec<f32> = (0..m * k).map(|_| random_input(&mut state)).collect();
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q8_0,
    };
    // r1 = 2/3/4 leave a partial final tile; r1 = 8 exceeds M outright.
    for r1 in [2u32, 3, 4, 8] {
        let mut encoder = device.command_encoder().expect("encoder");
        let reference = encode_single_row_reference(
            &mut encoder,
            &mut registry,
            &device,
            &weight,
            &input,
            m,
            n,
            k,
        );
        let batched_input = f32_buffer(&device, &input);
        // One extra guard row past M must stay untouched.
        let output = sentinel_buffer(&device, (m + 1) * n);
        dispatch_mv_q8_0_nr2_rows_for_test(
            &mut encoder,
            &mut registry,
            &device,
            &batched_input,
            &weight,
            &output,
            &params,
            r1,
        )
        .expect("partial-tile dispatch");
        encoder.commit_and_wait().expect("GPU execution");

        let actual = output.as_slice::<f32>().expect("partial-tile result");
        assert_rows_match(&format!("tail r1={r1}"), &reference, actual, n);
        assert!(
            actual[m * n..]
                .iter()
                .all(|value| value.to_bits() == UNWRITTEN),
            "r1={r1}: rows past M were written"
        );
    }

    let mut encoder = device.command_encoder().expect("encoder");
    let input_buffer = f32_buffer(&device, &input);
    let output = sentinel_buffer(&device, m * n);
    for bad_r1 in [0u32, 1, 9] {
        assert!(dispatch_mv_q8_0_nr2_rows_for_test(
            &mut encoder,
            &mut registry,
            &device,
            &input_buffer,
            &weight,
            &output,
            &params,
            bad_r1,
        )
        .is_err());
    }
}

#[test]
fn q8_0_batched_multi_row_is_byte_identical_to_single_row_nr2() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (batch, n, k) = (3usize, 63usize, 4128usize);
    for m in [2usize, 4, 5, 8] {
        let mut state = 0x5138_ba7c_0000_0000 ^ m as u64;
        let weights: Vec<Vec<u8>> = (0..batch).map(|_| random_q8_0(&mut state, n, k)).collect();
        let input: Vec<f32> = (0..batch * m * k)
            .map(|_| random_input(&mut state))
            .collect();

        let mut encoder = device.command_encoder().expect("encoder");
        let mut reference = Vec::with_capacity(batch);
        let mut weight_buffers = Vec::with_capacity(batch);
        for (index, bytes) in weights.iter().enumerate() {
            let weight = u8_buffer(&device, bytes);
            reference.push(encode_single_row_reference(
                &mut encoder,
                &mut registry,
                &device,
                &weight,
                &input[index * m * k..(index + 1) * m * k],
                m,
                n,
                k,
            ));
            weight_buffers.push(weight);
        }

        let stacked: Vec<u8> = weights.concat();
        let weight = u8_buffer(&device, &stacked);
        let batched_input = f32_buffer(&device, &input);
        let params = GgmlBatchedQuantizedMatmulParams {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type: GgmlType::Q8_0,
        };
        let multi_row = sentinel_buffer(&device, batch * m * n);
        quantized_matmul_ggml_batched_mv_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &batched_input,
            &weight,
            &multi_row,
            &params,
            &GgmlRoutingPolicy::default(),
        )
        .expect("batched multi-row dispatch");
        let per_row = sentinel_buffer(&device, batch * m * n);
        quantized_matmul_ggml_batched_mv_with_policy(
            &mut encoder,
            &mut registry,
            &device,
            &batched_input,
            &weight,
            &per_row,
            &params,
            &GgmlRoutingPolicy {
                dense_q8_0_mv_r1: false,
                ..GgmlRoutingPolicy::default()
            },
        )
        .expect("batched per-row NR2 dispatch");
        encoder.commit_and_wait().expect("GPU execution");

        let multi_row = multi_row.as_slice::<f32>().expect("multi-row result");
        let per_row = per_row.as_slice::<f32>().expect("per-row result");
        for (index, rows) in reference.iter().enumerate() {
            let slice = index * m * n..(index + 1) * m * n;
            assert_rows_match(
                &format!("batched m={m} batch_index={index}"),
                rows,
                &multi_row[slice.clone()],
                n,
            );
            assert_rows_match(
                &format!("batched opt-out m={m} batch_index={index}"),
                rows,
                &per_row[slice],
                n,
            );
        }
    }
}

#[test]
fn q8_0_multi_row_opt_out_restores_per_row_nr2_dispatch() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (n, k, m) = (63usize, 4128usize, 4usize);
    let mut state = 0x5138_0f70_0000_0004;
    let weight = u8_buffer(&device, &random_q8_0(&mut state, n, k));
    let input: Vec<f32> = (0..m * k).map(|_| random_input(&mut state)).collect();
    let input_buffer = f32_buffer(&device, &input);
    let params = GgmlQuantizedMatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        ggml_type: GgmlType::Q8_0,
    };
    let opt_out = GgmlRoutingPolicy {
        dense_q8_0_mv_r1: false,
        ..GgmlRoutingPolicy::default()
    };

    let mut encoder = device.command_encoder().expect("encoder");
    let per_row = sentinel_buffer(&device, m * n);
    let trace = quantized_matmul_ggml_with_policy_and_trace(
        &mut encoder,
        &mut registry,
        &device,
        &input_buffer,
        &weight,
        &per_row,
        &params,
        &opt_out,
        GgmlWorkloadClass::ContinuousWidth,
    )
    .expect("opt-out trace");
    assert_eq!(trace.resolved_route, GgmlResolvedKernelRoute::DenseMvNr2);
    assert_eq!(
        trace.dispatches[0].pipeline.kernel_name,
        "kernel_mul_mv_q8_0_f32_nr2"
    );
    assert_eq!(
        trace.dispatches[0].encoded.grid,
        [n.div_ceil(2) as u64, m as u64, 1]
    );
    let multi_row = sentinel_buffer(&device, m * n);
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input_buffer,
        &weight,
        &multi_row,
        &params,
        &GgmlRoutingPolicy::default(),
    )
    .expect("multi-row dispatch");
    encoder.commit_and_wait().expect("GPU execution");
    let per_row = per_row.as_slice::<f32>().expect("per-row result");
    let multi_row = multi_row.as_slice::<f32>().expect("multi-row result");
    for (index, (&want, &got)) in per_row.iter().zip(multi_row).enumerate() {
        assert_ne!(
            want.to_bits(),
            UNWRITTEN,
            "per-row output {index} unwritten"
        );
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "opt-out vs multi-row mismatch at {index}: {want:?} vs {got:?}"
        );
    }

    // Graph capture keeps the multi-row shared memory and dataflow edges.
    let capture_output = sentinel_buffer(&device, m * n);
    let mut encoder = device.command_encoder().expect("capture encoder");
    encoder.start_capture();
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        &mut registry,
        &device,
        &input_buffer,
        &weight,
        &capture_output,
        &params,
        &GgmlRoutingPolicy::default(),
    )
    .expect("capture multi-row dispatch");
    let captured = encoder.take_capture().expect("capture");
    assert_eq!(captured.len(), 1);
    match &captured[0] {
        CapturedNode::Dispatch {
            threadgroup_memory,
            reads,
            writes,
            ..
        } => {
            assert_eq!(threadgroup_memory, &vec![(0, 2 * m as u64 * 32 * 4)]);
            assert_eq!(reads.len(), 2, "weight and input must remain tracked");
            assert_eq!(writes.len(), 1, "output must remain tracked");
        }
        CapturedNode::Barrier => panic!("expected Q8_0 multi-row dispatch capture"),
    }
}
