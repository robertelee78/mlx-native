//! Exact CPU/GPU parity and dispatch-contract tests for Q/gate deinterleave.

#![allow(clippy::expect_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    dispatch_q_gate_deinterleave_f32, CapturedNode, DType, DispatchKind, KernelRegistry, MlxBuffer,
    MlxDevice, QGateDeinterleaveParams,
};

fn exact_payload(index: usize) -> u32 {
    const SPECIALS: [u32; 8] = [
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        0x3f80_0000, // +1.0
        0xbf80_0000, // -1.0
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_1234, // quiet NaN with payload
        0x7fa1_2345, // signaling-NaN bit pattern
    ];
    if index < SPECIALS.len() {
        SPECIALS[index]
    } else {
        (index as u32)
            .wrapping_mul(0x9e37_79b9)
            .rotate_left((index % 31) as u32)
    }
}

fn cpu_reference(fused: &[u32], m: usize, n_heads: usize, head_dim: usize) -> (Vec<u32>, Vec<u32>) {
    let mut q = vec![0u32; m * n_heads * head_dim];
    let mut gate = vec![0u32; q.len()];
    for row in 0..m {
        for head in 0..n_heads {
            let vector = row * n_heads + head;
            let src = vector * 2 * head_dim;
            let dst = vector * head_dim;
            q[dst..dst + head_dim].copy_from_slice(&fused[src..src + head_dim]);
            gate[dst..dst + head_dim].copy_from_slice(&fused[src + head_dim..src + 2 * head_dim]);
        }
    }
    (q, gate)
}

fn allocate_case(
    device: &MlxDevice,
    m: usize,
    n_heads: usize,
    head_dim: usize,
) -> (MlxBuffer, MlxBuffer, MlxBuffer, Vec<u32>) {
    let input_elements = m * n_heads * 2 * head_dim;
    let output_elements = m * n_heads * head_dim;
    let input_bits: Vec<u32> = (0..input_elements).map(exact_payload).collect();

    let mut fused = device
        .alloc_buffer(
            input_elements * 4,
            DType::F32,
            vec![m, n_heads, 2 * head_dim],
        )
        .expect("allocate fused Q/gate activation");
    fused
        .as_mut_slice::<u32>()
        .expect("map fused bits")
        .copy_from_slice(&input_bits);
    let q = device
        .alloc_buffer(output_elements * 4, DType::F32, vec![m, n_heads, head_dim])
        .expect("allocate Q output");
    let gate = device
        .alloc_buffer(output_elements * 4, DType::F32, vec![m, n_heads, head_dim])
        .expect("allocate gate output");
    (fused, q, gate, input_bits)
}

fn run_parity_case(m: usize, n_heads: usize, head_dim: usize) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (fused, q, gate, input_bits) = allocate_case(&device, m, n_heads, head_dim);

    let mut encoder = device.command_encoder().expect("command encoder");
    dispatch_q_gate_deinterleave_f32(
        &mut encoder,
        &mut registry,
        &device,
        &fused,
        &q,
        &gate,
        QGateDeinterleaveParams {
            m: m as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
        },
    )
    .expect("dispatch Q/gate deinterleave");
    encoder
        .commit_and_wait_labeled("test.q_gate_deinterleave")
        .expect("complete Q/gate deinterleave");

    let (expected_q, expected_gate) = cpu_reference(&input_bits, m, n_heads, head_dim);
    assert_eq!(q.as_slice::<u32>().expect("map Q bits"), expected_q);
    assert_eq!(
        gate.as_slice::<u32>().expect("map gate bits"),
        expected_gate
    );
}

#[test]
fn decode_m1_qwen38_shape_is_byte_exact() {
    run_parity_case(1, 24, 256);
}

#[test]
fn k3_verifier_m4_qwen38_shape_is_byte_exact() {
    run_parity_case(4, 24, 256);
}

#[test]
fn odd_geometry_exercises_tail_threads_byte_exactly() {
    run_parity_case(3, 5, 259);
}

#[test]
fn production_like_prefill_is_byte_exact() {
    run_parity_case(128, 24, 256);
}

#[test]
fn capture_records_one_tracked_dispatch_with_expected_geometry() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (fused, q, gate, _) = allocate_case(&device, 4, 24, 256);
    let mut encoder = device.command_encoder().expect("command encoder");
    encoder.start_capture();

    dispatch_q_gate_deinterleave_f32(
        &mut encoder,
        &mut registry,
        &device,
        &fused,
        &q,
        &gate,
        QGateDeinterleaveParams {
            m: 4,
            n_heads: 24,
            head_dim: 256,
        },
    )
    .expect("capture Q/gate deinterleave");

    let graph = encoder.take_capture().expect("captured graph");
    assert_eq!(graph.len(), 1, "deinterleave must be one dispatch");
    let CapturedNode::Dispatch {
        pipeline,
        threads_per_grid,
        threads_per_threadgroup,
        dispatch_kind,
        reads,
        writes,
        ..
    } = &graph[0]
    else {
        panic!("expected a captured dispatch");
    };
    assert_eq!(pipeline.label(), "q_gate_deinterleave_f32");
    assert!(matches!(dispatch_kind, DispatchKind::ThreadGroups));
    assert_eq!(threads_per_grid.width, 1);
    assert_eq!(threads_per_grid.height, 24);
    assert_eq!(threads_per_grid.depth, 4);
    assert_eq!(threads_per_threadgroup.width, 256);
    assert_eq!(threads_per_threadgroup.height, 1);
    assert_eq!(threads_per_threadgroup.depth, 1);
    assert_eq!(reads.len(), 1, "fused input must be tracked as one read");
    assert_eq!(writes.len(), 2, "Q and gate must be tracked as writes");
}

#[test]
fn validation_rejects_zero_overflow_dtype_shape_and_alias_before_dispatch() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let (fused, q, gate, _) = allocate_case(&device, 1, 2, 4);
    let valid = QGateDeinterleaveParams {
        m: 1,
        n_heads: 2,
        head_dim: 4,
    };

    let mut encoder = device.command_encoder().expect("zero encoder");
    let zero = dispatch_q_gate_deinterleave_f32(
        &mut encoder,
        &mut registry,
        &device,
        &fused,
        &q,
        &gate,
        QGateDeinterleaveParams { m: 0, ..valid },
    );
    assert!(zero.is_err(), "zero m must fail");

    let mut encoder = device.command_encoder().expect("overflow encoder");
    let overflow = dispatch_q_gate_deinterleave_f32(
        &mut encoder,
        &mut registry,
        &device,
        &fused,
        &q,
        &gate,
        QGateDeinterleaveParams {
            m: u32::MAX,
            n_heads: 2,
            head_dim: 1,
        },
    );
    assert!(overflow.is_err(), "u32 index overflow must fail");

    let wrong_dtype = device
        .alloc_buffer(16 * 4, DType::U32, vec![1, 2, 8])
        .expect("wrong dtype buffer");
    let mut encoder = device.command_encoder().expect("dtype encoder");
    let dtype = dispatch_q_gate_deinterleave_f32(
        &mut encoder,
        &mut registry,
        &device,
        &wrong_dtype,
        &q,
        &gate,
        valid,
    );
    assert!(dtype.is_err(), "non-F32 input must fail");

    let wrong_shape = fused
        .with_shape(vec![1, 16])
        .expect("flat fused view with same element count");
    let mut encoder = device.command_encoder().expect("shape encoder");
    let shape = dispatch_q_gate_deinterleave_f32(
        &mut encoder,
        &mut registry,
        &device,
        &wrong_shape,
        &q,
        &gate,
        valid,
    );
    assert!(shape.is_err(), "noncanonical input shape must fail");

    let mut encoder = device.command_encoder().expect("alias encoder");
    let alias = dispatch_q_gate_deinterleave_f32(
        &mut encoder,
        &mut registry,
        &device,
        &fused,
        &q,
        &q,
        valid,
    );
    assert!(alias.is_err(), "overlapping Q/gate outputs must fail");
}
