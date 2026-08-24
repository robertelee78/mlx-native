//! DeepSeek-V4 0731 asymmetric SwiGLU and routed reduction parity.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_moe_activation::{
    dispatch_deepseek_moe_swiglu, dispatch_deepseek_moe_weighted_reduce, DEEPSEEK_MOE_HIDDEN_DIM,
    DEEPSEEK_MOE_INTER_DIM,
};
use mlx_native::ops::deepseek_moe_routing::{DEEPSEEK_MOE_EXPERTS, DEEPSEEK_MOE_TOP_K};
use mlx_native::{CapturedNode, DType, KernelRegistry, MlxBuffer, MlxDevice};

const I: usize = DEEPSEEK_MOE_INTER_DIM;
const H: usize = DEEPSEEK_MOE_HIDDEN_DIM;
const K: usize = DEEPSEEK_MOE_TOP_K;

fn f32_buffer(device: &MlxDevice, values: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, shape)
        .unwrap();
    buffer.as_mut_slice().unwrap().copy_from_slice(values);
    buffer
}

fn i32_buffer(device: &MlxDevice, values: &[i32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::I32, shape)
        .unwrap();
    buffer.as_mut_slice().unwrap().copy_from_slice(values);
    buffer
}

fn empty_f32(device: &MlxDevice, shape: Vec<usize>) -> MlxBuffer {
    let elements = shape.iter().product::<usize>();
    device
        .alloc_buffer(elements * 4, DType::F32, shape)
        .unwrap()
}

fn status_buffer(device: &MlxDevice) -> MlxBuffer {
    let mut status = device.alloc_buffer(4, DType::U32, vec![1]).unwrap();
    status.as_mut_slice::<u32>().unwrap()[0] = 0;
    status
}

fn swiglu_reference(gate: &[f32], up: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .enumerate()
        .map(|(index, (&gate, &up))| {
            let gate = gate.min(10.0);
            let up = up.clamp(-10.0, 10.0);
            let weight = weights.map_or(1.0, |values| values[index / I]);
            gate / (1.0 + (-gate).exp()) * up * weight
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len());
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        let delta = (got - want).abs();
        assert!(
            delta <= tolerance,
            "value[{index}] delta={delta}: {got} != {want}"
        );
    }
}

#[test]
fn capture_annotates_swiglu_dependencies() {
    let device = MlxDevice::new().unwrap();
    let gate = empty_f32(&device, vec![1, I]);
    let up = empty_f32(&device, vec![1, I]);
    let output = empty_f32(&device, vec![1, I]);
    let status = status_buffer(&device);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    encoder.start_capture();
    dispatch_deepseek_moe_swiglu(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        None,
        &output,
        &status,
        1,
    )
    .unwrap();

    let captured = encoder.take_capture().unwrap();
    assert_eq!(captured.len(), 1);
    match &captured[0] {
        CapturedNode::Dispatch { reads, writes, .. } => {
            assert_eq!(reads.len(), 3);
            assert_eq!(writes.len(), 2);
        }
        CapturedNode::Barrier => panic!("expected SwiGLU dispatch"),
    }
}

#[test]
fn asymmetric_clamped_swiglu_with_selected_weights_matches_cpu() {
    let rows = 2;
    let mut gate = (0..rows * I)
        .map(|index| (index % 97) as f32 * 0.19 - 9.0)
        .collect::<Vec<_>>();
    let mut up = (0..rows * I)
        .map(|index| (index % 89) as f32 * 0.31 - 13.0)
        .collect::<Vec<_>>();
    gate[0] = 20.0;
    up[0] = 20.0;
    gate[1] = -12.0;
    up[1] = -20.0;
    let selected_weights = [0.25, 1.5];
    let want = swiglu_reference(&gate, &up, Some(&selected_weights));

    let device = MlxDevice::new().unwrap();
    let gate = f32_buffer(&device, &gate, vec![rows, I]);
    let up = f32_buffer(&device, &up, vec![rows, I]);
    let weights = f32_buffer(&device, &selected_weights, vec![rows]);
    let output = empty_f32(&device, vec![rows, I]);
    let status = status_buffer(&device);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_swiglu(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        Some(&weights),
        &output,
        &status,
        rows,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(output.as_slice::<f32>().unwrap(), &want, 3e-5);
    assert_eq!(status.as_slice::<u32>().unwrap(), &[0]);
    let got = output.as_slice::<f32>().unwrap();
    assert!((got[0] - 10.0 / (1.0 + (-10.0f32).exp()) * 10.0 * 0.25).abs() < 3e-5);
    assert!((got[1] - (-12.0 / (1.0 + 12.0f32.exp())) * -10.0 * 0.25).abs() < 3e-5);
}

#[test]
fn swiglu_nonfinite_input_fails_only_its_row_closed() {
    let rows = 2;
    let mut gate = vec![0.5; rows * I];
    let up = vec![0.75; rows * I];
    gate[73] = f32::NAN;
    let device = MlxDevice::new().unwrap();
    let gate = f32_buffer(&device, &gate, vec![rows, I]);
    let up = f32_buffer(&device, &up, vec![rows, I]);
    let output = f32_buffer(&device, &vec![1.0; rows * I], vec![rows, I]);
    let status = status_buffer(&device);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_swiglu(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &up,
        None,
        &output,
        &status,
        rows,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let got = output.as_slice::<f32>().unwrap();
    assert!(got[..I].iter().all(|&value| value == 0.0));
    assert!(got[I..]
        .iter()
        .all(|&value| value.is_finite() && value != 0.0));
    assert_eq!(status.as_slice::<u32>().unwrap(), &[1]);
}

#[test]
fn every_swiglu_nonfinite_source_sets_the_sticky_status() {
    for source in 0..3 {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut gate = vec![0.5; I];
            let mut up = vec![0.75; I];
            let mut weights = vec![0.25];
            match source {
                0 => gate[73] = poison,
                1 => up[79] = poison,
                2 => weights[0] = poison,
                _ => unreachable!(),
            }
            let device = MlxDevice::new().unwrap();
            let gate = f32_buffer(&device, &gate, vec![1, I]);
            let up = f32_buffer(&device, &up, vec![1, I]);
            let weights = f32_buffer(&device, &weights, vec![1]);
            let output = f32_buffer(&device, &vec![1.0; I], vec![1, I]);
            let status = status_buffer(&device);
            let mut registry = KernelRegistry::new();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_moe_swiglu(
                &mut encoder,
                &mut registry,
                &device,
                &gate,
                &up,
                Some(&weights),
                &output,
                &status,
                1,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            assert!(output.as_slice::<f32>().unwrap().iter().all(|&v| v == 0.0));
            assert_eq!(status.as_slice::<u32>().unwrap(), &[1]);
        }
    }
}

fn reduce_reference(
    indices: &[i32],
    weights: &[f32],
    routed: &[f32],
    shared: &[f32],
    tokens: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; tokens * H];
    for token in 0..tokens {
        let mut order = (0..K).collect::<Vec<_>>();
        order.sort_by_key(|&slot| indices[token * K + slot]);
        for feature in 0..H {
            let mut value = 0.0f32;
            for &slot in &order {
                value = weights[token * K + slot]
                    .mul_add(routed[(token * K + slot) * H + feature], value);
            }
            output[token * H + feature] = value + shared[token * H + feature];
        }
    }
    output
}

#[test]
fn weighted_top6_reduction_and_shared_add_match_official_order() {
    let tokens = 2;
    let indices = [9, 2, 17, 2, 1, 200, 255, 6, 8, 3, 77, 4];
    let weights = [
        0.1, 0.2, 0.3, 0.15, 0.25, 0.5, 0.4, 0.1, 0.2, 0.35, 0.3, 0.15,
    ];
    let routed = (0..tokens * K * H)
        .map(|index| (index % 113) as f32 * 0.007 - 0.39)
        .collect::<Vec<_>>();
    let shared = (0..tokens * H)
        .map(|index| (index % 79) as f32 * 0.004 - 0.12)
        .collect::<Vec<_>>();
    let want = reduce_reference(&indices, &weights, &routed, &shared, tokens);

    let device = MlxDevice::new().unwrap();
    let indices = i32_buffer(&device, &indices, vec![tokens, K]);
    let weights = f32_buffer(&device, &weights, vec![tokens, K]);
    let routed = f32_buffer(&device, &routed, vec![tokens, K, H]);
    let shared = f32_buffer(&device, &shared, vec![tokens, H]);
    let output = empty_f32(&device, vec![tokens, H]);
    let status = status_buffer(&device);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_weighted_reduce(
        &mut encoder,
        &mut registry,
        &device,
        &indices,
        &weights,
        &routed,
        &shared,
        &output,
        &status,
        tokens,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_close(output.as_slice::<f32>().unwrap(), &want, 2e-6);
    assert_eq!(status.as_slice::<u32>().unwrap(), &[0]);
}

#[test]
fn reduction_invalid_id_or_nonfinite_value_fails_token_closed() {
    let tokens = 2;
    let mut indices = vec![0i32; tokens * K];
    for (slot, value) in indices.iter_mut().enumerate() {
        *value = (slot % DEEPSEEK_MOE_EXPERTS) as i32;
    }
    indices[K] = 256;
    let weights = vec![0.25; tokens * K];
    let mut routed = vec![0.5; tokens * K * H];
    routed[H + 19] = f32::INFINITY;
    let shared = vec![0.1; tokens * H];

    let device = MlxDevice::new().unwrap();
    let indices = i32_buffer(&device, &indices, vec![tokens, K]);
    let weights = f32_buffer(&device, &weights, vec![tokens, K]);
    let routed = f32_buffer(&device, &routed, vec![tokens, K, H]);
    let shared = f32_buffer(&device, &shared, vec![tokens, H]);
    let output = f32_buffer(&device, &vec![1.0; tokens * H], vec![tokens, H]);
    let status = status_buffer(&device);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_weighted_reduce(
        &mut encoder,
        &mut registry,
        &device,
        &indices,
        &weights,
        &routed,
        &shared,
        &output,
        &status,
        tokens,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert!(output
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .all(|&value| value == 0.0));
    assert_eq!(status.as_slice::<u32>().unwrap(), &[1]);
}

#[test]
fn every_reduction_failure_source_sets_the_sticky_status() {
    for source in 0..5 {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut indices = (0..K).map(|slot| slot as i32).collect::<Vec<_>>();
            let mut weights = vec![0.25; K];
            let mut routed = vec![0.5; K * H];
            let mut shared = vec![0.1; H];
            match source {
                0 => indices[0] = i32::MIN,
                1 => indices[0] = i32::MAX,
                2 => weights[0] = poison,
                3 => routed[H + 19] = poison,
                4 => shared[23] = poison,
                _ => unreachable!(),
            }
            let device = MlxDevice::new().unwrap();
            let indices = i32_buffer(&device, &indices, vec![1, K]);
            let weights = f32_buffer(&device, &weights, vec![1, K]);
            let routed = f32_buffer(&device, &routed, vec![1, K, H]);
            let shared = f32_buffer(&device, &shared, vec![1, H]);
            let output = f32_buffer(&device, &vec![1.0; H], vec![1, H]);
            let status = status_buffer(&device);
            let mut registry = KernelRegistry::new();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_deepseek_moe_weighted_reduce(
                &mut encoder,
                &mut registry,
                &device,
                &indices,
                &weights,
                &routed,
                &shared,
                &output,
                &status,
                1,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            assert!(output.as_slice::<f32>().unwrap().iter().all(|&v| v == 0.0));
            assert_eq!(status.as_slice::<u32>().unwrap(), &[1]);
        }
    }
}

#[test]
fn invalid_then_valid_activation_in_one_command_buffer_keeps_status_sticky() {
    let device = MlxDevice::new().unwrap();
    let mut bad_gate = vec![0.5; I];
    bad_gate[0] = f32::NAN;
    let bad_gate = f32_buffer(&device, &bad_gate, vec![1, I]);
    let good_gate = f32_buffer(&device, &vec![0.5; I], vec![1, I]);
    let up = f32_buffer(&device, &vec![0.75; I], vec![1, I]);
    let bad_output = empty_f32(&device, vec![1, I]);
    let good_output = empty_f32(&device, vec![1, I]);
    let status = status_buffer(&device);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_swiglu(
        &mut encoder,
        &mut registry,
        &device,
        &bad_gate,
        &up,
        None,
        &bad_output,
        &status,
        1,
    )
    .unwrap();
    dispatch_deepseek_moe_swiglu(
        &mut encoder,
        &mut registry,
        &device,
        &good_gate,
        &up,
        None,
        &good_output,
        &status,
        1,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(status.as_slice::<u32>().unwrap(), &[1]);
    assert!(bad_output
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .all(|&v| v == 0.0));
    assert!(good_output
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .all(|&v| v.is_finite() && v != 0.0));
}

#[test]
fn malformed_activation_and_reduction_buffers_are_rejected() {
    let device = MlxDevice::new().unwrap();
    let gate = f32_buffer(&device, &vec![0.0; I], vec![1, I]);
    let bad_up = f32_buffer(&device, &vec![0.0; I], vec![I]);
    let output = empty_f32(&device, vec![1, I]);
    let status = status_buffer(&device);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    assert!(dispatch_deepseek_moe_swiglu(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &bad_up,
        None,
        &output,
        &status,
        1,
    )
    .is_err());
    let bad_status = f32_buffer(&device, &[0.0], vec![1]);
    assert!(dispatch_deepseek_moe_swiglu(
        &mut encoder,
        &mut registry,
        &device,
        &gate,
        &gate,
        None,
        &output,
        &bad_status,
        1,
    )
    .is_err());

    let indices = i32_buffer(&device, &[0; K], vec![1, K]);
    let weights = f32_buffer(&device, &[0.25; K], vec![1, K]);
    let routed = f32_buffer(&device, &vec![0.0; K * H], vec![K, H]);
    let shared = f32_buffer(&device, &vec![0.0; H], vec![1, H]);
    let reduced = empty_f32(&device, vec![1, H]);
    assert!(dispatch_deepseek_moe_weighted_reduce(
        &mut encoder,
        &mut registry,
        &device,
        &indices,
        &weights,
        &routed,
        &shared,
        &reduced,
        &status,
        1,
    )
    .is_err());
    assert!(dispatch_deepseek_moe_weighted_reduce(
        &mut encoder,
        &mut registry,
        &device,
        &indices,
        &weights,
        &f32_buffer(&device, &vec![0.0; K * H], vec![1, K, H]),
        &shared,
        &reduced,
        &bad_status,
        1,
    )
    .is_err());
}
