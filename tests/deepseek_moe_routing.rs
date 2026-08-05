//! DeepSeek-V4 0731 score/hash routing parity and fail-closed gates.

#![cfg(target_vendor = "apple")]

use mlx_native::ops::deepseek_moe_routing::{
    dispatch_deepseek_moe_hash_route, dispatch_deepseek_moe_score_route, DEEPSEEK_MOE_EXPERTS,
    DEEPSEEK_MOE_ROUTE_SCALE, DEEPSEEK_MOE_TOP_K,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const E: usize = DEEPSEEK_MOE_EXPERTS;
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

fn outputs(device: &MlxDevice, tokens: usize) -> (MlxBuffer, MlxBuffer) {
    let indices = device
        .alloc_buffer(tokens * K * 4, DType::I32, vec![tokens, K])
        .unwrap();
    let weights = device
        .alloc_buffer(tokens * K * 4, DType::F32, vec![tokens, K])
        .unwrap();
    (indices, weights)
}

fn sqrt_softplus(value: f32) -> f32 {
    (value.max(0.0) + (-value.abs()).exp().ln_1p()).sqrt()
}

fn score_reference(logits: &[f32], bias: &[f32], tokens: usize) -> (Vec<i32>, Vec<f32>) {
    let mut ids = vec![-1; tokens * K];
    let mut weights = vec![0.0; tokens * K];
    for token in 0..tokens {
        let scores = logits[token * E..(token + 1) * E]
            .iter()
            .map(|&x| sqrt_softplus(x))
            .collect::<Vec<_>>();
        let mut candidates = (0..E).collect::<Vec<_>>();
        candidates.sort_by(|&a, &b| {
            (scores[b] + bias[b])
                .total_cmp(&(scores[a] + bias[a]))
                .then(a.cmp(&b))
        });
        let selected = &candidates[..K];
        let sum = selected.iter().map(|&expert| scores[expert]).sum::<f32>();
        for (slot, &expert) in selected.iter().enumerate() {
            ids[token * K + slot] = expert as i32;
            weights[token * K + slot] = scores[expert] / sum * DEEPSEEK_MOE_ROUTE_SCALE;
        }
    }
    (ids, weights)
}

fn assert_weights(got: &[f32], want: &[f32]) {
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= 2e-6,
            "weight[{index}] {got} != {want}"
        );
    }
}

#[test]
fn score_route_uses_bias_only_for_deterministic_selection() {
    let tokens = 2;
    let mut logits = vec![-8.0; tokens * E];
    for expert in 0..8 {
        logits[expert] = 2.0;
    }
    for expert in 0..E {
        logits[E + expert] = expert as f32 * 0.003 - 0.4;
    }
    let mut bias = vec![0.0; E];
    bias[200] = 20.0;
    let (want_ids, want_weights) = score_reference(&logits, &bias, tokens);
    assert_eq!(&want_ids[..K], &[200, 0, 1, 2, 3, 4]);

    let device = MlxDevice::new().unwrap();
    let logits = f32_buffer(&device, &logits, vec![tokens, E]);
    let bias = f32_buffer(&device, &bias, vec![E]);
    let (ids, weights) = outputs(&device, tokens);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_score_route(
        &mut encoder,
        &mut registry,
        &device,
        &logits,
        &bias,
        &ids,
        &weights,
        tokens,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(ids.as_slice::<i32>().unwrap(), want_ids);
    assert_weights(weights.as_slice::<f32>().unwrap(), &want_weights);
    for row in weights.as_slice::<f32>().unwrap().chunks(K) {
        assert!((row.iter().sum::<f32>() - 1.5).abs() <= 2e-6);
    }
}

#[test]
fn hash_route_preserves_checkpoint_order_duplicates_and_normalizes() {
    let tokens = 2;
    let vocab = 4;
    let token_ids = [2, 0];
    let mut table = vec![0i32; vocab * K];
    table[..K].copy_from_slice(&[5, 4, 3, 2, 1, 0]);
    table[2 * K..3 * K].copy_from_slice(&[9, 3, 9, 255, 0, 44]);
    let logits = (0..tokens * E)
        .map(|i| (i % E) as f32 * 0.007 - 0.8)
        .collect::<Vec<_>>();
    let mut want_weights = Vec::new();
    for (token, &token_id) in token_ids.iter().enumerate() {
        let selected = &table[token_id as usize * K..(token_id as usize + 1) * K];
        let scores = selected
            .iter()
            .map(|&expert| sqrt_softplus(logits[token * E + expert as usize]))
            .collect::<Vec<_>>();
        let sum = scores.iter().sum::<f32>();
        want_weights.extend(scores.iter().map(|score| score / sum * 1.5));
    }

    let device = MlxDevice::new().unwrap();
    let logits = f32_buffer(&device, &logits, vec![tokens, E]);
    let token_ids = i32_buffer(&device, &token_ids, vec![tokens]);
    let table = i32_buffer(&device, &table, vec![vocab, K]);
    let (ids, weights) = outputs(&device, tokens);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_hash_route(
        &mut encoder,
        &mut registry,
        &device,
        &logits,
        &token_ids,
        &table,
        &ids,
        &weights,
        tokens,
        vocab,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(&ids.as_slice::<i32>().unwrap()[..K], &[9, 3, 9, 255, 0, 44]);
    assert_eq!(&ids.as_slice::<i32>().unwrap()[K..], &[5, 4, 3, 2, 1, 0]);
    assert_weights(weights.as_slice::<f32>().unwrap(), &want_weights);
}

#[test]
fn dynamic_invalid_values_fail_each_route_closed() {
    let device = MlxDevice::new().unwrap();
    let mut logits = vec![0.0; E];
    logits[7] = f32::INFINITY;
    let logits = f32_buffer(&device, &logits, vec![1, E]);
    let bias = f32_buffer(&device, &vec![0.0; E], vec![E]);
    let (ids, weights) = outputs(&device, 1);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_score_route(
        &mut encoder,
        &mut registry,
        &device,
        &logits,
        &bias,
        &ids,
        &weights,
        1,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(ids.as_slice::<i32>().unwrap(), &[-1; K]);
    assert_eq!(weights.as_slice::<f32>().unwrap(), &[0.0; K]);

    let token_ids = i32_buffer(&device, &[1], vec![1]);
    let table = i32_buffer(&device, &[0, 1, 2, 3, 4, 256], vec![1, K]);
    let mut encoder = device.command_encoder().unwrap();
    dispatch_deepseek_moe_hash_route(
        &mut encoder,
        &mut registry,
        &device,
        &logits,
        &token_ids,
        &table,
        &ids,
        &weights,
        1,
        1,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    assert_eq!(ids.as_slice::<i32>().unwrap(), &[-1; K]);
    assert_eq!(weights.as_slice::<f32>().unwrap(), &[0.0; K]);
}

#[test]
fn malformed_shapes_and_dtypes_are_rejected_before_encoding() {
    let device = MlxDevice::new().unwrap();
    let logits = f32_buffer(&device, &vec![0.0; E], vec![1, E]);
    let bad_bias = i32_buffer(&device, &vec![0; E], vec![E]);
    let (ids, weights) = outputs(&device, 1);
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().unwrap();
    assert!(dispatch_deepseek_moe_score_route(
        &mut encoder,
        &mut registry,
        &device,
        &logits,
        &bad_bias,
        &ids,
        &weights,
        1,
    )
    .is_err());
    let token_ids = i32_buffer(&device, &[0], vec![1]);
    let bad_table = i32_buffer(&device, &[0; K], vec![K]);
    assert!(dispatch_deepseek_moe_hash_route(
        &mut encoder,
        &mut registry,
        &device,
        &logits,
        &token_ids,
        &bad_table,
        &ids,
        &weights,
        1,
        1,
    )
    .is_err());
}
