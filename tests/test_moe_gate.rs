//! Tests for the MoE gate (top-K expert routing with softmax) GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice, MlxBuffer};
use mlx_native::ops::moe_gate::{moe_gate, MoeGateParams};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

fn f32_to_bf16_bytes(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let bf16 = (bits >> 16) as u16;
    bf16.to_le_bytes()
}

fn alloc_bf16(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let n = data.len();
    let mut buf = device
        .alloc_buffer(n * 2, DType::BF16, vec![n])
        .expect("alloc bf16");
    let bytes = buf.as_mut_slice::<u8>().expect("write bf16");
    for (i, &v) in data.iter().enumerate() {
        let b = f32_to_bf16_bytes(v);
        bytes[i * 2] = b[0];
        bytes[i * 2 + 1] = b[1];
    }
    buf
}

fn alloc_f32(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .expect("alloc f32");
    buf.as_mut_slice::<f32>().expect("w").copy_from_slice(data);
    buf
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let rms = (ss + eps).sqrt();
    x.iter().zip(weight).map(|(&xi, &wi)| (xi / rms) * wi).collect()
}

#[allow(dead_code)]
fn softmax(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

#[test]
fn test_moe_gate_basic() {
    let (device, mut registry) = setup();

    let hidden_dim = 16;
    let n_experts = 8;
    let top_k = 3;
    let rms_eps = 1e-6f32;

    let hidden_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1).collect();
    let norm_weight: Vec<f32> = vec![1.0; hidden_dim];
    let per_expert_scale: Vec<f32> = vec![1.0; n_experts];

    let mut router_data = vec![0.0f32; n_experts * hidden_dim];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            router_data[e * hidden_dim + d] = match e {
                5 => 2.0,
                2 => 1.5,
                7 => 1.0,
                _ => 0.1,
            };
        }
    }

    // Reference: RMS-norm the hidden state, then router matmul, then top-K + softmax
    let normed = rms_norm(&hidden_data, &norm_weight, rms_eps);
    let mut ref_logits = vec![0.0f32; n_experts];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            ref_logits[e] += normed[d] * router_data[e * hidden_dim + d];
        }
    }

    let mut indexed: Vec<(usize, f32)> = ref_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_indices: Vec<u32> = indexed[..top_k].iter().map(|&(i, _)| i as u32).collect();

    // GPU buffers — hidden_state must be bf16
    let hidden_buf = alloc_bf16(&device, &hidden_data);
    let router_buf = alloc_f32(&device, &router_data);
    let norm_weight_buf = alloc_f32(&device, &norm_weight);
    let per_expert_scale_buf = alloc_f32(&device, &per_expert_scale);
    let expert_ids_buf = device
        .alloc_buffer(top_k * 4, DType::U32, vec![top_k])
        .expect("ids");
    let weights_buf = device
        .alloc_buffer(top_k * 4, DType::F32, vec![top_k])
        .expect("weights");

    let mut encoder = device.command_encoder().expect("encoder");
    moe_gate(
        &mut encoder, &mut registry, device.metal_device(),
        &hidden_buf, &router_buf, &norm_weight_buf, &per_expert_scale_buf,
        &expert_ids_buf, &weights_buf,
        &MoeGateParams { hidden_dim, n_experts, top_k, seq_len: 1, rms_eps },
    ).expect("moe_gate");
    encoder.commit_and_wait().expect("commit");

    let out_ids: &[u32] = expert_ids_buf.as_slice().expect("read ids");
    assert_eq!(out_ids[0], top_indices[0], "top-1 expert should be {}", top_indices[0]);
    assert_eq!(out_ids[1], top_indices[1], "top-2 expert should be {}", top_indices[1]);
    assert_eq!(out_ids[2], top_indices[2], "top-3 expert should be {}", top_indices[2]);

    let out_weights: &[f32] = weights_buf.as_slice().expect("read weights");
    let sum: f32 = out_weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "weights should sum to 1.0, got {}",
        sum
    );
    for k in 0..top_k {
        assert!(
            out_weights[k] > 0.0,
            "weight {} should be positive, got {}",
            k, out_weights[k]
        );
    }
}

#[test]
fn test_moe_gate_128_experts_top8() {
    let (device, mut registry) = setup();

    let hidden_dim = 32;
    let n_experts = 128;
    let top_k = 8;
    let rms_eps = 1e-6f32;

    let hidden_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 - 16.0) * 0.01).collect();
    let norm_weight: Vec<f32> = vec![1.0; hidden_dim];
    let per_expert_scale: Vec<f32> = vec![1.0; n_experts];

    let mut router_data = vec![0.0f32; n_experts * hidden_dim];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            router_data[e * hidden_dim + d] = (e as f32) * 0.01;
        }
    }

    // Reference: RMS-norm + router matmul + top-K
    let normed = rms_norm(&hidden_data, &norm_weight, rms_eps);
    let mut ref_logits = vec![0.0f32; n_experts];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            ref_logits[e] += normed[d] * router_data[e * hidden_dim + d];
        }
    }
    let mut indexed: Vec<(usize, f32)> = ref_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let ref_top_ids: Vec<u32> = indexed[..top_k].iter().map(|&(i, _)| i as u32).collect();

    let hidden_buf = alloc_bf16(&device, &hidden_data);
    let router_buf = alloc_f32(&device, &router_data);
    let norm_weight_buf = alloc_f32(&device, &norm_weight);
    let per_expert_scale_buf = alloc_f32(&device, &per_expert_scale);
    let expert_ids_buf = device
        .alloc_buffer(top_k * 4, DType::U32, vec![top_k])
        .expect("ids");
    let weights_buf = device
        .alloc_buffer(top_k * 4, DType::F32, vec![top_k])
        .expect("weights");

    let mut encoder = device.command_encoder().expect("encoder");
    moe_gate(
        &mut encoder, &mut registry, device.metal_device(),
        &hidden_buf, &router_buf, &norm_weight_buf, &per_expert_scale_buf,
        &expert_ids_buf, &weights_buf,
        &MoeGateParams { hidden_dim, n_experts, top_k, seq_len: 1, rms_eps },
    ).expect("moe_gate");
    encoder.commit_and_wait().expect("commit");

    let out_ids: &[u32] = expert_ids_buf.as_slice().expect("read ids");
    for k in 0..top_k {
        assert_eq!(
            out_ids[k], ref_top_ids[k],
            "top-{} expert mismatch: expected {}, got {}",
            k + 1, ref_top_ids[k], out_ids[k]
        );
    }

    let out_weights: &[f32] = weights_buf.as_slice().expect("read weights");
    let sum: f32 = out_weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "weights should sum to 1.0, got {}",
        sum
    );
    for k in 0..top_k {
        assert!(
            out_weights[k] > 0.0,
            "weight {} should be positive, got {}",
            k, out_weights[k]
        );
    }
}

// ---- Validation tests ----

#[test]
fn test_moe_gate_topk_gt_nexperts() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(64, DType::F32, vec![16]).expect("buf");
    let ids = device.alloc_buffer(64, DType::U32, vec![16]).expect("ids");
    let mut encoder = device.command_encoder().expect("enc");

    let result = moe_gate(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, &buf, &buf, &ids, &buf,
        &MoeGateParams { hidden_dim: 4, n_experts: 4, top_k: 8, seq_len: 1, rms_eps: 1e-6 },
    );
    assert!(result.is_err(), "top_k > n_experts should error");
}

#[test]
fn test_moe_gate_too_many_experts() {
    let (device, mut registry) = setup();

    let buf = device.alloc_buffer(64, DType::F32, vec![16]).expect("buf");
    let ids = device.alloc_buffer(64, DType::U32, vec![16]).expect("ids");
    let mut encoder = device.command_encoder().expect("enc");

    let result = moe_gate(
        &mut encoder, &mut registry, device.metal_device(),
        &buf, &buf, &buf, &buf, &ids, &buf,
        &MoeGateParams { hidden_dim: 4, n_experts: 256, top_k: 1, seq_len: 1, rms_eps: 1e-6 },
    );
    assert!(result.is_err(), "n_experts > 128 should error");
}
