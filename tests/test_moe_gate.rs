//! Tests for the MoE gate (top-K expert routing with softmax) GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::{DType, KernelRegistry, MlxDevice};
use mlx_native::ops::moe_gate::{moe_gate, MoeGateParams};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

/// Reference softmax in Rust.
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

    // Create a simple hidden state
    let hidden_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1).collect();

    // Create router weights: each expert has hidden_dim weights
    // Make expert 5 have highest dot product, expert 2 second, expert 7 third
    let mut router_data = vec![0.0f32; n_experts * hidden_dim];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            router_data[e * hidden_dim + d] = match e {
                5 => 2.0,  // highest
                2 => 1.5,  // second
                7 => 1.0,  // third
                _ => 0.1,
            };
        }
    }

    // Compute reference logits
    let mut ref_logits = vec![0.0f32; n_experts];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            ref_logits[e] += hidden_data[d] * router_data[e * hidden_dim + d];
        }
    }

    // Find top-k
    let mut indexed: Vec<(usize, f32)> = ref_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_indices: Vec<u32> = indexed[..top_k].iter().map(|&(i, _)| i as u32).collect();
    let top_logits: Vec<f32> = indexed[..top_k].iter().map(|&(_, v)| v).collect();
    let ref_weights = softmax(&top_logits);

    // Create GPU buffers
    let mut hidden_buf = device
        .alloc_buffer(hidden_dim * 4, DType::F32, vec![hidden_dim])
        .expect("hidden");
    hidden_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&hidden_data);

    let mut router_buf = device
        .alloc_buffer(n_experts * hidden_dim * 4, DType::F32, vec![n_experts, hidden_dim])
        .expect("router");
    router_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&router_data);

    let expert_ids_buf = device
        .alloc_buffer(top_k * 4, DType::U32, vec![top_k])
        .expect("ids");
    let weights_buf = device
        .alloc_buffer(top_k * 4, DType::F32, vec![top_k])
        .expect("weights");

    // Dispatch
    let mut encoder = device.command_encoder().expect("encoder");
    moe_gate(
        &mut encoder, &mut registry, device.metal_device(),
        &hidden_buf, &router_buf, &expert_ids_buf, &weights_buf,
        &MoeGateParams { hidden_dim, n_experts, top_k },
    ).expect("moe_gate");
    encoder.commit_and_wait().expect("commit");

    // Verify expert IDs
    let out_ids: &[u32] = expert_ids_buf.as_slice().expect("read ids");
    assert_eq!(out_ids[0], top_indices[0], "top-1 expert should be {}", top_indices[0]);
    assert_eq!(out_ids[1], top_indices[1], "top-2 expert should be {}", top_indices[1]);
    assert_eq!(out_ids[2], top_indices[2], "top-3 expert should be {}", top_indices[2]);

    // Verify softmax weights
    let out_weights: &[f32] = weights_buf.as_slice().expect("read weights");
    for k in 0..top_k {
        let diff = (out_weights[k] - ref_weights[k]).abs();
        assert!(
            diff < 1e-5,
            "weight mismatch at k={}: expected {}, got {}, diff {}",
            k, ref_weights[k], out_weights[k], diff
        );
    }

    // Verify weights sum to 1
    let sum: f32 = out_weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "weights should sum to 1.0, got {}",
        sum
    );
}

#[test]
fn test_moe_gate_128_experts_top8() {
    let (device, mut registry) = setup();

    // Gemma 4 dimensions
    let hidden_dim = 32; // reduced for test speed
    let n_experts = 128;
    let top_k = 8;

    let hidden_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 - 16.0) * 0.01).collect();

    // Each expert gets distinct weights so top-K selection is deterministic
    let mut router_data = vec![0.0f32; n_experts * hidden_dim];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            // Make expert e have weight proportional to e
            router_data[e * hidden_dim + d] = (e as f32) * 0.01;
        }
    }

    // Create buffers
    let mut hidden_buf = device
        .alloc_buffer(hidden_dim * 4, DType::F32, vec![hidden_dim])
        .expect("hidden");
    hidden_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&hidden_data);

    let mut router_buf = device
        .alloc_buffer(n_experts * hidden_dim * 4, DType::F32, vec![n_experts, hidden_dim])
        .expect("router");
    router_buf.as_mut_slice::<f32>().expect("w").copy_from_slice(&router_data);

    let expert_ids_buf = device
        .alloc_buffer(top_k * 4, DType::U32, vec![top_k])
        .expect("ids");
    let weights_buf = device
        .alloc_buffer(top_k * 4, DType::F32, vec![top_k])
        .expect("weights");

    let mut encoder = device.command_encoder().expect("encoder");
    moe_gate(
        &mut encoder, &mut registry, device.metal_device(),
        &hidden_buf, &router_buf, &expert_ids_buf, &weights_buf,
        &MoeGateParams { hidden_dim, n_experts, top_k },
    ).expect("moe_gate");
    encoder.commit_and_wait().expect("commit");

    let out_ids: &[u32] = expert_ids_buf.as_slice().expect("read ids");
    let out_weights: &[f32] = weights_buf.as_slice().expect("read weights");

    // The top-8 experts should be the ones with the highest indices
    // (since weight is proportional to expert index and hidden_data has positive sum)
    // Compute reference logits to determine sign
    let mut ref_logits = vec![0.0f32; n_experts];
    for e in 0..n_experts {
        for d in 0..hidden_dim {
            ref_logits[e] += hidden_data[d] * router_data[e * hidden_dim + d];
        }
    }

    let mut indexed: Vec<(usize, f32)> = ref_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let ref_top_ids: Vec<u32> = indexed[..top_k].iter().map(|&(i, _)| i as u32).collect();

    for k in 0..top_k {
        assert_eq!(
            out_ids[k], ref_top_ids[k],
            "top-{} expert mismatch: expected {}, got {}",
            k + 1, ref_top_ids[k], out_ids[k]
        );
    }

    // Verify weights sum to 1
    let sum: f32 = out_weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "weights should sum to 1.0, got {}",
        sum
    );

    // All weights should be positive
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
        &buf, &buf, &ids, &buf,
        &MoeGateParams { hidden_dim: 4, n_experts: 4, top_k: 8 },
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
        &buf, &buf, &ids, &buf,
        &MoeGateParams { hidden_dim: 4, n_experts: 256, top_k: 1 },
    );
    assert!(result.is_err(), "n_experts > 128 should error");
}
