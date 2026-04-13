//! Tests for batched dispatch kernels:
//!   - kv_cache_copy_batch_f32
//!   - moe_swiglu_batch
//!   - moe_weighted_sum

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

/// Reference GELU approximation matching the shader.
fn gelu_ref(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
    0.5 * x * (1.0 + inner.tanh())
}

// =========================================================================
// Test: kv_cache_copy_batch_f32
// =========================================================================

#[test]
fn test_kv_cache_copy_batch_f32() {
    let (device, mut registry) = setup();
    let n_heads: usize = 4;
    let head_dim: usize = 8;
    let capacity: usize = 16;
    let seq_pos: u32 = 3;

    // Source: [n_heads * head_dim] flat
    let src_data: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| (i as f32) * 0.1 + 1.0)
        .collect();
    let src_bytes = n_heads * head_dim * 4;
    let mut src_buf = device
        .alloc_buffer(src_bytes, DType::F32, vec![n_heads * head_dim])
        .unwrap();
    {
        let dst: &mut [f32] = src_buf.as_mut_slice().unwrap();
        dst.copy_from_slice(&src_data);
    }

    // Cache: [n_heads, capacity, head_dim] — zero-init
    let cache_elements = n_heads * capacity * head_dim;
    let mut cache_buf = device
        .alloc_buffer(cache_elements * 4, DType::F32, vec![n_heads, capacity, head_dim])
        .unwrap();
    {
        let dst: &mut [f32] = cache_buf.as_mut_slice().unwrap();
        dst.fill(0.0);
    }

    // Dispatch
    {
        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_batch_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &src_buf,
            &cache_buf,
            n_heads as u32,
            head_dim as u32,
            capacity as u32,
            seq_pos,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // Verify
    let cache: &[f32] = cache_buf.as_slice().unwrap();
    for h in 0..n_heads {
        for elem in 0..head_dim {
            let expected = src_data[h * head_dim + elem];
            let idx = h * capacity * head_dim + (seq_pos as usize) * head_dim + elem;
            let actual = cache[idx];
            assert!(
                (actual - expected).abs() < 1e-6,
                "head={h}, elem={elem}: expected {expected}, got {actual}"
            );
        }
        // Verify other positions are still zero
        for pos in 0..capacity {
            if pos == seq_pos as usize {
                continue;
            }
            for elem in 0..head_dim {
                let idx = h * capacity * head_dim + pos * head_dim + elem;
                assert_eq!(cache[idx], 0.0, "pos={pos} should be zero");
            }
        }
    }
}

// =========================================================================
// Test: moe_swiglu_batch
// =========================================================================

#[test]
fn test_moe_swiglu_batch() {
    let (device, mut registry) = setup();
    let top_k: usize = 4;
    let intermediate: usize = 16;

    // gate_up: [top_k, 2*intermediate]
    let gu_len = top_k * 2 * intermediate;
    let gu_data: Vec<f32> = (0..gu_len)
        .map(|i| ((i as f32) * 0.05 - 1.5))
        .collect();
    let mut gu_buf = device
        .alloc_buffer(gu_len * 4, DType::F32, vec![gu_len])
        .unwrap();
    {
        let dst: &mut [f32] = gu_buf.as_mut_slice().unwrap();
        dst.copy_from_slice(&gu_data);
    }

    // Output: [top_k, intermediate]
    let out_len = top_k * intermediate;
    let out_buf = device
        .alloc_buffer(out_len * 4, DType::F32, vec![out_len])
        .unwrap();

    // Dispatch
    {
        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &gu_buf,
            &out_buf,
            intermediate,
            top_k,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // Verify against reference
    let output: &[f32] = out_buf.as_slice().unwrap();
    for slot in 0..top_k {
        for i in 0..intermediate {
            let gate = gu_data[slot * 2 * intermediate + i];
            let up = gu_data[slot * 2 * intermediate + intermediate + i];
            let expected = gelu_ref(gate) * up;
            let actual = output[slot * intermediate + i];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-4,
                "slot={slot}, i={i}: expected {expected}, got {actual} (diff={diff})"
            );
        }
    }
}

// =========================================================================
// Test: moe_weighted_sum
// =========================================================================

#[test]
fn test_moe_weighted_sum() {
    let (device, mut registry) = setup();
    let top_k: usize = 4;
    let hidden_size: usize = 32;

    // expert_outputs: [top_k, hidden_size]
    let exp_len = top_k * hidden_size;
    let exp_data: Vec<f32> = (0..exp_len)
        .map(|i| (i as f32) * 0.1 - 5.0)
        .collect();
    let mut exp_buf = device
        .alloc_buffer(exp_len * 4, DType::F32, vec![exp_len])
        .unwrap();
    {
        let dst: &mut [f32] = exp_buf.as_mut_slice().unwrap();
        dst.copy_from_slice(&exp_data);
    }

    // weights: [top_k]
    let weights_data: Vec<f32> = vec![0.4, 0.3, 0.2, 0.1];
    let mut w_buf = device
        .alloc_buffer(top_k * 4, DType::F32, vec![top_k])
        .unwrap();
    {
        let dst: &mut [f32] = w_buf.as_mut_slice().unwrap();
        dst.copy_from_slice(&weights_data);
    }

    // Output: [hidden_size]
    let out_buf = device
        .alloc_buffer(hidden_size * 4, DType::F32, vec![hidden_size])
        .unwrap();

    // Dispatch
    {
        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &exp_buf,
            &w_buf,
            &out_buf,
            hidden_size,
            top_k,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // Verify against reference
    let output: &[f32] = out_buf.as_slice().unwrap();
    for i in 0..hidden_size {
        let mut expected = 0.0f32;
        for k in 0..top_k {
            expected += exp_data[k * hidden_size + i] * weights_data[k];
        }
        let actual = output[i];
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-4,
            "i={i}: expected {expected}, got {actual} (diff={diff})"
        );
    }
}
