//! Tests for the Wave 2F bf16 kernel ports.
//!
//! Each new bf16 kernel is tested against a CPU reference that performs the
//! same operation in f32, then rounds the result to bf16 precision.
//!
//! Tolerance: atol=5e-3, rtol=2e-2 (same as flash_attn_prefill bf16 tests).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_batch_bf16;
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_bf16;
use mlx_native::ops::moe_dispatch::{
    fused_gelu_mul_bf16_encode, moe_swiglu_seq_bf16_encode,
    moe_weighted_sum_seq_bf16_input_encode,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---------------------------------------------------------------------------
// bf16 / f32 byte-level helpers (identical to test_fused_ops.rs pattern)
// ---------------------------------------------------------------------------

fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round-trip a float through bf16 precision.
fn f32_to_bf16_round(val: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(val))
}

fn write_bf16(buf: &mut [u8], values: &[f32]) {
    for (i, &v) in values.iter().enumerate() {
        let bytes = f32_to_bf16_bits(v).to_le_bytes();
        buf[i * 2] = bytes[0];
        buf[i * 2 + 1] = bytes[1];
    }
}

fn read_bf16(buf: &[u8], count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| bf16_bits_to_f32(u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]])))
        .collect()
}

fn read_f32(buf: &[u8], count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            f32::from_le_bytes([
                buf[i * 4],
                buf[i * 4 + 1],
                buf[i * 4 + 2],
                buf[i * 4 + 3],
            ])
        })
        .collect()
}

fn write_f32(buf: &mut [u8], values: &[f32]) {
    for (i, &v) in values.iter().enumerate() {
        let bytes = v.to_le_bytes();
        buf[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
}

fn write_u32(buf: &mut [u8], values: &[u32]) {
    for (i, &v) in values.iter().enumerate() {
        let bytes = v.to_le_bytes();
        buf[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
}

/// Simple deterministic pseudo-random f32 in [-0.5, 0.5].
fn pseudo_rand(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Allocate an MlxBuffer of `n` bf16 elements, writing `data` into it.
fn alloc_bf16(device: &MlxDevice, data: &[f32]) -> mlx_native::MlxBuffer {
    let n = data.len();
    let byte_len = n * 2;
    let mut buf = device
        .alloc_buffer(byte_len, DType::BF16, vec![n])
        .expect("alloc bf16");
    let bytes = buf.as_mut_slice::<u8>().expect("as_mut_slice");
    write_bf16(bytes, data);
    buf
}

/// Allocate an MlxBuffer of `n` f32 elements, writing `data` into it.
fn alloc_f32(device: &MlxDevice, data: &[f32]) -> mlx_native::MlxBuffer {
    let n = data.len();
    let byte_len = n * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![n])
        .expect("alloc f32");
    let bytes = buf.as_mut_slice::<u8>().expect("as_mut_slice");
    write_f32(bytes, data);
    buf
}

/// Allocate an MlxBuffer of `n` u32 elements, writing `data` into it.
fn alloc_u32(device: &MlxDevice, data: &[u32]) -> mlx_native::MlxBuffer {
    let n = data.len();
    let byte_len = n * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::U32, vec![n])
        .expect("alloc u32");
    let bytes = buf.as_mut_slice::<u8>().expect("as_mut_slice");
    write_u32(bytes, data);
    buf
}

/// Allocate a zeroed MlxBuffer.
fn alloc_zeroed(device: &MlxDevice, byte_len: usize, dtype: DType, n: usize) -> mlx_native::MlxBuffer {
    let mut buf = device
        .alloc_buffer(byte_len, dtype, vec![n])
        .expect("alloc zeroed");
    let bytes = buf.as_mut_slice::<u8>().expect("as_mut_slice");
    bytes.fill(0);
    buf
}

/// Check two float arrays are close; bf16 tolerance.
fn assert_allclose(got: &[f32], want: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{}: length mismatch", label);
    let mut max_err: f32 = 0.0;
    let mut fail_count = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs();
        let tol = atol + rtol * w.abs();
        if err > tol {
            if fail_count < 5 {
                eprintln!(
                    "{}: idx={} got={:.6} want={:.6} err={:.6} tol={:.6}",
                    label, i, g, w, err, tol
                );
            }
            fail_count += 1;
        }
        if err > max_err {
            max_err = err;
        }
    }
    if fail_count > 0 {
        panic!(
            "{}: {} elements out of tolerance (max_err={:.6})",
            label, fail_count, max_err
        );
    }
}

// ---------------------------------------------------------------------------
// Test 1: fused_head_norm_rope_batch_bf16
// ---------------------------------------------------------------------------

/// CPU reference: RMS norm per head + NeoX RoPE rotation on batched input.
/// Input layout: [seq_len, n_heads, head_dim] (token-major).
fn cpu_fused_head_norm_rope_batch(
    input: &[f32],
    norm_weight: Option<&[f32]>,
    positions: &[u32],
    n_heads: usize,
    head_dim: usize,
    half_rope: usize,
    eps: f32,
    theta: f32,
) -> Vec<f32> {
    let seq_len = positions.len();
    let half_dim = head_dim / 2;
    let mut output = vec![0.0f32; seq_len * n_heads * head_dim];

    for tok in 0..seq_len {
        let pos = positions[tok];
        for h in 0..n_heads {
            let base = (tok * n_heads + h) * head_dim;
            let head_in = &input[base..base + head_dim];

            // RMS norm
            let sq_sum: f32 = head_in.iter().map(|x| x * x).sum();
            let rms_inv = 1.0 / (sq_sum / head_dim as f32 + eps).sqrt();

            let mut normed = vec![0.0f32; head_dim];
            for i in 0..head_dim {
                normed[i] = head_in[i] * rms_inv;
                if let Some(w) = norm_weight {
                    normed[i] *= w[i];
                }
            }

            // NeoX RoPE: pairs (normed[i], normed[i + half_dim]) for i < half_rope.
            // Uses head_dim as denominator (ProportionalRoPE).
            let mut out_head = normed.clone();
            for i in 0..half_rope {
                let x0 = normed[i];
                let x1 = normed[i + half_dim];
                let dim_ratio = (2 * i) as f32 / head_dim as f32;
                let freq = pos as f32 / theta.powf(dim_ratio);
                let cos_a = freq.cos();
                let sin_a = freq.sin();
                out_head[i] = x0 * cos_a - x1 * sin_a;
                out_head[i + half_dim] = x1 * cos_a + x0 * sin_a;
            }
            output[base..base + head_dim].copy_from_slice(&out_head);
        }
    }
    output
}

#[test]
fn test_fused_head_norm_rope_batch_bf16_with_weight() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let seq_len: usize = 4;
    let n_heads: usize = 8;
    let head_dim: usize = 64;
    let half_rope: usize = 32; // = head_dim / 2
    let eps: f32 = 1e-6;
    let theta: f32 = 1_000_000.0;

    let total = seq_len * n_heads * head_dim;

    // Random input rounded to bf16 precision
    let input_f32: Vec<f32> = pseudo_rand(42, total)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();
    let norm_weight_f32: Vec<f32> = pseudo_rand(99, head_dim)
        .into_iter()
        .map(|x| f32_to_bf16_round(x + 1.0)) // shift to ~[0.5, 1.5]
        .collect();
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    // CPU reference, then round to bf16
    let cpu_ref_f32 = cpu_fused_head_norm_rope_batch(
        &input_f32,
        Some(&norm_weight_f32),
        &positions,
        n_heads,
        head_dim,
        half_rope,
        eps,
        theta,
    );
    let cpu_ref: Vec<f32> = cpu_ref_f32.iter().map(|&x| f32_to_bf16_round(x)).collect();

    // GPU buffers
    let input_buf = alloc_bf16(&device, &input_f32);
    let output_buf = alloc_zeroed(&device, total * 2, DType::BF16, total);
    let weight_buf = alloc_bf16(&device, &norm_weight_f32);
    let pos_buf = alloc_u32(&device, &positions);

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_fused_head_norm_rope_batch_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        Some(&weight_buf),
        &pos_buf,
        None, // no freq_factors
        n_heads as u32,
        head_dim as u32,
        half_rope as u32,
        seq_len as u32,
        eps,
        theta,
    )
    .expect("dispatch_fused_head_norm_rope_batch_bf16");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bytes = output_buf.as_slice::<u8>().expect("read output");
    let got = read_bf16(out_bytes, total);

    assert_allclose(&got, &cpu_ref, 5e-3, 2e-2, "fused_head_norm_rope_batch_bf16_with_weight");
}

#[test]
fn test_fused_head_norm_rope_batch_bf16_no_weight() {
    // V-norm path: no weight scale, half_rope = 0 (no rotation)
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let seq_len: usize = 2;
    let n_heads: usize = 4;
    let head_dim: usize = 32;
    let half_rope: usize = 0; // no rotation
    let eps: f32 = 1e-6;
    let theta: f32 = 10_000.0;

    let total = seq_len * n_heads * head_dim;
    let input_f32: Vec<f32> = pseudo_rand(7, total)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();
    let positions: Vec<u32> = vec![10, 11];

    let cpu_ref_f32 = cpu_fused_head_norm_rope_batch(
        &input_f32, None, &positions, n_heads, head_dim, half_rope, eps, theta,
    );
    let cpu_ref: Vec<f32> = cpu_ref_f32.iter().map(|&x| f32_to_bf16_round(x)).collect();

    let input_buf = alloc_bf16(&device, &input_f32);
    let output_buf = alloc_zeroed(&device, total * 2, DType::BF16, total);
    let pos_buf = alloc_u32(&device, &positions);

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_fused_head_norm_rope_batch_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        None,
        &pos_buf,
        None,
        n_heads as u32,
        head_dim as u32,
        half_rope as u32,
        seq_len as u32,
        eps,
        theta,
    )
    .expect("dispatch_fused_head_norm_rope_batch_bf16 (no weight)");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bytes = output_buf.as_slice::<u8>().expect("read output");
    let got = read_bf16(out_bytes, total);

    assert_allclose(&got, &cpu_ref, 5e-3, 2e-2, "fused_head_norm_rope_batch_bf16_no_weight");
}

// ---------------------------------------------------------------------------
// Test 2: fused_gelu_mul_bf16
// ---------------------------------------------------------------------------

fn gelu_approx(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let x3 = x * x * x;
    let inner = (sqrt_2_over_pi * (x + 0.044715 * x3)).clamp(-15.0, 15.0);
    0.5 * x * (1.0 + inner.tanh())
}

#[test]
fn test_fused_gelu_mul_bf16_basic() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n: usize = 256;
    let gate_f32: Vec<f32> = pseudo_rand(1, n)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();
    let up_f32: Vec<f32> = pseudo_rand(2, n)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();

    // CPU reference: compute in f32, round to bf16
    let cpu_ref: Vec<f32> = gate_f32
        .iter()
        .zip(up_f32.iter())
        .map(|(&g, &u)| f32_to_bf16_round(gelu_approx(g) * u))
        .collect();

    let gate_buf = alloc_bf16(&device, &gate_f32);
    let up_buf = alloc_bf16(&device, &up_f32);
    let out_buf = alloc_zeroed(&device, n * 2, DType::BF16, n);

    let mut encoder = device.command_encoder().expect("encoder");
    fused_gelu_mul_bf16_encode(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &gate_buf,
        &up_buf,
        &out_buf,
        n,
    )
    .expect("fused_gelu_mul_bf16_encode");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bytes = out_buf.as_slice::<u8>().expect("read output");
    let got = read_bf16(out_bytes, n);

    assert_allclose(&got, &cpu_ref, 5e-3, 2e-2, "fused_gelu_mul_bf16");
}

// ---------------------------------------------------------------------------
// Test 3: moe_swiglu_seq_bf16
// ---------------------------------------------------------------------------

#[test]
fn test_moe_swiglu_seq_bf16_basic() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n_tokens: usize = 3;
    let top_k: usize = 2;
    let intermediate: usize = 16;

    let total_in = n_tokens * top_k * 2 * intermediate;
    let total_out = n_tokens * top_k * intermediate;

    let gate_up_f32: Vec<f32> = pseudo_rand(3, total_in)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();

    // CPU reference: SwiGLU = GELU(gate) * up, rounded to bf16
    let mut cpu_ref = vec![0.0f32; total_out];
    for tok in 0..n_tokens {
        for slot in 0..top_k {
            let slot_base = (tok * top_k + slot) * 2 * intermediate;
            let out_base = (tok * top_k + slot) * intermediate;
            for i in 0..intermediate {
                let gate = gate_up_f32[slot_base + i];
                let up = gate_up_f32[slot_base + intermediate + i];
                cpu_ref[out_base + i] = f32_to_bf16_round(gelu_approx(gate) * up);
            }
        }
    }

    let gate_up_buf = alloc_bf16(&device, &gate_up_f32);
    let out_buf = alloc_zeroed(&device, total_out * 2, DType::BF16, total_out);

    let mut encoder = device.command_encoder().expect("encoder");
    moe_swiglu_seq_bf16_encode(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &gate_up_buf,
        &out_buf,
        intermediate,
        top_k,
        n_tokens,
    )
    .expect("moe_swiglu_seq_bf16_encode");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bytes = out_buf.as_slice::<u8>().expect("read output");
    let got = read_bf16(out_bytes, total_out);

    assert_allclose(&got, &cpu_ref, 5e-3, 2e-2, "moe_swiglu_seq_bf16");
}

// ---------------------------------------------------------------------------
// Test 4: kv_cache_copy_seq_bf16
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_copy_seq_bf16_linear() {
    // Copy bf16 source → f32 cache, no wrap, src_tok_offset=0.
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n_heads: usize = 4;
    let head_dim: usize = 16;
    let n_tokens: usize = 6;
    let capacity: usize = 32;
    let src_tok_offset: usize = 0;
    let seq_pos_start: usize = 0;

    let total_src = n_tokens * n_heads * head_dim;
    let total_cache = n_heads * capacity * head_dim;

    // Values already bf16-rounded so the cast to f32 is exact.
    let src_f32: Vec<f32> = pseudo_rand(5, total_src)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();

    let src_buf = alloc_bf16(&device, &src_f32);
    let cache_buf = alloc_zeroed(&device, total_cache * 4, DType::F32, total_cache);

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_kv_cache_copy_seq_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        n_heads as u32,
        head_dim as u32,
        capacity as u32,
        seq_pos_start as u32,
        n_tokens as u32,
        src_tok_offset as u32,
    )
    .expect("dispatch_kv_cache_copy_seq_bf16");
    encoder.commit_and_wait().expect("commit_and_wait");

    let cache_bytes = cache_buf.as_slice::<u8>().expect("read cache");
    let cache_vals = read_f32(cache_bytes, total_cache);

    // Verify each element copied to the correct cache slot.
    for tok in 0..n_tokens {
        for head in 0..n_heads {
            for elem in 0..head_dim {
                let src_idx = tok * (n_heads * head_dim) + head * head_dim + elem;
                let dst_pos = seq_pos_start + tok;
                let dst_idx = head * capacity * head_dim + dst_pos * head_dim + elem;
                let expected = src_f32[src_idx];
                let got = cache_vals[dst_idx];
                // bf16 → f32 promotion is exact for values that round-tripped through bf16.
                assert!(
                    (got - expected).abs() < 1e-5,
                    "kv_cache_copy_seq_bf16 mismatch at tok={} head={} elem={}: got={} want={}",
                    tok, head, elem, got, expected
                );
            }
        }
    }
}

#[test]
fn test_kv_cache_copy_seq_bf16_with_offset() {
    // src_tok_offset=2: writes tokens [2, 3] from a 4-token source buffer.
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n_heads: usize = 2;
    let head_dim: usize = 8;
    let n_src_tokens: usize = 4;
    let n_tokens: usize = 2;
    let src_tok_offset: usize = 2;
    let seq_pos_start: usize = 0;
    let capacity: usize = 16;

    let total_src = n_src_tokens * n_heads * head_dim;
    let total_cache = n_heads * capacity * head_dim;

    let src_f32: Vec<f32> = pseudo_rand(6, total_src)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();

    let src_buf = alloc_bf16(&device, &src_f32);
    let cache_buf = alloc_zeroed(&device, total_cache * 4, DType::F32, total_cache);

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_kv_cache_copy_seq_bf16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &src_buf,
        &cache_buf,
        n_heads as u32,
        head_dim as u32,
        capacity as u32,
        seq_pos_start as u32,
        n_tokens as u32,
        src_tok_offset as u32,
    )
    .expect("dispatch_kv_cache_copy_seq_bf16 (offset)");
    encoder.commit_and_wait().expect("commit_and_wait");

    let cache_bytes = cache_buf.as_slice::<u8>().expect("read cache");
    let cache_vals = read_f32(cache_bytes, total_cache);

    for tok in 0..n_tokens {
        for head in 0..n_heads {
            for elem in 0..head_dim {
                let src_tok = src_tok_offset + tok;
                let src_idx = src_tok * (n_heads * head_dim) + head * head_dim + elem;
                let dst_pos = seq_pos_start + tok;
                let dst_idx = head * capacity * head_dim + dst_pos * head_dim + elem;
                let expected = src_f32[src_idx];
                let got = cache_vals[dst_idx];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "kv_cache_copy_seq_bf16 (offset) mismatch at tok={} head={} elem={}: got={} want={}",
                    tok, head, elem, got, expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 5: moe_weighted_sum_seq_bf16_input
// ---------------------------------------------------------------------------

#[test]
fn test_moe_weighted_sum_seq_bf16_input_basic() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n_tokens: usize = 3;
    let top_k: usize = 4;
    let hidden_size: usize = 32;

    let total_expert = n_tokens * top_k * hidden_size;
    let total_weights = n_tokens * top_k;
    let total_out = n_tokens * hidden_size;

    // Expert outputs in bf16 (already rounded)
    let expert_f32: Vec<f32> = pseudo_rand(10, total_expert)
        .into_iter()
        .map(f32_to_bf16_round)
        .collect();
    // Routing weights in f32 (small positive values)
    let weights_f32: Vec<f32> = pseudo_rand(11, total_weights)
        .into_iter()
        .map(|x| (x + 0.5).abs())
        .collect();

    // CPU reference: for each (tok, d), sum over k of expert[tok,k,d] * weight[tok,k]
    let mut cpu_ref = vec![0.0f32; total_out];
    for tok in 0..n_tokens {
        for d in 0..hidden_size {
            let mut sum = 0.0f32;
            for k in 0..top_k {
                let in_idx = (tok * top_k + k) * hidden_size + d;
                let w_idx = tok * top_k + k;
                sum += expert_f32[in_idx] * weights_f32[w_idx];
            }
            cpu_ref[tok * hidden_size + d] = sum;
        }
    }

    let expert_buf = alloc_bf16(&device, &expert_f32);
    let weights_buf = alloc_f32(&device, &weights_f32);
    let out_buf = alloc_zeroed(&device, total_out * 4, DType::F32, total_out);

    let mut encoder = device.command_encoder().expect("encoder");
    moe_weighted_sum_seq_bf16_input_encode(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &expert_buf,
        &weights_buf,
        &out_buf,
        hidden_size,
        top_k,
        n_tokens,
    )
    .expect("moe_weighted_sum_seq_bf16_input_encode");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bytes = out_buf.as_slice::<u8>().expect("read output");
    let got = read_f32(out_bytes, total_out);

    // Output is f32 accumulation; atol is slightly relaxed due to bf16 input rounding.
    assert_allclose(&got, &cpu_ref, 5e-3, 2e-2, "moe_weighted_sum_seq_bf16_input");
}
