//! Integration tests for SDPA (scaled dot-product attention) kernels.
//!
//! These tests run the Metal GPU kernels and compare against CPU reference
//! implementations for correctness.

// `sdpa_sliding` is currently `#[deprecated]` pending repair-or-remove
// (see `src/ops/sdpa_sliding.rs` module docs and audit
// `cfa-20260425-fix-audit-findings`). These tests intentionally exercise
// it as the correctness baseline for the eventual `flash_attn_prefill`
// replacement, so the deprecation warning is silenced at file scope.
#![allow(deprecated)]

use mlx_native::ops::sdpa::{self, SdpaParams};
use mlx_native::ops::sdpa_sliding::{self, SdpaSlidingParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};

// --------------------------------------------------------------------------
// CPU reference implementations
// --------------------------------------------------------------------------

/// CPU reference SDPA: softmax(Q * K^T / sqrt(head_dim)) * V with causal mask.
///
/// All tensors are in layout [batch, heads, seq, head_dim] (contiguous f32).
/// GQA is handled by mapping Q head indices to KV head indices.
fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; batch_size * n_heads * seq_len * head_dim];

    for b in 0..batch_size {
        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            for q_pos in 0..seq_len {
                // Compute Q * K^T / sqrt(d) for all valid key positions.
                // In decode mode (seq_len < kv_seq_len), q_pos=0 maps to the
                // end of the KV sequence: abs_pos = kv_seq_len - seq_len + q_pos.
                let abs_pos = kv_seq_len - seq_len + q_pos;
                let causal_max_k = std::cmp::min(abs_pos + 1, kv_seq_len);
                let mut scores = Vec::with_capacity(causal_max_k);

                let q_offset =
                    b * (n_heads * seq_len * head_dim) + h * (seq_len * head_dim) + q_pos * head_dim;
                let k_head_base =
                    b * (n_kv_heads * kv_seq_len * head_dim) + kv_h * (kv_seq_len * head_dim);

                for k_pos in 0..causal_max_k {
                    let k_offset = k_head_base + k_pos * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k[k_offset + d];
                    }
                    scores.push(dot * scale);
                }

                // Numerically stable softmax.
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                // Weighted sum of V.
                let v_head_base = k_head_base; // V has same layout as K
                let o_offset = q_offset;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (k_pos, &w) in weights.iter().enumerate() {
                        acc += w * v[v_head_base + k_pos * head_dim + d];
                    }
                    output[o_offset + d] = acc;
                }
            }
        }
    }
    output
}

/// CPU reference sliding-window SDPA.
///
/// Same as `cpu_sdpa` but with a sliding window mask: for query position q_pos,
/// keys at k_pos < q_pos - window_size are masked.
fn cpu_sdpa_sliding(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    window_size: usize,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; batch_size * n_heads * seq_len * head_dim];

    for b in 0..batch_size {
        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            for q_pos in 0..seq_len {
                let abs_pos = kv_seq_len - seq_len + q_pos;
                let causal_max_k = std::cmp::min(abs_pos + 1, kv_seq_len);
                let window_start = if abs_pos >= window_size {
                    abs_pos - window_size
                } else {
                    0
                };

                let q_offset =
                    b * (n_heads * seq_len * head_dim) + h * (seq_len * head_dim) + q_pos * head_dim;
                let k_head_base =
                    b * (n_kv_heads * kv_seq_len * head_dim) + kv_h * (kv_seq_len * head_dim);

                let mut scores = Vec::new();
                let mut valid_positions = Vec::new();

                for k_pos in window_start..causal_max_k {
                    let k_offset = k_head_base + k_pos * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k[k_offset + d];
                    }
                    scores.push(dot * scale);
                    valid_positions.push(k_pos);
                }

                if scores.is_empty() {
                    // No valid keys: write zeros.
                    let o_offset = q_offset;
                    for d in 0..head_dim {
                        output[o_offset + d] = 0.0;
                    }
                    continue;
                }

                // Numerically stable softmax.
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                // Weighted sum of V.
                let v_head_base = k_head_base;
                let o_offset = q_offset;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (i, &k_pos) in valid_positions.iter().enumerate() {
                        acc += weights[i] * v[v_head_base + k_pos * head_dim + d];
                    }
                    output[o_offset + d] = acc;
                }
            }
        }
    }
    output
}

// --------------------------------------------------------------------------
// Helper: generate deterministic pseudo-random f32 data
// --------------------------------------------------------------------------

/// Simple LCG PRNG that generates reproducible f32 values in [-0.5, 0.5].
fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            // LCG: state = state * 6364136223846793005 + 1442695040888963407
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-0.5, 0.5] using top bits.
            let frac = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
            frac
        })
        .collect()
}

// --------------------------------------------------------------------------
// Helper: set up device and registry with SDPA shaders registered
// --------------------------------------------------------------------------

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new should succeed on Apple Silicon");
    let mut registry = KernelRegistry::new();
    sdpa::register(&mut registry);
    sdpa_sliding::register(&mut registry);
    (device, registry)
}

/// Compare GPU output against CPU reference within tolerance.
fn assert_close(actual: &[f32], expected: &[f32], tol: f32, test_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{test_name}: length mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    assert!(
        max_diff < tol,
        "{test_name}: max difference {max_diff} at index {max_diff_idx} exceeds tolerance {tol} \
         (actual={}, expected={})",
        actual[max_diff_idx],
        expected[max_diff_idx]
    );
}

/// Helper to run an SDPA test case.
fn run_sdpa_test(
    batch_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_seq_len: u32,
    test_name: &str,
) {
    let (device, mut registry) = setup();

    let q_elements = batch_size as usize * n_heads as usize * seq_len as usize * head_dim as usize;
    let kv_elements =
        batch_size as usize * n_kv_heads as usize * kv_seq_len as usize * head_dim as usize;

    // Generate test data.
    let q_data = pseudo_random_f32(42, q_elements);
    let k_data = pseudo_random_f32(137, kv_elements);
    let v_data = pseudo_random_f32(999, kv_elements);

    // CPU reference.
    let expected = cpu_sdpa(
        &q_data,
        &k_data,
        &v_data,
        batch_size as usize,
        n_heads as usize,
        n_kv_heads as usize,
        seq_len as usize,
        kv_seq_len as usize,
        head_dim as usize,
    );

    // Allocate GPU buffers.
    let q_bytes = q_elements * 4;
    let kv_bytes = kv_elements * 4;
    let out_bytes = q_elements * 4;

    let mut q_buf = device
        .alloc_buffer(q_bytes, DType::F32, vec![q_elements])
        .expect("alloc Q");
    let mut k_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elements])
        .expect("alloc K");
    let mut v_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elements])
        .expect("alloc V");
    let output_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elements])
        .expect("alloc output");

    // Fill GPU buffers.
    q_buf.as_mut_slice::<f32>().expect("q slice").copy_from_slice(&q_data);
    k_buf.as_mut_slice::<f32>().expect("k slice").copy_from_slice(&k_data);
    v_buf.as_mut_slice::<f32>().expect("v slice").copy_from_slice(&v_data);

    // Dispatch.
    let mut encoder = device.command_encoder().expect("encoder");
    let params = SdpaParams {
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        kv_seq_len,
        scale: 1.0 / (head_dim as f32).sqrt(),
        kv_capacity: kv_seq_len,
    };
    sdpa::sdpa(
        &mut encoder,
        &mut registry,
        &device,
        &q_buf,
        &k_buf,
        &v_buf,
        &output_buf,
        &params,
        batch_size,
    )
    .expect("sdpa dispatch");

    encoder.commit_and_wait().expect("commit_and_wait");

    // Read back and compare.
    let actual: Vec<f32> = output_buf.as_slice::<f32>().expect("read output").to_vec();
    assert_close(&actual, &expected, 1e-3, test_name);
}

/// Helper to run a sliding-window SDPA test case.
fn run_sdpa_sliding_test(
    batch_size: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_seq_len: u32,
    window_size: u32,
    test_name: &str,
) {
    let (device, mut registry) = setup();

    let q_elements = batch_size as usize * n_heads as usize * seq_len as usize * head_dim as usize;
    let kv_elements =
        batch_size as usize * n_kv_heads as usize * kv_seq_len as usize * head_dim as usize;

    let q_data = pseudo_random_f32(42, q_elements);
    let k_data = pseudo_random_f32(137, kv_elements);
    let v_data = pseudo_random_f32(999, kv_elements);

    let expected = cpu_sdpa_sliding(
        &q_data,
        &k_data,
        &v_data,
        batch_size as usize,
        n_heads as usize,
        n_kv_heads as usize,
        seq_len as usize,
        kv_seq_len as usize,
        head_dim as usize,
        window_size as usize,
    );

    let q_bytes = q_elements * 4;
    let kv_bytes = kv_elements * 4;
    let out_bytes = q_elements * 4;

    let mut q_buf = device
        .alloc_buffer(q_bytes, DType::F32, vec![q_elements])
        .expect("alloc Q");
    let mut k_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elements])
        .expect("alloc K");
    let mut v_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elements])
        .expect("alloc V");
    let output_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elements])
        .expect("alloc output");

    q_buf.as_mut_slice::<f32>().expect("q slice").copy_from_slice(&q_data);
    k_buf.as_mut_slice::<f32>().expect("k slice").copy_from_slice(&k_data);
    v_buf.as_mut_slice::<f32>().expect("v slice").copy_from_slice(&v_data);

    let mut encoder = device.command_encoder().expect("encoder");
    let params = SdpaSlidingParams {
        n_heads,
        n_kv_heads,
        head_dim,
        seq_len,
        kv_seq_len,
        window_size,
        scale: 1.0 / (head_dim as f32).sqrt(),
        kv_capacity: kv_seq_len,
    };
    sdpa_sliding::sdpa_sliding(
        &mut encoder,
        &mut registry,
        &device,
        &q_buf,
        &k_buf,
        &v_buf,
        &output_buf,
        &params,
        batch_size,
    )
    .expect("sdpa_sliding dispatch");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual: Vec<f32> = output_buf.as_slice::<f32>().expect("read output").to_vec();
    assert_close(&actual, &expected, 1e-3, test_name);
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

// AC-1: Standard multi-head attention (16/16 heads, head_dim=64, small for speed)
#[test]
fn test_sdpa_standard_mha() {
    run_sdpa_test(1, 4, 4, 64, 32, 32, "standard_mha");
}

// AC-2: GQA with 16 Q heads / 8 KV heads (Gemma 4 sliding config)
#[test]
fn test_sdpa_gqa_16_8() {
    run_sdpa_test(1, 16, 8, 64, 32, 32, "gqa_16_8");
}

// AC-3: GQA with 16 Q heads / 2 KV heads (Gemma 4 global config)
#[test]
fn test_sdpa_gqa_16_2() {
    run_sdpa_test(1, 16, 2, 64, 32, 32, "gqa_16_2");
}

// AC-5: Single token decode (seq_len=1, kv_seq_len=64)
#[test]
fn test_sdpa_single_token_decode() {
    run_sdpa_test(1, 4, 4, 64, 1, 64, "single_token_decode");
}

// AC-5: GQA single token decode
#[test]
fn test_sdpa_gqa_single_token_decode() {
    run_sdpa_test(1, 16, 8, 64, 1, 128, "gqa_single_token_decode");
}

// AC-5: Various sequence lengths
#[test]
fn test_sdpa_seq_len_128() {
    run_sdpa_test(1, 4, 4, 32, 128, 128, "seq_len_128");
}

// AC-4: Sliding window attention
#[test]
fn test_sdpa_sliding_window() {
    // Use smaller dimensions for test speed but still exercise the sliding logic.
    // seq_len=64, window_size=16 so many positions are masked.
    run_sdpa_sliding_test(1, 4, 4, 32, 64, 64, 16, "sliding_window");
}

// AC-4: Sliding window with GQA (16/8 heads)
#[test]
fn test_sdpa_sliding_gqa_16_8() {
    run_sdpa_sliding_test(1, 16, 8, 32, 64, 64, 16, "sliding_gqa_16_8");
}

// T7.7: Sliding window where seq_len < window_size (window has no effect,
// should match full causal attention).
#[test]
fn test_sdpa_sliding_short_seq() {
    // With window_size=1024 and seq_len=32, the window never clips.
    // Result should match full SDPA.
    let batch_size = 1u32;
    let n_heads = 4u32;
    let n_kv_heads = 4u32;
    let head_dim = 32u32;
    let seq_len = 32u32;
    let kv_seq_len = 32u32;
    let window_size = 1024u32;

    let q_elements = (batch_size * n_heads * seq_len * head_dim) as usize;
    let kv_elements = (batch_size * n_kv_heads * kv_seq_len * head_dim) as usize;
    let q_data = pseudo_random_f32(42, q_elements);
    let k_data = pseudo_random_f32(137, kv_elements);
    let v_data = pseudo_random_f32(999, kv_elements);

    // CPU full SDPA (no window).
    let expected_full = cpu_sdpa(
        &q_data, &k_data, &v_data,
        batch_size as usize, n_heads as usize, n_kv_heads as usize,
        seq_len as usize, kv_seq_len as usize, head_dim as usize,
    );

    // CPU sliding window SDPA (window larger than seq).
    let expected_sliding = cpu_sdpa_sliding(
        &q_data, &k_data, &v_data,
        batch_size as usize, n_heads as usize, n_kv_heads as usize,
        seq_len as usize, kv_seq_len as usize, head_dim as usize,
        window_size as usize,
    );

    // Verify CPU references match (the window doesn't clip anything).
    assert_close(&expected_sliding, &expected_full, 1e-6, "cpu_sliding_vs_full");

    // Now run on GPU and verify.
    run_sdpa_sliding_test(
        batch_size, n_heads, n_kv_heads, head_dim,
        seq_len, kv_seq_len, window_size,
        "sliding_short_seq",
    );
}

// T7.8: Sliding window single decode (seq_len=1, kv_seq_len=64, window=16)
#[test]
fn test_sdpa_sliding_single_decode() {
    run_sdpa_sliding_test(1, 4, 4, 32, 1, 64, 16, "sliding_single_decode");
}

// T7.9: Invalid head ratio returns error.
#[test]
fn test_invalid_head_ratio() {
    let (device, mut registry) = setup();

    let params = SdpaParams {
        n_heads: 16,
        n_kv_heads: 7,
        head_dim: 64,
        seq_len: 32,
        kv_seq_len: 32,
        scale: 1.0 / (64.0_f32).sqrt(),
        kv_capacity: 32,
    };

    // Allocate minimal buffers (they won't actually be used).
    let buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("buf");
    let out = device.alloc_buffer(4, DType::F32, vec![1]).expect("out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = sdpa::sdpa(
        &mut encoder, &mut registry, &device,
        &buf, &buf, &buf, &out, &params, 1,
    );

    assert!(result.is_err());
    match result {
        Err(mlx_native::MlxError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("divisible"),
                "Error message should mention divisibility: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got: {:?}", other),
    }
}

// T7.10: Zero head_dim returns error.
#[test]
fn test_zero_head_dim() {
    let (device, mut registry) = setup();

    let params = SdpaParams {
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 0,
        seq_len: 32,
        kv_seq_len: 32,
        scale: 1.0,
        kv_capacity: 32,
    };

    let buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("buf");
    let out = device.alloc_buffer(4, DType::F32, vec![1]).expect("out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = sdpa::sdpa(
        &mut encoder, &mut registry, &device,
        &buf, &buf, &buf, &out, &params, 1,
    );

    assert!(result.is_err());
    match result {
        Err(mlx_native::MlxError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("head_dim"),
                "Error message should mention head_dim: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got: {:?}", other),
    }
}

// Test sliding window with invalid (zero) window_size.
#[test]
fn test_sliding_zero_window_size() {
    let (device, mut registry) = setup();

    let params = SdpaSlidingParams {
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 64,
        seq_len: 32,
        kv_seq_len: 32,
        window_size: 0,
        scale: 1.0 / (64.0_f32).sqrt(),
        kv_capacity: 32,
    };

    let buf = device.alloc_buffer(4, DType::F32, vec![1]).expect("buf");
    let out = device.alloc_buffer(4, DType::F32, vec![1]).expect("out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = sdpa_sliding::sdpa_sliding(
        &mut encoder, &mut registry, &device,
        &buf, &buf, &buf, &out, &params, 1,
    );

    assert!(result.is_err());
    match result {
        Err(mlx_native::MlxError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("window_size"),
                "Error message should mention window_size: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got: {:?}", other),
    }
}

// Test with batch_size > 1.
#[test]
fn test_sdpa_batch_2() {
    run_sdpa_test(2, 4, 4, 32, 16, 16, "batch_2");
}

// Test sliding with batch_size > 1.
#[test]
fn test_sdpa_sliding_batch_2() {
    run_sdpa_sliding_test(2, 4, 4, 32, 32, 32, 8, "sliding_batch_2");
}

// Test with larger head_dim (128) matching real-world configs.
#[test]
fn test_sdpa_head_dim_128() {
    run_sdpa_test(1, 4, 2, 128, 16, 16, "head_dim_128");
}

// Shader compilation smoke test: ensure both shaders compile and produce valid pipelines.
#[test]
fn test_shader_compilation() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    sdpa::register(&mut registry);
    sdpa_sliding::register(&mut registry);

    let _p1 = registry
        .get_pipeline("sdpa", device.metal_device())
        .expect("sdpa shader should compile");
    let _p2 = registry
        .get_pipeline("sdpa_sliding", device.metal_device())
        .expect("sdpa_sliding shader should compile");
}
