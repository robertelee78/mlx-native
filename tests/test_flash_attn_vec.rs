//! Integration tests for flash_attn_vec (SIMD-vectorized decode-path SDPA).
//!
//! Tests the Metal GPU kernel against a CPU reference implementation.

use mlx_native::ops::flash_attn_vec::{self, FlashAttnVecParams};
use mlx_native::{DType, KernelRegistry, MlxDevice};

// --------------------------------------------------------------------------
// CPU reference: softmax(Q * K^T * scale) * V with causal + sliding window
// --------------------------------------------------------------------------

/// CPU reference SDPA for decode mode (single query).
///
/// Q: [num_heads, 1, head_dim]  (contiguous)
/// K: [num_kv_heads, kv_capacity, head_dim]  (only first kv_seq_len valid)
/// V: [num_kv_heads, kv_capacity, head_dim]
/// Output: [num_heads, 1, head_dim]
fn cpu_flash_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
    kv_capacity: usize,
    scale: f32,
    mask_type: u32,
    sliding_window: usize,
) -> Vec<f32> {
    let heads_per_kv = num_heads / num_kv_heads;
    let mut output = vec![0.0f32; num_heads * head_dim];

    // Decode: query is at the last position.
    let abs_pos = kv_seq_len - 1;

    let window_start = if mask_type == 2 && sliding_window > 0 {
        abs_pos.saturating_sub(sliding_window - 1)
    } else {
        0
    };
    let causal_max_k = std::cmp::min(abs_pos + 1, kv_seq_len);

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;

        let q_offset = h * head_dim;
        let k_head_base = kv_h * kv_capacity * head_dim;

        let mut scores = Vec::new();
        for k_pos in window_start..causal_max_k {
            let k_offset = k_head_base + k_pos * head_dim;
            let mut dot = 0.0f64; // use f64 for reference accuracy
            for d in 0..head_dim {
                dot += q[q_offset + d] as f64 * k[k_offset + d] as f64;
            }
            scores.push(dot as f32 * scale);
        }

        if scores.is_empty() {
            continue;
        }

        // Softmax.
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of V.
        let v_head_base = kv_h * kv_capacity * head_dim;
        let o_offset = h * head_dim;
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (i, &w) in weights.iter().enumerate() {
                let k_pos = window_start + i;
                acc += w * v[v_head_base + k_pos * head_dim + d];
            }
            output[o_offset + d] = acc;
        }
    }

    output
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/// Simple deterministic pseudo-random f32 in [-1, 1] from an index.
fn pseudo_random(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((x >> 33) as u32) & 0x7FFFFF;
    (bits as f32 / 0x7FFFFF as f32) * 2.0 - 1.0
}

fn fill_random(buf: &mut [f32], seed: u64) {
    for (i, val) in buf.iter_mut().enumerate() {
        *val = pseudo_random(seed + i as u64);
    }
}

/// Run a flash_attn_vec test case and compare to CPU reference.
fn run_test(
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    seed: u64,
    label: &str,
    epsilon: f32,
) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let q_elems = num_heads as usize * head_dim as usize;
    let kv_elems = num_kv_heads as usize * kv_capacity as usize * head_dim as usize;

    let mut q_data = vec![0.0f32; q_elems];
    let mut k_data = vec![0.0f32; kv_elems];
    let mut v_data = vec![0.0f32; kv_elems];
    fill_random(&mut q_data, seed);
    fill_random(&mut k_data, seed + 10000);
    fill_random(&mut v_data, seed + 20000);

    // CPU reference.
    let expected = cpu_flash_attn(
        &q_data,
        &k_data,
        &v_data,
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
        kv_seq_len as usize,
        kv_capacity as usize,
        scale,
        mask_type,
        sliding_window as usize,
    );

    // GPU buffers.
    let q_bytes = q_elems * 4;
    let kv_bytes = kv_elems * 4;
    let out_bytes = q_elems * 4;

    let mut q_buf = device
        .alloc_buffer(q_bytes, DType::F32, vec![q_elems])
        .expect("alloc Q");
    let mut k_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elems])
        .expect("alloc K");
    let mut v_buf = device
        .alloc_buffer(kv_bytes, DType::F32, vec![kv_elems])
        .expect("alloc V");
    let output_buf = device
        .alloc_buffer(out_bytes, DType::F32, vec![q_elems])
        .expect("alloc output");

    q_buf
        .as_mut_slice::<f32>()
        .expect("q slice")
        .copy_from_slice(&q_data);
    k_buf
        .as_mut_slice::<f32>()
        .expect("k slice")
        .copy_from_slice(&k_data);
    v_buf
        .as_mut_slice::<f32>()
        .expect("v slice")
        .copy_from_slice(&v_data);

    let tmp_bytes = flash_attn_vec::tmp_buffer_bytes(num_heads, head_dim);
    let tmp_elems = tmp_bytes / 4;
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_elems])
        .expect("alloc tmp");

    let params = FlashAttnVecParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type,
        sliding_window,
        softcap: 0.0,
    };

    // Dispatch.
    let mut encoder = device.command_encoder().expect("encoder");
    flash_attn_vec::flash_attn_vec(
        &mut encoder,
        &mut registry,
        &device,
        &q_buf,
        &k_buf,
        &v_buf,
        &output_buf,
        &tmp_buf,
        &params,
    )
    .expect("flash_attn_vec dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    // Compare.
    let gpu_output: &[f32] = output_buf.as_slice::<f32>().expect("output slice");
    let mut max_diff = 0.0f32;
    for (i, (&gpu, &cpu)) in gpu_output.iter().zip(expected.iter()).enumerate() {
        let diff = (gpu - cpu).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > epsilon {
            panic!(
                "{label}: mismatch at index {i}: gpu={gpu:.6}, cpu={cpu:.6}, diff={diff:.6e}"
            );
        }
    }
    eprintln!("{label}: max_diff = {max_diff:.6e}");
}

// --------------------------------------------------------------------------
// dk256 tests
// --------------------------------------------------------------------------

#[test]
fn test_flash_attn_vec_dk256_basic() {
    run_test(
        4,   // num_heads
        4,   // num_kv_heads
        256, // head_dim
        32,  // kv_seq_len
        64,  // kv_capacity
        1.0 / (256.0_f32).sqrt(), // scale
        1,   // mask_type (causal)
        0,   // sliding_window
        42,  // seed
        "dk256 basic",
        1e-2,
    );
}

#[test]
fn test_flash_attn_vec_dk256_gqa() {
    run_test(
        16,  // num_heads
        8,   // num_kv_heads
        256, // head_dim
        48,  // kv_seq_len
        64,  // kv_capacity
        1.0, // scale
        1,   // mask_type
        0,   // sliding_window
        100, // seed
        "dk256 GQA",
        1e-2,
    );
}

#[test]
fn test_flash_attn_vec_dk256_sliding_window() {
    run_test(
        4,   // num_heads
        4,   // num_kv_heads
        256, // head_dim
        64,  // kv_seq_len
        128, // kv_capacity
        1.0, // scale
        2,   // mask_type (sliding_window)
        16,  // sliding_window
        777, // seed
        "dk256 sliding window",
        1e-2,
    );
}

// --------------------------------------------------------------------------
// dk512 tests
// --------------------------------------------------------------------------

#[test]
fn test_flash_attn_vec_dk512_basic() {
    run_test(
        4,   // num_heads
        4,   // num_kv_heads
        512, // head_dim
        32,  // kv_seq_len
        64,  // kv_capacity
        1.0 / (512.0_f32).sqrt(), // scale
        1,   // mask_type
        0,   // sliding_window
        1111, // seed
        "dk512 basic",
        1e-2,
    );
}

#[test]
fn test_flash_attn_vec_dk512_gqa() {
    run_test(
        16,  // num_heads
        8,   // num_kv_heads
        512, // head_dim
        48,  // kv_seq_len
        64,  // kv_capacity
        1.0, // scale
        1,   // mask_type
        0,   // sliding_window
        2222, // seed
        "dk512 GQA",
        1e-2,
    );
}
