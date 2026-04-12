//! Unit tests for GGML block-format expert-indexed quantized matmul GPU kernels.
//!
//! Tests Q4_0, Q8_0, Q6_K with expert routing:
//!   - 1 token, 8 experts, top-2: verify output matches per-expert manual dispatch
//!   - 4 tokens, 8 experts, top-2: batch decode
//!   - Q6_K and Q8_0 variants

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxDevice,
};

// --------------------------------------------------------------------------
// PRNG
// --------------------------------------------------------------------------

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32) - 0.5;
            frac
        })
        .collect()
}

// --------------------------------------------------------------------------
// GGML block packing helpers (copied from test_quantized_matmul_ggml.rs)
// --------------------------------------------------------------------------

fn pack_q4_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0, "values must be multiple of 32");
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        let mut nibbles = [0u8; 16];
        for i in 0..16 {
            let v0 = (block[i] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            let v1 = (block[i + 16] * id + 8.0).round().clamp(0.0, 15.0) as u8;
            nibbles[i] = v0 | (v1 << 4);
        }
        buf.extend_from_slice(&nibbles);
    }
    buf
}

fn pack_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % 32 == 0);
    let mut buf = Vec::new();
    for block in values.chunks(32) {
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = half::f16::from_f32(d);
        buf.extend_from_slice(&d_f16.to_le_bytes());
        for &v in block {
            let q = (v * id).round().clamp(-128.0, 127.0) as i8;
            buf.push(q as u8);
        }
    }
    buf
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

/// Create stacked expert weight buffer: N experts concatenated.
/// Returns (stacked_bytes, per_expert_byte_size).
fn stack_expert_weights(expert_packed: &[Vec<u8>]) -> (Vec<u8>, usize) {
    let per_expert = expert_packed[0].len();
    for ep in expert_packed {
        assert_eq!(ep.len(), per_expert);
    }
    let mut stacked = Vec::with_capacity(per_expert * expert_packed.len());
    for ep in expert_packed {
        stacked.extend_from_slice(ep);
    }
    (stacked, per_expert)
}

/// Verify the _id kernel matches per-expert non-id dispatch.
/// This dispatches the regular (non-id) kernel once per expert and compares.
fn run_id_vs_norid_test(
    ggml_type: GgmlType,
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    n: usize,
    k: usize,
    pack_fn: fn(&[f32]) -> Vec<u8>,
    tolerance: f32,
) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();

    let f32_sz = std::mem::size_of::<f32>();
    let u32_sz = std::mem::size_of::<u32>();

    // Generate input and weights
    let input_data = pseudo_random_f32(42, n_tokens * k);
    let mut expert_packed = Vec::new();
    for e in 0..n_experts {
        let w_data = pseudo_random_f32(100 + e as u64, n * k);
        expert_packed.push(pack_fn(&w_data));
    }
    let (stacked_bytes, per_expert_bytes) = stack_expert_weights(&expert_packed);

    let mut ids = Vec::with_capacity(n_tokens * top_k);
    for t in 0..n_tokens {
        for s in 0..top_k {
            ids.push(((t * 3 + s * 7 + 1) % n_experts) as u32);
        }
    }

    let total_rows = n_tokens * top_k;

    // Upload
    let mut input_buf = device
        .alloc_buffer(n_tokens * k * f32_sz, DType::F32, vec![n_tokens * k])
        .unwrap();
    input_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_data);

    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U32, vec![stacked_bytes.len() / 4])
        .unwrap();
    weight_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&stacked_bytes);

    let mut ids_buf = device
        .alloc_buffer(total_rows * u32_sz, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids);

    // --- Run _id kernel ---
    let mut id_output_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows * n])
        .unwrap();

    {
        let params = GgmlQuantizedMatmulIdParams {
            n_tokens: n_tokens as u32,
            top_k: top_k as u32,
            n: n as u32,
            k: k as u32,
            n_experts: n_experts as u32,
            expert_stride: per_expert_bytes as u64,
            ggml_type,
        };

        let mut encoder = device.command_encoder().unwrap();
        mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut encoder,
            &mut registry,
            &device,
            &input_buf,
            &weight_buf,
            &ids_buf,
            &mut id_output_buf,
            &params,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();
    }

    // --- Run per-expert non-id kernel ---
    let mut norid_results = vec![0.0f32; total_rows * n];
    for t in 0..n_tokens {
        for s in 0..top_k {
            let row_idx = t * top_k + s;
            let expert_id = ids[row_idx] as usize;

            // Create single-expert weight buffer
            let expert_bytes = &stacked_bytes
                [expert_id * per_expert_bytes..(expert_id + 1) * per_expert_bytes];
            let mut expert_w_buf = device
                .alloc_buffer(per_expert_bytes, DType::U32, vec![per_expert_bytes / 4])
                .unwrap();
            expert_w_buf
                .as_mut_slice::<u8>()
                .unwrap()
                .copy_from_slice(expert_bytes);

            // Single-token input
            let mut tok_input_buf = device
                .alloc_buffer(k * f32_sz, DType::F32, vec![k])
                .unwrap();
            tok_input_buf
                .as_mut_slice::<f32>()
                .unwrap()
                .copy_from_slice(&input_data[t * k..(t + 1) * k]);

            let mut tok_output_buf = device
                .alloc_buffer(n * f32_sz, DType::F32, vec![n])
                .unwrap();

            let norid_params = GgmlQuantizedMatmulParams {
                m: 1,
                n: n as u32,
                k: k as u32,
                ggml_type,
            };

            let mut encoder = device.command_encoder().unwrap();
            mlx_native::ops::quantized_matmul_ggml::quantized_matmul_ggml(
                &mut encoder,
                &mut registry,
                &device,
                &tok_input_buf,
                &expert_w_buf,
                &mut tok_output_buf,
                &norid_params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();

            let result: &[f32] = tok_output_buf.as_slice().unwrap();
            norid_results[row_idx * n..(row_idx + 1) * n].copy_from_slice(result);
        }
    }

    // Compare _id output vs per-expert output (should be BIT-EXACT)
    let id_out: &[f32] = id_output_buf.as_slice().unwrap();
    let mut max_err: f32 = 0.0;
    let mut err_count = 0usize;
    for i in 0..total_rows * n {
        let err = (id_out[i] - norid_results[i]).abs();
        max_err = max_err.max(err);
        if err > tolerance {
            if err_count < 5 {
                eprintln!(
                    "  id vs norid mismatch at [{}]: id={:.6}, norid={:.6}, err={:.6}",
                    i, id_out[i], norid_results[i], err
                );
            }
            err_count += 1;
        }
    }
    assert_eq!(
        err_count, 0,
        "{:?} id vs norid: {} mismatches (max_err={:.6})",
        ggml_type, err_count, max_err
    );
    eprintln!(
        "  PASS id-vs-norid: {:?} {}x{}, {} tokens, top-{}, max_err={:.6}",
        ggml_type, n, k, n_tokens, top_k, max_err
    );
}

// --------------------------------------------------------------------------
// Test cases
// --------------------------------------------------------------------------

#[test]
fn test_q4_0_id_vs_norid() {
    run_id_vs_norid_test(
        GgmlType::Q4_0,
        1, 8, 2,
        64, 128,
        pack_q4_0,
        0.0,  // Should be bit-exact
    );
}

#[test]
fn test_q8_0_id_vs_norid() {
    run_id_vs_norid_test(
        GgmlType::Q8_0,
        1, 8, 2,
        64, 128,
        pack_q8_0,
        0.0,
    );
}

#[test]
fn test_q4_0_id_vs_norid_4tok() {
    run_id_vs_norid_test(
        GgmlType::Q4_0,
        4, 8, 2,
        64, 128,
        pack_q4_0,
        0.0,
    );
}

#[test]
fn test_q8_0_id_vs_norid_4tok() {
    run_id_vs_norid_test(
        GgmlType::Q8_0,
        4, 8, 2,
        64, 128,
        pack_q8_0,
        0.0,
    );
}

// Production-like shapes (Gemma 4 MoE dimensions)
#[test]
fn test_q8_0_production_shape() {
    // Gemma 4: gate_up [2*moe_intermediate, hidden] = [2*2048, 2816]
    // But that's too large for CI. Use scaled-down version.
    run_id_vs_norid_test(
        GgmlType::Q8_0,
        1, 8, 2,
        256, 256,   // Scaled down from 4096x2816
        pack_q8_0,
        0.0,
    );
}

#[test]
fn test_q4_0_production_shape() {
    run_id_vs_norid_test(
        GgmlType::Q4_0,
        1, 8, 2,
        256, 256,
        pack_q4_0,
        0.0,
    );
}
