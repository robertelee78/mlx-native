//! Parity test for `quantized_matmul_id_swiglu_q4_0`.
//!
//! The fused kernel computes
//!     output[r][n] = sum_k(dequant(W_q4_0[ids[r]][n][k])
//!                          * (silu(gate[r][k]) * up[r][k]))
//!
//! in a single dispatch.  This test validates it bit-equivalent (within
//! float-precision tolerance) to the un-fused
//! `silu_mul_f32 → quantized_matmul_id_ggml` two-dispatch sequence.
//!
//! Closes ADR-012 §Optimize / Task #15: the fused variant saves one
//! dispatch + one memory_barrier per MoE layer in dwq46 decode (40
//! layers × ~5-10µs ≈ 0.3-0.4ms saved per token).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::{
    DType, GgmlQuantizedMatmulIdParams, GgmlType, KernelRegistry, MlxDevice,
};

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
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

/// CPU silu_mul reference.
fn silu_mul_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

#[test]
fn test_swiglu_q4_0_vs_unfused_reference() {
    // Decode-shape: 1 token, top_k=8, intermediate dim, hidden dim.
    // Use small but realistic shapes so the test is fast and deterministic.
    let n_tokens: usize = 1;
    let top_k: usize = 8;
    let n_experts: usize = 8;
    let k: usize = 256; // intermediate dim (must be multiple of 32 for Q4_0)
    let n: usize = 128; // hidden dim
    let total_rows = n_tokens * top_k;

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let f32_sz = std::mem::size_of::<f32>();

    // Random gate + up vectors per output row (one gate/up vector per
    // (token, expert_slot) pair, since they are pre-routed in MoE FFN).
    let gate_data = pseudo_random_f32(42, total_rows * k);
    let up_data = pseudo_random_f32(43, total_rows * k);

    // Random Q4_0 weights per expert.
    let mut expert_packed: Vec<Vec<u8>> = Vec::new();
    for e in 0..n_experts {
        let w_data = pseudo_random_f32(100 + e as u64, n * k);
        expert_packed.push(pack_q4_0(&w_data));
    }
    let per_expert_bytes = expert_packed[0].len();
    let mut stacked_bytes = Vec::with_capacity(per_expert_bytes * n_experts);
    for ep in &expert_packed {
        stacked_bytes.extend_from_slice(ep);
    }

    // Deterministic ids: each output row routes to a different expert.
    let mut ids: Vec<u32> = Vec::with_capacity(total_rows);
    for r in 0..total_rows {
        ids.push((r as u32) % (n_experts as u32));
    }

    // Upload buffers.
    let mut gate_buf = device
        .alloc_buffer(total_rows * k * f32_sz, DType::F32, vec![total_rows, k])
        .unwrap();
    gate_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&gate_data);

    let mut up_buf = device
        .alloc_buffer(total_rows * k * f32_sz, DType::F32, vec![total_rows, k])
        .unwrap();
    up_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&up_data);

    let mut weight_buf = device
        .alloc_buffer(stacked_bytes.len(), DType::U32, vec![stacked_bytes.len() / 4])
        .unwrap();
    weight_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&stacked_bytes);

    let mut ids_buf = device
        .alloc_buffer(total_rows * 4, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&ids);

    // Production MoE FFN expert_down trick (gpu_ffn.rs:1212-1217):
    //   n_tokens = total_rows (n_real_tokens × top_k)
    //   top_k    = 1
    // This makes each output row read its own input row from a flat
    // [total_rows, K] buffer (gate_all/up_all/h_all are pre-routed
    // per (real_token, expert_slot) by the upstream gate_all/up_all
    // mv_id calls).  We mirror that calling convention here so the
    // unfused reference path's `token_idx = output_row / top_k` ends
    // up reading the same per-row vector that the fused kernel does.
    let params = GgmlQuantizedMatmulIdParams {
        n_tokens: total_rows as u32,
        top_k: 1,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::Q4_0,
    };

    // ---- Unfused reference: silu_mul on CPU + quantized_matmul_id_ggml ----
    let h_all = silu_mul_cpu(&gate_data, &up_data);
    let mut h_buf = device
        .alloc_buffer(total_rows * k * f32_sz, DType::F32, vec![total_rows, k])
        .unwrap();
    h_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&h_all);

    let mut ref_out_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows, n])
        .unwrap();
    {
        let mut enc = device.command_encoder().unwrap();
        mlx_native::ops::quantized_matmul_id_ggml::quantized_matmul_id_ggml(
            &mut enc, &mut registry, &device,
            &h_buf, &weight_buf, &ids_buf, &mut ref_out_buf, &params,
        ).unwrap();
        enc.commit_and_wait().unwrap();
    }

    // ---- Fused swiglu_q4_0 ----
    let mut fused_out_buf = device
        .alloc_buffer(total_rows * n * f32_sz, DType::F32, vec![total_rows, n])
        .unwrap();
    {
        let mut enc = device.command_encoder().unwrap();
        mlx_native::quantized_matmul_id_swiglu_q4_0(
            &mut enc, &mut registry, &device,
            &gate_buf, &up_buf, &weight_buf, &ids_buf, &mut fused_out_buf, &params,
        ).unwrap();
        enc.commit_and_wait().unwrap();
    }

    // ---- Compare ----
    let ref_out: &[f32] = ref_out_buf.as_slice().unwrap();
    let fused_out: &[f32] = fused_out_buf.as_slice().unwrap();
    let mut max_err: f32 = 0.0;
    let mut err_count = 0usize;
    // Tolerance: the fused kernel uses Metal's metal::exp (vs CPU's f32::exp).
    // Both are approximate but the difference is bounded by ~1ulp per silu eval
    // accumulated across k=256 elements → expect <1e-4 max abs error.
    let tolerance = 5e-3;
    for i in 0..total_rows * n {
        let err = (ref_out[i] - fused_out[i]).abs();
        max_err = max_err.max(err);
        if err > tolerance {
            if err_count < 5 {
                eprintln!(
                    "  swiglu mismatch at [{}]: ref={:.6}, fused={:.6}, err={:.6}",
                    i, ref_out[i], fused_out[i], err,
                );
            }
            err_count += 1;
        }
    }
    eprintln!(
        "  PASS swiglu Q4_0: shape={}x{} (rows×N), max_err={:.6e} (tol={})",
        total_rows, n, max_err, tolerance
    );
    assert_eq!(
        err_count, 0,
        "swiglu fused vs unfused reference: {} mismatches (max_err={:.6e})",
        err_count, max_err
    );
}
