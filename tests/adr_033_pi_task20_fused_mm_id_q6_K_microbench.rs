//! ADR-033 §Pi Task #20 iter 7 — synthetic microbench for the fused MoE
//! gate+up+silu_mul mm_id Q6_K kernel at Qwen3.6 35B-A3B prefill shapes.
//!
//! Compares the fused single-dispatch path (1 map0 + 1 mm_id) against
//! the unfused 3-dispatch reference (2 map0 + 2 mm_id + 1 silu_mul).
//! Reports total GPU wall time + tokens/sec equivalent across n_tokens
//! ∈ {64, 256, 1024} at production Qwen3.6 MoE shape.
//!
//! Not run by default (gated behind HF2Q_RUN_FUSED_BENCH=1) — bench
//! cycles are expensive and the parity test
//! (adr_033_pi_task20_fused_mm_id_q6_K_parity) is the gating
//! correctness check.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, non_snake_case)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_for_test, dispatch_id_mm_fused_gate_up_silu_for_test, GgmlIdMmDispatchParams,
};
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{DType, GgmlType, KernelRegistry, MlxDevice};
use std::time::Instant;

const QK_K: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;

fn xs64(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 13;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545F4914F6CDD1D)
}

fn random_pm1(state: &mut u64) -> f32 {
    let bits = xs64(state);
    ((bits >> 11) as f32) / (1u64 << 53) as f32 * 2.0 - 1.0
}

fn pack_q6_K(values: &[f32]) -> Vec<u8> {
    assert!(values.len() % QK_K == 0);
    let mut bytes = Vec::with_capacity(values.len() / QK_K * Q6_K_BLOCK_BYTES);
    for block in values.chunks(QK_K) {
        let mut sub_scales = [0.0f32; 16];
        let mut sub_scale_int = [0i8; 16];
        let mut max_scale: f32 = 0.0;

        for (s, sub) in block.chunks(16).enumerate() {
            let amax = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            sub_scales[s] = amax;
            if amax > max_scale {
                max_scale = amax;
            }
        }

        let d = max_scale / (32.0 * 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        for s in 0..16 {
            sub_scale_int[s] = if sub_scales[s] != 0.0 {
                (sub_scales[s] * id / 32.0).round().clamp(-128.0, 127.0) as i8
            } else {
                0
            };
        }

        let mut q6 = [0u8; 256];
        for (s, sub) in block.chunks(16).enumerate() {
            let sc = sub_scale_int[s] as f32;
            let sub_d = d * sc;
            let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * sub_id + 32.0).round().clamp(0.0, 63.0) as u8;
                q6[s * 16 + i] = q;
            }
        }

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for ip in 0..2usize {
            let base = ip * 128;
            for l in 0..32usize {
                let v0 = q6[base + l];
                let v32 = q6[base + l + 32];
                let v64 = q6[base + l + 64];
                let v96 = q6[base + l + 96];
                ql[ip * 64 + l] = (v0 & 0xF) | ((v64 & 0xF) << 4);
                ql[ip * 64 + l + 32] = (v32 & 0xF) | ((v96 & 0xF) << 4);
                qh[ip * 32 + l] = (v0 >> 4)
                    | ((v32 >> 4) << 2)
                    | ((v64 >> 4) << 4)
                    | ((v96 >> 4) << 6);
            }
        }

        bytes.extend_from_slice(&ql);
        bytes.extend_from_slice(&qh);
        for s in sub_scale_int.iter() {
            bytes.push(*s as u8);
        }
        bytes.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    }
    bytes
}

fn bench_at_shape(
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    n: usize, // intermediate_size
    k: usize, // hidden_size
    reps: usize,
    warmup: usize,
    seed: u64,
) {
    assert_eq!(k % QK_K, 0);
    let blocks_per_row = k / QK_K;
    let per_expert_bytes = n * blocks_per_row * Q6_K_BLOCK_BYTES;

    let mut state = seed;

    // Synthesize gate + up weights for all experts. This is the cold
    // setup cost — bench timer starts AFTER allocation.
    eprintln!(
        "[bench] generating Q6_K weight slabs ({:.2} MB total)...",
        (2 * n_experts * per_expert_bytes) as f64 / (1 << 20) as f64
    );
    let mut gate_bytes = Vec::with_capacity(n_experts * per_expert_bytes);
    let mut up_bytes = Vec::with_capacity(n_experts * per_expert_bytes);
    for _expert in 0..n_experts {
        for _row in 0..n {
            let mut row_g = vec![0.0_f32; k];
            let mut row_u = vec![0.0_f32; k];
            for v in row_g.iter_mut() {
                *v = random_pm1(&mut state) * 0.5;
            }
            for v in row_u.iter_mut() {
                *v = random_pm1(&mut state) * 0.5;
            }
            gate_bytes.extend(pack_q6_K(&row_g));
            up_bytes.extend(pack_q6_K(&row_u));
        }
    }

    let mut input_data = vec![0.0_f32; n_tokens * k];
    for v in input_data.iter_mut() {
        *v = random_pm1(&mut state);
    }

    let total_rows = n_tokens * top_k;
    let mut ids = vec![0_u32; total_rows];
    {
        let mut pool = vec![0_u32; n_experts];
        for t in 0..n_tokens {
            for j in 0..n_experts {
                pool[j] = j as u32;
            }
            for j in 0..top_k.min(n_experts) {
                let r = (xs64(&mut state) as usize) % (n_experts - j);
                let pick = pool[j + r];
                pool[j + r] = pool[j];
                pool[j] = pick;
                ids[t * top_k + j] = pick;
            }
        }
    }

    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();

    let mut gate_buf = device
        .alloc_buffer(gate_bytes.len(), DType::U8, vec![gate_bytes.len()])
        .unwrap();
    gate_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&gate_bytes);
    let mut up_buf = device
        .alloc_buffer(up_bytes.len(), DType::U8, vec![up_bytes.len()])
        .unwrap();
    up_buf
        .as_mut_slice::<u8>()
        .unwrap()
        .copy_from_slice(&up_bytes);

    let mut input_buf = device
        .alloc_buffer(input_data.len() * 4, DType::F32, vec![input_data.len()])
        .unwrap();
    input_buf
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&input_data);

    let mut ids_buf = device
        .alloc_buffer(total_rows * 4, DType::U32, vec![total_rows])
        .unwrap();
    ids_buf
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(&ids);

    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: n_tokens as u32,
        top_k: top_k as u32,
        n: n as u32,
        k: k as u32,
        n_experts: n_experts as u32,
        expert_stride: per_expert_bytes as u64,
        ggml_type: GgmlType::Q6_K,
    };

    let mut htpe_buf = device
        .alloc_buffer(dispatch.htpe_bytes(), DType::U32, vec![n_experts])
        .unwrap();
    let mut hids_buf = device
        .alloc_buffer(dispatch.hids_bytes(), DType::U32, vec![n_experts, n_tokens])
        .unwrap();
    let mut tmp_gate_buf = device
        .alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n])
        .unwrap();
    let mut tmp_up_buf = device
        .alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n])
        .unwrap();
    let mut h_all_buf = device
        .alloc_buffer(total_rows * n * 4, DType::F32, vec![total_rows * n])
        .unwrap();
    let mut silu_params_buf = device
        .alloc_buffer(4, DType::U32, vec![1])
        .unwrap();
    silu_params_buf
        .as_mut_slice::<u32>()
        .unwrap()[0] = (total_rows * n) as u32;

    fn zero_htpe(buf: &mut mlx_native::MlxBuffer) {
        for v in buf.as_mut_slice::<u32>().unwrap().iter_mut() {
            *v = 0;
        }
    }

    // Inline both benches to avoid closure borrow conflicts.
    let n_outputs = (total_rows * n) as u32;
    let do_unfused = |device: &MlxDevice,
                      registry: &mut KernelRegistry,
                      input_buf: &mlx_native::MlxBuffer,
                      gate_buf: &mlx_native::MlxBuffer,
                      up_buf: &mlx_native::MlxBuffer,
                      ids_buf: &mlx_native::MlxBuffer,
                      htpe_buf: &mut mlx_native::MlxBuffer,
                      hids_buf: &mut mlx_native::MlxBuffer,
                      tmp_gate_buf: &mut mlx_native::MlxBuffer,
                      tmp_up_buf: &mut mlx_native::MlxBuffer,
                      h_all_buf: &mut mlx_native::MlxBuffer,
                      silu_params_buf: &mlx_native::MlxBuffer,
                      reps: usize| -> f64 {
        let mut total_ns = 0u128;
        for _ in 0..reps {
            zero_htpe(htpe_buf);
            let t0 = Instant::now();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_id_mm_for_test(
                &mut encoder, registry, device,
                input_buf, gate_buf, ids_buf,
                htpe_buf, hids_buf, tmp_gate_buf,
                &dispatch,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            zero_htpe(htpe_buf);
            let mut encoder = device.command_encoder().unwrap();
            dispatch_id_mm_for_test(
                &mut encoder, registry, device,
                input_buf, up_buf, ids_buf,
                htpe_buf, hids_buf, tmp_up_buf,
                &dispatch,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_silu_mul(
                &mut encoder, registry, device.metal_device(),
                tmp_gate_buf, tmp_up_buf, h_all_buf,
                silu_params_buf, n_outputs,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            total_ns += t0.elapsed().as_nanos();
        }
        total_ns as f64 / reps as f64 / 1e6
    };

    let do_fused = |device: &MlxDevice,
                    registry: &mut KernelRegistry,
                    input_buf: &mlx_native::MlxBuffer,
                    gate_buf: &mlx_native::MlxBuffer,
                    up_buf: &mlx_native::MlxBuffer,
                    ids_buf: &mlx_native::MlxBuffer,
                    htpe_buf: &mut mlx_native::MlxBuffer,
                    hids_buf: &mut mlx_native::MlxBuffer,
                    h_all_buf: &mut mlx_native::MlxBuffer,
                    reps: usize| -> f64 {
        let mut total_ns = 0u128;
        for _ in 0..reps {
            zero_htpe(htpe_buf);
            let t0 = Instant::now();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_id_mm_fused_gate_up_silu_for_test(
                &mut encoder, registry, device,
                input_buf, gate_buf, up_buf, ids_buf,
                htpe_buf, hids_buf, h_all_buf,
                &dispatch,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            total_ns += t0.elapsed().as_nanos();
        }
        total_ns as f64 / reps as f64 / 1e6
    };

    // Warmup.
    let _ = do_unfused(
        &device, &mut registry, &input_buf, &gate_buf, &up_buf, &ids_buf,
        &mut htpe_buf, &mut hids_buf, &mut tmp_gate_buf, &mut tmp_up_buf,
        &mut h_all_buf, &silu_params_buf, warmup,
    );
    let _ = do_fused(
        &device, &mut registry, &input_buf, &gate_buf, &up_buf, &ids_buf,
        &mut htpe_buf, &mut hids_buf, &mut h_all_buf, warmup,
    );

    let unfused_ms = do_unfused(
        &device, &mut registry, &input_buf, &gate_buf, &up_buf, &ids_buf,
        &mut htpe_buf, &mut hids_buf, &mut tmp_gate_buf, &mut tmp_up_buf,
        &mut h_all_buf, &silu_params_buf, reps,
    );
    let fused_ms = do_fused(
        &device, &mut registry, &input_buf, &gate_buf, &up_buf, &ids_buf,
        &mut htpe_buf, &mut hids_buf, &mut h_all_buf, reps,
    );
    let speedup = unfused_ms / fused_ms;

    eprintln!(
        "[fused-mm_id-Q6_K bench] n_tokens={n_tokens:5} top_k={top_k} n_experts={n_experts:3} \
         n={n:5} k={k:5}  unfused={unfused_ms:6.2}ms  fused={fused_ms:6.2}ms  speedup={speedup:.3}x"
    );
}

#[test]
fn adr033_pi_task20_fused_mm_id_q6_K_microbench() {
    if std::env::var("HF2Q_RUN_FUSED_BENCH").is_err() {
        eprintln!(
            "[fused-mm_id-Q6_K bench] gated — set HF2Q_RUN_FUSED_BENCH=1 to run"
        );
        return;
    }

    // Qwen3.6 35B-A3B production MoE shape (per ADR-033 §Pi memory):
    //   hidden_size = 4096
    //   moe_intermediate_size = 768
    //   num_experts = 128
    //   num_experts_per_tok = 8
    //
    // Bench across a range of prefill chunk sizes. Use n_tokens >= 32
    // so the fused mm_id path is engaged (matches gpu_ffn.rs gate).

    let reps = 5;
    let warmup = 2;

    // Smaller shape (fewer experts) for fast iteration validation.
    bench_at_shape(128, 8, 32, 768, QK_K * 4 /* k=1024 */, reps, warmup, 0xBE_A6CF_1024);

    // Production Qwen3.6 35B-A3B shape — pp64 chunk.
    bench_at_shape(64, 8, 128, 768, QK_K * 16 /* k=4096 */, reps, warmup, 0xBE_A6CF_64);

    // Production Qwen3.6 35B-A3B shape — pp256 chunk.
    bench_at_shape(256, 8, 128, 768, QK_K * 16, reps, warmup, 0xBE_A6CF_256);

    // Production Qwen3.6 35B-A3B shape — pp1024 chunk.
    bench_at_shape(1024, 8, 128, 768, QK_K * 16, reps, warmup, 0xBE_A6CF_1024_2);
}
