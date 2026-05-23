//! ADR-033 §Pi Task #25 iter 24 (2026-05-23) — K=256 chunk vs autoregressive
//! perf bench at production-relevant shapes.
//!
//! Goal: empirically demonstrate that the new K=256 chunk-parallel pipeline
//! (iters 19-22) closes the autoregressive-fallback gap that caused
//! hf2q to lag llama.cpp at Qwen3.6 prefill. Llama.cpp uses a chunk-scan
//! path natively; with iter 22's K=256 dispatch wired up via hf2q
//! iter 23 chunk_path_eligible, hf2q can now match (or exceed) that path.
//!
//! This is a KERNEL-LEVEL perf test (not end-to-end model) — gives precise
//! control over seq_len (avoids tokenization off-by-N from the production
//! binary), and isolates the chunk-vs-autoreg performance delta.
//!
//! Gated behind HF2Q_RUN_K256_BENCH=1 to keep CI fast.
//!
//! Shape: B=1, T=512 (= 8 chunks of 64), Hg=Hv=1, K=256, V=64.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::ops::chunk_gated_delta_rule::{ChunkGatedDeltaRuleParams, FIXED_BT};
use mlx_native::ops::chunk_gated_delta_rule_bank_split::{
    dispatch_chunk_gated_delta_rule_fwd_k256_bank_split, BANK_SPLIT_K,
};
use mlx_native::ops::gated_delta_net::{
    build_gated_delta_net_params, dispatch_gated_delta_net, GatedDeltaNetParams,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};
use std::time::Instant;

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

fn random_bf16_vec(n: usize, seed: u64) -> Vec<half::bf16> {
    let mut state = seed;
    (0..n).map(|_| half::bf16::from_f32(random_pm1(&mut state) * 0.1)).collect()
}

fn random_f32_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| random_pm1(&mut state) * 0.1).collect()
}

fn bench_k256_at_t(t: u32, reps: usize, warmup: usize) -> (f64, f64) {
    let p = ChunkGatedDeltaRuleParams {
        b: 1,
        t,
        hg: 1,
        h: 1,
        k: BANK_SPLIT_K, // 256
        v: 64,
        bt: FIXED_BT, // 64
        scale: 1.0,
        use_qk_l2norm: true,
    };

    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    // Buffer sizes for the chunk pipeline contract.
    let qk_elems = (p.b * p.t * p.hg * p.k) as usize;
    let v_elems = (p.b * p.t * p.h * p.v) as usize;
    let g_elems = (p.b * p.t * p.h) as usize;
    let h0_elems = (p.b * p.h * p.v * p.k) as usize;
    let o_elems = v_elems;

    // Inputs (random bf16 / f32).
    let q_data = random_bf16_vec(qk_elems, 0x6789_ABCD);
    let k_data = random_bf16_vec(qk_elems, 0xABCD_6789);
    let v_data = random_bf16_vec(v_elems, 0xDEF0_1234);
    let g_data = random_f32_vec(g_elems, 0x1111_2222);
    let beta_data = random_f32_vec(g_elems, 0x3333_4444);
    let h0_data = random_f32_vec(h0_elems, 0x5555_6666);

    let mut q_buf = device.alloc_buffer(qk_elems * 2, DType::BF16, vec![qk_elems]).unwrap();
    q_buf.as_mut_slice::<half::bf16>().unwrap().copy_from_slice(&q_data);
    let mut k_buf = device.alloc_buffer(qk_elems * 2, DType::BF16, vec![qk_elems]).unwrap();
    k_buf.as_mut_slice::<half::bf16>().unwrap().copy_from_slice(&k_data);
    let mut v_buf = device.alloc_buffer(v_elems * 2, DType::BF16, vec![v_elems]).unwrap();
    v_buf.as_mut_slice::<half::bf16>().unwrap().copy_from_slice(&v_data);
    let mut g_buf = device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems]).unwrap();
    g_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&g_data);
    let mut beta_buf = device.alloc_buffer(g_elems * 4, DType::F32, vec![g_elems]).unwrap();
    beta_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&beta_data);
    let mut h0_buf = device.alloc_buffer(h0_elems * 4, DType::F32, vec![h0_elems]).unwrap();
    h0_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&h0_data);
    let o_buf = device.alloc_buffer(o_elems * 2, DType::BF16, vec![o_elems]).unwrap();
    let final_state_buf =
        device.alloc_buffer(h0_elems * 4, DType::F32, vec![h0_elems]).unwrap();

    // ---- Chunk path: warmup + reps ----
    // Inline the bench loop to avoid closure borrow issues.
    fn bench_chunk_inline(
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        q: &mlx_native::MlxBuffer,
        k: &mlx_native::MlxBuffer,
        v: &mlx_native::MlxBuffer,
        g: &mlx_native::MlxBuffer,
        beta: &mlx_native::MlxBuffer,
        h0: &mlx_native::MlxBuffer,
        o: &mlx_native::MlxBuffer,
        final_state: &mlx_native::MlxBuffer,
        p: ChunkGatedDeltaRuleParams,
        reps: usize,
    ) -> f64 {
        let mut total_ns = 0u128;
        for _ in 0..reps {
            let t0 = Instant::now();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_chunk_gated_delta_rule_fwd_k256_bank_split(
                &mut encoder, registry, device,
                q, k, v, g, beta, h0, o, final_state, p,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            total_ns += t0.elapsed().as_nanos();
        }
        total_ns as f64 / reps as f64 / 1e6
    }

    // ---- Autoregressive path: also warmup + reps ----
    //
    // The autoregressive `dispatch_gated_delta_net` takes the same inputs
    // but processes tokens one at a time inside the kernel. It's the
    // FALLBACK path currently used by Qwen3.6 when the chunk path doesn't
    // engage (e.g. seq_len % 64 != 0). Direct A/B with the chunk path at
    // the same T/K=256/V=64 shape isolates the per-token cost of the
    // autoreg approach.
    let autoreg_params_value = GatedDeltaNetParams {
        d_k: p.k,
        d_v: p.v,
        n_k_heads: p.hg,
        n_v_heads: p.h,
        n_tokens: p.t,
        n_seqs: p.b,
    };
    let autoreg_params_buf = build_gated_delta_net_params(&device, autoreg_params_value).unwrap();
    let autoreg_o_buf =
        device.alloc_buffer(o_elems * 4, DType::F32, vec![o_elems]).unwrap();
    let autoreg_state_out_buf =
        device.alloc_buffer(h0_elems * 4, DType::F32, vec![h0_elems]).unwrap();
    // The autoreg path takes f32 inputs (q, k, v, g, beta, state_in,
    // output, state_out). Convert our bf16 inputs to f32 by uploading
    // fresh f32 buffers (the gated_delta_net_f32 kernel expects f32).
    let mut q_f32 = device.alloc_buffer(qk_elems * 4, DType::F32, vec![qk_elems]).unwrap();
    q_f32.as_mut_slice::<f32>().unwrap().copy_from_slice(
        &q_data.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
    );
    let mut k_f32 = device.alloc_buffer(qk_elems * 4, DType::F32, vec![qk_elems]).unwrap();
    k_f32.as_mut_slice::<f32>().unwrap().copy_from_slice(
        &k_data.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
    );
    let mut v_f32 = device.alloc_buffer(v_elems * 4, DType::F32, vec![v_elems]).unwrap();
    v_f32.as_mut_slice::<f32>().unwrap().copy_from_slice(
        &v_data.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
    );

    #[allow(clippy::too_many_arguments)]
    fn bench_autoreg_inline(
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        q_f32: &mlx_native::MlxBuffer,
        k_f32: &mlx_native::MlxBuffer,
        v_f32: &mlx_native::MlxBuffer,
        g: &mlx_native::MlxBuffer,
        beta: &mlx_native::MlxBuffer,
        h0: &mlx_native::MlxBuffer,
        o: &mlx_native::MlxBuffer,
        state_out: &mlx_native::MlxBuffer,
        params_buf: &mlx_native::MlxBuffer,
        params: GatedDeltaNetParams,
        reps: usize,
    ) -> f64 {
        let mut total_ns = 0u128;
        for _ in 0..reps {
            let t0 = Instant::now();
            let mut encoder = device.command_encoder().unwrap();
            dispatch_gated_delta_net(
                &mut encoder, registry, device.metal_device(),
                q_f32, k_f32, v_f32, g, beta, h0, o, state_out,
                params_buf, params,
            )
            .unwrap();
            encoder.commit_and_wait().unwrap();
            total_ns += t0.elapsed().as_nanos();
        }
        total_ns as f64 / reps as f64 / 1e6
    }

    // Warmup.
    let _ = bench_chunk_inline(
        &device, &mut registry,
        &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &h0_buf,
        &o_buf, &final_state_buf, p, warmup,
    );
    let _ = bench_autoreg_inline(
        &device, &mut registry,
        &q_f32, &k_f32, &v_f32, &g_buf, &beta_buf, &h0_buf,
        &autoreg_o_buf, &autoreg_state_out_buf,
        &autoreg_params_buf, autoreg_params_value, warmup,
    );

    let chunk_ms = bench_chunk_inline(
        &device, &mut registry,
        &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &h0_buf,
        &o_buf, &final_state_buf, p, reps,
    );
    let autoreg_ms = bench_autoreg_inline(
        &device, &mut registry,
        &q_f32, &k_f32, &v_f32, &g_buf, &beta_buf, &h0_buf,
        &autoreg_o_buf, &autoreg_state_out_buf,
        &autoreg_params_buf, autoreg_params_value, reps,
    );
    (chunk_ms, autoreg_ms)
}

#[test]
fn adr_033_pi_task25_chunk_k256_perf() {
    if std::env::var("HF2Q_RUN_K256_BENCH").is_err() {
        eprintln!(
            "[k256 perf bench] gated — set HF2Q_RUN_K256_BENCH=1 to run \
             (this test is excluded from default CI to keep runs fast)"
        );
        return;
    }

    let reps = 10;
    let warmup = 3;

    eprintln!(
        "[k256 perf bench] B=1, K=256, V=64, BT=64, Hg=Hv=1 — reps={reps}, warmup={warmup}"
    );
    eprintln!(
        "{:<8} {:>18} {:>20} {:>14}",
        "T", "chunk_k256 (ms/iter)", "autoregressive (ms/iter)", "speedup"
    );

    // Production-relevant prefill sizes: 64, 128, 256, 512, 1024.
    // All are multiples of FIXED_BT=64 (chunk-eligible).
    for t in &[64u32, 128, 256, 512, 1024] {
        let (chunk_ms, autoreg_ms) = bench_k256_at_t(*t, reps, warmup);
        let speedup = autoreg_ms / chunk_ms;
        eprintln!(
            "{:<8} {:>18.3} {:>20.3} {:>13.2}x",
            t, chunk_ms, autoreg_ms, speedup
        );
    }
}
