//! Tests for the fused Gated DeltaNet kernel (ADR-013 Decision 6).
//!
//! Acceptance criteria:
//! 1. Spec-driven: 1-seq, 1-head, 4-token, 8-dim example; hand-compute state
//!    evolution and output token-by-token; match to 1e-3 (F32).
//! 2. CPU parity test: pure-Rust scalar implementation of the spec; GPU
//!    output matches the scalar CPU impl on random inputs to 1e-3 (F32).
//! 3. Performance microbench (deferred to perf-follow-up iter).
//!
//! Recurrence:
//! ```
//! alpha = exp(-g[t])
//! delta = v[t] - state @ k[t]
//! state' = alpha * state + beta[t] * outer(delta, k[t])
//! output[t] = state' @ q[t]
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::gated_delta_net::{
    build_gated_delta_net_params, cpu_reference_f32, dispatch_gated_delta_net,
    GatedDeltaNetParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

fn upload_f32(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .expect("alloc");
    buf.as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(data);
    buf
}

fn run_gdn(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    p: GatedDeltaNetParams,
) -> (Vec<f32>, Vec<f32>) {
    let q_buf = upload_f32(device, q);
    let k_buf = upload_f32(device, k);
    let v_buf = upload_f32(device, v);
    let g_buf = upload_f32(device, g);
    let beta_buf = upload_f32(device, beta);
    let si_buf = upload_f32(device, state_in);

    let v_elems = (p.d_v * p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let state_elems = (p.d_k * p.d_v * p.n_v_heads * p.n_seqs) as usize;

    let out_buf = device
        .alloc_buffer(v_elems * 4, DType::F32, vec![v_elems])
        .expect("out");
    let so_buf = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("so");
    let params = build_gated_delta_net_params(device, p).expect("params");

    let mut enc = device.command_encoder().expect("enc");
    dispatch_gated_delta_net(
        &mut enc, registry, device.metal_device(),
        &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &si_buf, &out_buf, &so_buf,
        &params, p,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    (
        out_buf.as_slice::<f32>().expect("read out").to_vec(),
        so_buf.as_slice::<f32>().expect("read state").to_vec(),
    )
}

// ==================================================================
// Spec-driven: 1 seq, 1 head, 4 tokens, D=8, hand-computed state evolution
// ==================================================================
//
// Configuration:
//   D_k = D_v = 8, n_k_heads = n_v_heads = 1, n_tokens = 4, n_seqs = 1.
//   Initial state = 0 matrix of shape [8, 8]. All g[t] = 0 (alpha = 1,
//   no decay). All beta[t] = 1 (full delta injection).
//
// With state_0 = 0 and alpha=1, beta=1:
//   delta_0 = v[0] - 0 @ k[0] = v[0]
//   state_1 = 0 + outer(v[0], k[0]) = v[0] ⊗ k[0]
//   output_0 = state_1 @ q[0] = (v[0] ⊗ k[0]) @ q[0] = v[0] * <k[0], q[0]>
//
//   delta_1 = v[1] - state_1 @ k[1]
//           = v[1] - v[0] * <k[0], k[1]>
//   state_2 = state_1 + outer(delta_1, k[1])
//           = v[0]⊗k[0] + (v[1] - v[0]*<k[0], k[1]>) ⊗ k[1]
//   output_1 = state_2 @ q[1]
//
// With specifically chosen k vectors — each k[t] one-hot at position t —
// the recurrence simplifies dramatically:
//   k[0] = e_0 = [1,0,0,0,0,0,0,0]
//   k[1] = e_1
//   k[2] = e_2
//   k[3] = e_3
//   <k[t], k[t']> = δ_{t,t'}, so state is:
//   state_t = sum_{s<=t-1} v[s] ⊗ k[s]  (no cross-talk, no decay)
//   output_t = state_{t+1} @ q[t] = sum_{s<=t} v[s] * q[t][s]  (if k[s] = e_s)
//            = v[t] * q[t][t] + sum_{s<t} v[s] * q[t][s]
//
// This gives a clean hand-computable ground truth.
#[test]
fn test_gdn_spec_driven_1seq_1head_4tok_d8() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 8, d_v: 8, n_k_heads: 1, n_v_heads: 1, n_tokens: 4, n_seqs: 1,
    };

    // K: k[t] = e_t (one-hot). Layout [D_k, 1, 4, 1]; d_k fastest.
    let mut k = vec![0.0f32; 8 * 4];
    for t in 0..4 {
        k[t * 8 + t] = 1.0;
    }
    // V: arbitrary deterministic values.
    // v[t] = [t+1, t+2, ... t+8] (floats).
    let mut v = vec![0.0f32; 8 * 4];
    for t in 0..4 {
        for i in 0..8 {
            v[t * 8 + i] = (t + i + 1) as f32;
        }
    }
    // Q: arbitrary. Layout [D_k, 1, 4, 1].
    // q[t] = [10*t, 10*t+1, ..., 10*t+7].
    let mut q = vec![0.0f32; 8 * 4];
    for t in 0..4 {
        for i in 0..8 {
            q[t * 8 + i] = (10 * t + i) as f32;
        }
    }
    // g = 0 (alpha=1), beta = 1.
    let g = vec![0.0f32; 4];
    let beta = vec![1.0f32; 4];
    let state_in = vec![0.0f32; 8 * 8];

    let (got_out, got_state) = run_gdn(
        &device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p,
    );

    // Hand-computed expected:
    //   state_4[i, j] = v[j][i]  for j in 0..4, else 0
    //   (state_{t+1} accumulates outer(v[t], e_t); outer(v[t], e_t)[i, j] = v[t][i] if j==t)
    //   -> state[j=t, i=anything] column t = v[t].
    // So final state (D_v=8, D_k=8 with d_k fastest, i.e., state[d_k=j, d_v=i]):
    //   state[j, i] = v[j][i] if j < 4, else 0.
    let mut expected_state = vec![0.0f32; 8 * 8];
    for j in 0..4 {
        for i in 0..8 {
            // state[j, i] at offset i * D_k + j (d_k innermost)
            expected_state[i * 8 + j] = v[j * 8 + i];
        }
    }
    for (idx, (&g_s, &e_s)) in got_state.iter().zip(expected_state.iter()).enumerate() {
        let d = (g_s - e_s).abs();
        assert!(
            d < 1e-4,
            "state[{}]: got {}, expected {}",
            idx, g_s, e_s
        );
    }

    // output[t][i] = sum_{s<=t} v[s][i] * q[t][s]  (since k[s] = e_s).
    //              = v[t][i] * q[t][t] + sum_{s<t} v[s][i] * q[t][s]
    let mut expected_out = vec![0.0f32; 8 * 4];
    for t in 0..4 {
        for i in 0..8 {
            let mut acc = 0.0f32;
            for s in 0..=t {
                acc += v[s * 8 + i] * q[t * 8 + s];
            }
            expected_out[t * 8 + i] = acc;
        }
    }
    for (idx, (&g_o, &e_o)) in got_out.iter().zip(expected_out.iter()).enumerate() {
        let d = (g_o - e_o).abs();
        assert!(
            d < 1e-3,
            "output[{}]: got {}, expected {}",
            idx, g_o, e_o
        );
    }
}

// ==================================================================
// CPU-parity test with random inputs (ADR acceptance #2)
// ==================================================================

#[test]
fn test_gdn_cpu_parity_random_small() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 8, d_v: 8, n_k_heads: 2, n_v_heads: 4, n_tokens: 6, n_seqs: 2,
    };

    let qk_elems = (p.d_k * p.n_k_heads * p.n_tokens * p.n_seqs) as usize;
    let v_elems = (p.d_v * p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let sc_elems = (p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let state_elems = (p.d_k * p.d_v * p.n_v_heads * p.n_seqs) as usize;

    let mut seed = 0xABCD_1234u32;
    let step = |seed: &mut u32| {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    };
    let rand_signed = |seed: &mut u32, scale: f32| -> f32 {
        step(seed);
        ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
    };
    let rand_unsigned = |seed: &mut u32, lo: f32, hi: f32| -> f32 {
        step(seed);
        let u = ((*seed >> 8) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    };

    let q: Vec<f32> = (0..qk_elems).map(|_| rand_signed(&mut seed, 0.3)).collect();
    let k: Vec<f32> = (0..qk_elems).map(|_| rand_signed(&mut seed, 0.3)).collect();
    let v: Vec<f32> = (0..v_elems).map(|_| rand_signed(&mut seed, 0.5)).collect();
    // g in [0.05, 0.5] keeps alpha = exp(-g) in a stable numerical range.
    let g: Vec<f32> = (0..sc_elems).map(|_| rand_unsigned(&mut seed, 0.05, 0.5)).collect();
    // beta in [0.1, 0.9].
    let beta: Vec<f32> = (0..sc_elems).map(|_| rand_unsigned(&mut seed, 0.1, 0.9)).collect();
    let state_in: Vec<f32> = (0..state_elems).map(|_| rand_signed(&mut seed, 0.2)).collect();

    let (out_gpu, state_gpu) = run_gdn(
        &device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p,
    );
    let (out_cpu, state_cpu) = cpu_reference_f32(&q, &k, &v, &g, &beta, &state_in, p);

    for (idx, (&gu, &cu)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        let d = (gu - cu).abs();
        assert!(
            d < 1e-3,
            "output[{}]: gpu={}, cpu={}, diff={}",
            idx, gu, cu, d
        );
    }
    for (idx, (&gs, &cs)) in state_gpu.iter().zip(state_cpu.iter()).enumerate() {
        let d = (gs - cs).abs();
        assert!(
            d < 1e-3,
            "state[{}]: gpu={}, cpu={}, diff={}",
            idx, gs, cs, d
        );
    }
}

// ==================================================================
// Edge case: n_tokens=1 (decode regime)
// ==================================================================

#[test]
fn test_gdn_single_token_decode() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 8, d_v: 8, n_k_heads: 1, n_v_heads: 1, n_tokens: 1, n_seqs: 1,
    };

    let q: Vec<f32> = (0..8).map(|i| 0.1 * (i as f32)).collect();
    let k: Vec<f32> = (0..8).map(|i| -0.05 * (i as f32)).collect();
    let v: Vec<f32> = (0..8).map(|i| 0.5 - 0.1 * (i as f32)).collect();
    let g = vec![0.2f32];
    let beta = vec![0.7f32];
    let mut state_in = vec![0.0f32; 64];
    // Put some non-zero state to make the decode non-trivial.
    for (i, s) in state_in.iter_mut().enumerate() {
        *s = 0.01 * (i as f32);
    }

    let (out_gpu, state_gpu) = run_gdn(
        &device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p,
    );
    let (out_cpu, state_cpu) = cpu_reference_f32(&q, &k, &v, &g, &beta, &state_in, p);

    for (idx, (&gu, &cu)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            (gu - cu).abs() < 1e-4,
            "decode out[{}]: gpu={}, cpu={}",
            idx, gu, cu
        );
    }
    for (idx, (&gs, &cs)) in state_gpu.iter().zip(state_cpu.iter()).enumerate() {
        assert!(
            (gs - cs).abs() < 1e-4,
            "decode state[{}]: gpu={}, cpu={}",
            idx, gs, cs
        );
    }
}

// ==================================================================
// Multi-seq independence
// ==================================================================

#[test]
fn test_gdn_multi_seq_independence() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 8, d_v: 8, n_k_heads: 1, n_v_heads: 1, n_tokens: 3, n_seqs: 2,
    };

    // Two sequences with different state and different token content.
    let mut q = vec![0.0f32; 8 * 3 * 2];
    let mut k = vec![0.0f32; 8 * 3 * 2];
    let mut v = vec![0.0f32; 8 * 3 * 2];
    for s in 0..2 {
        for t in 0..3 {
            for i in 0..8 {
                let base = s * 8 * 3 + t * 8 + i;
                q[base] = (s as f32) * 0.5 + (t as f32) * 0.1 + 0.01 * (i as f32);
                k[base] = 0.1 * (i as f32) + 0.2 * (t as f32) - 0.3 * (s as f32);
                v[base] = 1.0 - 0.05 * ((s + t + i) as f32);
            }
        }
    }
    let g: Vec<f32> = (0..(3 * 2)).map(|i| 0.1 + 0.05 * i as f32).collect();
    let beta = vec![0.5f32; 3 * 2];
    let mut state_in = vec![0.0f32; 8 * 8 * 2];
    for (i, s) in state_in.iter_mut().enumerate() {
        *s = ((i % 7) as f32) * 0.01;
    }

    let (out_gpu, state_gpu) = run_gdn(
        &device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p,
    );
    let (out_cpu, state_cpu) = cpu_reference_f32(&q, &k, &v, &g, &beta, &state_in, p);

    for (idx, (&gu, &cu)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            (gu - cu).abs() < 1e-3,
            "multi-seq out[{}]: gpu={}, cpu={}",
            idx, gu, cu
        );
    }
    for (idx, (&gs, &cs)) in state_gpu.iter().zip(state_cpu.iter()).enumerate() {
        assert!(
            (gs - cs).abs() < 1e-3,
            "multi-seq state[{}]: gpu={}, cpu={}",
            idx, gs, cs
        );
    }
}

// ==================================================================
// GQA broadcast (n_v_heads > n_k_heads)
// ==================================================================

#[test]
fn test_gdn_gqa_broadcast_correct_k_head_picked() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 4, d_v: 4, n_k_heads: 2, n_v_heads: 6, n_tokens: 2, n_seqs: 1,
    };
    // group_ratio = 3. v_heads [0,1,2] -> k_head 0; v_heads [3,4,5] -> k_head 1.

    let mut k = vec![0.0f32; p.d_k as usize * 2 * 2]; // D_k * n_k_heads * n_tokens * 1
    // k_head 0, t=0: all ones. k_head 1, t=0: all tens. t=1 pattern different.
    for t in 0..2 {
        let base_t = t * (p.d_k * p.n_k_heads) as usize;
        for d in 0..p.d_k as usize {
            k[base_t + 0 * p.d_k as usize + d] = 1.0 + t as f32; // k_head 0
            k[base_t + 1 * p.d_k as usize + d] = 10.0 + t as f32; // k_head 1
        }
    }
    let q = k.clone();
    let v = vec![0.1f32; p.d_v as usize * p.n_v_heads as usize * 2];
    let g = vec![0.1f32; p.n_v_heads as usize * 2];
    let beta = vec![0.5f32; p.n_v_heads as usize * 2];
    let state_in = vec![0.0f32; p.d_k as usize * p.d_v as usize * p.n_v_heads as usize];

    let (out_gpu, _) = run_gdn(
        &device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p,
    );
    let (out_cpu, _) = cpu_reference_f32(&q, &k, &v, &g, &beta, &state_in, p);

    for (idx, (&gu, &cu)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            (gu - cu).abs() < 1e-4,
            "gqa out[{}]: gpu={}, cpu={}",
            idx, gu, cu
        );
    }

    // Verify that v_heads 0..3 produce identical outputs (shared k_head 0,
    // identical v/q/g/beta) AND differ from v_heads 3..6.
    let d_v = p.d_v as usize;
    let t = 0;
    let base = t * (p.n_v_heads as usize * d_v);
    for vh in 1..3 {
        for i in 0..d_v {
            assert!(
                (out_gpu[base + 0 * d_v + i] - out_gpu[base + vh * d_v + i]).abs() < 1e-5,
                "v_heads 0 and {} in same GQA group should produce identical output",
                vh
            );
        }
    }
    // v_head 0 and v_head 3 are in different groups — should differ.
    let mut any_differ = false;
    for i in 0..d_v {
        if (out_gpu[base + 0 * d_v + i] - out_gpu[base + 3 * d_v + i]).abs() > 1e-3 {
            any_differ = true;
            break;
        }
    }
    assert!(any_differ, "v_heads in different GQA groups produced identical output");
}

// ==================================================================
// Qwen3.5 full-shape smoke (D=128, many heads)
// ==================================================================

#[test]
fn test_gdn_qwen35_shape_smoke() {
    let (device, mut registry) = setup();
    // Scaled-down from Qwen3.5-MoE: n_k_heads=4, n_v_heads=8 (group_ratio=2);
    // D=128 matches real model. n_tokens=2 keeps test fast.
    let p = GatedDeltaNetParams {
        d_k: 128, d_v: 128,
        n_k_heads: 4, n_v_heads: 8,
        n_tokens: 2, n_seqs: 1,
    };

    let qk_elems = (p.d_k * p.n_k_heads * p.n_tokens * p.n_seqs) as usize;
    let v_elems = (p.d_v * p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let sc_elems = (p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let state_elems = (p.d_k * p.d_v * p.n_v_heads * p.n_seqs) as usize;

    let mut seed = 0x5EED_u32;
    let mut rand_signed = |scale: f32| -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((seed as i32 as f32) / (i32::MAX as f32)) * scale
    };
    let q: Vec<f32> = (0..qk_elems).map(|_| rand_signed(0.05)).collect();
    let k: Vec<f32> = (0..qk_elems).map(|_| rand_signed(0.05)).collect();
    let v: Vec<f32> = (0..v_elems).map(|_| rand_signed(0.1)).collect();
    let g: Vec<f32> = (0..sc_elems).map(|_| 0.15 + rand_signed(0.02)).collect();
    let beta: Vec<f32> = (0..sc_elems).map(|_| 0.6 + rand_signed(0.01)).collect();
    let state_in: Vec<f32> = (0..state_elems).map(|_| rand_signed(0.01)).collect();

    let (out_gpu, state_gpu) = run_gdn(
        &device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p,
    );
    let (out_cpu, state_cpu) = cpu_reference_f32(&q, &k, &v, &g, &beta, &state_in, p);

    for (idx, (&gu, &cu)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        let d = (gu - cu).abs();
        assert!(d < 5e-3, "qwen35 out[{}]: gpu={}, cpu={}, diff={}", idx, gu, cu, d);
    }
    for (idx, (&gs, &cs)) in state_gpu.iter().zip(state_cpu.iter()).enumerate() {
        let d = (gs - cs).abs();
        assert!(d < 5e-3, "qwen35 state[{}]: gpu={}, cpu={}, diff={}", idx, gs, cs, d);
    }
}

// ==================================================================
// Error handling
// ==================================================================

#[test]
fn test_gdn_rejects_zero_dim() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 0, d_v: 8, n_k_heads: 1, n_v_heads: 1, n_tokens: 1, n_seqs: 1,
    };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let params = build_gated_delta_net_params(&device, p).expect("pb");
    let mut enc = device.command_encoder().expect("enc");
    let res = dispatch_gated_delta_net(
        &mut enc, &mut registry, device.metal_device(),
        &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &params, p,
    );
    assert!(res.is_err(), "zero d_k should error");
}

#[test]
fn test_gdn_rejects_non_multiple_heads() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 4, d_v: 4, n_k_heads: 3, n_v_heads: 5, n_tokens: 1, n_seqs: 1,
    };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let params = build_gated_delta_net_params(&device, p).expect("pb");
    let mut enc = device.command_encoder().expect("enc");
    let res = dispatch_gated_delta_net(
        &mut enc, &mut registry, device.metal_device(),
        &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &params, p,
    );
    assert!(res.is_err(), "n_v_heads not multiple of n_k_heads should error");
}

#[test]
fn test_gdn_rejects_d_exceeds_max() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 256, d_v: 256, n_k_heads: 1, n_v_heads: 1, n_tokens: 1, n_seqs: 1,
    };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let params = build_gated_delta_net_params(&device, p).expect("pb");
    let mut enc = device.command_encoder().expect("enc");
    let res = dispatch_gated_delta_net(
        &mut enc, &mut registry, device.metal_device(),
        &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &params, p,
    );
    assert!(res.is_err(), "d > MAX_STATE_D should error");
}
