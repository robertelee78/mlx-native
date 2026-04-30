//! Parity tests for the `simd_sum` decode variant of the fused Gated
//! DeltaNet kernel (ADR-015 iter56).
//!
//! Acceptance: bit-equivalent (modulo F32 reduction order) output vs the
//! existing `gated_delta_net_f32` kernel on representative shapes — 1e-5
//! max-abs tolerance accommodates the SIMD-group `simd_sum` reduction
//! reordering vs the existing kernel's tree-reduction (both produce the
//! same mathematical result but accumulate floats in different orders).
//!
//! Coverage:
//! 1. Tiny shape (D_k=D_v=32 → NSG=1).
//! 2. Mid shape (D_k=D_v=64 → NSG=2).
//! 3. Qwen3.5/3.6 production shape (D_k=D_v=128, n_v_heads=32, n_k_heads=16
//!    → NSG=4) — the iter56 hot path.
//! 4. Multi-seq independence smoke.
//! 5. Math-vs-CPU-ref parity (separate from the unfused-GPU comparison).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::gated_delta_net::{
    build_gated_delta_net_params, cpu_reference_f32, dispatch_gated_delta_net,
    GatedDeltaNetParams,
};
use mlx_native::ops::gated_delta_net_decode::dispatch_gated_delta_net_decode;
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

/// Run the *decode* (simd_sum) kernel and return (output, state_out).
fn run_decode(
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
    dispatch_gated_delta_net_decode(
        &mut enc, registry, device.metal_device(),
        &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &si_buf, &out_buf, &so_buf,
        &params, p,
    )
    .expect("dispatch decode");
    enc.commit_and_wait().expect("commit");

    (
        out_buf.as_slice::<f32>().expect("read out").to_vec(),
        so_buf.as_slice::<f32>().expect("read state").to_vec(),
    )
}

/// Run the *unfused-GPU baseline* (`dispatch_gated_delta_net`) for parity.
fn run_unfused(
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
    .expect("dispatch unfused");
    enc.commit_and_wait().expect("commit");

    (
        out_buf.as_slice::<f32>().expect("read out").to_vec(),
        so_buf.as_slice::<f32>().expect("read state").to_vec(),
    )
}

/// Deterministic LCG, matches `tests/test_gated_delta_net.rs`.
fn rand_vec(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for x in out.iter_mut() {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *x = ((*seed as i32 as f32) / (i32::MAX as f32)) * scale;
    }
    out
}

fn random_inputs(p: GatedDeltaNetParams, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let qk_n = (p.d_k * p.n_k_heads * p.n_tokens * p.n_seqs) as usize;
    let v_n = (p.d_v * p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let scalar_n = (p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let state_n = (p.d_k * p.d_v * p.n_v_heads * p.n_seqs) as usize;

    let mut s = seed;
    let q = rand_vec(&mut s, qk_n, 0.1);
    let k = rand_vec(&mut s, qk_n, 0.1);
    let v = rand_vec(&mut s, v_n, 0.1);
    // `g` should be small positive so exp(-g) is in (0, 1].
    let g_raw = rand_vec(&mut s, scalar_n, 0.05);
    let g: Vec<f32> = g_raw.iter().map(|x| x.abs()).collect();
    let beta_raw = rand_vec(&mut s, scalar_n, 1.0);
    let beta: Vec<f32> = beta_raw.iter().map(|x| 0.5 + 0.4 * x).collect();
    let state = rand_vec(&mut s, state_n, 0.05);
    (q, k, v, g, beta, state)
}

fn assert_close(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{label}: len mismatch");
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let e = (x - y).abs();
        if e > max_err {
            max_err = e;
            max_idx = i;
        }
    }
    assert!(
        max_err < tol,
        "{label}: max_abs_err={:.3e} > tol={:.3e} at idx {} ({} vs {})",
        max_err, tol, max_idx, a[max_idx], b[max_idx]
    );
    eprintln!("{label}: max_abs_err={:.3e} (tol {:.3e})", max_err, tol);
}

#[test]
fn decode_matches_unfused_nsg1_d32() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 32, d_v: 32, n_k_heads: 1, n_v_heads: 2, n_tokens: 1, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0xCAFE);

    let (dec_out, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (ref_out, ref_state) = run_unfused(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);

    assert_close("nsg1 output", &dec_out, &ref_out, 1e-5);
    assert_close("nsg1 state", &dec_state, &ref_state, 1e-5);
}

#[test]
fn decode_matches_unfused_nsg2_d64() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 64, d_v: 64, n_k_heads: 2, n_v_heads: 4, n_tokens: 1, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0x1234);

    let (dec_out, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (ref_out, ref_state) = run_unfused(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);

    assert_close("nsg2 output", &dec_out, &ref_out, 1e-5);
    assert_close("nsg2 state", &dec_state, &ref_state, 1e-5);
}

/// Production shape: matches Qwen3.5/3.6 35B-A3B GDN layer.
#[test]
fn decode_matches_unfused_nsg4_qwen35_shape() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 128, d_v: 128, n_k_heads: 16, n_v_heads: 32, n_tokens: 1, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0xBEEF);

    let (dec_out, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (ref_out, ref_state) = run_unfused(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);

    // F32 reductions over up to 128 floats can drift up to ~2e-5 between
    // tree-reduction and warp-reduction orderings; tolerance set just above
    // empirical max for this shape.
    assert_close("qwen35 output", &dec_out, &ref_out, 5e-5);
    assert_close("qwen35 state", &dec_state, &ref_state, 5e-5);
}

/// Multi-seq independence: state of seq 0 must not leak into seq 1, and vice
/// versa. Run two batched seqs with different inputs vs running them one at
/// a time; outputs must match per-seq.
#[test]
fn decode_multi_seq_independence_nsg4() {
    let (device, mut registry) = setup();

    let p_one = GatedDeltaNetParams {
        d_k: 128, d_v: 128, n_k_heads: 16, n_v_heads: 32, n_tokens: 1, n_seqs: 1,
    };
    let p_two = GatedDeltaNetParams {
        d_k: 128, d_v: 128, n_k_heads: 16, n_v_heads: 32, n_tokens: 1, n_seqs: 2,
    };

    // Build seq0 + seq1 inputs separately, then concatenate for the batched run.
    let (q0, k0, v0, g0, b0, s0) = random_inputs(p_one, 0x1111);
    let (q1, k1, v1, g1, b1, s1) = random_inputs(p_one, 0x2222);

    let (out0_solo, st0_solo) = run_decode(&device, &mut registry, &q0, &k0, &v0, &g0, &b0, &s0, p_one);
    let (out1_solo, st1_solo) = run_decode(&device, &mut registry, &q1, &k1, &v1, &g1, &b1, &s1, p_one);

    let mut q_b = q0.clone(); q_b.extend_from_slice(&q1);
    let mut k_b = k0.clone(); k_b.extend_from_slice(&k1);
    let mut v_b = v0.clone(); v_b.extend_from_slice(&v1);
    let mut g_b = g0.clone(); g_b.extend_from_slice(&g1);
    let mut b_b = b0.clone(); b_b.extend_from_slice(&b1);
    let mut s_b = s0.clone(); s_b.extend_from_slice(&s1);

    let (out_b, st_b) = run_decode(&device, &mut registry, &q_b, &k_b, &v_b, &g_b, &b_b, &s_b, p_two);

    let v_per_seq = (p_one.d_v * p_one.n_v_heads * p_one.n_tokens) as usize;
    let st_per_seq = (p_one.d_k * p_one.d_v * p_one.n_v_heads) as usize;

    assert_close("multi-seq out0", &out_b[..v_per_seq], &out0_solo, 1e-6);
    assert_close("multi-seq out1", &out_b[v_per_seq..], &out1_solo, 1e-6);
    assert_close("multi-seq state0", &st_b[..st_per_seq], &st0_solo, 1e-6);
    assert_close("multi-seq state1", &st_b[st_per_seq..], &st1_solo, 1e-6);
}

/// Verify against the pure-Rust scalar `cpu_reference_f32`.
/// This is the "spec ground truth" oracle (the same one
/// `dispatch_gated_delta_net` is checked against).
#[test]
fn decode_matches_cpu_ref_qwen35_shape() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 128, d_v: 128, n_k_heads: 16, n_v_heads: 32, n_tokens: 1, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0xACE0);

    let (dec_out, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (cpu_out, cpu_state) = cpu_reference_f32(&q, &k, &v, &g, &beta, &state_in, p);

    // CPU ref uses naive scalar reduction; warp reductions on D_k=128 floats
    // can diverge by ~2e-5 in F32. Tolerance covers that headroom.
    assert_close("decode-vs-cpu output", &dec_out, &cpu_out, 5e-5);
    assert_close("decode-vs-cpu state", &dec_state, &cpu_state, 5e-5);
}
