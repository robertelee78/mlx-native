//! ADR-034 task #90 (2026-05-21) — parity tests for the per-position
//! state-capture variant of `dispatch_gated_delta_net_decode`.
//!
//! Acceptance criteria:
//! 1. **Output parity**: capture variant's `output` byte-identical to
//!    non-capture `dispatch_gated_delta_net_decode` for the same inputs.
//! 2. **State_out parity**: capture variant's `state_out` byte-identical
//!    to non-capture path.
//! 3. **state_capture[..., n_tokens-1, ...] == state_out** (final-token
//!    capture slice matches the final state — the determinism contract
//!    documented in `dispatch_gated_delta_net_decode_with_capture`).
//! 4. **Pipeline loads** for all 3 NSG variants (1/2/4): kernel source
//!    compiles cleanly.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::gated_delta_net::{build_gated_delta_net_params, GatedDeltaNetParams};
use mlx_native::ops::gated_delta_net_decode::{
    dispatch_gated_delta_net_decode, dispatch_gated_delta_net_decode_with_capture,
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
    buf.as_mut_slice::<f32>().expect("mut").copy_from_slice(data);
    buf
}

fn rand_vec(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (*seed >> 8) as f32 / ((1u32 << 24) as f32);
            (r * 2.0 - 1.0) * scale
        })
        .collect()
}

fn random_inputs(
    p: GatedDeltaNetParams,
    seed: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let qk_n = (p.d_k * p.n_k_heads * p.n_tokens * p.n_seqs) as usize;
    let v_n = (p.d_v * p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let scalar_n = (p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let state_n = (p.d_k * p.d_v * p.n_v_heads * p.n_seqs) as usize;
    let mut s = seed;
    let q = rand_vec(&mut s, qk_n, 0.1);
    let k = rand_vec(&mut s, qk_n, 0.1);
    let v = rand_vec(&mut s, v_n, 0.1);
    let g: Vec<f32> = rand_vec(&mut s, scalar_n, 0.05)
        .iter()
        .map(|x| x.abs())
        .collect();
    let beta: Vec<f32> = rand_vec(&mut s, scalar_n, 1.0)
        .iter()
        .map(|x| 0.5 + 0.4 * x)
        .collect();
    let state = rand_vec(&mut s, state_n, 0.05);
    (q, k, v, g, beta, state)
}

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

fn run_decode_with_capture(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    p: GatedDeltaNetParams,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_buf = upload_f32(device, q);
    let k_buf = upload_f32(device, k);
    let v_buf = upload_f32(device, v);
    let g_buf = upload_f32(device, g);
    let beta_buf = upload_f32(device, beta);
    let si_buf = upload_f32(device, state_in);
    let v_elems = (p.d_v * p.n_v_heads * p.n_tokens * p.n_seqs) as usize;
    let state_elems = (p.d_k * p.d_v * p.n_v_heads * p.n_seqs) as usize;
    let capture_elems = state_elems * (p.n_tokens as usize);
    let out_buf = device
        .alloc_buffer(v_elems * 4, DType::F32, vec![v_elems])
        .expect("out");
    let so_buf = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("so");
    let sc_buf = device
        .alloc_buffer(capture_elems * 4, DType::F32, vec![capture_elems])
        .expect("sc");
    let params = build_gated_delta_net_params(device, p).expect("params");
    let mut enc = device.command_encoder().expect("enc");
    dispatch_gated_delta_net_decode_with_capture(
        &mut enc, registry, device.metal_device(),
        &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &si_buf, &out_buf, &so_buf,
        &params, &sc_buf, p,
    )
    .expect("dispatch decode_with_capture");
    enc.commit_and_wait().expect("commit");
    (
        out_buf.as_slice::<f32>().expect("read out").to_vec(),
        so_buf.as_slice::<f32>().expect("read state").to_vec(),
        sc_buf.as_slice::<f32>().expect("read capture").to_vec(),
    )
}

fn assert_byte_identical(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: len mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: byte mismatch at idx {} ({} vs {})",
            i, x, y
        );
    }
}

/// AC#1+2: capture variant's output + state_out byte-identical to
/// non-capture path on a NSG=1 (D_k=32) tiny shape.
#[test]
fn capture_output_state_byte_identical_nsg1_d32() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 32, d_v: 32, n_k_heads: 1, n_v_heads: 2, n_tokens: 1, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0xCAFE);
    let (dec_out, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (cap_out, cap_state, cap_capture) =
        run_decode_with_capture(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    assert_byte_identical("nsg1 output", &cap_out, &dec_out);
    assert_byte_identical("nsg1 state_out", &cap_state, &dec_state);
    // AC#3: state_capture[..., n_tokens-1, ...] == state_out (final slice
    // matches final state). With n_tokens=1 the capture buffer IS the
    // state_out exactly.
    assert_byte_identical("nsg1 capture[last] == state_out", &cap_capture, &dec_state);
}

/// AC#1+2+3 at NSG=2 (D_k=64).
#[test]
fn capture_output_state_byte_identical_nsg2_d64() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 64, d_v: 64, n_k_heads: 2, n_v_heads: 4, n_tokens: 1, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0x1234);
    let (dec_out, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (cap_out, cap_state, cap_capture) =
        run_decode_with_capture(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    assert_byte_identical("nsg2 output", &cap_out, &dec_out);
    assert_byte_identical("nsg2 state_out", &cap_state, &dec_state);
    assert_byte_identical("nsg2 capture[last] == state_out", &cap_capture, &dec_state);
}

/// AC#1+2+3 at NSG=4 (D_k=128) — Qwen3.5/3.6 production shape.
#[test]
fn capture_output_state_byte_identical_qwen35_shape() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 128, d_v: 128, n_k_heads: 16, n_v_heads: 32, n_tokens: 1, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0xBEEF);
    let (dec_out, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (cap_out, cap_state, cap_capture) =
        run_decode_with_capture(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    assert_byte_identical("qwen35 output", &cap_out, &dec_out);
    assert_byte_identical("qwen35 state_out", &cap_state, &dec_state);
    assert_byte_identical("qwen35 capture[last] == state_out", &cap_capture, &dec_state);
}

/// ADR-034 task #90 Step 5 (2026-05-21) — STRONG parity test for the
/// INTERMEDIATE recurrent captures: capture[t] for any t ∈ [0, n_tokens)
/// must equal state_out from a separate dispatch_gated_delta_net_decode
/// call with n_tokens=t+1 (truncated q/k/v/g/beta inputs). This is the
/// ground truth for K=N partial-reject rollback.
#[test]
fn capture_intermediate_t_matches_truncated_dispatch() {
    let (device, mut registry) = setup();
    let p_full = GatedDeltaNetParams {
        d_k: 128, d_v: 128, n_k_heads: 16, n_v_heads: 32, n_tokens: 4, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p_full, 0x517E);
    let (_, _, cap_capture) =
        run_decode_with_capture(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p_full);
    let state_elems = (p_full.d_k * p_full.d_v * p_full.n_v_heads * p_full.n_seqs) as usize;

    for t in 0..(p_full.n_tokens as usize - 1) {
        // Truncate inputs to n_tokens=t+1: each (q, k, v, g, beta) is
        // [..., n_tokens, n_seqs] with n_tokens varying. The arrays are
        // laid out token-major within each head, so truncating to t+1
        // tokens means slicing [0 .. (t+1) * per_token_stride] for each.
        let qk_per_tok = (p_full.d_k * p_full.n_k_heads * p_full.n_seqs) as usize;
        let v_per_tok = (p_full.d_v * p_full.n_v_heads * p_full.n_seqs) as usize;
        let sc_per_tok = (p_full.n_v_heads * p_full.n_seqs) as usize;
        let q_trunc = q[..qk_per_tok * (t + 1)].to_vec();
        let k_trunc = k[..qk_per_tok * (t + 1)].to_vec();
        let v_trunc = v[..v_per_tok * (t + 1)].to_vec();
        let g_trunc = g[..sc_per_tok * (t + 1)].to_vec();
        let beta_trunc = beta[..sc_per_tok * (t + 1)].to_vec();
        let p_trunc = GatedDeltaNetParams {
            d_k: p_full.d_k, d_v: p_full.d_v,
            n_k_heads: p_full.n_k_heads, n_v_heads: p_full.n_v_heads,
            n_tokens: (t + 1) as u32, n_seqs: p_full.n_seqs,
        };
        let (_y_trunc, state_trunc) = run_decode(
            &device, &mut registry,
            &q_trunc, &k_trunc, &v_trunc, &g_trunc, &beta_trunc,
            &state_in, p_trunc,
        );

        // capture[t] must byte-match state_trunc.
        let cap_t_offset = t * state_elems;
        let cap_t = &cap_capture[cap_t_offset..cap_t_offset + state_elems];
        assert_byte_identical(
            &format!("recurrent capture[t={t}] vs trunc(n_tokens={})", t + 1),
            cap_t,
            &state_trunc,
        );
    }
}

/// AC#3 with n_tokens > 1: the LAST slice of state_capture must equal
/// state_out. The earlier slices should NOT equal final state (they
/// represent intermediate per-position states).
#[test]
fn capture_last_slice_equals_state_out_n_tokens_4() {
    let (device, mut registry) = setup();
    let p = GatedDeltaNetParams {
        d_k: 128, d_v: 128, n_k_heads: 16, n_v_heads: 32, n_tokens: 4, n_seqs: 1,
    };
    let (q, k, v, g, beta, state_in) = random_inputs(p, 0xDEAD);
    let (_, dec_state) = run_decode(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    let (_, cap_state, cap_capture) =
        run_decode_with_capture(&device, &mut registry, &q, &k, &v, &g, &beta, &state_in, p);
    assert_byte_identical("nt4 state_out", &cap_state, &dec_state);

    let state_elems = (p.d_k * p.d_v * p.n_v_heads * p.n_seqs) as usize;
    assert_eq!(cap_capture.len(), state_elems * (p.n_tokens as usize));
    // Last slice = state after token (n_tokens-1) = final state.
    let last_slice = &cap_capture[state_elems * 3..state_elems * 4];
    assert_byte_identical("nt4 capture[3] == state_out", last_slice, &dec_state);
    // First slice should differ from final state (otherwise the recurrence
    // is degenerate and the test inputs are too uniform).
    let first_slice = &cap_capture[0..state_elems];
    let differs = first_slice
        .iter()
        .zip(dec_state.iter())
        .any(|(&a, &b)| a.to_bits() != b.to_bits());
    assert!(
        differs,
        "capture[0] unexpectedly equals state_out — recurrence may be degenerate"
    );
}
