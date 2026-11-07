//! ADR-065 E6 probe 3: `gated_delta_net_decode` with four sequences versus
//! four single-sequence dispatches, at Qwen3.8-27B DeltaNet shape
//! (16 key heads, 48 value heads, D_k = D_v = 128, q scale folded in).
//!
//! Asserted bitwise (output and `state_out`, plus `state_capture` for the
//! capture kernel):
//!
//! * `n_seqs = 4, n_tokens = 1` versus four `n_seqs = 1, n_tokens = 1`;
//! * `n_seqs = 4, n_tokens = 4` versus four `n_seqs = 1, n_tokens = 4`;
//! * the capture kernel at `n_seqs = 4, n_tokens = 4` versus four
//!   single-sequence capture dispatches.
//!
//! Recorded (bitwise verdict; bounded numerically): a 4-token window
//! versus four sequential 1-token decodes of the same sequence — token `t`'s
//! output and `state_capture[t]` against the state after `t + 1` steps.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, dead_code)]

include!("support/adr065_e6_common.rs");

use mlx_native::ops::gated_delta_net::{
    build_gated_delta_net_params_with_q_scale, GatedDeltaNetParams,
};
use mlx_native::ops::gated_delta_net_decode::{
    dispatch_gated_delta_net_decode, dispatch_gated_delta_net_decode_with_capture,
    SEQ_KERNEL_MIN_TOKENS,
};

const D: u32 = 128;
const N_K_HEADS: u32 = 16;
const N_V_HEADS: u32 = 48;
const SEQS: usize = 4;
const STATE: usize = (D * D * N_V_HEADS) as usize;

/// Per-sequence inputs for `tokens` tokens, in kernel layout.
#[derive(Clone)]
struct SeqInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

fn seq_inputs(rng: &mut E6Rng, tokens: usize) -> SeqInputs {
    let qk = (D * N_K_HEADS) as usize * tokens;
    let v = (D * N_V_HEADS) as usize * tokens;
    let scalars = N_V_HEADS as usize * tokens;
    SeqInputs {
        q: (0..qk).map(|_| rng.signed(1.0)).collect(),
        k: (0..qk).map(|_| rng.signed(0.1)).collect(),
        v: (0..v).map(|_| rng.signed(1.0)).collect(),
        g: (0..scalars).map(|_| 0.1 * rng.unit()).collect(),
        beta: (0..scalars).map(|_| 0.1 + 0.8 * rng.unit()).collect(),
        state: (0..STATE).map(|_| rng.signed(0.05)).collect(),
    }
}

/// Token `t` of a sequence as a one-token input (state left to the caller).
fn token_slice(inputs: &SeqInputs, t: usize) -> SeqInputs {
    let qk = (D * N_K_HEADS) as usize;
    let v = (D * N_V_HEADS) as usize;
    let s = N_V_HEADS as usize;
    SeqInputs {
        q: inputs.q[t * qk..(t + 1) * qk].to_vec(),
        k: inputs.k[t * qk..(t + 1) * qk].to_vec(),
        v: inputs.v[t * v..(t + 1) * v].to_vec(),
        g: inputs.g[t * s..(t + 1) * s].to_vec(),
        beta: inputs.beta[t * s..(t + 1) * s].to_vec(),
        state: Vec::new(),
    }
}

fn concat(seqs: &[SeqInputs]) -> SeqInputs {
    let mut out = SeqInputs {
        q: Vec::new(),
        k: Vec::new(),
        v: Vec::new(),
        g: Vec::new(),
        beta: Vec::new(),
        state: Vec::new(),
    };
    for seq in seqs {
        out.q.extend_from_slice(&seq.q);
        out.k.extend_from_slice(&seq.k);
        out.v.extend_from_slice(&seq.v);
        out.g.extend_from_slice(&seq.g);
        out.beta.extend_from_slice(&seq.beta);
        out.state.extend_from_slice(&seq.state);
    }
    out
}

struct GdnResult {
    output: Vec<f32>,
    state_out: Vec<f32>,
    capture: Option<Vec<f32>>,
}

fn run_gdn(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    inputs: &SeqInputs,
    n_seqs: u32,
    n_tokens: u32,
    capture: bool,
) -> GdnResult {
    let p = GatedDeltaNetParams {
        d_k: D,
        d_v: D,
        n_k_heads: N_K_HEADS,
        n_v_heads: N_V_HEADS,
        n_tokens,
        n_seqs,
    };
    let q = e6_f32_buffer(device, &inputs.q);
    let k = e6_f32_buffer(device, &inputs.k);
    let v = e6_f32_buffer(device, &inputs.v);
    let g = e6_f32_buffer(device, &inputs.g);
    let beta = e6_f32_buffer(device, &inputs.beta);
    let state_in = e6_f32_buffer(device, &inputs.state);
    let v_elems = (D * N_V_HEADS * n_tokens * n_seqs) as usize;
    let state_elems = STATE * n_seqs as usize;
    let output = device
        .alloc_buffer(v_elems * 4, DType::F32, vec![v_elems])
        .expect("output");
    let state_out = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("state_out");
    let params = build_gated_delta_net_params_with_q_scale(device, p, 1.0 / (D as f32).sqrt())
        .expect("params");
    let mut encoder = device.command_encoder().expect("encoder");
    let capture_buf = if capture {
        let elems = state_elems * n_tokens as usize;
        let buffer = device
            .alloc_buffer(elems * 4, DType::F32, vec![elems])
            .expect("capture");
        dispatch_gated_delta_net_decode_with_capture(
            &mut encoder,
            registry,
            device.metal_device(),
            &q,
            &k,
            &v,
            &g,
            &beta,
            &state_in,
            &output,
            &state_out,
            &params,
            &buffer,
            p,
        )
        .expect("GDN capture dispatch");
        Some(buffer)
    } else {
        dispatch_gated_delta_net_decode(
            &mut encoder,
            registry,
            device.metal_device(),
            &q,
            &k,
            &v,
            &g,
            &beta,
            &state_in,
            &output,
            &state_out,
            &params,
            p,
        )
        .expect("GDN dispatch");
        None
    };
    encoder.commit_and_wait().expect("GDN execution");
    GdnResult {
        output: e6_read_f32(&output),
        state_out: e6_read_f32(&state_out),
        capture: capture_buf.as_ref().map(e6_read_f32),
    }
}

/// Origin of the pipeline the dispatcher selected: the capture kernel, or
/// for the plain kernel the index-rebuild variant at one token and the
/// pointer-addressing variant from two tokens on.
fn kernel_origin(registry: &KernelRegistry, capture: bool, n_tokens: u32) -> String {
    let label = if capture {
        "gated_delta_net_decode_capture_f32_4"
    } else if n_tokens >= SEQ_KERNEL_MIN_TOKENS {
        "gated_delta_net_decode_seq_f32_4"
    } else {
        "gated_delta_net_decode_f32_4"
    };
    e6_pipeline_origin(registry, label, false)
}

fn record_and_assert(case: &str, part: &str, authority: &[f32], candidate: &[f32], origin: &str) {
    let compare = E6Compare::new(authority, candidate);
    let mut fields = e6_compare_fields(&compare);
    fields.push(("origin", origin.to_owned()));
    e6_record(
        "gated_delta_net_decode_seqs",
        &format!("{case}/{part}"),
        &fields,
    );
    assert!(
        compare.bitwise(),
        "{case} {part}: multi-sequence differs from single-sequence dispatch \
         ({} of {} mismatched, first {:?})",
        compare.mismatched,
        compare.elements,
        compare.first_mismatch
    );
}

fn probe_seqs(n_tokens: u32, capture: bool) {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut rng = E6Rng::new(0xe6_0d17 ^ (n_tokens as u64) << 8 ^ capture as u64);
    let seqs: Vec<SeqInputs> = (0..SEQS)
        .map(|_| seq_inputs(&mut rng, n_tokens as usize))
        .collect();
    let joint = run_gdn(
        &device,
        &mut registry,
        &concat(&seqs),
        SEQS as u32,
        n_tokens,
        capture,
    );
    let origin = kernel_origin(&registry, capture, n_tokens);
    let case = format!(
        "{}seqs4_tokens{n_tokens}",
        if capture { "capture_" } else { "" }
    );
    let v_per_seq = (D * N_V_HEADS * n_tokens) as usize;
    let capture_per_seq = STATE * n_tokens as usize;
    for (s, seq) in seqs.iter().enumerate() {
        let alone = run_gdn(&device, &mut registry, seq, 1, n_tokens, capture);
        record_and_assert(
            &case,
            &format!("seq{s}/output"),
            &alone.output,
            &joint.output[s * v_per_seq..(s + 1) * v_per_seq],
            &origin,
        );
        record_and_assert(
            &case,
            &format!("seq{s}/state_out"),
            &alone.state_out,
            &joint.state_out[s * STATE..(s + 1) * STATE],
            &origin,
        );
        if capture {
            record_and_assert(
                &case,
                &format!("seq{s}/state_capture"),
                alone.capture.as_ref().unwrap(),
                &joint.capture.as_ref().unwrap()[s * capture_per_seq..(s + 1) * capture_per_seq],
                &origin,
            );
        }
    }
}

#[test]
fn e6_gated_delta_net_decode_four_sequences_match_single_sequences() {
    probe_seqs(1, false);
}

#[test]
fn e6_gated_delta_net_decode_four_sequence_windows_match_single_sequences() {
    probe_seqs(4, false);
}

#[test]
fn e6_gated_delta_net_decode_capture_four_sequence_windows_match_single_sequences() {
    probe_seqs(4, true);
}

#[test]
fn e6_gated_delta_net_decode_window_versus_sequential_tokens() {
    const T: usize = 4;
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut rng = E6Rng::new(0xe6_05e9);
    let inputs = seq_inputs(&mut rng, T);
    let window = run_gdn(&device, &mut registry, &inputs, 1, T as u32, true);
    let origin = format!("{},{}", kernel_origin(&registry, true, T as u32), {
        // Make sure the plain kernel is compiled before reading its origin.
        let mut one = token_slice(&inputs, 0);
        one.state = inputs.state.clone();
        let _ = run_gdn(&device, &mut registry, &one, 1, 1, false);
        kernel_origin(&registry, false, 1)
    });
    let capture = window.capture.as_ref().unwrap();
    let v_per_token = (D * N_V_HEADS) as usize;
    let mut state = inputs.state.clone();
    for t in 0..T {
        let mut one = token_slice(&inputs, t);
        one.state = state;
        let step = run_gdn(&device, &mut registry, &one, 1, 1, false);
        for (part, authority, candidate) in [
            (
                "output",
                &step.output[..],
                &window.output[t * v_per_token..(t + 1) * v_per_token],
            ),
            (
                "state",
                &step.state_out[..],
                &capture[t * STATE..(t + 1) * STATE],
            ),
        ] {
            let compare = E6Compare::new(authority, candidate);
            let mut fields = e6_compare_fields(&compare);
            fields.push(("origin", origin.clone()));
            e6_record(
                "gated_delta_net_decode_window_vs_sequential",
                &format!("token{t}/{part}"),
                &fields,
            );
            assert!(
                candidate.iter().all(|value| value.is_finite()) && compare.max_abs_diff <= 1e-4,
                "token {t} {part}: window outside the numeric bound (max abs diff {:e})",
                compare.max_abs_diff
            );
        }
        state = step.state_out;
    }
    // The capture contract: the last captured state is state_out.
    let last = E6Compare::new(&window.state_out, &capture[(T - 1) * STATE..T * STATE]);
    assert!(last.bitwise(), "state_capture[T-1] must equal state_out");
}
