//! `gated_delta_net_decode_f32_<NSG>` (running per-token pointers,
//! compile-time D_k) versus the pre-rewrite reference
//! `gated_delta_net_decode_legacy_f32_<NSG>` (per-token `uint` index
//! rebuild): output `y` and final state must be bitwise identical.
//!
//! Matrix: Qwen3.8 heads (16 k, 48 v, D 128) and Qwen3.6 heads (16 k,
//! 32 v, D 128) × n_tokens {1, 2, 17, 512} × n_seqs {1, 4} × q-scale mode
//! (`params[8]` = 0 with q pre-scaled, and `params[8]` = bits(1/√D_k)) ×
//! buffer placement (whole buffers, and slice views at distinct non-zero,
//! non-16-byte-aligned offsets inside padded parents). Inputs are random
//! with g/beta extremes mixed in. For slice views the whole parent buffers
//! (padding included) are compared, so neither kernel may write outside
//! its view.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::ops::gated_delta_net::{
    build_gated_delta_net_params, build_gated_delta_net_params_with_q_scale, GatedDeltaNetParams,
};
use mlx_native::ops::gated_delta_net_decode::{
    dispatch_gated_delta_net_decode, dispatch_gated_delta_net_decode_legacy,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const D: u32 = 128;
/// Sentinel bit pattern for padding and untouched outputs.
const SENTINEL: u32 = 0x7fc0_dead;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn signed(&mut self, scale: f32) -> f32 {
        (2.0 * self.unit() - 1.0) * scale
    }
}

/// g as the production caller produces it (softplus·(−ssm_a) ≥ 0, so
/// alpha = exp(−g) ≤ 1), with extremes: 0, tiny, large (alpha ≈ 1e−13),
/// huge (alpha underflows), and an occasional negative (alpha > 1).
fn g_value(rng: &mut Rng) -> f32 {
    match rng.next() % 16 {
        0 => 0.0,
        1 => 1.0e-8,
        2 => 30.0,
        3 => 104.0,
        4 => -0.5,
        _ => 2.0 * rng.unit(),
    }
}

/// beta in [0, 1] with extremes 0, 1, tiny and just below 1.
fn beta_value(rng: &mut Rng) -> f32 {
    match rng.next() % 12 {
        0 => 0.0,
        1 => 1.0,
        2 => 1.0e-7,
        3 => 0.999_999_9,
        _ => rng.unit(),
    }
}

struct Inputs {
    p: GatedDeltaNetParams,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

fn inputs(n_k_heads: u32, n_v_heads: u32, n_tokens: u32, n_seqs: u32, seed: u64) -> Inputs {
    let p = GatedDeltaNetParams {
        d_k: D,
        d_v: D,
        n_k_heads,
        n_v_heads,
        n_tokens,
        n_seqs,
    };
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let tokens = (n_tokens * n_seqs) as usize;
    let qk = (D * n_k_heads) as usize * tokens;
    let v = (D * n_v_heads) as usize * tokens;
    let scalars = n_v_heads as usize * tokens;
    let state = (D * D * n_v_heads * n_seqs) as usize;
    // q/k look like l2-normalised rows (|x_i| ≲ 1/√D·few).
    let kq_scale = 3.0 / (D as f32).sqrt();
    Inputs {
        p,
        q: (0..qk).map(|_| rng.signed(kq_scale)).collect(),
        k: (0..qk).map(|_| rng.signed(kq_scale)).collect(),
        v: (0..v).map(|_| rng.signed(2.0)).collect(),
        g: (0..scalars).map(|_| g_value(&mut rng)).collect(),
        beta: (0..scalars).map(|_| beta_value(&mut rng)).collect(),
        state: (0..state).map(|_| rng.signed(0.5)).collect(),
    }
}

/// A kernel argument: the parent allocation and the view bound to the kernel.
struct Arg {
    parent: MlxBuffer,
    view: MlxBuffer,
}

/// `pad_elems` = 0 binds the whole buffer; otherwise the view starts
/// `pad_elems` f32 into a parent with `pad_elems + 7` elements of padding.
fn arg(device: &MlxDevice, data: Option<&[f32]>, n: usize, pad_elems: usize) -> Arg {
    let tail = if pad_elems == 0 { 0 } else { 7 };
    let total = pad_elems + n + tail;
    let mut parent = device
        .alloc_buffer(total * 4, DType::F32, vec![total])
        .expect("alloc");
    {
        let all = parent.as_mut_slice::<f32>().expect("parent slice");
        all.fill(f32::from_bits(SENTINEL));
        if let Some(data) = data {
            all[pad_elems..pad_elems + n].copy_from_slice(data);
        }
    }
    let view = if pad_elems == 0 {
        parent.clone()
    } else {
        parent.slice_view((pad_elems * 4) as u64, n)
    };
    Arg { parent, view }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Legacy,
    New,
}

/// Run one kernel; returns the full parent contents of output and state_out.
fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    inp: &Inputs,
    kernel: Kernel,
    folded_scale: bool,
    offsets: bool,
) -> (Vec<f32>, Vec<f32>) {
    let p = inp.p;
    // Distinct odd element offsets (4-byte aligned, not 16-byte aligned).
    let pad = |i: usize| if offsets { 1 + 6 * i } else { 0 };
    let q_data: Vec<f32> = if folded_scale {
        inp.q.clone()
    } else {
        let s = 1.0 / (p.d_k as f32).sqrt();
        inp.q.iter().map(|x| x * s).collect()
    };
    let out_n = inp.v.len();
    let q = arg(device, Some(&q_data), q_data.len(), pad(0));
    let k = arg(device, Some(&inp.k), inp.k.len(), pad(1));
    let v = arg(device, Some(&inp.v), inp.v.len(), pad(2));
    let g = arg(device, Some(&inp.g), inp.g.len(), pad(3));
    let beta = arg(device, Some(&inp.beta), inp.beta.len(), pad(4));
    let state_in = arg(device, Some(&inp.state), inp.state.len(), pad(5));
    let output = arg(device, None, out_n, pad(6));
    let state_out = arg(device, None, inp.state.len(), pad(7));
    let params = if folded_scale {
        build_gated_delta_net_params_with_q_scale(device, p, 1.0 / (p.d_k as f32).sqrt())
    } else {
        build_gated_delta_net_params(device, p)
    }
    .expect("params");

    let dispatch = match kernel {
        Kernel::Legacy => dispatch_gated_delta_net_decode_legacy,
        Kernel::New => dispatch_gated_delta_net_decode,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch(
        &mut encoder,
        registry,
        device.metal_device(),
        &q.view,
        &k.view,
        &v.view,
        &g.view,
        &beta.view,
        &state_in.view,
        &output.view,
        &state_out.view,
        &params,
        p,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("execute");
    (
        output.parent.as_slice::<f32>().expect("output").to_vec(),
        state_out
            .parent
            .as_slice::<f32>()
            .expect("state_out")
            .to_vec(),
    )
}

fn assert_bitwise(label: &str, legacy: &[f32], new: &[f32]) {
    assert_eq!(legacy.len(), new.len(), "{label}: length");
    let mismatches: Vec<usize> = legacy
        .iter()
        .zip(new)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .collect();
    if let Some(&first) = mismatches.first() {
        panic!(
            "{label}: {} of {} elements differ; first at {first}: legacy {:#010x} ({}) vs new {:#010x} ({})",
            mismatches.len(),
            legacy.len(),
            legacy[first].to_bits(),
            legacy[first],
            new[first].to_bits(),
            new[first],
        );
    }
}

/// The result must be a real recurrence, not an untouched or all-zero buffer.
fn assert_written(label: &str, values: &[f32]) {
    let written = values.iter().filter(|x| x.to_bits() != SENTINEL).count();
    let nonzero = values
        .iter()
        .filter(|x| x.to_bits() != SENTINEL && **x != 0.0)
        .count();
    assert!(
        nonzero * 2 > written && written > 0,
        "{label}: {written} written, {nonzero} non-zero — degenerate"
    );
}

fn check_heads(n_k_heads: u32, n_v_heads: u32, model: &str) {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    for (case, &n_tokens) in [1u32, 2, 17, 512].iter().enumerate() {
        for &n_seqs in &[1u32, 4] {
            let seed = (u64::from(n_k_heads) << 32)
                | (u64::from(n_v_heads) << 16)
                | ((case as u64) << 4)
                | u64::from(n_seqs);
            let inp = inputs(n_k_heads, n_v_heads, n_tokens, n_seqs, seed);
            for folded_scale in [false, true] {
                for offsets in [false, true] {
                    let label = format!(
                        "{model} n_tokens={n_tokens} n_seqs={n_seqs} folded_scale={folded_scale} \
                         offsets={offsets}"
                    );
                    let (y_old, s_old) = run(
                        &device,
                        &mut registry,
                        &inp,
                        Kernel::Legacy,
                        folded_scale,
                        offsets,
                    );
                    let (y_new, s_new) = run(
                        &device,
                        &mut registry,
                        &inp,
                        Kernel::New,
                        folded_scale,
                        offsets,
                    );
                    assert_written(&format!("{label} legacy y"), &y_old);
                    assert_written(&format!("{label} legacy state"), &s_old);
                    assert_bitwise(&format!("{label} y"), &y_old, &y_new);
                    assert_bitwise(&format!("{label} state"), &s_old, &s_new);
                }
            }
        }
    }
}

#[test]
fn gdn_decode_addressing_bitwise_qwen38_heads() {
    check_heads(16, 48, "qwen3.8");
}

#[test]
fn gdn_decode_addressing_bitwise_qwen36_heads() {
    check_heads(16, 32, "qwen3.6");
}

/// NSG 1 and 2 (D_k 32 / 64) share the rewritten impl; cover them too.
#[test]
fn gdn_decode_addressing_bitwise_small_nsg() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    for &(d, n_k_heads, n_v_heads) in &[(32u32, 2u32, 4u32), (64, 2, 6)] {
        for &(n_tokens, n_seqs) in &[(1u32, 1u32), (17, 4)] {
            let mut inp = inputs(
                n_k_heads,
                n_v_heads,
                n_tokens,
                n_seqs,
                u64::from(d * n_tokens),
            );
            // Re-shape the D=128 generator output to this head dim.
            let tokens = (n_tokens * n_seqs) as usize;
            inp.p.d_k = d;
            inp.p.d_v = d;
            inp.q.truncate((d * n_k_heads) as usize * tokens);
            inp.k.truncate((d * n_k_heads) as usize * tokens);
            inp.v.truncate((d * n_v_heads) as usize * tokens);
            inp.state.truncate((d * d * n_v_heads * n_seqs) as usize);
            for offsets in [false, true] {
                let label = format!("d={d} n_tokens={n_tokens} n_seqs={n_seqs} offsets={offsets}");
                let (y_old, s_old) =
                    run(&device, &mut registry, &inp, Kernel::Legacy, true, offsets);
                let (y_new, s_new) = run(&device, &mut registry, &inp, Kernel::New, true, offsets);
                assert_written(&format!("{label} legacy y"), &y_old);
                assert_bitwise(&format!("{label} y"), &y_old, &y_new);
                assert_bitwise(&format!("{label} state"), &s_old, &s_new);
            }
        }
    }
}
