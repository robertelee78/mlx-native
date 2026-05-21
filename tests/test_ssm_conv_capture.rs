//! ADR-034 task #90 Step 4b (2026-05-21) — parity tests for the
//! per-position conv-state capture variant of `dispatch_ssm_conv`.
//!
//! Acceptance criteria:
//! 1. **Output parity**: capture variant's `y` byte-identical to
//!    non-capture `dispatch_ssm_conv` for the same inputs.
//! 2. **state_capture[..., n_tokens-1, ...] == new_state**: the final
//!    per-position slice matches the new_state produced by
//!    `ssm_conv_state_update_f32` (the determinism contract in the
//!    `dispatch_ssm_conv_with_capture` docstring).
//! 3. **Pipeline loads**: kernel source compiles + dispatches cleanly.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::ssm_conv::{
    dispatch_ssm_conv, dispatch_ssm_conv_with_capture, SsmConvParams,
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

fn build_params_buf(device: &MlxDevice, p: SsmConvParams) -> MlxBuffer {
    let raw = [p.channels, p.n_tokens, p.n_seqs, p.k_width];
    let mut buf = device
        .alloc_buffer(16, DType::U32, vec![4])
        .expect("params buf");
    let dst = buf.as_mut_slice::<u32>().expect("params mut");
    dst.copy_from_slice(&raw);
    buf
}

fn run_legacy(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &[f32],
    kernel_w: &[f32],
    state: &[f32],
    p: SsmConvParams,
) -> (Vec<f32>, Vec<f32>) {
    let x_buf = upload_f32(device, x);
    let kw_buf = upload_f32(device, kernel_w);
    let old_state = upload_f32(device, state);
    let x_elems = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let s_elems = ((p.k_width - 1) * p.channels * p.n_seqs) as usize;
    let y_buf = device
        .alloc_buffer(x_elems * 4, DType::F32, vec![x_elems])
        .expect("y alloc");
    let new_state_buf = device
        .alloc_buffer(s_elems * 4, DType::F32, vec![s_elems])
        .expect("new state alloc");
    let params_buf = build_params_buf(device, p);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_ssm_conv(
        &mut enc, registry, device.metal_device(),
        &x_buf, &kw_buf, &old_state, &new_state_buf, &y_buf,
        &params_buf, p,
    )
    .expect("dispatch ssm_conv");
    enc.commit_and_wait().expect("commit");

    (
        y_buf.as_slice::<f32>().expect("y read").to_vec(),
        new_state_buf.as_slice::<f32>().expect("new_state read").to_vec(),
    )
}

fn run_capture(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &[f32],
    kernel_w: &[f32],
    state: &[f32],
    p: SsmConvParams,
) -> (Vec<f32>, Vec<f32>) {
    let x_buf = upload_f32(device, x);
    let kw_buf = upload_f32(device, kernel_w);
    let old_state = upload_f32(device, state);
    let x_elems = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let capture_elems = (p.n_seqs as usize)
        * (p.n_tokens as usize)
        * ((p.k_width - 1) as usize)
        * (p.channels as usize);
    let y_buf = device
        .alloc_buffer(x_elems * 4, DType::F32, vec![x_elems])
        .expect("y alloc");
    let cap_buf = device
        .alloc_buffer(capture_elems * 4, DType::F32, vec![capture_elems])
        .expect("cap alloc");
    let params_buf = build_params_buf(device, p);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_ssm_conv_with_capture(
        &mut enc, registry, device.metal_device(),
        &x_buf, &kw_buf, &old_state, &y_buf, &cap_buf,
        &params_buf, p,
    )
    .expect("dispatch ssm_conv_with_capture");
    enc.commit_and_wait().expect("commit");

    (
        y_buf.as_slice::<f32>().expect("y read").to_vec(),
        cap_buf.as_slice::<f32>().expect("cap read").to_vec(),
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

/// AC#1: y output byte-identical between capture and non-capture at
/// production Qwen 3.5/3.6 shape (K=4, channels=8192, n_tokens=4 for K=N spec).
#[test]
fn capture_y_byte_identical_qwen35_shape() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 8192,
        n_tokens: 4,
        n_seqs: 1,
        k_width: 4,
    };
    let x_n = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let w_n = (p.k_width * p.channels) as usize;
    let s_n = ((p.k_width - 1) * p.channels * p.n_seqs) as usize;
    let mut seed = 0xCAFE;
    let x = rand_vec(&mut seed, x_n, 0.1);
    let w = rand_vec(&mut seed, w_n, 0.05);
    let s = rand_vec(&mut seed, s_n, 0.05);

    let (legacy_y, legacy_state) = run_legacy(&device, &mut registry, &x, &w, &s, p);
    let (cap_y, cap_capture) = run_capture(&device, &mut registry, &x, &w, &s, p);

    assert_byte_identical("qwen35 y", &cap_y, &legacy_y);

    // AC#2: capture[..., n_tokens-1, ...] == new_state. Capture buffer
    // layout: [n_seqs, n_tokens, K-1, channels]. The last token slice
    // starts at (s=0, t=n_tokens-1) → offset (n_tokens-1) * (K-1) *
    // channels for n_seqs=1.
    let per_t = ((p.k_width - 1) * p.channels) as usize;
    let last_t_offset = ((p.n_tokens - 1) as usize) * per_t;
    let cap_last_t = &cap_capture[last_t_offset..last_t_offset + per_t];
    // legacy_state layout: [n_seqs, channels, K-1] (channels-major, K-1
    // stride 1). cap_last_t layout: [K-1, channels] (channels innermost).
    // Need to compare them with a re-indexed comparison.
    let k_minus1 = (p.k_width - 1) as usize;
    let channels = p.channels as usize;
    for i in 0..k_minus1 {
        for c in 0..channels {
            let legacy_idx = c * k_minus1 + i; // [channels, K-1] with K-1 stride 1
            let cap_idx = i * channels + c;    // [K-1, channels] with channels stride 1
            assert_eq!(
                legacy_state[legacy_idx].to_bits(),
                cap_last_t[cap_idx].to_bits(),
                "qwen35 last-t capture vs legacy state at i={i} c={c}",
            );
        }
    }
}

/// AC#1+2 at smaller shape (channels=32, n_tokens=4, K=4) for fast iteration.
#[test]
fn capture_y_byte_identical_small_shape() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 32,
        n_tokens: 4,
        n_seqs: 1,
        k_width: 4,
    };
    let x_n = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let w_n = (p.k_width * p.channels) as usize;
    let s_n = ((p.k_width - 1) * p.channels * p.n_seqs) as usize;
    let mut seed = 0x1234;
    let x = rand_vec(&mut seed, x_n, 0.1);
    let w = rand_vec(&mut seed, w_n, 0.05);
    let s = rand_vec(&mut seed, s_n, 0.05);

    let (legacy_y, legacy_state) = run_legacy(&device, &mut registry, &x, &w, &s, p);
    let (cap_y, cap_capture) = run_capture(&device, &mut registry, &x, &w, &s, p);

    assert_byte_identical("small y", &cap_y, &legacy_y);

    // capture[last_t] vs legacy_state — channel/k re-index.
    let per_t = ((p.k_width - 1) * p.channels) as usize;
    let last_t_offset = ((p.n_tokens - 1) as usize) * per_t;
    let cap_last_t = &cap_capture[last_t_offset..last_t_offset + per_t];
    let k_minus1 = (p.k_width - 1) as usize;
    let channels = p.channels as usize;
    for i in 0..k_minus1 {
        for c in 0..channels {
            let legacy_idx = c * k_minus1 + i;
            let cap_idx = i * channels + c;
            assert_eq!(
                legacy_state[legacy_idx].to_bits(),
                cap_last_t[cap_idx].to_bits(),
                "small last-t capture vs legacy state at i={i} c={c}",
            );
        }
    }
}

/// AC#3: capture[t=0] DIFFERS from capture[last_t] for n_tokens > 1 —
/// confirms the recurrence is non-degenerate and per-position writes
/// are happening at every t (not just the last).
#[test]
fn capture_intermediate_positions_differ() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 64,
        n_tokens: 4,
        n_seqs: 1,
        k_width: 4,
    };
    let x_n = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let w_n = (p.k_width * p.channels) as usize;
    let s_n = ((p.k_width - 1) * p.channels * p.n_seqs) as usize;
    let mut seed = 0xDEAD;
    let x = rand_vec(&mut seed, x_n, 0.5); // larger scale for variety
    let w = rand_vec(&mut seed, w_n, 0.5);
    let s = rand_vec(&mut seed, s_n, 0.5);

    let (_cap_y, cap_capture) = run_capture(&device, &mut registry, &x, &w, &s, p);

    let per_t = ((p.k_width - 1) * p.channels) as usize;
    let first_t = &cap_capture[0..per_t];
    let last_t_offset = ((p.n_tokens - 1) as usize) * per_t;
    let last_t = &cap_capture[last_t_offset..last_t_offset + per_t];

    // At least one entry must differ between first and last token's state.
    let differs = first_t
        .iter()
        .zip(last_t.iter())
        .any(|(&a, &b)| a.to_bits() != b.to_bits());
    assert!(
        differs,
        "capture[0] unexpectedly equals capture[last] — recurrence may be degenerate"
    );
}
