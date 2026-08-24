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
    dispatch_ssm_conv, dispatch_ssm_conv_with_capture, dispatch_ssm_conv_with_selected_capture,
    SsmConvParams,
};
use mlx_native::{CapturedNode, DType, DispatchKind, KernelRegistry, MlxBuffer, MlxDevice};

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
        &mut enc,
        registry,
        device.metal_device(),
        &x_buf,
        &kw_buf,
        &old_state,
        &new_state_buf,
        &y_buf,
        &params_buf,
        p,
    )
    .expect("dispatch ssm_conv");
    enc.commit_and_wait().expect("commit");

    (
        y_buf.as_slice::<f32>().expect("y read").to_vec(),
        new_state_buf
            .as_slice::<f32>()
            .expect("new_state read")
            .to_vec(),
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
        &mut enc,
        registry,
        device.metal_device(),
        &x_buf,
        &kw_buf,
        &old_state,
        &y_buf,
        &cap_buf,
        &params_buf,
        p,
    )
    .expect("dispatch ssm_conv_with_capture");
    enc.commit_and_wait().expect("commit");

    (
        y_buf.as_slice::<f32>().expect("y read").to_vec(),
        cap_buf.as_slice::<f32>().expect("cap read").to_vec(),
    )
}

fn run_selected_capture(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &[f32],
    kernel_w: &[f32],
    state: &[f32],
    capture_token: u32,
    p: SsmConvParams,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x_buf = upload_f32(device, x);
    let kw_buf = upload_f32(device, kernel_w);
    let old_state = upload_f32(device, state);
    let x_elems = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let capture_elems = (p.n_seqs as usize) * ((p.k_width - 1) as usize) * (p.channels as usize);
    let y_buf = device
        .alloc_buffer(x_elems * 4, DType::F32, vec![x_elems])
        .expect("y alloc");
    let cap_buf = device
        .alloc_buffer(capture_elems * 4, DType::F32, vec![capture_elems])
        .expect("selected cap alloc");
    let state_buf = device
        .alloc_buffer(capture_elems * 4, DType::F32, vec![capture_elems])
        .expect("selected state alloc");
    let params_buf = build_params_buf(device, p);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_ssm_conv_with_selected_capture(
        &mut enc,
        registry,
        device.metal_device(),
        &x_buf,
        &kw_buf,
        &old_state,
        &state_buf,
        &y_buf,
        &cap_buf,
        &params_buf,
        capture_token,
        p,
    )
    .expect("dispatch selected ssm capture");
    enc.commit_and_wait().expect("commit");

    (
        y_buf.as_slice::<f32>().expect("y read").to_vec(),
        state_buf.as_slice::<f32>().expect("state read").to_vec(),
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
            i,
            x,
            y
        );
    }
}

#[test]
fn selected_capture_matches_all_position_capture_for_each_sequence() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 64,
        n_tokens: 7,
        n_seqs: 2,
        k_width: 4,
    };
    let x_n = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let w_n = (p.k_width * p.channels) as usize;
    let s_n = ((p.k_width - 1) * p.channels * p.n_seqs) as usize;
    let mut seed = 0xB0A7_DA7A;
    let x = rand_vec(&mut seed, x_n, 0.1);
    let w = rand_vec(&mut seed, w_n, 0.05);
    let s = rand_vec(&mut seed, s_n, 0.05);
    let (ordinary_y, ordinary_state) = run_legacy(&device, &mut registry, &x, &w, &s, p);
    let (all_y, all_capture) = run_capture(&device, &mut registry, &x, &w, &s, p);
    assert_byte_identical("all-capture output", &all_y, &ordinary_y);

    let per_token = ((p.k_width - 1) * p.channels) as usize;
    let all_seq_stride = p.n_tokens as usize * per_token;
    for capture_token in [0u32, 3, 6] {
        let (selected_y, selected_state, selected_capture) =
            run_selected_capture(&device, &mut registry, &x, &w, &s, capture_token, p);
        assert_byte_identical("selected output", &selected_y, &ordinary_y);
        assert_byte_identical("selected final state", &selected_state, &ordinary_state);
        for seq in 0..p.n_seqs as usize {
            let all_start = seq * all_seq_stride + capture_token as usize * per_token;
            let selected_start = seq * per_token;
            assert_byte_identical(
                &format!("selected token {capture_token} seq {seq}"),
                &selected_capture[selected_start..selected_start + per_token],
                &all_capture[all_start..all_start + per_token],
            );
        }
    }
}

#[test]
fn selected_capture_rejects_invalid_index_and_destination_before_encoding() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 32,
        n_tokens: 3,
        n_seqs: 1,
        k_width: 4,
    };
    let x_elems = (p.channels * p.n_tokens) as usize;
    let state_elems = ((p.k_width - 1) * p.channels) as usize;
    let x = upload_f32(&device, &vec![0.1; x_elems]);
    let kernel = upload_f32(&device, &vec![0.2; (p.k_width * p.channels) as usize]);
    let old_state = upload_f32(&device, &vec![0.0; state_elems]);
    let y = device
        .alloc_buffer(x_elems * 4, DType::F32, vec![x_elems])
        .expect("y");
    let new_state = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("new state");
    let capture = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("capture");
    let short_capture = device
        .alloc_buffer((state_elems - 1) * 4, DType::F32, vec![state_elems - 1])
        .expect("short capture");
    let params = build_params_buf(&device, p);
    let mut encoder = device.command_encoder().expect("encoder");
    let invalid_index = dispatch_ssm_conv_with_selected_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x,
        &kernel,
        &old_state,
        &new_state,
        &y,
        &capture,
        &params,
        p.n_tokens,
        p,
    )
    .expect_err("capture token at n_tokens must fail");
    assert!(invalid_index.to_string().contains("capture_token"));
    let invalid_shape = dispatch_ssm_conv_with_selected_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x,
        &kernel,
        &old_state,
        &new_state,
        &y,
        &short_capture,
        &params,
        1,
        p,
    )
    .expect_err("short selected capture must fail");
    assert!(invalid_shape.to_string().contains("must be F32"));
}

#[test]
fn selected_capture_rejects_bf16_and_aliases_before_encoding() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 32,
        n_tokens: 3,
        n_seqs: 1,
        k_width: 4,
    };
    let x_elems = (p.channels * p.n_tokens) as usize;
    let state_elems = ((p.k_width - 1) * p.channels) as usize;
    let kernel_elems = (p.k_width * p.channels) as usize;
    let x = upload_f32(&device, &vec![0.1; x_elems]);
    let kernel = upload_f32(&device, &vec![0.2; kernel_elems]);
    let old_state = upload_f32(&device, &vec![0.0; state_elems]);
    let y = device
        .alloc_buffer(x_elems * 4, DType::F32, vec![x_elems])
        .expect("y");
    let new_state = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("new state");
    let capture = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("capture");
    let params = build_params_buf(&device, p);

    let bf16_x = device
        .alloc_buffer(x_elems * 2, DType::BF16, vec![x_elems])
        .expect("bf16 x");
    let bf16_kernel = device
        .alloc_buffer(kernel_elems * 2, DType::BF16, vec![kernel_elems])
        .expect("bf16 kernel");
    let bf16_old_state = device
        .alloc_buffer(state_elems * 2, DType::BF16, vec![state_elems])
        .expect("bf16 old state");
    let bf16_new_state = device
        .alloc_buffer(state_elems * 2, DType::BF16, vec![state_elems])
        .expect("bf16 new state");
    let bf16_y = device
        .alloc_buffer(x_elems * 2, DType::BF16, vec![x_elems])
        .expect("bf16 y");
    let mut encoder = device.command_encoder().expect("bf16 encoder");
    encoder.start_capture();
    let error = dispatch_ssm_conv_with_selected_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &bf16_x,
        &bf16_kernel,
        &bf16_old_state,
        &bf16_new_state,
        &bf16_y,
        &capture,
        &params,
        1,
        p,
    )
    .expect_err("BF16 selected capture must fail");
    assert!(error.to_string().contains("only F32"));
    assert!(encoder.take_capture().expect("BF16 capture").is_empty());

    let mut encoder = device.command_encoder().expect("state alias encoder");
    encoder.start_capture();
    let error = dispatch_ssm_conv_with_selected_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x,
        &kernel,
        &old_state,
        &old_state,
        &y,
        &capture,
        &params,
        1,
        p,
    )
    .expect_err("old/new state alias must fail");
    assert!(error.to_string().contains("must not overlap"));
    assert!(encoder
        .take_capture()
        .expect("state alias capture")
        .is_empty());

    let mut encoder = device.command_encoder().expect("x/y alias encoder");
    encoder.start_capture();
    let error = dispatch_ssm_conv_with_selected_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x,
        &kernel,
        &old_state,
        &new_state,
        &x,
        &capture,
        &params,
        1,
        p,
    )
    .expect_err("x/y alias must fail");
    assert!(error.to_string().contains("must not overlap"));
    assert!(encoder
        .take_capture()
        .expect("x/y alias capture")
        .is_empty());

    let parent = device
        .alloc_buffer((x_elems + 1) * 4, DType::F32, vec![x_elems + 1])
        .expect("overlap parent");
    let x_view = parent.slice_view(0, x_elems);
    let y_view = parent.slice_view(4, x_elems);
    let mut encoder = device.command_encoder().expect("partial alias encoder");
    encoder.start_capture();
    let error = dispatch_ssm_conv_with_selected_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x_view,
        &kernel,
        &old_state,
        &new_state,
        &y_view,
        &capture,
        &params,
        1,
        p,
    )
    .expect_err("partially overlapping x/y views must fail");
    assert!(error.to_string().contains("must not overlap"));
    assert!(encoder
        .take_capture()
        .expect("partial alias capture")
        .is_empty());
}

#[test]
fn all_position_capture_rejects_x_y_alias_before_encoding() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 32,
        n_tokens: 3,
        n_seqs: 1,
        k_width: 4,
    };
    let x_elems = (p.channels * p.n_tokens) as usize;
    let state_elems = ((p.k_width - 1) * p.channels) as usize;
    let capture_elems = p.n_tokens as usize * state_elems;
    let x = upload_f32(&device, &vec![0.1; x_elems]);
    let kernel = upload_f32(&device, &vec![0.2; (p.k_width * p.channels) as usize]);
    let old_state = upload_f32(&device, &vec![0.0; state_elems]);
    let capture = device
        .alloc_buffer(capture_elems * 4, DType::F32, vec![capture_elems])
        .expect("capture");
    let params = build_params_buf(&device, p);
    let mut encoder = device.command_encoder().expect("encoder");
    encoder.start_capture();
    let error = dispatch_ssm_conv_with_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x,
        &kernel,
        &old_state,
        &x,
        &capture,
        &params,
        p,
    )
    .expect_err("x/y alias must fail");
    assert!(error.to_string().contains("must not overlap"));
    assert!(encoder.take_capture().expect("capture").is_empty());
}

#[test]
fn selected_capture_records_a_raw_thread_dispatch() {
    let (device, mut registry) = setup();
    let p = SsmConvParams {
        channels: 32,
        n_tokens: 3,
        n_seqs: 2,
        k_width: 4,
    };
    let x_elems = (p.channels * p.n_tokens * p.n_seqs) as usize;
    let state_elems = ((p.k_width - 1) * p.channels * p.n_seqs) as usize;
    let x = upload_f32(&device, &vec![0.1; x_elems]);
    let kernel = upload_f32(&device, &vec![0.2; (p.k_width * p.channels) as usize]);
    let old_state = upload_f32(&device, &vec![0.0; state_elems]);
    let y = device
        .alloc_buffer(x_elems * 4, DType::F32, vec![x_elems])
        .expect("y");
    let new_state = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("new state");
    let capture = device
        .alloc_buffer(state_elems * 4, DType::F32, vec![state_elems])
        .expect("capture");
    let params = build_params_buf(&device, p);
    let mut encoder = device.command_encoder().expect("encoder");
    encoder.start_capture();
    dispatch_ssm_conv_with_selected_capture(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x,
        &kernel,
        &old_state,
        &new_state,
        &y,
        &capture,
        &params,
        1,
        p,
    )
    .expect("encode selected capture");
    let nodes = encoder.take_capture().expect("captured dispatch");
    let [CapturedNode::Dispatch {
        threads_per_grid,
        threads_per_threadgroup,
        dispatch_kind,
        ..
    }] = nodes.as_slice()
    else {
        panic!("expected exactly one selected-capture dispatch")
    };
    assert!(matches!(dispatch_kind, DispatchKind::Threads));
    assert_eq!(
        (
            threads_per_grid.width,
            threads_per_grid.height,
            threads_per_grid.depth
        ),
        (p.channels as u64, p.n_tokens as u64, p.n_seqs as u64)
    );
    assert!(
        threads_per_threadgroup.width
            * threads_per_threadgroup.height
            * threads_per_threadgroup.depth
            <= 256
    );
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
            let cap_idx = i * channels + c; // [K-1, channels] with channels stride 1
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

/// ADR-034 task #90 Step 5 (2026-05-21) — STRONG parity test for the
/// INTERMEDIATE captures: capture[t] for any t ∈ [0, n_tokens) must
/// equal new_state from a separate dispatch_ssm_conv call with
/// n_tokens = t+1. This is the ground truth for K=N partial-reject
/// rollback (caller selects capture[accepted_idx]).
///
/// Without this test, only capture[last_t] was verified.
#[test]
fn capture_intermediate_t_matches_truncated_dispatch() {
    let (device, mut registry) = setup();
    let n_full = 4u32;
    let p_full = SsmConvParams {
        channels: 64,
        n_tokens: n_full,
        n_seqs: 1,
        k_width: 4,
    };
    let x_n = (p_full.channels * n_full * p_full.n_seqs) as usize;
    let w_n = (p_full.k_width * p_full.channels) as usize;
    let s_n = ((p_full.k_width - 1) * p_full.channels * p_full.n_seqs) as usize;
    let mut seed = 0x517E;
    let x = rand_vec(&mut seed, x_n, 0.1);
    let w = rand_vec(&mut seed, w_n, 0.05);
    let s = rand_vec(&mut seed, s_n, 0.05);

    let (_cap_y, cap_capture) = run_capture(&device, &mut registry, &x, &w, &s, p_full);

    let per_t = ((p_full.k_width - 1) * p_full.channels) as usize;
    let k_minus1 = (p_full.k_width - 1) as usize;
    let channels = p_full.channels as usize;

    // For each intermediate t in 0..n_full-1, run the legacy
    // dispatch_ssm_conv with n_tokens = t+1 (truncated x) and verify
    // that its new_state byte-matches capture[t] (after re-indexing).
    for t in 0..(n_full as usize - 1) {
        let truncated_x_n = (p_full.channels as usize) * (t + 1) * (p_full.n_seqs as usize);
        let truncated_x = x[..truncated_x_n].to_vec();
        let p_trunc = SsmConvParams {
            channels: p_full.channels,
            n_tokens: (t + 1) as u32,
            n_seqs: p_full.n_seqs,
            k_width: p_full.k_width,
        };
        let (_y_trunc, state_trunc) =
            run_legacy(&device, &mut registry, &truncated_x, &w, &s, p_trunc);

        // capture[t] layout: per_t elements in [K-1, channels] order
        // (channels innermost). Re-index to compare with state_trunc
        // which is [channels, K-1] (K-1 innermost).
        let cap_t_offset = t * per_t;
        let cap_t = &cap_capture[cap_t_offset..cap_t_offset + per_t];
        for i in 0..k_minus1 {
            for c in 0..channels {
                let cap_idx = i * channels + c;
                let legacy_idx = c * k_minus1 + i;
                assert_eq!(
                    cap_t[cap_idx].to_bits(),
                    state_trunc[legacy_idx].to_bits(),
                    "capture[t={t}] vs trunc(n_tokens={trunc_n}) at i={i} c={c}",
                    trunc_n = t + 1,
                );
            }
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
