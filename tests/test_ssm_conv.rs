//! Tests for the SSM causal depthwise conv1d + SiLU kernel (ADR-013 Decision 7).
//!
//! Spec (verbatim from the ADR):
//!
//! ```text
//! ssm_conv(x, kernel, state) -> (y, new_state)
//!   x:        [channels, n_tokens, n_seqs]
//!   kernel:   [K, channels]
//!   state:    [K-1, channels, n_seqs]
//!   extended(c, t_ext, s) = state(t_ext, c, s)        if t_ext < K-1
//!                           x(c, t_ext - (K-1), s)    otherwise
//!   y(c, t, s) = silu( sum_{k=0..K} kernel(k, c) * extended(c, t + k, s) )
//!   new_state(i, c, s) = extended(c, n_tokens + i, s) for i in 0..K-1
//! ```
//!
//! Acceptance criteria from ADR-013 Decision 7:
//! 1. Shader + op wrapper exist (covered by the fact these tests compile).
//! 2. Spec-driven test: 1-seq, 4-channel, 6-token input with random kernel;
//!    hand-compute causal conv output with SiLU; match to 1e-4 (F32).
//! 3. Ring-buffer correctness test: run conv on tokens [0..4], save state,
//!    run conv on tokens [4..8] using saved state, compare against monolithic
//!    run on [0..8]; outputs byte-identical (within fp rounding).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::ssm_conv::SsmConvParams;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

fn alloc_params(device: &MlxDevice, p: SsmConvParams) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(4 * 4, DType::U32, vec![4])
        .expect("alloc params");
    {
        let s = buf.as_mut_slice::<u32>().expect("mut params");
        s[0] = p.channels;
        s[1] = p.n_tokens;
        s[2] = p.n_seqs;
        s[3] = p.k_width;
    }
    buf
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Pure-Rust scalar CPU reference implementation of the spec.
/// Returns (y, new_state) in the layouts used by the Metal kernel.
fn cpu_reference(
    x: &[f32],
    kernel_w: &[f32],
    state: &[f32],
    p: SsmConvParams,
) -> (Vec<f32>, Vec<f32>) {
    let c_n = p.channels as usize;
    let t_n = p.n_tokens as usize;
    let s_n = p.n_seqs as usize;
    let k = p.k_width as usize;
    let km1 = k - 1;

    let x_seq_stride = t_n * c_n;
    let s_seq_stride = km1 * c_n;

    // Virtual extended access:
    //   extended(c, t_ext, s)
    //     = state[s * s_seq_stride + c * km1 + t_ext]       if t_ext < km1
    //     = x[s * x_seq_stride + (t_ext - km1) * c_n + c]   otherwise
    let extended = |c: usize, t_ext: usize, s: usize| -> f32 {
        if t_ext < km1 {
            state[s * s_seq_stride + c * km1 + t_ext]
        } else {
            x[s * x_seq_stride + (t_ext - km1) * c_n + c]
        }
    };

    let mut y = vec![0.0f32; s_n * x_seq_stride];
    for s in 0..s_n {
        for t in 0..t_n {
            for c in 0..c_n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += kernel_w[c * k + kk] * extended(c, t + kk, s);
                }
                y[s * x_seq_stride + t * c_n + c] = silu(sum);
            }
        }
    }

    let mut new_state = vec![0.0f32; s_n * s_seq_stride];
    for s in 0..s_n {
        for c in 0..c_n {
            for i in 0..km1 {
                new_state[s * s_seq_stride + c * km1 + i] = extended(c, t_n + i, s);
            }
        }
    }

    (y, new_state)
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

// ==================================================================
// Spec-driven: 1-seq, 4-channel, 6-token, K=4, random-ish kernel/state
// ==================================================================

#[test]
fn test_ssm_conv_spec_driven_1seq_4ch_6tok() {
    let (device, mut registry) = setup();
    let params = SsmConvParams {
        channels: 4,
        n_tokens: 6,
        n_seqs: 1,
        k_width: 4,
    };

    // Deterministic pseudo-random fill. Small magnitudes keep SiLU in the
    // numerically interesting range (near x/2 for small x, near x for large).
    let mut seed = 0x4242u32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let v = (seed as i32 as f32) / (i32::MAX as f32);
        v * 0.8 // [-0.8, 0.8]
    };

    let n_x = (params.channels * params.n_tokens * params.n_seqs) as usize;
    let n_w = (params.k_width * params.channels) as usize;
    let n_state = ((params.k_width - 1) * params.channels * params.n_seqs) as usize;

    let x_data: Vec<f32> = (0..n_x).map(|_| rand()).collect();
    let w_data: Vec<f32> = (0..n_w).map(|_| rand()).collect();
    let state_data: Vec<f32> = (0..n_state).map(|_| rand()).collect();

    let (y_ref, state_ref) = cpu_reference(&x_data, &w_data, &state_data, params);

    let x_buf = upload_f32(&device, &x_data);
    let w_buf = upload_f32(&device, &w_data);
    let old_state_buf = upload_f32(&device, &state_data);
    let new_state_buf = device
        .alloc_buffer(n_state * 4, DType::F32, vec![n_state])
        .expect("new_state");
    let y_buf = device
        .alloc_buffer(n_x * 4, DType::F32, vec![n_x])
        .expect("y");
    let p_buf = alloc_params(&device, params);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &w_buf,
        &old_state_buf,
        &new_state_buf,
        &y_buf,
        &p_buf,
        params,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let y_got: &[f32] = y_buf.as_slice().expect("read y");
    let state_got: &[f32] = new_state_buf.as_slice().expect("read state");

    for (i, (&got, &exp)) in y_got.iter().zip(y_ref.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-4,
            "y mismatch at {}: got {}, expected {}, diff {}",
            i, got, exp, diff
        );
    }
    for (i, (&got, &exp)) in state_got.iter().zip(state_ref.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-5,
            "state mismatch at {}: got {}, expected {}, diff {}",
            i, got, exp, diff
        );
    }
}

// ==================================================================
// Hand-computed tiny example
// ==================================================================

/// Simplest exhaustive check: 1-seq, 1-channel, 2-token, K=4.
///
/// state = [s0, s1, s2]
/// kernel = [k0, k1, k2, k3]
/// x = [x0, x1]
///
/// extended = [s0, s1, s2, x0, x1]
/// y[0] = silu(k0*s0 + k1*s1 + k2*s2 + k3*x0)
/// y[1] = silu(k0*s1 + k1*s2 + k2*x0 + k3*x1)
/// new_state = extended[2..5] = [s2, x0, x1]
#[test]
fn test_ssm_conv_tiny_hand_computed() {
    let (device, mut registry) = setup();
    let params = SsmConvParams {
        channels: 1,
        n_tokens: 2,
        n_seqs: 1,
        k_width: 4,
    };

    let s0 = 0.1f32;
    let s1 = 0.2f32;
    let s2 = 0.3f32;
    let x0 = 0.4f32;
    let x1 = 0.5f32;
    let k0 = 0.5f32;
    let k1 = -0.25f32;
    let k2 = 0.125f32;
    let k3 = 0.1f32;

    let x_data = [x0, x1];
    let w_data = [k0, k1, k2, k3];
    let state_data = [s0, s1, s2];

    let expected_y0 = silu(k0 * s0 + k1 * s1 + k2 * s2 + k3 * x0);
    let expected_y1 = silu(k0 * s1 + k1 * s2 + k2 * x0 + k3 * x1);
    let expected_state = [s2, x0, x1];

    let x_buf = upload_f32(&device, &x_data);
    let w_buf = upload_f32(&device, &w_data);
    let old_state_buf = upload_f32(&device, &state_data);
    let new_state_buf = device
        .alloc_buffer(12, DType::F32, vec![3])
        .expect("new_state");
    let y_buf = device.alloc_buffer(8, DType::F32, vec![2]).expect("y");
    let p_buf = alloc_params(&device, params);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &w_buf,
        &old_state_buf,
        &new_state_buf,
        &y_buf,
        &p_buf,
        params,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let y: &[f32] = y_buf.as_slice().expect("read");
    assert!(
        (y[0] - expected_y0).abs() < 1e-5,
        "y[0]: got {}, expected {}",
        y[0], expected_y0
    );
    assert!(
        (y[1] - expected_y1).abs() < 1e-5,
        "y[1]: got {}, expected {}",
        y[1], expected_y1
    );

    let st: &[f32] = new_state_buf.as_slice().expect("read state");
    for i in 0..3 {
        assert!(
            (st[i] - expected_state[i]).abs() < 1e-6,
            "state[{}]: got {}, expected {}",
            i, st[i], expected_state[i]
        );
    }
}

// ==================================================================
// Ring-buffer correctness (ADR acceptance criterion #3)
// ==================================================================

/// Split one logical 8-token sequence into two 4-token chunks. The second
/// chunk's conv state is the first chunk's `new_state`. Monolithic vs.
/// chunked should produce identical outputs.
#[test]
fn test_ssm_conv_ring_buffer_chunk_equivalence() {
    let (device, mut registry) = setup();
    let channels = 4u32;
    let n_seqs = 1u32;
    let k = 4u32;
    let total_tokens = 8u32;

    // Pseudo-random fixture.
    let mut seed = 0xBEEFu32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32) * 0.5
    };

    let n_x_total = (channels * total_tokens * n_seqs) as usize;
    let n_w = (k * channels) as usize;
    let n_state = ((k - 1) * channels * n_seqs) as usize;

    let x_full: Vec<f32> = (0..n_x_total).map(|_| rand()).collect();
    let w_data: Vec<f32> = (0..n_w).map(|_| rand()).collect();
    let state0_data: Vec<f32> = (0..n_state).map(|_| rand()).collect();

    // --- Monolithic run: all 8 tokens in one dispatch ---
    let params_full = SsmConvParams {
        channels,
        n_tokens: total_tokens,
        n_seqs,
        k_width: k,
    };
    let x_buf = upload_f32(&device, &x_full);
    let w_buf = upload_f32(&device, &w_data);
    let old_state_buf = upload_f32(&device, &state0_data);
    let new_state_full_buf = device
        .alloc_buffer(n_state * 4, DType::F32, vec![n_state])
        .expect("new_state full");
    let y_full_buf = device
        .alloc_buffer(n_x_total * 4, DType::F32, vec![n_x_total])
        .expect("y full");
    let p_full = alloc_params(&device, params_full);
    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &w_buf,
        &old_state_buf,
        &new_state_full_buf,
        &y_full_buf,
        &p_full,
        params_full,
    )
    .expect("full dispatch");
    enc.commit_and_wait().expect("commit");
    let y_full: Vec<f32> = y_full_buf.as_slice::<f32>().expect("read").to_vec();

    // --- Chunked run: 4 + 4 tokens ---
    let c_n = channels as usize;
    let chunk_tokens = 4u32;
    let chunk_n_x = (channels * chunk_tokens * n_seqs) as usize;
    let x_chunk0 = &x_full[0..chunk_n_x];
    let x_chunk1 = &x_full[chunk_n_x..2 * chunk_n_x];

    let params_chunk = SsmConvParams {
        channels,
        n_tokens: chunk_tokens,
        n_seqs,
        k_width: k,
    };
    let p_chunk = alloc_params(&device, params_chunk);

    // First chunk.
    let x0_buf = upload_f32(&device, x_chunk0);
    let state0_buf = upload_f32(&device, &state0_data);
    let state_after0_buf = device
        .alloc_buffer(n_state * 4, DType::F32, vec![n_state])
        .expect("state0 out");
    let y0_buf = device
        .alloc_buffer(chunk_n_x * 4, DType::F32, vec![chunk_n_x])
        .expect("y0");
    let mut enc = device.command_encoder().expect("enc0");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &x0_buf,
        &w_buf,
        &state0_buf,
        &state_after0_buf,
        &y0_buf,
        &p_chunk,
        params_chunk,
    )
    .expect("chunk0 dispatch");
    enc.commit_and_wait().expect("commit0");
    let y0: Vec<f32> = y0_buf.as_slice::<f32>().expect("y0 read").to_vec();

    // Second chunk uses new_state from chunk 0.
    let x1_buf = upload_f32(&device, x_chunk1);
    let state_after1_buf = device
        .alloc_buffer(n_state * 4, DType::F32, vec![n_state])
        .expect("state1 out");
    let y1_buf = device
        .alloc_buffer(chunk_n_x * 4, DType::F32, vec![chunk_n_x])
        .expect("y1");
    let mut enc = device.command_encoder().expect("enc1");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &x1_buf,
        &w_buf,
        &state_after0_buf,
        &state_after1_buf,
        &y1_buf,
        &p_chunk,
        params_chunk,
    )
    .expect("chunk1 dispatch");
    enc.commit_and_wait().expect("commit1");
    let y1: Vec<f32> = y1_buf.as_slice::<f32>().expect("y1 read").to_vec();

    // --- Compare chunked concatenation vs monolithic ---
    //
    // x_seq_stride = total_tokens * channels for monolithic;
    // chunk0 covers tokens 0..4; chunk1 covers tokens 4..8.
    //
    // y_full layout: [s=0][t=0..8][c=0..4]
    //   = token-major, with channels contiguous per-token.
    // y0 / y1 have the same layout for their 4 tokens.
    //
    // Compare tokenwise.
    for t in 0..chunk_tokens as usize {
        for c in 0..c_n {
            let full = y_full[t * c_n + c];
            let chunk = y0[t * c_n + c];
            let diff = (full - chunk).abs();
            assert!(
                diff < 1e-6,
                "chunk-vs-full mismatch chunk0 at t={}, c={}: full={} chunk={}",
                t, c, full, chunk
            );
        }
    }
    for t in 0..chunk_tokens as usize {
        for c in 0..c_n {
            let full = y_full[(chunk_tokens as usize + t) * c_n + c];
            let chunk = y1[t * c_n + c];
            let diff = (full - chunk).abs();
            assert!(
                diff < 1e-6,
                "chunk-vs-full mismatch chunk1 at t={}, c={}: full={} chunk={}",
                t, c, full, chunk
            );
        }
    }

    // Final state must also match monolithic new_state.
    let state_chunked: &[f32] = state_after1_buf.as_slice().expect("state1 read");
    let state_mono: &[f32] = new_state_full_buf.as_slice().expect("mono state read");
    for i in 0..n_state {
        let diff = (state_chunked[i] - state_mono[i]).abs();
        assert!(
            diff < 1e-6,
            "final state mismatch at {}: mono={} chunked={}",
            i, state_mono[i], state_chunked[i]
        );
    }
}

// ==================================================================
// Multi-seq independence
// ==================================================================

/// Two sequences with DIFFERENT state should produce different outputs
/// but each seq's output must match its own cpu_reference.
#[test]
fn test_ssm_conv_multi_seq_independence() {
    let (device, mut registry) = setup();
    let params = SsmConvParams {
        channels: 3,
        n_tokens: 5,
        n_seqs: 2,
        k_width: 4,
    };

    let n_x = (params.channels * params.n_tokens * params.n_seqs) as usize;
    let n_w = (params.k_width * params.channels) as usize;
    let n_state = ((params.k_width - 1) * params.channels * params.n_seqs) as usize;

    let mut seed = 0xCAFE_u32;
    let mut rand = || -> f32 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed as i32 as f32) / (i32::MAX as f32) * 0.6
    };

    let x_data: Vec<f32> = (0..n_x).map(|_| rand()).collect();
    let w_data: Vec<f32> = (0..n_w).map(|_| rand()).collect();
    let state_data: Vec<f32> = (0..n_state).map(|_| rand()).collect();

    let (y_ref, _state_ref) = cpu_reference(&x_data, &w_data, &state_data, params);

    let x_buf = upload_f32(&device, &x_data);
    let w_buf = upload_f32(&device, &w_data);
    let old_state_buf = upload_f32(&device, &state_data);
    let new_state_buf = device
        .alloc_buffer(n_state * 4, DType::F32, vec![n_state])
        .expect("new_state");
    let y_buf = device
        .alloc_buffer(n_x * 4, DType::F32, vec![n_x])
        .expect("y");
    let p_buf = alloc_params(&device, params);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &w_buf,
        &old_state_buf,
        &new_state_buf,
        &y_buf,
        &p_buf,
        params,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let y: &[f32] = y_buf.as_slice().expect("read");
    for (i, (&got, &exp)) in y.iter().zip(y_ref.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 1e-4,
            "multi-seq mismatch at {}: got {}, expected {}",
            i, got, exp
        );
    }
}

// ==================================================================
// Decode regime: n_tokens < K - 1 (state-heavy path)
// ==================================================================

/// n_tokens = 1 (single-token decode). The state update reads mostly from
/// old_state; only the last slot gets x[0]. Ensures the branch in the
/// state-update kernel that reads from old_state is exercised.
#[test]
fn test_ssm_conv_decode_single_token() {
    let (device, mut registry) = setup();
    let params = SsmConvParams {
        channels: 2,
        n_tokens: 1,
        n_seqs: 1,
        k_width: 4,
    };

    let x_data = [0.5f32, -0.25f32];
    let w_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let state_data = [
        0.1f32, 0.2, 0.3, // c=0: s0, s1, s2
        -0.1f32, -0.2, -0.3, // c=1
    ];

    let (y_ref, state_ref) = cpu_reference(&x_data, &w_data, &state_data, params);

    let x_buf = upload_f32(&device, &x_data);
    let w_buf = upload_f32(&device, &w_data);
    let old_state_buf = upload_f32(&device, &state_data);
    let new_state_buf = device
        .alloc_buffer(6 * 4, DType::F32, vec![6])
        .expect("new_state");
    let y_buf = device.alloc_buffer(2 * 4, DType::F32, vec![2]).expect("y");
    let p_buf = alloc_params(&device, params);

    let mut encoder = device.command_encoder().expect("enc");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &w_buf,
        &old_state_buf,
        &new_state_buf,
        &y_buf,
        &p_buf,
        params,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");

    let y: &[f32] = y_buf.as_slice().expect("read");
    for i in 0..2 {
        assert!(
            (y[i] - y_ref[i]).abs() < 1e-5,
            "decode y[{}]: got {}, expected {}",
            i, y[i], y_ref[i]
        );
    }
    let st: &[f32] = new_state_buf.as_slice().expect("read state");
    for i in 0..6 {
        assert!(
            (st[i] - state_ref[i]).abs() < 1e-6,
            "decode state[{}]: got {}, expected {}",
            i, st[i], state_ref[i]
        );
    }
}

// ==================================================================
// BF16 path
// ==================================================================

#[test]
fn test_ssm_conv_bf16_tiny() {
    use half::bf16;
    let (device, mut registry) = setup();
    let params = SsmConvParams {
        channels: 1,
        n_tokens: 2,
        n_seqs: 1,
        k_width: 4,
    };

    let x_data_f32 = [0.5f32, 0.25f32];
    let w_data_f32 = [0.1f32, 0.2, 0.3, 0.4];
    let state_data_f32 = [0.1f32, 0.2, 0.3];

    let (y_ref_f32, state_ref_f32) = cpu_reference(&x_data_f32, &w_data_f32, &state_data_f32, params);

    let x_bf: Vec<bf16> = x_data_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_bf: Vec<bf16> = w_data_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let s_bf: Vec<bf16> = state_data_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let mut x_buf = device
        .alloc_buffer(4, DType::BF16, vec![2])
        .expect("x");
    x_buf.as_mut_slice::<bf16>().expect("mut").copy_from_slice(&x_bf);
    let mut w_buf = device
        .alloc_buffer(8, DType::BF16, vec![4])
        .expect("w");
    w_buf.as_mut_slice::<bf16>().expect("mut").copy_from_slice(&w_bf);
    let mut s_buf = device
        .alloc_buffer(6, DType::BF16, vec![3])
        .expect("s");
    s_buf.as_mut_slice::<bf16>().expect("mut").copy_from_slice(&s_bf);
    let ns_buf = device
        .alloc_buffer(6, DType::BF16, vec![3])
        .expect("ns");
    let y_buf = device.alloc_buffer(4, DType::BF16, vec![2]).expect("y");
    let p_buf = alloc_params(&device, params);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &w_buf,
        &s_buf,
        &ns_buf,
        &y_buf,
        &p_buf,
        params,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let y: &[bf16] = y_buf.as_slice().expect("read");
    for i in 0..2 {
        let got = y[i].to_f32();
        let diff = (got - y_ref_f32[i]).abs();
        assert!(diff < 1e-2, "bf16 y[{}]: got {}, expected {}", i, got, y_ref_f32[i]);
    }
    let st: &[bf16] = ns_buf.as_slice().expect("read state");
    for i in 0..3 {
        let got = st[i].to_f32();
        let diff = (got - state_ref_f32[i]).abs();
        assert!(diff < 1e-2, "bf16 state[{}]: got {}", i, got);
    }
}

// ==================================================================
// Error handling
// ==================================================================

#[test]
fn test_ssm_conv_rejects_zero_channels() {
    let (device, mut registry) = setup();
    let params = SsmConvParams {
        channels: 0,
        n_tokens: 1,
        n_seqs: 1,
        k_width: 4,
    };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let p_buf = alloc_params(&device, params);
    let mut enc = device.command_encoder().expect("enc");
    let res = mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        &p_buf,
        params,
    );
    assert!(res.is_err(), "zero channels should error");
}

#[test]
fn test_ssm_conv_rejects_k_too_small() {
    let (device, mut registry) = setup();
    let params = SsmConvParams {
        channels: 1,
        n_tokens: 1,
        n_seqs: 1,
        k_width: 1,
    };
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let p_buf = alloc_params(&device, params);
    let mut enc = device.command_encoder().expect("enc");
    let res = mlx_native::ops::ssm_conv::dispatch_ssm_conv(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        &p_buf,
        params,
    );
    assert!(res.is_err(), "K=1 should error");
}
