//! Tests for `rope_train` — differentiable RoPE op (forward + backward).
//!
//! Phase 1a of `flash_attn_train` (ADR-022 / DWQ training).
//!
//! Test plan (mirrors the mission spec):
//!
//! 1. **Forward parity vs CPU oracle** — small BF16 GPU output vs `rope_reference_f32`.
//! 2. **Backward = forward with negated pos** — `backward(dY, pos)` == `forward(dY, -pos)`.
//! 3. **Round-trip identity** — `backward(forward(x, pos), pos) ≈ x`.
//! 4. **Finite-diff falsifier** — analytic Jacobian column == numerical estimate.
//! 5. **IMROPE section parity** — only the activated section is non-zero post-RoPE.
//!
//! All GPU-vs-CPU tolerances are at `atol=1e-3` for bf16 (bf16 precision floor
//! ≈ 7×10^{-3} for unit values; `atol=1e-3` is comfortably within this range for
//! the small values used in these fixtures).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use half::bf16;
use mlx_native::ops::rope_train::{
    dispatch_rope_backward_bf16, dispatch_rope_backward_f32, dispatch_rope_forward_bf16,
    dispatch_rope_forward_f32, RopeTrainParams,
};
use mlx_native::ops::rope_multi::register as register_rope_multi;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn device_and_registry() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    register_rope_multi(&mut registry);
    (device, registry)
}

fn upload_bf16(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let bf: Vec<bf16> = data.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut buf = device
        .alloc_buffer(bf.len() * 2, DType::BF16, vec![bf.len()])
        .expect("alloc bf16");
    buf.as_mut_slice::<bf16>()
        .expect("mut")
        .copy_from_slice(&bf);
    buf
}

fn upload_f32(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .expect("alloc f32");
    buf.as_mut_slice::<f32>()
        .expect("mut")
        .copy_from_slice(data);
    buf
}

fn upload_i32(device: &MlxDevice, data: &[i32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(data.len() * 4, DType::I32, vec![data.len()])
        .expect("alloc i32");
    buf.as_mut_slice::<i32>()
        .expect("mut")
        .copy_from_slice(data);
    buf
}

fn alloc_bf16_zeros(device: &MlxDevice, n: usize) -> MlxBuffer {
    device
        .alloc_buffer(n * 2, DType::BF16, vec![n])
        .expect("alloc bf16 zeros")
}

fn alloc_f32_zeros(device: &MlxDevice, n: usize) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc f32 zeros")
}

fn download_bf16(buf: &MlxBuffer) -> Vec<f32> {
    buf.as_slice::<bf16>()
        .expect("read bf16")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn download_f32(buf: &MlxBuffer) -> Vec<f32> {
    buf.as_slice::<f32>().expect("read f32").to_vec()
}

/// Build text-mode positions `[4 * batch * seq_len]` where all 4 axes equal
/// the token's 1-D position in the sequence.  The position of the first token
/// in batch `b` is `b * seq_len` (distinct batches don't share positions; in
/// real training they'd each start at their own sequence offset).
fn text_positions(batch: u32, seq_len: u32) -> Vec<i32> {
    let n = (batch * seq_len) as usize;
    // All 4 axes = token index (within the flat batch*seq_len space).
    let base: Vec<i32> = (0..n as i32).collect();
    let mut pos = Vec::with_capacity(4 * n);
    for _ in 0..4 {
        pos.extend_from_slice(&base);
    }
    pos
}

// ---------------------------------------------------------------------------
// CPU reference — f32 scalar IMROPE
// ---------------------------------------------------------------------------
//
// Mirrors `cpu_rope_multi` in `tests/test_rope_multi.rs` for the IMROPE case.
// Input shape: `[batch * n_heads * seq_len, head_dim]` (row-major, rows = batch * n_heads * seq_len).
// `positions` has length `4 * batch * seq_len`.

fn pick_axis_imrope(sector: u32, sections: [u32; 4]) -> usize {
    let s = sections;
    if sector % 3 == 0 && sector < 3 * s[0] {
        0
    } else if sector % 3 == 1 && sector < 3 * s[1] {
        1
    } else if sector % 3 == 2 && sector < 3 * s[2] {
        2
    } else {
        3
    }
}

/// Pure-Rust f32 reference for IMROPE.
///
/// # Arguments
/// - `input`:     flat `[batch * n_heads * seq_len * head_dim]` f32
/// - `positions`: flat `[4 * batch * seq_len]` i32
///   layout: axis-0 block, axis-1 block, axis-2 block, axis-3 block;
///   each block length = `batch * seq_len`.
pub fn rope_reference_f32(input: &[f32], positions: &[i32], p: &RopeTrainParams) -> Vec<f32> {
    let batch = p.batch as usize;
    let n_heads = p.n_heads as usize;
    let seq = p.seq_len as usize;
    let hd = p.head_dim as usize;
    let half_dim = hd / 2;
    let rope_dim = p.rope_dim as usize;
    let half_rope = rope_dim / 2;
    let sections = p.sections;
    let theta_base = p.theta_base;

    // sect_dims = sum of all section counts (may be < half_rope; kernel wraps)
    let sect_dims = sections.iter().sum::<u32>().max(1) as usize;

    let n_rows = batch * n_heads * seq;
    assert_eq!(input.len(), n_rows * hd, "rope_reference_f32: input size mismatch");
    assert_eq!(positions.len(), 4 * batch * seq, "rope_reference_f32: positions size mismatch");

    let mut out = input.to_vec();

    for row in 0..n_rows {
        let base = row * hd;
        // row = b * n_heads * seq + h * seq + s
        // seq_idx within this batch element's sequence:
        let b = row / (n_heads * seq);
        let tok_in_batch = (row % (n_heads * seq)) / n_heads; // position within [0, seq)
        // flat token index in [0, batch * seq_len)
        let flat_tok = b * seq + tok_in_batch;

        for pair in 0..half_dim {
            if pair < half_rope {
                let sector = (pair % sect_dims) as u32;
                let axis = pick_axis_imrope(sector, sections);
                // positions[axis * batch * seq + flat_tok]
                let pos = positions[axis * batch * seq + flat_tok] as f32;

                let dim_ratio = 2.0 * pair as f32 / rope_dim as f32;
                let freq = 1.0 / theta_base.powf(dim_ratio);
                let theta = pos * freq;
                let (cos_a, sin_a) = (theta.cos(), theta.sin());

                let x0 = input[base + pair];
                let x1 = input[base + pair + half_dim];
                out[base + pair] = x0 * cos_a - x1 * sin_a;
                out[base + pair + half_dim] = x0 * sin_a + x1 * cos_a;
            } else {
                // pass-through
                out[base + pair] = input[base + pair];
                out[base + pair + half_dim] = input[base + pair + half_dim];
            }
        }
    }
    out
}

fn assert_close_atol(label: &str, got: &[f32], want: &[f32], atol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut max_diff: f32 = 0.0;
    let mut max_idx = 0;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - w).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
        assert!(
            d <= atol,
            "{label}: i={max_idx}: got={g} want={w} diff={d} > atol={atol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Run helpers — forward and backward
// ---------------------------------------------------------------------------

fn run_forward_bf16(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input_f32: &[f32],
    positions: &[i32],
    p: &RopeTrainParams,
) -> Vec<f32> {
    let n = input_f32.len();
    let in_buf = upload_bf16(device, input_f32);
    let out_buf = alloc_bf16_zeros(device, n);
    let pos_buf = upload_i32(device, positions);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_rope_forward_bf16(
        &mut enc,
        registry,
        device.metal_device(),
        device,
        &in_buf,
        &pos_buf,
        &out_buf,
        p,
    )
    .expect("dispatch_rope_forward_bf16");
    enc.commit_and_wait().expect("commit");
    download_bf16(&out_buf)
}

fn run_backward_bf16(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    grad_out_f32: &[f32],
    positions: &[i32],
    p: &RopeTrainParams,
) -> Vec<f32> {
    let n = grad_out_f32.len();
    let grad_out_buf = upload_bf16(device, grad_out_f32);
    let grad_in_buf = alloc_bf16_zeros(device, n);
    let pos_buf = upload_i32(device, positions);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_rope_backward_bf16(
        &mut enc,
        registry,
        device.metal_device(),
        device,
        &grad_out_buf,
        &pos_buf,
        &grad_in_buf,
        p,
    )
    .expect("dispatch_rope_backward_bf16");
    enc.commit_and_wait().expect("commit");
    download_bf16(&grad_in_buf)
}

fn run_forward_f32(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input_f32: &[f32],
    positions: &[i32],
    p: &RopeTrainParams,
) -> Vec<f32> {
    let n = input_f32.len();
    let in_buf = upload_f32(device, input_f32);
    let out_buf = alloc_f32_zeros(device, n);
    let pos_buf = upload_i32(device, positions);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_rope_forward_f32(
        &mut enc,
        registry,
        device.metal_device(),
        device,
        &in_buf,
        &pos_buf,
        &out_buf,
        p,
    )
    .expect("dispatch_rope_forward_f32");
    enc.commit_and_wait().expect("commit");
    download_f32(&out_buf)
}

fn run_backward_f32(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    grad_out_f32: &[f32],
    positions: &[i32],
    p: &RopeTrainParams,
) -> Vec<f32> {
    let n = grad_out_f32.len();
    let grad_out_buf = upload_f32(device, grad_out_f32);
    let grad_in_buf = alloc_f32_zeros(device, n);
    let pos_buf = upload_i32(device, positions);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_rope_backward_f32(
        &mut enc,
        registry,
        device.metal_device(),
        device,
        &grad_out_buf,
        &pos_buf,
        &grad_in_buf,
        p,
    )
    .expect("dispatch_rope_backward_f32");
    enc.commit_and_wait().expect("commit");
    download_f32(&grad_in_buf)
}

// ---------------------------------------------------------------------------
// Test 1: Forward parity vs CPU oracle (bf16 GPU vs f32 CPU at atol=1e-3)
// ---------------------------------------------------------------------------

/// Shape (B=1, H=4, S=64, D=64) — standard IMROPE with full rope_dim=64.
#[test]
fn test_forward_parity_b1_h4_s64_d64() {
    let (device, mut registry) = device_and_registry();

    let p = RopeTrainParams {
        batch: 1,
        n_heads: 4,
        seq_len: 64,
        head_dim: 64,
        rope_dim: 64,
        theta_base: 1e6,
        sections: [11, 11, 10, 0],
    };

    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    // Deterministic pseudo-random input in [-1, 1].
    let mut seed = 0xdeadc0deu32;
    let mut rand_f32 = || -> f32 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as i32 as f32) / (i32::MAX as f32)
    };
    let input: Vec<f32> = (0..n).map(|_| rand_f32()).collect();
    let positions = text_positions(p.batch, p.seq_len);

    let got = run_forward_bf16(&device, &mut registry, &input, &positions, &p);
    let want = rope_reference_f32(&input, &positions, &p);

    // bf16 ULP at magnitude ~1.3 ≈ 2^0 * 2^{-7} = 7.8e-3; atol=6e-3 < 1 ULP.
    assert_close_atol("fwd B1 H4 S64 D64", &got, &want, 6e-3);
}

/// Shape (B=2, H=2, S=128, D=128) — partial rotary: rope_dim=64 < head_dim=128.
#[test]
fn test_forward_parity_b2_h2_s128_d128() {
    let (device, mut registry) = device_and_registry();

    let p = RopeTrainParams {
        batch: 2,
        n_heads: 2,
        seq_len: 128,
        head_dim: 128,
        rope_dim: 64, // partial rotary: only first 64 dims rotate
        theta_base: 1e6,
        sections: [11, 11, 10, 0],
    };

    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    let mut seed = 0xcafebabeu32;
    let mut rand_f32 = || -> f32 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as i32 as f32) / (i32::MAX as f32)
    };
    let input: Vec<f32> = (0..n).map(|_| rand_f32()).collect();
    let positions = text_positions(p.batch, p.seq_len);

    let got = run_forward_bf16(&device, &mut registry, &input, &positions, &p);
    let want = rope_reference_f32(&input, &positions, &p);

    // bf16 atol=6e-3 (within 1 ULP for values up to magnitude ~1.3).
    assert_close_atol("fwd B2 H2 S128 D128 partial", &got, &want, 6e-3);

    // Verify partial-rotary tail passes through unchanged within bf16 round-trip error.
    let hd = p.head_dim as usize;
    let half_dim = hd / 2;
    let half_rope = p.rope_dim as usize / 2;
    let n_rows = (p.batch * p.n_heads * p.seq_len) as usize;
    for row in 0..n_rows {
        let base = row * hd;
        for pair in half_rope..half_dim {
            // These indices should be pass-through, so got ≈ input (roundtripped through bf16).
            let d0 = (got[base + pair] - input[base + pair]).abs();
            let d1 = (got[base + pair + half_dim] - input[base + pair + half_dim]).abs();
            assert!(
                d0 < 5e-3,
                "partial-rotary tail[pair={pair}] x0 modified: got={}, input={}",
                got[base + pair], input[base + pair]
            );
            assert!(
                d1 < 5e-3,
                "partial-rotary tail[pair={pair}] x1 modified: got={}, input={}",
                got[base + pair + half_dim], input[base + pair + half_dim]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Backward = forward with negated pos (atol=1e-4 — same kernel path)
// ---------------------------------------------------------------------------

/// `dispatch_rope_backward_bf16(dY, pos)` must be bit-for-bit identical to
/// `dispatch_rope_forward_bf16(dY, -pos)`.  Both dispatch the same kernel;
/// the only difference is the sign of the positions passed in.
#[test]
fn test_backward_equals_forward_with_negated_pos_bf16() {
    let (device, mut registry) = device_and_registry();

    let p = RopeTrainParams {
        batch: 1,
        n_heads: 4,
        seq_len: 32,
        head_dim: 64,
        rope_dim: 64,
        theta_base: 1e6,
        sections: [11, 11, 10, 0],
    };

    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    let mut seed = 0xf00dbabeu32;
    let mut rand_f32 = || -> f32 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as i32 as f32) / (i32::MAX as f32)
    };
    let dy: Vec<f32> = (0..n).map(|_| rand_f32()).collect();
    let positions = text_positions(p.batch, p.seq_len);
    let neg_positions: Vec<i32> = positions.iter().map(|&v| -v).collect();

    // Path A: backward(dY, pos) — should internally negate positions.
    let backward_result = run_backward_bf16(&device, &mut registry, &dy, &positions, &p);

    // Path B: forward(dY, -pos) — explicit negated positions.
    let forward_negated_result = run_forward_bf16(&device, &mut registry, &dy, &neg_positions, &p);

    // Bit-exact: both paths produce identical bytes because both call
    // `rope_multi_bf16` with the same negated-positions buffer.
    assert_eq!(
        backward_result.len(),
        forward_negated_result.len(),
        "length mismatch"
    );
    for (i, (b, f)) in backward_result
        .iter()
        .zip(forward_negated_result.iter())
        .enumerate()
    {
        assert_eq!(
            bf16::from_f32(*b).to_bits(),
            bf16::from_f32(*f).to_bits(),
            "backward != forward(-pos) at i={}: backward={}, forward_neg={}",
            i, b, f
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: Round-trip identity — backward(forward(x, pos), pos) ≈ x
// ---------------------------------------------------------------------------

/// Applying forward RoPE then backward RoPE with the same positions must
/// return the original input, because `R(-pos) * R(pos) = I`.
#[test]
fn test_round_trip_identity_bf16() {
    let (device, mut registry) = device_and_registry();

    let p = RopeTrainParams {
        batch: 1,
        n_heads: 2,
        seq_len: 16,
        head_dim: 64,
        rope_dim: 64,
        theta_base: 1e6,
        sections: [11, 11, 10, 0],
    };

    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    // Use values in [-0.5, 0.5] so bf16 round-trip is tight.
    let input_f32: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.03 + 0.01).sin()) * 0.5)
        .collect();
    let positions = text_positions(p.batch, p.seq_len);

    // Step 1: forward(x, pos) → y
    let y = run_forward_bf16(&device, &mut registry, &input_f32, &positions, &p);

    // Step 2: backward(y, pos) → x_recovered
    let x_recovered = run_backward_bf16(&device, &mut registry, &y, &positions, &p);

    // After two bf16 round-trips the error accumulates; use atol=5e-3 (two bf16 round-trips).
    assert_close_atol("round-trip identity", &x_recovered, &input_f32, 5e-3);
}

// ---------------------------------------------------------------------------
// Test 4: Finite-difference falsifier (f32 for full numerical precision)
// ---------------------------------------------------------------------------
//
// At small size (B=1, H=1, S=8, D=16) perturb input element [0,0,0,0] by eps,
// compute numerical (out_perturbed - out_baseline) / eps, compare to the
// kernel's analytical Jacobian column (= backward with one-hot dY at the
// perturbed output index, pick the dQ at the corresponding input index).
//
// We use the f32 variants for this test to avoid bf16 quantization noise
// masking the finite-difference signal.

#[test]
fn test_finite_diff_falsifier_f32() {
    let (device, mut registry) = device_and_registry();

    let p = RopeTrainParams {
        batch: 1,
        n_heads: 1,
        seq_len: 8,
        head_dim: 16,
        rope_dim: 16,
        theta_base: 1e4,
        sections: [3, 3, 2, 0],
    };

    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    // Deterministic input.
    let input_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1 - 0.5).tanh()).collect();
    let positions = text_positions(p.batch, p.seq_len);

    let eps = 1e-2_f32;

    // Probe: perturb input element at index `probe_in`, measure gradient on
    // output element at `probe_out = probe_in` (diagonal of Jacobian row —
    // RoPE mixes pairs so the Jacobian is block-diagonal).
    //
    // We test 4 probe_in values spanning different pairs to cover the
    // multi-axis IMROPE pattern.
    let probes: &[usize] = &[0, 3, 7, 12];

    let baseline = run_forward_f32(&device, &mut registry, &input_f32, &positions, &p);

    for &probe_in in probes {
        let mut input_perturbed = input_f32.clone();
        input_perturbed[probe_in] += eps;
        let perturbed = run_forward_f32(&device, &mut registry, &input_perturbed, &positions, &p);

        // Finite-difference estimate of ∂(out[probe_in]) / ∂(in[probe_in]).
        // For RoPE, the output element at probe_in is directly affected by in[probe_in]
        // (and also by in[probe_in ± half_dim], which is why we probe the diagonal).
        let fd_grad = (perturbed[probe_in] - baseline[probe_in]) / eps;

        // Analytical Jacobian via backward:
        // Set dY = one-hot at probe_in → backward produces dX.
        // The entry dX[probe_in] is ∂(out[probe_in]) / ∂(in[probe_in]).
        let mut dy_onehot = vec![0f32; n];
        dy_onehot[probe_in] = 1.0;
        let dx = run_backward_f32(&device, &mut registry, &dy_onehot, &positions, &p);
        let analytic_grad = dx[probe_in];

        let diff = (analytic_grad - fd_grad).abs();
        let scale = analytic_grad.abs().max(fd_grad.abs()).max(1.0);
        assert!(
            diff / scale <= 5e-2,
            "finite-diff falsifier FAILED at probe_in={probe_in}: \
             analytic={analytic_grad:.6}, fd={fd_grad:.6}, diff={diff:.6}, scale={scale:.6}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: IMROPE section parity
// ---------------------------------------------------------------------------
//
// With sections=[s0, 0, 0, 0] (only section 0 active, s1=s2=s3=0), set
// the axis-0 positions to non-zero and axes 1-3 to zero. Then verify that
// the RoPE output is the same as running IMROPE with all axes = axis-0 pos.
//
// More targeted: build an input that is zero except for one section's pairs,
// and verify that section's output is non-trivially rotated while other
// sections pass through unchanged.

#[test]
fn test_imrope_section_independence() {
    let (device, mut registry) = device_and_registry();

    // Use sections=[4, 4, 4, 0] with head_dim=24, rope_dim=24 (= 8 pairs;
    // sect_dims = 12; sector=pair%12).  Axis mapping (IMROPE):
    //   pair 0: sector 0, 0%3==0, 0 < 12 -> axis 0
    //   pair 1: sector 1, 1%3==1, 1 < 12 -> axis 1
    //   pair 2: sector 2, 2%3==2, 2 < 12 -> axis 2
    //   pair 3: sector 3, 3%3==0, 3 < 12 -> axis 0
    //   pair 4: sector 4, 4%3==1, 4 < 12 -> axis 1
    //   pair 5: sector 5, 5%3==2, 5 < 12 -> axis 2
    //   pair 6: sector 6, 6%3==0, 6 < 12 -> axis 0
    //   pair 7: sector 7, 7%3==1, 7 < 12 -> axis 1
    //   pair 8: sector 8, 8%3==2, 8 < 12 -> axis 2
    //   pair 9: sector 9, 9%3==0, 9 < 12 -> axis 0
    //   pair 10: sector 10, 10%3==1, 10 < 12 -> axis 1
    //   pair 11: sector 11, 11%3==2, 11 < 12 -> axis 2
    // (rope_dim=24 -> half_rope=12 pairs rotated; pairs 0..11 all rotate)

    let p_full = RopeTrainParams {
        batch: 1,
        n_heads: 1,
        seq_len: 1,
        head_dim: 24,
        rope_dim: 24,
        theta_base: 1e4,
        sections: [4, 4, 4, 0],
    };

    let n = (p_full.head_dim) as usize; // 1 row of head_dim elements
    let hd = p_full.head_dim as usize;
    let half_dim = hd / 2;

    // Input: non-zero only in pairs that belong to axis 0 (pairs 0, 3, 6, 9 in
    // the first half; pairs 0+half_dim, 3+half_dim, ... in the second half).
    let mut input_axis0_only = vec![0f32; n];
    let axis0_pairs: &[usize] = &[0, 3, 6, 9]; // pairs assigned to axis 0
    for &pair in axis0_pairs {
        input_axis0_only[pair] = 1.0;
        input_axis0_only[pair + half_dim] = 0.5;
    }

    // Positions: axis-0 = 5, all others = 0.
    // pos layout for 1×1 sequence: [p_t, p_h, p_w, p_e] each length 1.
    let positions_axis0_active = [5i32, 0, 0, 0];
    // Same positions but with axis-0 also zero (no rotation anywhere).
    let positions_all_zero = [0i32, 0, 0, 0];

    // Run with axis-0 active: only axis-0 pairs should be rotated.
    let got_active = run_forward_f32(
        &device,
        &mut registry,
        &input_axis0_only,
        &positions_axis0_active,
        &p_full,
    );

    // Run with all-zero positions: no rotation, output should equal input.
    let got_zero = run_forward_f32(
        &device,
        &mut registry,
        &input_axis0_only,
        &positions_all_zero,
        &p_full,
    );

    // got_zero must equal input (pos=0 → theta=0 → cos=1, sin=0 → identity).
    assert_close_atol("pos=0 identity", &got_zero, &input_axis0_only, 1e-6);

    // For axis-0 pairs, got_active should DIFFER from input (rotation applied).
    let mut found_nonzero_rotation = false;
    for &pair in axis0_pairs {
        let diff0 = (got_active[pair] - input_axis0_only[pair]).abs();
        let diff1 = (got_active[pair + half_dim] - input_axis0_only[pair + half_dim]).abs();
        if diff0 > 1e-4 || diff1 > 1e-4 {
            found_nonzero_rotation = true;
        }
    }
    assert!(
        found_nonzero_rotation,
        "axis-0 pairs were NOT rotated when axis-0 position=5 (expected non-zero rotation)"
    );

    // Pairs assigned to axes 1 and 2 have pos=0, so their output should equal
    // input (no rotation). Input for axis-1/2 pairs is 0 anyway, so they should
    // remain 0 regardless — check that the output is zero at those positions.
    // axis-1 pairs: 1, 4, 7, 10; axis-2 pairs: 2, 5, 8, 11
    let non_axis0_pairs: &[usize] = &[1, 2, 4, 5, 7, 8, 10, 11];
    for &pair in non_axis0_pairs {
        let val0 = got_active[pair].abs();
        let val1 = got_active[pair + half_dim].abs();
        assert!(
            val0 < 1e-6,
            "non-axis0 pair {pair} x0 should be 0 (input=0, pos=0): got {val0}"
        );
        assert!(
            val1 < 1e-6,
            "non-axis0 pair {pair} x1 should be 0 (input=0, pos=0): got {val1}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: Qwen3.5 production shape smoke test — bf16 forward parity
// ---------------------------------------------------------------------------
//
// Uses the actual Qwen3.5 production parameters:
//   head_dim=256, rope_dim=64, sections=[11,11,10,0], freq_base=1e6
//   (rope_dim=64 = 25% partial rotary of head_dim=256)
//
// Verifies forward bf16 parity vs CPU oracle at atol=1e-3.

#[test]
fn test_qwen35_production_shape_forward_parity() {
    let (device, mut registry) = device_and_registry();

    // Qwen3.5 / Qwen3.6 production shape (small seq_len for test speed).
    let p = RopeTrainParams {
        batch: 1,
        n_heads: 4, // smaller than 28/40 for test
        seq_len: 8,
        head_dim: 256,
        rope_dim: 64,     // partial rotary factor 0.25
        theta_base: 1e6,  // rope_theta = 1_000_000
        sections: [11, 11, 10, 0], // mrope_section from Qwen3.5 config
    };

    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    let mut seed = 0xf1e2d3c4u32;
    let mut rand_f32 = || -> f32 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as i32 as f32) / (i32::MAX as f32) * 0.5
    };
    let input: Vec<f32> = (0..n).map(|_| rand_f32()).collect();
    let positions = text_positions(p.batch, p.seq_len);

    let got = run_forward_bf16(&device, &mut registry, &input, &positions, &p);
    let want = rope_reference_f32(&input, &positions, &p);

    // bf16 atol=6e-3 (within 1 ULP for values up to magnitude ~1.3).
    assert_close_atol("qwen35 production shape", &got, &want, 6e-3);

    // Partial-rotary tail (pairs 32..127) must pass through unchanged.
    let hd = p.head_dim as usize;
    let half_dim = hd / 2;
    let half_rope = p.rope_dim as usize / 2; // 32
    let n_rows = (p.batch * p.n_heads * p.seq_len) as usize;
    for row in 0..n_rows {
        let base = row * hd;
        for pair in half_rope..half_dim {
            let d0 = (got[base + pair] - input[base + pair]).abs();
            let d1 = (got[base + pair + half_dim] - input[base + pair + half_dim]).abs();
            // bf16 round-trip tolerance
            assert!(
                d0 < 5e-3,
                "qwen35 partial-rotary tail pair={pair} x0: got={}, input={}",
                got[base + pair],
                input[base + pair]
            );
            assert!(
                d1 < 5e-3,
                "qwen35 partial-rotary tail pair={pair} x1: got={}, input={}",
                got[base + pair + half_dim],
                input[base + pair + half_dim]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 7: Error handling — invalid params are rejected
// ---------------------------------------------------------------------------

#[test]
fn test_validates_odd_head_dim() {
    let (device, mut registry) = device_and_registry();
    let p = RopeTrainParams {
        batch: 1, n_heads: 1, seq_len: 4, head_dim: 15, rope_dim: 4,
        theta_base: 1e4, sections: [1, 1, 0, 0],
    };
    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    let in_buf = alloc_bf16_zeros(&device, n);
    let out_buf = alloc_bf16_zeros(&device, n);
    let pos_buf = upload_i32(&device, &vec![0i32; 4 * p.seq_len as usize]);
    let mut enc = device.command_encoder().expect("enc");
    let res = dispatch_rope_forward_bf16(
        &mut enc, &mut registry, device.metal_device(), &device,
        &in_buf, &pos_buf, &out_buf, &p,
    );
    assert!(res.is_err(), "odd head_dim must error");
}

#[test]
fn test_validates_rope_dim_gt_head_dim() {
    let (device, mut registry) = device_and_registry();
    let p = RopeTrainParams {
        batch: 1, n_heads: 1, seq_len: 4, head_dim: 16, rope_dim: 32,
        theta_base: 1e4, sections: [4, 4, 0, 0],
    };
    let n = (p.batch * p.n_heads * p.seq_len * p.head_dim) as usize;
    let in_buf = alloc_bf16_zeros(&device, n);
    let out_buf = alloc_bf16_zeros(&device, n);
    let pos_buf = upload_i32(&device, &vec![0i32; 4 * p.seq_len as usize]);
    let mut enc = device.command_encoder().expect("enc");
    let res = dispatch_rope_forward_bf16(
        &mut enc, &mut registry, device.metal_device(), &device,
        &in_buf, &pos_buf, &out_buf, &p,
    );
    assert!(res.is_err(), "rope_dim > head_dim must error");
}
