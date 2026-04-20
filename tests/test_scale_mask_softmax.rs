//! Correctness tests for `dispatch_scale_mask_softmax_f32`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::scale_mask_softmax::{
    dispatch_scale_mask_softmax_f32, ScaleMaskSoftmaxParams,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    (MlxDevice::new().expect("MlxDevice::new"), KernelRegistry::new())
}

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// CPU reference: for each row, apply scale, add mask (cast bf16→f32),
/// then numerically-stable softmax.  Mask is shared across heads
/// (identical math to the kernel).
fn cpu_ref(
    input: &[f32],
    mask_bf16: &[bf16],
    rows: usize,
    cols: usize,
    seq_q: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let q = r % seq_q;
        let mut mx = f32::NEG_INFINITY;
        for k in 0..cols {
            let v = input[r * cols + k] * scale + mask_bf16[q * cols + k].to_f32();
            if v > mx {
                mx = v;
            }
        }
        let mut sum = 0.0f32;
        for k in 0..cols {
            let v = input[r * cols + k] * scale + mask_bf16[q * cols + k].to_f32();
            let e = (v - mx).exp();
            out[r * cols + k] = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for k in 0..cols {
            out[r * cols + k] *= inv;
        }
    }
    out
}

fn run_case(rows: u32, cols: u32, seq_q: u32, scale: f32, seed: u64, tol: f32) {
    let (device, mut registry) = setup();

    let input_f32 = pseudo_random_f32(seed, (rows * cols) as usize);

    // Build a bf16 mask with some masked positions (-inf) to exercise
    // the -inf handling path.  Causal-like triangular: for each q, mask
    // out k > q.
    let mut mask_bf16 = vec![bf16::ZERO; (seq_q * cols) as usize];
    for q in 0..seq_q {
        for k in 0..cols {
            if k > q {
                mask_bf16[(q * cols + k) as usize] = bf16::NEG_INFINITY;
            }
        }
    }

    let expected = cpu_ref(
        &input_f32,
        &mask_bf16,
        rows as usize,
        cols as usize,
        seq_q as usize,
        scale,
    );

    let mut input_buf = device
        .alloc_buffer(
            (rows * cols * 4) as usize,
            DType::F32,
            vec![rows as usize, cols as usize],
        )
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("input mut")
        .copy_from_slice(&input_f32);

    let mask_u16: Vec<u16> = mask_bf16.iter().map(|b| b.to_bits()).collect();
    let mut mask_buf = device
        .alloc_buffer(
            (seq_q * cols * 2) as usize,
            DType::BF16,
            vec![seq_q as usize, cols as usize],
        )
        .expect("alloc mask");
    mask_buf
        .as_mut_slice::<u16>()
        .expect("mask mut")
        .copy_from_slice(&mask_u16);

    let mut output_buf = device
        .alloc_buffer(
            (rows * cols * 4) as usize,
            DType::F32,
            vec![rows as usize, cols as usize],
        )
        .expect("alloc output");

    let params = ScaleMaskSoftmaxParams { rows, cols, seq_q, scale };
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_scale_mask_softmax_f32(
        &mut encoder, &mut registry, &device,
        &input_buf, &mut output_buf, &mask_buf, &params,
    ).expect("dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let actual = output_buf.as_slice::<f32>().expect("read").to_vec();
    let mut max_err = 0.0f32;
    for i in 0..(rows * cols) as usize {
        let err = (actual[i] - expected[i]).abs();
        if err > max_err { max_err = err; }
    }
    assert!(
        max_err < tol,
        "max error {} > tol {} (rows={} cols={} seq_q={} scale={})",
        max_err, tol, rows, cols, seq_q, scale
    );
}

#[test]
fn single_head_single_query() {
    // rows=1, one query row, full-open mask (no masking since q=0 and k=0).
    run_case(1, 64, 1, 0.125, 11, 1e-5);
}

#[test]
fn single_head_triangular_mask() {
    run_case(8, 8, 8, 0.125, 12, 1e-5);
}

#[test]
fn multi_head_shared_mask() {
    // nh=4, seq_q=16 → rows=64, heads share the same triangular mask.
    run_case(64, 16, 16, 0.0625, 13, 1e-5);
}

#[test]
fn prefill_like_shape() {
    // nh=4, seq_q=128, seq_k=128 → rows=512.  Representative of the
    // shape the non-FA prefill path will hit (scaled down to keep the
    // test fast).
    run_case(512, 128, 128, 0.0625, 14, 1e-5);
}
