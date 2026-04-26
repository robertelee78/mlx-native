//! Correctness tests for `dense_matmul_f32_f32_tensor`
//! (hf2q ADR-005 iter-118 BF16-vs-F32 ViT attention A/B diagnostic
//! kernel).
//!
//! Cases:
//! 1. Small random (M=64, K=128, N=64) — multi-tile single batch.
//! 2. ViT attention shape (M=N=256, K=72) — gemma4v per-head shape.
//! 3. Rectangular (M=64, K=256, N=128) — non-square.
//! 4. Zero input (all zeros) — sanity.
//! 5. Identity (B = I) — output must equal A.
//!
//! Tolerance: F32 GEMM should be near-bit-exact; we assert max-abs
//! diff < 1e-4 and (where shape is large enough to make cosine
//! meaningful) cosine >= 0.999999.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::dense_mm_f32_f32::{
    dense_matmul_f32_f32_tensor, DenseMmF32F32Params,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
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

/// CPU reference: same dst[b, m, n] = sum_k src0[b/r2, n, k] * src1[b, m, k]
/// contract as the BF16 sibling.
fn cpu_ref(
    src0: &[f32],
    src1: &[f32],
    m: usize,
    n: usize,
    k: usize,
    src0_batch: usize,
    src1_batch: usize,
) -> Vec<f32> {
    assert_eq!(src1_batch % src0_batch, 0, "r2 must divide evenly");
    let r2 = src1_batch / src0_batch;
    let mut dst = vec![0.0f32; src1_batch * m * n];
    for b in 0..src1_batch {
        let b_src0 = b / r2;
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    let a = src0[b_src0 * n * k + ni * k + ki];
                    let bv = src1[b * m * k + mi * k + ki];
                    acc += a * bv;
                }
                dst[b * m * n + mi * n + ni] = acc;
            }
        }
    }
    dst
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        dot += (av as f64) * (bv as f64);
        na += (av as f64) * (av as f64);
        nb += (bv as f64) * (bv as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Run a generic correctness case with explicit src0/src1 contents.
fn run_case_explicit(
    src0_f32: Vec<f32>,
    src1_f32: Vec<f32>,
    m: u32,
    n: u32,
    k: u32,
    src0_batch: u32,
    src1_batch: u32,
    max_abs_tol: f32,
) -> Vec<f32> {
    let (device, mut registry) = setup();

    let expected = cpu_ref(
        &src0_f32,
        &src1_f32,
        m as usize,
        n as usize,
        k as usize,
        src0_batch as usize,
        src1_batch as usize,
    );

    let src0_bytes = (src0_batch * n * k) as usize * 4;
    let mut src0_buf = device
        .alloc_buffer(
            src0_bytes,
            DType::F32,
            vec![src0_batch as usize, n as usize, k as usize],
        )
        .expect("alloc src0");
    src0_buf
        .as_mut_slice::<f32>()
        .expect("write src0")
        .copy_from_slice(&src0_f32);

    let src1_bytes = (src1_batch * m * k) as usize * 4;
    let mut src1_buf = device
        .alloc_buffer(
            src1_bytes,
            DType::F32,
            vec![src1_batch as usize, m as usize, k as usize],
        )
        .expect("alloc src1");
    src1_buf
        .as_mut_slice::<f32>()
        .expect("write src1")
        .copy_from_slice(&src1_f32);

    let dst_bytes = (src1_batch * m * n) as usize * 4;
    let mut dst_buf = device
        .alloc_buffer(
            dst_bytes,
            DType::F32,
            vec![src1_batch as usize, m as usize, n as usize],
        )
        .expect("alloc dst");

    let params = DenseMmF32F32Params {
        m,
        n,
        k,
        src0_batch,
        src1_batch,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dense_matmul_f32_f32_tensor(
        &mut encoder,
        &mut registry,
        &device,
        &src0_buf,
        &src1_buf,
        &mut dst_buf,
        &params,
    )
    .expect("dispatch dense_matmul_f32_f32_tensor");
    encoder.commit_and_wait().expect("commit_and_wait");

    let actual = dst_buf.as_slice::<f32>().expect("read dst").to_vec();
    let total = (src1_batch * m * n) as usize;
    let mut max_err = 0.0f32;
    for i in 0..total {
        let err = (actual[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < max_abs_tol,
        "max error {} > tolerance {} (m={} n={} k={} src0_b={} src1_b={})",
        max_err,
        max_abs_tol,
        m,
        n,
        k,
        src0_batch,
        src1_batch
    );
    actual
}

fn run_random_case(
    m: u32,
    n: u32,
    k: u32,
    src0_batch: u32,
    src1_batch: u32,
    seed_a: u64,
    seed_b: u64,
    max_abs_tol: f32,
    cosine_floor: f32,
) {
    let src0_f32 = pseudo_random_f32(seed_a, (src0_batch * n * k) as usize);
    let src1_f32 = pseudo_random_f32(seed_b, (src1_batch * m * k) as usize);
    let expected = cpu_ref(
        &src0_f32,
        &src1_f32,
        m as usize,
        n as usize,
        k as usize,
        src0_batch as usize,
        src1_batch as usize,
    );
    let actual = run_case_explicit(
        src0_f32,
        src1_f32,
        m,
        n,
        k,
        src0_batch,
        src1_batch,
        max_abs_tol,
    );
    let cos = cosine(&actual, &expected);
    assert!(
        cos >= cosine_floor,
        "cosine {} < floor {} (m={} n={} k={})",
        cos,
        cosine_floor,
        m,
        n,
        k
    );
}

// ----------------------------------------------------------------------
// Required test cases from W48 iter-119 spec.
// ----------------------------------------------------------------------

#[test]
fn dense_mm_f32_f32_correctness_small() {
    // M=64, K=128, N=64 — multi-tile single batch.
    // F32 GEMM should be near-bit-exact: max-abs-diff < 1e-4 across
    // the K=128 = 4 NK-tiles accumulation.
    run_random_case(64, 64, 128, 1, 1, 1, 2, 1e-4, 0.999999);
}

#[test]
fn dense_mm_f32_f32_correctness_vit_attention_shape() {
    // M=N=256 (ViT image-tokens), K=72 (head_dim) — gemma4v per-head
    // attention shape.  This is the call site that ADR-005 iter-118
    // wants to A/B against the BF16 path; correctness here is the
    // gating bar for iter-120 to wire it in.
    //
    // K=72 = 2 full NK=32 tiles + 8-element partial tile, so this
    // simultaneously exercises the partial-K-tile path that the BF16
    // sibling locked down in iter-67.
    run_random_case(256, 256, 72, 1, 1, 17, 18, 1e-3, 0.999999);
}

#[test]
fn dense_mm_f32_f32_correctness_rectangular() {
    // M=64, K=256, N=128 — non-square.  Tests the M < N tiling path.
    run_random_case(64, 128, 256, 1, 1, 3, 4, 1e-3, 0.999999);
}

#[test]
fn dense_mm_f32_f32_zero_input() {
    // All zeros input -> all zeros output.  Smallest legal K=32.
    let src0 = vec![0.0f32; 64 * 32]; // n=64, k=32
    let src1 = vec![0.0f32; 64 * 32]; // m=64, k=32
    let actual = run_case_explicit(src0, src1, 64, 64, 32, 1, 1, 1e-6);
    for &v in &actual {
        assert_eq!(v, 0.0, "zero-input case must produce exactly 0.0");
    }
}

#[test]
fn dense_mm_f32_f32_identity() {
    // M=K=N=64.  src0 = identity matrix [N=64, K=64], src1 = random
    // [M=64, K=64].  Output[m, n] = sum_k I[n,k] * src1[m,k] = src1[m, n].
    // (i.e. dst = src1 viewed as [M, N=K]).
    let nk: u32 = 64;
    let mut src0 = vec![0.0f32; (nk * nk) as usize]; // [N=64, K=64]
    for i in 0..nk as usize {
        src0[i * (nk as usize) + i] = 1.0;
    }
    let src1 = pseudo_random_f32(0xBEEF, (nk * nk) as usize); // [M=64, K=64]

    let actual = run_case_explicit(
        src0,
        src1.clone(),
        nk, // m
        nk, // n
        nk, // k
        1,
        1,
        1e-5,
    );
    // For identity src0 (n,k) and src1 (m,k):
    //   dst[m, n] = sum_k I[n,k] * src1[m,k] = src1[m, n]
    // So dst should equal src1 element-wise (viewed [M,N]==[M,K]).
    let mut max_err = 0.0f32;
    for i in 0..(nk * nk) as usize {
        let err = (actual[i] - src1[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 1e-5,
        "identity case max_err {} >= 1e-5",
        max_err
    );
}

// ----------------------------------------------------------------------
// Additional safety / GQA / partial-tile cases for parity with the BF16
// sibling's regression suite.
// ----------------------------------------------------------------------

#[test]
fn dense_mm_f32_f32_single_tile_minimum() {
    // Smallest legal geometry: M=32 (= 1 NR1 tile), N=64 (= 1 NR0
    // tile), K=32 (= 1 NK tile).  Single iteration of the K-loop.
    run_random_case(32, 64, 32, 1, 1, 5, 6, 1e-4, 0.999999);
}

#[test]
fn dense_mm_f32_f32_partial_output_tile() {
    // M=35, N=67, K=64 — both M and N have partial tiles, exercising
    // the shmem-copy write-back fallback in the kernel's else branch.
    run_random_case(35, 67, 64, 1, 1, 11, 12, 1e-4, 0.999999);
}

#[test]
fn dense_mm_f32_f32_gqa_broadcast_r2_eq_4() {
    // 2 src0 batches, 8 src1 batches -> r2=4.  Mirrors gemma4v's
    // GQA broadcast contract end-to-end for the F32 attention path.
    run_random_case(32, 64, 64, 2, 8, 7, 8, 1e-4, 0.999999);
}

#[test]
fn dense_mm_f32_f32_partial_k_tile_k_eq_72() {
    // K=72 = 2 NK + 8 partial.  ViT head_dim shape; ensures the
    // partial-K guard zero-pads correctly in F32 just like BF16.
    run_random_case(64, 128, 72, 1, 1, 17, 18, 1e-3, 0.999999);
}
