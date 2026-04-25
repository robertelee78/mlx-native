//! Correctness tests for `dense_matmul_bf16_f32_tensor`
//! (hf2q non-flash-attention prefill matmul).
//!
//! Cases:
//! 1. Single-tile shape (M=32, N=64, K=32) — smallest legal geometry.
//! 2. Multi-tile shape, single batch (M=64, N=128, K=128) — verifies the
//!    NK-loop and write-back paths.
//! 3. Partial-tile write-back (M=35, N=67, K=64) — exercises the
//!    shmem-copy fallback write-back path.
//! 4. GQA broadcast (src0_batch=2, src1_batch=8, r2=4) — verifies the
//!    head-broadcast offset math matches the CPU reference.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::dense_mm_bf16::{
    dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params,
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

/// CPU reference: for each batch b, compute
///   dst[b, m, n] = sum_k src0[b/r2, n, k] * src1[b, m, k]
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

fn run_case(
    m: u32,
    n: u32,
    k: u32,
    src0_batch: u32,
    src1_batch: u32,
    seed_a: u64,
    seed_b: u64,
    tol: f32,
) {
    let (device, mut registry) = setup();

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

    // Convert src0 f32 -> bf16 via the half crate (bit-exact with
    // Metal's bfloat cast).
    let src0_bf16: Vec<u16> = src0_f32
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect();

    let src0_bytes = (src0_batch * n * k) as usize * 2;
    let mut src0_buf = device
        .alloc_buffer(src0_bytes, DType::BF16, vec![
            src0_batch as usize, n as usize, k as usize,
        ])
        .expect("alloc src0");
    src0_buf
        .as_mut_slice::<u16>()
        .expect("write src0")
        .copy_from_slice(&src0_bf16);

    let src1_bytes = (src1_batch * m * k) as usize * 4;
    let mut src1_buf = device
        .alloc_buffer(src1_bytes, DType::F32, vec![
            src1_batch as usize, m as usize, k as usize,
        ])
        .expect("alloc src1");
    src1_buf
        .as_mut_slice::<f32>()
        .expect("write src1")
        .copy_from_slice(&src1_f32);

    let dst_bytes = (src1_batch * m * n) as usize * 4;
    let mut dst_buf = device
        .alloc_buffer(dst_bytes, DType::F32, vec![
            src1_batch as usize, m as usize, n as usize,
        ])
        .expect("alloc dst");

    let params = DenseMmBf16F32Params {
        m,
        n,
        k,
        src0_batch,
        src1_batch,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dense_matmul_bf16_f32_tensor(
        &mut encoder,
        &mut registry,
        &device,
        &src0_buf,
        &src1_buf,
        &mut dst_buf,
        &params,
    )
    .expect("dispatch dense_matmul_bf16_f32_tensor");
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
        max_err < tol,
        "max error {} > tolerance {} (m={} n={} k={} src0_b={} src1_b={})",
        max_err, tol, m, n, k, src0_batch, src1_batch
    );
}

#[test]
fn single_tile_32x64x32() {
    run_case(32, 64, 32, 1, 1, 1, 2, 1e-1);
}

#[test]
fn multi_tile_64x128x128() {
    run_case(64, 128, 128, 1, 1, 3, 4, 2e-1);
}

#[test]
fn partial_tile_35x67x64() {
    // Partial edge tiles exercise the shmem-copy write-back fallback.
    run_case(35, 67, 64, 1, 1, 5, 6, 2e-1);
}

#[test]
fn gqa_broadcast_r2_eq_4() {
    // 2 src0 batches, 8 src1 batches -> each src0 slice feeds 4 src1
    // slices.  Mirrors Gemma 4's nkv=4 / nh=16 group-query attention.
    run_case(32, 64, 64, 2, 8, 7, 8, 1e-1);
}

#[test]
fn prefill_attn_shape() {
    // Close-to-production shape: M=seq_q=128, N=seq_kv=128, K=hd=256,
    // 4 KV heads broadcast to 16 Q heads.
    run_case(128, 128, 256, 4, 16, 9, 10, 4e-1);
}

// ------------------------------------------------------------------------
// Partial K-tile coverage — K = ne00 not a multiple of NK=32. Without the
// kernel's `loop_k + NK <= args.ne00` guard, the in-tile unconditional
// 16-element / 8-element loads read past the end of the src0/src1
// buffers on the trailing partial tile, accumulating garbage into the
// output. Discovered externally by hf2q ADR-005 iter 67 on
// bge-small-en-v1.5 BERT embeddings (cosine 0.99999 at K=32 → 0.75-0.92
// at K=33-200). These tests lock the fix at the kernel level.
//
// The CPU reference does an exact F32 multiply-accumulate; the GPU path
// casts src0 to bfloat for the tensor-core matmul, so the tolerance
// reflects per-element BF16 round-off accumulated over K terms.

#[test]
fn partial_k_tile_k_eq_33() {
    // K=33: one full NK=32 tile + one 1-element trailing tile. The
    // simplest non-multiple-of-32 case. Pre-fix this test fails with
    // unbounded error from past-end reads; post-fix the trailing tile
    // is zero-padded and the matmul accumulates only valid k=0..32.
    run_case(32, 64, 33, 1, 1, 11, 12, 2e-1);
}

#[test]
fn partial_k_tile_k_eq_47() {
    // K=47: one full tile + a 15-element partial. Exercises every
    // intra-tile-position threshold (15 of 32 positions valid) so the
    // per-element gate has to evaluate true and false within a single
    // iteration's load.
    run_case(32, 64, 47, 1, 1, 13, 14, 2e-1);
}

#[test]
fn partial_k_tile_k_eq_63() {
    // K=63: one full tile + a 31-element partial. Off-by-one boundary
    // — only the very last position (k=31 within the trailing tile) is
    // out of range; the gate must zero exactly that single element.
    run_case(32, 64, 63, 1, 1, 15, 16, 2e-1);
}

#[test]
fn partial_k_tile_k_eq_72_vit_attention_path() {
    // K=72 mirrors Gemma 4 ViT's head_dim=72 attention `scores @ V`
    // (where K = seq_len = 49 also produces partial tiles, but K=72
    // is the more numerically-stressed first matmul in the chain).
    // Two full tiles + 8-element partial.
    run_case(64, 128, 72, 1, 1, 17, 18, 3e-1);
}

#[test]
fn partial_k_tile_k_eq_100_long_bert_seq() {
    // K=100 mirrors hf2q BERT `scores @ V` matmul on a 100-token
    // input — three full tiles + 4-element partial. End-to-end this
    // was the iter-67 cosine-fail regime.
    run_case(64, 128, 100, 1, 1, 19, 20, 3e-1);
}
