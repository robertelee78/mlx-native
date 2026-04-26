//! Correctness tests for `dense_matmul_f16_f32_tensor`
//! (hf2q ADR-005 Phase 2c iter-128 F16-staging GEMM, mirrors peer's
//! `kernel_mul_mm_f16_f32`).
//!
//! Cases mirror the BF16 sibling's regression suite shape-for-shape so
//! both kernels exercise the same partial-K, GQA-broadcast, and write-
//! back paths.  Tolerances are tightened ~5× vs BF16 because F16
//! retains 10-bit mantissa per shmem element vs BF16's 7-bit (8× more
//! precision per element of weight + activation, ~5× tighter on
//! accumulated K-error after the simdgroup reduction).
//!
//! Each test failing on its tightened tolerance proves the kernel is
//! actually using F16 staging (not silently downcasting to BF16); the
//! tolerance gap between this suite and the BF16 sibling's is the
//! mathematical gate that iter-128's hf2q ViT integration relies on.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::f16;
use mlx_native::ops::dense_mm_f16::{
    dense_matmul_f16_f32_tensor, DenseMmF16F32Params,
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
/// All math is exact F32 (the CPU reference does NOT mimic the GPU's
/// F16 staging — the test asserts the GPU is within F16's accumulated
/// round-off tolerance of the bit-exact F32 result).
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

    // Convert src0 f32 -> f16 via the half crate (bit-exact with
    // Metal's half cast).
    let src0_f16: Vec<u16> = src0_f32
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect();

    let src0_bytes = (src0_batch * n * k) as usize * 2;
    let mut src0_buf = device
        .alloc_buffer(src0_bytes, DType::F16, vec![
            src0_batch as usize, n as usize, k as usize,
        ])
        .expect("alloc src0");
    src0_buf
        .as_mut_slice::<u16>()
        .expect("write src0")
        .copy_from_slice(&src0_f16);

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

    let params = DenseMmF16F32Params {
        m,
        n,
        k,
        src0_batch,
        src1_batch,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dense_matmul_f16_f32_tensor(
        &mut encoder,
        &mut registry,
        &device,
        &src0_buf,
        &src1_buf,
        &mut dst_buf,
        &params,
    )
    .expect("dispatch dense_matmul_f16_f32_tensor");
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

// ----------------------------------------------------------------------
// Shape-by-shape parity with BF16 sibling, ~5× tighter tolerances.
// ----------------------------------------------------------------------

#[test]
fn single_tile_32x64x32() {
    // BF16 sibling: 1e-1.  F16 retains 10-bit mantissa vs BF16's 7-bit
    // (8× tighter per-element rounding); after K=32 Kahan-like
    // tensor-core reduction, accumulated error is ~5× tighter.
    run_case(32, 64, 32, 1, 1, 1, 2, 2e-2);
}

#[test]
fn multi_tile_64x128x128() {
    // BF16 sibling: 2e-1.
    run_case(64, 128, 128, 1, 1, 3, 4, 4e-2);
}

#[test]
fn partial_tile_35x67x64() {
    // Partial output-tile exercises the shmem-copy write-back fallback.
    // BF16 sibling: 2e-1.
    run_case(35, 67, 64, 1, 1, 5, 6, 4e-2);
}

#[test]
fn gqa_broadcast_r2_eq_4() {
    // 2 src0 batches, 8 src1 batches -> each src0 slice feeds 4 src1
    // slices.  Mirrors Gemma 4's nkv=4 / nh=16 GQA shape (vision tower
    // doesn't currently use GQA but the broadcast plumbing parallels
    // BF16 sibling's coverage and validates a future GQA vision tower).
    // BF16 sibling: 1e-1.
    run_case(32, 64, 64, 2, 8, 7, 8, 2e-2);
}

#[test]
fn prefill_attn_shape() {
    // Close-to-production shape: M=seq_q=128, N=seq_kv=128, K=hd=256,
    // 4 KV heads broadcast to 16 Q heads.  BF16 sibling: 4e-1.
    run_case(128, 128, 256, 4, 16, 9, 10, 8e-2);
}

// ------------------------------------------------------------------------
// Partial K-tile coverage — K = ne00 not a multiple of NK=32. Without the
// kernel's `loop_k + NK <= args.ne00` guard, the in-tile unconditional
// 16-element / 8-element loads read past the end of the src0/src1
// buffers on the trailing partial tile, accumulating garbage into the
// output. Same regression as BF16 sibling's iter-67 lock-in.

#[test]
fn partial_k_tile_k_eq_33() {
    // K=33: one full NK=32 tile + one 1-element trailing tile.
    // BF16 sibling: 2e-1.
    run_case(32, 64, 33, 1, 1, 11, 12, 4e-2);
}

#[test]
fn partial_k_tile_k_eq_47() {
    // K=47: one full tile + a 15-element partial.
    // BF16 sibling: 2e-1.
    run_case(32, 64, 47, 1, 1, 13, 14, 4e-2);
}

#[test]
fn partial_k_tile_k_eq_63() {
    // K=63: one full tile + a 31-element partial.
    // BF16 sibling: 2e-1.
    run_case(32, 64, 63, 1, 1, 15, 16, 4e-2);
}

#[test]
fn partial_k_tile_k_eq_72_vit_attention_path() {
    // K=72 mirrors Gemma 4 ViT's head_dim=72 attention `scores @ V`
    // shape — two full tiles + 8-element partial.
    // BF16 sibling: 3e-1.
    run_case(64, 128, 72, 1, 1, 17, 18, 6e-2);
}

#[test]
fn partial_k_tile_k_eq_100_long_bert_seq() {
    // K=100 — three full tiles + 4-element partial.
    // BF16 sibling: 3e-1.
    run_case(64, 128, 100, 1, 1, 19, 20, 6e-2);
}

// ------------------------------------------------------------------------
// gemma4v ViT production shapes — the actual call sites that iter-128
// closes the cascade on (vit_linear_gpu QKV/O at hidden=1152, ffn at
// gate/up/down).

#[test]
fn gemma4v_attn_qkv_shape_seq_2304_hidden_1152() {
    // M=seq=196 (W57 baseline single-image; full 2304 grid would OOM
    // the test runner), N=hidden=1152, K=hidden=1152.  Exercises the
    // production attention QKV/O matmul shape per
    // `vit_attention_block_forward_gpu` in vit_gpu.rs (Gemma 4
    // SigLIP, 27 blocks).
    //
    // F16 tolerance: K=1152 → ~36 NK-tiles accumulated error;
    // ~5e-2 is conservative given F16's 10-bit mantissa.
    run_case(196, 1152, 1152, 1, 1, 21, 22, 1e-1);
}

#[test]
fn gemma4v_ffn_gate_up_shape_hidden_1152_inter_4304() {
    // M=seq=196, N=intermediate=4304, K=hidden=1152.  Gemma 4 ViT
    // FFN gate / up projection shape; hf2q dispatches both via
    // `vit_linear_gpu`.  Heaviest production matmul (largest N).
    run_case(196, 4304, 1152, 1, 1, 23, 24, 2e-1);
}
