//! Correctness tests for `dense_matmul_bf16_f32_tensor`
//! (hf2q non-flash-attention prefill matmul).
//!
//! Cases:
//! 1. Short reductions K∈{1,2,3,4,8,16} — one zero-padded K tile.
//! 2. Single-tile shape (M=32, N=64, K=32).
//! 3. Multi-tile shape, single batch (M=64, N=128, K=128) — verifies the
//!    NK-loop and write-back paths.
//! 4. Partial-tile write-back (M=35, N=67, K=64) — exercises the
//!    shmem-copy fallback write-back path.
//! 5. GQA broadcast (src0_batch=2, src1_batch=8, r2=4) — verifies the
//!    head-broadcast offset math matches the CPU reference.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use half::bf16;
use mlx_native::ops::dense_bf16_auto::{dense_matmul_bf16_f32_forced, DenseBf16Route};
use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}

#[test]
fn tensor_and_simdgroup_reject_invalid_logical_contracts_before_encoding() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let params = DenseMmBf16F32Params {
        m: 2,
        n: 8,
        k: 32,
        src0_batch: 1,
        src1_batch: 1,
    };
    let weight = device
        .alloc_buffer((params.n * params.k * 2) as usize, DType::BF16, vec![256])
        .expect("weight");
    let input = device
        .alloc_buffer((params.m * params.k * 4) as usize, DType::F32, vec![64])
        .expect("input");
    let output = device
        .alloc_buffer((params.m * params.n * 4) as usize, DType::F32, vec![16])
        .expect("output");
    let short_weight = weight.slice_view(0, 255);
    let wrong_input = device
        .alloc_buffer(input.data_byte_len(), DType::BF16, vec![128])
        .expect("wrong input dtype");
    let oversized = DenseMmBf16F32Params {
        n: i32::MAX as u32 + 1,
        ..params
    };
    let misaligned_tensor_rows = DenseMmBf16F32Params { k: 33, ..params };

    for route in [DenseBf16Route::TensorV1, DenseBf16Route::Simdgroup] {
        for (candidate_weight, candidate_input, candidate_params) in [
            (&short_weight, &input, &params),
            (&weight, &wrong_input, &params),
            (&weight, &input, &oversized),
        ] {
            let mut encoder = device.command_encoder().expect("encoder");
            assert!(dense_matmul_bf16_f32_forced(
                route,
                &mut encoder,
                &mut registry,
                &device,
                candidate_weight,
                candidate_input,
                &output,
                candidate_params,
            )
            .is_err());
        }
    }

    let k33_weight = device
        .alloc_buffer(
            (params.n * 33 * 2) as usize,
            DType::BF16,
            vec![(params.n * 33) as usize],
        )
        .expect("K=33 weight");
    let k33_input = device
        .alloc_buffer(
            (params.m * 33 * 4) as usize,
            DType::F32,
            vec![(params.m * 33) as usize],
        )
        .expect("K=33 input");
    let mut encoder = device.command_encoder().expect("K=33 tensor encoder");
    assert!(dense_matmul_bf16_f32_forced(
        DenseBf16Route::TensorV1,
        &mut encoder,
        &mut registry,
        &device,
        &k33_weight,
        &k33_input,
        &output,
        &misaligned_tensor_rows,
    )
    .is_err());
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
        .alloc_buffer(
            src0_bytes,
            DType::BF16,
            vec![src0_batch as usize, n as usize, k as usize],
        )
        .expect("alloc src0");
    src0_buf
        .as_mut_slice::<u16>()
        .expect("write src0")
        .copy_from_slice(&src0_bf16);

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
        max_err,
        tol,
        m,
        n,
        k,
        src0_batch,
        src1_batch
    );
}

#[test]
fn single_tile_32x64x32() {
    run_case(32, 64, 32, 1, 1, 1, 2, 1e-1);
}

#[test]
fn one_zero_padded_tile_supports_short_reductions() {
    for (case, k) in [1, 2, 3, 4, 8, 16].into_iter().enumerate() {
        run_case(
            5,
            17,
            k,
            1,
            1,
            0x51A0 + case as u64,
            0xB170 + case as u64,
            5e-2,
        );
    }
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
// Partial K-tile coverage — K = ne00 not a multiple of NK=32. Auto routing
// uses the scalar-staging simdgroup kernel when K is not divisible by four,
// because Tensor V1's full-tile `float4` input loads require aligned row
// strides. Divisible-by-four partial tiles such as K=72 and K=100 continue to
// cover Tensor V1's gated tail. Without either kernel's bounds guard, the
// trailing tile reads past src0/src1 and accumulates garbage. Discovered on
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

/// Regression test for hf2q nomic-bert iter-79 cosine-parity bug.
///
/// `MlxBuffer::slice_view(byte_offset, n_elements)` documents that the
/// kernel sees memory starting at the slice's offset. Pre-fix
/// (encoder.rs:166), `KernelArg::Buffer(buf)` bound with hardcoded
/// offset=0 — silently exposing the entire underlying allocation
/// regardless of `buf.byte_offset()`. Symptom in hf2q: three Q/K/V
/// slices of a fused QKV weight all delivered Q's bytes to the matmul,
/// collapsing K and V projections onto Q's, gutting attention.
///
/// This test exercises the fixed contract directly:
///   1. Allocate a 3-block fused weight `[3*N, K]` with deterministic
///      content per block (block i filled with seed = i).
///   2. Slice block 1 (the middle block) via `slice_view(N*K*2,
///      N*K)`.
///   3. Run `dense_matmul_bf16_f32_tensor` with the SLICE as src0.
///   4. Compute the CPU reference for ONLY the middle block.
///   5. Assert max_err is within the standard bf16 tolerance.
///
/// Pre-fix this test would fail max_err ≈ 1.0+ (the kernel saw block
/// 0's bytes regardless of slice offset). Post-fix max_err ≤ 0.2.
#[test]
fn slice_view_kernel_arg_buffer_propagates_byte_offset() {
    use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};

    let (device, mut registry) = setup();
    let m = 32u32;
    let n = 64u32;
    let k = 32u32;
    let n_blocks = 3u32;

    // Three concatenated `[N, K]` blocks, each filled from a
    // distinct seed so they're guaranteed numerically distinct.
    let block_elems = (n * k) as usize;
    let mut fused_f32: Vec<f32> = Vec::with_capacity(n_blocks as usize * block_elems);
    let mut block_seeds = vec![];
    for b in 0..n_blocks {
        let seed = 0xC0FFEE_u64.wrapping_add((b as u64) * 31);
        block_seeds.push(seed);
        fused_f32.extend(pseudo_random_f32(seed, block_elems));
    }
    let fused_bf16: Vec<u16> = fused_f32
        .iter()
        .map(|&v| bf16::from_f32(v).to_bits())
        .collect();

    // Single fused buffer of 3 × N × K bf16 elements.
    let fused_bytes = (n_blocks as usize) * block_elems * 2;
    let mut fused_buf = device
        .alloc_buffer(
            fused_bytes,
            DType::BF16,
            vec![n_blocks as usize, n as usize, k as usize],
        )
        .expect("alloc fused");
    fused_buf
        .as_mut_slice::<u16>()
        .expect("write fused")
        .copy_from_slice(&fused_bf16);

    // Slice the MIDDLE block via slice_view at block-1's byte offset.
    let block_byte_off = (block_elems * 2) as u64; // 1 block × bf16
    let middle_slice = fused_buf.slice_view(block_byte_off, block_elems);

    // src1 (M=32, K=32) and dst (M=32, N=64).
    let src1_f32 = pseudo_random_f32(0xDEAD_C0DE, (m * k) as usize);
    let src1_bytes = (m * k) as usize * 4;
    let mut src1_buf = device
        .alloc_buffer(src1_bytes, DType::F32, vec![m as usize, k as usize])
        .expect("alloc src1");
    src1_buf
        .as_mut_slice::<f32>()
        .expect("write src1")
        .copy_from_slice(&src1_f32);

    let dst_bytes = (m * n) as usize * 4;
    let mut dst_buf = device
        .alloc_buffer(dst_bytes, DType::F32, vec![m as usize, n as usize])
        .expect("alloc dst");

    let params = DenseMmBf16F32Params {
        m,
        n,
        k,
        src0_batch: 1,
        src1_batch: 1,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dense_matmul_bf16_f32_tensor(
        &mut encoder,
        &mut registry,
        &device,
        &middle_slice, // ← slice_view, byte_offset = block_byte_off
        &src1_buf,
        &mut dst_buf,
        &params,
    )
    .expect("dispatch on slice_view");
    encoder.commit_and_wait().expect("commit_and_wait");

    // CPU reference: matmul against the MIDDLE block only.
    let middle_f32: &[f32] = &fused_f32[block_elems..2 * block_elems];
    let expected = cpu_ref(
        middle_f32, &src1_f32, m as usize, n as usize, k as usize, 1, 1,
    );
    let actual = dst_buf.as_slice::<f32>().expect("read dst").to_vec();
    let max_err = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Pre-fix: kernel sees block 0's bytes (offset ignored), max_err
    // is ~RMS of (block0 - block1) projection differences ≈ 1.0+.
    // Post-fix: kernel sees block 1's bytes (offset honored), max_err
    // is the standard bf16 tolerance.
    assert!(
        max_err < 1e-1,
        "slice_view byte_offset NOT propagated to dense_matmul: max_err = {}",
        max_err
    );

    // Sanity: also verify the WRONG-block matmul (using full buffer
    // pointing at block 0) gives a CLEARLY-different result, so the
    // test isn't trivially passing.
    let block0_f32: &[f32] = &fused_f32[..block_elems];
    let wrong_expected = cpu_ref(
        block0_f32, &src1_f32, m as usize, n as usize, k as usize, 1, 1,
    );
    let wrong_max_err = actual
        .iter()
        .zip(wrong_expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        wrong_max_err > 0.2,
        "block-0 vs block-1 outputs are too close — test is not discriminating: \
         wrong_max_err = {}",
        wrong_max_err
    );
}
