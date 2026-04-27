//! Integration tests for the flash-attention prefill kernel.
//!
//! Tests the `attention<BQ, BK, BD, WM, WN>` kernel in
//! `src/shaders/flash_attn_prefill.metal` against a CPU reference that
//! mirrors the exact mathematical idioms used in the GPU shader (base-2
//! softmax, llama.cpp finite-M-sentinel convention with ONE output-side
//! guard at the final normalisation — see kernel preamble and
//! ADR-011-phase2-port-sentinel.md).
//!
//! # Test coverage
//!
//! 1. Structural — `AttnParamsGpu` layout (size = 160 bytes, field offsets).
//! 2. Library compilation — the bf16 D=256 pipeline is obtainable.
//! 3. Error paths — invalid head_dim, invalid GQA ratio, undersized buffers,
//!    wrong dtype.
//! 4. CPU reference correctness — CPU-only self-consistency between the
//!    `sdpa_reference_f32` (kernel-equivalent) and `sdpa_naive_scalar_f32`
//!    (textbook-equivalent) implementations, plus a NaN-guard test on a
//!    fully-masked input.
//! 5. GPU correctness at bf16 D=256 — the production dispatcher path.
//!    Unmasked, causal, additive-mask, fully-masked NaN-guard, GQA, custom
//!    scale, determinism, and unaligned-ql / unaligned-kl scenarios.
//!
//! # Why bf16 and not f32 for GPU tests
//!
//! At D=256 with our tile geometry (BQ=32, BK=16) the f32 Qs tile is
//! `32 * 256 * 4 = 32 KB` — the entire Apple Silicon threadgroup memory
//! budget, before KV_smem (~16 KB more), scratch, and padding.  Empirically
//! verified on M5 Max: f32 D=256 requires ~53.7 KB and fails at library
//! compilation.  No M-series chip can run an f32 D=256 attention kernel at
//! this tile shape — M1..M5 all share the 32 KB
//! `MTLDevice.maxThreadgroupMemoryLength` limit.
//!
//! See `ADR-011-phase1-port-source-decision.md` §3 for the full
//! threadgroup-memory analysis.
//!
//! Phase 1a GPU correctness therefore runs at bf16 (~29 KB, fits within
//! the budget).  f32 correctness is verified at the CPU reference layer
//! only.
//!
//! # CPU reference design
//!
//! `sdpa_reference_f32` mirrors the GPU kernel idiom step-for-step:
//!
//! - Q is pre-scaled by `scale * log2(e)` before QK^T (matches the kernel's
//!   `TransformScale` applied to Q on load).
//! - Softmax uses `f32::exp2(x - max)` (matches `ExpSubOp::apply`).
//! - Row max `max_score` initialised to `f32::MIN / 2.0` = `-FLT_MAX/2`,
//!   the llama.cpp finite sentinel (ggml-metal.metal:5891).  This lets
//!   `exp2(score - max_score)` evaluate cleanly for any masked score
//!   (`exp2(-inf - finite) = 0.0`, IEEE-754 exact) without an intermediate
//!   NaN guard.
//! - ONE output-side guard at the final normalisation:
//!   `safe_sum = sum_exp == 0 ? 1 : sum_exp` — mirrors llama.cpp's
//!   `scale = S == 0 ? 0 : 1/S` at ggml-metal.metal:6358, which handles
//!   fully-masked rows where every exp is bit-exact 0.
//! - Additive (float) mask is multiplied by `log2(e)` before being added to
//!   the score, matching the kernel's mask-addition step.
//! - Causal masking: positions with `k_pos > q_abs_pos` get scores set to
//!   -inf before exp (the mask sentinel; absorbed by the finite
//!   `max_score` via the max() reduction).
//!
//! # Tolerance
//!
//! CPU-only self-consistency (reference vs naive, both in f32): `atol=5e-5`.
//!
//! GPU bf16 vs CPU reference f32: bf16 has a 7-bit mantissa, so per-element
//! round-off on inputs is bounded by `2^-7 * |x| ≈ 7.8e-3 * |x|`.  The
//! kernel's MMA accumulators are f32 internally (the `T_accum` template
//! parameter is `float` for every instantiation) so accumulation does not
//! widen further — only the final bf16 store at output rounds.
//! `atol=5e-3, rtol=2e-2` captures both input-cast and output-cast precision
//! loss at typical attention output magnitudes.  On bursty residuals
//! (outputs near the dtype edge) this may need to relax to 1e-2 — update
//! on observation, not speculation.

// ─── macOS guard ─────────────────────────────────────────────────────────────
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)] // test helpers mirror dispatcher signatures
mod flash_attn_prefill_tests {

use half::bf16;
use mlx_native::ops::flash_attn_prefill::{
    self, AttnParamsGpu, AttnMaskParamsGpu, FlashAttnPrefillParams,
    dispatch_flash_attn_prefill_bf16_d256,
};
use mlx_native::ops::flash_attn_prefill_d512::{
    self as flash_attn_prefill_d512,
    dispatch_flash_attn_prefill_bf16_d512,
};
use mlx_native::ops::flash_attn_prefill_mask::{
    self as flash_attn_prefill_mask,
    build_sdpa_mask_bf16, SdpaMaskParams,
};
use mlx_native::ops::flash_attn_prefill_blk::{
    self as flash_attn_prefill_blk,
    dispatch_flash_attn_prefill_blk, alloc_blk_buffer, BlkParams,
};
use mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_bf16_d256_with_blk;
use mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_bf16_d512_with_blk;
use mlx_native::{DType, KernelRegistry, MlxDevice, MlxError};

// ─────────────────────────────────────────────────────────────────────────────
// § 1  STRUCTURAL VERIFICATION
// ─────────────────────────────────────────────────────────────────────────────

/// Verify `AttnParamsGpu` is exactly 160 bytes.
///
/// The MSL `AttnParams` struct has the same layout as the Rust repr(C)
/// mirror.  If any field is added, reordered, or its type changed, the
/// layout drifts from the MSL struct and the kernel silently reads garbage
/// — CI must fail loudly.
///
/// Layout:
///   B(4) H(4) D(4) qL(4) kL(4) gqa(4) scale(4) softcap(4)   = 32
///   NQ(4) NK(4) NQ_aligned(4) NK_aligned(4)                    = 16
///   qL_rem(4) kL_rem(4) qL_off(4) _pad(4)                      = 16
///   Q_strides(24) K_strides(24) V_strides(24) O_strides(24)    = 96
///   Total                                                        = 160
#[test]
fn test_attn_params_gpu_layout() {
    assert_eq!(
        std::mem::size_of::<AttnParamsGpu>(),
        160,
        "AttnParamsGpu must be exactly 160 bytes to match MSL AttnParams layout"
    );

    // Spot-check field offsets using pointer arithmetic.
    // This catches field-reorder regressions that keep the total size the same.
    let dummy = AttnParamsGpu {
        b: 0, h: 0, d: 0, ql: 0, kl: 0, gqa_factor: 0,
        scale: 0.0, softcapping: 0.0,
        nq: 0, nk: 0, nq_aligned: 0, nk_aligned: 0,
        ql_rem: 0, kl_rem: 0, ql_off: 0, _pad: 0,
        q_strides: [0; 3], k_strides: [0; 3], v_strides: [0; 3], o_strides: [0; 3],
    };
    let base = &dummy as *const AttnParamsGpu as usize;

    // `scale` is at offset 24 (after 6 × i32).
    let scale_offset = &dummy.scale as *const f32 as usize - base;
    assert_eq!(scale_offset, 24, "AttnParamsGpu::scale must be at offset 24");

    // `nq` is at offset 32 (after 6×i32 + 2×f32).
    let nq_offset = &dummy.nq as *const i32 as usize - base;
    assert_eq!(nq_offset, 32, "AttnParamsGpu::nq must be at offset 32");

    // `_pad` is at offset 60.
    let pad_offset = &dummy._pad as *const i32 as usize - base;
    assert_eq!(pad_offset, 60, "AttnParamsGpu::_pad must be at offset 60");

    // `q_strides` is at offset 64 (after 16 × i32 = 64 bytes).
    let qs_offset = dummy.q_strides.as_ptr() as usize - base;
    assert_eq!(qs_offset, 64, "AttnParamsGpu::q_strides must be at offset 64");

    // `k_strides` is at offset 88 (64 + 3×8).
    let ks_offset = dummy.k_strides.as_ptr() as usize - base;
    assert_eq!(ks_offset, 88, "AttnParamsGpu::k_strides must be at offset 88");

    // `v_strides` is at offset 112.
    let vs_offset = dummy.v_strides.as_ptr() as usize - base;
    assert_eq!(vs_offset, 112, "AttnParamsGpu::v_strides must be at offset 112");

    // `o_strides` is at offset 136.
    let os_offset = dummy.o_strides.as_ptr() as usize - base;
    assert_eq!(os_offset, 136, "AttnParamsGpu::o_strides must be at offset 136");
}

/// Verify `AttnMaskParamsGpu` is exactly 24 bytes (3 × i64).
#[test]
fn test_attn_mask_params_gpu_layout() {
    assert_eq!(
        std::mem::size_of::<AttnMaskParamsGpu>(),
        24,
        "AttnMaskParamsGpu must be exactly 24 bytes (3 × i64)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  SHADER LIBRARY LOADS SUCCESSFULLY
// ─────────────────────────────────────────────────────────────────────────────

/// Verify the flash_attn_prefill.metal library compiles and the bf16 D=256
/// pipeline is obtainable at the canonical function-constant combination.
///
/// # Background
///
/// The f32 D=256 instantiation was removed from the shader because its
/// Qs threadgroup tile (32×256×4 = 32 KB) consumes the entire Apple Silicon
/// threadgroup memory budget before KV_smem or scratch — the library failed
/// to compile as long as the f32 instantiation remained.  This test is the
/// regression gate: if f32 instantiations are re-added, the library will
/// again fail to compile and this test will fail with
/// `MlxError::ShaderCompilationError`.
#[test]
fn test_bf16_d256_library_compiles() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    // Fetch the pipeline at the canonical function-constant combination for
    // the unmasked, non-causal, aligned bf16 D=256 path.
    let result = registry.get_pipeline_with_bool_constants(
        "flash_attn_prefill_bf16_d256",
        device.metal_device(),
        &[(200, true), (201, true), (300, false), (301, false)],
    );

    match result {
        Ok(_pipeline) => {
            eprintln!("test_bf16_d256_library_compiles: OK — library + pipeline compiled");
        }
        Err(MlxError::ShaderCompilationError { name, message }) => {
            panic!(
                "bf16 D=256 pipeline compilation failed — if this mentions threadgroup \
                 memory, an f32 instantiation likely re-entered the shader. \
                 name={name}, message={message}"
            );
        }
        Err(other) => {
            panic!("Unexpected error compiling bf16 D=256 pipeline: {other:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  ERROR PATH TESTS (validation before GPU, no shader needed)
// ─────────────────────────────────────────────────────────────────────────────

/// Set up device + registry (without attempting pipeline compilation).
fn setup_no_gpu() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new should succeed on Apple Silicon");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    (device, registry)
}

/// Allocate a BF16 buffer of `elems` elements (2 bytes each), zero-initialised.
fn alloc_bf16(device: &MlxDevice, elems: usize, name: &str) -> mlx_native::MlxBuffer {
    device
        .alloc_buffer(elems * 2, DType::BF16, vec![elems])
        .unwrap_or_else(|e| panic!("alloc_buffer({name}, {elems}) failed: {e:?}"))
}

/// Error path: head_dim != 256 must return MlxError::InvalidArgument.
///
/// This validation fires BEFORE any GPU resource is created.
#[test]
fn test_error_wrong_head_dim_128() {
    let (device, mut registry) = setup_no_gpu();

    let params = FlashAttnPrefillParams {
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 128, // wrong: this dispatcher is D=256 only
        seq_len_q: 32,
        seq_len_k: 32,
        batch: 1,
        scale: 1.0,
        do_causal: false,
    };

    let q_elems = 2 * 32 * 128;
    let kv_elems = 2 * 32 * 128;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = dispatch_flash_attn_prefill_bf16_d256(
        &mut encoder, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, None, &mut out_buf, &params,
    );

    assert!(result.is_err(), "Expected error for head_dim=128, got Ok");
    match result {
        Err(MlxError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("256") || msg.contains("head_dim"),
                "Error message should reference head_dim or 256: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got: {other:?}"),
    }
}

/// Error path: head_dim = 512 must return MlxError::InvalidArgument.
#[test]
fn test_error_wrong_head_dim_512() {
    let (device, mut registry) = setup_no_gpu();

    let params = FlashAttnPrefillParams {
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 512,
        seq_len_q: 8,
        seq_len_k: 8,
        batch: 1,
        scale: 1.0,
        do_causal: false,
    };

    let elems = 2 * 8 * 512;
    let buf = alloc_bf16(&device, elems, "buf");
    let mut out = alloc_bf16(&device, elems, "out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = dispatch_flash_attn_prefill_bf16_d256(
        &mut encoder, &device, &mut registry,
        &buf, &buf, &buf, None, &mut out, &params,
    );

    assert!(result.is_err(), "Expected error for head_dim=512, got Ok");
    match result {
        Err(MlxError::InvalidArgument(_)) => {}
        other => panic!("Expected InvalidArgument, got: {other:?}"),
    }
}

/// Error path: n_heads not divisible by n_kv_heads (invalid GQA ratio).
///
/// n_heads=3, n_kv_heads=2 → not integer divisible → must reject.
#[test]
fn test_error_invalid_gqa_ratio() {
    let (device, mut registry) = setup_no_gpu();

    let params = FlashAttnPrefillParams {
        n_heads: 3,
        n_kv_heads: 2,
        head_dim: 256,
        seq_len_q: 32,
        seq_len_k: 32,
        batch: 1,
        scale: 1.0,
        do_causal: false,
    };

    let q_elems = 3 * 32 * 256;
    let kv_elems = 2 * 32 * 256;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = dispatch_flash_attn_prefill_bf16_d256(
        &mut encoder, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, None, &mut out_buf, &params,
    );

    assert!(result.is_err(), "Expected error for invalid GQA ratio, got Ok");
    match result {
        Err(MlxError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("divisible") || msg.contains("kv_heads"),
                "Error message should mention GQA/divisibility: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got: {other:?}"),
    }
}

/// Error path: Q buffer dtype != BF16 must be rejected.
///
/// The bf16 dispatcher rejects any non-BF16 buffer (F32, F16, ...) before
/// pipeline compilation.  This is the direct counterpart of the removed
/// f32 path: calling the bf16 dispatcher with f32 buffers must fail cleanly.
#[test]
fn test_error_wrong_dtype_f32() {
    let (device, mut registry) = setup_no_gpu();

    let params = FlashAttnPrefillParams {
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 256,
        seq_len_q: 32,
        seq_len_k: 32,
        batch: 1,
        scale: 1.0 / (256.0_f32).sqrt(),
        do_causal: false,
    };

    let q_elems = 2 * 32 * 256;
    let kv_elems = 2 * 32 * 256;
    // F32 buffers (wrong dtype — dispatcher requires BF16).
    let q_buf = device.alloc_buffer(q_elems * 4, DType::F32, vec![q_elems]).expect("Q");
    let k_buf = device.alloc_buffer(kv_elems * 4, DType::F32, vec![kv_elems]).expect("K");
    let v_buf = device.alloc_buffer(kv_elems * 4, DType::F32, vec![kv_elems]).expect("V");
    let mut out_buf = device.alloc_buffer(q_elems * 4, DType::F32, vec![q_elems]).expect("out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = dispatch_flash_attn_prefill_bf16_d256(
        &mut encoder, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, None, &mut out_buf, &params,
    );

    assert!(result.is_err(), "Expected error for F32 buffers, got Ok");
    match result {
        Err(MlxError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("BF16") || msg.contains("dtype"),
                "Error message should mention BF16 or dtype: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got: {other:?}"),
    }
}

/// Error path: Q buffer too small for declared shape.
///
/// Allocate a Q buffer sized for seq_len=16 but declare seq_len=32.
/// The dispatcher validates buffer sizes BEFORE attempting pipeline compilation.
#[test]
fn test_error_q_buffer_too_small() {
    let (device, mut registry) = setup_no_gpu();

    let params = FlashAttnPrefillParams {
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 256,
        seq_len_q: 32,
        seq_len_k: 32,
        batch: 1,
        scale: 1.0,
        do_causal: false,
    };

    // Q buffer sized for seq_len=16, but params says seq_len=32.
    let q_small_elems = 2 * 16 * 256;
    let kv_elems = 2 * 32 * 256;
    let q_buf = alloc_bf16(&device, q_small_elems, "Q (small)");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let q_elems = 2 * 32 * 256;
    let mut out_buf = alloc_bf16(&device, q_elems, "out");

    let mut encoder = device.command_encoder().expect("encoder");
    let result = dispatch_flash_attn_prefill_bf16_d256(
        &mut encoder, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, None, &mut out_buf, &params,
    );

    assert!(result.is_err(), "Expected error for undersized Q buffer, got Ok");
    match result {
        Err(MlxError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("Q") || msg.contains("buffer") || msg.contains("small"),
                "Error message should mention Q or buffer size: {msg}"
            );
        }
        other => panic!("Expected InvalidArgument, got: {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  CPU REFERENCE SELF-CONSISTENCY TESTS
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests validate the CPU reference `sdpa_reference_f32` without
// touching the GPU.
//
// The CPU reference uses the kernel's exact math idioms:
// - Pre-scale Q by `scale * log2(e)` (matches the kernel's TransformScale).
// - Softmax via `f32::exp2(x - max)` (matches the kernel's `ExpSubOp`).
// - NaN guards on exp and rescale factor (kernel's guards #1 and #2).
// - Additive mask multiplied by `log2(e)` before being added to the score.
// - Causal masking: k_pos > q_abs_pos → score = -inf.
//
// Self-consistency is verified by comparing `sdpa_reference_f32` against
// a naive exp-based scalar SDPA at the same input.  With f32 precision and
// small head_dim, both should agree to within ~1e-5 (accumulation rounding).

/// Naive SDPA using `exp()` and standard scale placement (NOT the kernel's
/// base-2 idioms).
///
/// Used as an independent ground-truth to validate the kernel-flavored CPU
/// reference (`sdpa_reference_f32`).  At f32 precision, `exp(x)` ≈
/// `exp2(x * log2(e))` to within ~1 ULP, so the two implementations should
/// produce very close outputs (not byte-identical but within atol=5e-5).
fn sdpa_naive_scalar_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> Vec<f32> {
    let heads_per_kv = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; batch * n_heads * ql * head_dim];

    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / heads_per_kv;
            for q_pos in 0..ql {
                let q_base = b * n_heads * ql * head_dim + h * ql * head_dim + q_pos * head_dim;
                let kv_base = b * n_kv_heads * kl * head_dim + kv_h * kl * head_dim;

                // Standard QK^T * scale (no log2(e) pre-scale).
                let mut scores = vec![0.0f32; kl];
                for (k_pos, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[kv_base + k_pos * head_dim + d];
                    }
                    *score = dot * scale;
                }

                // Additive mask (natural-log scale, added directly, no log2(e) factor).
                if let Some(m) = mask {
                    let m_base = b * n_heads * ql * kl + h * ql * kl + q_pos * kl;
                    for k_pos in 0..kl {
                        scores[k_pos] += m[m_base + k_pos];
                    }
                }

                // Causal masking.
                if do_causal {
                    let q_abs = kl.saturating_sub(ql) + q_pos;
                    for (k_pos, score) in scores.iter_mut().enumerate() {
                        if k_pos > q_abs {
                            *score = f32::NEG_INFINITY;
                        }
                    }
                }

                // Standard numerically-stable softmax using exp() — updated
                // to the llama.cpp finite-sentinel convention for parity with
                // the kernel-equivalent reference.  For any non-fully-masked
                // input, `max_s` is the real row-max (identical to the
                // previous NEG_INFINITY init, which would have been overwritten
                // by the first score).  For a fully-masked row, max_s stays
                // at -FLT_MAX/2, `exp(-inf - -FLT_MAX/2) = exp(-inf) = 0.0`
                // (IEEE-754 exact), sum = 0, safe_sum = 1, output = 0 — same
                // final answer as the previous branch-guarded form.
                let mut max_s = f32::MIN / 2.0; // = -FLT_MAX/2 sentinel (llama.cpp ggml-metal.metal:5891)
                for &s in &scores {
                    if s > max_s {
                        max_s = s;
                    }
                }
                let exp_scores: Vec<f32> = scores
                    .iter()
                    .map(|&s| (s - max_s).exp())
                    .collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                // Mirror of llama.cpp's `S == 0 ? 0 : 1/S` at ggml-metal.metal:6358.
                let safe_sum = if sum_exp == 0.0 { 1.0 } else { sum_exp };

                let o_base = b * n_heads * ql * head_dim + h * ql * head_dim + q_pos * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for k_pos in 0..kl {
                        acc += (exp_scores[k_pos] / safe_sum) * v[kv_base + k_pos * head_dim + d];
                    }
                    out[o_base + d] = acc;
                }
            }
        }
    }

    out
}

/// CPU reference that mirrors the GPU kernel math exactly.
///
/// Uses base-2 softmax with the kernel's three NaN guards.  See module doc
/// for the full design.
fn sdpa_reference_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> Vec<f32> {
    // log2(e) = 1 / ln(2) ≈ 1.44269504089.
    // flash_attn_prefill.metal:1186 — `params->scale * 1.44269504089`.
    // Use std's canonical constant (clippy::approximate_constants).
    const LOG2E: f32 = std::f32::consts::LOG2_E;

    // Pre-scale factor applied to Q before QK^T:
    // flash_attn_prefill.metal:1186 — TransformScale<T> ts(scale * log2(e))
    // flash_attn_prefill.metal:1221 — loader_q.apply_inplace_op(ts)
    let q_scale = scale * LOG2E;

    let heads_per_kv = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; batch * n_heads * ql * head_dim];

    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / heads_per_kv; // GQA: kv_head_idx = simd_group_id / gqa_factor

            for q_pos in 0..ql {
                let q_base = b * n_heads * ql * head_dim + h * ql * head_dim + q_pos * head_dim;
                let kv_base = b * n_kv_heads * kl * head_dim + kv_h * kl * head_dim;

                // ── Q·K^T (base-2 scale via q_scale = scale * log2(e)) ────
                // flash_attn_prefill.metal:1282 — tile_matmad(Stile, Qtile, Ktile, Stile)
                let mut scores = vec![0.0f32; kl];
                for k_pos in 0..kl {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[kv_base + k_pos * head_dim + d];
                    }
                    // Q pre-scaled by scale * log2(e), so dot product is already in base-2 scale.
                    // flash_attn_prefill.metal:1186,1221
                    scores[k_pos] = dot * q_scale;
                }

                // ── Additive mask (float, natural-log scale → base-2 via × log2(e)) ──
                // flash_attn_prefill.metal:1378 — `Stile[jj] += 1.44269504089 * mfrag[jj]`
                if let Some(m) = mask {
                    let m_base = b * n_heads * ql * kl + h * ql * kl + q_pos * kl;
                    for k_pos in 0..kl {
                        // Mask in natural-log scale, multiplied by log2(e) for base-2 space.
                        scores[k_pos] += LOG2E * m[m_base + k_pos];
                    }
                }

                // ── Causal masking ─────────────────────────────────────────
                // Mirrors the kernel's in-template causal guard: positions
                // with k_pos > q_abs_pos receive a score of -inf before exp.
                if do_causal {
                    let q_abs = kl.saturating_sub(ql) + q_pos;
                    for (k_pos, score) in scores.iter_mut().enumerate() {
                        if k_pos > q_abs {
                            *score = f32::NEG_INFINITY;
                        }
                    }
                }

                // ── Online softmax (base-2, llama.cpp finite-M regime) ────
                // Row max reduction:
                // flash_attn_prefill.metal — row_reduce<MaxOp>(new_max) with
                // max_score init = -FLT_MAX/2 (llama.cpp convention,
                // ggml-metal.metal:5891).  A finite sentinel absorbs -inf
                // scores via max() without ever letting max_score become
                // -inf, so the subsequent exp2(score - max) path never sees
                // exp2(-inf - -inf) = exp2(NaN).  See
                // ADR-011-phase2-port-sentinel.md §1.3.
                let mut max_score = f32::MIN / 2.0; // = -FLT_MAX/2, matches llama.cpp + our kernel
                for &s in &scores {
                    if s > max_score {
                        max_score = s;
                    }
                }

                // exp2(score - max) unguarded: max_score is finite by
                // construction (simd_max of -FLT_MAX/2 floor and real
                // scores), so exp2(-inf - finite) = exp2(-inf) = +0.0
                // (IEEE-754 exact) for any masked position, never NaN.
                // Matches llama.cpp's unguarded `exp(s2 - M[jj])` at
                // ggml-metal.metal:6156.
                let exp_scores: Vec<f32> = scores
                    .iter()
                    .map(|&s| f32::exp2(s - max_score))
                    .collect();

                // Sum: flash_attn_prefill.metal — row_reduce<SumOp>
                let sum_exp: f32 = exp_scores.iter().sum();
                // THE single surviving guard — mirrors llama.cpp's
                // `scale = S == 0 ? 0 : 1/S` at ggml-metal.metal:6358.
                // For a fully-masked row every exp is 0 so sum_exp = 0;
                // downstream the weighted sum becomes (exp / safe_sum) = 0
                // and `* 0` when sum_exp was 0, i.e. 0 output.  We keep
                // the `safe_sum = 1` form here because the weighted loop
                // below does `(exp / safe_sum) * v`: when sum_exp == 0
                // every `exp` is also 0, so `0 / 1 * v = 0` — same final
                // behaviour as llama.cpp's `reciprocal-then-multiply`.
                let safe_sum = if sum_exp == 0.0 { 1.0 } else { sum_exp };

                // ── Weighted sum of V ─────────────────────────────────────
                // flash_attn_prefill.metal:1444-1457 — MMAFrag_acc_t::mma(Otile, Stile, Vtile)
                let o_base = b * n_heads * ql * head_dim + h * ql * head_dim + q_pos * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for k_pos in 0..kl {
                        acc += (exp_scores[k_pos] / safe_sum) * v[kv_base + k_pos * head_dim + d];
                    }
                    out[o_base + d] = acc;
                }
            }
        }
    }

    out
}

// ─── LCG PRNG (identical to test_sdpa.rs for cross-test seed consistency) ───

/// Fixed seed for all tests — encodes "ADR-011" symbolically.
/// 0xAD0011 = 11337745 decimal.
const SEED: u64 = 0x00AD_0011_u64;

/// LCG pseudo-random f32 in `[-0.5, 0.5]` — same algorithm as `test_sdpa.rs`.
fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Assert elementwise closeness between two f32 slices.
fn assert_close_cpu(actual: &[f32], expected: &[f32], atol: f32, test_name: &str) {
    assert_eq!(actual.len(), expected.len(), "{test_name}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff <= atol,
        "{test_name}: max_abs_error={max_diff:.3e} at index {max_idx} \
         (reference={:.6}, naive={:.6}) exceeds atol={atol:.3e}",
        actual[max_idx],
        expected[max_idx]
    );
    eprintln!("{test_name}: max_abs_error={max_diff:.3e} (budget: atol={atol:.3e})");
}

/// Assert all elements are finite (no NaN, no Inf).
fn assert_all_finite(data: &[f32], test_name: &str) {
    let nans: Vec<usize> = data.iter().enumerate().filter(|(_, v)| v.is_nan()).map(|(i, _)| i).collect();
    let infs: Vec<usize> = data.iter().enumerate().filter(|(_, v)| v.is_infinite()).map(|(i, _)| i).collect();
    assert!(nans.is_empty(), "{test_name}: output has {} NaN(s), first at index {}", nans.len(), nans[0]);
    assert!(infs.is_empty(), "{test_name}: output has {} Inf(s), first at index {}", infs.len(), infs[0]);
}

// CPU self-consistency tolerance: exp vs exp2 differ by ~1-2 ULP at f32.
// For small head_dims the error compounds less; use 5e-5 as the ceiling.
const CPU_SELF_CONSISTENCY_ATOL: f32 = 5e-5;

/// CPU self-consistency: reference vs naive, unmasked, no causal.
///
/// Verifies the kernel-equivalent reference produces the same output as the
/// naive `exp`-based reference for unmasked attention.  Any discrepancy
/// > 5e-5 means the reference has a bug in the math translation.
#[test]
fn test_cpu_ref_self_consistency_unmasked() {
    let batch = 1; let h = 2; let kv_h = 2;
    let ql = 32; let kl = 32; let d = 64; // small d to make CPU fast
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 1, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 2, batch * kv_h * kl * d);

    let ref_out = sdpa_reference_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, false,
    );
    let naive_out = sdpa_naive_scalar_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_cpu(&ref_out, &naive_out, CPU_SELF_CONSISTENCY_ATOL, "cpu_self_unmasked");
}

/// CPU self-consistency: causal masking.
///
/// With causal masking, the first Q positions have very few attended keys.
/// The NaN guard paths in the reference may be exercised depending on inputs.
#[test]
fn test_cpu_ref_self_consistency_causal() {
    let batch = 1; let h = 2; let kv_h = 2;
    let ql = 32; let kl = 32; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 10, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 11, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 12, batch * kv_h * kl * d);

    let ref_out = sdpa_reference_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, true,
    );
    let naive_out = sdpa_naive_scalar_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, true,
    );

    assert_close_cpu(&ref_out, &naive_out, CPU_SELF_CONSISTENCY_ATOL, "cpu_self_causal");
}

/// CPU self-consistency: additive mask (checkerboard pattern).
///
/// Tests the mask path: the kernel-equivalent reference multiplies mask
/// values by log2(e) (because Q is base-2 pre-scaled), while the naive
/// reference adds them directly in natural-log space.  These are not
/// equivalent — for mask=-0.1: reference adds -0.1 * 1.44269 = -0.14427,
/// naive adds -0.1.  This test therefore checks only that both produce
/// finite output (no NaN/Inf), not closeness against each other.
#[test]
fn test_cpu_ref_self_consistency_additive_mask() {
    let batch = 1; let h = 2; let kv_h = 2;
    let ql = 16; let kl = 16; let d = 32;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 20, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 21, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 22, batch * kv_h * kl * d);

    // Small additive bias so the log2(e) factor doesn't dominate.
    let mask: Vec<f32> = (0..batch * h * ql * kl)
        .map(|i| if i % 2 == 0 { 0.0f32 } else { -0.1f32 })
        .collect();

    let ref_out = sdpa_reference_f32(
        &q, &k, &v, Some(&mask), batch, h, kv_h, ql, kl, d, scale, false,
    );
    let naive_out = sdpa_naive_scalar_f32(
        &q, &k, &v, Some(&mask), batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_all_finite(&ref_out, "cpu_self_mask_reference");
    assert_all_finite(&naive_out, "cpu_self_mask_naive");
    eprintln!("test_cpu_ref_self_consistency_additive_mask: PASS (both refs produce finite output)");
}

/// CPU fully-masked input must produce zero output (llama.cpp sentinel regime).
///
/// Under the llama.cpp finite-M convention (ADR-011-phase2-port-sentinel.md),
/// when ALL K positions are masked to -infinity the fully-masked-row path is:
///
/// 1. Row max `max_score` initialised to `-FLT_MAX/2` (finite).  Every score
///    is `-inf` from the mask, so `max(-FLT_MAX/2, -inf) = -FLT_MAX/2`:
///    max_score stays finite.  No NaN.
/// 2. `exp2(score - max_score) = exp2(-inf - -FLT_MAX/2) = exp2(-inf) = 0.0`
///    (IEEE-754 exact).  No NaN.
/// 3. `sum_exp = sum of zeros = 0`.  Output-side guard
///    (`safe_sum = sum_exp == 0 ? 1 : sum_exp`) sets divisor to 1.
/// 4. `(0 / 1) * v = 0` per output component → output = all 0.0.
///
/// Mirrors llama.cpp's end-to-end trace at ggml-metal.metal:5888-6374 for a
/// fully-masked row (final `scale = S == 0 ? 0 : 1/S` at :6358 yields 0).
#[test]
fn test_cpu_ref_nan_guard_fully_masked() {
    let batch = 1; let h = 2; let kv_h = 2;
    let ql = 16; let kl = 16; let d = 32;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 30, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 31, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 32, batch * kv_h * kl * d);

    // All mask values = -inf → all K positions masked → all exp scores = 0.
    let mask = vec![f32::NEG_INFINITY; batch * h * ql * kl];

    let ref_out = sdpa_reference_f32(
        &q, &k, &v, Some(&mask), batch, h, kv_h, ql, kl, d, scale, false,
    );

    // NaN guards must prevent exp2(NaN) propagation.
    assert_all_finite(&ref_out, "cpu_nan_guard_fully_masked");

    // All outputs should be 0 (no valid keys → no attention → zero output).
    for (i, &v) in ref_out.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "cpu_nan_guard: element {i} should be 0.0 (no valid keys), got {v}"
        );
    }
    eprintln!("test_cpu_ref_nan_guard_fully_masked: PASS — all outputs are 0.0 (NaN guards work)");
}

/// CPU self-consistency: GQA (n_heads=8, n_kv_heads=2).
///
/// Verifies the KV head index mapping in the CPU reference.
/// Each Q head h maps to KV head `h / (n_heads / n_kv_heads) = h / 4`.
#[test]
fn test_cpu_ref_gqa_8q_2kv() {
    let batch = 1; let h = 8; let kv_h = 2; let heads_per_kv = 4;
    let ql = 16; let kl = 16; let d = 32;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 40, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 41, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 42, batch * kv_h * kl * d);

    let ref_out = sdpa_reference_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, false,
    );
    let naive_out = sdpa_naive_scalar_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_cpu(&ref_out, &naive_out, CPU_SELF_CONSISTENCY_ATOL, "cpu_gqa_8q_2kv");
    let _ = heads_per_kv; // suppress unused warning
}

/// CPU self-consistency: custom scale (not 1/sqrt(d)).
///
/// Tests the scale placement path: the reference pre-scales Q by
/// scale * log2(e); the naive form applies scale after QK^T.  Both should
/// produce close outputs.
#[test]
fn test_cpu_ref_custom_scale() {
    let batch = 1; let h = 2; let kv_h = 2;
    let ql = 16; let kl = 16; let d = 32;
    let scale = 0.25_f32; // non-standard scale

    let q = pseudo_random_f32(SEED + 50, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 51, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 52, batch * kv_h * kl * d);

    let ref_out = sdpa_reference_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, false,
    );
    let naive_out = sdpa_naive_scalar_f32(
        &q, &k, &v, None, batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_cpu(&ref_out, &naive_out, CPU_SELF_CONSISTENCY_ATOL, "cpu_custom_scale");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  GPU CORRECTNESS — BF16 D=256
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests compare `dispatch_flash_attn_prefill_bf16_d256` output against
// the kernel-equivalent CPU reference computed on bf16-rounded inputs
// (`sdpa_reference_f32` applied to bf16→f32 inputs, output cast to bf16).
//
// Tolerance: bf16 has a 7-bit mantissa; the final store rounds each output
// element into bf16.  MMA accumulation is f32 internally so it does not
// widen the per-element error.  Accumulator bound for attention outputs
// (magnitudes ≲ 1): atol=5e-3 + rtol=2e-2 captures input+output round-off.

// GPU tolerance bounds — see module doc §Tolerance.
const BF16_GPU_ATOL: f32 = 5e-3;
const BF16_GPU_RTOL: f32 = 2e-2;

// Tightened D=512 tolerance — exercised only by the D=512 llama.cpp-derived
// kernel (`flash_attn_prefill_d512.metal`, NSG=8).
//
// Why tighter than D=256?  The original D=512 tests inherited the D=256
// tolerance (5e-3 / 2e-2), which turned out to be too loose to catch a
// real bug: a double-application of `log2(e)` inside the softmax (see
// /opt/hf2q/docs/ADR-011-phase2-wave2c-d512-bug-fix.md).  The defect
// effectively produced a sharper-than-correct softmax, which the loose
// tolerance absorbed on (ql=kl=32/128) random inputs but which compounded
// across Gemma 4's 5 global layers enough to flip sourdough_gate argmaxes.
//
// Under the corrected kernel (single log2(e) application — Q pre-scaled
// once on load, `exp2(s - M)` with no additional factor) the residual
// per-element error is dominated by bf16 I/O rounding (~4e-3 on magnitudes
// near 1) and f32 MMA accumulation (~1 ULP per frag).  The 1e-3 / 5e-3
// budget is tight enough to catch any future softmax / accumulator
// regression while still absorbing legitimate bf16 rounding.  Mirrors the
// tolerance bar the user directive asked for in the fix brief.
const BF16_GPU_ATOL_D512: f32 = 1e-3;
const BF16_GPU_RTOL_D512: f32 = 5e-3;

/// Cast an f32 slice to bf16 using round-to-nearest-even (the default for
/// `bf16::from_f32`).
fn f32_to_bf16(xs: &[f32]) -> Vec<bf16> {
    xs.iter().map(|&x| bf16::from_f32(x)).collect()
}

/// Cast a bf16 slice back to f32 (lossless widening).
fn bf16_to_f32(xs: &[bf16]) -> Vec<f32> {
    xs.iter().map(|&x| x.to_f32()).collect()
}

/// Fill a Metal buffer with a bf16 slice via the CPU-visible contents pointer.
fn fill_bf16_buffer(buf: &mlx_native::MlxBuffer, data: &[bf16]) {
    let ptr = buf.contents_ptr() as *mut bf16;
    assert!(!ptr.is_null(), "buffer contents pointer is null (storage mode?)");
    // SAFETY: Shared-storage MlxBuffers expose CPU-visible memory; the
    // allocation's byte length covers `data.len() * size_of::<bf16>()`.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
}

/// Read a Metal buffer into a bf16 Vec.
fn read_bf16_buffer(buf: &mlx_native::MlxBuffer, elems: usize) -> Vec<bf16> {
    let ptr = buf.contents_ptr() as *const bf16;
    assert!(!ptr.is_null(), "buffer contents pointer is null");
    // SAFETY: see fill_bf16_buffer.
    let slice = unsafe { std::slice::from_raw_parts(ptr, elems) };
    slice.to_vec()
}

/// Closeness check for two bf16 outputs converted to f32.
///
/// Asserts `|actual - expected| <= atol + rtol * |expected|` elementwise.
/// Prints the worst offender on failure.
fn assert_close_gpu(
    actual_f32: &[f32],
    expected_f32: &[f32],
    atol: f32,
    rtol: f32,
    test_name: &str,
) {
    assert_eq!(actual_f32.len(), expected_f32.len(), "{test_name}: length mismatch");
    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    let mut worst_idx = 0usize;
    let mut fail_count = 0usize;
    for (i, (&a, &e)) in actual_f32.iter().zip(expected_f32.iter()).enumerate() {
        let abs_diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        if abs_diff > tol {
            fail_count += 1;
            if abs_diff > worst_abs {
                worst_abs = abs_diff;
                worst_rel = abs_diff / e.abs().max(1e-20);
                worst_idx = i;
            }
        } else if abs_diff > worst_abs {
            worst_abs = abs_diff;
            worst_rel = abs_diff / e.abs().max(1e-20);
            worst_idx = i;
        }
    }
    if fail_count > 0 {
        panic!(
            "{test_name}: {fail_count}/{} elements exceed tolerance. \
             worst: idx={worst_idx} actual={:.6e} expected={:.6e} abs_err={:.3e} \
             rel_err={:.3e} (budget: atol={atol:.3e} + rtol={rtol:.3e})",
            actual_f32.len(), actual_f32[worst_idx], expected_f32[worst_idx],
            worst_abs, worst_rel,
        );
    }
    eprintln!(
        "{test_name}: PASS — max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} \
         (budget: atol={atol:.3e} + rtol={rtol:.3e})"
    );
}

/// Run the bf16 D=256 kernel once and return the bf16 output as f32.
///
/// Inputs are bf16 (already rounded).  The reference is computed from the
/// same bf16→f32 inputs to isolate kernel error from input-cast error.
#[allow(clippy::too_many_arguments)]
fn run_bf16_gpu(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_bf16: &[bf16],
    k_bf16: &[bf16],
    v_bf16: &[bf16],
    mask_bf16: Option<&[bf16]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> Vec<f32> {
    let q_elems = batch * n_heads * ql * head_dim;
    let kv_elems = batch * n_kv_heads * kl * head_dim;

    let q_buf = alloc_bf16(device, q_elems, "Q");
    let k_buf = alloc_bf16(device, kv_elems, "K");
    let v_buf = alloc_bf16(device, kv_elems, "V");
    let mut out_buf = alloc_bf16(device, q_elems, "out");

    fill_bf16_buffer(&q_buf, q_bf16);
    fill_bf16_buffer(&k_buf, k_bf16);
    fill_bf16_buffer(&v_buf, v_bf16);

    let mask_buf = mask_bf16.map(|m| {
        let mask_elems = batch * n_heads * ql * kl;
        assert_eq!(m.len(), mask_elems, "mask length mismatch");
        let buf = alloc_bf16(device, mask_elems, "mask");
        fill_bf16_buffer(&buf, m);
        buf
    });

    let params = FlashAttnPrefillParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d256(
        &mut encoder, device, registry,
        &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut out_buf, &params,
    ).expect("bf16 D=256 dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bf16 = read_bf16_buffer(&out_buf, q_elems);
    bf16_to_f32(&out_bf16)
}

/// Compute the CPU reference for a bf16 GPU run.
///
/// Takes the bf16-rounded inputs (so input precision loss is captured
/// identically), widens to f32, runs `sdpa_reference_f32`, then casts
/// the output to bf16 and back to f32 to simulate the GPU's final store.
#[allow(clippy::too_many_arguments)]
fn reference_for_bf16(
    q_bf16: &[bf16],
    k_bf16: &[bf16],
    v_bf16: &[bf16],
    mask_bf16: Option<&[bf16]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> Vec<f32> {
    let q = bf16_to_f32(q_bf16);
    let k = bf16_to_f32(k_bf16);
    let v = bf16_to_f32(v_bf16);
    let mask_f32 = mask_bf16.map(bf16_to_f32);

    let out_f32 = sdpa_reference_f32(
        &q, &k, &v, mask_f32.as_deref(),
        batch, n_heads, n_kv_heads, ql, kl, head_dim, scale, do_causal,
    );

    // Simulate the GPU's final bf16 store by round-tripping the output.
    let out_bf16 = f32_to_bf16(&out_f32);
    bf16_to_f32(&out_bf16)
}

/// GPU correctness: unmasked, non-causal, seq_len ∈ {32, 128, 512}.
#[test]
fn test_gpu_bf16_d256_unmasked() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    for &(ql, kl) in &[(32usize, 32usize), (128, 128), (512, 512)] {
        let batch = 1; let h = 2; let kv_h = 2; let d = 256;
        let scale = 1.0 / (d as f32).sqrt();

        let q = pseudo_random_f32(SEED, batch * h * ql * d);
        let k = pseudo_random_f32(SEED + 1, batch * kv_h * kl * d);
        let v = pseudo_random_f32(SEED + 2, batch * kv_h * kl * d);
        let q_bf = f32_to_bf16(&q);
        let k_bf = f32_to_bf16(&k);
        let v_bf = f32_to_bf16(&v);

        let gpu_out = run_bf16_gpu(
            &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );
        let ref_out = reference_for_bf16(
            &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );

        assert_close_gpu(
            &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
            &format!("gpu_bf16_unmasked_ql{ql}_kl{kl}"),
        );
    }
}

/// GPU correctness: in-kernel causal masking at seq_len ∈ {32, 128, 512}.
#[test]
fn test_gpu_bf16_d256_causal() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    for &(ql, kl) in &[(32usize, 32usize), (128, 128), (512, 512)] {
        let batch = 1; let h = 2; let kv_h = 2; let d = 256;
        let scale = 1.0 / (d as f32).sqrt();

        let q = pseudo_random_f32(SEED + 10, batch * h * ql * d);
        let k = pseudo_random_f32(SEED + 11, batch * kv_h * kl * d);
        let v = pseudo_random_f32(SEED + 12, batch * kv_h * kl * d);
        let q_bf = f32_to_bf16(&q);
        let k_bf = f32_to_bf16(&k);
        let v_bf = f32_to_bf16(&v);

        let gpu_out = run_bf16_gpu(
            &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, true,
        );
        let ref_out = reference_for_bf16(
            &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, true,
        );

        assert_close_gpu(
            &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
            &format!("gpu_bf16_causal_ql{ql}_kl{kl}"),
        );
    }
}

/// GPU correctness: external additive mask (seq_len=128).
///
/// A random sparse mask blocks ~50% of positions with a large negative value
/// (-1e4) plus 0.0 elsewhere.  This exercises the `has_mask=true` path
/// without triggering the fully-masked NaN guard.
#[test]
fn test_gpu_bf16_d256_additive_mask() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1; let h = 2; let kv_h = 2; let ql = 128; let kl = 128; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 20, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 21, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 22, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Checkerboard mask: even positions attend (0.0), odd positions blocked (-1e4).
    let mask_f32: Vec<f32> = (0..batch * h * ql * kl)
        .map(|i| if i % 2 == 0 { 0.0_f32 } else { -1.0e4_f32 })
        .collect();
    let mask_bf = f32_to_bf16(&mask_f32);

    let gpu_out = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "gpu_bf16_additive_mask",
    );
}

/// GPU correctness: fully-masked KV tile under the llama.cpp sentinel regime.
///
/// The kernel uses llama.cpp's finite-M convention (ADR-011-phase2-port-sentinel.md):
/// the running row-max `M` is initialised to `-FLT_MAX/2` (finite), so
/// `simd_max(-FLT_MAX/2, -inf) = -FLT_MAX/2` keeps `M` finite throughout
/// the K-sweep, which in turn makes `exp2(-inf - M) = exp2(-inf) = +0.0`
/// (IEEE-754 exact) — never `exp2(NaN)`.  The K-sweep accumulates
/// `sum_score = bit-exact 0` with no intermediate NaN.  The ONE remaining
/// guard is the final output normalisation (`DivOp`: `sum_score == 0 ?
/// 0 : x/sum_score`), mirroring llama.cpp's
/// `scale = S == 0 ? 0 : 1/S` at ggml-metal.metal:6358.
///
/// This test sets the entire mask to -inf; the expected output is all zeros
///   (no valid keys attended).  Any NaN or Inf in the output = the sentinel
///   chain is broken somewhere.
#[test]
fn test_gpu_bf16_d256_fully_masked_nan_guard() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1; let h = 2; let kv_h = 2; let ql = 32; let kl = 32; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 30, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 31, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 32, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // All positions masked with -inf.
    let mask_f32 = vec![f32::NEG_INFINITY; batch * h * ql * kl];
    let mask_bf = f32_to_bf16(&mask_f32);

    let gpu_out = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    // No NaN, no Inf anywhere.
    assert_all_finite(&gpu_out, "gpu_bf16_nan_guard_fully_masked");
    // All outputs should be exactly 0.0.
    for (i, &val) in gpu_out.iter().enumerate() {
        assert_eq!(
            val, 0.0,
            "gpu_bf16_nan_guard: element {i} expected 0.0, got {val} \
             (NaN guard may have been removed from kernel)"
        );
    }
    eprintln!("test_gpu_bf16_d256_fully_masked_nan_guard: PASS — all outputs 0.0, guards intact");
}

/// GPU correctness: GQA with n_heads=8, n_kv_heads=2 (heads_per_kv=4).
#[test]
fn test_gpu_bf16_d256_gqa_8q_2kv() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1; let h = 8; let kv_h = 2; let ql = 128; let kl = 128; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 40, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 41, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 42, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let gpu_out = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "gpu_bf16_gqa_8q_2kv",
    );
}

/// GPU correctness: custom scale (not `1/sqrt(head_dim)`).
#[test]
fn test_gpu_bf16_d256_custom_scale() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1; let h = 2; let kv_h = 2; let ql = 128; let kl = 128; let d = 256;
    let scale = 0.05_f32; // non-standard

    let q = pseudo_random_f32(SEED + 50, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 51, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 52, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let gpu_out = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "gpu_bf16_custom_scale",
    );
}

/// GPU correctness: unaligned ql (`ql % BQ != 0`) exercises the
/// `align_Q=false` function-constant path and the `store_safe` epilogue.
///
/// The kernel compiles a distinct pipeline for `align_Q=false` (function
/// constant index 200) and uses `store_safe` instead of plain `store` to
/// handle the partial last Q tile.  Without this test the cache only sees
/// `align_Q=true`, leaving the unaligned path untested in Phase 1a.
/// (Reviewer should-fix S1.)
#[test]
fn test_gpu_bf16_d256_unaligned_ql() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    // ql=50 → 50 % BQ(32) = 18 → align_Q=false, qL_rem=18.
    let batch = 1; let h = 2; let kv_h = 2; let ql = 50; let kl = 64; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 70, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 71, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 72, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let gpu_out = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "gpu_bf16_unaligned_ql50_kl64",
    );
}

/// GPU correctness: unaligned kl (`kl % BK != 0`) exercises the
/// `align_K=false` function-constant path and the in-kernel KV bounds check.
///
/// The kernel compiles a distinct pipeline for `align_K=false` (function
/// constant index 201) and applies a per-position validity mask on the
/// trailing partial KV tile.  Without this test the cache only sees
/// `align_K=true`, leaving the unaligned path untested in Phase 1a.
/// (Reviewer should-fix S2.)
#[test]
fn test_gpu_bf16_d256_unaligned_kl() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    // kl=50 → 50 % BK(16) = 2 → align_K=false, kL_rem=2.
    let batch = 1; let h = 2; let kv_h = 2; let ql = 32; let kl = 50; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 80, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 81, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 82, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let gpu_out = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "gpu_bf16_unaligned_ql32_kl50",
    );
}

/// GPU determinism: two dispatches with identical inputs produce identical output.
///
/// Any divergence indicates the kernel has a non-deterministic reduction order
/// (e.g. unordered atomics) — which would be a correctness regression.
#[test]
fn test_gpu_bf16_d256_determinism() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1; let h = 2; let kv_h = 2; let ql = 128; let kl = 128; let d = 256;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 60, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 61, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 62, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let run1 = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let run2 = run_bf16_gpu(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_eq!(run1.len(), run2.len());
    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "determinism violated at index {i}: run1={a} run2={b}"
        );
    }
    eprintln!("test_gpu_bf16_d256_determinism: PASS — two runs byte-identical");
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  GPU CORRECTNESS — BF16 D=512 (NSG=8, llama.cpp-derived kernel)
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests exercise `dispatch_flash_attn_prefill_bf16_d512` — the
// llama.cpp-derived NSG=8 kernel at `shaders/flash_attn_prefill_d512.metal`.
// The tolerance budget is the same as D=256 (BF16_GPU_ATOL=5e-3, _RTOL=2e-2)
// because the kernel body is a direct port of llama.cpp's proven math and
// uses the same bf16 I/O + f32 accumulation pipeline.
//
// Unlike the D=256 kernel, D=512 ships with only ONE entry point per dtype
// (additive-mask; the boolean variant is separate) with NSG selected via an
// int function constant at pipeline-creation time.  These tests dispatch
// at the default NSG=8.

/// Run the bf16 D=512 kernel once and return the bf16 output as f32.
///
/// Mirrors `run_bf16_gpu` but routes through the D=512 dispatcher.
#[allow(clippy::too_many_arguments)]
fn run_bf16_gpu_d512(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_bf16: &[bf16],
    k_bf16: &[bf16],
    v_bf16: &[bf16],
    mask_bf16: Option<&[bf16]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> Vec<f32> {
    assert_eq!(head_dim, 512, "run_bf16_gpu_d512 is D=512 only");

    let q_elems = batch * n_heads * ql * head_dim;
    let kv_elems = batch * n_kv_heads * kl * head_dim;

    let q_buf = alloc_bf16(device, q_elems, "Q");
    let k_buf = alloc_bf16(device, kv_elems, "K");
    let v_buf = alloc_bf16(device, kv_elems, "V");
    let mut out_buf = alloc_bf16(device, q_elems, "out");

    fill_bf16_buffer(&q_buf, q_bf16);
    fill_bf16_buffer(&k_buf, k_bf16);
    fill_bf16_buffer(&v_buf, v_bf16);

    let mask_buf = mask_bf16.map(|m| {
        let mask_elems = batch * n_heads * ql * kl;
        assert_eq!(m.len(), mask_elems, "mask length mismatch");
        let buf = alloc_bf16(device, mask_elems, "mask");
        fill_bf16_buffer(&buf, m);
        buf
    });

    let params = FlashAttnPrefillParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d512(
        &mut encoder, device, registry,
        &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut out_buf, &params,
    ).expect("bf16 D=512 dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bf16 = read_bf16_buffer(&out_buf, q_elems);
    bf16_to_f32(&out_bf16)
}

/// Measure the compiled pipeline's static threadgroup memory footprint and
/// max threads/TG.  This is the regression gate for the TG-memory claim:
/// llama.cpp reports 28,672 B for (DK=DV=512, Q=8, C=64, NSG=8, bf16,
/// is_q=0); if this measurement is larger, the kernel is allocating
/// statically-sized threadgroup arrays that don't exist in llama.cpp's
/// kernel (or our shmem aliasing has drifted).
#[test]
fn test_d512_pipeline_tg_memory_and_threads() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);

    let pipeline = registry
        .get_pipeline_with_constants(
            "flash_attn_prefill_llamacpp_bf16_d512",
            device.metal_device(),
            &[(200, true), (201, true), (300, false), (301, false)],
            &[(322, 8_i32)],
        )
        .expect("bf16 d512 nsg=8 pipeline compiles");

    // NOTE: `static_threadgroup_memory_length` reports threadgroup memory
    // declared STATICALLY inside the shader (via `threadgroup T arr[N]`
    // declarations).  Our kernel declares its buffers DYNAMICALLY
    // (`threadgroup half* shmem_f16 [[threadgroup(0)]]`) and sizes them at
    // encode time via `setThreadgroupMemoryLength` — so this value is
    // expected to be 0, and the real allocation (28,672 B) is done at the
    // dispatcher level.  We still assert the pipeline compiled.
    let static_tg = pipeline.static_threadgroup_memory_length();
    let max_total = pipeline.max_total_threads_per_threadgroup();
    let exec_width = pipeline.thread_execution_width();

    eprintln!(
        "D=512 NSG=8 pipeline: static_tg={} bytes, max_total_threads_per_tg={}, \
         thread_execution_width={}",
        static_tg, max_total, exec_width
    );

    // Apple GPU simdgroup width is always 32.
    assert_eq!(exec_width, 32, "simdgroup width must be 32");

    // Our shader annotated `max_total_threads_per_threadgroup(32 * 8) = 256`
    // on the kernel.  Metal compiler may report a smaller value if it can
    // prove register pressure forces occupancy-down, but 256 is the upper
    // bound we declared.
    assert!(
        max_total >= 256,
        "max_total_threads_per_threadgroup >= 256 required for NSG=8 × 32 lanes"
    );

    // Dynamic allocation is used (matches llama.cpp), so static size should
    // be small or zero.
    assert!(
        static_tg < 1024,
        "static threadgroup memory should be near-zero (we allocate dynamically); \
         got {static_tg} bytes"
    );
}

/// Verify the llama.cpp-derived D=512 shader library compiles and the
/// bf16 NSG=8 pipeline is obtainable at the canonical function-constant
/// combination.
///
/// Acts as the regression gate for shader-source errors — if the kernel
/// template has an MSL syntax bug, this test fails before any of the
/// correctness tests have a chance to run.
#[test]
fn test_bf16_d512_llamacpp_library_compiles() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);

    // Canonical FC combo: align_Q=true, align_K=true, has_mask=false,
    // do_causal=false, nsg=8.
    let result = registry.get_pipeline_with_constants(
        "flash_attn_prefill_llamacpp_bf16_d512",
        device.metal_device(),
        &[(200, true), (201, true), (300, false), (301, false)],
        &[(322, 8_i32)],
    );
    match result {
        Ok(_) => eprintln!("test_bf16_d512_llamacpp_library_compiles: OK"),
        Err(MlxError::ShaderCompilationError { name, message }) => {
            panic!(
                "bf16 D=512 NSG=8 pipeline compilation failed: name={name}, \
                 message={message}"
            );
        }
        Err(other) => panic!("Unexpected error: {other:?}"),
    }
}

/// GPU correctness: unmasked, non-causal, at D=512 with (ql, kl) ∈
/// {(32, 32), (128, 128)}.
///
/// Small head count keeps CPU reference reasonable while exercising the
/// 8 simdgroups × 8 Q-rows × 64 KV-cols tile geometry.
#[test]
fn test_gpu_bf16_d512_unmasked() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);

    for &(ql, kl) in &[(32usize, 32usize), (128, 128)] {
        let batch = 1; let h = 2; let kv_h = 2; let d = 512;
        let scale = 1.0 / (d as f32).sqrt();

        let q = pseudo_random_f32(SEED + 100, batch * h * ql * d);
        let k = pseudo_random_f32(SEED + 101, batch * kv_h * kl * d);
        let v = pseudo_random_f32(SEED + 102, batch * kv_h * kl * d);
        let q_bf = f32_to_bf16(&q);
        let k_bf = f32_to_bf16(&k);
        let v_bf = f32_to_bf16(&v);

        let gpu_out = run_bf16_gpu_d512(
            &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );
        let ref_out = reference_for_bf16(
            &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );

        assert_close_gpu(
            &gpu_out, &ref_out, BF16_GPU_ATOL_D512, BF16_GPU_RTOL_D512,
            &format!("gpu_bf16_d512_unmasked_ql{ql}_kl{kl}"),
        );
    }
}

/// GPU correctness: in-kernel causal masking at D=512.
#[test]
fn test_gpu_bf16_d512_causal() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);

    for &(ql, kl) in &[(32usize, 32usize), (128, 128)] {
        let batch = 1; let h = 2; let kv_h = 2; let d = 512;
        let scale = 1.0 / (d as f32).sqrt();

        let q = pseudo_random_f32(SEED + 110, batch * h * ql * d);
        let k = pseudo_random_f32(SEED + 111, batch * kv_h * kl * d);
        let v = pseudo_random_f32(SEED + 112, batch * kv_h * kl * d);
        let q_bf = f32_to_bf16(&q);
        let k_bf = f32_to_bf16(&k);
        let v_bf = f32_to_bf16(&v);

        let gpu_out = run_bf16_gpu_d512(
            &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, true,
        );
        let ref_out = reference_for_bf16(
            &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, true,
        );

        assert_close_gpu(
            &gpu_out, &ref_out, BF16_GPU_ATOL_D512, BF16_GPU_RTOL_D512,
            &format!("gpu_bf16_d512_causal_ql{ql}_kl{kl}"),
        );
    }
}

/// GPU correctness: fully-masked KV tile at D=512 — exercises the sentinel
/// DivOp guard (`S == 0 ? 0 : 1/S`).
///
/// Under the finite-M regime, the entire score row degenerates to `-inf`
/// (from the mask), exp2(-inf - -FLT_MAX/2) = 0 (IEEE-754 exact), sum = 0.
/// The final `output / sum` guard returns 0 — output row must be all zeros
/// with no NaN / Inf leakage.
#[test]
fn test_gpu_bf16_d512_fully_masked_sentinel() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);

    let batch = 1; let h = 2; let kv_h = 2; let ql = 32; let kl = 32; let d = 512;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 130, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 131, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 132, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Fully -inf additive mask.
    let mask_f32 = vec![f32::NEG_INFINITY; batch * h * ql * kl];
    let mask_bf = f32_to_bf16(&mask_f32);

    let gpu_out = run_bf16_gpu_d512(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    // No NaN, no Inf.
    assert_all_finite(&gpu_out, "gpu_bf16_d512_fully_masked_sentinel");

    // All outputs must be exactly 0.0.
    for (i, &val) in gpu_out.iter().enumerate() {
        assert_eq!(
            val, 0.0,
            "gpu_bf16_d512_fully_masked_sentinel: element {i} expected 0.0, got {val} \
             (DivOp sentinel may be broken)"
        );
    }
    eprintln!("test_gpu_bf16_d512_fully_masked_sentinel: PASS — all zeros, DivOp guard intact");
}

/// GPU determinism: two dispatches with identical inputs produce identical
/// output at D=512 NSG=8.  Any divergence = non-deterministic reduction
/// order (e.g. unordered atomics) — would be a correctness regression.
#[test]
fn test_gpu_bf16_d512_determinism() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);

    let batch = 1; let h = 2; let kv_h = 2; let ql = 128; let kl = 128; let d = 512;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 160, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 161, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 162, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let run1 = run_bf16_gpu_d512(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let run2 = run_bf16_gpu_d512(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_eq!(run1.len(), run2.len());
    for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "determinism violated at index {i}: run1={a} run2={b}"
        );
    }
    eprintln!("test_gpu_bf16_d512_determinism: PASS — two runs byte-identical");
}

/// Regression gate for the "double log2(e)" softmax bug fixed in
/// `/opt/hf2q/docs/ADR-011-phase2-wave2c-d512-bug-fix.md`.
///
/// The bug — multiplying `(s2 - new_max)` by `log2(e)` inside `exp2` after
/// Q had already been pre-scaled by `scale * log2(e)` — produced a softmax
/// effectively scaled by `log2(e)^2 ≈ 2.08` rather than `log2(e)` once.
/// On low-variance inputs (the original `(ql=32, kl=32)` random PRNG test)
/// the score range is small enough that the sharpened distribution stays
/// inside the `5e-3 / 2e-2` tolerance.  But on HIGH-variance inputs — the
/// regime real attention sees, with `score = Q·K^T / sqrt(d)` ranging over
/// several units of magnitude — the bug shifts the softmax output by tens
/// of percent at the top-attended positions.
///
/// This test constructs an input with deliberately high score variance:
/// Q and K from `pseudo_random_f32 ∈ [-1, 1]`, scaled by `2/sqrt(d)` (4×
/// the standard `1/sqrt(d)`).  The pre-softmax scores are O(2-4) range,
/// so a single softmax-sharpness factor of `log2(e)` distorts the top
/// score's weight by `e^(2 * (log2(e) - 1)) ≈ e^0.885 ≈ 2.42`.  This
/// shows up as multi-percent output shifts that violate the tightened
/// `1e-3 / 5e-3` budget, ensuring the bug class can't recur silently.
///
/// Tolerance: same `BF16_GPU_ATOL_D512 / BF16_GPU_RTOL_D512` budget as
/// the standard D=512 tests — these ARE the bf16 round-off floor; any
/// extra softmax-shape error is a kernel bug.
#[test]
fn test_gpu_bf16_d512_high_variance_softmax() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);

    // Larger ql, kl + two heads to exercise the per-head softmax more
    // thoroughly.  Includes a partial trailing KV chunk (kl=72 → one full
    // chunk of C=64 + 8 leftover) to also exercise the align_K=false guard.
    for &(ql, kl) in &[(64usize, 72usize), (128, 192)] {
        let batch = 1; let h = 2; let kv_h = 2; let d = 512;
        // 4× the standard scale → score variance ≈ 16× wider, exercising
        // the regime where the double-log2(e) bug produces the largest
        // softmax-shape distortion.
        let scale = 4.0 / (d as f32).sqrt();

        let q = pseudo_random_f32(SEED + 200, batch * h * ql * d);
        let k = pseudo_random_f32(SEED + 201, batch * kv_h * kl * d);
        let v = pseudo_random_f32(SEED + 202, batch * kv_h * kl * d);
        let q_bf = f32_to_bf16(&q);
        let k_bf = f32_to_bf16(&k);
        let v_bf = f32_to_bf16(&v);

        let gpu_out = run_bf16_gpu_d512(
            &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );
        let ref_out = reference_for_bf16(
            &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );

        assert_close_gpu(
            &gpu_out, &ref_out, BF16_GPU_ATOL_D512, BF16_GPU_RTOL_D512,
            &format!("gpu_bf16_d512_high_variance_softmax_ql{ql}_kl{kl}"),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  SWA / CAUSAL MASK BUILDER (Wave 2D, ADR-011 Phase 2)
// ─────────────────────────────────────────────────────────────────────────────
//
// Verifies the port of llama.cpp's `llm_graph_input_attn_no_cache::set_input`
// mask-fill algorithm (llama-graph.cpp:380-444) and the `is_masked_swa`
// predicate (llama-hparams.h:316-328) into a bf16 GPU-fill kernel.
//
// The builder produces a `[seq_len_q, seq_len_k]` bf16 mask with discrete
// cell values: 0.0 for attended, -inf for masked.  Correctness is EXACT
// (no floating-point tolerance) — any cell that disagrees with the CPU
// predicate is a bug.  The integration test then runs the mask through
// `dispatch_flash_attn_prefill_bf16_d256` with `do_causal=false` to confirm
// the mask-carried causal + SWA gating produces the same attention output
// as the CPU reference with the same semantics, within bf16 tolerance.
//
// Sentinel verification: bf16(-INFINITY) must encode as bit pattern 0xFF80
// and bf16(0.0) as 0x0000.  Any drift here is a silent correctness bug
// because the flash_attn_prefill kernel relies on the finite-M-sentinel
// regime's handling of exact bf16 -inf in mask cells.

/// bf16(-INFINITY) must encode as 0xFF80 (sign=1, exponent=0xFF, mantissa=0).
/// bf16(0.0) must encode as 0x0000.
///
/// Both half (F16) and bfloat16 have real `-inf` representations; verifies
/// that the `half::bf16::from_f32` path preserves the infinity through the
/// cast.  If this ever changes the mask would silently start producing a
/// finite sentinel (e.g. -FLT_MAX saturation), breaking the kernel's
/// finite-M design and causing NaN propagation on fully-masked rows.
///
/// Cross-checks our Rust-side cast against the Metal-side cast at
/// `flash_attn_prefill_mask.metal:bfloat16_t(-INFINITY)`: both must produce
/// 0xFF80.
#[test]
fn test_mask_sentinel_bit_patterns() {
    let ninf = bf16::from_f32(f32::NEG_INFINITY);
    assert_eq!(
        ninf.to_bits(),
        0xFF80,
        "bf16(-INFINITY) must be 0xFF80 (sign=1, exp=0xFF, mantissa=0)"
    );
    assert!(ninf.is_infinite(), "bf16(-INFINITY) must be infinite");
    assert!(ninf.is_sign_negative(), "bf16(-INFINITY) must be negative");

    let zero = bf16::from_f32(0.0);
    assert_eq!(zero.to_bits(), 0x0000, "bf16(0.0) must be 0x0000");

    eprintln!("test_mask_sentinel_bit_patterns: PASS — bf16 sentinels are correct (0xFF80 / 0x0000)");
}

/// Read a mask buffer and assert elementwise that the semantics match the
/// expected (causal + SWA) predicate.  0.0 cells must compare exactly zero;
/// -inf cells must compare as negative infinity.  No tolerance — these
/// are discrete values.
fn assert_mask_matches_predicate(
    buf: &mlx_native::MlxBuffer,
    seq_len_q: usize,
    seq_len_k: usize,
    window_size: Option<u32>,
    causal: bool,
    q_abs_offset: u32,
    test_name: &str,
) {
    let data = read_bf16_buffer(buf, seq_len_q * seq_len_k);
    let mut first_mismatch: Option<(usize, usize, f32, bool)> = None;
    for q in 0..seq_len_q {
        for k in 0..seq_len_k {
            let cell = data[q * seq_len_k + k];
            let q_abs = q as i64 + q_abs_offset as i64;
            let k_pos = k as i64;

            // Mirror is_masked_swa + causal.
            let mut expected_masked = false;
            if causal && k_pos > q_abs {
                expected_masked = true;
            }
            if let Some(w) = window_size {
                if (q_abs - k_pos) >= w as i64 {
                    expected_masked = true;
                }
            }

            let actual_masked = cell.is_infinite() && cell.is_sign_negative();
            let actual_attended = cell.to_f32() == 0.0;

            let ok = if expected_masked {
                actual_masked
            } else {
                actual_attended
            };

            if !ok && first_mismatch.is_none() {
                first_mismatch = Some((q, k, cell.to_f32(), expected_masked));
            }
        }
    }
    if let Some((q, k, got, expected_masked)) = first_mismatch {
        panic!(
            "{test_name}: mismatch at (q={q}, k={k}): got={got} \
             (bits=0x{:04X}), expected {} (causal={causal}, \
             window_size={:?}, q_abs_offset={q_abs_offset})",
            data[q * seq_len_k + k].to_bits(),
            if expected_masked { "-inf" } else { "0.0" },
            window_size,
        );
    }
    eprintln!(
        "{test_name}: PASS — all {} cells match predicate",
        seq_len_q * seq_len_k
    );
}

/// 2D mask layout: seq_len=8, causal-only, no window.
/// Expected: lower-triangle (inclusive diag) = 0.0, upper-triangle = -inf.
#[test]
fn test_mask_global_causal() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_mask::register(&mut registry);

    let params = SdpaMaskParams {
        seq_len_q: 8,
        seq_len_k: 8,
        window_size: None,
        causal: true,
        q_abs_offset: 0,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    let mask = build_sdpa_mask_bf16(&device, &mut registry, &mut encoder, &params)
        .expect("build_sdpa_mask_bf16");
    encoder.commit_and_wait().expect("commit_and_wait");

    assert_eq!(mask.dtype(), DType::BF16);
    assert_eq!(mask.shape(), &[8, 8]);
    assert_eq!(mask.byte_len(), 8 * 8 * 2);

    assert_mask_matches_predicate(&mask, 8, 8, None, true, 0, "mask_global_causal");

    // Spot-check a few specific cells for the triangle pattern.
    let data = read_bf16_buffer(&mask, 64);
    // (0, 0): attended (k=0 <= q=0).
    assert_eq!(data[0 * 8 + 0].to_f32(), 0.0);
    // (0, 7): masked (k=7 > q=0).
    assert!(data[0 * 8 + 7].is_infinite() && data[0 * 8 + 7].is_sign_negative());
    // (7, 0): attended (k=0 <= q=7).
    assert_eq!(data[7 * 8 + 0].to_f32(), 0.0);
    // (7, 7): attended (k=7 <= q=7, diagonal).
    assert_eq!(data[7 * 8 + 7].to_f32(), 0.0);
    // (3, 5): masked (k=5 > q=3).
    assert!(data[3 * 8 + 5].is_infinite());
}

/// 2D mask layout: seq_len=16, window=4, causal.
/// Expected: for each q, attended iff (k <= q) AND (q - k < 4).
/// So row q=10 has attended cells k ∈ {7, 8, 9, 10}; k ≤ 6 is masked (outside
/// window) and k ≥ 11 is masked (future, causal).
#[test]
fn test_mask_sliding_window_4() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_mask::register(&mut registry);

    let params = SdpaMaskParams {
        seq_len_q: 16,
        seq_len_k: 16,
        window_size: Some(4),
        causal: true,
        q_abs_offset: 0,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    let mask = build_sdpa_mask_bf16(&device, &mut registry, &mut encoder, &params)
        .expect("build_sdpa_mask_bf16");
    encoder.commit_and_wait().expect("commit_and_wait");

    assert_mask_matches_predicate(
        &mask, 16, 16, Some(4), true, 0, "mask_sliding_window_4",
    );

    // Spot checks on the window boundary (the exclusive upper bound is the
    // primary failure mode for off-by-one bugs).
    let data = read_bf16_buffer(&mask, 16 * 16);
    // Row q=10:
    //   k=6 → distance 4 → masked (EXCLUSIVE upper bound).
    //   k=7 → distance 3 → attended.
    //   k=10 → distance 0 → attended.
    //   k=11 → future → masked.
    assert!(
        data[10 * 16 + 6].is_infinite(),
        "(q=10, k=6) distance=4 must be masked (exclusive upper bound)"
    );
    assert_eq!(
        data[10 * 16 + 7].to_f32(), 0.0,
        "(q=10, k=7) distance=3 must be attended (within window)"
    );
    assert_eq!(
        data[10 * 16 + 10].to_f32(), 0.0,
        "(q=10, k=10) diagonal must be attended"
    );
    assert!(
        data[10 * 16 + 11].is_infinite(),
        "(q=10, k=11) future must be masked"
    );
}

/// Integration: build a mask, feed it to `dispatch_flash_attn_prefill_bf16_d256`
/// with `do_causal=false` (the causal+window constraints are baked into the
/// mask).  Compare GPU output against the CPU reference with the same
/// semantics applied via the `mask` argument.  Must agree at bf16 tolerance.
///
/// The mask builder produces a `[qL, kL]` layout.  For the integration we
/// set `n_heads = n_kv_heads = batch = 1` so the flash_attn_prefill
/// dispatcher's internal stride calculation
/// (`m_batch_stride = h * qL * kL = qL * kL`, `m_head_stride = qL * kL`) is
/// degenerate: (batch=0, head=0) always indexes the same `[qL, kL]` plane.
/// This is the simplest integration that exercises the full mask → kernel
/// path without requiring the broadcast-stride plumbing that Wave 2E/3
/// will add at the hf2q call-site.
#[test]
fn test_mask_integrates_with_flash_attn_prefill() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1;
    let ql = 128; let kl = 128; let d = 256;
    let window = 64_u32;
    let scale = 1.0 / (d as f32).sqrt();

    // Random Q/K/V.
    let q = pseudo_random_f32(SEED + 300, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 301, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 302, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // ── Build the mask ────────────────────────────────────────────────────
    let mask_params = SdpaMaskParams {
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        window_size: Some(window),
        causal: true,
        q_abs_offset: 0,
    };
    let mut mask_enc = device.command_encoder().expect("mask encoder");
    let mask_buf = build_sdpa_mask_bf16(&device, &mut registry, &mut mask_enc, &mask_params)
        .expect("build_sdpa_mask_bf16");
    mask_enc.commit_and_wait().expect("mask commit");

    // Verify mask correctness before using it (cheap; localises failures).
    assert_mask_matches_predicate(
        &mask_buf, ql, kl, Some(window), true, 0, "integration_mask_precheck",
    );

    // ── GPU flash_attn_prefill with the external mask, do_causal=false ────
    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");
    fill_bf16_buffer(&q_buf, &q_bf);
    fill_bf16_buffer(&k_buf, &k_bf);
    fill_bf16_buffer(&v_buf, &v_bf);

    let params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false,
    };

    let mut enc = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d256(
        &mut enc, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, Some(&mask_buf), &mut out_buf, &params,
    ).expect("dispatch bf16 d256");
    enc.commit_and_wait().expect("commit");

    let gpu_out_bf = read_bf16_buffer(&out_buf, q_elems);
    let gpu_out = bf16_to_f32(&gpu_out_bf);

    // ── CPU reference: apply the same mask as an additive term ────────────
    //
    // sdpa_reference_f32 takes a [batch, h, qL, kL] mask; our mask is
    // [qL, kL].  Since batch=h=1 the layouts are byte-identical — pass the
    // mask f32 directly.
    let mask_f32: Vec<f32> = read_bf16_buffer(&mask_buf, ql * kl)
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&f32_to_bf16(&mask_f32)),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "mask_integrates_with_flash_attn_prefill",
    );
}

/// Gemma 4 sliding-prefill shape: seq_len=2455, window=1024 (per ADR-011
/// Phase 2 §3.2).  Head count is minimised (h=1, kv_h=1, d=256) because
/// correctness scales with sequence length, not head count; adding heads
/// would pad the runtime with no additional mask coverage.
///
/// This is the largest shape we exercise in this test file and validates
/// that the mask-fill kernel + the flash_attn_prefill consumer both behave
/// correctly at realistic Gemma 4 prompt lengths, at the seq_len > n_swa
/// boundary where the SWA mask actually differs from plain causal.
#[test]
fn test_mask_gemma4_sliding_prefill() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1;
    let ql = 2455; let kl = 2455; let d = 256;
    let window = 1024_u32;
    let scale = 1.0 / (d as f32).sqrt();

    // Random Q/K/V — pseudo_random is deterministic under a fixed seed.
    let q = pseudo_random_f32(SEED + 400, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 401, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 402, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Build the mask.
    let mask_params = SdpaMaskParams {
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        window_size: Some(window),
        causal: true,
        q_abs_offset: 0,
    };
    let mut mask_enc = device.command_encoder().expect("mask encoder");
    let mask_buf = build_sdpa_mask_bf16(&device, &mut registry, &mut mask_enc, &mask_params)
        .expect("build_sdpa_mask_bf16");
    mask_enc.commit_and_wait().expect("mask commit");

    // Predicate cross-check on a coarse sample — full-pass predicate test is
    // O(ql*kl) = 6 M cells and would dwarf the actual GPU runtime; the
    // smaller test_mask_sliding_window_4 already validates the predicate
    // at every cell for a small shape.  For this large-shape test we sample
    // the seq_len > n_swa boundary, where the SWA mask actually differs
    // from plain causal (cells (q, q-window) and (q, q-window-1) flip
    // across the SWA boundary).
    let mask_data = read_bf16_buffer(&mask_buf, ql * kl);
    for q_row in [1024usize, 1500, 2000, 2454].iter() {
        let q_row = *q_row;
        // (q, q - window): distance = window → EXCLUSIVE upper bound → masked.
        let k_pos = q_row - window as usize;
        assert!(
            mask_data[q_row * kl + k_pos].is_infinite(),
            "(q={q_row}, k={k_pos}) distance={window} must be masked (exclusive)"
        );
        // (q, q - window + 1): distance = window - 1 → attended.
        let k_pos = q_row - window as usize + 1;
        assert_eq!(
            mask_data[q_row * kl + k_pos].to_f32(), 0.0,
            "(q={q_row}, k={k_pos}) distance={}  must be attended (within window)",
            window - 1
        );
        // (q, q + 1): future → masked.
        if q_row + 1 < kl {
            let k_pos = q_row + 1;
            assert!(
                mask_data[q_row * kl + k_pos].is_infinite(),
                "(q={q_row}, k={k_pos}) future must be masked"
            );
        }
    }

    // ── Run flash_attn_prefill with the mask ──────────────────────────────
    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");
    fill_bf16_buffer(&q_buf, &q_bf);
    fill_bf16_buffer(&k_buf, &k_bf);
    fill_bf16_buffer(&v_buf, &v_bf);

    let params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false,
    };

    let mut enc = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d256(
        &mut enc, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, Some(&mask_buf), &mut out_buf, &params,
    ).expect("dispatch bf16 d256");
    enc.commit_and_wait().expect("commit");

    let gpu_out_bf = read_bf16_buffer(&out_buf, q_elems);
    let gpu_out = bf16_to_f32(&gpu_out_bf);

    // CPU reference.
    let mask_f32: Vec<f32> = mask_data.iter().map(|x| x.to_f32()).collect();
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&f32_to_bf16(&mask_f32)),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "mask_gemma4_sliding_prefill",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  WAVE 2E — TILE-SKIP PRE-PASS TESTS
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: read the entire blk byte buffer as a Vec<u8>.
fn read_u8_buffer(buf: &mlx_native::MlxBuffer, elems: usize) -> Vec<u8> {
    let ptr = buf.contents_ptr() as *const u8;
    assert!(!ptr.is_null(), "buffer contents pointer is null");
    // SAFETY: buffer is shared-storage and covers at least `elems` bytes.
    let slice = unsafe { std::slice::from_raw_parts(ptr, elems) };
    slice.to_vec()
}

/// Helper: build the Wave 2D mask + Wave 2E blk buffer and return both.
fn build_mask_and_blk(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ql: u32,
    kl: u32,
    window_size: Option<u32>,
    causal: bool,
    bq: u32,
    bk: u32,
) -> (mlx_native::MlxBuffer, mlx_native::MlxBuffer) {
    // Build the bf16 mask.
    let mask_params = SdpaMaskParams {
        seq_len_q: ql,
        seq_len_k: kl,
        window_size,
        causal,
        q_abs_offset: 0,
    };
    let mut mask_enc = device.command_encoder().expect("mask encoder");
    let mask = build_sdpa_mask_bf16(device, registry, &mut mask_enc, &mask_params)
        .expect("build_sdpa_mask_bf16");
    mask_enc.commit_and_wait().expect("mask commit");

    // Allocate and fill the blk classification buffer.
    let blk_params = BlkParams {
        seq_len_q: ql,
        seq_len_k: kl,
        bq,
        bk,
    };
    let blk = alloc_blk_buffer(device, &blk_params).expect("alloc_blk_buffer");
    let mut blk_enc = device.command_encoder().expect("blk encoder");
    dispatch_flash_attn_prefill_blk(
        &mut blk_enc,
        device,
        registry,
        &mask,
        &blk,
        &blk_params,
    ).expect("dispatch_flash_attn_prefill_blk");
    blk_enc.commit_and_wait().expect("blk commit");

    (mask, blk)
}

/// Reference classifier: given a bf16 mask + tile shape, compute the
/// per-tile byte on the CPU.  Byte values match the GPU pre-pass:
///   0 — every cell is -inf (or < -1e30f).
///   1 — mixed (at least one finite, at least one non-zero).
///   2 — every cell is exactly 0.0.
/// Partial right-edge tiles (tile_k_end > kL) default to 1.
fn cpu_classify_blk(
    mask: &[bf16],
    ql: usize,
    kl: usize,
    bq: usize,
    bk: usize,
) -> Vec<u8> {
    let nq = (ql + bq - 1) / bq;
    let nk = (kl + bk - 1) / bk;
    let mut out = vec![0u8; nq * nk];
    for qt in 0..nq {
        for kt in 0..nk {
            let tile_k_start = kt * bk;
            let tile_k_end = tile_k_start + bk;
            let res;
            if tile_k_start >= kl {
                res = 0;
            } else if tile_k_end > kl {
                res = 1;
            } else {
                let q_rows = bq.min(ql.saturating_sub(qt * bq));
                let mut mmin = f32::INFINITY;
                let mut mmax = f32::NEG_INFINITY;
                for j in 0..q_rows {
                    for i in 0..bk {
                        let row = qt * bq + j;
                        let col = tile_k_start + i;
                        let v = mask[row * kl + col].to_f32();
                        mmin = mmin.min(v);
                        mmax = mmax.max(v);
                    }
                }
                res = if mmax <= -1.0e30f32 {
                    0
                } else if mmin == 0.0 && mmax == 0.0 {
                    2
                } else {
                    1
                };
            }
            out[qt * nk + kt] = res;
        }
    }
    out
}

/// Test: global causal mask at seq_len=64 D=256 — verify classification.
/// Expected: upper-triangle tiles = 0, lower-triangle full tiles = 2,
/// diagonal tiles = 1.  Tile shape (BQ=32, BK=16) gives NQ=2, NK=4.
#[test]
fn test_blk_global_causal_d256() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);

    let ql = 64_u32;
    let kl = 64_u32;
    let bq = 32_u32;
    let bk = 16_u32;

    let (mask, blk) = build_mask_and_blk(&device, &mut registry, ql, kl, None, true, bq, bk);

    // Read and classify on CPU for cross-check.
    let mask_data = read_bf16_buffer(&mask, (ql * kl) as usize);
    let cpu_blk = cpu_classify_blk(&mask_data, ql as usize, kl as usize,
                                    bq as usize, bk as usize);

    let nq_tiles = ((ql + bq - 1) / bq) as usize;
    let nk_tiles = ((kl + bk - 1) / bk) as usize;
    let gpu_blk = read_u8_buffer(&blk, nq_tiles * nk_tiles);

    // Verify GPU matches CPU classifier byte-for-byte.
    for qt in 0..nq_tiles {
        for kt in 0..nk_tiles {
            let i = qt * nk_tiles + kt;
            assert_eq!(
                gpu_blk[i], cpu_blk[i],
                "blk[{qt}][{kt}] mismatch: GPU={}, CPU={}",
                gpu_blk[i], cpu_blk[i]
            );
        }
    }

    // Spot-check the expected structural pattern.  At ql=kl=64, bq=32, bk=16:
    //   Q-tile 0 (rows 0-31):  above-diagonal K-tiles (kt >= 2) fully masked;
    //                          diagonal K-tile (kt=1) mixed (kt*bk=16 ≤ row 31 < 32);
    //                          kt=0 (rows 0-31, cols 0-15) — mixed because rows 0-15
    //                          have some masked cells (causal: col > row).
    //                          Actually kt=0 has causal triangle: every row has
    //                          attended cols [0..=row], masked cols [row+1..16].
    //                          For row 15 all cols 0-15 are attended → last row.
    //                          For row 0 only col 0 attended, rest masked → mixed.
    //   Q-tile 1 (rows 32-63): kt=0, kt=1 fully-attended (row ≥ 32 always > col ≤ 31);
    //                          kt=2 mixed (diagonal spans rows 32-63 × cols 32-47);
    //                          kt=3 fully masked (cols 48-63, rows 32-47 have col>row).

    // qt=0, kt=2 (cols 32-47, rows 0-31): all col > row → fully masked.
    assert_eq!(gpu_blk[0 * nk_tiles + 2], 0);
    // qt=0, kt=3 (cols 48-63, rows 0-31): all col > row → fully masked.
    assert_eq!(gpu_blk[0 * nk_tiles + 3], 0);
    // qt=1, kt=0 (cols 0-15, rows 32-63): all col <= row → all attended.
    assert_eq!(gpu_blk[1 * nk_tiles + 0], 2);
    // qt=1, kt=1 (cols 16-31, rows 32-63): all col <= row → all attended.
    assert_eq!(gpu_blk[1 * nk_tiles + 1], 2);
    // qt=1, kt=2 (cols 32-47, rows 32-63): diagonal tile → mixed.
    assert_eq!(gpu_blk[1 * nk_tiles + 2], 1);
    // qt=1, kt=3 (cols 48-63, rows 32-47 masked, 48-63 mixed) → mixed.
    assert_eq!(gpu_blk[1 * nk_tiles + 3], 1);
}

/// Test: sliding window mask at seq_len=256, window=64 D=256 — verify
/// sliding pattern.  Tile shape (32, 16) gives NQ=8, NK=16.
#[test]
fn test_blk_sliding_window_d256() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);

    let ql = 256_u32;
    let kl = 256_u32;
    let bq = 32_u32;
    let bk = 16_u32;
    let window = 64_u32;

    let (mask, blk) =
        build_mask_and_blk(&device, &mut registry, ql, kl, Some(window), true, bq, bk);

    let mask_data = read_bf16_buffer(&mask, (ql * kl) as usize);
    let cpu_blk = cpu_classify_blk(&mask_data, ql as usize, kl as usize,
                                    bq as usize, bk as usize);

    let nq_tiles = ((ql + bq - 1) / bq) as usize;
    let nk_tiles = ((kl + bk - 1) / bk) as usize;
    let gpu_blk = read_u8_buffer(&blk, nq_tiles * nk_tiles);

    for qt in 0..nq_tiles {
        for kt in 0..nk_tiles {
            let i = qt * nk_tiles + kt;
            assert_eq!(
                gpu_blk[i], cpu_blk[i],
                "blk[{qt}][{kt}] mismatch: GPU={}, CPU={}",
                gpu_blk[i], cpu_blk[i]
            );
        }
    }

    // Count class-0 (skip) tiles: must be > 0 (tiles far above diagonal
    // AND tiles outside the SWA window below the diagonal).
    let zero_count = gpu_blk.iter().filter(|&&b| b == 0).count();
    let two_count  = gpu_blk.iter().filter(|&&b| b == 2).count();
    let one_count  = gpu_blk.iter().filter(|&&b| b == 1).count();

    eprintln!(
        "test_blk_sliding_window_d256: class counts — 0(skip)={zero_count} \
         1(mixed)={one_count} 2(all_attended)={two_count} \
         total={} (NQ*NK={})",
        zero_count + one_count + two_count,
        nq_tiles * nk_tiles
    );

    assert!(zero_count > 0, "SWA mask must produce some fully-masked tiles");
    assert!(two_count > 0,
            "SWA mask must produce some all-attended tiles (interior of window)");
    assert!(one_count > 0, "SWA mask must produce some mixed tiles (diagonal)");
    assert_eq!(zero_count + one_count + two_count, nq_tiles * nk_tiles);
}

/// Correctness gate: D=256 main kernel output MUST be bit-identical with
/// blk=None vs blk=Some(built_blk).  The blk path is a skip optimisation,
/// NEVER a correctness change.
///
/// Uses a modest shape (qL=kL=128, d=256) so bit-exact comparison is fast.
#[test]
fn test_gpu_bf16_d256_with_blk_matches_no_blk() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1;
    let ql = 128usize; let kl = 128usize; let d = 256;
    let window = 48_u32;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 600, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 601, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 602, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Build mask + blk once.
    let (mask_buf, blk_buf) = build_mask_and_blk(
        &device, &mut registry,
        ql as u32, kl as u32,
        Some(window), true,
        32, 16,
    );

    // Pre-fill two separate output buffers so they don't interfere.
    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;

    let q_buf_a = alloc_bf16(&device, q_elems, "Q_a");
    let k_buf_a = alloc_bf16(&device, kv_elems, "K_a");
    let v_buf_a = alloc_bf16(&device, kv_elems, "V_a");
    let mut out_a = alloc_bf16(&device, q_elems, "out_a");
    fill_bf16_buffer(&q_buf_a, &q_bf);
    fill_bf16_buffer(&k_buf_a, &k_bf);
    fill_bf16_buffer(&v_buf_a, &v_bf);

    let q_buf_b = alloc_bf16(&device, q_elems, "Q_b");
    let k_buf_b = alloc_bf16(&device, kv_elems, "K_b");
    let v_buf_b = alloc_bf16(&device, kv_elems, "V_b");
    let mut out_b = alloc_bf16(&device, q_elems, "out_b");
    fill_bf16_buffer(&q_buf_b, &q_bf);
    fill_bf16_buffer(&k_buf_b, &k_bf);
    fill_bf16_buffer(&v_buf_b, &v_bf);

    let params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false,
    };

    // Run A — blk = None (pre-Wave-2E path).
    let mut enc_a = device.command_encoder().expect("encoder_a");
    dispatch_flash_attn_prefill_bf16_d256_with_blk(
        &mut enc_a, &device, &mut registry,
        &q_buf_a, &k_buf_a, &v_buf_a,
        Some(&mask_buf), None,
        &mut out_a, &params,
    ).expect("dispatch without blk");
    enc_a.commit_and_wait().expect("commit_a");

    // Run B — blk = Some(built_blk).
    let mut enc_b = device.command_encoder().expect("encoder_b");
    dispatch_flash_attn_prefill_bf16_d256_with_blk(
        &mut enc_b, &device, &mut registry,
        &q_buf_b, &k_buf_b, &v_buf_b,
        Some(&mask_buf), Some(&blk_buf),
        &mut out_b, &params,
    ).expect("dispatch with blk");
    enc_b.commit_and_wait().expect("commit_b");

    // BIT-EXACT comparison — the blk path must not change any output bit.
    let out_a_data = read_bf16_buffer(&out_a, q_elems);
    let out_b_data = read_bf16_buffer(&out_b, q_elems);

    let mut mismatches = 0usize;
    let mut first_bad = None;
    for i in 0..q_elems {
        if out_a_data[i].to_bits() != out_b_data[i].to_bits() {
            if first_bad.is_none() {
                first_bad = Some((i, out_a_data[i], out_b_data[i]));
            }
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        if let Some((idx, a, b)) = first_bad {
            panic!(
                "test_gpu_bf16_d256_with_blk_matches_no_blk: \
                 {mismatches}/{q_elems} elements differ (blk=None vs \
                 blk=Some). first mismatch at idx={idx}: \
                 no_blk=0x{:04X} ({}) vs with_blk=0x{:04X} ({})",
                a.to_bits(), a.to_f32(),
                b.to_bits(), b.to_f32()
            );
        }
    }
    eprintln!(
        "test_gpu_bf16_d256_with_blk_matches_no_blk: PASS — all {q_elems} \
         bf16 elements bit-identical"
    );
}

/// Correctness gate: D=512 main kernel output MUST be bit-identical with
/// blk=None vs blk=Some(built_blk).  The D=512 analogue of the D=256 test.
///
/// Uses a modest shape (qL=kL=64, d=512) so bit-exact comparison is fast.
/// Tile shape for D=512 pre-pass: (BQ=8, BK=64) — matches the main kernel's
/// NQPSG=8 Q-tile and NCPSG=64 KV-chunk size.
#[test]
fn test_gpu_bf16_d512_with_blk_matches_no_blk() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1;
    // D=512 main kernel's NQPSG=8 and NCPSG=64 — size ql so the trailing
    // Q-chunk is exact (ql % 8 == 0) and kl so the trailing KV-chunk is
    // partial (kl=64 → exactly one full chunk, no partial).
    let ql = 64usize; let kl = 64usize; let d = 512;
    let window = 32_u32;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 700, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 701, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 702, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Build mask + blk for D=512 geometry: BQ=8, BK=64 (one per KV-chunk).
    let (mask_buf, blk_buf) = build_mask_and_blk(
        &device, &mut registry,
        ql as u32, kl as u32,
        Some(window), true,
        8, 64,
    );

    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;

    let q_buf_a = alloc_bf16(&device, q_elems, "Q_a");
    let k_buf_a = alloc_bf16(&device, kv_elems, "K_a");
    let v_buf_a = alloc_bf16(&device, kv_elems, "V_a");
    let mut out_a = alloc_bf16(&device, q_elems, "out_a");
    fill_bf16_buffer(&q_buf_a, &q_bf);
    fill_bf16_buffer(&k_buf_a, &k_bf);
    fill_bf16_buffer(&v_buf_a, &v_bf);

    let q_buf_b = alloc_bf16(&device, q_elems, "Q_b");
    let k_buf_b = alloc_bf16(&device, kv_elems, "K_b");
    let v_buf_b = alloc_bf16(&device, kv_elems, "V_b");
    let mut out_b = alloc_bf16(&device, q_elems, "out_b");
    fill_bf16_buffer(&q_buf_b, &q_bf);
    fill_bf16_buffer(&k_buf_b, &k_bf);
    fill_bf16_buffer(&v_buf_b, &v_bf);

    let params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false,
    };

    let mut enc_a = device.command_encoder().expect("encoder_a");
    dispatch_flash_attn_prefill_bf16_d512_with_blk(
        &mut enc_a, &device, &mut registry,
        &q_buf_a, &k_buf_a, &v_buf_a,
        Some(&mask_buf), None,
        &mut out_a, &params,
    ).expect("dispatch D512 without blk");
    enc_a.commit_and_wait().expect("commit_a");

    let mut enc_b = device.command_encoder().expect("encoder_b");
    dispatch_flash_attn_prefill_bf16_d512_with_blk(
        &mut enc_b, &device, &mut registry,
        &q_buf_b, &k_buf_b, &v_buf_b,
        Some(&mask_buf), Some(&blk_buf),
        &mut out_b, &params,
    ).expect("dispatch D512 with blk");
    enc_b.commit_and_wait().expect("commit_b");

    let out_a_data = read_bf16_buffer(&out_a, q_elems);
    let out_b_data = read_bf16_buffer(&out_b, q_elems);

    let mut mismatches = 0usize;
    let mut first_bad = None;
    for i in 0..q_elems {
        if out_a_data[i].to_bits() != out_b_data[i].to_bits() {
            if first_bad.is_none() {
                first_bad = Some((i, out_a_data[i], out_b_data[i]));
            }
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        if let Some((idx, a, b)) = first_bad {
            panic!(
                "test_gpu_bf16_d512_with_blk_matches_no_blk: \
                 {mismatches}/{q_elems} elements differ (blk=None vs \
                 blk=Some).  first mismatch at idx={idx}: \
                 no_blk=0x{:04X} ({}) vs with_blk=0x{:04X} ({})",
                a.to_bits(), a.to_f32(),
                b.to_bits(), b.to_f32()
            );
        }
    }
    eprintln!(
        "test_gpu_bf16_d512_with_blk_matches_no_blk: PASS — all {q_elems} \
         bf16 elements bit-identical"
    );
}

/// Gemma 4 sliding-prefill shape: seq_len=2455, window=1024 at D=256.
/// Build mask + blk + dispatch flash_attn_prefill.  Verify:
///   1. Pre-pass classification matches CPU reference (spot-checked rows).
///   2. Tile-skip rate is ≥ 55% (ADR-011 predicts ~58.5%).
///   3. GPU output with blk enabled matches the bf16 tolerance reference.
#[test]
fn test_blk_gemma4_sliding_prefill() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);

    let batch = 1; let h = 1; let kv_h = 1;
    let ql = 2455usize; let kl = 2455usize; let d = 256;
    let window = 1024_u32;
    let scale = 1.0 / (d as f32).sqrt();

    // Build mask + blk.
    let (mask_buf, blk_buf) = build_mask_and_blk(
        &device, &mut registry,
        ql as u32, kl as u32,
        Some(window), true,
        32, 16,
    );

    // Measure tile-skip rate.
    let nq_tiles = (ql + 31) / 32;
    let nk_tiles = (kl + 15) / 16;
    let blk_data = read_u8_buffer(&blk_buf, nq_tiles * nk_tiles);
    let zero_count = blk_data.iter().filter(|&&b| b == 0).count();
    let two_count  = blk_data.iter().filter(|&&b| b == 2).count();
    let one_count  = blk_data.iter().filter(|&&b| b == 1).count();
    let total = nq_tiles * nk_tiles;
    let skip_pct = (zero_count as f64 / total as f64) * 100.0;
    let attn_pct = (two_count  as f64 / total as f64) * 100.0;
    let mix_pct  = (one_count  as f64 / total as f64) * 100.0;

    eprintln!(
        "test_blk_gemma4_sliding_prefill: ql={ql} kl={kl} window={window} \
         nq={nq_tiles} nk={nk_tiles} total={total}\n\
         classification — skip(0): {zero_count} ({:.1}%), \
         mixed(1): {one_count} ({:.1}%), \
         all_attended(2): {two_count} ({:.1}%)",
        skip_pct, mix_pct, attn_pct
    );

    // ADR-011 predicts ~58.5% skip for Gemma 4 at ql=2455, window=1024.
    // Require ≥55% as a practical floor (slight variation from boundary
    // rounding at tile edges).
    assert!(
        skip_pct >= 55.0,
        "Gemma 4 sliding mask tile-skip rate too low: {skip_pct:.1}% \
         (ADR-011 predicts ~58.5%)"
    );

    // Cross-check a few tiles with CPU classifier (not all 11858 — that
    // would dwarf the actual GPU dispatch cost).
    let mask_data = read_bf16_buffer(&mask_buf, ql * kl);
    let cpu_blk = cpu_classify_blk(&mask_data, ql, kl, 32, 16);
    for &(qt, kt) in &[
        (0usize, 0usize),
        (0, nk_tiles - 1),
        (nq_tiles / 2, 0),
        (nq_tiles / 2, nk_tiles / 2),
        (nq_tiles - 1, nk_tiles - 1),
        (10, 5),
        (50, 80),
    ] {
        let i = qt * nk_tiles + kt;
        assert_eq!(
            blk_data[i], cpu_blk[i],
            "spot-check mismatch at (qt={qt}, kt={kt}): GPU={}, CPU={}",
            blk_data[i], cpu_blk[i]
        );
    }

    // ── End-to-end GPU dispatch with blk enabled ──────────────────────────
    let q = pseudo_random_f32(SEED + 800, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 801, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 802, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");
    fill_bf16_buffer(&q_buf, &q_bf);
    fill_bf16_buffer(&k_buf, &k_bf);
    fill_bf16_buffer(&v_buf, &v_bf);

    let params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false,
    };

    let mut enc = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d256_with_blk(
        &mut enc, &device, &mut registry,
        &q_buf, &k_buf, &v_buf,
        Some(&mask_buf), Some(&blk_buf),
        &mut out_buf, &params,
    ).expect("gemma4 dispatch with blk");
    enc.commit_and_wait().expect("commit");

    let gpu_out_bf = read_bf16_buffer(&out_buf, q_elems);
    let gpu_out = bf16_to_f32(&gpu_out_bf);

    // CPU reference: apply the same mask as an additive term.
    let mask_f32: Vec<f32> = mask_data.iter().map(|x| x.to_f32()).collect();
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&f32_to_bf16(&mask_f32)),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "blk_gemma4_sliding_prefill",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  WAVE 4.1 — RANK-2 BROADCAST MASK TESTS (ADR-011 Phase 2, Wave 4.1)
// ─────────────────────────────────────────────────────────────────────────────
//
// These tests verify that `dispatch_flash_attn_prefill_bf16_d{256,512}` accept
// a rank-2 `[qL, kL]` mask produced by `build_sdpa_mask_bf16` and broadcast
// it correctly across all (batch, head) pairs via stride-0 in the batch and
// head dimensions.
//
// The key property: for multi-head (n_heads > 1) attention with a single shared
// mask, every head must attend to exactly the same (q, k) positions — the GPU
// output must match a CPU reference that replicates the mask across all heads.
//
// Back-compat test: a rank-4 `[1, H, qL, kL]` mask (per-head layout) must
// still work via the old code path and produce the same result.

/// CPU reference helper for rank-2 broadcast mask: replicate `[ql, kl]` mask
/// across all batch items and heads to produce the `[B, H, qL, kL]` layout
/// that `sdpa_reference_f32` / `reference_for_bf16` expects.
fn broadcast_mask_to_rank4(
    mask_rank2: &[f32],
    batch: usize,
    n_heads: usize,
    ql: usize,
    kl: usize,
) -> Vec<f32> {
    assert_eq!(mask_rank2.len(), ql * kl, "broadcast_mask_to_rank4: mask must be [qL, kL]");
    let mut out = vec![0.0f32; batch * n_heads * ql * kl];
    for b in 0..batch {
        for h in 0..n_heads {
            let dst_base = b * n_heads * ql * kl + h * ql * kl;
            out[dst_base..dst_base + ql * kl].copy_from_slice(mask_rank2);
        }
    }
    out
}

/// Wave 4.1 / Test 1: rank-2 broadcast mask with multi-head D=256.
///
/// batch=1, n_heads=8 (multi-head), ql=kl=128, D=256.  Builds a causal+SWA
/// mask via `build_sdpa_mask_bf16` (which produces rank-2 `[128, 128]`).
/// Dispatches `dispatch_flash_attn_prefill_bf16_d256` and verifies GPU output
/// against a CPU reference that replicates the same mask across all 8 heads.
///
/// Critical: this test catches incorrect stride calculations that would cause
/// per-head indexing to walk off the single-plane buffer.
#[test]
fn test_mask_rank2_broadcast_d256_multihead() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);

    let batch = 1;
    let h = 8;
    let kv_h = 8; // MHA (no GQA) to keep the reference simple.
    let ql = 128usize;
    let kl = 128usize;
    let d = 256usize;
    let window = 64_u32;
    let scale = 1.0 / (d as f32).sqrt();

    // Build the rank-2 `[ql, kl]` causal+SWA mask.
    let mask_params = SdpaMaskParams {
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        window_size: Some(window),
        causal: true,
        q_abs_offset: 0,
    };
    let mut mask_enc = device.command_encoder().expect("mask encoder");
    let mask_buf = build_sdpa_mask_bf16(&device, &mut registry, &mut mask_enc, &mask_params)
        .expect("build_sdpa_mask_bf16");
    mask_enc.commit_and_wait().expect("mask commit");

    // Verify the mask shape is rank-2 (confirms Wave 2D contract).
    assert_eq!(mask_buf.shape(), &[ql, kl], "mask must be rank-2 [qL, kL]");

    // Random Q/K/V.
    let q = pseudo_random_f32(SEED + 900, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 901, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 902, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // GPU dispatch — rank-2 mask, multi-head.
    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");
    fill_bf16_buffer(&q_buf, &q_bf);
    fill_bf16_buffer(&k_buf, &k_bf);
    fill_bf16_buffer(&v_buf, &v_bf);

    let attn_params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false, // causal baked into mask
    };

    let mut enc = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d256(
        &mut enc, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, Some(&mask_buf), &mut out_buf, &attn_params,
    ).expect("dispatch bf16 d256 rank-2 broadcast");
    enc.commit_and_wait().expect("commit");

    let gpu_out_bf = read_bf16_buffer(&out_buf, q_elems);
    let gpu_out = bf16_to_f32(&gpu_out_bf);

    // CPU reference: broadcast the rank-2 mask across all heads.
    let mask_f32: Vec<f32> = read_bf16_buffer(&mask_buf, ql * kl)
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let mask_rank4_f32 = broadcast_mask_to_rank4(&mask_f32, batch, h, ql, kl);
    let mask_rank4_bf = f32_to_bf16(&mask_rank4_f32);
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&mask_rank4_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_all_finite(&gpu_out, "rank2_broadcast_d256_multihead");
    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "rank2_broadcast_d256_multihead",
    );
}

/// Wave 4 Phase B regression: pp65536 mask indexing must not overflow i32.
///
/// # Bug being guarded
///
/// Phase A (`/tmp/cfa-cfa-20260427-adr005-wave4/phase-A-report.md` §2.5)
/// identified two i32 multiplication overflow sites in the D=256 prefill path
/// that fire at `seq_len_q >= 32768` when the mask is the rank-2 broadcast
/// layout `[qL, kL]` (the layout the batched-prefill production code path
/// uses):
///
/// 1. `flash_attn_prefill.metal:1487-1495` — `MMAFrag_t::load_safe` computes
///    `(off_x + i) * str_x` where `off_x = row_pos` (int) and `str_x =
///    int(M_strides[2]) = kL`.  At `row_pos * kL >= i32::MAX = 2^31 - 1`,
///    i.e. `row_pos >= 32768` for `kL = 65536`, the i32 product wraps
///    negative.  The resulting pointer offset addresses memory **before**
///    the mask buffer; the bf16 garbage read perturbs the additive mask
///    contribution and corrupts the attention output for every Q row in
///    the upper half of the prompt.
///
/// 2. `flash_attn_prefill_blk.metal:181` — `mask_src = mask + (qt * BQ) *
///    M_stride + ...` where `qt`, `BQ`, `M_stride` are all `int`.  At pp65536
///    with `qt >= 1024`, `qt * BQ * M_stride >= 1024 * 32 * 65536 =
///    2,147,483,648 > i32::MAX`, same wraparound, same corruption pattern.
///
/// The D=512 sibling kernel is correct because
/// `flash_attn_prefill_d512.metal:411-413` casts each multiplicand to
/// `ulong` before multiplication.  Phase B's fix mirrors that idiom in the
/// two D=256 sites.
///
/// # Failure mode pre-patch
///
/// On a from-scratch single-pass prefill at `seq_len_q = seq_len_k = 65536`
/// with a Gemma-4-style sliding-window+causal rank-2 mask (`window = 1024`,
/// `causal = true`), the LAST Q row (index 65535) attends exactly 1024 K
/// positions (`[64512, 65535]` inclusive) — **all of which** lie inside the
/// upper-half overflow regime (row_pos >= 32768).  The bf16 output for row
/// 65535 diverges
/// from the CPU reference by orders of magnitude (random bf16 garbage in
/// the additive-mask term flips softmax winners arbitrarily).
///
/// Reproduces the empirical observation in
/// `project_long_prefill_parity_inverts.md` that pp65536 emits
/// `first_decode_token = 0` deterministically across 5 cold-process runs:
/// the corrupted-but-finite forward pass yields a stable wrong argmax over
/// the lm_head logits.
///
/// # Why we check only the last Q row
///
/// Memory budget (cold synthetic test):
///   - bf16 rank-2 mask buffer: 65536 * 65536 * 2 B = 8.6 GB
///   - bf16 Q/K/V/O (n_heads = 1, head_dim = 256): 4 * 65536 * 256 * 2 B = 128 MB
///   - f32 K/V for CPU reference (last-row-only): 2 * 65536 * 256 * 4 B = 128 MB
///   - f32 mask row 65535 only: 65536 * 4 B = 256 KB
///   - Total peak: ~9 GB GPU + ~256 MB CPU.
///
/// CPU reference for the full GPU output would be 65536 * 33 M ops =
/// 2.1 T ops at f32 — minutes on a single core.  Reference for the LAST
/// row only is 33 M ops, ~50 ms on Apple Silicon.  The last row is the
/// row that `forward_prefill_batched.rs` argmax-samples to compute
/// `first_decode_token`, so it is the production-relevant row — perfect
/// for a focused regression test.
///
/// # Acceptance
///
/// Pre-patch (current HEAD): FAILS with very large max abs error in row
/// 65535 (the corrupted region).  Post-patch (i64/ulong cast in the two
/// shaders): PASSES within the standard `BF16_GPU_ATOL / _RTOL` budget.
#[test]
fn flash_attn_prefill_pp65536_no_overflow_in_mask_indexing() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);

    let batch = 1;
    let h = 1;
    let kv_h = 1;
    let ql = 65536usize; // The smallest seq_len at which kL * row_pos > i32::MAX.
    let kl = 65536usize;
    let d = 256usize; // Affected path; D=512 is correct already.
    let window = 1024_u32; // Gemma-4 sliding-layer window.
    let scale = 1.0 / (d as f32).sqrt();

    eprintln!(
        "pp65536_overflow_test: allocating ~{} GB GPU (mask {} GB + Q/K/V/O {} MB)",
        ((kl * ql * 2 + 4 * h * ql * d * 2) as f64) / 1e9,
        (kl * ql * 2) as f64 / 1e9,
        (4 * h * ql * d * 2) / 1_048_576,
    );

    // ── Build the rank-2 [qL, kL] sliding+causal mask via the production builder.
    //     This is the same builder forward_prefill_batched.rs uses, so the
    //     mask layout under test is bit-identical to the production layout.
    let mask_params = SdpaMaskParams {
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        window_size: Some(window),
        causal: true,
        q_abs_offset: 0,
    };
    let mut mask_enc = device.command_encoder().expect("mask encoder");
    let mask_buf = build_sdpa_mask_bf16(&device, &mut registry, &mut mask_enc, &mask_params)
        .expect("build_sdpa_mask_bf16 @ pp65536");
    mask_enc.commit_and_wait().expect("mask commit");
    assert_eq!(
        mask_buf.shape(),
        &[ql, kl],
        "mask must be rank-2 [qL, kL] for the rank-2 broadcast code path"
    );

    // ── Random Q/K/V (deterministic seed) ────────────────────────────────
    let q = pseudo_random_f32(SEED + 4096, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 4097, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 4098, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // ── GPU dispatch — single-pass prefill, rank-2 broadcast mask ────────
    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");
    fill_bf16_buffer(&q_buf, &q_bf);
    fill_bf16_buffer(&k_buf, &k_bf);
    fill_bf16_buffer(&v_buf, &v_bf);

    let attn_params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false, // causal+SWA both encoded into the rank-2 mask
    };

    let mut enc = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d256(
        &mut enc,
        &device,
        &mut registry,
        &q_buf,
        &k_buf,
        &v_buf,
        Some(&mask_buf),
        &mut out_buf,
        &attn_params,
    )
    .expect("dispatch bf16 d256 @ pp65536");
    enc.commit_and_wait().expect("commit");

    // ── Read the LAST Q row only ─────────────────────────────────────────
    //     forward_prefill_batched.rs argmax-samples row (seq_len_q - 1) to
    //     compute first_decode_token, so this is the production-relevant
    //     row.  Layout is [B, H, qL, D] contiguous in D, so row 65535 starts
    //     at element offset (qL - 1) * D when batch=1, h=1.
    let last_row_offset = (ql - 1) * d;
    let out_all_bf = read_bf16_buffer(&out_buf, q_elems);
    let gpu_last_row: Vec<f32> = out_all_bf[last_row_offset..last_row_offset + d]
        .iter()
        .map(|x| x.to_f32())
        .collect();

    assert_all_finite(&gpu_last_row, "pp65536_no_overflow_last_row_finite");

    // ── CPU reference for the LAST row only ──────────────────────────────
    //     attention(q[L-1], K, V, mask[L-1, :]) at scale * log2(e), base-2
    //     softmax, finite-M sentinel — mirrors `sdpa_reference_f32` for one
    //     row only to keep memory and compute bounded.
    let q_last: Vec<f32> = q_bf[last_row_offset..last_row_offset + d]
        .iter()
        .map(|x| x.to_f32())
        .collect();
    // K/V are MHA (kv_h = 1), layout [B, kv_h, kL, D].
    let k_f32 = bf16_to_f32(&k_bf);
    let v_f32 = bf16_to_f32(&v_bf);
    // Read the full mask back (8.6 GB) then extract the last row.
    // Note: read_bf16_buffer reads the entire buffer into a Vec — this IS an
    // 8.6 GB host-side allocation.  The mask layout is contiguous [qL, kL]
    // row-major, so row (qL-1) occupies elements [(qL-1)*kL .. qL*kL].
    // A future optimisation could map only those kL elements via contents_ptr
    // to avoid the full copy; for now correctness is the priority.
    let mask_last_row_bf: Vec<half::bf16> = {
        let mask_all = read_bf16_buffer(&mask_buf, ql * kl);
        mask_all[(ql - 1) * kl..ql * kl].to_vec()
    };
    let mask_last_row_f32: Vec<f32> = mask_last_row_bf.iter().map(|x| x.to_f32()).collect();

    const LOG2E: f32 = std::f32::consts::LOG2_E;
    let q_scale = scale * LOG2E;

    // Q · K^T for the last query row, base-2 scale.
    let mut scores = vec![0.0f32; kl];
    for k_pos in 0..kl {
        let mut dot = 0.0f32;
        let kv_base = k_pos * d;
        for di in 0..d {
            dot += q_last[di] * k_f32[kv_base + di];
        }
        scores[k_pos] = dot * q_scale + LOG2E * mask_last_row_f32[k_pos];
    }

    // Online softmax (base-2, finite-M sentinel).
    let mut max_score = f32::MIN / 2.0;
    for &s in &scores {
        if s > max_score {
            max_score = s;
        }
    }
    let exp_scores: Vec<f32> = scores.iter().map(|&s| f32::exp2(s - max_score)).collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    let safe_sum = if sum_exp == 0.0 { 1.0 } else { sum_exp };

    // Weighted sum of V for the output row.
    let mut ref_last_row = vec![0.0f32; d];
    for di in 0..d {
        let mut acc = 0.0f32;
        for k_pos in 0..kl {
            acc += (exp_scores[k_pos] / safe_sum) * v_f32[k_pos * d + di];
        }
        ref_last_row[di] = acc;
    }

    // Apply the bf16 store-rounding the GPU does on output.
    let ref_last_row_bf = f32_to_bf16(&ref_last_row);
    let ref_last_row_after_store: Vec<f32> = ref_last_row_bf.iter().map(|x| x.to_f32()).collect();

    // Pre-patch this assertion will fail with a very large max_abs_error
    // (the bf16 garbage perturbation pushes outputs far past BF16_GPU_ATOL).
    // Post-patch (i64/ulong cast) it should pass with normal bf16 budget.
    assert_close_gpu(
        &gpu_last_row,
        &ref_last_row_after_store,
        BF16_GPU_ATOL,
        BF16_GPU_RTOL,
        "pp65536_no_overflow_in_mask_indexing_last_row",
    );
}

/// Wave 4.5 / Item 1: blk pre-pass pp65536 site-2 regression test.
///
/// ## What this tests
///
/// `flash_attn_prefill_blk.metal` site-2 (line 188-189, fixed in commit
/// 459f550): the pre-pass tile-classification kernel constructs a per-lane
/// mask pointer as
///
/// ```text
/// mask + (int64_t)(qt * BQ) * (int64_t)M_stride + tile_k_start + tiisg;
/// ```
///
/// Before the fix the two operands were plain `int` (i32), making the product
/// `qt * BQ * M_stride` overflow at `qt >= 1024`:
/// `1024 * 32 * 65536 = 2 147 483 648 > i32::MAX = 2 147 483 647`.
/// The signed-overflow wraps to a large negative pointer offset, causing the
/// kernel to read bytes far before the mask buffer's base — producing garbage
/// tile classifications for every Q-tile at index >= 1024.
///
/// ## Analytical correctness predicate (no full-mask readback)
///
/// At `seq_len_q = seq_len_k = 65536`, `window = 1024`, `causal = true`,
/// `BQ = 32`, `BK = 16`:
///
/// - Q-tile `qt = 1024` covers rows `32768..32800` (0-indexed).
/// - For row 32768 with SWA causal mask, the attended K-range is
///   `[32768 - 1024 + 1, 32768] = [31745, 32768]`.
/// - Any K-tile with `tile_k_end = (kt + 1) * 16 <= 31744` lies entirely
///   before the attended window for EVERY row in this Q-tile, so every mask
///   cell in that tile is `-inf` and the tile must be classified **0 (skip)**.
/// - `kt <= 1983` satisfies `(kt + 1) * 16 <= 31744`.
///
/// Pre-patch: the i32 overflow makes the kernel read garbage bytes for rows at
/// qt=1024 (and all higher Q-tiles).  Garbage mask values are non-`-inf`,
/// flipping some or all of these tiles from class 0 to class 1 or 2 — the
/// assertion below catches that.
///
/// Post-patch (459f550): the i64-widened pointer arithmetic is correct; every
/// tile at `qt=1024, kt<=1983` reads genuine `-inf` cells and classifies as 0.
///
/// ## Memory budget
///
/// - bf16 mask buffer: 65536 × 65536 × 2 B = 8.6 GB
/// - blk classification buffer: ceil(65536/32) × ceil(65536/16) = 2048 × 4096
///   = 8 MB
/// - Total peak: ~8.6 GB GPU.  Test is guarded by a 14 GB free-RAM check.
///
/// ## Acceptance
///
/// Pre-patch: FAILS — tiles at `qt=1024, kt<=1983` that should be 0 are
/// reported as non-zero (garbage read).  Post-patch: PASSES — all such tiles
/// are 0.
#[test]
fn flash_attn_prefill_blk_pp65536_no_overflow_in_mask_indexing() {
    // ── RAM guard ────────────────────────────────────────────────────────────
    // The mask buffer alone is 8.6 GB.  Refuse to run if available
    // (free + inactive) physical pages sum to less than 14 GB to avoid
    // jetsam-killing the test process.
    {
        use std::process::Command;
        let vm = Command::new("vm_stat")
            .output()
            .expect("vm_stat failed");
        let out = String::from_utf8_lossy(&vm.stdout);
        let pages: u64 = {
            let mut free: u64 = 0;
            let mut inactive: u64 = 0;
            for line in out.lines() {
                if line.starts_with("Pages free:") {
                    free = line
                        .split_whitespace()
                        .last()
                        .and_then(|s| s.trim_end_matches('.').parse().ok())
                        .unwrap_or(0);
                }
                if line.starts_with("Pages inactive:") {
                    inactive = line
                        .split_whitespace()
                        .last()
                        .and_then(|s| s.trim_end_matches('.').parse().ok())
                        .unwrap_or(0);
                }
            }
            free + inactive
        };
        // 16 KiB pages; 14 GB = 14 * 1024 * 1024 * 1024 / 16384 pages
        let min_pages: u64 = (14u64 * 1024 * 1024 * 1024) / 16384;
        let free_gb = (pages as f64 * 16384.0) / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "flash_attn_prefill_blk_pp65536: free+inactive = {:.1} GB (need >= 14 GB)",
            free_gb
        );
        if pages < min_pages {
            eprintln!(
                "SKIP: only {:.1} GB free — need >= 14 GB for the 8.6 GB mask buffer",
                free_gb
            );
            return;
        }
    }

    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);

    let ql = 65536_u32;
    let kl = 65536_u32;
    let window = 1024_u32; // Gemma-4 SWA window.
    // BQ=32 / BK=16 must match the tile shape used by the D=256 main kernel
    // (dispatch_flash_attn_prefill_bf16_d256_with_blk).
    let bq = 32_u32;
    let bk = 16_u32;

    eprintln!(
        "flash_attn_prefill_blk_pp65536: allocating mask {:.1} GB + blk {} KB",
        (kl as f64 * ql as f64 * 2.0) / 1e9,
        (ql.div_ceil(bq) as usize * kl.div_ceil(bk) as usize) / 1024,
    );

    // ── Build the SWA+causal mask (same builder as production) ───────────────
    let (mask_buf, blk_buf) = build_mask_and_blk(
        &device,
        &mut registry,
        ql,
        kl,
        Some(window),
        true,
        bq,
        bk,
    );
    let _ = &mask_buf; // keep alive

    // ── Read back the blk classification buffer (8 MB — cheap) ──────────────
    let nq_tiles = ql.div_ceil(bq) as usize; // 2048
    let nk_tiles = kl.div_ceil(bk) as usize; // 4096
    let blk_data = read_u8_buffer(&blk_buf, nq_tiles * nk_tiles);

    // ── Analytical correctness check at the overflow boundary ────────────────
    //
    // Q-tile qt=1024 covers rows 32768..32800.  The earliest attended K
    // position for any of those rows (worst case: row 32768, SWA window=1024)
    // is K=31745.  K-tiles with tile_k_end = (kt+1)*16 <= 31744 are entirely
    // outside the attended window for every row in qt=1024, so their mask
    // cells are all -inf and classification MUST be 0 (skip).
    //
    // kt <= 1983  ↔  (1983+1)*16 = 31744 <= 31744  ✓
    //
    // We check a representative sample of such tiles: kt=0 (the far corner,
    // maximum distance from the diagonal) and kt=1983 (the tile just before the
    // window boundary) at qt=1024.
    //
    // Pre-patch: i32 overflow at qt=1024 corrupts the mask pointer → garbage
    // reads → mmin/mmax are non-inf → tile class becomes 1 or 2 instead of 0.
    // Post-patch (459f550): i64-widened arithmetic reads genuine -inf cells →
    // tile class is 0.
    let qt_boundary = 1024_usize; // first Q-tile that overflows pre-patch
    let kt_far     = 0_usize;    // K-tile far from diagonal (should be 0)
    let kt_near_edge = 1983_usize; // last K-tile fully outside the window (should be 0)

    let class_far = blk_data[qt_boundary * nk_tiles + kt_far];
    let class_near_edge = blk_data[qt_boundary * nk_tiles + kt_near_edge];

    assert_eq!(
        class_far, 0,
        "blk[qt=1024][kt=0] must be 0 (fully-masked: far from diagonal at pp65536). \
         Got {}. Pre-patch i32 overflow reads garbage mask bytes and produces wrong \
         classification (non-zero) at this tile.",
        class_far
    );
    assert_eq!(
        class_near_edge, 0,
        "blk[qt=1024][kt=1983] must be 0 (fully-masked: still outside SWA window). \
         Got {}. Pre-patch i32 overflow reads garbage mask bytes.",
        class_near_edge
    );

    // ── Sanity: tiles near the diagonal at qt=1024 must NOT all be 0 ─────────
    // Tiles on or just below the diagonal should be class 1 (mixed) or 2
    // (all-attended).  Verify at least one non-zero classification exists in
    // the rows around qt=1024 to confirm the blk buffer is generally alive.
    let qt_range_start = 1020_usize;
    let qt_range_end   = 1028_usize.min(nq_tiles);
    let any_nonzero = (qt_range_start..qt_range_end).any(|qt| {
        let row_start = qt * bq as usize; // first q row of this tile
        // K-tiles that overlap the window: kt where (kt*BK) is around row_start
        let kt_window = row_start.saturating_sub(window as usize) / bk as usize;
        let kt_end = (row_start / bk as usize + 2).min(nk_tiles);
        (kt_window..kt_end).any(|kt| blk_data[qt * nk_tiles + kt] != 0)
    });

    assert!(
        any_nonzero,
        "Expected some non-zero blk classifications near the diagonal at qt~1024 \
         (window/mixed tiles). All zero would indicate the blk buffer is zeroed out \
         or the dispatch didn't run."
    );

    eprintln!(
        "flash_attn_prefill_blk_pp65536: PASS — \
         blk[1024][0]={class_far}, blk[1024][1983]={class_near_edge} (both 0 as required)"
    );
}

/// Wave 4.1 / Test 2: rank-2 broadcast mask with multi-head D=512.
///
/// Same as Test 1 but routes through `dispatch_flash_attn_prefill_bf16_d512`.
/// batch=1, n_heads=8, ql=kl=128, D=512.
#[test]
fn test_mask_rank2_broadcast_d512_multihead() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill_d512::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);

    let batch = 1;
    let h = 8;
    let kv_h = 8;
    let ql = 128usize;
    let kl = 128usize;
    let d = 512usize;
    let window = 64_u32;
    let scale = 1.0 / (d as f32).sqrt();

    // Build the rank-2 `[ql, kl]` causal+SWA mask.
    let mask_params = SdpaMaskParams {
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        window_size: Some(window),
        causal: true,
        q_abs_offset: 0,
    };
    let mut mask_enc = device.command_encoder().expect("mask encoder");
    let mask_buf = build_sdpa_mask_bf16(&device, &mut registry, &mut mask_enc, &mask_params)
        .expect("build_sdpa_mask_bf16");
    mask_enc.commit_and_wait().expect("mask commit");

    assert_eq!(mask_buf.shape(), &[ql, kl], "mask must be rank-2 [qL, kL]");

    let q = pseudo_random_f32(SEED + 910, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 911, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 912, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // GPU dispatch — rank-2 mask, multi-head, D=512.
    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");
    fill_bf16_buffer(&q_buf, &q_bf);
    fill_bf16_buffer(&k_buf, &k_bf);
    fill_bf16_buffer(&v_buf, &v_bf);

    let attn_params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false,
    };

    let mut enc = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d512(
        &mut enc, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, Some(&mask_buf), &mut out_buf, &attn_params,
    ).expect("dispatch bf16 d512 rank-2 broadcast");
    enc.commit_and_wait().expect("commit");

    let gpu_out_bf = read_bf16_buffer(&out_buf, q_elems);
    let gpu_out = bf16_to_f32(&gpu_out_bf);

    // CPU reference: broadcast the rank-2 mask across all heads.
    let mask_f32: Vec<f32> = read_bf16_buffer(&mask_buf, ql * kl)
        .iter()
        .map(|x| x.to_f32())
        .collect();
    let mask_rank4_f32 = broadcast_mask_to_rank4(&mask_f32, batch, h, ql, kl);
    let mask_rank4_bf = f32_to_bf16(&mask_rank4_f32);
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&mask_rank4_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_all_finite(&gpu_out, "rank2_broadcast_d512_multihead");
    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL_D512, BF16_GPU_RTOL_D512,
        "rank2_broadcast_d512_multihead",
    );
}

/// Wave 4.1 / Test 3: rank-4 per-head mask is still accepted and honoured
/// (back-compat regression gate).
///
/// Manually allocates a rank-4 `[1, H, qL, kL]` mask buffer with a
/// checkerboard bias pattern, dispatches `dispatch_flash_attn_prefill_bf16_d256`,
/// and verifies the output matches the CPU reference.  Any regression in the
/// rank-4 code path will be caught here.
#[test]
fn test_mask_rank4_preserved_regression() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1;
    let h = 4;
    let kv_h = 4;
    let ql = 128usize;
    let kl = 128usize;
    let d = 256usize;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 920, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 921, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 922, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Build a rank-4 [B, H, qL, kL] checkerboard mask directly.
    // Odd positions get -1e4 (effectively masked), even get 0.0.
    let mask_elems = batch * h * ql * kl;
    let mask_f32: Vec<f32> = (0..mask_elems)
        .map(|i| if i % 2 == 0 { 0.0_f32 } else { -1.0e4_f32 })
        .collect();
    let mask_bf = f32_to_bf16(&mask_f32);

    // Allocate with a rank-4 shape so the dispatcher takes the per-head path.
    let mask_buf = device
        .alloc_buffer(
            mask_elems * 2,
            DType::BF16,
            vec![batch, h, ql, kl],
        )
        .expect("alloc rank-4 mask buffer");
    fill_bf16_buffer(&mask_buf, &mask_bf);

    // Verify the buffer was allocated with rank-4 shape.
    assert_eq!(mask_buf.shape(), &[batch, h, ql, kl], "mask shape must be rank-4");
    assert_eq!(mask_buf.shape().len(), 4, "mask must have 4 dimensions");

    let q_elems = batch * h * ql * d;
    let kv_elems = batch * kv_h * kl * d;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");
    fill_bf16_buffer(&q_buf, &q_bf);
    fill_bf16_buffer(&k_buf, &k_bf);
    fill_bf16_buffer(&v_buf, &v_bf);

    let attn_params = FlashAttnPrefillParams {
        n_heads: h as u32,
        n_kv_heads: kv_h as u32,
        head_dim: d as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal: false,
    };

    let mut enc = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d256(
        &mut enc, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, Some(&mask_buf), &mut out_buf, &attn_params,
    ).expect("dispatch bf16 d256 rank-4 back-compat");
    enc.commit_and_wait().expect("commit");

    let gpu_out_bf = read_bf16_buffer(&out_buf, q_elems);
    let gpu_out = bf16_to_f32(&gpu_out_bf);

    // CPU reference uses the same rank-4 mask directly (already [B*H*qL*kL]).
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_all_finite(&gpu_out, "rank4_preserved_regression");
    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "rank4_preserved_regression",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// § D=64 GPU CORRECTNESS — BERT/embedding family
// ─────────────────────────────────────────────────────────────────────────────
//
// Validates the new D=64 dispatcher (`dispatch_flash_attn_prefill_bf16_d64`)
// against the bf16-rounded CPU reference at the same tolerance budget as
// D=256.  Two layout paths (HeadMajor and SeqMajor) are tested separately
// — both share the kernel binary but use different stride math, so a
// stride bug in one cannot be masked by a passing test on the other.

use mlx_native::ops::flash_attn_prefill::{
    dispatch_flash_attn_prefill_bf16_d64,
    FlashAttnPrefillLayout,
};

/// Run the D=64 dispatcher in HeadMajor layout (`[B, H, L, D]`) — the same
/// layout the D=256/D=512 dispatchers consume.  Returns the bf16-rounded
/// output as f32.
#[allow(clippy::too_many_arguments)]
fn run_bf16_gpu_d64_head_major(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_bf16: &[bf16],
    k_bf16: &[bf16],
    v_bf16: &[bf16],
    mask_bf16: Option<&[bf16]>,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> Vec<f32> {
    assert_eq!(head_dim, 64, "D=64 dispatcher requires head_dim=64");

    let q_elems = batch * n_heads * ql * head_dim;
    let kv_elems = batch * n_kv_heads * kl * head_dim;

    let q_buf = alloc_bf16(device, q_elems, "Q");
    let k_buf = alloc_bf16(device, kv_elems, "K");
    let v_buf = alloc_bf16(device, kv_elems, "V");
    let mut out_buf = alloc_bf16(device, q_elems, "out");

    fill_bf16_buffer(&q_buf, q_bf16);
    fill_bf16_buffer(&k_buf, k_bf16);
    fill_bf16_buffer(&v_buf, v_bf16);

    let mask_buf = mask_bf16.map(|m| {
        let mask_elems = batch * n_heads * ql * kl;
        assert_eq!(m.len(), mask_elems, "mask length mismatch");
        let buf = alloc_bf16(device, mask_elems, "mask");
        fill_bf16_buffer(&buf, m);
        buf
    });

    let params = FlashAttnPrefillParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d64(
        &mut encoder, device, registry,
        &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut out_buf, &params,
        FlashAttnPrefillLayout::HeadMajor,
    ).expect("bf16 D=64 head-major dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    let out_bf16 = read_bf16_buffer(&out_buf, q_elems);
    bf16_to_f32(&out_bf16)
}

/// Transpose `[B, H, L, D]` → `[B, L, H, D]` (head-major to seq-major).
fn head_to_seq_major(
    src: &[bf16], batch: usize, n_heads: usize, l: usize, d: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); batch * l * n_heads * d];
    for b in 0..batch {
        for h in 0..n_heads {
            for s in 0..l {
                for k in 0..d {
                    let src_idx = b * n_heads * l * d + h * l * d + s * d + k;
                    let dst_idx = b * l * n_heads * d + s * n_heads * d + h * d + k;
                    out[dst_idx] = src[src_idx];
                }
            }
        }
    }
    out
}

/// Transpose `[B, L, H, D]` → `[B, H, L, D]` (seq-major back to head-major).
fn seq_to_head_major(
    src: &[bf16], batch: usize, n_heads: usize, l: usize, d: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); batch * n_heads * l * d];
    for b in 0..batch {
        for s in 0..l {
            for h in 0..n_heads {
                for k in 0..d {
                    let src_idx = b * l * n_heads * d + s * n_heads * d + h * d + k;
                    let dst_idx = b * n_heads * l * d + h * l * d + s * d + k;
                    out[dst_idx] = src[src_idx];
                }
            }
        }
    }
    out
}

/// Run the D=64 dispatcher in SeqMajor layout (`[B, L, H, D]`) — the
/// natural BERT/embedding layout.  Caller passes head-major bf16 inputs;
/// helper transposes them in/out so the test math stays in head-major
/// (matching the CPU reference).
#[allow(clippy::too_many_arguments)]
fn run_bf16_gpu_d64_seq_major(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q_bf16_head_major: &[bf16],
    k_bf16_head_major: &[bf16],
    v_bf16_head_major: &[bf16],
    mask_bf16_rank2: Option<&[bf16]>, // rank-2 [qL, kL] broadcast across batch+heads
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    ql: usize,
    kl: usize,
    head_dim: usize,
    scale: f32,
    do_causal: bool,
) -> Vec<f32> {
    assert_eq!(head_dim, 64, "D=64 dispatcher requires head_dim=64");

    let q_elems = batch * n_heads * ql * head_dim;
    let kv_elems = batch * n_kv_heads * kl * head_dim;

    // Transpose inputs to seq-major before binding.
    let q_seq = head_to_seq_major(q_bf16_head_major, batch, n_heads, ql, head_dim);
    let k_seq = head_to_seq_major(k_bf16_head_major, batch, n_kv_heads, kl, head_dim);
    let v_seq = head_to_seq_major(v_bf16_head_major, batch, n_kv_heads, kl, head_dim);

    let q_buf = alloc_bf16(device, q_elems, "Q");
    let k_buf = alloc_bf16(device, kv_elems, "K");
    let v_buf = alloc_bf16(device, kv_elems, "V");
    let mut out_buf = alloc_bf16(device, q_elems, "out");

    fill_bf16_buffer(&q_buf, &q_seq);
    fill_bf16_buffer(&k_buf, &k_seq);
    fill_bf16_buffer(&v_buf, &v_seq);

    // Rank-2 mask buffer: shape [qL, kL].
    let mask_buf = mask_bf16_rank2.map(|m| {
        let mask_elems = ql * kl;
        assert_eq!(m.len(), mask_elems, "rank-2 mask length mismatch");
        let buf = device
            .alloc_buffer(mask_elems * 2, DType::BF16, vec![ql, kl])
            .unwrap_or_else(|e| panic!("alloc rank-2 mask: {e:?}"));
        fill_bf16_buffer(&buf, m);
        buf
    });

    let params = FlashAttnPrefillParams {
        n_heads: n_heads as u32,
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len_q: ql as u32,
        seq_len_k: kl as u32,
        batch: batch as u32,
        scale,
        do_causal,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_flash_attn_prefill_bf16_d64(
        &mut encoder, device, registry,
        &q_buf, &k_buf, &v_buf, mask_buf.as_ref(), &mut out_buf, &params,
        FlashAttnPrefillLayout::SeqMajor,
    ).expect("bf16 D=64 seq-major dispatch");
    encoder.commit_and_wait().expect("commit_and_wait");

    // Read seq-major output, transpose back to head-major for comparison.
    let out_seq = read_bf16_buffer(&out_buf, q_elems);
    let out_head = seq_to_head_major(&out_seq, batch, n_heads, ql, head_dim);
    bf16_to_f32(&out_head)
}

/// D=64 HeadMajor: unmasked, non-causal, multiple seq lengths.
#[test]
fn test_gpu_bf16_d64_head_major_unmasked() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    for &(ql, kl) in &[(32usize, 32usize), (128, 128), (512, 512)] {
        let batch = 1; let h = 12; let kv_h = 12; let d = 64;
        let scale = 1.0 / (d as f32).sqrt();

        let q = pseudo_random_f32(SEED + 90, batch * h * ql * d);
        let k = pseudo_random_f32(SEED + 91, batch * kv_h * kl * d);
        let v = pseudo_random_f32(SEED + 92, batch * kv_h * kl * d);
        let q_bf = f32_to_bf16(&q);
        let k_bf = f32_to_bf16(&k);
        let v_bf = f32_to_bf16(&v);

        let gpu_out = run_bf16_gpu_d64_head_major(
            &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );
        let ref_out = reference_for_bf16(
            &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );

        assert_all_finite(&gpu_out, &format!("d64_hm_unmasked_ql{ql}_kl{kl}"));
        assert_close_gpu(
            &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
            &format!("d64_hm_unmasked_ql{ql}_kl{kl}"),
        );
    }
}

/// D=64 HeadMajor: with rank-4 additive mask (per-head BERT-style padding).
#[test]
fn test_gpu_bf16_d64_head_major_additive_mask() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1; let h = 12; let kv_h = 12; let ql = 128; let kl = 128; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 100, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 101, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 102, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    let mask_f32: Vec<f32> = (0..batch * h * ql * kl)
        .map(|i| if i % 2 == 0 { 0.0_f32 } else { -1.0e4_f32 })
        .collect();
    let mask_bf = f32_to_bf16(&mask_f32);

    let gpu_out = run_bf16_gpu_d64_head_major(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&mask_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_all_finite(&gpu_out, "d64_hm_additive_mask");
    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "d64_hm_additive_mask",
    );
}

/// D=64 SeqMajor: unmasked — verifies the seq-major stride math agrees
/// with head-major (and with the CPU reference).  This is the production
/// path for nomic-bert and friends.
#[test]
fn test_gpu_bf16_d64_seq_major_unmasked() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    for &(ql, kl) in &[(32usize, 32usize), (128, 128), (512, 512)] {
        let batch = 1; let h = 12; let kv_h = 12; let d = 64;
        let scale = 1.0 / (d as f32).sqrt();

        let q = pseudo_random_f32(SEED + 110, batch * h * ql * d);
        let k = pseudo_random_f32(SEED + 111, batch * kv_h * kl * d);
        let v = pseudo_random_f32(SEED + 112, batch * kv_h * kl * d);
        let q_bf = f32_to_bf16(&q);
        let k_bf = f32_to_bf16(&k);
        let v_bf = f32_to_bf16(&v);

        let gpu_out = run_bf16_gpu_d64_seq_major(
            &device, &mut registry, &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );
        let ref_out = reference_for_bf16(
            &q_bf, &k_bf, &v_bf, None,
            batch, h, kv_h, ql, kl, d, scale, false,
        );

        assert_all_finite(&gpu_out, &format!("d64_sm_unmasked_ql{ql}_kl{kl}"));
        assert_close_gpu(
            &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
            &format!("d64_sm_unmasked_ql{ql}_kl{kl}"),
        );
    }
}

/// D=64 SeqMajor: with rank-2 broadcast mask (BERT padding-mask shape).
///
/// nomic-bert and BERT family use a single `[seq_len, seq_len]` padding
/// mask broadcast across batch + heads — verifies the rank-2 path in the
/// dispatcher works at the seq-major stride convention.
#[test]
fn test_gpu_bf16_d64_seq_major_rank2_mask() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let batch = 1; let h = 12; let kv_h = 12; let ql = 64; let kl = 64; let d = 64;
    let scale = 1.0 / (d as f32).sqrt();

    let q = pseudo_random_f32(SEED + 120, batch * h * ql * d);
    let k = pseudo_random_f32(SEED + 121, batch * kv_h * kl * d);
    let v = pseudo_random_f32(SEED + 122, batch * kv_h * kl * d);
    let q_bf = f32_to_bf16(&q);
    let k_bf = f32_to_bf16(&k);
    let v_bf = f32_to_bf16(&v);

    // Rank-2 mask `[ql, kl]`: even-position keys attend, odd-position keys
    // are blocked.  Same checkerboard pattern as the additive_mask test
    // but expressed once (broadcast across all heads).
    let mask_rank2_f32: Vec<f32> = (0..ql * kl)
        .map(|i| if i % 2 == 0 { 0.0_f32 } else { -1.0e4_f32 })
        .collect();
    let mask_rank2_bf = f32_to_bf16(&mask_rank2_f32);

    // CPU reference takes a per-head rank-4 mask: replicate the rank-2
    // plane across all heads + batches so the reference and GPU compute
    // identical things.
    let mut mask_rank4_f32 = vec![0.0_f32; batch * h * ql * kl];
    for b in 0..batch {
        for hd in 0..h {
            for q_pos in 0..ql {
                for k_pos in 0..kl {
                    let dst = b * h * ql * kl + hd * ql * kl + q_pos * kl + k_pos;
                    let src = q_pos * kl + k_pos;
                    mask_rank4_f32[dst] = mask_rank2_f32[src];
                }
            }
        }
    }
    let mask_rank4_bf = f32_to_bf16(&mask_rank4_f32);

    let gpu_out = run_bf16_gpu_d64_seq_major(
        &device, &mut registry, &q_bf, &k_bf, &v_bf, Some(&mask_rank2_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );
    let ref_out = reference_for_bf16(
        &q_bf, &k_bf, &v_bf, Some(&mask_rank4_bf),
        batch, h, kv_h, ql, kl, d, scale, false,
    );

    assert_all_finite(&gpu_out, "d64_sm_rank2_mask");
    assert_close_gpu(
        &gpu_out, &ref_out, BF16_GPU_ATOL, BF16_GPU_RTOL,
        "d64_sm_rank2_mask",
    );
}

/// Wrong head_dim must reject before any GPU resource is touched.
#[test]
fn test_d64_dispatcher_rejects_wrong_head_dim() {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    flash_attn_prefill::register(&mut registry);

    let q_elems = 1 * 2 * 32 * 128;
    let kv_elems = 1 * 2 * 32 * 128;
    let q_buf = alloc_bf16(&device, q_elems, "Q");
    let k_buf = alloc_bf16(&device, kv_elems, "K");
    let v_buf = alloc_bf16(&device, kv_elems, "V");
    let mut out_buf = alloc_bf16(&device, q_elems, "out");

    let params = FlashAttnPrefillParams {
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 128, // wrong: D=64 dispatcher requires 64
        seq_len_q: 32,
        seq_len_k: 32,
        batch: 1,
        scale: 1.0,
        do_causal: false,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    let result = dispatch_flash_attn_prefill_bf16_d64(
        &mut encoder, &device, &mut registry,
        &q_buf, &k_buf, &v_buf, None, &mut out_buf, &params,
        FlashAttnPrefillLayout::HeadMajor,
    );

    assert!(result.is_err(), "Expected error for head_dim=128 on D=64 dispatcher");
    if let Err(MlxError::InvalidArgument(msg)) = result {
        assert!(
            msg.contains("64") || msg.contains("head_dim"),
            "Error should reference head_dim or 64: {msg}"
        );
    } else {
        panic!("Expected InvalidArgument, got: {result:?}");
    }
}

} // mod flash_attn_prefill_tests

// ─── Non-macOS stub ───────────────────────────────────────────────────────────
#[cfg(not(target_os = "macos"))]
#[test]
fn test_flash_attn_prefill_requires_macos() {
    eprintln!("Skipping flash_attn_prefill tests — Metal requires macOS/Apple Silicon");
}
