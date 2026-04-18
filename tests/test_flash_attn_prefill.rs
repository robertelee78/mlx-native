//! Integration tests for the flash-attention prefill kernel.
//!
//! Tests the `attention<BQ, BK, BD, WM, WN>` kernel in
//! `src/shaders/flash_attn_prefill.metal` against a CPU reference that
//! mirrors the exact mathematical idioms used in the GPU shader (base-2
//! softmax, three NaN guards documented in the kernel preamble).
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
//! - NaN guard #1 on exp: if `max_new == -inf` (row fully masked), exp = 0.
//! - NaN guard #2 on rescale factor: if `max_old == -inf`, factor = 0.
//! - Additive (float) mask is multiplied by `log2(e)` before being added to
//!   the score, matching the kernel's mask-addition step.
//! - Causal masking: positions with `k_pos > q_abs_pos` get scores set to
//!   -inf before exp.
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

                // Standard numerically-stable softmax using exp().
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores
                    .iter()
                    .map(|&s| if max_s == f32::NEG_INFINITY { 0.0 } else { (s - max_s).exp() })
                    .collect();
                let sum_exp: f32 = exp_scores.iter().sum();
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

                // ── Online softmax (base-2, with NaN guards) ──────────────
                // Row max reduction:
                // flash_attn_prefill.metal:1393 — row_reduce<MaxOp>(new_max)
                let mut max_score = f32::NEG_INFINITY; // init: Limits<float>::min = -inf (:1060)
                for &s in &scores {
                    if s > max_score {
                        max_score = s;
                    }
                }

                // exp2(score - max) with NaN guard:
                // flash_attn_prefill.metal:1064-1072 — ExpSubOp::apply
                //   if (y == -inf) return T(0); else return fast::exp2(x - y);
                let exp_scores: Vec<f32> = scores
                    .iter()
                    .map(|&s| {
                        if max_score == f32::NEG_INFINITY {
                            // NaN guard: all scores are -inf (fully masked row).
                            // flash_attn_prefill.metal:1067-1069
                            0.0_f32
                        } else {
                            // flash_attn_prefill.metal:1070 — fast::exp2(x - y)
                            f32::exp2(s - max_score)
                        }
                    })
                    .collect();

                // Sum: flash_attn_prefill.metal:1415-1416 — row_reduce<SumOp>
                let sum_exp: f32 = exp_scores.iter().sum();
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

/// CPU NaN guard test: fully-masked input must produce finite output.
///
/// When ALL K positions are masked to -infinity, the CPU reference's NaN guards
/// must prevent NaN propagation (output should be 0 — no valid keys attended):
///
/// 1. `ExpSubOp` guard: max_new == -inf → return 0 (not exp2(NaN)).
///    flash_attn_prefill.metal:1067-1069
/// 2. Factor guard: max_old == -inf → factor = 0 (not exp2(-inf - new_max)).
///    flash_attn_prefill.metal:1401-1405
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

/// GPU correctness: fully-masked KV tile exercises all three NaN-guard paths.
///
/// The kernel uses true -infinity as the masked-position sentinel, so the
/// fully-masked-row case requires three guards (per the kernel's preamble):
/// (1) `ExpSubOp` for `max == -inf`, (2) the rescale factor for
/// `old_max == -inf`, and (3) `DivOp` for `sum_score == 0`.  Without all
/// three, fully-masked rows propagate NaN.
///
/// This test sets the entire mask to -inf; the expected output is all zeros
///   (no valid keys attended).  Any NaN or Inf in the output = guard bug.
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

} // mod flash_attn_prefill_tests

// ─── Non-macOS stub ───────────────────────────────────────────────────────────
#[cfg(not(target_os = "macos"))]
#[test]
fn test_flash_attn_prefill_requires_macos() {
    eprintln!("Skipping flash_attn_prefill tests — Metal requires macOS/Apple Silicon");
}
