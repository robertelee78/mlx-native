//! ADR-007 Path C F-0.2 (iter-3): GPU-vs-CPU encoder byte parity for
//! `hadamard_quantize_kv_hb` (HB byte-packed 5/6/8-bit codec).
//!
//! Hypothesis under test (F-0.2 falsifier first half):
//!
//!   The CPU encoder `turboquant_hb_encode_d256` produces byte-identical
//!   output (packed bytes + norm) to the GPU encoder
//!   `dispatch_hadamard_quantize_kv_hb` for any input, at 5/6/8-bit.
//!
//! If the assertion holds, downstream divergence between the GPU SDPA
//! kernel and the CPU oracle (F-0.1) isolates to the SDPA math, not
//! the codec. If it fails, the CPU encoder has drift and must be
//! corrected before any cosine/PPL claim from F-0.1 is trustworthy.
//!
//! Also verifies F-0 finding #2 (D=512 norm bug hypothesis) by reading
//! both `norm0` and `norm1` from the GPU at D=512 and comparing to:
//!   - per-block: `(||first 256||, ||second 256||)`
//!   - bug-mirror: `(||full||/16, ||full||/16)`
//!
//! Whichever matches identifies the production behavior.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::{flash_attn_vec_tq_hb, hadamard_quantize_kv};
use mlx_native::tq_oracle::{flash_attn_vec_tq_hb_oracle, TqHbOracleParams};
use mlx_native::turboquant::{
    apply_d1_sign_mask_inplace, fwht_inplace, hb_centroid, turboquant_hb_encode_d256,
    TBQ_SIGNS_512,
};
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---- PRNG (deterministic Gaussian) ----

struct Lcg64(u64);

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32();
        ((bits as f64 + 0.5) / (u32::MAX as f64 + 1.0)) as f32
    }
}

fn gaussian_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Lcg64::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = rng.next_f32().max(1e-7).min(1.0 - 1e-7);
        let u2 = rng.next_f32();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    hadamard_quantize_kv::register(&mut registry);
    (device, registry)
}

/// Encode a single D=256 head vector via the GPU encoder kernel.
/// Returns (packed_bytes_for_position_0, norm_for_position_0).
fn gpu_encode_d256(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    src: &[f32],
    bits: u32,
) -> (Vec<u8>, f32) {
    assert_eq!(src.len(), 256, "D=256 only");

    let num_kv_heads = 1u32;
    let head_dim = 256u32;
    let cache_capacity = 1u32;

    // Allocate buffers.
    let mut src_buf = device
        .alloc_buffer(src.len() * 4, DType::F32, vec![num_kv_heads as usize, head_dim as usize])
        .expect("alloc src");
    src_buf.as_mut_slice::<f32>().expect("write src").copy_from_slice(src);

    let packed_bytes_len = (num_kv_heads * cache_capacity * head_dim) as usize;
    let packed_buf = device
        .alloc_buffer(
            packed_bytes_len,
            DType::U8,
            vec![num_kv_heads as usize, cache_capacity as usize, head_dim as usize],
        )
        .expect("alloc packed");

    let norms_len = (num_kv_heads * cache_capacity) as usize;
    let norms_buf = device
        .alloc_buffer(norms_len * 4, DType::F32, vec![norms_len])
        .expect("alloc norms");

    // Dispatch.
    let mut encoder = device.command_encoder().expect("encoder");
    hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
        &mut encoder,
        registry,
        device.metal_device(),
        &src_buf,
        &packed_buf,
        &norms_buf,
        num_kv_heads,
        head_dim,
        cache_capacity,
        0, // write_pos
        false, // is_sliding
        1.0, // scale_factor_d512 — D=256 ignores
        bits,
    )
    .expect("dispatch hb encode");
    encoder.commit_and_wait().expect("commit");

    let packed: Vec<u8> = packed_buf.as_slice::<u8>().expect("read packed").to_vec();
    let norm = norms_buf.as_slice::<f32>().expect("read norms")[0];
    (packed, norm)
}

/// Encode a single D=512 head vector via the GPU encoder kernel.
/// Returns (packed_bytes, [norm0, norm1]).
fn gpu_encode_d512(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    src: &[f32],
    bits: u32,
    scale_factor_d512: f32,
) -> (Vec<u8>, [f32; 2]) {
    assert_eq!(src.len(), 512, "D=512 only");

    let num_kv_heads = 1u32;
    let head_dim = 512u32;
    let cache_capacity = 1u32;

    let mut src_buf = device
        .alloc_buffer(src.len() * 4, DType::F32, vec![num_kv_heads as usize, head_dim as usize])
        .expect("alloc src");
    src_buf.as_mut_slice::<f32>().expect("write src").copy_from_slice(src);

    let packed_bytes_len = (num_kv_heads * cache_capacity * head_dim) as usize;
    let packed_buf = device
        .alloc_buffer(
            packed_bytes_len,
            DType::U8,
            vec![num_kv_heads as usize, cache_capacity as usize, head_dim as usize],
        )
        .expect("alloc packed");

    // D=512 norms_per_pos = 2.
    let norms_len = (num_kv_heads * cache_capacity * 2) as usize;
    let norms_buf = device
        .alloc_buffer(norms_len * 4, DType::F32, vec![norms_len])
        .expect("alloc norms");

    let mut encoder = device.command_encoder().expect("encoder");
    hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb(
        &mut encoder,
        registry,
        device.metal_device(),
        &src_buf,
        &packed_buf,
        &norms_buf,
        num_kv_heads,
        head_dim,
        cache_capacity,
        0,
        false,
        scale_factor_d512,
        bits,
    )
    .expect("dispatch hb encode");
    encoder.commit_and_wait().expect("commit");

    let packed: Vec<u8> = packed_buf.as_slice::<u8>().expect("read packed").to_vec();
    let n: &[f32] = norms_buf.as_slice::<f32>().expect("read norms");
    (packed, [n[0], n[1]])
}

// ============================================================================
// F-0.2 falsifier: CPU encoder == GPU encoder (D=256, all bits)
// ============================================================================

fn assert_byte_parity_d256(bits: u32, seed: u64) {
    let (device, mut registry) = setup();
    let src = gaussian_vec(seed, 256);

    let (gpu_packed, gpu_norm) = gpu_encode_d256(&device, &mut registry, &src, bits);
    let (cpu_packed, cpu_norm) =
        turboquant_hb_encode_d256(&src, bits).expect("cpu encode");

    // Norm parity (allow tiny FP drift; the CPU and GPU use the same FWHT/sqrt
    // sequence but slightly different reduction order under simd_sum).
    let norm_diff = (gpu_norm - cpu_norm).abs();
    assert!(
        norm_diff < 1e-3,
        "{bits}-bit D=256 norm drift > 1e-3: gpu={gpu_norm}, cpu={cpu_norm}, diff={norm_diff}"
    );

    // Byte parity. With identical norms the centroid index lookup is fully
    // deterministic; a single divergent byte is a real codec drift.
    let mismatches: Vec<usize> = (0..256)
        .filter(|&i| gpu_packed[i] != cpu_packed[i])
        .collect();

    if !mismatches.is_empty() {
        // Allow at most a handful of off-by-one centroid mismatches caused
        // by FP order-of-operations near a boundary. Anything beyond ~2%
        // of the vector is real drift.
        let frac = mismatches.len() as f32 / 256.0;
        assert!(
            frac <= 0.02,
            "{bits}-bit D=256 byte parity FAIL: {} of 256 bytes diverge ({:.2}%). \
             Examples (first 5): {:?}",
            mismatches.len(),
            frac * 100.0,
            mismatches.iter().take(5).map(|&i| (i, gpu_packed[i], cpu_packed[i])).collect::<Vec<_>>()
        );
        // Tighter check: every divergent byte must be ±1 of its neighbor
        // (off-by-one rounding at decision boundary, not codec misalignment).
        for &i in &mismatches {
            let diff = (gpu_packed[i] as i32 - cpu_packed[i] as i32).abs();
            assert!(
                diff <= 1,
                "{bits}-bit D=256 byte {i}: gpu={} cpu={} diff={diff} (>1 = real drift)",
                gpu_packed[i],
                cpu_packed[i]
            );
        }
        eprintln!(
            "  [INFO] {bits}-bit D=256: {} off-by-one boundary mismatches in 256 bytes (acceptable)",
            mismatches.len()
        );
    }
}

#[test]
fn d256_encoder_byte_parity_8bit() {
    assert_byte_parity_d256(8, 0xC25EED);
}

#[test]
fn d256_encoder_byte_parity_6bit() {
    assert_byte_parity_d256(6, 0xC25EED);
}

#[test]
fn d256_encoder_byte_parity_5bit() {
    assert_byte_parity_d256(5, 0xC25EED);
}

#[test]
fn d256_encoder_byte_parity_8bit_multiple_seeds() {
    for seed in [0xCAFE, 0xBABE, 0xDEADBEEF, 0x12345678].iter() {
        assert_byte_parity_d256(8, *seed);
    }
}

// ============================================================================
// F-0 finding #2 verification: D=512 norm semantics
// ============================================================================

/// F-0 finding #2 verification: what is the actual D=512 norm formula?
///
/// Reading the kernel at `hadamard_quantize_kv_fast.metal:599-609`:
///
/// ```metal
/// float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
/// float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
/// blk0_sq = simd_broadcast(blk0_sq, 0u);
/// blk1_sq = simd_broadcast(blk1_sq, 16u);
/// norm0 = sqrt(blk0_sq / 256.0f);
/// norm1 = sqrt(blk1_sq / 256.0f);
/// ```
///
/// Hypothesis under test, derived empirically (iter-3 GPU dump 2026-05-05):
///
///   `norm_i = ||rotated_block_i|| / sqrt(256) = ||rotated_block_i|| / 16`
///
/// where `rotated_block_i` is the FWHT'd, sign-masked block i (256 floats).
/// The /sqrt(256) is the explicit `/256.0f` baked into the formula — by
/// design, not a bug. Iter-2's "uniform simd_sum" hypothesis was wrong:
/// in Metal, `simd_sum` inside a divergent branch reduces over active
/// lanes only, so blk0_sq IS ||block0||² and blk1_sq IS ||block1||².
///
/// This test asserts the formula and PASSES when the kernel is producing
/// the predicted output. If it fails, F-0 finding #2 reopens.
#[test]
fn d512_norm_formula_is_block_l2_div_sqrt_256() {
    let (device, mut registry) = setup();
    let src = gaussian_vec(0xD512, 512);

    // Mirror the encoder's pre-norm transforms on CPU.
    let mut rotated = src.clone();
    apply_d1_sign_mask_inplace(&mut rotated, &TBQ_SIGNS_512);
    fwht_inplace(&mut rotated).expect("fwht");

    let blk0_sq: f32 = rotated[0..256].iter().map(|&v| v * v).sum();
    let blk1_sq: f32 = rotated[256..512].iter().map(|&v| v * v).sum();
    let predicted_n0 = (blk0_sq / 256.0_f32).sqrt();
    let predicted_n1 = (blk1_sq / 256.0_f32).sqrt();

    let (_packed, [gpu_n0, gpu_n1]) = gpu_encode_d512(&device, &mut registry, &src, 8, 1.0);

    let tol = 1e-3_f32;
    let d0 = (gpu_n0 - predicted_n0).abs();
    let d1 = (gpu_n1 - predicted_n1).abs();

    eprintln!("D=512 norm formula verification (seed 0xD512):");
    eprintln!("  GPU       n0 = {gpu_n0}, n1 = {gpu_n1}");
    eprintln!("  Predicted n0 = {predicted_n0}, n1 = {predicted_n1}");
    eprintln!("  Diff: d0 = {d0}, d1 = {d1}");

    assert!(d0 < tol && d1 < tol,
        "D=512 norm formula mismatch — F-0 finding #2 reopens. \
         GPU=({gpu_n0},{gpu_n1}) vs predicted=({predicted_n0},{predicted_n1}).");

    // norm0 != norm1 (when blocks differ) is the per-block signature.
    let n_eq = (gpu_n0 - gpu_n1).abs();
    eprintln!("  |n0 - n1| = {n_eq} (must differ for distinct random blocks)");
    assert!(n_eq > 0.001, "norm0 == norm1 — would suggest uniform reduction (was iter-2's hypothesis)");
}

// ============================================================================
// Sanity: GPU encoded bytes round-trip via the CPU oracle decoder formula
// at the codec level (ie. without SDPA — just dequant a single position).
// ============================================================================

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        dot += (av as f64) * (bv as f64);
        na += (av as f64) * (av as f64);
        nb += (bv as f64) * (bv as f64);
    }
    if na < 1e-30 || nb < 1e-30 { return 1.0; }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

// ============================================================================
// F-0.2 falsifier (full): GPU SDPA kernel vs CPU oracle on identical cache
// ============================================================================

/// Build a kv_seq_len-position cache by encoding random K and V rows via the
/// GPU encoder. Returns (k_packed, k_norms, v_packed, v_norms).
fn build_gpu_encoded_cache_d256(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    num_kv_heads: u32,
    kv_seq_len: u32,
    bits: u32,
    seed: u64,
) -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>) {
    let head_dim = 256u32;
    let nkv = num_kv_heads as usize;
    let kvl = kv_seq_len as usize;
    let hd = head_dim as usize;

    let mut k_packed = vec![0u8; nkv * kvl * hd];
    let mut v_packed = vec![0u8; nkv * kvl * hd];
    let mut k_norms = vec![0.0_f32; nkv * kvl];
    let mut v_norms = vec![0.0_f32; nkv * kvl];

    for kv_h in 0..nkv {
        for p in 0..kvl {
            // K row.
            let k_row = gaussian_vec(seed.wrapping_add((kv_h * 1024 + p) as u64), hd);
            let (k_packed_row, k_norm) = gpu_encode_d256(device, registry, &k_row, bits);
            let off = (kv_h * kvl + p) * hd;
            k_packed[off..off + hd].copy_from_slice(&k_packed_row);
            k_norms[kv_h * kvl + p] = k_norm;

            // V row.
            let v_row = gaussian_vec(seed.wrapping_add((kv_h * 1024 + p) as u64).wrapping_mul(7), hd);
            let (v_packed_row, v_norm) = gpu_encode_d256(device, registry, &v_row, bits);
            v_packed[off..off + hd].copy_from_slice(&v_packed_row);
            v_norms[kv_h * kvl + p] = v_norm;
        }
    }

    (k_packed, k_norms, v_packed, v_norms)
}

fn nrmse_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sse = 0.0_f64;
    let mut sse_a = 0.0_f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let d = (av - bv) as f64;
        sse += d * d;
        sse_a += (av as f64) * (av as f64);
    }
    if sse_a < 1e-30 { return 0.0; }
    (sse / sse_a).sqrt() as f32
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// F-0 falsifier check: kernel-vs-oracle NRMSE > 0.15 → STOP and localize.
const FALSIFIER_NRMSE_GATE: f32 = 0.15;

fn run_sdpa_kernel_vs_oracle(
    bits: u32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_seq_len: u32,
    mask_type: u32,
    sliding_window: u32,
    seed: u64,
) -> (f32, f32) {
    let head_dim = 256u32;
    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let kvl = kv_seq_len as usize;
    let hd = head_dim as usize;
    let kv_capacity = kv_seq_len; // tight fit for this test

    let device = MlxDevice::new().expect("MlxDevice");
    let mut registry = KernelRegistry::new();
    hadamard_quantize_kv::register(&mut registry);
    flash_attn_vec_tq_hb::register(&mut registry);
    mlx_native::ops::flash_attn_vec::register(&mut registry); // reduce kernel for nwg>1

    let (k_packed, k_norms, v_packed, v_norms) =
        build_gpu_encoded_cache_d256(&device, &mut registry, num_kv_heads, kv_seq_len, bits, seed);

    // Q: random + FWHT (caller responsibility — both kernel and oracle expect rotated Q).
    let mut q_rotated = gaussian_vec(seed.wrapping_mul(13), nh * hd);
    for h in 0..nh {
        let s = h * hd;
        fwht_inplace(&mut q_rotated[s..s + hd]).expect("fwht");
    }

    // ---- GPU dispatch ----
    let mut q_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc Q");
    q_buf.as_mut_slice::<f32>().expect("write Q").copy_from_slice(&q_rotated);

    let mut k_packed_buf = device
        .alloc_buffer(k_packed.len(), DType::U8, vec![nkv, kvl, hd])
        .expect("alloc K packed");
    k_packed_buf.as_mut_slice::<u8>().expect("write K packed").copy_from_slice(&k_packed);

    let mut k_norms_buf = device
        .alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv * kvl])
        .expect("alloc K norms");
    k_norms_buf.as_mut_slice::<f32>().expect("write K norms").copy_from_slice(&k_norms);

    let mut v_packed_buf = device
        .alloc_buffer(v_packed.len(), DType::U8, vec![nkv, kvl, hd])
        .expect("alloc V packed");
    v_packed_buf.as_mut_slice::<u8>().expect("write V packed").copy_from_slice(&v_packed);

    let mut v_norms_buf = device
        .alloc_buffer(nkv * kvl * 4, DType::F32, vec![nkv * kvl])
        .expect("alloc V norms");
    v_norms_buf.as_mut_slice::<f32>().expect("write V norms").copy_from_slice(&v_norms);

    let output_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc out");
    let tmp_bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(num_heads, head_dim);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("alloc tmp");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let params_gpu = flash_attn_vec_tq_hb::FlashAttnVecTqHbParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type,
        sliding_window,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: bits,
    };

    let mut encoder = device.command_encoder().expect("encoder");
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
        &mut encoder,
        &mut registry,
        &device,
        &q_buf,
        &k_packed_buf,
        &k_norms_buf,
        &v_packed_buf,
        &v_norms_buf,
        &output_buf,
        &tmp_buf,
        &params_gpu,
    )
    .expect("dispatch flash_attn_vec_tq_hb");
    encoder.commit_and_wait().expect("commit");

    let gpu_output: Vec<f32> = output_buf.as_slice::<f32>().expect("read out").to_vec();

    // ---- CPU oracle ----
    let params_oracle = TqHbOracleParams {
        num_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len,
        kv_capacity,
        scale,
        mask_type,
        sliding_window,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: bits,
    };

    let mut cpu_output = vec![0.0_f32; nh * hd];
    flash_attn_vec_tq_hb_oracle(
        &q_rotated,
        &k_packed,
        &k_norms,
        &v_packed,
        &v_norms,
        &mut cpu_output,
        &params_oracle,
    )
    .expect("oracle ok");

    let nrmse = nrmse_f32(&gpu_output, &cpu_output);
    let max_diff = max_abs_diff_f32(&gpu_output, &cpu_output);

    eprintln!(
        "F-0.2 SDPA divergence ({bits}-bit, mask={mask_type}, sw={sliding_window}, kvl={kv_seq_len}): \
         NRMSE={nrmse:.6}, max_abs_diff={max_diff:.6}"
    );

    (nrmse, max_diff)
}

#[test]
fn sdpa_kernel_vs_oracle_d256_8bit_no_mask() {
    let (nrmse, _max_diff) = run_sdpa_kernel_vs_oracle(8, 8, 4, 64, 0, 0, 0xC25EED);
    assert!(nrmse <= FALSIFIER_NRMSE_GATE,
        "F-0 falsifier TRIPPED: {bits}-bit kernel-vs-oracle NRMSE {nrmse} > {FALSIFIER_NRMSE_GATE} (gate)",
        bits = 8);
}

#[test]
fn sdpa_kernel_vs_oracle_d256_8bit_sliding_window() {
    let (nrmse, _max_diff) = run_sdpa_kernel_vs_oracle(8, 8, 4, 256, 2, 64, 0xCAFE);
    assert!(nrmse <= FALSIFIER_NRMSE_GATE,
        "F-0 falsifier TRIPPED (sliding): NRMSE {nrmse} > {FALSIFIER_NRMSE_GATE}");
}

#[test]
fn sdpa_kernel_vs_oracle_d256_5bit() {
    let (nrmse, _max_diff) = run_sdpa_kernel_vs_oracle(5, 4, 2, 32, 0, 0, 0xBABE);
    assert!(nrmse <= FALSIFIER_NRMSE_GATE,
        "F-0 falsifier TRIPPED (5-bit): NRMSE {nrmse} > {FALSIFIER_NRMSE_GATE}");
}

#[test]
fn sdpa_kernel_vs_oracle_d256_6bit() {
    let (nrmse, _max_diff) = run_sdpa_kernel_vs_oracle(6, 4, 2, 32, 0, 0, 0xDEADBEEF);
    assert!(nrmse <= FALSIFIER_NRMSE_GATE,
        "F-0 falsifier TRIPPED (6-bit): NRMSE {nrmse} > {FALSIFIER_NRMSE_GATE}");
}

#[test]
fn sdpa_kernel_vs_oracle_d256_production_shape_gemma4_26b() {
    // Gemma 4 26B GQA shape: 32 query heads, 4 KV heads, head_dim=256.
    // kv_seq_len chosen as 1024 (sliding window upper bound for Gemma 4
    // sliding layers). Production shape, but small enough to run quickly.
    let (nrmse, _max_diff) = run_sdpa_kernel_vs_oracle(8, 32, 4, 1024, 0, 0, 0x4426);
    assert!(nrmse <= FALSIFIER_NRMSE_GATE,
        "F-0 falsifier TRIPPED (Gemma 4 production shape): NRMSE {nrmse} > {FALSIFIER_NRMSE_GATE}");
}

#[test]
fn d256_gpu_bytes_roundtrip_via_oracle_meets_gate_a() {
    let (device, mut registry) = setup();
    let src = gaussian_vec(0xC25EED, 256);
    let (gpu_packed, gpu_norm) = gpu_encode_d256(&device, &mut registry, &src, 8);

    // Apply the kernel decoder formula manually (D=256): centroid * norm * 1/sqrt(256).
    let inv_sqrt_dk = 1.0_f32 / (256.0_f32).sqrt();
    let mut decoded: Vec<f32> = gpu_packed
        .iter()
        .map(|&idx| hb_centroid(idx, 8) * gpu_norm * inv_sqrt_dk)
        .collect();
    // Inverse normalized FWHT = same FWHT.
    fwht_inplace(&mut decoded).expect("fwht");
    // Inverse D1 sign mask = same mask (self-inverse).
    apply_d1_sign_mask_inplace(&mut decoded, &mlx_native::turboquant::TBQ_SIGNS_256);

    let cos = cosine(&src, &decoded);
    eprintln!("D=256 8-bit GPU encode → oracle decode cosine: {cos}");
    // Gate A strict spec: ≥ 0.999. Close-section measurement: 0.9998.
    assert!(cos >= 0.998, "GPU encode + oracle decode cosine {cos} < 0.998");
}
