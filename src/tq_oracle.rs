//! ADR-007 Path C F-0.1: CPU F32 oracle for `flash_attn_vec_tq_hb` decode.
//!
//! Mirrors the math in `src/shaders/flash_attn_vec_tq_hb.metal::flash_attn_vec_tq_hb_impl`
//! exactly, in pure F32, with deterministic serial reduction. Used by the
//! Path C F-0.2 layer-by-layer divergence audit and by codec roundtrip
//! correctness gates.
//!
//! Inputs match the kernel call signature 1:1 (see
//! `crate::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams` and
//! `flash_attn_vec_tq_hb`). `Q` is consumed as F32 (the kernel converts to
//! `half` in shared memory; the oracle does NOT mirror that precision loss
//! since it serves as the ground truth — the F16-Q precision gap is one of
//! the divergence sources F-0.2 measures).
//!
//! No GPU dispatch, no Metal, no panics. Pure-Rust, single-threaded, branchless
//! at the inner loop where possible. Bit-for-bit deterministic across runs at
//! fixed inputs.
//!
//! ## Buffer layouts (must match the kernel exactly)
//!
//! - `q`:        `[num_heads, head_dim]` F32 (one query per head, decode).
//! - `k_packed`: `[num_kv_heads, kv_capacity, head_dim]` U8 (1 byte per element).
//! - `k_norms`:
//!     - D=256: `[num_kv_heads, kv_capacity]` F32 (1 norm per position).
//!     - D=512: `[num_kv_heads, kv_capacity, 2]` F32 (2 per-block norms per
//!       position, block 0 = coords 0..255, block 1 = coords 256..511).
//! - `v_packed`, `v_norms`: same as K.
//! - `output`:   `[num_heads, head_dim]` F32 written by the oracle.
//!
//! ## Codec dequant formula (mirrors kernel lines 170-202)
//!
//! For position `kv_pos`, head `kv_head`, coordinate `d`:
//! - D=256: `value = codebook[byte_idx] * (norm * inv_sqrt(DK))`
//! - D=512: `value = codebook[byte_idx] * (norm[block_idx] / scale_factor_d512)`
//!     where `block_idx = d / 256` (0 for coords 0..255, 1 for 256..511).
//!
//! ## Mask semantics (mirrors kernel lines 295-318)
//!
//! - `mask_type == 2 && sliding_window > 0 && kv_seq_len > sliding_window`:
//!   set `window_start_logical = kv_seq_len - sliding_window`. Otherwise 0.
//! - For each k_pos in `0..kv_seq_len`:
//!     - `logical_idx = (k_pos - ring_start + kv_capacity) % kv_capacity`
//!     - Position is valid iff `logical_idx >= window_start_logical &&
//!       logical_idx < kv_seq_len`. Otherwise masked (-65504.0).

use crate::error::{MlxError, Result};
use crate::turboquant::hb_centroid;

/// Parameters for the HB TQ flash attention decode oracle.
///
/// Field-for-field mirror of `crate::ops::flash_attn_vec_tq_hb::FlashAttnVecTqHbParams`
/// — kept independent so the oracle has zero dependency on Metal types.
#[derive(Debug, Clone, Copy)]
pub struct TqHbOracleParams {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub kv_seq_len: u32,
    pub kv_capacity: u32,
    pub scale: f32,
    pub mask_type: u32,
    pub sliding_window: u32,
    /// Note: present in kernel params but never read in the kernel body.
    /// Oracle mirrors the kernel by NOT applying softcap. Tracked as F-0
    /// finding: contractual drift vs `flash_attn_vec.metal` where softcap is
    /// also documented but unimplemented.
    pub softcap: f32,
    pub ring_start: u32,
    /// Only used when `head_dim == 512`. For D=256 set to any value.
    pub scale_factor_d512: f32,
    /// Codebook bit-width: 5, 6, or 8.
    pub codebook_bits: u32,
}

fn validate(params: &TqHbOracleParams, q_len: usize, k_packed_len: usize, k_norms_len: usize, v_packed_len: usize, v_norms_len: usize, output_len: usize) -> Result<()> {
    if params.head_dim != 256 && params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: head_dim must be 256 or 512, got {}",
            params.head_dim
        )));
    }
    if params.num_heads == 0 || params.num_kv_heads == 0 {
        return Err(MlxError::InvalidArgument(
            "tq_oracle: num_heads and num_kv_heads must be > 0".into(),
        ));
    }
    if params.num_heads % params.num_kv_heads != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: num_heads ({}) % num_kv_heads ({}) != 0",
            params.num_heads, params.num_kv_heads
        )));
    }
    if params.kv_seq_len == 0 {
        return Err(MlxError::InvalidArgument(
            "tq_oracle: kv_seq_len must be > 0".into(),
        ));
    }
    if params.kv_capacity < params.kv_seq_len {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: kv_capacity ({}) < kv_seq_len ({})",
            params.kv_capacity, params.kv_seq_len
        )));
    }
    if !matches!(params.codebook_bits, 5 | 6 | 8) {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: codebook_bits must be 5, 6, or 8, got {}",
            params.codebook_bits
        )));
    }
    let dk = params.head_dim as usize;
    let nh = params.num_heads as usize;
    let nkv = params.num_kv_heads as usize;
    let cap = params.kv_capacity as usize;
    let norms_per_pos = if dk == 512 { 2 } else { 1 };

    let need_q = nh * dk;
    let need_packed = nkv * cap * dk;
    let need_norms = nkv * cap * norms_per_pos;
    let need_output = nh * dk;

    if q_len < need_q {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: q has {q_len} < {need_q} required"
        )));
    }
    if k_packed_len < need_packed {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: k_packed has {k_packed_len} < {need_packed} required"
        )));
    }
    if v_packed_len < need_packed {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: v_packed has {v_packed_len} < {need_packed} required"
        )));
    }
    if k_norms_len < need_norms {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: k_norms has {k_norms_len} < {need_norms} required"
        )));
    }
    if v_norms_len < need_norms {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: v_norms has {v_norms_len} < {need_norms} required"
        )));
    }
    if output_len < need_output {
        return Err(MlxError::InvalidArgument(format!(
            "tq_oracle: output has {output_len} < {need_output} required"
        )));
    }
    Ok(())
}

/// CPU F32 oracle for `flash_attn_vec_tq_hb` decode.
///
/// Computes `output = softmax(Q @ K^T * scale + mask) @ V` where K and V are
/// dequantized from the byte-packed HB codec on-the-fly per the kernel formula.
///
/// Caller is responsible for applying FWHT to `q` BEFORE calling this oracle
/// (mirrors kernel contract, see `flash_attn_vec_tq_hb.rs:128-130`). Inverse
/// FWHT of `output` is also the caller's responsibility.
///
/// Determinism: bit-identical across runs at fixed inputs (serial reduction,
/// no parallelism, no NaN sources at validated inputs).
pub fn flash_attn_vec_tq_hb_oracle(
    q: &[f32],
    k_packed: &[u8],
    k_norms: &[f32],
    v_packed: &[u8],
    v_norms: &[f32],
    output: &mut [f32],
    params: &TqHbOracleParams,
) -> Result<()> {
    validate(
        params,
        q.len(),
        k_packed.len(),
        k_norms.len(),
        v_packed.len(),
        v_norms.len(),
        output.len(),
    )?;

    let dk = params.head_dim as usize;
    let nh = params.num_heads as usize;
    let nkv = params.num_kv_heads as usize;
    let kv_seq_len = params.kv_seq_len as usize;
    let kv_capacity = params.kv_capacity as usize;
    let ring_start = params.ring_start as usize;
    let cbits = params.codebook_bits;
    let heads_per_kv = nh / nkv;

    // Mirror the kernel's window_start_logical computation (lines 295-298).
    let window_start_logical: usize = if params.mask_type == 2
        && params.sliding_window > 0
        && (kv_seq_len as u32) > params.sliding_window
    {
        kv_seq_len - params.sliding_window as usize
    } else {
        0
    };

    let is_d512 = dk == 512;
    let inv_sqrt_dk: f32 = 1.0_f32 / (dk as f32).sqrt();
    // For D=256, V dequant scales by inv_sqrt_dv where DV=DK in our code path.
    // Mirror line 418 of the kernel: `const float inv_sqrt_dv = rsqrt(float(DV));`
    let inv_sqrt_dv: f32 = inv_sqrt_dk; // DV == DK in flash_attn_vec_tq_hb
    let sf_d512: f32 = params.scale_factor_d512;

    // Pre-compute mask: `mask[kv_pos] = 0.0 if valid, -65504.0 if invalid`.
    // Same predicate as the kernel (lines 308-318).
    let neg_inf_proxy: f32 = -65504.0_f32;
    let mut mask_vec: Vec<f32> = vec![0.0_f32; kv_seq_len];
    for kv_pos in 0..kv_seq_len {
        // logical_idx = (kv_pos - ring_start + kv_capacity) % kv_capacity
        // (signed-safe mod via wrapping arithmetic on u64)
        let logical_idx = ((kv_pos as i64 - ring_start as i64).rem_euclid(kv_capacity as i64))
            as usize;
        let valid = logical_idx >= window_start_logical && logical_idx < kv_seq_len;
        mask_vec[kv_pos] = if valid { 0.0_f32 } else { neg_inf_proxy };
    }

    // Per-head SDPA loop. Order: q_head h → kv_head h/heads_per_kv → kv_pos →
    // accumulate scores[kv_pos] → online softmax → output.
    for h in 0..nh {
        let kv_head = h / heads_per_kv;
        let q_offset = h * dk;
        let q_row: &[f32] = &q[q_offset..q_offset + dk];

        // Pass 1: Q @ K^T * scale + mask = scores[kv_pos]
        let mut scores: Vec<f32> = vec![neg_inf_proxy; kv_seq_len];
        for kv_pos in 0..kv_seq_len {
            if mask_vec[kv_pos] <= neg_inf_proxy {
                // Already -65504; leave it.
                continue;
            }
            let k_packed_offset = (kv_head * kv_capacity + kv_pos) * dk;
            let k_packed_row: &[u8] = &k_packed[k_packed_offset..k_packed_offset + dk];

            let mut dot: f32 = 0.0_f32;
            if is_d512 {
                // D=512: per-block norms.
                let knorm_offset = (kv_head * kv_capacity + kv_pos) * 2;
                let n0 = k_norms[knorm_offset];
                let n1 = k_norms[knorm_offset + 1];
                let sn0 = n0 / sf_d512;
                let sn1 = n1 / sf_d512;
                // Block 0: coords 0..256
                for d in 0..256 {
                    let centroid = hb_centroid(k_packed_row[d], cbits);
                    dot += q_row[d] * centroid * sn0;
                }
                // Block 1: coords 256..512
                for d in 256..dk {
                    let centroid = hb_centroid(k_packed_row[d], cbits);
                    dot += q_row[d] * centroid * sn1;
                }
            } else {
                // D=256: single norm per position.
                let n = k_norms[kv_head * kv_capacity + kv_pos];
                let sn = n * inv_sqrt_dk;
                for d in 0..dk {
                    let centroid = hb_centroid(k_packed_row[d], cbits);
                    dot += q_row[d] * centroid * sn;
                }
            }
            scores[kv_pos] = dot * params.scale + mask_vec[kv_pos];
        }

        // Pass 2: stable softmax (max-subtraction).
        // Mirror the kernel's online softmax outcome but use the equivalent
        // batch form for clarity. Both are deterministic in F32 for our serial
        // reduction.
        let mut m: f32 = f32::NEG_INFINITY;
        for &s in scores.iter() {
            if s > m {
                m = s;
            }
        }
        // If every position is masked (m == neg_inf_proxy or worse), the kernel
        // writes 0.0 to output (inv_S = 0 path, line 475).
        let all_masked = m <= neg_inf_proxy;

        let mut sum: f32 = 0.0_f32;
        let mut weights: Vec<f32> = vec![0.0_f32; kv_seq_len];
        if !all_masked {
            for (i, &s) in scores.iter().enumerate() {
                let w = (s - m).exp();
                weights[i] = w;
                sum += w;
            }
        }
        let inv_sum: f32 = if sum > 0.0_f32 { 1.0_f32 / sum } else { 0.0_f32 };

        // Pass 3: accumulate weighted V into output[h, :].
        let out_offset = h * dk;
        for d in 0..dk {
            output[out_offset + d] = 0.0_f32;
        }

        if !all_masked {
            for kv_pos in 0..kv_seq_len {
                let w = weights[kv_pos];
                if w == 0.0_f32 {
                    continue;
                }
                let v_packed_offset = (kv_head * kv_capacity + kv_pos) * dk;
                let v_packed_row: &[u8] = &v_packed[v_packed_offset..v_packed_offset + dk];

                if is_d512 {
                    let vnorm_offset = (kv_head * kv_capacity + kv_pos) * 2;
                    let vn0 = v_norms[vnorm_offset];
                    let vn1 = v_norms[vnorm_offset + 1];
                    let sn0 = vn0 / sf_d512;
                    let sn1 = vn1 / sf_d512;
                    for d in 0..256 {
                        let centroid = hb_centroid(v_packed_row[d], cbits);
                        output[out_offset + d] += centroid * sn0 * w;
                    }
                    for d in 256..dk {
                        let centroid = hb_centroid(v_packed_row[d], cbits);
                        output[out_offset + d] += centroid * sn1 * w;
                    }
                } else {
                    let vn = v_norms[kv_head * kv_capacity + kv_pos];
                    let sn = vn * inv_sqrt_dv;
                    for d in 0..dk {
                        let centroid = hb_centroid(v_packed_row[d], cbits);
                        output[out_offset + d] += centroid * sn * w;
                    }
                }
            }

            // Final divide by total softmax denominator (NWG=1 inv_S path).
            for d in 0..dk {
                output[out_offset + d] *= inv_sum;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::turboquant::{
        hb_nearest_centroid, CODEBOOK_HB_5BIT, CODEBOOK_HB_6BIT, CODEBOOK_HB_8BIT,
    };

    /// Helper: encode a single F32 row (head_dim=256) via the CPU encode path.
    /// FWHT + L2 norm + per-element nearest-centroid lookup. Mirrors what the
    /// `hadamard_quantize_kv_hb_d256` Metal kernel produces.
    fn encode_row_d256(x: &[f32], bits: u32) -> (Vec<u8>, f32) {
        let mut rotated = x.to_vec();
        crate::turboquant::fwht_inplace(&mut rotated).expect("fwht");
        let norm_sq: f32 = rotated.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt();
        if norm < 1e-30 {
            return (vec![0u8; x.len()], 0.0);
        }
        let scale = (x.len() as f32).sqrt() / norm;
        let mut packed = Vec::with_capacity(x.len());
        for &v in rotated.iter() {
            let scaled = v * scale;
            packed.push(hb_nearest_centroid(scaled, bits));
        }
        (packed, norm)
    }

    fn deterministic_gaussian(seed: u64, n: usize) -> Vec<f32> {
        // Box-Muller from a seeded LCG. Deterministic, no dependencies.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let next_u32 = |s: &mut u64| -> u32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s >> 32) as u32
        };
        let next_f32 = |s: &mut u64| -> f32 {
            let bits = next_u32(s);
            // Open-interval (0, 1).
            ((bits as f64 + 0.5) / (u32::MAX as f64 + 1.0)) as f32
        };
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = next_f32(&mut state).max(1e-7).min(1.0 - 1e-7);
            let u2 = next_f32(&mut state);
            let r = (-2.0_f32 * u1.ln()).sqrt();
            let theta = 2.0_f32 * std::f32::consts::PI * u2;
            out.push(r * theta.cos());
            if out.len() < n {
                out.push(r * theta.sin());
            }
        }
        out
    }

    #[test]
    fn codebooks_match_metal_shader_constants() {
        // Cross-check load-bearing values from the metal shader.
        // 5-bit endpoints (line 53 / 60 of flash_attn_vec_tq_hb.metal).
        assert!((CODEBOOK_HB_5BIT[0] - (-3.2606790)).abs() < 1e-6);
        assert!((CODEBOOK_HB_5BIT[31] - 3.2606790).abs() < 1e-6);
        // 6-bit endpoints.
        assert!((CODEBOOK_HB_6BIT[0] - (-3.6996161)).abs() < 1e-6);
        assert!((CODEBOOK_HB_6BIT[63] - 3.6996161).abs() < 1e-6);
        // 8-bit endpoints (line 92 / 155).
        assert!((CODEBOOK_HB_8BIT[0] - (-5.0652659)).abs() < 1e-6);
        assert!((CODEBOOK_HB_8BIT[255] - 5.0652659).abs() < 1e-6);
        // 8-bit symmetry (declared 3.41e-10 in the shader comment, line 88).
        for i in 0..128 {
            let sum = CODEBOOK_HB_8BIT[i] + CODEBOOK_HB_8BIT[255 - i];
            assert!(sum.abs() < 1e-5, "8-bit asymmetry at i={i}: {sum}");
        }
    }

    #[test]
    fn hb_centroid_lookup_matches_index() {
        // Spot-check a few indices vs the codebook arrays.
        for &idx in &[0u8, 1u8, 16u8, 31u8] {
            let v = hb_centroid(idx, 5);
            assert!((v - CODEBOOK_HB_5BIT[(idx & 0x1F) as usize]).abs() < 1e-7);
        }
        for &idx in &[0u8, 1u8, 32u8, 63u8] {
            let v = hb_centroid(idx, 6);
            assert!((v - CODEBOOK_HB_6BIT[(idx & 0x3F) as usize]).abs() < 1e-7);
        }
        for idx in 0u8..=255u8 {
            let v = hb_centroid(idx, 8);
            assert!((v - CODEBOOK_HB_8BIT[idx as usize]).abs() < 1e-7);
        }
    }

    #[test]
    fn hb_centroid_unsupported_bits_returns_zero() {
        // No-panic guarantee for invalid bits.
        assert_eq!(hb_centroid(0, 4), 0.0);
        assert_eq!(hb_centroid(255, 7), 0.0);
        assert_eq!(hb_nearest_centroid(0.0, 4), 0u8);
    }

    #[test]
    fn nearest_centroid_finds_closest() {
        // Index 128 is the centroid closest to zero on the 8-bit codebook
        // (positive side of the symmetric pair).
        // CODEBOOK_HB_8BIT[127] = -0.0135717, [128] = +0.0135717.
        // For value 0.005, the nearest is index 128 (positive side, dist 0.008572).
        // Wait: dist to [127] = abs(0.005 - (-0.0135717)) = 0.0185717
        //       dist to [128] = abs(0.005 - 0.0135717)   = 0.0085717
        // So nearest is 128.
        assert_eq!(hb_nearest_centroid(0.005, 8), 128);
        // Value 5.5 saturates to the high endpoint (index 255, value 5.0652659).
        assert_eq!(hb_nearest_centroid(5.5, 8), 255);
        // Value -5.5 saturates to the low endpoint (index 0).
        assert_eq!(hb_nearest_centroid(-5.5, 8), 0);
    }

    /// Sanity: oracle on a single-position cache with a known unit vector
    /// equals the manually computed attention.
    #[test]
    fn oracle_single_position_uniform_v_matches_manual() {
        let head_dim = 256u32;
        let num_heads = 1u32;
        let num_kv_heads = 1u32;
        let kv_capacity = 4u32;
        let kv_seq_len = 1u32;
        let bits = 8u32;

        // Encode a known K row (random gaussian, deterministic).
        let k_row = deterministic_gaussian(0xC25EED, head_dim as usize);
        let v_row = deterministic_gaussian(0xC25EED ^ 0xDEADBEEF, head_dim as usize);

        let (k_packed_row, k_norm) = encode_row_d256(&k_row, bits);
        let (v_packed_row, v_norm) = encode_row_d256(&v_row, bits);

        // Build the cache buffers (positions 1..3 zeroed).
        let mut k_packed = vec![0u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let mut k_norms = vec![0.0f32; (num_kv_heads * kv_capacity) as usize];
        let mut v_packed = vec![0u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let mut v_norms = vec![0.0f32; (num_kv_heads * kv_capacity) as usize];

        for d in 0..head_dim as usize {
            k_packed[d] = k_packed_row[d];
            v_packed[d] = v_packed_row[d];
        }
        k_norms[0] = k_norm;
        v_norms[0] = v_norm;

        // Q is a chosen unit vector (post-FWHT — caller's responsibility).
        let mut q = vec![0.0_f32; (num_heads * head_dim) as usize];
        for d in 0..head_dim as usize {
            q[d] = 1.0_f32 / (head_dim as f32).sqrt();
        }

        let params = TqHbOracleParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len,
            kv_capacity,
            scale: 1.0_f32 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: bits,
        };

        let mut output = vec![0.0_f32; (num_heads * head_dim) as usize];
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut output, &params).expect("oracle ok");

        // With a single valid kv_pos, softmax weight = 1.0, so output = V_dequant_row.
        // Compute expected V_dequant manually: centroid * v_norm * inv_sqrt(DK).
        let inv_sqrt_dk = 1.0_f32 / (head_dim as f32).sqrt();
        for d in 0..head_dim as usize {
            let expected = hb_centroid(v_packed_row[d], bits) * v_norm * inv_sqrt_dk;
            let actual = output[d];
            let diff = (actual - expected).abs();
            assert!(
                diff < 1e-5,
                "oracle output mismatch at d={d}: expected={expected}, actual={actual}, diff={diff}"
            );
        }
    }

    /// Determinism: same inputs produce bit-identical outputs across runs.
    #[test]
    fn oracle_is_bit_deterministic() {
        let head_dim = 256u32;
        let num_heads = 4u32;
        let num_kv_heads = 2u32;
        let kv_capacity = 16u32;
        let kv_seq_len = 8u32;

        let k_packed: Vec<u8> = (0..(num_kv_heads * kv_capacity * head_dim))
            .map(|i| (i.wrapping_mul(31) ^ 0xA5) as u8)
            .collect();
        let v_packed: Vec<u8> = (0..(num_kv_heads * kv_capacity * head_dim))
            .map(|i| (i.wrapping_mul(37) ^ 0x5A) as u8)
            .collect();
        let k_norms: Vec<f32> = (0..(num_kv_heads * kv_capacity))
            .map(|i| 1.0 + (i as f32) * 0.01)
            .collect();
        let v_norms: Vec<f32> = (0..(num_kv_heads * kv_capacity))
            .map(|i| 1.0 + (i as f32) * 0.02)
            .collect();
        let q: Vec<f32> = deterministic_gaussian(0xBEEF, (num_heads * head_dim) as usize);

        let params = TqHbOracleParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len,
            kv_capacity,
            scale: 0.0625,
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };

        let mut out_a = vec![0.0_f32; (num_heads * head_dim) as usize];
        let mut out_b = vec![0.0_f32; (num_heads * head_dim) as usize];
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut out_a, &params).expect("a");
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut out_b, &params).expect("b");

        for i in 0..out_a.len() {
            assert_eq!(out_a[i].to_bits(), out_b[i].to_bits(),
                "non-deterministic at i={i}: a={}, b={}", out_a[i], out_b[i]);
        }
    }

    /// Sliding-window mask: with mask_type=2 and a small window, only the
    /// most-recent `sliding_window` positions contribute.
    #[test]
    fn oracle_sliding_window_masks_old_positions() {
        let head_dim = 256u32;
        let num_heads = 1u32;
        let num_kv_heads = 1u32;
        let kv_capacity = 32u32;
        let kv_seq_len = 16u32;
        let sliding_window = 4u32;
        let bits = 8u32;

        // Build cache where positions 0..15 each store a distinguishable V
        // (we'll set v_norm to identify which position dominated).
        let k_row = deterministic_gaussian(0xCAFE, head_dim as usize);
        let v_row = deterministic_gaussian(0xBABE, head_dim as usize);
        let (k_packed_row, k_norm) = encode_row_d256(&k_row, bits);
        let (v_packed_row, v_norm) = encode_row_d256(&v_row, bits);

        let mut k_packed = vec![0u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let mut k_norms = vec![0.0f32; (num_kv_heads * kv_capacity) as usize];
        let mut v_packed = vec![0u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let mut v_norms = vec![0.0f32; (num_kv_heads * kv_capacity) as usize];
        for kv_pos in 0..kv_seq_len as usize {
            let off = kv_pos * head_dim as usize;
            for d in 0..head_dim as usize {
                k_packed[off + d] = k_packed_row[d];
                v_packed[off + d] = v_packed_row[d];
            }
            // Make v_norm per-position so the output reveals contributions.
            v_norms[kv_pos] = v_norm * (1.0 + kv_pos as f32);
            k_norms[kv_pos] = k_norm;
        }

        let mut q = vec![1.0_f32 / (head_dim as f32).sqrt(); (num_heads * head_dim) as usize];
        // Re-FWHT q so it correlates with the encoded K (caller responsibility).
        crate::turboquant::fwht_inplace(&mut q[..head_dim as usize]).expect("fwht");

        let params = TqHbOracleParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len,
            kv_capacity,
            scale: 1.0_f32 / (head_dim as f32).sqrt(),
            mask_type: 2,
            sliding_window,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: bits,
        };

        let mut out_windowed = vec![0.0_f32; (num_heads * head_dim) as usize];
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut out_windowed, &params).expect("ok");

        // Now disable masking and confirm output differs (sanity that masking
        // was actually applied).
        let params_no_mask = TqHbOracleParams { mask_type: 0, sliding_window: 0, ..params };
        let mut out_full = vec![0.0_f32; (num_heads * head_dim) as usize];
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut out_full, &params_no_mask).expect("ok");

        // The two outputs must differ when sliding_window < kv_seq_len.
        let mut max_diff = 0.0_f32;
        for i in 0..out_windowed.len() {
            max_diff = max_diff.max((out_windowed[i] - out_full[i]).abs());
        }
        assert!(max_diff > 1e-3, "sliding window had no effect: max_diff={max_diff}");
    }

    /// All-masked: when every position is masked out, output should be all zeros.
    /// Mirrors the kernel's `inv_S = 0` path (line 475).
    #[test]
    fn oracle_all_masked_returns_zeros() {
        let head_dim = 256u32;
        let num_heads = 1u32;
        let num_kv_heads = 1u32;
        let kv_capacity = 4u32;
        let kv_seq_len = 1u32;

        let k_packed = vec![128u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let v_packed = vec![128u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let k_norms = vec![1.0f32; (num_kv_heads * kv_capacity) as usize];
        let v_norms = vec![1.0f32; (num_kv_heads * kv_capacity) as usize];
        let q = vec![0.5_f32; (num_heads * head_dim) as usize];

        // ring_start outside the valid range so logical_idx is always >= kv_seq_len → masked.
        let params = TqHbOracleParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len,
            kv_capacity,
            scale: 1.0,
            mask_type: 2,
            sliding_window: kv_seq_len, // non-zero so window predicate engages
            softcap: 0.0,
            // logical_idx = (0 - 2 + 4) % 4 = 2, which >= kv_seq_len (1) → masked.
            ring_start: 2,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
        };

        let mut output = vec![1.0_f32; (num_heads * head_dim) as usize]; // pre-fill to detect zeroing
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut output, &params).expect("ok");
        for &v in output.iter() {
            assert_eq!(v.to_bits(), 0u32, "expected 0.0 in all-masked output, got {v}");
        }
    }

    /// D=512 path: per-block norms. Sanity check that the oracle splits coords
    /// 0..256 / 256..512 with separate norm scales.
    #[test]
    fn oracle_d512_per_block_norms() {
        let head_dim = 512u32;
        let num_heads = 1u32;
        let num_kv_heads = 1u32;
        let kv_capacity = 4u32;
        let kv_seq_len = 1u32;
        let bits = 8u32;
        let sf_d512: f32 = 16.0; // sqrt(256), matches AmesianX convention; orthogonal to oracle math.

        // Encode a 512-vec as two 256-blocks.
        let k_row = deterministic_gaussian(0x01234567, head_dim as usize);
        let mut k_b0 = k_row[0..256].to_vec();
        let mut k_b1 = k_row[256..512].to_vec();
        crate::turboquant::fwht_inplace(&mut k_b0).expect("fwht");
        crate::turboquant::fwht_inplace(&mut k_b1).expect("fwht");
        let n0 = k_b0.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let n1 = k_b1.iter().map(|&v| v * v).sum::<f32>().sqrt();
        // Encode each block as if it's an independent unit vector at sf_d512=16
        // ⇒ scaled coord = rotated * 16 / norm.
        let mut k_packed_row = vec![0u8; head_dim as usize];
        for d in 0..256 {
            let s = k_b0[d] * sf_d512 / n0;
            k_packed_row[d] = hb_nearest_centroid(s, bits);
        }
        for d in 0..256 {
            let s = k_b1[d] * sf_d512 / n1;
            k_packed_row[256 + d] = hb_nearest_centroid(s, bits);
        }

        let mut k_packed = vec![0u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let mut k_norms = vec![0.0f32; (num_kv_heads * kv_capacity * 2) as usize];
        // Position 0 with our encoded row + per-block norms.
        for d in 0..head_dim as usize {
            k_packed[d] = k_packed_row[d];
        }
        k_norms[0] = n0;
        k_norms[1] = n1;

        let v_packed = k_packed.clone();
        let v_norms = k_norms.clone();
        let q = vec![1.0_f32 / (head_dim as f32).sqrt(); (num_heads * head_dim) as usize];

        let params = TqHbOracleParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len,
            kv_capacity,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: sf_d512,
            codebook_bits: bits,
        };

        let mut out = vec![0.0f32; (num_heads * head_dim) as usize];
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut out, &params).expect("ok");

        // Single position → output[d] = V_dequant_row[d].
        // Block 0: centroid * (n0 / sf_d512). Block 1: centroid * (n1 / sf_d512).
        for d in 0..256 {
            let expected = hb_centroid(k_packed_row[d], bits) * (n0 / sf_d512);
            assert!((out[d] - expected).abs() < 1e-5,
                "d512 block0 mismatch d={d}: expected={expected}, actual={}", out[d]);
        }
        for d in 256..head_dim as usize {
            let expected = hb_centroid(k_packed_row[d], bits) * (n1 / sf_d512);
            assert!((out[d] - expected).abs() < 1e-5,
                "d512 block1 mismatch d={d}: expected={expected}, actual={}", out[d]);
        }
    }

    /// GQA: with num_heads=8, num_kv_heads=2, heads_per_kv=4. Heads 0..3 share
    /// kv_head=0, heads 4..7 share kv_head=1.
    #[test]
    fn oracle_gqa_routes_heads_to_correct_kv_head() {
        let head_dim = 256u32;
        let num_heads = 8u32;
        let num_kv_heads = 2u32;
        let kv_capacity = 4u32;
        let kv_seq_len = 1u32;
        let bits = 8u32;

        // Two distinguishable K/V rows: one per kv_head. We'll set v_norm so
        // the V row magnitude reveals which kv_head was consulted.
        let k_row = deterministic_gaussian(0x111, head_dim as usize);
        let v_row = deterministic_gaussian(0x222, head_dim as usize);
        let (k_packed_row, k_norm) = encode_row_d256(&k_row, bits);
        let (v_packed_row, v_norm) = encode_row_d256(&v_row, bits);

        let mut k_packed = vec![0u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let mut k_norms = vec![0.0f32; (num_kv_heads * kv_capacity) as usize];
        let mut v_packed = vec![0u8; (num_kv_heads * kv_capacity * head_dim) as usize];
        let mut v_norms = vec![0.0f32; (num_kv_heads * kv_capacity) as usize];

        // kv_head 0 at position 0: v_norm = 1.0 * v_norm
        for d in 0..head_dim as usize {
            k_packed[d] = k_packed_row[d];
            v_packed[d] = v_packed_row[d];
        }
        k_norms[0] = k_norm;
        v_norms[0] = v_norm;

        // kv_head 1 at position 0: v_norm = 10.0 * v_norm (distinguishable scale)
        let kv1_off = (kv_capacity * head_dim) as usize;
        for d in 0..head_dim as usize {
            k_packed[kv1_off + d] = k_packed_row[d];
            v_packed[kv1_off + d] = v_packed_row[d];
        }
        k_norms[(kv_capacity) as usize] = k_norm;
        v_norms[(kv_capacity) as usize] = 10.0 * v_norm;

        let q = vec![1.0_f32 / (head_dim as f32).sqrt(); (num_heads * head_dim) as usize];
        let params = TqHbOracleParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len,
            kv_capacity,
            scale: 1.0 / (head_dim as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: bits,
        };

        let mut out = vec![0.0f32; (num_heads * head_dim) as usize];
        flash_attn_vec_tq_hb_oracle(&q, &k_packed, &k_norms, &v_packed, &v_norms, &mut out, &params).expect("ok");

        // Heads 0..3 use kv_head 0 (v_norm baseline).
        // Heads 4..7 use kv_head 1 (v_norm 10× baseline).
        // For the first non-zero output dim, ratio of head4/head0 should be ≈ 10.
        let inv_sqrt_dk = 1.0_f32 / (head_dim as f32).sqrt();
        let expected_h0 = hb_centroid(v_packed_row[0], bits) * v_norm * inv_sqrt_dk;
        let expected_h4 = hb_centroid(v_packed_row[0], bits) * (10.0 * v_norm) * inv_sqrt_dk;

        let h0_d0 = out[(0 * head_dim) as usize];
        let h4_d0 = out[(4 * head_dim) as usize];

        assert!((h0_d0 - expected_h0).abs() < 1e-4,
            "h0 mismatch: expected={expected_h0}, actual={h0_d0}");
        assert!((h4_d0 - expected_h4).abs() < 1e-3,
            "h4 mismatch: expected={expected_h4}, actual={h4_d0}");
    }
}
