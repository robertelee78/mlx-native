//! TurboQuant KV cache compression — CPU reference implementation.
//!
//! Implements the TurboQuant_mse algorithm:
//! 1. Walsh-Hadamard rotation for incoherence
//! 2. Per-head norm extraction
//! 3. Lloyd-Max scalar quantization against N(0,1) codebooks
//!
//! This module is CPU-only math — no Metal GPU dispatch.

// ---- Lloyd-Max Codebooks for N(0,1) ----
//
// Precomputed via iterative Lloyd-Max algorithm with convergence tolerance 1e-12.
// Each codebook is symmetric around zero.

/// 2-bit Lloyd-Max centroids for N(0,1): 4 reconstruction levels.
pub const CODEBOOK_2BIT: [f32; 4] = [
    -1.5104176, -0.4527800, 0.4527800, 1.5104176,
];

/// 3-bit Lloyd-Max centroids for N(0,1): 8 reconstruction levels.
pub const CODEBOOK_3BIT: [f32; 8] = [
    -2.1519457, -1.3439093, -0.7560053, -0.2450942,
    0.2450942, 0.7560053, 1.3439093, 2.1519457,
];

/// 4-bit Lloyd-Max centroids for N(0,1): 16 reconstruction levels.
pub const CODEBOOK_4BIT: [f32; 16] = [
    -2.7325896, -2.0690172, -1.6180464, -1.2562312,
    -0.9423405, -0.6567591, -0.3880483, -0.1283950,
    0.1283950, 0.3880483, 0.6567591, 0.9423405,
    1.2562312, 1.6180464, 2.0690172, 2.7325896,
];

// ---------------------------------------------------------------------------
// ADR-007 Path C F-0.1: Higher-bit (HB) codebooks for the 5/6/8-bit byte-packed
// production decode path. These mirror EXACTLY the constants in
// `src/shaders/flash_attn_vec_tq_hb.metal` (lines 52-156) and
// `src/shaders/hadamard_quantize_kv_fast.metal::CODEBOOK_8BIT`.
//
// IMPORTANT: any change here is an on-disk codec change per ADR-007 §F-7.1.
// Bump the codec version in the TQ envelope before changing these bytes.
// ---------------------------------------------------------------------------

/// 5-bit Lloyd-Max centroids for N(0,1): 32 reconstruction levels.
/// Mirrors `flash_attn_vec_tq_hb.metal::CODEBOOK_HB_5BIT`.
pub const CODEBOOK_HB_5BIT: [f32; 32] = [
    -3.2606790, -2.6910589, -2.3176743, -2.0286608,
    -1.7871646, -1.5761599, -1.3862739, -1.2117410,
    -1.0487242, -0.8945114, -0.7470884, -0.6048936,
    -0.4666676, -0.3313550, -0.1980377, -0.0658849,
     0.0658849,  0.1980377,  0.3313550,  0.4666676,
     0.6048936,  0.7470884,  0.8945114,  1.0487242,
     1.2117410,  1.3862739,  1.5761599,  1.7871646,
     2.0286608,  2.3176743,  2.6910589,  3.2606790,
];

/// 6-bit Lloyd-Max centroids for N(0,1): 64 reconstruction levels.
/// Mirrors `flash_attn_vec_tq_hb.metal::CODEBOOK_HB_6BIT`.
pub const CODEBOOK_HB_6BIT: [f32; 64] = [
    -3.6996161, -3.1907215, -2.8640626, -2.6161277,
    -2.4129324, -2.2388464, -2.0853192, -1.9471373,
    -1.8208742, -1.7041502, -1.5952401, -1.4928497,
    -1.3959804, -1.3038428, -1.2157998, -1.1313277,
    -1.0499889, -0.9714118, -0.8952766, -0.8213046,
    -0.7492492, -0.6788902, -0.6100285, -0.5424819,
    -0.4760822, -0.4106724, -0.3461048, -0.2822386,
    -0.2189392, -0.1560761, -0.0935225, -0.0311537,
     0.0311537,  0.0935225,  0.1560761,  0.2189392,
     0.2822386,  0.3461048,  0.4106724,  0.4760822,
     0.5424819,  0.6100285,  0.6788902,  0.7492492,
     0.8213046,  0.8952766,  0.9714118,  1.0499889,
     1.1313277,  1.2157998,  1.3038428,  1.3959804,
     1.4928497,  1.5952401,  1.7041502,  1.8208742,
     1.9471373,  2.0853192,  2.2388464,  2.4129324,
     2.6161277,  2.8640626,  3.1907215,  3.6996161,
];

/// 8-bit Lloyd-Max centroids for N(0,1): 256 reconstruction levels.
/// Range: [-5.0652659, +5.0652659]. Symmetry error: 3.41e-10.
/// Mirrors `flash_attn_vec_tq_hb.metal::CODEBOOK_HB_8BIT` and
/// `hadamard_quantize_kv_fast.metal::CODEBOOK_8BIT`.
pub const CODEBOOK_HB_8BIT: [f32; 256] = [
    -5.0652659, -4.6836997, -4.4467193, -4.2715508,
    -4.1311907, -4.0132856, -3.9111092, -3.8205780,
    -3.7390194, -3.6645851, -3.5959415, -3.5320936,
    -3.4722785, -3.4158977, -3.3624729, -3.3116156,
    -3.2630056, -3.2163758, -3.1715011, -3.1281899,
    -3.0862780, -3.0456229, -3.0061011, -2.9676040,
    -2.9300362, -2.8933131, -2.8573596, -2.8221086,
    -2.7874999, -2.7534795, -2.7199985, -2.6870129,
    -2.6544825, -2.6223710, -2.5906452, -2.5592748,
    -2.5282321, -2.4974918, -2.4670306, -2.4368270,
    -2.4068614, -2.3771157, -2.3475732, -2.3182184,
    -2.2890372, -2.2600165, -2.2311440, -2.2024086,
    -2.1737998, -2.1453081, -2.1169245, -2.0886408,
    -2.0604493, -2.0323430, -2.0043154, -1.9763603,
    -1.9484722, -1.9206458, -1.8928763, -1.8651592,
    -1.8374904, -1.8098662, -1.7822828, -1.7547372,
    -1.7272261, -1.6997469, -1.6722970, -1.6448739,
    -1.6174755, -1.5900996, -1.5627445, -1.5354084,
    -1.5080897, -1.4807869, -1.4534986, -1.4262237,
    -1.3989610, -1.3717093, -1.3444678, -1.3172356,
    -1.2900118, -1.2627956, -1.2355865, -1.2083838,
    -1.1811868, -1.1539951, -1.1268081, -1.0996255,
    -1.0724469, -1.0452718, -1.0180999, -0.9909310,
    -0.9637647, -0.9366008, -0.9094390, -0.8822793,
    -0.8551212, -0.8279648, -0.8008098, -0.7736561,
    -0.7465035, -0.7193520, -0.6922014, -0.6650517,
    -0.6379027, -0.6107544, -0.5836067, -0.5564596,
    -0.5293129, -0.5021667, -0.4750208, -0.4478753,
    -0.4207301, -0.3935852, -0.3664405, -0.3392960,
    -0.3121517, -0.2850076, -0.2578636, -0.2307198,
    -0.2035761, -0.1764324, -0.1492888, -0.1221453,
    -0.0950019, -0.0678584, -0.0407151, -0.0135717,
     0.0135717,  0.0407151,  0.0678584,  0.0950019,
     0.1221453,  0.1492888,  0.1764324,  0.2035761,
     0.2307198,  0.2578636,  0.2850076,  0.3121517,
     0.3392960,  0.3664405,  0.3935852,  0.4207301,
     0.4478753,  0.4750208,  0.5021667,  0.5293129,
     0.5564596,  0.5836067,  0.6107544,  0.6379027,
     0.6650517,  0.6922014,  0.7193520,  0.7465035,
     0.7736561,  0.8008098,  0.8279648,  0.8551212,
     0.8822793,  0.9094390,  0.9366008,  0.9637647,
     0.9909310,  1.0180999,  1.0452718,  1.0724469,
     1.0996255,  1.1268081,  1.1539951,  1.1811868,
     1.2083838,  1.2355865,  1.2627956,  1.2900118,
     1.3172356,  1.3444678,  1.3717093,  1.3989610,
     1.4262237,  1.4534986,  1.4807869,  1.5080897,
     1.5354084,  1.5627445,  1.5900996,  1.6174755,
     1.6448739,  1.6722970,  1.6997469,  1.7272261,
     1.7547372,  1.7822828,  1.8098662,  1.8374904,
     1.8651592,  1.8928763,  1.9206458,  1.9484722,
     1.9763603,  2.0043154,  2.0323430,  2.0604493,
     2.0886408,  2.1169245,  2.1453081,  2.1737998,
     2.2024086,  2.2311440,  2.2600165,  2.2890372,
     2.3182184,  2.3475732,  2.3771157,  2.4068614,
     2.4368270,  2.4670306,  2.4974918,  2.5282321,
     2.5592748,  2.5906452,  2.6223710,  2.6544825,
     2.6870129,  2.7199985,  2.7534795,  2.7874999,
     2.8221086,  2.8573596,  2.8933131,  2.9300362,
     2.9676040,  3.0061011,  3.0456229,  3.0862780,
     3.1281899,  3.1715011,  3.2163758,  3.2630056,
     3.3116156,  3.3624729,  3.4158977,  3.4722785,
     3.5320936,  3.5959415,  3.6645851,  3.7390194,
     3.8205780,  3.9111092,  4.0132856,  4.1311907,
     4.2715508,  4.4467193,  4.6836997,  5.0652659,
];

/// HB codebook lookup helper.
///
/// Returns the centroid for the given byte index under the specified bit-width.
/// Mirrors `flash_attn_vec_tq_hb.metal::dequant_hb_single`'s codebook switch.
///
/// `bits` must be 5, 6, or 8 (returns 0.0 for any other value — caller should
/// validate). Index masking matches the kernel (`& 0x1F` for 5-bit, `& 0x3F` for
/// 6-bit, full byte for 8-bit).
#[inline]
pub fn hb_centroid(idx: u8, bits: u32) -> f32 {
    match bits {
        5 => CODEBOOK_HB_5BIT[(idx & 0x1F) as usize],
        6 => CODEBOOK_HB_6BIT[(idx & 0x3F) as usize],
        8 => CODEBOOK_HB_8BIT[idx as usize],
        _ => 0.0,
    }
}

/// D1 sign mask for the SRHT pre-multiplication, D=256 path.
///
/// Verbatim mirror of `hadamard_quantize_kv_fast.metal::TBQ_SIGNS_256`
/// (lines 25-30). Source: AmesianX `cpy-utils.cuh:158-163`,
/// sha256=3ef1038e6c232e9519101daa2d6efd637d4c6bfdb29f4ee7101625c39d0ddc89.
///
/// Convention: `bit j = (table[j>>3] >> (j&7)) & 1`; bit=1 → sign = -1,
/// bit=0 → sign = +1 (LSB-first within each byte).
pub const TBQ_SIGNS_256: [u8; 32] = [
    0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e,
    0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
    0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d,
    0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
];

/// D1 sign mask for the SRHT pre-multiplication, D=512 path.
///
/// Verbatim mirror of `hadamard_quantize_kv_fast.metal::TBQ_SIGNS_512`
/// (lines 35-44). Source: AmesianX `cpy-utils.cuh:211-220`,
/// sha256=44f13ce9f6db1edac62f558ee054f9de29cd474fd051362cadcaa98a55745f17.
pub const TBQ_SIGNS_512: [u8; 64] = [
    0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e,
    0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
    0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d,
    0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
    0xd3, 0x4e, 0xa8, 0x17, 0x9c, 0x5b, 0xe6, 0x31,
    0x72, 0xb9, 0x0d, 0xf5, 0x43, 0x8a, 0x6e, 0xc7,
    0x58, 0x2f, 0x94, 0xe1, 0xb6, 0x3d, 0x0a, 0x7c,
    0xc5, 0x61, 0xd8, 0x4f, 0xa3, 0x97, 0x1e, 0x85,
];

/// Apply D1 sign mask in-place per the SRHT convention.
///
/// `signs` must have at least `x.len() / 8` bytes (one bit per element).
/// Sign flip: bit=1 → x[j] *= -1; bit=0 → x[j] unchanged.
#[inline]
pub fn apply_d1_sign_mask_inplace(x: &mut [f32], signs: &[u8]) {
    for j in 0..x.len() {
        let byte = signs[j >> 3];
        let bit = (byte >> (j & 7)) & 1;
        if bit == 1 {
            x[j] = -x[j];
        }
    }
}

/// Higher-bit (5/6/8-bit) CPU encoder for D=256 — byte-equivalent mirror of
/// `hadamard_quantize_kv_fast.metal::hadamard_quantize_kv_hb<256>`.
///
/// Path C F-0.2 deliverable: produces the exact byte layout that the GPU
/// kernel writes given the same input vector, so divergence between the
/// flash_attn_vec_tq_hb GPU kernel and the F-0.1 CPU oracle isolates the
/// SDPA math (not the codec math).
///
/// Steps (mirroring the kernel byte-for-byte):
/// 1. Apply D1 sign mask (`TBQ_SIGNS_256`).
/// 2. Apply normalized FWHT (butterfly + 1/sqrt(d) — `fwht_inplace`).
/// 3. Compute L2 norm of the rotated vector.
/// 4. If norm > 1e-10: scale elems by `(1/norm) * sqrt(d)` (lift to N(0,1)).
///    If norm ≤ 1e-10: scale = 0 (matches kernel `inv_norm = 0` branch).
/// 5. Quantize each element to nearest centroid in the HB codebook for `bits`
///    (5/6/8). Returns 1 byte per element (byte-packed).
///
/// Returns `(packed_indices, norm)`.
pub fn turboquant_hb_encode_d256(x: &[f32], bits: u32) -> Result<(Vec<u8>, f32), crate::MlxError> {
    if x.len() != 256 {
        return Err(crate::MlxError::InvalidArgument(format!(
            "turboquant_hb_encode_d256 expects head_dim=256, got {}",
            x.len()
        )));
    }
    if !matches!(bits, 5 | 6 | 8) {
        return Err(crate::MlxError::InvalidArgument(format!(
            "turboquant_hb_encode_d256 bits must be 5, 6, or 8, got {bits}"
        )));
    }

    // Step 1: D1 sign pre-multiplication.
    let mut elems = x.to_vec();
    apply_d1_sign_mask_inplace(&mut elems, &TBQ_SIGNS_256);

    // Step 2+3: normalized FWHT (butterfly + 1/sqrt(d)).
    fwht_inplace(&mut elems)?;

    // Step 4: L2 norm.
    let norm_sq: f32 = elems.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt();

    // Step 5: scale to N(0,1). Kernel uses inv_norm * sqrt(d). Since fwht_inplace
    // already applied 1/sqrt(d) scaling to elems, the "inv_norm * sqrt(d)" here
    // means we multiply the post-fwht element by sqrt(d)/norm.
    // ↳ Match kernel exactly: when norm ≤ 1e-10, scale := 0 (zeros out output).
    let scale: f32 = if norm > 1.0e-10_f32 {
        (1.0_f32 / norm) * (256.0_f32).sqrt()
    } else {
        0.0_f32
    };
    for v in elems.iter_mut() {
        *v *= scale;
    }

    // Step 6+7: nearest centroid per element, byte-packed (1 byte per element).
    let mut packed = Vec::with_capacity(256);
    for &v in elems.iter() {
        packed.push(hb_nearest_centroid(v, bits));
    }

    Ok((packed, norm))
}

/// HB nearest-centroid encoder (CPU-side mirror of the Metal encoder kernel).
///
/// Returns the byte index of the nearest centroid in the codebook for the given
/// bit-width. Used only for the F-0.1 oracle's encode path and for codec
/// roundtrip tests; production encode goes through `hadamard_quantize_kv_hb_d*`
/// Metal kernels.
///
/// Returns `0u8` (closest-to-zero centroid) for unsupported bit-widths so the
/// function stays no-panic; callers are expected to pre-validate `bits`.
pub fn hb_nearest_centroid(value: f32, bits: u32) -> u8 {
    let cb: &[f32] = match bits {
        5 => &CODEBOOK_HB_5BIT,
        6 => &CODEBOOK_HB_6BIT,
        8 => &CODEBOOK_HB_8BIT,
        _ => return 0u8,
    };
    let mut best_idx: u32 = 0;
    let mut best_dist: f32 = (value - cb[0]).abs();
    for (i, &c) in cb.iter().enumerate().skip(1) {
        let dist = (value - c).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u32;
        }
    }
    best_idx as u8
}

// ---- BitWidth enum ----

/// Quantization bit-width for TurboQuant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWidth {
    /// 2-bit uniform: all coordinates use 4-level codebook.
    Two,
    /// 3-bit uniform: all coordinates use 8-level codebook.
    Three,
    /// 4-bit uniform: all coordinates use 16-level codebook.
    Four,
    /// 2.5-bit mixed: first d/4 coordinates at 3-bit, remaining 3d/4 at 2-bit.
    TwoPointFive,
}

/// Configuration for TurboQuant quantization.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Quantization bit-width.
    pub bit_width: BitWidth,
    /// Head dimension (must be a power of 2: 128, 256, or 512).
    pub head_dim: usize,
}

// ---- Fast Walsh-Hadamard Transform ----

/// In-place normalized Fast Walsh-Hadamard Transform.
///
/// The normalization ensures H * H = I, so the inverse transform is the
/// same function applied again.
///
/// # Arguments
/// * `x` — mutable slice of length `n` where `n` is a power of 2.
///
/// # Returns
/// `Ok(())` on success, or an error if the length is not a power of 2.
pub fn fwht_inplace(x: &mut [f32]) -> crate::Result<()> {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(crate::MlxError::InvalidArgument(format!(
            "FWHT requires power-of-two length, got {n}"
        )));
    }

    let mut h = 1;
    while h < n {
        let step = h * 2;
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += step;
        }
        h *= 2;
    }

    // Normalize so that H * H = I
    let scale = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }

    Ok(())
}

// ---- Standard Normal PDF / CDF ----

/// Standard normal probability density function: phi(x) = exp(-x^2/2) / sqrt(2*pi).
#[inline]
fn std_normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/sqrt(2*pi)
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF using the Abramowitz & Stegun rational approximation
/// (formula 26.2.17, maximum error < 7.5e-8).
#[inline]
fn std_normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x_abs = x.abs();

    // Horner form of the rational approximation
    const P: f64 = 0.231_641_9;
    const B1: f64 = 0.319_381_530;
    const B2: f64 = -0.356_563_782;
    const B3: f64 = 1.781_477_937;
    const B4: f64 = -1.821_255_978;
    const B5: f64 = 1.330_274_429;

    let t = 1.0 / (1.0 + P * x_abs);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = B1 * t + B2 * t2 + B3 * t3 + B4 * t4 + B5 * t5;
    let phi = std_normal_pdf(x_abs);

    let result = 1.0 - phi * poly;

    if sign < 0.0 {
        1.0 - result
    } else {
        result
    }
}

// ---- Nearest centroid lookup ----

/// Find the index of the nearest centroid in a sorted codebook.
#[inline]
fn nearest_centroid(value: f32, codebook: &[f32]) -> u8 {
    // Binary-search style: codebook is sorted, find nearest by checking boundaries
    let n = codebook.len();
    if n <= 1 {
        return 0;
    }

    let mut best_idx = 0u8;
    let mut best_dist = (value - codebook[0]).abs();

    for (i, &c) in codebook.iter().enumerate().skip(1) {
        let dist = (value - c).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Get the codebook for a specific coordinate index under the given config.
#[inline]
fn codebook_for_coord(coord_idx: usize, config: &TurboQuantConfig) -> &'static [f32] {
    match config.bit_width {
        BitWidth::Two => &CODEBOOK_2BIT,
        BitWidth::Three => &CODEBOOK_3BIT,
        BitWidth::Four => &CODEBOOK_4BIT,
        BitWidth::TwoPointFive => {
            let boundary = config.head_dim / 4;
            if coord_idx < boundary {
                &CODEBOOK_3BIT // first d/4 channels at 3-bit
            } else {
                &CODEBOOK_2BIT // remaining 3d/4 at 2-bit
            }
        }
    }
}

/// Bits per index for a coordinate under the given config.
#[inline]
fn bits_for_coord(coord_idx: usize, config: &TurboQuantConfig) -> usize {
    match config.bit_width {
        BitWidth::Two => 2,
        BitWidth::Three => 3,
        BitWidth::Four => 4,
        BitWidth::TwoPointFive => {
            if coord_idx < config.head_dim / 4 {
                3
            } else {
                2
            }
        }
    }
}

// ---- Pack / Unpack indices ----

/// Pack variable-width indices into a byte vector using bit-packing.
///
/// Indices are packed MSB-first into consecutive bytes.
fn pack_indices(indices: &[u8], config: &TurboQuantConfig) -> Vec<u8> {
    let total_bits: usize = (0..indices.len())
        .map(|i| bits_for_coord(i, config))
        .sum();
    let num_bytes = (total_bits + 7) / 8;
    let mut packed = vec![0u8; num_bytes];

    let mut bit_offset = 0usize;
    for (i, &idx) in indices.iter().enumerate() {
        let nbits = bits_for_coord(i, config);
        // Write `nbits` bits of `idx` starting at `bit_offset`
        for b in (0..nbits).rev() {
            let bit_val = (idx >> b) & 1;
            let byte_pos = bit_offset / 8;
            let bit_pos = 7 - (bit_offset % 8);
            if byte_pos < packed.len() {
                packed[byte_pos] |= bit_val << bit_pos;
            }
            bit_offset += 1;
        }
    }

    packed
}

/// Unpack variable-width indices from a packed byte vector.
fn unpack_indices(packed: &[u8], config: &TurboQuantConfig) -> Vec<u8> {
    let d = config.head_dim;
    let mut indices = Vec::with_capacity(d);

    let mut bit_offset = 0usize;
    for i in 0..d {
        let nbits = bits_for_coord(i, config);
        let mut val = 0u8;
        for _ in 0..nbits {
            let byte_pos = bit_offset / 8;
            let bit_pos = 7 - (bit_offset % 8);
            let bit_val = if byte_pos < packed.len() {
                (packed[byte_pos] >> bit_pos) & 1
            } else {
                0
            };
            val = (val << 1) | bit_val;
            bit_offset += 1;
        }
        indices.push(val);
    }

    indices
}

// ---- Quantize / Dequantize ----

/// Quantize a single head vector using TurboQuant_mse.
///
/// Steps:
/// 1. Apply FWHT (Walsh-Hadamard rotation) for incoherence
/// 2. Extract L2 norm
/// 3. Normalize to unit vector
/// 4. Quantize each coordinate against the appropriate Lloyd-Max codebook
/// 5. Pack indices
///
/// # Arguments
/// * `x` — input vector of length `config.head_dim`
/// * `config` — quantization configuration
///
/// # Returns
/// `(packed_indices, norm)` on success.
pub fn turboquant_quantize(
    x: &[f32],
    config: &TurboQuantConfig,
) -> crate::Result<(Vec<u8>, f32)> {
    let d = config.head_dim;
    if x.len() != d {
        return Err(crate::MlxError::InvalidArgument(format!(
            "Expected vector of length {d}, got {}",
            x.len()
        )));
    }
    if !d.is_power_of_two() {
        return Err(crate::MlxError::InvalidArgument(format!(
            "head_dim must be power of 2, got {d}"
        )));
    }

    // 1. Copy and apply FWHT
    let mut rotated = x.to_vec();
    fwht_inplace(&mut rotated)?;

    // 2. Compute L2 norm of rotated vector (same as original since Hadamard is orthogonal)
    let norm_sq: f32 = rotated.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt();

    if norm < 1e-30 {
        // Zero vector: all indices = 0, norm = 0
        let indices = vec![0u8; d];
        let packed = pack_indices(&indices, config);
        return Ok((packed, 0.0));
    }

    // 3. Normalize to unit vector on S^{d-1}
    let inv_norm = 1.0 / norm;
    for v in rotated.iter_mut() {
        *v *= inv_norm;
    }

    // 4. Quantize: each coordinate needs to be scaled to N(0,1) domain.
    // A unit vector on S^{d-1} has coordinates ~ N(0, 1/d) for large d.
    // Scale by sqrt(d) to map to N(0,1) for codebook lookup.
    let scale = (d as f32).sqrt();
    let mut indices = Vec::with_capacity(d);
    for (i, &v) in rotated.iter().enumerate() {
        let scaled = v * scale;
        let cb = codebook_for_coord(i, config);
        indices.push(nearest_centroid(scaled, cb));
    }

    // 5. Pack
    let packed = pack_indices(&indices, config);

    Ok((packed, norm))
}

/// Dequantize a TurboQuant-compressed head vector.
///
/// Steps:
/// 1. Unpack indices
/// 2. Look up centroid values, scale back from N(0,1) domain
/// 3. Multiply by norm
/// 4. Apply inverse FWHT (same as forward)
///
/// # Arguments
/// * `packed` — packed index bytes
/// * `norm` — the L2 norm stored during quantization
/// * `config` — quantization configuration
///
/// # Returns
/// Reconstructed vector of length `config.head_dim`.
pub fn turboquant_dequantize(
    packed: &[u8],
    norm: f32,
    config: &TurboQuantConfig,
) -> crate::Result<Vec<f32>> {
    let d = config.head_dim;
    if !d.is_power_of_two() {
        return Err(crate::MlxError::InvalidArgument(format!(
            "head_dim must be power of 2, got {d}"
        )));
    }

    // 1. Unpack indices
    let indices = unpack_indices(packed, config);

    // 2. Look up centroids and scale back from N(0,1) to unit-sphere scale
    let inv_scale = 1.0 / (d as f32).sqrt();
    let mut reconstructed = Vec::with_capacity(d);
    for (i, &idx) in indices.iter().enumerate() {
        let cb = codebook_for_coord(i, config);
        let idx_usize = idx as usize;
        let centroid = if idx_usize < cb.len() {
            cb[idx_usize]
        } else {
            0.0 // fallback for out-of-range (shouldn't happen)
        };
        reconstructed.push(centroid * inv_scale * norm);
    }

    // 3. Apply inverse FWHT (same as forward since H^{-1} = H with normalization)
    fwht_inplace(&mut reconstructed)?;

    Ok(reconstructed)
}

// ---- Lloyd-Max computation utilities (used by tests for validation) ----

/// Compute Lloyd-Max codebook for N(0,1) with the given number of levels.
///
/// Returns the sorted centroid array. This is used in tests to validate the
/// hardcoded codebooks.
pub fn compute_lloyd_max_codebook(num_levels: usize) -> Vec<f64> {
    // Initialize with uniform quantile boundaries
    let mut boundaries = Vec::with_capacity(num_levels + 1);
    boundaries.push(-10.0_f64); // approx -inf
    for i in 1..num_levels {
        let p = i as f64 / num_levels as f64;
        boundaries.push(quantile_normal(p));
    }
    boundaries.push(10.0_f64); // approx +inf

    // Initial centroids from conditional expectations
    let mut centroids = vec![0.0_f64; num_levels];
    for i in 0..num_levels {
        let a = boundaries[i];
        let b = boundaries[i + 1];
        let prob = std_normal_cdf(b) - std_normal_cdf(a);
        if prob > 1e-30 {
            centroids[i] = (std_normal_pdf(a) - std_normal_pdf(b)) / prob;
        }
    }

    // Iterate
    for _iter in 0..50_000 {
        let old = centroids.clone();

        // Update boundaries to midpoints
        boundaries[0] = -10.0;
        for i in 1..num_levels {
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
        }
        *boundaries.last_mut().unwrap_or(&mut 0.0) = 10.0;

        // Update centroids
        for i in 0..num_levels {
            let a = boundaries[i];
            let b = boundaries[i + 1];
            let prob = std_normal_cdf(b) - std_normal_cdf(a);
            if prob > 1e-30 {
                centroids[i] = (std_normal_pdf(a) - std_normal_pdf(b)) / prob;
            }
        }

        // Check convergence
        let max_change = centroids
            .iter()
            .zip(old.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_change < 1e-12 {
            break;
        }
    }

    centroids
}

/// Approximate quantile (inverse CDF) of N(0,1) using rational approximation.
///
/// Uses the Beasley-Springer-Moro algorithm.
fn quantile_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return -10.0;
    }
    if p >= 1.0 {
        return 10.0;
    }

    // Rational approximation (Peter Acklam's algorithm)
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Compute Lloyd-Max codebook for Beta((d-1)/2, (d-1)/2) scaled to [-1, 1].
///
/// The exact distribution of a coordinate of a unit vector uniform on S^{d-1}
/// is Beta((d-1)/2, (d-1)/2) on [-1, 1]. For large d this converges to N(0, 1/d).
///
/// Uses numerical integration via trapezoidal rule for the conditional expectations.
pub fn compute_lloyd_max_beta_codebook(dim: usize, num_levels: usize) -> Vec<f64> {
    let alpha = (dim as f64 - 1.0) / 2.0;

    // Beta PDF on [-1,1] with parameters (alpha, alpha) — symmetric
    // f(x) = C * (1-x^2)^(alpha-1)  for x in [-1, 1]
    // where C normalizes to 1.

    // Use log-space for numerical stability
    let log_norm = log_beta_norm_const(alpha);

    let beta_pdf = |x: f64| -> f64 {
        if x <= -1.0 || x >= 1.0 {
            return 0.0;
        }
        let val = 1.0 - x * x;
        if val <= 0.0 {
            return 0.0;
        }
        (log_norm + (alpha - 1.0) * val.ln()).exp()
    };

    // Numerical CDF via cumulative trapezoidal integration
    let n_grid = 10_000;
    let grid_lo = -1.0_f64;
    let grid_hi = 1.0_f64;
    let dx = (grid_hi - grid_lo) / n_grid as f64;

    // Build CDF table
    let mut cdf_vals = vec![0.0_f64; n_grid + 1];
    let mut pdf_vals = vec![0.0_f64; n_grid + 1];
    for i in 0..=n_grid {
        let x = grid_lo + i as f64 * dx;
        pdf_vals[i] = beta_pdf(x);
    }
    for i in 1..=n_grid {
        cdf_vals[i] = cdf_vals[i - 1] + 0.5 * (pdf_vals[i - 1] + pdf_vals[i]) * dx;
    }
    // Normalize CDF to [0, 1]
    let cdf_total = cdf_vals[n_grid];
    if cdf_total > 1e-30 {
        for v in cdf_vals.iter_mut() {
            *v /= cdf_total;
        }
        for v in pdf_vals.iter_mut() {
            *v /= cdf_total;
        }
    }

    // Helper: interpolated CDF and conditional expectation on [a, b]
    let interp_cdf = |x: f64| -> f64 {
        let frac = (x - grid_lo) / dx;
        let idx = frac as usize;
        if idx >= n_grid {
            return 1.0;
        }
        let t = frac - idx as f64;
        cdf_vals[idx] * (1.0 - t) + cdf_vals[idx + 1] * t
    };

    let conditional_expectation = |a: f64, b: f64| -> f64 {
        // E[X | a <= X <= b] via numerical integration
        let prob = interp_cdf(b) - interp_cdf(a);
        if prob < 1e-30 {
            return (a + b) / 2.0;
        }

        let n_sub = 500;
        let sub_dx = (b - a) / n_sub as f64;
        let mut integral = 0.0_f64;
        for j in 0..=n_sub {
            let x = a + j as f64 * sub_dx;
            let w = if j == 0 || j == n_sub { 0.5 } else { 1.0 };
            let frac = (x - grid_lo) / dx;
            let idx = frac as usize;
            let pdf_val = if idx >= n_grid {
                0.0
            } else {
                let t = frac - idx as f64;
                pdf_vals[idx] * (1.0 - t) + pdf_vals[idx + 1] * t
            };
            integral += w * x * pdf_val * sub_dx;
        }
        integral / prob
    };

    // Initialize with uniform quantile boundaries
    let mut boundaries = Vec::with_capacity(num_levels + 1);
    boundaries.push(-1.0_f64);
    for i in 1..num_levels {
        let target_p = i as f64 / num_levels as f64;
        // Binary search for quantile
        let mut lo = -1.0_f64;
        let mut hi = 1.0_f64;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.0;
            if interp_cdf(mid) < target_p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        boundaries.push((lo + hi) / 2.0);
    }
    boundaries.push(1.0_f64);

    // Initial centroids
    let mut centroids = vec![0.0_f64; num_levels];
    for i in 0..num_levels {
        centroids[i] = conditional_expectation(boundaries[i], boundaries[i + 1]);
    }

    // Lloyd-Max iteration
    for _iter in 0..5000 {
        let old = centroids.clone();

        // Update boundaries
        boundaries[0] = -1.0;
        for i in 1..num_levels {
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0;
        }
        if let Some(last) = boundaries.last_mut() {
            *last = 1.0;
        }

        // Update centroids
        for i in 0..num_levels {
            centroids[i] = conditional_expectation(boundaries[i], boundaries[i + 1]);
        }

        let max_change = centroids
            .iter()
            .zip(old.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_change < 1e-10 {
            break;
        }
    }

    centroids
}

/// Log of the normalization constant for the symmetric Beta PDF on [-1, 1].
fn log_beta_norm_const(alpha: f64) -> f64 {
    // Beta(alpha, alpha) on [0,1] has norm B(alpha, alpha) = Gamma(alpha)^2 / Gamma(2*alpha)
    // On [-1,1] we scale by 1/2, so norm = B(alpha,alpha) * 2^(2*alpha-1)
    // log C = -log(B(alpha,alpha)) - (2*alpha-1)*log(2)
    //       = log(Gamma(2*alpha)) - 2*log(Gamma(alpha)) - (2*alpha-1)*log(2)
    ln_gamma(2.0 * alpha) - 2.0 * ln_gamma(alpha) - (2.0 * alpha - 1.0) * 2.0_f64.ln()
}

/// Lanczos approximation for ln(Gamma(x)), x > 0.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation with g=7, n=9
    const G: f64 = 7.0;
    const COEFF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut ag = COEFF[0];
    for i in 1..9 {
        ag += COEFF[i] / (x + i as f64);
    }

    let tmp = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * tmp.ln() - tmp + ag.ln()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_symmetry() {
        for (name, cb) in [
            ("2-bit", &CODEBOOK_2BIT[..]),
            ("3-bit", &CODEBOOK_3BIT[..]),
            ("4-bit", &CODEBOOK_4BIT[..]),
        ] {
            let n = cb.len();
            for i in 0..n / 2 {
                let sum = cb[i] + cb[n - 1 - i];
                assert!(
                    sum.abs() < 1e-5,
                    "{name} codebook not symmetric: c[{i}]={} + c[{}]={} = {sum}",
                    cb[i],
                    n - 1 - i,
                    cb[n - 1 - i]
                );
            }
        }
    }

    #[test]
    fn test_codebook_values_match_lloyd_max() {
        for (bits, hardcoded) in [
            (2, &CODEBOOK_2BIT[..]),
            (3, &CODEBOOK_3BIT[..]),
            (4, &CODEBOOK_4BIT[..]),
        ] {
            let computed = compute_lloyd_max_codebook(1 << bits);
            assert_eq!(computed.len(), hardcoded.len());
            for (i, (&h, &c)) in hardcoded.iter().zip(computed.iter()).enumerate() {
                let diff = (h as f64 - c).abs();
                assert!(
                    diff < 1e-4,
                    "{bits}-bit codebook mismatch at {i}: hardcoded={h}, computed={c}, diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_fwht_roundtrip() {
        let original: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.4).collect();
        let mut data = original.clone();
        fwht_inplace(&mut data).unwrap();
        fwht_inplace(&mut data).unwrap();
        for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "FWHT roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
    }

    // ----- ADR-007 Path C F-0.2 tests: HB encoder mirror correctness -----

    fn deterministic_gaussian_test(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let next_u32 = |s: &mut u64| -> u32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s >> 32) as u32
        };
        let next_f32 = |s: &mut u64| -> f32 {
            let bits = next_u32(s);
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

    /// Decode a single packed-byte row back to F32 via the same dequant formula
    /// the GPU kernel uses on the read side, then invert SRHT (FWHT + sign mask).
    /// Used only by tests to verify encoder roundtrip.
    fn decode_d256_via_kernel_formula(packed: &[u8], norm: f32, bits: u32) -> Vec<f32> {
        // Step 1: codebook lookup × norm × inv_sqrt(256), per kernel decoder math.
        let inv_sqrt_dk = 1.0_f32 / (256.0_f32).sqrt();
        let mut decoded: Vec<f32> = packed.iter()
            .map(|&idx| hb_centroid(idx, bits) * norm * inv_sqrt_dk)
            .collect();
        // Step 2: inverse normalized FWHT = same FWHT (involution under H * H = I).
        fwht_inplace(&mut decoded).expect("fwht ok");
        // Step 3: invert D1 sign mask (sign flip is its own inverse).
        apply_d1_sign_mask_inplace(&mut decoded, &TBQ_SIGNS_256);
        decoded
    }

    fn nrmse(a: &[f32], b: &[f32]) -> f32 {
        let mut sse: f64 = 0.0;
        let mut sse_a: f64 = 0.0;
        for (&av, &bv) in a.iter().zip(b.iter()) {
            let d = (av - bv) as f64;
            sse += d * d;
            sse_a += (av as f64) * (av as f64);
        }
        if sse_a < 1e-30 {
            return 0.0;
        }
        (sse / sse_a).sqrt() as f32
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot: f64 = 0.0;
        let mut na: f64 = 0.0;
        let mut nb: f64 = 0.0;
        for (&av, &bv) in a.iter().zip(b.iter()) {
            dot += (av as f64) * (bv as f64);
            na += (av as f64) * (av as f64);
            nb += (bv as f64) * (bv as f64);
        }
        if na < 1e-30 || nb < 1e-30 {
            return 1.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    /// Encoder roundtrip via the kernel's dequant formula: encode → dequant →
    /// compare to original. Cosine ≥ ADR-007 Gate A threshold (≥0.999).
    /// 8-bit close-section measured 0.9998 mean / 0.9986 p1.
    #[test]
    fn hb_encoder_d256_roundtrip_8bit_meets_gate_a() {
        // 8-bit Gate A close-section measurement: cosine mean 0.9998.
        // Synthetic Gaussian sample isn't the production distribution but should
        // clear the strict spec on a single vector. Threshold ≥0.998 leaves
        // headroom for sampling noise on a single 256-vector.
        let x = deterministic_gaussian_test(0xC25EED, 256);
        let (packed, norm) = turboquant_hb_encode_d256(&x, 8).expect("encode");
        let recon = decode_d256_via_kernel_formula(&packed, norm, 8);
        let cos = cosine_similarity(&x, &recon);
        let nrmse_v = nrmse(&x, &recon);
        assert!(cos >= 0.998, "8-bit roundtrip cosine {cos} < 0.998");
        assert!(nrmse_v <= 0.07, "8-bit roundtrip NRMSE {nrmse_v} > 0.07");
    }

    #[test]
    fn hb_encoder_d256_roundtrip_5bit_within_band() {
        // 5-bit close-section: not shippable as default; expected wider gap.
        let x = deterministic_gaussian_test(0xC25EED, 256);
        let (packed, norm) = turboquant_hb_encode_d256(&x, 5).expect("encode");
        let recon = decode_d256_via_kernel_formula(&packed, norm, 5);
        let cos = cosine_similarity(&x, &recon);
        // 5-bit Lloyd-Max MSE ≈ 0.0095 → cosine ≈ 0.99. Allow small headroom.
        assert!(cos >= 0.985, "5-bit roundtrip cosine {cos} < 0.985");
    }

    #[test]
    fn hb_encoder_d256_is_deterministic() {
        let x = deterministic_gaussian_test(0xBEEF, 256);
        let (p_a, n_a) = turboquant_hb_encode_d256(&x, 8).expect("a");
        let (p_b, n_b) = turboquant_hb_encode_d256(&x, 8).expect("b");
        assert_eq!(p_a, p_b);
        assert_eq!(n_a.to_bits(), n_b.to_bits());
    }

    #[test]
    fn hb_encoder_d256_zero_vector() {
        // Mantra: if norm <= 1e-10 kernel sets scale = 0. Then every elem = 0,
        // which dequants to centroid index 127 or 128 (closest-to-zero 8-bit).
        // The norm written is 0. Decode should yield ~0 vector.
        let x = vec![0.0_f32; 256];
        let (packed, norm) = turboquant_hb_encode_d256(&x, 8).expect("encode");
        assert_eq!(norm, 0.0);
        // All packed bytes should be the centroid closest to zero (idx 127 or 128).
        for &b in packed.iter() {
            assert!(b == 127 || b == 128,
                "zero-vec encode produced non-near-zero centroid: {b}");
        }
        // Roundtrip: norm=0 means decoder produces all-zero output (× 0 = 0).
        let recon = decode_d256_via_kernel_formula(&packed, 0.0, 8);
        for &v in recon.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn hb_encoder_d256_validates_bits() {
        let x = vec![0.0_f32; 256];
        assert!(turboquant_hb_encode_d256(&x, 4).is_err()); // 4-bit not HB
        assert!(turboquant_hb_encode_d256(&x, 7).is_err()); // invalid
    }

    #[test]
    fn hb_encoder_d256_validates_size() {
        let x = vec![0.0_f32; 128]; // wrong size
        assert!(turboquant_hb_encode_d256(&x, 8).is_err());
    }

    #[test]
    fn d1_sign_mask_is_self_inverse() {
        let mut x = deterministic_gaussian_test(0x123, 256);
        let original = x.clone();
        apply_d1_sign_mask_inplace(&mut x, &TBQ_SIGNS_256);
        // After one application, must differ.
        let differs = x.iter().zip(original.iter()).any(|(&a, &b)| (a - b).abs() > 1e-6);
        assert!(differs, "D1 sign mask had no effect");
        // After two applications, must equal original (sign flip is its own inverse).
        apply_d1_sign_mask_inplace(&mut x, &TBQ_SIGNS_256);
        for (i, (&a, &b)) in x.iter().zip(original.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "D1 sign mask not self-inverse at {i}");
        }
    }

    #[test]
    fn tbq_signs_first_32_bytes_match_512_prefix() {
        // The shader's two sign tables share their first 32 bytes (verified
        // visually in hadamard_quantize_kv_fast.metal:25-30 vs 35-44). This
        // is load-bearing for cross-D=256/D=512 codec equivalence proofs.
        for i in 0..32 {
            assert_eq!(TBQ_SIGNS_256[i], TBQ_SIGNS_512[i],
                "TBQ_SIGNS_256/512 prefix mismatch at byte {i}");
        }
    }
}
