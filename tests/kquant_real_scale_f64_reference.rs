#![cfg(target_vendor = "apple")]
//! F64 gates for the 16-weight tile dequantizers of Q4_K, Q5_K and Q4_0 on
//! blocks whose F16 scale `d` is in the range real model files carry.
//!
//! Why a separate fixture. The tile dequantizers scale the high-nibble half
//! of a block by `d / 16`. Every other F64 gate in this crate draws `d` near
//! 1e-2, where `d / 16` is a normal F16 and the quotient is exact in any
//! precision. A census of every Q4_K / Q5_K block of four shipped GGUF files
//! (85 M to 113 M blocks each) gives `|d|` of
//!
//! | file                         | p10    | median | p90    |
//! |------------------------------|--------|--------|--------|
//! | 27B dense, Q4_K              | 5.5e-5 | 6.7e-5 | 9.3e-5 |
//! | 35B MoE, Q4_K                | 5.0e-5 | 5.9e-5 | 7.5e-5 |
//! | 26B MoE, Q4_K                | 1.3e-4 | 1.6e-4 | 2.0e-4 |
//! | 35B MoE, Q5_K                | 2.4e-5 | 2.9e-5 | 3.6e-5 |
//!
//! so `|d| / 16 < 2^-14` (the smallest normal F16) in 99.99% to 100% of the
//! blocks: evaluated in F16 the quotient is a subnormal with an absolute
//! step of 2^-24 and a relative error of 4e-3 to 1e-2 RMS. The fixture
//! draws `|d|` log-uniformly from [2e-5, 3e-4] (the census range) for the
//! K-quants. No Q4_0 file was in the census; Q4_0 has the same expression,
//! live when `|d| < 2^-10`, and the fixture draws its `|d|` log-uniformly
//! from [6e-5, 9e-4]. `dmin` (not in the census) is `d` times a uniform
//! factor in [5, 10): a 4-bit code's offset is about half its range.
//!
//! Gates, per kernel family and codec (each prints one `REALSCALE` line):
//!
//! * P — dequantizer probe (Class B, deterministic, per element). The
//!   activation rows are one-hot (1.0 in one column), so every output
//!   element is one weight exactly as the kernel staged it: no activation
//!   rounding, no accumulation error. Against the F64 value `w` of that
//!   weight, a kernel that evaluates the block affine map in F32 and stages
//!   the tile in F16 satisfies
//!   `|y − w| ≤ u_h·|w| + 2^-25 + 4·u_f·W`, and one that keeps an F32 tile
//!   `|y − w| ≤ 4·u_f·W`; `u_h = 2^-11`, `u_f = 2^-24`, 2^-25 is half the
//!   spacing of subnormal F16, and `W` bounds the magnitude of the terms of
//!   the affine map of that block (`|d|·63·q_max + |dmin|·63` for the
//!   K-quants, `23·|d|` for Q4_0; the map is one F32 product and one F32
//!   sum, at most `2·u_f·W`, and the factor 4 leaves the same again for a
//!   contracted or reordered evaluation). A scale wrong by 4e-3 moves a
//!   weight by up to 4e-3 of its magnitude, eight times `u_h`.
//! * C — scale covariance on the probe (Class A where the tile is F32).
//!   The same blocks with `d` and `dmin` multiplied by 2^8 (exact in F16)
//!   hold weights exactly 2^8 times larger, and every F32 operation of the
//!   dequantizer commutes with a power-of-two scaling. A kernel with an F32
//!   tile must therefore return bitwise `2^8·y`; one with an F16 tile must
//!   too, except where `|y| ≤ 2^-14` (staged on the subnormal F16 grid before
//!   the shift),
//!   where the two differ by at most `2^8·2^-24`.
//! * D — dense activations (uniform in [-4, 4)) against the F64 reference
//!   with the K-scaled bounds the other gates of these kernels use:
//!   `(2·u_h + u_h² + K·u_f)·Σ|x·w| + 2^-25·Σ|x|` for F16-staged kernels,
//!   `2·K·ε_f32·Σ|x·w|` for F32-tile kernels.
//! * R — relative RMS error on real-range blocks against the same blocks
//!   shifted by 2^8, F16-staged kernels only (F32-tile kernels are covered
//!   bitwise by C on the dense product too). The two products differ only
//!   by the subnormal staging of weights below 2^-14, an absolute 2^-25 per
//!   weight against weights of 1e-3 to 1e-1, so the two relative errors
//!   agree to well under a percent (they are equal to four digits on an
//!   M5 Max); the gate allows 10%. A scale computed in F16 raises the
//!   real-range error to 1.0e-3 - 2.1e-3, three to nine times the shifted
//!   blocks' 1.9e-4 - 4.2e-4, and leaves the shifted one unchanged.
//! * determinism (Class A): the dense product repeated is bitwise equal.
//!   Each line also carries an FNV-1a digest of every real-range output, to
//!   compare a plain run with runs under `MTL_SHADER_VALIDATION=1` and
//!   `MTL_DEBUG_LAYER=1`.
//!
//! D alone does not see a 4e-3 scale error (it is a worst-case bound over
//! K products); P, C and R do, at every site.
//!
//! ```text
//! cargo test --release --locked --test kquant_real_scale_f64_reference -- --test-threads=1 --nocapture
//! ```

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use half::f16;
use mlx_native::gguf::test_only_dequantize;
use mlx_native::ops::dequant_to_f16::dispatch_dequant_to_f16;
use mlx_native::{
    dispatch_mm_q4_0_tensor_64x32_for_test, quantized_matmul_ggml_column_invariant_mma,
    quantized_matmul_ggml_with_policy, quantized_matmul_id_ggml_with_policy,
    quantized_matmul_mm_tensor_perm021, DType, ExpertMmSchedule, GgmlQuantizedMatmulIdParams,
    GgmlQuantizedMatmulParams, GgmlQuantizedMatmulPerm021Params, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};

const U_H: f64 = 1.0 / 2048.0;
const U_F: f64 = 1.0 / 16_777_216.0;
/// Half the spacing of subnormal F16.
const HALF_SUBNORMAL_STEP: f64 = 1.0 / 33_554_432.0;
/// Smallest normal F16.
const F16_MIN_NORMAL: f64 = 1.0 / 16_384.0;
/// The exact power-of-two shift of the control blocks.
const SHIFT: f64 = 256.0;
/// Allowed growth of the relative RMS error from shifted to real-range blocks.
const RMS_RATIO_MAX: f64 = 1.10;
/// Columns of the probe matrix: two K-quant blocks per row.
const PROBE_K: usize = 512;
const DENSE_K: usize = 2048;

const CODECS: [GgmlType; 3] = [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q4_0];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn block_values(ggml_type: GgmlType) -> usize {
    ggml_type.block_values() as usize
}

fn block_bytes(ggml_type: GgmlType) -> usize {
    ggml_type.block_bytes() as usize
}

fn row_bytes(ggml_type: GgmlType, k: usize) -> usize {
    k / block_values(ggml_type) * block_bytes(ggml_type)
}

/// `|d|` range of the fixture (module doc).
fn scale_range(ggml_type: GgmlType) -> (f64, f64) {
    match ggml_type {
        GgmlType::Q4_K | GgmlType::Q5_K => (2.0e-5, 3.0e-4),
        GgmlType::Q4_0 => (6.0e-5, 9.0e-4),
        other => panic!("no real-range fixture for {other:?}"),
    }
}

fn read_f16(block: &[u8], offset: usize) -> f64 {
    f16::from_bits(u16::from_le_bytes([block[offset], block[offset + 1]])).to_f64()
}

/// `rows × k` random blocks whose scales follow the real-file range. One
/// block in eight has a negative `d`.
fn real_range_blocks(ggml_type: GgmlType, rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let size = block_bytes(ggml_type);
    let mut bytes = vec![0u8; rows * row_bytes(ggml_type, k)];
    let mut rng = Rng(seed);
    let (low, high) = scale_range(ggml_type);
    let mut subnormal_quotients = 0usize;
    let mut blocks = 0usize;
    for block in bytes.chunks_exact_mut(size) {
        for chunk in block.chunks_mut(8) {
            let word = rng.next_u64().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        let magnitude = low * (high / low).powf(rng.unit());
        let sign = if rng.next_u64() % 8 == 0 { -1.0 } else { 1.0 };
        let d = f16::from_f64(sign * magnitude);
        block[..2].copy_from_slice(&d.to_bits().to_le_bytes());
        if ggml_type != GgmlType::Q4_0 {
            let dmin = f16::from_f64(d.to_f64() * (5.0 + 5.0 * rng.unit()));
            block[2..4].copy_from_slice(&dmin.to_bits().to_le_bytes());
        }
        blocks += 1;
        if d.to_f64().abs() / 16.0 < F16_MIN_NORMAL {
            subnormal_quotients += 1;
        }
    }
    assert_eq!(
        subnormal_quotients, blocks,
        "{ggml_type:?}: every fixture block must have a subnormal F16 d/16, as the census found"
    );
    bytes
}

/// The same blocks with every F16 scale field multiplied by 2^8.
fn shifted_blocks(ggml_type: GgmlType, bytes: &[u8]) -> Vec<u8> {
    let offsets: &[usize] = if ggml_type == GgmlType::Q4_0 {
        &[0]
    } else {
        &[0, 2]
    };
    let mut out = bytes.to_vec();
    for block in out.chunks_exact_mut(block_bytes(ggml_type)) {
        for &offset in offsets {
            let value = read_f16(block, offset);
            let shifted = f16::from_f64(value * SHIFT);
            assert_eq!(
                shifted.to_f64(),
                value * SHIFT,
                "the shift must be exact in F16"
            );
            assert!(
                offset != 0 || shifted.to_f64().abs() / 16.0 >= F16_MIN_NORMAL,
                "the shifted control must have a normal F16 d/16"
            );
            block[offset..offset + 2].copy_from_slice(&shifted.to_bits().to_le_bytes());
        }
    }
    out
}

/// 6-bit (scale, min) pair `j` of a Q4_K / Q5_K block.
fn scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Exact F64 weights of one block (every operation below is exact in F64:
/// at most 11 + 6 + 5 significant bits per product) and the magnitude bound
/// `W` of the terms of its affine map.
fn dequant_block_f64(ggml_type: GgmlType, block: &[u8], out: &mut [f64]) -> f64 {
    let d = read_f16(block, 0);
    match ggml_type {
        GgmlType::Q4_0 => {
            let qs = &block[2..18];
            for j in 0..16 {
                out[j] = d * (f64::from(qs[j] & 0x0f) - 8.0);
                out[j + 16] = d * (f64::from(qs[j] >> 4) - 8.0);
            }
            23.0 * d.abs()
        }
        GgmlType::Q4_K | GgmlType::Q5_K => {
            let dmin = read_f16(block, 2);
            let scales = &block[4..16];
            let five_bit = ggml_type == GgmlType::Q5_K;
            let (qh, qs) = if five_bit {
                (&block[16..48], &block[48..176])
            } else {
                (&block[0..0], &block[16..144])
            };
            for group in 0..4 {
                let (sc1, m1) = scale_min_k4(2 * group, scales);
                let (sc2, m2) = scale_min_k4(2 * group + 1, scales);
                let (d1, min1) = (d * f64::from(sc1), dmin * f64::from(m1));
                let (d2, min2) = (d * f64::from(sc2), dmin * f64::from(m2));
                for l in 0..32 {
                    let byte = qs[group * 32 + l];
                    let (high1, high2) = if five_bit {
                        (
                            if qh[l] & (1 << (2 * group)) != 0 {
                                16.0
                            } else {
                                0.0
                            },
                            if qh[l] & (2 << (2 * group)) != 0 {
                                16.0
                            } else {
                                0.0
                            },
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    out[group * 64 + l] = d1 * (f64::from(byte & 0x0f) + high1) - min1;
                    out[group * 64 + 32 + l] = d2 * (f64::from(byte >> 4) + high2) - min2;
                }
            }
            let q_max = if five_bit { 31.0 } else { 15.0 };
            d.abs() * 63.0 * q_max + dmin.abs() * 63.0
        }
        other => panic!("no F64 dequantizer for {other:?}"),
    }
}

/// Exact weights and per-element term-magnitude bounds of `rows × k` blocks.
/// Cross-checks the decoder against the crate's F32 CPU dequantizer.
fn dequant_f64(ggml_type: GgmlType, bytes: &[u8], rows: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let values = block_values(ggml_type);
    let mut weights = vec![0.0f64; rows * k];
    let mut terms = vec![0.0f64; rows * k];
    for (index, block) in bytes.chunks_exact(block_bytes(ggml_type)).enumerate() {
        let range = index * values..(index + 1) * values;
        let w = dequant_block_f64(ggml_type, block, &mut weights[range.clone()]);
        terms[range].fill(w);
    }
    let mut cpu = vec![0.0f32; rows * k];
    test_only_dequantize(bytes, ggml_type, &mut cpu).expect("CPU dequantize");
    for (index, (&exact, &single)) in weights.iter().zip(&cpu).enumerate() {
        assert!(
            (exact - f64::from(single)).abs() <= 4.0 * U_F * terms[index],
            "{ggml_type:?} element {index}: F64 decoder {exact:e} vs CPU dequantizer {single:e}"
        );
    }
    (weights, terms)
}

fn dense_inputs(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..len)
        .map(|_| ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0)
        .collect()
}

fn u8_buffer(device: &MlxDevice, bytes: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("u8 buffer");
    buffer
        .as_mut_slice::<u8>()
        .expect("u8")
        .copy_from_slice(bytes);
    buffer
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32")
        .copy_from_slice(values);
    buffer
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("u32 buffer");
    buffer
        .as_mut_slice::<u32>()
        .expect("u32")
        .copy_from_slice(values);
    buffer
}

/// Output buffer pre-filled with a NaN sentinel so unwritten elements fail.
fn output_buffer(device: &MlxDevice, len: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("output buffer");
    buffer
        .as_mut_slice::<u32>()
        .expect("words")
        .fill(0x7fc0_dead);
    buffer
}

fn tensor_available(device: &MlxDevice) -> bool {
    let mut registry = KernelRegistry::new();
    let available = registry
        .get_pipeline("kernel_mul_mm_q4_K_tensor_f32", device.metal_device())
        .is_ok()
        && registry
            .get_pipeline("kernel_mul_mm_id_q4_K_tensor_f32", device.metal_device())
            .is_ok();
    if !available {
        assert!(
            std::env::var("MLX_MM_ID_TEST_ALLOW_NO_TENSOR").as_deref() == Ok("1"),
            "the tensor kernels are unavailable on {}: set MLX_MM_ID_TEST_ALLOW_NO_TENSOR=1 to skip them",
            device.name()
        );
        eprintln!("RECORDED SKIP: tensor pipeline unavailable; tensor kernels not gated");
    }
    available
}

/// Whether `kernel` resolved, with or without the dense MM function-constant
/// specialisation suffix.
fn ran(registry: &KernelRegistry, kernel: &str) -> bool {
    registry.is_cached(kernel) || registry.is_cached(&format!("{kernel}|700:i1|701:i1|702:i1"))
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

/// Precision of the dequantized tile a kernel multiplies with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tile {
    F16,
    F32,
}

struct Report {
    /// FNV-1a over the bits of every real-range output, for comparing runs
    /// under the Metal validation layers.
    digest: u64,
    probe_elements: usize,
    probe_violations: usize,
    probe_worst: f64,
    covariance_violations: usize,
    dense_worst: f64,
    dense_violations: usize,
    rms_real: f64,
    rms_shifted: f64,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            digest: 0xcbf2_9ce4_8422_2325,
            probe_elements: 0,
            probe_violations: 0,
            probe_worst: 0.0,
            covariance_violations: 0,
            dense_worst: 0.0,
            dense_violations: 0,
            rms_real: 0.0,
            rms_shifted: 0.0,
        }
    }
}

impl Report {
    fn hash(&mut self, value: f32) {
        for byte in value.to_bits().to_le_bytes() {
            self.digest = (self.digest ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// P and C for one probed weight: `real` and `shifted` are the kernel's
    /// outputs for the real-range block and its 2^8 control.
    fn probe(&mut self, tile: Tile, real: f32, shifted: f32, weight: f64, terms: f64) {
        let bound = match tile {
            Tile::F16 => U_H * weight.abs() + HALF_SUBNORMAL_STEP + 4.0 * U_F * terms,
            Tile::F32 => 4.0 * U_F * terms,
        };
        let fraction = (f64::from(real) - weight).abs() / bound;
        self.hash(real);
        self.probe_elements += 1;
        self.probe_worst = self.probe_worst.max(fraction);
        if !(fraction <= 1.0) {
            self.probe_violations += 1;
        }
        let expected = f64::from(real) * SHIFT;
        let covariant = if tile == Tile::F16 && f64::from(real).abs() <= F16_MIN_NORMAL {
            (f64::from(shifted) - expected).abs() <= SHIFT * 2.0 * HALF_SUBNORMAL_STEP
        } else {
            f64::from(shifted) == expected
        };
        if !covariant {
            self.covariance_violations += 1;
        }
    }

    /// D, C (F32 tiles) and the sums for R over one dense product.
    /// `reference[i] = (Σ x·w, Σ|x·w|)` for the real-range weights.
    fn dense(
        &mut self,
        tile: Tile,
        k: usize,
        real: &[f32],
        shifted: &[f32],
        reference: &[(f64, f64)],
        abs_x: &[f64],
        n: usize,
    ) {
        let (mut err_real, mut err_shifted, mut norm) = (0.0f64, 0.0f64, 0.0f64);
        for (index, &(want, abs)) in reference.iter().enumerate() {
            let bound = match tile {
                Tile::F16 => {
                    (2.0 * U_H + U_H * U_H + k as f64 * U_F) * abs
                        + HALF_SUBNORMAL_STEP * abs_x[index / n]
                }
                Tile::F32 => 2.0 * k as f64 * f64::from(f32::EPSILON) * abs,
            };
            let error = (f64::from(real[index]) - want).abs();
            self.hash(real[index]);
            self.dense_worst = self.dense_worst.max(error / bound);
            if !(error <= bound) {
                self.dense_violations += 1;
            }
            if tile == Tile::F32 && f64::from(shifted[index]) != f64::from(real[index]) * SHIFT {
                self.covariance_violations += 1;
            }
            err_real += error * error;
            err_shifted += (f64::from(shifted[index]) - want * SHIFT).powi(2);
            norm += want * want;
        }
        self.rms_real = self.rms_real.max((err_real / norm).sqrt());
        self.rms_shifted = self.rms_shifted.max((err_shifted / norm).sqrt() / SHIFT);
    }

    /// Prints the `REALSCALE` line and returns the failed gates.
    fn finish(&self, label: &str, tile: Tile) -> Vec<String> {
        eprintln!(
            "REALSCALE {label} tile={tile:?}: probe elements={} worst/bound={:.3} violations={} covariance_violations={} dense worst/bound={:.3} violations={} rel_rms real={:.3e} shifted={:.3e} digest={:016x}",
            self.probe_elements,
            self.probe_worst,
            self.probe_violations,
            self.covariance_violations,
            self.dense_worst,
            self.dense_violations,
            self.rms_real,
            self.rms_shifted,
            self.digest,
        );
        let mut failed = Vec::new();
        if self.probe_violations > 0 {
            failed.push(format!(
                "{label}: P {} of {} probed weights outside the F64 bound (worst {:.1}x)",
                self.probe_violations, self.probe_elements, self.probe_worst
            ));
        }
        if self.covariance_violations > 0 {
            failed.push(format!(
                "{label}: C {} outputs are not 2^8 times the real-range output",
                self.covariance_violations
            ));
        }
        if self.dense_violations > 0 {
            failed.push(format!(
                "{label}: D {} outputs outside the K-scaled F64 bound (worst {:.2}x)",
                self.dense_violations, self.dense_worst
            ));
        }
        if tile == Tile::F16 && !(self.rms_real <= RMS_RATIO_MAX * self.rms_shifted) {
            failed.push(format!(
                "{label}: R relative RMS error {:.3e} on real-range blocks exceeds {RMS_RATIO_MAX} x {:.3e} on the shifted blocks",
                self.rms_real, self.rms_shifted
            ));
        }
        failed
    }
}

/// One-hot probe rows: row `r` of dispatch `step` selects column
/// `step·m + r`; rows past K repeat the last column.
fn probe_rows(m: usize, k: usize, step: usize) -> (Vec<f32>, Vec<usize>) {
    let mut rows = vec![0.0f32; m * k];
    let mut columns = Vec::with_capacity(m);
    for r in 0..m {
        let column = (step * m + r).min(k - 1);
        rows[r * k + column] = 1.0;
        columns.push(column);
    }
    (rows, columns)
}

fn reference_products(x: &[f32], weights: &[f64], m: usize, n: usize, k: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(m * n);
    for row in x.chunks_exact(k).take(m) {
        for w in weights.chunks_exact(k).take(n) {
            let (mut dot, mut abs) = (0.0f64, 0.0f64);
            for (&a, &b) in row.iter().zip(w) {
                let product = f64::from(a) * b;
                dot += product;
                abs += product.abs();
            }
            out.push((dot, abs));
        }
    }
    out
}

fn abs_row_sums(x: &[f32], k: usize) -> Vec<f64> {
    x.chunks_exact(k)
        .map(|row| row.iter().map(|&v| f64::from(v).abs()).sum())
        .collect()
}

/// A dense `[m, K] × [N, K]ᵀ` entry under test.
type DenseEntry<'a> = dyn Fn(
        &mut KernelRegistry,
        &MlxDevice,
        &MlxBuffer,
        &MlxBuffer,
        &MlxBuffer,
        GgmlQuantizedMatmulParams,
    ) + 'a;

fn run_dense(
    entry: &DenseEntry<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    x: &[f32],
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let input = f32_buffer(device, x);
    let output = output_buffer(device, m * n);
    entry(
        registry,
        device,
        &input,
        weight,
        &output,
        GgmlQuantizedMatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type,
        },
    );
    output.as_slice::<f32>().expect("output").to_vec()
}

/// Runs P, C, D, R and determinism for one dense entry, codec, N and width.
fn dense_case(
    label: &str,
    entry: &DenseEntry<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    ggml_type: GgmlType,
    tile: Tile,
    m: usize,
    n: usize,
    dense_k: usize,
    seed: u64,
) -> Vec<String> {
    let mut report = Report::default();

    // P and C.
    let k = PROBE_K;
    let real = real_range_blocks(ggml_type, n, k, seed);
    let shifted = shifted_blocks(ggml_type, &real);
    let (weights, terms) = dequant_f64(ggml_type, &real, n, k);
    let (real_buffer, shifted_buffer) = (u8_buffer(device, &real), u8_buffer(device, &shifted));
    for step in 0..k.div_ceil(m) {
        let (rows, columns) = probe_rows(m, k, step);
        let y = run_dense(
            entry,
            registry,
            device,
            &rows,
            &real_buffer,
            ggml_type,
            m,
            n,
            k,
        );
        let y_shifted = run_dense(
            entry,
            registry,
            device,
            &rows,
            &shifted_buffer,
            ggml_type,
            m,
            n,
            k,
        );
        for (r, &column) in columns.iter().enumerate() {
            for i in 0..n {
                report.probe(
                    tile,
                    y[r * n + i],
                    y_shifted[r * n + i],
                    weights[i * k + column],
                    terms[i * k + column],
                );
            }
        }
    }

    // D, R and determinism.
    let k = dense_k;
    let real = real_range_blocks(ggml_type, n, k, seed ^ 0xd5d5);
    let shifted = shifted_blocks(ggml_type, &real);
    let (weights, _) = dequant_f64(ggml_type, &real, n, k);
    let x = dense_inputs(m * k, seed ^ 0x1111);
    let (real_buffer, shifted_buffer) = (u8_buffer(device, &real), u8_buffer(device, &shifted));
    let y = run_dense(
        entry,
        registry,
        device,
        &x,
        &real_buffer,
        ggml_type,
        m,
        n,
        k,
    );
    let again = run_dense(
        entry,
        registry,
        device,
        &x,
        &real_buffer,
        ggml_type,
        m,
        n,
        k,
    );
    let y_shifted = run_dense(
        entry,
        registry,
        device,
        &x,
        &shifted_buffer,
        ggml_type,
        m,
        n,
        k,
    );
    let reference = reference_products(&x, &weights, m, n, k);
    report.dense(tile, k, &y, &y_shifted, &reference, &abs_row_sums(&x, k), n);

    let label = format!("{label} {ggml_type:?} m={m} N={n}");
    let mut failed = report.finish(&label, tile);
    if y.iter()
        .zip(&again)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        failed.push(format!(
            "{label}: the repeated dense product is not bitwise equal"
        ));
    }
    failed
}

fn policy_entry(
    policy: GgmlRoutingPolicy,
) -> impl Fn(
    &mut KernelRegistry,
    &MlxDevice,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    GgmlQuantizedMatmulParams,
) {
    move |registry, device, input, weight, output, params| {
        let mut encoder = device.command_encoder().expect("encoder");
        quantized_matmul_ggml_with_policy(
            &mut encoder,
            registry,
            device,
            input,
            weight,
            output,
            &params,
            &policy,
        )
        .expect("dense dispatch");
        encoder.commit_and_wait().expect("command buffer");
    }
}

fn assert_no_failures(failed: Vec<String>) {
    assert!(
        failed.is_empty(),
        "{} gate failures:\n{}",
        failed.len(),
        failed.join("\n")
    );
}

fn codec_tag(ggml_type: GgmlType) -> &'static str {
    match ggml_type {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q5_K => "q5_K",
        GgmlType::Q4_0 => "q4_0",
        other => panic!("no kernel tag for {other:?}"),
    }
}

/// `kernel_mul_mm_<q>_f32`: the simdgroup matrix kernel (devices without the
/// tensor pipeline, and `HF2Q_DISABLE_TENSOR_MM=1`).
#[test]
fn real_scale_dense_mm_simdgroup() {
    let device = MlxDevice::new().expect("device");
    let entry = policy_entry(GgmlRoutingPolicy {
        dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
        ..GgmlRoutingPolicy::default()
    });
    let mut failed = Vec::new();
    for ggml_type in CODECS {
        let mut registry = KernelRegistry::new();
        for (m, n) in [(9, 67), (33, 70)] {
            failed.extend(dense_case(
                "mm simdgroup",
                &entry,
                &mut registry,
                &device,
                ggml_type,
                Tile::F16,
                m,
                n,
                DENSE_K,
                0x51d0 + m as u64,
            ));
        }
        let kernel = format!("kernel_mul_mm_{}_f32", codec_tag(ggml_type));
        assert!(ran(&registry, &kernel), "{kernel} did not run");
    }
    assert_no_failures(failed);
}

/// `kernel_mul_mm_<q>_tensor_f32`: the 32-row tensor tile.
#[test]
fn real_scale_dense_mm_tensor_v1() {
    let device = MlxDevice::new().expect("device");
    if !tensor_available(&device) {
        return;
    }
    let entry = policy_entry(GgmlRoutingPolicy {
        allow_dense_large_tile_mm: false,
        ..GgmlRoutingPolicy::default()
    });
    let mut failed = Vec::new();
    for ggml_type in CODECS {
        let mut registry = KernelRegistry::new();
        for (m, n) in [(32, 67), (41, 130)] {
            failed.extend(dense_case(
                "mm tensor v1",
                &entry,
                &mut registry,
                &device,
                ggml_type,
                Tile::F16,
                m,
                n,
                DENSE_K,
                0x7e51 + m as u64,
            ));
        }
        let kernel = format!("kernel_mul_mm_{}_tensor_f32", codec_tag(ggml_type));
        assert!(ran(&registry, &kernel), "{kernel} did not run");
    }
    assert_no_failures(failed);
}

/// `kernel_mul_mm_<q>_tensor_v2_f32` (128-row tile) and the Q4_0 64x32
/// candidate tile.
#[test]
fn real_scale_dense_mm_tensor_v2() {
    let device = MlxDevice::new().expect("device");
    if !tensor_available(&device) {
        return;
    }
    let entry = policy_entry(GgmlRoutingPolicy::default());
    let mut failed = Vec::new();
    for ggml_type in CODECS {
        let mut registry = KernelRegistry::new();
        for (m, n) in [(33, 67), (160, 130)] {
            failed.extend(dense_case(
                "mm tensor v2",
                &entry,
                &mut registry,
                &device,
                ggml_type,
                Tile::F16,
                m,
                n,
                DENSE_K,
                0x7e52 + m as u64,
            ));
        }
        let kernel = format!("kernel_mul_mm_{}_tensor_v2_f32", codec_tag(ggml_type));
        assert!(ran(&registry, &kernel), "{kernel} did not run");
    }
    let short_tile = |registry: &mut KernelRegistry,
                      device: &MlxDevice,
                      input: &MlxBuffer,
                      weight: &MlxBuffer,
                      output: &MlxBuffer,
                      params: GgmlQuantizedMatmulParams| {
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_mm_q4_0_tensor_64x32_for_test(
            &mut encoder,
            registry,
            device,
            input,
            weight,
            output,
            &params,
        )
        .expect("64x32 dispatch");
        encoder.commit_and_wait().expect("command buffer");
    };
    let mut registry = KernelRegistry::new();
    failed.extend(dense_case(
        "mm tensor 64x32",
        &short_tile,
        &mut registry,
        &device,
        GgmlType::Q4_0,
        Tile::F16,
        33,
        67,
        DENSE_K,
        0x6432,
    ));
    assert!(
        ran(&registry, "kernel_mul_mm_q4_0_tensor_64x32_f32"),
        "kernel_mul_mm_q4_0_tensor_64x32_f32 did not run"
    );
    assert_no_failures(failed);
}

/// `kernel_mul_mv_ext_<q>_f32_r1_*`: the few-row kernels with an F32 tile.
#[test]
fn real_scale_mul_mv_ext() {
    let device = MlxDevice::new().expect("device");
    let entry = policy_entry(GgmlRoutingPolicy {
        dense_decode_mvn: false,
        dense_decode_mv_ext: true,
        ..GgmlRoutingPolicy::default()
    });
    let mut failed = Vec::new();
    // Width selects the columns per threadgroup (1 -> r1_1, 2 -> r1_2,
    // 3 and 6 -> r1_3, 4, 7 and 8 -> r1_4, 5 -> r1_5); the pipelines are
    // specialised per shape, so the registry cannot be asked by name.
    let cases: [(GgmlType, &[usize]); 3] = [
        (GgmlType::Q4_K, &[4, 5, 6, 8]),
        (GgmlType::Q5_K, &[1, 2, 3, 4, 5, 8]),
        (GgmlType::Q4_0, &[2, 3, 4, 5, 8]),
    ];
    let mut registry = KernelRegistry::new();
    for (ggml_type, widths) in cases {
        for &m in widths {
            failed.extend(dense_case(
                "mul_mv_ext",
                &entry,
                &mut registry,
                &device,
                ggml_type,
                Tile::F32,
                m,
                67,
                DENSE_K,
                0xe870 + m as u64,
            ));
        }
    }
    assert_no_failures(failed);
}

/// `kernel_mul_mv_mma_q4_K_f32_nt{1,2,4}_rt1`: the column-invariant few-row
/// kernel with an F32 tile.
#[test]
fn real_scale_mul_mv_mma() {
    let device = MlxDevice::new().expect("device");
    let entry = |registry: &mut KernelRegistry,
                 device: &MlxDevice,
                 input: &MlxBuffer,
                 weight: &MlxBuffer,
                 output: &MlxBuffer,
                 params: GgmlQuantizedMatmulParams| {
        let mut encoder = device.command_encoder().expect("encoder");
        quantized_matmul_ggml_column_invariant_mma(
            &mut encoder,
            registry,
            device,
            input,
            weight,
            output,
            &params,
        )
        .expect("mma dispatch");
        encoder.commit_and_wait().expect("command buffer");
    };
    let mut failed = Vec::new();
    let mut registry = KernelRegistry::new();
    // N selects the 8-row tiles per threadgroup (nt1 below 2048 rows, nt2
    // below 4096, nt4 from there); the pipelines are specialised per shape,
    // so the registry cannot be asked by name.
    for (m, n, dense_k) in [(1, 70, DENSE_K), (4, 2112, 1024), (8, 4160, 1024)] {
        failed.extend(dense_case(
            "mul_mv_mma",
            &entry,
            &mut registry,
            &device,
            GgmlType::Q4_K,
            Tile::F32,
            m,
            n,
            dense_k,
            0x33a0 + m as u64,
        ));
    }
    assert_no_failures(failed);
}

// ---------------------------------------------------------------------------
// Expert `mm_id`
// ---------------------------------------------------------------------------

const EXPERTS: usize = 9;
const TOP_K: usize = 6;
const EXPERT_N: usize = 70;

fn route_ids(n_tokens: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed);
    let mut ids = Vec::with_capacity(n_tokens * TOP_K);
    for _ in 0..n_tokens {
        let start = ids.len();
        while ids.len() - start < TOP_K {
            let expert = (rng.next_u64() % EXPERTS as u64) as u32;
            if !ids[start..].contains(&expert) {
                ids.push(expert);
            }
        }
    }
    ids
}

fn run_expert(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    policy: &GgmlRoutingPolicy,
    x: &[f32],
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    ggml_type: GgmlType,
    n_tokens: usize,
    k: usize,
) -> Vec<f32> {
    let input = f32_buffer(device, x);
    let output = output_buffer(device, n_tokens * TOP_K * EXPERT_N);
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_id_ggml_with_policy(
        &mut encoder,
        registry,
        device,
        &input,
        weight,
        ids,
        &output,
        &GgmlQuantizedMatmulIdParams {
            n_tokens: n_tokens as u32,
            top_k: TOP_K as u32,
            n: EXPERT_N as u32,
            k: k as u32,
            n_experts: EXPERTS as u32,
            expert_stride: (EXPERT_N * row_bytes(ggml_type, k)) as u64,
            ggml_type,
        },
        policy,
    )
    .expect("mm_id dispatch");
    encoder.commit_and_wait().expect("command buffer");
    output.as_slice::<f32>().expect("output").to_vec()
}

fn expert_case(
    label: &str,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    policy: &GgmlRoutingPolicy,
    ggml_type: GgmlType,
    n_tokens: usize,
    seed: u64,
) -> Vec<String> {
    let mut report = Report::default();
    let n = EXPERT_N;
    let ids = route_ids(n_tokens, seed ^ 0x1d5);
    let ids_buffer = u32_buffer(device, &ids);

    // P and C.
    let k = PROBE_K;
    let real = real_range_blocks(ggml_type, EXPERTS * n, k, seed);
    let shifted = shifted_blocks(ggml_type, &real);
    let (weights, terms) = dequant_f64(ggml_type, &real, EXPERTS * n, k);
    let (real_buffer, shifted_buffer) = (u8_buffer(device, &real), u8_buffer(device, &shifted));
    for step in 0..k.div_ceil(n_tokens) {
        let (rows, columns) = probe_rows(n_tokens, k, step);
        let y = run_expert(
            registry,
            device,
            policy,
            &rows,
            &real_buffer,
            &ids_buffer,
            ggml_type,
            n_tokens,
            k,
        );
        let y_shifted = run_expert(
            registry,
            device,
            policy,
            &rows,
            &shifted_buffer,
            &ids_buffer,
            ggml_type,
            n_tokens,
            k,
        );
        for (t, &column) in columns.iter().enumerate() {
            for slot in 0..TOP_K {
                let expert = ids[t * TOP_K + slot] as usize;
                for i in 0..n {
                    let at = (expert * n + i) * k + column;
                    let out = (t * TOP_K + slot) * n + i;
                    report.probe(Tile::F16, y[out], y_shifted[out], weights[at], terms[at]);
                }
            }
        }
    }

    // D, R and determinism.
    let k = 1024;
    let real = real_range_blocks(ggml_type, EXPERTS * n, k, seed ^ 0xd5d5);
    let shifted = shifted_blocks(ggml_type, &real);
    let (weights, _) = dequant_f64(ggml_type, &real, EXPERTS * n, k);
    let x = dense_inputs(n_tokens * k, seed ^ 0x1111);
    let (real_buffer, shifted_buffer) = (u8_buffer(device, &real), u8_buffer(device, &shifted));
    let y = run_expert(
        registry,
        device,
        policy,
        &x,
        &real_buffer,
        &ids_buffer,
        ggml_type,
        n_tokens,
        k,
    );
    let again = run_expert(
        registry,
        device,
        policy,
        &x,
        &real_buffer,
        &ids_buffer,
        ggml_type,
        n_tokens,
        k,
    );
    let y_shifted = run_expert(
        registry,
        device,
        policy,
        &x,
        &shifted_buffer,
        &ids_buffer,
        ggml_type,
        n_tokens,
        k,
    );
    let mut reference = Vec::with_capacity(n_tokens * TOP_K * n);
    for t in 0..n_tokens {
        for slot in 0..TOP_K {
            let expert = ids[t * TOP_K + slot] as usize;
            reference.extend(reference_products(
                &x[t * k..(t + 1) * k],
                &weights[expert * n * k..(expert + 1) * n * k],
                1,
                n,
                k,
            ));
        }
    }
    let abs_x: Vec<f64> = abs_row_sums(&x, k)
        .into_iter()
        .flat_map(|sum| std::iter::repeat(sum).take(TOP_K))
        .collect();
    report.dense(Tile::F16, k, &y, &y_shifted, &reference, &abs_x, n);

    let label = format!("{label} {ggml_type:?} tokens={n_tokens} top_k={TOP_K} N={n}");
    let mut failed = report.finish(&label, Tile::F16);
    if y.iter()
        .zip(&again)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        failed.push(format!(
            "{label}: the repeated product is not bitwise equal"
        ));
    }
    failed
}

fn expert_family(tensor: bool) {
    let device = MlxDevice::new().expect("device");
    if tensor && !tensor_available(&device) {
        return;
    }
    let mut failed = Vec::new();
    for ggml_type in CODECS {
        for (schedule, suffix) in [
            (ExpertMmSchedule::CompactTiles, "_c"),
            (ExpertMmSchedule::UniformGrid, ""),
        ] {
            let policy = GgmlRoutingPolicy {
                expert_tensor_mm: if tensor {
                    GgmlTensorMmPreference::AutoProbe
                } else {
                    GgmlTensorMmPreference::ForceSimd
                },
                expert_mm_schedule: schedule,
                ..GgmlRoutingPolicy::default()
            };
            let kernel = format!(
                "kernel_mul_mm_id_{}{}_f32{suffix}",
                codec_tag(ggml_type),
                if tensor { "_tensor" } else { "" }
            );
            let mut registry = KernelRegistry::new();
            for n_tokens in [40, 72] {
                failed.extend(expert_case(
                    &kernel,
                    &mut registry,
                    &device,
                    &policy,
                    ggml_type,
                    n_tokens,
                    0x1d00 + n_tokens as u64,
                ));
            }
            assert!(ran(&registry, &kernel), "{kernel} did not run");
        }
    }
    assert_no_failures(failed);
}

/// `kernel_mul_mm_id_<q>_tensor_f32` and `_c`: the default expert matrix
/// kernels above 32 rows.
#[test]
fn real_scale_mm_id_tensor() {
    expert_family(true);
}

/// `kernel_mul_mm_id_<q>_f32` and `_c`: the simdgroup expert kernels.
#[test]
fn real_scale_mm_id_simdgroup() {
    expert_family(false);
}

// ---------------------------------------------------------------------------
// Q4_0 only: BF16 head-major tensor kernel and the F16 shadow
// ---------------------------------------------------------------------------

/// `kernel_mul_mm_q4_0_tensor_bf16_perm021`: activations are BF16 at
/// `[n_heads, m, head_dim]`. The reference uses the BF16-rounded values, so
/// the bounds are those of an F16-staged kernel.
#[cfg(mlx_native_has_metal_tensor_artifact)]
#[test]
fn real_scale_q4_0_tensor_bf16_perm021() {
    const HEAD_DIM: usize = 64;
    let device = MlxDevice::new().expect("device");
    if !tensor_available(&device) {
        return;
    }
    let bf16_input = |device: &MlxDevice, logical: &[f32], m: usize, k: usize| {
        let heads = k / HEAD_DIM;
        let mut buffer = device
            .alloc_buffer(m * k * 2, DType::BF16, vec![heads, m, HEAD_DIM])
            .expect("bf16 buffer");
        let stored = buffer.as_mut_slice::<u16>().expect("bf16 words");
        let mut rounded = vec![0.0f32; m * k];
        for head in 0..heads {
            for token in 0..m {
                for column in 0..HEAD_DIM {
                    let source = token * k + head * HEAD_DIM + column;
                    let bits = half::bf16::from_f32(logical[source]).to_bits();
                    stored[(head * m + token) * HEAD_DIM + column] = bits;
                    rounded[source] = half::bf16::from_bits(bits).to_f32();
                }
            }
        }
        (buffer, rounded)
    };
    let mut registry = KernelRegistry::new();
    let mut run = |x: &[f32], weight: &MlxBuffer, m: usize, n: usize, k: usize| {
        let (input, rounded) = bf16_input(&device, x, m, k);
        let output = output_buffer(&device, m * n);
        let mut encoder = device.command_encoder().expect("encoder");
        quantized_matmul_mm_tensor_perm021(
            &mut encoder,
            &mut registry,
            &device,
            &input,
            weight,
            &output,
            &GgmlQuantizedMatmulPerm021Params {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                head_dim: HEAD_DIM as u32,
                ggml_type: GgmlType::Q4_0,
            },
        )
        .expect("perm021 dispatch");
        encoder.commit_and_wait().expect("command buffer");
        (output.as_slice::<f32>().expect("output").to_vec(), rounded)
    };
    let ggml_type = GgmlType::Q4_0;
    let (m, n) = (33, 67);
    let mut report = Report::default();

    let k = PROBE_K;
    let real = real_range_blocks(ggml_type, n, k, 0x0210);
    let shifted = shifted_blocks(ggml_type, &real);
    let (weights, terms) = dequant_f64(ggml_type, &real, n, k);
    let (real_buffer, shifted_buffer) = (u8_buffer(&device, &real), u8_buffer(&device, &shifted));
    for step in 0..k.div_ceil(m) {
        let (rows, columns) = probe_rows(m, k, step);
        let (y, _) = run(&rows, &real_buffer, m, n, k);
        let (y_shifted, _) = run(&rows, &shifted_buffer, m, n, k);
        for (r, &column) in columns.iter().enumerate() {
            for i in 0..n {
                let at = i * k + column;
                report.probe(
                    Tile::F16,
                    y[r * n + i],
                    y_shifted[r * n + i],
                    weights[at],
                    terms[at],
                );
            }
        }
    }

    let k = DENSE_K;
    let real = real_range_blocks(ggml_type, n, k, 0x0211);
    let shifted = shifted_blocks(ggml_type, &real);
    let (weights, _) = dequant_f64(ggml_type, &real, n, k);
    let x = dense_inputs(m * k, 0x0212);
    let (real_buffer, shifted_buffer) = (u8_buffer(&device, &real), u8_buffer(&device, &shifted));
    let (y, rounded) = run(&x, &real_buffer, m, n, k);
    let (again, _) = run(&x, &real_buffer, m, n, k);
    let (y_shifted, _) = run(&x, &shifted_buffer, m, n, k);
    let reference = reference_products(&rounded, &weights, m, n, k);
    report.dense(
        Tile::F16,
        k,
        &y,
        &y_shifted,
        &reference,
        &abs_row_sums(&rounded, k),
        n,
    );

    let label = format!("mm tensor bf16 perm021 Q4_0 m={m} N={n}");
    let mut failed = report.finish(&label, Tile::F16);
    if y.iter()
        .zip(&again)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        failed.push(format!(
            "{label}: the repeated product is not bitwise equal"
        ));
    }
    assert_no_failures(failed);
}

/// `hf2q_dequant_q4_0_to_f16`: every weight of the F16 shadow against F64
/// (gate P on the stored value) and against the shifted control (gate C).
#[test]
fn real_scale_dequant_q4_0_to_f16() {
    let device = MlxDevice::new().expect("device");
    let ggml_type = GgmlType::Q4_0;
    let (n, k) = (67, 512);
    let real = real_range_blocks(ggml_type, n, k, 0xf16);
    let shifted = shifted_blocks(ggml_type, &real);
    let (weights, terms) = dequant_f64(ggml_type, &real, n, k);
    let mut registry = KernelRegistry::new();
    let mut shadow = |bytes: &[u8]| {
        let weight = u8_buffer(&device, bytes);
        let mut output = device
            .alloc_buffer(n * k * 2, DType::F16, vec![n, k])
            .expect("shadow buffer");
        output.as_mut_slice::<u16>().expect("words").fill(0x7e00);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_dequant_to_f16(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &weight,
            &output,
            n as u32,
            k as u32,
            ggml_type,
        )
        .expect("dequant dispatch");
        encoder.commit_and_wait().expect("command buffer");
        output
            .as_slice::<u16>()
            .expect("words")
            .iter()
            .map(|&bits| f16::from_bits(bits).to_f32())
            .collect::<Vec<f32>>()
    };
    let y = shadow(&real);
    let y_shifted = shadow(&shifted);
    let mut report = Report::default();
    for index in 0..n * k {
        report.probe(
            Tile::F16,
            y[index],
            y_shifted[index],
            weights[index],
            terms[index],
        );
    }
    let label = format!("dequant_to_f16 Q4_0 rows={n} K={k}");
    eprintln!(
        "REALSCALE {label}: probe elements={} worst/bound={:.3} violations={} covariance_violations={} digest={:016x}",
        report.probe_elements,
        report.probe_worst,
        report.probe_violations,
        report.covariance_violations,
        report.digest
    );
    assert!(
        report.probe_violations == 0 && report.covariance_violations == 0,
        "{label}: {} of {} weights outside the F64 bound (worst {:.1}x), {} not 2^8-covariant",
        report.probe_violations,
        report.probe_elements,
        report.probe_worst,
        report.covariance_violations
    );
}
