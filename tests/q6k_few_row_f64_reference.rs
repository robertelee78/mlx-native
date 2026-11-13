#![cfg(target_vendor = "apple")]
//! Q6_K few-row decode kernels against a CPU F64 reference (Class B), with
//! the Class A width and row independence they promise.
//!
//! Default routing sends a Q6_K dense product of one row to
//! `kernel_mul_mv_q6_K_f32_nr2` and of 2 to 8 rows to
//! `kernel_mul_mv_q6_K_f32_mN_r1_{2..5}` (6, 7 and 8 rows are tiled). Both
//! evaluate the same written block term, `hf2q_q6k_few_row_block_term`: per
//! 16-weight slice four lane sums of `activation × integer code` by explicit
//! fma, one fma per sub-block scale, one fma for the block scale, with
//! contraction and reassociation off.
//!
//! * F64: every output element of every width satisfies
//!   `|y − Σ x·w| ≤ 2·K·ε_f32·Σ|x·w|` against exactly dequantized weights.
//!   With unit roundoff `u = ε_f32/2`, the integer codes and the 7-bit
//!   sub-block scales are exact in F32, each fma rounds once, and a term of
//!   the dot product passes through at most four lane fmas, four scale fmas,
//!   one block fma and the F32 additions of `simd_sum` over the K/16 slices
//!   of a row: at most `(K/16 + 9)·u` relative to `Σ|x·w|` to first order,
//!   inside `K·ε_f32` for K ≥ 256 with the factor 2 left as margin.
//! * Width and row independence: column `j` of the width-`m` product is
//!   bitwise equal to the one-row product of that activation row, for every
//!   `m` in 2..=8 and every `j`.
//! * Determinism: each product repeated is bitwise equal.
//!
//! Each shape prints a `Q6K_FEW_ROW` line with the worst fraction of the
//! bound used and an FNV-1a digest of every output. The contract is about
//! the written arithmetic, so it must hold however Metal compiles the
//! kernels: run plain, under `MTL_SHADER_VALIDATION=1`, under
//! `MTL_DEBUG_LAYER=1` and with `MLX_PRECOMPILED_METALLIB=0`, and compare
//! the digests.
//!
//! ```text
//! cargo test --release --locked --test q6k_few_row_f64_reference -- --test-threads=1 --nocapture
//! ```

#![allow(clippy::expect_used, clippy::panic)]

use half::f16;
use mlx_native::gguf::test_only_dequantize;
use mlx_native::{
    quantized_matmul_ggml_with_policy_and_trace, DType, GgmlQuantizedMatmulParams,
    GgmlResolvedKernelRoute, GgmlRoutingPolicy, GgmlType, GgmlWorkloadClass, KernelRegistry,
    MlxBuffer, MlxDevice,
};

const BLOCK: usize = 210;
const QK: usize = 256;
const MAX_WIDTH: usize = 8;

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

/// `rows × k` random Q6_K blocks: random codes and sub-block scales, block
/// scale log-uniform in [1e-5, 1e-3] (one in eight negative).
fn q6k_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut bytes = vec![0u8; rows * k / QK * BLOCK];
    for block in bytes.chunks_exact_mut(BLOCK) {
        for chunk in block.chunks_mut(8) {
            let word = rng.next_u64().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        let sign = if rng.next_u64() % 8 == 0 { -1.0 } else { 1.0 };
        let d = f16::from_f64(sign * 1.0e-5 * 100f64.powf(rng.unit()));
        block[208..210].copy_from_slice(&d.to_bits().to_le_bytes());
    }
    bytes
}

/// Exact F64 weights (`d · scale · (code − 32)`: 11 + 8 + 6 significant
/// bits), cross-checked against the crate's F32 CPU dequantizer.
fn dequant_f64(bytes: &[u8], rows: usize, k: usize) -> Vec<f64> {
    let mut weights = vec![0.0f64; rows * k];
    for (index, block) in bytes.chunks_exact(BLOCK).enumerate() {
        let out = &mut weights[index * QK..(index + 1) * QK];
        let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f64();
        for half_block in 0..2 {
            let ql = &block[64 * half_block..64 * half_block + 64];
            let qh = &block[128 + 32 * half_block..128 + 32 * half_block + 32];
            let sc = &block[192 + 8 * half_block..192 + 8 * half_block + 8];
            for l in 0..32 {
                let scale = |lane: usize| d * f64::from(sc[l / 16 + 2 * lane] as i8);
                let high = |shift: u8| ((qh[l] >> shift) & 3) << 4;
                let codes = [
                    (ql[l] & 0x0f) | high(0),
                    (ql[l + 32] & 0x0f) | high(2),
                    (ql[l] >> 4) | high(4),
                    (ql[l + 32] >> 4) | high(6),
                ];
                for (lane, &code) in codes.iter().enumerate() {
                    out[128 * half_block + 32 * lane + l] = scale(lane) * (f64::from(code) - 32.0);
                }
            }
        }
    }
    let mut cpu = vec![0.0f32; rows * k];
    test_only_dequantize(bytes, GgmlType::Q6_K, &mut cpu).expect("CPU dequantize");
    for (index, (&exact, &single)) in weights.iter().zip(&cpu).enumerate() {
        assert!(
            (exact - f64::from(single)).abs() <= 4.0 * f64::from(f32::EPSILON) * exact.abs(),
            "element {index}: F64 decoder {exact:e} vs CPU dequantizer {single:e}"
        );
    }
    weights
}

struct Gpu {
    device: MlxDevice,
    registry: KernelRegistry,
}

impl Gpu {
    /// The default-routed product of `m` activation rows, as F32 bits.
    fn run(&mut self, weight: &MlxBuffer, x: &[f32], m: usize, n: usize, k: usize) -> Vec<u32> {
        let mut input = self
            .device
            .alloc_buffer(m * k * 4, DType::F32, vec![m, k])
            .expect("input buffer");
        input
            .as_mut_slice::<f32>()
            .expect("input")
            .copy_from_slice(&x[..m * k]);
        let mut output = self
            .device
            .alloc_buffer(m * n * 4, DType::F32, vec![m, n])
            .expect("output buffer");
        output
            .as_mut_slice::<u32>()
            .expect("output words")
            .fill(0x7fc0_dead);
        let mut encoder = self.device.command_encoder().expect("encoder");
        let trace = quantized_matmul_ggml_with_policy_and_trace(
            &mut encoder,
            &mut self.registry,
            &self.device,
            &input,
            weight,
            &output,
            &GgmlQuantizedMatmulParams {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                ggml_type: GgmlType::Q6_K,
            },
            &GgmlRoutingPolicy::default(),
            if m == 1 {
                GgmlWorkloadClass::DecodeSingle
            } else {
                GgmlWorkloadClass::ContinuousWidth
            },
        )
        .expect("Q6_K few-row dispatch");
        encoder.commit_and_wait().expect("command buffer");
        let expected = if m == 1 {
            "kernel_mul_mv_q6_K_f32_nr2"
        } else {
            assert_eq!(
                trace.resolved_route,
                GgmlResolvedKernelRoute::DenseQ6kWidthMn
            );
            "kernel_mul_mv_q6_K_f32_mN_r1_"
        };
        assert!(
            !trace.dispatches.is_empty()
                && trace
                    .dispatches
                    .iter()
                    .all(|dispatch| dispatch.pipeline.kernel_name.starts_with(expected)),
            "width {m}: expected {expected}*, got {:?}",
            trace
                .dispatches
                .iter()
                .map(|dispatch| dispatch.pipeline.kernel_name.as_str())
                .collect::<Vec<_>>()
        );
        output.as_slice::<u32>().expect("output words").to_vec()
    }
}

fn check_shape(gpu: &mut Gpu, n: usize, k: usize, seed: u64) {
    let bytes = q6k_bytes(n, k, seed);
    let weights = dequant_f64(&bytes, n, k);
    let mut rng = Rng(seed ^ 0x1111);
    let x: Vec<f32> = (0..MAX_WIDTH * k)
        .map(|_| ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0)
        .collect();
    let mut weight = gpu
        .device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("weight buffer");
    weight
        .as_mut_slice::<u8>()
        .expect("weight bytes")
        .copy_from_slice(&bytes);

    // F64 reference per activation row.
    let reference: Vec<Vec<(f64, f64)>> = x
        .chunks_exact(k)
        .map(|row| {
            weights
                .chunks_exact(k)
                .map(|w| {
                    row.iter().zip(w).fold((0.0, 0.0), |(dot, abs), (&a, &b)| {
                        let product = f64::from(a) * b;
                        (dot + product, abs + product.abs())
                    })
                })
                .collect()
        })
        .collect();

    // One-row authority for every activation row.
    let single: Vec<Vec<u32>> = (0..MAX_WIDTH)
        .map(|j| gpu.run(&weight, &x[j * k..(j + 1) * k], 1, n, k))
        .collect();

    let scale = 2.0 * k as f64 * f64::from(f32::EPSILON);
    let mut worst = 0.0f64;
    let mut digest = 0xcbf2_9ce4_8422_2325u64;
    for m in 1..=MAX_WIDTH {
        let bits = gpu.run(&weight, &x, m, n, k);
        let again = gpu.run(&weight, &x, m, n, k);
        assert_eq!(
            bits, again,
            "N={n} K={k} width {m}: repeat is not bitwise equal"
        );
        for j in 0..m {
            for i in 0..n {
                let word = bits[j * n + i];
                assert_eq!(
                    word, single[j][i],
                    "N={n} K={k} width {m} column {j} row {i}: {word:#010x} differs from the one-row kernel's {:#010x}",
                    single[j][i]
                );
                let (want, abs) = reference[j][i];
                let error = (f64::from(f32::from_bits(word)) - want).abs();
                let bound = scale * abs;
                assert!(
                    error <= bound,
                    "N={n} K={k} width {m} column {j} row {i}: |{} − {want}| = {error:e} > {bound:e}",
                    f32::from_bits(word)
                );
                if bound > 0.0 {
                    worst = worst.max(error / bound);
                }
                for byte in word.to_le_bytes() {
                    digest = (digest ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
    }
    eprintln!(
        "Q6K_FEW_ROW N={n} K={k} widths=1..=8: worst error/bound={worst:.4} digest={digest:016x}"
    );
}

/// Small shapes, an N that is not a multiple of the four-row threadgroup,
/// and the Gemma 4 projection width.
#[test]
fn q6k_few_row_kernels_match_f64_and_are_width_independent() {
    let mut gpu = Gpu {
        device: MlxDevice::new().expect("device"),
        registry: KernelRegistry::new(),
    };
    for (n, k, seed) in [
        (67, 256, 0x6001),
        (130, 512, 0x6002),
        (1024, 2816, 0x6003),
        (2048, 5120, 0x6004),
    ] {
        check_shape(&mut gpu, n, k, seed);
    }
}
