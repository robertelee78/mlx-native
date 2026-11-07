#![cfg(target_vendor = "apple")]
//! Q5_0 tensor-MM qualification against a CPU F64 reference (Class B), with
//! the Class A determinism and row-invariance gates of the tensor route.
//!
//! Every arm runs on the same bytes and activations; the Q8_0 arms are the
//! control: each Q5_0 block is re-encoded losslessly as a Q8_0 block with the
//! same F16 scale and `q - 16` as its int8 payload, so both codecs carry
//! bit-identical dequantized weights and only the dequant code differs. Each
//! Q5_0 kernel must equal its Q8_0 counterpart bit for bit: a Q5_0 dequant or
//! staging defect cannot hide behind the tolerance.
//!
//! Bound. For an output `y = sum_k x_k w_k` with `S = sum_k |x_k w_k|`, a
//! kernel that rounds both operands to F16 (unit roundoff `u_h = 2^-11`) and
//! accumulates in F32 (`u_f = 2^-24`) satisfies, to first order,
//! `|y_gpu - y_f64| <= (2 u_h + u_h^2 + K u_f) * S`. The simdgroup kernels
//! round only the weight operand, so the same bound covers them. Every
//! element of every arm must stay inside it; the report prints the maximum
//! fraction of the bound used.
//!
//! * R0 — the fixture and widths (N = 17, K = 64, M ∈ {9, 33}) whose fixed
//!   `2e-3 * max(|y|, 1)` F32 bound excluded Q5_0 from the tensor route; the
//!   report counts that bound's violations next to the F64 bound.
//! * R1 — dense Gemma 4 MLP down (2112 -> 2816) at n in {232, 1180, 1832}:
//!   Q5_0 tensor V2 (production large tile), Q5_0 tensor V1, Q5_0 simdgroup,
//!   Q5_0 default policy (must be the simdgroup bytes), Q8_0 tensor V2 and
//!   Q8_0 simdgroup.
//! * R2 — K-scaling: the same arms at n = 232, N = 512 and K in
//!   {704, 2112, 4224, 8448}.
//! * R3 — MoE down `mm_id` (704 -> 2816, 128 experts, top-8 slotted) at the
//!   same widths: Q5_0 and Q8_0 on the tensor and simdgroup compact kernels.
//! * A1 — determinism: every tensor arm repeated is bitwise equal.
//! * A2 — dense row invariance of the Q5_0 tensor route: a token's output
//!   row is bitwise independent of the batch width, of its position in the
//!   batch, and equals a serial width-9 call over the same rows.
//!   (`tests/test_mm_id_compact_schedule.rs` carries the `mm_id` gates.)
//!
//! Run on both metallib paths:
//!
//! ```text
//! MLX_PRECOMPILED_METALLIB=1 cargo test --release --locked --test q5_0_tensor_f64_reference -- --test-threads=1 --nocapture
//! MLX_PRECOMPILED_METALLIB=0 cargo test --release --locked --test q5_0_tensor_f64_reference -- --test-threads=1 --nocapture
//! ```

use mlx_native::gguf::test_only_dequantize;
use mlx_native::{
    quantized_matmul_ggml_with_policy, quantized_matmul_id_ggml_pooled_slotted_with_policy, DType,
    GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlRoutingPolicy,
    GgmlTensorMmPreference, GgmlType, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice,
};

const U_H: f64 = 1.0 / 2048.0;
const U_F: f64 = 1.0 / 16_777_216.0;
const Q5_0_BLOCK: usize = 22;
const Q8_0_BLOCK: usize = 34;
const QK: usize = 32;

const WIDTHS: [usize; 3] = [232, 1180, 1832];
const DENSE_N: usize = 2816;
const DENSE_K: usize = 2112;
const MOE_N: usize = 2816;
const MOE_K: usize = 704;
const MOE_EXPERTS: usize = 128;
const TOP_K: usize = 8;

fn bound_factor(k: usize) -> f64 {
    2.0 * U_H + U_H * U_H + k as f64 * U_F
}

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

    /// Signed f16 bits with magnitude in [2^-12, 2^-4).
    fn scale_f16_bits(&mut self) -> u16 {
        let r = self.next_u64();
        let exponent = 3 + (r % 8) as u16;
        let mantissa = ((r >> 8) & 0x3ff) as u16;
        let sign = if (r >> 20) & 1 == 1 { 0x8000 } else { 0 };
        sign | (exponent << 10) | mantissa
    }

    /// Uniform in [-4, 4).
    fn input_value(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 8.0 - 4.0
    }
}

/// `rows x k` random Q5_0 blocks with finite normal scales.
fn q5_0_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks = rows * k / QK;
    let mut rng = Rng(seed);
    let mut bytes = vec![0u8; blocks * Q5_0_BLOCK];
    for block in bytes.chunks_exact_mut(Q5_0_BLOCK) {
        block[..2].copy_from_slice(&rng.scale_f16_bits().to_le_bytes());
        let payload = rng.next_u64().to_le_bytes();
        let more = rng.next_u64().to_le_bytes();
        let last = rng.next_u64().to_le_bytes();
        block[2..10].copy_from_slice(&payload);
        block[10..18].copy_from_slice(&more);
        block[18..22].copy_from_slice(&last[..4]);
    }
    bytes
}

/// Lossless Q5_0 -> Q8_0 re-encoding: same scale, int8 payload `q - 16`.
fn q8_0_from_q5_0(q5: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(q5.len() / Q5_0_BLOCK * Q8_0_BLOCK);
    for block in q5.chunks_exact(Q5_0_BLOCK) {
        out.extend_from_slice(&block[..2]);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        for j in 0..QK {
            let nibble = if j < 16 {
                block[6 + j] & 0x0f
            } else {
                block[6 + j - 16] >> 4
            };
            let q = i32::from(nibble) | ((((qh >> j) & 1) as i32) << 4);
            out.push((q - 16) as i8 as u8);
        }
    }
    out
}

fn dequant_rows(bytes: &[u8], ggml_type: GgmlType, rows: usize, k: usize) -> Vec<f32> {
    let row_bytes = k / QK * ggml_type.block_bytes() as usize;
    let mut out = vec![0.0f32; rows * k];
    for row in 0..rows {
        test_only_dequantize(
            &bytes[row * row_bytes..(row + 1) * row_bytes],
            ggml_type,
            &mut out[row * k..(row + 1) * k],
        )
        .expect("dequantize row");
    }
    out
}

fn inputs(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..len).map(|_| rng.input_value()).collect()
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

fn words(buffer: &MlxBuffer, start: usize, len: usize) -> &[u32] {
    &buffer.as_slice::<u32>().expect("words")[start..start + len]
}

fn assert_bitwise(label: &str, left: &[u32], right: &[u32]) {
    assert_eq!(left.len(), right.len(), "{label}: lengths");
    if let Some(first) = left.iter().zip(right).position(|(a, b)| a != b) {
        let differing = left.iter().zip(right).filter(|(a, b)| a != b).count();
        panic!(
            "{label}: {differing} of {} words differ; first at {first}: {:#010x} vs {:#010x}",
            left.len(),
            left[first],
            right[first]
        );
    }
}

/// Tokens checked against the F64 reference: every token of a short batch;
/// otherwise tile edges plus a stride.
fn sample_tokens(max_width: usize) -> Vec<usize> {
    if max_width <= 64 {
        return (0..max_width).collect();
    }
    let mut tokens: Vec<usize> = (0..max_width).step_by(29).collect();
    for edge in [1, 31, 32, 63, 64, 127, 128, 129, 231, 1179, 1831] {
        if edge < max_width {
            tokens.push(edge);
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

fn parallel_map<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1);
    let mut items = items;
    let mut chunks = Vec::new();
    while !items.is_empty() {
        let rest = items.split_off(chunk.min(items.len()));
        chunks.push(std::mem::replace(&mut items, rest));
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| scope.spawn(|| chunk.into_iter().map(&f).collect::<Vec<R>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("reference thread"))
            .collect()
    })
}

/// F64 dot products of one activation row against `n` weight rows:
/// `(sum x w, sum |x w|)` per output column.
fn reference_row(x: &[f32], weights: &[f32], n: usize, k: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|col| {
            let w = &weights[col * k..(col + 1) * k];
            let mut dot = 0.0f64;
            let mut abs = 0.0f64;
            for (&a, &b) in x.iter().zip(w) {
                let product = f64::from(a) * f64::from(b);
                dot += product;
                abs += product.abs();
            }
            (dot, abs)
        })
        .collect()
}

#[derive(Default)]
struct Stats {
    max_abs: f64,
    max_norm: f64,
    max_bound_fraction: f64,
    err2: f64,
    ref2: f64,
    elements: usize,
    /// Elements outside the original fixed `2e-3 * max(|y|, 1)` bound that
    /// excluded Q5_0 from the tensor route (reported, not asserted).
    legacy_violations: usize,
}

impl Stats {
    fn add(&mut self, got: f32, (want, abs): (f64, f64), k: usize) {
        assert!(got.is_finite(), "non-finite output {got}");
        let err = (f64::from(got) - want).abs();
        self.max_abs = self.max_abs.max(err);
        self.max_norm = self.max_norm.max(err / abs);
        self.max_bound_fraction = self.max_bound_fraction.max(err / (bound_factor(k) * abs));
        self.err2 += err * err;
        self.ref2 += want * want;
        self.elements += 1;
        if err > 2.0e-3 * want.abs().max(1.0) {
            self.legacy_violations += 1;
        }
    }

    fn report(&self, label: &str, k: usize) -> String {
        format!(
            "F64REF {label} K={k}: max_abs={:.3e} rel_rms={:.3e} max_err/sum|xw|={:.3e} bound_fraction={:.3} (bound factor {:.3e}) legacy_2e-3_violations={}/{}",
            self.max_abs,
            (self.err2 / self.ref2).sqrt(),
            self.max_norm,
            self.max_bound_fraction,
            bound_factor(k),
            self.legacy_violations,
            self.elements,
        )
    }
}

// ---------------------------------------------------------------------------
// Dense arms
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum DenseArm {
    Q5TensorV2,
    Q5TensorV1,
    Q5Simd,
    Q5Default,
    Q8TensorV2,
    Q8Simd,
}

const DENSE_ARMS: [DenseArm; 6] = [
    DenseArm::Q5TensorV2,
    DenseArm::Q5TensorV1,
    DenseArm::Q5Simd,
    DenseArm::Q5Default,
    DenseArm::Q8TensorV2,
    DenseArm::Q8Simd,
];

impl DenseArm {
    fn ggml_type(self) -> GgmlType {
        match self {
            Self::Q8TensorV2 | Self::Q8Simd => GgmlType::Q8_0,
            _ => GgmlType::Q5_0,
        }
    }

    fn tensor(self) -> bool {
        matches!(self, Self::Q5TensorV2 | Self::Q5TensorV1 | Self::Q8TensorV2)
    }

    fn policy(self) -> GgmlRoutingPolicy {
        let base = GgmlRoutingPolicy::default();
        match self {
            Self::Q5TensorV2 => GgmlRoutingPolicy {
                q5_0_tensor_mm: true,
                ..base
            },
            Self::Q5TensorV1 => GgmlRoutingPolicy {
                q5_0_tensor_mm: true,
                allow_dense_large_tile_mm: false,
                ..base
            },
            Self::Q5Simd | Self::Q8Simd => GgmlRoutingPolicy {
                dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
                ..base
            },
            Self::Q5Default | Self::Q8TensorV2 => base,
        }
    }

    /// The kernel the arm must resolve, and one it must not.
    fn kernels(self) -> (&'static str, &'static str) {
        match self {
            Self::Q5TensorV2 => ("kernel_mul_mm_q5_0_tensor_v2_f32", "kernel_mul_mm_q5_0_f32"),
            Self::Q5TensorV1 => (
                "kernel_mul_mm_q5_0_tensor_f32",
                "kernel_mul_mm_q5_0_tensor_v2_f32",
            ),
            Self::Q5Simd | Self::Q5Default => {
                ("kernel_mul_mm_q5_0_f32", "kernel_mul_mm_q5_0_tensor_v2_f32")
            }
            Self::Q8TensorV2 => ("kernel_mul_mm_q8_0_tensor_v2_f32", "kernel_mul_mm_q8_0_f32"),
            Self::Q8Simd => ("kernel_mul_mm_q8_0_f32", "kernel_mul_mm_q8_0_tensor_v2_f32"),
        }
    }
}

struct Dense {
    device: MlxDevice,
}

impl Dense {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        registry: &mut KernelRegistry,
        arm: DenseArm,
        input: &MlxBuffer,
        weight: &MlxBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> MlxBuffer {
        let output = output_buffer(&self.device, m * n);
        let mut encoder = self.device.command_encoder().expect("encoder");
        quantized_matmul_ggml_with_policy(
            &mut encoder,
            registry,
            &self.device,
            input,
            weight,
            &output,
            &GgmlQuantizedMatmulParams {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                ggml_type: arm.ggml_type(),
            },
            &arm.policy(),
        )
        .expect("dense dispatch");
        encoder.commit_and_wait().expect("dense command buffer");
        output
    }
}

fn tensor_available(device: &MlxDevice) -> bool {
    let mut registry = KernelRegistry::new();
    let available = registry
        .get_pipeline("kernel_mul_mm_q4_0_tensor_f32", device.metal_device())
        .is_ok()
        && registry
            .get_pipeline("kernel_mul_mm_id_q4_0_tensor_f32", device.metal_device())
            .is_ok();
    eprintln!(
        "device={} tensor_available={available} MLX_PRECOMPILED_METALLIB={:?}",
        device.name(),
        std::env::var("MLX_PRECOMPILED_METALLIB").ok()
    );
    // The qualification is of the tensor path; a device without it must opt
    // out explicitly (as the compact-schedule suite does), and the skip is
    // recorded.
    if !available {
        assert!(
            std::env::var("MLX_MM_ID_TEST_ALLOW_NO_TENSOR").as_deref() == Ok("1"),
            "the Q5_0 tensor qualification needs the tensor pipeline on {}: set MLX_MM_ID_TEST_ALLOW_NO_TENSOR=1 to skip it",
            device.name()
        );
        eprintln!("RECORDED SKIP: tensor pipeline unavailable; Q5_0 tensor qualification not run");
    }
    available
}

/// Whether `kernel` resolved in `registry`, with or without the dense MM
/// function-constant specialisation suffix.
fn dense_ran(registry: &KernelRegistry, kernel: &str) -> bool {
    registry.is_cached(kernel) || registry.is_cached(&format!("{kernel}|700:i1|701:i1|702:i1"))
}

/// Runs every dense arm over `widths` on `q5` (and its lossless Q8_0
/// re-encoding) and the activations `x`: checks the F64 bound, determinism,
/// that the Q5_0 default policy is the simdgroup bytes, and that each Q5_0
/// kernel equals its Q8_0 counterpart bit for bit. Returns the labels of
/// arms outside the bound.
fn dense_case(
    dense: &Dense,
    n: usize,
    k: usize,
    widths: &[usize],
    q5: &[u8],
    x: &[f32],
) -> Vec<String> {
    let mut violations = Vec::new();
    let q8 = q8_0_from_q5_0(q5);
    let weights = dequant_rows(q5, GgmlType::Q5_0, n, k);
    let control = dequant_rows(&q8, GgmlType::Q8_0, n, k);
    assert!(
        weights
            .iter()
            .zip(&control)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the Q8_0 control must carry the Q5_0 values bit for bit"
    );
    let max_width = *widths.iter().max().expect("widths");
    assert_eq!(x.len(), max_width * k, "activation rows");
    let tokens = sample_tokens(max_width);
    let reference = parallel_map(tokens.clone(), |t| {
        reference_row(&x[t * k..(t + 1) * k], &weights, n, k)
    });
    let input = f32_buffer(&dense.device, x);
    let q5_buffer = u8_buffer(&dense.device, q5);
    let q8_buffer = u8_buffer(&dense.device, &q8);
    for &m in widths {
        let mut outputs: Vec<(DenseArm, Vec<u32>)> = Vec::new();
        for arm in DENSE_ARMS {
            let weight = match arm.ggml_type() {
                GgmlType::Q5_0 => &q5_buffer,
                _ => &q8_buffer,
            };
            let mut registry = KernelRegistry::new();
            let output = dense.run(&mut registry, arm, &input, weight, m, n, k);
            let (want, absent) = arm.kernels();
            assert!(dense_ran(&registry, want), "{arm:?}: {want} did not run");
            assert!(!dense_ran(&registry, absent), "{arm:?}: {absent} ran");
            let got = output.as_slice::<f32>().expect("output");
            let mut stats = Stats::default();
            for (index, &t) in tokens.iter().enumerate().filter(|(_, &t)| t < m) {
                for col in 0..n {
                    stats.add(got[t * n + col], reference[index][col], k);
                }
            }
            let label = format!("dense {arm:?} n={m} N={n}");
            eprintln!("{}", stats.report(&label, k));
            if stats.max_bound_fraction > 1.0 {
                violations.push(label.clone());
            }
            let all = words(&output, 0, m * n).to_vec();
            if arm.tensor() {
                let again = dense.run(&mut registry, arm, &input, weight, m, n, k);
                assert_bitwise(
                    &format!("{label} determinism"),
                    &all,
                    words(&again, 0, m * n),
                );
            }
            outputs.push((arm, all));
        }
        let of = |arm: DenseArm| {
            outputs
                .iter()
                .find(|(a, _)| *a == arm)
                .map(|(_, words)| words.as_slice())
                .expect("arm ran")
        };
        for (left, right, what) in [
            (
                DenseArm::Q5Default,
                DenseArm::Q5Simd,
                "default policy == simdgroup",
            ),
            (
                DenseArm::Q5TensorV2,
                DenseArm::Q8TensorV2,
                "Q5_0 tensor == Q8_0 tensor",
            ),
            (
                DenseArm::Q5Simd,
                DenseArm::Q8Simd,
                "Q5_0 simdgroup == Q8_0 simdgroup",
            ),
        ] {
            assert_bitwise(
                &format!("dense n={m} N={n} K={k} {what}"),
                of(left),
                of(right),
            );
        }
        eprintln!("dense n={m} N={n} K={k}: Q5_0 == Q8_0 control bitwise on the tensor V2 and simdgroup kernels");
    }
    violations
}

/// R0 — the fixture that excluded Q5_0 (`tests/q5_0_metal_parity.rs`:
/// N = 17, K = 64, scales up to 1.0, inputs `(v - 48) / 23`) at the failing
/// widths M = 9 and M = 33, rebuilt byte for byte.
#[test]
fn r0_original_exclusion_fixture_against_f64_reference() {
    const N: usize = 17;
    const K: usize = 64;
    let block = |seed: usize| {
        let mut bytes = [0u8; Q5_0_BLOCK];
        let scale = [0.125f32, -0.25, 0.5, -1.0][seed % 4];
        bytes[..2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        let qh = (0..32).fold(0u32, |mask, bit| {
            mask | ((((seed * 13 + bit * 7 + 3) >> 2) as u32 & 1) << bit)
        });
        bytes[2..6].copy_from_slice(&qh.to_le_bytes());
        for lane in 0..16 {
            let low = ((seed * 5 + lane * 11 + 1) & 0x0f) as u8;
            let high = ((seed * 17 + lane * 3 + 9) & 0x0f) as u8;
            bytes[6 + lane] = low | (high << 4);
        }
        bytes
    };
    let mut q5 = Vec::new();
    for row in 0..N {
        for block_index in 0..K / QK {
            q5.extend_from_slice(&block(7 + row * 19 + block_index));
        }
    }
    let input_rows = |rows: usize, salt: usize| -> Vec<f32> {
        (0..rows * K)
            .map(|index| {
                let value = ((index * 29 + salt * 37 + index / K * 11) % 97) as f32;
                (value - 48.0) / 23.0
            })
            .collect()
    };
    let dense = Dense {
        device: MlxDevice::new().expect("device"),
    };
    if !tensor_available(&dense.device) {
        return;
    }
    let mut violations = Vec::new();
    for m in [9usize, 33] {
        violations.extend(dense_case(&dense, N, K, &[m], &q5, &input_rows(m, m)));
    }
    assert!(
        violations.is_empty(),
        "outside the K-scaled F64 bound: {violations:?}"
    );
}

#[test]
fn r1_dense_gemma_down_against_f64_reference() {
    let dense = Dense {
        device: MlxDevice::new().expect("device"),
    };
    if !tensor_available(&dense.device) {
        return;
    }
    let violations = dense_case(
        &dense,
        DENSE_N,
        DENSE_K,
        &WIDTHS,
        &q5_0_bytes(DENSE_N, DENSE_K, 0xd0d0_2112),
        &inputs(WIDTHS[2] * DENSE_K, 0xd0d0_2112 ^ 0xa11ce),
    );
    assert!(
        violations.is_empty(),
        "outside the K-scaled F64 bound: {violations:?}"
    );
}

#[test]
fn r2_dense_k_scaling_against_f64_reference() {
    let dense = Dense {
        device: MlxDevice::new().expect("device"),
    };
    if !tensor_available(&dense.device) {
        return;
    }
    let mut violations = Vec::new();
    for k in [704, 2112, 4224, 8448] {
        let seed = 0x5ca1e ^ k as u64;
        violations.extend(dense_case(
            &dense,
            512,
            k,
            &[232],
            &q5_0_bytes(512, k, seed),
            &inputs(232 * k, seed ^ 0xa11ce),
        ));
    }
    assert!(
        violations.is_empty(),
        "outside the K-scaled F64 bound: {violations:?}"
    );
}

// ---------------------------------------------------------------------------
// A2 — dense row invariance of the Q5_0 tensor route
// ---------------------------------------------------------------------------

#[test]
fn a2_dense_q5_0_tensor_rows_are_width_position_and_serial_invariant() {
    let dense = Dense {
        device: MlxDevice::new().expect("device"),
    };
    if !tensor_available(&dense.device) {
        return;
    }
    let (n, k) = (DENSE_N, DENSE_K);
    let q5 = q5_0_bytes(n, k, 0xa2a2);
    let weight = u8_buffer(&dense.device, &q5);
    let full = *WIDTHS.iter().max().expect("widths");
    let x = inputs(full * k, 0xa2a2_0001);
    for arm in [DenseArm::Q5TensorV2, DenseArm::Q5TensorV1] {
        let mut registry = KernelRegistry::new();
        let reference = dense.run(
            &mut registry,
            arm,
            &f32_buffer(&dense.device, &x),
            &weight,
            full,
            n,
            k,
        );
        // (first row, width): prefixes at every Gemma width, windows that
        // move rows across tile positions, and serial width-9 calls.
        let mut windows = vec![
            (0usize, 232usize),
            (0, 1180),
            (37, 1180),
            (101, 232),
            (1, 1831),
        ];
        windows.extend([(0, 9), (9, 9), (123, 9), (1823, 9)]);
        for (first, width) in windows {
            let slice = f32_buffer(&dense.device, &x[first * k..(first + width) * k]);
            let output = dense.run(&mut registry, arm, &slice, &weight, width, n, k);
            assert_bitwise(
                &format!("A2 {arm:?} rows {first}..{}", first + width),
                words(&reference, first * n, width * n),
                words(&output, 0, width * n),
            );
        }
        eprintln!("A2 {arm:?}: every window is bitwise equal to the n={full} batch");
    }
}

// ---------------------------------------------------------------------------
// R3 — MoE down mm_id
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum MoeArm {
    Q5Tensor,
    Q5Simd,
    Q8Tensor,
    Q8Simd,
}

impl MoeArm {
    fn ggml_type(self) -> GgmlType {
        match self {
            Self::Q5Tensor | Self::Q5Simd => GgmlType::Q5_0,
            Self::Q8Tensor | Self::Q8Simd => GgmlType::Q8_0,
        }
    }

    fn tensor(self) -> bool {
        matches!(self, Self::Q5Tensor | Self::Q8Tensor)
    }

    fn policy(self) -> GgmlRoutingPolicy {
        GgmlRoutingPolicy {
            expert_tensor_mm: if self.tensor() {
                GgmlTensorMmPreference::AutoProbe
            } else {
                GgmlTensorMmPreference::ForceSimd
            },
            q5_0_tensor_mm: true,
            ..GgmlRoutingPolicy::default()
        }
    }

    fn kernel(self) -> &'static str {
        match self {
            Self::Q5Tensor => "kernel_mul_mm_id_q5_0_tensor_f32_c",
            Self::Q5Simd => "kernel_mul_mm_id_q5_0_f32_c",
            Self::Q8Tensor => "kernel_mul_mm_id_q8_0_tensor_f32_c",
            Self::Q8Simd => "kernel_mul_mm_id_q8_0_f32_c",
        }
    }
}

fn route_ids(n_tokens: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed);
    let mut ids = Vec::with_capacity(n_tokens * TOP_K);
    for _ in 0..n_tokens {
        let start = ids.len();
        while ids.len() - start < TOP_K {
            let expert = (rng.next_u64() % MOE_EXPERTS as u64) as u32;
            if !ids[start..].contains(&expert) {
                ids.push(expert);
            }
        }
    }
    ids
}

#[test]
fn r3_moe_down_mm_id_against_f64_reference() {
    let device = MlxDevice::new().expect("device");
    if !tensor_available(&device) {
        return;
    }
    let (n, k) = (MOE_N, MOE_K);
    let expert_q5 = n * k / QK * Q5_0_BLOCK;
    let q5 = q5_0_bytes(MOE_EXPERTS * n, k, 0x0e0e_0704);
    let q8 = q8_0_from_q5_0(&q5);
    let expert_q8 = n * k / QK * Q8_0_BLOCK;
    let full = *WIDTHS.iter().max().expect("widths");
    let x = inputs(full * TOP_K * k, 0x0e0e_1111);
    let ids = route_ids(full, 0x0e0e_2222);
    let tokens = sample_tokens(full);

    // Reference per (token, slot), one expert dequantized at a time.
    let mut pairs_by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); MOE_EXPERTS];
    for (index, &t) in tokens.iter().enumerate() {
        for slot in 0..TOP_K {
            pairs_by_expert[ids[t * TOP_K + slot] as usize].push((index, slot));
        }
    }
    let per_expert = parallel_map((0..MOE_EXPERTS).collect(), |expert| {
        let bytes = &q5[expert * expert_q5..(expert + 1) * expert_q5];
        let weights = dequant_rows(bytes, GgmlType::Q5_0, n, k);
        let control = dequant_rows(
            &q8[expert * expert_q8..(expert + 1) * expert_q8],
            GgmlType::Q8_0,
            n,
            k,
        );
        assert!(
            weights
                .iter()
                .zip(&control)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "expert {expert}: the Q8_0 control must carry the Q5_0 values bit for bit"
        );
        pairs_by_expert[expert]
            .iter()
            .map(|&(index, slot)| {
                let t = tokens[index];
                let row = &x[(t * TOP_K + slot) * k..(t * TOP_K + slot + 1) * k];
                ((index, slot), reference_row(row, &weights, n, k))
            })
            .collect::<Vec<_>>()
    });
    let mut reference = vec![vec![Vec::new(); TOP_K]; tokens.len()];
    for ((index, slot), row) in per_expert.into_iter().flatten() {
        reference[index][slot] = row;
    }

    let input = f32_buffer(&device, &x);
    let ids_buffer = u32_buffer(&device, &ids);
    let q5_buffer = u8_buffer(&device, &q5);
    let q8_buffer = u8_buffer(&device, &q8);
    let mut violations = Vec::new();
    for &m in &WIDTHS {
        // Pairs (Q5_0, then its Q8_0 control) on one kernel path each.
        let mut q5_output: Option<MlxBuffer> = None;
        for arm in [
            MoeArm::Q5Tensor,
            MoeArm::Q8Tensor,
            MoeArm::Q5Simd,
            MoeArm::Q8Simd,
        ] {
            let (weight, stride) = match arm.ggml_type() {
                GgmlType::Q5_0 => (&q5_buffer, expert_q5),
                _ => (&q8_buffer, expert_q8),
            };
            let params = GgmlQuantizedMatmulIdParams {
                n_tokens: m as u32,
                top_k: TOP_K as u32,
                n: n as u32,
                k: k as u32,
                n_experts: MOE_EXPERTS as u32,
                expert_stride: stride as u64,
                ggml_type: arm.ggml_type(),
            };
            let mut registry = KernelRegistry::new();
            let mut scratch =
                IdMmScratch::alloc(&device, MOE_EXPERTS as u32, m as u32).expect("scratch");
            let mut run = |registry: &mut KernelRegistry| {
                let output = output_buffer(&device, m * TOP_K * n);
                let mut encoder = device.command_encoder().expect("encoder");
                quantized_matmul_id_ggml_pooled_slotted_with_policy(
                    &mut encoder,
                    registry,
                    &device,
                    &input,
                    weight,
                    &ids_buffer,
                    &output,
                    &mut scratch,
                    &params,
                    &arm.policy(),
                )
                .expect("mm_id dispatch");
                encoder.commit_and_wait().expect("mm_id command buffer");
                output
            };
            let output = run(&mut registry);
            assert!(
                registry.is_cached(arm.kernel()),
                "{arm:?}: {} did not run",
                arm.kernel()
            );
            let got = output.as_slice::<f32>().expect("output");
            let mut stats = Stats::default();
            for (index, &t) in tokens.iter().enumerate().filter(|(_, &t)| t < m) {
                for slot in 0..TOP_K {
                    for col in 0..n {
                        stats.add(
                            got[(t * TOP_K + slot) * n + col],
                            reference[index][slot][col],
                            k,
                        );
                    }
                }
            }
            let label = format!("mm_id {arm:?} n={m} N={n} top_k={TOP_K}");
            eprintln!("{}", stats.report(&label, k));
            if stats.max_bound_fraction > 1.0 {
                violations.push(label.clone());
            }
            if arm.tensor() {
                let again = run(&mut registry);
                assert_bitwise(
                    &format!("{label} determinism"),
                    words(&output, 0, m * TOP_K * n),
                    words(&again, 0, m * TOP_K * n),
                );
            }
            match arm.ggml_type() {
                GgmlType::Q5_0 => q5_output = Some(output),
                _ => {
                    let q5 = q5_output.take().expect("Q5_0 arm ran first");
                    assert_bitwise(
                        &format!("{label}: Q5_0 == Q8_0 control"),
                        words(&q5, 0, m * TOP_K * n),
                        words(&output, 0, m * TOP_K * n),
                    );
                    eprintln!("mm_id n={m} {arm:?}: Q5_0 == Q8_0 control bitwise");
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "outside the K-scaled F64 bound: {violations:?}"
    );
}
