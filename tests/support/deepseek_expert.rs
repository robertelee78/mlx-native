// Shared helpers for the ADR-066 N-IQ2XXS DeepSeek-V4-Flash expert-shape
// tests (`tests/iq2_xxs_deepseek.rs`, `tests/q2_k_down_deepseek.rs`).
//
// Value ranges (design §3.7, fixed before any Metal run): IQ2_XXS `d` and
// Q2_K `d`/`dmin` magnitudes are drawn from [2^-12, 2^-4), inputs uniformly
// from [-4, 4] with magnitudes below 2^-14 set to 0, so half staging in the
// mm_id templates neither overflows nor goes subnormal.

#![allow(dead_code)]

use mlx_native::gguf::test_only_dequantize;
use mlx_native::{
    DType, GgmlRoutingPolicy, GgmlTensorMmPreference, GgmlType, KernelRegistry, MlxBuffer,
    MlxDevice,
};

pub const QK_K: usize = 256;
pub const EXPERTS: usize = 256;
pub const TOP_K: usize = 6;

/// splitmix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// f16 bits with magnitude in [2^-12, 2^-4) and the requested sign.
    pub fn scale_f16_bits(&mut self, allow_negative: bool) -> u16 {
        let r = self.next_u64();
        let exponent = 3 + (r % 8) as u16; // biased 3..=10: [2^-12, 2^-4)
        let mantissa = ((r >> 8) & 0x3ff) as u16;
        let sign = if allow_negative && (r >> 20) & 1 == 1 {
            0x8000
        } else {
            0
        };
        sign | (exponent << 10) | mantissa
    }

    pub fn input_value(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        let value = unit * 8.0 - 4.0;
        if value.abs() < 1.0 / 16384.0 {
            0.0
        } else {
            value
        }
    }
}

/// Expert weight geometry of one projection.
#[derive(Clone, Copy, Debug)]
pub struct ExpertShape {
    pub ggml_type: GgmlType,
    /// Output rows per expert.
    pub n: usize,
    /// Input width.
    pub k: usize,
}

impl ExpertShape {
    pub fn block_bytes(self) -> usize {
        self.ggml_type.block_bytes() as usize
    }

    pub fn row_bytes(self) -> usize {
        self.k / QK_K * self.block_bytes()
    }

    pub fn expert_bytes(self) -> usize {
        self.n * self.row_bytes()
    }

    pub fn total_bytes(self) -> usize {
        EXPERTS * self.expert_bytes()
    }
}

/// Seeded random expert stack written straight into a Metal buffer: every
/// byte random, then each block's scale fields replaced by in-range values.
pub fn expert_weights(device: &MlxDevice, shape: ExpertShape, seed: u64) -> MlxBuffer {
    let total = shape.total_bytes();
    let mut buffer = device
        .alloc_buffer(total, DType::U8, vec![total])
        .expect("expert weight buffer");
    let bytes = buffer.as_mut_slice::<u8>().expect("weight bytes");
    let mut rng = Rng::new(seed);
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next_u64().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    let block_bytes = shape.block_bytes();
    for block in bytes.chunks_exact_mut(block_bytes) {
        match shape.ggml_type {
            GgmlType::IQ2_XXS => {
                let d = rng.scale_f16_bits(true);
                block[..2].copy_from_slice(&d.to_le_bytes());
            }
            GgmlType::Q2_K => {
                let d = rng.scale_f16_bits(false);
                let dmin = rng.scale_f16_bits(false);
                block[80..82].copy_from_slice(&d.to_le_bytes());
                block[82..84].copy_from_slice(&dmin.to_le_bytes());
            }
            other => panic!("unsupported expert test type {other:?}"),
        }
    }
    buffer
}

pub fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32 slice")
        .copy_from_slice(values);
    buffer
}

pub fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("u32 buffer");
    buffer
        .as_mut_slice::<u32>()
        .expect("u32 slice")
        .copy_from_slice(values);
    buffer
}

pub fn zero_f32_buffer(device: &MlxDevice, len: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("output buffer");
    buffer.as_mut_slice::<f32>().expect("output").fill(0.0);
    buffer
}

pub fn inputs(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * k).map(|_| rng.input_value()).collect()
}

/// Seeded top-k over `EXPERTS` experts, distinct within each token.
pub fn routes(tokens: usize, top_k: usize, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut ids = Vec::with_capacity(tokens * top_k);
    for _ in 0..tokens {
        let start = ids.len();
        while ids.len() - start < top_k {
            let expert = rng.below(EXPERTS) as u32;
            if !ids[start..].contains(&expert) {
                ids.push(expert);
            }
        }
    }
    ids
}

/// Adversarial routing: every token routes to `shared` plus distinct others.
pub fn shared_expert_routes(tokens: usize, top_k: usize, shared: u32, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut ids = Vec::with_capacity(tokens * top_k);
    for token in 0..tokens {
        let start = ids.len();
        let shared_slot = token % top_k;
        for slot in 0..top_k {
            if slot == shared_slot {
                ids.push(shared);
                continue;
            }
            loop {
                let expert = rng.below(EXPERTS) as u32;
                if expert != shared && !ids[start..].contains(&expert) {
                    ids.push(expert);
                    break;
                }
            }
        }
    }
    ids
}

pub fn simd_policy() -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        expert_tensor_mm: GgmlTensorMmPreference::ForceSimd,
        ..GgmlRoutingPolicy::default()
    }
}

pub fn auto_policy() -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        expert_tensor_mm: GgmlTensorMmPreference::AutoProbe,
        ..GgmlRoutingPolicy::default()
    }
}

/// Whether this device builds tensor-API expert MM pipelines. The IQ2_XXS
/// and Q2_K tensor kernels must build exactly when the reference Q4_0
/// tensor kernel does.
pub fn tensor_mm_id_available(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    tensor_kernel: &str,
) -> bool {
    let reference = registry
        .get_pipeline("kernel_mul_mm_id_q4_0_tensor_f32", device.metal_device())
        .is_ok();
    let candidate = registry.get_pipeline(tensor_kernel, device.metal_device());
    assert_eq!(
        candidate.is_ok(),
        reference,
        "{tensor_kernel} availability must match the Q4_0 tensor probe: {:?}",
        candidate.err()
    );
    reference
}

/// Output columns checked against the CPU oracle: an odd stride covers
/// every residue of the 8-row mv and 64-row mm tiles, plus the last row.
pub fn sample_columns(n: usize, count: usize) -> Vec<usize> {
    let mut columns: Vec<usize> = (0..count.min(n)).map(|i| (i * 1031 + 7) % n).collect();
    columns.push(n - 1);
    columns.sort_unstable();
    columns.dedup();
    columns
}

/// Routed rows checked against the CPU oracle: all when few, otherwise the
/// ends plus a seeded spread.
pub fn sample_rows(total: usize, cap: usize, seed: u64) -> Vec<usize> {
    if total <= cap {
        return (0..total).collect();
    }
    let mut rng = Rng::new(seed);
    let mut rows = vec![0, total - 1];
    while rows.len() < cap {
        rows.push(rng.below(total));
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// CPU dequant (bit-exact against the reference) of one expert row.
pub fn dequant_row(weights: &[u8], shape: ExpertShape, expert: usize, row: usize) -> Vec<f32> {
    let start = expert * shape.expert_bytes() + row * shape.row_bytes();
    let mut out = vec![0.0f32; shape.k];
    test_only_dequantize(
        &weights[start..start + shape.row_bytes()],
        shape.ggml_type,
        &mut out,
    )
    .expect("CPU dequant");
    out
}

/// Assert `got` against the F64 oracle with `|got - want| <= bound * sum|w x|`
/// for the sampled routed rows and columns.
///
/// `input_row(r)` maps routed row `r` to its input row index.
#[allow(clippy::too_many_arguments)]
pub fn assert_oracle(
    label: &str,
    weights: &[u8],
    shape: ExpertShape,
    input: &[f32],
    ids: &[u32],
    input_row: impl Fn(usize) -> usize,
    got: &[f32],
    rows: &[usize],
    columns: &[usize],
    bound: f64,
) {
    let mut worst = 0.0f64;
    for &routed in rows {
        let expert = ids[routed] as usize;
        let x = &input[input_row(routed) * shape.k..(input_row(routed) + 1) * shape.k];
        for &column in columns {
            let w = dequant_row(weights, shape, expert, column);
            let mut want = 0.0f64;
            let mut magnitude = 0.0f64;
            for (&wi, &xi) in w.iter().zip(x) {
                let product = wi as f64 * xi as f64;
                want += product;
                magnitude += product.abs();
            }
            let value = got[routed * shape.n + column];
            assert!(
                value.is_finite(),
                "{label}: routed row {routed} column {column} is {value}"
            );
            let error = (value as f64 - want).abs();
            let allowed = bound * magnitude;
            assert!(
                error <= allowed,
                "{label}: routed row {routed} expert {expert} column {column}: got {value}, \
                 want {want}, |err| {error:e} > {bound:e} * {magnitude:e}"
            );
            if magnitude > 0.0 {
                worst = worst.max(error / magnitude);
            }
        }
    }
    eprintln!(
        "{label}: {} routed rows x {} columns, worst |err|/sum|wx| = {worst:e} (bound {bound:e})",
        rows.len(),
        columns.len()
    );
}

pub fn assert_bitwise(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (index, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: element {index}: {a:?} != {b:?}"
        );
    }
}
