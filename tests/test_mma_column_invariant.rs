//! Class B gates for `quantized_matmul_ggml_column_invariant_mma`, the
//! peer's few-row MMA technique (`mul_mv_mma`, 8×8 simdgroup matrices)
//! with a pipeline chosen from (type, N, K) only.
//!
//! It is not bitwise equal to the canonical mv routes, so correctness is
//! proven by:
//!
//! (a) an f64 CPU reference over the exactly dequantized weights with the
//!     K-scaled bound `|gpu − ref| ≤ C·K·ε_f32·Σ|w·x|` per output element;
//! (b) column invariance (Class A): for widths 1..=8, column `j` of the
//!     width-`w` result is bitwise equal to the width-1 result for that
//!     column's input row, also under a permuted column order;
//! (c) determinism (Class A): three repeats are bitwise identical;
//! (d) an output tail canary: nothing is written past `m·n`.
//!
//! Gemma 4 decode shapes: lm head Q6_K 262144×2816, Q4_K dense 4096×2816,
//! 8192×2816, 2048×2816, 2112×2816, 2816×4096, 2816×8192, Q6_K 2048×2816,
//! Q5_0/Q8_0 2816×2112
//! (K = 64·33), plus a small N that is not a multiple of the row tile for
//! every admitted type.
//!
//! GPU tests. Run with one test thread so the lm-head weight is the only
//! large allocation alive:
//!
//! ```text
//! cargo test --release --locked --test test_mma_column_invariant -- --test-threads=1
//! ```

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic)]

use mlx_native::gguf::test_only_dequantize;
use mlx_native::{
    quantized_matmul_ggml_column_invariant_mma, CommandEncoder, DType, GgmlQuantizedMatmulParams,
    GgmlType, KernelRegistry, MlxBuffer, MlxDevice,
};

/// A column-invariant entry under test.
type Entry = fn(
    &mut CommandEncoder,
    &mut KernelRegistry,
    &MlxDevice,
    &MlxBuffer,
    &MlxBuffer,
    &MlxBuffer,
    &GgmlQuantizedMatmulParams,
) -> mlx_native::Result<()>;

const MMA: Entry = quantized_matmul_ggml_column_invariant_mma;

/// Bound constant for `|gpu − ref| ≤ C·K·ε_f32·Σ|w·x|`, fixed here and never
/// loosened.
///
/// Justification: with unit roundoff `u = ε_f32/2`, any f32 summation order
/// of K products (the kernel's per-lane serial sums followed by the
/// simdgroup shuffle tree) has forward error at most `γ_K·Σ|w·x|` with
/// `γ_K = K·u/(1 − K·u) ≈ K·u`, and rounding each product (or fusing it)
/// adds at most `u·Σ|w·x|`. Together `(K + 1)·u·Σ|w·x| ≤ K·ε_f32·Σ|w·x|`,
/// i.e. C = 1 already covers the worst case for weights identical to the
/// CPU dequantization. C = 2 leaves one more `K·ε_f32` for the GPU
/// dequantizer evaluating the block affine map (`d·sc·q − dmin·m`) in a
/// different f32 order than the CPU helper, which perturbs each weight by a
/// few `u` of its terms — far below `K·u` per element at K ≥ 512.
///
/// The MMA entry's orders are summation orders of the same kind: 8-term
/// MMA dot steps accumulated per simdgroup over its round-robin K slice,
/// then a fixed-order f32 sum over simdgroups. Its Q8_0 block kind forms
/// `d·Σq·x` per 32-weight block (exact-integer q, f32 x) and fma-accumulates
/// the blocks, which applies the scale with one rounding per block instead
/// of one per product — still inside `(K + 1)·u` per element.
const BOUND_C: f64 = 2.0;

/// Input columns available to the width sweep.
const MAX_WIDTH: usize = 8;

/// Output tail canary: the kernel must never write past `m·n`.
const TAIL_CANARY: u32 = 0x7fc0_beef;
const TAIL_LEN: usize = 64;

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Byte offsets of the f16 scale fields within one block. Random payloads
/// keep these finite and small so every dequantized weight is finite.
fn half_scale_offsets(ggml_type: GgmlType) -> &'static [usize] {
    match ggml_type {
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => &[0],
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        other => panic!("no weight generator for {other:?}"),
    }
}

fn row_bytes(ggml_type: GgmlType, k: usize) -> usize {
    k / ggml_type.block_values() as usize * ggml_type.block_bytes() as usize
}

fn random_weight_bytes(ggml_type: GgmlType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let total = n * row_bytes(ggml_type, k);
    let mut bytes = vec![0u8; total];
    let mut rng = Xorshift(seed | 1);
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    let block = ggml_type.block_bytes() as usize;
    for block_bytes in bytes.chunks_exact_mut(block) {
        for &offset in half_scale_offsets(ggml_type) {
            let scale = 0.002 + 0.006 * rng.unit();
            let bits = half::f16::from_f64(scale).to_bits();
            block_bytes[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    bytes
}

fn random_inputs(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift(seed | 1);
    (0..MAX_WIDTH * k)
        .map(|_| (rng.unit() - 0.5) as f32)
        .collect()
}

/// f64 reference over the CPU-dequantized weights for every input column.
/// Returns `(ref, abs_sum)` laid out `[column][row]`.
fn cpu_reference(
    ggml_type: GgmlType,
    weights: &[u8],
    inputs: &[f32],
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let per_row = row_bytes(ggml_type, k);
    let mut reference = vec![0.0f64; MAX_WIDTH * n];
    let mut abs_sum = vec![0.0f64; MAX_WIDTH * n];
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(16);
    let rows_per_thread = n.div_ceil(threads);
    let results: Vec<(usize, Vec<f64>, Vec<f64>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let start = (t * rows_per_thread).min(n);
                let end = ((t + 1) * rows_per_thread).min(n);
                scope.spawn(move || {
                    let mut dequantized = vec![0.0f32; k];
                    let mut local_ref = vec![0.0f64; MAX_WIDTH * (end - start)];
                    let mut local_abs = vec![0.0f64; MAX_WIDTH * (end - start)];
                    for row in start..end {
                        test_only_dequantize(
                            &weights[row * per_row..(row + 1) * per_row],
                            ggml_type,
                            &mut dequantized,
                        )
                        .expect("CPU dequantize");
                        for column in 0..MAX_WIDTH {
                            let x = &inputs[column * k..(column + 1) * k];
                            let mut dot = 0.0f64;
                            let mut abs = 0.0f64;
                            for (w, v) in dequantized.iter().zip(x) {
                                let product = f64::from(*w) * f64::from(*v);
                                dot += product;
                                abs += product.abs();
                            }
                            local_ref[column * (end - start) + row - start] = dot;
                            local_abs[column * (end - start) + row - start] = abs;
                        }
                    }
                    (start, local_ref, local_abs)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("reference thread"))
            .collect()
    });
    for (start, local_ref, local_abs) in results {
        let rows = local_ref.len() / MAX_WIDTH;
        for column in 0..MAX_WIDTH {
            for r in 0..rows {
                reference[column * n + start + r] = local_ref[column * rows + r];
                abs_sum[column * n + start + r] = local_abs[column * rows + r];
            }
        }
    }
    (reference, abs_sum)
}

struct Gpu {
    device: MlxDevice,
    registry: KernelRegistry,
    entry: Entry,
}

impl Gpu {
    fn new(entry: Entry) -> Self {
        Self {
            device: MlxDevice::new().expect("Metal device"),
            registry: KernelRegistry::new(),
            entry,
        }
    }

    fn weight(&self, bytes: &[u8]) -> MlxBuffer {
        let mut buffer = self
            .device
            .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
            .expect("weight buffer");
        buffer
            .as_mut_slice::<u8>()
            .expect("weight bytes")
            .copy_from_slice(bytes);
        buffer
    }

    /// Run the entry on `columns` (each a K-length row) and return the
    /// `[columns.len(), n]` output as raw f32 bits.
    fn run(
        &mut self,
        weight: &MlxBuffer,
        ggml_type: GgmlType,
        columns: &[&[f32]],
        n: usize,
        k: usize,
    ) -> Vec<u32> {
        let m = columns.len();
        let mut input = self
            .device
            .alloc_buffer(m * k * 4, DType::F32, vec![m, k])
            .expect("input buffer");
        {
            let values = input.as_mut_slice::<f32>().expect("input values");
            for (j, column) in columns.iter().enumerate() {
                values[j * k..(j + 1) * k].copy_from_slice(column);
            }
        }
        let elements = m * n + TAIL_LEN;
        let mut output = self
            .device
            .alloc_buffer(elements * 4, DType::F32, vec![elements])
            .expect("output buffer");
        output
            .as_mut_slice::<u32>()
            .expect("output bits")
            .fill(TAIL_CANARY);
        let params = GgmlQuantizedMatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type,
        };
        let mut encoder = self.device.command_encoder().expect("encoder");
        (self.entry)(
            &mut encoder,
            &mut self.registry,
            &self.device,
            &input,
            weight,
            &output,
            &params,
        )
        .expect("column-invariant dispatch");
        encoder.commit_and_wait().expect("command buffer");
        let bits = output.as_slice::<u32>().expect("output bits");
        assert!(
            bits[m * n..].iter().all(|&b| b == TAIL_CANARY),
            "{ggml_type:?} n={n} k={k} m={m}: kernel wrote past m·n"
        );
        bits[..m * n].to_vec()
    }
}

fn assert_reference_bound(
    label: &str,
    gpu_bits: &[u32],
    reference: &[f64],
    abs_sum: &[f64],
    n: usize,
    k: usize,
) {
    let scale = BOUND_C * k as f64 * f64::from(f32::EPSILON);
    let mut worst_ratio = 0.0f64;
    for column in 0..MAX_WIDTH {
        for row in 0..n {
            let value = f32::from_bits(gpu_bits[column * n + row]);
            assert!(value.is_finite(), "{label}: non-finite output col {column} row {row}");
            let index = column * n + row;
            let error = (f64::from(value) - reference[index]).abs();
            let bound = scale * abs_sum[index];
            assert!(
                error <= bound,
                "{label}: col {column} row {row}: |{value} − {}| = {error:e} > {bound:e} \
                 (C={BOUND_C}, K={k}, Σ|w·x|={:e})",
                reference[index],
                abs_sum[index]
            );
            if bound > 0.0 {
                worst_ratio = worst_ratio.max(error / bound);
            }
        }
    }
    eprintln!("[{label}] reference bound OK, worst error/bound = {worst_ratio:.3e}");
}

/// Reversed column order: at widths ≥ 5 inputs also cross threadgroup
/// tiles (input 0 lands in slot 3 of the second r1 tile at width 8).
fn permutation(w: usize) -> Vec<usize> {
    (0..w).rev().collect()
}

fn check_shape(gpu: &mut Gpu, ggml_type: GgmlType, n: usize, k: usize, seed: u64) {
    let label = format!("{ggml_type:?} n={n} k={k}");
    let weights = random_weight_bytes(ggml_type, n, k, seed);
    let inputs = random_inputs(k, seed ^ 0x9e37_79b9_7f4a_7c15);
    let column = |j: usize| &inputs[j * k..(j + 1) * k];
    let weight = gpu.weight(&weights);

    // Width-1 authority for every input column.
    let singles: Vec<Vec<u32>> = (0..MAX_WIDTH)
        .map(|j| gpu.run(&weight, ggml_type, &[column(j)], n, k))
        .collect();

    // (b) column invariance, natural and permuted order, widths 1..=8.
    for w in 1..=MAX_WIDTH {
        let natural: Vec<&[f32]> = (0..w).map(column).collect();
        let out = gpu.run(&weight, ggml_type, &natural, n, k);
        for j in 0..w {
            assert!(
                out[j * n..(j + 1) * n] == singles[j][..],
                "{label}: width {w} column {j} differs bitwise from width 1"
            );
        }
        let order = permutation(w);
        let permuted: Vec<&[f32]> = order.iter().map(|&j| column(j)).collect();
        let out = gpu.run(&weight, ggml_type, &permuted, n, k);
        for (slot, &j) in order.iter().enumerate() {
            assert!(
                out[slot * n..(slot + 1) * n] == singles[j][..],
                "{label}: width {w} permuted slot {slot} (input {j}) differs bitwise from width 1"
            );
        }
    }

    // (c) determinism: three repeats at widths 8 and 3 are bitwise equal.
    for w in [MAX_WIDTH, 3] {
        let columns: Vec<&[f32]> = (0..w).map(column).collect();
        let first = gpu.run(&weight, ggml_type, &columns, n, k);
        for repeat in 1..3 {
            let again = gpu.run(&weight, ggml_type, &columns, n, k);
            assert!(
                again == first,
                "{label}: width {w} repeat {repeat} is not bitwise identical"
            );
        }
    }
    drop(weight);

    // (a) f64 reference with the K-scaled bound. Columns are width
    // invariant (checked above), so the width-1 results cover every width.
    let (reference, abs_sum) = cpu_reference(ggml_type, &weights, &inputs, n, k);
    let gpu_bits: Vec<u32> = singles.concat();
    assert_reference_bound(&label, &gpu_bits, &reference, &abs_sum, n, k);
}

#[test]
fn mma_lm_head_q6_k_262144x2816() {
    let mut gpu = Gpu::new(MMA);
    check_shape(&mut gpu, GgmlType::Q6_K, 262_144, 2816, 0x1a6d_0101);
}

#[test]
fn mma_q4_k_gemma_dense_shapes() {
    let mut gpu = Gpu::new(MMA);
    for (index, (n, k)) in [
        (4096, 2816),
        (8192, 2816),
        (2816, 4096),
        (2816, 8192),
        (2048, 2816),
        (2112, 2816),
    ]
    .into_iter()
    .enumerate()
    {
        check_shape(&mut gpu, GgmlType::Q4_K, n, k, 0x4b00_0101 + index as u64);
    }
}

#[test]
fn mma_q6_k_2048x2816() {
    let mut gpu = Gpu::new(MMA);
    check_shape(&mut gpu, GgmlType::Q6_K, 2048, 2816, 0x6b00_0101);
}

#[test]
fn mma_q5_0_and_q8_0_down_2816x2112() {
    let mut gpu = Gpu::new(MMA);
    check_shape(&mut gpu, GgmlType::Q5_0, 2816, 2112, 0x5000_0101);
    check_shape(&mut gpu, GgmlType::Q8_0, 2816, 2112, 0x8000_0101);
}

#[test]
fn mma_small_odd_n_every_admitted_type() {
    let mut gpu = Gpu::new(MMA);
    // N = 37 is not a multiple of the 8-row tile (the last tile clamps its
    // loads to row 36 and masks the store). K = 512 and, for the 32-weight
    // types, K = 320 (64·5); Q8_0's block kind also admits K = 96 (32·3,
    // fewer blocks than simdgroups would split).
    for (index, ggml_type) in [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q5_0, GgmlType::Q8_0]
        .into_iter()
        .enumerate()
    {
        check_shape(&mut gpu, ggml_type, 37, 512, 0x0dd2_0000 + index as u64);
        if ggml_type.block_values() == 32 {
            check_shape(&mut gpu, ggml_type, 37, 320, 0x0dd3_0000 + index as u64);
        }
    }
    check_shape(&mut gpu, GgmlType::Q8_0, 37, 96, 0x0dd4_0000);
}

/// Calls `entry` on a request that must be rejected.
struct Rejector<'a> {
    device: &'a MlxDevice,
    registry: KernelRegistry,
    weight: MlxBuffer,
    n: u32,
    entry: Entry,
}

impl Rejector<'_> {
    fn reject(
        &mut self,
        label: &str,
        input: &MlxBuffer,
        output: &MlxBuffer,
        m: u32,
        k: u32,
        t: GgmlType,
    ) {
        let params = GgmlQuantizedMatmulParams {
            m,
            n: self.n,
            k,
            ggml_type: t,
        };
        let mut encoder = self.device.command_encoder().expect("encoder");
        let result = (self.entry)(
            &mut encoder,
            &mut self.registry,
            self.device,
            input,
            &self.weight,
            output,
            &params,
        );
        assert!(result.is_err(), "{label} must be rejected");
        eprintln!("[{label}] {}", result.err().map(|e| e.to_string()).unwrap_or_default());
    }
}

const REJECT_K: usize = 512;
const REJECT_N: usize = 16;

fn rejector(device: &MlxDevice, entry: Entry) -> Rejector<'_> {
    // Large enough for every request below (largest block size × rows).
    let weight = device
        .alloc_buffer(REJECT_N * REJECT_K * 4, DType::U8, vec![REJECT_N * REJECT_K * 4])
        .expect("weight buffer");
    Rejector {
        device,
        registry: KernelRegistry::new(),
        weight,
        n: REJECT_N as u32,
        entry,
    }
}

fn alloc_f32(device: &MlxDevice, elements: usize) -> MlxBuffer {
    device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("f32 buffer")
}

#[test]
fn mma_rejects_unsupported_requests() {
    let device = MlxDevice::new().expect("Metal device");
    let mut r = rejector(&device, MMA);
    let input = alloc_f32(&device, 9 * REJECT_K);
    let output = alloc_f32(&device, 9 * REJECT_N);
    let k32 = REJECT_K as u32;
    r.reject("mma m=0", &input, &output, 0, k32, GgmlType::Q4_K);
    r.reject("mma m=9", &input, &output, 9, k32, GgmlType::Q4_K);
    for t in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q5_K,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q5_1,
        GgmlType::IQ4_NL,
        GgmlType::IQ4_XS,
    ] {
        r.reject(&format!("mma type {t:?}"), &input, &output, 4, k32, t);
    }
    r.reject("mma Q4_K k=128 (not a block multiple)", &input, &output, 4, 128, GgmlType::Q4_K);
    r.reject("mma Q5_0 k=96 (not a multiple of 64)", &input, &output, 4, 96, GgmlType::Q5_0);
    r.reject("mma Q8_0 k=80 (not a block multiple)", &input, &output, 4, 80, GgmlType::Q8_0);
    // An input view 4 bytes into its allocation breaks the 16-byte vector loads.
    let misaligned = input.slice_view(4, 4 * REJECT_K);
    r.reject("mma input offset 4 bytes", &misaligned, &output, 4, k32, GgmlType::Q4_K);
    r.reject("mma output aliases input", &input, &input, 4, k32, GgmlType::Q8_0);
    let short_output = alloc_f32(&device, 4 * REJECT_N - 1);
    r.reject("mma undersized output", &input, &short_output, 4, k32, GgmlType::Q8_0);
}
