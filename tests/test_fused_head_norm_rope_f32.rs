//! `fused_head_norm_rope_f32` — the batched prefill per-head RMS norm + NeoX
//! RoPE (Gemma 4 / Qwen Q and K), on every output the prefill uses: f32
//! natural `[token, head, d]`, BF16 and f32 at the permuted
//! `[head, token, d]` layout.
//!
//! * CPU F64 reference with a K-scaled bound (K = head_dim).
//! * Determinism: repeated dispatches are bitwise identical.
//! * Row independence (batched ≡ serial): a row of a `seq_len = n` batch is
//!   bitwise the row of a `seq_len = 1` dispatch at the same position, so
//!   neither the batch width nor the row grouping per threadgroup reaches
//!   the numbers.
//! * Layout: the permuted outputs are the natural f32 output moved (and
//!   rounded once to BF16 for the BF16 output).
//! * Lane mappings: the float4 and scalar lane mappings agree bitwise.
//!
//! Run on both shader paths:
//! `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test test_fused_head_norm_rope_f32`
//! and the same with `MLX_PRECOMPILED_METALLIB=0`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::fused_head_norm_rope::{
    dispatch_fused_head_norm_rope_batch_f32,
    dispatch_fused_head_norm_rope_batch_f32_with_bf16_f32_perm,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
struct Shape {
    n_heads: u32,
    head_dim: u32,
    half_rope: u32,
    seq_len: u32,
    weight: bool,
    /// Gemma 4 global layers: proportional RoPE, 1.0 for the first
    /// `half_rope / 4` pairs and 1e30 (no rotation) for the rest.
    freq_factors: bool,
    theta: f32,
}

impl Shape {
    fn elements(self) -> usize {
        (self.n_heads * self.head_dim * self.seq_len) as usize
    }
}

/// Gemma 4 sliding Q/K, global Q/K, plus partial rotary, no weight, a
/// non-power-of-two head, a head below the simdgroup width and one above
/// 512 (several elements per reduction slot).
fn shapes(seq_len: u32) -> Vec<Shape> {
    let s = |n_heads, head_dim, half_rope, weight, freq_factors, theta| Shape {
        n_heads,
        head_dim,
        half_rope,
        seq_len,
        weight,
        freq_factors,
        theta,
    };
    vec![
        s(16, 256, 128, true, false, 10_000.0),
        s(8, 256, 128, true, false, 10_000.0),
        s(16, 512, 256, true, true, 1_000_000.0),
        s(2, 512, 256, true, true, 1_000_000.0),
        s(4, 256, 32, true, false, 10_000_000.0),
        s(3, 128, 64, false, false, 10_000.0),
        s(5, 80, 16, true, false, 10_000.0),
        s(4, 16, 8, true, false, 10_000.0),
        s(2, 1024, 512, true, false, 10_000.0),
    ]
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: u32) -> u32 {
        (self.next() % u64::from(bound)) as u32
    }

    /// Mantissa in [-1, 1] times 2^[-6, 5]: rows with a wide dynamic range.
    fn value(&mut self) -> f32 {
        let mantissa = (self.below(2_000_001) as f32 - 1_000_000.0) / 1_000_000.0;
        mantissa * 2f32.powi(self.below(12) as i32 - 6)
    }
}

struct Inputs {
    input: Vec<f32>,
    weight: Vec<f32>,
    positions: Vec<u32>,
    freq_factors: Vec<f32>,
}

fn inputs(shape: Shape, seed: u64) -> Inputs {
    let mut rng = XorShift(seed);
    let input = (0..shape.elements()).map(|_| rng.value()).collect();
    let weight = (0..shape.head_dim)
        .map(|_| 0.5 + rng.below(1001) as f32 / 1000.0)
        .collect();
    // Prefill positions up to the 4096-token chunk range, first one at 0.
    let positions = (0..shape.seq_len)
        .map(|t| if t == 0 { 0 } else { rng.below(4096) })
        .collect();
    let mut freq_factors = vec![1e30f32; (shape.head_dim / 2) as usize];
    freq_factors[..(shape.half_rope / 4) as usize].fill(1.0);
    Inputs {
        input,
        weight,
        positions,
        freq_factors,
    }
}

fn upload<T: bytemuck::Pod>(device: &MlxDevice, values: &[T], dtype: DType) -> MlxBuffer {
    let bytes = std::mem::size_of_val(values);
    let mut buffer = device
        .alloc_buffer(bytes, dtype, vec![values.len()])
        .expect("alloc");
    buffer
        .as_mut_slice::<T>()
        .expect("mut slice")
        .copy_from_slice(values);
    buffer
}

/// Output buffer pre-filled with a sentinel byte pattern, so an element the
/// kernel fails to write cannot match a reference.
fn sentinel(device: &MlxDevice, elements: usize, dtype: DType) -> MlxBuffer {
    let bytes = elements * dtype.size_of();
    let mut buffer = device
        .alloc_buffer(bytes, dtype, vec![elements])
        .expect("alloc");
    buffer.as_mut_slice::<u8>().expect("bytes").fill(0x7F);
    buffer
}

struct Outputs {
    natural: Vec<f32>,
    bf16_perm: Vec<u16>,
    f32_perm: Vec<f32>,
}

/// The prefill Q entry point: f32 natural + BF16 permuted + f32 permuted.
fn run(device: &MlxDevice, registry: &mut KernelRegistry, shape: Shape, data: &Inputs) -> Outputs {
    run_with_input_offset(device, registry, shape, data, false)
}

/// `shifted_input`: bind the input as a view 4 bytes into its allocation,
/// which rules out the float4 lane mapping (scalar lanes instead).
fn run_with_input_offset(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    shape: Shape,
    data: &Inputs,
    shifted_input: bool,
) -> Outputs {
    let elements = shape.elements();
    let input = if shifted_input {
        let mut padded = vec![0.0f32];
        padded.extend_from_slice(&data.input);
        upload(device, &padded, DType::F32).slice_view(4, elements)
    } else {
        upload(device, &data.input, DType::F32)
    };
    let weight = upload(device, &data.weight, DType::F32);
    let positions = upload(device, &data.positions, DType::U32);
    let freq_factors = upload(device, &data.freq_factors, DType::F32);
    let natural = sentinel(device, elements, DType::F32);
    let bf16_perm = sentinel(device, elements, DType::BF16);
    let f32_perm = sentinel(device, elements, DType::F32);
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_fused_head_norm_rope_batch_f32_with_bf16_f32_perm(
        &mut encoder,
        registry,
        device.metal_device(),
        &input,
        &natural,
        Some(&bf16_perm),
        Some(&f32_perm),
        shape.weight.then_some(&weight),
        &positions,
        shape.freq_factors.then_some(&freq_factors),
        shape.n_heads,
        shape.head_dim,
        shape.half_rope,
        shape.seq_len,
        EPS,
        shape.theta,
        true,
    )
    .expect("dispatch");
    encoder.commit_and_wait().expect("commit");
    Outputs {
        natural: natural.as_slice::<f32>().expect("read").to_vec(),
        bf16_perm: bf16_perm.as_slice::<u16>().expect("read").to_vec(),
        f32_perm: f32_perm.as_slice::<f32>().expect("read").to_vec(),
    }
}

fn registry() -> (MlxDevice, KernelRegistry) {
    (MlxDevice::new().expect("device"), KernelRegistry::new())
}

/// F64 reference of one row: (outputs, RoPE angle per element).
fn reference_row(shape: Shape, data: &Inputs, row: usize) -> (Vec<f64>, Vec<f64>) {
    let d = shape.head_dim as usize;
    let half = d / 2;
    let x = &data.input[row * d..(row + 1) * d];
    let mean = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
    let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
    let y: Vec<f64> = (0..d)
        .map(|i| {
            let w = if shape.weight {
                f64::from(data.weight[i])
            } else {
                1.0
            };
            f64::from(x[i]) * inv * w
        })
        .collect();
    let pos = f64::from(data.positions[row / shape.n_heads as usize]);
    let mut out = y.clone();
    let mut angle = vec![0.0; d];
    for i in 0..shape.half_rope as usize {
        let mut a = pos / f64::from(shape.theta).powf(2.0 * i as f64 / d as f64);
        if shape.freq_factors {
            a /= f64::from(data.freq_factors[i]);
        }
        let (sin, cos) = a.sin_cos();
        out[i] = y[i] * cos - y[i + half] * sin;
        out[i + half] = y[i + half] * cos + y[i] * sin;
        angle[i] = a;
        angle[i + half] = a;
    }
    (out, angle)
}

/// Per-element bound against the F64 reference, for the pair (i, i+half)
/// with normalized magnitude `m = |y_i| + |y_{i+half}|` and angle `a`:
///
/// `m * (K * eps + 16 * eps + 64 * eps * |a|) + 2^-20 * m`
///
/// * `K * eps` (K = head_dim): the f32 sum of squares over K terms, any order;
/// * `16 * eps`: rsqrt, the weight and the rotation products;
/// * `64 * eps * |a|`: the f32 angle `pos / pow(theta, 2i/K) / ff` (fast
///   `pow` exponent error grows with |a|);
/// * `2^-20`: fast-math `sin`/`cos` absolute error.
fn bound(m: f64, angle: f64, k: u32) -> f64 {
    let eps = f64::from(f32::EPSILON) / 2.0;
    m * (f64::from(k) * eps + 16.0 * eps + 64.0 * eps * angle.abs() + 2f64.powi(-20))
}

#[test]
fn f64_reference_k_scaled_bound() {
    let (device, mut registry) = registry();
    for (index, shape) in shapes(37).into_iter().enumerate() {
        let data = inputs(shape, 0x5EED + index as u64);
        let got = run(&device, &mut registry, shape, &data);
        let d = shape.head_dim as usize;
        let half = d / 2;
        let mut worst = 0.0f64;
        for row in 0..(shape.n_heads * shape.seq_len) as usize {
            let (want, angle) = reference_row(shape, &data, row);
            let y_mag = |i: usize| {
                let (yi, yj) = (want[i], want[i + half]);
                // |rotation| preserves the pair norm; bound with the 1-norm.
                (yi * yi + yj * yj).sqrt() * std::f64::consts::SQRT_2
            };
            for i in 0..2 * half {
                let pair = if i < half { i } else { i - half };
                let limit = bound(y_mag(pair), angle[i], shape.head_dim);
                let err = (f64::from(got.natural[row * d + i]) - want[i]).abs();
                assert!(
                    err <= limit,
                    "{shape:?} row {row} elem {i}: got {} want {} err {err:e} > bound {limit:e}",
                    got.natural[row * d + i],
                    want[i]
                );
                worst = worst.max(err / limit);
            }
        }
        println!("{shape:?}: max err / bound = {worst:.3}");
    }
}

#[test]
fn deterministic_and_row_independent() {
    let (device, mut registry) = registry();
    // 29 tokens: the row count is not a multiple of the rows per threadgroup.
    for (index, shape) in shapes(29).into_iter().enumerate() {
        let data = inputs(shape, 0xD0 + index as u64);
        let first = run(&device, &mut registry, shape, &data);
        for _ in 0..2 {
            let again = run(&device, &mut registry, shape, &data);
            assert_eq!(
                bits(&first.natural),
                bits(&again.natural),
                "{shape:?}: f32 natural not deterministic"
            );
            assert_eq!(
                first.bf16_perm, again.bf16_perm,
                "{shape:?}: bf16 not deterministic"
            );
            assert_eq!(
                bits(&first.f32_perm),
                bits(&again.f32_perm),
                "{shape:?}: f32 perm not deterministic"
            );
        }
        let row_elems = (shape.n_heads * shape.head_dim) as usize;
        for t in 0..shape.seq_len as usize {
            let single_shape = Shape {
                seq_len: 1,
                ..shape
            };
            let single = Inputs {
                input: data.input[t * row_elems..(t + 1) * row_elems].to_vec(),
                weight: data.weight.clone(),
                positions: vec![data.positions[t]],
                freq_factors: data.freq_factors.clone(),
            };
            let one = run(&device, &mut registry, single_shape, &single);
            assert_eq!(
                bits(&one.natural),
                bits(&first.natural[t * row_elems..(t + 1) * row_elems]),
                "{shape:?}: token {t} differs between seq_len=1 and the batch"
            );
        }
    }
}

#[test]
fn permuted_outputs_are_the_natural_output_moved() {
    let (device, mut registry) = registry();
    for (index, shape) in shapes(23).into_iter().enumerate() {
        let data = inputs(shape, 0x1A + index as u64);
        let got = run(&device, &mut registry, shape, &data);
        let (h, d, n) = (
            shape.n_heads as usize,
            shape.head_dim as usize,
            shape.seq_len as usize,
        );
        // Odd head_dim never occurs; every element of the row is written.
        for t in 0..n {
            for head in 0..h {
                for i in 0..d {
                    let natural = got.natural[(t * h + head) * d + i];
                    let perm = head * n * d + t * d + i;
                    assert_eq!(
                        got.f32_perm[perm].to_bits(),
                        natural.to_bits(),
                        "{shape:?}: f32 perm token {t} head {head} elem {i}"
                    );
                    assert_eq!(
                        got.bf16_perm[perm],
                        half::bf16::from_f32(natural).to_bits(),
                        "{shape:?}: bf16 perm token {t} head {head} elem {i}"
                    );
                }
            }
        }
    }
}

/// The plain batched entry point (f32 natural only) writes the same f32
/// values as the prefill entry point.
#[test]
fn plain_batch_entry_matches() {
    let (device, mut registry) = registry();
    for (index, shape) in shapes(19).into_iter().enumerate() {
        let data = inputs(shape, 0xB0 + index as u64);
        let full = run(&device, &mut registry, shape, &data);
        let input = upload(&device, &data.input, DType::F32);
        let weight = upload(&device, &data.weight, DType::F32);
        let positions = upload(&device, &data.positions, DType::U32);
        let freq_factors = upload(&device, &data.freq_factors, DType::F32);
        let out = sentinel(&device, shape.elements(), DType::F32);
        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_fused_head_norm_rope_batch_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &input,
            &out,
            shape.weight.then_some(&weight),
            &positions,
            shape.freq_factors.then_some(&freq_factors),
            shape.n_heads,
            shape.head_dim,
            shape.half_rope,
            shape.seq_len,
            EPS,
            shape.theta,
        )
        .expect("dispatch");
        encoder.commit_and_wait().expect("commit");
        assert_eq!(
            bits(out.as_slice::<f32>().expect("read")),
            bits(&full.natural),
            "{shape:?}"
        );
    }
}

/// head_dim % 256 == 0 rows run the float4 lane mapping when the buffers are
/// 16-byte aligned and the scalar one otherwise; both evaluate the same f32
/// operation sequence, so every output is bitwise identical.
#[test]
fn float4_and_scalar_lanes_agree() {
    let (device, mut registry) = registry();
    for (index, shape) in shapes(31)
        .into_iter()
        .filter(|shape| shape.head_dim % 256 == 0)
        .enumerate()
    {
        let data = inputs(shape, 0x4A + index as u64);
        let vec4 = run_with_input_offset(&device, &mut registry, shape, &data, false);
        let scalar = run_with_input_offset(&device, &mut registry, shape, &data, true);
        assert_eq!(
            bits(&vec4.natural),
            bits(&scalar.natural),
            "{shape:?}: f32 natural"
        );
        assert_eq!(vec4.bf16_perm, scalar.bf16_perm, "{shape:?}: bf16 perm");
        assert_eq!(
            bits(&vec4.f32_perm),
            bits(&scalar.f32_perm),
            "{shape:?}: f32 perm"
        );
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
