//! `rms_norm_no_scale_f32_dual_perm` — the Gemma 4 prefill V unit norm:
//! f32 at the natural `[token, head, d]` layout (KV-cache copy source) and
//! BF16 at the permuted `[head, token, d]` layout (attention input).
//!
//! * CPU F64 reference with a K-scaled bound (K = dim).
//! * Determinism and row independence (batched ≡ serial).
//! * Geometry independence: `dispatch_rms_norm_no_scale_f32_dual_perm`
//!   (rows packed per threadgroup) and the one-threadgroup-per-row raw encode
//!   hf2q uses write bitwise identical outputs.
//! * Layout: the BF16 output is the f32 output moved and rounded once.
//!
//! Run on both shader paths: `MLX_PRECOMPILED_METALLIB=1` and `=0`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::encode_helpers::{
    as_bytes, encode_threadgroups_with_args_and_shared, KernelArg,
};
use mlx_native::ops::rms_norm::dispatch_rms_norm_no_scale_f32_dual_perm;
use mlx_native::{DType, KernelRegistry, MTLSize, MlxBuffer, MlxDevice};

const EPS: f32 = 1e-6;

/// (n_heads, dim): Gemma 4 sliding and global V, then a non-float4 dim, a
/// non-power-of-two dim, a dim below the simdgroup width and one with several
/// elements per reduction slot.
const SHAPES: [(u32, u32); 6] = [(8, 256), (2, 512), (4, 128), (3, 80), (2, 16), (1, 1024)];

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

    /// Mantissa in [-1, 1] times 2^[-6, 5].
    fn value(&mut self) -> f32 {
        let mantissa = ((self.next() % 2_000_001) as f32 - 1_000_000.0) / 1_000_000.0;
        mantissa * 2f32.powi((self.next() % 12) as i32 - 6)
    }
}

fn upload<T: bytemuck::Pod>(device: &MlxDevice, values: &[T], dtype: DType) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(std::mem::size_of_val(values), dtype, vec![values.len()])
        .expect("alloc");
    buffer
        .as_mut_slice::<T>()
        .expect("mut slice")
        .copy_from_slice(values);
    buffer
}

fn sentinel(device: &MlxDevice, elements: usize, dtype: DType) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * dtype.size_of(), dtype, vec![elements])
        .expect("alloc");
    buffer.as_mut_slice::<u8>().expect("bytes").fill(0x7F);
    buffer
}

struct Outputs {
    natural: Vec<u32>,
    bf16_perm: Vec<u16>,
}

/// `packed`: the mlx-native dispatcher; otherwise hf2q's raw encode (one
/// threadgroup of min(256, next_pow2(dim)) threads per row).
fn run(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &[f32],
    n_heads: u32,
    seq_len: u32,
    dim: u32,
    packed: bool,
) -> Outputs {
    let elements = input.len();
    let input = upload(device, input, DType::F32);
    let params = upload(device, &[EPS, dim as f32], DType::F32);
    let natural = sentinel(device, elements, DType::F32);
    let bf16_perm = sentinel(device, elements, DType::BF16);
    let mut encoder = device.command_encoder().expect("encoder");
    if packed {
        dispatch_rms_norm_no_scale_f32_dual_perm(
            &mut encoder,
            registry,
            device.metal_device(),
            &input,
            &natural,
            &bf16_perm,
            &params,
            n_heads,
            seq_len,
            dim,
        )
        .expect("dispatch");
    } else {
        let pipeline = registry
            .get_pipeline("rms_norm_no_scale_f32_dual_perm", device.metal_device())
            .expect("pipeline");
        let tg = u64::from(dim.next_power_of_two().min(256));
        let aux: [u32; 2] = [n_heads, seq_len];
        encode_threadgroups_with_args_and_shared(
            &mut encoder,
            pipeline,
            &[
                (0, KernelArg::Buffer(&input)),
                (1, KernelArg::Buffer(&natural)),
                (2, KernelArg::Buffer(&params)),
                (3, KernelArg::Buffer(&bf16_perm)),
                (4, KernelArg::Bytes(as_bytes(&aux))),
            ],
            &[(0, tg * 4)],
            MTLSize::new(u64::from(n_heads * seq_len), 1, 1),
            MTLSize::new(tg, 1, 1),
        );
    }
    encoder.commit_and_wait().expect("commit");
    Outputs {
        natural: natural.as_slice::<u32>().expect("read").to_vec(),
        bf16_perm: bf16_perm.as_slice::<u16>().expect("read").to_vec(),
    }
}

fn random_rows(elements: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift(seed);
    (0..elements).map(|_| rng.value()).collect()
}

#[test]
fn f64_reference_k_scaled_bound() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let seq_len = 37;
    let eps64 = f64::from(f32::EPSILON) / 2.0;
    for (index, &(n_heads, dim)) in SHAPES.iter().enumerate() {
        let input = random_rows((n_heads * dim * seq_len) as usize, 0xF64 + index as u64);
        let got = run(&device, &mut registry, &input, n_heads, seq_len, dim, true);
        let d = dim as usize;
        let mut worst = 0.0f64;
        for (row, x) in input.chunks_exact(d).enumerate() {
            let mean = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
            let inv = 1.0 / (mean + f64::from(EPS)).sqrt();
            for (i, &v) in x.iter().enumerate() {
                let want = f64::from(v) * inv;
                let got = f64::from(f32::from_bits(got.natural[row * d + i]));
                // K * eps: f32 sum of squares over K = dim terms; 16 * eps:
                // the division, eps add, fast rsqrt and the final product.
                let bound = want.abs() * (f64::from(dim) * eps64 + 16.0 * eps64)
                    + f64::from(f32::MIN_POSITIVE);
                let err = (got - want).abs();
                assert!(
                    err <= bound,
                    "heads {n_heads} dim {dim} row {row} elem {i}: got {got} want {want} err {err:e} > {bound:e}"
                );
                worst = worst.max(err / bound);
            }
        }
        println!("heads {n_heads} dim {dim}: max err / bound = {worst:.3}");
    }
}

#[test]
fn deterministic_row_independent_and_geometry_independent() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    // 29 tokens: the row count is not a multiple of the rows per threadgroup.
    let seq_len = 29;
    for (index, &(n_heads, dim)) in SHAPES.iter().enumerate() {
        let input = random_rows((n_heads * dim * seq_len) as usize, 0xDE7 + index as u64);
        let first = run(&device, &mut registry, &input, n_heads, seq_len, dim, true);
        for _ in 0..2 {
            let again = run(&device, &mut registry, &input, n_heads, seq_len, dim, true);
            assert_eq!(
                first.natural, again.natural,
                "heads {n_heads} dim {dim}: f32 not deterministic"
            );
            assert_eq!(
                first.bf16_perm, again.bf16_perm,
                "heads {n_heads} dim {dim}: bf16 not deterministic"
            );
        }
        let raw = run(&device, &mut registry, &input, n_heads, seq_len, dim, false);
        assert_eq!(
            first.natural, raw.natural,
            "heads {n_heads} dim {dim}: f32 depends on the geometry"
        );
        assert_eq!(
            first.bf16_perm, raw.bf16_perm,
            "heads {n_heads} dim {dim}: bf16 depends on the geometry"
        );

        let token_elems = (n_heads * dim) as usize;
        for t in 0..seq_len as usize {
            let one = run(
                &device,
                &mut registry,
                &input[t * token_elems..(t + 1) * token_elems],
                n_heads,
                1,
                dim,
                true,
            );
            assert_eq!(
                one.natural,
                first.natural[t * token_elems..(t + 1) * token_elems],
                "heads {n_heads} dim {dim}: token {t} differs between seq_len=1 and the batch"
            );
        }
    }
}

#[test]
fn bf16_output_is_the_f32_output_permuted() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let seq_len = 23usize;
    for (index, &(n_heads, dim)) in SHAPES.iter().enumerate() {
        let (h, d) = (n_heads as usize, dim as usize);
        let input = random_rows(h * d * seq_len, 0xB16 + index as u64);
        let got = run(
            &device,
            &mut registry,
            &input,
            n_heads,
            seq_len as u32,
            dim,
            true,
        );
        for t in 0..seq_len {
            for head in 0..h {
                for i in 0..d {
                    let natural = f32::from_bits(got.natural[(t * h + head) * d + i]);
                    assert_eq!(
                        got.bf16_perm[head * seq_len * d + t * d + i],
                        half::bf16::from_f32(natural).to_bits(),
                        "heads {n_heads} dim {dim}: token {t} head {head} elem {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn rejects_misaligned_float4_bindings() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let (n_heads, seq_len, dim) = (2u32, 3u32, 256u32);
    let elements = (n_heads * seq_len * dim) as usize;
    let backing = upload(&device, &vec![1.0f32; elements + 1], DType::F32);
    let shifted = backing.slice_view(4, elements);
    let params = upload(&device, &[EPS, dim as f32], DType::F32);
    let natural = sentinel(&device, elements, DType::F32);
    let bf16_perm = sentinel(&device, elements, DType::BF16);
    let mut encoder = device.command_encoder().expect("encoder");
    let result = dispatch_rms_norm_no_scale_f32_dual_perm(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &shifted,
        &natural,
        &bf16_perm,
        &params,
        n_heads,
        seq_len,
        dim,
    );
    assert!(
        result.is_err(),
        "a 4-byte-offset input must be rejected for dim % 256 == 0"
    );
}
