//! The dense quantized matmul where the Metal tensor API is unavailable.
//!
//! `dispatch_mm_impl` asks one probe (`probe_tensor_mm`, the pipeline
//! `kernel_mul_mm_q4_0_tensor_f32` whatever the weight's type) before it names
//! a kernel, so an unavailable tensor API must send every row count to the
//! simdgroup kernel of the weight's type: the rows that otherwise take the
//! large tile, and the 32 rows that otherwise take tensor-v1. The route and
//! its bits are pinned against the policy that forces the simdgroup kernels.
//!
//! Two hosts have no tensor API (`dense_mm_capability`): a runtime compiler
//! without `<metal_tensor>`, and a device outside the Metal 4 GPU family. The
//! first is reproduced on any host by a probe source that stops the compiler
//! with that diagnostic. The second cannot be reproduced on a device that
//! supports Metal 4, so `default_policy_dense_mm_encodes_on_this_device`
//! runs the unmodified registry: CI runs it on the hosted `macos-26` GPU,
//! which has the header and no Metal 4 family, with
//! `MLX_DENSE_MM_TEST_EXPECT_NO_TENSOR=1` so that a runner that gains the
//! tensor API cannot turn the gate into a tensor run unnoticed.

#![allow(clippy::expect_used, clippy::panic)]

use super::quantized_matmul_ggml::{
    quantized_matmul_ggml_with_policy, GgmlQuantizedMatmulParams, GgmlType,
};
use crate::{DType, GgmlRoutingPolicy, GgmlTensorMmPreference, KernelRegistry, MlxDevice};

const TENSOR_PROBE: &str = "kernel_mul_mm_q4_0_tensor_f32";
/// The runtime compiler's diagnostic on an SDK without the tensor header.
const MISSING_HEADER_PROBE_SOURCE: &str = "#error fatal error: 'metal_tensor' file not found\n";

const N: usize = 70;
const K: usize = 256;
/// Below, at and above the 32-row tensor-v1 tile and the 128-row large tile.
const ROWS: [usize; 5] = [9, 32, 33, 64, 130];
const TYPES: [GgmlType; 5] = [
    GgmlType::Q4_0,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
];

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

/// Random blocks whose f16 scales are small and finite, so every weight is.
fn weight_bytes(ggml_type: GgmlType, seed: u64) -> Vec<u8> {
    let block = ggml_type.block_bytes() as usize;
    let mut bytes = vec![0u8; N * (K / ggml_type.block_values() as usize) * block];
    let mut rng = Xorshift(seed | 1);
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    let scale_offsets: &[usize] = match ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 => &[0],
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        other => panic!("no weight generator for {other:?}"),
    };
    for block_bytes in bytes.chunks_exact_mut(block) {
        for &offset in scale_offsets {
            let scale = half::f16::from_f64(0.002 + 0.006 * rng.unit()).to_bits();
            block_bytes[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
        }
    }
    bytes
}

fn inputs(m: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift(seed | 1);
    (0..m * K).map(|_| (rng.unit() - 0.5) as f32).collect()
}

/// One dense matmul: the kernel it encoded and its output bits.
fn dense_mm(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    policy: &GgmlRoutingPolicy,
    ggml_type: GgmlType,
    m: usize,
) -> (String, Vec<u32>) {
    let seed = 0xC1_0835 ^ ((ggml_type.block_bytes() as u64) << 16) ^ m as u64;
    let weights = weight_bytes(ggml_type, seed);
    let rows = inputs(m, seed.rotate_left(17));
    let mut weight = device
        .alloc_buffer(weights.len(), DType::U8, vec![weights.len()])
        .expect("weight");
    weight
        .as_mut_slice::<u8>()
        .expect("weight bytes")
        .copy_from_slice(&weights);
    let mut input = device
        .alloc_buffer(m * K * 4, DType::F32, vec![m, K])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("input rows")
        .copy_from_slice(&rows);
    let output = device
        .alloc_buffer(m * N * 4, DType::F32, vec![m, N])
        .expect("output");
    let label = format!("{ggml_type:?} at {m} rows");
    let mut encoder = device.command_encoder().expect("encoder");
    encoder.start_encoded_dispatch_receipt(1).expect("receipt");
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        registry,
        device,
        &input,
        &weight,
        &output,
        &GgmlQuantizedMatmulParams {
            m: m as u32,
            n: N as u32,
            k: K as u32,
            ggml_type,
        },
        policy,
    )
    .unwrap_or_else(|error| panic!("{label}: {error}"));
    let receipt = encoder
        .take_encoded_dispatch_receipt()
        .expect("receipt result");
    assert_eq!(receipt.len(), 1, "{label}: one dispatch");
    encoder
        .commit_and_wait()
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let kernel = receipt[0]
        .pipeline_label
        .split('|')
        .next()
        .expect("kernel name")
        .to_string();
    let bits = output
        .as_slice::<f32>()
        .expect("output rows")
        .iter()
        .map(|value| value.to_bits())
        .collect();
    (kernel, bits)
}

fn simdgroup_policy() -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        dense_tensor_mm: GgmlTensorMmPreference::ForceSimd,
        ..GgmlRoutingPolicy::default()
    }
}

#[test]
fn missing_tensor_header_sends_every_row_count_to_the_simdgroup_kernel() {
    let device = MlxDevice::new().expect("device");
    let mut unavailable = KernelRegistry::new();
    unavailable.register_source(TENSOR_PROBE, MISSING_HEADER_PROBE_SOURCE);
    let mut simdgroup = KernelRegistry::new();
    for ggml_type in TYPES {
        for m in ROWS {
            let label = format!("{ggml_type:?} at {m} rows");
            let (kernel, bits) = dense_mm(
                &device,
                &mut unavailable,
                &GgmlRoutingPolicy::default(),
                ggml_type,
                m,
            );
            assert_eq!(kernel, ggml_type.mm_kernel_name(), "{label}");
            let (forced_kernel, forced_bits) =
                dense_mm(&device, &mut simdgroup, &simdgroup_policy(), ggml_type, m);
            assert_eq!(forced_kernel, ggml_type.mm_kernel_name(), "{label}");
            assert!(
                bits == forced_bits,
                "{label}: differs from the simdgroup kernel"
            );
        }
    }
}

#[test]
fn default_policy_dense_mm_encodes_on_this_device() {
    let device = MlxDevice::new().expect("device");
    let mut auto = KernelRegistry::new();
    let mut simdgroup = KernelRegistry::new();
    let mut fell_back = 0;
    for ggml_type in TYPES {
        for m in ROWS {
            let label = format!("{ggml_type:?} at {m} rows");
            let (kernel, bits) = dense_mm(
                &device,
                &mut auto,
                &GgmlRoutingPolicy::default(),
                ggml_type,
                m,
            );
            if kernel != ggml_type.mm_kernel_name() {
                continue;
            }
            fell_back += 1;
            let (_, forced_bits) =
                dense_mm(&device, &mut simdgroup, &simdgroup_policy(), ggml_type, m);
            assert!(
                bits == forced_bits,
                "{label}: differs from the simdgroup kernel"
            );
        }
    }
    let cases = TYPES.len() * ROWS.len();
    let expected = if std::env::var("MLX_DENSE_MM_TEST_EXPECT_NO_TENSOR").as_deref() == Ok("1") {
        fell_back == cases
    } else {
        fell_back == 0 || fell_back == cases
    };
    assert!(
        expected,
        "{}: {fell_back} of {cases} dense matmuls took the simdgroup kernel",
        device.name()
    );
    eprintln!(
        "{}: Metal 4 family {}, dense matmul route: {}",
        device.name(),
        device
            .metal_device()
            .supports_family(metal::MTLGPUFamily::Metal4),
        if fell_back == 0 { "tensor" } else { "simdgroup" }
    );
}
