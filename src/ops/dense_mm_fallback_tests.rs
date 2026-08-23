#![allow(clippy::expect_used, clippy::panic)]

use half::{bf16, f16};

use super::dense_mm_bf16::{dense_matmul_bf16_f32_with_backend, DenseMmBf16F32Params};
use super::dense_mm_capability::{is_unavailable_tensor_header, DenseMmBackend};
use super::dense_mm_f16::{dense_matmul_f16_f32_with_backend, DenseMmF16F32Params};
use super::dense_mm_f32_f32::{dense_matmul_f32_f32_with_backend, DenseMmF32F32Params};
use crate::{DType, KernelRegistry, MlxDevice};

const M: u32 = 35;
const N: u32 = 67;
const K: u32 = 72;
const SRC0_BATCH: u32 = 2;
const SRC1_BATCH: u32 = 4;

fn values(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 32) as u32) as f64 / u32::MAX as f64;
            (unit * 2.0 - 1.0) as f32
        })
        .collect()
}

fn assert_close(label: &str, tensor: &[f32], fallback: &[f32], tolerance: f32) {
    assert_eq!(tensor.len(), fallback.len());
    let max_abs = tensor
        .iter()
        .zip(fallback)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= tolerance,
        "{label}: tensor/fallback max_abs {max_abs} exceeds {tolerance}"
    );
}

fn assert_short_bf16_backend_matches_cpu(k: u32, backend: DenseMmBackend) {
    let device = MlxDevice::new().expect("device");
    let m = 5u32;
    let n = 17u32;
    let weights_f32 = values(0x5100 + u64::from(k), (n * k) as usize);
    let weights: Vec<u16> = weights_f32
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let input = values(0xB170 + u64::from(k), (m * k) as usize);
    let mut expected = vec![0.0f32; (m * n) as usize];
    for row in 0..m as usize {
        for output in 0..n as usize {
            let mut sum = 0.0f32;
            for depth in 0..k as usize {
                sum += bf16::from_bits(weights[output * k as usize + depth]).to_f32()
                    * input[row * k as usize + depth];
            }
            expected[row * n as usize + output] = sum;
        }
    }

    let mut weight_buffer = device
        .alloc_buffer(
            weights.len() * 2,
            DType::BF16,
            vec![1, n as usize, k as usize],
        )
        .expect("weight buffer");
    weight_buffer
        .as_mut_slice::<u16>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut input_buffer = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![1, m as usize, k as usize])
        .expect("input buffer");
    input_buffer
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&input);
    let output_buffer = device
        .alloc_buffer(
            expected.len() * 4,
            DType::F32,
            vec![1, m as usize, n as usize],
        )
        .expect("output buffer");
    let params = DenseMmBf16F32Params {
        m,
        n,
        k,
        src0_batch: 1,
        src1_batch: 1,
    };
    let mut registry = KernelRegistry::new();
    let mut encoder = device.command_encoder().expect("encoder");
    if let Err(error) = dense_matmul_bf16_f32_with_backend(
        &mut encoder,
        &mut registry,
        &device,
        &weight_buffer,
        &input_buffer,
        &output_buffer,
        &params,
        backend,
    ) {
        assert!(
            backend == DenseMmBackend::TensorRequired && is_unavailable_tensor_header(&error),
            "short-K {backend:?} dispatch failed: {error}"
        );
        return;
    }
    encoder.commit_and_wait().expect("completion");
    let actual = output_buffer.as_slice::<f32>().expect("output slice");
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 5e-2,
        "short-K {backend:?} K={k} max_abs {max_abs} exceeds 0.05"
    );
}

#[test]
fn bf16_short_reductions_are_correct_on_forced_backends() {
    for k in [1, 2, 3, 4, 8, 16] {
        assert_short_bf16_backend_matches_cpu(k, DenseMmBackend::FallbackRequired);
    }
    for k in [4, 8, 16] {
        assert_short_bf16_backend_matches_cpu(k, DenseMmBackend::TensorRequired);
    }
}

#[test]
fn bf16_tensor_and_tiled_fallback_preserve_numerical_contract() {
    let device = MlxDevice::new().expect("device");
    let weights_f32 = values(11, (SRC0_BATCH * N * K) as usize);
    let weights: Vec<u16> = weights_f32
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let input = values(12, (SRC1_BATCH * M * K) as usize);

    let mut weight_buffer = device
        .alloc_buffer(weights.len() * 2, DType::BF16, vec![weights.len()])
        .expect("weight buffer");
    weight_buffer
        .as_mut_slice::<u16>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut input_buffer = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![input.len()])
        .expect("input buffer");
    input_buffer
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&input);
    let output_len = (SRC1_BATCH * M * N) as usize;
    let tensor_output = device
        .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
        .expect("tensor output");
    let fallback_output = device
        .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
        .expect("fallback output");
    let params = DenseMmBf16F32Params {
        m: M,
        n: N,
        k: K,
        src0_batch: SRC0_BATCH,
        src1_batch: SRC1_BATCH,
    };

    let mut fallback_registry = KernelRegistry::new();
    let mut fallback_encoder = device.command_encoder().expect("fallback encoder");
    dense_matmul_bf16_f32_with_backend(
        &mut fallback_encoder,
        &mut fallback_registry,
        &device,
        &weight_buffer,
        &input_buffer,
        &fallback_output,
        &params,
        DenseMmBackend::FallbackRequired,
    )
    .expect("fallback dispatch");
    fallback_encoder
        .commit_and_wait()
        .expect("fallback completion");

    let mut tensor_registry = KernelRegistry::new();
    let mut tensor_encoder = device.command_encoder().expect("tensor encoder");
    if let Err(error) = dense_matmul_bf16_f32_with_backend(
        &mut tensor_encoder,
        &mut tensor_registry,
        &device,
        &weight_buffer,
        &input_buffer,
        &tensor_output,
        &params,
        DenseMmBackend::TensorRequired,
    ) {
        assert!(is_unavailable_tensor_header(&error), "{error}");
        return;
    }
    tensor_encoder.commit_and_wait().expect("tensor completion");

    assert_close(
        "bf16",
        tensor_output.as_slice::<f32>().expect("tensor slice"),
        fallback_output.as_slice::<f32>().expect("fallback slice"),
        1e-4,
    );
}

#[test]
fn f16_tensor_and_tiled_fallback_preserve_numerical_contract() {
    let device = MlxDevice::new().expect("device");
    let weights_f32 = values(21, (SRC0_BATCH * N * K) as usize);
    let weights: Vec<u16> = weights_f32
        .iter()
        .map(|&value| f16::from_f32(value).to_bits())
        .collect();
    let input = values(22, (SRC1_BATCH * M * K) as usize);
    let mut weight_buffer = device
        .alloc_buffer(weights.len() * 2, DType::F16, vec![weights.len()])
        .expect("weight buffer");
    weight_buffer
        .as_mut_slice::<u16>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut input_buffer = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![input.len()])
        .expect("input buffer");
    input_buffer
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&input);
    let output_len = (SRC1_BATCH * M * N) as usize;
    let tensor_output = device
        .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
        .expect("tensor output");
    let fallback_output = device
        .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
        .expect("fallback output");
    let params = DenseMmF16F32Params {
        m: M,
        n: N,
        k: K,
        src0_batch: SRC0_BATCH,
        src1_batch: SRC1_BATCH,
    };

    let mut fallback_registry = KernelRegistry::new();
    let mut fallback_encoder = device.command_encoder().expect("fallback encoder");
    dense_matmul_f16_f32_with_backend(
        &mut fallback_encoder,
        &mut fallback_registry,
        &device,
        &weight_buffer,
        &input_buffer,
        &fallback_output,
        &params,
        DenseMmBackend::FallbackRequired,
    )
    .expect("fallback dispatch");
    fallback_encoder
        .commit_and_wait()
        .expect("fallback completion");

    let mut tensor_registry = KernelRegistry::new();
    let mut tensor_encoder = device.command_encoder().expect("tensor encoder");
    if let Err(error) = dense_matmul_f16_f32_with_backend(
        &mut tensor_encoder,
        &mut tensor_registry,
        &device,
        &weight_buffer,
        &input_buffer,
        &tensor_output,
        &params,
        DenseMmBackend::TensorRequired,
    ) {
        assert!(is_unavailable_tensor_header(&error), "{error}");
        return;
    }
    tensor_encoder.commit_and_wait().expect("tensor completion");

    assert_close(
        "f16",
        tensor_output.as_slice::<f32>().expect("tensor slice"),
        fallback_output.as_slice::<f32>().expect("fallback slice"),
        1e-4,
    );
}

#[test]
fn f32_tensor_and_tiled_fallback_preserve_numerical_contract() {
    let device = MlxDevice::new().expect("device");
    let weights = values(31, (SRC0_BATCH * N * K) as usize);
    let input = values(32, (SRC1_BATCH * M * K) as usize);
    let mut weight_buffer = device
        .alloc_buffer(weights.len() * 4, DType::F32, vec![weights.len()])
        .expect("weight buffer");
    weight_buffer
        .as_mut_slice::<f32>()
        .expect("weight slice")
        .copy_from_slice(&weights);
    let mut input_buffer = device
        .alloc_buffer(input.len() * 4, DType::F32, vec![input.len()])
        .expect("input buffer");
    input_buffer
        .as_mut_slice::<f32>()
        .expect("input slice")
        .copy_from_slice(&input);
    let output_len = (SRC1_BATCH * M * N) as usize;
    let tensor_output = device
        .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
        .expect("tensor output");
    let fallback_output = device
        .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
        .expect("fallback output");
    let params = DenseMmF32F32Params {
        m: M,
        n: N,
        k: K,
        src0_batch: SRC0_BATCH,
        src1_batch: SRC1_BATCH,
    };

    let mut fallback_registry = KernelRegistry::new();
    let mut fallback_encoder = device.command_encoder().expect("fallback encoder");
    dense_matmul_f32_f32_with_backend(
        &mut fallback_encoder,
        &mut fallback_registry,
        &device,
        &weight_buffer,
        &input_buffer,
        &fallback_output,
        &params,
        DenseMmBackend::FallbackRequired,
    )
    .expect("fallback dispatch");
    fallback_encoder
        .commit_and_wait()
        .expect("fallback completion");

    let mut tensor_registry = KernelRegistry::new();
    let mut tensor_encoder = device.command_encoder().expect("tensor encoder");
    if let Err(error) = dense_matmul_f32_f32_with_backend(
        &mut tensor_encoder,
        &mut tensor_registry,
        &device,
        &weight_buffer,
        &input_buffer,
        &tensor_output,
        &params,
        DenseMmBackend::TensorRequired,
    ) {
        assert!(is_unavailable_tensor_header(&error), "{error}");
        return;
    }
    tensor_encoder.commit_and_wait().expect("tensor completion");

    assert_close(
        "f32",
        tensor_output.as_slice::<f32>().expect("tensor slice"),
        fallback_output.as_slice::<f32>().expect("fallback slice"),
        1e-4,
    );
}

#[test]
#[ignore = "performance diagnostic; run on an idle Apple GPU"]
fn bf16_production_shape_tensor_vs_tiled_fallback_benchmark() {
    use std::time::Instant;

    const BM: u32 = 128;
    const BN: u32 = 128;
    const BK: u32 = 256;
    const B0: u32 = 4;
    const B1: u32 = 16;

    let device = MlxDevice::new().expect("device");
    let weight_len = (B0 * BN * BK) as usize;
    let input_len = (B1 * BM * BK) as usize;
    let output_len = (B1 * BM * BN) as usize;
    let weight = device
        .alloc_buffer(weight_len * 2, DType::BF16, vec![weight_len])
        .expect("weight");
    let input = device
        .alloc_buffer(input_len * 4, DType::F32, vec![input_len])
        .expect("input");
    let output = device
        .alloc_buffer(output_len * 4, DType::F32, vec![output_len])
        .expect("output");
    let params = DenseMmBf16F32Params {
        m: BM,
        n: BN,
        k: BK,
        src0_batch: B0,
        src1_batch: B1,
    };
    let mut tensor_registry = KernelRegistry::new();
    let mut fallback_registry = KernelRegistry::new();

    let dispatch = |registry: &mut KernelRegistry, backend| {
        let mut encoder = device.command_encoder().expect("encoder");
        let started = Instant::now();
        dense_matmul_bf16_f32_with_backend(
            &mut encoder,
            registry,
            &device,
            &weight,
            &input,
            &output,
            &params,
            backend,
        )?;
        encoder.commit_and_wait()?;
        Ok::<_, crate::MlxError>(started.elapsed())
    };

    for _ in 0..5 {
        if let Err(error) = dispatch(&mut tensor_registry, DenseMmBackend::TensorRequired) {
            assert!(is_unavailable_tensor_header(&error), "{error}");
            eprintln!("tensor API unavailable; M1 fallback benchmark requires hosted evidence");
            return;
        }
        dispatch(&mut fallback_registry, DenseMmBackend::FallbackRequired)
            .expect("fallback warmup");
    }

    let mut tensor_us = Vec::with_capacity(21);
    let mut fallback_us = Vec::with_capacity(21);
    for _ in 0..21 {
        tensor_us.push(
            dispatch(&mut tensor_registry, DenseMmBackend::TensorRequired)
                .expect("tensor sample")
                .as_micros(),
        );
        fallback_us.push(
            dispatch(&mut fallback_registry, DenseMmBackend::FallbackRequired)
                .expect("fallback sample")
                .as_micros(),
        );
    }
    tensor_us.sort_unstable();
    fallback_us.sort_unstable();
    let tensor_median = tensor_us[tensor_us.len() / 2];
    let fallback_median = fallback_us[fallback_us.len() / 2];
    eprintln!(
        "dense BF16 M={BM} N={BN} K={BK} b={B1}: tensor={tensor_median}us \
         fallback={fallback_median}us ratio={:.3}",
        fallback_median as f64 / tensor_median as f64
    );
}
