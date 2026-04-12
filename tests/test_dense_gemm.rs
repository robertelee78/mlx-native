//! Tests for the dense F16 GEMM GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::dense_gemm::{self, DenseGemmF16Params};
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    dense_gemm::register(&mut registry);
    (device, registry)
}

/// Convert f32 to f16 bytes (IEEE 754 half-precision).
fn f32_to_f16(val: f32) -> u16 {
    half::f16::from_f32(val).to_bits()
}

/// Convert f16 bytes back to f32.
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// CPU reference: C = A * B^T, all in f32 for precision.
fn cpu_gemm_abt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk]; // B^T: B[j, kk]
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[test]
fn test_dense_gemm_small_known() {
    let (device, mut registry) = setup();
    let m: u32 = 2;
    let n: u32 = 3;
    let k: u32 = 4;

    // A = [[1, 2, 3, 4],
    //      [5, 6, 7, 8]]
    let a_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // B = [[1, 0, 1, 0],
    //      [0, 1, 0, 1],
    //      [1, 1, 0, 0]]
    let b_f32: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];

    let expected = cpu_gemm_abt(&a_f32, &b_f32, m as usize, n as usize, k as usize);

    // Convert to f16 byte arrays.
    let a_f16: Vec<u16> = a_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let b_f16: Vec<u16> = b_f32.iter().map(|&v| f32_to_f16(v)).collect();

    let a_bytes = m as usize * k as usize * 2;
    let mut a_buf = device
        .alloc_buffer(a_bytes, DType::F16, vec![m as usize, k as usize])
        .expect("alloc A");
    a_buf
        .as_mut_slice::<u16>()
        .expect("write A")
        .copy_from_slice(&a_f16);

    let b_bytes = n as usize * k as usize * 2;
    let mut b_buf = device
        .alloc_buffer(b_bytes, DType::F16, vec![n as usize, k as usize])
        .expect("alloc B");
    b_buf
        .as_mut_slice::<u16>()
        .expect("write B")
        .copy_from_slice(&b_f16);

    let c_bytes = m as usize * n as usize * 2;
    let c_buf = device
        .alloc_buffer(c_bytes, DType::F16, vec![m as usize, n as usize])
        .expect("alloc C");

    let params = DenseGemmF16Params { m, n, k };

    let mut encoder = device.command_encoder().expect("encoder");
    dense_gemm::dispatch_dense_gemm_f16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &a_buf,
        &b_buf,
        &c_buf,
        &params,
    )
    .expect("dispatch_dense_gemm_f16");

    encoder.commit_and_wait().expect("commit_and_wait");

    let output_f16 = c_buf.as_slice::<u16>().expect("read C");

    for i in 0..m as usize {
        for j in 0..n as usize {
            let actual = f16_to_f32(output_f16[i * n as usize + j]);
            let exp = expected[i * n as usize + j];
            assert!(
                (actual - exp).abs() < 1e-2,
                "GEMM mismatch at [{}, {}]: GPU={}, CPU={}",
                i, j, actual, exp
            );
        }
    }
}

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn test_dense_gemm_decode_shape_argmax() {
    let (device, mut registry) = setup();
    // M=1 decode shape, small K and N to keep test fast.
    let m: u32 = 1;
    let k: u32 = 64;
    let n: u32 = 256;

    let a_f32 = pseudo_random_f32(42, m as usize * k as usize);
    let b_f32 = pseudo_random_f32(99, n as usize * k as usize);

    let expected = cpu_gemm_abt(&a_f32, &b_f32, m as usize, n as usize, k as usize);

    // CPU argmax of expected.
    let cpu_argmax = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let a_f16: Vec<u16> = a_f32.iter().map(|&v| f32_to_f16(v)).collect();
    let b_f16: Vec<u16> = b_f32.iter().map(|&v| f32_to_f16(v)).collect();

    let a_bytes = m as usize * k as usize * 2;
    let mut a_buf = device
        .alloc_buffer(a_bytes, DType::F16, vec![m as usize, k as usize])
        .expect("alloc A");
    a_buf
        .as_mut_slice::<u16>()
        .expect("write A")
        .copy_from_slice(&a_f16);

    let b_bytes = n as usize * k as usize * 2;
    let mut b_buf = device
        .alloc_buffer(b_bytes, DType::F16, vec![n as usize, k as usize])
        .expect("alloc B");
    b_buf
        .as_mut_slice::<u16>()
        .expect("write B")
        .copy_from_slice(&b_f16);

    let c_bytes = m as usize * n as usize * 2;
    let c_buf = device
        .alloc_buffer(c_bytes, DType::F16, vec![m as usize, n as usize])
        .expect("alloc C");

    let params = DenseGemmF16Params { m, n, k };

    let mut encoder = device.command_encoder().expect("encoder");
    dense_gemm::dispatch_dense_gemm_f16(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &a_buf,
        &b_buf,
        &c_buf,
        &params,
    )
    .expect("dispatch_dense_gemm_f16");

    encoder.commit_and_wait().expect("commit_and_wait");

    let output_f16 = c_buf.as_slice::<u16>().expect("read C");
    let output_f32: Vec<f32> = output_f16.iter().map(|&b| f16_to_f32(b)).collect();

    let gpu_argmax = output_f32
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(
        gpu_argmax, cpu_argmax,
        "Decode argmax mismatch: GPU={}, CPU={}",
        gpu_argmax, cpu_argmax
    );
}
