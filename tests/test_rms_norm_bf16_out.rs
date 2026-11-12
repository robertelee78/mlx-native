//! `dispatch_rms_norm_f32_to_bf16` is bitwise `dispatch_rms_norm` on F32
//! buffers followed by `cast_f32_to_bf16` (Class A), at the Qwen hidden sizes
//! and at dimensions around the kernel's threadgroup geometry, for every row
//! count of a joint step. A row's bits do not depend on the row count or on
//! the row's position, and rows beyond the dispatch are not written.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::rms_norm::{
    dispatch_rms_norm, dispatch_rms_norm_f32_to_bf16, rms_norm_f32_to_bf16_available,
    rms_norm_f32_to_bf16_supported,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const EPS: f32 = 1e-6;
const SENTINEL: u16 = 0x7777;

fn value(row: usize, col: usize, salt: u32) -> f32 {
    let mut h = (row as u32).wrapping_mul(0x9E37_79B1)
        ^ (col as u32).wrapping_mul(0x85EB_CA77)
        ^ salt.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    // Mixed magnitudes, so the BF16 rounding is exercised over many exponents.
    let unit = (h & 0xFFFF) as f32 / 32768.0 - 1.0;
    unit * [0.01, 1.0, 37.0, 900.0][(h >> 16) as usize % 4]
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .unwrap();
    buffer
        .as_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(values);
    buffer
}

fn bf16_buffer(device: &MlxDevice, elements: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * 2, DType::BF16, vec![elements])
        .unwrap();
    buffer.as_mut_slice::<u16>().unwrap().fill(SENTINEL);
    buffer
}

/// Rows `rows` of the synthetic activation, normalized by the stock kernel
/// and cast.
fn stock(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    rows: &[usize],
    dim: usize,
) -> Vec<u16> {
    let input: Vec<f32> = rows
        .iter()
        .flat_map(|&row| (0..dim).map(move |col| value(row, col, 1)))
        .collect();
    let weight: Vec<f32> = (0..dim)
        .map(|col| 1.0 + 0.5 * value(0, col, 2).signum())
        .collect();
    let input = f32_buffer(device, &input);
    let weight = f32_buffer(device, &weight);
    let normed = f32_buffer(device, &vec![0.0; rows.len() * dim]);
    let output = bf16_buffer(device, rows.len() * dim);
    let params = f32_buffer(device, &[EPS, dim as f32]);
    let mut encoder = device.command_encoder().unwrap();
    dispatch_rms_norm(
        &mut encoder,
        registry,
        device.metal_device(),
        &input,
        &weight,
        &normed,
        &params,
        rows.len() as u32,
        dim as u32,
    )
    .unwrap();
    encoder.memory_barrier();
    cast(
        &mut encoder,
        registry,
        device.metal_device(),
        &normed,
        &output,
        rows.len() * dim,
        CastDirection::F32ToBF16,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    output.as_slice::<u16>().unwrap().to_vec()
}

/// The same rows through the BF16-output kernel, into a buffer with one
/// extra row that must stay untouched.
fn fused(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    rows: &[usize],
    dim: usize,
) -> Vec<u16> {
    let input: Vec<f32> = rows
        .iter()
        .flat_map(|&row| (0..dim).map(move |col| value(row, col, 1)))
        .collect();
    let weight: Vec<f32> = (0..dim)
        .map(|col| 1.0 + 0.5 * value(0, col, 2).signum())
        .collect();
    let input = f32_buffer(device, &input);
    let weight = f32_buffer(device, &weight);
    let storage = bf16_buffer(device, (rows.len() + 1) * dim);
    let output = storage.slice_view(0, rows.len() * dim);
    let params = f32_buffer(device, &[EPS, dim as f32]);
    let mut encoder = device.command_encoder().unwrap();
    dispatch_rms_norm_f32_to_bf16(
        &mut encoder,
        registry,
        device.metal_device(),
        &input,
        &weight,
        &output,
        &params,
        rows.len() as u32,
        dim as u32,
    )
    .unwrap();
    encoder.commit_and_wait().unwrap();
    let all = storage.as_slice::<u16>().unwrap();
    assert!(
        all[rows.len() * dim..].iter().all(|&bits| bits == SENTINEL),
        "dim {dim}, {} rows: a row beyond the dispatch was written",
        rows.len()
    );
    all[..rows.len() * dim].to_vec()
}

#[test]
fn bf16_output_norm_is_bitwise_the_norm_then_the_cast() {
    let Ok(device) = MlxDevice::new() else { return };
    let mut registry = KernelRegistry::new();
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    // The Qwen hidden sizes, then every threadgroup size of the kernel (32,
    // 64, 128 and 256 threads) on both sides of its boundaries.
    for dim in [
        5_120, 2_048, 1_024, 256, 132, 128, 124, 100, 68, 64, 36, 32, 28, 4,
    ] {
        assert!(rms_norm_f32_to_bf16_available(dim as u32));
        assert!(
            rms_norm_f32_to_bf16_supported(&mut registry, device.metal_device(), dim as u32)
                .unwrap(),
            "dim {dim}: the device limits refuse the BF16-output norm"
        );
        const POOL: usize = 40;
        let pool: Vec<usize> = (0..POOL).collect();
        let reference = stock(&device, &mut registry, &pool, dim);
        assert!(reference.iter().any(|&bits| bits != reference[0]));
        for m in [1, 4, 8, 16, 24, 32, POOL] {
            // Rows from the middle of the pool: a row's bits follow the row,
            // not its position or the row count.
            let first = (POOL - m) / 2;
            let rows: Vec<usize> = (first..first + m).collect();
            let got = fused(&device, &mut registry, &rows, dim);
            let want = &reference[first * dim..(first + m) * dim];
            let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
            assert!(
                differing == 0,
                "dim {dim}, {m} rows: {differing} of {} elements differ from norm + cast",
                got.len()
            );
            assert!(
                fused(&device, &mut registry, &rows, dim) == got,
                "dim {dim}, {m} rows: a repeated run differs"
            );
            for &bits in &got {
                digest = (digest ^ u64::from(bits)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    println!("rms_norm_f32_to_bf16 digest: {digest:016x}");
}

#[test]
fn bf16_output_norm_rejects_other_geometries() {
    let Ok(device) = MlxDevice::new() else { return };
    let mut registry = KernelRegistry::new();
    assert!(!rms_norm_f32_to_bf16_available(0));
    assert!(!rms_norm_f32_to_bf16_available(130));
    assert!(!rms_norm_f32_to_bf16_supported(&mut registry, device.metal_device(), 130).unwrap());
    let input = f32_buffer(&device, &[1.0; 16]);
    let weight = f32_buffer(&device, &[1.0; 8]);
    let params = f32_buffer(&device, &[EPS, 8.0]);
    let output = bf16_buffer(&device, 16);
    let f32_output = f32_buffer(&device, &[0.0; 16]);
    let short = bf16_buffer(&device, 8);
    let mut attempt = |input: &MlxBuffer, weight: &MlxBuffer, output: &MlxBuffer, rows, dim| {
        let mut encoder = device.command_encoder().unwrap();
        dispatch_rms_norm_f32_to_bf16(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            input,
            weight,
            output,
            &params,
            rows,
            dim,
        )
        .is_err()
    };
    assert!(
        !attempt(&input, &weight, &output, 2, 8),
        "a valid norm was rejected"
    );
    assert!(attempt(&input, &weight, &output, 0, 8), "zero rows");
    assert!(attempt(&input, &weight, &f32_output, 2, 8), "F32 output");
    assert!(attempt(&input, &weight, &short, 2, 8), "short output");
    assert!(attempt(&output, &weight, &output, 2, 8), "BF16 input");
    assert!(attempt(&input, &weight, &output, 1, 16), "short weight");
    assert!(
        attempt(&input, &weight, &output, 8, 2),
        "dim not a multiple of four"
    );
}
