//! Correctness tests for the gather_bench kernels.
//!
//! Uses a small, fully-deterministic layout so the expected output can be
//! computed on the CPU and compared element-by-element.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::gather_bench;
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    gather_bench::register(&mut registry);
    (device, registry)
}

// ---------------------------------------------------------------------------
// Nibble-gather correctness
// ---------------------------------------------------------------------------

/// Small case: capacity=4, head_dim=8
///
/// packed layout: each byte holds two 4-bit indices.
///   byte = (hi_nibble << 4) | lo_nibble
///   lo_nibble → even coordinate
///   hi_nibble → odd  coordinate
///
/// centroids layout: [16, head_dim] f32; centroid[k][c] = (k + 1) * (c + 1) as f32
///   — simple product so we can verify the gather without ambiguity.
#[test]
fn test_gather_nibble_correctness() {
    let (device, mut registry) = setup();

    let capacity: u32 = 4;
    let head_dim: u32 = 8;
    let n_centroids: usize = 16;

    // Build centroid table on CPU: centroid[k][c] = (k+1) * (c+1) as f32
    let mut centroids_cpu = vec![0.0f32; n_centroids * head_dim as usize];
    for k in 0..n_centroids {
        for c in 0..head_dim as usize {
            centroids_cpu[k * head_dim as usize + c] = ((k + 1) * (c + 1)) as f32;
        }
    }

    // Build packed nibble buffer.
    // For each position p and each pair of coordinates (2c, 2c+1):
    //   index_even = p % 16          (index for coord 2c)
    //   index_odd  = (p + 1) % 16   (index for coord 2c+1)
    //   packed_byte = (index_odd << 4) | index_even
    let packed_len = capacity as usize * (head_dim as usize / 2);
    let mut packed_cpu = vec![0u8; packed_len];
    for p in 0..capacity as usize {
        for pair in 0..(head_dim as usize / 2) {
            let idx_even = (p % 16) as u8;
            let idx_odd = ((p + 1) % 16) as u8;
            packed_cpu[p * (head_dim as usize / 2) + pair] = (idx_odd << 4) | idx_even;
        }
    }

    // Compute expected output on CPU.
    let mut expected = vec![0.0f32; capacity as usize * head_dim as usize];
    for p in 0..capacity as usize {
        for c in 0..head_dim as usize {
            // Extract nibble the same way the Metal kernel does.
            let byte_idx = p * (head_dim as usize / 2) + c / 2;
            let byte = packed_cpu[byte_idx];
            let idx = if c % 2 == 0 { byte & 0xF } else { (byte >> 4) & 0xF } as usize;
            expected[p * head_dim as usize + c] = centroids_cpu[idx * head_dim as usize + c];
        }
    }

    // Allocate GPU buffers.
    let mut packed_buf = device
        .alloc_buffer(packed_len, DType::U8, vec![packed_len])
        .expect("alloc packed");
    packed_buf
        .as_mut_slice::<u8>()
        .expect("write packed")
        .copy_from_slice(&packed_cpu);

    let centroid_byte_len = centroids_cpu.len() * 4;
    let mut centroid_buf = device
        .alloc_buffer(centroid_byte_len, DType::F32, vec![centroids_cpu.len()])
        .expect("alloc centroids");
    {
        let slice: &mut [f32] = centroid_buf.as_mut_slice().expect("write centroids");
        slice.copy_from_slice(&centroids_cpu);
    }

    let out_len = capacity as usize * head_dim as usize;
    let out_buf = device
        .alloc_buffer(out_len * 4, DType::F32, vec![out_len])
        .expect("alloc out");

    // Dispatch.
    let mut encoder = device.command_encoder().expect("encoder");
    gather_bench::dispatch_gather_nibble(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &packed_buf,
        &centroid_buf,
        &out_buf,
        capacity,
        head_dim,
    )
    .expect("dispatch_gather_nibble");
    encoder.commit_and_wait().expect("commit_and_wait");

    // Verify.
    let result: Vec<f32> = out_buf.as_slice::<f32>().expect("read out").to_vec();
    assert_eq!(result.len(), expected.len());
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at element {i}: got {got}, expected {exp}"
        );
    }

    println!("test_gather_nibble_correctness: PASSED (capacity={capacity}, head_dim={head_dim})");
    println!("  First 8 output values: {:?}", &result[..8]);
}

// ---------------------------------------------------------------------------
// F16-seq correctness
// ---------------------------------------------------------------------------

/// Verify that gather_bench_f16_seq correctly widens F16 → F32.
///
/// Uses a small 4×8 cache filled with the F16 encoding of 1.0 (0x3C00).
#[test]
fn test_gather_f16_seq_correctness() {
    let (device, mut registry) = setup();

    let capacity: u32 = 4;
    let head_dim: u32 = 8;
    let n = capacity as usize * head_dim as usize;

    // Build an F16 cache buffer: all elements = 1.0 (0x3C00 in little-endian)
    let cache_byte_len = n * 2;
    let mut cache_buf = device
        .alloc_buffer(cache_byte_len, DType::F16, vec![n])
        .expect("alloc cache");
    {
        let raw: &mut [u8] = cache_buf.as_mut_slice().expect("write cache");
        for chunk in raw.chunks_exact_mut(2) {
            // F16 1.0 = 0x3C00, stored little-endian
            chunk[0] = 0x00;
            chunk[1] = 0x3C;
        }
    }

    let out_buf = device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc out");

    let mut encoder = device.command_encoder().expect("encoder");
    gather_bench::dispatch_gather_f16_seq(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &cache_buf,
        &out_buf,
        capacity,
        head_dim,
    )
    .expect("dispatch_gather_f16_seq");
    encoder.commit_and_wait().expect("commit_and_wait");

    let result: Vec<f32> = out_buf.as_slice::<f32>().expect("read out").to_vec();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 1.0f32).abs() < 1e-4,
            "element {i}: expected 1.0, got {v}"
        );
    }

    println!("test_gather_f16_seq_correctness: PASSED (capacity={capacity}, head_dim={head_dim})");
    println!("  All {n} elements = {:.4}", result[0]);
}
