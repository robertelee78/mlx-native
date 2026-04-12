//! Tests for the argsort (descending) GPU kernel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::argsort;
use mlx_native::{DType, KernelRegistry, MlxDevice};

fn pseudo_random_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    argsort::register(&mut registry);
    (device, registry)
}

/// CPU reference: argsort descending.  Returns indices such that
/// values[indices[0]] >= values[indices[1]] >= ...
fn cpu_argsort_desc(data: &[f32]) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..data.len() as u32).collect();
    indices.sort_unstable_by(|&a, &b| {
        data[b as usize]
            .partial_cmp(&data[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

#[test]
fn test_argsort_random_128() {
    let (device, mut registry) = setup();
    let row_len: u32 = 128;
    let batch_size: u32 = 1;

    let data = pseudo_random_f32(42, row_len as usize);
    let expected = cpu_argsort_desc(&data);

    let byte_len = row_len as usize * 4;
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![row_len as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&data);

    let output_buf = device
        .alloc_buffer(byte_len, DType::U32, vec![row_len as usize])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    argsort::dispatch_argsort_desc_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        batch_size,
        row_len,
    )
    .expect("dispatch_argsort_desc_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual = output_buf.as_slice::<u32>().expect("read output");

    // Verify that the GPU output sorts values in descending order.
    for i in 0..row_len as usize - 1 {
        assert!(
            data[actual[i] as usize] >= data[actual[i + 1] as usize],
            "argsort descending violated at position {}: val[{}]={} < val[{}]={}",
            i,
            actual[i],
            data[actual[i] as usize],
            actual[i + 1],
            data[actual[i + 1] as usize]
        );
    }

    // Verify indices match CPU reference exactly.
    for i in 0..row_len as usize {
        assert_eq!(
            actual[i], expected[i],
            "argsort index mismatch at position {}: GPU={}, CPU={}",
            i, actual[i], expected[i]
        );
    }
}

#[test]
fn test_argsort_batch_4x8() {
    let (device, mut registry) = setup();
    let row_len: u32 = 8;
    let batch_size: u32 = 4;
    let total = batch_size as usize * row_len as usize;

    let data = pseudo_random_f32(123, total);

    let byte_len = total * 4;
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![batch_size as usize, row_len as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&data);

    let output_buf = device
        .alloc_buffer(byte_len, DType::U32, vec![batch_size as usize, row_len as usize])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    argsort::dispatch_argsort_desc_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        batch_size,
        row_len,
    )
    .expect("dispatch_argsort_desc_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual = output_buf.as_slice::<u32>().expect("read output");

    // Verify each row is sorted independently in descending order.
    for row in 0..batch_size as usize {
        let row_data = &data[row * row_len as usize..(row + 1) * row_len as usize];
        let row_indices = &actual[row * row_len as usize..(row + 1) * row_len as usize];

        let _expected = cpu_argsort_desc(row_data);

        for i in 0..row_len as usize - 1 {
            assert!(
                row_data[row_indices[i] as usize] >= row_data[row_indices[i + 1] as usize],
                "batch row {}: descending order violated at {}: val[{}]={} < val[{}]={}",
                row,
                i,
                row_indices[i],
                row_data[row_indices[i] as usize],
                row_indices[i + 1],
                row_data[row_indices[i + 1] as usize]
            );
        }

        // Verify indices form a valid permutation of [0, row_len).
        let mut seen = vec![false; row_len as usize];
        for &idx in row_indices {
            assert!(
                (idx as usize) < row_len as usize,
                "batch row {}: index {} out of range",
                row, idx
            );
            seen[idx as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "batch row {}: indices are not a valid permutation",
            row
        );
    }
}

#[test]
fn test_argsort_already_sorted() {
    let (device, mut registry) = setup();
    let row_len: u32 = 8;
    let batch_size: u32 = 1;

    // Already sorted in descending order: [7, 6, 5, 4, 3, 2, 1, 0]
    let data: Vec<f32> = (0..row_len).rev().map(|x| x as f32).collect();

    let byte_len = row_len as usize * 4;
    let mut input_buf = device
        .alloc_buffer(byte_len, DType::F32, vec![row_len as usize])
        .expect("alloc input");
    input_buf
        .as_mut_slice::<f32>()
        .expect("write input")
        .copy_from_slice(&data);

    let output_buf = device
        .alloc_buffer(byte_len, DType::U32, vec![row_len as usize])
        .expect("alloc output");

    let mut encoder = device.command_encoder().expect("encoder");
    argsort::dispatch_argsort_desc_f32(
        &mut encoder,
        &mut registry,
        device.metal_device(),
        &input_buf,
        &output_buf,
        batch_size,
        row_len,
    )
    .expect("dispatch_argsort_desc_f32");

    encoder.commit_and_wait().expect("commit_and_wait");

    let actual = output_buf.as_slice::<u32>().expect("read output");

    // Already sorted descending, so indices should be identity: [0, 1, 2, ...]
    for i in 0..row_len as usize {
        assert_eq!(
            actual[i], i as u32,
            "already-sorted: expected identity permutation at {}, got {}",
            i, actual[i]
        );
    }
}
