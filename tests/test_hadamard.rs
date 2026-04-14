//! Correctness tests for the Fast Walsh-Hadamard Transform (FWHT) GPU kernel.
//!
//! Tests:
//!   a) Known-values:   FWHT([1,1,1,1]) == [2,0,0,0] (normalized by 1/√4).
//!   b) Involution:     H(H(x)) == x  (H·H = I, within ε < 1e-4).
//!   c) Energy:         ‖H(x)‖₂ == ‖x‖₂ (orthogonal transform).
//!   d) Dimensions:     d=128, d=256, d=512 all pass (a)–(c).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::hadamard;
use mlx_native::{DType, KernelRegistry, MlxDevice};

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Sequential in-place unnormalized Walsh-Hadamard Transform.
fn cpu_fwht_unnormalized(x: &mut [f32]) {
    let n = x.len();
    assert!(n.is_power_of_two(), "FWHT length must be a power of two");
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Normalized FWHT reference: applies unnormalized FWHT then scales by 1/√n.
fn cpu_fwht(x: &mut [f32]) {
    let n = x.len();
    cpu_fwht_unnormalized(x);
    let scale = 1.0_f32 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    hadamard::register(&mut registry);
    (device, registry)
}

/// Run the GPU FWHT on `data` (length = num_heads × head_dim) in-place.
fn run_gpu_fwht(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    data: &mut Vec<f32>,
    head_dim: u32,
    num_heads: u32,
) {
    let n = data.len();
    let byte_len = n * 4;
    let mut buf = device
        .alloc_buffer(byte_len, DType::F32, vec![num_heads as usize, head_dim as usize])
        .expect("alloc_buffer");

    {
        let slice: &mut [f32] = buf.as_mut_slice().expect("as_mut_slice");
        slice.copy_from_slice(data);
    }

    {
        let mut enc = device.command_encoder().expect("command_encoder");
        hadamard::dispatch_hadamard_transform(
            &mut enc,
            registry,
            device.metal_device(),
            &buf,
            head_dim,
            num_heads,
        )
        .expect("dispatch_hadamard_transform");
        enc.commit_and_wait().expect("commit_and_wait");
    }

    {
        let slice: &[f32] = buf.as_slice().expect("as_slice");
        data.copy_from_slice(slice);
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// a) Known-values test
// ---------------------------------------------------------------------------
//
// H_4 · [1,1,1,1]^T = [4,0,0,0]^T  (unnormalized)
// Normalized by 1/√4 = 0.5 → [2,0,0,0].

#[test]
fn test_hadamard_known_values_d4() {
    let (device, mut registry) = setup();
    let mut data = vec![1.0f32, 1.0, 1.0, 1.0];
    run_gpu_fwht(&device, &mut registry, &mut data, 4, 1);

    assert!(
        (data[0] - 2.0).abs() < 1e-4,
        "data[0] expected 2.0, got {}",
        data[0]
    );
    for i in 1..4 {
        assert!(
            data[i].abs() < 1e-4,
            "data[{}] expected 0.0, got {}",
            i,
            data[i]
        );
    }
}

// ---------------------------------------------------------------------------
// b) Involution test: H(H(x)) == x  (across multiple dims and heads)
// ---------------------------------------------------------------------------

fn involution_test(head_dim: u32, num_heads: u32) {
    let (device, mut registry) = setup();
    let n = (num_heads as usize) * (head_dim as usize);

    // Deterministic pseudo-random input.
    let original: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 1.618_034 + 0.5).sin()) * 2.0)
        .collect();

    // First transform.
    let mut data = original.clone();
    run_gpu_fwht(&device, &mut registry, &mut data, head_dim, num_heads);

    // Second transform (should return original).
    run_gpu_fwht(&device, &mut registry, &mut data, head_dim, num_heads);

    for (i, (&got, &exp)) in data.iter().zip(original.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "Involution failed at dim={head_dim} heads={num_heads} idx={i}: \
             got {got:.6}, expected {exp:.6}"
        );
    }
}

#[test]
fn test_hadamard_involution_d128() {
    involution_test(128, 4);
}

#[test]
fn test_hadamard_involution_d256() {
    involution_test(256, 8);
}

#[test]
fn test_hadamard_involution_d512() {
    involution_test(512, 2);
}

// ---------------------------------------------------------------------------
// c) Energy preservation: ‖H(x)‖₂ == ‖x‖₂
// ---------------------------------------------------------------------------

fn energy_test(head_dim: u32, num_heads: u32) {
    let (device, mut registry) = setup();
    let n = (num_heads as usize) * (head_dim as usize);

    let original: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 2.718_28 + 1.0).cos()) * 3.0)
        .collect();

    let original_norm = l2_norm(&original);

    let mut data = original.clone();
    run_gpu_fwht(&device, &mut registry, &mut data, head_dim, num_heads);

    let transformed_norm = l2_norm(&data);

    let rel_err = (transformed_norm - original_norm).abs() / original_norm.max(1e-8);
    assert!(
        rel_err < 1e-4,
        "Energy not preserved at dim={head_dim} heads={num_heads}: \
         original_norm={original_norm:.6} transformed_norm={transformed_norm:.6} \
         rel_err={rel_err:.6}"
    );
}

#[test]
fn test_hadamard_energy_d128() {
    energy_test(128, 4);
}

#[test]
fn test_hadamard_energy_d256() {
    energy_test(256, 8);
}

#[test]
fn test_hadamard_energy_d512() {
    energy_test(512, 2);
}

// ---------------------------------------------------------------------------
// d) GPU vs CPU reference comparison
// ---------------------------------------------------------------------------

fn gpu_vs_cpu_test(head_dim: u32, num_heads: u32) {
    let (device, mut registry) = setup();
    let n = (num_heads as usize) * (head_dim as usize);

    let input: Vec<f32> = (0..n)
        .map(|i| i as f32 * 0.123 - (n as f32) * 0.0615)
        .collect();

    // CPU reference: apply per-head.
    let mut cpu_out = input.clone();
    for h in 0..num_heads as usize {
        let base = h * head_dim as usize;
        cpu_fwht(&mut cpu_out[base..base + head_dim as usize]);
    }

    // GPU.
    let mut gpu_out = input.clone();
    run_gpu_fwht(&device, &mut registry, &mut gpu_out, head_dim, num_heads);

    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-4,
            "GPU vs CPU mismatch at dim={head_dim} heads={num_heads} idx={i}: \
             gpu={g:.6} cpu={c:.6}"
        );
    }
}

#[test]
fn test_hadamard_gpu_vs_cpu_d128() {
    gpu_vs_cpu_test(128, 4);
}

#[test]
fn test_hadamard_gpu_vs_cpu_d256() {
    gpu_vs_cpu_test(256, 8);
}

#[test]
fn test_hadamard_gpu_vs_cpu_d512() {
    gpu_vs_cpu_test(512, 2);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn test_hadamard_non_power_of_two_fails() {
    let (device, mut registry) = setup();
    let byte_len = 6 * 4;
    let buf = device
        .alloc_buffer(byte_len, DType::F32, vec![1, 6])
        .expect("alloc_buffer");
    let mut enc = device.command_encoder().expect("command_encoder");
    let result = hadamard::dispatch_hadamard_transform(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &buf,
        6,  // not a power of two
        1,
    );
    assert!(result.is_err(), "Non-power-of-two head_dim should fail");
}

#[test]
fn test_hadamard_zero_heads_is_noop() {
    let (device, mut registry) = setup();
    let byte_len = 256 * 4;
    let buf = device
        .alloc_buffer(byte_len, DType::F32, vec![256])
        .expect("alloc_buffer");
    let mut enc = device.command_encoder().expect("command_encoder");
    let result = hadamard::dispatch_hadamard_transform(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &buf,
        256,
        0,  // zero heads — should be a no-op, not an error
    );
    assert!(result.is_ok(), "Zero num_heads should be a no-op, not an error");
    enc.commit_and_wait().expect("commit_and_wait");
}
