//! Tests for sigmoid-gated elementwise multiply (ADR-013 Decision 9).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::ops::sigmoid_mul::dispatch_sigmoid_mul;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

fn setup() -> (MlxDevice, KernelRegistry) {
    (
        MlxDevice::new().expect("device"),
        KernelRegistry::new(),
    )
}

fn upload_f32(device: &MlxDevice, data: &[f32]) -> MlxBuffer {
    let mut b = device
        .alloc_buffer(data.len() * 4, DType::F32, vec![data.len()])
        .expect("alloc");
    b.as_mut_slice::<f32>().expect("mut").copy_from_slice(data);
    b
}

fn alloc_params(device: &MlxDevice, n: u32) -> MlxBuffer {
    let mut p = device.alloc_buffer(4, DType::U32, vec![1]).expect("p");
    p.as_mut_slice::<u32>().expect("mut")[0] = n;
    p
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn sigmoid_mul_f32_matches_cpu() {
    let (device, mut registry) = setup();
    let x: Vec<f32> = (0..100).map(|i| (i as f32) * 0.05 - 2.5).collect();
    let gate: Vec<f32> = (0..100).map(|i| (i as f32) * 0.03 - 1.5).collect();
    let n = x.len() as u32;

    let x_buf = upload_f32(&device, &x);
    let gate_buf = upload_f32(&device, &gate);
    let out_buf = device
        .alloc_buffer(x.len() * 4, DType::F32, vec![x.len()])
        .expect("out");
    let p = alloc_params(&device, n);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_sigmoid_mul(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &gate_buf,
        &out_buf,
        &p,
        n,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let got: &[f32] = out_buf.as_slice().expect("read");
    for i in 0..x.len() {
        let expected = x[i] * sigmoid(gate[i]);
        let d = (got[i] - expected).abs();
        assert!(d < 1e-6, "mismatch at {}: got {}, expected {}", i, got[i], expected);
    }
}

#[test]
fn sigmoid_mul_saturation_bounds() {
    // Gate = large positive → sigmoid ≈ 1 → out ≈ x.
    // Gate = large negative → sigmoid ≈ 0 → out ≈ 0.
    let (device, mut registry) = setup();
    let x = vec![5.0f32, 5.0, 5.0, 5.0];
    let gate = vec![100.0f32, -100.0, 0.0, 0.0];
    let x_buf = upload_f32(&device, &x);
    let gate_buf = upload_f32(&device, &gate);
    let out_buf = device.alloc_buffer(16, DType::F32, vec![4]).expect("out");
    let p = alloc_params(&device, 4);

    let mut enc = device.command_encoder().expect("enc");
    dispatch_sigmoid_mul(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &x_buf,
        &gate_buf,
        &out_buf,
        &p,
        4,
    )
    .expect("dispatch");
    enc.commit_and_wait().expect("commit");

    let got: &[f32] = out_buf.as_slice().expect("read");
    assert!((got[0] - 5.0).abs() < 1e-3, "gate=+inf should give x: got {}", got[0]);
    assert!(got[1].abs() < 1e-3, "gate=-inf should give 0: got {}", got[1]);
    assert!((got[2] - 2.5).abs() < 1e-5, "gate=0 should give x*0.5 = 2.5: got {}", got[2]);
    assert!((got[3] - 2.5).abs() < 1e-5);
}

#[test]
fn sigmoid_mul_rejects_zero_n() {
    let (device, mut registry) = setup();
    let dummy = device.alloc_buffer(4, DType::F32, vec![1]).expect("d");
    let p = alloc_params(&device, 0);
    let mut enc = device.command_encoder().expect("enc");
    let res = dispatch_sigmoid_mul(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &dummy, &dummy, &dummy, &p, 0,
    );
    assert!(res.is_err());
}

#[test]
fn sigmoid_mul_rejects_dtype_mismatch() {
    let (device, mut registry) = setup();
    let x = device.alloc_buffer(4, DType::F32, vec![1]).expect("x");
    let gate = device.alloc_buffer(2, DType::BF16, vec![1]).expect("g");
    let out = device.alloc_buffer(4, DType::F32, vec![1]).expect("o");
    let p = alloc_params(&device, 1);
    let mut enc = device.command_encoder().expect("enc");
    let res = dispatch_sigmoid_mul(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &x, &gate, &out, &p, 1,
    );
    assert!(res.is_err(), "dtype mismatch should error");
}
