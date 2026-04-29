//! Smoke test for ADR-015 iter16 — CB / encoder label propagation.
//!
//! `mlx_native::CommandEncoder::commit_labeled(label)` and
//! `commit_and_wait_labeled(label)` MUST set both `MTLCommandBuffer.label` and
//! the active `MTLComputeCommandEncoder.label` so xctrace's
//! `metal-application-encoders-list` table populates `cmdbuffer-label` and
//! `encoder-label` columns with the semantic phase name.  This unblocks the
//! per-phase µs/token attribution path described in iter15 §E ("iter16
//! ATTRIBUTION PATH").
//!
//! What this test verifies:
//! * After `commit_and_wait_labeled("phase.unit_test_token")`, the underlying
//!   metal-rs `MTLCommandBuffer::label()` getter returns the same string.
//!   This is the simplest invariant of the propagation path — if it passes,
//!   the ObjC `setLabel:` call is wired through to the live MTL object that
//!   xctrace will see when MST is recording.
//!
//! What this test does NOT cover:
//! * Per-encoder `MTLComputeCommandEncoder::label()` after `endEncoding` is
//!   not directly readable — Metal does not expose a getter on already-ended
//!   encoders. iter16 §B verifies this leg via xctrace MST capture instead.
//! * The xctrace MST capture itself (that's an integration concern; verified
//!   under `/tmp/adr015-iter16/` by the iter16 implementer's profile run).
//!
//! ADR-015 iter16 — claude implementer worktree
//! `cfa-20260428-adr015-iter16-claude`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

#[test]
fn commit_and_wait_labeled_sets_cmdbuffer_label() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n = 4usize;
    let byte_len = n * std::mem::size_of::<f32>();
    let mut a = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");
    a.as_mut_slice::<f32>().unwrap().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    b.as_mut_slice::<f32>().unwrap().copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::elementwise::elementwise_add(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &a,
        &b,
        &out,
        n,
        DType::F32,
    )
    .expect("dispatch elementwise_add");

    // The label this test pins on the CB.  Use a phase-style name that
    // mirrors what hf2q's qwen35 forward path passes to commit_*_labeled
    // so a future grep for the test fixture ↔ production parity is obvious.
    let phase_label = "phase.iter16_smoke_token";

    enc.commit_and_wait_labeled(phase_label).expect("commit_and_wait_labeled");

    // After commit_and_wait_labeled, the underlying MTLCommandBuffer's label
    // getter should round-trip the value.  metal-rs 0.33's CommandBuffer::label
    // returns &str backed by the ObjC string property.
    let cb_label = enc.metal_command_buffer().label();
    assert_eq!(
        cb_label, phase_label,
        "MTLCommandBuffer.label must round-trip the value passed to commit_and_wait_labeled"
    );

    // Also verify the dispatch produced correct results — propagation must not
    // perturb the compute path.
    let result = out.as_slice::<f32>().expect("read");
    assert_eq!(
        result,
        &[6.0, 8.0, 10.0, 12.0],
        "commit_and_wait_labeled must produce correct elementwise_add results"
    );
}

#[test]
fn commit_labeled_sets_cmdbuffer_label() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    let n = 4usize;
    let byte_len = n * std::mem::size_of::<f32>();
    let mut a = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("a");
    let mut b = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("b");
    let out = device.alloc_buffer(byte_len, DType::F32, vec![n]).expect("out");
    a.as_mut_slice::<f32>().unwrap().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    b.as_mut_slice::<f32>().unwrap().copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

    let mut enc = device.command_encoder().expect("enc");
    mlx_native::ops::elementwise::elementwise_add(
        &mut enc,
        &mut registry,
        device.metal_device(),
        &a,
        &b,
        &out,
        n,
        DType::F32,
    )
    .expect("dispatch elementwise_add");

    let phase_label = "phase.iter16_smoke_async";

    enc.commit_labeled(phase_label);

    // Async commit path also sets the label — verify before wait.
    let cb_label = enc.metal_command_buffer().label();
    assert_eq!(
        cb_label, phase_label,
        "MTLCommandBuffer.label must round-trip the value passed to commit_labeled"
    );

    enc.wait_until_completed().expect("wait_until_completed");

    let result = out.as_slice::<f32>().expect("read");
    assert_eq!(
        result,
        &[6.0, 8.0, 10.0, 12.0],
        "commit_labeled must produce correct elementwise_add results"
    );
}
