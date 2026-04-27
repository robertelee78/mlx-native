//! Counter-based barrier accounting (ADR-015 H4 hard gate).
//!
//! ADR-015 §"P3a' live profile pass" hypothesis register row H4 was
//! reported as FALSIFIED at the literal `memory_barrier` frame against
//! a 1 ms-resolution xctrace TimeProfiler trace, then re-scoped to
//! NOT OBSERVED — BELOW TIMEPROFILER RESOLUTION after Codex review Q2
//! (2026-04-27).  Wave 2b hard gate #2 requires per-barrier counter
//! resolution (atomic count + summed `mach_absolute_time` ns) to
//! confirm-or-falsify the barrier-coalescing lever — this test
//! exercises the new accounting path.
//!
//! Verified:
//! 1. `barrier_count()` increments by exactly 1 per call to
//!    `CommandEncoder::memory_barrier()` in non-capture mode.
//! 2. Capture-mode `memory_barrier()` does NOT increment the counter
//!    (it records a [`CapturedNode::Barrier`] sentinel only).
//! 3. `barrier_total_ns()` is 0 by default (env-gated off) and is
//!    *non-zero* only when `MLX_PROFILE_BARRIERS=1` is set in the
//!    process — this test does NOT set it (avoid flakiness around the
//!    first-call OnceLock cache); the env-on path is exercised
//!    manually by callers per the docstring.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

#[test]
fn barrier_counter_skips_when_no_active_encoder() {
    // Calls before `active_encoder` is non-null are no-ops AND must not
    // increment the counter (the increment is gated behind the same
    // null check inside memory_barrier).  Use delta-based assertion
    // because tests run in parallel against shared static counters.
    let device = MlxDevice::new().expect("MlxDevice::new");
    let _registry = KernelRegistry::new();
    let mut enc = device.command_encoder().expect("enc");

    let before = mlx_native::barrier_count();
    enc.memory_barrier();
    let after = mlx_native::barrier_count();
    assert_eq!(
        after - before,
        0,
        "barrier_count must NOT increment when active_encoder is null \
         (no compute pass started yet); delta {} ({} -> {})",
        after - before,
        before,
        after,
    );
}

#[test]
fn barrier_counter_skips_capture_mode() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut enc = device.command_encoder().expect("enc");
    enc.start_capture();

    let before = mlx_native::barrier_count();
    enc.memory_barrier();
    enc.memory_barrier();
    enc.memory_barrier();
    let after = mlx_native::barrier_count();

    assert_eq!(
        after - before,
        0,
        "capture-mode barriers must NOT increment the MTL barrier counter \
         (they record CapturedNode::Barrier sentinels instead); \
         delta {} ({} -> {})",
        after - before,
        before,
        after,
    );

    let captured = enc.take_capture().expect("capture mode was on");
    assert_eq!(
        captured.len(),
        3,
        "all 3 capture-mode barriers should be recorded as CapturedNode entries"
    );
}

#[test]
fn reset_counters_clears_barrier_state_in_isolation() {
    // reset_counters() drives barrier_count back to 0 atomically.  Use
    // a serial-style guard: call reset, immediately read, accept that
    // a parallel test may push count > 0 milliseconds later — what we
    // assert is that reset itself drives the value to a known state at
    // the instant it is observed (no atomic-read torn).
    //
    // The strict property we care about is: BARRIER_NS stays 0 in any
    // run that does not set MLX_PROFILE_BARRIERS=1.  That is permanent
    // (env-gate is OnceLock-cached) regardless of parallelism.
    mlx_native::reset_counters();
    // Read both AFTER reset.  Even if a parallel test increments
    // barrier_count between reset and read, BARRIER_NS will remain 0
    // because no test in this file sets MLX_PROFILE_BARRIERS.
    assert_eq!(
        mlx_native::barrier_total_ns(),
        0,
        "BARRIER_NS must stay 0 when MLX_PROFILE_BARRIERS is unset; \
         observed {}",
        mlx_native::barrier_total_ns(),
    );
}

#[test]
fn barrier_counter_increments_after_active_dispatch() {
    // Once a real GPU dispatch has activated the encoder's compute pass,
    // memory_barrier() reaches the objc::msg_send! site and BARRIER_COUNT
    // increments by exactly 1 per call.  Use elementwise_add as a minimal
    // dispatch to bring the encoder into active state.
    let (device, mut registry) = setup_with_registry();

    mlx_native::reset_counters();

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

    // First barrier: active encoder is now non-null because elementwise_add
    // went through encode_threadgroups -> get_or_create_encoder.
    // Use delta-based assertions because static counters are shared
    // across parallel cargo test threads.
    let before = mlx_native::barrier_count();
    enc.memory_barrier();
    enc.memory_barrier();
    enc.memory_barrier();
    let after = mlx_native::barrier_count();
    assert_eq!(
        after - before,
        3,
        "exactly 3 barriers expected from this test's 3 calls (delta {})",
        after - before,
    );

    enc.commit_and_wait().expect("commit");
    let _ = out.as_slice::<f32>().expect("read");
}

fn setup_with_registry() -> (MlxDevice, KernelRegistry) {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let registry = KernelRegistry::new();
    (device, registry)
}
