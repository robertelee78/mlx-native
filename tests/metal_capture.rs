//! Programmatic Metal frame capture tests (ADR-015 iter63 Phase B.3).
//!
//! Verifies that `mlx_native::metal_capture::MetalCapture::from_env`:
//!
//! 1. Returns `None` when `MLX_METAL_CAPTURE` is unset or empty
//!    (default production path — zero overhead).
//! 2. Returns `None` (or a captured `Some` whose `end()` is a clean
//!    no-op) when `MLX_METAL_CAPTURE` is set but
//!    `METAL_CAPTURE_ENABLED=1` is NOT — this is the "user forgot
//!    the framework gate" path; we must NOT panic and we must
//!    print an actionable stderr warning.
//! 3. The "happy path" (capture file actually written) requires
//!    `METAL_CAPTURE_ENABLED=1` to be set in the shell at process
//!    start — that env var is read by Apple's framework BEFORE main()
//!    runs, so a `set_var` from inside a test does NOT enable
//!    capture.  We document the manual recipe in the docstring of
//!    [`mlx_native::metal_capture::MetalCapture::from_env`] instead
//!    of asserting on it from CI.
//!
//! ## Env-cache + one-shot interaction
//!
//! `MetalCapture::from_env` is one-shot per process via a global
//! `AtomicBool` latch.  We expose `reset_capture_consumed_for_test`
//! (hidden from rustdoc) to reset the latch between test bodies.
//! `MLX_METAL_CAPTURE` itself is read fresh on every call so we can
//! `set_var` / `remove_var` mid-test.
//!
//! Cargo runs each `tests/*.rs` file in its own binary, so the latch
//! and AtomicI8 caches in this file do not collide with
//! `dispatch_profile.rs`.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::metal_capture::{reset_capture_consumed_for_test, MetalCapture};
use mlx_native::MlxDevice;

#[test]
fn from_env_returns_none_when_unset() {
    unsafe { std::env::remove_var("MLX_METAL_CAPTURE") };
    reset_capture_consumed_for_test();
    let device = MlxDevice::new().expect("device");
    let cap = MetalCapture::from_env(&device);
    assert!(
        cap.is_none(),
        "MLX_METAL_CAPTURE unset → from_env must return None"
    );
}

#[test]
fn from_env_returns_none_on_empty_string() {
    unsafe { std::env::set_var("MLX_METAL_CAPTURE", "") };
    reset_capture_consumed_for_test();
    let device = MlxDevice::new().expect("device");
    let cap = MetalCapture::from_env(&device);
    assert!(
        cap.is_none(),
        "MLX_METAL_CAPTURE=\"\" → from_env must return None"
    );
    unsafe { std::env::remove_var("MLX_METAL_CAPTURE") };
}

#[test]
fn from_env_one_shot_latch_is_consumed_after_first_call() {
    // First call with MLX_METAL_CAPTURE set may or may not return
    // Some(MetalCapture) depending on whether METAL_CAPTURE_ENABLED=1
    // was set at process start.  Either way, the consumer-latch
    // should flip; the SECOND call (without resetting) must return
    // None even with the same env state.
    unsafe { std::env::set_var("MLX_METAL_CAPTURE", "/tmp/mlx-iter63-noop.gputrace") };
    reset_capture_consumed_for_test();
    let device = MlxDevice::new().expect("device");
    let _first = MetalCapture::from_env(&device); // may be Some or None — both fine
    // Don't reset; subsequent call should be latched.
    let second = MetalCapture::from_env(&device);
    assert!(
        second.is_none(),
        "second from_env without reset → must return None (one-shot latch)"
    );
    unsafe { std::env::remove_var("MLX_METAL_CAPTURE") };
}

#[test]
fn from_env_does_not_panic_without_metal_capture_enabled() {
    // Smoke test: with MLX_METAL_CAPTURE set but METAL_CAPTURE_ENABLED
    // unset (typical user mistake), from_env should return None
    // gracefully and print a stderr warning — NOT panic.  This
    // exercises Risk R2 (PROFILING-KIT-DESIGN §B.6 "panic if Metal
    // capture is not enabled").  metal-rs 0.33 wraps start_capture
    // in try_objc! so the failure is a String error, not a panic.
    unsafe {
        std::env::set_var("MLX_METAL_CAPTURE", "/tmp/mlx-iter63-no-enable.gputrace");
        std::env::remove_var("METAL_CAPTURE_ENABLED");
    }
    reset_capture_consumed_for_test();
    let device = MlxDevice::new().expect("device");
    // Either None (start_capture rejected gracefully) OR Some+end-noop
    // is acceptable.  The contract is: do not panic.
    let cap = MetalCapture::from_env(&device);
    if let Some(mut c) = cap {
        c.end();
    }
    unsafe { std::env::remove_var("MLX_METAL_CAPTURE") };
}

#[test]
fn graph_session_finish_with_capture_unset_is_clean() {
    // Verify that GraphSession's MetalCapture wiring does not
    // perturb the default-off path.  Just begin → finish a trivial
    // session without any env vars set — must succeed.
    unsafe { std::env::remove_var("MLX_METAL_CAPTURE") };
    reset_capture_consumed_for_test();
    let device = MlxDevice::new().expect("device");
    let exec = mlx_native::GraphExecutor::new(device);
    let session = exec.begin().expect("begin");
    session.finish().expect("finish");
}
