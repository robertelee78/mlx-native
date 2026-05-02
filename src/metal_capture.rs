//! Programmatic Metal Frame Capture wrapping (ADR-015 iter63 Part B).
//!
//! Mirrors llama.cpp's `GGML_METAL_CAPTURE_COMPUTE` env-driven capture
//! pattern (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m`
//! lines 161-170 + 488-608).  Triggered by:
//!
//! ```bash
//! METAL_CAPTURE_ENABLED=1 \
//! MLX_METAL_CAPTURE=/path/to/output.gputrace \
//!     cargo run --release --bin hf2q -- generate ...
//! ```
//!
//! The resulting `.gputrace` document opens in Xcode → Performance →
//! GPU for full GPU-timeline analysis (idle gaps, scheduling stalls,
//! memory pressure visualization).
//!
//! ## One-shot semantics
//!
//! Capture is one-shot per process: the FIRST [`MetalCapture::from_env`]
//! call after the env var is set returns `Some(MetalCapture)`; every
//! subsequent call returns `None` (a process-global `AtomicBool`
//! latches after the first consume).  This mirrors llama.cpp's
//! `capture_compute = 1; capture_compute--` countdown semantics —
//! the very first decode/prefill forward pass is captured, and the
//! kit goes quiet thereafter.  Re-running capture requires a fresh
//! process.
//!
//! ## Permissions / entitlements
//!
//! From metal-rs 0.33 capturemanager.rs:75-79:
//! > *Capture can be enabled by either:*
//! > *1. Running from Xcode*
//! > *2. Setting the environment variable `METAL_CAPTURE_ENABLED=1`*
//! > *3. Adding an info.plist file containing the `MetalCaptureEnabled`
//! >    key set to `YES`*
//!
//! For Cargo-built binaries (no Xcode), users MUST `export
//! METAL_CAPTURE_ENABLED=1` alongside `MLX_METAL_CAPTURE=...`.
//! Writing a `.gputrace` to disk requires no special entitlements
//! (only `GpuTraceDocument`; the `DeveloperTools` destination would
//! need Xcode running).
//!
//! [`MetalCapture::from_env`] is defensive: it checks
//! `mgr.supports_destination(GpuTraceDocument)` first and inspects the
//! `Result<(), String>` from `start_capture` (no panic).  On any
//! failure path the function returns `None` after a one-shot stderr
//! warning so callers can continue without capture.

use std::sync::atomic::{AtomicBool, Ordering};

use metal::{CaptureDescriptor, CaptureManager, CaptureScope, MTLCaptureDestination};

use crate::MlxDevice;

/// Process-global one-shot latch.  Set to `true` the first time a
/// `MetalCapture` is constructed; subsequent `from_env` calls return
/// `None` so a single bench run captures exactly the first forward
/// pass (mirrors llama.cpp's countdown semantics).
static CAPTURE_CONSUMED: AtomicBool = AtomicBool::new(false);

/// A live programmatic capture session backed by an `MTLCaptureScope`.
///
/// Construct via [`MetalCapture::from_env`].  Pair `begin` /
/// `end` calls around the unit of work to capture (e.g. a single
/// forward pass).  `Drop` runs `end` defensively in case the caller
/// forgets, so a panic mid-forward-pass still flushes a partial trace.
pub struct MetalCapture {
    scope: CaptureScope,
    /// Whether the scope is currently open (`begin_scope` called
    /// without a matching `end_scope`).  Set true by
    /// [`Self::begin`]; cleared by [`Self::end`] / `Drop`.
    started: bool,
    /// The output URL for the trace document.  Stored for stderr
    /// reporting at `end()`.
    output_path: String,
}

impl MetalCapture {
    /// Initialize a capture from the env vars `MLX_METAL_CAPTURE` (output
    /// path) and `METAL_CAPTURE_ENABLED` (Apple's framework-level
    /// permission gate).
    ///
    /// Returns `Some(MetalCapture)` only when ALL of:
    /// 1. `MLX_METAL_CAPTURE` is set to a non-empty path,
    /// 2. The capture manager supports
    ///    `MTLCaptureDestination::GpuTraceDocument`,
    /// 3. `start_capture` succeeds (which requires
    ///    `METAL_CAPTURE_ENABLED=1` or running under Xcode), and
    /// 4. The process-global one-shot latch [`CAPTURE_CONSUMED`] has
    ///    not yet flipped to `true`.
    ///
    /// On any failure path returns `None` after a one-shot stderr
    /// warning describing the cause.  The caller is expected to
    /// ignore the `None` and proceed without capture — never panic.
    pub fn from_env(device: &MlxDevice) -> Option<Self> {
        // 1. Env-var read.  Empty string treated as unset.
        let path = match std::env::var("MLX_METAL_CAPTURE") {
            Ok(s) if !s.is_empty() => s,
            _ => return None,
        };
        // 4. One-shot latch (checked early so the warning fires only
        // for the first consumer of the env, not every forward pass).
        if CAPTURE_CONSUMED
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }
        // 2. Destination support check.
        let mgr = CaptureManager::shared();
        if !mgr.supports_destination(MTLCaptureDestination::GpuTraceDocument) {
            eprintln!(
                "[mlx-native] MLX_METAL_CAPTURE={} ignored: \
                 GpuTraceDocument destination unsupported on this device",
                path
            );
            return None;
        }
        // Build descriptor + scope.  We scope to the device's command
        // queue so any CB enqueued through the existing
        // CommandEncoder (which uses device.metal_queue() under the
        // hood) is captured.
        let scope = mgr.new_capture_scope_with_command_queue(device.metal_queue());
        let descriptor = CaptureDescriptor::new();
        descriptor.set_capture_scope(&scope);
        descriptor.set_destination(MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_output_url(&path);
        // 3. Start capture (returns Result<(), String>).  Failure
        // typically means METAL_CAPTURE_ENABLED is unset.
        match mgr.start_capture(&descriptor) {
            Ok(()) => {
                eprintln!(
                    "[mlx-native] MTLCaptureManager: starting capture to {}",
                    path
                );
                Some(Self {
                    scope,
                    started: false,
                    output_path: path,
                })
            }
            Err(e) => {
                eprintln!(
                    "[mlx-native] MLX_METAL_CAPTURE={} capture start failed: {} \
                     (set METAL_CAPTURE_ENABLED=1?)",
                    path, e
                );
                None
            }
        }
    }

    /// Open the capture scope.  Idempotent: a second `begin` without
    /// an intervening `end` is a no-op so callers can wrap nested
    /// forward passes safely.
    ///
    /// Call at the start of the unit of work to capture (e.g.
    /// `GraphExecutor::begin` — see graph.rs wire-up).
    pub fn begin(&mut self) {
        if self.started {
            return;
        }
        self.scope.begin_scope();
        self.started = true;
    }

    /// Close the capture scope and stop the underlying
    /// `MTLCaptureManager`.  After `end` returns the `.gputrace` file
    /// is finalized and openable in Xcode.
    ///
    /// Idempotent.  Calling `end` without a preceding `begin` is a
    /// no-op (defensive — ensures `Drop` after a panic between
    /// construction and `begin` doesn't `endScope` an unopened scope).
    pub fn end(&mut self) {
        if !self.started {
            return;
        }
        self.scope.end_scope();
        CaptureManager::shared().stop_capture();
        self.started = false;
        eprintln!(
            "[mlx-native] MTLCaptureManager: stopped (trace at {})",
            self.output_path
        );
    }
}

impl Drop for MetalCapture {
    fn drop(&mut self) {
        // Defensive flush: if the caller forgot `end`, still finalize
        // the trace.  Idempotent so this is safe even when end() was
        // already called explicitly.
        self.end();
    }
}

/// Test-only reset of the one-shot latch.  Hidden from rustdoc and
/// gated behind `#[doc(hidden)]` because production callers must NOT
/// flip the latch manually — the once-per-process semantics are part
/// of the contract.  Cargo runs each test binary in a fresh process,
/// but unit tests inside the same binary need this hook to exercise
/// the `from_env` path more than once.
#[doc(hidden)]
pub fn reset_capture_consumed_for_test() {
    CAPTURE_CONSUMED.store(false, Ordering::SeqCst);
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn from_env_returns_none_when_unset() {
        // Ensure the var is unset (does not affect other tests because
        // the latch is process-global; this test just verifies the
        // env-empty path).
        unsafe { std::env::remove_var("MLX_METAL_CAPTURE") };
        reset_capture_consumed_for_test();
        let device = MlxDevice::new().expect("MlxDevice::new");
        assert!(
            MetalCapture::from_env(&device).is_none(),
            "MLX_METAL_CAPTURE unset → from_env must return None"
        );
    }

    #[test]
    fn from_env_returns_none_on_empty_string() {
        unsafe { std::env::set_var("MLX_METAL_CAPTURE", "") };
        reset_capture_consumed_for_test();
        let device = MlxDevice::new().expect("device");
        assert!(
            MetalCapture::from_env(&device).is_none(),
            "MLX_METAL_CAPTURE=\"\" → from_env must return None"
        );
        unsafe { std::env::remove_var("MLX_METAL_CAPTURE") };
    }
}
