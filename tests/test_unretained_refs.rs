//! Smoke test for ADR-015 iter13 — `MLX_UNRETAINED_REFS=1` env gate.
//!
//! When the env var is set at process start, `CommandEncoder::new_with_residency`
//! constructs each `MTLCommandBuffer` via
//! `CommandQueueRef::new_command_buffer_with_unretained_references` instead of
//! the default `commandBuffer`.  llama.cpp uses this same call at
//! `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:512` to skip
//! per-buffer-binding ARC retains on submit; the docstring on
//! `mlx-native/src/encoder.rs:392-411` cites ~3-5% wall on M-series GPUs.
//!
//! What this test verifies:
//! * Default-off: a process WITHOUT `MLX_UNRETAINED_REFS=1` runs an
//!   elementwise_add round-trip end-to-end and produces correct results.
//!   This is the sourdough-safe path — every iter to date has implicitly
//!   exercised this; the test pins it explicitly so any regression to the
//!   gate plumbing surfaces here.
//!
//! What this test does NOT cover (intentional):
//! * Cargo's parallel test runner does not allow per-test environment
//!   variables to be reliably set BEFORE the OnceLock cache fires —
//!   `unretained_refs_enabled()` reads the var on first call and caches.
//!   In a process where any earlier test has constructed an encoder, the
//!   cache is already poisoned to the process-startup value.  An
//!   integration-level smoke for the gate-on path is exercised at the
//!   hf2q decode bench level (paired baseline vs `MLX_UNRETAINED_REFS=1`
//!   end-to-end run with parity gate); see ADR-015 iter13 §Bench
//!   methodology.
//!
//! ADR-015 iter13 — claude implementer worktree
//! `cfa-20260428-adr015-iter13-claude`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::{DType, KernelRegistry, MlxDevice};

#[test]
fn default_off_path_runs_elementwise_add_correctly() {
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
    enc.commit_and_wait().expect("commit");

    let result = out.as_slice::<f32>().expect("read");
    assert_eq!(
        result,
        &[6.0, 8.0, 10.0, 12.0],
        "default-path elementwise_add must produce correct results regardless of \
         MLX_UNRETAINED_REFS state"
    );
}
