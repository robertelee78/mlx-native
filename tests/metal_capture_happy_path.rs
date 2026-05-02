//! Happy-path Metal frame capture smoke test (ADR-015 iter63 Phase B).
//!
//! This test produces a real `.gputrace` file ONLY when both env vars
//! are set BEFORE invoking `cargo test`:
//!
//! ```bash
//! METAL_CAPTURE_ENABLED=1 \
//! MLX_METAL_CAPTURE=/tmp/iter63-smoke.gputrace \
//!     cargo test --release --test metal_capture_happy_path -- --nocapture
//! ```
//!
//! When `METAL_CAPTURE_ENABLED` is unset, the test passes trivially
//! (Apple's framework reads that var BEFORE main() runs, so a
//! `set_var` from inside the test cannot enable capture — see Phase
//! B.6 of PROFILING-KIT-DESIGN).  This shape is intentional: the
//! sourdough-safe default is for `cargo test` (which never sets that
//! var) to skip the assertion, while operators running the recipe
//! get a real file written.
//!
//! The test does NOT verify Xcode-openability — that requires a GUI.
//! It only checks that:
//! - the file exists at the configured path,
//! - the file is non-empty,
//! - the test harness exits cleanly after capture finalization.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::PathBuf;

use mlx_native::{GraphExecutor, MlxDevice};

#[test]
fn capture_writes_gputrace_when_envs_set() {
    let path = match std::env::var("MLX_METAL_CAPTURE") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!(
                "[test] MLX_METAL_CAPTURE unset — skipping happy-path \
                 capture verification (sourdough-safe default; the \
                 operator recipe sets METAL_CAPTURE_ENABLED=1 + \
                 MLX_METAL_CAPTURE=/path/to/file.gputrace before \
                 cargo test)"
            );
            return;
        }
    };
    if std::env::var("METAL_CAPTURE_ENABLED").ok().as_deref() != Some("1") {
        eprintln!(
            "[test] METAL_CAPTURE_ENABLED is not '1' — start_capture \
             will reject; skipping happy-path verification"
        );
        return;
    }

    let trace_path = PathBuf::from(&path);
    if trace_path.exists() {
        // Clean stale trace so the size check below is meaningful.
        // .gputrace is a directory (bundle); ignore errors.
        let _ = std::fs::remove_dir_all(&trace_path);
    }

    let device = MlxDevice::new().expect("device");
    // Reset latch so this happy-path runs even if a prior test in
    // the same process consumed it.  Cargo runs each test binary
    // in its own process, but defensive reset keeps the test
    // location-independent.
    mlx_native::metal_capture::reset_capture_consumed_for_test();

    {
        let exec = GraphExecutor::new(device);
        let session = exec.begin().expect("GraphExecutor::begin");
        // session.finish() is the natural commit boundary; the
        // capture was started inside begin() (since the env vars are
        // set) and ends inside Drop after commit_and_wait completes.
        session.finish().expect("session.finish");
    }
    // Brief sleep is NOT needed — Drop's stop_capture is synchronous.
    // Verify the trace file landed.
    assert!(
        trace_path.exists(),
        "expected .gputrace at {} after GraphSession::finish under \
         METAL_CAPTURE_ENABLED=1 + MLX_METAL_CAPTURE; not found",
        trace_path.display()
    );
    // .gputrace is a bundle (directory); a non-empty bundle has at
    // least one entry.
    let entries = std::fs::read_dir(&trace_path)
        .expect("read_dir")
        .count();
    assert!(
        entries > 0,
        ".gputrace bundle at {} is empty (size 0); capture didn't \
         finalize",
        trace_path.display()
    );
    eprintln!(
        "[test] capture verified: {} ({} bundle entries)",
        trace_path.display(),
        entries
    );
}
