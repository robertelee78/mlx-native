//! Regression for command-buffer autorelease accumulation on long-lived Rust threads.
//!
//! `-[MTLCommandQueue commandBuffer]` returns an autoreleased object.  A
//! worker thread without a local autorelease pool can therefore retain every
//! command buffer until AGX blocks the next allocation.  The production Qwen
//! serving receipt reached that ceiling with 37,999 live command-buffer
//! objects and a worker blocked in `-[MTLCommandQueue commandBuffer]`.

#![allow(clippy::expect_used)]

use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use metal::ComputePipelineDescriptor;
use mlx_native::MlxDevice;

const CHILD_ENV: &str = "MLX_COMMAND_BUFFER_AUTORELEASE_CHILD";
const TEST_NAME: &str = "uncommitted_command_buffers_are_reclaimed_on_poolless_workers";
const ITERATIONS: usize = 50_000;
const CHILD_TIMEOUT: Duration = Duration::from_secs(120);

fn build_noop_pipeline(device: &metal::DeviceRef) -> metal::ComputePipelineState {
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void command_buffer_autorelease_noop() {}
    "#;
    let library = device
        .new_library_with_source(source, &metal::CompileOptions::new())
        .expect("compile autorelease no-op kernel");
    let function = library
        .get_function("command_buffer_autorelease_noop", None)
        .expect("load autorelease no-op kernel");
    let descriptor = ComputePipelineDescriptor::new();
    descriptor.set_compute_function(Some(&function));
    device
        .new_compute_pipeline_state(&descriptor)
        .expect("create autorelease no-op pipeline")
}

/// Exercise both production lifetime shapes far beyond the observed cliff.
///
/// The child process is load-bearing: a regression blocks inside Objective-C
/// before Rust can return an error, so an in-process timeout would leave a
/// wedged Metal thread behind in the test harness.  The parent kills the
/// isolated child and fails on a generous hosted-runner deadline instead.
#[test]
fn uncommitted_command_buffers_are_reclaimed_on_poolless_workers() {
    if std::env::var_os(CHILD_ENV).is_some() {
        let device = MlxDevice::new().expect("create Metal device");
        let pipeline = build_noop_pipeline(device.metal_device());
        for ordinal in 0..ITERATIONS {
            let mut encoder = device
                .command_encoder()
                .unwrap_or_else(|error| panic!("create command buffer {ordinal}: {error}"));
            // `set_pipeline` lazily opens the production concurrent compute
            // encoder. Dropping it without committing covers the raw retained
            // pointer crossing the local autorelease-pool drain.
            encoder.set_pipeline(&pipeline);
            drop(encoder);
        }

        // Exercise the Metal label bridge at the same cumulative scale. The
        // async commits preserve throughput; a periodic synchronous sentinel
        // drains the queue and keeps this a lifetime test rather than an
        // in-flight-depth test.
        for ordinal in 0..ITERATIONS {
            let mut encoder = device
                .command_encoder()
                .unwrap_or_else(|error| panic!("create labeled command buffer {ordinal}: {error}"));
            // Keep a compute encoder active so `commit_labeled` exercises
            // both command-buffer and compute-encoder label setters.
            encoder.set_pipeline(&pipeline);
            encoder.commit_labeled("autorelease.label.churn");
            if ordinal % 32 == 31 {
                let mut drain = device
                    .command_encoder()
                    .unwrap_or_else(|error| panic!("create label drain {ordinal}: {error}"));
                drain
                    .commit_and_wait()
                    .unwrap_or_else(|error| panic!("drain labeled commands {ordinal}: {error}"));
            }
        }

        // Reproduce hf2q's exact former final-layer lifecycle: drain the
        // session, rotate to a fresh empty CB, then drop without submitting
        // that replacement.  Before the scoped pools, the autoreleased +0
        // object survived every Rust-owned +1 drop on this pool-less thread.
        for ordinal in 0..ITERATIONS {
            let mut session = device
                .encoder_session()
                .unwrap_or_else(|error| panic!("create encoder session {ordinal}: {error}"))
                .expect("child enables encoder sessions");
            session
                .commit_and_wait()
                .unwrap_or_else(|error| panic!("commit encoder session {ordinal}: {error}"));
            session
                .reset_for_next_stage()
                .unwrap_or_else(|error| panic!("reset encoder session {ordinal}: {error}"));
            drop(session);
        }

        let mut sentinel = device
            .command_encoder()
            .expect("create sentinel command buffer");
        sentinel
            .commit_and_wait()
            .expect("commit sentinel command buffer");
        return;
    }

    let executable = std::env::current_exe().expect("locate test executable");
    let mut child = Command::new(executable)
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env("HF2Q_ENCODER_SESSION", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn isolated command-buffer regression child");

    let deadline = Instant::now() + CHILD_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait().expect("poll regression child") {
            assert!(status.success(), "regression child failed with {status}");
            break;
        }
        if Instant::now() >= deadline {
            child.kill().expect("kill wedged regression child");
            let _ = child.wait();
            panic!("command-buffer allocation wedged after autorelease accumulation");
        }
        thread::sleep(Duration::from_millis(10));
    }
}
