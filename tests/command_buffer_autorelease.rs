//! Regression for command-buffer autorelease accumulation on long-lived Rust threads.
//!
//! `-[MTLCommandQueue commandBuffer]` returns an autoreleased object.  A
//! worker thread without a local autorelease pool can therefore retain every
//! command buffer until AGX blocks the next allocation.  The production Qwen
//! serving receipt reached that ceiling with 37,999 live command-buffer
//! objects and a worker blocked in `-[MTLCommandQueue commandBuffer]`.

#![allow(clippy::expect_used, unexpected_cfgs)]

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use metal::ComputePipelineDescriptor;
use mlx_native::MlxDevice;
use objc::{msg_send, runtime::Object, sel, sel_impl};

const CHILD_ENV: &str = "MLX_COMMAND_BUFFER_AUTORELEASE_CHILD";
const TEST_NAME: &str = "uncommitted_command_buffers_are_reclaimed_on_poolless_workers";
const ITERATIONS: usize = 50_000;
const CHILD_TIMEOUT: Duration = Duration::from_secs(120);
const LABEL_CHILD_ENV: &str = "MLX_LABEL_CFSTRING_CHILD_MODE";
const LABEL_SOCKET_ENV: &str = "MLX_LABEL_CFSTRING_CONTROL_SOCKET";
const LABEL_TEST_NAME: &str = "labeled_commits_have_bounded_cfstring_population";
const LABEL_WARMUP_ITERATIONS: usize = 256;
const LABEL_WAVE_ITERATIONS: usize = 10_000;
const LABEL_CHILD_TIMEOUT: Duration = Duration::from_secs(120);
const HEAP_TIMEOUT: Duration = Duration::from_secs(20);
const CHECKPOINT_PREFIX: &str = "MLX_CFSTRING_CHECKPOINT";
const LABEL_MODES: [&str; 6] = [
    "sync",
    "async-drop",
    "async-wait",
    "unlabeled-async",
    "command-buffer-only",
    "compute-encoder-only",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeapPopulation {
    cfstrings: usize,
    command_buffers: usize,
    autorelease_pool_pages: usize,
}

fn parse_count(token: &str) -> Result<usize, String> {
    token
        .replace(',', "")
        .parse::<usize>()
        .map_err(|error| format!("invalid heap count {token:?}: {error}"))
}

fn parse_heap_population(report: &str) -> Result<HeapPopulation, String> {
    let mut cfstrings = None;
    let mut command_buffers = 0usize;
    let mut autorelease_pool_pages = None;

    for line in report.lines() {
        let fields = line.split_ascii_whitespace().collect::<Vec<_>>();
        if fields.len() < 5 {
            continue;
        }
        let object_type = fields[3];
        if object_type == "CFString" && fields[4] == "ObjC" {
            let count = parse_count(fields[0])?;
            if cfstrings.replace(count).is_some() {
                return Err("duplicate exact `CFString ObjC` heap row".to_string());
            }
            continue;
        }
        if (object_type.contains("FamilyCommandBuffer")
            || object_type.starts_with("IOGPUMetalCommandBuffer"))
            && !object_type.contains("StoragePool")
        {
            command_buffers = command_buffers
                .checked_add(parse_count(fields[0])?)
                .ok_or_else(|| "command-buffer heap count overflow".to_string())?;
            continue;
        }
        if object_type == "@autoreleasepool" && fields[4] == "content" {
            let count = parse_count(fields[0])?;
            if autorelease_pool_pages.replace(count).is_some() {
                return Err("duplicate `@autoreleasepool content` heap row".to_string());
            }
        }
    }

    Ok(HeapPopulation {
        cfstrings: cfstrings.ok_or_else(|| "missing exact `CFString ObjC` heap row".to_string())?,
        command_buffers,
        autorelease_pool_pages: autorelease_pool_pages
            .ok_or_else(|| "missing exact `@autoreleasepool content` heap row".to_string())?,
    })
}

fn wait_for_process(child: &mut Child, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        match child
            .try_wait()
            .map_err(|error| format!("poll child process: {error}"))?
        {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("child process failed with {status}")),
            None if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("child process exceeded {timeout:?}"));
            }
        }
    }
}

fn heap_population(pid: u32) -> Result<HeapPopulation, String> {
    let mut heap = Command::new("/usr/bin/heap")
        .args(["-q", &pid.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("spawn /usr/bin/heap for pid {pid}: {error}"))?;
    let mut stdout = heap
        .stdout
        .take()
        .ok_or_else(|| "heap stdout pipe missing".to_string())?;
    let mut stderr = heap
        .stderr
        .take()
        .ok_or_else(|| "heap stderr pipe missing".to_string())?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let process_result = wait_for_process(&mut heap, HEAP_TIMEOUT);
    let stdout = stdout_reader
        .join()
        .map_err(|_| "heap stdout reader panicked".to_string())?
        .map_err(|error| format!("read heap stdout: {error}"))?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| "heap stderr reader panicked".to_string())?
        .map_err(|error| format!("read heap stderr: {error}"))?;
    process_result?;
    if !stderr.is_empty() {
        return Err(format!(
            "/usr/bin/heap emitted stderr: {}",
            String::from_utf8_lossy(&stderr)
        ));
    }
    let report =
        String::from_utf8(stdout).map_err(|error| format!("heap stdout is not UTF-8: {error}"))?;
    parse_heap_population(&report)
}

fn emit_checkpoint(control: &mut UnixStream, phase: usize) {
    let pid = std::process::id();
    writeln!(control, "{CHECKPOINT_PREFIX}\tv1\t{phase}\t{pid}").expect("write heap checkpoint");
    control.flush().expect("flush heap checkpoint");
    let mut response = String::new();
    BufReader::new(control.try_clone().expect("clone heap control socket"))
        .read_line(&mut response)
        .expect("read heap checkpoint continuation");
    assert_eq!(
        response.trim_end(),
        format!("CONTINUE\tv1\t{phase}"),
        "parent must acknowledge the exact heap checkpoint"
    );
}

fn drain_queue(device: &MlxDevice, ordinal: usize) {
    let mut drain = device
        .command_encoder()
        .unwrap_or_else(|error| panic!("create label-lifetime drain {ordinal}: {error}"));
    drain
        .commit_and_wait()
        .unwrap_or_else(|error| panic!("drain label-lifetime queue {ordinal}: {error}"));
}

fn run_label_iterations(
    mode: &str,
    device: &MlxDevice,
    pipeline: &metal::ComputePipelineStateRef,
    start: usize,
    count: usize,
) {
    for ordinal in start..start + count {
        let mut encoder = device
            .command_encoder()
            .unwrap_or_else(|error| panic!("create {mode} command buffer {ordinal}: {error}"));
        if mode == "command-buffer-only" {
            objc::rc::autoreleasepool(|| {
                encoder
                    .metal_command_buffer()
                    .set_label("autorelease.label.population");
            });
            encoder
                .commit_and_wait()
                .unwrap_or_else(|error| panic!("commit command-buffer label {ordinal}: {error}"));
            drop(encoder);
            continue;
        }
        if mode == "compute-encoder-only" {
            let active_encoder = objc::rc::autoreleasepool(|| {
                let active_encoder = encoder.metal_command_buffer().new_compute_command_encoder();
                // Mirror CommandEncoder's production borrowed +0 -> explicit
                // +1 lifetime extension across the factory pool.
                let _: *mut Object = unsafe { msg_send![active_encoder, retain] };
                active_encoder as *const metal::ComputeCommandEncoderRef
            });
            objc::rc::autoreleasepool(|| unsafe {
                (&*active_encoder).set_label("autorelease.label.population");
                (&*active_encoder).end_encoding();
                let nil_label: *const Object = std::ptr::null();
                let _: () = msg_send![active_encoder, setLabel: nil_label];
                let _: () = msg_send![active_encoder, release];
            });
            encoder
                .commit_and_wait()
                .unwrap_or_else(|error| panic!("commit compute-encoder label {ordinal}: {error}"));
            drop(encoder);
            continue;
        }
        encoder.set_pipeline(pipeline);
        match mode {
            "sync" => encoder
                .commit_and_wait_labeled("autorelease.label.population")
                .unwrap_or_else(|error| panic!("sync labeled commit {ordinal}: {error}")),
            "async-drop" => encoder.commit_labeled("autorelease.label.population"),
            "async-wait" => {
                encoder.commit_labeled("autorelease.label.population");
                encoder
                    .wait_until_completed()
                    .unwrap_or_else(|error| panic!("wait labeled commit {ordinal}: {error}"));
            }
            "unlabeled-async" => encoder.commit(),
            other => panic!("unknown label-lifetime mode {other}"),
        }
        drop(encoder);
        if !matches!(mode, "sync" | "async-wait") && ordinal % 32 == 31 {
            drain_queue(device, ordinal);
        }
    }
    drain_queue(device, start + count);
}

fn run_label_population_child(mode: &str) {
    let worker_mode = mode.to_string();
    thread::spawn(move || {
        let socket_path = std::env::var(LABEL_SOCKET_ENV)
            .expect("parent must provide the heap control socket path");
        let mut control = UnixStream::connect(&socket_path)
            .unwrap_or_else(|error| panic!("connect heap control socket {socket_path}: {error}"));
        control
            .set_read_timeout(Some(LABEL_CHILD_TIMEOUT))
            .expect("set heap control read timeout");
        let device = MlxDevice::new().expect("create Metal device");
        let pipeline = build_noop_pipeline(device.metal_device());
        if worker_mode == "negative-control" {
            emit_checkpoint(&mut control, 0);
            let encoder = device
                .command_encoder()
                .expect("create negative-control command buffer");
            for _ in 0..2_048 {
                encoder
                    .metal_command_buffer()
                    .set_label("autorelease.label.negative_control");
            }
            emit_checkpoint(&mut control, 2_048);
            drop(encoder);
            return;
        }

        run_label_iterations(&worker_mode, &device, &pipeline, 0, LABEL_WARMUP_ITERATIONS);
        emit_checkpoint(&mut control, 0);
        run_label_iterations(
            &worker_mode,
            &device,
            &pipeline,
            LABEL_WARMUP_ITERATIONS,
            LABEL_WAVE_ITERATIONS,
        );
        emit_checkpoint(&mut control, LABEL_WAVE_ITERATIONS);
        run_label_iterations(
            &worker_mode,
            &device,
            &pipeline,
            LABEL_WARMUP_ITERATIONS + LABEL_WAVE_ITERATIONS,
            LABEL_WAVE_ITERATIONS,
        );
        emit_checkpoint(&mut control, 2 * LABEL_WAVE_ITERATIONS);
    })
    .join()
    .expect("label-population worker panicked");
}

fn wait_for_checkpoint(
    receiver: &mpsc::Receiver<String>,
    expected_phase: usize,
    expected_pid: u32,
) -> Result<(), String> {
    let deadline = Instant::now() + LABEL_CHILD_TIMEOUT;
    loop {
        if Instant::now() >= deadline {
            return Err(format!(
                "heap checkpoint {expected_phase} exceeded {LABEL_CHILD_TIMEOUT:?}"
            ));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        let line = receiver
            .recv_timeout(remaining)
            .map_err(|error| format!("wait for heap checkpoint {expected_phase}: {error}"))?;
        if line.starts_with("READ_ERROR:") {
            return Err(format!("read heap control socket: {line}"));
        }
        if !line.starts_with(CHECKPOINT_PREFIX) {
            continue;
        }
        let fields = line.trim_end().split('\t').collect::<Vec<_>>();
        if fields.len() != 4
            || fields[0] != CHECKPOINT_PREFIX
            || fields[1] != "v1"
            || fields[2] != expected_phase.to_string()
            || fields[3] != expected_pid.to_string()
        {
            return Err(format!(
                "malformed/out-of-order heap checkpoint: expected phase={expected_phase} pid={expected_pid}, got {line:?}"
            ));
        }
        return Ok(());
    }
}

fn signed_delta(after: usize, before: usize) -> i64 {
    i64::try_from(after).expect("heap count fits i64")
        - i64::try_from(before).expect("heap count fits i64")
}

fn run_heap_child(mode: &str, phases: &[usize]) -> Result<Vec<HeapPopulation>, String> {
    let executable =
        std::env::current_exe().map_err(|error| format!("locate test binary: {error}"))?;
    let socket_path = format!("/tmp/mlx-label-cfstring-{}-{mode}.sock", std::process::id());
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)
        .map_err(|error| format!("bind heap control socket {socket_path}: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("set heap control listener nonblocking: {error}"))?;
    let mut child = match Command::new(executable)
        .args([
            "--exact",
            LABEL_TEST_NAME,
            "--nocapture",
            "--test-threads=1",
        ])
        .env(LABEL_CHILD_ENV, mode)
        .env(LABEL_SOCKET_ENV, &socket_path)
        .env_remove("MLX_PROFILE_CB")
        .env_remove("MLX_PROFILE_DISPATCH")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let _ = std::fs::remove_file(&socket_path);
            return Err(format!("spawn label-population child for {mode}: {error}"));
        }
    };
    let pid = child.id();
    let accept_deadline = Instant::now() + LABEL_CHILD_TIMEOUT;
    let mut control = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock
                    && Instant::now() < accept_deadline =>
            {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        let _ = std::fs::remove_file(&socket_path);
                        return Err(format!(
                            "label-population child exited before socket accept with {status}"
                        ));
                    }
                    Ok(None) => {}
                    Err(poll_error) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = std::fs::remove_file(&socket_path);
                        return Err(format!(
                            "poll label-population child before socket accept: {poll_error}"
                        ));
                    }
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&socket_path);
                return Err(format!("accept heap control socket: {error}"));
            }
        }
    };
    let _ = std::fs::remove_file(&socket_path);
    let control_reader = match (|| {
        control
            .set_nonblocking(false)
            .map_err(|error| format!("restore blocking heap control socket: {error}"))?;
        control
            .set_write_timeout(Some(LABEL_CHILD_TIMEOUT))
            .map_err(|error| format!("set heap control write timeout: {error}"))?;
        control
            .try_clone()
            .map_err(|error| format!("clone parent heap control socket: {error}"))
    })() {
        Ok(reader) => reader,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(control_reader).lines() {
            if sender
                .send(line.unwrap_or_else(|error| format!("READ_ERROR:{error}")))
                .is_err()
            {
                break;
            }
        }
    });

    let result = (|| {
        let mut populations = Vec::with_capacity(phases.len());
        for &phase in phases {
            wait_for_checkpoint(&receiver, phase, pid)?;
            populations.push(heap_population(pid)?);
            writeln!(control, "CONTINUE\tv1\t{phase}")
                .map_err(|error| format!("continue child at phase {phase}: {error}"))?;
            control
                .flush()
                .map_err(|error| format!("flush child continuation at phase {phase}: {error}"))?;
        }
        drop(control);
        wait_for_process(&mut child, LABEL_CHILD_TIMEOUT)?;
        Ok(populations)
    })();

    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let _ = reader.join();
    result
}

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

#[test]
fn labeled_commits_have_bounded_cfstring_population() {
    if let Ok(mode) = std::env::var(LABEL_CHILD_ENV) {
        run_label_population_child(&mode);
        return;
    }

    let negative = run_heap_child("negative-control", &[0, 2_048])
        .expect("heap must observe the deliberately unpooled label control");
    let negative_delta = signed_delta(negative[1].cfstrings, negative[0].cfstrings);
    assert!(
        negative_delta >= 1_024,
        "heap detector is blind to the known unpooled label leak: {negative:?}, delta={negative_delta}"
    );

    let mut violations = Vec::new();
    for mode in LABEL_MODES {
        let populations =
            run_heap_child(mode, &[0, LABEL_WAVE_ITERATIONS, 2 * LABEL_WAVE_ITERATIONS])
                .unwrap_or_else(|error| panic!("run {mode} label-population child: {error}"));
        let first_delta = signed_delta(populations[1].cfstrings, populations[0].cfstrings);
        let second_delta = signed_delta(populations[2].cfstrings, populations[1].cfstrings);
        let total_delta = signed_delta(populations[2].cfstrings, populations[0].cfstrings);
        let first_pool_delta = signed_delta(
            populations[1].autorelease_pool_pages,
            populations[0].autorelease_pool_pages,
        );
        let second_pool_delta = signed_delta(
            populations[2].autorelease_pool_pages,
            populations[1].autorelease_pool_pages,
        );
        eprintln!(
            "label lifetime mode={mode} populations={populations:?} cfstring_deltas=[{first_delta},{second_delta}] pool_page_deltas=[{first_pool_delta},{second_pool_delta}]"
        );

        if !populations.iter().all(|sample| sample.command_buffers == 0) {
            violations.push(format!(
                "{mode} retained live Metal command buffers at a drained checkpoint: {populations:?}"
            ));
        }
        if first_delta > 256 || second_delta > 256 || total_delta > 512 {
            violations.push(format!(
                "{mode} retained workload-linear CFStrings: populations={populations:?}, deltas=[{first_delta},{second_delta},{total_delta}]"
            ));
        }
        if first_pool_delta > 8 || second_pool_delta > 8 {
            violations.push(format!(
                "{mode} retained workload-linear autorelease-pool pages: populations={populations:?}, deltas=[{first_pool_delta},{second_pool_delta}]"
            ));
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

#[test]
fn heap_population_parser_is_fail_closed() {
    let report = r#"
      101,820    4,837,312      47.5   CFString                                          ObjC    CoreFoundation
          210      860,160    4096.0   @autoreleasepool content                          C       libobjc.A.dylib
            4          224      56.0   CFString (Storage)                                C       CoreFoundation
            2        1,792     896.0   AGXG17XFamilyCommandBuffer                       ObjC    AGXMetalG17X
            1          896     896.0   AGXG17XFamilyCommandBuffer._impl                 C++     AGXMetalG17X
            1          128     128.0   AGXG17CDevice._commandBufferStoragePool          C++     AGXMetalG17X
    "#;
    assert_eq!(
        parse_heap_population(report).expect("parse representative heap rows"),
        HeapPopulation {
            cfstrings: 101_820,
            command_buffers: 3,
            autorelease_pool_pages: 210,
        }
    );

    let missing = "4 224 56.0 CFString (Storage) C CoreFoundation";
    assert!(
        parse_heap_population(missing).is_err(),
        "storage-only CFString row must not satisfy the exact ObjC population gate"
    );
    let duplicate =
        "1 48 48.0 CFString ObjC CoreFoundation\n2 96 48.0 CFString ObjC CoreFoundation";
    assert!(
        parse_heap_population(duplicate).is_err(),
        "duplicate exact CFString rows must fail closed"
    );
    let missing_pool = "1 48 48.0 CFString ObjC CoreFoundation";
    assert!(
        parse_heap_population(missing_pool).is_err(),
        "missing autorelease-pool population row must fail closed"
    );
}
