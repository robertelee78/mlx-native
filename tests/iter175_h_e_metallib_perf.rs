//! ADR-029 iter-175 Step 1k — H-E empirical test:
//! does `xcrun metal -O3` precompiled .metallib produce a faster kernel
//! at runtime than Apple's runtime `newLibraryWithSource:`?
//!
//! Compiles `quantized_matmul_ggml.metal` two ways at test runtime:
//!   1. xcrun-precompiled (-O3) .metallib loaded via `new_library_with_file`
//!   2. Runtime source-compiled via `new_library_with_source` (default options)
//!
//! Creates a pipeline for `kernel_mul_mv_q6_K_f32_nr2` with the same function
//! constants (700:1, 701:1, 702:1) used in production, then dispatches each
//! N=1000 times at gemma4 Q_sliding decode shape (m=1, n=4096, k=2816).
//! Prints per-call median µs for each path.
//!
//! Run via: `cargo test --release --test iter175_h_e_metallib_perf -- --nocapture`
//!
//! Expected outcomes:
//! - Per-call delta > 5%: H-E CONFIRMED → multi-day port to precompiled metallib path
//! - Per-call delta within noise: H-E FALSIFIED → close iter-175 at structural parity

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use metal::{ComputePipelineDescriptor, FunctionConstantValues, MTLDataType, MTLSize};
use mlx_native::{DType, MlxBuffer, MlxDevice};

const KERNEL_NAME: &str = "kernel_mul_mv_q6_K_f32_nr2";
const SHADER_PATH: &str = "src/shaders/quantized_matmul_ggml.metal";

// gemma4 Q_sliding decode shape: m=1 (decode), n=4096 (q-heads × head_dim), k=2816 (hidden).
const Q_M: u32 = 1;
const Q_N: u32 = 4096;
const Q_K: u32 = 2816;

const WARMUP: usize = 10;
const MEASURE: usize = 50;
const BATCH: usize = 32; // dispatches per CB to amortize sync

// Q6_K block layout: 256 elements per block, 210 bytes per block.
const QK_K: u64 = 256;
const Q6_K_BLOCK_BYTES: u64 = 210;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GgmlMatvecGpuParams {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne10: i64,
    ne12: i64,
    ne0: i64,
    ne1: i64,
    r2: i32,
    r3: i32,
}

fn build_metallib(shader_path: &str, out_dir: &str) -> PathBuf {
    let abs_shader = std::fs::canonicalize(shader_path).expect("shader exists");
    let air_path = format!("{}/iter175_h_e.air", out_dir);
    let metallib_path = format!("{}/iter175_h_e.metallib", out_dir);

    let air_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "-O3", "-c"])
        .arg(&abs_shader)
        .arg("-o")
        .arg(&air_path)
        .status()
        .expect("run xcrun metal");
    assert!(air_status.success(), "xcrun metal -O3 failed");

    let metallib_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&metallib_path)
        .status()
        .expect("run xcrun metallib");
    assert!(metallib_status.success(), "xcrun metallib failed");

    PathBuf::from(metallib_path)
}

fn make_function_constants() -> FunctionConstantValues {
    let fcv = FunctionConstantValues::new();
    // 700:1 (ne12), 701:1 (r2), 702:1 (r3) — production FC-bake at decode.
    for idx in [700u64, 701u64, 702u64] {
        let v: i32 = 1;
        fcv.set_constant_value_at_index(
            (&v as *const i32).cast::<std::ffi::c_void>(),
            MTLDataType::Int,
            idx,
        );
    }
    fcv
}

fn build_pipeline_from_source(
    device: &metal::DeviceRef,
    source: &str,
    label: &str,
) -> metal::ComputePipelineState {
    let opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(source, &opts)
        .expect("runtime source compile");
    let fcv = make_function_constants();
    let function = lib
        .get_function(KERNEL_NAME, Some(fcv))
        .expect("get_function from runtime lib");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&function));
    desc.set_label(label);
    device
        .new_compute_pipeline_state(&desc)
        .expect("pipeline from runtime")
}

fn build_pipeline_from_metallib(
    device: &metal::DeviceRef,
    metallib_path: &PathBuf,
    label: &str,
) -> metal::ComputePipelineState {
    let lib = device
        .new_library_with_file(metallib_path)
        .expect("load .metallib");
    let fcv = make_function_constants();
    let function = lib
        .get_function(KERNEL_NAME, Some(fcv))
        .expect("get_function from precompiled lib");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&function));
    desc.set_label(label);
    device
        .new_compute_pipeline_state(&desc)
        .expect("pipeline from precompiled")
}

fn alloc_weight_q6_k(device: &MlxDevice, n: u32, k: u32) -> MlxBuffer {
    let blocks_per_row = (k as u64) / QK_K;
    let total_bytes = (n as u64) * blocks_per_row * Q6_K_BLOCK_BYTES;
    device
        .alloc_buffer(total_bytes as usize, DType::U8, vec![total_bytes as usize])
        .expect("alloc weight")
}

fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .expect("alloc f32")
}

fn bench_pipeline(
    label: &str,
    pipeline: &metal::ComputePipelineStateRef,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlMatvecGpuParams,
) -> f64 {
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            (params as *const GgmlMatvecGpuParams).cast::<u8>(),
            std::mem::size_of::<GgmlMatvecGpuParams>(),
        )
    };
    let queue = device.metal_queue();

    // Threadgroup geometry per dispatch_mv() for q6_K_nr2: nth0=2, nth1=32, align=4.
    let n = params.ne01 as u64;
    let m = params.ne1 as u64;
    let align: u64 = 4;
    let threadgroups = MTLSize::new((n + align - 1) / align, m, 1);
    let threads_per_tg = MTLSize::new(2, 32, 1);

    // Warmup
    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(weight.metal_buffer()), weight.byte_offset());
        enc.set_buffer(1, Some(input.metal_buffer()), input.byte_offset());
        enc.set_buffer(2, Some(output.metal_buffer()), output.byte_offset());
        enc.set_bytes(3, params_bytes.len() as u64, params_bytes.as_ptr().cast());
        enc.dispatch_thread_groups(threadgroups, threads_per_tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let t0 = Instant::now();
        for _ in 0..BATCH {
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(weight.metal_buffer()), weight.byte_offset());
            enc.set_buffer(1, Some(input.metal_buffer()), input.byte_offset());
            enc.set_buffer(2, Some(output.metal_buffer()), output.byte_offset());
            enc.set_bytes(3, params_bytes.len() as u64, params_bytes.as_ptr().cast());
            enc.dispatch_thread_groups(threadgroups, threads_per_tg);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
        samples.push(elapsed_us);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let p10 = samples[samples.len() / 10];
    let p90 = samples[samples.len() * 9 / 10];
    eprintln!(
        "  {:<24} median={:>7.2}us  p10={:>7.2}  p90={:>7.2}  (n={})",
        label, median, p10, p90, MEASURE
    );
    median
}

#[test]
fn h_e_metallib_vs_runtime_q6_k_nr2() {
    let device = MlxDevice::new().expect("MlxDevice::new");

    let shader_source = std::fs::read_to_string(SHADER_PATH).expect("read shader source");
    let out_dir = std::env::temp_dir().to_string_lossy().to_string();
    let metallib_path = build_metallib(SHADER_PATH, &out_dir);
    eprintln!("[H-E] built .metallib: {}", metallib_path.display());

    let pipeline_runtime = build_pipeline_from_source(
        device.metal_device(),
        &shader_source,
        "runtime-source",
    );
    let pipeline_precompiled = build_pipeline_from_metallib(
        device.metal_device(),
        &metallib_path,
        "precompiled-O3",
    );

    let input = alloc_f32(&device, (Q_M * Q_K) as usize);
    let output = alloc_f32(&device, (Q_M * Q_N) as usize);
    let weight = alloc_weight_q6_k(&device, Q_N, Q_K);

    let params = GgmlMatvecGpuParams {
        ne00: Q_K as i64,
        ne01: Q_N as i64,
        ne02: 1,
        ne10: Q_K as i64,
        ne12: 1,
        ne0: Q_N as i64,
        ne1: Q_M as i64,
        r2: 1,
        r3: 1,
    };

    eprintln!("\n[H-E] kernel: {}", KERNEL_NAME);
    eprintln!("[H-E] shape: m={} n={} k={}", Q_M, Q_N, Q_K);
    eprintln!(
        "[H-E] batched per-call timing (BATCH={}, MEASURE={}):",
        BATCH, MEASURE
    );

    let median_runtime = bench_pipeline(
        "RUNTIME-SOURCE",
        &pipeline_runtime,
        &device,
        &weight,
        &input,
        &output,
        &params,
    );
    let median_precompiled = bench_pipeline(
        "PRECOMPILED -O3",
        &pipeline_precompiled,
        &device,
        &weight,
        &input,
        &output,
        &params,
    );

    let delta_pct = 100.0 * (median_runtime - median_precompiled) / median_runtime;
    eprintln!(
        "\n[H-E] delta: precompiled is {:+.2}% vs runtime-source",
        delta_pct
    );
    eprintln!(
        "[H-E] verdict: {}",
        if delta_pct.abs() < 1.0 {
            "FALSIFIED (within noise)"
        } else if delta_pct > 0.0 {
            "CONFIRMED — precompiled is faster"
        } else {
            "INVERTED — runtime-source is faster (unexpected)"
        }
    );
}
