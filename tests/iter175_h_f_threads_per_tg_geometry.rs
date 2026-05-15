//! ADR-029 iter-175 Step 1w — H-F empirical test:
//! does `(32, NSG, 1)` (peer geometry) vs `(2, 32, 1)` (hf2q current)
//! threads_per_threadgroup change `kernel_mul_mv_q6_K_f32_nr2` runtime?
//!
//! Background (from Step 1 + iter-175 reasoning):
//!   - Peer dispatches with `MTLSize(32, nsg, 1)` (ggml-metal-ops.cpp:2249).
//!   - hf2q dispatches with `MTLSize(2, 32, 1)` (quantized_matmul_ggml.rs:680
//!     when use_q6k_nr2=true, nth0=2, nth1=32).
//!   - The kernel uses only `tiisg` (thread_index_in_simdgroup) and `sgitg`
//!     (simdgroup_index_in_threadgroup), which are derived from the threadgroup
//!     LINEAR position — both layouts produce 2 simdgroups of 32 threads.
//!   - Hypothesis: functionally identical, but the Metal compiler/scheduler
//!     may emit different memory-access patterns depending on which axis the
//!     simdgroup boundary lands on.  Worth measuring.
//!
//! Bench plan: same shape and kernel as iter175_h_e (gemma4 Q_sliding,
//! m=1, n=4096, k=2816, q6_K_nr2 with FC slots 700/701/702=1).
//! Two arms: dispatch at `(2,32,1)` vs `(32,2,1)`.  Each arm:
//!   - precompiled .metallib (-O3) only (matches production after Step 1m default-flip)
//!   - BATCH=32 per CB, MEASURE=50 CBs, WARMUP=10.
//!
//! Decision criteria:
//!   - delta > 1% favoring (32,2,1): testable lever — open follow-up to flip
//!     all q6_K/q5_K/q4_K mv dispatches to peer geometry, re-bench at wall.
//!   - delta within ±1%: H-F FALSIFIED — geometry doesn't matter at this kernel.
//!
//! Run via: `cargo test --release --test iter175_h_f_threads_per_tg_geometry -- --nocapture`

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use metal::{ComputePipelineDescriptor, FunctionConstantValues, MTLDataType, MTLSize};
use mlx_native::{DType, MlxBuffer, MlxDevice};

const KERNEL_NAME: &str = "kernel_mul_mv_q6_K_f32_nr2";
const SHADER_PATH: &str = "src/shaders/quantized_matmul_ggml.metal";

// gemma4 Q_sliding decode shape: m=1 (decode), n=4096, k=2816.
const Q_M: u32 = 1;
const Q_N: u32 = 4096;
const Q_K: u32 = 2816;

const WARMUP: usize = 20;   // larger warmup to stabilize PSO cache
const MEASURE: usize = 80;  // larger sample for tighter CI
const BATCH: usize = 32;

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
    let air_path = format!("{}/iter175_h_f.air", out_dir);
    let metallib_path = format!("{}/iter175_h_f.metallib", out_dir);

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
    threads_per_tg: MTLSize,
) -> (f64, f64, f64) {
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            (params as *const GgmlMatvecGpuParams).cast::<u8>(),
            std::mem::size_of::<GgmlMatvecGpuParams>(),
        )
    };
    let queue = device.metal_queue();

    let n = params.ne01 as u64;
    let m = params.ne1 as u64;
    // Both arms cover the same N rows: ceil(N / (nr0*nsg)) = ceil(4096/4) = 1024 TGs.
    let align: u64 = 4;
    let threadgroups = MTLSize::new((n + align - 1) / align, m, 1);

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
        "  {:<32} median={:>7.2}us  p10={:>7.2}  p90={:>7.2}  (n={})",
        label, median, p10, p90, MEASURE
    );
    (median, p10, p90)
}

#[test]
fn h_f_threads_per_tg_geometry_q6_k_nr2() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let out_dir = std::env::temp_dir().to_string_lossy().to_string();
    let metallib_path = build_metallib(SHADER_PATH, &out_dir);
    eprintln!("[H-F] built .metallib: {}", metallib_path.display());

    let pipeline = build_pipeline_from_metallib(
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

    eprintln!("\n[H-F] kernel: {} (precompiled)", KERNEL_NAME);
    eprintln!("[H-F] shape: m={} n={} k={}", Q_M, Q_N, Q_K);
    eprintln!(
        "[H-F] threadgroups=(1024,1,1), BATCH={}, WARMUP={}, MEASURE={}",
        BATCH, WARMUP, MEASURE
    );

    // Run multiple alternating cycles to control for thermal drift.
    let mut hf2q_medians = Vec::new();
    let mut peer_medians = Vec::new();
    for cycle in 0..3 {
        eprintln!("\n--- cycle {} ---", cycle);
        // alternate order each cycle to remove residual order bias
        if cycle % 2 == 0 {
            let (h, _, _) = bench_pipeline(
                "hf2q (2, 32, 1)",
                &pipeline, &device, &weight, &input, &output, &params,
                MTLSize::new(2, 32, 1),
            );
            std::thread::sleep(std::time::Duration::from_secs(3));
            let (p, _, _) = bench_pipeline(
                "peer (32, 2, 1)",
                &pipeline, &device, &weight, &input, &output, &params,
                MTLSize::new(32, 2, 1),
            );
            hf2q_medians.push(h);
            peer_medians.push(p);
        } else {
            let (p, _, _) = bench_pipeline(
                "peer (32, 2, 1)",
                &pipeline, &device, &weight, &input, &output, &params,
                MTLSize::new(32, 2, 1),
            );
            std::thread::sleep(std::time::Duration::from_secs(3));
            let (h, _, _) = bench_pipeline(
                "hf2q (2, 32, 1)",
                &pipeline, &device, &weight, &input, &output, &params,
                MTLSize::new(2, 32, 1),
            );
            hf2q_medians.push(h);
            peer_medians.push(p);
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    let hf2q_mean = hf2q_medians.iter().sum::<f64>() / hf2q_medians.len() as f64;
    let peer_mean = peer_medians.iter().sum::<f64>() / peer_medians.len() as f64;
    let delta_pct = 100.0 * (hf2q_mean - peer_mean) / hf2q_mean;
    eprintln!("\n[H-F] aggregate (3 cycles, alt-paired):");
    eprintln!("  hf2q (2,32,1)  mean: {:.2}us  samples: {:?}", hf2q_mean, hf2q_medians);
    eprintln!("  peer (32,2,1)  mean: {:.2}us  samples: {:?}", peer_mean, peer_medians);
    eprintln!(
        "  delta: (32,2,1) is {:+.2}% vs (2,32,1)",
        delta_pct
    );
    eprintln!(
        "  verdict: {}",
        if delta_pct.abs() < 1.0 {
            "FALSIFIED (within noise — geometry doesn't matter)"
        } else if delta_pct > 0.0 {
            "CONFIRMED — peer geometry (32,2,1) is faster"
        } else {
            "INVERTED — hf2q geometry (2,32,1) is faster"
        }
    );
}
