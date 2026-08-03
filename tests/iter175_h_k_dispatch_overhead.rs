//! ADR-029 iter-175 Step 1aj — H-K: empirical measurement of per-dispatch
//! CPU overhead by isolating the dispatch path from kernel execution.
//!
//! Validates Step 1ai's 690 ns/dispatch estimate.  Strategy:
//! 1. Compile a trivial Metal kernel (single thread, immediate return).
//! 2. Dispatch it N times in a tight loop within one command-buffer.
//! 3. Time the loop; divide by N to get per-dispatch CPU+GPU cost.
//! 4. Compare against a near-empty pipeline at minimum dispatch size.
//!
//! Caveat: GPU still runs the trivial kernel, so the measurement includes
//! per-dispatch GPU scheduler cost (not just CPU encode).  But for a 1-thread
//! dispatch, GPU cost should be tiny (~5-10 ns) vs CPU ~690 ns.  The
//! measurement gives a tight upper bound on CPU encode cost.
//!
//! Run: `cargo test --release --test iter175_h_k_dispatch_overhead -- --nocapture`

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use metal::{ComputePipelineDescriptor, MTLSize};
use mlx_native::{DType, MlxDevice};

const TRIVIAL_KERNEL_SRC: &str = "
#include <metal_stdlib>
using namespace metal;

struct P { uint a; };

kernel void noop_dispatch(
    device const uint* x [[buffer(0)]],
    device const uint* y [[buffer(1)]],
    device uint*       z [[buffer(2)]],
    constant P&        p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.a) return;
    z[gid] = x[gid] + y[gid];
}
";

const WARMUP: usize = 10;
const MEASURE: usize = 50;
const N_DISPATCH_PER_CB: usize = 1000;  // many dispatches per command buffer

fn build_noop_metallib(out_dir: &str) -> PathBuf {
    let src_path = format!("{}/noop.metal", out_dir);
    std::fs::write(&src_path, TRIVIAL_KERNEL_SRC).expect("write noop.metal");
    let air = format!("{}/noop.air", out_dir);
    let metallib = format!("{}/noop.metallib", out_dir);

    let air_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "-O3", "-c", &src_path,
                "-o", &air])
        .status().expect("xcrun metal");
    assert!(air_status.success());

    let lib_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib", &air, "-o", &metallib])
        .status().expect("xcrun metallib");
    assert!(lib_status.success());
    PathBuf::from(metallib)
}

fn build_pipeline(device: &metal::DeviceRef, lib: &PathBuf) -> metal::ComputePipelineState {
    let l = device.new_library_with_file(lib).expect("load lib");
    let f = l.get_function("noop_dispatch", None).expect("get_function");
    let d = ComputePipelineDescriptor::new();
    d.set_compute_function(Some(&f));
    d.set_label("noop-dispatch");
    device.new_compute_pipeline_state(&d).expect("pipeline")
}

#[test]
fn h_k_dispatch_overhead() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let out_dir = std::env::temp_dir().to_string_lossy().to_string();

    let lib = build_noop_metallib(&out_dir);
    let pipe = build_pipeline(device.metal_device(), &lib);

    // Smallest possible buffers.
    let x = device.alloc_buffer(64, DType::U32, vec![16]).expect("alloc x");
    let y = device.alloc_buffer(64, DType::U32, vec![16]).expect("alloc y");
    let z = device.alloc_buffer(64, DType::U32, vec![16]).expect("alloc z");

    // 1-byte params.
    #[repr(C)]
    struct P { a: u32 }
    let p = P { a: 1 };
    let p_bytes = unsafe {
        std::slice::from_raw_parts(
            (&p as *const P).cast::<u8>(),
            std::mem::size_of::<P>(),
        )
    };

    // Minimal dispatch: 1 threadgroup × 1 thread.
    let tg = MTLSize::new(1, 1, 1);
    let threads = MTLSize::new(1, 1, 1);

    let queue = device.metal_queue();

    // Warmup
    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..N_DISPATCH_PER_CB {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(x.metal_buffer()), x.byte_offset());
            enc.set_buffer(1, Some(y.metal_buffer()), y.byte_offset());
            enc.set_buffer(2, Some(z.metal_buffer()), z.byte_offset());
            enc.set_bytes(3, p_bytes.len() as u64, p_bytes.as_ptr().cast());
            enc.dispatch_thread_groups(tg, threads);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Measure CPU-only encode (without waiting for GPU)
    let mut cpu_only_samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let t0 = Instant::now();
        for _ in 0..N_DISPATCH_PER_CB {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(x.metal_buffer()), x.byte_offset());
            enc.set_buffer(1, Some(y.metal_buffer()), y.byte_offset());
            enc.set_buffer(2, Some(z.metal_buffer()), z.byte_offset());
            enc.set_bytes(3, p_bytes.len() as u64, p_bytes.as_ptr().cast());
            enc.dispatch_thread_groups(tg, threads);
        }
        let cpu_us = t0.elapsed().as_secs_f64() * 1e6;
        cpu_only_samples.push(cpu_us);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    cpu_only_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_cpu = cpu_only_samples[cpu_only_samples.len() / 2];
    let p10_cpu = cpu_only_samples[cpu_only_samples.len() / 10];
    let p90_cpu = cpu_only_samples[cpu_only_samples.len() * 9 / 10];

    eprintln!("\n[H-K] Dispatch overhead microbench (production-equivalent path)");
    eprintln!("[H-K] N={} dispatches per CB, MEASURE={}", N_DISPATCH_PER_CB, MEASURE);
    eprintln!("[H-K] Per-CB CPU encode time (excludes GPU wait):");
    eprintln!("  median={:.1}us  p10={:.1}  p90={:.1}", median_cpu, p10_cpu, p90_cpu);
    let per_dispatch_ns = median_cpu * 1000.0 / N_DISPATCH_PER_CB as f64;
    eprintln!("  per-dispatch CPU: {:.1} ns", per_dispatch_ns);
    eprintln!("  step 1ai estimate: ~690 ns/dispatch (with 4 ffi calls)");
    let delta = 100.0 * (per_dispatch_ns - 690.0) / 690.0;
    eprintln!("  delta vs Step 1ai: {:+.1}% (likely lower b/c no tracker overhead)", delta);
}
