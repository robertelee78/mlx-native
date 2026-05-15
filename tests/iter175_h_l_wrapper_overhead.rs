//! ADR-029 iter-175 Step 1ak — H-L: bisect within hf2q's CommandEncoder
//! wrapper to find the 520 ns/dispatch overhead vs raw Metal FFI.
//!
//! Step 1aj microbenched raw metal-rs path = 162 ns/dispatch.
//! Step 1ah measured production = 0.68 µs/dispatch.
//! Delta = 520 ns/dispatch in hf2q's encoder.rs wrapper.
//!
//! H-L A: bench through `CommandEncoder::encode_threadgroups_with_args`
//!        (production wrapper, no tracker layer).
//! H-L B: bench through `CommandEncoder::dispatch_tracked_threadgroups_with_args`
//!        (production hot path used by Step 1e q6_K_nr2 site).
//!
//! Each arm uses the same trivial noop kernel.  Both call paths go through
//! the wrapper; B adds the dispatch_tracked_* layer on top of A.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use metal::{ComputePipelineDescriptor, MTLSize};
use mlx_native::{DType, MlxDevice, KernelArg};

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
const N_DISPATCH_PER_CB: usize = 1000;

fn build_pipeline(device: &metal::DeviceRef) -> metal::ComputePipelineState {
    let out_dir = std::env::temp_dir().to_string_lossy().to_string();
    let src_path = format!("{}/noop_h_l.metal", out_dir);
    std::fs::write(&src_path, TRIVIAL_KERNEL_SRC).expect("write metal");
    let air = format!("{}/noop_h_l.air", out_dir);
    let metallib = format!("{}/noop_h_l.metallib", out_dir);
    Command::new("xcrun").args(&["-sdk", "macosx", "metal", "-O3", "-c", &src_path, "-o", &air])
        .status().expect("xcrun metal");
    Command::new("xcrun").args(&["-sdk", "macosx", "metallib", &air, "-o", &metallib])
        .status().expect("xcrun metallib");
    let lib = device.new_library_with_file(&PathBuf::from(metallib)).expect("load lib");
    let f = lib.get_function("noop_dispatch", None).expect("get_function");
    let d = ComputePipelineDescriptor::new();
    d.set_compute_function(Some(&f));
    d.set_label("noop-h-l");
    device.new_compute_pipeline_state(&d).expect("pipeline")
}

#[test]
fn h_l_wrapper_overhead() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let pipe = build_pipeline(device.metal_device());

    let x = device.alloc_buffer(64, DType::U32, vec![16]).expect("x");
    let y = device.alloc_buffer(64, DType::U32, vec![16]).expect("y");
    let z = device.alloc_buffer(64, DType::U32, vec![16]).expect("z");

    #[repr(C)] struct P { a: u32 }
    let p = P { a: 1 };
    let p_bytes_owned: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            (&p as *const P).cast::<u8>(),
            std::mem::size_of::<P>(),
        ).to_vec()
    };

    let tg = MTLSize::new(1, 1, 1);
    let threads = MTLSize::new(1, 1, 1);

    // -------- ARM A: raw metal-rs (re-measure baseline) --------
    eprintln!("\n[H-L] N={} dispatches/CB, MEASURE={}", N_DISPATCH_PER_CB, MEASURE);
    let queue = device.metal_queue();
    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..N_DISPATCH_PER_CB {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(x.metal_buffer()), x.byte_offset());
            enc.set_buffer(1, Some(y.metal_buffer()), y.byte_offset());
            enc.set_buffer(2, Some(z.metal_buffer()), z.byte_offset());
            enc.set_bytes(3, p_bytes_owned.len() as u64, p_bytes_owned.as_ptr().cast());
            enc.dispatch_thread_groups(tg, threads);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut raw_samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let t0 = Instant::now();
        for _ in 0..N_DISPATCH_PER_CB {
            enc.set_compute_pipeline_state(&pipe);
            enc.set_buffer(0, Some(x.metal_buffer()), x.byte_offset());
            enc.set_buffer(1, Some(y.metal_buffer()), y.byte_offset());
            enc.set_buffer(2, Some(z.metal_buffer()), z.byte_offset());
            enc.set_bytes(3, p_bytes_owned.len() as u64, p_bytes_owned.as_ptr().cast());
            enc.dispatch_thread_groups(tg, threads);
        }
        raw_samples.push(t0.elapsed().as_secs_f64() * 1e6);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    raw_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let raw_median = raw_samples[MEASURE/2];
    let raw_per_dispatch_ns = raw_median * 1000.0 / N_DISPATCH_PER_CB as f64;
    eprintln!("  ARM A raw FFI            : {:.1} us/CB = {:.1} ns/dispatch", raw_median, raw_per_dispatch_ns);

    // -------- ARM B: hf2q CommandEncoder.encode_threadgroups_with_args --------
    // Warmup
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("command_encoder");
        for _ in 0..N_DISPATCH_PER_CB {
            let bindings = [
                (0u64, KernelArg::Buffer(&x)),
                (1u64, KernelArg::Buffer(&y)),
                (2u64, KernelArg::Buffer(&z)),
                (3u64, KernelArg::Bytes(&p_bytes_owned)),
            ];
            enc.encode_threadgroups_with_args(&pipe, &bindings, tg, threads);
        }
        enc.commit_and_wait();
    }
    let mut wrap_samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("command_encoder");
        let t0 = Instant::now();
        for _ in 0..N_DISPATCH_PER_CB {
            let bindings = [
                (0u64, KernelArg::Buffer(&x)),
                (1u64, KernelArg::Buffer(&y)),
                (2u64, KernelArg::Buffer(&z)),
                (3u64, KernelArg::Bytes(&p_bytes_owned)),
            ];
            enc.encode_threadgroups_with_args(&pipe, &bindings, tg, threads);
        }
        wrap_samples.push(t0.elapsed().as_secs_f64() * 1e6);
        enc.commit_and_wait();
    }
    wrap_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let wrap_median = wrap_samples[MEASURE/2];
    let wrap_per_dispatch_ns = wrap_median * 1000.0 / N_DISPATCH_PER_CB as f64;
    eprintln!("  ARM B CommandEncoder    : {:.1} us/CB = {:.1} ns/dispatch", wrap_median, wrap_per_dispatch_ns);

    // -------- ARM C: hf2q dispatch_tracked_threadgroups_with_args (Step 1e production hot path) --------
    let reads = [&x, &y];
    let writes = [&z];
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("command_encoder");
        for _ in 0..N_DISPATCH_PER_CB {
            let bindings = [
                (0u64, KernelArg::Buffer(&x)),
                (1u64, KernelArg::Buffer(&y)),
                (2u64, KernelArg::Buffer(&z)),
                (3u64, KernelArg::Bytes(&p_bytes_owned)),
            ];
            enc.dispatch_tracked_threadgroups_with_args(&pipe, &bindings, &reads, &writes, tg, threads);
        }
        enc.commit_and_wait();
    }
    let mut tracked_samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("command_encoder");
        let t0 = Instant::now();
        for _ in 0..N_DISPATCH_PER_CB {
            let bindings = [
                (0u64, KernelArg::Buffer(&x)),
                (1u64, KernelArg::Buffer(&y)),
                (2u64, KernelArg::Buffer(&z)),
                (3u64, KernelArg::Bytes(&p_bytes_owned)),
            ];
            enc.dispatch_tracked_threadgroups_with_args(&pipe, &bindings, &reads, &writes, tg, threads);
        }
        tracked_samples.push(t0.elapsed().as_secs_f64() * 1e6);
        enc.commit_and_wait();
    }
    tracked_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tracked_median = tracked_samples[MEASURE/2];
    let tracked_per_dispatch_ns = tracked_median * 1000.0 / N_DISPATCH_PER_CB as f64;
    eprintln!("  ARM C dispatch_tracked  : {:.1} us/CB = {:.1} ns/dispatch", tracked_median, tracked_per_dispatch_ns);

    eprintln!("\n[H-L] Per-dispatch breakdown (medians):");
    eprintln!("  Raw FFI                : {:.1} ns", raw_per_dispatch_ns);
    eprintln!("  + CommandEncoder wrap   : {:.1} ns  (Δ +{:.1} ns)", wrap_per_dispatch_ns, wrap_per_dispatch_ns - raw_per_dispatch_ns);
    eprintln!("  + dispatch_tracked      : {:.1} ns  (Δ +{:.1} ns)", tracked_per_dispatch_ns, tracked_per_dispatch_ns - wrap_per_dispatch_ns);
    eprintln!("\n[H-L] Step 1ah production : ~680 ns/dispatch");
    eprintln!("  This bench's `dispatch_tracked` is the SAME path used in production for q6_K_nr2 (Step 1e).");
    eprintln!("  If tracked ≈ 680 ns → confirms production attribution.");
    eprintln!("  If tracked < 680 ns → some overhead is in forward_mlx.rs orchestration, not encoder.rs.");
}
