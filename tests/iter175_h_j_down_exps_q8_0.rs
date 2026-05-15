//! ADR-029 iter-175 Step 1ag — H-J: peer vs hf2q `_id_q8_0` kernel at
//! CORRECT gemma4 down_exps shape (K=704, N=2816, top_k=8).
//!
//! Step 1af corrected the shape error: gemma4 down_exps is Q8_0, not Q6_K,
//! and K=704, not 8192.  This is the actual #1 FFN kernel per Step 1y.
//!
//! Both kernels compiled with `xcrun metal -O3` precompiled metallib
//! (matches Step 1m production).  Apples-to-apples kernel-quality comparison.
//!
//! Run: `cargo test --release --test iter175_h_j_down_exps_q8_0 -- --nocapture`

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use metal::{ComputePipelineDescriptor, FunctionConstantValues, MTLDataType, MTLSize};
use mlx_native::{DType, MlxBuffer, MlxDevice};

// gemma4 ffn_down_exps shape
const N: u32 = 2816;
const K: u32 = 704;
const N_EXPERTS: u32 = 128;
const TOP_K: u32 = 8;
const N_TOKENS: u32 = 1;

const QK8_0: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;  // 32 int8 + 2 byte fp16 scale

const WARMUP: usize = 20;
const MEASURE: usize = 80;
const BATCH: usize = 32;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
struct PeerMulMvIdKargs {
    nei0: i32, nei1: i32, nbi1: u64,
    ne00: i32, ne01: i32, ne02: i32,
    nb00: u64, nb01: u64, nb02: u64,
    ne10: i32, ne11: i32, ne12: i32, ne13: i32,
    nb10: u64, nb11: u64, nb12: u64,
    ne0: i32, ne1: i32, nb1: u64,
    nr0: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Hf2qMatvecIdParams {
    ne00: i64, ne01: i64, ne02: i64, ne10: i64, ne12: i64,
    ne0: i64, ne1: i64, r2: u32, r3: u32,
    top_k: u32, n_tokens: u32, expert_stride: i64,
}

fn build_metallib(src: &str, includes: &[&str], suffix: &str, out_dir: &str) -> PathBuf {
    let abs = std::fs::canonicalize(src).expect("source");
    let air = format!("{}/{}.air", out_dir, suffix);
    let metallib = format!("{}/{}.metallib", out_dir, suffix);

    let mut args: Vec<String> = vec!["-sdk".into(), "macosx".into(),
        "metal".into(), "-O3".into()];
    for inc in includes {
        args.push("-I".into());
        args.push(inc.to_string());
    }
    args.push("-c".into());

    let air_status = Command::new("xcrun")
        .args(&args)
        .arg(&abs).arg("-o").arg(&air)
        .status().expect("xcrun metal");
    assert!(air_status.success(), "metal compile failed for {}", src);

    let lib_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib"])
        .arg(&air).arg("-o").arg(&metallib)
        .status().expect("xcrun metallib");
    assert!(lib_status.success(), "metallib failed for {}", src);
    PathBuf::from(metallib)
}

fn peer_fcs() -> FunctionConstantValues {
    let fcv = FunctionConstantValues::new();
    // peer FC_MUL_MV = 600; nsg=4 for Q8_0 (N_SG_Q8_0=4), r2=1, r3=1, ne12=1
    let vals: &[(u64, i16)] = &[
        (600, 4),  // FC_mul_mv_nsg
        (601, 1),  // FC_mul_mv_nxpsg
        (602, 1),  // FC_mul_mv_ne12
        (603, 1),  // FC_mul_mv_r2
        (604, 1),  // FC_mul_mv_r3
    ];
    for (slot, v) in vals {
        let val = *v;
        fcv.set_constant_value_at_index(
            (&val as *const i16).cast::<std::ffi::c_void>(),
            MTLDataType::Short, *slot);
    }
    fcv
}

fn build_pipeline(
    device: &metal::DeviceRef,
    lib: &PathBuf,
    kernel: &str,
    label: &str,
    fcs: Option<FunctionConstantValues>,
) -> metal::ComputePipelineState {
    let l = device.new_library_with_file(lib).expect("load lib");
    let f = match fcs {
        Some(fc) => l.get_function(kernel, Some(fc)).expect("get_function"),
        None => l.get_function(kernel, None).expect("get_function"),
    };
    let d = ComputePipelineDescriptor::new();
    d.set_compute_function(Some(&f));
    d.set_label(label);
    device.new_compute_pipeline_state(&d).expect("pipeline")
}

fn alloc_weights(device: &MlxDevice) -> MlxBuffer {
    let blocks_per_row = K as u64 / QK8_0;
    let per_expert = (N as u64) * blocks_per_row * Q8_0_BLOCK_BYTES;
    let total = (N_EXPERTS as u64) * per_expert;
    device.alloc_buffer(total as usize, DType::U8, vec![total as usize]).expect("alloc weights")
}

fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n*4, DType::F32, vec![n]).expect("alloc f32")
}

fn alloc_u32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n*4, DType::U32, vec![n]).expect("alloc u32")
}

fn bench(
    label: &str,
    pipeline: &metal::ComputePipelineStateRef,
    device: &MlxDevice,
    buffers: &[(u64, &MlxBuffer)],
    params_bytes: &[u8],
    params_slot: u64,
    threadgroups: MTLSize,
    threads: MTLSize,
    smem: u64,
) -> f64 {
    let queue = device.metal_queue();
    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (s, b) in buffers { enc.set_buffer(*s, Some(b.metal_buffer()), b.byte_offset()); }
        enc.set_bytes(params_slot, params_bytes.len() as u64, params_bytes.as_ptr().cast());
        if smem > 0 { enc.set_threadgroup_memory_length(0, smem); }
        enc.dispatch_thread_groups(threadgroups, threads);
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
            for (s, b) in buffers { enc.set_buffer(*s, Some(b.metal_buffer()), b.byte_offset()); }
            enc.set_bytes(params_slot, params_bytes.len() as u64, params_bytes.as_ptr().cast());
            if smem > 0 { enc.set_threadgroup_memory_length(0, smem); }
            enc.dispatch_thread_groups(threadgroups, threads);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        samples.push(t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len()/2];
    let p10 = samples[samples.len()/10];
    eprintln!("  {:<28} median={:>8.3}us  p10={:>7.3}", label, median, p10);
    median
}

#[test]
fn h_j_down_exps_q8_0() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let out_dir = std::env::temp_dir().to_string_lossy().to_string();

    let peer_lib = build_metallib(
        "/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal",
        &["/opt/llama.cpp/ggml/src/ggml-metal", "/opt/llama.cpp/ggml/src", "/opt/llama.cpp/ggml/include"],
        "peer_q8_0", &out_dir);
    let hf2q_lib = build_metallib(
        "src/shaders/quantized_matmul_id_ggml.metal", &[],
        "hf2q_q8_0", &out_dir);

    let peer_pipe = build_pipeline(device.metal_device(), &peer_lib,
        "kernel_mul_mv_id_q8_0_f32", "peer-O3", Some(peer_fcs()));
    let hf2q_pipe = build_pipeline(device.metal_device(), &hf2q_lib,
        "kernel_mul_mv_id_q8_0_f32", "hf2q-O3", None);

    let weights = alloc_weights(&device);
    let input = alloc_f32(&device, (N_TOKENS * K) as usize);
    let dst = alloc_f32(&device, (N_TOKENS * TOP_K * N) as usize);
    let ids = alloc_u32(&device, (N_TOKENS * TOP_K) as usize);

    let blocks_per_row = K as u64 / QK8_0;
    let per_expert = (N as u64) * blocks_per_row * Q8_0_BLOCK_BYTES;

    let hf2q_params = Hf2qMatvecIdParams {
        ne00: K as i64, ne01: N as i64, ne02: 1,
        ne10: K as i64, ne12: 1,
        ne0: N as i64, ne1: (N_TOKENS * TOP_K) as i64,
        r2: 1, r3: 1, top_k: TOP_K, n_tokens: N_TOKENS,
        expert_stride: per_expert as i64,
    };
    let hf2q_bytes = unsafe { std::slice::from_raw_parts(
        (&hf2q_params as *const Hf2qMatvecIdParams).cast::<u8>(),
        std::mem::size_of::<Hf2qMatvecIdParams>()) };

    let peer_params = PeerMulMvIdKargs {
        nei0: TOP_K as i32, nei1: N_TOKENS as i32, nbi1: (TOP_K as u64)*4,
        ne00: K as i32, ne01: N as i32, ne02: 1,
        nb00: Q8_0_BLOCK_BYTES / QK8_0,
        nb01: blocks_per_row * Q8_0_BLOCK_BYTES,
        nb02: per_expert,
        ne10: K as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: (K as u64)*4, nb12: (K as u64)*4,
        ne0: N as i32, ne1: 1, nb1: (N as u64)*4,
        nr0: 2,  // N_R0_Q8_0
    };
    let peer_bytes = unsafe { std::slice::from_raw_parts(
        (&peer_params as *const PeerMulMvIdKargs).cast::<u8>(),
        std::mem::size_of::<PeerMulMvIdKargs>()) };

    // hf2q dispatch: align=N_DST*N_SIMDGROUP=8, (ceil(N/8), n_tokens*top_k, 1) = (352, 8, 1)
    let hf2q_align: u64 = 8;
    let hf2q_tg = MTLSize::new((N as u64 + hf2q_align - 1)/hf2q_align, (N_TOKENS*TOP_K) as u64, 1);
    let hf2q_threads = MTLSize::new(8, 8, 1);  // 64 threads = 2 SGs × 32

    // peer dispatch (Q8_0 is in the F32/F16/BF16/Q8_0 branch using nr0 not nr0*nsg):
    //   tg = (ceil(N/nr0), 1, n_tokens*top_k) with threads (32, nsg=4, 1)
    let peer_tg = MTLSize::new((N as u64 + 2 - 1)/2, 1, (N_TOKENS*TOP_K) as u64);
    let peer_threads = MTLSize::new(32, 4, 1);

    eprintln!("\n[H-J] gemma4 ffn_down_exps shape: N={} K={} top_k={} (Q8_0)", N, K, TOP_K);
    eprintln!("[H-J] hf2q: tg={:?}, threads={:?}", (hf2q_tg.width, hf2q_tg.height, hf2q_tg.depth), (hf2q_threads.width, hf2q_threads.height, hf2q_threads.depth));
    eprintln!("[H-J] peer: tg={:?}, threads={:?}", (peer_tg.width, peer_tg.height, peer_tg.depth), (peer_threads.width, peer_threads.height, peer_threads.depth));

    let mut hf2q_meds = Vec::new();
    let mut peer_meds = Vec::new();
    for cycle in 0..3 {
        eprintln!("\n--- cycle {} ---", cycle);
        if cycle % 2 == 0 {
            let mh = bench("hf2q _id_q8_0", &hf2q_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (3, &ids)],
                hf2q_bytes, 4, hf2q_tg, hf2q_threads, 0);
            std::thread::sleep(std::time::Duration::from_secs(3));
            let mp = bench("peer _id_q8_0", &peer_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (4, &ids)],
                peer_bytes, 0, peer_tg, peer_threads, 8192);
            hf2q_meds.push(mh); peer_meds.push(mp);
        } else {
            let mp = bench("peer _id_q8_0", &peer_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (4, &ids)],
                peer_bytes, 0, peer_tg, peer_threads, 8192);
            std::thread::sleep(std::time::Duration::from_secs(3));
            let mh = bench("hf2q _id_q8_0", &hf2q_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (3, &ids)],
                hf2q_bytes, 4, hf2q_tg, hf2q_threads, 0);
            hf2q_meds.push(mh); peer_meds.push(mp);
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    let mh = hf2q_meds.iter().sum::<f64>() / hf2q_meds.len() as f64;
    let mp = peer_meds.iter().sum::<f64>() / peer_meds.len() as f64;
    let delta = 100.0 * (mp - mh) / mh;
    eprintln!("\n[H-J] aggregate (3 alt-paired cycles):");
    eprintln!("  hf2q : mean {:.3}us  samples {:?}", mh, hf2q_meds);
    eprintln!("  peer : mean {:.3}us  samples {:?}", mp, peer_meds);
    eprintln!("  delta: peer {:+.2}% vs hf2q", delta);
    let v = if delta.abs() < 2.0 { "TIED (±2%)" }
            else if delta < 0.0 { "PEER FASTER — gemma4 down_exps lever exists" }
            else { "HF2Q FASTER — kernel quality not the bottleneck here" };
    eprintln!("  verdict: {}", v);
}
