//! ADR-029 iter-175 Step 1ae — H-I: peer vs hf2q _id kernel head-to-head.
//!
//! Loads peer's `kernel_mul_mv_id_q6_K_f32` (precompiled from peer source via
//! xcrun -O3) and hf2q's `kernel_mul_mv_id_q6_K_f32_nr2` (production), benches
//! both at IDENTICAL gemma4 MoE down_exps shape on the same buffers.
//!
//! Question: ignoring compile-flag differences (both -O3 precompiled), is
//! peer's kernel algorithmically faster than hf2q's?  Or is the residual
//! 6.35% decode gap from non-kernel overhead (encode, scheduler, dispatcher)?
//!
//! Run: `cargo test --release --test iter175_h_i_peer_id_kernel_compare -- --nocapture`

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use metal::{ComputePipelineDescriptor, FunctionConstantValues, MTLDataType, MTLSize};
use mlx_native::{DType, MlxBuffer, MlxDevice};

const N: u32 = 2816;
const K: u32 = 8192;
const N_EXPERTS: u32 = 128;
const TOP_K: u32 = 8;
const N_TOKENS: u32 = 1;

const QK_K: u64 = 256;
const Q6_K_BLOCK_BYTES: u64 = 210;

const WARMUP: usize = 20;
const MEASURE: usize = 80;
const BATCH: usize = 32;

// Peer's mul_mv_id kargs (40 bytes packed).
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

// hf2q's GgmlMatvecIdGpuParams.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Hf2qMatvecIdParams {
    ne00: i64, ne01: i64, ne02: i64, ne10: i64, ne12: i64,
    ne0: i64, ne1: i64, r2: u32, r3: u32,
    top_k: u32, n_tokens: u32, expert_stride: i64,
}

fn build_peer_metallib(out_dir: &str) -> PathBuf {
    let peer_src = "/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal";
    let abs_src = std::fs::canonicalize(peer_src).expect("peer source exists");
    let air = format!("{}/peer_ggml.air", out_dir);
    let metallib = format!("{}/peer_ggml.metallib", out_dir);

    let air_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "-O3",
                "-I", "/opt/llama.cpp/ggml/src/ggml-metal",
                "-I", "/opt/llama.cpp/ggml/src",
                "-I", "/opt/llama.cpp/ggml/include",
                "-c"])
        .arg(&abs_src)
        .arg("-o")
        .arg(&air)
        .status()
        .expect("run xcrun metal on peer source");
    assert!(air_status.success(), "peer xcrun metal -O3 failed");

    let lib_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&metallib)
        .status()
        .expect("run xcrun metallib");
    assert!(lib_status.success(), "peer xcrun metallib failed");
    PathBuf::from(metallib)
}

fn build_hf2q_metallib(out_dir: &str) -> PathBuf {
    let src = "src/shaders/quantized_matmul_id_ggml.metal";
    let abs = std::fs::canonicalize(src).expect("hf2q source");
    let air = format!("{}/hf2q_id.air", out_dir);
    let metallib = format!("{}/hf2q_id.metallib", out_dir);

    let air_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "-O3", "-c"])
        .arg(&abs).arg("-o").arg(&air)
        .status().expect("xcrun metal hf2q");
    assert!(air_status.success(), "hf2q xcrun metal failed");

    let lib_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib"])
        .arg(&air).arg("-o").arg(&metallib)
        .status().expect("xcrun metallib hf2q");
    assert!(lib_status.success(), "hf2q xcrun metallib failed");
    PathBuf::from(metallib)
}

fn peer_fcs() -> FunctionConstantValues {
    let fcv = FunctionConstantValues::new();
    // FC_MUL_MV = 600; nsg=2, nxpsg=1, ne12=1, r2=1, r3=1
    let vals: &[(u64, i16)] = &[
        (600, 2),  // FC_mul_mv_nsg
        (601, 1),  // FC_mul_mv_nxpsg
        (602, 1),  // FC_mul_mv_ne12
        (603, 1),  // FC_mul_mv_r2
        (604, 1),  // FC_mul_mv_r3
    ];
    for (slot, v) in vals {
        let val = *v;
        fcv.set_constant_value_at_index(
            (&val as *const i16).cast::<std::ffi::c_void>(),
            MTLDataType::Short,
            *slot,
        );
    }
    fcv
}

fn hf2q_fcs() -> FunctionConstantValues {
    let fcv = FunctionConstantValues::new();
    for slot in [700u64, 701u64, 702u64] {
        let v: i32 = 1;
        fcv.set_constant_value_at_index(
            (&v as *const i32).cast::<std::ffi::c_void>(),
            MTLDataType::Int, slot,
        );
    }
    fcv
}

fn build_pipeline(
    device: &metal::DeviceRef,
    lib_path: &PathBuf,
    kernel_name: &str,
    label: &str,
    fcs: Option<FunctionConstantValues>,
) -> metal::ComputePipelineState {
    let lib = device.new_library_with_file(lib_path).expect("load lib");
    let function = match fcs {
        Some(f) => lib.get_function(kernel_name, Some(f)).expect("get_function"),
        None => lib.get_function(kernel_name, None).expect("get_function (none)"),
    };
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&function));
    desc.set_label(label);
    device.new_compute_pipeline_state(&desc).expect("pipeline")
}

fn alloc_weights(device: &MlxDevice) -> MlxBuffer {
    let blocks_per_row = K as u64 / QK_K;
    let per_expert = (N as u64) * blocks_per_row * Q6_K_BLOCK_BYTES;
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
    smem_bytes: u64,
) -> f64 {
    let queue = device.metal_queue();
    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (s, b) in buffers { enc.set_buffer(*s, Some(b.metal_buffer()), b.byte_offset()); }
        enc.set_bytes(params_slot, params_bytes.len() as u64, params_bytes.as_ptr().cast());
        if smem_bytes > 0 { enc.set_threadgroup_memory_length(0, smem_bytes); }
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
            if smem_bytes > 0 { enc.set_threadgroup_memory_length(0, smem_bytes); }
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
    eprintln!("  {:<32} median={:>8.2}us  p10={:>7.2}", label, median, p10);
    median
}

#[test]
fn h_i_peer_vs_hf2q_id_kernel() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let out_dir = std::env::temp_dir().to_string_lossy().to_string();

    let peer_lib = build_peer_metallib(&out_dir);
    let hf2q_lib = build_hf2q_metallib(&out_dir);

    let peer_pipe = build_pipeline(device.metal_device(), &peer_lib,
        "kernel_mul_mv_id_q6_K_f32", "peer-O3", Some(peer_fcs()));
    let hf2q_pipe = build_pipeline(device.metal_device(), &hf2q_lib,
        "kernel_mul_mv_id_q6_K_f32_nr2", "hf2q-O3", Some(hf2q_fcs()));

    let weights = alloc_weights(&device);
    let input = alloc_f32(&device, (N_TOKENS * K) as usize);
    let dst = alloc_f32(&device, (N_TOKENS * TOP_K * N) as usize);
    let ids = alloc_u32(&device, (N_TOKENS * TOP_K) as usize);

    // hf2q params
    let blocks_per_row = (K as u64) / QK_K;
    let per_expert = (N as u64) * blocks_per_row * Q6_K_BLOCK_BYTES;
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

    // Peer params
    let peer_params = PeerMulMvIdKargs {
        nei0: TOP_K as i32, nei1: N_TOKENS as i32, nbi1: (TOP_K as u64)*4,
        ne00: K as i32, ne01: N as i32, ne02: 1,
        nb00: Q6_K_BLOCK_BYTES / QK_K, // approx; per-element byte stride
        nb01: blocks_per_row * Q6_K_BLOCK_BYTES,
        nb02: per_expert,
        ne10: K as i32, ne11: 1, ne12: 1, ne13: 1,
        nb10: 4, nb11: (K as u64)*4, nb12: (K as u64)*4,
        ne0: N as i32, ne1: 1, nb1: (N as u64)*4,
        nr0: 2,
    };
    let peer_bytes = unsafe { std::slice::from_raw_parts(
        (&peer_params as *const PeerMulMvIdKargs).cast::<u8>(),
        std::mem::size_of::<PeerMulMvIdKargs>()) };

    // hf2q dispatch: (704, 8, 1), threads (2, 32, 1)
    let align: u64 = 4;
    let hf2q_tg = MTLSize::new((N as u64 + align - 1)/align, (N_TOKENS*TOP_K) as u64, 1);
    let hf2q_threads = MTLSize::new(2, 32, 1);

    // peer dispatch: (704, 1, 8), threads (32, 2, 1)
    let peer_tg = MTLSize::new((N as u64 + 2*2 - 1)/(2*2), 1, (N_TOKENS*TOP_K) as u64);
    let peer_threads = MTLSize::new(32, 2, 1);

    eprintln!("\n[H-I] gemma4 MoE down_exps: N={} K={} top_k={} (5632 TGs each)", N, K, TOP_K);
    eprintln!("[H-I] hf2q: tg={:?}, threads={:?}", (hf2q_tg.width, hf2q_tg.height, hf2q_tg.depth), (hf2q_threads.width, hf2q_threads.height, hf2q_threads.depth));
    eprintln!("[H-I] peer: tg={:?}, threads={:?}", (peer_tg.width, peer_tg.height, peer_tg.depth), (peer_threads.width, peer_threads.height, peer_threads.depth));

    let mut hf2q_meds = Vec::new();
    let mut peer_meds = Vec::new();

    for cycle in 0..3 {
        eprintln!("\n--- cycle {} ---", cycle);
        if cycle % 2 == 0 {
            let mh = bench("hf2q _id_nr2", &hf2q_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (3, &ids)],
                hf2q_bytes, 4, hf2q_tg, hf2q_threads, 0);
            std::thread::sleep(std::time::Duration::from_secs(3));
            let mp = bench("peer _id", &peer_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (4, &ids)],
                peer_bytes, 0, peer_tg, peer_threads, 8192);
            hf2q_meds.push(mh); peer_meds.push(mp);
        } else {
            let mp = bench("peer _id", &peer_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (4, &ids)],
                peer_bytes, 0, peer_tg, peer_threads, 8192);
            std::thread::sleep(std::time::Duration::from_secs(3));
            let mh = bench("hf2q _id_nr2", &hf2q_pipe, &device,
                &[(0, &weights), (1, &input), (2, &dst), (3, &ids)],
                hf2q_bytes, 4, hf2q_tg, hf2q_threads, 0);
            hf2q_meds.push(mh); peer_meds.push(mp);
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    let mh = hf2q_meds.iter().sum::<f64>() / hf2q_meds.len() as f64;
    let mp = peer_meds.iter().sum::<f64>() / peer_meds.len() as f64;
    let delta = 100.0 * (mp - mh) / mh;
    eprintln!("\n[H-I] aggregate (3 cycles):");
    eprintln!("  hf2q : mean {:.2}us  samples {:?}", mh, hf2q_meds);
    eprintln!("  peer : mean {:.2}us  samples {:?}", mp, peer_meds);
    eprintln!("  delta: peer is {:+.2}% vs hf2q", delta);
    let v = if delta.abs() < 2.0 { "TIED (both within ±2%)" }
            else if delta < 0.0 { "PEER FASTER — algorithmic lever exists" }
            else { "HF2Q FASTER — kernel quality not the bottleneck" };
    eprintln!("  verdict: {}", v);
}
