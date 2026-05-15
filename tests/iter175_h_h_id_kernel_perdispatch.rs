//! ADR-029 iter-175 Step 1ad — H-H: per-dispatch isolated timing for
//! `kernel_mul_mv_id_q6_K_f32_nr2` at gemma4 MoE-decode shape.
//!
//! Background: Step 1y profile showed FFN dominates ATTN ~2:1
//! (~195 µs/layer vs ~100 µs/layer).  FFN is MoE experts; the
//! `_id` kernels carry an expert_id indirection on every weight
//! access.  Step 1y named this kernel as the #1 FFN candidate.
//!
//! Hypothesis: the `_id` indirection costs measurable per-dispatch
//! time vs the non-`_id` variant at the same dimension count.
//! Quantify the delta to know how much the indirection is worth.
//!
//! gemma4 A4B MoE down_exps shape (decode):
//!   - 128 total experts, 8 active (top_k=8)
//!   - Per expert: [N=2816, K=8192] Q6_K  (compresses ffn_dim back to hidden)
//!   - At decode: n_tokens=1, total_output_rows = 1*8 = 8
//!
//! Run: `cargo test --release --test iter175_h_h_id_kernel_perdispatch -- --nocapture`

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use metal::{ComputePipelineDescriptor, FunctionConstantValues, MTLDataType, MTLSize};
use mlx_native::{DType, MlxBuffer, MlxDevice};

const KERNEL_ID: &str = "kernel_mul_mv_id_q6_K_f32_nr2";
const KERNEL_NON_ID: &str = "kernel_mul_mv_q6_K_f32_nr2";
const SHADER_ID: &str = "src/shaders/quantized_matmul_id_ggml.metal";
const SHADER_NON_ID: &str = "src/shaders/quantized_matmul_ggml.metal";

// gemma4 MoE down_exps decode shape.
const N: u32 = 2816;  // hidden
const K: u32 = 8192;  // ffn_dim
const N_EXPERTS: u32 = 128;
const TOP_K: u32 = 8;
const N_TOKENS: u32 = 1;

const QK_K: u64 = 256;
const Q6_K_BLOCK_BYTES: u64 = 210;

const WARMUP: usize = 20;
const MEASURE: usize = 80;
const BATCH: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GgmlMatvecIdGpuParams {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne10: i64,
    ne12: i64,
    ne0: i64,
    ne1: i64,
    r2: u32,
    r3: u32,
    top_k: u32,
    n_tokens: u32,
    expert_stride: i64,
}

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

fn build_metallib(shader_path: &str, suffix: &str, out_dir: &str) -> PathBuf {
    let abs_shader = std::fs::canonicalize(shader_path).expect("shader exists");
    let air_path = format!("{}/iter175_h_h_{}.air", out_dir, suffix);
    let metallib_path = format!("{}/iter175_h_h_{}.metallib", out_dir, suffix);

    let air_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal", "-O3", "-c"])
        .arg(&abs_shader)
        .arg("-o")
        .arg(&air_path)
        .status()
        .expect("run xcrun metal");
    assert!(air_status.success(), "xcrun metal -O3 failed for {}", shader_path);

    let metallib_status = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&metallib_path)
        .status()
        .expect("run xcrun metallib");
    assert!(metallib_status.success(), "xcrun metallib failed for {}", shader_path);

    PathBuf::from(metallib_path)
}

fn make_fcs() -> FunctionConstantValues {
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

fn build_pipeline(
    device: &metal::DeviceRef,
    lib_path: &PathBuf,
    kernel_name: &str,
    label: &str,
    use_fcs: bool,
) -> metal::ComputePipelineState {
    let lib = device.new_library_with_file(lib_path).expect("load lib");
    let function = if use_fcs {
        lib.get_function(kernel_name, Some(make_fcs())).expect("get_function (FC)")
    } else {
        lib.get_function(kernel_name, None).expect("get_function")
    };
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&function));
    desc.set_label(label);
    device.new_compute_pipeline_state(&desc).expect("pipeline")
}

fn alloc_weights_q6_k(device: &MlxDevice, n_experts: u32, n: u32, k: u32) -> MlxBuffer {
    let blocks_per_row = (k as u64) / QK_K;
    let per_expert_bytes = (n as u64) * blocks_per_row * Q6_K_BLOCK_BYTES;
    let total_bytes = (n_experts as u64) * per_expert_bytes;
    device.alloc_buffer(total_bytes as usize, DType::U8, vec![total_bytes as usize]).expect("alloc weights")
}

fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("alloc f32")
}

fn alloc_u32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n * 4, DType::U32, vec![n]).expect("alloc u32")
}

fn bench(
    label: &str,
    pipeline: &metal::ComputePipelineStateRef,
    device: &MlxDevice,
    buffers: &[(u64, &MlxBuffer)],
    params_bytes: &[u8],
    params_slot: u64,
    threadgroups: MTLSize,
    threads_per_tg: MTLSize,
) -> (f64, f64) {
    let queue = device.metal_queue();

    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (slot, buf) in buffers {
            enc.set_buffer(*slot, Some(buf.metal_buffer()), buf.byte_offset());
        }
        enc.set_bytes(params_slot, params_bytes.len() as u64, params_bytes.as_ptr().cast());
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
            for (slot, buf) in buffers {
                enc.set_buffer(*slot, Some(buf.metal_buffer()), buf.byte_offset());
            }
            enc.set_bytes(params_slot, params_bytes.len() as u64, params_bytes.as_ptr().cast());
            enc.dispatch_thread_groups(threadgroups, threads_per_tg);
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        samples.push(t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let p10 = samples[samples.len() / 10];
    eprintln!("  {:<28} median={:>8.2}us  p10={:>7.2}", label, median, p10);
    (median, p10)
}

#[test]
fn h_h_id_kernel_perdispatch() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let out_dir = std::env::temp_dir().to_string_lossy().to_string();

    let lib_id = build_metallib(SHADER_ID, "id", &out_dir);
    let lib_non_id = build_metallib(SHADER_NON_ID, "nonid", &out_dir);
    eprintln!("[H-H] built libs at: {}", out_dir);

    let pipe_id = build_pipeline(device.metal_device(), &lib_id, KERNEL_ID, "id-precompiled", false);
    let pipe_non_id = build_pipeline(device.metal_device(), &lib_non_id, KERNEL_NON_ID, "non-id-precompiled", true);

    // Common allocations
    let weights = alloc_weights_q6_k(&device, N_EXPERTS, N, K);
    let input = alloc_f32(&device, (N_TOKENS * K) as usize);
    let dst_id = alloc_f32(&device, (N_TOKENS * TOP_K * N) as usize);
    let dst_non_id = alloc_f32(&device, (N_TOKENS * N) as usize);
    let ids = alloc_u32(&device, (N_TOKENS * TOP_K) as usize);

    // _id kernel params
    let blocks_per_row = (K as u64) / QK_K;
    let per_expert_bytes = (N as u64) * blocks_per_row * Q6_K_BLOCK_BYTES;
    let params_id = GgmlMatvecIdGpuParams {
        ne00: K as i64,
        ne01: N as i64,
        ne02: 1,
        ne10: K as i64,
        ne12: 1,
        ne0: N as i64,
        ne1: (N_TOKENS * TOP_K) as i64,
        r2: 1, r3: 1,
        top_k: TOP_K,
        n_tokens: N_TOKENS,
        expert_stride: per_expert_bytes as i64,
    };
    let params_id_bytes = unsafe {
        std::slice::from_raw_parts(
            (&params_id as *const GgmlMatvecIdGpuParams).cast::<u8>(),
            std::mem::size_of::<GgmlMatvecIdGpuParams>(),
        )
    };

    // non-_id kernel params (no top_k, no expert_stride, no n_tokens)
    let params_non_id = GgmlMatvecGpuParams {
        ne00: K as i64,
        ne01: N as i64,
        ne02: 1,
        ne10: K as i64,
        ne12: 1,
        ne0: N as i64,
        ne1: N_TOKENS as i64,
        r2: 1, r3: 1,
    };
    let params_non_id_bytes = unsafe {
        std::slice::from_raw_parts(
            (&params_non_id as *const GgmlMatvecGpuParams).cast::<u8>(),
            std::mem::size_of::<GgmlMatvecGpuParams>(),
        )
    };

    // _id kernel dispatch geometry: threadgroups = (ceil(N/4), n_tokens*top_k, 1)
    let align: u64 = 4;
    let tg_id = MTLSize::new(((N as u64) + align - 1) / align, (N_TOKENS * TOP_K) as u64, 1);
    let threads = MTLSize::new(2, 32, 1);

    // non-_id kernel dispatch geometry: threadgroups = (ceil(N/4), n_tokens, 1)
    let tg_non_id = MTLSize::new(((N as u64) + align - 1) / align, N_TOKENS as u64, 1);

    eprintln!("\n[H-H] gemma4 MoE down_exps shape: N={} K={} n_experts={} top_k={} n_tokens={}",
              N, K, N_EXPERTS, TOP_K, N_TOKENS);
    eprintln!("[H-H] _id   dispatch: tgs={:?}, threads={:?}", (tg_id.width, tg_id.height, tg_id.depth), (threads.width, threads.height, threads.depth));
    eprintln!("[H-H] non-id dispatch: tgs={:?}, threads={:?}", (tg_non_id.width, tg_non_id.height, tg_non_id.depth), (threads.width, threads.height, threads.depth));
    eprintln!("[H-H] BATCH={}, WARMUP={}, MEASURE={}", BATCH, WARMUP, MEASURE);

    // Run 3 cycles alt-paired.
    let mut id_meds = Vec::new();
    let mut non_id_meds = Vec::new();
    for cycle in 0..3 {
        eprintln!("\n--- cycle {} ---", cycle);
        if cycle % 2 == 0 {
            let (m_id, _) = bench(
                "_id kernel (top_k=8)",
                &pipe_id, &device,
                &[(0, &weights), (1, &input), (2, &dst_id), (3, &ids)],
                params_id_bytes, 4, tg_id, threads,
            );
            std::thread::sleep(std::time::Duration::from_secs(3));
            let (m_non_id, _) = bench(
                "non-_id kernel (single)",
                &pipe_non_id, &device,
                &[(0, &weights), (1, &input), (2, &dst_non_id)],
                params_non_id_bytes, 3, tg_non_id, threads,
            );
            id_meds.push(m_id); non_id_meds.push(m_non_id);
        } else {
            let (m_non_id, _) = bench(
                "non-_id kernel (single)",
                &pipe_non_id, &device,
                &[(0, &weights), (1, &input), (2, &dst_non_id)],
                params_non_id_bytes, 3, tg_non_id, threads,
            );
            std::thread::sleep(std::time::Duration::from_secs(3));
            let (m_id, _) = bench(
                "_id kernel (top_k=8)",
                &pipe_id, &device,
                &[(0, &weights), (1, &input), (2, &dst_id), (3, &ids)],
                params_id_bytes, 4, tg_id, threads,
            );
            id_meds.push(m_id); non_id_meds.push(m_non_id);
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    let m_id = id_meds.iter().sum::<f64>() / id_meds.len() as f64;
    let m_non = non_id_meds.iter().sum::<f64>() / non_id_meds.len() as f64;

    eprintln!("\n[H-H] aggregate (3 alt-paired cycles):");
    eprintln!("  _id    mean: {:.2}us  samples: {:?}", m_id, id_meds);
    eprintln!("  non-id mean: {:.2}us  samples: {:?}", m_non, non_id_meds);
    // _id processes top_k=8x more rows; expected ~8x time if perfect scaling.
    let per_row_id = m_id / (TOP_K as f64);
    eprintln!("  _id per-row (÷top_k={}): {:.3}us", TOP_K, per_row_id);
    eprintln!("  non-id per-row:            {:.3}us", m_non);
    let overhead_pct = 100.0 * (per_row_id - m_non) / m_non;
    eprintln!("  _id indirection overhead per-row: {:+.2}%", overhead_pct);
}
