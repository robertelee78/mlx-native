//! ADR-029 iter-84: per-call isolated bench of `dense_matmul_bf16_f32_tensor`
//! at the NO_FA attention shapes used at gemma4 pp8333 / pp4173.
//!
//! Purpose: localize whether the NO_FA performance gap vs FA lives in the
//! matmul kernel itself or in the surrounding dispatch sequence
//! (barriers, intermediate buffer materialization, pipeline overhead).
//!
//! At pp8333 the global-attn NO_FA path has 2 dispatches whose throughput
//! we measure here:
//!
//!   Q @ K^T:   src0 = K  [nkv=2,  kL=8333,  hd=512] bf16
//!              src1 = Q  [nh=16,  qL=8333,  hd=512] f32
//!              dst  = kq [nh=16,  qL=8333,  kL=8333] f32
//!              Params: m=8333 (qL), n=8333 (kL), k=512 (hd),
//!                      src0_batch=2 (nkv), src1_batch=16 (nh, r2=8)
//!
//!   scores @ V: src0 = V_t [nkv=2,  hd=512,  kL=8333] bf16
//!               src1 = kq  [nh=16,  qL=8333, kL=8333] f32
//!               dst  = attn[nh=16,  qL=8333, hd=512] f32
//!               Params: m=8333 (qL), n=512 (hd), k=8333 (kL),
//!                       src0_batch=2 (nkv), src1_batch=16 (nh, r2=8)
//!
//! Per layer FLOPs: 2 * 8333 * 8333 * 512 * 16 = 1.14 TFLOPS (each dispatch).
//! Theoretical at 25 TFLOPS bf16 peak = ~46 ms per call.
//!
//! Reports median ms / GFLOPS achieved / TFLOPS achieved / % of 25-TFLOP peak.
//!
//! Run: cargo bench --bench bench_dense_mm_bf16_nofa_shapes --release

use mlx_native::ops::dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

struct ShapeCase {
    label:       &'static str,
    m:           u32,
    n:           u32,
    k:           u32,
    src0_batch:  u32,
    src1_batch:  u32,
}

const SHAPES: &[ShapeCase] = &[
    // pp4173 shapes
    ShapeCase { label: "Q@K_pp4K",   m: 4173, n: 4173, k:  512, src0_batch: 2, src1_batch: 16 },
    ShapeCase { label: "scores@V_4K", m: 4173, n:  512, k: 4173, src0_batch: 2, src1_batch: 16 },
    // pp8333 shapes
    ShapeCase { label: "Q@K_pp8K",   m: 8333, n: 8333, k:  512, src0_batch: 2, src1_batch: 16 },
    ShapeCase { label: "scores@V_8K", m: 8333, n:  512, k: 8333, src0_batch: 2, src1_batch: 16 },
];

const WARMUP_ITERS: usize = 3;
const MEASURE_ITERS: usize = 10;

fn alloc_bf16(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 2, DType::BF16, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

fn alloc_f32(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

fn bench_one(case: &ShapeCase, device: &MlxDevice, registry: &mut KernelRegistry) -> f64 {
    let src0 = alloc_bf16(device, (case.src0_batch as usize) * (case.n as usize) * (case.k as usize), "src0");
    let src1 = alloc_f32 (device, (case.src1_batch as usize) * (case.m as usize) * (case.k as usize), "src1");
    let dst  = alloc_f32 (device, (case.src1_batch as usize) * (case.m as usize) * (case.n as usize), "dst");

    let params = DenseMmBf16F32Params {
        m: case.m, n: case.n, k: case.k,
        src0_batch: case.src0_batch, src1_batch: case.src1_batch,
    };

    // Warmup (triggers PSO compile + thermal warmup).
    for _ in 0..WARMUP_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        dense_matmul_bf16_f32_tensor(&mut enc, registry, device, &src0, &src1, &dst, &params)
            .expect("dispatch warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    // Measure — per-iteration commit_and_wait for isolated GPU time.
    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        dense_matmul_bf16_f32_tensor(&mut enc, registry, device, &src0, &src1, &dst, &params)
            .expect("dispatch measure");
        enc.commit_and_wait().expect("measure commit");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        samples.push(elapsed);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!(
        "{:<16} {:>5} {:>5} {:>5} {:>4} {:>4} {:>9} {:>10} {:>9} {:>9}",
        "shape", "M", "N", "K", "S0B", "S1B", "median_ms", "GFLOPs", "TFLOP/s", "%peak25"
    );
    println!("{}", "-".repeat(102));

    for case in SHAPES {
        let median = bench_one(case, &device, &mut registry);
        // FLOPs: 2 * M * N * K * src1_batch (output broadcast across heads)
        let flops = 2.0 * (case.m as f64) * (case.n as f64) * (case.k as f64) * (case.src1_batch as f64);
        let gflops = flops / 1e9;
        let tflops = flops / (median / 1000.0) / 1e12;
        let pct_peak25 = tflops / 25.0 * 100.0;
        println!(
            "{:<16} {:>5} {:>5} {:>5} {:>4} {:>4} {:>9.3} {:>10.1} {:>9.2} {:>8.1}%",
            case.label, case.m, case.n, case.k,
            case.src0_batch, case.src1_batch,
            median, gflops, tflops, pct_peak25,
        );
    }
    println!("{}", "-".repeat(102));
    println!(
        "Per-call GPU-isolated throughput at NO_FA attention shapes. \n\
         Compare to ADR-029 iter-81 NOFA_QK avg=21.3 ms/call, NOFA_SV avg=35.5 ms/call (pp8333, mixed sliding+global)."
    );
}
