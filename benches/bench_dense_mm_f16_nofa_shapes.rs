//! ADR-029 iter-88 H68: per-call isolated bench of `dense_matmul_f16_f32_tensor`
//! at the NO_FA attention shapes used at gemma4 pp8333 / pp4173.
//!
//! Purpose: compare f16-matmul throughput vs the bf16-matmul throughput
//! (iter-84 bench) at identical shapes. If f16 is significantly faster,
//! peer's split-attn-beats-FA advantage at gemma4 may be due to f16-typed
//! KV cache (per iter-87 finding: peer's common.h:317 defaults
//! cache_type_k/v to GGML_TYPE_F16; peer's split-attn uses
//! kernel_mul_mm_f16_f32 NOT bf16).
//!
//! Shapes match iter-84's bench_dense_mm_bf16_nofa_shapes exactly so
//! the bf16/f16 throughput ratio is the SOLE differentiator (kernel
//! tile geometry, dispatch grid, batch broadcast all identical).
//!
//! Run: cargo bench --bench bench_dense_mm_f16_nofa_shapes --release

use mlx_native::ops::dense_mm_f16::{dense_matmul_f16_f32_tensor, DenseMmF16F32Params};
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
    ShapeCase { label: "Q@K_pp4K",   m: 4173, n: 4173, k:  512, src0_batch: 2, src1_batch: 16 },
    ShapeCase { label: "scores@V_4K", m: 4173, n:  512, k: 4173, src0_batch: 2, src1_batch: 16 },
    ShapeCase { label: "Q@K_pp8K",   m: 8333, n: 8333, k:  512, src0_batch: 2, src1_batch: 16 },
    ShapeCase { label: "scores@V_8K", m: 8333, n:  512, k: 8333, src0_batch: 2, src1_batch: 16 },
];

const WARMUP_ITERS: usize = 3;
const MEASURE_ITERS: usize = 10;

fn alloc_f16(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 2, DType::F16, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

fn alloc_f32(device: &MlxDevice, n: usize, label: &str) -> MlxBuffer {
    device
        .alloc_buffer(n * 4, DType::F32, vec![n])
        .unwrap_or_else(|e| panic!("alloc {label}: {e}"))
}

fn bench_one(case: &ShapeCase, device: &MlxDevice, registry: &mut KernelRegistry) -> f64 {
    let src0 = alloc_f16(device, (case.src0_batch as usize) * (case.n as usize) * (case.k as usize), "src0");
    let src1 = alloc_f32(device, (case.src1_batch as usize) * (case.m as usize) * (case.k as usize), "src1");
    let dst  = alloc_f32(device, (case.src1_batch as usize) * (case.m as usize) * (case.n as usize), "dst");

    let params = DenseMmF16F32Params {
        m: case.m, n: case.n, k: case.k,
        src0_batch: case.src0_batch, src1_batch: case.src1_batch,
    };

    for _ in 0..WARMUP_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        dense_matmul_f16_f32_tensor(&mut enc, registry, device, &src0, &src1, &dst, &params)
            .expect("dispatch warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        dense_matmul_f16_f32_tensor(&mut enc, registry, device, &src0, &src1, &dst, &params)
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
        "H68 falsifier: compare to iter-84 bench_dense_mm_bf16_nofa_shapes baseline.\n\
         If f16 t/s == bf16 t/s -> dtype is NOT the lever; H68 falsified.\n\
         If f16 t/s > bf16 t/s by >10% -> H68 lever, prototype HF2Q_NOFA_F16 path."
    );
}
