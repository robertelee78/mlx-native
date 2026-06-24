//! ADR-040 Phase F M2 / F3 microbench — mv vs mm GEMM at DECODE BATCH sizes.
//!
//! Tests hypothesis H-F3: at continuous-batching decode batch m=2..8 the
//! current dispatch routes to the `mv` kernel (re-reads every weight block
//! once PER ROW → no amortization), while the `mm` kernel stages a weight
//! tile in threadgroup memory and reads it ONCE for the m-row block. So
//! routing decode-batch m>=2 to mm should amortize the (decode-dominant)
//! weight read and BEAT mv, despite mm's staging overhead.
//!
//! Measures the crossover: for each gemma4 projection shape, time
//! `quantized_matmul_ggml` (= mv at m<=8) vs `dispatch_mm_for_test` (= forced
//! mm) at m=1,2,4,8. If mm beats mv at m>=2 → lower MM_ROUTING_THRESHOLD for
//! the decode path (F3). If not → F3 needs a small-m fused decode GEMM.
//!
//! Run: cargo bench -p mlx-native --bench bench_f3_decode_mv_vs_mm --release

use mlx_native::ops::quantized_matmul_ggml::{
    dispatch_mm_for_test, quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const WARMUP: usize = 10;
const MEASURE: usize = 50;
const BATCH: usize = 32;

struct Shape {
    label: &'static str,
    n: u32,
    k: u32,
    qtype: GgmlType,
}

// gemma4 26B-A4B APEX decode projection shapes (hidden=2816). Q6_K dominant.
const SHAPES: &[Shape] = &[
    Shape { label: "Q_proj", n: 4096, k: 2816, qtype: GgmlType::Q6_K },
    Shape { label: "O_proj", n: 2816, k: 4096, qtype: GgmlType::Q6_K },
    Shape { label: "lmhead", n: 262144, k: 2816, qtype: GgmlType::Q6_K },
];
const M_VALUES: &[u32] = &[1, 2, 4, 8];

fn alloc_weight(device: &MlxDevice, n: u32, k: u32, qt: GgmlType) -> MlxBuffer {
    let blocks_per_row = (k as u64) / (qt.block_values() as u64);
    let total = (n as u64) * blocks_per_row * (qt.block_bytes() as u64);
    device
        .alloc_buffer(total as usize, DType::U8, vec![total as usize])
        .expect("alloc weight")
}
fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("alloc f32")
}

/// Median batched-per-call microseconds. `use_mm=false` → mv (the production
/// path at m<=8); `use_mm=true` → forced mm via the test hook.
fn time_kernel(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &GgmlQuantizedMatmulParams,
    use_mm: bool,
) -> f64 {
    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("encoder");
        if use_mm {
            dispatch_mm_for_test(&mut enc, registry, device, input, weight, output, params)
                .expect("mm warmup");
        } else {
            quantized_matmul_ggml(&mut enc, registry, device, input, weight, output, params)
                .expect("mv warmup");
        }
        enc.commit_and_wait().expect("warmup commit");
    }
    let mut samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        for _ in 0..BATCH {
            if use_mm {
                dispatch_mm_for_test(&mut enc, registry, device, input, weight, output, params)
                    .expect("mm");
            } else {
                quantized_matmul_ggml(&mut enc, registry, device, input, weight, output, params)
                    .expect("mv");
            }
        }
        enc.commit_and_wait().expect("commit");
        samples.push(t0.elapsed().as_secs_f64() * 1.0e6 / BATCH as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!("F3 microbench — mv vs mm at decode batch m (gemma4 Q6_K shapes), M5 Max");
    println!(
        "{:<8} {:>3} {:>7} {:>7} {:>9} {:>9} {:>8}",
        "shape", "m", "N", "K", "mv_us", "mm_us", "mm_x"
    );
    println!("{}", "-".repeat(60));

    for s in SHAPES {
        let weight = alloc_weight(&device, s.n, s.k, s.qtype);
        for &m in M_VALUES {
            let input = alloc_f32(&device, (m as usize) * (s.k as usize));
            let output = alloc_f32(&device, (m as usize) * (s.n as usize));
            let params = GgmlQuantizedMatmulParams { m, n: s.n, k: s.k, ggml_type: s.qtype };
            let mv_us = time_kernel(&device, &mut registry, &input, &weight, &output, &params, false);
            let mm_us = time_kernel(&device, &mut registry, &input, &weight, &output, &params, true);
            let speedup = mv_us / mm_us;
            println!(
                "{:<8} {:>3} {:>7} {:>7} {:>9.1} {:>9.1} {:>7.2}x{}",
                s.label, m, s.n, s.k, mv_us, mm_us, speedup,
                if m >= 2 && speedup > 1.05 { "  <- mm wins" } else { "" }
            );
        }
        println!();
    }
    println!(
        "H-F3 holds if mm_x > 1 at m>=2 (mm amortizes the weight read). If so,\n\
         route decode-batch m>=2 to mm (lower MM_ROUTING_THRESHOLD for decode)."
    );
}
