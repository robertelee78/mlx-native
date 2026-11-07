//! Row-exact few-row mat-vec microbenchmark on the DeepSeek-V4 decode shapes.
//!
//! Model-free (synthetic weights). Needs a quiet host (HOST-LOCK held by the
//! runner). One trial encodes `REPS` dispatches in one command buffer with a
//! memory barrier between them (serialized, like dependent decode ops) and
//! divides the command buffer's GPU time (`GPUStartTime`..`GPUEndTime`) by
//! `REPS`; a cell is the median (and p10/p90) of `TRIALS` trials after
//! `WARMUP` untimed ones. A short blit burst first brings the GPU clock up.
//! `cold` cells rotate over enough weight copies to exceed 256 MiB, so each
//! dispatch reads its weights from DRAM; `warm` cells reuse one copy.
//!
//! Families (`ROW_EXACT_BENCH_FAMILIES`, comma list; default
//! `f16,q8_0,grouped,baseline`):
//! - `f16`, `q8_0`: `dispatch_row_exact_matvec` at m = 1..=4.
//! - `grouped`: DeepSeek-V4 attention output-A, 8 groups of Q8_0
//!   [1024 x 4096] on token-major input/output, one dispatch, m = 1..=4.
//! - `baseline`: the existing kernels on the same cells
//!   (`dense_matvec_f16w_f32io` once per row; `quantized_matmul_ggml` at m).
//! - `sweep`: every (nsg, nr0) schedule at m = 1 and 4 (schedule tuning) for
//!   the enabled families.
//!
//! Output: one JSON line per cell to `ROW_EXACT_BENCH_OUT` (default
//! `target/row_exact_matvec/cells.jsonl`) and a table on stdout. Reference
//! implementation numbers on the same shapes: `docs/peer-benchmarks.md`.
//!
//! Run: `cargo bench --bench bench_row_exact_matvec`

#![allow(clippy::expect_used, clippy::too_many_arguments)]

use std::io::Write;

use mlx_native::ops::dense_gemm::{dispatch_dense_matvec_f16w_f32io, DenseGemmF16Params};
use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, dispatch_row_exact_matvec_scheduled, row_exact_matvec_nsg_limit,
    RowExactMatvecSchedule, RowExactMatvecShape, RowExactWeight,
};
use mlx_native::{
    quantized_matmul_ggml, CommandEncoder, DType, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxBuffer, MlxDevice,
};

const WARMUP: usize = 5;
const TRIALS: usize = 15;
const REPS: usize = 16;
const COLD_BYTES: usize = 256 << 20;
const MAX_COPIES: usize = 64;

const F16_SHAPES: [(&str, usize, usize); 7] = [
    ("hc_fn", 24, 16384),
    ("output_hc_fn", 4, 16384),
    ("router", 256, 4096),
    ("compressor_r4", 1024, 4096),
    ("compressor_r128", 512, 4096),
    ("indexer_q_b", 8192, 1024),
    ("indexer_proj", 64, 4096),
];

const Q8_0_SHAPES: [(&str, usize, usize); 7] = [
    ("attn_q_b", 32768, 1024),
    ("attn_output_b", 4096, 8192),
    ("shexp_gate_up", 2048, 4096),
    ("shexp_down", 4096, 2048),
    ("attn_q_a", 1024, 4096),
    ("attn_kv", 512, 4096),
    ("output_head", 129_280, 4096),
];

#[derive(Clone, Copy)]
struct Stats {
    median: f64,
    p10: f64,
    p90: f64,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn weight(device: &MlxDevice, ty: RowExactWeight, rows: usize, k: usize, seed: u64) -> MlxBuffer {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    match ty {
        RowExactWeight::F16 => {
            let mut buffer = device
                .alloc_buffer(rows * k * 2, DType::F16, vec![rows, k])
                .expect("f16 weight");
            // Finite halves in ±[2^-8, 2^-7): sign + mantissa random, fixed exponent.
            for chunk in buffer.as_mut_slice::<u64>().expect("f16 words") {
                *chunk = (rng.next() & 0x83ff_83ff_83ff_83ff) | 0x1c00_1c00_1c00_1c00;
            }
            buffer
        }
        RowExactWeight::Q8_0 => {
            let blocks = rows * k / 32;
            let mut buffer = device
                .alloc_buffer(blocks * 34, DType::U8, vec![blocks * 34])
                .expect("q8_0 weight");
            let bytes = buffer.as_mut_slice::<u8>().expect("q8_0 bytes");
            for chunk in bytes.chunks_mut(8) {
                let r = rng.next().to_le_bytes();
                chunk.copy_from_slice(&r[..chunk.len()]);
            }
            let d = half::f16::from_f32(1.0 / 512.0).to_bits().to_le_bytes();
            for block in bytes.chunks_exact_mut(34) {
                block[..2].copy_from_slice(&d);
            }
            buffer
        }
        // The gemma decode op-parity bench covers the block-quant types.
        other => {
            panic!("bench_row_exact_matvec generates F16 and Q8_0 weights only, not {other:?}")
        }
    }
}

fn f32_buffer(device: &MlxDevice, len: usize, seed: u64) -> MlxBuffer {
    let mut rng = Rng(seed | 1);
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("f32");
    for v in buffer.as_mut_slice::<f32>().expect("f32 slice") {
        *v = (rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
    }
    buffer
}

/// Every schedule the `sweep` family times for inner dimension `k`.
fn sweep_schedules(k: usize) -> Vec<RowExactMatvecSchedule> {
    let mut out = Vec::new();
    for nsg in [1u32, 2, 4, 8, 16, 32] {
        if nsg > 1 && k < nsg as usize * 32 * 16 {
            continue;
        }
        for nr0 in [1u32, 2] {
            out.push(RowExactMatvecSchedule { nsg, nr0 });
        }
    }
    out
}

fn copies_for(bytes: usize, warm: bool) -> usize {
    if warm {
        1
    } else {
        COLD_BYTES.div_ceil(bytes).clamp(1, MAX_COPIES)
    }
}

struct Bench {
    device: MlxDevice,
    registry: KernelRegistry,
    out: std::fs::File,
    families: Vec<String>,
}

impl Bench {
    fn wants(&self, family: &str) -> bool {
        self.families.iter().any(|f| f == family)
    }

    /// The schedule `dispatch_row_exact_matvec` launches on this device.
    fn device_schedule(
        &mut self,
        ty: RowExactWeight,
        shape: &RowExactMatvecShape,
    ) -> RowExactMatvecSchedule {
        RowExactMatvecSchedule::for_device(
            &mut self.registry,
            self.device.metal_device(),
            ty,
            shape,
        )
        .expect("device schedule")
    }

    /// Sweep schedules this device's `ty` pipelines can launch.
    fn launchable_sweep(&mut self, ty: RowExactWeight, k: usize) -> Vec<RowExactMatvecSchedule> {
        let limit = row_exact_matvec_nsg_limit(&mut self.registry, self.device.metal_device(), ty)
            .expect("nsg limit");
        sweep_schedules(k)
            .into_iter()
            .filter(|s| s.nsg <= limit)
            .collect()
    }

    /// GPU microseconds per dispatch: (median, p10, p90).
    fn time(
        &mut self,
        mut encode: impl FnMut(&mut CommandEncoder, &mut KernelRegistry, usize),
    ) -> Stats {
        let mut samples = Vec::with_capacity(TRIALS);
        for trial in 0..WARMUP + TRIALS {
            let mut encoder = self.device.command_encoder().expect("encoder");
            for rep in 0..REPS {
                if rep > 0 {
                    encoder.memory_barrier();
                }
                encode(&mut encoder, &mut self.registry, rep);
            }
            let (start, end) = encoder.commit_wait_with_gpu_time().expect("GPU");
            if trial >= WARMUP {
                samples.push((end - start) * 1e6 / REPS as f64);
            }
        }
        samples.sort_by(f64::total_cmp);
        let at = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
        Stats {
            median: at(0.5),
            p10: at(0.1),
            p90: at(0.9),
        }
    }

    /// Keeps the GPU busy for a few hundred ms so clocks ramp before timing.
    fn spin_up(&mut self) {
        let bytes = 256usize << 20;
        let src = self
            .device
            .alloc_buffer(bytes, DType::U8, vec![bytes])
            .expect("spin src");
        let dst = self
            .device
            .alloc_buffer(bytes, DType::U8, vec![bytes])
            .expect("spin dst");
        for _ in 0..200 {
            let mut encoder = self.device.command_encoder().expect("encoder");
            encoder
                .blit_copy_bytes(&src, 0, &dst, 0, bytes as u64)
                .expect("blit");
            encoder.commit_and_wait().expect("spin");
        }
    }

    fn cell(&mut self, cell: serde_json::Value) {
        writeln!(self.out, "{cell}").expect("cell");
    }

    fn sweep_cell(
        &mut self,
        tag: &str,
        op: &str,
        n: usize,
        k: usize,
        batch: usize,
        m: usize,
        warm: bool,
        s: RowExactMatvecSchedule,
        t: Stats,
    ) {
        println!(
            "sweep {tag} {op:16} {batch} x [{n:6} x {k:5}] m={m} {} {s:?}: {:8.2} us",
            if warm { "warm" } else { "cold" },
            t.median
        );
        self.cell(serde_json::json!({"family": "sweep", "weight": tag, "op": op, "batch": batch, "n": n, "k": k,
            "m": m, "warm": warm, "nsg": s.nsg, "nr0": s.nr0,
            "us": t.median, "p10_us": t.p10, "p90_us": t.p90}));
    }

    /// Times `dispatch_row_exact_matvec` (or an explicit schedule) at m.
    fn time_row_exact(
        &mut self,
        ty: RowExactWeight,
        weights: &[MlxBuffer],
        x: &MlxBuffer,
        y: &MlxBuffer,
        shape: RowExactMatvecShape,
        schedule: Option<RowExactMatvecSchedule>,
    ) -> Stats {
        let device = self.device.metal_device().to_owned();
        self.time(|enc, reg, rep| {
            let w = &weights[rep % weights.len()];
            match schedule {
                Some(s) => {
                    dispatch_row_exact_matvec_scheduled(enc, reg, &device, ty, w, x, y, &shape, &s)
                }
                None => dispatch_row_exact_matvec(enc, reg, &device, ty, w, x, y, &shape),
            }
            .expect("row-exact dispatch");
        })
    }

    fn baseline(
        &mut self,
        ty: RowExactWeight,
        weights: &[MlxBuffer],
        x: &MlxBuffer,
        y: &MlxBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Stats {
        let device = self.device.clone();
        let x_rows: Vec<MlxBuffer> = (0..m)
            .map(|j| x.slice_view((j * k * 4) as u64, k))
            .collect();
        let y_rows: Vec<MlxBuffer> = (0..m)
            .map(|j| y.slice_view((j * n * 4) as u64, n))
            .collect();
        self.time(|enc, reg, rep| {
            let w = &weights[rep % weights.len()];
            match ty {
                RowExactWeight::F16 => {
                    for j in 0..m {
                        dispatch_dense_matvec_f16w_f32io(
                            enc,
                            reg,
                            device.metal_device(),
                            &x_rows[j],
                            w,
                            &y_rows[j],
                            &DenseGemmF16Params {
                                m: 1,
                                n: n as u32,
                                k: k as u32,
                            },
                        )
                        .expect("f16 matvec");
                    }
                }
                quantized => quantized_matmul_ggml(
                    enc,
                    reg,
                    &device,
                    x,
                    w,
                    y,
                    &GgmlQuantizedMatmulParams {
                        m: m as u32,
                        n: n as u32,
                        k: k as u32,
                        ggml_type: match quantized {
                            RowExactWeight::Q8_0 => GgmlType::Q8_0,
                            RowExactWeight::Q4_K => GgmlType::Q4_K,
                            RowExactWeight::Q6_K => GgmlType::Q6_K,
                            RowExactWeight::Q5_0 => GgmlType::Q5_0,
                            RowExactWeight::F16 => unreachable!("handled above"),
                        },
                    },
                )
                .expect("quantized matmul"),
            }
        })
    }

    fn dense_family(&mut self, ty: RowExactWeight, shapes: &[(&str, usize, usize)], sweep: bool) {
        let tag = match ty {
            RowExactWeight::F16 => "f16",
            RowExactWeight::Q8_0 => "q8_0",
            RowExactWeight::Q4_K => "q4_K",
            RowExactWeight::Q6_K => "q6_K",
            RowExactWeight::Q5_0 => "q5_0",
        };
        for &(op, n, k) in shapes {
            let bytes = ty.row_bytes(k as u32) as usize * n;
            let warm_modes: &[bool] = if bytes > COLD_BYTES {
                &[false]
            } else {
                &[true, false]
            };
            for &warm in warm_modes {
                let copies = copies_for(bytes, warm);
                let weights: Vec<MlxBuffer> = (0..copies)
                    .map(|c| weight(&self.device, ty, n, k, c as u64 + 1))
                    .collect();
                let x = f32_buffer(&self.device, 8 * k, 7);
                let y = self
                    .device
                    .alloc_buffer(8 * n * 4, DType::F32, vec![8 * n])
                    .expect("y");
                for m in 1..=4usize {
                    let shape = RowExactMatvecShape::contiguous(m as u32, n as u32, k as u32);
                    if sweep {
                        if m != 1 && m != 4 {
                            continue;
                        }
                        for s in self.launchable_sweep(ty, k) {
                            let t = self.time_row_exact(ty, &weights, &x, &y, shape, Some(s));
                            self.sweep_cell(tag, op, n, k, 1, m, warm, s, t);
                        }
                        continue;
                    }
                    let auto = self.device_schedule(ty, &shape);
                    let t = self.time_row_exact(ty, &weights, &x, &y, shape, None);
                    let us = t.median;
                    let base = if self.wants("baseline") {
                        Some(self.baseline(ty, &weights, &x, &y, m, n, k))
                    } else {
                        None
                    };
                    println!(
                        "{tag:4} {op:16} [{n:6} x {k:5}] m={m} {}: row_exact {us:8.2} us (nsg {:2}, nr0 {})  baseline {}",
                        if warm { "warm" } else { "cold" },
                        auto.nsg,
                        auto.nr0,
                        base.map_or("-".into(), |b| format!("{:8.2} us", b.median)),
                    );
                    self.cell(serde_json::json!({"family": tag, "op": op, "n": n, "k": k, "m": m, "warm": warm,
                        "copies": copies, "weight_bytes": bytes, "nsg": auto.nsg, "nr0": auto.nr0, "row_exact_us": us,
                        "row_exact_p10_us": t.p10, "row_exact_p90_us": t.p90,
                        "baseline_us": base.map(|b| b.median), "baseline_p10_us": base.map(|b| b.p10),
                        "baseline_p90_us": base.map(|b| b.p90), "gbps": bytes as f64 / us / 1e3}));
                }
            }
        }
    }

    /// Grouped output-A: one row-exact dispatch on token-major activations;
    /// baseline = the 8 per-group `quantized_matmul_ggml` dispatches on
    /// group-major views (the two activation permutes it needs excluded).
    fn grouped(&mut self, sweep: bool) {
        let (groups, rank, gw) = (8usize, 1024usize, 4096usize);
        let ty = RowExactWeight::Q8_0;
        let row_bytes = gw / 32 * 34;
        let bytes = groups * rank * row_bytes;
        for warm in [true, false] {
            let tag = if warm { "warm" } else { "cold" };
            let copies = copies_for(bytes, warm);
            let weights: Vec<MlxBuffer> = (0..copies)
                .map(|c| weight(&self.device, ty, groups * rank, gw, 50 + c as u64))
                .collect();
            let x = f32_buffer(&self.device, 8 * groups * gw, 5);
            let y = self
                .device
                .alloc_buffer(8 * groups * rank * 4, DType::F32, vec![8 * groups * rank])
                .expect("y");
            for m in 1..=4usize {
                let shape = RowExactMatvecShape::grouped_token_major(
                    m as u32,
                    groups as u32,
                    rank as u32,
                    gw as u32,
                );
                if sweep {
                    if m != 1 && m != 4 {
                        continue;
                    }
                    for s in self.launchable_sweep(ty, gw) {
                        let t = self.time_row_exact(ty, &weights, &x, &y, shape, Some(s));
                        self.sweep_cell(
                            "q8_0",
                            "attn_output_a_grouped",
                            rank,
                            gw,
                            groups,
                            m,
                            warm,
                            s,
                            t,
                        );
                    }
                    continue;
                }
                let auto = self.device_schedule(ty, &shape);
                let t = self.time_row_exact(ty, &weights, &x, &y, shape, None);
                let base = if self.wants("baseline") {
                    let device = self.device.clone();
                    let xg = x.slice_view(0, m * groups * gw);
                    let yg = y.slice_view(0, m * groups * rank);
                    Some(self.time(|enc, reg, rep| {
                        let w = &weights[rep % weights.len()];
                        for g in 0..groups {
                            let xv = xg.slice_view((g * m * gw * 4) as u64, m * gw);
                            let wv = w.slice_view((g * rank * row_bytes) as u64, rank * row_bytes);
                            let yv = yg.slice_view((g * m * rank * 4) as u64, m * rank);
                            quantized_matmul_ggml(
                                enc,
                                reg,
                                &device,
                                &xv,
                                &wv,
                                &yv,
                                &GgmlQuantizedMatmulParams {
                                    m: m as u32,
                                    n: rank as u32,
                                    k: gw as u32,
                                    ggml_type: GgmlType::Q8_0,
                                },
                            )
                            .expect("q8_0 group matmul");
                        }
                    }))
                } else {
                    None
                };
                println!(
                    "q8_0 attn_output_a    8 x [{rank} x {gw}] m={m} {tag}: row_exact {:8.2} us (nsg {}, nr0 {})  baseline {}",
                    t.median,
                    auto.nsg,
                    auto.nr0,
                    base.map_or("-".into(), |b| format!("{:8.2} us", b.median)),
                );
                self.cell(serde_json::json!({"family": "grouped", "op": "attn_output_a_grouped", "groups": groups,
                    "n": rank, "k": gw, "m": m, "warm": warm, "copies": copies, "weight_bytes": bytes,
                    "nsg": auto.nsg, "nr0": auto.nr0, "row_exact_us": t.median,
                    "row_exact_p10_us": t.p10, "row_exact_p90_us": t.p90,
                    "baseline_us": base.map(|b| b.median), "baseline_p10_us": base.map(|b| b.p10),
                    "baseline_p90_us": base.map(|b| b.p90), "gbps": bytes as f64 / t.median / 1e3}));
            }
        }
    }
}

fn main() {
    let out_path: std::path::PathBuf = std::env::var_os("ROW_EXACT_BENCH_OUT")
        .map(Into::into)
        .unwrap_or_else(|| "target/row_exact_matvec/cells.jsonl".into());
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("out dir");
    }
    let families = std::env::var("ROW_EXACT_BENCH_FAMILIES")
        .unwrap_or_else(|_| "f16,q8_0,grouped,baseline".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let device = MlxDevice::new().expect("device");
    let mut bench = Bench {
        device,
        registry: KernelRegistry::new(),
        out: std::fs::File::create(&out_path).expect("cells file"),
        families,
    };
    bench.spin_up();
    let sweep = bench.wants("sweep");
    if bench.wants("f16") {
        bench.dense_family(RowExactWeight::F16, &F16_SHAPES, sweep);
    }
    if bench.wants("q8_0") {
        bench.dense_family(RowExactWeight::Q8_0, &Q8_0_SHAPES, sweep);
    }
    if bench.wants("grouped") {
        bench.grouped(sweep);
    }
    println!("cells: {}", out_path.display());
}
