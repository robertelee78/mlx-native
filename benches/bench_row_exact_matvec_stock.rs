//! Stock-order row-exact mat-vec microbenchmark on the DeepSeek-V4 decode
//! shapes: `row_exact_matvec_stock` against the stock kernels and the
//! Class-B `row_exact_matvec` kernels on the same cells.
//!
//! Timing as `bench_row_exact_matvec`: one trial encodes `REPS` dispatches in
//! one command buffer with a memory barrier between them and divides the
//! command buffer's GPU time by `REPS`; a cell is the median (p10/p90) of
//! `TRIALS` trials after `WARMUP` untimed ones, after a blit burst brings the
//! GPU clock up. `cold` cells rotate over enough weight copies to exceed
//! 256 MiB; `warm` cells reuse one copy.
//!
//! Arms per cell (m = 1..=4):
//! - `stock`: the kernels hf2q uses with the switch off. F16:
//!   `dense_matvec_f16w_f32io` once per row; Q8_0: `quantized_matmul_ggml`
//!   at m (the nr2 matvec, rows on grid y); grouped output-A:
//!   `quantized_matmul_ggml_batched_mv` at m on group-major buffers (the two
//!   activation permutes hf2q adds are not timed).
//! - `class_b`: `dispatch_row_exact_matvec` (the A+C kernels).
//! - `stock_order`: `dispatch_row_exact_matvec_stock` (default packing).
//!
//! `ROW_EXACT_STOCK_BENCH_FAMILIES` (comma list; default `arms`): `arms`
//! times the three arms; `sweep` times every stock-order packing at m = 1 and
//! 4. `ROW_EXACT_STOCK_BENCH_WEIGHTS` (`f16`, `q8_0` or both; default both)
//! limits the cells. Output: JSON lines to `ROW_EXACT_STOCK_BENCH_OUT`
//! (default `target/row_exact_matvec_stock/cells.jsonl`) and a table on
//! stdout.
//!
//! Run: `cargo bench --bench bench_row_exact_matvec_stock`

#![allow(clippy::expect_used, clippy::too_many_arguments)]

use std::io::Write;

use mlx_native::ops::dense_gemm::{dispatch_dense_matvec_f16w_f32io, DenseGemmF16Params};
use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, RowExactMatvecShape, RowExactWeight,
};
use mlx_native::ops::row_exact_matvec_stock::{
    dispatch_row_exact_matvec_stock, dispatch_row_exact_matvec_stock_packed, RowExactStockPacking,
};
use mlx_native::{
    quantized_matmul_ggml, quantized_matmul_ggml_batched_mv, CommandEncoder, DType,
    GgmlBatchedQuantizedMatmulParams, GgmlQuantizedMatmulParams, GgmlType, KernelRegistry,
    MlxBuffer, MlxDevice,
};

const WARMUP: usize = 5;
const TRIALS: usize = 15;
const REPS: usize = 16;
const COLD_BYTES: usize = 256 << 20;
const MAX_COPIES: usize = 64;

/// (op, weight, groups, n, k): deficit-table-v2 matvec rows.
const CELLS: [(&str, RowExactWeight, usize, usize, usize); 15] = [
    ("hc_fn", RowExactWeight::F16, 1, 24, 16384),
    ("output_hc_fn", RowExactWeight::F16, 1, 4, 16384),
    ("router", RowExactWeight::F16, 1, 256, 4096),
    ("compressor_r4", RowExactWeight::F16, 1, 1024, 4096),
    ("compressor_r128", RowExactWeight::F16, 1, 512, 4096),
    ("indexer_q_b", RowExactWeight::F16, 1, 8192, 1024),
    ("indexer_proj", RowExactWeight::F16, 1, 64, 4096),
    ("attn_q_a", RowExactWeight::Q8_0, 1, 1024, 4096),
    ("attn_kv", RowExactWeight::Q8_0, 1, 512, 4096),
    ("attn_q_b", RowExactWeight::Q8_0, 1, 32768, 1024),
    ("attn_output_b", RowExactWeight::Q8_0, 1, 4096, 8192),
    ("shexp_gate_up", RowExactWeight::Q8_0, 1, 2048, 4096),
    ("shexp_down", RowExactWeight::Q8_0, 1, 4096, 2048),
    ("output_head", RowExactWeight::Q8_0, 1, 129_280, 4096),
    ("attn_output_a", RowExactWeight::Q8_0, 8, 1024, 4096),
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
            for chunk in buffer.as_mut_slice::<u64>().expect("f16 words") {
                *chunk = (rng.next() & 0x83ff_83ff_83ff_83ff) | 0x1c00_1c00_1c00_1c00;
            }
            buffer
        }
        RowExactWeight::Q8_0 => {
            let blocks = rows * k / 32;
            let mut buffer = device
                .alloc_buffer(blocks * 34, DType::U8, vec![rows, k])
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
        other => unreachable!("the stock-order family has no {other:?} kernels"),
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

fn copies_for(bytes: usize, warm: bool) -> usize {
    if warm {
        1
    } else {
        COLD_BYTES.div_ceil(bytes).clamp(1, MAX_COPIES)
    }
}

fn sweep_packings(ty: RowExactWeight) -> Vec<RowExactStockPacking> {
    let mut out = Vec::new();
    match ty {
        RowExactWeight::F16 => {
            for rows in [1, 4] {
                for unroll in [1, 4] {
                    for simdgroups in [1, 2, 4] {
                        out.push(RowExactStockPacking::F16 {
                            rows,
                            simdgroups,
                            unroll,
                        });
                    }
                }
            }
        }
        RowExactWeight::Q8_0 => {
            for rows in [1, 2] {
                for virtual_simdgroups in [1, 2, 4] {
                    out.push(RowExactStockPacking::Q8_0 {
                        rows,
                        virtual_simdgroups,
                    });
                }
            }
        }
        other => unreachable!("the stock-order family has no {other:?} kernels"),
    }
    out
}

struct Bench {
    device: MlxDevice,
    registry: KernelRegistry,
    out: std::fs::File,
    weights: Vec<String>,
}

impl Bench {
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

    fn stock(
        &mut self,
        ty: RowExactWeight,
        weights: &[MlxBuffer],
        x: &MlxBuffer,
        y: &MlxBuffer,
        m: usize,
        groups: usize,
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
            match (ty, groups) {
                (RowExactWeight::F16, _) => {
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
                        .expect("stock f16");
                    }
                }
                (RowExactWeight::Q8_0, 1) => quantized_matmul_ggml(
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
                        ggml_type: GgmlType::Q8_0,
                    },
                )
                .expect("stock q8_0"),
                (RowExactWeight::Q8_0, _) => quantized_matmul_ggml_batched_mv(
                    enc,
                    reg,
                    &device,
                    x,
                    w,
                    y,
                    &GgmlBatchedQuantizedMatmulParams {
                        batch: groups as u32,
                        m: m as u32,
                        n: n as u32,
                        k: k as u32,
                        ggml_type: GgmlType::Q8_0,
                    },
                )
                .expect("stock grouped q8_0"),
                (other, _) => unreachable!("the stock-order family has no {other:?} kernels"),
            }
        })
    }

    fn row_exact(
        &mut self,
        ty: RowExactWeight,
        weights: &[MlxBuffer],
        x: &MlxBuffer,
        y: &MlxBuffer,
        shape: RowExactMatvecShape,
        arm: &str,
        packing: Option<RowExactStockPacking>,
    ) -> Stats {
        let device = self.device.metal_device().to_owned();
        let class_b = arm == "class_b";
        self.time(|enc, reg, rep| {
            let w = &weights[rep % weights.len()];
            if class_b {
                dispatch_row_exact_matvec(enc, reg, &device, ty, w, x, y, &shape)
            } else if let Some(p) = packing {
                dispatch_row_exact_matvec_stock_packed(enc, reg, &device, ty, w, x, y, &shape, &p)
            } else {
                dispatch_row_exact_matvec_stock(enc, reg, &device, ty, w, x, y, &shape)
            }
            .expect("row-exact dispatch");
        })
    }

    fn run(&mut self, sweep: bool) {
        for (op, ty, groups, n, k) in CELLS {
            let tag = match ty {
                RowExactWeight::F16 => "f16",
                RowExactWeight::Q8_0 => "q8_0",
                other => unreachable!("the stock-order family has no {other:?} kernels"),
            };
            if !self.weights.iter().any(|w| w == tag) {
                continue;
            }
            let bytes = ty.row_bytes(k as u32) as usize * n * groups;
            let warm_modes: &[bool] = if bytes > COLD_BYTES {
                &[false]
            } else {
                &[true, false]
            };
            for &warm in warm_modes {
                let copies = copies_for(bytes, warm);
                let weights: Vec<MlxBuffer> = (0..copies)
                    .map(|c| weight(&self.device, ty, n * groups, k, c as u64 + 1))
                    .collect();
                let x = f32_buffer(&self.device, 8 * groups * k, 7);
                let y = self
                    .device
                    .alloc_buffer(8 * groups * n * 4, DType::F32, vec![8 * groups * n])
                    .expect("y");
                let state = if warm { "warm" } else { "cold" };
                for m in 1..=4usize {
                    let shape = if groups == 1 {
                        RowExactMatvecShape::contiguous(m as u32, n as u32, k as u32)
                    } else {
                        RowExactMatvecShape::grouped_token_major(
                            m as u32,
                            groups as u32,
                            n as u32,
                            k as u32,
                        )
                    };
                    if sweep {
                        if m != 1 && m != 4 {
                            continue;
                        }
                        for p in sweep_packings(ty) {
                            let t =
                                self.row_exact(ty, &weights, &x, &y, shape, "stock_order", Some(p));
                            println!(
                                "sweep {tag} {op:16} m={m} {state} {p:?}: {:8.2} us",
                                t.median
                            );
                            self.cell(serde_json::json!({"family": "sweep", "weight": tag, "op": op,
                                "groups": groups, "n": n, "k": k, "m": m, "warm": warm,
                                "packing": format!("{p:?}"), "us": t.median, "p10_us": t.p10, "p90_us": t.p90}));
                        }
                        continue;
                    }
                    let stock = self.stock(ty, &weights, &x, &y, m, groups, n, k);
                    let class_b = self.row_exact(ty, &weights, &x, &y, shape, "class_b", None);
                    let stock_order =
                        self.row_exact(ty, &weights, &x, &y, shape, "stock_order", None);
                    let packing = RowExactStockPacking::auto(ty, &shape);
                    println!(
                        "{tag:4} {op:16} {groups}x[{n:6} x {k:5}] m={m} {state}: stock {:8.2}  class_b {:8.2}  stock_order {:8.2} us ({packing:?})",
                        stock.median, class_b.median, stock_order.median
                    );
                    self.cell(serde_json::json!({"family": "arms", "weight": tag, "op": op,
                        "groups": groups, "n": n, "k": k, "m": m, "warm": warm, "copies": copies,
                        "weight_bytes": bytes, "packing": format!("{packing:?}"),
                        "stock_us": stock.median, "stock_p10_us": stock.p10, "stock_p90_us": stock.p90,
                        "class_b_us": class_b.median, "class_b_p10_us": class_b.p10, "class_b_p90_us": class_b.p90,
                        "stock_order_us": stock_order.median, "stock_order_p10_us": stock_order.p10,
                        "stock_order_p90_us": stock_order.p90}));
                }
            }
        }
    }
}

fn main() {
    let out_path: std::path::PathBuf = std::env::var_os("ROW_EXACT_STOCK_BENCH_OUT")
        .map(Into::into)
        .unwrap_or_else(|| "target/row_exact_matvec_stock/cells.jsonl".into());
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("out dir");
    }
    let families: Vec<String> = std::env::var("ROW_EXACT_STOCK_BENCH_FAMILIES")
        .unwrap_or_else(|_| "arms".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let mut bench = Bench {
        device: MlxDevice::new().expect("device"),
        registry: KernelRegistry::new(),
        out: std::fs::File::create(&out_path).expect("cells file"),
        weights: std::env::var("ROW_EXACT_STOCK_BENCH_WEIGHTS")
            .unwrap_or_else(|_| "f16,q8_0".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect(),
    };
    bench.spin_up();
    if families.iter().any(|f| f == "sweep") {
        bench.run(true);
    }
    if families.iter().any(|f| f == "arms") {
        bench.run(false);
    }
    println!("cells: {}", out_path.display());
}
