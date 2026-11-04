//! Q4-GEMMA-DECODE M0 microbenchmark (ADR-063 design §7).
//!
//! Model-free. Needs the root's GPU lane and a quiet host (HOST-LOCK held by
//! the runner). Timings come from command-buffer `GPUStartTime`/`GPUEndTime`
//! with 30 dispatches per command buffer (one per "layer"), so the fixed
//! command-buffer cost is amortized. Every cell cycles 30 distinct weight
//! buffers (2 for the 605.6 MB lm head) so its working set exceeds the system
//! cache, alternates two activation sets, runs `WARMUP` untimed and `ITERS`
//! timed command buffers, and writes one JSON receipt line.
//!
//! Environment:
//! - `GEMMA_DECODE_BENCH_OUT` — receipt directory (default
//!   `target/gemma_decode_w4`); `cells.jsonl` and `identity.json` go there.
//! - `GEMMA_DECODE_BENCH_FAMILIES` — comma list of `roofline`, `d1`,
//!   `lmhead`, `d2`, `e1` (default: all).
//! - `GEMMA_DECODE_BENCH_V2_OVERLAP` — file with the hf2q `[ROUTING_OVERLAP]`
//!   line from the V2 run; adds the empirical E1 patterns (`v2_u*`).
//! - `GEMMA_DECODE_BENCH_ITERS` (default 50, minimum 50) and
//!   `GEMMA_DECODE_BENCH_WARMUP` (default 10).
//! - `HF2Q_HOST_LOCK_HOLDER` — recorded verbatim (the runner's HOST-LOCK name).
//!
//! Run (precompiled metallib, the production default):
//!   cargo bench --bench bench_gemma_decode_w4

#![allow(clippy::expect_used, clippy::too_many_arguments)]

use std::io::Write;
use std::process::Command;

use mlx_native::{
    quantized_matmul_ggml_dense_mn_explicit, quantized_matmul_ggml_row_tiled_with_geometry,
    quantized_matmul_ggml_with_policy, quantized_matmul_id_ggml_mv_grouped_with_geometry,
    quantized_matmul_id_ggml_mv_with_policy, row_tiled_geometry, DType,
    GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlRoutingPolicy, GgmlType,
    KernelRegistry, MlxBuffer, MlxDevice, RowTiledGeometry,
};

const LAYERS: usize = 30;

struct Config {
    out: std::path::PathBuf,
    families: Vec<String>,
    iters: usize,
    warmup: usize,
}

impl Config {
    fn from_env() -> Self {
        let out = std::env::var_os("GEMMA_DECODE_BENCH_OUT")
            .map(Into::into)
            .unwrap_or_else(|| "target/gemma_decode_w4".into());
        let families = std::env::var("GEMMA_DECODE_BENCH_FAMILIES")
            .unwrap_or_else(|_| "roofline,d1,lmhead,d2,e1".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let iters = std::env::var("GEMMA_DECODE_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50usize)
            .max(50);
        let warmup = std::env::var("GEMMA_DECODE_BENCH_WARMUP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        Self {
            out,
            families,
            iters,
            warmup,
        }
    }

    fn wants(&self, family: &str) -> bool {
        self.families.iter().any(|f| f == family)
    }
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

fn weight_buffer(device: &MlxDevice, ty: GgmlType, n: usize, k: usize, rng: &mut Rng) -> MlxBuffer {
    let block_bytes = ty.block_bytes() as usize;
    let blocks = n * (k / ty.block_values() as usize);
    let len = blocks * block_bytes;
    let mut buffer = device.alloc_buffer(len, DType::U8, vec![len]).expect("weight");
    let bytes = buffer.as_mut_slice::<u8>().expect("weight slice");
    for chunk in bytes.chunks_mut(8) {
        let v = rng.next().to_le_bytes();
        chunk.copy_from_slice(&v[..chunk.len()]);
    }
    // Finite small half scales at each block's scale offset(s).
    let scale = half::f16::from_f32(0.002).to_bits().to_le_bytes();
    let offsets: &[usize] = match ty {
        GgmlType::Q4_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        _ => &[0],
    };
    for b in 0..blocks {
        for &o in offsets {
            bytes[b * block_bytes + o..b * block_bytes + o + 2].copy_from_slice(&scale);
        }
    }
    buffer
}

fn f32_buffer(device: &MlxDevice, len: usize, rng: &mut Rng) -> MlxBuffer {
    let mut buffer = device.alloc_buffer(len * 4, DType::F32, vec![len]).expect("f32");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32 slice")
        .iter_mut()
        .for_each(|v| *v = (rng.next() >> 40) as f32 / (1u32 << 24) as f32 - 0.5);
    buffer
}

#[derive(Clone, Copy)]
enum Arm {
    /// Stock route under the default policy.
    Stock,
    /// D1 at an explicit geometry.
    RowTiled(RowTiledGeometry),
    /// D2.
    DenseMn,
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Stock => "stock".into(),
            Arm::RowTiled(g) => format!("d1_rt{}_nsg{}", g.r0, g.nsg),
            Arm::DenseMn => "d2_mn".into(),
        }
    }
}

struct Stats {
    median_ns: f64,
    p10_ns: f64,
    p90_ns: f64,
}

fn stats(mut per_dispatch_ns: Vec<f64>) -> Stats {
    per_dispatch_ns.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| per_dispatch_ns[((per_dispatch_ns.len() - 1) as f64 * q).round() as usize];
    Stats {
        median_ns: at(0.5),
        p10_ns: at(0.1),
        p90_ns: at(0.9),
    }
}

fn tiles(m: usize) -> usize {
    match m {
        0..=4 => 1,
        _ => 2,
    }
}

/// Weight reads issued per call by one arm (logical / unique).
fn weight_reads(arm: Arm, ty: GgmlType, m: usize) -> usize {
    match (arm, ty) {
        (Arm::Stock, GgmlType::Q5_0 | GgmlType::Q8_0) => m,
        (Arm::Stock, _) if m == 1 => 1,
        // Stock K-quant mN tiles 5 as [5] and 6..8 as two tiles.
        (Arm::Stock, _) => {
            if m <= 5 {
                1
            } else {
                2
            }
        }
        _ => tiles(m),
    }
}

struct Bench {
    device: MlxDevice,
    registry: KernelRegistry,
    cfg: Config,
    receipts: std::fs::File,
}

impl Bench {
    fn time_cell(
        &mut self,
        family: &str,
        shape: &str,
        ty: GgmlType,
        n: usize,
        k: usize,
        m: usize,
        weights: &[MlxBuffer],
        inputs: &[MlxBuffer; 2],
        output: &MlxBuffer,
        arm: Arm,
    ) {
        let params = GgmlQuantizedMatmulParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            ggml_type: ty,
        };
        let stock = GgmlRoutingPolicy::default();
        let mut samples = Vec::with_capacity(self.cfg.iters);
        for iter in 0..self.cfg.warmup + self.cfg.iters {
            let input = &inputs[iter % 2];
            let mut encoder = self.device.command_encoder().expect("encoder");
            for layer in 0..LAYERS {
                let weight = &weights[(iter * LAYERS + layer) % weights.len()];
                let result = match arm {
                    Arm::Stock => quantized_matmul_ggml_with_policy(
                        &mut encoder,
                        &mut self.registry,
                        &self.device,
                        input,
                        weight,
                        output,
                        &params,
                        &stock,
                    ),
                    Arm::RowTiled(geometry) => quantized_matmul_ggml_row_tiled_with_geometry(
                        &mut encoder,
                        &mut self.registry,
                        &self.device,
                        input,
                        weight,
                        output,
                        &params,
                        geometry,
                    ),
                    Arm::DenseMn => quantized_matmul_ggml_dense_mn_explicit(
                        &mut encoder,
                        &mut self.registry,
                        &self.device,
                        input,
                        weight,
                        output,
                        &params,
                    ),
                };
                result.expect("dispatch");
            }
            let (start, end) = encoder.commit_wait_with_gpu_time().expect("GPU");
            if iter >= self.cfg.warmup {
                samples.push((end - start) * 1e9 / LAYERS as f64);
            }
        }
        let s = stats(samples);
        let unique = weights[0].byte_len() as f64;
        let logical = unique * weight_reads(arm, ty, m) as f64;
        let element_columns = (n * k * m) as f64;
        let line = serde_json::json!({
            "schema": "hf2q.q4-gemma-decode.m0.cell.v1",
            "family": family,
            "shape": shape,
            "type": format!("{ty:?}"),
            "n": n, "k": k, "m": m,
            "arm": arm.label(),
            "geometry": match arm {
                Arm::RowTiled(g) => serde_json::json!({"r0": g.r0, "nsg": g.nsg}),
                _ => serde_json::Value::Null,
            },
            "weight_buffers_cycled": weights.len(),
            "dispatches_per_command_buffer": LAYERS,
            "iters": self.cfg.iters,
            "warmup": self.cfg.warmup,
            "median_ns": s.median_ns,
            "p10_ns": s.p10_ns,
            "p90_ns": s.p90_ns,
            "unique_weight_bytes": unique,
            "logical_weight_bytes": logical,
            "weight_gb_per_s_logical": logical / s.median_ns,
            "weight_gb_per_s_unique": unique / s.median_ns,
            "element_columns_per_s": element_columns / (s.median_ns * 1e-9),
        });
        writeln!(self.receipts, "{line}").expect("receipt");
        println!(
            "{family:7} {shape:14} {ty:?} m={m} {:14} median {:9.1} µs  p10 {:9.1}  p90 {:9.1}  {:6.1} GB/s  {:5.2} T el-col/s",
            arm.label(),
            s.median_ns / 1e3,
            s.p10_ns / 1e3,
            s.p90_ns / 1e3,
            logical / s.median_ns,
            element_columns / s.median_ns / 1e3,
        );
    }

    fn dense_family(
        &mut self,
        family: &str,
        shapes: &[(&str, GgmlType, usize, usize)],
        widths: &[usize],
        buffers: usize,
        arms_for: &dyn Fn(GgmlType, usize) -> Vec<Arm>,
    ) {
        let mut rng = Rng(0x6d30);
        for &(shape, ty, n, k) in shapes {
            let weights: Vec<MlxBuffer> = (0..buffers)
                .map(|_| weight_buffer(&self.device, ty, n, k, &mut rng))
                .collect();
            let max_m = widths.iter().copied().max().unwrap_or(1);
            let inputs = [
                f32_buffer(&self.device, max_m * k, &mut rng),
                f32_buffer(&self.device, max_m * k, &mut rng),
            ];
            let output = f32_buffer(&self.device, max_m * n, &mut rng);
            for &m in widths {
                for arm in arms_for(ty, m) {
                    self.time_cell(family, shape, ty, n, k, m, &weights, &inputs, &output, arm);
                }
            }
        }
    }

    fn roofline(&mut self) {
        let bytes: u64 = 2 << 30;
        let src = self
            .device
            .alloc_buffer(bytes as usize, DType::U8, vec![bytes as usize])
            .expect("roofline src");
        let dst = self
            .device
            .alloc_buffer(bytes as usize, DType::U8, vec![bytes as usize])
            .expect("roofline dst");
        let mut samples = Vec::new();
        for iter in 0..self.cfg.warmup + self.cfg.iters {
            let mut encoder = self.device.command_encoder().expect("encoder");
            encoder.blit_copy_bytes(&src, 0, &dst, 0, bytes).expect("blit");
            let (start, end) = encoder.commit_wait_with_gpu_time().expect("GPU");
            if iter >= self.cfg.warmup {
                samples.push((end - start) * 1e9);
            }
        }
        let s = stats(samples);
        let line = serde_json::json!({
            "schema": "hf2q.q4-gemma-decode.m0.roofline.v1",
            "kind": "blit_copy",
            "bytes_read": bytes,
            "bytes_written": bytes,
            "median_ns": s.median_ns,
            "p10_ns": s.p10_ns,
            "p90_ns": s.p90_ns,
            "read_plus_write_gb_per_s": (2 * bytes) as f64 / s.median_ns,
        });
        writeln!(self.receipts, "{line}").expect("receipt");
        println!(
            "roofline blit 2 GiB: median {:.2} ms, {:.1} GB/s read+write",
            s.median_ns / 1e6,
            (2 * bytes) as f64 / s.median_ns
        );
    }
}

fn command_output(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn row_tiled_arms(ty: GgmlType, n: usize, k: usize) -> Vec<Arm> {
    let mut arms = vec![Arm::Stock];
    for r0 in RowTiledGeometry::R0_CELLS {
        for nsg in RowTiledGeometry::NSG_CELLS {
            arms.push(Arm::RowTiled(RowTiledGeometry { r0, nsg }));
        }
    }
    debug_assert!(row_tiled_geometry(ty, n as u32, k as u32).is_some());
    arms
}

/// One routing pattern: `[n_tokens][8]` expert ids, distinct within a token.
struct Pattern {
    name: String,
    ids: Vec<u32>,
    /// Weight of this pattern in the V2 empirical mix (1 for synthetic).
    weight: f64,
}

const N_EXPERTS: u32 = 128;
const TOP_K: usize = 8;

/// Exactly `u` distinct experts over `n_tokens` tokens (8 ≤ u ≤ 8·n).
fn pattern_with_u(n_tokens: usize, u: usize, rng: &mut Rng) -> Vec<u32> {
    let u = u.clamp(TOP_K, (TOP_K * n_tokens).min(N_EXPERTS as usize));
    let mut pool: Vec<u32> = (0..N_EXPERTS).collect();
    for i in (1..pool.len()).rev() {
        pool.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    pool.truncate(u);
    // Deal the pool round-robin so every member is used, then top each
    // token up with distinct pool members.
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n_tokens];
    for (i, &e) in pool.iter().enumerate() {
        let t = i % n_tokens;
        if rows[t].len() < TOP_K {
            rows[t].push(e);
        }
    }
    for row in &mut rows {
        let mut j = 0;
        while row.len() < TOP_K {
            let e = pool[(rng.next() as usize + j) % pool.len()];
            if !row.contains(&e) {
                row.push(e);
            }
            j += 1;
        }
    }
    rows.concat()
}

fn synthetic_patterns(n_tokens: usize, rng: &mut Rng) -> Vec<Pattern> {
    let distinct = pattern_with_u(n_tokens, TOP_K * n_tokens, rng);
    let shared: Vec<u32> = (0..n_tokens).flat_map(|_| 0..TOP_K as u32).collect();
    // Skewed: expert 0 in every row, the rest distinct.
    let mut next = 1u32;
    let skewed: Vec<u32> = (0..n_tokens)
        .flat_map(|_| {
            let mut row = vec![0u32];
            for _ in 1..TOP_K {
                row.push(next % N_EXPERTS);
                next += 1;
            }
            row
        })
        .collect();
    vec![
        Pattern {
            name: format!("distinct_u{}", (TOP_K * n_tokens).min(N_EXPERTS as usize)),
            ids: distinct,
            weight: 1.0,
        },
        Pattern { name: "shared_u8".into(), ids: shared, weight: 1.0 },
        Pattern { name: "skewed".into(), ids: skewed, weight: 1.0 },
    ]
}

/// V2 empirical patterns: one pattern per observed U at this width, weighted
/// by its count summed over layers, from the `[ROUTING_OVERLAP]` line
/// (`GEMMA_DECODE_BENCH_V2_OVERLAP` names a file holding it).
fn v2_patterns(n_tokens: usize, rng: &mut Rng) -> Vec<Pattern> {
    let Some(path) = std::env::var_os("GEMMA_DECODE_BENCH_V2_OVERLAP") else {
        return Vec::new();
    };
    let text = std::fs::read_to_string(&path).expect("V2 overlap JSON");
    let json_text = text.trim().trim_start_matches("[ROUTING_OVERLAP]").trim();
    let report: serde_json::Value = serde_json::from_str(json_text).expect("V2 overlap JSON parse");
    let mut counts = std::collections::BTreeMap::<usize, f64>::new();
    for entry in report["distinct_experts"].as_array().into_iter().flatten() {
        if entry["width"].as_u64() != Some(n_tokens as u64) {
            continue;
        }
        for (u, count) in entry["u_hist"].as_object().into_iter().flatten() {
            *counts.entry(u.parse().expect("U key")).or_default() += count.as_f64().unwrap_or(0.0);
        }
    }
    let total: f64 = counts.values().sum();
    counts
        .into_iter()
        .map(|(u, count)| Pattern {
            name: format!("v2_u{u}"),
            ids: pattern_with_u(n_tokens, u, rng),
            weight: count / total,
        })
        .collect()
}

#[derive(Clone, Copy)]
enum ExpertArm {
    Stock,
    Grouped(u32),
}

impl Bench {
    fn expert_family(&mut self) {
        let mut rng = Rng(0xe1);
        // gate_up: token rows broadcast to top-8 pairs; down: one row per pair.
        let shapes = [
            ("gate_up_q4k", GgmlType::Q4_K, 1408usize, 2816usize, false),
            ("down_q5_0", GgmlType::Q5_0, 2816, 704, true),
            ("down_q8_0", GgmlType::Q8_0, 2816, 704, true),
        ];
        for (shape, ty, n, k, per_pair) in shapes {
            let row_bytes = k / ty.block_values() as usize * ty.block_bytes() as usize;
            let stride = (n * row_bytes) as u64;
            // Four distinct stacks (each > the system cache) cycled per layer.
            let stacks: Vec<MlxBuffer> = (0..4)
                .map(|_| weight_buffer(&self.device, ty, n * N_EXPERTS as usize, k, &mut rng))
                .collect();
            for n_tokens in [4usize, 8, 16] {
                let pairs = n_tokens * TOP_K;
                let rows = if per_pair { pairs } else { n_tokens };
                let inputs = [
                    f32_buffer(&self.device, rows * k, &mut rng),
                    f32_buffer(&self.device, rows * k, &mut rng),
                ];
                let output = f32_buffer(&self.device, pairs * n, &mut rng);
                let mut patterns = synthetic_patterns(n_tokens, &mut rng);
                patterns.extend(v2_patterns(n_tokens, &mut rng));
                for pattern in patterns {
                    let mut ids = self
                        .device
                        .alloc_buffer(pairs * 4, DType::U32, vec![pairs])
                        .expect("ids");
                    ids.as_mut_slice::<u32>().expect("ids").copy_from_slice(&pattern.ids);
                    let unique_experts = {
                        let mut v = pattern.ids.clone();
                        v.sort_unstable();
                        v.dedup();
                        v.len()
                    };
                    let params = GgmlQuantizedMatmulIdParams {
                        n_tokens: if per_pair { pairs as u32 } else { n_tokens as u32 },
                        top_k: if per_pair { 1 } else { TOP_K as u32 },
                        n: n as u32,
                        k: k as u32,
                        n_experts: N_EXPERTS,
                        expert_stride: stride,
                        ggml_type: ty,
                    };
                    let mut arms = vec![ExpertArm::Stock];
                    let cells: &[u32] = if ty == GgmlType::Q4_K { &[1, 2, 4] } else { &[4] };
                    arms.extend(cells.iter().map(|&r0| ExpertArm::Grouped(r0)));
                    for arm in arms {
                        let s = self.time_expert(&stacks, &inputs, &ids, &output, &params, arm);
                        let expert_bytes = stride as f64;
                        let logical = expert_bytes * pairs as f64;
                        let unique = expert_bytes * unique_experts as f64;
                        let label = match arm {
                            ExpertArm::Stock => "stock".to_string(),
                            ExpertArm::Grouped(r0) => format!("e1_grouped_rt{r0}"),
                        };
                        let line = serde_json::json!({
                            "schema": "hf2q.q4-gemma-decode.m0.cell.v1",
                            "family": "e1",
                            "shape": shape,
                            "type": format!("{ty:?}"),
                            "n": n, "k": k, "n_tokens": n_tokens, "top_k": TOP_K,
                            "pattern": pattern.name,
                            "pattern_weight": pattern.weight,
                            "distinct_experts": unique_experts,
                            "arm": label,
                            "weight_stacks_cycled": stacks.len(),
                            "dispatches_per_command_buffer": LAYERS,
                            "iters": self.cfg.iters,
                            "warmup": self.cfg.warmup,
                            "median_ns": s.median_ns,
                            "p10_ns": s.p10_ns,
                            "p90_ns": s.p90_ns,
                            "logical_weight_bytes_stock": logical,
                            "unique_weight_bytes": unique,
                            "weight_gb_per_s_logical": logical / s.median_ns,
                            "weight_gb_per_s_unique": unique / s.median_ns,
                            "element_columns_per_s": (n * k * pairs) as f64 / (s.median_ns * 1e-9),
                        });
                        writeln!(self.receipts, "{line}").expect("receipt");
                        println!(
                            "e1      {shape:14} n_tok={n_tokens:2} {:12} U={unique_experts:3} {label:16} median {:9.1} µs",
                            pattern.name,
                            s.median_ns / 1e3
                        );
                    }
                }
            }
        }
    }

    fn time_expert(
        &mut self,
        stacks: &[MlxBuffer],
        inputs: &[MlxBuffer; 2],
        ids: &MlxBuffer,
        output: &MlxBuffer,
        params: &GgmlQuantizedMatmulIdParams,
        arm: ExpertArm,
    ) -> Stats {
        let stock = GgmlRoutingPolicy::default();
        let mut samples = Vec::with_capacity(self.cfg.iters);
        for iter in 0..self.cfg.warmup + self.cfg.iters {
            let mut encoder = self.device.command_encoder().expect("encoder");
            for layer in 0..LAYERS {
                let weight = &stacks[(iter * LAYERS + layer) % stacks.len()];
                let input = &inputs[iter % 2];
                match arm {
                    ExpertArm::Stock => quantized_matmul_id_ggml_mv_with_policy(
                        &mut encoder,
                        &mut self.registry,
                        &self.device,
                        input,
                        weight,
                        ids,
                        output,
                        params,
                        &stock,
                    ),
                    ExpertArm::Grouped(r0) => quantized_matmul_id_ggml_mv_grouped_with_geometry(
                        &mut encoder,
                        &mut self.registry,
                        &self.device,
                        input,
                        weight,
                        ids,
                        output,
                        params,
                        r0,
                    ),
                }
                .expect("expert dispatch");
            }
            let (start, end) = encoder.commit_wait_with_gpu_time().expect("GPU");
            if iter >= self.cfg.warmup {
                samples.push((end - start) * 1e9 / LAYERS as f64);
            }
        }
        stats(samples)
    }
}

fn main() {
    let cfg = Config::from_env();
    std::fs::create_dir_all(&cfg.out).expect("bench output dir");
    let device = MlxDevice::new().expect("Metal device");
    let receipts = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out.join("cells.jsonl"))
        .expect("cells.jsonl");
    let mut bench = Bench {
        device,
        registry: KernelRegistry::new(),
        cfg,
        receipts,
    };

    if bench.cfg.wants("roofline") {
        bench.roofline();
    }
    let widths = [1usize, 2, 3, 4, 5, 6, 7, 8];
    if bench.cfg.wants("d1") {
        let shapes = [
            ("q_sliding", GgmlType::Q4_K, 4096, 2816),
            ("q_global", GgmlType::Q4_K, 8192, 2816),
            ("k_sliding", GgmlType::Q4_K, 2048, 2816),
            ("k_global", GgmlType::Q4_K, 1024, 2816),
            ("v_q4k", GgmlType::Q4_K, 2048, 2816),
            ("v_q6k", GgmlType::Q6_K, 2048, 2816),
            ("o_sliding", GgmlType::Q4_K, 2816, 4096),
            ("o_global", GgmlType::Q4_K, 2816, 8192),
            ("gate_up", GgmlType::Q4_K, 2112, 2816),
        ];
        // Each shape's own arms; stock at m = 1 is the m = 1 reference kernel.
        let table: Vec<_> = shapes.to_vec();
        for (shape, ty, n, k) in table {
            bench.dense_family("d1", &[(shape, ty, n, k)], &widths, LAYERS, &|ty, _| {
                row_tiled_arms(ty, n, k)
            });
        }
    }
    if bench.cfg.wants("lmhead") {
        bench.dense_family(
            "lmhead",
            &[("lm_head", GgmlType::Q6_K, 262_144, 2816)],
            &[1, 2, 3, 4, 8],
            2,
            &|ty, _| row_tiled_arms(ty, 262_144, 2816),
        );
    }
    if bench.cfg.wants("d2") {
        bench.dense_family(
            "d2",
            &[
                ("down_q5_0", GgmlType::Q5_0, 2816, 2112),
                ("down_q8_0", GgmlType::Q8_0, 2816, 2112),
            ],
            &widths,
            LAYERS,
            &|_, m| {
                if m >= 2 {
                    vec![Arm::Stock, Arm::DenseMn]
                } else {
                    vec![Arm::Stock]
                }
            },
        );
    }
    if bench.cfg.wants("e1") {
        bench.expert_family();
    }

    let identity = serde_json::json!({
        "schema": "hf2q.q4-gemma-decode.m0.identity.v1",
        "mlx_native_version": env!("CARGO_PKG_VERSION"),
        "mlx_native_commit": command_output("git", &["rev-parse", "HEAD"]),
        "embedded_metallib_sha256": bench
            .registry
            .pipeline_identity("kernel_mul_mv_q4_K_f32|700:i1|701:i1|702:i1")
            .ok()
            .and_then(|id| id.embedded_metallib_sha256),
        "metallib_env": std::env::var("MLX_PRECOMPILED_METALLIB").unwrap_or_else(|_| "default".into()),
        "host": command_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "hw_model": command_output("sysctl", &["-n", "hw.model"]),
        "macos_build": command_output("sw_vers", &["-buildVersion"]),
        "metal_compiler": command_output("xcrun", &["-sdk", "macosx", "metal", "--version"]),
        "host_lock_holder": std::env::var("HF2Q_HOST_LOCK_HOLDER").ok(),
        "iters": bench.cfg.iters,
        "warmup": bench.cfg.warmup,
        "families": bench.cfg.families,
    });
    std::fs::write(
        bench.cfg.out.join("identity.json"),
        serde_json::to_string_pretty(&identity).expect("identity json"),
    )
    .expect("identity.json");
}
