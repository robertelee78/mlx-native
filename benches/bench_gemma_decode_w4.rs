//! Q4-GEMMA-DECODE microbenchmark for the adopted levers (ADR-063 design §7).
//!
//! Model-free. Needs the root's GPU lane and a quiet host (HOST-LOCK held by
//! the runner). Timings come from command-buffer `GPUStartTime`/`GPUEndTime`
//! with 30 dispatches per command buffer (one per "layer"), so the fixed
//! command-buffer cost is amortized. Every cell cycles distinct weight
//! buffers (four 128-expert stacks for `f2`, 30 KV caches for `attn`) so its
//! working set exceeds the system cache, alternates two activation sets, runs
//! `WARMUP` untimed and `ITERS` timed command buffers, and writes one JSON
//! receipt line.
//!
//! `f2`: stock gate_up `mv_id` + `moe_swiglu_batch` against the fused F2
//! kernel (R0 = 4) on the Gemma 4 gate_up shape at n_tokens 4, 8 and 16.
//!
//! Environment:
//! - `GEMMA_DECODE_BENCH_OUT` — receipt directory (default
//!   `target/gemma_decode_w4`); `cells.jsonl` and `identity.json` go there.
//! - `GEMMA_DECODE_BENCH_FAMILIES` — comma list of `roofline`, `f2`, `attn`
//!   (default: all).
//! - `GEMMA_DECODE_BENCH_V2_OVERLAP` — file with the hf2q `[ROUTING_OVERLAP]`
//!   line from the V2 run; adds the empirical F2 patterns (`v2_u*`) and the
//!   V2 KV-length histogram steps of `attn` (`v2_le*`, weighted by count).
//! - `GEMMA_DECODE_BENCH_ITERS` (default 50, minimum 50) and
//!   `GEMMA_DECODE_BENCH_WARMUP` (default 10).
//! - `HF2Q_HOST_LOCK_HOLDER` — recorded verbatim (the runner's HOST-LOCK name).
//!
//! `attn` (the GQA-grouped vec-hybrid flash; dropped for decode by M0, kept
//! as the warm-prefill primitive): d256 sliding (16 q / 8 KV heads, ring
//! 1024) and d512 global (16 q / 2 KV heads), n_q ∈ {1,4,8,16},
//! uniform-length steps (stock batched vs GQA at every legal G) and
//! per-slot-bucket steps (stock per-slot fallback vs GQA at every legal G);
//! reduce dispatches included; 30 distinct KV caches cycled.
//!
//! Run (precompiled metallib, the production default):
//!   cargo bench --bench bench_gemma_decode_w4

#![allow(clippy::expect_used, clippy::too_many_arguments)]

use std::io::Write;
use std::process::Command;

use mlx_native::ops::flash_attn_vec_hybrid::{
    flash_attn_vec_hybrid, flash_attn_vec_hybrid_batched, flash_attn_vec_hybrid_batched_gqa,
    flash_attn_vec_hybrid_batched_gqa_group_is_legal, HybridSlotBucket,
    FLASH_ATTN_VEC_HYBRID_GQA_GROUPS,
};
use mlx_native::ops::flash_attn_vec_tq_hb::{compute_nsg, FlashAttnVecTqHbParams};
use mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode;
use mlx_native::{
    quantized_matmul_id_ggml_mv_grouped_geglu, quantized_matmul_id_ggml_mv_with_policy, DType,
    GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy, GgmlType, KernelRegistry, MlxBuffer,
    MlxDevice,
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
            .unwrap_or_else(|_| "roofline,f2,attn".into())
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

struct Bench {
    device: MlxDevice,
    registry: KernelRegistry,
    cfg: Config,
    receipts: std::fs::File,
}

impl Bench {
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
    /// Stock gate_up then stock `moe_swiglu_batch` (the route F2 replaces).
    StockSwiglu,
    /// F2 fused gate_up + GELU·mul at its frozen geometry (R0 = 4).
    Geglu,
}

impl ExpertArm {
    fn label(self) -> &'static str {
        match self {
            ExpertArm::StockSwiglu => "stock+swiglu",
            ExpertArm::Geglu => "f2_geglu_rt4",
        }
    }
}

impl Bench {
    /// F2: Gemma 4 gate_up (Q4_K, N = 1408, K = 2816, top-8 of 128) at
    /// n_tokens 4, 8 and 16 over the synthetic and V2 patterns.
    fn f2_family(&mut self) {
        let mut rng = Rng(0xe1);
        let (shape, ty, n, k) = ("gate_up_q4k", GgmlType::Q4_K, 1408usize, 2816usize);
        let row_bytes = k / ty.block_values() as usize * ty.block_bytes() as usize;
        let stride = (n * row_bytes) as u64;
        // Four distinct stacks (each > the system cache) cycled per layer.
        let stacks: Vec<MlxBuffer> = (0..4)
            .map(|_| weight_buffer(&self.device, ty, n * N_EXPERTS as usize, k, &mut rng))
            .collect();
        for n_tokens in [4usize, 8, 16] {
            let pairs = n_tokens * TOP_K;
            let inputs = [
                f32_buffer(&self.device, n_tokens * k, &mut rng),
                f32_buffer(&self.device, n_tokens * k, &mut rng),
            ];
            let gate_up = f32_buffer(&self.device, pairs * n, &mut rng);
            let output = f32_buffer(&self.device, pairs * n / 2, &mut rng);
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
                    n_tokens: n_tokens as u32,
                    top_k: TOP_K as u32,
                    n: n as u32,
                    k: k as u32,
                    n_experts: N_EXPERTS,
                    expert_stride: stride,
                    ggml_type: ty,
                };
                for arm in [ExpertArm::StockSwiglu, ExpertArm::Geglu] {
                    let s = self.time_expert(&stacks, &inputs, &ids, &gate_up, &output, &params, arm);
                    let expert_bytes = stride as f64;
                    let logical = expert_bytes * pairs as f64;
                    let unique = expert_bytes * unique_experts as f64;
                    let line = serde_json::json!({
                        "schema": "hf2q.q4-gemma-decode.m0.cell.v1",
                        "family": "f2",
                        "shape": shape,
                        "type": format!("{ty:?}"),
                        "n": n, "k": k, "n_tokens": n_tokens, "top_k": TOP_K,
                        "pattern": pattern.name,
                        "pattern_weight": pattern.weight,
                        "distinct_experts": unique_experts,
                        "arm": arm.label(),
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
                        "f2      {shape:14} n_tok={n_tokens:2} {:12} U={unique_experts:3} {:16} median {:9.1} µs",
                        pattern.name,
                        arm.label(),
                        s.median_ns / 1e3
                    );
                }
            }
        }
    }

    fn time_expert(
        &mut self,
        stacks: &[MlxBuffer],
        inputs: &[MlxBuffer; 2],
        ids: &MlxBuffer,
        gate_up: &MlxBuffer,
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
                    ExpertArm::StockSwiglu => quantized_matmul_id_ggml_mv_with_policy(
                        &mut encoder,
                        &mut self.registry,
                        &self.device,
                        input,
                        weight,
                        ids,
                        gate_up,
                        params,
                        &stock,
                    )
                    .and_then(|()| {
                        encoder.memory_barrier();
                        moe_swiglu_batch_encode(
                            &mut encoder,
                            &mut self.registry,
                            self.device.metal_device(),
                            gate_up,
                            output,
                            params.n as usize / 2,
                            (params.n_tokens * params.top_k) as usize,
                        )
                    }),
                    ExpertArm::Geglu => quantized_matmul_id_ggml_mv_grouped_geglu(
                        &mut encoder,
                        &mut self.registry,
                        &self.device,
                        input,
                        weight,
                        ids,
                        output,
                        params,
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

/// One attention layer type of the `attn` family.
struct AttnShape {
    name: &'static str,
    head_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    sliding: bool,
}

const ATTN_SLOTS: usize = 16;
const SLIDING_WINDOW: u32 = 1024;

#[derive(Clone, Copy)]
enum AttnArm {
    /// Stock `flash_attn_vec_hybrid_batched` (uniform bucket).
    StockBatched,
    /// Stock per-slot `flash_attn_vec_hybrid` fallback (mixed buckets).
    StockPerSlot,
    /// `flash_attn_vec_hybrid_batched_gqa` at group G.
    Gqa(u32),
}

impl AttnArm {
    fn label(self) -> String {
        match self {
            AttnArm::StockBatched => "stock_batched".into(),
            AttnArm::StockPerSlot => "stock_per_slot".into(),
            AttnArm::Gqa(g) => format!("gqa_g{g}"),
        }
    }
}

struct KvCache {
    k: MlxBuffer,
    v: MlxBuffer,
    norms: MlxBuffer,
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("u32");
    buffer.as_mut_slice::<u32>().expect("u32 slice").copy_from_slice(values);
    buffer
}

fn random_bytes(buffer: &mut MlxBuffer, rng: &mut Rng, f16: bool) {
    let bytes = buffer.as_mut_slice::<u8>().expect("byte slice");
    for chunk in bytes.chunks_mut(8) {
        let mut v = rng.next();
        if f16 {
            // Finite halves in ±[0.125, 0.25).
            v = (v & 0x83ff_83ff_83ff_83ff) | 0x3000_3000_3000_3000;
        }
        chunk.copy_from_slice(&v.to_le_bytes()[..chunk.len()]);
    }
}

/// V2 KV-length histogram of one layer type: (representative length, weight).
fn v2_kv_lengths(layer_type: &str) -> Vec<(u32, f64)> {
    let Some(path) = std::env::var_os("GEMMA_DECODE_BENCH_V2_OVERLAP") else {
        return Vec::new();
    };
    let text = std::fs::read_to_string(&path).expect("V2 overlap JSON");
    let json_text = text.trim().trim_start_matches("[ROUTING_OVERLAP]").trim();
    let report: serde_json::Value = serde_json::from_str(json_text).expect("V2 overlap JSON parse");
    let hist: Vec<(u32, f64)> = report[layer_type]["kv_len_pow2"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(key, count)| {
            let len = key.trim_start_matches("le").parse().expect("kv_len_pow2 key");
            (len, count.as_f64().unwrap_or(0.0))
        })
        .collect();
    let total: f64 = hist.iter().map(|(_, c)| c).sum();
    hist.into_iter().map(|(l, c)| (l, c / total.max(1.0))).collect()
}

impl Bench {
    fn attn_family(&mut self) {
        let shapes = [
            AttnShape { name: "d256_sliding", head_dim: 256, num_heads: 16, num_kv_heads: 8, sliding: true },
            AttnShape { name: "d512_global", head_dim: 512, num_heads: 16, num_kv_heads: 2, sliding: false },
        ];
        let mut rng = Rng(0xa77);
        for shape in &shapes {
            // (step name, per-slot seq_pos+1 pattern, weight)
            let mut steps: Vec<(String, Vec<u32>, Option<f64>)> = [512u32, 1024, 2048, 4096]
                .iter()
                .map(|&l| (format!("uniform_{l}"), vec![l], None))
                .collect();
            let layer_type = if shape.sliding { "sliding" } else { "global" };
            for (l, w) in v2_kv_lengths(layer_type) {
                steps.push((format!("v2_le{l}"), vec![l], Some(w)));
            }
            steps.push(("mixed_300_700_1100_3000".into(), vec![300, 700, 1100, 3000], None));
            let max_len = steps.iter().flat_map(|s| s.1.iter().copied()).max().unwrap_or(1);
            let cap = if shape.sliding { SLIDING_WINDOW } else { max_len };
            let (hd, nkv) = (shape.head_dim as usize, shape.num_kv_heads as usize);
            let npp = if hd == 512 { 2 } else { 1 };
            let rows = ATTN_SLOTS * nkv * cap as usize;
            // `GEMMA_W4_KV_CACHES=N` (default LAYERS = 30): distinct KV caches
            // cycled; 1 keeps one cache hot (the peer's perf mode reuses its
            // tensors), for the cache-temperature comparison.
            let n_caches = std::env::var("GEMMA_W4_KV_CACHES")
                .ok()
                .map(|v| v.parse::<usize>().expect("GEMMA_W4_KV_CACHES=<positive integer>"))
                .unwrap_or(LAYERS)
                .max(1);
            let caches: Vec<KvCache> = (0..n_caches)
                .map(|_| {
                    let mut k = self.device.alloc_buffer(rows * hd * 2, DType::F16, vec![rows * hd]).expect("k");
                    random_bytes(&mut k, &mut rng, true);
                    let mut v = self.device.alloc_buffer(rows * hd, DType::U8, vec![rows * hd]).expect("v");
                    random_bytes(&mut v, &mut rng, false);
                    KvCache { k, v, norms: f32_buffer(&self.device, rows * npp, &mut rng) }
                })
                .collect();
            let q_len = ATTN_SLOTS * shape.num_heads as usize * hd;
            let inputs = [f32_buffer(&self.device, q_len, &mut rng), f32_buffer(&self.device, q_len, &mut rng)];
            let output = f32_buffer(&self.device, q_len, &mut rng);
            let tmp = f32_buffer(&self.device, ATTN_SLOTS * shape.num_heads as usize * (hd + 2) * 32, &mut rng);
            let slot_ids: Vec<u32> = (0..ATTN_SLOTS as u32).collect();
            let slot_id_arr = u32_buffer(&self.device, &slot_ids);
            for n_q in [1usize, 4, 8, 16] {
                for (step, pattern, weight) in &steps {
                    let seq_lens: Vec<u32> = (0..n_q).map(|i| pattern[i % pattern.len()]).collect();
                    let kv_lens: Vec<u32> =
                        seq_lens.iter().map(|&l| if shape.sliding { l.min(cap) } else { l }).collect();
                    let buckets: Vec<HybridSlotBucket> =
                        kv_lens.iter().map(|&l| HybridSlotBucket::for_kv_seq_len(l)).collect();
                    let uniform = buckets.iter().all(|b| *b == buckets[0]);
                    if pattern.len() > 1 && uniform {
                        continue; // a "mixed" step that collapses to one bucket
                    }
                    let nsg_max = buckets.iter().map(|b| b.nsg).max().unwrap_or(1);
                    let mut arms = vec![if uniform { AttnArm::StockBatched } else { AttnArm::StockPerSlot }];
                    arms.extend(FLASH_ATTN_VEC_HYBRID_GQA_GROUPS.iter().copied().filter(|&g| {
                        flash_attn_vec_hybrid_batched_gqa_group_is_legal(
                            shape.num_heads,
                            shape.num_kv_heads,
                            shape.head_dim,
                            g,
                            nsg_max,
                        )
                    }).map(AttnArm::Gqa));
                    let seq_pos: Vec<u32> = seq_lens.iter().map(|l| l - 1).collect();
                    let seq_pos_arr = u32_buffer(&self.device, &seq_pos);
                    let bytes_per_row = (hd * 2 + hd + npp * 4) as f64;
                    let kv_rows: f64 = kv_lens.iter().map(|&l| l as f64).sum();
                    let logical = kv_rows * shape.num_heads as f64 * bytes_per_row;
                    let unique = kv_rows * shape.num_kv_heads as f64 * bytes_per_row;
                    for arm in arms {
                        let s = self.time_attn(
                            shape, cap, &caches, &inputs, &output, &tmp, &slot_id_arr, &seq_pos_arr, &kv_lens,
                            &buckets, arm,
                        );
                        let arm_reads = match arm {
                            AttnArm::Gqa(g) => logical / g as f64,
                            _ => logical,
                        };
                        let line = serde_json::json!({
                            "schema": "hf2q.q4-gemma-decode.m0.cell.v1",
                            "family": "attn",
                            "shape": shape.name,
                            "step": step,
                            "n_q": n_q,
                            "head_dim": shape.head_dim,
                            "num_heads": shape.num_heads,
                            "num_kv_heads": shape.num_kv_heads,
                            "kv_capacity": cap,
                            "kv_lens": kv_lens,
                            "uniform_bucket": uniform,
                            "weight": weight,
                            "arm": arm.label(),
                            "kv_caches_cycled": caches.len(),
                            "dispatches_per_command_buffer": LAYERS,
                            "iters": self.cfg.iters,
                            "warmup": self.cfg.warmup,
                            "median_ns": s.median_ns,
                            "p10_ns": s.p10_ns,
                            "p90_ns": s.p90_ns,
                            "kv_bytes_logical_stock": logical,
                            "kv_bytes_logical_arm": arm_reads,
                            "kv_bytes_unique": unique,
                            "kv_gb_per_s_unique": unique / s.median_ns,
                        });
                        writeln!(self.receipts, "{line}").expect("receipt");
                        println!(
                            "attn    {:12} {:24} n_q={n_q:2} {:14} median {:9.1} µs  p10 {:9.1}  p90 {:9.1}  {:6.1} GB/s unique",
                            shape.name,
                            step,
                            arm.label(),
                            s.median_ns / 1e3,
                            s.p10_ns / 1e3,
                            s.p90_ns / 1e3,
                            unique / s.median_ns,
                        );
                    }
                }
            }
        }
    }

    fn time_attn(
        &mut self,
        shape: &AttnShape,
        cap: u32,
        caches: &[KvCache],
        inputs: &[MlxBuffer; 2],
        output: &MlxBuffer,
        tmp: &MlxBuffer,
        slot_id_arr: &MlxBuffer,
        seq_pos_arr: &MlxBuffer,
        kv_lens: &[u32],
        buckets: &[HybridSlotBucket],
        arm: AttnArm,
    ) -> Stats {
        let n_q = kv_lens.len();
        let (hd, nh, nkv) = (shape.head_dim as usize, shape.num_heads as usize, shape.num_kv_heads as usize);
        let npp = if hd == 512 { 2 } else { 1 };
        let max_kv = kv_lens.iter().copied().max().unwrap_or(1);
        let params_for = |kv_seq_len: u32, ring_start: u32, nsg: u32| FlashAttnVecTqHbParams {
            num_heads: shape.num_heads,
            num_kv_heads: shape.num_kv_heads,
            head_dim: shape.head_dim,
            kv_seq_len,
            kv_capacity: cap,
            scale: 1.0,
            mask_type: if shape.sliding { 2 } else { 1 },
            sliding_window: if shape.sliding { SLIDING_WINDOW } else { 0 },
            softcap: 0.0,
            ring_start,
            scale_factor_d512: 1.0,
            codebook_bits: 8,
            fuse_fwht_pre: 0,
            nsg,
        };
        let batched = params_for(max_kv, 0, buckets.iter().map(|b| b.nsg).max().unwrap_or(1));
        let seq_pos: Vec<u32> = seq_pos_arr.as_slice::<u32>().expect("seq_pos").to_vec();
        let mut samples = Vec::with_capacity(self.cfg.iters);
        for iter in 0..self.cfg.warmup + self.cfg.iters {
            let q = &inputs[iter % 2];
            let mut encoder = self.device.command_encoder().expect("encoder");
            for layer in 0..LAYERS {
                let kv = &caches[(iter * LAYERS + layer) % caches.len()];
                match arm {
                    AttnArm::StockBatched => flash_attn_vec_hybrid_batched(
                        &mut encoder, &mut self.registry, &self.device, n_q as u32, q, &kv.k, &kv.v,
                        &kv.norms, output, tmp, slot_id_arr, seq_pos_arr, &batched,
                    )
                    .expect("batched flash"),
                    AttnArm::Gqa(g) => flash_attn_vec_hybrid_batched_gqa(
                        &mut encoder, &mut self.registry, &self.device, q, &kv.k, &kv.v, &kv.norms,
                        output, tmp, slot_id_arr, seq_pos_arr, &batched, g, buckets,
                    )
                    .expect("gqa flash"),
                    AttnArm::StockPerSlot => {
                        // hf2q's mixed-bucket fallback: one solo dispatch (+ reduce) per slot.
                        let slot_rows = nkv * cap as usize;
                        let tmp_per_slot = nh * (hd + 2) * 32;
                        for (i, &kv_len) in kv_lens.iter().enumerate() {
                            let ring_start = if shape.sliding && kv_len >= cap {
                                (seq_pos[i] + 1) % cap
                            } else {
                                0
                            };
                            let p = params_for(kv_len, ring_start, compute_nsg(kv_len));
                            flash_attn_vec_hybrid(
                                &mut encoder,
                                &mut self.registry,
                                &self.device,
                                &q.slice_view((i * nh * hd * 4) as u64, nh * hd),
                                &kv.k.slice_view((i * slot_rows * hd * 2) as u64, slot_rows * hd),
                                &kv.v.slice_view((i * slot_rows * hd) as u64, slot_rows * hd),
                                &kv.norms.slice_view((i * slot_rows * npp * 4) as u64, slot_rows * npp),
                                &output.slice_view((i * nh * hd * 4) as u64, nh * hd),
                                &tmp.slice_view((i * tmp_per_slot * 4) as u64, tmp_per_slot),
                                &p,
                            )
                            .expect("per-slot flash");
                        }
                    }
                }
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
    if bench.cfg.wants("f2") {
        bench.f2_family();
    }
    if bench.cfg.wants("attn") {
        bench.attn_family();
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
