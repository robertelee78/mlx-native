//! ADR-063 Q4-GEMMA-PREFILL-MOE M0 microbenchmark: expert `mm_id` launch
//! geometry, model-free.
//!
//! Suites (select any subset on the command line; default `m0a`):
//!
//! * `m0a` — empty-threadgroup cost on today's kernels. A synthetic
//!   uniform-grid schedule is written from the host: one expert owns 32
//!   routed rows, every other expert owns none. The busy work is fixed at
//!   one row tile × ⌈N/64⌉ N tiles while the launched grid grows as
//!   `n_experts × ⌈n_tokens/32⌉ × ⌈N/64⌉`. The slope of GPU time against
//!   launched threadgroups is the empty-threadgroup cost `c_e`. The sweep
//!   runs with the busy expert at 0 and at `n_experts − 1`, for Q8_0
//!   (device-selected route) and Q5_0 (simdgroup route), on the Gemma 4
//!   down shape (704 → 2816, 128 experts), `top_k = 1`, `n_tokens` ∈ {33,
//!   256, 1024, 4096, 11360}.
//! * `m0b` — served shapes on today's route (`UniformGrid`), `n` ∈ {232,
//!   528, 764, 880, 1172, 1412, 1420}: gate/up (Q4_K 2816 → 1408, top_k 8),
//!   down (Q8_0 and Q5_0 704 → 2816) flattened (`8n` rows, top_k 1) and
//!   slotted (`n` tokens, top_k 8), `map0_ne20_8` and `map0_ne20_1` alone,
//!   the activation pass and the fused weighted-sum pass. Routing tables:
//!   uniform random, a bootstrap of a routing-witness JSONL
//!   (`--witness PATH`, rows drawn per layer with replacement), and a
//!   recorded agent table (`--recorded PATH`, same JSONL format, rows
//!   grouped by invocation and layer and used as recorded).
//! * `m0c` — the same shapes and tables, `UniformGrid` against
//!   `CompactTiles` (tile-fastest grid, the only order shipped since M0c
//!   measured it faster than N tile fastest) with the half-tile skip off
//!   and on (`Forced`: on regardless of whether it shipped on that path),
//!   for gate/up and slotted down; `map0` against
//!   `map0c` alone; and ± schedule reuse, timed as one command buffer of
//!   gate/up followed by down with its own `map0` or with the reusing call.
//!
//! Clock: `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime`, one command
//! buffer per measured op, `--warmup` (default 20) warm-up iterations and
//! `--iters` (default 200) measured iterations; median and p90 reported.
//!
//! Receipt fields: mlx-native commit (`GIT_COMMIT_SHA` or `git rev-parse`),
//! working-tree dirtiness, metallib SHA-256 and pipeline origin per kernel,
//! `MLX_PRECOMPILED_METALLIB`, device name, thermal state before and after,
//! macOS product and build version.
//!
//! Output: one JSON document (`--json PATH`, otherwise stdout); a
//! human-readable table goes to stderr.
//!
//! Needs the GPU lane and a quiet host. Invoke:
//!
//! ```text
//! cargo bench --locked --bench bench_mm_id_schedule -- m0a --json m0a.json
//! cargo bench --locked --bench bench_mm_id_schedule -- m0b \
//!     --witness $E/q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl \
//!     --json m0b.json
//! cargo bench --locked --bench bench_mm_id_schedule -- m0c \
//!     --witness $E/q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl \
//!     --json m0c.json
//! ```

use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use mlx_native::ops::fused_norm_add::dispatch_fused_moe_wsum_dnorm_add_f32;
use mlx_native::ops::moe_dispatch::moe_swiglu_seq_encode;
use mlx_native::ops::quantized_matmul_id_ggml::{
    dispatch_id_mm_map0_for_test, dispatch_id_mm_pooled_with_options_for_test,
    dispatch_id_mm_with_prepared_schedule_for_test, tensor_half_tile_skip_validated,
    GgmlIdMmDispatchParams, HalfTileSkip, IdMmScheduleOptions,
};
use mlx_native::{
    ggml_routing_policy_from_environment, quantized_matmul_id_ggml_pooled_slotted_with_policy,
    quantized_matmul_id_ggml_pooled_with_policy, CommandEncoder, DType, ExpertMmSchedule,
    GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy, GgmlType, IdMmScratch, KernelRegistry,
    MlxBuffer, MlxDevice,
};
use objc::runtime::{Class, Object, BOOL, NO};
use objc::{msg_send, sel, sel_impl};
use serde_json::{json, Value};

#[link(name = "Foundation", kind = "framework")]
extern "C" {}

const NR0: u64 = 64;
const NR1: u64 = 32;

const GEMMA_EXPERTS: u32 = 128;
const GEMMA_TOP_K: u32 = 8;
const GEMMA_HIDDEN: u32 = 2816;
const GEMMA_EXPERT_FF: u32 = 704;
const GEMMA_GATE_UP_N: u32 = 2 * GEMMA_EXPERT_FF;

const M0A_TOKENS: [u32; 5] = [33, 256, 1024, 4096, 11360];
const M0B_ROWS: [u32; 7] = [232, 528, 764, 880, 1172, 1412, 1420];

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Options {
    suites: Vec<String>,
    json: Option<String>,
    witness: Option<String>,
    recorded: Option<String>,
    layers: Option<Vec<u32>>,
    warmup: usize,
    iters: usize,
}

fn parse_options() -> Options {
    let mut options = Options {
        suites: Vec::new(),
        json: None,
        witness: None,
        recorded: None,
        layers: None,
        warmup: 20,
        iters: 200,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => options.json = args.next(),
            "--witness" => options.witness = args.next(),
            "--recorded" => options.recorded = args.next(),
            "--layers" => {
                options.layers = args.next().map(|list| {
                    list.split(',')
                        .map(|layer| layer.trim().parse().expect("--layers takes u32 values"))
                        .collect()
                })
            }
            "--warmup" => {
                options.warmup = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .expect("--warmup takes a count")
            }
            "--iters" => {
                options.iters = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .expect("--iters takes a count")
            }
            "m0a" | "m0b" | "m0c" => options.suites.push(arg),
            // cargo bench appends `--bench`; libtest-style flags are ignored.
            other if other.starts_with("--") => {}
            other => panic!("unknown argument {other:?}; suites are m0a, m0b, m0c"),
        }
    }
    if options.suites.is_empty() {
        options.suites.push("m0a".into());
    }
    assert!(options.iters > 0, "--iters must be positive");
    options
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn thermal_state() -> Value {
    let Some(class) = Class::get("NSProcessInfo") else {
        return Value::Null;
    };
    // SAFETY: selectors are probed before use; `thermalState` is an
    // NSInteger enum and `isLowPowerModeEnabled` a BOOL on macOS.
    unsafe {
        let info: *mut Object = msg_send![class, processInfo];
        if info.is_null() {
            return Value::Null;
        }
        let has_thermal: BOOL = msg_send![info, respondsToSelector: sel!(thermalState)];
        if has_thermal == NO {
            return Value::Null;
        }
        let thermal: isize = msg_send![info, thermalState];
        let has_power: BOOL = msg_send![info, respondsToSelector: sel!(isLowPowerModeEnabled)];
        let low_power = if has_power == NO {
            Value::Null
        } else {
            let low: BOOL = msg_send![info, isLowPowerModeEnabled];
            Value::Bool(low != NO)
        };
        json!({ "thermal_state": thermal, "low_power_mode": low_power })
    }
}

fn receipt(device: &MlxDevice) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let commit = std::env::var("GIT_COMMIT_SHA")
        .ok()
        .or_else(|| command_stdout("git", &["-C", manifest, "rev-parse", "HEAD"]));
    let dirty = command_stdout("git", &["-C", manifest, "status", "--porcelain"])
        .map(|status| !status.is_empty());
    json!({
        "schema": "mlx-native.bench_mm_id_schedule.v1",
        "mlx_native_version": env!("CARGO_PKG_VERSION"),
        "mlx_native_commit": commit,
        "working_tree_dirty": dirty,
        "mlx_precompiled_metallib": std::env::var("MLX_PRECOMPILED_METALLIB").ok(),
        "device_name": device.name(),
        "tensor_half_tile_skip_ships": tensor_half_tile_skip_validated(device),
        "macos_product_version": command_stdout("sw_vers", &["-productVersion"]),
        "macos_build_version": command_stdout("sw_vers", &["-buildVersion"]),
        "thermal_at_start": thermal_state(),
    })
}

/// Origin and metallib digest of every compiled pipeline a case used.
fn pipeline_receipts(registry: &KernelRegistry, kernels: &[&str]) -> Value {
    let mut map = serde_json::Map::new();
    for kernel in kernels {
        if let Ok(identity) = registry.pipeline_identity(kernel) {
            map.insert(
                (*kernel).to_string(),
                serde_json::to_value(identity).expect("pipeline identity serializes"),
            );
        }
    }
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Stats {
    median_us: f64,
    p90_us: f64,
    min_us: f64,
    max_us: f64,
}

impl Stats {
    fn to_json(self) -> Value {
        json!({
            "median_us": self.median_us,
            "p90_us": self.p90_us,
            "min_us": self.min_us,
            "max_us": self.max_us,
        })
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

/// One command buffer per sample; GPU time from the buffer's timestamps.
fn time_gpu(
    device: &MlxDevice,
    options: &Options,
    mut encode: impl FnMut(&mut CommandEncoder),
) -> Stats {
    let mut sample = || {
        let mut encoder = device.command_encoder().expect("command encoder");
        encode(&mut encoder);
        let (start, end) = encoder
            .commit_wait_with_gpu_time()
            .expect("measured command buffer");
        (end - start) * 1e6
    };
    for _ in 0..options.warmup {
        sample();
    }
    let mut samples: Vec<f64> = (0..options.iters).map(|_| sample()).collect();
    samples.sort_by(|left, right| left.partial_cmp(right).expect("finite GPU time"));
    Stats {
        median_us: percentile(&samples, 0.5),
        p90_us: percentile(&samples, 0.9),
        min_us: samples[0],
        max_us: samples[samples.len() - 1],
    }
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: u32) -> u32 {
        (self.next() % u64::from(bound)) as u32
    }
}

/// Offsets of the f16 scale fields within one block, so random payloads
/// keep finite scales.
fn half_scale_offsets(ggml_type: GgmlType) -> &'static [usize] {
    match ggml_type {
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 => &[0],
        GgmlType::Q4_K => &[0, 2],
        other => panic!("bench weight generator does not cover {other:?}"),
    }
}

fn expert_matrix_bytes(ggml_type: GgmlType, n: u32, k: u32) -> u64 {
    u64::from(n) * u64::from(k / ggml_type.block_values()) * u64::from(ggml_type.block_bytes())
}

fn weight_buffer(device: &MlxDevice, ggml_type: GgmlType, n: u32, k: u32, seed: u64) -> MlxBuffer {
    let total = expert_matrix_bytes(ggml_type, n, k) as usize * GEMMA_EXPERTS as usize;
    let mut buffer = device
        .alloc_buffer(total, DType::U8, vec![total])
        .expect("weight buffer");
    let bytes: &mut [u8] = buffer.as_mut_slice().expect("weight bytes");
    let mut rng = XorShift(seed);
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    let block = ggml_type.block_bytes() as usize;
    for block_bytes in bytes.chunks_exact_mut(block) {
        for &offset in half_scale_offsets(ggml_type) {
            // 0x2C00 == 0.0625 in IEEE half.
            block_bytes[offset..offset + 2].copy_from_slice(&0x2C00u16.to_le_bytes());
        }
    }
    buffer
}

fn f32_buffer(device: &MlxDevice, elements: usize, seed: u64) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("f32 buffer");
    let values: &mut [f32] = buffer.as_mut_slice().expect("f32 values");
    let mut rng = XorShift(seed);
    for value in values.iter_mut() {
        *value = (rng.below(2001) as f32 - 1000.0) / 4000.0;
    }
    buffer
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(
            values.len().max(1) * 4,
            DType::U32,
            vec![values.len().max(1)],
        )
        .expect("u32 buffer");
    buffer.as_mut_slice::<u32>().expect("u32 values")[..values.len()].copy_from_slice(values);
    buffer
}

// ---------------------------------------------------------------------------
// Routing tables
// ---------------------------------------------------------------------------

struct RoutingTable {
    source: String,
    layer: Option<u32>,
    /// `[n_tokens, top_k]` distinct expert ids per token.
    ids: Vec<u32>,
}

impl RoutingTable {
    fn n_tokens(&self) -> u32 {
        (self.ids.len() / GEMMA_TOP_K as usize) as u32
    }

    fn busy_row_tiles(&self) -> u64 {
        let mut counts = vec![0u64; GEMMA_EXPERTS as usize];
        for &id in &self.ids {
            counts[id as usize] += 1;
        }
        counts.iter().map(|&count| count.div_ceil(NR1)).sum()
    }

    fn to_json(&self) -> Value {
        json!({ "source": self.source, "layer": self.layer, "n": self.n_tokens() })
    }
}

fn uniform_table(n: u32, seed: u64) -> RoutingTable {
    let mut rng = XorShift(seed);
    let mut ids = Vec::with_capacity((n * GEMMA_TOP_K) as usize);
    for _ in 0..n {
        let mut row: Vec<u32> = Vec::with_capacity(GEMMA_TOP_K as usize);
        while row.len() < GEMMA_TOP_K as usize {
            let id = rng.below(GEMMA_EXPERTS);
            if !row.contains(&id) {
                row.push(id);
            }
        }
        ids.extend(row);
    }
    RoutingTable {
        source: "uniform".into(),
        layer: None,
        ids,
    }
}

/// Rows of a routing-witness JSONL keyed by `(invocation, layer)`.
fn witness_rows(path: &str) -> BTreeMap<(u64, u32), Vec<[u32; 8]>> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut rows: BTreeMap<(u64, u32), Vec<[u32; 8]>> = BTreeMap::new();
    for line in std::io::BufReader::new(file).lines() {
        let line = line.expect("witness line");
        let record: Value = serde_json::from_str(&line).expect("witness JSON");
        if record["kind"] != "routing_row" {
            continue;
        }
        let at = &record["at"];
        let invocation = at["invocation"].as_u64().expect("invocation");
        let layer = at["layer"].as_u64().expect("layer") as u32;
        let selected: Vec<u32> = record["selected_ids"]
            .as_array()
            .expect("selected_ids")
            .iter()
            .map(|id| id.as_u64().expect("expert id") as u32)
            .collect();
        let row: [u32; 8] = selected.try_into().expect("top-8 witness row");
        rows.entry((invocation, layer)).or_default().push(row);
    }
    rows
}

fn witness_bootstrap_tables(
    path: &str,
    n: u32,
    layers: Option<&[u32]>,
    seed: u64,
) -> Vec<RoutingTable> {
    let mut per_layer: BTreeMap<u32, Vec<[u32; 8]>> = BTreeMap::new();
    for ((_, layer), rows) in witness_rows(path) {
        per_layer.entry(layer).or_default().extend(rows);
    }
    let mut rng = XorShift(seed);
    per_layer
        .into_iter()
        .filter(|(layer, _)| layers.map_or(true, |wanted| wanted.contains(layer)))
        .map(|(layer, rows)| {
            let ids = (0..n)
                .flat_map(|_| rows[rng.below(rows.len() as u32) as usize])
                .collect();
            RoutingTable {
                source: "witness_bootstrap".into(),
                layer: Some(layer),
                ids,
            }
        })
        .collect()
}

fn recorded_tables(path: &str, layers: Option<&[u32]>) -> Vec<RoutingTable> {
    witness_rows(path)
        .into_iter()
        .filter(|((_, layer), _)| layers.map_or(true, |wanted| wanted.contains(layer)))
        .filter(|(_, rows)| rows.len() as u32 > mlx_native::MM_ID_ROUTING_THRESHOLD)
        .map(|((invocation, layer), rows)| RoutingTable {
            source: format!("recorded:invocation={invocation}"),
            layer: Some(layer),
            ids: rows.into_iter().flatten().collect(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Kernel names (receipt only)
// ---------------------------------------------------------------------------

fn type_tag(ggml_type: GgmlType) -> &'static str {
    match ggml_type {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q5_0 => "q5_0",
        GgmlType::Q8_0 => "q8_0",
        other => panic!("bench does not cover {other:?}"),
    }
}

fn mm_kernel_candidates(ggml_type: GgmlType) -> Vec<String> {
    let tag = type_tag(ggml_type);
    vec![
        format!("kernel_mul_mm_id_{tag}_tensor_f32"),
        format!("kernel_mul_mm_id_{tag}_f32"),
    ]
}

fn resolved_mm_kernel(registry: &KernelRegistry, ggml_type: GgmlType) -> Option<String> {
    mm_kernel_candidates(ggml_type)
        .into_iter()
        .find(|name| registry.is_cached(name))
}

fn launched_grid(n_tokens: u32, n: u32) -> u64 {
    u64::from(n_tokens).div_ceil(NR1) * u64::from(n).div_ceil(NR0) * u64::from(GEMMA_EXPERTS)
}

// ---------------------------------------------------------------------------
// M0a — empty-threadgroup cost
// ---------------------------------------------------------------------------

fn least_squares_slope(points: &[(f64, f64)]) -> f64 {
    let count = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / count;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / count;
    let covariance: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let variance: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    covariance / variance
}

fn run_m0a(device: &MlxDevice, registry: &mut KernelRegistry, options: &Options) -> Value {
    let (n, k) = (GEMMA_HIDDEN, GEMMA_EXPERT_FF);
    let max_tokens = *M0A_TOKENS.iter().max().expect("sweep");
    let input = f32_buffer(device, max_tokens as usize * k as usize, 11);
    let output = f32_buffer(device, max_tokens as usize * n as usize, 12);
    let ids = u32_buffer(device, &vec![0u32; max_tokens as usize]);
    let mut cases = Vec::new();
    let mut fits = Vec::new();
    for ggml_type in [GgmlType::Q8_0, GgmlType::Q5_0] {
        let weight = weight_buffer(device, ggml_type, n, k, 13);
        for busy_expert in [0, GEMMA_EXPERTS - 1] {
            let mut points = Vec::new();
            for &n_tokens in &M0A_TOKENS {
                let params = GgmlIdMmDispatchParams {
                    n_tokens,
                    top_k: 1,
                    n,
                    k,
                    n_experts: GEMMA_EXPERTS,
                    expert_stride: expert_matrix_bytes(ggml_type, n, k),
                    ggml_type,
                };
                let mut counts = vec![0u32; GEMMA_EXPERTS as usize];
                counts[busy_expert as usize] = NR1 as u32;
                let htpe = u32_buffer(device, &counts);
                let mut lists = vec![0u32; GEMMA_EXPERTS as usize * n_tokens as usize];
                let base = busy_expert as usize * n_tokens as usize;
                for row in 0..NR1 as usize {
                    lists[base + row] = row as u32;
                }
                let hids = u32_buffer(device, &lists);
                let stats = time_gpu(device, options, |encoder| {
                    dispatch_id_mm_with_prepared_schedule_for_test(
                        encoder, registry, device, &input, &weight, &ids, &htpe, &hids, &output,
                        &params,
                    )
                    .expect("synthetic-schedule mm_id");
                });
                let launched = launched_grid(n_tokens, n);
                let busy = u64::from(n).div_ceil(NR0);
                points.push((launched as f64, stats.median_us));
                eprintln!(
                    "m0a {:>5} busy_expert={:>3} n_tokens={:>6} launched_tg={:>9} busy_tg={:>3} median={:>10.2}us p90={:>10.2}us",
                    type_tag(ggml_type), busy_expert, n_tokens, launched, busy,
                    stats.median_us, stats.p90_us,
                );
                cases.push(json!({
                    "ggml_type": type_tag(ggml_type),
                    "kernel": resolved_mm_kernel(registry, ggml_type),
                    "busy_expert": busy_expert,
                    "n_tokens": n_tokens,
                    "top_k": 1,
                    "n": n,
                    "k": k,
                    "n_experts": GEMMA_EXPERTS,
                    "launched_threadgroups": launched,
                    "busy_threadgroups": busy,
                    "gpu": stats.to_json(),
                }));
            }
            let slope_us = least_squares_slope(&points);
            eprintln!(
                "m0a {:>5} busy_expert={:>3} c_e = {:.4} ns per launched threadgroup",
                type_tag(ggml_type),
                busy_expert,
                slope_us * 1e3,
            );
            fits.push(json!({
                "ggml_type": type_tag(ggml_type),
                "busy_expert": busy_expert,
                "c_e_ns_per_threadgroup": slope_us * 1e3,
                "fit": "least squares of median GPU us against launched threadgroups",
            }));
        }
    }
    let kernels: Vec<String> = [GgmlType::Q8_0, GgmlType::Q5_0]
        .into_iter()
        .flat_map(mm_kernel_candidates)
        .collect();
    let kernel_refs: Vec<&str> = kernels.iter().map(String::as_str).collect();
    json!({
        "cases": cases,
        "c_e": fits,
        "pipelines": pipeline_receipts(registry, &kernel_refs),
    })
}

// ---------------------------------------------------------------------------
// M0b — served shapes on today's route
// ---------------------------------------------------------------------------

struct ServedWeights {
    gate_up_q4k: MlxBuffer,
    down_q8_0: MlxBuffer,
    down_q5_0: MlxBuffer,
}

fn expert_params(
    ggml_type: GgmlType,
    n_tokens: u32,
    top_k: u32,
    n: u32,
    k: u32,
) -> GgmlQuantizedMatmulIdParams {
    GgmlQuantizedMatmulIdParams {
        n_tokens,
        top_k,
        n,
        k,
        n_experts: GEMMA_EXPERTS,
        expert_stride: expert_matrix_bytes(ggml_type, n, k),
        ggml_type,
    }
}

fn case_json(
    table: &RoutingTable,
    op: &str,
    ggml_type: Option<GgmlType>,
    launched: Option<u64>,
    busy: Option<u64>,
    stats: Stats,
    kernel: Option<String>,
) -> Value {
    json!({
        "routing": table.to_json(),
        "op": op,
        "ggml_type": ggml_type.map(type_tag),
        "kernel": kernel,
        "launched_threadgroups": launched,
        "busy_threadgroups": busy,
        "gpu": stats.to_json(),
    })
}

fn run_m0b_table(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    options: &Options,
    weights: &ServedWeights,
    table: &RoutingTable,
) -> Vec<Value> {
    let rows = table.n_tokens();
    let routed = rows * GEMMA_TOP_K;
    let busy_tiles = table.busy_row_tiles();
    let ids = u32_buffer(device, &table.ids);
    let hidden_in = f32_buffer(device, (rows * GEMMA_HIDDEN) as usize, 21);
    let gate_up_out = f32_buffer(device, (routed * GEMMA_GATE_UP_N) as usize, 22);
    let down_in = f32_buffer(device, (routed * GEMMA_EXPERT_FF) as usize, 23);
    let down_out = f32_buffer(device, (routed * GEMMA_HIDDEN) as usize, 24);
    let mut scratch =
        IdMmScratch::alloc(device, GEMMA_EXPERTS, routed).expect("flattened-capacity scratch");
    let mut cases = Vec::new();
    let grid = routing_policy(ExpertMmSchedule::UniformGrid);

    let gate_up = expert_params(
        GgmlType::Q4_K,
        rows,
        GEMMA_TOP_K,
        GEMMA_GATE_UP_N,
        GEMMA_HIDDEN,
    );
    let stats = time_gpu(device, options, |encoder| {
        quantized_matmul_id_ggml_pooled_with_policy(
            encoder,
            registry,
            device,
            &hidden_in,
            &weights.gate_up_q4k,
            &ids,
            &gate_up_out,
            &mut scratch,
            &gate_up,
            &grid,
        )
        .expect("gate/up mm_id");
    });
    let n_tiles = u64::from(GEMMA_GATE_UP_N).div_ceil(NR0);
    cases.push(case_json(
        table,
        "gate_up_pooled",
        Some(GgmlType::Q4_K),
        Some(launched_grid(rows, GEMMA_GATE_UP_N)),
        Some(busy_tiles * n_tiles),
        stats,
        resolved_mm_kernel(registry, GgmlType::Q4_K),
    ));

    for (ggml_type, weight) in [
        (GgmlType::Q8_0, &weights.down_q8_0),
        (GgmlType::Q5_0, &weights.down_q5_0),
    ] {
        let n_tiles = u64::from(GEMMA_HIDDEN).div_ceil(NR0);
        let flat = expert_params(ggml_type, routed, 1, GEMMA_HIDDEN, GEMMA_EXPERT_FF);
        let stats = time_gpu(device, options, |encoder| {
            quantized_matmul_id_ggml_pooled_with_policy(
                encoder,
                registry,
                device,
                &down_in,
                weight,
                &ids,
                &down_out,
                &mut scratch,
                &flat,
                &grid,
            )
            .expect("flattened down mm_id");
        });
        cases.push(case_json(
            table,
            "down_flattened",
            Some(ggml_type),
            Some(launched_grid(routed, GEMMA_HIDDEN)),
            Some(busy_tiles * n_tiles),
            stats,
            resolved_mm_kernel(registry, ggml_type),
        ));
        let slotted = expert_params(ggml_type, rows, GEMMA_TOP_K, GEMMA_HIDDEN, GEMMA_EXPERT_FF);
        let stats = time_gpu(device, options, |encoder| {
            quantized_matmul_id_ggml_pooled_slotted_with_policy(
                encoder,
                registry,
                device,
                &down_in,
                weight,
                &ids,
                &down_out,
                &mut scratch,
                &slotted,
                &grid,
            )
            .expect("slotted down mm_id");
        });
        cases.push(case_json(
            table,
            "down_slotted",
            Some(ggml_type),
            Some(launched_grid(rows, GEMMA_HIDDEN)),
            Some(busy_tiles * n_tiles),
            stats,
            resolved_mm_kernel(registry, ggml_type),
        ));
    }

    for (op, n_tokens, top_k) in [
        ("map0_ne20_8", rows, GEMMA_TOP_K),
        ("map0_ne20_1", routed, 1),
    ] {
        let params = GgmlIdMmDispatchParams {
            n_tokens,
            top_k,
            n: GEMMA_HIDDEN,
            k: GEMMA_EXPERT_FF,
            n_experts: GEMMA_EXPERTS,
            expert_stride: expert_matrix_bytes(GgmlType::Q8_0, GEMMA_HIDDEN, GEMMA_EXPERT_FF),
            ggml_type: GgmlType::Q8_0,
        };
        let mut htpe = u32_buffer(device, &vec![0; params.htpe_bytes() / 4]);
        let mut hids = u32_buffer(device, &vec![0; params.hids_bytes() / 4]);
        let stats = time_gpu(device, options, |encoder| {
            dispatch_id_mm_map0_for_test(
                encoder,
                registry,
                device,
                &ids,
                &mut htpe,
                &mut hids,
                &params,
                ExpertMmSchedule::UniformGrid,
            )
            .expect("map0 alone");
        });
        cases.push(case_json(
            table,
            op,
            None,
            None,
            None,
            stats,
            Some(op.into()),
        ));
    }

    let stats = time_gpu(device, options, |encoder| {
        moe_swiglu_seq_encode(
            encoder,
            registry,
            device.metal_device(),
            &gate_up_out,
            &down_in,
            GEMMA_EXPERT_FF as usize,
            GEMMA_TOP_K as usize,
            rows as usize,
        )
        .expect("activation pass");
    });
    cases.push(case_json(
        table,
        "activation_moe_swiglu_seq",
        None,
        None,
        None,
        stats,
        None,
    ));

    let routing_weights = f32_buffer(device, routed as usize, 25);
    let residual = f32_buffer(device, (rows * GEMMA_HIDDEN) as usize, 26);
    let res_norm = f32_buffer(device, GEMMA_HIDDEN as usize, 27);
    let moe_norm = f32_buffer(device, GEMMA_HIDDEN as usize, 28);
    let summed = f32_buffer(device, (rows * GEMMA_HIDDEN) as usize, 29);
    let stats = time_gpu(device, options, |encoder| {
        dispatch_fused_moe_wsum_dnorm_add_f32(
            encoder,
            registry,
            device.metal_device(),
            &down_out,
            &routing_weights,
            &residual,
            &res_norm,
            &moe_norm,
            &summed,
            GEMMA_HIDDEN,
            GEMMA_TOP_K,
            rows,
            1e-6,
        )
        .expect("fused weighted sum");
    });
    cases.push(case_json(
        table,
        "fused_moe_wsum_dnorm_add",
        None,
        None,
        None,
        stats,
        None,
    ));

    for case in &cases {
        eprintln!(
            "m0b {:<26} n={:>5} {:<5} layer={:>4} median={:>10.2}us p90={:>10.2}us launched={} busy={}",
            case["op"].as_str().unwrap_or(""),
            rows,
            case["ggml_type"].as_str().unwrap_or("-"),
            case["routing"]["layer"],
            case["gpu"]["median_us"].as_f64().unwrap_or(f64::NAN),
            case["gpu"]["p90_us"].as_f64().unwrap_or(f64::NAN),
            case["launched_threadgroups"],
            case["busy_threadgroups"],
        );
    }
    cases
}

fn served_weights(device: &MlxDevice) -> ServedWeights {
    ServedWeights {
        gate_up_q4k: weight_buffer(device, GgmlType::Q4_K, GEMMA_GATE_UP_N, GEMMA_HIDDEN, 31),
        down_q8_0: weight_buffer(device, GgmlType::Q8_0, GEMMA_HIDDEN, GEMMA_EXPERT_FF, 32),
        down_q5_0: weight_buffer(device, GgmlType::Q5_0, GEMMA_HIDDEN, GEMMA_EXPERT_FF, 33),
    }
}

fn served_tables(options: &Options) -> Vec<RoutingTable> {
    let mut tables = Vec::new();
    for (index, &n) in M0B_ROWS.iter().enumerate() {
        tables.push(uniform_table(n, 1000 + index as u64));
        if let Some(path) = &options.witness {
            tables.extend(witness_bootstrap_tables(
                path,
                n,
                options.layers.as_deref(),
                2000 + index as u64,
            ));
        }
    }
    if let Some(path) = &options.recorded {
        tables.extend(recorded_tables(path, options.layers.as_deref()));
    }
    tables
}

/// The process routing policy with an explicit expert `mm_id` schedule.
fn routing_policy(schedule: ExpertMmSchedule) -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        expert_mm_schedule: schedule,
        ..ggml_routing_policy_from_environment()
    }
}

fn run_m0b(device: &MlxDevice, registry: &mut KernelRegistry, options: &Options) -> Value {
    let weights = served_weights(device);
    let tables = served_tables(options);
    let cases: Vec<Value> = tables
        .iter()
        .flat_map(|table| run_m0b_table(device, registry, options, &weights, table))
        .collect();
    let mut kernels: Vec<String> = [GgmlType::Q4_K, GgmlType::Q8_0, GgmlType::Q5_0]
        .into_iter()
        .flat_map(mm_kernel_candidates)
        .collect();
    kernels.extend(
        [
            "kernel_mul_mm_id_map0_ne20_1",
            "kernel_mul_mm_id_map0_ne20_8",
            "moe_swiglu_seq",
            "fused_moe_wsum_dnorm_add_f32",
        ]
        .map(String::from),
    );
    let kernel_refs: Vec<&str> = kernels.iter().map(String::as_str).collect();
    json!({
        "routing_sources": {
            "uniform": true,
            "witness": options.witness,
            "recorded": options.recorded,
            "layers": options.layers,
        },
        "cases": cases,
        "pipelines": pipeline_receipts(registry, &kernel_refs),
    })
}

// ---------------------------------------------------------------------------
// M0c — uniform grid against the compact schedule
// ---------------------------------------------------------------------------

fn compact_launch(rows: u32, n: u32) -> u64 {
    let rows = u64::from(rows);
    let max_tiles = rows.min(rows.div_ceil(NR1) + u64::from(GEMMA_EXPERTS).min(rows) - 1);
    max_tiles * u64::from(n).div_ceil(NR0)
}

/// Schedule variants M0c times: label, options.
fn m0c_variants() -> Vec<(String, IdMmScheduleOptions)> {
    let mut variants = vec![(
        "uniform_grid".to_string(),
        IdMmScheduleOptions::production(ExpertMmSchedule::UniformGrid),
    )];
    for half_tile_skip in [HalfTileSkip::Disabled, HalfTileSkip::Forced] {
        variants.push((
            format!("compact_skip_{half_tile_skip:?}"),
            IdMmScheduleOptions {
                schedule: ExpertMmSchedule::CompactTiles,
                half_tile_skip,
            },
        ));
    }
    variants
}

#[allow(clippy::too_many_arguments)]
fn encode_expert(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
    slotted: bool,
    reuse: bool,
    options: IdMmScheduleOptions,
) {
    dispatch_id_mm_pooled_with_options_for_test(
        encoder,
        registry,
        device,
        input,
        weight,
        ids,
        output,
        scratch,
        params,
        slotted,
        reuse,
        &routing_policy(options.schedule),
        options,
    )
    .expect("expert mm_id");
}

fn run_m0c_table(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    options: &Options,
    weights: &ServedWeights,
    table: &RoutingTable,
) -> Vec<Value> {
    let rows = table.n_tokens();
    let routed = rows * GEMMA_TOP_K;
    let busy_tiles = table.busy_row_tiles();
    let ids = u32_buffer(device, &table.ids);
    let hidden_in = f32_buffer(device, (rows * GEMMA_HIDDEN) as usize, 41);
    let gate_up_out = f32_buffer(device, (routed * GEMMA_GATE_UP_N) as usize, 42);
    let down_in = f32_buffer(device, (routed * GEMMA_EXPERT_FF) as usize, 43);
    let down_out = f32_buffer(device, (routed * GEMMA_HIDDEN) as usize, 44);
    let mut scratch = IdMmScratch::alloc(device, GEMMA_EXPERTS, rows).expect("scratch");
    let gate_up = expert_params(
        GgmlType::Q4_K,
        rows,
        GEMMA_TOP_K,
        GEMMA_GATE_UP_N,
        GEMMA_HIDDEN,
    );
    let mut cases = Vec::new();
    let mut push = |op: &str,
                    variant: &str,
                    ggml_type: Option<GgmlType>,
                    launched,
                    busy,
                    stats: Stats| {
        let mut case = case_json(table, op, ggml_type, launched, busy, stats, None);
        case["variant"] = json!(variant);
        eprintln!(
            "m0c {op:<22} {variant:<40} n={rows:>5} {:<5} layer={:>4} median={:>10.2}us p90={:>10.2}us",
            ggml_type.map(type_tag).unwrap_or("-"),
            table.layer.map_or("-".to_string(), |layer| layer.to_string()),
            stats.median_us,
            stats.p90_us,
        );
        cases.push(case);
    };

    for (variant, schedule) in m0c_variants() {
        let compact = schedule.schedule == ExpertMmSchedule::CompactTiles;
        let launched = |n: u32| {
            if compact {
                compact_launch(routed, n)
            } else {
                launched_grid(rows, n)
            }
        };
        let stats = time_gpu(device, options, |encoder| {
            encode_expert(
                encoder,
                registry,
                device,
                &hidden_in,
                &weights.gate_up_q4k,
                &ids,
                &gate_up_out,
                &mut scratch,
                &gate_up,
                false,
                false,
                schedule,
            );
        });
        push(
            "gate_up",
            &variant,
            Some(GgmlType::Q4_K),
            Some(launched(GEMMA_GATE_UP_N)),
            Some(busy_tiles * u64::from(GEMMA_GATE_UP_N).div_ceil(NR0)),
            stats,
        );
        for (ggml_type, weight) in [
            (GgmlType::Q8_0, &weights.down_q8_0),
            (GgmlType::Q5_0, &weights.down_q5_0),
        ] {
            let down = expert_params(ggml_type, rows, GEMMA_TOP_K, GEMMA_HIDDEN, GEMMA_EXPERT_FF);
            let busy = Some(busy_tiles * u64::from(GEMMA_HIDDEN).div_ceil(NR0));
            let stats = time_gpu(device, options, |encoder| {
                encode_expert(
                    encoder,
                    registry,
                    device,
                    &down_in,
                    weight,
                    &ids,
                    &down_out,
                    &mut scratch,
                    &down,
                    true,
                    false,
                    schedule,
                );
            });
            push(
                "down_slotted",
                &variant,
                Some(ggml_type),
                Some(launched(GEMMA_HIDDEN)),
                busy,
                stats,
            );
            // ± P2b: gate/up then down in one command buffer; down builds its
            // own schedule or reuses the gate/up schedule.
            for reuse in [false, true] {
                let stats = time_gpu(device, options, |encoder| {
                    encode_expert(
                        encoder,
                        registry,
                        device,
                        &hidden_in,
                        &weights.gate_up_q4k,
                        &ids,
                        &gate_up_out,
                        &mut scratch,
                        &gate_up,
                        false,
                        false,
                        schedule,
                    );
                    // The production graph orders down after gate/up (and the
                    // activation) with a barrier; without it the fresh arm's
                    // map0 could overwrite the schedule gate/up still reads.
                    encoder.memory_barrier();
                    encode_expert(
                        encoder,
                        registry,
                        device,
                        &down_in,
                        weight,
                        &ids,
                        &down_out,
                        &mut scratch,
                        &down,
                        true,
                        reuse,
                        schedule,
                    );
                });
                push(
                    if reuse {
                        "gate_up+down_reuse"
                    } else {
                        "gate_up+down_fresh"
                    },
                    &variant,
                    Some(ggml_type),
                    None,
                    None,
                    stats,
                );
            }
        }
    }

    for schedule in [
        ExpertMmSchedule::UniformGrid,
        ExpertMmSchedule::CompactTiles,
    ] {
        for (n_tokens, top_k) in [(rows, GEMMA_TOP_K), (routed, 1)] {
            let params = GgmlIdMmDispatchParams {
                n_tokens,
                top_k,
                n: GEMMA_HIDDEN,
                k: GEMMA_EXPERT_FF,
                n_experts: GEMMA_EXPERTS,
                expert_stride: expert_matrix_bytes(GgmlType::Q8_0, GEMMA_HIDDEN, GEMMA_EXPERT_FF),
                ggml_type: GgmlType::Q8_0,
            };
            let mut map_scratch =
                IdMmScratch::alloc(device, GEMMA_EXPERTS, n_tokens).expect("map0 scratch");
            let (htpe, hids) = map_scratch.buffers_mut();
            let stats = time_gpu(device, options, |encoder| {
                dispatch_id_mm_map0_for_test(
                    encoder, registry, device, &ids, &mut *htpe, &mut *hids, &params, schedule,
                )
                .expect("schedule stage alone");
            });
            let op = match schedule {
                ExpertMmSchedule::UniformGrid => format!("map0_ne20_{top_k}"),
                ExpertMmSchedule::CompactTiles => format!("map0c_ne20_{top_k}"),
            };
            push(&op, &format!("{schedule:?}"), None, None, None, stats);
        }
    }
    cases
}

fn run_m0c(device: &MlxDevice, registry: &mut KernelRegistry, options: &Options) -> Value {
    let weights = served_weights(device);
    let tables = served_tables(options);
    let cases: Vec<Value> = tables
        .iter()
        .flat_map(|table| run_m0c_table(device, registry, options, &weights, table))
        .collect();
    let mut kernels: Vec<String> = [GgmlType::Q4_K, GgmlType::Q8_0, GgmlType::Q5_0]
        .into_iter()
        .flat_map(|ggml_type| {
            mm_kernel_candidates(ggml_type)
                .into_iter()
                .flat_map(|name| [format!("{name}_c"), name])
        })
        .collect();
    kernels.extend(
        [
            "kernel_mul_mm_id_map0_ne20_1",
            "kernel_mul_mm_id_map0_ne20_8",
            "kernel_mul_mm_id_map0c_ne20_1",
            "kernel_mul_mm_id_map0c_ne20_8",
        ]
        .map(String::from),
    );
    let kernel_refs: Vec<&str> = kernels.iter().map(String::as_str).collect();
    json!({
        "routing_sources": {
            "uniform": true,
            "witness": options.witness,
            "recorded": options.recorded,
            "layers": options.layers,
        },
        "variants": m0c_variants().into_iter().map(|(label, _)| label).collect::<Vec<_>>(),
        "cases": cases,
        "pipelines": pipeline_receipts(registry, &kernel_refs),
    })
}

fn main() {
    let options = parse_options();
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    let mut document = receipt(&device);
    document["clock"] = json!({
        "source": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "command_buffers_per_sample": 1,
        "warmup": options.warmup,
        "measured": options.iters,
    });
    let mut suites = serde_json::Map::new();
    for suite in &options.suites {
        let result = match suite.as_str() {
            "m0a" => run_m0a(&device, &mut registry, &options),
            "m0b" => run_m0b(&device, &mut registry, &options),
            "m0c" => run_m0c(&device, &mut registry, &options),
            _ => unreachable!("validated suite"),
        };
        suites.insert(suite.clone(), result);
    }
    document["suites"] = Value::Object(suites);
    document["thermal_at_end"] = thermal_state();
    let text = serde_json::to_string_pretty(&document).expect("serialize receipt");
    match &options.json {
        Some(path) => std::fs::write(path, text).unwrap_or_else(|e| panic!("write {path}: {e}")),
        None => {
            let mut stdout = std::io::stdout().lock();
            writeln!(stdout, "{text}").expect("write receipt");
        }
    }
}
