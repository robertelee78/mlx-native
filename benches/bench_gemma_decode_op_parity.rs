//! Gemma 4 decode op parity: hf2q's dense and expert quantized mat-vec on
//! the exact shapes of the peer reference's test-backend-ops perf cases
//! (`v0130/q4-gemma-decode/m0p-*/gemma-m0p-cases.patch`), model-free.
//!
//! Families (`GEMMA_OP_PARITY_FAMILIES`, comma list, default
//! `dense,lmhead,experts`):
//!
//! * `dense` — Q4_K 4096×2816, 8192×2816, 2048×2816, 1024×2816, 2816×4096,
//!   2816×8192, 2112×2816; Q6_K 2048×2816; Q5_0 / Q8_0 2816×2112
//!   (N×K), widths 1..=8.
//! * `lmhead` — Q6_K 262144×2816, widths 1..=8.
//! * `experts` — `mul_mat_id` over 128 experts, top-8, fixed random ids
//!   (distinct per token), n_tokens ∈ {1, 4, 8, 16}: gate_up Q4_K
//!   1408×2816 with each token's row broadcast to its 8 pairs; down
//!   Q5_0 / Q8_0 2816×704 with one input row per (token, expert) pair
//!   (the served flattening: `n_tokens·8` rows, top_k 1).
//! * `qwen` (opt-in) — Qwen decode: Q4_K 8192×2048, 4096×2048, 2048×4096;
//!   Q6_K 8192×2048, 512×2048; Q6_K lm head 248320×2048, widths 1..=8.
//!
//! Arms: `stock` = `quantized_matmul_ggml` / `quantized_matmul_id_ggml`
//! default routing (process environment applies); `mma_ci` =
//! `quantized_matmul_ggml_column_invariant_mma`; `row_exact` =
//! `ops::row_exact_matvec` with its default schedule (cells record the
//! resolved `nr0` / `nsg`); `row_exact_r<NR0>_s<NSG>` = the same op pinned
//! to NR0 ∈ {1, 2, 4} rows per threadgroup (Q8_0: {1, 2}) and NSG ∈ {1, 2, 4}
//! simdgroups. All but `stock` run on the `dense`, `lmhead` and `qwen` cells only.
//!
//! Cache modes: `warm` reuses one weight buffer (and fixed ids) for every
//! dispatch, matching the peer's test-backend-ops perf; `cold` rotates
//! `min(30, ⌊8 GiB / weight bytes⌋)` weight buffers, one per dispatch.
//! Cold is skipped when a single call already reads ≥ 256 MiB of weights
//! (the lm head: no cache holds it, so cold ≡ warm) or when fewer than two
//! buffers fit the budget; skips are recorded in the metadata.
//!
//! Clock: `DISPATCHES_PER_CB` dispatches per command buffer (one output
//! buffer, so consecutive dispatches serialize like the peer's repeated
//! graph node), `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime` per buffer
//! divided by the dispatch count; `WARMUP` warm-up buffers, then
//! `GEMMA_OP_PARITY_ITERS` (default and minimum 50) measured buffers;
//! median, p10, p90 in ns per dispatch.
//!
//! Output (`GEMMA_OP_PARITY_OUT`, a directory, required): `cells.jsonl`
//! (one object per cell: family, shape, type, n, k, width or n_tokens,
//! arm, cache, median_ns, p10, p90, rotation_buffers, dispatches_per_cb,
//! iters) and `metadata.json` (git head, dirtiness, metallib environment,
//! `HF2Q_HOST_LOCK_HOLDER`, device, environment snapshot, cold skips,
//! pipeline identities of the `mul_mv_mma` kernels).
//!
//! Needs the GPU lane and a quiet host:
//!
//! ```text
//! GEMMA_OP_PARITY_OUT=$DIR HF2Q_HOST_LOCK_HOLDER=$NAME \
//!     cargo bench --locked --bench bench_gemma_decode_op_parity
//! ```

use std::io::Write;

use mlx_native::ops::row_exact_matvec::{
    dispatch_row_exact_matvec, dispatch_row_exact_matvec_scheduled, row_exact_matvec_kernel_names,
    RowExactMatvecSchedule, RowExactMatvecShape, RowExactWeight,
};
use mlx_native::{
    quantized_matmul_ggml, quantized_matmul_ggml_column_invariant_mma, quantized_matmul_id_ggml,
    CommandEncoder, DType, GgmlQuantizedMatmulIdParams, GgmlQuantizedMatmulParams, GgmlType,
    KernelRegistry, MlxBuffer, MlxDevice,
};
use serde_json::{json, Value};

/// One timed implementation of a dense cell.
#[derive(Clone, Copy)]
enum Arm {
    Stock,
    MmaCi,
    /// `ops::row_exact_matvec`; `None` = the default schedule.
    RowExact(RowExactWeight, Option<RowExactMatvecSchedule>),
}

/// Row-exact schedules swept per type: rows per threadgroup × simdgroups.
const ROW_EXACT_NR0: [u32; 3] = [1, 2, 4];
const ROW_EXACT_NSG: [u32; 3] = [1, 2, 4];

const WARMUP: usize = 10;
const MIN_ITERS: usize = 50;
const DISPATCHES_PER_CB: usize = 30;
const COLD_ROTATION: usize = 30;
const COLD_BUDGET_BYTES: u64 = 8 << 30;
/// A call reading at least this many weight bytes cannot be cache resident.
const CACHE_RESIDENT_LIMIT_BYTES: u64 = 256 << 20;

const EXPERTS: u32 = 128;
const TOP_K: u32 = 8;
const HIDDEN: u32 = 2816;
const EXPERT_FF: u32 = 704;
const EXPERT_TOKENS: [u32; 4] = [1, 4, 8, 16];

struct DenseShape {
    ggml_type: GgmlType,
    n: u32,
    k: u32,
}

const DENSE: &[DenseShape] = &[
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 4096,
        k: 2816,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 8192,
        k: 2816,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 2048,
        k: 2816,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 1024,
        k: 2816,
    },
    DenseShape {
        ggml_type: GgmlType::Q6_K,
        n: 2048,
        k: 2816,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 2816,
        k: 4096,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 2816,
        k: 8192,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 2112,
        k: 2816,
    },
    DenseShape {
        ggml_type: GgmlType::Q5_0,
        n: 2816,
        k: 2112,
    },
    DenseShape {
        ggml_type: GgmlType::Q8_0,
        n: 2816,
        k: 2112,
    },
];

const LM_HEAD: DenseShape = DenseShape {
    ggml_type: GgmlType::Q6_K,
    n: 262_144,
    k: 2816,
};

const QWEN_DENSE: &[DenseShape] = &[
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 8192,
        k: 2048,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 4096,
        k: 2048,
    },
    DenseShape {
        ggml_type: GgmlType::Q4_K,
        n: 2048,
        k: 4096,
    },
    DenseShape {
        ggml_type: GgmlType::Q6_K,
        n: 8192,
        k: 2048,
    },
    DenseShape {
        ggml_type: GgmlType::Q6_K,
        n: 512,
        k: 2048,
    },
    DenseShape {
        ggml_type: GgmlType::Q6_K,
        n: 248_320,
        k: 2048,
    },
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cache {
    Warm,
    Cold,
}

impl Cache {
    fn tag(self) -> &'static str {
        match self {
            Cache::Warm => "warm",
            Cache::Cold => "cold",
        }
    }
}

struct Options {
    out_dir: std::path::PathBuf,
    families: Vec<String>,
    iters: usize,
    /// `GEMMA_OP_PARITY_ARMS`: dense/lm-head arm names to run (default all).
    arms: Option<Vec<String>>,
    /// `GEMMA_OP_PARITY_WMAX`: widest dense width (default 8).
    wmax: u32,
    /// `GEMMA_OP_PARITY_CACHES`: `warm`, `cold` or both (default both).
    caches: Vec<String>,
    /// `GEMMA_OP_PARITY_REPEATS`: passes over the arm × width grid per cache
    /// (default 1); cells carry `rep` so interleaved passes expose noise.
    repeats: u32,
    /// `GEMMA_OP_PARITY_COOLDOWN_MS`: idle time before each dense/lm-head cell
    /// (default 0). DRAM-streaming cells such as the lm head heat the package;
    /// a cooldown keeps interleaved arms comparable.
    cooldown: std::time::Duration,
}

fn options() -> Options {
    let out_dir = std::env::var_os("GEMMA_OP_PARITY_OUT")
        .map(std::path::PathBuf::from)
        .expect("set GEMMA_OP_PARITY_OUT to the output directory");
    let families: Vec<String> = std::env::var("GEMMA_OP_PARITY_FAMILIES")
        .unwrap_or_else(|_| "dense,lmhead,experts".into())
        .split(',')
        .map(|family| family.trim().to_string())
        .filter(|family| !family.is_empty())
        .collect();
    for family in &families {
        assert!(
            matches!(family.as_str(), "dense" | "lmhead" | "experts" | "qwen"),
            "unknown family {family:?}; families are dense, lmhead, experts, qwen"
        );
    }
    let iters = std::env::var("GEMMA_OP_PARITY_ITERS")
        .ok()
        .map(|value| value.parse().expect("GEMMA_OP_PARITY_ITERS takes a count"))
        .unwrap_or(MIN_ITERS);
    assert!(
        iters >= MIN_ITERS,
        "GEMMA_OP_PARITY_ITERS must be ≥ {MIN_ITERS}"
    );
    let list = |name: &str| -> Option<Vec<String>> {
        std::env::var(name).ok().map(|value| {
            value
                .split(',')
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect()
        })
    };
    let arms = list("GEMMA_OP_PARITY_ARMS");
    let wmax = std::env::var("GEMMA_OP_PARITY_WMAX")
        .ok()
        .map(|value| value.parse().expect("GEMMA_OP_PARITY_WMAX takes a width"))
        .unwrap_or(8u32);
    assert!(
        (1..=8).contains(&wmax),
        "GEMMA_OP_PARITY_WMAX must be in 1..=8"
    );
    let caches =
        list("GEMMA_OP_PARITY_CACHES").unwrap_or_else(|| vec!["warm".into(), "cold".into()]);
    let repeats = std::env::var("GEMMA_OP_PARITY_REPEATS")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("GEMMA_OP_PARITY_REPEATS takes a count")
        })
        .unwrap_or(1u32)
        .max(1);
    Options {
        out_dir,
        families,
        iters,
        arms,
        wmax,
        caches,
        repeats,
        cooldown: std::time::Duration::from_millis(
            std::env::var("GEMMA_OP_PARITY_COOLDOWN_MS")
                .ok()
                .map(|value| value.parse().expect("GEMMA_OP_PARITY_COOLDOWN_MS takes milliseconds"))
                .unwrap_or(0u64),
        ),
    }
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

fn environment_snapshot() -> Value {
    let mut map = serde_json::Map::new();
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("HF2Q_")
                || key.starts_with("MLX_")
                || key.starts_with("GEMMA_OP_PARITY_")
        })
        .collect();
    vars.sort();
    for (key, value) in vars {
        map.insert(key, Value::String(value));
    }
    Value::Object(map)
}

fn metadata(device: &MlxDevice, options: &Options) -> Value {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let git_head = command_stdout("git", &["-C", manifest, "rev-parse", "HEAD"]);
    let dirty = command_stdout("git", &["-C", manifest, "status", "--porcelain"])
        .map(|status| !status.is_empty());
    json!({
        "schema": "mlx-native.bench_gemma_decode_op_parity.v1",
        "mlx_native_version": env!("CARGO_PKG_VERSION"),
        "git_head": git_head,
        "git_commit_sha_env": std::env::var("GIT_COMMIT_SHA").ok(),
        "working_tree_dirty": dirty,
        "mlx_precompiled_metallib": std::env::var("MLX_PRECOMPILED_METALLIB").ok(),
        "mlx_precompiled_metallib_fcv": std::env::var("MLX_PRECOMPILED_METALLIB_FCV").ok(),
        "host_lock_holder": std::env::var("HF2Q_HOST_LOCK_HOLDER").ok(),
        "device_name": device.name(),
        "macos_product_version": command_stdout("sw_vers", &["-productVersion"]),
        "macos_build_version": command_stdout("sw_vers", &["-buildVersion"]),
        "families": options.families,
        "warmup": WARMUP,
        "iters": options.iters,
        "dispatches_per_cb": DISPATCHES_PER_CB,
        "cold_rotation_max": COLD_ROTATION,
        "cold_budget_bytes": COLD_BUDGET_BYTES,
        "cache_resident_limit_bytes": CACHE_RESIDENT_LIMIT_BYTES,
        "environment": environment_snapshot(),
    })
}

/// Identities of the pipelines the `mma_ci` arm (`mul_mv_mma` rt1; labels
/// carry nsg and the block kind's K) and the `row_exact` arms compiled.
fn pipeline_identities(registry: &KernelRegistry) -> Value {
    let mut map = serde_json::Map::new();
    let mut insert = |label: String| {
        if let Ok(identity) = registry.pipeline_identity(&label) {
            map.insert(
                label,
                serde_json::to_value(identity).expect("pipeline identity serializes"),
            );
        }
    };
    let mut mma_k: Vec<u32> = DENSE
        .iter()
        .chain([&LM_HEAD])
        .chain(QWEN_DENSE)
        .map(|s| s.k)
        .collect();
    mma_k.sort_unstable();
    mma_k.dedup();
    for ty in ["q4_K", "q6_K", "q5_0", "q8_0"] {
        for nt in [1, 2, 4] {
            for nsg in [1, 2, 4, 8, 16, 32] {
                // Only the block kind (q8_0) specializes K; the others pass 0.
                let ks: Vec<u32> = if ty == "q8_0" { mma_k.clone() } else { vec![0] };
                for k in ks {
                    insert(format!(
                        "kernel_mul_mv_mma_{ty}_f32_nt{nt}_rt1|2300:i{nsg}|2304:i{k}"
                    ));
                }
            }
        }
    }
    for name in row_exact_matvec_kernel_names() {
        insert(name.to_string());
    }
    Value::Object(map)
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
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        other => panic!("bench weight generator does not cover {other:?}"),
    }
}

fn matrix_bytes(ggml_type: GgmlType, n: u32, k: u32) -> u64 {
    u64::from(n) * u64::from(k / ggml_type.block_values()) * u64::from(ggml_type.block_bytes())
}

fn weight_pattern(ggml_type: GgmlType, bytes: usize, seed: u64) -> Vec<u8> {
    let mut pattern = vec![0u8; bytes];
    let mut rng = XorShift(seed | 1);
    for chunk in pattern.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    for block in pattern.chunks_exact_mut(ggml_type.block_bytes() as usize) {
        for &offset in half_scale_offsets(ggml_type) {
            // 0x2C00 == 0.0625 in IEEE half.
            block[offset..offset + 2].copy_from_slice(&0x2C00u16.to_le_bytes());
        }
    }
    pattern
}

/// `count` distinct weight buffers with identical contents.
fn weight_buffers(device: &MlxDevice, pattern: &[u8], count: usize) -> Vec<MlxBuffer> {
    (0..count)
        .map(|_| {
            let mut buffer = device
                .alloc_buffer(pattern.len(), DType::U8, vec![pattern.len()])
                .expect("weight buffer");
            buffer
                .as_mut_slice::<u8>()
                .expect("weight bytes")
                .copy_from_slice(pattern);
            buffer
        })
        .collect()
}

fn f32_buffer(device: &MlxDevice, elements: usize, seed: u64) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("f32 buffer");
    let mut rng = XorShift(seed | 1);
    for value in buffer.as_mut_slice::<f32>().expect("f32 values") {
        *value = (rng.below(2001) as f32 - 1000.0) / 4000.0;
    }
    buffer
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("u32 buffer");
    buffer
        .as_mut_slice::<u32>()
        .expect("u32 values")
        .copy_from_slice(values);
    buffer
}

/// Fixed random expert ids, `TOP_K` distinct experts per token.
fn expert_ids(n_tokens: u32, seed: u64) -> Vec<u32> {
    let mut rng = XorShift(seed | 1);
    let mut ids = Vec::with_capacity((n_tokens * TOP_K) as usize);
    for _ in 0..n_tokens {
        let mut experts: Vec<u32> = (0..EXPERTS).collect();
        for slot in 0..TOP_K as usize {
            let pick = slot + rng.below(EXPERTS - slot as u32) as usize;
            experts.swap(slot, pick);
        }
        ids.extend_from_slice(&experts[..TOP_K as usize]);
    }
    ids
}

/// Cold rotation count for one weight buffer size, or the skip reason.
fn cold_rotation(buffer_bytes: u64, bytes_read_per_call: u64) -> Result<usize, String> {
    if bytes_read_per_call >= CACHE_RESIDENT_LIMIT_BYTES {
        return Err(format!(
            "one call reads {bytes_read_per_call} weight bytes (≥ {CACHE_RESIDENT_LIMIT_BYTES}); \
             no cache holds it, so cold ≡ warm"
        ));
    }
    let fit = (COLD_BUDGET_BYTES / buffer_bytes.max(1)) as usize;
    let rotation = fit.min(COLD_ROTATION);
    if rotation < 2 {
        return Err(format!(
            "weight buffer of {buffer_bytes} bytes leaves {rotation} buffers in the \
             {COLD_BUDGET_BYTES}-byte cold budget"
        ));
    }
    Ok(rotation)
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct Stats {
    median_ns: f64,
    p10_ns: f64,
    p90_ns: f64,
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

/// `DISPATCHES_PER_CB` encodes per command buffer; `encode(encoder, i)` gets
/// the dispatch index so cold mode can rotate weights.
fn time_gpu(
    device: &MlxDevice,
    iters: usize,
    mut encode: impl FnMut(&mut CommandEncoder, usize),
) -> Stats {
    let mut sample = || {
        let mut encoder = device.command_encoder().expect("command encoder");
        for dispatch in 0..DISPATCHES_PER_CB {
            encode(&mut encoder, dispatch);
        }
        let (start, end) = encoder
            .commit_wait_with_gpu_time()
            .expect("measured command buffer");
        (end - start) * 1e9 / DISPATCHES_PER_CB as f64
    };
    for _ in 0..WARMUP {
        sample();
    }
    let mut samples: Vec<f64> = (0..iters).map(|_| sample()).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite GPU time"));
    Stats {
        median_ns: percentile(&samples, 0.5),
        p10_ns: percentile(&samples, 0.1),
        p90_ns: percentile(&samples, 0.9),
    }
}

struct Sink {
    cells: std::fs::File,
}

impl Sink {
    fn emit(&mut self, mut cell: Value, stats: &Stats, rotation: usize, iters: usize) {
        let object = cell.as_object_mut().expect("cell object");
        object.insert("median_ns".into(), json!(stats.median_ns));
        object.insert("p10".into(), json!(stats.p10_ns));
        object.insert("p90".into(), json!(stats.p90_ns));
        object.insert("rotation_buffers".into(), json!(rotation));
        object.insert("dispatches_per_cb".into(), json!(DISPATCHES_PER_CB));
        object.insert("iters".into(), json!(iters));
        eprintln!(
            "  {:<8} {:<5} {:>22} w/t={:>2} {:<9} {:<4} median={:>10.0}ns p10={:>10.0} p90={:>10.0}",
            cell["family"].as_str().unwrap_or(""),
            cell["type"].as_str().unwrap_or(""),
            cell["shape"].as_str().unwrap_or(""),
            cell.get("width").or(cell.get("n_tokens")).and_then(Value::as_u64).unwrap_or(0),
            cell["arm"].as_str().unwrap_or(""),
            cell["cache"].as_str().unwrap_or(""),
            stats.median_ns,
            stats.p10_ns,
            stats.p90_ns,
        );
        writeln!(self.cells, "{cell}").expect("write cell");
        self.cells.flush().expect("flush cells");
    }
}

fn type_tag(ggml_type: GgmlType) -> &'static str {
    match ggml_type {
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q5_0 => "q5_0",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q5_K => "q5_K",
        GgmlType::Q6_K => "q6_K",
        other => panic!("no tag for {other:?}"),
    }
}

fn row_exact_weight(ggml_type: GgmlType) -> Option<RowExactWeight> {
    match ggml_type {
        GgmlType::Q8_0 => Some(RowExactWeight::Q8_0),
        GgmlType::Q4_K => Some(RowExactWeight::Q4_K),
        GgmlType::Q6_K => Some(RowExactWeight::Q6_K),
        GgmlType::Q5_0 => Some(RowExactWeight::Q5_0),
        _ => None,
    }
}

/// `stock`, `mma_ci`, then the row-exact arms the type supports.
fn dense_arms(ggml_type: GgmlType) -> Vec<(String, Arm)> {
    let mut arms = vec![
        ("stock".to_string(), Arm::Stock),
        ("mma_ci".to_string(), Arm::MmaCi),
    ];
    if let Some(weight) = row_exact_weight(ggml_type) {
        arms.push(("row_exact".to_string(), Arm::RowExact(weight, None)));
        for nr0 in ROW_EXACT_NR0
            .into_iter()
            .filter(|&nr0| weight.supports_nr0(nr0))
        {
            for nsg in ROW_EXACT_NSG {
                arms.push((
                    format!("row_exact_r{nr0}_s{nsg}"),
                    Arm::RowExact(weight, Some(RowExactMatvecSchedule { nsg, nr0 })),
                ));
            }
        }
    }
    arms
}

// ---------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------

fn run_dense_shape(
    family: &str,
    shape: &DenseShape,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    options: &Options,
    sink: &mut Sink,
    skips: &mut Vec<Value>,
) {
    let bytes = matrix_bytes(shape.ggml_type, shape.n, shape.k);
    let shape_tag = format!("{}x{}", shape.n, shape.k);
    let pattern = weight_pattern(shape.ggml_type, bytes as usize, u64::from(shape.n) ^ 0xd5e);
    let input = f32_buffer(device, 8 * shape.k as usize, 0x1f32);
    let output = device
        .alloc_buffer(
            8 * shape.n as usize * 4,
            DType::F32,
            vec![8 * shape.n as usize],
        )
        .expect("output buffer");
    let cold = cold_rotation(bytes, bytes);
    if let Err(reason) = &cold {
        skips.push(json!({
            "family": family, "shape": shape_tag, "type": type_tag(shape.ggml_type),
            "cache": "cold", "reason": reason,
        }));
    }
    for cache in [Cache::Warm, Cache::Cold] {
        if !options.caches.iter().any(|c| c == cache.tag()) {
            continue;
        }
        let rotation = match (cache, &cold) {
            (Cache::Warm, _) => 1,
            (Cache::Cold, Ok(rotation)) => *rotation,
            (Cache::Cold, Err(_)) => continue,
        };
        let weights = weight_buffers(device, &pattern, rotation);
        let arms: Vec<(String, Arm)> = dense_arms(shape.ggml_type)
            .into_iter()
            .filter(|(name, _)| options.arms.as_ref().is_none_or(|keep| keep.contains(name)))
            .collect();
        for rep in 0..options.repeats {
            for (arm_name, arm) in arms.iter().cloned() {
                for width in 1..=options.wmax {
                    let params = GgmlQuantizedMatmulParams {
                        m: width,
                        n: shape.n,
                        k: shape.k,
                        ggml_type: shape.ggml_type,
                    };
                    let row_shape = RowExactMatvecShape::contiguous(width, shape.n, shape.k);
                    std::thread::sleep(options.cooldown);
                    let stats = time_gpu(device, options.iters, |encoder, dispatch| {
                        let weight = &weights[dispatch % rotation];
                        match arm {
                            Arm::Stock => quantized_matmul_ggml(
                                encoder, registry, device, &input, weight, &output, &params,
                            ),
                            Arm::MmaCi => quantized_matmul_ggml_column_invariant_mma(
                                encoder, registry, device, &input, weight, &output, &params,
                            ),
                            Arm::RowExact(ty, None) => dispatch_row_exact_matvec(
                                encoder,
                                registry,
                                device.metal_device(),
                                ty,
                                weight,
                                &input,
                                &output,
                                &row_shape,
                            ),
                            Arm::RowExact(ty, Some(schedule)) => {
                                dispatch_row_exact_matvec_scheduled(
                                    encoder,
                                    registry,
                                    device.metal_device(),
                                    ty,
                                    weight,
                                    &input,
                                    &output,
                                    &row_shape,
                                    &schedule,
                                )
                            }
                        }
                        .unwrap_or_else(|e| panic!("{arm_name} {shape_tag} m={width}: {e}"));
                    });
                    let mut cell = json!({
                        "family": family,
                        "shape": shape_tag,
                        "type": type_tag(shape.ggml_type),
                        "n": shape.n,
                        "k": shape.k,
                        "width": width,
                        "arm": arm_name,
                        "cache": cache.tag(),
                        "rep": rep,
                    });
                    if let Arm::RowExact(ty, schedule) = arm {
                        // The default arm launches `for_device`'s schedule.
                        let schedule = match schedule {
                            Some(schedule) => schedule,
                            None => RowExactMatvecSchedule::for_device(
                                registry,
                                device.metal_device(),
                                ty,
                                &row_shape,
                            )
                            .expect("row-exact device schedule"),
                        };
                        let object = cell.as_object_mut().expect("cell object");
                        object.insert("nr0".into(), json!(schedule.nr0));
                        object.insert("nsg".into(), json!(schedule.nsg));
                    }
                    sink.emit(cell, &stats, rotation, options.iters);
                }
            }
        }
    }
}

struct ExpertCase {
    label: &'static str,
    ggml_type: GgmlType,
    n: u32,
    k: u32,
    /// `true`: each token's row is broadcast to its pairs (gate_up);
    /// `false`: one input row per (token, expert) pair (down).
    broadcast: bool,
}

const EXPERT_CASES: &[ExpertCase] = &[
    ExpertCase {
        label: "gate_up",
        ggml_type: GgmlType::Q4_K,
        n: 2 * EXPERT_FF,
        k: HIDDEN,
        broadcast: true,
    },
    ExpertCase {
        label: "down",
        ggml_type: GgmlType::Q5_0,
        n: HIDDEN,
        k: EXPERT_FF,
        broadcast: false,
    },
    ExpertCase {
        label: "down",
        ggml_type: GgmlType::Q8_0,
        n: HIDDEN,
        k: EXPERT_FF,
        broadcast: false,
    },
];

fn run_experts(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    options: &Options,
    sink: &mut Sink,
    skips: &mut Vec<Value>,
) {
    for case in EXPERT_CASES {
        let expert_bytes = matrix_bytes(case.ggml_type, case.n, case.k);
        let stack_bytes = expert_bytes * u64::from(EXPERTS);
        let shape_tag = format!(
            "{}x{}x{} top{} {}",
            EXPERTS,
            case.n,
            case.k,
            TOP_K,
            if case.broadcast {
                "broadcast"
            } else {
                "per_pair"
            }
        );
        let pattern = weight_pattern(
            case.ggml_type,
            stack_bytes as usize,
            u64::from(case.n) ^ 0xe7,
        );
        // Distinct experts read by the largest call bound the cold decision
        // for every n_tokens of the case (rotation must not change per cell).
        let max_ids = expert_ids(*EXPERT_TOKENS.last().expect("token counts"), 0x1d5);
        let mut distinct = max_ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let cold = cold_rotation(stack_bytes, expert_bytes * distinct.len() as u64);
        if let Err(reason) = &cold {
            skips.push(json!({
                "family": "experts", "shape": shape_tag, "type": type_tag(case.ggml_type),
                "cache": "cold", "reason": reason,
            }));
        }
        for cache in [Cache::Warm, Cache::Cold] {
            let rotation = match (cache, &cold) {
                (Cache::Warm, _) => 1,
                (Cache::Cold, Ok(rotation)) => *rotation,
                (Cache::Cold, Err(_)) => continue,
            };
            let weights = weight_buffers(device, &pattern, rotation);
            for n_tokens in EXPERT_TOKENS {
                let ids_host = expert_ids(n_tokens, 0x1d5 ^ u64::from(n_tokens));
                let ids = u32_buffer(device, &ids_host);
                let pairs = n_tokens * TOP_K;
                let params = if case.broadcast {
                    GgmlQuantizedMatmulIdParams {
                        n_tokens,
                        top_k: TOP_K,
                        n: case.n,
                        k: case.k,
                        n_experts: EXPERTS,
                        expert_stride: expert_bytes,
                        ggml_type: case.ggml_type,
                    }
                } else {
                    GgmlQuantizedMatmulIdParams {
                        n_tokens: pairs,
                        top_k: 1,
                        n: case.n,
                        k: case.k,
                        n_experts: EXPERTS,
                        expert_stride: expert_bytes,
                        ggml_type: case.ggml_type,
                    }
                };
                let input_rows = if case.broadcast { n_tokens } else { pairs };
                let input = f32_buffer(device, (input_rows * case.k) as usize, 0xe1f32);
                let output_elements = (pairs * case.n) as usize;
                let output = device
                    .alloc_buffer(output_elements * 4, DType::F32, vec![output_elements])
                    .expect("expert output");
                let stats = time_gpu(device, options.iters, |encoder, dispatch| {
                    quantized_matmul_id_ggml(
                        encoder,
                        registry,
                        device,
                        &input,
                        &weights[dispatch % rotation],
                        &ids,
                        &output,
                        &params,
                    )
                    .unwrap_or_else(|e| {
                        panic!("stock {} {shape_tag} t={n_tokens}: {e}", case.label)
                    });
                });
                sink.emit(
                    json!({
                        "family": "experts",
                        "op": case.label,
                        "shape": shape_tag,
                        "type": type_tag(case.ggml_type),
                        "n": case.n,
                        "k": case.k,
                        "n_tokens": n_tokens,
                        "arm": "stock",
                        "cache": cache.tag(),
                    }),
                    &stats,
                    rotation,
                    options.iters,
                );
            }
        }
    }
}

fn main() {
    let options = options();
    std::fs::create_dir_all(&options.out_dir).expect("create GEMMA_OP_PARITY_OUT");
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    let mut meta = metadata(&device, &options);
    let mut sink = Sink {
        cells: std::fs::File::create(options.out_dir.join("cells.jsonl")).expect("cells.jsonl"),
    };
    let mut skips = Vec::new();
    let started = std::time::Instant::now();

    let enabled = |family: &str| options.families.iter().any(|f| f == family);
    if enabled("dense") {
        for shape in DENSE {
            run_dense_shape(
                "dense",
                shape,
                &device,
                &mut registry,
                &options,
                &mut sink,
                &mut skips,
            );
        }
    }
    if enabled("lmhead") {
        run_dense_shape(
            "lmhead",
            &LM_HEAD,
            &device,
            &mut registry,
            &options,
            &mut sink,
            &mut skips,
        );
    }
    if enabled("qwen") {
        for shape in QWEN_DENSE {
            run_dense_shape(
                "qwen",
                shape,
                &device,
                &mut registry,
                &options,
                &mut sink,
                &mut skips,
            );
        }
    }
    if enabled("experts") {
        run_experts(&device, &mut registry, &options, &mut sink, &mut skips);
    }

    let object = meta.as_object_mut().expect("metadata object");
    object.insert("cold_skips".into(), Value::Array(skips));
    object.insert("pipeline_identities".into(), pipeline_identities(&registry));
    object.insert("elapsed_s".into(), json!(started.elapsed().as_secs_f64()));
    std::fs::write(
        options.out_dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).expect("metadata json"),
    )
    .expect("write metadata.json");
    eprintln!("wrote {}", options.out_dir.display());
}
