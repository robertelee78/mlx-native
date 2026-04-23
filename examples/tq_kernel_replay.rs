//! TQ kernel replay binary for ADR-007 C-1-unlock harness fix.
//!
//! Runs the flash_attn_vec_tq / flash_attn_vec kernels against captured inputs
//! and compares to a CPU reference SDPA computed from the same TQ-packed data.
//!
//! Fixes applied in C-1-unlock:
//!   D1 - encoder.memory_barrier() inserted at 3 sites (mirroring forward_mlx.rs:1429-1431,
//!        1441-1446, 1477-1480).
//!   D2 - Variation C replaced with true dense control: flash_attn_vec on dequantized F32 K/V.
//!   D3 - Canary in-range: --canary in-range mutates k_norms[head=0, pos=10] *= 2.0.
//!   D4 - Raw sdpa_out .bin written per variation alongside metrics JSON.
//!   D5 - kv_seq_len=23 accepted from manifest; CPU reference loops 0..kvl.
//!
//! Usage:
//!   cargo run --release --example tq_kernel_replay -- \
//!     --manifest /tmp/cfa-20260422-C1-unlock/manifest.json \
//!     --variation A \
//!     [--canary in-range] \
//!     --out /tmp/cfa-20260422-C1-unlock/out/claude/A
//!
//! Variations:
//!   A  Full production path: forward-FWHT(Q) + TQ kernel + inverse-FWHT(output)
//!   B  FWHT-disabled: skip both FWHT dispatches; pass Q as-is to TQ kernel
//!   C  Dense control: flash_attn_vec (F32 K/V, natural basis) — no FWHT on either side
//!
//! Canary (--canary in-range): k_norms[head=0 * kv_capacity + pos=10] *= 2.0 before H2D.
//!   In-range mutation (pos=10 < kv_seq_len=23); expected nrmse_delta vs A baseline > 0.01.
//!
//! Exit codes:
//!   0  Success
//!   1  Argument / IO error
//!   2  GPU dispatch error or NaN/Inf in output

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

use mlx_native::ops::flash_attn_vec::{self, FlashAttnVecParams};
use mlx_native::ops::flash_attn_vec_tq::{self, FlashAttnVecTqParams};
use mlx_native::ops::fwht_standalone;
use mlx_native::turboquant::{fwht_inplace, CODEBOOK_4BIT};
use mlx_native::{DType, KernelRegistry, MlxDevice};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// iter-5 pre-registered NRMSE band (catalog #11: never widen after measurement)
//
// These constants are COMMITTED here, BEFORE any measurement is run.
// If any sweep point returns nrmse outside [LOWER, UPPER], the binary panics
// with exit code 2 and reports BAND_PRE_FALSIFIED — NO band edits permitted.
// Violating this rule is catalog #11 (post-measurement widening, iter-4 HIGH-1 defect).
// ---------------------------------------------------------------------------

/// Lower bound of pre-registered iter-5 NRMSE band.
/// Catalog #11: pre-registered, no post-measurement widening.
const NRMSE_BAND_LOWER: f32 = 0.05;

/// Upper bound of pre-registered iter-5 NRMSE band.
/// Catalog #11: pre-registered, no post-measurement widening.
const NRMSE_BAND_UPPER: f32 = 0.35;

// ---------------------------------------------------------------------------
// CLI parsing (no clap dep — simple std::env)
// ---------------------------------------------------------------------------

/// Oracle mode: what reference to compare the TQ GPU output against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OracleMode {
    /// Dequant oracle only (C-1-unlock behavior, default for backward compat).
    Dequant,
    /// Independent-floor oracle only: dense flash_attn_vec on pre-quant F32 K/V.
    IndependentFloor,
    /// Both oracles — C-2 happy path; emits two nrmse columns.
    Both,
}

/// Replay mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplayMode {
    /// Single-step: load a manifest and replay it (backward-compat, default).
    Singlestep,
    /// Multi-step: synthesize K/V from seed and replay at 4 canonical positions.
    Multistep,
    /// Production-faithful v2: iter-5 controlled sweep with pre-registered band,
    /// subprocess regression gates, and single-seed deterministic draws.
    ProductionFaithful,
}

struct Args {
    manifest: Option<PathBuf>,
    variation: Variation,
    canary: CanaryMode,
    out: PathBuf,
    oracle: OracleMode,
    mode: ReplayMode,
    seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanaryMode {
    None,
    InRange,
    OutOfRange, // legacy: k_norms at positions >= kv_seq_len set to 1e9
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variation {
    A,
    B,
    C,
}

impl std::fmt::Display for Variation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variation::A => write!(f, "A"),
            Variation::B => write!(f, "B"),
            Variation::C => write!(f, "C (dense control)"),
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut manifest: Option<PathBuf> = None;
    let mut variation: Option<Variation> = None;
    let mut canary = CanaryMode::None;
    let mut out: Option<PathBuf> = None;
    let mut oracle = OracleMode::Dequant;
    let mut mode = ReplayMode::Singlestep;
    let mut seed: u64 = 0x00C2_5EED;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest = Some(PathBuf::from(argv.get(i).ok_or("--manifest needs a value")?));
            }
            "--variation" => {
                i += 1;
                variation = Some(match argv.get(i).map(|s| s.as_str()) {
                    Some("A") => Variation::A,
                    Some("B") => Variation::B,
                    Some("C") => Variation::C,
                    other => return Err(format!("unknown variation {:?}; expected A, B, or C", other)),
                });
            }
            "--canary" => {
                // Accept "--canary in-range", "--canary out-of-range", or bare "--canary" (= in-range)
                if let Some(next) = argv.get(i + 1) {
                    match next.as_str() {
                        "in-range" => {
                            canary = CanaryMode::InRange;
                            i += 1;
                        }
                        "out-of-range" => {
                            canary = CanaryMode::OutOfRange;
                            i += 1;
                        }
                        s if !s.starts_with('-') => {
                            // Legacy: numeric value like "1e9" → treat as out-of-range
                            canary = CanaryMode::OutOfRange;
                            i += 1;
                        }
                        _ => {
                            // Next arg is a flag — bare --canary defaults to in-range
                            canary = CanaryMode::InRange;
                        }
                    }
                } else {
                    canary = CanaryMode::InRange;
                }
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(argv.get(i).ok_or("--out needs a value")?));
            }
            "--oracle" => {
                i += 1;
                oracle = match argv.get(i).map(|s| s.as_str()) {
                    Some("dequant") => OracleMode::Dequant,
                    Some("independent-floor") => OracleMode::IndependentFloor,
                    Some("both") => OracleMode::Both,
                    other => return Err(format!("unknown --oracle {:?}; expected dequant, independent-floor, or both", other)),
                };
            }
            "--singlestep" => {
                mode = ReplayMode::Singlestep;
            }
            "--multistep" => {
                mode = ReplayMode::Multistep;
            }
            "--production-faithful" => {
                mode = ReplayMode::ProductionFaithful;
            }
            "--seed" => {
                i += 1;
                let s = argv.get(i).ok_or("--seed needs a value")?;
                seed = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                    u64::from_str_radix(hex, 16)
                        .map_err(|e| format!("--seed hex parse error: {}", e))?
                } else {
                    s.parse::<u64>()
                        .map_err(|e| format!("--seed decimal parse error: {}", e))?
                };
            }
            other => return Err(format!("unknown argument: {}", other)),
        }
        i += 1;
    }

    // Validate: singlestep requires --manifest; multistep and production-faithful do not.
    if mode == ReplayMode::Singlestep && manifest.is_none() {
        return Err("--singlestep (or default) mode requires --manifest".into());
    }

    Ok(Args {
        manifest,
        variation: variation.unwrap_or(Variation::A),
        canary,
        out: out.ok_or("--out is required")?,
        oracle,
        mode,
        seed,
    })
}

// ---------------------------------------------------------------------------
// Manifest schema — supports the instrumenter's C-1-unlock format.
//
// The instrumenter manifest uses `dump_paths` (not `inputs`) and has no
// `compact_sources` section. Compact K/V for CPU reference is derived
// in-memory by slicing rows 0..kvl from the padded buffers.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ManifestParams {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    softcap: f32,
    ring_start: u32,
}

/// Paths section — accepts both the new `dump_paths` key (instrumenter format)
/// and the old `inputs` key (C-1 format) via `#[serde(alias)]`.
#[derive(Debug, Deserialize)]
struct ManifestPaths {
    #[serde(alias = "k_packed_post_quant", alias = "k_packed_padded")]
    k_packed_padded: String,
    #[serde(alias = "v_packed_post_quant", alias = "v_packed_padded")]
    v_packed_padded: String,
    #[serde(alias = "k_norms_post_quant", alias = "k_norms_padded")]
    k_norms_padded: String,
    #[serde(alias = "v_norms_post_quant", alias = "v_norms_padded")]
    v_norms_padded: String,
    q_natural: String,
    // Optional legacy canary files (old format only)
    #[serde(default)]
    k_norms_canary: String,
    #[serde(default)]
    v_norms_canary: String,
    /// Optional pre-quant F32 K dump (from HF2Q_DUMP_PRE_QUANT=1).
    /// When both k_pre_quant and v_pre_quant are present, the independent-floor oracle is available.
    /// Layout: [nkv, hd] F32 little-endian (current token only; NOT the full ring buffer).
    #[serde(default)]
    k_pre_quant: Option<String>,
    #[serde(default)]
    v_pre_quant: Option<String>,
}

/// Top-level manifest. Accepts both:
///   - New format: `dump_paths` key (instrumenter C-1-unlock)
///   - Old format: `inputs` key (C-1 harness)
#[derive(Debug, Deserialize)]
struct Manifest {
    params: ManifestParams,
    /// New instrumenter format uses `dump_paths`; old harness format uses `inputs`.
    #[serde(alias = "inputs")]
    dump_paths: ManifestPaths,
    /// Old format only — if absent, compact sources are derived in-memory.
    #[serde(default)]
    compact_sources: Option<LegacyCompactSources>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
struct LegacyCompactSources {
    k_packed_compact: String,
    v_packed_compact: String,
    k_norms_compact: String,
    v_norms_compact: String,
}

// ---------------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct PerHeadDiff {
    head: usize,
    max_abs_diff: f32,
}

#[derive(Debug, Serialize)]
struct ReplayMetrics {
    variation: String,
    canary: String,
    ran_at: String,
    /// Primary dequant oracle nrmse (nrmse(gpu_out, cpu_sdpa_from_dequant)).
    /// Alias for backward compatibility: was `nrmse` in C-1-unlock output.
    #[serde(rename = "dequant_oracle_nrmse")]
    nrmse: f64,
    max_abs_diff: f32,
    per_head_max_abs_diff: Vec<PerHeadDiff>,
    any_nan_inf_in_gpu_output: bool,
    exit_status: String,
    bin_path: String,
    /// Independent-floor oracle nrmse: nrmse(gpu_out, flash_attn_vec on pre-quant F32 K/V).
    /// None when pre-quant paths are absent or --oracle dequant.
    independent_floor_nrmse: Option<f64>,
}

// ---------------------------------------------------------------------------
// CPU helpers (mirror test_flash_attn_vec_tq.rs)
// ---------------------------------------------------------------------------

fn boundaries_4bit() -> [f32; 15] {
    let mut b = [0.0f32; 15];
    for i in 0..15 {
        b[i] = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i + 1]) / 2.0;
    }
    b
}

fn nearest_centroid_4bit(value: f32) -> u8 {
    let boundaries = boundaries_4bit();
    let mut idx: u8 = 0;
    for &b in &boundaries {
        if value > b {
            idx += 1;
        }
    }
    idx
}

/// Quantize a head vector into nibble-packed format (mirrors test file).
fn nibble_quantize(x: &[f32], head_dim: usize) -> (Vec<u8>, f32) {
    let mut rotated = x.to_vec();
    fwht_inplace(&mut rotated).unwrap();

    let norm: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-30 {
        return (vec![0u8; head_dim / 2], 0.0);
    }

    let inv_norm = 1.0 / norm;
    let scale = (head_dim as f32).sqrt();

    let mut packed = vec![0u8; head_dim / 2];
    for c in 0..head_dim {
        let scaled = rotated[c] * inv_norm * scale;
        let idx = nearest_centroid_4bit(scaled);
        let byte_idx = c / 2;
        if c % 2 == 0 {
            packed[byte_idx] = idx & 0xF;
        } else {
            packed[byte_idx] |= (idx & 0xF) << 4;
        }
    }

    (packed, norm)
}

/// Dequantize from nibble-packed format (mirrors test file).
fn nibble_dequantize(packed: &[u8], norm: f32, head_dim: usize) -> Vec<f32> {
    let inv_scale = 1.0 / (head_dim as f32).sqrt();
    let mut rotated = Vec::with_capacity(head_dim);

    for c in 0..head_dim {
        let byte_idx = c / 2;
        let idx = if c % 2 == 0 {
            (packed[byte_idx] & 0xF) as usize
        } else {
            ((packed[byte_idx] >> 4) & 0xF) as usize
        };
        rotated.push(CODEBOOK_4BIT[idx] * inv_scale * norm);
    }

    fwht_inplace(&mut rotated).unwrap();
    rotated
}

/// CPU SDPA reference (mirrors test_flash_attn_vec_tq.rs cpu_sdpa).
///
/// Q: flat [num_heads * head_dim] F32 (natural basis)
/// k_dequant: [num_kv_heads * kvl_logical] entries of [head_dim] each,
///            indexed in CHRONOLOGICAL order (pos 0 = oldest, pos kvl_logical-1 = newest).
/// v_dequant: same layout as k_dequant.
/// kvl_logical: number of valid chronological positions (= min(abs_pos+1, kv_capacity)).
/// kv_capacity: physical ring buffer capacity (used only for ring_start modulo).
/// mask_type: 0=none/dense (attend all), 1=causal (all <= current step), 2=sliding_window.
/// sliding_window: only last sliding_window chronological positions attend (mask_type=2 only).
/// ring_start: chronological position 0 maps to physical row ring_start. For the dequant
///             oracle path, k_dequant is already compact (chronological order), so ring_start
///             does NOT remap into k_dequant — it is passed here for interface symmetry and
///             used only by the independent-floor path where physical layout matters.
///             In the dequant oracle, iterate p in 0..kvl_logical directly.
/// softcap: logit soft-capping. When > 0: score = softcap * tanh(score * scale / softcap).
///          When 0: score *= scale (standard).
///
/// Returns: flat [num_heads * head_dim] F32
fn cpu_sdpa(
    q: &[f32],
    k_dequant: &[Vec<f32>],
    v_dequant: &[Vec<f32>],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kvl_logical: usize,
    kv_capacity: usize,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    _ring_start: u32,  // unused in dequant oracle path (k_dequant is already chronological)
    softcap: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * head_dim];
    let heads_per_kv = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_offset = h * head_dim;

        let mut scores: Vec<f32> = Vec::with_capacity(kvl_logical);
        // Bitmask: which chronological positions are masked in.
        // For sliding (mask_type=2): only last sliding_window positions attend.
        // For causal (mask_type=1) and none (mask_type=0): all positions attend.
        let first_valid: usize = if mask_type == 2 {
            let sw = sliding_window as usize;
            if kvl_logical > sw { kvl_logical - sw } else { 0 }
        } else {
            0
        };

        for p in 0..kvl_logical {
            if p < first_valid {
                // Masked out — push NEG_INFINITY so softmax weight → 0.
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let mut dot = 0.0f32;
            for c in 0..head_dim {
                dot += q[q_offset + c] * k_dequant[kv_h * kvl_logical + p][c];
            }
            let score = if softcap > 0.0 {
                softcap * (dot * scale / softcap).tanh()
            } else {
                dot * scale
            };
            scores.push(score);
        }

        // Online softmax: ignore -inf entries (masked positions).
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores
            .iter()
            .map(|&s| if s == f32::NEG_INFINITY { 0.0f32 } else { (s - max_score).exp() })
            .collect();
        let sum: f32 = exp_scores.iter().sum();
        if sum > 0.0 {
            for e in &mut exp_scores {
                *e /= sum;
            }
        }

        let o_offset = h * head_dim;
        for p in 0..kvl_logical {
            let w = exp_scores[p];
            if w == 0.0 {
                continue;
            }
            for c in 0..head_dim {
                output[o_offset + c] += w * v_dequant[kv_h * kvl_logical + p][c];
            }
        }
    }

    // kv_capacity is retained as a parameter for interface symmetry with the
    // independent-floor oracle path; suppress the unused-variable warning.
    let _ = kv_capacity;

    output
}

// ---------------------------------------------------------------------------
// Load binary files as typed slices
// ---------------------------------------------------------------------------

fn load_f32(path: &str) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    assert!(bytes.len() % 4 == 0, "file {} is not 4-byte aligned", path);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn load_u8(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e))
}

// ---------------------------------------------------------------------------
// Compute diff metrics
// ---------------------------------------------------------------------------

fn compute_metrics(
    cpu_ref: &[f32],
    gpu_out: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> (f64, f32, Vec<PerHeadDiff>) {
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut global_max = 0.0f32;
    let mut per_head = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let mut head_max = 0.0f32;
        for c in 0..head_dim {
            let i = h * head_dim + c;
            let diff = (cpu_ref[i] - gpu_out[i]).abs();
            if diff > head_max {
                head_max = diff;
            }
            if diff > global_max {
                global_max = diff;
            }
            sum_sq_diff += (diff as f64) * (diff as f64);
            sum_sq_ref += (cpu_ref[i] as f64) * (cpu_ref[i] as f64);
        }
        per_head.push(PerHeadDiff {
            head: h,
            max_abs_diff: head_max,
        });
    }

    let nrmse = if sum_sq_ref > 0.0 {
        (sum_sq_diff / sum_sq_ref).sqrt()
    } else {
        0.0
    };

    (nrmse, global_max, per_head)
}

// ---------------------------------------------------------------------------
// Derive compact K/V from padded buffers by slicing rows 0..kvl
//
// Padded layout:  [nkv, kv_capacity, hd/2] u8 — stride h*kv_capacity*(hd/2) + pos*(hd/2)
// Compact layout: [nkv, kvl, hd/2] u8         — stride h*kvl*(hd/2) + pos*(hd/2)
// ---------------------------------------------------------------------------

fn compact_from_padded_u8(
    padded: &[u8],
    nkv: usize,
    kv_capacity: usize,
    kvl: usize,
    hd: usize,
) -> Vec<u8> {
    let half_hd = hd / 2;
    let mut compact = vec![0u8; nkv * kvl * half_hd];
    for kv_h in 0..nkv {
        for pos in 0..kvl {
            let src_off = kv_h * kv_capacity * half_hd + pos * half_hd;
            let dst_off = kv_h * kvl * half_hd + pos * half_hd;
            compact[dst_off..dst_off + half_hd]
                .copy_from_slice(&padded[src_off..src_off + half_hd]);
        }
    }
    compact
}

fn compact_from_padded_f32(
    padded: &[f32],
    nkv: usize,
    kv_capacity: usize,
    kvl: usize,
) -> Vec<f32> {
    let mut compact = vec![0.0f32; nkv * kvl];
    for kv_h in 0..nkv {
        for pos in 0..kvl {
            let src_off = kv_h * kv_capacity + pos;
            let dst_off = kv_h * kvl + pos;
            compact[dst_off] = padded[src_off];
        }
    }
    compact
}

// ---------------------------------------------------------------------------
// Core replay logic
// ---------------------------------------------------------------------------

fn run_variation(
    manifest: &Manifest,
    variation: Variation,
    canary: CanaryMode,
    oracle_mode: OracleMode,
    out_path: &PathBuf,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> ReplayMetrics {
    let p = &manifest.params;
    let paths = &manifest.dump_paths;

    let nh = p.num_heads as usize;
    let nkv = p.num_kv_heads as usize;
    let hd = p.head_dim as usize;
    let kvl = p.kv_seq_len as usize; // 23 in C-1-unlock
    let kv_capacity = p.kv_capacity as usize;

    // --- Load padded inputs ---
    let q_natural: Vec<f32> = load_f32(&paths.q_natural);
    assert_eq!(q_natural.len(), nh * hd, "q_natural size mismatch");

    let k_packed_padded: Vec<u8> = load_u8(&paths.k_packed_padded);
    let v_packed_padded: Vec<u8> = load_u8(&paths.v_packed_padded);
    assert_eq!(k_packed_padded.len(), nkv * kv_capacity * (hd / 2),
        "k_packed_padded size mismatch: expected {} got {}",
        nkv * kv_capacity * (hd / 2), k_packed_padded.len());
    assert_eq!(v_packed_padded.len(), nkv * kv_capacity * (hd / 2));

    let k_norms_padded_base: Vec<f32> = load_f32(&paths.k_norms_padded);
    let v_norms_padded_base: Vec<f32> = load_f32(&paths.v_norms_padded);
    assert_eq!(k_norms_padded_base.len(), nkv * kv_capacity);
    assert_eq!(v_norms_padded_base.len(), nkv * kv_capacity);

    // --- Derive compact K/V (rows 0..kvl) for CPU reference ---
    // New instrumenter format has no compact_sources — derive in-memory by slicing.
    // Legacy format may have compact_sources on disk.
    let (k_packed_compact, v_packed_compact, k_norms_compact, v_norms_compact) =
        if let Some(ref cs) = manifest.compact_sources {
            if !cs.k_packed_compact.is_empty() {
                // Legacy: load from disk
                let kp = load_u8(&cs.k_packed_compact);
                let vp = load_u8(&cs.v_packed_compact);
                let kn = load_f32(&cs.k_norms_compact);
                let vn = load_f32(&cs.v_norms_compact);
                (kp, vp, kn, vn)
            } else {
                // Empty legacy struct — derive in-memory
                let kp = compact_from_padded_u8(&k_packed_padded, nkv, kv_capacity, kvl, hd);
                let vp = compact_from_padded_u8(&v_packed_padded, nkv, kv_capacity, kvl, hd);
                let kn = compact_from_padded_f32(&k_norms_padded_base, nkv, kv_capacity, kvl);
                let vn = compact_from_padded_f32(&v_norms_padded_base, nkv, kv_capacity, kvl);
                (kp, vp, kn, vn)
            }
        } else {
            // No compact_sources key — instrumenter format, derive in-memory
            let kp = compact_from_padded_u8(&k_packed_padded, nkv, kv_capacity, kvl, hd);
            let vp = compact_from_padded_u8(&v_packed_padded, nkv, kv_capacity, kvl, hd);
            let kn = compact_from_padded_f32(&k_norms_padded_base, nkv, kv_capacity, kvl);
            let vn = compact_from_padded_f32(&v_norms_padded_base, nkv, kv_capacity, kvl);
            (kp, vp, kn, vn)
        };

    assert_eq!(k_packed_compact.len(), nkv * kvl * (hd / 2));
    assert_eq!(v_packed_compact.len(), nkv * kvl * (hd / 2));
    assert_eq!(k_norms_compact.len(), nkv * kvl);
    assert_eq!(v_norms_compact.len(), nkv * kvl);

    // --- P2 canary symmetry fix: pre-mutate compact norms BEFORE building k_dequant ---
    // When canary=InRange, both the GPU path AND the dequant CPU reference must see the
    // mutation at (head=0, pos=10). We apply it here to k_norms_compact so that k_dequant
    // is rebuilt from the mutated norms. This produces a symmetric canary: both oracle and
    // kernel see the 2x norm at head=0/pos=10, so nrmse returns to the baseline ~5.1e-5.
    //
    // To recover the ASYMMETRIC (C-1-unlock) behavior and reproduce ~0.111 nrmse, set the
    // env var HF2Q_REPLAY_CANARY_ASYMMETRIC=1. This skips the compact-norm mutation so the
    // CPU oracle sees the unmutated norm while the GPU sees the 2x version.
    // P2 canary asymmetric debug flag: set HF2Q_REPLAY_CANARY_ASYMMETRIC=1 to reproduce
    // C-1-unlock's 0.111 nrmse (one-sided mutation: GPU sees 2x, CPU oracle does not).
    // Expected: symmetric run → nrmse ≤ 1e-4; asymmetric → ~0.111.
    let canary_asymmetric_mode =
        std::env::var("HF2Q_REPLAY_CANARY_ASYMMETRIC").is_ok_and(|v| v == "1");
    let mut k_norms_compact = k_norms_compact; // make mutable
    if canary == CanaryMode::InRange && !canary_asymmetric_mode {
        // Symmetric fix: also mutate compact norm so CPU reference is consistent.
        let compact_canary_idx = 0 * kvl + 10;
        if compact_canary_idx < k_norms_compact.len() {
            let old_val = k_norms_compact[compact_canary_idx];
            k_norms_compact[compact_canary_idx] *= 2.0;
            eprintln!(
                "[canary symmetric] k_norms_compact[head=0, pos=10] *= 2.0: {} → {}",
                old_val, k_norms_compact[compact_canary_idx]
            );
        }
    } else if canary == CanaryMode::InRange {
        eprintln!("[canary ASYMMETRIC] HF2Q_REPLAY_CANARY_ASYMMETRIC=1: skipping compact norm mutation (reproduces C-1-unlock 0.111 nrmse)");
    }

    // --- Compute CPU reference: dequantize TQ-packed (kvl rows) → natural-basis K/V ---
    // CPU reference is the same for all variations (A, B, C): natural-basis SDPA from TQ dequant.
    // NOTE: uses k_norms_compact AFTER the canary mutation above (symmetric fix).
    let mut k_dequant: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl);
    let mut v_dequant: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl);

    for kv_h in 0..nkv {
        for pos in 0..kvl {
            let packed_offset = (kv_h * kvl + pos) * (hd / 2);
            let norm_offset = kv_h * kvl + pos;

            let k_vec = nibble_dequantize(
                &k_packed_compact[packed_offset..packed_offset + hd / 2],
                k_norms_compact[norm_offset],
                hd,
            );
            k_dequant.push(k_vec);

            let v_vec = nibble_dequantize(
                &v_packed_compact[packed_offset..packed_offset + hd / 2],
                v_norms_compact[norm_offset],
                hd,
            );
            v_dequant.push(v_vec);
        }
    }

    // CPU SDPA in natural basis (same reference for all variations A/B/C)
    let cpu_ref = cpu_sdpa(
        &q_natural,
        &k_dequant,
        &v_dequant,
        nh,
        nkv,
        hd,
        kvl,
        kv_capacity,
        p.scale,
        p.mask_type,
        p.sliding_window,
        p.ring_start,
        p.softcap,
    );

    // --- Prepare norms with optional canary mutation (GPU path) ---
    // Start from the padded baseline norms
    let mut k_norms_gpu: Vec<f32> = k_norms_padded_base.clone();
    let mut v_norms_gpu: Vec<f32> = v_norms_padded_base.clone();

    match canary {
        CanaryMode::None => {
            // No mutation — use baseline norms as-is
        }
        CanaryMode::InRange => {
            // D3: in-range canary — mutate k_norms at (head=0, pos=10) in the GPU buffer.
            // pos=10 is within kv_seq_len=23, so the kernel provably reads this position.
            // Mutation: scale norm by 2x → dequantized K[h=0, pos=10, :] magnitudes ~2x.
            // Mirror canary_spec: k_norms_padded[0 * kv_capacity + 10] *= 2.0
            let canary_idx = 0 * kv_capacity + 10;
            k_norms_gpu[canary_idx] *= 2.0;
            eprintln!(
                "[canary in-range GPU] k_norms[head=0, pos=10] *= 2.0 → new value = {}",
                k_norms_gpu[canary_idx]
            );
        }
        CanaryMode::OutOfRange => {
            // Legacy out-of-range canary: positions >= kvl set to 1e9.
            // If manifest has old canary files, load them; otherwise construct in-memory.
            if !paths.k_norms_canary.is_empty() && !paths.v_norms_canary.is_empty() {
                k_norms_gpu = load_f32(&paths.k_norms_canary);
                v_norms_gpu = load_f32(&paths.v_norms_canary);
                assert_eq!(k_norms_gpu.len(), nkv * kv_capacity);
                assert_eq!(v_norms_gpu.len(), nkv * kv_capacity);
            } else {
                for kv_h in 0..nkv {
                    for pos in kvl..kv_capacity {
                        k_norms_gpu[kv_h * kv_capacity + pos] = 1e9;
                        v_norms_gpu[kv_h * kv_capacity + pos] = 1e9;
                    }
                }
            }
        }
    }

    // --- GPU buffer allocation ---
    // Q: [nh, 1, hd] F32 — production shape
    let mut q_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc Q");
    q_buf.as_mut_slice::<f32>().expect("write Q")[..nh * hd]
        .copy_from_slice(&q_natural);

    // K/V packed: [nkv, kv_capacity, hd/2] u8 (used by A/B)
    let k_packed_bytes = nkv * kv_capacity * (hd / 2);
    let v_packed_bytes = nkv * kv_capacity * (hd / 2);

    let mut k_packed_buf = device
        .alloc_buffer(k_packed_bytes, DType::U8, vec![nkv, kv_capacity, hd / 2])
        .expect("alloc K packed");
    k_packed_buf.as_mut_slice::<u8>().expect("write K packed")
        .copy_from_slice(&k_packed_padded);

    let mut v_packed_buf = device
        .alloc_buffer(v_packed_bytes, DType::U8, vec![nkv, kv_capacity, hd / 2])
        .expect("alloc V packed");
    v_packed_buf.as_mut_slice::<u8>().expect("write V packed")
        .copy_from_slice(&v_packed_padded);

    // Norms: [nkv, kv_capacity] f32 (includes canary mutation if active)
    let norms_bytes = nkv * kv_capacity * 4;

    let mut k_norms_buf = device
        .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kv_capacity])
        .expect("alloc K norms");
    k_norms_buf.as_mut_slice::<f32>().expect("write K norms")
        .copy_from_slice(&k_norms_gpu);

    let mut v_norms_buf = device
        .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kv_capacity])
        .expect("alloc V norms");
    v_norms_buf.as_mut_slice::<f32>().expect("write V norms")
        .copy_from_slice(&v_norms_gpu);

    // Output buffer: [nh, 1, hd] F32
    let output_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc output");

    // Tmp buffer for TQ SDPA kernel
    let tmp_bytes_tq = flash_attn_vec_tq::tmp_buffer_bytes(p.num_heads, p.head_dim);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes_tq, DType::F32, vec![tmp_bytes_tq / 4])
        .expect("alloc tmp");

    // --- TQ SDPA params from manifest ---
    let tq_params = FlashAttnVecTqParams {
        num_heads: p.num_heads,
        num_kv_heads: p.num_kv_heads,
        head_dim: p.head_dim,
        kv_seq_len: p.kv_seq_len,
        kv_capacity: p.kv_capacity,
        scale: p.scale,
        mask_type: p.mask_type,
        sliding_window: p.sliding_window,
        softcap: p.softcap,
        ring_start: p.ring_start,
    };

    // --- Dispatch ---
    let mut encoder = device.command_encoder().expect("command_encoder");

    match variation {
        Variation::A => {
            // Mirror forward_mlx.rs:1429-1431 — RAW on q_buf before in-place forward FWHT
            encoder.memory_barrier(); // BARRIER 1 (D1): before forward FWHT on Q

            // Forward FWHT on Q (in-place) — mirrors forward_mlx.rs:1433-1437
            fwht_standalone::dispatch_fwht_f32(
                &mut encoder,
                registry,
                device.metal_device(),
                &q_buf,
                p.num_heads,
                p.head_dim,
            )
            .expect("FWHT forward-Q dispatch");

            // Mirror forward_mlx.rs:1441-1446 — publish Q (post-FWHT) + packed K/V + norms
            encoder.memory_barrier(); // BARRIER 2 (D1): before TQ SDPA

            // TQ SDPA kernel — mirrors forward_mlx.rs:1464-1474
            flash_attn_vec_tq::flash_attn_vec_tq(
                &mut encoder,
                registry,
                device,
                &q_buf,
                &k_packed_buf,
                &k_norms_buf,
                &v_packed_buf,
                &v_norms_buf,
                &output_buf,
                &tmp_buf,
                &tq_params,
            )
            .expect("flash_attn_vec_tq dispatch");

            // Mirror forward_mlx.rs:1477-1480 — RAW on sdpa_out before in-place inverse FWHT
            encoder.memory_barrier(); // BARRIER 3 (D1): before inverse FWHT on output

            // Inverse FWHT on output (in-place) — mirrors forward_mlx.rs:1481-1485
            fwht_standalone::dispatch_fwht_f32(
                &mut encoder,
                registry,
                device.metal_device(),
                &output_buf,
                p.num_heads,
                p.head_dim,
            )
            .expect("FWHT inverse-output dispatch");
        }

        Variation::B => {
            // FWHT-disabled: pass Q in natural basis; no FWHT on either side.
            // Only barrier_2 equivalent: publish packed K/V + norms before kernel reads.
            //
            // Mirror forward_mlx.rs:1441-1446 — publish packed K/V + norms before TQ SDPA
            encoder.memory_barrier(); // BARRIER 1 of B (D1): before TQ SDPA

            flash_attn_vec_tq::flash_attn_vec_tq(
                &mut encoder,
                registry,
                device,
                &q_buf,
                &k_packed_buf,
                &k_norms_buf,
                &v_packed_buf,
                &v_norms_buf,
                &output_buf,
                &tmp_buf,
                &tq_params,
            )
            .expect("flash_attn_vec_tq dispatch (no FWHT)");
            // No inverse FWHT — output remains in rotated domain.
        }

        Variation::C => {
            // D2: Dense control — flash_attn_vec on dequantized F32 K/V.
            // Natural basis on both sides (no FWHT on Q or output).
            // Allocate F32 dense K/V: [nkv, kv_capacity, hd]; fill 0..kvl from k_dequant/v_dequant.
            let dense_kv_bytes = nkv * kv_capacity * hd * 4;
            let mut k_dense_buf = device
                .alloc_buffer(dense_kv_bytes, DType::F32, vec![nkv, kv_capacity, hd])
                .expect("alloc K dense");
            let mut v_dense_buf = device
                .alloc_buffer(dense_kv_bytes, DType::F32, vec![nkv, kv_capacity, hd])
                .expect("alloc V dense");

            {
                let k_slice = k_dense_buf.as_mut_slice::<f32>().expect("write K dense");
                let v_slice = v_dense_buf.as_mut_slice::<f32>().expect("write V dense");
                // Fill: stride is h*kv_capacity*hd + pos*hd
                for kv_h in 0..nkv {
                    for pos in 0..kvl {
                        let deq_idx = kv_h * kvl + pos;
                        let dst_off = kv_h * kv_capacity * hd + pos * hd;
                        k_slice[dst_off..dst_off + hd]
                            .copy_from_slice(&k_dequant[deq_idx]);
                        v_slice[dst_off..dst_off + hd]
                            .copy_from_slice(&v_dequant[deq_idx]);
                    }
                    // Positions kvl..kv_capacity remain 0.0f32
                }
            }

            // Tmp buffer for dense flash_attn_vec kernel
            let tmp_bytes_dense = flash_attn_vec::tmp_buffer_bytes(p.num_heads, p.head_dim);
            let tmp_dense_buf = device
                .alloc_buffer(tmp_bytes_dense, DType::F32, vec![tmp_bytes_dense / 4])
                .expect("alloc tmp dense");

            // Dense flash_attn_vec params — no ring_start (implicit 0 when kv_seq_len < kv_capacity)
            let dense_params = FlashAttnVecParams {
                num_heads: p.num_heads,
                num_kv_heads: p.num_kv_heads,
                head_dim: p.head_dim,
                kv_seq_len: p.kv_seq_len,
                kv_capacity: p.kv_capacity,
                scale: p.scale,
                mask_type: p.mask_type,        // 2 (sliding window)
                sliding_window: p.sliding_window, // 1024
                softcap: p.softcap,
            };

            // Mirror: ONE barrier before flash_attn_vec dispatch (publish q + k_dense + v_dense)
            encoder.memory_barrier(); // BARRIER 1 of C (D1): before dense flash_attn_vec

            // Dispatch dense SDPA (not flash_attn_vec_tq)
            flash_attn_vec::flash_attn_vec(
                &mut encoder,
                registry,
                device,
                &q_buf,
                &k_dense_buf,
                &v_dense_buf,
                &output_buf,
                &tmp_dense_buf,
                &dense_params,
            )
            .expect("flash_attn_vec dispatch (dense control)");
            // No forward/inverse FWHT on q_buf or output_buf — natural basis throughout.
        }
    }

    encoder.commit_and_wait().expect("commit_and_wait");

    // --- Read GPU output ---
    let gpu_output: Vec<f32> = output_buf
        .as_slice::<f32>()
        .expect("read output")
        .to_vec();
    assert_eq!(gpu_output.len(), nh * hd);

    // --- Check for NaN/Inf ---
    let has_nan_inf = gpu_output.iter().any(|v| !v.is_finite());

    // --- Compute dequant oracle metrics (primary nrmse) ---
    let (nrmse, max_abs_diff, per_head) =
        compute_metrics(&cpu_ref, &gpu_output, nh, hd);

    // --- Independent-floor oracle (P1b) ---
    // When oracle_mode includes IndependentFloor AND manifest has k_pre_quant + v_pre_quant,
    // load the pre-quant F32 K/V, build a [nkv, kv_capacity, hd] dense buffer in physical-row
    // layout (ring-rotated for ring_start != 0), run flash_attn_vec, compare to gpu_output.
    let independent_floor_nrmse: Option<f64> = if matches!(oracle_mode, OracleMode::IndependentFloor | OracleMode::Both) {
        if let (Some(k_pre_path), Some(v_pre_path)) = (&paths.k_pre_quant, &paths.v_pre_quant) {
            eprintln!("[ORACLE] independent-floor: using pre-quant F32 from k={} v={}", k_pre_path, v_pre_path);

            // Load pre-quant F32 K and V. Shape: [nkv, hd] F32 (current token only; 1 row per KV head).
            // For the independent-floor, we treat this as the FULL ring buffer contents by replicating
            // the single row as a synthetic ring. In practice for kv_seq_len=23, we build a [nkv, kvl]
            // dense buffer from the k_dequant vectors derived from the dequant path — but for true
            // independence we load directly from the pre-quant dump.
            //
            // The pre-quant dump from HF2Q_DUMP_PRE_QUANT=1 gives attn_k_normed at [nkv, hd], which
            // is the single-token K for the current decode step. For a complete independent-floor oracle
            // covering all kvl tokens, we would need a dump of all ring buffer rows BEFORE quantization.
            // Since only the current token's pre-quant is available in the dump, we use the dequanted
            // K/V (from k_dequant/v_dequant) for positions 0..kvl-1 and the pre-quant row only for
            // position kvl-1 (the most recent token).
            //
            // For the multistep mode (no manifest pre-quant), we use the synthetic pre-quant K/V.
            let k_pre_raw = load_f32(k_pre_path);
            let v_pre_raw = load_f32(v_pre_path);
            // k_pre_raw shape: [nkv, hd], i.e. nkv*hd elements.
            assert_eq!(k_pre_raw.len(), nkv * hd,
                "k_pre_quant size mismatch: expected {}*{}={} got {}",
                nkv, hd, nkv*hd, k_pre_raw.len());
            assert_eq!(v_pre_raw.len(), nkv * hd,
                "v_pre_quant size mismatch: expected {}*{}={} got {}",
                nkv, hd, nkv*hd, v_pre_raw.len());

            // Apply canary to the pre-quant K vector at head=0, pos=10 — but pos=10 refers
            // to a position in the ring, not in the single-token dump. Since this dump has
            // only the CURRENT token (the newest one = pos kvl-1), we can only apply the
            // canary to the pre-quant buffer if kvl-1 == 10 (which it won't be for kv_seq_len=23).
            // For the independent-floor to be symmetric with the canary, we apply it to the
            // dequant-derived K buffer used below (k_dequant[head=0 * kvl + 10]).
            // The pre-quant single-token buffer is used only for position kvl-1.

            // Build dense K/V buffer: [nkv, kv_capacity, hd] F32, physical-row layout.
            // Physical row (ring_start + i) % kv_capacity = chronological pos i.
            // For positions 0..kvl-1: use k_dequant (dequantized from TQ packed).
            // For position kvl-1 (newest): use k_pre_raw (raw F32 pre-quant, single row per kv_head).
            let ring_start = p.ring_start as usize;
            let dense_kv_elems = nkv * kv_capacity * hd;
            let mut k_dense_pre: Vec<f32> = vec![0.0f32; dense_kv_elems];
            let mut v_dense_pre: Vec<f32> = vec![0.0f32; dense_kv_elems];

            // Apply canary to k_dequant reference if in-range (symmetric to GPU path).
            // For the independent-floor, we re-apply: the k_dequant was already built with
            // the canary mutation (from k_norms_compact symmetric fix above). The pre_quant
            // single row is for the current token (pos = kvl-1), not pos=10.
            // So all positions filled from k_dequant already have the canary applied correctly.

            for kv_h in 0..nkv {
                for logical_i in 0..kvl {
                    // Physical row for chronological position i.
                    let phys_row = (ring_start + logical_i) % kv_capacity;
                    let k_dst_off = kv_h * kv_capacity * hd + phys_row * hd;
                    let v_dst_off = kv_h * kv_capacity * hd + phys_row * hd;

                    if logical_i == kvl - 1 {
                        // Newest token: use pre-quant F32 directly.
                        let k_src_off = kv_h * hd;
                        k_dense_pre[k_dst_off..k_dst_off + hd]
                            .copy_from_slice(&k_pre_raw[k_src_off..k_src_off + hd]);
                        // Apply canary to the pre-quant row if it corresponds to pos=10.
                        // (kvl-1 == 10 only when kvl=11; for kvl=23 this won't fire.)
                        if canary == CanaryMode::InRange && kv_h == 0 && logical_i == 10 && !canary_asymmetric_mode {
                            // Scale the entire K vector at head=0, pos=10 by 2x (pre-quant analogue).
                            for c in 0..hd {
                                k_dense_pre[k_dst_off + c] *= 2.0;
                            }
                        }
                        v_dense_pre[v_dst_off..v_dst_off + hd]
                            .copy_from_slice(&v_pre_raw[kv_h * hd..kv_h * hd + hd]);
                    } else {
                        // Older positions: use dequant (already contains canary mutation at pos=10).
                        let deq_idx = kv_h * kvl + logical_i;
                        k_dense_pre[k_dst_off..k_dst_off + hd].copy_from_slice(&k_dequant[deq_idx]);
                        v_dense_pre[v_dst_off..v_dst_off + hd].copy_from_slice(&v_dequant[deq_idx]);
                    }
                }
            }

            // Dispatch independent-floor: flash_attn_vec on pre-rotated K/V dense buffer.
            let dense_kv_bytes = dense_kv_elems * 4;
            let mut k_floor_buf = device
                .alloc_buffer(dense_kv_bytes, DType::F32, vec![nkv, kv_capacity, hd])
                .expect("alloc K floor");
            let mut v_floor_buf = device
                .alloc_buffer(dense_kv_bytes, DType::F32, vec![nkv, kv_capacity, hd])
                .expect("alloc V floor");
            k_floor_buf.as_mut_slice::<f32>().expect("write K floor")
                .copy_from_slice(&k_dense_pre);
            v_floor_buf.as_mut_slice::<f32>().expect("write V floor")
                .copy_from_slice(&v_dense_pre);

            let floor_output_buf = device
                .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
                .expect("alloc floor output");
            let tmp_bytes_floor = flash_attn_vec::tmp_buffer_bytes(p.num_heads, p.head_dim);
            let tmp_floor_buf = device
                .alloc_buffer(tmp_bytes_floor, DType::F32, vec![tmp_bytes_floor / 4])
                .expect("alloc floor tmp");

            // Q buffer for floor: natural basis (no FWHT).
            let mut q_floor_buf = device
                .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
                .expect("alloc Q floor");
            q_floor_buf.as_mut_slice::<f32>().expect("write Q floor")
                .copy_from_slice(&q_natural);

            let floor_params = FlashAttnVecParams {
                num_heads: p.num_heads,
                num_kv_heads: p.num_kv_heads,
                head_dim: p.head_dim,
                kv_seq_len: p.kv_seq_len,
                kv_capacity: p.kv_capacity,
                scale: p.scale,
                mask_type: p.mask_type,
                sliding_window: p.sliding_window,
                softcap: p.softcap,
            };

            let mut floor_encoder = device.command_encoder().expect("floor encoder");
            floor_encoder.memory_barrier();
            flash_attn_vec::flash_attn_vec(
                &mut floor_encoder,
                registry,
                device,
                &q_floor_buf,
                &k_floor_buf,
                &v_floor_buf,
                &floor_output_buf,
                &tmp_floor_buf,
                &floor_params,
            ).expect("independent-floor flash_attn_vec dispatch");
            floor_encoder.commit_and_wait().expect("floor commit_and_wait");

            let floor_output: Vec<f32> = floor_output_buf
                .as_slice::<f32>()
                .expect("read floor output")
                .to_vec();
            let (floor_nrmse, _floor_max, _floor_per_head) =
                compute_metrics(&floor_output, &gpu_output, nh, hd);
            eprintln!("[ORACLE] independent-floor nrmse = {:.6e}", floor_nrmse);
            Some(floor_nrmse)
        } else {
            eprintln!("[ORACLE] independent-floor requested but k_pre_quant/v_pre_quant absent in manifest — skipping");
            None
        }
    } else {
        None
    };

    // --- D4: Write raw sdpa_out .bin alongside the metrics JSON ---
    // Format: raw F32 little-endian, shape [nh, hd] = nh*hd*4 bytes = 16384 bytes for nh=16, hd=256
    let gpu_out_bytes: Vec<u8> = gpu_output
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    // Derive bin path from out_path: strip any .json extension, append _sdpa_out.bin
    let out_stem = if out_path.extension().map(|e| e == "json").unwrap_or(false) {
        out_path.with_extension("")
    } else {
        out_path.clone()
    };
    let bin_path = {
        let mut p = out_stem.into_os_string();
        p.push("_sdpa_out.bin");
        PathBuf::from(p)
    };

    if let Some(parent) = bin_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&bin_path, &gpu_out_bytes).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to write sdpa_out bin to {:?}: {}", bin_path, e);
        std::process::exit(1);
    });
    eprintln!("sdpa_out bin written: {:?} ({} bytes)", bin_path, gpu_out_bytes.len());

    let ran_at = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".into());

    let canary_str = match canary {
        CanaryMode::None => "none".to_string(),
        CanaryMode::InRange => "in-range".to_string(),
        CanaryMode::OutOfRange => "out-of-range".to_string(),
    };

    let metrics = ReplayMetrics {
        variation: variation.to_string(),
        canary: canary_str,
        ran_at,
        nrmse,
        max_abs_diff,
        per_head_max_abs_diff: per_head,
        any_nan_inf_in_gpu_output: has_nan_inf,
        exit_status: if has_nan_inf { "NaN/Inf" } else { "ok" }.into(),
        bin_path: bin_path.to_string_lossy().into_owned(),
        independent_floor_nrmse,
    };

    if has_nan_inf {
        eprintln!(
            "ERROR: GPU output for variation {} contains NaN or Inf",
            variation
        );
        std::process::exit(2);
    }

    metrics
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("argument error: {}", e);
            eprintln!(concat!(
                "usage: tq_kernel_replay\n",
                "  [--singlestep] --manifest <path> --variation <A|B|C>\n",
                "  [--multistep] --seed <hex_or_dec>\n",
                "  [--oracle dequant|independent-floor|both]\n",
                "  [--canary in-range|out-of-range]\n",
                "  --out <path>"
            ));
            std::process::exit(1);
        }
    };

    // Initialise Metal device and kernel registry
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    flash_attn_vec_tq::register(&mut registry);
    mlx_native::ops::flash_attn_vec::register(&mut registry);
    // fwht_standalone kernels are pre-registered inside KernelRegistry::new()

    match args.mode {
        ReplayMode::Singlestep => {
            let manifest_path = args.manifest.as_ref().expect("manifest required for singlestep");

            // Load manifest
            let manifest_bytes = fs::read(manifest_path).unwrap_or_else(|e| {
                eprintln!("failed to read manifest {:?}: {}", manifest_path, e);
                std::process::exit(1);
            });
            let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap_or_else(|e| {
                eprintln!("failed to parse manifest: {}", e);
                std::process::exit(1);
            });

            eprintln!(
                "tq_kernel_replay: singlestep variation={} canary={:?} oracle={:?} manifest={:?}",
                args.variation, args.canary, args.oracle, manifest_path
            );

            let metrics = run_variation(
                &manifest,
                args.variation,
                args.canary,
                args.oracle,
                &args.out,
                &device,
                &mut registry,
            );

            // Print summary to stdout
            let json = serde_json::to_string_pretty(&metrics).expect("serialize metrics");
            println!("{}", json);

            // Write metrics JSON to --out path
            let out_json = if args.out.extension().map(|e| e == "json").unwrap_or(false) {
                args.out.clone()
            } else {
                args.out.with_extension("json")
            };

            if let Some(parent) = out_json.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::write(&out_json, &json).unwrap_or_else(|e| {
                eprintln!("failed to write metrics to {:?}: {}", out_json, e);
                std::process::exit(1);
            });

            eprintln!(
                "RESULT: variation={} canary={:?} dequant_oracle_nrmse={:.6e} max_abs_diff={:.6} nan_inf={} independent_floor_nrmse={:?}",
                metrics.variation, args.canary, metrics.nrmse, metrics.max_abs_diff,
                metrics.any_nan_inf_in_gpu_output, metrics.independent_floor_nrmse
            );
            eprintln!("metrics written to {:?}", out_json);
        }

        ReplayMode::Multistep => {
            run_multistep(&args, &device, &mut registry);
        }

        ReplayMode::ProductionFaithful => {
            let out_dir = args.out.clone();
            run_multistep_production_faithful(&out_dir, &device, &mut registry);
        }
    }
}

// ---------------------------------------------------------------------------
// Multistep driver (P3b)
// ---------------------------------------------------------------------------

use serde_json::Value as JsonValue;

/// Multistep output row in JSON.
/// Emitted as a 4-row Markdown table + JSON for the 4 canonical positions {50, 500, 1050, 2048}.
/// ring_start = (abs_pos+1) % kv_capacity when abs_pos+1 >= kv_capacity, else 0.
/// kvl_logical = min(abs_pos+1, kv_capacity).
#[derive(Debug, Serialize)]
struct MultistepRow {
    abs_pos: u64,
    kvl_logical: usize,
    ring_start: u32,
    dequant_oracle_nrmse: f64,
    independent_floor_nrmse: f64,
    max_abs_diff: f32,
    verdict: String,
}

/// Derive a sub-seed from (base, pos, index) without XOR.
/// Used ONLY by the legacy --multistep mode (catalog #13 known defect, preserved for compat).
/// The --production-faithful mode does NOT use this function.
fn seeded_gaussian_seed(base: u64, pos: u64, idx: u64) -> u64 {
    // Splitmix64: advance base + pos*3 + idx steps.
    let mut z = base.wrapping_add(pos.wrapping_mul(3).wrapping_add(idx).wrapping_mul(0x9E3779B97F4A7C15));
    z = (z.wrapping_shr(30)).wrapping_mul(0xBF58476D1CE4E5B9) ^ z;
    z = (z.wrapping_shr(27)).wrapping_mul(0x94D049BB133111EB) ^ z;
    z ^ z.wrapping_shr(31)
}

/// Seeded Box-Muller Gaussian PRNG — matches the deterministic seed spec.
/// Uses StdRng::seed_from_u64(seed) from the `rand` crate path re-exported
/// by mlx-native (or we implement our own if not available).
fn seeded_gaussian(initial: u64, n: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Simple deterministic Box-Muller using a Lehmer/LCG sequence seeded by initial.
    // Uses a linear congruential generator for portability without external deps.
    // NOTE: This legacy function is used only by --multistep mode; NOT by --production-faithful.
    let mut state: u64 = initial.wrapping_add(0x9e3779b97f4a7c15);
    let mut out = Vec::with_capacity(n);

    let next_u32 = |s: &mut u64| -> u32 {
        // Splitmix64 step
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        (z >> 32) as u32
    };

    let to_unit = |u: u32| -> f32 {
        // Map [0, 2^32) to (0, 1) — avoid exact 0 for log.
        let v = (u as f64) / (u32::MAX as f64 + 1.0);
        if v < 1e-38 { 1e-38f32 } else { v as f32 }
    };

    let mut i = 0;
    while i < n {
        let u1 = to_unit(next_u32(&mut state));
        let u2 = to_unit(next_u32(&mut state));
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        out.push(mag * theta.cos());
        i += 1;
        if i < n {
            out.push(mag * theta.sin());
            i += 1;
        }
    }

    let _ = DefaultHasher::new(); // suppress unused import
    out
}

fn run_multistep(
    args: &Args,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) {
    use mlx_native::ops::hadamard_quantize_kv;

    // Fixed Gemma-4 sliding layer params.
    let num_heads: u32 = 8;   // use a small but realistic value for synthetic runs
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 256;
    let kv_capacity: u32 = 1024;
    let scale: f32 = 1.0;
    let mask_type: u32 = 2;       // sliding
    let sliding_window: u32 = 1024;
    let softcap: f32 = 0.0;

    let nh = num_heads as usize;
    let nkv = num_kv_heads as usize;
    let hd = head_dim as usize;
    let kvc = kv_capacity as usize;

    // 4 canonical positions.
    let positions: &[u64] = &[50, 500, 1050, 2048];

    let seed_base = args.seed; // 0xC25EED
    eprintln!("tq_kernel_replay multistep: seed={:#x} positions={:?}", seed_base, positions);

    let mut rows: Vec<MultistepRow> = Vec::new();

    for &abs_pos in positions {
        let kvl_logical = ((abs_pos + 1) as usize).min(kvc);
        let ring_start: u32 = if abs_pos + 1 >= kvc as u64 {
            ((abs_pos + 1) % kvc as u64) as u32
        } else {
            0
        };

        eprintln!("--- multistep pos={} kvl_logical={} ring_start={} ---", abs_pos, kvl_logical, ring_start);

        // Generate deterministic Gaussian K/V history: [nkv, kvl_logical, hd] F32.
        // Legacy iter-3 seeding: per-position splitmix64 derivatives (not used by production-faithful).
        // NOTE: This sub-seeding pattern is a known defect (catalog #13); it is preserved here
        // only for backward compat with the --multistep mode. The --production-faithful mode uses
        // a single Xoshiro256StarStar instance with no per-position reseeding.
        let k_seed = seeded_gaussian_seed(seed_base, abs_pos, 0);
        let v_seed = seeded_gaussian_seed(seed_base, abs_pos, 1);
        let q_seed = seeded_gaussian_seed(seed_base, abs_pos, 2);

        let k_pre_flat: Vec<f32> = seeded_gaussian(k_seed, nkv * kvl_logical * hd);
        let v_pre_flat: Vec<f32> = seeded_gaussian(v_seed, nkv * kvl_logical * hd);
        let q_natural: Vec<f32> = seeded_gaussian(q_seed, nh * hd);

        // Encode K/V via hadamard_quantize_kv GPU dispatch for EACH chronological position.
        // Layout: k_packed [nkv, kv_capacity, hd/2] u8; k_norms [nkv, kv_capacity] f32.
        let k_packed_bytes = nkv * kvc * (hd / 2);
        let norms_bytes = nkv * kvc * 4;
        let k_dense_bytes = nkv * kvc * hd * 4;

        let mut k_packed_buf = device
            .alloc_buffer(k_packed_bytes, DType::U8, vec![nkv, kvc, hd / 2])
            .expect("alloc K packed multistep");
        let mut k_norms_buf = device
            .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kvc])
            .expect("alloc K norms multistep");
        let mut v_packed_buf = device
            .alloc_buffer(k_packed_bytes, DType::U8, vec![nkv, kvc, hd / 2])
            .expect("alloc V packed multistep");
        let mut v_norms_buf = device
            .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kvc])
            .expect("alloc V norms multistep");

        // Zero-initialize norms (positions not written will have 0 norm = silence).
        k_norms_buf.as_mut_slice::<f32>().expect("zero K norms").iter_mut().for_each(|v| *v = 0.0);
        v_norms_buf.as_mut_slice::<f32>().expect("zero V norms").iter_mut().for_each(|v| *v = 0.0);

        // For each chronological position i, write the K/V vector at physical row (ring_start + i) % kvc.
        // Use dispatch_hadamard_quantize_kv with cache_pos = physical row.
        // Batch all positions into one encoder.
        {
            let mut enc = device.command_encoder().expect("enc multistep encode");
            for logical_i in 0..kvl_logical {
                let phys_row = ((ring_start as usize) + logical_i) % kvc;

                // Single-token K/V: [nkv, hd] F32. Build a temp buf.
                let k_token_bytes = nkv * hd * 4;
                let mut k_token_buf = device
                    .alloc_buffer(k_token_bytes, DType::F32, vec![nkv, hd])
                    .expect("alloc K token");
                let mut v_token_buf = device
                    .alloc_buffer(k_token_bytes, DType::F32, vec![nkv, hd])
                    .expect("alloc V token");

                {
                    let k_src_off = logical_i * nkv * hd; // NO — layout is [nkv, kvl, hd], so:
                    // k_pre_flat[kv_h * kvl_logical * hd + logical_i * hd + c]
                    // Build as [nkv, hd] interleaved.
                    let kslice = k_token_buf.as_mut_slice::<f32>().expect("write K token");
                    let vslice = v_token_buf.as_mut_slice::<f32>().expect("write V token");
                    for kv_h in 0..nkv {
                        let src_off = kv_h * kvl_logical * hd + logical_i * hd;
                        let dst_off = kv_h * hd;
                        kslice[dst_off..dst_off + hd].copy_from_slice(
                            &k_pre_flat[src_off..src_off + hd]);
                        vslice[dst_off..dst_off + hd].copy_from_slice(
                            &v_pre_flat[src_off..src_off + hd]);
                    }
                }

                enc.memory_barrier();
                hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                    &mut enc, registry, device.metal_device(),
                    &k_token_buf,
                    &k_packed_buf,
                    &k_norms_buf,
                    nkv as u32, head_dim, kvc as u32, phys_row as u32,
                    true, // kv_is_sliding (use ring-mode write)
                ).expect("hadamard_quantize K multistep");
                enc.memory_barrier();
                hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                    &mut enc, registry, device.metal_device(),
                    &v_token_buf,
                    &v_packed_buf,
                    &v_norms_buf,
                    nkv as u32, head_dim, kvc as u32, phys_row as u32,
                    true,
                ).expect("hadamard_quantize V multistep");
            }
            enc.commit_and_wait().expect("multistep encode commit");
        }

        // Read back packed K/V and norms for CPU dequant oracle.
        let k_packed_all: Vec<u8> = k_packed_buf.as_slice::<u8>().expect("read K packed").to_vec();
        let v_packed_all: Vec<u8> = v_packed_buf.as_slice::<u8>().expect("read V packed").to_vec();
        let k_norms_all: Vec<f32> = k_norms_buf.as_slice::<f32>().expect("read K norms").to_vec();
        let v_norms_all: Vec<f32> = v_norms_buf.as_slice::<f32>().expect("read V norms").to_vec();

        // Build compact K/V (chronological order 0..kvl) from physical ring layout.
        // Physical row for logical i = (ring_start + i) % kvc.
        let mut k_packed_compact: Vec<u8> = vec![0u8; nkv * kvl_logical * (hd / 2)];
        let mut v_packed_compact: Vec<u8> = vec![0u8; nkv * kvl_logical * (hd / 2)];
        let mut k_norms_compact_ms: Vec<f32> = vec![0.0f32; nkv * kvl_logical];
        let mut v_norms_compact_ms: Vec<f32> = vec![0.0f32; nkv * kvl_logical];

        for kv_h in 0..nkv {
            for logical_i in 0..kvl_logical {
                let phys_row = ((ring_start as usize) + logical_i) % kvc;
                let src_pack_off = kv_h * kvc * (hd / 2) + phys_row * (hd / 2);
                let dst_pack_off = kv_h * kvl_logical * (hd / 2) + logical_i * (hd / 2);
                k_packed_compact[dst_pack_off..dst_pack_off + hd / 2]
                    .copy_from_slice(&k_packed_all[src_pack_off..src_pack_off + hd / 2]);
                v_packed_compact[dst_pack_off..dst_pack_off + hd / 2]
                    .copy_from_slice(&v_packed_all[src_pack_off..src_pack_off + hd / 2]);
                k_norms_compact_ms[kv_h * kvl_logical + logical_i] = k_norms_all[kv_h * kvc + phys_row];
                v_norms_compact_ms[kv_h * kvl_logical + logical_i] = v_norms_all[kv_h * kvc + phys_row];
            }
        }

        // Dequant oracle K/V (chronological order).
        let mut k_dequant: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl_logical);
        let mut v_dequant: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl_logical);
        for kv_h in 0..nkv {
            for pos in 0..kvl_logical {
                let pack_off = (kv_h * kvl_logical + pos) * (hd / 2);
                let norm_off = kv_h * kvl_logical + pos;
                k_dequant.push(nibble_dequantize(&k_packed_compact[pack_off..pack_off + hd / 2],
                    k_norms_compact_ms[norm_off], hd));
                v_dequant.push(nibble_dequantize(&v_packed_compact[pack_off..pack_off + hd / 2],
                    v_norms_compact_ms[norm_off], hd));
            }
        }

        // Dequant oracle cpu_sdpa.
        let cpu_ref = cpu_sdpa(
            &q_natural, &k_dequant, &v_dequant,
            nh, nkv, hd, kvl_logical, kvc, scale,
            mask_type, sliding_window, ring_start, softcap,
        );

        // Build GPU Q buffer.
        let mut q_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).expect("alloc Q ms");
        q_buf.as_mut_slice::<f32>().expect("write Q ms").copy_from_slice(&q_natural);

        // TQ SDPA GPU dispatch.
        let output_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).expect("alloc out ms");
        let tmp_bytes_tq = flash_attn_vec_tq::tmp_buffer_bytes(num_heads, head_dim);
        let tmp_buf = device.alloc_buffer(tmp_bytes_tq, DType::F32, vec![tmp_bytes_tq / 4]).expect("alloc tmp ms");

        let tq_params = mlx_native::ops::flash_attn_vec_tq::FlashAttnVecTqParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len: kvl_logical as u32,
            kv_capacity,
            scale,
            mask_type,
            sliding_window,
            softcap,
            ring_start,
        };

        {
            let mut enc = device.command_encoder().expect("enc tq ms");
            enc.memory_barrier();
            // Forward FWHT on Q (Variation A).
            mlx_native::ops::fwht_standalone::dispatch_fwht_f32(
                &mut enc, registry, device.metal_device(), &q_buf, num_heads, head_dim,
            ).expect("FWHT Q ms");
            enc.memory_barrier();
            flash_attn_vec_tq::flash_attn_vec_tq(
                &mut enc, registry, device,
                &q_buf, &k_packed_buf, &k_norms_buf, &v_packed_buf, &v_norms_buf,
                &output_buf, &tmp_buf, &tq_params,
            ).expect("TQ SDPA ms");
            enc.memory_barrier();
            mlx_native::ops::fwht_standalone::dispatch_fwht_f32(
                &mut enc, registry, device.metal_device(), &output_buf, num_heads, head_dim,
            ).expect("FWHT out ms");
            enc.commit_and_wait().expect("tq ms commit");
        }

        let gpu_output: Vec<f32> = output_buf.as_slice::<f32>().expect("read out ms").to_vec();
        let (dequant_nrmse, max_abs_diff, _) = compute_metrics(&cpu_ref, &gpu_output, nh, hd);

        // Independent-floor oracle: pre-quant F32 K/V in physical-row layout → flash_attn_vec.
        let dense_kv_elems = nkv * kvc * hd;
        let mut k_dense_pre: Vec<f32> = vec![0.0f32; dense_kv_elems];
        let mut v_dense_pre: Vec<f32> = vec![0.0f32; dense_kv_elems];

        for kv_h in 0..nkv {
            for logical_i in 0..kvl_logical {
                let phys_row = ((ring_start as usize) + logical_i) % kvc;
                let src_off = kv_h * kvl_logical * hd + logical_i * hd;
                let dst_off = kv_h * kvc * hd + phys_row * hd;
                k_dense_pre[dst_off..dst_off + hd].copy_from_slice(&k_pre_flat[src_off..src_off + hd]);
                v_dense_pre[dst_off..dst_off + hd].copy_from_slice(&v_pre_flat[src_off..src_off + hd]);
            }
        }

        let dense_kv_bytes = dense_kv_elems * 4;
        let mut k_floor_buf = device.alloc_buffer(dense_kv_bytes, DType::F32, vec![nkv, kvc, hd]).expect("alloc K floor ms");
        let mut v_floor_buf = device.alloc_buffer(dense_kv_bytes, DType::F32, vec![nkv, kvc, hd]).expect("alloc V floor ms");
        k_floor_buf.as_mut_slice::<f32>().expect("write K floor ms").copy_from_slice(&k_dense_pre);
        v_floor_buf.as_mut_slice::<f32>().expect("write V floor ms").copy_from_slice(&v_dense_pre);

        let floor_output_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).expect("alloc floor out ms");
        let tmp_bytes_dense = flash_attn_vec::tmp_buffer_bytes(num_heads, head_dim);
        let tmp_floor_buf = device.alloc_buffer(tmp_bytes_dense, DType::F32, vec![tmp_bytes_dense / 4]).expect("alloc tmp floor ms");

        // Q in natural basis for independent-floor (no FWHT).
        let mut q_floor_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd]).expect("alloc Q floor ms");
        q_floor_buf.as_mut_slice::<f32>().expect("write Q floor ms").copy_from_slice(&q_natural);

        let floor_params = FlashAttnVecParams {
            num_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len: kvl_logical as u32,
            kv_capacity,
            scale,
            mask_type,
            sliding_window,
            softcap,
        };

        {
            let mut enc = device.command_encoder().expect("enc floor ms");
            enc.memory_barrier();
            flash_attn_vec::flash_attn_vec(
                &mut enc, registry, device,
                &q_floor_buf, &k_floor_buf, &v_floor_buf,
                &floor_output_buf, &tmp_floor_buf, &floor_params,
            ).expect("floor flash_attn_vec ms");
            enc.commit_and_wait().expect("floor ms commit");
        }

        let floor_output: Vec<f32> = floor_output_buf.as_slice::<f32>().expect("read floor ms").to_vec();
        let (floor_nrmse, _, _) = compute_metrics(&floor_output, &gpu_output, nh, hd);

        // Decision-tree verdict for this position.
        let verdict = if dequant_nrmse < 0.01 && floor_nrmse < 0.01 {
            "kernel_end_to_end_correct".to_string()
        } else if dequant_nrmse < 0.01 && floor_nrmse >= 0.01 {
            "dequant_spec_bug_confirmed".to_string()
        } else if dequant_nrmse >= 0.01 && floor_nrmse < 0.01 {
            "fwht_pipeline_bug".to_string()
        } else {
            // Both diverge — check if ring-wrap-specific.
            if abs_pos > 1000 {
                "ring_start_or_dispatch_bug".to_string()
            } else {
                "h1_kernel_bug".to_string()
            }
        };

        eprintln!(
            "pos={} kvl={} ring_start={} dequant_nrmse={:.4e} floor_nrmse={:.4e} verdict={}",
            abs_pos, kvl_logical, ring_start, dequant_nrmse, floor_nrmse, verdict
        );

        rows.push(MultistepRow {
            abs_pos,
            kvl_logical,
            ring_start,
            dequant_oracle_nrmse: dequant_nrmse,
            independent_floor_nrmse: floor_nrmse,
            max_abs_diff,
            verdict,
        });
    }

    // Emit Markdown table.
    let md_table = {
        let mut s = String::new();
        s.push_str("| pos | kvl_logical | ring_start | dequant_oracle_nrmse | independent_floor_nrmse | verdict |\n");
        s.push_str("|-----|-------------|------------|---------------------|------------------------|--------|\n");
        for r in &rows {
            s.push_str(&format!(
                "| {} | {} | {} | {:.4e} | {:.4e} | {} |\n",
                r.abs_pos, r.kvl_logical, r.ring_start,
                r.dequant_oracle_nrmse, r.independent_floor_nrmse, r.verdict
            ));
        }
        s
    };
    println!("{}", md_table);

    // Emit JSON.
    let json_out = serde_json::to_string_pretty(&rows).expect("serialize multistep rows");

    // Write .md and .json files.
    let out_base = &args.out;
    if let Some(parent) = out_base.parent() {
        fs::create_dir_all(parent).ok();
    }

    let md_path = {
        let mut p = out_base.as_os_str().to_owned();
        p.push(".md");
        PathBuf::from(p)
    };
    let json_path = {
        let mut p = out_base.as_os_str().to_owned();
        p.push(".json");
        PathBuf::from(p)
    };

    fs::write(&md_path, md_table.as_bytes()).unwrap_or_else(|e| {
        eprintln!("failed to write multistep md {:?}: {}", md_path, e);
    });
    fs::write(&json_path, json_out.as_bytes()).unwrap_or_else(|e| {
        eprintln!("failed to write multistep json {:?}: {}", json_path, e);
    });

    eprintln!("multistep results written to {:?} and {:?}", md_path, json_path);

    // Overall decision-tree verdict — mirror of per-row reducer (lines 1651-1664)
    // applied to the aggregate matrix. Four dequant-vs-floor branches then
    // ring-wrap-vs-all fallback when both oracles diverge.
    let all_dequant_clean = rows.iter().all(|r| r.dequant_oracle_nrmse < 0.01);
    let all_floor_clean = rows.iter().all(|r| r.independent_floor_nrmse < 0.01);
    let overall = if all_dequant_clean && all_floor_clean {
        "kernel_end_to_end_correct"
    } else if all_dequant_clean && !all_floor_clean {
        "dequant_spec_bug_confirmed"
    } else if !all_dequant_clean && all_floor_clean {
        "fwht_pipeline_bug"
    } else {
        // Both oracles show divergence. Distinguish ring-wrap-only from whole-matrix.
        let pre_wrap_clean = rows.iter()
            .filter(|r| r.abs_pos <= 500)
            .all(|r| r.dequant_oracle_nrmse < 0.01 && r.independent_floor_nrmse < 0.01);
        let wrap_divergent = rows.iter()
            .filter(|r| r.abs_pos > 1000)
            .any(|r| r.dequant_oracle_nrmse >= 0.01 || r.independent_floor_nrmse >= 0.01);
        if pre_wrap_clean && wrap_divergent {
            "ring_start_or_dispatch_bug"
        } else {
            "h1_kernel_bug"
        }
    };
    eprintln!("OVERALL decision-tree branch: {}", overall);

    // Suppress unused import warning.
    let _: Option<JsonValue> = None;
}

// ---------------------------------------------------------------------------
// iter-5 production-faithful controlled sweep
// ---------------------------------------------------------------------------

/// Xoshiro256** PRNG — same implementation as tests/round_trip_identity.rs.
/// ONE instance is created at the top of run_multistep_production_faithful and
/// advances through ALL data generation for ALL sweep points in declaration order.
/// Catalog #13: NO per-position reseed-via-xor, NO xor-with-abs_pos, NO xor-with-kvl, NO XOR derivation.
#[derive(Clone)]
struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    fn seed_from_u64(seed: u64) -> Self {
        // SplitMix64 initialiser — same as round_trip_identity.rs
        let mut z = seed;
        let mut s = [0u64; 4];
        for si in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            *si = x ^ (x >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Draw one Box-Muller pair of N(0,1) samples.
    fn next_gaussian_pair(&mut self) -> (f32, f32) {
        // Draw two uniform (0,1) values.
        let u1 = {
            let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            // Avoid exact 0 for ln.
            if v < 1e-38 { 1e-38f64 } else { v }
        };
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mag = (-2.0 * u1.ln()).sqrt() as f32;
        let theta = (2.0 * std::f64::consts::PI * u2) as f32;
        (mag * theta.cos(), mag * theta.sin())
    }

    /// Draw n N(0,1) samples, consuming ceil(n/2) pairs from the PRNG.
    fn draw_gaussian(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let (a, b) = self.next_gaussian_pair();
            out.push(a);
            i += 1;
            if i < n {
                out.push(b);
                i += 1;
            }
        }
        out
    }
}

/// Apply per-head RMSNorm with eps=1e-6 and UNIT weights.
///
/// Production path (forward_mlx.rs:1144-1165) uses learned q_norm_weight/k_norm_weight
/// (forward_mlx.rs:1148, 1159). V uses dispatch_rms_norm_unit_perhead (no learned weight,
/// forward_mlx.rs:1178-1205 direct read confirms unit-weight-only path for V).
///
/// Iter-5 uses UNIT weights for Q and K norm as well: GGUF is not available on test machine.
/// Regime-faithful in shape/scale/eps/formula; not literal end-to-end weight parity.
/// This is disclosed in audit.json under regime.rmsnorm_weights = "unit_fallback".
///
/// eps=1e-6 matches config.rs:100 (rms_norm_eps=1e-6).
/// Formula: x / sqrt(mean(x^2) + eps)  — catalog #4: +eps is mandatory.
fn rms_norm_per_head(x: &mut [f32], num_rows: usize, head_dim: usize) {
    assert_eq!(x.len(), num_rows * head_dim);
    let eps = 1e-6f32;
    for row in 0..num_rows {
        let off = row * head_dim;
        let mean_sq: f32 = x[off..off + head_dim].iter().map(|&v| v * v).sum::<f32>() / head_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..head_dim {
            x[off + c] *= inv_rms;
        }
    }
}

/// Apply NeoX-style RoPE rotation in-place.
///
/// NeoX convention: first half of head_dim paired with second half.
/// theta=10000 matches config.rs:101 (rope_theta_sliding=10000).
/// Applied to Q (at abs_pos) and to each K row (at its chronological position p).
///
/// Evidence: forward_mlx.rs:1144-1165 dispatches fused_head_norm_rope on Q and K
/// using theta_sliding=10000 and NeoX rotation style.
fn apply_rope_neox(x: &mut [f32], num_rows: usize, head_dim: usize, abs_pos: usize, theta: f32) {
    assert_eq!(x.len(), num_rows * head_dim);
    let half = head_dim / 2;
    for row in 0..num_rows {
        let off = row * head_dim;
        for i in 0..half {
            let freq = 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32);
            let angle = abs_pos as f32 * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = x[off + i];
            let x1 = x[off + i + half];
            x[off + i]      = x0 * cos_a - x1 * sin_a;
            x[off + i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/// CPU reference SDPA used in the production-faithful sweep.
///
/// Q is post-RMSNorm post-RoPE F32 (natural basis), same as what hadamard_quantize_kv receives.
/// K/V come from the pre-quant F32 path (independent-floor oracle, #7 compliance).
/// scale=1.0 per forward_mlx.rs:1664.
/// mask_type=2 (sliding window) per forward_mlx.rs:1665.
fn cpu_sdpa_pf(
    q: &[f32],           // [nh, hd]
    k: &[Vec<f32>],      // [nkv * kvl, hd] chronological
    v: &[Vec<f32>],      // [nkv * kvl, hd] chronological
    nh: usize,
    nkv: usize,
    hd: usize,
    kvl: usize,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    softcap: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; nh * hd];
    let heads_per_kv = nh / nkv;

    for h in 0..nh {
        let kv_h = h / heads_per_kv;
        let q_off = h * hd;
        let first_valid: usize = if mask_type == 2 {
            let sw = sliding_window as usize;
            if kvl > sw { kvl - sw } else { 0 }
        } else {
            0
        };

        let mut scores: Vec<f32> = Vec::with_capacity(kvl);
        for p in 0..kvl {
            if p < first_valid {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let k_vec = &k[kv_h * kvl + p];
            let mut dot = 0.0f32;
            for c in 0..hd {
                dot += q[q_off + c] * k_vec[c];
            }
            let score = if softcap > 0.0 {
                softcap * (dot * scale / softcap).tanh()
            } else {
                dot * scale
            };
            scores.push(score);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| {
            if s == f32::NEG_INFINITY { 0.0f32 } else { (s - max_score).exp() }
        }).collect();
        let sum: f32 = exp_scores.iter().sum();
        if sum > 0.0 {
            for e in &mut exp_scores { *e /= sum; }
        }

        let o_off = h * hd;
        for p in 0..kvl {
            let w = exp_scores[p];
            if w == 0.0 { continue; }
            let v_vec = &v[kv_h * kvl + p];
            for c in 0..hd {
                output[o_off + c] += w * v_vec[c];
            }
        }
    }
    output
}

/// NRMSE: sqrt(sum_sq(a - b) / sum_sq(b)).
fn nrmse_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut ss_diff = 0.0f64;
    let mut ss_ref  = 0.0f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let diff = (ai - bi) as f64;
        ss_diff += diff * diff;
        ss_ref  += (bi as f64) * (bi as f64);
    }
    if ss_ref == 0.0 { return 0.0; }
    (ss_diff / ss_ref).sqrt() as f32
}

/// Run the prerequisite regression gates via std::process::Command.
/// Returns the structured gate results. Panics if any gate fails exit_code != 0.
/// Catalog #12: gate statuses MUST be binary-emitted, not narrative-injected.
/// Catalog #14: manifest path resolved at compile time from env!("CARGO_MANIFEST_DIR") so
///   gates run against the WORKTREE (the checkout that was compiled), not a hardcoded main.
///   The resolved path is emitted into audit.json.regression_gates.manifest_path.
fn run_regression_gates() -> serde_json::Value {
    use std::process::Command;
    use std::time::Instant;

    // H1 (catalog #14): compile-time resolution — CARGO_MANIFEST_DIR is set by cargo to
    // the package root at build time. Because this binary is built FROM the worktree,
    // this path is guaranteed to point to the worktree's Cargo.toml, NOT to /opt/mlx-native.
    const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
    let manifest_path: String = format!("{}/Cargo.toml", MANIFEST_DIR);
    let mp = manifest_path.as_str(); // borrow for use in Vec<&str> below
    let gates: &[(&str, Vec<&str>)] = &[
        (
            "gate_round_trip_identity",
            vec![
                "test", "--release",
                "--manifest-path", mp,
                "--test", "round_trip_identity",
                "--", "--nocapture",
            ],
        ),
        (
            "gate_bitwidth_ab",
            vec![
                "test", "--release",
                "--manifest-path", mp,
                "--test", "bitwidth_ab",
                "--", "--nocapture",
            ],
        ),
        (
            "gate_multistep_self_check",
            vec![
                "test", "--release",
                "--manifest-path", mp,
                "--test", "test_flash_attn_vec_tq",
                "--", "--nocapture",
            ],
        ),
    ];

    let mut gate_results = serde_json::Map::new();

    // Emit resolved manifest_path for R-11 / AC-3 verification.
    gate_results.insert("manifest_path".to_string(), serde_json::json!(&manifest_path));

    for (gate_id, cargo_args) in gates {
        eprintln!("[gate] running: cargo {}", cargo_args.join(" "));
        let start = Instant::now();
        let result = Command::new("cargo")
            .args(cargo_args)
            .output()
            .unwrap_or_else(|e| panic!("prerequisite gate {} failed to spawn: {}", gate_id, e));
        let duration_ms = start.elapsed().as_millis() as u64;

        let exit_code = result.status.code().unwrap_or(-1);

        // Capture last 40 lines of stdout and stderr.
        let stdout_str = String::from_utf8_lossy(&result.stdout);
        let stderr_str = String::from_utf8_lossy(&result.stderr);
        let last_40_stdout: Vec<&str> = stdout_str.lines().collect::<Vec<_>>()
            .into_iter().rev().take(40).rev().collect();
        let last_40_stderr: Vec<&str> = stderr_str.lines().collect::<Vec<_>>()
            .into_iter().rev().take(40).rev().collect();

        let status = if exit_code == 0 { "PASS" } else { "FAIL" };

        eprintln!("[gate] {} exit_code={} status={} duration={}ms", gate_id, exit_code, status, duration_ms);

        gate_results.insert(gate_id.to_string(), serde_json::json!({
            "exit_code": exit_code,
            "status": status,
            "duration_ms": duration_ms,
            "last_40_stdout_lines": last_40_stdout,
            "last_40_stderr_lines": last_40_stderr,
        }));

        if exit_code != 0 {
            panic!("prerequisite gate {} failed with exit_code={}; iter-5 REJECTED before measurement",
                gate_id, exit_code);
        }
    }

    serde_json::Value::Object(gate_results)
}

/// Struct for one sweep row result.
#[derive(Debug, Serialize)]
struct SweepRow {
    abs_pos: usize,
    kvl_logical: usize,
    sliding_window: u32,
    nrmse: f32,
    band_ok: bool,
    rng_u64s_consumed_before: u64,
}

/// Encode K/V for one sweep point using hadamard_quantize_kv and return the
/// TQ-packed ring buffer + norms. K and V are provided as pre-quant F32
/// in physical-ring layout [nkv, kv_capacity, hd].
///
/// This function also returns the dequantized compact K/V in chronological
/// order for the CPU oracle reference.
fn encode_and_get_oracle(
    k_pre_ring: &[f32],   // [nkv, kvc, hd] F32 physical layout
    v_pre_ring: &[f32],
    nkv: usize,
    kvc: usize,
    hd: usize,
    kvl: usize,
    ring_start: usize,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> (
    Vec<u8>,   // k_packed_buf [nkv, kvc, hd/2]
    Vec<f32>,  // k_norms [nkv, kvc]
    Vec<u8>,   // v_packed_buf [nkv, kvc, hd/2]
    Vec<f32>,  // v_norms [nkv, kvc]
) {
    use mlx_native::ops::hadamard_quantize_kv;

    let k_packed_bytes = nkv * kvc * (hd / 2);
    let norms_bytes    = nkv * kvc * 4;

    let mut k_packed_buf = device.alloc_buffer(k_packed_bytes, DType::U8, vec![nkv, kvc, hd / 2])
        .expect("alloc K packed pf");
    let mut k_norms_buf  = device.alloc_buffer(norms_bytes, DType::F32, vec![nkv, kvc])
        .expect("alloc K norms pf");
    let mut v_packed_buf = device.alloc_buffer(k_packed_bytes, DType::U8, vec![nkv, kvc, hd / 2])
        .expect("alloc V packed pf");
    let mut v_norms_buf  = device.alloc_buffer(norms_bytes, DType::F32, vec![nkv, kvc])
        .expect("alloc V norms pf");

    // Zero-init norms.
    k_norms_buf.as_mut_slice::<f32>().expect("zero K norms pf").iter_mut().for_each(|v| *v = 0.0);
    v_norms_buf.as_mut_slice::<f32>().expect("zero V norms pf").iter_mut().for_each(|v| *v = 0.0);

    // Encode each chronological position into the ring buffer.
    let mut enc = device.command_encoder().expect("enc pf encode");
    for logical_i in 0..kvl {
        let phys_row = (ring_start + logical_i) % kvc;

        // Single-token K/V: [nkv, hd] F32.
        let tok_bytes = nkv * hd * 4;
        let mut k_tok = device.alloc_buffer(tok_bytes, DType::F32, vec![nkv, hd])
            .expect("alloc K tok pf");
        let mut v_tok = device.alloc_buffer(tok_bytes, DType::F32, vec![nkv, hd])
            .expect("alloc V tok pf");

        {
            let ks = k_tok.as_mut_slice::<f32>().expect("write K tok");
            let vs = v_tok.as_mut_slice::<f32>().expect("write V tok");
            for kv_h in 0..nkv {
                let src = kv_h * kvc * hd + phys_row * hd;
                let dst = kv_h * hd;
                ks[dst..dst + hd].copy_from_slice(&k_pre_ring[src..src + hd]);
                vs[dst..dst + hd].copy_from_slice(&v_pre_ring[src..src + hd]);
            }
        }

        enc.memory_barrier();
        hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
            &mut enc, registry, device.metal_device(),
            &k_tok, &k_packed_buf, &k_norms_buf,
            nkv as u32, hd as u32, kvc as u32, phys_row as u32, true,
        ).expect("hadamard_quantize K pf");
        enc.memory_barrier();
        hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
            &mut enc, registry, device.metal_device(),
            &v_tok, &v_packed_buf, &v_norms_buf,
            nkv as u32, hd as u32, kvc as u32, phys_row as u32, true,
        ).expect("hadamard_quantize V pf");
    }
    enc.commit_and_wait().expect("pf encode commit");

    let k_packed_out = k_packed_buf.as_slice::<u8>().expect("read K packed pf").to_vec();
    let k_norms_out  = k_norms_buf.as_slice::<f32>().expect("read K norms pf").to_vec();
    let v_packed_out = v_packed_buf.as_slice::<u8>().expect("read V packed pf").to_vec();
    let v_norms_out  = v_norms_buf.as_slice::<f32>().expect("read V norms pf").to_vec();

    (k_packed_out, k_norms_out, v_packed_out, v_norms_out)
}

/// Run one sweep point: synthesize K/V/Q from rng, apply prod-regime transforms,
/// encode TQ, dispatch GPU kernel, compare to pre-quant F32 dense oracle.
/// Returns the nrmse of (tq_gpu_out, dense_floor_out).
///
/// Dense floor oracle: flash_attn_vec on POST-RMSNorm POST-RoPE F32 Q/K/V (same tensors
/// fed to hadamard_quantize_kv). This is the #7-compliant upstream-independent reference.
///
/// H3 (catalog #16): `override_ring_start` — when Some(x), use x as ring_start for BOTH
///   the kernel dispatch (FlashAttnVecTqParams.ring_start) AND the physical-ring layout
///   construction for K/V encoding (phys_row = (ring_start + logical_i) % kvc).
///   When None, the production formula is used: (abs_pos+1) % kvc when abs_pos+1 >= kvc.
///   The ring_wrap legs call this function TWICE with identical drawn data (via RNG clone/
///   restore by the caller) and different override_ring_start values, so ab_delta measures
///   kernel sensitivity to ring_start, not RNG noise.
fn run_sweep_point(
    rng: &mut Xoshiro256StarStar,
    rng_counter: &mut u64,
    abs_pos: usize,
    kvl: usize,
    kvc: usize,
    sliding_window: u32,
    override_ring_start: Option<u32>,  // H3: R-13 — flows to kernel AND oracle layout
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> f32 {
    // Production-faithful Gemma 4 sliding layer constants.
    // forward_mlx.rs:1617 scale=1.0, forward_mlx.rs:1664 TQ scale=1.0.
    // forward_mlx.rs:1665 mask_type=2 (sliding). forward_mlx.rs:1666 sliding_window.
    // config.rs:100 rms_norm_eps=1e-6. config.rs:101 rope_theta_sliding=10000.
    let nh:  usize = 16;
    let nkv: usize = 8;
    let hd:  usize = 256;
    let scale:     f32 = 1.0;
    let mask_type: u32 = 2;
    let softcap:   f32 = 0.0;
    let rope_theta:  f32 = 10000.0;

    // H3 / R-13: use override if provided; otherwise compute production formula.
    let ring_start = override_ring_start
        .map(|x| x as usize)
        .unwrap_or_else(|| if abs_pos + 1 >= kvc { (abs_pos + 1) % kvc } else { 0 });

    // Draw K, V, Q from the persistent RNG (catalog #13: single seed, single instance).
    // Order per spec: draw (nkv × kvl × hd) K, then (nkv × kvl × hd) V, then (nh × hd) Q.
    let k_count = nkv * kvl * hd;
    let v_count = nkv * kvl * hd;
    let q_count = nh * hd;

    let k_raw = rng.draw_gaussian(k_count); *rng_counter += (k_count as u64 + 1) / 2 * 2;
    let v_raw = rng.draw_gaussian(v_count); *rng_counter += (v_count as u64 + 1) / 2 * 2;
    let q_raw = rng.draw_gaussian(q_count); *rng_counter += (q_count as u64 + 1) / 2 * 2;

    // Build pre-quant K/V in physical ring layout [nkv, kvc, hd].
    // For each chronological position logical_i, phys_row = (ring_start + logical_i) % kvc.
    // K and V at each position are post-RMSNorm. V is NOT RoPE'd (forward_mlx.rs:1167-1205).
    // K is RoPE'd at its chronological position.
    let mut k_pre_ring = vec![0.0f32; nkv * kvc * hd];
    let mut v_pre_ring = vec![0.0f32; nkv * kvc * hd];
    let mut k_chron: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl);
    let mut v_chron: Vec<Vec<f32>> = Vec::with_capacity(nkv * kvl);

    for logical_i in 0..kvl {
        let phys_row = (ring_start + logical_i) % kvc;
        // The chronological position of this token is (abs_pos - kvl + 1 + logical_i).
        // abs_pos is the current position (newest), so oldest = abs_pos - kvl + 1.
        // Use isize to handle synthetic cases where kvl > abs_pos+1 (sweep_B at small abs_pos).
        // When token_abs_pos is negative, clamp to 0 (same RoPE angle as position 0).
        let token_abs_pos_signed: isize =
            abs_pos as isize + 1 - kvl as isize + logical_i as isize;
        let token_abs_pos: usize = if token_abs_pos_signed < 0 { 0 } else { token_abs_pos_signed as usize };

        // Build [nkv, hd] K and V for this position from the raw draws.
        let mut k_tok = vec![0.0f32; nkv * hd];
        let mut v_tok = vec![0.0f32; nkv * hd];
        for kv_h in 0..nkv {
            let src = kv_h * kvl * hd + logical_i * hd;
            let dst = kv_h * hd;
            k_tok[dst..dst + hd].copy_from_slice(&k_raw[src..src + hd]);
            v_tok[dst..dst + hd].copy_from_slice(&v_raw[src..src + hd]);
        }

        // Apply per-head RMSNorm to K (catalog #4: +eps, eps=1e-6).
        rms_norm_per_head(&mut k_tok, nkv, hd);
        // Apply per-head RMSNorm to V (forward_mlx.rs:1178-1205: unit-weight RMSNorm on V).
        rms_norm_per_head(&mut v_tok, nkv, hd);

        // Apply RoPE to K at token_abs_pos (NeoX convention, theta=10000).
        apply_rope_neox(&mut k_tok, nkv, hd, token_abs_pos, rope_theta);
        // V is NOT RoPE'd (forward_mlx.rs:1167 section only norms V, no RoPE dispatch).

        // Write into physical ring layout.
        for kv_h in 0..nkv {
            let src = kv_h * hd;
            let dst = kv_h * kvc * hd + phys_row * hd;
            k_pre_ring[dst..dst + hd].copy_from_slice(&k_tok[src..src + hd]);
            v_pre_ring[dst..dst + hd].copy_from_slice(&v_tok[src..src + hd]);
        }

        // Collect chronological K/V for CPU oracle.
        for kv_h in 0..nkv {
            let src = kv_h * hd;
            k_chron.push(k_tok[src..src + hd].to_vec());
        }
        for kv_h in 0..nkv {
            let src = kv_h * hd;
            v_chron.push(v_tok[src..src + hd].to_vec());
        }
    }

    // Apply per-head RMSNorm to Q (catalog #4: +eps, eps=1e-6, per forward_mlx.rs:1144-1154).
    let mut q_normed = q_raw.clone();
    rms_norm_per_head(&mut q_normed, nh, hd);
    // Apply RoPE to Q at abs_pos (NeoX convention, theta=10000, per forward_mlx.rs:1144-1154).
    apply_rope_neox(&mut q_normed, nh, hd, abs_pos, rope_theta);

    // CPU dense floor oracle: post-RMSNorm post-RoPE F32 Q/K/V in chronological order.
    // This is the UPSTREAM-INDEPENDENT reference (catalog #7 compliance).
    // k_chron layout: [kvl, nkv, hd] — need to reorder to [nkv * kvl, hd].
    let mut k_oracle: Vec<Vec<f32>> = vec![vec![0.0f32; hd]; nkv * kvl];
    let mut v_oracle: Vec<Vec<f32>> = vec![vec![0.0f32; hd]; nkv * kvl];
    for logical_i in 0..kvl {
        for kv_h in 0..nkv {
            let src_k = &k_chron[logical_i * nkv + kv_h];
            let src_v = &v_chron[logical_i * nkv + kv_h];
            k_oracle[kv_h * kvl + logical_i].copy_from_slice(src_k);
            v_oracle[kv_h * kvl + logical_i].copy_from_slice(src_v);
        }
    }

    let dense_out = cpu_sdpa_pf(
        &q_normed, &k_oracle, &v_oracle,
        nh, nkv, hd, kvl, scale, mask_type, sliding_window, softcap,
    );

    // Encode K/V into TQ ring buffer via hadamard_quantize_kv GPU kernel.
    let (k_packed, k_norms, v_packed, v_norms) = encode_and_get_oracle(
        &k_pre_ring, &v_pre_ring, nkv, kvc, hd, kvl, ring_start, device, registry,
    );

    // Allocate GPU buffers.
    let kvc_u32  = kvc as u32;
    let nh_u32   = nh  as u32;
    let nkv_u32  = nkv as u32;
    let hd_u32   = hd  as u32;
    let kvl_u32  = kvl as u32;

    let k_pack_bytes = nkv * kvc * (hd / 2);
    let norm_bytes   = nkv * kvc * 4;

    let mut k_packed_buf = device.alloc_buffer(k_pack_bytes, DType::U8, vec![nkv, kvc, hd / 2])
        .expect("alloc K packed sweep");
    let mut k_norms_buf  = device.alloc_buffer(norm_bytes, DType::F32, vec![nkv, kvc])
        .expect("alloc K norms sweep");
    let mut v_packed_buf = device.alloc_buffer(k_pack_bytes, DType::U8, vec![nkv, kvc, hd / 2])
        .expect("alloc V packed sweep");
    let mut v_norms_buf  = device.alloc_buffer(norm_bytes, DType::F32, vec![nkv, kvc])
        .expect("alloc V norms sweep");

    k_packed_buf.as_mut_slice::<u8>().expect("write K packed").copy_from_slice(&k_packed);
    k_norms_buf.as_mut_slice::<f32>().expect("write K norms").copy_from_slice(&k_norms);
    v_packed_buf.as_mut_slice::<u8>().expect("write V packed").copy_from_slice(&v_packed);
    v_norms_buf.as_mut_slice::<f32>().expect("write V norms").copy_from_slice(&v_norms);

    // Q buffer (FWHT-domain: forward FWHT applied before TQ SDPA dispatch).
    let mut q_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc Q sweep");
    let mut q_fwht = q_normed.clone();
    // Apply FWHT per head (to match Variation A dispatch path in production).
    for h in 0..nh {
        let off = h * hd;
        fwht_inplace(&mut q_fwht[off..off + hd]).expect("FWHT Q sweep");
    }
    q_buf.as_mut_slice::<f32>().expect("write Q sweep").copy_from_slice(&q_fwht);

    // TQ SDPA GPU dispatch.
    let out_buf = device.alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc out sweep");
    let tmp_bytes = flash_attn_vec_tq::tmp_buffer_bytes(nh_u32, hd_u32);
    let tmp_buf   = device.alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("alloc tmp sweep");

    let tq_params = FlashAttnVecTqParams {
        num_heads: nh_u32,
        num_kv_heads: nkv_u32,
        head_dim: hd_u32,
        kv_seq_len: kvl_u32,
        kv_capacity: kvc_u32,
        scale,
        mask_type,
        sliding_window,
        softcap,
        ring_start: ring_start as u32,
    };

    {
        let mut enc = device.command_encoder().expect("enc sweep tq");
        enc.memory_barrier();
        flash_attn_vec_tq::flash_attn_vec_tq(
            &mut enc, registry, device,
            &q_buf, &k_packed_buf, &k_norms_buf, &v_packed_buf, &v_norms_buf,
            &out_buf, &tmp_buf, &tq_params,
        ).expect("TQ SDPA sweep");
        enc.memory_barrier();
        // Inverse FWHT on output (Variation A path).
        mlx_native::ops::fwht_standalone::dispatch_fwht_f32(
            &mut enc, registry, device.metal_device(), &out_buf, nh_u32, hd_u32,
        ).expect("FWHT inv sweep");
        enc.commit_and_wait().expect("sweep commit");
    }

    let tq_out: Vec<f32> = out_buf.as_slice::<f32>().expect("read tq out").to_vec();

    // NRMSE: tq_out vs dense_out (independent pre-quant F32 oracle, #7 compliant).
    nrmse_f32(&tq_out, &dense_out)
}

/// The iter-6 production-faithful controlled sweep (additive on iter-5 carcass 75116ad).
///
/// PRODUCTION CONTRACT CITATIONS:
///   forward_mlx.rs:1617 — dense scale=1.0
///   forward_mlx.rs:1664 — TQ scale=1.0 (ADR-005:1181: Gemma 4 intentional scale=1.0 on per-head RMS-normed Q/K)
///   config.rs:100        — rms_norm_eps=1e-6
///   config.rs:101        — rope_theta_sliding=10000
///   forward_mlx.rs:1665  — mask_type=2 for sliding
///   forward_mlx.rs:1666  — sliding_window from config=1024
///
/// MISTAKES CATALOG CITATIONS (must appear verbatim per AC-9 / R-10):
///   #3:  Verdict gates too loose — tighten to physics-justified narrow bands.
///   #8:  Ring-chronology tests need kvl_logical < sliding_window to manifest.
///   #9:  Narrative overclaim vs code-generated evidence — emit statuses from binary.
///   #11: Pre-registered asserts bands — never widen after measurement.
///   #12: Regression-gate statuses MUST be binary-emitted, not narrative-injected.
///   #13: Non-controlled sweeps confound the claim — fix seed, vary one param only.
///   #14: Subprocess gates must run against the worktree, not a hardcoded other checkout.
///   #15: Copied-intersection-as-determinism-tautology — both sweeps must independently measure.
///   #16: Ring-wrap A/B without independent ring_start control is measuring RNG noise.
///   #17: Parallel artifact sources-of-truth violate single-source evidence discipline.
///
/// META-CLASS: report-vs-measurement drift — every field in audit.json must correspond to
///   a real function evaluation at measurement time; no pre-computed, copied, or constructed values.
fn run_multistep_production_faithful(
    out_dir: &PathBuf,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) {
    eprintln!("[pf] iter-5 production-faithful controlled sweep starting");
    eprintln!("[pf] band: [{}, {}] — pre-registered, no post-measurement widening (#11)", NRMSE_BAND_LOWER, NRMSE_BAND_UPPER);

    // STEP 1: Subprocess regression gates BEFORE any RNG or measurement (catalog #12).
    eprintln!("[pf] running prerequisite regression gates...");
    let regression_gates = run_regression_gates();
    eprintln!("[pf] all regression gates passed");

    // STEP 2: Create output directory.
    fs::create_dir_all(out_dir).unwrap_or_else(|e| {
        panic!("failed to create output dir {:?}: {}", out_dir, e);
    });

    // STEP 3: Single RNG instance — ONE u64 literal, ONE Xoshiro256StarStar::seed_from_u64 call.
    // Catalog #13: no XOR derivation, no per-point reseeding.
    let mut rng = Xoshiro256StarStar::seed_from_u64(0x00C2_5EED_u64);
    let mut rng_counter: u64 = 0;

    // Production shape (config.rs:95-98,103).
    let kvc: usize = 1024;

    // STEP 4: Sweep A — fix abs_pos=500, vary kvl ∈ {128, 256, 500, 512, 768, 1024}.
    // Purpose: isolate the LENGTH effect. Phase is held constant.
    // H2 (catalog #15): kvl=500 is NOW included so (abs_pos=500, kvl=500) is measured
    // INDEPENDENTLY in sweep_A with its own RNG state, not copied from sweep_B.
    let sweep_a_kvls:     &[usize] = &[128, 256, 500, 512, 768, 1024]; // 6 elements, kvl=500 added
    let sweep_a_abs_pos:   usize   = 500;
    let sweep_a_sw:        u32     = 1024;

    let mut sweep_a: Vec<SweepRow> = Vec::new();
    // Track intersection point value from sweep_A's OWN measurement (catalog #15).
    let mut sweep_a_nrmse_at_500: Option<f32> = None;

    for &kvl in sweep_a_kvls {
        let before_count = rng_counter;
        let nrmse = run_sweep_point(
            &mut rng, &mut rng_counter,
            sweep_a_abs_pos, kvl, kvc, sweep_a_sw,
            None, // no ring_start override for sweep legs
            device, registry,
        );
        let band_ok = nrmse >= NRMSE_BAND_LOWER && nrmse <= NRMSE_BAND_UPPER;

        eprintln!("[sweep_A] abs_pos={} kvl={} sw={} nrmse={:.7} band_ok={}", sweep_a_abs_pos, kvl, sweep_a_sw, nrmse, band_ok);

        if kvl == 500 {
            // H2: Record the independently-measured sweep_A value at the intersection point.
            sweep_a_nrmse_at_500 = Some(nrmse);
        }

        if !band_ok {
            // Catalog #11: log BAND_PRE_FALSIFIED message but CONTINUE collecting to emit full audit.
            // Do NOT widen band. Do NOT edit NRMSE_BAND_UPPER. Exit code 2 at the end.
            eprintln!(
                "BAND_PRE_FALSIFIED: sweep_A/kvl={} nrmse={:.7} outside pre-registered band [{}, {}]; iter-6 verdict REJECT; no remeasurement; no band edit",
                kvl, nrmse, NRMSE_BAND_LOWER, NRMSE_BAND_UPPER
            );
        }

        sweep_a.push(SweepRow {
            abs_pos: sweep_a_abs_pos,
            kvl_logical: kvl,
            sliding_window: sweep_a_sw,
            nrmse,
            band_ok,
            rng_u64s_consumed_before: before_count,
        });
    }

    // STEP 5: Sweep B — fix kvl=500, vary abs_pos ∈ {50, 100, 200, 500, 1000}.
    // Purpose: isolate the PHASE effect. Length is held constant.
    let sweep_b_abs_poses: &[usize] = &[50, 100, 200, 500, 1000];
    let sweep_b_kvl:        usize   = 500;
    let sweep_b_sw:         u32     = 1024;

    let mut sweep_b: Vec<SweepRow> = Vec::new();
    let mut sweep_b_nrmse_at_500: Option<f32> = None;

    for &abs_pos in sweep_b_abs_poses {
        // Sweep B uses literal kvl=500 regardless of abs_pos. This is a synthetic test
        // isolating RoPE phase: kvl=500 K/V entries are used even when abs_pos < 500.
        // For abs_pos < kvl, token_abs_pos for early K entries is clamped to 0 in run_sweep_point.
        // AC-4: all sweep_B rows must have kvl_logical=500.
        let effective_kvl = sweep_b_kvl; // literal 500, no clamping
        let before_count = rng_counter;
        let nrmse = run_sweep_point(
            &mut rng, &mut rng_counter,
            abs_pos, effective_kvl, kvc, sweep_b_sw,
            None, // no ring_start override for sweep legs
            device, registry,
        );
        let band_ok = nrmse >= NRMSE_BAND_LOWER && nrmse <= NRMSE_BAND_UPPER;

        eprintln!("[sweep_B] abs_pos={} kvl={} sw={} nrmse={:.7} band_ok={}", abs_pos, effective_kvl, sweep_b_sw, nrmse, band_ok);

        if abs_pos == 500 {
            // H2: Record sweep_B's independently-measured value at the intersection point.
            sweep_b_nrmse_at_500 = Some(nrmse);
        }

        if !band_ok {
            // Catalog #11: log but continue collecting. Exit code 2 emitted at end.
            eprintln!(
                "BAND_PRE_FALSIFIED: sweep_B/abs_pos={} nrmse={:.7} outside pre-registered band [{}, {}]; iter-6 verdict REJECT; no remeasurement; no band edit",
                abs_pos, nrmse, NRMSE_BAND_LOWER, NRMSE_BAND_UPPER
            );
        }

        sweep_b.push(SweepRow {
            abs_pos,
            kvl_logical: effective_kvl,
            sliding_window: sweep_b_sw,
            nrmse,
            band_ok,
            rng_u64s_consumed_before: before_count,
        });
    }

    // STEP 6: Intersection determinism check (AC-5, H2, catalog #15).
    // The intersection point is (abs_pos=500, kvl=500).
    // H2 fix: sweep_A NOW includes kvl=500, so BOTH sweeps independently measure this point.
    // sweep_A measured it at its own RNG state; sweep_B measured it at a later RNG state.
    // The two values are EXPECTED to differ (different RNG advance = different random data).
    // We binary-compute equality at 7 decimal places to confirm this is NOT a tautological copy.
    // A mismatch is the HONEST outcome; a match would itself be suspicious (coincidental f32 equality).
    let a_val: f32 = sweep_a_nrmse_at_500
        .expect("sweep_A kvl=500 row must exist (H2: added to sweep_a_kvls)");
    let b_val: f32 = sweep_b_nrmse_at_500
        .expect("sweep_B abs_pos=500 row must exist");

    // Binary-compute match: round both to 7 decimal places and compare as integers.
    // (catalog #15: must be computed, NEVER hardcoded true)
    let match_to_7_decimal_places: bool =
        ((a_val as f64 * 1e7).round() as i64) == ((b_val as f64 * 1e7).round() as i64);
    let absdiff: f64 = ((a_val as f64) - (b_val as f64)).abs();

    let intersection_band_ok = a_val >= NRMSE_BAND_LOWER && a_val <= NRMSE_BAND_UPPER;

    eprintln!(
        "[intersection] sweep_A/kvl=500 nrmse_A={:.7} sweep_B/abs_pos=500 nrmse_B={:.7} absdiff={:.2e} match_7dp={} (H2: two independent RNG states; mismatch is expected)",
        a_val, b_val, absdiff, match_to_7_decimal_places
    );

    // Band check for intersection (sweep_A measurement).
    if !intersection_band_ok {
        eprintln!(
            "BAND_PRE_FALSIFIED: intersection abs_pos=500 kvl=500 sweep_A_nrmse={:.7} outside band [{}, {}]",
            a_val, NRMSE_BAND_LOWER, NRMSE_BAND_UPPER
        );
    }

    // STEP 7: Ring-wrap legs.
    // M1 (catalog #8): kvl_logical MUST be < sliding_window for mask to differentiate slot
    //   chronology. iter-5 had kvl=1024 >= sliding_window=512 — degenerate (both ring_start
    //   formulas expose the full slot set). Fixed: kvl=256 < sliding_window=512.
    //   This is synthetic (production at abs_pos=1024 would have kvl=1024) but required by
    //   catalog #8 for chronology differences to physically manifest in the mask.
    // H3 (catalog #16): draw K/V/Q ONCE per abs_pos using an RNG clone/restore mechanism,
    //   then dispatch kernel TWICE with override_ring_start=Some(ring_start_a) and
    //   override_ring_start=Some(ring_start_b) on BYTE-IDENTICAL data.
    //   ab_delta = |nrmse_a - nrmse_b| measures kernel sensitivity to ring_start, not RNG noise.
    let ring_wrap_points = [(1024usize, 512u32), (1050usize, 512u32)];
    let ring_wrap_kvl: usize = 256; // strictly < sliding_window=512 (catalog #8 / M1)
    let mut ring_wrap: Vec<serde_json::Value> = Vec::new();

    for (abs_pos, sw) in ring_wrap_points {
        let kvl = ring_wrap_kvl; // 256 < 512 (M1 fix: catalog #8)
        let ring_start_a: u32 = if abs_pos + 1 >= kvc { ((abs_pos + 1) % kvc) as u32 } else { 0 };
        let ring_start_b: u32 = if abs_pos + 1 > kvc  { (abs_pos % kvc) as u32 } else { 0 };
        let before_count = rng_counter;

        // H3: Save RNG state before drawing data for this abs_pos.
        // Xoshiro256StarStar derives Clone (added in iter-6), so we can snapshot and restore.
        let rng_snapshot = rng.clone();
        let counter_snapshot = rng_counter;

        // First invocation: ring_start_a (production formula). Advances rng.
        let nrmse_a = run_sweep_point(
            &mut rng, &mut rng_counter,
            abs_pos, kvl, kvc, sw,
            Some(ring_start_a), // H3: override_ring_start flows to kernel dispatch + CPU oracle
            device, registry,
        );

        // H3: Restore RNG to pre-draw state so nrmse_b uses BYTE-IDENTICAL K/V/Q data.
        // The rng state is reset to exactly what it was before the A draw, then B draws
        // the same sequence — only ring_start differs in the kernel dispatch and oracle layout.
        rng = rng_snapshot;
        rng_counter = counter_snapshot;

        // Second invocation: ring_start_b (alternative formula). Same data as A.
        let nrmse_b = run_sweep_point(
            &mut rng, &mut rng_counter,
            abs_pos, kvl, kvc, sw,
            Some(ring_start_b), // H3: different ring_start, same K/V/Q data
            device, registry,
        );

        let ab_delta = (nrmse_a - nrmse_b).abs();

        eprintln!(
            "[ring_wrap] abs_pos={} kvl={} sw={} ring_start_A={} ring_start_B={} nrmse_a={:.7} nrmse_b={:.7} ab_delta={:.2e} (H3: byte-identical data, different ring_start)",
            abs_pos, kvl, sw, ring_start_a, ring_start_b, nrmse_a, nrmse_b, ab_delta
        );

        let band_ok_a = nrmse_a >= NRMSE_BAND_LOWER && nrmse_a <= NRMSE_BAND_UPPER;
        let band_ok_b = nrmse_b >= NRMSE_BAND_LOWER && nrmse_b <= NRMSE_BAND_UPPER;

        if !band_ok_a {
            eprintln!(
                "BAND_PRE_FALSIFIED: ring_wrap abs_pos={} nrmse_a={:.7} outside band [{}, {}]",
                abs_pos, nrmse_a, NRMSE_BAND_LOWER, NRMSE_BAND_UPPER
            );
        }
        if !band_ok_b {
            eprintln!(
                "BAND_PRE_FALSIFIED: ring_wrap abs_pos={} nrmse_b={:.7} outside band [{}, {}]",
                abs_pos, nrmse_b, NRMSE_BAND_LOWER, NRMSE_BAND_UPPER
            );
        }

        ring_wrap.push(serde_json::json!({
            "abs_pos": abs_pos,
            "kvl_logical": kvl,          // 256 < sliding_window=512 (M1 / catalog #8)
            "sliding_window": sw,         // 512
            "ring_start_a": ring_start_a,
            "ring_start_b": ring_start_b,
            "ring_start_A_passed_to_kernel": ring_start_a, // R-13: emitted for AC-11 verification
            "ring_start_B_passed_to_kernel": ring_start_b, // R-13: emitted for AC-11 verification
            "ring_start_A_nrmse": nrmse_a,
            "ring_start_B_nrmse": nrmse_b,
            "ab_delta": ab_delta,
            "band_ok": band_ok_a && band_ok_b,
            "rng_u64s_consumed_before": before_count,
            "h3_data_reuse": "byte-identical K/V/Q via RNG clone/restore; only ring_start differs",
        }));
    }

    // STEP 8: Verdict classification — deterministic from measured matrix (catalog #9).
    // Exactly one of four declared strings.
    let all_band_ok = sweep_a.iter().all(|r| r.band_ok)
        && sweep_b.iter().all(|r| r.band_ok)
        && ring_wrap.iter().all(|v| v["band_ok"].as_bool().unwrap_or(false));

    let verdict: &str = if !all_band_ok {
        // One or more sweep points are out of band: verdict is BAND_PRE_FALSIFIED.
        // Binary will exit with code 2 after writing audit.json.
        "BAND_PRE_FALSIFIED"
    } else {
        // Spearman rho for sweep_A (length effect): monotone-rising → rho > 0.7
        // H2: kvl=500 is NOW a real sweep_A row; include all 6 rows in the Spearman analysis.
        let sweep_a_nrmse: Vec<f32> = sweep_a.iter()
            .map(|r| r.nrmse).collect();
        let n_a = sweep_a_nrmse.len() as f32;
        let spearman_rho_a = if n_a >= 2.0 {
            // Rank correlation: rank each element, compute rho.
            let mut indexed: Vec<(usize, f32)> = sweep_a_nrmse.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut ranks = vec![0.0f32; indexed.len()];
            for (rank, (idx, _)) in indexed.iter().enumerate() {
                ranks[*idx] = rank as f32 + 1.0;
            }
            // Spearman = 1 - 6*sum_d2 / (n*(n^2-1))
            let natural_ranks: Vec<f32> = (1..=indexed.len()).map(|i| i as f32).collect();
            let sum_d2: f32 = ranks.iter().zip(natural_ranks.iter())
                .map(|(r, nr)| (r - nr).powi(2)).sum();
            1.0 - 6.0 * sum_d2 / (n_a * (n_a * n_a - 1.0))
        } else { 0.0 };

        // Sweep B range for phase effect.
        let sweep_b_nrmse_core: Vec<f32> = sweep_b.iter().map(|r| r.nrmse).collect();
        let sweep_b_max = sweep_b_nrmse_core.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sweep_b_min = sweep_b_nrmse_core.iter().cloned().fold(f32::INFINITY, f32::min);
        let sweep_b_range = sweep_b_max - sweep_b_min;

        // H2: all 6 sweep_A rows included (kvl=500 is now a real row, not a phantom).
        let sweep_a_range_core: Vec<f32> = sweep_a.iter()
            .map(|r| r.nrmse).collect();
        let sweep_a_max = sweep_a_range_core.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sweep_a_min = sweep_a_range_core.iter().cloned().fold(f32::INFINITY, f32::min);
        let sweep_a_range = sweep_a_max - sweep_a_min;

        eprintln!("[verdict] spearman_rho_A={:.4} sweep_B_range={:.4} sweep_A_range={:.4}", spearman_rho_a, sweep_b_range, sweep_a_range);

        if spearman_rho_a > 0.7 && sweep_b_range < 0.05 {
            "LENGTH_EFFECT_CONFIRMED"
        } else if sweep_b_range > 0.10 && sweep_a_range < 0.05 {
            "PHASE_EFFECT_CONFIRMED"
        } else {
            "FLOOR_IS_PHYSICS_CONSISTENT"
        }
    };

    eprintln!("[pf] verdict = {}", verdict);

    // STEP 9: Regime documentation.
    // V-norm policy: forward_mlx.rs:1167-1205 direct read confirms V gets
    // dispatch_rms_norm_unit_perhead (unit weights, no learned weight tensor for V).
    // Q/K: forward_mlx.rs:1144-1165 uses learned q_norm_weight/k_norm_weight.
    // Iter-6 uses unit weights for Q/K (GGUF not present on test machine).
    let regime = serde_json::json!({
        "rmsnorm_weights": "unit_fallback",
        "rmsnorm_weights_reason": "Gemma-4-27B GGUF not present on test machine; GGUF extraction path non-trivial. Q/K use unit RMSNorm weights (learned q_norm_weight/k_norm_weight available in production at forward_mlx.rs:1148,1159). Delta vs learned: unit weights normalize Q/K to unit sphere before RoPE; learned weights add a per-element multiplicative scale. For N(0,1) synthetic inputs the difference is O(weight_scale - 1), typically <5% for near-unit weights in trained models. Regime-faithful in shape/scale/eps/formula.",
        "v_norm_policy": "unit_weights_per_production",
        "v_norm_evidence": "forward_mlx.rs:1178-1205 direct read: dispatch_rms_norm_unit_perhead on V (no learned v_norm_weight tensor; Gemma 4 has no v_norm_weight per spec grep confirming only q_norm_weight and k_norm_weight at forward_mlx.rs:289-290,714-719).",
        "scale": 1.0,
        "scale_evidence": "forward_mlx.rs:1617 (dense), forward_mlx.rs:1664 (TQ), ADR-005:1181",
        "rms_norm_eps": 1e-6,
        "rms_norm_eps_evidence": "config.rs:100",
        "rope_theta": 10000.0,
        "rope_theta_evidence": "config.rs:101 (rope_theta_sliding=10000)",
        "rope_convention": "NeoX half-split, applied to Q at abs_pos and K at chronological position",
        "shapes": {"num_heads": 16, "num_kv_heads": 8, "head_dim": 256, "kv_capacity": 1024},
        "dense_floor_reference": "POST-RMSNorm POST-RoPE F32 Q/K/V (same tensors fed to hadamard_quantize_kv) — catalog #7 upstream-independent reference",
        "mask_type": 2,
        "mask_type_evidence": "forward_mlx.rs:1665 (mask_type=2 for sliding layers)",
        "softcap": 0.0,
        // M1 / catalog #8: ring_wrap uses kvl=256 < sliding_window=512 so chronology manifests.
        "ring_wrap_kvl_reason": "catalog #8: ring-chronology tests need kvl_logical < sliding_window to manifest. At kvl_logical >= sliding_window both ring_start formulas expose the full slot set; chronology differences physically cannot show. ring_wrap uses kvl=256 < sliding_window=512 (synthetic; production at abs_pos=1024 would have kvl=1024, but the A/B test is measuring kernel dispatch sensitivity to ring_start, which requires mask differentiation of slot chronology).",
    });

    // STEP 10: Write audit.json — SOLE reporting artifact (catalog #17 / M2).
    // No "pending", "TBD", or "pending_manual_run" strings anywhere (catalog #12, AC-3).
    // M2: sweep_A and sweep_B are embedded as arrays in audit.json; NO sidecar CSVs written.
    let audit = serde_json::json!({
        "session": "cfa-20260422-C4t3i6-evidence-package-integrity",
        "iter": 6,
        "ran_at": SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "verdict": verdict,
        "band": {
            "lower": NRMSE_BAND_LOWER,
            "upper": NRMSE_BAND_UPPER,
            "registration": "pre-registered as const f32 at module scope before any measurement; catalog #11",
        },
        "regression_gates": regression_gates,
        "regime": regime,
        "sweep_A": sweep_a,
        "sweep_B": sweep_b,
        "ring_wrap": ring_wrap,
        "rng": {
            "seed_literal": "0x00C2_5EED_u64",
            "algorithm": "Xoshiro256StarStar",
            "single_instance": true,
            "total_u64s_consumed": rng_counter,
        },
        "intersection_check": {
            "abs_pos": 500,
            "kvl_logical": 500,
            // H2 (catalog #15): sweep_A_intersection_nrmse and sweep_B_intersection_nrmse are
            // INDEPENDENTLY MEASURED from distinct RNG states (sweep_A draws first; sweep_B draws
            // after sweep_A has advanced the RNG). A mismatch is the EXPECTED honest outcome.
            "sweep_A_intersection_nrmse": a_val,   // from sweep_A's kvl=500 row (independently measured)
            "sweep_B_intersection_nrmse": b_val,   // from sweep_B's abs_pos=500 row (different RNG state)
            // Binary-computed equality — NEVER hardcoded (catalog #15 / AC-8).
            "match_to_7_decimal_places": match_to_7_decimal_places,
            "absdiff": absdiff,     // numeric distance for AC-7 verification
            "band_ok": intersection_band_ok,
            "note": "H2 / catalog #15: (abs_pos=500, kvl=500) is measured TWICE with distinct RNG states. sweep_A includes kvl=500 as of iter-6 (6 rows total). sweep_B has abs_pos=500 as its 4th row. The two values come from different RNG advances so a numerical mismatch is expected and is the HONEST outcome. match_to_7_decimal_places is COMPUTED, not hardcoded.",
        },
        "mistakes_catalog_citations": [
            "#3: Verdict gates too loose — tighten to physics-justified narrow bands",
            "#8: Ring-chronology tests need kvl_logical < sliding_window to manifest",
            "#9: Narrative overclaim vs code-generated evidence — emit statuses from binary",
            "#11: Pre-registered asserts bands — never widen after measurement (iter-4 HIGH-1 defect)",
            "#12: Regression-gate statuses MUST be binary-emitted, not narrative-injected (iter-4 HIGH-2 defect)",
            "#13: Non-controlled sweeps confound the claim — fix seed, vary one param only (iter-4 MED defect)",
            "#14: Subprocess gates must run against the worktree, not a hardcoded other checkout (iter-5 HIGH-1 defect)",
            "#15: Copied-intersection-as-determinism-tautology — both sweeps must independently measure (iter-5 HIGH-2 defect)",
            "#16: Ring-wrap A/B without independent ring_start control is measuring RNG noise (iter-5 HIGH-3 defect)",
            "#17: Parallel artifact sources-of-truth violate single-source evidence discipline (iter-5 MED-2 defect)",
            "meta-class: report-vs-measurement drift — every field in audit.json must correspond to a real function evaluation at measurement time; no pre-computed, copied, or constructed values",
        ],
        // M2 (catalog #17): CSV-equivalent documentation for downstream jq post-processing.
        // The binary writes ONLY audit.json. No sidecar CSVs. R-15 / AC-13 / AC-14.
        "csv_equivalent": {
            "sweep_a_columns": ["abs_pos", "kvl_logical", "sliding_window", "nrmse", "band_ok", "rng_u64s_consumed_before"],
            "sweep_b_columns": ["abs_pos", "kvl_logical", "sliding_window", "nrmse", "band_ok", "rng_u64s_consumed_before"],
            "ring_wrap_columns": ["abs_pos", "kvl_logical", "sliding_window", "ring_start_a", "ring_start_b", "ring_start_A_nrmse", "ring_start_B_nrmse", "ab_delta", "band_ok", "rng_u64s_consumed_before"],
            "note": "sweep_A and sweep_B arrays in this audit.json are the canonical source. jq one-liner: jq -r '.sweep_A[] | [.abs_pos,.kvl_logical,.sliding_window,.nrmse,.band_ok,.rng_u64s_consumed_before] | @csv' audit.json",
        },
    });

    let audit_json = serde_json::to_string_pretty(&audit).expect("serialize audit");

    // M2 (catalog #17): write ONLY audit.json — the SOLE reporting artifact.
    // Sidecar sweep_a.csv / sweep_b.csv REMOVED in iter-6. Use csv_equivalent.note for jq.
    let audit_path = out_dir.join("audit.json");
    fs::write(&audit_path, audit_json.as_bytes())
        .unwrap_or_else(|e| panic!("failed to write audit.json: {}", e));
    eprintln!("[pf] audit.json written to {:?} (sole artifact; no sidecar CSVs — catalog #17 / M2)", audit_path);

    // Print summary to stdout.
    println!("=== iter-5 production-faithful verdict: {} ===", verdict);
    println!("sweep_A (abs_pos=500, vary kvl):");
    for r in &sweep_a {
        println!("  kvl={} nrmse={:.7} band_ok={}", r.kvl_logical, r.nrmse, r.band_ok);
    }
    println!("sweep_B (kvl=~500, vary abs_pos):");
    for r in &sweep_b {
        println!("  abs_pos={} kvl={} nrmse={:.7} band_ok={}", r.abs_pos, r.kvl_logical, r.nrmse, r.band_ok);
    }
    println!("ring_wrap:");
    for rw in &ring_wrap {
        println!("  abs_pos={} sw={} nrmse_a={:.7} nrmse_b={:.7} ab_delta={:.2e}",
            rw["abs_pos"], rw["sliding_window"],
            rw["ring_start_A_nrmse"].as_f64().unwrap_or(0.0),
            rw["ring_start_B_nrmse"].as_f64().unwrap_or(0.0),
            rw["ab_delta"].as_f64().unwrap_or(0.0));
    }

    // Exit with appropriate code.
    let exit_code = if verdict == "BAND_PRE_FALSIFIED" { 2i32 } else { 0i32 };
    if exit_code != 0 {
        eprintln!("[pf] exiting with code {} (BAND_PRE_FALSIFIED)", exit_code);
        std::process::exit(exit_code);
    }
    eprintln!("[pf] complete — exit 0");
}
