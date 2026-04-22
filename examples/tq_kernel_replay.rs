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

    // Validate: singlestep requires --manifest; multistep does not.
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
    }
}

// ---------------------------------------------------------------------------
// Multistep driver (P3b)
// ---------------------------------------------------------------------------

use serde_json::Value as JsonValue;

/// Multistep output row in JSON.
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

/// Seeded Box-Muller Gaussian PRNG — matches the deterministic seed spec.
/// Uses StdRng::seed_from_u64(seed) from the `rand` crate path re-exported
/// by mlx-native (or we implement our own if not available).
fn seeded_gaussian(seed: u64, n: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Simple deterministic Box-Muller using a Lehmer/LCG sequence seeded by seed.
    // Uses a linear congruential generator for portability without external deps.
    let mut state: u64 = seed ^ 0x9e3779b97f4a7c15;
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
        // Sub-seed: seed_base XOR (abs_pos << 16) for K; XOR (abs_pos << 32) for V; XOR (abs_pos << 48) for Q.
        let k_seed = seed_base ^ (abs_pos << 16);
        let v_seed = seed_base ^ (abs_pos << 32);
        let q_seed = seed_base ^ (abs_pos << 48) ^ 0xABCDEF;

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

    // Overall decision-tree verdict.
    let has_divergence = rows.iter().any(|r| r.dequant_oracle_nrmse >= 0.01 || r.independent_floor_nrmse >= 0.01);
    let ring_wrap_only = rows.iter().filter(|r| r.abs_pos > 1000).any(|r| r.dequant_oracle_nrmse >= 0.01)
        && rows.iter().filter(|r| r.abs_pos <= 500).all(|r| r.dequant_oracle_nrmse < 0.01);
    let overall = if !has_divergence {
        "kernel_end_to_end_correct"
    } else if ring_wrap_only {
        "ring_start_or_dispatch_bug"
    } else {
        "h1_kernel_bug"
    };
    eprintln!("OVERALL decision-tree branch: {}", overall);

    // Suppress unused import warning.
    let _: Option<JsonValue> = None;
}
