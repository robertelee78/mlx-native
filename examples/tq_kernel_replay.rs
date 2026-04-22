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
//!   D5 - kv_seq_len=23 accepted from manifest (was 22); CPU reference loops 0..kvl.
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

struct Args {
    manifest: PathBuf,
    variation: Variation,
    canary: CanaryMode,
    out: PathBuf,
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
            other => return Err(format!("unknown argument: {}", other)),
        }
        i += 1;
    }

    Ok(Args {
        manifest: manifest.ok_or("--manifest is required")?,
        variation: variation.ok_or("--variation is required")?,
        canary,
        out: out.ok_or("--out is required")?,
    })
}

// ---------------------------------------------------------------------------
// Manifest schema
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

#[derive(Debug, Deserialize)]
struct ManifestInputs {
    q_natural: String,
    k_packed_padded: String,
    v_packed_padded: String,
    k_norms_padded: String,
    v_norms_padded: String,
    // Legacy out-of-range canary files (optional — kept for backward compat)
    #[serde(default)]
    k_norms_canary: String,
    #[serde(default)]
    v_norms_canary: String,
    // In-range canary: harness constructs this in-memory; path in manifest is optional
    #[serde(default)]
    k_norms_canary_in_range: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ManifestCompactSources {
    k_packed_compact: String,
    v_packed_compact: String,
    k_norms_compact: String,
    v_norms_compact: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    params: ManifestParams,
    inputs: ManifestInputs,
    compact_sources: ManifestCompactSources,
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
    nrmse: f64,
    max_abs_diff: f32,
    per_head_max_abs_diff: Vec<PerHeadDiff>,
    any_nan_inf_in_gpu_output: bool,
    exit_status: String,
    bin_path: String,
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
/// k_dequant: [num_kv_heads * kv_seq_len] entries of [head_dim] each
/// v_dequant: same layout
///
/// Returns: flat [num_heads * head_dim] F32
fn cpu_sdpa(
    q: &[f32],
    k_dequant: &[Vec<f32>],
    v_dequant: &[Vec<f32>],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; num_heads * head_dim];
    let heads_per_kv = num_heads / num_kv_heads;

    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_offset = h * head_dim;

        let mut scores = Vec::with_capacity(kv_seq_len);
        for p in 0..kv_seq_len {
            let mut dot = 0.0f32;
            for c in 0..head_dim {
                dot += q[q_offset + c] * k_dequant[kv_h * kv_seq_len + p][c];
            }
            scores.push(dot * scale);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        if sum > 0.0 {
            for e in &mut exp_scores {
                *e /= sum;
            }
        }

        let o_offset = h * head_dim;
        for p in 0..kv_seq_len {
            let w = exp_scores[p];
            for c in 0..head_dim {
                output[o_offset + c] += w * v_dequant[kv_h * kv_seq_len + p][c];
            }
        }
    }

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
// Core replay logic
// ---------------------------------------------------------------------------

fn run_variation(
    manifest: &Manifest,
    variation: Variation,
    canary: CanaryMode,
    out_path: &PathBuf,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> ReplayMetrics {
    let p = &manifest.params;
    let inp = &manifest.inputs;

    let nh = p.num_heads as usize;
    let nkv = p.num_kv_heads as usize;
    let hd = p.head_dim as usize;
    let kvl = p.kv_seq_len as usize; // now 23 per C-1-unlock spec
    let kv_capacity = p.kv_capacity as usize;

    // --- Load inputs ---
    let q_natural: Vec<f32> = load_f32(&inp.q_natural);
    assert_eq!(q_natural.len(), nh * hd, "q_natural size mismatch");

    // Padded packed K/V buffers [nkv, kv_capacity, hd/2]
    let k_packed_padded: Vec<u8> = load_u8(&inp.k_packed_padded);
    let v_packed_padded: Vec<u8> = load_u8(&inp.v_packed_padded);
    assert_eq!(k_packed_padded.len(), nkv * kv_capacity * (hd / 2));
    assert_eq!(v_packed_padded.len(), nkv * kv_capacity * (hd / 2));

    // --- Load compact TQ-packed K/V for CPU reference (kvl rows, compact stride) ---
    // Compact stride: element (kv_h, pos, byte) at offset kv_h*kvl*(hd/2) + pos*(hd/2) + byte
    let k_packed_compact: Vec<u8> = load_u8(&manifest.compact_sources.k_packed_compact);
    let v_packed_compact: Vec<u8> = load_u8(&manifest.compact_sources.v_packed_compact);
    let k_norms_compact: Vec<f32> = load_f32(&manifest.compact_sources.k_norms_compact);
    let v_norms_compact: Vec<f32> = load_f32(&manifest.compact_sources.v_norms_compact);
    assert_eq!(k_packed_compact.len(), nkv * kvl * (hd / 2));
    assert_eq!(v_packed_compact.len(), nkv * kvl * (hd / 2));
    assert_eq!(k_norms_compact.len(), nkv * kvl);
    assert_eq!(v_norms_compact.len(), nkv * kvl);

    // --- Compute CPU reference: dequantize TQ-packed (kvl rows) → natural-basis K/V ---
    // This is the authoritative reference — mirrors test_flash_attn_vec_tq.rs lines 127-200.
    // CPU reference is the same for all variations (A, B, C): natural-basis SDPA from TQ dequant.
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
        p.scale,
    );

    // --- Load norms (padded); apply canary mutation if requested ---
    // Baseline norms from padded files
    let mut k_norms_padded: Vec<f32> = load_f32(&inp.k_norms_padded);
    let mut v_norms_padded: Vec<f32> = load_f32(&inp.v_norms_padded);
    assert_eq!(k_norms_padded.len(), nkv * kv_capacity);
    assert_eq!(v_norms_padded.len(), nkv * kv_capacity);

    match canary {
        CanaryMode::None => {
            // No mutation — use baseline norms as-is
        }
        CanaryMode::InRange => {
            // D3: in-range canary — mutate k_norms at (head=0, pos=10).
            // pos=10 is within kv_seq_len=23, so the kernel provably reads this position.
            // Mutation: scale norm by 2x → dequantized K[h=0, pos=10, :] magnitudes ~2x.
            // Expected: nrmse_delta vs A baseline > 0.01 (kernel reads the mutated value).
            // Mirror canary_spec from queen's spec: k_norms_padded[0 * kv_capacity + 10] *= 2.0
            let canary_idx = 0 * kv_capacity + 10;
            k_norms_padded[canary_idx] *= 2.0;
            eprintln!(
                "canary in-range: k_norms[head=0, pos=10] *= 2.0 → new value = {}",
                k_norms_padded[canary_idx]
            );
        }
        CanaryMode::OutOfRange => {
            // Legacy out-of-range canary: load from manifest file if available,
            // otherwise set norms at positions >= kvl to 1e9.
            if !inp.k_norms_canary.is_empty() && !inp.v_norms_canary.is_empty() {
                k_norms_padded = load_f32(&inp.k_norms_canary);
                v_norms_padded = load_f32(&inp.v_norms_canary);
                assert_eq!(k_norms_padded.len(), nkv * kv_capacity);
                assert_eq!(v_norms_padded.len(), nkv * kv_capacity);
            } else {
                // Construct in-memory: positions >= kvl set to 1e9
                for kv_h in 0..nkv {
                    for pos in kvl..kv_capacity {
                        k_norms_padded[kv_h * kv_capacity + pos] = 1e9;
                        v_norms_padded[kv_h * kv_capacity + pos] = 1e9;
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

    // Norms: [nkv, kv_capacity] f32 (used by A/B; includes canary mutation if active)
    let norms_bytes = nkv * kv_capacity * 4;

    let mut k_norms_buf = device
        .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kv_capacity])
        .expect("alloc K norms");
    k_norms_buf.as_mut_slice::<f32>().expect("write K norms")
        .copy_from_slice(&k_norms_padded);

    let mut v_norms_buf = device
        .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kv_capacity])
        .expect("alloc V norms");
    v_norms_buf.as_mut_slice::<f32>().expect("write V norms")
        .copy_from_slice(&v_norms_padded);

    // Output buffer: [nh, 1, hd] F32
    let output_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc output");

    // Tmp buffer for TQ SDPA kernel
    let tmp_bytes_tq = flash_attn_vec_tq::tmp_buffer_bytes(p.num_heads, p.head_dim);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes_tq, DType::F32, vec![tmp_bytes_tq / 4])
        .expect("alloc tmp");

    // --- Build TQ SDPA params struct from manifest (mirrors forward_mlx.rs:1452-1462) ---
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
            // This deliberately mismatches the kernel's assumption that Q is pre-rotated.
            // Only barrier_2 equivalent needed: publish packed K/V + norms before kernel reads.
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
            // This isolates whether TQ packed-read path / FWHT bracketing is the locus.
            //
            // Allocate F32 dense K/V buffers: [nkv, kv_capacity, hd]
            // Fill positions 0..kvl from k_dequant / v_dequant (already natural basis).
            // Leave positions kvl..kv_capacity as 0.0f32.
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
                    // Positions kvl..kv_capacity remain 0.0f32 (already zero from alloc)
                }
            }

            // Tmp buffer for dense flash_attn_vec kernel
            let tmp_bytes_dense = flash_attn_vec::tmp_buffer_bytes(p.num_heads, p.head_dim);
            let tmp_dense_buf = device
                .alloc_buffer(tmp_bytes_dense, DType::F32, vec![tmp_bytes_dense / 4])
                .expect("alloc tmp dense");

            // Dense flash_attn_vec params — no ring_start field (implicit 0 when kv_seq_len < kv_capacity)
            let dense_params = FlashAttnVecParams {
                num_heads: p.num_heads,
                num_kv_heads: p.num_kv_heads,
                head_dim: p.head_dim,
                kv_seq_len: p.kv_seq_len,
                kv_capacity: p.kv_capacity,
                scale: p.scale,
                mask_type: p.mask_type,       // 2 (sliding window)
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

    // --- Compute metrics ---
    let (nrmse, max_abs_diff, per_head) =
        compute_metrics(&cpu_ref, &gpu_output, nh, hd);

    // --- D4: Write raw sdpa_out .bin alongside the metrics JSON ---
    // Format: raw F32 little-endian, shape [nh, 1, hd] = nh*hd*4 bytes = 16384 bytes for nh=16, hd=256
    let gpu_out_bytes: Vec<u8> = gpu_output
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    // Derive bin path from out_path: strip any extension, append _sdpa_out.bin
    let out_stem = out_path.with_extension("");
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
            eprintln!(
                "usage: tq_kernel_replay --manifest <path> --variation <A|B|C> [--canary in-range|out-of-range] --out <path>"
            );
            std::process::exit(1);
        }
    };

    // Load manifest
    let manifest_bytes = fs::read(&args.manifest).unwrap_or_else(|e| {
        eprintln!("failed to read manifest {:?}: {}", args.manifest, e);
        std::process::exit(1);
    });
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap_or_else(|e| {
        eprintln!("failed to parse manifest: {}", e);
        std::process::exit(1);
    });

    // Initialise Metal device and kernel registry
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();
    // TQ SDPA kernel
    flash_attn_vec_tq::register(&mut registry);
    // Dense flash_attn_vec kernel (used by Variation C)
    mlx_native::ops::flash_attn_vec::register(&mut registry);
    // fwht_standalone kernels are pre-registered inside KernelRegistry::new()

    eprintln!(
        "tq_kernel_replay: variation={} canary={:?} manifest={:?}",
        args.variation, args.canary, args.manifest
    );

    let metrics = run_variation(
        &manifest,
        args.variation,
        args.canary,
        &args.out,
        &device,
        &mut registry,
    );

    // Print summary to stdout
    let json = serde_json::to_string_pretty(&metrics).expect("serialize metrics");
    println!("{}", json);

    // Write metrics JSON to --out path (with .json extension if not already present)
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
        "RESULT: variation={} canary={:?} nrmse={:.6} max_abs_diff={:.6} nan_inf={}",
        metrics.variation, args.canary, metrics.nrmse, metrics.max_abs_diff,
        metrics.any_nan_inf_in_gpu_output
    );
    eprintln!("metrics written to {:?}", out_json);
}
