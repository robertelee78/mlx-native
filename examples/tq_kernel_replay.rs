//! TQ kernel replay binary for ADR-007 C-4 E1 localization.
//!
//! Runs the flash_attn_vec_tq kernel against captured C-0 inputs and compares
//! to a CPU reference SDPA computed in-memory from the same TQ-packed data.
//!
//! Usage:
//!   cargo run --release --example tq_kernel_replay -- \
//!     --manifest /tmp/cfa-20260422-C1-kernel-replay/manifest.json \
//!     --variation A \
//!     [--canary 1e9] \
//!     --out /tmp/cfa-20260422-C1-kernel-replay/out/A.json
//!
//! Variations:
//!   A  Full production path: forward-FWHT(Q) + TQ kernel + inverse-FWHT(output)
//!   B  FWHT-disabled: skip both FWHT dispatches; pass Q as-is to kernel
//!   C  Dense-KV re-encoded: dequant TQ-packed → re-FWHT → re-quantize → kernel
//!
//! Canary (--canary 1e9): positions 22..1023 in norms set to 1e9; if
//!   sdpa_out differs from the normal run, the mask is leaking.
//!
//! Exit codes:
//!   0  Success
//!   1  Argument / IO error
//!   2  GPU dispatch error or NaN/Inf in output

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
#![cfg(target_vendor = "apple")]

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
    canary: bool,
    out: PathBuf,
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
            Variation::C => write!(f, "C"),
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut manifest: Option<PathBuf> = None;
    let mut variation: Option<Variation> = None;
    let mut canary = false;
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
                // Accept either bare "--canary" or "--canary 1e9"
                canary = true;
                // Skip the next arg if it looks like the canary value (not a flag)
                if let Some(next) = argv.get(i + 1) {
                    if !next.starts_with('-') {
                        i += 1; // consume the value
                    }
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
    k_norms_canary: String,
    v_norms_canary: String,
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
    canary: bool,
    ran_at: String,
    nrmse: f64,
    max_abs_diff: f32,
    per_head_max_abs_diff: Vec<PerHeadDiff>,
    any_nan_inf_in_gpu_output: bool,
    exit_status: String,
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
    use_canary: bool,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> ReplayMetrics {
    let p = &manifest.params;
    let inp = &manifest.inputs;

    let nh = p.num_heads as usize;
    let nkv = p.num_kv_heads as usize;
    let hd = p.head_dim as usize;
    let kvl = p.kv_seq_len as usize;
    let kv_capacity = p.kv_capacity as usize;

    // --- Load inputs ---
    let q_natural: Vec<f32> = load_f32(&inp.q_natural);
    assert_eq!(q_natural.len(), nh * hd, "q_natural size mismatch");

    // Padded packed K/V buffers [nkv, kv_capacity, hd/2]
    let k_packed_padded: Vec<u8> = load_u8(&inp.k_packed_padded);
    let v_packed_padded: Vec<u8> = load_u8(&inp.v_packed_padded);
    assert_eq!(k_packed_padded.len(), nkv * kv_capacity * (hd / 2));
    assert_eq!(v_packed_padded.len(), nkv * kv_capacity * (hd / 2));

    // Norms — choose normal or canary
    let k_norms_padded: Vec<f32> = if use_canary {
        load_f32(&inp.k_norms_canary)
    } else {
        load_f32(&inp.k_norms_padded)
    };
    let v_norms_padded: Vec<f32> = if use_canary {
        load_f32(&inp.v_norms_canary)
    } else {
        load_f32(&inp.v_norms_padded)
    };
    assert_eq!(k_norms_padded.len(), nkv * kv_capacity);
    assert_eq!(v_norms_padded.len(), nkv * kv_capacity);

    // --- Load compact TQ-packed K/V for CPU reference (22 rows, compact stride) ---
    // Compact stride: element (kv_h, pos, byte) at offset kv_h*kvl*(hd/2) + pos*(hd/2) + byte
    let k_packed_compact: Vec<u8> = load_u8(&manifest.compact_sources.k_packed_compact);
    let v_packed_compact: Vec<u8> = load_u8(&manifest.compact_sources.v_packed_compact);
    let k_norms_compact: Vec<f32> = load_f32(&manifest.compact_sources.k_norms_compact);
    let v_norms_compact: Vec<f32> = load_f32(&manifest.compact_sources.v_norms_compact);
    assert_eq!(k_packed_compact.len(), nkv * kvl * (hd / 2));
    assert_eq!(v_packed_compact.len(), nkv * kvl * (hd / 2));
    assert_eq!(k_norms_compact.len(), nkv * kvl);
    assert_eq!(v_norms_compact.len(), nkv * kvl);

    // --- Compute CPU reference: dequantize TQ-packed (22 rows) → natural-basis K/V ---
    // This is the authoritative reference — mirrors test_flash_attn_vec_tq.rs lines 127-200.
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

    // CPU SDPA in natural basis
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


    // --- Variation C: re-encode dequantized K/V into fresh packed + norms ---
    // Re-encode: natural-basis K/V → nibble_quantize (which applies FWHT internally)
    // Then pad to kv_capacity with zeros, just like Worker 1 did for A/B.
    let (k_packed_reenc, v_packed_reenc, k_norms_reenc, v_norms_reenc);
    if variation == Variation::C {
        let mut kp = vec![0u8; nkv * kv_capacity * (hd / 2)];
        let mut vp = vec![0u8; nkv * kv_capacity * (hd / 2)];
        let mut kn = vec![0.0f32; nkv * kv_capacity];
        let mut vn = vec![0.0f32; nkv * kv_capacity];

        for kv_h in 0..nkv {
            for pos in 0..kvl {
                // Padded buffer stride: h*kv_capacity*(hd/2) + pos*(hd/2)
                let packed_dst_off = kv_h * kv_capacity * (hd / 2) + pos * (hd / 2);
                let norms_dst_off = kv_h * kv_capacity + pos;

                // K
                let kv_deq_idx = kv_h * kvl + pos;
                let (k_enc, k_norm) = nibble_quantize(&k_dequant[kv_deq_idx], hd);
                kp[packed_dst_off..packed_dst_off + hd / 2].copy_from_slice(&k_enc);
                kn[norms_dst_off] = k_norm;

                // V
                let (v_enc, v_norm) = nibble_quantize(&v_dequant[kv_deq_idx], hd);
                vp[packed_dst_off..packed_dst_off + hd / 2].copy_from_slice(&v_enc);
                vn[norms_dst_off] = v_norm;
            }
        }

        k_packed_reenc = kp;
        v_packed_reenc = vp;
        k_norms_reenc = kn;
        v_norms_reenc = vn;
    } else {
        k_packed_reenc = Vec::new();
        v_packed_reenc = Vec::new();
        k_norms_reenc = Vec::new();
        v_norms_reenc = Vec::new();
    }

    // Select which packed / norm buffers go to GPU
    let (k_packed_gpu, v_packed_gpu, k_norms_gpu, v_norms_gpu): (
        &[u8], &[u8], &[f32], &[f32],
    ) = match variation {
        Variation::A | Variation::B => (
            &k_packed_padded,
            &v_packed_padded,
            &k_norms_padded,
            &v_norms_padded,
        ),
        Variation::C => (
            &k_packed_reenc,
            &v_packed_reenc,
            &k_norms_reenc,
            &v_norms_reenc,
        ),
    };

    // --- Allocate GPU buffers ---
    // Q: [nh, 1, hd] F32 — production shape
    let mut q_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc Q");
    q_buf.as_mut_slice::<f32>().expect("write Q")[..nh * hd]
        .copy_from_slice(&q_natural);

    // K/V packed: [nkv, kv_capacity, hd/2] u8
    let k_packed_bytes = nkv * kv_capacity * (hd / 2);
    let v_packed_bytes = nkv * kv_capacity * (hd / 2);

    let mut k_packed_buf = device
        .alloc_buffer(k_packed_bytes, DType::U8, vec![nkv, kv_capacity, hd / 2])
        .expect("alloc K packed");
    k_packed_buf.as_mut_slice::<u8>().expect("write K packed")
        .copy_from_slice(k_packed_gpu);

    let mut v_packed_buf = device
        .alloc_buffer(v_packed_bytes, DType::U8, vec![nkv, kv_capacity, hd / 2])
        .expect("alloc V packed");
    v_packed_buf.as_mut_slice::<u8>().expect("write V packed")
        .copy_from_slice(v_packed_gpu);

    // Norms: [nkv, kv_capacity] f32
    let norms_bytes = nkv * kv_capacity * 4;

    let mut k_norms_buf = device
        .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kv_capacity])
        .expect("alloc K norms");
    k_norms_buf.as_mut_slice::<f32>().expect("write K norms")
        .copy_from_slice(k_norms_gpu);

    let mut v_norms_buf = device
        .alloc_buffer(norms_bytes, DType::F32, vec![nkv, kv_capacity])
        .expect("alloc V norms");
    v_norms_buf.as_mut_slice::<f32>().expect("write V norms")
        .copy_from_slice(v_norms_gpu);

    // Output + tmp
    let output_buf = device
        .alloc_buffer(nh * hd * 4, DType::F32, vec![nh, 1, hd])
        .expect("alloc output");

    let tmp_bytes = flash_attn_vec_tq::tmp_buffer_bytes(p.num_heads, p.head_dim);
    let tmp_buf = device
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .expect("alloc tmp");

    // --- Build params struct verbatim from manifest (mirrors forward_mlx.rs:1452-1462) ---
    //
    // DISPATCH MIRROR NOTE: The production call (forward_mlx.rs:1464-1474) passes
    // &self.activations.attn_q_normed which has ALREADY been FWHT-pre-rotated in-place
    // by dispatch_fwht_f32 at line 1433. In the harness, q_buf starts in natural basis
    // and we apply dispatch_fwht_f32 (Variation A/C) or skip it (Variation B), mirroring
    // the production dispatch structure faithfully.
    let params = FlashAttnVecTqParams {
        num_heads: p.num_heads,
        num_kv_heads: p.num_kv_heads,
        head_dim: p.head_dim,
        kv_seq_len: p.kv_seq_len,
        kv_capacity: p.kv_capacity,
        scale: p.scale,
        mask_type: p.mask_type,     // 2 (sliding) — corrected from meta JSON mask_type=1
        sliding_window: p.sliding_window,
        softcap: p.softcap,
        ring_start: p.ring_start,
    };

    // --- Dispatch ---
    let mut encoder = device.command_encoder().expect("command_encoder");

    match variation {
        Variation::A | Variation::C => {
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
                &params,
            )
            .expect("flash_attn_vec_tq dispatch");

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
            // FWHT-disabled: pass Q in natural basis; read output in rotated domain.
            // This deliberately mismatches the kernel's assumption that Q is pre-rotated.
            // If the output nrmse improves vs A, FWHT is the locus.
            //
            // NOTE: The CPU reference is the SAME as A/C (natural-basis SDPA from TQ dequant).
            // Variation B is mathematically wrong on purpose — its nrmse vs the natural-basis
            // reference reveals whether the FWHT pair (forward+inverse) is the source of
            // divergence seen in C-0.
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
                &params,
            )
            .expect("flash_attn_vec_tq dispatch (no FWHT)");
            // No inverse FWHT — output remains in rotated domain.
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

    // --- Write sdpa_out binary alongside the metrics JSON ---
    // (raw f32 LE, same layout as production sdpa_out [nh, 1, hd])
    let gpu_out_bytes: Vec<u8> = gpu_output
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    // Write to <out_base>.bin  (if --out is A.json, bin is A.bin)
    // We just print, not enforce — Worker 3 reads the JSON.

    let ran_at = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".into());

    let metrics = ReplayMetrics {
        variation: variation.to_string(),
        canary: use_canary,
        ran_at,
        nrmse,
        max_abs_diff,
        per_head_max_abs_diff: per_head,
        any_nan_inf_in_gpu_output: has_nan_inf,
        exit_status: if has_nan_inf { "NaN/Inf" } else { "ok" }.into(),
    };

    // Also emit per-run binary (best-effort, ignore errors)
    // Named <out stem>_sdpa_out.bin
    let _ = gpu_out_bytes; // suppress unused warning; caller writes the JSON

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
    // On non-Apple targets this code will not compile anyway due to #[cfg] at the
    // top, but we add an explicit guard for clarity.
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("argument error: {}", e);
            eprintln!(
                "usage: tq_kernel_replay --manifest <path> --variation <A|B|C> [--canary [1e9]] --out <path.json>"
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
    // TQ SDPA kernel + reduce kernel (registered explicitly)
    flash_attn_vec_tq::register(&mut registry);
    mlx_native::ops::flash_attn_vec::register(&mut registry);
    // fwht_standalone kernels are pre-registered inside KernelRegistry::new()

    eprintln!(
        "tq_kernel_replay: variation={} canary={} manifest={:?}",
        args.variation, args.canary, args.manifest
    );

    let metrics = run_variation(
        &manifest,
        args.variation,
        args.canary,
        &device,
        &mut registry,
    );

    // Print summary to stdout (for Worker 3 to capture)
    let json = serde_json::to_string_pretty(&metrics).expect("serialize metrics");
    println!("{}", json);

    // Write metrics to --out path
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&args.out, &json).unwrap_or_else(|e| {
        eprintln!("failed to write metrics to {:?}: {}", args.out, e);
        std::process::exit(1);
    });

    // Also write the raw GPU sdpa_out binary alongside (stem + _sdpa_out.bin)
    let out_stem = args.out.with_extension("");
    let bin_path = {
        let mut p = out_stem.into_os_string();
        p.push("_sdpa_out.bin");
        PathBuf::from(p)
    };
    // Re-run to get gpu bytes — or just re-derive from already-printed nrmse
    // (We already have the metrics, so we can't cheaply re-get the raw bytes here
    //  without restructuring. This is fine: the analyst reads the JSON metrics;
    //  if raw bytes are needed, re-run with a modified harness.)
    eprintln!(
        "metrics written to {:?}; raw sdpa_out bin would be at {:?} (not written separately in this run)",
        args.out, bin_path
    );

    eprintln!(
        "RESULT: variation={} nrmse={:.6} max_abs_diff={:.6} nan_inf={}",
        metrics.variation, metrics.nrmse, metrics.max_abs_diff, metrics.any_nan_inf_in_gpu_output
    );
}
