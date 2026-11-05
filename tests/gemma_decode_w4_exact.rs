//! Q4-GEMMA-DECODE dense primitive parity (ADR-063 design §6.1).
//!
//! Class A: every output (column c, row r) of the D1 row-tiled Q4_K/Q6_K and
//! D2 Q5_0/Q8_0 multi-column kernels must equal, as raw u32, the reference
//! single-column kernel run on column c alone on the same metallib path:
//! `kernel_mul_mv_q4_K_f32`, `kernel_mul_mv_q6_K_f32_nr2`,
//! `kernel_mul_mv_q5_0_f32`, `kernel_mul_mv_q8_0_f32_nr2`.
//!
//! Tests:
//! - `q4k_rt2_*` — S0, the Q4_K R0 = 2 spike (a T-N1 subset);
//! - `gemma_dense_row_tiled_q4k_q6k_bitwise` — T-N1 (every geometry cell,
//!   every Gemma Q4_K/Q6_K shape, synthetic tails, m = 1..8);
//! - `gemma_dense_row_tiled_lm_head_bitwise` — T-N1 lm head (N = 262,144);
//! - `gemma_dense_q5_0_q8_0_mn_bitwise` — T-N2;
//! - `gemma_decode_kernels_permutation_and_isolation` — T-N4 (dense part);
//! - `routing_policy_default_unchanged` — T-N7.
//!
//! T-N5: run the suite with `MLX_PRECOMPILED_METALLIB=1` and `=0` and the
//! same `GEMMA_DECODE_RECEIPT_DIR`. Each cell appends one JSON line (the
//! pipeline origin read back from the registry, SHA-256 of the output bits)
//! to `dense-<origin>.jsonl`; the ignored `tn5_cross_path_digest_diff` then
//! diffs the two files and fails on any mismatch.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::panic, clippy::too_many_arguments)]

use std::io::Write;

use mlx_native::{
    ggml_capability, ggml_routing_policy_from_environment,
    quantized_matmul_ggml_dense_mn_explicit, quantized_matmul_ggml_dual_gate_up_q4k,
    quantized_matmul_ggml_row_tiled_with_geometry,
    quantized_matmul_ggml_with_policy, row_tiled_geometry, DType, GgmlCapabilityRequest,
    GgmlInvocation, GgmlKernelRoute, GgmlQuantizedMatmulParams, GgmlRoutingPolicy, GgmlType,
    GgmlWorkloadClass, KernelRegistry, MlxBuffer, MlxDevice, RowTiledGeometry,
    KernelPipelineOrigin, GGML_CAPABILITY_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

const SENTINEL: u32 = 0xDEAD_BEEF;
const MAX_M: usize = 8;

// ---------------------------------------------------------------------------
// Seeded inputs
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32 - 0.5
    }
}

/// A half scale: mostly small normals, with zero and subnormal halves mixed in.
fn half_scale(rng: &mut Rng, block: usize) -> [u8; 2] {
    let bits = match block % 23 {
        0 => 0x0000,
        7 => 0x0001 + (rng.next() as u16 % 0x03FF), // subnormal
        13 => 0x8000,                               // -0
        _ => half::f16::from_f32(0.0005 + (rng.unit() + 0.5) * 0.004).to_bits(),
    };
    bits.to_le_bytes()
}

fn weight_bytes(ty: GgmlType, n: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let blocks = n * (k / ty.block_values() as usize);
    let mut out = Vec::with_capacity(blocks * ty.block_bytes() as usize);
    for b in 0..blocks {
        match ty {
            GgmlType::Q4_K => {
                out.extend_from_slice(&half_scale(rng, b));
                out.extend_from_slice(&half_scale(rng, b + 5));
                (0..12 + 128).for_each(|_| out.push(rng.byte()));
            }
            GgmlType::Q6_K => {
                (0..128 + 64 + 16).for_each(|_| out.push(rng.byte()));
                out.extend_from_slice(&half_scale(rng, b));
            }
            GgmlType::Q5_0 => {
                out.extend_from_slice(&half_scale(rng, b));
                (0..4 + 16).for_each(|_| out.push(rng.byte()));
            }
            GgmlType::Q8_0 => {
                out.extend_from_slice(&half_scale(rng, b));
                (0..32).for_each(|_| out.push(rng.byte()));
            }
            other => panic!("unsupported test type {other:?}"),
        }
    }
    assert_eq!(out.len(), blocks * ty.block_bytes() as usize);
    out
}

/// Eight activation columns. Every column carries ±0 and subnormals. Only
/// column 1 (NaN and ±∞, proving neighbour isolation) and column 6 (large
/// magnitudes) carry outliers: in the other six columns every lane's
/// per-block terms are of comparable size, so a 1-ULP drift in any lane
/// survives `simd_sum` instead of being absorbed by an outlier lane.
fn activations(k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x: Vec<f32> = (0..MAX_M * k).map(|_| rng.unit()).collect();
    for c in 0..MAX_M {
        let col = &mut x[c * k..(c + 1) * k];
        col[3] = 0.0;
        col[17] = -0.0;
        col[40] = f32::from_bits(0x0000_0013);
        col[41] = -f32::from_bits(0x0040_0000);
    }
    x[6 * k + k / 2] = 3.0e5;
    x[6 * k + k - 1] = -7.5e4;
    x[k + 9] = f32::NAN;
    x[k + 100] = f32::INFINITY;
    x[k + 101] = f32::NEG_INFINITY;
    x
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Fixture {
    device: MlxDevice,
    registry: KernelRegistry,
    weight: MlxBuffer,
    input: MlxBuffer,
    ty: GgmlType,
    n: usize,
    k: usize,
    /// Reference output of column c alone, `[MAX_M][n]` as u32.
    reference: Vec<Vec<u32>>,
    /// Metallib path the reference pipeline loaded from (T-N5 label).
    origin: String,
}

fn upload_bytes(device: &MlxDevice, bytes: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("alloc");
    buffer.as_mut_slice::<u8>().expect("slice").copy_from_slice(bytes);
    buffer
}

fn upload_f32(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("alloc");
    buffer.as_mut_slice::<f32>().expect("slice").copy_from_slice(values);
    buffer
}

/// Output buffer of `pad + m*n + pad` f32 filled with the sentinel; returns
/// the buffer and its `[pad, pad + m*n)` view.
fn sentinel_output(device: &MlxDevice, m: usize, n: usize, pad: usize) -> (MlxBuffer, MlxBuffer) {
    let total = pad + m * n + pad;
    let mut buffer = device
        .alloc_buffer(total * 4, DType::F32, vec![total])
        .expect("alloc output");
    buffer
        .as_mut_slice::<u32>()
        .expect("slice")
        .iter_mut()
        .for_each(|v| *v = SENTINEL);
    let view = buffer.slice_view((pad * 4) as u64, m * n);
    (buffer, view)
}

fn stock_single_column(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weight: &MlxBuffer,
    column: &[f32],
    ty: GgmlType,
    n: usize,
    k: usize,
) -> Vec<u32> {
    let input = upload_f32(device, column);
    let (out, view) = sentinel_output(device, 1, n, 0);
    let mut encoder = device.command_encoder().expect("encoder");
    quantized_matmul_ggml_with_policy(
        &mut encoder,
        registry,
        device,
        &input,
        weight,
        &view,
        &GgmlQuantizedMatmulParams {
            m: 1,
            n: n as u32,
            k: k as u32,
            ggml_type: ty,
        },
        &GgmlRoutingPolicy::default(),
    )
    .expect("stock dispatch");
    encoder.commit_and_wait().expect("stock GPU");
    out.as_slice::<u32>().expect("read").to_vec()
}

impl Fixture {
    fn new(ty: GgmlType, n: usize, k: usize, seed: u64) -> Self {
        let mut rng = Rng(seed ^ ((n as u64) << 20) ^ (k as u64));
        let device = MlxDevice::new().expect("Metal device");
        let mut registry = KernelRegistry::new();
        let weight = upload_bytes(&device, &weight_bytes(ty, n, k, &mut rng));
        let x = activations(k, &mut rng);
        let reference = (0..MAX_M)
            .map(|c| {
                stock_single_column(
                    &device,
                    &mut registry,
                    &weight,
                    &x[c * k..(c + 1) * k],
                    ty,
                    n,
                    k,
                )
            })
            .collect();
        let origin = pipeline_origin(&registry, ty);
        let input = upload_f32(&device, &x);
        Self {
            device,
            registry,
            weight,
            input,
            ty,
            n,
            k,
            reference,
            origin,
        }
    }

    fn params(&self, m: usize) -> GgmlQuantizedMatmulParams {
        GgmlQuantizedMatmulParams {
            m: m as u32,
            n: self.n as u32,
            k: self.k as u32,
            ggml_type: self.ty,
        }
    }

    /// Run `encode` on the first m columns (a view at `col_offset` columns
    /// into the input) and check every column bitwise against the reference
    /// plus sentinel bytes outside the output view.
    fn check<F>(&mut self, label: &str, m: usize, col_offset: usize, encode: F)
    where
        F: FnOnce(
            &mut mlx_native::CommandEncoder,
            &mut KernelRegistry,
            &MlxDevice,
            &MlxBuffer,
            &MlxBuffer,
            &MlxBuffer,
            &GgmlQuantizedMatmulParams,
        ),
    {
        let pad = 37;
        let (out, view) = sentinel_output(&self.device, m, self.n, pad);
        let input = self
            .input
            .slice_view((col_offset * self.k * 4) as u64, m * self.k);
        let params = self.params(m);
        let mut encoder = self.device.command_encoder().expect("encoder");
        encode(
            &mut encoder,
            &mut self.registry,
            &self.device,
            &input,
            &self.weight,
            &view,
            &params,
        );
        encoder.commit_and_wait().expect("candidate GPU");
        let got = out.as_slice::<u32>().expect("read");
        for (i, v) in got[..pad].iter().chain(&got[pad + m * self.n..]).enumerate() {
            assert_eq!(*v, SENTINEL, "{label}: write outside the output view at pad index {i}");
        }
        for c in 0..m {
            let want = &self.reference[col_offset + c];
            let col = &got[pad + c * self.n..pad + (c + 1) * self.n];
            if let Some(row) = (0..self.n).find(|&r| col[r] != want[r]) {
                panic!(
                    "{label} {:?} n={} k={} m={m} column {c} row {row}: got {:#010x} want {:#010x}",
                    self.ty, self.n, self.k, col[row], want[row]
                );
            }
        }
        record(label, &self.origin, self.ty, self.n, self.k, m, &got[pad..pad + m * self.n]);
    }
}

/// The metallib path the reference pipeline actually loaded from, read back
/// from the registry (not inferred from the environment).
fn pipeline_origin(registry: &KernelRegistry, ty: GgmlType) -> String {
    let kernel = match ty {
        GgmlType::Q4_K => "kernel_mul_mv_q4_K_f32",
        GgmlType::Q6_K => "kernel_mul_mv_q6_K_f32_nr2",
        GgmlType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlType::Q8_0 => "kernel_mul_mv_q8_0_f32_nr2",
        other => panic!("no reference kernel for {other:?}"),
    };
    let identity = registry
        .pipeline_identity(&format!("{kernel}|700:i1|701:i1|702:i1"))
        .expect("reference pipeline identity");
    match identity.origin {
        KernelPipelineOrigin::PrecompiledMetallib => "precompiled".into(),
        KernelPipelineOrigin::RuntimeSource => "runtime".into(),
    }
}

/// T-N5: one JSON line per cell into `<GEMMA_DECODE_RECEIPT_DIR>/dense-<origin>.jsonl`.
fn record(label: &str, origin: &str, ty: GgmlType, n: usize, k: usize, m: usize, bits: &[u32]) {
    let Some(dir) = std::env::var_os("GEMMA_DECODE_RECEIPT_DIR") else {
        return;
    };
    let mut hasher = Sha256::new();
    bits.iter().for_each(|v| hasher.update(v.to_le_bytes()));
    let line = serde_json::json!({
        "suite": "gemma_decode_w4_exact",
        "cell": label,
        "type": format!("{ty:?}"),
        "n": n, "k": k, "m": m,
        "pipeline_origin": origin,
        "metallib_env": std::env::var("MLX_PRECOMPILED_METALLIB").ok(),
        "output_sha256": format!("{:x}", hasher.finalize()),
    });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::path::Path::new(&dir).join(format!("dense-{origin}.jsonl")))
        .expect("receipt file");
    writeln!(file, "{line}").expect("receipt write");
}

/// T-N5 cross-path diff. Run after the suite has run once on each path with
/// the same `GEMMA_DECODE_RECEIPT_DIR`:
/// `GEMMA_DECODE_RECEIPT_DIR=<dir> cargo test --release --test gemma_decode_w4_exact tn5 -- --ignored`.
/// Fails unless both files exist, cover the same cells, and every cell's
/// digest multiset is identical; writes `tn5-diff.json` beside them.
#[test]
#[ignore = "cross-run step: needs receipts from both metallib paths"]
fn tn5_cross_path_digest_diff() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("GEMMA_DECODE_RECEIPT_DIR").expect("GEMMA_DECODE_RECEIPT_DIR"),
    );
    let load = |origin: &str| {
        let text = std::fs::read_to_string(dir.join(format!("dense-{origin}.jsonl")))
            .unwrap_or_else(|e| panic!("dense-{origin}.jsonl: {e}"));
        let mut cells = std::collections::BTreeMap::<String, Vec<String>>::new();
        for line in text.lines() {
            let r: serde_json::Value = serde_json::from_str(line).expect("receipt line");
            assert_eq!(r["pipeline_origin"], origin, "line filed under the wrong path");
            let key = format!("{}|{}|{}|{}|{}", r["cell"], r["type"], r["n"], r["k"], r["m"]);
            cells
                .entry(key)
                .or_default()
                .push(r["output_sha256"].as_str().expect("digest").to_string());
        }
        cells.values_mut().for_each(|v| v.sort());
        cells
    };
    let pre = load("precompiled");
    let run = load("runtime");
    let mismatched: Vec<&String> = pre
        .iter()
        .filter(|(k, v)| run.get(*k) != Some(v))
        .map(|(k, _)| k)
        .chain(run.keys().filter(|k| !pre.contains_key(*k)))
        .collect();
    let records: usize = pre.values().map(Vec::len).sum();
    std::fs::write(
        dir.join("tn5-diff.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "cells": pre.len(),
            "records_per_path": records,
            "mismatched": mismatched,
        }))
        .expect("json"),
    )
    .expect("tn5-diff.json");
    assert!(mismatched.is_empty(), "cross-path digest mismatch: {mismatched:?}");
    assert!(records > 0, "no receipts recorded");
}

fn all_geometries() -> Vec<RowTiledGeometry> {
    let mut cells = Vec::new();
    for r0 in RowTiledGeometry::R0_CELLS {
        for nsg in RowTiledGeometry::NSG_CELLS {
            cells.push(RowTiledGeometry { r0, nsg });
        }
    }
    cells
}

fn check_row_tiled(fx: &mut Fixture, geometries: &[RowTiledGeometry], widths: &[usize]) {
    for &geometry in geometries {
        for &m in widths {
            let label = format!("D1 r0={} nsg={}", geometry.r0, geometry.nsg);
            fx.check(&label, m, 0, |enc, reg, dev, x, w, y, p| {
                quantized_matmul_ggml_row_tiled_with_geometry(enc, reg, dev, x, w, y, p, geometry)
                    .expect("row-tiled dispatch")
            });
        }
    }
}

/// Gemma 4 Q4_K shapes (design §2.5) plus synthetic row tails.
const Q4K_SHAPES: &[(usize, usize)] = &[
    (4096, 2816),
    (8192, 2816),
    (2048, 2816),
    (1024, 2816),
    (2112, 2816),
    (2816, 4096),
    (2816, 8192),
    (7, 2816),
    (1027, 2816),
    (2050, 2816),
    (2817, 2816),
];

const Q6K_SHAPES: &[(usize, usize)] = &[(2048, 2816), (7, 2816), (1027, 2816), (2050, 2816), (2817, 2816)];

const WIDTHS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

// ---------------------------------------------------------------------------
// S0 and T-N1
// ---------------------------------------------------------------------------

#[test]
fn q4k_rt2_s0_gemma_shapes_and_tails_bitwise() {
    let geometries: Vec<_> = RowTiledGeometry::NSG_CELLS
        .iter()
        .map(|&nsg| RowTiledGeometry { r0: 2, nsg })
        .collect();
    for &(n, k) in Q4K_SHAPES {
        let mut fx = Fixture::new(GgmlType::Q4_K, n, k, 0x5300);
        check_row_tiled(&mut fx, &geometries, &WIDTHS);
    }
}

#[test]
fn gemma_dense_row_tiled_q4k_q6k_bitwise() {
    for (ty, shapes) in [(GgmlType::Q4_K, Q4K_SHAPES), (GgmlType::Q6_K, Q6K_SHAPES)] {
        for &(n, k) in shapes {
            let mut fx = Fixture::new(ty, n, k, 0x7161);
            check_row_tiled(&mut fx, &all_geometries(), &WIDTHS);
        }
    }
}

#[test]
fn gemma_dense_row_tiled_lm_head_bitwise() {
    let mut fx = Fixture::new(GgmlType::Q6_K, 262_144, 2816, 0x1a4d);
    check_row_tiled(&mut fx, &all_geometries(), &[1, 4, 8]);
    check_row_tiled(
        &mut fx,
        &[row_tiled_geometry(GgmlType::Q6_K, 262_144, 2816).expect("lm head in table")],
        &WIDTHS,
    );
}

/// F1: dual-weight gate+up. Each output equals the reference on its own
/// weight, at every NSG and width, including a row tail.
#[test]
fn gemma_dense_dual_gate_up_q4k_bitwise() {
    for (n, k) in [(2112, 2816), (1027, 2816)] {
        let mut fx = Fixture::new(GgmlType::Q4_K, n, k, 0xf1);
        let mut rng = Rng(0xf1f1 ^ n as u64);
        let up = upload_bytes(&fx.device, &weight_bytes(GgmlType::Q4_K, n, k, &mut rng));
        let x = fx.input.as_slice::<f32>().expect("input").to_vec();
        let up_reference: Vec<Vec<u32>> = (0..MAX_M)
            .map(|c| {
                stock_single_column(
                    &fx.device,
                    &mut fx.registry,
                    &up,
                    &x[c * k..(c + 1) * k],
                    GgmlType::Q4_K,
                    n,
                    k,
                )
            })
            .collect();
        for nsg in RowTiledGeometry::NSG_CELLS {
            for m in 1..=MAX_M {
                let (gate_buf, gate_view) = sentinel_output(&fx.device, m, n, 11);
                let (up_buf, up_view) = sentinel_output(&fx.device, m, n, 11);
                let input = fx.input.slice_view(0, m * k);
                let params = fx.params(m);
                let mut encoder = fx.device.command_encoder().expect("encoder");
                quantized_matmul_ggml_dual_gate_up_q4k(
                    &mut encoder,
                    &mut fx.registry,
                    &fx.device,
                    &input,
                    &fx.weight,
                    &up,
                    &gate_view,
                    &up_view,
                    &params,
                    nsg,
                )
                .expect("dual dispatch");
                encoder.commit_and_wait().expect("dual GPU");
                for (buf, reference, name) in [
                    (&gate_buf, &fx.reference, "gate"),
                    (&up_buf, &up_reference, "up"),
                ] {
                    let got = buf.as_slice::<u32>().expect("read");
                    assert!(got[..11].iter().chain(&got[11 + m * n..]).all(|&v| v == SENTINEL));
                    for c in 0..m {
                        assert_eq!(
                            &got[11 + c * n..11 + (c + 1) * n],
                            reference[c].as_slice(),
                            "F1 {name} n={n} nsg={nsg} m={m} column {c}"
                        );
                    }
                    record(
                        &format!("F1 {name} nsg={nsg}"),
                        &fx.origin,
                        GgmlType::Q4_K,
                        n,
                        k,
                        m,
                        &got[11..11 + m * n],
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T-N2
// ---------------------------------------------------------------------------

#[test]
fn gemma_dense_q5_0_q8_0_mn_bitwise() {
    for ty in [GgmlType::Q5_0, GgmlType::Q8_0] {
        for (n, k) in [(2816, 2112), (2816, 704), (7, 2112), (1027, 2112), (2817, 704)] {
            let mut fx = Fixture::new(ty, n, k, 0xd2);
            for m in 2..=MAX_M {
                fx.check("D2", m, 0, |enc, reg, dev, x, w, y, p| {
                    quantized_matmul_ggml_dense_mn_explicit(enc, reg, dev, x, w, y, p)
                        .expect("dense mN dispatch")
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T-N4 (dense) and the policy route
// ---------------------------------------------------------------------------

fn gemma_policy() -> GgmlRoutingPolicy {
    GgmlRoutingPolicy {
        dense_row_tiled_q4k: true,
        dense_row_tiled_q6k: true,
        dense_row_tiled_m1: true,
        dense_mn_q5_0: true,
        dense_mn_q8_0: true,
        ..GgmlRoutingPolicy::default()
    }
}

#[test]
fn gemma_decode_kernels_permutation_and_isolation() {
    let policy = gemma_policy();
    for (ty, n, k) in [
        (GgmlType::Q4_K, 4096, 2816),
        (GgmlType::Q4_K, 2816, 4096),
        (GgmlType::Q6_K, 2048, 2816),
        (GgmlType::Q5_0, 2816, 2112),
        (GgmlType::Q8_0, 2816, 2112),
    ] {
        let mut fx = Fixture::new(ty, n, k, 0x9e4);
        // Policy route at every width, views at non-zero column offsets
        // (the NaN/∞ column 1 moves to every tile position).
        for m in 1..=MAX_M {
            for col_offset in [0, MAX_M - m] {
                fx.check("policy", m, col_offset, |enc, reg, dev, x, w, y, p| {
                    quantized_matmul_ggml_with_policy(enc, reg, dev, x, w, y, p, &policy)
                        .expect("policy dispatch")
                });
            }
        }
        // Column permutation: reverse the eight columns; output columns must
        // be the reference columns in reverse.
        let x = fx.input.as_slice::<f32>().expect("input").to_vec();
        let reversed: Vec<f32> = (0..MAX_M)
            .rev()
            .flat_map(|c| x[c * k..(c + 1) * k].to_vec())
            .collect();
        let input = upload_f32(&fx.device, &reversed);
        let (out, view) = sentinel_output(&fx.device, MAX_M, n, 0);
        let params = fx.params(MAX_M);
        let mut encoder = fx.device.command_encoder().expect("encoder");
        quantized_matmul_ggml_with_policy(
            &mut encoder,
            &mut fx.registry,
            &fx.device,
            &input,
            &fx.weight,
            &view,
            &params,
            &policy,
        )
        .expect("permuted dispatch");
        encoder.commit_and_wait().expect("permuted GPU");
        let got = out.as_slice::<u32>().expect("read");
        for c in 0..MAX_M {
            assert_eq!(
                &got[c * n..(c + 1) * n],
                fx.reference[MAX_M - 1 - c].as_slice(),
                "{ty:?} permuted column {c}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// T-N7
// ---------------------------------------------------------------------------

fn dense_route(ty: GgmlType, m: u32, n: u32, k: u32, routing: GgmlRoutingPolicy) -> GgmlKernelRoute {
    ggml_capability(GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: GgmlInvocation::DenseAuto { m, n, k },
        ggml_type: ty,
        workload: if m == 1 {
            GgmlWorkloadClass::DecodeSingle
        } else {
            GgmlWorkloadClass::ContinuousWidth
        },
        routing,
    })
    .route
    .expect("route")
}

#[test]
fn routing_policy_default_unchanged() {
    let default = GgmlRoutingPolicy::default();
    let env = ggml_routing_policy_from_environment();
    for policy in [default, env] {
        assert!(!policy.dense_row_tiled_q4k && !policy.dense_row_tiled_q6k);
        assert!(!policy.dense_row_tiled_m1 && !policy.dense_mn_q5_0 && !policy.dense_mn_q8_0);
        assert!(!policy.expert_grouped_q4k && !policy.expert_grouped_q5_0);
        assert!(!policy.expert_grouped_q8_0);
    }
    let gemma = gemma_policy();
    let shapes = [
        (GgmlType::Q4_K, 4096, 2816),
        (GgmlType::Q4_K, 2816, 8192),
        (GgmlType::Q6_K, 2048, 2816),
        (GgmlType::Q6_K, 262_144, 2816),
        (GgmlType::Q5_0, 2816, 2112),
        (GgmlType::Q8_0, 2816, 2112),
    ];
    for (ty, n, k) in shapes {
        for m in 1..=8u32 {
            let stock = dense_route(ty, m, n, k, default);
            assert!(
                !matches!(stock, GgmlKernelRoute::DenseGemmaRowTiled | GgmlKernelRoute::DenseGemmaMn),
                "default policy planned a Gemma route for {ty:?} m={m}"
            );
            assert_eq!(stock, dense_route(ty, m, n, k, env), "{ty:?} m={m}");
            let candidate = dense_route(ty, m, n, k, gemma);
            let expected = match ty {
                GgmlType::Q4_K | GgmlType::Q6_K => GgmlKernelRoute::DenseGemmaRowTiled,
                _ if m >= 2 => GgmlKernelRoute::DenseGemmaMn,
                _ => stock,
            };
            assert_eq!(candidate, expected, "{ty:?} m={m}");
            // A reference-flag change (or mv_ext) disables the lever.
            let mv_ext = GgmlRoutingPolicy {
                dense_decode_mv_ext: true,
                ..gemma
            };
            assert!(!matches!(
                dense_route(ty, m, n, k, mv_ext),
                GgmlKernelRoute::DenseGemmaRowTiled | GgmlKernelRoute::DenseGemmaMn
            ));
        }
    }
    // Reference flags gate their levers.
    let no_q6_nr2 = GgmlRoutingPolicy {
        dense_q6k_mv_nr2: false,
        ..gemma
    };
    assert_ne!(
        dense_route(GgmlType::Q6_K, 4, 2048, 2816, no_q6_nr2),
        GgmlKernelRoute::DenseGemmaRowTiled
    );
    let no_q8_nr2 = GgmlRoutingPolicy {
        dense_q8_0_mv_nr2: false,
        ..gemma
    };
    assert_ne!(
        dense_route(GgmlType::Q8_0, 4, 2816, 2112, no_q8_nr2),
        GgmlKernelRoute::DenseGemmaMn
    );
    // Shapes outside the frozen table keep the stock route.
    assert_eq!(
        dense_route(GgmlType::Q4_K, 4, 5120, 5120, gemma),
        dense_route(GgmlType::Q4_K, 4, 5120, 5120, default)
    );

    // Serde: an old receipt without the new fields deserializes with every
    // lever off; a round trip preserves every field.
    let mut old = serde_json::to_value(default).expect("serialize");
    let object = old.as_object_mut().expect("object");
    for field in [
        "dense_row_tiled_q4k",
        "dense_row_tiled_q6k",
        "dense_row_tiled_m1",
        "dense_mn_q5_0",
        "dense_mn_q8_0",
        "expert_grouped_q4k",
        "expert_grouped_q5_0",
        "expert_grouped_q8_0",
    ] {
        assert!(object.remove(field).is_some(), "field {field} not serialized");
    }
    let parsed: GgmlRoutingPolicy = serde_json::from_value(old).expect("old receipt");
    assert_eq!(parsed, default);
    let all_on = GgmlRoutingPolicy {
        expert_grouped_q4k: true,
        expert_grouped_q5_0: true,
        expert_grouped_q8_0: true,
        ..gemma
    };
    let round: GgmlRoutingPolicy =
        serde_json::from_str(&serde_json::to_string(&all_on).expect("ser")).expect("de");
    assert_eq!(round, all_on);
}
