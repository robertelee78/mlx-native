//! ADR-007 Path C F-0.3: empirical KV distribution analyzer.
//!
//! Reads pre_quant K/V F32 dumps produced by hf2q's
//! `HF2Q_DUMP_PRE_QUANT=1` infrastructure (extended in iter-4 to support
//! `HF2Q_DUMP_PRE_QUANT_LAYERS` + `HF2Q_DUMP_PRE_QUANT_POSITIONS`).
//!
//! For each (layer, kv_head, position):
//!  1. Apply the production encoder pipeline (D1 SRHT + FWHT + norm
//!     extraction + scale-to-N(0,1)) using `turboquant_hb_encode_d256`'s
//!     intermediate steps.
//!  2. Capture the post-scale, pre-quantize values (the actual codebook
//!     lookup inputs).
//!  3. Aggregate per-(layer, kv_head, coord) stats: mean, std, p1, p99,
//!     min, max, outlier rate (|x| > 5.07 saturates 8-bit codebook range).
//!  4. Compute per-(layer, kv_head) overall stats vs N(0,1) reference.
//!
//! Output: JSON report with per-(layer, kv_head) statistics.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --example tq_distribution_analyze -- \
//!   --dump-dir /tmp/f03-dumps/pre_quant \
//!   --output docs/adr007-pathC/F-0/empirical_kv_distribution.json
//! ```
//!
//! ## Dump file format (from forward_mlx.rs)
//!
//! - `L{layer:02}_p{pos:04}_k_pre_quant.f32.bin` — raw F32 K, [nkv, hd]
//! - `L{layer:02}_p{pos:04}_v_pre_quant.f32.bin` — raw F32 V, [nkv, hd]
//! - `L{layer:02}_p{pos:04}_meta.json` — sidecar with shape, position, etc.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use mlx_native::turboquant::{
    apply_d1_sign_mask_inplace, fwht_inplace, TBQ_SIGNS_256,
};

use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct DumpFile {
    layer: usize,
    pos: usize,
    k_path: PathBuf,
    v_path: PathBuf,
    meta_path: PathBuf,
}

fn discover_dumps(dump_dir: &Path) -> Vec<DumpFile> {
    let mut files: BTreeMap<(usize, usize), DumpFile> = BTreeMap::new();
    if !dump_dir.is_dir() {
        eprintln!("dump_dir {} does not exist or is not a directory", dump_dir.display());
        return Vec::new();
    }
    for entry in fs::read_dir(dump_dir).expect("read_dir") {
        let entry = entry.expect("entry");
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Pattern: L{LL}_p{PPPP}_{k|v|meta}_pre_quant.f32.bin or _meta.json
        if !name.starts_with('L') { continue; }
        // Parse "L{LL}_p{PPPP}_..." prefix.
        let rest = &name[1..]; // drop 'L'
        let (l_str, rest) = match rest.find('_') {
            Some(i) => (&rest[..i], &rest[i+1..]),
            None => continue,
        };
        let layer: usize = match l_str.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let rest = match rest.strip_prefix('p') { Some(r) => r, None => continue };
        let (p_str, suffix) = match rest.find('_') {
            Some(i) => (&rest[..i], &rest[i+1..]),
            None => continue,
        };
        let pos: usize = match p_str.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };

        let path = entry.path();
        let entry = files.entry((layer, pos)).or_insert(DumpFile {
            layer, pos,
            k_path: PathBuf::new(),
            v_path: PathBuf::new(),
            meta_path: PathBuf::new(),
        });

        if suffix.starts_with("k_pre_quant") {
            entry.k_path = path;
        } else if suffix.starts_with("v_pre_quant") {
            entry.v_path = path;
        } else if suffix.starts_with("meta") {
            entry.meta_path = path;
        }
    }
    files.into_values().filter(|d| {
        d.k_path.exists() && d.v_path.exists() && d.meta_path.exists()
    }).collect()
}

fn read_f32_bin(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).expect("read bin");
    assert!(bytes.len() % 4 == 0, "non-multiple-of-4 file");
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &bytes[i*4..i*4+4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    out
}

fn read_meta(path: &Path) -> Value {
    let s = fs::read_to_string(path).expect("read meta");
    serde_json::from_str(&s).expect("parse meta")
}

/// Apply the production HB encoder pipeline up to "post-scale, pre-quantize".
/// Mirrors `turboquant_hb_encode_d256` steps 1-4 but stops before the
/// nearest-centroid lookup. Returns the post-scale values (length 256).
fn encode_to_post_scale_d256(x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), 256);
    let mut elems = x.to_vec();
    apply_d1_sign_mask_inplace(&mut elems, &TBQ_SIGNS_256);
    fwht_inplace(&mut elems).expect("fwht");
    let norm_sq: f32 = elems.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt();
    let scale: f32 = if norm > 1.0e-10_f32 {
        (1.0_f32 / norm) * (256.0_f32).sqrt()
    } else {
        0.0_f32
    };
    for v in elems.iter_mut() { *v *= scale; }
    elems
}

#[derive(Debug, Default, Clone)]
struct Stats {
    n: usize,
    sum: f64,
    sum_sq: f64,
    min: f32,
    max: f32,
    outliers_8bit: usize,    // |x| > 5.0652659 (8-bit codebook range)
    outliers_5bit: usize,    // |x| > 3.2606790 (5-bit range)
    samples: Vec<f32>,        // for percentile estimation
}

impl Stats {
    fn new() -> Self {
        Self {
            n: 0, sum: 0.0, sum_sq: 0.0,
            min: f32::INFINITY, max: f32::NEG_INFINITY,
            outliers_8bit: 0, outliers_5bit: 0,
            samples: Vec::new(),
        }
    }
    fn push(&mut self, v: f32) {
        self.n += 1;
        self.sum += v as f64;
        self.sum_sq += (v as f64) * (v as f64);
        if v < self.min { self.min = v; }
        if v > self.max { self.max = v; }
        if v.abs() > 5.0652659 { self.outliers_8bit += 1; }
        if v.abs() > 3.2606790 { self.outliers_5bit += 1; }
        self.samples.push(v);
    }
    fn mean(&self) -> f64 { if self.n == 0 { 0.0 } else { self.sum / self.n as f64 } }
    fn var(&self) -> f64 {
        if self.n < 2 { return 0.0; }
        let m = self.mean();
        (self.sum_sq / self.n as f64) - m * m
    }
    fn std(&self) -> f64 { self.var().max(0.0).sqrt() }
    fn pct(&mut self, p: f32) -> f32 {
        if self.samples.is_empty() { return 0.0; }
        if !self.samples.is_sorted() {
            self.samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
        let idx = (((self.samples.len() as f32 - 1.0) * p).round() as usize)
            .min(self.samples.len() - 1);
        self.samples[idx]
    }
    fn outlier_rate_8bit(&self) -> f64 { if self.n == 0 { 0.0 } else { self.outliers_8bit as f64 / self.n as f64 } }
    fn outlier_rate_5bit(&self) -> f64 { if self.n == 0 { 0.0 } else { self.outliers_5bit as f64 / self.n as f64 } }
}

fn main() {
    let mut dump_dir = PathBuf::from("/tmp/f03-dumps/pre_quant");
    let mut output = PathBuf::from("/tmp/f03-distribution-report.json");
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dump-dir" => { dump_dir = PathBuf::from(&args[i+1]); i += 2; }
            "--output"   => { output = PathBuf::from(&args[i+1]); i += 2; }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(1); }
        }
    }

    let files = discover_dumps(&dump_dir);
    eprintln!("F-0.3 analyzer: discovered {} dump triples in {}", files.len(), dump_dir.display());
    if files.is_empty() {
        eprintln!("no dumps found; exiting");
        std::process::exit(1);
    }

    // Per-(layer, tensor=K|V, kv_head, coord_bin) stats — for D=256 only;
    // we drop D=512 (architectural variant) with a warning.
    // Bucket coords into 8 groups of 32 to bound sample count.
    let mut per_layer_k: BTreeMap<usize, Stats> = BTreeMap::new();
    let mut per_layer_v: BTreeMap<usize, Stats> = BTreeMap::new();
    // Per-coord aggregate (across all positions, kv_heads, layers) for first/last layer comparison.
    let mut layer_kv_head_stats: BTreeMap<(usize, char, usize), Stats> = BTreeMap::new();

    let mut skipped_d512 = 0usize;

    for f in &files {
        let meta = read_meta(&f.meta_path);
        let nkv = meta["nkv"].as_u64().expect("nkv") as usize;
        let hd = meta["hd"].as_u64().expect("hd") as usize;
        if hd != 256 {
            skipped_d512 += 1;
            continue;
        }
        let k_raw = read_f32_bin(&f.k_path);
        let v_raw = read_f32_bin(&f.v_path);
        assert_eq!(k_raw.len(), nkv * hd);
        assert_eq!(v_raw.len(), nkv * hd);

        for kv_h in 0..nkv {
            let k_row: &[f32] = &k_raw[kv_h*hd..(kv_h+1)*hd];
            let v_row: &[f32] = &v_raw[kv_h*hd..(kv_h+1)*hd];
            let k_post = encode_to_post_scale_d256(k_row);
            let v_post = encode_to_post_scale_d256(v_row);

            let k_layer = per_layer_k.entry(f.layer).or_insert_with(Stats::new);
            for &x in &k_post { k_layer.push(x); }
            let v_layer = per_layer_v.entry(f.layer).or_insert_with(Stats::new);
            for &x in &v_post { v_layer.push(x); }

            let key_k = (f.layer, 'K', kv_h);
            let lkh_k = layer_kv_head_stats.entry(key_k).or_insert_with(Stats::new);
            for &x in &k_post { lkh_k.push(x); }
            let key_v = (f.layer, 'V', kv_h);
            let lkh_v = layer_kv_head_stats.entry(key_v).or_insert_with(Stats::new);
            for &x in &v_post { lkh_v.push(x); }
        }
    }

    if skipped_d512 > 0 {
        eprintln!("skipped {} dumps with hd != 256 (architectural variants)", skipped_d512);
    }

    // Build report.
    let mut report = serde_json::Map::new();
    report.insert("dump_dir".into(), Value::String(dump_dir.display().to_string()));
    report.insert("dump_count".into(), Value::Number((files.len() as i64).into()));
    report.insert("d512_skipped".into(), Value::Number((skipped_d512 as i64).into()));
    report.insert("codebook_range_8bit".into(), Value::Number(serde_json::Number::from_f64(5.0652659).unwrap()));
    report.insert("codebook_range_5bit".into(), Value::Number(serde_json::Number::from_f64(3.2606790).unwrap()));

    let mut per_layer_arr: Vec<Value> = Vec::new();
    let mut all_layers: Vec<usize> = per_layer_k.keys().cloned().collect();
    all_layers.sort();
    for layer in &all_layers {
        let mut k_stats = per_layer_k.remove(layer).unwrap();
        let mut v_stats = per_layer_v.remove(layer).unwrap();
        let row = serde_json::json!({
            "layer": layer,
            "K": {
                "n_samples": k_stats.n,
                "mean": k_stats.mean(),
                "std": k_stats.std(),
                "min": k_stats.min,
                "max": k_stats.max,
                "p1": k_stats.pct(0.01),
                "p50": k_stats.pct(0.50),
                "p99": k_stats.pct(0.99),
                "outlier_rate_8bit": k_stats.outlier_rate_8bit(),
                "outlier_rate_5bit": k_stats.outlier_rate_5bit(),
            },
            "V": {
                "n_samples": v_stats.n,
                "mean": v_stats.mean(),
                "std": v_stats.std(),
                "min": v_stats.min,
                "max": v_stats.max,
                "p1": v_stats.pct(0.01),
                "p50": v_stats.pct(0.50),
                "p99": v_stats.pct(0.99),
                "outlier_rate_8bit": v_stats.outlier_rate_8bit(),
                "outlier_rate_5bit": v_stats.outlier_rate_5bit(),
            },
        });
        per_layer_arr.push(row);
    }
    report.insert("per_layer".into(), Value::Array(per_layer_arr));

    // Per (layer, K|V, kv_head) stats — coarser-grained but useful for
    // detecting per-head outliers.
    let mut per_layer_head_arr: Vec<Value> = Vec::new();
    let keys: Vec<_> = layer_kv_head_stats.keys().cloned().collect();
    for key in &keys {
        let mut s = layer_kv_head_stats.remove(key).unwrap();
        let row = serde_json::json!({
            "layer": key.0,
            "tensor": key.1.to_string(),
            "kv_head": key.2,
            "n_samples": s.n,
            "mean": s.mean(),
            "std": s.std(),
            "p1": s.pct(0.01),
            "p99": s.pct(0.99),
            "outlier_rate_8bit": s.outlier_rate_8bit(),
            "outlier_rate_5bit": s.outlier_rate_5bit(),
        });
        per_layer_head_arr.push(row);
    }
    report.insert("per_layer_head".into(), Value::Array(per_layer_head_arr));

    // Verdict: how does the empirical distribution compare to N(0,1)?
    // For each layer's K stream, expected: mean ~ 0, std ~ 1, |x| > 5.07
    // outlier rate ~ 5.7e-7 (one-sided integral of N(0,1) tail at 5.07).
    let n01_outlier_rate_8bit = 2.0 * 0.5 * (1.0 - libm_erf(5.0652659 / std::f32::consts::SQRT_2));
    report.insert("n01_reference".into(), serde_json::json!({
        "outlier_rate_8bit_expected": n01_outlier_rate_8bit,
        "note": "If empirical outlier rate is much higher than this, the N(0,1) Lloyd-Max codebook is suboptimal — F-2 calibration could help.",
    }));

    let json = serde_json::to_string_pretty(&report).expect("serialize");
    fs::write(&output, &json).expect("write output");
    eprintln!("F-0.3 report written to {}", output.display());

    // Stdout summary.
    println!("=== F-0.3 Empirical KV Distribution Summary ===");
    println!("Dumps:              {}", files.len());
    println!("D=512 skipped:      {}", skipped_d512);
    println!("8-bit N(0,1) outlier rate (theoretical): {:.3e}", n01_outlier_rate_8bit);
}

/// Approx erf for f32 (Abramowitz & Stegun 7.1.26).
fn libm_erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.3275911_f32;
    let a1 = 0.254829592_f32;
    let a2 = -0.284496736_f32;
    let a3 = 1.421413741_f32;
    let a4 = -1.453152027_f32;
    let a5 = 1.061405429_f32;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
