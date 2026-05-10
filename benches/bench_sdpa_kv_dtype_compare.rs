//! ADR-028 iter-257 — TQ-HB SDPA vs F16/F32 SDPA at gemma4 sliding decode shape.
//!
//! Hypothesis under test (from iter-191/250):
//!   "TQ-HB SDPA is structurally ≥2× SLOWER than F16 SDPA at the same
//!    decode shape — the byte-packed quant + per-pos F32 norms +
//!    Lloyd-Max codebook lookup cannot be done as fast as raw F16 reads."
//!
//! Falsification plan:
//!   - Bench all three (F32, F16, TQ-HB) at SAME gemma4 sliding decode
//!     shape: head_dim=256, kv_seq=1024, n_heads=16, n_kv_heads=8.
//!   - Measure: time per dispatch (BATCH=200 in 1 CB).
//!   - Compute: TQ-HB / F16 ratio.
//!
//! Verdict thresholds:
//!   - ratio > 2.0×: H confirmed (structural cost real, no kernel lever)
//!   - ratio in (1.0×, 2.0×): partial cost, possible kernel lever
//!   - ratio < 1.0×: TQ-HB FASTER (bandwidth-bound; 4× less bytes read)
//!
//! Bytes read per SDPA call (production layout, gemma4 sliding):
//!   F32 K+V: 2 × (8 × 1024 × 256 × 4) = 16.0 MB
//!   F16 K+V: 2 × (8 × 1024 × 256 × 2) =  8.0 MB
//!   TQ-HB:   2 × (1.0 MB + 32 KB)     =  2.06 MB (4× less than F16)
//!
//! If TQ-HB is bandwidth-bound, ratio TQ-HB/F16 ≈ 0.25× (4× faster).
//! If TQ-HB is compute-bound (dequant arithmetic), ratio could be > 1×.
//!
//! Run:
//!   cargo bench -p mlx-native --bench bench_sdpa_kv_dtype_compare

use mlx_native::ops::flash_attn_vec::{flash_attn_vec, FlashAttnVecParams};
use mlx_native::ops::flash_attn_vec_tq_hb::{
    self, flash_attn_vec_tq_hb, FlashAttnVecTqHbParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

// gemma4 sliding-layer decode shape (from forward_mlx.rs production).
const NUM_HEADS: u32 = 16;
const NUM_KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 256;
const KV_SEQ_LEN: u32 = 1024;
const KV_CAPACITY: u32 = 1024;
const SLIDING_WINDOW: u32 = 1024;
const NWG: u32 = 32;

const WARMUP: usize = 10;
const MEASURE: usize = 30;
const BATCH: usize = 50;

fn alloc_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("alloc f32")
}

fn alloc_f16(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n * 2, DType::F16, vec![n]).expect("alloc f16")
}

fn alloc_u8(device: &MlxDevice, n: usize) -> MlxBuffer {
    device.alloc_buffer(n, DType::U8, vec![n]).expect("alloc u8")
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64) * p).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bench_sdpa_f32(device: &MlxDevice, registry: &mut KernelRegistry) -> (f64, u64) {
    mlx_native::ops::flash_attn_vec::register(registry);

    let q = alloc_f32(device, (NUM_HEADS * HEAD_DIM) as usize);
    let k = alloc_f32(device, (NUM_KV_HEADS * KV_CAPACITY * HEAD_DIM) as usize);
    let v = alloc_f32(device, (NUM_KV_HEADS * KV_CAPACITY * HEAD_DIM) as usize);
    let out = alloc_f32(device, (NUM_HEADS * HEAD_DIM) as usize);
    let tmp = alloc_f32(device, (NUM_HEADS * NWG * (HEAD_DIM + 2)) as usize);

    let params = FlashAttnVecParams {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        kv_seq_len: KV_SEQ_LEN,
        kv_capacity: KV_CAPACITY,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        mask_type: 2,
        sliding_window: SLIDING_WINDOW,
        softcap: 0.0,
    };

    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("encoder");
        flash_attn_vec(&mut enc, registry, device, &q, &k, &v, &out, &tmp, &params)
            .expect("warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    let mut samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        for _ in 0..BATCH {
            flash_attn_vec(&mut enc, registry, device, &q, &k, &v, &out, &tmp, &params)
                .expect("dispatch");
        }
        enc.commit_and_wait().expect("commit");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
        samples.push(elapsed_us);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let kv_bytes = 2 * (NUM_KV_HEADS as u64) * (KV_SEQ_LEN as u64) * (HEAD_DIM as u64) * 4;
    (percentile(&samples, 0.5), kv_bytes)
}

fn bench_sdpa_f16(device: &MlxDevice, registry: &mut KernelRegistry) -> (f64, u64) {
    mlx_native::ops::flash_attn_vec::register(registry);

    let q = alloc_f32(device, (NUM_HEADS * HEAD_DIM) as usize);
    // K, V at f16 — kernel auto-routes via dtype check.
    let k = alloc_f16(device, (NUM_KV_HEADS * KV_CAPACITY * HEAD_DIM) as usize);
    let v = alloc_f16(device, (NUM_KV_HEADS * KV_CAPACITY * HEAD_DIM) as usize);
    let out = alloc_f32(device, (NUM_HEADS * HEAD_DIM) as usize);
    let tmp = alloc_f32(device, (NUM_HEADS * NWG * (HEAD_DIM + 2)) as usize);

    let params = FlashAttnVecParams {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        kv_seq_len: KV_SEQ_LEN,
        kv_capacity: KV_CAPACITY,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        mask_type: 2,
        sliding_window: SLIDING_WINDOW,
        softcap: 0.0,
    };

    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("encoder");
        flash_attn_vec(&mut enc, registry, device, &q, &k, &v, &out, &tmp, &params)
            .expect("warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    let mut samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        for _ in 0..BATCH {
            flash_attn_vec(&mut enc, registry, device, &q, &k, &v, &out, &tmp, &params)
                .expect("dispatch");
        }
        enc.commit_and_wait().expect("commit");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
        samples.push(elapsed_us);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let kv_bytes = 2 * (NUM_KV_HEADS as u64) * (KV_SEQ_LEN as u64) * (HEAD_DIM as u64) * 2;
    (percentile(&samples, 0.5), kv_bytes)
}

fn bench_sdpa_tq_hb(device: &MlxDevice, registry: &mut KernelRegistry) -> (f64, u64) {
    flash_attn_vec_tq_hb::register(registry);

    let q = alloc_f32(device, (NUM_HEADS * HEAD_DIM) as usize);
    // Production layout: k_packed [nkv, cap, hd/2] U8, k_norms [nkv, cap] F32.
    let k_packed = alloc_u8(device, (NUM_KV_HEADS * KV_CAPACITY * HEAD_DIM / 2) as usize);
    let k_norms = alloc_f32(device, (NUM_KV_HEADS * KV_CAPACITY) as usize);
    let v_packed = alloc_u8(device, (NUM_KV_HEADS * KV_CAPACITY * HEAD_DIM / 2) as usize);
    let v_norms = alloc_f32(device, (NUM_KV_HEADS * KV_CAPACITY) as usize);
    let out = alloc_f32(device, (NUM_HEADS * HEAD_DIM) as usize);
    // tmp size: same as flash_attn_vec
    let tmp = alloc_f32(device, (NUM_HEADS * NWG * (HEAD_DIM + 2)) as usize);

    let params = FlashAttnVecTqHbParams {
        num_heads: NUM_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        head_dim: HEAD_DIM,
        kv_seq_len: KV_SEQ_LEN,
        kv_capacity: KV_CAPACITY,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        mask_type: 2,
        sliding_window: SLIDING_WINDOW,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: 5,
        fuse_fwht_pre: 0,
        nsg: flash_attn_vec_tq_hb::compute_nsg(KV_SEQ_LEN),
    };

    for _ in 0..WARMUP {
        let mut enc = device.command_encoder().expect("encoder");
        flash_attn_vec_tq_hb(
            &mut enc, registry, device,
            &q, &k_packed, &k_norms, &v_packed, &v_norms, &out, &tmp, &params,
        ).expect("warmup");
        enc.commit_and_wait().expect("warmup commit");
    }

    let mut samples = Vec::with_capacity(MEASURE);
    for _ in 0..MEASURE {
        let mut enc = device.command_encoder().expect("encoder");
        let t0 = std::time::Instant::now();
        for _ in 0..BATCH {
            flash_attn_vec_tq_hb(
                &mut enc, registry, device,
                &q, &k_packed, &k_norms, &v_packed, &v_norms, &out, &tmp, &params,
            ).expect("dispatch");
        }
        enc.commit_and_wait().expect("commit");
        let elapsed_us = t0.elapsed().as_secs_f64() * 1e6 / BATCH as f64;
        samples.push(elapsed_us);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Bytes: 2 × (k_packed + k_norms) for K+V combined.
    let packed_bytes = (NUM_KV_HEADS as u64) * (KV_SEQ_LEN as u64) * (HEAD_DIM as u64) / 2;
    let norms_bytes = (NUM_KV_HEADS as u64) * (KV_SEQ_LEN as u64) * 4;
    let kv_bytes = 2 * (packed_bytes + norms_bytes);
    (percentile(&samples, 0.5), kv_bytes)
}

fn main() {
    let device = MlxDevice::new().expect("MlxDevice::new");
    let mut registry = KernelRegistry::new();

    println!("ADR-028 iter-257: SDPA KV dtype comparison at gemma4 sliding decode shape");
    println!("  num_heads={} num_kv_heads={} head_dim={}", NUM_HEADS, NUM_KV_HEADS, HEAD_DIM);
    println!("  kv_seq_len={} sliding_window={}", KV_SEQ_LEN, SLIDING_WINDOW);
    println!("  Method: BATCH={} dispatches per CB, {} measures, {} warmup",
        BATCH, MEASURE, WARMUP);
    println!();

    let (us_f32, b_f32) = bench_sdpa_f32(&device, &mut registry);
    let (us_f16, b_f16) = bench_sdpa_f16(&device, &mut registry);
    let (us_tqhb, b_tqhb) = bench_sdpa_tq_hb(&device, &mut registry);

    let gbs = |bytes: u64, us: f64| (bytes as f64) / (us / 1e6) / 1e9;

    println!(
        "{:<10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "kv_dtype", "us/call", "MB read", "GB/s", "vs F16"
    );
    println!("{}", "-".repeat(70));
    println!("{:<10} | {:>10.2} | {:>10.2} | {:>10.1} | {:>9.2}×",
        "F32",  us_f32,  b_f32  as f64 / 1e6, gbs(b_f32, us_f32),  us_f32  / us_f16);
    println!("{:<10} | {:>10.2} | {:>10.2} | {:>10.1} | {:>9.2}×",
        "F16",  us_f16,  b_f16  as f64 / 1e6, gbs(b_f16, us_f16),  1.0);
    println!("{:<10} | {:>10.2} | {:>10.2} | {:>10.1} | {:>9.2}×",
        "TQ-HB", us_tqhb, b_tqhb as f64 / 1e6, gbs(b_tqhb, us_tqhb), us_tqhb / us_f16);

    let ratio = us_tqhb / us_f16;
    println!();
    println!("Hypothesis: TQ-HB / F16 ratio");
    if ratio > 2.0 {
        println!("  ratio = {:.2}× → CONFIRMED: TQ-HB is structurally slower than F16.", ratio);
        println!("  The byte-packed quant + dequant arithmetic dominates over bandwidth savings.");
    } else if ratio > 1.0 {
        println!("  ratio = {:.2}× → PARTIAL: TQ-HB slower than F16 but < 2×.", ratio);
        println!("  Possible kernel-level lever via dequant optimization.");
    } else {
        println!("  ratio = {:.2}× → FALSIFIED: TQ-HB is FASTER than F16!", ratio);
        println!("  TQ-HB's bandwidth savings (4× less bytes) more than compensate for dequant.");
        println!("  Production decode actually benefits from TQ-HB on this kernel.");
    }
    println!();
    println!("Bandwidth ratio (F16 / TQ-HB bytes) = {:.2}× (theoretical bandwidth advantage)",
        b_f16 as f64 / b_tqhb as f64);
}
