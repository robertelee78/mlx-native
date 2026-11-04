//! Warm-resume attention microbenchmark at Gemma 4 served shapes (hf2q
//! ADR-063 Q4-GEMMA-KERNEL design §6).
//!
//! Loads no model, but times the GPU: run it only with the GPU lane and a
//! quiet host.  Arms, each timed as the full per-chunk chain hf2q encodes
//! (`chain`) and as the attention dispatches alone (`attn`):
//!
//! | Arm | Sliding (d256) chain | Global (d512) chain |
//! |---|---|---|
//! | `vec` | vec-hybrid in 256-query chunks split at 512/1024 | same |
//! | `tiled_f16` | K compaction, F16 V dequant, Q permute, mask, fence, blk, F16 d256, O cast, O permute | K compaction, F16 V dequant, Q permute, F16 d512 resume, O cast, O permute |
//! | `precise_{v32,vcodes}_{window,mask}` | K compaction, F32 V dequant (`v32`) or code/norm compaction (`vcodes`), Q permute, [mask, fence, blk], precise d256, O permute | Q permute, precise d512 `_vcodes` in place, O permute |
//!
//! Cells:
//! - `four_lane`: the canonical run's warm transactions, the four lanes'
//!   chains back to back in one command buffer.  Lanes come from
//!   `GEMMA_WARM_LANES=<json>` (`{"transactions":[{"kind":..,"lanes":[[history,suffix],..]},..]}`),
//!   written by the extraction script kept with the receipt.
//! - `single`: the canonical and gate lanes one at a time.
//! - `long`: the long-context stress lanes, with the precise arm's
//!   transient scratch against the design §3.8 TOTAL.
//!
//! Method: per cell 10 warmup and 50 measured command buffers per arm,
//! arms interleaved forward then reverse, GPU timestamps; median and p90.
//! `BENCH_CELLS=four_lane,single,long` selects cell groups and
//! `BENCH_OUT=<path>` writes the JSON receipt.  Run once per metallib path
//! (`MLX_PRECOMPILED_METALLIB=0` for the runtime-source path).

#[path = "../tests/precise_attn_support/mod.rs"]
#[allow(dead_code, unused_imports)]
mod support;

use mlx_native::ops::elementwise::{cast, CastDirection};
use mlx_native::ops::flash_attn_prefill::{FlashAttnPrefillParams, FlashAttnPrefillResumeParams};
use mlx_native::ops::flash_attn_prefill_blk::{self, alloc_blk_buffer};
use mlx_native::ops::flash_attn_prefill_mask::{self, SdpaMaskParams};
use mlx_native::ops::flash_attn_prefill_precise::{
    self as precise, FlashAttnPrecisePrefillParams, PreciseD256Mask, PreciseV,
};
use mlx_native::ops::flash_attn_vec_hybrid::{self, FlashAttnVecTqHbParams};
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_linearize_ring_bytes;
use mlx_native::ops::transpose::{permute_021_f32, permute_021_f32_to_f16};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};
use sha2::{Digest, Sha256};
use support::{Rng, CODEBOOK_BITS, GLOBAL_KV_HEADS, N_HEADS, SCALE_FACTOR_D512, SLIDING_KV_HEADS};

const WARMUP: usize = 10;
const MEASURED: usize = 50;
const SLIDING_CAP: usize = 1024;
const VEC_CHUNK: usize = 256;
const TILED_SLIDING_CHUNK: usize = 1024;
const TILED_GLOBAL_CHUNK: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Vec,
    TiledF16,
    Precise { codes: bool, window: bool },
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Vec => "vec".into(),
            Arm::TiledF16 => "tiled_f16".into(),
            Arm::Precise { codes, window } => format!(
                "precise_{}_{}",
                if codes { "vcodes" } else { "v32" },
                if window { "window" } else { "mask" }
            ),
        }
    }
}

const ARMS: [Arm; 6] = [
    Arm::Vec,
    Arm::TiledF16,
    Arm::Precise {
        codes: false,
        window: true,
    },
    Arm::Precise {
        codes: false,
        window: false,
    },
    Arm::Precise {
        codes: true,
        window: true,
    },
    Arm::Precise {
        codes: true,
        window: false,
    },
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Sliding,
    Global,
}

fn alloc(dev: &MlxDevice, dtype: DType, n: usize) -> MlxBuffer {
    dev.alloc_buffer(n.max(1) * dtype.size_of(), dtype, vec![n.max(1)])
        .unwrap()
}

fn filled_f16(dev: &MlxDevice, rng: &mut Rng, n: usize) -> MlxBuffer {
    support::upload_f16(dev, &(0..n).map(|_| rng.normal()).collect::<Vec<_>>())
}

/// One sequence's buffers for one layer kind: the KV source the route reads
/// (sliding ring stage, or the global cache), the layer's Q/O stream rows
/// and per-route scratch.
struct Lane {
    kind: Kind,
    history: usize,
    suffix: usize,
    d: usize,
    h_kv: usize,
    /// Source K F16 / codes / norms `[h_kv, cap, d]`.
    cap: usize,
    k: MlxBuffer,
    codes: MlxBuffer,
    norms: MlxBuffer,
    q: MlxBuffer,
    out: MlxBuffer,
    // Scratch (sized for the largest chunk).
    dense_k: MlxBuffer,
    dense_v32: MlxBuffer,
    dense_v16: MlxBuffer,
    dense_codes: MlxBuffer,
    dense_norms: MlxBuffer,
    q16: MlxBuffer,
    o16: MlxBuffer,
    o32: MlxBuffer,
    vec_tmp: MlxBuffer,
    vec_slots: MlxBuffer,
    vec_pos: MlxBuffer,
    /// Persistent mask/blk per sliding chunk for the `attn` mode.
    masks: Vec<(MlxBuffer, MlxBuffer)>,
}

/// `(local_start, len, q_offset, kv_len)` chunks of a lane for an arm.
fn chunks(lane: &Lane, arm: Arm) -> Vec<(usize, usize, usize, usize)> {
    let mut out = Vec::new();
    let mut local = 0;
    while local < lane.suffix {
        let start = lane.history + local;
        let remaining = lane.suffix - local;
        let len = match (arm, lane.kind) {
            (Arm::Vec, _) => {
                let mut len = remaining.min(VEC_CHUNK);
                for boundary in [512usize, 1024] {
                    if start < boundary && start + len > boundary {
                        len = boundary - start;
                        break;
                    }
                }
                len
            }
            (_, Kind::Sliding) => remaining.min(TILED_SLIDING_CHUNK),
            (_, Kind::Global) => remaining.min(TILED_GLOBAL_CHUNK),
        };
        let q_offset = match lane.kind {
            Kind::Sliding => start.min(SLIDING_CAP - 1),
            Kind::Global => start,
        };
        out.push((local, len, q_offset, q_offset + len));
        local += len;
    }
    out
}

fn lane(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    kind: Kind,
    history: usize,
    suffix: usize,
    seed: u64,
) -> Lane {
    let mut rng = Rng(seed | 1);
    let (d, h_kv) = match kind {
        Kind::Sliding => (256, SLIDING_KV_HEADS),
        Kind::Global => (512, GLOBAL_KV_HEADS),
    };
    // The tiled sliding stage holds cap - 1 history rows plus one 1,024-row
    // chunk; the global cache holds every row.
    let cap = match kind {
        Kind::Sliding => SLIDING_CAP - 1 + suffix.min(TILED_SLIDING_CHUNK),
        Kind::Global => history + suffix,
    };
    let max_rows = suffix.min(match kind {
        Kind::Sliding => TILED_SLIDING_CHUNK,
        Kind::Global => TILED_GLOBAL_CHUNK,
    });
    let max_kl = match kind {
        Kind::Sliding => SLIDING_CAP - 1 + max_rows,
        Kind::Global => history + suffix,
    };
    let k = filled_f16(dev, &mut rng, h_kv * cap * d);
    let codes = alloc(dev, DType::U8, h_kv * cap * d);
    let norms = alloc(dev, DType::F32, h_kv * cap * (d / 256));
    // Real TQ-HB codes and norms for the source rows.
    let v_rows: Vec<f32> = (0..cap * h_kv * d).map(|_| rng.normal()).collect();
    let v_src = support::upload_f32(dev, &v_rows);
    let mut enc = dev.command_encoder().unwrap();
    mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
        &mut enc,
        reg,
        dev.metal_device(),
        &v_src,
        &codes,
        &norms,
        h_kv as u32,
        d as u32,
        cap as u32,
        0,
        cap as u32,
        0,
        false,
        SCALE_FACTOR_D512,
        CODEBOOK_BITS,
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    let row = N_HEADS * d;
    let q_vals: Vec<f32> = rng.rms_rows(suffix * N_HEADS, d);
    let vec_rows = suffix.min(VEC_CHUNK);
    let mut vec_slots = alloc(dev, DType::U32, suffix);
    vec_slots.as_mut_slice::<u32>().unwrap().fill(0);
    let mut lane = Lane {
        kind,
        history,
        suffix,
        d,
        h_kv,
        cap,
        k,
        codes,
        norms,
        q: support::upload_f32(dev, &q_vals),
        out: alloc(dev, DType::F32, suffix * row),
        dense_k: alloc(dev, DType::F16, h_kv * max_kl * d),
        dense_v32: alloc(dev, DType::F32, h_kv * max_kl * d),
        dense_v16: alloc(dev, DType::F16, h_kv * max_kl * d),
        dense_codes: alloc(dev, DType::U8, h_kv * max_kl * d),
        dense_norms: alloc(dev, DType::F32, h_kv * max_kl * (d / 256)),
        q16: alloc(dev, DType::F16, max_rows * row),
        o16: alloc(dev, DType::F16, max_rows * row),
        o32: alloc(dev, DType::F32, max_rows * row),
        vec_tmp: alloc(
            dev,
            DType::F32,
            vec_rows * flash_attn_vec_hybrid::tmp_buffer_bytes(N_HEADS as u32, d as u32) / 4,
        ),
        vec_slots,
        vec_pos: alloc(dev, DType::U32, suffix),
        masks: Vec::new(),
    };
    let vec_chunks = chunks(&lane, Arm::Vec);
    let positions = lane.vec_pos.as_mut_slice::<u32>().unwrap();
    for (local, len, q_offset, _) in vec_chunks {
        for i in 0..len {
            positions[local + i] = (q_offset + i) as u32;
        }
    }
    if kind == Kind::Sliding {
        let tiled = chunks(&lane, Arm::TiledF16);
        let mut enc = dev.command_encoder().unwrap();
        for (_, len, q_offset, kl) in tiled {
            let (mask, blk) = encode_mask_blk(dev, reg, &mut enc, len, kl, q_offset);
            lane.masks.push((mask, blk));
        }
        enc.commit_and_wait().unwrap();
    }
    lane
}

fn encode_mask_blk(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    enc: &mut CommandEncoder,
    len: usize,
    kl: usize,
    q_offset: usize,
) -> (MlxBuffer, MlxBuffer) {
    let mask = flash_attn_prefill_mask::build_sdpa_mask_f16(
        dev,
        reg,
        enc,
        &SdpaMaskParams {
            seq_len_q: len as u32,
            seq_len_k: kl as u32,
            window_size: Some(SLIDING_CAP as u32),
            causal: true,
            q_abs_offset: q_offset as u32,
        },
    )
    .unwrap();
    enc.memory_barrier();
    let params = precise::precise_d256_blk_params(len as u32, kl as u32);
    let blk = alloc_blk_buffer(dev, &params).unwrap();
    flash_attn_prefill_blk::dispatch_flash_attn_prefill_blk_f16(
        enc, dev, reg, &mask, &blk, &params,
    )
    .unwrap();
    enc.memory_barrier();
    (mask, blk)
}

/// Encode one lane's chain (`attn_only = false`) or its attention
/// dispatches alone (`attn_only = true`, inputs prepared by a prior chain).
fn encode_lane(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    enc: &mut CommandEncoder,
    lane: &Lane,
    arm: Arm,
    attn_only: bool,
) {
    let md = dev.metal_device();
    let (d, h, h_kv) = (lane.d, N_HEADS, lane.h_kv);
    let row = h * d;
    for (chunk_idx, (local, len, q_offset, kl)) in chunks(lane, arm).into_iter().enumerate() {
        let q = lane.q.slice_view((local * row * 4) as u64, len * row);
        let out = lane.out.slice_view((local * row * 4) as u64, len * row);
        if arm == Arm::Vec {
            let params = FlashAttnVecTqHbParams {
                num_heads: h as u32,
                num_kv_heads: h_kv as u32,
                head_dim: d as u32,
                kv_seq_len: kl as u32,
                kv_capacity: lane.cap as u32,
                scale: 1.0,
                mask_type: if lane.kind == Kind::Sliding { 3 } else { 1 },
                sliding_window: if lane.kind == Kind::Sliding {
                    SLIDING_CAP as u32
                } else {
                    0
                },
                softcap: 0.0,
                ring_start: 0,
                scale_factor_d512: SCALE_FACTOR_D512,
                codebook_bits: CODEBOOK_BITS,
                fuse_fwht_pre: 0,
                nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(kl as u32),
            };
            let slots = lane.vec_slots.slice_view(0, len);
            let pos = lane.vec_pos.slice_view((local * 4) as u64, len);
            flash_attn_vec_hybrid::flash_attn_vec_hybrid_batched(
                enc,
                reg,
                dev,
                len as u32,
                &q,
                &lane.k,
                &lane.codes,
                &lane.norms,
                &out,
                &lane.vec_tmp,
                &slots,
                &pos,
                &params,
            )
            .unwrap();
            enc.memory_barrier();
            continue;
        }
        let in_place_global = lane.kind == Kind::Global && matches!(arm, Arm::Precise { .. });
        if !attn_only {
            if !in_place_global {
                let f16_row = (d * 2) as u32;
                dispatch_kv_cache_linearize_ring_bytes(
                    enc,
                    reg,
                    md,
                    &lane.k,
                    &lane.dense_k,
                    h_kv as u32,
                    lane.cap as u32,
                    kl as u32,
                    kl as u32,
                    kl as u32,
                    f16_row,
                )
                .unwrap();
                match arm {
                    Arm::TiledF16 => {
                        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq_f16(
                            enc,
                            reg,
                            md,
                            &lane.codes,
                            &lane.norms,
                            &lane.dense_v16,
                            h_kv as u32,
                            d as u32,
                            lane.cap as u32,
                            0,
                            kl as u32,
                            SCALE_FACTOR_D512,
                            CODEBOOK_BITS,
                        )
                        .unwrap()
                    }
                    Arm::Precise { codes: false, .. } => {
                        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq(
                            enc,
                            reg,
                            md,
                            &lane.codes,
                            &lane.norms,
                            &lane.dense_v32,
                            h_kv as u32,
                            d as u32,
                            lane.cap as u32,
                            0,
                            kl as u32,
                            SCALE_FACTOR_D512,
                            CODEBOOK_BITS,
                        )
                        .unwrap()
                    }
                    Arm::Precise { codes: true, .. } => {
                        for (src, dst, row_bytes) in [
                            (&lane.codes, &lane.dense_codes, d as u32),
                            (&lane.norms, &lane.dense_norms, 4 * (d / 256) as u32),
                        ] {
                            dispatch_kv_cache_linearize_ring_bytes(
                                enc,
                                reg,
                                md,
                                src,
                                dst,
                                h_kv as u32,
                                lane.cap as u32,
                                kl as u32,
                                kl as u32,
                                kl as u32,
                                row_bytes,
                            )
                            .unwrap();
                        }
                    }
                    Arm::Vec => unreachable!(),
                }
            }
            permute_021_f32_to_f16(enc, reg, md, &q, &lane.q16, len, h, d).unwrap();
            enc.memory_barrier();
        }
        let window = matches!(arm, Arm::Precise { window: true, .. });
        let fresh;
        let mask_blk = if lane.kind == Kind::Sliding && !window {
            if attn_only {
                Some((&lane.masks[chunk_idx].0, &lane.masks[chunk_idx].1))
            } else {
                fresh = encode_mask_blk(dev, reg, enc, len, kl, q_offset);
                Some((&fresh.0, &fresh.1))
            }
        } else {
            None
        };
        match (arm, lane.kind) {
            (Arm::TiledF16, Kind::Sliding) => {
                mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_f16_d256_with_blk(
                    enc, dev, reg, &lane.q16, &lane.dense_k, &lane.dense_v16,
                    mask_blk.map(|m| m.0), mask_blk.map(|m| m.1), &lane.o16,
                    &FlashAttnPrefillParams {
                        n_heads: h as u32,
                        n_kv_heads: h_kv as u32,
                        head_dim: 256,
                        seq_len_q: len as u32,
                        seq_len_k: kl as u32,
                        batch: 1,
                        scale: 1.0,
                        do_causal: false,
                    },
                )
                .unwrap()
            }
            (Arm::TiledF16, Kind::Global) => {
                mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_f16_d512_resume(
                    enc, dev, reg, &lane.q16, &lane.dense_k, &lane.dense_v16, &lane.o16,
                    &resume_params(len, kl, q_offset, kl),
                )
                .unwrap()
            }
            (Arm::Precise { codes, .. }, Kind::Sliding) => {
                let v = if codes {
                    PreciseV::Codes {
                        codes: &lane.dense_codes,
                        norms: &lane.dense_norms,
                        codebook_bits: CODEBOOK_BITS,
                        scale_factor_d512: SCALE_FACTOR_D512,
                    }
                } else {
                    PreciseV::F32(&lane.dense_v32)
                };
                let visibility = if window {
                    PreciseD256Mask::Window { window: SLIDING_CAP as u32 }
                } else {
                    let (mask, blk) = mask_blk.unwrap();
                    PreciseD256Mask::External { mask, blk: Some(blk) }
                };
                precise::dispatch_flash_attn_prefill_precise_d256(
                    enc, dev, reg, &lane.q16, &lane.dense_k, v, visibility, &lane.o32,
                    &FlashAttnPrecisePrefillParams {
                        n_heads: h as u32,
                        n_kv_heads: h_kv as u32,
                        head_dim: 256,
                        seq_len_q: len as u32,
                        seq_len_k: kl as u32,
                        batch: 1,
                        scale: 1.0,
                        q_offset: q_offset as u32,
                    },
                )
                .unwrap()
            }
            (Arm::Precise { .. }, Kind::Global) => {
                precise::dispatch_flash_attn_prefill_precise_d512_resume(
                    enc,
                    dev,
                    reg,
                    &lane.q16,
                    &lane.k,
                    PreciseV::Codes {
                        codes: &lane.codes,
                        norms: &lane.norms,
                        codebook_bits: CODEBOOK_BITS,
                        scale_factor_d512: SCALE_FACTOR_D512,
                    },
                    &lane.o32,
                    &resume_params(len, kl, q_offset, lane.cap),
                )
                .unwrap()
            }
            (Arm::Vec, _) => unreachable!(),
        }
        enc.memory_barrier();
        if attn_only {
            continue;
        }
        if arm == Arm::TiledF16 {
            cast(
                enc,
                reg,
                md,
                &lane.o16,
                &lane.o32,
                len * row,
                CastDirection::F16ToF32,
            )
            .unwrap();
            enc.memory_barrier();
        }
        permute_021_f32(enc, reg, md, &lane.o32, &out, h, len, d).unwrap();
        enc.memory_barrier();
    }
}

fn resume_params(
    len: usize,
    kl: usize,
    q_offset: usize,
    cap: usize,
) -> FlashAttnPrefillResumeParams {
    FlashAttnPrefillResumeParams {
        n_heads: N_HEADS as u32,
        n_kv_heads: GLOBAL_KV_HEADS as u32,
        head_dim: 512,
        seq_len_q: len as u32,
        seq_len_k: kl as u32,
        batch: 1,
        scale: 1.0,
        do_causal: true,
        q_offset_in_k: q_offset as u32,
        kv_capacity: cap as u32,
    }
}

fn time_cb(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    lanes: &[Lane],
    arm: Arm,
    attn_only: bool,
) -> f64 {
    let mut enc = dev.command_encoder().unwrap();
    for lane in lanes {
        encode_lane(dev, reg, &mut enc, lane, arm, attn_only);
    }
    let (start, end) = enc.commit_wait_with_gpu_time().unwrap();
    (end - start) * 1e3
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Time every arm on `lanes` (one command buffer per sample, the lanes back
/// to back), interleaved forward then reverse.
fn measure(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    lanes: &[Lane],
    attn_only: bool,
) -> serde_json::Value {
    // Prime the attn-only inputs with one full chain per arm.
    for arm in ARMS {
        time_cb(dev, reg, lanes, arm, false);
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
    let order: Vec<usize> = (0..ARMS.len()).chain((0..ARMS.len()).rev()).collect();
    for round in 0..(WARMUP + MEASURED).div_ceil(2) {
        for &a in &order {
            let ms = time_cb(dev, reg, lanes, ARMS[a], attn_only);
            if round * 2 >= WARMUP {
                samples[a].push(ms);
            }
        }
    }
    let mut arms = serde_json::Map::new();
    for (a, mut s) in samples.into_iter().enumerate() {
        s.sort_by(f64::total_cmp);
        s.truncate(MEASURED);
        arms.insert(
            ARMS[a].name(),
            serde_json::json!({"median_ms": quantile(&s, 0.5), "p90_ms": quantile(&s, 0.9), "n": s.len()}),
        );
    }
    serde_json::Value::Object(arms)
}

/// Design §3.8 per-chunk scratch of the precise sliding chain (`v32`, mask)
/// and global chain, summed over a lane's chunks.
fn total_formula(lane: &Lane, arm: Arm) -> usize {
    chunks(lane, arm)
        .into_iter()
        .map(|(_, len, _, kl)| match lane.kind {
            Kind::Sliding => {
                let blk = precise::precise_d256_blk_params(len as u32, kl as u32);
                4096 * kl
                    + 8192 * kl
                    + 8192 * len
                    + 16384 * len
                    + 2 * len * kl
                    + flash_attn_prefill_blk::blk_buffer_byte_len(&blk).unwrap()
            }
            Kind::Global => 16384 * len + 32768 * len,
        })
        .sum()
}

/// The precise chain's per-chunk transient scratch as this bench allocates
/// it (scoped as §3.8: K, V, Q, O32, mask, blk; Q and O32 for global).
fn observed_scratch(lane: &Lane, arm: Arm) -> usize {
    chunks(lane, arm)
        .into_iter()
        .map(|(_, len, _, kl)| {
            let row = N_HEADS * lane.d;
            let (q, o32) = (len * row * 2, len * row * 4);
            match lane.kind {
                Kind::Sliding => {
                    let blk = precise::precise_d256_blk_params(len as u32, kl as u32);
                    lane.h_kv * kl * lane.d * (2 + 4)
                        + q
                        + o32
                        + 2 * len * kl
                        + flash_attn_prefill_blk::blk_buffer_byte_len(&blk).unwrap()
                }
                Kind::Global => q + o32,
            }
        })
        .sum()
}

fn sha256_file(path: &std::path::Path) -> String {
    std::fs::read(path)
        .map(|b| hex::encode(Sha256::digest(b)))
        .unwrap_or_default()
}

fn main() {
    let groups = std::env::var("BENCH_CELLS").unwrap_or_else(|_| "four_lane,single,long".into());
    let dev = MlxDevice::new().unwrap();
    let mut reg = KernelRegistry::new();
    mlx_native::ops::flash_attn_prefill::register(&mut reg);
    mlx_native::ops::flash_attn_prefill_d512::register(&mut reg);
    flash_attn_prefill_mask::register(&mut reg);
    flash_attn_prefill_blk::register(&mut reg);

    let mut cells = Vec::new();
    let mut seed = 0xBE7C_u64;
    let mut run_cell =
        |group: &str, label: String, pairs: &[(usize, usize)], reg: &mut KernelRegistry| {
            for kind in [Kind::Sliding, Kind::Global] {
                let lanes: Vec<Lane> = pairs
                    .iter()
                    .map(|&(hist, suf)| {
                        seed += 1;
                        lane(&dev, reg, kind, hist, suf, seed)
                    })
                    .collect();
                let kind_name = if kind == Kind::Sliding {
                    "sliding"
                } else {
                    "global"
                };
                let chain = measure(&dev, reg, &lanes, false);
                let attn = measure(&dev, reg, &lanes, true);
                let precise_arm = Arm::Precise {
                    codes: false,
                    window: false,
                };
                let scratch: Vec<_> = lanes
                    .iter()
                    .map(|l| {
                        serde_json::json!({
                            "observed_peak_bytes": observed_scratch(l, precise_arm),
                            "total_formula_bytes": total_formula(l, precise_arm),
                        })
                    })
                    .collect();
                eprintln!("{group} {label} {kind_name}: chain {chain}");
                cells.push(serde_json::json!({
                    "group": group, "label": label, "kind": kind_name, "lanes": pairs,
                    "chain": chain, "attn": attn, "precise_scratch": scratch,
                }));
            }
        };

    if groups.contains("four_lane") {
        let path =
            std::env::var("GEMMA_WARM_LANES").expect("GEMMA_WARM_LANES=<extracted lanes json>");
        let lanes: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for (i, t) in lanes["transactions"].as_array().unwrap().iter().enumerate() {
            let pairs: Vec<(usize, usize)> = t["lanes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    (
                        p[0].as_u64().unwrap() as usize,
                        p[1].as_u64().unwrap() as usize,
                    )
                })
                .collect();
            run_cell(
                "four_lane",
                format!("t{}-{}", i + 1, t["kind"].as_str().unwrap()),
                &pairs,
                &mut reg,
            );
        }
    }
    if groups.contains("single") {
        for pair in [
            (458, 58),
            (516, 191),
            (707, 220),
            (927, 288),
            (1215, 150),
            (1379, 348),
            (465, 56),
            (714, 200),
            (943, 311),
            (1254, 385),
        ] {
            run_cell(
                "single",
                format!("{}+{}", pair.0, pair.1),
                &[pair],
                &mut reg,
            );
        }
    }
    if groups.contains("long") {
        for pair in [(4096, 1024), (8192, 4096), (32768, 4096)] {
            run_cell("long", format!("{}+{}", pair.0, pair.1), &[pair], &mut reg);
        }
    }

    // Model-level projection and the frozen M1/M2 ratios on the four-lane
    // cells: 25 sliding + 5 global layers per transaction.
    let mut totals = std::collections::BTreeMap::<String, f64>::new();
    for cell in cells.iter().filter(|c| c["group"] == "four_lane") {
        let layers = if cell["kind"] == "sliding" { 25.0 } else { 5.0 };
        for arm in ARMS {
            *totals.entry(arm.name()).or_default() +=
                layers * cell["chain"][arm.name()]["median_ms"].as_f64().unwrap();
        }
    }
    let ratios: serde_json::Map<String, serde_json::Value> = totals
        .iter()
        .filter(|(name, _)| name.starts_with("precise"))
        .map(|(name, ms)| {
            (
                name.clone(),
                serde_json::json!({
                    "m1_precise_over_vec": ms / totals["vec"],
                    "m2_precise_over_tiled_f16": ms / totals["tiled_f16"],
                }),
            )
        })
        .collect();

    let identity = precise::prewarm_flash_attn_prefill_precise(
        &mut reg,
        &dev,
        Some((
            precise::PreciseVForm::Codes,
            precise::PreciseD256MaskKind::Window,
        )),
        Some(precise::PreciseVForm::Codes),
    )
    .unwrap()
    .into_iter()
    .map(|label| {
        let id = reg.pipeline_identity(&label).unwrap();
        serde_json::json!({"label": label, "origin": format!("{:?}", id.origin),
            "embedded_metallib_sha256": id.embedded_metallib_sha256})
    })
    .collect::<Vec<_>>();
    let exe = std::env::current_exe().unwrap();
    let os = std::process::Command::new("sw_vers")
        .arg("-buildVersion")
        .output()
        .ok();
    let receipt = serde_json::json!({
        "schema": "mlx-native.bench_flash_attn_warm_precise.v1",
        "device": dev.name(),
        "macos_build": os.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()),
        "mlx_native_version": env!("CARGO_PKG_VERSION"),
        "mlx_native_commit": option_env!("GIT_COMMIT_SHA"),
        "bench_binary_sha256": sha256_file(&exe),
        "precompiled_metallib": std::env::var("MLX_PRECOMPILED_METALLIB").ok(),
        "pipelines": identity,
        "warmup": WARMUP, "measured": MEASURED,
        "four_lane_projection_ms": totals,
        "four_lane_ratios": ratios,
        "cells": cells,
    });
    let text = serde_json::to_string_pretty(&receipt).unwrap();
    match std::env::var("BENCH_OUT") {
        Ok(path) => std::fs::write(path, text + "\n").unwrap(),
        Err(_) => println!("{text}"),
    }
}
