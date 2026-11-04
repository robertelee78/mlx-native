//! Warm-resume attention microbenchmark at Gemma 4 served shapes (hf2q
//! ADR-063 Q4-GEMMA-KERNEL design §6).
//!
//! Loads no model, but times the GPU: run it only with the GPU lane and a
//! quiet host.  Each arm is timed as the full per-chunk chain hf2q encodes
//! (`chain`, scratch allocated per chunk exactly as the route would) and as
//! its attention dispatches alone (`attn`, inputs prepared once):
//!
//! | Arm | Sliding (d256) chain | Global (d512) chain |
//! |---|---|---|
//! | `vec` | vec-hybrid in 256-query chunks split at 512/1024 | same |
//! | `tiled_f16` | K compaction, F16 V dequant, Q permute, mask, fence, blk, F16 d256, O cast, O permute | K compaction, F16 V dequant, Q permute, F16 d512 resume, O cast, O permute |
//! | `precise_vcodes_{window,mask}` | Q permute, [mask, fence, blk], precise d256 `_vcodes` reading the stage in place, O permute | Q permute, precise d512 `_vcodes` in place, O permute |
//! | `precise_v32_window` | F32 V dequant, Q permute, precise d256 `_v32` (K in place), O permute | as `precise_vcodes` (d512 `_v32` is parity-only) |
//! | `precise_v16cache_window` | as `precise_vcodes_window` on an F16 V cache | precise d512 `_v16` on an F16 V cache in place |
//!
//! Sliding lanes read hf2q's ring stage (`min(history, 1023)` history rows
//! plus the chunk, capacity `1023 + min(suffix, 1024)`); global lanes read
//! the cache (capacity history + suffix).
//!
//! Cells:
//! - `four_lane`: the canonical run's warm transactions, the four lanes'
//!   chains back to back in one command buffer.  Lanes come from
//!   `GEMMA_WARM_LANES=<json>` (`{"transactions":[{"kind":..,"lanes":[[history,suffix],..]},..]}`),
//!   written by the extraction script kept with the receipt.
//! - `single`: the canonical and gate lanes one at a time.
//! - `long`: the long-context stress lanes.
//!
//! Every cell records, per arm and lane, the observed transient scratch: the
//! sum of `byte_len()` of every buffer the chain allocated (§3.8 scope), and
//! for the precise arms the §3.8 TOTAL of the same chunks.
//!
//! Method: per cell 10 warmup and 50 measured command buffers per arm,
//! arms interleaved forward then reverse, GPU timestamps; median and p90.
//! `BENCH_CELLS=four_lane,single,long` selects cell groups and
//! `BENCH_OUT=<path>` writes the JSON receipt.  The receipt's commit is
//! `GIT_COMMIT_SHA` from the run environment, else `git rev-parse HEAD` of
//! the crate checkout.  Run once per metallib path
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
enum PreciseVIn {
    Codes,
    V32,
    V16Cache,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Vec,
    TiledF16,
    Precise { v: PreciseVIn, window: bool },
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Vec => "vec".into(),
            Arm::TiledF16 => "tiled_f16".into(),
            Arm::Precise { v, window } => format!(
                "precise_{}_{}",
                match v {
                    PreciseVIn::Codes => "vcodes",
                    PreciseVIn::V32 => "v32",
                    PreciseVIn::V16Cache => "v16cache",
                },
                if window { "window" } else { "mask" }
            ),
        }
    }
}

const ARMS: [Arm; 6] = [
    Arm::Vec,
    Arm::TiledF16,
    Arm::Precise {
        v: PreciseVIn::Codes,
        window: true,
    },
    Arm::Precise {
        v: PreciseVIn::Codes,
        window: false,
    },
    Arm::Precise {
        v: PreciseVIn::V32,
        window: true,
    },
    Arm::Precise {
        v: PreciseVIn::V16Cache,
        window: true,
    },
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Sliding,
    Global,
}

fn alloc(dev: &MlxDevice, dtype: DType, n: usize) -> MlxBuffer {
    // SAFETY: every scratch buffer is fully written by the dispatch that
    // produces it before any dispatch reads it.
    unsafe { dev.alloc_buffer_for_overwrite(n.max(1) * dtype.size_of(), dtype, vec![n.max(1)]) }
        .unwrap()
}

/// One sequence's persistent inputs for one layer kind: the KV source the
/// routes read (sliding ring stage, or the global cache), the layer's Q/O
/// stream rows and the vec-hybrid slot/position arrays.
struct Lane {
    kind: Kind,
    history: usize,
    suffix: usize,
    d: usize,
    h_kv: usize,
    cap: usize,
    k: MlxBuffer,
    codes: MlxBuffer,
    norms: MlxBuffer,
    v16: MlxBuffer,
    q: MlxBuffer,
    out: MlxBuffer,
    vec_slots: MlxBuffer,
    vec_pos: MlxBuffer,
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
    let cap = match kind {
        Kind::Sliding => SLIDING_CAP - 1 + suffix.min(TILED_SLIDING_CHUNK),
        Kind::Global => history + suffix,
    };
    let normal = |rng: &mut Rng, n: usize| (0..n).map(|_| rng.normal()).collect::<Vec<_>>();
    let k = support::upload_f16(dev, &normal(&mut rng, h_kv * cap * d));
    let v16 = support::upload_f16(dev, &normal(&mut rng, h_kv * cap * d));
    let codes = alloc(dev, DType::U8, h_kv * cap * d);
    let norms = alloc(dev, DType::F32, h_kv * cap * (d / 256));
    let v_src = support::upload_f32(dev, &normal(&mut rng, cap * h_kv * d));
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
        v16,
        q: support::upload_f32(dev, &rng.rms_rows(suffix * N_HEADS, d)),
        out: alloc(dev, DType::F32, suffix * row),
        vec_slots,
        vec_pos: alloc(dev, DType::U32, suffix),
    };
    let vec_chunks = chunks(&lane, Arm::Vec);
    let positions = lane.vec_pos.as_mut_slice::<u32>().unwrap();
    for (local, len, q_offset, _) in vec_chunks {
        for i in 0..len {
            positions[local + i] = (q_offset + i) as u32;
        }
    }
    lane
}

/// The transient buffers one chunk of one arm allocates.
#[derive(Default)]
struct ChunkScratch {
    q16: Option<MlxBuffer>,
    o16: Option<MlxBuffer>,
    o32: Option<MlxBuffer>,
    dense_k: Option<MlxBuffer>,
    dense_v: Option<MlxBuffer>,
    mask: Option<MlxBuffer>,
    blk: Option<MlxBuffer>,
    vec_tmp: Option<MlxBuffer>,
}

impl ChunkScratch {
    fn byte_len(&self) -> usize {
        [
            &self.q16,
            &self.o16,
            &self.o32,
            &self.dense_k,
            &self.dense_v,
            &self.mask,
            &self.blk,
            &self.vec_tmp,
        ]
        .iter()
        .filter_map(|b| b.as_ref())
        .map(MlxBuffer::byte_len)
        .sum()
    }
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

/// Encode one chunk's attention dispatch(es) from prepared scratch.
fn encode_attention(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    enc: &mut CommandEncoder,
    lane: &Lane,
    arm: Arm,
    (local, len, q_offset, kl): (usize, usize, usize, usize),
    s: &ChunkScratch,
) {
    let (d, h, h_kv) = (lane.d, N_HEADS, lane.h_kv);
    let row = h * d;
    match (arm, lane.kind) {
        (Arm::Vec, _) => {
            let q = lane.q.slice_view((local * row * 4) as u64, len * row);
            let out = lane.out.slice_view((local * row * 4) as u64, len * row);
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
                s.vec_tmp.as_ref().unwrap(),
                &slots,
                &pos,
                &params,
            )
            .unwrap();
        }
        (Arm::TiledF16, Kind::Sliding) => {
            mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_f16_d256_with_blk(
                enc,
                dev,
                reg,
                s.q16.as_ref().unwrap(),
                s.dense_k.as_ref().unwrap(),
                s.dense_v.as_ref().unwrap(),
                s.mask.as_ref(),
                s.blk.as_ref(),
                s.o16.as_ref().unwrap(),
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
                enc,
                dev,
                reg,
                s.q16.as_ref().unwrap(),
                s.dense_k.as_ref().unwrap(),
                s.dense_v.as_ref().unwrap(),
                s.o16.as_ref().unwrap(),
                &resume_params(len, kl, q_offset, kl),
            )
            .unwrap()
        }
        (Arm::Precise { v, window }, kind) => {
            let codes = PreciseV::Codes {
                codes: &lane.codes,
                norms: &lane.norms,
                codebook_bits: CODEBOOK_BITS,
                scale_factor_d512: SCALE_FACTOR_D512,
            };
            let v_in = match (v, kind) {
                (PreciseVIn::V16Cache, _) => PreciseV::F16(&lane.v16),
                (PreciseVIn::V32, Kind::Sliding) => PreciseV::F32(s.dense_v.as_ref().unwrap()),
                _ => codes,
            };
            let q16 = s.q16.as_ref().unwrap();
            let o32 = s.o32.as_ref().unwrap();
            if kind == Kind::Global {
                precise::dispatch_flash_attn_prefill_precise_d512_resume(
                    enc,
                    dev,
                    reg,
                    q16,
                    &lane.k,
                    v_in,
                    o32,
                    &resume_params(len, kl, q_offset, lane.cap),
                )
                .unwrap();
                return;
            }
            let visibility = if window {
                PreciseD256Mask::Window {
                    window: SLIDING_CAP as u32,
                }
            } else {
                PreciseD256Mask::External {
                    mask: s.mask.as_ref().unwrap(),
                    blk: s.blk.as_ref(),
                }
            };
            precise::dispatch_flash_attn_prefill_precise_d256(
                enc,
                dev,
                reg,
                q16,
                &lane.k,
                v_in,
                visibility,
                o32,
                &FlashAttnPrecisePrefillParams {
                    n_heads: h as u32,
                    n_kv_heads: h_kv as u32,
                    head_dim: 256,
                    seq_len_q: len as u32,
                    seq_len_k: kl as u32,
                    kv_capacity: lane.cap as u32,
                    batch: 1,
                    scale: 1.0,
                    q_offset: q_offset as u32,
                },
            )
            .unwrap()
        }
    }
}

/// Encode one lane's full chain, allocating its per-chunk scratch.
fn encode_chain(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    enc: &mut CommandEncoder,
    lane: &Lane,
    arm: Arm,
) -> Vec<ChunkScratch> {
    let md = dev.metal_device();
    let (d, h, h_kv) = (lane.d, N_HEADS, lane.h_kv);
    let row = h * d;
    let mut all = Vec::new();
    for chunk in chunks(lane, arm) {
        let (local, len, q_offset, kl) = chunk;
        let mut s = ChunkScratch::default();
        if arm == Arm::Vec {
            s.vec_tmp = Some(alloc(
                dev,
                DType::F32,
                len * flash_attn_vec_hybrid::tmp_buffer_bytes(h as u32, d as u32) / 4,
            ));
            encode_attention(dev, reg, enc, lane, arm, chunk, &s);
            enc.memory_barrier();
            all.push(s);
            continue;
        }
        let q = lane.q.slice_view((local * row * 4) as u64, len * row);
        let out = lane.out.slice_view((local * row * 4) as u64, len * row);
        let q16 = alloc(dev, DType::F16, len * row);
        let o32 = alloc(dev, DType::F32, len * row);
        let rows = (lane.cap as u32, kl as u32, kl as u32, kl as u32);
        match arm {
            Arm::TiledF16 => {
                let dense_k = alloc(dev, DType::F16, h_kv * kl * d);
                let dense_v = alloc(dev, DType::F16, h_kv * kl * d);
                dispatch_kv_cache_linearize_ring_bytes(
                    enc,
                    reg,
                    md,
                    &lane.k,
                    &dense_k,
                    h_kv as u32,
                    rows.0,
                    rows.1,
                    rows.2,
                    rows.3,
                    (d * 2) as u32,
                )
                .unwrap();
                mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq_f16(
                    enc,
                    reg,
                    md,
                    &lane.codes,
                    &lane.norms,
                    &dense_v,
                    h_kv as u32,
                    d as u32,
                    lane.cap as u32,
                    0,
                    kl as u32,
                    SCALE_FACTOR_D512,
                    CODEBOOK_BITS,
                )
                .unwrap();
                s.dense_k = Some(dense_k);
                s.dense_v = Some(dense_v);
                s.o16 = Some(alloc(dev, DType::F16, len * row));
            }
            Arm::Precise {
                v: PreciseVIn::V32, ..
            } if lane.kind == Kind::Sliding => {
                let dense_v = alloc(dev, DType::F32, h_kv * kl * d);
                mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq(
                    enc,
                    reg,
                    md,
                    &lane.codes,
                    &lane.norms,
                    &dense_v,
                    h_kv as u32,
                    d as u32,
                    lane.cap as u32,
                    0,
                    kl as u32,
                    SCALE_FACTOR_D512,
                    CODEBOOK_BITS,
                )
                .unwrap();
                s.dense_v = Some(dense_v);
            }
            _ => {}
        }
        permute_021_f32_to_f16(enc, reg, md, &q, &q16, len, h, d).unwrap();
        enc.memory_barrier();
        let needs_mask = lane.kind == Kind::Sliding
            && matches!(arm, Arm::TiledF16 | Arm::Precise { window: false, .. });
        if needs_mask {
            let (mask, blk) = encode_mask_blk(dev, reg, enc, len, kl, q_offset);
            s.mask = Some(mask);
            s.blk = Some(blk);
        }
        s.q16 = Some(q16);
        s.o32 = Some(o32);
        encode_attention(dev, reg, enc, lane, arm, chunk, &s);
        enc.memory_barrier();
        if arm == Arm::TiledF16 {
            cast(
                enc,
                reg,
                md,
                s.o16.as_ref().unwrap(),
                s.o32.as_ref().unwrap(),
                len * row,
                CastDirection::F16ToF32,
            )
            .unwrap();
            enc.memory_barrier();
        }
        permute_021_f32(enc, reg, md, s.o32.as_ref().unwrap(), &out, h, len, d).unwrap();
        enc.memory_barrier();
        all.push(s);
    }
    all
}

/// Design §3.8 TOTAL of a lane's precise chunks (sliding with the external
/// mask, or global).
fn total_formula(lane: &Lane, arm: Arm) -> usize {
    let Arm::Precise { v, window } = arm else {
        unreachable!()
    };
    chunks(lane, arm)
        .into_iter()
        .map(|(_, len, _, kl)| match lane.kind {
            Kind::Sliding => {
                let v_scratch = if v == PreciseVIn::V32 { 8192 * kl } else { 0 };
                let (mask, blk) = if window {
                    (0, 0)
                } else {
                    let p = precise::precise_d256_blk_params(len as u32, kl as u32);
                    (
                        2 * len * kl,
                        flash_attn_prefill_blk::blk_buffer_byte_len(&p).unwrap(),
                    )
                };
                4096 * kl + v_scratch + 8192 * len + 16384 * len + mask + blk
            }
            Kind::Global => 16384 * len + 32768 * len,
        })
        .sum()
}

fn time_cb(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    lanes: &[Lane],
    arm: Arm,
    prepared: Option<&[Vec<ChunkScratch>]>,
) -> f64 {
    let mut enc = dev.command_encoder().unwrap();
    let mut keep = Vec::new();
    for (i, lane) in lanes.iter().enumerate() {
        match prepared {
            Some(p) => {
                for (chunk, s) in chunks(lane, arm).into_iter().zip(&p[i]) {
                    encode_attention(dev, reg, &mut enc, lane, arm, chunk, s);
                    enc.memory_barrier();
                }
            }
            None => keep.push(encode_chain(dev, reg, &mut enc, lane, arm)),
        }
    }
    let (start, end) = enc.commit_wait_with_gpu_time().unwrap();
    drop(keep);
    (end - start) * 1e3
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Time every arm on `lanes` (one command buffer per sample, the lanes back
/// to back), interleaved forward then reverse.  Returns the chain and
/// attention-only timings and each arm's observed per-lane scratch.
fn measure(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    lanes: &[Lane],
) -> (serde_json::Value, serde_json::Value, serde_json::Value) {
    // Prepare each arm's attention inputs with one full chain.
    let mut prepared: Vec<Vec<Vec<ChunkScratch>>> = Vec::new();
    for arm in ARMS {
        let mut enc = dev.command_encoder().unwrap();
        let p: Vec<_> = lanes
            .iter()
            .map(|lane| encode_chain(dev, reg, &mut enc, lane, arm))
            .collect();
        enc.commit_and_wait().unwrap();
        prepared.push(p);
    }
    let mut scratch = serde_json::Map::new();
    for (a, arm) in ARMS.into_iter().enumerate() {
        let per_lane: Vec<_> = lanes
            .iter()
            .zip(&prepared[a])
            .map(|(lane, p)| {
                let observed: usize = p.iter().map(ChunkScratch::byte_len).sum();
                match arm {
                    Arm::Precise { .. } => {
                        let total = total_formula(lane, arm);
                        serde_json::json!({"observed_bytes": observed, "total_formula_bytes": total,
                            "within_total": observed <= total})
                    }
                    _ => serde_json::json!({"observed_bytes": observed}),
                }
            })
            .collect();
        scratch.insert(arm.name(), serde_json::Value::Array(per_lane));
    }
    let mut out = Vec::new();
    for attn_only in [false, true] {
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
        let order: Vec<usize> = (0..ARMS.len()).chain((0..ARMS.len()).rev()).collect();
        for round in 0..(WARMUP + MEASURED).div_ceil(2) {
            for &a in &order {
                let p = attn_only.then(|| prepared[a].as_slice());
                let ms = time_cb(dev, reg, lanes, ARMS[a], p);
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
        out.push(serde_json::Value::Object(arms));
    }
    let attn = out.pop().unwrap();
    let chain = out.pop().unwrap();
    (chain, attn, serde_json::Value::Object(scratch))
}

fn sha256_file(path: &std::path::Path) -> String {
    std::fs::read(path)
        .map(|b| hex::encode(Sha256::digest(b)))
        .unwrap_or_default()
}

fn source_commit() -> Option<String> {
    if let Ok(sha) = std::env::var("GIT_COMMIT_SHA") {
        return Some(sha);
    }
    let out = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let groups = std::env::var("BENCH_CELLS").unwrap_or_else(|_| "four_lane,single,long".into());
    let commit = source_commit();
    if commit.is_none() {
        eprintln!("WARNING: no GIT_COMMIT_SHA and no git checkout; the receipt has no commit");
    }
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
                let (chain, attn, scratch) = measure(&dev, reg, &lanes);
                eprintln!("{group} {label} {kind_name}: chain {chain}");
                cells.push(serde_json::json!({
                    "group": group, "label": label, "kind": kind_name, "lanes": pairs,
                    "chain": chain, "attn": attn, "scratch": scratch,
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
    // cells: 25 sliding + 5 global layers per transaction, chain medians.
    let mut totals = std::collections::BTreeMap::<String, f64>::new();
    for cell in cells.iter().filter(|c| c["group"] == "four_lane") {
        let layers = if cell["kind"] == "sliding" { 25.0 } else { 5.0 };
        for arm in ARMS {
            *totals.entry(arm.name()).or_default() +=
                layers * cell["chain"][arm.name()]["median_ms"].as_f64().unwrap();
        }
    }
    let ratios: serde_json::Map<String, serde_json::Value> = if totals.is_empty() {
        serde_json::Map::new()
    } else {
        totals
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
            .collect()
    };

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
        "schema": "mlx-native.bench_flash_attn_warm_precise.v2",
        "device": dev.name(),
        "macos_build": os.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()),
        "mlx_native_version": env!("CARGO_PKG_VERSION"),
        "mlx_native_commit": commit,
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
