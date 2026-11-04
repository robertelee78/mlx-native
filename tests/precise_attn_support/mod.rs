//! Shared fixtures for the precise tiled warm attention parity tests and
//! microbenchmark (hf2q ADR-063 Q4-GEMMA-KERNEL design §5–§6).
//!
//! Inputs follow §5: Q and K rows RMS-normalized like Gemma's q/k norms, V
//! rows TQ-HB encoded by `dispatch_hadamard_quantize_kv_hb_seq` (the encoder
//! hf2q's `write_kv_rows` uses).  Every route is fed the same F16 K bytes,
//! the same TQ-HB codes and norms, and the same single F32 → F16 rounding of
//! the unscaled Q.

#![allow(dead_code)]

use half::f16;
use mlx_native::ops::flash_attn_prefill::{FlashAttnPrefillParams, FlashAttnPrefillResumeParams};
use mlx_native::ops::flash_attn_prefill_blk::{self, alloc_blk_buffer};
use mlx_native::ops::flash_attn_prefill_mask::{self, SdpaMaskParams};
use mlx_native::ops::flash_attn_prefill_precise::{
    self as precise, FlashAttnPrecisePrefillParams, PreciseD256Mask, PreciseV,
};
use mlx_native::ops::flash_attn_vec_hybrid::{self, FlashAttnVecTqHbParams};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// Gemma 4 26B-A4B attention geometry (design §3.2).
pub const N_HEADS: usize = 16;
pub const SLIDING_KV_HEADS: usize = 8;
pub const GLOBAL_KV_HEADS: usize = 2;
pub const SLIDING_WINDOW: usize = 1024;
/// TQ-HB parameters hf2q serves Gemma 4 with.
pub const CODEBOOK_BITS: u32 = 8;
pub const SCALE_FACTOR_D512: f32 = 1.0;

pub struct Ctx {
    pub device: MlxDevice,
    pub registry: KernelRegistry,
}

/// A device and a registry with every kernel the routes use.  The precise
/// entry points come from `KernelRegistry::new()`'s source map; the
/// registrations below are for the comparison routes only.
pub fn ctx() -> Ctx {
    let device = MlxDevice::new().expect("Metal device");
    let mut registry = KernelRegistry::new();
    mlx_native::ops::flash_attn_prefill::register(&mut registry);
    mlx_native::ops::flash_attn_prefill_d512::register(&mut registry);
    flash_attn_prefill_mask::register(&mut registry);
    flash_attn_prefill_blk::register(&mut registry);
    Ctx { device, registry }
}

pub struct Rng(pub u64);

impl Rng {
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.uniform(), self.uniform());
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }

    /// `rows` rows of `d` standard-normal values, each RMS-normalized.
    pub fn rms_rows(&mut self, rows: usize, d: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * d);
        for _ in 0..rows {
            let row: Vec<f32> = (0..d).map(|_| self.normal()).collect();
            let rms = (row.iter().map(|x| x * x).sum::<f32>() / d as f32).sqrt();
            out.extend(row.iter().map(|x| x / rms));
        }
        out
    }
}

pub fn f16_round(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

pub fn upload_f32(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .unwrap();
    buf.as_mut_slice::<f32>().unwrap().copy_from_slice(values);
    buf
}

pub fn upload_f16(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(values.len() * 2, DType::F16, vec![values.len()])
        .unwrap();
    for (dst, &v) in buf.as_mut_slice::<f16>().unwrap().iter_mut().zip(values) {
        *dst = f16::from_f32(v);
    }
    buf
}

pub fn upload_u8(device: &MlxDevice, values: &[u8]) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(values.len(), DType::U8, vec![values.len()])
        .unwrap();
    buf.as_mut_slice::<u8>().unwrap().copy_from_slice(values);
    buf
}

pub fn read_f32(buf: &MlxBuffer, n: usize) -> Vec<f32> {
    buf.as_slice::<f32>().unwrap()[..n].to_vec()
}

pub fn read_f16(buf: &MlxBuffer, n: usize) -> Vec<f32> {
    buf.as_slice::<f16>().unwrap()[..n]
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

pub fn zeros_f32(device: &MlxDevice, n: usize) -> MlxBuffer {
    upload_f32(device, &vec![0.0; n])
}

/// `[rows, cols, inner]` → `[cols, rows, inner]`.
pub fn transpose_01(src: &[f32], rows: usize, cols: usize, inner: usize) -> Vec<f32> {
    let mut out = vec![0.0; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            let s = (r * cols + c) * inner;
            let d = (c * rows + r) * inner;
            out[d..d + inner].copy_from_slice(&src[s..s + inner]);
        }
    }
    out
}

/// One attention problem: a single sequence (batch 1) of `ql` query rows
/// at absolute positions `q_offset..`, attending keys `0..kl`, with the
/// KV cache allocated at `cap >= kl` rows.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub d: usize,
    pub h: usize,
    pub h_kv: usize,
    pub ql: usize,
    pub kl: usize,
    pub cap: usize,
    pub q_offset: usize,
    /// Sliding window (d256); `None` = causal only.
    pub window: Option<usize>,
}

impl Shape {
    /// Sliding layer of a warm resume with `history` cached tokens and
    /// `suffix` new rows, read in place from hf2q's ring stage: it holds
    /// `min(history, 1023)` history rows and the chunk, with capacity
    /// `1023 + min(suffix, 1024)`.
    pub fn sliding(history: usize, suffix: usize) -> Self {
        let q_offset = history.min(SLIDING_WINDOW - 1);
        let kl = q_offset + suffix;
        Self {
            d: 256,
            h: N_HEADS,
            h_kv: SLIDING_KV_HEADS,
            ql: suffix,
            kl,
            cap: (SLIDING_WINDOW - 1 + suffix.min(SLIDING_WINDOW)).max(kl),
            q_offset,
            window: Some(SLIDING_WINDOW),
        }
    }

    /// Global layer of a warm resume: every cached token is a key.
    pub fn global(history: usize, suffix: usize, cap: usize) -> Self {
        let kl = history + suffix;
        Self {
            d: 512,
            h: N_HEADS,
            h_kv: GLOBAL_KV_HEADS,
            ql: suffix,
            kl,
            cap: cap.max(kl),
            q_offset: history,
            window: None,
        }
    }

    pub fn visible(&self, i: usize, k: usize) -> bool {
        let q = self.q_offset + i;
        k <= q && self.window.is_none_or(|w| q - k < w)
    }

    pub fn o_len(&self) -> usize {
        self.h * self.ql * self.d
    }
}

/// Host and device inputs of one problem.
pub struct Problem {
    pub shape: Shape,
    /// F32 Q, token-major `[ql, h, d]` (the vec-hybrid input).
    pub q32_tokens: Vec<f32>,
    /// F16-rounded Q, head-major `[h, ql, d]` (the tiled input).
    pub q16: Vec<f32>,
    /// F16 K values, `[h_kv, kl, d]`.
    pub k16: Vec<f32>,
    /// F32 TQ-HB dequantized V, `[h_kv, kl, d]`.
    pub v32: Vec<f32>,
    pub q32_tokens_buf: MlxBuffer,
    pub q16_buf: MlxBuffer,
    /// F16 K cache `[h_kv, cap, d]`; rows `[kl, cap)` are NaN.
    pub k_cache: MlxBuffer,
    /// Compacted F16 K `[h_kv, kl, d]`.
    pub k_dense: MlxBuffer,
    /// TQ-HB codes `[h_kv, cap, d]` and norms `[h_kv, cap, d/256]`; rows
    /// `[kl, cap)` are poisoned (codes 0xFF, norms NaN).
    pub codes: MlxBuffer,
    pub norms: MlxBuffer,
    /// `tq_dequantize_hb_kv_seq` output `[h_kv, kl, d]`, F32 and F16.
    pub v32_buf: MlxBuffer,
    pub v16_buf: MlxBuffer,
    /// The F16 dequant values as an F16 V cache `[h_kv, cap, d]`; rows
    /// `[kl, cap)` are NaN.
    pub v16_cache: MlxBuffer,
    /// TQ-HB codebook width and D=512 norm divisor of the codes.
    pub bits: u32,
    pub sf: f32,
}

/// Build a problem with hf2q's Gemma TQ-HB parameters.
pub fn problem(c: &mut Ctx, shape: Shape, seed: u64) -> Problem {
    problem_tq(c, shape, seed, CODEBOOK_BITS, SCALE_FACTOR_D512)
}

/// Build a problem: Gemma-like Q/K, V through the real TQ-HB encoder with
/// `bits`-wide codes and D=512 norm divisor `sf`.
pub fn problem_tq(c: &mut Ctx, shape: Shape, seed: u64, bits: u32, sf: f32) -> Problem {
    let Shape {
        d,
        h,
        h_kv,
        ql,
        kl,
        cap,
        ..
    } = shape;
    let mut rng = Rng(seed | 1);
    let q32_tokens = rng.rms_rows(ql * h, d);
    let k_rows = rng.rms_rows(h_kv * kl, d);
    let v_tokens = rng.rms_rows(kl * h_kv, d);
    let q16 = f16_round(&transpose_01(&q32_tokens, ql, h, d));
    let k16 = f16_round(&k_rows);

    // Lay `[h_kv, kl, d]` rows out as a `[h_kv, cap, d]` cache, poisoning
    // rows `[kl, cap)`.
    let cache = |rows: &[f32]| {
        let mut out = vec![f32::NAN; h_kv * cap * d];
        for hk in 0..h_kv {
            let src = &rows[hk * kl * d..(hk + 1) * kl * d];
            out[hk * cap * d..hk * cap * d + kl * d].copy_from_slice(src);
        }
        out
    };
    let dev = &c.device;
    let k_cache = upload_f16(dev, &cache(&k16));
    let k_dense = upload_f16(dev, &k16);
    let codes = upload_u8(dev, &vec![0u8; h_kv * cap * d]);
    let norms = zeros_f32(dev, h_kv * cap * (d / 256));
    let v_src = upload_f32(dev, &v_tokens);
    let v32_buf = zeros_f32(dev, h_kv * kl * d);
    let v16_buf = upload_f16(dev, &vec![0.0; h_kv * kl * d]);
    let mut enc = dev.command_encoder().unwrap();
    mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
        &mut enc,
        &mut c.registry,
        dev.metal_device(),
        &v_src,
        &codes,
        &norms,
        h_kv as u32,
        d as u32,
        cap as u32,
        0,
        kl as u32,
        0,
        false,
        sf,
        bits,
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    let q32_tokens_buf = upload_f32(dev, &q32_tokens);
    let q16_buf = upload_f16(dev, &q16);
    let mut enc = dev.command_encoder().unwrap();
    dequant(
        c,
        &mut enc,
        &shape,
        (&codes, &norms),
        (&v32_buf, &v16_buf),
        (bits, sf),
    );
    enc.commit_and_wait().unwrap();
    let v32 = read_f32(&v32_buf, h_kv * kl * d);
    let v16_cache = upload_f16(&c.device, &cache(&read_f16(&v16_buf, h_kv * kl * d)));
    // Poison the cache rows past kl after encoding and dequantizing.
    let mut codes = codes;
    let mut norms = norms;
    let npp = d / 256;
    let (codes_mut, norms_mut) = (
        codes.as_mut_slice::<u8>().unwrap(),
        norms.as_mut_slice::<f32>().unwrap(),
    );
    for hk in 0..h_kv {
        codes_mut[(hk * cap + kl) * d..(hk + 1) * cap * d].fill(0xFF);
        norms_mut[(hk * cap + kl) * npp..(hk + 1) * cap * npp].fill(f32::NAN);
    }
    Problem {
        shape,
        q32_tokens_buf,
        q16_buf,
        q32_tokens,
        q16,
        k16,
        v32,
        k_cache,
        k_dense,
        codes,
        norms,
        v32_buf,
        v16_buf,
        v16_cache,
        bits,
        sf,
    }
}

/// Encode the F32 and F16 dequantization of rows `[0, kl)`.
pub fn dequant(
    c: &mut Ctx,
    enc: &mut mlx_native::CommandEncoder,
    s: &Shape,
    (codes, norms): (&MlxBuffer, &MlxBuffer),
    (v32, v16): (&MlxBuffer, &MlxBuffer),
    (bits, sf): (u32, f32),
) {
    let dev = c.device.metal_device();
    let args = (s.h_kv as u32, s.d as u32, s.cap as u32, 0, s.kl as u32);
    mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq(
        enc,
        &mut c.registry,
        dev,
        codes,
        norms,
        v32,
        args.0,
        args.1,
        args.2,
        args.3,
        args.4,
        sf,
        bits,
    )
    .unwrap();
    mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq_f16(
        enc,
        &mut c.registry,
        dev,
        codes,
        norms,
        v16,
        args.0,
        args.1,
        args.2,
        args.3,
        args.4,
        sf,
        bits,
    )
    .unwrap();
}

/// CPU F64 attention over the given exact inputs (head-major Q `[h, ql, d]`,
/// K and V `[h_kv, kl, d]`), visibility from the shape's predicate.
pub fn reference(s: &Shape, q: &[f32], k: &[f32], v: &[f32], scale: f64) -> Vec<f64> {
    let Shape {
        d, h, h_kv, ql, kl, ..
    } = *s;
    let gqa = h / h_kv;
    let mut out = vec![0.0f64; h * ql * d];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let rows_per = (h * ql).div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_idx, chunk) in out.chunks_mut(rows_per * d).enumerate() {
            scope.spawn(move || {
                let mut p = vec![0.0f64; kl];
                for (r, o) in chunk.chunks_mut(d).enumerate() {
                    let row = chunk_idx * rows_per + r;
                    let (head, i) = (row / ql, row % ql);
                    let hk = head / gqa;
                    let qrow = &q[(head * ql + i) * d..(head * ql + i + 1) * d];
                    let mut max = f64::NEG_INFINITY;
                    for key in 0..kl {
                        p[key] = f64::NEG_INFINITY;
                        if s.visible(i, key) {
                            let krow = &k[(hk * kl + key) * d..(hk * kl + key + 1) * d];
                            let dot: f64 = qrow
                                .iter()
                                .zip(krow)
                                .map(|(&a, &b)| a as f64 * b as f64)
                                .sum();
                            p[key] = dot * scale;
                            max = max.max(p[key]);
                        }
                    }
                    if max == f64::NEG_INFINITY {
                        continue; // no visible key: O row is 0
                    }
                    let mut sum = 0.0;
                    for key in 0..kl {
                        if p[key] != f64::NEG_INFINITY {
                            p[key] = (p[key] - max).exp();
                            sum += p[key];
                            let vrow = &v[(hk * kl + key) * d..(hk * kl + key + 1) * d];
                            for (acc, &x) in o.iter_mut().zip(vrow) {
                                *acc += p[key] * x as f64;
                            }
                        }
                    }
                    o.iter_mut().for_each(|x| *x /= sum);
                }
            });
        }
    });
    out
}

/// `(rel_rms, nmax)` of `got` against `want` (design §5.1 metrics).
pub fn metrics<A: Copy + Into<f64>, B: Copy + Into<f64>>(got: &[A], want: &[B]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut err2, mut ref2, mut err_max, mut ref_max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w): (f64, f64) = (g.into(), w.into());
        assert!(g.is_finite(), "non-finite output");
        err2 += (g - w) * (g - w);
        ref2 += w * w;
        err_max = err_max.max((g - w).abs());
        ref_max = ref_max.max(w.abs());
    }
    ((err2 / ref2).sqrt(), err_max / ref_max)
}

/// Which precise V input a dispatch uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VIn {
    /// `_v32` on the dense F32 dequant output; K read in place.
    F32,
    /// `_v16` on the dense F16 dequant output with the compacted K (the
    /// N2 input).
    F16Dequant,
    /// `_v16` on the F16 V cache, read in place like K.
    F16Cache,
    /// `_vcodes` on the codes and norms, read in place like K.
    Codes,
}

/// d256 visibility of a precise dispatch (d512 is always causal).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vis {
    Window,
    MaskBlk,
    MaskOnly,
    Causal,
}

/// Encode the external SWA mask (and, with `blk`, the fence + classifier
/// hf2q uses) for a sliding shape.
pub fn encode_mask(
    c: &mut Ctx,
    enc: &mut mlx_native::CommandEncoder,
    s: &Shape,
    with_blk: bool,
) -> (MlxBuffer, Option<MlxBuffer>) {
    let mask = flash_attn_prefill_mask::build_sdpa_mask_f16(
        &c.device,
        &mut c.registry,
        enc,
        &SdpaMaskParams {
            seq_len_q: s.ql as u32,
            seq_len_k: s.kl as u32,
            window_size: s.window.map(|w| w as u32),
            causal: true,
            q_abs_offset: s.q_offset as u32,
        },
    )
    .unwrap();
    if !with_blk {
        return (mask, None);
    }
    // The mask fill is not a declared write: fence before classifying.
    enc.memory_barrier();
    let params = precise::precise_d256_blk_params(s.ql as u32, s.kl as u32);
    let blk = alloc_blk_buffer(&c.device, &params).unwrap();
    flash_attn_prefill_blk::dispatch_flash_attn_prefill_blk_f16(
        enc,
        &c.device,
        &mut c.registry,
        &mask,
        &blk,
        &params,
    )
    .unwrap();
    enc.memory_barrier();
    (mask, Some(blk))
}

/// Run one precise dispatch for `p` with Q from `q` (F16 `[h, ql, d]`) and
/// scale `scale`; returns O `[h, ql, d]`.
pub fn run_precise(
    c: &mut Ctx,
    p: &Problem,
    q: &MlxBuffer,
    vin: VIn,
    vis: Vis,
    scale: f32,
) -> Vec<f32> {
    let s = p.shape;
    let out = zeros_f32(&c.device, s.o_len());
    let mut enc = c.device.command_encoder().unwrap();
    encode_precise(c, &mut enc, p, q, vin, vis, scale, &out);
    enc.commit_and_wait().unwrap();
    read_f32(&out, s.o_len())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_precise(
    c: &mut Ctx,
    enc: &mut mlx_native::CommandEncoder,
    p: &Problem,
    q: &MlxBuffer,
    vin: VIn,
    vis: Vis,
    scale: f32,
    out: &MlxBuffer,
) {
    let s = p.shape;
    let v = match vin {
        VIn::F32 => PreciseV::F32(&p.v32_buf),
        VIn::F16Dequant => PreciseV::F16(&p.v16_buf),
        VIn::F16Cache => PreciseV::F16(&p.v16_cache),
        VIn::Codes => PreciseV::Codes {
            codes: &p.codes,
            norms: &p.norms,
            codebook_bits: p.bits,
            scale_factor_d512: p.sf,
        },
    };
    // Only the N2 input (dense F16 dequant) attends the compacted K.
    let (k, cap) = if vin == VIn::F16Dequant {
        (&p.k_dense, s.kl)
    } else {
        (&p.k_cache, s.cap)
    };
    if s.d == 512 {
        let params = FlashAttnPrefillResumeParams {
            n_heads: s.h as u32,
            n_kv_heads: s.h_kv as u32,
            head_dim: 512,
            seq_len_q: s.ql as u32,
            seq_len_k: s.kl as u32,
            batch: 1,
            scale,
            do_causal: true,
            q_offset_in_k: s.q_offset as u32,
            kv_capacity: cap as u32,
        };
        precise::dispatch_flash_attn_prefill_precise_d512_resume(
            enc,
            &c.device,
            &mut c.registry,
            q,
            k,
            v,
            out,
            &params,
        )
        .unwrap();
        return;
    }
    let params = FlashAttnPrecisePrefillParams {
        n_heads: s.h as u32,
        n_kv_heads: s.h_kv as u32,
        head_dim: 256,
        seq_len_q: s.ql as u32,
        seq_len_k: s.kl as u32,
        kv_capacity: cap as u32,
        batch: 1,
        scale,
        q_offset: s.q_offset as u32,
    };
    let built;
    let mask = match vis {
        Vis::Window => PreciseD256Mask::Window {
            window: s.window.unwrap() as u32,
        },
        Vis::Causal => {
            assert!(s.window.is_none(), "causal-only visibility needs no window");
            PreciseD256Mask::Causal
        }
        Vis::MaskBlk | Vis::MaskOnly => {
            built = encode_mask(c, enc, &s, vis == Vis::MaskBlk);
            PreciseD256Mask::External {
                mask: &built.0,
                blk: built.1.as_ref(),
            }
        }
    };
    precise::dispatch_flash_attn_prefill_precise_d256(
        enc,
        &c.device,
        &mut c.registry,
        q,
        k,
        v,
        mask,
        out,
        &params,
    )
    .unwrap();
}

/// Encode the vec-hybrid route over the whole suffix: F32 Q token-major,
/// F16 K cache and TQ-HB codes in place.  Output token-major `[ql, h, d]`.
pub fn encode_vec(
    c: &mut Ctx,
    enc: &mut mlx_native::CommandEncoder,
    p: &Problem,
    out: &MlxBuffer,
    tmp: &MlxBuffer,
    slots: &MlxBuffer,
    positions: &MlxBuffer,
) {
    let s = p.shape;
    let params = FlashAttnVecTqHbParams {
        num_heads: s.h as u32,
        num_kv_heads: s.h_kv as u32,
        head_dim: s.d as u32,
        kv_seq_len: s.kl as u32,
        kv_capacity: s.cap as u32,
        scale: 1.0,
        mask_type: if s.window.is_some() { 3 } else { 1 },
        sliding_window: s.window.unwrap_or(0) as u32,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: p.sf,
        codebook_bits: p.bits,
        fuse_fwht_pre: 0,
        nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(s.kl as u32),
    };
    flash_attn_vec_hybrid::flash_attn_vec_hybrid_batched(
        enc,
        &mut c.registry,
        &c.device,
        s.ql as u32,
        &p.q32_tokens_buf,
        &p.k_cache,
        &p.codes,
        &p.norms,
        out,
        tmp,
        slots,
        positions,
        &params,
    )
    .unwrap();
}

/// Scratch for [`encode_vec`].
pub fn vec_scratch(c: &Ctx, s: &Shape) -> (MlxBuffer, MlxBuffer, MlxBuffer, MlxBuffer) {
    let dev = &c.device;
    let out = zeros_f32(dev, s.o_len());
    let tmp_bytes = s.ql * flash_attn_vec_hybrid::tmp_buffer_bytes(s.h as u32, s.d as u32);
    let tmp = zeros_f32(dev, tmp_bytes / 4);
    let mut slots = dev.alloc_buffer(s.ql * 4, DType::U32, vec![s.ql]).unwrap();
    slots.as_mut_slice::<u32>().unwrap().fill(0);
    let mut positions = dev.alloc_buffer(s.ql * 4, DType::U32, vec![s.ql]).unwrap();
    for (i, x) in positions
        .as_mut_slice::<u32>()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *x = (s.q_offset + i) as u32;
    }
    (out, tmp, slots, positions)
}

/// The vec-hybrid route's O, head-major `[h, ql, d]`.
pub fn run_vec(c: &mut Ctx, p: &Problem) -> Vec<f32> {
    let s = p.shape;
    let (out, tmp, slots, positions) = vec_scratch(c, &s);
    let mut enc = c.device.command_encoder().unwrap();
    encode_vec(c, &mut enc, p, &out, &tmp, &slots, &positions);
    enc.commit_and_wait().unwrap();
    transpose_01(&read_f32(&out, s.o_len()), s.ql, s.h, s.d)
}

/// The `TiledF16` chain's attention (F16 Q pre-scaled in the d256 kernel,
/// F16 V, F16 O), widened to F32, head-major.
pub fn run_tiled_f16(c: &mut Ctx, p: &Problem) -> Vec<f32> {
    let s = p.shape;
    let out = upload_f16(&c.device, &vec![0.0; s.o_len()]);
    let mut enc = c.device.command_encoder().unwrap();
    if s.d == 256 {
        let (mask, blk) = encode_mask(c, &mut enc, &s, true);
        mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_f16_d256_with_blk(
            &mut enc,
            &c.device,
            &mut c.registry,
            &p.q16_buf,
            &p.k_dense,
            &p.v16_buf,
            Some(&mask),
            blk.as_ref(),
            &out,
            &FlashAttnPrefillParams {
                n_heads: s.h as u32,
                n_kv_heads: s.h_kv as u32,
                head_dim: 256,
                seq_len_q: s.ql as u32,
                seq_len_k: s.kl as u32,
                batch: 1,
                scale: 1.0,
                do_causal: false,
            },
        )
        .unwrap();
    } else {
        mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_f16_d512_resume(
            &mut enc,
            &c.device,
            &mut c.registry,
            &p.q16_buf,
            &p.k_dense,
            &p.v16_buf,
            &out,
            &FlashAttnPrefillResumeParams {
                n_heads: s.h as u32,
                n_kv_heads: s.h_kv as u32,
                head_dim: 512,
                seq_len_q: s.ql as u32,
                seq_len_k: s.kl as u32,
                batch: 1,
                scale: 1.0,
                do_causal: true,
                q_offset_in_k: s.q_offset as u32,
                kv_capacity: s.kl as u32,
            },
        )
        .unwrap();
    }
    enc.commit_and_wait().unwrap();
    read_f16(&out, s.o_len())
}
