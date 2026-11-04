//! Precise tiled warm attention parity tests (hf2q ADR-063
//! Q4-GEMMA-KERNEL, design §5).  Model-free, GPU.
//!
//! Run on both metallib paths; each must pass independently:
//!
//! ```text
//! cargo test --release --test test_flash_attn_prefill_precise -- --include-ignored --test-threads=1
//! MLX_PRECOMPILED_METALLIB=0 cargo test --release --test test_flash_attn_prefill_precise -- --include-ignored --test-threads=1
//! ```
//!
//! The served-shape tests (Gemma 4 heads, warm-resume lanes up to 1,024
//! query rows) carry an F64 CPU reference of up to ~10^10 multiply-adds,
//! so they are `#[ignore]`d for debug-build CI and run with `--release
//! --include-ignored`.  Each test prints `RECEIPT {json}` lines with the
//! measured metrics.
//!
//! §5.1 bounds (frozen): P1/P1′ `rel_rms <= 1e-5`, `nmax <= 5e-5` against
//! the F64 reference; P2 `rel_rms(precise) <= max(2 rel_rms(vec), 2e-6)`;
//! P3 `rel_rms` and `nmax` of precise against vec-hybrid `<= 2e-5`.

#![cfg(target_os = "macos")]

#[path = "precise_attn_support/mod.rs"]
mod support;

use mlx_native::ops::flash_attn_prefill::FlashAttnPrefillResumeParams;
use mlx_native::ops::flash_attn_prefill_blk::{alloc_blk_buffer, BlkParams};
use mlx_native::ops::flash_attn_prefill_precise::{
    self as precise, FlashAttnPrecisePrefillParams, PreciseD256Mask, PreciseD256MaskKind, PreciseV,
    PreciseVForm,
};
use mlx_native::{KernelPipelineOrigin, MlxError};
use sha2::{Digest, Sha256};
use support::*;

const P1_REL_RMS: f64 = 1e-5;
const P1_NMAX: f64 = 5e-5;
const P3_BOUND: f64 = 2e-5;

fn p1_pass((rel, nmax): (f64, f64)) -> bool {
    rel <= P1_REL_RMS && nmax <= P1_NMAX
}

fn p3_pass((rel, nmax): (f64, f64)) -> bool {
    rel <= P3_BOUND && nmax <= P3_BOUND
}

fn receipt(check: &str, s: &Shape, what: &str, (rel, nmax): (f64, f64), pass: bool) {
    eprintln!(
        "RECEIPT {{\"check\":\"{check}\",\"d\":{},\"ql\":{},\"kl\":{},\"q_offset\":{},\
         \"what\":\"{what}\",\"rel_rms\":{rel:.3e},\"nmax\":{nmax:.3e},\"pass\":{pass}}}",
        s.d, s.ql, s.kl, s.q_offset
    );
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn digest(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    hex::encode(Sha256::digest(&bytes))
}

/// The precise pipelines are function-constant specializations, so either
/// switch selects the runtime-source path.
fn runtime_path() -> bool {
    ["MLX_PRECOMPILED_METALLIB", "MLX_PRECOMPILED_METALLIB_FCV"].iter().any(|name| {
        matches!(std::env::var(name).as_deref(), Ok("0") | Ok("false") | Ok("off"))
    })
}

/// P1, P1′, P2, P3, C+, C−, N1, N2, N3 at one shape.
fn full_parity(c: &mut Ctx, s: Shape, seed: u64) {
    let p = problem(c, s, seed);
    let reference32 = reference(&s, &p.q16, &p.k16, &p.v32, 1.0);
    let v16_values = read_f16(&p.v16_buf, s.h_kv * s.kl * s.d);
    let reference16 = reference(&s, &p.q16, &p.k16, &v16_values, 1.0);
    let vis = if s.d == 256 {
        Vis::Window
    } else {
        Vis::MaskBlk
    };

    // P1 (F32 V forms) and P1′ (F16 V cache input).
    let mut forms = vec![("v32", VIn::F32), ("vcodes", VIn::Codes)];
    if s.d == 512 {
        forms.retain(|(name, _)| *name == "vcodes"); // d512 `_v32` is parity-only (P9)
    }
    let mut precise_out = Vec::new();
    for (name, vin) in forms {
        let out = run_precise(c, &p, &p.q16_buf, vin, vis, 1.0);
        let m = metrics(&out, &reference32);
        receipt("P1", &s, name, m, p1_pass(m));
        assert!(p1_pass(m), "P1 {name} {m:?}");
        precise_out.push(out);
    }
    let out16 = run_precise(c, &p, &p.q16_buf, VIn::F16Dequant, vis, 1.0);
    let m = metrics(&out16, &reference16);
    receipt("P1'", &s, "v16", m, p1_pass(m));
    assert!(p1_pass(m), "P1' {m:?}");

    // C+, P2, P3 against vec-hybrid on the same codes and K.
    let vec_out = run_vec(c, &p);
    let vec_ref = metrics(&vec_out, &reference32);
    receipt("C+", &s, "vec-hybrid", vec_ref, p1_pass(vec_ref));
    assert!(p1_pass(vec_ref), "C+ vec-hybrid fails P1 {vec_ref:?}");
    for out in &precise_out {
        let rel = metrics(out, &reference32).0;
        let p2 = rel <= (2.0 * vec_ref.0).max(2e-6);
        // The P2 receipt's two metrics are rel_rms(precise) and rel_rms(vec).
        receipt(
            "P2",
            &s,
            "rel_rms-precise|rel_rms-vec",
            (rel, vec_ref.0),
            p2,
        );
        assert!(p2, "P2: precise {rel:e} vs vec {:e}", vec_ref.0);
        let m = metrics(out, &vec_out);
        receipt("P3", &s, "precise-vs-vec", m, p3_pass(m));
        assert!(p3_pass(m), "P3 {m:?}");
    }

    // C−: the TiledF16 chain fails P1 and P3.
    let tiled = run_tiled_f16(c, &p);
    let (m1, m3) = (metrics(&tiled, &reference32), metrics(&tiled, &vec_out));
    receipt("C-", &s, "tiled-f16-vs-ref", m1, !p1_pass(m1));
    receipt("C-", &s, "tiled-f16-vs-vec", m3, !p3_pass(m3));
    assert!(
        !p1_pass(m1) && !p3_pass(m3),
        "C− TiledF16 passed {m1:?} {m3:?}"
    );

    // N1: Q mismatch only — Q = round(q · log2 e), scale = ln 2.
    let q_log2e: Vec<f32> = support::transpose_01(&p.q32_tokens, s.ql, s.h, s.d)
        .iter()
        .map(|&x| x * std::f32::consts::LOG2_E)
        .collect();
    let q_log2e_buf = upload_f16(&c.device, &q_log2e);
    let n1 = run_precise(c, &p, &q_log2e_buf, VIn::Codes, vis, std::f32::consts::LN_2);
    let m = metrics(&n1, &vec_out);
    receipt("N1", &s, "q-log2e-vs-vec", m, !p3_pass(m));
    assert!(!p3_pass(m), "N1 unexpectedly passed P3 {m:?}");

    // N2: V mismatch only — `_v16` on the F16 dequant against the F32-V reference.
    let m = metrics(&out16, &reference32);
    receipt("N2", &s, "v16-dequant-vs-ref32", m, !p1_pass(m));
    assert!(!p1_pass(m), "N2 unexpectedly passed P1 {m:?}");

    // N3: O mismatch only — precise O rounded to F16 and widened back.
    let o16 = f16_round(precise_out.last().unwrap());
    let m = metrics(&o16, &reference32);
    receipt("N3", &s, "o-f16-vs-ref", m, !p1_pass(m));
    assert!(!p1_pass(m), "N3 unexpectedly passed P1 {m:?}");
}

#[test]
#[ignore = "served shapes: run with --release --include-ignored"]
fn served_sliding_lanes_meet_primitive_bounds_and_controls() {
    let mut c = ctx();
    // Gate lanes (714, 200) and (1254, 385), and the full 1,024-row window.
    for (i, (history, suffix)) in [(714, 200), (1254, 385), (1023, 1024)]
        .into_iter()
        .enumerate()
    {
        full_parity(&mut c, Shape::sliding(history, suffix), 0x51D0 + i as u64);
    }
}

#[test]
#[ignore = "served shapes: run with --release --include-ignored"]
fn served_global_lanes_meet_primitive_bounds_and_controls() {
    let mut c = ctx();
    for (i, (history, suffix)) in [(714, 200), (1254, 385), (1023, 1024)]
        .into_iter()
        .enumerate()
    {
        let s = Shape::global(history, suffix, history + suffix + 77);
        full_parity(&mut c, s, 0x610B + i as u64);
    }
}

/// Small d256 shapes: unaligned qL/kL, window edges mid-tile, fully masked
/// leading tiles, q_offset > 0, GQA 2.  P1/P1′ and P6 (mask + blk, mask
/// alone and the in-kernel window are bitwise equal).
#[test]
fn edge_sliding_shapes_meet_p1_and_masks_agree_bitwise() {
    let mut c = ctx();
    for (seed, (ql, q_offset, window)) in [
        (1, 140, 40),
        (31, 119, 37),
        (32, 96, 24),
        (33, 211, 50),
        (33, 0, 1024),
    ]
    .into_iter()
    .enumerate()
    {
        let kl = q_offset + ql;
        let s = Shape {
            d: 256,
            h: N_HEADS,
            h_kv: SLIDING_KV_HEADS,
            ql,
            kl,
            cap: kl,
            q_offset,
            window: Some(window),
        };
        let p = problem(&mut c, s, 0xED6E + seed as u64);
        let reference32 = reference(&s, &p.q16, &p.k16, &p.v32, 1.0);
        for vin in [VIn::F32, VIn::Codes] {
            let window_out = run_precise(&mut c, &p, &p.q16_buf, vin, Vis::Window, 1.0);
            let m = metrics(&window_out, &reference32);
            receipt("P1", &s, &format!("{vin:?}"), m, p1_pass(m));
            assert!(p1_pass(m), "P1 {vin:?} {s:?} {m:?}");
            for vis in [Vis::MaskBlk, Vis::MaskOnly] {
                let out = run_precise(&mut c, &p, &p.q16_buf, vin, vis, 1.0);
                assert!(
                    bits_equal(&out, &window_out),
                    "P6 {vis:?} != window {vin:?} {s:?}"
                );
            }
        }
        let v16_values = read_f16(&p.v16_buf, s.h_kv * s.kl * s.d);
        let reference16 = reference(&s, &p.q16, &p.k16, &v16_values, 1.0);
        let out16 = run_precise(&mut c, &p, &p.q16_buf, VIn::F16Dequant, Vis::Window, 1.0);
        let m = metrics(&out16, &reference16);
        assert!(p1_pass(m), "P1' {s:?} {m:?}");
    }
}

/// Small d512 shapes: unaligned qL/kL (kL not a multiple of 64), q_offset
/// > 0, GQA 8, in place with kv_capacity > kL.
#[test]
fn edge_global_shapes_meet_p1() {
    let mut c = ctx();
    for (seed, (ql, history)) in [(1, 100), (31, 70), (32, 64), (33, 97), (9, 0)]
        .into_iter()
        .enumerate()
    {
        let s = Shape::global(history, ql, history + ql + 13);
        let p = problem(&mut c, s, 0xD512 + seed as u64);
        let reference32 = reference(&s, &p.q16, &p.k16, &p.v32, 1.0);
        let out = run_precise(&mut c, &p, &p.q16_buf, VIn::Codes, Vis::MaskBlk, 1.0);
        let m = metrics(&out, &reference32);
        receipt("P1", &s, "vcodes", m, p1_pass(m));
        assert!(p1_pass(m), "P1 {s:?} {m:?}");
        let v16_values = read_f16(&p.v16_buf, s.h_kv * s.kl * s.d);
        let reference16 = reference(&s, &p.q16, &p.k16, &v16_values, 1.0);
        let out16 = run_precise(&mut c, &p, &p.q16_buf, VIn::F16Dequant, Vis::MaskBlk, 1.0);
        let m = metrics(&out16, &reference16);
        assert!(p1_pass(m), "P1' {s:?} {m:?}");
    }
}

/// P9: `_vcodes` O is bitwise the `_v32` O of the same template fed the
/// `tq_dequantize_hb_kv_seq` output, for every codebook width, on d256 and
/// d512 (parity-only `_v32`).  Only the V load differs, so equal O proves
/// the in-kernel V values equal the dequant output.
#[test]
fn vcodes_equals_v32_on_dequant_output_bitwise() {
    let mut c = ctx();
    for bits in [5, 6, 8] {
        for (sf, s) in [
            (1.0, Shape::sliding(300, 77)),
            (1.0, Shape::global(300, 77, 377)),
            (3.0, Shape::global(150, 45, 195)),
        ] {
            let p = problem_tq(&mut c, s, 0x9 + bits as u64, bits, sf);
            let vis = if s.d == 256 {
                Vis::Window
            } else {
                Vis::MaskBlk
            };
            let codes = run_precise(&mut c, &p, &p.q16_buf, VIn::Codes, vis, 1.0);
            let dense = run_precise(&mut c, &p, &p.q16_buf, VIn::F32, vis, 1.0);
            assert!(
                bits_equal(&codes, &dense),
                "P9 d{} bits {bits} sf {sf}",
                s.d
            );
        }
    }
}

/// P4: 20 repeats are bitwise equal, per specialization.
#[test]
fn repeats_are_bitwise_deterministic() {
    let mut c = ctx();
    let sliding = problem(&mut c, Shape::sliding(500, 61), 0xDE7);
    let global = problem(&mut c, Shape::global(500, 61, 600), 0xDE8);
    let cases: Vec<(&Problem, VIn, Vis)> = vec![
        (&sliding, VIn::F32, Vis::Window),
        (&sliding, VIn::Codes, Vis::Window),
        (&sliding, VIn::F16Dequant, Vis::Window),
        (&sliding, VIn::Codes, Vis::MaskBlk),
        (&sliding, VIn::F32, Vis::MaskOnly),
        (&global, VIn::Codes, Vis::MaskBlk),
        (&global, VIn::F16Dequant, Vis::MaskBlk),
        (&global, VIn::F32, Vis::MaskBlk),
    ];
    for (p, vin, vis) in cases {
        let first = run_precise(&mut c, p, &p.q16_buf, vin, vis, 1.0);
        for _ in 1..20 {
            let again = run_precise(&mut c, p, &p.q16_buf, vin, vis, 1.0);
            assert!(
                bits_equal(&first, &again),
                "P4 d{} {vin:?} {vis:?}",
                p.shape.d
            );
        }
    }
}

/// Copy `values` and append `extra` poison values.
fn with_tail<T: Copy>(values: &[T], extra: usize, poison: T) -> Vec<T> {
    let mut out = values.to_vec();
    out.extend(std::iter::repeat_n(poison, extra));
    out
}

/// P5 (d256): NaN K/V/Q rows past kL / qL, 0xFF codes and NaN norms, give
/// bitwise the clean O.  One KV head, so the last head's tail tile reads
/// run into the poison.
#[test]
fn d256_tail_poison_does_not_reach_o() {
    let mut c = ctx();
    let s = Shape {
        h: 2,
        h_kv: 1,
        ..Shape::sliding(203, 33)
    };
    let p = problem(&mut c, s, 0x7A11);
    let pad = 64 * s.d;
    let dev = &c.device;
    let q = upload_f16(dev, &with_tail(&p.q16, pad, f32::NAN));
    let k = upload_f16(dev, &with_tail(&p.k16, pad, f32::NAN));
    let v32 = upload_f32(dev, &with_tail(&p.v32, pad, f32::NAN));
    let v16_clean = read_f16(&p.v16_buf, s.kl * s.d);
    let v16 = upload_f16(dev, &with_tail(&v16_clean, pad, f32::NAN));
    let codes_clean = p.codes.as_slice::<u8>().unwrap().to_vec();
    let codes = upload_u8(dev, &with_tail(&codes_clean, pad, 0xFF));
    let norms_clean = read_f32(&p.norms, s.kl);
    let norms = upload_f32(dev, &with_tail(&norms_clean, 64, f32::NAN));
    let params = FlashAttnPrecisePrefillParams {
        n_heads: s.h as u32,
        n_kv_heads: 1,
        head_dim: 256,
        seq_len_q: s.ql as u32,
        seq_len_k: s.kl as u32,
        batch: 1,
        scale: 1.0,
        q_offset: s.q_offset as u32,
    };
    let codes_v = PreciseV::Codes {
        codes: &codes,
        norms: &norms,
        codebook_bits: 8,
        scale_factor_d512: 1.0,
    };
    let forms = [
        (VIn::F32, PreciseV::F32(&v32)),
        (VIn::F16Dequant, PreciseV::F16(&v16)),
        (VIn::Codes, codes_v),
    ];
    for (vin, v) in forms {
        let clean = run_precise(&mut c, &p, &p.q16_buf, vin, Vis::Window, 1.0);
        let out = zeros_f32(&c.device, s.o_len());
        let mut enc = c.device.command_encoder().unwrap();
        let mask = PreciseD256Mask::Window {
            window: SLIDING_WINDOW as u32,
        };
        precise::dispatch_flash_attn_prefill_precise_d256(
            &mut enc,
            &c.device,
            &mut c.registry,
            &q,
            &k,
            v,
            mask,
            &out,
            &params,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
        assert!(
            bits_equal(&read_f32(&out, s.o_len()), &clean),
            "P5 d256 {vin:?}"
        );
    }
}

/// P10 (and P5 for d512): `_vcodes` in place with kv_capacity > kL gives
/// bitwise the O of a compacted copy (capacity = kL).  Rows
/// `[kL, capacity)` are poisoned (K NaN, norms NaN, codes 0xFF) and Q rows
/// past qL are NaN.
#[test]
fn d512_in_place_read_matches_compacted_copy_under_poison() {
    let mut c = ctx();
    // (95, 33): kL = 128 is a multiple of the 64-key chunk and the last
    // partial Q tile's causal limit overshoots kL, the case the precise
    // chunk clamp exists for.
    for (history, suffix) in [(140, 33), (64, 64), (300, 1), (95, 33)] {
        let compact = Shape::global(history, suffix, 0);
        let p = problem(&mut c, compact, 0x1010 + history as u64);
        let clean_codes = run_precise(&mut c, &p, &p.q16_buf, VIn::Codes, Vis::MaskBlk, 1.0);
        let clean_v16 = run_precise(&mut c, &p, &p.q16_buf, VIn::F16Dequant, Vis::MaskBlk, 1.0);

        let (s, cap, d) = (compact, compact.kl + 70, 512);
        let mut k = vec![f32::NAN; s.h_kv * cap * d];
        let mut v16 = vec![f32::NAN; s.h_kv * cap * d];
        let mut codes = vec![0xFFu8; s.h_kv * cap * d];
        let mut norms = vec![f32::NAN; s.h_kv * cap * 2];
        let codes_clean = p.codes.as_slice::<u8>().unwrap().to_vec();
        let norms_clean = read_f32(&p.norms, s.h_kv * s.kl * 2);
        let v16_clean = read_f16(&p.v16_buf, s.h_kv * s.kl * d);
        for hk in 0..s.h_kv {
            let (src, dst) = (hk * s.kl * d, hk * cap * d);
            k[dst..dst + s.kl * d].copy_from_slice(&p.k16[src..src + s.kl * d]);
            v16[dst..dst + s.kl * d].copy_from_slice(&v16_clean[src..src + s.kl * d]);
            codes[dst..dst + s.kl * d].copy_from_slice(&codes_clean[src..src + s.kl * d]);
            let (src, dst) = (hk * s.kl * 2, hk * cap * 2);
            norms[dst..dst + s.kl * 2].copy_from_slice(&norms_clean[src..src + s.kl * 2]);
        }
        let dev = &c.device;
        let (k, codes, norms) = (
            upload_f16(dev, &k),
            upload_u8(dev, &codes),
            upload_f32(dev, &norms),
        );
        let v16 = upload_f16(dev, &v16);
        let q = upload_f16(dev, &with_tail(&p.q16, 8 * d, f32::NAN));
        let params = FlashAttnPrefillResumeParams {
            n_heads: s.h as u32,
            n_kv_heads: s.h_kv as u32,
            head_dim: 512,
            seq_len_q: s.ql as u32,
            seq_len_k: s.kl as u32,
            batch: 1,
            scale: 1.0,
            do_causal: true,
            q_offset_in_k: s.q_offset as u32,
            kv_capacity: cap as u32,
        };
        let codes_v = PreciseV::Codes {
            codes: &codes,
            norms: &norms,
            codebook_bits: 8,
            scale_factor_d512: 1.0,
        };
        for (v, clean, what) in [
            (codes_v, &clean_codes, "vcodes"),
            (PreciseV::F16(&v16), &clean_v16, "v16"),
        ] {
            let out = zeros_f32(&c.device, s.o_len());
            let mut enc = c.device.command_encoder().unwrap();
            precise::dispatch_flash_attn_prefill_precise_d512_resume(
                &mut enc,
                &c.device,
                &mut c.registry,
                &q,
                &k,
                v,
                &out,
                &params,
            )
            .unwrap();
            enc.commit_and_wait().unwrap();
            let got = read_f32(&out, s.o_len());
            assert!(bits_equal(&got, clean), "P10 {what} history {history}");
        }
    }
}

/// A row with no visible key is written as zeros (external mask).
#[test]
fn row_without_visible_key_is_zero() {
    let mut c = ctx();
    let s = Shape::sliding(40, 33);
    let p = problem(&mut c, s, 0x2E70);
    let mut mask = vec![0.0f32; s.ql * s.kl];
    for i in 0..s.ql {
        for k in 0..s.kl {
            let hidden = !s.visible(i, k) || i == 5;
            mask[i * s.kl + k] = if hidden { f32::NEG_INFINITY } else { 0.0 };
        }
    }
    let mask_buf = upload_f16(&c.device, &mask)
        .with_shape(vec![s.ql, s.kl])
        .unwrap();
    let out = zeros_f32(&c.device, s.o_len());
    let mut enc = c.device.command_encoder().unwrap();
    precise::dispatch_flash_attn_prefill_precise_d256(
        &mut enc,
        &c.device,
        &mut c.registry,
        &p.q16_buf,
        &p.k_cache,
        PreciseV::F32(&p.v32_buf),
        PreciseD256Mask::External {
            mask: &mask_buf,
            blk: None,
        },
        &out,
        &FlashAttnPrecisePrefillParams {
            n_heads: s.h as u32,
            n_kv_heads: s.h_kv as u32,
            head_dim: 256,
            seq_len_q: s.ql as u32,
            seq_len_k: s.kl as u32,
            batch: 1,
            scale: 1.0,
            q_offset: 0,
        },
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    let o = read_f32(&out, s.o_len());
    for head in 0..s.h {
        for i in 0..s.ql {
            let row = &o[(head * s.ql + i) * s.d..(head * s.ql + i + 1) * s.d];
            if i == 5 {
                assert!(row.iter().all(|&x| x.to_bits() == 0), "masked row not +0");
            } else {
                assert!(row.iter().all(|x| x.is_finite()) && row.iter().any(|&x| x != 0.0));
            }
        }
    }
}

/// Batch 2 once: two sequences in one dispatch, each against its own
/// reference.
#[test]
fn batch_two_matches_each_sequence_reference() {
    let mut c = ctx();
    let s = Shape::sliding(90, 45);
    let (a, b) = (problem(&mut c, s, 0xBA1), problem(&mut c, s, 0xBA2));
    let cat = |x: &[f32], y: &[f32]| [x, y].concat();
    let dev = &c.device;
    let q = upload_f16(dev, &cat(&a.q16, &b.q16));
    let k = upload_f16(dev, &cat(&a.k16, &b.k16));
    let codes = upload_u8(
        dev,
        &[
            a.codes.as_slice::<u8>().unwrap(),
            b.codes.as_slice::<u8>().unwrap(),
        ]
        .concat(),
    );
    let n = s.h_kv * s.kl;
    let norms = upload_f32(dev, &cat(&read_f32(&a.norms, n), &read_f32(&b.norms, n)));
    let out = zeros_f32(dev, 2 * s.o_len());
    let mut enc = c.device.command_encoder().unwrap();
    precise::dispatch_flash_attn_prefill_precise_d256(
        &mut enc,
        &c.device,
        &mut c.registry,
        &q,
        &k,
        PreciseV::Codes {
            codes: &codes,
            norms: &norms,
            codebook_bits: 8,
            scale_factor_d512: 1.0,
        },
        PreciseD256Mask::Window {
            window: SLIDING_WINDOW as u32,
        },
        &out,
        &FlashAttnPrecisePrefillParams {
            n_heads: s.h as u32,
            n_kv_heads: s.h_kv as u32,
            head_dim: 256,
            seq_len_q: s.ql as u32,
            seq_len_k: s.kl as u32,
            batch: 2,
            scale: 1.0,
            q_offset: s.q_offset as u32,
        },
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    let o = read_f32(&out, 2 * s.o_len());
    for (seq, p) in [&a, &b].into_iter().enumerate() {
        let want = reference(&s, &p.q16, &p.k16, &p.v32, 1.0);
        let m = metrics(&o[seq * s.o_len()..(seq + 1) * s.o_len()], &want);
        assert!(p1_pass(m), "batch 2 sequence {seq} {m:?}");
    }
}

/// P8: every precise specialization resolves from the expected origin
/// (precompiled metallib by default, runtime source under
/// `MLX_PRECOMPILED_METALLIB=0`), and the O digest of a fixed case is
/// recorded for the cross-path comparison.
#[test]
fn pipelines_resolve_from_the_expected_metallib_path() {
    let mut c = ctx();
    let expected = if runtime_path() {
        KernelPipelineOrigin::RuntimeSource
    } else {
        KernelPipelineOrigin::PrecompiledMetallib
    };
    let kinds = [
        PreciseD256MaskKind::Window,
        PreciseD256MaskKind::Causal,
        PreciseD256MaskKind::External { blk: true },
        PreciseD256MaskKind::External { blk: false },
    ];
    for form in [PreciseVForm::F32, PreciseVForm::F16, PreciseVForm::Codes] {
        for kind in kinds {
            let labels = precise::prewarm_flash_attn_prefill_precise(
                &mut c.registry,
                &c.device,
                Some((form, kind)),
                Some(form),
            )
            .unwrap();
            assert_eq!(labels.len(), 8);
            for label in labels {
                let identity = c.registry.pipeline_identity(&label).unwrap();
                assert_eq!(identity.origin, expected, "{label}");
            }
        }
    }
    for s in [Shape::sliding(714, 200), Shape::global(714, 200, 1000)] {
        let p = problem(&mut c, s, 0xD16);
        let vis = if s.d == 256 {
            Vis::Window
        } else {
            Vis::MaskBlk
        };
        let out = run_precise(&mut c, &p, &p.q16_buf, VIn::Codes, vis, 1.0);
        eprintln!(
            "RECEIPT {{\"check\":\"P8\",\"origin\":\"{expected:?}\",\"d\":{},\"o_sha256\":\"{}\"}}",
            s.d,
            digest(&out)
        );
    }
}

fn is_invalid<T>(r: mlx_native::Result<T>) -> bool {
    matches!(r, Err(MlxError::InvalidArgument(_)))
}

/// Contract violations are `InvalidArgument` before any encoding.
#[test]
fn contract_violations_are_rejected() {
    let mut c = ctx();
    let s = Shape::sliding(64, 40);
    let p = problem(&mut c, s, 0xBAD);
    let out = zeros_f32(&c.device, s.o_len());
    let base = FlashAttnPrecisePrefillParams {
        n_heads: s.h as u32,
        n_kv_heads: s.h_kv as u32,
        head_dim: 256,
        seq_len_q: s.ql as u32,
        seq_len_k: s.kl as u32,
        batch: 1,
        scale: 1.0,
        q_offset: s.q_offset as u32,
    };
    let window = PreciseD256Mask::Window { window: 1024 };
    let mut enc = c.device.command_encoder().unwrap();
    let mut d256 = |params: FlashAttnPrecisePrefillParams,
                    mask: PreciseD256Mask<'_>,
                    out: &mlx_native::MlxBuffer| {
        precise::dispatch_flash_attn_prefill_precise_d256(
            &mut enc,
            &c.device,
            &mut c.registry,
            &p.q16_buf,
            &p.k_cache,
            PreciseV::F32(&p.v32_buf),
            mask,
            out,
            &params,
        )
    };
    assert!(is_invalid(d256(
        FlashAttnPrecisePrefillParams {
            head_dim: 512,
            ..base
        },
        window,
        &out
    )));
    assert!(is_invalid(d256(
        FlashAttnPrecisePrefillParams { scale: 0.0, ..base },
        window,
        &out
    )));
    assert!(is_invalid(d256(
        FlashAttnPrecisePrefillParams {
            n_kv_heads: 3,
            ..base
        },
        window,
        &out
    )));
    assert!(is_invalid(d256(
        FlashAttnPrecisePrefillParams {
            q_offset: 65,
            ..base
        },
        window,
        &out
    )));
    assert!(is_invalid(d256(
        base,
        PreciseD256Mask::Window { window: 0 },
        &out
    )));
    let out16 = upload_f16(&c.device, &vec![0.0; s.o_len()]);
    assert!(is_invalid(d256(base, window, &out16)));
    // A blk classified with another tile is rejected, even when its byte
    // length would cover the precise grid.
    let mask = upload_f16(&c.device, &vec![0.0; s.ql * s.kl])
        .with_shape(vec![s.ql, s.kl])
        .unwrap();
    let other = BlkParams {
        seq_len_q: s.ql as u32,
        seq_len_k: s.kl as u32,
        bq: 8,
        bk: 8,
    };
    let wrong_blk = alloc_blk_buffer(&c.device, &other).unwrap();
    let external = PreciseD256Mask::External {
        mask: &mask,
        blk: Some(&wrong_blk),
    };
    assert!(is_invalid(d256(base, external, &out)));
    let right = precise::precise_d256_blk_params(s.ql as u32, s.kl as u32);
    let right_blk = alloc_blk_buffer(&c.device, &right).unwrap();
    let external = PreciseD256Mask::External {
        mask: &mask,
        blk: Some(&right_blk),
    };
    assert!(d256(base, external, &out).is_ok());

    let g = Shape::global(64, 40, 120);
    let pg = problem(&mut c, g, 0xBAE);
    let out = zeros_f32(&c.device, g.o_len());
    let resume = FlashAttnPrefillResumeParams {
        n_heads: g.h as u32,
        n_kv_heads: g.h_kv as u32,
        head_dim: 512,
        seq_len_q: g.ql as u32,
        seq_len_k: g.kl as u32,
        batch: 1,
        scale: 1.0,
        do_causal: true,
        q_offset_in_k: g.q_offset as u32,
        kv_capacity: g.cap as u32,
    };
    let mut enc = c.device.command_encoder().unwrap();
    let mut d512 = |params: FlashAttnPrefillResumeParams, bits: u32, sf: f32| {
        precise::dispatch_flash_attn_prefill_precise_d512_resume(
            &mut enc,
            &c.device,
            &mut c.registry,
            &pg.q16_buf,
            &pg.k_cache,
            PreciseV::Codes {
                codes: &pg.codes,
                norms: &pg.norms,
                codebook_bits: bits,
                scale_factor_d512: sf,
            },
            &out,
            &params,
        )
    };
    assert!(is_invalid(d512(
        FlashAttnPrefillResumeParams {
            kv_capacity: g.kl as u32 - 1,
            ..resume
        },
        8,
        1.0
    )));
    assert!(is_invalid(d512(
        FlashAttnPrefillResumeParams {
            q_offset_in_k: 65,
            ..resume
        },
        8,
        1.0
    )));
    assert!(is_invalid(d512(
        FlashAttnPrefillResumeParams {
            head_dim: 256,
            ..resume
        },
        8,
        1.0
    )));
    assert!(is_invalid(d512(resume, 4, 1.0)));
    assert!(is_invalid(d512(resume, 8, 0.0)));
    assert!(d512(resume, 8, 1.0).is_ok());
}
