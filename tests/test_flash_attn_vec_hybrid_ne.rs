//! NE head-split batched decode flash (`flash_attn_vec_hybrid_batched_ne`,
//! hf2q gap program, decode sliding attention). Gemma 4 sliding shape: 16
//! query heads, 8 KV heads, d256, 4 slots.
//! - F64 reference (Class B), F16 V: every output within `REF_BOUND · max|V|`
//!   of a CPU F64 attention over the same F16-rounded Q, K and V.
//! - Against NE = 1 on TQ-HB V codes (the serving V form): max |Δ| within
//!   `NE1_BOUND · max|NE1 output|` (only reduction orders differ).
//! - Determinism (Class A): a repeat is bitwise equal.
//! - Slot invariance (Class A): a query's output is bitwise equal alone or
//!   in a 4-query dispatch at the same split count.
//! Masks: linear causal (mask_type 1) and linear sliding window (mask_type 3)
//! at kv 300 / 1000 / 1024 and a split boundary of 517.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use half::f16;
use mlx_native::ops::flash_attn_vec_hybrid::{
    self as fa, flash_attn_vec_hybrid_batched, flash_attn_vec_hybrid_batched_ne,
    FlashAttnVecTqHbParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const NH: usize = 16;
const NKV: usize = 8;
const HD: usize = 256;
const SLOTS: usize = 4;
const CAP: usize = 1024;
const REF_BOUND: f64 = 4.0 / 2048.0;
const NE1_BOUND: f64 = 1e-5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload<T: bytemuck::Pod>(dev: &MlxDevice, data: &[T], dtype: DType) -> MlxBuffer {
    let mut b = dev
        .alloc_buffer(std::mem::size_of_val(data).max(16), dtype, vec![data.len()])
        .unwrap();
    b.as_mut_slice::<T>().unwrap()[..data.len()].copy_from_slice(data);
    b
}

struct Kv {
    k: Vec<f16>,
    v16: Vec<f16>,
    k_buf: MlxBuffer,
    v16_buf: MlxBuffer,
    codes: MlxBuffer,
    norms: MlxBuffer,
}

fn make_kv(dev: &MlxDevice, reg: &mut KernelRegistry, seed: u64) -> Kv {
    let mut rng = Rng(seed | 1);
    let n = SLOTS * NKV * CAP * HD;
    let k: Vec<f16> = (0..n).map(|_| f16::from_f32(rng.next())).collect();
    let v16: Vec<f16> = (0..n).map(|_| f16::from_f32(rng.next())).collect();
    let v_src: Vec<f32> = (0..n).map(|_| rng.next()).collect();
    let k_buf = upload(dev, &k, DType::F16);
    let v16_buf = upload(dev, &v16, DType::F16);
    let codes = dev.alloc_buffer(n, DType::U8, vec![n]).unwrap();
    let norms = dev
        .alloc_buffer(SLOTS * NKV * CAP * 4, DType::F32, vec![SLOTS * NKV * CAP])
        .unwrap();
    // One quantize per slot (slot-major [slot, nkv, cap, hd] layout).
    for slot in 0..SLOTS {
        let src = upload(
            dev,
            &v_src[slot * NKV * CAP * HD..(slot + 1) * NKV * CAP * HD],
            DType::F32,
        );
        let c = codes.slice_view((slot * NKV * CAP * HD) as u64, NKV * CAP * HD);
        let nr = norms.slice_view((slot * NKV * CAP * 4) as u64, NKV * CAP);
        let mut enc = dev.command_encoder().unwrap();
        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
            &mut enc,
            reg,
            dev.metal_device(),
            &src,
            &c,
            &nr,
            NKV as u32,
            HD as u32,
            CAP as u32,
            0,
            CAP as u32,
            0,
            false,
            1.0,
            8,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
    }
    Kv {
        k,
        v16,
        k_buf,
        v16_buf,
        codes,
        norms,
    }
}

#[derive(Clone, Copy, Debug)]
struct Step {
    mask_type: u32,
    window: u32,
    /// seq_pos + 1 of each query.
    ksl: [u32; SLOTS],
}

fn params(step: &Step) -> FlashAttnVecTqHbParams {
    let max = *step.ksl.iter().max().unwrap();
    FlashAttnVecTqHbParams {
        num_heads: NH as u32,
        num_kv_heads: NKV as u32,
        head_dim: HD as u32,
        kv_seq_len: max,
        kv_capacity: CAP as u32,
        scale: 1.0,
        mask_type: step.mask_type,
        sliding_window: step.window,
        softcap: 0.0,
        ring_start: 0,
        scale_factor_d512: 1.0,
        codebook_bits: 8,
        fuse_fwht_pre: 0,
        nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(max),
    }
}

/// Run queries `slots` (query i reads slot slots[i] at ksl[slots[i]]).
fn run(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    kv: &Kv,
    q: &[f32],
    step: &Step,
    slots: &[usize],
    ne: u32,
    f16_v: bool,
) -> Vec<u32> {
    let n = slots.len();
    let qs: Vec<f32> = slots
        .iter()
        .flat_map(|&s| q[s * NH * HD..(s + 1) * NH * HD].iter().copied())
        .collect();
    let q_buf = upload(dev, &qs, DType::F32);
    let slot_ids = upload(
        dev,
        &slots.iter().map(|&s| s as u32).collect::<Vec<_>>(),
        DType::U32,
    );
    let seq_pos = upload(
        dev,
        &slots.iter().map(|&s| step.ksl[s] - 1).collect::<Vec<_>>(),
        DType::U32,
    );
    let out = dev
        .alloc_buffer(n * NH * HD * 4, DType::F32, vec![n * NH * HD])
        .unwrap();
    let tmp_bytes = n * fa::tmp_buffer_bytes(NH as u32, HD as u32);
    let tmp = dev
        .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
        .unwrap();
    let p = params(step);
    let v = if f16_v { &kv.v16_buf } else { &kv.codes };
    let mut enc = dev.command_encoder().unwrap();
    if ne == 1 {
        flash_attn_vec_hybrid_batched(
            &mut enc, reg, dev, n as u32, &q_buf, &kv.k_buf, v, &kv.norms, &out, &tmp, &slot_ids,
            &seq_pos, &p,
        )
        .unwrap();
    } else {
        flash_attn_vec_hybrid_batched_ne(
            &mut enc, reg, dev, n as u32, &q_buf, &kv.k_buf, v, &kv.norms, &out, &tmp, &slot_ids,
            &seq_pos, &p, ne,
        )
        .unwrap();
    }
    enc.commit_and_wait().unwrap();
    out.as_slice::<f32>().unwrap()[..n * NH * HD]
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn reference(kv: &Kv, q: &[f32], step: &Step, slot: usize, h: usize) -> Vec<f64> {
    let kvh = h / (NH / NKV);
    let ksl = step.ksl[slot] as usize;
    let lo = if step.mask_type == 3 && step.window > 0 && ksl > step.window as usize {
        ksl - step.window as usize
    } else {
        0
    };
    let qv: Vec<f64> = (0..HD)
        .map(|d| f16::from_f32(q[(slot * NH + h) * HD + d]).to_f64())
        .collect();
    let base = |j: usize| ((slot * NKV + kvh) * CAP + j) * HD;
    let logits: Vec<f64> = (lo..ksl)
        .map(|j| (0..HD).map(|d| qv[d] * kv.k[base(j) + d].to_f64()).sum())
        .collect();
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = logits.iter().map(|l| (l - m).exp()).sum();
    let mut out = vec![0.0; HD];
    for (i, j) in (lo..ksl).enumerate() {
        let p = (logits[i] - m).exp() / z;
        for d in 0..HD {
            out[d] += p * kv.v16[base(j) + d].to_f64();
        }
    }
    out
}

fn steps() -> Vec<Step> {
    vec![
        Step {
            mask_type: 1,
            window: 0,
            ksl: [300, 300, 300, 300],
        },
        Step {
            mask_type: 1,
            window: 0,
            ksl: [1000, 1000, 1000, 1000],
        },
        Step {
            mask_type: 3,
            window: 1024,
            ksl: [1024, 1024, 1024, 1024],
        },
        Step {
            mask_type: 3,
            window: 700,
            ksl: [1024, 900, 1010, 1024],
        },
        Step {
            mask_type: 1,
            window: 0,
            ksl: [517, 600, 700, 1000],
        },
    ]
}

fn setup() -> (MlxDevice, KernelRegistry) {
    (
        MlxDevice::new().expect("Metal device"),
        KernelRegistry::new(),
    )
}

#[test]
fn ne_matches_f64_reference_on_f16_v() {
    let (dev, mut reg) = setup();
    let kv = make_kv(&dev, &mut reg, 0xA11);
    let mut rng = Rng(0xB22);
    let q: Vec<f32> = (0..SLOTS * NH * HD).map(|_| rng.next()).collect();
    let vmax = kv.v16.iter().map(|x| x.to_f64().abs()).fold(0.0, f64::max);
    for ne in [2u32, 4] {
        for step in steps() {
            let out = run(&dev, &mut reg, &kv, &q, &step, &[0, 1, 2, 3], ne, true);
            let mut worst = 0.0f64;
            for s in 0..SLOTS {
                for h in 0..NH {
                    let r = reference(&kv, &q, &step, s, h);
                    for d in 0..HD {
                        let got = f32::from_bits(out[(s * NH + h) * HD + d]) as f64;
                        assert!(got.is_finite());
                        worst = worst.max((got - r[d]).abs());
                    }
                }
            }
            eprintln!(
                "FA_VEC_NE_REF ne={ne} step={step:?} max_abs_err={worst:.3e} bound={:.3e}",
                REF_BOUND * vmax
            );
            assert!(worst <= REF_BOUND * vmax, "ne {ne} {step:?}: {worst:e}");
        }
    }
}

#[test]
fn ne_tracks_ne1_on_tq_codes_and_is_deterministic_and_slot_invariant() {
    let (dev, mut reg) = setup();
    let kv = make_kv(&dev, &mut reg, 0xC33);
    let mut rng = Rng(0xD44);
    let q: Vec<f32> = (0..SLOTS * NH * HD).map(|_| rng.next()).collect();
    for step in steps() {
        let base = run(&dev, &mut reg, &kv, &q, &step, &[0, 1, 2, 3], 1, false);
        let scale = base
            .iter()
            .map(|&b| f32::from_bits(b).abs() as f64)
            .fold(0.0, f64::max);
        for ne in [2u32, 4] {
            let out = run(&dev, &mut reg, &kv, &q, &step, &[0, 1, 2, 3], ne, false);
            let worst = out
                .iter()
                .zip(&base)
                .map(|(&a, &b)| (f32::from_bits(a) as f64 - f32::from_bits(b) as f64).abs())
                .fold(0.0, f64::max);
            eprintln!(
                "FA_VEC_NE_VS_NE1 ne={ne} step={step:?} max_abs_diff={worst:.3e} rel={:.3e}",
                worst / scale
            );
            assert!(
                worst <= NE1_BOUND * scale,
                "ne {ne} {step:?}: {worst:e} vs scale {scale:e}"
            );
            assert!(
                run(&dev, &mut reg, &kv, &q, &step, &[0, 1, 2, 3], ne, false) == out,
                "ne {ne} {step:?}: repeat differs"
            );
            // the slot with the dispatch-max ksl keeps the same split count alone
            let s = (0..SLOTS).max_by_key(|&s| step.ksl[s]).unwrap();
            let solo = run(&dev, &mut reg, &kv, &q, &step, &[s], ne, false);
            assert!(
                solo[..] == out[s * NH * HD..(s + 1) * NH * HD],
                "ne {ne} {step:?}: slot {s} alone differs"
            );
        }
    }
}
