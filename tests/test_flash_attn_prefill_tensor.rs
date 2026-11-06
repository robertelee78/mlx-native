//! Tensor flash attention for prefill (hf2q ADR-063 gap program F1,
//! `docs/research/adr063-q4-gemma-attn-tensor-design.md` §3–§4).
//!
//! GPU tests at Gemma 4 shapes (sliding: 16 query / 8 KV heads, d256,
//! window 1024; global: 16 / 2, d512), with serving masks (causal, window,
//! padding -inf) and `blk` from `flash_attn_prefill_blk_f16`:
//! - F64 reference (Class B): every checked output element within
//!   `REF_BOUND · max|V|` of a CPU F64 attention over the same F16-rounded
//!   Q·scale, K and V;
//! - determinism (Class A): a repeat is bitwise equal;
//! - width invariance (Class A): a row's output is bitwise equal whether it
//!   is computed in the full dispatch or in a dispatch of 1, 7, 31, 32, 58
//!   rows at any offset (any position in a query tile);
//! - chunking invariance (Class A): a suffix split into chunks, each
//!   attending only `pad64(hist + chunk_end)` keys, equals the full dispatch
//!   bitwise.
//!
//! Skipped when the device lacks the Metal tensor API, unless
//! `MLX_REQUIRE_TENSOR=1` (set it on Apple10 hosts). Run once per metallib
//! path (`MLX_PRECOMPILED_METALLIB=0` for runtime source).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use half::f16;
use mlx_native::ops::flash_attn_prefill_blk::{
    alloc_blk_buffer, dispatch_flash_attn_prefill_blk_f16, BlkParams,
};
use mlx_native::ops::flash_attn_prefill_tensor::{
    self as tensor, dispatch_flash_attn_prefill_tensor_f16, FlashAttnPrefillTensorParams,
    CACHE_ROWS_PER_STEP,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// `|gpu − f64| ≤ REF_BOUND · max|V|` per output element. P is half (2^-11
/// relative) and the lazy rescale keeps P ≤ e^8; 8·2^-11 leaves headroom.
const REF_BOUND: f64 = 8.0 / 2048.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    nh: usize,
    nkv: usize,
    hd: usize,
    window: usize,
}

const SLIDING: Shape = Shape {
    name: "sliding-d256",
    nh: 16,
    nkv: 8,
    hd: 256,
    window: 1024,
};
const GLOBAL: Shape = Shape {
    name: "global-d512",
    nh: 16,
    nkv: 2,
    hd: 512,
    window: 0,
};

/// One sequence: `rows` queries at absolute positions `hist..hist+rows`,
/// keys at absolute positions `0..kv_len` (`kv_len = hist + rows`).
struct Case {
    shape: Shape,
    hist: usize,
    rows: usize,
    /// `[rows, nh, hd]` F32.
    q: Vec<f32>,
    /// `[nkv, kv_pad, hd]` F16 (rows past `kv_len` zero).
    k: Vec<f16>,
    v: Vec<f16>,
    kv_pad: usize,
    scale: f32,
}

impl Case {
    fn new(shape: Shape, hist: usize, rows: usize, q_amp: f32, seed: u64) -> Self {
        let mut rng = Rng(seed | 1);
        let kv_len = hist + rows;
        let kv_pad = tensor::kv_pad(kv_len as u32) as usize;
        let q = (0..rows * shape.nh * shape.hd)
            .map(|_| rng.next() * q_amp)
            .collect();
        let mut k = vec![f16::ZERO; shape.nkv * kv_pad * shape.hd];
        let mut v = vec![f16::ZERO; shape.nkv * kv_pad * shape.hd];
        for h in 0..shape.nkv {
            for j in 0..kv_len {
                for d in 0..shape.hd {
                    let i = (h * kv_pad + j) * shape.hd + d;
                    k[i] = f16::from_f32(rng.next());
                    v[i] = f16::from_f32(rng.next());
                }
            }
        }
        Self {
            shape,
            hist,
            rows,
            q,
            k,
            v,
            kv_pad,
            scale: 1.0,
        }
    }

    fn visible(&self, row_abs: usize, j: usize) -> bool {
        j <= row_abs && (self.shape.window == 0 || row_abs - j < self.shape.window)
    }

    /// F16 mask for query rows `r0..r0+n` over `kv_pad` columns.
    fn mask(&self, r0: usize, n: usize, kv_pad: usize) -> Vec<f16> {
        let mut m = vec![f16::NEG_INFINITY; n * kv_pad];
        let kv_len = self.hist + self.rows;
        for i in 0..n {
            let row_abs = self.hist + r0 + i;
            for j in 0..kv_len.min(kv_pad) {
                if self.visible(row_abs, j) {
                    m[i * kv_pad + j] = f16::ZERO;
                }
            }
        }
        m
    }

    /// F64 reference of one output row and head.
    fn reference(&self, r: usize, h: usize) -> Vec<f64> {
        let s = self.shape;
        let kvh = h / (s.nh / s.nkv);
        let row_abs = self.hist + r;
        let q: Vec<f64> = (0..s.hd)
            .map(|d| f16::from_f32(self.q[(r * s.nh + h) * s.hd + d] * self.scale).to_f64())
            .collect();
        let mut logits = Vec::new();
        for j in 0..self.hist + self.rows {
            if !self.visible(row_abs, j) {
                continue;
            }
            let base = (kvh * self.kv_pad + j) * s.hd;
            let dot: f64 = (0..s.hd).map(|d| q[d] * self.k[base + d].to_f64()).sum();
            logits.push((j, dot));
        }
        let max = logits.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = logits.iter().map(|x| (x.1 - max).exp()).sum();
        let mut out = vec![0.0f64; s.hd];
        for &(j, l) in &logits {
            let p = (l - max).exp() / z;
            let base = (kvh * self.kv_pad + j) * s.hd;
            for d in 0..s.hd {
                out[d] += p * self.v[base + d].to_f64();
            }
        }
        out
    }
}

struct Gpu {
    device: MlxDevice,
    reg: KernelRegistry,
}

fn gpu(head_dim: u32) -> Option<Gpu> {
    let device = MlxDevice::new().expect("Metal device");
    let mut reg = KernelRegistry::new();
    let available =
        tensor::flash_attn_prefill_tensor_available(&mut reg, &device, head_dim).expect("probe");
    if !available {
        assert!(
            std::env::var("MLX_REQUIRE_TENSOR").as_deref() != Ok("1"),
            "MLX_REQUIRE_TENSOR=1 but the tensor pipeline is unavailable"
        );
        eprintln!("flash_attn_prefill_tensor: Metal tensor API unavailable; skipped");
        return None;
    }
    Some(Gpu { device, reg })
}

fn upload<T: bytemuck::Pod>(dev: &MlxDevice, data: &[T], dtype: DType) -> MlxBuffer {
    let mut b = dev
        .alloc_buffer(std::mem::size_of_val(data).max(16), dtype, vec![data.len()])
        .expect("alloc");
    b.as_mut_slice::<T>().unwrap()[..data.len()].copy_from_slice(data);
    b
}

/// Run query rows `r0..r0+n` of `c` against its first `kv_pad` cache rows;
/// returns `[n, nh, hd]` output bits.
fn run(g: &mut Gpu, c: &Case, r0: usize, n: usize, kv_pad: usize) -> Vec<u32> {
    let s = c.shape;
    let q = upload(
        &g.device,
        &c.q[r0 * s.nh * s.hd..(r0 + n) * s.nh * s.hd],
        DType::F32,
    );
    let k = upload(&g.device, &c.k, DType::F16);
    let v = upload(&g.device, &c.v, DType::F16);
    let mask = upload(&g.device, &c.mask(r0, n, kv_pad), DType::F16);
    let bq = tensor::queries_per_threadgroup(s.hd as u32).unwrap();
    let blk_params = BlkParams {
        seq_len_q: n as u32,
        seq_len_k: kv_pad as u32,
        bq,
        bk: CACHE_ROWS_PER_STEP,
    };
    let blk = alloc_blk_buffer(&g.device, &blk_params).unwrap();
    let out_elems = n * s.nh * s.hd;
    let out = g
        .device
        .alloc_buffer(out_elems * 4, DType::F32, vec![out_elems])
        .unwrap();
    let mut enc = g.device.command_encoder().unwrap();
    dispatch_flash_attn_prefill_blk_f16(&mut enc, &g.device, &mut g.reg, &mask, &blk, &blk_params)
        .unwrap();
    enc.memory_barrier();
    dispatch_flash_attn_prefill_tensor_f16(
        &mut enc,
        &mut g.reg,
        &g.device,
        &q,
        &k,
        &v,
        &mask,
        &blk,
        &out,
        &FlashAttnPrefillTensorParams {
            rows: n as u32,
            n_heads: s.nh as u32,
            n_kv_heads: s.nkv as u32,
            head_dim: s.hd as u32,
            kv_pad: kv_pad as u32,
            q_row_stride: (s.nh * s.hd) as u32,
            q_head_stride: s.hd as u32,
            k_row_stride: s.hd as u32,
            k_head_stride: (c.kv_pad * s.hd) as u32,
            v_row_stride: s.hd as u32,
            v_head_stride: (c.kv_pad * s.hd) as u32,
            scale: c.scale,
        },
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    out.as_slice::<f32>().unwrap()[..out_elems]
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn row_bits<'a>(all: &'a [u32], c: &Case, r: usize) -> &'a [u32] {
    let w = c.shape.nh * c.shape.hd;
    &all[r * w..(r + 1) * w]
}

/// F64 reference on a row subset (every 13th row plus the last rows).
fn check_reference(c: &Case, out: &[u32]) {
    let s = c.shape;
    let vmax = c.v.iter().map(|x| x.to_f64().abs()).fold(0.0, f64::max);
    let mut worst = 0.0f64;
    let rows: Vec<usize> = (0..c.rows)
        .step_by(13)
        .chain(c.rows.saturating_sub(3)..c.rows)
        .collect();
    for &r in &rows {
        for h in 0..s.nh {
            let reference = c.reference(r, h);
            for d in 0..s.hd {
                let got = f32::from_bits(out[(r * s.nh + h) * s.hd + d]) as f64;
                assert!(
                    got.is_finite(),
                    "{} hist {} row {r} head {h}: non-finite",
                    s.name,
                    c.hist
                );
                worst = worst.max((got - reference[d]).abs());
            }
        }
    }
    eprintln!(
        "FA_TENSOR_REF shape={} hist={} rows={} max_abs_err={worst:.3e} bound={:.3e}",
        s.name,
        c.hist,
        c.rows,
        REF_BOUND * vmax
    );
    assert!(
        worst <= REF_BOUND * vmax,
        "{} hist {}: max |Δ| {worst:e} > bound",
        s.name,
        c.hist
    );
}

fn full_checks(shape: Shape, hist: usize, rows: usize, q_amp: f32, seed: u64) {
    let Some(mut g) = gpu(shape.hd as u32) else {
        return;
    };
    let c = Case::new(shape, hist, rows, q_amp, seed);
    let full = run(&mut g, &c, 0, rows, c.kv_pad);
    check_reference(&c, &full);
    // determinism
    assert!(
        run(&mut g, &c, 0, rows, c.kv_pad) == full,
        "{} hist {hist}: repeat differs",
        shape.name
    );
    // width invariance
    for (r0, n) in [
        (0, 1),
        (5, 7),
        (rows - 1, 1),
        (3, 31),
        (17, 32),
        (rows / 2, 58),
    ] {
        if r0 + n > rows {
            continue;
        }
        let part = run(&mut g, &c, r0, n, c.kv_pad);
        for i in 0..n {
            assert!(
                row_bits(&part, &c, i) == row_bits(&full, &c, r0 + i),
                "{} hist {hist}: row {} differs in a {n}-row dispatch at offset {r0}",
                shape.name,
                r0 + i
            );
        }
    }
    // chunking invariance: each chunk attends only its own causal key range
    let cuts = [0, rows / 3, rows * 2 / 3, rows];
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a == b {
            continue;
        }
        let kv_pad = tensor::kv_pad((hist + b) as u32) as usize;
        let part = run(&mut g, &c, a, b - a, kv_pad);
        for i in 0..b - a {
            assert!(
                row_bits(&part, &c, i) == row_bits(&full, &c, a + i),
                "{} hist {hist}: row {} differs when chunked [{a}, {b}) over {kv_pad} keys",
                shape.name,
                a + i
            );
        }
    }
    eprintln!(
        "FA_TENSOR_OK shape={} hist={hist} rows={rows} q_amp={q_amp}",
        shape.name
    );
}

#[test]
fn sliding_warm_lanes() {
    for (i, &(hist, rows)) in [(465, 56), (927, 288), (1023, 348)].iter().enumerate() {
        full_checks(SLIDING, hist, rows, 1.0, 0x5100 + i as u64);
    }
}

#[test]
fn global_warm_lanes() {
    for (i, &(hist, rows)) in [(465, 56), (927, 288), (1379, 348)].iter().enumerate() {
        full_checks(GLOBAL, hist, rows, 1.0, 0x6100 + i as u64);
    }
}

#[test]
fn cold_prefill_rows() {
    full_checks(SLIDING, 0, 458, 1.0, 0x7101);
    full_checks(GLOBAL, 0, 458, 1.0, 0x7102);
    full_checks(SLIDING, 0, 256, 1.0, 0x7103);
}

/// Large logits: the running max moves by more than 8 inside a tile row, so
/// the lazy rescale path runs.
#[test]
fn large_logits_exercise_the_rescale() {
    full_checks(SLIDING, 300, 100, 6.0, 0x8101);
    full_checks(GLOBAL, 300, 100, 6.0, 0x8102);
}

#[test]
fn refuses_unpadded_kv_and_wrong_dtypes() {
    let Some(mut g) = gpu(256) else { return };
    let c = Case::new(SLIDING, 10, 20, 1.0, 1);
    let q = upload(&g.device, &c.q, DType::F32);
    let k = upload(&g.device, &c.k, DType::F16);
    let mask = upload(&g.device, &c.mask(0, 20, c.kv_pad), DType::F16);
    let blk = g.device.alloc_buffer(64, DType::U8, vec![64]).unwrap();
    let out = g
        .device
        .alloc_buffer(20 * 16 * 256 * 4, DType::F32, vec![20 * 16 * 256])
        .unwrap();
    let mut p = FlashAttnPrefillTensorParams {
        rows: 20,
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 256,
        kv_pad: 30,
        q_row_stride: 16 * 256,
        q_head_stride: 256,
        k_row_stride: 256,
        k_head_stride: (c.kv_pad * 256) as u32,
        v_row_stride: 256,
        v_head_stride: (c.kv_pad * 256) as u32,
        scale: 1.0,
    };
    let mut enc = g.device.command_encoder().unwrap();
    let r = dispatch_flash_attn_prefill_tensor_f16(
        &mut enc, &mut g.reg, &g.device, &q, &k, &k, &mask, &blk, &out, &p,
    );
    assert!(r.is_err(), "kv_pad 30 must be refused");
    p.kv_pad = c.kv_pad as u32;
    let r = dispatch_flash_attn_prefill_tensor_f16(
        &mut enc, &mut g.reg, &g.device, &q, &q, &k, &mask, &blk, &out, &p,
    );
    assert!(r.is_err(), "F32 K must be refused");
    p.head_dim = 128;
    let r = dispatch_flash_attn_prefill_tensor_f16(
        &mut enc, &mut g.reg, &g.device, &q, &k, &k, &mask, &blk, &out, &p,
    );
    assert!(r.is_err(), "head_dim 128 must be refused");
}
