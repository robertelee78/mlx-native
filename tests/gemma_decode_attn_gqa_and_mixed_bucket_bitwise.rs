//! T-N8 (ADR-063 Q4-GEMMA-DECODE lever A): GQA-grouped and per-slot-bucket
//! batched decode flash are Class A against the stock batched vec-hybrid
//! kernel + reduce run per slot at that slot's own (nwg, nsg).
//!
//! Every comparison is on the raw u32 of each F32 word:
//! * the output O of every slot row against that slot's solo stock dispatch
//!   (`flash_attn_vec_hybrid_batched`, n_q = 1) and against the per-slot
//!   fallback today's caller runs for mixed buckets (`flash_attn_vec_hybrid`);
//! * every split-K partial and S/M word in `tmp` against the solo dispatch's;
//! * with one shared bucket, the whole `tmp` layout against the stock batched
//!   dispatch over all slots;
//! * sentinel words outside the output and `tmp` views stay untouched.
//!
//! Run on both metallib paths:
//! `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test gemma_decode_attn_gqa_and_mixed_bucket_bitwise -- --test-threads=1`
//! and the same with `MLX_PRECOMPILED_METALLIB=0`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use mlx_native::ops::flash_attn_vec_hybrid::{
    self as fa, FlashAttnVecTqHbParams, HybridSlotBucket,
};
use mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg;
use mlx_native::{
    DType, KernelArg, KernelPipelineOrigin, KernelRegistry, MTLSize, MlxBuffer, MlxDevice,
};

const SENTINEL: u32 = 0x7FA5_5A5A;
const GUARD: usize = 64;
const PHYS_SLOTS: u32 = 16;

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    dk: u32,
    nh: u32,
    nkv: u32,
    cap: u32,
    mask_type: u32,
    window: u32,
}

/// Gemma 4 sliding layer: 8 KV heads, 16 query heads, ring of 1024.
const SLIDING: Shape = Shape {
    name: "d256-sliding-ring",
    dk: 256,
    nh: 16,
    nkv: 8,
    cap: 1024,
    mask_type: 2,
    window: 1024,
};

/// Gemma 4 global layer: 2 KV heads, 16 query heads, linear storage.
const GLOBAL: Shape = Shape {
    name: "d512-global",
    dk: 512,
    nh: 16,
    nkv: 2,
    cap: 4160,
    mask_type: 1,
    window: 0,
};

/// Sliding window over linear (staging) storage: mask_type 3.
const SLIDING_LINEAR: Shape = Shape {
    name: "d256-sliding-linear",
    dk: 256,
    nh: 16,
    nkv: 8,
    cap: 1024,
    mask_type: 3,
    window: 512,
};

/// d256 with 8 query heads per KV head (16 q / 2 KV), ring of 1024: makes
/// the d256 G = 4 and G = 8 kernels compare instead of only being refused.
const SLIDING_GQA8: Shape = Shape {
    name: "d256-gqa8-ring",
    dk: 256,
    nh: 16,
    nkv: 2,
    cap: 1024,
    mask_type: 2,
    window: 1024,
};

struct Xor(u64);

impl Xor {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 16) as u32
    }
    fn signed(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
    }
}

/// Finite half bits: mostly |x| in [2^-5, 4), with ±0 and subnormals.
fn rand_half_bits(rng: &mut Xor) -> u16 {
    let r = rng.next_u32();
    let sign = (r & 1) << 15;
    let exp = if (r >> 1) & 63 == 0 {
        0
    } else {
        10 + ((r >> 7) % 7)
    };
    let mant = (r >> 10) & 0x3FF;
    (sign | (exp << 10) | mant) as u16
}

fn words_buffer(device: &MlxDevice, words: &[u32], dtype: DType) -> MlxBuffer {
    let mut b = device
        .alloc_buffer(words.len() * 4, dtype, vec![words.len()])
        .expect("alloc");
    b.as_mut_slice::<u32>().expect("map").copy_from_slice(words);
    b
}

fn sentinel_buffer(device: &MlxDevice, floats: usize) -> MlxBuffer {
    words_buffer(device, &vec![SENTINEL; floats.max(1)], DType::F32)
}

fn words(b: &MlxBuffer) -> Vec<u32> {
    b.as_slice::<u32>().expect("read").to_vec()
}

struct Kv {
    k: MlxBuffer,
    v: MlxBuffer,
    vn: MlxBuffer,
    v_f16: bool,
}

fn npp(shape: &Shape) -> u32 {
    if shape.dk == 512 {
        2
    } else {
        1
    }
}

fn make_kv(device: &MlxDevice, shape: &Shape, v_f16: bool, seed: u64) -> Kv {
    let mut rng = Xor::new(seed);
    let elems = (PHYS_SLOTS * shape.nkv * shape.cap * shape.dk) as usize;
    let mut k = device
        .alloc_buffer(elems * 2, DType::F16, vec![elems])
        .expect("k");
    for h in k.as_mut_slice::<u16>().unwrap() {
        *h = rand_half_bits(&mut rng);
    }
    let v = if v_f16 {
        let mut v = device
            .alloc_buffer(elems * 2, DType::F16, vec![elems])
            .expect("v");
        for h in v.as_mut_slice::<u16>().unwrap() {
            *h = rand_half_bits(&mut rng);
        }
        v
    } else {
        let mut v = device
            .alloc_buffer(elems, DType::U8, vec![elems])
            .expect("v");
        for b in v.as_mut_slice::<u8>().unwrap() {
            *b = rng.next_u32() as u8;
        }
        v
    };
    let n_norms = (PHYS_SLOTS * shape.nkv * shape.cap * npp(shape)) as usize;
    let mut vn = device
        .alloc_buffer(n_norms * 4, DType::F32, vec![n_norms])
        .expect("vn");
    for (i, x) in vn.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        *x = if i % 251 == 0 {
            0.0
        } else {
            0.25 + rng.signed().abs() * 1.5
        };
    }
    Kv { k, v, vn, v_f16 }
}

/// Query rows with ±0, f32 subnormals (→ half 0) and large magnitudes.
fn make_q(n: usize, shape: &Shape, seed: u64) -> Vec<f32> {
    let mut rng = Xor::new(seed);
    let specials = [
        0.0f32, -0.0, 1.0e-40, -3.0e-39, 24.0, -17.5, 6.0e-5, -6.0e-8,
    ];
    (0..n * (shape.nh * shape.dk) as usize)
        .map(|i| {
            if i % 61 == 0 {
                specials[(i / 61) % specials.len()]
            } else {
                rng.signed() * 1.5
            }
        })
        .collect()
}

fn ksl(shape: &Shape, sp: u32) -> u32 {
    if shape.mask_type == 2 {
        (sp + 1).min(shape.cap)
    } else {
        sp + 1
    }
}

fn ring_start(shape: &Shape, sp: u32) -> u32 {
    if shape.mask_type == 2 && ksl(shape, sp) >= shape.cap {
        (sp + 1) % shape.cap
    } else {
        0
    }
}

#[derive(Clone)]
struct Cell {
    shape: Shape,
    /// (physical slot, seq position) per query row.
    slots: Vec<(u32, u32)>,
    group: u32,
    cbits: u32,
    scale: f32,
    fuse_fwht: bool,
    /// `scale_factor_d512` (D = 512 V-norm divisor).
    sf_d512: f32,
    /// Explicit per-slot buckets; `None` = stock heuristics per slot.
    buckets: Option<Vec<HybridSlotBucket>>,
    q: Vec<f32>,
}

impl Cell {
    fn new(shape: Shape, slots: Vec<(u32, u32)>, group: u32, seed: u64) -> Self {
        let q = make_q(slots.len(), &shape, seed);
        Self {
            shape,
            slots,
            group,
            cbits: 8,
            scale: 1.0,
            fuse_fwht: false,
            sf_d512: 22.627417,
            buckets: None,
            q,
        }
    }

    fn buckets(&self) -> Vec<HybridSlotBucket> {
        self.buckets.clone().unwrap_or_else(|| {
            self.slots
                .iter()
                .map(|&(_, sp)| HybridSlotBucket::for_kv_seq_len(ksl(&self.shape, sp)))
                .collect()
        })
    }

    fn label(&self) -> String {
        format!(
            "{} G={} cbits={} fwht={} scale={} slots={:?} buckets={:?}",
            self.shape.name,
            self.group,
            self.cbits,
            self.fuse_fwht,
            self.scale,
            self.slots,
            self.buckets()
        )
    }

    fn params(&self, kv_seq_len: u32, ring_start: u32, nsg: u32) -> FlashAttnVecTqHbParams {
        FlashAttnVecTqHbParams {
            num_heads: self.shape.nh,
            num_kv_heads: self.shape.nkv,
            head_dim: self.shape.dk,
            kv_seq_len,
            kv_capacity: self.shape.cap,
            scale: self.scale,
            mask_type: self.shape.mask_type,
            sliding_window: self.shape.window,
            softcap: 0.0,
            ring_start,
            scale_factor_d512: self.sf_d512,
            codebook_bits: self.cbits,
            fuse_fwht_pre: self.fuse_fwht as u32,
            nsg,
        }
    }
}

struct Ctx {
    device: MlxDevice,
    registry: KernelRegistry,
    /// Stock split-K partial words seen with 0 < |x| < 1e-37 (underflow
    /// reach of the cells run so far).
    tiny_stock_partials: usize,
}

impl Ctx {
    fn new() -> Self {
        Self {
            device: MlxDevice::new().expect("Metal device"),
            registry: KernelRegistry::new(),
            tiny_stock_partials: 0,
        }
    }
}

/// Inputs bound at non-zero offsets (views), shared by candidate and refs.
struct Inputs {
    q: MlxBuffer,
    slot_ids: MlxBuffer,
    seq_pos: MlxBuffer,
}

impl Inputs {
    fn new(device: &MlxDevice, cell: &Cell) -> Self {
        let mut q_words = vec![SENTINEL; GUARD];
        q_words.extend(cell.q.iter().map(|x| x.to_bits()));
        let q_base = words_buffer(device, &q_words, DType::F32);
        let q = q_base.slice_view((GUARD * 4) as u64, cell.q.len());
        let ids: Vec<u32> = cell.slots.iter().map(|s| s.0).collect();
        let pos: Vec<u32> = cell.slots.iter().map(|s| s.1).collect();
        let mut id_words = vec![0xFFFF_FFFFu32; 4];
        id_words.extend(&ids);
        let mut pos_words = vec![0xFFFF_FFFFu32; 4];
        pos_words.extend(&pos);
        let slot_ids = words_buffer(device, &id_words, DType::U32).slice_view(16, ids.len());
        let seq_pos = words_buffer(device, &pos_words, DType::U32).slice_view(16, pos.len());
        Self {
            q,
            slot_ids,
            seq_pos,
        }
    }
}

struct Candidate {
    /// Whole output allocation (GUARD + rows + GUARD words).
    out: Vec<u32>,
    /// Whole tmp allocation (GUARD + tmp_floats + GUARD words).
    tmp: Vec<u32>,
    tmp_floats: usize,
}

fn run_candidate(ctx: &mut Ctx, cell: &Cell, kv: &Kv, inputs: &Inputs) -> Candidate {
    let shape = &cell.shape;
    let n = cell.slots.len();
    let row = (shape.nh * shape.dk) as usize;
    let buckets = cell.buckets();
    let tmp_floats =
        fa::flash_attn_vec_hybrid_batched_gqa_tmp_bytes(shape.nh, shape.dk, &buckets).unwrap() / 4;
    let out_base = sentinel_buffer(&ctx.device, 2 * GUARD + n * row);
    let tmp_base = sentinel_buffer(&ctx.device, 2 * GUARD + tmp_floats);
    let out = out_base.slice_view((GUARD * 4) as u64, n * row);
    let tmp = tmp_base.slice_view((GUARD * 4) as u64, tmp_floats);
    let max_ksl = cell
        .slots
        .iter()
        .map(|&(_, sp)| ksl(shape, sp))
        .max()
        .unwrap();
    let params = cell.params(max_ksl, 0, compute_nsg(max_ksl));
    let mut enc = ctx.device.command_encoder().unwrap();
    fa::flash_attn_vec_hybrid_batched_gqa(
        &mut enc,
        &mut ctx.registry,
        &ctx.device,
        &inputs.q,
        &kv.k,
        &kv.v,
        &kv.vn,
        &out,
        &tmp,
        &inputs.slot_ids,
        &inputs.seq_pos,
        &params,
        cell.group,
        &buckets,
    )
    .unwrap_or_else(|e| panic!("{}: candidate dispatch: {e}", cell.label()));
    enc.commit_and_wait().unwrap();
    Candidate {
        out: words(&out_base),
        tmp: words(&tmp_base),
        tmp_floats,
    }
}

/// GPU params of the stock kernels (byte layout of the shader struct).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StockGpuParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_seq_len: u32,
    kv_capacity: u32,
    scale: f32,
    mask_type: u32,
    sliding_window: u32,
    softcap: f32,
    nwg: u32,
    ring_start: u32,
    scale_factor_d512: f32,
    codebook_bits: u32,
    fuse_fwht_pre: u32,
    nsg: u32,
    n_queries: u32,
}

/// The stock batched kernel + stock reduce for ONE slot at an explicit
/// (nwg, nsg): the same encode `flash_attn_vec_hybrid_batched` performs,
/// with the bucket given instead of derived (buckets the heuristics never
/// pick, e.g. nwg = 1 or nsg = 2).
fn encode_stock_solo_explicit(
    ctx: &mut Ctx,
    enc: &mut mlx_native::CommandEncoder,
    cell: &Cell,
    kv: &Kv,
    q_i: &MlxBuffer,
    slot_i: &MlxBuffer,
    pos_i: &MlxBuffer,
    out_i: &MlxBuffer,
    tmp_i: &MlxBuffer,
    b: HybridSlotBucket,
) {
    let shape = &cell.shape;
    let p = StockGpuParams {
        n_heads: shape.nh,
        n_kv_heads: shape.nkv,
        head_dim: shape.dk,
        kv_seq_len: 1,
        kv_capacity: shape.cap,
        scale: cell.scale,
        mask_type: shape.mask_type,
        sliding_window: shape.window,
        softcap: 0.0,
        nwg: b.nwg,
        ring_start: 0,
        scale_factor_d512: cell.sf_d512,
        codebook_bits: cell.cbits,
        fuse_fwht_pre: cell.fuse_fwht as u32,
        nsg: b.nsg,
        n_queries: 1,
    };
    let name = if shape.dk == 256 {
        "flash_attn_vec_hybrid_batched_dk256"
    } else {
        "flash_attn_vec_hybrid_batched_dk512"
    };
    let pipeline = ctx
        .registry
        .get_pipeline_with_constants(
            name,
            ctx.device.metal_device(),
            &[],
            &[(50, cell.cbits as i32), (51, kv.v_f16 as i32)],
        )
        .unwrap();
    let pad = (shape.dk as usize).div_ceil(128) * 128;
    let shmem = (pad + b.nsg as usize * (128 + 2 * pad)) * 2;
    let dst = if b.nwg == 1 { out_i } else { tmp_i };
    enc.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(bytemuck::bytes_of(&p))),
            (1, KernelArg::Buffer(q_i)),
            (2, KernelArg::Buffer(&kv.k)),
            (3, KernelArg::Buffer(&kv.v)),
            (4, KernelArg::Buffer(&kv.vn)),
            (5, KernelArg::Buffer(dst)),
            (6, KernelArg::Buffer(slot_i)),
            (7, KernelArg::Buffer(pos_i)),
        ],
        &[(0, shmem as u64)],
        MTLSize::new(1, shape.nh as u64, b.nwg as u64),
        MTLSize::new(32, b.nsg as u64, 1),
    );
    if b.nwg > 1 {
        enc.memory_barrier();
        let reduce = if shape.dk == 256 {
            "flash_attn_vec_reduce_dk256"
        } else {
            "flash_attn_vec_reduce_dk512"
        };
        let pipeline = ctx
            .registry
            .get_pipeline(reduce, ctx.device.metal_device())
            .unwrap();
        let nrows: u32 = shape.nh;
        enc.encode_threadgroups_with_args(
            pipeline,
            &[
                (0, KernelArg::Bytes(bytemuck::bytes_of(&nrows))),
                (1, KernelArg::Buffer(tmp_i)),
                (2, KernelArg::Buffer(out_i)),
                (3, KernelArg::Bytes(bytemuck::bytes_of(&b.nwg))),
            ],
            MTLSize::new(shape.nh as u64, 1, 1),
            MTLSize::new(32 * b.nwg as u64, 1, 1),
        );
    }
}

/// Test-side derivation of the documented `tmp` layout: maximal runs of
/// consecutive slots sharing an NWG > 1, each run base 16-byte aligned, each
/// run the stock `nrows = run_len * heads` layout. Returns per-slot
/// (partials offset, S/M offset) in floats and the total.
fn expected_layout(
    shape: &Shape,
    buckets: &[HybridSlotBucket],
) -> (Vec<Option<(usize, usize)>>, usize) {
    let rows = shape.nh as usize;
    let dv = shape.dk as usize;
    let mut offs = vec![None; buckets.len()];
    let mut off = 0usize;
    let mut s = 0;
    while s < buckets.len() {
        let w = buckets[s].nwg;
        let mut e = s + 1;
        while e < buckets.len() && buckets[e].nwg == w {
            e += 1;
        }
        if w > 1 {
            let w = w as usize;
            off = off.div_ceil(4) * 4;
            let l = e - s;
            let part = l * rows * dv * w;
            for k in 0..l {
                offs[s + k] = Some((off + k * rows * dv * w, off + part + k * rows * 2 * w));
            }
            off += part + l * rows * 2 * w;
        }
        s = e;
    }
    (offs, off)
}

struct SoloRef {
    out: Vec<u32>,
    tmp: Vec<u32>,
}

/// Run every per-slot reference for `cell` and check the candidate bitwise.
/// Returns the candidate's output rows (without guards).
fn check_cell(ctx: &mut Ctx, cell: &Cell, kv: &Kv) -> Vec<u32> {
    let label = cell.label();
    let shape = cell.shape;
    let n = cell.slots.len();
    let row = (shape.nh * shape.dk) as usize;
    let buckets = cell.buckets();
    assert!(
        fa::flash_attn_vec_hybrid_batched_gqa_group_is_legal(
            shape.nh,
            shape.nkv,
            shape.dk,
            cell.group,
            buckets.iter().map(|b| b.nsg).max().unwrap()
        ),
        "{label}: illegal group in a positive cell"
    );
    let inputs = Inputs::new(&ctx.device, cell);
    let cand = run_candidate(ctx, cell, kv, &inputs);

    let (layout, total) = expected_layout(&shape, &buckets);
    assert_eq!(total, cand.tmp_floats, "{label}: tmp size");

    // Per-slot stock references.
    let mut solos = Vec::with_capacity(n);
    let mut fallbacks = Vec::with_capacity(n);
    let mut enc = ctx.device.command_encoder().unwrap();
    let mut keep = Vec::new();
    for (i, (&(phys, sp), &b)) in cell.slots.iter().zip(&buckets).enumerate() {
        let q_i = inputs.q.slice_view((i * row * 4) as u64, row);
        let slot_i = inputs.slot_ids.slice_view((i * 4) as u64, 1);
        let pos_i = inputs.seq_pos.slice_view((i * 4) as u64, 1);
        let out_i = sentinel_buffer(&ctx.device, row);
        let tmp_i = sentinel_buffer(
            &ctx.device,
            row * b.nwg as usize + shape.nh as usize * 2 * b.nwg as usize,
        );
        let k_i = ksl(&shape, sp);
        let heuristic = HybridSlotBucket::for_kv_seq_len(k_i);
        if b == heuristic {
            // Exactly the stock public call at this slot's own bucket.
            let p = cell.params(k_i, 0, b.nsg);
            fa::flash_attn_vec_hybrid_batched(
                &mut enc,
                &mut ctx.registry,
                &ctx.device,
                1,
                &q_i,
                &kv.k,
                &kv.v,
                &kv.vn,
                &out_i,
                &tmp_i,
                &slot_i,
                &pos_i,
                &p,
            )
            .unwrap();
            // Today's per-slot fallback (non-batched kernel on slot views).
            // Its window applies only to mask_type 2, so mask_type 3 has no
            // fallback twin.
            if shape.mask_type != 3 {
                let kv_elems = (shape.nkv * shape.cap * shape.dk) as usize;
                let k_view = kv.k.slice_view(phys as u64 * kv_elems as u64 * 2, kv_elems);
                let v_size = if kv.v_f16 { 2 } else { 1 };
                let v_view =
                    kv.v.slice_view(phys as u64 * kv_elems as u64 * v_size, kv_elems);
                let n_norms = (shape.nkv * shape.cap * npp(&shape)) as usize;
                let vn_view = kv.vn.slice_view(phys as u64 * n_norms as u64 * 4, n_norms);
                let fb_out = sentinel_buffer(&ctx.device, row);
                let fb_tmp =
                    sentinel_buffer(&ctx.device, fa::tmp_buffer_bytes(shape.nh, shape.dk) / 4);
                let pf = cell.params(k_i, ring_start(&shape, sp), compute_nsg(k_i));
                fa::flash_attn_vec_hybrid(
                    &mut enc,
                    &mut ctx.registry,
                    &ctx.device,
                    &q_i,
                    &k_view,
                    &v_view,
                    &vn_view,
                    &fb_out,
                    &fb_tmp,
                    &pf,
                )
                .unwrap();
                fallbacks.push(Some(fb_out));
                keep.push((k_view, v_view, vn_view, fb_tmp));
            } else {
                fallbacks.push(None);
            }
        } else {
            encode_stock_solo_explicit(
                ctx, &mut enc, cell, kv, &q_i, &slot_i, &pos_i, &out_i, &tmp_i, b,
            );
            fallbacks.push(None);
        }
        solos.push((out_i, tmp_i));
        keep.push((q_i, slot_i, pos_i, sentinel_buffer(&ctx.device, 1)));
    }
    enc.commit_and_wait().unwrap();
    let solos: Vec<SoloRef> = solos
        .iter()
        .map(|(o, t)| SoloRef {
            out: words(o),
            tmp: words(t),
        })
        .collect();
    for (i, b) in buckets.iter().enumerate() {
        if layout[i].is_some() {
            let part = row * b.nwg as usize;
            ctx.tiny_stock_partials += solos[i].tmp[..part]
                .iter()
                .filter(|&&w| {
                    let x = f32::from_bits(w);
                    x != 0.0 && x.abs() < 1.0e-37
                })
                .count();
        }
    }

    // Partials and S/M against each slot's solo tmp (S/M first: they carry
    // only the QK/softmax path); untouched words stay sentinel.
    let mut written = vec![false; cand.tmp_floats];
    for (i, b) in buckets.iter().enumerate() {
        let Some((part_off, sm_off)) = layout[i] else {
            continue;
        };
        let w = b.nwg as usize;
        let part = row * w;
        let sm = shape.nh as usize * 2 * w;
        let got_part = &cand.tmp[GUARD + part_off..GUARD + part_off + part];
        let got_sm = &cand.tmp[GUARD + sm_off..GUARD + sm_off + sm];
        for (j, (&g, &e)) in got_sm
            .iter()
            .zip(&solos[i].tmp[part..part + sm])
            .enumerate()
        {
            assert_eq!(
                g, e,
                "{label}: tmp S/M slot {i} word {j}: {g:#010x} != {e:#010x}"
            );
        }
        for (j, (&g, &e)) in got_part.iter().zip(&solos[i].tmp[..part]).enumerate() {
            assert_eq!(
                g, e,
                "{label}: tmp partial slot {i} word {j}: {g:#010x} != {e:#010x}"
            );
        }
        written[part_off..part_off + part]
            .iter_mut()
            .for_each(|x| *x = true);
        written[sm_off..sm_off + sm]
            .iter_mut()
            .for_each(|x| *x = true);
    }
    for (j, &wr) in written.iter().enumerate() {
        if !wr {
            assert_eq!(
                cand.tmp[GUARD + j],
                SENTINEL,
                "{label}: tmp word {j} outside every slot region written"
            );
        }
    }
    assert!(
        cand.tmp[..GUARD].iter().all(|&w| w == SENTINEL),
        "{label}: tmp guard before"
    );
    assert!(
        cand.tmp[GUARD + cand.tmp_floats..]
            .iter()
            .all(|&w| w == SENTINEL),
        "{label}: tmp guard after"
    );
    // O rows.
    for i in 0..n {
        let got = &cand.out[GUARD + i * row..GUARD + (i + 1) * row];
        for (j, (&g, &e)) in got.iter().zip(&solos[i].out).enumerate() {
            assert_eq!(
                g, e,
                "{label}: O slot {i} word {j} (head {}): candidate {g:#010x} != solo stock {e:#010x}",
                j / shape.dk as usize
            );
        }
        if let Some(fb) = &fallbacks[i] {
            let fb = words(fb);
            assert_eq!(
                got,
                &fb[..],
                "{label}: O slot {i} differs from the per-slot fallback"
            );
        }
    }
    // Guards around O.
    assert!(
        cand.out[..GUARD].iter().all(|&w| w == SENTINEL),
        "{label}: O guard before"
    );
    assert!(
        cand.out[GUARD + n * row..].iter().all(|&w| w == SENTINEL),
        "{label}: O guard after"
    );

    // One shared heuristic bucket: the whole tmp equals the stock batched
    // dispatch's (same layout, same single reduce).
    let uniform = buckets.windows(2).all(|w| w[0] == w[1]);
    if uniform && cell.buckets.is_none() && buckets[0].nwg > 1 {
        let max_ksl = cell
            .slots
            .iter()
            .map(|&(_, sp)| ksl(&shape, sp))
            .max()
            .unwrap();
        let p = cell.params(max_ksl, 0, buckets[0].nsg);
        let out_b = sentinel_buffer(&ctx.device, n * row);
        let tmp_b = sentinel_buffer(&ctx.device, cand.tmp_floats);
        let mut enc = ctx.device.command_encoder().unwrap();
        fa::flash_attn_vec_hybrid_batched(
            &mut enc,
            &mut ctx.registry,
            &ctx.device,
            n as u32,
            &inputs.q,
            &kv.k,
            &kv.v,
            &kv.vn,
            &out_b,
            &tmp_b,
            &inputs.slot_ids,
            &inputs.seq_pos,
            &p,
        )
        .unwrap();
        enc.commit_and_wait().unwrap();
        assert_eq!(
            &cand.out[GUARD..GUARD + n * row],
            &words(&out_b)[..],
            "{label}: O differs from the stock batched dispatch"
        );
        assert_eq!(
            &cand.tmp[GUARD..GUARD + cand.tmp_floats],
            &words(&tmp_b)[..],
            "{label}: tmp differs from the stock batched dispatch"
        );
    }
    drop(keep);
    cand.out[GUARD..GUARD + n * row].to_vec()
}

fn legal_groups(shape: &Shape, nsg_max: u32) -> Vec<u32> {
    fa::FLASH_ATTN_VEC_HYBRID_GQA_GROUPS
        .iter()
        .copied()
        .filter(|&g| {
            fa::flash_attn_vec_hybrid_batched_gqa_group_is_legal(
                shape.nh, shape.nkv, shape.dk, g, nsg_max,
            )
        })
        .collect()
}

fn nsg_max_of(cell: &Cell) -> u32 {
    cell.buckets().iter().map(|b| b.nsg).max().unwrap()
}

/// Run `base` once per legal group size (and check the illegal ones are
/// refused at the API).
fn check_all_groups(ctx: &mut Ctx, base: &Cell, kv: &Kv) {
    let nsg_max = nsg_max_of(base);
    let legal = legal_groups(&base.shape, nsg_max);
    assert!(legal.contains(&1));
    for g in fa::FLASH_ATTN_VEC_HYBRID_GQA_GROUPS {
        let mut cell = base.clone();
        cell.group = g;
        if legal.contains(&g) {
            check_cell(ctx, &cell, kv);
        } else {
            assert_illegal(ctx, &cell, kv);
        }
    }
}

/// `check_all_groups`, asserting every group in `required` is legal (and so
/// compared bitwise) for `base`.
fn check_all_groups_requiring(ctx: &mut Ctx, base: &Cell, kv: &Kv, required: &[u32]) {
    let legal = legal_groups(&base.shape, nsg_max_of(base));
    for g in required {
        assert!(legal.contains(g), "{}: G={g} not legal ({legal:?})", base.label());
    }
    check_all_groups(ctx, base, kv);
}

fn assert_illegal(ctx: &mut Ctx, cell: &Cell, kv: &Kv) {
    let inputs = Inputs::new(&ctx.device, cell);
    let shape = &cell.shape;
    let n = cell.slots.len();
    let out = sentinel_buffer(&ctx.device, n * (shape.nh * shape.dk) as usize);
    let tmp = sentinel_buffer(
        &ctx.device,
        n * fa::tmp_buffer_bytes(shape.nh, shape.dk) / 4,
    );
    let max_ksl = cell
        .slots
        .iter()
        .map(|&(_, sp)| ksl(shape, sp))
        .max()
        .unwrap();
    let params = cell.params(max_ksl, 0, compute_nsg(max_ksl));
    let mut enc = ctx.device.command_encoder().unwrap();
    let r = fa::flash_attn_vec_hybrid_batched_gqa(
        &mut enc,
        &mut ctx.registry,
        &ctx.device,
        &inputs.q,
        &kv.k,
        &kv.v,
        &kv.vn,
        &out,
        &tmp,
        &inputs.slot_ids,
        &inputs.seq_pos,
        &params,
        cell.group,
        &cell.buckets(),
    );
    assert!(r.is_err(), "{}: illegal arguments accepted", cell.label());
    enc.commit_and_wait().unwrap();
    assert!(
        words(&out).iter().all(|&w| w == SENTINEL),
        "{}: refused call wrote O",
        cell.label()
    );
}

fn uniform_slots(sp: u32, n: usize) -> Vec<(u32, u32)> {
    // Distinct, scrambled physical slots.
    [3u32, 0, 7, 12, 9, 15, 1, 5, 14, 2, 11, 6, 13, 4, 10, 8]
        .iter()
        .take(n)
        .map(|&p| (p, sp))
        .collect()
}

#[test]
fn sliding_d256_kv_lengths_ring_wrap_and_groups_bitwise() {
    let mut ctx = Ctx::new();
    let kv = make_kv(&ctx.device, &SLIDING, false, 0x5117_D256);
    // Seq positions: every nwg edge below the ring size, then ring wrap at
    // and past capacity (window 1024), up to 4096 decoded positions.
    let sps = [
        0u32, 30, 31, 32, 510, 511, 512, 1022, 1023, 1024, 1025, 1500, 2047, 2048, 4095, 4096,
    ];
    for (k, &sp) in sps.iter().enumerate() {
        let base = Cell::new(SLIDING, uniform_slots(sp, 4), 1, 0xA000 + k as u64);
        check_all_groups(&mut ctx, &base, &kv);
    }
}

#[test]
fn global_d512_kv_lengths_and_groups_bitwise() {
    let mut ctx = Ctx::new();
    let kv = make_kv(&ctx.device, &GLOBAL, false, 0x610B_D512);
    // KV lengths 1, 31, 32, 33, 511, 512, 513 (nwg edge), 1024, 1025 (nsg
    // edge), 2048, 4096, 4097.
    let sps = [
        0u32, 30, 31, 32, 510, 511, 512, 1023, 1024, 2047, 4095, 4096,
    ];
    for (k, &sp) in sps.iter().enumerate() {
        let base = Cell::new(GLOBAL, uniform_slots(sp, 4), 1, 0xB000 + k as u64);
        check_all_groups(&mut ctx, &base, &kv);
    }
}

#[test]
fn mixed_buckets_match_per_slot_solo_dispatches() {
    let mut ctx = Ctx::new();
    let kv_s = make_kv(&ctx.device, &SLIDING, false, 0x5117_0001);
    let kv_g = make_kv(&ctx.device, &GLOBAL, false, 0x610B_0001);

    // Sliding: nwg 16/32 interleaved, ring-wrapped slots included.
    let cell = Cell::new(
        SLIDING,
        vec![(4, 99), (0, 700), (9, 5000), (15, 19), (2, 511), (11, 512)],
        1,
        0xC001,
    );
    check_all_groups(&mut ctx, &cell, &kv_s);

    // Global: nwg 16/32 and nsg 1/4 interleaved.
    let cell = Cell::new(
        GLOBAL,
        vec![
            (5, 99),
            (1, 600),
            (12, 1500),
            (3, 3000),
            (8, 0),
            (0, 512),
            (6, 1023),
            (13, 1024),
            (10, 4095),
        ],
        1,
        0xC002,
    );
    check_all_groups(&mut ctx, &cell, &kv_g);

    // Global with nsg_max = 1: G = 8 fits threadgroup memory.
    let cell = Cell::new(
        GLOBAL,
        vec![(5, 99), (1, 600), (12, 0), (3, 1000)],
        1,
        0xC003,
    );
    check_all_groups(&mut ctx, &cell, &kv_g);

    // Explicit buckets the heuristics never pick: nwg 1 (direct output),
    // odd nwg (run-base alignment), nsg 2, and inactive SIMD groups.
    let b = |nwg, nsg| HybridSlotBucket { nwg, nsg };
    let mut cell = Cell::new(
        GLOBAL,
        vec![(5, 99), (1, 600), (12, 1500), (3, 3000), (7, 40)],
        1,
        0xC004,
    );
    cell.buckets = Some(vec![b(1, 1), b(8, 2), b(32, 4), b(3, 2), b(3, 1)]);
    check_all_groups(&mut ctx, &cell, &kv_g);
    let mut cell = Cell::new(
        SLIDING,
        vec![(2, 1500), (6, 64), (14, 333), (0, 900)],
        1,
        0xC005,
    );
    cell.buckets = Some(vec![b(1, 2), b(16, 4), b(5, 1), b(32, 2)]);
    check_all_groups(&mut ctx, &cell, &kv_s);
}

#[test]
fn n_q_1_through_16_bitwise() {
    let mut ctx = Ctx::new();
    for (shape, max_sp) in [(SLIDING, 5000u32), (GLOBAL, 4096u32)] {
        let kv = make_kv(&ctx.device, &shape, false, 0x0016_0000 ^ shape.dk as u64);
        let mut rng = Xor::new(0x9E37_79B9 ^ shape.dk as u64);
        for n in 1..=16usize {
            let slots: Vec<(u32, u32)> = uniform_slots(0, 16)
                .into_iter()
                .take(n)
                .map(|(p, _)| (p, rng.next_u32() % (max_sp + 1)))
                .collect();
            let base = Cell::new(shape, slots, 1, 0xD000 + n as u64);
            let nsg_max = nsg_max_of(&base);
            let picked = fa::flash_attn_vec_hybrid_batched_gqa_pick_group(
                shape.nh, shape.nkv, shape.dk, nsg_max,
            );
            assert_eq!(picked, *legal_groups(&shape, nsg_max).last().unwrap());
            for g in [1, picked] {
                let mut cell = base.clone();
                cell.group = g;
                check_cell(&mut ctx, &cell, &kv);
            }
        }
    }
}

#[test]
fn codebooks_f16_v_fused_fwht_and_scale_bitwise() {
    let mut ctx = Ctx::new();
    for shape in [SLIDING, GLOBAL] {
        let kv = make_kv(&ctx.device, &shape, false, 0xCB00 ^ shape.dk as u64);
        let kv16 = make_kv(&ctx.device, &shape, true, 0xF160 ^ shape.dk as u64);
        let slots = vec![(1, 40), (8, 777), (3, 1200), (10, 2500)];
        for cbits in [5, 6] {
            let mut cell = Cell::new(shape, slots.clone(), 1, 0xE000 + cbits as u64);
            cell.cbits = cbits;
            check_all_groups(&mut ctx, &cell, &kv);
        }
        let cell = Cell::new(shape, slots.clone(), 1, 0xE100);
        check_all_groups(&mut ctx, &cell, &kv16);
        let mut cell = Cell::new(shape, slots.clone(), 1, 0xE200);
        cell.fuse_fwht = true;
        check_all_groups(&mut ctx, &cell, &kv);
        let mut cell = Cell::new(shape, slots.clone(), 1, 0xE300);
        cell.scale = 0.0625;
        check_all_groups(&mut ctx, &cell, &kv);
        // D = 512 norm divisor: non-powers of two expose the association
        // of `norm / sf * w`; a power of two must match as well.
        for (k, sf) in [3.0f32, 16.0, 0.7071068].into_iter().enumerate() {
            let mut cell = Cell::new(shape, slots.clone(), 1, 0xE400 + k as u64);
            cell.sf_d512 = sf;
            check_all_groups(&mut ctx, &cell, &kv);
        }
    }
}

#[test]
fn sliding_linear_mask_type3_bitwise() {
    let mut ctx = Ctx::new();
    let kv = make_kv(&ctx.device, &SLIDING_LINEAR, false, 0x5117_0003);
    let cell = Cell::new(
        SLIDING_LINEAR,
        vec![(0, 0), (5, 300), (9, 511), (2, 512), (7, 1023)],
        1,
        0xF003,
    );
    check_all_groups(&mut ctx, &cell, &kv);
}

/// Underflow KV: K is random finite f16 bit patterns (extreme scores, very
/// peaked softmax, weights down to FLT_MIN); V norms are 0 one in 251 and
/// otherwise log-uniform in [1e-30, 1e5].
fn make_kv_underflow(device: &MlxDevice, shape: &Shape, v_f16: bool, seed: u64) -> Kv {
    let mut rng = Xor::new(seed);
    let elems = (PHYS_SLOTS * shape.nkv * shape.cap * shape.dk) as usize;
    let mut k = device
        .alloc_buffer(elems * 2, DType::F16, vec![elems])
        .expect("k");
    for h in k.as_mut_slice::<u16>().unwrap() {
        *h = loop {
            let bits = rng.next_u32() as u16;
            if (bits >> 10) & 0x1F != 0x1F {
                break bits;
            }
        };
    }
    let v = if v_f16 {
        let mut v = device
            .alloc_buffer(elems * 2, DType::F16, vec![elems])
            .expect("v");
        for h in v.as_mut_slice::<u16>().unwrap() {
            *h = rand_half_bits(&mut rng);
        }
        v
    } else {
        let mut v = device
            .alloc_buffer(elems, DType::U8, vec![elems])
            .expect("v");
        for b in v.as_mut_slice::<u8>().unwrap() {
            *b = rng.next_u32() as u8;
        }
        v
    };
    let n_norms = (PHYS_SLOTS * shape.nkv * shape.cap * npp(shape)) as usize;
    let mut vn = device
        .alloc_buffer(n_norms * 4, DType::F32, vec![n_norms])
        .expect("vn");
    for (i, x) in vn.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        *x = if i % 251 == 0 {
            0.0
        } else {
            10f64.powf(-30.0 + 35.0 * rng.signed().abs() as f64) as f32
        };
    }
    Kv { k, v, vn, v_f16 }
}

/// KV lengths of the underflow cells (seq position = length - 1).
const UNDERFLOW_KV_LENS: [u32; 4] = [783, 905, 1024, 2048];

/// Every legal G at KV lengths 783/905/1024/2048, n_q 1/4/16, uniform and
/// mixed buckets; then require the cells to have reached underflow.
fn check_underflow_sweep(ctx: &mut Ctx, shape: Shape, kv: &Kv, tweak: &dyn Fn(&mut Cell), seed: u64) {
    let before = ctx.tiny_stock_partials;
    let mut k = 0u64;
    for &len in &UNDERFLOW_KV_LENS {
        for n in [1usize, 4, 16] {
            let mut cell = Cell::new(shape, uniform_slots(len - 1, n), 1, seed + k);
            k += 1;
            tweak(&mut cell);
            check_all_groups(ctx, &cell, kv);
        }
    }
    for n in [4usize, 16] {
        let slots: Vec<(u32, u32)> = uniform_slots(0, n)
            .into_iter()
            .enumerate()
            .map(|(i, (p, _))| (p, UNDERFLOW_KV_LENS[(i * 3 + 1) % 4] - 1))
            .collect();
        // Heuristic buckets (mixed at d512, where 2048 crosses the nsg edge;
        // the d256 ring caps every length at 1024), then explicit mixed
        // buckets: nwg 1 (direct O), odd nwg, nsg 1/2/4.
        let b = |nwg, nsg| HybridSlotBucket { nwg, nsg };
        let explicit = [b(32, 1), b(16, 2), b(1, 1), b(8, 4), b(3, 2), b(32, 4)];
        let mut cell = Cell::new(shape, slots, 1, seed + k);
        k += 1;
        tweak(&mut cell);
        check_all_groups(ctx, &cell, kv);
        cell.buckets = Some((0..n).map(|i| explicit[i % explicit.len()]).collect());
        check_all_groups(ctx, &cell, kv);
    }
    assert!(
        ctx.tiny_stock_partials > before,
        "{} seed {seed:#x}: underflow cells vacuous (no stock partial word with 0 < |x| < 1e-37)",
        shape.name
    );
}

#[test]
fn underflow_d256_sliding_bitwise() {
    let mut ctx = Ctx::new();
    for (shape, seed) in [(SLIDING, 0x0F10_0256u64), (SLIDING_GQA8, 0x0F10_8256)] {
        let kv = make_kv_underflow(&ctx.device, &shape, false, seed);
        check_underflow_sweep(&mut ctx, shape, &kv, &|_| {}, seed);
    }
}

#[test]
fn underflow_d512_global_sf_and_f16_v_bitwise() {
    let mut ctx = Ctx::new();
    let kv = make_kv_underflow(&ctx.device, &GLOBAL, false, 0x0F10_0512);
    // Non-power-of-two norm divisors (the default 22.627417 is one too).
    for (k, sf) in [22.627417f32, 3.0, 0.7071068].into_iter().enumerate() {
        check_underflow_sweep(
            &mut ctx,
            GLOBAL,
            &kv,
            &move |c: &mut Cell| c.sf_d512 = sf,
            0x0F20_0000 + ((k as u64) << 8),
        );
    }
    let kv16 = make_kv_underflow(&ctx.device, &GLOBAL, true, 0x0F16_0512);
    check_underflow_sweep(&mut ctx, GLOBAL, &kv16, &|_| {}, 0x0F30_0000);
}

#[test]
fn d256_eight_heads_per_kv_head_groups_4_and_8_bitwise() {
    let mut ctx = Ctx::new();
    let kv = make_kv(&ctx.device, &SLIDING_GQA8, false, 0x5117_6A08);
    for (k, &sp) in [0u32, 31, 511, 512, 1023, 1024, 2047, 4096].iter().enumerate() {
        let base = Cell::new(SLIDING_GQA8, uniform_slots(sp, 4), 1, 0x6A00 + k as u64);
        check_all_groups_requiring(&mut ctx, &base, &kv, &[1, 2, 4, 8]);
    }
    let base = Cell::new(
        SLIDING_GQA8,
        vec![(4, 99), (0, 700), (9, 5000), (15, 19), (2, 511), (11, 512)],
        1,
        0x6A10,
    );
    check_all_groups_requiring(&mut ctx, &base, &kv, &[1, 2, 4, 8]);
}

#[test]
fn d512_variants_with_nsg_max_1_groups_4_and_8_bitwise() {
    let mut ctx = Ctx::new();
    let kv = make_kv(&ctx.device, &GLOBAL, false, 0xCB51_2001);
    let kv16 = make_kv(&ctx.device, &GLOBAL, true, 0xF165_1201);
    // Every slot at KV <= 1024: nsg 1 everywhere.
    let slots = vec![(1, 40), (8, 777), (3, 1023), (10, 300)];
    let req = [1, 2, 4, 8];
    let check = |ctx: &mut Ctx, cell: &Cell, kv: &Kv| {
        assert_eq!(nsg_max_of(cell), 1, "{}", cell.label());
        check_all_groups_requiring(ctx, cell, kv, &req);
    };
    for cbits in [5, 6] {
        let mut cell = Cell::new(GLOBAL, slots.clone(), 1, 0xE500 + cbits as u64);
        cell.cbits = cbits;
        check(&mut ctx, &cell, &kv);
    }
    let cell = Cell::new(GLOBAL, slots.clone(), 1, 0xE600);
    check(&mut ctx, &cell, &kv16);
    let mut cell = Cell::new(GLOBAL, slots.clone(), 1, 0xE700);
    cell.fuse_fwht = true;
    check(&mut ctx, &cell, &kv);
    let mut cell = Cell::new(GLOBAL, slots.clone(), 1, 0xE800);
    cell.scale = 0.0625;
    check(&mut ctx, &cell, &kv);
    for (k, sf) in [3.0f32, 16.0, 0.7071068].into_iter().enumerate() {
        let mut cell = Cell::new(GLOBAL, slots.clone(), 1, 0xE900 + k as u64);
        cell.sf_d512 = sf;
        check(&mut ctx, &cell, &kv);
    }
}

#[test]
fn d512_explicit_nsg_2_group_4_cross_simd_merge_bitwise() {
    let mut ctx = Ctx::new();
    let kv = make_kv(&ctx.device, &GLOBAL, false, 0x610B_0E02);
    let b = |nwg, nsg| HybridSlotBucket { nwg, nsg };
    let mut cell = Cell::new(
        GLOBAL,
        vec![(5, 99), (1, 600), (12, 1500), (3, 3000), (7, 40)],
        1,
        0xC104,
    );
    cell.buckets = Some(vec![b(1, 2), b(8, 2), b(32, 2), b(3, 2), b(16, 2)]);
    check_all_groups_requiring(&mut ctx, &cell, &kv, &[1, 2, 4]);
    let mut cell = Cell::new(GLOBAL, vec![(2, 4095), (9, 1024), (6, 0)], 1, 0xC105);
    cell.buckets = Some(vec![b(32, 2), b(16, 1), b(1, 2)]);
    check_all_groups_requiring(&mut ctx, &cell, &kv, &[1, 2, 4]);
}

#[test]
fn slot_and_head_permutation_and_nan_isolation() {
    let mut ctx = Ctx::new();
    for shape in [SLIDING, GLOBAL] {
        let kv = make_kv(&ctx.device, &shape, false, 0x9E20 ^ shape.dk as u64);
        let row = (shape.nh * shape.dk) as usize;
        let dk = shape.dk as usize;
        let slots = vec![(2, 50), (9, 900), (4, 2000), (13, 513), (0, 3500)];
        let base = Cell::new(shape, slots.clone(), 1, 0x1234);
        let nsg_max = nsg_max_of(&base);
        for g in legal_groups(&shape, nsg_max) {
            let mut cell = base.clone();
            cell.group = g;
            let out = check_cell(&mut ctx, &cell, &kv);

            // Slot permutation: reversed rows, reversed slots.
            let mut rev = cell.clone();
            rev.slots.reverse();
            rev.q = (0..slots.len())
                .rev()
                .flat_map(|i| cell.q[i * row..(i + 1) * row].to_vec())
                .collect();
            let out_rev = check_cell(&mut ctx, &rev, &kv);
            let n = slots.len();
            for i in 0..n {
                assert_eq!(
                    &out_rev[i * row..(i + 1) * row],
                    &out[(n - 1 - i) * row..(n - i) * row],
                    "{}: slot permutation row {i}",
                    cell.label()
                );
            }

            // Head permutation within each KV group (heads sharing a KV head).
            let hpk = (shape.nh / shape.nkv) as usize;
            let perm = |h: usize| (h / hpk) * hpk + (hpk - 1 - h % hpk);
            let mut hp = cell.clone();
            hp.q = (0..n)
                .flat_map(|i| {
                    (0..shape.nh as usize)
                        .flat_map(|h| {
                            let src = i * row + perm(h) * dk;
                            cell.q[src..src + dk].to_vec()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let out_hp = check_cell(&mut ctx, &hp, &kv);
            for i in 0..n {
                for h in 0..shape.nh as usize {
                    let a = i * row + h * dk;
                    let b = i * row + perm(h) * dk;
                    assert_eq!(
                        &out_hp[a..a + dk],
                        &out[b..b + dk],
                        "{}: head permutation slot {i} head {h}",
                        cell.label()
                    );
                }
            }

            // NaN in one slot's query: that slot matches its own solo NaN
            // reference (check_cell), every other slot is bit-identical to
            // the clean run.
            let mut nan = cell.clone();
            nan.q[row + 5 * dk + 7] = f32::NAN;
            let out_nan = check_cell(&mut ctx, &nan, &kv);
            for i in (0..n).filter(|&i| i != 1) {
                assert_eq!(
                    &out_nan[i * row..(i + 1) * row],
                    &out[i * row..(i + 1) * row],
                    "{}: NaN in slot 1 perturbed slot {i}",
                    cell.label()
                );
            }
        }
    }
}

#[test]
fn pipelines_resolve_on_the_selected_metallib_path() {
    let precompiled = !matches!(
        std::env::var("MLX_PRECOMPILED_METALLIB").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    );
    let fcv = !matches!(
        std::env::var("MLX_PRECOMPILED_METALLIB_FCV").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    );
    let expected = if precompiled && fcv {
        KernelPipelineOrigin::PrecompiledMetallib
    } else {
        KernelPipelineOrigin::RuntimeSource
    };
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let mut names = vec![
        "flash_attn_vec_hybrid_batched_dk256".to_string(),
        "flash_attn_vec_hybrid_batched_dk512".to_string(),
    ];
    for dk in [256, 512] {
        for g in fa::FLASH_ATTN_VEC_HYBRID_GQA_GROUPS {
            names.push(format!("flash_attn_vec_hybrid_batched_gqa{g}_dk{dk}"));
        }
    }
    for name in &names {
        let label = registry
            .get_pipeline_with_constants(name, device.metal_device(), &[], &[(50, 8), (51, 0)])
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .label()
            .to_owned();
        let identity = registry.pipeline_identity(&label).unwrap();
        assert_eq!(identity.origin, expected, "{name}: pipeline origin");
    }
    eprintln!("metallib path: {expected:?}");
}

#[test]
fn rejects_illegal_arguments() {
    let mut ctx = Ctx::new();
    let kv = make_kv(&ctx.device, &SLIDING, false, 0xBAD0);
    let b = |nwg, nsg| HybridSlotBucket { nwg, nsg };
    let slots = vec![(0, 10), (1, 20)];
    // G must divide heads_per_kv (2 on sliding layers).
    for g in [3, 4, 8, 16] {
        let cell = Cell::new(SLIDING, slots.clone(), g, 1);
        assert_illegal(&mut ctx, &cell, &kv);
    }
    // Bucket bounds.
    for bad in [
        vec![b(0, 1), b(16, 1)],
        vec![b(16, 3), b(16, 1)],
        vec![b(33, 1), b(16, 1)],
        vec![b(16, 8), b(16, 1)],
    ] {
        let mut cell = Cell::new(SLIDING, slots.clone(), 1, 2);
        cell.buckets = Some(bad);
        assert_illegal(&mut ctx, &cell, &kv);
    }
    // One bucket per query row.
    let mut cell = Cell::new(SLIDING, slots.clone(), 1, 3);
    cell.buckets = Some(vec![]);
    assert_illegal(&mut ctx, &cell, &kv);
    // G = 8 at d512 only fits threadgroup memory at nsg 1.
    assert!(fa::flash_attn_vec_hybrid_batched_gqa_group_is_legal(
        16, 2, 512, 8, 1
    ));
    assert!(!fa::flash_attn_vec_hybrid_batched_gqa_group_is_legal(
        16, 2, 512, 8, 4
    ));
    assert_eq!(
        fa::flash_attn_vec_hybrid_batched_gqa_pick_group(16, 2, 512, 4),
        2
    );
    assert_eq!(
        fa::flash_attn_vec_hybrid_batched_gqa_pick_group(16, 8, 256, 4),
        2
    );
    // tmp too small is refused.
    let cell = Cell::new(SLIDING, slots.clone(), 1, 4);
    let inputs = Inputs::new(&ctx.device, &cell);
    let row = (SLIDING.nh * SLIDING.dk) as usize;
    let out = sentinel_buffer(&ctx.device, 2 * row);
    let need =
        fa::flash_attn_vec_hybrid_batched_gqa_tmp_bytes(SLIDING.nh, SLIDING.dk, &cell.buckets())
            .unwrap()
            / 4;
    let tmp = sentinel_buffer(&ctx.device, need - 1);
    let mut enc = ctx.device.command_encoder().unwrap();
    let r = fa::flash_attn_vec_hybrid_batched_gqa(
        &mut enc,
        &mut ctx.registry,
        &ctx.device,
        &inputs.q,
        &kv.k,
        &kv.v,
        &kv.vn,
        &out,
        &tmp,
        &inputs.slot_ids,
        &inputs.seq_pos,
        &cell.params(21, 0, 1),
        1,
        &cell.buckets(),
    );
    assert!(r.is_err());
    enc.commit_and_wait().unwrap();
}
