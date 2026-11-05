//! S0-a (hf2q ADR-063 Q4-GEMMA-ATTN-EXACT, design
//! `docs/research/adr063-q4-gemma-attn-exact-design.md` §5): the R = 1 case
//! of the bitwise-exact warm-prefill attention, i.e. Q4-GEMMA-DECODE lever
//! A1 `flash_attn_vec_hybrid_batched_gqa`, at the shapes hf2q's warm route
//! (`gemma_live_attention::encode_live_seq`, `VecHybrid` arm) issues:
//!
//! * one sequence (one slot) per dispatch, n_q = the query rows of one
//!   256-row chunk (split at 512/1024), every row its own stage position;
//! * sliding layers: `mask_type` 3, window 1024, over the chronological ring
//!   stage `[nkv, 1023 + min(rows, 1024), 256]` (`kv_capacity` = stage
//!   capacity), history `min(start, 1023)`;
//! * global layers: `mask_type` 1 over the cache, positions
//!   `graft + start + i`;
//! * one (nwg, nsg) bucket per dispatch: `compute_nwg`/`compute_nsg` of the
//!   chunk's key count (the three reachable pairs (16,1), (32,1), (32,4)).
//!
//! Every comparison is on raw u32: the reduced O of every row and every
//! split-K partial and S/M word of `tmp` against the stock
//! `flash_attn_vec_hybrid_batched` + reduce over the same rows, with
//! sentinels around both views. Block composition: a row's bits do not change
//! when its neighbours change (NaN neighbour rows; the same rows dispatched
//! from a shifted chunk start).
//!
//! Run on both metallib paths, one process each:
//! `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test gemma_warm_attn_exact_s0 -- --test-threads=1`
//! and the same with `MLX_PRECOMPILED_METALLIB=0`.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use mlx_native::ops::flash_attn_vec_hybrid::{
    self as fa, FlashAttnVecTqHbParams, HybridSlotBucket,
};
use mlx_native::{DType, KernelPipelineOrigin, KernelRegistry, MlxBuffer, MlxDevice};

const SENTINEL: u32 = 0x7FA5_5A5A;
const GUARD: usize = 64;
/// hf2q `TILED_SLIDING_QUERY_CHUNK`-independent vec chunk (`VEC_QUERY_CHUNK`).
const VEC_QUERY_CHUNK: u32 = 256;
const SLIDING_CAP: u32 = 1024;
const WINDOW: u32 = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Layer {
    /// d256, 16 query heads, 8 KV heads, ring stage, mask_type 3.
    Sliding,
    /// d512, 16 query heads, 2 KV heads, linear cache, mask_type 1.
    Global,
}

impl Layer {
    fn dk(self) -> u32 {
        match self {
            Self::Sliding => 256,
            Self::Global => 512,
        }
    }
    fn nkv(self) -> u32 {
        match self {
            Self::Sliding => 8,
            Self::Global => 2,
        }
    }
    fn npp(self) -> u32 {
        self.dk() / 256
    }
}

const NH: u32 = 16;

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

fn words(b: &MlxBuffer) -> Vec<u32> {
    b.as_slice::<u32>().expect("read").to_vec()
}

/// One slot's KV: K F16 and V (TQ-HB codes + F32 norms, or F16) of
/// `[nkv, cap, dk]`.
struct Kv {
    k: MlxBuffer,
    v: MlxBuffer,
    vn: MlxBuffer,
    cap: u32,
}

fn make_kv(device: &MlxDevice, layer: Layer, cap: u32, v_f16: bool, seed: u64) -> Kv {
    let mut rng = Xor::new(seed);
    let elems = (layer.nkv() * cap * layer.dk()) as usize;
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
    let n_norms = (layer.nkv() * cap * layer.npp()) as usize;
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
    Kv { k, v, vn, cap }
}

fn make_q(rows: usize, layer: Layer, seed: u64) -> Vec<f32> {
    let mut rng = Xor::new(seed);
    let specials = [
        0.0f32, -0.0, 1.0e-40, -3.0e-39, 24.0, -17.5, 6.0e-5, -6.0e-8,
    ];
    (0..rows * (NH * layer.dk()) as usize)
        .map(|i| {
            if i % 61 == 0 {
                specials[(i / 61) % specials.len()]
            } else {
                rng.signed() * 1.5
            }
        })
        .collect()
}

/// One warm dispatch: `positions[i]` is row i's stage/cache position (the
/// row attends keys `[.., positions[i]]`), `kv_seq_len` the chunk's key
/// count (the bucket).
#[derive(Clone)]
struct Cell {
    layer: Layer,
    label: String,
    positions: Vec<u32>,
    kv_seq_len: u32,
    cbits: u32,
    q: Vec<f32>,
}

impl Cell {
    fn bucket(&self) -> HybridSlotBucket {
        HybridSlotBucket::for_kv_seq_len(self.kv_seq_len)
    }

    fn params(&self, kv: &Kv) -> FlashAttnVecTqHbParams {
        let b = self.bucket();
        FlashAttnVecTqHbParams {
            num_heads: NH,
            num_kv_heads: self.layer.nkv(),
            head_dim: self.layer.dk(),
            kv_seq_len: self.kv_seq_len,
            kv_capacity: kv.cap,
            scale: 1.0,
            mask_type: if self.layer == Layer::Sliding { 3 } else { 1 },
            sliding_window: if self.layer == Layer::Sliding {
                WINDOW
            } else {
                0
            },
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 22.627417,
            codebook_bits: self.cbits,
            fuse_fwht_pre: 0,
            nsg: b.nsg,
        }
    }
}

/// hf2q's sliding chunk at absolute start `start` with `rows` rows: the
/// stage holds `min(start, 1023)` history rows, row i sits at history + i.
fn sliding_cell(start: u32, rows: u32, cbits: u32, seed: u64) -> Cell {
    let history = start.min(SLIDING_CAP - 1);
    Cell {
        layer: Layer::Sliding,
        label: format!("sliding start={start} history={history} rows={rows} cbits={cbits}"),
        positions: (0..rows).map(|i| history + i).collect(),
        kv_seq_len: history + rows,
        cbits,
        q: make_q(rows as usize, Layer::Sliding, seed),
    }
}

/// hf2q's global chunk: rows at `graft + start + i`.
fn global_cell(graft: u32, start: u32, rows: u32, cbits: u32, seed: u64) -> Cell {
    let base = graft + start;
    Cell {
        layer: Layer::Global,
        label: format!("global graft={graft} start={start} rows={rows} cbits={cbits}"),
        positions: (0..rows).map(|i| base + i).collect(),
        kv_seq_len: base + rows,
        cbits,
        q: make_q(rows as usize, Layer::Global, seed),
    }
}

struct Run {
    out: Vec<u32>,
    tmp: Vec<u32>,
}

struct Ctx {
    device: MlxDevice,
    registry: KernelRegistry,
}

fn view(device: &MlxDevice, floats: usize) -> (MlxBuffer, MlxBuffer) {
    let base = words_buffer(device, &vec![SENTINEL; floats + 2 * GUARD], DType::F32);
    let v = base.slice_view((GUARD * 4) as u64, floats);
    (base, v)
}

/// The kernel under test.
#[derive(Clone, Copy, Debug)]
enum Variant {
    /// Stock `flash_attn_vec_hybrid_batched` + reduce (the reference).
    Stock,
    /// Lever A1 `flash_attn_vec_hybrid_batched_gqa` (R = 1) at group G.
    Gqa(u32),
    /// `flash_attn_vec_hybrid_batched_rows` at (group G, rows R).
    Rows(u32, u32),
}

fn run(ctx: &mut Ctx, cell: &Cell, kv: &Kv, variant: Variant) -> Run {
    let n = cell.positions.len();
    let row = (NH * cell.layer.dk()) as usize;
    let mut q_words = vec![SENTINEL; GUARD];
    q_words.extend(cell.q.iter().map(|x| x.to_bits()));
    let q = words_buffer(&ctx.device, &q_words, DType::F32)
        .slice_view((GUARD * 4) as u64, cell.q.len());
    let slot_ids = words_buffer(&ctx.device, &vec![0u32; n], DType::U32);
    let seq_pos = words_buffer(&ctx.device, &cell.positions, DType::U32);
    let params = cell.params(kv);
    let buckets = vec![cell.bucket(); n];
    // One shared bucket: the grouped layout equals the stock batched one
    // (`n_q · heads · nwg · (head_dim + 2)` floats), so both arms get the
    // same exactly sized view and every word is compared.
    let tmp_floats =
        fa::flash_attn_vec_hybrid_batched_gqa_tmp_bytes(NH, cell.layer.dk(), &buckets).unwrap() / 4;
    let b = cell.bucket();
    assert_eq!(
        tmp_floats,
        n * (NH * b.nwg * (cell.layer.dk() + 2)) as usize,
        "{}: tmp layout",
        cell.label
    );
    let (out_base, out) = view(&ctx.device, n * row);
    let (tmp_base, tmp) = view(&ctx.device, tmp_floats);
    let mut enc = ctx.device.command_encoder().unwrap();
    match variant {
        Variant::Stock => fa::flash_attn_vec_hybrid_batched(
            &mut enc,
            &mut ctx.registry,
            &ctx.device,
            n as u32,
            &q,
            &kv.k,
            &kv.v,
            &kv.vn,
            &out,
            &tmp,
            &slot_ids,
            &seq_pos,
            &params,
        ),
        Variant::Gqa(g) => fa::flash_attn_vec_hybrid_batched_gqa(
            &mut enc,
            &mut ctx.registry,
            &ctx.device,
            &q,
            &kv.k,
            &kv.v,
            &kv.vn,
            &out,
            &tmp,
            &slot_ids,
            &seq_pos,
            &params,
            g,
            &buckets,
        ),
        Variant::Rows(g, r) => fa::flash_attn_vec_hybrid_batched_rows(
            &mut enc,
            &mut ctx.registry,
            &ctx.device,
            n as u32,
            &q,
            &kv.k,
            &kv.v,
            &kv.vn,
            &out,
            &tmp,
            &slot_ids,
            &seq_pos,
            &params,
            g,
            r,
            b,
        ),
    }
    .unwrap_or_else(|e| panic!("{} {variant:?}: dispatch: {e}", cell.label));
    enc.commit_and_wait().unwrap();
    let (out, tmp) = (words(&out_base), words(&tmp_base));
    for (name, w) in [("O", &out), ("tmp", &tmp)] {
        assert!(
            w[..GUARD]
                .iter()
                .chain(&w[w.len() - GUARD..])
                .all(|&x| x == SENTINEL),
            "{} {variant:?}: wrote outside the {name} view",
            cell.label
        );
    }
    Run { out, tmp }
}

fn legal_groups(layer: Layer, nsg: u32) -> Vec<u32> {
    fa::FLASH_ATTN_VEC_HYBRID_GQA_GROUPS
        .iter()
        .copied()
        .filter(|&g| {
            fa::flash_attn_vec_hybrid_batched_gqa_group_is_legal(
                NH,
                layer.nkv(),
                layer.dk(),
                g,
                nsg,
            )
        })
        .collect()
}

/// Every legal (G, R) pair at the cell's NSG: (G, 1) through A1 and the rows
/// kernel; R > 1 through the rows kernel.
fn legal_variants(layer: Layer, nsg: u32) -> Vec<Variant> {
    let mut v: Vec<Variant> = legal_groups(layer, nsg)
        .into_iter()
        .map(Variant::Gqa)
        .collect();
    v.extend(
        fa::FLASH_ATTN_VEC_HYBRID_ROWS_CONFIGS
            .iter()
            .filter(|&&(g, r)| {
                fa::flash_attn_vec_hybrid_batched_rows_is_legal(
                    NH,
                    layer.nkv(),
                    layer.dk(),
                    g,
                    r,
                    nsg,
                )
            })
            .map(|&(g, r)| Variant::Rows(g, r)),
    );
    v
}

/// Every legal variant against the stock dispatch: O and tmp, u32-equal.
/// Returns the stock O rows (without guards).
fn check(ctx: &mut Ctx, cell: &Cell, kv: &Kv) -> Vec<u32> {
    let stock = run(ctx, cell, kv, Variant::Stock);
    let variants = legal_variants(cell.layer, cell.bucket().nsg);
    assert!(
        variants
            .iter()
            .any(|v| matches!(v, Variant::Rows(_, r) if *r > 1)),
        "{}: some R > 1 must be legal",
        cell.label
    );
    for variant in variants {
        let cand = run(ctx, cell, kv, variant);
        if let Some(i) = cand.out.iter().zip(&stock.out).position(|(a, b)| a != b) {
            panic!(
                "{} {variant:?}: O word {} differs ({:#x} vs stock {:#x})",
                cell.label,
                i - GUARD,
                cand.out[i],
                stock.out[i]
            );
        }
        assert_eq!(
            cand.tmp.len(),
            stock.tmp.len(),
            "{} {variant:?}: tmp layout size",
            cell.label
        );
        let diffs = cand
            .tmp
            .iter()
            .zip(&stock.tmp)
            .filter(|(a, b)| a != b)
            .count();
        if let Some(i) = cand.tmp.iter().zip(&stock.tmp).position(|(a, b)| a != b) {
            panic!(
                "{} {variant:?}: {diffs} tmp words differ; first word {} ({:#x} vs stock {:#x})",
                cell.label,
                i - GUARD,
                cand.tmp[i],
                stock.tmp[i]
            );
        }
    }
    stock.out[GUARD..stock.out.len() - GUARD].to_vec()
}

fn ctx() -> Ctx {
    Ctx {
        device: MlxDevice::new().expect("Metal device"),
        registry: KernelRegistry::new(),
    }
}

/// Row counts of S0-a: canonical chunk lengths and non-multiples of every R.
const ROWS: [u32; 15] = [1, 3, 7, 9, 31, 32, 54, 56, 92, 97, 132, 191, 200, 220, 256];

#[test]
fn sliding_ring_stage_histories_rows_and_buckets_bitwise() {
    let mut ctx = ctx();
    // Starts give stage histories 0, 1, 465, 714, 943, 1022, 1023 (full) and
    // the far past (history saturated at 1023).
    let starts = [0u32, 1, 465, 714, 943, 1022, 1023, 5000];
    let mut buckets = std::collections::BTreeSet::new();
    for (si, &start) in starts.iter().enumerate() {
        for (ri, &rows) in ROWS.iter().enumerate() {
            let history = start.min(SLIDING_CAP - 1);
            let stage_cap = SLIDING_CAP - 1 + rows.min(1024);
            let kv = make_kv(
                &ctx.device,
                Layer::Sliding,
                stage_cap,
                false,
                0x5000 + (si * 32 + ri) as u64,
            );
            let cell = sliding_cell(start, rows, 8, 0x6000 + (si * 32 + ri) as u64);
            assert!(cell.kv_seq_len <= stage_cap && history + rows == cell.kv_seq_len);
            let b = cell.bucket();
            buckets.insert((b.nwg, b.nsg));
            check(&mut ctx, &cell, &kv);
        }
    }
    let reached: Vec<_> = buckets.into_iter().collect();
    assert_eq!(
        reached,
        vec![(16, 1), (32, 1), (32, 4)],
        "sliding buckets reached"
    );
}

#[test]
fn global_cache_kv_edges_grafts_and_rows_bitwise() {
    let mut ctx = ctx();
    let cap = 4096 + 512;
    let kv = make_kv(&ctx.device, Layer::Global, cap, false, 0x610B);
    // Key counts at every bucket edge (kL = start + rows): 1, 31, 32, 33,
    // 511, 512, 513, 1023, 1024, 1025, 2047, 4096.
    let edges = [
        1u32, 31, 32, 33, 511, 512, 513, 1023, 1024, 1025, 2047, 4096,
    ];
    let mut buckets = std::collections::BTreeSet::new();
    for (ei, &kl) in edges.iter().enumerate() {
        for &rows in ROWS.iter().filter(|&&r| r <= kl) {
            for graft in [0u32, 37] {
                if graft + kl > cap {
                    continue;
                }
                let start = kl - rows;
                let cell = global_cell(
                    graft,
                    start,
                    rows,
                    8,
                    0x7000 + ei as u64 * 1000 + rows as u64 + graft as u64,
                );
                let b = cell.bucket();
                buckets.insert((b.nwg, b.nsg));
                check(&mut ctx, &cell, &kv);
            }
        }
    }
    let reached: Vec<_> = buckets.into_iter().collect();
    assert_eq!(
        reached,
        vec![(16, 1), (32, 1), (32, 4)],
        "global buckets reached"
    );
}

#[test]
fn canonical_four_lane_chunks_bitwise() {
    // The canonical run's warm lanes (history, suffix) chunked as hf2q's vec
    // route does: 256-row chunks split at the 512/1024 key buckets.
    let mut ctx = ctx();
    let lanes = [
        (458u32, 58u32),
        (516, 191),
        (707, 220),
        (927, 288),
        (1215, 150),
        (1379, 348),
        (465, 56),
        (714, 200),
        (943, 311),
        (1254, 385),
    ];
    for (li, &(start, rows)) in lanes.iter().enumerate() {
        let mut local = 0;
        while local < rows {
            let pos = start + local;
            let mut len = (rows - local).min(VEC_QUERY_CHUNK);
            for boundary in [512u32, 1024] {
                if pos < boundary && pos + len > boundary {
                    len = boundary - pos;
                    break;
                }
            }
            let stage_cap = SLIDING_CAP - 1 + rows.min(VEC_QUERY_CHUNK);
            let kv_s = make_kv(
                &ctx.device,
                Layer::Sliding,
                stage_cap,
                false,
                0x8000 + li as u64,
            );
            check(
                &mut ctx,
                &sliding_cell(pos, len, 8, 0x8100 + (li * 8) as u64 + local as u64),
                &kv_s,
            );
            let kv_g = make_kv(
                &ctx.device,
                Layer::Global,
                pos + len + 1,
                false,
                0x8200 + li as u64,
            );
            check(
                &mut ctx,
                &global_cell(0, pos, len, 8, 0x8300 + (li * 8) as u64 + local as u64),
                &kv_g,
            );
            local += len;
        }
    }
}

#[test]
fn codebook_widths_and_f16_v_bitwise() {
    let mut ctx = ctx();
    for v_f16 in [false, true] {
        for cbits in [5u32, 6, 8] {
            if v_f16 && cbits != 8 {
                continue;
            }
            let rows = 97;
            let kv = make_kv(
                &ctx.device,
                Layer::Sliding,
                SLIDING_CAP - 1 + rows,
                v_f16,
                0x9000 + cbits as u64,
            );
            for start in [300u32, 1400] {
                check(
                    &mut ctx,
                    &sliding_cell(start, rows, cbits, 0x9100 + start as u64),
                    &kv,
                );
            }
            let kv = make_kv(
                &ctx.device,
                Layer::Global,
                2048,
                v_f16,
                0x9200 + cbits as u64,
            );
            for start in [300u32, 1400] {
                check(
                    &mut ctx,
                    &global_cell(0, start, rows, cbits, 0x9300 + start as u64),
                    &kv,
                );
            }
        }
    }
}

#[test]
fn block_composition_nan_neighbours_and_shifted_starts() {
    let mut ctx = ctx();
    for layer in [Layer::Sliding, Layer::Global] {
        // Keep the bucket fixed across the variants by fixing kv_seq_len.
        for &(start, rows) in &[(714u32, 56u32), (1300, 97)] {
            let make = |seed| match layer {
                Layer::Sliding => sliding_cell(start, rows, 8, seed),
                Layer::Global => global_cell(0, start, rows, 8, seed),
            };
            let cap = match layer {
                Layer::Sliding => SLIDING_CAP - 1 + rows,
                Layer::Global => start + rows + 1,
            };
            let kv = make_kv(&ctx.device, layer, cap, false, 0xA000 + start as u64);
            let base = make(0xA100 + start as u64);
            let row = (NH * layer.dk()) as usize;
            let reference = check(&mut ctx, &base, &kv);
            // NaN neighbours: rows 0 and 2 poisoned; row 1 and rows >= 3 keep their bits.
            let mut poisoned = base.clone();
            for r in [0usize, 2] {
                poisoned.q[r * row..(r + 1) * row].fill(f32::NAN);
            }
            poisoned.label += " nan-rows-0-2";
            let out = check(&mut ctx, &poisoned, &kv);
            for r in (1..rows as usize).filter(|&r| r != 2) {
                assert_eq!(
                    out[r * row..(r + 1) * row],
                    reference[r * row..(r + 1) * row],
                    "{}: row {r} changed with NaN neighbours",
                    poisoned.label
                );
            }
            // Shifted chunk starts (by 1 and by 7 rows), same bucket: the
            // overlapping rows keep their bits.
            for shift in [1usize, 7] {
                let mut shifted = base.clone();
                shifted.positions = base.positions[shift..].to_vec();
                shifted.q = base.q[shift * row..].to_vec();
                shifted.label += &format!(" shift={shift}");
                assert_eq!(shifted.bucket(), base.bucket());
                let out = check(&mut ctx, &shifted, &kv);
                assert_eq!(
                    out[..],
                    reference[shift * row..],
                    "{}: rows changed with the chunk start",
                    shifted.label
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
    let mut ctx = ctx();
    let mut names = Vec::new();
    for dk in [256u32, 512] {
        for g in fa::FLASH_ATTN_VEC_HYBRID_GQA_GROUPS {
            names.push(format!("flash_attn_vec_hybrid_batched_gqa{g}_dk{dk}"));
        }
        for (g, r) in fa::FLASH_ATTN_VEC_HYBRID_ROWS_CONFIGS {
            names.push(format!(
                "flash_attn_vec_hybrid_batched_rows_g{g}_r{r}_dk{dk}"
            ));
        }
    }
    {
        for name in names {
            for (cbits, v16) in [(8, 0), (6, 0), (5, 0), (8, 1)] {
                let label = ctx
                    .registry
                    .get_pipeline_with_constants(
                        &name,
                        ctx.device.metal_device(),
                        &[],
                        &[(50, cbits), (51, v16)],
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
                    .label()
                    .to_owned();
                assert_eq!(
                    ctx.registry.pipeline_identity(&label).unwrap().origin,
                    expected,
                    "{label}"
                );
            }
        }
    }
    eprintln!("GEMMA_WARM_ATTN_EXACT_S0A_ORIGIN {expected:?}");
}

#[test]
fn rows_kernel_refuses_mixed_slots_unordered_rows_and_ring_masks() {
    let mut ctx = ctx();
    let rows = 8u32;
    let kv = make_kv(
        &ctx.device,
        Layer::Sliding,
        SLIDING_CAP - 1 + rows,
        false,
        0xBAD1,
    );
    let cell = sliding_cell(100, rows, 8, 0xBAD2);
    let n = rows as usize;
    let row = (NH * Layer::Sliding.dk()) as usize;
    let q = words_buffer(
        &ctx.device,
        &cell.q.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        DType::F32,
    );
    let (out_base, out) = view(&ctx.device, n * row);
    let (_tmp_base, tmp) = view(&ctx.device, n * (NH * 32 * (256 + 2)) as usize);
    let b = cell.bucket();
    let attempt = |ctx: &mut Ctx, slots: Vec<u32>, pos: Vec<u32>, mask_type: u32| {
        let slot_ids = words_buffer(&ctx.device, &slots, DType::U32);
        let seq_pos = words_buffer(&ctx.device, &pos, DType::U32);
        let mut params = cell.params(&kv);
        params.mask_type = mask_type;
        let mut enc = ctx.device.command_encoder().unwrap();
        let r = fa::flash_attn_vec_hybrid_batched_rows(
            &mut enc,
            &mut ctx.registry,
            &ctx.device,
            rows,
            &q,
            &kv.k,
            &kv.v,
            &kv.vn,
            &out,
            &tmp,
            &slot_ids,
            &seq_pos,
            &params,
            2,
            4,
            b,
        );
        enc.commit_and_wait().unwrap();
        r
    };
    let good: Vec<u32> = cell.positions.clone();
    let mut mixed = vec![0u32; n];
    mixed[2] = 1;
    assert!(
        attempt(&mut ctx, mixed, good.clone(), 3).is_err(),
        "mixed slots in a row block"
    );
    let mut unordered = good.clone();
    unordered.swap(1, 2);
    assert!(
        attempt(&mut ctx, vec![0; n], unordered, 3).is_err(),
        "unordered rows in a block"
    );
    assert!(
        attempt(&mut ctx, vec![0; n], good.clone(), 2).is_err(),
        "ring mask refused"
    );
    assert!(
        words(&out_base).iter().all(|&w| w == SENTINEL),
        "refused calls wrote O"
    );
}
