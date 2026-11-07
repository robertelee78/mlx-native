//! Row-exactness gate for the row-batched Qwen3.6 full-attention decode chain.
//!
//! hf2q's joint full-attention layer used to run, for each of its w rows in
//! turn and behind a barrier: the TQ-HB K and V encode at the row's cursor,
//! the Q sign premultiply over the row's 16 heads, the row's split-K attention
//! (scalar kernel, or the GQA-Q2 kernel from kv 8192), its reduce, the undo
//! rotation, a copy into the packed attention rows, and then one packed
//! sigmoid gate multiply. The row-batched chain runs one batched K and one V
//! encode, one premultiply over all w * 16 heads, `flash_attn_vec_tq_hb_rows`
//! (all mains, one barrier, all reduces, straight into the packed rows) and
//! one fused undo + sigmoid gate.
//!
//! Every row's gated output and every cache byte must be bitwise those of the
//! per-row chain run for that row alone, at every width 1..=4, in any row
//! order and for any cohort — `f32::to_bits` / byte equality, zero tolerance.
//! The rows span every split-K schedule bucket (nwg 16 / 32, nsg 1 / 4) and
//! both attention kernels.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_native::ops::flash_attn_vec_tq_hb::{
    self, FlashAttnVecTqHbParams, GqaTile, TqHbDecodeRow, TqHbRowKernel,
};
use mlx_native::ops::fwht_standalone;
use mlx_native::ops::hadamard_quantize_kv;
use mlx_native::ops::sigmoid_mul::dispatch_sigmoid_mul;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

const NH: u32 = 16;
const NKV: u32 = 2;
const HD: u32 = 256;
const Q_TOTAL: usize = (NH * HD) as usize;
const KV_TOTAL: usize = (NKV * HD) as usize;
const CBITS: u32 = 8;
const SLOTS: u32 = 4;
/// Covers the GQA-Q2 row (kv 8192) plus headroom.
const CAPACITY: u32 = 8320;
/// hf2q `GQA_Q2_MIN_KV_SEQ_LEN` (`qwen35/gqa_q2_policy.rs`).
const GQA_Q2_MIN_KV: u32 = 8192;

struct Xor(u64);

impl Xor {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x as u32
    }

    fn signed(&mut self, scale: f32) -> f32 {
        scale * (self.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
    }
}

fn f32_buffer(device: &MlxDevice, values: &[f32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, vec![values.len()])
        .expect("allocate f32");
    buffer.as_mut_slice::<f32>().expect("map f32").copy_from_slice(values);
    buffer
}

fn u8_buffer(device: &MlxDevice, values: &[u8]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len(), DType::U8, vec![values.len()])
        .expect("allocate u8");
    buffer.as_mut_slice::<u8>().expect("map u8").copy_from_slice(values);
    buffer
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("allocate u32");
    buffer.as_mut_slice::<u32>().expect("map u32").copy_from_slice(values);
    buffer
}

fn zeros(device: &MlxDevice, elements: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(elements * 4, DType::F32, vec![elements])
        .expect("allocate zeros");
    buffer.as_mut_slice::<f32>().expect("map zeros").fill(0.0);
    buffer
}

fn tmp_buffer(device: &MlxDevice) -> MlxBuffer {
    let bytes = flash_attn_vec_tq_hb::tmp_buffer_bytes(NH, HD);
    device
        .alloc_buffer(bytes, DType::F32, vec![bytes / 4])
        .expect("allocate tmp")
}

/// One decode row: the inputs of the attention core for one sequence.
struct RowInput {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    slot: u32,
    cursor: u32,
}

impl RowInput {
    fn kv_len(&self) -> u32 {
        self.cursor + 1
    }

    fn kernel(&self) -> TqHbRowKernel {
        if self.kv_len() >= GQA_Q2_MIN_KV {
            TqHbRowKernel::Gqa(GqaTile::Q2)
        } else {
            TqHbRowKernel::Scalar
        }
    }

    fn params(&self) -> FlashAttnVecTqHbParams {
        FlashAttnVecTqHbParams {
            num_heads: NH,
            num_kv_heads: NKV,
            head_dim: HD,
            kv_seq_len: self.kv_len(),
            kv_capacity: CAPACITY,
            scale: 1.0 / (HD as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: CBITS,
            fuse_fwht_pre: 0,
            nsg: flash_attn_vec_tq_hb::compute_nsg(self.kv_len()),
        }
    }
}

/// Initial content of the multi-slot cache `[SLOTS, NKV, CAPACITY, HD]`.
struct CacheInit {
    k_packed: Vec<u8>,
    v_packed: Vec<u8>,
    k_norms: Vec<f32>,
    v_norms: Vec<f32>,
}

struct Cache {
    k_packed: MlxBuffer,
    v_packed: MlxBuffer,
    k_norms: MlxBuffer,
    v_norms: MlxBuffer,
}

const SLOT_CODES: usize = (NKV * CAPACITY * HD) as usize;
const SLOT_NORMS: usize = (NKV * CAPACITY) as usize;

impl Cache {
    fn new(device: &MlxDevice, init: &CacheInit) -> Self {
        Self {
            k_packed: u8_buffer(device, &init.k_packed),
            v_packed: u8_buffer(device, &init.v_packed),
            k_norms: f32_buffer(device, &init.k_norms),
            v_norms: f32_buffer(device, &init.v_norms),
        }
    }

    /// hf2q `TqFullAttnKvBuffers::slot_views`: zero-copy slot regions.
    fn slot(&self, slot: u32) -> [MlxBuffer; 4] {
        let s = slot as usize;
        [
            self.k_packed.slice_view((s * SLOT_CODES) as u64, SLOT_CODES),
            self.k_norms.slice_view((s * SLOT_NORMS * 4) as u64, SLOT_NORMS),
            self.v_packed.slice_view((s * SLOT_CODES) as u64, SLOT_CODES),
            self.v_norms.slice_view((s * SLOT_NORMS * 4) as u64, SLOT_NORMS),
        ]
    }

    fn snapshot(&self) -> (Vec<u8>, Vec<u8>, Vec<u32>, Vec<u32>) {
        let bits = |b: &MlxBuffer| b.as_slice::<f32>().unwrap().iter().map(|x| x.to_bits()).collect();
        (
            self.k_packed.as_slice::<u8>().unwrap().to_vec(),
            self.v_packed.as_slice::<u8>().unwrap().to_vec(),
            bits(&self.k_norms),
            bits(&self.v_norms),
        )
    }
}

fn cache_init(seed: u64) -> CacheInit {
    let mut rng = Xor(seed);
    let codes = (SLOTS as usize) * SLOT_CODES;
    let norms = (SLOTS as usize) * SLOT_NORMS;
    CacheInit {
        k_packed: (0..codes).map(|_| rng.next_u32() as u8).collect(),
        v_packed: (0..codes).map(|_| rng.next_u32() as u8).collect(),
        k_norms: (0..norms).map(|_| 0.5 + rng.signed(1.0).abs()).collect(),
        v_norms: (0..norms).map(|_| 0.5 + rng.signed(1.0).abs()).collect(),
    }
}

/// Rows on distinct slots whose kv lengths cover nwg 16 (<= 512), nwg 32 with
/// nsg 1 (<= 1024), nsg 4, and the GQA-Q2 kernel (>= 8192).
fn row_inputs(seed: u64) -> Vec<RowInput> {
    let mut rng = Xor(seed);
    let cursors = [8191u32, 299, 2999, 799];
    let slots = [2u32, 0, 3, 1];
    cursors
        .iter()
        .zip(slots)
        .map(|(&cursor, slot)| RowInput {
            q: (0..Q_TOTAL).map(|_| rng.signed(2.0)).collect(),
            k: (0..KV_TOTAL).map(|_| rng.signed(2.0)).collect(),
            v: (0..KV_TOTAL).map(|_| rng.signed(2.0)).collect(),
            gate: (0..Q_TOTAL).map(|_| rng.signed(8.0)).collect(),
            slot,
            cursor,
        })
        .collect()
}

struct Ctx {
    device: MlxDevice,
    registry: KernelRegistry,
}

impl Ctx {
    fn new() -> Self {
        let device = MlxDevice::new().expect("Metal device");
        let mut registry = KernelRegistry::new();
        flash_attn_vec_tq_hb::register(&mut registry);
        mlx_native::ops::flash_attn_vec::register(&mut registry);
        Self { device, registry }
    }

    fn encoder(&self) -> CommandEncoder {
        self.device.command_encoder().expect("encoder")
    }
}

/// The per-row chain hf2q ran before (`write_kv_with_optional_tq_encode` +
/// `dispatch_decode_sdpa_with_optional_tq` + packed `sigmoid_mul`), for one
/// row alone. Returns the gated output; the row's K/V land in `cache`.
fn per_row_reference(ctx: &mut Ctx, row: &RowInput, cache: &Cache) -> Vec<u32> {
    let device = &ctx.device;
    let q = f32_buffer(device, &row.q);
    let k = f32_buffer(device, &row.k);
    let v = f32_buffer(device, &row.v);
    let gate = f32_buffer(device, &row.gate);
    let out = zeros(device, Q_TOTAL);
    let gated = zeros(device, Q_TOTAL);
    let tmp = tmp_buffer(device);
    let [k_packed, k_norms, v_packed, v_norms] = cache.slot(row.slot);
    let params = row.params();
    let sig_params = u32_buffer(device, &[Q_TOTAL as u32]);

    let mut enc = ctx.encoder();
    let md = device.metal_device();
    for (src, packed, norms) in [(&k, &k_packed, &k_norms), (&v, &v_packed, &v_norms)] {
        hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
            &mut enc, &mut ctx.registry, md, src, packed, norms, NKV, HD, CAPACITY, row.cursor, 1,
            0, false, 1.0, CBITS,
        )
        .expect("per-row encode");
    }
    enc.memory_barrier();
    fwht_standalone::dispatch_fwht_sign_premult_f32(&mut enc, &mut ctx.registry, md, &q, NH, HD)
        .expect("per-row premult");
    enc.memory_barrier();
    match row.kernel() {
        TqHbRowKernel::Scalar => flash_attn_vec_tq_hb::flash_attn_vec_tq_hb(
            &mut enc, &mut ctx.registry, device, &q, &k_packed, &k_norms, &v_packed, &v_norms,
            &out, &tmp, &params,
        ),
        TqHbRowKernel::Gqa(tile) => flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_gqa(
            &mut enc, &mut ctx.registry, device, &q, &k_packed, &k_norms, &v_packed, &v_norms,
            &out, &tmp, &params, tile,
        ),
    }
    .expect("per-row attention");
    enc.memory_barrier();
    fwht_standalone::dispatch_fwht_sign_undo_f32(&mut enc, &mut ctx.registry, md, &out, NH, HD)
        .expect("per-row undo");
    enc.memory_barrier();
    dispatch_sigmoid_mul(&mut enc, &mut ctx.registry, md, &out, &gate, &gated, &sig_params, Q_TOTAL as u32)
        .expect("per-row sigmoid_mul");
    enc.commit_and_wait().expect("per-row completion");
    gated.as_slice::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
}

/// The row-batched chain over `rows` (in this order). Returns each row's
/// gated output; the rows' K/V land in `cache`.
fn rows_chain(ctx: &mut Ctx, rows: &[&RowInput], cache: &Cache) -> Vec<Vec<u32>> {
    let device = &ctx.device;
    let w = rows.len();
    let pack = |field: fn(&RowInput) -> &[f32]| -> MlxBuffer {
        let values: Vec<f32> = rows.iter().flat_map(|r| field(r).iter().copied()).collect();
        f32_buffer(device, &values)
    };
    let q = pack(|r| &r.q);
    let k = pack(|r| &r.k);
    let v = pack(|r| &r.v);
    let gate = pack(|r| &r.gate);
    let attention = zeros(device, w * Q_TOTAL);
    let gated = zeros(device, w * Q_TOTAL);
    let tmps: Vec<MlxBuffer> = (0..w).map(|_| tmp_buffer(device)).collect();
    let slot_ids = u32_buffer(device, &rows.iter().map(|r| r.slot).collect::<Vec<_>>());
    let cursors = u32_buffer(device, &rows.iter().map(|r| r.cursor).collect::<Vec<_>>());
    let views: Vec<(MlxBuffer, MlxBuffer, [MlxBuffer; 4])> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let off = (i * Q_TOTAL * 4) as u64;
            (q.slice_view(off, Q_TOTAL), attention.slice_view(off, Q_TOTAL), cache.slot(r.slot))
        })
        .collect();
    let decode_rows: Vec<TqHbDecodeRow<'_>> = views
        .iter()
        .zip(&tmps)
        .zip(rows)
        .map(|(((q, out, [kp, kn, vp, vn]), tmp), r)| TqHbDecodeRow {
            q,
            k_packed: kp,
            k_norms: kn,
            v_packed: vp,
            v_norms: vn,
            output: out,
            tmp,
            params: r.params(),
            kernel: r.kernel(),
        })
        .collect();

    let mut enc = ctx.encoder();
    let md = device.metal_device();
    for (src, packed, norms) in [
        (&k, &cache.k_packed, &cache.k_norms),
        (&v, &cache.v_packed, &cache.v_norms),
    ] {
        hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_batched(
            &mut enc, &mut ctx.registry, md, src, packed, norms, &slot_ids, &cursors, w as u32,
            NKV, HD, CAPACITY, false, 1.0, CBITS,
        )
        .expect("batched encode");
    }
    fwht_standalone::dispatch_fwht_sign_premult_f32(
        &mut enc, &mut ctx.registry, md, &q, w as u32 * NH, HD,
    )
    .expect("batched premult");
    enc.memory_barrier();
    flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_rows(&mut enc, &mut ctx.registry, device, &decode_rows)
        .expect("rows attention");
    enc.memory_barrier();
    fwht_standalone::dispatch_fwht_sign_undo_sigmoid_mul_f32(
        &mut enc, &mut ctx.registry, md, &attention, &gate, &gated, w as u32 * NH, HD,
    )
    .expect("fused undo + gate");
    enc.commit_and_wait().expect("rows completion");
    gated
        .as_slice::<f32>()
        .unwrap()
        .chunks_exact(Q_TOTAL)
        .map(|row| row.iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn assert_bits(label: &str, expected: &[u32], actual: &[u32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: length");
    if let Some(i) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        panic!(
            "{label}: element {i} differs: expected {:?} ({:#010x}), got {:?} ({:#010x})",
            f32::from_bits(expected[i]),
            expected[i],
            f32::from_bits(actual[i]),
            actual[i]
        );
    }
}

/// Every row of every cohort (widths 1..=4, several orders) equals the
/// per-row chain bitwise, and so do the caches.
#[test]
fn qwen36_attn_rows_match_per_row_chain_at_every_width_and_order() {
    let mut ctx = Ctx::new();
    let init = cache_init(0x5EED_A77E);
    let inputs = row_inputs(0x0DD_BA11);

    // Per-row reference: each row alone on its own fresh cache copy; the
    // expected final cache applies every row's own write.
    let expected: Vec<Vec<u32>> = inputs
        .iter()
        .map(|row| {
            let cache = Cache::new(&ctx.device, &init);
            per_row_reference(&mut ctx, row, &cache)
        })
        .collect();
    let all_rows_cache = Cache::new(&ctx.device, &init);
    for row in &inputs {
        per_row_reference(&mut ctx, row, &all_rows_cache);
    }
    let expected_cache = all_rows_cache.snapshot();
    let cohorts: &[&[usize]] = &[
        &[0],
        &[1],
        &[2],
        &[3],
        &[0, 1],
        &[3, 0],
        &[1, 2, 3],
        &[2, 0, 3],
        &[0, 1, 2, 3],
        &[3, 2, 1, 0],
        &[1, 3, 0, 2],
    ];
    for cohort in cohorts {
        let cache = Cache::new(&ctx.device, &init);
        let rows: Vec<&RowInput> = cohort.iter().map(|&i| &inputs[i]).collect();
        let got = rows_chain(&mut ctx, &rows, &cache);
        for (pos, &i) in cohort.iter().enumerate() {
            assert_bits(
                &format!("cohort {cohort:?} row {i} (kv {})", inputs[i].kv_len()),
                &expected[i],
                &got[pos],
            );
        }
        if cohort.len() == inputs.len() {
            assert!(cache.snapshot() == expected_cache, "cohort {cohort:?}: cache bytes differ");
        }
    }
}

/// The fused undo + gate equals `fwht_sign_undo_f32` then `sigmoid_mul_f32`
/// bitwise, including saturating gates, and does not depend on the head count.
#[test]
fn fused_undo_sigmoid_matches_undo_then_sigmoid_mul() {
    let mut ctx = Ctx::new();
    let heads = 4 * NH;
    let n = (heads * HD) as usize;
    let mut rng = Xor(0xFEED_FACE);
    let data: Vec<f32> = (0..n).map(|_| rng.signed(3.0)).collect();
    let mut gate: Vec<f32> = (0..n).map(|_| rng.signed(12.0)).collect();
    gate[..8].copy_from_slice(&[0.0, -0.0, 88.0, -88.0, 104.0, -104.0, 1e-30, -1e-30]);
    let device = &ctx.device;
    let undone = f32_buffer(device, &data);
    let gate_buf = f32_buffer(device, &gate);
    let expected = zeros(device, n);
    let sig_params = u32_buffer(device, &[n as u32]);
    let md = device.metal_device();

    let mut enc = ctx.encoder();
    fwht_standalone::dispatch_fwht_sign_undo_f32(&mut enc, &mut ctx.registry, md, &undone, heads, HD)
        .unwrap();
    enc.memory_barrier();
    dispatch_sigmoid_mul(&mut enc, &mut ctx.registry, md, &undone, &gate_buf, &expected, &sig_params, n as u32)
        .unwrap();
    enc.commit_and_wait().unwrap();
    let expected: Vec<u32> = expected.as_slice::<f32>().unwrap().iter().map(|x| x.to_bits()).collect();

    // All heads in one dispatch, then each head group of NH alone.
    let data_buf = f32_buffer(device, &data);
    let fused = zeros(device, n);
    let mut enc = ctx.encoder();
    fwht_standalone::dispatch_fwht_sign_undo_sigmoid_mul_f32(
        &mut enc, &mut ctx.registry, md, &data_buf, &gate_buf, &fused, heads, HD,
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    let got: Vec<u32> = fused.as_slice::<f32>().unwrap().iter().map(|x| x.to_bits()).collect();
    assert_bits("fused undo + gate, all heads", &expected, &got);

    let per_row = zeros(device, n);
    let mut enc = ctx.encoder();
    for r in 0..4usize {
        let off = (r * Q_TOTAL * 4) as u64;
        fwht_standalone::dispatch_fwht_sign_undo_sigmoid_mul_f32(
            &mut enc,
            &mut ctx.registry,
            md,
            &data_buf.slice_view(off, Q_TOTAL),
            &gate_buf.slice_view(off, Q_TOTAL),
            &per_row.slice_view(off, Q_TOTAL),
            NH,
            HD,
        )
        .unwrap();
    }
    enc.commit_and_wait().unwrap();
    let got: Vec<u32> = per_row.as_slice::<f32>().unwrap().iter().map(|x| x.to_bits()).collect();
    assert_bits("fused undo + gate, per row", &expected, &got);
}

/// The rows entry refuses rows whose writes overlap.
#[test]
fn rows_entry_rejects_aliased_writes() {
    let mut ctx = Ctx::new();
    let init = cache_init(1);
    let cache = Cache::new(&ctx.device, &init);
    let inputs = row_inputs(2);
    let q = f32_buffer(&ctx.device, &inputs[1].q);
    let q2 = f32_buffer(&ctx.device, &inputs[3].q);
    let out = zeros(&ctx.device, Q_TOTAL);
    let out2 = zeros(&ctx.device, Q_TOTAL);
    let tmp = tmp_buffer(&ctx.device);
    let tmp2 = tmp_buffer(&ctx.device);
    let [kp, kn, vp, vn] = cache.slot(0);
    let row = |q, output, tmp| TqHbDecodeRow {
        q,
        k_packed: &kp,
        k_norms: &kn,
        v_packed: &vp,
        v_norms: &vn,
        output,
        tmp,
        params: inputs[1].params(),
        kernel: TqHbRowKernel::Scalar,
    };
    let device = &ctx.device;
    for (label, rows) in [
        ("shared output", vec![row(&q, &out, &tmp), row(&q2, &out, &tmp2)]),
        ("shared tmp", vec![row(&q, &out, &tmp), row(&q2, &out2, &tmp)]),
        ("output over another row's q", vec![row(&q, &out, &tmp), row(&q2, &q, &tmp2)]),
    ] {
        let mut enc = device.command_encoder().unwrap();
        let result = flash_attn_vec_tq_hb::flash_attn_vec_tq_hb_rows(&mut enc, &mut ctx.registry, device, &rows);
        assert!(result.is_err(), "{label} must be rejected");
    }
}
