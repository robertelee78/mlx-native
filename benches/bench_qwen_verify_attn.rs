//! Full-attention kernels of one joint DFlash2 verify layer over the TQ-HB
//! cache, model-free, at the served Qwen shapes (hf2q Splash-gap program).
//!
//! A verify step carries 1-4 lanes of 8 rows at consecutive positions of the
//! lane's slot. This bench encodes the attention kernels of one layer for such
//! a step (K/V cache write, Q rotation, attention, output un-rotation, gate)
//! three ways and times them on the GPU:
//!
//! - `prep`: the part before them that every route shares, the per-head RMS
//!   norm of Q and K and their IMROPE (four dispatches, two stages).
//! - `serial`: hf2q's per-row chain (`apply_sdpa_with_kv_cache_decode_into`
//!   per row with a barrier between rows, then row copies and the gate).
//! - `rows`: batched K/V encode, one Q rotation, every row's split-K main
//!   behind one barrier and every reduce behind another
//!   (`flash_attn_vec_tq_hb_rows`), one fused un-rotation + gate. Bitwise the
//!   serial chain.
//! - `tile`: batched K/V encode and `tq_hb_verify_tile_attention` (one pack,
//!   one split, one reduce, one fused un-rotation + gate).
//!
//! Per case it prints the fused GPU time (one command buffer, the layer's own
//! barriers; median and minimum of `--repeat`) and, per dispatch kind, the
//! count and the summed GPU time with every dispatch submitted as its own
//! command buffer (median of five rounds). Every sample is the
//! `GPUEndTime - GPUStartTime` of a command buffer committed right behind a
//! filler command buffer of six repeats of the same layer, so the GPU is at
//! the clocks of a serving cycle; without the filler the same kernels read
//! two to three times slower and unstable (GPU performance-state ramp).
//!
//! Usage: `cargo bench --bench bench_qwen_verify_attn -- [--shapes
//! qwen38,qwen36] [--lanes 1,2,4] [--histories 512,2048,8192] [--repeat 20]
//! [--routes prep,serial,rows,tile] [--json PATH]`

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use std::collections::BTreeMap;

use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::ops::flash_attn_vec_tq_hb::{
    compute_nsg, flash_attn_vec_tq_hb_rows, tmp_buffer_bytes, FlashAttnVecTqHbParams, GqaTile,
    TqHbDecodeRow, TqHbRowKernel,
};
use mlx_native::ops::fwht_standalone::{
    dispatch_fwht_sign_premult_f32, dispatch_fwht_sign_undo_f32,
    dispatch_fwht_sign_undo_sigmoid_mul_f32,
};
use mlx_native::ops::hadamard_quantize_kv::{
    dispatch_hadamard_quantize_kv_hb_batched, dispatch_hadamard_quantize_kv_hb_seq,
};
use mlx_native::ops::rms_norm::dispatch_rms_norm;
use mlx_native::ops::rope_multi::{dispatch_rope_multi_cached, RopeMultiMode, RopeMultiParams};
use mlx_native::ops::sigmoid_mul::dispatch_sigmoid_mul;
use mlx_native::ops::tq_hb_verify_tile::{
    tq_hb_verify_tile_attention, tq_hb_verify_tile_available, verify_tile_scratch_bytes,
    verify_tile_slot_splits, TqHbVerifyLane, TqHbVerifyTileCache, TqHbVerifyTileScratch,
    TqHbVerifyTileShape, VERIFY_TILE_ROWS,
};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

type R<T> = Result<T, Box<dyn std::error::Error>>;

const HEAD_DIM: u32 = 256;
const CODEBOOK_BITS: u32 = 8;
const SLOTS: u32 = 4;
const ROWS: u32 = VERIFY_TILE_ROWS;
const WARMUP_MS: u128 = 300;
/// Layers of GPU work queued ahead of every sample.
const FILLER_LAYERS: usize = 6;
/// Rounds of the per-dispatch attribution (each round submits every dispatch).
const SEPARATE_REPEAT: usize = 5;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    n_heads: u32,
    n_kv: u32,
    /// hf2q pads the dense target's verify tile to 32 rows; the MoE target
    /// uses the smallest power of two that holds the rows (at least 4).
    dense: bool,
    /// hf2q `gqa_q2_policy`: shortest KV length the per-row chain runs the
    /// cooperative GQA-Q2 kernel at (rows = 1).
    gqa_q2_min_kv: u32,
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "qwen38",
        n_heads: 24,
        n_kv: 4,
        dense: true,
        gqa_q2_min_kv: 8_192,
    },
    Shape {
        name: "qwen36",
        n_heads: 16,
        n_kv: 2,
        dense: false,
        gqa_q2_min_kv: 4_096,
    },
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn alloc(dev: &MlxDevice, dtype: DType, elements: usize) -> R<MlxBuffer> {
    Ok(dev.alloc_buffer(elements * dtype.size_of(), dtype, vec![elements])?)
}

fn f32_buf(dev: &MlxDevice, elements: usize, rng: &mut Rng, lo: f32, hi: f32) -> R<MlxBuffer> {
    let mut b = alloc(dev, DType::F32, elements)?;
    for v in b.as_mut_slice::<f32>()? {
        *v = lo + (hi - lo) * 0.5 * (rng.unit() + 1.0);
    }
    Ok(b)
}

fn u32_buf(dev: &MlxDevice, values: &[u32]) -> R<MlxBuffer> {
    let mut b = alloc(dev, DType::U32, values.len())?;
    b.as_mut_slice::<u32>()?.copy_from_slice(values);
    Ok(b)
}

/// The step's inputs and the layer's TQ cache (`SLOTS` slots).
struct Case {
    shape: Shape,
    lanes: u32,
    history: u32,
    capacity: u32,
    width: u32,
    q: MlxBuffer,
    k: MlxBuffer,
    v: MlxBuffer,
    gate: MlxBuffer,
    k_packed: MlxBuffer,
    k_norms: MlxBuffer,
    v_packed: MlxBuffer,
    v_norms: MlxBuffer,
}

impl Case {
    fn new(dev: &MlxDevice, shape: Shape, lanes: u32, history: u32, rng: &mut Rng) -> R<Self> {
        let capacity = (history + ROWS).div_ceil(32) * 32 + 32;
        let real = lanes * ROWS;
        let width = if shape.dense {
            32
        } else {
            real.next_power_of_two().clamp(4, 32)
        };
        let q_total = (shape.n_heads * HEAD_DIM) as usize;
        let kv_total = (shape.n_kv * HEAD_DIM) as usize;
        let tokens = (SLOTS * shape.n_kv * capacity) as usize;
        let mut k_packed = alloc(dev, DType::U8, tokens * HEAD_DIM as usize)?;
        let mut v_packed = alloc(dev, DType::U8, tokens * HEAD_DIM as usize)?;
        for packed in [&mut k_packed, &mut v_packed] {
            for chunk in packed.as_mut_slice::<u8>()?.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rng.next().to_le_bytes());
            }
        }
        Ok(Self {
            shape,
            lanes,
            history,
            capacity,
            width,
            q: f32_buf(dev, width as usize * q_total, rng, -1.0, 1.0)?,
            k: f32_buf(dev, width as usize * kv_total, rng, -1.0, 1.0)?,
            v: f32_buf(dev, width as usize * kv_total, rng, -1.0, 1.0)?,
            gate: f32_buf(dev, width as usize * q_total, rng, -2.0, 2.0)?,
            k_packed,
            k_norms: f32_buf(dev, tokens, rng, 0.5, 1.5)?,
            v_packed,
            v_norms: f32_buf(dev, tokens, rng, 0.5, 1.5)?,
        })
    }

    fn q_total(&self) -> usize {
        (self.shape.n_heads * HEAD_DIM) as usize
    }

    fn kv_total(&self) -> usize {
        (self.shape.n_kv * HEAD_DIM) as usize
    }

    /// Zero-copy views of one slot's cache.
    fn slot(&self, slot: u32) -> [MlxBuffer; 4] {
        let tokens = (self.shape.n_kv * self.capacity) as usize;
        let codes = tokens * HEAD_DIM as usize;
        let s = slot as usize;
        [
            self.k_packed.slice_view((s * codes) as u64, codes),
            self.k_norms.slice_view((s * tokens * 4) as u64, tokens),
            self.v_packed.slice_view((s * codes) as u64, codes),
            self.v_norms.slice_view((s * tokens * 4) as u64, tokens),
        ]
    }

    fn params(&self, kv_seq_len: u32) -> FlashAttnVecTqHbParams {
        FlashAttnVecTqHbParams {
            num_heads: self.shape.n_heads,
            num_kv_heads: self.shape.n_kv,
            head_dim: HEAD_DIM,
            kv_seq_len,
            kv_capacity: self.capacity,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            mask_type: 0,
            sliding_window: 0,
            softcap: 0.0,
            ring_start: 0,
            scale_factor_d512: 1.0,
            codebook_bits: CODEBOOK_BITS,
            fuse_fwht_pre: 0,
            nsg: compute_nsg(kv_seq_len),
        }
    }

    fn kernel(&self, kv_seq_len: u32) -> TqHbRowKernel {
        if kv_seq_len >= self.shape.gqa_q2_min_kv {
            TqHbRowKernel::Gqa(GqaTile::Q2)
        } else {
            TqHbRowKernel::Scalar
        }
    }
}

type Encode = Box<dyn Fn(&mut CommandEncoder, &mut KernelRegistry, &MlxDevice) -> R<()>>;

/// One dispatch group of a layer. `barrier` precedes it in the fused run;
/// `dispatches` is what it adds to the kind's dispatch count.
struct Step {
    kind: &'static str,
    barrier: bool,
    dispatches: u32,
    encode: Encode,
}

fn step(kind: &'static str, barrier: bool, dispatches: u32, encode: Encode) -> Step {
    Step {
        kind,
        barrier,
        dispatches,
        encode,
    }
}

/// hf2q's per-row fallback (`encode_joint_full_attn_layer`, F32-control /
/// repeated-slot branch) for a TQ-only slot.
fn serial_route(dev: &MlxDevice, case: &Case) -> R<Vec<Step>> {
    let mut steps = Vec::new();
    let (nh, nkv) = (case.shape.n_heads, case.shape.n_kv);
    let (q_total, kv_total) = (case.q_total(), case.kv_total());
    let real = (case.lanes * ROWS) as usize;
    let attention = alloc(dev, DType::F32, case.width as usize * q_total)?;
    let gated = alloc(dev, DType::F32, case.width as usize * q_total)?;
    let gate_params = u32_buf(dev, &[(case.width as usize * q_total) as u32])?;
    let mut outputs = Vec::with_capacity(real);
    for row in 0..real {
        let lane = row as u32 / ROWS;
        let position = case.history + row as u32 % ROWS;
        let [k_packed, k_norms, v_packed, v_norms] = case.slot(lane);
        let q_row = case.q.slice_view((row * q_total * 4) as u64, q_total);
        let out = alloc(dev, DType::F32, q_total)?;
        let tmp = alloc(dev, DType::F32, tmp_buffer_bytes(nh, HEAD_DIM) / 4)?;
        let capacity = case.capacity;
        // The chain's barrier before its K encode also separates the rows.
        for (barrier, src, packed, norms) in [
            (true, &case.k, &k_packed, &k_norms),
            (false, &case.v, &v_packed, &v_norms),
        ] {
            let src = src.slice_view((row * kv_total * 4) as u64, kv_total);
            let (packed, norms) = (packed.clone(), norms.clone());
            steps.push(step(
                "kv_encode",
                barrier,
                1,
                Box::new(move |enc, reg, dev| {
                    dispatch_hadamard_quantize_kv_hb_seq(
                        enc,
                        reg,
                        dev.metal_device(),
                        &src,
                        &packed,
                        &norms,
                        nkv,
                        HEAD_DIM,
                        capacity,
                        position,
                        1,
                        0,
                        false,
                        1.0,
                        CODEBOOK_BITS,
                    )?;
                    Ok(())
                }),
            ));
        }
        let q = q_row.clone();
        steps.push(step(
            "q_rotate",
            true,
            1,
            Box::new(move |enc, reg, dev| {
                dispatch_fwht_sign_premult_f32(enc, reg, dev.metal_device(), &q, nh, HEAD_DIM)?;
                Ok(())
            }),
        ));
        let params = case.params(position + 1);
        let kernel = case.kernel(position + 1);
        let (q, o) = (q_row.clone(), out.clone());
        steps.push(step(
            "attention(main+reduce)",
            true,
            2,
            Box::new(move |enc, reg, dev| {
                flash_attn_vec_tq_hb_rows(
                    enc,
                    reg,
                    dev,
                    &[TqHbDecodeRow {
                        q: &q,
                        k_packed: &k_packed,
                        k_norms: &k_norms,
                        v_packed: &v_packed,
                        v_norms: &v_norms,
                        output: &o,
                        tmp: &tmp,
                        params,
                        kernel,
                    }],
                )?;
                Ok(())
            }),
        ));
        let o = out.clone();
        steps.push(step(
            "out_rotate",
            true,
            1,
            Box::new(move |enc, reg, dev| {
                dispatch_fwht_sign_undo_f32(enc, reg, dev.metal_device(), &o, nh, HEAD_DIM)?;
                Ok(())
            }),
        ));
        outputs.push(out);
    }
    for row in 0..case.width as usize {
        let src = outputs[if row < real { row } else { 0 }].clone();
        let dst = attention.clone();
        steps.push(step(
            "row_copy",
            row == 0,
            1,
            Box::new(move |enc, reg, dev| {
                dispatch_copy_f32(
                    enc,
                    reg,
                    dev.metal_device(),
                    &src,
                    &dst,
                    0,
                    row * q_total,
                    q_total,
                )?;
                Ok(())
            }),
        ));
    }
    let gate = case.gate.clone();
    let n = (case.width as usize * q_total) as u32;
    steps.push(step(
        "gate",
        true,
        1,
        Box::new(move |enc, reg, dev| {
            dispatch_sigmoid_mul(
                enc,
                reg,
                dev.metal_device(),
                &attention,
                &gate,
                &gated,
                &gate_params,
                n,
            )?;
            Ok(())
        }),
    ));
    Ok(steps)
}

/// The stage's unchanged prefix: per-head RMS norm of Q and K, then IMROPE
/// of both at the rows' positions (hf2q `apply_q_or_k_per_head_rms_norm`,
/// `apply_imrope`).
fn prep_route(dev: &MlxDevice, case: &Case) -> R<Vec<Step>> {
    let mut steps = Vec::new();
    let w = case.width;
    let (nh, nkv) = (case.shape.n_heads, case.shape.n_kv);
    let weight = f32_buf(dev, HEAD_DIM as usize, &mut Rng(7), 0.5, 1.5)?;
    let mut norm_params = alloc(dev, DType::F32, 2)?;
    norm_params
        .as_mut_slice::<f32>()?
        .copy_from_slice(&[1e-6, HEAD_DIM as f32]);
    let mut positions = alloc(dev, DType::I32, 4 * w as usize)?;
    for (i, v) in positions.as_mut_slice::<i32>()?.iter_mut().enumerate() {
        *v = (case.history + (i as u32 % w) % ROWS) as i32;
    }
    let q_normed = alloc(dev, DType::F32, w as usize * case.q_total())?;
    let k_normed = alloc(dev, DType::F32, w as usize * case.kv_total())?;
    let q_rope = alloc(dev, DType::F32, w as usize * case.q_total())?;
    let k_rope = alloc(dev, DType::F32, w as usize * case.kv_total())?;
    for (input, output, heads) in [(&case.q, &q_normed, nh), (&case.k, &k_normed, nkv)] {
        let (input, output) = (input.clone(), output.clone());
        let (weight, params) = (weight.clone(), norm_params.clone());
        steps.push(step(
            "head_norm",
            false,
            1,
            Box::new(move |enc, reg, dev| {
                dispatch_rms_norm(
                    enc,
                    reg,
                    dev.metal_device(),
                    &input,
                    &weight,
                    &output,
                    &params,
                    w * heads,
                    HEAD_DIM,
                )?;
                Ok(())
            }),
        ));
    }
    for (index, (input, output, heads)) in [(q_normed, q_rope, nh), (k_normed, k_rope, nkv)]
        .into_iter()
        .enumerate()
    {
        let positions = positions.clone();
        steps.push(step(
            "imrope",
            index == 0,
            1,
            Box::new(move |enc, reg, dev| {
                let params = RopeMultiParams {
                    head_dim: HEAD_DIM,
                    rope_dim: 64,
                    n_heads: heads,
                    seq_len: w,
                    freq_base: 10_000_000.0,
                    mode: RopeMultiMode::Imrope,
                    sections: [11, 11, 10, 0],
                };
                dispatch_rope_multi_cached(enc, reg, dev, &input, &output, &positions, params)?;
                Ok(())
            }),
        ));
    }
    Ok(steps)
}

/// The two batched K/V encodes of every window row at its position.
fn batched_encode(dev: &MlxDevice, case: &Case, steps: &mut Vec<Step>) -> R<()> {
    let real = case.lanes * ROWS;
    let slot_ids: Vec<u32> = (0..real).map(|row| row / ROWS).collect();
    let positions: Vec<u32> = (0..real).map(|row| case.history + row % ROWS).collect();
    let slot_ids = u32_buf(dev, &slot_ids)?;
    let positions = u32_buf(dev, &positions)?;
    let (nkv, capacity) = (case.shape.n_kv, case.capacity);
    for (src, packed, norms) in [
        (&case.k, &case.k_packed, &case.k_norms),
        (&case.v, &case.v_packed, &case.v_norms),
    ] {
        let (src, packed, norms) = (src.clone(), packed.clone(), norms.clone());
        let (slot_ids, positions) = (slot_ids.clone(), positions.clone());
        steps.push(step(
            "kv_encode",
            false,
            1,
            Box::new(move |enc, reg, dev| {
                dispatch_hadamard_quantize_kv_hb_batched(
                    enc,
                    reg,
                    dev.metal_device(),
                    &src,
                    &packed,
                    &norms,
                    &slot_ids,
                    &positions,
                    real,
                    nkv,
                    HEAD_DIM,
                    capacity,
                    false,
                    1.0,
                    CODEBOOK_BITS,
                )?;
                Ok(())
            }),
        ));
    }
    Ok(())
}

/// Padding copies and the fused un-rotation + gate over all packed rows.
fn undo_gate(dev: &MlxDevice, case: &Case, attention: &MlxBuffer, steps: &mut Vec<Step>) -> R<()> {
    let q_total = case.q_total();
    let real = (case.lanes * ROWS) as usize;
    {
        for row in real..case.width as usize {
            let a = attention.clone();
            steps.push(step(
                "row_copy",
                row == real,
                1,
                Box::new(move |enc, reg, dev| {
                    dispatch_copy_f32(
                        enc,
                        reg,
                        dev.metal_device(),
                        &a,
                        &a,
                        0,
                        row * q_total,
                        q_total,
                    )?;
                    Ok(())
                }),
            ));
        }
    }
    let gated = alloc(dev, DType::F32, case.width as usize * q_total)?;
    let (a, gate) = (attention.clone(), case.gate.clone());
    let heads = case.width * case.shape.n_heads;
    steps.push(step(
        "out_rotate+gate",
        true,
        1,
        Box::new(move |enc, reg, dev| {
            dispatch_fwht_sign_undo_sigmoid_mul_f32(
                enc,
                reg,
                dev.metal_device(),
                &a,
                &gate,
                &gated,
                heads,
                HEAD_DIM,
            )?;
            Ok(())
        }),
    ));
    Ok(())
}

/// Row-batched per-row kernels: bitwise the serial chain, two barriers.
fn rows_route(dev: &MlxDevice, case: &Case) -> R<Vec<Step>> {
    let mut steps = Vec::new();
    batched_encode(dev, case, &mut steps)?;
    let nh = case.shape.n_heads;
    let q_total = case.q_total();
    let real = (case.lanes * ROWS) as usize;
    let q = case.q.clone();
    let heads = real as u32 * nh;
    steps.push(step(
        "q_rotate",
        false,
        1,
        Box::new(move |enc, reg, dev| {
            dispatch_fwht_sign_premult_f32(enc, reg, dev.metal_device(), &q, heads, HEAD_DIM)?;
            Ok(())
        }),
    ));
    let attention = alloc(dev, DType::F32, case.width as usize * q_total)?;
    struct Row {
        q: MlxBuffer,
        out: MlxBuffer,
        tmp: MlxBuffer,
        cache: [MlxBuffer; 4],
        params: FlashAttnVecTqHbParams,
        kernel: TqHbRowKernel,
    }
    let mut rows = Vec::with_capacity(real);
    for row in 0..real {
        let kv = case.history + row as u32 % ROWS + 1;
        rows.push(Row {
            q: case.q.slice_view((row * q_total * 4) as u64, q_total),
            out: attention.slice_view((row * q_total * 4) as u64, q_total),
            tmp: alloc(dev, DType::F32, tmp_buffer_bytes(nh, HEAD_DIM) / 4)?,
            cache: case.slot(row as u32 / ROWS),
            params: case.params(kv),
            kernel: case.kernel(kv),
        });
    }
    steps.push(step(
        "attention(main+reduce)",
        true,
        2 * real as u32,
        Box::new(move |enc, reg, dev| {
            let bound: Vec<TqHbDecodeRow<'_>> = rows
                .iter()
                .map(|row| TqHbDecodeRow {
                    q: &row.q,
                    k_packed: &row.cache[0],
                    k_norms: &row.cache[1],
                    v_packed: &row.cache[2],
                    v_norms: &row.cache[3],
                    output: &row.out,
                    tmp: &row.tmp,
                    params: row.params,
                    kernel: row.kernel,
                })
                .collect();
            flash_attn_vec_tq_hb_rows(enc, reg, dev, &bound)?;
            Ok(())
        }),
    ));
    undo_gate(dev, case, &attention, &mut steps)?;
    Ok(steps)
}

/// The tiled verify attention.
fn tile_route(dev: &MlxDevice, case: &Case) -> R<Vec<Step>> {
    let mut steps = Vec::new();
    batched_encode(dev, case, &mut steps)?;
    let shape = TqHbVerifyTileShape {
        n_heads: case.shape.n_heads,
        n_kv_heads: case.shape.n_kv,
        head_dim: HEAD_DIM,
        kv_capacity: case.capacity,
        codebook_bits: CODEBOOK_BITS,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        width: case.width,
    };
    let lanes: Vec<TqHbVerifyLane> = (0..case.lanes)
        .map(|lane| TqHbVerifyLane {
            slot: lane,
            committed: case.history,
            rows: ROWS,
            first_row: lane * ROWS,
        })
        .collect();
    let sizes = verify_tile_scratch_bytes(
        &shape,
        lanes.len(),
        verify_tile_slot_splits(&lanes, case.capacity),
    );
    let queries = alloc(dev, DType::F16, sizes.queries / 2)?;
    let partials = alloc(dev, DType::F32, sizes.partials / 4)?;
    let statistics = alloc(dev, DType::F32, sizes.statistics / 4)?;
    let attention = alloc(dev, DType::F32, sizes.attention / 4)?;
    let gated = alloc(dev, DType::F32, case.width as usize * case.q_total())?;
    let (q, gate) = (case.q.clone(), case.gate.clone());
    let cache = [
        case.k_packed.clone(),
        case.k_norms.clone(),
        case.v_packed.clone(),
        case.v_norms.clone(),
    ];
    steps.push(step(
        "tile(pack+split+reduce+gate)",
        false,
        4,
        Box::new(move |enc, reg, dev| {
            tq_hb_verify_tile_attention(
                enc,
                reg,
                dev,
                &q,
                &gate,
                &TqHbVerifyTileCache {
                    k_packed: &cache[0],
                    k_norms: &cache[1],
                    v_packed: &cache[2],
                    v_norms: &cache[3],
                },
                &gated,
                &TqHbVerifyTileScratch {
                    queries: &queries,
                    partials: &partials,
                    statistics: &statistics,
                    attention: &attention,
                },
                &shape,
                &lanes,
            )?;
            Ok(())
        }),
    ));
    Ok(steps)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn gpu_ms(enc: &mut CommandEncoder) -> R<f64> {
    let (start, end) = enc.commit_wait_with_gpu_time()?;
    Ok((end - start) * 1e3)
}

struct Measured {
    fused_ms: f64,
    fused_min_ms: f64,
    dispatches: u32,
    kinds: BTreeMap<&'static str, (u32, f64)>,
}

/// GPU time of `measured` (its own command buffer) right behind a filler
/// command buffer that repeats the whole layer `FILLER_LAYERS` times. The
/// filler keeps the GPU busy between samples, as a serving cycle does, so
/// the sample runs at the GPU's full clocks: a lone sub-millisecond command
/// buffer per host round trip leaves the GPU in a low performance state and
/// reads two to three times slower.
fn hot_gpu_ms(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    steps: &[Step],
    measured: &[&Step],
) -> R<f64> {
    let mut filler = dev.command_encoder()?;
    for _ in 0..FILLER_LAYERS {
        filler.memory_barrier();
        for step in steps {
            if step.barrier {
                filler.memory_barrier();
            }
            (step.encode)(&mut filler, reg, dev)?;
        }
    }
    filler.commit();
    let mut enc = dev.command_encoder()?;
    for step in measured {
        if step.barrier {
            enc.memory_barrier();
        }
        (step.encode)(&mut enc, reg, dev)?;
    }
    let ms = gpu_ms(&mut enc)?;
    filler.wait_until_completed()?;
    Ok(ms)
}

fn measure(
    dev: &MlxDevice,
    reg: &mut KernelRegistry,
    steps: &[Step],
    repeat: usize,
) -> R<Measured> {
    let all: Vec<&Step> = steps.iter().collect();
    let warm = std::time::Instant::now();
    while warm.elapsed().as_millis() < WARMUP_MS {
        hot_gpu_ms(dev, reg, steps, &all)?;
    }
    let mut fused = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        fused.push(hot_gpu_ms(dev, reg, steps, &all)?);
    }
    let mut sums: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut counts: BTreeMap<&'static str, u32> = BTreeMap::new();
    for step in steps {
        *counts.entry(step.kind).or_default() += step.dispatches;
    }
    for _ in 0..repeat.min(SEPARATE_REPEAT) {
        let mut per_kind: BTreeMap<&'static str, f64> = BTreeMap::new();
        for step in steps {
            *per_kind.entry(step.kind).or_default() += hot_gpu_ms(dev, reg, steps, &[step])?;
        }
        for (kind, ms) in per_kind {
            sums.entry(kind).or_default().push(ms);
        }
    }
    let kinds = sums
        .into_iter()
        .map(|(kind, mut values)| (kind, (counts[kind], median(&mut values))))
        .collect();
    Ok(Measured {
        fused_min_ms: fused.iter().copied().fold(f64::INFINITY, f64::min),
        fused_ms: median(&mut fused),
        dispatches: counts.values().sum(),
        kinds,
    })
}

fn list<T: std::str::FromStr>(args: &[String], flag: &str, default: &[T]) -> Vec<T>
where
    T: Clone,
{
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn main() -> R<()> {
    let args: Vec<String> = std::env::args().collect();
    let shapes: Vec<String> = list(
        &args,
        "--shapes",
        &["qwen38".to_string(), "qwen36".to_string()],
    );
    let lanes: Vec<u32> = list(&args, "--lanes", &[1, 2, 4]);
    let histories: Vec<u32> = list(&args, "--histories", &[512, 2048, 8192]);
    let routes: Vec<String> = list(
        &args,
        "--routes",
        &[
            "prep".to_string(),
            "serial".to_string(),
            "rows".to_string(),
            "tile".to_string(),
        ],
    );
    let repeat = list(&args, "--repeat", &[20usize])[0];
    let json_path: Option<String> = list::<String>(&args, "--json", &[]).into_iter().next();

    let dev = MlxDevice::new()?;
    let mut reg = KernelRegistry::new();
    println!(
        "device {}, one full-attention layer, TQ-HB {CODEBOOK_BITS}-bit KV, {SLOTS} slots, {ROWS} rows per lane, median of {repeat}",
        dev.name()
    );
    let mut records = Vec::new();
    for shape in SHAPES.iter().filter(|s| shapes.iter().any(|n| n == s.name)) {
        for &history in &histories {
            for &lane_count in &lanes {
                let mut rng =
                    Rng(0x9E37_79B9_7F4A_7C15 ^ u64::from(history) << 8 ^ u64::from(lane_count));
                let case = Case::new(&dev, *shape, lane_count, history, &mut rng)?;
                for route in &routes {
                    let steps = match route.as_str() {
                        "prep" => prep_route(&dev, &case)?,
                        "serial" => serial_route(&dev, &case)?,
                        "rows" => rows_route(&dev, &case)?,
                        "tile" => {
                            let probe = TqHbVerifyTileShape {
                                n_heads: shape.n_heads,
                                n_kv_heads: shape.n_kv,
                                head_dim: HEAD_DIM,
                                kv_capacity: case.capacity,
                                codebook_bits: CODEBOOK_BITS,
                                scale: 0.0625,
                                width: case.width,
                            };
                            if !tq_hb_verify_tile_available(&mut reg, &dev, &probe)? {
                                println!(
                                    "{} tile: RECORDED SKIP (no Metal tensor API)",
                                    shape.name
                                );
                                continue;
                            }
                            tile_route(&dev, &case)?
                        }
                        other => return Err(format!("unknown route {other}").into()),
                    };
                    let m = measure(&dev, &mut reg, &steps, repeat)?;
                    let kinds: Vec<String> = m
                        .kinds
                        .iter()
                        .map(|(kind, (count, ms))| format!("{kind} x{count} {ms:.4}"))
                        .collect();
                    println!(
                        "{:<7} {}/{}/256 history={:<5} lanes={} width={:<2} {:<6} fused={:.4} ms (min {:.4}) dispatches={:<3} separate: {}",
                        shape.name,
                        shape.n_heads,
                        shape.n_kv,
                        history,
                        lane_count,
                        case.width,
                        route,
                        m.fused_ms,
                        m.fused_min_ms,
                        m.dispatches,
                        kinds.join(", ")
                    );
                    records.push(format!(
                        "{{\"shape\":\"{}\",\"history\":{},\"lanes\":{},\"width\":{},\"route\":\"{}\",\"fused_ms\":{:.6},\"fused_min_ms\":{:.6},\"dispatches\":{},\"kinds\":{{{}}}}}",
                        shape.name,
                        history,
                        lane_count,
                        case.width,
                        route,
                        m.fused_ms,
                        m.fused_min_ms,
                        m.dispatches,
                        m.kinds
                            .iter()
                            .map(|(kind, (count, ms))| format!("\"{kind}\":{{\"count\":{count},\"ms\":{ms:.6}}}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    ));
                }
            }
        }
    }
    if let Some(path) = json_path {
        std::fs::write(&path, format!("[\n{}\n]\n", records.join(",\n")))?;
        println!("wrote {path}");
    }
    Ok(())
}
