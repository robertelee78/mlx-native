//! Gates of the tiled TQ-HB verify attention (`ops::tq_hb_verify_tile`).
//!
//! The route differs in bits from the per-row decode kernel (Class B), so it
//! carries its own invariants at the served Qwen shapes (24 query / 4 KV
//! heads and 16 / 2, head dim 256, 8-bit codes, 8 rows per lane, 1-4 lanes):
//!
//! 1. F64 reference with a K-scaled bound, on the rotated attention (the
//!    reduce's output). The kernel's matmul operands are half: the rotated
//!    queries, the centroids, and each probability times its token's value
//!    norm; scores and both accumulations are F32. The reference evaluates
//!    the same attention in F64 over the half queries and centroids and
//!    bounds the kernel by its roundings: `(256 + 8) * 2^-24` per score
//!    relative to `sum |q||k|`; per output, relative to `sum p|v| / sum p`,
//!    the propagated score bound, `2^-11` for the half probability operand
//!    and `(visible tokens + pages + splits) * 2^-23` for the F32
//!    accumulation, plus `2^-25 |v|` for every token whose operand is below
//!    the half normal range. The pack is checked separately against an F64
//!    sign-Hadamard transform to half precision.
//! 2. Distance to the cache's F32 definition (F32 centroids, unrounded
//!    queries): bounded by the two half roundings, `2^-10` per score, and,
//!    as a regression gate that does not loosen with the norms, a relative
//!    RMS error of every (row, head) and of every case below fixed limits.
//! 3. The un-rotation + gate dispatch is bitwise
//!    `fwht_sign_undo_sigmoid_mul_f32` on the same rotated rows.
//! 4. Determinism: repeated runs are bitwise equal. A value norm beyond the
//!    half operand's range saturates and stays finite.
//! 5. Row and cohort independence, bitwise: a row's output does not depend
//!    on the other lanes, the lane order, the packed row it occupies, the
//!    padding, the window's later rows (present or not) or cache bytes at or
//!    after its causal end.
//! 6. Window independence, bitwise: a token's output does not depend on the
//!    window that verifies it (its start, the token's row in it, its length).
//!
//! A window whose rows ask for two split lengths runs as two tiles (a cut
//! window; the first is the one that crosses 1,536 tokens). Every invariant
//! above has such windows among its fixtures.

#![cfg(target_vendor = "apple")]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_arguments)]

use mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_sigmoid_mul_f32;
use mlx_native::ops::tq_hb_verify_tile::{
    tq_hb_verify_tile_attention, tq_hb_verify_tile_available, verify_tile_pages_per_split,
    verify_tile_scratch_bytes, verify_tile_slot_splits, TqHbVerifyLane, TqHbVerifyTileCache,
    TqHbVerifyTileScratch, TqHbVerifyTileShape, MAX_SPLITS, PAGE_TOKENS, VERIFY_TILE_ROWS,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

const HD: usize = 256;
const SLOTS: u32 = 4;
const SCALE: f32 = 0.0625;

/// The 256 float literals of a shader's codebook table named `table`.
fn codebook_table(source: &str, table: &str) -> Vec<f32> {
    let start = source.find(table).expect("codebook table");
    let body = &source[start..start + source[start..].find("};").expect("codebook end")];
    let body = &body[body.find('{').expect("codebook body")..];
    let values: Vec<f32> = body
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .filter(|cell| cell.contains('.'))
        .map(|cell| cell.parse::<f32>().expect("centroid"))
        .collect();
    assert_eq!(values.len(), 256, "{table}");
    values
}

/// The tile shader's centroids, which must be the per-row kernel's
/// (`CODEBOOK_HB_8BIT` of `shaders/flash_attn_vec_tq_hb.metal`).
fn codebook() -> Vec<f32> {
    let tile = codebook_table(
        mlx_native::ops::tq_hb_verify_tile::TQ_HB_VERIFY_TILE_SHADER_SOURCE,
        "TQV_CODEBOOK_8BIT[256]",
    );
    let per_row = codebook_table(
        mlx_native::ops::flash_attn_vec_tq_hb::FLASH_ATTN_VEC_TQ_HB_SHADER_SOURCE,
        "CODEBOOK_HB_8BIT[256]",
    );
    assert_eq!(
        tile, per_row,
        "tile codebook differs from the TQ-HB codebook"
    );
    tile
}

/// `TBQ_SIGNS_256` of `shaders/fwht_standalone.metal`.
const SIGNS: [u8; 32] = [
    0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e, 0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
    0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d, 0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
];

struct Xor(u64);

impl Xor {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * 0.5 * (self.unit() + 1.0)
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    n_heads: u32,
    n_kv: u32,
}

const QWEN38: Geometry = Geometry {
    n_heads: 24,
    n_kv: 4,
};
const QWEN36: Geometry = Geometry {
    n_heads: 16,
    n_kv: 2,
};

impl Geometry {
    fn group(self) -> usize {
        (self.n_heads / self.n_kv) as usize
    }

    fn shape(self, capacity: u32, width: u32) -> TqHbVerifyTileShape {
        TqHbVerifyTileShape {
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv,
            head_dim: HD as u32,
            kv_capacity: capacity,
            codebook_bits: 8,
            scale: SCALE,
            width,
        }
    }
}

/// CPU copy of a layer's multi-slot TQ-HB cache.
#[derive(Clone)]
struct Cache {
    geometry: Geometry,
    capacity: u32,
    k_codes: Vec<u8>,
    k_norms: Vec<f32>,
    v_codes: Vec<u8>,
    v_norms: Vec<f32>,
}

impl Cache {
    /// Random codes everywhere; norms in `norms` (K) and `[0.25, 4]` (V).
    fn random(geometry: Geometry, capacity: u32, norms: (f32, f32), rng: &mut Xor) -> Self {
        let tokens = (SLOTS * geometry.n_kv * capacity) as usize;
        let codes = |rng: &mut Xor| {
            let mut bytes = vec![0u8; tokens * HD];
            for chunk in bytes.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rng.next().to_le_bytes());
            }
            bytes
        };
        let k_codes = codes(rng);
        let v_codes = codes(rng);
        Self {
            geometry,
            capacity,
            k_codes,
            k_norms: (0..tokens).map(|_| rng.range(norms.0, norms.1)).collect(),
            v_codes,
            v_norms: (0..tokens).map(|_| rng.range(0.25, 4.0)).collect(),
        }
    }

    /// Index of `(slot, kv_head, token)` in the norm arrays.
    fn token(&self, slot: u32, kv_head: usize, token: usize) -> usize {
        (slot as usize * self.geometry.n_kv as usize + kv_head) * self.capacity as usize + token
    }

    /// Overwrite every token of `slot` from `from` on with new random content.
    fn scramble_from(&mut self, slot: u32, from: u32, rng: &mut Xor) {
        for kv_head in 0..self.geometry.n_kv as usize {
            for token in from as usize..self.capacity as usize {
                let index = self.token(slot, kv_head, token);
                self.k_norms[index] = rng.range(-1e30, 1e30);
                self.v_norms[index] = rng.range(-1e30, 1e30);
                for d in 0..HD {
                    self.k_codes[index * HD + d] = rng.next() as u8;
                    self.v_codes[index * HD + d] = rng.next() as u8;
                }
            }
        }
    }
}

fn buffer<T: bytemuck::Pod>(device: &MlxDevice, dtype: DType, values: &[T]) -> MlxBuffer {
    let bytes = std::mem::size_of_val(values);
    let mut buffer = device
        .alloc_buffer(bytes.max(16), dtype, vec![values.len()])
        .expect("allocate");
    buffer.as_mut_slice::<T>().expect("map")[..values.len()].copy_from_slice(values);
    buffer
}

struct Gpu {
    device: MlxDevice,
    registry: KernelRegistry,
}

/// The device and registry, or `None` (recorded) without the tensor API.
fn gpu() -> Option<Gpu> {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    for geometry in [QWEN38, QWEN36] {
        let available = tq_hb_verify_tile_available(&mut registry, &device, &geometry.shape(64, 8))
            .expect("probe");
        if !available {
            eprintln!("RECORDED SKIP: tq_hb_verify_tile unavailable (no Metal tensor API)");
            return None;
        }
    }
    Some(Gpu { device, registry })
}

struct TileRun {
    /// Gated attention `[width, n_heads, 256]`.
    output: Vec<f32>,
    /// Rotated attention `[lane, row, head, 256]` (the reduce's output).
    attention: Vec<f32>,
    /// Rotated half queries `[lane, kv_head, row, head in group, 256]`.
    queries: Vec<half::f16>,
}

/// Run the op once over NaN-filled output and scratch.
fn run_tile(
    gpu: &mut Gpu,
    cache: &Cache,
    q: &[f32],
    gate: &[f32],
    lanes: &[TqHbVerifyLane],
    width: u32,
) -> TileRun {
    let geometry = cache.geometry;
    let shape = geometry.shape(cache.capacity, width);
    let q_total = geometry.n_heads as usize * HD;
    assert_eq!(q.len(), width as usize * q_total);
    assert_eq!(gate.len(), q.len());
    let device = &gpu.device;
    let sizes = verify_tile_scratch_bytes(
        &shape,
        lanes.len(),
        verify_tile_slot_splits(lanes, cache.capacity),
    );
    let k_packed = buffer(device, DType::U8, &cache.k_codes);
    let k_norms = buffer(device, DType::F32, &cache.k_norms);
    let v_packed = buffer(device, DType::U8, &cache.v_codes);
    let v_norms = buffer(device, DType::F32, &cache.v_norms);
    let q_rows = buffer(device, DType::F32, q);
    let gate_rows = buffer(device, DType::F32, gate);
    let output = buffer(device, DType::F32, &vec![f32::NAN; q.len()]);
    let queries = buffer(device, DType::F16, &vec![half::f16::NAN; sizes.queries / 2]);
    let partials = buffer(device, DType::F32, &vec![f32::NAN; sizes.partials / 4]);
    let statistics = buffer(device, DType::F32, &vec![f32::NAN; sizes.statistics / 4]);
    let attention = buffer(device, DType::F32, &vec![f32::NAN; sizes.attention / 4]);
    let mut encoder = device.command_encoder().expect("encoder");
    tq_hb_verify_tile_attention(
        &mut encoder,
        &mut gpu.registry,
        device,
        &q_rows,
        &gate_rows,
        &TqHbVerifyTileCache {
            k_packed: &k_packed,
            k_norms: &k_norms,
            v_packed: &v_packed,
            v_norms: &v_norms,
        },
        &output,
        &TqHbVerifyTileScratch {
            queries: &queries,
            partials: &partials,
            statistics: &statistics,
            attention: &attention,
        },
        &shape,
        lanes,
    )
    .expect("encode tile attention");
    encoder.commit_and_wait().expect("commit");
    TileRun {
        output: output.as_slice::<f32>().expect("output")[..q.len()].to_vec(),
        attention: attention.as_slice::<f32>().expect("attention")[..sizes.attention / 4].to_vec(),
        queries: queries.as_slice::<half::f16>().expect("queries")[..sizes.queries / 2].to_vec(),
    }
}

/// `fwht_sign_undo_sigmoid_mul_f32` of rotated rows `[rows, n_heads, 256]`.
fn stock_undo_gate(gpu: &mut Gpu, rotated: &[f32], gate: &[f32]) -> Vec<f32> {
    let device = &gpu.device;
    let data = buffer(device, DType::F32, rotated);
    let gate_rows = buffer(device, DType::F32, gate);
    let output = buffer(device, DType::F32, &vec![f32::NAN; rotated.len()]);
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_fwht_sign_undo_sigmoid_mul_f32(
        &mut encoder,
        &mut gpu.registry,
        device.metal_device(),
        &data,
        &gate_rows,
        &output,
        (rotated.len() / HD) as u32,
        HD as u32,
    )
    .expect("encode stock undo + gate");
    encoder.commit_and_wait().expect("commit");
    output.as_slice::<f32>().expect("output")[..rotated.len()].to_vec()
}

/// The rotated attention `run` holds for each packed row (padding repeats
/// the first lane's first row), `[width, n_heads, 256]`.
fn packed_rotated(
    run: &TileRun,
    geometry: Geometry,
    lanes: &[TqHbVerifyLane],
    width: u32,
) -> Vec<f32> {
    let q_total = geometry.n_heads as usize * HD;
    let mut source = vec![0usize; width as usize];
    for (index, lane) in lanes.iter().enumerate() {
        for row in 0..lane.rows as usize {
            source[lane.first_row as usize + row] = index * VERIFY_TILE_ROWS as usize + row;
        }
    }
    source
        .iter()
        .flat_map(|&row| {
            run.attention[row * q_total..(row + 1) * q_total]
                .iter()
                .copied()
        })
        .collect()
}

/// The gated output must be the stock un-rotation + gate of the rotated rows.
fn check_undo_gate(
    gpu: &mut Gpu,
    run: &TileRun,
    geometry: Geometry,
    gate: &[f32],
    lanes: &[TqHbVerifyLane],
    width: u32,
    label: &str,
) {
    let rotated = packed_rotated(run, geometry, lanes, width);
    let want = stock_undo_gate(gpu, &rotated, gate);
    assert_eq!(
        bits(&run.output),
        bits(&want),
        "{label}: un-rotation + gate differs from the stock kernel"
    );
}

/// F64 sign-Hadamard rotation of one head: `H (s ⊙ q) / 16`.
fn rotate(q: &[f32]) -> Vec<f64> {
    let mut x: Vec<f64> = q
        .iter()
        .enumerate()
        .map(|(j, &v)| {
            let negative = (SIGNS[j >> 3] >> (j & 7)) & 1 == 1;
            if negative {
                -f64::from(v)
            } else {
                f64::from(v)
            }
        })
        .collect();
    let mut h = 1;
    while h < HD {
        for i in (0..HD).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    x.iter().map(|v| v / 16.0).collect()
}

/// The reference of one (row, head): output in the rotated domain, its
/// per-dimension bound scale `sum p |v| / sum p`, the per-dimension absolute
/// bound of the half probability operands below the normal range, and the
/// largest `sum |q||k|`-scaled score magnitude among the visible tokens.
struct Reference {
    output: Vec<f64>,
    magnitude: Vec<f64>,
    subnormal: Vec<f64>,
    score_magnitude: f64,
}

/// Below this a half operand may be subnormal on the device (smallest normal
/// half is `2^-14`; the margin covers the device's own F32 probability).
const HALF_NORMAL_FLOOR: f64 = 1.0 / (1u64 << 13) as f64;
/// Spacing of subnormal halves: the absolute rounding error there is half.
const HALF_SUBNORMAL_ULP: f64 = 1.0 / (1u64 << 24) as f64;

/// Attention of query `q` (rotated, any precision) over the first `visible`
/// tokens of `(slot, kv_head)` with centroids `centroids`.
fn reference(
    cache: &Cache,
    centroids: &[f64],
    q: &[f64],
    slot: u32,
    kv_head: usize,
    visible: usize,
) -> Reference {
    let mut scores = Vec::with_capacity(visible);
    let mut score_magnitude = 0.0f64;
    for token in 0..visible {
        let index = cache.token(slot, kv_head, token);
        let codes = &cache.k_codes[index * HD..(index + 1) * HD];
        let (mut dot, mut abs) = (0.0f64, 0.0f64);
        for d in 0..HD {
            let term = q[d] * centroids[codes[d] as usize];
            dot += term;
            abs += term.abs();
        }
        let factor = f64::from(cache.k_norms[index]) / 16.0 * f64::from(SCALE);
        scores.push(dot * factor);
        score_magnitude = score_magnitude.max(abs * factor.abs());
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut output = vec![0.0f64; HD];
    let mut magnitude = vec![0.0f64; HD];
    let mut subnormal = vec![0.0f64; HD];
    let mut sum = 0.0f64;
    for (token, score) in scores.iter().enumerate() {
        let p = (score - maximum).exp();
        sum += p;
        let index = cache.token(slot, kv_head, token);
        let weight = p * f64::from(cache.v_norms[index]) / 16.0;
        assert!(
            weight < 65_504.0,
            "fixture value norm saturates the half operand"
        );
        // The device's probability is relative to a running maximum, so its
        // operand is at least this one: normal here means normal there.
        let below = weight < HALF_NORMAL_FLOOR;
        let codes = &cache.v_codes[index * HD..(index + 1) * HD];
        for d in 0..HD {
            let centroid = centroids[codes[d] as usize];
            let value = weight * centroid;
            output[d] += value;
            magnitude[d] += value.abs();
            if below {
                subnormal[d] += 0.5 * HALF_SUBNORMAL_ULP * centroid.abs();
            }
        }
    }
    for d in 0..HD {
        output[d] /= sum;
        magnitude[d] /= sum;
        subnormal[d] /= sum;
    }
    Reference {
        output,
        magnitude,
        subnormal,
        score_magnitude,
    }
}

/// Regression limits of the relative RMS error against the F32 definition:
/// one case pooled over its rows and heads, and its worst (row, head).
#[derive(Clone, Copy)]
struct RmsLimits {
    pooled: f64,
    head: f64,
}

/// Key norms of 4 to 24 (the stored norms of the served models are 16 to 29):
/// measured 3.4e-4 to 7.1e-4 pooled and 7.9e-4 to 1.4e-3 in the worst head.
const SERVED_RMS: RmsLimits = RmsLimits {
    pooled: 1.2e-3,
    head: 2.5e-3,
};
/// Key norms of 100 to 400, a softmax dominated by a few tokens: measured
/// 1.7e-3 to 1.8e-3 pooled and 1.5e-2 to 2.0e-2 in the worst head.
const PEAKED_RMS: RmsLimits = RmsLimits {
    pooled: 3e-3,
    head: 3e-2,
};

/// Relative accuracy assumed of the device `exp` (fast math).
const EXP_RELATIVE: f64 = 1.0 / (1u64 << 19) as f64;

/// Check `run` against the F64 reference for `heads` of every window row.
/// Returns the largest observed error / bound ratio.
fn check_reference(
    cache: &Cache,
    q: &[f32],
    lanes: &[TqHbVerifyLane],
    run: &TileRun,
    heads: &[usize],
    limits: RmsLimits,
    label: &str,
) -> f64 {
    let geometry = cache.geometry;
    let (nh, group) = (geometry.n_heads as usize, geometry.group());
    let book = codebook();
    let half_centroids: Vec<f64> = book
        .iter()
        .map(|&c| f64::from(half::f16::from_f32(c).to_f32()))
        .collect();
    let f32_centroids: Vec<f64> = book.iter().map(|&c| f64::from(c)).collect();
    let eps = 2f64.powi(-24);
    let half_eps = 2f64.powi(-11);
    let mut worst = 0.0f64;
    let mut worst_definition = 0.0f64;
    let (mut pooled_error, mut pooled_signal, mut worst_head) = (0.0f64, 0.0f64, 0.0f64);
    for (lane_index, lane) in lanes.iter().enumerate() {
        let splits = verify_tile_slot_splits(&[*lane], cache.capacity) as usize;
        for row in 0..lane.rows as usize {
            let packed = lane.first_row as usize + row;
            let visible = lane.committed as usize + row + 1;
            let pages = visible.div_ceil(PAGE_TOKENS as usize);
            for &head in heads {
                let kv_head = head / group;
                let q_f32 = &q[(packed * nh + head) * HD..(packed * nh + head + 1) * HD];
                let exact = rotate(q_f32);
                let tile = ((lane_index * geometry.n_kv as usize + kv_head)
                    * VERIFY_TILE_ROWS as usize
                    + row)
                    * group
                    + head % group;
                let packed_q: Vec<f64> = run.queries[tile * HD..(tile + 1) * HD]
                    .iter()
                    .map(|v| f64::from(v.to_f32()))
                    .collect();
                // (1a) The pack: F32 butterflies then one half rounding.
                let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for d in 0..HD {
                    let bound = half_eps * exact[d].abs() + 16.0 * eps * scale + 1e-7;
                    assert!(
                        (packed_q[d] - exact[d]).abs() <= bound,
                        "{label}: pack lane {lane_index} row {row} head {head} dim {d}: {} vs {} (bound {bound:e})",
                        packed_q[d],
                        exact[d]
                    );
                }
                let rotated_row = lane_index * VERIFY_TILE_ROWS as usize + row;
                let got = &run.attention
                    [(rotated_row * nh + head) * HD..(rotated_row * nh + head + 1) * HD];
                // (1b) Same operands in F64; the bound is the F32 accumulation.
                let same = reference(
                    cache,
                    &half_centroids,
                    &packed_q,
                    lane.slot,
                    kv_head,
                    visible,
                );
                let delta = (HD as f64 + 8.0) * eps * same.score_magnitude;
                let relative = 2.0 * ((2.0 * delta).exp() - 1.0)
                    + 4.0 * EXP_RELATIVE
                    + half_eps
                    + (visible + pages + splits + 16) as f64 * 2.0 * eps;
                // (2) The cache's F32 definition: two half roundings per score
                // term, one per value term, on top of (1b).
                let definition =
                    reference(cache, &f32_centroids, &exact, lane.slot, kv_head, visible);
                let delta_definition =
                    delta + (2.0 * half_eps + 32.0 * eps) * definition.score_magnitude;
                // The value centroids' half rounding adds one more `2^-11`.
                let relative_definition =
                    2.0 * ((2.0 * delta_definition).exp() - 1.0) + half_eps + relative;
                for d in 0..HD {
                    assert!(got[d].is_finite(), "{label}: non-finite output");
                    let error = (f64::from(got[d]) - same.output[d]).abs();
                    let bound = relative * same.magnitude[d] + same.subnormal[d] + 1e-30;
                    assert!(
                        error <= bound,
                        "{label}: lane {lane_index} row {row} head {head} dim {d}: gpu {} ref {} error {error:e} > bound {bound:e}",
                        got[d],
                        same.output[d]
                    );
                    worst = worst.max(error / bound);
                    let error = (f64::from(got[d]) - definition.output[d]).abs();
                    let bound = relative_definition * definition.magnitude[d]
                        + definition.subnormal[d]
                        + 1e-30;
                    assert!(
                        error <= bound,
                        "{label}: lane {lane_index} row {row} head {head} dim {d}: gpu {} F32-definition {} error {error:e} > bound {bound:e}",
                        got[d],
                        definition.output[d]
                    );
                    worst_definition = worst_definition.max(error / bound);
                }
                let error: f64 = (0..HD)
                    .map(|d| (f64::from(got[d]) - definition.output[d]).powi(2))
                    .sum();
                let signal: f64 = definition.output.iter().map(|v| v * v).sum();
                pooled_error += error;
                pooled_signal += signal;
                worst_head = worst_head.max((error / signal).sqrt());
            }
        }
    }
    let pooled = (pooled_error / pooled_signal).sqrt();
    eprintln!(
        "{label}: max error/bound {worst:.4} (same operands), {worst_definition:.4} (F32 definition); \
         relative RMS against the F32 definition {pooled:.3e} pooled, {worst_head:.3e} worst head"
    );
    assert!(
        pooled <= limits.pooled && worst_head <= limits.head,
        "{label}: relative RMS against the F32 definition {pooled:.3e} pooled, {worst_head:.3e} worst head"
    );
    worst
}

fn random_q(geometry: Geometry, width: u32, rng: &mut Xor) -> Vec<f32> {
    (0..width as usize * geometry.n_heads as usize * HD)
        .map(|_| rng.unit())
        .collect()
}

fn random_gate(geometry: Geometry, width: u32, rng: &mut Xor) -> Vec<f32> {
    (0..width as usize * geometry.n_heads as usize * HD)
        .map(|_| rng.range(-6.0, 6.0))
        .collect()
}

fn lanes_of(committed: &[u32], rows: &[u32]) -> Vec<TqHbVerifyLane> {
    let mut first_row = 0;
    committed
        .iter()
        .zip(rows)
        .enumerate()
        .map(|(slot, (&committed, &rows))| {
            let lane = TqHbVerifyLane {
                slot: slot as u32,
                committed,
                rows,
                first_row,
            };
            first_row += rows;
            lane
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Rows `rows` of packed lane output starting at `first_row`.
fn rows_of(output: &[f32], geometry: Geometry, first_row: u32, rows: u32) -> &[f32] {
    let q_total = geometry.n_heads as usize * HD;
    &output[first_row as usize * q_total..(first_row + rows) as usize * q_total]
}

#[test]
fn f64_reference_at_served_shapes() {
    let Some(mut gpu) = gpu() else { return };
    let mut rng = Xor(0x5EED_1234_ABCD_0001);
    let mut cut_rng = Xor(0x5EED_1234_ABCD_0152);
    for (name, geometry) in [("qwen38", QWEN38), ("qwen36", QWEN36)] {
        // Short histories: window inside, across and at page boundaries, an
        // empty history, one to four lanes, short windows. Every head.
        let all_heads: Vec<usize> = (0..geometry.n_heads as usize).collect();
        for (committed, rows) in [
            (vec![0u32], vec![8u32]),
            (vec![1, 24], vec![8, 8]),
            (vec![31, 32, 500, 63], vec![8, 8, 8, 8]),
            (vec![40, 7, 100], vec![1, 3, 8]),
        ] {
            let cache = Cache::random(geometry, 576, (4.0, 24.0), &mut rng);
            let lanes = lanes_of(&committed, &rows);
            let width = 32;
            let q = random_q(geometry, width, &mut rng);
            let gate = random_gate(geometry, width, &mut rng);
            let run = run_tile(&mut gpu, &cache, &q, &gate, &lanes, width);
            let label = format!("{name} committed {committed:?} rows {rows:?}");
            check_reference(&cache, &q, &lanes, &run, &all_heads, SERVED_RMS, &label);
            check_undo_gate(&mut gpu, &run, geometry, &gate, &lanes, width, &label);
        }
        // Peaked softmax (large key norms): rescales on most pages.
        {
            let cache = Cache::random(geometry, 576, (100.0, 400.0), &mut rng);
            let lanes = lanes_of(&[500, 131], &[8, 8]);
            let q = random_q(geometry, 16, &mut rng);
            let gate = random_gate(geometry, 16, &mut rng);
            let run = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 16);
            let label = format!("{name} peaked");
            check_reference(&cache, &q, &lanes, &run, &all_heads, PEAKED_RMS, &label);
            check_undo_gate(&mut gpu, &run, geometry, &gate, &lanes, 16, &label);
        }
        // Value norms in the thousands: the half probability operand stays
        // far from its range limit and the bound is relative.
        {
            let mut cache = Cache::random(geometry, 576, (4.0, 24.0), &mut rng);
            for norm in &mut cache.v_norms {
                *norm *= 2_000.0;
            }
            let lanes = lanes_of(&[200], &[8]);
            let q = random_q(geometry, 8, &mut rng);
            let gate = random_gate(geometry, 8, &mut rng);
            let run = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 8);
            let label = format!("{name} large value norms");
            check_reference(&cache, &q, &lanes, &run, &all_heads, SERVED_RMS, &label);
        }
        // Served histories: several pages per split (2,048: 3; 8,192: 11;
        // 20,000: 27). First and last query head of the first and last KV
        // head.
        let group = geometry.group();
        let nh = geometry.n_heads as usize;
        let sampled = [0, group - 1, nh - group, nh - 1];
        for committed in [2_048u32, 8_192, 20_000] {
            let capacity = (committed + 8).div_ceil(32) * 32 + 32;
            let cache = Cache::random(geometry, capacity, (4.0, 24.0), &mut rng);
            let lanes = [
                TqHbVerifyLane {
                    slot: 2,
                    committed,
                    rows: 8,
                    first_row: 0,
                },
                TqHbVerifyLane {
                    slot: 0,
                    committed: committed - 37,
                    rows: 8,
                    first_row: 8,
                },
            ];
            let q = random_q(geometry, 16, &mut rng);
            let gate = random_gate(geometry, 16, &mut rng);
            let run = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 16);
            let label = format!("{name} committed {committed}");
            check_reference(&cache, &q, &lanes, &run, &sampled, SERVED_RMS, &label);
            check_undo_gate(&mut gpu, &run, geometry, &gate, &lanes, 16, &label);
        }
        // Cut windows: rows on both sides of the page boundaries at 1,536 and
        // 2,304 tokens, where the split length steps (two tiles per lane),
        // full and short, beside windows that end or start at the boundary;
        // peaked; and the step at 23,040 tokens. Their own generator, so the
        // fixtures above keep their inputs.
        for (committed, rows, key_norms, limits) in [
            (
                vec![1_531u32, 1_529, 2_299, 1_535],
                vec![8u32, 8, 8, 8],
                (4.0, 24.0),
                SERVED_RMS,
            ),
            (
                vec![2_303, 1_533, 1_530, 1_528],
                vec![2, 5, 7, 8],
                (4.0, 24.0),
                SERVED_RMS,
            ),
            (vec![1_532, 2_300], vec![8, 8], (100.0, 400.0), PEAKED_RMS),
        ] {
            let cache = Cache::random(geometry, 2_336, key_norms, &mut cut_rng);
            let lanes = lanes_of(&committed, &rows);
            let q = random_q(geometry, 32, &mut cut_rng);
            let gate = random_gate(geometry, 32, &mut cut_rng);
            let run = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 32);
            let label = format!("{name} cut windows {committed:?} rows {rows:?}");
            check_reference(&cache, &q, &lanes, &run, &all_heads, limits, &label);
            check_undo_gate(&mut gpu, &run, geometry, &gate, &lanes, 32, &label);
        }
        {
            let cache = Cache::random(geometry, 23_104, (4.0, 24.0), &mut cut_rng);
            let lanes = lanes_of(&[23_037, 23_033], &[8, 8]);
            let q = random_q(geometry, 16, &mut cut_rng);
            let gate = random_gate(geometry, 16, &mut cut_rng);
            let run = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 16);
            let label = format!("{name} cut windows [23037, 23033]");
            check_reference(&cache, &q, &lanes, &run, &sampled, SERVED_RMS, &label);
            check_undo_gate(&mut gpu, &run, geometry, &gate, &lanes, 16, &label);
        }
    }
}

#[test]
fn a_saturating_value_norm_stays_finite() {
    let Some(mut gpu) = gpu() else { return };
    let mut rng = Xor(0x5EED_1234_ABCD_0004);
    // One tile, and a cut window (two tiles).
    for (geometry, capacity, committed) in [
        (QWEN38, 128, 40),
        (QWEN38, 1_600, 1_531),
        (QWEN36, 1_600, 1_531),
    ] {
        let mut cache = Cache::random(geometry, capacity, (4.0, 24.0), &mut rng);
        // Far beyond the half operand's range for any probability near one.
        for norm in &mut cache.v_norms {
            *norm = 1e9;
        }
        let lanes = lanes_of(&[committed], &[8]);
        let q = random_q(geometry, 8, &mut rng);
        let gate = random_gate(geometry, 8, &mut rng);
        let run = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 8);
        assert!(
            run.attention.iter().all(|v| v.is_finite()),
            "saturated operand overflowed at {committed}"
        );
        assert!(run.output.iter().all(|v| v.is_finite()));
    }
}

/// The split's page, score and probability storage is bound threadgroup
/// memory. Shader validation doubles static threadgroup memory, which would
/// take a kernel that declares that storage statically past the device limit
/// (and abort under the Metal debug layer).
#[test]
fn split_threadgroup_memory_is_bound_not_static() {
    let Some(mut gpu) = gpu() else { return };
    let limit = gpu.device.metal_device().max_threadgroup_memory_length();
    for name in ["tq_hb_verify_tile_split_g6", "tq_hb_verify_tile_split_g8"] {
        let pipeline = gpu
            .registry
            .get_pipeline(name, gpu.device.metal_device())
            .expect("split pipeline");
        let fixed = pipeline.static_threadgroup_memory_length();
        eprintln!(
            "{name}: static threadgroup memory {fixed} bytes, device limit {limit}, {} threads allowed",
            pipeline.max_total_threads_per_threadgroup()
        );
        assert!(
            fixed <= 64,
            "{name}: {fixed} bytes of static threadgroup memory"
        );
        assert!(pipeline.max_total_threads_per_threadgroup() >= 256);
    }
}

#[test]
fn repeated_runs_are_bitwise_equal() {
    let Some(mut gpu) = gpu() else { return };
    let mut rng = Xor(0x5EED_1234_ABCD_0002);
    // Windows that are cut at 1,536 and 2,304 tokens, one that ends at the
    // boundary and one that starts just before its last token. Their own
    // generator: the first fixture's digests are those of 0.21.8.
    let mut cut_rng = Xor(0x5EED_1234_ABCD_0153);
    for geometry in [QWEN38, QWEN36] {
        for (label, capacity, committed, rng) in [
            ("output", 2_112, [2_048, 513, 0, 1_000], &mut rng),
            (
                "cut windows",
                2_336,
                [1_531, 2_299, 1_529, 1_535],
                &mut cut_rng,
            ),
        ] {
            let cache = Cache::random(geometry, capacity, (4.0, 200.0), rng);
            let lanes = lanes_of(&committed, &[8, 8, 8, 8]);
            let q = random_q(geometry, 32, rng);
            let gate = random_gate(geometry, 32, rng);
            let first = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 32);
            for _ in 0..4 {
                let again = run_tile(&mut gpu, &cache, &q, &gate, &lanes, 32);
                assert_eq!(bits(&first.output), bits(&again.output));
                assert_eq!(bits(&first.attention), bits(&again.attention));
            }
            // Compared across runs: plain, shader validation, debug layer,
            // runtime-compiled source.
            let digest = bits(&first.output)
                .iter()
                .chain(&bits(&first.attention))
                .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &word| {
                    (hash ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3)
                });
            eprintln!(
                "{label} digest {}/{} heads: {digest:016x}",
                geometry.n_heads, geometry.n_kv
            );
        }
    }
}

#[test]
fn a_row_is_independent_of_its_cohort() {
    let Some(mut gpu) = gpu() else { return };
    let mut rng = Xor(0x5EED_1234_ABCD_0003);
    for (name, geometry) in [("qwen38", QWEN38), ("qwen36", QWEN36)] {
        let q_total = geometry.n_heads as usize * HD;
        // 1,531: a cut window (rows 0-4 and 5-7 are two tiles).
        for committed in [0u32, 27, 1_500, 1_531] {
            let capacity = 1_600;
            let cache = Cache::random(geometry, capacity, (4.0, 200.0), &mut rng);
            // The subject: 8 rows of slot 1, alone, at packed rows 0..8.
            let subject_q: Vec<f32> = (0..8 * q_total).map(|_| rng.unit()).collect();
            let subject_gate: Vec<f32> = (0..8 * q_total).map(|_| rng.range(-6.0, 6.0)).collect();
            let subject = TqHbVerifyLane {
                slot: 1,
                committed,
                rows: 8,
                first_row: 0,
            };
            let alone = run_tile(&mut gpu, &cache, &subject_q, &subject_gate, &[subject], 8);
            let want = bits(&alone.output);

            // (a) Three other lanes around it, another packed position, padding.
            let mut q = random_q(geometry, 32, &mut rng);
            let mut gate = random_gate(geometry, 32, &mut rng);
            q[11 * q_total..19 * q_total].copy_from_slice(&subject_q);
            gate[11 * q_total..19 * q_total].copy_from_slice(&subject_gate);
            let cohort = [
                TqHbVerifyLane {
                    slot: 3,
                    committed: 700,
                    rows: 8,
                    first_row: 0,
                },
                TqHbVerifyLane {
                    slot: 0,
                    committed: 3,
                    rows: 3,
                    first_row: 8,
                },
                TqHbVerifyLane {
                    slot: 1,
                    committed,
                    rows: 8,
                    first_row: 11,
                },
                TqHbVerifyLane {
                    slot: 2,
                    committed: 1_592,
                    rows: 8,
                    first_row: 19,
                },
            ];
            let joint = run_tile(&mut gpu, &cache, &q, &gate, &cohort, 32);
            assert_eq!(
                bits(rows_of(&joint.output, geometry, 11, 8)),
                want,
                "{name} committed {committed}: cohort changed the subject"
            );
            // Padding rows: the first lane's first row under their own gates.
            check_undo_gate(&mut gpu, &joint, geometry, &gate, &cohort, 32, name);

            // (b) Other slots' contents (non-finite-prone norms included).
            let mut other = cache.clone();
            for slot in [0, 2, 3] {
                other.scramble_from(slot, 0, &mut rng);
            }
            let mut q = random_q(geometry, 16, &mut rng);
            let mut gate = random_gate(geometry, 16, &mut rng);
            q[..8 * q_total].copy_from_slice(&subject_q);
            gate[..8 * q_total].copy_from_slice(&subject_gate);
            let padded = run_tile(&mut gpu, &other, &q, &gate, &[subject], 16);
            assert_eq!(
                bits(rows_of(&padded.output, geometry, 0, 8)),
                want,
                "{name} committed {committed}: other slots or padding changed the subject"
            );

            // (c) A shorter window, and cache content at or after a row's
            // causal end: rows 0..=r keep their bits.
            for r in 0..8u32 {
                let kept = (r as usize + 1) * q_total;
                let mut later = cache.clone();
                later.scramble_from(1, committed + r + 1, &mut rng);
                let short = TqHbVerifyLane {
                    rows: r + 1,
                    ..subject
                };
                let run = run_tile(
                    &mut gpu,
                    &later,
                    &subject_q[..kept],
                    &subject_gate[..kept],
                    &[short],
                    r + 1,
                );
                assert_eq!(
                    bits(&run.output),
                    want[..kept],
                    "{name} committed {committed}: window of {} rows changed a row",
                    r + 1
                );
                let full = run_tile(&mut gpu, &later, &subject_q, &subject_gate, &[subject], 8);
                assert_eq!(
                    bits(&full.output)[..kept],
                    want[..kept],
                    "{name} committed {committed}: tokens after row {r} changed it"
                );
            }
        }
    }
}

#[test]
fn invalid_steps_are_rejected() {
    let Some(mut gpu) = gpu() else { return };
    let geometry = QWEN36;
    let mut rng = Xor(7);
    let cache = Cache::random(geometry, 64, (1.0, 2.0), &mut rng);
    let q = random_q(geometry, 8, &mut rng);
    let device = &gpu.device;
    let shape = geometry.shape(64, 8);
    let sizes = verify_tile_scratch_bytes(&shape, 2, 1);
    let k_packed = buffer(device, DType::U8, &cache.k_codes);
    let k_norms = buffer(device, DType::F32, &cache.k_norms);
    let v_packed = buffer(device, DType::U8, &cache.v_codes);
    let v_norms = buffer(device, DType::F32, &cache.v_norms);
    let q_rows = buffer(device, DType::F32, &q);
    let output = buffer(device, DType::F32, &vec![0f32; q.len()]);
    let queries = buffer(
        device,
        DType::F16,
        &vec![half::f16::ZERO; sizes.queries / 2],
    );
    let partials = buffer(device, DType::F32, &vec![0f32; sizes.partials / 4]);
    let statistics = buffer(device, DType::F32, &vec![0f32; sizes.statistics / 4]);
    let attention = buffer(device, DType::F32, &vec![0f32; sizes.attention / 4]);
    let mut encode = |shape: &TqHbVerifyTileShape, lanes: &[TqHbVerifyLane]| {
        let mut encoder = device.command_encoder().expect("encoder");
        tq_hb_verify_tile_attention(
            &mut encoder,
            &mut gpu.registry,
            device,
            &q_rows,
            &q_rows,
            &TqHbVerifyTileCache {
                k_packed: &k_packed,
                k_norms: &k_norms,
                v_packed: &v_packed,
                v_norms: &v_norms,
            },
            &output,
            &TqHbVerifyTileScratch {
                queries: &queries,
                partials: &partials,
                statistics: &statistics,
                attention: &attention,
            },
            shape,
            lanes,
        )
    };
    let lane = TqHbVerifyLane {
        slot: 0,
        committed: 10,
        rows: 4,
        first_row: 0,
    };
    assert!(encode(&shape, &[lane]).is_ok());
    assert!(encode(&shape, &[]).is_err(), "no lanes");
    assert!(
        encode(&shape, &[TqHbVerifyLane { rows: 0, ..lane }]).is_err(),
        "no rows"
    );
    assert!(
        encode(&shape, &[TqHbVerifyLane { rows: 9, ..lane }]).is_err(),
        "nine rows"
    );
    assert!(
        encode(
            &shape,
            &[TqHbVerifyLane {
                committed: 61,
                ..lane
            }]
        )
        .is_err(),
        "past capacity"
    );
    assert!(
        encode(
            &shape,
            &[TqHbVerifyLane {
                first_row: 5,
                ..lane
            }]
        )
        .is_err(),
        "past width"
    );
    assert!(
        encode(
            &shape,
            &[TqHbVerifyLane {
                slot: SLOTS,
                ..lane
            }]
        )
        .is_err(),
        "slot outside cache"
    );
    assert!(
        encode(
            &shape,
            &[
                lane,
                TqHbVerifyLane {
                    slot: 1,
                    first_row: 3,
                    ..lane
                }
            ]
        )
        .is_err(),
        "overlapping windows"
    );
    assert!(
        encode(
            &TqHbVerifyTileShape {
                kv_capacity: 48,
                ..shape
            },
            &[lane]
        )
        .is_err(),
        "capacity not a page multiple"
    );
    assert!(
        encode(
            &TqHbVerifyTileShape {
                codebook_bits: 6,
                ..shape
            },
            &[lane]
        )
        .is_err(),
        "6-bit codes"
    );
    assert!(
        encode(
            &TqHbVerifyTileShape {
                n_heads: 8,
                ..shape
            },
            &[lane]
        )
        .is_err(),
        "group of four"
    );
}

/// A slot's cache and scratch kept on the device across many steps, with a
/// query and a gate per position, for sweeps of window starts.
struct Sweep {
    geometry: Geometry,
    capacity: u32,
    k_packed: MlxBuffer,
    k_norms: MlxBuffer,
    v_packed: MlxBuffer,
    v_norms: MlxBuffer,
    q_rows: MlxBuffer,
    gate_rows: MlxBuffer,
    output: MlxBuffer,
    queries: MlxBuffer,
    partials: MlxBuffer,
    statistics: MlxBuffer,
    attention: MlxBuffer,
    /// Queries and gates of positions `p % SWEEP_PERIOD`.
    q_table: Vec<f32>,
    gate_table: Vec<f32>,
}

/// Positions `p` and `p + SWEEP_PERIOD` share a query and a gate.
const SWEEP_PERIOD: usize = 61;
/// The sweeps use one slot and up to four lanes of eight rows.
const SWEEP_SLOT: u32 = 1;
const SWEEP_WIDTH: u32 = 32;

impl Sweep {
    fn new(gpu: &Gpu, geometry: Geometry, capacity: u32, rng: &mut Xor) -> Self {
        let device = &gpu.device;
        let cache = Cache::random(geometry, capacity, (4.0, 40.0), rng);
        let q_total = geometry.n_heads as usize * HD;
        let shape = geometry.shape(capacity, SWEEP_WIDTH);
        // Four lanes at the most splits a tile can have.
        let sizes = verify_tile_scratch_bytes(&shape, 4, MAX_SPLITS);
        let rows = SWEEP_WIDTH as usize * q_total;
        Self {
            geometry,
            capacity,
            k_packed: buffer(device, DType::U8, &cache.k_codes),
            k_norms: buffer(device, DType::F32, &cache.k_norms),
            v_packed: buffer(device, DType::U8, &cache.v_codes),
            v_norms: buffer(device, DType::F32, &cache.v_norms),
            q_rows: buffer(device, DType::F32, &vec![0f32; rows]),
            gate_rows: buffer(device, DType::F32, &vec![0f32; rows]),
            output: buffer(device, DType::F32, &vec![f32::NAN; rows]),
            queries: buffer(device, DType::F16, &vec![half::f16::NAN; sizes.queries / 2]),
            partials: buffer(device, DType::F32, &vec![f32::NAN; sizes.partials / 4]),
            statistics: buffer(device, DType::F32, &vec![f32::NAN; sizes.statistics / 4]),
            attention: buffer(device, DType::F32, &vec![f32::NAN; sizes.attention / 4]),
            q_table: (0..SWEEP_PERIOD * q_total).map(|_| rng.unit()).collect(),
            gate_table: (0..SWEEP_PERIOD * q_total)
                .map(|_| rng.range(-6.0, 6.0))
                .collect(),
        }
    }

    /// One step of windows `(start, rows)` of the sweep slot, one lane each.
    /// Returns, per lane and window row, the digest of the token's rotated
    /// attention and gated output bits.
    fn step(&mut self, gpu: &mut Gpu, windows: &[(u32, u32)]) -> Vec<Vec<u64>> {
        let geometry = self.geometry;
        let q_total = geometry.n_heads as usize * HD;
        let mut lanes = Vec::with_capacity(windows.len());
        let mut first_row = 0;
        for &(start, rows) in windows {
            lanes.push(TqHbVerifyLane {
                slot: SWEEP_SLOT,
                committed: start,
                rows,
                first_row,
            });
            for row in 0..rows as usize {
                let packed = (first_row as usize + row) * q_total;
                let entry = (start as usize + row) % SWEEP_PERIOD * q_total;
                self.q_rows.as_mut_slice::<f32>().expect("q")[packed..packed + q_total]
                    .copy_from_slice(&self.q_table[entry..entry + q_total]);
                self.gate_rows.as_mut_slice::<f32>().expect("gate")[packed..packed + q_total]
                    .copy_from_slice(&self.gate_table[entry..entry + q_total]);
            }
            first_row += rows;
        }
        let shape = geometry.shape(self.capacity, SWEEP_WIDTH);
        let mut encoder = gpu.device.command_encoder().expect("encoder");
        tq_hb_verify_tile_attention(
            &mut encoder,
            &mut gpu.registry,
            &gpu.device,
            &self.q_rows,
            &self.gate_rows,
            &TqHbVerifyTileCache {
                k_packed: &self.k_packed,
                k_norms: &self.k_norms,
                v_packed: &self.v_packed,
                v_norms: &self.v_norms,
            },
            &self.output,
            &TqHbVerifyTileScratch {
                queries: &self.queries,
                partials: &self.partials,
                statistics: &self.statistics,
                attention: &self.attention,
            },
            &shape,
            &lanes,
        )
        .expect("encode sweep step");
        encoder.commit_and_wait().expect("commit");
        let output = self.output.as_slice::<f32>().expect("output");
        let attention = self.attention.as_slice::<f32>().expect("attention");
        lanes
            .iter()
            .enumerate()
            .map(|(index, lane)| {
                (0..lane.rows as usize)
                    .map(|row| {
                        let gated = (lane.first_row as usize + row) * q_total;
                        let rotated = (index * VERIFY_TILE_ROWS as usize + row) * q_total;
                        output[gated..gated + q_total]
                            .iter()
                            .chain(&attention[rotated..rotated + q_total])
                            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
                                (hash ^ u64::from(value.to_bits()))
                                    .wrapping_mul(0x0000_0100_0000_01b3)
                            })
                    })
                    .collect()
            })
            .collect()
    }
}

/// Digest of a position at each row index of the windows that verified it.
type Alignments = Vec<[Option<u64>; 8]>;

/// Page boundaries (in tokens) below `capacity` where the split length steps.
fn split_length_steps(capacity: u32) -> Vec<u32> {
    (1..capacity / PAGE_TOKENS)
        .filter(|&pages| {
            verify_tile_pages_per_split(pages) != verify_tile_pages_per_split(pages + 1)
        })
        .map(|pages| pages * PAGE_TOKENS)
        .collect()
}

/// Run every start of `starts` as a full window, four windows a step, and
/// require that every position has the same bits at every row index it was
/// verified at.
fn full_windows(sweep: &mut Sweep, gpu: &mut Gpu, starts: &[u32], name: &str) -> Alignments {
    let mut digests: Alignments = vec![[None; 8]; sweep.capacity as usize];
    for chunk in starts.chunks(4) {
        let windows: Vec<(u32, u32)> = chunk.iter().map(|&start| (start, 8)).collect();
        for (lane, rows) in sweep.step(gpu, &windows).into_iter().enumerate() {
            for (row, digest) in rows.into_iter().enumerate() {
                digests[chunk[lane] as usize + row][row] = Some(digest);
            }
        }
    }
    let differing: Vec<usize> = digests
        .iter()
        .enumerate()
        .filter(|(_, seen)| {
            let mut seen = seen.iter().flatten();
            let first = seen.next();
            seen.any(|digest| Some(digest) != first)
        })
        .map(|(position, _)| position)
        .collect();
    assert!(
        differing.is_empty(),
        "{name}: {} tokens depend on their window alignment, first at positions {:?}",
        differing.len(),
        &differing[..differing.len().min(8)]
    );
    digests
}

/// Windows of one to eight rows in steps of one to three lanes that end just
/// before, at and across each of `boundaries`: every row must have the bits
/// `digests` holds for its position. Returns the rows checked.
fn short_windows(
    sweep: &mut Sweep,
    gpu: &mut Gpu,
    digests: &Alignments,
    boundaries: &[u32],
    name: &str,
) -> usize {
    let capacity = sweep.capacity;
    let mut checked = 0;
    for &boundary in boundaries {
        for rows in 1..=8u32 {
            for start in [
                boundary.saturating_sub(rows),
                (boundary + 1).saturating_sub(rows),
                boundary.saturating_sub(rows / 2),
                boundary,
            ] {
                let start = start.min(capacity - rows);
                let lanes = 1 + (start + rows) as usize % 3;
                let windows: Vec<(u32, u32)> = (0..lanes as u32)
                    .map(|lane| ((start + 5 * lane).min(capacity - rows), rows))
                    .collect();
                for (lane, row_digests) in sweep.step(gpu, &windows).into_iter().enumerate() {
                    for (row, digest) in row_digests.into_iter().enumerate() {
                        let position = windows[lane].0 + row as u32;
                        let expected = digests[position as usize]
                            .iter()
                            .flatten()
                            .next()
                            .expect("a position the full windows verified");
                        assert_eq!(
                            digest, *expected,
                            "{name}: position {position} as row {row} of a {rows}-row window at \
                             {} in a {lanes}-lane step",
                            windows[lane].0
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    checked
}

/// A token's bits are a function of its own position, query, gate and causal
/// history: the same whichever row of whichever window verifies it. Every
/// window start from 0 to past 9,000 tokens of history is run as a full
/// window (four starts a step, one lane each), so every token is seen at
/// every row index 0-7 and every page boundary, including those where the
/// split length steps, is crossed at every alignment; then windows of every
/// length and steps of one to three lanes are checked against the same table
/// around every such step.
#[test]
fn a_token_is_independent_of_its_window_alignment() {
    let Some(mut gpu) = gpu() else { return };
    let mut rng = Xor(0x5EED_1234_ABCD_0005);
    const CAPACITY: u32 = 9_152;
    const LAST_START: u32 = CAPACITY - 8;
    for (name, geometry) in [("qwen38", QWEN38), ("qwen36", QWEN36)] {
        let mut sweep = Sweep::new(&gpu, geometry, CAPACITY, &mut rng);
        let starts: Vec<u32> = (0..=LAST_START).collect();
        let digests = full_windows(&mut sweep, &mut gpu, &starts, name);
        for (position, seen) in digests.iter().enumerate() {
            // Positions 7..=LAST_START sit at all eight row indices.
            let alignments = seen.iter().flatten().count();
            assert!(
                alignments == 8 || position < 7 || position > LAST_START as usize,
                "{name}: position {position} seen at {alignments} alignments"
            );
        }
        let steps = split_length_steps(CAPACITY);
        assert_eq!(steps.len(), 10, "{name}: split-length steps {steps:?}");
        // Around every step of the split length and a few plain starts.
        let boundaries: Vec<u32> = steps
            .iter()
            .copied()
            .chain([32, 1_000, 5_555, LAST_START])
            .collect();
        let checked = short_windows(&mut sweep, &mut gpu, &digests, &boundaries, name);
        eprintln!(
            "{name}: {} tokens bitwise equal at every row index over window starts 0..={LAST_START} \
             ({} split-length steps); {checked} rows of 1-8 row windows in 1-3 lane steps equal",
            CAPACITY as usize,
            steps.len()
        );
    }
}

/// The same up to 33,824 tokens of history, sampled: every window start
/// within 48 tokens of each of the 43 page boundaries where the split length
/// steps (1,536 to 33,792 in steps of 768), and the last starts of the cache.
/// The 7 tokens before each boundary are the head rows of cut windows.
#[test]
fn a_token_is_independent_of_its_window_alignment_at_every_split_length_step() {
    let Some(mut gpu) = gpu() else { return };
    let mut rng = Xor(0x5EED_1234_ABCD_0006);
    const CAPACITY: u32 = 33_824;
    const LAST_START: u32 = CAPACITY - 8;
    const NEAR: u32 = 48;
    let steps = split_length_steps(CAPACITY);
    assert_eq!(steps.len(), 43, "split-length steps {steps:?}");
    assert_eq!((steps[0], steps[42]), (1_536, 33_792));
    let starts: Vec<u32> = (0..=LAST_START)
        .filter(|&start| {
            start + 80 >= CAPACITY
                || steps
                    .iter()
                    .any(|&step| start + NEAR >= step && start <= step + NEAR)
        })
        .collect();
    for (name, geometry) in [("qwen38", QWEN38), ("qwen36", QWEN36)] {
        let mut sweep = Sweep::new(&gpu, geometry, CAPACITY, &mut rng);
        let digests = full_windows(&mut sweep, &mut gpu, &starts, name);
        let mut observations = 0;
        for &step in &steps {
            // The tokens on both sides of the boundary sit at all eight row
            // indices: the head rows of every cut window among them.
            for position in step - 8..(step + 8).min(LAST_START) {
                let alignments = digests[position as usize].iter().flatten().count();
                assert_eq!(alignments, 8, "{name}: position {position}");
            }
        }
        for seen in &digests {
            observations += seen.iter().flatten().count();
        }
        let checked = short_windows(&mut sweep, &mut gpu, &digests, &steps, name);
        eprintln!(
            "{name}: {observations} observations of full windows at {} starts around {} \
             split-length steps up to {CAPACITY} tokens bitwise equal; {checked} rows of 1-8 row \
             windows in 1-3 lane steps equal",
            starts.len(),
            steps.len()
        );
    }
}
