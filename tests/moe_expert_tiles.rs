//! MoE expert live tiles (`moe_tiles_*`): F64 reference with a K-scaled
//! bound, determinism, row/cohort independence and masked routes.
#![cfg(mlx_native_has_metal_tensor_artifact)]

use half::{bf16, f16};
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{
    ggml_expert_bytes, moe_tiles_column_tile_exact, moe_tiles_down, moe_tiles_gate_up,
    moe_tiles_gather, moe_tiles_grouped_rows, moe_tiles_schedule,
    prepare_quantized_matmul_id_ggml_grouped_m8, quantized_matmul_id_ggml_grouped_m8_epilogue,
    quantized_matmul_id_ggml_grouped_m8_gate_up, DType, GgmlQuantizedMatmulIdParams, GgmlType,
    GroupedM8Epilogue, GroupedM8Input, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice,
    MoeTilesBuffers, MoeTilesParams, PreparedGroupedM8Experts, MOE_TILES_GRID_WORDS,
};

const HIDDEN: usize = 512;
const INTERMEDIATE: usize = 256;
const EXPERTS: usize = 24;
const TOP_K: usize = 8;
const TOKENS: usize = 32;
const MASK: u32 = EXPERTS as u32;
/// Bit pattern no kernel writes: an untouched output element.
const UNWRITTEN: u32 = 0x7fc0_dead;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (-1, 1) times a power of two in 2^-3..=2^3.
    fn activation(&mut self) -> f32 {
        let unit = ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        unit * 2.0f32.powi((self.next() % 7) as i32 - 3)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Random GGML blocks with half block scales of magnitude 2^-8..2^-7.
fn quantized(
    device: &MlxDevice,
    ggml_type: GgmlType,
    n: usize,
    k: usize,
    rng: &mut Rng,
) -> MlxBuffer {
    let stride =
        (n * (k / ggml_type.block_values() as usize) * ggml_type.block_bytes() as usize) as u64;
    let bytes = ggml_expert_bytes(ggml_type, EXPERTS as u32, n as u32, k as u32, stride)
        .expect("bytes") as usize;
    let mut buffer = device
        .alloc_buffer(bytes, DType::U8, vec![bytes])
        .expect("weights");
    let data = buffer.as_mut_slice::<u8>().expect("weight slice");
    for chunk in data.chunks_mut(8) {
        chunk.copy_from_slice(&rng.next().to_le_bytes()[..chunk.len()]);
    }
    let scale_offsets: &[usize] = match ggml_type {
        GgmlType::Q4_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        _ => unreachable!(),
    };
    for block in data.chunks_exact_mut(ggml_type.block_bytes() as usize) {
        for &offset in scale_offsets {
            let sign = if rng.next() & 1 != 0 { 0x8000 } else { 0 };
            let bits = sign | (7 << 10) | ((rng.next() >> 8) as u16 & 0x03ff);
            block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    buffer
}

/// The exact weights of one `[n, k]` expert matrix, row-major.
fn dequantize(ggml_type: GgmlType, bytes: &[u8], n: usize, k: usize) -> Vec<f64> {
    let half = |at: &[u8]| f64::from(f16::from_bits(u16::from_le_bytes([at[0], at[1]])));
    let mut weights = Vec::with_capacity(n * k);
    let block_bytes = ggml_type.block_bytes() as usize;
    for block in bytes[..n * (k / 256) * block_bytes].chunks_exact(block_bytes) {
        let mut values = [0.0f64; 256];
        match ggml_type {
            GgmlType::Q4_K => {
                let (d, dmin) = (half(&block[0..2]), half(&block[2..4]));
                let scales = &block[4..16];
                let qs = &block[16..144];
                for j in 0..8 {
                    let (scale, minimum) = if j < 4 {
                        (scales[j] & 63, scales[j + 4] & 63)
                    } else {
                        (
                            (scales[j + 4] & 0xf) | ((scales[j - 4] >> 6) << 4),
                            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
                        )
                    };
                    for l in 0..32 {
                        let code = (qs[32 * (j / 2) + l] >> (4 * (j % 2))) & 0xf;
                        values[32 * j + l] =
                            d * f64::from(scale) * f64::from(code) - dmin * f64::from(minimum);
                    }
                }
            }
            GgmlType::Q6_K => {
                let d = half(&block[208..210]);
                for half_block in 0..2 {
                    let ql = &block[64 * half_block..];
                    let qh = &block[128 + 32 * half_block..];
                    let scales = &block[192 + 8 * half_block..];
                    for l in 0..32 {
                        let is = l / 16;
                        let codes = [
                            (ql[l] & 0xf) | ((qh[l] & 3) << 4),
                            (ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4),
                            (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4),
                            (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4),
                        ];
                        for (quarter, code) in codes.into_iter().enumerate() {
                            let scale = f64::from(scales[is + 2 * quarter] as i8);
                            values[128 * half_block + 32 * quarter + l] =
                                d * scale * (f64::from(code) - 32.0);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
        weights.extend_from_slice(&values);
    }
    weights
}

struct Layer {
    device: MlxDevice,
    registry: KernelRegistry,
    gate_up_type: GgmlType,
    down_type: GgmlType,
    sources: [MlxBuffer; 3],
    prepared: [PreparedGroupedM8Experts; 3],
    column_tile: bool,
}

impl Layer {
    fn new(gate_up_type: GgmlType, down_type: GgmlType, seed: u64) -> Self {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut rng = Rng(seed);
        let sources = [
            quantized(&device, gate_up_type, INTERMEDIATE, HIDDEN, &mut rng),
            quantized(&device, gate_up_type, INTERMEDIATE, HIDDEN, &mut rng),
            quantized(&device, down_type, HIDDEN, INTERMEDIATE, &mut rng),
        ];
        let shape = |ggml_type: GgmlType, n: usize, k: usize| GgmlQuantizedMatmulIdParams {
            n_tokens: TOKENS as u32,
            top_k: TOP_K as u32,
            n: n as u32,
            k: k as u32,
            n_experts: EXPERTS as u32,
            expert_stride: (n * (k / 256) * ggml_type.block_bytes() as usize) as u64,
            ggml_type,
        };
        let mut encoder = device.command_encoder().expect("prepare encoder");
        let mut prepare = |source: &MlxBuffer, shape: GgmlQuantizedMatmulIdParams| {
            prepare_quantized_matmul_id_ggml_grouped_m8(
                &mut encoder,
                &mut registry,
                &device,
                source,
                &shape,
            )
            .expect("prepare")
        };
        let prepared = [
            prepare(&sources[0], shape(gate_up_type, INTERMEDIATE, HIDDEN)),
            prepare(&sources[1], shape(gate_up_type, INTERMEDIATE, HIDDEN)),
            prepare(&sources[2], shape(down_type, HIDDEN, INTERMEDIATE)),
        ];
        encoder.commit_and_wait().expect("commit preparation");
        let column_tile =
            moe_tiles_column_tile_exact(&device, &mut registry).expect("column probe");
        Self {
            device,
            registry,
            gate_up_type,
            down_type,
            sources,
            prepared,
            column_tile,
        }
    }

    /// Exact weights of `expert`'s projection 0 (gate), 1 (up) or 2 (down).
    fn weights(&self, projection: usize, expert: usize) -> Vec<f64> {
        let (ggml_type, n, k) = if projection == 2 {
            (self.down_type, HIDDEN, INTERMEDIATE)
        } else {
            (self.gate_up_type, INTERMEDIATE, HIDDEN)
        };
        let stride = n * (k / 256) * ggml_type.block_bytes() as usize;
        let bytes = self.sources[projection]
            .as_slice::<u8>()
            .expect("source bytes");
        dequantize(
            ggml_type,
            &bytes[expert * stride..(expert + 1) * stride],
            n,
            k,
        )
    }

    fn run(&mut self, x: &[f32], ids: &[u32], column_tile: bool) -> Step {
        let tokens = ids.len() / TOP_K;
        let params = MoeTilesParams {
            n_tokens: tokens as u32,
            top_k: TOP_K as u32,
            n_experts: EXPERTS as u32,
            hidden: HIDDEN as u32,
            intermediate: INTERMEDIATE as u32,
            masked_routes: true,
            column_tile,
            shared_planes: false,
        };
        let device = &self.device;
        let rows =
            moe_tiles_grouped_rows(params.n_tokens, params.top_k, params.n_experts).expect("rows");
        // Every scratch row starts as NaN: rows a tile reads past its routes
        // and rows no pass writes must not reach a result.
        let nan_rows = |width: usize| {
            let mut buffer = device
                .alloc_buffer(rows * width * 2, DType::BF16, vec![rows, width])
                .expect("grouped rows");
            buffer
                .as_mut_slice::<u16>()
                .expect("grouped slice")
                .fill(0x7fc0);
            buffer
        };
        let gathered = nan_rows(HIDDEN);
        let intermediate = nan_rows(INTERMEDIATE);
        let grids = device
            .alloc_buffer(
                MOE_TILES_GRID_WORDS * 4,
                DType::U32,
                vec![MOE_TILES_GRID_WORDS],
            )
            .expect("grids");
        let mut input = device
            .alloc_buffer(tokens * HIDDEN * 4, DType::F32, vec![tokens, HIDDEN])
            .expect("input");
        input
            .as_mut_slice::<f32>()
            .expect("input slice")
            .copy_from_slice(x);
        let mut id_buffer = device
            .alloc_buffer(ids.len() * 4, DType::U32, vec![ids.len()])
            .expect("ids");
        id_buffer
            .as_mut_slice::<u32>()
            .expect("id slice")
            .copy_from_slice(ids);
        let mut output = device
            .alloc_buffer(ids.len() * HIDDEN * 4, DType::F32, vec![ids.len(), HIDDEN])
            .expect("output");
        output
            .as_mut_slice::<u32>()
            .expect("output slice")
            .fill(UNWRITTEN);
        let mut scratch =
            IdMmScratch::alloc(device, EXPERTS as u32, tokens as u32).expect("scratch");
        let buffers = MoeTilesBuffers {
            gathered: &gathered,
            intermediate: &intermediate,
            grids: &grids,
        };
        let mut encoder = device.command_encoder().expect("encoder");
        moe_tiles_schedule(
            &mut encoder,
            &mut self.registry,
            device,
            &id_buffer,
            &mut scratch,
            &params,
        )
        .expect("schedule");
        moe_tiles_gather(
            &mut encoder,
            &mut self.registry,
            device,
            &input,
            &id_buffer,
            &buffers,
            &scratch,
            &params,
        )
        .expect("gather");
        encoder.memory_barrier();
        moe_tiles_gate_up(
            &mut encoder,
            &mut self.registry,
            device,
            &self.prepared[0],
            &self.prepared[1],
            &id_buffer,
            &buffers,
            &scratch,
            &params,
        )
        .expect("gate/up");
        encoder.memory_barrier();
        moe_tiles_down(
            &mut encoder,
            &mut self.registry,
            device,
            &self.prepared[2],
            &id_buffer,
            &buffers,
            &scratch,
            &output,
            &params,
        )
        .expect("down");
        encoder.commit_and_wait().expect("commit step");

        // Each route's grouped intermediate row, through the schedule.
        let schedule = scratch.htpe().as_slice::<u32>().expect("schedule");
        let list = scratch.hids().as_slice::<u32>().expect("route list");
        let grouped = intermediate.as_slice::<u16>().expect("intermediate");
        let mut swiglu = vec![Vec::new(); ids.len()];
        let tiles = schedule[0] as usize;
        assert!(tiles & 0x8000_0000 == 0, "poisoned schedule");
        for tile in 0..tiles {
            let (first, count) = (
                schedule[2 + 3 * tile] as usize,
                schedule[3 + 3 * tile] as usize,
            );
            for row in 0..count {
                let route = list[first + row] as usize;
                swiglu[route] = grouped
                    [(first + row) * INTERMEDIATE..(first + row + 1) * INTERMEDIATE]
                    .to_vec();
            }
        }
        Step {
            output: output.as_slice::<u32>().expect("output bits").to_vec(),
            swiglu,
            grids: grids.as_slice::<u32>().expect("grid words").to_vec(),
        }
    }
}

struct Step {
    /// FP32 bits of `[routes, HIDDEN]`.
    output: Vec<u32>,
    /// BF16 bits of each scheduled route's SwiGLU row; empty for a masked route.
    swiglu: Vec<Vec<u16>>,
    grids: Vec<u32>,
}

impl Step {
    fn route(&self, route: usize) -> &[u32] {
        &self.output[route * HIDDEN..(route + 1) * HIDDEN]
    }
}

fn activations(tokens: usize, rng: &mut Rng) -> Vec<f32> {
    (0..tokens * HIDDEN).map(|_| rng.activation()).collect()
}

/// Routes whose experts hold 1, 2, 3..8, 9..16 and 17..32 routes: slot s of
/// token t draws from a pool that widens with the slot.
fn spread_routes(rng: &mut Rng) -> Vec<u32> {
    let mut ids = Vec::with_capacity(TOKENS * TOP_K);
    for token in 0..TOKENS {
        let mut row: Vec<u32> = vec![0, if token < 17 { 1 } else { 2 }];
        if token < 9 {
            row.push(3);
        }
        if token < 3 {
            row.push(4);
        }
        if token == 20 || token == 21 {
            row.push(5);
        }
        if token == 30 {
            row.push(6);
        }
        while row.len() < TOP_K {
            let expert = (7 + rng.below(EXPERTS - 7)) as u32;
            if !row.contains(&expert) {
                row.push(expert);
            }
        }
        // A token's slots are not ordered by expert.
        for index in (1..TOP_K).rev() {
            row.swap(index, rng.below(index + 1));
        }
        ids.extend_from_slice(&row);
    }
    let mut counts = [0usize; EXPERTS];
    ids.iter().for_each(|&id| counts[id as usize] += 1);
    for class in [1..=1, 2..=2, 3..=8, 9..=16, 17..=32] {
        assert!(
            counts.iter().any(|count| class.contains(count)),
            "no expert of {class:?} routes: {counts:?}"
        );
    }
    ids
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

/// Relative rounding error of BF16 (8 significand bits).
const BF16_ULP: f64 = 1.0 / 256.0;
/// A half below this magnitude is subnormal: its rounding error is absolute.
const HALF_MIN_NORMAL: f64 = 1.0 / 16384.0;

/// Worst-case relative error of one dot product of `k` terms against
/// `sum |x| (|w| + HALF_MIN_NORMAL)`: BF16 inputs when `rounded_input`, half
/// weights (each decoded in FP32 and rounded once), an FP32 sum of `k` exact
/// products. The bound grows with K through the sum.
fn dot_epsilon(k: usize, rounded_input: bool) -> f64 {
    let input = if rounded_input { BF16_ULP } else { 0.0 };
    let weight = 2.0f64.powi(-11) + 2.0f64.powi(-23);
    let sum = (k as f64 + 2.0) * 2.0f64.powi(-24);
    (1.0 + input) * (1.0 + weight) * (1.0 + sum) - 1.0
}

/// `(sum w x, sum |x| (|w| + HALF_MIN_NORMAL))`.
fn dot(w: &[f64], x: &[f64]) -> (f64, f64) {
    w.iter()
        .zip(x)
        .fold((0.0f64, 0.0f64), |(sum, magnitude), (w, x)| {
            (
                sum + w * x,
                magnitude + x.abs() * (w.abs() + HALF_MIN_NORMAL),
            )
        })
}

fn matches_f64_reference(gate_up_type: GgmlType, down_type: GgmlType) {
    let mut layer = Layer::new(gate_up_type, down_type, 0x51);
    let mut rng = Rng(0x52);
    let x = activations(TOKENS, &mut rng);
    let ids = spread_routes(&mut rng);
    let column_tile = layer.column_tile;
    let step = layer.run(&x, &ids, column_tile);
    let mut worst = (0.0f64, 0.0f64);
    let (mut error_h, mut norm_h, mut error_y, mut norm_y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let weights: Vec<[Vec<f64>; 3]> = (0..EXPERTS)
        .map(|expert| {
            [
                layer.weights(0, expert),
                layer.weights(1, expert),
                layer.weights(2, expert),
            ]
        })
        .collect();
    for (route, &expert) in ids.iter().enumerate() {
        let row: Vec<f64> = x[route / TOP_K * HIDDEN..][..HIDDEN]
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let [gate_w, up_w, down_w] = &weights[expert as usize];
        let epsilon = dot_epsilon(HIDDEN, true);
        let mut h = vec![0.0f64; INTERMEDIATE];
        let mut h_bound = vec![0.0f64; INTERMEDIATE];
        for column in 0..INTERMEDIATE {
            let (gate, gate_abs) = dot(&gate_w[column * HIDDEN..][..HIDDEN], &row);
            let (up, up_abs) = dot(&up_w[column * HIDDEN..][..HIDDEN], &row);
            let (gate_bound, up_bound) = (epsilon * gate_abs, epsilon * up_abs);
            h[column] = silu(gate) * up;
            // |SiLU'| <= 1.1; SiLU and the product are FP32 (within 2^-16
            // relative, fast exp included) and the result rounds to BF16 once.
            let exact = 1.1 * gate_bound * (up.abs() + up_bound) + silu(gate).abs() * up_bound;
            h_bound[column] = exact + (BF16_ULP + 2.0f64.powi(-16)) * (h[column].abs() + exact);
            let actual = f64::from(bf16::from_bits(step.swiglu[route][column]).to_f32());
            let error = (actual - h[column]).abs();
            assert!(
                error <= h_bound[column],
                "route {route} SwiGLU column {column}: {actual} vs {} (error {error:e}, bound {:e})",
                h[column],
                h_bound[column]
            );
            worst.0 = worst.0.max(error / h_bound[column].max(f64::MIN_POSITIVE));
            error_h += error * error;
            norm_h += h[column] * h[column];
        }
        let epsilon = dot_epsilon(INTERMEDIATE, false);
        for column in 0..HIDDEN {
            let w = &down_w[column * INTERMEDIATE..][..INTERMEDIATE];
            let reference = dot(w, &h).0;
            // The pass reads the BF16 SwiGLU exactly: its own error is the
            // weights' and the sum's, on top of the SwiGLU's.
            let bound: f64 = w
                .iter()
                .zip(h.iter().zip(&h_bound))
                .map(|(w, (h, bound))| {
                    w.abs() * bound + epsilon * (w.abs() + HALF_MIN_NORMAL) * (h.abs() + bound)
                })
                .sum();
            let actual = f64::from(f32::from_bits(step.route(route)[column]));
            let error = (actual - reference).abs();
            assert!(
                error <= bound,
                "route {route} output column {column}: {actual} vs {reference} (error {error:e}, bound {bound:e})"
            );
            worst.1 = worst.1.max(error / bound.max(f64::MIN_POSITIVE));
            error_y += error * error;
            norm_y += reference * reference;
        }
    }
    eprintln!(
        "{gate_up_type:?}/{down_type:?}: SwiGLU relative RMS error {:.3e} (worst error/bound {:.3}), \
         output relative RMS error {:.3e} (worst error/bound {:.3})",
        (error_h / norm_h).sqrt(),
        worst.0,
        (error_y / norm_y).sqrt(),
        worst.1
    );
}

#[test]
fn q4_k_matches_f64_reference() {
    matches_f64_reference(GgmlType::Q4_K, GgmlType::Q4_K);
}

#[test]
fn q6_k_down_matches_f64_reference() {
    matches_f64_reference(GgmlType::Q4_K, GgmlType::Q6_K);
}

#[test]
fn q6_k_gate_up_matches_f64_reference() {
    matches_f64_reference(GgmlType::Q6_K, GgmlType::Q4_K);
}

#[test]
fn register_path_is_the_matmul_on_this_device() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let exact = moe_tiles_column_tile_exact(&device, &mut registry).expect("probe");
    eprintln!(
        "{}: register path bitwise the matmul tile: {exact}",
        device.name()
    );
    // The order was derived on Apple10; a device that reorders must say so
    // here rather than silently lose the register path.
    if device.name().contains("M5") {
        assert!(
            exact,
            "the register path no longer matches matmul2d on {}",
            device.name()
        );
    }
}

fn is_deterministic_and_row_independent(gate_up_type: GgmlType, down_type: GgmlType) {
    let mut layer = Layer::new(gate_up_type, down_type, 0x61);
    let mut rng = Rng(0x62);
    let x = activations(TOKENS, &mut rng);
    let ids = spread_routes(&mut rng);
    let column_tile = layer.column_tile;
    let full = layer.run(&x, &ids, column_tile);
    assert!(
        full.output
            .iter()
            .all(|bits| *bits != UNWRITTEN && f32::from_bits(*bits).is_finite()),
        "a live route was not written or is not finite"
    );

    // Determinism: the same step twice, and without the register path.
    let again = layer.run(&x, &ids, column_tile);
    assert!(
        again.output == full.output && again.swiglu == full.swiglu,
        "two runs differ"
    );
    let matmul_only = layer.run(&x, &ids, false);
    assert!(
        matmul_only.output == full.output && matmul_only.swiglu == full.swiglu,
        "the register path changed a result"
    );

    // Each row alone, every other row masked (and NaN): its experts each
    // hold one route, so the row moves from its 2-, 8-, 16- and 32-row
    // tiles to the register path (or a one-row matmul without it).
    for token in 0..TOKENS {
        let mut alone_ids = vec![MASK; TOKENS * TOP_K];
        alone_ids[token * TOP_K..(token + 1) * TOP_K]
            .copy_from_slice(&ids[token * TOP_K..(token + 1) * TOP_K]);
        let mut alone_x = vec![f32::NAN; TOKENS * HIDDEN];
        alone_x[token * HIDDEN..(token + 1) * HIDDEN]
            .copy_from_slice(&x[token * HIDDEN..(token + 1) * HIDDEN]);
        for register in [column_tile, false] {
            let alone = layer.run(&alone_x, &alone_ids, register);
            for route in token * TOP_K..(token + 1) * TOP_K {
                assert!(
                    alone.route(route) == full.route(route) && alone.swiglu[route] == full.swiglu[route],
                    "row {token} route {route} depends on the other rows (register path {register})"
                );
            }
        }
    }

    // The rows in another order (other lanes, other places in their tiles)
    // and as a narrower step.
    let order: Vec<usize> = (0..TOKENS).rev().collect();
    let permuted_x: Vec<f32> = order
        .iter()
        .flat_map(|&t| x[t * HIDDEN..(t + 1) * HIDDEN].to_vec())
        .collect();
    let permuted_ids: Vec<u32> = order
        .iter()
        .flat_map(|&t| ids[t * TOP_K..(t + 1) * TOP_K].to_vec())
        .collect();
    let permuted = layer.run(&permuted_x, &permuted_ids, column_tile);
    for (place, &token) in order.iter().enumerate() {
        for slot in 0..TOP_K {
            assert!(
                permuted.route(place * TOP_K + slot) == full.route(token * TOP_K + slot),
                "row {token} slot {slot} depends on its place in the step"
            );
        }
    }
    let narrow = layer.run(&x[..8 * HIDDEN], &ids[..8 * TOP_K], column_tile);
    assert!(
        narrow.output[..] == full.output[..8 * TOP_K * HIDDEN],
        "the first lane depends on the step's width"
    );
}

#[test]
fn q4_k_is_deterministic_and_row_independent() {
    is_deterministic_and_row_independent(GgmlType::Q4_K, GgmlType::Q4_K);
}

#[test]
fn q6_k_is_deterministic_and_row_independent() {
    is_deterministic_and_row_independent(GgmlType::Q6_K, GgmlType::Q6_K);
}

#[test]
fn masked_rows_cost_no_tile_and_are_neither_read_nor_written() {
    let mut layer = Layer::new(GgmlType::Q4_K, GgmlType::Q6_K, 0x71);
    let mut rng = Rng(0x72);
    let x = activations(TOKENS, &mut rng);
    let ids = spread_routes(&mut rng);
    let column_tile = layer.column_tile;
    let full = layer.run(&x, &ids, column_tile);

    // Each lane of eight rows verifies a prefix; the rest are masked and
    // their inputs NaN.
    let verify_rows = [5usize, 8, 1, 3];
    let live = |token: usize| token % 8 < verify_rows[token / 8];
    let mut masked_ids = ids.clone();
    let mut masked_x = x.clone();
    for token in (0..TOKENS).filter(|token| !live(*token)) {
        masked_ids[token * TOP_K..(token + 1) * TOP_K].fill(MASK);
        masked_x[token * HIDDEN..(token + 1) * HIDDEN].fill(f32::NAN);
    }
    let masked = layer.run(&masked_x, &masked_ids, column_tile);
    for route in 0..TOKENS * TOP_K {
        if live(route / TOP_K) {
            assert!(
                masked.route(route) == full.route(route),
                "live route {route} changed when other rows were masked"
            );
        } else {
            assert!(
                masked.route(route).iter().all(|bits| *bits == UNWRITTEN),
                "masked route {route} was written"
            );
            assert!(
                masked.swiglu[route].is_empty(),
                "masked route {route} was scheduled"
            );
        }
    }

    // The passes launch one tile per live expert (no expert holds more than
    // 32 routes here) and nothing for the masked rows.
    let mut experts = [false; EXPERTS];
    masked_ids
        .iter()
        .filter(|id| **id != MASK)
        .for_each(|id| experts[*id as usize] = true);
    let tiles = experts.iter().filter(|live| **live).count() as u32;
    assert_eq!(
        masked.grids,
        [
            (INTERMEDIATE / 32) as u32,
            tiles,
            1,
            (HIDDEN / 64) as u32,
            tiles,
            1
        ],
        "the indirect grids are not the live tiles"
    );
    assert!(tiles < full.grids[1], "masking rows removed no tile");

    // Every row masked: the grids keep one tile row, which returns at once.
    let none = layer.run(
        &vec![f32::NAN; TOKENS * HIDDEN],
        &vec![MASK; TOKENS * TOP_K],
        column_tile,
    );
    assert!(
        none.output.iter().all(|bits| *bits == UNWRITTEN),
        "a fully masked step wrote output"
    );
    assert_eq!(none.grids[1], 1);
}

#[test]
fn rejects_undersized_buffers_and_unscheduled_passes() {
    let mut layer = Layer::new(GgmlType::Q4_K, GgmlType::Q4_K, 0x81);
    let device = &layer.device;
    let params = MoeTilesParams {
        n_tokens: TOKENS as u32,
        top_k: TOP_K as u32,
        n_experts: EXPERTS as u32,
        hidden: HIDDEN as u32,
        intermediate: INTERMEDIATE as u32,
        masked_routes: false,
        column_tile: false,
        shared_planes: false,
    };
    let rows =
        moe_tiles_grouped_rows(params.n_tokens, params.top_k, params.n_experts).expect("rows");
    assert_eq!(rows, TOKENS * TOP_K + EXPERTS + 31);
    let bf16_rows = |rows: usize, width: usize| {
        device
            .alloc_buffer(rows * width * 2, DType::BF16, vec![rows, width])
            .expect("rows")
    };
    let gathered = bf16_rows(rows, HIDDEN);
    let intermediate = bf16_rows(rows, INTERMEDIATE);
    let short = bf16_rows(rows - 1, HIDDEN);
    let grids = device
        .alloc_buffer(
            MOE_TILES_GRID_WORDS * 4,
            DType::U32,
            vec![MOE_TILES_GRID_WORDS],
        )
        .expect("grids");
    let input = device
        .alloc_buffer(TOKENS * HIDDEN * 4, DType::F32, vec![TOKENS, HIDDEN])
        .expect("input");
    let ids = device
        .alloc_buffer(TOKENS * TOP_K * 4, DType::U32, vec![TOKENS * TOP_K])
        .expect("ids");
    let mut output = device
        .alloc_buffer(
            TOKENS * TOP_K * HIDDEN * 4,
            DType::F32,
            vec![TOKENS * TOP_K, HIDDEN],
        )
        .expect("output");
    output
        .as_mut_slice::<u32>()
        .expect("output slice")
        .fill(UNWRITTEN);
    let mut scratch = IdMmScratch::alloc(device, EXPERTS as u32, TOKENS as u32).expect("scratch");
    let mut encoder = device.command_encoder().expect("encoder");

    let undersized = MoeTilesBuffers {
        gathered: &short,
        intermediate: &intermediate,
        grids: &grids,
    };
    let error = moe_tiles_gather(
        &mut encoder,
        &mut layer.registry,
        device,
        &input,
        &ids,
        &undersized,
        &scratch,
        &params,
    )
    .expect_err("the tile slack must be reserved");
    assert!(
        error.to_string().contains("gathered"),
        "unexpected error: {error}"
    );

    let buffers = MoeTilesBuffers {
        gathered: &gathered,
        intermediate: &intermediate,
        grids: &grids,
    };
    let error = moe_tiles_gate_up(
        &mut encoder,
        &mut layer.registry,
        device,
        &layer.prepared[0],
        &layer.prepared[1],
        &ids,
        &buffers,
        &scratch,
        &params,
    )
    .expect_err("a pass needs the schedule of this command");
    assert!(
        error.to_string().contains("schedule"),
        "unexpected error: {error}"
    );

    moe_tiles_schedule(
        &mut encoder,
        &mut layer.registry,
        device,
        &ids,
        &mut scratch,
        &params,
    )
    .expect("schedule");
    moe_tiles_gather(
        &mut encoder,
        &mut layer.registry,
        device,
        &input,
        &ids,
        &buffers,
        &scratch,
        &params,
    )
    .expect("gather");
    encoder.memory_barrier();
    let error = moe_tiles_gate_up(
        &mut encoder,
        &mut layer.registry,
        device,
        &layer.prepared[0],
        &layer.prepared[2],
        &ids,
        &buffers,
        &scratch,
        &params,
    )
    .expect_err("up planes of the down shape");
    assert!(
        error.to_string().contains("up planes"),
        "unexpected error: {error}"
    );
    let error = moe_tiles_down(
        &mut encoder,
        &mut layer.registry,
        device,
        &layer.prepared[2],
        &ids,
        &buffers,
        &scratch,
        &input,
        &params,
    )
    .expect_err("an output of one row per token");
    assert!(
        error.to_string().contains("output"),
        "unexpected error: {error}"
    );
    encoder.memory_barrier();
    moe_tiles_down(
        &mut encoder,
        &mut layer.registry,
        device,
        &layer.prepared[2],
        &ids,
        &buffers,
        &scratch,
        &output,
        &params,
    )
    .expect("down");
    // The zero route table repeats an expert within a row: the schedule is
    // poisoned and the passes write nothing.
    encoder.commit_and_wait().expect("commit poisoned step");
    assert!(output
        .as_slice::<u32>()
        .expect("output")
        .iter()
        .all(|bits| *bits == UNWRITTEN));
}

/// A shared expert as its own step: one route per row whose id is the row's
/// block of eight, all blocks over one set of planes. Returns the FP32 bits
/// of `[rows, HIDDEN]`.
fn run_shared(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    prepared: &[PreparedGroupedM8Experts; 3],
    x: &[f32],
    column_tile: bool,
) -> Vec<u32> {
    let rows = x.len() / HIDDEN;
    let blocks = rows.div_ceil(8) as u32;
    let params = MoeTilesParams {
        n_tokens: rows as u32,
        top_k: 1,
        n_experts: blocks,
        hidden: HIDDEN as u32,
        intermediate: INTERMEDIATE as u32,
        masked_routes: false,
        column_tile,
        shared_planes: true,
    };
    let grouped = moe_tiles_grouped_rows(params.n_tokens, 1, blocks).expect("rows");
    let nan_rows = |width: usize| {
        let mut buffer = device
            .alloc_buffer(grouped * width * 2, DType::BF16, vec![grouped, width])
            .expect("grouped rows");
        buffer
            .as_mut_slice::<u16>()
            .expect("grouped slice")
            .fill(0x7fc0);
        buffer
    };
    let (gathered, intermediate) = (nan_rows(HIDDEN), nan_rows(INTERMEDIATE));
    let grids = device
        .alloc_buffer(
            MOE_TILES_GRID_WORDS * 4,
            DType::U32,
            vec![MOE_TILES_GRID_WORDS],
        )
        .expect("grids");
    let mut input = device
        .alloc_buffer(rows * HIDDEN * 4, DType::F32, vec![rows, HIDDEN])
        .expect("input");
    input
        .as_mut_slice::<f32>()
        .expect("input")
        .copy_from_slice(x);
    let mut ids = device
        .alloc_buffer(rows * 4, DType::U32, vec![rows])
        .expect("ids");
    for (row, id) in ids
        .as_mut_slice::<u32>()
        .expect("ids")
        .iter_mut()
        .enumerate()
    {
        *id = (row / 8) as u32;
    }
    let mut output = device
        .alloc_buffer(rows * HIDDEN * 4, DType::F32, vec![rows, HIDDEN])
        .expect("output");
    output
        .as_mut_slice::<u32>()
        .expect("output")
        .fill(UNWRITTEN);
    let mut scratch =
        IdMmScratch::alloc_with_top_k(device, blocks, rows as u32, 1).expect("scratch");
    let buffers = MoeTilesBuffers {
        gathered: &gathered,
        intermediate: &intermediate,
        grids: &grids,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    moe_tiles_schedule(&mut encoder, registry, device, &ids, &mut scratch, &params)
        .expect("schedule");
    moe_tiles_gather(
        &mut encoder,
        registry,
        device,
        &input,
        &ids,
        &buffers,
        &scratch,
        &params,
    )
    .expect("gather");
    encoder.memory_barrier();
    moe_tiles_gate_up(
        &mut encoder,
        registry,
        device,
        &prepared[0],
        &prepared[1],
        &ids,
        &buffers,
        &scratch,
        &params,
    )
    .expect("gate/up");
    encoder.memory_barrier();
    moe_tiles_down(
        &mut encoder,
        registry,
        device,
        &prepared[2],
        &ids,
        &buffers,
        &scratch,
        &output,
        &params,
    )
    .expect("down");
    encoder.commit_and_wait().expect("commit shared step");
    let bits = output.as_slice::<u32>().expect("output bits").to_vec();
    bits
}

#[test]
fn shared_planes_rows_are_independent_and_match_f64() {
    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mut rng = Rng(0xa1);
    let one_expert = |ggml_type: GgmlType, n: usize, k: usize, rng: &mut Rng| {
        let bytes = n * (k / 256) * ggml_type.block_bytes() as usize;
        let stack = quantized(&device, ggml_type, n, k, rng);
        stack.slice_view(0, bytes)
    };
    let sources = [
        one_expert(GgmlType::Q4_K, INTERMEDIATE, HIDDEN, &mut rng),
        one_expert(GgmlType::Q4_K, INTERMEDIATE, HIDDEN, &mut rng),
        one_expert(GgmlType::Q6_K, HIDDEN, INTERMEDIATE, &mut rng),
    ];
    let shape = |ggml_type: GgmlType, n: usize, k: usize| GgmlQuantizedMatmulIdParams {
        n_tokens: 1,
        top_k: 1,
        n: n as u32,
        k: k as u32,
        n_experts: 1,
        expert_stride: (n * (k / 256) * ggml_type.block_bytes() as usize) as u64,
        ggml_type,
    };
    let mut encoder = device.command_encoder().expect("prepare encoder");
    let mut prepare = |source: &MlxBuffer, shape: GgmlQuantizedMatmulIdParams| {
        prepare_quantized_matmul_id_ggml_grouped_m8(
            &mut encoder,
            &mut registry,
            &device,
            source,
            &shape,
        )
        .expect("prepare")
    };
    let prepared = [
        prepare(&sources[0], shape(GgmlType::Q4_K, INTERMEDIATE, HIDDEN)),
        prepare(&sources[1], shape(GgmlType::Q4_K, INTERMEDIATE, HIDDEN)),
        prepare(&sources[2], shape(GgmlType::Q6_K, HIDDEN, INTERMEDIATE)),
    ];
    encoder.commit_and_wait().expect("commit preparation");
    let column_tile = moe_tiles_column_tile_exact(&device, &mut registry).expect("probe");

    let x = activations(TOKENS, &mut rng);
    let full = run_shared(&device, &mut registry, &prepared, &x, column_tile);
    assert!(full
        .iter()
        .all(|bits| *bits != UNWRITTEN && f32::from_bits(*bits).is_finite()));
    assert!(
        full == run_shared(&device, &mut registry, &prepared, &x, column_tile),
        "two runs differ"
    );
    // Every row alone, and steps whose last block holds one, two and three rows.
    for row in 0..TOKENS {
        let alone = run_shared(
            &device,
            &mut registry,
            &prepared,
            &x[row * HIDDEN..(row + 1) * HIDDEN],
            column_tile,
        );
        assert!(
            alone[..] == full[row * HIDDEN..(row + 1) * HIDDEN],
            "row {row} depends on its block"
        );
    }
    for rows in [9, 10, 11, 17, 24] {
        let step = run_shared(
            &device,
            &mut registry,
            &prepared,
            &x[..rows * HIDDEN],
            column_tile,
        );
        assert!(
            step[..] == full[..rows * HIDDEN],
            "a {rows}-row step changed a row"
        );
    }
    // F64 reference of the first rows.
    let weights: Vec<Vec<f64>> = [
        (GgmlType::Q4_K, INTERMEDIATE, HIDDEN),
        (GgmlType::Q4_K, INTERMEDIATE, HIDDEN),
        (GgmlType::Q6_K, HIDDEN, INTERMEDIATE),
    ]
    .iter()
    .zip(&sources)
    .map(|((ggml_type, n, k), source)| {
        dequantize(*ggml_type, source.as_slice::<u8>().expect("bytes"), *n, *k)
    })
    .collect();
    let (mut squared, mut norm) = (0.0f64, 0.0f64);
    for row in 0..4 {
        let exact: Vec<f64> = x[row * HIDDEN..(row + 1) * HIDDEN]
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let h: Vec<f64> = (0..INTERMEDIATE)
            .map(|column| {
                silu(dot(&weights[0][column * HIDDEN..][..HIDDEN], &exact).0)
                    * dot(&weights[1][column * HIDDEN..][..HIDDEN], &exact).0
            })
            .collect();
        for column in 0..HIDDEN {
            let reference = dot(&weights[2][column * INTERMEDIATE..][..INTERMEDIATE], &h).0;
            let actual = f64::from(f32::from_bits(full[row * HIDDEN + column]));
            squared += (actual - reference).powi(2);
            norm += reference * reference;
        }
    }
    let relative = (squared / norm).sqrt();
    eprintln!("shared planes: relative RMS error against F64 {relative:.3e}");
    assert!(
        relative < 1.0e-2,
        "shared planes: relative RMS error {relative:e}"
    );
}

/// The two SwiGLU orders the earlier grouped-M8 kernels offered, per route:
/// BF16 SwiGLU rows and FP32 down rows in route order.
struct GroupedM8Step {
    swiglu: Vec<f32>,
    output: Vec<f32>,
    /// FP32 gate and up sums (the served route only).
    gate_up: Option<(Vec<f32>, Vec<f32>)>,
}

impl Layer {
    /// `quantized_matmul_id_ggml_grouped_m8_*` over the same planes: the
    /// served route (FP32 gate and up, FP32 SwiGLU, BF16 gather for down) or
    /// the withdrawn one-pass route (`fused`: BF16 gate and up before a BF16
    /// SwiGLU).
    fn run_grouped_m8(&mut self, x: &[f32], ids: &[u32], fused: bool) -> GroupedM8Step {
        let device = &self.device;
        let registry = &mut self.registry;
        let routes = ids.len();
        let tokens = routes / TOP_K;
        let f32_rows = |rows: usize, width: usize| {
            device
                .alloc_buffer(rows * width * 4, DType::F32, vec![rows, width])
                .expect("f32 rows")
        };
        let bf16_rows = |rows: usize, width: usize| {
            device
                .alloc_buffer(rows * width * 2, DType::BF16, vec![rows, width])
                .expect("bf16 rows")
        };
        let mut input = f32_rows(tokens, HIDDEN);
        input
            .as_mut_slice::<f32>()
            .expect("input")
            .copy_from_slice(x);
        let mut id_buffer = device
            .alloc_buffer(routes * 4, DType::U32, vec![routes])
            .expect("ids");
        id_buffer
            .as_mut_slice::<u32>()
            .expect("ids")
            .copy_from_slice(ids);
        let rows = mlx_native::grouped_m8_gathered_rows(tokens as u32, TOP_K as u32, EXPERTS as u32)
            .expect("gathered rows");
        let gathered = bf16_rows(rows, HIDDEN);
        let output = f32_rows(routes, HIDDEN);
        let mut scratch =
            IdMmScratch::alloc(device, EXPERTS as u32, tokens as u32).expect("scratch");
        let gate_up = GgmlQuantizedMatmulIdParams {
            n_tokens: tokens as u32,
            top_k: TOP_K as u32,
            n: INTERMEDIATE as u32,
            k: HIDDEN as u32,
            n_experts: EXPERTS as u32,
            expert_stride: 0,
            ggml_type: self.gate_up_type,
        };
        let down = GgmlQuantizedMatmulIdParams {
            n: HIDDEN as u32,
            k: INTERMEDIATE as u32,
            ggml_type: self.down_type,
            ..gate_up
        };
        let mut encoder = device.command_encoder().expect("encoder");
        let mut project = |encoder: &mut mlx_native::CommandEncoder,
                           scratch: &mut IdMmScratch,
                           source: &MlxBuffer,
                           prepared: &PreparedGroupedM8Experts,
                           target: &MlxBuffer,
                           aux: Option<&MlxBuffer>,
                           params: &GgmlQuantizedMatmulIdParams,
                           slotted: bool,
                           reuse: (bool, bool),
                           kinds: (GroupedM8Input, GroupedM8Epilogue)| {
            quantized_matmul_id_ggml_grouped_m8_epilogue(
                encoder, registry, device, source, &gathered, prepared, &id_buffer, target, aux,
                scratch, params, slotted, reuse.0, reuse.1, kinds.0, kinds.1,
            )
            .expect("grouped M8 projection");
        };
        let swiglu_bf16 = bf16_rows(routes, INTERMEDIATE);
        let swiglu_f32 = f32_rows(routes, INTERMEDIATE);
        let gate = f32_rows(routes, INTERMEDIATE);
        let up = f32_rows(routes, INTERMEDIATE);
        if fused {
            drop(project);
            quantized_matmul_id_ggml_grouped_m8_gate_up(
                &mut encoder,
                &mut self.registry,
                device,
                &input,
                &gathered,
                &self.prepared[0],
                &self.prepared[1],
                &id_buffer,
                &swiglu_bf16,
                &mut scratch,
                &gate_up,
                GroupedM8Input::F32,
            )
            .expect("one-pass gate/up");
            quantized_matmul_id_ggml_grouped_m8_epilogue(
                &mut encoder,
                &mut self.registry,
                device,
                &swiglu_bf16,
                &gathered,
                &self.prepared[2],
                &id_buffer,
                &output,
                None,
                &mut scratch,
                &down,
                true,
                true,
                false,
                GroupedM8Input::Bf16,
                GroupedM8Epilogue::F32,
            )
            .expect("BF16-input down");
        } else {
            let mut count = device
                .alloc_buffer(4, DType::U32, vec![1])
                .expect("silu count");
            let elements = (routes * INTERMEDIATE) as u32;
            count.as_mut_slice::<u32>().expect("count")[0] = elements;
            project(
                &mut encoder,
                &mut scratch,
                &input,
                &self.prepared[0],
                &gate,
                None,
                &gate_up,
                false,
                (false, false),
                (GroupedM8Input::F32, GroupedM8Epilogue::F32),
            );
            project(
                &mut encoder,
                &mut scratch,
                &input,
                &self.prepared[1],
                &up,
                None,
                &gate_up,
                false,
                (true, true),
                (GroupedM8Input::F32, GroupedM8Epilogue::F32),
            );
            encoder.memory_barrier();
            drop(project);
            dispatch_silu_mul(
                &mut encoder,
                &mut self.registry,
                device.metal_device(),
                &gate,
                &up,
                &swiglu_f32,
                &count,
                elements,
            )
            .expect("FP32 SwiGLU");
            encoder.memory_barrier();
            quantized_matmul_id_ggml_grouped_m8_epilogue(
                &mut encoder,
                &mut self.registry,
                device,
                &swiglu_f32,
                &gathered,
                &self.prepared[2],
                &id_buffer,
                &output,
                None,
                &mut scratch,
                &down,
                true,
                true,
                false,
                GroupedM8Input::F32,
                GroupedM8Epilogue::F32,
            )
            .expect("FP32-input down");
        }
        encoder.commit_and_wait().expect("commit grouped M8 step");
        let swiglu = if fused {
            swiglu_bf16
                .as_slice::<u16>()
                .expect("BF16 SwiGLU")
                .iter()
                .map(|bits| bf16::from_bits(*bits).to_f32())
                .collect()
        } else {
            // The down pass reads this row rounded to BF16.
            swiglu_f32
                .as_slice::<f32>()
                .expect("FP32 SwiGLU")
                .iter()
                .map(|value| bf16::from_f32(*value).to_f32())
                .collect()
        };
        GroupedM8Step {
            swiglu,
            output: output.as_slice::<f32>().expect("output").to_vec(),
            gate_up: (!fused).then(|| {
                (
                    gate.as_slice::<f32>().expect("gate").to_vec(),
                    up.as_slice::<f32>().expect("up").to_vec(),
                )
            }),
        }
    }
}

/// One 32-input step of `matmul2d` multiply-accumulate onto `sum`.
fn matmul_step(sum: f32, products: &[f32]) -> f32 {
    let q = |k: usize| products[2 * k] + products[2 * k + 1];
    let mut sum = sum;
    for half in 0..2 {
        for m in 0..4 {
            sum += q(8 * half + m) + q(8 * half + m + 4);
        }
    }
    sum
}

/// Splash's arithmetic for one route on the CPU (`moe_gguf.metal:69-115`,
/// `gguf_staged.h`, `gguf_column_tile.h:20-68`, `gguf_tile.h:10-14`): BF16
/// input, weights rounded to half, FP32 sums in the staged tile's order or
/// (`column`, an expert of one or two routes) one FP32 chain per group,
/// gate and up each rounded to BF16 before `BF16(up * SiLU(gate))`, and a
/// down pass over that row. Returns the BF16 SwiGLU row and the FP32 down
/// sums (which Splash then stores as BF16).
fn splash_order(x: &[f32], weights: &[Vec<f64>; 3], column: bool) -> (Vec<f32>, Vec<f32>) {
    let half = |value: f64| f16::from_f32(value as f32).to_f32();
    let dot = |w: &[f64], input: &[f32]| {
        let mut sum = 0.0f32;
        for (w, input) in w.chunks(32).zip(input.chunks(32)) {
            let products: Vec<f32> = w.iter().zip(input).map(|(w, x)| half(*w) * x).collect();
            sum = if column {
                sum + products
                    .iter()
                    .fold(0.0f32, |chain, product| chain + product)
            } else {
                matmul_step(sum, &products)
            };
        }
        sum
    };
    let input: Vec<f32> = x
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    let round = |value: f32| bf16::from_f32(value).to_f32();
    let swiglu: Vec<f32> = (0..INTERMEDIATE)
        .map(|column| {
            let gate = round(dot(&weights[0][column * HIDDEN..][..HIDDEN], &input));
            let up = round(dot(&weights[1][column * HIDDEN..][..HIDDEN], &input));
            round(up * (gate / (1.0 + (-gate).exp())))
        })
        .collect();
    let output = (0..HIDDEN)
        .map(|column| {
            dot(
                &weights[2][column * INTERMEDIATE..][..INTERMEDIATE],
                &swiglu,
            )
        })
        .collect();
    (swiglu, output)
}

/// ADR-067 withdrew the one-pass BF16 gate/up route after one failed task
/// run. This measures, on one set of inputs: whether that kernel stores
/// `BF16(SiLU(BF16(gate)) * BF16(up))` of the very sums the served route's
/// kernels produce (a kernel defect would show here), and the error against
/// F64 of the served route, the withdrawn route, Splash's arithmetic order
/// and the live tiles.
///
/// `cargo +1.96.0 test --release --locked --test moe_expert_tiles -- --ignored --nocapture fused_bf16`
#[test]
#[ignore = "arithmetic evidence for the withdrawn one-pass BF16 route"]
fn fused_bf16_route_against_f64_and_splash_order() {
    for (gate_up_type, down_type) in [
        (GgmlType::Q4_K, GgmlType::Q4_K),
        (GgmlType::Q4_K, GgmlType::Q6_K),
    ] {
        let mut layer = Layer::new(gate_up_type, down_type, 0x91);
        let mut rng = Rng(0x92);
        let x = activations(TOKENS, &mut rng);
        let ids = spread_routes(&mut rng);
        let mut counts = [0usize; EXPERTS];
        ids.iter().for_each(|&id| counts[id as usize] += 1);

        let served = layer.run_grouped_m8(&x, &ids, false);
        let withdrawn = layer.run_grouped_m8(&x, &ids, true);
        let again = layer.run_grouped_m8(&x, &ids, true);
        let column_tile = layer.column_tile;
        let tiles = layer.run(&x, &ids, column_tile);
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let repeatable = bits(&withdrawn.swiglu) == bits(&again.swiglu)
            && bits(&withdrawn.output) == bits(&again.output);
        // The one-pass kernel's SwiGLU from the served kernels' FP32 sums.
        // The CPU's exp is not Metal's: an element may land on the
        // neighbouring BF16 value, never further.
        let round = |value: f32| bf16::from_f32(value).to_f32();
        let (gate, up) = served.gate_up.as_ref().expect("served sums");
        let (mut exact, mut off_by_more) = (0usize, 0usize);
        // Deviations by the routes of the element's expert and by its place
        // in the expert's route list (ascending token order).
        let mut by_count = std::collections::BTreeMap::<usize, (usize, usize)>::new();
        let mut by_place = std::collections::BTreeMap::<usize, usize>::new();
        let mut seen = [0usize; EXPERTS];
        let mut worst = (0.0f32, 0usize, 0.0f32, 0.0f32);
        let mut worst_absolute = 0.0f32;
        let swiglu_rms = (withdrawn
            .swiglu
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            / withdrawn.swiglu.len() as f64)
            .sqrt();
        for (route, &expert) in ids.iter().enumerate() {
            let place = seen[expert as usize];
            seen[expert as usize] += 1;
            for column in 0..INTERMEDIATE {
                let index = route * INTERMEDIATE + column;
                let (gate, up) = (round(gate[index]), round(up[index]));
                let expected = round((gate / (1.0 + (-gate).exp())) * up);
                let actual = withdrawn.swiglu[index];
                let ulp = expected.abs().max(f32::MIN_POSITIVE) * 2.0f32.powi(-7);
                exact += usize::from(actual.to_bits() == expected.to_bits());
                let entry = by_count.entry(counts[expert as usize]).or_default();
                entry.1 += 1;
                if (actual - expected).abs() > ulp {
                    worst_absolute = worst_absolute.max((actual - expected).abs());
                    off_by_more += 1;
                    entry.0 += 1;
                    *by_place.entry(place).or_default() += 1;
                    let relative =
                        (actual - expected).abs() / expected.abs().max(f32::MIN_POSITIVE);
                    if relative > worst.0 {
                        worst = (relative, index, actual, expected);
                    }
                }
            }
        }
        eprintln!(
            "  one-pass deviations by routes of the expert (deviating, elements): {by_count:?}"
        );
        eprintln!("  one-pass deviations by place in the expert's route list: {by_place:?}");
        eprintln!(
            "  largest relative deviation: element {} (route {}, column {}): {} against {} ({:.3e} relative); \
             largest deviation beyond one ulp over the SwiGLU's RMS: {:.3e}",
            worst.1,
            worst.1 / INTERMEDIATE,
            worst.1 % INTERMEDIATE,
            worst.2,
            worst.3,
            worst.0,
            f64::from(worst_absolute) / swiglu_rms
        );
        // A deviation beyond one BF16 ulp is a SwiGLU below the rounding
        // noise of its row: Metal's exp and the host's differ where
        // SiLU(gate) underflows.
        assert!(
            f64::from(worst_absolute) < 1.0e-3 * swiglu_rms,
            "the one-pass kernel does not store the SwiGLU of its sums"
        );

        // F64 reference and Splash's order per route.
        let weights: Vec<[Vec<f64>; 3]> = (0..EXPERTS)
            .map(|expert| {
                [
                    layer.weights(0, expert),
                    layer.weights(1, expert),
                    layer.weights(2, expert),
                ]
            })
            .collect();
        let mut reference = GroupedM8Step {
            swiglu: Vec::new(),
            output: Vec::new(),
            gate_up: None,
        };
        let mut splash = GroupedM8Step {
            swiglu: Vec::new(),
            output: Vec::new(),
            gate_up: None,
        };
        let mut splash_bf16_output = Vec::new();
        for (route, &expert) in ids.iter().enumerate() {
            let row = &x[route / TOP_K * HIDDEN..][..HIDDEN];
            let exact: Vec<f64> = row.iter().map(|value| f64::from(*value)).collect();
            let [gate_w, up_w, down_w] = &weights[expert as usize];
            let h: Vec<f64> = (0..INTERMEDIATE)
                .map(|column| {
                    silu(dot(&gate_w[column * HIDDEN..][..HIDDEN], &exact).0)
                        * dot(&up_w[column * HIDDEN..][..HIDDEN], &exact).0
                })
                .collect();
            reference.output.extend(
                (0..HIDDEN).map(|column| {
                    dot(&down_w[column * INTERMEDIATE..][..INTERMEDIATE], &h).0 as f32
                }),
            );
            reference.swiglu.extend(h.iter().map(|value| *value as f32));
            let (swiglu, output) =
                splash_order(row, &weights[expert as usize], counts[expert as usize] <= 2);
            splash_bf16_output.extend(output.iter().map(|value| bf16::from_f32(*value).to_f32()));
            splash.swiglu.extend(swiglu);
            splash.output.extend(output);
        }
        let tiles_step = GroupedM8Step {
            swiglu: tiles
                .swiglu
                .iter()
                .flat_map(|row| row.iter().map(|bits| bf16::from_bits(*bits).to_f32()))
                .collect(),
            output: tiles
                .output
                .iter()
                .map(|bits| f32::from_bits(*bits))
                .collect(),
            gate_up: None,
        };
        // Relative RMS error and the largest error over the RMS of the reference.
        let error = |actual: &[f32], expected: &[f32]| {
            let (mut squared, mut norm, mut worst) = (0.0f64, 0.0f64, 0.0f64);
            for (actual, expected) in actual.iter().zip(expected) {
                let difference = f64::from(*actual) - f64::from(*expected);
                squared += difference * difference;
                norm += f64::from(*expected) * f64::from(*expected);
                worst = worst.max(difference.abs());
            }
            let rms = (norm / expected.len() as f64).sqrt();
            ((squared / norm).sqrt(), worst / rms)
        };
        eprintln!(
            "{gate_up_type:?}/{down_type:?}, {} routes ({} of experts with one or two routes): the one-pass \
             SwiGLU is BF16(SiLU(BF16(gate)) * BF16(up)) of the served sums in {exact} of {} elements \
             bitwise, {off_by_more} further than one BF16 ulp; repeatable: {repeatable}",
            ids.len(),
            ids.iter().filter(|id| counts[**id as usize] <= 2).count(),
            withdrawn.swiglu.len()
        );
        eprintln!("  route                                   SwiGLU rel RMS  (max/rms)   down rel RMS  (max/rms)");
        for (name, step) in [
            ("served: FP32 gate/up, FP32 SwiGLU", &served),
            ("withdrawn: one pass, BF16 gate/up", &withdrawn),
            ("Splash order: half weights, BF16", &splash),
            ("live tiles: half weights, FP32 SwiGLU", &tiles_step),
        ] {
            let (h_rms, h_max) = error(&step.swiglu, &reference.swiglu);
            let (y_rms, y_max) = error(&step.output, &reference.output);
            eprintln!("  {name:<38} {h_rms:>10.3e} {h_max:>10.3e} {y_rms:>14.3e} {y_max:>10.3e}");
        }
        let (y_rms, y_max) = error(&splash_bf16_output, &reference.output);
        eprintln!(
            "  {:<38} {:>10} {:>10} {y_rms:>14.3e} {y_max:>10.3e}",
            "Splash order, BF16 down store", "", ""
        );
        let (h_rms, _) = error(&withdrawn.swiglu, &splash.swiglu);
        let (y_rms, _) = error(&withdrawn.output, &splash.output);
        eprintln!("  withdrawn route against Splash's order: SwiGLU {h_rms:.3e}, down {y_rms:.3e}");
        assert!(repeatable, "the one-pass kernel is not repeatable");
    }
}
