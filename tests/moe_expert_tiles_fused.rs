//! Fused MoE expert tiles (`moe_tiles_fused`): bitwise the live-tile stages
//! (`moe_tiles_*`, whose F64 reference, determinism and row independence
//! `tests/moe_expert_tiles.rs` proves) for the routed experts and the shared
//! expert, the verify-row mask applied by the schedule, the schedule words,
//! route list and grids of a 256-expert step against a CPU reference,
//! determinism, row independence, and what the route refuses.
//!
//! Every comparison prints an FNV-1a digest of the compared bits, so runs
//! under `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1` can be compared
//! with a plain run.
#![cfg(mlx_native_has_metal_tensor_artifact)]

use mlx_native::ops::dflash2_attention_pack::{
    dispatch_dflash2_mask_moe_routes, DFlash2MoeRouteMaskParams,
};
use mlx_native::{
    ggml_expert_bytes, moe_tiles_column_tile_exact, moe_tiles_down, moe_tiles_fused,
    moe_tiles_fused_grouped_rows, moe_tiles_fused_output_rows, moe_tiles_fused_schedule_words,
    moe_tiles_gate_up, moe_tiles_gather, moe_tiles_grouped_rows, moe_tiles_schedule,
    prepare_quantized_matmul_id_ggml_grouped_m8, DType, GgmlQuantizedMatmulIdParams, GgmlType,
    IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice, MoeTilesBuffers, MoeTilesFusedBuffers,
    MoeTilesFusedMask, MoeTilesFusedParams, MoeTilesFusedStage, MoeTilesFusedWeights,
    MoeTilesParams, PreparedGroupedM8Experts, MOE_TILES_FUSED_GRID_WORDS, MOE_TILES_GRID_WORDS,
};

const HIDDEN: usize = 512;
const INTERMEDIATE: usize = 256;
const EXPERTS: usize = 24;
/// The widest schedule: one expert per thread of its threadgroup, eight
/// simdgroups.
const WIDE_EXPERTS: usize = 256;
const TOP_K: usize = 8;
const TOKENS: usize = 32;
const LANE_ROWS: usize = 8;
const MASK: u32 = EXPERTS as u32;
/// Bit pattern no kernel writes: an untouched output element.
const UNWRITTEN: u32 = 0x7fc0_dead;
const POISON: u32 = 0x8000_0000;

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

fn digest(words: &[u32]) -> u64 {
    words.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, word| {
        word.to_le_bytes().iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

/// Random GGML blocks with half block scales of magnitude 2^-8..2^-7.
fn quantized(
    device: &MlxDevice,
    ggml_type: GgmlType,
    experts: usize,
    n: usize,
    k: usize,
    rng: &mut Rng,
) -> MlxBuffer {
    let stride =
        (n * (k / ggml_type.block_values() as usize) * ggml_type.block_bytes() as usize) as u64;
    let bytes = ggml_expert_bytes(ggml_type, experts as u32, n as u32, k as u32, stride)
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

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::U32, vec![values.len()])
        .expect("u32 buffer");
    buffer
        .as_mut_slice::<u32>()
        .expect("u32 slice")
        .copy_from_slice(values);
    buffer
}

fn f32_buffer(device: &MlxDevice, values: &[f32], shape: Vec<usize>) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(values.len() * 4, DType::F32, shape)
        .expect("f32 buffer");
    buffer
        .as_mut_slice::<f32>()
        .expect("f32 slice")
        .copy_from_slice(values);
    buffer
}

/// `[rows, width]` BF16 rows of NaN: rows a tile reads past its routes and
/// rows no pass writes must not reach a result.
fn nan_rows(device: &MlxDevice, rows: usize, width: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(rows * width * 2, DType::BF16, vec![rows, width])
        .expect("grouped rows");
    buffer
        .as_mut_slice::<u16>()
        .expect("grouped slice")
        .fill(0x7fc0);
    buffer
}

fn unwritten(device: &MlxDevice, rows: usize) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(rows * HIDDEN * 4, DType::F32, vec![rows, HIDDEN])
        .expect("output");
    buffer
        .as_mut_slice::<u32>()
        .expect("output slice")
        .fill(UNWRITTEN);
    buffer
}

/// The verify-row mask of a step: `verify_rows` per physical lane.
#[derive(Clone, Copy)]
struct Mask<'a> {
    verify_rows: &'a [u32],
    physical_lane: [u32; 4],
    lanes: u32,
}

impl Mask<'_> {
    fn live(&self, token: usize) -> bool {
        let lane = token / LANE_ROWS;
        lane < self.lanes as usize
            && token % LANE_ROWS
                < (self.verify_rows[self.physical_lane[lane] as usize] as usize).min(LANE_ROWS)
    }
}

struct Layer {
    device: MlxDevice,
    registry: KernelRegistry,
    experts: usize,
    /// Gate, up, down of the routed experts and of the shared expert.
    routed: [PreparedGroupedM8Experts; 3],
    shared: [PreparedGroupedM8Experts; 3],
    column_tile: bool,
}

/// What one step wrote.
struct Step {
    /// FP32 bits of `[routes, HIDDEN]`: each route's down row.
    routed: Vec<u32>,
    /// FP32 bits of `[tokens, HIDDEN]`: the shared expert's row of each token.
    shared: Vec<u32>,
    /// The route table after the step.
    ids: Vec<u32>,
    weights: Vec<u32>,
    /// Live tiles of the step (the fused schedule's word 0).
    tiles: u32,
    grids: Vec<u32>,
    /// The fused step's schedule words and route list (empty for the
    /// live-tile stages).
    schedule: Vec<u32>,
    route_list: Vec<u32>,
}

impl Step {
    fn route(&self, route: usize) -> &[u32] {
        &self.routed[route * HIDDEN..(route + 1) * HIDDEN]
    }

    fn shared_row(&self, token: usize) -> &[u32] {
        &self.shared[token * HIDDEN..(token + 1) * HIDDEN]
    }
}

impl Layer {
    fn new(
        gate_up_type: GgmlType,
        down_type: GgmlType,
        shared_down_type: GgmlType,
        seed: u64,
    ) -> Self {
        Self::with_experts(gate_up_type, down_type, shared_down_type, seed, EXPERTS)
    }

    fn with_experts(
        gate_up_type: GgmlType,
        down_type: GgmlType,
        shared_down_type: GgmlType,
        seed: u64,
        experts: usize,
    ) -> Self {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut rng = Rng(seed);
        let shape =
            |ggml_type: GgmlType, experts: usize, n: usize, k: usize| GgmlQuantizedMatmulIdParams {
                n_tokens: TOKENS as u32,
                top_k: TOP_K as u32,
                n: n as u32,
                k: k as u32,
                n_experts: experts as u32,
                expert_stride: (n * (k / 256) * ggml_type.block_bytes() as usize) as u64,
                ggml_type,
            };
        let mut encoder = device.command_encoder().expect("prepare encoder");
        let mut prepare = |ggml_type: GgmlType, experts: usize, n: usize, k: usize| {
            let source = quantized(&device, ggml_type, experts, n, k, &mut rng);
            prepare_quantized_matmul_id_ggml_grouped_m8(
                &mut encoder,
                &mut registry,
                &device,
                &source,
                &shape(ggml_type, experts, n, k),
            )
            .expect("prepare")
        };
        let routed = [
            prepare(gate_up_type, experts, INTERMEDIATE, HIDDEN),
            prepare(gate_up_type, experts, INTERMEDIATE, HIDDEN),
            prepare(down_type, experts, HIDDEN, INTERMEDIATE),
        ];
        let shared = [
            prepare(gate_up_type, 1, INTERMEDIATE, HIDDEN),
            prepare(gate_up_type, 1, INTERMEDIATE, HIDDEN),
            prepare(shared_down_type, 1, HIDDEN, INTERMEDIATE),
        ];
        encoder.commit_and_wait().expect("commit preparation");
        let column_tile =
            moe_tiles_column_tile_exact(&device, &mut registry).expect("column probe");
        Self {
            device,
            registry,
            experts,
            routed,
            shared,
            column_tile,
        }
    }

    fn fused_params(&self, tokens: usize, column_tile: bool) -> MoeTilesFusedParams {
        MoeTilesFusedParams {
            n_tokens: tokens as u32,
            top_k: TOP_K as u32,
            n_experts: self.experts as u32,
            hidden: HIDDEN as u32,
            intermediate: INTERMEDIATE as u32,
            column_tile,
        }
    }

    /// The live-tile stages over `ids` (a masked route holds `MASK`), the
    /// routed experts and the shared expert each with its own schedule.
    fn live(&mut self, x: &[f32], ids: &[u32], column_tile: bool) -> Step {
        let tokens = ids.len() / TOP_K;
        let device = &self.device;
        let input = f32_buffer(device, x, vec![tokens, HIDDEN]);
        let mut outputs = Vec::new();
        let mut tiles = 0;
        let mut grid_words = Vec::new();
        let blocks = tokens.div_ceil(LANE_ROWS);
        let shared_ids: Vec<u32> = (0..tokens)
            .map(|token| (token / LANE_ROWS) as u32)
            .collect();
        for shared in [false, true] {
            let params = MoeTilesParams {
                n_tokens: tokens as u32,
                top_k: if shared { 1 } else { TOP_K as u32 },
                n_experts: if shared {
                    blocks as u32
                } else {
                    self.experts as u32
                },
                hidden: HIDDEN as u32,
                intermediate: INTERMEDIATE as u32,
                masked_routes: !shared,
                column_tile,
                shared_planes: shared,
            };
            let planes = if shared { &self.shared } else { &self.routed };
            let table = u32_buffer(device, if shared { &shared_ids } else { ids });
            let rows = moe_tiles_grouped_rows(params.n_tokens, params.top_k, params.n_experts)
                .expect("rows");
            let gathered = nan_rows(device, rows, HIDDEN);
            let intermediate = nan_rows(device, rows, INTERMEDIATE);
            let grids = device
                .alloc_buffer(
                    MOE_TILES_GRID_WORDS * 4,
                    DType::U32,
                    vec![MOE_TILES_GRID_WORDS],
                )
                .expect("grids");
            let output = unwritten(device, params.n_tokens as usize * params.top_k as usize);
            let mut scratch = IdMmScratch::alloc_with_top_k(
                device,
                params.n_experts,
                params.n_tokens,
                params.top_k,
            )
            .expect("scratch");
            let buffers = MoeTilesBuffers {
                gathered: &gathered,
                intermediate: &intermediate,
                grids: &grids,
            };
            let mut encoder = device.command_encoder().expect("encoder");
            let registry = &mut self.registry;
            moe_tiles_schedule(
                &mut encoder,
                registry,
                device,
                &table,
                &mut scratch,
                &params,
            )
            .expect("schedule");
            moe_tiles_gather(
                &mut encoder,
                registry,
                device,
                &input,
                &table,
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
                &planes[0],
                &planes[1],
                &table,
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
                &planes[2],
                &table,
                &buffers,
                &scratch,
                &output,
                &params,
            )
            .expect("down");
            encoder.commit_and_wait().expect("commit live step");
            let word0 = scratch.htpe().as_slice::<u32>().expect("schedule")[0];
            assert!(word0 & POISON == 0, "poisoned live schedule");
            tiles += word0;
            grid_words.extend_from_slice(grids.as_slice::<u32>().expect("grid words"));
            outputs.push(output.as_slice::<u32>().expect("output bits").to_vec());
        }
        let shared = outputs.pop().expect("shared output");
        Step {
            routed: outputs.pop().expect("routed output"),
            shared,
            ids: ids.to_vec(),
            weights: Vec::new(),
            tiles,
            grids: grid_words,
            schedule: Vec::new(),
            route_list: Vec::new(),
        }
    }

    /// One fused step over the unmasked table `ids` with `routing_weights`.
    fn fused(
        &mut self,
        x: &[f32],
        ids: &[u32],
        routing_weights: &[f32],
        mask: Option<Mask<'_>>,
        column_tile: bool,
    ) -> Step {
        let tokens = ids.len() / TOP_K;
        let params = self.fused_params(tokens, column_tile);
        let device = &self.device;
        let rows = moe_tiles_fused_grouped_rows(&params).expect("rows");
        let gathered = nan_rows(device, rows, HIDDEN);
        let intermediate = nan_rows(device, rows, INTERMEDIATE);
        let words = moe_tiles_fused_schedule_words(&params).expect("schedule words");
        let schedule = u32_buffer(device, &vec![0xdead_beef; words]);
        let routes = u32_buffer(device, &vec![0xdead_beef; rows]);
        let grids = u32_buffer(device, &[0xdead_beef; MOE_TILES_FUSED_GRID_WORDS]);
        let input = f32_buffer(device, x, vec![tokens, HIDDEN]);
        let table = u32_buffer(device, ids);
        let weights = f32_buffer(device, routing_weights, vec![ids.len()]);
        let output = unwritten(
            device,
            moe_tiles_fused_output_rows(&params).expect("output rows"),
        );
        let verify_rows = mask.map(|mask| u32_buffer(device, mask.verify_rows));
        let gpu_mask = mask.map(|mask| MoeTilesFusedMask {
            verify_rows: verify_rows.as_ref().expect("verify rows"),
            physical_lane: mask.physical_lane,
            lanes: mask.lanes,
            rows_per_lane: LANE_ROWS as u32,
        });
        let mut encoder = device.command_encoder().expect("encoder");
        moe_tiles_fused(
            &mut encoder,
            &mut self.registry,
            device,
            &input,
            &table,
            &weights,
            gpu_mask.as_ref(),
            &MoeTilesFusedWeights {
                gate: &self.routed[0],
                up: &self.routed[1],
                down: &self.routed[2],
                shared_gate: &self.shared[0],
                shared_up: &self.shared[1],
                shared_down: &self.shared[2],
            },
            &MoeTilesFusedBuffers {
                schedule: &schedule,
                routes: &routes,
                grids: &grids,
                gathered: &gathered,
                intermediate: &intermediate,
            },
            &output,
            &params,
            MoeTilesFusedStage::Down,
        )
        .expect("fused step");
        encoder.commit_and_wait().expect("commit fused step");
        let bits = output.as_slice::<u32>().expect("output bits");
        let routed_words = ids.len() * HIDDEN;
        Step {
            routed: bits[..routed_words].to_vec(),
            shared: bits[routed_words..].to_vec(),
            ids: table.as_slice::<u32>().expect("ids").to_vec(),
            weights: weights.as_slice::<u32>().expect("weights").to_vec(),
            tiles: schedule.as_slice::<u32>().expect("schedule")[0],
            grids: grids.as_slice::<u32>().expect("grid words").to_vec(),
            schedule: schedule.as_slice::<u32>().expect("schedule").to_vec(),
            route_list: routes.as_slice::<u32>().expect("route list").to_vec(),
        }
    }
}

fn activations(tokens: usize, rng: &mut Rng) -> Vec<f32> {
    (0..tokens * HIDDEN).map(|_| rng.activation()).collect()
}

fn weights(routes: usize, rng: &mut Rng) -> Vec<f32> {
    (0..routes)
        .map(|_| rng.activation().abs() + 0.125)
        .collect()
}

/// `tokens` rows of distinct random experts.
fn random_routes(tokens: usize, rng: &mut Rng) -> Vec<u32> {
    let mut ids = Vec::with_capacity(tokens * TOP_K);
    for _ in 0..tokens {
        let mut row: Vec<u32> = Vec::with_capacity(TOP_K);
        while row.len() < TOP_K {
            let expert = rng.below(EXPERTS) as u32;
            if !row.contains(&expert) {
                row.push(expert);
            }
        }
        ids.extend_from_slice(&row);
    }
    ids
}

/// Routes whose experts hold 1, 2, 3..8, 9..16 and 17..32 routes.
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

/// Tiles of a route table: an expert of `c` live routes takes `ceil(c / 8)`,
/// the shared expert one per block of eight rows.
fn expected_tiles(ids: &[u32]) -> u32 {
    let mut counts = [0u32; EXPERTS];
    ids.iter()
        .filter(|id| **id != MASK)
        .for_each(|id| counts[*id as usize] += 1);
    counts.iter().map(|count| count.div_ceil(8)).sum::<u32>()
        + (ids.len() / TOP_K).div_ceil(LANE_ROWS) as u32
}

fn assert_written(step: &Step, label: &str) {
    for (name, bits) in [("routed", &step.routed), ("shared", &step.shared)] {
        assert!(
            bits.iter()
                .all(|bits| *bits != UNWRITTEN && f32::from_bits(*bits).is_finite()),
            "{label}: a {name} row was not written or is not finite"
        );
    }
}

fn equals_live_tiles(gate_up_type: GgmlType, down_type: GgmlType, seed: u64) {
    let mut layer = Layer::new(gate_up_type, down_type, down_type, seed);
    let mut rng = Rng(seed ^ 0xa5a5);
    let probed = layer.column_tile;
    let mut cases: Vec<(String, Vec<f32>, Vec<u32>)> = vec![(
        "spread".into(),
        activations(TOKENS, &mut rng),
        spread_routes(&mut rng),
    )];
    for tokens in [1, 2, 3, 7, 8, 9, 16, 17, 24, 31, 32] {
        cases.push((
            format!("random {tokens}"),
            activations(tokens, &mut rng),
            random_routes(tokens, &mut rng),
        ));
    }
    let mut all = Vec::new();
    for (name, x, ids) in &cases {
        let routing_weights = weights(ids.len(), &mut rng);
        for column_tile in [probed, false] {
            let label = format!("{gate_up_type:?}/{down_type:?} {name} register {column_tile}");
            let live = layer.live(x, ids, column_tile);
            let fused = layer.fused(x, ids, &routing_weights, None, column_tile);
            assert_written(&live, &label);
            assert!(fused.routed == live.routed, "{label}: a routed row differs");
            assert!(fused.shared == live.shared, "{label}: a shared row differs");
            assert!(
                fused.ids == *ids
                    && fused.weights
                        == routing_weights
                            .iter()
                            .map(|weight| weight.to_bits())
                            .collect::<Vec<_>>(),
                "{label}: an unmasked step rewrote its route table"
            );
            let tiles = expected_tiles(ids);
            assert_eq!(fused.tiles, tiles, "{label}: tile count");
            assert_eq!(
                fused.grids,
                [
                    (HIDDEN / 256) as u32,
                    tiles,
                    1,
                    (INTERMEDIATE / 32) as u32,
                    tiles,
                    1,
                    (HIDDEN / 64) as u32,
                    tiles,
                    1
                ],
                "{label}: indirect grids"
            );
            all.extend_from_slice(&fused.routed);
            all.extend_from_slice(&fused.shared);
        }
    }
    eprintln!(
        "digest fused_equals_live {gate_up_type:?}/{down_type:?}: {:016x}",
        digest(&all)
    );
}

#[test]
fn q4_k_fused_route_is_the_live_tile_route_bitwise() {
    equals_live_tiles(GgmlType::Q4_K, GgmlType::Q4_K, 0x91);
}

#[test]
fn q6_k_down_fused_route_is_the_live_tile_route_bitwise() {
    equals_live_tiles(GgmlType::Q4_K, GgmlType::Q6_K, 0x92);
}

#[test]
fn q6_k_gate_up_fused_route_is_the_live_tile_route_bitwise() {
    equals_live_tiles(GgmlType::Q6_K, GgmlType::Q4_K, 0x93);
}

const SHARED_TILE: u32 = 0x8000_0000;
const COLUMN_TILE: u32 = 0x4000_0000;

/// The schedule words and route list `moe_tiles_fused_schedule` documents,
/// for a route table whose masked routes hold the id `experts`.
fn reference_schedule(
    masked_ids: &[u32],
    experts: usize,
    column_tile: bool,
) -> (Vec<u32>, Vec<u32>) {
    let tokens = masked_ids.len() / TOP_K;
    let column_rows = if column_tile { 2 } else { 0 };
    let mut by_expert: Vec<Vec<u32>> = vec![Vec::new(); experts];
    for (route, id) in masked_ids.iter().enumerate() {
        if (*id as usize) < experts {
            by_expert[*id as usize].push(route as u32);
        }
    }
    let mut tiles = Vec::new();
    let mut routes = Vec::new();
    for (expert, list) in by_expert.iter().enumerate() {
        // The register path is an expert's, not a tile's: nine routes are
        // two matmul tiles.
        let flag = if list.len() <= column_rows {
            COLUMN_TILE
        } else {
            0
        };
        for chunk in list.chunks(8) {
            tiles.extend_from_slice(&[
                expert as u32 | flag,
                routes.len() as u32,
                chunk.len() as u32,
            ]);
            routes.extend_from_slice(chunk);
        }
    }
    let routed_rows = routes.len();
    for block in 0..tokens.div_ceil(LANE_ROWS) {
        let rows = LANE_ROWS.min(tokens - block * LANE_ROWS);
        let flag = if rows <= column_rows { COLUMN_TILE } else { 0 };
        tiles.extend_from_slice(&[
            SHARED_TILE | flag,
            (routed_rows + block * LANE_ROWS) as u32,
            rows as u32,
        ]);
    }
    routes.extend((0..tokens).map(|token| (masked_ids.len() + token) as u32));
    let mut schedule = vec![(tiles.len() / 3) as u32];
    schedule.extend(tiles);
    (schedule, routes)
}

/// A 32-row table over `WIDE_EXPERTS` experts in which expert `e` of `plan`
/// holds exactly the rows of its range, the experts of `unused` hold none
/// and every other slot holds an expert of one route, drawn from all of the
/// schedule's simdgroups (a stride of 37 walks every expert once).
fn edge_table(plan: &[(u32, std::ops::Range<usize>)], unused: &[u32], seed: u64) -> Vec<u32> {
    let mut rng = Rng(seed);
    let mut filler = (0..WIDE_EXPERTS as u32)
        .map(|index| index * 37 % WIDE_EXPERTS as u32)
        .filter(|expert| {
            !unused.contains(expert) && plan.iter().all(|(planned, _)| planned != expert)
        });
    let mut ids = Vec::with_capacity(TOKENS * TOP_K);
    for token in 0..TOKENS {
        let mut row: Vec<u32> = plan
            .iter()
            .filter(|(_, range)| range.contains(&token))
            .map(|(expert, _)| *expert)
            .collect();
        while row.len() < TOP_K {
            row.push(filler.next().expect("filler expert"));
        }
        assert_eq!(
            row.len(),
            TOP_K,
            "row {token} holds too many planned experts"
        );
        for index in (1..TOP_K).rev() {
            row.swap(index, rng.below(index + 1));
        }
        ids.extend_from_slice(&row);
    }
    ids
}

/// One fused step of the 256-expert layer against the CPU reference of its
/// schedule and against the live-tile stages; returns the compared bits.
fn check_wide_step(
    layer: &mut Layer,
    label: &str,
    x: &[f32],
    ids: &[u32],
    routing_weights: &[f32],
    mask: Option<Mask<'_>>,
    column_tile: bool,
) -> Vec<u32> {
    let tokens = ids.len() / TOP_K;
    let masked_id = layer.experts as u32;
    let live = |token: usize| mask.map_or(true, |mask| mask.live(token));
    let masked_ids: Vec<u32> = ids
        .iter()
        .enumerate()
        .map(|(route, id)| if live(route / TOP_K) { *id } else { masked_id })
        .collect();
    // Masked rows carry NaN inputs: no live route may read them.
    let mut masked_x = x.to_vec();
    for token in (0..tokens).filter(|token| !live(*token)) {
        masked_x[token * HIDDEN..(token + 1) * HIDDEN].fill(f32::NAN);
    }
    let fused = layer.fused(&masked_x, ids, routing_weights, mask, column_tile);
    assert_eq!(fused.ids, masked_ids, "{label}: route table");

    let (schedule, route_list) = reference_schedule(&masked_ids, layer.experts, column_tile);
    assert_eq!(
        fused.schedule[..schedule.len()],
        schedule[..],
        "{label}: schedule words"
    );
    assert_eq!(
        fused.route_list[..route_list.len()],
        route_list[..],
        "{label}: route list"
    );
    let tiles = schedule[0];
    assert_eq!(
        fused.grids,
        [
            (HIDDEN / 256) as u32,
            tiles,
            1,
            (INTERMEDIATE / 32) as u32,
            tiles,
            1,
            (HIDDEN / 64) as u32,
            tiles,
            1
        ],
        "{label}: indirect grids"
    );

    let stages = layer.live(&masked_x, &masked_ids, column_tile);
    let mut compared = Vec::new();
    for route in 0..ids.len() {
        if masked_ids[route] == masked_id {
            assert!(
                fused.route(route).iter().all(|bits| *bits == UNWRITTEN),
                "{label}: masked route {route} was written"
            );
        } else {
            assert!(
                fused
                    .route(route)
                    .iter()
                    .all(|bits| *bits != UNWRITTEN && f32::from_bits(*bits).is_finite()),
                "{label}: route {route} was not written or is not finite"
            );
            assert!(
                fused.route(route) == stages.route(route),
                "{label}: route {route} (expert {}) differs from the live-tile stages",
                masked_ids[route]
            );
            compared.extend_from_slice(fused.route(route));
        }
    }
    for token in (0..tokens).filter(|token| live(*token)) {
        assert!(
            fused.shared_row(token) == stages.shared_row(token),
            "{label}: the shared expert's row {token} differs from the live-tile stages"
        );
        compared.extend_from_slice(fused.shared_row(token));
    }
    compared
}

/// Experts in every simdgroup of the schedule's threadgroup, the last index
/// included: the first row and first tile of an expert sum over the
/// simdgroups before its own.
#[test]
fn schedule_of_256_experts_matches_the_cpu_reference() {
    let mut layer = Layer::with_experts(
        GgmlType::Q4_K,
        GgmlType::Q6_K,
        GgmlType::Q6_K,
        0xe0,
        WIDE_EXPERTS,
    );
    let probed = layer.column_tile;
    let first = edge_table(
        &[
            (0, 0..32),
            (255, 0..17),
            (1, 0..16),
            (254, 0..9),
            (2, 0..8),
            (3, 30..32),
            (4, 31..32),
        ],
        &[5, 128, 253],
        0xe1,
    );
    let second = edge_table(
        &[
            (255, 0..32),
            (254, 5..6),
            (0, 3..5),
            (253, 0..9),
            (127, 8..24),
            (128, 15..32),
            (64, 0..8),
            (31, 2..12),
            (32, 20..31),
        ],
        &[1, 2, 252],
        0xe2,
    );
    for table in [&first, &second] {
        let mut counts = vec![0usize; WIDE_EXPERTS];
        table.iter().for_each(|id| counts[*id as usize] += 1);
        for wanted in [0usize, 1, 2, 8, 9, 16, 17, 32] {
            assert!(counts.contains(&wanted), "no expert of {wanted} routes");
        }
        // Live experts in each of the eight simdgroups.
        for group in 0..WIDE_EXPERTS / 32 {
            assert!(
                counts[group * 32..(group + 1) * 32]
                    .iter()
                    .any(|count| *count > 0),
                "no live expert in simdgroup {group}"
            );
        }
    }
    assert_eq!(first.iter().filter(|id| **id == 255).count(), 17);
    assert_eq!(second.iter().filter(|id| **id == 255).count(), 32);
    let mut rng = Rng(0xe3);
    let x = activations(TOKENS, &mut rng);
    let routing_weights = weights(TOKENS * TOP_K, &mut rng);
    let mut all = Vec::new();
    let mut steps = 0;

    // Unmasked: every prefix width, with and without the register path.
    for (name, table) in [("first", &first), ("second", &second)] {
        for tokens in [1usize, 2, 3, 7, 8, 9, 10, 16, 17, 24, 31, 32] {
            for column_tile in [probed, false] {
                all.extend(check_wide_step(
                    &mut layer,
                    &format!("{name} table, {tokens} rows, register {column_tile}"),
                    &x[..tokens * HIDDEN],
                    &table[..tokens * TOP_K],
                    &routing_weights[..tokens * TOP_K],
                    None,
                    column_tile,
                ));
                steps += 1;
            }
        }
    }

    // Masked: (rows, lanes, verify rows per physical lane, lane order).
    let cases: [(usize, u32, [u32; 4], [u32; 4]); 10] = [
        (32, 4, [0, 0, 0, 0], [0, 1, 2, 3]),
        (32, 4, [0, 0, 1, 0], [0, 1, 2, 3]),
        (32, 4, [0, 0, 0, 1], [3, 2, 1, 0]),
        (32, 1, [3, 8, 8, 8], [0, 0, 0, 0]),
        (32, 2, [8, 5, 8, 8], [3, 1, 0, 2]),
        (32, 4, [8, 8, 8, 8], [0, 1, 2, 3]),
        (32, 4, [100, 9, 8, 7], [0, 1, 2, 3]),
        (32, 4, [5, 8, 1, 3], [0, 1, 2, 3]),
        (16, 2, [3, 8, 0, 0], [0, 1, 0, 0]),
        (24, 3, [1, 0, 7, 0], [0, 1, 2, 0]),
    ];
    for (name, table) in [("first", &first), ("second", &second)] {
        for (tokens, lanes, verify_rows, physical_lane) in cases {
            all.extend(check_wide_step(
                &mut layer,
                &format!(
                    "{name} table, {tokens} rows, lanes {lanes} {verify_rows:?} {physical_lane:?}"
                ),
                &x[..tokens * HIDDEN],
                &table[..tokens * TOP_K],
                &routing_weights[..tokens * TOP_K],
                Some(Mask {
                    verify_rows: &verify_rows,
                    physical_lane,
                    lanes,
                }),
                probed,
            ));
            steps += 1;
        }
    }
    eprintln!(
        "digest fused_schedule_256 ({steps} steps): {:016x}",
        digest(&all)
    );
}

/// What `dflash2_mask_moe_routes` leaves in the route table.
fn mask_kernel(
    layer: &mut Layer,
    ids: &[u32],
    routing_weights: &[f32],
    mask: Mask<'_>,
) -> (Vec<u32>, Vec<u32>) {
    let device = &layer.device;
    let table = u32_buffer(device, ids);
    let weights = f32_buffer(device, routing_weights, vec![ids.len()]);
    let verify_rows = u32_buffer(device, mask.verify_rows);
    let mut encoder = device.command_encoder().expect("encoder");
    dispatch_dflash2_mask_moe_routes(
        &mut encoder,
        &mut layer.registry,
        device,
        &verify_rows,
        &table,
        &weights,
        DFlash2MoeRouteMaskParams {
            physical_lane: mask.physical_lane,
            lanes: mask.lanes,
            rows_per_lane: LANE_ROWS as u32,
            top_k: TOP_K as u32,
            n_experts: EXPERTS as u32,
            width: (ids.len() / TOP_K) as u32,
        },
    )
    .expect("route mask");
    encoder.commit_and_wait().expect("commit mask");
    (
        table.as_slice::<u32>().expect("ids").to_vec(),
        weights.as_slice::<u32>().expect("weights").to_vec(),
    )
}

#[test]
fn fused_schedule_applies_the_verify_row_mask() {
    let mut layer = Layer::new(GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q6_K, 0xa1);
    let mut rng = Rng(0xa2);
    let column_tile = layer.column_tile;
    let mut all = Vec::new();
    // (rows of the step, lanes, verify rows per physical lane, lane order).
    let cases: [(usize, u32, [u32; 4], [u32; 4]); 5] = [
        (32, 4, [5, 8, 1, 3], [0, 1, 2, 3]),
        (32, 4, [8, 2, 7, 4], [2, 0, 3, 1]),
        (24, 3, [6, 9, 2, 0], [3, 1, 0, 0]),
        // The last two blocks belong to no lane.
        (32, 2, [4, 8, 8, 8], [1, 0, 0, 0]),
        (32, 4, [0, 0, 0, 0], [0, 1, 2, 3]),
    ];
    for (tokens, lanes, verify_rows, physical_lane) in cases {
        let label = format!("{tokens} rows, lanes {lanes} {verify_rows:?} {physical_lane:?}");
        let mask = Mask {
            verify_rows: &verify_rows,
            physical_lane,
            lanes,
        };
        let x = activations(tokens, &mut rng);
        let ids = if tokens == TOKENS {
            spread_routes(&mut rng)
        } else {
            random_routes(tokens, &mut rng)
        };
        let routing_weights = weights(ids.len(), &mut rng);
        let full = layer.live(&x, &ids, column_tile);

        // Masked rows carry NaN inputs: no live route may read them.
        let mut masked_x = x.clone();
        for token in (0..tokens).filter(|token| !mask.live(*token)) {
            masked_x[token * HIDDEN..(token + 1) * HIDDEN].fill(f32::NAN);
        }
        let (masked_ids, masked_weights) = mask_kernel(&mut layer, &ids, &routing_weights, mask);
        let fused = layer.fused(&masked_x, &ids, &routing_weights, Some(mask), column_tile);
        assert!(
            fused.ids == masked_ids && fused.weights == masked_weights,
            "{label}: the route table is not what the mask kernel writes"
        );
        let live = layer.live(&masked_x, &masked_ids, column_tile);
        for route in 0..ids.len() {
            if mask.live(route / TOP_K) {
                assert!(
                    fused.route(route) == full.route(route)
                        && fused.route(route) == live.route(route),
                    "{label}: live route {route} changed under the mask"
                );
            } else {
                assert_eq!(masked_ids[route], MASK, "{label}: route {route} not masked");
                assert!(
                    fused.route(route).iter().all(|bits| *bits == UNWRITTEN),
                    "{label}: masked route {route} was written"
                );
            }
        }
        for token in (0..tokens).filter(|token| mask.live(*token)) {
            assert!(
                fused.shared_row(token) == full.shared_row(token),
                "{label}: the shared expert's row {token} changed under the mask"
            );
        }
        assert_eq!(fused.tiles, expected_tiles(&masked_ids), "{label}: tiles");
        assert_eq!(fused.grids[4], fused.tiles, "{label}: gate/up grid");
        for route in (0..ids.len()).filter(|route| mask.live(route / TOP_K)) {
            all.extend_from_slice(fused.route(route));
        }
    }
    eprintln!("digest fused_mask: {:016x}", digest(&all));
}

#[test]
fn fused_route_is_deterministic_and_row_independent() {
    let mut layer = Layer::new(GgmlType::Q6_K, GgmlType::Q6_K, GgmlType::Q6_K, 0xb1);
    let mut rng = Rng(0xb2);
    let column_tile = layer.column_tile;
    let x = activations(TOKENS, &mut rng);
    let ids = spread_routes(&mut rng);
    let routing_weights = weights(ids.len(), &mut rng);
    let full = layer.fused(&x, &ids, &routing_weights, None, column_tile);
    assert_written(&full, "full step");
    let again = layer.fused(&x, &ids, &routing_weights, None, column_tile);
    assert!(
        again.routed == full.routed && again.shared == full.shared,
        "two runs differ"
    );
    let matmul_only = layer.fused(&x, &ids, &routing_weights, None, false);
    assert!(
        matmul_only.routed == full.routed && matmul_only.shared == full.shared,
        "the register path changed a result"
    );

    // Each row as a step of its own: its experts hold one route each and
    // the shared expert one row.
    for token in 0..TOKENS {
        let alone = layer.fused(
            &x[token * HIDDEN..(token + 1) * HIDDEN],
            &ids[token * TOP_K..(token + 1) * TOP_K],
            &routing_weights[token * TOP_K..(token + 1) * TOP_K],
            None,
            column_tile,
        );
        assert!(
            alone.routed == full.routed[token * TOP_K * HIDDEN..(token + 1) * TOP_K * HIDDEN]
                && alone.shared == full.shared_row(token),
            "row {token} depends on the other rows of its step"
        );
    }

    // The rows in another order: other lanes, other places in their tiles.
    let order: Vec<usize> = (0..TOKENS).rev().collect();
    let permuted_x: Vec<f32> = order
        .iter()
        .flat_map(|&token| x[token * HIDDEN..(token + 1) * HIDDEN].to_vec())
        .collect();
    let permuted_ids: Vec<u32> = order
        .iter()
        .flat_map(|&token| ids[token * TOP_K..(token + 1) * TOP_K].to_vec())
        .collect();
    let permuted = layer.fused(
        &permuted_x,
        &permuted_ids,
        &routing_weights,
        None,
        column_tile,
    );
    for (place, &token) in order.iter().enumerate() {
        for slot in 0..TOP_K {
            assert!(
                permuted.route(place * TOP_K + slot) == full.route(token * TOP_K + slot),
                "row {token} slot {slot} depends on its place in the step"
            );
        }
        assert!(
            permuted.shared_row(place) == full.shared_row(token),
            "the shared expert's row {token} depends on its place in the step"
        );
    }
    eprintln!(
        "digest fused_independent: {:016x}",
        digest(&[full.routed.as_slice(), full.shared.as_slice()].concat())
    );
}

#[test]
fn fused_route_refuses_what_it_cannot_run() {
    let mut layer = Layer::new(GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K, 0xc1);
    // The same shapes with the shared expert's down in another format.
    let other = Layer::new(GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q6_K, 0xc1);
    let mut rng = Rng(0xc2);
    let device = &layer.device;
    let params = layer.fused_params(TOKENS, false);
    let rows = moe_tiles_fused_grouped_rows(&params).expect("rows");
    let words = moe_tiles_fused_schedule_words(&params).expect("words");
    let x = activations(TOKENS, &mut rng);
    let ids = random_routes(TOKENS, &mut rng);
    let input = f32_buffer(device, &x, vec![TOKENS, HIDDEN]);
    let table = u32_buffer(device, &ids);
    let routing_weights = f32_buffer(device, &weights(ids.len(), &mut rng), vec![ids.len()]);
    let schedule = u32_buffer(device, &vec![0; words]);
    let routes = u32_buffer(device, &vec![0; rows]);
    let grids = u32_buffer(device, &[0; MOE_TILES_FUSED_GRID_WORDS]);
    let gathered = nan_rows(device, rows, HIDDEN);
    let intermediate = nan_rows(device, rows, INTERMEDIATE);
    let output = unwritten(
        device,
        moe_tiles_fused_output_rows(&params).expect("output rows"),
    );
    let buffers = MoeTilesFusedBuffers {
        schedule: &schedule,
        routes: &routes,
        grids: &grids,
        gathered: &gathered,
        intermediate: &intermediate,
    };
    let mut encoder = device.command_encoder().expect("encoder");
    let mut run = |weights: &MoeTilesFusedWeights<'_>,
                   buffers: &MoeTilesFusedBuffers<'_>,
                   mask: Option<&MoeTilesFusedMask<'_>>,
                   params: &MoeTilesFusedParams| {
        moe_tiles_fused(
            &mut encoder,
            &mut layer.registry,
            device,
            &input,
            &table,
            &routing_weights,
            mask,
            weights,
            buffers,
            &output,
            params,
            MoeTilesFusedStage::Down,
        )
    };
    let (routed, shared) = (&layer.routed, &layer.shared);
    let valid = MoeTilesFusedWeights {
        gate: &routed[0],
        up: &routed[1],
        down: &routed[2],
        shared_gate: &shared[0],
        shared_up: &shared[1],
        shared_down: &shared[2],
    };
    assert!(
        run(&valid, &buffers, None, &params).is_ok(),
        "the valid step was refused"
    );
    let mixed = MoeTilesFusedWeights {
        shared_down: &other.shared[2],
        ..valid
    };
    assert!(
        run(&mixed, &buffers, None, &params).is_err(),
        "routed and shared down in two formats were accepted"
    );
    // Gate planes stand in for the down stacks: one format, the wrong shape.
    let misshaped = MoeTilesFusedWeights {
        down: &routed[0],
        shared_down: &shared[0],
        ..valid
    };
    assert!(
        run(&misshaped, &buffers, None, &params).is_err(),
        "down stacks of gate shape were accepted"
    );
    // A mask of more rows than the step holds.
    let verify_rows = u32_buffer(device, &[8; 4]);
    let wide_mask = MoeTilesFusedMask {
        verify_rows: &verify_rows,
        physical_lane: [0, 1, 2, 3],
        lanes: 4,
        rows_per_lane: 16,
    };
    assert!(
        run(&valid, &buffers, Some(&wide_mask), &params).is_err(),
        "a mask past the step's rows was accepted"
    );
    // A lane outside the verify rows.
    let stray_mask = MoeTilesFusedMask {
        physical_lane: [0, 1, 2, 4],
        rows_per_lane: 8,
        ..wide_mask
    };
    assert!(
        run(&valid, &buffers, Some(&stray_mask), &params).is_err(),
        "a physical lane outside the verify rows was accepted"
    );
    // The schedule is one word short.
    let short = u32_buffer(device, &vec![0; words - 1]);
    assert!(
        run(
            &valid,
            &MoeTilesFusedBuffers {
                schedule: &short,
                ..buffers
            },
            None,
            &params
        )
        .is_err(),
        "an undersized schedule was accepted"
    );
    // 33 rows of eight routes exceed the schedule's threadgroup.
    let wide = MoeTilesFusedParams {
        n_tokens: 33,
        ..params
    };
    assert!(moe_tiles_fused_grouped_rows(&wide).is_err());
    assert!(moe_tiles_fused_schedule_words(&wide).is_err());
    let odd = MoeTilesFusedParams {
        hidden: HIDDEN as u32 + 64,
        ..params
    };
    assert!(moe_tiles_fused_output_rows(&odd).is_err());
}

#[test]
fn fused_route_poisons_an_invalid_route_table() {
    let mut layer = Layer::new(GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q4_K, 0xd1);
    let mut rng = Rng(0xd2);
    let x = activations(TOKENS, &mut rng);
    let mut ids = random_routes(TOKENS, &mut rng);
    ids[77] = EXPERTS as u32 + 3;
    let routing_weights = weights(ids.len(), &mut rng);
    let step = layer.fused(&x, &ids, &routing_weights, None, false);
    assert_eq!(
        step.tiles, POISON,
        "an out-of-range id did not poison the schedule"
    );
    assert_eq!(&step.grids[..3], &[(HIDDEN / 256) as u32, 1, 1]);
    assert!(
        step.routed
            .iter()
            .chain(&step.shared)
            .all(|bits| *bits == UNWRITTEN),
        "a poisoned step wrote output"
    );
    // An out-of-range id in a masked row is rewritten by the mask, not read.
    let verify_rows = [8u32, 1, 8, 8];
    let mask = Mask {
        verify_rows: &verify_rows,
        physical_lane: [0, 1, 2, 3],
        lanes: 4,
    };
    assert!(!mask.live(77 / TOP_K));
    let step = layer.fused(&x, &ids, &routing_weights, Some(mask), false);
    let masked: Vec<u32> = ids
        .iter()
        .enumerate()
        .map(|(route, id)| if mask.live(route / TOP_K) { *id } else { MASK })
        .collect();
    assert_eq!(step.ids, masked, "the masked table");
    assert_eq!(step.tiles, expected_tiles(&masked));
}
