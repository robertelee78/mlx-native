//! Model-free GPU time of one Qwen3.6-35B-A3B MoE FFN layer at the served
//! shape (hidden 2048, 256 experts, top 8, expert intermediate 512, shared
//! expert 512, F32 router), as the width-32 DFlash2 verifier dispatches it.
//!
//! ```text
//! cargo +1.96.0 bench --locked --bench bench_qwen36_moe_layer -- \
//!     [routes=<dump>|pooled|random] [rows=32] [down=q4k|q6k] [rounds=20] [layers=16] \
//!     [weights=1] [route=all|staged|live|fused] [stages=1|0] [idle=<ms>]
//! ```
//!
//! `route` selects the routes to time (the staged-M8 route hf2q serves by
//! default, the live-tile route, the fused-tile route); `stages=0` skips the
//! per-stage commands.
//!
//! `routes=<dump>` replays expert ids recorded from real verify steps: a
//! sequence of little-endian records `layer u32, rows u32, top_k u32,
//! ids[rows * top_k] u32`, where every slot of a row the verify-row mask
//! removed holds the id 256. `pooled` draws each lane's eight rows from 24
//! experts and `random` draws every row from all 256; both keep every row
//! live.
//!
//! `weights=<n>` allocates `n` sets of expert weights (about 0.47 GiB each);
//! layer `i` of a command reads set `i % n`. A served cycle reads each
//! layer's own weights once, so `weights=<layers>` is the served memory
//! pattern, and `weights=1` re-reads one set `layers` times per command.
//!
//! Three measurements, all GPU time (`GPUEndTime - GPUStartTime`):
//!
//! * per stage: one command buffer that runs the stage for each of the
//!   `layers` sampled layers, a barrier between them, median of `rounds`
//!   commands divided by `layers` (a stage that reuses the route schedule of
//!   an earlier stage runs after that stage and reports the difference; a
//!   command of one small dispatch alone measures the GPU's wake-up);
//! * per layer: `layers` layers of different routes in one command buffer,
//!   median of `rounds` commands;
//! * per layer, every layer its own command that the host waits for (how a
//!   stage profile commits a layer), median of `rounds` passes.

use mlx_native::ops::dense_matvec_f32_rows::{
    dispatch_dense_matvec_f32_row_blocks, dispatch_dense_matvec_f32_rows,
};
use mlx_native::ops::dflash2_attention_pack::{
    dispatch_dflash2_mask_moe_routes, DFlash2MoeRouteMaskParams,
};
use mlx_native::ops::moe_softmax_topk::dispatch_moe_softmax_topk_simd;
use mlx_native::ops::moe_weighted_reduce::dispatch_moe_weighted_reduce_masked;
use mlx_native::ops::quantized_matmul_ggml::{
    quantized_matmul_ggml, GgmlQuantizedMatmulParams, GgmlType,
};
use mlx_native::ops::silu_mul::dispatch_silu_mul;
use mlx_native::{
    dense_matmul_f32_f32_tensor, ggml_expert_bytes, moe_tiles_column_tile_exact, moe_tiles_down,
    moe_tiles_fused, moe_tiles_fused_grouped_rows, moe_tiles_fused_output_rows,
    moe_tiles_fused_schedule_words, moe_tiles_gate_up, moe_tiles_gather, moe_tiles_grouped_rows,
    moe_tiles_schedule, prepare_quantized_matmul_id_ggml_grouped_m8,
    quantized_matmul_id_ggml_grouped_m8_epilogue_masked, CommandEncoder, DType,
    DenseMmF32F32Params, GgmlQuantizedMatmulIdParams, GroupedM8Epilogue, GroupedM8Input,
    IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice, MoeTilesBuffers, MoeTilesFusedBuffers,
    MoeTilesFusedMask, MoeTilesFusedParams, MoeTilesFusedStage, MoeTilesFusedWeights,
    MoeTilesParams, PreparedGroupedM8Experts, MOE_TILES_FUSED_GRID_WORDS, MOE_TILES_GRID_WORDS,
};

const H: u32 = 2048;
const EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const INTERMEDIATE: u32 = 512;
const SHARED: u32 = 512;
const LANE_ROWS: u32 = 8;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            values.swap(index, self.below(index + 1));
        }
    }
}

fn f32_buffer(device: &MlxDevice, len: usize, rng: &mut Rng, scale: f32) -> MlxBuffer {
    let mut buffer = device
        .alloc_buffer(len * 4, DType::F32, vec![len])
        .expect("f32 buffer");
    for value in buffer.as_mut_slice::<f32>().expect("f32 slice") {
        *value = rng.unit() * scale;
    }
    buffer
}

fn zeros(device: &MlxDevice, len: usize, dtype: DType) -> MlxBuffer {
    device
        .alloc_buffer(len * dtype.size_of(), dtype, vec![len])
        .expect("scratch buffer")
}

fn u32_buffer(device: &MlxDevice, values: &[u32]) -> MlxBuffer {
    let mut buffer = zeros(device, values.len(), DType::U32);
    buffer
        .as_mut_slice::<u32>()
        .expect("u32 slice")
        .copy_from_slice(values);
    buffer
}

/// Random GGML blocks with model-magnitude block scales.
fn quantized(
    device: &MlxDevice,
    ggml_type: GgmlType,
    experts: u32,
    n: u32,
    k: u32,
    rng: &mut Rng,
) -> (MlxBuffer, u64) {
    let stride =
        u64::from(n) * u64::from(k / ggml_type.block_values()) * u64::from(ggml_type.block_bytes());
    let bytes = ggml_expert_bytes(ggml_type, experts, n, k, stride).expect("weight bytes") as usize;
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
        _ => unreachable!("bench formats"),
    };
    for block in data.chunks_exact_mut(ggml_type.block_bytes() as usize) {
        for (index, offset) in scale_offsets.iter().copied().enumerate() {
            let sign = if index == 0 && rng.next() & 1 != 0 {
                0x8000
            } else {
                0
            };
            // half in [2^-8, 2^-7): weights of a few hundredths.
            let bits = sign | (7 << 10) | ((rng.next() >> 8) as u16 & 0x03ff);
            block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    (buffer, stride)
}

/// One layer's routes: `rows * TOP_K` ids, 256 in every slot of a masked row.
#[derive(Clone)]
struct Routes {
    rows: u32,
    ids: Vec<u32>,
}

impl Routes {
    fn live_rows(&self) -> usize {
        self.ids
            .chunks(TOP_K as usize)
            .filter(|row| row[0] != EXPERTS)
            .count()
    }

    /// (live experts, experts of one or two routes, eight-row tiles).
    fn tiles(&self) -> (usize, usize, usize) {
        let mut counts = [0usize; EXPERTS as usize];
        for &id in self.ids.iter().filter(|&&id| id < EXPERTS) {
            counts[id as usize] += 1;
        }
        let live = counts.iter().filter(|&&count| count > 0).count();
        let column = counts
            .iter()
            .filter(|&&count| (1..=2).contains(&count))
            .count();
        let tiles = counts.iter().map(|count| count.div_ceil(8)).sum();
        (live, column, tiles)
    }
}

fn load_routes(path: &str, rows: u32) -> Vec<Routes> {
    let bytes = std::fs::read(path).expect("route dump");
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("word"));
    let mut routes = Vec::new();
    let mut at = 0;
    while at + 12 <= bytes.len() {
        let (record_rows, top_k) = (word(at + 4), word(at + 8));
        let count = (record_rows * top_k) as usize;
        if at + 12 + count * 4 > bytes.len() {
            break;
        }
        if record_rows == rows && top_k == TOP_K {
            routes.push(Routes {
                rows,
                ids: (0..count).map(|index| word(at + 12 + index * 4)).collect(),
            });
        }
        at += 12 + count * 4;
    }
    routes
}

fn synthetic_routes(rows: u32, pooled: bool, count: usize, rng: &mut Rng) -> Vec<Routes> {
    (0..count)
        .map(|_| {
            let mut experts: Vec<u32> = (0..EXPERTS).collect();
            let mut ids = Vec::with_capacity((rows * TOP_K) as usize);
            for row in 0..rows {
                if !pooled || row % LANE_ROWS == 0 {
                    rng.shuffle(&mut experts);
                }
                let pool = if pooled {
                    3 * TOP_K as usize
                } else {
                    EXPERTS as usize
                };
                let mut candidates = experts[..pool].to_vec();
                rng.shuffle(&mut candidates);
                ids.extend_from_slice(&candidates[..TOP_K as usize]);
            }
            Routes { rows, ids }
        })
        .collect()
}

struct Weights {
    router: MlxBuffer,
    shared_logit: MlxBuffer,
    shared_gate: MlxBuffer,
    shared_up: MlxBuffer,
    shared_down: MlxBuffer,
    shared_down_type: GgmlType,
    gate: PreparedGroupedM8Experts,
    up: PreparedGroupedM8Experts,
    down: PreparedGroupedM8Experts,
    gate_up_params: GgmlQuantizedMatmulIdParams,
    down_params: GgmlQuantizedMatmulIdParams,
    column_tile: bool,
    /// The shared expert's gate, up and down as one-expert tile planes.
    shared_tiles: [PreparedGroupedM8Experts; 3],
}

/// One layer's activations, routes and scratch.
struct Layer {
    rows: u32,
    x: MlxBuffer,
    residual: MlxBuffer,
    logits: MlxBuffer,
    selected: MlxBuffer,
    selected_weights: MlxBuffer,
    verify_rows: MlxBuffer,
    ids: MlxBuffer,
    routing_weights: MlxBuffer,
    sh_logit: MlxBuffer,
    a_s: MlxBuffer,
    b_s: MlxBuffer,
    h_s: MlxBuffer,
    y_s: MlxBuffer,
    gate_all: MlxBuffer,
    up_all: MlxBuffer,
    h_all: MlxBuffer,
    y_all: MlxBuffer,
    gathered: MlxBuffer,
    silu_shared: MlxBuffer,
    silu_expert: MlxBuffer,
    out: MlxBuffer,
    scratch: IdMmScratch,
    tile_gathered: MlxBuffer,
    tile_intermediate: MlxBuffer,
    tile_grids: MlxBuffer,
    shared_ids: MlxBuffer,
    shared_scratch: IdMmScratch,
    shared_gathered: MlxBuffer,
    shared_intermediate: MlxBuffer,
    shared_grids: MlxBuffer,
    /// The fused route's inputs and scratch: the route table before the
    /// verify-row mask (a masked row holds experts 0..TOP_K), each lane's
    /// live rows, and one schedule for routed and shared tiles.
    fused_ids: MlxBuffer,
    fused_weights: MlxBuffer,
    fused_verify_rows: MlxBuffer,
    fused_schedule: MlxBuffer,
    fused_routes: MlxBuffer,
    fused_grids: MlxBuffer,
    fused_gathered: MlxBuffer,
    fused_intermediate: MlxBuffer,
    fused_out: MlxBuffer,
}

impl Layer {
    fn new(device: &MlxDevice, routes: &Routes, rng: &mut Rng) -> Self {
        let rows = routes.rows;
        let r = rows as usize;
        let pairs = r * TOP_K as usize;
        let (h, i, s) = (H as usize, INTERMEDIATE as usize, SHARED as usize);
        let grouped = moe_tiles_grouped_rows(rows, TOP_K, EXPERTS).expect("grouped rows");
        let shared_rows =
            moe_tiles_grouped_rows(rows, 1, rows.div_ceil(LANE_ROWS)).expect("shared rows");
        let routing_weights: Vec<f32> = routes
            .ids
            .iter()
            .map(|&id| {
                if id < EXPERTS {
                    1.0 / TOP_K as f32
                } else {
                    0.0
                }
            })
            .collect();
        let mut weights = zeros(device, pairs, DType::F32);
        weights
            .as_mut_slice::<f32>()
            .expect("routing weights")
            .copy_from_slice(&routing_weights);
        let fused = fused_params(rows, false);
        let fused_rows = moe_tiles_fused_grouped_rows(&fused).expect("fused rows");
        // A lane's live rows are a prefix of its block.
        let lane_rows: Vec<u32> = (0..4)
            .map(|lane| {
                (0..LANE_ROWS)
                    .filter(|row| {
                        let token = lane * LANE_ROWS + row;
                        token < rows && routes.ids[(token * TOP_K) as usize] < EXPERTS
                    })
                    .count() as u32
            })
            .collect();
        let unmasked: Vec<u32> = routes
            .ids
            .iter()
            .enumerate()
            .map(|(route, &id)| {
                if id < EXPERTS {
                    id
                } else {
                    route as u32 % TOP_K
                }
            })
            .collect();
        let mut fused_weights = zeros(device, pairs, DType::F32);
        fused_weights
            .as_mut_slice::<f32>()
            .expect("fused routing weights")
            .fill(1.0 / TOP_K as f32);
        Self {
            rows,
            x: f32_buffer(device, r * h, rng, 1.0),
            residual: f32_buffer(device, r * h, rng, 1.0),
            logits: zeros(device, r * EXPERTS as usize, DType::F32),
            selected: zeros(device, pairs, DType::U32),
            selected_weights: zeros(device, pairs, DType::F32),
            verify_rows: u32_buffer(device, &[LANE_ROWS; 4]),
            ids: u32_buffer(device, &routes.ids),
            routing_weights: weights,
            sh_logit: zeros(device, r, DType::F32),
            a_s: zeros(device, r * s, DType::F32),
            b_s: zeros(device, r * s, DType::F32),
            h_s: zeros(device, r * s, DType::F32),
            y_s: zeros(device, r * h, DType::F32),
            gate_all: zeros(device, pairs * i, DType::F32),
            up_all: zeros(device, pairs * i, DType::F32),
            h_all: zeros(device, pairs * i, DType::F32),
            y_all: zeros(device, pairs * h, DType::F32),
            gathered: zeros(
                device,
                mlx_native::grouped_m8_gathered_rows(rows, TOP_K, EXPERTS).expect("gathered rows")
                    * h,
                DType::BF16,
            ),
            silu_shared: u32_buffer(device, &[(r * s) as u32]),
            silu_expert: u32_buffer(device, &[(pairs * i) as u32]),
            out: zeros(device, r * h, DType::F32),
            scratch: IdMmScratch::alloc(device, EXPERTS, rows).expect("scratch"),
            tile_gathered: zeros(device, grouped * h, DType::BF16),
            tile_intermediate: zeros(device, grouped * i, DType::BF16),
            tile_grids: zeros(device, MOE_TILES_GRID_WORDS, DType::U32),
            shared_ids: u32_buffer(
                device,
                &(0..rows).map(|row| row / LANE_ROWS).collect::<Vec<_>>(),
            ),
            shared_scratch: IdMmScratch::alloc_with_top_k(
                device,
                rows.div_ceil(LANE_ROWS),
                rows,
                1,
            )
            .expect("shared scratch"),
            shared_gathered: zeros(device, shared_rows * h, DType::BF16),
            shared_intermediate: zeros(device, shared_rows * s, DType::BF16),
            shared_grids: zeros(device, MOE_TILES_GRID_WORDS, DType::U32),
            fused_ids: u32_buffer(device, &unmasked),
            fused_weights,
            fused_verify_rows: u32_buffer(device, &lane_rows),
            fused_schedule: zeros(
                device,
                moe_tiles_fused_schedule_words(&fused).expect("fused schedule"),
                DType::U32,
            ),
            fused_routes: zeros(device, fused_rows, DType::U32),
            fused_grids: zeros(device, MOE_TILES_FUSED_GRID_WORDS, DType::U32),
            fused_gathered: zeros(device, fused_rows * h, DType::BF16),
            fused_intermediate: zeros(device, fused_rows * i, DType::BF16),
            fused_out: zeros(
                device,
                moe_tiles_fused_output_rows(&fused).expect("fused output") * h,
                DType::F32,
            ),
        }
    }
}

/// The stages of the staged-M8 route of hf2q's `moe_batched.rs`
/// (`HF2Q_QWEN_MOE_BATCHED=1 HF2Q_QWEN_MOE_STAGED_M8=1` under a route mask).
const STAGED_F32: &[Stage] = &[
    ("router", Base::Alone, true),
    ("shared_gate_logit", Base::Alone, true),
    ("shared_gate", Base::Alone, true),
    ("shared_up", Base::Alone, true),
    ("route_select", Base::Alone, true),
    ("route_mask", Base::Alone, true),
    ("shared_swiglu", Base::Alone, true),
    ("expert_gate(+schedule,gather)", Base::Alone, true),
    ("expert_up", Base::After(7), true),
    ("shared_down", Base::Alone, true),
    ("expert_swiglu", Base::Alone, true),
    ("expert_down(+gather)", Base::After(7), true),
    ("weighted_reduce", Base::Alone, true),
];

fn dense_f32(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    rows: u32,
    n: u32,
) {
    if rows <= 8 {
        dispatch_dense_matvec_f32_rows(enc, registry, device, input, weight, output, rows, n, H)
            .expect("dense matvec");
    } else {
        dense_matmul_f32_f32_tensor(
            enc,
            registry,
            device,
            weight,
            input,
            output,
            &DenseMmF32F32Params {
                m: rows,
                n,
                k: H,
                src0_batch: 1,
                src1_batch: 1,
            },
        )
        .expect("dense matmul");
    }
}

/// Encode stage `stage` of `STAGED_F32` with the barrier that follows it in
/// the served graph.
fn encode_staged_f32(
    stage: usize,
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    w: &Weights,
    l: &mut Layer,
) {
    let rows = l.rows;
    let shared = |ggml_type, n, k| GgmlQuantizedMatmulParams {
        m: rows,
        n,
        k,
        ggml_type,
    };
    match stage {
        0 => dense_f32(
            enc, registry, device, &l.x, &w.router, &l.logits, rows, EXPERTS,
        ),
        1 => dense_f32(
            enc,
            registry,
            device,
            &l.x,
            &w.shared_logit,
            &l.sh_logit,
            rows,
            1,
        ),
        2 => quantized_matmul_ggml(
            enc,
            registry,
            device,
            &l.x,
            &w.shared_gate,
            &l.a_s,
            &shared(GgmlType::Q4_K, SHARED, H),
        )
        .expect("shared gate"),
        3 => {
            quantized_matmul_ggml(
                enc,
                registry,
                device,
                &l.x,
                &w.shared_up,
                &l.b_s,
                &shared(GgmlType::Q4_K, SHARED, H),
            )
            .expect("shared up");
            enc.memory_barrier();
        }
        4 => {
            // The selection runs on its own outputs: the expert stages replay
            // the recorded routes in `l.ids`.
            dispatch_moe_softmax_topk_simd(
                enc,
                registry,
                device,
                &l.logits,
                &l.selected,
                &l.selected_weights,
                rows,
                EXPERTS,
                TOP_K,
            )
            .expect("route select");
            enc.memory_barrier();
        }
        5 => dispatch_dflash2_mask_moe_routes(
            enc,
            registry,
            device,
            &l.verify_rows,
            &l.selected,
            &l.selected_weights,
            DFlash2MoeRouteMaskParams {
                physical_lane: [0, 1, 2, 3],
                lanes: rows / LANE_ROWS,
                rows_per_lane: LANE_ROWS,
                top_k: TOP_K,
                n_experts: EXPERTS,
                width: rows,
            },
        )
        .expect("route mask"),
        6 => {
            dispatch_silu_mul(
                enc,
                registry,
                device.metal_device(),
                &l.a_s,
                &l.b_s,
                &l.h_s,
                &l.silu_shared,
                rows * SHARED,
            )
            .expect("shared swiglu");
            enc.memory_barrier();
        }
        7 | 8 => {
            let (prepared, output, reuse) = if stage == 7 {
                (&w.gate, &l.gate_all, false)
            } else {
                (&w.up, &l.up_all, true)
            };
            quantized_matmul_id_ggml_grouped_m8_epilogue_masked(
                enc,
                registry,
                device,
                &l.x,
                &l.gathered,
                prepared,
                &l.ids,
                output,
                None,
                &mut l.scratch,
                &GgmlQuantizedMatmulIdParams {
                    n_tokens: rows,
                    ..w.gate_up_params
                },
                false,
                reuse,
                reuse,
                GroupedM8Input::F32,
                GroupedM8Epilogue::F32,
            )
            .expect("expert gate/up");
        }
        9 => {
            quantized_matmul_ggml(
                enc,
                registry,
                device,
                &l.h_s,
                &w.shared_down,
                &l.y_s,
                &shared(w.shared_down_type, H, SHARED),
            )
            .expect("shared down");
            enc.memory_barrier();
        }
        10 => {
            dispatch_silu_mul(
                enc,
                registry,
                device.metal_device(),
                &l.gate_all,
                &l.up_all,
                &l.h_all,
                &l.silu_expert,
                rows * TOP_K * INTERMEDIATE,
            )
            .expect("expert swiglu");
            enc.memory_barrier();
        }
        11 => {
            quantized_matmul_id_ggml_grouped_m8_epilogue_masked(
                enc,
                registry,
                device,
                &l.h_all,
                &l.gathered,
                &w.down,
                &l.ids,
                &l.y_all,
                None,
                &mut l.scratch,
                &GgmlQuantizedMatmulIdParams {
                    n_tokens: rows,
                    ..w.down_params
                },
                true,
                true,
                false,
                GroupedM8Input::F32,
                GroupedM8Epilogue::F32,
            )
            .expect("expert down");
            enc.memory_barrier();
        }
        12 => {
            dispatch_moe_weighted_reduce_masked(
                enc,
                registry,
                device,
                &l.routing_weights,
                &l.y_all,
                &l.sh_logit,
                &l.y_s,
                &l.residual,
                &mut l.out,
                rows,
                TOP_K,
                H,
                true,
            )
            .expect("weighted reduce");
            enc.memory_barrier();
        }
        _ => unreachable!("staged F32 stage"),
    }
}

/// The stage-by-stage live-tile form of hf2q's `moe_live_tiles.rs`
/// (`moe_tiles_*`, the fallback of its live-tile route): router and
/// shared-gate logit per block of eight rows on the single-row kernel, one
/// route schedule, one gate/up pass with the BF16 SwiGLU left in the
/// schedule's rows, and down on the same rows. The shared expert runs the
/// same three stages beside the routed experts, a tile per block of rows.
const LIVE_TILES: &[Stage] = &[
    ("router", Base::Alone, true),
    ("shared_gate_logit", Base::Alone, true),
    ("shared_schedule", Base::Alone, true),
    ("route_select", Base::Alone, true),
    ("shared_gather", Base::After(2), true),
    ("route_mask", Base::Alone, true),
    ("schedule", Base::Alone, true),
    ("gather", Base::After(6), true),
    ("expert_gate_up", Base::After(6), true),
    ("shared_gate_up", Base::After(2), true),
    ("expert_down", Base::After(6), true),
    ("shared_down", Base::After(2), true),
    ("weighted_reduce", Base::Alone, true),
];

fn tile_params(w: &Weights, rows: u32, shared: bool) -> MoeTilesParams {
    MoeTilesParams {
        n_tokens: rows,
        top_k: if shared { 1 } else { TOP_K },
        n_experts: if shared {
            rows.div_ceil(LANE_ROWS)
        } else {
            EXPERTS
        },
        hidden: H,
        intermediate: if shared { SHARED } else { INTERMEDIATE },
        masked_routes: !shared,
        column_tile: w.column_tile,
        shared_planes: shared,
    }
}

/// Encode stage `stage` of `LIVE_TILES` with the barrier that follows it.
fn encode_live_tiles(
    stage: usize,
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    w: &Weights,
    l: &mut Layer,
) {
    let rows = l.rows;
    let lane_rows = |buffer: &MlxBuffer, lane: u32, width: u32| {
        buffer.slice_view(
            u64::from(lane * LANE_ROWS * width) * 4,
            (LANE_ROWS * width) as usize,
        )
    };
    let shared = matches!(stage, 2 | 4 | 9 | 11);
    let params = tile_params(w, rows, shared);
    let buffers = if shared {
        MoeTilesBuffers {
            gathered: &l.shared_gathered,
            intermediate: &l.shared_intermediate,
            grids: &l.shared_grids,
        }
    } else {
        MoeTilesBuffers {
            gathered: &l.tile_gathered,
            intermediate: &l.tile_intermediate,
            grids: &l.tile_grids,
        }
    };
    match stage {
        0 | 1 => {
            let (weight, output, n) = if stage == 0 {
                (&w.router, &l.logits, EXPERTS)
            } else {
                (&w.shared_logit, &l.sh_logit, 1)
            };
            for lane in 0..rows / LANE_ROWS {
                dispatch_dense_matvec_f32_rows(
                    enc,
                    registry,
                    device,
                    &lane_rows(&l.x, lane, H),
                    weight,
                    &lane_rows(output, lane, n),
                    LANE_ROWS,
                    n,
                    H,
                )
                .expect("row-exact router");
            }
        }
        // The schedule's own barrier closes the first phase.
        2 => moe_tiles_schedule(
            enc,
            registry,
            device,
            &l.shared_ids,
            &mut l.shared_scratch,
            &params,
        )
        .expect("shared schedule"),
        3 => dispatch_moe_softmax_topk_simd(
            enc,
            registry,
            device,
            &l.logits,
            &l.selected,
            &l.selected_weights,
            rows,
            EXPERTS,
            TOP_K,
        )
        .expect("route select"),
        4 => {
            moe_tiles_gather(
                enc,
                registry,
                device,
                &l.x,
                &l.shared_ids,
                &buffers,
                &l.shared_scratch,
                &params,
            )
            .expect("shared gather");
            enc.memory_barrier();
        }
        5 => {
            encode_staged_f32(5, enc, registry, device, w, l);
            enc.memory_barrier();
        }
        6 => moe_tiles_schedule(enc, registry, device, &l.ids, &mut l.scratch, &params)
            .expect("schedule"),
        7 => {
            moe_tiles_gather(
                enc, registry, device, &l.x, &l.ids, &buffers, &l.scratch, &params,
            )
            .expect("gather");
            enc.memory_barrier();
        }
        8 => moe_tiles_gate_up(
            enc, registry, device, &w.gate, &w.up, &l.ids, &buffers, &l.scratch, &params,
        )
        .expect("gate/up"),
        9 => {
            moe_tiles_gate_up(
                enc,
                registry,
                device,
                &w.shared_tiles[0],
                &w.shared_tiles[1],
                &l.shared_ids,
                &buffers,
                &l.shared_scratch,
                &params,
            )
            .expect("shared gate/up");
            enc.memory_barrier();
        }
        10 => moe_tiles_down(
            enc, registry, device, &w.down, &l.ids, &buffers, &l.scratch, &l.y_all, &params,
        )
        .expect("down"),
        11 => {
            moe_tiles_down(
                enc,
                registry,
                device,
                &w.shared_tiles[2],
                &l.shared_ids,
                &buffers,
                &l.shared_scratch,
                &l.y_s,
                &params,
            )
            .expect("shared down");
            enc.memory_barrier();
        }
        12 => encode_staged_f32(12, enc, registry, device, w, l),
        _ => unreachable!("live tiles stage"),
    }
}

/// The live-tile route of hf2q's `moe_live_tiles_fused.rs`
/// (`moe_tiles_fused`): router and shared-gate logit as one dispatch each
/// over the blocks of eight rows, the selection, then one schedule (with the
/// verify-row mask), one gather and one dispatch per pass for the routed
/// experts and the shared expert together. The expert stages are timed
/// cumulatively: each row is the stage's command minus the one that stops a
/// stage earlier.
const FUSED_TILES: &[Stage] = &[
    ("router", Base::Alone, true),
    ("shared_gate_logit", Base::Alone, true),
    ("route_select", Base::Alone, true),
    ("schedule(+mask,shared)", Base::Alone, false),
    ("gather(+shared)", Base::Minus(3), false),
    ("gate_up(+shared)", Base::Minus(4), false),
    ("down(+shared)", Base::Minus(5), true),
    ("weighted_reduce", Base::Alone, true),
];

fn fused_params(rows: u32, column_tile: bool) -> MoeTilesFusedParams {
    MoeTilesFusedParams {
        n_tokens: rows,
        top_k: TOP_K,
        n_experts: EXPERTS,
        hidden: H,
        intermediate: INTERMEDIATE,
        column_tile,
    }
}

/// Encode stage `stage` of `FUSED_TILES` with the barrier that follows it.
fn encode_fused_tiles(
    stage: usize,
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    w: &Weights,
    l: &mut Layer,
) {
    let rows = l.rows;
    match stage {
        // Router and shared-gate logit: every block of eight rows in one
        // dispatch each.
        0 | 1 => {
            let (weight, output, n) = if stage == 0 {
                (&w.router, &l.logits, EXPERTS)
            } else {
                (&w.shared_logit, &l.sh_logit, 1)
            };
            dispatch_dense_matvec_f32_row_blocks(
                enc,
                registry,
                device,
                &l.x,
                weight,
                output,
                LANE_ROWS,
                rows / LANE_ROWS,
                n,
                H,
            )
            .expect("row-exact router blocks");
            if stage == 1 {
                enc.memory_barrier();
            }
        }
        2 => {
            encode_live_tiles(3, enc, registry, device, w, l);
            enc.memory_barrier();
        }
        3..=6 => {
            moe_tiles_fused(
                enc,
                registry,
                device,
                &l.x,
                &l.fused_ids,
                &l.fused_weights,
                Some(&MoeTilesFusedMask {
                    verify_rows: &l.fused_verify_rows,
                    physical_lane: [0, 1, 2, 3],
                    lanes: rows / LANE_ROWS,
                    rows_per_lane: LANE_ROWS,
                }),
                &MoeTilesFusedWeights {
                    gate: &w.gate,
                    up: &w.up,
                    down: &w.down,
                    shared_gate: &w.shared_tiles[0],
                    shared_up: &w.shared_tiles[1],
                    shared_down: &w.shared_tiles[2],
                },
                &MoeTilesFusedBuffers {
                    schedule: &l.fused_schedule,
                    routes: &l.fused_routes,
                    grids: &l.fused_grids,
                    gathered: &l.fused_gathered,
                    intermediate: &l.fused_intermediate,
                },
                &l.fused_out,
                &fused_params(rows, w.column_tile),
                [
                    MoeTilesFusedStage::Schedule,
                    MoeTilesFusedStage::Gather,
                    MoeTilesFusedStage::GateUp,
                    MoeTilesFusedStage::Down,
                ][stage - 3],
            )
            .expect("fused tiles");
            enc.memory_barrier();
        }
        7 => {
            let pairs = (rows * TOP_K) as usize;
            let shared = l.fused_out.slice_view(
                (pairs * H as usize * 4) as u64,
                rows as usize * H as usize,
            );
            dispatch_moe_weighted_reduce_masked(
                enc,
                registry,
                device,
                &l.fused_weights,
                &l.fused_out,
                &l.sh_logit,
                &shared,
                &l.residual,
                &mut l.out,
                rows,
                TOP_K,
                H,
                true,
            )
            .expect("weighted reduce");
            enc.memory_barrier();
        }
        _ => unreachable!("fused tiles stage"),
    }
}

/// How a stage's time is taken: its own command, its command after an
/// earlier stage whose schedule it reuses (minus that stage alone), or its
/// own command minus an earlier stage's that it repeats.
#[derive(Clone, Copy)]
enum Base {
    Alone,
    After(usize),
    Minus(usize),
}

/// A stage's name, how it is timed, and whether a whole layer encodes it.
type Stage = (&'static str, Base, bool);

type Encode = fn(usize, &mut CommandEncoder, &mut KernelRegistry, &MlxDevice, &Weights, &mut Layer);

fn gpu_ms(enc: &mut CommandEncoder) -> f64 {
    let (start, end) = enc.commit_wait_with_gpu_time().expect("commit");
    (end - start) * 1.0e3
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    samples[samples.len() / 2]
}

/// Mean GPU ms per layer of every stage as its own command, the median GPU
/// ms per layer of `layers.len()` whole layers in one command, and the same
/// with every layer committed and waited for as its own command (how a
/// stage profile commits it). Layer `i` reads weight set `i % sets.len()`.
fn measure(
    label: &str,
    stages: &[Stage],
    encode: Encode,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    sets: &[Weights],
    layers: &mut [Layer],
    rounds: usize,
    with_stages: bool,
    idle_ms: f64,
) {
    let whole = |registry: &mut KernelRegistry, layers: &mut [Layer]| {
        let mut enc = device.command_encoder().expect("encoder");
        for (index, layer) in layers.iter_mut().enumerate() {
            for stage in (0..stages.len()).filter(|stage| stages[*stage].2) {
                encode(
                    stage,
                    &mut enc,
                    registry,
                    device,
                    &sets[index % sets.len()],
                    layer,
                );
            }
        }
        gpu_ms(&mut enc) / layers.len() as f64
    };
    whole(registry, layers); // compile pipelines, fill every buffer
    let separate = |registry: &mut KernelRegistry, layers: &mut [Layer]| {
        let mut total = 0.0;
        for (index, layer) in layers.iter_mut().enumerate() {
            let mut enc = device.command_encoder().expect("encoder");
            for stage in (0..stages.len()).filter(|stage| stages[*stage].2) {
                encode(
                    stage,
                    &mut enc,
                    registry,
                    device,
                    &sets[index % sets.len()],
                    layer,
                );
            }
            total += gpu_ms(&mut enc);
        }
        total / layers.len() as f64
    };
    // `stage` alone (or with `prefix` before it) for every layer, serialized.
    let mut serial = |prefix: Option<usize>, stage: Option<usize>, layers: &mut [Layer]| {
        let mut samples: Vec<f64> = (0..rounds)
            .map(|_| {
                let mut enc = device.command_encoder().expect("encoder");
                for (index, layer) in layers.iter_mut().enumerate() {
                    for step in [prefix, stage].into_iter().flatten() {
                        encode(
                            step,
                            &mut enc,
                            registry,
                            device,
                            &sets[index % sets.len()],
                            layer,
                        );
                        enc.memory_barrier();
                    }
                }
                gpu_ms(&mut enc) / layers.len() as f64
            })
            .collect();
        median(&mut samples)
    };
    let per_stage: Vec<f64> = if with_stages {
        stages
            .iter()
            .enumerate()
            .map(|(stage, (_, base, _))| match base {
                Base::After(earlier) => {
                    serial(Some(*earlier), Some(stage), layers)
                        - serial(Some(*earlier), None, layers)
                }
                Base::Minus(earlier) => {
                    serial(None, Some(stage), layers) - serial(None, Some(*earlier), layers)
                }
                Base::Alone => serial(None, Some(stage), layers),
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut samples: Vec<f64> = (0..rounds).map(|_| whole(registry, layers)).collect();
    let layer_ms = median(&mut samples);
    let fastest = samples[0];
    let mut samples: Vec<f64> = (0..rounds)
        .map(|_| separate(registry, layers))
        .collect();
    let separate_ms = median(&mut samples);
    println!(
        "{label}: {layer_ms:.4} ms per layer in one command (fastest {fastest:.4}), \
         {separate_ms:.4} with every layer its own waited command (medians of {rounds})"
    );
    if idle_ms > 0.0 {
        // The GPU left idle before every command, as a served cycle leaves
        // it while the host plans, commits and drafts.
        let mut samples: Vec<f64> = (0..rounds)
            .map(|_| {
                std::thread::sleep(std::time::Duration::from_secs_f64(idle_ms / 1.0e3));
                whole(registry, layers)
            })
            .collect();
        println!(
            "    one command after {idle_ms} ms of idle GPU: {:.4} ms per layer (median of {rounds})",
            median(&mut samples)
        );
    }
    if with_stages {
        for ((name, ..), ms) in stages.iter().zip(&per_stage) {
            println!("    {name:<32} {ms:.4}");
        }
        println!(
            "    {:<32} {:.4}",
            "sum of stages",
            per_stage.iter().sum::<f64>()
        );
    }
}

/// One set of expert, shared-expert and router weights.
fn weight_set(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    rng: &mut Rng,
    rows: u32,
    down_type: GgmlType,
    column_tile: bool,
) -> Weights {
    let (gate_q, gate_stride) = quantized(device, GgmlType::Q4_K, EXPERTS, INTERMEDIATE, H, rng);
    let (up_q, _) = quantized(device, GgmlType::Q4_K, EXPERTS, INTERMEDIATE, H, rng);
    let (down_q, down_stride) = quantized(device, down_type, EXPERTS, H, INTERMEDIATE, rng);
    let gate_up_params = GgmlQuantizedMatmulIdParams {
        n_tokens: rows,
        top_k: TOP_K,
        n: INTERMEDIATE,
        k: H,
        n_experts: EXPERTS,
        expert_stride: gate_stride,
        ggml_type: GgmlType::Q4_K,
    };
    let down_params = GgmlQuantizedMatmulIdParams {
        n: H,
        k: INTERMEDIATE,
        expert_stride: down_stride,
        ggml_type: down_type,
        ..gate_up_params
    };
    let shared_gate = quantized(device, GgmlType::Q4_K, 1, SHARED, H, rng).0;
    let shared_up = quantized(device, GgmlType::Q4_K, 1, SHARED, H, rng).0;
    let shared_down = quantized(device, down_type, 1, H, SHARED, rng).0;
    let one = |params: &GgmlQuantizedMatmulIdParams| GgmlQuantizedMatmulIdParams {
        n_experts: 1,
        ..*params
    };
    let mut enc = device.command_encoder().expect("prepare encoder");
    let mut prepare = |weight: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams| {
        prepare_quantized_matmul_id_ggml_grouped_m8(&mut enc, registry, device, weight, params)
            .expect("prepare experts")
    };
    let gate = prepare(&gate_q, &gate_up_params);
    let up = prepare(&up_q, &gate_up_params);
    let down = prepare(&down_q, &down_params);
    let shared_tiles = [
        prepare(&shared_gate, &one(&gate_up_params)),
        prepare(&shared_up, &one(&gate_up_params)),
        prepare(&shared_down, &one(&down_params)),
    ];
    enc.commit_and_wait().expect("commit preparation");
    Weights {
        router: f32_buffer(device, (EXPERTS * H) as usize, rng, 0.05),
        shared_logit: f32_buffer(device, H as usize, rng, 0.05),
        shared_gate,
        shared_up,
        shared_down,
        shared_down_type: down_type,
        gate,
        up,
        down,
        gate_up_params,
        down_params,
        column_tile,
        shared_tiles,
    }
}

fn main() {
    let mut source = "pooled".to_string();
    let (mut rows, mut rounds, mut layer_count, mut set_count) = (32u32, 20usize, 16usize, 1usize);
    let mut down_type = GgmlType::Q4_K;
    let (mut route, mut with_stages) = ("all".to_string(), true);
    let mut idle_ms = 0.0f64;
    for argument in std::env::args().skip(1) {
        match argument.split_once('=') {
            Some(("routes", value)) => source = value.to_string(),
            Some(("rows", value)) => rows = value.parse().expect("rows"),
            Some(("rounds", value)) => rounds = value.parse().expect("rounds"),
            Some(("layers", value)) => layer_count = value.parse().expect("layers"),
            Some(("weights", value)) => set_count = value.parse().expect("weights"),
            Some(("route", value)) => route = value.to_string(),
            Some(("stages", value)) => with_stages = value != "0",
            Some(("idle", value)) => idle_ms = value.parse().expect("idle"),
            Some(("down", "q4k")) => down_type = GgmlType::Q4_K,
            Some(("down", "q6k")) => down_type = GgmlType::Q6_K,
            // `cargo bench` passes `--bench`.
            _ if argument.starts_with("--") => {}
            _ => panic!("unknown argument {argument}"),
        }
    }
    assert!(
        rows % LANE_ROWS == 0 && (LANE_ROWS..=32).contains(&rows),
        "rows is 8, 16, 24 or 32"
    );
    assert!(set_count >= 1, "weights is at least 1");

    let device = MlxDevice::new().expect("device");
    let mut registry = KernelRegistry::new();
    let mut rng = Rng(7);
    let mut routes = match source.as_str() {
        "pooled" => synthetic_routes(rows, true, layer_count, &mut rng),
        "random" => synthetic_routes(rows, false, layer_count, &mut rng),
        path => load_routes(path, rows),
    };
    assert!(!routes.is_empty(), "no routes of {rows} rows in {source}");
    // An even sample over the recorded steps.
    if routes.len() > layer_count {
        let step = routes.len() / layer_count;
        routes = (0..layer_count)
            .map(|index| routes[index * step].clone())
            .collect();
    }
    let n = routes.len() as f64;
    let mean = |f: &dyn Fn(&Routes) -> usize| routes.iter().map(|r| f(r)).sum::<usize>() as f64 / n;
    println!(
        "{} on {}: {rows} rows, {} layers, down {down_type:?}: live rows {:.1}, live experts {:.1}, \
         experts of 1-2 routes {:.1}, eight-row tiles {:.1}",
        source,
        device.name(),
        routes.len(),
        mean(&|r| r.live_rows()),
        mean(&|r| r.tiles().0),
        mean(&|r| r.tiles().1),
        mean(&|r| r.tiles().2),
    );

    let column_tile = moe_tiles_column_tile_exact(&device, &mut registry).expect("column probe");
    println!("register path bitwise the matmul tile: {column_tile}; weight sets {set_count}");
    let sets: Vec<Weights> = (0..set_count)
        .map(|_| {
            weight_set(
                &device,
                &mut registry,
                &mut rng,
                rows,
                down_type,
                column_tile,
            )
        })
        .collect();
    let mut layers: Vec<Layer> = routes
        .iter()
        .map(|routes| Layer::new(&device, routes, &mut rng))
        .collect();

    let runs: [(&str, &str, &[Stage], Encode); 3] = [
        (
            "staged",
            "staged M8, F32 gate/up (served)",
            STAGED_F32,
            encode_staged_f32,
        ),
        (
            "live",
            "live tiles, one gate/up pass",
            LIVE_TILES,
            encode_live_tiles,
        ),
        (
            "fused",
            "fused tiles, shared expert in the routed grid",
            FUSED_TILES,
            encode_fused_tiles,
        ),
    ];
    assert!(
        route == "all" || runs.iter().any(|(name, ..)| *name == route),
        "route is all, staged, live or fused"
    );
    for (name, label, stages, encode) in runs {
        if route == "all" || route == name {
            measure(
                label,
                stages,
                encode,
                &device,
                &mut registry,
                &sets,
                &mut layers,
                rounds,
                with_stages,
                idle_ms,
            );
        }
    }
}
