//! Fused MoE expert tiles: the routed experts and the shared expert of one
//! layer as one schedule, one gather and one dispatch per expert pass
//! (`shaders/moe_expert_tiles.metal`, the `moe_tiles_fused_*` kernels).
//!
//! [`moe_tiles_fused`] encodes four dispatches with a barrier between them:
//!
//! 1. the schedule: applies the verify-row mask to the route table, groups
//!    the live routes by expert into tiles of at most eight rows, appends the
//!    shared expert's tiles (one per block of eight rows of the step) and
//!    writes the indirect grids of the three dispatches that follow;
//! 2. the gather of every tile's input rows as BF16;
//! 3. gate and up in one pass, `BF16(SiLU(gate) * up)` left in the grouped
//!    rows;
//! 4. down, each route's FP32 output at its own row: route `token * top_k +
//!    slot` at that row, the shared expert's output of `token` at row
//!    `n_tokens * top_k + token`.
//!
//! Every route is computed by the arithmetic of the live-tile stages
//! ([`super::moe_tiles_gate_up`], [`super::moe_tiles_down`]): an expert of at
//! most two routes in registers when [`MoeTilesFusedParams::column_tile`] is
//! set, any other in eight-row matmul tiles, whose rows are independent of
//! one another. The two routes therefore produce the same bits; what differs
//! is that an expert of more than eight routes is several threadgroups, the
//! shared expert shares the routed experts' grid, and the mask, the two
//! schedules and the grid writes are one dispatch.

use super::{
    as_bytes, checked_byte_extent, CommandEncoder, DType, GgmlType, KernelArg, KernelRegistry,
    MlxBuffer, MlxDevice, MlxError, PreparedGroupedM8Experts, Result,
};

/// Words of the three indirect grids the schedule writes.
pub const MOE_TILES_FUSED_GRID_WORDS: usize = 9;
/// Routes one step may hold: one per thread of the schedule's threadgroup.
pub const MOE_TILES_FUSED_MAX_ROUTES: u32 = 256;
/// Experts the schedule's threadgroup scans.
pub const MOE_TILES_FUSED_MAX_EXPERTS: u32 = 256;

const TILE_ROWS: u32 = 8;
/// Routes of an expert that the register path runs when it is enabled.
const COLUMN_ROWS: u32 = 2;
const SCHEDULE_KERNEL: &str = "moe_tiles_fused_schedule";
const GATHER_KERNEL: &str = "moe_tiles_fused_gather_f32";

/// One step's shape.
#[derive(Debug, Clone, Copy)]
pub struct MoeTilesFusedParams {
    pub n_tokens: u32,
    pub top_k: u32,
    pub n_experts: u32,
    /// Input width of gate/up and output width of down, routed and shared.
    pub hidden: u32,
    /// Output width of gate/up and input width of down, routed and shared.
    pub intermediate: u32,
    /// Run experts of one or two routes in registers; pass
    /// [`super::moe_tiles_column_tile_exact`] (see
    /// [`super::MoeTilesParams::column_tile`]).
    pub column_tile: bool,
}

/// The DFlash2 verify-row mask: a row of a lane's block of `rows_per_lane`
/// rows is live while its index is below `verify_rows[physical_lane[lane]]`;
/// rows past `lanes * rows_per_lane` are masked.
#[derive(Clone, Copy)]
pub struct MoeTilesFusedMask<'a> {
    /// U32, one entry per physical lane.
    pub verify_rows: &'a MlxBuffer,
    pub physical_lane: [u32; 4],
    pub lanes: u32,
    pub rows_per_lane: u32,
}

/// The prepared planes of one layer: the routed experts' stacks and the
/// shared expert's (prepared with `n_experts = 1`).
#[derive(Clone, Copy)]
pub struct MoeTilesFusedWeights<'a> {
    pub gate: &'a PreparedGroupedM8Experts,
    pub up: &'a PreparedGroupedM8Experts,
    pub down: &'a PreparedGroupedM8Experts,
    pub shared_gate: &'a PreparedGroupedM8Experts,
    pub shared_up: &'a PreparedGroupedM8Experts,
    pub shared_down: &'a PreparedGroupedM8Experts,
}

/// Caller-owned scratch of one step; the contents need no initialization.
/// `schedule` is U32 `[moe_tiles_fused_schedule_words]`, `routes` U32
/// `[moe_tiles_fused_grouped_rows]`, `grids` U32
/// `[MOE_TILES_FUSED_GRID_WORDS]`, `gathered` BF16
/// `[moe_tiles_fused_grouped_rows, hidden]` and `intermediate` BF16
/// `[moe_tiles_fused_grouped_rows, intermediate]`.
#[derive(Clone, Copy)]
pub struct MoeTilesFusedBuffers<'a> {
    pub schedule: &'a MlxBuffer,
    pub routes: &'a MlxBuffer,
    pub grids: &'a MlxBuffer,
    pub gathered: &'a MlxBuffer,
    pub intermediate: &'a MlxBuffer,
}

/// The last stage [`moe_tiles_fused`] encodes. Everything but
/// [`MoeTilesFusedStage::Down`] is for benchmarks that time the stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MoeTilesFusedStage {
    Schedule,
    Gather,
    GateUp,
    Down,
}

fn invalid(message: String) -> MlxError {
    MlxError::InvalidArgument(message)
}

fn check_shape(params: &MoeTilesFusedParams) -> Result<()> {
    let routes = u64::from(params.n_tokens) * u64::from(params.top_k);
    if params.n_tokens == 0
        || params.top_k == 0
        || params.top_k > params.n_experts
        || params.n_experts > MOE_TILES_FUSED_MAX_EXPERTS
        || routes > u64::from(MOE_TILES_FUSED_MAX_ROUTES)
        || params.hidden == 0
        || params.hidden % 256 != 0
        || params.intermediate == 0
        || params.intermediate % 256 != 0
    {
        return Err(invalid(format!(
            "fused MoE tiles require n_tokens > 0, 0 < top_k <= n_experts <= \
             {MOE_TILES_FUSED_MAX_EXPERTS}, n_tokens * top_k <= {MOE_TILES_FUSED_MAX_ROUTES} and \
             widths that are multiples of 256; got {params:?}"
        )));
    }
    Ok(())
}

/// Most tiles a step can hold: an expert of `c` routes takes `ceil(c / 8)`
/// tiles and no tile is empty; then one shared tile per block of eight rows.
fn max_tiles(params: &MoeTilesFusedParams) -> u32 {
    let routes = params.n_tokens * params.top_k;
    let routed = routes.min(routes / TILE_ROWS + params.n_experts.min(routes));
    routed + params.n_tokens.div_ceil(TILE_ROWS)
}

/// Rows the grouped buffers of [`MoeTilesFusedBuffers`] hold (and entries of
/// its `routes`): every route, the shared expert's row of every token, and
/// the seven rows an eight-row matmul tile reads, and drops, past its last
/// live row.
pub fn moe_tiles_fused_grouped_rows(params: &MoeTilesFusedParams) -> Result<usize> {
    check_shape(params)?;
    Ok((params.n_tokens * params.top_k + params.n_tokens + TILE_ROWS - 1) as usize)
}

/// Words of [`MoeTilesFusedBuffers::schedule`]: the tile count and three
/// words per tile the step can hold.
pub fn moe_tiles_fused_schedule_words(params: &MoeTilesFusedParams) -> Result<usize> {
    check_shape(params)?;
    Ok(1 + 3 * max_tiles(params) as usize)
}

/// Rows of the FP32 output of [`moe_tiles_fused`]: the routed routes, then
/// the shared expert's row of every token.
pub fn moe_tiles_fused_output_rows(params: &MoeTilesFusedParams) -> Result<usize> {
    check_shape(params)?;
    Ok((params.n_tokens * params.top_k + params.n_tokens) as usize)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScheduleGpuParams {
    n_tokens: u32,
    top_k: u32,
    n_experts: u32,
    max_tiles: u32,
    column_rows: u32,
    shared_tiles: u32,
    masked: u32,
    lanes: u32,
    rows_per_lane: u32,
    gather_columns: u32,
    gate_up_columns: u32,
    down_columns: u32,
    physical_lane: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherGpuParams {
    k: u32,
    top_k: u32,
    route_count: u32,
    _pad: u32,
}

/// `MoeTilesParams` of the shader; the fused kernels read the column-tile
/// choice from the schedule.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PassGpuParams {
    k: u32,
    n: u32,
    column_rows: u32,
    plane1_offset: u32,
    meta_offset: u32,
    _pad: u32,
    prepared_expert_stride: u64,
}

fn format_name(ggml_type: GgmlType) -> &'static str {
    match ggml_type {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q6_K => "q6_K",
        _ => unreachable!("prepared expert planes are Q4_K or Q6_K"),
    }
}

/// `routed` holds `[n_experts, n, k]` and `shared` `[1, n, k]` in one
/// format, hence one plane layout.
fn check_planes(
    label: &str,
    routed: &PreparedGroupedM8Experts,
    shared: &PreparedGroupedM8Experts,
    n_experts: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if routed.n_experts != n_experts
        || shared.n_experts != 1
        || [routed.n, shared.n] != [n, n]
        || [routed.k, shared.k] != [k, k]
        || routed.ggml_type != shared.ggml_type
        || routed.plane1_offset != shared.plane1_offset
        || routed.meta_offset != shared.meta_offset
    {
        return Err(invalid(format!(
            "fused MoE tiles {label} planes are [{}, {}, {}] {:?} and shared [{}, {}, {}] {:?}; the \
             step needs [{n_experts}, {n}, {k}] and [1, {n}, {k}] in one format",
            routed.n_experts,
            routed.n,
            routed.k,
            routed.ggml_type,
            shared.n_experts,
            shared.n,
            shared.k,
            shared.ggml_type,
        )));
    }
    Ok(())
}

fn check_buffer(buffer: &MlxBuffer, dtype: DType, elements: &[usize], label: &str) -> Result<()> {
    let mut extent = elements.to_vec();
    extent.push(dtype.size_of());
    let bytes = checked_byte_extent(label, &extent)?;
    if buffer.dtype() != dtype || buffer.data_byte_len() < bytes {
        return Err(invalid(format!(
            "fused MoE tiles need {dtype:?} {label} of {bytes} bytes, got {:?} {}",
            buffer.dtype(),
            buffer.data_byte_len()
        )));
    }
    Ok(())
}

/// Whether the fused route can run `weights` on a step of shape `params`:
/// the shape is within the schedule's threadgroup and each projection's
/// routed and shared planes share a format. `Err` names what does not fit.
pub fn moe_tiles_fused_admits(
    weights: &MoeTilesFusedWeights<'_>,
    params: &MoeTilesFusedParams,
) -> Result<()> {
    check_shape(params)?;
    let (h, m, experts) = (params.hidden, params.intermediate, params.n_experts);
    check_planes("gate", weights.gate, weights.shared_gate, experts, m, h)?;
    check_planes("up", weights.up, weights.shared_up, experts, m, h)?;
    check_planes("down", weights.down, weights.shared_down, experts, h, m)?;
    if weights.gate.ggml_type != weights.up.ggml_type {
        return Err(invalid(format!(
            "fused MoE tiles gate/up pass needs one format, got {:?} and {:?}",
            weights.gate.ggml_type, weights.up.ggml_type
        )));
    }
    Ok(())
}

/// Encode the expert stages of one layer over `input` (`[n_tokens, hidden]`
/// F32) and the route table `ids` / `routing_weights` (`[n_tokens * top_k]`
/// U32 / F32, written by an earlier dispatch that the caller ordered before
/// this call), into `output` (`[moe_tiles_fused_output_rows, hidden]` F32).
///
/// With `mask`, the schedule rewrites the table's masked routes to the id
/// `n_experts` and the weight 0 and leaves their output rows unwritten.
/// Every id of a live route must be below `n_experts`; otherwise no tile
/// runs and `output` is left as it was.
///
/// Ends without a barrier: order the reader of `output` after it.
#[allow(clippy::too_many_arguments)]
pub fn moe_tiles_fused(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    ids: &MlxBuffer,
    routing_weights: &MlxBuffer,
    mask: Option<&MoeTilesFusedMask<'_>>,
    weights: &MoeTilesFusedWeights<'_>,
    buffers: &MoeTilesFusedBuffers<'_>,
    output: &MlxBuffer,
    params: &MoeTilesFusedParams,
    through: MoeTilesFusedStage,
) -> Result<()> {
    moe_tiles_fused_admits(weights, params)?;
    let routes = (params.n_tokens * params.top_k) as usize;
    let tokens = params.n_tokens as usize;
    let (h, m) = (params.hidden as usize, params.intermediate as usize);
    let grouped = moe_tiles_fused_grouped_rows(params)?;
    let tiles = max_tiles(params);
    check_buffer(input, DType::F32, &[tokens, h], "input")?;
    check_buffer(ids, DType::U32, &[routes], "ids")?;
    check_buffer(routing_weights, DType::F32, &[routes], "routing weights")?;
    check_buffer(
        buffers.schedule,
        DType::U32,
        &[1 + 3 * tiles as usize],
        "schedule",
    )?;
    check_buffer(buffers.routes, DType::U32, &[grouped], "routes")?;
    check_buffer(
        buffers.grids,
        DType::U32,
        &[MOE_TILES_FUSED_GRID_WORDS],
        "grids",
    )?;
    check_buffer(
        buffers.gathered,
        DType::BF16,
        &[grouped, h],
        "gathered rows",
    )?;
    check_buffer(
        buffers.intermediate,
        DType::BF16,
        &[grouped, m],
        "intermediate rows",
    )?;
    check_buffer(output, DType::F32, &[routes + tokens, h], "output")?;
    if let Some(mask) = mask {
        let physical = mask.verify_rows.element_count();
        if mask.lanes == 0
            || mask.lanes > 4
            || mask.rows_per_lane == 0
            || mask.lanes * mask.rows_per_lane > params.n_tokens
            || mask.verify_rows.dtype() != DType::U32
            || mask.physical_lane[..mask.lanes as usize]
                .iter()
                .any(|&lane| lane as usize >= physical)
        {
            return Err(invalid(format!(
                "fused MoE tiles mask needs 1..=4 lanes of rows_per_lane rows within {} rows and U32 \
                 verify rows covering physical lanes {:?}; got lanes {} rows_per_lane {} verify rows \
                 {:?} {physical}",
                params.n_tokens,
                &mask.physical_lane[..mask.lanes.min(4) as usize],
                mask.lanes,
                mask.rows_per_lane,
                mask.verify_rows.dtype(),
            )));
        }
    }

    let gate_up_kernel = format!(
        "moe_tiles_fused_gate_up_{}",
        format_name(weights.gate.ggml_type)
    );
    let down_kernel = format!(
        "moe_tiles_fused_down_{}",
        format_name(weights.down.ggml_type)
    );
    // A kernel's static threadgroup memory is fixed when it is compiled (and
    // doubled by shader validation), and its thread limit with it; refuse
    // before anything is encoded.
    let memory_limit = device.metal_device().max_threadgroup_memory_length();
    for (kernel, threads) in [
        (SCHEDULE_KERNEL, u64::from(MOE_TILES_FUSED_MAX_ROUTES)),
        (GATHER_KERNEL, 256),
        (gate_up_kernel.as_str(), 64),
        (down_kernel.as_str(), 64),
    ] {
        let pipeline = registry.get_pipeline(kernel, device.metal_device())?;
        let memory = pipeline.static_threadgroup_memory_length();
        if memory > memory_limit || pipeline.max_total_threads_per_threadgroup() < threads {
            return Err(invalid(format!(
                "{kernel} needs {memory} bytes of threadgroup memory and {threads} threads; the \
                 device allows {memory_limit} bytes and the pipeline {} threads",
                pipeline.max_total_threads_per_threadgroup()
            )));
        }
    }

    // ---- 1: mask, schedule, grids ----
    let gpu = ScheduleGpuParams {
        n_tokens: params.n_tokens,
        top_k: params.top_k,
        n_experts: params.n_experts,
        max_tiles: tiles,
        column_rows: if params.column_tile { COLUMN_ROWS } else { 0 },
        shared_tiles: 1,
        masked: u32::from(mask.is_some()),
        lanes: mask.map_or(0, |mask| mask.lanes),
        rows_per_lane: mask.map_or(1, |mask| mask.rows_per_lane),
        gather_columns: params.hidden / 256,
        gate_up_columns: params.intermediate / 32,
        down_columns: params.hidden / 64,
        physical_lane: mask.map_or([0; 4], |mask| mask.physical_lane),
    };
    // Unmasked steps never read the verify rows; any buffer binds the slot.
    let verify_rows = mask.map_or(ids, |mask| mask.verify_rows);
    let pipeline = registry.get_pipeline(SCHEDULE_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(ids)),
            (1, KernelArg::Buffer(routing_weights)),
            (2, KernelArg::Buffer(verify_rows)),
            (3, KernelArg::Buffer(buffers.schedule)),
            (4, KernelArg::Buffer(buffers.routes)),
            (5, KernelArg::Buffer(buffers.grids)),
            (6, KernelArg::Bytes(as_bytes(&gpu))),
        ],
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(u64::from(MOE_TILES_FUSED_MAX_ROUTES), 1, 1),
    );
    if through == MoeTilesFusedStage::Schedule {
        return Ok(());
    }
    encoder.memory_barrier();

    // ---- 2: gather ----
    let grid_bytes = (3 * DType::U32.size_of()) as u64;
    let gpu = GatherGpuParams {
        k: params.hidden,
        top_k: params.top_k,
        route_count: params.n_tokens * params.top_k,
        _pad: 0,
    };
    let pipeline = registry.get_pipeline(GATHER_KERNEL, device.metal_device())?;
    encoder.encode_threadgroups_indirect_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(buffers.gathered)),
            (2, KernelArg::Buffer(buffers.schedule)),
            (3, KernelArg::Buffer(buffers.routes)),
            (4, KernelArg::Bytes(as_bytes(&gpu))),
        ],
        buffers.grids,
        0,
        metal::MTLSize::new(u64::from(params.hidden / 256), u64::from(tiles), 1),
        metal::MTLSize::new(256, 1, 1),
    );
    if through == MoeTilesFusedStage::Gather {
        return Ok(());
    }
    encoder.memory_barrier();

    // ---- 3: gate and up ----
    let pass = |planes: &PreparedGroupedM8Experts| PassGpuParams {
        k: planes.k,
        n: planes.n,
        column_rows: 0,
        plane1_offset: planes.plane1_offset,
        meta_offset: planes.meta_offset,
        _pad: 0,
        prepared_expert_stride: planes.prepared_expert_stride,
    };
    let gpu = pass(weights.gate);
    let pipeline = registry.get_pipeline(&gate_up_kernel, device.metal_device())?;
    encoder.encode_threadgroups_indirect_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(&weights.gate.bytes)),
            (1, KernelArg::Buffer(&weights.up.bytes)),
            (2, KernelArg::Buffer(&weights.shared_gate.bytes)),
            (3, KernelArg::Buffer(&weights.shared_up.bytes)),
            (4, KernelArg::Buffer(buffers.gathered)),
            (5, KernelArg::Buffer(buffers.intermediate)),
            (6, KernelArg::Buffer(buffers.schedule)),
            (7, KernelArg::Bytes(as_bytes(&gpu))),
        ],
        buffers.grids,
        grid_bytes,
        metal::MTLSize::new(u64::from(params.intermediate / 32), u64::from(tiles), 1),
        metal::MTLSize::new(64, 1, 1),
    );
    if through == MoeTilesFusedStage::GateUp {
        return Ok(());
    }
    encoder.memory_barrier();

    // ---- 4: down ----
    let gpu = pass(weights.down);
    let pipeline = registry.get_pipeline(&down_kernel, device.metal_device())?;
    encoder.encode_threadgroups_indirect_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(&weights.down.bytes)),
            (1, KernelArg::Buffer(&weights.shared_down.bytes)),
            (2, KernelArg::Buffer(buffers.intermediate)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Buffer(buffers.schedule)),
            (5, KernelArg::Buffer(buffers.routes)),
            (6, KernelArg::Bytes(as_bytes(&gpu))),
        ],
        buffers.grids,
        2 * grid_bytes,
        metal::MTLSize::new(u64::from(params.hidden / 64), u64::from(tiles), 1),
        metal::MTLSize::new(64, 1, 1),
    );
    Ok(())
}
