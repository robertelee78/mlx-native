//! MoE expert live tiles over prepared Q4_K/Q6_K planes
//! (`shaders/moe_expert_tiles.metal`).
//!
//! One layer's routed experts run as four stages over one compact route
//! schedule:
//!
//! 1. [`moe_tiles_schedule`] builds the schedule from the route table;
//! 2. [`moe_tiles_gather`] copies each scheduled route's input row into the
//!    schedule's grouped rows as BF16 and writes the live-tile grids of the
//!    two passes;
//! 3. [`moe_tiles_gate_up`] runs gate and up in one pass and leaves
//!    `BF16(SiLU(gate) * up)` in the grouped rows;
//! 4. [`moe_tiles_down`] runs down over those rows and writes each route's
//!    FP32 output to its own row (`token * top_k + slot`).
//!
//! The schedule ends with its own barrier; the caller orders each later
//! stage after the one before it, and may encode independent work between.
//!
//! Both passes launch the live tiles alone through an indirect grid. A route
//! the table masks (id `n_experts`, [`MoeTilesParams::masked_routes`]) joins
//! no tile: its input row is not read and its output row is not written.
//!
//! A shared expert runs through the same stages as its own small step
//! ([`MoeTilesParams::shared_planes`]): a route table of one route per row
//! whose id is the row's block of eight, so every block is one tile over the
//! shared expert's single set of planes.
//!
//! A route's result depends only on its input row and its expert's weights,
//! not on the rows or routes that share the step; see the shader header and
//! [`moe_tiles_column_tile_exact`].

use super::{
    as_bytes, checked_byte_extent, compact_max_tiles, encode_compact_map0,
    prepare_quantized_matmul_id_ggml_grouped_m8, CommandEncoder, DType, ExpertMmSchedule,
    GgmlIdMmDispatchParams, GgmlQuantizedMatmulIdParams, GgmlType, IdMmScratch, KernelArg,
    KernelRegistry, MlxBuffer, MlxDevice, MlxError, PreparedGroupedM8Experts, Result,
};

/// Rows past an expert's last route that its 32-row matmul tile reads (and
/// discards): the grouped buffers hold this many rows beyond the route list.
const ROW_SLACK: usize = 31;

/// Routes of an expert that the register path runs when it is enabled.
const COLUMN_ROWS: u32 = 2;

/// Words of the two indirect grids [`moe_tiles_gather`] writes.
pub const MOE_TILES_GRID_WORDS: usize = 6;

/// Rows the grouped buffers of [`MoeTilesBuffers`] hold for `n_tokens` rows of
/// `top_k` routes over `n_experts` experts: the route list (one guard entry
/// per expert) and the slack of the widest matmul tile.
pub fn moe_tiles_grouped_rows(n_tokens: u32, top_k: u32, n_experts: u32) -> Result<usize> {
    (n_tokens as usize)
        .checked_mul(top_k as usize)
        .and_then(|routes| routes.checked_add(n_experts as usize))
        .and_then(|entries| entries.checked_add(ROW_SLACK))
        .ok_or_else(|| MlxError::InvalidArgument("MoE tiles grouped rows overflow".into()))
}

/// One step's shape and route-table semantics.
#[derive(Debug, Clone, Copy)]
pub struct MoeTilesParams {
    pub n_tokens: u32,
    pub top_k: u32,
    pub n_experts: u32,
    /// Input width of gate/up and output width of down.
    pub hidden: u32,
    /// Output width of gate/up and input width of down.
    pub intermediate: u32,
    /// An id equal to `n_experts` marks a masked route instead of poisoning
    /// the schedule.
    pub masked_routes: bool,
    /// Run experts of one or two routes in registers. Pass
    /// [`moe_tiles_column_tile_exact`]: the register path is bitwise the
    /// matmul only on devices where that probe holds.
    pub column_tile: bool,
    /// Every scheduled id reads the planes of one expert (prepared with
    /// `n_experts = 1`): the ids only group rows into tiles.
    pub shared_planes: bool,
}

/// Caller-owned scratch of one step. `gathered` is BF16
/// `[moe_tiles_grouped_rows, hidden]`, `intermediate` BF16
/// `[moe_tiles_grouped_rows, intermediate]` and `grids` U32
/// `[MOE_TILES_GRID_WORDS]`. The contents need no initialization.
#[derive(Clone, Copy)]
pub struct MoeTilesBuffers<'a> {
    pub gathered: &'a MlxBuffer,
    pub intermediate: &'a MlxBuffer,
    pub grids: &'a MlxBuffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherGpuParams {
    k: u32,
    top_k: u32,
    gate_up_columns: u32,
    down_columns: u32,
}

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

fn invalid(message: String) -> MlxError {
    MlxError::InvalidArgument(message)
}

fn check_shape(params: &MoeTilesParams) -> Result<usize> {
    if params.shared_planes && (params.top_k != 1 || params.masked_routes) {
        return Err(invalid(
            "MoE tiles over shared planes take one unmasked route per row".into(),
        ));
    }
    if !matches!(params.top_k, 1 | 6 | 8) {
        return Err(invalid(format!(
            "MoE tiles have no route schedule for top_k={}",
            params.top_k
        )));
    }
    if params.n_tokens == 0
        || params.top_k > params.n_experts
        || params.hidden == 0
        || params.hidden % 256 != 0
        || params.intermediate == 0
        || params.intermediate % 256 != 0
    {
        return Err(invalid(format!(
            "MoE tiles require n_tokens > 0, top_k <= n_experts and widths that are multiples of 256; got {params:?}"
        )));
    }
    moe_tiles_grouped_rows(params.n_tokens, params.top_k, params.n_experts)
}

fn check_buffers(buffers: &MoeTilesBuffers<'_>, params: &MoeTilesParams) -> Result<()> {
    let rows = check_shape(params)?;
    let bf16 = DType::BF16.size_of();
    let gathered = checked_byte_extent(
        "MoE tiles gathered rows",
        &[rows, params.hidden as usize, bf16],
    )?;
    let intermediate = checked_byte_extent(
        "MoE tiles intermediate rows",
        &[rows, params.intermediate as usize, bf16],
    )?;
    let grids = MOE_TILES_GRID_WORDS * DType::U32.size_of();
    if buffers.gathered.dtype() != DType::BF16
        || buffers.intermediate.dtype() != DType::BF16
        || buffers.grids.dtype() != DType::U32
        || buffers.gathered.data_byte_len() < gathered
        || buffers.intermediate.data_byte_len() < intermediate
        || buffers.grids.data_byte_len() < grids
    {
        return Err(invalid(format!(
            "MoE tiles need BF16 gathered {gathered} bytes, BF16 intermediate {intermediate} bytes and U32 grids {grids} bytes; got {:?} {}, {:?} {}, {:?} {}",
            buffers.gathered.dtype(),
            buffers.gathered.data_byte_len(),
            buffers.intermediate.dtype(),
            buffers.intermediate.data_byte_len(),
            buffers.grids.dtype(),
            buffers.grids.data_byte_len(),
        )));
    }
    Ok(())
}

fn check_prepared(
    label: &str,
    prepared: &PreparedGroupedM8Experts,
    params: &MoeTilesParams,
    n: u32,
    k: u32,
) -> Result<()> {
    let n_experts = if params.shared_planes {
        1
    } else {
        params.n_experts
    };
    if prepared.n_experts != n_experts || prepared.n != n || prepared.k != k {
        return Err(invalid(format!(
            "MoE tiles {label} planes are [{}, {}, {}], the step needs [{n_experts}, {n}, {k}]",
            prepared.n_experts, prepared.n, prepared.k
        )));
    }
    Ok(())
}

fn maximum_tiles(params: &MoeTilesParams) -> u64 {
    compact_max_tiles(
        u64::from(params.n_tokens) * u64::from(params.top_k),
        u64::from(params.n_experts),
    )
}

fn format_name(ggml_type: GgmlType) -> &'static str {
    match ggml_type {
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q6_K => "q6_K",
        _ => unreachable!("prepared expert planes are Q4_K or Q6_K"),
    }
}

fn pass_params(prepared: &PreparedGroupedM8Experts, params: &MoeTilesParams) -> PassGpuParams {
    PassGpuParams {
        k: prepared.k,
        n: prepared.n,
        column_rows: if params.column_tile { COLUMN_ROWS } else { 0 },
        plane1_offset: prepared.plane1_offset,
        meta_offset: prepared.meta_offset,
        _pad: 0,
        prepared_expert_stride: if params.shared_planes {
            0
        } else {
            prepared.prepared_expert_stride
        },
    }
}

/// Stage 1: schedule the routes of `ids` (`[n_tokens * top_k]` U32) into
/// `scratch`. Ends with a barrier.
pub fn moe_tiles_schedule(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    ids: &MlxBuffer,
    scratch: &mut IdMmScratch,
    params: &MoeTilesParams,
) -> Result<()> {
    check_shape(params)?;
    let routes = params.n_tokens as usize * params.top_k as usize;
    let required_ids = checked_byte_extent("MoE tiles ids", &[routes, DType::U32.size_of()])?;
    if ids.dtype() != DType::U32 || ids.data_byte_len() < required_ids {
        return Err(invalid(format!(
            "MoE tiles need U32 ids {required_ids} bytes, got {:?} {}",
            ids.dtype(),
            ids.data_byte_len(),
        )));
    }
    scratch.check_capacity(params.n_experts, params.n_tokens, params.top_k)?;
    scratch.invalidate_schedule();
    encode_compact_map0(
        encoder,
        registry,
        device,
        ids,
        &scratch.htpe,
        &scratch.hids,
        &GgmlIdMmDispatchParams {
            n_tokens: params.n_tokens,
            top_k: params.top_k,
            n: params.intermediate,
            k: params.hidden,
            n_experts: params.n_experts,
            expert_stride: 0,
            ggml_type: GgmlType::Q4_K,
        },
        maximum_tiles(params),
        params.masked_routes,
    )?;
    scratch.encode_serial = scratch.encode_serial.wrapping_add(1);
    scratch.schedule_key = Some(scratch.schedule_key_for(
        encoder,
        ids,
        params.n_tokens,
        params.top_k,
        params.n_experts,
        ExpertMmSchedule::CompactTiles,
    ));
    Ok(())
}

/// Stage 2: gather the scheduled rows of `input` (`[n_tokens, hidden]` F32)
/// into `buffers.gathered` and write the pass grids, for the schedule
/// [`moe_tiles_schedule`] encoded for `ids` in this command. Encodes no
/// barrier: order the passes after it with one.
#[allow(clippy::too_many_arguments)]
pub fn moe_tiles_gather(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    ids: &MlxBuffer,
    buffers: &MoeTilesBuffers<'_>,
    scratch: &IdMmScratch,
    params: &MoeTilesParams,
) -> Result<()> {
    check_buffers(buffers, params)?;
    let required_input = checked_byte_extent(
        "MoE tiles input",
        &[
            params.n_tokens as usize,
            params.hidden as usize,
            DType::F32.size_of(),
        ],
    )?;
    if input.dtype() != DType::F32 || input.data_byte_len() < required_input {
        return Err(invalid(format!(
            "MoE tiles need F32 input {required_input} bytes, got {:?} {}",
            input.dtype(),
            input.data_byte_len(),
        )));
    }
    scratch.validate_schedule_reuse(
        encoder,
        ids,
        params.n_tokens,
        params.top_k,
        params.n_experts,
        ExpertMmSchedule::CompactTiles,
    )?;
    let gpu = GatherGpuParams {
        k: params.hidden,
        top_k: params.top_k,
        gate_up_columns: params.intermediate / 32,
        down_columns: params.hidden / 64,
    };
    let pipeline = registry.get_pipeline("moe_tiles_gather_f32", device.metal_device())?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(buffers.gathered)),
            (2, KernelArg::Buffer(&scratch.htpe)),
            (3, KernelArg::Buffer(&scratch.hids)),
            (4, KernelArg::Buffer(buffers.grids)),
            (5, KernelArg::Bytes(as_bytes(&gpu))),
        ],
        metal::MTLSize::new(
            u64::from(params.hidden / 256),
            maximum_tiles(params).max(1),
            1,
        ),
        metal::MTLSize::new(256, 1, 1),
    );
    Ok(())
}

/// Stage 3: gate and up over the gathered rows, into `buffers.intermediate`.
/// Encodes no barrier: order [`moe_tiles_down`] after it with one.
#[allow(clippy::too_many_arguments)]
pub fn moe_tiles_gate_up(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate: &PreparedGroupedM8Experts,
    up: &PreparedGroupedM8Experts,
    ids: &MlxBuffer,
    buffers: &MoeTilesBuffers<'_>,
    scratch: &IdMmScratch,
    params: &MoeTilesParams,
) -> Result<()> {
    check_buffers(buffers, params)?;
    check_prepared("gate", gate, params, params.intermediate, params.hidden)?;
    check_prepared("up", up, params, params.intermediate, params.hidden)?;
    if gate.ggml_type != up.ggml_type {
        return Err(invalid(format!(
            "MoE tiles gate/up pass needs one format, got {:?} and {:?}",
            gate.ggml_type, up.ggml_type
        )));
    }
    scratch.validate_schedule_reuse(
        encoder,
        ids,
        params.n_tokens,
        params.top_k,
        params.n_experts,
        ExpertMmSchedule::CompactTiles,
    )?;
    let gpu = pass_params(gate, params);
    let kernel = format!("moe_tiles_gate_up_{}", format_name(gate.ggml_type));
    let pipeline = registry.get_pipeline(&kernel, device.metal_device())?;
    encoder.encode_threadgroups_indirect_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(&gate.bytes)),
            (1, KernelArg::Buffer(&up.bytes)),
            (2, KernelArg::Buffer(buffers.gathered)),
            (3, KernelArg::Buffer(buffers.intermediate)),
            (4, KernelArg::Buffer(&scratch.htpe)),
            (5, KernelArg::Bytes(as_bytes(&gpu))),
        ],
        buffers.grids,
        0,
        metal::MTLSize::new(
            u64::from(params.intermediate / 32),
            maximum_tiles(params).max(1),
            1,
        ),
        metal::MTLSize::new(64, 1, 1),
    );
    Ok(())
}

/// Stage 4: down of the same schedule over `buffers.intermediate`, into
/// `output` (`[n_tokens * top_k, hidden]` F32, a route's row at its slot).
/// Masked routes' rows are left unwritten. Encodes no barrier.
#[allow(clippy::too_many_arguments)]
pub fn moe_tiles_down(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    down: &PreparedGroupedM8Experts,
    ids: &MlxBuffer,
    buffers: &MoeTilesBuffers<'_>,
    scratch: &IdMmScratch,
    output: &MlxBuffer,
    params: &MoeTilesParams,
) -> Result<()> {
    check_buffers(buffers, params)?;
    check_prepared("down", down, params, params.hidden, params.intermediate)?;
    let routes = params.n_tokens as usize * params.top_k as usize;
    let required_output = checked_byte_extent(
        "MoE tiles output",
        &[routes, params.hidden as usize, DType::F32.size_of()],
    )?;
    if output.dtype() != DType::F32 || output.data_byte_len() < required_output {
        return Err(invalid(format!(
            "MoE tiles need F32 output {required_output} bytes, got {:?} {}",
            output.dtype(),
            output.data_byte_len()
        )));
    }
    scratch.validate_schedule_reuse(
        encoder,
        ids,
        params.n_tokens,
        params.top_k,
        params.n_experts,
        ExpertMmSchedule::CompactTiles,
    )?;
    let gpu = pass_params(down, params);
    let kernel = format!("moe_tiles_down_{}", format_name(down.ggml_type));
    let pipeline = registry.get_pipeline(&kernel, device.metal_device())?;
    encoder.encode_threadgroups_indirect_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(&down.bytes)),
            (1, KernelArg::Buffer(buffers.intermediate)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Buffer(&scratch.htpe)),
            (4, KernelArg::Buffer(&scratch.hids)),
            (5, KernelArg::Bytes(as_bytes(&gpu))),
        ],
        buffers.grids,
        (3 * DType::U32.size_of()) as u64,
        metal::MTLSize::new(
            u64::from(params.hidden / 64),
            maximum_tiles(params).max(1),
            1,
        ),
        metal::MTLSize::new(64, 1, 1),
    );
    Ok(())
}

/// Whether the register path of one- and two-route experts is bitwise the
/// matmul tile on `device`: runs both passes of a small Q4_K and Q6_K layer
/// with and without the register path and compares every output bit. The
/// register path spells out the order in which this device's `matmul2d`
/// adds a group's products; a device or OS whose order differs returns
/// `false` and its steps must keep [`MoeTilesParams::column_tile`] off.
///
/// Blocks on four small command buffers the first time it is called for a
/// device; the answer is cached per device.
pub fn moe_tiles_column_tile_exact(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> Result<bool> {
    static CACHE: std::sync::Mutex<Vec<(u64, bool)>> = std::sync::Mutex::new(Vec::new());
    let registry_id = device.registry_id();
    let cached = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .find(|(id, _)| *id == registry_id)
        .map(|(_, exact)| *exact);
    if let Some(exact) = cached {
        return Ok(exact);
    }
    let exact = probe_column_tile(device, registry, GgmlType::Q4_K)?
        && probe_column_tile(device, registry, GgmlType::Q6_K)?;
    CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push((registry_id, exact));
    Ok(exact)
}

/// SplitMix64, the probe's fixed data.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn probe_column_tile(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    ggml_type: GgmlType,
) -> Result<bool> {
    // Three rows, one route each: expert 0 takes one, expert 1 two.
    const TOKENS: u32 = 3;
    const EXPERTS: u32 = 2;
    const WIDTH: u32 = 256;
    let mut state = 0x6d6f_655f_7469_6c65 ^ u64::from(ggml_type.block_bytes());
    let block_bytes = ggml_type.block_bytes() as usize;
    let stride =
        u64::from(WIDTH) * u64::from(WIDTH / ggml_type.block_values()) * block_bytes as u64;
    let weights = |state: &mut u64| -> Result<MlxBuffer> {
        let bytes = (stride * u64::from(EXPERTS)) as usize;
        let mut buffer = device.alloc_buffer(bytes, DType::U8, vec![bytes])?;
        let data = buffer.as_mut_slice::<u8>()?;
        for chunk in data.chunks_mut(8) {
            chunk.copy_from_slice(&mix(state).to_le_bytes()[..chunk.len()]);
        }
        // Finite half block scales of mixed magnitude.
        let scale_offsets: &[usize] = if ggml_type == GgmlType::Q4_K {
            &[0, 2]
        } else {
            &[208]
        };
        for block in data.chunks_exact_mut(block_bytes) {
            for &offset in scale_offsets {
                let word = mix(state);
                let bits = ((word & 0x8000)
                    | (((5 + (word >> 16) % 6) << 10) & 0x7c00)
                    | ((word >> 32) & 0x03ff)) as u16;
                block[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        Ok(buffer)
    };
    let shape = GgmlQuantizedMatmulIdParams {
        n_tokens: TOKENS,
        top_k: 1,
        n: WIDTH,
        k: WIDTH,
        n_experts: EXPERTS,
        expert_stride: stride,
        ggml_type,
    };
    let sources = [
        weights(&mut state)?,
        weights(&mut state)?,
        weights(&mut state)?,
    ];
    let mut input = device.alloc_buffer(
        (TOKENS * WIDTH) as usize * 4,
        DType::F32,
        vec![TOKENS as usize, WIDTH as usize],
    )?;
    for value in input.as_mut_slice::<f32>()? {
        let word = mix(&mut state);
        let unit = ((word >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        *value = unit * 2.0f32.powi((word % 13) as i32 - 6);
    }
    let mut ids = device.alloc_buffer(TOKENS as usize * 4, DType::U32, vec![TOKENS as usize])?;
    ids.as_mut_slice::<u32>()?.copy_from_slice(&[0, 1, 1]);
    let rows = moe_tiles_grouped_rows(TOKENS, 1, EXPERTS)?;
    let grouped = |device: &MlxDevice| {
        device.alloc_buffer(
            rows * WIDTH as usize * 2,
            DType::BF16,
            vec![rows, WIDTH as usize],
        )
    };
    let gathered = grouped(device)?;
    let grids = device.alloc_buffer(
        MOE_TILES_GRID_WORDS * 4,
        DType::U32,
        vec![MOE_TILES_GRID_WORDS],
    )?;
    let mut scratch = IdMmScratch::alloc_with_top_k(device, EXPERTS, TOKENS, 1)?;

    let mut encoder = device.command_encoder()?;
    let mut prepare = |source: &MlxBuffer| {
        prepare_quantized_matmul_id_ggml_grouped_m8(&mut encoder, registry, device, source, &shape)
    };
    let prepared = [
        prepare(&sources[0])?,
        prepare(&sources[1])?,
        prepare(&sources[2])?,
    ];
    encoder.commit_and_wait()?;

    let mut run = |column_tile: bool| -> Result<(Vec<u16>, Vec<u32>)> {
        let intermediate = grouped(device)?;
        let output = device.alloc_buffer(
            (TOKENS * WIDTH) as usize * 4,
            DType::F32,
            vec![TOKENS as usize, WIDTH as usize],
        )?;
        let buffers = MoeTilesBuffers {
            gathered: &gathered,
            intermediate: &intermediate,
            grids: &grids,
        };
        let params = MoeTilesParams {
            n_tokens: TOKENS,
            top_k: 1,
            n_experts: EXPERTS,
            hidden: WIDTH,
            intermediate: WIDTH,
            masked_routes: false,
            column_tile,
            shared_planes: false,
        };
        let mut encoder = device.command_encoder()?;
        moe_tiles_schedule(&mut encoder, registry, device, &ids, &mut scratch, &params)?;
        moe_tiles_gather(
            &mut encoder,
            registry,
            device,
            &input,
            &ids,
            &buffers,
            &scratch,
            &params,
        )?;
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
        )?;
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
        )?;
        encoder.commit_and_wait()?;
        // Routes sit at list entries 0 (expert 0) and 2, 3 (expert 1).
        let grouped_bits = intermediate.as_slice::<u16>()?;
        let mut intermediate_bits = Vec::with_capacity((TOKENS * WIDTH) as usize);
        for entry in [0usize, 2, 3] {
            intermediate_bits.extend_from_slice(
                &grouped_bits[entry * WIDTH as usize..(entry + 1) * WIDTH as usize],
            );
        }
        let output_bits = output
            .as_slice::<f32>()?
            .iter()
            .map(|value| value.to_bits())
            .collect();
        Ok((intermediate_bits, output_bits))
    };
    let register = run(true)?;
    let matmul = run(false)?;
    let finite = matmul
        .1
        .iter()
        .all(|bits| f32::from_bits(*bits).is_finite());
    Ok(finite && register == matmul)
}
