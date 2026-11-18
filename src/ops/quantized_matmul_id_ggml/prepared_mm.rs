//! Unpublished full-MM experiment: prepared addresses, original F16 reductions.

use super::{
    as_bytes, checked_byte_extent, compact_max_tiles, encode_compact_map0,
    encode_legacy_map0, has_mm_id_map0, id_mm_scratch_bytes, plan_expert_auto_route,
    probe_tensor_mm_id, required_expert_weight_bytes, tensor_half_tile_skip_validated,
    tensor_mm_auto_selected, CommandEncoder, DType, ExpertAutoPlan, ExpertMmSchedule,
    GgmlIdMmDispatchParams, GgmlIdMmMmGpuParams, GgmlQuantizedMatmulIdParams,
    GgmlRoutingPolicy, GgmlType, HalfTileSkip, IdMmScheduleOptions, IdMmScratch,
    KernelArg, KernelRegistry, MlxBuffer, MlxDevice, MlxError, GroupedM8ExpertPlanes,
    Result, HALF_TILE_SKIP_SIMDGROUP, SCHED_FLAG_HALF_TILE_SKIP,
};
use crate::device::metal_device_registry_id;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PreparedGpuParams {
    n_experts: u32,
    n: u32,
    blocks: u32,
    format: u32,
    raw_expert_stride: u64,
    prepared_expert_stride: u64,
    plane1_offset: u32,
    meta_offset: u32,
}

type Range = (usize, usize);

fn invalid(message: &str) -> MlxError {
    MlxError::InvalidArgument(format!("prepared mm_id experiment: {message}"))
}

fn buffer_range(
    buffer: &MlxBuffer,
    dtype: DType,
    required: usize,
    alignment: usize,
    writable: bool,
    device_id: u64,
) -> Result<Range> {
    if buffer.dtype() != dtype || buffer.data_byte_len() < required {
        return Err(invalid("buffer dtype or logical extent does not match the dispatch"));
    }
    if writable && (!buffer.is_cpu_writable() || buffer.is_file_backed()) {
        return Err(invalid("output and scratch must be writable, non-file-backed allocations"));
    }
    if metal_device_registry_id(buffer.metal_buffer().device()) != device_id {
        return Err(invalid("all buffers must belong to the encoder's device"));
    }
    let offset = usize::try_from(buffer.byte_offset())
        .map_err(|_| invalid("buffer offset exceeds usize"))?;
    let physical_end = offset.checked_add(buffer.data_byte_len())
        .ok_or_else(|| invalid("physical buffer extent overflow"))?;
    if physical_end > buffer.byte_len() {
        return Err(invalid("logical buffer window exceeds its allocation"));
    }
    let base = buffer.contents_ptr() as usize;
    if base == 0 {
        return Err(invalid("buffers must have CPU-addressable storage"));
    }
    let start = base.checked_add(offset).ok_or_else(|| invalid("buffer address overflow"))?;
    if start % alignment != 0 || offset % alignment != 0 {
        return Err(invalid("buffer window is not aligned for its shader accesses"));
    }
    let end = start.checked_add(buffer.data_byte_len())
        .ok_or_else(|| invalid("logical buffer address overflow"))?;
    Ok((start, end))
}

fn overlaps(a: Range, b: Range) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Native-only candidate for complete classic/tensor Q4_K/Q6_K MM reductions.
///
/// Output is F32 `[n_tokens, top_k, N]`. Input, schedule reuse and half-tile
/// options have the raw test dispatch's semantics. Only naturally selected MM
/// geometries are accepted. This does not change routing or prepared ownership.
/// No GPU storage is allocated and no raw weights are bound or reconstructed.
/// Validation and pipeline lookup finish before any command is encoded. As for
/// the raw API, callers must order producers across encoders and invalidate the
/// scratch schedule before mutating IDs in place.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_prepared_id_mm_pooled_with_options_for_test(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &GroupedM8ExpertPlanes,
    ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
    slotted: bool,
    reuse_schedule: bool,
    routing: &GgmlRoutingPolicy,
    options: IdMmScheduleOptions,
) -> Result<()> {
    // Preserve the raw pooled helper's early-error schedule ownership rules.
    super::validate_native_expert_dtypes(
        "dispatch_prepared_id_mm_pooled_with_options_for_test",
        input, &prepared.bytes, ids, output,
    )?;
    if plan_expert_auto_route(params.n_tokens, params.top_k, params.k, false, routing)
        != ExpertAutoPlan::Mm
    {
        return Err(invalid("requires the naturally selected mm_id route"));
    }
    if reuse_schedule {
        scratch.validate_schedule_reuse(encoder, ids, params.n_tokens, params.top_k,
            params.n_experts, options.schedule)?;
    }
    scratch.check_capacity(params.n_experts, params.n_tokens, params.top_k)?;
    let result = dispatch(
        encoder, registry, device, input, prepared, ids, output, scratch,
        params, slotted, reuse_schedule, routing, options,
    );
    match &result {
        Ok(()) if !reuse_schedule => {
            scratch.encode_serial = scratch.encode_serial.wrapping_add(1);
            scratch.schedule_key = Some(scratch.schedule_key_for(
                encoder, ids, params.n_tokens, params.top_k, params.n_experts,
                options.schedule,
            ));
        }
        Ok(()) => {}
        Err(_) => scratch.invalidate_schedule(),
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    prepared: &GroupedM8ExpertPlanes,
    ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: &IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
    slotted: bool,
    reuse_schedule: bool,
    routing: &GgmlRoutingPolicy,
    options: IdMmScheduleOptions,
) -> Result<()> {
    let format = match params.ggml_type {
        GgmlType::Q4_K => 0,
        GgmlType::Q6_K => 1,
        _ => return Err(invalid("only Q4_K and Q6_K are supported")),
    };
    if params.n_experts == 0 || params.n == 0 || params.n % 64 != 0
        || params.k == 0 || params.k % 256 != 0 || params.n_tokens == 0
        || params.top_k == 0 || params.top_k > params.n_experts
        || !has_mm_id_map0(params.top_k)
    {
        return Err(invalid("requires nonzero dimensions, N % 64 == 0, K % 256 == 0 and top_k in {1,6,8} <= experts"));
    }
    for value in [params.n_tokens, params.top_k, params.n, params.k, params.n_experts] {
        if value > i32::MAX as u32 {
            return Err(invalid("dimension exceeds the signed Metal ABI"));
        }
    }
    // The unchanged map loops increment an int by up to 1024 threads. Leave
    // room for their final increment and inactive lanes in that final group.
    if params.n_tokens > (i32::MAX as u32).saturating_sub(1024) {
        return Err(invalid("token count exceeds safe map-loop indexing"));
    }
    if prepared.n_experts != params.n_experts || prepared.n != params.n
        || prepared.k != params.k || prepared.ggml_type != params.ggml_type
    {
        return Err(invalid("params do not match the opaque prepared geometry/type"));
    }
    // Validate the descriptive raw stride without touching the retained owner.
    required_expert_weight_bytes(params.ggml_type, params.n_experts, params.n,
        params.k, params.expert_stride)?;
    if params.expert_stride % 2 != 0 {
        return Err(invalid("expert stride must be half-aligned"));
    }
    let compact = options.schedule == ExpertMmSchedule::CompactTiles;
    let rows = checked_byte_extent("prepared MM rows", &[params.n_tokens as usize, params.top_k as usize])?;
    let list_bound = if compact {
        rows.checked_add(params.n_experts as usize)
    } else {
        (params.n_tokens as usize).checked_mul(params.n_experts as usize)
    }.ok_or_else(|| invalid("schedule list indexing overflow"))?;
    if rows > i32::MAX as usize || list_bound > i32::MAX as usize {
        return Err(invalid("schedule list exceeds signed Metal indexing"));
    }
    let max_tiles = compact_max_tiles(rows as u64, u64::from(params.n_experts));
    if compact && (max_tiles >= 0x8000_0000 || max_tiles > (u32::MAX as u64 - 1) / 3) {
        return Err(invalid("compact descriptor indexing overflow"));
    }
    let experts = params.n_experts as usize;
    let n = params.n as usize;
    let blocks = (params.k / 256) as usize;
    let plane0 = checked_byte_extent("prepared MM low plane", &[n, blocks, 8, 16])?;
    let plane1 = if format == 1 {
        checked_byte_extent("prepared MM high plane", &[n, blocks, 8, 8])?
    } else { 0 };
    let meta_offset = plane0.checked_add(plane1).ok_or_else(|| invalid("metadata offset overflow"))?;
    let metadata = checked_byte_extent("prepared MM metadata", &[n, blocks, if format == 1 { 20 } else { 16 }])?;
    let stride = meta_offset.checked_add(metadata).ok_or_else(|| invalid("prepared stride overflow"))?;
    if u32::try_from(plane0).ok() != Some(prepared.plane1_offset)
        || u32::try_from(meta_offset).ok() != Some(prepared.meta_offset)
        || u64::try_from(stride).ok() != Some(prepared.prepared_expert_stride)
    {
        return Err(invalid("opaque prepared plane offsets/stride do not match its geometry"));
    }
    let prepared_bytes = checked_byte_extent("prepared MM weights", &[experts, stride])?;
    let input_bytes = checked_byte_extent("prepared MM input", &[
        if slotted { rows } else { params.n_tokens as usize }, params.k as usize, 4,
    ])?;
    let ids_bytes = checked_byte_extent("prepared MM IDs", &[rows, 4])?;
    let output_bytes = checked_byte_extent("prepared MM output", &[rows, n, 4])?;
    let (htpe_bytes, hids_bytes) = id_mm_scratch_bytes(options.schedule,
        params.n_experts, params.n_tokens, params.top_k)?;
    let device_id = encoder.device_registry_id();
    if device_id != device.registry_id() {
        return Err(invalid("encoder and supplied device differ"));
    }
    let prepared_range = buffer_range(&prepared.bytes, DType::U8, prepared_bytes, 4, false, device_id)?;
    let input_range = buffer_range(input, DType::F32, input_bytes, 16, false, device_id)?;
    let ids_range = buffer_range(ids, DType::U32, ids_bytes, 4, false, device_id)?;
    let output_range = buffer_range(output, DType::F32, output_bytes, 16, true, device_id)?;
    let htpe_range = buffer_range(&scratch.htpe, DType::U32, htpe_bytes, 4, true, device_id)?;
    let hids_range = buffer_range(&scratch.hids, DType::U32, hids_bytes, 4, true, device_id)?;
    let read_ranges = [prepared_range, input_range, ids_range];
    let write_ranges = [output_range, htpe_range, hids_range];
    for (index, &write) in write_ranges.iter().enumerate() {
        if read_ranges.iter().any(|&read| overlaps(write, read))
            || write_ranges[..index].iter().any(|&other| overlaps(write, other))
        {
            return Err(invalid("output/scratch must not overlap each other or any input window"));
        }
    }
    let tensor_path = tensor_mm_auto_selected(params.ggml_type, routing.expert_tensor_mm,
        routing.q5_0_tensor_mm) && probe_tensor_mm_id(registry, device)?;
    let kernel_name = match (format, tensor_path, compact) {
        (0, false, false) => "kernel_mul_mm_id_prepared_q4_K_f32",
        (0, false, true) => "kernel_mul_mm_id_prepared_q4_K_f32_c",
        (1, false, false) => "kernel_mul_mm_id_prepared_q6_K_f32",
        (1, false, true) => "kernel_mul_mm_id_prepared_q6_K_f32_c",
        (0, true, false) => "kernel_mul_mm_id_prepared_q4_K_tensor_f32",
        (0, true, true) => "kernel_mul_mm_id_prepared_q4_K_tensor_f32_c",
        (1, true, false) => "kernel_mul_mm_id_prepared_q6_K_tensor_f32",
        (1, true, true) => "kernel_mul_mm_id_prepared_q6_K_tensor_f32_c",
        _ => unreachable!("format validated above"),
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?.to_owned();
    let memory_limit = device.metal_device().max_threadgroup_memory_length();
    if pipeline.max_total_threads_per_threadgroup() < 128
        || pipeline.thread_execution_width() != 32
        || pipeline.static_threadgroup_memory_length().checked_add(8192)
            .filter(|&bytes| bytes <= memory_limit).is_none()
    {
        return Err(invalid("MM pipeline cannot support the original thread/memory geometry"));
    }
    // The existing map helper owns map lookup, validation and encoding. Check
    // the uniform helper's dynamic-memory limit here (compact checks its own).
    if !reuse_schedule && !compact {
        let map_name = match params.top_k {
            1 => "kernel_mul_mm_id_map0_ne20_1",
            6 => "kernel_mul_mm_id_map0_ne20_6",
            8 => "kernel_mul_mm_id_map0_ne20_8",
            _ => unreachable!("top_k validated above"),
        };
        let map = registry.get_pipeline(map_name, device.metal_device())?;
        let shared = (u64::from(params.n_experts) * u64::from(params.top_k) * 4).div_ceil(16) * 16;
        if params.n_experts > i16::MAX as u32
            || u64::from(params.n_experts) > map.max_total_threads_per_threadgroup()
            || map.static_threadgroup_memory_length().checked_add(shared)
                .filter(|&bytes| bytes <= memory_limit).is_none()
        {
            return Err(invalid("uniform map cannot support the original thread/memory geometry"));
        }
    }
    let half_skip = compact && match options.half_tile_skip {
        HalfTileSkip::Production => if tensor_path {
            tensor_half_tile_skip_validated(device)
        } else { HALF_TILE_SKIP_SIMDGROUP },
        HalfTileSkip::Disabled => false,
        HalfTileSkip::Forced => true,
    };
    let row_bytes = u64::from(params.k) * 4;
    let mm = GgmlIdMmMmGpuParams {
        ne00: params.k as i32, ne02: params.n_experts as i32,
        nb01: blocks as u64 * u64::from(params.ggml_type.block_bytes()),
        nb02: params.expert_stride, nb03: 0, ne11: params.top_k as i32, _pad0: 0,
        nb10: 4, nb11: if slotted { row_bytes } else { 0 },
        nb12: if slotted { row_bytes * u64::from(params.top_k) } else { row_bytes },
        nb13: 0, ne20: params.top_k as i32, ne21: params.n_tokens as i32,
        ne0: params.n as i32, ne1: params.top_k as i32, r2: 1, r3: 1,
        sched_flags: if half_skip { SCHED_FLAG_HALF_TILE_SKIP } else { 0 }, _pad1: 0,
    };
    let planes = PreparedGpuParams {
        n_experts: params.n_experts, n: params.n, blocks: params.k / 256, format,
        raw_expert_stride: params.expert_stride,
        prepared_expert_stride: prepared.prepared_expert_stride,
        plane1_offset: prepared.plane1_offset, meta_offset: prepared.meta_offset,
    };
    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: params.n_tokens, top_k: params.top_k, n: params.n, k: params.k,
        n_experts: params.n_experts, expert_stride: params.expert_stride,
        ggml_type: params.ggml_type,
    };
    let grid = if compact {
        metal::MTLSize::new(max_tiles, u64::from(params.n / 64), 1)
    } else {
        metal::MTLSize::new(u64::from(params.n_tokens).div_ceil(32),
            u64::from(params.n / 64), u64::from(params.n_experts))
    };
    // Last fallible operation: these unchanged helpers validate and look up
    // their pipeline before encoding their first command.
    if !reuse_schedule {
        if compact {
            encode_compact_map0(encoder, registry, device, ids, &scratch.htpe,
                &scratch.hids, &dispatch, max_tiles, false)?;
        } else {
            encode_legacy_map0(encoder, registry, device, ids, &scratch.htpe,
                &scratch.hids, &dispatch)?;
        }
    }
    if encoder.is_capturing() {
        // Exclude the Q6 metadata padding, just as the fragment proof does.
        let mut reads = Vec::with_capacity(experts * (if format == 0 { 2 } else { 1 + n * blocks }) + 3);
        for expert in 0..experts {
            let start = prepared_range.0 + expert * stride;
            reads.push((start, start + meta_offset));
            let meta = start + meta_offset;
            if format == 0 {
                reads.push((meta, meta + metadata));
            } else {
                for block in 0..n * blocks {
                    reads.push((meta + block * 20, meta + block * 20 + 18));
                }
            }
        }
        reads.extend([
            (input_range.0, input_range.0 + input_bytes),
            (htpe_range.0, htpe_range.0 + htpe_bytes),
            (hids_range.0, hids_range.0 + hids_bytes),
        ]);
        encoder.set_pending_buffer_ranges(reads,
            vec![(output_range.0, output_range.0 + output_bytes)]);
    }
    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&mm))),
            (1, KernelArg::Buffer(&prepared.bytes)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(&scratch.htpe)),
            (4, KernelArg::Buffer(&scratch.hids)),
            (5, KernelArg::Buffer(output)),
            (6, KernelArg::Bytes(as_bytes(&planes))),
        ],
        &[(0, 8192)], grid, metal::MTLSize::new(128, 1, 1),
    );
    Ok(())
}
