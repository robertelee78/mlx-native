//! Canonical expert storage with the existing legacy arithmetic and routing.
//!
//! Automatic MM dispatch requires caller-owned scratch. No entry point allocates
//! GPU storage, waits, binds a raw expert source, or reconstructs raw blocks.
use super::{
    as_bytes, checked_byte_extent, encode_legacy_map0, ggml_routing_policy_for_registry,
    id_mm_scratch_bytes, mv_id_gpu_params, plan_expert_auto_route, required_expert_weight_bytes,
    CommandEncoder, DType, ExpertAutoPlan, ExpertMmSchedule, GgmlIdMmDispatchParams,
    GgmlIdMmMmGpuParams, GgmlMatvecIdGpuParams, GgmlQuantizedMatmulIdParams,
    GgmlRoutingPolicy, GgmlType, GroupedM8ExpertPlanes, IdMmScheduleOptions,
    IdMmScratch, KernelArg, KernelRegistry, MlxBuffer, MlxDevice, Result,
};
use super::prepared_mm::{buffer_range, invalid, overlaps, PreparedGpuParams};

type BufferRange = (usize, usize);

struct ValidatedCall {
    planes: PreparedGpuParams,
    weights: BufferRange,
    input: BufferRange,
    ids: BufferRange,
    output: BufferRange,
    input_bytes: usize,
    ids_bytes: usize,
    output_bytes: usize,
}

fn validate(
    encoder: &CommandEncoder, device: &MlxDevice, input: &MlxBuffer,
    prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer, output: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams, output_cols: u32, slotted: bool,
) -> Result<ValidatedCall> {
    let format = match params.ggml_type {
        GgmlType::Q4_K => 0,
        GgmlType::Q6_K => 1,
        _ => return Err(invalid("legacy prepared consumers require Q4_K or Q6_K")),
    };
    if params.n == 0 || params.n % 64 != 0 || params.k == 0 || params.k % 256 != 0
        || params.n_tokens == 0 || params.n_experts == 0 || params.top_k == 0
        || params.top_k > params.n_experts || output_cols == 0 || output_cols > params.n
    {
        return Err(invalid("invalid canonical expert geometry or routed row count"));
    }
    if [params.n, params.k, params.n_tokens, params.n_experts, params.top_k]
        .iter().any(|&x| x > i32::MAX as u32)
    {
        return Err(invalid("dimension exceeds signed Metal indexing"));
    }
    if prepared.n != params.n || prepared.k != params.k
        || prepared.n_experts != params.n_experts || prepared.ggml_type != params.ggml_type
    {
        return Err(invalid("params do not match canonical expert geometry/type"));
    }
    required_expert_weight_bytes(params.ggml_type, params.n_experts, params.n,
        params.k, params.expert_stride)?;
    if params.expert_stride % 2 != 0 || params.expert_stride > i64::MAX as u64 {
        return Err(invalid("descriptive raw expert stride is not half-aligned or exceeds ABI"));
    }
    let n = params.n as usize;
    let blocks = (params.k / 256) as usize;
    let low = checked_byte_extent("canonical low plane", &[n, blocks, 8, 16])?;
    let high = if format == 1 {
        checked_byte_extent("canonical high plane", &[n, blocks, 8, 8])?
    } else { 0 };
    let metadata = checked_byte_extent("canonical metadata", &[n, blocks, if format == 1 { 20 } else { 16 }])?;
    let meta = low.checked_add(high).ok_or_else(|| invalid("metadata offset overflow"))?;
    let stride = meta.checked_add(metadata).ok_or_else(|| invalid("canonical stride overflow"))?;
    if u32::try_from(low).ok() != Some(prepared.plane1_offset)
        || u32::try_from(meta).ok() != Some(prepared.meta_offset)
        || u64::try_from(stride).ok() != Some(prepared.prepared_expert_stride)
    {
        return Err(invalid("canonical plane offsets/stride do not match geometry"));
    }
    let rows = checked_byte_extent("legacy prepared rows", &[params.n_tokens as usize, params.top_k as usize])?;
    if rows > i32::MAX as usize { return Err(invalid("routed rows exceed signed indexing")); }
    let input_bytes = checked_byte_extent("legacy prepared input", &[
        if slotted { rows } else { params.n_tokens as usize }, params.k as usize, 4])?;
    let ids_bytes = checked_byte_extent("legacy prepared IDs", &[rows, 4])?;
    let output_bytes = checked_byte_extent("legacy prepared output", &[rows, output_cols as usize, 4])?;
    let device_id = encoder.device_registry_id();
    if device_id != device.registry_id() { return Err(invalid("encoder and supplied device differ")); }
    let weights = buffer_range(&prepared.bytes, DType::U8,
        checked_byte_extent("canonical weights", &[params.n_experts as usize, stride])?, 4, false, device_id)?;
    let input = buffer_range(input, DType::F32, input_bytes, 4, false, device_id)?;
    let ids = buffer_range(ids, DType::U32, ids_bytes, 4, false, device_id)?;
    let output = buffer_range(output, DType::F32, output_bytes, 4, true, device_id)?;
    if [weights, input, ids].iter().any(|&read| overlaps(output, read)) {
        return Err(invalid("output must not overlap expert planes, input or IDs"));
    }
    Ok(ValidatedCall {
        planes: PreparedGpuParams { n_experts: params.n_experts, n: params.n,
            blocks: params.k / 256, format, raw_expert_stride: params.expert_stride,
            prepared_expert_stride: prepared.prepared_expert_stride,
            plane1_offset: prepared.plane1_offset, meta_offset: prepared.meta_offset },
        weights, input, ids, output, input_bytes, ids_bytes, output_bytes,
    })
}

// Keep Q6's two padding bytes per metadata record out of capture read ranges.
fn capture_reads(call: &ValidatedCall) -> Vec<BufferRange> {
    let p = call.planes;
    let mut reads = Vec::new();
    for expert in 0..p.n_experts as usize {
        let base = call.weights.0 + expert * p.prepared_expert_stride as usize;
        let meta = base + p.meta_offset as usize;
        reads.push((base, meta));
        if p.format == 0 {
            reads.push((meta, base + p.prepared_expert_stride as usize));
        } else {
            for block in 0..p.n as usize * p.blocks as usize {
                reads.push((meta + block * 20, meta + block * 20 + 18));
            }
        }
    }
    reads.push((call.input.0, call.input.0 + call.input_bytes));
    reads.push((call.ids.0, call.ids.0 + call.ids_bytes));
    reads
}

#[derive(Clone, Copy)]
enum Matvec { Selected, Q4Nr2, Geglu }

fn dispatch_mv(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams,
    routing: &GgmlRoutingPolicy, kind: Matvec, masked: bool,
) -> Result<()> {
    let geglu = matches!(kind, Matvec::Geglu);
    if (geglu || matches!(kind, Matvec::Q4Nr2)) && params.ggml_type != GgmlType::Q4_K {
        return Err(invalid("Q4 NR2 and grouped GEGLU require Q4_K"));
    }
    let call = validate(encoder, device, input, prepared, ids, output, params,
        if geglu { params.n / 2 } else { params.n }, false)?;
    let q6_nr2 = params.ggml_type == GgmlType::Q6_K && routing.expert_q6k_mv_nr2;
    let (name, columns, threads) = match kind {
        Matvec::Geglu => ("kernel_mul_mv_id_prepared_q4_K_f32_grouped_geglu_rt4", 8, metal::MTLSize::new(2, 32, 1)),
        Matvec::Q4Nr2 => {
            let fold = match params.k / 256 { 1 => 4, 2 => 2, _ => 1 };
            ("kernel_mul_mv_id_prepared_q4_K_f32_nr2", 4 * fold, metal::MTLSize::new(32, 2, 1))
        }
        Matvec::Selected if q6_nr2 => ("kernel_mul_mv_id_prepared_q6_K_f32_nr2", 4, metal::MTLSize::new(2, 32, 1)),
        Matvec::Selected if params.ggml_type == GgmlType::Q6_K => ("kernel_mul_mv_id_prepared_q6_K_f32", 2, metal::MTLSize::new(2, 32, 1)),
        Matvec::Selected => ("kernel_mul_mv_id_prepared_q4_K_f32", 2, metal::MTLSize::new(2, 32, 1)),
    };
    let pipeline = registry.get_pipeline(name, device.metal_device())?;
    if pipeline.max_total_threads_per_threadgroup() < 64 || pipeline.thread_execution_width() != 32 {
        return Err(invalid("legacy MV pipeline does not support its original launch geometry"));
    }
    let gpu = GgmlMatvecIdGpuParams {
        ne0: if geglu { i64::from(params.n / 2) } else { i64::from(params.n) },
        ..mv_id_gpu_params(params, masked)
    };
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(capture_reads(&call),
            vec![(call.output.0, call.output.0 + call.output_bytes)]);
    }
    encoder.encode_threadgroups_with_args(pipeline, &[
        (0, KernelArg::Buffer(&prepared.bytes)), (1, KernelArg::Buffer(input)),
        (2, KernelArg::Buffer(output)), (3, KernelArg::Buffer(ids)),
        (4, KernelArg::Bytes(as_bytes(&gpu))),
        (5, KernelArg::Bytes(as_bytes(&call.planes))),
    ], metal::MTLSize::new(u64::from(if geglu { params.n / 2 } else { params.n }).div_ceil(columns),
        u64::from(params.n_tokens) * u64::from(params.top_k), 1), threads);
    Ok(())
}

fn dispatch_auto(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams, slotted: bool, reuse_schedule: bool,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    if plan_expert_auto_route(params.n_tokens, params.top_k, params.k, false, routing) == ExpertAutoPlan::Mm {
        return super::prepared_mm::dispatch_pooled(encoder, registry, device,
            input, prepared, ids, output, scratch, params, slotted, reuse_schedule, routing,
            IdMmScheduleOptions::production(routing.expert_mm_schedule));
    }
    if slotted || reuse_schedule {
        return Err(invalid("slotted input and schedule reuse require the selected MM route"));
    }
    dispatch_mv(encoder, registry, device, input, prepared, ids, output, params,
        routing, Matvec::Selected, false)
}

/// Auto-select legacy MV/MM over canonical planes, using caller-owned MM scratch.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_id_ggml_prepared_with_policy(encoder, registry, device,
        input, prepared, ids, output, scratch, params, &routing)
}

/// Explicit-policy auto-select, without per-call GPU allocation.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_with_policy(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_auto(encoder, registry, device, input, prepared, ids, output, scratch,
        params, false, false, routing)
}

/// Caller-scratch spelling matching the native raw API.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    quantized_matmul_id_ggml_prepared(encoder, registry, device, input, prepared,
        ids, output, scratch, params)
}

/// Explicit-policy caller-scratch shared input.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled_with_policy(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_auto(encoder, registry, device, input, prepared, ids, output, scratch,
        params, false, false, routing)
}

/// Slotted input `[tokens, top_k, K]`, selected MM route only.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled_slotted(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_id_ggml_prepared_pooled_slotted_with_policy(encoder, registry,
        device, input, prepared, ids, output, scratch, params, &routing)
}

/// Explicit-policy slotted input.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled_slotted_with_policy(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_auto(encoder, registry, device, input, prepared, ids, output, scratch,
        params, true, false, routing)
}

/// Reuse the unchanged ID schedule in this encoder; IDs must not be mutated.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled_slotted_reusing_schedule(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_id_ggml_prepared_pooled_slotted_reusing_schedule_with_policy(
        encoder, registry, device, input, prepared, ids, output, scratch, params, &routing)
}

/// Explicit-policy slotted schedule reuse, preserving the native schedule key.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled_slotted_reusing_schedule_with_policy(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_auto(encoder, registry, device, input, prepared, ids, output, scratch,
        params, true, true, routing)
}

/// Forced selected MV (Q4 stock; Q6 stock or NR2 under registry policy).
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_mv(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_id_ggml_prepared_mv_with_policy(encoder, registry, device,
        input, prepared, ids, output, params, &routing)
}

/// Explicit-policy forced MV; out-of-range IDs write the native NaN poison.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_mv_with_policy(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams, routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_mv(encoder, registry, device, input, prepared, ids, output, params,
        routing, Matvec::Selected, false)
}

/// Forced MV with zero-writing padded/invalid expert IDs.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_mv_masked(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_id_ggml_prepared_mv_masked_with_policy(encoder, registry, device,
        input, prepared, ids, output, params, &routing)
}

/// Explicit-policy masked MV, including stock and NR2 Q6 selection.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_mv_masked_with_policy(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams, routing: &GgmlRoutingPolicy,
) -> Result<()> {
    dispatch_mv(encoder, registry, device, input, prepared, ids, output, params,
        routing, Matvec::Selected, true)
}

/// Explicit Q4 NR2 lane layout, including K=256/512 row-group folding.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_mv_q4_k_nr2(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    dispatch_mv(encoder, registry, device, input, prepared, ids, output, params,
        &GgmlRoutingPolicy::default(), Matvec::Q4Nr2, false)
}

/// Explicit masked Q4 NR2 lane layout.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_mv_q4_k_nr2_masked(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    dispatch_mv(encoder, registry, device, input, prepared, ids, output, params,
        &GgmlRoutingPolicy::default(), Matvec::Q4Nr2, true)
}

/// Grouped Q4 gate/up with the original GELU-tanh times up epilogue.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_mv_grouped_geglu(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, prepared: &GroupedM8ExpertPlanes, ids: &MlxBuffer,
    output: &MlxBuffer, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    dispatch_mv(encoder, registry, device, input, prepared, ids, output, params,
        &GgmlRoutingPolicy::default(), Matvec::Geglu, false)
}

/// Paired shared-input projections. Both are validated before the single map;
/// the second projection reads the same schedule without a projection barrier.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled_pair(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, first: &GroupedM8ExpertPlanes, second: &GroupedM8ExpertPlanes,
    ids: &MlxBuffer, first_output: &MlxBuffer, second_output: &MlxBuffer,
    scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let routing = ggml_routing_policy_for_registry(registry);
    quantized_matmul_id_ggml_prepared_pooled_pair_with_policy(encoder, registry, device,
        input, first, second, ids, first_output, second_output, scratch, params, &routing)
}

/// Explicit-policy paired MM. A failed validation invalidates the scratch key.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_pooled_pair_with_policy(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, first: &GroupedM8ExpertPlanes, second: &GroupedM8ExpertPlanes,
    ids: &MlxBuffer, first_output: &MlxBuffer, second_output: &MlxBuffer,
    scratch: &mut IdMmScratch, params: &GgmlQuantizedMatmulIdParams,
    routing: &GgmlRoutingPolicy,
) -> Result<()> {
    let result = (|| {
        if plan_expert_auto_route(params.n_tokens, params.top_k, params.k, false, routing) != ExpertAutoPlan::Mm {
            return Err(invalid("paired prepared projections require the selected MM route"));
        }
        scratch.check_capacity(params.n_experts, params.n_tokens, params.top_k)?;
        let options = IdMmScheduleOptions::production(routing.expert_mm_schedule);
        let first_plan = super::prepared_mm::plan(encoder, registry, device, input,
            first, ids, first_output, scratch, params, false, false, routing, options)?;
        let second_plan = super::prepared_mm::plan(encoder, registry, device, input,
            second, ids, second_output, scratch, params, false, true, routing, options)?;
        let range = |b: &MlxBuffer| -> Result<BufferRange> {
            buffer_range(b, b.dtype(), b.data_byte_len(), 4, false, device.registry_id())
        };
        let a = range(first_output)?;
        let b = range(second_output)?;
        let aw = range(&first.bytes)?;
        let bw = range(&second.bytes)?;
        if overlaps(a, b) || overlaps(a, bw) || overlaps(b, aw) {
            return Err(invalid("paired outputs must not overlap each other or either expert plane allocation"));
        }
        first_plan.encode(encoder, registry, device, false)?;
        second_plan.encode(encoder, registry, device, true)
    })();
    match &result {
        Ok(()) => {
            scratch.encode_serial = scratch.encode_serial.wrapping_add(1);
            scratch.schedule_key = Some(scratch.schedule_key_for(encoder, ids,
                params.n_tokens, params.top_k, params.n_experts, routing.expert_mm_schedule));
        }
        Err(_) => scratch.invalidate_schedule(),
    }
    result
}

/// Original fused classic Q6 gate/up/SiLU MM with a uniform reusable schedule.
/// Top-k remains the raw fused kernel's supported set {1, 8}.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_prepared_fused_gate_up_silu_mm_q6_k(
    encoder: &mut CommandEncoder, registry: &mut KernelRegistry, device: &MlxDevice,
    input: &MlxBuffer, gate: &GroupedM8ExpertPlanes, up: &GroupedM8ExpertPlanes,
    ids: &MlxBuffer, output: &MlxBuffer, scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams, reuse_schedule: bool,
) -> Result<()> {
    let result = (|| {
        if params.ggml_type != GgmlType::Q6_K || !matches!(params.top_k, 1 | 8) {
            return Err(invalid("fused prepared MM requires Q6_K and top_k in {1, 8}"));
        }
        let gate_call = validate(encoder, device, input, gate, ids, output, params, params.n, false)?;
        let up_call = validate(encoder, device, input, up, ids, output, params, params.n, false)?;
        if output.byte_offset() % 16 != 0 {
            return Err(invalid("fused prepared MM requires a 16-byte-aligned output"));
        }
        if params.n_tokens > (i32::MAX as u32).saturating_sub(1024)
            || u64::from(params.n_tokens) * u64::from(params.n_experts) > i32::MAX as u64
            || params.n_experts > i16::MAX as u32
        {
            return Err(invalid("fused uniform map exceeds signed indexing"));
        }
        scratch.check_capacity(params.n_experts, params.n_tokens, params.top_k)?;
        if reuse_schedule {
            scratch.validate_schedule_reuse(encoder, ids, params.n_tokens,
                params.top_k, params.n_experts, ExpertMmSchedule::UniformGrid)?;
        }
        let (htpe_bytes, hids_bytes) = id_mm_scratch_bytes(ExpertMmSchedule::UniformGrid,
            params.n_experts, params.n_tokens, params.top_k)?;
        let htpe = buffer_range(&scratch.htpe, DType::U32, htpe_bytes, 4, true, device.registry_id())?;
        let hids = buffer_range(&scratch.hids, DType::U32, hids_bytes, 4, true, device.registry_id())?;
        let reads = [gate_call.weights, up_call.weights, gate_call.input, gate_call.ids];
        let writes = [gate_call.output, htpe, hids];
        for (index, &write) in writes.iter().enumerate() {
            if reads.iter().any(|&read| overlaps(write, read))
                || writes[..index].iter().any(|&other| overlaps(write, other))
            {
                return Err(invalid("fused output and scratch overlap another resource"));
            }
        }
        let pipeline = registry.get_pipeline("kernel_fused_gate_up_silu_mm_id_prepared_q6_K_f32",
            device.metal_device())?.to_owned();
        let memory_limit = device.metal_device().max_threadgroup_memory_length();
        if pipeline.max_total_threads_per_threadgroup() < 128 || pipeline.thread_execution_width() != 32
            || pipeline.static_threadgroup_memory_length().checked_add(16384)
                .filter(|&bytes| bytes <= memory_limit).is_none()
        {
            return Err(invalid("fused pipeline does not support original thread/memory geometry"));
        }
        if !reuse_schedule {
            let map = registry.get_pipeline(if params.top_k == 1 { "kernel_mul_mm_id_map0_ne20_1" }
                else { "kernel_mul_mm_id_map0_ne20_8" }, device.metal_device())?;
            let shared = (u64::from(params.n_experts) * u64::from(params.top_k) * 4).div_ceil(16) * 16;
            if u64::from(params.n_experts) > map.max_total_threads_per_threadgroup()
                || map.static_threadgroup_memory_length().checked_add(shared)
                    .filter(|&bytes| bytes <= memory_limit).is_none()
            {
                return Err(invalid("fused map exceeds original thread/memory geometry"));
            }
        }
        let mm = GgmlIdMmMmGpuParams {
            ne00: params.k as i32, ne02: params.n_experts as i32,
            nb01: u64::from(params.k / 256) * 210, nb02: params.expert_stride, nb03: 0,
            ne11: params.top_k as i32, _pad0: 0, nb10: 4, nb11: 0,
            nb12: u64::from(params.k) * 4, nb13: 0, ne20: params.top_k as i32,
            ne21: params.n_tokens as i32, ne0: params.n as i32, ne1: params.top_k as i32,
            r2: 1, r3: 1, sched_flags: 0, _pad1: 0,
        };
        if !reuse_schedule {
            let dispatch = GgmlIdMmDispatchParams { n_tokens: params.n_tokens,
                top_k: params.top_k, n: params.n, k: params.k, n_experts: params.n_experts,
                expert_stride: params.expert_stride, ggml_type: params.ggml_type };
            encode_legacy_map0(encoder, registry, device, ids, &scratch.htpe, &scratch.hids, &dispatch)?;
        }
        if encoder.is_capturing() {
            let mut capture = capture_reads(&gate_call);
            capture.extend(capture_reads(&up_call));
            capture.extend([(htpe.0, htpe.0 + htpe_bytes), (hids.0, hids.0 + hids_bytes)]);
            encoder.set_pending_buffer_ranges(capture,
                vec![(gate_call.output.0, gate_call.output.0 + gate_call.output_bytes)]);
        }
        encoder.encode_threadgroups_with_args_and_shared(&pipeline, &[
            (0, KernelArg::Bytes(as_bytes(&mm))), (1, KernelArg::Buffer(&gate.bytes)),
            (2, KernelArg::Buffer(&up.bytes)), (3, KernelArg::Buffer(input)),
            (4, KernelArg::Buffer(&scratch.htpe)), (5, KernelArg::Buffer(&scratch.hids)),
            (6, KernelArg::Buffer(output)), (7, KernelArg::Bytes(as_bytes(&gate_call.planes))),
            (8, KernelArg::Bytes(as_bytes(&up_call.planes))),
        ], &[(0, 16384)], metal::MTLSize::new(u64::from(params.n_tokens).div_ceil(32),
            u64::from(params.n / 64), u64::from(params.n_experts)), metal::MTLSize::new(128, 1, 1));
        Ok(())
    })();
    match &result {
        Ok(()) if !reuse_schedule => {
            scratch.encode_serial = scratch.encode_serial.wrapping_add(1);
            scratch.schedule_key = Some(scratch.schedule_key_for(encoder, ids,
                params.n_tokens, params.top_k, params.n_experts, ExpertMmSchedule::UniformGrid));
        }
        Ok(()) => {}
        Err(_) => scratch.invalidate_schedule(),
    }
    result
}
