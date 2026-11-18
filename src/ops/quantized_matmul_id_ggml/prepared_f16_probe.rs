//! Unpublished, native-only F16 decoder-fragment experiment; no serving route.

use super::{
    as_bytes, checked_byte_extent, CommandEncoder, DType, GgmlQuantizedMatmulIdParams,
    GgmlType, KernelArg, KernelRegistry, MlxBuffer, MlxError, GroupedM8ExpertPlanes, Result,
};
use crate::device::metal_device_registry_id;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbeGpuParams {
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
    MlxError::InvalidArgument(format!("grouped M8 F16 tile probe: {message}"))
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
        return Err(invalid("buffer dtype or logical extent does not match the probe"));
    }
    if writable && (!buffer.is_cpu_writable() || buffer.is_file_backed()) {
        return Err(invalid("outputs must be writable, non-file-backed allocations"));
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
    let start = base.checked_add(offset)
        .ok_or_else(|| invalid("buffer address overflow"))?;
    if start % alignment != 0 || offset % alignment != 0 {
        return Err(invalid("buffer offset is not aligned for its shader reads/writes"));
    }
    let end = start.checked_add(required)
        .ok_or_else(|| invalid("buffer range overflow"))?;
    // Validate the whole logical window as well, for alias checking below.
    start.checked_add(buffer.data_byte_len())
        .ok_or_else(|| invalid("logical buffer address overflow"))?;
    Ok((start, end))
}

fn overlaps(a: Range, b: Range) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Compare actual raw classic/tensor F16 decoder fragments with direct readers
/// of the lossless prepared Q4_K/Q6_K planes. This is an unpublished spike, not
/// an inference selector or a prepared-source lifetime/release API.
///
/// Both outputs are writable F16 with at least
/// `n_experts * n * (k / 256) * 16 * 16` elements. Their order is expert, row,
/// block, `il` (0..16), then matrix column/row (`reg[i / 4][i % 4]`). Buffer
/// offsets and padded raw expert strides are honored. The raw/prepared owners
/// remain unchanged; only `prepared.bytes` and its layout are consumed here.
/// `n_tokens` and `top_k` do not participate in this decoder-only experiment.
///
/// All validation and pipeline lookup precede encoding. Barriers order prior
/// same-encoder producers and later consumers. Cross-encoder submission and
/// CPU/GPU synchronization remain the caller's responsibility. No allocation
/// of GPU storage, host wait, or temporary raw reconstruction is performed.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml_grouped_m8_f16_tiles_probe_into(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    raw: &MlxBuffer,
    prepared: &GroupedM8ExpertPlanes,
    reference: &MlxBuffer,
    candidate: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
    tensor_reference: bool,
) -> Result<()> {
    let format = match params.ggml_type {
        GgmlType::Q4_K => 0,
        GgmlType::Q6_K => 1,
        _ => return Err(invalid("only Q4_K and Q6_K are supported")),
    };
    if params.n_experts == 0 || params.n == 0 || params.n % 64 != 0
        || params.k == 0 || params.k % 256 != 0
    {
        return Err(invalid("requires experts > 0, N > 0 divisible by 64 and K > 0 divisible by 256"));
    }
    if prepared.n_experts != params.n_experts || prepared.n != params.n
        || prepared.k != params.k || prepared.ggml_type != params.ggml_type
    {
        return Err(invalid("params do not match the opaque prepared geometry/type"));
    }
    let rows = params.n as usize;
    let experts = params.n_experts as usize;
    let blocks = (params.k / 256) as usize;
    let matrix_bytes = checked_byte_extent("F16 probe raw matrix",
        &[rows, blocks, params.ggml_type.block_bytes() as usize])?;
    let raw_stride = usize::try_from(params.expert_stride)
        .map_err(|_| invalid("raw expert stride exceeds usize"))?;
    if raw_stride < matrix_bytes || raw_stride % 2 != 0 {
        return Err(invalid("raw expert stride must cover a matrix and be half-aligned"));
    }
    let raw_bytes = (experts - 1).checked_mul(raw_stride)
        .and_then(|offset| offset.checked_add(matrix_bytes))
        .ok_or_else(|| invalid("raw extent overflow"))?;
    let plane0 = checked_byte_extent("F16 probe low plane", &[rows, blocks, 8, 16])?;
    let plane1 = if format == 1 {
        checked_byte_extent("F16 probe high plane", &[rows, blocks, 8, 8])?
    } else { 0 };
    let meta_offset = plane0.checked_add(plane1)
        .ok_or_else(|| invalid("metadata offset overflow"))?;
    let metadata = checked_byte_extent("F16 probe metadata", &[rows, blocks, if format == 1 { 20 } else { 16 }])?;
    let prepared_stride = meta_offset.checked_add(metadata)
        .ok_or_else(|| invalid("prepared stride overflow"))?;
    if u32::try_from(plane0).ok() != Some(prepared.plane1_offset)
        || u32::try_from(meta_offset).ok() != Some(prepared.meta_offset)
        || u64::try_from(prepared_stride).ok() != Some(prepared.prepared_expert_stride)
    {
        return Err(invalid("opaque prepared plane offsets/stride do not match its geometry"));
    }
    let prepared_bytes = prepared_stride.checked_mul(experts)
        .ok_or_else(|| invalid("prepared extent overflow"))?;
    let output_bytes = checked_byte_extent("F16 probe output", &[experts, rows, blocks, 16, 16, 2])?;
    let grid_x = u32::try_from(blocks).ok().and_then(|b| b.checked_mul(16))
        .ok_or_else(|| invalid("fragment grid width overflow"))?;
    checked_byte_extent("F16 probe grid", &[experts, rows, grid_x as usize])?;
    let device_id = encoder.device_registry_id();
    let raw_range = buffer_range(raw, DType::U8, raw_bytes, 2, false, device_id)?;
    let prepared_range = buffer_range(&prepared.bytes, DType::U8, prepared_bytes, 4, false, device_id)?;
    let reference_range = buffer_range(reference, DType::F16, output_bytes, 2, true, device_id)?;
    let candidate_range = buffer_range(candidate, DType::F16, output_bytes, 2, true, device_id)?;
    let raw_window = (raw_range.0, raw_range.0 + raw.data_byte_len());
    let prepared_window = (prepared_range.0, prepared_range.0 + prepared.bytes.data_byte_len());
    if overlaps(reference_range, candidate_range)
        || overlaps(reference_range, raw_window) || overlaps(candidate_range, raw_window)
        || overlaps(reference_range, prepared_window) || overlaps(candidate_range, prepared_window)
        || overlaps(raw_window, prepared_window)
    {
        return Err(invalid("reference/candidate and raw/prepared storage must not overlap"));
    }
    let gpu = ProbeGpuParams {
        n_experts: params.n_experts, n: params.n, blocks: params.k / 256, format,
        raw_expert_stride: params.expert_stride,
        prepared_expert_stride: prepared.prepared_expert_stride,
        plane1_offset: prepared.plane1_offset, meta_offset: prepared.meta_offset,
    };
    let device = raw.metal_buffer().device();
    let reference_name = if tensor_reference {
        "grouped_m8_f16_tiles_tensor_reference"
    } else { "grouped_m8_f16_tiles_reference" };
    let reference_pipeline = registry.get_pipeline(reference_name, device)?.to_owned();
    let candidate_pipeline = registry.get_pipeline("grouped_m8_f16_tiles_candidate", device)?.to_owned();
    let threads = u64::from(grid_x).min(64)
        .min(reference_pipeline.max_total_threads_per_threadgroup())
        .min(candidate_pipeline.max_total_threads_per_threadgroup());
    if threads == 0 {
        return Err(invalid("probe pipelines admit no threads"));
    }
    let grid = metal::MTLSize::new(u64::from(grid_x), u64::from(params.n), u64::from(params.n_experts));
    let group = metal::MTLSize::new(threads, 1, 1);
    encoder.memory_barrier();
    if encoder.is_capturing() {
        let reads = (0..experts).map(|expert| {
            let start = raw_range.0 + expert * raw_stride;
            (start, start + matrix_bytes)
        }).collect();
        encoder.set_pending_buffer_ranges(reads, vec![reference_range]);
    }
    encoder.encode_with_args(&reference_pipeline, &[
        (0, KernelArg::Buffer(raw)), (1, KernelArg::Buffer(reference)),
        (2, KernelArg::Bytes(as_bytes(&gpu))),
    ], grid, group);
    if encoder.is_capturing() {
        let mut reads = Vec::with_capacity(experts * if format == 0 { 2 } else { 1 + rows * blocks });
        for expert in 0..experts {
            let start = prepared_range.0 + expert * prepared_stride;
            reads.push((start, start + meta_offset));
            let meta = start + meta_offset;
            if format == 0 {
                reads.push((meta, meta + metadata));
            } else {
                // Q6's two metadata padding bytes are never read.
                for block in 0..rows * blocks {
                    reads.push((meta + block * 20, meta + block * 20 + 18));
                }
            }
        }
        encoder.set_pending_buffer_ranges(reads, vec![candidate_range]);
    }
    encoder.encode_with_args(&candidate_pipeline, &[
        (0, KernelArg::Buffer(&prepared.bytes)), (1, KernelArg::Buffer(candidate)),
        (2, KernelArg::Bytes(as_bytes(&gpu))),
    ], grid, group);
    encoder.memory_barrier();
    Ok(())
}
