//! Shared byte layout and full/live expert-plane preparation.

use super::{
    as_bytes, checked_byte_extent, compact_max_tiles, encode_prepared_compact_map0,
    id_mm_scratch_bytes, prepare_compact_map0, require_tensor_kernel_family,
    validate_id_mm_scratch_dtypes, CommandEncoder, DType,
    ExpertMmSchedule, GgmlIdMmDispatchParams, GgmlQuantizedMatmulIdParams, GgmlType, IdMmScratch,
    KernelArg, KernelRegistry, MlxBuffer, MlxDevice, MlxError, PreparedGroupedM8Experts, Result,
    TensorKernelFamily,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GroupedM8PrepareGpuParams {
    n_experts: u32,
    n: u32,
    k: u32,
    format: u32,
    source_expert_stride: u64,
    prepared_expert_stride: u64,
    plane1_offset: u32,
    meta_offset: u32,
    source_row_bytes: u32,
    max_descriptors: u32,
}

struct Layout {
    gpu: GroupedM8PrepareGpuParams,
    total: usize,
    matrix_units: u32,
    matrix_bytes: usize,
}

impl Layout {
    fn new(n_experts: u32, n: u32, k: u32, ggml_type: GgmlType) -> Result<Self> {
        let format = match ggml_type {
            GgmlType::Q4_K => 0,
            GgmlType::Q6_K => 1,
            other => return Err(MlxError::InvalidArgument(format!(
                "grouped M8 preparation requires Q4_K or Q6_K, got {other:?}"
            ))),
        };
        if n_experts == 0 || n == 0 || n % 64 != 0 || k == 0 || k % 256 != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "grouped M8 preparation requires experts > 0, N % 64 == 0 and K % 256 == 0; got experts={n_experts} N={n} K={k}"
            )));
        }
        let rows = n as usize;
        let groups = (k / 32) as usize;
        let matrix_units = n.checked_mul(k / 32).ok_or_else(|| {
            MlxError::InvalidArgument("grouped M8 matrix units exceed u32".into())
        })?;
        let matrix_bytes = checked_byte_extent(
            "grouped M8 source matrix",
            &[rows, (k / ggml_type.block_values()) as usize, ggml_type.block_bytes() as usize],
        )?;
        let plane0 = checked_byte_extent("grouped M8 low plane", &[rows, groups, 16])?;
        let plane1 = if format == 1 {
            checked_byte_extent("grouped M8 high plane", &[rows, groups, 8])?
        } else {
            0
        };
        let metadata = checked_byte_extent(
            "grouped M8 metadata plane",
            &[rows, groups / 8, if format == 1 { 20 } else { 16 }],
        )?;
        let meta_offset = plane0.checked_add(plane1).ok_or_else(|| {
            MlxError::InvalidArgument("grouped M8 metadata offset overflow".into())
        })?;
        let stride = meta_offset.checked_add(metadata).ok_or_else(|| {
            MlxError::InvalidArgument("grouped M8 prepared stride overflow".into())
        })?;
        let total = stride.checked_mul(n_experts as usize).ok_or_else(|| {
            MlxError::InvalidArgument("grouped M8 prepared size overflow".into())
        })?;
        Ok(Self {
            gpu: GroupedM8PrepareGpuParams {
                n_experts, n, k, format,
                source_expert_stride: 0,
                prepared_expert_stride: stride as u64,
                plane1_offset: u32::try_from(plane0).map_err(|_| {
                    MlxError::InvalidArgument("grouped M8 plane offset exceeds u32".into())
                })?,
                meta_offset: u32::try_from(meta_offset).map_err(|_| {
                    MlxError::InvalidArgument("grouped M8 metadata offset exceeds u32".into())
                })?,
                source_row_bytes: u32::try_from(matrix_bytes / rows).map_err(|_| {
                    MlxError::InvalidArgument("grouped M8 row bytes exceed u32".into())
                })?,
                max_descriptors: 0,
            },
            total, matrix_units, matrix_bytes,
        })
    }

    fn source(&mut self, weight: &MlxBuffer, stride: u64) -> Result<usize> {
        if weight.dtype() != DType::U8 {
            return Err(MlxError::InvalidArgument(format!(
                "grouped M8 preparation requires U8 expert weights, got {:?}", weight.dtype()
            )));
        }
        if stride < self.matrix_bytes as u64 {
            return Err(MlxError::InvalidArgument(format!(
                "grouped M8 source expert stride {stride} is smaller than matrix bytes {}",
                self.matrix_bytes
            )));
        }
        let source_bytes = u64::from(self.gpu.n_experts - 1)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(self.matrix_bytes as u64))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| MlxError::InvalidArgument("grouped M8 source size overflow".into()))?;
        if weight.data_byte_len() < source_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "grouped M8 source needs {source_bytes} bytes, got {}", weight.data_byte_len()
            )));
        }
        self.gpu.source_expert_stride = stride;
        Ok(source_bytes)
    }

    fn handle(&self, weight: &MlxBuffer, bytes: MlxBuffer, ggml_type: GgmlType) -> PreparedGroupedM8Experts {
        PreparedGroupedM8Experts {
            _source: weight.clone(), bytes,
            n_experts: self.gpu.n_experts,
            n: self.gpu.n,
            k: self.gpu.k,
            ggml_type,
            prepared_expert_stride: self.gpu.prepared_expert_stride,
            plane1_offset: self.gpu.plane1_offset,
            meta_offset: self.gpu.meta_offset,
        }
    }
}

/// Exact storage for a prepared Q4_K/Q6_K expert stack, including every plane.
/// Geometry and all layout/indexing arithmetic are checked; no storage is allocated.
pub fn grouped_m8_prepared_stack_bytes(n_experts: u32, n: u32, k: u32, ggml_type: GgmlType) -> Result<usize> {
    Ok(Layout::new(n_experts, n, k, ggml_type)?.total)
}

fn range(buffer: &MlxBuffer, extent: usize) -> (usize, usize) {
    let start = (buffer.contents_ptr() as usize).saturating_add(buffer.byte_offset() as usize);
    (start, start.saturating_add(extent))
}

/// Prepare one complete Q4_K or Q6_K expert stack once for the grouped
/// eight-row Metal tensor tile.
pub fn prepare_quantized_matmul_id_ggml_grouped_m8(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<PreparedGroupedM8Experts> {
    let mut layout = Layout::new(params.n_experts, params.n, params.k, params.ggml_type)?;
    let source_bytes = layout.source(weight, params.expert_stride)?;
    require_tensor_kernel_family(registry, device, TensorKernelFamily::GroupedM8Experts)?;
    let pipeline = registry.get_pipeline("grouped_m8_prepare_q4k_q6k", device.metal_device())?;
    let bytes = device.alloc_buffer(layout.total, DType::U8, vec![layout.total])?;
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(vec![range(weight, source_bytes)], vec![range(&bytes, layout.total)]);
    }
    encoder.encode_with_args(
        pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(&bytes)),
            (2, KernelArg::Bytes(as_bytes(&layout.gpu))),
        ],
        metal::MTLSize::new(u64::from(layout.matrix_units), u64::from(params.n_experts), 1),
        metal::MTLSize::new(256.min(u64::from(layout.matrix_units)), 1, 1),
    );
    encoder.memory_barrier();
    Ok(layout.handle(weight, bytes, params.ggml_type))
}

/// Repack only experts present in the native compact route schedule into
/// caller-owned U8 storage. Unselected experts and bytes after the required
/// stack extent are untouched. The returned handle aliases `output` without
/// allocating or copying prepared weight storage.
///
/// `reuse_schedule` requires the same ids view, geometry, scratch generation
/// and command encoder as the schedule's producer, with no intervening ids
/// writes. `masked_routes` has the native map0 meaning (id == n_experts is
/// skipped); it must match the reused schedule's route interpretation.
///
/// WAR and RAW barriers order reuse within this command stream, without host
/// waits. Finish all consumers of the prior contents before calling again;
/// old handles alias the overwritten bytes. Consume only experts from the
/// prepared schedule until another preparation fills their planes.
#[allow(clippy::too_many_arguments)]
pub fn prepare_quantized_matmul_id_ggml_grouped_m8_live_into(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    ids: &MlxBuffer,
    scratch: &mut IdMmScratch,
    params: &GgmlQuantizedMatmulIdParams,
    masked_routes: bool,
    reuse_schedule: bool,
) -> Result<PreparedGroupedM8Experts> {
    let mut layout = Layout::new(params.n_experts, params.n, params.k, params.ggml_type)?;
    let source_bytes = layout.source(weight, params.expert_stride)?;
    if output.dtype() != DType::U8 || output.data_byte_len() < layout.total
        || output.byte_offset() % 16 != 0 || output.is_file_backed() || !output.is_cpu_writable()
    {
        return Err(MlxError::InvalidArgument(format!(
            "grouped M8 live preparation requires writable, 16-byte-aligned U8 output of at least {} bytes", layout.total
        )));
    }
    if params.n_tokens == 0 || !matches!(params.top_k, 1 | 6 | 8)
        || params.top_k > params.n_experts || params.n_experts > i32::MAX as u32
    {
        return Err(MlxError::InvalidArgument(
            "grouped M8 live preparation requires tokens > 0, top_k in {1,6,8} <= experts, and signed expert indexing".into()
        ));
    }
    let rows = u64::from(params.n_tokens) * u64::from(params.top_k);
    if rows > i32::MAX as u64 || rows + u64::from(params.n_experts) > u64::from(u32::MAX) {
        return Err(MlxError::InvalidArgument("grouped M8 live route indexing overflow".into()));
    }
    let ids_bytes = checked_byte_extent("grouped M8 live ids", &[rows as usize, DType::U32.size_of()])?;
    if ids.dtype() != DType::U32 || ids.data_byte_len() < ids_bytes || ids.byte_offset() % 4 != 0 {
        return Err(MlxError::InvalidArgument("grouped M8 live preparation requires sufficiently large, aligned U32 ids".into()));
    }
    scratch.check_capacity(params.n_experts, params.n_tokens, params.top_k)?;
    validate_id_mm_scratch_dtypes("grouped M8 live preparation", &scratch.htpe, &scratch.hids)?;
    let (htpe_bytes, hids_bytes) = id_mm_scratch_bytes(
        ExpertMmSchedule::CompactTiles, params.n_experts, params.n_tokens, params.top_k,
    )?;
    if scratch.htpe.data_byte_len() < htpe_bytes || scratch.hids.data_byte_len() < hids_bytes
        || scratch.htpe.byte_offset() % 4 != 0 || scratch.hids.byte_offset() % 4 != 0
    {
        return Err(MlxError::InvalidArgument("grouped M8 live preparation scratch extent/alignment mismatch".into()));
    }
    let reads = [("source", range(weight, source_bytes)), ("ids", range(ids, ids_bytes))];
    let writes = [
        ("output", range(output, layout.total)),
        ("htpe", range(&scratch.htpe, htpe_bytes)),
        ("hids", range(&scratch.hids, hids_bytes)),
    ];
    let overlaps = |a: (usize, usize), b: (usize, usize)| a.0 < b.1 && b.0 < a.1;
    for (i, (name, write)) in writes.iter().enumerate() {
        if reads.iter().chain(writes[..i].iter()).any(|(_, other)| overlaps(*write, *other)) {
            return Err(MlxError::InvalidArgument(format!("grouped M8 live preparation {name} overlaps another buffer")));
        }
    }
    let maximum_tiles = compact_max_tiles(rows, u64::from(params.n_experts));
    layout.gpu.max_descriptors = u32::try_from(maximum_tiles).map_err(|_| {
        MlxError::InvalidArgument("grouped M8 live descriptor bound exceeds u32".into())
    })?;
    if reuse_schedule {
        scratch.validate_schedule_reuse(
            encoder, ids, params.n_tokens, params.top_k, params.n_experts, ExpertMmSchedule::CompactTiles,
        )?;
    }
    require_tensor_kernel_family(registry, device, TensorKernelFamily::GroupedM8Experts)?;
    // Resolve every repack pipeline failure before map0 can encode anything.
    let pipeline = registry.get_pipeline("grouped_m8_prepare_live_q4k_q6k", device.metal_device())?.clone();
    let threads = 256.min(pipeline.max_total_threads_per_threadgroup()).min(u64::from(layout.matrix_units));
    if threads == 0 {
        return Err(MlxError::InvalidArgument("grouped M8 live repack has no available threads".into()));
    }
    let dispatch = GgmlIdMmDispatchParams {
        n_tokens: params.n_tokens, top_k: params.top_k, n: params.n, k: params.k,
        n_experts: params.n_experts, expert_stride: params.expert_stride, ggml_type: params.ggml_type,
    };
    let launch = if reuse_schedule {
        None
    } else {
        Some(prepare_compact_map0(registry, device, &dispatch, maximum_tiles, masked_routes)?)
    };
    let bytes = output.slice_view(0, layout.total);
    // Protect both the prepared root and a freshly overwritten route schedule
    // from prior consumers on Metal's concurrent-dispatch compute encoder.
    encoder.memory_barrier();
    if let Some(launch) = launch {
        scratch.invalidate_schedule();
        encode_prepared_compact_map0(encoder, ids, &scratch.htpe, &scratch.hids, &launch);
        scratch.encode_serial = scratch.encode_serial.wrapping_add(1);
        scratch.schedule_key = Some(scratch.schedule_key_for(
            encoder, ids, params.n_tokens, params.top_k, params.n_experts, ExpertMmSchedule::CompactTiles,
        ));
    }
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(
            vec![range(weight, source_bytes), range(&scratch.htpe, htpe_bytes)],
            vec![range(&bytes, layout.total)],
        );
    }
    encoder.encode_with_args(
        &pipeline,
        &[
            (0, KernelArg::Buffer(weight)),
            (1, KernelArg::Buffer(&bytes)),
            (2, KernelArg::Bytes(as_bytes(&layout.gpu))),
            (3, KernelArg::Buffer(&scratch.htpe)),
        ],
        metal::MTLSize::new(u64::from(layout.matrix_units), maximum_tiles, 1),
        metal::MTLSize::new(threads, 1, 1),
    );
    encoder.memory_barrier();
    Ok(layout.handle(weight, bytes, params.ggml_type))
}
