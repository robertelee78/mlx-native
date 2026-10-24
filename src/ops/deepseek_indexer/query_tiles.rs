//! Fixed-256 query tiling. Validation/preparation errors encode no commands.
use super::*;
use metal::ComputePipelineState;
mod encode;
mod plan;
#[cfg(test)]
mod tests;
use plan::{checked_product, invalid, Tile};
pub use plan::{deepseek_indexer_query_tile_plan, DeepSeekIndexerQueryTilePlan};

struct PreparedTile {
    params: DeepSeekIndexerParams,
    q: MlxBuffer,
    kv: MlxBuffer,
    weights: MlxBuffer,
    scratch: MlxBuffer,
    bank_a: MlxBuffer,
    bank_b: MlxBuffer,
    output: MlxBuffer,
    ranges: [(usize, usize); 7],
}
struct Pipelines {
    score: ComputePipelineState,
    block: ComputePipelineState,
    merge: ComputePipelineState,
    finalize: ComputePipelineState,
}

// Shared by production and fault controls: nothing is encoded until both
// the LAST view and LAST pipeline have successfully prepared.
fn prepare_all<T, P>(
    tiles: &[Tile],
    mut tile: impl FnMut(&Tile) -> Result<T>,
    pipelines: impl FnOnce() -> Result<P>,
) -> Result<(Vec<T>, P)> {
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(tiles.len())
        .map_err(|_| invalid("tile metadata allocation failed"))?;
    for item in tiles {
        prepared.push(tile(item)?);
    }
    Ok((prepared, pipelines()?))
}

fn prepared_dispatch<T, P>(
    tiles: &[Tile],
    tile: impl FnMut(&Tile) -> Result<T>,
    pipelines: impl FnOnce() -> Result<P>,
    encode: impl FnOnce(&[T], &P),
) -> Result<()> {
    let (tiles, pipelines) = prepare_all(tiles, tile, pipelines)?;
    encode(&tiles, &pipelines);
    Ok(())
}

fn checked_range(
    base: usize,
    origin: usize,
    bytes: usize,
    physical_bytes: usize,
) -> Result<(usize, usize)> {
    if base == 0 {
        return Err(invalid("null physical buffer address"));
    }
    let end = origin
        .checked_add(bytes)
        .ok_or_else(|| invalid("physical end overflow"))?;
    if end > physical_bytes {
        return Err(invalid("view exceeds physical resource"));
    }
    let start = base
        .checked_add(origin)
        .ok_or_else(|| invalid("address origin overflow"))?;
    Ok((
        start,
        start
            .checked_add(bytes)
            .ok_or_else(|| invalid("address overflow"))?,
    ))
}
fn physical_range(buffer: &MlxBuffer) -> Result<(usize, usize)> {
    let origin =
        usize::try_from(buffer.byte_offset()).map_err(|_| invalid("physical offset overflow"))?;
    checked_range(
        buffer.contents_ptr() as usize,
        origin,
        buffer.data_byte_len(),
        buffer.byte_len(),
    )
}
fn validate_spec(
    actual_dtype: DType,
    actual_shape: &[usize],
    actual_bytes: usize,
    dtype: DType,
    shape: &[usize],
) -> Result<()> {
    let bytes = checked_product(&[checked_product(shape)?, dtype.size_of()])?;
    if actual_dtype != dtype || actual_shape != shape || actual_bytes != bytes {
        return Err(invalid("buffer dtype, logical length or shape mismatch"));
    }
    Ok(())
}
fn validate(buf: &MlxBuffer, dtype: DType, shape: &[usize]) -> Result<(usize, usize)> {
    validate_spec(buf.dtype(), buf.shape(), buf.data_byte_len(), dtype, shape)?;
    let range = physical_range(buf)?;
    if range.0 == 0 || range.0 % dtype.size_of() != 0 {
        return Err(invalid("null or unaligned logical buffer address"));
    }
    Ok(range)
}

fn view(buf: &MlxBuffer, offset: usize, shape: Vec<usize>) -> Result<MlxBuffer> {
    let bytes = checked_product(&[checked_product(&shape)?, buf.dtype().size_of()])?;
    buf.data_view(offset, bytes, buf.dtype(), shape)
}
fn validate_writable(file_backed: bool, cpu_writable: bool) -> Result<()> {
    if file_backed || !cpu_writable {
        return Err(invalid(
            "scratch, top-k scratch and output require writable non-file-backed storage",
        ));
    }
    Ok(())
}

fn no_alias(ranges: &[(usize, usize); 6]) -> Result<()> {
    // Read-only q/KV/weights may overlap; all writable ranges must be disjoint.
    for a in 3..6 {
        for b in 0..a {
            if ranges[a].0 < ranges[b].1 && ranges[b].0 < ranges[a].1 {
                return Err(invalid("writable range aliases another argument"));
            }
        }
    }
    Ok(())
}

/// Allocation-free Metal dispatch using fixed 256-row score/top-k workspaces.
///
/// Batch one only. Inputs/output describe the COMPLETE transaction. Workspace
/// shapes come from [`deepseek_indexer_query_tile_plan`]. All views, bounds,
/// alias ranges and four pipelines prepare before any command is encoded.
/// Writable arguments must be non-file-backed and CPU-writable. Known shared
/// views are checked by virtual address ranges; distinct mappings/external raw
/// backings can physically alias without overlapping virtual addresses. Callers
/// must supply known nonaliasing backing provenance; unknown aliasing is forbidden.
/// CPU metadata is proportional to the checked tile count. Driver failures and
/// panics are not transactional. Callers retain normal command-lifetime owners.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_indexer_into_mma_tiled(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    weights: &MlxBuffer,
    scratch: &MlxBuffer,
    topk_scratch: &MlxBuffer,
    output: &MlxBuffer,
    output_row_stride: usize,
    output_column_offset: usize,
    params: &DeepSeekIndexerParams,
) -> Result<()> {
    let plan = deepseek_indexer_query_tile_plan(params, output_row_stride, output_column_offset)?;
    let queries = params.query_len as usize;
    let keys = params.kv_len as usize;
    let ranges = [
        validate(q, DType::BF16, &[1, queries, 64, 128])?,
        validate(kv, DType::BF16, &[1, keys, 128])?,
        validate(weights, DType::F32, &[1, queries, 64])?,
        validate(scratch, DType::F32, &[1, plan.workspace_rows, keys])?,
        validate(
            topk_scratch,
            DType::I32,
            &[1, plan.workspace_rows, plan.topk_row_elements],
        )?,
        validate(output, DType::I32, &[1, queries, output_row_stride])?,
    ];
    for writable in [scratch, topk_scratch, output] {
        validate_writable(writable.is_file_backed(), writable.is_cpu_writable())?;
    }
    no_alias(&ranges)?;
    prepared_dispatch(
        &plan.tiles,
        |t| {
            debug_assert_eq!(t.start_pos, params.start_pos + t.origin as u32);
            let q = view(q, t.q_byte_offset, vec![1, t.rows, 64, 128])?;
            let kv = kv.clone();
            let weights = view(weights, t.weight_byte_offset, vec![1, t.rows, 64])?;
            let scratch = view(scratch, 0, vec![1, t.rows, keys])?;
            let bank_elements = checked_product(&[t.rows, plan.topk_row_elements / 2])?;
            let bank_a = view(topk_scratch, 0, vec![bank_elements])?;
            let bank_b = view(
                topk_scratch,
                checked_product(&[bank_elements, 4])?,
                vec![bank_elements],
            )?;
            let output = view(
                output,
                t.output_byte_offset,
                vec![1, t.rows, output_row_stride],
            )?;
            let ranges = [
                physical_range(&q)?,
                physical_range(&kv)?,
                physical_range(&weights)?,
                physical_range(&scratch)?,
                physical_range(&bank_a)?,
                physical_range(&bank_b)?,
                physical_range(&output)?,
            ];
            Ok(PreparedTile {
                params: DeepSeekIndexerParams {
                    query_len: t.rows as u32,
                    start_pos: t.start_pos,
                    ..*params
                },
                q,
                kv,
                weights,
                scratch,
                bank_a,
                bank_b,
                output,
                ranges,
            })
        },
        || {
            let score = registry
                .get_pipeline(DEEPSEEK_INDEXER_SCORE_MMA_KERNEL, device.metal_device())?
                .clone();
            let block = registry
                .get_pipeline(DEEPSEEK_INDEXER_TOPK_BLOCK_KERNEL, device.metal_device())?
                .clone();
            let merge = registry
                .get_pipeline(DEEPSEEK_INDEXER_TOPK_MERGE_KERNEL, device.metal_device())?
                .clone();
            let finalize = registry
                .get_pipeline(DEEPSEEK_INDEXER_TOPK_FINALIZE_KERNEL, device.metal_device())?
                .clone();
            if block.max_total_threads_per_threadgroup() < 1024
                || score.max_total_threads_per_threadgroup() < 256
                || merge.max_total_threads_per_threadgroup() < 256
                || finalize.max_total_threads_per_threadgroup() < 256
            {
                return Err(invalid("pipeline thread limits do not support indexer"));
            }
            let memory_limit = device.metal_device().max_threadgroup_memory_length();
            for (pipeline, dynamic) in
                [(&score, 0_u64), (&block, 4096), (&merge, 0), (&finalize, 0)]
            {
                if pipeline
                    .static_threadgroup_memory_length()
                    .checked_add(dynamic)
                    .map_or(true, |bytes| bytes > memory_limit)
                {
                    return Err(invalid("pipeline threadgroup memory exceeds device limit"));
                }
            }
            Ok(Pipelines {
                score,
                block,
                merge,
                finalize,
            })
        },
        |tiles, pipelines| {
            let layout = DeepSeekIndexerOutputLayout {
                row_stride: output_row_stride as u32,
                column_offset: output_column_offset as u32,
            };
            for tile in tiles {
                encode::encode(encoder, pipelines, tile, &layout);
            }
        },
    )
}
