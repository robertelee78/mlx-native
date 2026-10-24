use super::super::*;
pub(super) fn invalid(message: &str) -> MlxError {
    MlxError::InvalidArgument(format!("deepseek_indexer_tiles: {message}"))
}
pub(super) fn checked_product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| invalid("shape/byte product overflow"))
    })
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Tile {
    pub origin: usize,
    pub rows: usize,
    pub start_pos: u32,
    pub q_byte_offset: usize,
    pub weight_byte_offset: usize,
    pub output_byte_offset: usize,
}
/// Checked logical workspace payload; excludes complete inputs/output and CPU metadata.
#[derive(Clone, Debug)]
pub struct DeepSeekIndexerQueryTilePlan {
    pub workspace_rows: usize,
    pub score_elements: usize,
    pub topk_elements: usize,
    pub topk_row_elements: usize,
    pub workspace_bytes: usize,
    pub(super) tiles: Vec<Tile>,
}
/// Fixed 256-query tiles preserve the eight-query MMA grouping. Batch one only.
pub fn deepseek_indexer_query_tile_plan(
    p: &DeepSeekIndexerParams,
    stride: usize,
    column: usize,
) -> Result<DeepSeekIndexerQueryTilePlan> {
    if p.batch != 1 {
        return Err(invalid("only batch one is supported"));
    }
    let (_, queries, keys) = validate_params(p)?;
    let tail = column
        .checked_add(512)
        .ok_or_else(|| invalid("output tail overflow"))?;
    if tail > stride || u32::try_from(stride).is_err() || u32::try_from(column).is_err() {
        return Err(invalid("invalid output layout"));
    }
    // Validate COMPLETE extents even though individual shader grids are tiled.
    checked_product(&[queries, 64, 128, 2])?;
    checked_product(&[queries, 64, 4])?;
    checked_product(&[queries, stride, 4])?;
    let rows = queries.min(256);
    let score_elements = checked_product(&[rows, keys])?;
    let topk_elements = deepseek_indexer_topk_scratch_elements(1, rows, keys)?;
    let workspace_bytes = score_elements
        .checked_add(topk_elements)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("workspace byte overflow"))?;
    let count = queries.div_ceil(256);
    let mut tiles = Vec::new();
    tiles
        .try_reserve_exact(count)
        .map_err(|_| invalid("plan metadata allocation failed"))?;
    for origin in (0..queries).step_by(256) {
        let n = (queries - origin).min(256);
        let start_pos = p
            .start_pos
            .checked_add(origin as u32)
            .ok_or_else(|| invalid("tile position overflow"))?;
        start_pos
            .checked_add(n as u32)
            .ok_or_else(|| invalid("tile end overflow"))?;
        tiles.push(Tile {
            origin,
            rows: n,
            start_pos,
            q_byte_offset: checked_product(&[origin, 64, 128, 2])?,
            weight_byte_offset: checked_product(&[origin, 64, 4])?,
            output_byte_offset: checked_product(&[origin, stride, 4])?,
        });
    }
    Ok(DeepSeekIndexerQueryTilePlan {
        workspace_rows: rows,
        score_elements,
        topk_elements,
        topk_row_elements: topk_elements / rows,
        workspace_bytes,
        tiles,
    })
}
