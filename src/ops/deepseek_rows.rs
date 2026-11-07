//! DeepSeek-V4 0731 decode cores over independent rows, for joint decode.
//!
//! Joint decode advances up to eight independent requests by one token each.
//! Every request owns separate cache buffers, so the existing one-row kernels
//! run once per row. The dispatches here run those kernels' one-row bodies
//! for all rows at once: row `r` reads its cache through binding slot `r`
//! (unused slots rebind row 0) and its shape from a per-row constant table.
//! Row `r`'s threadgroups execute the one-row kernel's code with that row's
//! loop bounds, tiling, reduction order and tie order and read only row
//! `r`'s bindings, so every row of a rows dispatch is bitwise equal to the
//! existing one-row dispatch on that row (tests/deepseek_rows.rs).
//!
//! - [`dispatch_deepseek_sparse_attention_rows`]: selected sparse attention
//!   (score + reduce) at a per-row width; the one-row
//!   `dispatch_deepseek_sparse_attention_with_scratch`.
//! - [`dispatch_deepseek_sparse_attention_flash_rows`]: gather + the D=512
//!   flash kernel + sanitize; the one-row
//!   `dispatch_deepseek_sparse_attention_flash_decode`. The flash kernel takes
//!   one key length per dispatch, so rows run as one flash batch per distinct
//!   width (no padding).
//! - [`dispatch_deepseek_indexer_rows`]: lightning-indexer MMA score and the
//!   deterministic top-512 (block sort, merge tree, finalize); the one-row
//!   `dispatch_deepseek_indexer_into_mma_tiled` at one query.
//! - [`dispatch_deepseek_compressor_rows`]: one decode token of the learned KV
//!   compressor per row, each with its own recurrent state; the one-row
//!   `dispatch_deepseek_compressor` at `seq_len = 1`, nonzero position,
//!   `write_cache = 0`.
//! - [`dispatch_deepseek_rows_gather_f32`]: copy B separate one-row F32
//!   buffers into one `[B, n]` operand (exact copy).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CapturedOpKind, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::deepseek_sparse_attention::{
    DeepSeekSparseAttentionParams, DEEPSEEK_SPARSE_ATTENTION_SANITIZE_KERNEL,
    DEEPSEEK_SPARSE_ATTENTION_VALIDATE_Q_KERNEL, DEEPSEEK_SPARSE_HEADS, DEEPSEEK_SPARSE_HEAD_DIM,
};
use crate::ops::flash_attn_prefill::FlashAttnPrefillParams;
use crate::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_bf16_d512_heads_as_rows_with_sinks;

/// Most rows one rows dispatch accepts (binding slots per cache family).
pub const DEEPSEEK_ROWS_MAX: usize = 8;

pub const DEEPSEEK_SPARSE_ATTENTION_SCORE_ROWS_KERNEL: &str =
    "deepseek_sparse_attention_score_rows_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_REDUCE_ROWS_KERNEL: &str =
    "deepseek_sparse_attention_reduce_rows_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_GATHER_ROWS_KERNEL: &str =
    "deepseek_sparse_attention_gather_rows_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_PERMUTE_Q_ROWS_KERNEL: &str =
    "deepseek_sparse_attention_permute_q_rows_bf16";
pub const DEEPSEEK_SPARSE_ATTENTION_SANITIZE_ROWS_KERNEL: &str =
    "deepseek_sparse_attention_sanitize_rows_bf16";
pub const DEEPSEEK_INDEXER_SCORE_MMA_ROWS_KERNEL: &str = "deepseek_indexer_score_mma_rows_bf16";
pub const DEEPSEEK_INDEXER_TOPK_BLOCK_ROWS_KERNEL: &str = "deepseek_indexer_topk_block_rows_i32";
pub const DEEPSEEK_INDEXER_TOPK_MERGE_ROWS_KERNEL: &str = "deepseek_indexer_topk_merge_rows_i32";
pub const DEEPSEEK_INDEXER_TOPK_FINALIZE_ROWS_KERNEL: &str =
    "deepseek_indexer_topk_finalize_rows_i32";
pub const DEEPSEEK_COMPRESSOR_ROWS_KERNEL: &str = "deepseek_compressor_rows_bf16";
pub const DEEPSEEK_ROWS_GATHER_F32_KERNEL: &str = "deepseek_rows_gather_f32";

pub static DEEPSEEK_ROWS_SHADER_SOURCE: &str = include_str!("../shaders/deepseek_rows.metal");

const THREADS: u64 = 256;
const SPARSE_SLOTS_PER_GROUP: usize = 32;
const INDEXER_HEADS: usize = 64;
const INDEXER_DIM: usize = 128;
const INDEXER_TOP_K: usize = 512;
const INDEXER_KEYS_PER_GROUP: usize = 64;
const INDEXER_BLOCK_THREADS: usize = 1024;
const INDEXER_MERGE_THREADS: u64 = 256;
const SPARSE_ROW_ELEMENTS: usize = DEEPSEEK_SPARSE_HEADS * DEEPSEEK_SPARSE_HEAD_DIM;

/// Every kernel in [`DEEPSEEK_ROWS_SHADER_SOURCE`].
pub fn deepseek_rows_kernel_names() -> [&'static str; 11] {
    [
        DEEPSEEK_SPARSE_ATTENTION_SCORE_ROWS_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_REDUCE_ROWS_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_GATHER_ROWS_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_PERMUTE_Q_ROWS_KERNEL,
        DEEPSEEK_SPARSE_ATTENTION_SANITIZE_ROWS_KERNEL,
        DEEPSEEK_INDEXER_SCORE_MMA_ROWS_KERNEL,
        DEEPSEEK_INDEXER_TOPK_BLOCK_ROWS_KERNEL,
        DEEPSEEK_INDEXER_TOPK_MERGE_ROWS_KERNEL,
        DEEPSEEK_INDEXER_TOPK_FINALIZE_ROWS_KERNEL,
        DEEPSEEK_COMPRESSOR_ROWS_KERNEL,
        DEEPSEEK_ROWS_GATHER_F32_KERNEL,
    ]
}

/// Register the rows kernels' source (the `KernelRegistry` constructor
/// already carries it; this is for registries built without it).
pub fn register(registry: &mut KernelRegistry) {
    for name in deepseek_rows_kernel_names() {
        registry.register_source(name, DEEPSEEK_ROWS_SHADER_SOURCE);
    }
}

fn invalid(message: impl Into<String>) -> MlxError {
    MlxError::InvalidArgument(format!("deepseek_rows: {}", message.into()))
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{what} exceeds u32")))
}

fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("shape product overflows: {dims:?}")))
    })
}

/// `buffer` must be `dtype` and hold at least `elements` logical elements.
fn require(buffer: &MlxBuffer, name: &str, dtype: DType, elements: usize) -> Result<()> {
    if buffer.dtype() != dtype {
        return Err(invalid(format!(
            "{name} must be {dtype}, got {}",
            buffer.dtype()
        )));
    }
    let bytes = product(&[elements, dtype.size_of()])?;
    if buffer.data_byte_len() < bytes {
        return Err(invalid(format!(
            "{name} holds {} bytes, needs {bytes}",
            buffer.data_byte_len()
        )));
    }
    Ok(())
}

fn range(buffer: &MlxBuffer) -> (usize, usize) {
    let start = buffer.contents_ptr() as usize + buffer.byte_offset() as usize;
    (start, start + buffer.data_byte_len())
}

fn capture(
    encoder: &mut CommandEncoder,
    kind: CapturedOpKind,
    reads: Vec<(usize, usize)>,
    writes: Vec<(usize, usize)>,
) {
    encoder.set_op_kind(kind);
    if encoder.is_capturing() {
        encoder.set_pending_buffer_ranges(reads, writes);
    }
}

fn check_rows(rows: usize) -> Result<()> {
    if rows == 0 || rows > DEEPSEEK_ROWS_MAX {
        return Err(invalid(format!(
            "rows must be 1..={DEEPSEEK_ROWS_MAX}, got {rows}"
        )));
    }
    Ok(())
}

/// Per-row binding slots: row `r` at slot `r`, unused slots rebind row 0.
fn slots<'a>(buffers: &[&'a MlxBuffer]) -> [&'a MlxBuffer; DEEPSEEK_ROWS_MAX] {
    std::array::from_fn(|slot| buffers.get(slot).copied().unwrap_or(buffers[0]))
}

fn push_slots<'a>(
    args: &mut Vec<(u64, KernelArg<'a>)>,
    first: u64,
    buffers: &[&'a MlxBuffer; DEEPSEEK_ROWS_MAX],
) {
    for (slot, buffer) in buffers.iter().enumerate() {
        args.push((first + slot as u64, KernelArg::Buffer(buffer)));
    }
}

fn disjoint(ranges: &[(usize, usize)], what: &str) -> Result<()> {
    for (a, lhs) in ranges.iter().enumerate() {
        for rhs in &ranges[..a] {
            if lhs.0 < rhs.1 && rhs.0 < lhs.1 {
                return Err(invalid(format!("{what} of two rows overlap")));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sparse attention
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SparseRowsParams {
    rows: u32,
    max_width: u32,
    scale: f32,
    reserved: u32,
    kv_len: [u32; DEEPSEEK_ROWS_MAX],
    width: [u32; DEEPSEEK_ROWS_MAX],
    index_base: [u32; DEEPSEEK_ROWS_MAX],
    gathered_base: [u32; DEEPSEEK_ROWS_MAX],
    mask_base: [u32; DEEPSEEK_ROWS_MAX],
    slot: [u32; DEEPSEEK_ROWS_MAX],
    q_row: [u32; DEEPSEEK_ROWS_MAX],
}

/// One row of a sparse-attention rows dispatch.
#[derive(Clone, Copy, Debug)]
pub struct DeepSeekSparseAttentionRow<'a> {
    /// The row's BF16 shared KV, at least `kv_len * 512` elements (e.g. the
    /// row's window or attention cache).
    pub kv: &'a MlxBuffer,
    pub kv_len: usize,
    /// Selected slots; the one-row kernel's `top_k`.
    pub width: usize,
    /// Element offset of the row's `width` I32 indices in `indices`.
    pub index_offset: usize,
    /// Row of the `[_, 64, 512]` query and output buffers this entry uses.
    pub q_row: usize,
}

fn sparse_params(rows: &[DeepSeekSparseAttentionRow<'_>], scale: f32) -> Result<SparseRowsParams> {
    check_rows(rows.len())?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid("scale must be finite and positive"));
    }
    let mut params = SparseRowsParams {
        rows: rows.len() as u32,
        scale,
        ..Default::default()
    };
    for (entry, row) in rows.iter().enumerate() {
        if row.kv_len == 0 || row.width == 0 {
            return Err(invalid(format!(
                "row {entry}: kv_len and width must be nonzero"
            )));
        }
        require(
            row.kv,
            "kv",
            DType::BF16,
            product(&[row.kv_len, DEEPSEEK_SPARSE_HEAD_DIM])?,
        )?;
        if rows[..entry].iter().any(|other| other.q_row == row.q_row) {
            return Err(invalid(format!(
                "row {entry}: q_row {} is used twice",
                row.q_row
            )));
        }
        params.kv_len[entry] = to_u32(row.kv_len, "kv_len")?;
        params.width[entry] = to_u32(row.width, "width")?;
        params.index_base[entry] = to_u32(row.index_offset, "index offset")?;
        params.q_row[entry] = to_u32(row.q_row, "q_row")?;
        params.max_width = params.max_width.max(params.width[entry]);
    }
    Ok(params)
}

fn sparse_io_checks(
    rows: &[DeepSeekSparseAttentionRow<'_>],
    q: &MlxBuffer,
    indices: &MlxBuffer,
    output: &MlxBuffer,
) -> Result<()> {
    let q_rows = rows.iter().map(|row| row.q_row).max().unwrap_or(0) + 1;
    require(
        q,
        "q",
        DType::BF16,
        product(&[q_rows, SPARSE_ROW_ELEMENTS])?,
    )?;
    require(
        output,
        "output",
        DType::BF16,
        product(&[q_rows, SPARSE_ROW_ELEMENTS])?,
    )?;
    let index_end = rows
        .iter()
        .map(|row| row.index_offset.checked_add(row.width))
        .try_fold(0usize, |end, row_end| {
            row_end.map(|row_end| end.max(row_end))
        })
        .ok_or_else(|| invalid("index range overflows"))?;
    require(indices, "indices", DType::I32, index_end)
}

/// F32 logits required by [`dispatch_deepseek_sparse_attention_rows`].
pub fn deepseek_sparse_attention_rows_scratch_elements(rows: usize, max_width: usize) -> usize {
    rows * DEEPSEEK_SPARSE_HEADS * max_width
}

/// Sparse attention (score then reduce) for each row at its own width.
///
/// `q`/`output` are BF16 `[_, 64, 512]` (entry `e` uses row `rows[e].q_row`),
/// `sinks` F32 `[64]`, `indices` I32 holding every row's `width` indices at
/// its offset, and `scratch` F32 with at least
/// [`deepseek_sparse_attention_rows_scratch_elements`]. Each row equals
/// `dispatch_deepseek_sparse_attention_with_scratch` at batch 1, one query,
/// `kv_len` and `top_k = width` bitwise, including its fail-closed zeroing.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_sparse_attention_rows(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    scratch: &MlxBuffer,
    output: &MlxBuffer,
    rows: &[DeepSeekSparseAttentionRow<'_>],
    scale: f32,
) -> Result<()> {
    let params = sparse_params(rows, scale)?;
    sparse_io_checks(rows, q, indices, output)?;
    require(sinks, "sinks", DType::F32, DEEPSEEK_SPARSE_HEADS)?;
    let max_width = params.max_width as usize;
    require(
        scratch,
        "scratch",
        DType::F32,
        deepseek_sparse_attention_rows_scratch_elements(rows.len(), max_width),
    )?;
    let kv: Vec<&MlxBuffer> = rows.iter().map(|row| row.kv).collect();
    let kv = slots(&kv);
    let kv_ranges: Vec<_> = rows.iter().map(|row| range(row.kv)).collect();
    let score = registry
        .get_pipeline(
            DEEPSEEK_SPARSE_ATTENTION_SCORE_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();
    let reduce = registry
        .get_pipeline(
            DEEPSEEK_SPARSE_ATTENTION_REDUCE_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();

    let head_groups = rows.len() * DEEPSEEK_SPARSE_HEADS;
    let blocks_per_head = max_width.div_ceil(SPARSE_SLOTS_PER_GROUP);
    let mut reads = vec![range(q), range(indices)];
    reads.extend(&kv_ranges);
    capture(encoder, CapturedOpKind::Sdpa, reads, vec![range(scratch)]);
    let mut args = vec![
        (0, KernelArg::Bytes(as_bytes(&params))),
        (1, KernelArg::Buffer(q)),
        (2, KernelArg::Buffer(indices)),
        (3, KernelArg::Buffer(scratch)),
    ];
    push_slots(&mut args, 4, &kv);
    encoder.encode_threadgroups_with_args(
        &score,
        &args,
        MTLSize::new((head_groups * blocks_per_head) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    encoder.memory_barrier();

    let mut reads = vec![range(sinks), range(indices), range(scratch)];
    reads.extend(&kv_ranges);
    capture(
        encoder,
        CapturedOpKind::Sdpa,
        reads,
        vec![range(output), range(scratch)],
    );
    let mut args = vec![
        (0, KernelArg::Bytes(as_bytes(&params))),
        (1, KernelArg::Buffer(sinks)),
        (2, KernelArg::Buffer(indices)),
        (3, KernelArg::Buffer(scratch)),
        (4, KernelArg::Buffer(output)),
    ];
    push_slots(&mut args, 5, &kv);
    encoder.encode_threadgroups_with_args(
        &reduce,
        &args,
        MTLSize::new(head_groups as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}

/// Caller-owned buffers of [`dispatch_deepseek_sparse_attention_flash_rows`].
pub struct DeepSeekSparseFlashRowsWorkspace<'a> {
    /// BF16, at least `sum(width) * 512` elements.
    pub gathered_kv: &'a MlxBuffer,
    /// BF16, at least `sum(width)` elements.
    pub mask: &'a MlxBuffer,
    /// U32, at least `rows` elements, zeroed before encoding.
    pub invalid_global: &'a MlxBuffer,
    /// U32, at least `rows * 64` elements, zeroed before encoding.
    pub invalid_heads: &'a MlxBuffer,
    /// BF16, at least `rows * 64 * 512` elements; read only when the rows'
    /// flash batches are not already contiguous in `q` (see
    /// [`deepseek_sparse_flash_rows_contiguous`]).
    pub q_scratch: &'a MlxBuffer,
    /// BF16, at least `rows * 64 * 512` elements; same condition.
    pub output_scratch: &'a MlxBuffer,
}

/// Flash-batch placement: entries grouped by width in first-appearance
/// order, entry order inside a group. Returns (slot per entry, groups as
/// `(first slot, entries, width)`).
fn flash_groups(
    rows: &[DeepSeekSparseAttentionRow<'_>],
) -> (Vec<usize>, Vec<(usize, usize, usize)>) {
    let mut widths: Vec<usize> = Vec::new();
    for row in rows {
        if !widths.contains(&row.width) {
            widths.push(row.width);
        }
    }
    let mut slot = vec![0; rows.len()];
    let mut groups = Vec::with_capacity(widths.len());
    let mut next = 0;
    for width in widths {
        let first = next;
        for (entry, row) in rows.iter().enumerate() {
            if row.width == width {
                slot[entry] = next;
                next += 1;
            }
        }
        groups.push((first, next - first, width));
    }
    (slot, groups)
}

/// Whether the rows' flash batches read `q` and write `output` in place:
/// entry `e` is query row `e` and equal widths form one contiguous run each.
pub fn deepseek_sparse_flash_rows_contiguous(rows: &[DeepSeekSparseAttentionRow<'_>]) -> bool {
    let (slot, _) = flash_groups(rows);
    rows.iter()
        .enumerate()
        .all(|(entry, row)| row.q_row == entry && slot[entry] == entry)
}

/// The scalable gather + D=512 flash sparse-attention path for each row at
/// its own width: gather every row's selected KV into `gathered_kv`, run the
/// flash kernel once per distinct width over that width's rows, and sanitize.
///
/// Inputs as [`dispatch_deepseek_sparse_attention_rows`]. Each row equals
/// `dispatch_deepseek_sparse_attention_flash_decode` at `kv_len` and
/// `top_k = width` bitwise, including its fail-closed zeroing.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_sparse_attention_flash_rows(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    sinks: &MlxBuffer,
    indices: &MlxBuffer,
    workspace: &DeepSeekSparseFlashRowsWorkspace<'_>,
    output: &MlxBuffer,
    rows: &[DeepSeekSparseAttentionRow<'_>],
    scale: f32,
) -> Result<()> {
    let mut params = sparse_params(rows, scale)?;
    sparse_io_checks(rows, q, indices, output)?;
    require(sinks, "sinks", DType::F32, DEEPSEEK_SPARSE_HEADS)?;
    let n = rows.len();
    let (slot, groups) = flash_groups(rows);
    let contiguous = deepseek_sparse_flash_rows_contiguous(rows);
    let mut gathered_elements = 0usize;
    let mut mask_elements = 0usize;
    let mut group_base = Vec::with_capacity(groups.len());
    for &(first, count, width) in &groups {
        group_base.push((gathered_elements, mask_elements));
        for (entry, row) in rows.iter().enumerate() {
            if row.width == width {
                let rank = slot[entry] - first;
                params.gathered_base[entry] = to_u32(
                    gathered_elements + rank * width * DEEPSEEK_SPARSE_HEAD_DIM,
                    "gathered offset",
                )?;
                params.mask_base[entry] = to_u32(mask_elements + rank * width, "mask offset")?;
                params.slot[entry] = slot[entry] as u32;
            }
        }
        gathered_elements += count * width * DEEPSEEK_SPARSE_HEAD_DIM;
        mask_elements += count * width;
    }
    to_u32(gathered_elements, "gathered elements")?;
    require(
        workspace.gathered_kv,
        "gathered_kv",
        DType::BF16,
        gathered_elements,
    )?;
    require(workspace.mask, "mask", DType::BF16, mask_elements)?;
    require(workspace.invalid_global, "invalid_global", DType::U32, n)?;
    require(
        workspace.invalid_heads,
        "invalid_heads",
        DType::U32,
        n * DEEPSEEK_SPARSE_HEADS,
    )?;
    let row_bytes = SPARSE_ROW_ELEMENTS * DType::BF16.size_of();
    let (flash_q, flash_out) = if contiguous {
        (q, output)
    } else {
        require(
            workspace.q_scratch,
            "q_scratch",
            DType::BF16,
            n * SPARSE_ROW_ELEMENTS,
        )?;
        require(
            workspace.output_scratch,
            "output_scratch",
            DType::BF16,
            n * SPARSE_ROW_ELEMENTS,
        )?;
        (workspace.q_scratch, workspace.output_scratch)
    };
    // Flash views: q/output rows of each group, its gathered KV and mask.
    let mut flash_views = Vec::with_capacity(groups.len());
    for (&(first, count, width), &(gathered, mask)) in groups.iter().zip(&group_base) {
        let rows_elements = count * SPARSE_ROW_ELEMENTS;
        let q_view = flash_q
            .slice_view((first * row_bytes) as u64, rows_elements)
            .with_shape(vec![count, DEEPSEEK_SPARSE_HEADS, DEEPSEEK_SPARSE_HEAD_DIM])?;
        let out_view = flash_out
            .slice_view((first * row_bytes) as u64, rows_elements)
            .with_shape(vec![count, DEEPSEEK_SPARSE_HEADS, DEEPSEEK_SPARSE_HEAD_DIM])?;
        let kv_view = workspace
            .gathered_kv
            .slice_view(
                (gathered * 2) as u64,
                count * width * DEEPSEEK_SPARSE_HEAD_DIM,
            )
            .with_shape(vec![count, 1, width, DEEPSEEK_SPARSE_HEAD_DIM])?;
        let mask_view = workspace
            .mask
            .slice_view((mask * 2) as u64, count * width)
            .with_shape(vec![count, 1, width])?;
        flash_views.push((count, width, q_view, kv_view, mask_view, out_view));
    }
    let one_row = |batch: usize| DeepSeekSparseAttentionParams {
        batch: batch as u32,
        query_len: 1,
        kv_len: 1,
        top_k: 1,
        heads: DEEPSEEK_SPARSE_HEADS as u32,
        head_dim: DEEPSEEK_SPARSE_HEAD_DIM as u32,
        scale,
    };
    let validate = registry
        .get_pipeline(
            DEEPSEEK_SPARSE_ATTENTION_VALIDATE_Q_KERNEL,
            device.metal_device(),
        )?
        .clone();
    let gather = registry
        .get_pipeline(
            DEEPSEEK_SPARSE_ATTENTION_GATHER_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();
    let permute = registry
        .get_pipeline(
            DEEPSEEK_SPARSE_ATTENTION_PERMUTE_Q_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();
    let sanitize = registry
        .get_pipeline(
            if contiguous {
                DEEPSEEK_SPARSE_ATTENTION_SANITIZE_KERNEL
            } else {
                DEEPSEEK_SPARSE_ATTENTION_SANITIZE_ROWS_KERNEL
            },
            device.metal_device(),
        )?
        .clone();
    let kv: Vec<&MlxBuffer> = rows.iter().map(|row| row.kv).collect();
    let kv = slots(&kv);
    let kv_ranges: Vec<_> = rows.iter().map(|row| range(row.kv)).collect();
    let row_elements_total = n * SPARSE_ROW_ELEMENTS;

    if !contiguous {
        capture(
            encoder,
            CapturedOpKind::Sdpa,
            vec![range(q)],
            vec![range(workspace.q_scratch)],
        );
        encoder.encode_threadgroups_with_args(
            &permute,
            &[
                (0, KernelArg::Bytes(as_bytes(&params))),
                (1, KernelArg::Buffer(q)),
                (2, KernelArg::Buffer(workspace.q_scratch)),
            ],
            MTLSize::new(row_elements_total.div_ceil(THREADS as usize) as u64, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
        encoder.memory_barrier();
    }

    let validate_params = one_row(n);
    capture(
        encoder,
        CapturedOpKind::Sdpa,
        vec![range(flash_q), range(sinks)],
        vec![range(workspace.invalid_heads)],
    );
    encoder.encode_threadgroups_with_args(
        &validate,
        &[
            (0, KernelArg::Bytes(as_bytes(&validate_params))),
            (1, KernelArg::Buffer(flash_q)),
            (2, KernelArg::Buffer(sinks)),
            (3, KernelArg::Buffer(workspace.invalid_heads)),
        ],
        MTLSize::new((n * DEEPSEEK_SPARSE_HEADS) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );

    let mut reads = vec![range(indices)];
    reads.extend(&kv_ranges);
    capture(
        encoder,
        CapturedOpKind::Sdpa,
        reads,
        vec![
            range(workspace.gathered_kv),
            range(workspace.mask),
            range(workspace.invalid_global),
        ],
    );
    let mut args = vec![
        (0, KernelArg::Bytes(as_bytes(&params))),
        (1, KernelArg::Buffer(indices)),
        (2, KernelArg::Buffer(workspace.gathered_kv)),
        (3, KernelArg::Buffer(workspace.mask)),
        (4, KernelArg::Buffer(workspace.invalid_global)),
    ];
    push_slots(&mut args, 5, &kv);
    encoder.encode_threadgroups_with_args(
        &gather,
        &args,
        MTLSize::new(gathered_elements.div_ceil(THREADS as usize) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    encoder.memory_barrier();

    for (count, width, q_view, kv_view, mask_view, out_view) in &flash_views {
        dispatch_flash_attn_prefill_bf16_d512_heads_as_rows_with_sinks(
            encoder,
            device,
            registry,
            q_view,
            kv_view,
            kv_view,
            mask_view,
            sinks,
            out_view,
            &FlashAttnPrefillParams {
                n_heads: 8,
                n_kv_heads: 1,
                head_dim: DEEPSEEK_SPARSE_HEAD_DIM as u32,
                seq_len_q: 8,
                seq_len_k: *width as u32,
                batch: *count as u32,
                scale,
                do_causal: false,
            },
        )?;
    }
    encoder.memory_barrier();

    if contiguous {
        let sanitize_params = one_row(n);
        capture(
            encoder,
            CapturedOpKind::Sdpa,
            vec![
                range(workspace.invalid_global),
                range(workspace.invalid_heads),
                range(output),
            ],
            vec![range(output)],
        );
        encoder.encode_threadgroups_with_args(
            &sanitize,
            &[
                (0, KernelArg::Bytes(as_bytes(&sanitize_params))),
                (1, KernelArg::Buffer(workspace.invalid_global)),
                (2, KernelArg::Buffer(workspace.invalid_heads)),
                (3, KernelArg::Buffer(output)),
            ],
            MTLSize::new(row_elements_total.div_ceil(THREADS as usize) as u64, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
    } else {
        capture(
            encoder,
            CapturedOpKind::Sdpa,
            vec![
                range(workspace.invalid_global),
                range(workspace.invalid_heads),
                range(workspace.output_scratch),
            ],
            vec![range(output)],
        );
        encoder.encode_threadgroups_with_args(
            &sanitize,
            &[
                (0, KernelArg::Bytes(as_bytes(&params))),
                (1, KernelArg::Buffer(workspace.invalid_global)),
                (2, KernelArg::Buffer(workspace.invalid_heads)),
                (3, KernelArg::Buffer(workspace.output_scratch)),
                (4, KernelArg::Buffer(output)),
            ],
            MTLSize::new(row_elements_total.div_ceil(THREADS as usize) as u64, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
    }
    encoder.memory_barrier();
    Ok(())
}

// ---------------------------------------------------------------------------
// Lightning indexer
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct IndexerRowsParams {
    rows: u32,
    query_len: u32,
    ratio: u32,
    score_stride: u32,
    max_blocks: u32,
    merge_round: u32,
    reserved0: u32,
    reserved1: u32,
    kv_len: [u32; DEEPSEEK_ROWS_MAX],
    start_pos: [u32; DEEPSEEK_ROWS_MAX],
    offset: [i32; DEEPSEEK_ROWS_MAX],
    q_row: [u32; DEEPSEEK_ROWS_MAX],
    output_base: [u32; DEEPSEEK_ROWS_MAX],
}

/// One row (one query) of an indexer rows dispatch.
#[derive(Clone, Copy, Debug)]
pub struct DeepSeekIndexerRow<'a> {
    /// The row's BF16 indexer cache, at least `kv_len * 128` elements.
    pub kv: &'a MlxBuffer,
    /// Valid compressed keys (the one-row dispatch's `kv_len`), nonzero.
    pub kv_len: usize,
    /// The query's absolute position (the one-row `start_pos`).
    pub start_pos: u32,
    /// Added to every emitted index (the one-row `offset`).
    pub offset: i32,
    /// Row of the `[_, 64, 128]` query and `[_, 64]` head weights.
    pub q_row: usize,
    /// Element offset in `output` of the row's 512 output indices (the
    /// one-row `column_offset` within the row's output).
    pub output_offset: usize,
}

/// `(scores F32, top-k scratch I32)` elements required by
/// [`dispatch_deepseek_indexer_rows`] for `rows` rows of at most
/// `max_kv_len` keys.
pub fn deepseek_indexer_rows_workspace_elements(rows: usize, max_kv_len: usize) -> (usize, usize) {
    let blocks = max_kv_len.div_ceil(INDEXER_BLOCK_THREADS);
    (rows * max_kv_len, 2 * rows * blocks * INDEXER_TOP_K)
}

/// Indexer score (simdgroup MMA) and deterministic causal top-512 for one
/// query per row against the row's own indexer cache.
///
/// `q` is BF16 `[_, 64, 128]`, `weights` F32 `[_, 64]` (entry `e` uses row
/// `rows[e].q_row`); `scores`/`topk` hold at least
/// [`deepseek_indexer_rows_workspace_elements`]. Each row's 512 indices equal
/// `dispatch_deepseek_indexer_into_mma_tiled` at batch 1, one query, ratio 4
/// bitwise, including the tie order and `-1` for unfilled slots.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_indexer_rows(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q: &MlxBuffer,
    weights: &MlxBuffer,
    scores: &MlxBuffer,
    topk: &MlxBuffer,
    output: &MlxBuffer,
    rows: &[DeepSeekIndexerRow<'_>],
) -> Result<()> {
    check_rows(rows.len())?;
    let mut params = IndexerRowsParams {
        rows: rows.len() as u32,
        query_len: 1,
        ratio: 4,
        ..Default::default()
    };
    let mut max_kv = 0usize;
    let mut output_ranges = Vec::with_capacity(rows.len());
    for (entry, row) in rows.iter().enumerate() {
        if row.kv_len == 0 {
            return Err(invalid(format!(
                "indexer row {entry}: kv_len must be nonzero"
            )));
        }
        require(
            row.kv,
            "indexer kv",
            DType::BF16,
            product(&[row.kv_len, INDEXER_DIM])?,
        )?;
        let causal = row
            .start_pos
            .checked_add(1)
            .ok_or_else(|| invalid("indexer position overflows"))?
            / 4;
        if (row.kv_len as u64) < u64::from(causal) {
            return Err(invalid(format!(
                "indexer row {entry}: kv_len {} is shorter than {causal} causal compressed entries",
                row.kv_len
            )));
        }
        // As the one-row indexer: -1 is the sentinel, and every emitted
        // `index + offset` must fit i32.
        if row.offset < 0 {
            return Err(invalid(format!(
                "indexer row {entry}: offset must be nonnegative; -1 is reserved for the sentinel"
            )));
        }
        if causal > 0 && i64::from(row.offset) + i64::from(causal) - 1 > i64::from(i32::MAX) {
            return Err(invalid(format!(
                "indexer row {entry}: offset plus compressed index exceeds i32"
            )));
        }
        params.kv_len[entry] = to_u32(row.kv_len, "indexer kv_len")?;
        params.start_pos[entry] = row.start_pos;
        params.offset[entry] = row.offset;
        params.q_row[entry] = to_u32(row.q_row, "indexer q_row")?;
        params.output_base[entry] = to_u32(row.output_offset, "indexer output offset")?;
        output_ranges.push((row.output_offset, row.output_offset + INDEXER_TOP_K));
        max_kv = max_kv.max(row.kv_len);
    }
    disjoint(&output_ranges, "indexer outputs")?;
    let blocks = max_kv.div_ceil(INDEXER_BLOCK_THREADS);
    params.score_stride = to_u32(max_kv, "indexer keys")?;
    params.max_blocks = blocks as u32;
    let q_rows = rows.iter().map(|row| row.q_row).max().unwrap_or(0) + 1;
    require(
        q,
        "indexer q",
        DType::BF16,
        product(&[q_rows, INDEXER_HEADS, INDEXER_DIM])?,
    )?;
    require(
        weights,
        "indexer weights",
        DType::F32,
        product(&[q_rows, INDEXER_HEADS])?,
    )?;
    let (score_elements, topk_elements) =
        deepseek_indexer_rows_workspace_elements(rows.len(), max_kv);
    require(scores, "indexer scores", DType::F32, score_elements)?;
    require(topk, "indexer top-k scratch", DType::I32, topk_elements)?;
    let output_end = output_ranges.iter().map(|range| range.1).max().unwrap_or(0);
    require(output, "indexer output", DType::I32, output_end)?;
    let bank_elements = topk_elements / 2;
    let bank_a = topk.slice_view(0, bank_elements);
    let bank_b = topk.slice_view((bank_elements * 4) as u64, bank_elements);

    let score = registry
        .get_pipeline(
            DEEPSEEK_INDEXER_SCORE_MMA_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();
    let block = registry
        .get_pipeline(
            DEEPSEEK_INDEXER_TOPK_BLOCK_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();
    let merge = registry
        .get_pipeline(
            DEEPSEEK_INDEXER_TOPK_MERGE_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();
    let finalize = registry
        .get_pipeline(
            DEEPSEEK_INDEXER_TOPK_FINALIZE_ROWS_KERNEL,
            device.metal_device(),
        )?
        .clone();
    if block.max_total_threads_per_threadgroup() < INDEXER_BLOCK_THREADS as u64 {
        return Err(invalid("pipeline thread limits do not support the indexer"));
    }
    let kv: Vec<&MlxBuffer> = rows.iter().map(|row| row.kv).collect();
    let kv = slots(&kv);
    let mut reads = vec![range(q), range(weights)];
    reads.extend(rows.iter().map(|row| range(row.kv)));
    capture(encoder, CapturedOpKind::Other, reads, vec![range(scores)]);
    let mut args = vec![
        (0, KernelArg::Bytes(as_bytes(&params))),
        (1, KernelArg::Buffer(q)),
        (2, KernelArg::Buffer(weights)),
        (3, KernelArg::Buffer(scores)),
    ];
    push_slots(&mut args, 4, &kv);
    encoder.encode_threadgroups_with_args(
        &score,
        &args,
        MTLSize::new(
            max_kv.div_ceil(INDEXER_KEYS_PER_GROUP) as u64,
            1,
            rows.len() as u64,
        ),
        MTLSize::new(THREADS, 1, 1),
    );
    encoder.memory_barrier();

    capture(
        encoder,
        CapturedOpKind::Other,
        vec![range(scores)],
        vec![range(&bank_a)],
    );
    encoder.encode_threadgroups_with_args_and_shared(
        &block,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(scores)),
            (2, KernelArg::Buffer(&bank_a)),
        ],
        &[(0, (INDEXER_BLOCK_THREADS * 4) as u64)],
        MTLSize::new((rows.len() * blocks) as u64, 1, 1),
        MTLSize::new(INDEXER_BLOCK_THREADS as u64, 1, 1),
    );
    encoder.memory_barrier();

    let mut input = &bank_a;
    let mut output_bank = &bank_b;
    let mut lists = blocks;
    let mut round = 0u32;
    while lists > 1 {
        let merged = lists.div_ceil(2);
        let merge_params = IndexerRowsParams {
            merge_round: round,
            ..params
        };
        capture(
            encoder,
            CapturedOpKind::Other,
            vec![range(scores), range(input)],
            vec![range(output_bank)],
        );
        encoder.encode_threadgroups_with_args(
            &merge,
            &[
                (0, KernelArg::Bytes(as_bytes(&merge_params))),
                (1, KernelArg::Buffer(scores)),
                (2, KernelArg::Buffer(input)),
                (3, KernelArg::Buffer(output_bank)),
            ],
            MTLSize::new((rows.len() * merged) as u64, 1, 1),
            MTLSize::new(INDEXER_MERGE_THREADS, 1, 1),
        );
        encoder.memory_barrier();
        std::mem::swap(&mut input, &mut output_bank);
        lists = merged;
        round += 1;
    }

    capture(
        encoder,
        CapturedOpKind::Other,
        vec![range(input)],
        vec![range(output)],
    );
    encoder.encode_with_args(
        &finalize,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Buffer(input)),
            (2, KernelArg::Buffer(output)),
        ],
        MTLSize::new(INDEXER_TOP_K as u64, rows.len() as u64, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    encoder.memory_barrier();
    Ok(())
}

// ---------------------------------------------------------------------------
// Row gather
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RowsGatherParams {
    rows: u32,
    row_elements: u32,
}

/// Copy `sources[r]` (F32, at least `row_elements` each) to row `r` of
/// `output` (F32 `[sources.len(), row_elements]`) in one dispatch, bit for
/// bit. Lets callers produce rows into separate buffers (so their producers
/// carry no write hazard on each other) and still hand a rows kernel one
/// contiguous operand.
pub fn dispatch_deepseek_rows_gather_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    sources: &[&MlxBuffer],
    output: &MlxBuffer,
    row_elements: usize,
) -> Result<()> {
    check_rows(sources.len())?;
    if row_elements == 0 {
        return Err(invalid("gather row_elements must be nonzero"));
    }
    for source in sources {
        require(source, "gather source", DType::F32, row_elements)?;
    }
    let total = product(&[sources.len(), row_elements])?;
    require(output, "gather output", DType::F32, total)?;
    let params = RowsGatherParams {
        rows: sources.len() as u32,
        row_elements: to_u32(row_elements, "gather row_elements")?,
    };
    to_u32(total, "gather elements")?;
    let pipeline = registry
        .get_pipeline(DEEPSEEK_ROWS_GATHER_F32_KERNEL, device.metal_device())?
        .clone();
    let slotted = slots(sources);
    capture(
        encoder,
        CapturedOpKind::Other,
        sources.iter().map(|source| range(source)).collect(),
        vec![range(output)],
    );
    let mut args = vec![
        (0, KernelArg::Bytes(as_bytes(&params))),
        (1, KernelArg::Buffer(output)),
    ];
    push_slots(&mut args, 2, &slotted);
    encoder.encode_threadgroups_with_args(
        &pipeline,
        &args,
        MTLSize::new(total.div_ceil(THREADS as usize) as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Learned KV compressor
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CompressorRowsParams {
    rows: u32,
    seq_len: u32,
    ratio: u32,
    head_dim: u32,
    epsilon: f32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
    start_pos: [u32; DEEPSEEK_ROWS_MAX],
}

/// One row of a compressor rows dispatch: its recurrent state and position.
#[derive(Clone, Copy, Debug)]
pub struct DeepSeekCompressorRow<'a> {
    /// F32 `[coff * ratio, coff * head_dim]` (`coff = 2` for ratio 4).
    pub kv_state: &'a MlxBuffer,
    /// F32, same shape as `kv_state`.
    pub score_state: &'a MlxBuffer,
    /// The decode token's absolute position; nonzero.
    pub start_pos: u32,
}

/// One decode token of the learned KV compressor per row.
///
/// `kv`/`score` are F32 `[rows, coff * head_dim]` (row `r` is entry `r`),
/// `ape` F32 `[ratio, coff * head_dim]`, `norm` F32 `[head_dim]`, and
/// `output` BF16 `[rows, 1, head_dim]`: the normalized pre-RoPE compressed
/// row for rows whose token completes a block, zero otherwise. Each row's
/// state and output equal `dispatch_deepseek_compressor` at batch 1,
/// `seq_len = 1`, `start_pos`, `write_cache = 0` bitwise.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_deepseek_compressor_rows(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    kv: &MlxBuffer,
    score: &MlxBuffer,
    ape: &MlxBuffer,
    norm: &MlxBuffer,
    output: &MlxBuffer,
    rows: &[DeepSeekCompressorRow<'_>],
    ratio: u32,
    head_dim: u32,
    epsilon: f32,
) -> Result<()> {
    check_rows(rows.len())?;
    let supported =
        (ratio == 4 && (head_dim == 128 || head_dim == 512)) || (ratio == 128 && head_dim == 512);
    if !supported {
        return Err(invalid(format!(
            "compressor (ratio, head_dim) must be (4,128), (4,512) or (128,512), got ({ratio},{head_dim})"
        )));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(invalid("compressor epsilon must be finite and positive"));
    }
    let coff = if ratio == 4 { 2 } else { 1 };
    let dim = head_dim as usize;
    let projected = coff * dim;
    let state = product(&[coff * ratio as usize, projected])?;
    let mut params = CompressorRowsParams {
        rows: rows.len() as u32,
        seq_len: 1,
        ratio,
        head_dim,
        epsilon,
        ..Default::default()
    };
    let mut state_ranges = Vec::with_capacity(2 * rows.len());
    for (entry, row) in rows.iter().enumerate() {
        if row.start_pos == 0 || row.start_pos.checked_add(1).is_none() {
            return Err(invalid(format!(
                "compressor row {entry}: position must be in 1..u32::MAX, got {}",
                row.start_pos
            )));
        }
        require(row.kv_state, "kv_state", DType::F32, state)?;
        require(row.score_state, "score_state", DType::F32, state)?;
        state_ranges.push(range(row.kv_state));
        state_ranges.push(range(row.score_state));
        params.start_pos[entry] = row.start_pos;
    }
    disjoint(&state_ranges, "compressor states")?;
    require(kv, "kv", DType::F32, rows.len() * projected)?;
    require(score, "score", DType::F32, rows.len() * projected)?;
    require(ape, "ape", DType::F32, ratio as usize * projected)?;
    require(norm, "norm", DType::F32, dim)?;
    require(output, "output", DType::BF16, rows.len() * dim)?;
    let pipeline = registry
        .get_pipeline(DEEPSEEK_COMPRESSOR_ROWS_KERNEL, device.metal_device())?
        .clone();
    let kv_states: Vec<&MlxBuffer> = rows.iter().map(|row| row.kv_state).collect();
    let score_states: Vec<&MlxBuffer> = rows.iter().map(|row| row.score_state).collect();
    let (kv_states, score_states) = (slots(&kv_states), slots(&score_states));
    let mut reads = vec![range(kv), range(score), range(ape), range(norm)];
    reads.extend(&state_ranges);
    let mut writes = vec![range(output)];
    writes.extend(&state_ranges);
    capture(encoder, CapturedOpKind::Other, reads, writes);
    let mut args = vec![
        (0, KernelArg::Bytes(as_bytes(&params))),
        (1, KernelArg::Buffer(kv)),
        (2, KernelArg::Buffer(score)),
        (3, KernelArg::Buffer(ape)),
        (4, KernelArg::Buffer(norm)),
        (5, KernelArg::Buffer(output)),
    ];
    push_slots(&mut args, 6, &kv_states);
    push_slots(&mut args, 14, &score_states);
    encoder.encode_threadgroups_with_args(
        &pipeline,
        &args,
        MTLSize::new(rows.len() as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    Ok(())
}
