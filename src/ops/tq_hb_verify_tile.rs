//! Tiled verify attention over the TQ-HB KV cache.
//!
//! One joint verify step attends, per lane, a window of up to
//! [`VERIFY_TILE_ROWS`] rows at consecutive positions of the lane's slot.
//! [`crate::ops::flash_attn_vec_tq_hb`] scans the slot's history once per row
//! and query head. Here one threadgroup computes all rows of a lane's window
//! and all query heads of one KV head over one split of the history, with
//! Metal 4 `mpp::tensor_ops::matmul2d` for Q·Kᵀ and P·V over pages of
//! [`PAGE_TOKENS`] cache tokens, and a second dispatch combines the splits in
//! split order. The structure is Splash's paged verify attention
//! (`runtime/metal/kernels/common/paged_attention_tile.h`); the cache is the
//! TQ-HB one, unchanged (8-bit codes, one F32 norm per KV head and token,
//! rotated domain). See `shaders/tq_hb_verify_tile.metal`.
//!
//! [`tq_hb_verify_tile_attention`] encodes four dispatches: the query
//! rotation and pack, the split, the reduce, and the un-rotation fused with
//! the sigmoid output gate. Its output is the gated attention of every packed
//! row.
//!
//! Numerics (differs in bits from the per-row kernel): the rotated queries,
//! the centroids and each probability times its token's value norm are half
//! operands of matmuls that accumulate in F32 (the per-row kernel also reads
//! Q as half; its centroids and probabilities are F32), scores are F32, and
//! the online softmax runs per page and split instead of per 32-token chunk
//! and workgroup. The half probability operand saturates at 65504, which a
//! value norm below about 1e6 cannot reach. The un-rotation and gate are
//! bitwise
//! [`crate::ops::fwht_standalone::dispatch_fwht_sign_undo_sigmoid_mul_f32`].
//!
//! Invariants (`tests/test_tq_hb_verify_tile.rs`): a row's output is a
//! function of its lane's committed length, its offset in the window, its
//! query, its gate and the cache tokens before its causal end only. It does
//! not depend on the other lanes, on the window's later rows or row count, on
//! the packed row it occupies or on the padding.
//!
//! The pipelines exist only on devices with the Metal tensor API; use
//! [`tq_hb_verify_tile_available`] before offering the route.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::DType;

/// MSL source.
pub static TQ_HB_VERIFY_TILE_SHADER_SOURCE: &str =
    include_str!("../shaders/tq_hb_verify_tile.metal");

/// Rows of one lane's verify tile (the kernel's `TQV_ROWS`).
pub const VERIFY_TILE_ROWS: u32 = 8;
/// Cache tokens per page (the kernel's `TQV_PAGE`).
pub const PAGE_TOKENS: u32 = 32;
/// Most history splits of one lane (the kernel's `TQV_MAX_SPLITS` bounds it).
pub const MAX_SPLITS: u32 = 24;
/// Fewest pages of one split.
pub const MIN_PAGES_PER_SPLIT: u32 = 2;

/// The kernel's `TQV_MAX_SPLITS`: its reduce holds a weight per split.
const KERNEL_MAX_SPLITS: u32 = 128;
const HEAD_DIM: u32 = 256;
const CODEBOOK_BITS: u32 = 8;
const SPLIT_THREADS: u64 = 256;

const UNDO_GATE: &str = "tq_hb_verify_tile_undo_gate";

const KERNELS: [(u32, &str, &str, &str); 2] = [
    (
        6,
        "tq_hb_verify_tile_pack_g6",
        "tq_hb_verify_tile_split_g6",
        "tq_hb_verify_tile_reduce_g6",
    ),
    (
        8,
        "tq_hb_verify_tile_pack_g8",
        "tq_hb_verify_tile_split_g8",
        "tq_hb_verify_tile_reduce_g8",
    ),
];

/// Register every entry point against the shader source.
pub fn register(registry: &mut KernelRegistry) {
    for (_, pack, split, reduce) in KERNELS {
        registry.register_source(pack, TQ_HB_VERIFY_TILE_SHADER_SOURCE);
        registry.register_source(split, TQ_HB_VERIFY_TILE_SHADER_SOURCE);
        registry.register_source(reduce, TQ_HB_VERIFY_TILE_SHADER_SOURCE);
    }
    registry.register_source(UNDO_GATE, TQ_HB_VERIFY_TILE_SHADER_SOURCE);
}

fn invalid(msg: String) -> MlxError {
    MlxError::InvalidArgument(format!("tq_hb_verify_tile: {msg}"))
}

/// Attention geometry and cache layout of one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TqHbVerifyTileShape {
    /// Query heads.
    pub n_heads: u32,
    /// KV heads; `n_heads / n_kv_heads` must be a compiled group (6 or 8).
    pub n_kv_heads: u32,
    /// Must be 256.
    pub head_dim: u32,
    /// Tokens per slot of the cache buffers; a multiple of [`PAGE_TOKENS`].
    pub kv_capacity: u32,
    /// Must be 8.
    pub codebook_bits: u32,
    /// Score scale (`1 / sqrt(head_dim)` for plain attention).
    pub scale: f32,
    /// Packed rows of `q_rows` and `output`, real and padding.
    pub width: u32,
}

impl TqHbVerifyTileShape {
    fn kernels(&self) -> Result<(&'static str, &'static str, &'static str)> {
        if self.head_dim != HEAD_DIM || self.codebook_bits != CODEBOOK_BITS {
            return Err(invalid(format!(
                "needs head_dim {HEAD_DIM} and {CODEBOOK_BITS}-bit codes, got head_dim {} and {} bits",
                self.head_dim, self.codebook_bits
            )));
        }
        if self.n_kv_heads == 0 || self.n_heads % self.n_kv_heads != 0 {
            return Err(invalid(format!(
                "n_heads {} is not a multiple of n_kv_heads {}",
                self.n_heads, self.n_kv_heads
            )));
        }
        let group = self.n_heads / self.n_kv_heads;
        KERNELS
            .iter()
            .find(|(g, ..)| *g == group)
            .map(|&(_, pack, split, reduce)| (pack, split, reduce))
            .ok_or_else(|| invalid(format!("no kernel for {group} query heads per KV head")))
    }

    /// Whether a kernel is compiled for this geometry and layout.
    pub fn supported(&self) -> bool {
        self.kernels().is_ok()
            && self.kv_capacity > 0
            && self.kv_capacity % PAGE_TOKENS == 0
            && self.width > 0
    }

    fn group(&self) -> u64 {
        u64::from(self.n_heads / self.n_kv_heads)
    }

    /// Fused rows of one tile: window rows times query heads per KV head.
    fn fused_rows(&self) -> u64 {
        u64::from(VERIFY_TILE_ROWS) * self.group()
    }
}

/// Whether the tile pipelines for `shape` build on this device. `Ok(false)`
/// when the shape has no kernel or the toolchain or device lacks the Metal
/// tensor API; any other compile failure is an error.
pub fn tq_hb_verify_tile_available(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    shape: &TqHbVerifyTileShape,
) -> Result<bool> {
    if !shape.supported() {
        return Ok(false);
    }
    let (pack, split, reduce) = shape.kernels()?;
    for name in [pack, split, reduce, UNDO_GATE] {
        let probe = registry.probe_optional_pipeline(
            name,
            device.metal_device(),
            device.registry_id(),
            crate::ops::dense_mm_capability::is_unavailable_tensor_header,
        )?;
        if !probe.available {
            return Ok(false);
        }
    }
    Ok(true)
}

/// One lane of a verify step: a window of `rows` rows at positions
/// `committed..committed + rows` of cache slot `slot`, held in packed rows
/// `first_row..first_row + rows` of the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TqHbVerifyLane {
    pub slot: u32,
    pub committed: u32,
    pub rows: u32,
    pub first_row: u32,
}

/// Pages a lane with `committed` tokens before its window reads: those of its
/// history and a full window, whatever the window's row count.
fn verify_tile_pages(committed: u32, kv_capacity: u32) -> u32 {
    committed
        .saturating_add(VERIFY_TILE_ROWS)
        .min(kv_capacity)
        .div_ceil(PAGE_TOKENS)
}

/// History splits of a lane with `committed` tokens before its window: its
/// pages in runs of equal length, at least [`MIN_PAGES_PER_SPLIT`] pages
/// each and at most [`MAX_SPLITS`] runs. Every split owns a page. A function
/// of the lane's own committed length only, so the cohort never changes a
/// lane's arithmetic.
pub fn verify_tile_splits(committed: u32, kv_capacity: u32) -> u32 {
    let pages = verify_tile_pages(committed, kv_capacity).max(1);
    let per_split = pages.div_ceil(MAX_SPLITS).max(MIN_PAGES_PER_SPLIT);
    pages.div_ceil(per_split)
}

/// Partial slots per (lane, KV head) of a step: the most splits of its lanes.
pub fn verify_tile_slot_splits(lanes: &[TqHbVerifyLane], kv_capacity: u32) -> u32 {
    lanes
        .iter()
        .map(|lane| verify_tile_splits(lane.committed, kv_capacity))
        .max()
        .unwrap_or(1)
}

/// Scratch sizes of one step, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TqHbVerifyTileScratchBytes {
    /// Rotated queries, F16 `[lane, kv_head, row, head in group, 256]`.
    pub queries: usize,
    /// Split partials, F32 `[lane, kv_head, split, fused row, 256]`.
    pub partials: usize,
    /// Split statistics, F32 `[lane, kv_head, split, fused row, 2]`.
    pub statistics: usize,
    /// Rotated attention, F32 `[lane, row, head, 256]`.
    pub attention: usize,
}

/// Scratch a step of `lanes` lanes with `slot_splits` partial slots needs.
pub fn verify_tile_scratch_bytes(
    shape: &TqHbVerifyTileShape,
    lanes: usize,
    slot_splits: u32,
) -> TqHbVerifyTileScratchBytes {
    let tiles = lanes * shape.n_kv_heads as usize;
    let fused = shape.fused_rows() as usize;
    let slots = tiles * slot_splits as usize;
    TqHbVerifyTileScratchBytes {
        queries: tiles * fused * HEAD_DIM as usize * 2,
        partials: slots * fused * HEAD_DIM as usize * 4,
        statistics: slots * fused * 2 * 4,
        attention: lanes
            * VERIFY_TILE_ROWS as usize
            * shape.n_heads as usize
            * HEAD_DIM as usize
            * 4,
    }
}

/// Scratch buffers of [`tq_hb_verify_tile_attention`]; sizes from
/// [`verify_tile_scratch_bytes`]. Their contents are undefined afterwards.
#[derive(Clone, Copy)]
pub struct TqHbVerifyTileScratch<'a> {
    pub queries: &'a MlxBuffer,
    pub partials: &'a MlxBuffer,
    pub statistics: &'a MlxBuffer,
    pub attention: &'a MlxBuffer,
}

/// The TQ-HB cache of one attention layer: codes `[slots, n_kv_heads,
/// kv_capacity, 256]` U8 and norms `[slots, n_kv_heads, kv_capacity]` F32.
#[derive(Clone, Copy)]
pub struct TqHbVerifyTileCache<'a> {
    pub k_packed: &'a MlxBuffer,
    pub k_norms: &'a MlxBuffer,
    pub v_packed: &'a MlxBuffer,
    pub v_norms: &'a MlxBuffer,
}

/// Rust mirror of the MSL `TqHbVerifyLane`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TqHbVerifyLaneGpu {
    slot: u32,
    committed: u32,
    rows: u32,
    first_row: u32,
    splits: u32,
}

/// Rust mirror of the MSL `TqHbVerifyParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TqHbVerifyParamsGpu {
    n_heads: u32,
    n_kv_heads: u32,
    kv_capacity: u32,
    slot_splits: u32,
    scale: f32,
}

fn need(name: &str, buffer: &MlxBuffer, dtype: DType, bytes: u64, align: u64) -> Result<()> {
    if buffer.dtype() != dtype {
        return Err(invalid(format!(
            "{name} must be {dtype}, got {}",
            buffer.dtype()
        )));
    }
    if (buffer.data_byte_len() as u64) < bytes {
        return Err(invalid(format!(
            "{name} has {} bytes but needs {bytes}",
            buffer.data_byte_len()
        )));
    }
    if buffer.byte_offset() % align != 0 {
        return Err(invalid(format!(
            "{name} is bound at byte offset {}, not a multiple of {align}",
            buffer.byte_offset()
        )));
    }
    Ok(())
}

/// Encode the tiled verify attention and output gate of one layer.
///
/// `q_rows` is `[width, n_heads, 256]` F32, unrotated, and `gate_rows` the
/// same shape (both only read); `output` receives every packed row's
/// attention times `sigmoid(gate)`, `[width, n_heads, 256]` F32. The lanes'
/// windows are disjoint ranges of the packed rows; a packed row outside
/// every window (padding) receives the attention of packed row
/// `lanes[0].first_row` under its own gate. The cache must already hold (or
/// have encoded before this call, in the same encoder) the K/V of every
/// window row at its position: the barrier before the split orders it after
/// those writes.
///
/// Encodes: pack, barrier, split, barrier, reduce, barrier, un-rotation +
/// gate. The caller orders readers of `output` with a barrier.
#[allow(clippy::too_many_arguments)]
pub fn tq_hb_verify_tile_attention(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    q_rows: &MlxBuffer,
    gate_rows: &MlxBuffer,
    cache: &TqHbVerifyTileCache<'_>,
    output: &MlxBuffer,
    scratch: &TqHbVerifyTileScratch<'_>,
    shape: &TqHbVerifyTileShape,
    lanes: &[TqHbVerifyLane],
) -> Result<()> {
    let (pack, split, reduce) = shape.kernels()?;
    if !shape.supported() {
        return Err(invalid(format!(
            "kv_capacity {} must be a positive multiple of {PAGE_TOKENS} and width {} positive",
            shape.kv_capacity, shape.width
        )));
    }
    if lanes.is_empty() {
        return Err(invalid("no lanes".into()));
    }
    let width = shape.width as usize;
    // Packed row -> its rotated row `lane * VERIFY_TILE_ROWS + window row`;
    // padding repeats the first lane's first row.
    let mut row_map = vec![u32::MAX; width];
    let head_tokens = u64::from(shape.n_kv_heads) * u64::from(shape.kv_capacity);
    let mut slots_needed = 0u64;
    for (index, lane) in lanes.iter().enumerate() {
        if lane.rows == 0 || lane.rows > VERIFY_TILE_ROWS {
            return Err(invalid(format!("lane {index} has {} rows", lane.rows)));
        }
        if u64::from(lane.committed) + u64::from(lane.rows) > u64::from(shape.kv_capacity) {
            return Err(invalid(format!(
                "lane {index} window {}+{} exceeds kv_capacity {}",
                lane.committed, lane.rows, shape.kv_capacity
            )));
        }
        if u64::from(lane.first_row) + u64::from(lane.rows) > u64::from(shape.width) {
            return Err(invalid(format!(
                "lane {index} rows {}+{} exceed width {}",
                lane.first_row, lane.rows, shape.width
            )));
        }
        for row in 0..lane.rows {
            let cell = &mut row_map[(lane.first_row + row) as usize];
            if *cell != u32::MAX {
                return Err(invalid(format!(
                    "lane {index} shares packed row {} with another lane",
                    lane.first_row + row
                )));
            }
            *cell = index as u32 * VERIFY_TILE_ROWS + row;
        }
        slots_needed = slots_needed.max(u64::from(lane.slot) + 1);
    }
    for cell in &mut row_map {
        if *cell == u32::MAX {
            *cell = 0;
        }
    }

    let slot_splits = verify_tile_slot_splits(lanes, shape.kv_capacity);
    if slot_splits > KERNEL_MAX_SPLITS {
        return Err(invalid(format!(
            "{slot_splits} splits exceed the kernel's {KERNEL_MAX_SPLITS}"
        )));
    }
    let sizes = verify_tile_scratch_bytes(shape, lanes.len(), slot_splits);
    let rows_bytes = u64::from(shape.width) * u64::from(shape.n_heads) * u64::from(HEAD_DIM) * 4;
    need("q_rows", q_rows, DType::F32, rows_bytes, 4)?;
    need("gate_rows", gate_rows, DType::F32, rows_bytes, 4)?;
    need("output", output, DType::F32, rows_bytes, 4)?;
    need(
        "queries",
        scratch.queries,
        DType::F16,
        sizes.queries as u64,
        16,
    )?;
    need(
        "partials",
        scratch.partials,
        DType::F32,
        sizes.partials as u64,
        16,
    )?;
    need(
        "statistics",
        scratch.statistics,
        DType::F32,
        sizes.statistics as u64,
        16,
    )?;
    need(
        "attention",
        scratch.attention,
        DType::F32,
        sizes.attention as u64,
        4,
    )?;
    let code_bytes = slots_needed * head_tokens * u64::from(HEAD_DIM);
    let norm_bytes = slots_needed * head_tokens * 4;
    need("k_packed", cache.k_packed, DType::U8, code_bytes, 16)?;
    need("v_packed", cache.v_packed, DType::U8, code_bytes, 16)?;
    need("k_norms", cache.k_norms, DType::F32, norm_bytes, 16)?;
    need("v_norms", cache.v_norms, DType::F32, norm_bytes, 16)?;

    let gpu_lanes: Vec<TqHbVerifyLaneGpu> = lanes
        .iter()
        .map(|lane| TqHbVerifyLaneGpu {
            slot: lane.slot,
            committed: lane.committed,
            rows: lane.rows,
            first_row: lane.first_row,
            splits: verify_tile_splits(lane.committed, shape.kv_capacity),
        })
        .collect();
    let lane_bytes: &[u8] = bytemuck::cast_slice(&gpu_lanes);
    let row_map_bytes: &[u8] = bytemuck::cast_slice(&row_map);
    // Inline constant data is limited to 4 KiB per binding.
    if lane_bytes.len() > 4096 || row_map_bytes.len() > 4096 {
        return Err(invalid(format!(
            "{} lanes over {} packed rows exceed the inline parameter limit",
            lanes.len(),
            shape.width
        )));
    }
    let params = TqHbVerifyParamsGpu {
        n_heads: shape.n_heads,
        n_kv_heads: shape.n_kv_heads,
        kv_capacity: shape.kv_capacity,
        slot_splits,
        scale: shape.scale,
    };
    let metal_device = device.metal_device();
    let lane_count = lanes.len() as u64;
    let (heads, rows) = (u64::from(shape.n_heads), u64::from(VERIFY_TILE_ROWS));

    let pipeline = registry.get_pipeline(pack, metal_device)?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Bytes(lane_bytes)),
            (2, KernelArg::Buffer(q_rows)),
            (3, KernelArg::Buffer(scratch.queries)),
        ],
        MTLSize::new(heads, rows, lane_count),
        MTLSize::new(32, 1, 1),
    );
    encoder.memory_barrier();

    let pipeline = registry.get_pipeline(split, metal_device)?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Bytes(lane_bytes)),
            (2, KernelArg::Buffer(scratch.queries)),
            (3, KernelArg::Buffer(cache.k_packed)),
            (4, KernelArg::Buffer(cache.k_norms)),
            (5, KernelArg::Buffer(cache.v_packed)),
            (6, KernelArg::Buffer(cache.v_norms)),
            (7, KernelArg::Buffer(scratch.partials)),
            (8, KernelArg::Buffer(scratch.statistics)),
        ],
        MTLSize::new(
            u64::from(shape.n_kv_heads),
            u64::from(slot_splits),
            lane_count,
        ),
        MTLSize::new(SPLIT_THREADS, 1, 1),
    );
    encoder.memory_barrier();

    let pipeline = registry.get_pipeline(reduce, metal_device)?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Bytes(lane_bytes)),
            (2, KernelArg::Buffer(scratch.partials)),
            (3, KernelArg::Buffer(scratch.statistics)),
            (4, KernelArg::Buffer(scratch.attention)),
        ],
        MTLSize::new(heads, rows, lane_count),
        MTLSize::new(u64::from(HEAD_DIM), 1, 1),
    );
    encoder.memory_barrier();

    let pipeline = registry.get_pipeline(UNDO_GATE, metal_device)?;
    encoder.encode_threadgroups_with_args(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&params))),
            (1, KernelArg::Bytes(row_map_bytes)),
            (2, KernelArg::Buffer(scratch.attention)),
            (3, KernelArg::Buffer(gate_rows)),
            (4, KernelArg::Buffer(output)),
        ],
        MTLSize::new(heads, u64::from(shape.width), 1),
        MTLSize::new(32, 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_follow_the_lane_history_only() {
        let splits = |committed| verify_tile_splits(committed, 262_144);
        // One page, then runs of at least MIN_PAGES_PER_SPLIT pages.
        assert_eq!(splits(0), 1);
        assert_eq!(splits(24), 1);
        assert_eq!(splits(512), 9); // 17 pages in runs of 2
        assert_eq!(splits(2_048), 22); // 65 pages in runs of 3
        assert_eq!(splits(8_192), 24); // 257 pages in runs of 11
        assert_eq!(splits(262_136), MAX_SPLITS);
        // The bound never passes the capacity.
        assert_eq!(verify_tile_splits(u32::MAX, 64), 1);
        for committed in (0..40_000).step_by(7) {
            let pages = verify_tile_pages(committed, 262_144);
            let count = splits(committed);
            let per_split = pages.div_ceil(count);
            assert!((1..=MAX_SPLITS).contains(&count));
            // Every split owns a page: the kernel's partition writes them all.
            assert_eq!(pages.div_ceil(per_split), count, "committed {committed}");
        }
    }

    #[test]
    fn scratch_sizes_match_the_kernel_layout() {
        let shape = TqHbVerifyTileShape {
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            kv_capacity: 4_096,
            codebook_bits: 8,
            scale: 0.0625,
            width: 32,
        };
        assert!(shape.supported());
        let sizes = verify_tile_scratch_bytes(&shape, 4, 22);
        assert_eq!(sizes.queries, 4 * 4 * 48 * 256 * 2);
        assert_eq!(sizes.partials, 4 * 4 * 22 * 48 * 256 * 4);
        assert_eq!(sizes.statistics, 4 * 4 * 22 * 48 * 2 * 4);
        assert_eq!(sizes.attention, 4 * 8 * 24 * 256 * 4);
        let odd = TqHbVerifyTileShape {
            n_heads: 20,
            ..shape
        };
        assert!(!odd.supported());
        let unaligned = TqHbVerifyTileShape {
            kv_capacity: 4_100,
            ..shape
        };
        assert!(!unaligned.supported());
    }
}
