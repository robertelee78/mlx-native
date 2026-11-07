//! Row-exact few-row mat-vec: F16 or Q8_0 weights × F32 input → F32 output,
//! `y[j, :] = W · x[j, :]` for `1 <= m <= 8` input rows in one dispatch.
//!
//! Wraps `shaders/row_exact_matvec.metal`. One dispatch reads every weight
//! row once for all `m` input rows (the peer's few-row `mul_mv_ext` idea) and
//! splits K over all threads of a threadgroup (the peer's `mul_mv` nsg
//! split), so skinny outputs still fill the GPU.
//!
//! # Row exactness
//!
//! Output row `j` is bitwise identical whether it is computed alone (`m = 1`)
//! or as one row of any `m <= 8`, and identical across repeated runs. Each
//! thread walks a fixed subset of K with explicit `fma` chains, and the
//! threadgroup reduction is `simd_sum` over fixed lanes. The only schedule
//! field that affects numerics is [`RowExactMatvecSchedule::nsg`], which
//! [`row_exact_reduction_nsg`] derives from the weight shape
//! `(weight, batch * n, k)` alone; [`RowExactMatvecSchedule::nr0`] (weight
//! rows per threadgroup) only packs rows.
//!
//! # Batched / grouped products
//!
//! `batch > 1` runs independent products in one grid: batch `b` reads weight
//! rows at `b * n * row_bytes`, input at `b * x_batch_stride` and writes
//! output at `b * y_batch_stride`. [`RowExactMatvecShape::grouped_token_major`]
//! describes a grouped projection whose input and output stay token-major
//! (no permutes), e.g. DeepSeek-V4 attention output-A.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// MSL source for the row-exact mat-vec kernels (embedded at compile time).
pub static ROW_EXACT_MATVEC_SHADER_SOURCE: &str = include_str!("../shaders/row_exact_matvec.metal");

/// Maximum input rows per dispatch.
pub const ROW_EXACT_MATVEC_MAX_ROWS: u32 = 8;

const Q8_0_BLOCK_VALUES: u32 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

macro_rules! row_exact_names {
    ($ty:literal, $r:literal) => {
        [
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m1"),
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m2"),
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m3"),
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m4"),
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m5"),
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m6"),
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m7"),
            concat!("row_exact_mv_", $ty, "_f32_r", $r, "_m8"),
        ]
    };
}

/// Kernel names indexed `[weight][nr0 - 1][m - 1]`.
const KERNEL_NAMES: [[[&str; 8]; 2]; 2] = [
    [row_exact_names!("f16", "1"), row_exact_names!("f16", "2")],
    [row_exact_names!("q8_0", "1"), row_exact_names!("q8_0", "2")],
];

/// Every kernel function name in `row_exact_matvec.metal`.
pub fn row_exact_matvec_kernel_names() -> impl Iterator<Item = &'static str> {
    KERNEL_NAMES.iter().flatten().flatten().copied()
}

/// Register the row-exact mat-vec shader source with the given registry.
///
/// [`KernelRegistry::new`] already carries these sources; this is for
/// registries built without the constructor's defaults.
pub fn register(registry: &mut KernelRegistry) {
    for name in row_exact_matvec_kernel_names() {
        registry.register_source(name, ROW_EXACT_MATVEC_SHADER_SOURCE);
    }
}

/// Weight storage of a row-exact mat-vec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowExactWeight {
    /// Row-major `[batch * n, k]` F16 (`DType::F16`).
    F16,
    /// Row-major `[batch * n, k / 32]` Q8_0 blocks (`DType::U8`, 34 bytes each).
    Q8_0,
}

impl RowExactWeight {
    fn index(self) -> usize {
        match self {
            Self::F16 => 0,
            Self::Q8_0 => 1,
        }
    }

    /// Bytes per weight row of `k` elements.
    pub fn row_bytes(self, k: u32) -> u64 {
        match self {
            Self::F16 => u64::from(k) * 2,
            Self::Q8_0 => u64::from(k / Q8_0_BLOCK_VALUES) * Q8_0_BLOCK_BYTES,
        }
    }

    /// Elements one thread consumes per step (matches the shader).
    fn values_per_thread_step(self) -> u32 {
        16
    }
}

/// Problem shape and F32 strides (in elements) of a row-exact mat-vec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowExactMatvecShape {
    /// Input rows (1..=8).
    pub m: u32,
    /// Output columns = weight rows per batch.
    pub n: u32,
    /// Inner dimension.
    pub k: u32,
    /// Independent products (>= 1), dispatched batch-major along grid x
    /// (all threadgroups of batch `b` before those of batch `b + 1`).
    pub batch: u32,
    /// Elements between input rows (multiple of 4).
    pub x_row_stride: u32,
    /// Elements between input batches (multiple of 4).
    pub x_batch_stride: u32,
    /// Elements between output rows.
    pub y_row_stride: u32,
    /// Elements between output batches.
    pub y_batch_stride: u32,
}

impl RowExactMatvecShape {
    /// Dense `x [m, k]`, `W [n, k]`, `y [m, n]`.
    pub fn contiguous(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            batch: 1,
            x_row_stride: k,
            x_batch_stride: 0,
            y_row_stride: n,
            y_batch_stride: 0,
        }
    }

    /// Grouped projection with token-major activations: `x [m, groups * k]`,
    /// `W [groups * n, k]` (group `g` owns rows `g*n..(g+1)*n`),
    /// `y [m, groups * n]` with `y[j, g*n + i] = W[g*n + i, :] · x[j, g*k..(g+1)*k]`.
    pub fn grouped_token_major(m: u32, groups: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            batch: groups,
            x_row_stride: groups * k,
            x_batch_stride: k,
            y_row_stride: groups * n,
            y_batch_stride: n,
        }
    }
}

/// Threadgroup schedule of a row-exact mat-vec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowExactMatvecSchedule {
    /// Simdgroups per threadgroup (1..=32). Fixes the per-row reduction
    /// order: row exactness holds only between dispatches with equal `nsg`.
    pub nsg: u32,
    /// Weight rows per threadgroup (1 or 2), sharing each input load.
    /// Numerically inert.
    pub nr0: u32,
}

/// Simdgroups per threadgroup for a weight of `rows = batch * n` rows of `k`.
///
/// Depends only on the weight shape, never on `m`, so every row of an
/// `m`-row dispatch matches the same row computed at `m = 1`.
///
/// Tuned on M5 Max with `bench_row_exact_matvec` (`sweep` family, warm and
/// cold weights, m = 1 and 4; best of three sweeps): split K over just
/// enough simdgroups that `rows * 32 * nsg` reaches 32768 threads (F16) /
/// 131072 threads (Q8_0), at most 4 simdgroups once `rows >= 256` (wide
/// outputs lose to the cross-simdgroup reduction beyond that) and at least
/// one full 16-value step per thread. An 8/16-lane-per-row layout (the
/// peer's few-row lane map) was measured too and never won by more than the
/// run-to-run noise, so it is not provided.
pub fn row_exact_reduction_nsg(weight: RowExactWeight, rows: u32, k: u32) -> u32 {
    let per_sg = 32 * weight.values_per_thread_step();
    let max_nsg = (k / per_sg).clamp(1, 32);
    let target_simdgroups: u32 = match weight {
        RowExactWeight::F16 => 1024,
        RowExactWeight::Q8_0 => 4096,
    };
    let cap = if rows >= 256 { 4 } else { 32 };
    target_simdgroups
        .div_ceil(rows.max(1))
        .next_power_of_two()
        .clamp(1, cap.min(max_nsg))
}

impl RowExactMatvecSchedule {
    /// Shape-derived schedule: `nsg` from [`row_exact_reduction_nsg`];
    /// `nr0 = 2` for Q8_0 (faster at m = 4, equal at m = 1), `nr0 = 1` for
    /// F16 (pairs of F16 rows lose at m = 4). Its `nsg * 32` threads may
    /// exceed what a device's pipelines can launch; dispatch with
    /// [`Self::for_device`] (what [`dispatch_row_exact_matvec`] uses).
    pub fn auto(weight: RowExactWeight, shape: &RowExactMatvecShape) -> Self {
        let rows = shape.n.saturating_mul(shape.batch);
        let nsg = row_exact_reduction_nsg(weight, rows, shape.k);
        let nr0 = match weight {
            RowExactWeight::Q8_0 if shape.n >= 2 => 2,
            _ => 1,
        };
        Self { nsg, nr0 }
    }

    /// [`Self::auto`] with `nsg` capped at [`row_exact_matvec_nsg_limit`],
    /// the largest reduction every `m` and packing of `weight` can launch
    /// on this device. The cap depends only on the weight type and device,
    /// so rows stay equal across `m = 1..=8` on one device.
    pub fn for_device(
        registry: &mut KernelRegistry,
        device: &metal::DeviceRef,
        weight: RowExactWeight,
        shape: &RowExactMatvecShape,
    ) -> Result<Self> {
        let auto = Self::auto(weight, shape);
        let limit = row_exact_matvec_nsg_limit(registry, device, weight)?;
        Ok(Self {
            nsg: auto.nsg.min(limit),
            ..auto
        })
    }
}

/// Largest power-of-two `nsg` (at most 32) whose `nsg * 32` threads every
/// row-exact pipeline of `weight` (all `m`, both `nr0`) can launch on
/// `device`: the minimum `max_total_threads_per_threadgroup` over those
/// pipelines. The pipeline limit depends on the compiled kernel's register
/// use and the GPU; larger `m` kernels hold more accumulators and can fall
/// below 1024 threads on some devices.
pub fn row_exact_matvec_nsg_limit(
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    weight: RowExactWeight,
) -> Result<u32> {
    let mut threads = 32 * 32;
    for name in KERNEL_NAMES[weight.index()].iter().flatten() {
        threads = threads.min(
            registry
                .get_pipeline(name, device)?
                .max_total_threads_per_threadgroup(),
        );
    }
    if threads < 32 {
        return Err(invalid(format!(
            "{weight:?} pipelines launch at most {threads} threads per threadgroup, below one simdgroup"
        )));
    }
    let simdgroups = (threads / 32) as u32;
    Ok(1 << (31 - simdgroups.leading_zeros()))
}

/// MSL-compatible params; must match `RowExactMvParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRowExactMvParams {
    k: u32,
    n: u32,
    x_row_stride: u32,
    x_batch_stride: u32,
    y_row_stride: u32,
    y_batch_stride: u32,
    tgs_per_batch: u32,
    pad: u32,
    w_row_bytes: u64,
    w_batch_bytes: u64,
}

fn invalid(message: String) -> MlxError {
    MlxError::InvalidArgument(format!("row_exact_matvec: {message}"))
}

/// Last element index + 1 touched by a strided `[batch, rows, cols]` view.
fn strided_extent(batch: u32, batch_stride: u32, rows: u32, row_stride: u32, cols: u32) -> u64 {
    u64::from(batch - 1) * u64::from(batch_stride)
        + u64::from(rows - 1) * u64::from(row_stride)
        + u64::from(cols)
}

fn validate(
    weight_type: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y: &MlxBuffer,
    shape: &RowExactMatvecShape,
    schedule: &RowExactMatvecSchedule,
) -> Result<u64> {
    let s = shape;
    if s.m == 0 || s.m > ROW_EXACT_MATVEC_MAX_ROWS {
        return Err(invalid(format!("m ({}) must be in 1..=8", s.m)));
    }
    if s.n == 0 || s.k == 0 || s.batch == 0 {
        return Err(invalid(format!(
            "n ({}), k ({}) and batch ({}) must be > 0",
            s.n, s.k, s.batch
        )));
    }
    if !(1..=32).contains(&schedule.nsg) || !(1..=2).contains(&schedule.nr0) {
        return Err(invalid(format!(
            "schedule nsg ({}) must be in 1..=32 and nr0 ({}) in 1..=2",
            schedule.nsg, schedule.nr0
        )));
    }
    let (weight_dtype, k_multiple, weight_align) = match weight_type {
        RowExactWeight::F16 => (DType::F16, 4, 8),
        RowExactWeight::Q8_0 => (DType::U8, Q8_0_BLOCK_VALUES, 2),
    };
    if w.dtype() != weight_dtype {
        return Err(invalid(format!(
            "{weight_type:?} weight must be {weight_dtype:?}, got {:?}",
            w.dtype()
        )));
    }
    if x.dtype() != DType::F32 || y.dtype() != DType::F32 {
        return Err(invalid(format!(
            "input and output must be F32, got {:?} and {:?}",
            x.dtype(),
            y.dtype()
        )));
    }
    if s.k % k_multiple != 0 {
        return Err(invalid(format!(
            "k ({}) must be a multiple of {k_multiple} for {weight_type:?}",
            s.k
        )));
    }
    if s.x_row_stride % 4 != 0 || s.x_batch_stride % 4 != 0 || x.byte_offset() % 16 != 0 {
        return Err(invalid(format!(
            "input strides ({}, {}) must be multiples of 4 and the input offset ({}) 16-byte aligned",
            s.x_row_stride,
            s.x_batch_stride,
            x.byte_offset()
        )));
    }
    if w.byte_offset() % weight_align != 0 || y.byte_offset() % 4 != 0 {
        return Err(invalid(format!(
            "weight offset ({}) must be {weight_align}-byte and output offset ({}) 4-byte aligned",
            w.byte_offset(),
            y.byte_offset()
        )));
    }
    // Output elements must not alias: rows of one product and the products of
    // one dispatch write disjoint ranges (either layout order is accepted).
    let rows_disjoint = s.m == 1 || s.y_row_stride >= s.n;
    let outputs_disjoint = if s.batch == 1 {
        rows_disjoint
    } else if s.m == 1 {
        s.y_batch_stride >= s.n
    } else if s.y_row_stride >= s.y_batch_stride {
        s.y_batch_stride >= s.n
            && u64::from(s.y_row_stride)
                >= u64::from(s.batch - 1) * u64::from(s.y_batch_stride) + u64::from(s.n)
    } else {
        rows_disjoint
            && u64::from(s.y_batch_stride)
                >= u64::from(s.m - 1) * u64::from(s.y_row_stride) + u64::from(s.n)
    };
    if !outputs_disjoint {
        return Err(invalid(format!(
            "output strides overlap: y_row_stride {}, y_batch_stride {}, m {}, n {}, batch {}",
            s.y_row_stride, s.y_batch_stride, s.m, s.n, s.batch
        )));
    }
    let row_bytes = weight_type.row_bytes(s.k);
    let weight_bytes = row_bytes * u64::from(s.n) * u64::from(s.batch);
    if (w.data_byte_len() as u64) < weight_bytes {
        return Err(invalid(format!(
            "weight holds {} bytes, needs {weight_bytes}",
            w.data_byte_len()
        )));
    }
    let x_elems = strided_extent(s.batch, s.x_batch_stride, s.m, s.x_row_stride, s.k);
    if (x.data_byte_len() as u64) < x_elems * 4 {
        return Err(invalid(format!(
            "input holds {} bytes, needs {}",
            x.data_byte_len(),
            x_elems * 4
        )));
    }
    let y_elems = strided_extent(s.batch, s.y_batch_stride, s.m, s.y_row_stride, s.n);
    if (y.data_byte_len() as u64) < y_elems * 4 {
        return Err(invalid(format!(
            "output holds {} bytes, needs {}",
            y.data_byte_len(),
            y_elems * 4
        )));
    }
    // In-batch output index is computed in 32 bits by the shader.
    if u64::from(s.m - 1) * u64::from(s.y_row_stride) + u64::from(s.n) > u64::from(u32::MAX) {
        return Err(invalid("output row extent exceeds u32".into()));
    }
    Ok(row_bytes)
}

/// Encode `y[b, j, :] = W[b] · x[b, j, :]` with the default schedule
/// ([`RowExactMatvecSchedule::for_device`]).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_row_exact_matvec(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    weight_type: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y: &MlxBuffer,
    shape: &RowExactMatvecShape,
) -> Result<()> {
    let schedule = RowExactMatvecSchedule::for_device(registry, device, weight_type, shape)?;
    dispatch_row_exact_matvec_scheduled(
        encoder,
        registry,
        device,
        weight_type,
        w,
        x,
        y,
        shape,
        &schedule,
    )
}

/// Encode a row-exact mat-vec with an explicit schedule.
///
/// Rows match across dispatches only when [`RowExactMatvecSchedule::nsg`]
/// matches; use
/// [`dispatch_row_exact_matvec`] unless pinning a schedule on purpose.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_row_exact_matvec_scheduled(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    weight_type: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y: &MlxBuffer,
    shape: &RowExactMatvecShape,
    schedule: &RowExactMatvecSchedule,
) -> Result<()> {
    let row_bytes = validate(weight_type, w, x, y, shape, schedule)?;
    let name = KERNEL_NAMES[weight_type.index()][schedule.nr0 as usize - 1][shape.m as usize - 1];
    let pipeline = registry.get_pipeline(name, device)?;
    // A launch above the pipeline's limit is dropped by Metal without an
    // error, leaving the output unwritten: refuse it here instead.
    let pipeline_threads = pipeline.max_total_threads_per_threadgroup();
    if u64::from(schedule.nsg) * 32 > pipeline_threads {
        return Err(invalid(format!(
            "schedule nsg {} needs {} threads but {name} launches at most {pipeline_threads} threads per threadgroup on this device",
            schedule.nsg,
            schedule.nsg * 32
        )));
    }

    let tgs_per_batch = shape.n.div_ceil(schedule.nr0);
    let params = GpuRowExactMvParams {
        k: shape.k,
        n: shape.n,
        x_row_stride: shape.x_row_stride,
        x_batch_stride: shape.x_batch_stride,
        y_row_stride: shape.y_row_stride,
        y_batch_stride: shape.y_batch_stride,
        tgs_per_batch,
        pad: 0,
        w_row_bytes: row_bytes,
        w_batch_bytes: row_bytes * u64::from(shape.n),
    };
    let smem_bytes = u64::from(schedule.nr0 * shape.m * 32) * 4;
    // Batches are flattened batch-major into grid x, so threadgroups in
    // flight share one batch's input rows.
    let threadgroups = MTLSize::new(u64::from(tgs_per_batch) * u64::from(shape.batch), 1, 1);
    let threads_per_tg = MTLSize::new(u64::from(schedule.nsg) * 32, 1, 1);
    encoder.dispatch_tracked_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Buffer(w)),
            (1, KernelArg::Buffer(x)),
            (2, KernelArg::Buffer(y)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        &[(0, smem_bytes)],
        &[w, x],
        &[y],
        threadgroups,
        threads_per_tg,
    );
    Ok(())
}
