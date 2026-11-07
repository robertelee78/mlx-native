//! Row-exact few-row mat-vec in the stock one-row order: F16 or Q8_0 weights
//! × F32 input → F32 output, `y[j, :] = W · x[j, :]` for `1 <= m <= 8`.
//!
//! Wraps `shaders/row_exact_matvec_stock.metal`. Every output row is bitwise
//! equal to the stock one-row kernel for that storage, whatever `m` is:
//!
//! - F16: [`crate::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io`]
//!   (`dense_matvec_f16w_f32io`);
//! - Q8_0: the canonical GGUF matvec at `m = 1`
//!   (`kernel_mul_mv_q8_0_f32_nr2`, the default `dense_q8_0_mv_nr2` route of
//!   [`crate::quantized_matmul_ggml`]), and per group
//!   [`crate::ops::quantized_matmul_ggml::quantized_matmul_ggml_batched_mv`]
//!   for grouped products.
//!
//! The kernels replay the stock per-row arithmetic (same thread → element
//! map, same expressions, same `simd_sum` combines) and share each weight
//! load across the `m` input rows. The schedule only packs rows; it never
//! changes a value. `tests/row_exact_matvec_stock.rs` proves the equality on
//! every DeepSeek-V4 decode shape.
//!
//! Shapes and strides are [`RowExactMatvecShape`], so grouped token-major
//! products ([`RowExactMatvecShape::grouped_token_major`]) need no permutes.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::row_exact_matvec::{
    validate, RowExactMatvecSchedule, RowExactMatvecShape, RowExactWeight,
};

/// MSL source for the stock-order row-exact kernels (embedded at compile time).
pub static ROW_EXACT_MATVEC_STOCK_SHADER_SOURCE: &str =
    include_str!("../shaders/row_exact_matvec_stock.metal");

macro_rules! stock_names {
    ($prefix:literal) => {
        [
            concat!($prefix, "_m1"),
            concat!($prefix, "_m2"),
            concat!($prefix, "_m3"),
            concat!($prefix, "_m4"),
            concat!($prefix, "_m5"),
            concat!($prefix, "_m6"),
            concat!($prefix, "_m7"),
            concat!($prefix, "_m8"),
        ]
    };
}

/// F16 kernel names indexed `[rows per simdgroup (1, 4)][unroll (1, 4)][m - 1]`.
const F16_NAMES: [[[&str; 8]; 2]; 2] = [
    [
        stock_names!("row_exact_stock_mv_f16_f32_r1_u1"),
        stock_names!("row_exact_stock_mv_f16_f32_r1_u4"),
    ],
    [
        stock_names!("row_exact_stock_mv_f16_f32_r4_u1"),
        stock_names!("row_exact_stock_mv_f16_f32_r4_u4"),
    ],
];

/// Q8_0 kernel names indexed `[rows per threadgroup - 1][virtual simdgroups
/// per simdgroup (1, 2, 4)][m - 1]`.
const Q8_0_NAMES: [[[&str; 8]; 3]; 2] = [
    [
        stock_names!("row_exact_stock_mv_q8_0_f32_r1_v1"),
        stock_names!("row_exact_stock_mv_q8_0_f32_r1_v2"),
        stock_names!("row_exact_stock_mv_q8_0_f32_r1_v4"),
    ],
    [
        stock_names!("row_exact_stock_mv_q8_0_f32_r2_v1"),
        stock_names!("row_exact_stock_mv_q8_0_f32_r2_v2"),
        stock_names!("row_exact_stock_mv_q8_0_f32_r2_v4"),
    ],
];

/// Every kernel function name in `row_exact_matvec_stock.metal`.
pub fn row_exact_matvec_stock_kernel_names() -> impl Iterator<Item = &'static str> {
    F16_NAMES
        .iter()
        .flatten()
        .flatten()
        .chain(Q8_0_NAMES.iter().flatten().flatten())
        .copied()
}

/// Register the stock-order shader source with the given registry.
///
/// [`KernelRegistry::new`] already carries these sources; this is for
/// registries built without the constructor's defaults.
pub fn register(registry: &mut KernelRegistry) {
    for name in row_exact_matvec_stock_kernel_names() {
        registry.register_source(name, ROW_EXACT_MATVEC_STOCK_SHADER_SOURCE);
    }
}

/// Work packing of a stock-order dispatch. Numerically inert: every packing
/// produces the stock one-row values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowExactStockPacking {
    /// F16: `rows` (1 or 4) weight rows per simdgroup, `simdgroups` (1..=32)
    /// per threadgroup, `unroll` (1 or 4) steps whose loads are issued
    /// before their in-order accumulations.
    F16 {
        rows: u32,
        simdgroups: u32,
        unroll: u32,
    },
    /// Q8_0: `rows` (1 or 2) weight rows per threadgroup; each simdgroup runs
    /// `virtual_simdgroups` (1, 2 or 4) of the stock four-simdgroup team, so
    /// a threadgroup has `4 / virtual_simdgroups` simdgroups.
    Q8_0 { rows: u32, virtual_simdgroups: u32 },
}

impl RowExactStockPacking {
    /// Default packing for `weight` at `shape`, from the
    /// `bench_row_exact_matvec_stock` `sweep` family (M5 Max, warm and cold
    /// weights, m = 1 and 4, two runs; cold cells decide).
    ///
    /// - F16: one row per simdgroup (spreads skinny outputs over more
    ///   simdgroups), two simdgroups per threadgroup, four steps' loads issued
    ///   before their accumulations (hides the load latency of the stock
    ///   per-lane chain).
    /// - Q8_0: one row per threadgroup, two for outputs of 65536 rows or
    ///   more (the vocabulary head at m = 4); the whole four-simdgroup team
    ///   on one simdgroup when K <= 1024 (one block per virtual thread), else
    ///   the stock four simdgroups.
    pub fn auto(weight: RowExactWeight, shape: &RowExactMatvecShape) -> Self {
        let rows = u64::from(shape.n) * u64::from(shape.batch);
        match weight {
            RowExactWeight::F16 => Self::F16 {
                rows: 1,
                simdgroups: 2,
                unroll: 4,
            },
            RowExactWeight::Q8_0 => Self::Q8_0 {
                rows: if rows >= 65_536 { 2 } else { 1 },
                virtual_simdgroups: if shape.k <= 1024 { 4 } else { 1 },
            },
        }
    }

    fn validate(&self, weight: RowExactWeight) -> Result<()> {
        let valid = match (*self, weight) {
            (
                Self::F16 {
                    rows,
                    simdgroups,
                    unroll,
                },
                RowExactWeight::F16,
            ) => matches!(rows, 1 | 4) && (1..=32).contains(&simdgroups) && matches!(unroll, 1 | 4),
            (
                Self::Q8_0 {
                    rows,
                    virtual_simdgroups,
                },
                RowExactWeight::Q8_0,
            ) => matches!(rows, 1 | 2) && matches!(virtual_simdgroups, 1 | 2 | 4),
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(MlxError::InvalidArgument(format!(
                "row_exact_matvec_stock: packing {self:?} is not valid for {weight:?}"
            )))
        }
    }

    /// Weight rows one threadgroup covers.
    fn rows_per_threadgroup(&self) -> u32 {
        match *self {
            Self::F16 {
                rows, simdgroups, ..
            } => rows * simdgroups,
            Self::Q8_0 { rows, .. } => rows,
        }
    }

    fn threads_per_threadgroup(&self) -> u32 {
        match *self {
            Self::F16 { simdgroups, .. } => 32 * simdgroups,
            Self::Q8_0 {
                virtual_simdgroups, ..
            } => 32 * (4 / virtual_simdgroups),
        }
    }

    fn kernel_name(&self, m: u32) -> &'static str {
        let m = m as usize - 1;
        match *self {
            Self::F16 { rows, unroll, .. } => {
                F16_NAMES[usize::from(rows == 4)][usize::from(unroll == 4)][m]
            }
            Self::Q8_0 {
                rows,
                virtual_simdgroups,
            } => {
                let v = match virtual_simdgroups {
                    1 => 0,
                    2 => 1,
                    _ => 2,
                };
                Q8_0_NAMES[rows as usize - 1][v][m]
            }
        }
    }
}

/// Refuse a packing whose threadgroup exceeds the pipeline's thread limit
/// (register-heavy instantiations can have a limit below 1024).
fn check_thread_limit(packing: &RowExactStockPacking, limit: u64) -> Result<u64> {
    let threads = u64::from(packing.threads_per_threadgroup());
    if threads > limit {
        return Err(MlxError::InvalidArgument(format!(
            "row_exact_matvec_stock: packing {packing:?} needs {threads} threads per threadgroup, \
             pipeline limit {limit}"
        )));
    }
    Ok(threads)
}

/// MSL-compatible params; must match `RowExactStockParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRowExactStockParams {
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

/// Encode `y[b, j, :] = W[b] · x[b, j, :]` in the stock one-row order with
/// the default packing ([`RowExactStockPacking::auto`]).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_row_exact_matvec_stock(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    weight_type: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y: &MlxBuffer,
    shape: &RowExactMatvecShape,
) -> Result<()> {
    let packing = RowExactStockPacking::auto(weight_type, shape);
    dispatch_row_exact_matvec_stock_packed(
        encoder,
        registry,
        device,
        weight_type,
        w,
        x,
        y,
        shape,
        &packing,
    )
}

/// Encode a stock-order row-exact mat-vec with an explicit packing.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_row_exact_matvec_stock_packed(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    weight_type: RowExactWeight,
    w: &MlxBuffer,
    x: &MlxBuffer,
    y: &MlxBuffer,
    shape: &RowExactMatvecShape,
    packing: &RowExactStockPacking,
) -> Result<()> {
    packing.validate(weight_type)?;
    // Buffer, stride, alignment and alias checks shared with
    // `row_exact_matvec` (the schedule argument is not used here).
    let row_bytes = validate(
        weight_type,
        w,
        x,
        y,
        shape,
        &RowExactMatvecSchedule { nsg: 1, nr0: 1 },
    )?;
    let pipeline = registry.get_pipeline(packing.kernel_name(shape.m), device)?;
    let threads = check_thread_limit(packing, pipeline.max_total_threads_per_threadgroup())?;

    let tgs_per_batch = shape.n.div_ceil(packing.rows_per_threadgroup());
    let params = GpuRowExactStockParams {
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
    let threadgroups = MTLSize::new(u64::from(tgs_per_batch) * u64::from(shape.batch), 1, 1);
    let threads_per_tg = MTLSize::new(threads, 1, 1);
    let bindings = [
        (0, KernelArg::Buffer(w)),
        (1, KernelArg::Buffer(x)),
        (2, KernelArg::Buffer(y)),
        (3, KernelArg::Bytes(as_bytes(&params))),
    ];
    match *packing {
        RowExactStockPacking::F16 { .. } => encoder.dispatch_tracked_threadgroups_with_args(
            pipeline,
            &bindings,
            &[w, x],
            &[y],
            threadgroups,
            threads_per_tg,
        ),
        RowExactStockPacking::Q8_0 { rows, .. } => {
            let smem_bytes = u64::from(rows * shape.m * 32) * 4;
            encoder.dispatch_tracked_threadgroups_with_args_and_shared(
                pipeline,
                &bindings,
                &[(0, smem_bytes)],
                &[w, x],
                &[y],
                threadgroups,
                threads_per_tg,
            )
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_limit_refuses_threadgroups_above_the_pipeline_limit() {
        let f16 = |simdgroups| RowExactStockPacking::F16 {
            rows: 1,
            simdgroups,
            unroll: 4,
        };
        assert_eq!(check_thread_limit(&f16(16), 512).unwrap(), 512);
        assert!(check_thread_limit(&f16(17), 512).is_err());
        assert!(check_thread_limit(&f16(32), 896).is_err());
        let q8 = RowExactStockPacking::Q8_0 {
            rows: 2,
            virtual_simdgroups: 1,
        };
        assert_eq!(check_thread_limit(&q8, 128).unwrap(), 128);
        assert!(check_thread_limit(&q8, 96).is_err());
    }
}
