//! Lower-triangular unit-diagonal solve: `X = L \ B`.
//!
//! Solves `L · X = B` where `L` is `N×N` lower-triangular with an *implicit*
//! unit diagonal (diagonal entries are not read). `B` is `N×M`. The kernel
//! is batched over a single leading dim; callers fold any additional leading
//! dims into `batch`.
//!
//! Spec source: ADR-013 Decision 5. Formula (forward substitution):
//!
//! ```text
//! x[0, :]       = b[0, :]
//! x[i, :]       = b[i, :] - sum_{j=0..i-1} L[i, j] * x[j, :]   for 1 <= i < N
//! ```
//!
//! # Memory layout (column-major, innermost-first)
//!
//! * `L[i, j, b]` at `b * N*N + i * N + j`  (row i contiguous, stride N)
//! * `B[i, m, b]` at `b * N*M + i * M + m`
//! * `X[i, m, b]` at `b * N*M + i * M + m`  (same shape + layout as B)
//!
//! This layout makes row-i slices of L contiguous (for the inner-j sum),
//! and makes all M RHS columns for row i adjacent (for the per-m parallel
//! loop).
//!
//! # Parallelism
//!
//! One thread per `(m, batch)` pair. Each thread walks rows 0..N serially,
//! accumulating in f32 regardless of input dtype. The sequential walk is
//! correct because thread-local `x[j]` for j < i has already been written
//! by the same thread in an earlier iteration.
//!
//! # Usage
//!
//! Consumed by the Gated DeltaNet **debug / reference** path (ADR-013
//! Decision 8 CPU parity). The fused production kernel (Decision 6) handles
//! this internally, so this op is not on the production hot path.
//!
//! # Errors
//!
//! - `N == 0`, `M == 0`, or `batch == 0`: returns `InvalidArgument`.
//! - Element counts mismatch `[N, N, batch]` / `[N, M, batch]`.
//! - Dtype mismatch between any of L, B, X.
//! - Unsupported dtype (only F32 and BF16 today).
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static TRI_SOLVE_SHADER_SOURCE: &str = include_str!("../shaders/tri_solve.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("tri_solve_lower_unit_f32", TRI_SOLVE_SHADER_SOURCE);
    registry.register_source("tri_solve_lower_unit_bf16", TRI_SOLVE_SHADER_SOURCE);
}

#[derive(Debug, Clone, Copy)]
pub struct TriSolveParams {
    /// System size (square `L` is `N×N`).
    pub n: u32,
    /// Number of right-hand-side columns.
    pub m: u32,
    /// Batch count (leading dim).
    pub batch: u32,
}

fn validate(
    p: &TriSolveParams,
    l: &MlxBuffer,
    b: &MlxBuffer,
    x: &MlxBuffer,
) -> Result<()> {
    if p.n == 0 || p.m == 0 || p.batch == 0 {
        return Err(MlxError::InvalidArgument(
            "tri_solve: n, m, and batch must all be > 0".into(),
        ));
    }

    let l_elems = (p.n as usize)
        .checked_mul(p.n as usize)
        .and_then(|v| v.checked_mul(p.batch as usize))
        .ok_or_else(|| MlxError::InvalidArgument("tri_solve: L shape overflow".into()))?;
    let bx_elems = (p.n as usize)
        .checked_mul(p.m as usize)
        .and_then(|v| v.checked_mul(p.batch as usize))
        .ok_or_else(|| MlxError::InvalidArgument("tri_solve: B/X shape overflow".into()))?;

    if l.element_count() != l_elems {
        return Err(MlxError::InvalidArgument(format!(
            "tri_solve: L element count {} != n({}) * n({}) * batch({}) = {}",
            l.element_count(),
            p.n,
            p.n,
            p.batch,
            l_elems
        )));
    }
    if b.element_count() != bx_elems {
        return Err(MlxError::InvalidArgument(format!(
            "tri_solve: B element count {} != n({}) * m({}) * batch({}) = {}",
            b.element_count(),
            p.n,
            p.m,
            p.batch,
            bx_elems
        )));
    }
    if x.element_count() != bx_elems {
        return Err(MlxError::InvalidArgument(format!(
            "tri_solve: X element count {} != {}",
            x.element_count(),
            bx_elems
        )));
    }
    if l.dtype() != b.dtype() || l.dtype() != x.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "tri_solve: dtype mismatch L={}, B={}, X={}",
            l.dtype(),
            b.dtype(),
            x.dtype()
        )));
    }
    Ok(())
}

/// Dispatch a lower-triangular unit-diagonal solve.
pub fn dispatch_tri_solve(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    l: &MlxBuffer,
    b: &MlxBuffer,
    x: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: TriSolveParams,
) -> Result<()> {
    validate(&p, l, b, x)?;

    let kernel_name = match l.dtype() {
        DType::F32 => "tri_solve_lower_unit_f32",
        DType::BF16 => "tri_solve_lower_unit_bf16",
        other => {
            return Err(MlxError::InvalidArgument(format!(
                "tri_solve: unsupported dtype {}",
                other
            )));
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // Grid: one thread per (col, batch); serialize over rows inside the thread.
    let grid = MTLSize::new(p.m as u64, p.batch as u64, 1);

    // Threadgroup packing: pack along m first; fill remaining along batch.
    let tg_m = std::cmp::min(p.m, 256).max(1);
    let remain = (256u32 / tg_m).max(1);
    let tg_b = std::cmp::min(p.batch, remain).max(1);
    let tg = MTLSize::new(tg_m as u64, tg_b as u64, 1);

    encoder.encode(
        pipeline,
        &[(0, l), (1, b), (2, x), (3, params_buf)],
        grid,
        tg,
    );

    Ok(())
}
