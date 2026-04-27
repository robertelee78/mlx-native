//! Wave 5b.1 iter 4 — per-chunk-block tri-solve invert wrapper.
//!
//! Computes `A_inv = (I + A_strict_lower)^-1` per `[BT, BT]` chunk-block
//! on the FLA-native A layout `[B, T, H, BT]`.
//!
//! # Why a dedicated kernel (instead of batched `tri_solve_lower_unit`)
//!
//! The mlx-native `tri_solve_lower_unit` kernel solves `L · X = B` with
//! implicit unit diagonal of L. To use it for `(I + A_strict)^-1` we'd
//! pass `A_strict` as `L` (its zero diagonal becomes the implicit unit
//! diagonal of `I + A_strict`) and `B = I`. The math works out — but
//! the memory layout doesn't:
//!
//! - `tri_solve_lower_unit` expects `L[i, j, batch]` at
//!   `batch * N*N + i * N + j` (rows of L are contiguous, batch outer).
//! - FLA's A is `[B, T, H, BT]` with row-stride `H*BT` (rows are NOT
//!   contiguous within a chunk-block).
//!
//! Batching the existing kernel would require a transpose pass to compact
//! each `(b, i_t, i_h)` block into a contiguous `[BT, BT]` slab. That
//! pass is itself a Metal kernel — at which point we may as well do the
//! invert in the same kernel and save the round-trip. So iter 4 ships a
//! dedicated kernel that solves directly on FLA's native layout.
//!
//! Iter 5 perf may revisit this: a fused recompute_w_u that reads A_inv
//! through a shared-memory stage avoids the global write entirely, OR a
//! solve_tril clone of FLA's `merge_16x16/32x32` block decomposition
//! (`/opt/vllm/vllm/model_executor/layers/fla/ops/solve_tril.py:30-503`)
//! could be faster on long-T workloads. Iter 4 baseline = O(BT^3) per
//! threadgroup; that's 262 144 FMAs × `B*NT*H = 8` blocks for the test
//! shape = 2 M FMAs total. Empirically negligible on M5 Max.
//!
//! # Spec source
//!
//! - FLA `solve_tril` semantics:
//!   `/opt/vllm/vllm/model_executor/layers/fla/ops/solve_tril.py:506-530`.
//!   "Compute the inverse of the matrix I + A. A should be strictly lower
//!   triangular, i.e., A.triu() == 0."

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static CHUNK_TRI_SOLVE_INVERT_SHADER_SOURCE: &str =
    include_str!("../shaders/chunk_gated_delta_rule_tri_solve_invert.metal");

/// Iter-4 fixed BT (matches the rest of the chunk pipeline).
pub const FIXED_BT: u32 = 64;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "chunk_tri_solve_invert_f32",
        CHUNK_TRI_SOLVE_INVERT_SHADER_SOURCE,
    );
}

/// Shape parameters for the per-chunk-block tri-solve invert kernel.
#[derive(Debug, Clone, Copy)]
pub struct ChunkTriSolveInvertParams {
    pub b: u32,
    pub t: u32,
    pub h: u32,
    pub bt: u32,
}

impl ChunkTriSolveInvertParams {
    pub fn num_chunks(&self) -> u32 {
        self.t.div_ceil(self.bt)
    }
}

fn validate(
    p: &ChunkTriSolveInvertParams,
    a_strict: &MlxBuffer,
    a_inv: &MlxBuffer,
) -> Result<()> {
    if p.b == 0 || p.t == 0 || p.h == 0 || p.bt == 0 {
        return Err(MlxError::InvalidArgument(
            "chunk_tri_solve_invert: all dims must be > 0".into(),
        ));
    }
    if p.bt != FIXED_BT {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_tri_solve_invert (iter 4): bt must be {} (got {})",
            FIXED_BT, p.bt
        )));
    }
    if p.t % p.bt != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "chunk_tri_solve_invert (iter 4): t ({}) must be a multiple of bt ({})",
            p.t, p.bt
        )));
    }

    let elems = (p.b * p.t * p.h * p.bt) as usize;
    for (name, buf) in [("A_strict", a_strict), ("A_inv", a_inv)] {
        if buf.element_count() != elems {
            return Err(MlxError::InvalidArgument(format!(
                "chunk_tri_solve_invert: {} element count {} != expected {}",
                name,
                buf.element_count(),
                elems
            )));
        }
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "chunk_tri_solve_invert: {} must be f32 (got {})",
                name,
                buf.dtype()
            )));
        }
    }
    Ok(())
}

/// Dispatch the per-chunk-block tri-solve invert kernel.
///
/// `params_buf` holds 4 u32: `[B, T, H, BT]`.
/// Use [`build_chunk_tri_solve_invert_params`] to build it.
pub fn dispatch_chunk_tri_solve_invert(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    a_strict: &MlxBuffer,
    a_inv: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: ChunkTriSolveInvertParams,
) -> Result<()> {
    validate(&p, a_strict, a_inv)?;

    let pipeline = registry.get_pipeline("chunk_tri_solve_invert_f32", device)?;

    // Grid: (NT, H, B). One threadgroup per (chunk, head, batch).
    let grid_tgs = MTLSize::new(p.num_chunks() as u64, p.h as u64, p.b as u64);

    // Threadgroup: BT threads. One thread per output column j ∈ [0, BT).
    let tg = MTLSize::new(p.bt as u64, 1, 1);

    // Threadgroup memory: l_tile + x_tile = 2 * BT*BT * 4 bytes.
    // At BT=64, that's 32 KB — exactly the M5 Max cap.
    let shared_bytes: u64 = 2 * (p.bt as u64) * (p.bt as u64) * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, a_strict), (1, a_inv), (2, params_buf)],
        &[(0, shared_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// Build the 4-u32 params buffer: `[B, T, H, BT]`.
pub fn build_chunk_tri_solve_invert_params(
    device: &crate::MlxDevice,
    p: ChunkTriSolveInvertParams,
) -> Result<MlxBuffer> {
    let mut buf = device.alloc_buffer(4 * 4, DType::U32, vec![4])?;
    {
        let s = buf.as_mut_slice::<u32>()?;
        s[0] = p.b;
        s[1] = p.t;
        s[2] = p.h;
        s[3] = p.bt;
    }
    Ok(buf)
}
