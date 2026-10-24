//! Pure checked payload plan for the existing direct-allocation arena.
//! No allocator/domain switch, GPU execution or complete memory-bound claim.
use super::{CUMSUM_PARAM_WORDS, FIXED_BT, MAX_K, MAX_V};
use crate::ops::chunk_gated_delta_rule_tri_solve_invert::INVERT_PARAM_WORDS;
use crate::ops::gated_delta_net_chunk::CHUNK_PARAM_WORDS;
use crate::ops::gated_delta_net_chunk_o::CHUNK_O_PARAM_WORDS;
use crate::ops::gated_delta_net_kkt::KKT_PARAM_WORDS;
use crate::ops::gated_delta_net_recompute_wu::RECOMPUTE_WU_PARAM_WORDS;
use crate::{DType, MlxError, Result};

/// Thirteen owned roots in the unchanged constructor allocation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum ChunkArenaRole {
    Cumsum,
    Strict,
    Inverse,
    W,
    U,
    Hidden,
    VNew,
    CumsumParams,
    KktParams,
    InvertParams,
    RecomputeWuParams,
    ChunkParams,
    ChunkOParams,
}
/// Exact requested flat buffer layout. This is not Metal allocatedSize/footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkArenaBufferPlan {
    role: ChunkArenaRole,
    dtype: DType,
    elements: usize,
    bytes: usize,
}
impl ChunkArenaBufferPlan {
    pub fn role(&self) -> ChunkArenaRole {
        self.role
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Existing allocator exposes flat one-dimensional shapes, not semantic axes.
    pub fn shape(&self) -> [usize; 1] {
        [self.elements]
    }
    pub fn requested_bytes(&self) -> usize {
        self.bytes
    }
}
/// Immutable thirteen-root constructor plan; no device, metadata or lifetime bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkInternalArenaPlan {
    roots: [ChunkArenaBufferPlan; 13],
    requested_bytes: u64,
}
impl ChunkInternalArenaPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn for_shape(b: u32, t: u32, hg: u32, h: u32, k: u32, v: u32, bt: u32) -> Result<Self> {
        if b == 0 || t == 0 || hg == 0 || h == 0 || k == 0 || v == 0 || bt == 0 {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: zero dim b={} t={} hg={} h={} k={} v={} bt={}",
                b, t, hg, h, k, v, bt,
            )));
        }
        if h % hg != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: H ({}) must be a multiple of Hg ({})",
                h, hg
            )));
        }
        if k != MAX_K {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: K ({}) must equal MAX_K = {} \
                 (sub-kernel hard constraint)",
                k, MAX_K
            )));
        }
        if v > MAX_V {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: V ({}) > MAX_V ({})",
                v, MAX_V
            )));
        }
        if bt != FIXED_BT {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: bt ({}) must equal FIXED_BT ({})",
                bt, FIXED_BT
            )));
        }
        if t % bt != 0 {
            return Err(MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: t ({}) must be a multiple of bt ({})",
                t, bt
            )));
        }

        let nt = t.div_ceil(bt);
        use ChunkArenaRole::*;
        let roots = [
            buffer(Cumsum, DType::F32, &[b, t, h])?,
            buffer(Strict, DType::F32, &[b, t, h, bt])?,
            buffer(Inverse, DType::F32, &[b, t, h, bt])?,
            buffer(W, DType::BF16, &[b, t, h, k])?,
            buffer(U, DType::BF16, &[b, t, h, v])?,
            buffer(Hidden, DType::BF16, &[b, nt, h, v, k])?,
            buffer(VNew, DType::BF16, &[b, t, h, v])?,
            parameter(CumsumParams, CUMSUM_PARAM_WORDS)?,
            parameter(KktParams, KKT_PARAM_WORDS)?,
            parameter(InvertParams, INVERT_PARAM_WORDS)?,
            parameter(RecomputeWuParams, RECOMPUTE_WU_PARAM_WORDS)?,
            parameter(ChunkParams, CHUNK_PARAM_WORDS)?,
            parameter(ChunkOParams, CHUNK_O_PARAM_WORDS)?,
        ];
        let requested_bytes = roots.iter().try_fold(0u64, |sum, root| {
            sum.checked_add(root.bytes as u64).ok_or_else(|| {
                MlxError::InvalidArgument("ChunkInternalArena::new: total byte overflow".into())
            })
        })?;
        Ok(Self {
            roots,
            requested_bytes,
        })
    }
    pub fn roots(&self) -> &[ChunkArenaBufferPlan; 13] {
        &self.roots
    }
    pub fn root(&self, role: ChunkArenaRole) -> &ChunkArenaBufferPlan {
        &self.roots[role as usize]
    }
    pub fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }
}
fn buffer(role: ChunkArenaRole, dtype: DType, dims: &[u32]) -> Result<ChunkArenaBufferPlan> {
    // Preserve the constructor's original u32 element-product domain. Reject
    // overflow explicitly instead of debug panic/release wrap or silent widening
    // into an unreviewed GPU uint-index range.
    let elements = dims.iter().try_fold(1u32, |n, d| {
        n.checked_mul(*d).ok_or_else(|| {
            MlxError::InvalidArgument(format!(
                "ChunkInternalArena::new: {role:?} element overflow"
            ))
        })
    })?;
    let elements = usize::try_from(elements).map_err(|_| {
        MlxError::InvalidArgument("ChunkInternalArena::new: element address-space overflow".into())
    })?;
    let bytes = elements.checked_mul(dtype.size_of()).ok_or_else(|| {
        MlxError::InvalidArgument(format!("ChunkInternalArena::new: {role:?} byte overflow"))
    })?;
    Ok(ChunkArenaBufferPlan {
        role,
        dtype,
        elements,
        bytes,
    })
}
fn parameter(role: ChunkArenaRole, words: usize) -> Result<ChunkArenaBufferPlan> {
    let count = u32::try_from(words).map_err(|_| {
        MlxError::InvalidArgument("ChunkInternalArena::new: parameter count overflow".into())
    })?;
    buffer(role, DType::U32, &[count])
}

#[cfg(test)]
#[path = "arena_plan_tests.rs"]
mod tests;
