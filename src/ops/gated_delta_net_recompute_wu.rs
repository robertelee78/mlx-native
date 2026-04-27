//! Wave 5b.1 iter 2 — recompute_w_u_fwd Metal kernel host dispatch.
//!
//! Spec source:
//! - FLA reference: `recompute_w_u_fwd_kernel` at
//!   /opt/vllm/vllm/model_executor/layers/fla/ops/wy_fast.py:29-117
//!
//! No FLA / Triton / CUDA code is copied — the Metal shader is a
//! re-derivation from the math + the structural pattern of FLA's kernel.
//!
//! # Algorithm
//!
//! For each `(batch b, V-head i_h, chunk i_t)`:
//!
//! ```text
//! kh        = i_h / (H / Hg)                                   # GQA-mapped K-head
//! b_beta    = beta[b, t_chunk, i_h]                            # [BT] f32
//! b_A       = A[b, t_chunk, i_h, :]                            # [BT, BT] f32 (post-solve)
//! b_g       = exp(g[b, t_chunk, i_h])                          # [BT] f32  (FLA :72)
//!
//! # u-loop (FLA wy_fast.py:74-94):
//! for i_v in 0..(V // BV):
//!     b_v       = v[b, t_chunk, i_h, i_v*BV:(i_v+1)*BV]        # [BT, BV] bf16
//!     b_vb_bf16 = bfloat(b_v.float() * b_beta[:, None])         # FLA :92 cast
//!     b_u       = b_A @ b_vb_bf16.float()                      # [BT, BV] f32
//!     u[b, t_chunk, i_h, i_v*BV:(i_v+1)*BV] = bf16(b_u)
//!
//! # w-loop (FLA wy_fast.py:96-116):
//! for i_k in 0..(K // BK):
//!     b_k       = k[b, t_chunk, kh, i_k*BK:(i_k+1)*BK]         # [BT, BK] bf16
//!     b_kb_bf16 = bfloat(b_k.float() * b_beta[:, None] * b_g[:, None])  # FLA :114
//!     b_w       = b_A @ b_kb_bf16.float()                      # [BT, BK] f32
//!     w[b, t_chunk, i_h, i_k*BK:(i_k+1)*BK] = bf16(b_w)
//! ```
//!
//! # Memory layouts (innermost-first)
//!
//! Inputs:
//! - `k`:    `[B, T, Hg, K]`  bf16  — K innermost
//! - `v`:    `[B, T, H,  V]`  bf16  — V innermost
//! - `beta`: `[B, T, H]`      f32   — H innermost
//! - `g`:    `[B, T, H]`      f32   — H innermost (cumsumed within chunk)
//! - `A`:    `[B, T, H, BT]`  f32   — BT innermost (post-solve_tril output)
//!
//! Outputs:
//! - `w`:    `[B, T, H, K]`   bf16  — K innermost
//! - `u`:    `[B, T, H, V]`   bf16  — V innermost
//!
//! # Threading model
//!
//! Grid: `(NT, H, B)`. One threadgroup per `(chunk, head, batch)`. Each
//! threadgroup processes both the V-loop and K-loop for its chunk, reusing
//! a shared b_A tile.
//!
//! Threadgroup size: 256 threads. The output tiles are [BT, BV] = [64, 64]
//! and [BT, BK] = [64, 64] = 4096 cells each, split 16 cells/thread.
//!
//! # Threadgroup memory budget (BT=64, BK=BV=64)
//!
//!   ba_tile  : BT * BT * 4 bytes (f32)  = 64 * 64 * 4 = 16 KB
//!   stage    : BT * max(BV, BK) * 2 bytes (bf16) = 64 * 64 * 2 = 8 KB
//!   Total: 24 KB.  M5 Max cap 32 KB → 8 KB headroom.
//!
//! # Validation
//!
//! - K ≤ 192 (matches iter-1 cap; iter-3 will autotune).
//! - V ≤ 256 (same).
//! - BT must be 64 (iter-2 fixed).
//! - BK fixed at 64; BV fixed at 64; T must be a multiple of BT.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static GATED_DELTA_NET_RECOMPUTE_WU_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net_recompute_wu.metal");

pub const MAX_K: u32 = 192;
pub const MAX_V: u32 = 256;
pub const DEFAULT_BK: u32 = 64;
pub const DEFAULT_BV: u32 = 64;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "gated_delta_net_recompute_wu_bf16",
        GATED_DELTA_NET_RECOMPUTE_WU_SHADER_SOURCE,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetRecomputeWuParams {
    pub b: u32,
    pub t: u32,
    pub hg: u32,
    pub h: u32,
    pub k: u32,
    pub v: u32,
    pub bt: u32,
}

impl GatedDeltaNetRecomputeWuParams {
    pub fn num_chunks(&self) -> u32 {
        self.t.div_ceil(self.bt)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate(
    p: &GatedDeltaNetRecomputeWuParams,
    k: &MlxBuffer,
    v: &MlxBuffer,
    beta: &MlxBuffer,
    g: &MlxBuffer,
    a: &MlxBuffer,
    w: &MlxBuffer,
    u: &MlxBuffer,
) -> Result<()> {
    if p.b == 0 || p.t == 0 || p.hg == 0 || p.h == 0 || p.k == 0 || p.v == 0 || p.bt == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_recompute_wu: all dims must be > 0".into(),
        ));
    }
    if p.h % p.hg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: h ({}) must be a multiple of hg ({})",
            p.h, p.hg
        )));
    }
    if p.k > MAX_K {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: K ({}) exceeds iter-2 32 KB threadgroup \
             memory budget (MAX_K = {}); iter-3 will autotune past this",
            p.k, MAX_K
        )));
    }
    if p.v > MAX_V {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: v ({}) must be <= MAX_V ({})",
            p.v, MAX_V
        )));
    }
    if p.bt != 64 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu (iter 2): bt must be 64 (got {})",
            p.bt
        )));
    }
    if p.t % p.bt != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu (iter 2): t ({}) must be a multiple of bt ({})",
            p.t, p.bt
        )));
    }
    if p.k % DEFAULT_BK != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu (iter 2): k ({}) must be a multiple of BK ({})",
            p.k, DEFAULT_BK
        )));
    }
    if p.v % DEFAULT_BV != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu (iter 2): v ({}) must be a multiple of BV ({})",
            p.v, DEFAULT_BV
        )));
    }

    // Defense-in-depth threadgroup-mem accounting.
    //   ba_tile  : BT * BT * 4 bytes (f32)
    //   stage    : BT * max(BV, BK) * 2 bytes (bf16)
    let stage_width = std::cmp::max(DEFAULT_BV, DEFAULT_BK);
    let shared_bytes: u64 =
        ((p.bt * p.bt) as u64) * 4 + ((p.bt * stage_width) as u64) * 2;
    const M5_MAX_TG_MEM_BYTES: u64 = 32 * 1024;
    if shared_bytes > M5_MAX_TG_MEM_BYTES {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: threadgroup memory {} bytes exceeds M5 Max \
             cap of {} bytes (bt={}, stage_width={})",
            shared_bytes, M5_MAX_TG_MEM_BYTES, p.bt, stage_width
        )));
    }

    let k_elems = (p.b * p.t * p.hg * p.k) as usize;
    let v_elems = (p.b * p.t * p.h * p.v) as usize;
    let beta_elems = (p.b * p.t * p.h) as usize;
    let g_elems = (p.b * p.t * p.h) as usize;
    let a_elems = (p.b * p.t * p.h * p.bt) as usize;
    let w_elems = (p.b * p.t * p.h * p.k) as usize;
    let u_elems = (p.b * p.t * p.h * p.v) as usize;

    if k.element_count() != k_elems || k.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: k must be bf16[{}] (got {} {})",
            k_elems,
            k.element_count(),
            k.dtype()
        )));
    }
    if v.element_count() != v_elems || v.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: v must be bf16[{}] (got {} {})",
            v_elems,
            v.element_count(),
            v.dtype()
        )));
    }
    if beta.element_count() != beta_elems || beta.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: beta must be f32[{}] (got {} {})",
            beta_elems,
            beta.element_count(),
            beta.dtype()
        )));
    }
    if g.element_count() != g_elems || g.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: g must be f32[{}] (got {} {})",
            g_elems,
            g.element_count(),
            g.dtype()
        )));
    }
    if a.element_count() != a_elems || a.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: A must be f32[{}] (got {} {})",
            a_elems,
            a.element_count(),
            a.dtype()
        )));
    }
    if w.element_count() != w_elems || w.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: w must be bf16[{}] (got {} {})",
            w_elems,
            w.element_count(),
            w.dtype()
        )));
    }
    if u.element_count() != u_elems || u.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_recompute_wu: u must be bf16[{}] (got {} {})",
            u_elems,
            u.element_count(),
            u.dtype()
        )));
    }

    Ok(())
}

/// Dispatch the recompute_w_u_fwd kernel.
///
/// `params_buf` holds 8 u32: `[B, T, Hg, H, K, V, BT, NT]`. Use
/// [`build_gated_delta_net_recompute_wu_params`] to build it.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net_recompute_wu(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    k: &MlxBuffer,
    v: &MlxBuffer,
    beta: &MlxBuffer,
    g: &MlxBuffer,
    a: &MlxBuffer,
    w: &MlxBuffer,
    u: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: GatedDeltaNetRecomputeWuParams,
) -> Result<()> {
    validate(&p, k, v, beta, g, a, w, u)?;

    let pipeline = registry.get_pipeline("gated_delta_net_recompute_wu_bf16", device)?;

    // Grid: one threadgroup per (chunk, head, batch).
    let grid_tgs = MTLSize::new(p.num_chunks() as u64, p.h as u64, p.b as u64);

    // Threadgroup: 256 threads.
    let tg = MTLSize::new(256, 1, 1);

    // Threadgroup memory:
    //   ba_tile (16 KB, f32) + stage (8 KB, bf16) = 24 KB
    let ba_tile_bytes: u64 = (p.bt as u64) * (p.bt as u64) * 4;
    let stage_width = std::cmp::max(DEFAULT_BV, DEFAULT_BK) as u64;
    let stage_bytes: u64 = (p.bt as u64) * stage_width * 2;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, k),
            (1, v),
            (2, beta),
            (3, g),
            (4, a),
            (5, w),
            (6, u),
            (7, params_buf),
        ],
        &[(0, ba_tile_bytes), (1, stage_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// Build the 8-u32 params buffer:
/// `[B, T, Hg, H, K, V, BT, NT]`.
pub fn build_gated_delta_net_recompute_wu_params(
    device: &crate::MlxDevice,
    p: GatedDeltaNetRecomputeWuParams,
) -> Result<MlxBuffer> {
    let mut buf = device.alloc_buffer(8 * 4, DType::U32, vec![8])?;
    {
        let s = buf.as_mut_slice::<u32>()?;
        s[0] = p.b;
        s[1] = p.t;
        s[2] = p.hg;
        s[3] = p.h;
        s[4] = p.k;
        s[5] = p.v;
        s[6] = p.bt;
        s[7] = p.num_chunks();
    }
    Ok(buf)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    //! Wave 5b.1 iter 2.5 — closes Codex audit "missed test" finding:
    //! the K=256 rejection path in `validate` had no Rust assertion.
    use super::*;
    use crate::MlxDevice;

    /// Allocate a 1-element dummy buffer of the given dtype. The K-cap
    /// check inside `validate` fires before any buffer-size check, so
    /// these placeholder buffers are sufficient to exercise the error
    /// path.
    fn dummy_buf(device: &MlxDevice, dtype: DType) -> MlxBuffer {
        device
            .alloc_buffer(2, dtype, vec![1])
            .expect("alloc dummy")
    }

    #[test]
    fn validate_rejects_k_above_max() {
        let device = MlxDevice::new().expect("MlxDevice::new");
        let k_buf = dummy_buf(&device, DType::BF16);
        let v_buf = dummy_buf(&device, DType::BF16);
        let beta_buf = dummy_buf(&device, DType::F32);
        let g_buf = dummy_buf(&device, DType::F32);
        let a_buf = dummy_buf(&device, DType::F32);
        let w_buf = dummy_buf(&device, DType::BF16);
        let u_buf = dummy_buf(&device, DType::BF16);

        let p = GatedDeltaNetRecomputeWuParams {
            b: 1,
            t: 128,
            hg: 2,
            h: 4,
            k: 256, // > MAX_K (192) — must reject.
            v: 128,
            bt: 64,
        };

        let err = validate(
            &p, &k_buf, &v_buf, &beta_buf, &g_buf, &a_buf, &w_buf, &u_buf,
        )
        .expect_err("validate must reject K=256");
        let msg = err.to_string();
        assert!(
            msg.contains("256"),
            "expected K=256 in error message, got: {msg}"
        );
        assert!(
            msg.contains("32 KB") || msg.contains("threadgroup"),
            "expected threadgroup-memory-budget context in error, got: {msg}"
        );
        assert!(
            msg.contains("MAX_K = 192") || msg.contains("MAX_K=192"),
            "expected explicit MAX_K cap in error, got: {msg}"
        );
    }
}
