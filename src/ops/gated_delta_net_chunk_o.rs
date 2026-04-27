//! Wave 5b.1 iter 3 — chunk_fwd_o Metal kernel host dispatch.
//!
//! Spec source:
//! - FLA reference: `chunk_fwd_kernel_o` at
//!   /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_o.py:42-138
//!
//! No FLA / Triton / CUDA code is copied — the Metal shader is a
//! re-derivation from the math + the structural pattern of FLA's kernel.
//!
//! # Algorithm
//!
//! For each `(batch b, V-head i_h, chunk i_t, V-tile i_v)`:
//!
//! ```text
//! kh         = i_h / (H / Hg)                                  # GQA-mapped K-head
//! bo_acc     = zeros([BT, BV])                                 # f32
//! bA_acc     = zeros([BT, BT])                                 # f32
//!
//! # K-tile loop (FLA chunk_o.py:93-113):
//! for i_k in 0..(K // BK):
//!     b_q = q[b, t_chunk, kh, i_k*BK:(i_k+1)*BK]               # [BT, BK] bf16
//!     b_k = k[b, t_chunk, kh, i_k*BK:(i_k+1)*BK]               # [BT, BK] bf16
//!     b_h = h[b, i_t, i_h, i_v*BV:(i_v+1)*BV, i_k*BK:...]      # [BV, BK] bf16
//!     bo_acc += b_q · b_h^T                                    # [BT, BV] f32
//!     bA_acc += b_q · b_k^T                                    # [BT, BT] f32
//!
//! # Gate (FLA :115-120; USE_G == True for GDN):
//! bo_acc[i, j] *= exp(g[t_start+i])
//! bA_acc[i, j] *= exp(g[t_start+i] - g[t_start+j])
//!
//! # Causal+diag mask (FLA :122-125 — `>=`, INCLUSIVE of diagonal):
//! bA_acc[i, j] = (i >= j) ? bA_acc[i, j] : 0
//!
//! # bf16 round-trip + closing dot (FLA :137 — both * scale terms preserved):
//! bA_bf16    = bfloat(bA_acc)                                  # post-mask cast
//! v_dot      = bA_bf16 · v_new[b, t_chunk, i_h, i_v*BV:...]    # [BT, BV] f32
//! o_val      = bo_acc * scale + v_dot * scale                  # f32
//! o[b, t_chunk, i_h, i_v*BV:(i_v+1)*BV] = bfloat(o_val)
//! ```
//!
//! # Memory layouts (innermost-first)
//!
//! Inputs:
//! - `q`:     `[B, T, Hg, K]`     bf16  — K innermost
//! - `k`:     `[B, T, Hg, K]`     bf16  — K innermost
//! - `v_new`: `[B, T, H,  V]`     bf16  — V innermost
//! - `h`:     `[B, NT, H, V, K]`  bf16  — K innermost (per chunk-head)
//! - `g`:     `[B, T, H]`         f32   — H innermost (chunk-cumsumed)
//!
//! Output:
//! - `o`:     `[B, T, H, V]`      bf16  — V innermost
//!
//! # Threading model
//!
//! Grid: `(NV, NT, B*H)` where `NV = V/BV`. One threadgroup per
//! `(V-tile, chunk, head)`. Each threadgroup owns one `[BT, BV]` output tile.
//!
//! Threadgroup size: 256 threads. The output tile [BT, BV] = [64, 32] = 2048
//! cells split 8 cells/thread; the [BT, BT] = 4096 cells in `bA_acc` split
//! 16 cells/thread.
//!
//! # Threadgroup memory budget (post Wave 5b.2 iter 2 MMA, BT=64)
//!
//!   bA_stage : BT * BT * 2 bytes (bf16) = 64 * 64 * 2 = 8 KB
//!   Total: 8 KB (down from 28 KB — bo_acc and bA_acc now live in
//!   simdgroup_matrix accumulator registers, not threadgroup memory).
//!   M5 Max cap 32 KB → 24 KB headroom.
//!
//! # Validation
//!
//! - K ≤ 128 (NARROWED from 192 in Wave 5b.2 iter 2, 2026-04-27).
//!   The MMA K-tile count (16 = K/8) is hard-coded in the shader; making
//!   it runtime collapses speedup ~3× (iter 1.5 lesson on inter_state).
//!   Iter-4 will lift this when the FLA bank-split lands.
//! - V ≤ 256 (matches iter-1+2 cap).
//! - BT must be 64 (the BT-tile count `8 = BT/8` is also compile-time).
//! - BK fixed at 32; BV fixed at 32; T must be a multiple of BT.
//! - K must be a multiple of BK; V must be a multiple of BV.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static GATED_DELTA_NET_CHUNK_O_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net_chunk_o.metal");

/// Hard cap on per-tile head-dim K.
///
/// NARROWED from 192 to 128 in Wave 5b.2 iter 2 (2026-04-27). The
/// `simdgroup_matrix<float, 8, 8>` MMA K-tile count is hard-coded at
/// 16 (= K=128 / 8) in `shaders/gated_delta_net_chunk_o.metal`; making
/// it runtime via `K/8u` collapses speedup ~3× because Metal's MMA
/// scheduler needs compile-time loop bounds for tile-sequence unrolling
/// (mirrors the iter 1.5 lesson on `gated_delta_net_chunk_inter_state_bf16`).
///
/// Iter-4 will lift this when the FLA-style bank-split lands (each bank
/// keeps a compile-time-known K-tile count).
pub const MAX_K: u32 = 128;

/// Hard cap on per-tile head-dim V (matches iter-1+2 chunk-pipeline cap).
pub const MAX_V: u32 = 256;

/// Default BK split (K-tile width). Iter-3 fixed; iter-4 will autotune.
pub const DEFAULT_BK: u32 = 32;

/// Default BV split (V-tile width). Iter-3 fixed; iter-4 will autotune.
pub const DEFAULT_BV: u32 = 32;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "gated_delta_net_chunk_o_bf16",
        GATED_DELTA_NET_CHUNK_O_SHADER_SOURCE,
    );
}

/// Shape parameters for the chunk_fwd_o kernel.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetChunkOParams {
    /// Batch size.
    pub b: u32,
    /// Sequence length (must be a multiple of `bt` for iter 3).
    pub t: u32,
    /// K-head count (Hg).
    pub hg: u32,
    /// V-head count (H). `H % Hg == 0` is required for GQA.
    pub h: u32,
    /// Per-head K dimension (must be a multiple of `DEFAULT_BK`).
    pub k: u32,
    /// Per-head V dimension (must be a multiple of `DEFAULT_BV`).
    pub v: u32,
    /// Chunk size (BT).
    pub bt: u32,
    /// Output scale (typically `K^-0.5`).
    pub scale: f32,
}

impl GatedDeltaNetChunkOParams {
    /// Number of chunks (`ceil(t / bt)`).
    pub fn num_chunks(&self) -> u32 {
        self.t.div_ceil(self.bt)
    }

    /// Number of V-tiles per chunk (`v / DEFAULT_BV`).
    pub fn num_v_tiles(&self) -> u32 {
        self.v / DEFAULT_BV
    }
}

#[allow(clippy::too_many_arguments)]
fn validate(
    p: &GatedDeltaNetChunkOParams,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    h: &MlxBuffer,
    g: &MlxBuffer,
    o: &MlxBuffer,
) -> Result<()> {
    if p.b == 0 || p.t == 0 || p.hg == 0 || p.h == 0 || p.k == 0 || p.v == 0 || p.bt == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_chunk_o: all dims must be > 0".into(),
        ));
    }
    if p.h % p.hg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: h ({}) must be a multiple of hg ({})",
            p.h, p.hg
        )));
    }
    // K must be EXACTLY 128 — the simdgroup_matrix MMA K-tile loop is
    // hard-coded at 16 tiles (= K=128/8). K<128 (e.g., 32/64/96) would have
    // the kernel read past the input arrays' rows = out-of-bounds.
    // Codex iter-7 audit (2026-04-27 HIGH-sev) caught this; iter-2.5 fixup
    // narrows from `K <= MAX_K` to `K == 128` exact match. To support
    // K=32/64/96/192/256, port FLA's b_h1..b_h4 bank-split for compile-time
    // tile counts per bank; out-of-scope for Wave 5b.2.
    if p.k != MAX_K {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: K ({}) must equal MAX_K = {} exactly. \
             The simdgroup_matrix MMA K-tile loop is compile-time hard-coded \
             at 16 (= K=128/8) in the shader; runtime loop bounds defeat the \
             MMA scheduler (3.15× regression measured 2026-04-27). Future \
             iters will lift this via FLA's b_h1..b_h4 bank-split.",
            p.k, MAX_K
        )));
    }
    if p.v > MAX_V {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: v ({}) must be <= MAX_V ({})",
            p.v, MAX_V
        )));
    }
    if p.bt != 64 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o (iter 3): bt must be 64 (got {})",
            p.bt
        )));
    }
    if p.t % p.bt != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o (iter 3): t ({}) must be a multiple of bt ({})",
            p.t, p.bt
        )));
    }
    if p.k % DEFAULT_BK != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o (iter 3): k ({}) must be a multiple of BK ({})",
            p.k, DEFAULT_BK
        )));
    }
    if p.v % DEFAULT_BV != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o (iter 3): v ({}) must be a multiple of BV ({})",
            p.v, DEFAULT_BV
        )));
    }
    if !p.scale.is_finite() {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: scale must be finite (got {})",
            p.scale
        )));
    }

    // Defense-in-depth threadgroup-mem accounting (post Wave 5b.2 iter 2).
    //   bA_stage : BT * BT * 2 bytes (bf16)
    let shared_bytes: u64 = ((p.bt * p.bt) as u64) * 2;
    const M5_MAX_TG_MEM_BYTES: u64 = 32 * 1024;
    if shared_bytes > M5_MAX_TG_MEM_BYTES {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: threadgroup memory {} bytes exceeds M5 Max \
             cap of {} bytes (bt={})",
            shared_bytes, M5_MAX_TG_MEM_BYTES, p.bt
        )));
    }

    let nt = p.num_chunks();
    let q_elems = (p.b * p.t * p.hg * p.k) as usize;
    let k_elems = (p.b * p.t * p.hg * p.k) as usize;
    let v_elems = (p.b * p.t * p.h * p.v) as usize;
    let h_elems = (p.b * nt * p.h * p.v * p.k) as usize;
    let g_elems = (p.b * p.t * p.h) as usize;
    let o_elems = (p.b * p.t * p.h * p.v) as usize;

    if q.element_count() != q_elems || q.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: q must be bf16[{}] (got {} {})",
            q_elems,
            q.element_count(),
            q.dtype()
        )));
    }
    if k.element_count() != k_elems || k.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: k must be bf16[{}] (got {} {})",
            k_elems,
            k.element_count(),
            k.dtype()
        )));
    }
    if v.element_count() != v_elems || v.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: v must be bf16[{}] (got {} {})",
            v_elems,
            v.element_count(),
            v.dtype()
        )));
    }
    if h.element_count() != h_elems || h.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: h must be bf16[{}] (got {} {})",
            h_elems,
            h.element_count(),
            h.dtype()
        )));
    }
    if g.element_count() != g_elems || g.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: g must be f32[{}] (got {} {})",
            g_elems,
            g.element_count(),
            g.dtype()
        )));
    }
    if o.element_count() != o_elems || o.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: o must be bf16[{}] (got {} {})",
            o_elems,
            o.element_count(),
            o.dtype()
        )));
    }

    Ok(())
}

/// Dispatch the chunk_fwd_o kernel.
///
/// `params_buf` holds 11 u32: `[B, T, Hg, H, K, V, BT, NT, BK, BV, scale_bits]`
/// where `scale_bits = scale.to_bits()`. Use
/// [`build_gated_delta_net_chunk_o_params`] to build it.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net_chunk_o(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    h: &MlxBuffer,
    g: &MlxBuffer,
    o: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: GatedDeltaNetChunkOParams,
) -> Result<()> {
    validate(&p, q, k, v, h, g, o)?;

    let pipeline = registry.get_pipeline("gated_delta_net_chunk_o_bf16", device)?;

    // Grid: (NV, NT, B*H). One threadgroup per (V-tile, chunk, head).
    let nv = p.num_v_tiles() as u64;
    let nt = p.num_chunks() as u64;
    let bh = (p.b * p.h) as u64;
    let grid_tgs = MTLSize::new(nv, nt, bh);

    // Threadgroup: 256 threads.
    let tg = MTLSize::new(256, 1, 1);

    // Threadgroup memory budget (8 KB at BT=64, post Wave 5b.2 iter 2 MMA).
    //   bA_stage : BT * BT * 2 bytes (bf16) = 8 KB
    let ba_stage_bytes: u64 = (p.bt as u64) * (p.bt as u64) * 2;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, q),
            (1, k),
            (2, v),
            (3, h),
            (4, g),
            (5, o),
            (6, params_buf),
        ],
        &[(0, ba_stage_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// Build the params buffer.
///
/// Layout: 11 u32 slots = `[B, T, Hg, H, K, V, BT, NT, BK, BV, scale_bits]`
/// where `scale_bits = scale.to_bits()`. The kernel reads `params[10]` as
/// `as_type<float>(uint)` to recover the scale value.
pub fn build_gated_delta_net_chunk_o_params(
    device: &crate::MlxDevice,
    p: GatedDeltaNetChunkOParams,
) -> Result<MlxBuffer> {
    let mut buf = device.alloc_buffer(11 * 4, DType::U32, vec![11])?;
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
        s[8] = DEFAULT_BK;
        s[9] = DEFAULT_BV;
        s[10] = p.scale.to_bits();
    }
    Ok(buf)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    //! Wave 5b.1 iter 3 — K=256 clamp unit test.
    //!
    //! Mirrors iter-2.5 discipline (see kkt + recompute_w_u for sibling
    //! tests at /opt/mlx-native/src/ops/gated_delta_net_kkt.rs and
    //! /opt/mlx-native/src/ops/gated_delta_net_recompute_wu.rs). Closes the
    //! validation unit-test gap proactively for the new chunk_fwd_o op.
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
        let q_buf = dummy_buf(&device, DType::BF16);
        let k_buf = dummy_buf(&device, DType::BF16);
        let v_buf = dummy_buf(&device, DType::BF16);
        let h_buf = dummy_buf(&device, DType::BF16);
        let g_buf = dummy_buf(&device, DType::F32);
        let o_buf = dummy_buf(&device, DType::BF16);

        let p = GatedDeltaNetChunkOParams {
            b: 1,
            t: 128,
            hg: 2,
            h: 4,
            k: 256, // > MAX_K (128, narrowed in Wave 5b.2 iter 2) — must reject.
            v: 128,
            bt: 64,
            scale: (128f32).powf(-0.5),
        };

        let err = validate(&p, &q_buf, &k_buf, &v_buf, &h_buf, &g_buf, &o_buf)
            .expect_err("validate must reject K=256");
        let msg = err.to_string();
        assert!(
            msg.contains("256"),
            "expected K=256 in error message, got: {msg}"
        );
        assert!(
            msg.contains("compile-time") || msg.contains("MMA"),
            "expected MMA-compile-time-bound context in error, got: {msg}"
        );
        assert!(
            msg.contains("MAX_K = 128") || msg.contains("MAX_K=128"),
            "expected explicit MAX_K cap in error, got: {msg}"
        );
    }
}

