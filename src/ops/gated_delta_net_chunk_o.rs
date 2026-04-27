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
//! # Threadgroup memory budget (BT=64, BK=32, BV=32)
//!
//!   bo_acc : BT * BV * 4 bytes (f32) = 64 * 32 * 4 = 8  KB
//!   ba_acc : BT * BT * 4 bytes (f32) = 64 * 64 * 4 = 16 KB
//!   stage  : BT * max(BK, BV) * 2 bytes (bf16) = 64 * 32 * 2 = 4 KB
//!   Total: 28 KB.  M5 Max cap 32 KB → 4 KB headroom.
//!
//! # Validation
//!
//! - K ≤ 192 (matches iter-1+2 cap; iter-4 will autotune past this).
//! - V ≤ 256 (same).
//! - BT must be 64 (iter-3 fixed).
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

/// Hard cap on per-tile head-dim K. Matches iter-1+2 chunk-pipeline cap;
/// the 32 KB threadgroup-memory budget is tight, with 4 KB headroom at
/// BK=BV=32. Iter-4 will lift this when the autotuned tile schedule lands.
pub const MAX_K: u32 = 192;

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
    if p.k > MAX_K {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: k ({}) exceeds iter-3 32 KB threadgroup \
             memory budget (max k = {}); iter-4 will autotune past this",
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

    // Defense-in-depth threadgroup-mem accounting.
    //   bo_acc : BT * BV * 4 bytes (f32)
    //   ba_acc : BT * BT * 4 bytes (f32)
    //   stage  : BT * max(BV, BK) * 2 bytes (bf16)
    let stage_width = std::cmp::max(DEFAULT_BV, DEFAULT_BK);
    let shared_bytes: u64 = ((p.bt * DEFAULT_BV) as u64) * 4
        + ((p.bt * p.bt) as u64) * 4
        + ((p.bt * stage_width) as u64) * 2;
    const M5_MAX_TG_MEM_BYTES: u64 = 32 * 1024;
    if shared_bytes > M5_MAX_TG_MEM_BYTES {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk_o: threadgroup memory {} bytes exceeds M5 Max \
             cap of {} bytes (bt={}, bk={}, bv={})",
            shared_bytes, M5_MAX_TG_MEM_BYTES, p.bt, DEFAULT_BK, DEFAULT_BV
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

    // Threadgroup memory budget (28 KB at BT=64, BK=BV=32).
    let bo_acc_bytes: u64 = (p.bt as u64) * (DEFAULT_BV as u64) * 4;
    let ba_acc_bytes: u64 = (p.bt as u64) * (p.bt as u64) * 4;
    let stage_width = std::cmp::max(DEFAULT_BV, DEFAULT_BK) as u64;
    let stage_bytes: u64 = (p.bt as u64) * stage_width * 2;

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
        &[(0, bo_acc_bytes), (1, ba_acc_bytes), (2, stage_bytes)],
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
            k: 256, // > MAX_K (192) — must reject.
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
            msg.contains("32 KB") || msg.contains("threadgroup"),
            "expected threadgroup-memory-budget context in error, got: {msg}"
        );
        assert!(
            msg.contains("max k = 192") || msg.contains("max k=192"),
            "expected explicit max-K cap in error, got: {msg}"
        );
    }
}

