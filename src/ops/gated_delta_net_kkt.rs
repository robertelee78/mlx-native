//! Wave 5b.1 iter 2 — chunk_scaled_dot_kkt kernel host dispatch.
//!
//! Spec source:
//! - FLA reference: `chunk_scaled_dot_kkt_fwd_kernel` at
//!   /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:36-99
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
//! b_g       = g[b, t_chunk, i_h]                               # [BT] f32 (cumsumed)
//! b_A       = zeros([BT, BT])                                  # f32
//! for i_k in 0..(K // BK):
//!     b_k       = k[b, t_chunk, kh, i_k*BK:(i_k+1)*BK]         # [BT, BK] bf16
//!     b_kb      = b_k.float() * b_beta[:, None]                 # f32
//!     b_kb_bf16 = bfloat(b_kb)                                  # FLA :86 cast
//!     b_A      += b_kb_bf16.float() @ b_k.float().T             # [BT, BT] f32
//! b_A *= exp(b_g[:, None] - b_g[None, :])                      # FLA :91-92
//! b_A  = where(row > col, b_A, 0)                              # FLA :94-95 strict-lower
//! store p_A : [B, T, H, BT] f32 (output_dtype = f32 per FLA :109)
//! ```
//!
//! # Memory layouts (innermost-first / column-major-ish)
//!
//! Inputs:
//! - `k`:    `[B, T, Hg, K]`  bf16  — K innermost
//! - `beta`: `[B, T, H]`      f32   — H innermost
//! - `g`:    `[B, T, H]`      f32   — H innermost
//!
//! Output:
//! - `A`:    `[B, T, H, BT]`  f32   — BT innermost, then H, then T, then B
//!
//! # Threading model
//!
//! Grid: `(NT, H, B)`. One threadgroup per `(chunk, head, batch)`. Each
//! threadgroup owns one `[BT, BT]` output tile.
//!
//! Threadgroup size: 256 threads (8 simdgroups × 32 lanes), flat 1D. With
//! BT=64, the [BT,BT]=4096 output cells are split 16 cells/thread. The
//! [BT,BK]=4096 input k cells are split 16 cells/thread for cooperative load.
//!
//! # Threadgroup memory budget (BT=64, BK=64)
//!
//!   bk_stage : BT × BK × 2 bytes (bf16) = 64 × 64 × 2 = 8 KB
//!   ba_acc   : BT × BT × 4 bytes (f32)  = 64 × 64 × 4 = 16 KB
//!   Total: 24 KB.  M5 Max threadgroup memory cap is 32 KB → fits with 8 KB headroom.
//!
//! # Validation
//!
//! - K cap matches iter-1: K ≤ 192 (iter-3 will autotune past this).
//! - BT must equal 64 (iter-2 fixed; iter-3 autotune).
//! - BK fixed at 64; the kernel iterates K/BK times internally.
//! - T must be a multiple of BT (no partial-chunk masking in iter 2).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static GATED_DELTA_NET_KKT_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net_kkt.metal");

/// Hard cap on per-tile head-dim K. Matches iter-1 chunk kernel cap; the
/// 32 KB threadgroup-memory budget is tight at K = 192 (24 KB used).
/// Iter-3 will lift this when the autotuned BK schedule lands.
pub const MAX_K: u32 = 192;

/// Default BK split — iterating K/BK times within the kernel.
pub const DEFAULT_BK: u32 = 64;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "gated_delta_net_kkt_bf16",
        GATED_DELTA_NET_KKT_SHADER_SOURCE,
    );
}

/// Shape parameters for the kkt kernel.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetKktParams {
    /// Batch size.
    pub b: u32,
    /// Sequence length (must be a multiple of `bt` for iter 2).
    pub t: u32,
    /// K-head count (Hg).
    pub hg: u32,
    /// V-head count (H). `H % Hg == 0` is required for GQA.
    pub h: u32,
    /// Per-head K dimension.
    pub k: u32,
    /// Chunk size (BT).
    pub bt: u32,
}

impl GatedDeltaNetKktParams {
    /// Number of chunks (`ceil(t / bt)`).
    pub fn num_chunks(&self) -> u32 {
        self.t.div_ceil(self.bt)
    }
}

fn validate(
    p: &GatedDeltaNetKktParams,
    k: &MlxBuffer,
    beta: &MlxBuffer,
    g: &MlxBuffer,
    a: &MlxBuffer,
) -> Result<()> {
    if p.b == 0 || p.t == 0 || p.hg == 0 || p.h == 0 || p.k == 0 || p.bt == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_kkt: all dims must be > 0".into(),
        ));
    }
    if p.h % p.hg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt: h ({}) must be a multiple of hg ({})",
            p.h, p.hg
        )));
    }
    if p.k > MAX_K {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt: k ({}) exceeds iter-2 32 KB threadgroup memory \
             budget (max k = {}); iter-3 will autotune past this",
            p.k, MAX_K
        )));
    }
    if p.bt != 64 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt (iter 2): bt must be 64 (got {})",
            p.bt
        )));
    }
    if p.t % p.bt != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt (iter 2): t ({}) must be a multiple of bt ({})",
            p.t, p.bt
        )));
    }
    if p.k % DEFAULT_BK != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt (iter 2): k ({}) must be a multiple of BK ({})",
            p.k, DEFAULT_BK
        )));
    }
    // Defense-in-depth threadgroup-mem accounting.
    //   bk_stage : BT * BK * 2 bytes (bf16) = 8 KB at BT=64, BK=64
    //   ba_acc   : BT * BT * 4 bytes (f32)  = 16 KB at BT=64
    let shared_bytes: u64 = ((p.bt * DEFAULT_BK) as u64) * 2 + ((p.bt * p.bt) as u64) * 4;
    const M5_MAX_TG_MEM_BYTES: u64 = 32 * 1024;
    if shared_bytes > M5_MAX_TG_MEM_BYTES {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt: threadgroup memory {} bytes exceeds M5 Max \
             cap of {} bytes (bt={}, bk={}, k={})",
            shared_bytes, M5_MAX_TG_MEM_BYTES, p.bt, DEFAULT_BK, p.k
        )));
    }

    let k_elems = (p.b * p.t * p.hg * p.k) as usize;
    let beta_elems = (p.b * p.t * p.h) as usize;
    let g_elems = (p.b * p.t * p.h) as usize;
    let a_elems = (p.b * p.t * p.h * p.bt) as usize;

    if k.element_count() != k_elems || k.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt: k must be bf16[{}] (got {} {})",
            k_elems,
            k.element_count(),
            k.dtype()
        )));
    }
    if beta.element_count() != beta_elems || beta.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt: beta must be f32[{}] (got {} {})",
            beta_elems,
            beta.element_count(),
            beta.dtype()
        )));
    }
    if g.element_count() != g_elems || g.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt: g must be f32[{}] (got {} {})",
            g_elems,
            g.element_count(),
            g.dtype()
        )));
    }
    if a.element_count() != a_elems || a.dtype() != DType::F32 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_kkt: A must be f32[{}] (got {} {})",
            a_elems,
            a.element_count(),
            a.dtype()
        )));
    }

    Ok(())
}

/// Dispatch the chunk_scaled_dot_kkt kernel.
///
/// `params_buf` holds 8 u32: `[B, T, Hg, H, K, BT, NT, BK]`.
/// Use [`build_gated_delta_net_kkt_params`] to build it.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net_kkt(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    k: &MlxBuffer,
    beta: &MlxBuffer,
    g: &MlxBuffer,
    a: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: GatedDeltaNetKktParams,
) -> Result<()> {
    validate(&p, k, beta, g, a)?;

    let pipeline = registry.get_pipeline("gated_delta_net_kkt_bf16", device)?;

    // Grid: one threadgroup per (chunk, head, batch).
    let grid_tgs = MTLSize::new(p.num_chunks() as u64, p.h as u64, p.b as u64);

    // Threadgroup: 256 threads (8 simdgroups × 32 lanes). The output [BT,BT]
    // = 4096 cells split 16 cells/thread; the bk-tile [BT,BK] = 4096 cells
    // also split 16/thread for cooperative load.
    let tg = MTLSize::new(256, 1, 1);

    // Threadgroup memory: bk_stage (BT*BK bf16 = 8 KB) + ba_acc (BT*BT f32 = 16 KB)
    let bk_stage_bytes: u64 = (p.bt as u64) * (DEFAULT_BK as u64) * 2;
    let ba_acc_bytes: u64 = (p.bt as u64) * (p.bt as u64) * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, k), (1, beta), (2, g), (3, a), (4, params_buf)],
        &[(0, bk_stage_bytes), (1, ba_acc_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// Build the 8-u32 params buffer:
/// `[B, T, Hg, H, K, BT, NT, BK]`.
pub fn build_gated_delta_net_kkt_params(
    device: &crate::MlxDevice,
    p: GatedDeltaNetKktParams,
) -> Result<MlxBuffer> {
    let mut buf = device.alloc_buffer(8 * 4, DType::U32, vec![8])?;
    {
        let s = buf.as_mut_slice::<u32>()?;
        s[0] = p.b;
        s[1] = p.t;
        s[2] = p.hg;
        s[3] = p.h;
        s[4] = p.k;
        s[5] = p.bt;
        s[6] = p.num_chunks();
        s[7] = DEFAULT_BK;
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

    /// Allocate a 1-byte dummy buffer of the given dtype. The K-cap check
    /// at the top of `validate` fires before any buffer-size check, so
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
        let beta_buf = dummy_buf(&device, DType::F32);
        let g_buf = dummy_buf(&device, DType::F32);
        let a_buf = dummy_buf(&device, DType::F32);

        let p = GatedDeltaNetKktParams {
            b: 1,
            t: 128,
            hg: 2,
            h: 4,
            k: 256, // > MAX_K (192) — must reject.
            bt: 64,
        };

        let err = validate(&p, &k_buf, &beta_buf, &g_buf, &a_buf)
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
