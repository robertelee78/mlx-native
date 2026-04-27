//! Wave 5b — chunk-parallel Gated DeltaNet inter-chunk state-recurrence kernel.
//!
//! This is the *one* kernel in the chunk-parallel pipeline that has no
//! off-the-shelf Apple Metal substitute: the inter-chunk state recurrence
//! that drives the FLA chunk_delta_h primitive. The other primitives
//! (cumsum, tri_solve, dense_mm_bf16, etc.) already exist in mlx-native
//! and are composed by the orchestrator (iter 2).
//!
//! Spec source:
//! - arXiv 2412.06464 §4 (Yang–Hatamizadeh 2024; chunkwise parallelization)
//! - FLA reference: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` at
//!   /opt/vllm/vllm/model_executor/layers/fla/ops/chunk_delta_h.py:43-298
//!
//! No FLA / Triton / CUDA code is copied — the Metal shader is a
//! re-derivation from the math + the structural pattern of FLA's kernel.
//!
//! # Algorithm
//!
//! For each `(batch, head)`:
//!
//! ```text
//! b_h := h0[b, head]                       # [V, K] f32, initial state
//! for i_t in 0..NT:
//!     h[b, i_t, head]  := bf16(b_h)        # snapshot at chunk start
//!     b_w := w[b, t_chunk, head, :]        # [BT, K] bf16
//!     b_u := u[b, t_chunk, head, :]        # [BT, V] bf16
//!     b_v := b_u - b_w @ b_h.T             # [BT, V] f32 -> bf16 store
//!     v_new[b, t_chunk, head] := bf16(b_v)
//!
//!     g_last := g[b, last_t, head]
//!     g_blk  := g[b, t_chunk, head]
//!     b_v *= exp(g_last - g_blk)[:, None]
//!     b_h *= exp(g_last)
//!
//!     b_k := k[b, t_chunk, head/group_ratio, :]    # GQA broadcast
//!     b_h += b_v.T @ b_k                            # outer accumulate
//!
//! final_state[b, head] := b_h              # f32 store
//! ```
//!
//! All matrix dots run bf16 → f32 accumulator → keep f32 in `b_h`.
//!
//! # Memory layouts (innermost-first / column-major-ish)
//!
//! Inputs:
//! - `k`     : `[B, T, Hg, K]` bf16  — token-major, K-head, K innermost
//! - `w`     : `[B, T, H,  K]` bf16  — WY-projected K (per V-head)
//! - `u`     : `[B, T, H,  V]` bf16  — WY-projected V (per V-head)
//! - `g`     : `[B, T, H]`     f32   — per-chunk-cumsumed log-decay
//! - `h0`    : `[B, H, V, K]`  f32   — initial state
//!
//! Outputs:
//! - `h_out`       : `[B, NT, H, V, K]` bf16  — chunk-start states
//! - `v_new`       : `[B, T, H, V]`     bf16  — post-recurrence new V
//! - `final_state` : `[B, H, V, K]`     f32   — state after last chunk
//!
//! These layouts intentionally match the FLA reference (so the Python
//! fixture is byte-comparable to the GPU output). Byte-identical layout
//! lets the test load fixtures directly into device buffers without any
//! transpose.
//!
//! # Threading model (will be implemented in commit 3)
//!
//! Grid: `(NV_TILES, H, B)` where `NV_TILES = ceil(V / BV)` and `BV = 32`
//! by default. Each threadgroup owns one `(V-tile, head, batch)` slice and
//! sweeps the chunk axis sequentially. The state b_h tile lives in
//! threadgroup memory; per-chunk K/W/U/G blocks are streamed.
//!
//! Threadgroup size: 128 threads (4 simdgroups × 32 lanes); same as
//! `flash_attn_prefill_d256`'s mid-tier config. No simdgroup MMA in iter 1
//! — start with explicit per-thread MAC loops; SIMD MMA is iter 2 perf.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static GATED_DELTA_NET_CHUNK_SHADER_SOURCE: &str =
    include_str!("../shaders/gated_delta_net_chunk.metal");

/// Hard cap on per-tile head-dim K for iter 1.
///
/// The threadgroup-memory budget for the `bh` tile is `BV * K * 4` bytes,
/// plus `BT * BV * 4` bytes for `bv_stage`. With `BT = 64`, `BV = 32`:
///   bv_stage = 64 * 32 * 4 = 8 KB
///   bh tile  = 32 * K  * 4 = 128 * K bytes
/// M5 Max threadgroup memory cap is 32 KB, so:
///   8192 + 128 * K <= 32768  =>  K <= 192
///
/// FLA's K=256 path uses a 4-bank `b_h1..b_h4` partition that we don't
/// implement in iter 1; iter 2 will lift this cap by porting the bank
/// split. Until then, we reject K > 192 with a clear error.
///
/// Qwen3.6 uses K = 128, well under the cap.
pub const MAX_K: u32 = 192;
/// Hard cap on per-tile head-dim V (same threadgroup-memory budget; V is
/// tiled by BV so any V <= 256 still produces 32-element tiles, but we
/// keep the cap symmetric with K for documentation purposes).
pub const MAX_V: u32 = 256;

/// Default V-tile width — matches FLA's mid-config autotune choice.
/// The shader hard-codes this for iter 1; future iters will autotune.
pub const DEFAULT_BV: u32 = 32;

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "gated_delta_net_chunk_inter_state_bf16",
        GATED_DELTA_NET_CHUNK_SHADER_SOURCE,
    );
}

/// Shape parameters for the chunk-parallel inter-chunk state kernel.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaNetChunkParams {
    /// Batch size.
    pub b: u32,
    /// Sequence length (must be a multiple of `bt` for iter 1; non-multiple
    /// support arrives in iter 2 with explicit boundary masks).
    pub t: u32,
    /// K-head count (Hg).
    pub hg: u32,
    /// V-head count (H). `H % Hg == 0` is required for GQA.
    pub h: u32,
    /// Per-head K dimension.
    pub k: u32,
    /// Per-head V dimension.
    pub v: u32,
    /// Chunk size (BT). Recommended 64; matches FLA / llama.cpp.
    pub bt: u32,
}

impl GatedDeltaNetChunkParams {
    /// Number of chunks (`ceil(t / bt)`).
    pub fn num_chunks(&self) -> u32 {
        self.t.div_ceil(self.bt)
    }

    /// GQA group ratio (V-heads per K-head).
    pub fn group_ratio(&self) -> u32 {
        self.h / self.hg
    }
}

fn validate(
    p: &GatedDeltaNetChunkParams,
    k: &MlxBuffer,
    w: &MlxBuffer,
    u: &MlxBuffer,
    g: &MlxBuffer,
    h0: &MlxBuffer,
    h_out: &MlxBuffer,
    v_new: &MlxBuffer,
    final_state: &MlxBuffer,
) -> Result<()> {
    if p.b == 0 || p.t == 0 || p.hg == 0 || p.h == 0 || p.k == 0 || p.v == 0 || p.bt == 0 {
        return Err(MlxError::InvalidArgument(
            "gated_delta_net_chunk: all dims must be > 0".into(),
        ));
    }
    if p.h % p.hg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk: h ({}) must be a multiple of hg ({})",
            p.h, p.hg
        )));
    }
    if p.k > MAX_K {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk: K ({}) exceeds iter-1 32 KB threadgroup memory \
             budget (MAX_K = {}); iter-2 will lift this when the FLA b_h1..b_h4 \
             bank split is ported",
            p.k, MAX_K
        )));
    }
    if p.v > MAX_V {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk: v ({}) must be <= MAX_V ({})",
            p.v, MAX_V
        )));
    }
    // Explicit threadgroup-memory accounting (defense-in-depth — MAX_K is
    // chosen so this branch is unreachable when `bt = 64`, `bv = 32`, but
    // future tile-size changes need to keep the inequality tight).
    let bv = DEFAULT_BV;
    let shared_bytes = ((p.bt * bv) + (bv * p.k)) as u64 * 4;
    const M5_MAX_TG_MEM_BYTES: u64 = 32 * 1024;
    if shared_bytes > M5_MAX_TG_MEM_BYTES {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk: threadgroup memory {} bytes exceeds M5 Max \
             cap of {} bytes (bt={}, bv={}, k={})",
            shared_bytes, M5_MAX_TG_MEM_BYTES, p.bt, bv, p.k
        )));
    }
    if p.t % p.bt != 0 {
        // Iter 1 limitation; iter 2 lifts via boundary masks.
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk (iter 1): t ({}) must be a multiple of bt ({})",
            p.t, p.bt
        )));
    }
    if p.bt != 64 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk (iter 1): bt must be 64 (got {})",
            p.bt
        )));
    }
    if p.v % DEFAULT_BV != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk (iter 1): v ({}) must be a multiple of BV ({})",
            p.v, DEFAULT_BV
        )));
    }

    let nt = p.num_chunks() as usize;

    let k_elems = (p.b * p.t * p.hg * p.k) as usize;
    let w_elems = (p.b * p.t * p.h * p.k) as usize;
    let u_elems = (p.b * p.t * p.h * p.v) as usize;
    let g_elems = (p.b * p.t * p.h) as usize;
    let h0_elems = (p.b * p.h * p.v * p.k) as usize;
    let h_out_elems = p.b as usize * nt * (p.h * p.v * p.k) as usize;
    let v_new_elems = u_elems;
    let final_elems = h0_elems;

    let bf16_inputs: [(&str, &MlxBuffer, usize); 4] = [
        ("k", k, k_elems),
        ("w", w, w_elems),
        ("u", u, u_elems),
        ("v_new", v_new, v_new_elems),
    ];
    for (name, buf, exp) in bf16_inputs {
        if buf.element_count() != exp {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_chunk: {} element count {} != expected {}",
                name,
                buf.element_count(),
                exp
            )));
        }
        if buf.dtype() != DType::BF16 {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_chunk: {} must be bf16 (got {})",
                name,
                buf.dtype()
            )));
        }
    }

    let f32_buffers: [(&str, &MlxBuffer, usize); 4] = [
        ("g", g, g_elems),
        ("h0", h0, h0_elems),
        ("final_state", final_state, final_elems),
        ("h_out_check", h_out, h_out_elems), // size only — dtype check below
    ];
    for (name, buf, exp) in f32_buffers {
        if buf.element_count() != exp {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_chunk: {} element count {} != expected {}",
                name,
                buf.element_count(),
                exp
            )));
        }
    }

    // h_out is bf16; the size check above covers it but the dtype is bf16.
    if h_out.dtype() != DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk: h_out must be bf16 (got {})",
            h_out.dtype()
        )));
    }

    for (name, buf) in [("g", g), ("h0", h0), ("final_state", final_state)] {
        if buf.dtype() != DType::F32 {
            return Err(MlxError::InvalidArgument(format!(
                "gated_delta_net_chunk: {} must be f32 (got {})",
                name,
                buf.dtype()
            )));
        }
    }

    Ok(())
}

/// Dispatch the chunk-parallel inter-chunk state recurrence kernel.
///
/// `params_buf` holds 8 u32: `[B, T, Hg, H, K, V, BT, NT]`.
/// Use [`build_gated_delta_net_chunk_params`] to build it.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gated_delta_net_chunk_inter_state(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    k: &MlxBuffer,
    w: &MlxBuffer,
    u: &MlxBuffer,
    g: &MlxBuffer,
    h0: &MlxBuffer,
    h_out: &MlxBuffer,
    v_new: &MlxBuffer,
    final_state: &MlxBuffer,
    params_buf: &MlxBuffer,
    p: GatedDeltaNetChunkParams,
) -> Result<()> {
    validate(&p, k, w, u, g, h0, h_out, v_new, final_state)?;

    let pipeline = registry.get_pipeline("gated_delta_net_chunk_inter_state_bf16", device)?;

    // Grid: (NV_TILES, H, B). Each threadgroup walks the T-axis chunks
    // serially.
    let nv_tiles = (p.v / DEFAULT_BV) as u64;
    let grid_tgs = MTLSize::new(nv_tiles, p.h as u64, p.b as u64);

    // Threadgroup: 128 threads (4 simdgroups × 32 lanes). Same as
    // flash_attn_prefill_d256 mid config.
    let tg = MTLSize::new(128, 1, 1);

    // Threadgroup memory budget — see header comment in
    // gated_delta_net_chunk.metal for the exact accounting.
    //   bv_stage : BT × BV × 4 = 64 × 32 × 4 = 8 KB   (per-thread bv -> all-threads)
    //   bh tile  : BV × K  × 4 = 32 × 128 × 4 = 16 KB (running f32 state)
    // Total: 24 KB.  M5 Max max threadgroup memory is 32 KB, so this fits
    // with 8 KB headroom.
    let shared_floats: u64 = ((p.bt * DEFAULT_BV) + (DEFAULT_BV * p.k)) as u64;
    let shared_bytes = shared_floats * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, k),
            (1, w),
            (2, u),
            (3, g),
            (4, h0),
            (5, h_out),
            (6, v_new),
            (7, final_state),
            (8, params_buf),
        ],
        &[(0, shared_bytes)],
        grid_tgs,
        tg,
    );

    Ok(())
}

/// Build the 8-u32 params buffer:
/// `[B, T, Hg, H, K, V, BT, NT]`.
pub fn build_gated_delta_net_chunk_params(
    device: &crate::MlxDevice,
    p: GatedDeltaNetChunkParams,
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
    //! Wave 5b.1 iter 4.5 (T2) — K=256 clamp unit test.
    //!
    //! Closes the consistency gap left by iter 4 commit 3e19baa: the K-cap
    //! error message in this op was tightened to uppercase "K", but no
    //! `validate_rejects_k_above_max` test was added (the other 3 ops in the
    //! chunk pipeline — kkt, recompute_w_u, chunk_o — all have one). This
    //! test mirrors their pattern.
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
        // Buffer dtypes match the op's actual signature (k/w/u/h0 are bf16,
        // g is f32, h_out/v_new/final_state are bf16/bf16/f32 respectively
        // per the iter-1 forward-pass contract).
        let k_buf = dummy_buf(&device, DType::BF16);
        let w_buf = dummy_buf(&device, DType::BF16);
        let u_buf = dummy_buf(&device, DType::BF16);
        let g_buf = dummy_buf(&device, DType::F32);
        let h0_buf = dummy_buf(&device, DType::F32);
        let h_out_buf = dummy_buf(&device, DType::BF16);
        let v_new_buf = dummy_buf(&device, DType::BF16);
        let final_state_buf = dummy_buf(&device, DType::F32);

        let p = GatedDeltaNetChunkParams {
            b: 1,
            t: 128,
            hg: 2,
            h: 4,
            k: 256, // > MAX_K (192) — must reject.
            v: 128,
            bt: 64,
        };

        let err = validate(
            &p,
            &k_buf,
            &w_buf,
            &u_buf,
            &g_buf,
            &h0_buf,
            &h_out_buf,
            &v_new_buf,
            &final_state_buf,
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
