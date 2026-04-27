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

/// Hard cap on per-tile head-dim K (matches FLA's `assert K <= 256`).
/// Qwen3.6 uses K = 128, well under the cap.
pub const MAX_K: u32 = 256;
/// Hard cap on per-tile head-dim V (same reasoning).
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
    if p.k > MAX_K || p.v > MAX_V {
        return Err(MlxError::InvalidArgument(format!(
            "gated_delta_net_chunk: k ({}) and v ({}) must be <= MAX (k:{}, v:{})",
            p.k, p.v, MAX_K, MAX_V
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
    //   bh tile      : BV × K × 4 = 32 × 128 × 4 = 16 KB
    //   bw tile      : BT × K × 2 = 64 × 128 × 2 = 16 KB ❌ over budget
    // We stage bw and bk in registers/private and only put bh in shared.
    let shared_floats: u64 = (DEFAULT_BV * p.k) as u64;
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
