//! GPU-accelerated tiled-GQA broadcast: `[T, Hg, K]` → `[T, H, K]` F32.
//!
//! Replaces the hf2q-side CPU triple-loop tiled-replicate at
//! `gpu_delta_net.rs:893-940` (`q_expanded` / `k_expanded` fill,
//! ~497 ms / 10.4 ms-per-layer at PP4106 per the W-5b.17 audit).
//!
//! Mapping:
//!
//! ```text
//! dst[t, h, k] = src[t, h % Hg, k]
//! ```
//!
//! Where `Hg = n_k_heads`, `H = n_v_heads`, `K = head_dim`. The "tiled"
//! variant matches Qwen3.6 GGUF tensor layout (per
//! `project_qwen36_gqa_tiled_vs_block` and `gpu_delta_net.rs:834-866`),
//! and is the same convention as llama.cpp's `ggml_repeat_4d` graph op.
//!
//! ADR-005 W-5b.19 (2026-04-27): single-dispatch GPU broadcast eliminates
//! the chunk-wrapper's CPU memcpy bucket. Production caller:
//! `hf2q::inference::models::qwen35::gpu_delta_net::apply_gated_delta_net_chunk`
//! (chunk-prefill GQA pre-expansion).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL source for the tiled-repeat kernel (embedded at compile time).
pub static REPEAT_TILED_SHADER_SOURCE: &str =
    include_str!("../shaders/repeat_tiled.metal");

/// Register the repeat-tiled shader source with the given kernel registry.
///
/// Idempotent — the source is also auto-registered by `KernelRegistry::new`,
/// but this helper exists to mirror the convention used by other op modules
/// (`copy::register`, `flash_attn_prefill::register`, ...).
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("repeat_tiled_f32", REPEAT_TILED_SHADER_SOURCE);
}

/// MSL-compatible params struct. Must match `RepeatTiledParams` in
/// `repeat_tiled.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRepeatTiledParams {
    seq: u32,
    hg: u32,
    h: u32,
    k: u32,
}

/// Parameters for a tiled-GQA broadcast operation.
#[derive(Clone, Copy, Debug)]
pub struct RepeatTiledParams {
    /// Number of tokens (T).
    pub seq: u32,
    /// Source head count (Hg = n_k_heads).
    pub hg: u32,
    /// Destination head count (H = n_v_heads). Must satisfy `H % Hg == 0`.
    pub h: u32,
    /// Per-head element count (K = head_dim).
    pub k: u32,
}

/// Dispatch a tiled-GQA broadcast on the GPU.
///
/// Expands a `[seq, hg, k]` f32 input to a `[seq, h, k]` f32 output via
/// `dst[t, h, k] = src[t, h % hg, k]` in a single dispatch — no compute,
/// no host round-trip.
///
/// # Arguments
///
/// * `encoder`  - Command encoder to record the dispatch into.
/// * `registry` - Kernel registry (`repeat_tiled_f32` is auto-registered).
/// * `device`   - Metal device for pipeline compilation.
/// * `src`      - Input buffer, f32, contiguous, ≥ `seq*hg*k` elements.
/// * `dst`      - Output buffer, f32, contiguous, ≥ `seq*h*k` elements.
/// * `params`   - Shape parameters.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if any dimension is zero, if
/// `h % hg != 0`, or if either buffer is too small for the declared shapes.
pub fn dispatch_repeat_tiled_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    params: &RepeatTiledParams,
) -> Result<()> {
    if params.seq == 0 || params.hg == 0 || params.h == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "repeat_tiled_f32: seq, hg, h, k must all be > 0".into(),
        ));
    }
    if params.h % params.hg != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "repeat_tiled_f32: h ({}) must be a multiple of hg ({})",
            params.h, params.hg
        )));
    }

    // Buffer-size sanity checks (in bytes; f32 = 4 B).
    let src_elems = (params.seq as usize)
        .checked_mul(params.hg as usize)
        .and_then(|v| v.checked_mul(params.k as usize))
        .ok_or_else(|| {
            MlxError::InvalidArgument(
                "repeat_tiled_f32: seq*hg*k overflows usize".into(),
            )
        })?;
    let dst_elems = (params.seq as usize)
        .checked_mul(params.h as usize)
        .and_then(|v| v.checked_mul(params.k as usize))
        .ok_or_else(|| {
            MlxError::InvalidArgument(
                "repeat_tiled_f32: seq*h*k overflows usize".into(),
            )
        })?;

    let src_bytes = src_elems * 4;
    if src.byte_len() < src_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "repeat_tiled_f32: src buffer too small: need {} bytes, have {}",
            src_bytes,
            src.byte_len()
        )));
    }
    let dst_bytes = dst_elems * 4;
    if dst.byte_len() < dst_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "repeat_tiled_f32: dst buffer too small: need {} bytes, have {}",
            dst_bytes,
            dst.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("repeat_tiled_f32", device)?;

    let gpu_params = GpuRepeatTiledParams {
        seq: params.seq,
        hg: params.hg,
        h: params.h,
        k: params.k,
    };

    // Grid: (K, H, T) — one thread per output element. Threadgroup width
    // along K dimension (innermost / contiguous in dst write) up to 256.
    let grid = MTLSize::new(params.k as u64, params.h as u64, params.seq as u64);
    let tg_x = std::cmp::min(256u64, params.k as u64);
    let tg = MTLSize::new(tg_x, 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}
