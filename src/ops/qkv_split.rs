//! GPU-accelerated split of a fused QKV tensor into separate Q/K/V outputs.
//!
//! Input layout (per token, contiguous f32):
//!
//! ```text
//! qkv[t, :] = [ Q (q_sp) | K (k_sp) | V (v_sp) ]   (length = qkv_ch)
//! ```
//!
//! Where `q_sp = n_k_heads * d_k`, `k_sp = n_k_heads * d_k`, and
//! `v_sp = n_v_heads * d_v`. The kernel writes each input element to exactly
//! one of `{q, k, v}` in a single dispatch — replacing the prior CPU
//! download → triple-loop split → 3× upload round-trip used by the qwen35
//! Gated DeltaNet prefill path.
//!
//! ADR-005 W-5b.18 (2026-04-27): targets the 838 ms / 17.5 ms-per-layer
//! `layer.qkv_deinterleave` bucket in `hf2q::gpu_delta_net`.
//!
//! Production caller: `hf2q::inference::models::qwen35::gpu_delta_net::
//! apply_proj` (prefill seq>1 branch).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL source for the QKV-split kernel (embedded at compile time).
pub static QKV_SPLIT_SHADER_SOURCE: &str = include_str!("../shaders/qkv_split.metal");

/// Register the QKV-split shader source with the given kernel registry.
///
/// Idempotent — the source is also auto-registered by `KernelRegistry::new`,
/// but this helper exists to mirror the convention used by other op modules
/// (`copy::register`, `flash_attn_prefill::register`, ...).
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("qkv_split_f32", QKV_SPLIT_SHADER_SOURCE);
}

/// MSL-compatible params struct for the QKV split kernel.
///
/// Must match `QkvSplitParams` in `qkv_split.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuQkvSplitParams {
    seq: u32,
    q_sp: u32,
    k_sp: u32,
    v_sp: u32,
    qkv_ch: u32,
}

/// Parameters for a fused-QKV split operation.
#[derive(Clone, Copy, Debug)]
pub struct QkvSplitParams {
    /// Number of tokens in the sequence dimension.
    pub seq: u32,
    /// Q span per token, in f32 elements (== `n_k_heads * d_k`).
    pub q_sp: u32,
    /// K span per token, in f32 elements (== `n_k_heads * d_k`).
    pub k_sp: u32,
    /// V span per token, in f32 elements (== `n_v_heads * d_v`).
    pub v_sp: u32,
}

/// Dispatch a fused-QKV split on the GPU.
///
/// Splits a `[seq, q_sp + k_sp + v_sp]` f32 input into three contiguous
/// outputs — `q [seq, q_sp]`, `k [seq, k_sp]`, `v [seq, v_sp]` — in a
/// single dispatch, no compute, no host round-trip.
///
/// # Arguments
///
/// * `encoder`  - Command encoder to record the dispatch into.
/// * `registry` - Kernel registry (`qkv_split_f32` is auto-registered).
/// * `device`   - Metal device for pipeline compilation.
/// * `qkv`      - Input fused-QKV buffer, f32, contiguous.
/// * `q`        - Output Q buffer, f32, contiguous.
/// * `k`        - Output K buffer, f32, contiguous.
/// * `v`        - Output V buffer, f32, contiguous.
/// * `params`   - Shape parameters.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if any dimension is zero or any
/// buffer is too small for the declared shapes.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qkv_split_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    qkv: &MlxBuffer,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    params: &QkvSplitParams,
) -> Result<()> {
    if params.seq == 0 || params.q_sp == 0 || params.k_sp == 0 || params.v_sp == 0 {
        return Err(MlxError::InvalidArgument(
            "qkv_split_f32: seq, q_sp, k_sp, v_sp must all be > 0".into(),
        ));
    }

    let qkv_ch = params
        .q_sp
        .checked_add(params.k_sp)
        .and_then(|qk| qk.checked_add(params.v_sp))
        .ok_or_else(|| {
            MlxError::InvalidArgument(
                "qkv_split_f32: q_sp + k_sp + v_sp overflows u32".into(),
            )
        })?;

    // Buffer-size sanity checks (all in bytes; f32 = 4 B).
    let in_bytes = (params.seq as usize) * (qkv_ch as usize) * 4;
    if qkv.byte_len() < in_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "qkv_split_f32: qkv buffer too small: need {} bytes, have {}",
            in_bytes,
            qkv.byte_len()
        )));
    }
    let q_bytes = (params.seq as usize) * (params.q_sp as usize) * 4;
    if q.byte_len() < q_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "qkv_split_f32: q buffer too small: need {} bytes, have {}",
            q_bytes,
            q.byte_len()
        )));
    }
    let k_bytes = (params.seq as usize) * (params.k_sp as usize) * 4;
    if k.byte_len() < k_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "qkv_split_f32: k buffer too small: need {} bytes, have {}",
            k_bytes,
            k.byte_len()
        )));
    }
    let v_bytes = (params.seq as usize) * (params.v_sp as usize) * 4;
    if v.byte_len() < v_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "qkv_split_f32: v buffer too small: need {} bytes, have {}",
            v_bytes,
            v.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("qkv_split_f32", device)?;

    let gpu_params = GpuQkvSplitParams {
        seq: params.seq,
        q_sp: params.q_sp,
        k_sp: params.k_sp,
        v_sp: params.v_sp,
        qkv_ch,
    };

    let grid = MTLSize::new(qkv_ch as u64, params.seq as u64, 1);
    let tg_x = std::cmp::min(256u64, qkv_ch as u64);
    let tg = MTLSize::new(tg_x, 1, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(qkv)),
            (1, KernelArg::Buffer(q)),
            (2, KernelArg::Buffer(k)),
            (3, KernelArg::Buffer(v)),
            (4, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}
