//! Fused per-head RMSNorm + SiLU gate kernel for DeltaNet op 8.
//!
//! Computes:
//! `normed[t, vh, d] = attn_out[t, vh, d] * rsqrt(mean(attn_out[t, vh, :]^2) + eps) * weight[d]`
//! `output[t, vh, d] = normed[t, vh, d] * silu(z[t, vh, d])`
//!
//! Replaces the CPU bridge in `apply_ssm_norm_and_gate`: eliminates 2 GPU
//! downloads + CPU compute + 1 upload per delta-net layer.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static SSM_NORM_GATE_SHADER_SOURCE: &str = include_str!("../shaders/ssm_norm_gate.metal");

/// Register `ssm_norm_gate_f32` shader with the kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("ssm_norm_gate_f32", SSM_NORM_GATE_SHADER_SOURCE);
}

/// Dispatch `ssm_norm_gate_f32` into an external encoder.
///
/// Inputs:
/// - `attn_out`: `[seq * n_v_heads, d_v]` f32 — GDN output
/// - `weight`:   `[d_v]` f32 — ssm_norm learned weight
/// - `z`:        `[seq * n_v_heads, d_v]` f32 — z gate projection (same layout)
/// - `output`:   `[seq * n_v_heads, d_v]` f32 — allocated output buffer
/// - `params_buf`: `[eps: f32, d_v_f: f32]` — 8 bytes
///
/// `rows = seq * n_v_heads`, one threadgroup per row.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_ssm_norm_gate(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    attn_out: &MlxBuffer,
    weight: &MlxBuffer,
    z: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,   // seq * n_v_heads
    d_v: u32,
) -> Result<()> {
    if rows == 0 || d_v == 0 {
        return Err(MlxError::InvalidArgument(
            "ssm_norm_gate: rows and d_v must be > 0".into(),
        ));
    }
    let pipeline = registry.get_pipeline("ssm_norm_gate_f32", device)?;
    // One threadgroup per row; threadgroup_size = next_pow2(d_v) capped at 256.
    let tg_size = d_v.next_power_of_two().min(256) as u64;
    let shared_bytes = tg_size * 4; // sizeof(float) per thread in shared memory
    // dispatch_thread_groups: grid = number of threadgroups, tg = threads per threadgroup
    let grid = MTLSize::new(rows as u64, 1, 1);
    let tg = MTLSize::new(tg_size, 1, 1);
    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[
            (0, attn_out),
            (1, weight),
            (2, z),
            (3, output),
            (4, params_buf),
        ],
        &[(0, shared_bytes)],  // threadgroup_mem: (index=0, byte_len=tg_size*4)
        grid,
        tg,
    );
    Ok(())
}

/// Allocate params buffer `[eps: f32, d_v_f: f32]` for `ssm_norm_gate_f32`.
pub fn build_ssm_norm_gate_params(
    device: &crate::device::MlxDevice,
    eps: f32,
    d_v: u32,
) -> Result<MlxBuffer> {
    let mut buf = device
        .alloc_buffer(8, DType::F32, vec![2])
        .map_err(|e| MlxError::InvalidArgument(format!("ssm_norm_gate params alloc: {e}")))?;
    let s = buf
        .as_mut_slice::<f32>()
        .map_err(|e| MlxError::InvalidArgument(format!("ssm_norm_gate params write: {e}")))?;
    s[0] = eps;
    s[1] = d_v as f32;
    Ok(buf)
}
