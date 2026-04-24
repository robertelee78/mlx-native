//! TQ KV dequantize kernel dispatch — iter-20 Leg F ablation.
//!
//! Reads nibble-packed TurboQuant KV cache at one position and writes a
//! dense F32 buffer of shape `[num_kv_heads, head_dim]`.
//!
//! This isolates the SDPA kernel math from TQ representation noise:
//!   1. K/V are still encoded via `dispatch_hadamard_quantize_kv` (TQ path).
//!   2. This kernel decodes them back to F32 in the FWHT-rotated domain.
//!   3. Caller dispatches `flash_attn_vec` (dense) with those F32 buffers.
//!
//! Decision interpretation:
//!   Leg_F ≥ 3094 → TQ SDPA kernel has a math bug; dense SDPA on TQ K/V recovers parity.
//!   Leg_F ≈ 127  → Representation floor: 4-bit encode/decode round-trip is the source
//!                  of the byte-prefix gap, not a kernel bug.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{encode_threadgroups_with_args, KernelArg};

/// MSL source for the TQ KV dequantize kernel (embedded at compile time).
pub static TQ_DEQUANTIZE_KV_SHADER_SOURCE: &str =
    include_str!("../shaders/tq_dequantize_kv.metal");

/// Register the `tq_dequantize_kv` shader with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("tq_dequantize_kv", TQ_DEQUANTIZE_KV_SHADER_SOURCE);
}

/// GPU-side parameter struct. Must match `TqDequantizeKvParams` in the MSL exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TqDequantizeKvParamsGpu {
    head_dim: u32,
    num_kv_heads: u32,
    read_pos: u32,
    cache_capacity: u32,
    norms_per_pos: u32,
    scale_factor_d512: f32,
}

/// Dispatch the TQ KV dequantize kernel.
///
/// Reads the TQ-encoded K or V at `read_pos` from `packed` and `norms` and
/// writes the dequantized F32 values to `dst` of shape `[num_kv_heads, head_dim]`.
///
/// The output is in the FWHT-rotated domain — the same domain the
/// `hadamard_quantize_kv` kernel encodes into. The caller must apply an
/// inverse FWHT if the original (non-rotated) K/V is needed; for the Leg F
/// ablation this is NOT required because Q was also pre-rotated by FWHT before
/// `flash_attn_vec`, so the rotated-domain dot products are correct.
///
/// # Arguments
///
/// * `encoder`           — Command encoder.
/// * `registry`          — Kernel registry.
/// * `device`            — Metal device.
/// * `packed`            — `[num_kv_heads, cache_capacity, head_dim/2]` u8.
/// * `norms`             — `[num_kv_heads, cache_capacity, norms_per_pos]` f32.
/// * `dst`               — `[num_kv_heads, head_dim]` f32 output buffer (pre-allocated).
/// * `num_kv_heads`      — Number of KV heads.
/// * `head_dim`          — Head dimension (256 or 512).
/// * `cache_capacity`    — KV cache capacity.
/// * `read_pos`          — Logical cache position to read (already wrapped for ring buffers).
/// * `scale_factor_d512` — D=512 per-block norm scale divisor (1.0 = bare, iter-16 control).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_tq_dequantize_kv(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    packed: &MlxBuffer,
    norms: &MlxBuffer,
    dst: &MlxBuffer,
    num_kv_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    read_pos: u32,
    scale_factor_d512: f32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    if !head_dim.is_power_of_two() {
        return Err(MlxError::InvalidArgument(format!(
            "tq_dequantize_kv: head_dim must be a power of two, got {}",
            head_dim
        )));
    }

    // Validate dst buffer has room for [num_kv_heads, head_dim] f32.
    let required_dst = (num_kv_heads as u64) * (head_dim as u64);
    if (dst.element_count() as u64) < required_dst {
        return Err(MlxError::InvalidArgument(format!(
            "tq_dequantize_kv: dst has {} elements, need {}",
            dst.element_count(),
            required_dst
        )));
    }

    let norms_per_pos = (head_dim / 256).max(1);

    let params = TqDequantizeKvParamsGpu {
        head_dim,
        num_kv_heads,
        read_pos,
        cache_capacity,
        norms_per_pos,
        scale_factor_d512,
    };
    let params_bytes = bytemuck::bytes_of(&params);

    let pipeline = registry.get_pipeline("tq_dequantize_kv", device)?;

    // One threadgroup per KV head; head_dim threads per threadgroup.
    let threadgroups = MTLSize { width: num_kv_heads as u64, height: 1, depth: 1 };
    let threadgroup_size = MTLSize {
        width: head_dim.min(1024) as u64,
        height: 1,
        depth: 1,
    };

    encode_threadgroups_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(packed)),
            (1, KernelArg::Buffer(norms)),
            (2, KernelArg::Buffer(dst)),
            (3, KernelArg::Bytes(params_bytes)),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}
