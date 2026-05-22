//! ADR-034 task #94 (2026-05-21) — Fused dual Q4_0 projection dispatch.
//!
//! Wraps `kernel_fused_dual_proj_q4_0_f32`. Replaces 2 separate
//! `quantized_matmul_ggml` (Q4_0) dispatches that share the same input
//! vector with a single fused dispatch.
//!
//! Designed to fuse:
//!   - Q + gate projections (both `[hidden, q_total]`)
//!   - K + V projections (both `[hidden, kv_total]`)
//! in the Qwen FA layer Op 2 sequence.
//!
//! Both weight matrices MUST share shape (ne00, ne01, ne02).
//!
//! Math contract: byte-identical to two `kernel_mul_mv_q4_0_f32` dispatches.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static FUSED_DUAL_PROJ_Q4_0_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_dual_proj_q4_0.metal");

/// Register the fused dual Q4_0 shader.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "kernel_fused_dual_proj_q4_0_f32",
        FUSED_DUAL_PROJ_Q4_0_SHADER_SOURCE,
    );
}

/// GPU-side params — same layout as `GgmlMatvecParams` in
/// `quantized_matmul_ggml.metal` (single struct shared across the matvec
/// kernel family).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedDualProjQ4_0Params {
    ne00: i64, // K (hidden_size)
    ne01: i64, // N (output_size, rows of weight_a / weight_b)
    ne02: i64, // batch (weights) = 1
    ne10: i64, // K (input dim) = ne00
    ne12: i64, // batch (input) = 1
    ne0:  i64, // output stride = ne01
    ne1:  i64, // M (number of input rows)
    r2:   u32, // ne12 / ne02 = 1
    r3:   u32, // 1
}

/// Public args for [`dispatch_fused_dual_proj_q4_0`].
#[derive(Debug, Clone, Copy)]
pub struct FusedDualProjQ4_0Args {
    /// Input row count (1 for decode; up to ~8 for batched verify).
    pub m: u32,
    /// Output dimension (rows of weight_a / weight_b). Must match for both.
    pub output_size: u32,
    /// Input/hidden dimension. Must be divisible by 32 (QK4_0).
    pub hidden_size: u32,
}

/// Dispatch the fused dual Q4_0 mat-vec kernel.
///
/// Preconditions:
///   - `weight_a` and `weight_b` are Q4_0 blocks, layout
///     `[output_size, hidden_size / 32]`, i.e.
///     `output_size * (hidden_size / 32) * 18` bytes each
///     (sizeof(half) + 16 nibbles = 18 bytes per Q4_0 block).
///   - `input` is F32, `hidden_size * m` elements.
///   - `dst_a` and `dst_b` are F32, `output_size * m` elements each.
///   - `hidden_size % 32 == 0`.
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] on shape mismatch or zero dims.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn dispatch_fused_dual_proj_q4_0(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight_a: &MlxBuffer,
    weight_b: &MlxBuffer,
    input: &MlxBuffer,
    dst_a: &MlxBuffer,
    dst_b: &MlxBuffer,
    args: FusedDualProjQ4_0Args,
) -> Result<()> {
    const QK4_0: u32 = 32;
    const BLOCK_Q4_0_BYTES: u32 = 18; // sizeof(half) + QK4_0/2 = 2 + 16

    if args.m == 0 || args.output_size == 0 || args.hidden_size == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_dual_proj_q4_0: m, output_size, hidden_size must be > 0".into(),
        ));
    }
    if args.hidden_size % QK4_0 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_dual_proj_q4_0: hidden_size {} must be divisible by QK4_0 ({})",
            args.hidden_size, QK4_0,
        )));
    }

    let blocks_per_row = args.hidden_size / QK4_0;
    let weight_bytes = (args.output_size as usize)
        * (blocks_per_row as usize)
        * (BLOCK_Q4_0_BYTES as usize);
    let input_bytes = (args.hidden_size as usize) * (args.m as usize) * 4;
    let output_bytes = (args.output_size as usize) * (args.m as usize) * 4;

    for (name, buf, expected) in [
        ("weight_a", weight_a, weight_bytes),
        ("weight_b", weight_b, weight_bytes),
        ("input", input, input_bytes),
        ("dst_a", dst_a, output_bytes),
        ("dst_b", dst_b, output_bytes),
    ] {
        if buf.byte_len() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_dual_proj_q4_0: {name} too small: need {expected} bytes, have {}",
                buf.byte_len()
            )));
        }
    }

    // PSO with FC slots 700/701/702 = 1 (matches the Q4_0 baseline kernel
    // dispatcher pattern; single-batch usage so divisors are 1).
    let pipeline = registry.get_pipeline_with_constants(
        "kernel_fused_dual_proj_q4_0_f32",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let gpu_params = FusedDualProjQ4_0Params {
        ne00: args.hidden_size as i64,
        ne01: args.output_size as i64,
        ne02: 1,
        ne10: args.hidden_size as i64,
        ne12: 1,
        ne0:  args.output_size as i64,
        ne1:  args.m as i64,
        r2:   1,
        r3:   1,
    };

    // Geometry: same as kernel_mul_mv_q4_0_f32 baseline — align=8, 8 rows/TG,
    // (8, 8, 1) threads = 64 = 2 simdgroups × 32. No threadgroup shared
    // memory required.
    let threadgroups = MTLSize::new(
        ((args.output_size as u64) + 7) / 8,
        args.m as u64,
        1,
    );
    let threads_per_tg = MTLSize::new(8, 8, 1);

    encoder.encode_threadgroups_with_args(
        &pipeline,
        &[
            (0, KernelArg::Buffer(weight_a)),
            (1, KernelArg::Buffer(weight_b)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(dst_a)),
            (4, KernelArg::Buffer(dst_b)),
            (5, KernelArg::Bytes(bytemuck::bytes_of(&gpu_params))),
        ],
        threadgroups,
        threads_per_tg,
    );
    Ok(())
}
