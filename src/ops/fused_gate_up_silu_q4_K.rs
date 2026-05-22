//! ADR-034 task #93 cont. 24 (2026-05-21) — Fused gate+up+silu_mul Q4_K dispatch.
//!
//! Q4_K variant of `fused_gate_up_silu_q8_0`. Wraps
//! `kernel_fused_gate_up_silu_q4_K_f32`. Replaces a 3-dispatch sequence
//! (gate matvec + up matvec + silu_mul) with a single fused dispatch.
//!
//! Designed to fire on Q4_K_M Qwen 3.5/3.6 GGUFs (the most common
//! production quant).
//!
//! Math contract: byte-identical (within F32 FMA tolerance ≤ 1e-5) to
//! the equivalent 3-dispatch unfused sequence.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static FUSED_GATE_UP_SILU_Q4_K_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_gate_up_silu_q4_K.metal");

/// Register the fused Q4_K shader.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "kernel_fused_gate_up_silu_q4_K_f32",
        FUSED_GATE_UP_SILU_Q4_K_SHADER_SOURCE,
    );
}

/// GPU-side params — same layout as `GgmlMatvecParams` in
/// `quantized_matmul_ggml.metal`.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedGateUpSiluQ4_KParams {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne10: i64,
    ne12: i64,
    ne0:  i64,
    ne1:  i64,
    r2:   u32,
    r3:   u32,
}

/// Public args for [`dispatch_fused_gate_up_silu_q4_K`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub struct FusedGateUpSiluQ4_KArgs {
    /// Input row count (1 for decode; up to ~8 for batched verify).
    pub m: u32,
    /// Intermediate dimension = rows of gate_w / up_w.
    pub intermediate_size: u32,
    /// Hidden dimension. Must be divisible by 256 (QK_K).
    pub hidden_size: u32,
}

/// Dispatch the fused Q4_K gate+up+silu_mul kernel.
///
/// Preconditions:
///   - `gate_w` and `up_w` are Q4_K super-blocks, layout
///     `[intermediate_size, hidden_size / 256]`, i.e.
///     `intermediate_size * (hidden_size / 256) * 144` bytes each
///     (2*sizeof(half) + 12 + 128 = 144 bytes per Q4_K block).
///   - `input` is F32, `hidden_size * m` elements.
///   - `output` is F32, `intermediate_size * m` elements.
///   - `hidden_size % 256 == 0`.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn dispatch_fused_gate_up_silu_q4_K(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate_w: &MlxBuffer,
    up_w: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    args: FusedGateUpSiluQ4_KArgs,
) -> Result<()> {
    const QK_K: u32 = 256;
    const BLOCK_Q4_K_BYTES: u32 = 144; // 2*sizeof(half) + 12 + 128

    if args.m == 0 || args.intermediate_size == 0 || args.hidden_size == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_gate_up_silu_q4_K: m, intermediate_size, hidden_size must be > 0".into(),
        ));
    }
    if args.hidden_size % QK_K != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gate_up_silu_q4_K: hidden_size {} must be divisible by QK_K ({})",
            args.hidden_size, QK_K,
        )));
    }

    let super_blocks_per_row = args.hidden_size / QK_K;
    let weight_bytes = (args.intermediate_size as usize)
        * (super_blocks_per_row as usize)
        * (BLOCK_Q4_K_BYTES as usize);
    let input_bytes = (args.hidden_size as usize) * (args.m as usize) * 4;
    let output_bytes = (args.intermediate_size as usize) * (args.m as usize) * 4;

    for (name, buf, expected) in [
        ("gate_w", gate_w, weight_bytes),
        ("up_w", up_w, weight_bytes),
        ("input", input, input_bytes),
        ("output", output, output_bytes),
    ] {
        if buf.byte_len() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_gate_up_silu_q4_K: {name} too small: need {expected} bytes, have {}",
                buf.byte_len()
            )));
        }
    }

    let pipeline = registry.get_pipeline_with_constants(
        "kernel_fused_gate_up_silu_q4_K_f32",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let gpu_params = FusedGateUpSiluQ4_KParams {
        ne00: args.hidden_size as i64,
        ne01: args.intermediate_size as i64,
        ne02: 1,
        ne10: args.hidden_size as i64,
        ne12: 1,
        ne0:  args.intermediate_size as i64,
        ne1:  args.m as i64,
        r2:   1,
        r3:   1,
    };

    // Same geometry as kernel_mul_mv_q4_K_f32: 2 simdgroups × 32 threads,
    // 2 rows/TG. NO threadgroup shared memory required (each SG writes
    // 1 distinct row).
    let threadgroups = MTLSize::new(
        ((args.intermediate_size as u64) + 1) / 2,
        args.m as u64,
        1,
    );
    let threads_per_tg = MTLSize::new(32, 2, 1);

    encoder.encode_threadgroups_with_args(
        &pipeline,
        &[
            (0, KernelArg::Buffer(gate_w)),
            (1, KernelArg::Buffer(up_w)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(bytemuck::bytes_of(&gpu_params))),
        ],
        threadgroups,
        threads_per_tg,
    );
    Ok(())
}
