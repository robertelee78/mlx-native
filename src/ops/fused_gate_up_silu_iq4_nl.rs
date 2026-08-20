//! ADR-034 task #93 cont. 26 (2026-05-21) — Fused IQ4_NL gate+up+silu_mul.
//!
//! IQ4_NL variant of `fused_gate_up_silu_q8_0`. Same Q4_0-style geometry
//! (8 rows/TG, 64 threads, no shmem) but uses the IQ4_NL codebook lookup
//! instead of biased nibble pairs.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static FUSED_GATE_UP_SILU_IQ4_NL_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_gate_up_silu_iq4_nl.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "kernel_fused_gate_up_silu_iq4_nl_f32",
        FUSED_GATE_UP_SILU_IQ4_NL_SHADER_SOURCE,
    );
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedGateUpSiluIq4NlParams {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne10: i64,
    ne12: i64,
    ne0: i64,
    ne1: i64,
    r2: u32,
    r3: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct FusedGateUpSiluIq4NlArgs {
    /// Input row count (1 for decode; up to ~8 for batched verify).
    pub m: u32,
    /// Intermediate dimension = rows of gate_w / up_w.
    pub intermediate_size: u32,
    /// Hidden dimension. Must be divisible by 32 (QK4_0).
    pub hidden_size: u32,
}

/// Dispatch the fused IQ4_NL gate+up+silu_mul kernel.
///
/// Preconditions:
///   - `gate_w` and `up_w` are IQ4_NL blocks, layout
///     `[intermediate_size, hidden_size / 32]`, 18 bytes per block
///     (sizeof(half) + 16 = 18 bytes per IQ4_NL block).
///   - `input` is F32, `hidden_size * m` elements.
///   - `output` is F32, `intermediate_size * m` elements.
///   - `hidden_size % 32 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_gate_up_silu_iq4_nl(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate_w: &MlxBuffer,
    up_w: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    args: FusedGateUpSiluIq4NlArgs,
) -> Result<()> {
    const QK4_0: u32 = 32;
    const BLOCK_IQ4_NL_BYTES: u32 = 18; // sizeof(half) + QK4_0/2 = 2 + 16

    if args.m == 0 || args.intermediate_size == 0 || args.hidden_size == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_gate_up_silu_iq4_nl: m, intermediate_size, hidden_size must be > 0".into(),
        ));
    }
    if args.hidden_size % QK4_0 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gate_up_silu_iq4_nl: hidden_size {} must be divisible by QK4_0 ({})",
            args.hidden_size, QK4_0,
        )));
    }

    let blocks_per_row = args.hidden_size / QK4_0;
    let weight_bytes = (args.intermediate_size as usize)
        .checked_mul(blocks_per_row as usize)
        .and_then(|value| value.checked_mul(BLOCK_IQ4_NL_BYTES as usize))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_iq4_nl: weight size overflow".into())
        })?;
    let input_bytes = (args.hidden_size as usize)
        .checked_mul(args.m as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_iq4_nl: input size overflow".into())
        })?;
    let output_bytes = (args.intermediate_size as usize)
        .checked_mul(args.m as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_iq4_nl: output size overflow".into())
        })?;

    for (name, buf, expected) in [
        ("gate_w", gate_w, weight_bytes),
        ("up_w", up_w, weight_bytes),
        ("input", input, input_bytes),
        ("output", output, output_bytes),
    ] {
        if buf.data_byte_len() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_gate_up_silu_iq4_nl: {name} too small: need {expected} bytes, have {}",
                buf.data_byte_len()
            )));
        }
    }

    let pipeline = registry.get_pipeline_with_constants(
        "kernel_fused_gate_up_silu_iq4_nl_f32",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let gpu_params = FusedGateUpSiluIq4NlParams {
        ne00: args.hidden_size as i64,
        ne01: args.intermediate_size as i64,
        ne02: 1,
        ne10: args.hidden_size as i64,
        ne12: 1,
        ne0: args.intermediate_size as i64,
        ne1: args.m as i64,
        r2: 1,
        r3: 1,
    };

    // Same geometry as kernel_mul_mv_iq4_nl_f32 / Q4_0 baseline:
    // align=8 (8 rows/TG), (8, 8, 1) threads = 64 = 2 SG × 32. No shmem.
    let threadgroups = MTLSize::new(((args.intermediate_size as u64) + 7) / 8, args.m as u64, 1);
    let threads_per_tg = MTLSize::new(8, 8, 1);

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
