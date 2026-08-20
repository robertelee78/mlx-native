//! ADR-034 task #93 cont. 28 (2026-05-21) — Fused Q6_K gate+up+silu_mul dispatch.
//!
//! Q6_K uses 2 SG × 32 threads with 1 row per simdgroup (2 rows/TG).
//! Different geometry from Q4_K/Q5_K (which had 2 rows/SG). 210 bytes per
//! 256-element super-block.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::ggml_capability::{GgmlRoutingPolicy, GgmlWorkloadClass};
use crate::ggml_dispatch_trace::{trace_dense_gate_up_silu_operation, GgmlResolvedDispatchTrace};
use crate::kernel_registry::KernelRegistry;
use crate::ops::quantized_matmul_ggml::GgmlType;

pub static FUSED_GATE_UP_SILU_Q6_K_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_gate_up_silu_q6_K.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "kernel_fused_gate_up_silu_q6_K_f32",
        FUSED_GATE_UP_SILU_Q6_K_SHADER_SOURCE,
    );
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedGateUpSiluQ6_KParams {
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

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub struct FusedGateUpSiluQ6_KArgs {
    pub m: u32,
    pub intermediate_size: u32,
    pub hidden_size: u32,
}

/// Dispatch the fused Q6_K gate+up+silu_mul kernel.
///
/// Preconditions:
///   - `gate_w` and `up_w` are Q6_K super-blocks, layout
///     `[intermediate_size, hidden_size / 256]`, 210 bytes per block
///     (`ql[128] + qh[64] + scales[16] + d`).
///   - `input` is F32, `hidden_size * m` elements.
///   - `output` is F32, `intermediate_size * m` elements.
///   - `hidden_size % 256 == 0`.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn dispatch_fused_gate_up_silu_q6_K(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate_w: &MlxBuffer,
    up_w: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    args: FusedGateUpSiluQ6_KArgs,
) -> Result<()> {
    const QK_K: u32 = 256;
    const BLOCK_Q6_K_BYTES: u32 = 210; // 128 + 64 + 16 + 2

    if args.m == 0 || args.intermediate_size == 0 || args.hidden_size == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_gate_up_silu_q6_K: m, intermediate_size, hidden_size must be > 0".into(),
        ));
    }
    if args.hidden_size % QK_K != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gate_up_silu_q6_K: hidden_size {} must be divisible by QK_K ({})",
            args.hidden_size, QK_K,
        )));
    }

    let super_blocks_per_row = args.hidden_size / QK_K;
    let weight_bytes = (args.intermediate_size as usize)
        .checked_mul(super_blocks_per_row as usize)
        .and_then(|value| value.checked_mul(BLOCK_Q6_K_BYTES as usize))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_q6_K: weight size overflow".into())
        })?;
    let input_bytes = (args.hidden_size as usize)
        .checked_mul(args.m as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_q6_K: input size overflow".into())
        })?;
    let output_bytes = (args.intermediate_size as usize)
        .checked_mul(args.m as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_q6_K: output size overflow".into())
        })?;

    for (name, buf, expected) in [
        ("gate_w", gate_w, weight_bytes),
        ("up_w", up_w, weight_bytes),
        ("input", input, input_bytes),
        ("output", output, output_bytes),
    ] {
        if buf.data_byte_len() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_gate_up_silu_q6_K: {name} too small: need {expected} bytes, have {}",
                buf.data_byte_len()
            )));
        }
    }

    let pipeline = registry.get_pipeline_with_constants(
        "kernel_fused_gate_up_silu_q6_K_f32",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let gpu_params = FusedGateUpSiluQ6_KParams {
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

    // Geometry: matches kernel_mul_mv_q6_K_f32. 2 SG × 32 threads = 64
    // threads/TG, 1 row/SG → 2 rows/TG. (Different from Q4_K/Q5_K's
    // 2 rows/SG → 4 rows/TG geometry.)
    let threadgroups = MTLSize::new(((args.intermediate_size as u64) + 1) / 2, args.m as u64, 1);
    let threads_per_tg = MTLSize::new(2, 32, 1);

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

#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn dispatch_fused_gate_up_silu_q6_K_with_trace(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate_w: &MlxBuffer,
    up_w: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    args: FusedGateUpSiluQ6_KArgs,
    routing: &GgmlRoutingPolicy,
    workload: GgmlWorkloadClass,
) -> Result<GgmlResolvedDispatchTrace> {
    trace_dense_gate_up_silu_operation(
        encoder,
        registry,
        device,
        GgmlType::Q6_K,
        args.m,
        args.intermediate_size,
        args.hidden_size,
        routing,
        workload,
        |encoder, registry| {
            dispatch_fused_gate_up_silu_q6_K(
                encoder, registry, device, gate_w, up_w, input, output, args,
            )
        },
    )
}
