//! ADR-034 task #93 cont. 27 (2026-05-21) — Fused Q5_K gate+up+silu_mul dispatch.
//!
//! Q5_K variant of `fused_gate_up_silu_q4_K`. Same super-block geometry
//! (2 SG × 32 threads/SG, 2 rows/TG) but the kernel adds the qh high-bit
//! accumulator (matches `kernel_mul_mv_q5_K_f32` canonical math).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::ggml_capability::{GgmlRoutingPolicy, GgmlWorkloadClass};
use crate::ggml_dispatch_trace::{trace_dense_gate_up_silu_operation, GgmlResolvedDispatchTrace};
use crate::kernel_registry::KernelRegistry;
use crate::ops::quantized_matmul_ggml::GgmlType;

pub static FUSED_GATE_UP_SILU_Q5_K_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_gate_up_silu_q5_K.metal");

pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "kernel_fused_gate_up_silu_q5_K_f32",
        FUSED_GATE_UP_SILU_Q5_K_SHADER_SOURCE,
    );
}

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedGateUpSiluQ5_KParams {
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

/// Public args for [`dispatch_fused_gate_up_silu_q5_K`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub struct FusedGateUpSiluQ5_KArgs {
    pub m: u32,
    pub intermediate_size: u32,
    pub hidden_size: u32,
}

/// Dispatch the fused Q5_K gate+up+silu_mul kernel.
///
/// Preconditions:
///   - `gate_w` and `up_w` are Q5_K super-blocks, layout
///     `[intermediate_size, hidden_size / 256]`, i.e.
///     `intermediate_size * (hidden_size / 256) * 176` bytes each
///     (2*sizeof(half) + 12 + 32 + 128 = 176 bytes per Q5_K block).
///   - `input` is F32, `hidden_size * m` elements.
///   - `output` is F32, `intermediate_size * m` elements.
///   - `hidden_size % 256 == 0`.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn dispatch_fused_gate_up_silu_q5_K(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate_w: &MlxBuffer,
    up_w: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    args: FusedGateUpSiluQ5_KArgs,
) -> Result<()> {
    const QK_K: u32 = 256;
    const BLOCK_Q5_K_BYTES: u32 = 176; // 2*sizeof(half) + 12 + 32 + 128

    if args.m == 0 || args.intermediate_size == 0 || args.hidden_size == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_gate_up_silu_q5_K: m, intermediate_size, hidden_size must be > 0".into(),
        ));
    }
    if args.hidden_size % QK_K != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gate_up_silu_q5_K: hidden_size {} must be divisible by QK_K ({})",
            args.hidden_size, QK_K,
        )));
    }

    let super_blocks_per_row = args.hidden_size / QK_K;
    let weight_bytes = (args.intermediate_size as usize)
        .checked_mul(super_blocks_per_row as usize)
        .and_then(|value| value.checked_mul(BLOCK_Q5_K_BYTES as usize))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_q5_K: weight size overflow".into())
        })?;
    let input_bytes = (args.hidden_size as usize)
        .checked_mul(args.m as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_q5_K: input size overflow".into())
        })?;
    let output_bytes = (args.intermediate_size as usize)
        .checked_mul(args.m as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            MlxError::InvalidArgument("fused_gate_up_silu_q5_K: output size overflow".into())
        })?;

    for (name, buf, expected) in [
        ("gate_w", gate_w, weight_bytes),
        ("up_w", up_w, weight_bytes),
        ("input", input, input_bytes),
        ("output", output, output_bytes),
    ] {
        if buf.data_byte_len() < expected {
            return Err(MlxError::InvalidArgument(format!(
                "fused_gate_up_silu_q5_K: {name} too small: need {expected} bytes, have {}",
                buf.data_byte_len()
            )));
        }
    }

    let pipeline = registry.get_pipeline_with_constants(
        "kernel_fused_gate_up_silu_q5_K_f32",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let gpu_params = FusedGateUpSiluQ5_KParams {
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

    let threadgroups = MTLSize::new(((args.intermediate_size as u64) + 1) / 2, args.m as u64, 1);
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

#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn dispatch_fused_gate_up_silu_q5_K_with_trace(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate_w: &MlxBuffer,
    up_w: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    args: FusedGateUpSiluQ5_KArgs,
    routing: &GgmlRoutingPolicy,
    workload: GgmlWorkloadClass,
) -> Result<GgmlResolvedDispatchTrace> {
    trace_dense_gate_up_silu_operation(
        encoder,
        registry,
        device,
        GgmlType::Q5_K,
        args.m,
        args.intermediate_size,
        args.hidden_size,
        routing,
        workload,
        |encoder, registry| {
            dispatch_fused_gate_up_silu_q5_K(
                encoder, registry, device, gate_w, up_w, input, output, args,
            )
        },
    )
}
