//! ADR-034 task #93 (2026-05-21) — Fused gate_proj + up_proj + silu_mul Q8_0.
//!
//! Dispatch wrapper for `kernel_fused_gate_up_silu_q8_0_f32`.
//!
//! Replaces a 3-dispatch sequence:
//!   `dispatch_quantized_matmul_ggml(Q8_0, gate_w, x)` → `tmp_gate`
//!   `dispatch_quantized_matmul_ggml(Q8_0, up_w, x)`   → `tmp_up`
//!   `dispatch_silu_mul(tmp_gate, tmp_up)`              → `out`
//!
//! With a single Metal kernel dispatch. Input vector `x` is loaded ONCE per
//! ib iteration and reused for both projections, saving ~50% of input
//! memory bandwidth.
//!
//! Dispatch geometry (peer-style NR=2 NSG=4):
//!   - threadgroups   = (ceil(N / 2), M, 1)
//!   - threads_per_tg = (32, 4, 1) = 128 threads
//!   - threadgroup memory = 4 * 32 * sizeof(f32) = 512 bytes
//!     (NR0=2 output rows × 2 (gate + up) × NW=32 threads × f32)
//!
//! Math contract: byte-identical to the unfused 3-dispatch sequence when
//! both weight matrices are Q8_0 with identical layout (ne00, ne01, ne02).

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::encoder::{CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static FUSED_GATE_UP_SILU_Q8_0_SHADER_SOURCE: &str =
    include_str!("../shaders/fused_gate_up_silu_q8_0.metal");

/// Register the fused gate+up+silu Q8_0 shader.
///
/// Must be called before dispatching. The PSO is compiled lazily on first
/// `get_pipeline_with_constants` call.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "kernel_fused_gate_up_silu_q8_0_f32",
        FUSED_GATE_UP_SILU_Q8_0_SHADER_SOURCE,
    );
}

/// GPU-side params — matches `GgmlMatvecParams` in
/// `src/shaders/fused_gate_up_silu_q8_0.metal` (identical to the unfused
/// `GgmlMatvecParams` so the same struct layout works for both kernels).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedGateUpSiluQ8_0Params {
    ne00: i64, // K (hidden_size)
    ne01: i64, // N (intermediate_size, rows of gate_w / up_w)
    ne02: i64, // batch (weights) = 1
    ne10: i64, // K (input dim) = ne00
    ne12: i64, // batch (input) = 1
    ne0:  i64, // output stride = ne01
    ne1:  i64, // M (number of input rows; 1 for decode)
    r2:   u32, // ne12 / ne02 = 1
    r3:   u32, // 1 (always for non-batched)
}

/// Public params for [`dispatch_fused_gate_up_silu_q8_0`].
#[derive(Debug, Clone, Copy)]
pub struct FusedGateUpSiluQ8_0Args {
    /// Input rows (1 for decode; this kernel is designed for ne1==1).
    pub m: u32,
    /// Intermediate dimension = rows of gate_w / up_w.
    pub intermediate_size: u32,
    /// Hidden dimension = cols of gate_w / up_w before quantization.
    /// Must be divisible by 32 (QK8_0).
    pub hidden_size: u32,
}

/// Dispatch the fused gate+up+silu Q8_0 kernel.
///
/// All preconditions:
///   - `gate_w` and `up_w` are Q8_0 blocks, layout `[intermediate_size, hidden_size]`,
///     i.e. `intermediate_size * (hidden_size / 32) * 34` bytes each.
///   - `input` is F32, `hidden_size * m` elements.
///   - `output` is F32, `intermediate_size * m` elements (pre-allocated).
///   - `args.hidden_size % 32 == 0` (Q8_0 block size).
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] on shape mismatch or zero dims.
#[allow(clippy::too_many_arguments, non_snake_case)]
pub fn dispatch_fused_gate_up_silu_q8_0(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    gate_w: &MlxBuffer,
    up_w: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    args: FusedGateUpSiluQ8_0Args,
) -> Result<()> {
    const QK8_0: u32 = 32;
    const BLOCK_Q8_0_BYTES: u32 = 34; // sizeof(half) + QK8_0 = 2 + 32

    if args.m == 0 || args.intermediate_size == 0 || args.hidden_size == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_gate_up_silu_q8_0: m, intermediate_size, hidden_size must be > 0".into(),
        ));
    }
    if args.hidden_size % QK8_0 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gate_up_silu_q8_0: hidden_size {} must be divisible by QK8_0 ({})",
            args.hidden_size, QK8_0,
        )));
    }

    let blocks_per_row = args.hidden_size / QK8_0;
    let weight_bytes = (args.intermediate_size as usize)
        * (blocks_per_row as usize)
        * (BLOCK_Q8_0_BYTES as usize);
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
                "fused_gate_up_silu_q8_0: {name} buffer too small: need {expected} bytes, have {}",
                buf.byte_len()
            )));
        }
    }

    // PSO with FC slots 700/701/702 = 1 (matches the unfused Q8_0 NR2 kernel
    // dispatcher pattern; current usage is always single-batch so divisors
    // are 1 and the Metal compiler folds `im % 1 → 0` etc.).
    let pipeline = registry.get_pipeline_with_constants(
        "kernel_fused_gate_up_silu_q8_0_f32",
        device.metal_device(),
        &[],
        &[(700, 1), (701, 1), (702, 1)],
    )?;

    let gpu_params = FusedGateUpSiluQ8_0Params {
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

    // Same geometry as kernel_mul_mv_q8_0_f32_nr2: NR0=2 rows/TG, NSG=4
    // simdgroups × NW=32 threads = 128 threads/TG.
    let threadgroups = MTLSize::new(
        ((args.intermediate_size as u64) + 1) / 2,
        args.m as u64,
        1,
    );
    let threads_per_tg = MTLSize::new(32, 4, 1);

    // Threadgroup memory: NR0 (2) × 2 (gate + up) × NW (32) × f32 (4) = 256 bytes
    // for gate slots + 256 bytes for up slots = 512 bytes total.
    let smem_bytes: u64 = (2u64 * 2u64 * 32u64) * (std::mem::size_of::<f32>() as u64);

    encoder.encode_threadgroups_with_args_and_shared(
        &pipeline,
        &[
            (0, KernelArg::Buffer(gate_w)),
            (1, KernelArg::Buffer(up_w)),
            (2, KernelArg::Buffer(input)),
            (3, KernelArg::Buffer(output)),
            (4, KernelArg::Bytes(bytemuck::bytes_of(&gpu_params))),
        ],
        &[(0, smem_bytes)],
        threadgroups,
        threads_per_tg,
    );
    Ok(())
}
