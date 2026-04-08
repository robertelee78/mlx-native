//! GPU-accelerated MoE expert dispatch (Stage 1: loop over selected experts).
//!
//! For each of the K selected experts, runs:
//!   gate_out   = gate_proj_e(x)       [input_dim -> intermediate_dim]
//!   up_out     = up_proj_e(x)         [input_dim -> intermediate_dim]
//!   hidden     = GELU(gate_out) * up_out
//!   expert_out = down_proj_e(hidden)  [intermediate_dim -> input_dim]
//!   result    += routing_weight_e * expert_out
//!
//! Stage 1 uses individual kernel dispatches per expert and per projection.
//! The projections use float matmul (caller dequantizes or provides float
//! weights).  Stage 2 optimization (Epic 6) would fuse these.
//!
//! This module provides the high-level `moe_dispatch` function that
//! orchestrates the per-expert loop, using the fused_gelu_mul and
//! moe_accumulate shaders from moe_dispatch.metal.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// Parameters for MoE dispatch.
pub struct MoeDispatchParams {
    /// Input/output dimension (e.g. 2816 for Gemma 4 MoE layers).
    pub input_dim: usize,
    /// Intermediate FFN dimension per expert (e.g. 704 for Gemma 4).
    pub intermediate_dim: usize,
    /// Number of selected experts (top_k, e.g. 8).
    pub n_selected: usize,
}

/// A single expert's weight matrices (float32, pre-dequantized or float).
///
/// Each expert has three projection matrices:
/// * `gate_proj`: `[input_dim, intermediate_dim]` row-major
/// * `up_proj`:   `[input_dim, intermediate_dim]` row-major
/// * `down_proj`: `[intermediate_dim, input_dim]` row-major
pub struct ExpertWeights<'a> {
    pub gate_proj: &'a MlxBuffer,
    pub up_proj: &'a MlxBuffer,
    pub down_proj: &'a MlxBuffer,
}

/// MSL-compatible struct for fused_gelu_mul kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFusedGeluMulParams {
    n_elements: u32,
}

/// MSL-compatible struct for moe_accumulate kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMoeAccumParams {
    n_elements: u32,
    routing_weight: f32,
}

/// MSL-compatible struct for zero_buffer kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuZeroParams {
    n_elements: u32,
}

/// MSL-compatible struct for a simple matmul params.
/// This is used with the naive_matmul shader for expert projections.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMatmulParams {
    m: u32,  // rows of output (1 for single-token)
    k: u32,  // inner dimension
    n: u32,  // cols of output
}

/// Encode MoE dispatch: loop over selected experts, run FFN, accumulate.
///
/// This is the Stage 1 implementation that loops over each selected expert
/// and dispatches individual compute passes for each projection.
///
/// # Buffer expectations
///
/// * `input` — f32, `[input_dim]` (single token hidden state)
/// * `expert_weights` — slice of `n_selected` expert weight structs, each
///   containing gate_proj, up_proj, down_proj as f32 buffers
/// * `routing_weights` — f32, `[n_selected]` (softmax routing weights from moe_gate)
/// * `output` — f32, `[input_dim]` (output, will be zero-initialized)
/// * `scratch_gate` — f32, `[intermediate_dim]` (scratch buffer for gate_proj output)
/// * `scratch_up` — f32, `[intermediate_dim]` (scratch buffer for up_proj output)
/// * `scratch_hidden` — f32, `[intermediate_dim]` (scratch buffer for GELU*up output)
/// * `scratch_expert` — f32, `[input_dim]` (scratch buffer for down_proj output)
///
/// # Design Notes
///
/// The caller provides scratch buffers to avoid allocating inside the
/// encoding loop.  These can come from a `MlxBufferPool`.
///
/// For the matrix projections, we use a naive matmul kernel (single-token,
/// M=1, so it's really a matvec).  The quantized_matmul from Story 1.2
/// would be used when weights remain quantized.  Stage 1 assumes float
/// weights for simplicity.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are invalid.
#[allow(clippy::too_many_arguments)]
pub fn moe_dispatch(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    expert_weights: &[ExpertWeights<'_>],
    routing_weights: &[f32],
    output: &MlxBuffer,
    scratch_gate: &MlxBuffer,
    scratch_up: &MlxBuffer,
    scratch_hidden: &MlxBuffer,
    scratch_expert: &MlxBuffer,
    params: &MoeDispatchParams,
) -> Result<()> {
    // --- Validation ---
    if params.input_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_dispatch: input_dim must be > 0".into(),
        ));
    }
    if params.intermediate_dim == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_dispatch: intermediate_dim must be > 0".into(),
        ));
    }
    if params.n_selected == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_dispatch: n_selected must be > 0".into(),
        ));
    }
    if expert_weights.len() != params.n_selected {
        return Err(MlxError::InvalidArgument(format!(
            "moe_dispatch: expert_weights length ({}) must match n_selected ({})",
            expert_weights.len(),
            params.n_selected
        )));
    }
    if routing_weights.len() != params.n_selected {
        return Err(MlxError::InvalidArgument(format!(
            "moe_dispatch: routing_weights length ({}) must match n_selected ({})",
            routing_weights.len(),
            params.n_selected
        )));
    }

    // Validate buffer sizes
    let input_bytes = params.input_dim * std::mem::size_of::<f32>();
    if input.byte_len() < input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_dispatch: input buffer too small: need {} bytes, have {}",
            input_bytes,
            input.byte_len()
        )));
    }
    if output.byte_len() < input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "moe_dispatch: output buffer too small: need {} bytes, have {}",
            input_bytes,
            output.byte_len()
        )));
    }

    let intermediate_bytes = params.intermediate_dim * std::mem::size_of::<f32>();
    if scratch_gate.byte_len() < intermediate_bytes {
        return Err(MlxError::InvalidArgument(
            "moe_dispatch: scratch_gate buffer too small".into(),
        ));
    }
    if scratch_up.byte_len() < intermediate_bytes {
        return Err(MlxError::InvalidArgument(
            "moe_dispatch: scratch_up buffer too small".into(),
        ));
    }
    if scratch_hidden.byte_len() < intermediate_bytes {
        return Err(MlxError::InvalidArgument(
            "moe_dispatch: scratch_hidden buffer too small".into(),
        ));
    }
    if scratch_expert.byte_len() < input_bytes {
        return Err(MlxError::InvalidArgument(
            "moe_dispatch: scratch_expert buffer too small".into(),
        ));
    }

    // Validate expert weight sizes
    let gate_up_bytes = params.input_dim * params.intermediate_dim * std::mem::size_of::<f32>();
    let down_bytes = params.intermediate_dim * params.input_dim * std::mem::size_of::<f32>();

    for (i, ew) in expert_weights.iter().enumerate() {
        if ew.gate_proj.byte_len() < gate_up_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "moe_dispatch: expert {} gate_proj too small: need {} bytes, have {}",
                i, gate_up_bytes, ew.gate_proj.byte_len()
            )));
        }
        if ew.up_proj.byte_len() < gate_up_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "moe_dispatch: expert {} up_proj too small: need {} bytes, have {}",
                i, gate_up_bytes, ew.up_proj.byte_len()
            )));
        }
        if ew.down_proj.byte_len() < down_bytes {
            return Err(MlxError::InvalidArgument(format!(
                "moe_dispatch: expert {} down_proj too small: need {} bytes, have {}",
                i, down_bytes, ew.down_proj.byte_len()
            )));
        }
    }

    // --- Pre-warm pipeline cache to avoid compilation during hot loop ---
    // Each get_pipeline call borrows &mut registry; we must not hold multiple
    // returned references simultaneously.  Pre-warming ensures subsequent
    // get_pipeline calls are cache hits, and we retrieve each one just before
    // use (via helper closures or repeated single calls).
    {
        registry.get_pipeline("naive_matvec_f32", device)?;
        registry.get_pipeline("fused_gelu_mul", device)?;
        registry.get_pipeline("moe_accumulate", device)?;
        registry.get_pipeline("zero_buffer", device)?;
    }
    // SAFETY: After pre-warming, the pipelines are in the cache and the HashMap
    // entries will not be moved or removed for the lifetime of `registry`.
    // We convert to raw pointers to avoid the borrow checker's complaint about
    // multiple &mut borrows while holding references into the cache.
    let matvec_pipeline: *const metal::ComputePipelineState = {
        let p = registry.get_pipeline("naive_matvec_f32", device)?;
        p as *const _
    };
    let gelu_mul_pipeline: *const metal::ComputePipelineState = {
        let p = registry.get_pipeline("fused_gelu_mul", device)?;
        p as *const _
    };
    let accum_pipeline: *const metal::ComputePipelineState = {
        let p = registry.get_pipeline("moe_accumulate", device)?;
        p as *const _
    };
    let zero_pipeline: *const metal::ComputePipelineState = {
        let p = registry.get_pipeline("zero_buffer", device)?;
        p as *const _
    };
    // Re-borrow from the raw pointers.  This is safe because:
    //   1. The HashMap entries are stable (no insertions/removals below).
    //   2. The registry outlives these references.
    //   3. We only read through these pointers.
    let matvec_pipeline = unsafe { &*matvec_pipeline };
    let gelu_mul_pipeline = unsafe { &*gelu_mul_pipeline };
    let accum_pipeline = unsafe { &*accum_pipeline };
    let zero_pipeline = unsafe { &*zero_pipeline };

    // --- Zero-initialize output ---
    let zero_params = GpuZeroParams {
        n_elements: params.input_dim as u32,
    };
    encode_with_args(
        encoder,
        zero_pipeline,
        &[
            (0, KernelArg::Buffer(output)),
            (1, KernelArg::Bytes(as_bytes(&zero_params))),
        ],
        MTLSize::new(params.input_dim as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, params.input_dim as u64), 1, 1),
    );

    // --- Loop over selected experts ---
    for (i, ew) in expert_weights.iter().enumerate() {
        let w = routing_weights[i];

        // Skip experts with near-zero routing weight
        if w.abs() < 1e-10 {
            continue;
        }

        // gate_out = gate_proj @ input  (matvec: [intermediate_dim, input_dim] @ [input_dim] -> [intermediate_dim])
        let gate_params = GpuMatmulParams {
            m: 1,
            k: params.input_dim as u32,
            n: params.intermediate_dim as u32,
        };
        encode_with_args(
            encoder,
            matvec_pipeline,
            &[
                (0, KernelArg::Buffer(ew.gate_proj)),
                (1, KernelArg::Buffer(input)),
                (2, KernelArg::Buffer(scratch_gate)),
                (3, KernelArg::Bytes(as_bytes(&gate_params))),
            ],
            MTLSize::new(params.intermediate_dim as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, params.intermediate_dim as u64), 1, 1),
        );

        // up_out = up_proj @ input  (same shape as gate)
        let up_params = GpuMatmulParams {
            m: 1,
            k: params.input_dim as u32,
            n: params.intermediate_dim as u32,
        };
        encode_with_args(
            encoder,
            matvec_pipeline,
            &[
                (0, KernelArg::Buffer(ew.up_proj)),
                (1, KernelArg::Buffer(input)),
                (2, KernelArg::Buffer(scratch_up)),
                (3, KernelArg::Bytes(as_bytes(&up_params))),
            ],
            MTLSize::new(params.intermediate_dim as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, params.intermediate_dim as u64), 1, 1),
        );

        // hidden = GELU(gate_out) * up_out
        let gelu_params = GpuFusedGeluMulParams {
            n_elements: params.intermediate_dim as u32,
        };
        encode_with_args(
            encoder,
            gelu_mul_pipeline,
            &[
                (0, KernelArg::Buffer(scratch_gate)),
                (1, KernelArg::Buffer(scratch_up)),
                (2, KernelArg::Buffer(scratch_hidden)),
                (3, KernelArg::Bytes(as_bytes(&gelu_params))),
            ],
            MTLSize::new(params.intermediate_dim as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, params.intermediate_dim as u64), 1, 1),
        );

        // expert_out = down_proj @ hidden  (matvec: [input_dim, intermediate_dim] @ [intermediate_dim] -> [input_dim])
        let down_params = GpuMatmulParams {
            m: 1,
            k: params.intermediate_dim as u32,
            n: params.input_dim as u32,
        };
        encode_with_args(
            encoder,
            matvec_pipeline,
            &[
                (0, KernelArg::Buffer(ew.down_proj)),
                (1, KernelArg::Buffer(scratch_hidden)),
                (2, KernelArg::Buffer(scratch_expert)),
                (3, KernelArg::Bytes(as_bytes(&down_params))),
            ],
            MTLSize::new(params.input_dim as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, params.input_dim as u64), 1, 1),
        );

        // result += w * expert_out
        let accum_params = GpuMoeAccumParams {
            n_elements: params.input_dim as u32,
            routing_weight: w,
        };
        encode_with_args(
            encoder,
            accum_pipeline,
            &[
                (0, KernelArg::Buffer(output)),
                (1, KernelArg::Buffer(scratch_expert)),
                (2, KernelArg::Bytes(as_bytes(&accum_params))),
            ],
            MTLSize::new(params.input_dim as u64, 1, 1),
            MTLSize::new(std::cmp::min(256, params.input_dim as u64), 1, 1),
        );
    }

    Ok(())
}

/// Zero-initialize an f32 GPU buffer using the `zero_buffer` kernel.
///
/// This is useful for preparing an accumulator buffer before dispatching
/// weighted accumulation passes.
///
/// # Arguments
/// * `encoder`    — Command encoder to record into.
/// * `registry`   — Kernel registry for pipeline lookup.
/// * `device`     — Metal device reference.
/// * `output`     — f32 buffer to zero, must be at least `n_elements * 4` bytes.
/// * `n_elements` — Number of f32 elements to zero.
pub fn moe_zero_buffer_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    output: &MlxBuffer,
    n_elements: usize,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_zero_buffer_encode: n_elements must be > 0".into(),
        ));
    }
    let required = n_elements * std::mem::size_of::<f32>();
    if output.byte_len() < required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_zero_buffer_encode: buffer too small: need {} bytes, have {}",
            required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("zero_buffer", device)?;
    let params = GpuZeroParams { n_elements: n_elements as u32 };
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(output)),
            (1, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(n_elements as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_elements as u64), 1, 1),
    );
    Ok(())
}

/// Encode a weighted accumulation: `accumulator[i] += routing_weight * expert_output[i]`.
///
/// Uses the `moe_accumulate` kernel from moe_dispatch.metal.
///
/// # Arguments
/// * `encoder`        — Command encoder to record into.
/// * `registry`       — Kernel registry for pipeline lookup.
/// * `device`         — Metal device reference.
/// * `accumulator`    — f32 buffer `[n_elements]`, in/out.
/// * `expert_output`  — f32 buffer `[n_elements]`, input.
/// * `routing_weight` — Scalar weight for this expert.
/// * `n_elements`     — Number of f32 elements.
pub fn moe_accumulate_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    accumulator: &MlxBuffer,
    expert_output: &MlxBuffer,
    routing_weight: f32,
    n_elements: usize,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_accumulate_encode: n_elements must be > 0".into(),
        ));
    }
    let required = n_elements * std::mem::size_of::<f32>();
    if accumulator.byte_len() < required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_accumulate_encode: accumulator too small: need {} bytes, have {}",
            required, accumulator.byte_len()
        )));
    }
    if expert_output.byte_len() < required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_accumulate_encode: expert_output too small: need {} bytes, have {}",
            required, expert_output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_accumulate", device)?;
    let params = GpuMoeAccumParams {
        n_elements: n_elements as u32,
        routing_weight,
    };
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(accumulator)),
            (1, KernelArg::Buffer(expert_output)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(n_elements as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_elements as u64), 1, 1),
    );
    Ok(())
}
