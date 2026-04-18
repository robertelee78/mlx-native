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

        // Barrier after zero_buffer (first iteration) or after previous
        // accumulate (subsequent iterations) — the output buffer was just written.
        encoder.memory_barrier();

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

        // Barrier: gate_out and up_out must complete before gelu_mul reads them.
        encoder.memory_barrier();

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

        // Barrier: hidden must complete before down_proj reads it.
        encoder.memory_barrier();

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

        // Barrier: expert_out must complete before accumulate reads it.
        encoder.memory_barrier();

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

/// Encode a fused SwiGLU on a `[2*N]` gate_up buffer, producing `[N]` output.
///
/// Computes `output[i] = GELU(gate_up[i]) * gate_up[N + i]` for `i in 0..N`.
///
/// Uses the `moe_swiglu_fused` kernel from moe_dispatch.metal.
///
/// # Arguments
/// * `encoder`        -- Command encoder to record into.
/// * `registry`       -- Kernel registry for pipeline lookup.
/// * `device`         -- Metal device reference.
/// * `gate_up`        -- f32 buffer `[2 * n_elements]` (gate || up concatenated).
/// * `output`         -- f32 buffer `[n_elements]` (output).
/// * `n_elements`     -- Number of output elements (intermediate_dim).
pub fn moe_swiglu_fused_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    gate_up: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_swiglu_fused_encode: n_elements must be > 0".into(),
        ));
    }
    let gu_required = 2 * n_elements * std::mem::size_of::<f32>();
    if gate_up.byte_len() < gu_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_fused_encode: gate_up buffer too small: need {} bytes, have {}",
            gu_required, gate_up.byte_len()
        )));
    }
    let out_required = n_elements * std::mem::size_of::<f32>();
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_fused_encode: output buffer too small: need {} bytes, have {}",
            out_required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_swiglu_fused", device)?;
    let params = GpuFusedGeluMulParams {
        n_elements: n_elements as u32,
    };
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(gate_up)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(n_elements as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_elements as u64), 1, 1),
    );
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

/// Encode a batched SwiGLU across all top_k expert slots in one dispatch.
///
/// Takes a `[top_k, 2*intermediate]` gate_up buffer and produces
/// `[top_k, intermediate]` output: `GELU(gate[i]) * up[i]` per slot.
///
/// Replaces top_k separate `moe_swiglu_fused_encode_offset` dispatches with 1.
///
/// # Arguments
/// * `encoder`       -- Command encoder to record into.
/// * `registry`      -- Kernel registry for pipeline lookup.
/// * `device`        -- Metal device reference.
/// * `gate_up`       -- f32 buffer `[top_k * 2 * intermediate]`.
/// * `output`        -- f32 buffer `[top_k * intermediate]`.
/// * `intermediate`  -- Intermediate dimension per expert.
/// * `top_k`         -- Number of selected expert slots.
pub fn moe_swiglu_batch_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    gate_up: &MlxBuffer,
    output: &MlxBuffer,
    intermediate: usize,
    top_k: usize,
) -> Result<()> {
    if intermediate == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_swiglu_batch_encode: intermediate and top_k must be > 0".into(),
        ));
    }
    let gu_required = top_k * 2 * intermediate * std::mem::size_of::<f32>();
    if gate_up.byte_len() < gu_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_batch_encode: gate_up too small: need {} bytes, have {}",
            gu_required, gate_up.byte_len()
        )));
    }
    let out_required = top_k * intermediate * std::mem::size_of::<f32>();
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_batch_encode: output too small: need {} bytes, have {}",
            out_required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_swiglu_batch", device)?;
    let intermediate_bytes = (intermediate as u32).to_ne_bytes();
    let top_k_bytes = (top_k as u32).to_ne_bytes();

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(gate_up)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(&intermediate_bytes)),
            (3, KernelArg::Bytes(&top_k_bytes)),
        ],
        MTLSize::new(intermediate as u64, top_k as u64, 1),
        MTLSize::new(std::cmp::min(256, intermediate as u64), 1, 1),
    );
    Ok(())
}

/// MSL-compatible struct for moe_weighted_sum kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMoeWeightedSumParams {
    hidden_size: u32,
    top_k: u32,
}

/// Encode a weighted sum of all top_k expert outputs in one dispatch.
///
/// Replaces the zero_buffer + top_k * moe_accumulate pattern with 1 dispatch.
/// The weights buffer must contain pre-scaled routing weights for all top_k
/// experts (i.e. `routing_weight * per_expert_scale`).
///
/// # Arguments
/// * `encoder`        -- Command encoder to record into.
/// * `registry`       -- Kernel registry for pipeline lookup.
/// * `device`         -- Metal device reference.
/// * `expert_outputs` -- f32 buffer `[top_k * hidden_size]`.
/// * `weights`        -- f32 buffer `[top_k]` (pre-scaled routing weights).
/// * `output`         -- f32 buffer `[hidden_size]` (output weighted sum).
/// * `hidden_size`    -- Hidden dimension.
/// * `top_k`          -- Number of selected expert slots.
pub fn moe_weighted_sum_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    expert_outputs: &MlxBuffer,
    weights: &MlxBuffer,
    output: &MlxBuffer,
    hidden_size: usize,
    top_k: usize,
) -> Result<()> {
    if hidden_size == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_weighted_sum_encode: hidden_size and top_k must be > 0".into(),
        ));
    }
    let expert_required = top_k * hidden_size * std::mem::size_of::<f32>();
    if expert_outputs.byte_len() < expert_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_encode: expert_outputs too small: need {} bytes, have {}",
            expert_required, expert_outputs.byte_len()
        )));
    }
    let weights_required = top_k * std::mem::size_of::<f32>();
    if weights.byte_len() < weights_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_encode: weights too small: need {} bytes, have {}",
            weights_required, weights.byte_len()
        )));
    }
    let out_required = hidden_size * std::mem::size_of::<f32>();
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_encode: output too small: need {} bytes, have {}",
            out_required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_weighted_sum", device)?;
    let params = GpuMoeWeightedSumParams {
        hidden_size: hidden_size as u32,
        top_k: top_k as u32,
    };
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(expert_outputs)),
            (1, KernelArg::Buffer(weights)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(hidden_size as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, hidden_size as u64), 1, 1),
    );
    Ok(())
}

/// MSL-compatible struct for moe_gather_topk_weights kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMoeGatherTopkParams {
    n_experts: u32,
    top_k: u32,
}

/// Encode a GPU-side MoE top-K routing gather.
///
/// Reads softmax probs and sorted indices (both on GPU from prior dispatches),
/// gathers the top-K expert IDs and their weights, applies per_expert_scale,
/// and renormalizes.  This eliminates the CPU readback that previously forced
/// a session break between S1 and S4.
///
/// # Arguments
/// * `encoder`          -- Command encoder.
/// * `registry`         -- Kernel registry.
/// * `device`           -- Metal device.
/// * `softmax_probs`    -- f32 `[n_experts]` (output of dispatch_softmax).
/// * `sorted_indices`   -- u32 `[n_experts]` (output of dispatch_argsort_desc_f32).
/// * `per_expert_scale` -- f32 `[n_experts]` (learned per-expert scale).
/// * `out_expert_ids`   -- u32 `[top_k]` (output: selected expert indices).
/// * `out_weights`      -- f32 `[top_k]` (output: pre-scaled routing weights).
/// * `n_experts`        -- Total number of experts.
/// * `top_k`            -- Number of experts to select.
pub fn moe_gather_topk_weights_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    softmax_probs: &MlxBuffer,
    sorted_indices: &MlxBuffer,
    per_expert_scale: &MlxBuffer,
    out_expert_ids: &MlxBuffer,
    out_weights: &MlxBuffer,
    n_experts: usize,
    top_k: usize,
) -> Result<()> {
    if n_experts == 0 || top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_gather_topk_weights: n_experts and top_k must be > 0".into(),
        ));
    }
    if top_k > n_experts {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gather_topk_weights: top_k ({}) > n_experts ({})",
            top_k, n_experts,
        )));
    }
    if top_k > 8 {
        return Err(MlxError::InvalidArgument(format!(
            "moe_gather_topk_weights: top_k ({}) > 8 (shader fixed-size array limit)",
            top_k,
        )));
    }

    let f32_size = std::mem::size_of::<f32>();
    let u32_size = std::mem::size_of::<u32>();
    if softmax_probs.byte_len() < n_experts * f32_size {
        return Err(MlxError::InvalidArgument("softmax_probs too small".into()));
    }
    if sorted_indices.byte_len() < n_experts * u32_size {
        return Err(MlxError::InvalidArgument("sorted_indices too small".into()));
    }
    if per_expert_scale.byte_len() < n_experts * f32_size {
        return Err(MlxError::InvalidArgument("per_expert_scale too small".into()));
    }
    if out_expert_ids.byte_len() < top_k * u32_size {
        return Err(MlxError::InvalidArgument("out_expert_ids too small".into()));
    }
    if out_weights.byte_len() < top_k * f32_size {
        return Err(MlxError::InvalidArgument("out_weights too small".into()));
    }

    let pipeline = registry.get_pipeline("moe_gather_topk_weights", device)?;
    let params = GpuMoeGatherTopkParams {
        n_experts: n_experts as u32,
        top_k: top_k as u32,
    };
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(softmax_probs)),
            (1, KernelArg::Buffer(sorted_indices)),
            (2, KernelArg::Buffer(per_expert_scale)),
            (3, KernelArg::Buffer(out_expert_ids)),
            (4, KernelArg::Buffer(out_weights)),
            (5, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(1, 1, 1),  // single thread
        MTLSize::new(1, 1, 1),
    );
    Ok(())
}

/// Like [`moe_swiglu_fused_encode`] but reads from `gate_up` at `gu_byte_offset`
/// and writes to `output` at `out_byte_offset`.
///
/// This enables operating on slices within larger buffers (e.g. the _id kernel
/// output which contains top_k rows of gate_up data).
#[allow(clippy::too_many_arguments)]
pub fn moe_swiglu_fused_encode_offset(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    gate_up: &MlxBuffer,
    gu_byte_offset: usize,
    output: &MlxBuffer,
    out_byte_offset: usize,
    n_elements: usize,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_swiglu_fused_encode_offset: n_elements must be > 0".into(),
        ));
    }
    let gu_required = gu_byte_offset + 2 * n_elements * std::mem::size_of::<f32>();
    if gate_up.byte_len() < gu_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_fused_encode_offset: gate_up buffer too small: need {} bytes (offset {}), have {}",
            gu_required, gu_byte_offset, gate_up.byte_len()
        )));
    }
    let out_required = out_byte_offset + n_elements * std::mem::size_of::<f32>();
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_fused_encode_offset: output buffer too small: need {} bytes (offset {}), have {}",
            out_required, out_byte_offset, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_swiglu_fused", device)?;
    let params = GpuFusedGeluMulParams {
        n_elements: n_elements as u32,
    };
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::BufferWithOffset(gate_up, gu_byte_offset as u64)),
            (1, KernelArg::BufferWithOffset(output, out_byte_offset as u64)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(n_elements as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_elements as u64), 1, 1),
    );
    Ok(())
}

/// Like [`moe_accumulate_encode`] but reads `expert_output` from `src_byte_offset`.
///
/// This enables reading from a slice within a larger buffer (e.g. the down _id
/// kernel output which contains top_k rows of hidden data).
#[allow(clippy::too_many_arguments)]
pub fn moe_accumulate_encode_offset(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    accumulator: &MlxBuffer,
    expert_output: &MlxBuffer,
    src_byte_offset: usize,
    routing_weight: f32,
    n_elements: usize,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_accumulate_encode_offset: n_elements must be > 0".into(),
        ));
    }
    let required = n_elements * std::mem::size_of::<f32>();
    if accumulator.byte_len() < required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_accumulate_encode_offset: accumulator too small: need {} bytes, have {}",
            required, accumulator.byte_len()
        )));
    }
    let src_required = src_byte_offset + required;
    if expert_output.byte_len() < src_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_accumulate_encode_offset: expert_output too small: need {} bytes (offset {}), have {}",
            src_required, src_byte_offset, expert_output.byte_len()
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
            (1, KernelArg::BufferWithOffset(expert_output, src_byte_offset as u64)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(n_elements as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_elements as u64), 1, 1),
    );
    Ok(())
}

/// Encode a fused GELU-multiply on bf16 buffers.
///
/// Computes `output[i] = GELU(gate_out[i]) * up_out[i]` with bf16 I/O and
/// f32 accumulator.  Port of [`fused_gelu_mul`] for the bf16 activation path.
///
/// # Arguments
/// * `encoder`    — Command encoder.
/// * `registry`   — Kernel registry.
/// * `device`     — Metal device.
/// * `gate_out`   — bf16 buffer `[n_elements]`.
/// * `up_out`     — bf16 buffer `[n_elements]`.
/// * `output`     — bf16 buffer `[n_elements]`.
/// * `n_elements` — Number of elements.
pub fn fused_gelu_mul_bf16_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    gate_out: &MlxBuffer,
    up_out: &MlxBuffer,
    output: &MlxBuffer,
    n_elements: usize,
) -> Result<()> {
    if n_elements == 0 {
        return Err(MlxError::InvalidArgument(
            "fused_gelu_mul_bf16_encode: n_elements must be > 0".into(),
        ));
    }
    // bf16 is 2 bytes per element
    let required = n_elements * 2;
    if gate_out.byte_len() < required {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gelu_mul_bf16_encode: gate_out too small: need {} bytes, have {}",
            required, gate_out.byte_len()
        )));
    }
    if up_out.byte_len() < required {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gelu_mul_bf16_encode: up_out too small: need {} bytes, have {}",
            required, up_out.byte_len()
        )));
    }
    if output.byte_len() < required {
        return Err(MlxError::InvalidArgument(format!(
            "fused_gelu_mul_bf16_encode: output too small: need {} bytes, have {}",
            required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("fused_gelu_mul_bf16", device)?;
    let params = GpuFusedGeluMulParams {
        n_elements: n_elements as u32,
    };
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(gate_out)),
            (1, KernelArg::Buffer(up_out)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(n_elements as u64, 1, 1),
        MTLSize::new(std::cmp::min(256, n_elements as u64), 1, 1),
    );
    Ok(())
}

/// GPU params for batched SwiGLU over multiple tokens.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMoeSwigluSeqParams {
    intermediate: u32,
    top_k: u32,
    n_tokens: u32,
}

/// Multi-token SwiGLU for batched prefill.
///
/// Input:  `[n_tokens, top_k, 2*intermediate]`
/// Output: `[n_tokens, top_k, intermediate]`
#[allow(clippy::too_many_arguments)]
pub fn moe_swiglu_seq_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    gate_up: &MlxBuffer,
    output: &MlxBuffer,
    intermediate: usize,
    top_k: usize,
    n_tokens: usize,
) -> Result<()> {
    if intermediate == 0 || top_k == 0 || n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_swiglu_seq_encode: all dims must be > 0".into(),
        ));
    }
    let gu_required = n_tokens * top_k * 2 * intermediate * std::mem::size_of::<f32>();
    if gate_up.byte_len() < gu_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_seq_encode: gate_up too small: need {} bytes, have {}",
            gu_required, gate_up.byte_len()
        )));
    }
    let out_required = n_tokens * top_k * intermediate * std::mem::size_of::<f32>();
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_seq_encode: output too small: need {} bytes, have {}",
            out_required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_swiglu_seq", device)?;
    let gpu_params = GpuMoeSwigluSeqParams {
        intermediate: intermediate as u32,
        top_k: top_k as u32,
        n_tokens: n_tokens as u32,
    };

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(gate_up)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        MTLSize::new(intermediate as u64, top_k as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, intermediate as u64), 1, 1),
    );
    Ok(())
}

/// GPU params for batched weighted-sum over multiple tokens.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMoeWeightedSumSeqParams {
    hidden_size: u32,
    top_k: u32,
    n_tokens: u32,
}

/// Multi-token weighted sum of expert outputs for batched prefill.
///
/// * `expert_outputs` — `[n_tokens, top_k, hidden_size]`
/// * `weights`        — `[n_tokens, top_k]`
/// * `output`         — `[n_tokens, hidden_size]`
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_seq_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    expert_outputs: &MlxBuffer,
    weights: &MlxBuffer,
    output: &MlxBuffer,
    hidden_size: usize,
    top_k: usize,
    n_tokens: usize,
) -> Result<()> {
    if hidden_size == 0 || top_k == 0 || n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_weighted_sum_seq_encode: all dims must be > 0".into(),
        ));
    }
    let expert_required = n_tokens * top_k * hidden_size * std::mem::size_of::<f32>();
    if expert_outputs.byte_len() < expert_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_seq_encode: expert_outputs too small: need {} bytes, have {}",
            expert_required, expert_outputs.byte_len()
        )));
    }
    let weights_required = n_tokens * top_k * std::mem::size_of::<f32>();
    if weights.byte_len() < weights_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_seq_encode: weights too small: need {} bytes, have {}",
            weights_required, weights.byte_len()
        )));
    }
    let out_required = n_tokens * hidden_size * std::mem::size_of::<f32>();
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_seq_encode: output too small: need {} bytes, have {}",
            out_required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_weighted_sum_seq", device)?;
    let gpu_params = GpuMoeWeightedSumSeqParams {
        hidden_size: hidden_size as u32,
        top_k: top_k as u32,
        n_tokens: n_tokens as u32,
    };

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(expert_outputs)),
            (1, KernelArg::Buffer(weights)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        MTLSize::new(hidden_size as u64, n_tokens as u64, 1),
        MTLSize::new(std::cmp::min(256, hidden_size as u64), 1, 1),
    );
    Ok(())
}

/// Multi-token SwiGLU for batched prefill (bf16 I/O, f32 accumulator).
///
/// Port of [`moe_swiglu_seq_encode`] with bf16 buffers.
/// Input:  `[n_tokens, top_k, 2*intermediate]` bf16
/// Output: `[n_tokens, top_k, intermediate]`   bf16
#[allow(clippy::too_many_arguments)]
pub fn moe_swiglu_seq_bf16_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    gate_up: &MlxBuffer,
    output: &MlxBuffer,
    intermediate: usize,
    top_k: usize,
    n_tokens: usize,
) -> Result<()> {
    if intermediate == 0 || top_k == 0 || n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_swiglu_seq_bf16_encode: all dims must be > 0".into(),
        ));
    }
    // bf16 = 2 bytes per element
    let gu_required = n_tokens * top_k * 2 * intermediate * 2;
    if gate_up.byte_len() < gu_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_seq_bf16_encode: gate_up too small: need {} bytes, have {}",
            gu_required, gate_up.byte_len()
        )));
    }
    let out_required = n_tokens * top_k * intermediate * 2;
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_swiglu_seq_bf16_encode: output too small: need {} bytes, have {}",
            out_required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_swiglu_seq_bf16", device)?;
    let gpu_params = GpuMoeSwigluSeqParams {
        intermediate: intermediate as u32,
        top_k: top_k as u32,
        n_tokens: n_tokens as u32,
    };

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(gate_up)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        MTLSize::new(intermediate as u64, top_k as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, intermediate as u64), 1, 1),
    );
    Ok(())
}

/// Multi-token weighted sum of expert outputs for batched prefill (bf16 inputs).
///
/// Port of [`moe_weighted_sum_seq_encode`] that accepts bf16 expert_outputs
/// and produces f32 output — matching the convention where expert intermediates
/// are bf16 but the weighted accumulator (pf_moe_accum) stays f32 for residual
/// precision.
///
/// * `expert_outputs` — bf16 `[n_tokens, top_k, hidden_size]`
/// * `weights`        — f32  `[n_tokens, top_k]`
/// * `output`         — f32  `[n_tokens, hidden_size]`
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_seq_bf16_input_encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    expert_outputs: &MlxBuffer,
    weights: &MlxBuffer,
    output: &MlxBuffer,
    hidden_size: usize,
    top_k: usize,
    n_tokens: usize,
) -> Result<()> {
    if hidden_size == 0 || top_k == 0 || n_tokens == 0 {
        return Err(MlxError::InvalidArgument(
            "moe_weighted_sum_seq_bf16_input_encode: all dims must be > 0".into(),
        ));
    }
    // expert_outputs is bf16 (2 bytes per element)
    let expert_required = n_tokens * top_k * hidden_size * 2;
    if expert_outputs.byte_len() < expert_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_seq_bf16_input_encode: expert_outputs too small: need {} bytes, have {}",
            expert_required, expert_outputs.byte_len()
        )));
    }
    let weights_required = n_tokens * top_k * std::mem::size_of::<f32>();
    if weights.byte_len() < weights_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_seq_bf16_input_encode: weights too small: need {} bytes, have {}",
            weights_required, weights.byte_len()
        )));
    }
    let out_required = n_tokens * hidden_size * std::mem::size_of::<f32>();
    if output.byte_len() < out_required {
        return Err(MlxError::InvalidArgument(format!(
            "moe_weighted_sum_seq_bf16_input_encode: output too small: need {} bytes, have {}",
            out_required, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("moe_weighted_sum_seq_bf16_input", device)?;
    let gpu_params = GpuMoeWeightedSumSeqParams {
        hidden_size: hidden_size as u32,
        top_k: top_k as u32,
        n_tokens: n_tokens as u32,
    };

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(expert_outputs)),
            (1, KernelArg::Buffer(weights)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        MTLSize::new(hidden_size as u64, n_tokens as u64, 1),
        MTLSize::new(std::cmp::min(256, hidden_size as u64), 1, 1),
    );
    Ok(())
}
