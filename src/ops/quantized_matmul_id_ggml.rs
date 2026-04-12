//! GGML block-format expert-routed (MoE) quantized matrix-vector multiply dispatch.
//!
//! Encodes a GPU compute command that performs, for each (token, expert-slot):
//!   expert_id = ids[token * top_k + slot]
//!   output[token*top_k + slot][col] = sum_k(dequant(weight[expert_id][col][k]) * input[token][k])
//!
//! This is the _id variant of quantized_matmul_ggml: same GGML block dequantization
//! but with per-token expert selection via an ids buffer, enabling fused MoE dispatch.
//!
//! Derived from candle-metal-kernels (Apache-2.0) kernel_mul_mv_id template
//! and mlx-native's quantized_matmul_ggml kernels.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::quantized_matmul_ggml::GgmlType;

// ---- GPU params struct ----

/// GPU-side params struct — must match the Metal shader's `GgmlMatvecIdParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GgmlMatvecIdGpuParams {
    ne00: i64,           // K
    ne01: i64,           // N
    ne02: i64,           // 1 (unused)
    ne10: i64,           // K
    ne12: i64,           // 1 (unused)
    ne0: i64,            // N (output stride)
    ne1: i64,            // total output rows = n_tokens * top_k
    r2: u32,             // 1
    r3: u32,             // 1
    top_k: u32,          // experts per token
    n_tokens: u32,       // number of input tokens
    expert_stride: i64,  // bytes between expert weight slices
}

// ---- Public types ----

/// Parameters describing the expert-routed GGML quantized matmul dimensions.
#[derive(Debug, Clone, Copy)]
pub struct GgmlQuantizedMatmulIdParams {
    /// Number of input tokens.
    pub n_tokens: u32,
    /// Number of experts each token is routed to (top-k).
    pub top_k: u32,
    /// Number of output columns per expert (weight rows).
    pub n: u32,
    /// Input dimension (weight cols before quantization).
    /// Must be divisible by the GGML block QK value.
    pub k: u32,
    /// Total number of experts in the stacked weight buffer.
    pub n_experts: u32,
    /// Byte stride between expert weight slices in the stacked buffer.
    pub expert_stride: u64,
    /// GGML quantization type.
    pub ggml_type: GgmlType,
}

impl GgmlType {
    /// Metal kernel function name for the _id variant.
    fn id_kernel_name(self) -> &'static str {
        match self {
            GgmlType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
            GgmlType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
            GgmlType::Q6_K => "kernel_mul_mv_id_q6_K_f32",
        }
    }
}

/// Encode an expert-routed GGML quantized matrix-vector multiply.
///
/// Weight buffer contains raw GGML blocks stacked as `[n_experts, N, packed_K]`.
/// Input is f32 `[n_tokens, K]`, output is f32 `[n_tokens * top_k, N]`.
/// The `ids` buffer `[n_tokens * top_k]` u32 selects which expert to use for
/// each (token, slot) pair.
///
/// # Arguments
///
/// * `encoder`  -- Command encoder to record the dispatch into.
/// * `registry` -- Kernel registry (compiles shader on first call).
/// * `device`   -- Metal device.
/// * `input`    -- f32 input buffer, shape `[n_tokens, K]`.
/// * `weight`   -- Stacked GGML block weight buffer, `[n_experts, N, packed_K]`.
/// * `ids`      -- u32 expert index buffer, shape `[n_tokens * top_k]`.
/// * `output`   -- f32 output buffer, shape `[n_tokens * top_k, N]`.
/// * `params`   -- Dimensions and quantization parameters.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - K is not divisible by the GGML block QK value
/// - Buffer sizes don't match expected dimensions
/// - Any dimension is zero
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id_ggml(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    ids: &MlxBuffer,
    output: &mut MlxBuffer,
    params: &GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    let qk = params.ggml_type.block_values();
    let block_bytes = params.ggml_type.block_bytes();

    // --- Validate dimensions ---
    if params.n_tokens == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml: n_tokens, K, and N must all be > 0".into(),
        ));
    }
    if params.top_k == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml: top_k must be > 0".into(),
        ));
    }
    if params.n_experts == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id_ggml: n_experts must be > 0".into(),
        ));
    }
    if params.k % qk != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: K ({}) must be divisible by block QK ({})",
            params.k, qk
        )));
    }

    // --- Validate buffer sizes ---
    let expected_input_bytes =
        (params.n_tokens as usize) * (params.k as usize) * DType::F32.size_of();
    if input.byte_len() < expected_input_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: input buffer too small: expected {} bytes for [{} x {}] f32, got {}",
            expected_input_bytes, params.n_tokens, params.k, input.byte_len()
        )));
    }

    let blocks_per_row = params.k / qk;
    let per_expert_bytes =
        (params.n as usize) * (blocks_per_row as usize) * (block_bytes as usize);

    // Validate expert_stride is sane
    if params.expert_stride < per_expert_bytes as u64 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: expert_stride ({}) < per_expert_bytes ({})",
            params.expert_stride, per_expert_bytes
        )));
    }

    let total_weight_bytes = per_expert_bytes * (params.n_experts as usize);
    if weight.byte_len() < total_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: weight buffer too small: expected {} bytes for {} experts, got {}",
            total_weight_bytes, params.n_experts, weight.byte_len()
        )));
    }

    let total_rows = (params.n_tokens as usize) * (params.top_k as usize);
    let expected_ids_bytes = total_rows * DType::U32.size_of();
    if ids.byte_len() < expected_ids_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: ids buffer too small: expected {} bytes for [{} * {}] u32, got {}",
            expected_ids_bytes, params.n_tokens, params.top_k, ids.byte_len()
        )));
    }

    let expected_output_bytes = total_rows * (params.n as usize) * DType::F32.size_of();
    if output.byte_len() < expected_output_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id_ggml: output buffer too small: expected {} bytes for [{} x {}] f32, got {}",
            expected_output_bytes, total_rows, params.n, output.byte_len()
        )));
    }

    // --- Get pipeline ---
    let kernel_name = params.ggml_type.id_kernel_name();
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    // --- Build GPU params ---
    let gpu_params = GgmlMatvecIdGpuParams {
        ne00: params.k as i64,
        ne01: params.n as i64,
        ne02: 1,
        ne10: params.k as i64,
        ne12: 1,
        ne0: params.n as i64,
        ne1: total_rows as i64,
        r2: 1,
        r3: 1,
        top_k: params.top_k,
        n_tokens: params.n_tokens,
        expert_stride: params.expert_stride as i64,
    };
    let params_size = std::mem::size_of::<GgmlMatvecIdGpuParams>();
    let mut params_buf = device.alloc_buffer(params_size, DType::U32, vec![params_size / 4])?;
    {
        let slice: &mut [GgmlMatvecIdGpuParams] = bytemuck::cast_slice_mut(
            params_buf
                .as_mut_slice::<u8>()
                .map_err(|e| MlxError::InvalidArgument(format!("params buf write: {e}")))?,
        );
        slice[0] = gpu_params;
    }

    // --- Dispatch ---
    // Threadgroup geometry matches the non-id GGML kernels exactly,
    // but with the Y (row) dimension expanded to total_rows = n_tokens * top_k.
    let (nth0, nth1, align) = match params.ggml_type {
        GgmlType::Q4_0 | GgmlType::Q8_0 => (8u64, 8u64, 8usize),
        GgmlType::Q6_K => (2u64, 32u64, 2usize),
    };

    let n = params.n as usize;
    let m = total_rows;

    let threadgroups = metal::MTLSize::new(
        div_ceil(n, align) as u64,
        m as u64,
        1,
    );
    let threads_per_tg = metal::MTLSize::new(nth0, nth1, 1);

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, weight),    // src0 = stacked expert weights (GGML blocks)
            (1, input),     // src1 = input (f32)
            (2, output),    // dst  = output (f32)
            (3, ids),       // ids  = expert indices (u32)
            (4, &params_buf),
        ],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
