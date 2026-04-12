//! Expert-routed (MoE) quantized matrix-vector multiply dispatch.
//!
//! Encodes a GPU compute command that performs, for each (token, expert-slot):
//!   expert_id = ids[token * n_expert_used + slot]
//!   output[token][slot][col] = sum_k(dequant(expert_weight[expert_id][col][k]) * input[token][k])
//!
//! This is the _id variant of quantized_matmul: same dequantization logic but
//! with per-token expert selection via an ids buffer, enabling fused MoE dispatch.
//!
//! Portions derived from candle-metal-kernels v0.10.2 (Apache-2.0).
//! See src/shaders/quantized_matmul_id.metal for full attribution.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Parameters describing the expert-routed quantized matmul dimensions.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulIdParams {
    /// Number of input rows (tokens).
    pub m: u32,
    /// Inner dimension (shared between input and weight).
    pub k: u32,
    /// Number of output columns per expert.
    pub n: u32,
    /// Number of consecutive values sharing one scale/bias pair.
    pub group_size: u32,
    /// Quantization bit width (4, 6, or 8).
    pub bits: u32,
    /// Number of experts each token is routed to (top-k).
    pub n_expert_used: u32,
    /// Total number of experts in the weight tensor.
    pub num_experts: u32,
}

/// GPU-side params struct -- must match the Metal shader's QuantizedMatmulIdParams.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantizedMatmulIdGpuParams {
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    bits: u32,
    n_expert_used: u32,
    num_experts: u32,
    expert_weight_stride: u32,
    expert_scales_stride: u32,
    expert_biases_stride: u32,
}

/// Compute the expected weight buffer size in bytes for one expert.
fn expert_weight_bytes(k: u32, n: u32, bits: u32) -> usize {
    match bits {
        4 => {
            let values_per_pack = 8u32;
            let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
            (n as usize) * (packs_per_row as usize) * 4
        }
        6 => {
            let triplets_per_row = (k + 3) / 4;
            (n as usize) * (triplets_per_row as usize) * 3
        }
        8 => {
            let values_per_pack = 4u32;
            let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
            (n as usize) * (packs_per_row as usize) * 4
        }
        _ => 0,
    }
}

/// Compute the expected scales (or biases) element count for one expert.
/// Each output column has ceil(K / group_size) groups, each with one bf16 value.
fn expert_scales_elements(k: u32, n: u32, group_size: u32) -> usize {
    let num_groups = (k + group_size - 1) / group_size;
    (n as usize) * (num_groups as usize)
}

/// Encode an expert-routed quantized matrix multiplication onto the command encoder.
///
/// This does **not** commit the command buffer -- the caller is responsible for
/// calling `encoder.commit_and_wait()` after encoding all desired operations.
///
/// # Arguments
///
/// * `encoder`  -- The command encoder to record the dispatch into.
/// * `registry` -- Kernel registry (compiles the shader on first call).
/// * `device`   -- The Metal device (needed for pipeline compilation and output allocation).
/// * `input`    -- f32 input matrix buffer, shape `[M, K]`.
/// * `weight`   -- Packed quantized weight buffer, shape `[num_experts, N, packed_k]` contiguous.
/// * `scales`   -- bf16 scale buffer, shape `[num_experts, N, num_groups]` contiguous.
/// * `biases`   -- bf16 bias buffer, shape `[num_experts, N, num_groups]` contiguous.
/// * `ids`      -- u32 expert index buffer, shape `[M, n_expert_used]`.
/// * `params`   -- Dimensions and quantization parameters.
///
/// # Returns
///
/// A freshly allocated `MlxBuffer` for the output of shape `[M, n_expert_used, N]`
/// with dtype `F32`.
///
/// # Errors
///
/// * `MlxError::InvalidArgument` -- unsupported `bits` value, or buffer sizes
///   do not match the expected dimensions.
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul_id(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    ids: &MlxBuffer,
    params: &QuantizedMatmulIdParams,
) -> Result<MlxBuffer> {
    // --- Validate bits ---
    if params.bits != 4 && params.bits != 6 && params.bits != 8 {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id: unsupported bits value {}; only 4, 6, and 8 are supported",
            params.bits
        )));
    }

    // --- Validate dimensions are non-zero ---
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id: M, K, and N must all be > 0".into(),
        ));
    }
    if params.group_size == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id: group_size must be > 0".into(),
        ));
    }
    if params.n_expert_used == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id: n_expert_used must be > 0".into(),
        ));
    }
    if params.num_experts == 0 {
        return Err(MlxError::InvalidArgument(
            "quantized_matmul_id: num_experts must be > 0".into(),
        ));
    }

    // --- Validate buffer sizes ---
    let expected_input = (params.m as usize) * (params.k as usize) * DType::F32.size_of();
    if input.byte_len() < expected_input {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id: input buffer too small: expected at least {} bytes for [{}x{}] f32, got {}",
            expected_input, params.m, params.k, input.byte_len()
        )));
    }

    let per_expert_w = expert_weight_bytes(params.k, params.n, params.bits);
    let total_w = per_expert_w * (params.num_experts as usize);
    if weight.byte_len() < total_w {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id: weight buffer too small: expected at least {} bytes for {} experts, got {}",
            total_w, params.num_experts, weight.byte_len()
        )));
    }

    let per_expert_s = expert_scales_elements(params.k, params.n, params.group_size);
    let total_s_bytes = per_expert_s * (params.num_experts as usize) * 2; // 2 bytes per bf16
    if scales.byte_len() < total_s_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id: scales buffer too small: expected at least {} bytes, got {}",
            total_s_bytes, scales.byte_len()
        )));
    }
    if biases.byte_len() < total_s_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id: biases buffer too small: expected at least {} bytes, got {}",
            total_s_bytes, biases.byte_len()
        )));
    }

    let expected_ids = (params.m as usize) * (params.n_expert_used as usize) * DType::U32.size_of();
    if ids.byte_len() < expected_ids {
        return Err(MlxError::InvalidArgument(format!(
            "quantized_matmul_id: ids buffer too small: expected at least {} bytes for [{}x{}] u32, got {}",
            expected_ids, params.m, params.n_expert_used, ids.byte_len()
        )));
    }

    // --- Get (or compile) the pipeline ---
    let pipeline = registry.get_pipeline("quantized_matmul_id", device.metal_device())?;

    // --- Allocate output buffer ---
    let output_elems = (params.m as usize) * (params.n_expert_used as usize) * (params.n as usize);
    let output_bytes = output_elems * DType::F32.size_of();
    let output = device.alloc_buffer(
        output_bytes,
        DType::F32,
        vec![
            params.m as usize,
            params.n_expert_used as usize,
            params.n as usize,
        ],
    )?;

    // --- Create GPU params ---
    let gpu_params = QuantizedMatmulIdGpuParams {
        m: params.m,
        k: params.k,
        n: params.n,
        group_size: params.group_size,
        bits: params.bits,
        n_expert_used: params.n_expert_used,
        num_experts: params.num_experts,
        expert_weight_stride: per_expert_w as u32,
        expert_scales_stride: per_expert_s as u32,
        expert_biases_stride: per_expert_s as u32,
    };
    let params_bytes = std::mem::size_of::<QuantizedMatmulIdGpuParams>();
    let mut params_buf = device.alloc_buffer(params_bytes, DType::U32, vec![10])?;
    {
        let slice: &mut [QuantizedMatmulIdGpuParams] = bytemuck::cast_slice_mut(
            params_buf
                .as_mut_slice::<u8>()
                .map_err(|e| MlxError::InvalidArgument(format!("params buf write: {e}")))?,
        );
        slice[0] = gpu_params;
    }

    // --- Dispatch ---
    // Grid: (N, M * n_expert_used, 1)
    let total_rows = (params.m as u64) * (params.n_expert_used as u64);
    let tg_x = 16u64.min(params.n as u64);
    let tg_y = 16u64.min(total_rows);
    let threadgroup_size = metal::MTLSize::new(tg_x, tg_y, 1);

    let grid_groups = metal::MTLSize::new(
        (params.n as u64 + tg_x - 1) / tg_x,
        (total_rows + tg_y - 1) / tg_y,
        1,
    );

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, input),
            (1, weight),
            (2, scales),
            (3, biases),
            (4, ids),
            (5, &output),
            (6, &params_buf),
        ],
        grid_groups,
        threadgroup_size,
    );

    Ok(output)
}
