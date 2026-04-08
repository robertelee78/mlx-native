//! Quantized matrix multiplication host-side dispatch.
//!
//! Encodes a GPU compute command that performs:
//!   output[row][col] = sum_k(dequant(weight[col][k]) * input[row][k])
//!
//! Weights are stored in packed quantized format (4-bit or 6-bit) with per-group
//! bf16 scales and biases for affine dequantization.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Parameters describing the quantized matmul dimensions and format.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulParams {
    /// Number of input rows (tokens).
    pub m: u32,
    /// Inner dimension (shared between input and weight).
    pub k: u32,
    /// Number of output columns.
    pub n: u32,
    /// Number of consecutive values sharing one scale/bias pair.
    pub group_size: u32,
    /// Quantization bit width (4, 6, or 8).
    pub bits: u32,
}

/// GPU-side params struct — must match the Metal shader's `QuantizedMatmulParams`.
///
/// This is `#[repr(C)]` to guarantee C-compatible layout for Metal buffer binding.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantizedMatmulGpuParams {
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    bits: u32,
}

/// Compute the expected weight buffer size in bytes for the given parameters.
///
/// - 4-bit: 8 values per uint32, so each row of K values needs ceil(K/8) uint32s.
///   Total = N * ceil(K/8) * 4 bytes.
/// - 6-bit: 4 values per uint32, so each row needs ceil(K/4) uint32s.
///   Total = N * ceil(K/4) * 4 bytes.
/// - 8-bit: 4 values per uint32, so each row needs ceil(K/4) uint32s.
///   Total = N * ceil(K/4) * 4 bytes.
fn expected_weight_bytes(k: u32, n: u32, bits: u32) -> usize {
    let values_per_pack: u32 = match bits {
        4 => 8,
        6 | 8 => 4,
        _ => return 0,
    };
    let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
    (n as usize) * (packs_per_row as usize) * 4 // 4 bytes per uint32
}

/// Compute the expected scales (or biases) buffer size in bytes.
///
/// Each output column has ceil(K / group_size) groups, each with one bf16 value.
/// Total = N * ceil(K / group_size) * 2 bytes.
fn expected_scales_bytes(k: u32, n: u32, group_size: u32) -> usize {
    let num_groups = (k + group_size - 1) / group_size;
    (n as usize) * (num_groups as usize) * 2 // 2 bytes per bf16
}

/// Encode a quantized matrix multiplication onto the given command encoder.
///
/// This does **not** commit the command buffer — the caller is responsible for
/// calling `encoder.commit_and_wait()` after encoding all desired operations.
///
/// # Arguments
///
/// * `encoder`  — The command encoder to record the dispatch into.
/// * `registry` — Kernel registry (compiles the shader on first call).
/// * `device`   — The Metal device (needed for pipeline compilation and output allocation).
/// * `input`    — f32 input matrix buffer, shape `[M, K]`.
/// * `weight`   — Packed quantized weight buffer, shape `[N, packed_k]`.
/// * `scales`   — bf16 scale buffer, shape `[N, num_groups]`.
/// * `biases`   — bf16 bias buffer, shape `[N, num_groups]`.
/// * `params`   — Dimensions and quantization parameters.
///
/// # Returns
///
/// A freshly allocated `MlxBuffer` for the output of shape `[M, N]` with dtype `F32`.
///
/// # Errors
///
/// * `MlxError::InvalidArgument` — unsupported `bits` value, or buffer sizes
///   do not match the expected dimensions.
pub fn quantized_matmul(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    params: &QuantizedMatmulParams,
) -> Result<MlxBuffer> {
    // --- Validate bits ---
    if params.bits != 4 && params.bits != 6 && params.bits != 8 {
        return Err(MlxError::InvalidArgument(format!(
            "Unsupported bits value {}; only 4, 6, and 8 are supported",
            params.bits
        )));
    }

    // --- Validate dimensions are non-zero ---
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, and N must all be > 0".into(),
        ));
    }
    if params.group_size == 0 {
        return Err(MlxError::InvalidArgument(
            "group_size must be > 0".into(),
        ));
    }

    // --- Validate buffer sizes ---
    let expected_input = (params.m as usize) * (params.k as usize) * DType::F32.size_of();
    if input.byte_len() < expected_input {
        return Err(MlxError::InvalidArgument(format!(
            "Input buffer too small: expected at least {} bytes for [{}x{}] f32, got {}",
            expected_input, params.m, params.k, input.byte_len()
        )));
    }

    let expected_w = expected_weight_bytes(params.k, params.n, params.bits);
    if weight.byte_len() < expected_w {
        return Err(MlxError::InvalidArgument(format!(
            "Weight buffer too small: expected at least {} bytes for {}bit [{}x{}], got {}",
            expected_w, params.bits, params.n, params.k, weight.byte_len()
        )));
    }

    let expected_s = expected_scales_bytes(params.k, params.n, params.group_size);
    if scales.byte_len() < expected_s {
        return Err(MlxError::InvalidArgument(format!(
            "Scales buffer too small: expected at least {} bytes, got {}",
            expected_s, scales.byte_len()
        )));
    }
    if biases.byte_len() < expected_s {
        return Err(MlxError::InvalidArgument(format!(
            "Biases buffer too small: expected at least {} bytes, got {}",
            expected_s, biases.byte_len()
        )));
    }

    // --- Get (or compile) the pipeline ---
    let pipeline = registry.get_pipeline("quantized_matmul", device.metal_device())?;

    // --- Allocate output buffer ---
    // Output is f32 to avoid f16 overflow (max ~65504) on projections with large
    // accumulated values (e.g. attention output projections where K=4096).
    let output_bytes = (params.m as usize) * (params.n as usize) * DType::F32.size_of();
    let output = device.alloc_buffer(
        output_bytes,
        DType::F32,
        vec![params.m as usize, params.n as usize],
    )?;

    // --- Create GPU params buffer ---
    let gpu_params = QuantizedMatmulGpuParams {
        m: params.m,
        k: params.k,
        n: params.n,
        group_size: params.group_size,
        bits: params.bits,
    };
    let params_bytes = std::mem::size_of::<QuantizedMatmulGpuParams>();
    let mut params_buf = device.alloc_buffer(params_bytes, DType::U32, vec![5])?;
    {
        let slice: &mut [QuantizedMatmulGpuParams] = bytemuck::cast_slice_mut(
            params_buf
                .as_mut_slice::<u8>()
                .map_err(|e| MlxError::InvalidArgument(format!("params buf write: {e}")))?,
        );
        slice[0] = gpu_params;
    }

    // --- Dispatch ---
    // Grid: (N, M, 1) — one thread per output element.
    // Threadgroup: up to 256 threads, arranged as (tx, ty, 1).
    // We use encode_threadgroups with explicit threadgroup counts for non-even dims.

    // Choose threadgroup size: we want a 2D block that fits within 256 threads.
    // Use 16x16 = 256 as a good default for 2D dispatch.
    let tg_x = 16u64.min(params.n as u64);
    let tg_y = 16u64.min(params.m as u64);
    let threadgroup_size = metal::MTLSize::new(tg_x, tg_y, 1);

    // Use encode_threadgroups with ceil-division for non-even dimensions.
    let grid_groups = metal::MTLSize::new(
        (params.n as u64 + tg_x - 1) / tg_x,
        (params.m as u64 + tg_y - 1) / tg_y,
        1,
    );

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, input),
            (1, weight),
            (2, scales),
            (3, biases),
            (4, &output),
            (5, &params_buf),
        ],
        grid_groups,
        threadgroup_size,
    );

    Ok(output)
}

/// Check whether the SIMD-cooperative kernel can be used for the given params.
///
/// The SIMD path requires:
///   - bits is 4 or 8 (not 6)
///   - N is divisible by 8 (results_per_simdgroup * num_simdgroups)
///   - K is divisible by block_size:
///     - 4-bit: K % 512 == 0 (block_size = 16 values/thread * 32 SIMD = 512, qmv_fast)
///     - 8-bit: K % 256 == 0 (block_size = 8 * 32 = 256)
///
/// ADR-002: All SIMD kernels now use qmv_fast params (values_per_thread=16
/// for 4-bit), so 4-bit block_size is 512.  Dimensions like K=2816 that are
/// not 512-aligned fall back to the scalar kernel.
fn can_use_simd_kernel(params: &QuantizedMatmulParams) -> bool {
    let bn = 8u32; // num_simdgroups * results_per_simdgroup
    if params.n % bn != 0 {
        return false;
    }
    match params.bits {
        4 => params.k % 512 == 0,  // qmv_fast: block_size = 16 * 32 = 512
        8 => params.k % 256 == 0,
        _ => false,
    }
}

/// Encode a quantized matrix-vector multiply using the SIMD-cooperative kernel
/// that matches MLX's `qmv_fast` accumulation pattern exactly.
///
/// This kernel uses 2 simdgroups of 32 threads, each producing 4 output rows,
/// with `simd_sum()` reduction. The accumulation order matches MLX bit-for-bit.
///
/// Falls back to the scalar `quantized_matmul` kernel if the dimensions don't
/// meet the alignment requirements.
///
/// # Arguments
///
/// Same as [`quantized_matmul`].
///
/// # Returns
///
/// A freshly allocated `MlxBuffer` for the output of shape `[M, N]` with dtype `F32`.
pub fn quantized_matmul_simd(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    params: &QuantizedMatmulParams,
) -> Result<MlxBuffer> {
    // Fall back to scalar kernel if dimensions don't support SIMD path.
    if !can_use_simd_kernel(params) {
        return quantized_matmul(encoder, registry, device, input, weight, scales, biases, params);
    }

    // --- Validate bits ---
    if params.bits != 4 && params.bits != 8 {
        return Err(MlxError::InvalidArgument(format!(
            "SIMD kernel: unsupported bits value {}; only 4 and 8 are supported",
            params.bits
        )));
    }

    // --- Validate dimensions are non-zero ---
    if params.m == 0 || params.k == 0 || params.n == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, and N must all be > 0".into(),
        ));
    }
    if params.group_size == 0 {
        return Err(MlxError::InvalidArgument(
            "group_size must be > 0".into(),
        ));
    }

    // --- Validate buffer sizes ---
    let expected_input = (params.m as usize) * (params.k as usize) * DType::F32.size_of();
    if input.byte_len() < expected_input {
        return Err(MlxError::InvalidArgument(format!(
            "Input buffer too small: expected at least {} bytes for [{}x{}] f32, got {}",
            expected_input, params.m, params.k, input.byte_len()
        )));
    }

    let expected_w = expected_weight_bytes(params.k, params.n, params.bits);
    if weight.byte_len() < expected_w {
        return Err(MlxError::InvalidArgument(format!(
            "Weight buffer too small: expected at least {} bytes for {}bit [{}x{}], got {}",
            expected_w, params.bits, params.n, params.k, weight.byte_len()
        )));
    }

    let expected_s = expected_scales_bytes(params.k, params.n, params.group_size);
    if scales.byte_len() < expected_s {
        return Err(MlxError::InvalidArgument(format!(
            "Scales buffer too small: expected at least {} bytes, got {}",
            expected_s, scales.byte_len()
        )));
    }
    if biases.byte_len() < expected_s {
        return Err(MlxError::InvalidArgument(format!(
            "Biases buffer too small: expected at least {} bytes, got {}",
            expected_s, biases.byte_len()
        )));
    }

    // --- Get (or compile) the SIMD pipeline ---
    let pipeline = registry.get_pipeline("quantized_matmul_simd", device.metal_device())?;

    // --- Allocate output buffer ---
    let output_bytes = (params.m as usize) * (params.n as usize) * DType::F32.size_of();
    let output = device.alloc_buffer(
        output_bytes,
        DType::F32,
        vec![params.m as usize, params.n as usize],
    )?;

    // --- Create GPU params buffer ---
    let gpu_params = QuantizedMatmulGpuParams {
        m: params.m,
        k: params.k,
        n: params.n,
        group_size: params.group_size,
        bits: params.bits,
    };
    let params_bytes = std::mem::size_of::<QuantizedMatmulGpuParams>();
    let mut params_buf = device.alloc_buffer(params_bytes, DType::U32, vec![5])?;
    {
        let slice: &mut [QuantizedMatmulGpuParams] = bytemuck::cast_slice_mut(
            params_buf
                .as_mut_slice::<u8>()
                .map_err(|e| MlxError::InvalidArgument(format!("params buf write: {e}")))?,
        );
        slice[0] = gpu_params;
    }

    // --- Dispatch with MLX's qmv_fast pattern ---
    // threadgroup_size = (SIMD_SIZE, num_simdgroups, 1) = (32, 2, 1) = 64 threads
    // threadgroups     = (M, ceil(N / 8), 1)
    //
    // In the kernel:
    //   tid.x = input row (M dimension)
    //   tid.y = output column block (each block produces 8 output columns)
    //   simd_gid = which simdgroup (0 or 1), each handles 4 of the 8 columns
    //   simd_lid = thread index within simdgroup (0..31)
    let num_simdgroups = 2u64;
    let results_per_simdgroup = 4u64;
    let bn = num_simdgroups * results_per_simdgroup; // 8

    let threadgroup_size = metal::MTLSize::new(32, num_simdgroups, 1);
    let threadgroups = metal::MTLSize::new(
        params.m as u64,
        (params.n as u64 + bn - 1) / bn,
        1,
    );

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, input),
            (1, weight),
            (2, scales),
            (3, biases),
            (4, &output),
            (5, &params_buf),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(output)
}

// ---------------------------------------------------------------------------
// bf16 I/O variant — eliminates 2 cast dispatches per projection by accepting
// bfloat input and producing bfloat output directly.
// ---------------------------------------------------------------------------

/// GPU-side params for the bf16 kernels.  Identical layout to
/// `QuantizedMatmulGpuParams`; kept as a separate type for clarity.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QMatmulBf16GpuParams {
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    bits: u32,
}

/// Dispatch the bf16 I/O variant of the SIMD quantized matmul kernel.
///
/// Input and output are both bf16.  Accumulation happens in f32 inside the
/// shader for numerical stability, matching the precision of the f32 variant.
///
/// Falls back to the scalar `quantized_matmul` kernel (with f32 output) if the
/// dimensions don't satisfy SIMD alignment requirements.
///
/// # Arguments
///
/// * `encoder`      — The command encoder to record the dispatch into.
/// * `registry`     — Kernel registry (compiles the shader on first call).
/// * `device`       — The Metal device for buffer allocation.
/// * `input`        — bf16 input matrix buffer, shape `[M, K]`.
/// * `packed_weights` — Packed quantized weight buffer, shape `[N, packed_k]`.
/// * `scales`       — bf16 scale buffer, shape `[N, num_groups]`.
/// * `biases`       — bf16 bias buffer, shape `[N, num_groups]`.
/// * `params`       — Dimensions and quantization parameters.
///
/// # Returns
///
/// A freshly allocated `MlxBuffer` for the output of shape `[M, N]` with dtype `BF16`.
pub fn dispatch_quantized_matmul_simd_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    packed_weights: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    params: &QuantizedMatmulParams,
) -> Result<MlxBuffer> {
    // Fall back to scalar path for unsupported shapes.
    // The scalar kernel expects f32, so cast bf16 input first.
    if !can_use_simd_kernel(params) {
        let n_in = (params.m as usize) * (params.k as usize);
        let f32_input = if input.dtype() == DType::BF16 {
            let f32_buf = device.alloc_buffer(n_in * DType::F32.size_of(), DType::F32, vec![params.m as usize, params.k as usize])?;
            crate::ops::elementwise::cast(
                encoder, registry, device.metal_device(),
                input, &f32_buf, n_in,
                crate::ops::elementwise::CastDirection::BF16ToF32,
            )?;
            Some(f32_buf)
        } else {
            None
        };
        let actual_input = f32_input.as_ref().unwrap_or(input);
        let f32_result = quantized_matmul(encoder, registry, device, actual_input, packed_weights, scales, biases, params)?;
        // Cast f32 output back to bf16
        let n_out = (params.m as usize) * (params.n as usize);
        let bf16_out = device.alloc_buffer(n_out * DType::BF16.size_of(), DType::BF16, vec![params.m as usize, params.n as usize])?;
        crate::ops::elementwise::cast(
            encoder, registry, device.metal_device(),
            &f32_result, &bf16_out, n_out,
            crate::ops::elementwise::CastDirection::F32ToBF16,
        )?;
        return Ok(bf16_out);
    }

    if params.bits != 4 && params.bits != 8 {
        return Err(MlxError::InvalidArgument(format!(
            "bf16 SIMD kernel: unsupported bits value {}; only 4 and 8 are supported",
            params.bits
        )));
    }
    if params.m == 0 || params.k == 0 || params.n == 0 || params.group_size == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, N, and group_size must all be > 0".into(),
        ));
    }

    // Buffer size validation (input is bf16 = 2 bytes per element).
    let expected_input = (params.m as usize) * (params.k as usize) * DType::BF16.size_of();
    if input.byte_len() < expected_input {
        return Err(MlxError::InvalidArgument(format!(
            "bf16 input buffer too small: expected {} bytes for [{}x{}] bf16, got {}",
            expected_input, params.m, params.k, input.byte_len()
        )));
    }

    let expected_w = expected_weight_bytes(params.k, params.n, params.bits);
    if packed_weights.byte_len() < expected_w {
        return Err(MlxError::InvalidArgument(format!(
            "Weight buffer too small: expected {} bytes, got {}",
            expected_w, packed_weights.byte_len()
        )));
    }

    let expected_s = expected_scales_bytes(params.k, params.n, params.group_size);
    if scales.byte_len() < expected_s {
        return Err(MlxError::InvalidArgument(format!(
            "Scales buffer too small: expected {} bytes, got {}",
            expected_s, scales.byte_len()
        )));
    }
    if biases.byte_len() < expected_s {
        return Err(MlxError::InvalidArgument(format!(
            "Biases buffer too small: expected {} bytes, got {}",
            expected_s, biases.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("quantized_matmul_simd_bf16", device.metal_device())?;

    // Output is bf16.
    let output_bytes = (params.m as usize) * (params.n as usize) * DType::BF16.size_of();
    let output = device.alloc_buffer(
        output_bytes,
        DType::BF16,
        vec![params.m as usize, params.n as usize],
    )?;

    let gpu_params = QMatmulBf16GpuParams {
        m: params.m,
        k: params.k,
        n: params.n,
        group_size: params.group_size,
        bits: params.bits,
    };
    let params_bytes = std::mem::size_of::<QMatmulBf16GpuParams>();
    let mut params_buf = device.alloc_buffer(params_bytes, DType::U32, vec![5])?;
    {
        let slice: &mut [QMatmulBf16GpuParams] = bytemuck::cast_slice_mut(
            params_buf
                .as_mut_slice::<u8>()
                .map_err(|e| MlxError::InvalidArgument(format!("params buf write: {e}")))?,
        );
        slice[0] = gpu_params;
    }

    let num_simdgroups = 2u64;
    let results_per_simdgroup = 4u64;
    let bn = num_simdgroups * results_per_simdgroup; // 8

    let threadgroup_size = metal::MTLSize::new(32, num_simdgroups, 1);
    let threadgroups = metal::MTLSize::new(
        params.m as u64,
        (params.n as u64 + bn - 1) / bn,
        1,
    );

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, input),
            (1, packed_weights),
            (2, scales),
            (3, biases),
            (4, &output),
            (5, &params_buf),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(output)
}

/// Dispatch bf16 quantized matmul with expert offset for MoE inference.
///
/// Indexes into a 3D packed weight tensor `[n_experts, rows, packed_cols]` using
/// byte offsets, eliminating CPU memcpy for expert weight selection.
///
/// # Arguments
///
/// * `encoder`              — The command encoder to record the dispatch into.
/// * `registry`             — Kernel registry (compiles the shader on first call).
/// * `device`               — The Metal device for buffer allocation.
/// * `input`                — bf16 input matrix buffer, shape `[M, K]`.
/// * `packed_weights`       — Full 3D packed weight tensor for all experts.
/// * `scales`               — Full scales buffer for all experts.
/// * `biases`               — Full biases buffer for all experts.
/// * `params`               — Dimensions for this expert's projection (M, K, N).
/// * `expert_offset_bytes`  — Byte offset into `packed_weights` for this expert.
/// * `scales_offset_bytes`  — Byte offset into `scales` for this expert.
/// * `biases_offset_bytes`  — Byte offset into `biases` for this expert.
///
/// # Returns
///
/// A freshly allocated `MlxBuffer` for the output of shape `[M, N]` with dtype `BF16`.
pub fn dispatch_quantized_matmul_simd_bf16_expert(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    packed_weights: &MlxBuffer,
    scales: &MlxBuffer,
    biases: &MlxBuffer,
    params: &QuantizedMatmulParams,
    expert_offset_bytes: u32,
    scales_offset_bytes: u32,
    biases_offset_bytes: u32,
) -> Result<MlxBuffer> {
    // Expert-offset path requires SIMD alignment; no fallback because the
    // scalar kernel doesn't understand 3D expert packing.
    if !can_use_simd_kernel(params) {
        return Err(MlxError::InvalidArgument(
            "dispatch_quantized_matmul_simd_bf16_expert: dimensions do not satisfy SIMD \
             alignment requirements (N%8==0 and K%512==0 for 4-bit, K%256==0 for 8-bit)".into(),
        ));
    }

    if params.bits != 4 && params.bits != 8 {
        return Err(MlxError::InvalidArgument(format!(
            "bf16 expert kernel: unsupported bits value {}; only 4 and 8 are supported",
            params.bits
        )));
    }
    if params.m == 0 || params.k == 0 || params.n == 0 || params.group_size == 0 {
        return Err(MlxError::InvalidArgument(
            "M, K, N, and group_size must all be > 0".into(),
        ));
    }

    // We trust the caller to have sized the 3D buffers correctly.  Validate
    // that the requested slice (offset + one-expert size) fits.
    let expert_weight_bytes = expected_weight_bytes(params.k, params.n, params.bits);
    let expert_scales_bytes = expected_scales_bytes(params.k, params.n, params.group_size);

    if packed_weights.byte_len() < (expert_offset_bytes as usize) + expert_weight_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "packed_weights too small for expert slice: offset={} + size={} > buffer={}",
            expert_offset_bytes, expert_weight_bytes, packed_weights.byte_len()
        )));
    }
    if scales.byte_len() < (scales_offset_bytes as usize) + expert_scales_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "scales buffer too small for expert slice: offset={} + size={} > buffer={}",
            scales_offset_bytes, expert_scales_bytes, scales.byte_len()
        )));
    }
    if biases.byte_len() < (biases_offset_bytes as usize) + expert_scales_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "biases buffer too small for expert slice: offset={} + size={} > buffer={}",
            biases_offset_bytes, expert_scales_bytes, biases.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("quantized_matmul_simd_bf16_expert", device.metal_device())?;

    let output_bytes = (params.m as usize) * (params.n as usize) * DType::BF16.size_of();
    let output = device.alloc_buffer(
        output_bytes,
        DType::BF16,
        vec![params.m as usize, params.n as usize],
    )?;

    let gpu_params = QMatmulBf16GpuParams {
        m: params.m,
        k: params.k,
        n: params.n,
        group_size: params.group_size,
        bits: params.bits,
    };
    let params_bytes = std::mem::size_of::<QMatmulBf16GpuParams>();
    let mut params_buf = device.alloc_buffer(params_bytes, DType::U32, vec![5])?;
    {
        let slice: &mut [QMatmulBf16GpuParams] = bytemuck::cast_slice_mut(
            params_buf
                .as_mut_slice::<u8>()
                .map_err(|e| MlxError::InvalidArgument(format!("params buf write: {e}")))?,
        );
        slice[0] = gpu_params;
    }

    // Pack the three byte-offset values into individual u32 buffers.
    let mut expert_offset_buf = device.alloc_buffer(4, DType::U32, vec![1])?;
    {
        let s: &mut [u32] = expert_offset_buf
            .as_mut_slice()
            .map_err(|e| MlxError::InvalidArgument(format!("expert_offset buf: {e}")))?;
        s[0] = expert_offset_bytes;
    }
    let mut scales_offset_buf = device.alloc_buffer(4, DType::U32, vec![1])?;
    {
        let s: &mut [u32] = scales_offset_buf
            .as_mut_slice()
            .map_err(|e| MlxError::InvalidArgument(format!("scales_offset buf: {e}")))?;
        s[0] = scales_offset_bytes;
    }
    let mut biases_offset_buf = device.alloc_buffer(4, DType::U32, vec![1])?;
    {
        let s: &mut [u32] = biases_offset_buf
            .as_mut_slice()
            .map_err(|e| MlxError::InvalidArgument(format!("biases_offset buf: {e}")))?;
        s[0] = biases_offset_bytes;
    }

    let num_simdgroups = 2u64;
    let results_per_simdgroup = 4u64;
    let bn = num_simdgroups * results_per_simdgroup;

    let threadgroup_size = metal::MTLSize::new(32, num_simdgroups, 1);
    let threadgroups = metal::MTLSize::new(
        params.m as u64,
        (params.n as u64 + bn - 1) / bn,
        1,
    );

    encoder.encode_threadgroups(
        pipeline,
        &[
            (0, input),
            (1, packed_weights),
            (2, scales),
            (3, biases),
            (4, &output),
            (5, &params_buf),
            (6, &expert_offset_buf),
            (7, &scales_offset_buf),
            (8, &biases_offset_buf),
        ],
        threadgroups,
        threadgroup_size,
    );

    Ok(output)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::MlxDevice;

    // ---- f16 / bf16 conversion helpers (no external dependency) ----

    /// Convert an f32 to bfloat16 (bf16) bits.
    /// bf16 is simply the top 16 bits of the IEEE 754 f32 representation.
    fn f32_to_bf16_bits(val: f32) -> u16 {
        (val.to_bits() >> 16) as u16
    }

    /// Convert an f32 to IEEE 754 half-precision (f16) bits.
    /// Uses round-to-nearest-even.
    fn f32_to_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x007F_FFFF;

        if exp == 255 {
            // Inf or NaN
            let m = if mantissa != 0 { 0x0200 } else { 0 };
            return (sign | 0x7C00 | m) as u16;
        }

        // Rebias exponent from f32 (bias=127) to f16 (bias=15).
        let new_exp = exp - 127 + 15;

        if new_exp >= 31 {
            // Overflow → Inf
            return (sign | 0x7C00) as u16;
        }

        if new_exp <= 0 {
            // Denormalized or zero
            if new_exp < -10 {
                return sign as u16; // Too small → zero
            }
            let m = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
            return (sign | m) as u16;
        }

        // Normalized: round-to-nearest-even
        let m = mantissa >> 13;
        let round_bit = (mantissa >> 12) & 1;
        let sticky = if (mantissa & 0xFFF) != 0 { 1u32 } else { 0 };
        let round_up = round_bit & (sticky | m);
        let result = sign | ((new_exp as u32) << 10) | m;
        (result + round_up) as u16
    }

    /// Convert IEEE 754 half-precision (f16) bits to f32.
    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits as u32 & 0x8000) as u32) << 16;
        let exp = (bits >> 10) & 0x1F;
        let mantissa = (bits & 0x03FF) as u32;

        if exp == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign); // +/- zero
            }
            // Denormalized: normalize it.
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let f32_exp = ((e + 127) as u32) << 23;
            let f32_mantissa = m << 13;
            return f32::from_bits(sign | f32_exp | f32_mantissa);
        }

        if exp == 31 {
            let m = if mantissa != 0 { 0x007F_FFFF } else { 0 };
            return f32::from_bits(sign | 0x7F80_0000 | m);
        }

        let f32_exp = ((exp as u32 - 15 + 127) as u32) << 23;
        let f32_mantissa = mantissa << 13;
        f32::from_bits(sign | f32_exp | f32_mantissa)
    }

    // Helper: create an f16 buffer from f32 values.
    #[allow(dead_code)]
    fn f16_buffer(device: &MlxDevice, shape: Vec<usize>, values: &[f32]) -> MlxBuffer {
        let byte_len = values.len() * 2;
        let mut buf = device.alloc_buffer(byte_len, DType::F16, shape).expect("alloc");
        {
            let slice: &mut [u16] = buf.as_mut_slice().expect("as_mut_slice");
            for (i, &v) in values.iter().enumerate() {
                slice[i] = f32_to_f16_bits(v);
            }
        }
        buf
    }

    // Helper: create a bf16 buffer from f32 values (used for scales/biases).
    fn bf16_buffer(device: &MlxDevice, shape: Vec<usize>, values: &[f32]) -> MlxBuffer {
        let byte_len = values.len() * 2;
        let mut buf = device.alloc_buffer(byte_len, DType::BF16, shape).expect("alloc");
        {
            let slice: &mut [u16] = buf.as_mut_slice().expect("as_mut_slice");
            for (i, &v) in values.iter().enumerate() {
                slice[i] = f32_to_bf16_bits(v);
            }
        }
        buf
    }

    // Helper: create an f32 buffer from f32 values (used for input).
    fn f32_buffer(device: &MlxDevice, shape: Vec<usize>, values: &[f32]) -> MlxBuffer {
        let byte_len = values.len() * 4;
        let mut buf = device.alloc_buffer(byte_len, DType::F32, shape).expect("alloc");
        {
            let slice: &mut [f32] = buf.as_mut_slice().expect("as_mut_slice");
            slice.copy_from_slice(values);
        }
        buf
    }

    // Helper: pack 4-bit values into uint32 buffer.
    // `quant_values` is a flat array of quantized unsigned values (0..15),
    // laid out as weight[col][k] (N rows of K values each).
    fn pack_4bit_buffer(device: &MlxDevice, n: usize, k: usize, quant_values: &[u8]) -> MlxBuffer {
        let values_per_pack = 8;
        let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
        let total_packs = n * packs_per_row;
        let byte_len = total_packs * 4;

        let mut buf = device.alloc_buffer(byte_len, DType::U32, vec![n, packs_per_row]).expect("alloc");
        {
            let slice: &mut [u32] = buf.as_mut_slice().expect("as_mut_slice");
            for col in 0..n {
                for pack in 0..packs_per_row {
                    let mut packed: u32 = 0;
                    for i in 0..values_per_pack {
                        let k_idx = pack * values_per_pack + i;
                        if k_idx < k {
                            let val = quant_values[col * k + k_idx] as u32 & 0xF;
                            packed |= val << (4 * i);
                        }
                    }
                    slice[col * packs_per_row + pack] = packed;
                }
            }
        }
        buf
    }

    // Helper: pack 6-bit values into uint32 buffer.
    fn pack_6bit_buffer(device: &MlxDevice, n: usize, k: usize, quant_values: &[u8]) -> MlxBuffer {
        let values_per_pack = 4;
        let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
        let total_packs = n * packs_per_row;
        let byte_len = total_packs * 4;

        let mut buf = device.alloc_buffer(byte_len, DType::U32, vec![n, packs_per_row]).expect("alloc");
        {
            let slice: &mut [u32] = buf.as_mut_slice().expect("as_mut_slice");
            for col in 0..n {
                for pack in 0..packs_per_row {
                    let mut packed: u32 = 0;
                    for i in 0..values_per_pack {
                        let k_idx = pack * values_per_pack + i;
                        if k_idx < k {
                            let val = quant_values[col * k + k_idx] as u32 & 0x3F;
                            packed |= val << (6 * i);
                        }
                    }
                    slice[col * packs_per_row + pack] = packed;
                }
            }
        }
        buf
    }

    // Helper: pack 8-bit values into uint32 buffer.
    // 4 values per uint32 (8 bits each).
    fn pack_8bit_buffer(device: &MlxDevice, n: usize, k: usize, quant_values: &[u8]) -> MlxBuffer {
        let values_per_pack = 4;
        let packs_per_row = (k + values_per_pack - 1) / values_per_pack;
        let total_packs = n * packs_per_row;
        let byte_len = total_packs * 4;

        let mut buf = device.alloc_buffer(byte_len, DType::U32, vec![n, packs_per_row]).expect("alloc");
        {
            let slice: &mut [u32] = buf.as_mut_slice().expect("as_mut_slice");
            for col in 0..n {
                for pack in 0..packs_per_row {
                    let mut packed: u32 = 0;
                    for i in 0..values_per_pack {
                        let k_idx = pack * values_per_pack + i;
                        if k_idx < k {
                            let val = quant_values[col * k + k_idx] as u32 & 0xFF;
                            packed |= val << (8 * i);
                        }
                    }
                    slice[col * packs_per_row + pack] = packed;
                }
            }
        }
        buf
    }

    // Helper: read f16 buffer back as f32.
    #[allow(dead_code)]
    fn read_f16(buf: &MlxBuffer) -> Vec<f32> {
        let slice: &[u16] = buf.as_slice().expect("as_slice");
        slice.iter().map(|&bits| f16_bits_to_f32(bits)).collect()
    }

    // Helper: read f32 output buffer.
    fn read_f32(buf: &MlxBuffer) -> Vec<f32> {
        let slice: &[f32] = buf.as_slice().expect("as_slice");
        slice.to_vec()
    }

    /// Test 4-bit quantized matmul with a small known example.
    ///
    /// input = [[1.0, 2.0, 3.0, 4.0]]  (M=1, K=4)
    /// weight quantized values (N=2, K=4): [[1, 2, 3, 4], [5, 6, 7, 8]]
    /// scales = [[0.1], [0.2]]  (1 group per row since group_size=64 > K=4)
    /// biases = [[0.0], [0.0]]
    ///
    /// dequant weight row 0: [0.1, 0.2, 0.3, 0.4]
    /// dequant weight row 1: [1.0, 1.2, 1.4, 1.6]
    ///
    /// output[0][0] = 1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 4.0*0.4 = 0.1+0.4+0.9+1.6 = 3.0
    /// output[0][1] = 1.0*1.0 + 2.0*1.2 + 3.0*1.4 + 4.0*1.6 = 1.0+2.4+4.2+6.4 = 14.0
    #[test]
    fn test_4bit_matmul_small_known() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        let m = 1u32;
        let k = 4u32;
        let n = 2u32;
        let group_size = 64u32;
        let bits = 4u32;

        let input = f32_buffer(&device, vec![m as usize, k as usize], &[1.0, 2.0, 3.0, 4.0]);

        // Quantized weight values (unsigned): row0=[1,2,3,4], row1=[5,6,7,8]
        let quant_w: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let weight = pack_4bit_buffer(&device, n as usize, k as usize, &quant_w);

        let scales = bf16_buffer(&device, vec![n as usize, 1], &[0.1, 0.2]);
        let biases = bf16_buffer(&device, vec![n as usize, 1], &[0.0, 0.0]);

        let params = QuantizedMatmulParams { m, k, n, group_size, bits };

        let output = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        ).expect("quantized_matmul");

        encoder.commit_and_wait().expect("commit");

        let result = read_f32(&output);
        assert_eq!(result.len(), 2);

        // Tolerance: bf16 precision is ~1e-2 for these magnitudes.
        let tol = 1e-1; // bf16 has limited precision, be generous for this test
        assert!(
            (result[0] - 3.0).abs() < tol,
            "output[0]={}, expected ~3.0", result[0]
        );
        assert!(
            (result[1] - 14.0).abs() < tol,
            "output[1]={}, expected ~14.0", result[1]
        );
    }

    /// Test 6-bit quantized matmul with a small known example.
    #[test]
    fn test_6bit_matmul_small_known() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        let m = 1u32;
        let k = 4u32;
        let n = 2u32;
        let group_size = 64u32;
        let bits = 6u32;

        let input = f32_buffer(&device, vec![m as usize, k as usize], &[1.0, 2.0, 3.0, 4.0]);

        // 6-bit quantized weight values (0..63): row0=[1,2,3,4], row1=[10,20,30,40]
        let quant_w: Vec<u8> = vec![1, 2, 3, 4, 10, 20, 30, 40];
        let weight = pack_6bit_buffer(&device, n as usize, k as usize, &quant_w);

        let scales = bf16_buffer(&device, vec![n as usize, 1], &[0.1, 0.05]);
        let biases = bf16_buffer(&device, vec![n as usize, 1], &[0.0, 0.0]);

        let params = QuantizedMatmulParams { m, k, n, group_size, bits };

        let output = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        ).expect("quantized_matmul");

        encoder.commit_and_wait().expect("commit");

        let result = read_f32(&output);
        assert_eq!(result.len(), 2);

        // dequant row 0: [0.1, 0.2, 0.3, 0.4]
        // output[0] = 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 = 3.0
        // dequant row 1: [0.5, 1.0, 1.5, 2.0]
        // output[1] = 1*0.5 + 2*1.0 + 3*1.5 + 4*2.0 = 0.5+2.0+4.5+8.0 = 15.0
        let tol = 1e-1;
        assert!(
            (result[0] - 3.0).abs() < tol,
            "output[0]={}, expected ~3.0", result[0]
        );
        assert!(
            (result[1] - 15.0).abs() < tol,
            "output[1]={}, expected ~15.0", result[1]
        );
    }

    /// Test with non-zero biases.
    #[test]
    fn test_4bit_matmul_with_bias() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        let m = 1u32;
        let k = 4u32;
        let n = 1u32;
        let group_size = 64u32;
        let bits = 4u32;

        let input = f32_buffer(&device, vec![1, 4], &[1.0, 1.0, 1.0, 1.0]);

        // quant values all 0 → dequant = scale*0 + bias = bias
        let quant_w: Vec<u8> = vec![0, 0, 0, 0];
        let weight = pack_4bit_buffer(&device, 1, 4, &quant_w);

        let scales = bf16_buffer(&device, vec![1, 1], &[1.0]);
        let biases = bf16_buffer(&device, vec![1, 1], &[0.5]);

        let params = QuantizedMatmulParams { m, k, n, group_size, bits };

        let output = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        ).expect("quantized_matmul");

        encoder.commit_and_wait().expect("commit");

        let result = read_f32(&output);
        // Each weight dequantized to 0.5, dot with [1,1,1,1] = 2.0
        let tol = 1e-2;
        assert!(
            (result[0] - 2.0).abs() < tol,
            "output[0]={}, expected ~2.0", result[0]
        );
    }

    /// Test batch (M > 1).
    #[test]
    fn test_4bit_batch_matmul() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        let m = 2u32;
        let k = 4u32;
        let n = 1u32;
        let group_size = 64u32;
        let bits = 4u32;

        // Two input rows: [1,0,0,0] and [0,1,0,0]
        let input = f32_buffer(&device, vec![2, 4], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

        // quant weight row: [2, 4, 6, 8]
        let quant_w: Vec<u8> = vec![2, 4, 6, 8];
        let weight = pack_4bit_buffer(&device, 1, 4, &quant_w);

        let scales = bf16_buffer(&device, vec![1, 1], &[0.5]);
        let biases = bf16_buffer(&device, vec![1, 1], &[0.0]);

        let params = QuantizedMatmulParams { m, k, n, group_size, bits };

        let output = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        ).expect("quantized_matmul");

        encoder.commit_and_wait().expect("commit");

        let result = read_f32(&output);
        assert_eq!(result.len(), 2);

        // dequant: [1.0, 2.0, 3.0, 4.0]
        // row0: 1.0*1.0 = 1.0
        // row1: 1.0*2.0 = 2.0
        let tol = 1e-2;
        assert!((result[0] - 1.0).abs() < tol, "row0={}, expected 1.0", result[0]);
        assert!((result[1] - 2.0).abs() < tol, "row1={}, expected 2.0", result[1]);
    }

    /// Test invalid bits returns error.
    #[test]
    fn test_invalid_bits_returns_error() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        let input = f32_buffer(&device, vec![1, 4], &[1.0; 4]);
        // Minimal buffers — validation should fail before size checks matter.
        let weight = device.alloc_buffer(4, DType::U32, vec![1]).expect("alloc");
        let scales = bf16_buffer(&device, vec![1], &[1.0]);
        let biases = bf16_buffer(&device, vec![1], &[0.0]);

        let params = QuantizedMatmulParams {
            m: 1, k: 4, n: 1, group_size: 64, bits: 5,
        };

        let result = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        );

        assert!(result.is_err());
        match result {
            Err(MlxError::InvalidArgument(msg)) => {
                assert!(msg.contains("bits"), "Error should mention bits: {msg}");
            }
            other => panic!("Expected InvalidArgument, got {:?}", other),
        }
    }

    /// Test mismatched dimensions returns error.
    #[test]
    fn test_mismatched_dimensions_returns_error() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        // Input is 1x4 but we'll claim K=128 in params.
        let input = f32_buffer(&device, vec![1, 4], &[1.0; 4]);
        let weight = device.alloc_buffer(4, DType::U32, vec![1]).expect("alloc");
        let scales = bf16_buffer(&device, vec![1], &[1.0]);
        let biases = bf16_buffer(&device, vec![1], &[0.0]);

        let params = QuantizedMatmulParams {
            m: 1, k: 128, n: 1, group_size: 64, bits: 4,
        };

        let result = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        );

        assert!(result.is_err());
        match result {
            Err(MlxError::InvalidArgument(msg)) => {
                assert!(msg.contains("Input buffer too small"), "msg: {msg}");
            }
            other => panic!("Expected InvalidArgument for input size, got {:?}", other),
        }
    }

    /// Test 8-bit quantized matmul with a small known example.
    ///
    /// input = [[1.0, 2.0, 3.0, 4.0]]  (M=1, K=4)
    /// weight quantized values (N=2, K=4): [[10, 20, 30, 40], [50, 60, 70, 80]]
    /// scales = [[0.01], [0.02]]  (1 group per row since group_size=64 > K=4)
    /// biases = [[0.0], [0.0]]
    ///
    /// dequant weight row 0: [0.1, 0.2, 0.3, 0.4]
    /// dequant weight row 1: [1.0, 1.2, 1.4, 1.6]
    ///
    /// output[0][0] = 1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 4.0*0.4 = 3.0
    /// output[0][1] = 1.0*1.0 + 2.0*1.2 + 3.0*1.4 + 4.0*1.6 = 14.0
    #[test]
    fn test_8bit_matmul_small_known() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        let m = 1u32;
        let k = 4u32;
        let n = 2u32;
        let group_size = 64u32;
        let bits = 8u32;

        let input = f32_buffer(&device, vec![m as usize, k as usize], &[1.0, 2.0, 3.0, 4.0]);

        // 8-bit quantized weight values (0..255): row0=[10,20,30,40], row1=[50,60,70,80]
        let quant_w: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let weight = pack_8bit_buffer(&device, n as usize, k as usize, &quant_w);

        let scales = bf16_buffer(&device, vec![n as usize, 1], &[0.01, 0.02]);
        let biases = bf16_buffer(&device, vec![n as usize, 1], &[0.0, 0.0]);

        let params = QuantizedMatmulParams { m, k, n, group_size, bits };

        let output = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        ).expect("quantized_matmul");

        encoder.commit_and_wait().expect("commit");

        let result = read_f32(&output);
        assert_eq!(result.len(), 2);

        // Tolerance: bf16 precision is ~1e-2 for these magnitudes.
        let tol = 1e-1;
        assert!(
            (result[0] - 3.0).abs() < tol,
            "output[0]={}, expected ~3.0", result[0]
        );
        assert!(
            (result[1] - 14.0).abs() < tol,
            "output[1]={}, expected ~14.0", result[1]
        );
    }

    /// Test 8-bit with non-zero biases.
    #[test]
    fn test_8bit_matmul_with_bias() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        let m = 1u32;
        let k = 4u32;
        let n = 1u32;
        let group_size = 64u32;
        let bits = 8u32;

        let input = f32_buffer(&device, vec![1, 4], &[1.0, 1.0, 1.0, 1.0]);

        // quant values all 0 -> dequant = scale*0 + bias = bias
        let quant_w: Vec<u8> = vec![0, 0, 0, 0];
        let weight = pack_8bit_buffer(&device, 1, 4, &quant_w);

        let scales = bf16_buffer(&device, vec![1, 1], &[1.0]);
        let biases = bf16_buffer(&device, vec![1, 1], &[0.5]);

        let params = QuantizedMatmulParams { m, k, n, group_size, bits };

        let output = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        ).expect("quantized_matmul");

        encoder.commit_and_wait().expect("commit");

        let result = read_f32(&output);
        // Each weight dequantized to 0.5, dot with [1,1,1,1] = 2.0
        let tol = 1e-2;
        assert!(
            (result[0] - 2.0).abs() < tol,
            "output[0]={}, expected ~2.0", result[0]
        );
    }

    /// Test with multiple groups along K (K > group_size).
    #[test]
    fn test_4bit_multiple_groups() {
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();
        let mut encoder = device.command_encoder().expect("encoder");

        // Use K=8 with group_size=4 → 2 groups per column.
        let m = 1u32;
        let k = 8u32;
        let n = 1u32;
        let group_size = 4u32;
        let bits = 4u32;

        let input = f32_buffer(&device, vec![1, 8], &[1.0; 8]);

        // quant values: [1,1,1,1, 2,2,2,2]
        let quant_w: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let weight = pack_4bit_buffer(&device, 1, 8, &quant_w);

        // 2 groups: scale=[0.5, 1.0], bias=[0.0, 0.0]
        let scales = bf16_buffer(&device, vec![1, 2], &[0.5, 1.0]);
        let biases = bf16_buffer(&device, vec![1, 2], &[0.0, 0.0]);

        let params = QuantizedMatmulParams { m, k, n, group_size, bits };

        let output = quantized_matmul(
            &mut encoder, &mut registry, &device,
            &input, &weight, &scales, &biases, &params,
        ).expect("quantized_matmul");

        encoder.commit_and_wait().expect("commit");

        let result = read_f32(&output);
        // Group 0: dequant=[0.5,0.5,0.5,0.5], sum = 4*0.5 = 2.0
        // Group 1: dequant=[2.0,2.0,2.0,2.0], sum = 4*2.0 = 8.0
        // Total = 10.0
        let tol = 1e-1;
        assert!(
            (result[0] - 10.0).abs() < tol,
            "output[0]={}, expected ~10.0", result[0]
        );
    }
}
