//! GPU-accelerated 2D matrix transpose.
//!
//! Transposes a 2D matrix `[rows, cols]` to `[cols, rows]`.
//! Supports F32 and F16 dtypes.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL-compatible params struct for 2D transpose.
///
/// Must match `TransposeParams` in `elementwise.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTransposeParams {
    rows: u32,
    cols: u32,
}

/// Encode a 2D matrix transpose: `output[col, row] = input[row, col]`.
///
/// # Buffer expectations
///
/// * `input`  — `[rows, cols]` in the given dtype
/// * `output` — `[cols, rows]` in the given dtype (must be pre-allocated)
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// * `rows` or `cols` is zero
/// * `dtype` is not F32 or F16
/// * Buffers are too small
#[allow(clippy::too_many_arguments)]
pub fn transpose_2d(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<()> {
    if rows == 0 {
        return Err(MlxError::InvalidArgument(
            "transpose_2d: rows must be > 0".into(),
        ));
    }
    if cols == 0 {
        return Err(MlxError::InvalidArgument(
            "transpose_2d: cols must be > 0".into(),
        ));
    }

    let kernel_name = match dtype {
        DType::F32 => "transpose_2d_f32",
        DType::F16 => "transpose_2d_f16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "transpose_2d: unsupported dtype {dtype}"
            )));
        }
    };

    let elem_bytes = rows * cols * dtype.size_of();
    if input.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "transpose_2d: input buffer too small: need {} bytes, have {}",
            elem_bytes,
            input.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "transpose_2d: output buffer too small: need {} bytes, have {}",
            elem_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    let gpu_params = GpuTransposeParams {
        rows: rows as u32,
        cols: cols as u32,
    };

    // 2D grid: (cols, rows)
    let grid = MTLSize::new(cols as u64, rows as u64, 1);
    let tg = MTLSize::new(
        std::cmp::min(16, cols as u64),
        std::cmp::min(16, rows as u64),
        1,
    );

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}

/// MSL-compatible params struct for 3D permute [A, B, C] -> [B, A, C].
///
/// Must match `Permute021Params` in `elementwise.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPermute021Params {
    dim_a: u32,
    dim_b: u32,
    dim_c: u32,
}

/// Encode a 3D permutation: `[A, B, C] -> [B, A, C]` (bf16).
///
/// This is used to convert between `[seq_len, n_heads, head_dim]` and
/// `[n_heads, seq_len, head_dim]` layouts.
///
/// # Buffer expectations
///
/// * `input`  — `[dim_a, dim_b, dim_c]` in bf16
/// * `output` — `[dim_b, dim_a, dim_c]` in bf16 (must be pre-allocated)
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if any dimension is zero or buffers
/// are too small.
pub fn permute_021_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    dim_a: usize,
    dim_b: usize,
    dim_c: usize,
) -> Result<()> {
    if dim_a == 0 || dim_b == 0 || dim_c == 0 {
        return Err(MlxError::InvalidArgument(
            "permute_021_f32: all dimensions must be > 0".into(),
        ));
    }

    let total_elements = dim_a * dim_b * dim_c;
    let elem_bytes = total_elements * 4; // f32 = 4 bytes
    if input.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "permute_021_f32: input buffer too small: need {} bytes, have {}",
            elem_bytes,
            input.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "permute_021_f32: output buffer too small: need {} bytes, have {}",
            elem_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("permute_021_f32", device)?;

    let gpu_params = GpuPermute021Params {
        dim_a: dim_a as u32,
        dim_b: dim_b as u32,
        dim_c: dim_c as u32,
    };

    let grid = MTLSize::new(dim_c as u64, dim_b as u64, dim_a as u64);
    let tg = MTLSize::new(
        std::cmp::min(64, dim_c as u64),
        std::cmp::min(4, dim_b as u64),
        std::cmp::min(4, dim_a as u64),
    );

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}

/// Swap the last two axes of a 3D bf16 tensor: [A, B, C] -> [A, C, B].
///
/// Used by hf2q's non-flash-attention prefill path to transpose V from
/// its natural `[nkv, seq, hd]` post-RoPE layout to the `[nkv, hd, seq]`
/// layout the `scores @ V` matmul consumes (the contract dim of our
/// tensor-mm kernel is the inner-most axis of src0).
///
/// One dispatch covers all A batches; each thread copies a single bf16.
pub fn transpose_last2_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    dim_a: usize,
    dim_b: usize,
    dim_c: usize,
) -> Result<()> {
    if dim_a == 0 || dim_b == 0 || dim_c == 0 {
        return Err(MlxError::InvalidArgument(
            "transpose_last2_bf16: all dimensions must be > 0".into(),
        ));
    }

    let total_elements = dim_a * dim_b * dim_c;
    let elem_bytes = total_elements * 2;
    if input.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "transpose_last2_bf16: input buffer too small: need {} bytes, have {}",
            elem_bytes, input.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "transpose_last2_bf16: output buffer too small: need {} bytes, have {}",
            elem_bytes, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("transpose_last2_bf16", device)?;

    let gpu_params = GpuPermute021Params {
        dim_a: dim_a as u32,
        dim_b: dim_b as u32,
        dim_c: dim_c as u32,
    };

    // Grid: (dim_b, dim_c, dim_a).  Kernel maps (gid.x, gid.y, gid.z) →
    // (b, c, a); see shaders/elementwise.metal::transpose_last2_bf16.
    let grid = MTLSize::new(dim_b as u64, dim_c as u64, dim_a as u64);
    let tg = MTLSize::new(
        std::cmp::min(16, dim_b as u64),
        std::cmp::min(16, dim_c as u64),
        std::cmp::min(4, dim_a as u64),
    );

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}

pub fn permute_021_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    dim_a: usize,
    dim_b: usize,
    dim_c: usize,
) -> Result<()> {
    if dim_a == 0 || dim_b == 0 || dim_c == 0 {
        return Err(MlxError::InvalidArgument(
            "permute_021_bf16: all dimensions must be > 0".into(),
        ));
    }

    let total_elements = dim_a * dim_b * dim_c;
    let elem_bytes = total_elements * 2; // bf16 = 2 bytes
    if input.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "permute_021_bf16: input buffer too small: need {} bytes, have {}",
            elem_bytes,
            input.byte_len()
        )));
    }
    if output.byte_len() < elem_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "permute_021_bf16: output buffer too small: need {} bytes, have {}",
            elem_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("permute_021_bf16", device)?;

    let gpu_params = GpuPermute021Params {
        dim_a: dim_a as u32,
        dim_b: dim_b as u32,
        dim_c: dim_c as u32,
    };

    // 3D grid: (dim_c, dim_b, dim_a), each thread copies one element
    let grid = MTLSize::new(dim_c as u64, dim_b as u64, dim_a as u64);
    let tg = MTLSize::new(
        std::cmp::min(64, dim_c as u64),
        std::cmp::min(4, dim_b as u64),
        std::cmp::min(4, dim_a as u64),
    );

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}

/// Fused permute_021 + bf16→f32 cast.  Replaces the two-pass sequence
/// `permute_021_bf16(bf16 → bf16) ; cast_bf16_to_f32(bf16 → f32)` with a
/// single dispatch that reads bf16 in [A, B, C] order and writes f32 in
/// [B, A, C] order, halving the global-memory traffic on the post-FA SDPA
/// output buffer.  Wave P4.10.
pub fn permute_021_bf16_to_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    dim_a: usize,
    dim_b: usize,
    dim_c: usize,
) -> Result<()> {
    if dim_a == 0 || dim_b == 0 || dim_c == 0 {
        return Err(MlxError::InvalidArgument(
            "permute_021_bf16_to_f32: all dimensions must be > 0".into(),
        ));
    }

    let total_elements = dim_a * dim_b * dim_c;
    let in_bytes = total_elements * 2; // bf16
    let out_bytes = total_elements * 4; // f32
    if input.byte_len() < in_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "permute_021_bf16_to_f32: input buffer too small: need {} bytes, have {}",
            in_bytes, input.byte_len()
        )));
    }
    if output.byte_len() < out_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "permute_021_bf16_to_f32: output buffer too small: need {} bytes, have {}",
            out_bytes, output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("permute_021_bf16_to_f32", device)?;

    let gpu_params = GpuPermute021Params {
        dim_a: dim_a as u32,
        dim_b: dim_b as u32,
        dim_c: dim_c as u32,
    };

    let grid = MTLSize::new(dim_c as u64, dim_b as u64, dim_a as u64);
    let tg = MTLSize::new(
        std::cmp::min(64, dim_c as u64),
        std::cmp::min(4, dim_b as u64),
        std::cmp::min(4, dim_a as u64),
    );

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(input)),
            (1, KernelArg::Buffer(output)),
            (2, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}
