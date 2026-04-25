//! Dense bf16 × f32 → f32 GEMV (matrix-vector multiply) for M == 1 decode.
//!
//! Port of llama.cpp's `kernel_mul_mv_bf16_f32_4` kernel (bfloat4 vectorized
//! path).  Use this instead of [`dense_mm_bf16::dense_matmul_bf16_f32_tensor`]
//! when the number of input rows M == 1, i.e. single-token decode.
//!
//! # Layout
//!
//! | Tensor | Shape               | Dtype  | Note |
//! |--------|---------------------|--------|------|
//! | src0   | `[src0_batch, N, K]` | BF16  | weight matrix rows |
//! | src1   | `[src1_batch, M, K]` | F32   | input vectors |
//! | dst    | `[src1_batch, M, N]` | F32   | output vectors |
//!
//! This is the same contract as [`dense_mm_bf16::DenseMmBf16F32Params`].
//!
//! Derived from llama.cpp (MIT).  See `src/shaders/dense_gemv_bf16.metal`.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::dense_mm_bf16::DenseMmBf16F32Params;

/// GEMV kernel source (compiled lazily on first call).
pub static DENSE_GEMV_BF16_SHADER_SOURCE: &str =
    include_str!("../shaders/dense_gemv_bf16.metal");

/// Register the `hf2q_dense_gemv_bf16_f32_4` pipeline with a kernel registry.
///
/// Call this once during model init so the shader is compiled before the hot
/// decode path.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source(
        "hf2q_dense_gemv_bf16_f32_4",
        DENSE_GEMV_BF16_SHADER_SOURCE,
    );
}

/// GPU-side params struct; matches `DenseGemvBf16Params` in the Metal shader
/// byte-for-byte, which in turn matches `ggml_metal_kargs_mul_mv`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseGemvBf16GpuParams {
    ne00: i32,         // K
    ne01: i32,         // N
    ne02: i32,         // src0_batch
    _pad0: u32,        // align ne02 → nb00 (4 bytes, uint64_t needs 8-byte alignment)
    nb00: u64,         // sizeof(bfloat) = 2  (unused by kernel, kept for layout)
    nb01: u64,         // K * 2  (src0 row stride in bytes)
    nb02: u64,         // N * K * 2  (src0 batch stride in bytes)
    nb03: u64,         // 0  (super-batch unused)
    ne10: i32,         // K  (unused by kernel, kept for layout)
    ne11: i32,         // M
    ne12: i32,         // src1_batch
    _pad1: u32,        // align ne12 → nb10 (4 bytes, uint64_t needs 8-byte alignment)
    nb10: u64,         // sizeof(float) = 4  (unused by kernel)
    nb11: u64,         // K * 4  (src1 row stride in bytes)
    nb12: u64,         // M * K * 4  (src1 batch stride in bytes)
    nb13: u64,         // 0
    ne0:  i32,         // N
    ne1:  i32,         // M
    nr0:  i32,         // 2  (NR0 — weight rows per threadgroup)
    r2:   i16,         // src1_batch / src0_batch
    r3:   i16,         // 1
    // Total: 112 bytes, 8-byte aligned — no trailing pad needed.
}

/// Dense bf16 × f32 → f32 GEMV — optimized for M = 1 (single-token decode).
///
/// Accepts the same [`DenseMmBf16F32Params`] struct as
/// `dense_matmul_bf16_f32_tensor` so callers can switch between the two
/// paths without API changes.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` for any shape, dtype, or buffer-size
/// mismatch.
pub fn dense_gemv_bf16_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src0: &MlxBuffer,    // BF16 weight [src0_batch, N, K]
    src1: &MlxBuffer,    // F32  input  [src1_batch, M, K]
    dst: &mut MlxBuffer, // F32  output [src1_batch, M, N]
    params: &DenseMmBf16F32Params,
) -> Result<()> {
    let bf16_sz = DType::BF16.size_of();
    let f32_sz  = DType::F32.size_of();

    if params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_gemv_bf16_f32: M, N, K must all be > 0".into(),
        ));
    }
    if params.src0_batch == 0 || params.src1_batch == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_gemv_bf16_f32: batch counts must be > 0".into(),
        ));
    }
    if params.src1_batch % params.src0_batch != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_gemv_bf16_f32: src1_batch ({}) must be a multiple of \
             src0_batch ({}) for GQA broadcast",
            params.src1_batch, params.src0_batch
        )));
    }

    let expected_src0 = params.src0_batch as usize * params.n as usize * params.k as usize * bf16_sz;
    if src0.byte_len() < expected_src0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_gemv_bf16_f32: src0 too small: expected {expected_src0} bytes for \
             [{}×{}×{}] bf16, got {}",
            params.src0_batch, params.n, params.k, src0.byte_len()
        )));
    }
    let expected_src1 = params.src1_batch as usize * params.m as usize * params.k as usize * f32_sz;
    if src1.byte_len() < expected_src1 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_gemv_bf16_f32: src1 too small: expected {expected_src1} bytes for \
             [{}×{}×{}] f32, got {}",
            params.src1_batch, params.m, params.k, src1.byte_len()
        )));
    }
    let expected_dst = params.src1_batch as usize * params.m as usize * params.n as usize * f32_sz;
    if dst.byte_len() < expected_dst {
        return Err(MlxError::InvalidArgument(format!(
            "dense_gemv_bf16_f32: dst too small: expected {expected_dst} bytes for \
             [{}×{}×{}] f32, got {}",
            params.src1_batch, params.m, params.n, dst.byte_len()
        )));
    }

    let nb01 = params.k as u64 * bf16_sz as u64;            // src0 row stride
    let nb02 = params.n as u64 * nb01;                      // src0 batch stride
    let nb11 = params.k as u64 * f32_sz as u64;             // src1 row stride
    let nb12 = params.m as u64 * nb11;                      // src1 batch stride
    let r2   = (params.src1_batch / params.src0_batch) as i16;

    // NSG = min(4, ceil(K / 128))
    let nsg: u64 = ((params.k as u64 + 127) / 128).min(4);

    const NR0: u64 = 2; // weight rows per threadgroup

    let gpu_params = DenseGemvBf16GpuParams {
        ne00: params.k as i32,
        ne01: params.n as i32,
        ne02: params.src0_batch as i32,
        _pad0: 0,
        nb00: bf16_sz as u64,
        nb01,
        nb02,
        nb03: 0,
        ne10: params.k as i32,
        ne11: params.m as i32,
        ne12: params.src1_batch as i32,
        _pad1: 0,
        nb10: f32_sz as u64,
        nb11,
        nb12,
        nb13: 0,
        ne0:  params.n as i32,
        ne1:  params.m as i32,
        nr0:  NR0 as i32,
        r2,
        r3:   1,
    };

    let pipeline = registry.get_pipeline("hf2q_dense_gemv_bf16_f32_4", device.metal_device())?;

    // Grid: (ceil(N/NR0), M, src1_batch).
    // Threadgroup size: (32, NSG, 1) = 32 lanes × NSG simdgroups.
    // Shared memory: NR0 × 32 × sizeof(float) = 2 × 32 × 4 = 256 bytes.
    let threadgroups = metal::MTLSize::new(
        (params.n as u64 + NR0 - 1) / NR0,
        params.m as u64,
        params.src1_batch as u64,
    );
    let threadgroup_size = metal::MTLSize::new(32, nsg, 1);
    let shmem_bytes: u64 = NR0 * 32 * f32_sz as u64; // 256

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(src0)),
            (2, KernelArg::Buffer(src1)),
            (3, KernelArg::Buffer(dst)),
        ],
        &[(0, shmem_bytes)],
        threadgroups,
        threadgroup_size,
    );

    Ok(())
}
