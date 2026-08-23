//! Dense bf16 × f32 → f32 row-reduction GEMV for short-row workloads.
//!
//! Each threadgroup computes one input row using vectorized BF16 weight loads.
//! The route is optimized for M=1 decode and remains exact for larger M; the
//! frozen BF16 route plan decides when it is faster than tiled-four execution.
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
//! See `src/shaders/dense_gemv_bf16.metal` for the kernel source and its
//! attribution.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::Result;
use crate::kernel_registry::KernelRegistry;
use crate::ops::dense_bf16_contract::{validate_dense_bf16_contract, DenseBf16Contract};
use crate::ops::dense_mm_bf16::DenseMmBf16F32Params;

/// GEMV kernel source (compiled lazily on first call).
pub static DENSE_GEMV_BF16_SHADER_SOURCE: &str = include_str!("../shaders/dense_gemv_bf16.metal");

/// Register the BF16 GEMV pipelines with a kernel registry.
///
/// Call this once during model init so the shader is compiled before the hot
/// decode path.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("hf2q_dense_gemv_bf16_f32_4", DENSE_GEMV_BF16_SHADER_SOURCE);
    registry.register_source(
        "hf2q_dense_gemv_bf16_f32_r1_4",
        DENSE_GEMV_BF16_SHADER_SOURCE,
    );
}

/// GPU-side params struct; matches `DenseGemvBf16Params` in the Metal shader
/// byte-for-byte, which in turn matches `ggml_metal_kargs_mul_mv`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseGemvBf16GpuParams {
    ne00: i32,  // K
    ne01: i32,  // N
    ne02: i32,  // src0_batch
    _pad0: u32, // align ne02 → nb00 (4 bytes, uint64_t needs 8-byte alignment)
    nb00: u64,  // sizeof(bfloat) = 2  (unused by kernel, kept for layout)
    nb01: u64,  // K * 2  (src0 row stride in bytes)
    nb02: u64,  // N * K * 2  (src0 batch stride in bytes)
    nb03: u64,  // 0  (super-batch unused)
    ne10: i32,  // K  (unused by kernel, kept for layout)
    ne11: i32,  // M
    ne12: i32,  // src1_batch
    _pad1: u32, // align ne12 → nb10 (4 bytes, uint64_t needs 8-byte alignment)
    nb10: u64,  // sizeof(float) = 4  (unused by kernel)
    nb11: u64,  // K * 4  (src1 row stride in bytes)
    nb12: u64,  // M * K * 4  (src1 batch stride in bytes)
    nb13: u64,  // 0
    ne0: i32,   // N
    ne1: i32,   // M
    nr0: i32,   // 2  (NR0 — weight rows per threadgroup)
    r2: i16,    // src1_batch / src0_batch
    r3: i16,    // 1
                // Total: 112 bytes, 8-byte aligned — no trailing pad needed.
}

fn validate_and_build_params(
    operation: &str,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<DenseGemvBf16GpuParams> {
    validate_dense_bf16_contract(
        operation,
        DenseBf16Contract::RowReduction,
        src0,
        src1,
        dst,
        params,
    )?;
    let broadcast = params.src1_batch / params.src0_batch;
    let bf16_size = DType::BF16.size_of() as u64;
    let f32_size = DType::F32.size_of() as u64;
    let nb01 = params.k as u64 * bf16_size;
    let nb02 = params.n as u64 * nb01;
    let nb11 = params.k as u64 * f32_size;
    let nb12 = params.m as u64 * nb11;
    Ok(DenseGemvBf16GpuParams {
        ne00: params.k as i32,
        ne01: params.n as i32,
        ne02: params.src0_batch as i32,
        _pad0: 0,
        nb00: bf16_size,
        nb01,
        nb02,
        nb03: 0,
        ne10: params.k as i32,
        ne11: params.m as i32,
        ne12: params.src1_batch as i32,
        _pad1: 0,
        nb10: f32_size,
        nb11,
        nb12,
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        nr0: 2,
        r2: broadcast as i16,
        r3: 1,
    })
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
    src0: &MlxBuffer, // BF16 weight [src0_batch, N, K]
    src1: &MlxBuffer, // F32  input  [src1_batch, M, K]
    dst: &MlxBuffer,  // F32  output [src1_batch, M, N]
    params: &DenseMmBf16F32Params,
) -> Result<()> {
    let gpu_params = validate_and_build_params("dense_gemv_bf16_f32", src0, src1, dst, params)?;

    // NSG = min(4, ceil(K / 128))
    let nsg: u64 = ((params.k as u64 + 127) / 128).min(4);

    const NR0: u64 = 2; // weight rows per threadgroup

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
    let shmem_bytes: u64 = NR0 * 32 * DType::F32.size_of() as u64; // 256

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

/// Dense BF16-weight × F32-input GEMV in four-row tiles.
///
/// The kernel loads each weight vector once and evaluates up to four input
/// rows per tile with the same F32 reduction order as
/// [`dense_gemv_bf16_f32`]. It is intended for short verifier batches where a
/// 32-row matrix tile wastes most of its work. Callers remain responsible for
/// selecting it only for shapes where measured latency is lower than the row
/// or tensor path.
///
/// # Errors
///
/// Returns [`MlxError::InvalidArgument`] for any shape, dtype, alignment, or
/// logical buffer-size mismatch.
pub fn dense_gemv_bf16_f32_tiled4(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<()> {
    let gpu_params =
        validate_and_build_params("dense_gemv_bf16_f32_tiled4", src0, src1, dst, params)?;
    let pipeline = registry.get_pipeline("hf2q_dense_gemv_bf16_f32_r1_4", device.metal_device())?;
    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(src0)),
            (2, KernelArg::Buffer(src1)),
            (3, KernelArg::Buffer(dst)),
        ],
        &[(0, 2 * 4 * 32 * DType::F32.size_of() as u64)],
        metal::MTLSize::new(
            (params.n as u64 + 1) / 2,
            (params.m as u64 + 3) / 4,
            params.src1_batch as u64,
        ),
        metal::MTLSize::new(32, 4, 1),
    );
    Ok(())
}
