//! Dense f16 × f32 → f32 matmul using Apple M3+ tensor cores
//! (`mpp::tensor_ops::matmul2d`).
//!
//! F16-staging sibling of [`crate::ops::dense_mm_bf16`].  Used by hf2q's
//! gemma4v ViT precision-parity path (ADR-005 Phase 2c iter-128): every
//! mmproj weight is stored F16 in GGUF, peer's
//! `kernel_mul_mm_f16_f32` (`/opt/llama.cpp/ggml/src/ggml-metal/
//! ggml-metal.metal:10099`) stages BOTH A and B as `half` in shmem and
//! computes on `simdgroup_half8x8`.  Pre-iter-128, hf2q dequantized
//! F16→F32 at load and re-cast F32→BF16 at the matmul, costing 8× the
//! peer's per-element rounding budget per element — visible in the
//! gemma4v ViT cascade as a 1.16×/block compound (block_26 max_abs 733
//! vs peer ~25).  This kernel matches peer precision exactly.
//!
//! Computes `dst[b, m, n] = sum_k src0[b/r2, n, k] * src1[b, m, k]`
//! across all `b` in `[0, src1_batch)`.  Implements llama.cpp's
//! `kernel_mul_mm_f16_f32` template instantiation
//! (`ggml-metal.metal:10099`) on the `GGML_METAL_HAS_TENSOR` branch.
//!
//! Derived from llama.cpp (MIT).  See `src/shaders/dense_mm_f16_tensor.metal`.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Host-side parameters for [`dense_matmul_f16_f32_tensor`].
///
/// Field meanings match [`crate::ops::dense_mm_bf16::DenseMmBf16F32Params`]:
/// `m`/`n`/`k` are the matmul dims; `src0_batch` and `src1_batch`
/// implement GQA-style head broadcast where every src0 slice is shared
/// across `src1_batch / src0_batch` consecutive src1 slices.
#[derive(Debug, Clone, Copy)]
pub struct DenseMmF16F32Params {
    /// M — number of src1 rows (= output rows per batch).
    pub m: u32,
    /// N — number of src0 rows (= output cols per batch).
    pub n: u32,
    /// K — contract dim, shared between src0 and src1.
    pub k: u32,
    /// src0 batch count.  Each slice is `[n, k]` f16 row-major.
    pub src0_batch: u32,
    /// src1 batch count.  Each slice is `[m, k]` f32 row-major.
    /// Must be an integer multiple of `src0_batch`.
    pub src1_batch: u32,
}

/// GPU-side params struct; matches `DenseMmF16F32TensorParams` in
/// `shaders/dense_mm_f16_tensor.metal` byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseMmF16F32TensorGpuParams {
    ne00: i32,   // K (contract dim)
    ne02: i32,   // src0 batch count
    nb01: u64,   // src0 row stride (bytes)
    nb02: u64,   // src0 batch stride (bytes)
    nb03: u64,   // unused
    ne12: i32,   // src1 batch count
    _pad0: u32,
    nb10: u64,   // sizeof(float) = 4
    nb11: u64,   // src1 row stride (bytes)
    nb12: u64,   // src1 batch stride (bytes)
    nb13: u64,   // unused
    ne0: i32,    // N (output cols = src0 rows)
    ne1: i32,    // M (output rows = src1 rows)
    r2: i16,     // ne12 / ne02 (GQA head broadcast factor)
    r3: i16,
    _pad1: u32,
}

/// Dense f16 × f32 → f32 matmul, tensor-API path.
///
/// Computes `output[b, m, n] = sum_k src0[b/r2, n, k] * src1[b, m, k]`
/// for every `b` in `0..src1_batch`.  Implements llama.cpp's
/// `kernel_mul_mm_f16_f32` contract on the tensor-core path.
///
/// Dtype contract:
/// - `src0`: f16 `[src0_batch, n, k]` row-major.
/// - `src1`: f32 `[src1_batch, m, k]` row-major.
/// - `dst`:  f32 `[src1_batch, m, n]` row-major (output).
///
/// # Errors
///
/// `MlxError::InvalidArgument` for any shape, buffer-size, or dtype
/// mismatch, or if `k < 32` (kernel requires at least one NK=32 tile).
pub fn dense_matmul_f16_f32_tensor(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &mut MlxBuffer,
    params: &DenseMmF16F32Params,
) -> Result<()> {
    if params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_f16_f32_tensor: M, N, K must all be > 0".into(),
        ));
    }
    if params.k < 32 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_f16_f32_tensor: K ({}) must be >= 32",
            params.k
        )));
    }
    if params.src0_batch == 0 || params.src1_batch == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_f16_f32_tensor: batch counts must be > 0".into(),
        ));
    }
    if params.src1_batch % params.src0_batch != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_f16_f32_tensor: src1_batch ({}) must be a \
             multiple of src0_batch ({}) for GQA broadcast",
            params.src1_batch, params.src0_batch
        )));
    }

    let f16_sz = DType::F16.size_of();
    let f32_sz = DType::F32.size_of();

    let expected_src0_bytes =
        (params.src0_batch as usize) * (params.n as usize) * (params.k as usize) * f16_sz;
    if src0.byte_len() < expected_src0_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_f16_f32_tensor: src0 too small: expected {} bytes for \
             [{}x{}x{}] f16, got {}",
            expected_src0_bytes, params.src0_batch, params.n, params.k, src0.byte_len()
        )));
    }
    let expected_src1_bytes =
        (params.src1_batch as usize) * (params.m as usize) * (params.k as usize) * f32_sz;
    if src1.byte_len() < expected_src1_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_f16_f32_tensor: src1 too small: expected {} bytes for \
             [{}x{}x{}] f32, got {}",
            expected_src1_bytes, params.src1_batch, params.m, params.k, src1.byte_len()
        )));
    }
    let expected_dst_bytes =
        (params.src1_batch as usize) * (params.m as usize) * (params.n as usize) * f32_sz;
    if dst.byte_len() < expected_dst_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_f16_f32_tensor: dst too small: expected {} bytes for \
             [{}x{}x{}] f32, got {}",
            expected_dst_bytes, params.src1_batch, params.m, params.n, dst.byte_len()
        )));
    }

    let pipeline = registry
        .get_pipeline("hf2q_dense_mm_f16_f32_tensor", device.metal_device())?;

    let nb01 = (params.k as u64) * (f16_sz as u64);                  // src0 row
    let nb02 = (params.n as u64) * nb01;                              // src0 batch
    let nb11 = (params.k as u64) * (f32_sz as u64);                  // src1 row
    let nb12 = (params.m as u64) * nb11;                              // src1 batch
    let r2 = (params.src1_batch / params.src0_batch) as i16;

    let gpu_params = DenseMmF16F32TensorGpuParams {
        ne00: params.k as i32,
        ne02: params.src0_batch as i32,
        nb01,
        nb02,
        nb03: 0,
        ne12: params.src1_batch as i32,
        _pad0: 0,
        nb10: f32_sz as u64,
        nb11,
        nb12,
        nb13: 0,
        ne0: params.n as i32,
        ne1: params.m as i32,
        r2,
        r3: 1,
        _pad1: 0,
    };

    // Same tile geometry / shmem as the BF16 sibling.  half and bfloat
    // share the 16-bit storage size so the threadgroup memory layout
    // is byte-identical:
    //   sa: 64 * 32 * 2 = 4 KB
    //   sb: 32 * 32 * 2 = 4 KB
    //   sc reuses sa+sb (8 KB) — write-back of [NR0][NR1] floats
    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const THREADS_PER_TG: u64 = 128;
    const SHMEM_BYTES: u64 = 8192;

    let threadgroups = metal::MTLSize::new(
        (params.m as u64 + NR1 - 1) / NR1,
        (params.n as u64 + NR0 - 1) / NR0,
        params.src1_batch as u64,
    );
    let threads_per_tg = metal::MTLSize::new(THREADS_PER_TG, 1, 1);

    encoder.encode_threadgroups_with_args_and_shared(
        pipeline,
        &[
            (0, KernelArg::Bytes(as_bytes(&gpu_params))),
            (1, KernelArg::Buffer(src0)),
            (2, KernelArg::Buffer(src1)),
            (3, KernelArg::Buffer(dst)),
        ],
        &[(0, SHMEM_BYTES)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}
