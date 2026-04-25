//! Dense bf16 × f32 → f32 matmul using Apple M3+ tensor cores
//! (`mpp::tensor_ops::matmul2d`).
//!
//! Mirrors the API shape of `quantized_matmul_ggml::quantized_matmul_ggml`
//! (M-N-K with batch broadcasting via r2/r3) but operates on dense bf16
//! weights instead of GGML block-quantized weights.  Used by hf2q's
//! non-flash-attention prefill path for Q@K^T and scores@V, matching
//! llama.cpp's `ggml_mul_mat` dispatch when `-fa 0`.
//!
//! Derived from llama.cpp (MIT).  See `src/shaders/dense_mm_bf16_tensor.metal`.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{CommandEncoder, KernelArg, as_bytes};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

/// Host-side parameters for `dense_matmul_bf16_f32_tensor`.
#[derive(Debug, Clone, Copy)]
pub struct DenseMmBf16F32Params {
    /// M — number of src1 rows (= output rows per batch).
    pub m: u32,
    /// N — number of src0 rows (= output cols per batch).
    pub n: u32,
    /// K — contract dim, shared between src0 and src1.
    pub k: u32,
    /// src0 batch count (e.g. nkv for attention GQA).  Every batch slice
    /// is laid out contiguously as `[n, k]` bf16 row-major.
    pub src0_batch: u32,
    /// src1 batch count (e.g. nh for attention).  Every slice is
    /// `[m, k]` f32 row-major.  Must be an integer multiple of
    /// `src0_batch` — the kernel broadcasts each src0 slice across
    /// `src1_batch / src0_batch` consecutive src1 slices (GQA head
    /// broadcast).
    pub src1_batch: u32,
}

/// GPU-side params struct; matches `DenseMmBf16F32TensorParams` in
/// `shaders/dense_mm_bf16_tensor.metal` byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseMmBf16F32TensorGpuParams {
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

/// Dense bf16 × f32 → f32 matmul, tensor-API path.
///
/// Computes `output[b, m, n] = sum_k src0[b/r2, n, k] * src1[b, m, k]`
/// for every `b` in `0..src1_batch`.  Implements llama.cpp's
/// `kernel_mul_mm_bf16_f32` contract on the tensor-core path.
///
/// Dtype contract:
/// - `src0`: bf16 `[src0_batch, n, k]` row-major.
/// - `src1`: f32 `[src1_batch, m, k]` row-major.
/// - `dst`:  f32 `[src1_batch, m, n]` row-major (output).
///
/// # Errors
///
/// `MlxError::InvalidArgument` for any shape, buffer-size, or dtype
/// mismatch, or if `k < 32` (kernel requires at least one NK=32 tile).
pub fn dense_matmul_bf16_f32_tensor(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &mut MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<()> {
    if params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_bf16_f32_tensor: M, N, K must all be > 0".into(),
        ));
    }
    if params.k < 32 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_bf16_f32_tensor: K ({}) must be >= 32",
            params.k
        )));
    }
    if params.src0_batch == 0 || params.src1_batch == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_matmul_bf16_f32_tensor: batch counts must be > 0".into(),
        ));
    }
    if params.src1_batch % params.src0_batch != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_bf16_f32_tensor: src1_batch ({}) must be a \
             multiple of src0_batch ({}) for GQA broadcast",
            params.src1_batch, params.src0_batch
        )));
    }

    let bf16_sz = DType::BF16.size_of();
    let f32_sz = DType::F32.size_of();

    let expected_src0_bytes =
        (params.src0_batch as usize) * (params.n as usize) * (params.k as usize) * bf16_sz;
    if src0.byte_len() < expected_src0_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_bf16_f32_tensor: src0 too small: expected {} bytes for \
             [{}×{}×{}] bf16, got {}",
            expected_src0_bytes, params.src0_batch, params.n, params.k, src0.byte_len()
        )));
    }
    let expected_src1_bytes =
        (params.src1_batch as usize) * (params.m as usize) * (params.k as usize) * f32_sz;
    if src1.byte_len() < expected_src1_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_bf16_f32_tensor: src1 too small: expected {} bytes for \
             [{}×{}×{}] f32, got {}",
            expected_src1_bytes, params.src1_batch, params.m, params.k, src1.byte_len()
        )));
    }
    let expected_dst_bytes =
        (params.src1_batch as usize) * (params.m as usize) * (params.n as usize) * f32_sz;
    if dst.byte_len() < expected_dst_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_matmul_bf16_f32_tensor: dst too small: expected {} bytes for \
             [{}×{}×{}] f32, got {}",
            expected_dst_bytes, params.src1_batch, params.m, params.n, dst.byte_len()
        )));
    }

    let pipeline = registry
        .get_pipeline("hf2q_dense_mm_bf16_f32_tensor", device.metal_device())?;

    let nb01 = (params.k as u64) * (bf16_sz as u64);                 // src0 row
    let nb02 = (params.n as u64) * nb01;                             // src0 batch
    let nb11 = (params.k as u64) * (f32_sz as u64);                  // src1 row
    let nb12 = (params.m as u64) * nb11;                             // src1 batch
    let r2 = (params.src1_batch / params.src0_batch) as i16;

    let gpu_params = DenseMmBf16F32TensorGpuParams {
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

    // Same tile geometry / shmem as the quantized tensor kernel.
    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const THREADS_PER_TG: u64 = 128;
    const SHMEM_BYTES: u64 = 8192;

    // Grid: (ceil(M/32), ceil(N/64), src1_batch).  Note M -> x-axis,
    // N -> y-axis, im -> z-axis (matches the kernel's tgpig unpacking).
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
