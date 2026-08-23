//! Dense bf16 × f32 → f32 matmul using the optional Metal tensor API when
//! available and a tiled simdgroup-MMA fallback otherwise.
//!
//! Mirrors the API shape of `quantized_matmul_ggml::quantized_matmul_ggml`
//! (M-N-K with batch broadcasting via r2/r3) but operates on dense bf16
//! weights instead of GGML block-quantized weights.  Used by hf2q's
//! non-flash-attention prefill path for Q@K^T and scores@V, matching
//! the reference mat-mul dispatch when flash attention is disabled.
//!
//! See `src/shaders/dense_mm_bf16_tensor.metal` for the kernel source and
//! its attribution.

use crate::buffer::MlxBuffer;
use crate::device::MlxDevice;
use crate::dtypes::DType;
use crate::encoder::{as_bytes, CommandEncoder, KernelArg};
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;
use crate::ops::dense_bf16_contract::{validate_dense_bf16_contract, DenseBf16Contract};
use crate::ops::dense_mm_capability::{tensor_pipeline_available, DenseMmBackend};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseMmBf16TensorTile {
    Environment,
    V1,
}

/// GPU-side params struct; matches `DenseMmBf16F32TensorParams` in
/// `shaders/dense_mm_bf16_tensor.metal` byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DenseMmBf16F32TensorGpuParams {
    ne00: i32, // K (contract dim)
    ne02: i32, // src0 batch count
    nb01: u64, // src0 row stride (bytes)
    nb02: u64, // src0 batch stride (bytes)
    nb03: u64, // unused
    ne12: i32, // src1 batch count
    _pad0: u32,
    nb10: u64, // sizeof(float) = 4
    nb11: u64, // src1 row stride (bytes)
    nb12: u64, // src1 batch stride (bytes)
    nb13: u64, // unused
    ne0: i32,  // N (output cols = src0 rows)
    ne1: i32,  // M (output rows = src1 rows)
    r2: i16,   // ne12 / ne02 (GQA head broadcast factor)
    r3: i16,
    _pad1: u32,
}

/// Dense bf16 × f32 → f32 matmul with automatic tensor/simdgroup dispatch.
///
/// Computes `output[b, m, n] = sum_k src0[b/r2, n, k] * src1[b, m, k]`
/// for every `b` in `0..src1_batch`.  Implements the
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
/// mismatch. Reductions shorter than the NK=32 tile are zero-padded by the
/// simdgroup and TensorV1 kernels and remain legal; TensorV2 stays gated to
/// K>=32 until its direct cooperative B view has an explicit short-K proof.
pub fn dense_matmul_bf16_f32_tensor(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &MlxBuffer,
    params: &DenseMmBf16F32Params,
) -> Result<()> {
    dense_matmul_bf16_f32_with_tile(
        encoder,
        registry,
        device,
        src0,
        src1,
        dst,
        params,
        DenseMmBackend::Auto,
        DenseMmBf16TensorTile::Environment,
    )
}

#[cfg(test)]
pub(super) fn dense_matmul_bf16_f32_with_backend(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &MlxBuffer,
    params: &DenseMmBf16F32Params,
    backend: DenseMmBackend,
) -> Result<()> {
    dense_matmul_bf16_f32_with_tile(
        encoder,
        registry,
        device,
        src0,
        src1,
        dst,
        params,
        backend,
        DenseMmBf16TensorTile::Environment,
    )
}

pub(crate) fn dense_matmul_bf16_f32_with_tile(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    src0: &MlxBuffer,
    src1: &MlxBuffer,
    dst: &MlxBuffer,
    params: &DenseMmBf16F32Params,
    backend: DenseMmBackend,
    tile: DenseMmBf16TensorTile,
) -> Result<()> {
    // ADR-029 iter-80 H60: V2 large-tile (NRA=64, NRB=128) variant
    // env-gated by HF2Q_LARGE_TILE_MM.  Default OFF until coherence +
    // thermal-fair bench parity proven.  Treated as a fan-out shim: V1
    // pipeline + grid (NR0=64, NR1=32) when off, V2 pipeline + grid
    // (NRA=64, NRB=128) when on.  Truthy: "1", "true", "yes" (case-
    // insensitive); anything else → V1.
    let use_v2_large_tile = params.k >= 32
        && match tile {
            DenseMmBf16TensorTile::Environment => {
                matches!(
                    std::env::var("HF2Q_LARGE_TILE_MM").as_deref(),
                    Ok("1") | Ok("true") | Ok("True") | Ok("TRUE") | Ok("yes") | Ok("YES")
                )
            }
            DenseMmBf16TensorTile::V1 => false,
        };
    let tensor_kernel_name = if use_v2_large_tile {
        "hf2q_dense_mm_bf16_f32_tensor_v2"
    } else {
        "hf2q_dense_mm_bf16_f32_tensor"
    };
    // Tensor V1 performs aligned `float4` loads from every input row. A K
    // stride that is not a multiple of four F32 values is legal for the
    // scalar-staging simdgroup route, but cannot be sent to Tensor V1.
    let use_tensor = if params.k % 4 != 0 {
        match backend {
            DenseMmBackend::TensorRequired => {
                validate_dense_bf16_contract(
                    "dense_matmul_bf16_f32_tensor_v1",
                    DenseBf16Contract::MatrixTensor,
                    src0,
                    src1,
                    dst,
                    params,
                )?;
                return Err(MlxError::InvalidArgument(
                    "dense_matmul_bf16_f32_tensor_v1: non-vector-aligned K was not rejected".into(),
                ));
            }
            DenseMmBackend::Auto | DenseMmBackend::FallbackRequired => false,
        }
    } else {
        tensor_pipeline_available(backend, "hf2q_dense_mm_bf16_f32_tensor", registry, device)?
    };
    validate_dense_bf16_contract(
        if use_tensor {
            "dense_matmul_bf16_f32_tensor_v1"
        } else {
            "dense_matmul_bf16_f32_simdgroup"
        },
        if use_tensor {
            DenseBf16Contract::MatrixTensor
        } else {
            DenseBf16Contract::MatrixSimdgroup
        },
        src0,
        src1,
        dst,
        params,
    )?;
    let broadcast = params.src1_batch / params.src0_batch;

    let bf16_sz = DType::BF16.size_of();
    let f32_sz = DType::F32.size_of();

    let kernel_name = if use_tensor {
        tensor_kernel_name
    } else {
        "hf2q_dense_mm_bf16_f32_fallback"
    };
    let pipeline = registry.get_pipeline(kernel_name, device.metal_device())?;

    let nb01 = (params.k as u64) * (bf16_sz as u64); // src0 row
    let nb02 = (params.n as u64) * nb01; // src0 batch
    let nb11 = (params.k as u64) * (f32_sz as u64); // src1 row
    let nb12 = (params.m as u64) * nb11; // src1 batch
    let r2 = broadcast as i16;

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

    if !use_tensor {
        const NR0: u64 = 64;
        const NR1: u64 = 32;
        encoder.encode_threadgroups_with_args_and_shared(
            pipeline,
            &[
                (0, KernelArg::Bytes(as_bytes(&gpu_params))),
                (1, KernelArg::Buffer(src0)),
                (2, KernelArg::Buffer(src1)),
                (3, KernelArg::Buffer(dst)),
            ],
            &[(0, 8192)],
            metal::MTLSize::new(
                (params.m as u64 + NR1 - 1) / NR1,
                (params.n as u64 + NR0 - 1) / NR0,
                params.src1_batch as u64,
            ),
            metal::MTLSize::new(128, 1, 1),
        );
        return Ok(());
    }

    // V1 tile: NR0=64 (M_peer axis = hf2q-N), NR1=32 (N_peer axis = hf2q-M).
    // V2 tile: NRA=64 (M_peer = hf2q-N), NRB=128 (N_peer = hf2q-M).
    // Note hf2q axis swap: ne0 = hf2q-N (M_peer), ne1 = hf2q-M (N_peer);
    // tgpig.y covers M_peer-axis (NRA/NR0), tgpig.x covers N_peer-axis
    // (NRB/NR1).  Threads-per-TG = NUM_THREADS = 128 in both (4 simdgroups
    // × 32 lanes).  V2 shmem: A-tile only (NRA × NK = 64 × 32 × 2 B =
    // 4096 B), B read direct from device → halved shmem budget vs V1.
    const NR0: u64 = 64;
    const NR1_V1: u64 = 32;
    const NRB_V2: u64 = 128;
    const THREADS_PER_TG: u64 = 128;
    const SHMEM_V1: u64 = 8192;
    const SHMEM_V2: u64 = 4096;

    let (nr1, shmem_bytes) = if use_v2_large_tile {
        (NRB_V2, SHMEM_V2)
    } else {
        (NR1_V1, SHMEM_V1)
    };

    // Grid: (ceil(M/nr1), ceil(N/NR0), src1_batch).  M → tgpig.x (covers
    // N_peer = hf2q-M), N → tgpig.y (covers M_peer = hf2q-N), batch → z.
    let threadgroups = metal::MTLSize::new(
        (params.m as u64 + nr1 - 1) / nr1,
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
        &[(0, shmem_bytes)],
        threadgroups,
        threads_per_tg,
    );

    Ok(())
}
