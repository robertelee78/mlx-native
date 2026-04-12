//! Dense F16 matrix multiply for the lm_head vocabulary projection.
//!
//! Computes `C = A * B^T` where A is [M, K] f16, B is [N, K] f16,
//! and C is [M, N] f16.
//!
//! This is a simple tiled implementation for correctness.  Optimization
//! (e.g., larger tiles, shared memory tiling, vectorized loads) is deferred
//! to a later sprint.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{as_bytes, encode_with_args, KernelArg};

/// MSL source for the dense GEMM kernel (embedded at compile time).
pub static DENSE_GEMM_SHADER_SOURCE: &str = include_str!("../shaders/dense_gemm.metal");

/// Register dense GEMM shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("dense_gemm_f16", DENSE_GEMM_SHADER_SOURCE);
}

/// MSL-compatible params struct for dense GEMM.
///
/// Must match `DenseGemmParams` in `dense_gemm.metal`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDenseGemmParams {
    m: u32,
    n: u32,
    k: u32,
}

/// Parameters for a dense GEMM operation.
pub struct DenseGemmF16Params {
    /// Number of rows in A (and C).
    pub m: u32,
    /// Number of rows in B (columns in C).  C = A * B^T is [M, N].
    pub n: u32,
    /// Inner dimension (columns of A and B).
    pub k: u32,
}

/// Dispatch a dense F16 matrix multiply on the GPU: `C = A * B^T`.
///
/// A is `[M, K]` f16, B is `[N, K]` f16, C is `[M, N]` f16.
///
/// # Arguments
///
/// * `encoder`  - Command encoder to record the dispatch into.
/// * `registry` - Kernel registry (must have `dense_gemm_f16` registered).
/// * `device`   - Metal device for pipeline compilation.
/// * `a`        - Matrix A buffer `[M, K]` (f16).
/// * `b`        - Matrix B buffer `[N, K]` (f16).
/// * `output`   - Output buffer C `[M, N]` (f16).
/// * `params`   - GEMM dimensions.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if dimensions are 0 or buffers are
/// too small.
pub fn dispatch_dense_gemm_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    a: &MlxBuffer,
    b: &MlxBuffer,
    output: &MlxBuffer,
    params: &DenseGemmF16Params,
) -> Result<()> {
    if params.m == 0 || params.n == 0 || params.k == 0 {
        return Err(MlxError::InvalidArgument(
            "dense_gemm_f16: M, N, and K must all be > 0".into(),
        ));
    }

    let a_bytes = params.m as usize * params.k as usize * 2; // f16 = 2 bytes
    if a.byte_len() < a_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_gemm_f16: A buffer too small: need {} bytes, have {}",
            a_bytes,
            a.byte_len()
        )));
    }
    let b_bytes = params.n as usize * params.k as usize * 2;
    if b.byte_len() < b_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_gemm_f16: B buffer too small: need {} bytes, have {}",
            b_bytes,
            b.byte_len()
        )));
    }
    let c_bytes = params.m as usize * params.n as usize * 2;
    if output.byte_len() < c_bytes {
        return Err(MlxError::InvalidArgument(format!(
            "dense_gemm_f16: output buffer too small: need {} bytes, have {}",
            c_bytes,
            output.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("dense_gemm_f16", device)?;

    let gpu_params = GpuDenseGemmParams {
        m: params.m,
        n: params.n,
        k: params.k,
    };

    // Tile sizes match the shader constants (TILE_M=8, TILE_N=8).
    let tile_m: u64 = 8;
    let tile_n: u64 = 8;

    let grid = MTLSize::new(
        ((params.n as u64 + tile_n - 1) / tile_n) * tile_n,
        ((params.m as u64 + tile_m - 1) / tile_m) * tile_m,
        1,
    );
    let tg = MTLSize::new(tile_n, tile_m, 1);

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(a)),
            (1, KernelArg::Buffer(b)),
            (2, KernelArg::Buffer(output)),
            (3, KernelArg::Bytes(as_bytes(&gpu_params))),
        ],
        grid,
        tg,
    );

    Ok(())
}
