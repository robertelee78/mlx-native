//! Cumulative sum (inclusive prefix sum) along the last axis.
//!
//! Computes `out[r, i] = sum(x[r, 0..=i])` for every row `r`.
//!
//! Used by the chunked Gated DeltaNet path to produce the decay-mask base
//! (ADR-013 Decision 4). Spec derived from the definition of an inclusive
//! prefix scan.
//!
//! # Algorithm
//!
//! Hillis-Steele scan in threadgroup shared memory. Each thread owns a
//! contiguous chunk of the row; it scans locally first, exchanges chunk
//! totals via a shared-memory scan, then adds the exclusive prefix of
//! preceding threads to its chunk.
//!
//! # Threadgroup shape
//!
//! One threadgroup per row. `tg_size = min(256, next_power_of_two(dim))`
//! and each thread handles `ceil_div(dim, tg_size)` elements. The shader
//! caps the per-thread chunk at `CUMSUM_MAX_CHUNK = 32`, so for `tg_size =
//! 256` the maximum `dim` handled in a single dispatch is 8192.
//!
//! Reduction and prefix arithmetic are performed in f32 regardless of input
//! dtype for numerical stability.
use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::dtypes::DType;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

pub static CUMSUM_SHADER_SOURCE: &str = include_str!("../shaders/cumsum.metal");

/// Register cumsum shader sources with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("cumsum_f32", CUMSUM_SHADER_SOURCE);
    registry.register_source("cumsum_bf16", CUMSUM_SHADER_SOURCE);
}

/// Maximum per-thread chunk; keep in sync with `CUMSUM_MAX_CHUNK` in
/// `cumsum.metal`. Threads must not exceed this chunk size or they'll
/// overrun their private buffer.
const SHADER_MAX_CHUNK: u32 = 32;

/// Dispatch an inclusive prefix sum along the last axis of `[rows, dim]`.
///
/// # Arguments
///
/// * `input`      - shape `[rows, dim]`, f32 or bf16.
/// * `output`     - same shape + dtype as `input`.
/// * `params_buf` - one u32: `[dim]`. The kernel also reads `tg_size` from
///                  the Metal dispatch so only `dim` needs to be in-buffer.
/// * `rows`       - number of independent rows.
/// * `dim`        - length of each row; must satisfy
///                  `ceil_div(dim, tg_size) <= 32`.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if:
/// - `rows == 0` or `dim == 0`.
/// - input/output element counts disagree with `rows * dim`.
/// - input and output dtypes differ, or are not f32/bf16.
/// - `dim` exceeds `tg_size * SHADER_MAX_CHUNK` (8192 for the default
///   `tg_size = 256`).
pub fn dispatch_cumsum(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params_buf: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> Result<()> {
    if rows == 0 || dim == 0 {
        return Err(MlxError::InvalidArgument(
            "cumsum rows and dim must be > 0".into(),
        ));
    }

    let expected = (rows as usize) * (dim as usize);
    if input.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "cumsum input element count {} != rows({}) * dim({})",
            input.element_count(),
            rows,
            dim
        )));
    }
    if output.element_count() != expected {
        return Err(MlxError::InvalidArgument(format!(
            "cumsum output element count {} != rows({}) * dim({})",
            output.element_count(),
            rows,
            dim
        )));
    }
    if input.dtype() != output.dtype() {
        return Err(MlxError::InvalidArgument(format!(
            "cumsum input/output dtype mismatch: {} vs {}",
            input.dtype(),
            output.dtype()
        )));
    }

    let kernel_name = match input.dtype() {
        DType::F32 => "cumsum_f32",
        DType::BF16 => "cumsum_bf16",
        _ => {
            return Err(MlxError::InvalidArgument(format!(
                "cumsum unsupported dtype: {}",
                input.dtype()
            )));
        }
    };

    // Choose threadgroup size: smallest power of two >= dim, capped at 256.
    // Must be a power of two for the Hillis-Steele offset loop. Must be >= 1
    // (true because we rejected dim == 0 above).
    let tg_size = std::cmp::min(256u32, dim.next_power_of_two());
    let tg_size = std::cmp::max(tg_size, 1u32);

    let chunk = dim.div_ceil(tg_size);
    if chunk > SHADER_MAX_CHUNK {
        return Err(MlxError::InvalidArgument(format!(
            "cumsum dim {} exceeds supported limit: tg_size {} * chunk {} < dim",
            dim, tg_size, SHADER_MAX_CHUNK
        )));
    }

    let pipeline = registry.get_pipeline(kernel_name, device)?;

    // Shared memory: tg_size floats for the cross-thread scan.
    let shared_mem_bytes = (tg_size as u64) * 4;

    encoder.encode_threadgroups_with_shared(
        pipeline,
        &[(0, input), (1, output), (2, params_buf)],
        &[(0, shared_mem_bytes)],
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(tg_size as u64, 1, 1),
    );

    Ok(())
}
