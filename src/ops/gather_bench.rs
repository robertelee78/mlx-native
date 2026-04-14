//! Gather throughput microbenchmark dispatch.
//!
//! Provides two kernels for measuring KV cache read throughput:
//!
//! * `gather_bench_nibble`   — Simulates TurboQuant SDPA: unpack 4-bit nibble
//!   indices then gather from a 16-entry centroid table.
//! * `gather_bench_f16_seq`  — Baseline: sequential F16 read + widen to F32.
//!
//! The throughput ratio between the two kernels determines whether nibble-gather
//! meets the ADR-007 gate of ≥ 50% of sequential F16 throughput.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{encode_with_args, KernelArg};

/// MSL source for the gather benchmark kernels (embedded at compile time).
pub static GATHER_BENCH_SHADER_SOURCE: &str = include_str!("../shaders/gather_bench.metal");

/// Register both gather benchmark kernels with the given registry.
///
/// Each Metal kernel function name must be registered individually so that
/// `KernelRegistry::get_pipeline` can look up the source by function name.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("gather_bench_nibble", GATHER_BENCH_SHADER_SOURCE);
    registry.register_source("gather_bench_f16_seq", GATHER_BENCH_SHADER_SOURCE);
}

/// Dispatch the nibble-gather kernel.
///
/// Reads nibble-packed 4-bit indices and gathers from a centroid table,
/// simulating the TurboQuant SDPA KV-cache read path.
///
/// # Arguments
///
/// * `encoder`   — Command encoder to record the dispatch into.
/// * `registry`  — Kernel registry (must have gather_bench registered).
/// * `device`    — Metal device for pipeline compilation.
/// * `packed`    — Nibble-packed index buffer `[capacity × head_dim/2]` (u8).
/// * `centroids` — Centroid table buffer `[16 × head_dim]` (f32).
/// * `out`       — Output buffer `[capacity × head_dim]` (f32).
/// * `capacity`  — Number of token positions.
/// * `head_dim`  — Head dimension (must be even).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if `head_dim` is odd or parameters are
/// inconsistent with buffer sizes.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gather_nibble(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    packed: &MlxBuffer,
    centroids: &MlxBuffer,
    out: &MlxBuffer,
    capacity: u32,
    head_dim: u32,
) -> Result<()> {
    if capacity == 0 || head_dim == 0 {
        return Ok(());
    }
    if head_dim % 2 != 0 {
        return Err(MlxError::InvalidArgument(format!(
            "gather_bench_nibble: head_dim must be even, got {}",
            head_dim
        )));
    }

    let pipeline = registry.get_pipeline("gather_bench_nibble", device)?;

    let capacity_bytes = capacity.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();

    // Grid: x covers head_dim coordinates, y covers capacity positions.
    // Threadgroup x = min(256, head_dim), y = 1.
    let tg_x = std::cmp::min(256, head_dim as u64);
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(packed)),
            (1, KernelArg::Buffer(centroids)),
            (2, KernelArg::Bytes(&capacity_bytes)),
            (3, KernelArg::Bytes(&head_dim_bytes)),
            (4, KernelArg::Buffer(out)),
        ],
        MTLSize::new(head_dim as u64, capacity as u64, 1),
        MTLSize::new(tg_x, 1, 1),
    );

    Ok(())
}

/// Dispatch the sequential F16 read kernel.
///
/// Reads every F16 element of the KV cache and widens it to F32, providing a
/// throughput baseline against which `gather_bench_nibble` is compared.
///
/// # Arguments
///
/// * `encoder`   — Command encoder to record the dispatch into.
/// * `registry`  — Kernel registry (must have gather_bench registered).
/// * `device`    — Metal device for pipeline compilation.
/// * `cache`     — F16 KV cache buffer `[capacity × head_dim]` (half / u16).
/// * `out`       — Output buffer `[capacity × head_dim]` (f32).
/// * `capacity`  — Number of token positions.
/// * `head_dim`  — Head dimension.
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are inconsistent.
pub fn dispatch_gather_f16_seq(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    cache: &MlxBuffer,
    out: &MlxBuffer,
    capacity: u32,
    head_dim: u32,
) -> Result<()> {
    if capacity == 0 || head_dim == 0 {
        return Ok(());
    }

    let pipeline = registry.get_pipeline("gather_bench_f16_seq", device)?;

    let capacity_bytes = capacity.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();

    let tg_x = std::cmp::min(256, head_dim as u64);
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(cache)),
            (1, KernelArg::Bytes(&capacity_bytes)),
            (2, KernelArg::Bytes(&head_dim_bytes)),
            (3, KernelArg::Buffer(out)),
        ],
        MTLSize::new(head_dim as u64, capacity as u64, 1),
        MTLSize::new(tg_x, 1, 1),
    );

    Ok(())
}
