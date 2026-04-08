//! KV cache GPU copy dispatch.
//!
//! Copies new K or V data directly from a source GPU buffer into a
//! pre-allocated KV cache buffer at the correct write position, with
//! optional modulo wrapping for sliding window (ring buffer) caches.
//!
//! This eliminates the CPU round-trip that `append_bf16` requires:
//! instead of GPU -> CPU (as_slice) -> CPU (copy loop) -> shared buffer,
//! the GPU copies directly between two shared Metal buffers.

use metal::MTLSize;

use crate::buffer::MlxBuffer;
use crate::encoder::CommandEncoder;
use crate::error::{MlxError, Result};
use crate::kernel_registry::KernelRegistry;

use super::encode_helpers::{encode_with_args, KernelArg};

/// MSL source for the KV cache copy kernel (embedded at compile time).
pub static KV_CACHE_COPY_SHADER_SOURCE: &str = include_str!("../shaders/kv_cache_copy.metal");

/// Register KV cache copy shader source with the given kernel registry.
pub fn register(registry: &mut KernelRegistry) {
    registry.register_source("kv_cache_copy", KV_CACHE_COPY_SHADER_SOURCE);
}

/// Dispatch a GPU copy from a source bf16 buffer into a KV cache buffer.
///
/// Both `src` and `cache` must be bf16 Metal buffers in shared memory.
///
/// # Arguments
///
/// * `encoder`   - Command encoder to record the dispatch into.
/// * `registry`  - Kernel registry (must have kv_cache_copy registered).
/// * `device`    - Metal device for pipeline compilation.
/// * `src`       - Source buffer of shape `[n_new, row_size]` (bf16).
/// * `cache`     - Destination cache buffer (bf16, pre-allocated).
/// * `write_pos` - Starting write position in the cache (token index).
/// * `row_size`  - Elements per token row (`n_kv_heads * head_dim`).
/// * `n_new`     - Number of new tokens to copy.
/// * `cache_cap` - Cache capacity (window size for sliding, max_seq_len for global).
/// * `is_sliding`- Whether to use modulo wrapping (`true` for sliding window).
///
/// # Errors
///
/// Returns `MlxError::InvalidArgument` if parameters are inconsistent.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    cache: &MlxBuffer,
    write_pos: u32,
    row_size: u32,
    n_new: u32,
    cache_cap: u32,
    is_sliding: bool,
) -> Result<()> {
    if n_new == 0 || row_size == 0 {
        return Ok(()); // Nothing to copy
    }

    let total_elements = (n_new as u64) * (row_size as u64);
    let src_elements = src.element_count() as u64;
    if src_elements < total_elements {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy: src has {} elements but need {} (n_new={} * row_size={})",
            src_elements, total_elements, n_new, row_size
        )));
    }

    // For global (non-sliding) caches, check we won't write past capacity
    if !is_sliding && (write_pos as u64 + n_new as u64) > cache_cap as u64 {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy: global cache overflow: write_pos({}) + n_new({}) > cache_cap({})",
            write_pos, n_new, cache_cap
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy", device)?;

    let is_sliding_val: u32 = if is_sliding { 1 } else { 0 };

    // Pass each scalar as individual set_bytes calls matching buffer indices 2-6
    let write_pos_bytes = write_pos.to_ne_bytes();
    let row_size_bytes = row_size.to_ne_bytes();
    let n_new_bytes = n_new.to_ne_bytes();
    let cache_cap_bytes = cache_cap.to_ne_bytes();
    let is_sliding_bytes = is_sliding_val.to_ne_bytes();

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(cache)),
            (2, KernelArg::Bytes(&write_pos_bytes)),
            (3, KernelArg::Bytes(&row_size_bytes)),
            (4, KernelArg::Bytes(&n_new_bytes)),
            (5, KernelArg::Bytes(&cache_cap_bytes)),
            (6, KernelArg::Bytes(&is_sliding_bytes)),
        ],
        MTLSize::new(total_elements, 1, 1),
        MTLSize::new(std::cmp::min(256, total_elements), 1, 1),
    );

    Ok(())
}
