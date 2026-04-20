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

/// Dispatch a batched GPU copy from a source f32 buffer into a f32 KV cache.
///
/// Copies ALL heads in one dispatch instead of one dispatch per head.
///
/// Source layout: `[n_heads * head_dim]` flat (one token, all heads).
/// Cache layout: `[n_heads, capacity, head_dim]` head-major.
///
/// # Arguments
///
/// * `encoder`   - Command encoder to record the dispatch into.
/// * `registry`  - Kernel registry (must have kv_cache_copy_batch_f32 registered).
/// * `device`    - Metal device for pipeline compilation.
/// * `src`       - Source buffer of shape `[n_heads * head_dim]` (f32).
/// * `cache`     - Destination cache buffer (f32, pre-allocated).
/// * `n_heads`   - Number of KV heads.
/// * `head_dim`  - Elements per head.
/// * `capacity`  - Cache capacity (window size or max_seq_len).
/// * `seq_pos`   - Write position in cache (already wrapped for sliding).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_batch_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    cache: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    seq_pos: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    let total_src = (n_heads as u64) * (head_dim as u64);
    if (src.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_batch_f32: src has {} elements but need {} (n_heads={} * head_dim={})",
            src.element_count(), total_src, n_heads, head_dim
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_batch_f32", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_bytes = seq_pos.to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(cache)),
            (2, KernelArg::Bytes(&n_heads_bytes)),
            (3, KernelArg::Bytes(&head_dim_bytes)),
            (4, KernelArg::Bytes(&capacity_bytes)),
            (5, KernelArg::Bytes(&seq_pos_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, 1),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Dispatch a GPU copy from a source f32 buffer into a f32 KV cache buffer.
///
/// Identical to `dispatch_kv_cache_copy` but for F32 data (used when the
/// activation pipeline operates in F32 throughout).
///
/// Both `src` and `cache` must be f32 Metal buffers in shared memory.
///
/// # Arguments
///
/// * `encoder`   - Command encoder to record the dispatch into.
/// * `registry`  - Kernel registry (must have kv_cache_copy_f32 registered).
/// * `device`    - Metal device for pipeline compilation.
/// * `src`       - Source buffer of shape `[n_new, row_size]` (f32).
/// * `cache`     - Destination cache buffer (f32, pre-allocated).
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
pub fn dispatch_kv_cache_copy_f32(
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
            "kv_cache_copy_f32: src has {} elements but need {} (n_new={} * row_size={})",
            src_elements, total_elements, n_new, row_size
        )));
    }

    // For global (non-sliding) caches, check we won't write past capacity
    if !is_sliding && (write_pos as u64 + n_new as u64) > cache_cap as u64 {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_f32: global cache overflow: write_pos({}) + n_new({}) > cache_cap({})",
            write_pos, n_new, cache_cap
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_f32", device)?;

    let is_sliding_val: u32 = if is_sliding { 1 } else { 0 };

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

/// Dispatch a batched F32→F16 copy from a source f32 buffer into an f16 KV cache.
///
/// Copies ALL heads in one dispatch, casting float→half on write.
/// This halves KV cache memory bandwidth for SDPA reads (bandwidth-bound
/// at batch=1 decode). Reference: llama.cpp stores KV cache in F16.
///
/// Source layout: `[n_heads * head_dim]` flat F32 (one token, all heads).
/// Cache layout: `[n_heads, capacity, head_dim]` head-major F16.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_batch_f32_to_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    cache: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    seq_pos: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    let total_src = (n_heads as u64) * (head_dim as u64);
    if (src.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_batch_f32_to_f16: src has {} elements but need {} (n_heads={} * head_dim={})",
            src.element_count(), total_src, n_heads, head_dim
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_batch_f32_to_f16", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_bytes = seq_pos.to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(cache)),
            (2, KernelArg::Bytes(&n_heads_bytes)),
            (3, KernelArg::Bytes(&head_dim_bytes)),
            (4, KernelArg::Bytes(&capacity_bytes)),
            (5, KernelArg::Bytes(&seq_pos_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, 1),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Multi-position, all-heads KV cache copy (F32 → F32 cache, batched prefill).
///
/// Source layout: `[n_src_tokens, n_heads, head_dim]` (token-major). The
/// kernel reads `[src_tok_offset, src_tok_offset + n_tokens)` from it.
/// Cache layout:  `[n_heads, capacity, head_dim]` (head-major).
/// Writes absolute positions `[seq_pos_start, seq_pos_start + n_tokens)` into
/// cache slots `dst_pos % capacity`.
///
/// Global-layer contract: caller sets `seq_pos_start + n_tokens <= capacity`
/// so `dst_pos % capacity == dst_pos` and writes are linear. Typical call:
/// `src_tok_offset = 0`, `n_tokens = seq_len`, `seq_pos_start = 0`.
///
/// Sliding-window contract: caller sets `capacity = sliding_window`,
/// `n_tokens = min(seq_len, capacity)`, `src_tok_offset = seq_len - n_tokens`,
/// `seq_pos_start = seq_len - n_tokens`. This writes the last `n_tokens`
/// source tokens into modular slots exactly once — no intra-dispatch race.
/// Decode side reads via `ring_start = write_pos % capacity`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_seq_f32(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    cache: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    seq_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    let total_src = ((src_tok_offset as u64) + (n_tokens as u64))
        * (n_heads as u64) * (head_dim as u64);
    if (src.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_seq_f32: src has {} elements, need {} ((src_tok_offset={} + n_tokens={}) * n_heads={} * head_dim={})",
            src.element_count(), total_src, src_tok_offset, n_tokens, n_heads, head_dim
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_seq_f32", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_start_bytes = seq_pos_start.to_ne_bytes();
    let n_tokens_bytes = n_tokens.to_ne_bytes();
    let src_tok_offset_bytes = src_tok_offset.to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(cache)),
            (2, KernelArg::Bytes(&n_heads_bytes)),
            (3, KernelArg::Bytes(&head_dim_bytes)),
            (4, KernelArg::Bytes(&capacity_bytes)),
            (5, KernelArg::Bytes(&seq_pos_start_bytes)),
            (6, KernelArg::Bytes(&n_tokens_bytes)),
            (7, KernelArg::Bytes(&src_tok_offset_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Fused K + V cache copy (F32 source → F32 cache).  Wave P4.11.
///
/// Combines two `dispatch_kv_cache_copy_seq_f32` calls (one for K, one
/// for V) into one dispatch.  Both streams share identical metadata
/// (n_heads, head_dim, capacity, seq_pos_start, n_tokens,
/// src_tok_offset) and are independently addressed in src/cache, so a
/// single thread can copy one (K, V) element pair at the same
/// coordinates.  Saves 1 dispatch per layer (30/prefill on Gemma 4).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_seq_f32_dual(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src_k: &MlxBuffer,
    src_v: &MlxBuffer,
    cache_k: &MlxBuffer,
    cache_v: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    seq_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    let total_src = ((src_tok_offset as u64) + (n_tokens as u64))
        * (n_heads as u64) * (head_dim as u64);
    for (name, b) in [("src_k", src_k), ("src_v", src_v)] {
        if (b.element_count() as u64) < total_src {
            return Err(MlxError::InvalidArgument(format!(
                "kv_cache_copy_seq_f32_dual: {} has {} elements, need {}",
                name, b.element_count(), total_src
            )));
        }
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_seq_f32_kv_dual", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_start_bytes = seq_pos_start.to_ne_bytes();
    let n_tokens_bytes = n_tokens.to_ne_bytes();
    let src_tok_offset_bytes = src_tok_offset.to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src_k)),
            (1, KernelArg::Buffer(src_v)),
            (2, KernelArg::Buffer(cache_k)),
            (3, KernelArg::Buffer(cache_v)),
            (4, KernelArg::Bytes(&n_heads_bytes)),
            (5, KernelArg::Bytes(&head_dim_bytes)),
            (6, KernelArg::Bytes(&capacity_bytes)),
            (7, KernelArg::Bytes(&seq_pos_start_bytes)),
            (8, KernelArg::Bytes(&n_tokens_bytes)),
            (9, KernelArg::Bytes(&src_tok_offset_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Fused K + V cache copy (F32 source → F16 cache).  Wave P4.11
/// f16-cache variant of `dispatch_kv_cache_copy_seq_f32_dual`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_seq_f32_to_f16_dual(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src_k: &MlxBuffer,
    src_v: &MlxBuffer,
    cache_k: &MlxBuffer,
    cache_v: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    seq_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    let total_src = ((src_tok_offset as u64) + (n_tokens as u64))
        * (n_heads as u64) * (head_dim as u64);
    for (name, b) in [("src_k", src_k), ("src_v", src_v)] {
        if (b.element_count() as u64) < total_src {
            return Err(MlxError::InvalidArgument(format!(
                "kv_cache_copy_seq_f32_to_f16_dual: {} has {} elements, need {}",
                name, b.element_count(), total_src
            )));
        }
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_seq_f32_to_f16_kv_dual", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_start_bytes = seq_pos_start.to_ne_bytes();
    let n_tokens_bytes = n_tokens.to_ne_bytes();
    let src_tok_offset_bytes = src_tok_offset.to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src_k)),
            (1, KernelArg::Buffer(src_v)),
            (2, KernelArg::Buffer(cache_k)),
            (3, KernelArg::Buffer(cache_v)),
            (4, KernelArg::Bytes(&n_heads_bytes)),
            (5, KernelArg::Bytes(&head_dim_bytes)),
            (6, KernelArg::Bytes(&capacity_bytes)),
            (7, KernelArg::Bytes(&seq_pos_start_bytes)),
            (8, KernelArg::Bytes(&n_tokens_bytes)),
            (9, KernelArg::Bytes(&src_tok_offset_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Multi-position, all-heads KV cache copy (BF16 source → F32 cache, batched prefill).
///
/// Same layout and semantics as [`dispatch_kv_cache_copy_seq_f32`] — including
/// `src_tok_offset` source slicing and `dst_pos % capacity` ring-wrap for
/// sliding-window layers — but reads bfloat16 from the source and promotes to
/// float32 on write.
///
/// Used in the Phase 2 bf16 activation path where `pf_k_normed` / `pf_v_normed`
/// become bf16, but the KV cache (used by decode SDPA) stays f32.
///
/// Source layout: `[n_src_tokens, n_heads, head_dim]` bf16.
/// Cache layout:  `[n_heads, capacity, head_dim]`     f32.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_seq_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    cache: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    seq_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    // src is bf16 (2 bytes per element)
    let total_src = ((src_tok_offset as u64) + (n_tokens as u64))
        * (n_heads as u64) * (head_dim as u64);
    let src_bytes_needed = total_src * 2; // bf16 = 2 bytes
    if (src.byte_len() as u64) < src_bytes_needed {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_seq_bf16: src has {} bytes, need {} ((src_tok_offset={} + n_tokens={}) * n_heads={} * head_dim={} * 2)",
            src.byte_len(), src_bytes_needed, src_tok_offset, n_tokens, n_heads, head_dim
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_seq_bf16", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_start_bytes = seq_pos_start.to_ne_bytes();
    let n_tokens_bytes = n_tokens.to_ne_bytes();
    let src_tok_offset_bytes = src_tok_offset.to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(cache)),
            (2, KernelArg::Bytes(&n_heads_bytes)),
            (3, KernelArg::Bytes(&head_dim_bytes)),
            (4, KernelArg::Bytes(&capacity_bytes)),
            (5, KernelArg::Bytes(&seq_pos_start_bytes)),
            (6, KernelArg::Bytes(&n_tokens_bytes)),
            (7, KernelArg::Bytes(&src_tok_offset_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Multi-position, all-heads KV cache copy (F32 source → F16 cache, batched prefill).
///
/// Same semantics as [`dispatch_kv_cache_copy_seq_f32`] (including
/// `src_tok_offset` source slicing and `dst_pos % capacity` ring-wrap for
/// sliding-window layers) but writes half-precision values in the cache.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_seq_f32_to_f16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    cache: &MlxBuffer,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    seq_pos_start: u32,
    n_tokens: u32,
    src_tok_offset: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    let total_src = ((src_tok_offset as u64) + (n_tokens as u64))
        * (n_heads as u64) * (head_dim as u64);
    if (src.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_seq_f32_to_f16: src has {} elements, need {}",
            src.element_count(), total_src
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_seq_f32_to_f16", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_start_bytes = seq_pos_start.to_ne_bytes();
    let n_tokens_bytes = n_tokens.to_ne_bytes();
    let src_tok_offset_bytes = src_tok_offset.to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};

    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(cache)),
            (2, KernelArg::Bytes(&n_heads_bytes)),
            (3, KernelArg::Bytes(&head_dim_bytes)),
            (4, KernelArg::Bytes(&capacity_bytes)),
            (5, KernelArg::Bytes(&seq_pos_start_bytes)),
            (6, KernelArg::Bytes(&n_tokens_bytes)),
            (7, KernelArg::Bytes(&src_tok_offset_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}
