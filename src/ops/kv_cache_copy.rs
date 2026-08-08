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
    registry.register_source(
        "kv_cache_linearize_ring_bytes",
        KV_CACHE_COPY_SHADER_SOURCE,
    );
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
    if src.dtype() != crate::dtypes::DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy: src must be BF16, got {:?}",
            src.dtype()
        )));
    }
    if cache.dtype() != crate::dtypes::DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy: cache must be BF16, got {:?}",
            cache.dtype()
        )));
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

/// ADR-040 M4 — BATCHED multi-sequence F16-K copy: writes all `n_queries`
/// decode queries' K into their own physical-slot regions of the shared
/// multi_seq F16 cache in ONE dispatch (grid.z = N), replacing the per-slot
/// host-side loop. `src` is `[N, n_heads*head_dim]` F32; `cache` is the FULL
/// multi_seq buffer `[n_seqs, n_heads, capacity, head_dim]` F16; `slot_id`/
/// `seq_pos` are `[N]` u32. Byte-identical to N single-slot calls.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_batch_f32_to_f16_batched(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    cache: &MlxBuffer,
    slot_id: &MlxBuffer,
    seq_pos: &MlxBuffer,
    n_queries: u32,
    n_heads: u32,
    head_dim: u32,
    capacity: u32,
    is_ring: bool,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || n_queries == 0 {
        return Ok(());
    }
    let pipeline = registry.get_pipeline("kv_cache_copy_batch_f32_to_f16_batched", device)?;
    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let is_ring_bytes = (if is_ring { 1u32 } else { 0u32 }).to_ne_bytes();

    use super::encode_helpers::{encode_with_args, KernelArg};
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(cache)),
            (2, KernelArg::Buffer(slot_id)),
            (3, KernelArg::Buffer(seq_pos)),
            (4, KernelArg::Bytes(&n_heads_bytes)),
            (5, KernelArg::Bytes(&head_dim_bytes)),
            (6, KernelArg::Bytes(&capacity_bytes)),
            (7, KernelArg::Bytes(&is_ring_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, n_queries as u64),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Fused single-position K + V cache copy (F32 source → F32 cache) — DECODE shape.
///
/// ADR-028 iter-145: collapses the 2-dispatch pattern (1× K, 1× V) into a single
/// dispatch. Saves one kernel launch floor (~14 µs/Apple GPU) per layer per token.
/// At gemma4 30 layers, drops 60→30 KV-copy dispatches/decode-token.
///
/// Source layouts: `[n_heads * head_dim]` flat F32 each (one token, all heads).
/// Cache layouts:  `[n_heads, capacity, head_dim]` head-major F32 each.
///
/// Each thread copies one (K, V) element pair at the same coords; results are
/// byte-identical to two `dispatch_kv_cache_copy_batch_f32` calls.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_batch_f32_kv_dual(
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
    seq_pos: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    let total_src = (n_heads as u64) * (head_dim as u64);
    if (src_k.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_batch_f32_kv_dual: src_k has {} elements but need {} (n_heads={} * head_dim={})",
            src_k.element_count(), total_src, n_heads, head_dim
        )));
    }
    if (src_v.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_batch_f32_kv_dual: src_v has {} elements but need {} (n_heads={} * head_dim={})",
            src_v.element_count(), total_src, n_heads, head_dim
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_batch_f32_kv_dual", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_bytes = seq_pos.to_ne_bytes();

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
            (7, KernelArg::Bytes(&seq_pos_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, 1),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Fused single-position K + V cache copy (F32 source → F16 cache) — DECODE shape.
///
/// Same as `dispatch_kv_cache_copy_batch_f32_kv_dual` but casts F32→F16 on write
/// for the use_f16_kv branch. Halves SDPA-read bandwidth post-write.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_batch_f32_to_f16_kv_dual(
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
    seq_pos: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 {
        return Ok(());
    }

    let total_src = (n_heads as u64) * (head_dim as u64);
    if (src_k.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_batch_f32_to_f16_kv_dual: src_k has {} elements but need {} (n_heads={} * head_dim={})",
            src_k.element_count(), total_src, n_heads, head_dim
        )));
    }
    if (src_v.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_batch_f32_to_f16_kv_dual: src_v has {} elements but need {} (n_heads={} * head_dim={})",
            src_v.element_count(), total_src, n_heads, head_dim
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_batch_f32_to_f16_kv_dual", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_bytes = seq_pos.to_ne_bytes();

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
            (7, KernelArg::Bytes(&seq_pos_bytes)),
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

/// ADR-030 iter-95: bit-exact BF16→BF16 strided cache copy from pf_k_perm
/// (head-major BF16) to bf16_xlen_cache (head-major BF16).
///
/// Used by the DFlash spec-decode xlen verify path to persist BF16 K/V
/// across rounds without the F16-intermediate precision drift that
/// iter-92/93 root-caused for Option A's non-toy coherence failures.
/// Bit-identical to what Option C reads from pf_k_perm in the same call
/// (single rounding at head_norm_rope's F32→BF16 output).
///
/// `src` layout: `[n_heads, src_seq_len, head_dim]` BF16 head-major
/// (matches fused_head_norm_rope's bf16 permuted output `pf_k_perm` /
/// `pf_v_perm`).
/// `cache` layout: `[n_heads, capacity, head_dim]` BF16 head-major.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_copy_seq_bf16_to_bf16_head_major(
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
    src_seq_len: u32,
) -> Result<()> {
    if n_heads == 0 || head_dim == 0 || n_tokens == 0 {
        return Ok(());
    }
    let total_src = (n_heads as u64) * (src_seq_len as u64) * (head_dim as u64);
    if (src.element_count() as u64) < total_src {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_seq_bf16_to_bf16_head_major: src has {} elements, need {} \
             ({} heads × {} src_seq_len × {} head_dim)",
            src.element_count(), total_src, n_heads, src_seq_len, head_dim
        )));
    }
    if src.dtype() != crate::DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_seq_bf16_to_bf16_head_major: src must be BF16, got {:?}",
            src.dtype()
        )));
    }
    if cache.dtype() != crate::DType::BF16 {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_copy_seq_bf16_to_bf16_head_major: cache must be BF16, got {:?}",
            cache.dtype()
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_copy_seq_bf16_to_bf16_head_major", device)?;

    let n_heads_bytes = n_heads.to_ne_bytes();
    let head_dim_bytes = head_dim.to_ne_bytes();
    let capacity_bytes = capacity.to_ne_bytes();
    let seq_pos_start_bytes = seq_pos_start.to_ne_bytes();
    let n_tokens_bytes = n_tokens.to_ne_bytes();
    let src_tok_offset_bytes = src_tok_offset.to_ne_bytes();
    let src_seq_len_bytes = src_seq_len.to_ne_bytes();

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
            (8, KernelArg::Bytes(&src_seq_len_bytes)),
        ],
        MTLSize::new(head_dim as u64, n_heads as u64, n_tokens as u64),
        MTLSize::new(std::cmp::min(256, head_dim as u64), 1, 1),
    );

    Ok(())
}

/// Linearize the newest rows from a head-major ring into a head-major staging
/// buffer without a CPU synchronization or model-specific dtype conversion.
///
/// Each position contains `row_bytes` opaque bytes. This makes the primitive
/// reusable for F16 K, packed/F16 V, and F32 norm payloads while retaining the
/// same `[head, capacity, row]` addressing contract.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_kv_cache_linearize_ring_bytes(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &metal::DeviceRef,
    src: &MlxBuffer,
    dst: &MlxBuffer,
    n_heads: u32,
    src_capacity: u32,
    dst_capacity: u32,
    history_len: u32,
    logical_end: u32,
    row_bytes: u32,
) -> Result<()> {
    if n_heads == 0 || history_len == 0 || row_bytes == 0 {
        return Ok(());
    }
    if src_capacity == 0 || dst_capacity == 0 {
        return Err(MlxError::InvalidArgument(
            "kv_cache_linearize_ring_bytes: capacities must be > 0".into(),
        ));
    }
    if history_len > src_capacity || history_len > dst_capacity {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_linearize_ring_bytes: history_len ({history_len}) exceeds source ({src_capacity}) or destination ({dst_capacity}) capacity"
        )));
    }
    if logical_end < history_len {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_linearize_ring_bytes: logical_end ({logical_end}) < history_len ({history_len})"
        )));
    }
    let src_required = n_heads as u64 * src_capacity as u64 * row_bytes as u64;
    let dst_required = n_heads as u64 * dst_capacity as u64 * row_bytes as u64;
    if (src.byte_len() as u64) < src_required || (dst.byte_len() as u64) < dst_required {
        return Err(MlxError::InvalidArgument(format!(
            "kv_cache_linearize_ring_bytes: buffers too small (src {} < {src_required} or dst {} < {dst_required})",
            src.byte_len(),
            dst.byte_len()
        )));
    }

    let pipeline = registry.get_pipeline("kv_cache_linearize_ring_bytes", device)?;
    let n_heads_bytes = n_heads.to_ne_bytes();
    let src_capacity_bytes = src_capacity.to_ne_bytes();
    let dst_capacity_bytes = dst_capacity.to_ne_bytes();
    let history_len_bytes = history_len.to_ne_bytes();
    let logical_end_bytes = logical_end.to_ne_bytes();
    let row_bytes_bytes = row_bytes.to_ne_bytes();
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(src)),
            (1, KernelArg::Buffer(dst)),
            (2, KernelArg::Bytes(&n_heads_bytes)),
            (3, KernelArg::Bytes(&src_capacity_bytes)),
            (4, KernelArg::Bytes(&dst_capacity_bytes)),
            (5, KernelArg::Bytes(&history_len_bytes)),
            (6, KernelArg::Bytes(&logical_end_bytes)),
            (7, KernelArg::Bytes(&row_bytes_bytes)),
        ],
        MTLSize::new(row_bytes as u64, n_heads as u64, history_len as u64),
        MTLSize::new(std::cmp::min(256, row_bytes as u64), 1, 1),
    );
    Ok(())
}

#[cfg(test)]
mod linearize_tests {
    use super::*;
    use crate::{DType, MlxDevice};

    #[test]
    fn linearize_ring_bytes_preserves_head_major_chronology() {
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: no Metal device: {error}");
                return;
            }
        };
        let mut registry = KernelRegistry::new();
        let n_heads = 2u32;
        let src_capacity = 5u32;
        let dst_capacity = 6u32;
        let row_bytes = 3u32;
        let history_len = 4u32;
        let logical_end = 8u32;

        let src_len = (n_heads * src_capacity * row_bytes) as usize;
        let mut src = device
            .alloc_buffer(src_len, DType::U8, vec![src_len])
            .expect("source");
        for head in 0..n_heads {
            for slot in 0..src_capacity {
                for byte in 0..row_bytes {
                    let index = ((head * src_capacity + slot) * row_bytes + byte) as usize;
                    src.as_mut_slice::<u8>().unwrap()[index] =
                        (head * 80 + slot * 10 + byte) as u8;
                }
            }
        }
        let dst_len = (n_heads * dst_capacity * row_bytes) as usize;
        let dst = device
            .alloc_buffer(dst_len, DType::U8, vec![dst_len])
            .expect("destination");

        let mut encoder = device.command_encoder().expect("encoder");
        dispatch_kv_cache_linearize_ring_bytes(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &src,
            &dst,
            n_heads,
            src_capacity,
            dst_capacity,
            history_len,
            logical_end,
            row_bytes,
        )
        .expect("dispatch");
        encoder.commit_and_wait().expect("commit");

        let actual = dst.as_slice::<u8>().expect("destination slice");
        for head in 0..n_heads {
            for history_index in 0..history_len {
                let logical_position = logical_end - history_len + history_index;
                let source_slot = logical_position % src_capacity;
                for byte in 0..row_bytes {
                    let dst_index =
                        ((head * dst_capacity + history_index) * row_bytes + byte) as usize;
                    let expected = (head * 80 + source_slot * 10 + byte) as u8;
                    assert_eq!(actual[dst_index], expected);
                }
            }
        }
    }
}
