#include <metal_stdlib>
using namespace metal;

/// Copy new K or V data from a source GPU buffer into the KV cache buffer
/// at the correct write position, with optional modulo wrapping for sliding
/// window (ring buffer) caches.
///
/// Grid: 1D, total_elements = n_new * row_size
/// Each thread copies one element.
kernel void kv_cache_copy(
    device const half* src       [[buffer(0)]],   // source K or V [n_new, row_size]
    device half*       cache     [[buffer(1)]],   // destination cache buffer
    constant uint&     write_pos [[buffer(2)]],   // starting write position in cache
    constant uint&     row_size  [[buffer(3)]],   // n_kv_heads * head_dim
    constant uint&     n_new     [[buffer(4)]],   // number of new tokens
    constant uint&     cache_cap [[buffer(5)]],   // window size (sliding) or max_seq_len (global)
    constant uint&     is_sliding [[buffer(6)]],  // 1 = use modulo wrapping, 0 = linear
    uint tid [[thread_position_in_grid]]
) {
    uint total_elements = n_new * row_size;
    if (tid >= total_elements) return;

    uint token_idx = tid / row_size;
    uint elem_idx  = tid % row_size;

    uint dst_pos = is_sliding
        ? ((write_pos + token_idx) % cache_cap)
        : (write_pos + token_idx);

    cache[dst_pos * row_size + elem_idx] = src[token_idx * row_size + elem_idx];
}

/// Float32 variant of kv_cache_copy for F32 KV caches.
///
/// Identical logic to the half variant but operates on float data.
/// Used when the activation pipeline is F32 throughout (no bf16 casting).
kernel void kv_cache_copy_f32(
    device const float* src       [[buffer(0)]],   // source K or V [n_new, row_size]
    device float*       cache     [[buffer(1)]],   // destination cache buffer
    constant uint&     write_pos [[buffer(2)]],   // starting write position in cache
    constant uint&     row_size  [[buffer(3)]],   // n_kv_heads * head_dim
    constant uint&     n_new     [[buffer(4)]],   // number of new tokens
    constant uint&     cache_cap [[buffer(5)]],   // window size (sliding) or max_seq_len (global)
    constant uint&     is_sliding [[buffer(6)]],  // 1 = use modulo wrapping, 0 = linear
    uint tid [[thread_position_in_grid]]
) {
    uint total_elements = n_new * row_size;
    if (tid >= total_elements) return;

    uint token_idx = tid / row_size;
    uint elem_idx  = tid % row_size;

    uint dst_pos = is_sliding
        ? ((write_pos + token_idx) % cache_cap)
        : (write_pos + token_idx);

    cache[dst_pos * row_size + elem_idx] = src[token_idx * row_size + elem_idx];
}

/// Batched KV cache copy — copies ALL heads in one dispatch.
///
/// Source layout: [n_heads * head_dim] flat (one token, all heads).
/// Cache layout: [n_heads, capacity, head_dim] head-major.
///
/// Grid: 2D — x=element within head (head_dim), y=head index (n_heads).
/// Replaces n_heads separate kv_cache_copy_f32 dispatches with 1.
kernel void kv_cache_copy_batch_f32(
    device const float* src       [[buffer(0)]],   // [n_heads * head_dim] flat
    device float*       cache     [[buffer(1)]],   // [n_heads, capacity, head_dim]
    constant uint&     n_heads   [[buffer(2)]],   // number of KV heads
    constant uint&     head_dim  [[buffer(3)]],   // elements per head
    constant uint&     capacity  [[buffer(4)]],   // cache capacity (ring buffer size)
    constant uint&     seq_pos   [[buffer(5)]],   // write position (already wrapped)
    uint2 tid [[thread_position_in_grid]]          // x=elem, y=head
) {
    uint elem = tid.x;
    uint head = tid.y;
    if (head >= n_heads || elem >= head_dim) return;

    uint src_idx = head * head_dim + elem;
    uint dst_idx = head * capacity * head_dim + seq_pos * head_dim + elem;
    cache[dst_idx] = src[src_idx];
}

/// Batched KV cache copy with F32→F16 cast — copies ALL heads in one dispatch.
///
/// Source layout: [n_heads * head_dim] flat F32 (one token, all heads).
/// Cache layout: [n_heads, capacity, head_dim] head-major F16.
///
/// Casts float → half on write, halving cache memory bandwidth for SDPA reads.
/// Reference: llama.cpp stores KV cache in F16 for bandwidth-bound decode SDPA.
///
/// Grid: 2D — x=element within head (head_dim), y=head index (n_heads).
kernel void kv_cache_copy_batch_f32_to_f16(
    device const float* src       [[buffer(0)]],   // [n_heads * head_dim] flat F32
    device half*        cache     [[buffer(1)]],   // [n_heads, capacity, head_dim] F16
    constant uint&     n_heads   [[buffer(2)]],   // number of KV heads
    constant uint&     head_dim  [[buffer(3)]],   // elements per head
    constant uint&     capacity  [[buffer(4)]],   // cache capacity (ring buffer size)
    constant uint&     seq_pos   [[buffer(5)]],   // write position (already wrapped)
    uint2 tid [[thread_position_in_grid]]          // x=elem, y=head
) {
    uint elem = tid.x;
    uint head = tid.y;
    if (head >= n_heads || elem >= head_dim) return;

    uint src_idx = head * head_dim + elem;
    uint dst_idx = head * capacity * head_dim + seq_pos * head_dim + elem;
    cache[dst_idx] = half(src[src_idx]);
}

/// Multi-position, all-heads KV cache copy (F32 source → F32 cache).
///
/// Source layout: [n_src_tokens, n_heads, head_dim] — token-major, from
/// head_norm+RoPE. The caller selects the surviving slice of the source
/// via `src_tok_offset`; only tokens
/// `[src_tok_offset, src_tok_offset + n_tokens)` are read.
/// Cache layout:  [n_heads, capacity, head_dim] — head-major dense_kvs.
/// Writes absolute positions `[seq_pos_start, seq_pos_start + n_tokens)`
/// into cache slots `dst_pos % capacity`.
///
/// Global (non-sliding) layer contract: caller ensures
/// `seq_pos_start + n_tokens <= capacity` so `dst_pos % capacity == dst_pos`
/// and behaviour is linear/no-wrap.
///
/// Sliding-window contract: caller sets `capacity = sliding_window`,
/// `n_tokens = min(seq_len, capacity)`, `src_tok_offset = seq_len - n_tokens`,
/// `seq_pos_start = seq_len - n_tokens`. This writes the last `n_tokens`
/// source tokens into modular slots, each exactly once — no intra-dispatch
/// race. Decode side reads via `ring_start = write_pos % capacity`
/// (`hf2q:src/serve/forward_mlx.rs`), consistent with this layout.
///
/// Grid: 3D — x=elem within head, y=head, z=token.
kernel void kv_cache_copy_seq_f32(
    device const float* src       [[buffer(0)]],   // [n_src_tokens, n_heads, head_dim] F32
    device float*       cache     [[buffer(1)]],   // [n_heads, capacity, head_dim] F32
    constant uint&     n_heads   [[buffer(2)]],
    constant uint&     head_dim  [[buffer(3)]],
    constant uint&     capacity  [[buffer(4)]],
    constant uint&     seq_pos_start [[buffer(5)]],
    constant uint&     n_tokens  [[buffer(6)]],
    constant uint&     src_tok_offset [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint head = tid.y;
    uint tok  = tid.z;
    if (head >= n_heads || elem >= head_dim || tok >= n_tokens) return;

    uint src_tok = src_tok_offset + tok;
    uint src_idx = src_tok * (n_heads * head_dim) + head * head_dim + elem;
    uint dst_pos = seq_pos_start + tok;
    uint slot    = dst_pos % capacity;
    uint dst_idx = head * capacity * head_dim + slot * head_dim + elem;
    cache[dst_idx] = src[src_idx];
}

/// Fused K + V multi-position cache copy (F32 source → F32 cache).
///
/// Combines two kv_cache_copy_seq_f32 dispatches (one for K, one for V)
/// into a single kernel.  The two streams have identical shape, identical
/// dst layout (n_heads × capacity × head_dim), and shared metadata
/// (capacity, seq_pos_start, n_tokens, src_tok_offset) — so we can merge
/// them with no extra register pressure.  Each thread copies one (K, V)
/// element pair at the same coordinates.
///
/// Wave P4.11 — saves one dispatch per layer (30/prefill on Gemma 4).
///
/// Grid: same as kv_cache_copy_seq_f32 (3D — x=elem, y=head, z=token).
kernel void kv_cache_copy_seq_f32_kv_dual(
    device const float* src_k         [[buffer(0)]],
    device const float* src_v         [[buffer(1)]],
    device float*       cache_k       [[buffer(2)]],
    device float*       cache_v       [[buffer(3)]],
    constant uint&     n_heads        [[buffer(4)]],
    constant uint&     head_dim       [[buffer(5)]],
    constant uint&     capacity       [[buffer(6)]],
    constant uint&     seq_pos_start  [[buffer(7)]],
    constant uint&     n_tokens       [[buffer(8)]],
    constant uint&     src_tok_offset [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint head = tid.y;
    uint tok  = tid.z;
    if (head >= n_heads || elem >= head_dim || tok >= n_tokens) return;

    uint src_tok = src_tok_offset + tok;
    uint src_idx = src_tok * (n_heads * head_dim) + head * head_dim + elem;
    uint dst_pos = seq_pos_start + tok;
    uint slot    = dst_pos % capacity;
    uint dst_idx = head * capacity * head_dim + slot * head_dim + elem;
    cache_k[dst_idx] = src_k[src_idx];
    cache_v[dst_idx] = src_v[src_idx];
}

/// Fused K + V multi-position cache copy (F32 source → F16 cache).
///
/// Same as kv_cache_copy_seq_f32_kv_dual but casts to half on write —
/// for the use_f16_kv branch.  Wave P4.11.
kernel void kv_cache_copy_seq_f32_to_f16_kv_dual(
    device const float* src_k         [[buffer(0)]],
    device const float* src_v         [[buffer(1)]],
    device half*        cache_k       [[buffer(2)]],
    device half*        cache_v       [[buffer(3)]],
    constant uint&     n_heads        [[buffer(4)]],
    constant uint&     head_dim       [[buffer(5)]],
    constant uint&     capacity       [[buffer(6)]],
    constant uint&     seq_pos_start  [[buffer(7)]],
    constant uint&     n_tokens       [[buffer(8)]],
    constant uint&     src_tok_offset [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint head = tid.y;
    uint tok  = tid.z;
    if (head >= n_heads || elem >= head_dim || tok >= n_tokens) return;

    uint src_tok = src_tok_offset + tok;
    uint src_idx = src_tok * (n_heads * head_dim) + head * head_dim + elem;
    uint dst_pos = seq_pos_start + tok;
    uint slot    = dst_pos % capacity;
    uint dst_idx = head * capacity * head_dim + slot * head_dim + elem;
    cache_k[dst_idx] = half(src_k[src_idx]);
    cache_v[dst_idx] = half(src_v[src_idx]);
}

/// Multi-position, all-heads KV cache copy (BF16 source → F32 cache).
///
/// Same layout/semantics as kv_cache_copy_seq_f32 — including `src_tok_offset`
/// source slicing and `dst_pos % capacity` ring-wrap for sliding-window layers
/// — but reads bfloat16 from the source and promotes to float32 on write.
///
/// Used in the Phase 2 bf16 activation path where pf_k_normed and pf_v_normed
/// are bf16 but the KV cache (dense_kvs) remains f32 for decode SDPA precision.
///
/// Grid: 3D — x=elem within head, y=head, z=token.
kernel void kv_cache_copy_seq_bf16(
    device const bfloat* src              [[buffer(0)]],   // [n_src_tokens, n_heads, head_dim] BF16
    device float*        cache            [[buffer(1)]],   // [n_heads, capacity, head_dim] F32
    constant uint&       n_heads          [[buffer(2)]],
    constant uint&       head_dim         [[buffer(3)]],
    constant uint&       capacity         [[buffer(4)]],
    constant uint&       seq_pos_start    [[buffer(5)]],
    constant uint&       n_tokens         [[buffer(6)]],
    constant uint&       src_tok_offset   [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint head = tid.y;
    uint tok  = tid.z;
    if (head >= n_heads || elem >= head_dim || tok >= n_tokens) return;

    uint src_tok = src_tok_offset + tok;
    uint src_idx = src_tok * (n_heads * head_dim) + head * head_dim + elem;
    uint dst_pos = seq_pos_start + tok;
    uint slot    = dst_pos % capacity;
    uint dst_idx = head * capacity * head_dim + slot * head_dim + elem;
    // Promote bf16 → f32 on write to keep cache precision.
    cache[dst_idx] = static_cast<float>(src[src_idx]);
}

/// Multi-position, all-heads KV cache copy (F32 source → F16 cache).
///
/// Same layout/semantics as kv_cache_copy_seq_f32 (including `src_tok_offset`
/// source slicing and `dst_pos % capacity` ring-wrap for sliding layers)
/// but casts to half on write.
kernel void kv_cache_copy_seq_f32_to_f16(
    device const float* src       [[buffer(0)]],   // [n_src_tokens, n_heads, head_dim] F32
    device half*        cache     [[buffer(1)]],   // [n_heads, capacity, head_dim] F16
    constant uint&     n_heads   [[buffer(2)]],
    constant uint&     head_dim  [[buffer(3)]],
    constant uint&     capacity  [[buffer(4)]],
    constant uint&     seq_pos_start [[buffer(5)]],
    constant uint&     n_tokens  [[buffer(6)]],
    constant uint&     src_tok_offset [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint head = tid.y;
    uint tok  = tid.z;
    if (head >= n_heads || elem >= head_dim || tok >= n_tokens) return;

    uint src_tok = src_tok_offset + tok;
    uint src_idx = src_tok * (n_heads * head_dim) + head * head_dim + elem;
    uint dst_pos = seq_pos_start + tok;
    uint slot    = dst_pos % capacity;
    uint dst_idx = head * capacity * head_dim + slot * head_dim + elem;
    cache[dst_idx] = half(src[src_idx]);
}
