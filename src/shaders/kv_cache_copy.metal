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
