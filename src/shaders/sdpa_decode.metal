//! SDPA decode kernel — F32 Q/K/V, SIMD-vectorized, single query token.
//!
//! Optimized for the decode path (seq_len=1) where one token attends over
//! the full KV cache. Each threadgroup handles one query head; 32 threads
//! (one SIMD group) cooperate on the QK dot products via simd_sum.
//!
//! Compared to the generic sdpa.metal kernel:
//!   - Q is stored in F32 registers (no half-precision quantization)
//!   - QK dot product uses simd_sum across 32 threads for vectorization
//!   - Processes KV cache in chunks of 32 positions per iteration
//!   - Each thread owns `head_dim/32` Q/K/V elements (8 for head_dim=256)
//!
//! Supports head_dim = 128, 256, 512.
//!
//! Grid: (n_kv_groups, n_heads_per_kv, 1)  — one TG per query head
//! TG size: (32, 1, 1)  — one SIMD group
//!
//! Q layout: [n_heads, head_dim]  F32 (seq=1, so seq dim is dropped)
//! K layout: [n_kv_heads, kv_capacity, head_dim]  F32
//! V layout: [n_kv_heads, kv_capacity, head_dim]  F32
//! O layout: [n_heads, head_dim]  F32

#include <metal_stdlib>
using namespace metal;

struct SdpaDecodeParams {
    uint n_heads;        // total Q heads
    uint n_kv_heads;     // KV heads (GQA: n_kv_heads <= n_heads)
    uint head_dim;       // must be 128, 256, or 512
    uint kv_seq_len;     // valid KV positions in the cache
    uint kv_capacity;    // stride between KV heads (>= kv_seq_len)
    float scale;         // attention scale (typically 1/sqrt(head_dim))
};

// -------------------------------------------------------------------------
// Main decode SDPA kernel
// -------------------------------------------------------------------------
// Grid: (n_heads, 1, 1)   Threadgroup: (32, 1, 1)
//
// Each threadgroup = one SIMD group = 32 threads handles one query head.
// Threads cooperate on QK dot products via simd_sum.
// -------------------------------------------------------------------------
kernel void sdpa_decode(
    device const float  *Q      [[buffer(0)]],  // [n_heads, head_dim]
    device const float  *K      [[buffer(1)]],  // [n_kv_heads, kv_cap, head_dim]
    device const float  *V      [[buffer(2)]],  // [n_kv_heads, kv_cap, head_dim]
    device float        *O      [[buffer(3)]],  // [n_heads, head_dim]
    constant SdpaDecodeParams &p [[buffer(4)]],
    uint  head_idx   [[threadgroup_position_in_grid]],
    ushort lane      [[thread_index_in_threadgroup]]   // 0..31
) {
    const uint  n_heads    = p.n_heads;
    const uint  n_kv_heads = p.n_kv_heads;
    const uint  head_dim   = p.head_dim;
    const uint  kv_seq_len = p.kv_seq_len;
    const uint  kv_cap     = p.kv_capacity;
    const float scale      = p.scale;

    if (head_idx >= n_heads) return;

    // GQA: map Q head → KV head
    const uint kv_head = head_idx * n_kv_heads / n_heads;

    // Each thread owns `elem_per_lane = head_dim / 32` elements.
    // head_dim must be a multiple of 32.
    const uint EPL = head_dim / 32;   // elements per lane

    // Pointers to this head's Q, K, V, O.
    const uint q_off = head_idx * head_dim + lane * EPL;
    const uint kv_off = kv_head * kv_cap * head_dim;
    const uint o_off = head_idx * head_dim + lane * EPL;

    // Load Q elements into registers (EPL elements per lane).
    float q_reg[16];  // max EPL = 512/32 = 16
    for (uint e = 0; e < EPL; e++) {
        q_reg[e] = Q[q_off + e];
    }

    // ---- Online softmax over KV positions ----
    float running_max  = -INFINITY;
    float running_sum  = 0.0f;
    float acc[16];    // output accumulator, EPL elements per lane
    for (uint e = 0; e < EPL; e++) acc[e] = 0.0f;

    for (uint k_pos = 0; k_pos < kv_seq_len; k_pos++) {
        // Base offset for K at position k_pos.
        const uint kb = kv_off + k_pos * head_dim + lane * EPL;

        // Partial QK dot product for this lane's EPL elements.
        float partial = 0.0f;
        for (uint e = 0; e < EPL; e++) {
            partial += q_reg[e] * K[kb + e];
        }

        // Reduce across all 32 lanes to get the full dot product.
        float dot = simd_sum(partial) * scale;

        // Online softmax update.
        float old_max = running_max;
        running_max = max(running_max, dot);
        float correction = exp(old_max - running_max);
        running_sum = running_sum * correction + exp(dot - running_max);
        float weight = exp(dot - running_max);

        // Rescale and accumulate V.
        for (uint e = 0; e < EPL; e++) {
            acc[e] = acc[e] * correction + weight * V[kv_off + k_pos * head_dim + lane * EPL + e];
        }
    }

    // Normalize by running_sum.
    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint e = 0; e < EPL; e++) {
        O[o_off + e] = acc[e] * inv_sum;
    }
}
