// Scaled Dot-Product Attention with Sliding Window Mask — Metal kernel.
//
// Computes: softmax(Q * K^T / sqrt(head_dim)) * V
// with a sliding window mask that restricts attention to the last
// `window_size` key positions relative to each query position.
//
// For query position q_pos, keys at positions k_pos where:
//   k_pos < q_pos - window_size    (outside sliding window)
//   k_pos > q_pos                  (causal mask)
// are masked to -inf before softmax.
//
// Effective attention window for q_pos: [max(0, q_pos - window_size), q_pos]
//
// Supports grouped-query attention (GQA) where n_heads > n_kv_heads.
//
// Layout assumptions (same as sdpa.metal):
//   Q: [batch, n_heads, seq_len, head_dim]
//   K: [batch, n_kv_heads, kv_seq_len, head_dim]
//   V: [batch, n_kv_heads, kv_seq_len, head_dim]
//   O: [batch, n_heads, seq_len, head_dim]

#include <metal_stdlib>
using namespace metal;

// Parameters passed via buffer binding.
struct SdpaSlidingParams {
    uint n_heads;       // number of query heads
    uint n_kv_heads;    // number of key/value heads
    uint head_dim;      // dimension per head
    uint seq_len;       // query sequence length
    uint kv_seq_len;    // key/value sequence length
    uint window_size;   // sliding window size
};

// Tile size: number of query positions per threadgroup.
constant uint TILE_Q = 32;

kernel void sdpa_sliding(
    device const float *Q          [[buffer(0)]],
    device const float *K          [[buffer(1)]],
    device const float *V          [[buffer(2)]],
    device float       *O          [[buffer(3)]],
    device const SdpaSlidingParams *params [[buffer(4)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]]
) {
    // Unpack parameters.
    const uint n_heads     = params->n_heads;
    const uint n_kv_heads  = params->n_kv_heads;
    const uint head_dim    = params->head_dim;
    const uint seq_len     = params->seq_len;
    const uint kv_seq_len  = params->kv_seq_len;
    const uint window_size = params->window_size;

    // Threadgroup grid: (batch, head, query_tile).
    const uint batch_idx   = tgid.x;
    const uint head_idx    = tgid.y;
    const uint tile_idx    = tgid.z;

    // Which query position this thread handles within the tile.
    const uint q_pos = tile_idx * TILE_Q + tid;

    // Bounds check.
    if (q_pos >= seq_len) {
        return;
    }

    // GQA head broadcast.
    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kv_head_idx  = head_idx / heads_per_kv;

    // Base offsets.
    const uint q_base = batch_idx * (n_heads * seq_len * head_dim)
                      + head_idx * (seq_len * head_dim)
                      + q_pos * head_dim;

    const uint k_head_base = batch_idx * (n_kv_heads * kv_seq_len * head_dim)
                           + kv_head_idx * (kv_seq_len * head_dim);

    const uint v_head_base = k_head_base;

    const uint o_base = q_base;

    const float scale = rsqrt(float(head_dim));

    // Sliding window + causal mask bounds.
    // Causal: k_pos <= q_pos, so max key is min(q_pos + 1, kv_seq_len).
    // Sliding: k_pos >= q_pos - window_size (but q_pos - window_size may underflow).
    const uint causal_max_k = min(q_pos + 1, kv_seq_len);

    // Compute the start of the sliding window.
    // If q_pos < window_size, the window starts at 0 (no underflow with uint).
    const uint window_start = (q_pos >= window_size) ? (q_pos - window_size) : 0;

    // Effective range: [window_start, causal_max_k)
    // If window_start >= causal_max_k, there are no valid keys (shouldn't happen
    // in normal usage but handle gracefully).
    if (window_start >= causal_max_k) {
        // No keys to attend to; write zeros.
        for (uint d = 0; d < head_dim; d++) {
            O[o_base + d] = 0.0f;
        }
        return;
    }

    // ---- Pass 1: Find max score for numerical stability ----

    float max_score = -INFINITY;

    for (uint k_pos = window_start; k_pos < causal_max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        max_score = max(max_score, score);
    }

    // ---- Pass 2a: Compute sum of exp(score - max) ----

    float sum_exp = 0.0f;

    for (uint k_pos = window_start; k_pos < causal_max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        sum_exp += exp(score - max_score);
    }

    // ---- Pass 2b: Accumulate weighted V ----

    float acc[512];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    const float inv_sum = 1.0f / sum_exp;

    for (uint k_pos = window_start; k_pos < causal_max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        float weight = exp(score - max_score) * inv_sum;

        const uint v_offset = v_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            acc[d] += weight * float(V[v_offset + d]);
        }
    }

    // Write output.
    for (uint d = 0; d < head_dim; d++) {
        O[o_base + d] = acc[d];
    }
}
