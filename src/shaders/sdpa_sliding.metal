// Scaled Dot-Product Attention with Sliding Window Mask — Metal kernel.
//
// Computes: softmax(Q * K^T / sqrt(head_dim)) * V
// with a sliding window mask that restricts attention to the last
// `window_size` key positions relative to each query position.
//
// Uses single-pass online softmax (Milakov & Gimelshein 2018) to avoid
// reading the K cache multiple times.
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
    uint kv_seq_len;    // key/value sequence length (valid positions)
    uint window_size;   // sliding window size
    float scale;        // attention score scaling factor (e.g. 1/sqrt(head_dim) or 1.0)
    uint kv_capacity;   // stride between KV heads (in positions); >= kv_seq_len
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
    const uint n_heads      = params->n_heads;
    const uint n_kv_heads   = params->n_kv_heads;
    const uint head_dim     = params->head_dim;
    const uint seq_len      = params->seq_len;
    const uint kv_seq_len   = params->kv_seq_len;
    const uint window_size  = params->window_size;
    const uint kv_capacity  = params->kv_capacity;  // stride between KV heads

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

    // Base offsets.  Q is packed [batch, heads, seq, head_dim].
    // KV cache uses kv_capacity as stride between heads (>= kv_seq_len).
    const uint q_base = batch_idx * (n_heads * seq_len * head_dim)
                      + head_idx * (seq_len * head_dim)
                      + q_pos * head_dim;

    const uint k_head_base = batch_idx * (n_kv_heads * kv_capacity * head_dim)
                           + kv_head_idx * (kv_capacity * head_dim);

    const uint v_head_base = k_head_base;

    const uint o_base = q_base;

    const float scale = params->scale;

    // Sliding window + causal mask bounds.
    // When seq_len < kv_seq_len (decode mode with KV cache), the query
    // positions map to the END of the full sequence. q_pos=0 corresponds
    // to absolute position (kv_seq_len - seq_len).
    const uint abs_pos = kv_seq_len - seq_len + q_pos;
    // Causal: k_pos <= abs_pos, so max key is min(abs_pos + 1, kv_seq_len).
    const uint causal_max_k = min(abs_pos + 1, kv_seq_len);

    // Compute the start of the sliding window.
    // If abs_pos < window_size, the window starts at 0 (no underflow with uint).
    const uint window_start = (abs_pos >= window_size) ? (abs_pos - window_size) : 0;

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

    // ---- Single-pass online softmax (Milakov & Gimelshein 2018) ----

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    float acc[512];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    for (uint k_pos = window_start; k_pos < causal_max_k; k_pos++) {
        // Compute Q . K^T for this key position.
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;

        // Online softmax update.
        float old_max = running_max;
        running_max = max(running_max, score);

        // Correction factor: rescale previous accumulations.
        // First iteration: old_max=-inf, correction=exp(-inf - finite)=0.
        float correction = exp(old_max - running_max);

        running_sum = running_sum * correction + exp(score - running_max);

        float weight = exp(score - running_max);
        const uint v_offset = v_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction + weight * float(V[v_offset + d]);
        }
    }

    // Final normalization.
    float inv_sum = 1.0f / running_sum;

    // Write output.
    for (uint d = 0; d < head_dim; d++) {
        O[o_base + d] = acc[d] * inv_sum;
    }
}

// --------------------------------------------------------------------------
// sdpa_sliding_bf16 — Sliding-window attention for bfloat16 tensors.
//
// Same online softmax algorithm as sdpa_sliding, but reads/writes bfloat16.
// All accumulation and intermediate computation stays in float32.
// --------------------------------------------------------------------------
kernel void sdpa_sliding_bf16(
    device const bfloat *Q          [[buffer(0)]],
    device const bfloat *K          [[buffer(1)]],
    device const bfloat *V          [[buffer(2)]],
    device bfloat       *O          [[buffer(3)]],
    device const SdpaSlidingParams *params [[buffer(4)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]]
) {
    const uint n_heads      = params->n_heads;
    const uint n_kv_heads   = params->n_kv_heads;
    const uint head_dim     = params->head_dim;
    const uint seq_len      = params->seq_len;
    const uint kv_seq_len   = params->kv_seq_len;
    const uint window_size  = params->window_size;
    const uint kv_capacity  = params->kv_capacity;

    const uint batch_idx   = tgid.x;
    const uint head_idx    = tgid.y;
    const uint tile_idx    = tgid.z;

    const uint q_pos = tile_idx * TILE_Q + tid;

    if (q_pos >= seq_len) {
        return;
    }

    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kv_head_idx  = head_idx / heads_per_kv;

    const uint q_base = batch_idx * (n_heads * seq_len * head_dim)
                      + head_idx * (seq_len * head_dim)
                      + q_pos * head_dim;

    const uint k_head_base = batch_idx * (n_kv_heads * kv_capacity * head_dim)
                           + kv_head_idx * (kv_capacity * head_dim);

    const uint v_head_base = k_head_base;
    const uint o_base = q_base;

    const float scale = params->scale;

    const uint abs_pos = kv_seq_len - seq_len + q_pos;
    const uint causal_max_k = min(abs_pos + 1, kv_seq_len);
    const uint window_start = (abs_pos >= window_size) ? (abs_pos - window_size) : 0;

    if (window_start >= causal_max_k) {
        for (uint d = 0; d < head_dim; d++) {
            O[o_base + d] = bfloat(0.0f);
        }
        return;
    }

    // Single-pass online softmax.
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    float acc[512];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    for (uint k_pos = window_start; k_pos < causal_max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;

        float old_max = running_max;
        running_max = max(running_max, score);

        float correction = exp(old_max - running_max);
        running_sum = running_sum * correction + exp(score - running_max);

        float weight = exp(score - running_max);
        const uint v_offset = v_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction + weight * float(V[v_offset + d]);
        }
    }

    float inv_sum = 1.0f / running_sum;

    // Write output as bfloat16.
    for (uint d = 0; d < head_dim; d++) {
        O[o_base + d] = bfloat(acc[d] * inv_sum);
    }
}
