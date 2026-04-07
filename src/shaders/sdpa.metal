// Scaled Dot-Product Attention (SDPA) Metal kernel.
//
// Computes: softmax(Q * K^T / sqrt(head_dim)) * V
//
// Supports grouped-query attention (GQA) where n_heads > n_kv_heads.
// Each KV head serves (n_heads / n_kv_heads) Q heads via broadcast.
//
// Layout assumptions:
//   Q: [batch, n_heads, seq_len, head_dim]        (contiguous, row-major)
//   K: [batch, n_kv_heads, kv_seq_len, head_dim]  (contiguous, row-major)
//   V: [batch, n_kv_heads, kv_seq_len, head_dim]  (contiguous, row-major)
//   O: [batch, n_heads, seq_len, head_dim]         (contiguous, row-major)
//
// Tiling: each threadgroup processes one (batch, head, query_tile) combination.
// Within a threadgroup, each thread handles one query position from the tile
// and iterates over all key positions sequentially.

#include <metal_stdlib>
using namespace metal;

// Parameters passed via buffer binding.
struct SdpaParams {
    uint n_heads;       // number of query heads
    uint n_kv_heads;    // number of key/value heads
    uint head_dim;      // dimension per head
    uint seq_len;       // query sequence length
    uint kv_seq_len;    // key/value sequence length
};

// Tile size: number of query positions per threadgroup.
constant uint TILE_Q = 32;

kernel void sdpa(
    device const float *Q          [[buffer(0)]],
    device const float *K          [[buffer(1)]],
    device const float *V          [[buffer(2)]],
    device float       *O          [[buffer(3)]],
    device const SdpaParams *params [[buffer(4)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]]
) {
    // Unpack parameters.
    const uint n_heads     = params->n_heads;
    const uint n_kv_heads  = params->n_kv_heads;
    const uint head_dim    = params->head_dim;
    const uint seq_len     = params->seq_len;
    const uint kv_seq_len  = params->kv_seq_len;

    // Threadgroup grid: (batch, head, query_tile).
    const uint batch_idx   = tgid.x;
    const uint head_idx    = tgid.y;
    const uint tile_idx    = tgid.z;

    // Which query position this thread handles within the tile.
    const uint q_pos = tile_idx * TILE_Q + tid;

    // Bounds check: this thread may be past the end of the sequence.
    if (q_pos >= seq_len) {
        return;
    }

    // GQA head broadcast: map Q head index to KV head index.
    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kv_head_idx  = head_idx / heads_per_kv;

    // Compute base offsets into the contiguous [batch, heads, seq, head_dim] layout.
    // Q offset: batch_idx * (n_heads * seq_len * head_dim) + head_idx * (seq_len * head_dim) + q_pos * head_dim
    const uint q_base = batch_idx * (n_heads * seq_len * head_dim)
                      + head_idx * (seq_len * head_dim)
                      + q_pos * head_dim;

    // K offset base: batch_idx * (n_kv_heads * kv_seq_len * head_dim) + kv_head_idx * (kv_seq_len * head_dim)
    const uint k_head_base = batch_idx * (n_kv_heads * kv_seq_len * head_dim)
                           + kv_head_idx * (kv_seq_len * head_dim);

    // V has the same layout as K.
    const uint v_head_base = k_head_base;

    // Output offset: same layout as Q.
    const uint o_base = q_base;

    // Scale factor: 1 / sqrt(head_dim).
    const float scale = rsqrt(float(head_dim));

    // ---- Pass 1: Compute attention scores and find max (for numerical stability) ----

    // We apply a causal mask. When seq_len < kv_seq_len (decode mode with
    // KV cache), the query positions map to the END of the full sequence.
    // q_pos=0 corresponds to absolute position (kv_seq_len - seq_len).
    // The causal constraint: attend to k_pos <= abs_pos.
    const uint abs_pos = kv_seq_len - seq_len + q_pos;
    const uint max_k = min(abs_pos + 1, kv_seq_len);

    float max_score = -INFINITY;

    // First pass: find the maximum score.
    for (uint k_pos = 0; k_pos < max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        max_score = max(max_score, score);
    }

    // ---- Pass 2: Compute exp(score - max) and sum, accumulate weighted V ----

    float sum_exp = 0.0f;
    // Accumulate output in f32 for numerical stability.
    // We use a local array for head_dim values. For large head_dim (512),
    // this fits in thread-local memory.
    // Metal supports variable-length arrays on stack, but for safety we
    // accumulate in a streaming fashion: two passes over K.

    // Actually, we need V weighted sum. Let's do it in a single pass after
    // computing softmax weights. Since kv_seq_len can be large, we do:
    // Pass 2a: compute sum_exp
    // Pass 2b: accumulate output

    // Pass 2a: sum of exp(score - max)
    for (uint k_pos = 0; k_pos < max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        sum_exp += exp(score - max_score);
    }

    // Pass 2b: accumulate weighted V values.
    // For each output dimension d, compute sum over k_pos of softmax_weight * V[k_pos, d].
    // We iterate over output dimensions in the outer loop and key positions in
    // the inner loop. This is cache-friendly for V access since we walk
    // contiguously along head_dim for each k_pos.
    //
    // However, for large kv_seq_len this means re-computing Q*K^T per output dim.
    // Instead, iterate key positions in the outer loop and accumulate into the
    // output dimensions. We store the partial output in thread-local registers.
    //
    // Strategy: iterate k_pos once, for each compute the softmax weight,
    // then scatter-add weight * V[k_pos, :] into a thread-local output accumulator.
    // We limit head_dim to at most 512 (Gemma 4 max). Metal thread local memory
    // handles this fine.

    // Zero-initialize output accumulator.
    // Using a fixed-size array that covers the maximum head_dim we support.
    // The actual loop only iterates up to `head_dim`.
    float acc[512];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    const float inv_sum = 1.0f / sum_exp;

    for (uint k_pos = 0; k_pos < max_k; k_pos++) {
        // Recompute attention score (avoids storing all scores in memory).
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        float weight = exp(score - max_score) * inv_sum;

        // Accumulate weighted V.
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

// --------------------------------------------------------------------------
// sdpa_bf16 — Scaled dot-product attention for bfloat16 tensors.
//
// Same algorithm as sdpa, but reads/writes bfloat16. All accumulation
// and intermediate computation stays in float32 for numerical stability.
// --------------------------------------------------------------------------
kernel void sdpa_bf16(
    device const bfloat *Q          [[buffer(0)]],
    device const bfloat *K          [[buffer(1)]],
    device const bfloat *V          [[buffer(2)]],
    device bfloat       *O          [[buffer(3)]],
    device const SdpaParams *params [[buffer(4)]],
    uint3 tgid                     [[threadgroup_position_in_grid]],
    uint  tid                      [[thread_index_in_threadgroup]]
) {
    const uint n_heads     = params->n_heads;
    const uint n_kv_heads  = params->n_kv_heads;
    const uint head_dim    = params->head_dim;
    const uint seq_len     = params->seq_len;
    const uint kv_seq_len  = params->kv_seq_len;

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

    const uint k_head_base = batch_idx * (n_kv_heads * kv_seq_len * head_dim)
                           + kv_head_idx * (kv_seq_len * head_dim);

    const uint v_head_base = k_head_base;
    const uint o_base = q_base;

    const float scale = rsqrt(float(head_dim));

    const uint abs_pos = kv_seq_len - seq_len + q_pos;
    const uint max_k = min(abs_pos + 1, kv_seq_len);

    // Pass 1: find max score.
    float max_score = -INFINITY;
    for (uint k_pos = 0; k_pos < max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        max_score = max(max_score, score);
    }

    // Pass 2a: sum of exp(score - max).
    float sum_exp = 0.0f;
    for (uint k_pos = 0; k_pos < max_k; k_pos++) {
        float dot = 0.0f;
        const uint k_offset = k_head_base + k_pos * head_dim;
        for (uint d = 0; d < head_dim; d++) {
            dot += float(Q[q_base + d]) * float(K[k_offset + d]);
        }
        float score = dot * scale;
        sum_exp += exp(score - max_score);
    }

    // Pass 2b: accumulate weighted V.
    float acc[512];
    for (uint d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }

    const float inv_sum = 1.0f / sum_exp;

    for (uint k_pos = 0; k_pos < max_k; k_pos++) {
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

    // Write output as bfloat16.
    for (uint d = 0; d < head_dim; d++) {
        O[o_base + d] = bfloat(acc[d]);
    }
}
