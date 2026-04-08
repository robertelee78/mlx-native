#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// moe_gate — Parallel top-K expert routing with softmax weights.
//
// Operates in parallel: one threadgroup per token, 128 threads per group.
//
// Algorithm per token:
//   1. RMS-Norm of hidden state (parallel reduction in threadgroup).
//   2. Router matmul: each thread handles ceil(n_experts/128) experts.
//      Normed hidden is stored in threadgroup shared memory (tg_hidden).
//      Logits are accumulated in threadgroup shared memory (tg_logits).
//   3. Top-K selection (single thread, K=8 from 128 experts via insertion
//      sort — 8 × 128 = 1024 comparisons, trivial).
//   4. Apply per_expert_scale after softmax, then re-normalize.
//
// Buffers:
//   0: hidden_state      — bfloat [seq_len, hidden_dim]
//   1: router_weights    — float  [n_experts, hidden_dim] (row-major)
//   2: norm_weight       — float  [hidden_dim]
//   3: per_expert_scale  — float  [n_experts]
//   4: expert_ids        — uint   [seq_len, top_k]  (output)
//   5: expert_weights    — float  [seq_len, top_k]  (output)
//   6: hidden_dim        — constant uint
//   7: n_experts         — constant uint
//   8: top_k             — constant uint
//   9: rms_eps           — constant float
//
// Threadgroup shared memory layout (index 0):
//   [0 .. hidden_dim)           — float: normed hidden state
//   [hidden_dim .. hidden_dim+n_experts) — float: router logits
//
// Grid:     (seq_len, 1, 1)   — one threadgroup per token
// Threads:  (128, 1, 1)       — 128 threads per threadgroup
// --------------------------------------------------------------------------

kernel void moe_gate(
    device const bfloat* hidden_state       [[buffer(0)]],
    device const float*  router_weights     [[buffer(1)]],
    device const float*  norm_weight        [[buffer(2)]],
    device const float*  per_expert_scale   [[buffer(3)]],
    device uint*         expert_ids         [[buffer(4)]],
    device float*        expert_weights     [[buffer(5)]],
    constant uint&       hidden_dim         [[buffer(6)]],
    constant uint&       n_experts          [[buffer(7)]],
    constant uint&       top_k              [[buffer(8)]],
    constant float&      rms_eps            [[buffer(9)]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  token  [[threadgroup_position_in_grid]],
    uint  tg_size [[threads_per_threadgroup]],
    threadgroup float* shared               [[threadgroup(0)]]
) {
    // shared layout:
    //   shared[0 .. hidden_dim)                — normed hidden (f32)
    //   shared[hidden_dim .. hidden_dim+n_experts) — logits (f32)
    threadgroup float* tg_hidden = shared;
    threadgroup float* tg_logits = shared + hidden_dim;

    const uint token_base = token * hidden_dim;

    // -----------------------------------------------------------------------
    // Phase 1: RMS Norm
    //   Compute rms_inv in parallel, store normed*weight in tg_hidden.
    // -----------------------------------------------------------------------

    // Step 1a: partial sum of squares
    float partial_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        float v = static_cast<float>(hidden_state[token_base + i]);
        partial_sq += v * v;
    }

    // Reuse the logit region as a reduction scratch (n_experts >= tg_size=128).
    tg_logits[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over tg_size threads
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_logits[tid] += tg_logits[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(tg_logits[0] / float(hidden_dim) + rms_eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1b: normalize and multiply by norm_weight -> store in tg_hidden
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        float v = static_cast<float>(hidden_state[token_base + i]);
        tg_hidden[i] = v * rms_inv * norm_weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 2: Router matmul
    //   Each thread computes dot(tg_hidden, router_weights[e]) for its experts.
    // -----------------------------------------------------------------------
    for (uint e = tid; e < n_experts; e += tg_size) {
        float dot = 0.0f;
        device const float* w_row = router_weights + e * hidden_dim;
        for (uint d = 0; d < hidden_dim; d++) {
            dot += tg_hidden[d] * w_row[d];
        }
        tg_logits[e] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Phase 3: Top-K + softmax + per_expert_scale (single thread, tid == 0)
    // -----------------------------------------------------------------------
    if (tid == 0) {
        // MSL does not allow variable-length arrays; n_experts <= 128.
        bool  selected[128];
        float sel_logits[8];   // top_k <= 8

        for (uint e = 0; e < n_experts; e++) {
            selected[e] = false;
        }

        const uint out_base = token * top_k;

        // Insertion sort for top-K
        for (uint k = 0; k < top_k; k++) {
            float best_val = -INFINITY;
            uint  best_idx = 0;
            for (uint e = 0; e < n_experts; e++) {
                if (!selected[e] && tg_logits[e] > best_val) {
                    best_val = tg_logits[e];
                    best_idx = e;
                }
            }
            selected[best_idx] = true;
            expert_ids[out_base + k]  = best_idx;
            sel_logits[k]             = best_val;
        }

        // Standard softmax (no scale)
        float max_logit = sel_logits[0];
        for (uint k = 1; k < top_k; k++) {
            max_logit = max(max_logit, sel_logits[k]);
        }

        float exp_vals[8];
        float sum_exp = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            exp_vals[k] = exp(sel_logits[k] - max_logit);
            sum_exp += exp_vals[k];
        }
        float inv_sum = 1.0f / sum_exp;

        // Apply softmax, then scale, then re-normalize
        float scaled_weights[8];
        float scale_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            float softmax_val = exp_vals[k] * inv_sum;
            float scale = per_expert_scale[expert_ids[out_base + k]];
            scaled_weights[k] = softmax_val * scale;
            scale_sum += scaled_weights[k];
        }
        float inv_scale_sum = 1.0f / scale_sum;
        for (uint k = 0; k < top_k; k++) {
            expert_weights[out_base + k] = scaled_weights[k] * inv_scale_sum;
        }
    }
}
