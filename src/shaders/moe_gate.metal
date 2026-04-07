#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// moe_gate — Top-K expert routing with softmax weights
//
// Given a hidden state vector [hidden_dim] and a router weight matrix
// [hidden_dim x n_experts], compute router logits, select top-K experts,
// and return softmax-normalized weights over the selected experts.
//
// This kernel operates on a single token at a time. For batch processing,
// the Rust host dispatches one invocation per token.
//
// Stage 1: Compute router logits via dot products (one per expert)
// Stage 2: Find top-K experts (simple selection sort for K=8 out of 128)
// Stage 3: Softmax over the K selected logits
//
// Buffers:
//   0: hidden_state    — float [hidden_dim]
//   1: router_weight   — float [n_experts, hidden_dim] (row-major: each expert is one row)
//   2: out_expert_ids  — uint32 [top_k]  (output: selected expert indices)
//   3: out_weights     — float  [top_k]  (output: softmax routing weights)
//   4: params          — { hidden_dim, n_experts, top_k }
//
// Grid: (1, 1, 1)  — single threadgroup, single thread for correctness
//        (For large n_experts, a parallel reduction would be faster,
//         but 128 experts with dot products of size 2816 is manageable
//         for Stage 1. Epic 6 can optimize.)
// --------------------------------------------------------------------------

struct MoeGateParams {
    uint hidden_dim;
    uint n_experts;
    uint top_k;
};

kernel void moe_gate(
    device const float*    hidden_state    [[buffer(0)]],
    device const float*    router_weight   [[buffer(1)]],
    device uint32_t*       out_expert_ids  [[buffer(2)]],
    device float*          out_weights     [[buffer(3)]],
    constant MoeGateParams& params         [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // This kernel runs as a single thread for correctness.
    // It computes: logits[e] = dot(hidden_state, router_weight[e]) for each expert e
    // Then selects top-K and applies softmax over them.

    uint hidden_dim = params.hidden_dim;
    uint n_experts  = params.n_experts;
    uint top_k      = params.top_k;

    // --- Stage 1: Compute router logits ---
    // We use threadgroup memory to store all logits.
    // n_experts is at most 128, so this is fine.
    // NOTE: threadgroup memory is not available for a single-thread kernel
    // dispatched with threads=(1,1,1). Use device memory scratch or
    // local array.  128 floats = 512 bytes, fits in registers/stack.

    // We cannot use variable-length arrays in MSL, but n_experts <= 128.
    // Use a fixed-size array.
    float logits[128];

    for (uint e = 0; e < n_experts; e++) {
        float dot = 0.0f;
        device const float* w_row = router_weight + e * hidden_dim;
        for (uint d = 0; d < hidden_dim; d++) {
            dot += hidden_state[d] * w_row[d];
        }
        logits[e] = dot;
    }

    // --- Stage 2: Top-K selection (selection sort, K iterations) ---
    // For K=8 out of 128, this is 8*128 = 1024 comparisons — trivial.
    bool selected[128];
    for (uint e = 0; e < n_experts; e++) {
        selected[e] = false;
    }

    float selected_logits[8];  // top_k <= 8

    for (uint k = 0; k < top_k; k++) {
        float best_val = -INFINITY;
        uint best_idx = 0;
        for (uint e = 0; e < n_experts; e++) {
            if (!selected[e] && logits[e] > best_val) {
                best_val = logits[e];
                best_idx = e;
            }
        }
        selected[best_idx] = true;
        out_expert_ids[k] = best_idx;
        selected_logits[k] = best_val;
    }

    // --- Stage 3: Softmax over the K selected logits ---
    // Find max for numerical stability
    float max_logit = selected_logits[0];
    for (uint k = 1; k < top_k; k++) {
        max_logit = max(max_logit, selected_logits[k]);
    }

    float sum_exp = 0.0f;
    float exp_vals[8];
    for (uint k = 0; k < top_k; k++) {
        exp_vals[k] = exp(selected_logits[k] - max_logit);
        sum_exp += exp_vals[k];
    }

    for (uint k = 0; k < top_k; k++) {
        out_weights[k] = exp_vals[k] / sum_exp;
    }
}
