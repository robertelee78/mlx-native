// moe_softmax_topk.metal — Fused softmax + top-K + renorm for MoE routing.
//
// Per-token GPU kernel that replaces the CPU softmax_topk_renorm_cpu() call
// in build_moe_ffn_layer_gpu_q.  Eliminating this CPU round-trip allows the
// router logits matmul and the expert matmuls to live in the same command
// buffer, removing one commit_and_wait() per MoE layer.
//
// Algorithm (per token, one threadgroup):
//   1. Softmax over n_experts logits (numerically stable with max subtraction).
//   2. Top-K selection via insertion sort (K <= 64, works for K=8).
//   3. Renormalize: divide each top-K weight by sum of top-K weights.
//   4. Write flat ids[token*top_k .. (token+1)*top_k] and weights[...].
//
// Grid: (n_tokens, 1, 1).  Threadgroup: (min(n_experts, 128), 1, 1).
// One threadgroup per token; threads collaborate on softmax reduction.

#include <metal_stdlib>
using namespace metal;

struct MoeSoftmaxTopkParams {
    uint n_tokens;
    uint n_experts;
    uint top_k;
    float _pad;  // align to 16 bytes
};

kernel void moe_softmax_topk_f32(
        constant MoeSoftmaxTopkParams & p    [[buffer(0)]],
        device const float * logits          [[buffer(1)]],  // [n_tokens, n_experts]
        device       uint  * out_ids         [[buffer(2)]],  // [n_tokens * top_k]
        device       float * out_weights     [[buffer(3)]],  // [n_tokens * top_k]
        threadgroup  float * shmem           [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint   tiisg [[thread_index_in_simdgroup]],
        uint   tiitg [[thread_index_in_threadgroup]]) {

    const uint token_idx = tgpig.x;
    if (token_idx >= p.n_tokens) return;

    const uint ne = p.n_experts;
    const uint top_k = p.top_k;
    const uint tg_sz = ne < 128 ? ne : 128;  // threadgroup size (capped at 128)

    // Pointer to this token's logits.
    device const float * token_logits = logits + token_idx * ne;

    // ---- Phase 1: Numerically stable softmax ----
    // Each thread computes softmax for its slice of experts.
    // threadgroup memory layout:
    //   shmem[0..tg_sz-1]     = per-thread max reductions
    //   shmem[tg_sz..2*tg_sz-1] = per-thread sum reductions
    //   shmem[2*tg_sz..2*tg_sz+ne-1] = softmax values

    threadgroup float * tg_max  = shmem;
    threadgroup float * tg_sum  = shmem + tg_sz;
    threadgroup float * tg_prob = shmem + 2 * tg_sz;

    // Step 1a: Each thread finds max over its assigned experts.
    float local_max = -INFINITY;
    for (uint e = tiitg; e < ne; e += tg_sz) {
        float v = token_logits[e];
        if (v > local_max) local_max = v;
    }
    tg_max[tiitg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1b: Tree reduction for global max.
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tiitg < stride) {
            if (tg_max[tiitg + stride] > tg_max[tiitg])
                tg_max[tiitg] = tg_max[tiitg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float global_max = tg_max[0];

    // Step 1c: Each thread computes exp(v - max) and writes to tg_prob, accumulates sum.
    float local_sum = 0.f;
    for (uint e = tiitg; e < ne; e += tg_sz) {
        float ev = exp(token_logits[e] - global_max);
        tg_prob[e] = ev;
        local_sum += ev;
    }
    tg_sum[tiitg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1d: Tree reduction for global sum.
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tiitg < stride) {
            tg_sum[tiitg] += tg_sum[tiitg + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float global_sum = tg_sum[0];

    // Step 1e: Normalize probabilities.
    for (uint e = tiitg; e < ne; e += tg_sz) {
        tg_prob[e] /= global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 2: Top-K insertion sort (single thread, thread 0) ----
    // For k <= 64 and n_experts <= 256, single-thread insertion sort is fast.
    if (tiitg == 0) {
        // Use local arrays for top-K (k <= 64 enforced by caller).
        float top_vals[64];
        uint  top_idxs[64];

        // Initialize with -inf.
        for (uint k = 0; k < top_k; k++) {
            top_vals[k] = -INFINITY;
            top_idxs[k] = 0;
        }

        // Insertion sort: maintain sorted top-K (descending order).
        for (uint e = 0; e < ne; e++) {
            float prob = tg_prob[e];
            if (prob <= top_vals[top_k - 1]) continue;  // below threshold

            // Find insertion position (binary search would be overkill for k<=64).
            uint ins = top_k - 1;
            while (ins > 0 && prob > top_vals[ins - 1]) {
                ins--;
            }

            // Shift down to make room.
            for (uint k = top_k - 1; k > ins; k--) {
                top_vals[k] = top_vals[k - 1];
                top_idxs[k] = top_idxs[k - 1];
            }
            top_vals[ins] = prob;
            top_idxs[ins] = e;
        }

        // ---- Phase 3: Renormalize top-K weights ----
        float topk_sum = 0.f;
        for (uint k = 0; k < top_k; k++) {
            topk_sum += top_vals[k];
        }
        const float inv_topk_sum = (topk_sum > 1e-9f) ? (1.f / topk_sum) : 1.f;

        // ---- Phase 4: Write output ----
        const uint base = token_idx * top_k;
        for (uint k = 0; k < top_k; k++) {
            out_ids[base + k]     = top_idxs[k];
            out_weights[base + k] = top_vals[k] * inv_topk_sum;
        }
    }
}
