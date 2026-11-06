// moe_softmax_topk_simd.metal — fused softmax + top-K + renorm for MoE routing,
// one SIMD group per token row.
//
// Same contract as moe_softmax_topk_f32 (moe_softmax_topk.metal): for each row
// of router logits [n_tokens, n_experts], softmax over the experts, pick the
// top_k experts in descending probability (ties go to the lower expert index),
// renormalize their probabilities by their sum, and write
// ids[row * top_k + k] / weights[row * top_k + k] in that descending order.
//
// Technique (the peer's fused top-k MoE routing kernel): each of the 32 lanes
// holds n_experts / 32 logits in registers; max and sum use simd_max/simd_sum;
// each of the top_k rounds finds the best (value, index) pair per lane, then
// reduces it across the SIMD group with simd_shuffle_xor, and the owning lane
// retires that expert. No threadgroup memory, no barriers, no single-thread
// sort. One dispatch covers every row (grid = n_tokens threadgroups of 32).
//
// Numerics: the softmax sum is reduced in SIMD order (Class B against the
// 128-thread tree of moe_softmax_topk_f32). The top-k sum accumulates the
// selected probabilities in descending order on every lane, the same order
// as moe_softmax_topk_f32. Rows are independent, so a row's result does not
// depend on how many rows share the dispatch.
//
// Limits (checked on the host): n_experts <= 32 * MOE_SIMD_MAX_PER_LANE (1024),
// top_k <= 32.

#include <metal_stdlib>
using namespace metal;

#define MOE_SIMD_MAX_PER_LANE 32

struct MoeSoftmaxTopkParams {
    uint n_tokens;
    uint n_experts;
    uint top_k;
    float _pad;
};

kernel void moe_softmax_topk_simd_f32(
        constant MoeSoftmaxTopkParams & p    [[buffer(0)]],
        device const float * logits          [[buffer(1)]],  // [n_tokens, n_experts]
        device       uint  * out_ids         [[buffer(2)]],  // [n_tokens * top_k]
        device       float * out_weights     [[buffer(3)]],  // [n_tokens * top_k]
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]]) {

    const uint row = tgpig.x;
    if (row >= p.n_tokens) return;

    const uint ne        = p.n_experts;
    const uint top_k     = p.top_k;
    const uint per_lane  = (ne + 31) / 32;

    device const float * row_logits = logits + row * ne;

    // Lane `lane` owns experts lane, lane + 32, lane + 64, ...
    float v[MOE_SIMD_MAX_PER_LANE];
    for (uint i = 0; i < MOE_SIMD_MAX_PER_LANE; ++i) {
        v[i] = -INFINITY;
    }
    for (uint i = 0; i < per_lane; ++i) {
        const uint e = lane + i * 32;
        if (e < ne) {
            const float x = row_logits[e];
            v[i] = isnan(x) ? -FLT_MAX : x;
        }
    }

    // Softmax: max, exp, sum.
    float local_max = -INFINITY;
    for (uint i = 0; i < per_lane; ++i) {
        local_max = max(local_max, v[i]);
    }
    const float row_max = simd_max(local_max);

    float local_sum = 0.0f;
    for (uint i = 0; i < per_lane; ++i) {
        const uint e = lane + i * 32;
        const float ev = (e < ne) ? exp(v[i] - row_max) : 0.0f;
        v[i] = (e < ne) ? ev : -INFINITY;
        local_sum += ev;
    }
    const float row_sum = simd_sum(local_sum);
    for (uint i = 0; i < per_lane; ++i) {
        const uint e = lane + i * 32;
        if (e < ne) {
            v[i] = v[i] / row_sum;
        }
    }

    // Top-k: one SIMD-wide argmax per selected expert. Every lane ends each
    // round holding the same (best_val, best_e).
    float sel_val[32];
    uint  sel_id[32];
    float topk_sum = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        float best_val = -INFINITY;
        uint  best_e   = 0xFFFFFFFFu;
        for (uint i = 0; i < per_lane; ++i) {
            const uint e = lane + i * 32;
            if (e < ne && (v[i] > best_val || (v[i] == best_val && e < best_e))) {
                best_val = v[i];
                best_e   = e;
            }
        }
        for (ushort mask = 16; mask > 0; mask >>= 1) {
            const float ov = simd_shuffle_xor(best_val, mask);
            const uint  oe = simd_shuffle_xor(best_e, mask);
            if (ov > best_val || (ov == best_val && oe < best_e)) {
                best_val = ov;
                best_e   = oe;
            }
        }
        if ((best_e & 31u) == lane) {
            v[best_e / 32] = -INFINITY;
        }
        sel_val[k] = best_val;
        sel_id[k]  = best_e;
        topk_sum  += best_val;  // descending order, identical on every lane
    }

    const float inv_topk_sum = (topk_sum > 1e-9f) ? (1.0f / topk_sum) : 1.0f;
    const uint base = row * top_k;
    for (uint k = lane; k < top_k; k += 32) {
        out_ids[base + k]     = sel_id[k];
        out_weights[base + k] = sel_val[k] * inv_topk_sum;
    }
}
