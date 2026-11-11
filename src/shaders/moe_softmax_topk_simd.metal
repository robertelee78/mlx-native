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
// The per-lane count and top_k are function constants, so every register
// index is static after unrolling.
//
// Numerics: the softmax sum is reduced in SIMD order (Class B against the
// 128-thread tree of moe_softmax_topk_f32). The top-k sum accumulates the
// selected probabilities in descending order on every lane, the same order
// as moe_softmax_topk_f32. Rows are independent, so a row's result does not
// depend on how many rows share the dispatch.
//
// Limits (checked on the host): n_experts <= 32 * 32 (1024), top_k <= 32.

#include <metal_stdlib>
using namespace metal;

constant int FC_PER_LANE [[function_constant(0)]];  // ceil(n_experts / 32), 1..32
constant int FC_TOP_K    [[function_constant(1)]];  // 1..32

struct MoeSoftmaxTopkParams {
    uint n_tokens;
    uint n_experts;
    uint top_k;
    uint output_stride;
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

    const uint ne = p.n_experts;
    device const float * row_logits = logits + row * ne;

    // Lane `lane` owns experts lane, lane + 32, lane + 64, ...
    float v[32];  // only the first FC_PER_LANE entries are used
#pragma unroll
    for (int i = 0; i < FC_PER_LANE; ++i) {
        const uint e = lane + uint(i) * 32;
        const float x = (e < ne) ? row_logits[e] : -INFINITY;
        // NaN never wins; +/-inf are clamped to the finite range so every row keeps a
        // finite max and every selected id is a real expert (< n_experts).
        v[i] = isnan(x) ? -FLT_MAX : clamp(x, -FLT_MAX, FLT_MAX);
    }

    // Softmax: max, exp, sum.
    float local_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < FC_PER_LANE; ++i) {
        local_max = max(local_max, v[i]);
    }
    const float row_max = simd_max(local_max);

    float local_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < FC_PER_LANE; ++i) {
        const uint e = lane + uint(i) * 32;
        const float ev = (e < ne) ? exp(v[i] - row_max) : 0.0f;
        v[i] = (e < ne) ? ev : -INFINITY;
        local_sum += ev;
    }
    const float row_sum = simd_sum(local_sum);
#pragma unroll
    for (int i = 0; i < FC_PER_LANE; ++i) {
        const uint e = lane + uint(i) * 32;
        if (e < ne) {
            v[i] = v[i] / row_sum;
        }
    }

    // Top-k: one SIMD-wide argmax per selected expert. Every lane ends each
    // round holding the same (best_val, best_e); lane k keeps round k.
    float my_val = 0.0f;
    uint  my_id  = 0;
    float topk_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < FC_TOP_K; ++k) {
        float best_val = -INFINITY;
        uint  best_e   = 0xFFFFFFFFu;
#pragma unroll
        for (int i = 0; i < FC_PER_LANE; ++i) {
            const uint e = lane + uint(i) * 32;
            if (e < ne && (v[i] > best_val || (v[i] == best_val && e < best_e))) {
                best_val = v[i];
                best_e   = e;
            }
        }
#pragma unroll
        for (ushort mask = 16; mask > 0; mask >>= 1) {
            const float ov = simd_shuffle_xor(best_val, mask);
            const uint  oe = simd_shuffle_xor(best_e, mask);
            if (ov > best_val || (ov == best_val && oe < best_e)) {
                best_val = ov;
                best_e   = oe;
            }
        }
#pragma unroll
        for (int i = 0; i < FC_PER_LANE; ++i) {
            if (lane + uint(i) * 32 == best_e) {
                v[i] = -INFINITY;
            }
        }
        if (lane == ushort(k)) {
            my_val = best_val;
            my_id  = best_e;
        }
        topk_sum += best_val;  // descending order, identical on every lane
    }

    const float inv_topk_sum = (topk_sum > 1e-9f) ? (1.0f / topk_sum) : 1.0f;
    const uint base = row * p.output_stride;
    if (lane < ushort(FC_TOP_K)) {
        out_ids[base + lane]     = my_id;
        out_weights[base + lane] = my_val * inv_topk_sum;
    } else if (lane == ushort(FC_TOP_K) && p.output_stride > p.top_k) {
        out_ids[base + p.top_k] = p.n_experts;
        out_weights[base + p.top_k] = 0.0f;
    }
}
