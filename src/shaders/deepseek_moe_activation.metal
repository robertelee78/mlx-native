// Exact DeepSeek-V4 0731 expert activation and routed reduction.

#include <metal_stdlib>
using namespace metal;

struct DeepSeekMoeActivationParams {
    uint count;
    uint use_weights;
};

constant uint DSV4_THREADS = 256;
constant uint DSV4_TOP_K = 6;
constant uint DSV4_INTER_DIM = 2048;
constant uint DSV4_HIDDEN_DIM = 4096;
constant uint DSV4_EXPERTS = 256;
constant float DSV4_SWIGLU_LIMIT = 10.0f;

kernel void deepseek_moe_swiglu_f32(
        constant DeepSeekMoeActivationParams &p [[buffer(0)]],
        device const float *gate                [[buffer(1)]],
        device const float *up                  [[buffer(2)]],
        device const float *selected_weights    [[buffer(3)]],
        device float *output                    [[buffer(4)]],
        uint row                                [[threadgroup_position_in_grid]],
        uint tid                                [[thread_index_in_threadgroup]]) {
    if (row >= p.count) return;
    threadgroup uint bad[DSV4_THREADS];
    float activated[8];
    uint local_bad = 0;
    const float selected_weight = p.use_weights != 0 ? selected_weights[row] : 1.0f;
    local_bad |= !isfinite(selected_weight) ? 1u : 0u;
    const ulong base = ulong(row) * DSV4_INTER_DIM;
    for (uint part = 0; part < 8; ++part) {
        const uint feature = tid + part * DSV4_THREADS;
        const float raw_gate = gate[base + feature];
        const float raw_up = up[base + feature];
        const float clamped_gate = min(raw_gate, DSV4_SWIGLU_LIMIT);
        const float clamped_up = clamp(raw_up, -DSV4_SWIGLU_LIMIT, DSV4_SWIGLU_LIMIT);
        const float silu = clamped_gate / (1.0f + exp(-clamped_gate));
        activated[part] = silu * clamped_up * selected_weight;
        local_bad |= (!isfinite(raw_gate) || !isfinite(raw_up)
            || !isfinite(activated[part])) ? 1u : 0u;
    }
    bad[tid] = local_bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DSV4_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) bad[tid] += bad[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const bool invalid = bad[0] != 0;
    for (uint part = 0; part < 8; ++part) {
        const uint feature = tid + part * DSV4_THREADS;
        output[base + feature] = invalid ? 0.0f : activated[part];
    }
}

kernel void deepseek_moe_weighted_reduce_f32(
        constant DeepSeekMoeActivationParams &p [[buffer(0)]],
        device const int *indices               [[buffer(1)]],
        device const float *weights             [[buffer(2)]],
        device const float *routed              [[buffer(3)]],
        device const float *shared              [[buffer(4)]],
        device float *output                    [[buffer(5)]],
        uint token                              [[threadgroup_position_in_grid]],
        uint tid                                [[thread_index_in_threadgroup]]) {
    if (token >= p.count) return;
    threadgroup uint slot_order[DSV4_TOP_K];
    threadgroup uint route_bad;
    threadgroup uint bad[DSV4_THREADS];
    const ulong route_base = ulong(token) * DSV4_TOP_K;
    if (tid == 0) {
        route_bad = 0;
        for (uint slot = 0; slot < DSV4_TOP_K; ++slot) {
            slot_order[slot] = slot;
            route_bad |= (indices[route_base + slot] < 0
                || indices[route_base + slot] >= int(DSV4_EXPERTS)
                || !isfinite(weights[route_base + slot])) ? 1u : 0u;
        }
        // Stable insertion sort reproduces the official ascending expert loop;
        // equal expert IDs retain checkpoint/top-k slot order.
        for (uint slot = 1; slot < DSV4_TOP_K; ++slot) {
            const uint current = slot_order[slot];
            const int current_id = indices[route_base + current];
            uint position = slot;
            while (position > 0
                && indices[route_base + slot_order[position - 1]] > current_id) {
                slot_order[position] = slot_order[position - 1];
                --position;
            }
            slot_order[position] = current;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float reduced[16];
    uint local_bad = route_bad;
    const ulong shared_base = ulong(token) * DSV4_HIDDEN_DIM;
    for (uint part = 0; part < 16; ++part) {
        const uint feature = tid + part * DSV4_THREADS;
        float acc = 0.0f;
        for (uint order = 0; order < DSV4_TOP_K; ++order) {
            const uint slot = slot_order[order];
            const float weight = weights[route_base + slot];
            const float value = routed[(route_base + slot) * DSV4_HIDDEN_DIM + feature];
            local_bad |= !isfinite(value) ? 1u : 0u;
            acc = fma(weight, value, acc);
        }
        const float shared_value = shared[shared_base + feature];
        local_bad |= !isfinite(shared_value) ? 1u : 0u;
        reduced[part] = acc + shared_value;
        local_bad |= !isfinite(reduced[part]) ? 1u : 0u;
    }
    bad[tid] = local_bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DSV4_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) bad[tid] += bad[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const bool invalid = bad[0] != 0;
    for (uint part = 0; part < 16; ++part) {
        const uint feature = tid + part * DSV4_THREADS;
        output[shared_base + feature] = invalid ? 0.0f : reduced[part];
    }
}
