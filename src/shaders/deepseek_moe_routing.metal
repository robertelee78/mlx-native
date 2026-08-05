// Exact DeepSeek-V4 0731 sqrt-softplus and checkpoint-hash routing.

#include <metal_stdlib>
using namespace metal;

struct DeepSeekMoeRoutingParams {
    uint n_tokens;
    uint vocab_size;
};

constant uint DSV4_EXPERTS = 256;
constant uint DSV4_TOP_K = 6;
constant float DSV4_ROUTE_SCALE = 1.5f;

inline float dsv4_sqrt_softplus(float value) {
    // This is the stable form of log(1 + exp(value)). It has the same F32
    // limit as PyTorch softplus's linear branch for large positive values.
    const float tail = exp(-abs(value));
    const float log_tail = tail < 1.0e-4f
        ? tail * (1.0f + tail * (-0.5f + tail / 3.0f))
        : log(1.0f + tail);
    const float softplus = max(value, 0.0f) + log_tail;
    return sqrt(softplus);
}

inline void dsv4_zero_route(
        device int *indices,
        device float *weights,
        ulong base) {
    for (uint slot = 0; slot < DSV4_TOP_K; ++slot) {
        indices[base + slot] = -1;
        weights[base + slot] = 0.0f;
    }
}

kernel void deepseek_moe_score_route_f32(
        constant DeepSeekMoeRoutingParams &p [[buffer(0)]],
        device const float *logits           [[buffer(1)]],
        device const float *bias             [[buffer(2)]],
        device int *out_indices              [[buffer(3)]],
        device float *out_weights            [[buffer(4)]],
        uint token                           [[threadgroup_position_in_grid]],
        uint tid                             [[thread_index_in_threadgroup]]) {
    if (token >= p.n_tokens) return;
    threadgroup float unbiased[DSV4_EXPERTS];
    threadgroup float selection[DSV4_EXPERTS];
    threadgroup uint bad[DSV4_EXPERTS];

    const float logit = logits[ulong(token) * DSV4_EXPERTS + tid];
    const float learned_bias = bias[tid];
    const float score = dsv4_sqrt_softplus(logit);
    const float selected_score = score + learned_bias;
    unbiased[tid] = score;
    selection[tid] = selected_score;
    bad[tid] = (!isfinite(logit) || !isfinite(learned_bias)
        || !isfinite(score) || !isfinite(selected_score)) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = DSV4_EXPERTS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) bad[tid] += bad[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const ulong out_base = ulong(token) * DSV4_TOP_K;
        if (bad[0] != 0) {
            dsv4_zero_route(out_indices, out_weights, out_base);
            return;
        }
        int chosen[DSV4_TOP_K];
        float gathered[DSV4_TOP_K];
        for (uint slot = 0; slot < DSV4_TOP_K; ++slot) chosen[slot] = -1;
        float sum = 0.0f;
        for (uint slot = 0; slot < DSV4_TOP_K; ++slot) {
            float best = -INFINITY;
            int best_id = -1;
            for (uint expert = 0; expert < DSV4_EXPERTS; ++expert) {
                bool already = false;
                for (uint prior = 0; prior < slot; ++prior) {
                    already |= chosen[prior] == int(expert);
                }
                const float candidate = selection[expert];
                if (!already && (candidate > best
                    || (candidate == best && (best_id < 0 || int(expert) < best_id)))) {
                    best = candidate;
                    best_id = int(expert);
                }
            }
            chosen[slot] = best_id;
            gathered[slot] = unbiased[best_id];
            sum += gathered[slot];
        }
        if (!isfinite(sum) || sum <= 0.0f) {
            dsv4_zero_route(out_indices, out_weights, out_base);
            return;
        }
        const float factor = DSV4_ROUTE_SCALE / sum;
        for (uint slot = 0; slot < DSV4_TOP_K; ++slot) {
            const float weight = gathered[slot] * factor;
            if (!isfinite(weight)) {
                dsv4_zero_route(out_indices, out_weights, out_base);
                return;
            }
            out_indices[out_base + slot] = chosen[slot];
            out_weights[out_base + slot] = weight;
        }
    }
}

kernel void deepseek_moe_hash_route_f32(
        constant DeepSeekMoeRoutingParams &p [[buffer(0)]],
        device const float *logits           [[buffer(1)]],
        device const int *token_ids          [[buffer(2)]],
        device const int *tid2eid            [[buffer(3)]],
        device int *out_indices              [[buffer(4)]],
        device float *out_weights            [[buffer(5)]],
        uint token                           [[thread_position_in_grid]]) {
    if (token >= p.n_tokens) return;
    const ulong out_base = ulong(token) * DSV4_TOP_K;
    const int token_id = token_ids[token];
    if (token_id < 0 || uint(token_id) >= p.vocab_size) {
        dsv4_zero_route(out_indices, out_weights, out_base);
        return;
    }
    float gathered[DSV4_TOP_K];
    int selected[DSV4_TOP_K];
    float sum = 0.0f;
    bool invalid = false;
    const ulong table_base = ulong(token_id) * DSV4_TOP_K;
    for (uint slot = 0; slot < DSV4_TOP_K; ++slot) {
        const int expert = tid2eid[table_base + slot];
        selected[slot] = expert;
        if (expert < 0 || expert >= int(DSV4_EXPERTS)) {
            invalid = true;
            gathered[slot] = 0.0f;
            continue;
        }
        const float logit = logits[ulong(token) * DSV4_EXPERTS + uint(expert)];
        const float score = dsv4_sqrt_softplus(logit);
        gathered[slot] = score;
        sum += score;
        invalid |= !isfinite(logit) || !isfinite(score);
    }
    if (invalid || !isfinite(sum) || sum <= 0.0f) {
        dsv4_zero_route(out_indices, out_weights, out_base);
        return;
    }
    const float factor = DSV4_ROUTE_SCALE / sum;
    for (uint slot = 0; slot < DSV4_TOP_K; ++slot) {
        const float weight = gathered[slot] * factor;
        if (!isfinite(weight)) {
            dsv4_zero_route(out_indices, out_weights, out_base);
            return;
        }
        out_indices[out_base + slot] = selected[slot];
        out_weights[out_base + slot] = weight;
    }
}
