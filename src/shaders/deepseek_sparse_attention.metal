// DeepSeek-V4 0731 sparse attention.
// Public layouts: Q/O [B,Q,64,512], shared KV [B,N,512], indices [B,Q,K].

#include <metal_stdlib>
using namespace metal;

struct DeepSeekSparseAttentionParams {
    uint batch;
    uint query_len;
    uint kv_len;
    uint top_k;
    uint heads;
    uint head_dim;
    float scale;
};

constant uint DS_HEADS = 64;
constant uint DS_DIM = 512;
constant uint DS_THREADS = 256;

kernel void deepseek_sparse_attention_bf16(
        constant DeepSeekSparseAttentionParams &p [[buffer(0)]],
        device const bfloat *q                    [[buffer(1)]],
        device const bfloat *kv                   [[buffer(2)]],
        device const float *sinks                 [[buffer(3)]],
        device const int *indices                 [[buffer(4)]],
        device bfloat *output                     [[buffer(5)]],
        uint3 group                               [[threadgroup_position_in_grid]],
        uint tid                                  [[thread_index_in_threadgroup]]) {
    const uint flat = group.x;
    const uint head = flat % DS_HEADS;
    const uint query = (flat / DS_HEADS) % p.query_len;
    const uint batch = flat / (DS_HEADS * p.query_len);
    const ulong qbase = ((ulong(batch) * p.query_len + query) * DS_HEADS + head) * DS_DIM;
    const ulong ibase = (ulong(batch) * p.query_len + query) * p.top_k;

    threadgroup float reduce[DS_THREADS];
    threadgroup uint bad[DS_THREADS];
    threadgroup int selected_index;
    threadgroup uint selected;
    threadgroup uint invalid;
    threadgroup uint has_value;
    threadgroup float running_max;
    threadgroup float running_sum;
    threadgroup float old_scale;
    threadgroup float new_weight;
    threadgroup float final_scale;

    const float raw_q0 = float(q[qbase + tid]);
    const float raw_q1 = float(q[qbase + tid + DS_THREADS]);
    const bool q_bad = !isfinite(raw_q0) || !isfinite(raw_q1);
    const float q0 = isfinite(raw_q0) ? raw_q0 : 0.0f;
    const float q1 = isfinite(raw_q1) ? raw_q1 : 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    bad[tid] = q_bad ? 1u : 0u;
    if (tid == 0) {
        invalid = (!isfinite(p.scale) || p.scale <= 0.0f ||
                   !isfinite(sinks[head])) ? 1u : 0u;
        has_value = 0;
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DS_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) bad[tid] += bad[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && bad[0] != 0) invalid = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < p.top_k; ++slot) {
        if (tid == 0) {
            selected_index = indices[ibase + slot];
            selected = selected_index >= 0 && uint(selected_index) < p.kv_len;
            if (selected_index < -1 || selected_index >= int(p.kv_len)) invalid = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float value0 = 0.0f;
        float value1 = 0.0f;
        bool value_bad = false;
        if (selected != 0) {
            const ulong kbase = (ulong(batch) * p.kv_len + uint(selected_index)) * DS_DIM;
            value0 = float(kv[kbase + tid]);
            value1 = float(kv[kbase + tid + DS_THREADS]);
            value_bad = !isfinite(value0) || !isfinite(value1);
            value0 = isfinite(value0) ? value0 : 0.0f;
            value1 = isfinite(value1) ? value1 : 0.0f;
        }
        reduce[tid] = fma(q0, value0, q1 * value1);
        bad[tid] = value_bad ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = DS_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
                bad[tid] += bad[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            if (selected != 0 && bad[0] != 0) invalid = 1;
            const float logit = reduce[0] * p.scale;
            if (selected != 0 && isfinite(logit)) {
                const float next_max = has_value != 0 ? max(running_max, logit) : logit;
                old_scale = has_value != 0 ? exp(running_max - next_max) : 0.0f;
                new_weight = exp(logit - next_max);
                running_sum = running_sum * old_scale + new_weight;
                running_max = next_max;
                has_value = 1;
            } else {
                if (selected != 0) invalid = 1;
                old_scale = 1.0f;
                new_weight = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc0 = fma(new_weight, value0, acc0 * old_scale);
        acc1 = fma(new_weight, value1, acc1 * old_scale);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        if (has_value != 0) {
            const float sink = sinks[head];
            const float next_max = max(running_max, sink);
            const float numerator_scale = exp(running_max - next_max);
            const float denominator = running_sum * numerator_scale + exp(sink - next_max);
            final_scale = numerator_scale / denominator;
            if (!isfinite(final_scale)) invalid = 1;
        } else {
            final_scale = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float result0 = acc0 * final_scale;
    const float result1 = acc1 * final_scale;
    bad[tid] = (!isfinite(result0) || !isfinite(result1)) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DS_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) bad[tid] += bad[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && bad[0] != 0) invalid = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    output[qbase + tid] = bfloat(invalid == 0 ? result0 : 0.0f);
    output[qbase + tid + DS_THREADS] = bfloat(invalid == 0 ? result1 : 0.0f);
}
