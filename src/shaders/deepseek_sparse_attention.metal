// DeepSeek-V4 0731 selected sparse attention.
// Public layouts: Q/O [B,Q,64,512], shared KV [B,N,512], indices [B,Q,K].
//
// Pass 1 assigns one SIMD-group to each selected KV slot and writes its
// F32 logit. Pass 2 normalizes those logits once per head, then lets 256
// threads accumulate two output dimensions each. This removes the former
// eight threadgroup barriers for every selected slot.

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
constant uint DS_SIMDGROUPS = 8;
constant uint DS_SLOTS_PER_SIMDGROUP = 4;
constant uint DS_SLOTS_PER_GROUP = DS_SIMDGROUPS * DS_SLOTS_PER_SIMDGROUP;

kernel void deepseek_sparse_attention_validate_q_bf16(
        constant DeepSeekSparseAttentionParams &p [[buffer(0)]],
        device const bfloat *q                    [[buffer(1)]],
        device const float *sinks                 [[buffer(2)]],
        device atomic_uint *invalid_heads         [[buffer(3)]],
        uint head                                 [[threadgroup_position_in_grid]],
        uint tid                                  [[thread_index_in_threadgroup]]) {
    const ulong qbase = ulong(head) * DS_DIM;
    uint invalid = tid == 0 && !isfinite(sinks[head]) ? 1u : 0u;
    for (uint feature = tid; feature < DS_DIM; feature += DS_THREADS) {
        invalid += !isfinite(float(q[qbase + feature])) ? 1u : 0u;
    }
    if (invalid != 0) {
        atomic_store_explicit(&invalid_heads[head], 1u, memory_order_relaxed);
    }
}

kernel void deepseek_sparse_attention_gather_bf16(
        constant DeepSeekSparseAttentionParams &p [[buffer(0)]],
        device const bfloat *kv                   [[buffer(1)]],
        device const int *indices                 [[buffer(2)]],
        device bfloat *gathered                    [[buffer(3)]],
        device bfloat *mask                        [[buffer(4)]],
        device atomic_uint *invalid_global         [[buffer(5)]],
        uint gid                                   [[thread_position_in_grid]]) {
    const uint elements = p.top_k * DS_DIM;
    if (gid >= elements) return;
    const uint slot = gid / DS_DIM;
    const uint feature = gid % DS_DIM;
    const int selected = indices[slot];
    const bool sentinel = selected == -1;
    const bool index_bad = selected < -1 || selected >= int(p.kv_len);
    float value = 0.0f;
    if (!sentinel && !index_bad) {
        value = float(kv[ulong(uint(selected)) * DS_DIM + feature]);
    }
    const bool value_bad = !isfinite(value);
    gathered[gid] = bfloat(!index_bad && !value_bad ? value : 0.0f);
    if (feature == 0) {
        mask[slot] = bfloat(sentinel ? -INFINITY : 0.0f);
    }
    if (index_bad || value_bad) {
        atomic_store_explicit(invalid_global, 1u, memory_order_relaxed);
    }
}

kernel void deepseek_sparse_attention_sanitize_bf16(
        device const atomic_uint *invalid_global [[buffer(0)]],
        device const atomic_uint *invalid_heads  [[buffer(1)]],
        device bfloat *output                     [[buffer(2)]],
        uint gid                                  [[thread_position_in_grid]]) {
    const uint elements = DS_HEADS * DS_DIM;
    if (gid >= elements) return;
    const uint head = gid / DS_DIM;
    const float value = float(output[gid]);
    const bool invalid = atomic_load_explicit(invalid_global, memory_order_relaxed) != 0u
        || atomic_load_explicit(&invalid_heads[head], memory_order_relaxed) != 0u
        || !isfinite(value);
    if (invalid) output[gid] = bfloat(0.0f);
}

kernel void deepseek_sparse_attention_score_bf16(
        constant DeepSeekSparseAttentionParams &p [[buffer(0)]],
        device const bfloat *q                    [[buffer(1)]],
        device const bfloat *kv                   [[buffer(2)]],
        device const int *indices                 [[buffer(3)]],
        device float *scores                      [[buffer(4)]],
        uint group_id                             [[threadgroup_position_in_grid]],
        ushort lane                               [[thread_index_in_simdgroup]],
        ushort simdgroup                          [[simdgroup_index_in_threadgroup]]) {
    const uint blocks_per_head = (p.top_k + DS_SLOTS_PER_GROUP - 1) / DS_SLOTS_PER_GROUP;
    const uint slot_block = group_id % blocks_per_head;
    const uint head_group = group_id / blocks_per_head;
    const uint head = head_group % DS_HEADS;
    const uint query = (head_group / DS_HEADS) % p.query_len;
    const uint batch = head_group / (DS_HEADS * p.query_len);
    const ulong qbase = ((ulong(batch) * p.query_len + query) * DS_HEADS + head) * DS_DIM;
    const ulong ibase = (ulong(batch) * p.query_len + query) * p.top_k;
    const ulong score_base = ulong(head_group) * p.top_k;

    for (uint iteration = 0; iteration < DS_SLOTS_PER_SIMDGROUP; ++iteration) {
        const uint slot = slot_block * DS_SLOTS_PER_GROUP +
            iteration * DS_SIMDGROUPS + simdgroup;
        if (slot >= p.top_k) continue;

        const int selected = indices[ibase + slot];
        const bool sentinel = selected == -1;
        const bool index_bad = selected < -1 || selected >= int(p.kv_len);
        float dot = 0.0f;
        uint bad = index_bad ? 1u : 0u;
        if (!sentinel && !index_bad) {
            const ulong kbase = (ulong(batch) * p.kv_len + uint(selected)) * DS_DIM;
            for (uint feature = lane; feature < DS_DIM; feature += 32) {
                const float qv = float(q[qbase + feature]);
                const float kvv = float(kv[kbase + feature]);
                bad += (!isfinite(qv) || !isfinite(kvv)) ? 1u : 0u;
                dot = fma(isfinite(qv) ? qv : 0.0f,
                          isfinite(kvv) ? kvv : 0.0f,
                          dot);
            }
        }
        dot = simd_sum(dot);
        bad = simd_sum(bad);
        if (lane == 0) {
            const float logit = dot * p.scale;
            scores[score_base + slot] = sentinel
                ? -INFINITY
                : (bad == 0 && isfinite(logit) ? logit : NAN);
        }
    }
}

kernel void deepseek_sparse_attention_reduce_bf16(
        constant DeepSeekSparseAttentionParams &p [[buffer(0)]],
        device const bfloat *kv                   [[buffer(1)]],
        device const float *sinks                 [[buffer(2)]],
        device const int *indices                 [[buffer(3)]],
        device float *scores                      [[buffer(4)]],
        device bfloat *output                     [[buffer(5)]],
        uint head_group                           [[threadgroup_position_in_grid]],
        uint tid                                  [[thread_index_in_threadgroup]]) {
    const uint head = head_group % DS_HEADS;
    const uint query = (head_group / DS_HEADS) % p.query_len;
    const uint batch = head_group / (DS_HEADS * p.query_len);
    const ulong ibase = (ulong(batch) * p.query_len + query) * p.top_k;
    const ulong score_base = ulong(head_group) * p.top_k;
    const ulong output_base = ulong(head_group) * DS_DIM;

    threadgroup float reduction[DS_THREADS];
    threadgroup uint bad[DS_THREADS];
    threadgroup float maximum;
    threadgroup float denominator;
    threadgroup uint invalid;

    const float sink = sinks[head];
    float local_max = tid == 0 && isfinite(sink) ? sink : -INFINITY;
    uint local_bad = isfinite(sink) ? 0u : 1u;
    for (uint slot = tid; slot < p.top_k; slot += DS_THREADS) {
        const float logit = scores[score_base + slot];
        local_bad += isnan(logit) ? 1u : 0u;
        if (isfinite(logit)) local_max = max(local_max, logit);
    }
    reduction[tid] = local_max;
    bad[tid] = local_bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DS_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] = max(reduction[tid], reduction[tid + stride]);
            bad[tid] += bad[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        maximum = reduction[0];
        invalid = bad[0] != 0 || !isfinite(maximum) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = tid == 0 && isfinite(sink) && isfinite(maximum)
        ? exp(sink - maximum)
        : 0.0f;
    for (uint slot = tid; slot < p.top_k; slot += DS_THREADS) {
        const ulong location = score_base + slot;
        const float logit = scores[location];
        const float weight = isfinite(logit) && isfinite(maximum)
            ? exp(logit - maximum)
            : 0.0f;
        scores[location] = weight;
        local_sum += weight;
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DS_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        denominator = reduction[0];
        if (!isfinite(denominator) || denominator <= 0.0f) invalid = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint feature = tid; feature < DS_DIM; feature += DS_THREADS) {
        float numerator = 0.0f;
        for (uint slot = 0; slot < p.top_k; ++slot) {
            const float weight = scores[score_base + slot];
            if (weight > 0.0f) {
                const uint selected = uint(indices[ibase + slot]);
                const float value = float(kv[(ulong(batch) * p.kv_len + selected) * DS_DIM + feature]);
                numerator = fma(weight, value, numerator);
            }
        }
        const float result = numerator / denominator;
        output[output_base + feature] = bfloat(invalid == 0 && isfinite(result) ? result : 0.0f);
    }
}
