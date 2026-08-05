// DeepSeek-V4 0731 index score and deterministic causal top-512.

#include <metal_stdlib>
using namespace metal;

struct DeepSeekIndexerParams {
    uint batch;
    uint query_len;
    uint kv_len;
    uint start_pos;
    uint ratio;
    uint heads;
    uint head_dim;
    uint top_k;
    int offset;
};

struct DeepSeekIndexerOutputLayout {
    uint row_stride;
    uint column_offset;
};

constant uint IDX_HEADS = 64;
constant uint IDX_DIM = 128;
constant uint IDX_TOPK = 512;
constant uint IDX_THREADS = 256;
constant uint IDX_SIMDGROUPS = 8;

kernel void deepseek_indexer_score_bf16(
        constant DeepSeekIndexerParams &p [[buffer(0)]],
        device const bfloat *q             [[buffer(1)]],
        device const bfloat *kv            [[buffer(2)]],
        device const float *weights        [[buffer(3)]],
        device float *scores               [[buffer(4)]],
        uint3 group                        [[threadgroup_position_in_grid]],
        ushort lane                        [[thread_index_in_simdgroup]],
        ushort simdgroup                   [[simdgroup_index_in_threadgroup]]) {
    const uint candidate = group.x % p.kv_len;
    const uint query = (group.x / p.kv_len) % p.query_len;
    const uint batch = group.x / (p.kv_len * p.query_len);
    const ulong out_index = (ulong(batch) * p.query_len + query) * p.kv_len + candidate;
    const uint valid_count = (p.start_pos + query + 1) / p.ratio;
    if (candidate >= valid_count) {
        if (lane == 0 && simdgroup == 0) scores[out_index] = -INFINITY;
        return;
    }

    threadgroup float partial[IDX_SIMDGROUPS];
    threadgroup uint invalid[IDX_SIMDGROUPS];
    float head_sum = 0.0f;
    uint local_bad = 0;
    const ulong kv_base = (ulong(batch) * p.kv_len + candidate) * IDX_DIM;
    for (uint head = simdgroup; head < IDX_HEADS; head += IDX_SIMDGROUPS) {
        const ulong q_base = ((ulong(batch) * p.query_len + query) * IDX_HEADS + head) * IDX_DIM;
        float dot = 0.0f;
        uint bad = 0;
        for (uint feature = lane; feature < IDX_DIM; feature += 32) {
            const float qv = float(q[q_base + feature]);
            const float kvv = float(kv[kv_base + feature]);
            bad += (!isfinite(qv) || !isfinite(kvv)) ? 1u : 0u;
            dot = fma(isfinite(qv) ? qv : 0.0f, isfinite(kvv) ? kvv : 0.0f, dot);
        }
        dot = simd_sum(dot);
        bad = simd_sum(bad);
        if (lane == 0) {
            const float weight = weights[(ulong(batch) * p.query_len + query) * IDX_HEADS + head];
            const float contribution = max(dot, 0.0f) * weight;
            local_bad += bad + ((!isfinite(weight) || !isfinite(contribution)) ? 1u : 0u);
            head_sum += isfinite(contribution) ? contribution : 0.0f;
        }
    }
    if (lane == 0) {
        partial[simdgroup] = head_sum;
        invalid[simdgroup] = local_bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && simdgroup == 0) {
        float total = 0.0f;
        uint bad = 0;
        for (uint i = 0; i < IDX_SIMDGROUPS; ++i) {
            total += partial[i];
            bad += invalid[i];
        }
        scores[out_index] = bad == 0 && isfinite(total) ? total : -INFINITY;
    }
}

inline bool better_index(float score, int index, float other_score, int other_index) {
    return isfinite(score) && (score > other_score || (score == other_score && index < other_index));
}

kernel void deepseek_indexer_topk_i32(
        constant DeepSeekIndexerParams &p [[buffer(0)]],
        device float *scores               [[buffer(1)]],
        device int *output                 [[buffer(2)]],
        constant DeepSeekIndexerOutputLayout &layout [[buffer(3)]],
        uint3 group                        [[threadgroup_position_in_grid]],
        uint tid                           [[thread_index_in_threadgroup]]) {
    const uint query = group.x % p.query_len;
    const uint batch = group.x / p.query_len;
    const ulong score_base = (ulong(batch) * p.query_len + query) * p.kv_len;
    const ulong output_base =
        (ulong(batch) * p.query_len + query) * layout.row_stride + layout.column_offset;
    const uint valid_count = min(p.kv_len, (p.start_pos + query + 1) / p.ratio);

    threadgroup float best_scores[IDX_THREADS];
    threadgroup int best_indices[IDX_THREADS];
    threadgroup uint selected;
    for (uint slot = tid; slot < IDX_TOPK; slot += IDX_THREADS) output[output_base + slot] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint selections = min(valid_count, IDX_TOPK);
    for (uint slot = 0; slot < selections; ++slot) {
        float local_score = -INFINITY;
        int local_index = INT_MAX;
        for (uint candidate = tid; candidate < valid_count; candidate += IDX_THREADS) {
            const float score = scores[score_base + candidate];
            if (better_index(score, int(candidate), local_score, local_index)) {
                local_score = score;
                local_index = int(candidate);
            }
        }
        best_scores[tid] = local_score;
        best_indices[tid] = local_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = IDX_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride && better_index(
                    best_scores[tid + stride], best_indices[tid + stride],
                    best_scores[tid], best_indices[tid])) {
                best_scores[tid] = best_scores[tid + stride];
                best_indices[tid] = best_indices[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            selected = best_indices[0] != INT_MAX;
            if (selected != 0) {
                output[output_base + slot] = best_indices[0] + p.offset;
                scores[score_base + best_indices[0]] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (selected == 0) break;
    }
}
