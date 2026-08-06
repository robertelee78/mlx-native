// DeepSeek-V4 0731 tiled lightning indexer and deterministic causal top-512.
//
// The score kernel tiles 64 candidates while preserving the prior BF16->F32
// reduction order and fail-closed semantics.  The top-k path follows the
// block-sort/hierarchical-merge structure of llama.cpp's MIT-licensed Metal
// implementation (commit f9e832c1), adapted to mlx-native's strided output.

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

struct DeepSeekIndexerTopKPlan {
    uint block_threads;
    uint block_count;
    uint list_count;
    uint scratch_row_stride;
};

constant uint IDX_HEADS = 64;
constant uint IDX_DIM = 128;
constant uint IDX_TOPK = 512;
constant uint IDX_SCORE_SIMDGROUPS = 8;
constant uint IDX_KEYS_PER_SIMDGROUP = 8;
constant uint IDX_KEYS_PER_GROUP = IDX_SCORE_SIMDGROUPS * IDX_KEYS_PER_SIMDGROUP;
constant uint IDX_HEADS_PER_TILE = 8;
constant uint IDX_QUERIES_PER_GROUP = 8;

inline bool index_better(float score, int index, float other_score, int other_index) {
    if (index < 0) return false;
    if (other_index < 0) return true;
    return isfinite(score) &&
        (!isfinite(other_score) || score > other_score ||
         (score == other_score && index < other_index));
}

inline float index_score(device const float *scores, ulong row_base, int index) {
    return index < 0 ? -INFINITY : scores[row_base + uint(index)];
}

kernel void deepseek_indexer_score_bf16(
        constant DeepSeekIndexerParams &p [[buffer(0)]],
        device const bfloat *q             [[buffer(1)]],
        device const bfloat *kv            [[buffer(2)]],
        device const float *weights        [[buffer(3)]],
        device float *scores               [[buffer(4)]],
        uint3 group                        [[threadgroup_position_in_grid]],
        ushort tid                         [[thread_index_in_threadgroup]],
        ushort lane                        [[thread_index_in_simdgroup]],
        ushort simdgroup                   [[simdgroup_index_in_threadgroup]]) {
    const uint key_group_start = group.x * IDX_KEYS_PER_GROUP;
    const uint row_start = group.y * IDX_QUERIES_PER_GROUP;
    const uint total_rows = p.query_len;

    // Cache the complete query once per 64-candidate tile.  A bfloat cache is
    // exact: conversion to F32 still occurs at the same point as the original
    // one-threadgroup-per-candidate kernel.
    threadgroup bfloat staged_q[IDX_HEADS * IDX_DIM];
    for (uint i = tid; i < IDX_HEADS * IDX_DIM; i += 256) {
        staged_q[i] = q[ulong(row_start) * IDX_HEADS * IDX_DIM + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row_delta = 0; row_delta < IDX_QUERIES_PER_GROUP; ++row_delta) {
        const uint row = row_start + row_delta;
        if (row >= total_rows) break;
        const uint query = row % p.query_len;
        const uint valid_count = min(p.kv_len, (p.start_pos + query + 1) / p.ratio);
        if (row_delta != 0) {
            for (uint i = tid; i < IDX_HEADS * IDX_DIM; i += 256) {
                staged_q[i] = q[ulong(row) * IDX_HEADS * IDX_DIM + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint local_key = simdgroup; local_key < IDX_KEYS_PER_GROUP;
             local_key += IDX_SCORE_SIMDGROUPS) {
            const uint key = key_group_start + local_key;
            if (key >= p.kv_len) continue;
            float total = 0.0f;
            uint total_bad = 0;
            // Preserve the old eight-simdgroup head partition and its final
            // partial[0..8] accumulation order exactly.
            for (uint head_bucket = 0; head_bucket < IDX_SCORE_SIMDGROUPS; ++head_bucket) {
                float head_sum = 0.0f;
                uint local_bad = 0;
                for (uint head = head_bucket; head < IDX_HEADS; head += IDX_SCORE_SIMDGROUPS) {
                    float dot = 0.0f;
                    uint bad = 0;
                    for (uint feature = lane; feature < IDX_DIM; feature += 32) {
                        const float qv = float(staged_q[head * IDX_DIM + feature]);
                        const float kvv = float(kv[ulong(key) * IDX_DIM + feature]);
                        bad += (!isfinite(qv) || !isfinite(kvv)) ? 1u : 0u;
                        dot = fma(isfinite(qv) ? qv : 0.0f, isfinite(kvv) ? kvv : 0.0f, dot);
                    }
                    dot = simd_sum(dot);
                    bad = simd_sum(bad);
                    if (lane == 0) {
                        const float weight = weights[ulong(row) * IDX_HEADS + head];
                        const float contribution = max(dot, 0.0f) * weight;
                        local_bad += bad + ((!isfinite(weight) || !isfinite(contribution)) ? 1u : 0u);
                        head_sum += isfinite(contribution) ? contribution : 0.0f;
                    }
                }
                if (lane == 0) {
                    total += head_sum;
                    total_bad += local_bad;
                }
            }
            if (lane == 0) {
                scores[ulong(row) * p.kv_len + key] =
                    key < valid_count && total_bad == 0 && isfinite(total) ? total : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Fast finite-input path matching llama.cpp's simdgroup-MMA score geometry.
// It is opt-in until model-level coherence accepts the BF16->F16 tile staging.
kernel void deepseek_indexer_score_mma_bf16(
        constant DeepSeekIndexerParams &p [[buffer(0)]],
        device const bfloat *q             [[buffer(1)]],
        device const bfloat *kv            [[buffer(2)]],
        device const float *weights        [[buffer(3)]],
        device float *scores               [[buffer(4)]],
        uint3 group                        [[threadgroup_position_in_grid]],
        ushort tid                         [[thread_index_in_threadgroup]],
        ushort lane                        [[thread_index_in_simdgroup]],
        ushort simdgroup                   [[simdgroup_index_in_threadgroup]]) {
    const uint key_group_start = group.x * IDX_KEYS_PER_GROUP;
    const uint row_start = group.y * IDX_QUERIES_PER_GROUP;

    threadgroup half staged_k[IDX_KEYS_PER_GROUP * IDX_DIM];
    threadgroup half staged_q[IDX_HEADS_PER_TILE * IDX_DIM];
    threadgroup float staged_weights[IDX_HEADS_PER_TILE];
    threadgroup float qk_tiles[
        IDX_SCORE_SIMDGROUPS * IDX_HEADS_PER_TILE * IDX_KEYS_PER_SIMDGROUP];

    for (uint i = tid; i < IDX_KEYS_PER_GROUP * IDX_DIM; i += 256) {
        const uint local_key = i / IDX_DIM;
        const uint feature = i % IDX_DIM;
        const uint key = key_group_start + local_key;
        staged_k[i] = key < p.kv_len
            ? half(kv[ulong(key) * IDX_DIM + feature])
            : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 k_tiles[IDX_DIM / 8];
    for (uint tile = 0; tile < IDX_DIM / 8; ++tile) {
        simdgroup_load(
            k_tiles[tile],
            staged_k + simdgroup * IDX_KEYS_PER_SIMDGROUP * IDX_DIM + 8 * tile,
            IDX_DIM,
            0,
            true);
    }

    for (uint row = row_start; row < min(p.query_len, row_start + IDX_QUERIES_PER_GROUP); ++row) {
        const uint valid_count = min(p.kv_len, (p.start_pos + row + 1) / p.ratio);
        float score = 0.0f;
        for (uint head_start = 0; head_start < IDX_HEADS; head_start += IDX_HEADS_PER_TILE) {
            for (uint i = tid; i < IDX_HEADS_PER_TILE * IDX_DIM; i += 256) {
                const uint local_head = i / IDX_DIM;
                const uint feature = i % IDX_DIM;
                staged_q[i] = half(q[
                    (ulong(row) * IDX_HEADS + head_start + local_head) * IDX_DIM + feature]);
            }
            if (tid < IDX_HEADS_PER_TILE) {
                staged_weights[tid] = weights[ulong(row) * IDX_HEADS + head_start + tid];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 product = make_filled_simdgroup_matrix<float, 8>(0.0f);
            for (uint tile = 0; tile < IDX_DIM / 8; ++tile) {
                simdgroup_half8x8 q_tile;
                simdgroup_load(q_tile, staged_q + 8 * tile, IDX_DIM, 0, false);
                simdgroup_multiply_accumulate(product, q_tile, k_tiles[tile], product);
            }
            threadgroup float *tile_out =
                qk_tiles + simdgroup * IDX_HEADS_PER_TILE * IDX_KEYS_PER_SIMDGROUP;
            simdgroup_store(product, tile_out, IDX_KEYS_PER_SIMDGROUP, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < IDX_KEYS_PER_SIMDGROUP) {
                for (uint head = 0; head < IDX_HEADS_PER_TILE; ++head) {
                    score += max(tile_out[head * IDX_KEYS_PER_SIMDGROUP + lane], 0.0f) *
                        staged_weights[head];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane < IDX_KEYS_PER_SIMDGROUP) {
            const uint key = key_group_start + simdgroup * IDX_KEYS_PER_SIMDGROUP + lane;
            if (key < p.kv_len) {
                scores[ulong(row) * p.kv_len + key] =
                    key < valid_count && isfinite(score) ? score : -INFINITY;
            }
        }
    }
}

kernel void deepseek_indexer_topk_block_i32(
        constant DeepSeekIndexerParams &p [[buffer(0)]],
        constant DeepSeekIndexerTopKPlan &plan [[buffer(1)]],
        device const float *scores [[buffer(2)]],
        device int *indices [[buffer(3)]],
        threadgroup int *local_indices [[threadgroup(0)]],
        uint group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint row = group_id / plan.block_count;
    const uint block = group_id % plan.block_count;
    const uint block_start = block * plan.block_threads;
    const uint candidate = block_start + tid;
    const uint query = row % p.query_len;
    const uint valid_count = min(p.kv_len, (p.start_pos + query + 1) / p.ratio);
    const ulong score_base = ulong(row) * p.kv_len;

    local_indices[tid] = candidate < valid_count &&
        isfinite(scores[score_base + candidate]) ? int(candidate) : -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Descending bitonic sort, with ascending source index as the exact tie
    // breaker used by the prior repeated-argmax implementation.
    for (uint width = 2; width <= plan.block_threads; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            const uint peer = tid ^ stride;
            if (peer > tid) {
                const int lhs = local_indices[tid];
                const int rhs = local_indices[peer];
                const float lhs_score = index_score(scores, score_base, lhs);
                const float rhs_score = index_score(scores, score_base, rhs);
                const bool ascending_half = (tid & width) != 0;
                const bool rhs_better = index_better(rhs_score, rhs, lhs_score, lhs);
                const bool lhs_better = index_better(lhs_score, lhs, rhs_score, rhs);
                if ((!ascending_half && rhs_better) || (ascending_half && lhs_better)) {
                    local_indices[tid] = rhs;
                    local_indices[peer] = lhs;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < IDX_TOPK) {
        const ulong out = ulong(row) * plan.scratch_row_stride + ulong(block) * IDX_TOPK + tid;
        indices[out] = local_indices[tid];
    }
}

kernel void deepseek_indexer_topk_merge_i32(
        constant DeepSeekIndexerParams &p [[buffer(0)]],
        constant DeepSeekIndexerTopKPlan &plan [[buffer(1)]],
        device const float *scores [[buffer(2)]],
        device const int *input [[buffer(3)]],
        device int *output [[buffer(4)]],
        uint group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]]) {
    const uint merged_lists = (plan.list_count + 1) / 2;
    const uint row = group_id / merged_lists;
    const uint pair = group_id % merged_lists;
    const uint left_list = pair * 2;
    const uint right_list = left_list + 1;
    const ulong row_base = ulong(row) * plan.scratch_row_stride;
    const ulong score_base = ulong(row) * p.kv_len;
    device const int *left = input + row_base + ulong(left_list) * IDX_TOPK;
    device const int *right = right_list < plan.list_count
        ? input + row_base + ulong(right_list) * IDX_TOPK
        : nullptr;
    device int *dst = output + row_base + ulong(pair) * IDX_TOPK;

    const uint chunk = (IDX_TOPK + threads - 1) / threads;
    const uint out_begin = tid * chunk;
    const uint out_end = min(IDX_TOPK, out_begin + chunk);
    if (out_begin >= IDX_TOPK) return;

    if (right == nullptr) {
        for (uint out = out_begin; out < out_end; ++out) dst[out] = left[out];
        return;
    }

    // Merge-path partition for the first `out_begin` elements of two
    // descending 512-entry lists.
    int low = max(0, int(out_begin) - int(IDX_TOPK));
    int high = min(int(out_begin), int(IDX_TOPK));
    while (low < high) {
        const int take_left = (low + high) >> 1;
        const int take_right = int(out_begin) - take_left;
        const int left_idx = left[take_left];
        const int right_prev_idx = right[take_right - 1];
        const float left_score = index_score(scores, score_base, left_idx);
        const float right_prev_score = index_score(scores, score_base, right_prev_idx);
        if (index_better(left_score, left_idx, right_prev_score, right_prev_idx)) {
            low = take_left + 1;
        } else {
            high = take_left;
        }
    }

    int i = low;
    int j = int(out_begin) - i;
    for (uint out = out_begin; out < out_end; ++out) {
        const int left_idx = i < int(IDX_TOPK) ? left[i] : -1;
        const int right_idx = j < int(IDX_TOPK) ? right[j] : -1;
        const float left_score = index_score(scores, score_base, left_idx);
        const float right_score = index_score(scores, score_base, right_idx);
        if (index_better(left_score, left_idx, right_score, right_idx)) {
            dst[out] = left_idx;
            ++i;
        } else {
            dst[out] = right_idx;
            ++j;
        }
    }
}

kernel void deepseek_indexer_topk_finalize_i32(
        constant DeepSeekIndexerParams &p [[buffer(0)]],
        constant DeepSeekIndexerTopKPlan &plan [[buffer(1)]],
        constant DeepSeekIndexerOutputLayout &layout [[buffer(2)]],
        device const int *indices [[buffer(3)]],
        device int *output [[buffer(4)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint slot = pos.x;
    const uint row = pos.y;
    if (slot >= IDX_TOPK || row >= p.batch * p.query_len) return;
    const uint query = row % p.query_len;
    const uint valid_count = min(p.kv_len, (p.start_pos + query + 1) / p.ratio);
    const int index = indices[ulong(row) * plan.scratch_row_stride + slot];
    output[ulong(row) * layout.row_stride + layout.column_offset + slot] =
        slot < min(valid_count, IDX_TOPK) && index >= 0 ? index + p.offset : -1;
}
