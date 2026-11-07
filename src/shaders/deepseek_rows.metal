// DeepSeek-V4 0731 decode cores over independent rows (joint decode).
//
// Each kernel here runs the one-row (batch 1, one query) body of an existing
// DeepSeek kernel for up to eight independent rows in one dispatch. Row `r`
// reads its cache through fixed binding slot `r` and its shape (kv length,
// width, position, index offset) from a per-row constant table. Every row's
// threadgroups execute exactly the one-row kernel's code with that row's
// loop bounds, tiling, reduction order and tie order, and read only that
// row's bindings, so row `r` of a rows dispatch is bitwise equal to the
// one-row dispatch of the existing kernel on row `r`.
//
// Sources of the copied bodies (kept untouched):
// - deepseek_sparse_attention.metal: score, reduce, gather, sanitize;
// - deepseek_indexer.metal: score_mma, topk_block, topk_merge, topk_finalize;
// - deepseek_compressor.metal: the nonzero-position append branch.

#include <metal_stdlib>
using namespace metal;

constant uint DS_ROWS_MAX = 8;

// Binding slots for one per-row buffer family: `name0..name7` at
// consecutive buffer indices starting at `first`.
#define DS_ROW_SLOTS(qual, type, name, first)                       \
    qual type *name##0 [[buffer(first + 0)]],                       \
    qual type *name##1 [[buffer(first + 1)]],                       \
    qual type *name##2 [[buffer(first + 2)]],                       \
    qual type *name##3 [[buffer(first + 3)]],                       \
    qual type *name##4 [[buffer(first + 4)]],                       \
    qual type *name##5 [[buffer(first + 5)]],                       \
    qual type *name##6 [[buffer(first + 6)]],                       \
    qual type *name##7 [[buffer(first + 7)]]

#define DS_ROW_SELECT(row, name)                                    \
    ((row) == 0 ? name##0 : (row) == 1 ? name##1 :                  \
     (row) == 2 ? name##2 : (row) == 3 ? name##3 :                  \
     (row) == 4 ? name##4 : (row) == 5 ? name##5 :                  \
     (row) == 6 ? name##6 : name##7)

// ---------------------------------------------------------------------------
// Sparse attention (selected shared-KV rows with learned sinks)
// ---------------------------------------------------------------------------

struct DeepSeekSparseRowsParams {
    uint rows;
    uint max_width;
    float scale;
    uint reserved;
    uint kv_len[DS_ROWS_MAX];
    uint width[DS_ROWS_MAX];
    // Element offset of the row's `width` indices in the indices buffer.
    uint index_base[DS_ROWS_MAX];
    // Element offsets of the row's gathered `[width, 512]` KV and `[width]`
    // mask (flash path only).
    uint gathered_base[DS_ROWS_MAX];
    uint mask_base[DS_ROWS_MAX];
    // Flash-batch slot of the row (flash path with permuted rows only).
    uint slot[DS_ROWS_MAX];
    // Row of the `[_, 64, 512]` query and output buffers the entry reads
    // and writes.
    uint q_row[DS_ROWS_MAX];
};

constant uint DS_HEADS = 64;
constant uint DS_DIM = 512;
constant uint DS_THREADS = 256;
constant uint DS_SIMDGROUPS = 8;
constant uint DS_SLOTS_PER_SIMDGROUP = 4;
constant uint DS_SLOTS_PER_GROUP = DS_SIMDGROUPS * DS_SLOTS_PER_SIMDGROUP;

// Row r's logits live at [r, 64, max_width]; head h at h * width[r] within it.
inline ulong ds_rows_score_base(constant DeepSeekSparseRowsParams &p, uint row, uint head) {
    return ulong(row) * DS_HEADS * p.max_width + ulong(head) * p.width[row];
}

kernel void deepseek_sparse_attention_score_rows_bf16(
        constant DeepSeekSparseRowsParams &p      [[buffer(0)]],
        device const bfloat *q                    [[buffer(1)]],
        device const int *indices                 [[buffer(2)]],
        device float *scores                      [[buffer(3)]],
        DS_ROW_SLOTS(device const, bfloat, kv, 4),
        uint group_id                             [[threadgroup_position_in_grid]],
        ushort lane                               [[thread_index_in_simdgroup]],
        ushort simdgroup                          [[simdgroup_index_in_threadgroup]]) {
    const uint blocks_per_head = (p.max_width + DS_SLOTS_PER_GROUP - 1) / DS_SLOTS_PER_GROUP;
    const uint slot_block = group_id % blocks_per_head;
    const uint head_group = group_id / blocks_per_head;
    const uint head = head_group % DS_HEADS;
    const uint row = head_group / DS_HEADS;
    if (row >= p.rows) return;
    const uint top_k = p.width[row];
    const uint kv_len = p.kv_len[row];
    device const bfloat *kv = DS_ROW_SELECT(row, kv);
    const ulong qbase = (ulong(p.q_row[row]) * DS_HEADS + head) * DS_DIM;
    const ulong ibase = p.index_base[row];
    const ulong score_base = ds_rows_score_base(p, row, head);

    for (uint iteration = 0; iteration < DS_SLOTS_PER_SIMDGROUP; ++iteration) {
        const uint slot = slot_block * DS_SLOTS_PER_GROUP +
            iteration * DS_SIMDGROUPS + simdgroup;
        if (slot >= top_k) continue;

        const int selected = indices[ibase + slot];
        const bool sentinel = selected == -1;
        const bool index_bad = selected < -1 || selected >= int(kv_len);
        float dot = 0.0f;
        uint bad = index_bad ? 1u : 0u;
        if (!sentinel && !index_bad) {
            const ulong kbase = ulong(uint(selected)) * DS_DIM;
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

kernel void deepseek_sparse_attention_reduce_rows_bf16(
        constant DeepSeekSparseRowsParams &p      [[buffer(0)]],
        device const float *sinks                 [[buffer(1)]],
        device const int *indices                 [[buffer(2)]],
        device float *scores                      [[buffer(3)]],
        device bfloat *output                     [[buffer(4)]],
        DS_ROW_SLOTS(device const, bfloat, kv, 5),
        uint head_group                           [[threadgroup_position_in_grid]],
        uint tid                                  [[thread_index_in_threadgroup]]) {
    const uint head = head_group % DS_HEADS;
    const uint row = head_group / DS_HEADS;
    if (row >= p.rows) return;
    const uint top_k = p.width[row];
    device const bfloat *kv = DS_ROW_SELECT(row, kv);
    const ulong ibase = p.index_base[row];
    const ulong score_base = ds_rows_score_base(p, row, head);
    const ulong output_base = (ulong(p.q_row[row]) * DS_HEADS + head) * DS_DIM;

    threadgroup float reduction[DS_THREADS];
    threadgroup uint bad[DS_THREADS];
    threadgroup float maximum;
    threadgroup float denominator;
    threadgroup uint invalid;

    const float sink = sinks[head];
    float local_max = tid == 0 && isfinite(sink) ? sink : -INFINITY;
    uint local_bad = isfinite(sink) ? 0u : 1u;
    for (uint slot = tid; slot < top_k; slot += DS_THREADS) {
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
    for (uint slot = tid; slot < top_k; slot += DS_THREADS) {
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
        for (uint slot = 0; slot < top_k; ++slot) {
            const float weight = scores[score_base + slot];
            if (weight > 0.0f) {
                const uint selected = uint(indices[ibase + slot]);
                const float value = float(kv[ulong(selected) * DS_DIM + feature]);
                numerator = fma(weight, value, numerator);
            }
        }
        const float result = numerator / denominator;
        output[output_base + feature] = bfloat(invalid == 0 && isfinite(result) ? result : 0.0f);
    }
}

// One thread per gathered element; rows are concatenated in row order
// (`width[r] * 512` threads each) and written to `gathered_base[r]`.
kernel void deepseek_sparse_attention_gather_rows_bf16(
        constant DeepSeekSparseRowsParams &p      [[buffer(0)]],
        device const int *indices                 [[buffer(1)]],
        device bfloat *gathered                   [[buffer(2)]],
        device bfloat *mask                       [[buffer(3)]],
        device atomic_uint *invalid_global        [[buffer(4)]],
        DS_ROW_SLOTS(device const, bfloat, kv, 5),
        uint gid                                  [[thread_position_in_grid]]) {
    uint row = DS_ROWS_MAX;
    uint begin = 0;
    for (uint r = 0; r < p.rows; ++r) {
        const uint elements = p.width[r] * DS_DIM;
        if (gid < begin + elements) {
            row = r;
            break;
        }
        begin += elements;
    }
    if (row >= p.rows) return;
    device const bfloat *kv = DS_ROW_SELECT(row, kv);
    const uint within_row = gid - begin;
    const uint slot = within_row / DS_DIM;
    const uint feature = within_row % DS_DIM;
    const ulong index_offset = ulong(p.index_base[row]) + slot;
    const int selected = indices[index_offset];
    const bool sentinel = selected == -1;
    const bool index_bad = selected < -1 || selected >= int(p.kv_len[row]);
    float value = 0.0f;
    if (!sentinel && !index_bad) {
        value = float(kv[ulong(uint(selected)) * DS_DIM + feature]);
    }
    const bool value_bad = !isfinite(value);
    gathered[ulong(p.gathered_base[row]) + within_row] =
        bfloat(!index_bad && !value_bad ? value : 0.0f);
    if (feature == 0) {
        mask[ulong(p.mask_base[row]) + slot] = bfloat(sentinel ? -INFINITY : 0.0f);
    }
    if (index_bad || value_bad) {
        atomic_store_explicit(&invalid_global[row], 1u, memory_order_relaxed);
    }
}

// Copy entry e's query (row `q_row[e]`) to flash-batch slot `slot[e]`.
kernel void deepseek_sparse_attention_permute_q_rows_bf16(
        constant DeepSeekSparseRowsParams &p      [[buffer(0)]],
        device const bfloat *q                    [[buffer(1)]],
        device bfloat *permuted                   [[buffer(2)]],
        uint gid                                  [[thread_position_in_grid]]) {
    const uint row_elements = DS_HEADS * DS_DIM;
    const uint row = gid / row_elements;
    if (row >= p.rows) return;
    const uint within_row = gid % row_elements;
    permuted[ulong(p.slot[row]) * row_elements + within_row] =
        q[ulong(p.q_row[row]) * row_elements + within_row];
}

// Entry e's output row `q_row[e]` is flash slot `slot[e]` of `source`,
// zeroed exactly as the one-row sanitize kernel zeroes it in place.
// `invalid_global` is per entry, `invalid_heads` per flash slot and head.
kernel void deepseek_sparse_attention_sanitize_rows_bf16(
        constant DeepSeekSparseRowsParams &p      [[buffer(0)]],
        device const atomic_uint *invalid_global  [[buffer(1)]],
        device const atomic_uint *invalid_heads   [[buffer(2)]],
        device const bfloat *source               [[buffer(3)]],
        device bfloat *output                     [[buffer(4)]],
        uint gid                                  [[thread_position_in_grid]]) {
    const uint row_elements = DS_HEADS * DS_DIM;
    const uint row = gid / row_elements;
    if (row >= p.rows) return;
    const uint within_row = gid % row_elements;
    const uint slot = p.slot[row];
    const uint head_group = slot * DS_HEADS + within_row / DS_DIM;
    const bfloat stored = source[ulong(slot) * row_elements + within_row];
    const float value = float(stored);
    const bool invalid = atomic_load_explicit(&invalid_global[row], memory_order_relaxed) != 0u
        || atomic_load_explicit(&invalid_heads[head_group], memory_order_relaxed) != 0u
        || !isfinite(value);
    output[ulong(p.q_row[row]) * row_elements + within_row] = invalid ? bfloat(0.0f) : stored;
}

// ---------------------------------------------------------------------------
// Lightning indexer score and deterministic top-512 (one query per row)
// ---------------------------------------------------------------------------

struct DeepSeekIndexerRowsParams {
    uint rows;
    uint query_len;
    uint ratio;
    uint score_stride;
    uint max_blocks;
    uint merge_round;
    uint reserved0;
    uint reserved1;
    uint kv_len[DS_ROWS_MAX];
    uint start_pos[DS_ROWS_MAX];
    int offset[DS_ROWS_MAX];
    // Batch row of the entry's query and head weights.
    uint q_row[DS_ROWS_MAX];
    // Element offset of the entry's 512 output indices.
    uint output_base[DS_ROWS_MAX];
};

constant uint IDX_HEADS = 64;
constant uint IDX_DIM = 128;
constant uint IDX_TOPK = 512;
constant uint IDX_SCORE_SIMDGROUPS = 8;
constant uint IDX_KEYS_PER_SIMDGROUP = 8;
constant uint IDX_KEYS_PER_GROUP = IDX_SCORE_SIMDGROUPS * IDX_KEYS_PER_SIMDGROUP;
constant uint IDX_HEADS_PER_TILE = 8;
constant uint IDX_QUERIES_PER_GROUP = 8;
constant uint IDX_BLOCK_THREADS = 1024;

inline bool index_rows_better(float score, int index, float other_score, int other_index) {
    if (index < 0) return false;
    if (other_index < 0) return true;
    return isfinite(score) &&
        (!isfinite(other_score) || score > other_score ||
         (score == other_score && index < other_index));
}

inline float index_rows_score(device const float *scores, ulong row_base, int index) {
    return index < 0 ? -INFINITY : scores[row_base + uint(index)];
}

inline uint index_rows_block_count(constant DeepSeekIndexerRowsParams &p, uint entry) {
    return (p.kv_len[entry] + IDX_BLOCK_THREADS - 1) / IDX_BLOCK_THREADS;
}

kernel void deepseek_indexer_score_mma_rows_bf16(
        constant DeepSeekIndexerRowsParams &p [[buffer(0)]],
        device const bfloat *q_all          [[buffer(1)]],
        device const float *weights_all     [[buffer(2)]],
        device float *scores_all            [[buffer(3)]],
        DS_ROW_SLOTS(device const, bfloat, kv, 4),
        uint3 group                        [[threadgroup_position_in_grid]],
        ushort tid                         [[thread_index_in_threadgroup]],
        ushort lane                        [[thread_index_in_simdgroup]],
        ushort simdgroup                   [[simdgroup_index_in_threadgroup]]) {
    const uint entry = group.z;
    if (entry >= p.rows) return;
    const uint kv_len = p.kv_len[entry];
    const uint key_group_start = group.x * IDX_KEYS_PER_GROUP;
    if (key_group_start >= kv_len) return;
    const uint row_start = group.y * IDX_QUERIES_PER_GROUP;
    device const bfloat *kv = DS_ROW_SELECT(entry, kv);
    device const bfloat *q = q_all + ulong(p.q_row[entry]) * IDX_HEADS * IDX_DIM;
    device const float *weights = weights_all + ulong(p.q_row[entry]) * IDX_HEADS;
    device float *scores = scores_all + ulong(entry) * p.score_stride;
    const uint start_pos = p.start_pos[entry];

    threadgroup half staged_k[IDX_KEYS_PER_GROUP * IDX_DIM];
    threadgroup half staged_q[IDX_HEADS_PER_TILE * IDX_DIM];
    threadgroup float staged_weights[IDX_HEADS_PER_TILE];
    threadgroup float qk_tiles[
        IDX_SCORE_SIMDGROUPS * IDX_HEADS_PER_TILE * IDX_KEYS_PER_SIMDGROUP];

    for (uint i = tid; i < IDX_KEYS_PER_GROUP * IDX_DIM; i += 256) {
        const uint local_key = i / IDX_DIM;
        const uint feature = i % IDX_DIM;
        const uint key = key_group_start + local_key;
        staged_k[i] = key < kv_len
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
        const uint valid_count = min(kv_len, (start_pos + row + 1) / p.ratio);
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
            if (key < kv_len) {
                scores[ulong(row) * kv_len + key] =
                    key < valid_count && isfinite(score) ? score : -INFINITY;
            }
        }
    }
}

// Entry e owns `[2, rows, max_blocks * 512]` ping-pong lists at row e.
kernel void deepseek_indexer_topk_block_rows_i32(
        constant DeepSeekIndexerRowsParams &p [[buffer(0)]],
        device const float *scores_all [[buffer(1)]],
        device int *indices [[buffer(2)]],
        threadgroup int *local_indices [[threadgroup(0)]],
        uint group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint entry = group_id / p.max_blocks;
    if (entry >= p.rows) return;
    const uint block = group_id % p.max_blocks;
    if (block >= index_rows_block_count(p, entry)) return;
    const uint kv_len = p.kv_len[entry];
    const uint block_start = block * IDX_BLOCK_THREADS;
    const uint candidate = block_start + tid;
    const uint query = 0 % p.query_len;
    const uint valid_count = min(kv_len, (p.start_pos[entry] + query + 1) / p.ratio);
    const ulong score_base = ulong(entry) * p.score_stride;
    device const float *scores = scores_all;

    local_indices[tid] = candidate < valid_count &&
        isfinite(scores[score_base + candidate]) ? int(candidate) : -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 2; width <= IDX_BLOCK_THREADS; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            const uint peer = tid ^ stride;
            if (peer > tid) {
                const int lhs = local_indices[tid];
                const int rhs = local_indices[peer];
                const float lhs_score = index_rows_score(scores, score_base, lhs);
                const float rhs_score = index_rows_score(scores, score_base, rhs);
                const bool ascending_half = (tid & width) != 0;
                const bool rhs_better = index_rows_better(rhs_score, rhs, lhs_score, lhs);
                const bool lhs_better = index_rows_better(lhs_score, lhs, rhs_score, rhs);
                if ((!ascending_half && rhs_better) || (ascending_half && lhs_better)) {
                    local_indices[tid] = rhs;
                    local_indices[peer] = lhs;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < IDX_TOPK) {
        const ulong out = ulong(entry) * p.max_blocks * IDX_TOPK + ulong(block) * IDX_TOPK + tid;
        indices[out] = local_indices[tid];
    }
}

// Merge round `merge_round`: an entry with L lists before this round merges
// pairs into ceil(L / 2) lists; an entry already down to one list copies it.
kernel void deepseek_indexer_topk_merge_rows_i32(
        constant DeepSeekIndexerRowsParams &p [[buffer(0)]],
        device const float *scores [[buffer(1)]],
        device const int *input [[buffer(2)]],
        device int *output [[buffer(3)]],
        uint group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]]) {
    uint max_lists = p.max_blocks;
    for (uint round = 0; round < p.merge_round; ++round) max_lists = (max_lists + 1) / 2;
    const uint max_merged = (max_lists + 1) / 2;
    const uint entry = group_id / max_merged;
    if (entry >= p.rows) return;
    uint list_count = index_rows_block_count(p, entry);
    for (uint round = 0; round < p.merge_round; ++round) list_count = (list_count + 1) / 2;
    const uint merged_lists = (list_count + 1) / 2;
    const uint pair = group_id % max_merged;
    if (pair >= merged_lists) return;
    const uint left_list = pair * 2;
    const uint right_list = left_list + 1;
    const ulong row_base = ulong(entry) * p.max_blocks * IDX_TOPK;
    const ulong score_base = ulong(entry) * p.score_stride;
    device const int *left = input + row_base + ulong(left_list) * IDX_TOPK;
    device const int *right = right_list < list_count
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

    int low = max(0, int(out_begin) - int(IDX_TOPK));
    int high = min(int(out_begin), int(IDX_TOPK));
    while (low < high) {
        const int take_left = (low + high) >> 1;
        const int take_right = int(out_begin) - take_left;
        const int left_idx = left[take_left];
        const int right_prev_idx = right[take_right - 1];
        const float left_score = index_rows_score(scores, score_base, left_idx);
        const float right_prev_score = index_rows_score(scores, score_base, right_prev_idx);
        if (index_rows_better(left_score, left_idx, right_prev_score, right_prev_idx)) {
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
        const float left_score = index_rows_score(scores, score_base, left_idx);
        const float right_score = index_rows_score(scores, score_base, right_idx);
        if (index_rows_better(left_score, left_idx, right_score, right_idx)) {
            dst[out] = left_idx;
            ++i;
        } else {
            dst[out] = right_idx;
            ++j;
        }
    }
}

kernel void deepseek_indexer_topk_finalize_rows_i32(
        constant DeepSeekIndexerRowsParams &p [[buffer(0)]],
        device const int *indices [[buffer(1)]],
        device int *output [[buffer(2)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint slot = pos.x;
    const uint entry = pos.y;
    if (slot >= IDX_TOPK || entry >= p.rows) return;
    const uint query = 0 % p.query_len;
    const uint valid_count = min(p.kv_len[entry], (p.start_pos[entry] + query + 1) / p.ratio);
    const int index = indices[ulong(entry) * p.max_blocks * IDX_TOPK + slot];
    output[ulong(p.output_base[entry]) + slot] =
        slot < min(valid_count, IDX_TOPK) && index >= 0 ? index + p.offset[entry] : -1;
}

// ---------------------------------------------------------------------------
// Learned KV compressor, one decode token per row (state per row)
// ---------------------------------------------------------------------------

struct DeepSeekCompressorRowsParams {
    uint rows;
    uint seq_len;
    uint ratio;
    uint head_dim;
    float epsilon;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint start_pos[DS_ROWS_MAX];
};

constant uint COMP_THREADS = 256;

inline float safe_rows_state_value(float value) {
    return isfinite(value) ? value : 0.0f;
}

// The nonzero-position append branch of deepseek_compressor_bf16 at
// `seq_len` tokens (decode: one) for row `group.x`, with write_cache = 0.
kernel void deepseek_compressor_rows_bf16(
        constant DeepSeekCompressorRowsParams &p [[buffer(0)]],
        device const float *kv_all            [[buffer(1)]],
        device const float *score_all         [[buffer(2)]],
        device const float *ape               [[buffer(3)]],
        device const float *norm              [[buffer(4)]],
        device bfloat *output_all             [[buffer(5)]],
        DS_ROW_SLOTS(device, float, kv_state, 6),
        DS_ROW_SLOTS(device, float, score_state, 14),
        uint3 group                           [[threadgroup_position_in_grid]],
        uint tid                              [[thread_index_in_threadgroup]]) {
    const uint row_index = group.x;
    if (row_index >= p.rows) return;
    const uint start_pos = p.start_pos[row_index];
    const bool overlap = p.ratio == 4;
    const uint coff = overlap ? 2 : 1;
    const uint projected = coff * p.head_dim;
    const uint append_output_count =
        (start_pos + p.seq_len) / p.ratio - start_pos / p.ratio;
    const uint output_slots = max(1u, append_output_count);
    const uint batch = 0;
    device float *kv_state = DS_ROW_SELECT(row_index, kv_state);
    device float *score_state = DS_ROW_SELECT(row_index, score_state);
    device const float *kv = kv_all + ulong(row_index) * p.seq_len * projected;
    device const float *score = score_all + ulong(row_index) * p.seq_len * projected;
    device bfloat *output = output_all + ulong(row_index) * output_slots * p.head_dim;
    const ulong state_batch = ulong(batch) * coff * p.ratio * projected;
    const ulong input_batch = ulong(batch) * p.seq_len * projected;

    threadgroup float sums[COMP_THREADS];
    threadgroup uint bad[COMP_THREADS];
    threadgroup float rms_scale;
    threadgroup uint row_invalid;

    for (uint slot = 0; slot < output_slots; ++slot) {
        const ulong base = (ulong(batch) * output_slots + slot) * p.head_dim;
        for (uint feature = tid; feature < p.head_dim; feature += COMP_THREADS) {
            output[base + feature] = bfloat(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint emitted = 0;
    for (uint row = 0; row < p.seq_len; ++row) {
        const uint absolute = start_pos + row;
        const uint token = absolute % p.ratio;
        const uint slot = (overlap ? p.ratio : 0) + token;
        for (uint feature = tid; feature < projected; feature += COMP_THREADS) {
            const ulong src = input_batch + ulong(row) * projected + feature;
            const float v = kv[src];
            const float s = score[src] + ape[token * projected + feature];
            const ulong dst = state_batch + ulong(slot) * projected + feature;
            kv_state[dst] = safe_rows_state_value(v);
            score_state[dst] = isfinite(s) && isfinite(v) ? s : NAN;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((absolute + 1) % p.ratio == 0) {
            float compressed[2] = { 0.0f, 0.0f };
            uint local_bad = 0;
            for (uint part = 0; part < 2; ++part) {
                const uint feature = tid + part * COMP_THREADS;
                if (feature >= p.head_dim) continue;
                float maximum = -INFINITY;
                const uint window = overlap ? 2 * p.ratio : p.ratio;
                for (uint item = 0; item < window; ++item) {
                    const uint source_feature =
                        overlap && item >= p.ratio ? p.head_dim + feature : feature;
                    const ulong src = state_batch + ulong(item) * projected + source_feature;
                    const float s = score_state[src];
                    local_bad |=
                        (isnan(s) || s == INFINITY || !isfinite(kv_state[src])) ? 1u : 0u;
                    maximum = max(maximum, s);
                }
                float denominator = 0.0f;
                float numerator = 0.0f;
                for (uint item = 0; item < window; ++item) {
                    const uint source_feature =
                        overlap && item >= p.ratio ? p.head_dim + feature : feature;
                    const ulong src = state_batch + ulong(item) * projected + source_feature;
                    const float s = score_state[src];
                    const float v = kv_state[src];
                    const float weight = exp(s - maximum);
                    denominator += weight;
                    numerator = fma(weight, v, numerator);
                }
                const float pooled = numerator / denominator;
                compressed[part] = float(bfloat(isfinite(pooled) ? pooled : 0.0f));
                local_bad |= (!isfinite(pooled) || !isfinite(norm[feature])) ? 1u : 0u;
            }

            bad[tid] = local_bad;
            sums[tid] = compressed[0] * compressed[0] + compressed[1] * compressed[1];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = COMP_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    bad[tid] += bad[tid + stride];
                    sums[tid] += sums[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                row_invalid = bad[0] != 0 || !isfinite(sums[0]);
                rms_scale = row_invalid == 0
                    ? rsqrt(sums[0] / float(p.head_dim) + p.epsilon)
                    : 0.0f;
                if (!isfinite(rms_scale)) row_invalid = 1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float normalized[2] = { 0.0f, 0.0f };
            for (uint part = 0; part < 2; ++part) {
                const uint feature = tid + part * COMP_THREADS;
                if (feature < p.head_dim) {
                    normalized[part] = compressed[part] * rms_scale * norm[feature];
                }
            }
            bad[tid] = (!isfinite(normalized[0]) || !isfinite(normalized[1])) ? 1u : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = COMP_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) bad[tid] += bad[tid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0 && bad[0] != 0) row_invalid = 1;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const ulong output_base =
                (ulong(batch) * output_slots + emitted) * p.head_dim;
            for (uint part = 0; part < 2; ++part) {
                const uint feature = tid + part * COMP_THREADS;
                if (feature < p.head_dim) {
                    const bfloat result =
                        bfloat(row_invalid == 0 ? normalized[part] : 0.0f);
                    output[output_base + feature] = result;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (overlap) {
                const uint copy_count = p.ratio * projected;
                for (uint i = tid; i < copy_count; i += COMP_THREADS) {
                    kv_state[state_batch + i] =
                        kv_state[state_batch + ulong(p.ratio) * projected + i];
                    score_state[state_batch + i] =
                        score_state[state_batch + ulong(p.ratio) * projected + i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            emitted += 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
