#include <metal_stdlib>
using namespace metal;

constant uint DFLASH2_MAX_LANES = 4;
constant uint DFLASH2_SHARDS = 8;
constant uint DFLASH2_CANDIDATES = 16;
constant uint DFLASH2_SELECTOR_RANK = 256;

struct DFlash2SelectorParams {
    uint anchor[DFLASH2_MAX_LANES];
    float temperature[DFLASH2_MAX_LANES];
    uint lanes;
    uint rows_per_lane;
    uint vocabulary;
    uint sampling_mask;
};

inline bool dflash2_top_beats(float value, uint token, float other, uint other_token) {
    return value > other || (value == other && token < other_token);
}

inline void dflash2_top16_insert(thread float (&values)[16],
                                  thread uint (&ids)[16],
                                  float value, uint token) {
#pragma clang loop unroll(full)
    for (uint slot = 15; slot > 0; --slot) {
        bool here = dflash2_top_beats(value, token, values[slot], ids[slot]);
        bool above = dflash2_top_beats(value, token, values[slot - 1], ids[slot - 1]);
        values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
        ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
    }
    bool top = dflash2_top_beats(value, token, values[0], ids[0]);
    values[0] = top ? value : values[0];
    ids[0] = top ? token : ids[0];
}

template <uint Count>
inline void dflash2_top_pop(thread float (&values)[Count],
                            thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
    for (uint slot = 0; slot + 1 < Count; ++slot) {
        values[slot] = values[slot + 1];
        ids[slot] = ids[slot + 1];
    }
    values[Count - 1] = -INFINITY;
    ids[Count - 1] = 0xffffffffu;
}

inline void dflash2_simd_best(float value, uint token,
                               thread float &best, thread uint &best_token) {
    best = simd_max(value);
    best_token = simd_min(value == best ? token : 0xffffffffu);
}

kernel void dflash2_select_top16(
    device const float *logits [[buffer(0)]],
    device uint *partial_ids [[buffer(1)]],
    device float *partial_values [[buffer(2)]],
    constant DFlash2SelectorParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint K = DFLASH2_CANDIDATES;
    constexpr uint VectorTokens = 4;
    constexpr uint ChunkVectors = 16;
    const uint positions = params.rows_per_lane - 1;
    const uint batch = group / (positions * DFLASH2_SHARDS);
    const uint local = group % (positions * DFLASH2_SHARDS);
    const uint position = local / DFLASH2_SHARDS;
    const uint shard = local % DFLASH2_SHARDS;
    const ulong row_start = (ulong(batch) * params.rows_per_lane + position + 1) * params.vocabulary;
    const uint shard_tokens = (params.vocabulary + DFLASH2_SHARDS - 1) / DFLASH2_SHARDS;
    const uint begin = min(shard * shard_tokens, params.vocabulary);
    const uint end = min(begin + shard_tokens, params.vocabulary);
    uint head = uint((VectorTokens - (row_start + begin) % VectorTokens) % VectorTokens);
    head = min(head, end - begin);
    const uint vectors = (end - begin - head) / VectorTokens;
    const uint vector_begin = begin + head;
    const uint tail_begin = vector_begin + vectors * VectorTokens;
    device const float *row = logits + row_start;
    device const float4 *vector_row = reinterpret_cast<device const float4 *>(row + vector_begin);

    float values[K];
    uint ids[K];
    for (uint i = 0; i < K; ++i) {
        values[i] = -INFINITY;
        ids[i] = 0xffffffffu;
    }
    if (thread_index < head) {
        uint token = begin + thread_index;
        float value = row[token];
        if (dflash2_top_beats(value, token, values[K - 1], ids[K - 1]))
            dflash2_top16_insert(values, ids, value, token);
    }
    if (thread_index < end - tail_begin) {
        uint token = tail_begin + thread_index;
        float value = row[token];
        if (dflash2_top_beats(value, token, values[K - 1], ids[K - 1]))
            dflash2_top16_insert(values, ids, value, token);
    }

    threadgroup float maxima[256];
    threadgroup float thresholds[8];
    for (uint chunk = 0; chunk < vectors; chunk += 256 * ChunkVectors) {
        float4 loaded[ChunkVectors];
        float best = -INFINITY;
        for (uint i = 0; i < ChunkVectors; ++i) {
            uint index = chunk + thread_index + i * 256;
            loaded[i] = index < vectors ? vector_row[index] : float4(0.0f);
            if (index < vectors) {
                for (uint j = 0; j < VectorTokens; ++j)
                    if (loaded[i][j] > best) best = loaded[i][j];
            }
        }
        maxima[thread_index] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint above = 0;
        for (uint other = 0; other < 256; ++other)
            above += maxima[other] > best ? 1u : 0u;
        float candidate = simd_min(above < K ? best : INFINITY);
        if (lane == 0) thresholds[simd_group] = candidate;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float threshold = thresholds[0];
        for (uint other = 1; other < 8; ++other)
            threshold = min(threshold, thresholds[other]);
        for (uint i = 0; i < ChunkVectors; ++i) {
            uint index = chunk + thread_index + i * 256;
            if (index >= vectors) continue;
            uint token = vector_begin + index * VectorTokens;
            for (uint j = 0; j < VectorTokens; ++j) {
                float value = loaded[i][j];
                if (value >= threshold &&
                    dflash2_top_beats(value, token + j, values[K - 1], ids[K - 1]))
                    dflash2_top16_insert(values, ids, value, token + j);
            }
        }
    }

    threadgroup float round_values[2][8];
    threadgroup uint round_ids[2][8];
    for (uint rank = 0; rank < K; ++rank) {
        float head_value = values[0];
        uint head_id = ids[0];
        float simd_value;
        uint simd_id;
        dflash2_simd_best(head_value, head_id, simd_value, simd_id);
        uint slot = rank & 1;
        if (lane == 0) {
            round_values[slot][simd_group] = simd_value;
            round_ids[slot][simd_group] = simd_id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = round_values[slot][0];
        uint best_id = round_ids[slot][0];
        for (uint other = 1; other < 8; ++other) {
            float value = round_values[slot][other];
            uint token = round_ids[slot][other];
            if (dflash2_top_beats(value, token, best, best_id)) {
                best = value;
                best_id = token;
            }
        }
        if (thread_index == rank) {
            partial_ids[group * K + rank] = best_id;
            partial_values[group * K + rank] = best;
        }
        if (head_value == best && head_id == best_id)
            dflash2_top_pop<K>(values, ids);
    }
}

inline void dflash2_merge_shards(device const uint *partial_ids,
                                  device const float *partial_values,
                                  uint row, uint lane,
                                  thread float &value, thread uint &token) {
    constexpr uint K = DFLASH2_CANDIDATES;
    constexpr uint Entries = DFLASH2_SHARDS * K / 32;
    uint origin = row * DFLASH2_SHARDS * K + lane * Entries;
    float values[Entries];
    uint ids[Entries];
    for (uint i = 0; i < Entries; ++i) {
        values[i] = partial_values[origin + i];
        ids[i] = partial_ids[origin + i];
    }
    value = -INFINITY;
    token = 0xffffffffu;
    for (uint rank = 0; rank < K; ++rank) {
        float best;
        uint best_id;
        dflash2_simd_best(values[0], ids[0], best, best_id);
        if (lane == rank) {
            value = best;
            token = best_id;
        }
        if (values[0] == best && ids[0] == best_id)
            dflash2_top_pop<Entries>(values, ids);
    }
}

kernel void dflash2_select_edges(
    device const uint *partial_ids [[buffer(0)]],
    device const float *partial_values [[buffer(1)]],
    device uint *candidates [[buffer(2)]],
    device float *unary [[buffer(3)]],
    device float *edges [[buffer(4)]],
    device const bfloat *hidden [[buffer(5)]],
    device const bfloat *predecessor_codebook [[buffer(6)]],
    device const bfloat *successor_codebook [[buffer(7)]],
    constant DFlash2SelectorParams &params [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint positions = params.rows_per_lane - 1;
    const uint batch = row / positions;
    const uint position = row % positions;
    threadgroup uint successors[DFLASH2_CANDIDATES];
    threadgroup uint predecessors[DFLASH2_CANDIDATES];
    if (simd_group == 0) {
        float value;
        uint token;
        dflash2_merge_shards(partial_ids, partial_values, row, lane, value, token);
        if (lane < DFLASH2_CANDIDATES) {
            candidates[row * DFLASH2_CANDIDATES + lane] = token;
            unary[row * DFLASH2_CANDIDATES + lane] = value;
            successors[lane] = token;
        }
    } else if (simd_group == 1) {
        uint token = params.anchor[batch];
        if (position > 0) {
            float value;
            dflash2_merge_shards(partial_ids, partial_values, row - 1, lane, value, token);
        }
        if (lane < DFLASH2_CANDIDATES) predecessors[lane] = token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint TaskCandidates = 8;
    constexpr uint Dims = DFLASH2_SELECTOR_RANK / 32;
    uint tasks = (position > 0 ? DFLASH2_CANDIDATES : 1) *
                 (DFLASH2_CANDIDATES / TaskCandidates);
    device float *table = edges + ulong(row) * DFLASH2_CANDIDATES * DFLASH2_CANDIDATES;
    device const bfloat *row_hidden = hidden +
        (ulong(batch) * params.rows_per_lane + position + 1) * DFLASH2_SELECTOR_RANK;
    for (uint task = simd_group; task < tasks; task += threads / 32) {
        uint predecessor_index = task / (DFLASH2_CANDIDATES / TaskCandidates);
        uint first_candidate = task % (DFLASH2_CANDIDATES / TaskCandidates) * TaskCandidates;
        uint safe_predecessor = min(predecessors[predecessor_index], params.vocabulary - 1);
        float context[Dims];
        float successor[TaskCandidates][Dims];
        for (uint i = 0; i < Dims; ++i) {
            uint dim = lane + i * 32;
            context[i] = float(predecessor_codebook[safe_predecessor * DFLASH2_SELECTOR_RANK + dim]) *
                         float(row_hidden[dim]);
        }
        for (uint j = 0; j < TaskCandidates; ++j) {
            uint safe_candidate = min(successors[first_candidate + j], params.vocabulary - 1);
            for (uint i = 0; i < Dims; ++i)
                successor[j][i] = float(successor_codebook[
                    safe_candidate * DFLASH2_SELECTOR_RANK + lane + i * 32]);
        }
        for (uint j = 0; j < TaskCandidates; ++j) {
            float score = 0.0f;
            for (uint i = 0; i < Dims; ++i) score += context[i] * successor[j][i];
            score = simd_sum(score);
            if (lane == 0)
                table[predecessor_index * DFLASH2_CANDIDATES + first_candidate + j] = score;
        }
    }
}

kernel void dflash2_select_path(
    device const uint *candidates [[buffer(0)]],
    device const float *unary [[buffer(1)]],
    device const float *edges [[buffer(2)]],
    device const float *uniforms [[buffer(3)]],
    device uint *tokens [[buffer(4)]],
    device float *proposal_probabilities [[buffer(5)]],
    device float *confidence [[buffer(6)]],
    constant DFlash2SelectorParams &params [[buffer(7)]],
    uint batch [[thread_position_in_grid]]) {
    const uint positions = params.rows_per_lane - 1;
    candidates += batch * positions * DFLASH2_CANDIDATES;
    unary += batch * positions * DFLASH2_CANDIDATES;
    edges += batch * positions * DFLASH2_CANDIDATES * DFLASH2_CANDIDATES;
    uniforms += batch * positions;
    tokens += batch * positions;
    proposal_probabilities += batch * positions * DFLASH2_CANDIDATES;
    confidence += batch * positions;
    const bool sampling = (params.sampling_mask & (1u << batch)) != 0;
    uint predecessor_index = 0;
    for (uint position = 0; position < positions; ++position) {
        device const float *position_edges = edges +
            (position * DFLASH2_CANDIDATES + predecessor_index) * DFLASH2_CANDIDATES;
        float scores[DFLASH2_CANDIDATES];
        for (uint i = 0; i < DFLASH2_CANDIDATES; ++i)
            scores[i] = unary[position * DFLASH2_CANDIDATES + i] + position_edges[i];
        float best = scores[0];
        for (uint i = 1; i < DFLASH2_CANDIDATES; ++i)
            best = max(best, scores[i]);
        float mass = 0.0f;
        for (uint i = 0; i < DFLASH2_CANDIDATES; ++i)
            mass += exp(scores[i] - best);
        confidence[position] = 1.0f / mass;

        uint selected = 0;
        if (sampling) {
            float sum = 0.0f;
            for (uint i = 0; i < DFLASH2_CANDIDATES; ++i) {
                float probability = exp((scores[i] - best) / params.temperature[batch]);
                proposal_probabilities[position * DFLASH2_CANDIDATES + i] = probability;
                sum += probability;
            }
            float cumulative = 0.0f;
            selected = DFLASH2_CANDIDATES - 1;
            for (uint i = 0; i < DFLASH2_CANDIDATES; ++i) {
                float probability =
                    proposal_probabilities[position * DFLASH2_CANDIDATES + i] / sum;
                proposal_probabilities[position * DFLASH2_CANDIDATES + i] = probability;
                cumulative += probability;
                if (selected == DFLASH2_CANDIDATES - 1 &&
                    cumulative > uniforms[position])
                    selected = i;
            }
        } else {
            for (uint i = 1; i < DFLASH2_CANDIDATES; ++i)
                if (scores[i] > scores[selected]) selected = i;
        }

        predecessor_index = selected;
        tokens[position] = candidates[position * DFLASH2_CANDIDATES + selected];
    }
}
struct DFlash2VerifyRowsParams {
    float slope;
    float intercept[15];
    float cost[65];
    uint lanes;
    uint rows_per_lane;
    uint every_row;
    uint maximum[DFLASH2_MAX_LANES];
};

kernel void dflash2_select_verify_rows(
    device const float *confidence [[buffer(0)]],
    device uint *verify_rows [[buffer(1)]],
    constant DFlash2VerifyRowsParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    const uint positions = params.rows_per_lane - 1;
    uint rows[DFLASH2_MAX_LANES];
    uint total = 0;
    for (uint lane = 0; lane < params.lanes; ++lane) {
        rows[lane] = params.every_row != 0 ? params.maximum[lane] : 1;
        total += rows[lane];
    }
    if (params.every_row == 0) {
        float survivals[DFLASH2_MAX_LANES * 15];
        uint owners[DFLASH2_MAX_LANES * 15];
        uint count = 0;
        for (uint lane = 0; lane < params.lanes; ++lane) {
            float survival = 1.0f;
            for (uint position = 0; position + 1 < params.maximum[lane]; ++position) {
                const float q = clamp(
                    confidence[lane * positions + position],
                    1.0e-6f,
                    1.0f - 1.0e-6f);
                const float z = params.slope * log(q / (1.0f - q)) +
                                params.intercept[position];
                survival *= 1.0f / (1.0f + exp(-z));
                uint at = count++;
                for (; at > 0 && survivals[at - 1] < survival; --at) {
                    survivals[at] = survivals[at - 1];
                    owners[at] = owners[at - 1];
                }
                survivals[at] = survival;
                owners[at] = lane;
            }
        }
        float expected = float(params.lanes);
        float best = expected / params.cost[total];
        uint taken = 0;
        for (uint i = 0; i < count; ++i) {
            expected += survivals[i];
            const float value = expected / params.cost[total + i + 1];
            if (value > best) {
                best = value;
                taken = i + 1;
            }
        }
        for (uint i = 0; i < taken; ++i)
            ++rows[owners[i]];
    }
    for (uint lane = 0; lane < params.lanes; ++lane)
        verify_rows[lane] = rows[lane];
}

struct DFlash2ComposeVerifyInputParams {
    uint anchor[DFLASH2_MAX_LANES];
    uint physical_lane[DFLASH2_MAX_LANES];
    uint rows[DFLASH2_MAX_LANES];
    uint physical_lanes;
    uint lanes;
    uint rows_per_lane;
    uint width;
};

kernel void dflash2_compose_verify_input(
    device const uint *proposals [[buffer(0)]],
    device uint *target_tokens [[buffer(1)]],
    constant DFlash2ComposeVerifyInputParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.width) return;
    uint row = index;
    uint lane = 0;
    for (; lane < params.lanes; ++lane) {
        if (row < params.rows[lane]) break;
        row -= params.rows[lane];
    }
    // Padding rows replay the first anchor, matching the fixed-width target
    // graph's cache-free padding policy.
    if (lane == params.lanes) {
        target_tokens[index] = params.anchor[0];
        return;
    }
    if (row == 0) {
        target_tokens[index] = params.anchor[lane];
        return;
    }
    const uint source = params.physical_lane[lane] * (params.rows_per_lane - 1) + row - 1;
    target_tokens[index] = proposals[source];
}
