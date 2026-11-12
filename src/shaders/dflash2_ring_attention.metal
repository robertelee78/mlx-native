// DFlash2 draft attention over the persistent K/V rings, read in place.
//
// The split/reduce structure and the online softmax are derived from Splash's
// Apache-2.0 draft attention (https://github.com/incoai/splash, commit
// 031a25a6: runtime/metal/kernels/decode/draft.metal).
//
// One threadgroup runs one split of one KV head of one lane: the head's four
// query heads over the lane's eight query rows, as 32 attention rows. The
// live window occupies one or two physical ranges of the ring; the ring tiles
// of 128 slots touching them are dealt round-robin to the splits, each split
// runs an online softmax over its tiles, and the last split adds the lane's
// eight current rows. The reduce kernel combines the splits in split order.
//
// Layouts:
// - queries  BF16 [lanes][q_heads][8][128]
// - ring     BF16 [lanes][kv_heads][window][128], slot = position % window
// - current keys   BF16 [lanes][kv_heads][8][128]
// - current values BF16 [lanes][kv_heads][16][128], rows 8..15 zero
// - partials F32  [lanes][kv_heads][4 splits][32 * 128 + 2 * 32]
// - output   BF16 [lanes * 8][q_heads * 128]
//
// Query row r of a lane whose ring holds `length` rows is at position
// length + r and sees ring positions in [max(0, length + r - (window - 1)),
// length) and all eight current rows. An attention row's result depends on
// its query, the lane's ring and current rows and the lane's length alone.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr ushort kRows = 8;
constant constexpr ushort kGroupHeads = 4;
constant constexpr ushort kAttentionRows = kGroupHeads * kRows;
constant constexpr ushort kHeadDim = 128;
constant constexpr ushort kTile = 128;
constant constexpr ushort kStep = 64;
constant constexpr ushort kCurrentValues = 16;
constant constexpr uint kSplits = 4;
constant constexpr uint kPartialFloats = kAttentionRows * kHeadDim + 2 * kAttentionRows;
// 1 / sqrt(128)
constant constexpr float kScale = 0.08838834765f;

struct RingAttentionParams {
    uint lanes;
    uint q_heads;
    uint kv_heads;
    uint window;
};

// Rounds the current F32 query, key and value rows to BF16 in the layouts the
// split kernel multiplies in place.
kernel void dflash2_ring_attention_pack(
    device const float *query           [[buffer(0)]],
    device const float *key             [[buffer(1)]],
    device const float *value           [[buffer(2)]],
    device bfloat *queries              [[buffer(3)]],
    device bfloat *current_keys         [[buffer(4)]],
    device bfloat *current_values       [[buffer(5)]],
    constant RingAttentionParams &p     [[buffer(6)]],
    uint gid                            [[thread_position_in_grid]]) {
    if (gid < p.lanes * p.q_heads * kRows * kHeadDim) {
        uint x = gid;
        const uint dim = x % kHeadDim; x /= kHeadDim;
        const uint row = x % kRows; x /= kRows;
        const uint head = x % p.q_heads;
        const uint lane = x / p.q_heads;
        queries[gid] = bfloat(query[((lane * kRows + row) * p.q_heads + head) * kHeadDim + dim]);
    }
    if (gid < p.lanes * p.kv_heads * kRows * kHeadDim) {
        uint x = gid;
        const uint dim = x % kHeadDim; x /= kHeadDim;
        const uint row = x % kRows; x /= kRows;
        const uint head = x % p.kv_heads;
        const uint lane = x / p.kv_heads;
        current_keys[gid] = bfloat(key[((lane * kRows + row) * p.kv_heads + head) * kHeadDim + dim]);
    }
    if (gid < p.lanes * p.kv_heads * kCurrentValues * kHeadDim) {
        uint x = gid;
        const uint dim = x % kHeadDim; x /= kHeadDim;
        const uint row = x % kCurrentValues; x /= kCurrentValues;
        const uint head = x % p.kv_heads;
        const uint lane = x / p.kv_heads;
        current_values[gid] = row < kRows
            ? bfloat(value[((lane * kRows + row) * p.kv_heads + head) * kHeadDim + dim])
            : bfloat(0.0f);
    }
}

kernel void dflash2_ring_attention_split(
    device bfloat *all_queries          [[buffer(0)]],
    device bfloat *ring_keys            [[buffer(1)]],
    device bfloat *ring_values          [[buffer(2)]],
    device bfloat *all_current_keys     [[buffer(3)]],
    device bfloat *all_current_values   [[buffer(4)]],
    device const uint *cache_lengths    [[buffer(5)]],
    device float *all_partials          [[buffer(6)]],
    constant RingAttentionParams &p     [[buffer(7)]],
    uint3 group                         [[threadgroup_position_in_grid]],
    uint thread_index                   [[thread_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint simd_group                     [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort M = kAttentionRows, N = kTile, D = kHeadDim, TileK = kStep;
    const uint head = group.x, batch = group.y, split = group.z;
    const uint window = p.window;
    const uint cache_length = cache_lengths[batch];

    device bfloat *queries =
        all_queries + (ulong(batch) * p.q_heads + head * kGroupHeads) * kRows * D;
    const ulong ring = (ulong(batch) * p.kv_heads + head) * window * D;
    device bfloat *keys = ring_keys + ring;
    device bfloat *values = ring_values + ring;
    device bfloat *current_keys =
        all_current_keys + (ulong(batch) * p.kv_heads + head) * kRows * D;
    device bfloat *current_values =
        all_current_values + (ulong(batch) * p.kv_heads + head) * kCurrentValues * D;
    device float *partial =
        all_partials + ((ulong(batch) * p.kv_heads + head) * kSplits + split) * kPartialFloats;

    threadgroup float score_storage[M * N];
    threadgroup float row_max[M];
    threadgroup float row_sum[M];
    threadgroup float previous_scale[M];

    const uint common_start = cache_length >= window - 1 ? cache_length - (window - 1) : 0;
    const uint old_count = cache_length - common_start;
    const uint physical_start = common_start % window;
    // Live tiles: [0, wrapped) when the window wraps, then [resume, tail_end);
    // a tile the wrap and the tail both touch is visited once.
    const uint physical_end = physical_start + old_count;
    const uint wrapped = physical_end > window ? (physical_end - window + N - 1) / N : 0;
    const uint tail_end = (min(physical_end, window) + N - 1) / N;
    const uint resume = max(physical_start / N, wrapped);
    const uint live_tiles = wrapped + tail_end - resume;
    // A split with no tile that is not the last one leaves only row maxima of
    // -inf, which the reduce skips.
    if (split >= live_tiles && split + 1 != kSplits) {
        if (thread_index < M) partial[M * D + thread_index] = -INFINITY;
        return;
    }

    auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    constexpr auto qk_descriptor = matmul2d_descriptor(M, N, TileK, false, true, false);
    matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
    auto q0 = qt.slice<TileK, M>(0, 0);
    auto pt = tensor(score_storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto p0 = pt.slice<TileK, M>(0, 0);
    auto first_v = tensor(values, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto first_v0 = first_v.slice<D, TileK>(0, 0);
    constexpr auto pv_descriptor = matmul2d_descriptor(M, D, TileK, false, false, false);
    matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), float>();
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;
    if (thread_index < M) {
        row_max[thread_index] = -INFINITY;
        row_sum[thread_index] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile = split; tile < live_tiles; tile += kSplits) {
        const uint slot = (tile < wrapped ? tile : resume + tile - wrapped) * N;
        auto kt = tensor(keys + ulong(slot) * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
        auto k0 = kt.slice<TileK, N>(0, 0);
        auto scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(k0), float>();
        for (ushort i = 0; i < scores.get_capacity(); ++i) scores[i] = 0.0f;
        for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
            auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
            auto ks = kt.slice<TileK, N>(chunk * TileK, 0);
            auto partial_scores = qk.template get_destination_cooperative_tensor<
                decltype(qs), decltype(ks), float>();
            qk.run(qs, ks, partial_scores);
            for (ushort i = 0; i < scores.get_capacity(); ++i) scores[i] += partial_scores[i];
        }
        scores.store(pt.slice<N, M>(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort first = 0; first < M; first += 8) {
            const ushort matrix_row = first + simd_group;
            const uint query_position = cache_length + matrix_row % kRows;
            const uint row_start =
                query_position >= window - 1 ? query_position - (window - 1) : 0;
            const uint hidden_prefix = row_start - common_start;
            float local_scores[4];
            float tile_max = -INFINITY;
            for (ushort i = 0; i < 4; ++i) {
                const uint column = lane + i * 32;
                const uint key = slot + column;
                const uint logical_key =
                    key >= physical_start ? key - physical_start : key + window - physical_start;
                const bool valid = logical_key < old_count && logical_key >= hidden_prefix;
                const float score = score_storage[matrix_row * N + column] * kScale;
                local_scores[i] = valid ? score : -INFINITY;
                tile_max = max(tile_max, local_scores[i]);
            }
            tile_max = simd_max(tile_max);
            const float next_max = max(row_max[matrix_row], tile_max);
            // A row with no visible key so far keeps its running state instead
            // of scaling it by exp(-inf - -inf).
            const bool empty = next_max == -INFINITY;
            const float scale = empty ? 1.0f : exp(row_max[matrix_row] - next_max);
            float tile_sum = 0.0f;
            for (ushort i = 0; i < 4; ++i) {
                local_scores[i] = empty ? 0.0f : exp(local_scores[i] - next_max);
                tile_sum += local_scores[i];
            }
            tile_sum = simd_sum(tile_sum);
            if (lane == 0) {
                previous_scale[matrix_row] = scale;
                row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
                row_max[matrix_row] = next_max;
            }
            for (ushort i = 0; i < 4; ++i)
                score_storage[matrix_row * N + lane + i * 32] = local_scores[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto vt = tensor(values + ulong(slot) * D, dextents<int, 2>{D, N}, array<int, 2>{1, D});
        auto v0 = vt.slice<D, TileK>(0, 0);
        auto tile_output = pv.template get_destination_cooperative_tensor<
            decltype(p0), decltype(v0), float>();
        for (ushort i = 0; i < tile_output.get_capacity(); ++i) tile_output[i] = 0.0f;
        for (ushort chunk = 0; chunk < N / TileK; ++chunk) {
            auto ps = pt.slice<TileK, M>(chunk * TileK, 0);
            auto vs = vt.slice<D, TileK>(0, chunk * TileK);
            auto chunk_output = pv.template get_destination_cooperative_tensor<
                decltype(ps), decltype(vs), float>();
            pv.run(ps, vs, chunk_output);
            for (ushort i = 0; i < tile_output.get_capacity(); ++i)
                tile_output[i] += chunk_output[i];
        }
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            const auto index = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[index[1]] + tile_output[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (split + 1 == kSplits) {
        // The eight current rows: every one is visible to every query row.
        constexpr ushort CurrentN = kRows, CurrentV = kCurrentValues;
        threadgroup float *current_raw_scores = score_storage + M * CurrentV;
        auto current_scores = tensor(current_raw_scores, dextents<int, 2>{CurrentN, M},
                                     array<int, 2>{1, CurrentN});
        constexpr auto current_qk_descriptor =
            matmul2d_descriptor(M, CurrentN, TileK, false, true, false);
        matmul2d<current_qk_descriptor, execution_simdgroups<8>> current_qk;
        auto current_kt = tensor(current_keys, dextents<int, 2>{D, CurrentN},
                                 array<int, 2>{1, D});
        auto current_k0 = current_kt.slice<TileK, CurrentN>(0, 0);
        auto current = current_qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(current_k0), float>();
        for (ushort i = 0; i < current.get_capacity(); ++i) current[i] = 0.0f;
        for (ushort chunk = 0; chunk < D / TileK; ++chunk) {
            auto qs = qt.slice<TileK, M>(chunk * TileK, 0);
            auto ks = current_kt.slice<TileK, CurrentN>(chunk * TileK, 0);
            auto partial_scores = current_qk.template get_destination_cooperative_tensor<
                decltype(qs), decltype(ks), float>();
            current_qk.run(qs, ks, partial_scores);
            for (ushort i = 0; i < current.get_capacity(); ++i) current[i] += partial_scores[i];
        }
        current.store(current_scores.slice<CurrentN, M>(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort first = 0; first < M; first += 8) {
            const ushort matrix_row = first + simd_group;
            const float score = lane < CurrentN
                ? current_raw_scores[matrix_row * CurrentN + lane] * kScale
                : -INFINITY;
            const float tile_max = simd_max(score);
            const float next_max = max(row_max[matrix_row], tile_max);
            const float scale = exp(row_max[matrix_row] - next_max);
            const float probability = lane < CurrentN ? exp(score - next_max) : 0.0f;
            const float tile_sum = simd_sum(probability);
            if (lane == 0) {
                previous_scale[matrix_row] = scale;
                row_sum[matrix_row] = row_sum[matrix_row] * scale + tile_sum;
                row_max[matrix_row] = next_max;
            }
            if (lane < CurrentV) score_storage[matrix_row * CurrentV + lane] = probability;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto current_probabilities = tensor(score_storage, dextents<int, 2>{CurrentV, M},
                                            array<int, 2>{1, CurrentV});
        auto current_p0 = current_probabilities.slice<CurrentV, M>(0, 0);
        auto current_vt = tensor(current_values, dextents<int, 2>{D, CurrentV},
                                 array<int, 2>{1, D});
        auto current_v0 = current_vt.slice<D, CurrentV>(0, 0);
        constexpr auto current_pv_descriptor =
            matmul2d_descriptor(M, D, CurrentV, false, false, false);
        matmul2d<current_pv_descriptor, execution_simdgroups<8>> current_pv;
        auto current_output = current_pv.template get_destination_cooperative_tensor<
            decltype(current_p0), decltype(current_v0), float>();
        for (ushort i = 0; i < current_output.get_capacity(); ++i) current_output[i] = 0.0f;
        current_pv.run(current_p0, current_v0, current_output);
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            const auto index = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[index[1]] + current_output[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto accumulator = tensor(partial, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    running.store(accumulator.slice<D, M>(0, 0));
    if (thread_index < M) {
        partial[M * D + thread_index] = row_max[thread_index];
        partial[M * D + M + thread_index] = row_sum[thread_index];
    }
}

// Fixed-order combine of one (lane, KV head) pair's split partials into the
// normalized BF16 rows, in the row-major layout the output projection reads.
kernel void dflash2_ring_attention_reduce(
    device const float *all_partials    [[buffer(0)]],
    device bfloat *output               [[buffer(1)]],
    constant RingAttentionParams &p     [[buffer(2)]],
    uint2 group                         [[threadgroup_position_in_grid]],
    uint thread_index                   [[thread_index_in_threadgroup]]) {
    constexpr uint M = kAttentionRows, D = kHeadDim, Chunk = D / 8;
    const uint head = group.x, batch = group.y;
    device const float *partials =
        all_partials + (ulong(batch) * p.kv_heads + head) * kSplits * kPartialFloats;
    const uint row = thread_index / 8;
    const uint dim = (thread_index % 8) * Chunk;
    float maximum = -INFINITY;
    for (uint split = 0; split < kSplits; ++split)
        maximum = max(maximum, partials[split * kPartialFloats + M * D + row]);
    float total = 0.0f;
    float accumulated[Chunk];
    for (uint i = 0; i < Chunk; ++i) accumulated[i] = 0.0f;
    for (uint split = 0; split < kSplits; ++split) {
        device const float *partial = partials + split * kPartialFloats;
        const float split_max = partial[M * D + row];
        // A split with no visible key contributes nothing rather than
        // exp(-inf - max) times an unwritten accumulator.
        if (split_max == -INFINITY) continue;
        const float scale = exp(split_max - maximum);
        total += scale * partial[M * D + M + row];
        for (uint i = 0; i < Chunk; ++i) accumulated[i] += scale * partial[row * D + dim + i];
    }
    const uint query_head = head * kGroupHeads + row / kRows;
    device bfloat *destination = output +
        ((ulong(batch) * kRows + row % kRows) * p.q_heads + query_head) * D + dim;
    for (uint i = 0; i < Chunk; ++i) destination[i] = bfloat(accumulated[i] / total);
}
