#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Prepared AF4G64 projection: 64 output rows per storage tile. Code plane:
// [N / 64][K / 32][64][4 words]. Metadata plane:
// [N / 64][K / 64][64] packed as BF16 scale | BF16 bias.
constant constexpr uint OUTPUT_TILE = 64;
constant constexpr uint OUTPUTS_PER_SIMDGROUP = 16;
constant constexpr uint ROWS_PER_TILE = 8;
constant constexpr uint SPAN = 64;
constant constexpr uint SPAN_TABLE_VALUES = SPAN * ROWS_PER_TILE;

struct MatrixLane {
    ushort row;
    ushort column;
};

inline MatrixLane matrix_lane(uint lane) {
    const uint quad = lane >> 2;
    MatrixLane result;
    result.row = ushort((quad & 4) | ((lane >> 1) & 3));
    result.column = ushort(((quad & 2) << 1) | ((lane & 1) << 1));
    return result;
}

template <typename T>
inline thread vec<T, 2> &matrix_elements(thread simdgroup_matrix<T, 8, 8> &matrix) {
    return reinterpret_cast<thread vec<T, 2> &>(matrix.thread_elements());
}

inline void matrix_multiply_accumulate(
    thread float2 &accumulator,
    bfloat2 weights,
    bfloat2 activations) {
    simdgroup_matrix<bfloat, 8, 8> a, b;
    simdgroup_matrix<float, 8, 8> c, result;
    matrix_elements(a) = weights;
    matrix_elements(b) = activations;
    matrix_elements(c) = accumulator;
    simdgroup_multiply_accumulate(result, a, b, c);
    accumulator = matrix_elements(result);
}

inline uint table_offset(uint fragment_index, uint inner_row, uint request_row) {
    return (((fragment_index >> 2) * 8 + inner_row) * 4 + (request_row >> 1)) * 8 +
        (fragment_index & 3) * 2 + (request_row & 1);
}

inline uint2 logical_inner(uint inner) {
    const uint group_half = inner >> 5;
    const uint group = (inner >> 4) & 1;
    const uint chunk = (inner >> 2) & 3;
    const uint pair = (inner >> 1) & 1;
    const uint element = inner & 1;
    return uint2(4 * group_half + 2 * group + pair, 2 * chunk + element);
}

kernel void qmm_affine_t_packed_simd4_b4_gs64_bf16_prepare(
    device const float *x          [[buffer(0)]],
    device bfloat *table           [[buffer(1)]],
    device float *sums             [[buffer(2)]],
    constant uint4 &shape          [[buffer(3)]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint request_row               [[simdgroup_index_in_threadgroup]],
    uint2 tile                     [[threadgroup_position_in_grid]]) {
    const uint M = shape.x;
    const uint K = shape.z;
    const uint row = tile.y * ROWS_PER_TILE + request_row;
    if (row >= M) return;
    const uint inner = tile.x * SPAN + 2 * lane;
    const bfloat first = bfloat(x[ulong(row) * K + inner]);
    const bfloat second = bfloat(x[ulong(row) * K + inner + 1]);
    const uint2 logical = logical_inner(2 * lane);
    device bfloat *row_table =
        table + ulong(tile.y) * K * ROWS_PER_TILE + tile.x * SPAN_TABLE_VALUES;
    row_table[table_offset(logical.x, logical.y, request_row)] = first;
    row_table[table_offset(logical.x, logical.y + 1, request_row)] = second;

    float sum = float(first) + float(second);
    sum += simd_shuffle_xor(sum, 1u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 4u);
    sum += simd_shuffle_xor(sum, 8u);
    sum += simd_shuffle_xor(sum, 16u);
    if (lane == 0) {
        const uint groups64 = K / SPAN;
        sums[(tile.y * groups64 + tile.x) * ROWS_PER_TILE + request_row] = sum;
    }
}

kernel void qmm_affine_t_packed_simd4_b4_gs64_bf16(
    device const bfloat *table     [[buffer(0)]],
    device const float *sums       [[buffer(1)]],
    device const uint *codes       [[buffer(2)]],
    device const uint *metadata    [[buffer(3)]],
    device float *y                [[buffer(4)]],
    constant uint4 &shape          [[buffer(5)]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint simdgroup                 [[simdgroup_index_in_threadgroup]],
    uint2 tile                     [[threadgroup_position_in_grid]]) {
    const uint M = shape.x;
    const uint N = shape.y;
    const uint K = shape.z;
    const uint groups32 = K / 32;
    const uint groups64 = K / SPAN;
    const uint output_tile = tile.x;
    const uint output_origin =
        output_tile * OUTPUT_TILE + simdgroup * OUTPUTS_PER_SIMDGROUP;
    const uint row_origin = tile.y * ROWS_PER_TILE;
    const MatrixLane matrix = matrix_lane(lane);
    const uint chunk = matrix.column / 2;
    float2 accumulators[2] = {float2(0.0f), float2(0.0f)};
    device const bfloat *row_table = table + ulong(tile.y) * K * ROWS_PER_TILE;

    for (uint group64 = 0; group64 < groups64; ++group64) {
        float2 group_accumulators[2] = {float2(0.0f), float2(0.0f)};
        device const vec<bfloat, 8> *span_table =
            reinterpret_cast<device const vec<bfloat, 8> *>(
                row_table + group64 * SPAN_TABLE_VALUES);
#pragma unroll
        for (uint group_half = 0; group_half < 2; ++group_half) {
            const uint group32 = group64 * 2 + group_half;
            uint4 pairs[2];
#pragma unroll
            for (uint output_fragment = 0; output_fragment < 2; ++output_fragment) {
                const uint output_row =
                    simdgroup * OUTPUTS_PER_SIMDGROUP +
                    output_fragment * 8 + matrix.row;
                const ulong code_row =
                    (ulong(output_tile) * groups32 + group32) * OUTPUT_TILE +
                    output_row;
                const uint word = codes[code_row * 4 + chunk];
                pairs[output_fragment] =
                    (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu;
            }
            const vec<bfloat, 8> activations =
                span_table[(8 * group_half + matrix.row) * 4 + chunk];
#pragma unroll
            for (uint pair = 0; pair < 4; ++pair) {
                const bfloat2 activation =
                    reinterpret_cast<thread const bfloat2 *>(&activations)[pair];
#pragma unroll
                for (uint output_fragment = 0; output_fragment < 2; ++output_fragment) {
                    const bfloat2 weight = as_type<bfloat2>(
                        pairs[output_fragment][pair] + 0x43004300u);
                    matrix_multiply_accumulate(
                        group_accumulators[output_fragment],
                        weight,
                        activation);
                }
            }
        }
        const float2 input_sum = *reinterpret_cast<device const float2 *>(
            sums + (tile.y * groups64 + group64) * ROWS_PER_TILE + matrix.column);
#pragma unroll
        for (uint output_fragment = 0; output_fragment < 2; ++output_fragment) {
            const uint output_row =
                simdgroup * OUTPUTS_PER_SIMDGROUP +
                output_fragment * 8 + matrix.row;
            const ulong metadata_row =
                (ulong(output_tile) * groups64 + group64) * OUTPUT_TILE +
                output_row;
            const uint coefficients = metadata[metadata_row];
            const float scale =
                float(as_type<bfloat>(ushort(coefficients & 0xFFFFu)));
            const float bias =
                float(as_type<bfloat>(ushort(coefficients >> 16)));
            accumulators[output_fragment] = fma(
                group_accumulators[output_fragment],
                float2(scale),
                accumulators[output_fragment]);
            accumulators[output_fragment] = fma(
                input_sum,
                float2(fma(-128.0f, scale, bias)),
                accumulators[output_fragment]);
        }
    }

#pragma unroll
    for (uint output_fragment = 0; output_fragment < 2; ++output_fragment) {
        const uint output =
            output_origin + output_fragment * 8 + matrix.row;
        const uint row0 = row_origin + matrix.column;
        const uint row1 = row0 + 1;
        if (output < N && row0 < M)
            y[ulong(row0) * N + output] = accumulators[output_fragment].x;
        if (output < N && row1 < M)
            y[ulong(row1) * N + output] = accumulators[output_fragment].y;
    }
}
