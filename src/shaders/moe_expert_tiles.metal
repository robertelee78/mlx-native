// MoE expert live tiles over prepared Q4_K/Q6_K planes: one gate/up pass and
// a down pass over the compact route schedule (map0c), with the BF16
// SwiGLU intermediate kept in the schedule's grouped rows.
//
// The planes are the ones `grouped_m8_prepare_q4k_q6k` writes
// (quantized_matmul_id_grouped_mv.metal). Each thread decodes its output
// column's 32-weight groups to half, so every product with a BF16 activation
// is exact in FP32. An expert of up to `column_rows` routes runs in registers
// (no stage, no barrier); any other expert stages its weights and runs the
// 8-, 16- or 32-row `matmul2d` that holds its routes, decoding the weights
// once for all of them.
//
// Row independence. A route's result depends only on its own input row and
// its expert's weights: the register path adds the 32 exact products of a
// group in the order `matmul2d` multiply-accumulate adds them on the devices
// where the host enables it (`moe_tiles_column_tile_exact` proves the two
// bitwise equal on the running device; otherwise `column_rows` is 0 and every
// expert takes the matmul), and a matmul row is the same in every tile size
// and beside any other rows. `reassociate(off)` keeps the source order of the
// sums; products that underflow FP32 are outside the equality.

#pragma clang fp reassociate(off)

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint MT_PLANE_ROWS = 64;
constant constexpr uint MT_COLS = 32;
constant constexpr uint MT_STEP = 32;
constant constexpr uint MT_TILE_ROWS = 8;
constant constexpr uint MT_COLUMN_ROWS = 2;
constant constexpr uint MT_STAGE = MT_COLS * MT_STEP;

struct MoeTilesGatherParams {
    uint k;
    uint top_k;
    uint gate_up_columns;
    uint down_columns;
};

struct MoeTilesParams {
    uint k;
    uint n;
    uint column_rows;
    uint plane1_offset;
    uint meta_offset;
    uint _pad;
    uint64_t prepared_expert_stride;
};

// Copies each scheduled route's input row into the schedule's grouped rows
// (entry `e` of the route list at row `e`) as BF16, and writes the indirect
// grids of the two expert passes: (column tiles, live tiles, 1). A poisoned
// or empty schedule launches one tile row, which returns at once.
// Threadgroup (part, tile) copies the part's 256 columns of the tile's rows,
// eight rows at once, each thread eight columns of one row; grid
// (k / 256, tiles), 256 threads.
kernel void moe_tiles_gather_f32(
    device const float *input [[buffer(0)]],
    device bfloat *gathered [[buffer(1)]],
    device const uint *schedule [[buffer(2)]],
    device const int *routes [[buffer(3)]],
    device uint *grids [[buffer(4)]],
    constant MoeTilesGatherParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint state = schedule[0];
    const uint tiles = (state & 0x80000000u) != 0u ? 0u : state;
    if (group.x == 0 && group.y == 0 && thread_index == 0) {
        const uint launched = max(tiles, 1u);
        grids[0] = p.gate_up_columns;
        grids[1] = launched;
        grids[2] = 1;
        grids[3] = p.down_columns;
        grids[4] = launched;
        grids[5] = 1;
    }
    if (group.y >= tiles)
        return;
    device const uint *descriptor = schedule + 1 + 3 * group.y;
    const uint first = descriptor[1];
    const uint count = descriptor[2];
    const uint column = group.x * 256 + (thread_index % 32) * 8;
    for (uint row = thread_index / 32; row < count; row += 8) {
        const uint input_row = uint(routes[first + row]) / p.top_k;
        device const float *source = input + ulong(input_row) * p.k + column;
        device bfloat *target = gathered + (ulong(first) + row) * p.k + column;
#pragma unroll
        for (ushort i = 0; i < 8; ++i)
            target[i] = bfloat(source[i]);
    }
}

inline uint4 mt_nibble_pairs(uint word) {
    return (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu;
}

inline uint mt_spread2(uint bits) {
    return (bits & 0x3333u) | ((bits & 0xCCCCu) << 14);
}

inline uint mt_scale_pair(packed_uint4 bytes, ushort j) {
    const uint word = j < 2 ? bytes.x : j < 4 ? bytes.y : j < 6 ? bytes.z : bytes.w;
    return word >> (16u * (j & 1));
}

inline float2 mt_q4_coef(uint4 h, ushort j) {
    uint scale, minimum;
    if (j < 4) {
        const uint shift = 8u * j;
        scale = (h.y >> shift) & 63u;
        minimum = (h.z >> shift) & 63u;
    } else {
        const uint shift = 8u * (j - 4);
        const uint word = h.w >> shift;
        scale = (word & 15u) | ((((h.y >> shift) >> 6) & 3u) << 4);
        minimum = ((word >> 4) & 15u) | ((((h.z >> shift) >> 6) & 3u) << 4);
    }
    const half d = as_type<half>(ushort(h.x));
    const half dmin = as_type<half>(ushort(h.x >> 16));
    return float2(float(d) * float(scale), -float(dmin) * float(minimum));
}

// Two codes of one pair word to half, each value rounded once. 0x6400 is
// half 1024, whose ulp is 1: or-ing a code into its mantissa is 1024 + code.
inline half2 mt_linear_pair(uint pair, float scale, float minimum, ushort zero) {
    const float2 code =
        float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + zero)));
    return zero ? half2(code * scale) : half2(fma(code, float2(scale), float2(minimum)));
}

// One output column's walk over its prepared planes: the payload of the
// group it decodes next and that group's metadata unit.
template <bool Q6>
struct MtWalk {
    device const uchar *plane0;
    device const uchar *plane1;
    device const uchar *meta;
    uint4 low;
    uint2 high;
    uint4 q4_meta;
    packed_uint4 q6_scales;
    uint q6_d;
};

template <bool Q6>
inline void mt_load_payload(thread MtWalk<Q6> &walk, uint g) {
    walk.low = *reinterpret_cast<device const uint4 *>(
        walk.plane0 + ulong(g) * (MT_PLANE_ROWS * 16));
    if constexpr (Q6)
        walk.high = *reinterpret_cast<device const uint2 *>(
            walk.plane1 + ulong(g) * (MT_PLANE_ROWS * 8));
}

template <bool Q6>
inline void mt_load_meta(thread MtWalk<Q6> &walk, uint unit) {
    if constexpr (Q6) {
        device const uchar *at = walk.meta + ulong(unit) * (MT_PLANE_ROWS * 20);
        walk.q6_scales = *reinterpret_cast<device const packed_uint4 *>(at);
        walk.q6_d = *reinterpret_cast<device const uint *>(at + 16);
    } else {
        walk.q4_meta = *reinterpret_cast<device const uint4 *>(
            walk.meta + ulong(unit) * (MT_PLANE_ROWS * 16));
    }
}

template <bool Q6>
inline void mt_begin(
    thread MtWalk<Q6> &walk,
    device const uchar *weights,
    constant MoeTilesParams &p,
    uint column) {
    const uint groups = p.k / MT_STEP;
    const uint units = groups / 8;
    const ulong tile = column / MT_PLANE_ROWS;
    const ulong in_tile = column % MT_PLANE_ROWS;
    walk.plane0 = weights + (tile * groups * MT_PLANE_ROWS + in_tile) * 16;
    walk.plane1 = weights + p.plane1_offset + (tile * groups * MT_PLANE_ROWS + in_tile) * 8;
    walk.meta = weights + p.meta_offset +
        (tile * units * MT_PLANE_ROWS + in_tile) * (Q6 ? 20 : 16);
    mt_load_payload<Q6>(walk, 0);
    mt_load_meta<Q6>(walk, 0);
}

// After group `next - 1` is decoded: its successor's payload, and the
// metadata unit when the successor starts one.
template <bool Q6>
inline void mt_advance(thread MtWalk<Q6> &walk, uint next) {
    mt_load_payload<Q6>(walk, next);
    if (next % 8 == 0)
        mt_load_meta<Q6>(walk, next / 8);
}

// The walk's group `j` of its metadata unit: v[i] holds weights 4i..4i+3.
template <bool Q6>
inline void mt_decode(thread const MtWalk<Q6> &walk, ushort j, thread half4 *v) {
    if constexpr (Q6) {
        const float d = float(as_type<half>(ushort(walk.q6_d)));
        const uint pair = mt_scale_pair(walk.q6_scales, j);
        const float2 scale = float2(
            d * float(as_type<char>(uchar(pair))),
            d * float(as_type<char>(uchar(pair >> 8))));
#pragma unroll
        for (ushort chunk = 0; chunk < 4; ++chunk) {
            const uint high_bits =
                mt_spread2(walk.high[chunk >> 1] >> (16 * (chunk & 1)));
            const uint4 pairs = mt_nibble_pairs(walk.low[chunk]) |
                (((uint4(high_bits) >> uint4(0, 4, 8, 12)) & 0x00030003u) << 4);
            v[chunk] = half4(
                mt_linear_pair(pairs.x, scale.x, 0, 32),
                mt_linear_pair(pairs.y, scale.x, 0, 32));
            v[4 + chunk] = half4(
                mt_linear_pair(pairs.z, scale.y, 0, 32),
                mt_linear_pair(pairs.w, scale.y, 0, 32));
        }
    } else {
        const float2 coefficient = mt_q4_coef(walk.q4_meta, j);
#pragma unroll
        for (ushort chunk = 0; chunk < 4; ++chunk) {
            const uint4 pairs = mt_nibble_pairs(walk.low[chunk]);
            v[chunk] = half4(
                mt_linear_pair(pairs.x, coefficient.x, coefficient.y, 0),
                mt_linear_pair(pairs.y, coefficient.x, coefficient.y, 0));
            v[4 + chunk] = half4(
                mt_linear_pair(pairs.z, coefficient.x, coefficient.y, 0),
                mt_linear_pair(pairs.w, coefficient.x, coefficient.y, 0));
        }
    }
}

// The register path: `count` (at most MT_COLUMN_ROWS) rows of `x` against one
// weight column. Per group, with p[e] the exact product of element e:
//   q[k] = p[2k] + p[2k+1]                         k = 0..15
//   a[m] = q[m] + q[m+4], a[4+m] = q[8+m] + q[12+m]  m = 0..3
//   sum  = (((((((sum + a[0]) + a[1]) + a[2]) + a[3]) + a[4]) + a[5]) + a[6]) + a[7]
// which is the order of matmul2d's 32-input multiply-accumulate step.
template <bool Q6>
inline void mt_column(
    thread MtWalk<Q6> &walk,
    constant MoeTilesParams &p,
    device const bfloat *x,
    uint count,
    thread float *sums) {
#pragma unroll
    for (ushort row = 0; row < MT_COLUMN_ROWS; ++row)
        sums[row] = 0.0f;
    const uint groups = p.k / MT_STEP;
    for (uint g = 0; g < groups; ++g) {
        half4 v[8];
        mt_decode<Q6>(walk, ushort(g % 8), v);
        if (g + 1 < groups)
            mt_advance<Q6>(walk, g + 1);
#pragma unroll
        for (ushort row = 0; row < MT_COLUMN_ROWS; ++row) {
            if (row >= count)
                break;
            device const bfloat4 *xr = reinterpret_cast<device const bfloat4 *>(
                x + ulong(row) * p.k + g * MT_STEP);
            float4 c[8];
#pragma unroll
            for (ushort i = 0; i < 8; ++i)
                c[i] = float4(v[i]) * float4(xr[i]);
            float sum = sums[row];
#pragma unroll
            for (ushort half_index = 0; half_index < 2; ++half_index) {
                const ushort b = 4 * half_index;
                const float2 a01 = (c[b].xz + c[b].yw) + (c[b + 2].xz + c[b + 2].yw);
                const float2 a23 = (c[b + 1].xz + c[b + 1].yw) + (c[b + 3].xz + c[b + 3].yw);
                sum = sum + a01.x;
                sum = sum + a01.y;
                sum = sum + a23.x;
                sum = sum + a23.y;
            }
            sums[row] = sum;
        }
    }
}

// each(row, column, value) for this thread's elements of a tile's sums that
// belong to its first `count` rows.
template <class Sum, class Each>
inline void mt_elements(thread Sum &sum, uint count, Each each) {
    for (ushort i = 0; i < sum.get_capacity(); ++i) {
        if (!sum.is_valid_element(i))
            continue;
        const auto index = sum.get_multidimensional_index(i);
        if (uint(index[1]) < count)
            each(uint(index[1]), uint(index[0]), float(sum[i]));
    }
}

// One simdgroup's 32 columns (from `column - lane` on) against the `count`
// (at most MT_TILE_ROWS) routes at `x`.
//
// Register path (`column_tile`): each lane sums its own column. Staged path:
// lane l stages column l of each group into one of two stages and the
// simdgroup runs the group's MT_TILE_ROWS-row matmul (rows past `count` are
// read and dropped); the next group's payload is loaded while that matmul
// runs, and alternating stages leave one barrier per group.
//
// `emit(row, column in the 32, sum)` receives every result. In a gate/up
// pair (`Pair`) the gate simdgroup calls `park` instead, both simdgroups
// meet at a threadgroup barrier, and the up simdgroup emits after it.
template <bool Q6, bool Pair, class Park, class Emit>
inline void mt_project(
    device const uchar *weights,
    constant MoeTilesParams &p,
    device bfloat *x,
    uint count,
    bool column_tile,
    uint column,
    ushort lane,
    threadgroup half *stage,
    bool up,
    Park park,
    Emit emit) {
    MtWalk<Q6> walk;
    mt_begin<Q6>(walk, weights, p, column);
    if (column_tile) {
        float sums[MT_COLUMN_ROWS];
        mt_column<Q6>(walk, p, x, count, sums);
        if constexpr (Pair) {
            if (!up) {
                for (uint row = 0; row < count; ++row)
                    park(row, uint(lane), sums[row]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!up)
                return;
        }
        for (uint row = 0; row < count; ++row)
            emit(row, uint(lane), sums[row]);
        return;
    }
    matmul2d<matmul2d_descriptor(
        MT_TILE_ROWS, MT_COLS, MT_STEP, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;
    auto all_left = tensor(
        x, dextents<int32_t, 2>{int(p.k), MT_TILE_ROWS}, array<int32_t, 2>{1, int(p.k)});
    auto first_left = all_left.template slice<MT_STEP, MT_TILE_ROWS>(0, 0);
    auto right0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        stage, dextents<int32_t, 2>{MT_STEP, MT_COLS}, array<int32_t, 2>{1, MT_STEP});
    auto right1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        stage + MT_STAGE, dextents<int32_t, 2>{MT_STEP, MT_COLS},
        array<int32_t, 2>{1, MT_STEP});
    auto sum = mm.template get_destination_cooperative_tensor<
        decltype(first_left), decltype(right0), float>();
    for (ushort i = 0; i < sum.get_capacity(); ++i)
        sum[i] = 0.0f;
    const uint groups = p.k / MT_STEP;
    for (uint g = 0; g < groups; ++g) {
        half4 v[8];
        mt_decode<Q6>(walk, ushort(g % 8), v);
        threadgroup half4 *staged = reinterpret_cast<threadgroup half4 *>(
            stage + (g & 1) * MT_STAGE + uint(lane) * MT_STEP);
#pragma unroll
        for (ushort i = 0; i < 8; ++i)
            staged[i] = v[i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (g + 1 < groups)
            mt_advance<Q6>(walk, g + 1);
        auto left = all_left.template slice<MT_STEP, MT_TILE_ROWS>(g * MT_STEP, 0);
        if (g & 1)
            mm.run(left, right1, sum);
        else
            mm.run(left, right0, sum);
    }
    if constexpr (Pair) {
        if (!up)
            mt_elements(sum, count, park);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!up)
            return;
    }
    mt_elements(sum, count, emit);
}

// One definition of the SwiGLU store, so the register and staged paths
// round a route's intermediate identically.
__attribute__((noinline)) bfloat mt_swiglu(float gate, float up) {
    return bfloat((gate / (1.0f + exp(-gate))) * up);
}

// Gate and up in one pass: threadgroup (x, y) runs a gate simdgroup and an up
// simdgroup on the same 32 columns of live tile y, grid (n / 32, tiles). The
// gate simdgroup parks its FP32 sums and the up simdgroup writes
// BF16(SiLU(gate) * up) to the tile's grouped rows.
template <bool Q6>
kernel void moe_tiles_gate_up_impl(
    device const uchar *gate_weights [[buffer(0)]],
    device const uchar *up_weights [[buffer(1)]],
    device bfloat *input [[buffer(2)]],
    device bfloat *intermediate [[buffer(3)]],
    device const uint *schedule [[buffer(4)]],
    constant MoeTilesParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup half stage[2 * 2 * MT_STAGE];
    threadgroup float gates[MT_TILE_ROWS * MT_COLS];
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.y >= state)
        return;
    device const uint *descriptor = schedule + 1 + 3 * group.y;
    const uint expert = descriptor[0];
    const uint first = descriptor[1];
    const uint count = descriptor[2];
    const bool up = simdgroup != 0;
    const uint origin = group.x * MT_COLS;
    device const uchar *weights =
        (up ? up_weights : gate_weights) + ulong(expert) * p.prepared_expert_stride;
    const bool column_tile = count <= min(p.column_rows, MT_COLUMN_ROWS);
    for (uint base = 0; base < count; base += MT_TILE_ROWS) {
        const ulong row0 = ulong(first) + base;
        mt_project<Q6, true>(
            weights, p, input + row0 * p.k, min(MT_TILE_ROWS, count - base), column_tile,
            origin + lane, lane, stage + uint(simdgroup) * (2 * MT_STAGE), up,
            [&](uint row, uint column, float value) {
                gates[row * MT_COLS + column] = value;
            },
            [&](uint row, uint column, float value) {
                intermediate[(row0 + row) * p.n + origin + column] =
                    mt_swiglu(gates[row * MT_COLS + column], value);
            });
        if (base + MT_TILE_ROWS < count)
            threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Down: threadgroup (x, y) runs two simdgroups, each on its own 32 columns of
// live tile y, grid (n / 64, tiles), over the tile's grouped BF16 rows, and
// writes FP32 to each route's own row of `output` (route `token * top_k +
// slot` at row `route`).
template <bool Q6>
kernel void moe_tiles_down_impl(
    device const uchar *down_weights [[buffer(0)]],
    device bfloat *intermediate [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint *schedule [[buffer(3)]],
    device const int *routes [[buffer(4)]],
    constant MoeTilesParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup half stage[2 * 2 * MT_STAGE];
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.y >= state)
        return;
    device const uint *descriptor = schedule + 1 + 3 * group.y;
    const uint expert = descriptor[0];
    const uint first = descriptor[1];
    const uint count = descriptor[2];
    const uint origin = group.x * (2 * MT_COLS) + uint(simdgroup) * MT_COLS;
    const bool column_tile = count <= min(p.column_rows, MT_COLUMN_ROWS);
    for (uint base = 0; base < count; base += MT_TILE_ROWS) {
        const ulong row0 = ulong(first) + base;
        mt_project<Q6, false>(
            down_weights + ulong(expert) * p.prepared_expert_stride, p,
            intermediate + row0 * p.k, min(MT_TILE_ROWS, count - base), column_tile,
            origin + lane, lane, stage + uint(simdgroup) * (2 * MT_STAGE), false,
            [](uint, uint, float) {},
            [&](uint row, uint column, float value) {
                output[ulong(uint(routes[row0 + row])) * p.n + origin + column] = value;
            });
    }
}

template [[host_name("moe_tiles_gate_up_q4_K")]]
kernel void moe_tiles_gate_up_impl<false>(
    device const uchar *, device const uchar *, device bfloat *, device bfloat *,
    device const uint *, constant MoeTilesParams &, uint2, ushort, ushort);

template [[host_name("moe_tiles_gate_up_q6_K")]]
kernel void moe_tiles_gate_up_impl<true>(
    device const uchar *, device const uchar *, device bfloat *, device bfloat *,
    device const uint *, constant MoeTilesParams &, uint2, ushort, ushort);

template [[host_name("moe_tiles_down_q4_K")]]
kernel void moe_tiles_down_impl<false>(
    device const uchar *, device bfloat *, device float *, device const uint *,
    device const int *, constant MoeTilesParams &, uint2, ushort, ushort);

template [[host_name("moe_tiles_down_q6_K")]]
kernel void moe_tiles_down_impl<true>(
    device const uchar *, device bfloat *, device float *, device const uint *,
    device const int *, constant MoeTilesParams &, uint2, ushort, ushort);
