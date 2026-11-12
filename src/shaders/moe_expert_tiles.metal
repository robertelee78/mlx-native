// MoE expert live tiles over prepared Q4_K/Q6_K planes: one gate/up pass and
// a down pass over the compact route schedule (map0c), with the BF16
// SwiGLU intermediate kept in the schedule's grouped rows.
//
// Portions of this file are derived from Splash
// (https://github.com/incoai/splash, commit 031a25a6), Apache-2.0 licensed.
// Source: runtime/ops/MoE.cpp, runtime/metal/kernels/shared/moe_gguf.metal,
// runtime/metal/kernels/common/gguf_staged_tile.h, gguf_column_tile.h and
// moe_expert_slab.h (one gate/up pass with a grouped intermediate over an
// indirect live grid, the alternating half weight stage, the register path
// for experts of one or two routes, the shared expert as tiles).
// Modifications: the planes and compact route schedule of this crate's
// grouped M8 route; eight-row tiles only; the register path enabled by a
// device probe of the matmul's summation order; masked rows.
//
// Copyright the Splash authors. See LICENSE-APACHE-splash at the repository
// root.
//
// The planes are the ones `grouped_m8_prepare_q4k_q6k` writes
// (quantized_matmul_id_grouped_mv.metal). Each thread decodes its output
// column's 32-weight groups to half, so every product with a BF16 activation
// is exact in FP32. An expert of up to `column_rows` routes runs in registers
// (no stage, no barrier); any other expert stages its weights and runs
// eight-row `matmul2d` tiles over its routes, one after another.
//
// Row independence. A route's result depends only on its own input row and
// its expert's weights: the register path adds the 32 exact products of a
// group in the order `matmul2d` multiply-accumulate adds them on the devices
// where the host enables it (`moe_tiles_column_tile_exact` compares the two
// bitwise on the running device; otherwise `column_rows` is 0 and every
// expert takes the matmul), and a matmul row is the same beside any other
// rows and at any place in its tile. `reassociate(off)` keeps the source
// order of the sums. Products that underflow FP32 are outside the equality:
// the two paths may round a subnormal product differently, so a result of
// that magnitude (observed at most 4e-31) may depend on how many routes
// share its expert.

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

// ====================================================================
// Fused route: one schedule for the routed experts and the shared expert
// ====================================================================
//
// `moe_tiles_fused_schedule` does the work of the verify-row mask, both
// map0c schedules and the grid writes of the route above in one dispatch
// (Splash: runtime/metal/kernels/shared/moe.metal, moe_group_routes_verified):
// one threadgroup of MT_FUSED_THREADS threads, thread r owning route r of the
// table and, in the scan, expert r.
//
//   schedule word 0       tile count, or MT_FUSED_POISON when an unmasked id
//                         is out of range or the tiles exceed `max_tiles`
//   schedule[1 + 3t ..]   tile t: (expert | flags, first grouped row, rows
//                         <= 8). Routed tiles in ascending expert order, an
//                         expert of more than eight routes as several tiles,
//                         then the shared expert's tiles, one per block of
//                         eight rows of the step.
//   routes[g]             the route of grouped row g: `token * top_k + slot`,
//                         ascending within its expert, or `n_tokens * top_k +
//                         token` for a row of the shared expert.
//   grids                 the indirect grids of gather, gate/up and down.
//
// With `masked`, a route of a row outside its lane's verify rows is written
// back as the id `n_experts` with weight 0 (what `dflash2_mask_moe_routes`
// writes) and joins no tile.
//
// MT_FUSED_COLUMN marks a tile the passes run in registers: an expert, or a
// block of the shared expert, of at most `column_rows` routes in total. That
// is the rule of the route above, so both routes compute every route with
// the same arithmetic; an expert's tiles are all of one kind.

constant constexpr uint MT_FUSED_THREADS = 256;
constant constexpr uint MT_FUSED_GROUPS = MT_FUSED_THREADS / 32;
constant constexpr uint MT_FUSED_POISON = 0x80000000u;
constant constexpr uint MT_FUSED_SHARED = 0x80000000u;
constant constexpr uint MT_FUSED_COLUMN = 0x40000000u;
constant constexpr uint MT_FUSED_EXPERT = 0x3FFFFFFFu;

struct MoeTilesFusedScheduleParams {
    uint n_tokens;
    uint top_k;
    uint n_experts;
    uint max_tiles;
    uint column_rows;
    uint shared_tiles;
    uint masked;
    uint lanes;
    uint rows_per_lane;
    uint gather_columns;
    uint gate_up_columns;
    uint down_columns;
    uint physical_lane[4];
};

kernel void moe_tiles_fused_schedule(
    device uint *ids [[buffer(0)]],
    device float *routing_weights [[buffer(1)]],
    device const uint *verify_rows [[buffer(2)]],
    device uint *schedule [[buffer(3)]],
    device uint *routes [[buffer(4)]],
    device uint *grids [[buffer(5)]],
    constant MoeTilesFusedScheduleParams &p [[buffer(6)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    // masks[g][e]: the lanes of simdgroup g whose route goes to expert e.
    threadgroup atomic_uint masks[MT_FUSED_GROUPS * MT_FUSED_THREADS];
    threadgroup uint starts[MT_FUSED_THREADS];
    threadgroup uint totals[2 * MT_FUSED_GROUPS];
    threadgroup atomic_uint invalid;

    const uint route = uint(simdgroup) * 32 + lane;
    const uint route_count = p.n_tokens * p.top_k;
    for (uint g = 0; g < MT_FUSED_GROUPS; ++g)
        atomic_store_explicit(&masks[g * MT_FUSED_THREADS + route], 0u, memory_order_relaxed);
    if (route == 0)
        atomic_store_explicit(&invalid, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- The route's expert, after the verify-row mask ----
    uint id = p.n_experts;
    if (route < route_count) {
        const uint token = route / p.top_k;
        bool masked = false;
        if (p.masked != 0u) {
            const uint step_lane = token / p.rows_per_lane;
            const uint row = token % p.rows_per_lane;
            masked = !(step_lane < p.lanes &&
                row < min(verify_rows[p.physical_lane[min(step_lane, 3u)]], p.rows_per_lane));
        }
        if (masked) {
            ids[route] = p.n_experts;
            routing_weights[route] = 0.0f;
        } else {
            id = ids[route];
            if (id < p.n_experts) {
                atomic_fetch_or_explicit(
                    &masks[uint(simdgroup) * MT_FUSED_THREADS + id], 1u << lane,
                    memory_order_relaxed);
            } else {
                atomic_store_explicit(&invalid, 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Scan: thread e holds expert e's routes, first row and first tile ----
    const uint expert = route;
    uint count = 0;
    if (expert < p.n_experts) {
        for (uint g = 0; g < MT_FUSED_GROUPS; ++g)
            count += popcount(atomic_load_explicit(
                &masks[g * MT_FUSED_THREADS + expert], memory_order_relaxed));
    }
    const uint expert_tiles = (count + MT_TILE_ROWS - 1) / MT_TILE_ROWS;
    const uint rows_before = simd_prefix_exclusive_sum(count);
    const uint tiles_before = simd_prefix_exclusive_sum(expert_tiles);
    if (lane == 31) {
        totals[simdgroup] = rows_before + count;
        totals[MT_FUSED_GROUPS + simdgroup] = tiles_before + expert_tiles;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint first_row = rows_before;
    uint first_tile = tiles_before;
    uint routed_rows = 0;
    uint routed_tiles = 0;
    for (uint g = 0; g < MT_FUSED_GROUPS; ++g) {
        if (g < uint(simdgroup)) {
            first_row += totals[g];
            first_tile += totals[MT_FUSED_GROUPS + g];
        }
        routed_rows += totals[g];
        routed_tiles += totals[MT_FUSED_GROUPS + g];
    }
    starts[expert] = first_row;
    const uint shared_tiles =
        p.shared_tiles != 0u ? (p.n_tokens + MT_TILE_ROWS - 1) / MT_TILE_ROWS : 0u;
    const uint tiles = routed_tiles + shared_tiles;
    const bool poison =
        atomic_load_explicit(&invalid, memory_order_relaxed) != 0u || tiles > p.max_tiles;
    if (route == 0) {
        const uint launched = poison ? 1u : max(tiles, 1u);
        schedule[0] = poison ? MT_FUSED_POISON : tiles;
        grids[0] = p.gather_columns;
        grids[1] = launched;
        grids[2] = 1;
        grids[3] = p.gate_up_columns;
        grids[4] = launched;
        grids[5] = 1;
        grids[6] = p.down_columns;
        grids[7] = launched;
        grids[8] = 1;
    }
    if (poison)
        return;

    // ---- Tile descriptors and the shared expert's rows ----
    const uint column = count <= p.column_rows ? MT_FUSED_COLUMN : 0u;
    for (uint tile = 0; tile < expert_tiles; ++tile) {
        device uint *descriptor = schedule + 1 + 3 * (first_tile + tile);
        descriptor[0] = expert | column;
        descriptor[1] = first_row + tile * MT_TILE_ROWS;
        descriptor[2] = min(MT_TILE_ROWS, count - tile * MT_TILE_ROWS);
    }
    if (route < shared_tiles) {
        const uint rows = min(MT_TILE_ROWS, p.n_tokens - route * MT_TILE_ROWS);
        device uint *descriptor = schedule + 1 + 3 * (routed_tiles + route);
        descriptor[0] = MT_FUSED_SHARED | (rows <= p.column_rows ? MT_FUSED_COLUMN : 0u);
        descriptor[1] = routed_rows + route * MT_TILE_ROWS;
        descriptor[2] = rows;
    }
    if (p.shared_tiles != 0u && route < p.n_tokens)
        routes[routed_rows + route] = route_count + route;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- The route's grouped row: ascending route order within its expert ----
    if (id < p.n_experts) {
        uint rank = popcount(
            atomic_load_explicit(
                &masks[uint(simdgroup) * MT_FUSED_THREADS + id], memory_order_relaxed) &
            ((1u << lane) - 1u));
        for (uint g = 0; g < uint(simdgroup); ++g)
            rank += popcount(atomic_load_explicit(
                &masks[g * MT_FUSED_THREADS + id], memory_order_relaxed));
        routes[starts[id] + rank] = route;
    }
}

struct MoeTilesFusedGatherParams {
    uint k;
    uint top_k;
    uint route_count;
    uint _pad;
};

// Copies each tile's input rows into its grouped rows as BF16: threadgroup
// (part, tile) the part's 256 columns of the tile's rows, each thread eight
// columns of one row; grid (k / 256, tiles), 256 threads.
kernel void moe_tiles_fused_gather_f32(
    device const float *input [[buffer(0)]],
    device bfloat *gathered [[buffer(1)]],
    device const uint *schedule [[buffer(2)]],
    device const uint *routes [[buffer(3)]],
    constant MoeTilesFusedGatherParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint state = schedule[0];
    if ((state & MT_FUSED_POISON) != 0u || group.y >= state)
        return;
    device const uint *descriptor = schedule + 1 + 3 * group.y;
    const uint first = descriptor[1];
    const uint row = thread_index / 32;
    if (row >= descriptor[2])
        return;
    const uint route = routes[first + row];
    const uint input_row = route < p.route_count ? route / p.top_k : route - p.route_count;
    const uint column = group.x * 256 + (thread_index % 32) * 8;
    device const float *source = input + ulong(input_row) * p.k + column;
    device bfloat *target = gathered + (ulong(first) + row) * p.k + column;
#pragma unroll
    for (ushort i = 0; i < 8; ++i)
        target[i] = bfloat(source[i]);
}

// Gate and up of one tile, routed or shared: `moe_tiles_gate_up_impl` on the
// fused schedule's descriptors.
template <bool Q6>
kernel void moe_tiles_fused_gate_up_impl(
    device const uchar *gate_weights [[buffer(0)]],
    device const uchar *up_weights [[buffer(1)]],
    device const uchar *shared_gate_weights [[buffer(2)]],
    device const uchar *shared_up_weights [[buffer(3)]],
    device bfloat *input [[buffer(4)]],
    device bfloat *intermediate [[buffer(5)]],
    device const uint *schedule [[buffer(6)]],
    constant MoeTilesParams &p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup half stage[2 * 2 * MT_STAGE];
    threadgroup float gates[MT_TILE_ROWS * MT_COLS];
    const uint state = schedule[0];
    if ((state & MT_FUSED_POISON) != 0u || group.y >= state)
        return;
    device const uint *descriptor = schedule + 1 + 3 * group.y;
    const uint tile = descriptor[0];
    const ulong first = descriptor[1];
    const uint count = descriptor[2];
    const bool up = simdgroup != 0;
    const uint origin = group.x * MT_COLS;
    device const uchar *weights = (tile & MT_FUSED_SHARED) != 0u
        ? (up ? shared_up_weights : shared_gate_weights)
        : (up ? up_weights : gate_weights) +
            ulong(tile & MT_FUSED_EXPERT) * p.prepared_expert_stride;
    mt_project<Q6, true>(
        weights, p, input + first * p.k, count, (tile & MT_FUSED_COLUMN) != 0u,
        origin + lane, lane, stage + uint(simdgroup) * (2 * MT_STAGE), up,
        [&](uint row, uint column, float value) {
            gates[row * MT_COLS + column] = value;
        },
        [&](uint row, uint column, float value) {
            intermediate[(first + row) * p.n + origin + column] =
                mt_swiglu(gates[row * MT_COLS + column], value);
        });
}

// Down of one tile, routed or shared, to each route's own row of `output`
// (`[n_tokens * top_k + n_tokens, n]`: the routed routes, then the shared
// expert's row of every token).
template <bool Q6>
kernel void moe_tiles_fused_down_impl(
    device const uchar *down_weights [[buffer(0)]],
    device const uchar *shared_down_weights [[buffer(1)]],
    device bfloat *intermediate [[buffer(2)]],
    device float *output [[buffer(3)]],
    device const uint *schedule [[buffer(4)]],
    device const uint *routes [[buffer(5)]],
    constant MoeTilesParams &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup half stage[2 * 2 * MT_STAGE];
    const uint state = schedule[0];
    if ((state & MT_FUSED_POISON) != 0u || group.y >= state)
        return;
    device const uint *descriptor = schedule + 1 + 3 * group.y;
    const uint tile = descriptor[0];
    const ulong first = descriptor[1];
    const uint count = descriptor[2];
    const uint origin = group.x * (2 * MT_COLS) + uint(simdgroup) * MT_COLS;
    device const uchar *weights = (tile & MT_FUSED_SHARED) != 0u
        ? shared_down_weights
        : down_weights + ulong(tile & MT_FUSED_EXPERT) * p.prepared_expert_stride;
    mt_project<Q6, false>(
        weights, p, intermediate + first * p.k, count, (tile & MT_FUSED_COLUMN) != 0u,
        origin + lane, lane, stage + uint(simdgroup) * (2 * MT_STAGE), false,
        [](uint, uint, float) {},
        [&](uint row, uint column, float value) {
            output[ulong(routes[first + row]) * p.n + origin + column] = value;
        });
}

template [[host_name("moe_tiles_fused_gate_up_q4_K")]]
kernel void moe_tiles_fused_gate_up_impl<false>(
    device const uchar *, device const uchar *, device const uchar *, device const uchar *,
    device bfloat *, device bfloat *, device const uint *, constant MoeTilesParams &,
    uint2, ushort, ushort);

template [[host_name("moe_tiles_fused_gate_up_q6_K")]]
kernel void moe_tiles_fused_gate_up_impl<true>(
    device const uchar *, device const uchar *, device const uchar *, device const uchar *,
    device bfloat *, device bfloat *, device const uint *, constant MoeTilesParams &,
    uint2, ushort, ushort);

template [[host_name("moe_tiles_fused_down_q4_K")]]
kernel void moe_tiles_fused_down_impl<false>(
    device const uchar *, device const uchar *, device bfloat *, device float *,
    device const uint *, device const uint *, constant MoeTilesParams &, uint2, ushort,
    ushort);

template [[host_name("moe_tiles_fused_down_q6_K")]]
kernel void moe_tiles_fused_down_impl<true>(
    device const uchar *, device const uchar *, device bfloat *, device float *,
    device const uint *, device const uint *, constant MoeTilesParams &, uint2, ushort,
    ushort);
