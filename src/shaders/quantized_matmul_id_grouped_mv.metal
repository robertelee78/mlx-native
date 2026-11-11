// Prepared Q4_K/Q6_K expert planes and grouped staged-M8 decode tiles.
//
// The compact expert schedule groups live routed rows by expert. Decode keeps
// activations and expert intermediates in BF16, uses the one/two-row column
// path for sparse tiles, and otherwise runs exact eight-row tensor tiles.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint M8_TILE_ROWS = 64;
constant constexpr uint M8_Q4 = 0;
constant constexpr uint M8_Q6 = 1;

struct GroupedM8PrepareParams {
    uint n_experts;
    uint n;
    uint k;
    uint format;
    uint64_t source_expert_stride;
    uint64_t prepared_expert_stride;
    uint plane1_offset;
    uint meta_offset;
    uint source_row_bytes;
    uint _pad;
};

inline ulong grouped_m8_tiled_index(uint row, uint unit, uint units) {
    return (ulong(row / M8_TILE_ROWS) * units + unit) * M8_TILE_ROWS + row % M8_TILE_ROWS;
}

inline uint grouped_m8_quant_slot(uint e) {
    return 8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3);
}

inline uint grouped_m8_pair_word(thread const uchar *slots, uint c) {
    uint word = 0;
    for (uint i = 0; i < 8; ++i)
        word |= uint(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1));
    return word;
}

inline uint grouped_m8_bit_word(thread const uchar *slots, uint bits, uint word_index) {
    const uint per = 32 / bits;
    uint word = 0;
    for (uint i = 0; i < per; ++i)
        word |= uint(slots[word_index * per + i]) << (bits * i);
    return word;
}

kernel void grouped_m8_prepare_q4k_q6k(
    device const uchar *source [[buffer(0)]],
    device uchar *prepared [[buffer(1)]],
    constant GroupedM8PrepareParams &p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint groups = p.k / 32;
    const uint matrix_units = p.n * groups;
    if (index >= p.n_experts * matrix_units)
        return;
    const uint expert = index / matrix_units;
    const uint matrix_index = index % matrix_units;
    const uint row = matrix_index / groups;
    const uint group = matrix_index % groups;
    const uint block = group / 8;
    const uint within = group % 8;
    device const uchar *expert_source = source + uint64_t(expert) * p.source_expert_stride;
    device uchar *expert_prepared = prepared + uint64_t(expert) * p.prepared_expert_stride;
    device const uchar *b = expert_source + uint64_t(row) * p.source_row_bytes +
                            uint64_t(block) * (p.format == M8_Q4 ? 144 : 210);
    device uchar *plane0 = expert_prepared + grouped_m8_tiled_index(row, group, groups) * 16;
    uchar low[32], high[32];
    if (p.format == M8_Q4) {
        for (uint e = 0; e < 32; ++e)
            low[grouped_m8_quant_slot(e)] =
                (b[16 + (within / 2) * 32 + e] >> (4 * (within % 2))) & 15;
        for (uint c = 0; c < 4; ++c)
            reinterpret_cast<device uint *>(plane0)[c] = grouped_m8_pair_word(low, c);
        if (within == 0) {
            device uchar *meta = expert_prepared + p.meta_offset +
                grouped_m8_tiled_index(row, block, groups / 8) * 16;
            for (uint i = 0; i < 16; ++i)
                meta[i] = b[i];
        }
    } else {
        const uint high_block = within / 4;
        const uint quarter = within % 4;
        for (uint e = 0; e < 32; ++e) {
            low[grouped_m8_quant_slot(e)] =
                (b[64 * high_block + 32 * (quarter & 1) + e] >> (4 * (quarter >> 1))) & 15;
            high[grouped_m8_quant_slot(e)] =
                (b[128 + 32 * high_block + e] >> (2 * quarter)) & 3;
        }
        for (uint c = 0; c < 4; ++c)
            reinterpret_cast<device uint *>(plane0)[c] = grouped_m8_pair_word(low, c);
        device uchar *plane1 = expert_prepared + p.plane1_offset +
            grouped_m8_tiled_index(row, group, groups) * 8;
        reinterpret_cast<device uint *>(plane1)[0] = grouped_m8_bit_word(high, 2, 0);
        reinterpret_cast<device uint *>(plane1)[1] = grouped_m8_bit_word(high, 2, 1);
        if (within == 0) {
            device uchar *meta = expert_prepared + p.meta_offset +
                grouped_m8_tiled_index(row, block, groups / 8) * 20;
            for (uint i = 0; i < 16; ++i)
                meta[i] = b[192 + i];
            meta[16] = b[208];
            meta[17] = b[209];
            meta[18] = 0;
            meta[19] = 0;
        }
    }
}

inline uint4 grouped_m8_nibble_pairs(uint word) {
    return (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu;
}

inline uint grouped_m8_spread2(uint bits) {
    return (bits & 0x3333u) | ((bits & 0xCCCCu) << 14);
}

inline uint grouped_m8_scale_pair(packed_uint4 bytes, ushort j) {
    const uint word = j < 2 ? bytes.x : j < 4 ? bytes.y : j < 6 ? bytes.z : bytes.w;
    return word >> (16u * (j & 1));
}

inline float2 grouped_m8_q4_coef(uint4 h, ushort j) {
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

inline bfloat2 grouped_m8_linear_pair(uint pair, float scale, float minimum, ushort zero) {
    const float2 code =
        float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + zero)));
    return zero ? bfloat2(code * scale) : bfloat2(fma(code, float2(scale), float2(minimum)));
}

inline void grouped_m8_decode_q4(
    uint4 payload,
    uint4 meta,
    ushort group,
    ushort half_index,
    threadgroup bfloat *out) {
    const float2 coefficient = grouped_m8_q4_coef(meta, group);
    for (ushort chunk = 0; chunk < 4; ++chunk) {
        const uint4 pairs = grouped_m8_nibble_pairs(payload[chunk]);
        const uint2 selected = half_index ? pairs.zw : pairs.xy;
        *reinterpret_cast<threadgroup bfloat2 *>(out + 4 * chunk) =
            grouped_m8_linear_pair(selected.x, coefficient.x, coefficient.y, 0);
        *reinterpret_cast<threadgroup bfloat2 *>(out + 4 * chunk + 2) =
            grouped_m8_linear_pair(selected.y, coefficient.x, coefficient.y, 0);
    }
}

inline void grouped_m8_decode_q6(
    uint4 low,
    uint2 high,
    packed_uint4 scales,
    uint dh,
    ushort group,
    ushort half_index,
    threadgroup bfloat *out) {
    const float dscale = float(as_type<half>(ushort(dh)));
    const uint pair = grouped_m8_scale_pair(scales, group);
    const float2 scale = float2(
        dscale * float(as_type<char>(uchar(pair))),
        dscale * float(as_type<char>(uchar(pair >> 8))));
    for (ushort chunk = 0; chunk < 4; ++chunk) {
        const uint high_bits = grouped_m8_spread2(high[chunk >> 1] >> (16 * (chunk & 1)));
        const uint4 pairs = grouped_m8_nibble_pairs(low[chunk]) |
            (((uint4(high_bits) >> uint4(0, 4, 8, 12)) & 0x00030003u) << 4);
        const uint2 selected = half_index ? pairs.zw : pairs.xy;
        const float selected_scale = half_index ? scale.y : scale.x;
        *reinterpret_cast<threadgroup bfloat2 *>(out + 4 * chunk) =
            grouped_m8_linear_pair(selected.x, selected_scale, 0, 32);
        *reinterpret_cast<threadgroup bfloat2 *>(out + 4 * chunk + 2) =
            grouped_m8_linear_pair(selected.y, selected_scale, 0, 32);
    }
}

inline void grouped_m8_decode_q4_thread(
    uint4 payload,
    uint4 meta,
    ushort group,
    thread bfloat *out) {
    const float2 coefficient = grouped_m8_q4_coef(meta, group);
    for (ushort half_index = 0; half_index < 2; ++half_index) {
        for (ushort chunk = 0; chunk < 4; ++chunk) {
            const uint4 pairs = grouped_m8_nibble_pairs(payload[chunk]);
            const uint2 selected = half_index ? pairs.zw : pairs.xy;
            *reinterpret_cast<thread bfloat2 *>(out + half_index * 16 + 4 * chunk) =
                grouped_m8_linear_pair(selected.x, coefficient.x, coefficient.y, 0);
            *reinterpret_cast<thread bfloat2 *>(out + half_index * 16 + 4 * chunk + 2) =
                grouped_m8_linear_pair(selected.y, coefficient.x, coefficient.y, 0);
        }
    }
}

inline void grouped_m8_decode_q6_thread(
    uint4 low,
    uint2 high,
    packed_uint4 scales,
    uint dh,
    ushort group,
    thread bfloat *out) {
    const float dscale = float(as_type<half>(ushort(dh)));
    const uint pair = grouped_m8_scale_pair(scales, group);
    const float2 scale = float2(
        dscale * float(as_type<char>(uchar(pair))),
        dscale * float(as_type<char>(uchar(pair >> 8))));
    for (ushort half_index = 0; half_index < 2; ++half_index) {
        for (ushort chunk = 0; chunk < 4; ++chunk) {
            const uint high_bits = grouped_m8_spread2(high[chunk >> 1] >> (16 * (chunk & 1)));
            const uint4 pairs = grouped_m8_nibble_pairs(low[chunk]) |
                (((uint4(high_bits) >> uint4(0, 4, 8, 12)) & 0x00030003u) << 4);
            const uint2 selected = half_index ? pairs.zw : pairs.xy;
            const float selected_scale = half_index ? scale.y : scale.x;
            *reinterpret_cast<thread bfloat2 *>(out + half_index * 16 + 4 * chunk) =
                grouped_m8_linear_pair(selected.x, selected_scale, 0, 32);
            *reinterpret_cast<thread bfloat2 *>(out + half_index * 16 + 4 * chunk + 2) =
                grouped_m8_linear_pair(selected.y, selected_scale, 0, 32);
        }
    }
}

struct GroupedM8Params {
    int32_t k;
    int32_t n;
    int32_t n_tokens;
    int32_t top_k;
    int32_t n_experts;
    uint32_t slotted_input;
    uint64_t prepared_expert_stride;
    uint32_t plane1_offset;
    uint32_t meta_offset;
};

kernel void grouped_m8_gather_f32_bf16(
    device const float *input [[buffer(0)]],
    device bfloat *gathered [[buffer(1)]],
    device const uint32_t *schedule [[buffer(2)]],
    device const int32_t *routes [[buffer(3)]],
    constant GroupedM8Params &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.y >= state)
        return;
    device const uint32_t *descriptor = schedule + 1 + 3 * group.y;
    const uint first_route = descriptor[1];
    const uint route_count = descriptor[2];
    for (uint row = 0; row < route_count; ++row) {
        const int packed = routes[first_route + row];
        const int input_row = p.slotted_input ? packed : packed / p.top_k;
        for (uint column = thread_index; column < uint(p.k); column += 256) {
            gathered[(uint64_t(first_route) + row) * p.k + column] =
                bfloat(input[int64_t(input_row) * p.k + column]);
        }
    }
}

kernel void grouped_m8_gather_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *gathered [[buffer(1)]],
    device const uint32_t *schedule [[buffer(2)]],
    device const int32_t *routes [[buffer(3)]],
    constant GroupedM8Params &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.y >= state)
        return;
    device const uint32_t *descriptor = schedule + 1 + 3 * group.y;
    const uint first_route = descriptor[1];
    const uint route_count = descriptor[2];
    for (uint row = 0; row < route_count; ++row) {
        const int packed = routes[first_route + row];
        const int input_row = p.slotted_input ? packed : packed / p.top_k;
        for (uint column = thread_index; column < uint(p.k); column += 256) {
            gathered[(uint64_t(first_route) + row) * p.k + column] =
                input[int64_t(input_row) * p.k + column];
        }
    }
}


constant constexpr ushort M8_EP_F32 = 0;
constant constexpr ushort M8_EP_BF16 = 1;
constant constexpr ushort M8_EP_SILU_MUL_BF16 = 2;

template <bool Q6, ushort Ep>
kernel void grouped_m8_impl(
    device const uchar *weights [[buffer(0)]],
    device bfloat *input [[buffer(1)]],
    device uchar *output [[buffer(2)]],
    device const uint32_t *schedule [[buffer(3)]],
    device const int32_t *routes [[buffer(4)]],
    constant GroupedM8Params &p [[buffer(5)]],
    device const bfloat *aux [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint ROWS = 8;
    constexpr uint COLS = 32;
    constexpr uint STEP = 32;
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.y >= state)
        return;
    device const uint32_t *descriptor = schedule + 1 + 3 * group.y;
    const uint expert = descriptor[0];
    const uint first_route = descriptor[1];
    const uint route_count = descriptor[2];
    const uint output_origin = group.x * 64 + uint(simdgroup) * COLS;
    device const uchar *expert_weights =
        weights + uint64_t(expert) * p.prepared_expert_stride;
    const uint groups = uint(p.k) / STEP;
    const uint units = groups / 8;

    if (route_count <= 2) {
        int packed[2];
        float sums[2] = {0.0f, 0.0f};
        for (uint row = 0; row < route_count; ++row)
            packed[row] = routes[first_route + row];
        const uint weight_row = output_origin + lane;
        for (uint g = 0; g < groups; ++g) {
            const ulong tiled = grouped_m8_tiled_index(weight_row, g, groups);
            const uint4 low = *reinterpret_cast<device const uint4 *>(
                expert_weights + tiled * 16);
            bfloat4 decoded[8];
            if constexpr (Q6) {
                const uint2 high = *reinterpret_cast<device const uint2 *>(
                    expert_weights + p.plane1_offset + tiled * 8);
                device const uchar *meta = expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 20;
                const packed_uint4 scales =
                    *reinterpret_cast<device const packed_uint4 *>(meta);
                const uint dh = *reinterpret_cast<device const uint *>(meta + 16);
                grouped_m8_decode_q6_thread(
                    low, high, scales, dh, ushort(g % 8),
                    reinterpret_cast<thread bfloat *>(decoded));
            } else {
                const uint4 meta = *reinterpret_cast<device const uint4 *>(
                    expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 16);
                grouped_m8_decode_q4_thread(
                    low, meta, ushort(g % 8),
                    reinterpret_cast<thread bfloat *>(decoded));
            }
            for (uint row = 0; row < route_count; ++row) {
                device const bfloat4 *x = reinterpret_cast<device const bfloat4 *>(
                    input + (uint64_t(first_route) + row) * p.k + g * STEP);
                float group_sum = 0.0f;
#pragma unroll
                for (ushort i = 0; i < 8; ++i) {
                    const float4 xv = float4(x[i]);
                    const float4 wv = float4(decoded[i]);
                    group_sum += wv.x * xv.x;
                    group_sum += wv.y * xv.y;
                    group_sum += wv.z * xv.z;
                    group_sum += wv.w * xv.w;
                }
                sums[row] += group_sum;
            }
        }
        for (uint row = 0; row < route_count; ++row) {
            const int64_t output_index = int64_t(packed[row]) * p.n + weight_row;
            if constexpr (Ep == M8_EP_F32)
                reinterpret_cast<device float *>(output)[output_index] = sums[row];
            else if constexpr (Ep == M8_EP_BF16)
                reinterpret_cast<device bfloat *>(output)[output_index] = bfloat(sums[row]);
            else {
                const float gate = float(aux[output_index]);
                reinterpret_cast<device bfloat *>(output)[output_index] =
                    bfloat((gate / (1.0f + exp(-gate))) * sums[row]);
            }
        }
        return;
    }

    threadgroup bfloat weight_tile[2 * COLS * STEP];
    threadgroup float result_tile[2 * ROWS * COLS];
    threadgroup int packed_routes[ROWS];
    threadgroup ushort live_rows;

    matmul2d<matmul2d_descriptor(
        ROWS, COLS, STEP, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;

    for (uint route_base = 0; route_base < route_count; route_base += ROWS) {
        if (thread_index == 0)
            live_rows = ushort(min(ROWS, route_count - route_base));
        if (thread_index < ROWS)
            packed_routes[thread_index] =
                thread_index < min(ROWS, route_count - route_base)
                    ? routes[first_route + route_base + thread_index]
                    : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto all_left = tensor(
            input + (uint64_t(first_route) + route_base) * p.k,
            dextents<int32_t, 2>{p.k, ROWS},
            array<int32_t, 2>{1, p.k});
        auto first_left = all_left.template slice<STEP, ROWS>(0, 0);
        threadgroup bfloat *simd_weight = weight_tile + uint(simdgroup) * COLS * STEP;
        auto right = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(
            simd_weight, dextents<int32_t, 2>{STEP, COLS},
            array<int32_t, 2>{1, STEP});
        auto sum = mm.get_destination_cooperative_tensor<decltype(first_left), decltype(right), float>();
        for (ushort i = 0; i < sum.get_capacity(); ++i)
            sum[i] = 0.0f;

        for (uint g = 0; g < groups; ++g) {

            const uint weight_row = output_origin + lane;
            const ulong tiled = grouped_m8_tiled_index(weight_row, g, groups);
            const uint4 low = *reinterpret_cast<device const uint4 *>(
                expert_weights + tiled * 16);
            if constexpr (Q6) {
                const uint2 high = *reinterpret_cast<device const uint2 *>(
                    expert_weights + p.plane1_offset + tiled * 8);
                device const uchar *meta = expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 20;
                const packed_uint4 scales =
                    *reinterpret_cast<device const packed_uint4 *>(meta);
                const uint dh = *reinterpret_cast<device const uint *>(meta + 16);
                grouped_m8_decode_q6(
                    low, high, scales, dh, ushort(g % 8), 0,
                    simd_weight + lane * STEP);
                grouped_m8_decode_q6(
                    low, high, scales, dh, ushort(g % 8), 1,
                    simd_weight + lane * STEP + 16);
            } else {
                const uint4 meta = *reinterpret_cast<device const uint4 *>(
                    expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 16);
                grouped_m8_decode_q4(
                    low, meta, ushort(g % 8), 0,
                    simd_weight + lane * STEP);
                grouped_m8_decode_q4(
                    low, meta, ushort(g % 8), 1,
                    simd_weight + lane * STEP + 16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto left = all_left.template slice<STEP, ROWS>(g * STEP, 0);
            mm.run(left, right, sum);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float *simd_result = result_tile + uint(simdgroup) * ROWS * COLS;
        auto stored = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            simd_result, dextents<int32_t, 2>{COLS, ROWS},
            array<int32_t, 2>{1, COLS});
        sum.store(stored);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = thread_index; index < uint(live_rows) * 64; index += 64) {
            const uint row = index / 64;
            const uint column = index % 64;
            const uint source_simdgroup = column / COLS;
            const uint source_column = column % COLS;
            const int64_t output_index =
                int64_t(packed_routes[row]) * p.n + group.x * 64 + column;
            const float value =
                result_tile[source_simdgroup * ROWS * COLS + row * COLS + source_column];
            if constexpr (Ep == M8_EP_F32)
                reinterpret_cast<device float *>(output)[output_index] = value;
            else if constexpr (Ep == M8_EP_BF16)
                reinterpret_cast<device bfloat *>(output)[output_index] = bfloat(value);
            else {
                const float gate = float(aux[output_index]);
                reinterpret_cast<device bfloat *>(output)[output_index] =
                    bfloat((gate / (1.0f + exp(-gate))) * value);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <bool Q6>
kernel void grouped_m8_gate_up_impl(
    device const uchar *gate_weights [[buffer(0)]],
    device const uchar *up_weights [[buffer(1)]],
    device bfloat *input [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device const uint32_t *schedule [[buffer(4)]],
    device const int32_t *routes [[buffer(5)]],
    constant GroupedM8Params &p [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint ROWS = 8;
    constexpr uint COLS = 32;
    constexpr uint STEP = 32;
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.y >= state)
        return;
    device const uint32_t *descriptor = schedule + 1 + 3 * group.y;
    const uint expert = descriptor[0];
    const uint first_route = descriptor[1];
    const uint route_count = descriptor[2];
    const uint output_origin = group.x * COLS;
    const bool up = simdgroup != 0;
    device const uchar *weights = up ? up_weights : gate_weights;
    device const uchar *expert_weights =
        weights + uint64_t(expert) * p.prepared_expert_stride;
    const uint groups = uint(p.k) / STEP;
    const uint units = groups / 8;

    if (route_count <= 2) {
        threadgroup bfloat gates[2 * COLS];
        int packed[2];
        float sums[2] = {0.0f, 0.0f};
        for (uint row = 0; row < route_count; ++row)
            packed[row] = routes[first_route + row];
        const uint weight_row = output_origin + lane;
        for (uint g = 0; g < groups; ++g) {
            const ulong tiled = grouped_m8_tiled_index(weight_row, g, groups);
            const uint4 low = *reinterpret_cast<device const uint4 *>(
                expert_weights + tiled * 16);
            bfloat4 decoded[8];
            if constexpr (Q6) {
                const uint2 high = *reinterpret_cast<device const uint2 *>(
                    expert_weights + p.plane1_offset + tiled * 8);
                device const uchar *meta = expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 20;
                const packed_uint4 scales =
                    *reinterpret_cast<device const packed_uint4 *>(meta);
                const uint dh = *reinterpret_cast<device const uint *>(meta + 16);
                grouped_m8_decode_q6_thread(
                    low, high, scales, dh, ushort(g % 8),
                    reinterpret_cast<thread bfloat *>(decoded));
            } else {
                const uint4 meta = *reinterpret_cast<device const uint4 *>(
                    expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 16);
                grouped_m8_decode_q4_thread(
                    low, meta, ushort(g % 8),
                    reinterpret_cast<thread bfloat *>(decoded));
            }
            for (uint row = 0; row < route_count; ++row) {
                device const bfloat4 *x = reinterpret_cast<device const bfloat4 *>(
                    input + (uint64_t(first_route) + row) * p.k + g * STEP);
                float group_sum = 0.0f;
#pragma unroll
                for (ushort i = 0; i < 8; ++i) {
                    const float4 xv = float4(x[i]);
                    const float4 wv = float4(decoded[i]);
                    group_sum += wv.x * xv.x;
                    group_sum += wv.y * xv.y;
                    group_sum += wv.z * xv.z;
                    group_sum += wv.w * xv.w;
                }
                sums[row] += group_sum;
            }
        }
        if (!up)
            for (uint row = 0; row < route_count; ++row)
                gates[row * COLS + lane] = bfloat(sums[row]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (up) {
            for (uint row = 0; row < route_count; ++row) {
                const int64_t at = int64_t(packed[row]) * p.n + weight_row;
                const float gate = float(gates[row * COLS + lane]);
                const float up_value = float(bfloat(sums[row]));
                output[at] = bfloat((gate / (1.0f + exp(-gate))) * up_value);
            }
        }
        return;
    }

    threadgroup bfloat weight_tile[2 * COLS * STEP];
    threadgroup float result_tile[2 * ROWS * COLS];
    threadgroup int packed_routes[ROWS];
    threadgroup ushort live_rows;
    matmul2d<matmul2d_descriptor(
        ROWS, COLS, STEP, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<1>> mm;

    for (uint route_base = 0; route_base < route_count; route_base += ROWS) {
        if (thread_index == 0)
            live_rows = ushort(min(ROWS, route_count - route_base));
        if (thread_index < ROWS)
            packed_routes[thread_index] =
                thread_index < min(ROWS, route_count - route_base)
                    ? routes[first_route + route_base + thread_index]
                    : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto all_left = tensor(
            input + (uint64_t(first_route) + route_base) * p.k,
            dextents<int32_t, 2>{p.k, ROWS},
            array<int32_t, 2>{1, p.k});
        auto first_left = all_left.template slice<STEP, ROWS>(0, 0);
        threadgroup bfloat *simd_weight =
            weight_tile + uint(simdgroup) * COLS * STEP;
        auto right = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(
            simd_weight, dextents<int32_t, 2>{STEP, COLS},
            array<int32_t, 2>{1, STEP});
        auto sum = mm.get_destination_cooperative_tensor<
            decltype(first_left), decltype(right), float>();
        for (ushort i = 0; i < sum.get_capacity(); ++i)
            sum[i] = 0.0f;

        for (uint g = 0; g < groups; ++g) {
            const uint weight_row = output_origin + lane;
            const ulong tiled = grouped_m8_tiled_index(weight_row, g, groups);
            const uint4 low = *reinterpret_cast<device const uint4 *>(
                expert_weights + tiled * 16);
            if constexpr (Q6) {
                const uint2 high = *reinterpret_cast<device const uint2 *>(
                    expert_weights + p.plane1_offset + tiled * 8);
                device const uchar *meta = expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 20;
                const packed_uint4 scales =
                    *reinterpret_cast<device const packed_uint4 *>(meta);
                const uint dh = *reinterpret_cast<device const uint *>(meta + 16);
                grouped_m8_decode_q6(
                    low, high, scales, dh, ushort(g % 8), 0,
                    simd_weight + lane * STEP);
                grouped_m8_decode_q6(
                    low, high, scales, dh, ushort(g % 8), 1,
                    simd_weight + lane * STEP + 16);
            } else {
                const uint4 meta = *reinterpret_cast<device const uint4 *>(
                    expert_weights + p.meta_offset +
                    grouped_m8_tiled_index(weight_row, g / 8, units) * 16);
                grouped_m8_decode_q4(
                    low, meta, ushort(g % 8), 0,
                    simd_weight + lane * STEP);
                grouped_m8_decode_q4(
                    low, meta, ushort(g % 8), 1,
                    simd_weight + lane * STEP + 16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            auto left = all_left.template slice<STEP, ROWS>(g * STEP, 0);
            mm.run(left, right, sum);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float *simd_result =
            result_tile + uint(simdgroup) * ROWS * COLS;
        auto stored = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            simd_result, dextents<int32_t, 2>{COLS, ROWS},
            array<int32_t, 2>{1, COLS});
        sum.store(stored);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (up) {
            for (uint index = lane; index < uint(live_rows) * COLS; index += 32) {
                const uint row = index / COLS;
                const uint column = index % COLS;
                const int64_t at =
                    int64_t(packed_routes[row]) * p.n + output_origin + column;
                const float gate = float(bfloat(result_tile[row * COLS + column]));
                const float up_value =
                    float(bfloat(result_tile[ROWS * COLS + row * COLS + column]));
                output[at] =
                    bfloat((gate / (1.0f + exp(-gate))) * up_value);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}


template [[host_name("grouped_m8_id_q4_K_f32")]]
kernel void grouped_m8_impl<false, M8_EP_F32>(
    device const uchar *, device bfloat *, device uchar *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_id_q6_K_f32")]]
kernel void grouped_m8_impl<true, M8_EP_F32>(
    device const uchar *, device bfloat *, device uchar *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_id_q4_K_bf16")]]
kernel void grouped_m8_impl<false, M8_EP_BF16>(
    device const uchar *, device bfloat *, device uchar *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_id_q6_K_bf16")]]
kernel void grouped_m8_impl<true, M8_EP_BF16>(
    device const uchar *, device bfloat *, device uchar *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_id_q4_K_silu_mul_bf16")]]
kernel void grouped_m8_impl<false, M8_EP_SILU_MUL_BF16>(
    device const uchar *, device bfloat *, device uchar *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_id_q6_K_silu_mul_bf16")]]
kernel void grouped_m8_impl<true, M8_EP_SILU_MUL_BF16>(
    device const uchar *, device bfloat *, device uchar *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_gate_up_q4_K_bf16")]]
kernel void grouped_m8_gate_up_impl<false>(
    device const uchar *, device const uchar *, device bfloat *, device bfloat *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_gate_up_q6_K_bf16")]]
kernel void grouped_m8_gate_up_impl<true>(
    device const uchar *, device const uchar *, device bfloat *, device bfloat *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, uint2, ushort, ushort, ushort);
