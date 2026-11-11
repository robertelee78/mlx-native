// Expert-grouped Q4_K/Q6_K matvec tiles for decode-sized MoE batches.
//
// The compact mm_id map groups routed rows by expert into live descriptors.
// One threadgroup consumes one descriptor and keeps each expert weight row live
// while evaluating up to five routed activations at a time. Unlike the 32-row
// matrix tile, this path needs no activation staging and remains efficient when
// a decode step routes only a handful of rows to each expert.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define QK_K 256
#define GROUPED_ROWS 5

struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    half d;
};

struct GroupedMvParams {
    int32_t k;
    int32_t n;
    int32_t n_tokens;
    int32_t top_k;
    int32_t n_experts;
    uint32_t slotted_input;
    uint64_t expert_stride;
};

inline bool grouped_mv_descriptor(
    device float *output,
    device const uint32_t *schedule,
    constant GroupedMvParams &p,
    uint2 group,
    uint thread_index,
    thread uint &expert,
    thread uint &first_route,
    thread uint &route_count) {
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u) {
        if (group.x == 0 && group.y == 0) {
            const uint total = uint(p.n_tokens) * uint(p.top_k) * uint(p.n);
            for (uint i = thread_index; i < total; i += 64)
                output[i] = numeric_limits<float>::quiet_NaN();
        }
        return false;
    }
    if (group.y >= state)
        return false;
    device const uint *descriptor = schedule + 1 + 3 * group.y;
    expert = descriptor[0];
    first_route = descriptor[1];
    route_count = descriptor[2];
    return true;
}

kernel void grouped_mv_id_q4_K_f32_r5(
    device const char *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint32_t *schedule [[buffer(3)]],
    device const int32_t *routes [[buffer(4)]],
    constant GroupedMvParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    uint expert, first_route, route_count;
    if (!grouped_mv_descriptor(output, schedule, p, group, thread_index,
                               expert, first_route, route_count))
        return;

    const int output_row = 2 * int(group.x) + int(simdgroup);
    if (output_row >= p.n)
        return;
    const int blocks = p.k / QK_K;
    device const block_q4_K *weight =
        reinterpret_cast<device const block_q4_K *>(
            weights + uint64_t(expert) * p.expert_stride) + output_row * blocks;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;
    const short tid = lane / 4;
    const short ix = lane % 4;
    const short iq = tid / 4;
    const short ir = tid % 4;
    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    for (uint route_base = 0; route_base < route_count; route_base += GROUPED_ROWS) {
        const short live = short(min(uint(GROUPED_ROWS), route_count - route_base));
        int packed[GROUPED_ROWS];
        device const float *row_input[GROUPED_ROWS];
        float sums[GROUPED_ROWS];
        device const float *y[GROUPED_ROWS];
        for (short c = 0; c < live; ++c) {
            packed[c] = routes[first_route + route_base + uint(c)];
            const int input_row = p.slotted_input ? packed[c] : packed[c] / p.top_k;
            row_input[c] = input + int64_t(input_row) * p.k;
            y[c] = row_input[c] + ix * QK_K + y_offset;
            sums[c] = 0.0f;
        }

        for (int i = ix; i < blocks; i += 4) {
            device const uint8_t *q1 = weight[i].qs + q_offset;
            device const uint8_t *q2 = q1 + 64;
            device const half *dh = &weight[i].d;
            device const uint16_t *a =
                reinterpret_cast<device const uint16_t *>(weight[i].scales) + iq;
            uint16_t sc16[4];
            thread const uint8_t *sc8 = reinterpret_cast<thread const uint8_t *>(sc16);
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            const float dall = float(dh[0]);
            const float dmin = float(dh[1]);

            for (short c = 0; c < live; ++c) {
                device const float *y1 = y[c];
                device const float *y2 = y1 + 128;
                float yl[16], yh[16];
                float4 sumy = float4(0.0f);
                for (int l = 0; l < 8; ++l) {
                    yl[l] = y1[l];
                    sumy[0] += yl[l];
                    yl[l + 8] = y1[l + 32];
                    sumy[1] += yl[l + 8];
                    yh[l] = y2[l];
                    sumy[2] += yh[l];
                    yh[l + 8] = y2[l + 32];
                    sumy[3] += yh[l + 8];
                }
                float4 acc1 = float4(0.0f);
                for (int l = 0; l < 8; ++l) {
                    acc1[0] += yl[l] * float(q1[l] & 0x0f);
                    acc1[1] += yl[l + 8] * float(q1[l] & 0xf0);
                    acc1[2] += yh[l] * float(q2[l] & 0x0f);
                    acc1[3] += yh[l + 8] * float(q2[l] & 0xf0);
                }
                sums[c] += dall * (float(sc8[0]) * acc1[0] +
                                   float(sc8[1]) * (acc1[1] / 16.0f) +
                                   float(sc8[4]) * acc1[2] +
                                   float(sc8[5]) * (acc1[3] / 16.0f)) -
                           dmin * (sumy[0] * float(sc8[2]) +
                                   sumy[1] * float(sc8[3]) +
                                   sumy[2] * float(sc8[6]) +
                                   sumy[3] * float(sc8[7]));
                y[c] += 4 * QK_K;
            }
        }

        for (short c = 0; c < live; ++c) {
            const float total = simd_sum(sums[c]);
            if (lane == 0)
                output[int64_t(packed[c]) * p.n + output_row] = total;
        }
    }
}

kernel void grouped_mv_id_q6_K_f32_r5(
    device const char *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint32_t *schedule [[buffer(3)]],
    device const int32_t *routes [[buffer(4)]],
    constant GroupedMvParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    uint expert, first_route, route_count;
    if (!grouped_mv_descriptor(output, schedule, p, group, thread_index,
                               expert, first_route, route_count))
        return;

    constexpr short rows_per_simdgroup = 2;
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0c;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xc0;
    const int first_output = (int(group.x) * 2 + int(simdgroup)) * rows_per_simdgroup;
    if (first_output >= p.n)
        return;
    const int blocks = p.k / QK_K;
    device const block_q6_K *weight_base =
        reinterpret_cast<device const block_q6_K *>(
            weights + uint64_t(expert) * p.expert_stride) + first_output * blocks;

    const short tid = lane / 2;
    const short ix = lane % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    for (uint route_base = 0; route_base < route_count; route_base += GROUPED_ROWS) {
        const short live = short(min(uint(GROUPED_ROWS), route_count - route_base));
        int packed[GROUPED_ROWS];
        device const float *row_input[GROUPED_ROWS];
        float sums[rows_per_simdgroup][GROUPED_ROWS];
        float yl[GROUPED_ROWS][16];
        for (short c = 0; c < live; ++c) {
            packed[c] = routes[first_route + route_base + uint(c)];
            const int input_row = p.slotted_input ? packed[c] : packed[c] / p.top_k;
            row_input[c] = input + int64_t(input_row) * p.k;
        }
        for (short r = 0; r < rows_per_simdgroup; ++r)
            for (short c = 0; c < live; ++c)
                sums[r][c] = 0.0f;

        for (int i = ix; i < blocks; i += 2) {
            for (short c = 0; c < live; ++c) {
                device const float *y = row_input[c] + i * QK_K + y_offset;
                for (int l = 0; l < 4; ++l) {
                    yl[c][4 * l] = y[l];
                    yl[c][4 * l + 1] = y[l + 32];
                    yl[c][4 * l + 2] = y[l + 64];
                    yl[c][4 * l + 3] = y[l + 96];
                }
            }
            for (short r = 0; r < rows_per_simdgroup; ++r) {
                if (first_output + r >= p.n)
                    continue;
                device const block_q6_K *weight = weight_base + r * blocks;
                device const uint8_t *q1 = weight[i].ql + q_offset_l;
                device const uint8_t *q2 = q1 + 32;
                device const uint8_t *qh = weight[i].qh + q_offset_h;
                device const int8_t *sc = weight[i].scales + is;
                const float dall = weight[i].d;
                for (short c = 0; c < live; ++c) {
                    float4 dot = float4(0.0f);
                    for (int l = 0; l < 4; ++l) {
                        dot[0] += yl[c][4 * l] *
                                  (int8_t((q1[l] & 0x0f) | ((qh[l] & kmask1) << 4)) - 32);
                        dot[1] += yl[c][4 * l + 1] *
                                  (int8_t((q2[l] & 0x0f) | ((qh[l] & kmask2) << 2)) - 32);
                        dot[2] += yl[c][4 * l + 2] *
                                  (int8_t((q1[l] >> 4) | (qh[l] & kmask3)) - 32);
                        dot[3] += yl[c][4 * l + 3] *
                                  (int8_t((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
                    }
                    sums[r][c] += dall * (dot[0] * sc[0] + dot[1] * sc[2] +
                                          dot[2] * sc[4] + dot[3] * sc[6]);
                }
            }
        }

        for (short r = 0; r < rows_per_simdgroup; ++r) {
            const int output_row = first_output + r;
            for (short c = 0; c < live; ++c) {
                const float total = simd_sum(sums[r][c]);
                if (lane == 0 && output_row < p.n)
                    output[int64_t(packed[c]) * p.n + output_row] = total;
            }
        }
    }
}

// Prepared expert weights and an exact eight-row Metal tensor tile. The
// prepared layout is expert-local: low-code groups, optional Q6 high bits,
// then one metadata record per 256-value super-block.
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

// Apache-2.0 register-matrix structure adapted from incoai/splash
// 031a25a6, runtime/metal/kernels/{common/gguf_sgmatrix.h,
// decode/linear_gguf_sgmatrix.metal,shared/moe.metal}. The prepared Q4_K/Q6_K
// planes above have the same 64-row tiled image; this local specialization
// keeps only those two formats and one grouped eight-row MoE tile.
inline ulong grouped_m8_table_sums_per_tile(uint width) {
    return ulong(width) * 3 / 4;
}

inline uint grouped_m8_xt_offset(uint j, uint kp, uint m) {
    return (((j >> 2) * 8 + kp) * 4 + (m >> 1)) * 8 +
           (j & 3) * 2 + (m & 1);
}

inline uint2 grouped_m8_klogical(uint k) {
    const uint q = k >> 5;
    const uint h = (k >> 4) & 1;
    const uint a = (k >> 2) & 3;
    const uint b = (k >> 1) & 1;
    const uint e = k & 1;
    return uint2(4 * q + 2 * h + b, 2 * a + e);
}

inline void grouped_m8_write_table(
    device bfloat *table,
    device float *sums,
    uint width,
    uint span,
    uint row,
    uint lane,
    bfloat a,
    bfloat b) {
    const uint2 logical = grouped_m8_klogical(2 * lane);
    table[span * 512 + grouped_m8_xt_offset(logical.x, logical.y, row)] = a;
    table[span * 512 + grouped_m8_xt_offset(logical.x, logical.y + 1, row)] = b;
    float sum = float(a) + float(b);
    sum += simd_shuffle_xor(sum, 1u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 4u);
    if ((lane & 7) == 0)
        sums[(span * 4 + lane / 8) * 8 + row] = -160.0f * sum;
    const float sum32 = sum + simd_shuffle_xor(sum, 8u);
    if ((lane & 15) == 0)
        sums[ulong(width) / 2 + (span * 2 + lane / 16) * 8 + row] = sum32;
}

template <bool F32>
kernel void grouped_m8_gather_table_impl(
    device const uchar *input [[buffer(0)]],
    device bfloat *table [[buffer(1)]],
    device float *sums [[buffer(2)]],
    device const uint32_t *schedule [[buffer(3)]],
    device const int32_t *routes [[buffer(4)]],
    constant GroupedM8Params &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.x >= state)
        return;
    device const uint32_t *descriptor = schedule + 1 + 3 * group.x;
    const uint expert = descriptor[0];
    const uint first_route = descriptor[1];
    const uint route_count = descriptor[2];
    const uint spans = uint(p.k) / 64;
    for (uint route_base = 0; route_base < route_count; route_base += 8) {
        const uint live = min(8u, route_count - route_base);
        const uint route_index = first_route + route_base + simdgroup;
        const int packed = simdgroup < live ? routes[route_index] : -1;
        const int input_row = packed < 0 ? -1 :
            (p.slotted_input ? packed : packed / p.top_k);
        const ulong storage_row = ulong(first_route) + 7ul * expert + route_base;
        device bfloat *tile = table + storage_row * p.k;
        device float *tile_sums =
            sums + storage_row * grouped_m8_table_sums_per_tile(p.k) / 8;
        for (uint span = group.y * 4; span < min(group.y * 4 + 4, spans); ++span) {
            const uint column = span * 64 + 2 * lane;
            bfloat2 values = bfloat2(float2(0.0f));
            if (input_row >= 0) {
                const ulong at = ulong(input_row) * p.k + column;
                if constexpr (F32) {
                    device const float *source =
                        reinterpret_cast<device const float *>(input);
                    values = bfloat2(float2(source[at], source[at + 1]));
                } else {
                    device const bfloat *source =
                        reinterpret_cast<device const bfloat *>(input);
                    values = *reinterpret_cast<device const bfloat2 *>(source + at);
                }
            }
            grouped_m8_write_table(
                tile, tile_sums, p.k, span, simdgroup, lane, values.x, values.y);
        }
    }
}

struct GroupedM8RegisterLane {
    ushort fm;
    ushort fn;
};

inline GroupedM8RegisterLane grouped_m8_register_lane(uint lane) {
    const uint qid = lane >> 2;
    return {
        ushort((qid & 4) | ((lane >> 1) & 3)),
        ushort(((qid & 2) << 1) | ((lane & 1) << 1)),
    };
}

template <typename T>
inline thread vec<T, 2> &grouped_m8_thread_elements(
    thread simdgroup_matrix<T, 8, 8> &matrix) {
    return reinterpret_cast<thread vec<T, 2> &>(matrix.thread_elements());
}

inline void grouped_m8_register_mma(
    thread float2 &accumulator,
    bfloat2 a,
    bfloat2 b) {
    simdgroup_matrix<bfloat, 8, 8> left, right;
    simdgroup_matrix<float, 8, 8> current, result;
    grouped_m8_thread_elements(left) = a;
    grouped_m8_thread_elements(right) = b;
    grouped_m8_thread_elements(current) = accumulator;
    simdgroup_multiply_accumulate(result, left, right, current);
    accumulator = grouped_m8_thread_elements(result);
}

inline bfloat2 grouped_m8_register_operand(uint pair) {
    return as_type<bfloat2>(pair + 0x43004300u);
}

template <bool Q6, ushort Ep>
kernel void grouped_m8_register_impl(
    device const uchar *weights [[buffer(0)]],
    device const bfloat *table [[buffer(1)]],
    device const float *sums [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    device const uint32_t *schedule [[buffer(4)]],
    device const int32_t *routes [[buffer(5)]],
    constant GroupedM8Params &p [[buffer(6)]],
    device const bfloat *aux [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint state = schedule[0];
    if ((state & 0x80000000u) != 0u || group.y >= state)
        return;
    device const uint32_t *descriptor = schedule + 1 + 3 * group.y;
    const uint expert = descriptor[0];
    const uint first_route = descriptor[1];
    const uint route_count = descriptor[2];
    device const uchar *expert_weights =
        weights + ulong(expert) * p.prepared_expert_stride;
    const uint groups = uint(p.k) / 32;
    const uint units = groups / 8;
    const GroupedM8RegisterLane mapped = grouped_m8_register_lane(lane);
    const uint fm = mapped.fm;
    const uint fn = mapped.fn;
    const uint chunk = fn / 2;
    const uint plane_tile = group.x;
    const uint plane_row = uint(simdgroup) * 16 + fm;

    for (uint route_base = 0; route_base < route_count; route_base += 8) {
        const uint live = min(8u, route_count - route_base);
        const ulong storage_row = ulong(first_route) + 7ul * expert + route_base;
        device const bfloat *tile = table + storage_row * p.k;
        device const float *tile_sums =
            sums + storage_row * grouped_m8_table_sums_per_tile(p.k) / 8;
        float2 accumulator[2] = {float2(0.0f), float2(0.0f)};

        for (uint span = 0; span < uint(p.k) / 64; ++span) {
            device const vec<bfloat, 8> *xt =
                reinterpret_cast<device const vec<bfloat, 8> *>(tile + ulong(span) * 512);
            for (uint q = 0; q < 2; ++q) {
                const uint g = 2 * span + q;
                const vec<bfloat, 8> activation = xt[(8 * q + fm) * 4 + chunk];
                const bfloat2 *activation_pairs =
                    reinterpret_cast<thread const bfloat2 *>(&activation);
                for (uint nf = 0; nf < 2; ++nf) {
                    const uint column = plane_row + 8 * nf;
                    const ulong tiled =
                        (ulong(plane_tile) * groups + g) * 64 + column;
                    const uint low = *reinterpret_cast<device const uint *>(
                        expert_weights + tiled * 16 + chunk * 4);
                    const uint4 low_codes = grouped_m8_nibble_pairs(low);
                    if constexpr (Q6) {
                        const ushort high = *reinterpret_cast<device const ushort *>(
                            expert_weights + p.plane1_offset + tiled * 8 + chunk * 2);
                        const uint high_bits = grouped_m8_spread2(high);
                        const uint4 codes = low_codes |
                            (((uint4(high_bits) >> uint4(0, 4, 8, 12)) &
                              0x00030003u) << 4);
                        device const uchar *meta = expert_weights + p.meta_offset +
                            ((ulong(plane_tile) * units + g / 8) * 64 + column) * 20;
                        const packed_uint4 scales =
                            *reinterpret_cast<device const packed_uint4 *>(meta);
                        const uint dh = *reinterpret_cast<device const uint *>(meta + 16);
                        const float d = float(as_type<half>(ushort(dh)));
                        const uint pair = grouped_m8_scale_pair(scales, ushort(g % 8));
                        const float2 scale = float2(
                            d * float(as_type<char>(uchar(pair))),
                            d * float(as_type<char>(uchar(pair >> 8))));
                        for (uint half_index = 0; half_index < 2; ++half_index) {
                            float2 dot = *reinterpret_cast<device const float2 *>(
                                tile_sums + span * 32 + (2 * q + half_index) * 8 + fn);
                            for (uint f = 2 * half_index; f < 2 * half_index + 2; ++f) {
                                grouped_m8_register_mma(
                                    dot, grouped_m8_register_operand(codes[f]),
                                    activation_pairs[f]);
                            }
                            accumulator[nf] =
                                fma(dot, float2(scale[half_index]), accumulator[nf]);
                        }
                    } else {
                        const uint4 meta = *reinterpret_cast<device const uint4 *>(
                            expert_weights + p.meta_offset +
                            ((ulong(plane_tile) * units + g / 8) * 64 + column) * 16);
                        const float2 coefficient =
                            grouped_m8_q4_coef(meta, ushort(g % 8));
                        float2 dot = float2(0.0f);
                        for (uint f = 0; f < 4; ++f) {
                            grouped_m8_register_mma(
                                dot, grouped_m8_register_operand(low_codes[f]),
                                activation_pairs[f]);
                        }
                        const float2 sum32 =
                            *reinterpret_cast<device const float2 *>(
                                tile_sums + ulong(p.k) / 2 + span * 16 + q * 8 + fn);
                        accumulator[nf] =
                            fma(dot, float2(coefficient.x), accumulator[nf]);
                        accumulator[nf] =
                            fma(sum32, float2(coefficient.y - 128.0f * coefficient.x),
                                accumulator[nf]);
                    }
                }
            }
        }

        for (uint nf = 0; nf < 2; ++nf) {
            const uint column =
                group.x * 64 + uint(simdgroup) * 16 + nf * 8 + fm;
            for (uint i = 0; i < 2; ++i) {
                const uint row = fn + i;
                if (row >= live)
                    continue;
                const int packed = routes[first_route + route_base + row];
                const int64_t output_index = int64_t(packed) * p.n + column;
                const float value = accumulator[nf][i];
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
        }
    }
}

template <bool Q6>
kernel void grouped_native_m8_impl(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const uint32_t *schedule [[buffer(3)]],
    device const int32_t *routes [[buffer(4)]],
    constant GroupedMvParams &p [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint ROWS = 8;
    constexpr uint COLS = 32;
    constexpr uint STEP = 32;
    uint expert, first_route, route_count;
    if (!grouped_mv_descriptor(output, schedule, p, group, thread_index,
                               expert, first_route, route_count))
        return;
    const uint output_origin = group.x * 64 + uint(simdgroup) * COLS;
    device const uchar *expert_weights =
        weights + uint64_t(expert) * p.expert_stride;
    const uint blocks = uint(p.k) / QK_K;
    const uint groups = uint(p.k) / STEP;

    threadgroup half activation[ROWS * STEP];
    threadgroup half weight_tile[2 * COLS * STEP];
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

        auto left = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            activation, dextents<int32_t, 2>{STEP, ROWS},
            array<int32_t, 2>{1, STEP});
        threadgroup half *simd_weight = weight_tile + uint(simdgroup) * COLS * STEP;
        auto right = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            simd_weight, dextents<int32_t, 2>{STEP, COLS},
            array<int32_t, 2>{1, STEP});
        auto sum = mm.get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
        for (ushort i = 0; i < sum.get_capacity(); ++i)
            sum[i] = 0.0f;

        for (uint g = 0; g < groups; ++g) {
            for (uint index = thread_index; index < ROWS * STEP; index += 64) {
                const uint row = index / STEP;
                const uint column = index % STEP;
                const int packed = packed_routes[row];
                const int input_row = p.slotted_input ? packed : packed / p.top_k;
                activation[index] = packed < 0
                    ? half(0)
                    : half(input[int64_t(input_row) * p.k + g * STEP + column]);
            }

            const uint weight_row = output_origin + lane;
            if constexpr (Q6) {
                device const uchar *b = expert_weights +
                    uint64_t(weight_row * blocks + g / 8) * sizeof(block_q6_K);
                const uint within = g % 8;
                const uint high_block = within / 4;
                const uint quarter = within % 4;
                const float d = float(
                    *reinterpret_cast<device const half *>(b + 208));
                for (uint e = 0; e < STEP; ++e) {
                    const uint low =
                        (b[64 * high_block + 32 * (quarter & 1) + e] >>
                         (4 * (quarter >> 1))) & 15;
                    const uint high =
                        (b[128 + 32 * high_block + e] >> (2 * quarter)) & 3;
                    const char scale = as_type<char>(
                        b[192 + 2 * within + e / 16]);
                    simd_weight[lane * STEP + e] =
                        half(d * float(scale) * (float(low | (high << 4)) - 32.0f));
                }
            } else {
                device const uchar *b = expert_weights +
                    uint64_t(weight_row * blocks + g / 8) * sizeof(block_q4_K);
                const uint within = g % 8;
                const uint4 meta =
                    *reinterpret_cast<device const uint4 *>(b);
                const float2 coefficient =
                    grouped_m8_q4_coef(meta, ushort(within));
                for (uint e = 0; e < STEP; ++e) {
                    const uint code =
                        (b[16 + (within / 2) * 32 + e] >>
                         (4 * (within % 2))) & 15;
                    simd_weight[lane * STEP + e] =
                        half(fma(float(code), coefficient.x, coefficient.y));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mm.run(left, right, sum);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup float *simd_result =
            result_tile + uint(simdgroup) * ROWS * COLS;
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
            output[int64_t(packed_routes[row]) * p.n + group.x * 64 + column] =
                result_tile[source_simdgroup * ROWS * COLS +
                            row * COLS + source_column];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("grouped_native_m8_id_q4_K_f32")]]
kernel void grouped_native_m8_impl<false>(
    device const uchar *, device const float *, device float *,
    device const uint32_t *, device const int32_t *,
    constant GroupedMvParams &, uint2, ushort, ushort, ushort);

template [[host_name("grouped_native_m8_id_q6_K_f32")]]
kernel void grouped_native_m8_impl<true>(
    device const uchar *, device const float *, device float *,
    device const uint32_t *, device const int32_t *,
    constant GroupedMvParams &, uint2, ushort, ushort, ushort);

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

template [[host_name("grouped_m8_gather_table_f32")]]
kernel void grouped_m8_gather_table_impl<true>(
    device const uchar *, device bfloat *, device float *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, uint2, uint, uint);

template [[host_name("grouped_m8_gather_table_bf16")]]
kernel void grouped_m8_gather_table_impl<false>(
    device const uchar *, device bfloat *, device float *,
    device const uint32_t *, device const int32_t *,
    constant GroupedM8Params &, uint2, uint, uint);

template [[host_name("grouped_m8_register_q4_K_f32")]]
kernel void grouped_m8_register_impl<false, M8_EP_F32>(
    device const uchar *, device const bfloat *, device const float *, device uchar *,
    device const uint32_t *, device const int32_t *, constant GroupedM8Params &,
    device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_register_q6_K_f32")]]
kernel void grouped_m8_register_impl<true, M8_EP_F32>(
    device const uchar *, device const bfloat *, device const float *, device uchar *,
    device const uint32_t *, device const int32_t *, constant GroupedM8Params &,
    device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_register_q4_K_bf16")]]
kernel void grouped_m8_register_impl<false, M8_EP_BF16>(
    device const uchar *, device const bfloat *, device const float *, device uchar *,
    device const uint32_t *, device const int32_t *, constant GroupedM8Params &,
    device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_register_q6_K_bf16")]]
kernel void grouped_m8_register_impl<true, M8_EP_BF16>(
    device const uchar *, device const bfloat *, device const float *, device uchar *,
    device const uint32_t *, device const int32_t *, constant GroupedM8Params &,
    device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_register_q4_K_silu_mul_bf16")]]
kernel void grouped_m8_register_impl<false, M8_EP_SILU_MUL_BF16>(
    device const uchar *, device const bfloat *, device const float *, device uchar *,
    device const uint32_t *, device const int32_t *, constant GroupedM8Params &,
    device const bfloat *, uint2, ushort, ushort, ushort);

template [[host_name("grouped_m8_register_q6_K_silu_mul_bf16")]]
kernel void grouped_m8_register_impl<true, M8_EP_SILU_MUL_BF16>(
    device const uchar *, device const bfloat *, device const float *, device uchar *,
    device const uint32_t *, device const int32_t *, constant GroupedM8Params &,
    device const bfloat *, uint2, ushort, ushort, ushort);
