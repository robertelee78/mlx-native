// Expert-grouped Q4_K/Q6_K matvec tiles for decode-sized MoE batches.
//
// The compact mm_id map groups routed rows by expert into live descriptors.
// One threadgroup consumes one descriptor and keeps each expert weight row live
// while evaluating up to five routed activations at a time. Unlike the 32-row
// matrix tile, this path needs no activation staging and remains efficient when
// a decode step routes only a handful of rows to each expert.

#include <metal_stdlib>
using namespace metal;

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
