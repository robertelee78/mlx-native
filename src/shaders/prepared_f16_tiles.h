// Unpublished decoder-tile probe: lossless prepared addressing, legacy F16 math.
#ifndef MLX_NATIVE_PREPARED_F16_TILES_H
#define MLX_NATIVE_PREPARED_F16_TILES_H
#include <metal_stdlib>
using namespace metal;

struct PreparedF16TilesParams {
    uint n_experts;
    uint n;
    uint blocks;
    uint format;
    ulong raw_expert_stride;
    ulong prepared_expert_stride;
    uint plane1_offset;
    uint meta_offset;
};

static inline ulong prepared_f16_tiled_index(uint row, uint unit, uint units) {
    return (ulong(row / 64) * units + unit) * 64 + row % 64;
}

static inline uint prepared_f16_slot(uint e) {
    return 8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3);
}

static inline uint prepared_f16_low(
    device const uchar *expert, uint row, uint group, uint groups, uint e) {
    const uint slot = prepared_f16_slot(e);
    device const uint *words = reinterpret_cast<device const uint *>(
        expert + prepared_f16_tiled_index(row, group, groups) * 16);
    return (words[slot / 8] >> (16 * (slot % 2) + 4 * ((slot % 8) / 2))) & 15;
}

static inline uint prepared_f16_q6_code(
    device const uchar *expert, constant PreparedF16TilesParams &p,
    uint row, uint group, uint e) {
    const uint slot = prepared_f16_slot(e);
    device const uint *words = reinterpret_cast<device const uint *>(
        expert + p.plane1_offset + prepared_f16_tiled_index(row, group, p.blocks * 8) * 8);
    const uint high = (words[slot / 16] >> (2 * (slot % 16))) & 3;
    return prepared_f16_low(expert, row, group, p.blocks * 8, e) | (high << 4);
}

static inline void prepared_f16_q4_fragment(
    device const uchar *expert, constant PreparedF16TilesParams &p,
    uint row, uint block, short il, thread half4x4 &reg) {
    device const uchar *meta = expert + p.meta_offset +
        prepared_f16_tiled_index(row, block, p.blocks) * 16;
    device const uchar *scales = meta + 4;
    const short is = (il / 4) * 2;
    const short sub = il & 3;
    const short k = sub / 2;
    const uchar2 sc = is < 4
        ? uchar2{uchar(scales[is + k] & 63), uchar(scales[is + 4 + k] & 63)}
        : uchar2{uchar((scales[is + 4 + k] & 0xF) | ((scales[is - 4 + k] & 0xc0) >> 2)),
                 uchar((scales[is + 4 + k] >> 4) | ((scales[is + k] & 0xc0) >> 2))};
    const half d_all = *reinterpret_cast<device const half *>(meta);
    const half dmin = *reinterpret_cast<device const half *>(meta + 2);
    // Keep the original high-nibble operand and the F32 division by 16.
    const float d = sub < 2 ? float(d_all) : float(d_all) / 16.f;
    const float min = dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const uint group = block * 8 + uint(is + k);
    for (int i = 0; i < 16; ++i) {
        const ushort q = prepared_f16_low(expert, row, group, p.blocks * 8,
                                          16 * (il & 1) + uint(i)) << (sub < 2 ? 0 : 4);
        reg[i / 4][i % 4] = dl * q - ml;
    }
}

static inline void prepared_f16_q6_fragment(
    device const uchar *expert, constant PreparedF16TilesParams &p,
    uint row, uint block, short il, thread half4x4 &reg) {
    device const uchar *meta = expert + p.meta_offset +
        prepared_f16_tiled_index(row, block, p.blocks) * 20;
    const half d_all = *reinterpret_cast<device const half *>(meta + 16);
    const float sc = reinterpret_cast<device const char *>(meta)[il];
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint group = block * 8 + uint(il / 2);
    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        // This is the original packed, unsigned q operand, not scalar dequantization.
        uint q = 0;
        for (uint j = 0; j < 4; ++j)
            q |= prepared_f16_q6_code(expert, p, row, group,
                                     16 * (il & 1) + 4 * uint(i) + j) << (8 * j);
        reg_f[i][0] = dl0 * ((half)(q & 0xFF))       - ml;
        reg_f[i][1] = dl1 * ((float)(q & 0xFF00))    - ml;
        reg_f[i][2] = dl2 * ((float)(q & 0xFF0000))  - ml;
        reg_f[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
    reg = (half4x4)reg_f;
}

static inline ulong prepared_f16_fragment_offset(
    constant PreparedF16TilesParams &p, uint3 index) {
    return ((ulong(index.z) * p.n + index.y) * p.blocks * 16 + index.x) * 16;
}
#endif
