// Lossless canonical addressing and shared original legacy F16 fragment math.
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

// Both storage layouts feed the same F16 arithmetic, including high-nibble
// operands, widened d/16, Q6 casts, and byte-position-dependent scales.
template<typename Block>
struct RawF16Fragment {
    device const Block * x;
    ushort q4(short il, uint i) const {
        const uint offset = (il / 4) * 32 + 16 * (il & 1) + i;
        return x->qs[offset] & ((il & 3) < 2 ? 0x0F : 0xF0);
    }
    uint q6(short il, uint i) const {
        device const ushort * ql = reinterpret_cast<device const ushort *>(x->ql)
            + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
        device const ushort * qh = reinterpret_cast<device const ushort *>(x->qh)
            + 16 * (il / 8) + 8 * (il & 1);
        const short sub = (il / 2) & 3;
        const uint mask1 = sub > 1 ? (sub > 2 ? 0xC0C0C0C0 : 0x30303030)
                                  : (sub > 0 ? 0x0C0C0C0C : 0x03030303);
        const uint mask2 = sub > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
        const uint low = (ql[2*i] | uint(ql[2*i+1] << 16)) & mask2;
        const uint high = (qh[2*i] | uint(qh[2*i+1] << 16)) & mask1;
        return ((high << (sub > 1 ? 0 : (sub > 0 ? 2 : 4))) >> (sub > 2 ? 2 : 0))
            | (low >> (sub > 1 ? 4 : 0));
    }
};

struct PreparedF16Fragment {
    device const uchar * expert;
    constant PreparedF16TilesParams * p;
    uint row;
    uint block;
    ushort q4(short il, uint i) const {
        const uint group = block * 8 + (il / 4) * 2 + ((il & 3) / 2);
        return prepared_f16_low(expert, row, group, p->blocks * 8, 16 * (il & 1) + i)
            << ((il & 3) < 2 ? 0 : 4);
    }
    uint q6(short il, uint i) const {
        uint q = 0;
        for (uint j = 0; j < 4; ++j)
            q |= prepared_f16_q6_code(expert, *p, row, block * 8 + uint(il / 2),
                                     16 * (il & 1) + 4 * i + j) << (8 * j);
        return q;
    }
};

template<typename Reader, typename Tile>
static inline void legacy_f16_q4_fragment(
    half d_all, half dmin, device const uchar * scales,
    thread const Reader & reader, short il, thread Tile & reg) {
    const short is = (il / 4) * 2;
    const short sub = il & 3;
    const short k = sub / 2;
    const uchar2 sc = is < 4
        ? uchar2{uchar(scales[is + k] & 63), uchar(scales[is + 4 + k] & 63)}
        : uchar2{uchar((scales[is + 4 + k] & 0xF) | ((scales[is - 4 + k] & 0xc0) >> 2)),
                 uchar((scales[is + 4 + k] >> 4) | ((scales[is + k] & 0xc0) >> 2))};
    const float d = sub < 2 ? float(d_all) : float(d_all) / 16.f;
    const float min = dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    for (int i = 0; i < 16; ++i)
        reg[i / 4][i % 4] = dl * reader.q4(il, i) - ml;
}

template<typename Reader, typename Tile>
static inline void legacy_f16_q6_fragment(
    half d_all, device const signed char * scales,
    thread const Reader & reader, short il, thread Tile & reg) {
    const float sc = scales[il];
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        const uint q = reader.q6(il, i);
        reg_f[i][0] = dl0 * ((half)(q & 0xFF))        - ml;
        reg_f[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg_f[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg_f[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
    reg = (Tile)reg_f;
}

static inline void prepared_f16_q4_fragment(
    device const uchar *expert, constant PreparedF16TilesParams &p,
    uint row, uint block, short il, thread half4x4 &reg) {
    device const uchar *meta = expert + p.meta_offset +
        prepared_f16_tiled_index(row, block, p.blocks) * 16;
    const PreparedF16Fragment reader = {expert, &p, row, block};
    legacy_f16_q4_fragment(*reinterpret_cast<device const half *>(meta),
        *reinterpret_cast<device const half *>(meta + 2), meta + 4, reader, il, reg);
}

static inline void prepared_f16_q6_fragment(
    device const uchar *expert, constant PreparedF16TilesParams &p,
    uint row, uint block, short il, thread half4x4 &reg) {
    device const uchar *meta = expert + p.meta_offset +
        prepared_f16_tiled_index(row, block, p.blocks) * 20;
    const PreparedF16Fragment reader = {expert, &p, row, block};
    legacy_f16_q6_fragment(*reinterpret_cast<device const half *>(meta + 16),
        reinterpret_cast<device const signed char *>(meta), reader, il, reg);
}

static inline ulong prepared_f16_fragment_offset(
    constant PreparedF16TilesParams &p, uint3 index) {
    return ((ulong(index.z) * p.n + index.y) * p.blocks * 16 + index.x) * 16;
}
#endif
