// Integer-only canonical expert packing; independent of tensor arithmetic.
#include <metal_stdlib>
using namespace metal;
#ifndef HF2Q_PREPARED_EXPERT_LAYOUT
#include "prepared_expert_layout.h"
#endif

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

