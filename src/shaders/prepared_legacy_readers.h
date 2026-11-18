// Storage-only readers. No dequantization, reduction, or temporary raw blocks.
#ifndef MLX_NATIVE_PREPARED_LEGACY_READERS_H
#define MLX_NATIVE_PREPARED_LEGACY_READERS_H
#ifndef MLX_NATIVE_PREPARED_F16_TILES_H
#include "prepared_f16_tiles.h"
#endif

template<bool PREPARED, bool Q6>
struct LegacyExpertRow {
    device const char * bytes;
    uint expert;
    uint row;
    uint blocks;
    ulong raw_stride;
    constant PreparedF16TilesParams * planes;

    LegacyExpertRow with_row(uint next) const {
        return {bytes, expert, next, blocks, raw_stride, planes};
    }
    device const uchar * raw_block(uint block) const {
        return reinterpret_cast<device const uchar *>(bytes) + ulong(expert) * raw_stride
            + (ulong(row) * blocks + block) * (Q6 ? 210 : 144);
    }
    device const uchar * prepared_expert() const {
        return reinterpret_cast<device const uchar *>(bytes)
            + ulong(expert) * planes->prepared_expert_stride;
    }
    device const uchar * metadata(uint block) const {
        if (PREPARED) return prepared_expert() + planes->meta_offset
            + prepared_f16_tiled_index(row, block, blocks) * (Q6 ? 20 : 16);
        return raw_block(block) + (Q6 ? 192 : 0);
    }
    device const half * dh(uint block) const {
        return reinterpret_cast<device const half *>(metadata(block) + (Q6 ? 16 : 0));
    }
    device const uchar * scales(uint block) const {
        return metadata(block) + (Q6 ? 0 : 4);
    }
    uint code(uint block, uint group, uint e) const {
        return prepared_f16_q6_code(prepared_expert(), *planes, row, block * 8 + group, e);
    }
    uchar q4_byte(uint block, uint offset) const {
        if (!PREPARED) return raw_block(block)[16 + offset];
        const uint group = block * 8 + 2 * (offset / 32);
        device const uchar * base = prepared_expert();
        return uchar(prepared_f16_low(base, row, group, blocks * 8, offset % 32)
            | (prepared_f16_low(base, row, group + 1, blocks * 8, offset % 32) << 4));
    }
    ushort q4_word(uint block, uint offset) const {
        if (!PREPARED)
            return reinterpret_cast<device const ushort *>(raw_block(block) + 16)[offset];
        return ushort(q4_byte(block, 2 * offset)) | (ushort(q4_byte(block, 2 * offset + 1)) << 8);
    }
    uchar q6_low(uint block, uint offset) const {
        if (!PREPARED) return raw_block(block)[offset];
        const uint group = 4 * (offset / 64) + (offset % 64) / 32;
        const uint e = offset % 32;
        return uchar((code(block, group, e) & 15) | ((code(block, group + 2, e) & 15) << 4));
    }
    uchar q6_high(uint block, uint offset) const {
        if (!PREPARED) return raw_block(block)[128 + offset];
        const uint group = 4 * (offset / 32);
        const uint e = offset % 32;
        return uchar((code(block, group, e) >> 4)
            | ((code(block, group + 1, e) >> 4) << 2)
            | ((code(block, group + 2, e) >> 4) << 4)
            | ((code(block, group + 3, e) >> 4) << 6));
    }
};
#endif
