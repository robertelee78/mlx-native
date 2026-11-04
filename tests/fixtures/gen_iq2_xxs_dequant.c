// Developer-only generator for tests/fixtures/iq2_xxs_dequant_v1.bin
// (ADR-066 N-IQ2XXS bit-exact CPU dequant fixture).
//
// It links only the pinned peer checkout's `dequantize_row_iq2_xxs`
// (ggml/src/ggml-quants.c) and the `ggml-common.h` tables that file
// instantiates. Neither the generator nor peer code is part of the
// mlx-native build or test run; `tests/iq2_xxs_dequant.rs` reads the
// checked-in fixture.
//
// Build and run (P = peer checkout root):
//
//   cc -O2 -std=gnu11 -I"$P/ggml/include" -I"$P/ggml/src" \
//       -I"$P/ggml/src/ggml-cpu" -c "$P/ggml/src/ggml-quants.c" -o quants.o
//   cc -O2 -std=gnu11 -I"$P/ggml/include" -I"$P/ggml/src" \
//       tests/fixtures/gen_iq2_xxs_dequant.c quants.o -Wl,-dead_strip \
//       -o gen_iq2_xxs_dequant
//   ./gen_iq2_xxs_dequant tests/fixtures/iq2_xxs_dequant_v1.bin \
//       <peer-commit> \
//       "$(shasum -a 256 "$P/ggml/src/ggml-common.h" | cut -d' ' -f1)" \
//       "$(shasum -a 256 "$P/ggml/src/ggml-quants.c" | cut -d' ' -f1)"
//
// `-Wl,-dead_strip` drops the quantizers in ggml-quants.c that reference
// other ggml-base symbols; only the dequantizer and its tables remain.
//
// Fixture layout (little-endian):
//   magic   "IQ2XXSD1"                      8 bytes
//   u32     header length H
//   header  H bytes of UTF-8 `key=value\n` lines:
//           peer_commit, ggml_common_h_sha256, ggml_quants_c_sha256,
//           blocks, random_blocks, seed
//   u32     block count B
//   B * 66  input blocks
//   B * 256 expected f32 bit patterns (u32 each)

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dequantize_row_iq2_xxs(const block_iq2_xxs * x, float * y, int64_t k);

#define RANDOM_BLOCKS 4096
#define SEED 0x1d2a66ull

static uint64_t state = SEED;

static uint64_t next_u64(void) {
    // splitmix64
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static void set_sub_block(block_iq2_xxs * b, int ib32, const uint8_t grid[4],
                          const uint8_t signs[4], uint32_t scale) {
    uint32_t a0 = (uint32_t)grid[0] | ((uint32_t)grid[1] << 8) |
                  ((uint32_t)grid[2] << 16) | ((uint32_t)grid[3] << 24);
    uint32_t a1 = (scale & 15u) << 28;
    for (int l = 0; l < 4; ++l) a1 |= ((uint32_t)signs[l] & 127u) << (7 * l);
    memcpy(b->qs + 4 * ib32, &a0, 4);
    memcpy(b->qs + 4 * ib32 + 2, &a1, 4);
}

static void random_qs(block_iq2_xxs * b) {
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint64_t r = next_u64();
        uint8_t grid[4], signs[4];
        for (int l = 0; l < 4; ++l) {
            grid[l] = (uint8_t)(r >> (8 * l));
            signs[l] = (uint8_t)(r >> (32 + 7 * l)) & 127;
        }
        set_sub_block(b, ib32, grid, signs, (uint32_t)(r >> 60));
    }
}

// A random finite, normal f16 scale with a random sign.
static uint16_t random_normal_f16(void) {
    uint64_t r = next_u64();
    uint16_t sign = (uint16_t)((r & 1) << 15);
    uint16_t exponent = (uint16_t)(1 + (r >> 1) % 30);
    uint16_t mantissa = (uint16_t)((r >> 8) & 0x3ff);
    return sign | (uint16_t)(exponent << 10) | mantissa;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <out.bin> <peer-commit> <common-sha256> <quants-sha256>\n",
                argv[0]);
        return 2;
    }

    static const uint16_t edge_d[] = {
        0x0000, // +0
        0x8000, // -0
        0x0001, // smallest positive subnormal
        0x8001, // smallest negative subnormal
        0x3c00, // 1.0
        0x7bff, // largest finite
        0xfbff, // most negative finite
        0x7c00, // +inf
        0xfc00, // -inf
        0x7e00, // quiet NaN
        0xfe00, // negative quiet NaN
        0x7c01, // signalling NaN payload
    };
    const int n_edge_d = (int)(sizeof(edge_d) / sizeof(edge_d[0]));
    const int n_scale = 16;   // one block per scale nibble
    const int n_cover = 32;   // grid/sign index coverage at every l
    const int n_blocks = n_scale + n_cover + n_edge_d + RANDOM_BLOCKS;

    block_iq2_xxs * blocks = calloc((size_t)n_blocks, sizeof(block_iq2_xxs));
    float * values = malloc((size_t)n_blocks * QK_K * sizeof(float));
    if (!blocks || !values) return 1;

    int index = 0;
    // Every scale nibble 0..15, all eight sub-blocks.
    for (int s = 0; s < n_scale; ++s, ++index) {
        blocks[index].d = 0x3c00;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint64_t r = next_u64();
            uint8_t grid[4], signs[4];
            for (int l = 0; l < 4; ++l) {
                grid[l] = (uint8_t)(r >> (8 * l));
                signs[l] = (uint8_t)(r >> (32 + 7 * l)) & 127;
            }
            set_sub_block(&blocks[index], ib32, grid, signs, (uint32_t)s);
        }
    }
    // For each l, (8 * b + ib32) covers every grid index 0..255 and every
    // sign index 0..127 across the 32 coverage blocks.
    for (int b = 0; b < n_cover; ++b, ++index) {
        blocks[index].d = 0x3800; // 0.5
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint8_t grid[4], signs[4];
            for (int l = 0; l < 4; ++l) {
                grid[l] = (uint8_t)((8 * b + ib32 + 37 * l) & 255);
                signs[l] = (uint8_t)((8 * b + ib32 + 19 * l) & 127);
            }
            set_sub_block(&blocks[index], ib32, grid, signs, (uint32_t)((b + ib32) & 15));
        }
    }
    for (int e = 0; e < n_edge_d; ++e, ++index) {
        blocks[index].d = edge_d[e];
        random_qs(&blocks[index]);
    }
    for (int r = 0; r < RANDOM_BLOCKS; ++r, ++index) {
        blocks[index].d = random_normal_f16();
        random_qs(&blocks[index]);
    }

    dequantize_row_iq2_xxs(blocks, values, (int64_t)n_blocks * QK_K);

    char header[1024];
    int header_len = snprintf(header, sizeof(header),
                              "peer_commit=%s\n"
                              "ggml_common_h_sha256=%s\n"
                              "ggml_quants_c_sha256=%s\n"
                              "blocks=%d\n"
                              "random_blocks=%d\n"
                              "seed=0x%llx\n",
                              argv[2], argv[3], argv[4], n_blocks, RANDOM_BLOCKS,
                              (unsigned long long)SEED);
    if (header_len <= 0 || header_len >= (int)sizeof(header)) return 1;

    FILE * out = fopen(argv[1], "wb");
    if (!out) return 1;
    const uint32_t h = (uint32_t)header_len;
    const uint32_t count = (uint32_t)n_blocks;
    fwrite("IQ2XXSD1", 1, 8, out);
    fwrite(&h, 4, 1, out);
    fwrite(header, 1, (size_t)header_len, out);
    fwrite(&count, 4, 1, out);
    fwrite(blocks, sizeof(block_iq2_xxs), (size_t)n_blocks, out);
    fwrite(values, sizeof(float), (size_t)n_blocks * QK_K, out);
    fclose(out);
    fprintf(stderr, "wrote %d blocks to %s\n", n_blocks, argv[1]);
    return 0;
}
