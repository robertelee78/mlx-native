#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;

typedef struct {
    uchar scales[16];
    uchar qs[64];
    half d;
    half dmin;
} block_q2_K;
static_assert(sizeof(block_q2_K) == 84, "wrong q2_K block size");

struct EmbeddingQ2KParams {
    uint vocab_size;
    uint embed_dim;
    uint blocks_per_row;
    uint n_tokens;
};

kernel void embedding_gather_q2_k_f32(
    device const block_q2_K *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingQ2KParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint column = gid.x;
    const uint token_index = gid.y;
    if (column >= p.embed_dim || token_index >= p.n_tokens) {
        return;
    }
    const uint token = token_ids[token_index];
    if (token >= p.vocab_size) {
        output[token_index * p.embed_dim + column] = 0.0f;
        return;
    }

    const uint in_block = column % QK_K;
    const uint group = in_block / 16;
    const uint lane = in_block % 16;
    const uint half_index = group / 8;
    const uint group_in_half = group % 8;
    const uint shift = 2 * (group_in_half / 2);
    const uint q_offset = half_index * 32 + (group_in_half % 2) * 16 + lane;
    device const block_q2_K &block =
        weights[token * p.blocks_per_row + column / QK_K];
    const uchar packed_scale = block.scales[group];
    const uint quant = (block.qs[q_offset] >> shift) & 0x03;
    output[token_index * p.embed_dim + column] =
        float(block.d) * float(packed_scale & 0x0f) * float(quant)
        - float(block.dmin) * float(packed_scale >> 4);
}
