#include <metal_stdlib>
using namespace metal;

constant uint QK4_0 = 32;
constant uint VALUES_PER_THREAD = 16;

typedef struct {
    half d;
    uchar qs[QK4_0 / 2];
} block_q4_0;
static_assert(sizeof(block_q4_0) == 18, "wrong q4_0 block size");

struct EmbeddingQ4_0Params {
    uint vocab_size;
    uint embed_dim;
    uint blocks_per_row;
    uint n_tokens;
};

kernel void embedding_gather_q4_0_f32(
    device const block_q4_0 *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingQ4_0Params &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint chunk = gid.x;
    const uint token_index = gid.y;
    const uint chunks_per_row = p.embed_dim / VALUES_PER_THREAD;
    if (chunk >= chunks_per_row || token_index >= p.n_tokens) {
        return;
    }
    const uint token = token_ids[token_index];
    if (token >= p.vocab_size) {
        const uint output_base = token_index * p.embed_dim + chunk * VALUES_PER_THREAD;
        for (uint lane = 0; lane < VALUES_PER_THREAD; ++lane) {
            output[output_base + lane] = 0.0f;
        }
        return;
    }

    const uint column_base = chunk * VALUES_PER_THREAD;
    const uint block_index = column_base / QK4_0;
    const bool high_nibble = (column_base % QK4_0) >= VALUES_PER_THREAD;
    device const block_q4_0 &block =
        weights[token * p.blocks_per_row + block_index];
    const float scale = float(block.d);
    const uint output_base = token_index * p.embed_dim + column_base;
    for (uint lane = 0; lane < VALUES_PER_THREAD; ++lane) {
        const uchar packed_quant = block.qs[lane];
        const uchar quant = high_nibble ? packed_quant >> 4 : packed_quant & 0x0f;
        output[output_base + lane] = scale * (float(quant) - 8.0f);
    }
}
