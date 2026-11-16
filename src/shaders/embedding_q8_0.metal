#include <metal_stdlib>
using namespace metal;

constant uint QK8_0 = 32;

typedef struct {
    half d;
    char qs[QK8_0];
} block_q8_0;
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size");

struct EmbeddingQ8_0Params {
    uint vocab_size;
    uint embed_dim;
    uint blocks_per_row;
    uint n_tokens;
};

template <bool BF16_BOUNDARY>
kernel void embedding_gather_q8_0_impl(
    device const block_q8_0 *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingQ8_0Params &p [[buffer(3)]],
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

    device const block_q8_0 &block =
        weights[token * p.blocks_per_row + column / QK8_0];
    const float value = float(block.d) * float(block.qs[column % QK8_0]);
    output[token_index * p.embed_dim + column] =
        BF16_BOUNDARY ? float(bfloat(value)) : value;
}

template [[host_name("embedding_gather_q8_0_f32")]]
kernel void embedding_gather_q8_0_impl<false>(
    device const block_q8_0 *, device const uint *, device float *,
    constant EmbeddingQ8_0Params &, uint2);
template [[host_name("embedding_gather_q8_0_f32_bf16_boundary")]]
kernel void embedding_gather_q8_0_impl<true>(
    device const block_q8_0 *, device const uint *, device float *,
    constant EmbeddingQ8_0Params &, uint2);
