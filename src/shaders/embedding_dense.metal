#include <metal_stdlib>
using namespace metal;

struct EmbeddingDenseParams {
    uint vocab_size;
    uint embed_dim;
    uint n_tokens;
    uint reserved;
};

template <typename T>
inline void embedding_gather_dense_impl(
    device const T *weights,
    device const uint *token_ids,
    device float *output,
    constant EmbeddingDenseParams &p,
    uint2 gid) {
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
    output[token_index * p.embed_dim + column] =
        float(weights[token * p.embed_dim + column]);
}

kernel void embedding_gather_bf16_f32(
    device const bfloat *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingDenseParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    embedding_gather_dense_impl(weights, token_ids, output, p, gid);
}

kernel void embedding_gather_f16_f32(
    device const half *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingDenseParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    embedding_gather_dense_impl(weights, token_ids, output, p, gid);
}

kernel void embedding_gather_f32_f32(
    device const float *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingDenseParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    embedding_gather_dense_impl(weights, token_ids, output, p, gid);
}
