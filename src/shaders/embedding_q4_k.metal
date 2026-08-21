#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint VALUES_PER_THREAD = 16;

typedef struct {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K / 2];
} block_q4_K;
static_assert(sizeof(block_q4_K) == 144, "wrong q4_K block size");

struct EmbeddingQ4KParams {
    uint vocab_size;
    uint embed_dim;
    uint blocks_per_row;
    uint n_tokens;
};

static inline uchar2 q4_k_scale_min(
    device const uchar *scales,
    uint group) {
    if (group < 4) {
        return uchar2(scales[group] & 0x3f, scales[group + 4] & 0x3f);
    }
    return uchar2(
        (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
        (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4));
}

kernel void embedding_gather_q4_k_f32(
    device const block_q4_K *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingQ4KParams &p [[buffer(3)]],
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
    const uint in_block = column_base % QK_K;
    const uint group = in_block / 32;
    const uint lane_base = in_block % 32;
    device const block_q4_K &block =
        weights[token * p.blocks_per_row + column_base / QK_K];
    const uchar2 scale_min = q4_k_scale_min(block.scales, group);
    const float scale = float(block.d) * float(scale_min.x);
    const float minimum = float(block.dmin) * float(scale_min.y);
    const uint quant_base = (group / 2) * 32 + lane_base;
    const uint output_base = token_index * p.embed_dim + column_base;
    for (uint lane = 0; lane < VALUES_PER_THREAD; ++lane) {
        const uchar packed_quant = block.qs[quant_base + lane];
        const uchar quant = group & 1 ? packed_quant >> 4 : packed_quant & 0x0f;
        output[output_base + lane] = scale * float(quant) - minimum;
    }
}
