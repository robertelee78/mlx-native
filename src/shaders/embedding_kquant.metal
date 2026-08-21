#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint VALUES_PER_THREAD = 16;

typedef struct {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K / 8];
    uchar qs[QK_K / 2];
} block_q5_K;
static_assert(sizeof(block_q5_K) == 176, "wrong q5_K block size");

typedef struct {
    uchar ql[QK_K / 2];
    uchar qh[QK_K / 4];
    char scales[QK_K / 16];
    half d;
} block_q6_K;
static_assert(sizeof(block_q6_K) == 210, "wrong q6_K block size");

struct EmbeddingKQuantParams {
    uint vocab_size;
    uint embed_dim;
    uint blocks_per_row;
    uint n_tokens;
};

static inline uchar2 q5_k_scale_min(device const uchar *scales, uint group) {
    if (group < 4) {
        return uchar2(scales[group] & 0x3f, scales[group + 4] & 0x3f);
    }
    return uchar2(
        (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
        (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4));
}

kernel void embedding_gather_q5_k_f32(
    device const block_q5_K *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingKQuantParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint chunk = gid.x;
    const uint token_index = gid.y;
    const uint chunks_per_row = p.embed_dim / VALUES_PER_THREAD;
    if (chunk >= chunks_per_row || token_index >= p.n_tokens) return;
    const uint output_base = token_index * p.embed_dim + chunk * VALUES_PER_THREAD;
    const uint token = token_ids[token_index];
    if (token >= p.vocab_size) {
        for (uint lane = 0; lane < VALUES_PER_THREAD; ++lane) output[output_base + lane] = 0.0f;
        return;
    }

    const uint column_base = chunk * VALUES_PER_THREAD;
    const uint in_block = column_base % QK_K;
    const uint group = in_block / 32;
    const uint lane_base = in_block % 32;
    device const block_q5_K &block = weights[token * p.blocks_per_row + column_base / QK_K];
    const uchar2 scale_min = q5_k_scale_min(block.scales, group);
    const float scale = float(block.d) * float(scale_min.x);
    const float minimum = float(block.dmin) * float(scale_min.y);
    const uint quant_base = (group / 2) * 32 + lane_base;
    const uchar high_mask = uchar(1u << group);
    for (uint lane = 0; lane < VALUES_PER_THREAD; ++lane) {
        const uchar packed_quant = block.qs[quant_base + lane];
        const uchar low = group & 1 ? packed_quant >> 4 : packed_quant & 0x0f;
        const uchar high = block.qh[lane_base + lane] & high_mask ? 16 : 0;
        output[output_base + lane] = scale * float(low + high) - minimum;
    }
}

kernel void embedding_gather_q6_k_f32(
    device const block_q6_K *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant EmbeddingKQuantParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint chunk = gid.x;
    const uint token_index = gid.y;
    const uint chunks_per_row = p.embed_dim / VALUES_PER_THREAD;
    if (chunk >= chunks_per_row || token_index >= p.n_tokens) return;
    const uint output_base = token_index * p.embed_dim + chunk * VALUES_PER_THREAD;
    const uint token = token_ids[token_index];
    if (token >= p.vocab_size) {
        for (uint lane = 0; lane < VALUES_PER_THREAD; ++lane) output[output_base + lane] = 0.0f;
        return;
    }

    const uint column_base = chunk * VALUES_PER_THREAD;
    device const block_q6_K &block = weights[token * p.blocks_per_row + column_base / QK_K];
    for (uint lane = 0; lane < VALUES_PER_THREAD; ++lane) {
        const uint in_block = (column_base + lane) % QK_K;
        const uint half_index = in_block / 128;
        const uint position = in_block % 128;
        const uint segment = position / 32;
        const uint l = position % 32;
        const uint scale_pair = l / 16;
        const uint ql_base = half_index * 64;
        const uint qh_base = half_index * 32;
        const uint scale_base = half_index * 8;
        const uchar high_bits = block.qh[qh_base + l];
        uchar quant_bits;
        char sub_scale;
        if (segment == 0) {
            quant_bits = (block.ql[ql_base + l] & 0x0f) | ((high_bits & 0x03) << 4);
            sub_scale = block.scales[scale_base + scale_pair];
        } else if (segment == 1) {
            quant_bits = (block.ql[ql_base + l + 32] & 0x0f) | (((high_bits >> 2) & 0x03) << 4);
            sub_scale = block.scales[scale_base + scale_pair + 2];
        } else if (segment == 2) {
            quant_bits = (block.ql[ql_base + l] >> 4) | (((high_bits >> 4) & 0x03) << 4);
            sub_scale = block.scales[scale_base + scale_pair + 4];
        } else {
            quant_bits = (block.ql[ql_base + l + 32] >> 4) | (((high_bits >> 6) & 0x03) << 4);
            sub_scale = block.scales[scale_base + scale_pair + 6];
        }
        const int quant = int(quant_bits) - 32;
        const float scale = float(block.d) * float(sub_scale);
        output[output_base + lane] = scale * float(quant);
    }
}
