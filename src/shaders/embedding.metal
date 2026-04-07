#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// embedding_gather_4bit
//
// Quantized embedding table lookup for 4-bit packed weights.
//
// The embedding table has shape [vocab_size, embed_dim].  Weights are stored
// as packed uint32 values: 8 x 4-bit entries per uint32.
//
// Buffers:
//   0: weight_packed  — packed uint32 embedding table [vocab_size, packed_dim]
//   1: scales         — bf16 scales, [vocab_size, n_groups]
//   2: biases         — bf16 biases, [vocab_size, n_groups]
//   3: token_ids      — uint32 token IDs, [n_tokens]
//   4: output         — float output, [n_tokens, embed_dim]
//   5: params         — { embed_dim: uint32, group_size: uint32, packed_row_stride: uint32 }
//
// Grid: (embed_dim, n_tokens, 1)
// --------------------------------------------------------------------------

struct EmbeddingParams {
    uint embed_dim;
    uint group_size;
    uint packed_row_stride;   // number of uint32 values per row in packed table
    uint n_groups_per_row;    // ceil(embed_dim / group_size)
};

kernel void embedding_gather_4bit(
    device const uint32_t* weight_packed  [[buffer(0)]],
    device const uint16_t* scales         [[buffer(1)]],
    device const uint16_t* biases         [[buffer(2)]],
    device const uint32_t* token_ids      [[buffer(3)]],
    device float*          output         [[buffer(4)]],
    constant EmbeddingParams& params      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;  // which element in embed_dim
    uint row = gid.y;  // which token

    if (col >= params.embed_dim) return;

    uint token_id = token_ids[row];

    // Locate the packed data for this row
    device const uint32_t* row_data = weight_packed + token_id * params.packed_row_stride;

    // 4-bit: 8 values per uint32
    uint word_idx = col / 8;
    uint bit_idx  = col % 8;
    uint32_t word = row_data[word_idx];
    uint32_t uint_val = (word >> (bit_idx * 4)) & 0xF;

    // Get scale and bias for this element's group
    uint group_idx = col / params.group_size;
    uint scale_offset = token_id * params.n_groups_per_row + group_idx;

    // scales and biases are stored as bf16 — reinterpret as bfloat
    // metal has bfloat type on Apple Silicon
    float scale = static_cast<float>(as_type<bfloat>(scales[scale_offset]));
    float bias  = static_cast<float>(as_type<bfloat>(biases[scale_offset]));

    float dequant = static_cast<float>(uint_val) * scale + bias;

    output[row * params.embed_dim + col] = dequant;
}

// --------------------------------------------------------------------------
// embedding_gather_6bit
//
// 6-bit packing: 4 values per 3 bytes (24 bits).
// The packed data is stored as uint8 triplets.  But in the Metal buffer it
// is reinterpreted from the uint32 safetensors storage.
//
// For element `i` within a row:
//   pack_index = i / 4           (which 3-byte triplet)
//   sub_index  = i % 4           (which value within the triplet)
//   byte_offset = pack_index * 3 (byte offset into the row's packed data)
//   pack = byte0 | (byte1 << 8) | (byte2 << 16)
//   val = (pack >> (sub_index * 6)) & 0x3F
// --------------------------------------------------------------------------

kernel void embedding_gather_6bit(
    device const uint8_t*  weight_packed  [[buffer(0)]],
    device const uint16_t* scales         [[buffer(1)]],
    device const uint16_t* biases         [[buffer(2)]],
    device const uint32_t* token_ids      [[buffer(3)]],
    device float*          output         [[buffer(4)]],
    constant EmbeddingParams& params      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;  // which element in embed_dim
    uint row = gid.y;  // which token

    if (col >= params.embed_dim) return;

    uint token_id = token_ids[row];

    // For 6-bit, packed_row_stride is in bytes (number of bytes per row)
    // packed_row_stride = embed_dim * 3 / 4 bytes (since 4 values per 3 bytes)
    device const uint8_t* row_data = weight_packed + token_id * params.packed_row_stride;

    // 6-bit: 4 values per 3-byte triplet
    uint pack_index = col / 4;
    uint sub_index  = col % 4;
    uint byte_offset = pack_index * 3;

    uint32_t b0 = row_data[byte_offset];
    uint32_t b1 = row_data[byte_offset + 1];
    uint32_t b2 = row_data[byte_offset + 2];
    uint32_t pack = b0 | (b1 << 8) | (b2 << 16);

    uint32_t uint_val = (pack >> (sub_index * 6)) & 0x3F;

    // Get scale and bias for this element's group
    uint group_idx = col / params.group_size;
    uint scale_offset = token_id * params.n_groups_per_row + group_idx;

    float scale = static_cast<float>(as_type<bfloat>(scales[scale_offset]));
    float bias  = static_cast<float>(as_type<bfloat>(biases[scale_offset]));

    float dequant = static_cast<float>(uint_val) * scale + bias;

    output[row * params.embed_dim + col] = dequant;
}
