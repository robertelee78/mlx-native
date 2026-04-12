// Portions of this file are derived from candle-metal-kernels v0.10.2
// (https://github.com/huggingface/candle), Apache-2.0 licensed.
// Source: candle-metal-kernels/src/metal_src/quantized.metal:7544-7618
// Modifications: ported to mlx-native's dispatch path; argument-passing
// adapted to mlx-native's encoder API; threadgroup geometry preserved.
// The candle kernel uses ggml block types (block_q4_0, block_q6_K, etc.)
// with complex template machinery; this port uses mlx-native's own
// affine quantization format (packed uint32 with bf16 scale/bias) and
// a simplified per-token expert-routing dispatch.
//
// Copyright the candle Authors. See LICENSE-APACHE-candle in this directory.

// quantized_matmul_id.metal — Expert-routed (MoE) quantized matrix-vector
// multiply with per-token expert selection via an ids buffer.
//
// For each token t and expert slot s:
//   expert_id = ids[t * n_expert_used + s]
//   W_e       = weight + expert_id * expert_weight_stride
//   S_e       = scales + expert_id * expert_scales_stride
//   B_e       = biases + expert_id * expert_biases_stride
//   output[t * n_expert_used * N + s * N + col] = sum_k(dequant(W_e[col][k]) * input[t * K + k])
//
// This kernel supports 4-bit, 6-bit, and 8-bit affine quantization with
// bf16 scales/biases, matching the non-id quantized_matmul kernel exactly.

#include <metal_stdlib>
using namespace metal;

// Parameters struct — must match the Rust-side QuantizedMatmulIdGpuParams.
struct QuantizedMatmulIdParams {
    uint M;              // number of input rows (tokens)
    uint K;              // inner dimension
    uint N;              // number of output columns per expert
    uint group_size;     // values per scale/bias group
    uint bits;           // 4, 6, or 8
    uint n_expert_used;  // number of experts per token (top-k)
    uint num_experts;    // total number of experts
    // Per-expert byte strides (allows contiguous 3D weight layout)
    uint expert_weight_stride; // bytes per expert in weight buffer
    uint expert_scales_stride; // uint16 elements per expert in scales buffer
    uint expert_biases_stride; // uint16 elements per expert in biases buffer
};

// Dequantization helpers — identical to quantized_matmul.metal

inline bfloat dequant_4bit_id(uint packed, uint i, bfloat scale, bfloat bias) {
    uint val = (packed >> (4 * i)) & 0xFu;
    return bfloat(val) * scale + bias;
}

inline bfloat dequant_6bit_id(uint packed, uint i, bfloat scale, bfloat bias) {
    uint val = (packed >> (6 * i)) & 0x3Fu;
    return bfloat(val) * scale + bias;
}

inline bfloat dequant_8bit_id(uint packed, uint i, bfloat scale, bfloat bias) {
    uint val = (packed >> (8 * i)) & 0xFFu;
    return bfloat(val) * scale + bias;
}

// Expert-routed quantized matmul kernel.
//
// Grid: (N, M * n_expert_used, 1)
//   tid.x = output column
//   tid.y = flattened (token * n_expert_used + expert_slot)
//
// Buffer layout:
//   buffer(0): input     — float32[M][K]
//   buffer(1): weight    — packed uint32[num_experts][N][packed_k] (contiguous per expert)
//   buffer(2): scales    — bf16[num_experts][N][num_groups] (as uint16)
//   buffer(3): biases    — bf16[num_experts][N][num_groups] (as uint16)
//   buffer(4): ids       — uint32[M][n_expert_used] (expert indices)
//   buffer(5): output    — float32[M][n_expert_used][N]
//   buffer(6): params    — QuantizedMatmulIdParams
kernel void quantized_matmul_id(
    device const float*    input   [[buffer(0)]],
    device const uint*     weight  [[buffer(1)]],
    device const uint16_t* scales  [[buffer(2)]],
    device const uint16_t* biases  [[buffer(3)]],
    device const uint*     ids     [[buffer(4)]],
    device float*          output  [[buffer(5)]],
    constant QuantizedMatmulIdParams& params [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;   // output column
    uint flat = tid.y;  // token * n_expert_used + expert_slot

    uint n_expert_used = params.n_expert_used;
    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    if (col >= N || flat >= M * n_expert_used) {
        return;
    }

    uint token = flat / n_expert_used;
    uint slot  = flat % n_expert_used;

    // Look up which expert this (token, slot) pair routes to.
    uint expert_id = ids[token * n_expert_used + slot];

    // Bounds check expert_id (safety).
    if (expert_id >= params.num_experts) {
        return;
    }

    uint group_size = params.group_size;
    uint bits = params.bits;
    uint num_groups = (K + group_size - 1) / group_size;

    // Pointer to this expert's weight, scales, biases.
    // Weight buffer is uint32*, but expert_weight_stride is in bytes.
    const device uint8_t* w_bytes_base = (const device uint8_t*)weight;
    const device uint8_t* w_expert = w_bytes_base + expert_id * params.expert_weight_stride;

    const device uint16_t* s_expert = scales + expert_id * params.expert_scales_stride;
    const device uint16_t* b_expert = biases + expert_id * params.expert_biases_stride;

    // Scale/bias base for this column.
    uint sb_base = col * num_groups;

    float acc = 0.0f;

    if (bits == 6) {
        // 6-bit: 4 values per 3-byte triplet.
        uint triplets_per_row = (K + 3) / 4;
        uint row_bytes = triplets_per_row * 3;
        const device uint8_t* w_row = w_expert + col * row_bytes;

        for (uint k = 0; k < K; k++) {
            uint triplet_idx = k / 4;
            uint in_triplet = k % 4;
            uint byte_off = triplet_idx * 3;
            uint packed = uint(w_row[byte_off])
                        | (uint(w_row[byte_off + 1]) << 8)
                        | (uint(w_row[byte_off + 2]) << 16);

            uint g = k / group_size;
            bfloat scale = as_type<bfloat>(s_expert[sb_base + g]);
            bfloat bias  = as_type<bfloat>(b_expert[sb_base + g]);
            bfloat w = dequant_6bit_id(packed, in_triplet, scale, bias);

            bfloat x = bfloat(input[token * K + k]);
            acc += float(w) * float(x);
        }
    } else {
        // 4-bit and 8-bit: uint32 packed.
        uint values_per_pack = (bits == 4) ? 8u : 4u;
        uint packs_per_row = (K + values_per_pack - 1) / values_per_pack;
        // Cast expert weight to uint32* for packed access.
        const device uint* w_expert_u32 = (const device uint*)w_expert;
        uint w_base = col * packs_per_row;

        for (uint k = 0; k < K; k++) {
            uint pack_idx = k / values_per_pack;
            uint in_pack_idx = k % values_per_pack;
            uint packed = w_expert_u32[w_base + pack_idx];

            uint g = k / group_size;
            bfloat scale = as_type<bfloat>(s_expert[sb_base + g]);
            bfloat bias  = as_type<bfloat>(b_expert[sb_base + g]);

            bfloat w;
            if (bits == 4) {
                w = dequant_4bit_id(packed, in_pack_idx, scale, bias);
            } else {
                w = dequant_8bit_id(packed, in_pack_idx, scale, bias);
            }

            bfloat x = bfloat(input[token * K + k]);
            acc += float(w) * float(x);
        }
    }

    // Write to output[token][slot][col].
    output[token * n_expert_used * N + slot * N + col] = acc;
}
