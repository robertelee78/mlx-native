// quantized_matmul.metal — MSL shader for 4-bit, 6-bit, and 8-bit affine
// quantized matrix multiplication with on-the-fly dequantization.
//
// Computes: output[row][col] = sum_k(dequant(weight[col][k]) * input[row][k])
//
// Weight layout: (N, K) — each of N output columns is a row in the weight
// matrix, packed in quantized format.  Scales and biases are per-group
// (group_size consecutive values along K share one scale and one bias).
//
// Packing formats:
//   4-bit: 8 values per uint32, value i = (packed >> (4*i)) & 0xF
//   6-bit: 4 values per 3 bytes packed into uint32 (MLX triplet format)
//          val0 = packed & 0x3F, val1 = (packed>>6) & 0x3F, etc.
//   8-bit: 4 values per uint32, value i = (packed >> (8*i)) & 0xFF
//
// Dequantization: float_val = scale * quant_val + bias
//   where scale and bias are bf16, one per group of group_size values.
//
// Accumulation is done in f32 for numerical stability; output is written as f16.

#include <metal_stdlib>
using namespace metal;

// Parameters struct — must match the Rust-side QuantizedMatmulGpuParams.
struct QuantizedMatmulParams {
    uint M;           // number of input rows (tokens)
    uint K;           // inner dimension
    uint N;           // number of output columns
    uint group_size;  // values per scale/bias group
    uint bits;        // 4, 6, or 8
};

// Helper: read bf16 scales/biases.  Metal's `half` type is IEEE f16, which is
// a DIFFERENT format from bf16 (bfloat16).  MLX stores scales and biases as
// bf16, so we read them as raw uint16_t and reinterpret via as_type<bfloat>,
// then cast to float for dequantization arithmetic.

// ---- 4-bit dequantization ----
// Extract the i-th 4-bit value from a packed uint32.
inline float dequant_4bit(uint packed, uint i, float scale, float bias) {
    uint val = (packed >> (4 * i)) & 0xFu;
    return scale * float(val) + bias;
}

// ---- 6-bit dequantization (MLX 3-byte triplet) ----
// Extract the i-th 6-bit value from a packed uint32 (4 values per uint32).
inline float dequant_6bit(uint packed, uint i, float scale, float bias) {
    uint val = (packed >> (6 * i)) & 0x3Fu;
    return scale * float(val) + bias;
}

// ---- 8-bit dequantization ----
// Extract the i-th 8-bit value from a packed uint32 (4 values per uint32).
inline float dequant_8bit(uint packed, uint i, float scale, float bias) {
    uint val = (packed >> (8 * i)) & 0xFFu;
    return scale * float(val) + bias;
}

// Main quantized matmul kernel (f32 output).
//
// Each thread computes one element of the output: output[row][col].
// This is a simple, correct baseline — future optimization (tiling, SIMD groups)
// will come in Epic 6.
//
// Accumulation and output are both f32 to avoid f16 overflow (max ~65504) on
// projections with large intermediate values (e.g. attention output projections).
//
// Buffer layout:
//   buffer(0): input    — float32[M][K] (row-major)
//   buffer(1): weight   — packed uint32[N][packed_k] (row-major per output column)
//   buffer(2): scales   — bf16[N][num_groups_per_row] (one per group along K, stored as uint16)
//   buffer(3): biases   — bf16[N][num_groups_per_row] (stored as uint16)
//   buffer(4): output   — float32[M][N] (row-major)
//   buffer(5): params   — QuantizedMatmulParams
kernel void quantized_matmul(
    device const float*  input   [[buffer(0)]],
    device const uint*   weight  [[buffer(1)]],
    device const uint16_t* scales  [[buffer(2)]],
    device const uint16_t* biases  [[buffer(3)]],
    device float*        output  [[buffer(4)]],
    constant QuantizedMatmulParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.y;  // which input row (token)
    uint col = tid.x;  // which output column

    if (row >= params.M || col >= params.N) {
        return;
    }

    uint K = params.K;
    uint group_size = params.group_size;
    uint bits = params.bits;

    // Number of groups along K for one output column.
    uint num_groups = (K + group_size - 1) / group_size;

    // Scale/bias for this column: stored contiguously as scales[col * num_groups + g].
    uint sb_base = col * num_groups;

    float acc = 0.0f;

    // Determine packing parameters based on bit-width.
    // 4-bit: 8 values per uint32
    // 6-bit: 4 values per uint32 (MLX triplet format)
    // 8-bit: 4 values per uint32
    uint values_per_pack = (bits == 4) ? 8u : 4u;
    uint packs_per_row = (K + values_per_pack - 1) / values_per_pack;
    uint w_base = col * packs_per_row;

    for (uint k = 0; k < K; k++) {
        uint pack_idx = k / values_per_pack;
        uint in_pack_idx = k % values_per_pack;
        uint packed = weight[w_base + pack_idx];

        uint g = k / group_size;
        float scale = static_cast<float>(as_type<bfloat>(scales[sb_base + g]));
        float bias  = static_cast<float>(as_type<bfloat>(biases[sb_base + g]));

        float w;
        if (bits == 4) {
            w = dequant_4bit(packed, in_pack_idx, scale, bias);
        } else if (bits == 6) {
            w = dequant_6bit(packed, in_pack_idx, scale, bias);
        } else {
            w = dequant_8bit(packed, in_pack_idx, scale, bias);
        }

        float x = input[row * K + k];
        acc += w * x;
    }

    output[row * params.N + col] = acc;
}
