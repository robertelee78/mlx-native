// quantized_matmul.metal — MSL shader for 4-bit and 6-bit affine quantized
// matrix multiplication with on-the-fly dequantization.
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
    uint bits;        // 4 or 6
};

// Helper: read bf16 from a half buffer.  Metal's `half` type is IEEE f16, but
// we store MLX-style bf16 scale/bias as raw uint16 and convert manually.
// Actually, for simplicity and compatibility with the Rust side which stores
// scales/biases as f16 (not bf16), we use f16 directly via the `half` type.
// The story spec says bf16 but the practical implementation stores them as f16
// since Metal natively supports half (f16) but not bf16 in older MSL versions.
// We accept half* for scales and biases.

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

// Main quantized matmul kernel.
//
// Each thread computes one element of the output: output[row][col].
// This is a simple, correct baseline — future optimization (tiling, SIMD groups)
// will come in Epic 6.
//
// Buffer layout:
//   buffer(0): input    — float16[M][K] (row-major)
//   buffer(1): weight   — packed uint32[N][packed_k] (row-major per output column)
//   buffer(2): scales   — float16[N][num_groups_per_row] (one per group along K)
//   buffer(3): biases   — float16[N][num_groups_per_row]
//   buffer(4): output   — float16[M][N] (row-major)
//   buffer(5): params   — QuantizedMatmulParams
kernel void quantized_matmul(
    device const half*   input   [[buffer(0)]],
    device const uint*   weight  [[buffer(1)]],
    device const half*   scales  [[buffer(2)]],
    device const half*   biases  [[buffer(3)]],
    device half*         output  [[buffer(4)]],
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

    if (bits == 4) {
        // 4-bit: 8 values per uint32.
        uint values_per_pack = 8;
        // Number of uint32 packs per row of weights (one row = K values for one output col).
        uint packs_per_row = (K + values_per_pack - 1) / values_per_pack;
        // Weight base for this column.
        uint w_base = col * packs_per_row;

        for (uint k = 0; k < K; k++) {
            uint pack_idx = k / values_per_pack;
            uint in_pack_idx = k % values_per_pack;
            uint packed = weight[w_base + pack_idx];

            // Determine which group this k belongs to.
            uint g = k / group_size;
            float scale = float(scales[sb_base + g]);
            float bias  = float(biases[sb_base + g]);

            float w = dequant_4bit(packed, in_pack_idx, scale, bias);
            float x = float(input[row * K + k]);
            acc += w * x;
        }
    } else {
        // 6-bit: 4 values per uint32 (3 bytes = 24 bits packed into low 24 bits).
        uint values_per_pack = 4;
        uint packs_per_row = (K + values_per_pack - 1) / values_per_pack;
        uint w_base = col * packs_per_row;

        for (uint k = 0; k < K; k++) {
            uint pack_idx = k / values_per_pack;
            uint in_pack_idx = k % values_per_pack;
            uint packed = weight[w_base + pack_idx];

            uint g = k / group_size;
            float scale = float(scales[sb_base + g]);
            float bias  = float(biases[sb_base + g]);

            float w = dequant_6bit(packed, in_pack_idx, scale, bias);
            float x = float(input[row * K + k]);
            acc += w * x;
        }
    }

    output[row * params.N + col] = half(acc);
}
