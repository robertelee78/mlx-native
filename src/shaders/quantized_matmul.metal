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

// ---- Dequantization helpers ----
// All dequant functions operate in bf16 arithmetic to match MLX's precision.
// MLX dequantizes weights to bf16 (the model's native dtype), and multiplies
// bf16 weights with bf16 inputs, accumulating products in f32.
// Using bf16 dequant ensures our weight values match MLX's exactly.

// 4-bit: Extract the i-th 4-bit value from a packed uint32.
inline bfloat dequant_4bit(uint packed, uint i, bfloat scale, bfloat bias) {
    uint val = (packed >> (4 * i)) & 0xFu;
    return bfloat(val) * scale + bias;
}

// 6-bit: Extract the i-th 6-bit value from a packed uint32 (4 values per uint32).
inline bfloat dequant_6bit(uint packed, uint i, bfloat scale, bfloat bias) {
    uint val = (packed >> (6 * i)) & 0x3Fu;
    return bfloat(val) * scale + bias;
}

// 8-bit: Extract the i-th 8-bit value from a packed uint32 (4 values per uint32).
inline bfloat dequant_8bit(uint packed, uint i, bfloat scale, bfloat bias) {
    uint val = (packed >> (8 * i)) & 0xFFu;
    return bfloat(val) * scale + bias;
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
    // 6-bit: 4 values per 3 bytes (MLX triplet format, read as uint8_t triples)
    // 8-bit: 4 values per uint32
    const device uint8_t* w_bytes = (const device uint8_t*)weight;

    if (bits == 6) {
        // 6-bit: 4 values per 3-byte triplet. Row stride = ceil(K/4) * 3 bytes.
        uint triplets_per_row = (K + 3) / 4;
        uint row_bytes = triplets_per_row * 3;
        const device uint8_t* w_row = w_bytes + col * row_bytes;

        for (uint k = 0; k < K; k++) {
            uint triplet_idx = k / 4;
            uint in_triplet = k % 4;
            // Read 3 bytes as uint32 (high byte is zero-padded)
            uint byte_off = triplet_idx * 3;
            uint packed = uint(w_row[byte_off])
                        | (uint(w_row[byte_off + 1]) << 8)
                        | (uint(w_row[byte_off + 2]) << 16);

            uint g = k / group_size;
            bfloat scale = as_type<bfloat>(scales[sb_base + g]);
            bfloat bias  = as_type<bfloat>(biases[sb_base + g]);
            bfloat w = dequant_6bit(packed, in_triplet, scale, bias);

            bfloat x = bfloat(input[row * K + k]);
            acc += float(w) * float(x);
        }
    } else {
        // 4-bit and 8-bit: uint32 packed
        uint values_per_pack = (bits == 4) ? 8u : 4u;
        uint packs_per_row = (K + values_per_pack - 1) / values_per_pack;
        uint w_base = col * packs_per_row;

        for (uint k = 0; k < K; k++) {
            uint pack_idx = k / values_per_pack;
            uint in_pack_idx = k % values_per_pack;
            uint packed = weight[w_base + pack_idx];

            uint g = k / group_size;
            bfloat scale = as_type<bfloat>(scales[sb_base + g]);
            bfloat bias  = as_type<bfloat>(biases[sb_base + g]);

            bfloat w;
            if (bits == 4) {
                w = dequant_4bit(packed, in_pack_idx, scale, bias);
            } else {
                w = dequant_8bit(packed, in_pack_idx, scale, bias);
            }

            bfloat x = bfloat(input[row * K + k]);
            acc += float(w) * float(x);
        }
    }

    output[row * params.N + col] = acc;
}

// ---------------------------------------------------------------------------
// SIMD-cooperative quantized matrix-vector multiply kernel.
//
// Matches MLX's qmv_fast accumulation pattern exactly:
//   - 2 simdgroups per threadgroup, 4 output rows per simdgroup (8 total)
//   - Each thread in a simdgroup processes values_per_thread elements
//   - simd_sum() reduction across the 32 threads in a simdgroup
//
// This ensures bit-identical f32 accumulation order to MLX, which is
// critical for matching inference results.
//
// Dispatch (from Rust):
//   threadgroup_size = (32, 2, 1)   — 32 threads * 2 simdgroups = 64 threads
//   threadgroups     = (M, ceil(N / 8), 1)
//
// Buffer layout matches the original kernel exactly.
// ---------------------------------------------------------------------------

// Constants matching MLX's qmv_fast_impl
constant constexpr int SIMD_SIZE_CONST = 32;

// -- load_vector for 4-bit: pre-divide inputs for packed dot product --
// MLX loads values_per_thread (16 for 4-bit) consecutive input values,
// divides them by the positional shift factors, and sums the raw values.
// This allows qdot to multiply directly with masked (unshifted) packed weights.
//
// Input is f32 on device; we truncate to bfloat then promote to float to
// match MLX's bf16 pipeline exactly.

inline void load_vector_4bit(
    const device float* x_f32,
    thread float* x_thread,
    thread float& sum,
    int count  // values_per_thread, expected 16
) {
    float s = 0.0f;
    // 4-bit: pack_factor=8, values come in groups of 4 from uint16_t reads
    // Divisors: 1, 16, 256, 4096 (corresponding to nibble positions in uint16_t)
    for (int i = 0; i < count; i += 4) {
        // Truncate f32 -> bfloat -> float to match MLX's precision
        float v0 = float(bfloat(x_f32[i]));
        float v1 = float(bfloat(x_f32[i + 1]));
        float v2 = float(bfloat(x_f32[i + 2]));
        float v3 = float(bfloat(x_f32[i + 3]));
        s += v0 + v1 + v2 + v3;
        x_thread[i]     = v0;
        x_thread[i + 1] = v1 / 16.0f;
        x_thread[i + 2] = v2 / 256.0f;
        x_thread[i + 3] = v3 / 4096.0f;
    }
    sum = s;
}

// -- load_vector for 8-bit --
inline void load_vector_8bit(
    const device float* x_f32,
    thread float* x_thread,
    thread float& sum,
    int count  // values_per_thread, expected 8
) {
    float s = 0.0f;
    for (int i = 0; i < count; i++) {
        float v = float(bfloat(x_f32[i]));
        s += v;
        x_thread[i] = v;
    }
    sum = s;
}

// -- qdot for 4-bit: masked multiply-accumulate with packed uint16_t weights --
// Reads weight bytes as uint16_t*, multiplies masked values with pre-divided inputs.
// Result: scale * accum + sum * bias  (equivalent to proper dequant + dot)
inline float qdot_4bit(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int values_per_thread  // 16
) {
    float accum = 0.0f;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
        accum +=
            (x_thread[4 * i]     * float(ws[i] & 0x000fu) +
             x_thread[4 * i + 1] * float(ws[i] & 0x00f0u) +
             x_thread[4 * i + 2] * float(ws[i] & 0x0f00u) +
             x_thread[4 * i + 3] * float(ws[i] & 0xf000u));
    }
    return scale * accum + sum * bias;
}

// -- qdot for 8-bit --
inline float qdot_8bit(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int values_per_thread  // 8
) {
    float accum = 0.0f;
    for (int i = 0; i < values_per_thread; i++) {
        accum += x_thread[i] * float(w[i]);
    }
    return scale * accum + sum * bias;
}

// The main SIMD-cooperative kernel.
// Supports 4-bit and 8-bit (the two formats used by our model).
kernel void quantized_matmul_simd(
    device const float*    input   [[buffer(0)]],
    device const uint*     weight  [[buffer(1)]],
    device const uint16_t* scales  [[buffer(2)]],
    device const uint16_t* biases  [[buffer(3)]],
    device float*          output  [[buffer(4)]],
    constant QuantizedMatmulParams& params [[buffer(5)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]
) {
    // Constants matching MLX's qmv_fast_impl template parameters
    const uint num_simdgroups = 2;
    const uint results_per_simdgroup = 4;

    const uint K = params.K;
    const uint N = params.N;
    const uint bits = params.bits;
    const uint group_size = params.group_size;

    // Packing parameters — match MLX's qmv (NOT qmv_fast) for K=2816 compatibility
    // 4-bit: pack_factor=8, packs_per_thread=1, values_per_thread=8, bytes_per_pack=4
    // 8-bit: pack_factor=4, packs_per_thread=2, values_per_thread=8, bytes_per_pack=4
    uint pack_factor;
    uint packs_per_thread;
    uint values_per_thread;
    uint bytes_per_pack;

    if (bits == 4) {
        pack_factor = 8;
        packs_per_thread = 1;
        values_per_thread = 8;     // 8 * 1 = 8, block_size = 256
        bytes_per_pack = 4;
    } else if (bits == 8) {
        pack_factor = 4;
        packs_per_thread = 2;
        values_per_thread = 8;
        bytes_per_pack = 4;
    } else {
        // Unsupported in SIMD path — should not be called for 6-bit
        return;
    }

    const uint block_size = values_per_thread * SIMD_SIZE_CONST;
    const uint scale_step_per_thread = group_size / values_per_thread;

    // Weight bytes pointer
    const device uint8_t* ws = (const device uint8_t*)weight;

    // Compute row in input (tid.x in MLX's grid corresponds to M dimension)
    const uint row = tid.x;  // which input row
    if (row >= params.M) return;

    // Compute output column block
    const uint out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                         simd_gid * results_per_simdgroup;

    // Bounds check: if all 4 results are out of range, return
    if (out_row >= N) return;

    // Weight row stride in bytes: each output column row has (K / pack_factor) packs,
    // each pack is bytes_per_pack bytes.
    const uint in_vec_size_w = K * bytes_per_pack / pack_factor;  // bytes per weight row
    const uint in_vec_size_g = K / group_size;  // number of scale/bias groups per row

    // Position pointers for this thread
    // ws points to: out_row's weight data + this thread's offset
    const device uint8_t* ws_ptr = ws + out_row * in_vec_size_w +
                                   simd_lid * packs_per_thread * bytes_per_pack;
    // scales/biases: out_row's group data + this thread's group offset
    const device uint16_t* sc_ptr = scales + out_row * in_vec_size_g +
                                    simd_lid / scale_step_per_thread;
    const device uint16_t* bi_ptr = biases + out_row * in_vec_size_g +
                                    simd_lid / scale_step_per_thread;

    // Input pointer: row * K + this thread's element offset
    // MLX reads input as type T (bfloat), but our input is f32.
    // To match MLX's accumulation exactly, we must truncate to bfloat.
    // We'll read f32 and cast to bfloat in the load function.
    const device float* x_f32 = input + row * K + simd_lid * values_per_thread;

    // Output pointer
    device float* y_ptr = output + row * N + out_row;

    // Per-thread accumulation registers
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Main loop over K in blocks of block_size
    if (bits == 4) {
        float x_thread[8];  // qmv: 8 values per thread (block_size=256)
        for (uint k = 0; k < K; k += block_size) {
            // Load input values with pre-division for packed dot product
            // f32 -> bfloat truncation happens inside load_vector_4bit
            float x_sum;
            load_vector_4bit(x_f32, x_thread, x_sum, values_per_thread);

            // Dot product with each of the 4 output rows
            for (uint r = 0; r < results_per_simdgroup; r++) {
                const device uint8_t* wl = ws_ptr + r * in_vec_size_w;
                float s = float(as_type<bfloat>(sc_ptr[r * in_vec_size_g]));
                float b = float(as_type<bfloat>(bi_ptr[r * in_vec_size_g]));
                result[r] += qdot_4bit(wl, x_thread, s, b, x_sum, values_per_thread);
            }

            // Advance pointers by block_size
            ws_ptr += block_size * bytes_per_pack / pack_factor;
            sc_ptr += block_size / group_size;
            bi_ptr += block_size / group_size;
            x_f32  += block_size;
        }
    } else {
        // 8-bit path
        float x_thread[8];
        for (uint k = 0; k < K; k += block_size) {
            float x_sum;
            load_vector_8bit(x_f32, x_thread, x_sum, 8);

            for (uint r = 0; r < results_per_simdgroup; r++) {
                const device uint8_t* wl = ws_ptr + r * in_vec_size_w;
                float s = float(as_type<bfloat>(sc_ptr[r * in_vec_size_g]));
                float b = float(as_type<bfloat>(bi_ptr[r * in_vec_size_g]));
                result[r] += qdot_8bit(wl, x_thread, s, b, x_sum, 8);
            }

            ws_ptr += block_size * bytes_per_pack / pack_factor;
            sc_ptr += block_size / group_size;
            bi_ptr += block_size / group_size;
            x_f32  += block_size;
        }
    }

    // Reduce across simdgroup and write output
    for (uint r = 0; r < results_per_simdgroup; r++) {
        result[r] = simd_sum(result[r]);
        if (simd_lid == 0) {
            // Only write if this output column is in bounds
            if (out_row + r < N) {
                y_ptr[r] = result[r];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// load_vector helpers for bf16 input (used by bf16 variants below).
//
// These mirror load_vector_4bit / load_vector_8bit but read bfloat instead of
// float32.  The values are promoted to float for accumulation — identical
// arithmetic to the f32 path because both paths truncate to bfloat first.
// ---------------------------------------------------------------------------

inline void load_vector_4bit_bf16(
    const device bfloat* x_bf16,
    thread float* x_thread,
    thread float& sum,
    int count  // values_per_thread, expected 16 (same as f32 variant)
) {
    float s = 0.0f;
    for (int i = 0; i < count; i += 4) {
        float v0 = float(x_bf16[i]);
        float v1 = float(x_bf16[i + 1]);
        float v2 = float(x_bf16[i + 2]);
        float v3 = float(x_bf16[i + 3]);
        s += v0 + v1 + v2 + v3;
        x_thread[i]     = v0;
        x_thread[i + 1] = v1 / 16.0f;
        x_thread[i + 2] = v2 / 256.0f;
        x_thread[i + 3] = v3 / 4096.0f;
    }
    sum = s;
}

inline void load_vector_8bit_bf16(
    const device bfloat* x_bf16,
    thread float* x_thread,
    thread float& sum,
    int count  // values_per_thread, expected 8
) {
    float s = 0.0f;
    for (int i = 0; i < count; i++) {
        float v = float(x_bf16[i]);
        s += v;
        x_thread[i] = v;
    }
    sum = s;
}

// ---------------------------------------------------------------------------
// quantized_matmul_simd_bf16
//
// Identical computation to quantized_matmul_simd but with bf16 input and
// bf16 output.  Accumulation remains in f32 for numerical stability.
//
// Buffer layout:
//   buffer(0): input    — bfloat[M][K]
//   buffer(1): weight   — packed uint32[N][packed_k]
//   buffer(2): scales   — uint16_t[N][num_groups]  (bf16 bits)
//   buffer(3): biases   — uint16_t[N][num_groups]  (bf16 bits)
//   buffer(4): output   — bfloat[M][N]
//   buffer(5): params   — QuantizedMatmulParams
// ---------------------------------------------------------------------------
kernel void quantized_matmul_simd_bf16(
    device const bfloat*   input   [[buffer(0)]],
    device const uint*     weight  [[buffer(1)]],
    device const uint16_t* scales  [[buffer(2)]],
    device const uint16_t* biases  [[buffer(3)]],
    device bfloat*         output  [[buffer(4)]],
    constant QuantizedMatmulParams& params [[buffer(5)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]
) {
    const uint num_simdgroups = 2;
    const uint results_per_simdgroup = 4;

    const uint K = params.K;
    const uint N = params.N;
    const uint bits = params.bits;
    const uint group_size = params.group_size;

    uint pack_factor;
    uint packs_per_thread;
    uint values_per_thread;
    uint bytes_per_pack;

    // IMPORTANT: values_per_thread and packs_per_thread MUST match the f32
    // variant exactly.  A previous bug used values_per_thread=8 for 4-bit
    // which halved the block_size and corrupted all pointer arithmetic.
    if (bits == 4) {
        pack_factor = 8;
        packs_per_thread = 2;       // was incorrectly 1
        values_per_thread = 16;     // was incorrectly 8
        bytes_per_pack = 4;
    } else if (bits == 8) {
        pack_factor = 4;
        packs_per_thread = 2;
        values_per_thread = 8;
        bytes_per_pack = 4;
    } else {
        return;
    }

    const uint block_size = values_per_thread * SIMD_SIZE_CONST;
    const uint scale_step_per_thread = group_size / values_per_thread;

    const device uint8_t* ws = (const device uint8_t*)weight;

    const uint row = tid.x;
    if (row >= params.M) return;

    const uint out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                         simd_gid * results_per_simdgroup;
    if (out_row >= N) return;

    const uint in_vec_size_w = K * bytes_per_pack / pack_factor;
    const uint in_vec_size_g = K / group_size;

    const device uint8_t* ws_ptr = ws + out_row * in_vec_size_w +
                                   simd_lid * packs_per_thread * bytes_per_pack;
    const device uint16_t* sc_ptr = scales + out_row * in_vec_size_g +
                                    simd_lid / scale_step_per_thread;
    const device uint16_t* bi_ptr = biases + out_row * in_vec_size_g +
                                    simd_lid / scale_step_per_thread;

    // Input is bf16 — read directly without cast.
    const device bfloat* x_bf16 = input + row * K + simd_lid * values_per_thread;

    // Output is bf16.
    device bfloat* y_ptr = output + row * N + out_row;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (bits == 4) {
        float x_thread[16];  // was incorrectly [8]
        for (uint k = 0; k < K; k += block_size) {
            float x_sum;
            load_vector_4bit_bf16(x_bf16, x_thread, x_sum, values_per_thread);

            for (uint r = 0; r < results_per_simdgroup; r++) {
                const device uint8_t* wl = ws_ptr + r * in_vec_size_w;
                float s = float(as_type<bfloat>(sc_ptr[r * in_vec_size_g]));
                float b = float(as_type<bfloat>(bi_ptr[r * in_vec_size_g]));
                result[r] += qdot_4bit(wl, x_thread, s, b, x_sum, values_per_thread);
            }

            ws_ptr += block_size * bytes_per_pack / pack_factor;
            sc_ptr += block_size / group_size;
            bi_ptr += block_size / group_size;
            x_bf16 += block_size;
        }
    } else {
        float x_thread[8];
        for (uint k = 0; k < K; k += block_size) {
            float x_sum;
            load_vector_8bit_bf16(x_bf16, x_thread, x_sum, 8);

            for (uint r = 0; r < results_per_simdgroup; r++) {
                const device uint8_t* wl = ws_ptr + r * in_vec_size_w;
                float s = float(as_type<bfloat>(sc_ptr[r * in_vec_size_g]));
                float b = float(as_type<bfloat>(bi_ptr[r * in_vec_size_g]));
                result[r] += qdot_8bit(wl, x_thread, s, b, x_sum, 8);
            }

            ws_ptr += block_size * bytes_per_pack / pack_factor;
            sc_ptr += block_size / group_size;
            bi_ptr += block_size / group_size;
            x_bf16 += block_size;
        }
    }

    // Reduce and write bf16 output.
    for (uint r = 0; r < results_per_simdgroup; r++) {
        result[r] = simd_sum(result[r]);
        if (simd_lid == 0) {
            if (out_row + r < N) {
                y_ptr[r] = bfloat(result[r]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// quantized_matmul_simd_bf16_expert
//
// Same as quantized_matmul_simd_bf16 but accepts byte-offset parameters that
// allow indexing into a 3D packed weight tensor [n_experts, rows, packed_cols]
// without CPU memcpy.  The offsets are in bytes.
//
// Buffer layout:
//   buffer(0): input              — bfloat[M][K]
//   buffer(1): packed_weights     — packed uint32 for ALL experts (3D layout)
//   buffer(2): scales             — uint16_t scales for ALL experts
//   buffer(3): biases             — uint16_t biases for ALL experts
//   buffer(4): output             — bfloat[M][N]
//   buffer(5): params             — QuantizedMatmulParams
//   buffer(6): expert_offset      — byte offset into packed_weights for this expert
//   buffer(7): scales_offset      — byte offset into scales for this expert
//   buffer(8): biases_offset      — byte offset into biases for this expert
// ---------------------------------------------------------------------------
kernel void quantized_matmul_simd_bf16_expert(
    device const bfloat*   input             [[buffer(0)]],
    device const uint8_t*  packed_weights    [[buffer(1)]],
    device const uint8_t*  scales_raw        [[buffer(2)]],
    device const uint8_t*  biases_raw        [[buffer(3)]],
    device bfloat*         output            [[buffer(4)]],
    constant QuantizedMatmulParams& params   [[buffer(5)]],
    constant uint& expert_offset             [[buffer(6)]],
    constant uint& scales_offset             [[buffer(7)]],
    constant uint& biases_offset             [[buffer(8)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]]
) {
    const uint num_simdgroups = 2;
    const uint results_per_simdgroup = 4;

    const uint K = params.K;
    const uint N = params.N;
    const uint bits = params.bits;
    const uint group_size = params.group_size;

    uint pack_factor;
    uint packs_per_thread;
    uint values_per_thread;
    uint bytes_per_pack;

    // ADR-002 fix: 4-bit params must match non-expert bf16 kernel (qmv_fast).
    // Previously had packs_per_thread=1, values_per_thread=8 which halved
    // block_size and corrupted all pointer arithmetic for 4-bit MoE layers.
    if (bits == 4) {
        pack_factor = 8;
        packs_per_thread = 2;       // qmv_fast: 2 (was incorrectly 1)
        values_per_thread = 16;     // qmv_fast: 8*2=16 (was incorrectly 8)
        bytes_per_pack = 4;
    } else if (bits == 8) {
        pack_factor = 4;
        packs_per_thread = 2;
        values_per_thread = 8;
        bytes_per_pack = 4;
    } else {
        return;
    }

    const uint block_size = values_per_thread * SIMD_SIZE_CONST;
    const uint scale_step_per_thread = group_size / values_per_thread;

    // Apply byte offsets to reach this expert's slice in the 3D buffers.
    const device uint8_t* ws = packed_weights + expert_offset;
    const device uint16_t* scales = (const device uint16_t*)(scales_raw + scales_offset);
    const device uint16_t* biases = (const device uint16_t*)(biases_raw + biases_offset);

    const uint row = tid.x;
    if (row >= params.M) return;

    const uint out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
                         simd_gid * results_per_simdgroup;
    if (out_row >= N) return;

    const uint in_vec_size_w = K * bytes_per_pack / pack_factor;
    const uint in_vec_size_g = K / group_size;

    const device uint8_t* ws_ptr = ws + out_row * in_vec_size_w +
                                   simd_lid * packs_per_thread * bytes_per_pack;
    const device uint16_t* sc_ptr = scales + out_row * in_vec_size_g +
                                    simd_lid / scale_step_per_thread;
    const device uint16_t* bi_ptr = biases + out_row * in_vec_size_g +
                                    simd_lid / scale_step_per_thread;

    const device bfloat* x_bf16 = input + row * K + simd_lid * values_per_thread;
    device bfloat* y_ptr = output + row * N + out_row;

    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (bits == 4) {
        float x_thread[16];  // ADR-002 fix: was [8], must be [16] for qmv_fast
        for (uint k = 0; k < K; k += block_size) {
            float x_sum;
            load_vector_4bit_bf16(x_bf16, x_thread, x_sum, values_per_thread);

            for (uint r = 0; r < results_per_simdgroup; r++) {
                const device uint8_t* wl = ws_ptr + r * in_vec_size_w;
                float s = float(as_type<bfloat>(sc_ptr[r * in_vec_size_g]));
                float b = float(as_type<bfloat>(bi_ptr[r * in_vec_size_g]));
                result[r] += qdot_4bit(wl, x_thread, s, b, x_sum, values_per_thread);
            }

            ws_ptr += block_size * bytes_per_pack / pack_factor;
            sc_ptr += block_size / group_size;
            bi_ptr += block_size / group_size;
            x_bf16 += block_size;
        }
    } else {
        float x_thread[8];
        for (uint k = 0; k < K; k += block_size) {
            float x_sum;
            load_vector_8bit_bf16(x_bf16, x_thread, x_sum, 8);

            for (uint r = 0; r < results_per_simdgroup; r++) {
                const device uint8_t* wl = ws_ptr + r * in_vec_size_w;
                float s = float(as_type<bfloat>(sc_ptr[r * in_vec_size_g]));
                float b = float(as_type<bfloat>(bi_ptr[r * in_vec_size_g]));
                result[r] += qdot_8bit(wl, x_thread, s, b, x_sum, 8);
            }

            ws_ptr += block_size * bytes_per_pack / pack_factor;
            sc_ptr += block_size / group_size;
            bi_ptr += block_size / group_size;
            x_bf16 += block_size;
        }
    }

    for (uint r = 0; r < results_per_simdgroup; r++) {
        result[r] = simd_sum(result[r]);
        if (simd_lid == 0) {
            if (out_row + r < N) {
                y_ptr[r] = bfloat(result[r]);
            }
        }
    }
}
