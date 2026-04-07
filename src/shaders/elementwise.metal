#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// Elementwise operations: add, multiply, cast
//
// These are simple per-element kernels used for residual connections,
// scaling, and dtype conversion in the inference pipeline.
// --------------------------------------------------------------------------

struct ElementwiseParams {
    uint n_elements;
};

// --------------------------------------------------------------------------
// elementwise_add_f32 — out = a + b (float32)
//
// Buffers:
//   0: a      — float [n_elements]
//   1: b      — float [n_elements]
//   2: output — float [n_elements]
//   3: params — { n_elements }
// --------------------------------------------------------------------------
kernel void elementwise_add_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = a[gid] + b[gid];
}

// --------------------------------------------------------------------------
// elementwise_add_f16 — out = a + b (float16)
//
// Buffers:
//   0: a      — half [n_elements]
//   1: b      — half [n_elements]
//   2: output — half [n_elements]
//   3: params — { n_elements }
// --------------------------------------------------------------------------
kernel void elementwise_add_f16(
    device const half* a       [[buffer(0)]],
    device const half* b       [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = a[gid] + b[gid];
}

// --------------------------------------------------------------------------
// elementwise_mul_f32 — out = a * b (float32)
//
// Buffers:
//   0: a      — float [n_elements]
//   1: b      — float [n_elements]
//   2: output — float [n_elements]
//   3: params — { n_elements }
// --------------------------------------------------------------------------
kernel void elementwise_mul_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = a[gid] * b[gid];
}

// --------------------------------------------------------------------------
// elementwise_mul_f16 — out = a * b (float16)
//
// Buffers:
//   0: a      — half [n_elements]
//   1: b      — half [n_elements]
//   2: output — half [n_elements]
//   3: params — { n_elements }
// --------------------------------------------------------------------------
kernel void elementwise_mul_f16(
    device const half* a       [[buffer(0)]],
    device const half* b       [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = a[gid] * b[gid];
}

// --------------------------------------------------------------------------
// elementwise_add_bf16 — out = a + b (bfloat16)
//
// Upcasts to f32 for the addition, downcasts result to bfloat16.
//
// Buffers:
//   0: a      — bfloat [n_elements]
//   1: b      — bfloat [n_elements]
//   2: output — bfloat [n_elements]
//   3: params — { n_elements }
// --------------------------------------------------------------------------
kernel void elementwise_add_bf16(
    device const bfloat* a       [[buffer(0)]],
    device const bfloat* b       [[buffer(1)]],
    device bfloat*       output  [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = bfloat(float(a[gid]) + float(b[gid]));
}

// --------------------------------------------------------------------------
// elementwise_mul_bf16 — out = a * b (bfloat16)
//
// Upcasts to f32 for the multiply, downcasts result to bfloat16.
//
// Buffers:
//   0: a      — bfloat [n_elements]
//   1: b      — bfloat [n_elements]
//   2: output — bfloat [n_elements]
//   3: params — { n_elements }
// --------------------------------------------------------------------------
kernel void elementwise_mul_bf16(
    device const bfloat* a       [[buffer(0)]],
    device const bfloat* b       [[buffer(1)]],
    device bfloat*       output  [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = bfloat(float(a[gid]) * float(b[gid]));
}

// --------------------------------------------------------------------------
// cast_f16_to_f32 — convert half to float
//
// Buffers:
//   0: input  — half  [n_elements]
//   1: output — float [n_elements]
//   2: params — { n_elements }
// --------------------------------------------------------------------------
kernel void cast_f16_to_f32(
    device const half* input   [[buffer(0)]],
    device float*      output  [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = static_cast<float>(input[gid]);
}

// --------------------------------------------------------------------------
// cast_f32_to_f16 — convert float to half
//
// Buffers:
//   0: input  — float [n_elements]
//   1: output — half  [n_elements]
//   2: params — { n_elements }
// --------------------------------------------------------------------------
kernel void cast_f32_to_f16(
    device const float* input  [[buffer(0)]],
    device half*        output [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = static_cast<half>(input[gid]);
}

// --------------------------------------------------------------------------
// cast_bf16_to_f32 — convert bfloat16 to float
//
// Buffers:
//   0: input  — bfloat [n_elements]
//   1: output — float  [n_elements]
//   2: params — { n_elements }
// --------------------------------------------------------------------------
kernel void cast_bf16_to_f32(
    device const bfloat* input  [[buffer(0)]],
    device float*        output [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = static_cast<float>(input[gid]);
}

// --------------------------------------------------------------------------
// cast_f32_to_bf16 — convert float to bfloat16
//
// Buffers:
//   0: input  — float  [n_elements]
//   1: output — bfloat [n_elements]
//   2: params — { n_elements }
// --------------------------------------------------------------------------
kernel void cast_f32_to_bf16(
    device const float* input  [[buffer(0)]],
    device bfloat*      output [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = static_cast<bfloat>(input[gid]);
}

// --------------------------------------------------------------------------
// transpose_2d_f32 — Transpose a 2D matrix [rows, cols] -> [cols, rows]
//
// Buffers:
//   0: input  — float [rows * cols]
//   1: output — float [cols * rows]
//   2: params — { rows, cols } packed as uint2 in ElementwiseParams
//               We reuse n_elements for rows and add a second field.
// --------------------------------------------------------------------------

struct TransposeParams {
    uint rows;
    uint cols;
};

// --------------------------------------------------------------------------
// scalar_mul_bf16 — out = input * scalar (bfloat16 input/output, f32 scalar)
//
// Buffers:
//   0: input  — bfloat [count]
//   1: output — bfloat [count]
//   2: params — { float scalar; uint count; }
// --------------------------------------------------------------------------

struct ScalarMulParams {
    float scalar;
    uint count;
};

kernel void scalar_mul_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant ScalarMulParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    output[gid] = bfloat(static_cast<float>(input[gid]) * params.scalar);
}

// --------------------------------------------------------------------------
// permute_021_bf16 — Transpose [A, B, C] -> [B, A, C] for bfloat16
//
// Used to convert [seq_len, n_heads, head_dim] <-> [n_heads, seq_len, head_dim]
//
// Buffers:
//   0: input  — bfloat [A * B * C]
//   1: output — bfloat [B * A * C]
//   2: params — { uint dim_a; uint dim_b; uint dim_c; }
//
// Grid: (C, B, A), each thread copies one element
// --------------------------------------------------------------------------

struct Permute021Params {
    uint dim_a;
    uint dim_b;
    uint dim_c;
};

kernel void permute_021_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant Permute021Params& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint b = gid.y;
    const uint a = gid.z;

    if (a >= params.dim_a || b >= params.dim_b || c >= params.dim_c) return;

    // input[a, b, c]  -> offset = a * (B*C) + b * C + c
    // output[b, a, c] -> offset = b * (A*C) + a * C + c
    const uint in_idx  = a * (params.dim_b * params.dim_c) + b * params.dim_c + c;
    const uint out_idx = b * (params.dim_a * params.dim_c) + a * params.dim_c + c;

    output[out_idx] = input[in_idx];
}

kernel void transpose_2d_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant TransposeParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;

    if (row >= params.rows || col >= params.cols) return;

    // input[row, col] -> output[col, row]
    output[col * params.rows + row] = input[row * params.cols + col];
}

kernel void transpose_2d_f16(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant TransposeParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;

    if (row >= params.rows || col >= params.cols) return;

    output[col * params.rows + row] = input[row * params.cols + col];
}
