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
