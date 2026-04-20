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
// scalar_mul_f32 — out = input * scalar (float32 input/output, f32 scalar)
//
// Buffers:
//   0: input  — float [count]
//   1: output — float [count]
//   2: params — { float scalar; uint count; }
// --------------------------------------------------------------------------

kernel void scalar_mul_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant ScalarMulParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    output[gid] = input[gid] * params.scalar;
}

// --------------------------------------------------------------------------
// embedding_gather_scale_f32 — Gather one embedding row and scale it.
//
// out[i] = embed_table[token_id * hidden_size + i] * scale
//
// Buffers:
//   0: embed_table — float [vocab_size * hidden_size]
//   1: output      — float [hidden_size]
//   2: params      — { float scale; uint hidden_size; uint token_id; }
// --------------------------------------------------------------------------

struct EmbedGatherScaleParams {
    float scale;
    uint hidden_size;
    uint token_id;
};

kernel void embedding_gather_scale_f32(
    device const float* embed_table [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    constant EmbedGatherScaleParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.hidden_size) return;
    uint offset = params.token_id * params.hidden_size + gid;
    output[gid] = embed_table[offset] * params.scale;
}

// --------------------------------------------------------------------------
// embedding_gather_scale_batch_f32 — Gather N embedding rows and scale them.
//
// output[tok * hidden_size + i] = embed_table[token_ids[tok] * hidden_size + i] * scale
//
// Buffers:
//   0: embed_table — float [vocab_size * hidden_size]
//   1: token_ids   — uint  [n_tokens]
//   2: output      — float [n_tokens * hidden_size]
//   3: params      — { float scale; uint hidden_size; uint n_tokens; }
//
// Grid: 2D (hidden_size, n_tokens) — one thread per element.
// --------------------------------------------------------------------------

struct EmbedGatherScaleBatchParams {
    float scale;
    uint hidden_size;
    uint n_tokens;
};

kernel void embedding_gather_scale_batch_f32(
    device const float* embed_table [[buffer(0)]],
    device const uint*  token_ids   [[buffer(1)]],
    device float*       output      [[buffer(2)]],
    constant EmbedGatherScaleBatchParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint tok = gid.y;
    if (i >= params.hidden_size || tok >= params.n_tokens) return;
    const uint token_id = token_ids[tok];
    const uint src = token_id * params.hidden_size + i;
    const uint dst = tok * params.hidden_size + i;
    output[dst] = embed_table[src] * params.scale;
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

// --------------------------------------------------------------------------
// permute_021_f32 — Transpose [A, B, C] -> [B, A, C] for float32
//
// Same operation as permute_021_bf16 but for F32 buffers.
// Used for prefill Q/K/V layout conversion:
//   [seq_len, n_heads, head_dim] <-> [n_heads, seq_len, head_dim]
//
// Reuses Permute021Params (defined above permute_021_bf16).
//
// Grid: (C, B, A), each thread copies one element
// --------------------------------------------------------------------------

kernel void permute_021_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant Permute021Params& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint b = gid.y;
    const uint a = gid.z;

    if (a >= params.dim_a || b >= params.dim_b || c >= params.dim_c) return;

    const uint in_idx  = a * (params.dim_b * params.dim_c) + b * params.dim_c + c;
    const uint out_idx = b * (params.dim_a * params.dim_c) + a * params.dim_c + c;
    output[out_idx] = input[in_idx];
}

// --------------------------------------------------------------------------
// permute_021_bf16_to_f32 — fused permute + dtype cast.
//
// Combines the two post-FA SDPA-output passes (permute_021_bf16 then
// cast_bf16_to_f32) into one global-memory pass, halving bandwidth on the
// [n_heads, seq_len, head_dim] tensor and removing one dispatch per layer.
// Wave P4.10.
//
// Buffers:
//   0: input  — bfloat [A * B * C]
//   1: output — float  [B * A * C]
//   2: params — { uint dim_a; uint dim_b; uint dim_c; }
//
// Grid: (C, B, A) — same as permute_021_bf16; each thread reads one bf16
// element and writes one f32 element at the permuted position.
// --------------------------------------------------------------------------
kernel void permute_021_bf16_to_f32(
    device const bfloat* input  [[buffer(0)]],
    device float*        output [[buffer(1)]],
    constant Permute021Params& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint c = gid.x;
    const uint b = gid.y;
    const uint a = gid.z;

    if (a >= params.dim_a || b >= params.dim_b || c >= params.dim_c) return;

    const uint in_idx  = a * (params.dim_b * params.dim_c) + b * params.dim_c + c;
    const uint out_idx = b * (params.dim_a * params.dim_c) + a * params.dim_c + c;

    output[out_idx] = static_cast<float>(input[in_idx]);
}

// --------------------------------------------------------------------------
// transpose_last2_bf16 — swap the last two axes of a 3D bf16 tensor.
//
//   input:  bfloat [A, B, C] row-major
//   output: bfloat [A, C, B] row-major
//
// Used by hf2q's non-flash-attention prefill path to materialise V_t
// (V transposed over seq↔hd) so the scores@V matmul can consume it at
// the tile geometry the dense_mm_bf16_tensor kernel expects.
//
// Grid: (B, C, A) — each thread copies one element.  Threadgroup shape
// is a divisor of (B, C); typical dispatch uses (16, 16, 1) threadgroups.
//
// Buffers:
//   0: input  — bfloat [A * B * C]
//   1: output — bfloat [A * C * B]
//   2: params — Permute021Params { dim_a, dim_b, dim_c } (shared struct)
// --------------------------------------------------------------------------
kernel void transpose_last2_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant Permute021Params& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint b = gid.x;
    const uint c = gid.y;
    const uint a = gid.z;

    if (a >= params.dim_a || b >= params.dim_b || c >= params.dim_c) return;

    const uint in_idx  = a * (params.dim_b * params.dim_c) + b * params.dim_c + c;
    // output layout: [A, C, B].  dim_c rows of length dim_b per slice.
    const uint out_idx = a * (params.dim_c * params.dim_b) + c * params.dim_b + b;

    output[out_idx] = input[in_idx];
}
