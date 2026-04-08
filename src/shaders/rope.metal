#include <metal_stdlib>
using namespace metal;

/// Rotary Position Embedding (RoPE) kernel.
///
/// Applies rotation to pairs of elements (x[2i], x[2i+1]) using the angle:
///   angle = position * theta^(-2i / head_dim)
///
/// Buffer layout:
///   buffer(0): input   — float array of shape [seq_len, head_dim]
///   buffer(1): output  — float array of shape [seq_len, head_dim]
///   buffer(2): params  — float4: (theta, head_dim_f, seq_len_f, 0)
///   buffer(3): positions — uint array of shape [seq_len]
///
/// Each thread processes one pair (2 elements) at coordinate (pair_idx, seq_idx).
/// Grid: (head_dim / 2, seq_len, 1)

kernel void rope_f32(
    device const float *input      [[buffer(0)]],
    device float       *output     [[buffer(1)]],
    device const float *params     [[buffer(2)]],
    device const uint  *positions  [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint pair_idx  = tid.x;             // which pair within the head
    const uint seq_idx   = tid.y;             // which position in the sequence
    const float theta    = params[0];
    const uint head_dim  = uint(params[1]);
    const uint half_dim  = head_dim / 2;

    // Bounds check
    if (pair_idx >= half_dim) return;

    const uint pos = positions[seq_idx];

    // Compute the rotation angle:
    //   freq = theta^(-2 * pair_idx / head_dim)
    //   angle = pos * freq
    const float dim_ratio = float(2 * pair_idx) / float(head_dim);
    const float freq = 1.0f / pow(theta, dim_ratio);
    const float angle = float(pos) * freq;

    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    // Index into the flat [seq_len, head_dim] array
    const uint base = seq_idx * head_dim + 2 * pair_idx;
    const float x0 = input[base];
    const float x1 = input[base + 1];

    // Apply 2D rotation
    output[base]     = x0 * cos_a - x1 * sin_a;
    output[base + 1] = x0 * sin_a + x1 * cos_a;
}

kernel void rope_f16(
    device const half  *input      [[buffer(0)]],
    device half        *output     [[buffer(1)]],
    device const float *params     [[buffer(2)]],
    device const uint  *positions  [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint pair_idx  = tid.x;
    const uint seq_idx   = tid.y;
    const float theta    = params[0];
    const uint head_dim  = uint(params[1]);
    const uint half_dim  = head_dim / 2;

    if (pair_idx >= half_dim) return;

    const uint pos = positions[seq_idx];

    const float dim_ratio = float(2 * pair_idx) / float(head_dim);
    const float freq = 1.0f / pow(theta, dim_ratio);
    const float angle = float(pos) * freq;

    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    const uint base = seq_idx * head_dim + 2 * pair_idx;
    // Promote to f32 for computation, store back as f16
    const float x0 = float(input[base]);
    const float x1 = float(input[base + 1]);

    output[base]     = half(x0 * cos_a - x1 * sin_a);
    output[base + 1] = half(x0 * sin_a + x1 * cos_a);
}

kernel void rope_bf16(
    device const bfloat *input      [[buffer(0)]],
    device bfloat       *output     [[buffer(1)]],
    device const float  *params     [[buffer(2)]],
    device const uint   *positions  [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint pair_idx  = tid.x;
    const uint seq_idx   = tid.y;
    const float theta    = params[0];
    const uint head_dim  = uint(params[1]);
    const uint half_dim  = head_dim / 2;

    if (pair_idx >= half_dim) return;

    const uint pos = positions[seq_idx];

    const float dim_ratio = float(2 * pair_idx) / float(head_dim);
    const float freq = 1.0f / pow(theta, dim_ratio);
    const float angle = float(pos) * freq;

    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    const uint base = seq_idx * head_dim + 2 * pair_idx;
    // Promote to f32 for computation, store back as bfloat16
    const float x0 = static_cast<float>(input[base]);
    const float x1 = static_cast<float>(input[base + 1]);

    output[base]     = bfloat(x0 * cos_a - x1 * sin_a);
    output[base + 1] = bfloat(x0 * sin_a + x1 * cos_a);
}

/// Neox/split-convention Rotary Position Embedding for bfloat16.
///
/// Unlike the standard RoPE which pairs (d[2i], d[2i+1]), the Neox convention
/// pairs (d[i], d[i + half_rope_dim]).  This is required for Gemma 4.
///
/// Supports partial rotary: only the first rope_dim dimensions are rotated,
/// the remaining (head_dim - rope_dim) are passed through unchanged.
///
/// Buffer layout:
///   buffer(0): input     — bfloat array, shape [n_rows, head_dim] (n_rows = seq_len * n_heads)
///   buffer(1): output    — bfloat array, same shape
///   buffer(2): params    — float4: (theta, head_dim_f, rope_dim_f, 0)
///   buffer(3): positions — uint array of shape [seq_len]
///   buffer(4): rope_params — uint2: (n_heads, 0)
///
/// Grid: (rope_dim / 2, n_rows, 1)
/// Each thread processes one pair: (input[row, pair_idx], input[row, pair_idx + half_rope])

kernel void rope_neox_bf16(
    device const bfloat *input       [[buffer(0)]],
    device bfloat       *output      [[buffer(1)]],
    device const float  *params      [[buffer(2)]],
    device const uint   *positions   [[buffer(3)]],
    device const uint   *rope_params [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint pair_idx   = tid.x;          // which pair within rope_dim/2
    const uint row_idx    = tid.y;          // which row (flattened seq_len * n_heads)
    const float theta     = params[0];
    const uint head_dim   = uint(params[1]);
    const uint rope_dim   = uint(params[2]);
    const uint half_rope  = rope_dim / 2;
    const uint n_heads    = rope_params[0];

    if (pair_idx >= half_rope) return;

    // Determine seq_idx from the row: row_idx = seq_idx * n_heads + head_idx
    const uint seq_idx = row_idx / n_heads;
    const uint pos = positions[seq_idx];

    // Compute the rotation angle
    // NOTE: denominator is head_dim (not rope_dim) to match mlx-lm's
    // ProportionalRoPE which computes: exponents = arange(0, rotated_dims, 2) / dims
    // where dims = full head_dim.  This ensures correct frequency scaling for
    // partial-rotary global attention layers (e.g., 128 of 512 dims rotated).
    const float dim_ratio = float(2 * pair_idx) / float(head_dim);
    const float freq = 1.0f / pow(theta, dim_ratio);
    const float angle = float(pos) * freq;

    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    // Neox/split indexing: pair (d[pair_idx], d[pair_idx + half_rope])
    const uint base = row_idx * head_dim;
    const float x0 = static_cast<float>(input[base + pair_idx]);
    const float x1 = static_cast<float>(input[base + pair_idx + half_rope]);

    output[base + pair_idx]             = bfloat(x0 * cos_a - x1 * sin_a);
    output[base + pair_idx + half_rope] = bfloat(x1 * cos_a + x0 * sin_a);

    // Pass through non-rotated dimensions (only need one thread to handle this)
    // Each thread copies its corresponding element beyond rope_dim if it exists
    // Thread pair_idx handles element rope_dim + pair_idx (and + half_rope if in range)
    if (pair_idx < (head_dim - rope_dim)) {
        output[base + rope_dim + pair_idx] = input[base + rope_dim + pair_idx];
    }
    if (pair_idx + half_rope < (head_dim - rope_dim)) {
        output[base + rope_dim + pair_idx + half_rope] = input[base + rope_dim + pair_idx + half_rope];
    }
}
