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
