#include <metal_stdlib>
using namespace metal;

/// 2-D NeoX-convention Rotary Position Embedding for ViT vision towers.
///
/// Used by Gemma 4 Vision (gemma4v). The head_dim is split in half;
/// the first half rotates by `pos_x[p]`, the second half by `pos_y[p]`.
/// Each half is rotated NeoX-style with its OWN d-axis schedule:
///   pair (d[i], d[i + d_quarter]) for i ∈ [0, d_quarter)  (first half)
///   pair (d[d_half + i], d[d_half + i + d_quarter])       (second half)
/// where d_half = head_dim / 2 and d_quarter = head_dim / 4.
///
/// Rotation angle for index i within either half:
///   theta_i = base ^ (-2 * i / d_half)
///   angle   = position * theta_i
/// (denominator is d_half, NOT head_dim — each half is its own rotation
/// domain, mirroring `ggml_rope_ext(..., n_dims = n_dim/2, ...)` in
/// `/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp:59-86`.)
///
/// Buffer layout:
///   buffer(0): input    — float / bfloat array, shape [n_rows, head_dim]
///   buffer(1): output   — same shape, same dtype
///   buffer(2): params   — float4: (theta_base, head_dim_f, n_heads_f, 0)
///   buffer(3): pos_x    — uint array of shape [seq_len]
///   buffer(4): pos_y    — uint array of shape [seq_len]
///
/// Grid: (d_quarter, n_rows, 1) where d_quarter = head_dim / 4.
/// Each thread rotates ONE pair in the first half and ONE pair in the second half.
/// The row layout is row_idx = seq_idx * n_heads + head_idx.

kernel void vision_2d_rope_f32(
    device const float *input    [[buffer(0)]],
    device float       *output   [[buffer(1)]],
    device const float *params   [[buffer(2)]],
    device const uint  *pos_x    [[buffer(3)]],
    device const uint  *pos_y    [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint i         = tid.x;             // pair index within a half [0, d_quarter)
    const uint row_idx   = tid.y;             // [0, n_rows)
    const float theta    = params[0];
    const uint head_dim  = uint(params[1]);
    const uint n_heads   = uint(params[2]);
    const uint d_half    = head_dim / 2;
    const uint d_quarter = d_half / 2;

    if (i >= d_quarter) return;

    // Determine seq_idx from the row: row_idx = seq_idx * n_heads + head_idx
    const uint seq_idx = row_idx / n_heads;
    const uint p_x = pos_x[seq_idx];
    const uint p_y = pos_y[seq_idx];

    // Per-axis theta uses d_half as denominator (each half is its own
    // rotation domain). Same schedule used for first-half (with pos_x)
    // and second-half (with pos_y).
    const float dim_ratio = float(2 * i) / float(d_half);
    const float freq      = 1.0f / pow(theta, dim_ratio);
    const float angle_x   = float(p_x) * freq;
    const float angle_y   = float(p_y) * freq;
    const float cx = cos(angle_x);
    const float sx = sin(angle_x);
    const float cy = cos(angle_y);
    const float sy = sin(angle_y);

    const uint base = row_idx * head_dim;

    // First half: NeoX pair (i, i + d_quarter) within [0, d_half)
    {
        const float x0 = input[base + i];
        const float x1 = input[base + i + d_quarter];
        output[base + i]              = x0 * cx - x1 * sx;
        output[base + i + d_quarter]  = x0 * sx + x1 * cx;
    }
    // Second half: NeoX pair (d_half + i, d_half + i + d_quarter) within [d_half, head_dim)
    {
        const float y0 = input[base + d_half + i];
        const float y1 = input[base + d_half + i + d_quarter];
        output[base + d_half + i]                = y0 * cy - y1 * sy;
        output[base + d_half + i + d_quarter]    = y0 * sy + y1 * cy;
    }
}

kernel void vision_2d_rope_bf16(
    device const bfloat *input    [[buffer(0)]],
    device bfloat       *output   [[buffer(1)]],
    device const float  *params   [[buffer(2)]],
    device const uint   *pos_x    [[buffer(3)]],
    device const uint   *pos_y    [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint i         = tid.x;
    const uint row_idx   = tid.y;
    const float theta    = params[0];
    const uint head_dim  = uint(params[1]);
    const uint n_heads   = uint(params[2]);
    const uint d_half    = head_dim / 2;
    const uint d_quarter = d_half / 2;

    if (i >= d_quarter) return;

    const uint seq_idx = row_idx / n_heads;
    const uint p_x = pos_x[seq_idx];
    const uint p_y = pos_y[seq_idx];

    const float dim_ratio = float(2 * i) / float(d_half);
    const float freq      = 1.0f / pow(theta, dim_ratio);
    const float angle_x   = float(p_x) * freq;
    const float angle_y   = float(p_y) * freq;
    const float cx = cos(angle_x);
    const float sx = sin(angle_x);
    const float cy = cos(angle_y);
    const float sy = sin(angle_y);

    const uint base = row_idx * head_dim;

    {
        const float x0 = static_cast<float>(input[base + i]);
        const float x1 = static_cast<float>(input[base + i + d_quarter]);
        output[base + i]             = bfloat(x0 * cx - x1 * sx);
        output[base + i + d_quarter] = bfloat(x0 * sx + x1 * cx);
    }
    {
        const float y0 = static_cast<float>(input[base + d_half + i]);
        const float y1 = static_cast<float>(input[base + d_half + i + d_quarter]);
        output[base + d_half + i]                = bfloat(y0 * cy - y1 * sy);
        output[base + d_half + i + d_quarter]    = bfloat(y0 * sy + y1 * cy);
    }
}
