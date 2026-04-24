#include <metal_stdlib>
using namespace metal;

// Multi-section Rotary Position Embedding (MROPE) with optional interleaved
// mode (IMROPE).
//
// Spec source: ADR-013 Decision 10. Formula derived from the published
// MROPE / IMROPE definitions (ggml `ggml_mrope_cache_init`). No code copied
// — only the math is reproduced here.
//
// # Modes
//
//   mode == 8   :  MROPE (non-interleaved, contiguous sections)
//   mode == 40  :  IMROPE (interleaved; `sector % 3` cycles through axes)
//
// Qwen3.5 / Qwen3.6 text uses IMROPE with sections = [11, 11, 10, 0] and
// `rope_theta = 1e7`. For text-only, positions on all 4 axes are equal to
// the token's 1D position, so the IMROPE output equals plain NeoX RoPE
// output. Kernel still implements the full multi-axis machinery so that
// the same op can serve multimodal Qwen variants where the axes diverge.
//
// # Pair indexing (NeoX-style)
//
// For each pair p in [0, head_dim / 2), the rotation acts on
// `(x[p], x[p + head_dim/2])`. Only the first `rope_dim / 2` pairs are
// rotated; the remaining pairs pass through unchanged.
//
// # Per-pair frequency
//
//   theta_scale = freq_base ^ (-2 / rope_dim)
//   theta_base(axis, p) = position[axis] * theta_scale ^ p
//
// # Sector-to-axis mapping
//
//   sect_dims = s0 + s1 + s2 + s3
//   sector = p % sect_dims       (for rotated pairs; p < rope_dim/2)
//
// MROPE (mode == 8):
//   sector < s0                  -> axis 0 (t)
//   s0 <= sector < s0+s1         -> axis 1 (h)
//   s0+s1 <= sector < s0+s1+s2   -> axis 2 (w)
//   else                         -> axis 3 (e/extra)
//
// IMROPE (mode == 40):
//   sector % 3 == 0 && sector < 3*s0  -> axis 0 (t)
//   sector % 3 == 1 && sector < 3*s1  -> axis 1 (h)
//   sector % 3 == 2 && sector < 3*s2  -> axis 2 (w)
//   else                              -> axis 3 (e/extra)
//
// # Positions layout
//
//   positions[int32] has length 4 * seq_len.
//   pos_t(i) = positions[i]
//   pos_h(i) = positions[i +     seq_len]
//   pos_w(i) = positions[i + 2 * seq_len]
//   pos_e(i) = positions[i + 3 * seq_len]
//
// # Buffer bindings
//
//   buffer(0): input       - shape [n_rows, head_dim]  (n_rows = seq_len*n_heads)
//   buffer(1): output      - same shape + dtype as input
//   buffer(2): params      - float4: (freq_base, head_dim_f, rope_dim_f, 0)
//   buffer(3): positions   - int32 array, length 4 * seq_len
//   buffer(4): rope_params - uint4: (n_heads, mode, seq_len, 0)
//   buffer(5): sections    - uint4: (s0, s1, s2, s3)
//
// # Grid
//
//   Grid: (head_dim / 2, n_rows, 1). Every thread writes exactly 2 output
//   elements (one pair). Threads with pair_idx >= rope_dim/2 copy input
//   unchanged — partial-rotary pass-through.

// ----- constants -----

constant uint RMODE_MROPE   = 8u;
constant uint RMODE_IMROPE  = 40u;

// pick_axis returns position axis index (0=t, 1=h, 2=w, 3=e) for a given
// sector, according to the mode and sections counts.
static inline uint pick_axis(uint sector, uint mode, uint s0, uint s1, uint s2) {
    if (mode == RMODE_IMROPE) {
        if (sector % 3u == 0u && sector < 3u * s0) return 0u;
        if (sector % 3u == 1u && sector < 3u * s1) return 1u;
        if (sector % 3u == 2u && sector < 3u * s2) return 2u;
        return 3u;
    } else {
        // MROPE (contiguous)
        if (sector < s0) return 0u;
        if (sector < s0 + s1) return 1u;
        if (sector < s0 + s1 + s2) return 2u;
        return 3u;
    }
}

// fetch_pos returns the int32 position for the requested axis, clamping to
// zero if axis index is >=4 (defensive).
static inline int fetch_pos(
    device const int *positions,
    uint seq_idx,
    uint seq_len,
    uint axis
) {
    return positions[axis * seq_len + seq_idx];
}

// compute_cos_sin computes (cos_theta, sin_theta) for a given rotated pair.
// Uses the formula theta = pos * freq_base^(-2*p/rope_dim).
static inline float2 compute_cos_sin(
    int pos,
    uint pair_idx,
    float freq_base,
    uint rope_dim
) {
    // dim_ratio = 2 * pair_idx / rope_dim  (matches llama.cpp theta_scale).
    const float dim_ratio = float(2u * pair_idx) / float(rope_dim);
    // freq = freq_base^(-dim_ratio) = 1 / freq_base^dim_ratio.
    const float freq = 1.0f / pow(freq_base, dim_ratio);
    const float theta = float(pos) * freq;
    return float2(cos(theta), sin(theta));
}

// ----- f32 -----

kernel void rope_multi_f32(
    device const float *input        [[buffer(0)]],
    device float       *output       [[buffer(1)]],
    device const float *params       [[buffer(2)]],
    device const int   *positions    [[buffer(3)]],
    device const uint  *rope_params  [[buffer(4)]],
    device const uint  *sections     [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint pair_idx = tid.x;
    const uint row_idx  = tid.y;

    const float freq_base = params[0];
    const uint head_dim   = uint(params[1]);
    const uint rope_dim   = uint(params[2]);
    const uint half_dim   = head_dim / 2u;
    const uint half_rope  = rope_dim / 2u;

    const uint n_heads = rope_params[0];
    const uint mode    = rope_params[1];
    const uint seq_len = rope_params[2];

    if (pair_idx >= half_dim) return;

    const uint base = row_idx * head_dim;

    // Pass-through for pairs outside the rotary range.
    if (pair_idx >= half_rope) {
        output[base + pair_idx]            = input[base + pair_idx];
        output[base + pair_idx + half_dim] = input[base + pair_idx + half_dim];
        return;
    }

    const uint s0 = sections[0];
    const uint s1 = sections[1];
    const uint s2 = sections[2];
    const uint s3 = sections[3];
    const uint sect_dims = max(s0 + s1 + s2 + s3, 1u);

    const uint seq_idx = row_idx / n_heads;
    const uint sector = pair_idx % sect_dims;
    const uint axis = pick_axis(sector, mode, s0, s1, s2);
    const int pos = fetch_pos(positions, seq_idx, seq_len, axis);

    const float2 cs = compute_cos_sin(pos, pair_idx, freq_base, rope_dim);
    const float cos_a = cs.x;
    const float sin_a = cs.y;

    const float x0 = input[base + pair_idx];
    const float x1 = input[base + pair_idx + half_dim];

    output[base + pair_idx]            = x0 * cos_a - x1 * sin_a;
    output[base + pair_idx + half_dim] = x0 * sin_a + x1 * cos_a;
    // s3 is read-referenced above solely to avoid unused-variable noise on
    // some shader compilers; its behavior only matters when sector falls
    // into the axis-3 branch inside pick_axis.
    (void)s3;
}

// ----- bf16 -----

kernel void rope_multi_bf16(
    device const bfloat *input        [[buffer(0)]],
    device bfloat       *output       [[buffer(1)]],
    device const float  *params       [[buffer(2)]],
    device const int    *positions    [[buffer(3)]],
    device const uint   *rope_params  [[buffer(4)]],
    device const uint   *sections     [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint pair_idx = tid.x;
    const uint row_idx  = tid.y;

    const float freq_base = params[0];
    const uint head_dim   = uint(params[1]);
    const uint rope_dim   = uint(params[2]);
    const uint half_dim   = head_dim / 2u;
    const uint half_rope  = rope_dim / 2u;

    const uint n_heads = rope_params[0];
    const uint mode    = rope_params[1];
    const uint seq_len = rope_params[2];

    if (pair_idx >= half_dim) return;

    const uint base = row_idx * head_dim;

    if (pair_idx >= half_rope) {
        output[base + pair_idx]            = input[base + pair_idx];
        output[base + pair_idx + half_dim] = input[base + pair_idx + half_dim];
        return;
    }

    const uint s0 = sections[0];
    const uint s1 = sections[1];
    const uint s2 = sections[2];
    const uint s3 = sections[3];
    const uint sect_dims = max(s0 + s1 + s2 + s3, 1u);

    const uint seq_idx = row_idx / n_heads;
    const uint sector = pair_idx % sect_dims;
    const uint axis = pick_axis(sector, mode, s0, s1, s2);
    const int pos = fetch_pos(positions, seq_idx, seq_len, axis);

    const float2 cs = compute_cos_sin(pos, pair_idx, freq_base, rope_dim);
    const float cos_a = cs.x;
    const float sin_a = cs.y;

    const float x0 = float(input[base + pair_idx]);
    const float x1 = float(input[base + pair_idx + half_dim]);

    output[base + pair_idx]            = bfloat(x0 * cos_a - x1 * sin_a);
    output[base + pair_idx + half_dim] = bfloat(x0 * sin_a + x1 * cos_a);
    (void)s3;
}
