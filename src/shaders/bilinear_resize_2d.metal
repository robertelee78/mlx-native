// bilinear_resize_2d_f32 — Metal shader for ADR-021 K2.
//
// Antialiased bilinear resize (triangle-filter with support =
// max(1, 1/sf)) of a [H_src, W_src, C] row-major f32 tensor into
// a [H_dst, W_dst, C] row-major f32 tensor. Mirrors the CPU oracle
// `qwen3vl_resize_position_embeddings_bilinear` exactly:
//   - sample-coord mapping: `(dst + 0.5) / sf - 0.5` (PyTorch
//     align_corners=False / pixel_offset=0.5).
//   - antialias: triangle filter with support = max(1, 1/sf), sums
//     ALL contributions in the support window, then renormalizes by
//     total weight.
//   - For sf >= 1 (upsampling), support degenerates to 1 → 4-tap
//     bilinear; for sf < 1 (downsampling), support > 1 → wider
//     low-pass filter — load-bearing semantic match to llama.cpp's
//     `BILINEAR | ANTIALIAS` mode.
//
// Reference: /opt/llama.cpp/ggml/src/ggml-cpu/ops.cpp:7578-7637
// (the C++ source the CPU oracle ports). For the Qwen3-VL fixture
// trained_n == target_n on both axes so the fast path of the
// general formula collapses to pass-through (sy=floor(y), weight=1)
// — verified.
//
// Each thread emits one output element at (y_dst, x_dst, c). One
// thread per output element keeps the per-thread loop short for the
// canonical Qwen3-VL window sizes (sf ∈ [0.5, 4]) and avoids
// threadgroup-shared reduction.

#include <metal_stdlib>
using namespace metal;

// Must match `GpuBilinearResize2dParams` in src/ops/bilinear_resize_2d.rs.
struct BilinearResize2dParams {
    uint trained_n;       // edge length of the (square) source grid
    uint target_n_x;
    uint target_n_y;
    uint n_embd;          // channel count C
    float sf_x;           // target_n_x / trained_n
    float sf_y;           // target_n_y / trained_n
    float support_x;      // max(1.0, 1.0 / sf_x)
    float support_y;      // max(1.0, 1.0 / sf_y)
    float invscale_x;     // 1.0 / support_x
    float invscale_y;     // 1.0 / support_y
};

// Buffers:
//   0: params       — BilinearResize2dParams
//   1: src_table    — float [trained_n * trained_n * n_embd] row-major
//   2: dst_table    — float [target_n_y * target_n_x * n_embd] row-major
kernel void bilinear_resize_2d_f32(
    constant BilinearResize2dParams& params [[buffer(0)]],
    device const float*              src    [[buffer(1)]],
    device float*                    dst    [[buffer(2)]],
    uint                             gid    [[thread_position_in_grid]]
) {
    const uint total = params.target_n_y * params.target_n_x * params.n_embd;
    if (gid >= total) {
        return;
    }
    const uint trained = params.trained_n;
    const uint target_x = params.target_n_x;
    const uint n_embd = params.n_embd;

    const uint nx_c = target_x * n_embd;
    const uint y_dst = gid / nx_c;
    const uint within_y = gid - y_dst * nx_c;
    const uint x_dst = within_y / n_embd;
    const uint c = within_y - x_dst * n_embd;

    // Source-coord mapping (align_corners=False / pixel_offset=0.5).
    const float pixel_offset = 0.5f;
    const float y = ((float)y_dst + pixel_offset) / params.sf_y;
    const float x = ((float)x_dst + pixel_offset) / params.sf_x;

    int y_min = (int)max(y - params.support_y + pixel_offset, 0.0f);
    int y_max = (int)min(y + params.support_y + pixel_offset, (float)trained);
    int x_min = (int)max(x - params.support_x + pixel_offset, 0.0f);
    int x_max = (int)min(x + params.support_x + pixel_offset, (float)trained);

    float acc = 0.0f;
    float total_weight = 0.0f;
    for (int sy = y_min; sy < y_max; ++sy) {
        const float wy = max(1.0f - fabs(((float)sy - y + pixel_offset) * params.invscale_y), 0.0f);
        for (int sx = x_min; sx < x_max; ++sx) {
            const float wx = max(1.0f - fabs(((float)sx - x + pixel_offset) * params.invscale_x), 0.0f);
            const float w = wx * wy;
            if (w > 0.0f) {
                const uint src_idx = (uint)sy * trained * n_embd + (uint)sx * n_embd + c;
                acc += src[src_idx] * w;
                total_weight += w;
            }
        }
    }
    if (total_weight > 0.0f) {
        dst[gid] = acc / total_weight;
    } else {
        dst[gid] = 0.0f;
    }
}
