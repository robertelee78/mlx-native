// im2col_2d_3ch_f32 — Metal shader for ADR-021 K1.
//
// Unfolds a [3, H, W] f32 row-major pixel buffer into a
// [num_patches, 3*p²] f32 row-major im2col matrix matching
// `patch_embed_forward_hw`'s inner-kernel iteration order
// (channel-major, dy-major, dx-major). The output is the
// `src1` operand of the two `dense_matmul_f32_f32_tensor`
// dispatches that replace the dual-conv patch embed.
//
// Reference: src/inference/vision/vit.rs::patch_embed_forward_hw
//   for ic in 0..3:
//     for dy in 0..p:
//       for dx in 0..p:
//         k = ic*p² + dy*p + dx
// where output row index `m = patch_y * nps_x + patch_x` and the
// column index `k` iterates the unfolded patch in (ic, dy, dx)
// order — matching how `dense_matmul_f32_f32_tensor` consumes the
// `src1` slice as `[M=num_patches, K=3*p²]` row-major.

#include <metal_stdlib>
using namespace metal;

// Must match `GpuIm2col2d3chParams` in src/ops/im2col_2d_3ch.rs.
struct Im2col2d3chParams {
    uint pixel_h;
    uint pixel_w;
    uint patch_size;
    uint nps_x;
    uint nps_y;
    uint k_total;        // = 3 * p²
    uint num_patches;    // = nps_x * nps_y
    uint _pad;
};

// Buffers:
//   0: params      — Im2col2d3chParams
//   1: pixels      — float [3 * pixel_h * pixel_w] row-major (channel-major)
//   2: output      — float [num_patches * k_total] row-major
kernel void im2col_2d_3ch_f32(
    constant Im2col2d3chParams& params [[buffer(0)]],
    device const float*         pixels [[buffer(1)]],
    device float*               output [[buffer(2)]],
    uint                        gid    [[thread_position_in_grid]]
) {
    const uint total = params.num_patches * params.k_total;
    if (gid >= total) {
        return;
    }
    const uint p = params.patch_size;
    const uint w = params.pixel_w;
    const uint h = params.pixel_h;
    const uint nps_x = params.nps_x;
    const uint k_total = params.k_total;
    const uint p2 = p * p;
    const uint hw = h * w;

    const uint patch_idx = gid / k_total;
    const uint k = gid - patch_idx * k_total;

    const uint ic = k / p2;
    const uint within = k - ic * p2;
    const uint dy = within / p;
    const uint dx = within - dy * p;

    const uint patch_y = patch_idx / nps_x;
    const uint patch_x = patch_idx - patch_y * nps_x;

    const uint src_y = patch_y * p + dy;
    const uint src_x = patch_x * p + dx;
    const uint src_idx = ic * hw + src_y * w + src_x;

    output[gid] = pixels[src_idx];
}
