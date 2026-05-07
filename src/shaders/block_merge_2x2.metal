// block_merge_2x2_f32 — Metal shader for ADR-021 K4.
//
// 2×2 block-major reshape of a [ny, nx, n_embd] row-major f32 tensor
// into a [ny*nx, n_embd] row-major tensor. The dst patch index for
// src position (src_y, src_x) is:
//
//   by      = src_y / 2
//   bx      = src_x / 2
//   block_id = by * (nx/2) + bx
//   within   = (src_y & 1) * 2 + (src_x & 1)
//   dst_p    = block_id * 4 + within
//
// matching the permutation in
// `vit_gpu_qwen3vl.rs::qwen3vl_2x2_block_merge_reshape`. Pure
// permutation copy — no FP arithmetic — so AC-1 holds byte-identically.

#include <metal_stdlib>
using namespace metal;

// Must match `GpuBlockMerge2x2Params` in src/ops/block_merge_2x2.rs.
struct BlockMerge2x2Params {
    uint nx;
    uint ny;
    uint n_embd;
    uint half_x;     // nx / 2
};

// Buffers:
//   0: params — BlockMerge2x2Params
//   1: input  — float [ny * nx * n_embd] row-major
//   2: output — float [ny * nx * n_embd] row-major (block-merged order)
kernel void block_merge_2x2_f32(
    constant BlockMerge2x2Params& params [[buffer(0)]],
    device const float*           input  [[buffer(1)]],
    device float*                 output [[buffer(2)]],
    uint                          gid    [[thread_position_in_grid]]
) {
    const uint total = params.ny * params.nx * params.n_embd;
    if (gid >= total) {
        return;
    }
    const uint n_embd = params.n_embd;
    const uint nx = params.nx;
    const uint half_x = params.half_x;

    const uint patch_idx = gid / n_embd;
    const uint c = gid - patch_idx * n_embd;

    const uint block_id = patch_idx / 4;
    const uint within = patch_idx - block_id * 4;
    const uint by = block_id / half_x;
    const uint bx = block_id - by * half_x;
    const uint y_in = within >> 1;
    const uint x_in = within & 1u;

    const uint src_y = by * 2u + y_in;
    const uint src_x = bx * 2u + x_in;
    const uint src_idx = (src_y * nx + src_x) * n_embd + c;

    output[gid] = input[src_idx];
}
