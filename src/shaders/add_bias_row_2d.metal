// add_bias_row_2d_f32 — Metal shader for ADR-021 K1's bias broadcast.
//
// Computes `out[m, n] = a[m, n] + bias[n]` over a `[M, N]` row-major
// f32 matrix. Used to land the patch-embed bias atop the dual-stem
// accumulator (qwen3vl.cpp:41-43 equivalent in the GPU port).
//
// Independent of the existing `elementwise.metal` to keep ADR-021's
// surface from colliding with other concurrent edits there.

#include <metal_stdlib>
using namespace metal;

// Must match `GpuAddBiasRow2dParams` in src/ops/add_bias_row_2d.rs.
struct AddBiasRow2dParams {
    uint m;
    uint n;
};

// Buffers:
//   0: params — AddBiasRow2dParams
//   1: a      — float [M * N] row-major (in/out: read first, then overwrite)
//   2: bias   — float [N]
//   3: output — float [M * N] row-major
kernel void add_bias_row_2d_f32(
    constant AddBiasRow2dParams& params [[buffer(0)]],
    device const float*          a      [[buffer(1)]],
    device const float*          bias   [[buffer(2)]],
    device float*                output [[buffer(3)]],
    uint                         gid    [[thread_position_in_grid]]
) {
    const uint total = params.m * params.n;
    if (gid >= total) {
        return;
    }
    const uint col = gid - (gid / params.n) * params.n;
    output[gid] = a[gid] + bias[col];
}
