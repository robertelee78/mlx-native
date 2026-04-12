#include <metal_stdlib>
using namespace metal;

/// Gather rows from a 2D source tensor by index.
///
/// output[i, :] = src[indices[i], :]
///
/// Each thread copies one element of one output row.
///
/// Buffer layout:
///   buffer(0): src     — float [src_rows, row_width]
///   buffer(1): indices — uint  [n_indices]
///   buffer(2): output  — float [n_indices, row_width]
///   buffer(3): params  — uint  [3] — {row_width, n_indices, src_rows}
///
/// Grid:        (row_width, n_indices, 1)
/// Threadgroup: (min(256, row_width), 1, 1)

struct GatherParams {
    uint row_width;
    uint n_indices;
    uint src_rows;
};

kernel void gather_f32(
    device const float*     src     [[buffer(0)]],
    device const uint*      indices [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant GatherParams&  params  [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint col = pos.x;
    const uint idx = pos.y;

    if (col >= params.row_width || idx >= params.n_indices) return;

    uint src_row = indices[idx];
    // Clamp to valid range to prevent out-of-bounds access.
    if (src_row >= params.src_rows) src_row = params.src_rows - 1;

    output[idx * params.row_width + col] = src[src_row * params.row_width + col];
}
