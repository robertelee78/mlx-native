#include <metal_stdlib>
using namespace metal;

/// Generic strided copy for making tensors contiguous.
///
/// Converts a strided 2D tensor to a contiguous layout:
///   dst[row * cols + col] = src[row * stride_row + col * stride_col]
///
/// Buffer layout:
///   buffer(0): src    — float (strided layout)
///   buffer(1): dst    — float (contiguous output)
///   buffer(2): params — uint [4] — {rows, cols, stride_row, stride_col}
///
/// Grid:        (cols, rows, 1)
/// Threadgroup: (min(256, cols), 1, 1)

struct StridedCopyParams {
    uint rows;
    uint cols;
    uint stride_row;
    uint stride_col;
};

kernel void strided_copy_f32(
    device const float*         src    [[buffer(0)]],
    device float*               dst    [[buffer(1)]],
    constant StridedCopyParams& params [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint col = pos.x;
    const uint row = pos.y;

    if (col >= params.cols || row >= params.rows) return;

    uint src_idx = row * params.stride_row + col * params.stride_col;
    uint dst_idx = row * params.cols + col;
    dst[dst_idx] = src[src_idx];
}
