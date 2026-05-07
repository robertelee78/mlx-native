#include <metal_stdlib>
using namespace metal;

/// Slice a column range out of a 2D row-major tensor.
///
///   `output[r, c] = input[r, start_col + c]`     for 0 ≤ c < out_cols
///
/// Buffer layout:
///   buffer(0): input  — float[rows, in_cols]
///   buffer(1): output — float[rows, out_cols]
///   buffer(2): params — uint[3]: (in_cols, out_cols, start_col)
///
/// Grid: 2D (out_cols, rows); one thread per output element.
kernel void slice_2d_cols_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    device const uint  *params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint in_cols  = params[0];
    const uint out_cols = params[1];
    const uint start    = params[2];
    const uint col = tid.x;
    const uint row = tid.y;
    if (col >= out_cols || row >= grid_size.y) {
        return;
    }
    output[row * out_cols + col] = input[row * in_cols + start + col];
}

/// Copy a 2D source tensor into a column slab of a 2D destination
/// tensor.  The destination must be pre-zeroed (or pre-populated)
/// by the caller — this kernel writes ONLY the slab
/// `dst[:, start_col_in_dst .. start_col_in_dst + src_cols]`.
///
/// Used to implement `concat_along_cols` by calling this kernel
/// once per source slab into a single pre-zeroed output.
///
/// Buffer layout:
///   buffer(0): src    — float[rows, src_cols]
///   buffer(1): dst    — float[rows, dst_cols]
///   buffer(2): params — uint[3]: (src_cols, dst_cols, start_col_in_dst)
///
/// Grid: 2D (src_cols, rows); one thread per source element.
kernel void copy_2d_cols_into_f32(
    device const float *src    [[buffer(0)]],
    device float       *dst    [[buffer(1)]],
    device const uint  *params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    const uint src_cols = params[0];
    const uint dst_cols = params[1];
    const uint start    = params[2];
    const uint col = tid.x;
    const uint row = tid.y;
    if (col >= src_cols || row >= grid_size.y) {
        return;
    }
    dst[row * dst_cols + start + col] = src[row * src_cols + col];
}
