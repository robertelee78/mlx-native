#include <metal_stdlib>
using namespace metal;

/// Bitonic sort (descending) on a per-row basis.
///
/// Each threadgroup handles one row.  The indices array is initialized to
/// [0, 1, 2, ..., N-1] and then sorted so that values[indices[0]] >=
/// values[indices[1]] >= ...
///
/// For MoE routing N <= 128, so a single threadgroup with 128 threads suffices.
///
/// Buffer layout:
///   buffer(0): input  — float [batch_size, row_len] (values to sort by)
///   buffer(1): output — uint  [batch_size, row_len] (sorted indices)
///   buffer(2): params — uint  [2] — {row_len, batch_size}
///
/// Grid:        (1, batch_size, 1)
/// Threadgroup: (next_power_of_two(row_len), 1, 1)

struct ArgsortParams {
    uint row_len;
    uint batch_size;
};

kernel void argsort_desc_f32(
    device const float*        input  [[buffer(0)]],
    device uint*               output [[buffer(1)]],
    constant ArgsortParams&    params [[buffer(2)]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    uint row_id   [[threadgroup_position_in_grid]]
) {
    const uint row_len    = params.row_len;
    const uint batch_size = params.batch_size;

    if (row_id >= batch_size) return;

    // Pointers for this row.
    device const float* row_vals = input  + row_id * row_len;
    device uint*        row_out  = output + row_id * row_len;

    // Threadgroup shared memory for values and indices.
    // Allocated to next power of two of row_len.
    threadgroup float shared_vals[256];
    threadgroup uint  shared_idxs[256];

    // Initialize: each thread loads one element (pad with -INF for unused slots).
    if (tid < row_len) {
        shared_vals[tid] = row_vals[tid];
        shared_idxs[tid] = tid;
    } else {
        shared_vals[tid] = -INFINITY;
        shared_idxs[tid] = tid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort — descending order.
    // We sort `tg_size` elements (power-of-two padded).
    for (uint k = 2; k <= tg_size; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            uint ixj = tid ^ j;
            if (ixj > tid) {
                // Determine sort direction for this subsequence.
                bool ascending = ((tid & k) != 0);
                float a = shared_vals[tid];
                float b = shared_vals[ixj];
                // For descending: swap if a < b when we want descending.
                bool should_swap;
                if (ascending) {
                    // ascending subsequence: swap if a > b
                    should_swap = (a > b) || (a == b && shared_idxs[tid] > shared_idxs[ixj]);
                } else {
                    // descending subsequence: swap if a < b
                    should_swap = (a < b) || (a == b && shared_idxs[tid] < shared_idxs[ixj]);
                }
                if (should_swap) {
                    shared_vals[tid] = b;
                    shared_vals[ixj] = a;
                    uint tmp = shared_idxs[tid];
                    shared_idxs[tid] = shared_idxs[ixj];
                    shared_idxs[ixj] = tmp;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write sorted indices back to output (only valid elements).
    if (tid < row_len) {
        row_out[tid] = shared_idxs[tid];
    }
}
