#include <metal_stdlib>
using namespace metal;

/// Find the index of the maximum value in a float array.
///
/// Uses parallel reduction in threadgroup shared memory.  Each thread scans
/// a strided chunk of the input to find a local (value, index) pair, then all
/// threads cooperate in a tree reduction to find the global maximum.
///
/// Algorithm:
///   1. Each thread i scans elements i, i+tg_size, i+2*tg_size, ... to find
///      its local max and the index where it occurred.
///   2. Results are stored in threadgroup shared memory (interleaved float/uint).
///   3. Tree reduction (stride = tg_size/2 down to 1) finds the global max.
///   4. Thread 0 writes out_index and out_value.
///
/// For vocab_size=262144 and tg_size=1024: each thread scans 256 elements,
/// then 10 reduction steps.  Output is a single (index, value) pair —
/// 4+4 = 8 bytes instead of the full 1MB logits readback.
///
/// Buffer layout:
///   buffer(0): input     — float array [n_elements]
///   buffer(1): out_index — uint  [1] — index of maximum element
///   buffer(2): out_value — float [1] — value of maximum element
///   buffer(3): params    — uint  [1] — n_elements
///
/// Threadgroup: (min(1024, next_power_of_two(n_elements)), 1, 1)
/// Grid:        (1, 1, 1) — single threadgroup
/// Shared mem:  tg_size * (sizeof(float) + sizeof(uint)) bytes at index 0
///
/// IMPORTANT: tg_size must be a power of 2 for the tree reduction to be
/// correct.  The Rust dispatch ensures this.

kernel void argmax_f32(
    device const float* input      [[buffer(0)]],
    device uint*        out_index  [[buffer(1)]],
    device float*       out_value  [[buffer(2)]],
    device const uint*  params     [[buffer(3)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared_vals [[threadgroup(0)]],
    threadgroup uint*  shared_idxs [[threadgroup(1)]]
) {
    const uint n_elements = params[0];

    // Phase 1: each thread finds its local (max_val, max_idx) over its chunk.
    float local_max = -INFINITY;
    uint  local_idx = 0;

    for (uint i = tid; i < n_elements; i += tg_size) {
        float v = input[i];
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Store in shared memory for reduction.
    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree reduction — keep whichever slot has the larger value.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = shared_vals[tid + stride];
            if (other_val > shared_vals[tid]) {
                shared_vals[tid] = other_val;
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the result.
    if (tid == 0) {
        out_value[0] = shared_vals[0];
        out_index[0] = shared_idxs[0];
    }
}
