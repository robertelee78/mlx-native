#include <metal_stdlib>
using namespace metal;

// In-place Fast Walsh-Hadamard Transform using shared memory butterfly pattern.
//
// One threadgroup per head. D threads per threadgroup (D = head_dim).
// Input/output: device float buffer of shape [num_heads, head_dim].
//
// The transform is normalized: multiply by 1/sqrt(D) so H·H = I.
//
// Butterfly pattern: log2(head_dim) stages.  In each stage h, thread tid and
// its XOR partner (tid ^ h) form a butterfly pair.  Only the thread whose
// tid < partner (i.e. partner > tid) performs the update so each pair is
// processed exactly once.

kernel void hadamard_transform(
    device float       *data       [[buffer(0)]],
    constant uint      &head_dim   [[buffer(1)]],
    constant uint      &num_heads  [[buffer(2)]],
    threadgroup float  *shared     [[threadgroup(0)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]]
) {
    uint head_idx = tgid;
    if (head_idx >= num_heads) return;

    // Load this head's data into shared memory.
    uint base = head_idx * head_dim;
    if (tid < head_dim) {
        shared[tid] = data[base + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Butterfly stages: log2(head_dim) iterations.
    for (uint h = 1; h < head_dim; h <<= 1) {
        if (tid < head_dim) {
            uint partner = tid ^ h;
            if (partner > tid) {
                float a = shared[tid];
                float b = shared[partner];
                shared[tid]     = a + b;
                shared[partner] = a - b;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize by 1/sqrt(head_dim) and write back.
    if (tid < head_dim) {
        float scale = rsqrt(float(head_dim));
        data[base + tid] = shared[tid] * scale;
    }
}
