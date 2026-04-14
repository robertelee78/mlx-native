#include <metal_stdlib>
using namespace metal;

// Lloyd-Max codebook for N(0,1) at 4-bit (16 centroids).
// Must match CODEBOOK_4BIT in turboquant.rs exactly.
constant float CODEBOOK_4BIT[16] = {
    -2.7325896f, -2.0690172f, -1.6180464f, -1.2562312f,
    -0.9423405f, -0.6567591f, -0.3880483f, -0.1283950f,
     0.1283950f,  0.3880483f,  0.6567591f,  0.9423405f,
     1.2562312f,  1.6180464f,  2.0690172f,  2.7325896f,
};

// Decision boundaries: midpoints of adjacent codebook centroids.
// BOUNDARIES_4BIT[i] = (CODEBOOK_4BIT[i] + CODEBOOK_4BIT[i+1]) / 2
constant float BOUNDARIES_4BIT[15] = {
    -2.4008034f,  // midpoint(-2.7325896, -2.0690172)
    -1.8435318f,  // midpoint(-2.0690172, -1.6180464)
    -1.4371388f,  // midpoint(-1.6180464, -1.2562312)
    -1.0992859f,  // midpoint(-1.2562312, -0.9423405)
    -0.7995498f,  // midpoint(-0.9423405, -0.6567591)
    -0.5224037f,  // midpoint(-0.6567591, -0.3880483)
    -0.2582217f,  // midpoint(-0.3880483, -0.1283950)
     0.0000000f,  // midpoint(-0.1283950,  0.1283950)
     0.2582217f,  // midpoint( 0.1283950,  0.3880483)
     0.5224037f,  // midpoint( 0.3880483,  0.6567591)
     0.7995498f,  // midpoint( 0.6567591,  0.9423405)
     1.0992859f,  // midpoint( 0.9423405,  1.2562312)
     1.4371388f,  // midpoint( 1.2562312,  1.6180464)
     1.8435318f,  // midpoint( 1.6180464,  2.0690172)
     2.4008034f,  // midpoint( 2.0690172,  2.7325896)
};

struct HadamardQuantizeParams {
    uint head_dim;        // 256 or 512 (must be a power of two)
    uint num_kv_heads;    // number of KV heads to process
    uint write_pos;       // position in cache to write to (pre-wrapping for global)
    uint cache_capacity;  // total cache capacity
    uint is_sliding;      // 1 for sliding window (ring buffer), 0 for global
};

// Quantize KV data from F32 source into nibble-packed TurboQuant format.
//
// Input:  src     [num_kv_heads, head_dim] F32 — one token's KV for all heads
// Output: packed  [num_kv_heads, cache_capacity, head_dim/2] u8 — nibble-packed indices
// Output: norms   [num_kv_heads, cache_capacity] F32 — per-position L2 norm scalar
//
// One threadgroup per KV head. head_dim threads per threadgroup.
// Shared memory layout: [0 .. head_dim) = data region, [head_dim .. 2*head_dim) = norm reduction.
kernel void hadamard_quantize_kv(
    device const float             *src    [[buffer(0)]],
    device       uint8_t           *packed [[buffer(1)]],
    device       float             *norms  [[buffer(2)]],
    constant HadamardQuantizeParams &params [[buffer(3)]],
    threadgroup  float             *shared  [[threadgroup(0)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]]
) {
    uint head_idx = tgid;
    uint head_dim = params.head_dim;

    if (head_idx >= params.num_kv_heads || tid >= head_dim) return;

    // 1. Load this head's F32 source data into shared memory.
    uint src_offset = head_idx * head_dim;
    shared[tid] = src[src_offset + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Fast Walsh-Hadamard Transform butterfly (same XOR pattern as hadamard.metal).
    for (uint h = 1; h < head_dim; h <<= 1) {
        uint partner = tid ^ h;
        if (partner > tid) {
            float a = shared[tid];
            float b = shared[partner];
            shared[tid]     = a + b;
            shared[partner] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Normalize FWHT output by 1/sqrt(head_dim).
    float val = shared[tid] * rsqrt(float(head_dim));
    shared[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Compute L2 norm of the FWHT-rotated vector via parallel reduction.
    //    Use the second half of shared memory [head_dim .. 2*head_dim) as scratch.
    threadgroup float *norm_scratch = shared + head_dim;
    norm_scratch[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = head_dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            norm_scratch[tid] += norm_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = sqrt(norm_scratch[0]);

    // 5. Normalize each coordinate to unit sphere.
    float unit_val = (norm > 1.0e-10f) ? (val / norm) : 0.0f;

    // 6. Scale to approximate N(0,1) domain for codebook lookup.
    //    A coordinate of a unit vector on S^{d-1} is ~N(0, 1/d), so multiply
    //    by sqrt(d) to shift to ~N(0,1).  This matches turboquant.rs line 325.
    float scaled = unit_val * sqrt(float(head_dim));

    // 7. Find nearest Lloyd-Max centroid via linear scan through decision boundaries.
    uint centroid_idx = 0;
    for (uint b = 0; b < 15; b++) {
        if (scaled > BOUNDARIES_4BIT[b]) {
            centroid_idx = b + 1;
        }
    }

    // 8. Write nibble-packed output.
    //    Layout: packed[head_idx * cache_capacity * (head_dim/2) + pos * (head_dim/2) + tid/2]
    //    Even tid → low nibble (bits 3:0); odd tid → high nibble (bits 7:4).
    //
    //    To avoid a race on the shared byte, even threads zero the byte and write
    //    the low nibble first; after a barrier odd threads OR in the high nibble.
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;

    uint packed_row_stride = head_dim / 2;
    uint byte_idx = head_idx * params.cache_capacity * packed_row_stride
                  + actual_pos * packed_row_stride
                  + tid / 2;

    if (tid % 2 == 0) {
        // Even thread owns the byte: clear it and write the low nibble.
        packed[byte_idx] = uint8_t(centroid_idx & 0xFu);
    }
    threadgroup_barrier(mem_flags::mem_device);   // device memory fence for packed[]

    if (tid % 2 == 1) {
        // Odd thread ORs in the high nibble.
        packed[byte_idx] |= uint8_t((centroid_idx & 0xFu) << 4);
    }

    // 9. Store the L2 norm scalar (only thread 0 per head needs to do this).
    if (tid == 0) {
        uint norm_idx = head_idx * params.cache_capacity + actual_pos;
        norms[norm_idx] = norm;
    }
}
