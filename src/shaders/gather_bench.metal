#include <metal_stdlib>
using namespace metal;

/// Kernel A: gather_bench_nibble
///
/// Simulates TurboQuant SDPA reads: unpacks 4-bit nibble indices from a
/// nibble-packed buffer, then gathers the corresponding centroid row.
///
/// Layout:
///   packed    : [capacity, head_dim/2]  uint8 — low nibble = even coord, high nibble = odd coord
///   centroids : [16, head_dim]          float — pre-rotated centroid table (4-bit = 16 entries)
///   out       : [capacity, head_dim]    float — gathered output
///
/// Grid: 2D — x = coord index (head_dim), y = position index (capacity)
/// Threadgroup: [256, 1, 1]
kernel void gather_bench_nibble(
    device const uint8_t* packed        [[buffer(0)]],
    constant float*        centroids    [[buffer(1)]],
    constant uint&         capacity     [[buffer(2)]],
    constant uint&         head_dim     [[buffer(3)]],
    device float*          out          [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint c = tid.x; // coordinate index within head
    uint p = tid.y; // position (token) index

    if (p >= capacity || c >= head_dim) return;

    // Extract 4-bit index from nibble-packed buffer.
    // Low nibble  → even coordinate (c % 2 == 0)
    // High nibble → odd  coordinate (c % 2 == 1)
    uint byte_idx = p * (head_dim / 2u) + c / 2u;
    uint8_t byte  = packed[byte_idx];
    uint    idx   = (c % 2u == 0u) ? (byte & 0xFu) : ((byte >> 4u) & 0xFu);

    // Gather from centroid table and write output.
    out[p * head_dim + c] = centroids[idx * head_dim + c];
}

/// Kernel B: gather_bench_f16_seq
///
/// Baseline sequential F16 reads: reads every element of the F16 KV cache
/// and widens to F32 — this is the workload a standard F16 SDPA performs.
///
/// Layout:
///   cache : [capacity, head_dim]  half  — F16 KV cache
///   out   : [capacity, head_dim]  float — widened output
///
/// Grid: 2D — x = coord index (head_dim), y = position index (capacity)
/// Threadgroup: [256, 1, 1]
kernel void gather_bench_f16_seq(
    device const half* cache        [[buffer(0)]],
    constant uint&     capacity     [[buffer(1)]],
    constant uint&     head_dim     [[buffer(2)]],
    device float*      out          [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint c = tid.x;
    uint p = tid.y;

    if (p >= capacity || c >= head_dim) return;

    out[p * head_dim + c] = float(cache[p * head_dim + c]);
}
