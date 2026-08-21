#include <metal_stdlib>
using namespace metal;

// Exact activation layout transform for Qwen fused Q/gate projection rows.
//
// Logical input:  [m, n_heads, 2 * head_dim] F32
// Logical outputs: [m, n_heads, head_dim] F32 each
//
// Every payload is copied as uint rather than converted through a floating-
// point register. This preserves all 32 bits, including signed zero and NaN
// payloads. Rust validates dtype, exact logical shapes, lengths, and aliasing
// before encoding; these guards remain as defense against excess grid threads.

struct QGateDeinterleaveParams {
    uint m;
    uint n_heads;
    uint head_dim;
};

kernel void q_gate_deinterleave_f32(
    device const uint *fused [[buffer(0)]],
    device uint *q [[buffer(1)]],
    device uint *gate [[buffer(2)]],
    constant QGateDeinterleaveParams &p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint column = gid.x;
    const uint head = gid.y;
    const uint row = gid.z;
    if (column >= p.head_dim || head >= p.n_heads || row >= p.m) {
        return;
    }

    const uint vector = row * p.n_heads + head;
    const uint src_base = vector * (2u * p.head_dim);
    const uint dst = vector * p.head_dim + column;
    q[dst] = fused[src_base + column];
    gate[dst] = fused[src_base + p.head_dim + column];
}
