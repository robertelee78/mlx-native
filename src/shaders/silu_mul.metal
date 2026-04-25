#include <metal_stdlib>
using namespace metal;

// Fused SiLU-gated multiply: out[i] = gate[i] * sigmoid(gate[i]) * up[i].
//
// SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate)).
// Used by Qwen3.5 MoE FFN and shared expert SwiGLU activations.
//
// Buffer layout:
//   buffer(0): gate   - f32 array [n]
//   buffer(1): up     - f32 array [n]
//   buffer(2): output - f32 array [n]
//   buffer(3): n      - u32 (element count)
//
// Grid: 1D threads across n.

kernel void silu_mul_f32(
    device const float *gate   [[buffer(0)]],
    device const float *up     [[buffer(1)]],
    device float       *output [[buffer(2)]],
    device const uint  *params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    const float g = gate[tid];
    // SiLU(g) = g * sigmoid(g)
    const float silu_g = g / (1.0f + metal::exp(-g));
    output[tid] = silu_g * up[tid];
}
