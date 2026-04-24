#include <metal_stdlib>
using namespace metal;

// Elementwise sigmoid-gated multiply.
//
// Computes: out[i] = x[i] * sigmoid(gate[i])    (Qwen3.5 full-attention
//                                                 output gate, ADR-013 Decision 9).
//
// sigmoid is the authoritative activation per ADR-013 (citing HF
// transformers modeling_qwen3_5.py:689 + vLLM qwen3_next.py:312-314).
// Swish would be a silent-corruption bug.
//
// Buffer layout:
//   buffer(0): x       - f32 array [n]
//   buffer(1): gate    - f32 array [n]
//   buffer(2): output  - f32 array [n]
//   buffer(3): n       - u32 (element count)
//
// Grid: 1D threads across n.

kernel void sigmoid_mul_f32(
    device const float *x      [[buffer(0)]],
    device const float *gate   [[buffer(1)]],
    device float       *output [[buffer(2)]],
    device const uint  *params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    const float g = gate[tid];
    // sigmoid(g) = 1 / (1 + exp(-g))
    const float s = 1.0f / (1.0f + metal::exp(-g));
    output[tid] = x[tid] * s;
}

kernel void sigmoid_mul_bf16(
    device const bfloat *x      [[buffer(0)]],
    device const bfloat *gate   [[buffer(1)]],
    device bfloat       *output [[buffer(2)]],
    device const uint   *params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = params[0];
    if (tid >= n) return;
    const float g = float(gate[tid]);
    const float s = 1.0f / (1.0f + metal::exp(-g));
    output[tid] = bfloat(float(x[tid]) * s);
}
