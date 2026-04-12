#include <metal_stdlib>
using namespace metal;

/// Softcap kernel: tanh-based logit capping.
///
/// Computes: output = tanh(input / cap) * cap
///
/// This bounds output values to the range (-cap, +cap).
///
/// Buffer layout:
///   buffer(0): input  — float array
///   buffer(1): output — float array
///   buffer(2): params — float2: (cap, n_elements_as_float)
///
/// Grid: dispatched with enough threads to cover n_elements.
/// Threads beyond n_elements are no-ops (bounds check).

kernel void softcap_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    device const float *params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    const uint n_elements = as_type<uint>(params[1]);
    if (id >= n_elements) return;
    const float cap = params[0];
    const float x = input[id];
    output[id] = tanh(x / cap) * cap;
}

kernel void softcap_f16(
    device const half  *input  [[buffer(0)]],
    device half        *output [[buffer(1)]],
    device const float *params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    const uint n_elements = as_type<uint>(params[1]);
    if (id >= n_elements) return;
    const float cap = params[0];
    // Promote to f32 for accurate tanh computation
    const float x = float(input[id]);
    output[id] = half(tanh(x / cap) * cap);
}

kernel void softcap_bf16(
    device const bfloat *input  [[buffer(0)]],
    device bfloat       *output [[buffer(1)]],
    device const float  *params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    const uint n_elements = as_type<uint>(params[1]);
    if (id >= n_elements) return;
    const float cap = params[0];
    // Promote to f32 for accurate tanh computation; accumulate in f32
    const float x = static_cast<float>(input[id]);
    output[id] = bfloat(tanh(x / cap) * cap);
}
