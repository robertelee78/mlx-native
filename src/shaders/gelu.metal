#include <metal_stdlib>
using namespace metal;

/// GELU activation (pytorch_tanh variant).
///
/// Computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Buffer layout:
///   buffer(0): input  — float array
///   buffer(1): output — float array
///
/// Grid: (element_count, 1, 1)

constant float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;   // sqrt(2/pi)
constant float GELU_COEFF          = 0.044715f;
// Clamp threshold for tanh argument to prevent NaN from exp() overflow.
// tanh saturates at +/-1 well before |x| = 10, so clamping at 15 is safe.
constant float TANH_CLAMP          = 15.0f;

kernel void gelu_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    const float x = input[id];
    const float x_cubed = x * x * x;
    const float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    // Clamp to avoid NaN from tanh overflow on large |inner| values.
    const float clamped = clamp(inner, -TANH_CLAMP, TANH_CLAMP);
    output[id] = 0.5f * x * (1.0f + tanh(clamped));
}

kernel void gelu_f16(
    device const half  *input  [[buffer(0)]],
    device half        *output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // Promote to f32 for accurate computation
    const float x = float(input[id]);
    const float x_cubed = x * x * x;
    const float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    const float clamped = clamp(inner, -TANH_CLAMP, TANH_CLAMP);
    output[id] = half(0.5f * x * (1.0f + tanh(clamped)));
}

kernel void gelu_bf16(
    device const bfloat *input  [[buffer(0)]],
    device bfloat       *output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // Promote to f32 for accurate computation; accumulate and compute in f32
    const float x = static_cast<float>(input[id]);
    const float x_cubed = x * x * x;
    const float inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    const float clamped = clamp(inner, -TANH_CLAMP, TANH_CLAMP);
    output[id] = bfloat(0.5f * x * (1.0f + tanh(clamped)));
}
