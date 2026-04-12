#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// moe_expert_ffn — Single-expert FFN: gate_proj + up_proj -> GELU -> down_proj
//
// For a single expert, computes:
//   gate_out = gate_proj(x)        [input_dim -> intermediate_dim]
//   up_out   = up_proj(x)          [input_dim -> intermediate_dim]
//   hidden   = GELU(gate_out) * up_out
//   out      = down_proj(hidden)   [intermediate_dim -> input_dim]
//
// This shader works on float (f32) dequantized weights.  The Rust host
// is responsible for providing pre-dequantized weight slices for the
// selected expert, OR this shader is called after dequantization.
//
// For Stage 1, the Rust host loops over selected experts and dispatches
// the quantized_matmul kernel for each projection, then calls this
// elementwise kernel for the GELU + multiply fusion.
//
// This shader does the fused GELU-multiply: hidden = GELU(gate_out) * up_out
// --------------------------------------------------------------------------

// GELU approximation matching PyTorch/MLX: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Clamp threshold for tanh argument to prevent NaN from exp() overflow.
// tanh saturates at +/-1 well before |x| = 10, so clamping at 15 is safe.
inline float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float tanh_clamp = 15.0f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    inner = clamp(inner, -tanh_clamp, tanh_clamp);
    return 0.5f * x * (1.0f + tanh(inner));
}

struct FusedGeluMulParams {
    uint n_elements;
};

// Fused GELU(gate_out) * up_out
// Buffers:
//   0: gate_out  — float [n_elements]  (input, will be overwritten with result)
//   1: up_out    — float [n_elements]  (input)
//   2: output    — float [n_elements]  (output: GELU(gate_out) * up_out)
//   3: params    — { n_elements }
kernel void fused_gelu_mul(
    device const float* gate_out [[buffer(0)]],
    device const float* up_out   [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    constant FusedGeluMulParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    output[gid] = gelu_approx(gate_out[gid]) * up_out[gid];
}

// --------------------------------------------------------------------------
// moe_accumulate — Weighted accumulation: result += weight * expert_output
//
// Used by the Rust host to accumulate expert outputs with routing weights.
//
// Buffers:
//   0: accumulator   — float [n_elements]  (in/out)
//   1: expert_output — float [n_elements]  (input)
//   2: params        — { n_elements, routing_weight }
// --------------------------------------------------------------------------

struct MoeAccumParams {
    uint n_elements;
    float routing_weight;
};

kernel void moe_accumulate(
    device float*       accumulator   [[buffer(0)]],
    device const float* expert_output [[buffer(1)]],
    constant MoeAccumParams& params   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    accumulator[gid] += params.routing_weight * expert_output[gid];
}

// --------------------------------------------------------------------------
// zero_buffer — Zero-initialize a float buffer
//
// Buffers:
//   0: buffer      — float [n_elements]
//   1: params      — { n_elements }
// --------------------------------------------------------------------------

struct ZeroParams {
    uint n_elements;
};

kernel void zero_buffer(
    device float* buffer [[buffer(0)]],
    constant ZeroParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    buffer[gid] = 0.0f;
}

// --------------------------------------------------------------------------
// moe_swiglu_fused — SwiGLU on a fused [gate, up] buffer.
//
// Takes a buffer of 2*N elements where:
//   - First N elements are the gate projection output
//   - Last N elements are the up projection output
// Produces N elements: GELU(gate[i]) * up[i]
//
// This is the fused variant for models that use a single gate_up projection
// (e.g., Gemma 4 MoE experts with fused gate/up weights).
//
// Buffers:
//   0: gate_up — float [2 * N] (input: gate || up, concatenated)
//   1: output  — float [N]     (output: GELU(gate) * up)
//   2: params  — { n_elements = N }
// --------------------------------------------------------------------------

kernel void moe_swiglu_fused(
    device const float* gate_up [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant FusedGeluMulParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n_elements) return;
    float gate_val = gate_up[gid];
    float up_val   = gate_up[params.n_elements + gid];
    output[gid] = gelu_approx(gate_val) * up_val;
}

// --------------------------------------------------------------------------
// naive_matvec_f32 — Simple matrix-vector multiply for expert projections.
//
// Computes: output[row] = dot(weight[row, :], input[:])
//
// weight is [N, K] row-major, input is [K], output is [N].
// Each thread computes one output element.
//
// This is a naive implementation for Stage 1.  For large matrices,
// the quantized_matmul kernel from Story 1.2 would be used instead.
//
// Buffers:
//   0: weight — float [N, K] row-major
//   1: input  — float [K]
//   2: output — float [N]
//   3: params — { m (unused, always 1), k, n }
// --------------------------------------------------------------------------

struct MatvecParams {
    uint m;  // unused for matvec, included for struct compatibility
    uint k;  // inner dimension
    uint n;  // output dimension (number of rows in weight)
};

kernel void naive_matvec_f32(
    device const float* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant MatvecParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.n) return;

    float sum = 0.0f;
    device const float* row = weight + gid * params.k;
    for (uint i = 0; i < params.k; i++) {
        sum += row[i] * input[i];
    }
    output[gid] = sum;
}
