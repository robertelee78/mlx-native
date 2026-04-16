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
// moe_swiglu_batch — Batched SwiGLU across all top_k expert slots.
//
// Takes a [top_k, 2*intermediate] buffer where for each slot k:
//   - gate values are at [k, 0..intermediate)
//   - up values are at [k, intermediate..2*intermediate)
// Produces [top_k, intermediate] output: GELU(gate[i]) * up[i] per slot.
//
// Grid: 2D — x=element within intermediate, y=expert slot.
// Replaces top_k separate moe_swiglu_fused dispatches with 1.
// --------------------------------------------------------------------------

kernel void moe_swiglu_batch(
    device const float* gate_up_buf  [[buffer(0)]],  // [top_k, 2*intermediate]
    device float*       output_buf   [[buffer(1)]],  // [top_k, intermediate]
    constant uint&      intermediate [[buffer(2)]],
    constant uint&      top_k        [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]             // x=element, y=slot
) {
    uint i = tid.x;
    uint slot = tid.y;
    if (slot >= top_k || i >= intermediate) return;

    uint base = slot * 2 * intermediate;
    float gate = gate_up_buf[base + i];
    float up = gate_up_buf[base + intermediate + i];
    // SwiGLU = GELU(gate) * up
    float gelu = gate * 0.5f * (1.0f + precise::tanh(
        0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
    output_buf[slot * intermediate + i] = gelu * up;
}

// --------------------------------------------------------------------------
// moe_weighted_sum — Weighted sum of all top_k expert outputs in one dispatch.
//
// Replaces the zero_buffer + top_k * moe_accumulate pattern (9 dispatches)
// with a single dispatch that reads all expert outputs and routing weights.
//
// Grid: 1D — each thread computes one element of the output.
// --------------------------------------------------------------------------

struct MoeWeightedSumParams {
    uint hidden_size;
    uint top_k;
};

kernel void moe_weighted_sum(
    device const float*  expert_outputs [[buffer(0)]],  // [top_k, hidden_size]
    device const float*  weights        [[buffer(1)]],  // [top_k]
    device float*        output         [[buffer(2)]],  // [hidden_size]
    constant MoeWeightedSumParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.hidden_size) return;
    float sum = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        sum += expert_outputs[k * params.hidden_size + tid] * weights[k];
    }
    output[tid] = sum;
}

/// Multi-token SwiGLU for batched prefill.
/// Input:  [n_tokens, top_k, 2*intermediate]
/// Output: [n_tokens, top_k, intermediate]
/// Grid:   3D (intermediate, top_k, n_tokens)
struct MoeSwigluSeqParams {
    uint intermediate;
    uint top_k;
    uint n_tokens;
};
kernel void moe_swiglu_seq(
    device const float* gate_up_buf  [[buffer(0)]],
    device float*       output_buf   [[buffer(1)]],
    constant MoeSwigluSeqParams& params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint slot = tid.y;
    uint tok = tid.z;
    if (tok >= params.n_tokens || slot >= params.top_k || i >= params.intermediate) return;

    uint slot_base = (tok * params.top_k + slot) * 2 * params.intermediate;
    float gate = gate_up_buf[slot_base + i];
    float up   = gate_up_buf[slot_base + params.intermediate + i];
    float gelu = gate * 0.5f * (1.0f + precise::tanh(
        0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
    output_buf[(tok * params.top_k + slot) * params.intermediate + i] = gelu * up;
}

/// Multi-token weighted sum of expert outputs.
/// Input:  expert_outputs [n_tokens, top_k, hidden_size]
///         weights        [n_tokens, top_k]
/// Output: [n_tokens, hidden_size]
struct MoeWeightedSumSeqParams {
    uint hidden_size;
    uint top_k;
    uint n_tokens;
};
kernel void moe_weighted_sum_seq(
    device const float*  expert_outputs [[buffer(0)]],
    device const float*  weights        [[buffer(1)]],
    device float*        output         [[buffer(2)]],
    constant MoeWeightedSumSeqParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint tok = tid.y;
    if (tok >= params.n_tokens || d >= params.hidden_size) return;

    float sum = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        const uint in_idx = (tok * params.top_k + k) * params.hidden_size + d;
        const uint w_idx = tok * params.top_k + k;
        sum += expert_outputs[in_idx] * weights[w_idx];
    }
    output[tok * params.hidden_size + d] = sum;
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

// --------------------------------------------------------------------------
// moe_gather_topk_weights — Gather softmax probs at top-K sorted indices,
// multiply by per_expert_scale, and renormalize.
//
// Replaces the CPU softmax+argsort+gather+scale+renorm sequence for MoE
// routing.  Runs on GPU so the session never needs to break.
//
// Inputs:
//   softmax_probs   — f32 [n_experts]  (softmax output, from dispatch_softmax)
//   sorted_indices  — u32 [n_experts]  (descending sort, from dispatch_argsort)
//   per_expert_scale— f32 [n_experts]  (per-expert learned scale)
//
// Outputs:
//   out_expert_ids  — u32 [top_k]  (selected expert indices)
//   out_weights     — f32 [top_k]  (pre-scaled, renormalized routing weights)
//
// Grid: single thread (top_k <= 8, trivial work).
// --------------------------------------------------------------------------

struct MoeGatherTopkParams {
    uint n_experts;
    uint top_k;
};

kernel void moe_gather_topk_weights(
    device const float*  softmax_probs    [[buffer(0)]],
    device const uint*   sorted_indices   [[buffer(1)]],
    device const float*  per_expert_scale [[buffer(2)]],
    device uint*         out_expert_ids   [[buffer(3)]],
    device float*        out_weights      [[buffer(4)]],
    constant MoeGatherTopkParams& params  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // 1. Gather top-K expert ids and their softmax probabilities
    float top_probs[8];   // max top_k = 8
    float prob_sum = 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        uint eid = sorted_indices[k];
        out_expert_ids[k] = eid;
        top_probs[k] = softmax_probs[eid];
        prob_sum += top_probs[k];
    }

    // 2. Renormalize and apply per_expert_scale
    float inv_sum = (prob_sum > 0.0f) ? (1.0f / prob_sum) : 0.0f;
    for (uint k = 0; k < params.top_k; k++) {
        uint eid = out_expert_ids[k];
        out_weights[k] = (top_probs[k] * inv_sum) * per_expert_scale[eid];
    }
}
