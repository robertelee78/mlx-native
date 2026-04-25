// moe_weighted_reduce.metal — GPU weighted accumulate for MoE output.
//
// Replaces the CPU loop in build_moe_ffn_layer_gpu_q (Step 3e):
//   out[t, h] = sum_{k=0}^{top_k-1} weights[t*top_k + k] * y_all[(t*top_k + k), h]
//
// For decode (n_tokens=1, top_k=8, h=2048), this is 8×2048 multiply-accumulate
// operations — trivial GPU work.
//
// Also handles the shared expert weighted accumulate:
//   out[t, :] += sh_gate_val[t] * y_s[t, :]
//
// We use a separate fused kernel that adds both contributions:
//   out[t, h] = sum_{k=0}^{top_k-1} expert_w[t*top_k + k] * y_expert[(t*top_k + k), h]
//               + sh_gate[t] * y_shared[t, h]
//               + (optional) residual[t, h]    -- folded in when add_residual=1
//
// Grid: (ceil(h/TG_SIZE), n_tokens, 1)  where TG_SIZE = 256.
// Each thread computes one output element.

#include <metal_stdlib>
using namespace metal;

struct MoeWeightedReduceParams {
    uint n_tokens;  // number of tokens
    uint top_k;     // experts per token
    uint h;         // hidden size
    uint add_residual; // 1 = add residual, 0 = don't
};

// Fused MoE weighted accumulate + shared expert add + optional residual.
//
// sh_logit is the raw pre-sigmoid scalar; sigmoid is applied inline.
kernel void moe_weighted_reduce_f32(
        constant MoeWeightedReduceParams & p   [[buffer(0)]],
        device const float * expert_w          [[buffer(1)]],  // [n_tokens * top_k] routing weights
        device const float * y_expert          [[buffer(2)]],  // [n_tokens * top_k, h] expert outputs
        device const float * sh_logit          [[buffer(3)]],  // [n_tokens] raw shared gate logit (pre-sigmoid)
        device const float * y_shared          [[buffer(4)]],  // [n_tokens, h] shared expert output
        device const float * residual          [[buffer(5)]],  // [n_tokens, h] residual (if add_residual=1)
        device       float * output            [[buffer(6)]],  // [n_tokens, h] output
        uint2  tgpig [[threadgroup_position_in_grid]],
        uint   tiitg [[thread_index_in_threadgroup]]) {

    const uint h_idx = tgpig.x * 256 + tiitg;
    const uint t     = tgpig.y;

    if (h_idx >= p.h || t >= p.n_tokens) return;

    // Accumulate expert contributions.
    float acc = 0.f;
    for (uint k = 0; k < p.top_k; k++) {
        const uint slot = t * p.top_k + k;
        acc += expert_w[slot] * y_expert[slot * p.h + h_idx];
    }

    // Compute sigmoid of sh_logit and apply to shared expert output.
    const float sh_gate = 1.f / (1.f + exp(-sh_logit[t]));
    acc += sh_gate * y_shared[t * p.h + h_idx];

    // Optionally add residual.
    if (p.add_residual != 0) {
        acc += residual[t * p.h + h_idx];
    }

    output[t * p.h + h_idx] = acc;
}
