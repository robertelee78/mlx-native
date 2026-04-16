#include <metal_stdlib>
using namespace metal;

/// Fused RMS normalization + residual addition (float32).
///
/// Computes Gemma4's post-attention / post-FFN ordering:
///   normed[i] = rms_norm(input[i], weight[i], eps)
///   output[i] = residual[i] + normed[i]
///
/// Fuses two separate dispatches (rms_norm_f32 + elementwise_add_f32) into
/// one kernel launch per transformer sub-layer.
///
/// Buffer layout:
///   buffer(0): residual — float [rows * dim]   residual stream (unmodified)
///   buffer(1): input    — float [rows * dim]   sublayer output (to normalize)
///   buffer(2): weight   — float [dim]          RMS norm learned scale
///   buffer(3): output   — float [rows * dim]   residual + normed result
///   buffer(4): dim      — uint
///   buffer(5): rows     — uint
///   buffer(6): eps      — float
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1) — one threadgroup per row
/// Grid       : (rows, 1, 1)
/// Shared mem : tg_size * sizeof(float) for the sum-of-squares reduction

kernel void fused_norm_add_f32(
    device const float* residual [[buffer(0)]],
    device const float* input    [[buffer(1)]],
    device const float* weight   [[buffer(2)]],
    device float*       output   [[buffer(3)]],
    constant uint&      dim      [[buffer(4)]],
    constant uint&      rows     [[buffer(5)]],
    constant float&     eps      [[buffer(6)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    if (row_id >= rows) { return; }

    const uint base = row_id * dim;

    // Phase 1: accumulate partial sum-of-squares over input.
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = input[base + i];
        partial_sq += v * v;
    }

    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction to obtain total sum-of-squares.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // rms_inv = rsqrt(mean(input^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize input, apply weight, add residual, store output.
    for (uint i = tid; i < dim; i += tg_size) {
        const float normed = input[base + i] * rms_inv * weight[i];
        output[base + i] = residual[base + i] + normed;
    }
}

/// Fused residual addition + RMS normalization (float32).
///
/// Computes:
///   sum[i]    = residual[i] + input[i]
///   normed[i] = sum[i] * rsqrt(mean(sum^2) + eps) * weight[i]
///
/// Optionally writes `sum` to a separate output buffer for use as the next
/// layer's residual (avoids an extra elementwise kernel).
///
/// Buffer layout:
///   buffer(0): residual      — float [rows * dim]
///   buffer(1): input         — float [rows * dim]
///   buffer(2): weight        — float [dim]
///   buffer(3): normed_output — float [rows * dim]   normalized result
///   buffer(4): sum_output    — float [rows * dim]   residual+input (may be unused)
///   buffer(5): params        — { dim, rows, eps, write_sum }
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1)
/// Grid       : (rows, 1, 1)
///
/// Shared memory: tg_size * sizeof(float) at threadgroup(0) for the reduction.

struct FusedResidualNormF32Params {
    uint  dim;
    uint  rows;
    float eps;
    uint  write_sum;   // 0 = skip writing sum_output, nonzero = write it
};

kernel void fused_residual_norm_f32(
    device const float*               residual      [[buffer(0)]],
    device const float*               input         [[buffer(1)]],
    device const float*               weight        [[buffer(2)]],
    device float*                     normed_output [[buffer(3)]],
    device float*                     sum_output    [[buffer(4)]],
    constant FusedResidualNormF32Params& params     [[buffer(5)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared          [[threadgroup(0)]]
) {
    const uint  dim       = params.dim;
    const float eps       = params.eps;
    const bool  write_sum = (params.write_sum != 0u);

    const uint base = row_id * dim;

    // Phase 1: compute residual + input element-wise, accumulate sum-of-squares
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float s = residual[base + i] + input[base + i];
        shared[i] = s;
        partial_sq += s * s;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Optionally write the un-normed sum before the reduction overwrites shared.
    if (write_sum) {
        for (uint i = tid; i < dim; i += tg_size) {
            sum_output[base + i] = shared[i];
        }
    }

    // Reduction
    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: recompute element sums, normalize, write
    for (uint i = tid; i < dim; i += tg_size) {
        const float s = residual[base + i] + input[base + i];
        const float w = weight[i];
        normed_output[base + i] = s * rms_inv * w;
    }
}

/// Fused post-layer: residual add + RMS norm + scalar multiply (float32).
///
/// Computes the end-of-layer sequence in one pass:
///   sum[i]    = residual[i] + input[i]
///   normed[i] = rms_norm(sum, weight, eps)[i]
///   output[i] = normed[i] * scalar
///
/// When scalar_is_vector != 0, scalar is a per-element array of shape [dim].
/// Otherwise, scalar[0] is broadcast to all elements.
///
/// Buffer layout:
///   buffer(0): residual — float [rows * dim]
///   buffer(1): input    — float [rows * dim]
///   buffer(2): weight   — float [dim]
///   buffer(3): output   — float [rows * dim]
///   buffer(4): scalar   — float [1] or [dim]
///   buffer(5): params   — FusedResidualNormScalarF32Params
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1)
/// Grid       : (rows, 1, 1)

struct FusedResidualNormScalarF32Params {
    uint  dim;
    uint  rows;
    float eps;
    uint  scalar_is_vector; // 0 = broadcast scalar[0], nonzero = per-element
};

kernel void fused_residual_norm_scalar_f32(
    device const float*                       residual [[buffer(0)]],
    device const float*                       input    [[buffer(1)]],
    device const float*                       weight   [[buffer(2)]],
    device float*                             output   [[buffer(3)]],
    device const float*                       scalar   [[buffer(4)]],
    constant FusedResidualNormScalarF32Params& params  [[buffer(5)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared          [[threadgroup(0)]]
) {
    const uint  dim              = params.dim;
    const float eps              = params.eps;
    const bool  scalar_is_vector = (params.scalar_is_vector != 0u);

    const uint base = row_id * dim;

    // Phase 1: accumulate sum-of-squares of (residual + input)
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float s = residual[base + i] + input[base + i];
        shared[i] = s;
        partial_sq += s * s;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction
    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Load broadcast scalar if needed
    const float broadcast_scalar = scalar_is_vector ? 0.0f : scalar[0];

    // Phase 2: recompute sums, normalize, scale, write
    for (uint i = tid; i < dim; i += tg_size) {
        const float s = residual[base + i] + input[base + i];
        const float w = weight[i];
        const float sc = scalar_is_vector ? scalar[i] : broadcast_scalar;
        output[base + i] = s * rms_inv * w * sc;
    }
}

/// Fused MoE routing: softmax + argsort descending + gather top-K weights (float32).
///
/// Replaces 3 separate dispatches with one kernel. Operates on [1, num_experts]
/// logits (single token). Top-K is small (typically 2).
///
/// Buffer layout:
///   buffer(0): logits         — float [num_experts]  (input)
///   buffer(1): expert_ids     — uint  [top_k]        (output: sorted expert indices)
///   buffer(2): routing_weights— float [top_k]        (output: top-K softmax weights
///                                                    renormalized over the selected
///                                                    experts, then scaled by
///                                                    per_expert_scale)
///   buffer(3): per_expert_scale — float [num_experts] (input: per-expert scale factors)
///   buffer(4): params         — { num_experts, top_k }
///
/// Single threadgroup, tg_size threads.

struct FusedMoeRoutingParams {
    uint num_experts;
    uint top_k;
};

kernel void fused_moe_routing_f32(
    device const float*               logits          [[buffer(0)]],
    device uint*                      expert_ids      [[buffer(1)]],
    device float*                     routing_weights [[buffer(2)]],
    device const float*               per_expert_scale [[buffer(3)]],
    constant FusedMoeRoutingParams&   params          [[buffer(4)]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint num_experts = params.num_experts;
    const uint top_k       = params.top_k;

    // Step 1: find max for numerical stability (softmax)
    float local_max = -INFINITY;
    for (uint i = tid; i < num_experts; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_val = shared[0];

    // Step 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < num_experts; i += tg_size) {
        const float e = exp(logits[i] - max_val);
        shared[num_experts + i] = e;  // store exp values after first num_experts slots
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum_exp = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: compute softmax probabilities in shared[0..num_experts)
    for (uint i = tid; i < num_experts; i += tg_size) {
        shared[i] = shared[num_experts + i] / sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: find top-K (serial, only thread 0 — K is tiny, typically 2)
    if (tid == 0) {
        // Simple selection sort for K elements
        for (uint k = 0; k < top_k; k++) {
            float best_val = -1.0f;
            uint best_idx = 0;
            for (uint i = 0; i < num_experts; i++) {
                if (shared[i] > best_val) {
                    best_val = shared[i];
                    best_idx = i;
                }
            }
            expert_ids[k] = best_idx;
            // Match the old passing candle path:
            //   1. softmax over all experts
            //   2. renormalize the selected top-K slice
            //   3. apply per_expert_scale
            // Critically, do NOT renormalize after applying per_expert_scale.
            routing_weights[k] = best_val;
            shared[best_idx] = -1.0f;  // mark as used
        }

        // Renormalize the top-K weights before applying per_expert_scale.
        float topk_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            topk_sum += routing_weights[k];
        }
        if (topk_sum > 0.0f) {
            for (uint k = 0; k < top_k; k++) {
                const uint eid = expert_ids[k];
                routing_weights[k] = (routing_weights[k] / topk_sum) * per_expert_scale[eid];
            }
        } else {
            for (uint k = 0; k < top_k; k++) {
                routing_weights[k] = 0.0f;
            }
        }
    }
}

/// Batched fused MoE routing for prefill (float32).
///
/// Same semantics as fused_moe_routing_f32, but processes n_tokens at once.
/// Grid: (n_tokens, 1, 1). Each threadgroup handles one token's routing.
///
/// Buffer layout:
///   buffer(0): logits          — float [n_tokens, num_experts]
///   buffer(1): expert_ids      — uint  [n_tokens, top_k]
///   buffer(2): routing_weights — float [n_tokens, top_k]
///   buffer(3): per_expert_scale — float [num_experts]
///   buffer(4): params          — { num_experts, top_k }
///
/// Shared memory: (2 * num_experts + tg_size) floats.
kernel void fused_moe_routing_batch_f32(
    device const float*               logits_all      [[buffer(0)]],
    device uint*                      expert_ids_all  [[buffer(1)]],
    device float*                     routing_weights_all [[buffer(2)]],
    device const float*               per_expert_scale [[buffer(3)]],
    constant FusedMoeRoutingParams&   params          [[buffer(4)]],
    uint tok_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint num_experts = params.num_experts;
    const uint top_k       = params.top_k;

    device const float* logits          = logits_all       + tok_id * num_experts;
    device uint*        expert_ids      = expert_ids_all   + tok_id * top_k;
    device float*       routing_weights = routing_weights_all + tok_id * top_k;

    // Step 1: find max for numerical stability (softmax)
    float local_max = -INFINITY;
    for (uint i = tid; i < num_experts; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_val = shared[0];

    // Step 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < num_experts; i += tg_size) {
        const float e = exp(logits[i] - max_val);
        shared[num_experts + i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum_exp = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: compute softmax probabilities in shared[0..num_experts)
    for (uint i = tid; i < num_experts; i += tg_size) {
        shared[i] = shared[num_experts + i] / sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: find top-K (serial, only thread 0 — K is tiny, typically 2)
    if (tid == 0) {
        for (uint k = 0; k < top_k; k++) {
            float best_val = -1.0f;
            uint best_idx = 0;
            for (uint i = 0; i < num_experts; i++) {
                if (shared[i] > best_val) {
                    best_val = shared[i];
                    best_idx = i;
                }
            }
            expert_ids[k] = best_idx;
            routing_weights[k] = best_val;
            shared[best_idx] = -1.0f;
        }

        float topk_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            topk_sum += routing_weights[k];
        }
        if (topk_sum > 0.0f) {
            for (uint k = 0; k < top_k; k++) {
                const uint eid = expert_ids[k];
                routing_weights[k] = (routing_weights[k] / topk_sum) * per_expert_scale[eid];
            }
        } else {
            for (uint k = 0; k < top_k; k++) {
                routing_weights[k] = 0.0f;
            }
        }
    }
}

/// Fused RMS normalization + residual addition + scalar multiply (float32).
///
/// Computes:
///   normed[i] = rms_norm(input, weight, eps)[i]
///   output[i] = (residual[i] + normed[i]) * scalar[i or 0]
///
/// This is the correct end-of-layer operation for Gemma 4:
///   output = (residual + rms_norm(mlp_output)) * layer_scalar
///
/// Note: the norm is applied to `input` ALONE, not to the sum.
///
/// Buffer layout:
///   buffer(0): residual — float [rows * dim]
///   buffer(1): input    — float [rows * dim]   (sublayer output to normalize)
///   buffer(2): weight   — float [dim]          (RMS norm learned scale)
///   buffer(3): output   — float [rows * dim]
///   buffer(4): scalar   — float [1] or [dim]
///   buffer(5): params   — FusedNormAddScalarF32Params

struct FusedNormAddScalarF32Params {
    uint  dim;
    uint  rows;
    float eps;
    uint  scalar_is_vector; // 0 = broadcast scalar[0], nonzero = per-element
};

kernel void fused_norm_add_scalar_f32(
    device const float*                    residual [[buffer(0)]],
    device const float*                    input    [[buffer(1)]],
    device const float*                    weight   [[buffer(2)]],
    device float*                          output   [[buffer(3)]],
    device const float*                    scalar   [[buffer(4)]],
    constant FusedNormAddScalarF32Params&  params   [[buffer(5)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared          [[threadgroup(0)]]
) {
    const uint  dim              = params.dim;
    const float eps              = params.eps;
    const bool  scalar_is_vector = (params.scalar_is_vector != 0u);

    if (row_id >= params.rows) { return; }

    const uint base = row_id * dim;

    // Phase 1: accumulate sum-of-squares of input (NOT the sum with residual)
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = input[base + i];
        partial_sq += v * v;
    }

    shared[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    const float broadcast_scalar = scalar_is_vector ? 0.0f : scalar[0];

    // Phase 2: normalize input, add residual, scale, write
    for (uint i = tid; i < dim; i += tg_size) {
        const float normed = input[base + i] * rms_inv * weight[i];
        const float sum = residual[base + i] + normed;
        const float sc = scalar_is_vector ? scalar[i] : broadcast_scalar;
        output[base + i] = sum * sc;
    }
}
