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

/// Fused MoE-weighted-sum + RMS norm + residual add (float32).
///
/// Replaces the three-dispatch sequence in batched prefill:
///   1. moe_weighted_sum_seq:    sum[tok,d] = sum_k expert_outputs[tok,k,d] * weights[tok,k]
///   2. fused_norm_add_f32:      output[tok,d] = residual[tok,d] + rms_norm(sum, weight, eps)
///
/// (Steps 1+2 combined; the kernel reads expert_outputs+weights, accumulates
/// the weighted sum into a threadgroup-local buffer in phase 1, computes
/// RMS over that buffer, and writes residual + norm(sum)*weight in phase 2.)
///
/// Wave P4.13 — saves 1 dispatch per layer (30/prefill on Gemma 4) and one
/// global write+read of the [rows * dim] intermediate sum buffer (~5 MB at
/// pp2455 × 30 = 150 MB of read+write traffic eliminated).
///
/// Buffer layout:
///   buffer(0): expert_outputs — float [rows * top_k * dim]   MoE down outputs
///   buffer(1): weights        — float [rows * top_k]         MoE routing weights
///   buffer(2): residual       — float [rows * dim]           residual stream
///   buffer(3): norm_weight    — float [dim]                  RMS norm scale
///   buffer(4): output         — float [rows * dim]           residual + norm(sum)
///   buffer(5): dim            — uint
///   buffer(6): top_k          — uint
///   buffer(7): rows           — uint
///   buffer(8): eps            — float
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1) — one threadgroup per row
/// Grid       : (rows, 1, 1)
/// Shared mem : tg_size + dim floats (reduction scratch + weighted_sum buffer)
kernel void fused_moe_wsum_norm_add_f32(
    device const float* expert_outputs [[buffer(0)]],
    device const float* weights        [[buffer(1)]],
    device const float* residual       [[buffer(2)]],
    device const float* norm_weight    [[buffer(3)]],
    device float*       output         [[buffer(4)]],
    constant uint&      dim            [[buffer(5)]],
    constant uint&      top_k          [[buffer(6)]],
    constant uint&      rows           [[buffer(7)]],
    constant float&     eps            [[buffer(8)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared          [[threadgroup(0)]]
) {
    if (row_id >= rows) { return; }

    // Threadgroup shmem layout: first `tg_size` floats are the
    // sum-of-squares reduction scratch (reused for tree-reduce), then
    // `dim` floats for the weighted-sum result buffer.  Both regions are
    // only valid within Phase 1/2 of THIS row's computation; nothing
    // crosses threadgroup boundaries.
    threadgroup float* sum_scratch = shared;
    threadgroup float* sum_buf     = shared + tg_size;

    const uint base_w  = row_id * top_k;
    const uint base_eo = row_id * top_k * dim;
    const uint base_d  = row_id * dim;

    // Phase 1: each thread computes sum[i] = sum_k expert_outputs[i,k] *
    // weights[k] for its strided i, accumulates v*v for RMS, and stashes
    // v in sum_buf for phase 2 reuse.
    float partial_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = 0.0f;
        for (uint k = 0; k < top_k; ++k) {
            v += expert_outputs[base_eo + k * dim + i] * weights[base_w + k];
        }
        sum_buf[i] = v;
        partial_sq += v * v;
    }

    sum_scratch[tid] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(sum_scratch[0] / float(dim) + eps);

    // Phase 2: residual + rms_norm(sum) * weight, no re-compute of sum.
    for (uint i = tid; i < dim; i += tg_size) {
        const float normed = sum_buf[i] * rms_inv * norm_weight[i];
        output[base_d + i] = residual[base_d + i] + normed;
    }
}

/// Fused MoE-weighted-sum + double-RMS-norm + add (float32).  Wave P4.14.
///
/// Replaces the three-dispatch post-MoE-down sequence in batched prefill:
///   1. rms_norm_f32:                  pf_mlp_down → pf_mlp_down_out
///                                     (norm with post_feedforward_layernorm_1)
///   2. moe_weighted_sum_seq:          weighted = Σ_k pf_moe_down[k] * weights[k]
///   3. fused_norm_add_f32:            pf_mlp_down = pf_mlp_down_out + norm(weighted,
///                                                   post_feedforward_layernorm_2)
///
/// One kernel doing it all:
///   * Phase 1: per-thread accumulate (a) residual_sq for RMS(residual) and
///              (b) weighted_sum across top_k stored in sum_buf, plus
///              weighted_sq for RMS(weighted).
///   * Phase 2: two parallel tree reductions (residual + weighted) yield
///              both rms_inv values.
///   * Phase 3: output[d] = residual[d] * rms_inv_r * resnorm_weight[d] +
///                          sum_buf[d] * rms_inv_w * moenorm_weight[d]
///
/// Saves 2 dispatches per layer (60/prefill on Gemma 4) and eliminates
/// TWO [rows * dim] intermediate buffers (pf_mlp_down_out and
/// pf_moe_accum, ~10 MB at pp2455 × 30 = 300 MB of memory traffic).
///
/// Buffer layout:
///   buffer(0): expert_outputs    — float [rows * top_k * dim]
///   buffer(1): weights           — float [rows * top_k]
///   buffer(2): residual          — float [rows * dim]   (pre-norm, gets normed)
///   buffer(3): res_norm_weight   — float [dim]          (RMS norm scale for residual)
///   buffer(4): moe_norm_weight   — float [dim]          (RMS norm scale for weighted)
///   buffer(5): output            — float [rows * dim]
///   buffer(6): dim               — uint
///   buffer(7): top_k             — uint
///   buffer(8): rows              — uint
///   buffer(9): eps               — float
///
/// Threadgroup: (min(256, next_pow2(dim)), 1, 1) — one threadgroup per row.
/// Shared mem : 2*tg_size + dim floats (~10 KB at dim=2048, tg_size=256;
///              well under 32 KB budget).
kernel void fused_moe_wsum_dnorm_add_f32(
    device const float* expert_outputs   [[buffer(0)]],
    device const float* weights          [[buffer(1)]],
    device const float* residual         [[buffer(2)]],
    device const float* res_norm_weight  [[buffer(3)]],
    device const float* moe_norm_weight  [[buffer(4)]],
    device float*       output           [[buffer(5)]],
    constant uint&      dim              [[buffer(6)]],
    constant uint&      top_k            [[buffer(7)]],
    constant uint&      rows             [[buffer(8)]],
    constant float&     eps              [[buffer(9)]],
    uint row_id  [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared            [[threadgroup(0)]]
) {
    if (row_id >= rows) { return; }

    // Two parallel reduction scratch arrays + the per-row sum_buf.
    threadgroup float* sum_scratch_r = shared;
    threadgroup float* sum_scratch_w = shared + tg_size;
    threadgroup float* sum_buf       = shared + 2u * tg_size;

    const uint base_w  = row_id * top_k;
    const uint base_eo = row_id * top_k * dim;
    const uint base_d  = row_id * dim;

    // Phase 1: per-thread loads. Each thread handles strided d's.
    //   v_r = residual[d]                    -> partial_sq_r += v_r * v_r
    //   v_w = Σ_k expert_outputs[k,d] * weights[k]
    //                                        -> partial_sq_w += v_w * v_w
    //   sum_buf[d] = v_w  (stash for phase 3)
    //
    // Note: residual is loaded once and used both for RMS sum-sq AND
    // the phase-3 reweighting — but we can't keep it in registers across
    // the threadgroup_barrier, so phase 3 will re-read residual[base_d + i]
    // from global memory.  That's OK because the read is cached and we
    // saved 2 separate dispatches' worth of barrier+launch overhead.
    float partial_sq_r = 0.0f;
    float partial_sq_w = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v_r = residual[base_d + i];
        partial_sq_r += v_r * v_r;

        float v_w = 0.0f;
        for (uint k = 0; k < top_k; ++k) {
            v_w += expert_outputs[base_eo + k * dim + i] * weights[base_w + k];
        }
        sum_buf[i] = v_w;
        partial_sq_w += v_w * v_w;
    }

    sum_scratch_r[tid] = partial_sq_r;
    sum_scratch_w[tid] = partial_sq_w;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: two parallel tree reductions.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch_r[tid] += sum_scratch_r[tid + stride];
            sum_scratch_w[tid] += sum_scratch_w[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv_r = rsqrt(sum_scratch_r[0] / float(dim) + eps);
    const float rms_inv_w = rsqrt(sum_scratch_w[0] / float(dim) + eps);

    // Phase 3: combine per-element. residual is re-read from device
    // memory (cached); sum_buf is read from threadgroup memory.
    for (uint i = tid; i < dim; i += tg_size) {
        const float v_r = residual[base_d + i];
        const float v_w = sum_buf[i];
        const float normed_r = v_r * rms_inv_r * res_norm_weight[i];
        const float normed_w = v_w * rms_inv_w * moe_norm_weight[i];
        output[base_d + i] = normed_r + normed_w;
    }
}

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

    // All threads must complete the broadcast-read of shared[0] for max_val
    // BEFORE any thread (notably tid==0) overwrites shared[0] with its
    // local_sum. Same race class as the fused_head_norm_rope Phase-1→Phase-2
    // boundary (see b31505d / hf2q docs/spike-batched-prefill-race-rootcause.md).
    // Without this barrier, simdgroups that race ahead of simdgroup 0 read a
    // clobbered shared[0] and compute a corrupt max_val — produces
    // nondeterministic routing decisions at scale.
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
