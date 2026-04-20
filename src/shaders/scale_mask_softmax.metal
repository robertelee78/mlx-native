// scale_mask_softmax.metal — fused scale-then-mask-then-softmax over
// attention scores for the non-flash-attention prefill path.
//
// Replaces three sequential dispatches (scale, mask-add, softmax) with
// one row-local pass.  Each threadgroup processes one (head, query-
// position) row of the scores tensor and reduces over the key axis.
//
// Contract:
//   input  — f32 [nh, seq_q, seq_k] = [rows = nh*seq_q, cols = seq_k]
//   output — f32 same shape (row-normalized softmax probs; may alias
//            input for in-place operation)
//   mask   — bf16 [seq_q, seq_k] row-major.  Shared across heads: for
//            row_idx = h*seq_q + q, mask[q, :] is applied.  Masked
//            positions hold -INF (matches flash_attn_prefill_mask.metal's
//            sentinel: attended = 0.0f, masked = -INF).
//
// Math (per row):
//   tmp[k] = input[row, k] * scale + float(mask[q, k])
//   row_max = max_k tmp[k]
//   exp_k   = exp(tmp[k] - row_max)
//   row_sum = sum_k exp_k
//   output[row, k] = exp_k / row_sum
//
// Layout assumptions:
//   * Dispatcher sends threadgroups=(rows, 1, 1), tgsize=(N, 1, 1) with
//     N >= 32.  Each threadgroup gets `shared[N]` floats of scratch.
//   * rows is nh*seq_q; cols = seq_k.  The caller passes seq_q in
//     params so we can derive q = row_idx % seq_q for the mask index.
//
// Used for the non-FA prefill attention path (HF2Q_NO_FA=1).  Modelled
// on llama.cpp's kernel_soft_max_f32 (ggml-metal.metal:1855-1960),
// simplified for our specific case (no ALiBi, bf16 mask, fixed scale).

#include <metal_stdlib>
using namespace metal;

struct ScaleMaskSoftmaxParams {
    uint  cols;        // seq_k (size of reduction axis)
    uint  seq_q;       // number of rows per head (to compute q = row % seq_q)
    float scale;       // multiplicative scale applied to input pre-mask (= 1/sqrt(hd))
    uint  _pad;
};

kernel void scale_mask_softmax_f32(
    device const float  *input   [[buffer(0)]],
    device       float  *output  [[buffer(1)]],
    device const bfloat *mask    [[buffer(2)]],
    constant ScaleMaskSoftmaxParams & params [[buffer(3)]],
    uint  row_idx [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    threadgroup float *shared [[threadgroup(0)]]
) {
    const uint  cols       = params.cols;
    const uint  seq_q      = params.seq_q;
    const float scale      = params.scale;
    const uint  q          = row_idx % seq_q;
    const uint  scores_base = row_idx * cols;
    const uint  mask_base   = q * cols;

    // --- Phase 1: row-max over (input * scale + mask).  Accumulate in f32
    //     even though the mask is bf16 (-INF must not underflow when added).
    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tg_size) {
        float v = input[scores_base + i] * scale + float(mask[mask_base + i]);
        local_max = max(local_max, v);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 2: exp(scaled-masked - max); write intermediate to output;
    //     accumulate f32 sum for normalisation.
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        float v = input[scores_base + i] * scale + float(mask[mask_base + i]);
        float e = exp(v - row_max);
        output[scores_base + i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 3: normalise.  Guard against row_sum = 0 (fully-masked
    //     row: all -INF masked → row_max = -INF → exp(...) = 0 → sum = 0).
    //     A fully-masked row is impossible under causal+sliding-window
    //     attention on our shapes, but the guard keeps the output
    //     defined rather than NaN if the invariant ever breaks.
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        output[scores_base + i] *= inv_sum;
    }
}
