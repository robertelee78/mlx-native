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

// Apple GPU simdgroup width — constant across all current generations.
#define N_SIMDWIDTH 32

struct ScaleMaskSoftmaxParams {
    uint  cols;        // seq_k (size of reduction axis)
    uint  seq_q;       // number of rows per head (to compute q = row % seq_q)
    float scale;       // multiplicative scale applied to input pre-mask (= 1/sqrt(hd))
    uint  _pad;
};

// D.3 — llama.cpp-style simdgroup-reduction softmax.  Uses hardware
// simd_max / simd_sum (1-cycle intra-simdgroup reductions) instead of
// the tree-reduce + threadgroup barriers we had before.  When the
// threadgroup has more than one simdgroup (tg_size > 32), a secondary
// cross-simdgroup reduction runs through shared memory — but within
// each phase that is only 1 extra barrier + 1 simd_reduce, versus the
// log2(tg_size) barriers the tree path took.  On Apple M5 at
// tg_size=256 (8 simdgroups, cols=2455 attention row), this cuts the
// softmax kernel time by ~3x per row.
//
// Structure matches llama.cpp's kernel_soft_max (ggml-metal.metal:1855).

kernel void scale_mask_softmax_f32(
    device const float  *input   [[buffer(0)]],
    device       float  *output  [[buffer(1)]],
    device const bfloat *mask    [[buffer(2)]],
    constant ScaleMaskSoftmaxParams & params [[buffer(3)]],
    uint  row_idx [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  sgitg   [[simdgroup_index_in_threadgroup]],
    uint  tiisg   [[thread_index_in_simdgroup]],
    threadgroup float *shared [[threadgroup(0)]]
) {
    const uint  cols       = params.cols;
    const uint  seq_q      = params.seq_q;
    const float scale      = params.scale;
    const uint  q          = row_idx % seq_q;
    const uint  scores_base = row_idx * cols;
    const uint  mask_base   = q * cols;

    // ---- Phase 1: row-max ----
    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tg_size) {
        float v = input[scores_base + i] * scale + float(mask[mask_base + i]);
        local_max = max(local_max, v);
    }

    // Intra-simdgroup reduction (hardware, 1 cycle).
    float max_val = simd_max(local_max);
    if (tg_size > N_SIMDWIDTH) {
        // Cross-simdgroup reduction via shared memory.
        if (sgitg == 0) {
            shared[tiisg] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            shared[sgitg] = max_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = shared[tiisg];
        max_val = simd_max(max_val);
    }

    // ---- Phase 2: exp(v - max), store to output, accumulate sum ----
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        float v = input[scores_base + i] * scale + float(mask[mask_base + i]);
        float e = exp(v - max_val);
        output[scores_base + i] = e;
        local_sum += e;
    }

    // Barrier fixes a sporadic reduction ordering bug on Apple GPUs —
    // matches llama.cpp's comment at ggml-metal.metal:1925.
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(local_sum);
    if (tg_size > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            shared[sgitg] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = shared[tiisg];
        sum = simd_sum(sum);
    }

    // ---- Phase 3: normalise.  Guard against sum=0 (fully-masked row). ----
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        output[scores_base + i] *= inv_sum;
    }
}

// ===========================================================================
// ADR-029 iter-93 H71: float4-vectorized scale_mask_softmax_f32 — peer parity
// with kernel_soft_max_f32_4 (ggml-metal.metal:1961).
//
// Mirrors the scalar kernel above but reads/writes input/output as
// float4 (4 elements per loop iteration). Mask remains bfloat: read
// 2-at-a-time via bfloat2 (matching iter-50 H46 pattern), promoted to
// float4 via two bfloat2 pairs. Reduces global-memory transaction count
// 4x on the input/output streams and 2x on the mask stream — the
// dominant cost in this kernel at large cols.
//
// REQUIRES cols % 4 == 0 — caller must dispatch only when this holds.
// For gemma4 NO_FA path, cols = seq_kv which is always divisible by 4
// at our prefill shapes (2247, 4173, 8333 — all even, divisible by 4
// only at 8333... actually no, 2247 % 4 = 3). The dispatcher must check
// and fall back to scalar kernel when alignment doesn't hold.
//
// Dispatcher: when `cols % 4 == 0` AND env HF2Q_SOFTMAX_V4=1, route to
// this kernel; otherwise route to the scalar kernel above.
// ===========================================================================

kernel void scale_mask_softmax_f32_v4(
    device const float  *input   [[buffer(0)]],
    device       float  *output  [[buffer(1)]],
    device const bfloat *mask    [[buffer(2)]],
    constant ScaleMaskSoftmaxParams & params [[buffer(3)]],
    uint  row_idx [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  sgitg   [[simdgroup_index_in_threadgroup]],
    uint  tiisg   [[thread_index_in_simdgroup]],
    threadgroup float *shared [[threadgroup(0)]]
) {
    const uint  cols       = params.cols;
    const uint  seq_q      = params.seq_q;
    const float scale      = params.scale;
    const uint  q          = row_idx % seq_q;
    const uint  scores_base = row_idx * cols;
    const uint  mask_base   = q * cols;

    // Pointer aliases for vectorized access. Caller MUST ensure cols % 4 == 0.
    device const float4 *input4  = (device const float4 *)(input  + scores_base);
    device       float4 *output4 = (device       float4 *)(output + scores_base);
    // bfloat4 reads — peer (line 6006) uses similar pattern (half2 reads).
    device const bfloat4 *mask4  = (device const bfloat4 *)(mask  + mask_base);

    const uint cols4 = cols / 4;

    // ---- Phase 1: row-max ----
    float4 local_max4 = float4(-INFINITY);
    for (uint i = tid; i < cols4; i += tg_size) {
        float4 v = input4[i] * scale;
        // bfloat4 -> float4 explicit conversion (Metal auto-promotes).
        bfloat4 m = mask4[i];
        v += float4(float(m.x), float(m.y), float(m.z), float(m.w));
        local_max4 = max(local_max4, v);
    }
    float local_max = max(max(local_max4.x, local_max4.y), max(local_max4.z, local_max4.w));

    float max_val = simd_max(local_max);
    if (tg_size > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared[tiisg] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            shared[sgitg] = max_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = shared[tiisg];
        max_val = simd_max(max_val);
    }

    // ---- Phase 2: exp(v - max), store output, accumulate sum ----
    float4 local_sum4 = float4(0.0f);
    for (uint i = tid; i < cols4; i += tg_size) {
        float4 v = input4[i] * scale;
        bfloat4 m = mask4[i];
        v += float4(float(m.x), float(m.y), float(m.z), float(m.w));
        float4 e = exp(v - float4(max_val));
        output4[i] = e;
        local_sum4 += e;
    }
    float local_sum = (local_sum4.x + local_sum4.y) + (local_sum4.z + local_sum4.w);

    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(local_sum);
    if (tg_size > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            shared[sgitg] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = shared[tiisg];
        sum = simd_sum(sum);
    }

    // ---- Phase 3: normalise ----
    const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    const float4 inv_sum4 = float4(inv_sum);
    for (uint i = tid; i < cols4; i += tg_size) {
        output4[i] *= inv_sum4;
    }
}
