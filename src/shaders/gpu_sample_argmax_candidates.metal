#include <metal_stdlib>
using namespace metal;

/// ADR-040 §26 iter-M — GPU-side first-max argmax + threshold candidate collect.
///
/// Per slot (one threadgroup per slot, grid = (N,1,1)), over the slot's row of
/// the [N, vocab] post-softcap logits buffer:
///   1. first-max argmax — max VALUE, then LOWEST index achieving it. Byte-matches
///      the host `argmax_f32_first_max` (fold max + first index; strict `>` in the
///      strided local scan keeps the lowest in-stride index; the tree reduce breaks
///      ties by ACTUAL index, not slot position — the existing argmax_f32 kernel is
///      NOT first-max-safe because it ties by slot).
///   2. candidate collect — ids where logits[j] >= top1_val - 0.5f (exact f32),
///      atomically appended into a per-slot capped buffer; overflow flagged.
///
/// Output (small — replaces the full [N,vocab] readback):
///   out_top1_idx[N], out_top1_val[N], out_cand_count[N] (atomic), out_overflow[N],
///   out_cand_ids[N*CAP].  params = [vocab, cap].
///
/// Threadgroup: (tg_size,1,1), tg_size a power of two (1024). Shared: vals+idxs.
kernel void gpu_sample_argmax_candidates(
    device const float*  logits         [[buffer(0)]],
    device uint*         out_top1_idx   [[buffer(1)]],
    device float*        out_top1_val   [[buffer(2)]],
    device atomic_uint*  out_cand_count [[buffer(3)]],
    device uint*         out_overflow   [[buffer(4)]],
    device uint*         out_cand_ids   [[buffer(5)]],
    device const uint*   params         [[buffer(6)]],
    uint slot     [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared_vals [[threadgroup(0)]],
    threadgroup uint*  shared_idxs [[threadgroup(1)]]
) {
    const uint vocab = params[0];
    const uint cap   = params[1];
    device const float* row = logits + (ulong)slot * (ulong)vocab;

    // Phase 1 — per-thread local first-max over strided columns.
    float local_max = -INFINITY;
    uint  local_idx = 0;
    for (uint j = tid; j < vocab; j += tg_size) {
        float v = row[j];
        if (v > local_max) { local_max = v; local_idx = j; }
    }
    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduce — keep larger value; on EQUAL value keep the LOWER index.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float ov = shared_vals[tid + stride];
            uint  oi = shared_idxs[tid + stride];
            float cv = shared_vals[tid];
            uint  ci = shared_idxs[tid];
            if (ov > cv || (ov == cv && oi < ci)) {
                shared_vals[tid] = ov;
                shared_idxs[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float top1_val = shared_vals[0];
    const uint  top1_idx = shared_idxs[0];

    if (tid == 0) {
        out_top1_idx[slot] = top1_idx;
        out_top1_val[slot] = top1_val;
        out_overflow[slot] = 0u;
        atomic_store_explicit(&out_cand_count[slot], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // Phase 2 — collect candidates >= top1_val - 0.5f.
    const float threshold = top1_val - 0.5f;
    for (uint j = tid; j < vocab; j += tg_size) {
        if (row[j] >= threshold) {
            uint pos = atomic_fetch_add_explicit(&out_cand_count[slot], 1u, memory_order_relaxed);
            if (pos < cap) {
                out_cand_ids[(ulong)slot * (ulong)cap + (ulong)pos] = j;
            } else {
                out_overflow[slot] = 1u;
            }
        }
    }
}
