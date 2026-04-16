#include <metal_stdlib>
using namespace metal;

// top_k_f32 — return indices and values of the K largest elements.
//
// Used by the Q8 lm_head rerank path to avoid the 1 MB full-logits readback.
// After the Q8 matmul writes the full vocabulary of logits, this kernel
// selects K candidates on GPU; only K * 8 bytes of (index, value) pairs are
// read back to CPU for exact F32 reranking.
//
// Algorithm:
//   Phase 1: one threadgroup of tg_size threads. Each thread strides through
//            the input (ne elements) and maintains a local top-K window in
//            per-thread memory via replace-min insertion.
//   Phase 2: thread 0 performs a K-iteration selection over the tg_size * K
//            concatenated local top-Ks in threadgroup shared memory, emitting
//            the global top-K (unsorted).
//
// Order is NOT guaranteed — the caller does a CPU-side rerank anyway. If
// order matters, sort on the caller side.
//
// Buffer layout:
//   buffer(0): input        — float [ne]
//   buffer(1): out_indices  — uint  [K]
//   buffer(2): out_values   — float [K]
//   buffer(3): params       — uint  [2] = (ne, K)
//
// Threadgroup: (tg_size, 1, 1) — e.g. (32, 1, 1)
// Grid:        (1, 1, 1) — single threadgroup
// Shared mem:  tg_size * K * (sizeof(float) + sizeof(uint)) bytes
//              = tg_size * K * 8 bytes
//              (must fit in Apple Silicon's ~32 KB threadgroup memory)
//
// Constraints:
//   K <= MAX_K (compile-time constant below)
//   tg_size <= 32 to keep shared memory within 32 KB for K=64.
//
// Correctness: tg_size=32, K=64, ne=262144 → each thread scans 8192 elements
// and tracks its local top-64. No thread can hold more than K=64 of the true
// global top-K (since by pigeonhole each thread only sees ne/tg_size elements
// and K ≥ ne/tg_size / 128 = tiny). In practice the global top-K is strictly
// a subset of the union of per-thread local top-Ks.

#ifndef MAX_K
#define MAX_K 128
#endif

kernel void top_k_f32(
    device const float* input       [[buffer(0)]],
    device uint*        out_indices [[buffer(1)]],
    device float*       out_values  [[buffer(2)]],
    device const uint*  params      [[buffer(3)]],
    uint tid         [[thread_index_in_threadgroup]],
    uint tg_size_dyn [[threads_per_threadgroup]],
    threadgroup float* shared_vals [[threadgroup(0)]],  // [tg_size * K]
    threadgroup uint*  shared_idxs [[threadgroup(1)]]   // [tg_size * K]
) {
    const uint ne     = params[0];
    const uint K      = params[1];
    const uint tg_sz  = tg_size_dyn;

    // ---- Phase 1: per-thread local top-K via replace-min insertion ----
    float local_vals[MAX_K];
    uint  local_idxs[MAX_K];
    for (uint k = 0; k < K; k++) {
        local_vals[k] = -INFINITY;
        local_idxs[k] = 0;
    }
    // Track current min of the local top-K window for O(K) replace cost.
    float local_min = -INFINITY;
    uint  local_min_pos = 0;

    for (uint i = tid; i < ne; i += tg_sz) {
        float v = input[i];
        if (v > local_min) {
            local_vals[local_min_pos] = v;
            local_idxs[local_min_pos] = i;
            // Recompute min across the K-sized window.
            float new_min = local_vals[0];
            uint  new_min_pos = 0;
            for (uint k = 1; k < K; k++) {
                if (local_vals[k] < new_min) {
                    new_min = local_vals[k];
                    new_min_pos = k;
                }
            }
            local_min = new_min;
            local_min_pos = new_min_pos;
        }
    }

    // Write local top-K to shared memory at stride tid * K.
    const uint base = tid * K;
    for (uint k = 0; k < K; k++) {
        shared_vals[base + k] = local_vals[k];
        shared_idxs[base + k] = local_idxs[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 2: thread 0 extracts global top-K via K selections ----
    if (tid == 0) {
        const uint total = tg_sz * K;
        for (uint final_k = 0; final_k < K; final_k++) {
            uint  best_pos = 0;
            float best_val = shared_vals[0];
            for (uint i = 1; i < total; i++) {
                float v = shared_vals[i];
                if (v > best_val) {
                    best_val = v;
                    best_pos = i;
                }
            }
            out_indices[final_k] = shared_idxs[best_pos];
            out_values[final_k]  = best_val;
            shared_vals[best_pos] = -INFINITY;  // mark consumed
        }
    }
}
