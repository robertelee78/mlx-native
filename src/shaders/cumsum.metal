#include <metal_stdlib>
using namespace metal;

// Inclusive prefix sum (cumulative sum) along the last axis.
//
// Computes: out[r, i] = sum(x[r, 0..=i]) for every row r independently.
//
// Spec source: ADR-013 Decision 4. Formula derived from the definition of
// an inclusive prefix scan; no code copied from llama.cpp.
//
// Algorithm: per-row Hillis-Steele scan using threadgroup shared memory.
// One threadgroup per row; each thread owns CHUNK contiguous elements, loaded
// into private memory, reduced locally, then a Hillis-Steele scan runs across
// thread-local totals. Finally each thread adds the exclusive prefix of
// preceding threads' totals to its chunk and writes outputs.
//
// Buffer layout:
//   buffer(0): input   - shape [rows, dim]
//   buffer(1): output  - shape [rows, dim]
//   buffer(2): params  - uint2: (dim, tg_size)
//
// Threadgroup shape: (tg_size, 1, 1) - caller picks tg_size so that
//   tg_size * CHUNK >= dim. CHUNK is computed by the caller as
//   ceil_div(dim, tg_size). Kernel reads CHUNK from params[2] if present,
//   otherwise derives it from dim / tg_size at runtime.
//
// Accumulation is performed in f32 regardless of input dtype for numerical
// stability (critical for Gated DeltaNet's decay-mask which multiplies these
// sums later).

// Maximum per-thread chunk size. A threadgroup of 256 threads × 32 elements
// per thread handles dim up to 8192 in a single pass. Larger dims can be
// supported by the caller increasing tg_size up to 1024 (hardware max) or
// tiling across multiple kernel launches.
#define CUMSUM_MAX_CHUNK 32

kernel void cumsum_f32(
    device const float *input   [[buffer(0)]],
    device float       *output  [[buffer(1)]],
    device const uint  *params  [[buffer(2)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const uint dim = params[0];
    const uint base = row_idx * dim;

    // Per-thread chunk bounds. Each thread owns a contiguous range
    // [lo, hi) of the row.
    const uint chunk = (dim + tg_size - 1u) / tg_size;
    const uint lo = min(tid * chunk, dim);
    const uint hi = min(lo + chunk, dim);
    const uint len = hi - lo;

    // Load chunk into private memory and compute local prefix sum.
    thread float local_buf[CUMSUM_MAX_CHUNK];
    float local_sum = 0.0f;
    for (uint i = 0; i < len; ++i) {
        local_sum += float(input[base + lo + i]);
        local_buf[i] = local_sum;
    }
    // Thread's contribution to the row-wide running total.
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele INCLUSIVE scan across thread totals in shared memory.
    // Uses a temporary second buffer slot pattern: write to (tid + tg_size).
    // shared[0..tg_size)      = current values
    // shared[tg_size..2*tg_size) = previous-iteration values
    for (uint offset = 1u; offset < tg_size; offset <<= 1u) {
        float v = shared[tid];
        if (tid >= offset) {
            v += shared[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // shared[tid] now holds the inclusive prefix over threads 0..=tid of
    // their local sums. Exclusive prefix (needed to offset this thread's
    // local buf) is shared[tid] - local_sum = shared[tid-1] for tid>0.
    const float exclusive = (tid == 0u) ? 0.0f : shared[tid - 1u];

    for (uint i = 0; i < len; ++i) {
        output[base + lo + i] = local_buf[i] + exclusive;
    }
}

kernel void cumsum_bf16(
    device const bfloat *input   [[buffer(0)]],
    device bfloat       *output  [[buffer(1)]],
    device const uint   *params  [[buffer(2)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float *shared    [[threadgroup(0)]]
) {
    const uint dim = params[0];
    const uint base = row_idx * dim;

    const uint chunk = (dim + tg_size - 1u) / tg_size;
    const uint lo = min(tid * chunk, dim);
    const uint hi = min(lo + chunk, dim);
    const uint len = hi - lo;

    thread float local_buf[CUMSUM_MAX_CHUNK];
    float local_sum = 0.0f;
    for (uint i = 0; i < len; ++i) {
        local_sum += float(input[base + lo + i]);
        local_buf[i] = local_sum;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1u; offset < tg_size; offset <<= 1u) {
        float v = shared[tid];
        if (tid >= offset) {
            v += shared[tid - offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float exclusive = (tid == 0u) ? 0.0f : shared[tid - 1u];

    for (uint i = 0; i < len; ++i) {
        output[base + lo + i] = bfloat(local_buf[i] + exclusive);
    }
}
