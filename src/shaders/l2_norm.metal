#include <metal_stdlib>
using namespace metal;

// L2 Normalization kernel.
//
// Computes: output = x / sqrt(sum(x^2) + eps)
// The sum is computed over the last dimension (per-row).
//
// Spec source: ADR-013 Decision 3. Formula derived from the mathematical
// definition of L2 normalization (x / ||x||_2, with epsilon for stability).
// Used by Gated DeltaNet on Q and K after conv1d state update
// (delta-net-base.cpp:320-321 references; no code copied).
//
// Buffer layout:
//   buffer(0): input   - array of shape [rows, dim]  (element dtype varies)
//   buffer(1): output  - array of shape [rows, dim]
//   buffer(2): params  - float2: (eps, dim_f)
//
// Threadgroup: (threadgroup_size, 1, 1) - one threadgroup per row
// Grid threadgroups: (rows, 1, 1)
//
// Accumulation is always performed in f32 for numerical stability, regardless
// of the input dtype (matches ADR-011 convention).

kernel void l2_norm_f32(
    device const float *input   [[buffer(0)]],
    device float       *output  [[buffer(1)]],
    device const float *params  [[buffer(2)]],
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint base = row_idx * dim;

    // Phase 1: compute partial sum of squares in f32.
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = input[base + i];
        partial += v * v;
    }

    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // L2 norm uses sum-of-squares (not mean-of-squares like RMS norm).
    const float inv = rsqrt(shared[0] + eps);

    // Phase 2: write normalized output.
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * inv;
    }
}

kernel void l2_norm_f16(
    device const half  *input   [[buffer(0)]],
    device half        *output  [[buffer(1)]],
    device const float *params  [[buffer(2)]],
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint base = row_idx * dim;

    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = float(input[base + i]);
        partial += v * v;
    }

    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(shared[0] + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = half(float(input[base + i]) * inv);
    }
}

kernel void l2_norm_bf16(
    device const bfloat *input   [[buffer(0)]],
    device bfloat       *output  [[buffer(1)]],
    device const float  *params  [[buffer(2)]],
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared    [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint base = row_idx * dim;

    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = float(input[base + i]);
        partial += v * v;
    }

    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(shared[0] + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = bfloat(float(input[base + i]) * inv);
    }
}

// ---------------------------------------------------------------------------
// l2_norm_scale_f32 — fused L2 normalization with scalar multiply.
//
// ADR-015 iter59a — fuses the `dispatch_l2_norm` + `scalar_mul_f32` pair on
// the DeltaNet q-path into a single dispatch. Eliminates one dispatch per
// DN layer per prefill chunk (and per decode token).
//
// Computes: output = (x / sqrt(sum(x^2) + eps)) * scale
// The sum is computed over the last dimension (per-row).
//
// Same compute structure and numerics as `l2_norm_f32` followed by an
// elementwise scalar multiply; the scale is folded into the Phase 2 store
// so the L2-normalized intermediate never round-trips through device
// memory.  Matches CPU reference to within fp32 roundoff (1e-6 typical).
//
// Buffer layout:
//   buffer(0): input   - float [rows, dim]
//   buffer(1): output  - float [rows, dim]
//   buffer(2): params  - float3: (eps, dim_f, scale)
//
// Threadgroup: (threadgroup_size, 1, 1) - one threadgroup per row.
// Grid threadgroups: (rows, 1, 1).
//
// Kept as a separate kernel (rather than a templated parameter on
// l2_norm_f32) so existing l2_norm callers do not pay any extra param-buf
// register pressure.

kernel void l2_norm_scale_f32(
    device const float *input   [[buffer(0)]],
    device float       *output  [[buffer(1)]],
    device const float *params  [[buffer(2)]],
    uint row_idx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const float eps   = params[0];
    const uint  dim   = uint(params[1]);
    const float scale = params[2];
    const uint  base  = row_idx * dim;

    // Phase 1: compute partial sum of squares in f32.
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = input[base + i];
        partial += v * v;
    }

    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // L2 norm uses sum-of-squares (not mean-of-squares like RMS norm).
    const float inv = rsqrt(shared[0] + eps);

    // Phase 2: write scaled-normalized output.
    //
    // Bit-identity vs the unfused `l2_norm_f32` + `scalar_mul_f32` path is
    // required so iter59a's wired-in fused kernel does not flip
    // greedy-T=0 token-cliffs through 1-ulp drift in the GDN delta-rule
    // recurrence.  The unfused path goes
    //
    //     intermediate = input * inv      (l2_norm_f32, written to DRAM)
    //     output       = intermediate * scale  (scalar_mul_f32, separate dispatch)
    //
    // The DRAM round-trip between kernels forces the intermediate to be
    // rounded to a single f32 representation before the scale multiply.
    // We mirror that here with a two-pass write/read: store `input * inv`
    // to the output buffer, fence with a device-memory barrier (so the
    // Metal compiler cannot fold the two multiplies into a single FMA in
    // a register), then read the f32-rounded intermediate back and apply
    // the scale.  One extra device-memory write per element vs the
    // single-multiply form, which is still a net win at the dispatch
    // level (~30 µs per saved dispatch × dispatches eliminated >> ~1 µs
    // extra write bandwidth on M5 Max unified memory).
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * inv;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = output[base + i] * scale;
    }
}
