#include <metal_stdlib>
using namespace metal;

/// RMS Normalization — reverse-mode autograd kernels.
///
/// Forward:
///   ms[b]    = (1/D) * Σ_j x[b, j]²
///   r[b]     = rsqrt(ms[b] + eps)              (a per-row scalar)
///   y[b, i]  = x[b, i] * r[b] * w[i]
///
/// Backward — given dy[rows, dim], x[rows, dim], w[dim] →
/// produce dx[rows, dim] and dw[dim] via the analytical identities:
///
///   ∂y[b, i] / ∂w[i]      = x[b, i] * r[b]
///   ∂y[b, i] / ∂x[b, k]   = δ_{i,k} * r[b] * w[i]
///                          - x[b, i] * x[b, k] * r[b]³ * w[i] / D
///
///   dw[i] = Σ_b dy[b, i] * x[b, i] * r[b]
///   dx[b, k] = r[b] * (dy[b, k] * w[k] - x[b, k] * (s[b] * r[b]² / D))
///   where s[b] = Σ_i dy[b, i] * x[b, i] * w[i]
///
/// We split the computation across THREE kernels:
///   1. `rms_norm_compute_rms_inv_f32` — produces r[rows]
///   2. `rms_norm_backward_dx_f32`     — produces dx[rows, dim]
///   3. `rms_norm_backward_dw_f32`     — produces dw[dim]
///
/// Why three? r[b] is reused by both dx and dw; computing it once in
/// a helper avoids redundant TG-wide reductions in the larger
/// kernels, especially for dw which is dim-major (per-feature
/// threadgroups would otherwise re-reduce r[b] for every feature).

/// Helper: compute r[b] = rsqrt(mean(x[b, :]²) + eps) for each row.
///
/// Buffer layout:
///   buffer(0): x       — float[rows, dim]
///   buffer(1): r_out   — float[rows]
///   buffer(2): params  — float2: (eps, dim_f)
///
/// Threadgroup: (tg_size, 1, 1); one TG per row.
/// Grid threadgroups: (rows, 1, 1).
/// Threadgroup shared memory: tg_size * sizeof(float).
kernel void rms_norm_compute_rms_inv_f32(
    device const float *x       [[buffer(0)]],
    device float       *r_out   [[buffer(1)]],
    device const float *params  [[buffer(2)]],
    uint  row_idx [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint  dim = uint(params[1]);
    const uint  base = row_idx * dim;

    // Phase 1: per-thread partial sum of squares.
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = x[base + i];
        partial += v * v;
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        r_out[row_idx] = rsqrt(shared[0] / float(dim) + eps);
    }
}

/// dx[b, k] = r[b] * (dy[b, k] * w[k] - x[b, k] * (s[b] * r[b]² / D))
///   where s[b] = Σ_i dy[b, i] * x[b, i] * w[i]
///
/// Buffer layout:
///   buffer(0): x       — float[rows, dim]
///   buffer(1): w       — float[dim]
///   buffer(2): dy      — float[rows, dim]
///   buffer(3): r       — float[rows]   (precomputed per `rms_norm_compute_rms_inv_f32`)
///   buffer(4): dx      — float[rows, dim]   (output)
///   buffer(5): params  — float2: (dim_f, _padding)
///
/// Threadgroup: (tg_size, 1, 1); one TG per row.
/// Grid threadgroups: (rows, 1, 1).
/// Threadgroup shared memory: tg_size * sizeof(float) (used for the s[b] reduction).
kernel void rms_norm_backward_dx_f32(
    device const float *x       [[buffer(0)]],
    device const float *w       [[buffer(1)]],
    device const float *dy      [[buffer(2)]],
    device const float *r       [[buffer(3)]],
    device float       *dx      [[buffer(4)]],
    device const float *params  [[buffer(5)]],
    uint  row_idx [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const uint  dim = uint(params[0]);
    const uint  base = row_idx * dim;
    const float r_b = r[row_idx];

    // Phase 1: compute s[b] = Σ_i dy[b, i] * x[b, i] * w[i]
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        partial += dy[base + i] * x[base + i] * w[i];
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float s_b = shared[0];
    // Coefficient on the `x[b, k]` term: s[b] * r[b]² / D.
    const float coeff = s_b * r_b * r_b / float(dim);

    // Phase 2: dx[b, k] = r[b] * (dy[b, k] * w[k] - x[b, k] * coeff)
    for (uint k = tid; k < dim; k += tg_size) {
        const float val = r_b * (dy[base + k] * w[k] - x[base + k] * coeff);
        dx[base + k] = val;
    }
}

/// dw[i] = Σ_b dy[b, i] * x[b, i] * r[b]
///
/// Buffer layout:
///   buffer(0): x       — float[rows, dim]
///   buffer(1): dy      — float[rows, dim]
///   buffer(2): r       — float[rows]
///   buffer(3): dw      — float[dim]    (output)
///   buffer(4): params  — float2: (dim_f, rows_f)
///
/// Threadgroup: (tg_size, 1, 1); one TG per FEATURE i.
/// Grid threadgroups: (dim, 1, 1).
/// Threadgroup shared memory: tg_size * sizeof(float).
///
/// Each TG sums over `rows` for its feature; threads stride over `b`,
/// then tree-reduce.
kernel void rms_norm_backward_dw_f32(
    device const float *x       [[buffer(0)]],
    device const float *dy      [[buffer(1)]],
    device const float *r       [[buffer(2)]],
    device float       *dw      [[buffer(3)]],
    device const float *params  [[buffer(4)]],
    uint  feat_idx [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  tg_size  [[threads_per_threadgroup]],
    threadgroup float *shared   [[threadgroup(0)]]
) {
    const uint dim  = uint(params[0]);
    const uint rows = uint(params[1]);

    // Phase 1: per-thread partial sum over rows.
    //   contribution[b] = dy[b, feat_idx] * x[b, feat_idx] * r[b]
    float partial = 0.0f;
    for (uint b = tid; b < rows; b += tg_size) {
        const uint base = b * dim + feat_idx;
        partial += dy[base] * x[base] * r[b];
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        dw[feat_idx] = shared[0];
    }
}
