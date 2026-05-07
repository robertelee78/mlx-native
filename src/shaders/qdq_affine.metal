#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-13b — differentiable affine quant-dequant kernels for the
/// DWQ-proper training loop (Track 2).
///
/// Per mlx-lm's `dwq.py` + the `mx.QuantizedLinear` `unfreeze(keys=
/// ["scales","biases"])` semantics, **q_int is a precomputed FROZEN
/// integer tensor** and the *scales* + *biases* are the learnable
/// per-group affine parameters that flow gradients during DWQ
/// distillation.  The clean, mathematically-correct equivalent is:
///
///   forward:    qdq[i] = q_int[i] * scales[g(i)] + biases[g(i)]
///   backward:   d/d(scales[g]) = Σ_{i ∈ g} q_int[i] · dy[i]
///               d/d(biases[g]) = Σ_{i ∈ g} dy[i]
///
/// where `g(i) = i / group_size`.
///
/// This file ships FOUR kernels:
///
///   1. `qdq_affine_init_f32`         — one-shot init from a frozen FP32
///       weight: per-group min/max → s = (max-min) / (n_bins-1),
///       b = min, q_int = clip(round((w - b) / s), 0, n_bins-1).
///
///   2. `qdq_affine_forward_f32`      — qdq[i] = q_int[i]·s_g + b_g.
///
///   3. `qdq_affine_backward_scales_f32`  — per-group reduction
///       d_scales[g] = Σ q_int[i]·dy[i].
///
///   4. `qdq_affine_backward_biases_f32`  — per-group reduction
///       d_biases[g] = Σ dy[i].
///
/// All kernels are FP32-in / FP32-out for scales/biases/dy/qdq, with
/// `q_int` as `uchar` (one byte per element; supports up to 8-bit
/// quantization without packing — packing is deferred to a later
/// iteration once the differentiable training loop is proven correct).

/// One-shot per-group affine init.  Threadgroup: (group_size, 1, 1) —
/// one tg per group; tg-shared min/max reduction over the group's
/// `group_size` elements.  Grid: (n_groups, 1, 1) tgs.
///
/// Buffers:
///   buffer(0): w        — float[n_groups * group_size]
///   buffer(1): scales   — float[n_groups]   (out)
///   buffer(2): biases   — float[n_groups]   (out)
///   buffer(3): q_int    — uchar[n_groups * group_size]   (out)
///   buffer(4): meta     — uint[2] = { group_size, n_bins }
///
/// Threadgroup shared:  2 * group_size floats (min_arr + max_arr)
kernel void qdq_affine_init_f32(
    device const float *w        [[buffer(0)]],
    device float       *scales   [[buffer(1)]],
    device float       *biases   [[buffer(2)]],
    device uchar       *q_int    [[buffer(3)]],
    device const uint  *meta     [[buffer(4)]],
    uint  block_idx [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    threadgroup float *shared    [[threadgroup(0)]]
) {
    const uint group_size = meta[0];
    const uint n_bins     = meta[1];

    threadgroup float *min_arr = shared;
    threadgroup float *max_arr = shared + group_size;

    const uint base = block_idx * group_size + tid;
    const float v = w[base];
    min_arr[tid] = v;
    max_arr[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over `group_size` lanes for both min and max.  Works
    // for any power-of-two `group_size` <= 1024.
    for (uint stride = group_size >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            min_arr[tid] = min(min_arr[tid], min_arr[tid + stride]);
            max_arr[tid] = max(max_arr[tid], max_arr[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float w_min = min_arr[0];
    const float w_max = max_arr[0];

    // s = (max - min) / (n_bins - 1), b = min.  Degenerate min == max
    // (uniform group) → s := 1.0 (avoids 1/0 in the encode), b := w_min;
    // q_int will be 0 for every element which round-trips correctly to
    // qdq = 0·1 + w_min = w_min = original value (no quant error).
    float s = (w_max - w_min) / float(n_bins - 1u);
    if (!(s > 0.0f)) { s = 1.0f; }
    const float b = w_min;

    if (tid == 0u) {
        scales[block_idx] = s;
        biases[block_idx] = b;
    }

    // Encode this thread's element.  Round-half-away-from-zero to match
    // qdq_legacy / Rust f32::round semantics; saturate to [0, n_bins-1].
    const float z = (v - b) / s;
    int q = (z >= 0.0f) ? int(floor(z + 0.5f)) : int(ceil(z - 0.5f));
    q = clamp(q, 0, int(n_bins - 1u));
    q_int[base] = uchar(q);
}

/// Forward: qdq[i] = q_int[i] · scales[g(i)] + biases[g(i)].
///
/// Threadgroup: (256, 1, 1).  Grid: ceil(n_total / 256) tgs.  Each
/// thread handles one element; group index derived from thread linear
/// index.
///
/// Buffers:
///   buffer(0): q_int   — uchar[n_total]
///   buffer(1): scales  — float[n_groups]
///   buffer(2): biases  — float[n_groups]
///   buffer(3): qdq     — float[n_total]   (out)
///   buffer(4): meta    — uint[2] = { n_total, group_size }
kernel void qdq_affine_forward_f32(
    device const uchar *q_int   [[buffer(0)]],
    device const float *scales  [[buffer(1)]],
    device const float *biases  [[buffer(2)]],
    device float       *qdq     [[buffer(3)]],
    device const uint  *meta    [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n_total    = meta[0];
    const uint group_size = meta[1];
    if (gid >= n_total) return;

    const uint g = gid / group_size;
    const float s = scales[g];
    const float b = biases[g];
    const float q = float(q_int[gid]);
    qdq[gid] = q * s + b;
}

/// Backward w.r.t. scales — one threadgroup per group, tg-shared
/// reduction over the group's `group_size` elements.
///
/// d_scales[g] = Σ_{i ∈ g} q_int[i] · dy[i]
///
/// Buffers:
///   buffer(0): q_int    — uchar[n_groups * group_size]
///   buffer(1): dy       — float[n_groups * group_size]
///   buffer(2): d_scales — float[n_groups]    (out)
///   buffer(3): meta     — uint[1] = { group_size }
///
/// Threadgroup shared: group_size floats.
/// Threadgroup: (group_size, 1, 1).  Grid: (n_groups, 1, 1) tgs.
kernel void qdq_affine_backward_scales_f32(
    device const uchar *q_int    [[buffer(0)]],
    device const float *dy       [[buffer(1)]],
    device float       *d_scales [[buffer(2)]],
    device const uint  *meta     [[buffer(3)]],
    uint  block_idx [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    threadgroup float *shared    [[threadgroup(0)]]
) {
    const uint group_size = meta[0];
    const uint base = block_idx * group_size + tid;
    shared[tid] = float(q_int[base]) * dy[base];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        d_scales[block_idx] = shared[0];
    }
}

/// Backward w.r.t. biases — one threadgroup per group, tg-shared sum
/// reduction over the group's `group_size` elements.
///
/// d_biases[g] = Σ_{i ∈ g} dy[i]
///
/// Buffers:
///   buffer(0): dy        — float[n_groups * group_size]
///   buffer(1): d_biases  — float[n_groups]    (out)
///   buffer(2): meta      — uint[1] = { group_size }
///
/// Threadgroup shared: group_size floats.
/// Threadgroup: (group_size, 1, 1).  Grid: (n_groups, 1, 1) tgs.
kernel void qdq_affine_backward_biases_f32(
    device const float *dy        [[buffer(0)]],
    device float       *d_biases  [[buffer(1)]],
    device const uint  *meta      [[buffer(2)]],
    uint  block_idx [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const uint group_size = meta[0];
    const uint base = block_idx * group_size + tid;
    shared[tid] = dy[base];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        d_biases[block_idx] = shared[0];
    }
}
