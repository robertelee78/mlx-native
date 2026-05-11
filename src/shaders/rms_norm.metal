#include <metal_stdlib>
using namespace metal;

/// RMS Normalization kernel.
///
/// Computes: output = x * rsqrt(mean(x^2) + eps) * weight
/// The mean is computed over the last dimension.
///
/// Buffer layout:
///   buffer(0): input   — float array of shape [rows, dim]
///   buffer(1): weight  — float array of shape [dim]
///   buffer(2): output  — float array of shape [rows, dim]
///   buffer(3): params  — float2: (eps, dim_f)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void rms_norm_f32(
    device const float *input     [[buffer(0)]],
    device const float *weight    [[buffer(1)]],
    device float       *output    [[buffer(2)]],
    device const float *params    [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: compute partial sum of squares
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = input[base + i];
        partial_sum_sq += val * val;
    }

    // Reduction in threadgroup shared memory
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute the normalization factor: rsqrt(mean(x^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize and apply weight
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * rms_inv * weight[i];
    }
}

// ---------------------------------------------------------------------------
// rms_norm_f32_v2 (ADR-028 iter-310) — peer-pattern port.
//
// Replaces our scalar + threadgroup tree-reduction `rms_norm_f32` with:
//   1. float4 vector loads (4× memory throughput per thread)
//   2. simd_sum() in-simdgroup reduction (1 HW op, no barrier)
//   3. inter-simdgroup shuffle via shared memory (just 2 barriers total)
//
// Numerically equivalent to `rms_norm_f32` (same algebra, same f32
// accumulation), but ~2× faster in our hot path per peer's
// `kernel_rms_norm_fuse_impl<float4, 1>` benchmarks.
//
// REQUIREMENT: `dim % 4 == 0`.  All hf2q production shapes meet this
// (gemma4 hidden=3584, qwen3.6 hidden=2048).  Dispatcher must guard
// or fall back to scalar.
//
// Threadgroup geometry: same as scalar rms_norm_f32 — one TG per row,
// `min(256, dim.next_power_of_two())` threads.  Shared memory now
// only needs one float per simdgroup (32 floats max for 1024 threads),
// vs `tg_size * 4` bytes in scalar.
kernel void rms_norm_f32_v2(
    device const float4 *input  [[buffer(0)]],
    device const float4 *weight [[buffer(1)]],
    device float4       *output [[buffer(2)]],
    device const float  *params [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float *shared [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint dim4 = dim / 4u;

    const uint base4 = row_idx * dim4;

    // Phase 1: sum of squares using float4 vector loads.
    float sumf = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 v = input[base4 + i];
        sumf += dot(v, v);
    }

    // In-simdgroup reduction (1 HW op).
    sumf = simd_sum(sumf);

    // Stage per-simdgroup partial sums via threadgroup memory.
    if (tiisg == 0) {
        shared[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across simdgroups in the first simdgroup.
    // Number of active SGs = tg_size / 32 (≤ 32 for tg_size ≤ 1024).
    const uint n_sg = tg_size / 32u;
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize and apply weight, float4 vector store.
    for (uint i = tid; i < dim4; i += tg_size) {
        output[base4 + i] = (input[base4 + i] * rms_inv) * weight[i];
    }
}

// ---------------------------------------------------------------------------
// rms_norm_no_scale_f32_v2 (ADR-028 iter-310) — peer-pattern port,
// no-weight variant.  Same math as rms_norm_no_scale_f32, but float4 +
// simd_sum.  Used for the V-rms-norm site (no learnable weight).
kernel void rms_norm_no_scale_f32_v2(
    device const float4 *input  [[buffer(0)]],
    device float4       *output [[buffer(1)]],
    device const float  *params [[buffer(2)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float *shared [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint dim4 = dim / 4u;

    const uint base4 = row_idx * dim4;

    float sumf = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 v = input[base4 + i];
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);

    if (tiisg == 0) {
        shared[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_sg = tg_size / 32u;
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    for (uint i = tid; i < dim4; i += tg_size) {
        output[base4 + i] = input[base4 + i] * rms_inv;
    }
}

kernel void rms_norm_f16(
    device const half  *input     [[buffer(0)]],
    device const float *weight    [[buffer(1)]],
    device half        *output    [[buffer(2)]],
    device const float *params    [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: accumulate in f32 for numerical stability
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = float(input[base + i]);
        partial_sum_sq += val * val;
    }

    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize, compute in f32, store as f16
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = float(input[base + i]);
        output[base + i] = half(val * rms_inv * weight[i]);
    }
}

kernel void rms_norm_bf16(
    device const bfloat *input     [[buffer(0)]],
    device const bfloat *weight    [[buffer(1)]],
    device bfloat       *output    [[buffer(2)]],
    device const float  *params    [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared      [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: accumulate sum of squares in f32 for numerical stability
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = static_cast<float>(input[base + i]);
        partial_sum_sq += val * val;
    }

    // Reduction in threadgroup shared memory (f32 for accuracy)
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute the normalization factor: rsqrt(mean(x^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize in f32, apply bf16 weight, store as bf16
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = static_cast<float>(input[base + i]);
        output[base + i] = bfloat(val * rms_inv * static_cast<float>(weight[i]));
    }
}

/// RMS Normalization without learned scale (bfloat16).
///
/// Computes: output = input / sqrt(mean(input^2) + eps)
/// No weight multiplication — used for per-head V norm in Gemma 4.
///
/// Buffer layout:
///   buffer(0): input   — bfloat array of shape [rows, dim]
///   buffer(1): output  — bfloat array of shape [rows, dim]
///   buffer(2): params  — float2: (eps, dim_f)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void rms_norm_no_scale_bf16(
    device const bfloat *input     [[buffer(0)]],
    device bfloat       *output    [[buffer(1)]],
    device const float  *params    [[buffer(2)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared      [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: accumulate sum of squares in f32 for numerical stability
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = static_cast<float>(input[base + i]);
        partial_sum_sq += val * val;
    }

    // Reduction in threadgroup shared memory (f32 for accuracy)
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute the normalization factor: rsqrt(mean(x^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize in f32, store as bf16 — NO weight multiply
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = static_cast<float>(input[base + i]);
        output[base + i] = bfloat(val * rms_inv);
    }
}

/// Fused RMS Normalization + elementwise multiply kernel (float32).
///
/// Computes: output = (x * rsqrt(mean(x^2) + eps) * weight) * scale
/// where `weight` is the norm's learned scale and `scale` is an external
/// multiplicand (e.g. the gate output in SwiGLU or a per-element mask).
///
/// This fuses the pattern: rms_norm → barrier → elementwise_mul into a
/// single kernel pass, eliminating one barrier and one global memory
/// round-trip.
///
/// Inspired by llama.cpp's kernel_rms_norm_mul_f32 (ggml-metal.metal),
/// MIT licensed.  Copyright the llama.cpp Authors. See LICENSE-MIT-llamacpp.
/// Adapted for mlx-native's dispatch conventions.
///
/// Buffer layout:
///   buffer(0): input   — float array of shape [rows, dim]
///   buffer(1): weight  — float array of shape [dim] (norm weights)
///   buffer(2): scale   — float array of shape [rows, dim] (MUL operand)
///   buffer(3): output  — float array of shape [rows, dim]
///   buffer(4): params  — float2: (eps, dim_f)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void rms_norm_mul_f32(
    device const float *input     [[buffer(0)]],
    device const float *weight    [[buffer(1)]],
    device const float *scale     [[buffer(2)]],
    device float       *output    [[buffer(3)]],
    device const float *params    [[buffer(4)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: compute partial sum of squares
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = input[base + i];
        partial_sum_sq += val * val;
    }

    // Reduction in threadgroup shared memory
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute the normalization factor: rsqrt(mean(x^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize, apply weight, and multiply by scale
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * rms_inv * weight[i] * scale[base + i];
    }
}

/// Fused RMS Normalization + elementwise multiply kernel (bfloat16).
///
/// Same as rms_norm_mul_f32 but operates on bfloat16 inputs/outputs with
/// f32 accumulation for numerical stability.
///
/// Buffer layout:
///   buffer(0): input   — bfloat array of shape [rows, dim]
///   buffer(1): weight  — bfloat array of shape [dim] (norm weights)
///   buffer(2): scale   — bfloat array of shape [rows, dim] (MUL operand)
///   buffer(3): output  — bfloat array of shape [rows, dim]
///   buffer(4): params  — float2: (eps, dim_f)

kernel void rms_norm_mul_bf16(
    device const bfloat *input     [[buffer(0)]],
    device const bfloat *weight    [[buffer(1)]],
    device const bfloat *scale     [[buffer(2)]],
    device bfloat       *output    [[buffer(3)]],
    device const float  *params    [[buffer(4)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared      [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: accumulate in f32 for numerical stability
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = static_cast<float>(input[base + i]);
        partial_sum_sq += val * val;
    }

    // Reduction in threadgroup shared memory (f32 for accuracy)
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute the normalization factor: rsqrt(mean(x^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize in f32, apply weight and scale, store as bf16
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = static_cast<float>(input[base + i]);
        const float w = static_cast<float>(weight[i]);
        const float s = static_cast<float>(scale[base + i]);
        output[base + i] = bfloat(val * rms_inv * w * s);
    }
}

/// Fused RMS Normalization + elementwise multiply kernel (float16).
///
/// Same as rms_norm_mul_f32 but operates on half inputs/outputs with
/// f32 accumulation for numerical stability.
///
/// Buffer layout:
///   buffer(0): input   — half array of shape [rows, dim]
///   buffer(1): weight  — float array of shape [dim] (norm weights)
///   buffer(2): scale   — half array of shape [rows, dim] (MUL operand)
///   buffer(3): output  — half array of shape [rows, dim]
///   buffer(4): params  — float2: (eps, dim_f)

kernel void rms_norm_mul_f16(
    device const half  *input     [[buffer(0)]],
    device const float *weight    [[buffer(1)]],
    device const half  *scale     [[buffer(2)]],
    device half        *output    [[buffer(3)]],
    device const float *params    [[buffer(4)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: accumulate in f32 for numerical stability
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = float(input[base + i]);
        partial_sum_sq += val * val;
    }

    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize, compute in f32, store as f16
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = float(input[base + i]);
        const float s = float(scale[base + i]);
        output[base + i] = half(val * rms_inv * weight[i] * s);
    }
}

/// RMS Normalization without learned scale (float32).
///
/// Computes: output = input / sqrt(mean(input^2) + eps)
/// No weight multiplication — used for per-head V norm in Gemma 4.
///
/// Buffer layout:
///   buffer(0): input   — float array of shape [rows, dim]
///   buffer(1): output  — float array of shape [rows, dim]
///   buffer(2): params  — float2: (eps, dim_f)
///
/// Threadgroup: (threadgroup_size, 1, 1) — one threadgroup per row
/// Grid threadgroups: (rows, 1, 1)

kernel void rms_norm_no_scale_f32(
    device const float *input     [[buffer(0)]],
    device float       *output    [[buffer(1)]],
    device const float *params    [[buffer(2)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared     [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: accumulate sum of squares
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = input[base + i];
        partial_sum_sq += val * val;
    }

    // Reduction in threadgroup shared memory
    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute the normalization factor: rsqrt(mean(x^2) + eps)
    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: normalize — NO weight multiply
    for (uint i = tid; i < dim; i += tg_size) {
        output[base + i] = input[base + i] * rms_inv;
    }
}

// ---------------------------------------------------------------------------
// rms_norm_no_scale_f32_dual — co-writes bf16 output alongside the f32
// output (ADR-011 Phase 3 Wave P3b-tensor.3).
//
// Used by batched prefill's V-norm path to fuse the f32→bf16 cast that
// previously ran as a separate dispatch.  Same compute as
// rms_norm_no_scale_f32; one extra device write per element.  Memory
// traffic on Apple Silicon's unified memory is bandwidth-bound; the
// extra write is ~free since the f32 result is already in registers.
// ---------------------------------------------------------------------------
kernel void rms_norm_no_scale_f32_dual(
    device const float *input       [[buffer(0)]],
    device float       *output      [[buffer(1)]],
    device const float *params      [[buffer(2)]],
    device bfloat      *output_bf16 [[buffer(3)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared       [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = input[base + i];
        partial_sum_sq += val * val;
    }

    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        const float v = input[base + i] * rms_inv;
        output[base + i]      = v;
        output_bf16[base + i] = bfloat(v);
    }
}

// ---------------------------------------------------------------------------
// rms_norm_no_scale_f32_dual_perm — V-norm that writes bf16 output at the
// permuted [n_heads, seq_len, head_dim] layout instead of the natural
// [seq_len, n_heads, head_dim] layout.  Wave P4.16.
//
// Replaces the V-permute_021_bf16 dispatch (~30/prefill on Gemma 4) by
// having the V-norm itself write directly into the FA-expected head-major
// layout.  Same compute as rms_norm_no_scale_f32_dual; only the bf16
// output index changes.  f32 output (used by KV cache copy downstream)
// remains at natural layout — KV cache copy expects [seq_len, n_heads,
// head_dim] source.
//
// Buffers:
//   0: input        — float [rows * dim]   (rows = seq_len * n_heads)
//   1: output       — float [rows * dim]   (natural layout — KV cache src)
//   2: params       — float [eps, dim]
//   3: output_bf16  — bfloat [rows * dim]  (permuted layout — FA src)
//   4: aux_params   — uint  [n_heads, seq_len]  (permuted-index calc)
//
// Threadgroup: (min(256, next_pow2(dim)), 1, 1) — one threadgroup per row
// Grid       : (rows, 1, 1)
// ---------------------------------------------------------------------------
kernel void rms_norm_no_scale_f32_dual_perm(
    device const float *input       [[buffer(0)]],
    device float       *output      [[buffer(1)]],
    device const float *params      [[buffer(2)]],
    device bfloat      *output_bf16 [[buffer(3)]],
    constant uint2&     aux_params  [[buffer(4)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared       [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);
    const uint n_heads = aux_params.x;
    const uint seq_len = aux_params.y;

    const uint base = row_idx * dim;
    // Permuted bf16 base: rows are laid out [seq_len, n_heads, dim] in
    // input/output, so row_idx = token * n_heads + head.  The permuted
    // bf16 layout is [n_heads, seq_len, dim], so the bf16 base is
    //   head * (seq_len * dim) + token * dim.
    const uint head    = row_idx % n_heads;
    const uint token   = row_idx / n_heads;
    const uint base_bf = head * (seq_len * dim) + token * dim;

    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = input[base + i];
        partial_sum_sq += val * val;
    }

    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    for (uint i = tid; i < dim; i += tg_size) {
        const float v = input[base + i] * rms_inv;
        output[base + i]         = v;
        output_bf16[base_bf + i] = bfloat(v);
    }
}

// ---------------------------------------------------------------------------
// rms_norm_f32_triple — fused 3-output RMS norm with shared compute.
//
// Computes RMS(input) ONCE, then applies three different per-element
// weight vectors to produce three outputs.  Used by hf2q's batched
// prefill at the pre-FF point where the residual stream is normed three
// separate ways (pre_feedforward_layernorm for MLP, pre_feedforward_
// layernorm_2 for MoE input, router_combined_weight for MoE routing).
//
// Wave P4.9 — replaces three separate `rms_norm_f32` dispatches per
// layer (60 dispatches/prefill on Gemma 4) with one shared-compute
// dispatch.  Bandwidth: input is read once instead of three times,
// saving ~40 MB of read traffic per layer at pp2455 (input is
// 2455×2048×4 = 20 MB).
//
// Buffers:
//   0: input    — float [rows * dim]
//   1: weight_a — float [dim]
//   2: weight_b — float [dim]
//   3: weight_c — float [dim]
//   4: output_a — float [rows * dim]
//   5: output_b — float [rows * dim]
//   6: output_c — float [rows * dim]
//   7: params   — float [eps, dim]
//
// Same compute structure as rms_norm_f32 (Phase 1: sum of squares + tree
// reduce; Phase 2: rsqrt + 3 weight multiplies).  The Phase 2 loop
// reads input[i] once, multiplies by rms_inv and three different
// weight[i] values to produce three outputs.
// ---------------------------------------------------------------------------
kernel void rms_norm_f32_triple(
    device const float *input    [[buffer(0)]],
    device const float *weight_a [[buffer(1)]],
    device const float *weight_b [[buffer(2)]],
    device const float *weight_c [[buffer(3)]],
    device float       *output_a [[buffer(4)]],
    device float       *output_b [[buffer(5)]],
    device float       *output_c [[buffer(6)]],
    device const float *params   [[buffer(7)]],
    uint row_idx   [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    threadgroup float *shared    [[threadgroup(0)]]
) {
    const float eps = params[0];
    const uint dim  = uint(params[1]);

    const uint base = row_idx * dim;

    // Phase 1: sum of squares — read input once.
    float partial_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float val = input[base + i];
        partial_sum_sq += val * val;
    }

    shared[tid] = partial_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: read input again, multiply by rms_inv * weight, write
    // three outputs.  Compiler hoists the input load and rms_inv*input
    // factor across the three multiplies.
    for (uint i = tid; i < dim; i += tg_size) {
        const float vn = input[base + i] * rms_inv;
        output_a[base + i] = vn * weight_a[i];
        output_b[base + i] = vn * weight_b[i];
        output_c[base + i] = vn * weight_c[i];
    }
}

// ---------------------------------------------------------------------------
// fused_post_attn_triple_norm_f32 — fuses POST_ATTN_NORM_ADD + TRIPLE_RMS_NORM
// into one dispatch.  Replaces the pair:
//   residual = hidden + norm(attn_out, post_attn_w)           [fused_norm_add_f32]
//   [norm_a, norm_b, norm_c] = triple_norm(residual, w_a/b/c) [rms_norm_f32_triple]
// with one kernel that:
//   1. computes sum(attn_out^2)          — for attn_out rms
//   2. accumulates residual_new = hidden + attn_out*rms_attn*post_attn_w
//      AND sum(residual_new^2) in the same pass
//   3. writes residual_output            — for end-of-layer consumer
//   4. applies three weight vectors to residual * rms_res -> 3 outputs
//
// Eliminates:
//   - One dispatch per layer (30/prefill)
//   - One write+read of pf_residual (~60 MB/layer, 1.8 GB total @ pp2455)
//   - The serialization barrier between the two dispatches
//
// Buffers:
//   0: hidden         — float [rows * dim]  (pre-attention residual stream)
//   1: attn_out       — float [rows * dim]  (attention O-proj output)
//   2: post_attn_w    — float [dim]         (post-attention layernorm weight)
//   3: weight_a       — float [dim]         (pre-FF layernorm 1)
//   4: weight_b       — float [dim]         (pre-FF layernorm 2)
//   5: weight_c       — float [dim]         (router combined weight)
//   6: residual_out   — float [rows * dim]  (hidden + normed_attn, written)
//   7: output_a       — float [rows * dim]
//   8: output_b       — float [rows * dim]
//   9: output_c       — float [rows * dim]
//  10: params         — { float eps; uint dim; }
//
// Grid: (rows, 1, 1). Threadgroup: (min(tg_size, next_pow2(dim)), 1, 1).
// Shared memory: 2 × tg_size × sizeof(float) for two staggered reductions.
// ---------------------------------------------------------------------------

struct FusedPostAttnTripleNormParams {
    float eps;
    uint  dim;
};

kernel void fused_post_attn_triple_norm_f32(
    device const float* hidden       [[buffer(0)]],
    device const float* attn_out     [[buffer(1)]],
    device const float* post_attn_w  [[buffer(2)]],
    device const float* weight_a     [[buffer(3)]],
    device const float* weight_b     [[buffer(4)]],
    device const float* weight_c     [[buffer(5)]],
    device float*       residual_out [[buffer(6)]],
    device float*       output_a     [[buffer(7)]],
    device float*       output_b     [[buffer(8)]],
    device float*       output_c     [[buffer(9)]],
    constant FusedPostAttnTripleNormParams& params [[buffer(10)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const float eps = params.eps;
    const uint  dim = params.dim;
    const uint  base = row_id * dim;

    // Phase 1: sum of squares over attn_out (for the first rms norm).
    float partial_attn_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = attn_out[base + i];
        partial_attn_sq += v * v;
    }
    shared[tid] = partial_attn_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms_inv_attn = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: compute residual_new[i] = hidden[i] + normed_attn[i], write
    // it to residual_out, AND accumulate sum(residual_new^2) for the pre-FF
    // rms normalization.  We re-compute the residual in Phase 4 from
    // residual_out (device-memory re-read, hardware-cached) to avoid
    // needing shared memory for the full row.
    float partial_res_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float a = attn_out[base + i];
        const float normed_attn = a * rms_inv_attn * post_attn_w[i];
        const float r = hidden[base + i] + normed_attn;
        residual_out[base + i] = r;
        partial_res_sq += r * r;
    }
    shared[tid] = partial_res_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms_inv_res = rsqrt(shared[0] / float(dim) + eps);

    // Phase 3: read residual_out back, apply three weight vectors, write
    // three outputs.  Compiler hoists the shared (r * rms_inv_res) factor.
    for (uint i = tid; i < dim; i += tg_size) {
        const float r = residual_out[base + i];
        const float vn = r * rms_inv_res;
        output_a[base + i] = vn * weight_a[i];
        output_b[base + i] = vn * weight_b[i];
        output_c[base + i] = vn * weight_c[i];
    }
}

// ---------------------------------------------------------------------------
// fused_post_attn_triple_norm_f32_v2 (ADR-028 iter-370) — float4 + simd_sum
// rewrite of fused_post_attn_triple_norm_f32 above.
//
// Same math, same buffer layout, same FusedPostAttnTripleNormParams.  Only
// the reductions change: 2 scalar tree reductions (16 barriers at tg=256)
// → 2 simd_sum reductions (4 barriers, 75% reduction).  iter-186 V1
// regressed -1.0% on decode because parallelism loss outweighed dispatch
// savings.  V2 saves 12 barriers/dispatch which might flip the verdict.
//
// Dispatcher must guard `dim % 4 == 0`; gemma4 hidden=2816 (=704 × 4) ✓.
// ---------------------------------------------------------------------------
kernel void fused_post_attn_triple_norm_f32_v2(
    device const float4* hidden       [[buffer(0)]],
    device const float4* attn_out     [[buffer(1)]],
    device const float4* post_attn_w  [[buffer(2)]],
    device const float4* weight_a     [[buffer(3)]],
    device const float4* weight_b     [[buffer(4)]],
    device const float4* weight_c     [[buffer(5)]],
    device float4*       residual_out [[buffer(6)]],
    device float4*       output_a     [[buffer(7)]],
    device float4*       output_b     [[buffer(8)]],
    device float4*       output_c     [[buffer(9)]],
    constant FusedPostAttnTripleNormParams& params [[buffer(10)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const float eps  = params.eps;
    const uint  dim  = params.dim;
    const uint  dim4 = dim / 4u;
    const uint  base4 = row_id * dim4;

    // --- Phase 1: sum of squares over attn_out (FIRST RMS) ---
    float sumf_attn = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 v = attn_out[base4 + i];
        sumf_attn += dot(v, v);
    }
    sumf_attn = simd_sum(sumf_attn);
    if (tiisg == 0) {
        shared[sgitg] = sumf_attn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_sg = tg_size / 32u;
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_attn = rsqrt(shared[0] / float(dim) + eps);

    // --- Phase 2: residual_new = hidden + attn*rms_attn*post_attn_w; write
    //              + accumulate sum(residual_new^2) for SECOND RMS ---
    float sumf_res = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 a = attn_out[base4 + i];
        const float4 normed_attn = (a * rms_inv_attn) * post_attn_w[i];
        const float4 r = hidden[base4 + i] + normed_attn;
        residual_out[base4 + i] = r;
        sumf_res += dot(r, r);
    }
    sumf_res = simd_sum(sumf_res);
    if (tiisg == 0) {
        shared[sgitg] = sumf_res;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_res = rsqrt(shared[0] / float(dim) + eps);

    // --- Phase 3: re-read residual_out, apply 3 weight vectors ---
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 r = residual_out[base4 + i];
        const float4 vn = r * rms_inv_res;
        output_a[base4 + i] = vn * weight_a[i];
        output_b[base4 + i] = vn * weight_b[i];
        output_c[base4 + i] = vn * weight_c[i];
    }
}

// ---------------------------------------------------------------------------
// fused_post_ff_norm2_endlayer_f32 — fuse the gemma4 layer-end pair:
//   (a) mlp_down = attn_out + norm(moe_accum, w2)
//   (b) hidden   = (residual + norm(mlp_down, w3)) * layer_scalar
// into a single dispatch.  ADR-028 iter-217 — bisect-confirmed +2.7%
// throughput on gemma4 default path (saves the (b) launch ≈ 0.34 ms).
//
// Structural template: `fused_post_attn_triple_norm_f32` above (also
// 2 RMS reductions in 1 kernel).  Difference: scalar mul at the end.
//
// ⚠ Risk: iter-186's fused_post_attn_triple_norm REGRESSED on decode
// (-1.0%) because it forced 3 CONCURRENT norms into 1 sequential kernel.
// This kernel fuses 2 SEQUENTIAL norms (different scenario); fusion
// eliminates the second dispatch's launch latency.  iter-218+ will
// bench-validate before shipping default-on.
// ---------------------------------------------------------------------------

struct FusedPostFFNorm2EndlayerParams {
    float eps;
    uint  dim;
    uint  scalar_is_vector;  // 0 = broadcast scalar[0], 1 = per-channel scalar[i]
};

kernel void fused_post_ff_norm2_endlayer_f32(
    device const float* attn_out      [[buffer(0)]],
    device const float* moe_accum     [[buffer(1)]],
    device const float* residual      [[buffer(2)]],
    device const float* w2            [[buffer(3)]],
    device const float* w3            [[buffer(4)]],
    device const float* layer_scalar  [[buffer(5)]],
    device float*       mlp_down      [[buffer(6)]],
    device float*       hidden        [[buffer(7)]],
    constant FusedPostFFNorm2EndlayerParams& params [[buffer(8)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const float eps = params.eps;
    const uint  dim = params.dim;
    const bool  scalar_is_vec = (params.scalar_is_vector != 0u);
    const uint  base = row_id * dim;

    // Phase 1: sum of squares over moe_accum (FIRST RMS norm).
    float partial_moe_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float v = moe_accum[base + i];
        partial_moe_sq += v * v;
    }
    shared[tid] = partial_moe_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms_inv_moe = rsqrt(shared[0] / float(dim) + eps);

    // Phase 2: mlp_down[i] = attn_out + moe_accum * rms_inv_moe * w2[i],
    // write, accumulate sum(mlp_down^2) for SECOND RMS.
    float partial_mlp_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        const float m = moe_accum[base + i];
        const float a = attn_out[base + i];
        const float v = a + m * rms_inv_moe * w2[i];
        mlp_down[base + i] = v;
        partial_mlp_sq += v * v;
    }
    shared[tid] = partial_mlp_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float rms_inv_mlp = rsqrt(shared[0] / float(dim) + eps);

    // Phase 3: hidden[i] = (residual + mlp_down * rms_inv_mlp * w3[i]) * scalar
    for (uint i = tid; i < dim; i += tg_size) {
        const float m = mlp_down[base + i];
        const float r = residual[base + i];
        const float vn = m * rms_inv_mlp;
        const float h = r + vn * w3[i];
        const float s = scalar_is_vec ? layer_scalar[i] : layer_scalar[0];
        hidden[base + i] = h * s;
    }
}

// ---------------------------------------------------------------------------
// fused_post_ff_norm2_endlayer_f32_v2 (ADR-028 iter-362) — float4 + simd_sum
// rewrite of fused_post_ff_norm2_endlayer_f32 above.
//
// Same math, same buffer layout, same FusedPostFFNorm2EndlayerParams.  Only
// the reductions change: scalar tree reduction → simd_sum + per-simdgroup
// partial sum staging (mirrors rms_norm_f32_v2 structural pattern).
//
// Why this matters: original kernel does TWO tree reductions per dispatch,
// each with `log2(tg_size)` barriers (8 barriers at tg=256 → 16 barriers per
// dispatch).  V2 does 2 simd_sum reductions, each with 1 threadgroup_barrier
// for cross-SG broadcast (4 barriers per dispatch, 75% reduction).
//
// Dispatcher must guard `dim % 4 == 0`; gemma4 hidden=3584 (=896 × 4) ✓.
// At dim=3584 with tg=256 = 8 SGs, each SG processes 896/8 = 112 float4s.
// ---------------------------------------------------------------------------
kernel void fused_post_ff_norm2_endlayer_f32_v2(
    device const float4* attn_out      [[buffer(0)]],
    device const float4* moe_accum     [[buffer(1)]],
    device const float4* residual      [[buffer(2)]],
    device const float4* w2            [[buffer(3)]],
    device const float4* w3            [[buffer(4)]],
    device const float*  layer_scalar  [[buffer(5)]],
    device float4*       mlp_down      [[buffer(6)]],
    device float4*       hidden        [[buffer(7)]],
    constant FusedPostFFNorm2EndlayerParams& params [[buffer(8)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const float eps  = params.eps;
    const uint  dim  = params.dim;
    const uint  dim4 = dim / 4u;
    const bool  scalar_is_vec = (params.scalar_is_vector != 0u);
    const uint  base4 = row_id * dim4;

    // --- Phase 1: sum of squares over moe_accum (FIRST RMS norm) ---
    float sumf_moe = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 v = moe_accum[base4 + i];
        sumf_moe += dot(v, v);
    }
    sumf_moe = simd_sum(sumf_moe);
    if (tiisg == 0) {
        shared[sgitg] = sumf_moe;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_sg = tg_size / 32u;
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_moe = rsqrt(shared[0] / float(dim) + eps);

    // --- Phase 2: write mlp_down + accumulate sum(mlp_down^2) for SECOND RMS ---
    float sumf_mlp = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 m = moe_accum[base4 + i];
        const float4 a = attn_out[base4 + i];
        const float4 v = a + (m * rms_inv_moe) * w2[i];
        mlp_down[base4 + i] = v;
        sumf_mlp += dot(v, v);
    }
    sumf_mlp = simd_sum(sumf_mlp);
    if (tiisg == 0) {
        shared[sgitg] = sumf_mlp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_mlp = rsqrt(shared[0] / float(dim) + eps);

    // --- Phase 3: hidden[i] = (residual + mlp_down * rms_inv_mlp * w3[i]) * scalar ---
    if (scalar_is_vec) {
        // layer_scalar is a per-channel vector — read as float4 to vectorize.
        device const float4* layer_scalar4 = (device const float4*)layer_scalar;
        for (uint i = tid; i < dim4; i += tg_size) {
            const float4 m = mlp_down[base4 + i];
            const float4 r = residual[base4 + i];
            const float4 vn = m * rms_inv_mlp;
            const float4 h = r + vn * w3[i];
            hidden[base4 + i] = h * layer_scalar4[i];
        }
    } else {
        const float s = layer_scalar[0];
        for (uint i = tid; i < dim4; i += tg_size) {
            const float4 m = mlp_down[base4 + i];
            const float4 r = residual[base4 + i];
            const float4 vn = m * rms_inv_mlp;
            const float4 h = r + vn * w3[i];
            hidden[base4 + i] = h * s;
        }
    }
}

// ---------------------------------------------------------------------------
// fused_moe_wsum_post_ff_norm2_endlayer_f32_v2 (ADR-028 iter-367)
//
// Fuses moe_weighted_sum INTO the production-default Path A end-of-layer
// kernel (fused_post_ff_norm2_endlayer_f32_v2 above).  Eliminates one dispatch
// per layer (30 dispatches/decode-token on gemma4) and one full memory
// round-trip on the moe_accum buffer.
//
// Replaces the dispatch chain:
//   1. moe_weighted_sum: moe_down_id_out × routing_weights → moe_accum
//   2. fused_post_ff_norm2_endlayer_v2: attn_out + moe_accum + residual + ...
//                                       → mlp_down + hidden
//
// With a single dispatch that:
//   Phase 1: per-thread float4 weighted_sum from experts + routing weights,
//            stash in threadgroup `sum_buf`, accumulate dot(v,v) for first
//            RMS, simd_sum + per-SG reduce → rms_inv_moe.
//   Phase 2: read sum_buf (instead of moe_accum), write mlp_down, accumulate
//            second RMS as before.
//   Phase 3: hidden = (residual + mlp_down * rms_inv_mlp * w3) * scalar.
//
// Threadgroup memory: max(32, n_sg) + dim floats (~14.5 KB at gemma4 dim=3584).
// Under 32 KB Apple budget.
//
// Dispatcher must guard `dim % 4 == 0`; gemma4 hidden=3584 (=896 × 4) ✓.
// Parity tolerance: max_rel < 1e-4 (V2 simd_sum reduction order vs chain's
//                   V2 reduction order — small f32 rounding deltas).
// ---------------------------------------------------------------------------
struct FusedMoeWsumPostFFNorm2EndlayerParams {
    float eps;
    uint  dim;
    uint  top_k;
    uint  scalar_is_vector;
};

kernel void fused_moe_wsum_post_ff_norm2_endlayer_f32_v2(
    device const float4* expert_outputs   [[buffer(0)]],  // [top_k, dim/4] (per-row)
    device const float*  routing_weights  [[buffer(1)]],  // [top_k]
    device const float4* attn_out         [[buffer(2)]],  // [dim/4]
    device const float4* residual         [[buffer(3)]],  // [dim/4]
    device const float4* w2               [[buffer(4)]],  // [dim/4]
    device const float4* w3               [[buffer(5)]],  // [dim/4]
    device const float*  layer_scalar     [[buffer(6)]],  // [1] or [dim]
    device float4*       mlp_down         [[buffer(7)]],  // [dim/4]
    device float4*       hidden           [[buffer(8)]],  // [dim/4]
    constant FusedMoeWsumPostFFNorm2EndlayerParams& params [[buffer(9)]],
    uint row_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const float eps    = params.eps;
    const uint  dim    = params.dim;
    const uint  top_k  = params.top_k;
    const uint  dim4   = dim / 4u;
    const bool  scalar_is_vec = (params.scalar_is_vector != 0u);
    const uint  base4_row = row_id * dim4;
    const uint  base_eo4  = row_id * top_k * dim4;
    const uint  base_w    = row_id * top_k;

    // Threadgroup memory layout:
    //   shared[0 .. max(32, n_sg))      = SG reduction scratch
    //   shared[max(32, n_sg) .. +dim)   = sum_buf (per-element weighted_sum)
    // Use max(32, n_sg) so simd_sum's sgitg-indexed write is never OOB even
    // for partial-warp tg_sizes (mirrors fused_norm_add_f32_v2 pattern).
    const uint n_sg = tg_size / 32u;
    const uint sg_scratch_floats = max(32u, n_sg);
    threadgroup float* sg_scratch = shared;
    threadgroup float4* sum_buf4  = (threadgroup float4*)(shared + sg_scratch_floats);

    // --- Phase 1: weighted_sum + sum-of-squares for FIRST RMS ---
    float sumf_moe = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        float4 v = float4(0.0f);
        for (uint k = 0; k < top_k; ++k) {
            const float w = routing_weights[base_w + k];
            v += expert_outputs[base_eo4 + k * dim4 + i] * w;
        }
        sum_buf4[i] = v;
        sumf_moe += dot(v, v);
    }
    sumf_moe = simd_sum(sumf_moe);
    if (tiisg == 0) {
        sg_scratch[sgitg] = sumf_moe;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? sg_scratch[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            sg_scratch[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_moe = rsqrt(sg_scratch[0] / float(dim) + eps);

    // --- Phase 2: write mlp_down + sum-of-squares for SECOND RMS ---
    float sumf_mlp = 0.0f;
    for (uint i = tid; i < dim4; i += tg_size) {
        const float4 m = sum_buf4[i];
        const float4 a = attn_out[base4_row + i];
        const float4 v = a + (m * rms_inv_moe) * w2[i];
        mlp_down[base4_row + i] = v;
        sumf_mlp += dot(v, v);
    }
    sumf_mlp = simd_sum(sumf_mlp);
    if (tiisg == 0) {
        sg_scratch[sgitg] = sumf_mlp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? sg_scratch[tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            sg_scratch[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms_inv_mlp = rsqrt(sg_scratch[0] / float(dim) + eps);

    // --- Phase 3: hidden = (residual + mlp_down * rms_inv_mlp * w3) * scalar ---
    if (scalar_is_vec) {
        device const float4* layer_scalar4 = (device const float4*)layer_scalar;
        for (uint i = tid; i < dim4; i += tg_size) {
            const float4 m = mlp_down[base4_row + i];
            const float4 r = residual[base4_row + i];
            const float4 vn = m * rms_inv_mlp;
            const float4 h = r + vn * w3[i];
            hidden[base4_row + i] = h * layer_scalar4[i];
        }
    } else {
        const float s = layer_scalar[0];
        for (uint i = tid; i < dim4; i += tg_size) {
            const float4 m = mlp_down[base4_row + i];
            const float4 r = residual[base4_row + i];
            const float4 vn = m * rms_inv_mlp;
            const float4 h = r + vn * w3[i];
            hidden[base4_row + i] = h * s;
        }
    }
}
