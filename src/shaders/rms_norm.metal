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
