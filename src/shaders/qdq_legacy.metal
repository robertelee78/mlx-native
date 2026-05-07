#include <metal_stdlib>
using namespace metal;

/// Q4_0 quantize-dequantize round-trip in fp32.
///
/// For each block of QK4_0=32 fp32 input values, computes the GGUF
/// Q4_0 quant→dequant round-trip and writes the rounded fp32 values
/// back to the output buffer (same shape as input).  Byte-identical
/// to `quantize_row_q4_0 → dequantize_row_q4_0` from
/// `hf2q::quantize::q_legacy`.
///
/// Per-block formula (matches q_legacy.rs:329 + 380):
///   max  = signed value at the position with largest |.|  (ties go to LOWER index)
///   d    = max / -8.0                  (fp32)
///   id   = (d == 0) ? 0 : 1/d          (fp32)
///   d_h  = float(half(d))              (f16 round-trip — dequantize uses this)
///   For each element v:
///     q  = clamp((v * id + 8.5) as int, 0, 15)
///     dq = (q - 8) * d_h
///
/// Buffer layout:
///   buffer(0): input  — float[num_blocks * 32]
///   buffer(1): output — float[num_blocks * 32]
///
/// Threadgroup: (32, 1, 1) — one block per threadgroup.
/// Grid threadgroups: (num_blocks, 1, 1)
///
/// Threadgroup shared memory: 64 * sizeof(float) = 256 bytes
///   layout: [amax_arr (32 floats), max_arr (32 floats)]
kernel void qdq_q4_0_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    uint  block_idx [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    threadgroup float *shared  [[threadgroup(0)]]
) {
    threadgroup float *amax_arr = shared;          // [32]
    threadgroup float *max_arr  = shared + 32;     // [32]

    const uint base = block_idx * 32u + tid;
    const float v = input[base];
    amax_arr[tid] = fabs(v);
    max_arr[tid]  = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over 32 lanes — keep entry with LARGER amax;
    // on tie, keep LEFT (lower tid) to match CPU `>` semantics
    // (q_legacy.rs:351: `if av > amax { amax = av; max = v; }`).
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float r_amax = amax_arr[tid + stride];
            const float l_amax = amax_arr[tid];
            if (r_amax > l_amax) {
                amax_arr[tid] = r_amax;
                max_arr[tid]  = max_arr[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float max_val = max_arr[0];
    const float d  = max_val / -8.0f;
    const float id = (d == 0.0f) ? 0.0f : (1.0f / d);
    // f16 round-trip — dequant reads d as f16 (q_legacy.rs:358 + d() at decode).
    const float d_h = float(half(d));

    int q = int(floor(v * id + 8.5f));
    q = clamp(q, 0, 15);
    output[base] = float(q - 8) * d_h;
}

/// Q8_0 quantize-dequantize round-trip in fp32.
///
/// Per-block formula (matches q_legacy.rs:149 + 187):
///   amax = max(|v|)                     (fp32)
///   d    = amax / 127.0                 (fp32)
///   id   = (d == 0) ? 0 : 1/d           (fp32)
///   d_h  = float(half(d))               (f16 round-trip — dequant uses this)
///   For each element v:
///     q  = clamp(round(v * id) as int, -128, 127)
///     dq = q * d_h
///
/// Buffer layout:
///   buffer(0): input  — float[num_blocks * 32]
///   buffer(1): output — float[num_blocks * 32]
///
/// Threadgroup: (32, 1, 1) — one block per threadgroup.
/// Grid threadgroups: (num_blocks, 1, 1)
///
/// Threadgroup shared memory: 32 * sizeof(float) = 128 bytes
kernel void qdq_q8_0_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    uint  block_idx [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    threadgroup float *shared  [[threadgroup(0)]]
) {
    const uint base = block_idx * 32u + tid;
    const float v = input[base];
    shared[tid] = fabs(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for amax (max of |v|).
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float r = shared[tid + stride];
            const float l = shared[tid];
            if (r > l) {
                shared[tid] = r;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float amax = shared[0];
    const float d   = amax / 127.0f;
    const float id  = (d == 0.0f) ? 0.0f : (1.0f / d);
    const float d_h = float(half(d));

    // CPU uses `(v * id).round() as i32` then `clamp(-128, 127) as i8`.
    // Metal `rint` is round-to-nearest-even (banker's), but Rust's
    // `f32::round` is round-half-away-from-zero.  Use `floor(x + 0.5)`
    // for positive, `ceil(x - 0.5)` for negative — equivalent to
    // round-half-away-from-zero, matches Rust.
    const float scaled = v * id;
    int q = (scaled >= 0.0f) ? int(floor(scaled + 0.5f))
                              : int(ceil(scaled - 0.5f));
    q = clamp(q, -128, 127);
    output[base] = float(q) * d_h;
}
