#include <metal_stdlib>
using namespace metal;

struct DeepSeekMxfp8Params {
    uint rows;
    uint row_width;
    uint quantized_width;
    uint block_size;
};

inline float pow2_ceil_positive(float value) {
    const uint bits = as_type<uint>(value);
    const int exponent = int((bits >> 23) & 0xffu) - 127
        + int((bits & 0x7fffffu) != 0u);
    return as_type<float>(uint(exponent + 127) << 23);
}

inline float quantize_e4m3fn(float value) {
    const float magnitude = min(fabs(value), 448.0f);
    float step;
    if (magnitude < 0.015625f) {
        step = 0.001953125f;
    } else {
        const uint bits = as_type<uint>(magnitude);
        const int exponent = int((bits >> 23) & 0xffu) - 127;
        step = as_type<float>(uint(exponent - 3 + 127) << 23);
    }
    return copysign(min(rint(magnitude / step) * step, 448.0f), value);
}

inline float quantize_e2m1(float value) {
    const float x = min(fabs(value), 6.0f);
    float rounded;
    if (x <= 0.25f) rounded = 0.0f;
    else if (x < 0.75f) rounded = 0.5f;
    else if (x <= 1.25f) rounded = 1.0f;
    else if (x < 1.75f) rounded = 1.5f;
    else if (x <= 2.5f) rounded = 2.0f;
    else if (x < 3.5f) rounded = 3.0f;
    else if (x <= 5.0f) rounded = 4.0f;
    else rounded = 6.0f;
    return copysign(rounded, value);
}

kernel void deepseek_mxfp8_fake_quant_bf16(
        constant DeepSeekMxfp8Params &p [[buffer(0)]],
        device bfloat *data [[buffer(1)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    threadgroup float scales[16];
    const uint blocks = p.quantized_width / p.block_size;
    const ulong base = ulong(row) * p.row_width;
    if (tid < blocks) {
        float maximum = 0.0f;
        bool invalid = false;
        const uint start = tid * p.block_size;
        for (uint i = 0; i < p.block_size; ++i) {
            const float value = float(data[base + start + i]);
            invalid |= !isfinite(value);
            maximum = max(maximum, fabs(value));
        }
        maximum = max(maximum, 1.0e-4f);
        scales[tid] = invalid ? -1.0f : pow2_ceil_positive(maximum / 448.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < p.quantized_width; column += 256) {
        const float scale = scales[column / p.block_size];
        const float value = float(data[base + column]);
        const float result = scale > 0.0f
            ? quantize_e4m3fn(value / scale) * scale
            : 0.0f;
        data[base + column] = bfloat(isfinite(result) ? result : 0.0f);
    }
}

kernel void deepseek_hadamard_mxfp4_bf16(
        constant uint &rows [[buffer(0)]],
        device bfloat *data [[buffer(1)]],
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    if (row >= rows || tid >= 128) return;
    const ulong base = ulong(row) * 128;
    const float input = float(data[base + tid]);
    shared[tid] = isfinite(input) ? input : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 128; stride <<= 1) {
        const uint partner = tid ^ stride;
        if (tid < partner) {
            const float left = shared[tid];
            const float right = shared[partner];
            shared[tid] = left + right;
            shared[partner] = left - right;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    shared[tid] *= 0.08838834764831845f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float scales[4];
    if (tid < 4) {
        float maximum = 0.0f;
        const uint start = tid * 32;
        for (uint i = 0; i < 32; ++i) maximum = max(maximum, fabs(shared[start + i]));
        maximum = max(maximum, 7.052966104933725e-38f);
        scales[tid] = pow2_ceil_positive(maximum / 6.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = scales[tid / 32];
    const float result = quantize_e2m1(shared[tid] / scale) * scale;
    data[base + tid] = bfloat(isfinite(result) ? result : 0.0f);
}
