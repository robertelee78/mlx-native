#include <metal_stdlib>
using namespace metal;

// Portions derived from Splash (https://github.com/incoai/splash), commit
// 031a25a6125af2d7acf7096580afe56eb33d65bc: common/rms_inverse.h and
// shared/normalization.metal. Copyright the Splash authors.
// Apache-2.0; see LICENSE-APACHE-splash at the repository root.
// Storage remains F32; no caller operand is rounded or widened here. The
// completed weighted normalized result alone crosses the BF16 boundary.
kernel void dflash2_rms_norm_bf16(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *params [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp reassociate(off)
    const ulong base = ulong(row) * dim;
    float sum = 0.0f;
    for (uint column = tid; column < dim; column += 256) {
        const float value = input[base + column];
        sum += value * value;
    }
    sum = simd_sum(sum);
    threadgroup float partials[8];
    if (lane == 0) partials[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint group = 0; group < 8; ++group) total += partials[group];
        partials[0] = rsqrt(total / float(dim) + params[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse = partials[0];
    for (uint column = tid; column < dim; column += 256) {
        const float value = (input[base + column] * inverse) * weight[column];
        output[base + column] = float(bfloat(value));
    }
}
