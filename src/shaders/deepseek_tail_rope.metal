#include <metal_stdlib>
using namespace metal;

struct DeepSeekTailRopeParams {
    uint batch;
    uint seq_len;
    uint heads;
    uint head_dim;
    uint rope_dim;
    uint inverse;
};

#define DEEPSEEK_TAIL_ROPE_BODY(INPUT_TYPE, OUTPUT_TYPE)                         \
    const uint work = gid.x;                                                     \
    const uint vector = gid.y;                                                   \
    const uint vectors = p.batch * p.seq_len * p.heads;                          \
    const uint nope = p.head_dim - p.rope_dim;                                   \
    const uint work_width = nope + p.rope_dim / 2;                               \
    if (vector >= vectors || work >= work_width) return;                         \
    const uint base = vector * p.head_dim;                                       \
    if (work < nope) {                                                           \
        const float value = float(input[base + work]);                           \
        output[base + work] = OUTPUT_TYPE(isfinite(value) ? value : 0.0f);        \
        return;                                                                  \
    }                                                                            \
    const uint pair = work - nope;                                               \
    const uint column = nope + pair * 2;                                         \
    const uint seq = (vector / p.heads) % p.seq_len;                             \
    const float angle = (p.inverse != 0 ? -1.0f : 1.0f)                          \
        * float(positions[seq]) * frequencies[pair];                             \
    const float real = float(input[base + column]);                              \
    const float imag = float(input[base + column + 1]);                          \
    if (!isfinite(angle) || !isfinite(real) || !isfinite(imag)) {                \
        output[base + column] = OUTPUT_TYPE(0.0f);                               \
        output[base + column + 1] = OUTPUT_TYPE(0.0f);                           \
        return;                                                                  \
    }                                                                            \
    const float cosine = cos(angle);                                             \
    const float sine = sin(angle);                                               \
    output[base + column] = OUTPUT_TYPE(real * cosine - imag * sine);            \
    output[base + column + 1] = OUTPUT_TYPE(real * sine + imag * cosine)

kernel void deepseek_tail_rope_f32_to_bf16(
    device const float *input [[buffer(0)]],
    device const uint *positions [[buffer(1)]],
    device const float *frequencies [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant DeepSeekTailRopeParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    DEEPSEEK_TAIL_ROPE_BODY(float, bfloat);
}

kernel void deepseek_tail_rope_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uint *positions [[buffer(1)]],
    device const float *frequencies [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant DeepSeekTailRopeParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    DEEPSEEK_TAIL_ROPE_BODY(bfloat, bfloat);
}

kernel void deepseek_tail_rope_f32_to_f16(
    device const float *input [[buffer(0)]],
    device const uint *positions [[buffer(1)]],
    device const float *frequencies [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant DeepSeekTailRopeParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    DEEPSEEK_TAIL_ROPE_BODY(float, half);
}

kernel void deepseek_tail_rope_f16_to_bf16(
    device const half *input [[buffer(0)]],
    device const uint *positions [[buffer(1)]],
    device const float *frequencies [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant DeepSeekTailRopeParams &p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    DEEPSEEK_TAIL_ROPE_BODY(half, bfloat);
}
