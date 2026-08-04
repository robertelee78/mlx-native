// DeepSeek-V4 Hyper-Connection kernels.
//
// Row-major public layouts:
//   mixes    [token, 24]
//   pre/post [token, 4]
//   comb     [token, source, destination]
//   residual/output [token, branch, embedding]
//
// The split kernel implements the official DeepSeek-V4 order exactly:
// row softmax + eps, column normalization, then 19 row/column rounds.

#include <metal_stdlib>
using namespace metal;

struct HcSplitParams {
    uint n_tokens;
};

struct HcVectorParams {
    uint n_tokens;
    uint n_embd;
};

constant uint HC = 4;
constant uint MIX_WIDTH = 24;
constant uint SINKHORN_ITERS = 20;
constant float HC_EPS = 1.0e-6f;

inline float hc_sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

kernel void deepseek_hc_split_sinkhorn_f32(
        constant HcSplitParams &params       [[buffer(0)]],
        device const float *mixes            [[buffer(1)]],
        device const float *scale             [[buffer(2)]],
        device const float *base              [[buffer(3)]],
        device float *pre                     [[buffer(4)]],
        device float *post                    [[buffer(5)]],
        device float *comb                    [[buffer(6)]],
        uint3 threadgroup_id                  [[threadgroup_position_in_grid]],
        ushort lane                           [[thread_index_in_simdgroup]],
        ushort simdgroup_id                   [[simdgroup_index_in_threadgroup]],
        ushort3 threads_per_threadgroup       [[threads_per_threadgroup]]) {
    const uint token = threadgroup_id.x * threads_per_threadgroup.y + simdgroup_id;
    if (token >= params.n_tokens) {
        return;
    }

    float scale_lane = lane < 3 ? scale[lane] : 0.0f;
    const float scale_pre = simd_shuffle(scale_lane, 0);
    const float scale_post = simd_shuffle(scale_lane, 1);
    const float scale_comb = simd_shuffle(scale_lane, 2);

    float raw = 0.0f;
    float transformed = 0.0f;
    uint invalid = 0;
    if (lane < MIX_WIDTH) {
        const uint index = token * MIX_WIDTH + lane;
        raw = mixes[index];
        const float bias = base[lane];
        const float lane_scale = lane < HC ? scale_pre : (lane < 2 * HC ? scale_post : scale_comb);
        invalid = (!isfinite(raw) || !isfinite(bias) || !isfinite(lane_scale)) ? 1u : 0u;
        transformed = fma(raw, lane_scale, bias);
        invalid |= !isfinite(transformed) ? 1u : 0u;
    }

    // Any invalid value closes the entire token: no partial matrix or branch
    // weights are allowed to escape into the model state.
    if (simd_sum(invalid) != 0) {
        if (lane < HC) {
            pre[token * HC + lane] = 0.0f;
            post[token * HC + lane] = 0.0f;
        }
        if (lane < HC * HC) {
            comb[token * HC * HC + lane] = 0.0f;
        }
        return;
    }

    // Shuffle while all lanes are active. Values sourced from lanes that do
    // not execute a shuffle are undefined on Metal.
    const float post_value = simd_shuffle(transformed, min(lane + HC, 2 * HC - 1));
    const float comb_value = simd_shuffle(transformed, min(lane + 2 * HC, MIX_WIDTH - 1));
    if (lane < HC) {
        pre[token * HC + lane] = hc_sigmoid(transformed) + HC_EPS;
        post[token * HC + lane] = 2.0f * hc_sigmoid(post_value);
    }

    // Lanes 0..15 own [source, destination] in row-major order.
    float value = lane < HC * HC ? comb_value : 0.0f;

    // Softmax over destinations: four contiguous lanes per source row.
    float row_max = max(value, simd_shuffle_xor(value, 1));
    row_max = max(row_max, simd_shuffle_xor(row_max, 2));
    value = exp(value - row_max);
    float sum = value + simd_shuffle_xor(value, 1);
    sum += simd_shuffle_xor(sum, 2);
    value = value / sum + HC_EPS;

    // Equal destination indices are four lanes apart.
    sum = value + simd_shuffle_xor(value, 4);
    sum += simd_shuffle_xor(sum, 8);
    value /= sum + HC_EPS;

    for (uint iteration = 1; iteration < SINKHORN_ITERS; ++iteration) {
        sum = value + simd_shuffle_xor(value, 1);
        sum += simd_shuffle_xor(sum, 2);
        value /= sum + HC_EPS;

        sum = value + simd_shuffle_xor(value, 4);
        sum += simd_shuffle_xor(sum, 8);
        value /= sum + HC_EPS;
    }

    if (lane < HC * HC) {
        comb[token * HC * HC + lane] = isfinite(value) ? value : 0.0f;
    }
}

kernel void deepseek_hc_pre_f32(
        constant HcVectorParams &params      [[buffer(0)]],
        device const float *x                 [[buffer(1)]],
        device const float *weights           [[buffer(2)]],
        device float *output                  [[buffer(3)]],
        uint3 threadgroup_id                  [[threadgroup_position_in_grid]],
        ushort lane                           [[thread_index_in_simdgroup]],
        ushort simdgroup_id                   [[simdgroup_index_in_threadgroup]],
        ushort3 threads_per_threadgroup       [[threads_per_threadgroup]]) {
    const uint token = threadgroup_id.y;
    const uint embedding =
        ((threadgroup_id.x * threads_per_threadgroup.y + simdgroup_id) * 32) + lane;

    float weight_lane = lane < HC ? weights[token * HC + lane] : 0.0f;
    const uint invalid_weights = simd_sum(lane < HC && !isfinite(weight_lane) ? 1u : 0u);
    float weight_reg[HC];
    for (uint source = 0; source < HC; ++source) {
        weight_reg[source] = simd_shuffle(weight_lane, source);
    }
    if (embedding >= params.n_embd) {
        return;
    }
    if (invalid_weights != 0) {
        output[token * params.n_embd + embedding] = 0.0f;
        return;
    }

    float result = 0.0f;
    bool invalid = false;
    for (uint source = 0; source < HC; ++source) {
        const float value = x[(token * HC + source) * params.n_embd + embedding];
        invalid |= !isfinite(value);
        result = fma(value, weight_reg[source], result);
    }
    output[token * params.n_embd + embedding] =
        (!invalid && isfinite(result)) ? result : 0.0f;
}

kernel void deepseek_hc_post_f32(
        constant HcVectorParams &params      [[buffer(0)]],
        device const float *x                 [[buffer(1)]],
        device const float *residual          [[buffer(2)]],
        device const float *post              [[buffer(3)]],
        device const float *comb              [[buffer(4)]],
        device float *output                  [[buffer(5)]],
        uint3 threadgroup_id                  [[threadgroup_position_in_grid]],
        ushort lane                           [[thread_index_in_simdgroup]],
        ushort simdgroup_id                   [[simdgroup_index_in_threadgroup]],
        ushort3 threads_per_threadgroup       [[threads_per_threadgroup]]) {
    const uint token = threadgroup_id.y;
    const uint embedding =
        ((threadgroup_id.x * threads_per_threadgroup.y + simdgroup_id) * 32) + lane;

    float coefficient = 0.0f;
    if (lane < HC) {
        coefficient = post[token * HC + lane];
    } else if (lane < HC + HC * HC) {
        coefficient = comb[token * HC * HC + lane - HC];
    }
    const uint invalid_coefficients = simd_sum(
        lane < HC + HC * HC && !isfinite(coefficient) ? 1u : 0u);
    float post_reg[HC];
    float comb_reg[HC][HC];
    for (uint destination = 0; destination < HC; ++destination) {
        post_reg[destination] = simd_shuffle(coefficient, destination);
    }
    for (uint source = 0; source < HC; ++source) {
        for (uint destination = 0; destination < HC; ++destination) {
            comb_reg[source][destination] =
                simd_shuffle(coefficient, HC + source * HC + destination);
        }
    }
    if (embedding >= params.n_embd) {
        return;
    }
    if (invalid_coefficients != 0) {
        for (uint destination = 0; destination < HC; ++destination) {
            output[(token * HC + destination) * params.n_embd + embedding] = 0.0f;
        }
        return;
    }

    const float x_value = x[token * params.n_embd + embedding];
    bool invalid = !isfinite(x_value);
    float result[HC];
    for (uint destination = 0; destination < HC; ++destination) {
        result[destination] = x_value * post_reg[destination];
    }
    for (uint source = 0; source < HC; ++source) {
        const float residual_value =
            residual[(token * HC + source) * params.n_embd + embedding];
        invalid |= !isfinite(residual_value);
        for (uint destination = 0; destination < HC; ++destination) {
            result[destination] =
                fma(residual_value, comb_reg[source][destination], result[destination]);
        }
    }
    for (uint destination = 0; destination < HC; ++destination) {
        const float value = result[destination];
        output[(token * HC + destination) * params.n_embd + embedding] =
            (!invalid && isfinite(value)) ? value : 0.0f;
    }
}
