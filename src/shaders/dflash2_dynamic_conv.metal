#include <metal_stdlib>
using namespace metal;

struct DFlash2DynamicConvParams {
    uint lanes;
    uint rows_per_lane;
    uint hidden;
    uint groups;
    uint stage;
};

template <bool BF16_BOUNDARY>
kernel void dflash2_dynamic_conv_impl(
    device const float *input [[buffer(0)]],
    device const float *dynamic_weights [[buffer(1)]],
    device const bfloat *base_kernel [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant DFlash2DynamicConvParams &params [[buffer(5)]],
    uint element [[thread_position_in_grid]]) {
    const uint total_rows = params.lanes * params.rows_per_lane;
    const uint total = total_rows * params.hidden;
    if (element >= total) return;

    const uint row = element / params.hidden;
    const uint position = row % params.rows_per_lane;
    const uint channel = element % params.hidden;
    const uint group_size = params.hidden / params.groups;
    const uint group = channel / group_size;
    const uint stage_tap = params.stage * 2;
    const uint dynamic_width = 4 * params.groups;

    float value = input[element] *
        (float(base_kernel[stage_tap * params.hidden + channel]) +
         dynamic_weights[row * dynamic_width + stage_tap * params.groups + group]);
    if (position > 0) {
        value += input[element - params.hidden] *
            (float(base_kernel[(stage_tap + 1) * params.hidden + channel]) +
             dynamic_weights[row * dynamic_width + (stage_tap + 1) * params.groups + group]);
    }
    if (params.stage == 1) value += residual[element];
    output[element] = BF16_BOUNDARY ? float(bfloat(value)) : value;
}

template [[host_name("dflash2_dynamic_conv_f32")]]
kernel void dflash2_dynamic_conv_impl<false>(
    device const float *, device const float *, device const bfloat *,
    device const float *, device float *, constant DFlash2DynamicConvParams &, uint);
template [[host_name("dflash2_dynamic_conv_f32_bf16_boundary")]]
kernel void dflash2_dynamic_conv_impl<true>(
    device const float *, device const float *, device const bfloat *,
    device const float *, device float *, constant DFlash2DynamicConvParams &, uint);
