// Staged AF4G64 draft projections for Apple10-class GPUs.
//
// The tile and its split, walk and one-pass gate/up structure are derived
// from Splash's Apache-2.0 staged GGUF decode tile
// (https://github.com/incoai/splash, commit 031a25a6:
// runtime/metal/kernels/common/gguf_staged_tile.h and
// runtime/metal/kernels/shared/gguf_linear.metal).
//
// Weights are the prepared AF4G64 planes of
// qmm_affine_t_packed_simd4_b4_gs64_bf16.metal: codes
// [N / 64][K / 32][64][4 words], metadata [N / 64][K / 64][64] packed as
// BF16 scale | BF16 bias << 16. A weight is scale * code + bias, rounded once
// to half. Activations are BF16 rows read in place; sums are FP32.
//
// A threadgroup owns 64 output columns (two simdgroups of 32) and one run of
// K. Lane l of a simdgroup dequantizes column l of one 32-input step into the
// simdgroup's half stage, and the simdgroup multiplies the step's activation
// slice by the stage with matmul2d. Runs are added in run order by the reduce
// kernel, so an output's bits depend on its own activation row, its weight
// row and the split count alone.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint kStorageRows = 64;
constant constexpr int kColumns = 32;
constant constexpr int kStep = 32;
constant constexpr uint kStageHalves = kColumns * kStep;

struct Af4StagedParams {
    uint k;
    uint n;
    uint splits;
    uint spread;
};

struct Af4ReduceParams {
    uint rows;
    uint n;
    uint splits;
};

inline uint4 nibble_pairs(uint word) {
    return (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu;
}

// 0x6400 is half 1024, whose ulp is 1: a 4-bit code or-ed into its mantissa is
// 1024 + code exactly.
inline half2 affine_pair(uint pair, float scale, float bias) {
    const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024)));
    return half2(fma(code, float2(scale), float2(bias)));
}

// The step a column tile starts its walk over [begin, begin + per) at. In
// lockstep every tile fetches the same activation slice at once; spread, tile
// t starts (t * 40503 mod 2^16) / 2^16 of the way in.
inline uint first_step(uint tile, uint begin, uint per, uint spread) {
    return spread ? begin + uint(ulong((tile * 40503u) & 0xFFFFu) * per >> 16) : begin;
}

// One weight row's planes, the payload of the step it stages next and that
// step's scale and bias.
struct Af4Column {
    device const uint *codes;
    device const uint *metadata;
    uint4 payload;
    uint coefficients;

    void begin(device const uint *code_plane, device const uint *metadata_plane, uint k,
               uint row, uint first) thread {
        const ulong groups = k / 32;
        const ulong tile = row / kStorageRows, in_tile = row % kStorageRows;
        codes = code_plane + (tile * groups * kStorageRows + in_tile) * 4;
        metadata = metadata_plane + tile * (groups / 2) * kStorageRows + in_tile;
        payload = *((device const uint4 *)(codes + ulong(first) * kStorageRows * 4));
        coefficients = metadata[ulong(first / 2) * kStorageRows];
    }

    void stage(threadgroup half *column) thread {
        const float scale = float(as_type<bfloat>(ushort(coefficients & 0xFFFFu)));
        const float bias = float(as_type<bfloat>(ushort(coefficients >> 16)));
#pragma unroll
        for (ushort chunk = 0; chunk < 4; ++chunk) {
            const uint4 pairs = nibble_pairs(payload[chunk]);
            *((threadgroup half4 *)(column + 4 * chunk)) =
                half4(affine_pair(pairs.x, scale, bias), affine_pair(pairs.y, scale, bias));
            *((threadgroup half4 *)(column + 16 + 4 * chunk)) =
                half4(affine_pair(pairs.z, scale, bias), affine_pair(pairs.w, scale, bias));
        }
    }

    // After `step` is staged: the payload of `next`, and its coefficients when
    // it enters another 64-input group.
    void load(uint step, uint next) thread {
        payload = *((device const uint4 *)(codes + ulong(next) * kStorageRows * 4));
        if (next / 2 != step / 2) coefficients = metadata[ulong(next / 2) * kStorageRows];
    }
};

template <ushort Rows>
kernel void dflash2_af4g64_staged_impl(
    constant Af4StagedParams &p     [[buffer(0)]],
    device const uint *codes        [[buffer(1)]],
    device const uint *metadata     [[buffer(2)]],
    device bfloat *input            [[buffer(3)]],
    device float *output            [[buffer(4)]],
    device float *partials          [[buffer(5)]],
    threadgroup half *shared        [[threadgroup(0)]],
    uint2 group                     [[threadgroup_position_in_grid]],
    ushort tid                      [[thread_index_in_threadgroup]]) {
    const uint simdgroup = tid / 32, lane = tid % 32;
    threadgroup half *stage0 = shared + simdgroup * 2 * kStageHalves;
    threadgroup half *stage1 = stage0 + kStageHalves;
    const uint origin = group.y * 64 + simdgroup * kColumns;
    const uint per = p.k / kStep / p.splits;
    const uint begin = group.x * per, end = begin + per;

    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>{int(p.k), Rows}, array<int32_t, 2>{1, int(p.k)});
    auto w0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        stage0, dextents<int32_t, 2>{kStep, kColumns}, array<int32_t, 2>{1, kStep});
    auto w1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        stage1, dextents<int32_t, 2>{kStep, kColumns}, array<int32_t, 2>{1, kStep});
    matmul2d<matmul2d_descriptor(Rows, kColumns, kStep, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<1>> multiply;
    auto left0 = x.template slice<kStep, Rows>(0, 0);
    auto right0 = w0.template slice<kStep, kColumns>(0, 0);
    auto right1 = w1.template slice<kStep, kColumns>(0, 0);
    auto sum = multiply.template get_destination_cooperative_tensor<
        decltype(left0), decltype(right0), float>();
#pragma unroll
    for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.0f;

    uint step = first_step(group.y, begin, per, p.spread);
    Af4Column column;
    column.begin(codes, metadata, p.k, origin + lane, step);
    for (uint walked = 0; walked < per; ++walked) {
        const bool odd = walked & 1;
        column.stage((odd ? stage1 : stage0) + lane * kStep);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint next = step + 1 < end ? step + 1 : begin;
        if (walked + 1 < per) column.load(step, next);
        auto left = x.template slice<kStep, Rows>(step * kStep, 0);
        if (odd) multiply.run(left, right1, sum);
        else multiply.run(left, right0, sum);
        step = next;
    }

    device float *destination =
        (p.splits == 1 ? output : partials + ulong(group.x) * Rows * p.n) + origin;
#pragma unroll
    for (ushort i = 0; i < sum.get_capacity(); ++i) {
        if (!sum.is_valid_element(i)) continue;
        const auto index = sum.get_multidimensional_index(i);
        destination[ulong(index[1]) * p.n + index[0]] = sum[i];
    }
}

typedef decltype(dflash2_af4g64_staged_impl<32>) dflash2_af4g64_staged_t;
template [[host_name("dflash2_af4g64_staged_m8")]]
kernel dflash2_af4g64_staged_t dflash2_af4g64_staged_impl<8>;
template [[host_name("dflash2_af4g64_staged_m16")]]
kernel dflash2_af4g64_staged_t dflash2_af4g64_staged_impl<16>;
template [[host_name("dflash2_af4g64_staged_m32")]]
kernel dflash2_af4g64_staged_t dflash2_af4g64_staged_impl<32>;

// Adds the runs of one row tile in run order.
kernel void dflash2_af4g64_staged_reduce(
    device const float *partials    [[buffer(0)]],
    device float *output            [[buffer(1)]],
    constant Af4ReduceParams &p     [[buffer(2)]],
    uint i                          [[thread_position_in_grid]]) {
    const uint elements = p.rows * p.n;
    if (i >= elements) return;
    float sum = 0.0f;
    for (uint split = 0; split < p.splits; ++split) sum += partials[ulong(split) * elements + i];
    output[i] = sum;
}

inline bfloat gated(float gate, float up) {
    return bfloat(gate / (1.0f + exp(-gate)) * up);
}

// Gate and up in one dispatch: a threadgroup of one simdgroup stages the same
// 32 columns of both tensors, so each activation slice is fetched once for the
// pair. Each tensor sums its steps in the order of its own projection, and
// the output is BF16 SiLU(gate) * up. Partials are [split][row][2 * n]: the
// gate's columns, then the up's.
template <ushort Rows>
kernel void dflash2_af4g64_staged_gate_up_impl(
    constant Af4StagedParams &p     [[buffer(0)]],
    device const uint *gate_codes   [[buffer(1)]],
    device const uint *gate_meta    [[buffer(2)]],
    device const uint *up_codes     [[buffer(3)]],
    device const uint *up_meta      [[buffer(4)]],
    device bfloat *input            [[buffer(5)]],
    device bfloat *output           [[buffer(6)]],
    device float *partials          [[buffer(7)]],
    threadgroup half *shared        [[threadgroup(0)]],
    uint2 group                     [[threadgroup_position_in_grid]],
    ushort lane                     [[thread_index_in_threadgroup]]) {
    threadgroup half *gate_stage = shared;
    threadgroup half *up_stage = shared + kStageHalves;
    const uint origin = group.y * kColumns;
    const uint per = p.k / kStep / p.splits;
    const uint begin = group.x * per, end = begin + per;

    auto x = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>{int(p.k), Rows}, array<int32_t, 2>{1, int(p.k)});
    auto wg = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        gate_stage, dextents<int32_t, 2>{kStep, kColumns}, array<int32_t, 2>{1, kStep});
    auto wu = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        up_stage, dextents<int32_t, 2>{kStep, kColumns}, array<int32_t, 2>{1, kStep});
    matmul2d<matmul2d_descriptor(Rows, kColumns, kStep, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<1>> multiply;
    auto left0 = x.template slice<kStep, Rows>(0, 0);
    auto right_gate = wg.template slice<kStep, kColumns>(0, 0);
    auto right_up = wu.template slice<kStep, kColumns>(0, 0);
    auto gate = multiply.template get_destination_cooperative_tensor<
        decltype(left0), decltype(right_gate), float>();
    auto up = multiply.template get_destination_cooperative_tensor<
        decltype(left0), decltype(right_gate), float>();
#pragma unroll
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
        gate[i] = 0.0f;
        up[i] = 0.0f;
    }

    // The pair walks from the step its 64-column tile walks from.
    uint step = first_step(group.y / 2, begin, per, p.spread);
    Af4Column gate_column, up_column;
    gate_column.begin(gate_codes, gate_meta, p.k, origin + lane, step);
    up_column.begin(up_codes, up_meta, p.k, origin + lane, step);
    for (uint walked = 0; walked < per; ++walked) {
        gate_column.stage(gate_stage + lane * kStep);
        up_column.stage(up_stage + lane * kStep);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint next = step + 1 < end ? step + 1 : begin;
        if (walked + 1 < per) {
            gate_column.load(step, next);
            up_column.load(step, next);
        }
        auto left = x.template slice<kStep, Rows>(step * kStep, 0);
        multiply.run(left, right_gate, gate);
        multiply.run(left, right_up, up);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        step = next;
    }

#pragma unroll
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
        if (!gate.is_valid_element(i)) continue;
        const auto index = gate.get_multidimensional_index(i);
        const uint row = index[1], column = origin + index[0];
        if (p.splits == 1) {
            output[ulong(row) * p.n + column] = gated(gate[i], up[i]);
        } else {
            device float *partial =
                partials + (ulong(group.x) * Rows + row) * (2 * p.n) + column;
            partial[0] = gate[i];
            partial[p.n] = up[i];
        }
    }
}

typedef decltype(dflash2_af4g64_staged_gate_up_impl<32>) dflash2_af4g64_staged_gate_up_t;
template [[host_name("dflash2_af4g64_staged_gate_up_m8")]]
kernel dflash2_af4g64_staged_gate_up_t dflash2_af4g64_staged_gate_up_impl<8>;
template [[host_name("dflash2_af4g64_staged_gate_up_m16")]]
kernel dflash2_af4g64_staged_gate_up_t dflash2_af4g64_staged_gate_up_impl<16>;
template [[host_name("dflash2_af4g64_staged_gate_up_m32")]]
kernel dflash2_af4g64_staged_gate_up_t dflash2_af4g64_staged_gate_up_impl<32>;

// Adds the gate and up runs of one row tile in run order and applies the gate.
kernel void dflash2_af4g64_staged_gate_up_reduce(
    device const float *partials    [[buffer(0)]],
    device bfloat *output           [[buffer(1)]],
    constant Af4ReduceParams &p     [[buffer(2)]],
    uint i                          [[thread_position_in_grid]]) {
    if (i >= p.rows * p.n) return;
    const uint row = i / p.n, column = i % p.n;
    float gate = 0.0f, up = 0.0f;
    for (uint split = 0; split < p.splits; ++split) {
        device const float *partial =
            partials + (ulong(split) * p.rows + row) * (2 * p.n) + column;
        gate += partial[0];
        up += partial[p.n];
    }
    output[i] = gated(gate, up);
}
