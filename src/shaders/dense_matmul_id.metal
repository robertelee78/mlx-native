// Native scalar expert-ID matrix multiplication.
//
// Weight storage stays F32, F16, or BF16 exactly as supplied. F32 inputs and
// outputs are explicit. The direct kernel covers decode and repeat-allowed
// routing in one dispatch. The grouped path compacts distinct per-token expert
// selections, then reuses each expert weight tile across its routed rows.
// Both paths widen native scalar weights at the multiply and preserve F32
// activations without an intermediate F16/BF16 rounding step.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

struct DenseMatmulIdParams {
    uint m;
    uint n;
    uint k;
    uint top_k;
    uint n_experts;
    uint input_layout;
    uint64_t expert_stride_bytes;
    uint64_t input_token_stride_bytes;
    uint64_t input_slot_stride_bytes;
};

// Direct and grouped execution intentionally share these helpers.  This is
// the value-independence boundary used by activation route plans: for one
// exact dtype/shape/stride/layout/multiplicity and compiled pipeline set, both
// routes form the same input and weight indices, widen the same native scalar
// value to F32, apply the same F32 fused multiply-add, and finish with the same
// simd_sum reduction.  The grouped kernel may reorder independent routed rows
// and stage already-widened weights, but it cannot change scalar arithmetic.
inline uint64_t dense_matmul_id_input_byte_offset(
        constant DenseMatmulIdParams & params,
        uint flat) {
    const uint token = flat / params.top_k;
    const uint slot = flat - token * params.top_k;
    return uint64_t(token) * params.input_token_stride_bytes
        + uint64_t(slot) * params.input_slot_stride_bytes;
}

inline uint64_t dense_matmul_id_weight_scalar_index(
        constant DenseMatmulIdParams & params,
        uint col,
        uint inner) {
    return uint64_t(col) * params.k + inner;
}

inline uint64_t dense_matmul_id_expert_byte_offset(
        constant DenseMatmulIdParams & params,
        uint expert) {
    return uint64_t(expert) * params.expert_stride_bytes;
}

template<typename T>
inline float dense_matmul_id_widen_weight(T value) {
    return float(value);
}

inline float dense_matmul_id_f32_madd(
        float sum,
        float weight,
        float activation) {
    return fma(weight, activation, sum);
}

template<typename T>
kernel void dense_matmul_id_direct_impl(
        constant DenseMatmulIdParams & params [[buffer(0)]],
        device const char * weights          [[buffer(1)]],
        device const float * input           [[buffer(2)]],
        device const uint * expert_ids       [[buffer(3)]],
        device float * output                [[buffer(4)]],
        uint3 tgpig                          [[threadgroup_position_in_grid]],
        ushort lane                          [[thread_index_in_simdgroup]],
        ushort simdgroup                     [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_simdgroup = 4;
    constexpr uint simdgroups_per_threadgroup = 2;
    constexpr uint outputs_per_threadgroup =
        rows_per_simdgroup * simdgroups_per_threadgroup;

    const uint flat = uint(tgpig.y);
    const uint total_rows = params.m * params.top_k;
    if (flat >= total_rows) {
        return;
    }

    const uint expert = expert_ids[flat];
    const uint first_col =
        (uint(tgpig.x) * outputs_per_threadgroup)
        + uint(simdgroup) * rows_per_simdgroup;
    if (first_col >= params.n) {
        return;
    }

    const uint64_t input_byte_offset =
        dense_matmul_id_input_byte_offset(params, flat);
    device const float * input_row =
        (device const float *)((device const char *)input + input_byte_offset);

    float sums[rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (expert < params.n_experts) {
        device const T * expert_weight =
            (device const T *)(weights
                + dense_matmul_id_expert_byte_offset(params, expert));
        for (uint inner = uint(lane); inner < params.k; inner += 32u) {
            const float activation = input_row[inner];
            #pragma clang loop unroll(full)
            for (uint row = 0; row < rows_per_simdgroup; ++row) {
                const uint col = first_col + row;
                if (col < params.n) {
                    const float weight = dense_matmul_id_widen_weight(
                        expert_weight[dense_matmul_id_weight_scalar_index(
                            params, col, inner)]);
                    sums[row] = dense_matmul_id_f32_madd(
                        sums[row], weight, activation);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < rows_per_simdgroup; ++row) {
        const float total = simd_sum(sums[row]);
        const uint col = first_col + row;
        if (lane == 0 && col < params.n) {
            // Invalid expert IDs are defended in-kernel and deterministically
            // produce zero instead of forming an out-of-bounds weight address.
            output[uint64_t(flat) * params.n + col] = total;
        }
    }
}

template [[host_name("dense_matmul_id_direct_bf16_f32")]]
kernel void dense_matmul_id_direct_impl<bfloat>(
    constant DenseMatmulIdParams &, device const char *, device const float *,
    device const uint *, device float *, uint3, ushort, ushort);

template [[host_name("dense_matmul_id_direct_f16_f32")]]
kernel void dense_matmul_id_direct_impl<half>(
    constant DenseMatmulIdParams &, device const char *, device const float *,
    device const uint *, device float *, uint3, ushort, ushort);

template [[host_name("dense_matmul_id_direct_f32_f32")]]
kernel void dense_matmul_id_direct_impl<float>(
    constant DenseMatmulIdParams &, device const char *, device const float *,
    device const uint *, device float *, uint3, ushort, ushort);

// One thread owns one expert. Distinct-per-token routing bounds every expert
// list at M entries. The selected value stored in routed_rows is the flattened
// (token, slot) row, preserving both supported input layouts and output order.
kernel void dense_matmul_id_map_distinct(
        constant DenseMatmulIdParams & params [[buffer(0)]],
        device const uint * expert_ids       [[buffer(1)]],
        device uint * expert_counts          [[buffer(2)]],
        device uint * routed_rows            [[buffer(3)]],
        device float * output                [[buffer(4)]],
        uint expert                           [[thread_position_in_grid]]) {
    if (expert >= params.n_experts) {
        return;
    }
    uint count = 0;
    for (uint token = 0; token < params.m; ++token) {
        const uint row = token * params.top_k;
        for (uint slot = 0; slot < params.top_k; ++slot) {
            const uint flat = row + slot;
            if (expert_ids[flat] == expert) {
                routed_rows[uint64_t(expert) * params.m + count] = flat;
                ++count;
                break;
            }
        }
    }
    expert_counts[expert] = count;

    // The direct route deterministically zeros an invalid expert row. Match
    // that full-overwrite behavior without ever forming a weight address.
    // Valid calls only pay the small ID scan; the O(N) write is malformed-input
    // defense and is owned by one thread to avoid races.
    if (expert == 0) {
        const uint total_rows = params.m * params.top_k;
        for (uint flat = 0; flat < total_rows; ++flat) {
            if (expert_ids[flat] >= params.n_experts) {
                for (uint col = 0; col < params.n; ++col) {
                    output[uint64_t(flat) * params.n + col] = 0.0f;
                }
            }
        }
    }
}

template<typename T>
kernel void dense_matmul_id_grouped_impl(
        constant DenseMatmulIdParams & params [[buffer(0)]],
        device const char * weights          [[buffer(1)]],
        device const float * input           [[buffer(2)]],
        device const uint * expert_counts    [[buffer(3)]],
        device const uint * routed_rows      [[buffer(4)]],
        device float * output                [[buffer(5)]],
        threadgroup char * shmem             [[threadgroup(0)]],
        uint3 tgpig                          [[threadgroup_position_in_grid]],
        ushort thread_index                  [[thread_index_in_threadgroup]],
        ushort lane                          [[thread_index_in_simdgroup]],
        ushort simdgroup                     [[simdgroup_index_in_threadgroup]]) {
    constexpr uint output_cols = 8;
    constexpr uint routed_rows_per_threadgroup = 8;
    constexpr uint inner_tile = 128;

    const uint expert = uint(tgpig.z);
    const uint first_col = uint(tgpig.y) * output_cols;
    const uint first_routed = uint(tgpig.x) * routed_rows_per_threadgroup;
    const uint routed_count = expert_counts[expert];
    if (first_routed >= routed_count || first_col >= params.n) {
        return;
    }

    const uint routed_index = first_routed + uint(simdgroup);
    const bool active = routed_index < routed_count;
    uint flat = 0;
    device const float * input_row = input;
    if (active) {
        flat = routed_rows[uint64_t(expert) * params.m + routed_index];
        const uint64_t input_byte_offset =
            dense_matmul_id_input_byte_offset(params, flat);
        input_row =
            (device const float *)((device const char *)input + input_byte_offset);
    }
    device const T * expert_weight =
        (device const T *)(weights
            + dense_matmul_id_expert_byte_offset(params, expert));
    threadgroup float * weight_tile = (threadgroup float *)shmem;

    float sums[output_cols] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    for (uint loop_k = 0; loop_k < params.k; loop_k += inner_tile) {
        for (uint inner_offset = uint(thread_index); inner_offset < inner_tile;
             inner_offset += 256u) {
            const uint absolute_k = loop_k + inner_offset;
            #pragma clang loop unroll(full)
            for (uint col_offset = 0; col_offset < output_cols; ++col_offset) {
                const uint col = first_col + col_offset;
                weight_tile[col_offset * inner_tile + inner_offset] =
                    col < params.n && absolute_k < params.k
                    ? dense_matmul_id_widen_weight(
                        expert_weight[dense_matmul_id_weight_scalar_index(
                            params, col, absolute_k)])
                    : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint inner_offset = uint(lane); inner_offset < inner_tile;
                 inner_offset += 32u) {
                const uint absolute_k = loop_k + inner_offset;
                const float activation =
                    absolute_k < params.k ? input_row[absolute_k] : 0.0f;
                #pragma clang loop unroll(full)
                for (uint col_offset = 0; col_offset < output_cols; ++col_offset) {
                    sums[col_offset] = dense_matmul_id_f32_madd(
                        sums[col_offset],
                        weight_tile[col_offset * inner_tile + inner_offset],
                        activation);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint col_offset = 0; col_offset < output_cols; ++col_offset) {
        const float total = simd_sum(sums[col_offset]);
        const uint col = first_col + col_offset;
        if (active && lane == 0 && col < params.n) {
            output[uint64_t(flat) * params.n + col] = total;
        }
    }
}

template [[host_name("dense_matmul_id_grouped_bf16_f32")]]
kernel void dense_matmul_id_grouped_impl<bfloat>(
    constant DenseMatmulIdParams &, device const char *, device const float *,
    device const uint *, device const uint *, device float *, threadgroup char *,
    uint3, ushort, ushort, ushort);

template [[host_name("dense_matmul_id_grouped_f16_f32")]]
kernel void dense_matmul_id_grouped_impl<half>(
    constant DenseMatmulIdParams &, device const char *, device const float *,
    device const uint *, device const uint *, device float *, threadgroup char *,
    uint3, ushort, ushort, ushort);

template [[host_name("dense_matmul_id_grouped_f32_f32")]]
kernel void dense_matmul_id_grouped_impl<float>(
    constant DenseMatmulIdParams &, device const char *, device const float *,
    device const uint *, device const uint *, device float *, threadgroup char *,
    uint3, ushort, ushort, ushort);
