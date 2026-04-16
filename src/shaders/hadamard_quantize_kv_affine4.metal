// hadamard_quantize_kv_affine4.metal — FWHT + symmetric affine INT4 quantize
//
// Same SIMD shuffle FWHT as hadamard_quantize_kv_fast.metal, but replaces
// Lloyd-Max codebook quantization with symmetric uniform (affine) quantization.
//
// Encode:
//   1. FWHT rotate
//   2. Find absmax of rotated vector
//   3. Quantize: index = round(7.5 * (1 + x/absmax)), clamp [0, 15]
//   4. Pack nibbles
//   5. Store norm = absmax * sqrt(head_dim) / 7.5
//
// Decode (in SDPA kernel):
//   value = norm * rsqrt(head_dim) * (float(nibble) - 7.5)
//         = (absmax / 7.5) * (nibble - 7.5)
//         = absmax * (nibble/7.5 - 1)
//
// This maps [-absmax, +absmax] uniformly to 16 levels.
// Compared to Lloyd-Max: slightly higher MSE for Gaussian inputs,
// but decode is 1 FMA instead of 1 indexed lookup + 1 multiply.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct HadamardQuantizeParams {
    uint head_dim;
    uint num_kv_heads;
    uint write_pos;
    uint cache_capacity;
    uint is_sliding;
};

inline void butterfly_local(thread float &a, thread float &b) {
    float sum = a + b;
    float diff = a - b;
    a = sum;
    b = diff;
}

template<ushort EPT>
inline void fwht_simd(thread float *elems, uint lane) {
    for (ushort h = 1; h < EPT; h <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            ushort partner = i ^ h;
            if (partner > i) {
                butterfly_local(elems[i], elems[partner]);
            }
        }
    }

    for (ushort delta = 1; delta < 32; delta <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            float partner_val = simd_shuffle_xor(elems[i], delta);
            if (lane & delta) {
                elems[i] = partner_val - elems[i];
            } else {
                elems[i] = elems[i] + partner_val;
            }
        }
    }
}

template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_affine4(
    device const float             *src    [[buffer(0)]],
    device       uint8_t           *packed [[buffer(1)]],
    device       float             *norms  [[buffer(2)]],
    constant HadamardQuantizeParams &params [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgid;
    const uint lane = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    // 1. Load elements.
    const uint src_base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = src[src_base + i];
    }

    // 2. FWHT via SIMD shuffle.
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize by 1/sqrt(head_dim).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= inv_sqrt_d;
    }

    // 4. Find absolute maximum via SIMD reduction.
    float local_absmax = 0.0f;
    for (ushort i = 0; i < EPT; i++) {
        local_absmax = max(local_absmax, abs(elems[i]));
    }
    float absmax = simd_max(local_absmax);

    // 5. Symmetric affine quantize: map [-absmax, +absmax] to [0, 15].
    // index = round(7.5 * (1 + x/absmax)) = round(7.5 + 7.5 * x / absmax)
    float inv_absmax = (absmax > 1.0e-10f) ? (1.0f / absmax) : 0.0f;
    float scale_to_idx = 7.5f * inv_absmax;

    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float idx_f = fma(elems[i], scale_to_idx, 7.5f);
        // Clamp to [0, 15] and round.
        idx_f = clamp(idx_f, 0.0f, 15.0f);
        indices[i] = uint8_t(idx_f + 0.5f);
    }

    // 6. Pack nibbles and write.
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;

    const uint packed_row_stride = HEAD_DIM / 2;
    const uint packed_base = head_idx * params.cache_capacity * packed_row_stride
                           + actual_pos * packed_row_stride;

    const uint byte_base = packed_base + lane * (EPT / 2);
    for (ushort i = 0; i < EPT; i += 2) {
        uint8_t lo = indices[i] & 0xFu;
        uint8_t hi = (indices[i + 1] & 0xFu) << 4;
        packed[byte_base + i / 2] = lo | hi;
    }

    // 7. Store norm value that matches the decode formula:
    //    decode: value = norm * rsqrt(head_dim) * (nibble - 7.5)
    //    We need: norm * rsqrt(head_dim) = absmax / 7.5
    //    So: norm = absmax * sqrt(head_dim) / 7.5
    if (lane == 0) {
        uint norm_idx = head_idx * params.cache_capacity + actual_pos;
        norms[norm_idx] = absmax * sqrt(float(HEAD_DIM)) / 7.5f;
    }
}

template [[host_name("hadamard_quantize_kv_affine4_d256")]]
kernel void hadamard_quantize_kv_affine4<256>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeParams &, uint, uint);

template [[host_name("hadamard_quantize_kv_affine4_d512")]]
kernel void hadamard_quantize_kv_affine4<512>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeParams &, uint, uint);
