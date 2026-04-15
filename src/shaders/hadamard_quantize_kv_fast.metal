// hadamard_quantize_kv_fast.metal — FWHT + quantize using SIMD shuffle (zero threadgroup barriers)
//
// Replaces hadamard_quantize_kv.metal. Same algorithm, but the FWHT butterfly
// uses simd_shuffle_xor instead of shared memory + threadgroup barriers.
//
// Architecture: 1 simdgroup (32 threads) per KV head.
// Each thread holds head_dim/32 elements in registers.
// - head_dim=256: 8 elements/thread, 8 butterfly stages (3 local + 5 shuffle)
// - head_dim=512: 16 elements/thread, 9 butterfly stages (4 local + 5 shuffle)
//
// Reference: HadaCore (arxiv 2412.08832) SIMD butterfly pattern.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant float CODEBOOK_4BIT[16] = {
    -2.7325896f, -2.0690172f, -1.6180464f, -1.2562312f,
    -0.9423405f, -0.6567591f, -0.3880483f, -0.1283950f,
     0.1283950f,  0.3880483f,  0.6567591f,  0.9423405f,
     1.2562312f,  1.6180464f,  2.0690172f,  2.7325896f,
};

constant float BOUNDARIES_4BIT[15] = {
    -2.4008034f, -1.8435318f, -1.4371388f, -1.0992859f,
    -0.7995498f, -0.5224037f, -0.2582217f,  0.0000000f,
     0.2582217f,  0.5224037f,  0.7995498f,  1.0992859f,
     1.4371388f,  1.8435318f,  2.4008034f,
};

struct HadamardQuantizeParams {
    uint head_dim;
    uint num_kv_heads;
    uint write_pos;
    uint cache_capacity;
    uint is_sliding;
};

// Butterfly operation on a local element pair.
inline void butterfly_local(thread float &a, thread float &b) {
    float sum = a + b;
    float diff = a - b;
    a = sum;
    b = diff;
}

// FWHT using SIMD shuffle — zero threadgroup barriers.
// EPT = elements per thread (head_dim / 32).
// Each thread holds EPT consecutive elements from the head vector.
template<ushort EPT>
inline void fwht_simd(thread float *elems, uint lane) {
    // Stage 1: local butterfly stages (h < EPT)
    // h=1: pairs (0,1), (2,3), ...
    // h=2: pairs (0,2), (1,3), ...
    // ... up to h=EPT/2
    for (ushort h = 1; h < EPT; h <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            ushort partner = i ^ h;
            if (partner > i) {
                butterfly_local(elems[i], elems[partner]);
            }
        }
    }

    // Stage 2: cross-thread butterfly stages (h >= EPT)
    // h=EPT: exchange with thread lane^1
    // h=2*EPT: exchange with thread lane^2
    // h=4*EPT: exchange with thread lane^4
    // ... up to h=16*EPT (lane^16 for 32-thread simd)
    for (ushort delta = 1; delta < 32; delta <<= 1) {
        // Each element i in this thread exchanges with element i in thread (lane ^ delta).
        // The butterfly pair is (global_idx, global_idx ^ (delta * EPT)).
        // global_idx = lane * EPT + i, partner_global = (lane ^ delta) * EPT + i.
        for (ushort i = 0; i < EPT; i++) {
            float partner_val = simd_shuffle_xor(elems[i], delta);
            // Determine if this thread does (a+b) or (a-b).
            // The lower-indexed thread gets the sum, the higher gets the difference.
            if (lane & delta) {
                elems[i] = partner_val - elems[i];
            } else {
                elems[i] = elems[i] + partner_val;
            }
        }
    }
}

// Quantize one KV head's vector: load → FWHT → normalize → quantize → pack nibbles.
// 1 simdgroup per head. 32 threads. Each thread handles EPT = head_dim/32 elements.
template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_fast(
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

    // 1. Load EPT elements into registers.
    const uint src_base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = src[src_base + i];
    }

    // 2. FWHT via SIMD shuffle (ZERO threadgroup barriers).
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize by 1/sqrt(head_dim).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= inv_sqrt_d;
    }

    // 4. Compute L2 norm via SIMD reduction (ZERO threadgroup barriers).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) {
        local_sq_sum += elems[i] * elems[i];
    }
    float norm = sqrt(simd_sum(local_sq_sum));

    // 5. Normalize to unit sphere and scale to N(0,1).
    float inv_norm = (norm > 1.0e-10f) ? (1.0f / norm) : 0.0f;
    float scale = inv_norm * sqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= scale;
    }

    // 6. Quantize each element: find nearest Lloyd-Max centroid.
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx = 0;
        // Unrolled binary search (4 comparisons for 16 centroids).
        idx = (v > BOUNDARIES_4BIT[7]) ? 8 : 0;
        idx += (v > BOUNDARIES_4BIT[idx + 3]) ? 4 : 0;
        idx += (v > BOUNDARIES_4BIT[idx + 1]) ? 2 : 0;
        idx += (v > BOUNDARIES_4BIT[idx]) ? 1 : 0;
        indices[i] = idx;
    }

    // 7. Pack nibbles and write.
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;

    const uint packed_row_stride = HEAD_DIM / 2;
    const uint packed_base = head_idx * params.cache_capacity * packed_row_stride
                           + actual_pos * packed_row_stride;

    // Each thread writes EPT/2 bytes (EPT elements → EPT/2 nibble pairs).
    const uint byte_base = packed_base + lane * (EPT / 2);
    for (ushort i = 0; i < EPT; i += 2) {
        uint8_t lo = indices[i] & 0xFu;
        uint8_t hi = (indices[i + 1] & 0xFu) << 4;
        packed[byte_base + i / 2] = lo | hi;
    }

    // 8. Store norm (lane 0 only).
    if (lane == 0) {
        uint norm_idx = head_idx * params.cache_capacity + actual_pos;
        norms[norm_idx] = norm;
    }
}

// Instantiations for Gemma 4 head dimensions.
template [[host_name("hadamard_quantize_kv_fast_d256")]]
kernel void hadamard_quantize_kv_fast<256>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeParams &, uint, uint);

template [[host_name("hadamard_quantize_kv_fast_d512")]]
kernel void hadamard_quantize_kv_fast<512>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeParams &, uint, uint);
