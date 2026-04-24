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
    constant HadamardQuantizeParams &, device float *, uint, uint);

// ============================================================================
// Track B (iter-21): higher-bit codebooks for ablation.
// 5-bit (32 centroids) and 6-bit (64 centroids) Lloyd-Max codebooks for N(0,1).
// Byte-packed: 1 byte per element (upper 3 or 2 bits zeroed).
// Packed buffer layout: [num_kv_heads, capacity, head_dim] u8 (one byte per index).
// ============================================================================

constant float CODEBOOK_5BIT[32] = {
    -3.2606790f, -2.6910589f, -2.3176743f, -2.0286608f,
    -1.7871646f, -1.5761599f, -1.3862739f, -1.2117410f,
    -1.0487242f, -0.8945114f, -0.7470884f, -0.6048936f,
    -0.4666676f, -0.3313550f, -0.1980377f, -0.0658849f,
     0.0658849f,  0.1980377f,  0.3313550f,  0.4666676f,
     0.6048936f,  0.7470884f,  0.8945114f,  1.0487242f,
     1.2117410f,  1.3862739f,  1.5761599f,  1.7871646f,
     2.0286608f,  2.3176743f,  2.6910589f,  3.2606790f,
};

constant float BOUNDARIES_5BIT[31] = {
    -2.9758689f, -2.5043666f, -2.1731675f, -1.9079127f,
    -1.6816622f, -1.4812169f, -1.2990074f, -1.1302326f,
    -0.9716178f, -0.8207999f, -0.6759910f, -0.5357806f,
    -0.3990113f, -0.2646964f, -0.1319613f,  0.0000000f,
     0.1319613f,  0.2646964f,  0.3990113f,  0.5357806f,
     0.6759910f,  0.8207999f,  0.9716178f,  1.1302326f,
     1.2990074f,  1.4812169f,  1.6816622f,  1.9079127f,
     2.1731675f,  2.5043666f,  2.9758689f,
};

constant float CODEBOOK_6BIT[64] = {
    -3.6996161f, -3.1907215f, -2.8640626f, -2.6161277f,
    -2.4129324f, -2.2388464f, -2.0853192f, -1.9471373f,
    -1.8208742f, -1.7041502f, -1.5952401f, -1.4928497f,
    -1.3959804f, -1.3038428f, -1.2157998f, -1.1313277f,
    -1.0499889f, -0.9714118f, -0.8952766f, -0.8213046f,
    -0.7492492f, -0.6788902f, -0.6100285f, -0.5424819f,
    -0.4760822f, -0.4106724f, -0.3461048f, -0.2822386f,
    -0.2189392f, -0.1560761f, -0.0935225f, -0.0311537f,
     0.0311537f,  0.0935225f,  0.1560761f,  0.2189392f,
     0.2822386f,  0.3461048f,  0.4106724f,  0.4760822f,
     0.5424819f,  0.6100285f,  0.6788902f,  0.7492492f,
     0.8213046f,  0.8952766f,  0.9714118f,  1.0499889f,
     1.1313277f,  1.2157998f,  1.3038428f,  1.3959804f,
     1.4928497f,  1.5952401f,  1.7041502f,  1.8208742f,
     1.9471373f,  2.0853192f,  2.2388464f,  2.4129324f,
     2.6161277f,  2.8640626f,  3.1907215f,  3.6996161f,
};

constant float BOUNDARIES_6BIT[63] = {
    -3.4451688f, -3.0273920f, -2.7400952f, -2.5145300f,
    -2.3258894f, -2.1620828f, -2.0162282f, -1.8840057f,
    -1.7625122f, -1.6496952f, -1.5440449f, -1.4444151f,
    -1.3499116f, -1.2598213f, -1.1735638f, -1.0906583f,
    -1.0107003f, -0.9333442f, -0.8582906f, -0.7852769f,
    -0.7140697f, -0.6444593f, -0.5762552f, -0.5092820f,
    -0.4433773f, -0.3783886f, -0.3141717f, -0.2505889f,
    -0.1875076f, -0.1247993f, -0.0623381f,  0.0000000f,
     0.0623381f,  0.1247993f,  0.1875076f,  0.2505889f,
     0.3141717f,  0.3783886f,  0.4433773f,  0.5092820f,
     0.5762552f,  0.6444593f,  0.7140697f,  0.7852769f,
     0.8582906f,  0.9333442f,  1.0107003f,  1.0906583f,
     1.1735638f,  1.2598213f,  1.3499116f,  1.4444151f,
     1.5440449f,  1.6496952f,  1.7625122f,  1.8840057f,
     2.0162282f,  2.1620828f,  2.3258894f,  2.5145300f,
     2.7400952f,  3.0273920f,  3.4451688f,
};

struct HadamardQuantizeHbParams {
    uint head_dim;
    uint num_kv_heads;
    uint write_pos;
    uint cache_capacity;
    uint is_sliding;
    float scale_factor_d512; // Same semantics as 4-bit path
    uint codebook_bits;      // 5 or 6
};

// Higher-bit quantization kernel: same FWHT + norm as 4-bit, but quantizes to
// 5-bit (32 centroids) or 6-bit (64 centroids) and writes 1 byte per element.
// Packed buffer: [num_kv_heads, capacity, head_dim] u8 (byte-packed).
template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_hb(
    device const float                    *src    [[buffer(0)]],
    device       uint8_t                  *packed [[buffer(1)]],  // byte-packed (1 byte/elem)
    device       float                    *norms  [[buffer(2)]],
    constant HadamardQuantizeHbParams     &params [[buffer(3)]],
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
    for (ushort i = 0; i < EPT; i++) elems[i] = src[src_base + i];

    // 1b. D1 sign pre-multiplication (SRHT).
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = (HEAD_DIM == 256) ? TBQ_SIGNS_256[j >> 3] : TBQ_SIGNS_512[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    // 2. FWHT.
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize 1/sqrt(d).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) elems[i] *= inv_sqrt_d;

    // 4. Compute norm(s).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) local_sq_sum += elems[i] * elems[i];

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        norm0 = sqrt(simd_sum(local_sq_sum));
        norm1 = 0.0f;
    } else {
        float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
        float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
        blk0_sq = simd_broadcast(blk0_sq, 0u);
        blk1_sq = simd_broadcast(blk1_sq, 16u);
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 5. Scale elements to N(0,1) range for quantization.
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    } else {
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * params.scale_factor_d512;
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    }

    // 6. Quantize with higher-bit codebook.
    const bool use_5bit = (params.codebook_bits == 5u);
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx;
        if (use_5bit) {
            // 5-bit: 32 centroids, binary search with 5 levels
            idx = (v > BOUNDARIES_5BIT[15]) ? 16 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_5BIT[idx]) ? 1 : 0;
        } else {
            // 6-bit: 64 centroids, binary search with 6 levels
            idx = (v > BOUNDARIES_6BIT[31]) ? 32 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_6BIT[idx]) ? 1 : 0;
        }
        indices[i] = idx;
    }

    // 7. Write byte-packed output (1 byte per element).
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;
    // Packed layout: [head_idx, actual_pos, 0..HEAD_DIM] u8 — byte-packed.
    const uint packed_base = head_idx * params.cache_capacity * HEAD_DIM
                           + actual_pos * HEAD_DIM;
    const uint elem_base = packed_base + lane * EPT;
    for (ushort i = 0; i < EPT; i++) {
        packed[elem_base + i] = indices[i];
    }

    // 8. Store norm(s) — same as 4-bit path.
    if (HEAD_DIM == 256) {
        if (lane == 0) {
            norms[head_idx * params.cache_capacity + actual_pos] = norm0;
        }
    } else {
        if (lane == 0u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
    }
}

template [[host_name("hadamard_quantize_kv_hb_d256")]]
kernel void hadamard_quantize_kv_hb<256>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint, uint);

template [[host_name("hadamard_quantize_kv_hb_d512")]]
kernel void hadamard_quantize_kv_hb<512>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint, uint);
