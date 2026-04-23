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
//
// D1 random sign pre-multiplication (SRHT) per ADR-007 iter-13 shipping-impl study + iter-14 hypothesis test.
// Sign table verbatim from AmesianX TurboQuant at ggml-cuda/cpy-utils.cuh:158-163 (D=256) + :211-220 (D=512).
// Without D1, plain-WHT fails to decorrelate structured K/V (real Gemma activations, not random Gaussian).
// Sign is applied BEFORE WHT in encode + AFTER IWHT in decode (self-inverse since sign*sign=1).

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// D1 sign table for D=256 (32 bytes, 256 bits).
// Verbatim from AmesianX cpy-utils.cuh:158-163. sha256=3ef1038e6c232e9519101daa2d6efd637d4c6bfdb29f4ee7101625c39d0ddc89
// Bit j = (table[j>>3] >> (j&7)) & 1; bit=1 → sign=-1, bit=0 → sign=+1 (LSB-first).
constant uint8_t TBQ_SIGNS_256[32] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
};

// D1 sign table for D=512 (64 bytes, 512 bits).
// Verbatim from AmesianX cpy-utils.cuh:211-220. sha256=44f13ce9f6db1edac62f558ee054f9de29cd474fd051362cadcaa98a55745f17
// Same convention: bit j → table_512[j>>3] >> (j&7); bit=1 → sign=-1, bit=0 → sign=+1.
constant uint8_t TBQ_SIGNS_512[64] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
};

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
//
// D=256 path: single global L2 norm, stored at norms[head * capacity + pos].
//
// D=512 path (ADR-007 iter-15 per-256-block norm, per AmesianX cpy-utils.cuh:241-269):
//   After full 512-FWHT the vector is split into 2 contiguous 256-halves (block 0 = [0..255],
//   block 1 = [256..511]). Each half gets an independent RMS norm. The norms buffer is indexed
//   as norms[head * capacity * NORMS_PER_POS + pos * NORMS_PER_POS + blk] where
//   NORMS_PER_POS = 1 for D=256, NORMS_PER_POS = 2 for D=512.
//   Lane assignment: for EPT=16, lane 0..15 own elements 0..255 (block 0),
//                                lane 16..31 own elements 256..511 (block 1).
//   Cite: AmesianX cpy-utils.cuh:241-269 (queen-verified); ADR-007 iter-14 D1 SRHT + iter-15 per-block norm.
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

    // 1b. D1 sign pre-multiplication (SRHT) — applied BEFORE FWHT.
    // Select table by HEAD_DIM at compile time (constexpr branch).
    // Element global index j = lane * EPT + i.
    // sign_bit = (table[j>>3] >> (j&7)) & 1; sign = bit ? -1.0f : +1.0f.
    // Mirror of AmesianX cpy-utils.cuh:181 (D=256) / :229 (D=512).
    {
        for (ushort i = 0; i < EPT; i++) {
            ushort j = lane * EPT + i;  // global element index within head
            uint8_t sign_byte;
            if (HEAD_DIM == 256) {
                sign_byte = TBQ_SIGNS_256[j >> 3];
            } else {
                sign_byte = TBQ_SIGNS_512[j >> 3];
            }
            float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
            elems[i] *= sign_val;
        }
    }

    // 2. FWHT via SIMD shuffle (ZERO threadgroup barriers).
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize by 1/sqrt(head_dim) (normalized WHT convention).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= inv_sqrt_d;
    }

    // 4. Compute norm(s) via SIMD reduction (ZERO threadgroup barriers).
    //
    // D=256: single global L2 norm over all 256 elements.
    // D=512: 2 per-block RMS norms per AmesianX cpy-utils.cuh:241-269.
    //   Block 0 = elements [0..255], owned by lanes 0..15 (EPT=16 → lane*16+i in [0..255] iff lane<16).
    //   Block 1 = elements [256..511], owned by lanes 16..31.
    //   RMS norm: blk_norm[b] = sqrt(sum_sq[b] / 256.0f) where sum_sq[b] includes inv_sqrt_d factor.
    //   This matches AmesianX decode convention when decode uses: scale = blk_norm[b] * inv_sqrt(512).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) {
        local_sq_sum += elems[i] * elems[i];
    }

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        // Single global L2 norm (unchanged D=256 path).
        float norm = sqrt(simd_sum(local_sq_sum));
        norm0 = norm;
        norm1 = 0.0f;  // unused for D=256
    } else {
        // D=512: per-block RMS norms.
        // Lane 0..15 (block 0): contribute to blk0_sq; lanes 16..31 zero out.
        // Lane 16..31 (block 1): contribute to blk1_sq; lanes 0..15 zero out.
        float blk0_contribution = (lane < 16u) ? local_sq_sum : 0.0f;
        float blk1_contribution = (lane >= 16u) ? local_sq_sum : 0.0f;
        float blk0_sq = simd_sum(blk0_contribution);  // sum over all 32 lanes (blk1 contributes 0)
        float blk1_sq = simd_sum(blk1_contribution);  // sum over all 32 lanes (blk0 contributes 0)
        // RMS norm per block (256 elements each).
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 5. Normalize each element: scale to N(0,1) using per-block norm.
    //    D=256: scale = inv_norm0 * sqrt(256) (unchanged).
    //    D=512: element in block b gets scale = sqrt(512) / blk_norm[b].
    //    The sqrt(HEAD_DIM) factor cancels the inv_sqrt_d applied in step 3, so stored
    //    element = FWHT_unnorm(sign*x)[j] / blk_norm[b].
    //    Decode recovers: CODEBOOK[idx] * blk_norm[b] * inv_sqrt(HEAD_DIM) = FWHT_norm(sign*x)[j].
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) {
            elems[i] *= scale;
        }
    } else {
        // D=512: apply per-block scale. Lanes 0..15 use norm0, lanes 16..31 use norm1.
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) {
            elems[i] *= scale;
        }
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

    // 8. Store norm(s).
    //    D=256: 1 norm at norms[head * capacity + pos] (NORMS_PER_POS = 1).
    //    D=512: 2 norms at norms[head * capacity * 2 + pos * 2 + blk] (NORMS_PER_POS = 2).
    //    Per AmesianX cpy-utils.cuh:256 y[blk].d = __float2half(blk_norm).
    if (HEAD_DIM == 256) {
        if (lane == 0) {
            uint norm_idx = head_idx * params.cache_capacity + actual_pos;
            norms[norm_idx] = norm0;
        }
    } else {
        // D=512: lane 0 writes norm0 (block 0), lane 16 writes norm1 (block 1).
        if (lane == 0u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
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

// iter-24: 8-bit (256 centroids) Lloyd-Max codebook for N(0,1).
// Computed via Lloyd-Max iteration to convergence (tol=1e-12).
// Symmetry error: 3.41e-10.
constant float CODEBOOK_8BIT[256] = {
    -5.0652659f, -4.6836997f, -4.4467193f, -4.2715508f,
    -4.1311907f, -4.0132856f, -3.9111092f, -3.8205780f,
    -3.7390194f, -3.6645851f, -3.5959415f, -3.5320936f,
    -3.4722785f, -3.4158977f, -3.3624729f, -3.3116156f,
    -3.2630056f, -3.2163758f, -3.1715011f, -3.1281899f,
    -3.0862780f, -3.0456229f, -3.0061011f, -2.9676040f,
    -2.9300362f, -2.8933131f, -2.8573596f, -2.8221086f,
    -2.7874999f, -2.7534795f, -2.7199985f, -2.6870129f,
    -2.6544825f, -2.6223710f, -2.5906452f, -2.5592748f,
    -2.5282321f, -2.4974918f, -2.4670306f, -2.4368270f,
    -2.4068614f, -2.3771157f, -2.3475732f, -2.3182184f,
    -2.2890372f, -2.2600165f, -2.2311440f, -2.2024086f,
    -2.1737998f, -2.1453081f, -2.1169245f, -2.0886408f,
    -2.0604493f, -2.0323430f, -2.0043154f, -1.9763603f,
    -1.9484722f, -1.9206458f, -1.8928763f, -1.8651592f,
    -1.8374904f, -1.8098662f, -1.7822828f, -1.7547372f,
    -1.7272261f, -1.6997469f, -1.6722970f, -1.6448739f,
    -1.6174755f, -1.5900996f, -1.5627445f, -1.5354084f,
    -1.5080897f, -1.4807869f, -1.4534986f, -1.4262237f,
    -1.3989610f, -1.3717093f, -1.3444678f, -1.3172356f,
    -1.2900118f, -1.2627956f, -1.2355865f, -1.2083838f,
    -1.1811868f, -1.1539951f, -1.1268081f, -1.0996255f,
    -1.0724469f, -1.0452718f, -1.0180999f, -0.9909310f,
    -0.9637647f, -0.9366008f, -0.9094390f, -0.8822793f,
    -0.8551212f, -0.8279648f, -0.8008098f, -0.7736561f,
    -0.7465035f, -0.7193520f, -0.6922014f, -0.6650517f,
    -0.6379027f, -0.6107544f, -0.5836067f, -0.5564596f,
    -0.5293129f, -0.5021667f, -0.4750208f, -0.4478753f,
    -0.4207301f, -0.3935852f, -0.3664405f, -0.3392960f,
    -0.3121517f, -0.2850076f, -0.2578636f, -0.2307198f,
    -0.2035761f, -0.1764324f, -0.1492888f, -0.1221453f,
    -0.0950019f, -0.0678584f, -0.0407151f, -0.0135717f,
     0.0135717f,  0.0407151f,  0.0678584f,  0.0950019f,
     0.1221453f,  0.1492888f,  0.1764324f,  0.2035761f,
     0.2307198f,  0.2578636f,  0.2850076f,  0.3121517f,
     0.3392960f,  0.3664405f,  0.3935852f,  0.4207301f,
     0.4478753f,  0.4750208f,  0.5021667f,  0.5293129f,
     0.5564596f,  0.5836067f,  0.6107544f,  0.6379027f,
     0.6650517f,  0.6922014f,  0.7193520f,  0.7465035f,
     0.7736561f,  0.8008098f,  0.8279648f,  0.8551212f,
     0.8822793f,  0.9094390f,  0.9366008f,  0.9637647f,
     0.9909310f,  1.0180999f,  1.0452718f,  1.0724469f,
     1.0996255f,  1.1268081f,  1.1539951f,  1.1811868f,
     1.2083838f,  1.2355865f,  1.2627956f,  1.2900118f,
     1.3172356f,  1.3444678f,  1.3717093f,  1.3989610f,
     1.4262237f,  1.4534986f,  1.4807869f,  1.5080897f,
     1.5354084f,  1.5627445f,  1.5900996f,  1.6174755f,
     1.6448739f,  1.6722970f,  1.6997469f,  1.7272261f,
     1.7547372f,  1.7822828f,  1.8098662f,  1.8374904f,
     1.8651592f,  1.8928763f,  1.9206458f,  1.9484722f,
     1.9763603f,  2.0043154f,  2.0323430f,  2.0604493f,
     2.0886408f,  2.1169245f,  2.1453081f,  2.1737998f,
     2.2024086f,  2.2311440f,  2.2600165f,  2.2890372f,
     2.3182184f,  2.3475732f,  2.3771157f,  2.4068614f,
     2.4368270f,  2.4670306f,  2.4974918f,  2.5282321f,
     2.5592748f,  2.5906452f,  2.6223710f,  2.6544825f,
     2.6870129f,  2.7199985f,  2.7534795f,  2.7874999f,
     2.8221086f,  2.8573596f,  2.8933131f,  2.9300362f,
     2.9676040f,  3.0061011f,  3.0456229f,  3.0862780f,
     3.1281899f,  3.1715011f,  3.2163758f,  3.2630056f,
     3.3116156f,  3.3624729f,  3.4158977f,  3.4722785f,
     3.5320936f,  3.5959415f,  3.6645851f,  3.7390194f,
     3.8205780f,  3.9111092f,  4.0132856f,  4.1311907f,
     4.2715508f,  4.4467193f,  4.6836997f,  5.0652659f,
};

// iter-24: 8-bit boundaries (255 boundaries for 256 centroids).
// BOUNDARIES_8BIT[i] = midpoint(CODEBOOK_8BIT[i], CODEBOOK_8BIT[i+1]).
constant float BOUNDARIES_8BIT[255] = {
    -4.8744828f, -4.5652095f, -4.3591350f, -4.2013707f,
    -4.0722382f, -3.9621974f, -3.8658436f, -3.7797987f,
    -3.7018022f, -3.6302633f, -3.5640175f, -3.5021860f,
    -3.4440881f, -3.3891853f, -3.3370443f, -3.2873106f,
    -3.2396907f, -3.1939384f, -3.1498455f, -3.1072339f,
    -3.0659504f, -3.0258620f, -2.9868525f, -2.9488201f,
    -2.9116746f, -2.8753363f, -2.8397341f, -2.8048042f,
    -2.7704897f, -2.7367390f, -2.7035057f, -2.6707477f,
    -2.6384267f, -2.6065081f, -2.5749600f, -2.5437535f,
    -2.5128620f, -2.4822612f, -2.4519288f, -2.4218442f,
    -2.3919885f, -2.3623444f, -2.3328958f, -2.3036278f,
    -2.2745269f, -2.2455802f, -2.2167763f, -2.1881042f,
    -2.1595539f, -2.1311163f, -2.1027826f, -2.0745450f,
    -2.0463962f, -2.0183292f, -1.9903379f, -1.9624162f,
    -1.9345590f, -1.9067610f, -1.8790177f, -1.8513248f,
    -1.8236783f, -1.7960745f, -1.7685100f, -1.7409817f,
    -1.7134865f, -1.6860220f, -1.6585855f, -1.6311747f,
    -1.6037875f, -1.5764221f, -1.5490764f, -1.5217490f,
    -1.4944383f, -1.4671427f, -1.4398612f, -1.4125923f,
    -1.3853351f, -1.3580886f, -1.3308517f, -1.3036237f,
    -1.2764037f, -1.2491911f, -1.2219851f, -1.1947853f,
    -1.1675909f, -1.1404016f, -1.1132168f, -1.0860362f,
    -1.0588593f, -1.0316859f, -1.0045155f, -0.9773478f,
    -0.9501827f, -0.9230199f, -0.8958592f, -0.8687003f,
    -0.8415430f, -0.8143873f, -0.7872329f, -0.7600798f,
    -0.7329278f, -0.7057767f, -0.6786266f, -0.6514772f,
    -0.6243286f, -0.5971806f, -0.5700331f, -0.5428862f,
    -0.5157398f, -0.4885937f, -0.4614481f, -0.4343027f,
    -0.4071577f, -0.3800128f, -0.3528683f, -0.3257239f,
    -0.2985797f, -0.2714356f, -0.2442917f, -0.2171479f,
    -0.1900042f, -0.1628606f, -0.1357171f, -0.1085736f,
    -0.0814302f, -0.0542868f, -0.0271434f,  0.0000000f,
     0.0271434f,  0.0542868f,  0.0814302f,  0.1085736f,
     0.1357171f,  0.1628606f,  0.1900042f,  0.2171479f,
     0.2442917f,  0.2714356f,  0.2985797f,  0.3257239f,
     0.3528683f,  0.3800128f,  0.4071577f,  0.4343027f,
     0.4614481f,  0.4885937f,  0.5157398f,  0.5428862f,
     0.5700331f,  0.5971806f,  0.6243286f,  0.6514772f,
     0.6786266f,  0.7057767f,  0.7329278f,  0.7600798f,
     0.7872329f,  0.8143873f,  0.8415430f,  0.8687003f,
     0.8958592f,  0.9230199f,  0.9501827f,  0.9773478f,
     1.0045155f,  1.0316859f,  1.0588593f,  1.0860362f,
     1.1132168f,  1.1404016f,  1.1675909f,  1.1947853f,
     1.2219851f,  1.2491911f,  1.2764037f,  1.3036237f,
     1.3308517f,  1.3580886f,  1.3853351f,  1.4125923f,
     1.4398612f,  1.4671427f,  1.4944383f,  1.5217490f,
     1.5490764f,  1.5764221f,  1.6037875f,  1.6311747f,
     1.6585855f,  1.6860220f,  1.7134865f,  1.7409817f,
     1.7685100f,  1.7960745f,  1.8236783f,  1.8513248f,
     1.8790177f,  1.9067610f,  1.9345590f,  1.9624162f,
     1.9903379f,  2.0183292f,  2.0463962f,  2.0745450f,
     2.1027826f,  2.1311163f,  2.1595539f,  2.1881042f,
     2.2167763f,  2.2455802f,  2.2745269f,  2.3036278f,
     2.3328958f,  2.3623444f,  2.3919885f,  2.4218442f,
     2.4519288f,  2.4822612f,  2.5128620f,  2.5437535f,
     2.5749600f,  2.6065081f,  2.6384267f,  2.6707477f,
     2.7035057f,  2.7367390f,  2.7704897f,  2.8048042f,
     2.8397341f,  2.8753363f,  2.9116746f,  2.9488201f,
     2.9868525f,  3.0258620f,  3.0659504f,  3.1072339f,
     3.1498455f,  3.1939384f,  3.2396907f,  3.2873106f,
     3.3370443f,  3.3891853f,  3.4440881f,  3.5021860f,
     3.5640175f,  3.6302633f,  3.7018022f,  3.7797987f,
     3.8658436f,  3.9621974f,  4.0722382f,  4.2013707f,
     4.3591350f,  4.5652095f,  4.8744828f,
};

struct HadamardQuantizeHbParams {
    uint head_dim;
    uint num_kv_heads;
    uint write_pos;
    uint cache_capacity;
    uint is_sliding;
    float scale_factor_d512; // Same semantics as 4-bit path
    uint codebook_bits;      // 5, 6, or 8
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

    // 6. Quantize with higher-bit codebook (5, 6, or 8-bit).
    const uint cbits = params.codebook_bits;
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx;
        if (cbits == 5u) {
            // 5-bit: 32 centroids, binary search with 5 levels
            idx = (v > BOUNDARIES_5BIT[15]) ? 16 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_5BIT[idx]) ? 1 : 0;
        } else if (cbits == 6u) {
            // 6-bit: 64 centroids, binary search with 6 levels
            idx = (v > BOUNDARIES_6BIT[31]) ? 32 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_6BIT[idx]) ? 1 : 0;
        } else {
            // 8-bit: 256 centroids, binary search with 8 levels
            idx = (v > BOUNDARIES_8BIT[127]) ? 128 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 63]) ? 64 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 31]) ? 32 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 7])  ?  8 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 3])  ?  4 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 1])  ?  2 : 0;
            idx += (v > BOUNDARIES_8BIT[idx])      ?  1 : 0;
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
