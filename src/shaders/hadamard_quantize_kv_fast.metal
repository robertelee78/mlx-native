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

// ============================================================================
// 8-bit Lloyd-Max codebook for N(0,1): 256 reconstruction levels.
// Computed via Lloyd-Max iteration (iter-24). Byte-packed: 1 byte per element.
// Memory: 256 * 4 = 1KB for centroids, same for boundaries.
// ============================================================================

constant float CODEBOOK_8BIT[256] = {
    -4.0354801f,  -3.5656248f,  -3.2681869f,  -3.0454753f,
    -2.8654909f,  -2.7135514f,  -2.5816441f,  -2.4648947f,
    -2.3601072f,  -2.2650656f,  -2.1781659f,  -2.0982056f,
    -2.0242566f,  -1.9555844f,  -1.8915951f,  -1.8317989f,
    -1.7757850f,  -1.7232032f,  -1.6737511f,  -1.6271638f,
    -1.5832071f,  -1.5416716f,  -1.5023685f,  -1.4651264f,
    -1.4297889f,  -1.3962121f,  -1.3642638f,  -1.3338215f,
    -1.3047719f,  -1.2770099f,  -1.2504379f,  -1.2249654f,
    -1.2005084f,  -1.1769889f,  -1.1543349f,  -1.1324796f,
    -1.1113613f,  -1.0909233f,  -1.0711132f,  -1.0518828f,
    -1.0331881f,  -1.0149885f,  -0.9972470f,  -0.9799298f,
    -0.9630061f,  -0.9464477f,  -0.9302290f,  -0.9143267f,
    -0.8987195f,  -0.8833881f,  -0.8683149f,  -0.8534840f,
    -0.8388808f,  -0.8244918f,  -0.8103051f,  -0.7963095f,
    -0.7824948f,  -0.7688517f,  -0.7553715f,  -0.7420464f,
    -0.7288690f,  -0.7158325f,  -0.7029306f,  -0.6901574f,
    -0.6775076f,  -0.6649758f,  -0.6525573f,  -0.6402476f,
    -0.6280424f,  -0.6159377f,  -0.6039295f,  -0.5920144f,
    -0.5801888f,  -0.5684495f,  -0.5567932f,  -0.5452171f,
    -0.5337183f,  -0.5222939f,  -0.5109414f,  -0.4996583f,
    -0.4884421f,  -0.4772904f,  -0.4662010f,  -0.4551717f,
    -0.4442004f,  -0.4332850f,  -0.4224237f,  -0.4116144f,
    -0.4008554f,  -0.3901447f,  -0.3794808f,  -0.3688618f,
    -0.3582862f,  -0.3477523f,  -0.3372586f,  -0.3268035f,
    -0.3163855f,  -0.3060032f,  -0.2956552f,  -0.2853401f,
    -0.2750566f,  -0.2648032f,  -0.2545787f,  -0.2443818f,
    -0.2342113f,  -0.2240659f,  -0.2139444f,  -0.2038456f,
    -0.1937683f,  -0.1837115f,  -0.1736739f,  -0.1636545f,
    -0.1536520f,  -0.1436655f,  -0.1336938f,  -0.1237359f,
    -0.1137907f,  -0.1038571f,  -0.0939341f,  -0.0840208f,
    -0.0741159f,  -0.0642186f,  -0.0543279f,  -0.0444426f,
    -0.0345618f,  -0.0246845f,  -0.0148097f,  -0.0049364f,
     0.0049364f,   0.0148097f,   0.0246845f,   0.0345618f,
     0.0444426f,   0.0543279f,   0.0642186f,   0.0741159f,
     0.0840208f,   0.0939341f,   0.1038571f,   0.1137907f,
     0.1237359f,   0.1336938f,   0.1436655f,   0.1536520f,
     0.1636545f,   0.1736739f,   0.1837115f,   0.1937683f,
     0.2038456f,   0.2139444f,   0.2240659f,   0.2342113f,
     0.2443818f,   0.2545787f,   0.2648032f,   0.2750566f,
     0.2853401f,   0.2956552f,   0.3060032f,   0.3163855f,
     0.3268035f,   0.3372586f,   0.3477523f,   0.3582862f,
     0.3688618f,   0.3794808f,   0.3901447f,   0.4008554f,
     0.4116144f,   0.4224237f,   0.4332850f,   0.4442004f,
     0.4551717f,   0.4662010f,   0.4772904f,   0.4884421f,
     0.4996583f,   0.5109414f,   0.5222939f,   0.5337183f,
     0.5452171f,   0.5567932f,   0.5684495f,   0.5801888f,
     0.5920144f,   0.6039295f,   0.6159377f,   0.6280424f,
     0.6402476f,   0.6525573f,   0.6649758f,   0.6775076f,
     0.6901574f,   0.7029306f,   0.7158325f,   0.7288690f,
     0.7420464f,   0.7553715f,   0.7688517f,   0.7824948f,
     0.7963095f,   0.8103051f,   0.8244918f,   0.8388808f,
     0.8534840f,   0.8683149f,   0.8833881f,   0.8987195f,
     0.9143267f,   0.9302290f,   0.9464477f,   0.9630061f,
     0.9799298f,   0.9972470f,   1.0149885f,   1.0331881f,
     1.0518828f,   1.0711132f,   1.0909233f,   1.1113613f,
     1.1324796f,   1.1543349f,   1.1769889f,   1.2005084f,
     1.2249654f,   1.2504379f,   1.2770099f,   1.3047719f,
     1.3338215f,   1.3642638f,   1.3962121f,   1.4297889f,
     1.4651264f,   1.5023685f,   1.5416716f,   1.5832071f,
     1.6271638f,   1.6737511f,   1.7232032f,   1.7757850f,
     1.8317989f,   1.8915951f,   1.9555844f,   2.0242566f,
     2.0982056f,   2.1781659f,   2.2650656f,   2.3601072f,
     2.4648947f,   2.5816441f,   2.7135514f,   2.8654909f,
     3.0454753f,   3.2681869f,   3.5656248f,   4.0354801f,
};

constant float BOUNDARIES_8BIT[255] = {
    -3.8005524f,  -3.4169059f,  -3.1568311f,  -2.9554831f,
    -2.7895212f,  -2.6475977f,  -2.5232694f,  -2.4124010f,
    -2.3125864f,  -2.2216157f,  -2.1381857f,  -2.0612311f,
    -1.9899205f,  -1.9235897f,  -1.8616970f,  -1.8037919f,
    -1.7494941f,  -1.6984772f,  -1.6504574f,  -1.6051855f,
    -1.5624393f,  -1.5220201f,  -1.4837474f,  -1.4474577f,
    -1.4130005f,  -1.3802380f,  -1.3490426f,  -1.3192967f,
    -1.2908909f,  -1.2637239f,  -1.2377017f,  -1.2127369f,
    -1.1887487f,  -1.1656619f,  -1.1434072f,  -1.1219205f,
    -1.1011423f,  -1.0810183f,  -1.0615030f,  -1.0425530f,
    -1.0241353f,  -1.0062177f,  -0.9887684f,  -0.9717574f,
    -0.9551569f,  -0.9389384f,  -0.9230778f,  -0.9075531f,
    -0.8923438f,  -0.8774315f,  -0.8627994f,  -0.8483824f,
    -0.8341763f,  -0.8201665f,  -0.8063474f,  -0.7927022f,
    -0.7792233f,  -0.7659116f,  -0.7527560f,  -0.7397577f,
    -0.7269007f,  -0.7141815f,  -0.7015940f,  -0.6891325f,
    -0.6767917f,  -0.6645665f,  -0.6524524f,  -0.6404450f,
    -0.6285400f,  -0.6167336f,  -0.6050219f,  -0.5934019f,
    -0.5818691f,  -0.5704213f,  -0.5590564f,  -0.5477721f,
    -0.5365561f,  -0.5254177f,  -0.5143414f,  -0.5033399f,
    -0.4923663f,  -0.4814462f,  -0.4705957f,  -0.4598087f,
    -0.4490864f,  -0.4384327f,  -0.4278523f,  -0.4173191f,
    -0.4068480f,  -0.3964127f,  -0.3860278f,  -0.3756713f,
    -0.3653724f,  -0.3551054f,  -0.3448886f,  -0.3347310f,
    -0.3246244f,  -0.3145792f,  -0.3045792f,  -0.2946477f,
    -0.2847717f,  -0.2749409f,  -0.2651658f,  -0.2554303f,
    -0.2457480f,  -0.2361052f,  -0.2265154f,  -0.2169700f,
    -0.2074711f,  -0.1980201f,  -0.1886130f,  -0.1792522f,
    -0.1699320f,  -0.1606596f,  -0.1514298f,  -0.1422459f,
    -0.1331030f,  -0.1239975f,  -0.1149320f,  -0.1059030f,
    -0.0969144f,  -0.0879614f,  -0.0790463f,  -0.0701690f,
    -0.0613175f,  -0.0524850f,  -0.0436807f,  -0.0349011f,
    -0.0261497f,  -0.0174258f,  -0.0087231f,   0.0000000f,
     0.0087231f,   0.0174258f,   0.0261497f,   0.0349011f,
     0.0436807f,   0.0524850f,   0.0613175f,   0.0701690f,
     0.0790463f,   0.0879614f,   0.0969144f,   0.1059030f,
     0.1149320f,   0.1239975f,   0.1331030f,   0.1422459f,
     0.1514298f,   0.1606596f,   0.1699320f,   0.1792522f,
     0.1886130f,   0.1980201f,   0.2074711f,   0.2169700f,
     0.2265154f,   0.2361052f,   0.2457480f,   0.2554303f,
     0.2651658f,   0.2749409f,   0.2847717f,   0.2946477f,
     0.3045792f,   0.3145792f,   0.3246244f,   0.3347310f,
     0.3448886f,   0.3551054f,   0.3653724f,   0.3756713f,
     0.3860278f,   0.3964127f,   0.4068480f,   0.4173191f,
     0.4278523f,   0.4384327f,   0.4490864f,   0.4598087f,
     0.4705957f,   0.4814462f,   0.4923663f,   0.5033399f,
     0.5143414f,   0.5254177f,   0.5365561f,   0.5477721f,
     0.5590564f,   0.5704213f,   0.5818691f,   0.5934019f,
     0.6050219f,   0.6167336f,   0.6285400f,   0.6404450f,
     0.6524524f,   0.6645665f,   0.6767917f,   0.6891325f,
     0.7015940f,   0.7141815f,   0.7269007f,   0.7397577f,
     0.7527560f,   0.7659116f,   0.7792233f,   0.7927022f,
     0.8063474f,   0.8201665f,   0.8341763f,   0.8483824f,
     0.8627994f,   0.8774315f,   0.8923438f,   0.9075531f,
     0.9230778f,   0.9389384f,   0.9551569f,   0.9717574f,
     0.9887684f,   1.0062177f,   1.0241353f,   1.0425530f,
     1.0615030f,   1.0810183f,   1.1011423f,   1.1219205f,
     1.1434072f,   1.1656619f,   1.1887487f,   1.2127369f,
     1.2377017f,   1.2637239f,   1.2908909f,   1.3192967f,
     1.3490426f,   1.3802380f,   1.4130005f,   1.4474577f,
     1.4837474f,   1.5220201f,   1.5624393f,   1.6051855f,
     1.6504574f,   1.6984772f,   1.7494941f,   1.8037919f,
     1.8616970f,   1.9235897f,   1.9899205f,   2.0612311f,
     2.1381857f,   2.2216157f,   2.3125864f,   2.4124010f,
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
