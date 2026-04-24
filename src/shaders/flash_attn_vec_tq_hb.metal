// flash_attn_vec_tq_hb.metal — Native TQ SDPA for 5/6/8-bit byte-packed KV cache.
//
// Variant of flash_attn_vec_tq.metal that reads K/V from byte-packed (1 byte/element)
// higher-bit codebook indices instead of nibble-packed 4-bit indices.
//
// Bit-width is selected at compile time via template parameter CODEBOOK_BITS:
//   5  → 32 centroids  (Lloyd-Max N(0,1) optimal)
//   6  → 64 centroids
//   8  → 256 centroids
//
// Packed buffer layout: [num_kv_heads, capacity, head_dim] u8 (byte-packed)
//   One byte per element. For 5-bit only 5 LSBs are used (upper 3 zero).
//
// Dequant formula (same as tq_dequantize_hb_kv, which must match exactly):
//   D=256: scale_norm = norm * inv_sqrt(256)
//   D=512: scale_norm = norm / scale_factor_d512
//
// ADR-007 iter-24: measure Gate A/B/C at 5/6/8-bit to find shippable bit-width.

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32   // KV positions per simdgroup iteration
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

// Parameters — same layout as FlashAttnVecTqParams in flash_attn_vec_tq.metal.
struct FlashAttnVecTqHbParams {
    uint  n_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  kv_seq_len;
    uint  kv_capacity;
    float scale;
    uint  mask_type;
    uint  sliding_window;
    float softcap;
    uint  nwg;
    uint  ring_start;
    float scale_factor_d512;  // for D=512 norm dequant
    uint  codebook_bits;      // 5, 6, or 8 (runtime selector)
};

// Reduce params — shared with flash_attn_vec.
struct FlashAttnVecReduceParamsHb {
    uint nrows;
};

// ---------------------------------------------------------------------------
// 5-bit codebook (32 centroids, byte-packed — same as hadamard_quantize_kv_fast.metal)
// ---------------------------------------------------------------------------
constant float CODEBOOK_HB_5BIT[32] = {
    -3.2606790f, -2.6910589f, -2.3176743f, -2.0286608f,
    -1.7871646f, -1.5761599f, -1.3862739f, -1.2117410f,
    -1.0487242f, -0.8945114f, -0.7470884f, -0.6048936f,
    -0.4666676f, -0.3313550f, -0.1980377f, -0.0658849f,
     0.0658849f,  0.1980377f,  0.3313550f,  0.4666676f,
     0.6048936f,  0.7470884f,  0.8945114f,  1.0487242f,
     1.2117410f,  1.3862739f,  1.5761599f,  1.7871646f,
     2.0286608f,  2.3176743f,  2.6910589f,  3.2606790f,
};

// ---------------------------------------------------------------------------
// 6-bit codebook (64 centroids)
// ---------------------------------------------------------------------------
constant float CODEBOOK_HB_6BIT[64] = {
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

// ---------------------------------------------------------------------------
// 8-bit codebook (256 centroids, Lloyd-Max N(0,1), iter-24)
// ---------------------------------------------------------------------------
constant float CODEBOOK_HB_8BIT[256] = {
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

// ---------------------------------------------------------------------------
// Inline dequant: look up byte index in the selected codebook, scale by norm.
// CODEBOOK_BITS is a runtime value from params (not compile-time template),
// so we use if-else. The Metal compiler will constant-fold if the value is
// known constant per-dispatch via a push-constant variant, but runtime is fine
// for correctness.
//
// packed_base: pointer to start of this position's byte-packed data [head_dim bytes]
// coord:       coordinate index (0..head_dim-1)
// scale_norm:  pre-multiplied scale (norm * inv_sqrt_dk for D=256, norm/sf for D=512)
// cbits:       codebook_bits field from params (5, 6, or 8)
// ---------------------------------------------------------------------------
inline float dequant_hb_single(
    device const uint8_t *packed_pos,
    uint coord,
    float scale_norm,
    uint cbits
) {
    uint idx = (uint)packed_pos[coord];
    float centroid;
    if (cbits == 5u) {
        centroid = CODEBOOK_HB_5BIT[idx & 0x1Fu];
    } else if (cbits == 6u) {
        centroid = CODEBOOK_HB_6BIT[idx & 0x3Fu];
    } else {
        centroid = CODEBOOK_HB_8BIT[idx];  // 8-bit: full byte
    }
    return centroid * scale_norm;
}

// Reconstruct float4 from 4 consecutive byte-packed elements.
// coord_base must be a multiple of 4.
inline float4 dequant_hb_float4(
    device const uint8_t *packed_pos,
    uint coord_base,
    float scale_norm,
    uint cbits
) {
    return float4(
        dequant_hb_single(packed_pos, coord_base + 0, scale_norm, cbits),
        dequant_hb_single(packed_pos, coord_base + 1, scale_norm, cbits),
        dequant_hb_single(packed_pos, coord_base + 2, scale_norm, cbits),
        dequant_hb_single(packed_pos, coord_base + 3, scale_norm, cbits)
    );
}

// ---------------------------------------------------------------------------
// Main kernel: native HB (higher-bit) TQ flash attention vector.
//
// Same structure as flash_attn_vec_tq_impl but reads from byte-packed K/V.
// 5/6/8-bit controlled by params.codebook_bits at runtime.
//
// Norms layout:
//   D=256: [num_kv_heads, capacity]    f32 — 1 norm per position
//   D=512: [num_kv_heads, capacity, 2] f32 — 2 per-block norms per position
// ---------------------------------------------------------------------------
template<short DK, short DV>
kernel void flash_attn_vec_tq_hb_impl(
    constant FlashAttnVecTqHbParams  &params      [[buffer(0)]],
    device const float               *Q           [[buffer(1)]],
    device const uint8_t             *K_packed    [[buffer(2)]],  // byte-packed
    device const float               *K_norms     [[buffer(3)]],
    device const uint8_t             *V_packed    [[buffer(4)]],  // byte-packed
    device const float               *V_norms     [[buffer(5)]],
    device       float               *dst         [[buffer(6)]],
    threadgroup  half                *shmem       [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK / 4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV / 4;
    constexpr short SH  = 4 * C;  // 128 halfs = 64 floats

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");
    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    const uint NWG = params.nwg;
    const ushort iwg = tgpig[2] % NWG;
    const ushort iq2 = tgpig[1];  // head index
    const ushort iq1 = tgpig[0];  // query index (0 for decode)

    // GQA: map query head to KV head.
    const uint heads_per_kv = params.n_heads / params.n_kv_heads;
    const uint kv_head = iq2 / heads_per_kv;

    // Shared memory layout (same as flash_attn_vec_tq.metal):
    //   [0, PK)                     — Q as half4 (pre-rotated by caller)
    //   [PK, PK + SH)               — scratch for attention scores
    //   [PK + SH, PK + SH + 2*PV)   — output accumulator as float4
    threadgroup half4  *sq4 = (threadgroup half4  *)(shmem);
    threadgroup float  *ss  = (threadgroup float  *)(shmem + PK);
    threadgroup float4 *so4 = (threadgroup float4 *)(shmem + PK + SH);

    // Load pre-rotated Q into shared memory as half4.
    for (ushort i = tiisg; i < PK4; i += NW) {
        if (i < DK4) {
            float4 qval = *((device const float4 *)(Q + iq2 * DK + i * 4));
            sq4[i] = half4(qval);
        } else {
            sq4[i] = half4(0.0h);
        }
    }

    // Zero output accumulator.
    so4 += tiisg;
    for (short i = 0; i < DV4 / NL; ++i) {
        so4[i * NL] = float4(0.0f);
    }

    // Zero scratch buffer.
    for (ushort i = tiisg; i < SH / 4; i += NW) {
        ((threadgroup float *)(shmem + PK))[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state.
    float S = 0.0f;
    float M = -FLT_MAX / 2;

    const ushort tx = tiisg;
    const uint kv_seq_len = params.kv_seq_len;
    const uint kv_capacity = params.kv_capacity;
    const uint ring_start = params.ring_start;
    const uint cbits = params.codebook_bits;
    const float sf_d512 = params.scale_factor_d512;
    const bool is_d512 = (DK > 256);

    uint window_start_logical = 0;
    if (params.mask_type == 2 && params.sliding_window > 0 && kv_seq_len > params.sliding_window) {
        window_start_logical = kv_seq_len - params.sliding_window;
    }

    threadgroup const half4 *pq4 = sq4 + tx;

    // Main loop over KV cache in chunks of C=32.
    for (uint ic0 = iwg; ; ic0 += NWG) {
        uint ic = ic0 * C;
        if (ic >= kv_seq_len) break;

        // Compute mask for this chunk.
        {
            uint k_pos = ic + tx;
            float mask_val = 0.0f;
            if (k_pos >= kv_seq_len) {
                mask_val = -65504.0f;
            } else {
                uint logical_idx = (k_pos - ring_start + kv_capacity) % kv_capacity;
                if (logical_idx >= kv_seq_len || logical_idx < window_start_logical) {
                    mask_val = -65504.0f;
                }
            }
            ss[tx] = mask_val;
        }

        if (simd_max(ss[tiisg]) <= -65504.0f) continue;

        // ---- Q * K^T ----
        {
            float mqk[C];
            const float inv_sqrt_dk = rsqrt(float(DK));

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) {
                    mqk[cc] = 0.0f;
                    continue;
                }

                // Dequant scale for K.
                float k_sn;
                if (is_d512) {
                    // D=512: per-block norms; block 0 = coords 0..255, block 1 = 256..511
                    // For K*Q^T we need both blocks. The dot product spans all DK coords.
                    // We compute the block-0 portion and block-1 portion separately,
                    // each with their own scale_norm.
                    // norm_base points to: [kv_head, kv_pos, 0..2] f32
                    device const float *knorm = K_norms + (kv_head * kv_capacity + kv_pos) * 2u;
                    // k_sn unused in this branch — handled in the inner loop below
                    (void)k_sn;
                    (void)inv_sqrt_dk;

                    device const uint8_t *k_base =
                        K_packed + (kv_head * kv_capacity + kv_pos) * DK;

                    float partial = 0.0f;
                    // Block 0: coords 0..255
                    {
                        float sn0 = knorm[0] / sf_d512;
                        for (short ii = 0; ii < (DK/2) / 4 / NL; ++ii) {
                            uint coord = (uint)(tx * (DK/2/4) + ii) * 4;
                            if (coord + 3 < (uint)(DK/2)) {
                                float4 k_val = dequant_hb_float4(k_base, coord, sn0, cbits);
                                partial += dot(k_val, float4(pq4[ii * NL]));
                            }
                        }
                    }
                    // Block 1: coords 256..511
                    {
                        float sn1 = knorm[1] / sf_d512;
                        const uint blk1_start = DK / 2;
                        for (short ii = 0; ii < (DK/2) / 4 / NL; ++ii) {
                            uint coord = blk1_start + (uint)(tx * (DK/2/4) + ii) * 4;
                            if (coord + 3 < (uint)DK) {
                                float4 k_val = dequant_hb_float4(k_base, coord, sn1, cbits);
                                partial += dot(k_val, float4(pq4[(DK4/2/NL + ii) * NL]));
                            }
                        }
                    }
                    mqk[cc] = simd_sum(partial);
                } else {
                    // D=256: single norm per position.
                    float k_norm_val = K_norms[kv_head * kv_capacity + kv_pos];
                    k_sn = k_norm_val * inv_sqrt_dk;

                    device const uint8_t *k_base =
                        K_packed + (kv_head * kv_capacity + kv_pos) * DK + tx * 4u;

                    float partial = 0.0f;
                    for (short ii = 0; ii < DK4 / NL; ++ii) {
                        float4 k_val = dequant_hb_float4(k_base, (uint)(ii * NL) * 4u, k_sn, cbits);
                        partial += dot(k_val, float4(pq4[ii * NL]));
                    }
                    mqk[cc] = simd_sum(partial);
                }
            }

            ss[tx] = fma(mqk[tx], params.scale, ss[tx]);
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Online softmax ----
        {
            const float m_old = M;
            const float s_new = ss[tiisg];
            M = simd_max(max(M, s_new));
            const float ms = exp(m_old - M);
            const float vs = exp(s_new - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] *= ms;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O + softmax_weights * V ----
        {
            float4 lo[DV4 / NL];
            for (short ii = 0; ii < DV4 / NL; ++ii) lo[ii] = float4(0.0f);

            const float inv_sqrt_dv = rsqrt(float(DV));

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) continue;

                if (is_d512) {
                    device const float *vnorm = V_norms + (kv_head * kv_capacity + kv_pos) * 2u;
                    device const uint8_t *v_base =
                        V_packed + (kv_head * kv_capacity + kv_pos) * DV;
                    float w = ss[cc];

                    // Block 0: coords 0..255
                    float sn0 = vnorm[0] / sf_d512 * w;
                    for (short ii = 0; ii < (DV/2) / 4 / NL; ++ii) {
                        uint coord = (uint)(tx * (DV/2/4) + ii) * 4;
                        if (coord + 3 < (uint)(DV/2)) {
                            lo[ii] += dequant_hb_float4(v_base, coord, sn0, cbits);
                        }
                    }
                    // Block 1: coords 256..511
                    float sn1 = vnorm[1] / sf_d512 * w;
                    for (short ii = 0; ii < (DV/2) / 4 / NL; ++ii) {
                        uint coord = (uint)(DV/2) + (uint)(tx * (DV/2/4) + ii) * 4;
                        if (coord + 3 < (uint)DV) {
                            lo[DV4/2/NL + ii] += dequant_hb_float4(v_base, coord, sn1, cbits);
                        }
                    }
                } else {
                    float v_norm_val = V_norms[kv_head * kv_capacity + kv_pos];
                    float v_sw = v_norm_val * inv_sqrt_dv * ss[cc];
                    device const uint8_t *v_base =
                        V_packed + (kv_head * kv_capacity + kv_pos) * DV + tx * 4u;

                    for (short ii = 0; ii < DV4 / NL; ++ii) {
                        lo[ii] += dequant_hb_float4(v_base, (uint)(ii * NL) * 4u, v_sw, cbits);
                    }
                }
            }

            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] += lo[ii];
            }
        }
    }

    // Store M and S for the reduce kernel.
    if (tiisg == 0) {
        ss[0] = S;
        ss[1] = M;
    }

    so4 -= tiisg;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Write output ----
    if (sgitg == 0) {
        const int64_t nrows = params.n_heads;
        const int64_t rid = iq2 + (int64_t)iq1 * params.n_heads;
        const uint NWG_val = params.nwg;
        const float inv_S = (NWG_val == 1) ? ((S == 0.0f) ? 0.0f : 1.0f / S) : 1.0f;

        device float4 *dst4 = (device float4 *)dst;
        for (ushort i = tiisg; i < DV4; i += NW) {
            dst4[rid * DV4 * NWG_val + NWG_val * i + iwg] = so4[i] * inv_S;
        }

        if (NWG_val > 1 && tiisg == 0) {
            device float *dst1 = (device float *)dst + nrows * DV * NWG_val;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 0] = S;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 1] = M;
        }
    }
}

// --------------------------------------------------------------------------
// Kernel instantiations
// --------------------------------------------------------------------------

typedef decltype(flash_attn_vec_tq_hb_impl<256, 256>) flash_attn_vec_tq_hb_t;

template [[host_name("flash_attn_vec_tq_hb_dk256")]]
kernel flash_attn_vec_tq_hb_t flash_attn_vec_tq_hb_impl<256, 256>;

template [[host_name("flash_attn_vec_tq_hb_dk512")]]
kernel flash_attn_vec_tq_hb_t flash_attn_vec_tq_hb_impl<512, 512>;
