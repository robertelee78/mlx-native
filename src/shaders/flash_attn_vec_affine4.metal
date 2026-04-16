// Flash attention vector kernel with affine INT4 decode (speed test variant).
//
// Identical structure to flash_attn_vec_tq.metal, but replaces the Lloyd-Max
// codebook lookup with symmetric affine decode:
//
//   TQ:     value = CODEBOOK_4BIT[nibble] * scale_norm
//   Affine: value = scale_norm * (float(nibble) - 7.5f)
//
// This tests whether the constant-memory indexed lookup is the bottleneck.
// Same buffer layout: packed nibbles + per-position scale float.
// Quality will differ (affine assumes uniform spacing), but speed is the question.

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

struct FlashAttnVecTqParams {
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
};

struct FlashAttnVecReduceParams {
    uint nrows;
};

// ---------------------------------------------------------------------------
// Affine INT4 decode: value = scale_norm * (float(nibble) - 7.5)
//
// Compared to TQ codebook decode:
//   TQ:     4 indexed constant-memory lookups + 4 multiplies
//   Affine: 4 int-to-float casts + 4 FMAs (or 4 sub + 4 mul)
//
// The 7.5 center maps unsigned [0,15] to symmetric [-7.5, +7.5].
// For speed testing, the scale_norm absorbs the same norm * rsqrt(head_dim).
// ---------------------------------------------------------------------------
inline float4 dequant_affine4_float4(
    device const uint8_t *packed_base,
    uint byte_offset,
    float scale_norm
) {
    ushort packed = *((device const ushort *)(packed_base + byte_offset));

    uint idx0 = packed & 0xFu;
    uint idx1 = (packed >> 4u) & 0xFu;
    uint idx2 = (packed >> 8u) & 0xFu;
    uint idx3 = (packed >> 12u) & 0xFu;

    // Affine decode: scale * (nibble - center)
    // This is 1 subtract + 1 multiply per element, or 1 FMA.
    // No constant-memory indexed lookup.
    const float center = 7.5f;
    return float4(
        scale_norm * (float(idx0) - center),
        scale_norm * (float(idx1) - center),
        scale_norm * (float(idx2) - center),
        scale_norm * (float(idx3) - center)
    );
}

// ---------------------------------------------------------------------------
// Main kernel — identical to flash_attn_vec_tq_impl, only dequant changed.
// ---------------------------------------------------------------------------
template<short DK, short DV>
kernel void flash_attn_vec_affine4_impl(
    constant FlashAttnVecTqParams   &params      [[buffer(0)]],
    device const float              *Q           [[buffer(1)]],
    device const uint8_t            *K_packed    [[buffer(2)]],
    device const float              *K_norms     [[buffer(3)]],
    device const uint8_t            *V_packed    [[buffer(4)]],
    device const float              *V_norms     [[buffer(5)]],
    device       float              *dst         [[buffer(6)]],
    threadgroup  half               *shmem       [[threadgroup(0)]],
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
    constexpr short SH  = 4 * C;

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");
    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    const uint NWG = params.nwg;
    const ushort iwg = tgpig[2] % NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    const uint heads_per_kv = params.n_heads / params.n_kv_heads;
    const uint kv_head = iq2 / heads_per_kv;

    threadgroup half4  *sq4 = (threadgroup half4  *)(shmem);
    threadgroup float  *ss  = (threadgroup float  *)(shmem + PK);
    threadgroup float4 *so4 = (threadgroup float4 *)(shmem + PK + SH);

    {
        for (ushort i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                float4 qval = *((device const float4 *)(Q + iq2 * DK + i * 4));
                sq4[i] = half4(qval);
            } else {
                sq4[i] = half4(0.0h);
            }
        }
    }

    so4 += tiisg;
    for (short i = 0; i < DV4 / NL; ++i) {
        so4[i * NL] = float4(0.0f);
    }

    for (ushort i = tiisg; i < SH / 4; i += NW) {
        ((threadgroup float *)(shmem + PK))[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    const ushort tx = tiisg;

    const uint kv_seq_len = params.kv_seq_len;
    const uint abs_pos = kv_seq_len - 1;
    const uint causal_max_k = min(abs_pos + 1, kv_seq_len);

    uint window_start = 0;
    if (params.mask_type == 2 && params.sliding_window > 0) {
        window_start = (abs_pos >= params.sliding_window)
            ? (abs_pos - params.sliding_window + 1) : 0;
    }

    threadgroup const half4 *pq4 = sq4 + tx;

    for (uint ic0 = iwg; ; ic0 += NWG) {
        uint ic = ic0 * C;
        if (ic >= causal_max_k) break;

        {
            uint k_pos = ic + tx;
            float mask_val = 0.0f;
            if (k_pos >= causal_max_k || k_pos < window_start) {
                mask_val = -65504.0f;
            }
            ss[tx] = mask_val;
        }

        if (simd_max(ss[tiisg]) <= -65504.0f) continue;

        // ---- Q * K^T with AFFINE decode ----
        {
            float mqk[C];
            const float inv_sqrt_dk = rsqrt(float(DK));

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) {
                    mqk[cc] = 0.0f;
                    continue;
                }

                float k_sn = K_norms[kv_head * params.kv_capacity + kv_pos] * inv_sqrt_dk;

                device const uint8_t *k_base =
                    K_packed + (kv_head * params.kv_capacity + kv_pos) * (DK / 2) + tx * 2u;

                float partial = 0.0f;
                for (short ii = 0; ii < DK4 / NL; ++ii) {
                    float4 k_val = dequant_affine4_float4(k_base, (uint)(ii * NL) * 2u, k_sn);
                    partial += dot(k_val, float4(pq4[ii * NL]));
                }
                mqk[cc] = simd_sum(partial);
            }

            ss[tx] = fma(mqk[tx], params.scale, ss[tx]);
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Online softmax (identical) ----
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

        // ---- O += weights * V with AFFINE decode ----
        {
            float4 lo[DV4 / NL];
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                lo[ii] = float4(0.0f);
            }

            const float inv_sqrt_dv = rsqrt(float(DV));

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) continue;

                float v_sw = V_norms[kv_head * params.kv_capacity + kv_pos] * inv_sqrt_dv * ss[cc];
                device const uint8_t *v_base =
                    V_packed + (kv_head * params.kv_capacity + kv_pos) * (DV / 2) + tx * 2u;

                for (short ii = 0; ii < DV4 / NL; ++ii) {
                    lo[ii] += dequant_affine4_float4(v_base, (uint)(ii * NL) * 2u, v_sw);
                }
            }

            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] += lo[ii];
            }
        }
    }

    if (tiisg == 0) {
        ss[0] = S;
        ss[1] = M;
    }

    so4 -= tiisg;
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

template [[host_name("flash_attn_vec_affine4_dk256")]]
kernel decltype(flash_attn_vec_affine4_impl<256, 256>)
flash_attn_vec_affine4_impl<256, 256>;

template [[host_name("flash_attn_vec_affine4_dk512")]]
kernel decltype(flash_attn_vec_affine4_impl<512, 512>)
flash_attn_vec_affine4_impl<512, 512>;
