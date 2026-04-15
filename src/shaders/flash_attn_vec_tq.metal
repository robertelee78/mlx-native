// Flash attention vector kernel for TurboQuant-compressed KV cache (ADR-007 Phase 1.3).
//
// Fork of flash_attn_vec.metal that reads K and V from nibble-packed indices
// + per-position norms + pre-rotated centroid table, instead of F16/F32 buffers.
//
// The kernel operates in the Hadamard-rotated domain:
//   1. Q is rotated (FWHT) in shared memory at load time
//   2. K/V are gathered from the centroid table (already in rotated domain)
//   3. Dot products are computed in the rotated domain (orthogonal invariance)
//   4. Accumulated output is rotated back (inverse FWHT) before writing to dst
//
// Pre-rotated centroid table layout: [16, head_dim] F32
//   table[idx * head_dim + c] = codebook[idx] / sqrt(head_dim)
//
// Packed KV layout: [num_kv_heads, capacity, head_dim/2] u8
//   Low nibble = even coordinate index, high nibble = odd coordinate index
//
// Norms layout: [num_kv_heads, capacity] f32

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32   // KV positions per simdgroup iteration

// Pad x up to next multiple of n (n must be power of 2).
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

// Parameters — same layout as FlashAttnVecParams.
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

// Reduce params — shared with flash_attn_vec.
struct FlashAttnVecReduceParams {
    uint nrows;
};

// ---------------------------------------------------------------------------
// In-place FWHT on float data in shared memory.
//
// x: pointer to float array of length D in threadgroup memory.
// D: dimension (must be power of 2, known at compile time).
// NW: number of threads in the simdgroup (32).
// tx: thread index (0..31).
//
// Each butterfly stage requires a threadgroup barrier.
// For D=256: 8 stages. For D=512: 9 stages.
// ---------------------------------------------------------------------------
template<short D>
inline void fwht_shared(threadgroup float *x, ushort tx, ushort NW) {
    // Normalization factor: 1/sqrt(D)
    // Applied once at the end to avoid repeated multiplications.

    short h = 1;
    while (h < D) {
        short step = h * 2;
        // Each thread processes D/NW pairs per stage
        for (short idx = tx; idx < D / 2; idx += NW) {
            // Map flat index to butterfly pair
            short block = idx / h;
            short offset = idx % h;
            short j = block * step + offset;

            float a = x[j];
            float b = x[j + h];
            x[j]     = a + b;
            x[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // Normalize: multiply all elements by 1/sqrt(D)
    float inv_sqrt_d = rsqrt(float(D));
    for (short i = tx; i < D; i += NW) {
        x[i] *= inv_sqrt_d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


// ---------------------------------------------------------------------------
// Reconstruct float4 from 2 packed bytes (4 nibble indices) using centroid table.
//
// packed_base: pointer to start of this position's packed data [head_dim/2 bytes]
// coord_offset: starting coordinate index (must be multiple of 4)
// head_dim: head dimension
// centroids: pre-rotated centroid table [16, head_dim]
// norm: per-position norm scalar
// ---------------------------------------------------------------------------
inline float4 gather_tq_float4(
    device const uint8_t *packed_base,
    uint coord_offset,
    uint head_dim,
    constant float *centroids,
    float norm
) {
    // Read 2 bytes = 4 nibble indices
    // Layout: low nibble = even coord, high nibble = odd coord
    uint byte0_idx = coord_offset / 2;
    uint byte1_idx = (coord_offset + 2) / 2;
    uint8_t byte0 = packed_base[byte0_idx];
    uint8_t byte1 = packed_base[byte1_idx];

    uint idx0 = byte0 & 0xFu;          // coord_offset + 0 (even -> low nibble)
    uint idx1 = (byte0 >> 4u) & 0xFu;  // coord_offset + 1 (odd -> high nibble)
    uint idx2 = byte1 & 0xFu;          // coord_offset + 2 (even -> low nibble)
    uint idx3 = (byte1 >> 4u) & 0xFu;  // coord_offset + 3 (odd -> high nibble)

    float4 result;
    result.x = centroids[idx0 * head_dim + coord_offset + 0] * norm;
    result.y = centroids[idx1 * head_dim + coord_offset + 1] * norm;
    result.z = centroids[idx2 * head_dim + coord_offset + 2] * norm;
    result.w = centroids[idx3 * head_dim + coord_offset + 3] * norm;
    return result;
}


// ---------------------------------------------------------------------------
// Main TQ flash attention vector kernel.
//
// Same structure as flash_attn_vec_impl but:
//   - Q is FWHT-rotated in shared memory before dot products
//   - K/V are gathered from nibble-packed buffers + centroid table
//   - Accumulated output is inverse-FWHT-rotated before writing
// ---------------------------------------------------------------------------
template<short DK, short DV>
kernel void flash_attn_vec_tq_impl(
    constant FlashAttnVecTqParams   &params      [[buffer(0)]],
    device const float              *Q           [[buffer(1)]],
    device const uint8_t            *K_packed    [[buffer(2)]],
    device const float              *K_norms     [[buffer(3)]],
    constant float                  *K_centroids [[buffer(4)]],
    device const uint8_t            *V_packed    [[buffer(5)]],
    device const float              *V_norms     [[buffer(6)]],
    constant float                  *V_centroids [[buffer(7)]],
    device       float              *dst         [[buffer(8)]],
    threadgroup  half               *shmem       [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Compile-time constants.
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

    // Shared memory layout:
    //   [0, PK)                     — Q as half4 (pre-rotated, loaded directly)
    //   [PK, PK + SH)               — scratch for attention scores
    //   [PK + SH, PK + SH + 2*PV)   — output accumulator as float4
    //
    // No FWHT scratch needed — Q arrives pre-rotated from a standalone dispatch,
    // and the output is written in the rotated domain for a standalone inverse FWHT.

    // Pointers.
    threadgroup half4  *sq4 = (threadgroup half4  *)(shmem);
    threadgroup float  *ss  = (threadgroup float  *)(shmem + PK);
    threadgroup float4 *so4 = (threadgroup float4 *)(shmem + PK + SH);

    // Load PRE-ROTATED Q directly into shared memory as half4.
    // The caller has already applied FWHT to Q via a standalone dispatch.
    // No in-kernel rotation needed — just load and cast F32 → half4.
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

    // Zero the output accumulator.
    // Each thread owns its SIMD lane.
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

    // Masking bounds.
    const uint kv_seq_len = params.kv_seq_len;
    const uint abs_pos = kv_seq_len - 1;
    const uint causal_max_k = min(abs_pos + 1, kv_seq_len);

    uint window_start = 0;
    if (params.mask_type == 2 && params.sliding_window > 0) {
        window_start = (abs_pos >= params.sliding_window)
            ? (abs_pos - params.sliding_window + 1) : 0;
    }

    // Reference to Q in shared memory for dot products.
    threadgroup const half4 *pq4 = sq4 + tx;

    // Main loop over KV cache in chunks of C=32.
    for (uint ic0 = iwg; ; ic0 += NWG) {
        uint ic = ic0 * C;
        if (ic >= causal_max_k) {
            break;
        }

        // Compute implicit mask for this chunk.
        {
            uint k_pos = ic + tx;
            float mask_val = 0.0f;
            if (k_pos >= causal_max_k || k_pos < window_start) {
                mask_val = -65504.0f;
            }
            ss[tx] = mask_val;
        }

        // Skip all-masked chunks.
        if (simd_max(ss[tiisg]) <= -65504.0f) {
            continue;
        }

        // ---- Q * K^T (in rotated domain) ----
        {
            float mqk[C];

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) {
                    mqk[cc] = 0.0f;
                    continue;
                }

                // Get norm for this K position.
                float k_norm = K_norms[kv_head * params.kv_capacity + kv_pos];

                // Pointer to packed K data for this position.
                device const uint8_t *k_packed_pos =
                    K_packed + (kv_head * params.kv_capacity + kv_pos) * (DK / 2);

                float partial = 0.0f;
                for (short ii = 0; ii < DK4 / NL; ++ii) {
                    uint coord = (uint)(ii * NL + tx) * 4u;
                    float4 k_val = gather_tq_float4(
                        k_packed_pos, coord, (uint)DK, K_centroids, k_norm);
                    partial += dot(k_val, float4(pq4[ii * NL]));
                }
                mqk[cc] = simd_sum(partial);
            }

            // Combine with mask and scale.
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

            // Rescale previous output accumulation.
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] *= ms;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O + softmax_weights * V (in rotated domain) ----
        {
            float4 lo[DV4 / NL];
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                lo[ii] = float4(0.0f);
            }

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) continue;

                float v_norm = V_norms[kv_head * params.kv_capacity + kv_pos];
                device const uint8_t *v_packed_pos =
                    V_packed + (kv_head * params.kv_capacity + kv_pos) * (DV / 2);

                float weight = ss[cc];
                for (short ii = 0; ii < DV4 / NL; ++ii) {
                    uint coord = (uint)(ii * NL + tx) * 4u;
                    float4 v_val = gather_tq_float4(
                        v_packed_pos, coord, (uint)DV, V_centroids, v_norm);
                    lo[ii] += v_val * weight;
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

    // Remove per-thread offset before cross-simdgroup access.
    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Inverse FWHT on accumulated output ----
    // The output is in the rotated domain. Apply FWHT to rotate back.
    // (Since H^{-1} = H for normalized Hadamard, inverse = forward.)
    //
    // The output is stored as float4 in so4[0..DV4).
    // We need to copy it to a contiguous float array, apply FWHT, then write out.
    //
    // Reuse fwht_scratch = (float*)(shmem + PK + SH) which overlaps so4.
    // So first copy so4 to a temporary location... but we don't have one.
    //
    // Actually, so4 IS fwht_scratch (same memory). The float4 layout in so4
    // Output stays in rotated domain — the caller applies inverse FWHT
    // via a standalone dispatch after this kernel completes.
    // No in-kernel FWHT on the output.
    if (sgitg == 0) {
        const int64_t nrows = params.n_heads;
        const int64_t rid = iq2 + (int64_t)iq1 * params.n_heads;

        const float inv_S = (S == 0.0f) ? 0.0f : 1.0f / S;

        // Normalize the output by 1/S.
        for (ushort i = tiisg; i < DV4; i += NW) {
            so4[i] *= inv_S;
        }

        // Write to destination (still in rotated domain).
        device float4 *dst4 = (device float4 *)dst;
        const uint NWG_val = params.nwg;

        for (ushort i = tiisg; i < DV4; i += NW) {
            dst4[rid * DV4 * NWG_val + NWG_val * i + iwg] = so4[i];
        }

        // Store S and M for the reduce kernel (in case NWG > 1).
        if (NWG_val > 1 && tiisg == 0) {
            device float *dst1 = (device float *)dst + nrows * DV * NWG_val;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 0] = S;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 1] = M;
        }
    }
}


// --------------------------------------------------------------------------
// Kernel instantiations — TQ variants
// --------------------------------------------------------------------------

typedef decltype(flash_attn_vec_tq_impl<256, 256>) flash_attn_vec_tq_t;

template [[host_name("flash_attn_vec_tq_dk256")]]
kernel flash_attn_vec_tq_t flash_attn_vec_tq_impl<256, 256>;

template [[host_name("flash_attn_vec_tq_dk512")]]
kernel flash_attn_vec_tq_t flash_attn_vec_tq_impl<512, 512>;
