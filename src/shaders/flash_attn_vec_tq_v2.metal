// TQ SDPA v2: Tiled shared-memory dequant variant.
//
// Dequants a tile of K positions into shared memory as half4, then runs
// QK dot products from shared memory (same access pattern as F16 SDPA).
// V is handled the same way, reusing the tile buffer.
//
// Key difference from v1: dequant and attend are separate phases, not fused.
// This reduces register pressure and lets the dot product loop compile
// identically to the F16 path.
//
// Tile size: CT positions per tile (tunable, default 16 for dk512, 32 for dk256).
// Shared memory: CT * head_dim * 2 bytes (half) for the KV tile buffer.

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32

// Pad x up to next multiple of n (n must be power of 2).
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
    uint  ring_start;
};

struct FlashAttnVecReduceParams {
    uint nrows;
};

// 4-bit Lloyd-Max codebook for N(0,1).
constant float CODEBOOK_4BIT[16] = {
    -2.7325896f, -2.0690172f, -1.6180464f, -1.2562312f,
    -0.9423405f, -0.6567591f, -0.3880483f, -0.1283950f,
     0.1283950f,  0.3880483f,  0.6567591f,  0.9423405f,
     1.2562312f,  1.6180464f,  2.0690172f,  2.7325896f,
};

// ---------------------------------------------------------------------------
// Dequant a tile of KV positions into shared memory as half4.
//
// For each position in [tile_start, tile_start + CT):
//   Read packed nibbles, look up codebook, multiply by scale_norm,
//   write to tile_buf as half4.
//
// tile_buf layout: [CT, DK4] half4  (CT positions × DK4 half4 vectors)
// ---------------------------------------------------------------------------
template<short DK, short CT>
inline void dequant_tile_to_shmem(
    device const uint8_t *packed,       // [num_kv_heads, capacity, DK/2]
    device const float   *norms,        // [num_kv_heads, capacity]
    uint kv_head,
    uint kv_capacity,
    uint tile_start,
    uint kv_seq_len,
    threadgroup half4 *tile_buf,        // [CT, DK4] output
    ushort tx,
    ushort NW
) {
    constexpr short DK4 = DK / 4;
    const float inv_sqrt_dk = rsqrt(float(DK));

    for (short cc = 0; cc < CT; ++cc) {
        uint kv_pos = tile_start + cc;

        if (kv_pos >= kv_seq_len) {
            // Zero-fill beyond valid positions.
            for (short ii = tx; ii < DK4; ii += NW) {
                tile_buf[cc * DK4 + ii] = half4(0.0h);
            }
            continue;
        }

        float norm = norms[kv_head * kv_capacity + kv_pos];
        float scale_norm = norm * inv_sqrt_dk;

        device const uint8_t *pos_packed =
            packed + (kv_head * kv_capacity + kv_pos) * (DK / 2);

        for (short ii = tx; ii < DK4; ii += NW) {
            uint coord = (uint)ii * 4u;
            uint byte0_idx = coord / 2;
            uint byte1_idx = (coord + 2) / 2;
            uint8_t byte0 = pos_packed[byte0_idx];
            uint8_t byte1 = pos_packed[byte1_idx];

            uint idx0 = byte0 & 0xFu;
            uint idx1 = (byte0 >> 4u) & 0xFu;
            uint idx2 = byte1 & 0xFu;
            uint idx3 = (byte1 >> 4u) & 0xFu;

            tile_buf[cc * DK4 + ii] = half4(
                CODEBOOK_4BIT[idx0] * scale_norm,
                CODEBOOK_4BIT[idx1] * scale_norm,
                CODEBOOK_4BIT[idx2] * scale_norm,
                CODEBOOK_4BIT[idx3] * scale_norm
            );
        }
    }
}

// ---------------------------------------------------------------------------
// TQ SDPA v2: Tiled shared-memory dequant.
//
// Structure:
//   1. Load pre-rotated Q into sq4 (same as v1)
//   2. For each tile of CT KV positions:
//      a. Dequant K tile into shared memory as half4
//      b. Compute Q·K^T from shared memory (same loop structure as F16)
//      c. Online softmax update
//      d. Dequant V tile into shared memory as half4 (reuse buffer)
//      e. Accumulate O += softmax_weights * V from shared memory
//   3. Write output (rotated domain, caller handles inverse FWHT)
// ---------------------------------------------------------------------------
template<short DK, short DV, short CT>
kernel void flash_attn_vec_tq_v2_impl(
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
    constexpr short SH  = 4 * CT;  // CT * 4 halfs for scratch scores

    // Shared memory for KV tile: CT * max(DK4, DV4) half4 elements.
    // Reused for K tile, then V tile within each chunk.
    constexpr short TILE_SIZE = CT * (DK4 > DV4 ? DK4 : DV4); // in half4

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

    const uint NWG = params.nwg;
    const ushort iwg = tgpig[2] % NWG;
    const ushort iq2 = tgpig[1];  // head index
    const ushort iq1 = tgpig[0];  // query index (0 for decode)

    const uint heads_per_kv = params.n_heads / params.n_kv_heads;
    const uint kv_head = iq2 / heads_per_kv;

    // Shared memory layout:
    //   [0, PK)                            — Q as half4
    //   [PK, PK + SH)                      — scratch for scores
    //   [PK + SH, PK + SH + 2*PV)          — output accumulator (float4)
    //   [PK + SH + 2*PV, ...)               — KV tile buffer (half4)
    threadgroup half4  *sq4       = (threadgroup half4  *)(shmem);
    threadgroup float  *ss        = (threadgroup float  *)(shmem + PK);
    threadgroup float4 *so4       = (threadgroup float4 *)(shmem + PK + SH);
    threadgroup half4  *tile_buf  = (threadgroup half4  *)(shmem + PK + SH + 2 * PV);

    const ushort tx = tiisg;

    // Load pre-rotated Q.
    for (ushort i = tx; i < PK4; i += NW) {
        if (i < DK4) {
            float4 qval = *((device const float4 *)(Q + iq2 * DK + i * 4));
            sq4[i] = half4(qval);
        } else {
            sq4[i] = half4(0.0h);
        }
    }

    // Zero output accumulator.
    so4 += tx;
    for (short i = 0; i < DV4 / NL; ++i) {
        so4[i * NL] = float4(0.0f);
    }

    for (ushort i = tx; i < SH / 4; i += NW) {
        ((threadgroup float *)(shmem + PK))[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    const uint kv_seq_len = params.kv_seq_len;
    const uint abs_pos = kv_seq_len - 1;
    const uint causal_max_k = min(abs_pos + 1, kv_seq_len);

    uint window_start = 0;
    if (params.mask_type == 2 && params.sliding_window > 0) {
        window_start = (abs_pos >= params.sliding_window)
            ? (abs_pos - params.sliding_window + 1) : 0;
    }

    threadgroup const half4 *pq4 = sq4 + tx;

    // Main loop: process tiles of CT KV positions.
    for (uint ic0 = iwg; ; ic0 += NWG) {
        uint ic = ic0 * CT;
        if (ic >= causal_max_k) break;

        // Compute mask for this tile.
        for (ushort i = tx; i < CT; i += NW) {
            uint k_pos = ic + i;
            float mask_val = 0.0f;
            if (k_pos >= causal_max_k || k_pos < window_start) {
                mask_val = -65504.0f;
            }
            ss[i] = mask_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Skip all-masked tiles.
        // Check first CT elements for any non-masked.
        bool all_masked = true;
        for (short i = 0; i < CT && all_masked; ++i) {
            if (ss[i] > -65504.0f) all_masked = false;
        }
        if (all_masked) continue;

        // ---- Phase 1: Dequant K tile into shared memory ----
        dequant_tile_to_shmem<DK, CT>(
            K_packed, K_norms, kv_head, params.kv_capacity,
            ic, kv_seq_len, tile_buf, tx, NW
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase 2: Q·K^T from shared memory ----
        {
            float mqk[CT];
            for (short cc = 0; cc < CT; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) {
                    mqk[cc] = 0.0f;
                    continue;
                }

                float partial = 0.0f;
                for (short ii = 0; ii < DK4 / NL; ++ii) {
                    // Read from shared memory tile — same as F16 reads from device.
                    half4 k_h4 = tile_buf[cc * DK4 + ii * NL + tx];
                    partial += dot(float4(k_h4), float4(pq4[ii * NL]));
                }
                mqk[cc] = simd_sum(partial);
            }

            // Combine with mask and scale.
            for (ushort i = tx; i < CT; i += NW) {
                ss[i] = fma(mqk[i], params.scale, ss[i]);
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Online softmax ----
        {
            const float m_old = M;
            float s_new = -FLT_MAX / 2;
            for (short i = 0; i < CT; ++i) {
                s_new = max(s_new, ss[i]);
            }
            M = max(M, s_new);

            const float ms = exp(m_old - M);
            float vs_sum = 0.0f;
            for (short i = 0; i < CT; ++i) {
                float vs = exp(ss[i] - M);
                ss[i] = vs;
                vs_sum += vs;
            }
            S = S * ms + vs_sum;

            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] *= ms;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase 3: Dequant V tile into shared memory (reuse tile_buf) ----
        dequant_tile_to_shmem<DV, CT>(
            V_packed, V_norms, kv_head, params.kv_capacity,
            ic, kv_seq_len, tile_buf, tx, NW
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase 4: O += softmax_weights * V from shared memory ----
        {
            float4 lo[DV4 / NL];
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                lo[ii] = float4(0.0f);
            }

            for (short cc = 0; cc < CT; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) continue;

                float weight = ss[cc];
                for (short ii = 0; ii < DV4 / NL; ++ii) {
                    half4 v_h4 = tile_buf[cc * DV4 + ii * NL + tx];
                    lo[ii] += float4(v_h4) * weight;
                }
            }

            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] += lo[ii];
            }
        }
    }

    // Store M and S.
    if (tx == 0) {
        ss[0] = S;
        ss[1] = M;
    }

    so4 -= tx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write output (rotated domain).
    if (sgitg == 0) {
        const int64_t nrows = params.n_heads;
        const int64_t rid = iq2 + (int64_t)iq1 * params.n_heads;
        const uint NWG_val = params.nwg;

        const float inv_S = (NWG_val == 1) ? ((S == 0.0f) ? 0.0f : 1.0f / S) : 1.0f;

        device float4 *dst4 = (device float4 *)dst;
        for (ushort i = tx; i < DV4; i += NW) {
            dst4[rid * DV4 * NWG_val + NWG_val * i + iwg] = so4[i] * inv_S;
        }

        if (NWG_val > 1 && tx == 0) {
            device float *dst1 = (device float *)dst + nrows * DV * NWG_val;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 0] = S;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 1] = M;
        }
    }
}

// Instantiations:
// dk256: CT=32 (32 * 64 * 2 = 4KB tile — fits easily)
// dk512: CT=16 (16 * 128 * 2 = 4KB tile — fits easily)
// Both use half4 tile buffer.

template [[host_name("flash_attn_vec_tq_v2_dk256")]]
kernel decltype(flash_attn_vec_tq_v2_impl<256, 256, 32>)
flash_attn_vec_tq_v2_impl<256, 256, 32>;

template [[host_name("flash_attn_vec_tq_v2_dk512")]]
kernel decltype(flash_attn_vec_tq_v2_impl<512, 512, 16>)
flash_attn_vec_tq_v2_impl<512, 512, 16>;
