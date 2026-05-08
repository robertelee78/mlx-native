#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

/// ADR-020 AC#5 Iter A — packed-U32 variant of `qmm_affine_t_f32_simd4`
/// for bits=4.  Mirrors the canonical mlx upstream `affine_qmm_t`
/// (`/private/tmp/mlx-quant/quantized.h:1716`) on-disk packing
/// convention: weight tensor shape `[N, K/8]` U32 row-major, where
/// each u32 holds 8 consecutive 4-bit codes along K (low nibble at
/// k % 8 == 0, high nibble at k % 8 == 7), matching mlx's
/// `mlx/ops.cpp:4762-4772` `left_shift(i*bits)` packing.
///
/// This is the production decode/prefill path for serving DWQ-trained
/// safetensors output (`hf2q dwq-train --bits 4`) through hf2q's serve
/// runtime.  The unpacked iter-15c-2 (`qmm_affine_t_f32_simd4`) was a
/// correctness-first kernel; this packed variant reads the on-disk
/// layout directly without an unpack pass at load time, saving 8×
/// resident weight memory at bits=4.
///
/// Same threadgroup geometry / accumulator structure / write-back as
/// `qmm_affine_t_f32_simd4`.  Only the cooperative dequant load loop
/// differs (packed-U32 unpack instead of one-byte-per-code load).

constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;
constant constexpr uint PACK_FACTOR = 8;     // bits=4: 8 codes per u32
constant constexpr uint NIBBLE_MASK = 0x0F;

kernel void qmm_affine_t_packed_simd4_b4(
    device const float    *x          [[buffer(0)]],
    device const uint     *w_packed   [[buffer(1)]],
    device const float    *scales     [[buffer(2)]],
    device const float    *biases     [[buffer(3)]],
    device float          *y          [[buffer(4)]],
    device const uint     *meta       [[buffer(5)]],  // [M, N, K, group_size]
    threadgroup float     *shmem      [[threadgroup(0)]],
    uint   tid_in_tg                  [[thread_index_in_threadgroup]],
    uint   sgitg                      [[simdgroup_index_in_threadgroup]],
    uint3  tg_id                      [[threadgroup_position_in_grid]]
) {
    const uint M = meta[0];
    const uint N = meta[1];
    const uint K = meta[2];
    // BK is host-validated to equal group_size = 32.

    const uint groups_per_row = K / BK;
    const uint k_packed_per_row = K / PACK_FACTOR;  // = K/8 for bits=4

    // Threadgroup-shared layout (matches simd4 unpacked variant).
    threadgroup float *a_tile = shmem;
    threadgroup float *b_tile = shmem + BM * BK;

    // Output tile origin.
    const uint m_origin = tg_id.x * BM;
    const uint n_origin = tg_id.y * BN;

    // Simdgroup partitioning of the 32×32 tile into 2×2 of 16×16.
    const uint sg_row = sgitg / 2;
    const uint sg_col = sgitg % 2;
    const uint sg_m_origin = m_origin + sg_row * 16;
    const uint sg_n_origin = n_origin + sg_col * 16;

    // 4 MMA accumulators per simdgroup (2×2 grid of 8×8).
    simdgroup_float8x8 mc[4];
    for (uint i = 0; i < 4; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // Sweep K in BK-sized chunks (= group_size).
    for (uint kt = 0; kt < groups_per_row; kt++) {
        const uint k_base = kt * BK;

        // ---- Cooperative load: X tile [BM=32, BK=32] = 1024 floats ----
        for (uint slot = 0; slot < 8; slot++) {
            const uint lin = slot * 128 + tid_in_tg;
            const uint r = lin / BK;
            const uint c = lin % BK;
            const uint mm = m_origin + r;
            float v = 0.0f;
            if (mm < M && (k_base + c) < K) {
                v = x[mm * K + k_base + c];
            }
            a_tile[lin] = v;
        }

        // ---- Cooperative dequant: W tile [BN=32, BK=32] = 1024 floats ----
        // Packed-U32 inner load: 8 codes per u32 along K.  For each
        // (n_local=r, k_local=c) within the BN×BK tile, compute the
        // pack index pack = (k_base + c)/8 and the slot j = (k_base + c)%8,
        // then mask out 4 bits.
        for (uint slot = 0; slot < 8; slot++) {
            const uint lin = slot * 128 + tid_in_tg;
            const uint r = lin / BK;                           // 0..31 = n_local
            const uint c = lin % BK;                           // 0..31 = k_local
            const uint nn = n_origin + r;
            float v = 0.0f;
            if (nn < N && (k_base + c) < K) {
                const uint sb_idx = nn * groups_per_row + kt;
                const float s = scales[sb_idx];
                const float b = biases[sb_idx];
                const uint k_abs = k_base + c;
                const uint pack_idx = k_abs / PACK_FACTOR;
                const uint slot_in_pack = k_abs % PACK_FACTOR;
                const uint w_word = w_packed[nn * k_packed_per_row + pack_idx];
                const uint q_u = (w_word >> (slot_in_pack * 4)) & NIBBLE_MASK;
                const float q = float(q_u);
                v = q * s + b;
            }
            b_tile[lin] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Inner MMA: 4 sub-K-tiles × 4 MMAs/simdgroup ----
        for (uint ik = 0; ik < BK / 8; ik++) {
            simdgroup_float8x8 ma0, ma1;
            simdgroup_float8x8 mb0, mb1;

            simdgroup_load(ma0,
                a_tile + (sg_row * 16 + 0) * BK + ik * 8, BK, 0, false);
            simdgroup_load(ma1,
                a_tile + (sg_row * 16 + 8) * BK + ik * 8, BK, 0, false);

            simdgroup_load(mb0,
                b_tile + (sg_col * 16 + 0) * BK + ik * 8, BK, 0, true);
            simdgroup_load(mb1,
                b_tile + (sg_col * 16 + 8) * BK + ik * 8, BK, 0, true);

            simdgroup_multiply_accumulate(mc[0], ma0, mb0, mc[0]);
            simdgroup_multiply_accumulate(mc[1], ma0, mb1, mc[1]);
            simdgroup_multiply_accumulate(mc[2], ma1, mb0, mc[2]);
            simdgroup_multiply_accumulate(mc[3], ma1, mb1, mc[3]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write-back ---- (identical to unpacked simd4 variant)
    const bool full_tile_in_bounds =
        (m_origin + BM <= M) && (n_origin + BN <= N);
    if (full_tile_in_bounds) {
        device float *base = y + sg_m_origin * N + sg_n_origin;
        simdgroup_store(mc[0], base + 0 * N + 0, N, 0, false);
        simdgroup_store(mc[1], base + 0 * N + 8, N, 0, false);
        simdgroup_store(mc[2], base + 8 * N + 0, N, 0, false);
        simdgroup_store(mc[3], base + 8 * N + 8, N, 0, false);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *staging =
            a_tile + (sg_row * 16) * BM + (sg_col * 16);
        simdgroup_store(mc[0], staging + 0 * BM + 0, BM, 0, false);
        simdgroup_store(mc[1], staging + 0 * BM + 8, BM, 0, false);
        simdgroup_store(mc[2], staging + 8 * BM + 0, BM, 0, false);
        simdgroup_store(mc[3], staging + 8 * BM + 8, BM, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slot = 0; slot < 8; slot++) {
            const uint lin = slot * 128 + tid_in_tg;
            const uint r = lin / BM;
            const uint c = lin % BM;
            const uint mm = m_origin + r;
            const uint nn = n_origin + c;
            if (mm < M && nn < N) {
                y[mm * N + nn] = a_tile[r * BM + c];
            }
        }
    }
}
