#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

/// ADR-020 iter-15c-2b — `gs=64` variant of `qmm_affine_t_f32_simd4`.
///
/// Identical to `qmm_affine_simd4.metal` except the K-tile depth (=
/// per-row group size) is 64 instead of 32, matching mlx-lm's canonical
/// `dynamic_quant.py` default group size.  Apple GPU MMA tile width
/// is fixed at 8 → with BK=64 we run 8 sub-K-tiles per K-step instead
/// of 4.
///
/// Why ship both: GGUF Q4_0 uses gs=32, mlx-lm DWQ output uses gs=64.
/// hf2q's Track 2 inference path (loading mlx-format DWQ-trained
/// safetensors per iter-16) needs gs=64 to be production-fast.
///
/// Layout:
///   * BM = BN = 32, BK = 64 (= group_size)
///   * 4 simdgroups in 2x2 grid; each owns 16x16 sub-tile = 4 MMA accums
///   * a_tile[32][64] + b_tile[32][64] = 16384 bytes shared
///     (Apple Metal threadgroup-shared limit is 32 KB → safe)
///   * Cooperative load: 128 threads × 16 elements/thread = 2048 floats per tile
///   * Inner reduction: 8 sub-K-tiles × 4 MMAs/simdgroup = 32 MMAs/sg/K-step
///
/// Same fast/slow write-back paths as iter-15c-2 (uniform TG-wide
/// bounds gate to avoid divergent-barrier deadlock).

constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 64;

kernel void qmm_affine_t_f32_simd4_gs64(
    device const float    *x          [[buffer(0)]],
    device const uchar    *q_int      [[buffer(1)]],
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
    // BK is host-validated to equal group_size = 64.

    const uint groups_per_row = K / BK;

    threadgroup float *a_tile = shmem;
    threadgroup float *b_tile = shmem + BM * BK;

    const uint m_origin = tg_id.x * BM;
    const uint n_origin = tg_id.y * BN;

    const uint sg_row = sgitg / 2;
    const uint sg_col = sgitg % 2;
    const uint sg_m_origin = m_origin + sg_row * 16;
    const uint sg_n_origin = n_origin + sg_col * 16;

    simdgroup_float8x8 mc[4];
    for (uint i = 0; i < 4; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint kt = 0; kt < groups_per_row; kt++) {
        const uint k_base = kt * BK;

        // ---- Cooperative load: X tile [BM=32, BK=64] = 2048 floats ----
        // 128 threads × 16 elements/thread.
        for (uint slot = 0; slot < 16; slot++) {
            const uint lin = slot * 128 + tid_in_tg;          // 0..2047
            const uint r = lin / BK;                           // 0..31
            const uint c = lin % BK;                           // 0..63
            const uint mm = m_origin + r;
            float v = 0.0f;
            if (mm < M && (k_base + c) < K) {
                v = x[mm * K + k_base + c];
            }
            a_tile[lin] = v;
        }

        // ---- Cooperative dequant: W tile [BN=32, BK=64] = 2048 floats ----
        for (uint slot = 0; slot < 16; slot++) {
            const uint lin = slot * 128 + tid_in_tg;
            const uint r = lin / BK;                           // 0..31 = n_local
            const uint c = lin % BK;                           // 0..63 = k_local
            const uint nn = n_origin + r;
            float v = 0.0f;
            if (nn < N && (k_base + c) < K) {
                const uint sb_idx = nn * groups_per_row + kt;
                const float s = scales[sb_idx];
                const float b = biases[sb_idx];
                const float q = float(q_int[nn * K + k_base + c]);
                v = q * s + b;
            }
            b_tile[lin] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Inner MMA: BK/8 = 8 sub-K-tiles × 4 MMAs/simdgroup ----
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

    // ---- Write-back ----
    // TG-wide uniform bounds gate (same fix as 15c-2).
    const bool full_tile_in_bounds =
        (m_origin + BM <= M) && (n_origin + BN <= N);
    if (full_tile_in_bounds) {
        device float *base = y + sg_m_origin * N + sg_n_origin;
        simdgroup_store(mc[0], base + 0 * N + 0, N, 0, false);
        simdgroup_store(mc[1], base + 0 * N + 8, N, 0, false);
        simdgroup_store(mc[2], base + 8 * N + 0, N, 0, false);
        simdgroup_store(mc[3], base + 8 * N + 8, N, 0, false);
    } else {
        // Partial-tile path: stage 32×32 output to a_tile (4096 bytes
        // — first half of a_tile, well within 32×64×4=8192 budget),
        // then 128 threads cooperatively copy out (8 elements each
        // for a 32×32 = 1024-element output tile).
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
