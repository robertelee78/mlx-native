#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

/// ADR-020 iter-15c-2 — 4-simdgroup-per-threadgroup variant of
/// `qmm_affine_t_f32_simd`.  Mirrors the GGML qmm_mm reference
/// kernel structure (`quantized_matmul_mm.metal`) for full Apple GPU
/// warp-pool exploitation: 128 threads / TG (= 4 simdgroups), 32×32
/// output tile per TG, each simdgroup owns a 16×16 sub-tile via 4
/// `simdgroup_matrix<float, 8, 8>` accumulators in a 2×2 grid.
///
/// Same I/O contract as iter-15 / 15b / 15c-1.  Same constraint:
/// `group_size == 32` (= BK).  Differs only in tile geometry; the
/// math is identical.
///
/// ## Layout
///
///   * BM = BN = BK = 32.
///   * 4 simdgroups per TG, indexed by `sgitg` ∈ {0, 1, 2, 3}.
///   * 2×2 simdgroup grid: `sg_row = sgitg / 2`, `sg_col = sgitg % 2`.
///   * Each simdgroup owns output rows `sg_row*16 .. +16` and cols
///     `sg_col*16 .. +16` of the 32×32 TG output tile.
///   * Per simdgroup: 4 `simdgroup_float8x8` accumulators arranged as
///     a 2×2 grid:
///       `mc[0]` rows 0..7, cols 0..7    (relative to sub-tile origin)
///       `mc[1]` rows 0..7, cols 8..15
///       `mc[2]` rows 8..15, cols 0..7
///       `mc[3]` rows 8..15, cols 8..15
///
/// ## Algorithm (per K-step of `BK = 32`)
///
///   1. Cooperatively load `X[BM, BK]` into `a_tile` (128 threads × 8
///      elements/thread = 1024 floats).
///   2. Cooperatively dequant `W[BN, BK]` into `b_tile`
///      (`b_tile[n][k] = q_int[n, k]·scales[n, kt] + biases[n, kt]`).
///   3. Inner reduction over 4 sub-K-tiles (BK/8):
///        For each `ik` in 0..3:
///          Per simdgroup, load 2 `ma` (row-blocks 0 and 1 of own
///          sub-tile, K cols ik*8..ik*8+7) and 2 `mb` (col-blocks 0
///          and 1, transposed K-major from b_tile).  4 MMAs:
///          `mc[i*2 + j] += ma[i] * mb[j]`.
///
/// ## Performance expectation
///
/// Each Apple GPU core can issue 4 simdgroup_matrix MMAs concurrently.
/// iter-15c-1 used 1 simdgroup per TG → 1 MMA-issue/cycle/TG.
/// iter-15c-2 uses 4 simdgroups per TG → 4 MMA-issues/cycle/TG, with
/// the SAME shared-memory tile reuse cost.  Expected 3-4× over
/// iter-15c-1 → 7-10 GFLOPS class.
///
/// ## Threadgroup-shared memory budget
///
///   a_tile : 32×32 = 1024 floats = 4096 bytes
///   b_tile : 32×32 = 1024 floats = 4096 bytes
///   total                                = 8192 bytes
///
/// Apple Metal threadgroup-shared limit is 32 KB → 8 KB is safe.

constant constexpr uint BM = 32;
constant constexpr uint BN = 32;
constant constexpr uint BK = 32;

kernel void qmm_affine_t_f32_simd4(
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
    // BK is host-validated to equal group_size = 32.

    const uint groups_per_row = K / BK;

    // Threadgroup-shared layout.
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
        // 128 threads × 8 elements/thread.
        for (uint slot = 0; slot < 8; slot++) {
            const uint lin = slot * 128 + tid_in_tg;          // 0..1023
            const uint r = lin / BK;                           // 0..31
            const uint c = lin % BK;                           // 0..31
            const uint mm = m_origin + r;
            float v = 0.0f;
            if (mm < M && (k_base + c) < K) {
                v = x[mm * K + k_base + c];
            }
            a_tile[lin] = v;
        }

        // ---- Cooperative dequant: W tile [BN=32, BK=32] = 1024 floats ----
        // Each thread dequants 8 elements: q_int[n, k] · scales[n, kt] + biases[n, kt].
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
                const float q = float(q_int[nn * K + k_base + c]);
                v = q * s + b;
            }
            b_tile[lin] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Inner MMA: 4 sub-K-tiles × 4 MMAs/simdgroup ----
        // For this simdgroup at (sg_row, sg_col):
        //   ma[i]: rows in a_tile sg_row*16+i*8 .. +8, K cols ik*8..ik*8+7
        //   mb[j]: rows = K cols (transpose=true), N cols sg_col*16+j*8 .. +8
        //   mc[i*2 + j] += ma[i] * mb[j]
        for (uint ik = 0; ik < BK / 8; ik++) {
            simdgroup_float8x8 ma0, ma1;
            simdgroup_float8x8 mb0, mb1;

            // ma loads from a_tile [BM=32][BK=32], stride BK=32.
            simdgroup_load(ma0,
                a_tile + (sg_row * 16 + 0) * BK + ik * 8, BK, 0, false);
            simdgroup_load(ma1,
                a_tile + (sg_row * 16 + 8) * BK + ik * 8, BK, 0, false);

            // mb loads from b_tile [BN=32][BK=32], transpose=true.
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
    // The fast/slow path decision MUST be uniform across the
    // threadgroup so the slow path's threadgroup_barriers don't
    // deadlock — all 128 threads must hit the same barriers, so
    // we gate on the WHOLE 32×32 TG output tile being in bounds,
    // not per-simdgroup sub-tile bounds.
    const bool full_tile_in_bounds =
        (m_origin + BM <= M) && (n_origin + BN <= N);
    if (full_tile_in_bounds) {
        device float *base = y + sg_m_origin * N + sg_n_origin;
        // mc[0]: rows 0..7, cols 0..7
        simdgroup_store(mc[0], base + 0 * N + 0, N, 0, false);
        // mc[1]: rows 0..7, cols 8..15
        simdgroup_store(mc[1], base + 0 * N + 8, N, 0, false);
        // mc[2]: rows 8..15, cols 0..7
        simdgroup_store(mc[2], base + 8 * N + 0, N, 0, false);
        // mc[3]: rows 8..15, cols 8..15
        simdgroup_store(mc[3], base + 8 * N + 8, N, 0, false);
    } else {
        // Partial-tile path: stage 16×16 sub-tile to threadgroup-
        // shared memory (reusing a_tile region — 16×16 = 256 floats =
        // 1024 bytes, well within a_tile's 4096 bytes), then copy out
        // element-wise with bounds check.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Per simdgroup, stage to its own quadrant of a_tile.  Use
        // a_tile[sg_row * 16 * BM + sg_col * 16 + r * BM + c] —
        // treating a_tile as a [BM=32][BM=32]-strided buffer for
        // staging 4 sub-tiles side by side.
        threadgroup float *staging =
            a_tile + (sg_row * 16) * BM + (sg_col * 16);
        // Each mc[i] writes to staging at its (8 row × 8 col) offset.
        simdgroup_store(mc[0], staging + 0 * BM + 0, BM, 0, false);
        simdgroup_store(mc[1], staging + 0 * BM + 8, BM, 0, false);
        simdgroup_store(mc[2], staging + 8 * BM + 0, BM, 0, false);
        simdgroup_store(mc[3], staging + 8 * BM + 8, BM, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 128 threads cooperatively copy 32×32 = 1024 elements out
        // (8 each).
        for (uint slot = 0; slot < 8; slot++) {
            const uint lin = slot * 128 + tid_in_tg;
            const uint r = lin / BM;                           // 0..31
            const uint c = lin % BM;                           // 0..31
            const uint mm = m_origin + r;
            const uint nn = n_origin + c;
            if (mm < M && nn < N) {
                y[mm * N + nn] = a_tile[r * BM + c];
            }
        }
    }
}
