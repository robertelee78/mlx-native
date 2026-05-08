#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

/// ADR-020 iter-15c — simdgroup-MMA variant of `qmm_affine_t_f32`.
///
/// Same I/O contract as iter-15 / iter-15b: computes
/// `Y[m, n] = Σ_k X[m, k] · (q_int[n, k]·scales[n, g(k)] + biases[n, g(k)])`
/// for affine-quantized weight tensors.  Uses Apple GPU's hardware
/// `simdgroup_matrix<float, 8, 8>` MMA instead of the scalar
/// per-thread inner loop in iter-15b.
///
/// ## Layout
///
///   * Each threadgroup: ONE simdgroup (32 threads).
///   * Each TG produces an 8×8 tile of `Y` (via `simdgroup_store`).
///   * Grid: `(ceil(M/8), ceil(N/8), 1)` threadgroups.
///
/// ## Algorithm
///
///   * Per K-step of `BK = 32` (= group_size, same constraint as 15b):
///     - Cooperatively load `X[8][32]` into `a_tile` (32 floats per thread,
///       8 floats × 4 sub-blocks of 8 K-elements each).
///     - Cooperatively dequantize `W[8][32]` into `b_tile` (one 8-element
///       row per thread; per-element dequant `q · scale + bias`).
///     - Inner unroll over `BK / 8 = 4` sub-tiles:
///         simdgroup_load `ma` from `a_tile`, `mb` from `b_tile`,
///         `simdgroup_multiply_accumulate(mc, ma, mb, mc)`.
///   * Final: `simdgroup_store(mc → Y[8][8])` (with bounds check).
///
/// ## Performance expectation
///
/// Apple GPU hardware MMA does an 8x8x8 multiply-accumulate per simdgroup
/// per cycle.  iter-15b's 256-thread × 32-iter scalar inner loop is
/// compute-bound at ~2 TFLOPS on M5 Max.  iter-15c's MMA path does the
/// same work in `(32 × 4) = 128` MMA cycles per output tile (vs `256 × 32 = 8192`
/// scalar cycles), an 8× algorithmic improvement that lands as 3-4× wall
/// once threadgroup-launch + load/store overhead amortizes.
///
/// ## Constraints
///
///   * `group_size == 32` (= BK; one (scales, biases) pair per K-tile).
///   * `K` divisible by `group_size` (host-validated; same as 15b).
///   * `M, N` not required to be multiples of 8 — bounds-check at
///     write-back; partial-tile output uses scalar fallback path.
///   * Single simdgroup per TG (32 threads); shared memory budget
///     `8*32*4 (a_tile) + 8*32*4 (b_tile) = 2048 bytes`.

constant constexpr uint BM = 8;
constant constexpr uint BN = 8;
constant constexpr uint BK = 32;

kernel void qmm_affine_t_f32_simd(
    device const float    *x          [[buffer(0)]],
    device const uchar    *q_int      [[buffer(1)]],
    device const float    *scales     [[buffer(2)]],
    device const float    *biases     [[buffer(3)]],
    device float          *y          [[buffer(4)]],
    device const uint     *meta       [[buffer(5)]],  // [M, N, K, group_size]
    threadgroup float     *shmem      [[threadgroup(0)]],
    uint   tid_in_simd                [[thread_index_in_simdgroup]],
    uint3  tg_id                      [[threadgroup_position_in_grid]]
) {
    const uint M = meta[0];
    const uint N = meta[1];
    const uint K = meta[2];
    // BK is host-validated to equal group_size = 32.

    const uint groups_per_row = K / BK;

    // Threadgroup-shared layout:
    //   shmem[0 .. BM*BK)         = a_tile (X)
    //   shmem[BM*BK .. 2*BM*BK)   = b_tile (dequantized W)
    threadgroup float *a_tile = shmem;
    threadgroup float *b_tile = shmem + BM * BK;

    // Output tile origin in (M, N).
    const uint m_origin = tg_id.x * BM;
    const uint n_origin = tg_id.y * BN;

    // MMA accumulator (8x8 floats — one full output tile).
    simdgroup_float8x8 mc = make_filled_simdgroup_matrix<float, 8>(0.f);

    // Sweep K in BK-sized chunks (= group_size).
    for (uint kt = 0; kt < groups_per_row; kt++) {
        const uint k_base = kt * BK;

        // ---- Cooperative load: X tile [BM=8, BK=32] = 256 floats ----
        // 32 threads × 8 floats per thread.  Thread `tid_in_simd` loads
        // floats `tid_in_simd, tid_in_simd+32, ..., tid_in_simd + 7*32`
        // (column-stride layout matches `simdgroup_load` row-stride 8).
        for (uint slot = 0; slot < 8; slot++) {
            const uint lin = slot * 32 + tid_in_simd;       // 0..255
            const uint r = lin / BK;                         // 0..7
            const uint c = lin % BK;                         // 0..31
            const uint mm = m_origin + r;
            float v = 0.0f;
            if (mm < M && (k_base + c) < K) {
                v = x[mm * K + k_base + c];
            }
            a_tile[lin] = v;
        }

        // ---- Cooperative dequant: W tile [BN=8, BK=32] = 256 floats ----
        // Each thread dequants 8 elements.  Per output row n:
        //   w_dq[n, c] = q_int[n, k_base + c] * scales[n, kt] + biases[n, kt]
        // Layout in b_tile: b_tile[r * BK + c]   (same row-major).
        for (uint slot = 0; slot < 8; slot++) {
            const uint lin = slot * 32 + tid_in_simd;
            const uint r = lin / BK;                         // 0..7
            const uint c = lin % BK;                         // 0..31
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

        // ---- Inner MMA: 4 sub-tiles of 8x8 along the BK axis ----
        // a_tile is laid out as [BM=8 rows][BK=32 cols].
        // simdgroup_load reads an 8x8 tile from a contiguous stride-8 region.
        // For sub-tile `ik` (k = ik*8 .. ik*8+8):
        //   ma rows: a_tile[r][ik*8 .. ik*8+8]
        //   mb rows: b_tile[r][ik*8 .. ik*8+8]^T (transposed for k×n form)
        //
        // Since both tiles are row-major [8][32], the simdgroup-load source
        // pointer is `a_tile + ik*8` with row-stride BK (= 32) — but
        // simdgroup_load wants stride argument 8.  We handle this by
        // staging an explicit 8x8 sub-block to a contiguous register-width
        // pattern; here we use the in-place stride form directly: pass
        // `BK` as the matrix stride on a "row-major 8x8 view".
        for (uint ik = 0; ik < BK / 8; ik++) {
            simdgroup_float8x8 ma;
            simdgroup_float8x8 mb;
            simdgroup_load(ma, a_tile + ik * 8, BK, 0, false);
            // For mb we want the K axis as the OUTER (rows of the 8x8
            // matrix) so that ma * mb matches Y = X · W^T orientation.
            // b_tile is [n_row][k_col]; we want [k_col][n_row].  Pass
            // `transpose = true` to simdgroup_load to read with K axis on
            // the outer.
            simdgroup_load(mb, b_tile + ik * 8, BK, 0, true);
            simdgroup_multiply_accumulate(mc, ma, mb, mc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write-back ----
    // Fast path: full 8x8 tile inside (M, N) bounds.
    const bool full_tile_in_bounds =
        (m_origin + BM <= M) && (n_origin + BN <= N);
    if (full_tile_in_bounds) {
        device float *y_origin = y + m_origin * N + n_origin;
        simdgroup_store(mc, y_origin, N, 0, false);
    } else {
        // Partial-tile path: stage 8x8 tile to shared memory, copy out
        // element-wise with bounds check.
        threadgroup float *staging = a_tile;  // reuse a_tile for output
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(mc, staging, BN, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 32 threads cooperatively copy 8x8 = 64 elements (2 each).
        for (uint slot = 0; slot < 2; slot++) {
            const uint lin = slot * 32 + tid_in_simd;        // 0..63
            const uint r = lin / BN;                          // 0..7
            const uint c = lin % BN;                          // 0..7
            const uint mm = m_origin + r;
            const uint nn = n_origin + c;
            if (mm < M && nn < N) {
                y[mm * N + nn] = staging[r * BN + c];
            }
        }
    }
}
