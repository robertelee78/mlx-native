#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-15b — tiled variant of `qmm_affine_t_f32`.
///
/// Same math + I/O contract as iter-15's per-element kernel, but
/// shifts the inner reduction onto threadgroup-shared tiles so reads
/// of X and (scales, biases) are amortized across the 16x16 thread
/// block.  Per-element kernel reads X[m, k] independently for each
/// (m, n) pair; tiled variant loads X[BM, BK] once per K-tile and
/// reuses it across BN output columns of the tile (register-level
/// reuse via the inner loop).
///
/// Tile geometry (compile-time constants, must agree with the host
/// dispatcher's threadgroup size and shared-memory allocation):
///   BM = 16   output rows per threadgroup
///   BN = 16   output cols per threadgroup
///   BK = 32   K-tile depth = group_size (REQUIRED — see below)
///
/// **Constraint**: this kernel REQUIRES `group_size == 32`.  This
/// makes BK align exactly with one (scales, biases) pair per
/// K-tile per output row, eliminating the inner per-group fetch.
/// The host validates this; for `group_size != 32` callers fall back
/// to the iter-15 per-element kernel.
///
/// Threadgroup-shared budget:
///   x_tile : f32[BM][BK] = 16 * 32 * 4 =  2048 bytes
///   q_tile :  u8[BN][BK] = 16 * 32 * 1 =   512 bytes
///   s_tile : f32[BN]      =      16 * 4 =    64 bytes
///   b_tile : f32[BN]      =      16 * 4 =    64 bytes
///   total                                =  2688 bytes
///
/// Threadgroup: 16x16 = 256 threads.  Each thread computes one
/// output element y[m, n].
///
/// Performance vs iter-15 per-element kernel: ~2-5× depending on
/// matrix size (smaller speedup at small M/N where TG-overhead
/// dominates).  The simdgroup-MMA optimization (mlx::steel::BlockMMA
/// equivalent) is queued for iter-15c.

constant constexpr uint BM = 16;
constant constexpr uint BN = 16;
constant constexpr uint BK = 32;

kernel void qmm_affine_t_f32_tiled(
    device const float    *x          [[buffer(0)]],
    device const uchar    *q_int      [[buffer(1)]],
    device const float    *scales     [[buffer(2)]],
    device const float    *biases     [[buffer(3)]],
    device float          *y          [[buffer(4)]],
    device const uint     *meta       [[buffer(5)]],  // [M, N, K, group_size]
    threadgroup float     *shmem      [[threadgroup(0)]],
    uint2 tid                         [[thread_position_in_threadgroup]],
    uint2 bid                         [[threadgroup_position_in_grid]]
) {
    const uint M          = meta[0];
    const uint N          = meta[1];
    const uint K          = meta[2];
    // group_size is host-validated to be 32 for this kernel — the
    // shader skips the meta[3] dynamic check and uses BK directly.

    // Threadgroup-shared layout:
    //   shmem[0 .. BM*BK)         = x_tile  (f32, BM*BK floats)
    //   shmem[BM*BK .. BM*BK+BN]  = s_tile  (f32, BN floats)
    //   shmem[BM*BK+BN .. ...+BN] = b_tile  (f32, BN floats)
    //   tail: q_tile (u8, BN*BK bytes) — packed after the float regions
    threadgroup float *x_tile = shmem;
    threadgroup float *s_tile = shmem + BM * BK;
    threadgroup float *b_tile = s_tile + BN;
    threadgroup uchar *q_tile = (threadgroup uchar *)(b_tile + BN);

    const uint m = bid.x * BM + tid.x;
    const uint n = bid.y * BN + tid.y;
    const bool m_in_bounds = (m < M);
    const bool n_in_bounds = (n < N);

    const uint groups_per_row = K / BK;     // = K / group_size

    // Linear thread index in the 16x16 TG (0..255).
    const uint tlin = tid.x * BN + tid.y;

    float acc = 0.0f;

    // Sweep K in BK-sized tiles.
    for (uint kt = 0; kt < groups_per_row; kt++) {
        const uint k_base = kt * BK;

        // ---- Cooperative loads ----
        // X tile: [BM][BK] = 512 floats; 256 threads → 2 floats each.
        //   thread tlin loads positions tlin and tlin + 256 in row-major
        //   linear index over the tile.
        for (uint slot = 0; slot < 2; slot++) {
            const uint lin_idx = slot * 256 + tlin;
            const uint row = lin_idx / BK;       // 0..15
            const uint col = lin_idx % BK;       // 0..31
            const uint mm = bid.x * BM + row;
            const float v = (mm < M && (k_base + col) < K)
                ? x[mm * K + k_base + col]
                : 0.0f;
            x_tile[lin_idx] = v;
        }

        // q_int tile: [BN][BK] = 512 bytes; 256 threads → 2 bytes each.
        for (uint slot = 0; slot < 2; slot++) {
            const uint lin_idx = slot * 256 + tlin;
            const uint row = lin_idx / BK;       // 0..15
            const uint col = lin_idx % BK;       // 0..31
            const uint nn = bid.y * BN + row;
            const uchar q = (nn < N && (k_base + col) < K)
                ? q_int[nn * K + k_base + col]
                : (uchar)0;
            q_tile[lin_idx] = q;
        }

        // s/b tiles: BN floats each; first 16 threads in tlin order load.
        if (tlin < BN) {
            const uint nn = bid.y * BN + tlin;
            const uint sb_idx = nn * groups_per_row + kt;
            s_tile[tlin] = (nn < N) ? scales[sb_idx] : 0.0f;
            b_tile[tlin] = (nn < N) ? biases[sb_idx] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Inner reduction across BK ----
        if (m_in_bounds && n_in_bounds) {
            const float s = s_tile[tid.y];
            const float b = b_tile[tid.y];
            // Shared X row for this thread (depends only on tid.x).
            threadgroup float *x_row = x_tile + tid.x * BK;
            // Shared q row for this thread (depends only on tid.y).
            threadgroup uchar *q_row = q_tile + tid.y * BK;

            for (uint k = 0; k < BK; k++) {
                const float xv = x_row[k];
                const float qv = float(q_row[k]);
                const float wv = qv * s + b;
                acc += xv * wv;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m_in_bounds && n_in_bounds) {
        y[m * N + n] = acc;
    }
}
