#include <metal_stdlib>
using namespace metal;

/// ADR-020 iter-15 — fused affine quantized matmul for DWQ inference.
///
/// Computes `Y[m, n] = Σ_k X[m, k] · dequant(q_int[n, k], scales[n,
/// g(k)], biases[n, g(k)])` where `dequant(q, s, b) = q · s + b` and
/// `g(k) = k / group_size`.  Matches the canonical mlx affine
/// dequantization formula at `mlx/backend/metal/kernels/quantized.h:521-526`
/// (`s[0] * (b & 0x0f) + bias`), but operates on UNPACKED uint8
/// codes (one byte per nibble) for layout compatibility with
/// iter-13b's `qdq_affine` kernels.  A packed-uint8 variant (2
/// nibbles per byte, mlx's on-disk convention) is deferred to
/// iter-15b once the unpacked variant is proven correct.
///
/// Layout (matches iter-13b's `qdq_affine_forward_f32`):
///   - `x`      : f32[M, K]                                row-major
///   - `q_int`  : u8 [N, K]                                row-major; q ∈ [0, 2^bits - 1]
///   - `scales` : f32[N · K/group_size]                    row-major (n_outer, n_groups)
///   - `biases` : f32[N · K/group_size]                    row-major (same layout)
///   - `y`      : f32[M, N]                                row-major
///
/// Constraints (validated host-side):
///   - `K` must be divisible by `group_size`.
///   - `group_size` must be a power of two in [2, 1024].
///   - `bits` baked into the kernel; first cut supports 4 + 8 (others
///     deferred — same dispatch code, separate kernel entries).  This
///     SHADER doesn't bound q values: the host promises `q_int[i] <
///     2^bits` via the iter-13b init kernel; the kernel just casts to
///     float.
///
/// Threading: one thread per output element `(m, n)`.  Grid = (M, N,
/// 1).  Threadgroup = up to (16, 16, 1) — caller picks based on
/// device limits.  No threadgroup-shared memory; reduction is
/// per-thread accumulator.
///
/// Performance note: this is a CORRECTNESS-FIRST iter-15 kernel.
/// A tiled + simdgroup-MMA variant matching mlx's `affine_qmm_t`
/// (BM=BK=BN=32, WM=WN=2) lands in iter-15b.

kernel void qmm_affine_t_f32(
    device const float    *x          [[buffer(0)]],
    device const uchar    *q_int      [[buffer(1)]],
    device const float    *scales     [[buffer(2)]],
    device const float    *biases     [[buffer(3)]],
    device float          *y          [[buffer(4)]],
    device const uint     *meta       [[buffer(5)]],  // [M, N, K, group_size]
    uint2 gid [[thread_position_in_grid]]
) {
    const uint M          = meta[0];
    const uint N          = meta[1];
    const uint K          = meta[2];
    const uint group_size = meta[3];

    const uint m = gid.x;
    const uint n = gid.y;
    if (m >= M || n >= N) { return; }

    const uint groups_per_row = K / group_size;
    const uint q_row_base = n * K;
    const uint x_row_base = m * K;
    const uint sb_row_base = n * groups_per_row;

    float acc = 0.0f;
    // Inner loop unrolled by group_size so we can hoist the (scale,
    // bias) load to once per group.  For each group of `group_size`
    // contiguous K elements, all (q_int, x) reads share the same
    // (scale, bias) pair.
    for (uint g = 0; g < groups_per_row; g++) {
        const uint k0 = g * group_size;
        const float s = scales[sb_row_base + g];
        const float b = biases[sb_row_base + g];
        // Tight inner loop over the group's elements.  The dequant
        // formula `q*s + b` matches mlx's canonical affine layout
        // (the "/16 absorbed scale" trick in mlx's bits=4 path is
        // a packed-byte-specific optimization that doesn't apply
        // here — we have one byte per code).
        for (uint i = 0; i < group_size; i++) {
            const uint k = k0 + i;
            const float q = float(q_int[q_row_base + k]);
            const float w_dq = q * s + b;
            acc += x[x_row_base + k] * w_dq;
        }
    }

    y[m * N + n] = acc;
}
