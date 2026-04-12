#include <metal_stdlib>
using namespace metal;

/// Dense F16 matrix multiply: C = A * B^T
///
/// A is [M, K] half, B is [N, K] half, C is [M, N] half.
///
/// Simple tiled implementation — correctness first.
/// Each threadgroup computes a TILE_M x TILE_N block of the output.
///
/// Buffer layout:
///   buffer(0): A      — half [M, K]
///   buffer(1): B      — half [N, K]
///   buffer(2): C      — half [M, N]
///   buffer(3): params — uint [3] — {M, N, K}
///
/// Grid:        ceil(N/TILE_N) * ceil(M/TILE_M) threadgroups
/// Threadgroup: (TILE_N, TILE_M, 1)

struct DenseGemmParams {
    uint M;
    uint N;
    uint K;
};

constant uint TILE_M = 8;
constant uint TILE_N = 8;
constant uint TILE_K = 8;

kernel void dense_gemm_f16(
    device const half*       A      [[buffer(0)]],
    device const half*       B      [[buffer(1)]],
    device half*             C      [[buffer(2)]],
    constant DenseGemmParams& params [[buffer(3)]],
    uint2 group_pos [[threadgroup_position_in_grid]],
    uint2 local_pos [[thread_position_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint row = group_pos.y * TILE_M + local_pos.y;
    const uint col = group_pos.x * TILE_N + local_pos.x;

    if (row >= M || col >= N) return;

    // Accumulate in float for numerical stability.
    float acc = 0.0f;

    // Main loop over K dimension.
    for (uint k = 0; k < K; k += 1) {
        // C = A * B^T, so B is [N, K]: B[col, k] = B[col * K + k]
        acc += float(A[row * K + k]) * float(B[col * K + k]);
    }

    C[row * N + col] = half(acc);
}
