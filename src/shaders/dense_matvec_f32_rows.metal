// dense_matvec_f32_rows.metal — row-exact F32 mat-vec over a few input rows.
//
//   Y[r, n] = sum_k X[r, k] * W[n, k]      r in 0..ROWS, n in 0..N
//
// X [ROWS, K], W [N, K] (row-major), Y [ROWS, N], all F32. One dispatch reads
// each weight row once for every input row (the peer's F32 mat-vec with
// N = rows technique). ROWS is a function constant (1..8).
//
// Geometry: a threadgroup of NSG = 4 SIMD groups computes NR0 = 2 weight
// rows. K is cut into 128-value steps (32 lanes x float4); SIMD group g owns
// steps g, g + 4, g + 8, ... Each lane accumulates its float4 of every owned
// step, in increasing step order, with explicit fma in x, y, z, w order, one
// accumulator per (weight row, input row). The four per-group partials of a
// lane are then added in group order through threadgroup memory and the 32
// lanes reduced with simd_sum.
//
// Row exactness: the accumulation of output (r, n) reads only X[r] and W[n]
// and its order does not depend on ROWS, on which other rows share the
// dispatch, or on N. A row's result is therefore bitwise the same at
// ROWS = 1 and in a batch. (Not bitwise equal to dense_matvec_f32, which
// uses dot() and a different K split.)
//
// Host requirements: K % 4 == 0 (float4 rows, 16-byte aligned row starts),
// 1 <= ROWS <= 8.

#include <metal_stdlib>
using namespace metal;

constant int FC_ROWS [[function_constant(0)]];  // input rows, 1..8

struct DenseMatvecF32RowsParams {
    uint n;
    uint k;
    uint _pad0;
    uint _pad1;
};

constant constexpr short NSG = 4;
constant constexpr short NR0 = 2;
constant constexpr short NW = 32;
constant constexpr short MAX_ROWS = 8;

kernel void dense_matvec_f32_rows(
    device const float *               W      [[buffer(0)]],
    device const float *               X      [[buffer(1)]],
    device       float *               Y      [[buffer(2)]],
    constant DenseMatvecF32RowsParams & p     [[buffer(3)]],
    threadgroup float *                shmem  [[threadgroup(0)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    ushort  lane  [[thread_index_in_simdgroup]],
    ushort  sg    [[simdgroup_index_in_threadgroup]])
{
    const uint N = p.n;
    const uint K = p.k;
    const uint n0 = tgpig.x * NR0;

    float acc[NR0][MAX_ROWS];
#pragma unroll
    for (short i = 0; i < NR0; ++i) {
#pragma unroll
        for (short r = 0; r < FC_ROWS; ++r) {
            acc[i][r] = 0.0f;
        }
    }

    const uint steps = (K + NW * 4 - 1) / (NW * 4);
    for (uint step = sg; step < steps; step += NSG) {
        const uint k = step * (NW * 4) + uint(lane) * 4;
        if (k < K) {
            float4 xv[MAX_ROWS];
#pragma unroll
            for (short r = 0; r < FC_ROWS; ++r) {
                xv[r] = *((device const float4 *)(X + (ulong)r * K + k));
            }
#pragma unroll
            for (short i = 0; i < NR0; ++i) {
                const uint n = min(n0 + uint(i), N - 1);
                const float4 wv = *((device const float4 *)(W + (ulong)n * K + k));
#pragma unroll
                for (short r = 0; r < FC_ROWS; ++r) {
                    float a = acc[i][r];
                    a = fma(xv[r].x, wv.x, a);
                    a = fma(xv[r].y, wv.y, a);
                    a = fma(xv[r].z, wv.z, a);
                    a = fma(xv[r].w, wv.w, a);
                    acc[i][r] = a;
                }
            }
        }
    }

    // shmem layout: [NR0 * ROWS pairs][NSG groups][NW lanes]
#pragma unroll
    for (short i = 0; i < NR0; ++i) {
#pragma unroll
        for (short r = 0; r < FC_ROWS; ++r) {
            const int pair = int(i) * FC_ROWS + int(r);
            shmem[(pair * NSG + int(sg)) * NW + int(lane)] = acc[i][r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int pairs = NR0 * FC_ROWS;
    for (int pair = int(sg); pair < pairs; pair += NSG) {
        float s = shmem[(pair * NSG + 0) * NW + int(lane)];
        s += shmem[(pair * NSG + 1) * NW + int(lane)];
        s += shmem[(pair * NSG + 2) * NW + int(lane)];
        s += shmem[(pair * NSG + 3) * NW + int(lane)];
        s = simd_sum(s);
        const int i = pair / FC_ROWS;
        const int r = pair % FC_ROWS;
        const uint n = n0 + uint(i);
        if (lane == 0 && n < N) {
            Y[(ulong)r * N + n] = s;
        }
    }
}
