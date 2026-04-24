#include <metal_stdlib>
using namespace metal;

// Lower-triangular unit-diagonal solve: X = L \ B, where L is N×N, B is N×M.
//
// Spec source: ADR-013 Decision 5. Derived from the definition of forward
// substitution on a lower-triangular unit-diagonal system:
//
//   L · X = B   with L[i][i] = 1 (implicit), L[i][j] = 0 for j > i.
//
//   x[0, :]   = b[0, :]
//   x[i, :]   = b[i, :] - sum_{j=0..i-1} L[i, j] * x[j, :]    for i = 1..N-1
//
// The diagonal is UNIT (not read); only the strict lower triangle of L is
// consulted. Upper-triangle values are ignored.
//
// # Batching
//
// Batched over a single leading dim `batch`. The caller folds any additional
// leading dims into `batch`.
//
// # Memory layout (column-major, innermost-first)
//
//   L[i, j, b] at b * N*N + i * N + j      (row i stride N, each row contiguous)
//   B[i, m, b] at b * N*M + i * M + m      (rhs column m contiguous per row)
//   X[i, m, b] at b * N*M + i * M + m      (same layout as B)
//
// This layout makes row-i slices of L contiguous (for the inner-j dot product)
// and puts all M right-hand-side columns for row i adjacent (for the outer
// per-m parallel loop).
//
// # Parallelism
//
// One thread per (m, b) pair. Each thread walks rows 0..N-1 serially,
// accumulating the forward-substitution sum in f32. X is written in-place
// safe (the thread only reads previously-written rows j < i).
//
// # Buffer layout
//
//   buffer(0): L       — shape [batch, N, N]
//   buffer(1): B       — shape [batch, N, M]
//   buffer(2): X       — shape [batch, N, M] (output)
//   buffer(3): params  — uint3: (N, M, batch)
//
// Grid: (M, batch, 1) threads total. Threadgroup: (min(256, M), 1, 1).

kernel void tri_solve_lower_unit_f32(
    device const float *L       [[buffer(0)]],
    device const float *B       [[buffer(1)]],
    device float       *X       [[buffer(2)]],
    device const uint  *params  [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint n     = params[0];
    const uint m     = params[1];
    const uint batch = params[2];

    const uint col = tid.x;
    const uint b   = tid.y;
    if (col >= m || b >= batch) {
        return;
    }

    const uint l_batch_stride = n * n;
    const uint bx_batch_stride = n * m;

    // Row 0: unit-diagonal so x[0, col, b] = b[0, col, b] directly.
    // Then forward-substitute row by row.
    for (uint i = 0; i < n; ++i) {
        float acc = float(B[b * bx_batch_stride + i * m + col]);
        for (uint j = 0; j < i; ++j) {
            const float l_ij = L[b * l_batch_stride + i * n + j];
            const float x_j  = X[b * bx_batch_stride + j * m + col];
            acc -= l_ij * x_j;
        }
        X[b * bx_batch_stride + i * m + col] = acc;
    }
}

kernel void tri_solve_lower_unit_bf16(
    device const bfloat *L      [[buffer(0)]],
    device const bfloat *B      [[buffer(1)]],
    device bfloat       *X      [[buffer(2)]],
    device const uint   *params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint n     = params[0];
    const uint m     = params[1];
    const uint batch = params[2];

    const uint col = tid.x;
    const uint b   = tid.y;
    if (col >= m || b >= batch) {
        return;
    }

    const uint l_batch_stride = n * n;
    const uint bx_batch_stride = n * m;

    for (uint i = 0; i < n; ++i) {
        float acc = float(B[b * bx_batch_stride + i * m + col]);
        for (uint j = 0; j < i; ++j) {
            const float l_ij = float(L[b * l_batch_stride + i * n + j]);
            const float x_j  = float(X[b * bx_batch_stride + j * m + col]);
            acc -= l_ij * x_j;
        }
        X[b * bx_batch_stride + i * m + col] = bfloat(acc);
    }
}
