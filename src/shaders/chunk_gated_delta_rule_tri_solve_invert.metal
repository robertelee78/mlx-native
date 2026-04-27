#include <metal_stdlib>
using namespace metal;

// Wave 5b.1 iter 4 — per-chunk-block tri-solve invert kernel.
//
// Computes A_inv = (I + A_strict_lower)^-1 per [BT, BT] chunk-block on the
// FLA-native A layout [B, T, H, BT] (BT innermost, row-stride H*BT).
//
// Spec source: FLA solve_tril semantics at
//   /opt/vllm/vllm/model_executor/layers/fla/ops/solve_tril.py:506-530.
//
// FLA's solve_tril returns `(I + A_strict_lower)^-1`. The mlx-native
// tri_solve.metal shader solves L · X = B with implicit unit diagonal,
// but its memory layout (row-stride N, batch-major) does NOT match FLA's
// [B, T, H, BT] layout (row-stride H*BT). This kernel does the per-block
// invert directly on the FLA-native layout to avoid a transpose pass.
//
// # Algorithm
//
// For each (b, i_t, i_h) chunk-block, with t_start = i_t*BT:
//
//   # In shared memory, stage L = A_strict[b, t_start..t_start+BT, i_h, :]
//   # which has BT rows × BT cols, strict-lower (zero diagonal).
//
//   # X = (I + L)^-1, computed via forward substitution with B = I:
//   #   X[i, j] = (i == j ? 1 : 0) - sum_{k<i} L[i, k] * X[k, j]
//   #
//   # Each thread owns column j ∈ [0, BT) and walks rows i = 0..BT-1
//   # serially. Within a thread, X[k, j] for k<i is already written by
//   # the same thread in an earlier iteration — no cross-thread sync.
//
// # Memory layouts
//
//   A_strict (input)  : [B, T, H, BT] f32 — row-stride H*BT
//   A_inv    (output) : [B, T, H, BT] f32 — same layout
//
//   For block (b, i_t, i_h):
//     A_strict[b, t_start+i, i_h, j]
//       = A_strict_base[((b * T + t_start + i) * H + i_h) * BT + j]
//   We load each row contiguously (BT consecutive cols) into shared mem.
//
// # Threading model
//
//   Grid: (NT, H, B) — one threadgroup per (i_t, i_h, b).
//   Threadgroup size: BT threads (= 64 for iter 4). Each thread owns
//   column j of the [BT, BT] X tile.
//
// # Threadgroup memory
//
//   l_tile : BT * BT * 4 bytes (f32) = 64 * 64 * 4 = 16 KB
//   x_tile : BT * BT * 4 bytes (f32) = 64 * 64 * 4 = 16 KB
//   Total: 32 KB. M5 Max threadgroup-memory cap: 32 KB. Tight. (We could
//   compute X in-place in l_tile, dropping x_tile, but staging both is
//   simpler and 32 KB fits exactly.)
//
//   Iter 4 alternative — write X[i, j] back to global directly each row,
//   sacrificing 16 KB threadgroup memory. We use the in-shared variant
//   below to keep the inner loop tight; if iter 5 perf finds threadgroup
//   memory pressure, we drop x_tile.
//
// # Buffer layout
//
//   buffer(0): A_strict [B, T, H, BT] f32   (input)
//   buffer(1): A_inv    [B, T, H, BT] f32   (output)
//   buffer(2): params: uint4 = (B, T, H, BT)

constant constexpr uint MAX_BT = 64;  // iter-4 fixed

kernel void chunk_tri_solve_invert_f32(
    device const float *A_strict [[buffer(0)]],
    device       float *A_inv    [[buffer(1)]],
    device const uint  *params   [[buffer(2)]],
    threadgroup float  *shmem    [[threadgroup(0)]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]],
    uint3 t_pos                  [[thread_position_in_threadgroup]]
) {
    const uint B  = params[0];
    const uint T  = params[1];
    const uint H  = params[2];
    const uint BT = params[3];

    const uint i_t = tg_pos.x;       // chunk index   in [0, NT)
    const uint i_h = tg_pos.y;       // V-head index  in [0, H)
    const uint b   = tg_pos.z;       // batch index   in [0, B)
    const uint NT  = T / BT;
    if (i_t >= NT || i_h >= H || b >= B) {
        return;
    }

    const uint j   = t_pos.x;        // column owned by this thread
    if (j >= BT) {
        return;
    }

    // Shared-memory tiles (carved from a single shmem buffer).
    threadgroup float *l_tile = shmem;                   // [BT, BT]
    threadgroup float *x_tile = shmem + (BT * BT);       // [BT, BT]

    // Cooperative load of L: each thread loads BT entries — one full
    // column j across all rows i. (The kernel always uses BT threads, so
    // BT × BT cells are covered.)
    const uint t_start = i_t * BT;
    for (uint i = 0; i < BT; ++i) {
        const uint global_idx =
            ((b * T + t_start + i) * H + i_h) * BT + j;
        l_tile[i * BT + j] = A_strict[global_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward substitution: X[i, j] = (i == j) - sum_{k<i} L[i, k] * X[k, j].
    // Each thread owns column j and walks rows in order; thread-local
    // dependency only — no cross-thread sync needed.
    for (uint i = 0; i < BT; ++i) {
        float acc = (i == j) ? 1.0f : 0.0f;
        for (uint k = 0; k < i; ++k) {
            // L[i, k] is in row i of l_tile; X[k, j] is the value this same
            // thread (column j) already wrote at iteration k.
            acc -= l_tile[i * BT + k] * x_tile[k * BT + j];
        }
        x_tile[i * BT + j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative store: each thread writes its column back.
    for (uint i = 0; i < BT; ++i) {
        const uint global_idx =
            ((b * T + t_start + i) * H + i_h) * BT + j;
        A_inv[global_idx] = x_tile[i * BT + j];
    }
}
