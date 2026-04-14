// dense_gemm.metal — Dense F16 matrix multiply: C = A * B^T
//
// Two kernels:
//
// 1. `dense_matvec_f16`  — Specialized M=1 mat-vec (decode path).
//    Modeled after llama.cpp's kernel_mul_mv_f16_f32 pattern.
//    Each SIMD group (32 threads) processes N_DST rows using vectorized
//    half4 loads and simd_sum reduction.
//
// 2. `dense_gemm_f16`    — Fallback tiled GEMM for M>1.
//    Uses shared memory tiling with simdgroup_matrix MMA (8x8 hardware
//    accumulator) for high throughput.
//
// Buffer layout (both kernels):
//   buffer(0): A      — half [M, K]
//   buffer(1): B      — half [N, K]
//   buffer(2): C      — half [M, N]  (output)
//   buffer(3): params — DenseGemmParams {M, N, K}

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct DenseGemmParams {
    uint M;
    uint N;
    uint K;
};

// ============================================================
// Kernel 1: Specialized M=1 mat-vec — the hot path for lm_head decode
// ============================================================
//
// Architecture (matches llama.cpp pattern):
//   - N_DST=4 rows per SIMD group, N_SIMDGROUP=2 per threadgroup -> 8 rows/tg
//   - Each thread in a simdgroup handles K/(32*4) vectorized loads (half4)
//   - simd_sum reduces across 32 lanes
//   - Dispatch: threadgroups=(ceil(N/8), 1, 1), threads_per_tg=(32, 2, 1)
//     (32 lanes x 2 simdgroups)
//
// For lm_head [1, 2816] x [262144, 2816]^T:
//   32768 threadgroups, 64 threads each, each tg outputs 8 values.

constant uint MV_N_DST       = 4;   // rows per SIMD group
constant uint MV_N_SIMDWIDTH = 32;  // Apple GPU SIMD width

kernel void dense_matvec_f16(
    device const half*           A      [[buffer(0)]],
    device const half*           B      [[buffer(1)]],
    device half*                 C      [[buffer(2)]],
    constant DenseGemmParams&    params [[buffer(3)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    sgitg [[simdgroup_index_in_threadgroup]],
    uint    tiisg [[thread_index_in_simdgroup]]
) {
    const uint K = params.K;
    const uint N = params.N;

    // Which row block this threadgroup handles
    // N_SIMDGROUP = 2 (from dispatch: threads_per_tg.y = 2)
    const uint row_base = tgpig.x * (MV_N_DST * 2);
    // Which rows this simdgroup handles
    const uint row0 = row_base + sgitg * MV_N_DST;

    // Early exit if all rows are out of bounds
    if (row0 >= N) return;

    // The single input row (M=1)
    device const half* a_ptr = A;

    // Accumulators for N_DST rows (float for precision)
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // Pointers to weight rows
    device const half* b0 = B + (row0 + 0) * K;
    device const half* b1 = B + min(row0 + 1, N - 1) * K;
    device const half* b2 = B + min(row0 + 2, N - 1) * K;
    device const half* b3 = B + min(row0 + 3, N - 1) * K;

    // Main K-loop: vectorized half4 loads
    // Each of 32 threads handles every 32nd group of 4 elements
    const uint k_stride = MV_N_SIMDWIDTH * 4; // 128
    uint k = tiisg * 4;

    // Aligned loop (K divisible by k_stride handled automatically)
    for (; k + 3 < K; k += k_stride) {
        half4 av = *((device const half4*)(a_ptr + k));
        float4 afv = float4(av);

        half4 bv0 = *((device const half4*)(b0 + k));
        acc0 += dot(afv, float4(bv0));

        if (row0 + 1 < N) {
            half4 bv1 = *((device const half4*)(b1 + k));
            acc1 += dot(afv, float4(bv1));
        }
        if (row0 + 2 < N) {
            half4 bv2 = *((device const half4*)(b2 + k));
            acc2 += dot(afv, float4(bv2));
        }
        if (row0 + 3 < N) {
            half4 bv3 = *((device const half4*)(b3 + k));
            acc3 += dot(afv, float4(bv3));
        }
    }

    // Scalar remainder — at most 3 elements per thread can remain,
    // but since we stepped by k_stride (128) the remainder is at most
    // K % 128 elements spread across 32 threads. Each thread handles
    // at most ceil(remainder/32) elements.
    for (; k < K; k += MV_N_SIMDWIDTH) {
        if (k < K) {
            float av = float(a_ptr[k]);
            acc0 += av * float(b0[k]);
            if (row0 + 1 < N) acc1 += av * float(b1[k]);
            if (row0 + 2 < N) acc2 += av * float(b2[k]);
            if (row0 + 3 < N) acc3 += av * float(b3[k]);
        }
    }

    // SIMD reduction across 32 lanes
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    // Lane 0 writes results
    if (tiisg == 0) {
        C[row0 + 0] = half(acc0);
        if (row0 + 1 < N) C[row0 + 1] = half(acc1);
        if (row0 + 2 < N) C[row0 + 2] = half(acc2);
        if (row0 + 3 < N) C[row0 + 3] = half(acc3);
    }
}

// ============================================================
// Kernel 1b: Mixed-precision mat-vec — F16 weights × F32 input → F32 output
// ============================================================
//
// Identical to dense_matvec_f16 but:
//   - A (input)  is float* instead of half*
//   - C (output) is float* instead of half*
//   - B (weights) remains half*
// Eliminates the F32→F16 cast on input and F16→F32 cast on output.

kernel void dense_matvec_f16w_f32io(
    device const float*          A      [[buffer(0)]],
    device const half*           B      [[buffer(1)]],
    device float*                C      [[buffer(2)]],
    constant DenseGemmParams&    params [[buffer(3)]],
    uint3   tgpig [[threadgroup_position_in_grid]],
    uint    sgitg [[simdgroup_index_in_threadgroup]],
    uint    tiisg [[thread_index_in_simdgroup]]
) {
    const uint K = params.K;
    const uint N = params.N;

    const uint row_base = tgpig.x * (MV_N_DST * 2);
    const uint row0 = row_base + sgitg * MV_N_DST;

    if (row0 >= N) return;

    device const float* a_ptr = A;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    device const half* b0 = B + (row0 + 0) * K;
    device const half* b1 = B + min(row0 + 1, N - 1) * K;
    device const half* b2 = B + min(row0 + 2, N - 1) * K;
    device const half* b3 = B + min(row0 + 3, N - 1) * K;

    // Main K-loop: vectorized float4 loads for A, half4 for B weights
    const uint k_stride = MV_N_SIMDWIDTH * 4; // 128
    uint k = tiisg * 4;

    for (; k + 3 < K; k += k_stride) {
        float4 afv = *((device const float4*)(a_ptr + k));

        half4 bv0 = *((device const half4*)(b0 + k));
        acc0 += dot(afv, float4(bv0));

        if (row0 + 1 < N) {
            half4 bv1 = *((device const half4*)(b1 + k));
            acc1 += dot(afv, float4(bv1));
        }
        if (row0 + 2 < N) {
            half4 bv2 = *((device const half4*)(b2 + k));
            acc2 += dot(afv, float4(bv2));
        }
        if (row0 + 3 < N) {
            half4 bv3 = *((device const half4*)(b3 + k));
            acc3 += dot(afv, float4(bv3));
        }
    }

    // Scalar remainder
    for (; k < K; k += MV_N_SIMDWIDTH) {
        if (k < K) {
            float av = a_ptr[k];
            acc0 += av * float(b0[k]);
            if (row0 + 1 < N) acc1 += av * float(b1[k]);
            if (row0 + 2 < N) acc2 += av * float(b2[k]);
            if (row0 + 3 < N) acc3 += av * float(b3[k]);
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (tiisg == 0) {
        C[row0 + 0] = acc0;
        if (row0 + 1 < N) C[row0 + 1] = acc1;
        if (row0 + 2 < N) C[row0 + 2] = acc2;
        if (row0 + 3 < N) C[row0 + 3] = acc3;
    }
}

// ============================================================
// Kernel 2: Fallback tiled GEMM for M>1
// ============================================================
//
// Uses simdgroup_matrix for hardware-accelerated 8x8 MMA.
// 32x32 output tile, BK=16, 4 simdgroups (WM=2, WN=2).
//
// This is a simplified port of the MLX Steel GEMM for C = A * B^T.
// Attribution: Algorithm based on MLX (Apache-2.0), see
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm

constant uint BM = 32;
constant uint BN = 32;
constant uint BK = 16;
constant uint WM = 2;   // simdgroups along M
constant uint WN = 2;   // simdgroups along N
constant uint TGP_SIZE = WM * WN * 32; // 128 threads

// Simdgroup matrix tile strides
constant short GM_TM_stride = 8 * WM; // = 16
constant short GM_TN_stride = 8 * WN; // = 16

// Threadgroup memory sizes with padding to avoid bank conflicts
constant uint TGP_PAD = 8; // 16 / sizeof(half)
constant uint LDA_TGP = BK + TGP_PAD;  // A: [BM, BK+pad]  (A is not transposed)
constant uint LDB_TGP = BK + TGP_PAD;  // B^T: [BN, BK+pad] (B is transposed)
constant uint TGP_MEM_A = BM * LDA_TGP;
constant uint TGP_MEM_B = BN * LDB_TGP;

// Number of simdgroup matrices per simdgroup sub-tile
constant uint GM_TM = 2; // 16/8 along M
constant uint GM_TN = 2; // 16/8 along N

kernel void dense_gemm_f16(
    device const half*           A       [[buffer(0)]],
    device const half*           B       [[buffer(1)]],
    device half*                 C       [[buffer(2)]],
    constant DenseGemmParams&    params  [[buffer(3)]],
    uint2   group_pos  [[threadgroup_position_in_grid]],
    uint    sgid       [[simdgroup_index_in_threadgroup]],
    uint    slid       [[thread_index_in_simdgroup]],
    uint    tid_in_tg  [[thread_index_in_threadgroup]]
) {
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    // Block position in output
    const uint c_row = group_pos.y * BM;
    const uint c_col = group_pos.x * BN;

    // Early exit for out-of-bounds threadgroups
    if (c_row >= M && c_col >= N) return;

    // Threadgroup shared memory
    threadgroup half As[TGP_MEM_A];
    threadgroup half Bs[TGP_MEM_B];

    // Advance A and B pointers to this block's starting position
    device const half* A_block = A + c_row * K;
    device const half* B_block = B + c_col * K;

    // Simdgroup position within the WM x WN grid
    const uint wm_id = sgid / WN;  // 0 or 1
    const uint wn_id = sgid % WN;  // 0 or 1

    // Determine thread position within simdgroup matrix layout
    short qid = slid / 4;
    short sm = (qid & 4) + (slid / 2) % 4;
    short sn = (qid & 2) * 2 + (slid % 2) * 2;

    // Simdgroup tile offsets
    short tm = 8 * wm_id;
    short tn = 8 * wn_id;

    // Initialize result accumulators
    simdgroup_matrix<float, 8, 8> results[GM_TM * GM_TN] = {
        simdgroup_matrix<float, 8, 8>(0),
        simdgroup_matrix<float, 8, 8>(0),
        simdgroup_matrix<float, 8, 8>(0),
        simdgroup_matrix<float, 8, 8>(0)
    };

    // Cooperative tile loading constants
    const uint a_loads_per_thread = (BM * BK) / TGP_SIZE; // 4
    const uint b_loads_per_thread = (BN * BK) / TGP_SIZE; // 4

    const uint k_iterations = (K + BK - 1) / BK;

    for (uint k_iter = 0; k_iter < k_iterations; k_iter++) {
        const uint k_off = k_iter * BK;
        const uint k_remain = min(BK, K - k_off);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tile [BM, BK] into shared memory
        for (uint i = 0; i < a_loads_per_thread; i++) {
            uint flat_idx = tid_in_tg + i * TGP_SIZE;
            uint local_row = flat_idx / BK;
            uint local_col = flat_idx % BK;

            half val = 0;
            if (c_row + local_row < M && local_col < k_remain) {
                val = A_block[local_row * K + k_off + local_col];
            }
            As[local_row * LDA_TGP + local_col] = val;
        }

        // Load B tile [BN, BK] into shared memory
        for (uint i = 0; i < b_loads_per_thread; i++) {
            uint flat_idx = tid_in_tg + i * TGP_SIZE;
            uint local_row = flat_idx / BK;
            uint local_col = flat_idx % BK;

            half val = 0;
            if (c_col + local_row < N && local_col < k_remain) {
                val = B_block[local_row * K + k_off + local_col];
            }
            Bs[local_row * LDB_TGP + local_col] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA: iterate over BK in steps of 8
        simdgroup_matrix<float, 8, 8> Asimd[GM_TM];
        simdgroup_matrix<float, 8, 8> Bsimd[GM_TN];

        // A offsets: not transposed
        short As_off = sn + (tm + sm) * (short)LDA_TGP;
        // B offsets: transposed
        short Bs_off = (tn + sn) * (short)LDB_TGP + sm;

        threadgroup const half* As_ptr = As + As_off;
        threadgroup const half* Bs_ptr = Bs + Bs_off;

        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);

            // Load A simdgroup matrices
            for (uint i = 0; i < GM_TM; i++) {
                Asimd[i].thread_elements()[0] =
                    float(As_ptr[i * GM_TM_stride * (short)LDA_TGP + 0]);
                Asimd[i].thread_elements()[1] =
                    float(As_ptr[i * GM_TM_stride * (short)LDA_TGP + 1]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // Load B simdgroup matrices
            for (uint j = 0; j < GM_TN; j++) {
                Bsimd[j].thread_elements()[0] =
                    float(Bs_ptr[j * GM_TN_stride * (short)LDB_TGP + 0]);
                Bsimd[j].thread_elements()[1] =
                    float(Bs_ptr[j * GM_TN_stride * (short)LDB_TGP + (short)LDB_TGP]);
            }

            simdgroup_barrier(mem_flags::mem_none);

            // MMA with serpentine access pattern
            for (uint i = 0; i < GM_TM; i++) {
                for (uint j = 0; j < GM_TN; j++) {
                    short j_serp = (i % 2) ? (GM_TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * GM_TN + j_serp],
                        Asimd[i],
                        Bsimd[j_serp],
                        results[i * GM_TN + j_serp]);
                }
            }

            As_ptr += 8;
            Bs_ptr += 8;
        }
    }

    // Store results
    short tgp_bm = min((short)BM, (short)(M - c_row));
    short tgp_bn = min((short)BN, (short)(N - c_col));

    device half* D = C + (uint)(sm + tm + c_row) * N + (uint)(tn + sn + c_col);

    for (uint i = 0; i < GM_TM; i++) {
        for (uint j = 0; j < GM_TN; j++) {
            thread const auto& accum = results[i * GM_TN + j].thread_elements();
            int offset = (int)(i * GM_TM_stride) * (int)N + (int)(j * GM_TN_stride);

            short out_row = sm + tm + (short)(i * GM_TM_stride);
            short out_col = sn + tn + (short)(j * GM_TN_stride);

            if (out_row < tgp_bm && out_col < tgp_bn) {
                D[offset] = half(accum[0]);
            }
            if (out_row < tgp_bm && out_col + 1 < tgp_bn) {
                D[offset + 1] = half(accum[1]);
            }
        }
    }
}
