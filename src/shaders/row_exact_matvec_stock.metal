// Row-exact few-row mat-vec in the STOCK per-row order:
// y[j, n] = sum_k W[n, k] * x[j, k] for j < M <= 8.
//
// Every output row is bitwise equal to the stock one-row kernel hf2q uses
// today for that weight storage, and therefore to any other row count:
//
// - F16 (`row_exact_stock_mv_f16_f32_r<NR>_u<U>_m<M>`) replays
//   `dense_matvec_f16w_f32io` (dense_gemm.metal). Per row, the 32 lanes of one
//   simdgroup each run `acc += dot(float4 x[4t + 128 s], float4(half4 w))` for
//   s = 0, 1, ... while 4t + 128 s < K, then `simd_sum`. (The stock kernel's
//   scalar remainder never runs: the dispatch requires K % 4 == 0, so the
//   vector loop already covers every element; lanes may take different
//   step counts when K % 128 != 0.)
// - Q8_0 (`row_exact_stock_mv_q8_0_f32_r<NR0>_v<V>_m<M>`) replays
//   `kernel_mul_mv_q8_0_f32_nr2` (quantized_matmul_ggml.metal). Per row, a
//   team of 4 x 32 threads (virtual simdgroup s, lane l; ix = l / 4,
//   il = l % 4) visits blocks ib = 8 s + ix, + 32, ...; per block
//   `sumq += qs[i] * yl[i]` over 8 quants in order, then
//   `sumf += sumq * d`; `simd_sum` per virtual simdgroup; the final value is
//   `simd_sum` over [s0, s1, s2, s3, 0, ..., 0].
//
// The arithmetic is written as the same source expressions, in the same
// order, as the stock kernels. Only the work packing differs, and it does
// not touch any per-row sum:
// - each weight value is loaded once and used for all M input rows;
// - F16: NR rows per simdgroup, and U steps whose loads are issued before
//   their (in-order) accumulations;
// - Q8_0: NR0 rows per threadgroup, and V virtual simdgroups per physical
//   simdgroup (4 / V physical simdgroups per threadgroup).
// Both metallib paths compile this file with the stock files' options, and
// tests/row_exact_matvec_stock.rs checks every row against the stock kernel
// bitwise on each DeepSeek-V4 decode shape.
//
// Grid: (batch * tgs_per_batch, 1, 1), batch-major.

#include <metal_stdlib>
using namespace metal;

struct RowExactStockParams {
    uint  k;              // inner dimension (elements)
    uint  n;              // output columns = weight rows per batch
    uint  x_row_stride;   // f32 elements between input rows
    uint  x_batch_stride; // f32 elements between input batches
    uint  y_row_stride;   // f32 elements between output rows
    uint  y_batch_stride; // f32 elements between output batches
    uint  tgs_per_batch;  // threadgroups per batch
    uint  pad;
    ulong w_row_bytes;    // bytes between weight rows
    ulong w_batch_bytes;  // bytes between weight batches
};

// ---------------------------------------------------------------- F16 weights

template <short NR, short U, short M>
kernel void row_exact_stock_mv_f16_f32(
    device const char  * w     [[buffer(0)]],
    device const float * x     [[buffer(1)]],
    device       float * y     [[buffer(2)]],
    constant RowExactStockParams & p [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    uint   sgitg [[simdgroup_index_in_threadgroup]],
    uint   tiisg [[thread_index_in_simdgroup]],
    uint   nsg   [[simdgroups_per_threadgroup]]) {
    const uint K = p.k;
    const uint N = p.n;
    const uint b = tgpig.x / p.tgs_per_batch;
    const uint row0 = (tgpig.x % p.tgs_per_batch) * (NR * nsg) + sgitg * NR;
    if (row0 >= N) {
        return;
    }

    device const char * wb = w + (ulong)b * p.w_batch_bytes;
    device const half * bw[NR];
    for (short r = 0; r < NR; ++r) {
        bw[r] = (device const half *)(wb + (ulong)min(row0 + (uint)r, N - 1) * p.w_row_bytes);
    }
    device const float * a[M];
    for (short j = 0; j < M; ++j) {
        a[j] = x + (ulong)b * p.x_batch_stride + (ulong)j * p.x_row_stride;
    }

    float acc[NR][M];
    for (short r = 0; r < NR; ++r) {
        for (short j = 0; j < M; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    const uint k_stride = 32 * 4; // 128
    uint k = tiisg * 4;

    // U whole steps at a time: every load first, then the accumulations in
    // the stock step order.
    for (; k + (U - 1) * k_stride + 3 < K; k += U * k_stride) {
        float4 afv[U][M];
        half4 bv[U][NR];
        for (short u = 0; u < U; ++u) {
            for (short j = 0; j < M; ++j) {
                afv[u][j] = *((device const float4 *)(a[j] + k + u * k_stride));
            }
            for (short r = 0; r < NR; ++r) {
                bv[u][r] = *((device const half4 *)(bw[r] + k + u * k_stride));
            }
        }
        for (short u = 0; u < U; ++u) {
            for (short r = 0; r < NR; ++r) {
                for (short j = 0; j < M; ++j) {
                    acc[r][j] += dot(afv[u][j], float4(bv[u][r]));
                }
            }
        }
    }
    // Remaining whole steps, one at a time (the stock loop).
    for (; k + 3 < K; k += k_stride) {
        float4 afv[M];
        for (short j = 0; j < M; ++j) {
            afv[j] = *((device const float4 *)(a[j] + k));
        }
        for (short r = 0; r < NR; ++r) {
            half4 bv = *((device const half4 *)(bw[r] + k));
            for (short j = 0; j < M; ++j) {
                acc[r][j] += dot(afv[j], float4(bv));
            }
        }
    }

    for (short r = 0; r < NR; ++r) {
        for (short j = 0; j < M; ++j) {
            acc[r][j] = simd_sum(acc[r][j]);
        }
    }

    if (tiisg == 0) {
        device float * yb = y + (ulong)b * p.y_batch_stride;
        for (short r = 0; r < NR; ++r) {
            if (row0 + r < N) {
                for (short j = 0; j < M; ++j) {
                    yb[(ulong)j * p.y_row_stride + row0 + r] = acc[r][j];
                }
            }
        }
    }
}

// --------------------------------------------------------------- Q8_0 weights

#define ROW_EXACT_STOCK_QK8_0 32

typedef struct {
    half   d;
    int8_t qs[ROW_EXACT_STOCK_QK8_0];
} row_exact_stock_block_q8_0;
static_assert(sizeof(row_exact_stock_block_q8_0) == sizeof(half) + ROW_EXACT_STOCK_QK8_0,
              "wrong q8_0 block size");

template <short NR0, short V, short M>
kernel void row_exact_stock_mv_q8_0_f32(
    device const char  * w     [[buffer(0)]],
    device const float * x     [[buffer(1)]],
    device       float * y     [[buffer(2)]],
    constant RowExactStockParams & p [[buffer(3)]],
    threadgroup float  * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NSG = 4;  // stock N_SG_Q8_0: virtual simdgroups per row team
    constexpr int NW  = 32; // stock N_SIMDWIDTH
    constexpr int NQ  = 8;  // stock NQ_Q8_0

    const int nb = p.k / ROW_EXACT_STOCK_QK8_0;
    const uint b  = tgpig.x / p.tgs_per_batch;
    const int  r0 = (tgpig.x % p.tgs_per_batch) * NR0;
    const int  ne01 = p.n;

    device const char * wb = w + (ulong)b * p.w_batch_bytes;
    device const row_exact_stock_block_q8_0 * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const row_exact_stock_block_q8_0 *)(wb + (ulong)min(r0 + row, ne01 - 1) * p.w_row_bytes);
    }
    device const float * y_in[M];
    for (short j = 0; j < M; ++j) {
        y_in[j] = x + (ulong)b * p.x_batch_stride + (ulong)j * p.x_row_stride;
    }

    const int ix = tiisg / (NW / NQ);  // 0..7
    const int il = tiisg % (NW / NQ);  // 0..3

    float sumf[V][NR0][M];
    for (short v = 0; v < V; ++v) {
        for (int row = 0; row < NR0; ++row) {
            for (short j = 0; j < M; ++j) {
                sumf[v][row][j] = 0.f;
            }
        }
    }

    for (short v = 0; v < V; ++v) {
        const int vsg = (int)sgitg * V + v;
        const int ib0 = vsg * NQ + ix;
        for (int ib = ib0; ib < nb; ib += NSG * NQ) {
            float yl[M][NQ];
            for (short j = 0; j < M; ++j) {
                device const float * yb = y_in[j] + ib * ROW_EXACT_STOCK_QK8_0 + il * NQ;
                for (int i = 0; i < NQ; ++i) {
                    yl[j][i] = yb[i];
                }
            }
            for (int row = 0; row < NR0; ++row) {
                if (r0 + row < ne01) {
                    device const int8_t * qs = ax[row][ib].qs + il * NQ;
                    for (short j = 0; j < M; ++j) {
                        float sumq = 0.f;
                        for (int iq = 0; iq < NQ; ++iq) {
                            sumq += qs[iq] * yl[j][iq];
                        }
                        sumf[v][row][j] += sumq * ax[row][ib].d;
                    }
                }
            }
        }
    }

    // Stock combine: simd_sum per (virtual) simdgroup; threadgroup slot
    // [row][j][0..31] holds those sums at the virtual simdgroup index and
    // +0.0 elsewhere; the row value is simd_sum over the slot.
    for (short v = 0; v < V; ++v) {
        for (int row = 0; row < NR0; ++row) {
            for (short j = 0; j < M; ++j) {
                sumf[v][row][j] = simd_sum(sumf[v][row][j]);
            }
        }
    }
    device float * yb = y + (ulong)b * p.y_batch_stride;
    if (V == NSG) {
        // One physical simdgroup holds all four virtual sums: lane l < 4
        // takes sum l, the other lanes +0.0, as the stock threadgroup slot.
        for (int row = 0; row < NR0; ++row) {
            for (short j = 0; j < M; ++j) {
                float slot = 0.0f;
                for (short v = 0; v < V; ++v) {
                    slot = tiisg == (uint)v ? sumf[v][row][j] : slot;
                }
                const float tot = simd_sum(slot);
                if (tiisg == 0 && r0 + row < ne01) {
                    yb[(ulong)j * p.y_row_stride + r0 + row] = tot;
                }
            }
        }
        return;
    }
    const uint nthreads = NW * (NSG / V);
    const uint tid = sgitg * NW + tiisg;
    for (uint slot = tid; slot < (uint)(NR0 * M * NW); slot += nthreads) {
        shmem[slot] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        for (short v = 0; v < V; ++v) {
            const int vsg = (int)sgitg * V + v;
            for (int row = 0; row < NR0; ++row) {
                for (short j = 0; j < M; ++j) {
                    shmem[(row * M + j) * NW + vsg] = sumf[v][row][j];
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = sgitg; slot < (uint)(NR0 * M); slot += NSG / V) {
        const int row = slot / M;
        const int j = slot % M;
        const float tot = simd_sum(shmem[slot * NW + tiisg]);
        if (tiisg == 0 && r0 + row < ne01) {
            yb[(ulong)j * p.y_row_stride + r0 + row] = tot;
        }
    }
}

typedef decltype(row_exact_stock_mv_f16_f32<1, 1, 1>) row_exact_stock_mv_f16_t;
typedef decltype(row_exact_stock_mv_q8_0_f32<1, 1, 1>) row_exact_stock_mv_q8_0_t;

#define ROW_EXACT_STOCK_F16(NR, U, M) \
    template [[host_name("row_exact_stock_mv_f16_f32_r" #NR "_u" #U "_m" #M)]] \
    kernel row_exact_stock_mv_f16_t row_exact_stock_mv_f16_f32<NR, U, M>;
#define ROW_EXACT_STOCK_Q8_0(NR0, V, M) \
    template [[host_name("row_exact_stock_mv_q8_0_f32_r" #NR0 "_v" #V "_m" #M)]] \
    kernel row_exact_stock_mv_q8_0_t row_exact_stock_mv_q8_0_f32<NR0, V, M>;

#define ROW_EXACT_STOCK_ALL_M(MACRO, A, B) \
    MACRO(A, B, 1) MACRO(A, B, 2) MACRO(A, B, 3) MACRO(A, B, 4) \
    MACRO(A, B, 5) MACRO(A, B, 6) MACRO(A, B, 7) MACRO(A, B, 8)

ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_F16, 1, 1)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_F16, 1, 4)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_F16, 4, 1)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_F16, 4, 4)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_Q8_0, 1, 1)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_Q8_0, 1, 2)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_Q8_0, 1, 4)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_Q8_0, 2, 1)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_Q8_0, 2, 2)
ROW_EXACT_STOCK_ALL_M(ROW_EXACT_STOCK_Q8_0, 2, 4)
