// Row-exact few-row mat-vec: y[j, n] = sum_k W[n, k] * x[j, k] for j < M <= 8.
//
// Weight formats: F16 (`row_exact_mv_f16_f32_r<NR0>_m<M>`) and Q8_0
// (`row_exact_mv_q8_0_f32_r<NR0>_m<M>`). Input and output are F32. One
// dispatch reads each weight row once for all M input rows (the peer's
// few-row `mul_mv_ext` idea) and the whole threadgroup, T = 32 * nsg threads,
// splits K of its NR0 rows (the peer's `mul_mv` nsg split), so skinny
// outputs still fill the GPU.
//
// Row exactness: output row j depends only on (W, x[j], K, T); never on M,
// NR0 or the other input rows. Thread t walks a fixed subset of K in
// ascending order with explicit `fma` chains (no plain adds the fast-math
// compiler could reassociate); partial sums meet through `simd_sum` over
// fixed lanes and threadgroup memory. The host derives T from the weight
// shape only.
//
// Grid: (batch * ceil(N / NR0), 1, 1), batch-major. Threadgroup: 32 * nsg
// threads, nsg <= 32. Threadgroup memory: NR0 * M * 32 floats (only read
// when nsg > 1).

#include <metal_stdlib>
using namespace metal;

struct RowExactMvParams {
    uint  k;              // inner dimension (elements)
    uint  n;              // output columns = weight rows per batch
    uint  x_row_stride;   // f32 elements between input rows
    uint  x_batch_stride; // f32 elements between input batches
    uint  y_row_stride;   // f32 elements between output rows
    uint  y_batch_stride; // f32 elements between output batches
    uint  tgs_per_batch;  // threadgroups per batch = ceil(n / NR0)
    uint  pad;
    ulong w_row_bytes;    // bytes between weight rows
    ulong w_batch_bytes;  // bytes between weight batches
};

// ---------------------------------------------------------------- F16 weights
//
// Unit of work: one half4. Step s of thread t covers half4 indices
// s*4T + i*T + t for i = 0..3, so each load instruction is contiguous across
// the threadgroup. The four products of a half4 are fused into the accumulator in
// element order.

template <short NR0, short NR1>
inline void row_exact_acc_f16(
    device const char * wb,
    ulong row_bytes,
    thread const uint (&rows)[NR0],
    device const float * xb,
    uint x_row_stride,
    uint k,
    uint T,
    uint t,
    thread float (&acc)[NR0][NR1]) {
    device const half4 * w4[NR0];
    for (short r = 0; r < NR0; ++r) {
        w4[r] = (device const half4 *)(wb + (ulong)rows[r] * row_bytes);
    }
    device const float4 * x4[NR1];
    for (short j = 0; j < NR1; ++j) {
        x4[j] = (device const float4 *)(xb + (ulong)j * x_row_stride);
    }
    const uint q_total = k / 4;
    const uint step = 4 * T;
    const uint q_full = (q_total / step) * step;

    for (uint base = 0; base < q_full; base += step) {
        float4 wv[NR0][4];
        for (short r = 0; r < NR0; ++r) {
            for (short i = 0; i < 4; ++i) {
                wv[r][i] = float4(w4[r][base + i * T + t]);
            }
        }
        for (short j = 0; j < NR1; ++j) {
            float4 xv[4];
            for (short i = 0; i < 4; ++i) {
                xv[i] = x4[j][base + i * T + t];
            }
            for (short r = 0; r < NR0; ++r) {
                float a = acc[r][j];
                for (short i = 0; i < 4; ++i) {
                    a = fma(wv[r][i].x, xv[i].x, a);
                    a = fma(wv[r][i].y, xv[i].y, a);
                    a = fma(wv[r][i].z, xv[i].z, a);
                    a = fma(wv[r][i].w, xv[i].w, a);
                }
                acc[r][j] = a;
            }
        }
    }
    // Tail step: same index map, guarded per half4 (still ascending per thread).
    for (short i = 0; i < 4; ++i) {
        const uint q = q_full + i * T + t;
        if (q < q_total) {
            float4 wv[NR0];
            for (short r = 0; r < NR0; ++r) {
                wv[r] = float4(w4[r][q]);
            }
            for (short j = 0; j < NR1; ++j) {
                const float4 xv = x4[j][q];
                for (short r = 0; r < NR0; ++r) {
                    float a = acc[r][j];
                    a = fma(wv[r].x, xv.x, a);
                    a = fma(wv[r].y, xv.y, a);
                    a = fma(wv[r].z, xv.z, a);
                    a = fma(wv[r].w, xv.w, a);
                    acc[r][j] = a;
                }
            }
        }
    }
}

// --------------------------------------------------------------- Q8_0 weights
//
// Block: half d + int8 qs[32] (34 bytes). Unit of work: a quarter block
// (8 quants, the peer's NQ = 8 per thread). Step s of thread t covers
// quarter indices s*2T + i*T + t for i = 0..1, so consecutive threads read
// consecutive quarters (the peer's q8_0 `mul_mv` lane map). Per quarter:
// s = fma chain over the 8 q*x products in element order, then
// acc = fma(s, d, acc).

constant constexpr uint ROW_EXACT_Q8_0_BLOCK_BYTES = 34;

inline float row_exact_q8_0_quarter(device const char * blk, uint sub, float4 x0, float4 x1) {
    device const ushort * qs = (device const ushort *)(blk + 2 + 8 * sub);
    const char2 q01 = as_type<char2>(qs[0]);
    const char2 q23 = as_type<char2>(qs[1]);
    const char2 q45 = as_type<char2>(qs[2]);
    const char2 q67 = as_type<char2>(qs[3]);
    float s = (float)q01.x * x0.x;
    s = fma((float)q01.y, x0.y, s);
    s = fma((float)q23.x, x0.z, s);
    s = fma((float)q23.y, x0.w, s);
    s = fma((float)q45.x, x1.x, s);
    s = fma((float)q45.y, x1.y, s);
    s = fma((float)q67.x, x1.z, s);
    s = fma((float)q67.y, x1.w, s);
    return s;
}

template <short NR0, short NR1>
inline void row_exact_q8_0_unit(
    device const char * const (&wr)[NR0],
    device const float4 * const (&x4)[NR1],
    uint q,
    thread float (&acc)[NR0][NR1]) {
    device const char * blk[NR0];
    float d[NR0];
    for (short r = 0; r < NR0; ++r) {
        blk[r] = wr[r] + (ulong)(q >> 2) * ROW_EXACT_Q8_0_BLOCK_BYTES;
        d[r] = (float)(*(device const half *)blk[r]);
    }
    for (short j = 0; j < NR1; ++j) {
        const float4 x0 = x4[j][2 * q];
        const float4 x1 = x4[j][2 * q + 1];
        for (short r = 0; r < NR0; ++r) {
            const float s = row_exact_q8_0_quarter(blk[r], q & 3, x0, x1);
            acc[r][j] = fma(s, d[r], acc[r][j]);
        }
    }
}

template <short NR0, short NR1>
inline void row_exact_acc_q8_0(
    device const char * wb,
    ulong row_bytes,
    thread const uint (&rows)[NR0],
    device const float * xb,
    uint x_row_stride,
    uint k,
    uint T,
    uint t,
    thread float (&acc)[NR0][NR1]) {
    device const char * wr[NR0];
    for (short r = 0; r < NR0; ++r) {
        wr[r] = wb + (ulong)rows[r] * row_bytes;
    }
    device const float4 * x4[NR1];
    for (short j = 0; j < NR1; ++j) {
        x4[j] = (device const float4 *)(xb + (ulong)j * x_row_stride);
    }
    const uint q_total = k / 8;
    const uint step = 2 * T;
    const uint q_full = (q_total / step) * step;

    for (uint base = 0; base < q_full; base += step) {
        for (short i = 0; i < 2; ++i) {
            row_exact_q8_0_unit<NR0, NR1>(wr, x4, base + i * T + t, acc);
        }
    }
    for (short i = 0; i < 2; ++i) {
        const uint q = q_full + i * T + t;
        if (q < q_total) {
            row_exact_q8_0_unit<NR0, NR1>(wr, x4, q, acc);
        }
    }
}

// ----------------------------------------------------------------- dispatch

// WT: 0 = F16, 1 = Q8_0.
template <short WT, short NR0, short NR1>
inline void row_exact_acc(
    device const char * wb,
    ulong row_bytes,
    thread const uint (&rows)[NR0],
    device const float * xb,
    uint x_row_stride,
    uint k,
    uint T,
    uint t,
    thread float (&acc)[NR0][NR1]) {
    if (WT == 0) {
        row_exact_acc_f16<NR0, NR1>(wb, row_bytes, rows, xb, x_row_stride, k, T, t, acc);
    } else {
        row_exact_acc_q8_0<NR0, NR1>(wb, row_bytes, rows, xb, x_row_stride, k, T, t, acc);
    }
}

// The threadgroup reduces NR0 rows; threadgroup memory holds one partial per
// (row, input, simdgroup).
template <short WT, short NR0, short NR1>
kernel void row_exact_mv(
    device const char  * w     [[buffer(0)]],
    device const float * x     [[buffer(1)]],
    device       float * y     [[buffer(2)]],
    constant RowExactMvParams & p [[buffer(3)]],
    threadgroup float  * shmem [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sg    [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint b  = tgpig.x / p.tgs_per_batch;
    const uint r0 = (tgpig.x % p.tgs_per_batch) * NR0;
    uint rows[NR0];
    for (short r = 0; r < NR0; ++r) {
        rows[r] = min(r0 + (uint)r, p.n - 1);
    }
    float acc[NR0][NR1];
    for (short r = 0; r < NR0; ++r) {
        for (short j = 0; j < NR1; ++j) {
            acc[r][j] = 0.0f;
        }
    }
    row_exact_acc<WT, NR0, NR1>(
        w + (ulong)b * p.w_batch_bytes, p.w_row_bytes, rows,
        x + (ulong)b * p.x_batch_stride, p.x_row_stride, p.k, (uint)nsg * 32, tid, acc);

    device float * yb = y + (ulong)b * p.y_batch_stride;
    for (short r = 0; r < NR0; ++r) {
        for (short j = 0; j < NR1; ++j) {
            acc[r][j] = simd_sum(acc[r][j]);
        }
    }
    if (nsg == 1) {
        if (lane == 0) {
            for (short r = 0; r < NR0; ++r) {
                if (r0 + r < p.n) {
                    for (short j = 0; j < NR1; ++j) {
                        yb[(uint)j * p.y_row_stride + r0 + r] = acc[r][j];
                    }
                }
            }
        }
        return;
    }
    if (lane == 0) {
        for (short r = 0; r < NR0; ++r) {
            for (short j = 0; j < NR1; ++j) {
                shmem[(r * NR1 + j) * 32 + sg] = acc[r][j];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short v = sg; v < NR0 * NR1; v += nsg) {
        const float part = lane < nsg ? shmem[v * 32 + lane] : 0.0f;
        const float total = simd_sum(part);
        const short r = v / NR1;
        const short j = v % NR1;
        if (lane == 0 && r0 + r < p.n) {
            yb[(uint)j * p.y_row_stride + r0 + r] = total;
        }
    }
}

typedef decltype(row_exact_mv<0, 1, 1>) row_exact_mv_t;

#define ROW_EXACT_MV_M(WT, TAG, NR0, M) \
    template [[host_name("row_exact_mv_" TAG "_f32_r" #NR0 "_m" #M)]] kernel row_exact_mv_t row_exact_mv<WT, NR0, M>;

#define ROW_EXACT_MV_ALL_M(WT, TAG, NR0) \
    ROW_EXACT_MV_M(WT, TAG, NR0, 1)      \
    ROW_EXACT_MV_M(WT, TAG, NR0, 2)      \
    ROW_EXACT_MV_M(WT, TAG, NR0, 3)      \
    ROW_EXACT_MV_M(WT, TAG, NR0, 4)      \
    ROW_EXACT_MV_M(WT, TAG, NR0, 5)      \
    ROW_EXACT_MV_M(WT, TAG, NR0, 6)      \
    ROW_EXACT_MV_M(WT, TAG, NR0, 7)      \
    ROW_EXACT_MV_M(WT, TAG, NR0, 8)

ROW_EXACT_MV_ALL_M(0, "f16", 1)
ROW_EXACT_MV_ALL_M(0, "f16", 2)
ROW_EXACT_MV_ALL_M(1, "q8_0", 1)
ROW_EXACT_MV_ALL_M(1, "q8_0", 2)
