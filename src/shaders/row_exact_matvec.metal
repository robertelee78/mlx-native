// Row-exact few-row mat-vec: y[j, n] = sum_k W[n, k] * x[j, k] for j < M <= 8.
//
// Weight formats: F16 (`row_exact_mv_f16_f32_r<NR0>_m<M>`), Q8_0
// (`row_exact_mv_q8_0_f32_r<NR0>_m<M>`), and Q4_K / Q6_K / Q5_0
// (`row_exact_mv_{q4_K,q6_K,q5_0}_f32_r<NR0>_m<M>`, NR0 in {1, 2, 4}).
// Input and output are F32. One
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

// ------------------------------------------------- Q4_K / Q6_K / Q5_0 weights
//
// Unit of work: a fixed run of consecutive quants of one block (32 values
// for Q4_K / Q6_K, 16 for Q5_0). Unit u of a row is slice u % S of block
// u / S (S units per block) and thread t walks units t, t + T, t + 2T, ...
// in ascending order, so consecutive threads read consecutive slices of a
// block (the reference implementation's `mul_mv` lane maps for these types:
// 8 / 8 / 2 threads per block). Per unit and weight row, each group of four
// values is dequantized once into a float4 with a fixed formula (an explicit
// `fma` or an exact product; never touched by M), then every input row j
// runs `acc = fma(w, x, acc)` over the unit's values in the order listed
// per type. The integer unpacks work on four bytes at a time.

// Q4_K block (144 bytes): half d, half dmin, uchar scales[12], uchar
// qs[128]. Eight 32-value sub-blocks with 6-bit scale sc and min mn;
// w = d*sc * q - dmin*mn. qs byte 32g + l holds value 64g + l (sub-block
// 2g) in its low nibble and 64g + 32 + l (sub-block 2g + 1) in its high
// nibble. Unit (S = 8): the 16-byte qs slice p = u % 8, g = p / 2,
// h = p % 2: values 64g + 16h + i (low nibbles) then 64g + 32 + 16h + i
// (high nibbles), i = 0..15. Per value w = fma(d*sc, q, -dmin*mn): d*sc and
// dmin*mn are exact in F32 (11-bit by 6-bit), so w is the block's value
// rounded once.
//
// Loads: the 16-byte block header (d, dmin, scales) and the 16-byte qs
// slice, one uint4 each (blocks are 16-byte aligned: 144 = 9 * 16). Unpack:
// a nibble is masked in place inside its 16-bit half-word (masks 0x000F,
// 0x00F0, 0x0F00, 0xF000, the reference implementation's q4_K trick), so it
// converts as q * 2^s, and the scale is pre-multiplied by 2^-s:
// fma(d*sc * 2^-s, q * 2^s, -dmin*mn) has the same exact product as
// fma(d*sc, q, -dmin*mn) and therefore the same single rounding.
// Each thread loads two units before using either (more weight bytes in
// flight per thread); the accumulation order is unchanged.

constant constexpr uint ROW_EXACT_Q4_K_BLOCK_BYTES = 144;

#define ROW_EXACT_UNROLL _Pragma("clang loop unroll(full)")

template <short NR0>
inline void row_exact_q4_K_load(
    device const char * const (&wr)[NR0],
    uint u,
    thread uint4 (&hdr)[NR0],
    thread uint4 (&qs)[NR0]) {
    ROW_EXACT_UNROLL
    for (short r = 0; r < NR0; ++r) {
        device const char * blk = wr[r] + (ulong)(u >> 3) * ROW_EXACT_Q4_K_BLOCK_BYTES;
        hdr[r] = *(device const uint4 *)blk;
        qs[r] = *(device const uint4 *)(blk + 16 + 16 * (u & 7));
    }
}

template <short NR0, short NR1>
inline void row_exact_q4_K_unit(
    thread const uint4 (&hdr)[NR0],
    thread const uint4 (&qs)[NR0],
    device const float4 * const (&x4)[NR1],
    uint u,
    thread float (&acc)[NR0][NR1]) {
    const uint p = u & 7;
    const uint g = p >> 1;
    const bool odd = (g & 1) != 0;
    // float4 index of the low run's first value.
    const uint xq = (u >> 3) * 64 + g * 16 + (p & 1) * 4;
    // Per row: scales for the in-place nibble masks of the low run
    // (0x000F, 0x0F00; sub-block 2g) and the high run (0x00F0, 0xF000;
    // sub-block 2g + 1), and the negated mins.
    float sl[NR0][2][2];
    float nm[NR0][2];
    ROW_EXACT_UNROLL
    for (short r = 0; r < NR0; ++r) {
        const float2 dm = float2(as_type<half2>(hdr[r].x));
        // 16-bit words a0, a2, a4 of the scales (the reference
        // implementation's packed 6-bit decode; g & 1 picks the word half).
        const ushort2 s0 = as_type<ushort2>(hdr[r].y);
        const ushort2 s1 = as_type<ushort2>(hdr[r].z);
        const ushort2 s2 = as_type<ushort2>(hdr[r].w);
        const ushort a0 = odd ? s0.y : s0.x;
        const ushort a2 = odd ? s1.y : s1.x;
        const ushort a4 = odd ? s2.y : s2.x;
        const ushort sc = g < 2 ? ushort(a0 & 0x3f3f) : ushort((a4 & 0x0f0f) | ((a0 & 0xc0c0) >> 2));
        const ushort mn = g < 2 ? ushort(a2 & 0x3f3f) : ushort(((a4 >> 4) & 0x0f0f) | ((a2 & 0xc0c0) >> 2));
        const float d0 = dm.x * (float)(sc & 0xff);
        const float d1 = dm.x * (float)(sc >> 8);
        sl[r][0][0] = d0;
        sl[r][0][1] = d0 * (1.0f / 256.0f);
        sl[r][1][0] = d1 * (1.0f / 16.0f);
        sl[r][1][1] = d1 * (1.0f / 4096.0f);
        nm[r][0] = -(dm.y * (float)(mn & 0xff));
        nm[r][1] = -(dm.y * (float)(mn >> 8));
    }
    ROW_EXACT_UNROLL
    for (short hn = 0; hn < 2; ++hn) {
        const ushort mlo = hn == 0 ? 0x000F : 0x00F0;
        const ushort mhi = hn == 0 ? 0x0F00 : 0xF000;
        ROW_EXACT_UNROLL
        for (short i = 0; i < 4; ++i) {
            float4 w[NR0];
            ROW_EXACT_UNROLL
            for (short r = 0; r < NR0; ++r) {
                const ushort2 h = as_type<ushort2>(qs[r][i]);
                const float a = sl[r][hn][0];
                const float b = sl[r][hn][1];
                const float c = nm[r][hn];
                w[r] = float4(fma(a, (float)(ushort)(h.x & mlo), c),
                              fma(b, (float)(ushort)(h.x & mhi), c),
                              fma(a, (float)(ushort)(h.y & mlo), c),
                              fma(b, (float)(ushort)(h.y & mhi), c));
            }
            ROW_EXACT_UNROLL
            for (short j = 0; j < NR1; ++j) {
                const float4 xv = x4[j][xq + 8 * hn + i];
                ROW_EXACT_UNROLL
                for (short r = 0; r < NR0; ++r) {
                    float a = acc[r][j];
                    a = fma(w[r].x, xv.x, a);
                    a = fma(w[r].y, xv.y, a);
                    a = fma(w[r].z, xv.z, a);
                    a = fma(w[r].w, xv.w, a);
                    acc[r][j] = a;
                }
            }
        }
    }
}

template <short NR0, short NR1>
inline void row_exact_acc_q4_K(
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
    const uint u_total = k / 32;
    uint u = t;
    for (; u + T < u_total; u += 2 * T) {
        uint4 hdr_a[NR0], qs_a[NR0], hdr_b[NR0], qs_b[NR0];
        row_exact_q4_K_load<NR0>(wr, u, hdr_a, qs_a);
        row_exact_q4_K_load<NR0>(wr, u + T, hdr_b, qs_b);
        row_exact_q4_K_unit<NR0, NR1>(hdr_a, qs_a, x4, u, acc);
        row_exact_q4_K_unit<NR0, NR1>(hdr_b, qs_b, x4, u + T, acc);
    }
    if (u < u_total) {
        uint4 hdr_a[NR0], qs_a[NR0];
        row_exact_q4_K_load<NR0>(wr, u, hdr_a, qs_a);
        row_exact_q4_K_unit<NR0, NR1>(hdr_a, qs_a, x4, u, acc);
    }
}

// Q6_K block (210 bytes): uchar ql[128], uchar qh[64], int8 scales[16],
// half d. Value 128n + 32c + l (n = 0..1, c = 0..3, l = 0..31) is
// q = (ql[64n + 32(c & 1) + l] >> 4(c >> 1)) & 15 | ((qh[32n + l] >> 2c) & 3) << 4,
// w = d * scales[8n + 2c + l / 16] * (q - 32). Unit (S = 8): p = u % 8,
// n = p / 4, l0 = 8 (p % 4): values 128n + 32c + l0 + i for c = 0..3,
// i = 0..7 (ascending). Per value w = fma(d*sc, q, -32*d*sc): every term is
// exact and so is w (18-bit d*sc times |q - 32| <= 32, at most 5 significant
// bits). Blocks are only
// 2-byte aligned, so quants load as ushorts.

constant constexpr uint ROW_EXACT_Q6_K_BLOCK_BYTES = 210;

inline uint row_exact_u32(device const ushort * p) {
    return (uint)p[0] | ((uint)p[1] << 16);
}

template <short NR0, short NR1>
inline void row_exact_q6_K_unit(
    device const char * const (&wr)[NR0],
    device const float4 * const (&x4)[NR1],
    uint u,
    thread float (&acc)[NR0][NR1]) {
    const uint p = u & 7;
    const uint n = p >> 2;
    const uint l0 = 8 * (p & 3);
    // float4 index of value 128n + l0.
    const uint xq = (u >> 3) * 64 + 32 * n + l0 / 4;
    uint ql[NR0][2][2];
    uint qh[NR0][2];
    float dl[NR0][4];
    for (short r = 0; r < NR0; ++r) {
        device const char * blk = wr[r] + (ulong)(u >> 3) * ROW_EXACT_Q6_K_BLOCK_BYTES;
        device const ushort * qa = (device const ushort *)(blk + 64 * n + l0);
        device const ushort * qb = (device const ushort *)(blk + 64 * n + 32 + l0);
        device const ushort * hh = (device const ushort *)(blk + 128 + 32 * n + l0);
        for (short i = 0; i < 2; ++i) {
            ql[r][0][i] = row_exact_u32(qa + 2 * i);
            ql[r][1][i] = row_exact_u32(qb + 2 * i);
            qh[r][i] = row_exact_u32(hh + 2 * i);
        }
        device const char * sc = blk + 192 + 8 * n + l0 / 16;
        const float d = (float)(*(device const half *)(blk + 208));
        for (short c = 0; c < 4; ++c) {
            dl[r][c] = d * (float)sc[2 * c];
        }
    }
    for (short c = 0; c < 4; ++c) {
        for (short i = 0; i < 2; ++i) {
            float4 w[NR0];
            for (short r = 0; r < NR0; ++r) {
                const uint lo = (ql[r][c & 1][i] >> (4 * (c >> 1))) & 0x0f0f0f0f;
                // qh bits 2c..2c+1 of each byte moved to bits 4..5.
                const uint h = qh[r][i];
                const uint hs = c == 0 ? h << 4 : c == 1 ? h << 2 : c == 2 ? h : h >> 2;
                const uint hi = hs & 0x30303030;
                w[r] = fma(float4(dl[r][c]), float4(as_type<uchar4>(lo | hi)), float4(-32.0f * dl[r][c]));
            }
            for (short j = 0; j < NR1; ++j) {
                const float4 xv = x4[j][xq + 8 * c + i];
                for (short r = 0; r < NR0; ++r) {
                    float a = acc[r][j];
                    a = fma(w[r].x, xv.x, a);
                    a = fma(w[r].y, xv.y, a);
                    a = fma(w[r].z, xv.z, a);
                    a = fma(w[r].w, xv.w, a);
                    acc[r][j] = a;
                }
            }
        }
    }
}

// Q5_0 block (22 bytes): half d, uchar qh[4], uchar qs[16]. Value v
// (0..31) is q = (qs[v % 16] >> 4(v / 16)) & 15 | ((qh >> v) & 1) << 4 with
// qh read little-endian, w = d * (q - 16). Unit (S = 2): half block
// p = u % 2, il = 8p: values il + i then 16 + il + i, i = 0..7 (the
// reference implementation's two threads per block). Per value
// w = fma(d, q, -16d), exact (11-bit d times a 5-bit integer). Blocks are
// only 2-byte aligned, so qh and qs load as ushorts.

constant constexpr uint ROW_EXACT_Q5_0_BLOCK_BYTES = 22;

template <short NR0, short NR1>
inline void row_exact_q5_0_unit(
    device const char * const (&wr)[NR0],
    device const float4 * const (&x4)[NR1],
    uint u,
    thread float (&acc)[NR0][NR1]) {
    const uint p = u & 1;
    // float4 index of value 8p of the block.
    const uint xq = (u >> 1) * 8 + 2 * p;
    uint qs[NR0][2];
    uint qh[NR0];
    float d[NR0];
    for (short r = 0; r < NR0; ++r) {
        device const char * blk = wr[r] + (ulong)(u >> 1) * ROW_EXACT_Q5_0_BLOCK_BYTES;
        device const ushort * q = (device const ushort *)(blk + 6 + 8 * p);
        d[r] = (float)(*(device const half *)blk);
        qh[r] = row_exact_u32((device const ushort *)(blk + 2)) >> (8 * p);
        qs[r][0] = row_exact_u32(q);
        qs[r][1] = row_exact_u32(q + 2);
    }
    for (short hn = 0; hn < 2; ++hn) {
        for (short i = 0; i < 2; ++i) {
            float4 w[NR0];
            for (short r = 0; r < NR0; ++r) {
                const uint lo = (qs[r][i] >> (4 * hn)) & 0x0f0f0f0f;
                // Bits 4i..4i+3 of this run's qh nibble, spread to bit 4 of each byte.
                const uint b = (qh[r] >> (16 * hn + 4 * i)) & 0xf;
                const uint hi = ((b * 0x00204081u) & 0x01010101u) << 4;
                w[r] = fma(float4(d[r]), float4(as_type<uchar4>(lo | hi)), float4(-16.0f * d[r]));
            }
            for (short j = 0; j < NR1; ++j) {
                const float4 xv = x4[j][xq + 4 * hn + i];
                for (short r = 0; r < NR0; ++r) {
                    float a = acc[r][j];
                    a = fma(w[r].x, xv.x, a);
                    a = fma(w[r].y, xv.y, a);
                    a = fma(w[r].z, xv.z, a);
                    a = fma(w[r].w, xv.w, a);
                    acc[r][j] = a;
                }
            }
        }
    }
}

// WT: 3 = Q6_K, 4 = Q5_0 (Q4_K: row_exact_acc_q4_K).
template <short WT, short NR0, short NR1>
inline void row_exact_acc_blocks(
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
    const uint u_total = k / (WT == 4 ? 16 : 32);
    for (uint u = t; u < u_total; u += T) {
        if (WT == 3) {
            row_exact_q6_K_unit<NR0, NR1>(wr, x4, u, acc);
        } else {
            row_exact_q5_0_unit<NR0, NR1>(wr, x4, u, acc);
        }
    }
}

// ----------------------------------------------------------------- dispatch

// WT: 0 = F16, 1 = Q8_0, 2 = Q4_K, 3 = Q6_K, 4 = Q5_0.
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
    } else if (WT == 1) {
        row_exact_acc_q8_0<NR0, NR1>(wb, row_bytes, rows, xb, x_row_stride, k, T, t, acc);
    } else if (WT == 2) {
        row_exact_acc_q4_K<NR0, NR1>(wb, row_bytes, rows, xb, x_row_stride, k, T, t, acc);
    } else {
        row_exact_acc_blocks<WT, NR0, NR1>(wb, row_bytes, rows, xb, x_row_stride, k, T, t, acc);
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
ROW_EXACT_MV_ALL_M(2, "q4_K", 1)
ROW_EXACT_MV_ALL_M(2, "q4_K", 2)
ROW_EXACT_MV_ALL_M(2, "q4_K", 4)
ROW_EXACT_MV_ALL_M(3, "q6_K", 1)
ROW_EXACT_MV_ALL_M(3, "q6_K", 2)
ROW_EXACT_MV_ALL_M(3, "q6_K", 4)
ROW_EXACT_MV_ALL_M(4, "q5_0", 1)
ROW_EXACT_MV_ALL_M(4, "q5_0", 2)
ROW_EXACT_MV_ALL_M(4, "q5_0", 4)
