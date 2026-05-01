// quantized_matmul_ggml.metal — GGML block-format quantized mat-vec kernels.
//
// Portions of this file are derived from candle-metal-kernels v0.10.2
// (https://github.com/huggingface/candle), Apache-2.0 licensed.
// Original source: llama.cpp's ggml-metal.metal, vendored in candle.
// Source: candle-metal-kernels/src/metal_src/quantized.metal
//
// Block struct definitions and dequantization formulas are byte-for-byte
// compatible with GGUF on-disk format. The kernel dispatch pattern is
// adapted to mlx-native's CommandEncoder API.
//
// Copyright the candle Authors and llama.cpp Authors.
// See LICENSE-APACHE-candle in this directory.

#include <metal_stdlib>
using namespace metal;

// ---- Constants ----

#define QK4_0 32
#define QK8_0 32
#define QK_K  256

#define N_DST       4   // each SIMD group works on 4 rows (Q4_0, Q6_K)
#define N_SIMDGROUP 2   // number of SIMD groups per threadgroup (Q4_0, Q6_K)
#define N_SIMDWIDTH 32  // Apple GPU SIMD width

// Q8_0 uses wider threadgroups: 4 simdgroups × 2 rows = 8 rows/tg.
// Matches llama.cpp N_SG_Q8_0=4, N_R0_Q8_0=2.
#define N_DST_Q8       2   // each SIMD group works on 2 rows
#define N_SIMDGROUP_Q8 4   // 4 SIMD groups per threadgroup (128 threads)

// Packed parameter struct — matches Rust-side GgmlMatvecGpuParams.
struct GgmlMatvecParams {
    int64_t ne00; // K: number of values per weight row (before quantization)
    int64_t ne01; // N: number of weight rows (output dim)
    int64_t ne02; // batch dim for weights
    int64_t ne10; // K: number of values per input row
    int64_t ne12; // batch dim for input
    int64_t ne0;  // output stride (= ne01)
    int64_t ne1;  // M: number of input rows
    uint    r2;   // ne12 / ne02
    uint    r3;   // ne13 / ne03 (always 1 for non-batched)
};

// ---- GGML block struct definitions ----
// Byte-for-byte compatible with GGUF on-disk format.

typedef struct {
    half    d;              // delta (scale)
    uint8_t qs[QK4_0 / 2]; // 32 nibbles packed into 16 bytes
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(half) + QK4_0 / 2, "wrong q4_0 block size");

typedef struct {
    half   d;          // delta (scale)
    int8_t qs[QK8_0];  // 32 signed 8-bit quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(half) + QK8_0, "wrong q8_0 block size");

typedef struct {
    uint8_t ql[QK_K/2];      // lower 4 bits of 6-bit values
    uint8_t qh[QK_K/4];      // upper 2 bits of 6-bit values
    int8_t  scales[QK_K/16]; // 8-bit sub-block scales
    half    d;                // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(half) + QK_K/16 + 3*QK_K/4, "wrong q6_K block size");

// Q4_K: 256 values per block, 144 bytes per block.
// Layout: [half d][half dmin][uint8_t scales[12]][uint8_t qs[128]]
//   d     : super-block scale for the 6-bit quantized sub-block scales
//   dmin  : super-block scale for the 6-bit quantized sub-block mins
//   scales: packed 6-bit (sub-scale, sub-min) pairs for 8 sub-blocks
//           (same K_SCALE_SIZE=12 byte layout shared with Q5_K, decoded
//            via the kmask1/kmask2/kmask3 machinery below).
//   qs    : 128 bytes of 4-bit quantized values, low nibble = first half
//           of pair, high nibble = second half of pair.
//
// Q4_K is structurally Q5_K minus the 32-byte qh "high-bit" array.
//
// Source: ggml-common.h block_q4_K (llama.cpp).
#define K_SCALE_SIZE 12
typedef struct {
    half    d;                    // super-block scale for quantized scales
    half    dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // quants, low 4 bits (128 bytes)
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(half) + K_SCALE_SIZE + QK_K/2,
              "wrong q4_K block size");

// ---- Q4_0 mat-vec kernel ----
//
// Each SIMD group (32 threads) processes N_DST=4 rows.
// Two SIMD groups per threadgroup => 8 rows per threadgroup.
// Each thread processes half a Q4_0 block (16 nibbles).
//
// Dispatch: threadgroups=(ceil(N/8), M, B), threads_per_tg=(8, 8, 1)

// ADR-009 Phase 3A: match llama.cpp's 4-accumulator layout exactly.
// Using 4 separate accumulators (one per nibble position) instead of 2
// paired accumulators ensures identical floating-point rounding to
// llama.cpp's block_q_n_dot_y for block_q4_0.
inline float block_q4_0_dot_y(
    device const block_q4_0 * qb,
    float sumy,
    thread float * yl,
    int il
) {
    float d = qb->d;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device const uint16_t * qs = ((device const uint16_t *)qb + 1 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

kernel void kernel_mul_mv_q4_0_f32(
    device const  void  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    constant GgmlMatvecParams & p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = p.ne00 / QK4_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im % p.ne12;
    const uint i13 = im / p.ne12;

    const uint offset0 = first_row * nb + (i12/p.r2)*(nb*p.ne01) + (i13/p.r3)*(nb*p.ne01*p.ne02);

    device const block_q4_0 * x = (device const block_q4_0 *) src0 + offset0;
    device const float      * y = (device const float      *) src1 + r1*p.ne10 + im*p.ne00*p.ne1;

    float yl[16];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 2;
    const int il = (tiisg % 2) * 8;

    device const float * yb = y + ix * QK4_0 + il;

    // ADR-009 Phase 3A: match llama.cpp's two-accumulator sumy pattern.
    // llama.cpp accumulates sumy[0] (first half) and sumy[1] (second half)
    // separately, then combines. This ensures identical FP rounding.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy[2] = { 0.f, 0.f };
        for (int i = 0; i < 8; i += 2) {
            sumy[0] += yb[i] + yb[i+1];
            yl[i+0] = yb[i+0];
            yl[i+1] = yb[i+1] / 256.f;
            sumy[1] += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16] / 16.f;
            yl[i+9] = yb[i+17] / 4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q4_0_dot_y(x + ib + row*nb, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.ne01) {
            dst[im*p.ne0*p.ne1 + r1*p.ne0 + first_row + row] = tot;
        }
    }
}

// ---- Q8_0 mat-vec kernel ----
//
// This is the stock candle kernel geometry and reduction path used by the
// old passing TQ stack. Dispatch: threadgroups=(ceil(N/8), M, B),
// threads_per_tg=(8, 8, 1). No threadgroup shared memory.

#define NB_Q8_0 8

kernel void kernel_mul_mv_q8_0_f32(
    device const  void  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    constant GgmlMatvecParams & p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = p.ne00 / QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im % p.ne12;
    const uint i13 = im / p.ne12;

    const uint offset0 = first_row * nb + (i12 / p.r2) * (nb * p.ne01) + (i13 / p.r3) * (nb * p.ne01 * p.ne02);

    device const block_q8_0 * x = (device const block_q8_0 *) src0 + offset0;
    device const float      * y = (device const float      *) src1 + r1 * p.ne10 + im * p.ne00 * p.ne1;

    float yl[NB_Q8_0];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 4;
    const int il = tiisg % 4;

    device const float * yb = y + ix * QK8_0 + NB_Q8_0 * il;

    for (int ib = ix; ib < nb; ib += nw / 4) {
        for (int i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < nr; row++) {
            device const int8_t * qs = x[ib + row * nb].qs + NB_Q8_0 * il;
            float sumq = 0.f;
            for (int iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq * x[ib + row * nb].d;
        }

        yb += NB_Q8_0 * nw;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.ne01) {
            dst[r1 * p.ne0 + im * p.ne0 * p.ne1 + first_row + row] = tot;
        }
    }
}

// ---- Q6_K mat-vec kernel ----
//
// Dispatch: threadgroups=(ceil(N/2), M, B), threads_per_tg=(2, 32, 1)
// Each threadgroup handles 2 rows (one per SIMD group).

kernel void kernel_mul_mv_q6_K_f32(
    device const  void  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    constant GgmlMatvecParams & p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = p.ne00 / QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = tgpig.z;

    const int row = 2 * r0 + sgitg;

    const uint i12 = im % p.ne12;
    const uint i13 = im / p.ne12;

    const uint offset0 = (i12/p.r2)*(nb*p.ne01) + (i13/p.r3)*(nb*p.ne01*p.ne02);

    device const block_q6_K * x  = (device const block_q6_K *) src0 + row * nb + offset0;
    device const float      * yy = (device const float      *) src1 + r1*p.ne10 + im*p.ne00*p.ne1;

    float sumf = 0;

    const int tid  = tiisg / 2;
    const int ix   = tiisg % 2;
    const int ip   = tid / 8;
    const int il   = tid % 8;
    const int n    = 4;
    const int l0   = n * il;
    const int is   = 8*ip + l0/16;

    const int y_offset   = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*p.ne0 + im*p.ne0*p.ne1 + row] = tot;
    }
}

// ---- Q4_K mat-vec kernel ----
//
// ADR-013 P7 — port of llama.cpp `kernel_mul_mv_q4_K_f32_impl`
// (ggml-metal.metal:7715-7821). Algorithm: for each weight row, decode
// the 8 sub-block (scale, min) 6-bit pairs from the packed 12-byte
// `scales` array, dequant and dot-product against the input vector.
//
// Geometry (mirrors Q5_K mv_id pattern):
//   NSG        = 2 simdgroups per threadgroup
//   nr0_per_sg = 1 row per simdgroup
//   rows/tg    = 2  (one per simdgroup; row = 2*r0 + sgitg)
// Dispatch:    threadgroups=(ceil(N/2), M, B), threads_per_tg=(2, 32, 1)
//
// Scale-decode is identical to Q5_K's: same kmask1=0x3f3f, kmask2=0x0f0f,
// kmask3=0xc0c0, same `sc16[]` packing. Q4_K differs from Q5_K only by
// the absence of the `qh` (high-bit) accumulators — the inner loop
// reduces to (q1[l] & 0x0F) and (q1[l] & 0xF0) >> 4 paired with the
// pre-summed yl/yh/sumy.

kernel void kernel_mul_mv_q4_K_f32(
    device const  void  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    constant GgmlMatvecParams & p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nb = p.ne00 / QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = tgpig.z;

    const int row = 2 * (int)r0 + (int)sgitg;

    const uint i12 = im % p.ne12;
    const uint i13 = im / p.ne12;

    const uint offset0 = (i12/p.r2)*(nb*p.ne01) + (i13/p.r3)*(nb*p.ne01*p.ne02);

    device const block_q4_K * x  = (device const block_q4_K *) src0 + row * nb + offset0;
    device const float      * yy = (device const float      *) src1 + r1*p.ne10 + im*p.ne00*p.ne1;

    float sumf = 0.f;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    // tiisg ∈ [0, 31].  Same partitioning as Q5_K mv_id:
    //   tid = tiisg/4 (0..7)
    //   ix  = tiisg%4 (0..3)  → block stride = 4
    //   iq  = tid/4    (0..1) → which half of the super-block (low/high)
    //   ir  = tid%4    (0..3) → which 8-element slice within iq's half
    const int tid = tiisg / 4;
    const int ix  = tiisg % 4;
    const int iq  = tid / 4;
    const int ir  = tid % 4;
    const int n   = 8;

    const int l0       = n * ir;
    const int q_offset = 32 * iq + l0;
    const int y_offset = 64 * iq + l0;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix * QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t  * q1 = x[i].qs + q_offset;
        device const uint8_t  * q2 = q1 + 64;
        device const half     * dh = &x[i].d;
        // Read packed 6-bit scales/mins as 6 uint16_ts; iq selects
        // which half of the super-block we're decoding.
        device const uint16_t * a  = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            yl[l+0] = y1[l +  0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l + 32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l +  0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l + 32]; sumy[3] += yh[l+8];
        }

        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            // Low/high nibble pairs from q1 (first 32 vals) and q2 (third 32 vals).
            // No qh: Q4_K has no high-bit array, so the Q5_K formula's
            // acc2 (high-bit) accumulators collapse to zero; only the
            // raw nibble dot-products contribute.
            acc1[0] += yl[l+0] * (float)(q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (float)(q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (float)(q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (float)(q2[l] & 0xF0);
        }

        const float dall = (float)dh[0];
        const float dmin = (float)dh[1];
        sumf += dall * ((float)sc8[0] * (acc1[0]        ) +
                        (float)sc8[1] * (acc1[1] / 16.f ) +
                        (float)sc8[4] * (acc1[2]        ) +
                        (float)sc8[5] * (acc1[3] / 16.f )) -
               dmin * (sumy[0] * (float)sc8[2] + sumy[1] * (float)sc8[3] +
                       sumy[2] * (float)sc8[6] + sumy[3] * (float)sc8[7]);

        y1 += 4 * QK_K;
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0 && row < (int)p.ne01) {
        dst[r1*p.ne0 + im*p.ne0*p.ne1 + row] = tot;
    }
}
