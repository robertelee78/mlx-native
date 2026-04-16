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

// ---- Q4_0 mat-vec kernel ----
//
// Each SIMD group (32 threads) processes N_DST=4 rows.
// Two SIMD groups per threadgroup => 8 rows per threadgroup.
// Each thread processes half a Q4_0 block (16 nibbles).
//
// Dispatch: threadgroups=(ceil(N/8), M, B), threads_per_tg=(8, 8, 1)

inline float block_q4_0_dot_y(
    device const block_q4_0 * qb,
    float sumy,
    thread float * yl,
    int il
) {
    float d = qb->d;
    float2 acc = 0.f;
    device const uint16_t * qs = ((device const uint16_t *)qb + 1 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
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

    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+0];
            yl[i+1] = yb[i+1] / 256.f;
            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16] / 16.f;
            yl[i+9] = yb[i+17] / 4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q4_0_dot_y(x + ib + row*nb, sumy, yl, il);
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
