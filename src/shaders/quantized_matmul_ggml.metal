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

// ADR-029 iter-162 H93: peer-grounded port of llama.cpp commit da4495332
// ("metal : promote mul_mv/mul_mm batch divisors to function constants").
//
// `ne12`, `r2`, `r3` appear in offset arithmetic as integer divisors:
//   const uint i12 = im % p.ne12;
//   const uint offset0 = ... (i12/p.r2)*... + (i13/p.r3)*...;
//
// Integer division on Apple Silicon is expensive (~10-15 cycles, can't be
// pipelined). When the divisor is a function constant, the Metal compiler
// specializes the PSO with magic-number multiplication at compile time,
// reducing the cost to ~1-2 cycles. Hot path: ~210 matvec dispatches per
// decode token; each dispatch executes the div/mod per thread.
//
// Sentinel `-1` means "FC not set; fall back to runtime p.ne12 / p.r2 / p.r3
// from the args buffer" (backwards-compat — any dispatcher that hasn't yet
// been updated to set these FCs continues to work, just without the speedup).
// Production dispatchers should always set them.
//
// FC slot allocation: 700/701/702 — clear of all existing mlx-native FCs
// (highest currently used is 601 in mul_mv_ext).
constant int FC_qmatmul_ne12 [[function_constant(700)]];
constant int FC_qmatmul_r2   [[function_constant(701)]];
constant int FC_qmatmul_r3   [[function_constant(702)]];
constant int qmatmul_ne12_effective =
    is_function_constant_defined(FC_qmatmul_ne12) ? FC_qmatmul_ne12 : -1;
constant int qmatmul_r2_effective =
    is_function_constant_defined(FC_qmatmul_r2) ? FC_qmatmul_r2 : -1;
constant int qmatmul_r3_effective =
    is_function_constant_defined(FC_qmatmul_r3) ? FC_qmatmul_r3 : -1;

// Helper macros: return FC value if set, else the runtime arg. The branch
// is on a `constant` expression so the compiler DCEs the unused arm at PSO
// compile (once FCs are set, only the FC arm survives → no runtime branch).
#define QMM_NE12(p) ((qmatmul_ne12_effective >= 0) ? (uint)qmatmul_ne12_effective : (uint)(p).ne12)
#define QMM_R2(p)   ((qmatmul_r2_effective   >= 0) ? (uint)qmatmul_r2_effective   : (uint)(p).r2)
#define QMM_R3(p)   ((qmatmul_r3_effective   >= 0) ? (uint)qmatmul_r3_effective   : (uint)(p).r3)

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

// ADR-022 Phase 2 — Q5_K block (176 bytes).
// Layout: [half d][half dmin][uint8_t scales[12]][uint8_t qh[32]][uint8_t qs[128]]
// Adds a 32-byte qh "high-bit" array vs Q4_K. The high bit OR'd into each
// dequantized 4-bit nibble lifts the value range from [0,15] to [0,31].
typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(half) + K_SCALE_SIZE + QK_K/8 + QK_K/2,
              "wrong q5_K block size");

// Q5_1 (ADR-022 Phase 1). 32 values per block, 24 bytes per block.
typedef struct {
    half    d;
    half    m;
    uint    qh;
    uint8_t qs[QK4_0 / 2];
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2*sizeof(half) + 4 + QK4_0/2,
              "wrong q5_1 block size");

// IQ4_NL (ADR-022 Phase 1). 32 values per block, 18 bytes per block.
typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(half) + QK4_0/2,
              "wrong iq4_nl block size");

// Frozen IQ4_NL codebook (ggml-common.h:1109-1112). Lock-step with
// host-side `KVALUES_IQ4_NL` in src/gguf/mod.rs and the duplicate in
// quantized_matmul_id_ggml.metal.
constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// Q5_1 dot helper — see id_ggml.metal:135-179 for the formula derivation.
inline float block_q5_1_dot_y(
    device const block_q5_1 * qb,
    float sumy,
    thread float * yl,
    int il
) {
    float d = qb->d;
    float m = qb->m;
    float4 acc = 0.f;
    device const uint16_t * qs = ((device const uint16_t *)qb + 4 + il/2);
    const uint qh = qb->qh;
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0]
                * (float)((qs[i / 2] & 0x000F) | (((qh >> (i + 0 + il      )) << 4 ) & 0x0010));
        acc[1] += yl[i + 1]
                * (float)((qs[i / 2] & 0x0F00) | (((qh >> (i + 1 + il      )) << 12) & 0x1000));
        acc[2] += yl[i + 8]
                * (float)((qs[i / 2] & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8 ) & 0x0100));
        acc[3] += yl[i + 9]
                * (float)((qs[i / 2] & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
    }
    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// IQ4_NL dot helper — see id_ggml.metal:181-211 for the codebook-lookup
// rationale (raw yl[], no pre-scale, non-linear).
inline float block_iq4_nl_dot_y(
    device const block_iq4_nl * qb,
    thread float * yl_raw,
    int il
) {
    float d = qb->d;
    float acc = 0.f;
    device const uint8_t * qs = qb->qs + il;
    for (int i = 0; i < 8; i++) {
        const uint8_t b = qs[i];
        acc += yl_raw[i]     * (float)kvalues_iq4nl[b & 0x0F];
        acc += yl_raw[i + 8] * (float)kvalues_iq4nl[(b >> 4) & 0x0F];
    }
    return d * acc;
}

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

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = first_row * nb + (i12/QMM_R2(p))*(nb*p.ne01) + (i13/QMM_R3(p))*(nb*p.ne01*p.ne02);

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

// ---- Q5_1 mat-vec kernel (ADR-022 Phase 1) ----
//
// Same dispatch geometry as Q4_0; differs only in (a) block walked and
// (b) dot helper used. Q5_1 carries an `m` (min) term, contributing
// `m * sumy` to the dot product, plus the qh 5th-bit injection.

kernel void kernel_mul_mv_q5_1_f32(
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

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = first_row * nb + (i12/QMM_R2(p))*(nb*p.ne01) + (i13/QMM_R3(p))*(nb*p.ne01*p.ne02);

    device const block_q5_1 * x = (device const block_q5_1 *) src0 + offset0;
    device const float      * y = (device const float      *) src1 + r1*p.ne10 + im*p.ne00*p.ne1;

    float yl[16];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 2;
    const int il = (tiisg % 2) * 8;

    device const float * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0.f;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+0];
            yl[i+1] = yb[i+1] / 256.f;
            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16] / 16.f;
            yl[i+9] = yb[i+17] / 4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q5_1_dot_y(x + ib + row*nb, sumy, yl, il);
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

// ---- IQ4_NL mat-vec kernel (ADR-022 Phase 1) ----
//
// IQ4_NL's codebook lookup is non-linear; uses raw yl[] (no pre-scale).

kernel void kernel_mul_mv_iq4_nl_f32(
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

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = first_row * nb + (i12/QMM_R2(p))*(nb*p.ne01) + (i13/QMM_R3(p))*(nb*p.ne01*p.ne02);

    device const block_iq4_nl * x = (device const block_iq4_nl *) src0 + offset0;
    device const float        * y = (device const float        *) src1 + r1*p.ne10 + im*p.ne00*p.ne1;

    float yl_raw[16];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 2;
    const int il = (tiisg % 2) * 8;

    device const float * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += nw/2) {
        for (int i = 0; i < 8; i++) {
            yl_raw[i]     = yb[i];
            yl_raw[i + 8] = yb[i + 16];
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_iq4_nl_dot_y(x + ib + row*nb, yl_raw, il);
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

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = first_row * nb + (i12 / QMM_R2(p)) * (nb * p.ne01) + (i13 / QMM_R3(p)) * (nb * p.ne01 * p.ne02);

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

// ---- Q8_0 mat-vec kernel — peer-style NSG=4 NR=2 (ADR-028 iter-368) ----
//
// Port of llama.cpp's `kernel_mul_mv_q8_0_f32_impl` with N_R0_Q8_0=2 and
// N_SG_Q8_0=4 (functional constant equivalent).  Each TG covers 2 rows;
// 4 simdgroups collaborate on those 2 rows with cross-SG reduction via
// threadgroup memory.  Uses 128 threads/TG vs the existing kernel's 64 →
// better latency hiding on Apple Silicon.
//
// Math is mathematically equivalent to `kernel_mul_mv_q8_0_f32` (same
// row × col F32 dot products with identical accumulator order).  Difference
// is parallelism / dispatch geometry.
//
// Dispatch (host-side):
//   threadgroups   = (ceil(N/NR0), M, B)
//   threads_per_tg = (NW, NSG, 1) = (32, 4, 1)
//   shared memory  = NR0 * NW * sizeof(float) = 2 * 32 * 4 = 256 bytes
//
// Reference: /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:3572 (MIT).

#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define NQ_Q8_0   8

kernel void kernel_mul_mv_q8_0_f32_nr2(
    device const  void  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    constant GgmlMatvecParams & p [[buffer(3)]],
    threadgroup float   * shmem  [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NR0 = N_R0_Q8_0;   // 2
    constexpr int NSG = N_SG_Q8_0;   // 4
    constexpr int NW  = N_SIMDWIDTH; // 32
    constexpr int NQ  = NQ_Q8_0;     // 8

    const int nb = p.ne00 / QK8_0;
    const int r0 = tgpig.x * NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    // Per-row src0 pointers (unrolled NR0 iterations).
    device const block_q8_0 * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        const uint offset0 = (r0 + row) * nb
            + (i12 / QMM_R2(p)) * (nb * p.ne01)
            + (i13 / QMM_R3(p)) * (nb * p.ne01 * p.ne02);
        ax[row] = (device const block_q8_0 *) src0 + offset0;
    }

    device const float * y = (device const float *) src1
        + r1 * p.ne10
        + im * p.ne00 * p.ne1;

    float sumf[NR0] = { 0.f };

    const int ix = tiisg / (NW / NQ);  // 0..3
    const int il = tiisg % (NW / NQ);  // 0..3

    const int ib0 = sgitg * NQ + ix;

    float yl[NQ];
    device const float * yb = y + ib0 * QK8_0 + il * NQ;

    // Each thread covers NQ quants per iteration; SGs interleave across
    // ib by stride NSG*NQ.
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        for (int i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < NR0; ++row) {
            device const int8_t * qs = ax[row][ib].qs + il * NQ;
            float sumq = 0.f;
            for (int iq = 0; iq < NQ; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    // Cross-simdgroup reduction (peer's helper_mv_reduce_and_write pattern).
    threadgroup float * shmem_rows[NR0];
    for (int row = 0; row < NR0; ++row) {
        shmem_rows[row] = shmem + NW * row;
        // Pre-zero shmem for the final simd_sum read below.  Only sgitg==0
        // initializes; barrier serializes init+writes across SGs.
        if (sgitg == 0) {
            shmem_rows[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_rows[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = 0; row < NR0 && r0 + row < p.ne01; ++row) {
        const float tot = simd_sum(shmem_rows[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            dst[r1 * p.ne0 + im * p.ne0 * p.ne1 + r0 + row] = tot;
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

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = (i12/QMM_R2(p))*(nb*p.ne01) + (i13/QMM_R3(p))*(nb*p.ne01*p.ne02);

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

// ---- Q6_K mat-vec kernel, nr0=2 variant (ADR-028 iter-309) ----
//
// Ported from llama.cpp `kernel_mul_mv_q6_K_f32_impl` with N_R0_Q6_K=2.
// Doubles rows per simdgroup vs the baseline q6_K mv (1 row → 2) and
// caches `yl[16]` once per QK_K block, re-using it across both rows so
// the dequant unpack work amortizes.  4 rows per threadgroup (2 SGs ×
// 2 rows) vs 2 in the baseline.
//
// Dispatch: threadgroups=(ceil(N/4), M, B), threads_per_tg=(2, 32, 1).
// Each SIMD group handles 2 consecutive rows.
//
// Hypothesis (ADR-028 iter-308): cuts per-dispatch time on the biggest
// gemma4 APEX-Q5_K_M decode kernel (17.91% of dispatches) by closing
// the row-amortization gap with peer llama.cpp.
kernel void kernel_mul_mv_q6_K_f32_nr2(
    device const  void  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    constant GgmlMatvecParams & p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NSG = 2;   // simdgroups per threadgroup
    constexpr int nr0 = 2;   // rows per simdgroup
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = p.ne00 / QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = tgpig.z;

    const int first_row = (int)((r0 * NSG + sgitg) * nr0);

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = (i12/QMM_R2(p))*(nb*p.ne01) + (i13/QMM_R3(p))*(nb*p.ne01*p.ne02);

    device const block_q6_K * x_base = (device const block_q6_K *) src0 + first_row * nb + offset0;
    device const float      * yy = (device const float      *) src1 + r1*p.ne10 + im*p.ne00*p.ne1;

    float sumf[nr0] = {0.f, 0.f};
    float yl[16];

    // ADR-028 iter-401: use `short` (16-bit) for indexing to match peer's
    // ggml-metal.metal:8005-8014. Apple Metal compiler may emit more compact
    // 16-bit ALU ops; per peer's pattern.
    const short tid  = tiisg / 2;
    const short ix   = tiisg % 2;
    const short ip   = tid / 8;
    const short il   = tid % 8;
    const short l0   = 4 * il;
    const short is   = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l = 64*ip + l0;
    const short q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        // Y vector cached once per block, reused across nr0 rows.
        // ADR-028 iter-352: explicit `clang loop unroll(full)` (mirroring peer's
        // FOR_UNROLL macro at llama.cpp ggml-metal.metal:8035) was tested here
        // and FALSIFIED — measured -0.2-0.4 tok/s vs Apple Metal's auto-unroll.
        // Compiler was already doing the optimal thing without the hint, and the
        // explicit pragma may have hurt register allocation.  Removed; auto-unroll
        // retained as the production choice.
        device const float * y = yy + i * QK_K + y_offset;
        for (int l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (int row = 0; row < nr0; ++row) {
            device const block_q6_K * xr = x_base + row * nb;
            device const uint8_t * q1 = xr[i].ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = xr[i].qh + q_offset_h;
            device const int8_t  * sc = xr[i].scales + is;

            const float dall = xr[i].d;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (int l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    device float * dst_f32 = dst + im*p.ne0*p.ne1 + r1*p.ne0;
    for (int row = 0; row < nr0; ++row) {
        const int out_row = first_row + row;
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && out_row < p.ne01) {
            dst_f32[out_row] = tot;
        }
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

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = (i12/QMM_R2(p))*(nb*p.ne01) + (i13/QMM_R3(p))*(nb*p.ne01*p.ne02);

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
    // ADR-028 iter-406: short indexing matches peer Q4_K mv pattern.
    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0       = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

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
        for (int l = 0; l < 8; ++l) {
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
        for (int l = 0; l < 8; ++l) {
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

// ---- Q5_K dense mat-vec kernel (ADR-022 Phase 2) ----
//
// Port of llama.cpp `kernel_mul_mv_q5_K_f32_impl` (ggml-metal.metal:7837).
// Body is `kernel_mul_mv_q4_K_f32` (above) plus the Q5_K mv_id qh/acc2
// high-bit accumulation block — the only structural delta between Q4_K
// and Q5_K. The geometry, scale-decode (kmask1/2/3 + sc16 packing), and
// final dall/dmin reduction are byte-identical.

kernel void kernel_mul_mv_q5_K_f32(
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

    const uint i12 = im % QMM_NE12(p);
    const uint i13 = im / QMM_NE12(p);

    const uint offset0 = (i12/QMM_R2(p))*(nb*p.ne01) + (i13/QMM_R3(p))*(nb*p.ne01*p.ne02);

    device const block_q5_K * x  = (device const block_q5_K *) src0 + row * nb + offset0;
    device const float      * yy = (device const float      *) src1 + r1*p.ne10 + im*p.ne00*p.ne1;

    float sumf = 0.f;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    // ADR-028 iter-403: short indexing matches peer Q5_K mv (ggml-metal.metal:7873-7880).
    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0       = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix * QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t  * q1 = x[i].qs + q_offset;
        device const uint8_t  * q2 = q1 + 64;
        device const uint8_t  * qh = x[i].qh + l0;
        device const half     * dh = &x[i].d;
        device const uint16_t * a  = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
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
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            uint8_t h = qh[l];
            acc1[0] += yl[l+0] * (float)(q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (float)(q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (float)(q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (float)(q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
            acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
            acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
            acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
        }

        const float dall = (float)dh[0];
        const float dmin = (float)dh[1];
        sumf += dall * ((float)sc8[0] * (acc1[0]        + 16.f * acc2[0]) +
                        (float)sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                        (float)sc8[4] * (acc1[2]        + 16.f * acc2[2]) +
                        (float)sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
               dmin * (sumy[0] * (float)sc8[2] + sumy[1] * (float)sc8[3] +
                       sumy[2] * (float)sc8[6] + sumy[3] * (float)sc8[7]);

        y1 += 4 * QK_K;
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0 && row < (int)p.ne01) {
        dst[r1*p.ne0 + im*p.ne0*p.ne1 + row] = tot;
    }
}
