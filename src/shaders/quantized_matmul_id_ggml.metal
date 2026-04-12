// Derived from candle-metal-kernels (Apache-2.0) kernel_mul_mv_id template
// and mlx-native's quantized_matmul_ggml kernels.
// Combines GGML block format dequantization with expert-indexed dispatch.
//
// Original sources:
//   candle-metal-kernels/src/metal_src/quantized.metal:7544-7618 (kernel_mul_mv_id)
//   candle-metal-kernels/src/metal_src/quantized.metal:90-293    (Q4_0, Q8_0, Q6_K kernels)
//   mlx-native/src/shaders/quantized_matmul_ggml.metal           (GGML block dequant)
//
// This kernel performs expert-indexed (MoE) quantized matrix-vector multiply:
//   For each (token, slot) pair:
//     expert_id = ids[token * top_k + slot]
//     output[token*top_k + slot, :] = matmul(input[token, :], weight[expert_id])
//
// The key insight: instead of dispatching one kernel per expert, we dispatch once
// for ALL (token, slot) pairs. The kernel uses the ids buffer to route each output
// row to the correct expert's weight slice.
//
// Copyright the candle Authors and llama.cpp Authors.
// See LICENSE-APACHE-candle in this directory.

#include <metal_stdlib>
using namespace metal;

// ---- Constants (must match quantized_matmul_ggml.metal) ----

#define QK4_0 32
#define QK8_0 32
#define QK_K  256

#define N_DST       4
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 32

// ---- Parameters for expert-indexed GGML matmul ----

struct GgmlMatvecIdParams {
    int64_t ne00;           // K: input dimension
    int64_t ne01;           // N: output dimension per expert
    int64_t ne02;           // 1 (unused, kept for struct compat)
    int64_t ne10;           // K: input dimension (redundant, == ne00)
    int64_t ne12;           // 1 (unused)
    int64_t ne0;            // N: output stride
    int64_t ne1;            // total output rows = n_tokens * top_k
    uint    r2;             // 1
    uint    r3;             // 1
    uint    top_k;          // experts per token
    uint    n_tokens;       // number of input tokens
    int64_t expert_stride;  // bytes between expert weight slices
};

// ---- GGML block struct definitions (byte-for-byte with GGUF) ----

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    half   d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;

// ---- Q4_0 dot product helper (identical to quantized_matmul_ggml.metal) ----

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

// ====================================================================
// Q4_0 expert-indexed mat-vec kernel
// ====================================================================
//
// For each output row r (where r = token*top_k + slot):
//   expert_id = ids[r]   (ids is [n_tokens * top_k] flat, pre-expanded)
//   src0_cur  = src0 + expert_id * expert_stride
//   output[r] = matmul(src1[token], src0_cur)
//
// Dispatch geometry: threadgroups=(ceil(N/8), n_tokens*top_k, 1), tg=(8,8,1)

kernel void kernel_mul_mv_id_q4_0_f32(
    device const  char  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    device const  uint  * ids    [[buffer(3)]],
    constant GgmlMatvecIdParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = p.ne00 / QK4_0;
    const int r0 = tgpig.x;
    const int output_row = tgpig.y;  // flat index into output: token*top_k + slot

    // Bounds check
    if (output_row >= (int)p.ne1) return;

    // Determine which token this output row belongs to and which expert
    const uint token_idx = output_row / p.top_k;
    const uint expert_id = ids[output_row];

    const int first_row = (r0 * nsg + sgitg) * nr;

    // Point to the expert's weight slice
    device const block_q4_0 * x = (device const block_q4_0 *)((device const char *)src0 + expert_id * p.expert_stride) + first_row * nb;

    // Point to the input row for this token
    device const float * y = src1 + token_idx * p.ne10;

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
            dst[output_row * p.ne0 + first_row + row] = tot;
        }
    }
}

// ====================================================================
// Q8_0 expert-indexed mat-vec kernel
// ====================================================================

#define NB_Q8_0 8

kernel void kernel_mul_mv_id_q8_0_f32(
    device const  char  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    device const  uint  * ids    [[buffer(3)]],
    constant GgmlMatvecIdParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = p.ne00 / QK8_0;
    const int r0 = tgpig.x;
    const int output_row = tgpig.y;

    if (output_row >= (int)p.ne1) return;

    const uint token_idx = output_row / p.top_k;
    const uint expert_id = ids[output_row];

    const int first_row = (r0 * nsg + sgitg) * nr;

    device const block_q8_0 * x = (device const block_q8_0 *)((device const char *)src0 + expert_id * p.expert_stride) + first_row * nb;
    device const float      * y = src1 + token_idx * p.ne10;

    float yl[NB_Q8_0];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 4;
    const int il = tiisg % 4;

    device const float * yb = y + ix * QK8_0 + NB_Q8_0 * il;

    for (int ib = ix; ib < nb; ib += nw/4) {
        for (int i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < nr; row++) {
            device const int8_t * qs = x[ib + row*nb].qs + NB_Q8_0 * il;
            float sumq = 0.f;
            for (int iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            sumf[row] += sumq * x[ib + row*nb].d;
        }

        yb += NB_Q8_0 * nw;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.ne01) {
            dst[output_row * p.ne0 + first_row + row] = tot;
        }
    }
}

// ====================================================================
// Q6_K expert-indexed mat-vec kernel
// ====================================================================

kernel void kernel_mul_mv_id_q6_K_f32(
    device const  char  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    device const  uint  * ids    [[buffer(3)]],
    constant GgmlMatvecIdParams & p [[buffer(4)]],
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
    const int output_row_base = tgpig.y;

    if (output_row_base >= (int)p.ne1) return;

    // For Q6_K, each threadgroup handles 2 rows (one per SIMD group).
    // But we need to handle the _id dimension: output_row_base is the
    // flat output-row index. We process row = 2*r0 + sgitg within the
    // weight matrix, and the output goes to output_row_base.
    //
    // Actually, the Q6_K dispatch geometry is different from Q4_0/Q8_0.
    // In the non-id version: threadgroups = (ceil(N/2), M, 1)
    // Each threadgroup handles 2 adjacent weight rows (r0*2 + sgitg).
    //
    // For the _id version: threadgroups = (ceil(N/2), n_tokens*top_k, 1)
    // tgpig.y = output_row (flat: token*top_k + slot)
    // tgpig.x = weight row pair index
    // sgitg selects within the pair: row = 2*r0 + sgitg

    const uint token_idx = output_row_base / p.top_k;
    const uint expert_id = ids[output_row_base];

    const int row = 2 * r0 + sgitg;

    device const block_q6_K * x  = (device const block_q6_K *)((device const char *)src0 + expert_id * p.expert_stride) + row * nb;
    device const float      * yy = src1 + token_idx * p.ne10;

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
    if (tiisg == 0 && row < (int)p.ne01) {
        dst[output_row_base * p.ne0 + row] = tot;
    }
}
