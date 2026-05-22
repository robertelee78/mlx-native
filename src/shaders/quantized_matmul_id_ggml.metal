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

// K_SCALE_SIZE: bytes used for scales+mins in Q4_K and Q5_K super-blocks.
#define K_SCALE_SIZE 12

// ---- GGML block struct definitions (byte-for-byte with GGUF) ----

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    half   d;
    int8_t qs[QK8_0];
} block_q8_0;

// Q5_K: 256 values per block, 176 bytes per block.
// Layout: [half d][half dmin][uint8_t scales[12]][uint8_t qh[32]][uint8_t qs[128]]
typedef struct {
    half    d;                    // super-block scale for quantized scales
    half    dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit   (32 bytes)
    uint8_t qs[QK_K/2];           // quants, low 4 bits (128 bytes)
} block_q5_K;

// Q4_K: 256 values per block, 144 bytes per block.
// Layout: [half d][half dmin][uint8_t scales[12]][uint8_t qs[128]]
//   Structurally Q5_K minus the 32-byte qh "high-bit" array.
//   Same K_SCALE_SIZE=12 layout for packed (sub-scale, sub-min) 6-bit pairs.
//
// Source: ggml-common.h block_q4_K (llama.cpp).
typedef struct {
    half    d;                    // super-block scale for quantized scales
    half    dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // quants, low 4 bits (128 bytes)
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;

// Q5_1: 32 values per block, 24 bytes per block.
// Layout: [half d][half m][uint qh][uint8_t qs[16]]
// Per-element: x[i]      = (qs[i] & 0x0F) | (((qh >> i) & 1) << 4)
//              x[i+16]   = (qs[i] >> 4)   | (((qh >> (i+16)) & 1) << 4)
//              out[i]    = d * x[i]      + m
//              out[i+16] = d * x[i+16]   + m
// ADR-022 Phase 1.
typedef struct {
    half    d;
    half    m;
    uint    qh;
    uint8_t qs[QK4_0 / 2];  // 16 bytes — same QK as Q4_0
} block_q5_1;

// IQ4_NL: 32 values per block, 18 bytes per block.
// Layout: [half d][uint8_t qs[16]]
// Each qs byte holds 2 4-bit codebook indices into kvalues_iq4nl[16].
// Per-element: out[i]    = d * kvalues_iq4nl[qs[i] & 0x0F]
//              out[i+16] = d * kvalues_iq4nl[qs[i] >> 4]
// ADR-022 Phase 1.
typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];  // 16 bytes
} block_iq4_nl;

// IQ4_NL non-linear codebook (frozen by llama.cpp ggml-common.h:1109-1112).
// Any drift breaks every IQ4_NL GGUF on disk; do not edit without
// updating the host-side `KVALUES_IQ4_NL` in src/gguf/mod.rs in lock-step.
constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// IQ4_XS (ADR-033 §Pi 2026-05-22): super-block 4-bit codebook quant.
// 256 elements per super-block, 8 × 32-element sub-blocks each with a
// 6-bit signed scale (4-bit scales_l + 2-bit scales_h). 136 bytes.
// Reference: ggml-common.h:444-450 `block_iq4_xs`.
typedef struct {
    half     d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];  // 4
    uint8_t  qs[QK_K / 2];         // 128
} block_iq4_xs;

// ---- Q5_1 dot product helper (ADR-022 Phase 1) ----
//
// Mirrors `block_q_n_dot_y<block_q5_1>` from llama.cpp
// (ggml-metal.metal:3293-3310). Q5_1 differs from Q4_0 by carrying
// (a) an additional `m` (min) term contributing `m * sumy` to the dot,
// and (b) a 5th high-bit per element packed in `qh`.
//
// `il` is 0 or 8 (mlx-native convention: byte offset within the
// 16-byte qs array). The yl[] vector is pre-scaled by the caller
// (yl[i+1] /= 256, yl[i+8] /= 16, yl[i+9] /= 4096) so that masking
// nibbles via 0x000F / 0x0F00 / 0x00F0 / 0xF000 yields the correct
// sub-element-weighted partial sums.

inline float block_q5_1_dot_y(
    device const block_q5_1 * qb,
    float sumy,
    thread float * yl,
    int il
) {
    float d = qb->d;
    float m = qb->m;

    float4 acc = 0.f;

    // qs is 16 bytes starting at offset 8 in the block.
    // Cast block as uint16_t* to skip d (1 uint16) + m (1 uint16) +
    // qh (2 uint16) = 4 uint16; then add il/2 to land at the right
    // qs sub-region.
    device const uint16_t * qs = ((device const uint16_t *)qb + 4 + il/2);
    const uint qh = qb->qh;

    for (int i = 0; i < 8; i += 2) {
        // Low nibbles, sub-positions i + 0 and i + 1 (within block 0..15).
        // qh bit (i+0+il) contributes the 5th bit, placed at position 4
        // (mask 0x10) for the 0x000F-masked nibble.
        // qh bit (i+1+il) contributes the 5th bit, placed at position 12
        // (mask 0x1000) for the 0x0F00-masked nibble.
        acc[0] += yl[i + 0]
                * (float)((qs[i / 2] & 0x000F) | (((qh >> (i + 0 + il      )) << 4 ) & 0x0010));
        acc[1] += yl[i + 1]
                * (float)((qs[i / 2] & 0x0F00) | (((qh >> (i + 1 + il      )) << 12) & 0x1000));
        // High nibbles, sub-positions i + 8 and i + 9 (within block 16..31).
        // qh bit (i+0+il+16) for nibble at mask 0x00F0 → bit at 0x0100.
        // qh bit (i+1+il+16) for nibble at mask 0xF000 → bit at 0x10000.
        acc[2] += yl[i + 8]
                * (float)((qs[i / 2] & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8 ) & 0x0100));
        acc[3] += yl[i + 9]
                * (float)((qs[i / 2] & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// ---- IQ4_NL dot product helper (ADR-022 Phase 1) ----
//
// IQ4_NL is a 4-bit codebook quant: each qs nibble selects one of
// 16 entries from `kvalues_iq4nl`. There is no zero-point bias and no
// `m` term. The output formula is purely `out[i] = d * kvalues_iq4nl[idx]`.
//
// Caller convention is identical to Q4_0 / Q5_1: yl[] holds the
// pre-scaled input row, but for IQ4_NL the divisors used by Q4_0
// (`/256`, `/16`, `/4096`) do NOT compose because the codebook lookup
// is non-linear. This helper therefore reads the raw input via a
// caller-supplied `yl_raw[]` array of 16 unscaled values, multiplies
// by the codebook entry directly, and returns `d * sum`.

inline float block_iq4_nl_dot_y(
    device const block_iq4_nl * qb,
    thread float * yl_raw,
    int il
) {
    float d = qb->d;
    float acc = 0.f;

    // qs starts at byte 2 in the block; 16 bytes total.
    device const uint8_t * qs = qb->qs + il;

    for (int i = 0; i < 8; i++) {
        const uint8_t b = qs[i];
        const int lo = b & 0x0F;
        const int hi = (b >> 4) & 0x0F;
        // First half: position i within sub-region [il .. il+8).
        acc += yl_raw[i] * (float)kvalues_iq4nl[lo];
        // Second half: position i + 16 (the high half of the block).
        acc += yl_raw[i + 8] * (float)kvalues_iq4nl[hi];
    }

    return d * acc;
}

// IQ4_XS dot helper — see comment in quantized_matmul_ggml.metal for full
// docs. Processes ONE 32-element sub-block of a super-block.
// `ib32` ∈ [0,7] selects sub-block; `il` ∈ {0, 8} selects which 8-element
// half of the sub-block to process.
inline float block_iq4_xs_dot_y(
    device const block_iq4_xs * qb,
    thread float * yl_raw,
    int ib32,
    int il
) {
    const float d = qb->d;
    const int ls = ((qb->scales_l[ib32 / 2] >> 4*(ib32 % 2)) & 0xf)
                 | (((qb->scales_h >> 2*ib32) & 0x3) << 4);
    const float dl = d * (float)(ls - 32);
    float acc = 0.f;
    device const uint8_t * qs = qb->qs + 16*ib32 + il;
    for (int i = 0; i < 8; i++) {
        const uint8_t b = qs[i];
        acc += yl_raw[i]     * (float)kvalues_iq4nl[b & 0x0F];
        acc += yl_raw[i + 8] * (float)kvalues_iq4nl[(b >> 4) & 0x0F];
    }
    return dl * acc;
}

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
//
// Routing index is in dim Y, NOT dim Z (despite llama.cpp's mul_mv_id grid
// using ne123 in z at ggml-metal-ops.cpp:2452).  Tested 2026-04-26 on M5 Max
// dwq46 64-token decode: switching kernel to read tgpig.z + dispatcher
// MTLSize::new(N/8, 1, m) regressed throughput from 114 t/s to 90.9 t/s
// (-19%).  Apple GPU's threadgroup scheduler distributes this dispatch
// shape better via Y than Z — 7th confirmed static-evidence kernel
// hypothesis falsified per `project_metal_compiler_auto_optimizes_static_levers.md`.

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
// Q4_0 fused-SwiGLU expert-indexed mat-vec kernel
// ====================================================================
//
// Computes: dst[r][n] = sum_k(dequant(W_q4_0[expert_id][n][k])
//                              * (silu(gate[r][k]) * up[r][k]))
//
// where r = token*top_k + slot, expert_id = ids[r].
//
// Replaces the dispatch sequence:
//   silu_mul_f32(gate, up → h_all)        # 1 dispatch + memory_barrier
//   kernel_mul_mv_id_q4_0_f32(W, h_all → dst)  # 1 dispatch
//
// with a single dispatch that reads gate + up directly and computes
// swiglu inline before the dot product.  Closes ~5-10µs/layer × 40
// layers ≈ 0.3-0.4ms/token of CPU dispatch overhead in the dwq46
// decode hot path (ADR-012 §Optimize / Task #15).
//
// Buffer layout:
//   buffer(0): src0   - Q4_0 packed weight, [n_experts, N, K/QK4_0] blocks
//   buffer(1): gate   - f32 [n_tokens*top_k, K]
//   buffer(2): up     - f32 [n_tokens*top_k, K]
//   buffer(3): dst    - f32 [n_tokens*top_k, N]
//   buffer(4): ids    - u32 [n_tokens*top_k]
//   buffer(5): params - GgmlMatvecIdParams
//
// Dispatch geometry: identical to kernel_mul_mv_id_q4_0_f32 —
// threadgroups=(ceil(N/8), n_tokens*top_k, 1), tg=(8, 8, 1).
kernel void kernel_mul_mv_id_q4_0_f32_swiglu(
    device const  char  * src0   [[buffer(0)]],
    device const float  * gate   [[buffer(1)]],
    device const float  * up     [[buffer(2)]],
    device       float  * dst    [[buffer(3)]],
    device const  uint  * ids    [[buffer(4)]],
    constant GgmlMatvecIdParams & p [[buffer(5)]],
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

    if (output_row >= (int)p.ne1) return;

    // For expert_down in the decode-time MoE pipeline, the input row IS
    // the output row index (one h_all row per (token, expert_slot) pair),
    // not token_idx.  Each (token, expert_slot) has its own gate/up vectors.
    const uint expert_id = ids[output_row];
    const uint input_row = output_row;  // gate/up are pre-routed per (token, slot).

    const int first_row = (r0 * nsg + sgitg) * nr;

    // Expert's weight slice.
    device const block_q4_0 * x = (device const block_q4_0 *)((device const char *)src0 + expert_id * p.expert_stride) + first_row * nb;

    // Per-row gate and up vectors.
    device const float * gate_y = gate + input_row * p.ne10;
    device const float * up_y   = up   + input_row * p.ne10;

    float yl[16];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 2;
    const int il = (tiisg % 2) * 8;

    device const float * gb = gate_y + ix * QK4_0 + il;
    device const float * ub = up_y   + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0;
        // Compute swiglu = silu(gate) * up = gate * sigmoid(gate) * up
        // for each of the 16 active elements per simdthread, reusing the
        // same yl[] / sumy aggregation as the unfused kernel.
        for (int i = 0; i < 8; i += 2) {
            // Lane block 0 (i=0, i+1).
            float g0 = gb[i+0];
            float g1 = gb[i+1];
            float u0 = ub[i+0];
            float u1 = ub[i+1];
            float s0 = (g0 / (1.0f + metal::exp(-g0))) * u0;
            float s1 = (g1 / (1.0f + metal::exp(-g1))) * u1;
            sumy += s0 + s1;
            yl[i+0] = s0;
            yl[i+1] = s1 / 256.f;

            // Lane block 1 (i+16, i+17).
            float g2 = gb[i+16];
            float g3 = gb[i+17];
            float u2 = ub[i+16];
            float u3 = ub[i+17];
            float s2 = (g2 / (1.0f + metal::exp(-g2))) * u2;
            float s3 = (g3 / (1.0f + metal::exp(-g3))) * u3;
            sumy += s2 + s3;
            yl[i+8] = s2 / 16.f;
            yl[i+9] = s3 / 4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q4_0_dot_y(x + ib + row*nb, sumy, yl, il);
        }

        gb += QK4_0 * 16;
        ub += QK4_0 * 16;
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
// Q8_0 _id expert-indexed mat-vec kernel — NR0=2 NSG=4 variant
// (ADR-029 iter-6 port; peer N_R0_Q8_0=2 + N_SG_Q8_0=4 in
//  /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:27,40)
// ====================================================================
//
// Same row coverage as `kernel_mul_mv_id_q8_0_f32` (8 rows/TG) but
// distributes the K-dim across 4 simdgroups with shmem cross-SG reduce
// instead of 2 SGs each independently handling 4 rows.  Better latency
// hiding when memory-bandwidth-bound, which the gemma4 MoE down_exps
// Q8_0 mat-vec hits at hidden_size=2816, top_k=8.
//
// Dispatch geometry:
//   threadgroups = (ceil(N/N_R0_Q8_0=2), n_tokens*top_k, 1)
//   threads_per_tg = (32, 4, 1) = 128 threads = 4 simdgroups × 32
//   shmem = N_R0_Q8_0 * N_SIMDWIDTH * sizeof(float) = 256 bytes
//
// Env-gated via HF2Q_Q8_0_ID_MV_NR2=1 in dispatch_id_mv.

#define N_R0_Q8_0_ID 2
#define N_SG_Q8_0_ID 4
#define NQ_Q8_0_ID   8

kernel void kernel_mul_mv_id_q8_0_f32_nr2(
    device const  char  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    device const  uint  * ids    [[buffer(3)]],
    constant GgmlMatvecIdParams & p [[buffer(4)]],
    threadgroup float   * shmem  [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NR0 = N_R0_Q8_0_ID;   // 2
    constexpr int NSG = N_SG_Q8_0_ID;   // 4
    constexpr int NW  = N_SIMDWIDTH;    // 32
    constexpr int NQ  = NQ_Q8_0_ID;     // 8

    const int nb = p.ne00 / QK8_0;
    const int r0 = tgpig.x;
    const int output_row_base = tgpig.y;

    if (output_row_base >= (int)p.ne1) return;

    const uint token_idx = output_row_base / p.top_k;
    const uint expert_id = ids[output_row_base];

    const int first_row = r0 * NR0;

    // Per-row src0 pointers (unrolled NR0=2 iterations), with expert offset.
    device const block_q8_0 * ax[NR0];
    for (int row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q8_0 *)((device const char *)src0 + expert_id * p.expert_stride) + (first_row + row) * nb;
    }

    device const float * y = src1 + token_idx * p.ne10;

    float sumf[NR0] = { 0.f };

    const int ix = tiisg / (NW / NQ);  // 0..3
    const int il = tiisg % (NW / NQ);  // 0..3

    const int ib0 = sgitg * NQ + ix;

    float yl[NQ];
    device const float * yb = y + ib0 * QK8_0 + il * NQ;

    // Each thread covers NQ quants per iteration; SGs interleave across
    // ib by stride NSG*NQ. Mirrors kernel_mul_mv_q8_0_f32_nr2 (regular).
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

    for (int row = 0; row < NR0 && first_row + row < (int)p.ne01; ++row) {
        const float tot = simd_sum(shmem_rows[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            dst[output_row_base * p.ne0 + first_row + row] = tot;
        }
    }
}

// ====================================================================
// Q5_1 expert-indexed mat-vec kernel  (ADR-022 Phase 1)
// ====================================================================
//
// Same dispatch geometry as Q4_0 / Q8_0 (legacy 32-element formats):
//   threadgroups = (ceil(N/8), n_tokens*top_k, 1), tg = (8, 8, 1)
//   tgpig.x = block-row index
//   tgpig.y = flat output row (token*top_k + slot)
//
// Differs from Q4_0 only in (a) the block-q-n typedef walked, and
// (b) the dot helper used (`block_q5_1_dot_y` carries the m*sumy term
// and qh-bit injection; `block_q4_0_dot_y` does not).
//
// Reference: llama.cpp `mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1>`
// at ggml-metal.metal:3358-3443 + 3293-3310. Inlined here in mlx-native
// style (matching the Q4_0 / Q8_0 ports above) rather than templated.

kernel void kernel_mul_mv_id_q5_1_f32(
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

    // Q5_1 has the same QK as Q4_0 (32 elements per block).
    const int nb = p.ne00 / QK4_0;
    const int r0 = tgpig.x;
    const int output_row = tgpig.y;

    if (output_row >= (int)p.ne1) return;

    const uint token_idx = output_row / p.top_k;
    const uint expert_id = ids[output_row];

    const int first_row = (r0 * nsg + sgitg) * nr;

    // Point to the expert's weight slice. Q5_1 block stride is 24 bytes
    // (vs 18 for Q4_0); the expert_stride byte offset is honored by the
    // host-side dispatcher.
    device const block_q5_1 * x = (device const block_q5_1 *)((device const char *)src0
                                + expert_id * p.expert_stride) + first_row * nb;

    device const float * y = src1 + token_idx * p.ne10;

    float yl[16];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 2;
    const int il = (tiisg % 2) * 8;

    device const float * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0;
        // Pre-scale yl[] for the masked-nibble × scaled-input trick
        // (same as Q4_0 — works because Q5_1's qh-bit injection happens
        // at bit positions {4, 12, 8, 16} that align with the same
        // masks 0x000F / 0x0F00 / 0x00F0 / 0xF000).
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
            dst[output_row * p.ne0 + first_row + row] = tot;
        }
    }
}

// ====================================================================
// IQ4_NL expert-indexed mat-vec kernel  (ADR-022 Phase 1)
// ====================================================================
//
// Same dispatch geometry as Q4_0. IQ4_NL's codebook lookup is
// non-linear, so the masked-nibble × pre-scaled-yl trick that Q4_0 /
// Q5_1 use does not compose. We pass the raw input row to the dot
// helper, which multiplies element-wise by the looked-up codebook
// values.
//
// Reference: llama.cpp `kernel_mul_mv_iq4_nl_f32_impl` at
// ggml-metal.metal (template instantiated at line 10359 via
// kernel_mul_mv_id<mmv_fn<...>>); inlined here in mlx-native style.

// ====================================================================
// IQ4_XS expert-indexed mat-vec kernel (ADR-033 §Pi Task #16 2026-05-22)
// ====================================================================
//
// MoE-routed variant of `kernel_mul_mv_iq4_xs_f32` (see
// quantized_matmul_ggml.metal). Each output_row picks an expert id
// from `ids[output_row]` and reads that expert's weight slab
// (offset = expert_id * expert_stride). Geometry mirrors IQ4_NL's
// mv_id (N_SIMDGROUP=2, N_DST=4, threadgroup=(8,8,1)).
//
// Reference: llama.cpp `kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl>>`
// at ggml-metal.metal:10432. Inlined here in mlx-native style (no
// template metaprogramming).

kernel void kernel_mul_mv_id_iq4_xs_f32(
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

    const int nb = p.ne00 / QK_K; // super-blocks per row
    const int r0 = tgpig.x;
    const int output_row = tgpig.y;

    if (output_row >= (int)p.ne1) return;

    const uint token_idx = output_row / p.top_k;
    const uint expert_id = ids[output_row];

    const int first_row = (r0 * nsg + sgitg) * nr;

    device const block_iq4_xs * x = (device const block_iq4_xs *)((device const char *)src0
                                  + expert_id * p.expert_stride) + first_row * nb;

    device const float * y = src1 + token_idx * p.ne10;

    // tiisg partitions super-block × (sub-block, half) work — see
    // kernel_mul_mv_iq4_xs_f32 comment for derivation.
    const int ix  = tiisg / 16; // 0 or 1
    const int it  = tiisg % 16; // 0..15
    const int ib32 = it / 2;
    const int il   = (it % 2) * 8;

    float yl_raw[16];
    float sumf[nr] = {0.f};

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float * yb = y + ibl * QK_K + ib32 * 32 + il;
        for (int i = 0; i < 8; i++) {
            yl_raw[i]     = yb[i];
            yl_raw[i + 8] = yb[i + 16];
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_iq4_xs_dot_y(x + ibl + row*nb, yl_raw, ib32, il);
        }
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.ne01) {
            dst[output_row * p.ne0 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_id_iq4_nl_f32(
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
    const int output_row = tgpig.y;

    if (output_row >= (int)p.ne1) return;

    const uint token_idx = output_row / p.top_k;
    const uint expert_id = ids[output_row];

    const int first_row = (r0 * nsg + sgitg) * nr;

    device const block_iq4_nl * x = (device const block_iq4_nl *)((device const char *)src0
                                  + expert_id * p.expert_stride) + first_row * nb;

    device const float * y = src1 + token_idx * p.ne10;

    float yl_raw[16];
    float sumf[nr] = {0.f};

    const int ix = tiisg / 2;
    const int il = (tiisg % 2) * 8;

    device const float * yb = y + ix * QK4_0 + il;

    for (int ib = ix; ib < nb; ib += nw/2) {
        // Raw yl[] for IQ4_NL — no pre-scale, codebook lookup is
        // non-linear and would be polluted by the /256, /16, /4096
        // divisors used by the linear-quant helpers.
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
            dst[output_row * p.ne0 + first_row + row] = tot;
        }
    }
}

// ====================================================================
// Q5_K expert-indexed mat-vec kernel
// ====================================================================
//
// Dispatch geometry (same as Q6_K): threadgroups = (ceil(N/2), n_tokens*top_k, 1)
//   tgpig.x = weight-row-pair index        (two rows: 2*r0 + sgitg)
//   tgpig.y = flat output row              (token*top_k + slot)
//   sgitg   = selects which of the two rows this simdgroup processes
//
// Ported from candle-metal-kernels kernel_mul_mv_q5_K_f32_impl with
// the expert-routing indirection from the Q6_K _id kernel above.
// Copyright the candle Authors (Apache-2.0) and llama.cpp Authors (MIT).

kernel void kernel_mul_mv_id_q5_K_f32(
    device const  char  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    device const  uint  * ids    [[buffer(3)]],
    constant GgmlMatvecIdParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nb = p.ne00 / QK_K;

    const int64_t r0          = tgpig.x;
    const int     output_row  = tgpig.y;  // flat: token*top_k + slot

    if (output_row >= (int)p.ne1) return;

    const uint token_idx = output_row / p.top_k;
    const uint expert_id = ids[output_row];

    // Each threadgroup covers weight-row pair (2*r0, 2*r0+1);
    // sgitg selects which row this simdgroup computes.
    const int row = 2 * (int)r0 + (int)sgitg;

    // Point to the expert's weight slice and the token's input row.
    device const block_q5_K * x  = (device const block_q5_K *)((device const char *)src0 + expert_id * p.expert_stride) + row * nb;
    device const float      * yy = src1 + token_idx * p.ne10;

    float sumf = 0.f;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = tiisg / 4;
    const int ix  = tiisg % 4;
    const int iq  = tid / 4;
    const int ir  = tid % 4;
    const int n   = 8;

    const int l0       = n * ir;
    const int q_offset = 32 * iq + l0;
    const int y_offset = 64 * iq + l0;

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
        // scales array is uint8_t[12]; cast to uint16_t[6] for the
        // sc16 decoding identical to the reference candle kernel.
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
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
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
        dst[output_row * p.ne0 + row] = tot;
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

// ====================================================================
// Q6_K _id expert-indexed mat-vec kernel — nr0=2 variant (ADR-028 iter-321)
// ====================================================================
//
// Same as `kernel_mul_mv_id_q6_K_f32` above but processes nr0=2 rows
// per simdgroup with cached yl[16].  4 rows/TG (2 SGs × 2 rows) vs
// baseline 2 rows/TG.  Mirrors peer's
// `template kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K>>>`
// (ggml-metal.metal:10351).
//
// Dispatch: threadgroups=(ceil(N/4), n_tokens*top_k, 1), threads=(2, 32, 1).
// Env-gated via HF2Q_Q6K_ID_MV_NR2=1 in dispatch_id_mv.
kernel void kernel_mul_mv_id_q6_K_f32_nr2(
    device const  char  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    device const  uint  * ids    [[buffer(3)]],
    constant GgmlMatvecIdParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NSG = 2;
    constexpr int nr0 = 2;
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = p.ne00 / QK_K;

    const int64_t r0 = tgpig.x;
    const int output_row_base = tgpig.y;

    if (output_row_base >= (int)p.ne1) return;

    const uint token_idx = output_row_base / p.top_k;
    const uint expert_id = ids[output_row_base];

    const int first_row = (int)((r0 * NSG + sgitg) * nr0);

    device const block_q6_K * x_base = (device const block_q6_K *)((device const char *)src0 + expert_id * p.expert_stride) + first_row * nb;
    device const float      * yy = src1 + token_idx * p.ne10;

    float sumf[nr0] = {0.f, 0.f};
    float yl[16];

    // ADR-028 iter-402: short indexing matches peer's pattern (same as
    // iter-401 applied to the non-_id variant).
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
        // ADR-028 iter-352: explicit FOR_UNROLL pragma test FALSIFIED here too;
        // see non-_id variant (kernel_mul_mv_q6_K_f32_nr2) for the bench data.
        // Apple Metal's auto-unroll is already optimal; explicit hint is removed.
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

    for (int row = 0; row < nr0; ++row) {
        const int out_row = first_row + row;
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && out_row < (int)p.ne01) {
            dst[output_row_base * p.ne0 + out_row] = tot;
        }
    }
}

// ====================================================================
// Q4_K expert-indexed mat-vec kernel
// ====================================================================
//
// ADR-013 P7 — port of llama.cpp `kernel_mul_mv_id_q4_K_f32` (mv_id
// thunk in ggml-metal.metal:10349 wrapping `kernel_mul_mv_q4_K_f32_impl`).
//
// Mirrors the Q5_K mv_id kernel above, differing only in:
//   1. Block struct has no qh field (saves 32 bytes per block).
//   2. The acc2 high-bit accumulators collapse to zero — Q4_K stores
//      only the low 4 bits, no high-bit array.
//   3. The dot-product reduces to (q1[l] & 0x0F) and (q1[l] & 0xF0) >> 4
//      paired with the pre-summed yl/yh/sumy.
//
// Scale-decode is byte-identical to Q5_K: same kmask1/kmask2/kmask3,
// same sc16[] packing — verified against llama.cpp's
// kernel_mul_mv_q4_K_f32_impl at ggml-metal.metal:7727-7729.
//
// Geometry (mirrors Q5_K mv_id):
//   2 simdgroups per threadgroup, 1 row per simdgroup → 2 rows per tg.
//   tgpig.x = weight-row-pair index   (row = 2*r0 + sgitg)
//   tgpig.y = flat output row         (token*top_k + slot)
//
// Routing index dim Y, NOT Z — see the Q5_K kernel's note at line 124.

kernel void kernel_mul_mv_id_q4_K_f32(
    device const  char  * src0   [[buffer(0)]],
    device const float  * src1   [[buffer(1)]],
    device       float  * dst    [[buffer(2)]],
    device const  uint  * ids    [[buffer(3)]],
    constant GgmlMatvecIdParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nb = p.ne00 / QK_K;

    const int64_t r0          = tgpig.x;
    const int     output_row  = tgpig.y;  // flat: token*top_k + slot

    if (output_row >= (int)p.ne1) return;

    const uint token_idx = output_row / p.top_k;
    const uint expert_id = ids[output_row];

    // Each threadgroup covers weight-row pair (2*r0, 2*r0+1);
    // sgitg selects which row this simdgroup computes.
    const int row = 2 * (int)r0 + (int)sgitg;

    device const block_q4_K * x  = (device const block_q4_K *)((device const char *)src0 + expert_id * p.expert_stride) + row * nb;
    device const float      * yy = src1 + token_idx * p.ne10;

    float sumf = 0.f;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

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
        // scales array is uint8_t[12]; cast to uint16_t[6] for the
        // sc16 decoding identical to the reference candle kernel.
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
        dst[output_row * p.ne0 + row] = tot;
    }
}
