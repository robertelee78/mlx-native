// quantized_matmul_id_mm.metal — MoE-routed GGML-quantized matrix-matrix kernels.
//
// Implements the `kernel_mul_mm_id_<qtype>_f32` kernel shape and its
// preprocessing helper `kernel_mul_mm_id_map0_ne20_<N>` for mlx-native.
//
// Why mm_id?  The MoE variant of quantized_matmul needs to run the same
// expert weight tile against many tokens that routed to that expert.  The
// existing `kernel_mul_mv_id_<qtype>_f32` re-reads each expert's weight
// blocks once per routed (token, slot) pair.  The mm_id variant stages a
// 64x32 tile of the expert's weights into threadgroup shared memory once
// and reuses it across a 32-row block of the expert's routed tokens —
// the same win as the dense mm kernel, but per-expert.
//
// Two-stage dispatch:
//
//   1. `kernel_mul_mm_id_map0_ne20_8` — scans the (token, slot) -> expert
//       id table, for each expert builds a contiguous list of the packed
//       `(token_idx * ne20 + slot_idx)` values that routed to it, plus a
//       per-expert count.  Scratch buffers: `hids` `[n_experts, n_tokens]`
//       (int32, row-major), `htpe` `[n_experts]` (uint32).
//
//   2. `kernel_mul_mm_id_<qtype>_f32` — dispatched with one Z-per-expert.
//       Each threadgroup owns one (N-tile, M-tile, expert_id) and short-
//       circuits when the expert's routed-token count is below its M-tile
//       start.  Inside a tile, ALL 32 M-rows belong to the same expert,
//       so the 64x32 weight tile loaded into shmem is valid for every row.
//
// The preprocessing step is the key to making mm profitable at MoE:
// without it, 32 consecutive output rows in a tile could route to 32
// different experts, defeating weight reuse.
//
// Port rules identical to the dense mm port (see quantized_matmul_mm.metal
// for detail).  Both kernels are bit-compatible with the reference source;
// output tolerance-level matches with the existing mv_id kernel.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK4_0 32
#define QK8_0 32
#define QK_K  256

#define QK_NL 16

// ---- Params for map0 ----
//
// Maps the reference map0 kargs layout.  We omit `ne02` (== n_experts,
// used for a
// threadgroup-size assert upstream — we pass it host-side and dispatch
// exactly `n_experts` threads per threadgroup to mirror the upstream
// `ntg = n_experts` launch).

struct GgmlMatmulIdMm_Map0Params {
    int32_t  ne10;    // unused, kept for struct symmetry
    int32_t  ne11;    // n_expert_used (bcast), always == ne20 in our case
    uint64_t nb11;    // unused
    uint64_t nb12;    // unused
    int32_t  ne21;    // n_tokens
    int32_t  ne20;    // n_expert_used (== top_k)
    uint64_t nb21;    // bytes per token in the source ids table
                      //  (= ne20 * sizeof(int32_t) for our layout)
};

// ---- Params for mul_mm_id ----
//
// Maps the reference mm_id kargs layout.

struct GgmlMatmulIdMm_MmParams {
    int32_t  ne00;   // K
    int32_t  ne02;   // n_experts
    uint64_t nb01;   // bytes per weight row (within one expert)
    uint64_t nb02;   // bytes per expert weight slab (== nb01 * ne01)
    uint64_t nb03;   // unused
    int32_t  ne11;   // n_expert_used (bcast, == ne20)
    uint64_t nb10;   // = sizeof(float) = 4
    uint64_t nb11;   // bytes per input row = K * sizeof(float)
    uint64_t nb12;   // bytes per input batch (n_tokens * nb11)
    uint64_t nb13;   // unused
    int32_t  ne20;   // n_expert_used (== top_k)
    int32_t  ne21;   // n_tokens
    int32_t  ne0;    // N (per-expert output rows)
    int32_t  ne1;    // batch (always == ne21 * ne20 in our layout)
    int16_t  r2;     // 1
    int16_t  r3;     // 1
    int16_t  _pad0;
    int16_t  _pad1;
};

// ---- block_q struct layouts (byte-for-byte GGUF) ----

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    half    d;
    uint8_t qh[4];
    uint8_t qs[QK4_0 / 2];
} block_q5_0;

typedef struct {
    half   d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    half    d;
    half    dmin;
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    half    d;
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(half) + QK_K/4 + QK_K/8 + 12,
              "wrong q3_K block size");

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;

// ADR-013 P16 — Q4_K block (144 bytes) for mm_id port.
// Layout: [half d][half dmin][uint8_t scales[12]][uint8_t qs[128]]
#define K_SCALE_SIZE 12
typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

// ADR-022 Phase 2 — Q5_K block (176 bytes) for mm_id port.
// Layout: [half d][half dmin][uint8_t scales[12]][uint8_t qh[32]][uint8_t qs[128]]
// Same as Q4_K plus a 32-byte qh "high-bit" array. Mirrors
// quantized_matmul_id_ggml.metal:75 lock-step.
typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

// ADR-022 Phase 1 — Q5_1 / IQ4_NL block typedefs for the mm_id port.
typedef struct {
    half    d;
    half    m;
    uint    qh;
    uint8_t qs[QK4_0 / 2];
} block_q5_1;

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_iq4_nl;

// ADR-033 §Pi Task #20 — IQ4_XS block (136 bytes) for mm_id port.
// Layout: [half d][uint16_t scales_h][uint8_t scales_l[4]][uint8_t qs[128]]
// QK_K = 256 elements per super-block, organized as 8 sub-blocks of 32
// elements each. Each sub-block has a 6-bit signed scale (offset by 32)
// packed across scales_l (low 4 bits each, 2 per byte) + scales_h (top
// 2 bits each, 2 bits per sub-block in a uint16_t). Per-quant nibbles
// in qs are 4-bit codebook indexes into `kvalues_iq4nl` (shared with
// IQ4_NL). Lock-step with quantized_matmul_id_ggml.metal:139.
typedef struct {
    half     d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;

// IQ4_NL non-linear codebook (frozen by ggml-common.h:1109-1112).
// Lock-step with the duplicates in quantized_matmul_id_ggml.metal,
// quantized_matmul_ggml.metal, and the host-side `KVALUES_IQ4_NL`
// in src/gguf/mod.rs.
constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// Spec source: reference `get_scale_min_k4_just2`.
// Decodes the (sub-block scale, sub-block min) 6-bit pair at index `j`
// (within sub-block group `k`) from the packed K_SCALE_SIZE=12 array.
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// ---- Dequantize helpers (identical to quantized_matmul_mm.metal) ----

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint qh = *((device const uint *)xb->qh);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i    )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i + 1)) << gh_bk) & 0x10;
        const int x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }
    reg = (type4x4)reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }
    reg = (type4x4) reg_f;
}

// Q2_K dequantize, pinned to a fixed upstream revision.
template <typename type4x4>
void dequantize_q2_K(device const block_q2_K * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    const uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    const half coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uchar mask = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const float dl = d * (sc & 0xF) * coef;
    const float ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = xb->qs;
    device const uint8_t * h = xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;
    q += 32*(il/8) + 16*(il&1);
    h += 16*(il&1);
    const uint8_t m = 1 << (il/2);
    const uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : ((il/4)>0 ? 12 : 3);
    const uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    const uint16_t scale_2 = scales[il%8];
    const uint16_t scale_1 = scales[8 + il%4];
    const int16_t dl_int = (il/4)&1
        ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
        : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all*(dl_int - 32.f) : d_all*(dl_int/16.f - 32.f);
    const float ml = 4.f*dl;
    il = (il/2)&3;
    const half coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    dl *= coef;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl*(q[i]&mask) - (h[i]&m ? 0.f : ml);
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;

    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg_f[i][0] = dl0 *  ((half)(q & 0xFF))      - ml;
        reg_f[i][1] = dl1 * ((float)(q & 0xFF00))    - ml;
        reg_f[i][2] = dl2 * ((float)(q & 0xFF0000))  - ml;
        reg_f[i][3] = dl3 * ((float)(q & 0xFF000000))- ml;
    }
    reg = (type4x4) reg_f;
}

// ADR-022 Phase 1 — Q5_1 dequant for the mm_id MMA-tile path.
// Spec source: reference `dequantize_q5_1`.
// Fills a 4x4 tile (16 values) with positions [16*il .. 16*il+16) of
// the 32-element block. il=0 fills low half, il=1 fills high half.
template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = xb->qh;
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }
    reg = (type4x4) reg_f;
}

// ADR-022 Phase 1 — IQ4_NL dequant for the mm_id MMA-tile path.
// Spec source: reference `dequantize_iq4_nl`.
// 16 elements per call addressed by il ∈ {0, 1}; loops over 4 uint16
// chunks of qs, extracting 4 4-bit indices each via shift+mask, looking
// up into kvalues_iq4nl per nibble.
template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * (float)kvalues_iq4nl[q8[0]];
        reg[i][1] = d * (float)kvalues_iq4nl[q8[1]];
        reg[i][2] = d * (float)kvalues_iq4nl[q8[2]];
        reg[i][3] = d * (float)kvalues_iq4nl[q8[3]];
    }
}

// ADR-033 §Pi Task #20 — IQ4_XS float codebook table.
// Identical values to `kvalues_iq4nl` above but pre-converted to float
// to match the reference `kvalues_iq4nl_f` representation byte-for-byte.
// Used by `dequantize_iq4_xs` below — using the int8 table + cast was
// producing numerical divergence at top_k=8 mm_id batch shapes, so the
// safer mirror of canonical is to emit a float table directly.
constant float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// ADR-033 §Pi Task #20 — IQ4_XS dequant for the mm_id MMA-tile path.
// Spec source: reference `dequantize_iq4_xs`
// — reproduced verbatim modulo formatting.
//
// 16 elements per call addressed by `il ∈ [0,16)`:
//   - `ib32 = il/2` selects the 32-element sub-block (0..7) in qs;
//   - `il%2` selects which 16-elem half (low/high nibble) within that
//     sub-block.
//
// Per-sub-block 6-bit signed scale is reconstructed from 4 low bits in
// `scales_l[ib32/2]` (nibble-packed, 2 sub-blocks per byte) plus the top
// 2 bits in `scales_h` (2 bits per sub-block in a uint16_t). Scale value
// is `xb->d * (ls - 32)` — the offset of 32 is the signed-conversion.
template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0..15 for QK_K = 256 — index of 32-elem sub-block is il/2
    const int ib32 = il / 2;
    il = il % 2;
    // il = 0 or 1: il=0 → first 16 quants of the 32-elem sub-block,
    //              il=1 → second 16 quants.
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4 * ib32;
    const int ls = ((xb->scales_l[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)
                 | (((xb->scales_h >> 2 * ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4 * il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

// ADR-013 P16 — Q4_K dequant for the mm_id MMA-tile path.
// Spec source: reference `dequantize_q4_K`. Fills a 4x4 tile with the
// 16 dequantized values addressed by `il` (which selects sub-block + half
// nibble), using the same scale/min decode as the mat-vec-id Q4_K kernel
// already shipped at quantized_matmul_id_ggml.metal.
template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl  = d * sc[0];
    const float ml  = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// ADR-022 Phase 2 — Q5_K dequant for mm_id MMA-tile path.
// Spec source: reference `dequantize_q5_K`.
// Q5_K differs from Q4_K by adding a 32-byte qh "high-bit" array; each
// of the 16 elements per call OR's an extra 16 (low half) or 256 (high
// half) into the dequantized value when the corresponding qh bit is set.
template <typename type4x4>
void dequantize_q5_K(device const block_q5_K * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl  = d * sc[0];
    const float ml  = min * sc[1];

    const ushort mask  = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

// ====================================================================
// map0 — per-expert routed-token list builder
// ====================================================================
//
// Input:  src2 — per-token expert ids `[n_tokens, n_expert_used]` int32.
// Output: htpe — per-expert routed count `[n_experts]` uint32. Bit 31 is
//                set in every entry when any token has an out-of-range or
//                duplicate expert ID; consumers then poison the operation.
// Output: hids — per-expert routed-token list `[n_experts, n_tokens]` int32.
//         Each slot holds `(token_idx * ne20 + slot_idx)` packed.
//
// Dispatch geometry:
//   threadgroups = (1, 1, 1)
//   threads_per_threadgroup = (n_experts, 1, 1)     (ntg == ne02)
//
// One thread per expert id (ide = tpitg).  Threads scan the token axis
// in blocks of `ntg` tokens, stage each token's `ne20` expert-ids into
// threadgroup shmem, then each thread scans that block looking for its
// own expert-id and appends entries into its slot in `hids`.

template<short ne20>
kernel void hf2q_mul_mm_id_map0_impl(
        constant GgmlMatmulIdMm_Map0Params & args [[buffer(0)]],
        device const char * src2 [[buffer(1)]],
        device       char * htpe [[buffer(2)]],
        device       char * hids [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg;    // expert id owned by this thread

    uint32_t n_all = 0;
    bool routing_invalid = false;
    device int32_t * ids_i32 = (device int32_t *) hids + ide * args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const uint32_t * src2_u32 =
                (device const uint32_t *) (src2 + (i21 + tpitg) * args.nb21);
            threadgroup uint32_t * sids = (threadgroup uint32_t *) shmem + tpitg * ne20;
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_u32[i20];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) break;
            threadgroup const uint32_t * sids =
                (threadgroup const uint32_t *) shmem + t * ne20;

            bool token_invalid = false;
            for (short i20 = 0; i20 < ne20; i20++) {
                token_invalid |= sids[i20] >= ntg;
                for (short other = 0; other < i20; other++) {
                    token_invalid |= sids[i20] == sids[other];
                }
            }
            routing_invalid |= token_invalid;
            if (token_invalid) {
                continue;
            }

            short selected_slot = -1;
            for (short i20 = 0; i20 < ne20; i20++) {
                if (sids[i20] == ide) {
                    selected_slot = i20;
                    break;
                }
            }
            if (selected_slot >= 0) {
                ids_i32[n_all++] = (i21 + t) * ne20 + selected_slot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all | (routing_invalid ? 0x80000000u : 0u);
}

// Gemma 4 uses top_k = 8 for the MoE gate_up call and top_k = 1 for the
// MoE down call (where the gate_up output is flattened to
// [seq_len*top_k] rows, each one routed to a single expert via the
// expert-id table).  Both must hit the mm_id path — without ne20_1, the
// down call falls back to mv_id and re-reads each expert's weights once
// per (seq_len * top_k) row, defeating the whole point of mm.
template [[host_name("kernel_mul_mm_id_map0_ne20_1")]]
kernel void hf2q_mul_mm_id_map0_impl<1>(
    constant GgmlMatmulIdMm_Map0Params &,
    device const char *, device char *, device char *,
    threadgroup char *, ushort, ushort);

template [[host_name("kernel_mul_mm_id_map0_ne20_6")]]
kernel void hf2q_mul_mm_id_map0_impl<6>(
    constant GgmlMatmulIdMm_Map0Params &,
    device const char *, device char *, device char *,
    threadgroup char *, ushort, ushort);

template [[host_name("kernel_mul_mm_id_map0_ne20_8")]]
kernel void hf2q_mul_mm_id_map0_impl<8>(
    constant GgmlMatmulIdMm_Map0Params &,
    device const char *, device char *, device char *,
    threadgroup char *, ushort, ushort);

// ====================================================================
// mul_mm_id — MoE matmul using the map0-produced expert token lists
// ====================================================================
//
// Dispatch geometry:
//   threadgroups = (ceil(neh1/32), ceil(N/64), n_experts)
//   threads_per_threadgroup = (128, 1, 1)
//
// where `neh1 = htpe[expert]` (count of tokens routed to this expert).
//
// Each threadgroup owns:
//   * im       = tgpig.z     — expert index
//   * r0-block = tgpig.y     — output-N tile index
//   * r1-block = tgpig.x     — input-M (per-expert) tile index
//
// The kernel short-circuits inactive (expert, r1) combinations via the
// `r1 >= neh1` check — some threadgroups may exit early but the grid
// launch is uniform.
//
// Per-iteration of the K loop: stage a 64x32 A-tile (from the expert's
// weight slab) + 32x32 B-tile (from input rows selected via hids) into
// threadgroup memory, then run 8 simdgroup MMA ops per simdgroup.  Four
// simdgroups cooperate on one 64x32 output tile.
//
// Write-back staging ALWAYS goes through shmem because consecutive M-rows
// in this tile belong to the same expert but may correspond to different
// (token, slot) output rows (determined by hids[im * ne21 + r1 + j]).

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void hf2q_mul_mm_id_impl(
        constant GgmlMatmulIdMm_MmParams & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device const char * htpe [[buffer(3)]],
        device const char * hids [[buffer(4)]],
        device       char * dst  [[buffer(5)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;  // 2
    constexpr int NL1 = NK/8;   // 4

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const uint32_t route_state = tpe_u32[im];
    if ((route_state & 0x80000000u) != 0u) {
        // Invalid IDs poison the whole operation. Exactly one threadgroup
        // fills the complete logical output with NaN; every threadgroup exits
        // before constructing an expert weight or activation address. The
        // consuming graph is responsible for rejecting this sentinel at an
        // existing host-visible result readback.
        if (im == 0 && tgpig.x == 0 && tgpig.y == 0) {
            device float * out = (device float *) dst;
            const uint64_t total =
                (uint64_t)args.ne21 * (uint64_t)args.ne1 * (uint64_t)args.ne0;
            for (uint64_t index = tiitg; index < total; index += 128) {
                out[index] = NAN;
            }
        }
        return;
    }
    const int32_t neh1 = (int32_t)route_state;

    // Early exit: this expert has fewer routed tokens than our tile's
    // M-base.  Whole threadgroup returns.
    if (r1 >= neh1) return;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    // (token_idx, slot_idx) of the row this thread owns.
    const int id = ids_i32[im * args.ne21 + r1 + lr1];
    const short i11 = (id % args.ne20) % args.ne11;   // slot index
    // §0.19 FIX (2026-06-30): token/row index must be `int`, not `short`.
    // The flat [n_tokens, K] input layout makes i12 = id/ne20 range up to
    // n_tokens-1 (65535 at 8k, 131071 at 16k for the MoE down call where
    // n_tokens = seq_len*top_k). `short` overflows at 32768 → negative
    // i12 → `args.nb12*i12` wraps to an OOB device address (read of
    // run-varying memory). llama keeps this index small via its
    // [K, n_expert_used, n_tokens] multi-dim layout; our flat port must
    // widen it. Bit-identical for n_tokens<=32768 (parity gate).
    const int   i12 = (id / args.ne20);               // token index
    const short i13 = 0;

    // Base of expert `im`'s weight slab.
    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x =
        (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*(uint64_t)i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- A tile dequantize + stage ----
        {
            half4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ADR-033 §Pi Task #20 iter 8 — added #pragma unroll to match
            // the peer's FOR_UNROLL for the 16-iter A-tile staging loop.
            #pragma clang loop unroll(full)
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // ---- B tile stage ----
        //
        // ADR-033 §Pi Task #20 iter 8 — vectorized B-tile store fast path
        // matching the reference `*(threadgroup S1_2x4 *)(sb + ...) = (S1_2x4) *((device T1_2x4 *) y)`.
        // Each thread does ONE 8-float store instead of 8 scalar stores
        // when the K-loop iteration is fully in-bounds. For Q6_K with
        // K%QK_K==0 and NK=32, K is always divisible by NK, so the fast
        // path is taken every iteration (no tail). Kept the scalar
        // bounds-checked fallback for non-Q6_K shapes.
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            if (loop_k + NK <= args.ne00) {
                // Fast path: vectorized 8-float store (= S1_2x4 / float8).
                threadgroup float * dst_row = sb + 64*ib + 8*ly;
                device const float * src_row = (device const float *) y;
                ((threadgroup float4 *) dst_row)[0] = ((device const float4 *) src_row)[0];
                ((threadgroup float4 *) dst_row)[1] = ((device const float4 *) src_row)[1];
            } else {
                // Tail: scalar with bounds check.
                #pragma clang loop unroll(full)
                for (short i = 0; i < 8; ++i) {
                    const short lx = i;
                    *(sb + 64*ib + 8*ly + lx) =
                        (loop_k + iy + i < args.ne00) ? *((device float *) y + i) : 0.f;
                }
            }
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA accumulate ----
        threadgroup const half  * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const float * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // ---- Write-back (always through shmem staging: see header note) ----
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
        + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup `sgitg` strides through the tile in steps of 4 rows.
    // For each owned row j (within the tile), look up its (token, slot)
    // via hids, compute the destination address in dst, and copy.
    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;  // slot index (output row-major)
        // §0.19 FIX (2026-06-30): output token/batch index must be `int` —
        // same overflow as i12 above. For the MoE down call idt = id ranges
        // to n_tokens-1 (65535 at 8k); `short` overflows at 32768 → negative
        // idt → the `dst + idt*ne1*ne0` write-back wraps to an OOB device
        // address (corrupting/reading run-varying memory). Bit-identical for
        // n_tokens<=32768.
        const int   idt = id / args.ne20;  // token index (output batch)

        // §0.19 FIX (2026-06-30, codex SHIP-WITH-CHANGES): force the output
        // ELEMENT offset to 64-bit (promote each factor BEFORE multiply) so the
        // dst address never overflows int32 at large ne0/ne1/token counts —
        // mirrors the int32 dst-offset fix in dense_mm_bf16_tensor.metal.
        // Byte-identical for in-range values.
        const uint64_t dst_off =
            (uint64_t) r0
            + (uint64_t) ide * (uint64_t) args.ne0
            + (uint64_t) idt * (uint64_t) args.ne1 * (uint64_t) args.ne0;
        device float  * D  = (device float  *) dst + dst_off;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

// Template instantiations for the three quant types our GGUF uses.

template [[host_name("kernel_mul_mm_id_q4_0_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q4_0, 2, dequantize_q4_0>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q5_0_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q5_0, 2, dequantize_q5_0>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q8_0_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q8_0, 2, dequantize_q8_0>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q2_K_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q2_K, QK_NL, dequantize_q2_K>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q3_K_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q3_K, QK_NL, dequantize_q3_K>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q6_K_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q6_K, QK_NL, dequantize_q6_K>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-013 P16 — Q4_K mm_id template instantiation.
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q4_K, QK_NL, dequantize_q4_K>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-022 Phase 2 — Q5_K mm_id template instantiation.
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q5_K, QK_NL, dequantize_q5_K>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-022 Phase 1 P1.6 — Q5_1 / IQ4_NL mm_id template instantiations.
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q5_1, 2, dequantize_q5_1>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-033 §Pi Task #20 — IQ4_XS mm_id template instantiation.
// Matches the reference `kernel_mul_mm_id_iq4_xs_f32`
// (parameters: block_iq4_xs + QK_NL=16
// + dequantize_iq4_xs).
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]
kernel void hf2q_mul_mm_id_impl<block_iq4_xs, QK_NL, dequantize_iq4_xs>(
        constant GgmlMatmulIdMm_MmParams & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device const char * htpe [[buffer(3)]],
        device const char * hids [[buffer(4)]],
        device       char * dst  [[buffer(5)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]);

template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]
kernel void hf2q_mul_mm_id_impl<block_iq4_nl, 2, dequantize_iq4_nl>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ============================================================================
// ADR-033 §Pi Task #20 debug kernel — dequantize_iq4_xs dump
// ============================================================================
//
// Standalone diagnostic kernel that calls `dequantize_iq4_xs` at all
// 16 il values for ONE block_iq4_xs and writes the dequantized output
// (256 floats) to a buffer. Lets a Rust test compare the kernel's
// dequant output against the canonical CPU reference
// (`test_only_dequantize_iq4_xs`).
//
// Layout: dst[il * 16 + j] for il ∈ [0,16), j ∈ [0,16) = 256 floats.
// Each `il` call produces 16 elements in row-major reg[i/4][i%4] order,
// matching how the mm_id template stages them via the same indexing.
//
// Dispatch: one threadgroup, one thread. We're not benchmarking — just
// reading the dequant output for parity vs CPU.
kernel void hf2q_dequant_iq4_xs_dump(
        device const block_iq4_xs * src [[buffer(0)]],
        device float * dst [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    for (short il = 0; il < 16; ++il) {
        half4x4 reg;
        dequantize_iq4_xs(src, il, reg);
        for (int i = 0; i < 16; ++i) {
            dst[il * 16 + i] = (float)reg[i / 4][i % 4];
        }
    }
}
