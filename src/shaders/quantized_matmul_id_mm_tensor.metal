// quantized_matmul_id_mm_tensor.metal — MoE-routed GGML-quantized mat-mat
// kernels using the Apple Metal tensor_ops (MetalPerformancePrimitives)
// primitives (ADR-011 Phase 3 Wave P3b-tensor).
//
// Tensor-API equivalent of quantized_matmul_id_mm.metal — replaces the
// simdgroup_multiply_accumulate inner loop with `mpp::tensor_ops::matmul2d`
// which hits the M3+ hardware tensor cores for 2-3× the FLOP throughput.
//
// Only the mm_id kernel is ported here (map0 is a short pre-pass, no
// matmul — the existing simdgroup version is reused verbatim).  Shared-
// memory staging is the tensor-path row-major layout identical to the
// dense tensor mm kernel.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define QK4_0 32
#define QK8_0 32
#define QK_K  256
#define QK_NL 16

struct GgmlMatmulIdMmTensor_MmParams {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int16_t  sched_flags; // compact kernels only: bit 0 = half-tile skip (P2c)
    int16_t  _pad1;
};

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

// ADR-013 P16 — Q4_K block (144 bytes) for tensor-API mm_id port.
#define K_SCALE_SIZE 12
typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

// ADR-022 Phase 2 — Q5_K block (176 bytes) for tensor-API mm_id port.
typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

// ADR-022 Phase 1 — Q5_1 / IQ4_NL block typedefs for the tensor-API mm_id port.
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

// ADR-033 §Pi Task #20 tensor-API — IQ4_XS block (136 bytes).
// Layout: [half d][uint16 scales_h][uint8[4] scales_l][uint8[128] qs].
// Same struct as in quantized_matmul_id_mm.metal — each Metal file is
// compiled separately so per-file typedefs are required.
typedef struct {
    half     d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;

// ADR-066 N-IQ2XXS tensor-API — IQ2_XXS block (66 bytes per 256 values).
typedef struct {
    half     d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;

// BEGIN GENERATED iq2xxs
// Generated from src/quant/iq_tables.rs by tests/iq_tables.rs
// (MLX_BLESS_IQ_TABLES=1). Do not edit by hand.
constant uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};
constant uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
constant uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128,
};
// END GENERATED iq2xxs

constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// Spec source: reference `get_scale_min_k4_just2`.
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dq_q4_0_id(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    // The scale is widened before the division: |d| / 16 is a subnormal half
    // (absolute step 2^-24, percent-level error) for |d| < 2^-10, which real
    // model files carry (nearly every Q4_K / Q5_K block).
    const float d1 = il ? (float(xb->d) / 16.f) : float(xb->d);
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
void dq_q5_0_id(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
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
void dq_q8_0_id(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
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
void dq_q2_K_id(device const block_q2_K * xb, short il, thread type4x4 & reg) {
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
void dq_q3_K_id(device const block_q3_K * xb, short il, thread type4x4 & reg) {
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
void dq_q6_K_id(device const block_q6_K * xb, short il, thread type4x4 & reg) {
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

// ADR-022 Phase 1 — Q5_1 / IQ4_NL dequant for tensor-API MMA-tile path.
// These mirror the dequantize_q5_1 / dequantize_iq4_nl helpers in
// quantized_matmul_id_mm.metal. Renamed to dq_<type>_id to follow the
// file-local convention.

template <typename type4x4>
void dq_q5_1_id(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
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

template <typename type4x4>
void dq_iq4_nl_id(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
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

// ADR-033 §Pi Task #20 tensor-API — IQ4_XS dequant for the tensor-cored
// MMA path. Spec source: reference `dequantize_iq4_xs`
// — reproduced verbatim modulo formatting. Same
// algorithm as `dequantize_iq4_xs` in quantized_matmul_id_mm.metal but
// scoped to the tensor-API kernel's namespace.
template <typename type4x4>
void dq_iq4_xs_id(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    const int ib32 = il / 2;
    il = il % 2;
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4 * ib32;
    const int ls = ((xb->scales_l[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)
                 | (((xb->scales_h >> 2 * ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4 * il) & 0x0f0f0f0f;
        reg[i][0] = d * (float)kvalues_iq4nl[q8[0]];
        reg[i][1] = d * (float)kvalues_iq4nl[q8[1]];
        reg[i][2] = d * (float)kvalues_iq4nl[q8[2]];
        reg[i][3] = d * (float)kvalues_iq4nl[q8[3]];
    }
}

// ADR-066 N-IQ2XXS tensor-API — IQ2_XXS dequant; same algorithm as
// `dequantize_iq2_xxs` in quantized_matmul_id_mm.metal.
template <typename type4x4>
void dq_iq2_xxs_id(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il / 2;
    il = il % 2;
    device const uint16_t * q2 = xb->qs + 4 * ib32;
    const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2 * il + 0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14 * il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i / 4][i % 4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2 * il + 1]);
    signs = ksigns_iq2xs[(aux32_s >> (14 * il + 7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2 + i / 4][i % 4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

// ADR-013 P16 — Q4_K dequant for tensor-API MMA-tile path.
// Spec source: reference `dequantize_q4_K`.
template <typename type4x4>
void dq_q4_K_id(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    // The scale is widened before the division: |d| / 16 is a subnormal half
    // (absolute step 2^-24, percent-level error) for |d| < 2^-10, which real
    // model files carry (nearly every Q4_K / Q5_K block).
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float min = xb->dmin;
    const float dl  = d * sc[0];
    const float ml  = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// ADR-022 Phase 2 — Q5_K dequant for tensor-API mm_id MMA-tile path.
// Spec source: reference `dequantize_q5_K`.
template <typename type4x4>
void dq_q5_K_id(device const block_q5_K * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    // The scale is widened before the division: |d| / 16 is a subnormal half
    // (absolute step 2^-24, percent-level error) for |d| < 2^-10, which real
    // model files carry (nearly every Q4_K / Q5_K block).
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float min = xb->dmin;
    const float dl  = d * sc[0];
    const float ml  = min * sc[1];

    const ushort mask  = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

// K loop, `matmul2d` accumulate and shared-memory store of one tile.
//
// `TM` is the row count of the B operand and of the destination. The
// uniform grid and full compact tiles use TM = 32 (`matmul2d_descriptor(32,
// 64, 32)`, today's body). The half-tile skip (P2c) runs partial compact
// tiles with at most 16 rows at TM = 16 over the lower half of the staged
// B tile; staging is unchanged, so rows 16-31 are loaded but not multiplied
// and the destination rows 16-31 of `sc` are not written.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &), short TM>
inline void hf2q_mul_mm_id_tensor_tile(
        constant GgmlMatmulIdMmTensor_MmParams & args,
        device const block_q * x,
        device const float * y,
        short il0,
        threadgroup half  * sa,
        threadgroup half  * sb,
        threadgroup float * sc,
        ushort tiitg) {
    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    short il = il0;

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NK, TM));

    matmul2d<
        matmul2d_descriptor(TM, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // Stage A.
        //
        // ADR-029 iter-55 H47 FALSIFIED 2026-05-11: tested adding
        // `#pragma clang loop unroll(full)` here to mirror peer's
        // FOR_UNROLL on this loop.
        // Bench (post-H44+H46 baseline, 4K + 8K warmup-then-real):
        //   4K MOE_GATE_UP: 10.61 → 10.65 ms/call (+0.4%, σ)
        //   4K MOE_DOWN:    9.82  → 9.83  ms/call (+0.1%, σ)
        //   8K MOE_GATE_UP: 20.58 → 20.56 ms/call (-0.1%, σ)
        //   8K MOE_DOWN:    19.08 → 19.09 ms/call (+0.1%, σ)
        //   4K wall:        1549  → 1551 ms (+0.1%, σ)
        //   8K wall:        3307  → 3307 ms (0%, σ)
        // ALL WITHIN σ on both regimes.  Deep-research hypothesis was
        // that mm_id's extra pre-loop live registers (id, i11, i12,
        // offset0, offset1, ids_i32) inhibit Metal's auto-unroll
        // heuristic — measurement falsifies this.  Metal compiler
        // already auto-unrolls regardless of register pressure here.
        // The original P4.8-null-effect comment direction was correct
        // even though the cited measurement was on the dense path;
        // iter-55 confirms mm_id has the same null effect.
        {
            half4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        // Stage B (f32 → half, 8-wide vector store).  See the dense
        // tensor kernel's equivalent staging for the rationale:
        // K is always a multiple of NK=32 on our projections, so the
        // per-element K-tail bounds check that the scalar path needs is
        // never triggered — drop it and issue a single half2x4 store
        // per thread.  Matches the FC_mul_mm_bc_inp=false fast path.
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            *(threadgroup half2x4 *)(sb + NK*(8*sy + ly) + 8*sx) =
                (half2x4)(*((device float2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
        // ADR-040 §0.19: post-mm.run barrier — without it the next K-iter
        // overwrites sa/sb while this mm.run may still be reading them under
        // GPU contention (intra-kernel race; deterministic single-tenant by
        // timing-luck, flakes under multi-process scheduling). Mirrors the V2
        // path + llama's kernel_mul_mm (ggml-metal.metal:9549).
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write-back: always through shmem (scatter-by-hids) — same pattern as
    // the simdgroup mm_id version, just cooperative_tensor::store instead
    // of simdgroup_store for the shmem stage.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        auto tC_sm = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, TM));
        cT.store(tC_sm);
    }
}

// COMPACT (ADR-063 P2, `_c` host names): grid of `max_tiles` tile slots
// (grid x, fastest) × ⌈N/64⌉ N tiles (grid y) over the map0c schedule; see
// quantized_matmul_id_mm.metal for the descriptor layout. Bit 0 enables the
// half-tile skip (P2c) for tiles of <= 16 rows.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &), bool COMPACT = false>
kernel void hf2q_mul_mm_id_tensor_impl(
        constant GgmlMatmulIdMmTensor_MmParams & args [[buffer(0)]],
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
    threadgroup half  * sb = (threadgroup half  *)(shmem + 4096);
    threadgroup float * sc = (threadgroup float *)(shmem);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    int   im;
    int   r0;
    int   list_base;  // hids index of this tile's first routed row
    short nr1;
    if (COMPACT) {
        const uint32_t schedule_state = tpe_u32[0];
        const uint32_t tile  = tgpig.x;
        const uint32_t ntile = tgpig.y;
        if ((schedule_state & 0x80000000u) != 0u) {
            if (tgpig.x == 0 && tgpig.y == 0) {
                device float * out = (device float *) dst;
                const uint64_t total =
                    (uint64_t)args.ne21 * (uint64_t)args.ne1 * (uint64_t)args.ne0;
                for (uint64_t index = tiitg; index < total; index += 128) {
                    out[index] = NAN;
                }
            }
            return;
        }
        if (tile >= schedule_state) return;
        device const uint32_t * desc = tpe_u32 + 1 + 3*tile;
        im        = (int)desc[0];
        list_base = (int)desc[1];
        nr1       = (short)desc[2];
        r0        = ntile * NR0;
    } else {
        im = tgpig.z;
        r0 = tgpig.y * NR0;
        const int r1 = tgpig.x * NR1;

        const uint32_t route_state = tpe_u32[im];
        if ((route_state & 0x80000000u) != 0u) {
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
        if (r1 >= neh1) return;

        nr1 = (neh1 - r1 < NR1) ? (neh1 - r1) : NR1;
        list_base = im * args.ne21 + r1;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    const int id = ids_i32[list_base + lr1];
    const short i11 = (id % args.ne20) % args.ne11;
    // §0.19 FIX (2026-06-30): token/row index must be `int` — flat
    // [n_tokens, K] layout makes i12 reach n_tokens-1 (65535 at 8k); `short`
    // overflows at 32768 → OOB gather. See quantized_matmul_id_mm.metal.
    const int   i12 = (id / args.ne20);
    const short i13 = 0;

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

    if (COMPACT && (args.sched_flags & 1) != 0 && nr1 <= 16) {
        hf2q_mul_mm_id_tensor_tile<block_q, nl, dequantize_func, 16>(
            args, x, y, il0, sa, sb, sc, tiitg);
    } else {
        hf2q_mul_mm_id_tensor_tile<block_q, nl, dequantize_func, NR1>(
            args, x, y, il0, sa, sb, sc, tiitg);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[list_base + j];
        const short ide = id % args.ne20;
        // §0.19 FIX (2026-06-30): output token/batch index must be `int` —
        // `short` overflows at n_tokens>32768 → OOB write-back. See
        // quantized_matmul_id_mm.metal.
        const int   idt = id / args.ne20;

        // §0.19 FIX (2026-06-30, codex SHIP-WITH-CHANGES): 64-bit output
        // element offset (each factor promoted before multiply). See
        // quantized_matmul_id_mm.metal. Byte-identical for in-range values.
        const uint64_t dst_off =
            (uint64_t) r0
            + (uint64_t) ide * (uint64_t) args.ne0
            + (uint64_t) idt * (uint64_t) args.ne1 * (uint64_t) args.ne0;
        device float  * D  = (device float  *) dst + dst_off;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = sc + j*NR0;
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

template [[host_name("kernel_mul_mm_id_q4_0_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q4_0, 2, dq_q4_0_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q5_0_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q5_0, 2, dq_q5_0_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q8_0_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q8_0, 2, dq_q8_0_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q2_K_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q2_K, QK_NL, dq_q2_K_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q3_K_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q3_K, QK_NL, dq_q3_K_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q6_K_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q6_K, QK_NL, dq_q6_K_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-013 P16 — Q4_K tensor-API mm_id template instantiation.
// ADR-022 Phase 1 P1.6 — Q5_1 / IQ4_NL tensor-API mm_id template instantiations.
template [[host_name("kernel_mul_mm_id_q5_1_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q5_1, 2, dq_q5_1_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_iq4_nl_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_iq4_nl, 2, dq_iq4_nl_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-033 §Pi Task #20 tensor-API — IQ4_XS tensor-API mm_id
// template instantiation. Closes the prefill perf gap vs the peer
// at production-Qwen-MoE shape — the simdgroup variant alone clocks
// ~1465 tok/s on M5 Max while the peer's tensor-core kernel hits
// ~2127 tok/s. Tensor-core path uses Apple M3+'s simdgroup matrix
// multiply intrinsics via mlx's `tensor_ops::matmul2d` reduction.
template [[host_name("kernel_mul_mm_id_iq4_xs_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_iq4_xs, QK_NL, dq_iq4_xs_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-066 N-IQ2XXS tensor-API — IQ2_XXS tensor mm_id instantiation.
template [[host_name("kernel_mul_mm_id_iq2_xxs_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_iq2_xxs, QK_NL, dq_iq2_xxs_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q5_K_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q5_K, QK_NL, dq_q5_K_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q4_K_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q4_K, QK_NL, dq_q4_K_id>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

// ADR-063 P2 — compact-schedule (`_c`) instantiations of every mm_id type.
template [[host_name("kernel_mul_mm_id_q4_0_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q4_0, 2, dq_q4_0_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q5_0_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q5_0, 2, dq_q5_0_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q8_0_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q8_0, 2, dq_q8_0_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q2_K_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q2_K, QK_NL, dq_q2_K_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q3_K_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q3_K, QK_NL, dq_q3_K_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q4_K_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q4_K, QK_NL, dq_q4_K_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q5_K_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q5_K, QK_NL, dq_q5_K_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q6_K_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q6_K, QK_NL, dq_q6_K_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q5_1_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q5_1, 2, dq_q5_1_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_iq4_nl_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_iq4_nl, 2, dq_iq4_nl_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_iq4_xs_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_iq4_xs, QK_NL, dq_iq4_xs_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_iq2_xxs_tensor_f32_c")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_iq2_xxs, QK_NL, dq_iq2_xxs_id, true>(
    constant GgmlMatmulIdMmTensor_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);
