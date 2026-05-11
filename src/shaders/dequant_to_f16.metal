// dequant_to_f16.metal — whole-tensor dequantization from block-quantized
// formats (Q4_0, Q5_K, Q6_K, Q8_0, etc.) to F16 storage.
//
// ADR-029 iter-28 H29 — peer (llama.cpp on Apple Silicon) NEVER dispatches
// quantized mat-mat kernels for gemma4 attn weights (Q6_K).  Instead it
// pre-dequantizes Q6_K → F16 once at model load, then runs the F16-input
// `kernel_mul_mm_f16_f32_*` for every dense attn dispatch.  This trades
// ~1 GB of extra resident memory for 2-3× faster per-call dense MM at
// prefill — bandwidth-friendly F16 reads instead of per-call dequant work
// inside the matmul kernel.
//
// hf2q has 128 GB unified memory on the target M5 Max device; the memory
// trade is favorable.  This shader produces the F16 shadow buffer that
// the load-path materializes once per quantized weight tensor.
//
// Design (mirrors peer's `kernel_get_rows_q` at ggml-metal.metal:9164-9191):
//   * One thread per 16-element group.
//   * Each thread reads its corresponding `(block_idx, il)` slot:
//       block_idx = ind / nl
//       il        = ind % nl
//   * Calls `dequantize_func(block_ptr + block_idx, il, temp)` which
//     produces a `half4x4` (16 halfs).
//   * Writes the half4x4 to `dst[ind]` (treated as `device half4x4 *`).
//
// Dispatch: total threads = n_elements / 16.  Pick threadgroup size to
// saturate (e.g. 256 threads/tg).
//
// `nl` is the type-specific QK_NL constant (2 for legacy block-quant types
// at QK=32; 16 for K-quants at QK_K=256).  See callers for the per-type
// instantiation list.
//
// Coherence: F16 storage of dequantized Q6_K introduces F16-rounding drift
// vs the per-call float-precision dequant the V1/V2 matmul kernel does.
// Empirically peer ships this in production on gemma4-26B with no observable
// quality regression; the F16 mantissa (10 bits, ~1e-3 ulp) is well above
// the Q6_K quantization noise floor.  Sourdough byte-identity is NOT
// expected; coherence gate is "fluent output at temp=0 across regimes".

#include <metal_stdlib>
using namespace metal;

// ---- GGML block sizes (match quantized_matmul_mm_tensor.metal) ----
#define QK4_0 32
#define QK8_0 32
#define QK_K  256
#define K_SCALE_SIZE 12

// ---- Block struct definitions (byte-identical to quantized_matmul_mm_tensor.metal) ----

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

typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

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

constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4)  | ((q[j-0+k] & 0xc0) >> 2))};
}

// ---- Dequantize helpers (identical bodies to quantized_matmul_mm_tensor.metal) ----
// Each function dequantizes ONE 16-element sub-group of one block_q.

template <typename type4x4>
void dq_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
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
void dq_q8_0(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const half d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (float)(d * qs[16*il + i]);
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dq_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
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
void dq_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
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

template <typename type4x4>
void dq_q5_K(device const block_q5_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs + 32*(il/4) + 16*(il&1);
    device const uchar * qh = xb->qh + 16*(il&1);
    const uchar2 sc = get_scale_min_k4_just2(il/2, 8, xb->scales);
    const float d_all = (float)xb->d;
    const float dl = d_all * sc[0];
    const float ml = (float)xb->dmin * sc[1];
    const ushort mask = 1 << (il/2);
    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        const float val = dl * (((q[i] >> (4*(il&2))) & 0xF) + (qh[i] & mask ? 16 : 0)) - ml;
        reg_f[i/4][i%4] = val;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dq_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs + 32*(il/4) + 16*(il&1);
    const uchar2 sc = get_scale_min_k4_just2(il/2, 8, xb->scales);
    const float d_all = (float)xb->d;
    const float dl = d_all * sc[0];
    const float ml = (float)xb->dmin * sc[1];
    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        const float val = dl * ((q[i] >> (4*(il&2))) & 0xF) - ml;
        reg_f[i/4][i%4] = val;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dq_q6_K(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t  * sc = (device const int8_t  *)xb->scales;

    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);
    qh = qh + 32*(il/8)                  + 16*(il&1);
    sc = sc + 8*(il/8);

    // (matches dq_q6_K body in quantized_matmul_mm_tensor.metal — unused
    // local consts dropped to satisfy Metal -Werror unused-variable.)
    const short sh = (il & 2) ? 2 : 0;

    const float dl0 = d_all * sc[0] / 32.f;
    const float dl1 = d_all * sc[2] / 32.f;
    const float dl2 = d_all * sc[4] / 32.f;
    const float dl3 = d_all * sc[6] / 32.f;
    const float ml  = 32.f * d_all;

    const uint32_t kmask1 = 0x0F0F0F0F;
    const uint32_t kmask2 = 0xC0C0C0C0 >> (sh*8);

    const uchar shr_h = il & 4 ? 0 : 2;
    const uchar shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uchar shr_l = il>1 ? 4 : 0;

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

// ---- The whole-tensor dequant kernel ----
//
// Total dispatched threads = n_elements / 16 = n_blocks * nl
// Each thread:
//   block_idx = tid / nl
//   il        = tid % nl
//   dequantize_func(src + block_idx, il, temp_half4x4)
//   *(dst + tid) = temp_half4x4    (16 halfs at offset tid*16)

template<typename block_q, short nl,
         void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void hf2q_dequant_to_f16_impl(
        constant uint32_t & n_groups [[buffer(0)]],
        device const char * src      [[buffer(1)]],
        device       char * dst      [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= n_groups) return;

    const uint block_idx = tid / nl;
    const short il = (short)(tid % nl);

    device const block_q * blk = (device const block_q *)src + block_idx;
    device half4x4 * out = (device half4x4 *)dst + tid;

    half4x4 temp;
    dequantize_func(blk, il, temp);
    *out = temp;
}

// ---- Kernel instantiations (one per supported quant type) ----

template [[host_name("hf2q_dequant_q4_0_to_f16")]]
kernel void hf2q_dequant_to_f16_impl<block_q4_0, 2, dq_q4_0>(
    constant uint32_t &, device const char *, device char *, uint);

template [[host_name("hf2q_dequant_q8_0_to_f16")]]
kernel void hf2q_dequant_to_f16_impl<block_q8_0, 2, dq_q8_0>(
    constant uint32_t &, device const char *, device char *, uint);

template [[host_name("hf2q_dequant_q5_1_to_f16")]]
kernel void hf2q_dequant_to_f16_impl<block_q5_1, 2, dq_q5_1>(
    constant uint32_t &, device const char *, device char *, uint);

template [[host_name("hf2q_dequant_iq4_nl_to_f16")]]
kernel void hf2q_dequant_to_f16_impl<block_iq4_nl, 2, dq_iq4_nl>(
    constant uint32_t &, device const char *, device char *, uint);

template [[host_name("hf2q_dequant_q4_K_to_f16")]]
kernel void hf2q_dequant_to_f16_impl<block_q4_K, 16, dq_q4_K>(
    constant uint32_t &, device const char *, device char *, uint);

template [[host_name("hf2q_dequant_q5_K_to_f16")]]
kernel void hf2q_dequant_to_f16_impl<block_q5_K, 16, dq_q5_K>(
    constant uint32_t &, device const char *, device char *, uint);

template [[host_name("hf2q_dequant_q6_K_to_f16")]]
kernel void hf2q_dequant_to_f16_impl<block_q6_K, 16, dq_q6_K>(
    constant uint32_t &, device const char *, device char *, uint);
