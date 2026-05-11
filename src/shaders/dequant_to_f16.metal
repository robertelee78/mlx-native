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

// ADR-029 iter-29 — Q6_K dequant in LINEAR K-order, byte-identical to the
// CPU reference at /opt/mlx-native/src/gguf/mod.rs:667-720.
//
// Layout per Q6_K block (210 bytes, 256 elements):
//   ql[0..128]     — low nibbles
//   qh[0..64]      — high 2-bit pairs
//   sc[0..16] i8   — 16 sub-scales
//   d:half          — block scale
//
// Each `il ∈ [0, 16)` here covers a CONTIGUOUS 16-element K-strip,
// in linear K-position order.  Writing `*(half4x4 *)dst + tid = reg` then
// produces a row-major linear-K F16 tensor that matches what
// `dense_matmul_f16_f32_tensor` expects (peer's `kernel_get_rows_q6_K`
// pattern would re-pack at runtime; we do it once at load).
//
// Math is the CPU dequantize_q6_k code restructured to emit exactly the
// 16 elements at K-positions [il*16 .. il*16 + 16).  Index map (matches
// CPU loops at gguf/mod.rs:702-715):
//   block of 256 elements = 2 halves × 128 elements
//   half h ∈ {0, 1}: ql_base = ql + 64*h, qh_base = qh + 32*h,
//                    sc_base = sc + 8*h, out_base offset = 128*h
//   within a half, 32 'l' values (l ∈ [0, 32)) produce 4 elements each
//   at K-positions {l, l+32, l+64, l+96}.
//
// For il=0..15, the linear K-range is [il*16, il*16+16).
//   half h          = il / 8
//   within-half pos = il % 8 * 16 → 16-element strip starts at this
//
// Each strip's 16 elements are at K-positions s..s+15 where s = (il%8)*16.
// Within the strip's [0, 16) sub-index i, the CPU code maps to specific
// (l, group, q-index).  Reconstruct here:
//   If s + i < 32 (group 0 = q1):  l = s + i,        out_pos = l
//   If s + i < 64 (group 1 = q2):  l = (s + i) - 32, out_pos = l + 32
//   If s + i < 96 (group 2 = q3):  l = (s + i) - 64, out_pos = l + 64
//   If s + i < 128 (group 3 = q4): l = (s + i) - 96, out_pos = l + 96
template <typename type4x4>
void dq_q6_K(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const float d_all = (float)xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t  * sc = (device const int8_t  *)xb->scales;

    const short h     = il / 8;          // 0 or 1
    const short s     = (il % 8) * 16;   // 16-element strip start within half

    device const uint8_t * ql_base = ql + 64 * h;
    device const uint8_t * qh_base = qh + 32 * h;
    device const int8_t  * sc_base = sc + 8  * h;

    float4x4 reg_f;
    for (short i = 0; i < 16; ++i) {
        const short k_in_half = s + i;            // 0..127
        const short group = k_in_half / 32;       // 0..3
        const short l = k_in_half - group * 32;   // 0..31
        const short is = l / 16;                  // 0 for l<16, 1 for l>=16

        // Extract the 6-bit value for this position.  Matches CPU code:
        //   group 0: q1 = (ql_base[l]   & 0xF) | ((qh_base[l] & 3) << 4) - 32
        //   group 1: q2 = (ql_base[l+32]& 0xF) | ((qh_base[l]>>2 & 3) << 4) - 32
        //   group 2: q3 = (ql_base[l]   >> 4)  | ((qh_base[l]>>4 & 3) << 4) - 32
        //   group 3: q4 = (ql_base[l+32]>> 4)  | ((qh_base[l]>>6 & 3) << 4) - 32
        // CPU reference (gguf/mod.rs:705-710):
        //   group 0 (q1): ql_base[l]    & 0xF    | (qh_base[l] & 3)        << 4
        //   group 1 (q2): ql_base[l+32] & 0xF    | ((qh_base[l] >> 2) & 3) << 4
        //   group 2 (q3): ql_base[l]    >> 4     | ((qh_base[l] >> 4) & 3) << 4
        //   group 3 (q4): ql_base[l+32] >> 4     | ((qh_base[l] >> 6) & 3) << 4
        //   Then cast to i8 and subtract 32 to get signed [-32, 31].
        //
        // Mapping: groups 0,1 use LOW nibble of ql; groups 2,3 use HIGH nibble.
        //          groups 0,2 use ql_base[l]; groups 1,3 use ql_base[l+32].
        uint8_t ql_byte;
        uint8_t shift_h;
        switch (group) {
            case 0: ql_byte = ql_base[l];      shift_h = 0; break;
            case 1: ql_byte = ql_base[l + 32]; shift_h = 2; break;
            case 2: ql_byte = ql_base[l];      shift_h = 4; break;
            default: /* 3 */ ql_byte = ql_base[l + 32]; shift_h = 6; break;
        }
        const uint8_t high_bits = (qh_base[l] >> shift_h) & 0x3;
        // group < 2 → low nibble; group >= 2 → high nibble (>> 4).
        const uint8_t low_bits = (group < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);
        // Final 6-bit value: subtract 32 in i8 space → [-32, 31].
        const int q = (int)((int8_t)((low_bits | (high_bits << 4)) - 32));

        // Sub-scale: each group has 4 scales, indexed by `is` (l/16).
        const float scale = d_all * (float)(sc_base[group * 2 + is]);

        const float val = scale * (float)q;
        reg_f[i / 4][i % 4] = val;
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
