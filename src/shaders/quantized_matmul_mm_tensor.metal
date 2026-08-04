// quantized_matmul_mm_tensor.metal — GGML block-format quantized mat-mat
// kernels using the Apple Metal tensor_ops (MetalPerformancePrimitives)
// intrinsics.
//
// This is the Metal-tensor-API equivalent of our existing simdgroup-MMA
// mul_mm kernel (quantized_matmul_mm.metal) — ports llama.cpp's
// `kernel_mul_mm_impl<GGML_METAL_HAS_TENSOR>` branch
// (ggml/src/ggml-metal/ggml-metal.metal:9289+).  Same tile geometry
// (NR0=64, NR1=32, NK=32, 4 simdgroups / threadgroup, 128 threads), same
// dequantize functions.  The difference is the compute engine: instead
// of `simdgroup_multiply_accumulate` the kernel uses
// `mpp::tensor_ops::matmul2d<>` which on M3+ dispatches to the hardware
// tensor cores — 2-3x the effective FLOP throughput of the simdgroup
// MMA path.
//
// Shared-memory layout is different from the simdgroup path:
//   * sa (A/weight tile): half, `[NR0=64][NK=32]` row-major, 4096 B
//   * sb (B/input tile):  half, `[NR1=32][NK=32]` row-major, 2048 B
// Non-tensor path uses float for sb (staging the f32 input verbatim) —
// tensor_ops::matmul2d rejects mixed-precision operands (both A and B
// must be the same Metal type), so we cast f32 input → half at staging.
// The half intermediate has ample precision for the 32-wide K reduction
// (mantissa drift vs f32 is 1-2 ULPs in the 256-1152 K range).
//
// Gated at kernel-registry level: this file is only compiled / registered
// on devices where the tensor API is available (M3+).  At runtime the
// dispatcher picks between this tensor kernel and the simdgroup fallback
// based on a device-capability check.
//
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp), MIT licensed.
// Original source: ggml/src/ggml-metal/ggml-metal.metal.
// Copyright the llama.cpp Authors.  See LICENSE-MIT-llamacpp.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// ---- GGML block sizes (must match quantized_matmul_mm.metal) ----

#define QK4_0 32
#define QK8_0 32
#define QK_K  256
#define QK_NL 16

// ---- Host-facing params struct (shared layout with the non-tensor
//      kernel, see quantized_matmul_mm.metal::GgmlMatmulMmParams) ----

struct GgmlMatmulMmTensorParams {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

// ---- GGML block struct definitions (byte-for-byte GGUF layout) ----

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

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
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;

// ADR-022 Phase 2 — Q5_K block + helper for tensor mm.
#define K_SCALE_SIZE 12
typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

// ADR-022 Phase 3 — Q4_K block typedef for tensor mm.
typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4)  | ((q[j-0+k] & 0xc0) >> 2))};
}

// ADR-022 Phase 1 — Q5_1 / IQ4_NL block typedefs for tensor mm.
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
    1, 13, 25, 38, 53, 69, 89, 113
};

// ---- Dequantize helpers (identical to the non-tensor file; duplicated
//      so this file is self-contained and independently compilable) ----

template <typename type4x4>
void dequantize_q4_0_t(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
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
void dequantize_q8_0_t(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
}

// Pinned llama.cpp 6ea215d17 ggml-metal.metal::dequantize_q2_K.
template <typename type4x4>
void dequantize_q2_K_t(device const block_q2_K * xb, short il, thread type4x4 & reg) {
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
void dequantize_q6_K_t(device const block_q6_K * xb, short il, thread type4x4 & reg) {
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

// ADR-022 Phase 1 — Q5_1 dequant for tensor mm (identical math to non-tensor).
template <typename type4x4>
void dequantize_q5_1_t(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
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

// ADR-022 Phase 1 — IQ4_NL dequant for tensor mm.
template <typename type4x4>
void dequantize_iq4_nl_t(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
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

// ADR-022 Phase 2 — Q5_K dequant for tensor mm. Identical body to the
// non-tensor variant in quantized_matmul_mm.metal — the dequant math is
// type-agnostic; only the kernel template type signature differs.
template <typename type4x4>
void dequantize_q5_K_t(device const block_q5_K * xb, short il, thread type4x4 & reg) {
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

// ADR-022 Phase 3 — Q4_K dequant for tensor mm.
template <typename type4x4>
void dequantize_q4_K_t(device const block_q4_K * xb, short il, thread type4x4 & reg) {
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

// ---- tensor-API mul_mm template ----
//
// Direct port of llama.cpp's kernel_mul_mm with the GGML_METAL_HAS_TENSOR
// branches active.  Shared memory is `sa`/`sb` in row-major layout that
// the tensor<> views consume directly.  Every loop iteration stages a
// 64x32 (A) + 32x32 (B) tile, then runs `mm.run` which the compiler
// lowers to native M3+ tensor MMA.  Partial-tile (edge) write-back uses
// a threadgroup float buffer shared with sa+sb, matching llama.cpp's
// layout.

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void hf2q_mul_mm_tensor_impl(
        constant GgmlMatmulMmTensorParams & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup half  * sb = (threadgroup half  *)(shmem + 4096);
    threadgroup float * sc = (threadgroup float *)(shmem);  // partial-tile write-back reuses shmem base

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    // Tensor views over the shared staging buffers.  Both A and B are
    // half in shmem; tensor_ops::matmul2d requires operand-type match.
    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    // Matmul operator — transpose_right=true matches the layout where B's
    // inner dimension is NK and A's inner dimension is also NK (the K
    // axis); the compiler emits the tensor-cores matmul variant.
    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- Stage A tile (block_q -> half, via dequantize_func).
        // Tensor-path layout: sa is [NR0][NK] row-major — write every
        // element to `sa + NK*(8*sy + ly) + 8*sx + lx`.  Matches
        // llama.cpp ggml-metal.metal:9446-9456 (GGML_METAL_HAS_TENSOR
        // branch).
        //
        // NOTE: We DO NOT add llama.cpp's FOR_UNROLL pragma here.
        // Tested 2026-04-19 (P4.8): no measurable prefill delta on M5
        // Max (5-run median 2710 tok/s with vs 2710 without).  The
        // Metal compiler unrolls 16-iter constant-bound loops on its
        // own; the explicit pragma adds no value on this gen.  Per
        // project memory entry "Metal compiler auto-optimizes static
        // levers", we leave the source minimal rather than carrying a
        // null-effect annotation that suggests the compiler doesn't.
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

        // ---- Stage B tile (f32 input cast to half, vector store) ----
        //
        // Gemma 4 (and all Llama-style) projections have K divisible by
        // NK=32, so the per-element K-tail bounds check that the
        // per-element path needs is never triggered in practice.  Drop
        // it and issue a single 8-wide vector store per thread — this
        // is what llama.cpp's `FC_mul_mm_bc_inp=false` path does and is
        // 4-8x the per-element path's store throughput.
        //
        // Cast: `(half2x4)(*((device float2x4 *) y))` loads 8 f32 values
        // from the input row and packs them as 8 halfs into sb.  The
        // rest of the K-loop iteration layout is identical to the
        // per-element version; only the staging pattern changes.
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

        // ---- Multiply: matmul2d over the staged tiles ----
        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
        // ADR-040 §0.19: post-mm.run barrier (see V2 path + llama
        // ggml-metal.metal:9549) — prevents the next K-iter from overwriting
        // sa/sb while mm.run may still read them under GPU contention.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write-back ----
    // Fast path: full 64x32 tile, direct cooperative tensor store to device.
    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *) dst +
            r0 +
            r1 * args.ne0 + im*args.ne1*args.ne0;

        auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(args.ne0, NR1));
        cT.store(tC);
    } else {
        // Partial tile: stage to shmem (reusing sa+sb space), then the
        // first simdgroup copies rows out with M-bound.  Same approach as
        // llama.cpp's non-tensor path, just using cooperative_tensor::store
        // to shmem instead of simdgroup_store.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
        cT.store(tC);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = sc + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

// ===========================================================================
// ADR-029 iter-23 H28-A: large-tile v2 mm-tensor — 4× threadgroup reduction
// at gemma4 prefill shapes vs the legacy 32×64 tile.
//
// Geometry (ports llama.cpp ggml-metal.metal:9309-9431 line-for-line, with
// the type-template params collapsed to gemma4's fixed F32-in / F32-out /
// F16-shmem case):
//   NRA = SZ_SIMDGROUP × N_MM_BLOCK_Y × N_MM_SIMD_GROUP_Y = 16 × 2 × 2 = 64
//                                                          (peer-name "M tile")
//   NRB = SZ_SIMDGROUP × N_MM_BLOCK_X × N_MM_SIMD_GROUP_X = 16 × 4 × 2 = 128
//                                                          (peer-name "N tile")
//   NK_TOTAL = SZ_SIMDGROUP × N_MM_NK = 16 × 2 = 32
//   simdgroups/tg = N_MM_SIMD_GROUP_X × N_MM_SIMD_GROUP_Y = 2 × 2 = 4
//   threads/tg    = N_SIMDWIDTH × 4 = 128
//
// Notes on hf2q convention vs peer:
//   * hf2q dispatch passes args.ne0 = N (cols of output), ne1 = M (rows).
//     Peer passes ne0 = M, ne1 = N.  The kernel's internal ra/rb naming
//     matches peer (ra = M-axis offset, rb = N-axis offset) — the Rust
//     dispatcher uses peer-equivalent geometry so callers stay unchanged.
//   * No threadgroup B-staging.  Peer reads B (input activations, F32)
//     directly from device memory through an mpp `tensor` view (line
//     9358-9360 of ggml-metal.metal).  This eliminates the per-loop B
//     copy-into-shmem (lines 408-414 of the V1 kernel) and frees the
//     2-4 KB of shared memory it consumed.
//   * Output store uses mpp `cT.store(tD.slice(ra, rb))`.  The destination
//     tensor wraps the device buffer with the same column-major-over-M
//     stride peer uses (`array<2>({1, M})`); the data layout that hf2q's
//     downstream consumers see is unchanged — the kernel's M/N internal
//     naming differs from V1's r0/r1 but the bytes on the wire match.
//
// At prefill shape m=4213, n=5760:
//   V1 dispatches 132 × 90 = 11,880 threadgroups (NR0=64, NR1=32 tile)
//   V2 dispatches  66 × 45 =  2,970 threadgroups (NRA=64, NRB=128 tile)
//
// Gated at the Rust side by `HF2Q_LARGE_TILE_MM=1` (env opt-in).  Default
// OFF until coherence + multi-regime bench parity proven.
// ===========================================================================

template<typename block_q, short nl,
         void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void hf2q_mul_mm_tensor_v2_impl(
        constant GgmlMatmulMmTensorParams & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Peer's ggml convention has src0=[K, M_peer, batch], src1=[K, N_peer, batch],
    // output=[M_peer, N_peer] with column-major-over-M_peer storage.
    // hf2q's dispatch SWAPS M/N at the params layer (ne0 = hf2q N = peer M,
    // ne1 = hf2q M = peer N) so the downstream V1 store `dst + r0 + r1*ne0`
    // writes row-major-over-(hf2q-N) ≡ column-major-over-(peer-M).  V2 keeps
    // peer's internal naming (M_peer / N_peer) so the kernel body mirrors
    // ggml-metal.metal:9326-9329 line-for-line; the dispatcher accounts
    // for the hf2q axis swap.
    const int K      = args.ne00;
    const int M_peer = args.ne0;   // hf2q ne0 = hf2q-N = peer M
    const int N_peer = args.ne1;   // hf2q ne1 = hf2q-M = peer N

    const int im = tgpig.z;
    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t offset0 = (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;

    // Tile constants — peer's ggml-metal-impl.h.
    constexpr int SZ_SIMDGROUP        = 16;
    constexpr int N_MM_BLOCK_X        = 4;
    constexpr int N_MM_BLOCK_Y        = 2;
    constexpr int N_MM_SIMD_GROUP_X   = 2;
    constexpr int N_MM_SIMD_GROUP_Y   = 2;
    constexpr int N_MM_NK             = 2;
    constexpr int N_MM_NK_TOTAL       = SZ_SIMDGROUP * N_MM_NK;          // 32
    constexpr int N_SIMDWIDTH         = 32;

    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y; // 64 = M tile
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X; // 128 = N tile

    const int ra = tgpig.y * NRA;   // M_peer offset (gy covers M_peer = hf2q-N)
    const int rb = tgpig.x * NRB;   // N_peer offset (gx covers N_peer = hf2q-M)

    threadgroup half * sa = (threadgroup half *)(shmem);

    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;                              // 128
    constexpr int NUM_THREADS  = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y; // 128

    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // B is F32, read directly from device memory (peer ggml-metal.metal:9358-9360).
    // B has hf2q layout `B[N_peer_idx, K_idx] = base + N_peer_idx*K + K_idx`
    // (slow axis = N_peer = hf2q-M = tokens, fast axis = K).
    device float * ptrB = (device float *)(srcB + args.nb12 * i12 + args.nb13 * i13);
    const int strideB = (int)(args.nb11 / sizeof(float));   // = K floats per N_peer row
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N_peer), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NRB, NRA, N_MM_NK_TOTAL, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // PHASE 1: dequantize A tile into sa.
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row     = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos   = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            if (ra + row < M_peer) {
                const int block_idx = k_pos / (16 * nl);
                const short il      = (k_pos / 16) % nl;

                device const block_q * row_ptr =
                    (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                half4x4 temp_a;
                dequantize_func(row_ptr + block_idx, il, temp_a);

                // ADR-029 iter-61 H50: add #pragma unroll(full) to mirror peer's
                // FOR_UNROLL at ggml-metal.metal:9403.  iter-34 H36 + iter-55 H47
                // tested this on mm_id (FALSIFIED both times — Metal compiler
                // auto-unrolls there).  This is the FIRST test on V2 dense
                // mm-tensor.  Falsifier: bench HF2Q_F16_SHADOW=0 at pp4096
                // (exercises the direct-Q6_K V2 path); expected if real, delta
                // ≥ +5% t/s vs baseline 2607.4.
                #pragma clang loop unroll(full)
                for (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] =
                        (k_pos + i < K) ? temp_a[i / 4][i % 4] : (half)0;
                }
            } else {
                #pragma clang loop unroll(full)
                for (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PHASE 2: tensor matmul.
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);

        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store cooperative tensor to device output.
    // Output shape (peer convention): [M_peer, N_peer], column-major-over-M_peer.
    // In hf2q-equivalent terms: [hf2q-N, hf2q-M] column-major-over-N — i.e.,
    // `out[N_idx, M_idx] = base + N_idx + M_idx*N` (V1 layout exactly).
    device float * dstBatch = (device float *)dst +
        im * (uint64_t)M_peer * (uint64_t)N_peer;
    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M_peer, N_peer),
                     array<int, 2>({1, M_peer}));
    cT.store(tD.slice(ra, rb));
}

// ===========================================================================
// ADR-029 iter-30 H29-speed: F16-weight variant of the V2 large-tile mm.
//
// Identical geometry / semantics to `hf2q_mul_mm_tensor_v2_impl`, with
// the per-call dequantize_func replaced by a direct half load from the
// F16 weight shadow buffer.  Used when MlxQWeight.f16_shadow is Some.
// ===========================================================================

[[host_name("hf2q_mul_mm_tensor_v2_f16")]]
kernel void hf2q_mul_mm_tensor_v2_f16_impl(
        constant GgmlMatmulMmTensorParams & args,
        device const char * srcA,    // F16 weight [M_peer × K], nb01 = 2K bytes/row
        device const char * srcB,    // F32 input  [K × N_peer]
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    const int K      = args.ne00;
    const int M_peer = args.ne0;
    const int N_peer = args.ne1;

    const int im = tgpig.z;
    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;
    const uint64_t offset0 = (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;

    constexpr int SZ_SIMDGROUP        = 16;
    constexpr int N_MM_BLOCK_X        = 4;
    constexpr int N_MM_BLOCK_Y        = 2;
    constexpr int N_MM_SIMD_GROUP_X   = 2;
    constexpr int N_MM_SIMD_GROUP_Y   = 2;
    constexpr int N_MM_NK             = 2;
    constexpr int N_MM_NK_TOTAL       = SZ_SIMDGROUP * N_MM_NK;
    constexpr int N_SIMDWIDTH         = 32;

    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;   // 64
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;   // 128

    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    threadgroup half * sa = (threadgroup half *)(shmem);

    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS  = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    device float * ptrB = (device float *)(srcB + args.nb12 * i12 + args.nb13 * i13);
    const int strideB = (int)(args.nb11 / sizeof(float));
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N_peer), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NRB, NRA, N_MM_NK_TOTAL, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row     = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos   = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            if (ra + row < M_peer) {
                device const half * row_ptr =
                    (device const half *)(srcA + args.nb01 * (ra + row) + offset0);
                // ADR-029 iter-61 H50 (F16-shadow variant): same FOR_UNROLL fix
                #pragma clang loop unroll(full)
                for (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] =
                        (k_pos + i < K) ? row_ptr[k_pos + i] : (half)0;
                }
            } else {
                #pragma clang loop unroll(full)
                for (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, rb);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float * dstBatch = (device float *)dst +
        im * (uint64_t)M_peer * (uint64_t)N_peer;
    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M_peer, N_peer),
                     array<int, 2>({1, M_peer}));
    cT.store(tD.slice(ra, rb));
}

// (host_name set inline on the kernel above — non-template, no instantiation here)

// ---- V2 kernel instantiations (env-gated via HF2Q_LARGE_TILE_MM=1) ----
template [[host_name("kernel_mul_mm_q4_0_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_q4_0, 2, dequantize_q4_0_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q8_0_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_q8_0, 2, dequantize_q8_0_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q2_K_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_q2_K, QK_NL, dequantize_q2_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q6_K_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_q6_K, QK_NL, dequantize_q6_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q5_1_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_q5_1, 2, dequantize_q5_1_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q5_K_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_q5_K, QK_NL, dequantize_q5_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q4_K_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_q4_K, QK_NL, dequantize_q4_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_iq4_nl_tensor_v2_f32")]]
kernel void hf2q_mul_mm_tensor_v2_impl<block_iq4_nl, 2, dequantize_iq4_nl_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

// ---- Kernel instantiations (legacy V1, NR0=64 × NR1=32) ----

template [[host_name("kernel_mul_mm_q4_0_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q4_0, 2, dequantize_q4_0_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q8_0_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q8_0, 2, dequantize_q8_0_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q2_K_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q2_K, QK_NL, dequantize_q2_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q6_K_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q6_K, QK_NL, dequantize_q6_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

// ADR-022 Phase 1 — Q5_1 / IQ4_NL tensor-mm template instantiations.
template [[host_name("kernel_mul_mm_q5_1_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q5_1, 2, dequantize_q5_1_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

// ADR-022 Phase 2 — Q5_K tensor-mm template instantiation.
template [[host_name("kernel_mul_mm_q5_K_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q5_K, QK_NL, dequantize_q5_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

// ADR-022 Phase 3 — Q4_K tensor-mm template instantiation.
template [[host_name("kernel_mul_mm_q4_K_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q4_K, QK_NL, dequantize_q4_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_iq4_nl_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_iq4_nl, 2, dequantize_iq4_nl_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

// ===========================================================================
// Wave P4.19 — bf16-input permuted-021 variant.
//
// Identical compute kernel as `hf2q_mul_mm_tensor_impl` above, EXCEPT the
// B-stage reads src1 as bf16 at the permuted [n_heads, seq_len, head_dim]
// layout that flash-attention emits natively (pf_sdpa_out_perm).  The
// existing f32-input kernel is fed by a dedicated permute_021_bf16_to_f32
// dispatch that (a) transposes to [seq_len, n_heads, head_dim] and
// (b) casts bf16 → f32.  This variant folds both steps into the A-stage
// load by reshaping the threadgroup's index arithmetic and casting one
// bf16 at a time to half (the shmem format).
//
// Byte-exact equivalence (documented in /tmp/cfa-perf-parity/worker-4-spec.md):
//   original path : bf16 -> (cast kernel) f32 -> (mm B-stage) half
//   this variant  : bf16 -> (mm B-stage) half
// Both produce the same half bits.  bfloat->float is pure bit-expansion
// (high 16 bits of f32 = bfloat bits, low 16 = zero); float->half's
// RNE round drops the zero-pad bits without changing the result.  NaN /
// Inf / subnormal behaviour is identical.
//
// Preconditions (caller must guarantee):
//   * hd = args.head_dim is a multiple of NK=32 (sliding: 256, global: 512).
//   * args.ne00 == args.nh * args.head_dim (i.e. K equals the full hidden_size).
//
// When these hold the 8-wide K-stripe every thread loads stays inside a
// single head's contiguous [seq_len, head_dim] slab — the address
// computation cleanly splits into (h = loop_k / hd, f = loop_k % hd + iy),
// with h constant across the 32-wide tile.
// ===========================================================================

struct GgmlMatmulMmTensorPerm021Params {
    int32_t  ne00;       // K (= n_heads * head_dim)
    int32_t  ne02;       // weight batch (unused here; kept for call symmetry)
    uint64_t nb01;       // src0 row stride (bytes)
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;       // = sizeof(bfloat) = 2
    uint64_t nb11;       // src1 per-(head,token)-feature inner stride (bytes) — unused (addressing is manual below)
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;        // N (= hidden_size)
    int32_t  ne1;        // M (= seq_len)
    int16_t  r2;
    int16_t  r3;
    int32_t  head_dim;   // hd — new: feature width per head
    int32_t  seq_len;    // explicit copy of ne1 for strict per-K-tile math
};

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void hf2q_mul_mm_tensor_perm021_impl(
        constant GgmlMatmulMmTensorPerm021Params & args,
        device const char * src0,
        device const char * src1,        // bfloat* at [n_heads, seq_len, head_dim] physical layout
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup half  * sb = (threadgroup half  *)(shmem + 4096);
    threadgroup float * sc = (threadgroup float *)(shmem);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x =
        (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    // B-stage address components for the permuted bf16 src1.
    // iy ∈ {0, 8, 16, 24}; each thread loads 8 contiguous bf16 lanes
    // at K-offset loop_k + iy.  With args.head_dim a multiple of NK=32
    // and iy < NK, the 8-lane stripe stays inside head h = k / hd.
    const short iy = 8 * (tiitg % NL1);

    // Token index (M-row). Constant across K-loop.
    const int t = r1 + lr1;

    // Batch/(r2, r3) broadcast: identical to the f32 kernel.
    device const bfloat * y_base_bf16 = (device const bfloat *)
        (src1 + args.nb13*i13 + args.nb12*i12);

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- Stage A tile (block_q -> half, unchanged from f32 variant) ----
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

        // ---- Stage B tile (bfloat permuted -> half, 8-wide load) ----
        //
        // Map logical K-position k = loop_k + iy to physical:
        //   h = k / hd
        //   f = k % hd          ∈ [0, hd-8]  (since hd % NK==0 and iy+8<=NK<=hd)
        //   byte offset = ((h*seq + t)*hd + f) * 2
        //
        // 8-wide bf16 stripe (16 bytes) at f..f+7 is contiguous and
        // stays inside head h's [seq, hd] slab.  We use a `bfloat4` +
        // `bfloat4` pair to express the vector load without needing a
        // 128-bit bf16 type (Metal does not provide bfloat2x4).
        {
            const int k      = loop_k + (int)iy;
            const int h      = k / args.head_dim;
            const int f      = k - h * args.head_dim;

            device const bfloat * y_b =
                y_base_bf16 + ((long)h * args.seq_len + (long)t) * args.head_dim + f;

            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            threadgroup half * sb_ptr = sb + NK*(8*sy + ly) + 8*sx;

            // Per-element bf16 -> float -> half cast chain.  The explicit
            // bf16→float step is REQUIRED for byte-exact equivalence with
            // the eliminated `permute_021_bf16_to_f32` + f32-input-mm
            // pair: the old path routes through f32 before f32→half RNE
            // rounding in the original mm B-stage; a direct
            // `(half)(bfloat)` cast in some Metal SDKs uses a different
            // intermediate representation and produces different low-bit
            // rounding for ~1% of values in the tested Gemma 4 pp2455
            // trajectory (verified empirically 2026-04-20: direct cast
            // flipped first_token from 29294 to 236772; explicit float
            // step restores byte-identity).
            sb_ptr[0] = (half)(float)y_b[0];
            sb_ptr[1] = (half)(float)y_b[1];
            sb_ptr[2] = (half)(float)y_b[2];
            sb_ptr[3] = (half)(float)y_b[3];
            sb_ptr[4] = (half)(float)y_b[4];
            sb_ptr[5] = (half)(float)y_b[5];
            sb_ptr[6] = (half)(float)y_b[6];
            sb_ptr[7] = (half)(float)y_b[7];
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
        // ADR-040 §0.19: post-mm.run barrier (see V2 path + llama
        // ggml-metal.metal:9549) — prevents the next K-iter from overwriting
        // sa/sb while mm.run may still read them under GPU contention.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write-back (identical to f32 variant) ----
    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *) dst +
            r0 +
            r1 * args.ne0 + im*args.ne1*args.ne0;

        auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(args.ne0, NR1));
        cT.store(tC);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
        cT.store(tC);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = sc + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

template [[host_name("kernel_mul_mm_q4_0_tensor_bf16_perm021")]]
kernel void hf2q_mul_mm_tensor_perm021_impl<block_q4_0, 2, dequantize_q4_0_t>(
    constant GgmlMatmulMmTensorPerm021Params &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q6_K_tensor_bf16_perm021")]]
kernel void hf2q_mul_mm_tensor_perm021_impl<block_q6_K, QK_NL, dequantize_q6_K_t>(
    constant GgmlMatmulMmTensorPerm021Params &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

// ADR-022 Phase 3 — Q8_0 perm021 template instantiation. Used by ADR-013
// P21 attention Q@K^T when the K matrix is stored as Q8_0 (currently
// only Q4_0 + Q6_K had this kernel). Same impl as the dense tensor mm,
// only the B-stage reads bf16 from a permuted [n_heads, seq_len, head_dim]
// layout (see hf2q_mul_mm_tensor_perm021_impl in this file at line 528).
template [[host_name("kernel_mul_mm_q8_0_tensor_bf16_perm021")]]
kernel void hf2q_mul_mm_tensor_perm021_impl<block_q8_0, 2, dequantize_q8_0_t>(
    constant GgmlMatmulMmTensorPerm021Params &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

// ===========================================================================
// ADR-029 iter-36 H28-D: F16-shadow variant of the bf16-input perm021 mm.
//
// Identical geometry and B-stage to `hf2q_mul_mm_tensor_perm021_impl`, with
// the per-K-tile quantized dequant replaced by a direct half load from the
// F16 shadow buffer.  Used when `MlxQWeight.f16_shadow` is `Some` and the
// caller routes through `dispatch_mm_perm021_f16` (m > threshold).
//
// Layout invariants (must match the F16 shadow caller):
//   - src0 is `half [n_out, k]` row-major, nb01 = k * 2 bytes.
//   - src1 is bfloat at physical `[n_heads, seq_len, head_dim]`, same as
//     the quantized variant (B-stage logic is byte-identical).
//   - dst is f32 `[n_batch, m, n_out]` row-major, output layout unchanged.
//
// Byte-exact equivalence with quantized perm021 path: half values dequant'd
// from block_q at load time (H29) match the per-call dequantize_func output
// up to the same round-half-even truncation.  Verified at integration via
// first-decode-token byte-identity on gemma4 4K prefill.
// ===========================================================================

[[host_name("kernel_mul_mm_f16_tensor_bf16_perm021")]]
kernel void hf2q_mul_mm_tensor_perm021_f16_impl(
        constant GgmlMatmulMmTensorPerm021Params & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup half  * sb = (threadgroup half  *)(shmem + 4096);
    threadgroup float * sc = (threadgroup float *)(shmem);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;  // 2
    constexpr int NL1 = NK/8;   // 4

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    // F16 src0 row pointer for output-row (r0 + lr0).
    // nb01 (caller-set) is the byte stride between weight rows in the F16
    // shadow buffer = k * sizeof(half) = 2*k.
    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    device const half * x_f16 =
        (device const half *)(src0 + args.nb01*(r0 + lr0) + offset0);

    // B-stage address (unchanged from quantized variant).
    const short iy = 8 * (tiitg % NL1);
    const int t = r1 + lr1;
    device const bfloat * y_base_bf16 = (device const bfloat *)
        (src1 + args.nb13*i13 + args.nb12*i12);

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- Stage A tile: 16 halves directly from F16 weight at
        //      K-position loop_k + il0*16, no dequant. ----
        {
            const int k_start = loop_k + (int)il0 * 16;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = x_f16[k_start + i];
            }
        }

        // ---- Stage B tile (bfloat permuted -> half) — unchanged from quantized variant ----
        {
            const int k      = loop_k + (int)iy;
            const int h      = k / args.head_dim;
            const int f      = k - h * args.head_dim;

            device const bfloat * y_b =
                y_base_bf16 + ((long)h * args.seq_len + (long)t) * args.head_dim + f;

            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            threadgroup half * sb_ptr = sb + NK*(8*sy + ly) + 8*sx;

            sb_ptr[0] = (half)(float)y_b[0];
            sb_ptr[1] = (half)(float)y_b[1];
            sb_ptr[2] = (half)(float)y_b[2];
            sb_ptr[3] = (half)(float)y_b[3];
            sb_ptr[4] = (half)(float)y_b[4];
            sb_ptr[5] = (half)(float)y_b[5];
            sb_ptr[6] = (half)(float)y_b[6];
            sb_ptr[7] = (half)(float)y_b[7];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
        // ADR-040 §0.19: post-mm.run barrier (see V2 path + llama
        // ggml-metal.metal:9549) — prevents the next K-iter from overwriting
        // sa/sb while mm.run may still read them under GPU contention.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write-back (identical to quantized variant) ----
    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *) dst +
            r0 +
            r1 * args.ne0 + im*args.ne1*args.ne0;

        auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(args.ne0, NR1));
        cT.store(tC);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
        cT.store(tC);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = sc + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}
