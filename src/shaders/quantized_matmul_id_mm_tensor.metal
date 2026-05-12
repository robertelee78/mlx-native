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
//
// Portions derived from llama.cpp (MIT).  Copyright the llama.cpp Authors.

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
    int16_t  _pad0;
    int16_t  _pad1;
};

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

constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// Spec source: llama.cpp ggml-metal.metal:675.
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dq_q4_0_id(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
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
void dq_q8_0_id(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }
    reg = (type4x4) reg_f;
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
// quantized_matmul_id_mm.metal (which mirror llama.cpp's
// dequantize_q5_1 at ggml-metal.metal:511 and dequantize_iq4_nl at
// :920). Renamed to dq_<type>_id to follow the file-local convention.

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

// ADR-013 P16 — Q4_K dequant for tensor-API MMA-tile path.
// Spec source: llama.cpp ggml-metal.metal:681 (`dequantize_q4_K`).
template <typename type4x4>
void dq_q4_K_id(device const block_q4_K * xb, short il, thread type4x4 & reg) {
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

// ADR-022 Phase 2 — Q5_K dequant for tensor-API mm_id MMA-tile path.
// Spec source: llama.cpp ggml-metal.metal:699-720 (`dequantize_q5_K`).
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

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
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

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];
    if (r1 >= neh1) return;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im * args.ne21 + r1 + lr1];
    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x =
        (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // Stage A.
        //
        // ADR-029 iter-55 H47 FALSIFIED 2026-05-11: tested adding
        // `#pragma clang loop unroll(full)` here to mirror peer's
        // FOR_UNROLL at /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9903.
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
        // per thread.  Matches llama.cpp's FC_mul_mm_bc_inp=false path.
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
    }

    // Write-back: always through shmem (scatter-by-hids) — same pattern as
    // the simdgroup mm_id version, just cooperative_tensor::store instead
    // of simdgroup_store for the shmem stage.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        auto tC_sm = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NR0, NR1));
        cT.store(tC_sm);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];
        const short ide = id % args.ne20;
        const short idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
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

template [[host_name("kernel_mul_mm_id_q8_0_tensor_f32")]]
kernel void hf2q_mul_mm_id_tensor_impl<block_q8_0, 2, dq_q8_0_id>(
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
