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
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;

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

// ---- Kernel instantiations ----

template [[host_name("kernel_mul_mm_q4_0_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q4_0, 2, dequantize_q4_0_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q8_0_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q8_0, 2, dequantize_q8_0_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q6_K_tensor_f32")]]
kernel void hf2q_mul_mm_tensor_impl<block_q6_K, QK_NL, dequantize_q6_K_t>(
    constant GgmlMatmulMmTensorParams &, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);
