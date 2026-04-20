// dense_mm_bf16_tensor.metal — Dense bf16×f32 → f32 tensor-API matmul.
//
// Port of llama.cpp's `kernel_mul_mm_bf16_f32` template instantiation
// (ggml/src/ggml-metal/ggml-metal.metal:10032) with the
// `GGML_METAL_HAS_TENSOR` branch active.  Tile geometry, shared memory
// layout, and matmul2d descriptor are identical to our existing
// `quantized_matmul_mm_tensor.metal`:
//   * sa (A tile): bfloat, [NR0=64][NK=32] row-major, 4 KB
//   * sb (B tile): bfloat, [NR1=32][NK=32] row-major, 4 KB
//   * sc (partial-tile write-back): float, reuses shmem base
//
// The kernel is used on hf2q's non-flash-attention prefill path for
// BOTH of the two attention mat-muls:
//   1. Q @ K^T -> scores  (K is src0 bf16 weight; Q is src1 f32 input)
//   2. scores @ V -> out  (V is src0 bf16 weight; scores is src1 f32
//      input after softmax)
//
// The non-tensor simdgroup MMA fallback is intentionally NOT included
// — mlx-native targets M3+ where tensor-ops is always available, and
// keeping the kernel single-path avoids the two-branch duplication
// llama.cpp carries for backward compatibility.  If a pre-M3 user ever
// runs this build, kernel compile will fail cleanly and the host-side
// dispatcher (dense_matmul_bf16_f32_tensor_mm) returns an error, and
// the caller must use a different attention path (flash-attn or simd
// MMA mat-mul).
//
// ne02 / r2 broadcast:  hf2q's grouped-query attention has nh heads
// attending but nkv shared KV heads.  The attention mat-muls iterate
// over nh in the z-axis (im = tgpig.z); the src0 head offset divides
// by r2 = nh/nkv so the same KV head is broadcast across all heads in
// its GQA group.  This matches llama.cpp's ggml_mul_mat r2/r3 contract.
//
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp), MIT licensed.
// Copyright the llama.cpp Authors.  See LICENSE-MIT-llamacpp.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// ---- Host-facing params struct ---------------------------------------
//
// Mirrors the ggml matmul args layout we use in quantized_matmul_mm_tensor.
// ne00 = contract dim K (shared between src0 and src1).
// ne0  = output N (= src0.ne[1], number of weight rows).
// ne1  = output M (= src1.ne[1], number of input rows).
// ne02 = src0 batch count (GQA: nkv).
// ne12 = src1 batch count (GQA: nh); r2 = ne12 / ne02.
// nb01 = src0 row stride (bytes) = ne00 * sizeof(bfloat).
// nb02 = src0 batch stride (bytes).
// nb11 = src1 row stride (bytes) = ne00 * sizeof(float).
// nb12 = src1 batch stride (bytes).

struct DenseMmBf16F32TensorParams {
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

// ---- Kernel --------------------------------------------------------
//
// Directly modelled on hf2q_mul_mm_tensor_impl from
// quantized_matmul_mm_tensor.metal.  The ONLY structural differences:
//   * src0 is bfloat instead of block_q; no dequantize is needed, so
//     the A-tile staging is a plain copy loop (bfloat -> bfloat in
//     shmem) rather than the 16-element dequantize + permuted store
//     from the quantized path.
//   * A-stage tile stride: src0 is laid out row-major bfloat, so each
//     thread loads 16 consecutive bfloats from src0 and stores them
//     into sa at the llama.cpp tile-row/tile-col positions matching
//     the tensor_ops matmul2d contract (same as the quantized path
//     after dequantize).

kernel void hf2q_dense_mm_bf16_f32_tensor(
        constant DenseMmBf16F32TensorParams & args,
        device const char * src0,        // bfloat [ne02, ne01, ne00]
        device const char * src1,        // float  [ne12, ne11, ne10]
        device       char * dst,         // float  [batch, ne1, ne0]
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup bfloat * sa = (threadgroup bfloat *)(shmem);
    threadgroup bfloat * sb = (threadgroup bfloat *)(shmem + 4096);
    threadgroup float  * sc = (threadgroup float  *)(shmem);  // partial-tile write-back reuses shmem base

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;   // 2 — threads per A-tile row-block
    constexpr int NL1 = NK/8;    // 4 — threads per B-tile row-block

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

    const uint64_t offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;

    // src0: bfloat row-major, row = (r0 + lr0), batch offset from im->i12/r2.
    // Start at the first bfloat of this thread's row, offset by the K-tile
    // column group that this thread-index owns (16 bfloats per group, so
    // il0 * 16 values = il0 * 16 bfloats in).
    device const bfloat * x = (device const bfloat *)(src0 + args.nb01*(r0 + lr0) + offset0) + il0 * 16;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    auto tA = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- Stage A tile (bfloat -> bfloat copy into sa [NR0][NK]).
        // No dequantize: A is already bfloat in device memory.  Same
        // destination layout as quantized_matmul_mm_tensor.metal:
        //   sa[NK*(8*sy + ly) + 8*sx + lx] = x[i]
        // with (sx, sy, lx, ly) derived from (tiitg, i, il0).
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = x[i];
            }
        }

        // ---- Stage B tile (f32 -> bfloat per-element cast into sb) ----
        // Metal has `float2x4` but no `bfloat2x4` matrix type, so the
        // "single vector store" trick the quantized tensor kernel uses
        // (cast float2x4 -> bfloat2x4) is not available here.  We load
        // 8 f32 values as a float4×2 pair and store them as 8
        // individual bfloats.  The Metal compiler packs this into a
        // half8-equivalent (bfloat shares the 16-bit storage size so
        // the store lowers to a single 128-bit write).
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            float4 y_lo = *((device const float4 *) y);
            float4 y_hi = *((device const float4 *)(y + 4));

            threadgroup bfloat * sb_ptr = sb + NK*(8*sy + ly) + 8*sx;
            sb_ptr[0] = bfloat(y_lo[0]);
            sb_ptr[1] = bfloat(y_lo[1]);
            sb_ptr[2] = bfloat(y_lo[2]);
            sb_ptr[3] = bfloat(y_lo[3]);
            sb_ptr[4] = bfloat(y_hi[0]);
            sb_ptr[5] = bfloat(y_hi[1]);
            sb_ptr[6] = bfloat(y_hi[2]);
            sb_ptr[7] = bfloat(y_hi[3]);
        }

        x += NK;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB = tB.slice(0, 0);
        mm.run(sB, sA, cT);
    }

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
