// dense_mm_f32_f32.metal — Dense f32×f32 → f32 tensor-API matmul.
//
// Port of llama.cpp's `kernel_mul_mm_f32_f32` template instantiation
// (ggml/src/ggml-metal/ggml-metal.metal:10098):
//
//   template [[host_name("kernel_mul_mm_f32_f32")]]
//     kernel mul_mm_t kernel_mul_mm<
//         half, half4x4, simdgroup_half8x8,
//         half, half2x4, simdgroup_half8x8,
//         float4x4, 1, dequantize_f32,
//         float, float4x4, float, float2x4>;
//
// llama.cpp's f32_f32 specialization downcasts both A and B to half in
// shared memory before the simdgroup MMA, then accumulates into float.
// We instead keep the staged tiles as float for the tensor-API
// matmul2d path — the M3+ Apple Silicon tensor-ops dispatch supports
// `float`-typed shmem tiles natively, and avoids the half-cast
// precision loss that the F32-mode ViT attention diagnostic
// specifically wants to remove (ADR-005 iter-118 BF16-vs-F32 A/B).
//
// Tile geometry mirrors `dense_mm_bf16_tensor.metal` exactly:
//   NR0 = 64  (output rows per threadgroup, stride along ne0)
//   NR1 = 32  (output cols per threadgroup, stride along ne1)
//   NK  = 32  (K-dim tile depth)
//
// Threadgroup memory layout (16 KB total):
//   sa (A tile):   float [NR0=64][NK=32] row-major — 8 KB
//   sb (B tile):   float [NR1=32][NK=32] row-major — 4 KB
//   sc (writeback): float [NR0][NR1]               — 8 KB (reuses sa)
//
// Both A and B are f32 device-side; staging is a plain copy.  The
// BF16 sibling does an extra cast on the B-stage path; here that
// becomes a direct float load/store.
//
// ne02 / r2 broadcast: matches BF16 sibling.  hf2q's ViT and GQA
// code paths broadcast nkv KV heads across nh query heads via
// r2 = nh / nkv (same contract as llama.cpp's ggml_mul_mat).
//
// Portions derived from llama.cpp (https://github.com/ggml-org/llama.cpp),
// MIT licensed.  Copyright the llama.cpp Authors.  See LICENSE-MIT-llamacpp.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// ---- Host-facing params struct ---------------------------------------
//
// Byte-identical to the BF16 sibling (DenseMmBf16F32TensorParams) so
// the Rust dispatcher can share a #[repr(C)] mirror with the only
// difference being the dtype interpretation of the buffers.  See
// `dense_mm_bf16.rs` for the field-by-field commentary.

struct DenseMmF32F32TensorParams {
    int32_t  ne00;   // K (contract dim)
    int32_t  ne02;   // src0 batch count (e.g. nkv)
    uint64_t nb01;   // src0 row stride (bytes) = ne00 * sizeof(float)
    uint64_t nb02;   // src0 batch stride (bytes)
    uint64_t nb03;   // (unused, reserved for 4-D)
    int32_t  ne12;   // src1 batch count (e.g. nh)
    uint64_t nb10;   // src1 element stride = sizeof(float)
    uint64_t nb11;   // src1 row stride (bytes)
    uint64_t nb12;   // src1 batch stride (bytes)
    uint64_t nb13;   // (unused, reserved for 4-D)
    int32_t  ne0;    // N (output cols = src0 rows)
    int32_t  ne1;    // M (output rows = src1 rows)
    int16_t  r2;     // ne12 / ne02 (GQA head broadcast factor)
    int16_t  r3;     // (unused, reserved for higher-dim broadcast)
};

// ---- Kernel --------------------------------------------------------
//
// Computes  dst[b, m, n] = sum_k src0[b/r2, n, k] * src1[b, m, k]
// for every b in [0, ne12).  src0 is f32 [ne02, ne0, ne00] row-major,
// src1 is f32 [ne12, ne1, ne00] row-major, dst is f32 [ne12, ne1, ne0]
// row-major.

kernel void hf2q_dense_mm_f32_f32_tensor(
        constant DenseMmF32F32TensorParams & args,
        device const char * src0,        // float [ne02, ne0, ne00]
        device const char * src1,        // float [ne12, ne1, ne00]
        device       char * dst,         // float [ne12, ne1, ne0]
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup float * sa = (threadgroup float *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 8192); // sa = 64*32*4 = 8 KB
    threadgroup float * sc = (threadgroup float *)(shmem);        // partial-tile write-back reuses sa region

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

    // src0: f32 row-major, row = (r0 + lr0), batch offset from im->i12/r2.
    // Each thread owns a 16-element K-column group within a tile row;
    // x is initialized at offset il0*16 within the row so x[i] (i=0..15)
    // covers absolute K = loop_k + il0*16 + i.
    device const float * x = (device const float *)(src0 + args.nb01*(r0 + lr0) + offset0) + il0 * 16;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    auto tA = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // Full-tile fast path when the entire NK=32 K-block fits inside
        // ne00; gated slow path for the partial trailing tile when
        // ne00 is not a multiple of NK.  Same K-tile-bounds invariant
        // as the BF16 sibling (hf2q ADR-005 iter 67 bisection).
        const bool full_tile = (loop_k + NK <= args.ne00);

        // ---- Stage A tile (f32 -> f32 copy into sa [NR0][NK]).
        // Layout matches the BF16 sibling and the quantized tensor
        // kernel:  sa[NK*(8*sy + ly) + 8*sx + lx] = x[i]
        // with (sx, sy, lx, ly) derived from (tiitg, i, il0).
        {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (full_tile) {
                for (short i = 0; i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short sy = (tiitg/NL0)/8;
                    const short lx = i%8;
                    const short ly = (tiitg/NL0)%8;
                    *(sa + NK*(8*sy + ly) + 8*sx + lx) = x[i];
                }
            } else {
                for (short i = 0; i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short sy = (tiitg/NL0)/8;
                    const short lx = i%8;
                    const short ly = (tiitg/NL0)%8;
                    const int abs_k = loop_k + il0*16 + i;
                    const float v = (abs_k < args.ne00) ? x[i] : 0.0f;
                    *(sa + NK*(8*sy + ly) + 8*sx + lx) = v;
                }
            }
        }

        // ---- Stage B tile (f32 -> f32 copy into sb [NR1][NK]) ----
        // Each thread covers 8 contiguous f32 elements via two float4
        // loads, written as a single float8-equivalent store (Metal
        // compiler packs the two float4 stores into one 32-byte
        // threadgroup write).
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            threadgroup float * sb_ptr = sb + NK*(8*sy + ly) + 8*sx;

            if (full_tile) {
                float4 y_lo = *((device const float4 *) y);
                float4 y_hi = *((device const float4 *)(y + 4));

                sb_ptr[0] = y_lo[0];
                sb_ptr[1] = y_lo[1];
                sb_ptr[2] = y_lo[2];
                sb_ptr[3] = y_lo[3];
                sb_ptr[4] = y_hi[0];
                sb_ptr[5] = y_hi[1];
                sb_ptr[6] = y_hi[2];
                sb_ptr[7] = y_hi[3];
            } else {
                for (short i = 0; i < 8; i++) {
                    const int abs_k = loop_k + iy + i;
                    sb_ptr[i] = (abs_k < args.ne00) ? y[i] : 0.0f;
                }
            }
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
