// dense_mm_f16_tensor.metal — Dense f16×f32 → f32 tensor-API matmul.
//
// Port of llama.cpp's `kernel_mul_mm_f16_f32` template instantiation
// (ggml/src/ggml-metal/ggml-metal.metal:10099):
//
//   template [[host_name("kernel_mul_mm_f16_f32")]]
//     kernel mul_mm_t kernel_mul_mm<
//         half, half4x4, simdgroup_half8x8,
//         half, half2x4, simdgroup_half8x8,
//         half4x4, 1, dequantize_f16,
//         half, half4x4, float, float2x4>;
//
// Peer's `kernel_mul_mm_f16_f32` stages BOTH the A-tile (weight, half
// in DRAM) AND the B-tile (activation, float in DRAM, cast to half on
// stage) as `half` in shmem; the simdgroup MMA uses `simdgroup_half8x8`
// with a `float4x4` accumulator. Effective per-element rounding is
// 10-bit mantissa (half), 8× tighter than BF16 sibling's 7-bit mantissa.
//
// hf2q's gemma4v ViT: every weight (`v.blk.NN.attn_*.weight`,
// `v.blk.NN.ffn_*.weight`, `v.patch_embd.weight`,
// `mm.input_projection.weight`) is stored as F16 in the mmproj GGUF
// (verified via gguf_dump.py). Pre-iter-128, hf2q dequantized F16→F32
// at load and then re-cast F32→BF16 in `vit_linear_gpu`, accepting the
// 8× rounding budget delta. Per the iter-127 numerical bisect that
// confirmed BF16 staging compounds 1.16×/block × 27 blocks = 65× cascade
// (block_26 max_abs 733), this kernel keeps the F16 path end-to-end
// in shmem, matching the peer's per-element precision exactly.
//
// Tile geometry mirrors `dense_mm_bf16_tensor.metal` exactly:
//   NR0 = 64  (output rows per threadgroup, stride along ne0)
//   NR1 = 32  (output cols per threadgroup, stride along ne1)
//   NK  = 32  (K-dim tile depth)
//
// Threadgroup memory layout (8 KB total):
//   sa (A tile):   half [NR0=64][NK=32] row-major — 4 KB
//   sb (B tile):   half [NR1=32][NK=32] row-major — 4 KB
//   sc (writeback): float [NR0][NR1]              — 8 KB (reuses sa+sb)
//
// Identical to BF16 sibling's geometry — half and bfloat share the
// 16-bit storage size, so byte offsets and stride math are unchanged.
// The ONLY semantic differences are (a) src0 device reads are `half`
// (matches GGUF F16 storage), (b) shmem A/B tiles are `half`, and (c)
// the simdgroup MMA accumulator template parameter is `half`.
//
// ne02 / r2 broadcast: matches BF16 sibling.  hf2q's ViT does NOT use
// GQA (nh = nkv per `clip.cpp:Gemma4VHead`), so r2 = 1 in production;
// the broadcast plumbing is preserved for parity with the BF16/F32
// siblings and so a future GQA vision tower can reuse this kernel.
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
// Byte-identical to the BF16 sibling (DenseMmBf16F32TensorParams) so the
// Rust dispatcher can share a #[repr(C)] mirror with the only difference
// being the dtype interpretation of the buffers.  See `dense_mm_bf16.rs`
// for the field-by-field commentary.

struct DenseMmF16F32TensorParams {
    int32_t  ne00;   // K (contract dim)
    int32_t  ne02;   // src0 batch count (e.g. nkv)
    uint64_t nb01;   // src0 row stride (bytes) = ne00 * sizeof(half)
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
// for every b in [0, ne12).  src0 is half  [ne02, ne0, ne00] row-major,
// src1 is float [ne12, ne1, ne00] row-major, dst is float
// [ne12, ne1, ne0] row-major.

kernel void hf2q_dense_mm_f16_f32_tensor(
        constant DenseMmF16F32TensorParams & args,
        device const char * src0,        // half  [ne02, ne0, ne00]
        device const char * src1,        // float [ne12, ne1, ne00]
        device       char * dst,         // float [ne12, ne1, ne0]
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

    // src0: half row-major, row = (r0 + lr0), batch offset from im->i12/r2.
    // Each thread owns 16 consecutive halves within its tile row, offset
    // by il0 * 16 within the row so x[i] (i=0..15) covers absolute
    // K = loop_k + il0*16 + i.
    device const half * x = (device const half *)(src0 + args.nb01*(r0 + lr0) + offset0) + il0 * 16;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK,  NR0));
    auto tB = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sb, dextents<int32_t, 2>(NR1, NK ));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // Full-tile fast path when the entire NK=32 K-block fits inside
        // ne00; gated slow path for the partial trailing tile when ne00
        // is not a multiple of NK.  Same K-tile-bounds invariant as the
        // BF16 sibling (hf2q ADR-005 iter 67 bisection).
        const bool full_tile = (loop_k + NK <= args.ne00);

        // ---- Stage A tile (half -> half copy into sa [NR0][NK]).
        // No dequantize: A is already half in device memory.  Same
        // destination layout as the BF16 sibling (and the quantized
        // tensor kernel after dequant):
        //   sa[NK*(8*sy + ly) + 8*sx + lx] = x[i]
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
                // Partial tile: gate per-element. This thread's x[i]
                // covers absolute K = loop_k + il0*16 + i.
                for (short i = 0; i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short sy = (tiitg/NL0)/8;
                    const short lx = i%8;
                    const short ly = (tiitg/NL0)%8;
                    const int abs_k = loop_k + il0*16 + i;
                    const half v = (abs_k < args.ne00) ? x[i] : half(0.0);
                    *(sa + NK*(8*sy + ly) + 8*sx + lx) = v;
                }
            }
        }

        // ---- Stage B tile (f32 -> half per-element cast into sb) ----
        // Same per-element float→half pattern as the BF16 sibling's
        // float→bfloat path. Metal compiler packs 8 sequential 16-bit
        // stores into a single 128-bit threadgroup write; no half2x4
        // matrix store path is available (parallel to BF16's missing
        // bfloat2x4 — see BF16 sibling line 196-201 for context).
        {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            threadgroup half * sb_ptr = sb + NK*(8*sy + ly) + 8*sx;

            if (full_tile) {
                float4 y_lo = *((device const float4 *) y);
                float4 y_hi = *((device const float4 *)(y + 4));

                sb_ptr[0] = half(y_lo[0]);
                sb_ptr[1] = half(y_lo[1]);
                sb_ptr[2] = half(y_lo[2]);
                sb_ptr[3] = half(y_lo[3]);
                sb_ptr[4] = half(y_hi[0]);
                sb_ptr[5] = half(y_hi[1]);
                sb_ptr[6] = half(y_hi[2]);
                sb_ptr[7] = half(y_hi[3]);
            } else {
                // Partial tile: y[i] for thread (tiitg%NL1) covers
                // absolute K = loop_k + iy + i.
                for (short i = 0; i < 8; i++) {
                    const int abs_k = loop_k + iy + i;
                    sb_ptr[i] = (abs_k < args.ne00) ? half(y[i]) : half(0.0);
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
