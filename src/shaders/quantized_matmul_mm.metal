// quantized_matmul_mm.metal — GGML block-format quantized matrix-matrix kernels.
//
// Ports llama.cpp's `kernel_mul_mm_<qtype>_f32` matmul kernel
// (ggml/src/ggml-metal/ggml-metal.metal:9276) to mlx-native.
//
// Why mm and not mv?  The existing `kernel_mul_mv_q*_f32` family
// (quantized_matmul_ggml.metal) re-loads each weight block once per prompt
// token from DRAM.  At prefill m=2455 that is a ~32x read amplification
// vs llama.cpp's mm variant, which stages a 64x32 tile into threadgroup
// shared memory and reuses it across a whole 32-row block of the prompt.
// Switching the prefill path from mv to mm closes most of the 7x tok/s
// gap vs llama.cpp (see ADR-011 phase 3 investigation).
//
// Port rules:
//   * Bit-exact against llama.cpp at the kernel level: identical NK/NR0/NR1,
//     identical dequantize formulas, identical accumulation order.  Tests
//     verify the output matches the existing mv kernel (which is itself
//     byte-for-byte with the llama.cpp mv path) to f32 tolerance.
//   * We use the non-tensor (simdgroup_multiply_accumulate) code path.
//     llama.cpp gates its tensor-API path behind GGML_METAL_HAS_TENSOR;
//     mlx-native does not enable that path, so we drop it and use only
//     the simdgroup MMA implementation.
//   * We always enable the bounds-checked input / output paths.  llama.cpp
//     keys these on FC_mul_mm_bc_inp / FC_mul_mm_bc_out function constants
//     to skip the branches when shapes are aligned; for the port we keep
//     a single correct variant.  Overhead is a couple of predicates per
//     32-wide tile; orders of magnitude below the DRAM savings vs mv.
//
// Portions of this file are derived from llama.cpp
// (https://github.com/ggerganov/llama.cpp), MIT licensed.
// Original source: ggml/src/ggml-metal/ggml-metal.metal.
// Copyright the llama.cpp Authors.  See LICENSE-MIT-llamacpp.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// ---- GGML block sizes (must match quantized_matmul_ggml.metal) ----

#define QK4_0 32
#define QK8_0 32
#define QK_K  256

// Matches llama.cpp's QK_NL = 16 (the "nibbles per lane" step for K-quant
// block deqauntize).
#define QK_NL 16

// ---- Host-facing params struct ----
//
// Mirrors llama.cpp's `ggml_metal_kargs_mul_mm`.  We include a handful of
// extra fields (nb0x, ne0x) so mlx-native can pass the kernel the
// byte-strides it needs without a second dispatch.  Field names match
// llama.cpp 1:1 where possible.

struct GgmlMatmulMmParams {
    int32_t  ne00;  // K
    int32_t  ne02;  // batch(src0)  (always 1 for our projections)
    uint64_t nb01;  // bytes per weight row (= blocks_per_row * block_bytes)
    uint64_t nb02;  // bytes per weight batch (unused, kept for symmetry)
    uint64_t nb03;  // unused
    int32_t  ne12;  // batch(src1)  (always 1)
    uint64_t nb10;  // = sizeof(float) = 4
    uint64_t nb11;  // bytes per input row (= K * sizeof(float))
    uint64_t nb12;  // bytes per input batch (unused)
    uint64_t nb13;  // unused
    int32_t  ne0;   // N (output stride)
    int32_t  ne1;   // M (number of input rows)
    int16_t  r2;    // 1
    int16_t  r3;    // 1
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

// ---- Dequantize helpers (llama.cpp, ggml-metal.metal) ----
//
// Each helper reads a single 16-element slice of a block (indexed by
// `il`) and produces a 4x4 tile of values in the caller's output dtype.

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
void dequantize_q8_0(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }

    reg = (type4x4) reg_f;
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

// ---- kernel_mul_mm template ----
//
// Directly ports llama.cpp ggml-metal.metal:9276 (the non-tensor path, with
// FC_mul_mm_bc_inp = true and FC_mul_mm_bc_out = true always enabled for
// safety).  One threadgroup computes a 64x32 tile of the output (NR0 x NR1).
// Four simdgroups (128 threads total) cooperate on the tile.  Each loop
// iteration stages an NK=32 wide slice of A and B into threadgroup shared
// memory (via dequantize), then runs 8 simdgroup_multiply_accumulate
// 8x8 MMA ops covering the 64x32 output.

//
// Template parameters (reduced vs llama.cpp since we always use T0=half and
// f32 output):
//   * block_q           — e.g. block_q4_0, block_q8_0, block_q6_K.
//   * nl                — the number of 16-element slices per block
//                          (2 for Q4_0/Q8_0; QK_NL=16 for Q6_K).
//   * dequantize_func   — points at one of the dequantize helpers above.
//
// Runtime params (via GgmlMatmulMmParams):
//   * args.ne00         — K
//   * args.ne0          — N (output stride)
//   * args.ne1          — M (number of input rows)
//   * args.nb01         — bytes per weight row
//   * args.nb11         — bytes per input row
//   * args.ne12, r2, r3 — batch bookkeeping (always 1 for our projections)
//
// Dispatch geometry (threads_per_tg = 128, threadgroups = (ceil(M/32), ceil(N/64), 1)):
//   * tgpig.x = r1 / NR1      (m-tile index)
//   * tgpig.y = r0 / NR0      (n-tile index)
//   * tgpig.z = im            (batch, always 0)
//
// Threadgroup shmem layout (matches llama.cpp byte layout):
//   * sa: half at +0     — A tile (64 rows x 32 K-slots), 4096 bytes
//   * sb: float at +4096 — B tile (32 rows x 32 K-slots), 4096 bytes
//   * (Write-back on small tiles reuses sa+sb for partial-tile staging.)

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void hf2q_mul_mm_impl(
        constant GgmlMatmulMmParams & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    // A tile (half), 64 rows x 32 K-slots = 2048 halfs = 4096 bytes.
    threadgroup half  * sa = (threadgroup half  *)(shmem);
    // B tile (float), 32 rows x 32 K-slots = 1024 floats = 4096 bytes.
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;  // 2  — A tile slices (16 elements wide each)
    constexpr int NL1 = NK/8;   // 4  — B tile slices (8 elements wide each)

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;  // first output row in N
    const int r1 = tgpig.x * NR1;  // first output row in M

    // If this block is 64x32 or smaller, clamp valid extents.
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // A thread shouldn't load data outside of the matrix.
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0..63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0..31

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

    // MMA accumulators (non-tensor path).
    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- Stage A tile (block_q -> half, via dequantize_func) ----
        {
            half4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // ---- Stage B tile (float input, bounds-checked for K tail) ----
        //
        // llama.cpp switches between a fast-path 2x4 vector store and a
        // per-element loop keyed on FC_mul_mm_bc_inp.  For correctness at
        // K not divisible by NK, we always take the per-element path.
        // Our K values (2048, 2112, 2816, 4096) *are* multiples of NK=32,
        // but keep the bounds check for defensive safety.
        for (short i = 0; i < 8; ++i) {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short lx = i;
            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(sb + 64*ib + 8*ly + lx) =
                (loop_k + iy + i < args.ne00) ? *((device float *) y + i) : 0.f;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Multiply: 4 simdgroups each own 4 (A) x 2 (B) = 8 MMA tiles ----
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

    // ---- Write-back ----
    //
    // Fast path: full 64x32 tile, direct simdgroup_store to device memory.
    // Slow path: partial tile at M or N edges, stage to shmem and scalar-copy.
    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) +
            (r1 + 16*(sgitg >> 1)) * args.ne0 + im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage the output tile into shmem (reusing sa+sb space), then the
        // first simdgroup copies rows out with M-bound.
        threadgroup float * temp_str = ((threadgroup float *) shmem)
            + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            // When sgitg==0 the per-simdgroup offset below is 0, so temp_str
            // coincides with (threadgroup float *) shmem.  llama.cpp reads
            // via `temp_str + j*NR0` but effectively that equals
            // `(threadgroup float *) shmem + j*NR0` here.
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ne0 + im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
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

// ---- Template instantiations ----
//
// Follow llama.cpp's host_name convention: kernel_mul_mm_<qtype>_f32.  The
// kernel registry in src/kernel_registry.rs maps these names to the shader
// source.

template [[host_name("kernel_mul_mm_q4_0_f32")]]
kernel void hf2q_mul_mm_impl<block_q4_0, 2, dequantize_q4_0>(
    constant GgmlMatmulMmParams &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q8_0_f32")]]
kernel void hf2q_mul_mm_impl<block_q8_0, 2, dequantize_q8_0>(
    constant GgmlMatmulMmParams &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mm_q6_K_f32")]]
kernel void hf2q_mul_mm_impl<block_q6_K, QK_NL, dequantize_q6_K>(
    constant GgmlMatmulMmParams &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort);
