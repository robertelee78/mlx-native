// Dense BF16/F16/F32 matrix-multiply fallback for environments without the
// optional Metal tensor API.  This is the reference non-tensor shape:
// a 64x32 output tile, K=32 staging, and four simdgroups performing 8x8 MMA.
// BF16 and F16 activations are rounded while staged, matching the tensor
// kernels' element semantics; F32 stays F32 end-to-end.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

struct DenseMmFallbackParams {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint32_t _pad0;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    uint32_t _pad1;
};

template<typename S, typename S8x8>
kernel void hf2q_dense_mm_fallback_impl(
        constant DenseMmFallbackParams & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    // A is 64x32; B is 32x32.  For 16-bit S this consumes 6 KiB and
    // partial output uses 8 KiB.  For F32 this consumes 12 KiB.
    threadgroup S * sa = (threadgroup S *)shmem;
    threadgroup S * sb =
        (threadgroup S *)(shmem + sizeof(S) * NR0 * NK);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;
    const short nr0 = min((int)NR0, args.ne0 - r0);
    const short nr1 = min((int)NR1, args.ne1 - r1);
    const short lr0 = min((short)(tiitg / NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / NL1), (short)(nr1 - 1));
    const short il0 = tiitg % NL0;
    const short iy = 8 * (tiitg % NL1);

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;
    const uint64_t offset0 =
        (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;

    device const S * x =
        (device const S *)(src0 + args.nb01 * (r0 + lr0) + offset0)
        + il0 * 16;
    device const float * y = (device const float *)(
        src1 + args.nb13 * i13 + args.nb12 * i12
        + args.nb11 * (r1 + lr1) + args.nb10 * iy);

    S8x8 ma[4];
    S8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // Prevent the next K tile from overwriting shared data before all
        // simdgroups have consumed the previous one.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            const int abs_k = loop_k + il0 * 16 + i;
            sa[64 * ib + 8 * ly + lx] =
                abs_k < args.ne00 ? x[i] : S(0.0f);
        }

        for (short i = 0; i < 8; ++i) {
            const short sx = tiitg % NL1;
            const short sy = (tiitg / NL1) / 8;
            const short lx = i;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            const int abs_k = loop_k + iy + i;
            sb[64 * ib + 8 * ly + lx] =
                abs_k < args.ne00 ? S(y[i]) : S(0.0f);
        }

        x += NK;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const S * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(
                    mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    const uint64_t dst_base =
        (uint64_t)im * (uint64_t)args.ne1 * (uint64_t)args.ne0;
    if (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1) {
        device float * c = (device float *)dst
            + (uint64_t)r0 + (uint64_t)r1 * (uint64_t)args.ne0 + dst_base
            + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * args.ne0;
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(
                mc[i], c + 8 * (i % 4) + 8 * args.ne0 * (i / 4),
                args.ne0, 0, false);
        }
        return;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * tile = (threadgroup float *)shmem
        + 32 * (sgitg & 1) + 16 * (sgitg >> 1) * NR0;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(
            mc[i], tile + 8 * (i % 4) + 8 * NR0 * (i / 4),
            NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1) {
            device float * d = (device float *)dst
                + (uint64_t)r0 + (uint64_t)(r1 + j) * (uint64_t)args.ne0
                + dst_base;
            threadgroup float * c = (threadgroup float *)shmem + j * NR0;
            int i = 0;
            for (; i < nr0 / 4; ++i) {
                ((device float4 *)d)[i] = ((threadgroup float4 *)c)[i];
            }
            i *= 4;
            for (; i < nr0; ++i) {
                d[i] = c[i];
            }
        }
    }
}

template [[host_name("hf2q_dense_mm_bf16_f32_fallback")]]
kernel void hf2q_dense_mm_fallback_impl<bfloat, simdgroup_bfloat8x8>(
    constant DenseMmFallbackParams &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort);

template [[host_name("hf2q_dense_mm_f16_f32_fallback")]]
kernel void hf2q_dense_mm_fallback_impl<half, simdgroup_half8x8>(
    constant DenseMmFallbackParams &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort);

template [[host_name("hf2q_dense_mm_f32_f32_fallback")]]
kernel void hf2q_dense_mm_fallback_impl<float, simdgroup_float8x8>(
    constant DenseMmFallbackParams &, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort);
