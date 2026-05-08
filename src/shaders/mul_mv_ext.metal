// ADR-022 Phase 1 P1.7 — `mul_mv_ext` r1 family for Q5_1 + IQ4_NL.
//
// Direct port of llama.cpp's `kernel_mul_mv_ext_q4_f32_impl`
// (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:3662-3761`) +
// the eight `template [[host_name(...)]]` instantiations at :3936-3939
// (Q5_1 × r1∈{2,3,4,5}) and :3951-3954 (IQ4_NL × r1∈{2,3,4,5}).
//
// Why a separate kernel from `kernel_mul_mv_<q>_f32`:
//   The plain mv kernel computes one src1 row at a time (m=1 decode).
//   At small batch m∈{2,3,4,5} (speculative decode, batched server, MTP),
//   the per-row dispatch overhead dominates. mv_ext processes `r1ptg` src1
//   rows in parallel per simdgroup, sharing the dequantized src0 weights
//   across all r1ptg rows. llama.cpp's host dispatcher (ggml-metal-ops.cpp:2107)
//   routes the m=2..8 range to mv_ext when ne01 % nsg*nypsg == 0.
//
// Function constants:
//   FC_mul_mv_nsg   (i32 @ 600) — num simdgroups per threadgroup; llama.cpp
//                                 always passes 2 (`ggml-metal-ops.cpp:2090`).
//   FC_mul_mv_nxpsg (i32 @ 601) — num threads along row per simdgroup;
//                                 llama.cpp picks ∈ {4, 8, 16} by K-modulus.
//
// Note on data type: llama.cpp declares these as `short` (i16). mlx-native's
// `KernelRegistry::get_pipeline_with_constants` infrastructure currently
// supports bool + i32; we declare them as `int [[function_constant(...)]]`
// here and the kernel uses them as scalar `short`-range values via implicit
// conversion. Identical behavior at run time; nothing branches on the
// type difference.
//
// Mantra: chesterton's fence — every shape-rule, every offset, every shift
// here is a 1:1 byte-port from the llama.cpp peer source. No improvisation.

#include <metal_stdlib>
using namespace metal;

#define QK4_0 32
#define QK_K  256

// ------- block typedefs (byte-for-byte GGUF layout) -------

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

// IQ4_NL non-linear codebook (frozen in ggml-common.h:1109-1112). Lock-step
// duplicate of the values at:
//   - quantized_matmul_id_mm.metal kvalues_iq4nl
//   - quantized_matmul_mm.metal     kvalues_iq4nl
//   - quantized_matmul_mm_tensor.metal kvalues_iq4nl
//   - quantized_matmul_id_ggml.metal kvalues_iq4nl
//   - quantized_matmul_ggml.metal    kvalues_iq4nl
//   - host: src/gguf/mod.rs::KVALUES_IQ4_NL
//
// llama.cpp uses a `float` array here for fast multiply-accumulate in the
// dequant_t4 inner loop; the existing mlx-native callers use `int8_t` then
// cast to float per-element. Float-form matches llama.cpp's mv_ext path
// exactly, so we duplicate the pattern in this file rather than recompute.
constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// ------- 4-element dequantize helpers (peer: ggml-metal.metal:544 + :936) -------

// Q5_1: 32-element block as 8 float4 chunks. `il` ∈ [0, 8) selects which
// 4-element chunk to fill (low half: il < 4, high half: il ≥ 4).
template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)&xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

// IQ4_NL: 32-element block as 8 float4 chunks. `il` ∈ [0, 8) — il/4 selects
// nibble (low / high), il%4 selects the float4 chunk within the half.
template <typename type4>
void dequantize_iq4_nl_t4(device const block_iq4_nl * xb, short il, thread type4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

// ------- function constants (per-PSO compile-time) -------

constant int FC_mul_mv_nsg   [[function_constant(600)]];
constant int FC_mul_mv_nxpsg [[function_constant(601)]];

// ------- args struct (mirrors ggml_metal_kargs_mul_mv_ext) -------
//
// Same field order as `ggml-metal-impl.h:475-494`. Padding inserted by
// the Metal compiler to honor the natural alignment of int64_t fields
// matches the C++ layout (verified by sizeof(args) check on host side).

struct hf2q_mul_mv_ext_args {
    int      ne00;
    int      ne01;
    int      ne02;
    ulong    nb00;
    ulong    nb01;
    ulong    nb02;
    ulong    nb03;
    int      ne10;
    int      ne11;
    int      ne12;
    ulong    nb10;
    ulong    nb11;
    ulong    nb12;
    ulong    nb13;
    int      ne0;
    int      ne1;
    short    r2;
    short    r3;
};

// ------- kernel template (port of ggml-metal.metal:3662) -------
//
// r1ptg : number of src1 rows processed per threadgroup (matches FC's
//         template-parameter analogue but is a concrete template arg here
//         since the host name encodes it).
// q_t   : block weight type (block_q5_1 or block_iq4_nl).
// chpb  : chunks per block = block_QK / 4 = 8 for Q5_1 / IQ4_NL (32 / 4).
// deq_t4: 4-element dequant function pointer.
template <short r1ptg, typename q_t, short chpb,
          void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void hf2q_mul_mv_ext_q4_f32_impl(
        constant hf2q_mul_mv_ext_args & args   [[buffer(0)]],
        device const char * src0               [[buffer(1)]],
        device const char * src1               [[buffer(2)]],
        device       char * dst                [[buffer(3)]],
        uint3   tgpig                          [[threadgroup_position_in_grid]],
        ushort  tiisg                          [[thread_index_in_simdgroup]],
        ushort  sgitg                          [[simdgroup_index_in_threadgroup]]) {
    const short NSG   = (short) FC_mul_mv_nsg;
    const short nxpsg = (short) FC_mul_mv_nxpsg;

    const short chpt = 4;                 // chunks per thread
    const short nypsg = (32 / nxpsg);

    const short tx = tiisg % nxpsg;
    const short ty = tiisg / nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m % args.ne12;
    const int i13 = i1m / args.ne12;

    const ulong offset0 = i01*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const ulong offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01)
        ? (device const q_t *) (src0 + offset0) + tx/chpb
        : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11)
            ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx
            : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { 0.0f };

    short cch = tx % chpb;

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch / chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt * nxpsg;
        }
    }

    // simdgroup reduction along the row dimension (nxpsg threads).
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        if (nxpsg >= 16) sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        if (nxpsg >=  8) sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        if (nxpsg >=  4) sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        if (nxpsg >=  2) sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst
                + (ulong)i1m*args.ne0*args.ne1
                + (ulong)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// ------- 8 instantiations (Q5_1 × r1∈{2,3,4,5} + IQ4_NL × r1∈{2,3,4,5}) -------
//
// chpb = QK4_0 / 4 = 32 / 4 = 8 for both types (32-element block,
// 4-element chunk → 8 chunks per block).

// --- Q5_1 ---

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_2")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<2, block_q5_1, 8, dequantize_q5_1_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_3")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<3, block_q5_1, 8, dequantize_q5_1_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_4")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<4, block_q5_1, 8, dequantize_q5_1_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_5")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<5, block_q5_1, 8, dequantize_q5_1_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);

// --- IQ4_NL ---

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<2, block_iq4_nl, 8, dequantize_iq4_nl_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<3, block_iq4_nl, 8, dequantize_iq4_nl_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<4, block_iq4_nl, 8, dequantize_iq4_nl_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]]
kernel void hf2q_mul_mv_ext_q4_f32_impl<5, block_iq4_nl, 8, dequantize_iq4_nl_t4>(
    constant hf2q_mul_mv_ext_args &, device const char *, device const char *,
    device char *, uint3, ushort, ushort);
