// ADR-034 task #93 cont. 26 (2026-05-21) — Fused gate+up+silu_mul IQ4_NL.
//
// IQ4_NL variant of the shipped Q8_0 / Q4_K fused MLP pattern. Saves 2
// Metal launches per layer by computing both gate_proj and up_proj from
// the same input vector in a single kernel + applying silu_mul inline.
//
// Dispatch geometry (matches `kernel_mul_mv_iq4_nl_f32`):
//   - threadgroups   = (ceil(N / 8), M, 1)
//   - threads_per_tg = (8, 8, 1) = 64 threads = 2 simdgroups × 32 threads
//   - NO threadgroup shared memory (each simdgroup writes distinct rows).
//
// Buffer layout:
//   buffer(0): gate_w   device const block_iq4_nl * [I, H_blocks] IQ4_NL
//   buffer(1): up_w     device const block_iq4_nl * [I, H_blocks] IQ4_NL
//   buffer(2): x        device const float *         [H * M] F32
//   buffer(3): out      device       float *         [I * M] F32
//   buffer(4): p        constant GgmlMatvecParams &
//
// Math contract: byte-identical (within F32 FMA tolerance ≤ 1e-5) to:
//   `kernel_mul_mv_iq4_nl_f32(gate_w, x) → tmp_gate`
//   `kernel_mul_mv_iq4_nl_f32(up_w, x)   → tmp_up`
//   `silu_mul_f32(tmp_gate, tmp_up)       → out`

#include <metal_stdlib>
using namespace metal;

#define QK4_0       32
#define N_DST        4   // 4 rows per simdgroup (N_DST_Q4_0 baseline)
#define N_SIMDGROUP  2   // 2 simdgroups per threadgroup → 8 rows/TG
#define N_SIMDWIDTH 32

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(half) + QK4_0 / 2,
              "wrong iq4_nl block size");

// Frozen IQ4_NL codebook (ggml-common.h:1109-1112). Lock-step with the
// kvalues_iq4nl in quantized_matmul_ggml.metal.
constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

struct GgmlMatvecParams {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne10;
    int64_t ne12;
    int64_t ne0;
    int64_t ne1;
    uint    r2;
    uint    r3;
};

constant int FC_qmatmul_ne12 [[function_constant(700)]];
constant int FC_qmatmul_r2   [[function_constant(701)]];
constant int FC_qmatmul_r3   [[function_constant(702)]];
constant int qmatmul_ne12_effective =
    is_function_constant_defined(FC_qmatmul_ne12) ? FC_qmatmul_ne12 : -1;
constant int qmatmul_r2_effective =
    is_function_constant_defined(FC_qmatmul_r2) ? FC_qmatmul_r2 : -1;
constant int qmatmul_r3_effective =
    is_function_constant_defined(FC_qmatmul_r3) ? FC_qmatmul_r3 : -1;

#define QMM_NE12(p) ((qmatmul_ne12_effective >= 0) ? (uint)qmatmul_ne12_effective : (uint)(p).ne12)
#define QMM_R2(p)   ((qmatmul_r2_effective   >= 0) ? (uint)qmatmul_r2_effective   : (uint)(p).r2)
#define QMM_R3(p)   ((qmatmul_r3_effective   >= 0) ? (uint)qmatmul_r3_effective   : (uint)(p).r3)

// IQ4_NL dot helper — ported verbatim from quantized_matmul_ggml.metal:196-210.
inline float block_iq4_nl_dot_y(
    device const block_iq4_nl * qb,
    thread float * yl_raw,
    int il
) {
    float d = qb->d;
    float acc = 0.f;
    device const uint8_t * qs = qb->qs + il;
    for (int i = 0; i < 8; i++) {
        const uint8_t b = qs[i];
        acc += yl_raw[i]     * (float)kvalues_iq4nl[b & 0x0F];
        acc += yl_raw[i + 8] * (float)kvalues_iq4nl[(b >> 4) & 0x0F];
    }
    return d * acc;
}

kernel void kernel_fused_gate_up_silu_iq4_nl_f32(
    device const  void  * gate_w  [[buffer(0)]],
    device const  void  * up_w    [[buffer(1)]],
    device const float  * src1    [[buffer(2)]],
    device       float  * dst     [[buffer(3)]],
    constant GgmlMatvecParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = (int)(p.ne00 / QK4_0);
    const int r0 = (int)tgpig.x;
    const int r1 = (int)tgpig.y;
    const int im = (int)tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = (uint)im % QMM_NE12(p);
    const uint i13 = (uint)im / QMM_NE12(p);

    const uint offset0 = first_row * nb
        + (i12 / QMM_R2(p)) * (nb * p.ne01)
        + (i13 / QMM_R3(p)) * (nb * p.ne01 * p.ne02);

    device const block_iq4_nl * xg = (device const block_iq4_nl *) gate_w + offset0;
    device const block_iq4_nl * xu = (device const block_iq4_nl *) up_w   + offset0;
    device const float        * y  = (device const float        *) src1
        + r1 * p.ne10 + im * p.ne00 * p.ne1;

    float yl_raw[16];
    float sumf_gate[N_DST] = { 0.f, 0.f, 0.f, 0.f };
    float sumf_up  [N_DST] = { 0.f, 0.f, 0.f, 0.f };

    const int ix = (int)tiisg / 2;
    const int il = (int)((tiisg % 2) * 8);

    device const float * yb = y + ix * QK4_0 + il;

    // Stride-loop: each thread covers QK4_0 = 32 quants per iteration;
    // simdgroups interleave across ib by stride nw/2 = 16.
    // Per-ib input loaded ONCE into yl_raw and reused for both
    // weight_a (gate) and weight_b (up) dot products.
    for (int ib = ix; ib < nb; ib += nw / 2) {
        for (int i = 0; i < 8; i++) {
            yl_raw[i]     = yb[i];
            yl_raw[i + 8] = yb[i + 16];
        }

        for (int row = 0; row < nr; row++) {
            sumf_gate[row] += block_iq4_nl_dot_y(xg + ib + row * nb, yl_raw, il);
            sumf_up  [row] += block_iq4_nl_dot_y(xu + ib + row * nb, yl_raw, il);
        }

        yb += QK4_0 * 16;
    }

    // Final simd-level reduction + fused silu_mul + write. No threadgroup
    // shmem needed — each simdgroup writes its own first_row's rows.
    for (int row = 0; row < N_DST; ++row) {
        const float gate_tot = simd_sum(sumf_gate[row]);
        const float up_tot   = simd_sum(sumf_up  [row]);
        if (tiisg == 0 && first_row + row < (int)p.ne01) {
            const float silu_g = gate_tot / (1.0f + metal::exp(-gate_tot));
            dst[im * p.ne0 * p.ne1 + r1 * p.ne0 + first_row + row] =
                silu_g * up_tot;
        }
    }
}
