// ADR-034 task #93 cont. 28 (2026-05-21) — Fused gate+up+silu_mul Q6_K.
//
// Final super-block fused MLP family member (after Q4_K, Q5_K). Replaces
// 3 dispatches with 1: gate matvec + up matvec + silu_mul. Both weights
// share the same input vector y, so we load y values once per ib
// iteration and reuse for both projections.
//
// Dispatch geometry (matches `kernel_mul_mv_q6_K_f32`):
//   - threadgroups   = (ceil(N / 2), M, 1)
//   - threads_per_tg = (2, 32, 1) = 64 threads = 2 simdgroups × 32 threads
//   - NO threadgroup shared memory.
//
// Buffer layout:
//   buffer(0): gate_w   device const block_q6_K *  [I, H_super_blocks] Q6_K
//   buffer(1): up_w     device const block_q6_K *  [I, H_super_blocks] Q6_K
//   buffer(2): x        device const float *       [H * M] F32
//   buffer(3): out      device       float *       [I * M] F32
//   buffer(4): p        constant GgmlMatvecParams &
//
// Math contract: byte-identical (within F32 FMA tolerance ≤ 1e-4) to:
//   `kernel_mul_mv_q6_K_f32(gate_w, x) → tmp_gate`
//   `kernel_mul_mv_q6_K_f32(up_w, x)   → tmp_up`
//   `silu_mul_f32(tmp_gate, tmp_up)     → out`

#include <metal_stdlib>
using namespace metal;

#define QK_K 256

typedef struct {
    uint8_t ql[QK_K/2];      // 128 bytes — lower 4 bits of 6-bit values
    uint8_t qh[QK_K/4];      // 64 bytes — upper 2 bits packed
    int8_t  scales[QK_K/16]; // 16 bytes — signed 8-bit per-sub-block scales
    half    d;                // 2 bytes — super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(half) + QK_K/16 + 3*QK_K/4,
              "wrong q6_K block size");

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

kernel void kernel_fused_gate_up_silu_q6_K_f32(
    device const  void  * gate_w  [[buffer(0)]],
    device const  void  * up_w    [[buffer(1)]],
    device const float  * src1    [[buffer(2)]],
    device       float  * dst     [[buffer(3)]],
    constant GgmlMatvecParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = (int)(p.ne00 / QK_K);

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = (int)tgpig.z;

    const int row = 2 * (int)r0 + (int)sgitg;

    const uint i12 = (uint)im % QMM_NE12(p);
    const uint i13 = (uint)im / QMM_NE12(p);

    const uint offset0 = (i12 / QMM_R2(p)) * (nb * p.ne01)
        + (i13 / QMM_R3(p)) * (nb * p.ne01 * p.ne02);

    device const block_q6_K * xg = (device const block_q6_K *) gate_w + row * nb + offset0;
    device const block_q6_K * xu = (device const block_q6_K *) up_w   + row * nb + offset0;
    device const float      * yy = (device const float      *) src1
        + r1 * p.ne10 + im * p.ne00 * p.ne1;

    float sumf_gate = 0.f;
    float sumf_up   = 0.f;

    const int tid  = (int)tiisg / 2;
    const int ix   = (int)tiisg % 2;
    const int ip   = tid / 8;
    const int il   = tid % 8;
    const int n    = 4;
    const int l0   = n * il;
    const int is   = 8 * ip + l0 / 16;

    const int y_offset   = 128 * ip + l0;
    const int q_offset_l = 64 * ip + l0;
    const int q_offset_h = 32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1_g = xg[i].ql + q_offset_l;
        device const uint8_t * q2_g = q1_g + 32;
        device const uint8_t * qh_g = xg[i].qh + q_offset_h;
        device const int8_t  * sc_g = xg[i].scales + is;

        device const uint8_t * q1_u = xu[i].ql + q_offset_l;
        device const uint8_t * q2_u = q1_u + 32;
        device const uint8_t * qh_u = xu[i].qh + q_offset_h;
        device const int8_t  * sc_u = xu[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall_g = xg[i].d;
        const float dall_u = xu[i].d;

        // Per-projection partial sums (shared y reads).
        float4 sums_g = {0.f, 0.f, 0.f, 0.f};
        float4 sums_u = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            const float y0  = y[l +  0];
            const float y32 = y[l + 32];
            const float y64 = y[l + 64];
            const float y96 = y[l + 96];

            sums_g[0] += y0  * (float)((int8_t)((q1_g[l] & 0xF) | ((qh_g[l] & kmask1) << 4)) - 32);
            sums_g[1] += y32 * (float)((int8_t)((q2_g[l] & 0xF) | ((qh_g[l] & kmask2) << 2)) - 32);
            sums_g[2] += y64 * (float)((int8_t)((q1_g[l]  >> 4) | ((qh_g[l] & kmask3) << 0)) - 32);
            sums_g[3] += y96 * (float)((int8_t)((q2_g[l]  >> 4) | ((qh_g[l] & kmask4) >> 2)) - 32);

            sums_u[0] += y0  * (float)((int8_t)((q1_u[l] & 0xF) | ((qh_u[l] & kmask1) << 4)) - 32);
            sums_u[1] += y32 * (float)((int8_t)((q2_u[l] & 0xF) | ((qh_u[l] & kmask2) << 2)) - 32);
            sums_u[2] += y64 * (float)((int8_t)((q1_u[l]  >> 4) | ((qh_u[l] & kmask3) << 0)) - 32);
            sums_u[3] += y96 * (float)((int8_t)((q2_u[l]  >> 4) | ((qh_u[l] & kmask4) >> 2)) - 32);
        }

        sumf_gate += dall_g * (sums_g[0] * sc_g[0] + sums_g[1] * sc_g[2]
                             + sums_g[2] * sc_g[4] + sums_g[3] * sc_g[6]);
        sumf_up   += dall_u * (sums_u[0] * sc_u[0] + sums_u[1] * sc_u[2]
                             + sums_u[2] * sc_u[4] + sums_u[3] * sc_u[6]);
    }

    const float gate_tot = simd_sum(sumf_gate);
    const float up_tot   = simd_sum(sumf_up);
    if (tiisg == 0) {
        const float silu_g = gate_tot / (1.0f + metal::exp(-gate_tot));
        dst[r1 * p.ne0 + im * p.ne0 * p.ne1 + row] = silu_g * up_tot;
    }
}
