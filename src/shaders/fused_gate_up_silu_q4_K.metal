// ADR-034 task #93 cont. 24 (2026-05-21) — Fused gate_proj + up_proj + silu_mul Q4_K.
//
// Q4_K variant of the Q8_0 fused kernel. Replaces a 3-dispatch sequence
// (gate matvec + up matvec + silu_mul) with a single Metal kernel dispatch.
// Both gate_w and up_w consume the SAME input vector x, so y is loaded
// once per ib iteration and reused for both projections, saving ~50%
// input memory bandwidth.
//
// Dispatch geometry (matches `kernel_mul_mv_q4_K_f32`):
//   - threadgroups   = (ceil(N / 2), M, 1)
//   - threads_per_tg = (32, 2, 1) = 64 threads = 2 simdgroups × 32 threads
//   - NO threadgroup shared memory (each simdgroup writes 1 distinct row).
//
// Buffer layout:
//   buffer(0): gate_w   device const block_q4_K *  [I, H_super_blocks] Q4_K
//   buffer(1): up_w     device const block_q4_K *  [I, H_super_blocks] Q4_K
//   buffer(2): x        device const float *       [H * M] F32
//   buffer(3): out      device       float *       [I * M] F32 (intermediate)
//   buffer(4): p        constant GgmlMatvecParams &
//
// Constraints:
//   - `p.ne00` (hidden_size) must be a multiple of QK_K = 256.
//   - Both weight matrices MUST share `p` (same ne00, ne01, layout).
//
// Math contract: byte-identical (within F32 FMA tolerance ≤ 1e-5) to:
//   `kernel_mul_mv_q4_K_f32(gate_w, x) → tmp_gate`
//   `kernel_mul_mv_q4_K_f32(up_w, x)   → tmp_up`
//   `silu_mul_f32(tmp_gate, tmp_up)     → out`

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    half     d;
    half     dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K/2];
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(half) + K_SCALE_SIZE + QK_K/2,
              "wrong q4_K block size");

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

kernel void kernel_fused_gate_up_silu_q4_K_f32(
    device const  void  * gate_w  [[buffer(0)]],
    device const  void  * up_w    [[buffer(1)]],
    device const float  * src1    [[buffer(2)]],
    device       float  * dst     [[buffer(3)]],
    constant GgmlMatvecParams & p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nb = (int)(p.ne00 / QK_K);

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = (int)tgpig.z;

    const int row = 2 * (int)r0 + (int)sgitg;

    const uint i12 = (uint)im % QMM_NE12(p);
    const uint i13 = (uint)im / QMM_NE12(p);

    const uint offset0 = (i12 / QMM_R2(p)) * (nb * p.ne01)
        + (i13 / QMM_R3(p)) * (nb * p.ne01 * p.ne02);

    device const block_q4_K * xg = (device const block_q4_K *) gate_w + row * nb + offset0;
    device const block_q4_K * xu = (device const block_q4_K *) up_w   + row * nb + offset0;
    device const float      * yy = (device const float      *) src1
        + r1 * p.ne10 + im * p.ne00 * p.ne1;

    float sumf_gate = 0.f;
    float sumf_up   = 0.f;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const short tid = (short)tiisg / 4;
    const short ix  = (short)tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0       = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    uint16_t sc16_g[4], sc16_u[4];
    thread const uint8_t * sc8_g = (thread const uint8_t *)sc16_g;
    thread const uint8_t * sc8_u = (thread const uint8_t *)sc16_u;

    device const float * y1 = yy + ix * QK_K + y_offset;

    // Stride-loop over super-blocks; ix==0..3 lanes interleave across ib.
    for (int i = ix; i < nb; i += 4) {
        device const uint8_t  * q1_g = xg[i].qs + q_offset;
        device const uint8_t  * q2_g = q1_g + 64;
        device const uint8_t  * q1_u = xu[i].qs + q_offset;
        device const uint8_t  * q2_u = q1_u + 64;
        device const half     * dh_g = &xg[i].d;
        device const half     * dh_u = &xu[i].d;
        device const uint16_t * ag   = (device const uint16_t *)xg[i].scales + iq;
        device const uint16_t * au   = (device const uint16_t *)xu[i].scales + iq;

        device const float * y2 = y1 + 128;
        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l +  0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l + 32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l +  0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l + 32]; sumy[3] += yh[l+8];
        }

        // Scale decode (same formula for both gate and up).
        sc16_g[0] = ag[0] & kmask1;
        sc16_g[1] = ag[2] & kmask1;
        sc16_g[2] = ((ag[4] >> 0) & kmask2) | ((ag[0] & kmask3) >> 2);
        sc16_g[3] = ((ag[4] >> 4) & kmask2) | ((ag[2] & kmask3) >> 2);

        sc16_u[0] = au[0] & kmask1;
        sc16_u[1] = au[2] & kmask1;
        sc16_u[2] = ((au[4] >> 0) & kmask2) | ((au[0] & kmask3) >> 2);
        sc16_u[3] = ((au[4] >> 4) & kmask2) | ((au[2] & kmask3) >> 2);

        // Per-projection dot accumulators (shared yl/yh).
        float4 acc1_g = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_u = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            acc1_g[0] += yl[l+0] * (float)(q1_g[l] & 0x0F);
            acc1_g[1] += yl[l+8] * (float)(q1_g[l] & 0xF0);
            acc1_g[2] += yh[l+0] * (float)(q2_g[l] & 0x0F);
            acc1_g[3] += yh[l+8] * (float)(q2_g[l] & 0xF0);

            acc1_u[0] += yl[l+0] * (float)(q1_u[l] & 0x0F);
            acc1_u[1] += yl[l+8] * (float)(q1_u[l] & 0xF0);
            acc1_u[2] += yh[l+0] * (float)(q2_u[l] & 0x0F);
            acc1_u[3] += yh[l+8] * (float)(q2_u[l] & 0xF0);
        }

        const float dall_g = (float)dh_g[0];
        const float dmin_g = (float)dh_g[1];
        sumf_gate += dall_g * ((float)sc8_g[0] * (acc1_g[0]        ) +
                               (float)sc8_g[1] * (acc1_g[1] / 16.f ) +
                               (float)sc8_g[4] * (acc1_g[2]        ) +
                               (float)sc8_g[5] * (acc1_g[3] / 16.f )) -
                     dmin_g * (sumy[0] * (float)sc8_g[2] + sumy[1] * (float)sc8_g[3] +
                               sumy[2] * (float)sc8_g[6] + sumy[3] * (float)sc8_g[7]);

        const float dall_u = (float)dh_u[0];
        const float dmin_u = (float)dh_u[1];
        sumf_up   += dall_u * ((float)sc8_u[0] * (acc1_u[0]        ) +
                               (float)sc8_u[1] * (acc1_u[1] / 16.f ) +
                               (float)sc8_u[4] * (acc1_u[2]        ) +
                               (float)sc8_u[5] * (acc1_u[3] / 16.f )) -
                     dmin_u * (sumy[0] * (float)sc8_u[2] + sumy[1] * (float)sc8_u[3] +
                               sumy[2] * (float)sc8_u[6] + sumy[3] * (float)sc8_u[7]);

        y1 += 4 * QK_K;
    }

    // Final simd-level reduction + fused silu_mul + write. No threadgroup shmem
    // needed because each simdgroup writes a distinct row.
    const float gate_tot = simd_sum(sumf_gate);
    const float up_tot   = simd_sum(sumf_up);
    if (tiisg == 0 && row < (int)p.ne01) {
        const float silu_g = gate_tot / (1.0f + metal::exp(-gate_tot));
        dst[r1 * p.ne0 + im * p.ne0 * p.ne1 + row] = silu_g * up_tot;
    }
}
