// ADR-034 task #93 Step 4 (2026-05-21) — Fused dual projection Q4_0.
//
// Computes TWO Q4_0 mat-vec projections sharing the same input vector x:
//   dst_a[row] = dot(dequant(weight_a[row]), x)
//   dst_b[row] = dot(dequant(weight_b[row]), x)
//
// Both weight matrices MUST have identical shape (ne00, ne01, ne02) — i.e.
// same input dim AND same output dim. Designed to fuse:
//   - Q + gate projections (both size [hidden, q_total])
//   - K + V projections (both size [hidden, kv_total])
// in the Qwen FA layer Op 2 sequence.
//
// Replaces 2 separate dispatches of `kernel_mul_mv_q4_0_f32` with 1.
// Input vector x is loaded ONCE per ib iteration and reused for both
// projections, saving ~50% input memory bandwidth pressure.
//
// Dispatch geometry: identical to `kernel_mul_mv_q4_0_f32`:
//   - threadgroups   = (ceil(N / 8), M, 1)
//   - threads_per_tg = (8, 8, 1) = 64 threads = 2 simdgroups × 32 threads
//   - NO threadgroup shared memory (each simdgroup writes distinct rows).
//
// Math contract: byte-identical to two `kernel_mul_mv_q4_0_f32` dispatches
// (same accumulator order, same `block_q4_0_dot_y` helper logic).

#include <metal_stdlib>
using namespace metal;

#define QK4_0 32
#define N_DST_Q4 4
#define N_SIMDGROUP_Q4 2
#define N_SIMDWIDTH 32

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(half) + QK4_0 / 2,
              "wrong q4_0 block size");

// Matches `GgmlMatvecParams` in `quantized_matmul_ggml.metal` exactly so
// the Rust-side `GgmlMatvecGpuParams` struct can be reused without change.
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

// Function constants — same slots as the Q8_0 kernel for compatibility
// with the shared dispatch infrastructure.
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

// Q4_0 dot helper — ported from `quantized_matmul_ggml.metal:224-240`.
// 4-accumulator layout matches llama.cpp's block_q_n_dot_y for byte-identical
// FP rounding.
inline float block_q4_0_dot_y(
    device const block_q4_0 * qb,
    float sumy,
    thread float * yl,
    int il
) {
    float d = qb->d;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device const uint16_t * qs = ((device const uint16_t *)qb + 1 + il/2);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// Fused dual Q4_0 mat-vec.
//
// Buffer layout:
//   buffer(0): weight_a  device const block_q4_0 * [N, K_blocks]
//   buffer(1): weight_b  device const block_q4_0 * [N, K_blocks]  (same shape)
//   buffer(2): input     device const float *      [K * M] F32
//   buffer(3): dst_a     device       float *      [N * M] F32
//   buffer(4): dst_b     device       float *      [N * M] F32
//   buffer(5): params    constant GgmlMatvecParams &
kernel void kernel_fused_dual_proj_q4_0_f32(
    device const  void  * weight_a  [[buffer(0)]],
    device const  void  * weight_b  [[buffer(1)]],
    device const float  * src1      [[buffer(2)]],
    device       float  * dst_a     [[buffer(3)]],
    device       float  * dst_b     [[buffer(4)]],
    constant GgmlMatvecParams & p   [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int nr  = N_DST_Q4;        // 4 rows per simdgroup
    const int nsg = N_SIMDGROUP_Q4;  // 2 simdgroups per threadgroup
    const int nw  = N_SIMDWIDTH;     // 32 threads per simdgroup

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

    device const block_q4_0 * xa = (device const block_q4_0 *) weight_a + offset0;
    device const block_q4_0 * xb = (device const block_q4_0 *) weight_b + offset0;
    device const float      * y  = (device const float      *) src1
        + r1 * p.ne10
        + im * p.ne00 * p.ne1;

    float yl[16];
    float sumf_a[N_DST_Q4] = { 0.f, 0.f, 0.f, 0.f };
    float sumf_b[N_DST_Q4] = { 0.f, 0.f, 0.f, 0.f };

    const int ix = (int)tiisg / 2;
    const int il = (int)((tiisg % 2) * 8);

    device const float * yb = y + ix * QK4_0 + il;

    // Stride-loop: each thread covers QK4_0 = 32 quants per iteration; SGs
    // interleave across ib by stride nw/2 = 16. Per-ib input is loaded ONCE
    // into yl and reused for both weight_a and weight_b — the bandwidth win.
    for (int ib = ix; ib < nb; ib += nw / 2) {
        float sumy[2] = { 0.f, 0.f };
        for (int i = 0; i < 8; i += 2) {
            sumy[0] += yb[i] + yb[i+1];
            yl[i+0] = yb[i+0];
            yl[i+1] = yb[i+1] / 256.f;
            sumy[1] += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16] / 16.f;
            yl[i+9] = yb[i+17] / 4096.f;
        }

        const float sumy_total = sumy[0] + sumy[1];
        for (int row = 0; row < nr; row++) {
            sumf_a[row] += block_q4_0_dot_y(xa + ib + row * nb, sumy_total, yl, il);
            sumf_b[row] += block_q4_0_dot_y(xb + ib + row * nb, sumy_total, yl, il);
        }

        yb += QK4_0 * 16;
    }

    // Final simd-level reduction + write. No threadgroup-shared memory needed
    // because each simdgroup writes distinct rows (first_row differs per SG).
    for (int row = 0; row < nr; ++row) {
        const float tot_a = simd_sum(sumf_a[row]);
        const float tot_b = simd_sum(sumf_b[row]);
        if (tiisg == 0 && first_row + row < p.ne01) {
            const uint dst_idx = im * p.ne0 * p.ne1 + r1 * p.ne0 + first_row + row;
            dst_a[dst_idx] = tot_a;
            dst_b[dst_idx] = tot_b;
        }
    }
}
