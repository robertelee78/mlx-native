// ADR-034 task #93 (2026-05-21) — Fused gate_proj + up_proj + silu_mul Q8_0.
//
// Computes:  out[i] = silu(gate_w[i] @ x) * (up_w[i] @ x)
//   where silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
//
// Replaces a 3-dispatch sequence (gate matvec + up matvec + silu_mul) with
// a single Metal kernel dispatch. Both gate_w and up_w consume the SAME
// input vector x, so we load x once per ib iteration and reuse for both
// projections, saving ~50% of memory-bandwidth pressure on x compared to
// the unfused sequence.
//
// Dispatch geometry (matches `kernel_mul_mv_q8_0_f32_nr2`):
//   - threadgroups = (ceil(intermediate_size / (NSG * NR0)), 1, 1)
//   - threads_per_tg = (32, NSG, 1) = (32, 4, 1) = 128 threads
//   - NSG=4, NR0=2, NW=32, NQ=8 (same constants as peer-style Q8_0 mv).
//
// Buffer layout:
//   - buffer(0): gate_w  device const block_q8_0 *  [I, H_blocks] Q8_0
//   - buffer(1): up_w    device const block_q8_0 *  [I, H_blocks] Q8_0
//   - buffer(2): x       device const float *       [H] F32
//   - buffer(3): out     device       float *       [I] F32 (intermediate)
//   - buffer(4): p       constant GgmlMatvecParams &
//   - threadgroup(0): shmem  threadgroup float *    [NW * NR0 * 2] (gate + up reductions)
//
// Constraints:
//   - `p.ne00` (hidden_size) must be a multiple of QK8_0 = 32.
//   - `p.ne01` (intermediate_size) is the gate_w / up_w row count. SAME for both.
//   - Both weight matrices MUST share `p` (same ne00, ne01, layout).
//   - `p.ne1` = number of input rows; this kernel is for `ne1 == 1` (decode);
//     the multi-row case is via the dispatch wrapper running multiple TGs.
//
// Math contract: byte-identical to a 3-dispatch sequence of
//   dispatch_quantized_matmul_ggml(Q8_0, gate_w, x)  → tmp_gate
//   dispatch_quantized_matmul_ggml(Q8_0, up_w, x)    → tmp_up
//   dispatch_silu_mul(tmp_gate, tmp_up)              → out
// All accumulators are F32 with identical accumulator order to the
// peer-style `kernel_mul_mv_q8_0_f32_nr2` for gate and up independently;
// final silu_mul applies the IEEE-754 single-precision functions
// `g / (1 + exp(-g))` and `*`.

#include <metal_stdlib>
using namespace metal;

// These constants MUST match `quantized_matmul_ggml.metal`. Kept duplicated
// here (not via #include) because mlx-native build system compiles each
// .metal file independently into the metallib.
#define QK8_0 32

#define N_R0_Q8_0    2
#define N_SG_Q8_0    4
#define NQ_Q8_0      8
#define N_SIMDWIDTH 32

typedef struct {
    half   d;
    int8_t qs[QK8_0];
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(half) + QK8_0,
              "wrong q8_0 block size");

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

// FC slots match `quantized_matmul_ggml.metal`. Re-declared here so the
// function-constant specialization works for this PSO too.
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

kernel void kernel_fused_gate_up_silu_q8_0_f32(
    device const  void  * gate_w  [[buffer(0)]],
    device const  void  * up_w    [[buffer(1)]],
    device const float  * src1    [[buffer(2)]],
    device       float  * dst     [[buffer(3)]],
    constant GgmlMatvecParams & p [[buffer(4)]],
    threadgroup float   * shmem   [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr int NR0 = N_R0_Q8_0;   // 2 output rows per TG
    constexpr int NSG = N_SG_Q8_0;   // 4 simdgroups per TG
    constexpr int NW  = N_SIMDWIDTH; // 32 threads per simdgroup
    constexpr int NQ  = NQ_Q8_0;     // 8 quants per thread per ib

    const int nb = (int)(p.ne00 / QK8_0);  // blocks per weight row
    const int r0 = (int)tgpig.x * NR0;
    const int r1 = (int)tgpig.y;
    const int im = (int)tgpig.z;

    const uint i12 = (uint)im % QMM_NE12(p);
    const uint i13 = (uint)im / QMM_NE12(p);

    // Per-row weight pointers (gate AND up share the same p.ne01 / layout).
    device const block_q8_0 * ax_gate[NR0];
    device const block_q8_0 * ax_up[NR0];
    for (int row = 0; row < NR0; ++row) {
        const uint offset0 = (r0 + row) * nb
            + (i12 / QMM_R2(p)) * (nb * p.ne01)
            + (i13 / QMM_R3(p)) * (nb * p.ne01 * p.ne02);
        ax_gate[row] = (device const block_q8_0 *) gate_w + offset0;
        ax_up  [row] = (device const block_q8_0 *) up_w   + offset0;
    }

    device const float * y = (device const float *) src1
        + r1 * p.ne10
        + im * p.ne00 * p.ne1;

    // Per-row partial sums for gate and up projections.
    float sumf_gate[NR0] = { 0.f };
    float sumf_up  [NR0] = { 0.f };

    const int ix = (int)tiisg / (NW / NQ);  // 0..3
    const int il = (int)tiisg % (NW / NQ);  // 0..3

    const int ib0 = sgitg * NQ + ix;

    float yl[NQ];
    device const float * yb = y + ib0 * QK8_0 + il * NQ;

    // Stride-loop over weight blocks; each iter loads NQ y-values ONCE
    // and reuses for both projections, saving ~50% input bandwidth.
    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        for (int i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (int row = 0; row < NR0; ++row) {
            // gate projection accumulator
            {
                device const int8_t * qs = ax_gate[row][ib].qs + il * NQ;
                float sumq = 0.f;
                for (int iq = 0; iq < NQ; ++iq) {
                    sumq += (float)qs[iq] * yl[iq];
                }
                sumf_gate[row] += sumq * (float)ax_gate[row][ib].d;
            }
            // up projection accumulator (same yl, different weights)
            {
                device const int8_t * qs = ax_up[row][ib].qs + il * NQ;
                float sumq = 0.f;
                for (int iq = 0; iq < NQ; ++iq) {
                    sumq += (float)qs[iq] * yl[iq];
                }
                sumf_up[row] += sumq * (float)ax_up[row][ib].d;
            }
        }

        yb += NSG * NQ * QK8_0;
    }

    // Cross-simdgroup reduction: gate and up each get NW slots in shmem.
    // Layout: shmem[0 .. NW*NR0 .. NW*NR0*2] = gate rows 0..NR0-1, up rows 0..NR0-1.
    threadgroup float * shmem_gate[NR0];
    threadgroup float * shmem_up  [NR0];
    for (int row = 0; row < NR0; ++row) {
        shmem_gate[row] = shmem + NW * row;
        shmem_up  [row] = shmem + NW * NR0 + NW * row;
        if (sgitg == 0) {
            shmem_gate[row][tiisg] = 0.0f;
            shmem_up  [row][tiisg] = 0.0f;
        }
        sumf_gate[row] = simd_sum(sumf_gate[row]);
        sumf_up  [row] = simd_sum(sumf_up  [row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_gate[row][sgitg] = sumf_gate[row];
            shmem_up  [row][sgitg] = sumf_up  [row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduce + fused silu_mul + write.
    for (int row = 0; row < NR0 && r0 + row < p.ne01; ++row) {
        const float gate_tot = simd_sum(shmem_gate[row][tiisg]);
        const float up_tot   = simd_sum(shmem_up  [row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            // silu(g) = g / (1 + exp(-g)) (IEEE-754 single precision)
            const float silu_g = gate_tot / (1.0f + metal::exp(-gate_tot));
            dst[r1 * p.ne0 + im * p.ne0 * p.ne1 + r0 + row] = silu_g * up_tot;
        }
    }
}
