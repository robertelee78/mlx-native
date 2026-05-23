// ADR-033 §Pi Task #20 + ADR-034 task #93 prefill extension —
// Fused MoE gate+up+silu_mul mm_id kernel for Q6_K.
//
// Closes the hf2q-vs-llama.cpp prefill gap at production Qwen MoE
// shapes (~7-15% slower at pp1024-pp4096 pre-fusion). Replaces a
// 3-dispatch sequence (gate_mm_id + up_mm_id + silu_mul_id) with a
// single fused dispatch per layer:
//
//   ffn_gate_exps × x → tmp_gate
//   ffn_up_exps   × x → tmp_up
//   silu_mul(tmp_gate, tmp_up) → out
//
//   ↓ becomes ↓
//
//   fused_gate_up_silu_mm_id_q6_K(gate_w, up_w, x, hids, htpe, dst)
//
// Both weights read the same routed input row(s) per (token, slot),
// so we can amortize input-row staging across BOTH matmuls. Output
// is single [m, intermediate] not two — halves the output writeback
// bandwidth.
//
// Dispatch geometry (matches `hf2q_mul_mm_id_impl<block_q6_K, …>`):
//   threadgroups   = (ceil(n_tokens*top_k / NR1), ceil(N / NR0), n_experts)
//   threads_per_tg = 128 (4 simdgroups × 32 threads)
//   shmem          = 16 KB (2× the A-tile + 1× B-tile)
//
// Buffer layout:
//   buffer(0): args         constant GgmlMatmulIdMm_MmParams &
//   buffer(1): gate_w_src0  device const char * stacked expert weights for gate_proj
//   buffer(2): up_w_src0    device const char * stacked expert weights for up_proj
//   buffer(3): src1         device const char * input rows [n_tokens, K]
//   buffer(4): htpe         device const char * per-expert routed counts [n_experts]
//   buffer(5): hids         device const char * per-expert routed-token list
//   buffer(6): dst          device       char * output [n_tokens*top_k, N]
//
// Math contract: byte-identical (within F32 FMA tolerance) to the
// unfused 3-dispatch sequence. Parity tested via
// `adr_033_pi_task20_fused_mm_id_q6_K_parity` (TBD).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
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

// Per-call dequantize: writes 16 elements of one Q6_K block to a half4x4
// tile. Mirrors the dequantize_q6_K used in `quantized_matmul_id_mm.metal`.
template <typename type4x4>
void dequantize_q6_K_fused(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const float d_all = xb->d;
    device const uint8_t * ql = xb->ql + 64*(il/8) + 32*((il/2)%2);
    device const uint8_t * qh = xb->qh + 32*(il/8);
    const int kk = il % 2;
    const float dl = d_all * xb->scales[il*2];
    const float dl2 = d_all * xb->scales[il*2 + 1];
    for (int n = 0; n < 4; n++) {
        const float v0 =
            (float)((int8_t)((ql[16*kk + n + 0] & 0xF) | (((qh[n + 0] >> (4*kk + 0)) & 0x3) << 4)) - 32) * dl;
        const float v1 =
            (float)((int8_t)((ql[16*kk + n + 4] & 0xF) | (((qh[n + 4] >> (4*kk + 0)) & 0x3) << 4)) - 32) * dl;
        const float v2 =
            (float)((int8_t)((ql[16*kk + n + 8] & 0xF) | (((qh[n + 8] >> (4*kk + 0)) & 0x3) << 4)) - 32) * dl2;
        const float v3 =
            (float)((int8_t)((ql[16*kk + n + 12] & 0xF) | (((qh[n + 12] >> (4*kk + 0)) & 0x3) << 4)) - 32) * dl2;
        reg[n][0] = v0;
        reg[n][1] = v1;
        reg[n][2] = v2;
        reg[n][3] = v3;
    }
}

// Args struct — same shape as GgmlMatmulIdMm_MmParams in the
// unfused kernel (lock-step with src/ops/quantized_matmul_id_ggml.rs).
struct GgmlMatmulIdMm_MmParams {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  _pad0;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    uint     r2;
    uint     r3;
    uint     _pad1;
};

// Iter 2: dual-A-tile staging + shared B-tile + dual MMA accumulation.
//
// Shmem layout (12 KB used in K-loop, 4 KB used in writeback):
//   offset    0 .. 4095   sa_gate (32 K-elem × 64 N-row × half = 4 KB)
//   offset 4096 .. 8191   sa_up   (same shape)
//   offset 8192 .. 12287  sb      (32 K-elem × 32 M-row × float = 4 KB)
//
//   Writeback overlays sa_gate region (offset 0); the A-tiles are no
//   longer needed once we exit the K-loop.
//
// silu_mul fusion lands in iter 3 — for now writeback path is empty.
#define Q6K_NL 16  // 256/16 = 16 dequant ops per block

kernel void kernel_fused_gate_up_silu_mm_id_q6_K_f32(
        constant GgmlMatmulIdMm_MmParams & args [[buffer(0)]],
        device const char * gate_w [[buffer(1)]],
        device const char * up_w   [[buffer(2)]],
        device const char * src1   [[buffer(3)]],
        device const char * htpe   [[buffer(4)]],
        device const char * hids   [[buffer(5)]],
        device       char * dst    [[buffer(6)]],
        threadgroup  char * shmem  [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa_gate = (threadgroup half  *)(shmem);
    threadgroup half  * sa_up   = (threadgroup half  *)(shmem + 4096);
    threadgroup float * sb      = (threadgroup float *)(shmem + 8192);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;  // 2
    constexpr int NL1 = NK/8;   // 4

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) return;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im * args.ne21 + r1 + lr1];
    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    // Same expert's slab in BOTH weight buffers — gate_w and up_w have
    // identical layout (same shape, same expert stride).
    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/Q6K_NL;

    device const block_q6_K * x_gate =
        (device const block_q6_K *)(gate_w + args.nb01*(r0 + lr0) + offset0) + offset1;
    device const block_q6_K * x_up =
        (device const block_q6_K *)(up_w   + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8  ma_gate[4];
    simdgroup_half8x8  ma_up[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];

    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i]   = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- A-tile dequantize + stage for GATE ----
        {
            half4x4 temp_a;
            dequantize_q6_K_fused(x_gate, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa_gate + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // ---- A-tile dequantize + stage for UP ----
        {
            half4x4 temp_a;
            dequantize_q6_K_fused(x_up, il, temp_a);

            // No barrier between the two A-tile stages — they write to
            // disjoint shmem regions and the same thread owns the same
            // (sx, sy, lx, ly) cell in each. The post-stage barrier
            // before MMA covers the cross-thread visibility.

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa_up + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // ---- B-tile stage (shared input — same as unfused kernel) ----
        for (short i = 0; i < 8; ++i) {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short lx = i;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(sb + 64*ib + 8*ly + lx) =
                (loop_k + iy + i < args.ne00) ? *((device float *) y + i) : 0.f;
        }

        // Advance K-cursor for both gate and up in lock-step. They share
        // the same dequant block layout so `il`/`x` arithmetic is identical.
        const short il_next = (il + 2 < Q6K_NL) ? il + 2 : il % 2;
        if (il_next < 2) {
            x_gate += (2 + Q6K_NL - 1)/Q6K_NL;
            x_up   += (2 + Q6K_NL - 1)/Q6K_NL;
        }
        il = il_next;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Dual MMA accumulate ----
        threadgroup const half  * lsma_gate = (sa_gate + 4*64*(sgitg%2));
        threadgroup const half  * lsma_up   = (sa_up   + 4*64*(sgitg%2));
        threadgroup const float * lsmb      = (sb      + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma_gate[i], lsma_gate + 64*i, 8, 0, false);
                simdgroup_load(ma_up[i],   lsma_up   + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma_gate[i%4], mc_gate[i]);
                simdgroup_multiply_accumulate(mc_up[i],   mb[i/4], ma_up[i%4],   mc_up[i]);
            }
            lsma_gate += 8*64;
            lsma_up   += 8*64;
            lsmb      += 4*64;
        }
    }

    // Iter 3 will: apply silu_mul = (mc_gate * sigmoid(mc_gate)) * mc_up,
    // simdgroup_store to shmem, and write back to dst with hids lookup.
    // For now the writeback is empty so we can prove compile + grid wiring.
    (void)dst;
    (void)nr0;
}
