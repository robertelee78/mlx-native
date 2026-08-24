// ADR-033 §Pi Task #20 + ADR-034 task #93 prefill extension —
// Fused MoE gate+up+silu_mul mm_id kernel for Q6_K.
//
// Closes the hf2q-vs-peer prefill gap at production Qwen MoE
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
//   shmem          = 16 KB peak (writeback dominates)
//     K-loop:  4 KB sa_gate + 4 KB sa_up + 4 KB sb  = 12 KB
//     Writeback: 8 KB temp_str_gate + 8 KB temp_str_up = 16 KB (overlays)
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

// Per-call dequantize: writes 16 elements of one Q6_K block to a 4x4 tile.
// Verbatim port of `dequantize_q6_K` from quantized_matmul_id_mm.metal:213
// (itself the shared upstream template). DO NOT modify — the
// `il`/scales index arithmetic + kmask bit-packing is mm-template-specific.
template <typename type4x4>
void dequantize_q6_K_fused(device const block_q6_K * xb, short il, thread type4x4 & reg) {
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

    const uint32_t route_state = tpe_u32[im];
    if ((route_state & 0x80000000u) != 0u) {
        if (im == 0 && tgpig.x == 0 && tgpig.y == 0) {
            device float * out = (device float *) dst;
            const uint64_t total =
                (uint64_t)args.ne21 * (uint64_t)args.ne1 * (uint64_t)args.ne0;
            for (uint64_t index = tiitg; index < total; index += 128) {
                out[index] = NAN;
            }
        }
        return;
    }
    const int32_t neh1 = (int32_t)route_state;

    if (r1 >= neh1) return;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im * args.ne21 + r1 + lr1];
    const short i11 = (id % args.ne20) % args.ne11;
    const int   i12 = (id / args.ne20);
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

    // ---- Fused silu_mul writeback ----
    //
    // Shmem layout (writeback overlays the K-loop A-tile/B-tile slots,
    // which are no longer needed):
    //   offset    0 ..  8191   temp_str_gate (8 KB — 4 SG × 16×32 floats)
    //   offset 8192 .. 16383   temp_str_up   (8 KB)
    //
    // Each simdgroup writes its 8 mc-tiles into its quadrant of the
    // 64×32 output tile, then a single barrier publishes both buffers.
    // The per-row copy loop reads gate + up, applies silu_mul, writes dst.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_gate = ((threadgroup float *) shmem)
        + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
    threadgroup float * temp_str_up   = ((threadgroup float *)(shmem + 8192))
        + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc_gate[i], temp_str_gate + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        simdgroup_store(mc_up[i],   temp_str_up   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-row copy with fused silu_mul: each simdgroup strides through nr1.
    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;  // slot index
        const short idt = id / args.ne20;  // token index

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * Cg  = (threadgroup float *)  shmem          + j*NR0;
        threadgroup float  * Cu  = (threadgroup float *) (shmem + 8192)  + j*NR0;
        threadgroup float4 * Cg4 = (threadgroup float4 *) Cg;
        threadgroup float4 * Cu4 = (threadgroup float4 *) Cu;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            float4 g4 = *(Cg4 + i);
            float4 u4 = *(Cu4 + i);
            float4 r4;
            // silu(g) = g / (1 + exp(-g));  out = silu(g) * u
            r4.x = (g4.x / (1.0f + exp(-g4.x))) * u4.x;
            r4.y = (g4.y / (1.0f + exp(-g4.y))) * u4.y;
            r4.z = (g4.z / (1.0f + exp(-g4.z))) * u4.z;
            r4.w = (g4.w / (1.0f + exp(-g4.w))) * u4.w;
            *(D4 + i) = r4;
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            const float g = *(Cg + i);
            const float u = *(Cu + i);
            *(D + i) = (g / (1.0f + exp(-g))) * u;
        }
    }
}
