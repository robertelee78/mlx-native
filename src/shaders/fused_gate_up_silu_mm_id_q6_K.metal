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

// SCAFFOLDING — full kernel body deferred to follow-up iter.
//
// Plan:
//   1. Replicate the dequant + shmem staging from `hf2q_mul_mm_id_impl`
//      but for TWO weight buffers (gate_w + up_w), staging into two
//      separate A-tile regions in shmem.
//   2. Run MMA for both gate and up, accumulating into mc_gate[8] and
//      mc_up[8] simdgroup matrices.
//   3. In the writeback prelude, apply `silu(gate) * up` element-wise
//      via `tmp_gate * (1.0f / (1.0f + exp(-tmp_gate)))` * tmp_up.
//   4. Write the single fused result to dst — half the output bandwidth
//      vs writing tmp_gate then tmp_up separately.
//
// Shmem budget (4 SG × 64x32 output tile):
//   sa_gate: 4 KB  (32 K-elem × 64 N-row × half)
//   sa_up:   4 KB
//   sb:      4 KB  (32 K-elem × 32 M-row × float)
//   Total: 12 KB — well under 32 KB threadgroup limit.
//
// Parity test plan: bench-driven; will land alongside the body.
//
// STATUS: signature + dequant locked; body in next iter.
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
    // Scaffolding — body in follow-up iter. Suppress unused warnings.
    (void)args; (void)gate_w; (void)up_w; (void)src1;
    (void)htpe; (void)hids; (void)dst; (void)shmem;
    (void)tgpig; (void)tiitg; (void)tiisg; (void)sgitg;
}
