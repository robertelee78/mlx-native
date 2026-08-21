#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// moe_mm_id_q4_0 — Two-pass MoE main mm_id kernel for Q4_0 weights.
//
// ADR-033 §Pi iter B (SKELETON ONLY — body pending iter B-2). Consumes
// the (tpe, hids) output of iter A's `moe_mm_id_map0` to drive a per-
// expert NR0=64 / NR1=32 / NK=32 simdgroup-matmul tile.
//
// This file is iter B SCAFFOLDING. The args struct, name, and registration
// are committed so iter B-2 can fill the body without needing to also
// design the dispatch contract. The current entry point is a structural
// stub that early-returns when r1 >= neh1 (upstream boundary check)
// and writes the launched output to zero otherwise. Production wiring
// (iter C) MUST NOT engage this kernel until iter B-2 lands the actual
// simdgroup matmul body.
//
// Pending body work (iter B-2):
//   - 16-element block_q4_0 dequantization into shmem (sa) — 16 weights
//     per block × NK=32 K-elements per loop iteration.
//   - simdgroup_half8x8 ma[4] / mb[2] / mc[8] accumulator chain
//     (specialized here for half input + f32 accumulator instead of
//     templated element types).
//   - Inner loop k = 0 .. args.ne00 by NK, dequantize + simdgroup_load
//     + simdgroup_multiply_accumulate chain.
//   - Tail handling at boundary (output writeback to dst with neh1 mask).
// --------------------------------------------------------------------------

struct MoeMmIdQ4_0Params {
    // Mirrors the reference mm_id kargs layout.
    // Field order must NOT drift — the iter C Rust dispatch wrapper will
    // memcpy values into this layout directly.
    int32_t  ne00;  // K dim (input feature width)
    int32_t  ne02;  // n_experts (a-side stride dim)
    uint64_t nb01;  // src0 row stride bytes (per-expert weight row)
    uint64_t nb02;  // src0 expert stride bytes
    uint64_t nb03;  // src0 batch stride bytes (unused for MoE; carried)
    int32_t  ne11;  // n_expert_used (broadcast)
    uint64_t nb10;  // src1 elem stride bytes
    uint64_t nb11;  // src1 row stride bytes (per-token)
    uint64_t nb12;  // src1 batch-1 stride bytes (per-slot)
    uint64_t nb13;  // src1 batch-0 stride bytes (unused for MoE; carried)
    int32_t  ne20;  // n_expert_used (echo)
    int32_t  ne21;  // n_tokens
    int32_t  ne0;   // output row count (N dim)
    int32_t  ne1;   // output col count (M dim — packed token×slot)
    int16_t  r2;    // reserved
    int16_t  r3;    // reserved
};

// Output tile geometry — NR0/NR1/NK are load-bearing constants shared
// with the host dispatcher. Drift here changes the dispatch grid
// shape and silently produces wrong outputs.
constant constexpr int NR0 = 64;
constant constexpr int NR1 = 32;

// --------------------------------------------------------------------------
// SKELETON ENTRY POINT (iter B-1)
//
// Boundary checks + tile-index resolution per the reference scheme.
// Body deferred to iter B-2.
// --------------------------------------------------------------------------

kernel void moe_mm_id_q4_0_f32_skeleton(
    constant MoeMmIdQ4_0Params & args   [[buffer(0)]],
    device  const char         * src0   [[buffer(1)]], // packed Q4_0 weights
    device  const char         * src1   [[buffer(2)]], // f32 token activations
    device  const char         * htpe   [[buffer(3)]], // from iter A
    device  const char         * hids   [[buffer(4)]], // from iter A
    device        char         * dst    [[buffer(5)]], // f32 output
    threadgroup   char         * shmem  [[threadgroup(0)]],
    uint3  tgpig                       [[threadgroup_position_in_grid]],
    ushort tiitg                       [[thread_index_in_threadgroup]],
    ushort tiisg                       [[thread_index_in_simdgroup]],
    ushort sgitg                       [[simdgroup_index_in_threadgroup]])
{
    const int im = tgpig.z;       // expert index
    const int r0 = tgpig.y * NR0; // N-tile row
    const int r1 = tgpig.x * NR1; // M-tile col

    device const uint32_t * tpe_u32 = (device const uint32_t *) htpe;
    const int32_t neh1 = (int32_t) tpe_u32[im];

    // Early-exit when this expert has fewer than r1 tokens routed to it.
    if (r1 >= neh1) {
        return;
    }

    // STUB: write zeros so a downstream comparison against a CPU reference
    // produces a clean "skeleton produces zero output" diagnostic. Iter
    // B-2 replaces this with the real simdgroup matmul.
    if (r0 + (int) tiitg < args.ne0) {
        for (int r = 0; r + r1 < neh1 && r < NR1; ++r) {
            const int dst_row = r0 + tiitg;
            const int dst_col = r1 + r;
            const int dst_idx = im * args.ne0 * args.ne1 + dst_col * args.ne0 + dst_row;
            ((device float *) dst)[dst_idx] = 0.0f;
        }
    }

    // Suppress unused-parameter warnings until iter B-2 wires the full
    // ma/mb/mc simdgroup_matrix accumulator chain.
    (void) src0;
    (void) src1;
    (void) hids;
    (void) shmem;
    (void) tiisg;
    (void) sgitg;
}
