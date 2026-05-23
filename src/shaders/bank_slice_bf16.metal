// ADR-033 §Pi Task #25 iter 16 (2026-05-23) — K-bank slice copy for the
// chunk-scan bank-split arc. Extracts a [rows, K_bank] BF16 slice from
// a [rows, K_full] BF16 source where K_full is laid out innermost.
//
// Usage: in the K=256 bank-split path of the chunk-parallel gated
// DeltaNet, q and k are [B, T, Hg, K_full=256] row-major K-innermost.
// To dispatch the existing K=128 chunk pipeline twice (once per bank),
// we materialize per-bank temp buffers via this kernel: one call with
// `bank_offset=0` extracts K[0..128], another with `bank_offset=128`
// extracts K[128..256].
//
// Math:
//     dst[r, k] = src[r, bank_offset + k]    for r ∈ [0, rows), k ∈ [0, k_bank)
//
// Dispatch geometry:
//     threadgroups   = (ceil(k_bank / 32), rows, 1)
//     threads_per_tg = (32, 1, 1)
//
// Buffer layout:
//     buffer(0): src         device const bfloat *  [rows * k_full]
//     buffer(1): dst         device       bfloat *  [rows * k_bank]
//     buffer(2): params      constant BankSliceParams &
//
// BankSliceParams:
//     rows         — output row count
//     k_full       — source K dimension (innermost)
//     k_bank       — destination K dimension (slice length)
//     bank_offset  — source K offset to start reading from

#include <metal_stdlib>
using namespace metal;

struct BankSliceParams {
    uint rows;
    uint k_full;
    uint k_bank;
    uint bank_offset;
};

kernel void bank_slice_bf16(
    device const bfloat * src     [[buffer(0)]],
    device       bfloat * dst     [[buffer(1)]],
    constant BankSliceParams & p  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint k = gid.x;
    const uint r = gid.y;
    if (k >= p.k_bank || r >= p.rows) return;

    const uint src_idx = r * p.k_full + p.bank_offset + k;
    const uint dst_idx = r * p.k_bank + k;
    dst[dst_idx] = src[src_idx];
}
