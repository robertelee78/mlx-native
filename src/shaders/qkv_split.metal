#include <metal_stdlib>
using namespace metal;

// =============================================================================
// qkv_split_f32 — Strided split of a fused QKV tensor into Q/K/V outputs.
//
// Input layout (per token, contiguous f32):
//
//   qkv[t, :] = [ Q (q_sp) | K (k_sp) | V (v_sp) ]   (length = qkv_ch)
//
// where q_sp = n_k_heads * d_k, k_sp = n_k_heads * d_k, v_sp = n_v_heads * d_v,
// and qkv_ch = q_sp + k_sp + v_sp.
//
// Outputs:
//   q[t * q_sp + i] = qkv[t * qkv_ch + i]                       for i in 0..q_sp
//   k[t * k_sp + i] = qkv[t * qkv_ch + q_sp + i]                for i in 0..k_sp
//   v[t * v_sp + i] = qkv[t * qkv_ch + q_sp + k_sp + i]         for i in 0..v_sp
//
// Buffer bindings:
//   buffer(0): qkv    — float (input,  fused)
//   buffer(1): q      — float (output)
//   buffer(2): k      — float (output)
//   buffer(3): v      — float (output)
//   buffer(4): params — uint [5] — {seq, q_sp, k_sp, v_sp, qkv_ch}
//
// Grid:        (qkv_ch, seq, 1)   — one thread per element of the input
// Threadgroup: (min(256, qkv_ch), 1, 1)
//
// Each thread reads one input element and writes it to exactly one of {q,k,v}
// based on the column index. Perfect coalescing on the input read; the output
// write is strided (3 disjoint output regions) but each region remains
// contiguous within itself, so writes are still coalesced within a head-span.
//
// W-5b.18 worker (ADR-005, 2026-04-27): replaces the hf2q-side
// download_f32 + CPU triple-loop + 3× upload_f32 round-trip in
// `gpu_delta_net.rs::layer_qkv_deinterleave` (838 ms / 17.5 ms per layer).
// =============================================================================

struct QkvSplitParams {
    uint seq;
    uint q_sp;
    uint k_sp;
    uint v_sp;
    uint qkv_ch;
};

kernel void qkv_split_f32(
    device const float*       qkv    [[buffer(0)]],
    device float*             q      [[buffer(1)]],
    device float*             k      [[buffer(2)]],
    device float*             v      [[buffer(3)]],
    constant QkvSplitParams&  params [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint col = pos.x;       // index within qkv_ch
    const uint row = pos.y;       // token index (0..seq)

    if (col >= params.qkv_ch || row >= params.seq) return;

    const uint src_idx = row * params.qkv_ch + col;
    const float val    = qkv[src_idx];

    const uint qk_boundary = params.q_sp + params.k_sp;

    if (col < params.q_sp) {
        // Q region.
        const uint dst_idx = row * params.q_sp + col;
        q[dst_idx] = val;
    } else if (col < qk_boundary) {
        // K region.
        const uint k_off   = col - params.q_sp;
        const uint dst_idx = row * params.k_sp + k_off;
        k[dst_idx] = val;
    } else {
        // V region.
        const uint v_off   = col - qk_boundary;
        const uint dst_idx = row * params.v_sp + v_off;
        v[dst_idx] = val;
    }
}
