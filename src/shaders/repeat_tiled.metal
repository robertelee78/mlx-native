#include <metal_stdlib>
using namespace metal;

// =============================================================================
// repeat_tiled_f32 — Tiled GQA broadcast of a [T, Hg, K] F32 tensor into a
//                    [T, H, K] F32 output, where dst[t, h, k] = src[t, h % Hg, k].
//
// This is the GPU analog of llama.cpp's `ggml_repeat_4d` op as used in
// `delta-net-base.cpp:357 k = ggml_repeat(ctx0, k, s)` for the GQA broadcast
// of K (and the matching Q expansion in hf2q's `apply_gated_delta_net_chunk`
// wrapper, which uses TILED replication to match Qwen3.6's GGUF tensor
// layout — see `gpu_delta_net.rs:834-866` for the convention rationale).
//
// Layout (row-major contiguous F32, no strides):
//   src:  [T, Hg, K]   — element count = T * Hg * K
//   dst:  [T, H,  K]   — element count = T * H  * K   (H % Hg == 0)
//
// Mapping (TILED, per W-5b.4 / `project_qwen36_gqa_tiled_vs_block`):
//   dst[t, h, k] = src[t, h % Hg, k]
//
// Buffer bindings:
//   buffer(0): src    — float (input,  [T, Hg, K])
//   buffer(1): dst    — float (output, [T, H, K])
//   buffer(2): params — uint [3] — {seq=T, hg=Hg, h=H, k=K}
//
// Grid:        (K, H, T)            — one thread per OUTPUT element
// Threadgroup: (min(256, K), 1, 1)
//
// Each thread reads from src[t * Hg * K + (h % Hg) * K + k] and writes to
// dst[t * H * K + h * K + k]. The src reads collapse to Hg distinct
// addresses per (t,k) pair — well-cached on Apple unified memory.
//
// W-5b.19 worker (ADR-005, 2026-04-27): replaces the hf2q-side CPU triple-
// loop tiled-replicate at `gpu_delta_net.rs:893-940` (`q_expanded` /
// `k_expanded` fill, ~497 ms / 10.4 ms-per-layer at PP4106). Production
// caller: `hf2q::inference::models::qwen35::gpu_delta_net::
// apply_gated_delta_net_chunk` (chunk-prefill GQA pre-expansion).
// =============================================================================

struct RepeatTiledParams {
    uint seq;  // T
    uint hg;   // Hg (n_k_heads)
    uint h;    // H  (n_v_heads)
    uint k;    // K  (head dim)
};

kernel void repeat_tiled_f32(
    device const float*           src    [[buffer(0)]],
    device float*                 dst    [[buffer(1)]],
    constant RepeatTiledParams&   params [[buffer(2)]],
    uint3 pos [[thread_position_in_grid]]
) {
    const uint k = pos.x;        // index within head dim
    const uint h = pos.y;        // dst head index (0..H)
    const uint t = pos.z;        // token index    (0..T)

    if (k >= params.k || h >= params.h || t >= params.seq) return;

    // TILED mapping — `kh = h % Hg` (matches Qwen3.6 GGUF layout, llama's
    // ggml_repeat_4d, and the wrapper-side fix at gpu_delta_net.rs:927).
    const uint kh = h % params.hg;

    const uint src_idx = t * params.hg * params.k + kh * params.k + k;
    const uint dst_idx = t * params.h  * params.k + h  * params.k + k;

    dst[dst_idx] = src[src_idx];
}
