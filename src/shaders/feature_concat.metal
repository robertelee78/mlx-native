// feature_concat_f32 — Metal shader for ADR-021 K5.
//
// Strided copy of one chunk into its slice of the concatenated
// destination tensor. For a chunk of shape [T, D] f32 row-major,
// computes:
//
//   for t in 0..T:
//     for d in 0..D:
//       dst[t * dst_stride + dst_offset + d] = src[t * D + d]
//
// where `dst_stride = D_total = sum of all chunk D's`. Launching
// once per chunk (varying `dst_offset`) builds the full concatenated
// `[T, D_total]` row-major tensor with each row carrying
// `[chunk0[t]; chunk1[t]; ...; chunkN[t]]` — matching qwen3vl.cpp:186
// `ggml_concat(ctx0, embeddings, deepstack_features, 0)`.

#include <metal_stdlib>
using namespace metal;

// Must match `GpuFeatureConcatParams` in src/ops/feature_concat.rs.
struct FeatureConcatParams {
    uint n_tokens;       // T
    uint src_dim;        // D_i (chunk feature width)
    uint dst_offset;     // start column for this chunk in the [T, D_total] dst
    uint dst_stride;     // D_total (row stride of dst)
};

// Buffers:
//   0: params — FeatureConcatParams
//   1: src    — float [T * src_dim] row-major (one chunk)
//   2: dst    — float [T * dst_stride] row-major (concatenated tensor;
//               this kernel only writes to the [t, dst_offset .. dst_offset+src_dim] slice)
kernel void feature_concat_f32(
    constant FeatureConcatParams& params [[buffer(0)]],
    device const float*           src    [[buffer(1)]],
    device float*                 dst    [[buffer(2)]],
    uint                          gid    [[thread_position_in_grid]]
) {
    const uint total = params.n_tokens * params.src_dim;
    if (gid >= total) {
        return;
    }
    const uint d_dim = params.src_dim;
    const uint t = gid / d_dim;
    const uint d = gid - t * d_dim;
    const uint dst_idx = t * params.dst_stride + params.dst_offset + d;
    dst[dst_idx] = src[gid];
}
