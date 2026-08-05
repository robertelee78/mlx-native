// DeepSeek-V4 sparse-selection additive flash mask.

#include <metal_stdlib>
using namespace metal;

struct DeepSeekSparsePrefillMaskParams {
    uint batch;
    uint query_len;
    uint kv_len;
    uint top_k;
    uint heads;
};

kernel void deepseek_sparse_prefill_mask_fill_bf16(
        constant DeepSeekSparsePrefillMaskParams &p [[buffer(0)]],
        device bfloat *mask                          [[buffer(1)]],
        uint gid                                     [[thread_position_in_grid]]) {
    const ulong total = ulong(p.batch) * p.heads * p.query_len * p.kv_len;
    if (gid >= total) return;
    mask[gid] = bfloat(-INFINITY);
}

kernel void deepseek_sparse_prefill_mask_scatter_bf16(
        constant DeepSeekSparsePrefillMaskParams &p [[buffer(0)]],
        device const int *indices                    [[buffer(1)]],
        device bfloat *mask                          [[buffer(2)]],
        uint gid                                     [[thread_position_in_grid]]) {
    const ulong total = ulong(p.batch) * p.query_len * p.top_k * p.heads;
    if (gid >= total) return;
    const uint head = gid % p.heads;
    const uint slot = (gid / p.heads) % p.top_k;
    const uint query = (gid / (p.heads * p.top_k)) % p.query_len;
    const uint batch = gid / (p.heads * p.top_k * p.query_len);
    const int selected = indices[(ulong(batch) * p.query_len + query) * p.top_k + slot];
    if (selected < 0 || selected >= int(p.kv_len)) return;
    const ulong out = ((ulong(batch) * p.heads + head) * p.query_len + query)
        * p.kv_len + uint(selected);
    mask[out] = bfloat(0.0f);
}

kernel void deepseek_sparse_prefill_mask_fill_f16(
        constant DeepSeekSparsePrefillMaskParams &p [[buffer(0)]],
        device half *mask                            [[buffer(1)]],
        uint gid                                     [[thread_position_in_grid]]) {
    const ulong total = ulong(p.batch) * p.heads * p.query_len * p.kv_len;
    if (gid >= total) return;
    mask[gid] = half(-INFINITY);
}

kernel void deepseek_sparse_prefill_mask_scatter_f16(
        constant DeepSeekSparsePrefillMaskParams &p [[buffer(0)]],
        device const int *indices                    [[buffer(1)]],
        device half *mask                            [[buffer(2)]],
        uint gid                                     [[thread_position_in_grid]]) {
    const ulong total = ulong(p.batch) * p.query_len * p.top_k * p.heads;
    if (gid >= total) return;
    const uint head = gid % p.heads;
    const uint slot = (gid / p.heads) % p.top_k;
    const uint query = (gid / (p.heads * p.top_k)) % p.query_len;
    const uint batch = gid / (p.heads * p.top_k * p.query_len);
    const int selected = indices[(ulong(batch) * p.query_len + query) * p.top_k + slot];
    if (selected < 0 || selected >= int(p.kv_len)) return;
    const ulong out = ((ulong(batch) * p.heads + head) * p.query_len + query)
        * p.kv_len + uint(selected);
    mask[out] = half(0.0f);
}
