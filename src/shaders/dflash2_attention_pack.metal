#include <metal_stdlib>
using namespace metal;

struct DFlash2AttentionPackParams {
    uint lanes;
    uint q_heads;
    uint kv_heads;
    uint head_dim;
    uint rows;
    uint window;
    uint packed_len;
    uint max_commit;
};

struct DFlash2MaskRowsParams {
    uint lanes;
    uint rows;
    uint hidden;
};

struct DFlash2MoeRouteMaskParams {
    uint physical_lane[4];
    uint lanes;
    uint rows_per_lane;
    uint top_k;
    uint n_experts;
    uint width;
};

kernel void dflash2_pack_attention_bf16(
    device const bfloat *ring_keys [[buffer(0)]],
    device const bfloat *ring_values [[buffer(1)]],
    device const float *current_queries [[buffer(2)]],
    device const float *current_keys [[buffer(3)]],
    device const float *current_values [[buffer(4)]],
    device const uint *cache_lengths [[buffer(5)]],
    device const uint *write_positions [[buffer(6)]],
    device bfloat *packed_queries [[buffer(7)]],
    device bfloat *packed_keys [[buffer(8)]],
    device bfloat *packed_values [[buffer(9)]],
    constant DFlash2AttentionPackParams &p [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    const uint kv_elements = p.lanes * p.kv_heads * p.packed_len * p.head_dim;
    const uint query_elements = p.lanes * p.q_heads * p.rows * p.head_dim;
    if (gid < query_elements) {
        uint q = gid;
        const uint dim = q % p.head_dim; q /= p.head_dim;
        const uint row = q % p.rows; q /= p.rows;
        const uint head = q % p.q_heads;
        const uint lane = q / p.q_heads;
        const uint src = ((lane * p.rows + row) * p.q_heads + head) * p.head_dim + dim;
        packed_queries[gid] = bfloat(current_queries[src]);
    }
    if (gid >= kv_elements) return;
    uint x = gid;
    const uint dim = x % p.head_dim; x /= p.head_dim;
    const uint packed = x % p.packed_len; x /= p.packed_len;
    const uint head = x % p.kv_heads;
    const uint lane = x / p.kv_heads;
    bfloat key = bfloat(0.0f);
    bfloat value = bfloat(0.0f);
    if (packed < p.rows) {
        const uint src = ((lane * p.rows + packed) * p.kv_heads + head) * p.head_dim + dim;
        key = bfloat(current_keys[src]);
        value = bfloat(current_values[src]);
    } else {
        const uint age = packed - p.rows;
        const uint live = min(cache_lengths[lane], p.window - 1);
        if (age < live) {
            const uint slot = (write_positions[lane] + p.window - 1 - age) % p.window;
            const uint src = ((lane * p.kv_heads + head) * p.window + slot) * p.head_dim + dim;
            key = ring_keys[src];
            value = ring_values[src];
        }
    }
    packed_keys[gid] = key;
    packed_values[gid] = value;
}

kernel void dflash2_fill_attention_mask_bf16(
    device const uint *cache_lengths [[buffer(0)]],
    device bfloat *mask [[buffer(1)]],
    constant DFlash2AttentionPackParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const uint elements = p.lanes * p.q_heads * p.rows * p.packed_len;
    if (gid >= elements) return;
    uint x = gid;
    const uint packed = x % p.packed_len; x /= p.packed_len;
    const uint row = x % p.rows; x /= p.rows;
    x /= p.q_heads;
    const uint lane = x;
    bool visible = packed < p.rows;
    if (!visible) {
        const uint age = packed - p.rows;
        const uint old_visible = min(cache_lengths[lane], p.window - 1 - row);
        visible = age < old_visible;
    }
    mask[gid] = visible ? bfloat(0.0f) : bfloat(-INFINITY);
}

kernel void dflash2_commit_kv_bf16(
    device const float *new_keys [[buffer(0)]],
    device const float *new_values [[buffer(1)]],
    device const uint *commit_counts [[buffer(2)]],
    device const uint *write_positions [[buffer(3)]],
    device bfloat *ring_keys [[buffer(4)]],
    device bfloat *ring_values [[buffer(5)]],
    constant DFlash2AttentionPackParams &p [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint elements = p.lanes * p.max_commit * p.kv_heads * p.head_dim;
    if (gid >= elements) return;
    uint x = gid;
    const uint dim = x % p.head_dim; x /= p.head_dim;
    const uint head = x % p.kv_heads; x /= p.kv_heads;
    const uint token = x % p.max_commit;
    const uint lane = x / p.max_commit;
    if (token >= commit_counts[lane]) return;
    const uint src = ((lane * p.max_commit + token) * p.kv_heads + head) * p.head_dim + dim;
    const uint slot = (write_positions[lane] + token) % p.window;
    const uint dst = ((lane * p.kv_heads + head) * p.window + slot) * p.head_dim + dim;
    ring_keys[dst] = bfloat(new_keys[src]);
    ring_values[dst] = bfloat(new_values[src]);
}

kernel void dflash2_set_mask_rows_f32(
    device const bfloat *mask_embedding [[buffer(0)]],
    device float *hidden [[buffer(1)]],
    constant DFlash2MaskRowsParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const uint elements = p.lanes * (p.rows - 1) * p.hidden;
    if (gid >= elements) return;
    uint x = gid;
    const uint channel = x % p.hidden; x /= p.hidden;
    const uint masked_row = x % (p.rows - 1);
    const uint lane = x / (p.rows - 1);
    const uint dst = (lane * p.rows + masked_row + 1) * p.hidden + channel;
    hidden[dst] = float(mask_embedding[channel]);
}

kernel void dflash2_mask_moe_routes(
    device const uint *verify_rows [[buffer(0)]],
    device uint *ids [[buffer(1)]],
    device float *routing_weights [[buffer(2)]],
    constant DFlash2MoeRouteMaskParams &p [[buffer(3)]],
    uint pair [[thread_position_in_grid]]) {
    const uint pairs = p.width * p.top_k;
    if (pair >= pairs) return;
    const uint token = pair / p.top_k;
    const uint lane = token / p.rows_per_lane;
    const uint row = token % p.rows_per_lane;
    if (lane < p.lanes &&
        row < min(verify_rows[p.physical_lane[lane]], p.rows_per_lane)) return;
    ids[pair] = p.n_experts;
    routing_weights[pair] = 0.0f;
}
