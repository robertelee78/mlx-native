#include <metal_stdlib>
using namespace metal;

/// FP32 embedding table lookup.
///
///   output[b, h] = embedding[ids[b], h]   for 0 ≤ b < batch, 0 ≤ h < hidden
///
/// Buffer layout:
///   buffer(0): embedding — float[vocab, hidden]  (the lookup table)
///   buffer(1): ids       — uint32[batch]         (token IDs; must satisfy 0 ≤ ids[b] < vocab)
///   buffer(2): output    — float[batch, hidden]
///   buffer(3): params    — uint[2]: (vocab, hidden)
///
/// Grid: 2D (hidden, batch); one thread per output element.
kernel void embedding_lookup_f32(
    device const float    *embedding [[buffer(0)]],
    device const uint32_t *ids       [[buffer(1)]],
    device float          *output    [[buffer(2)]],
    device const uint     *params    [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint vocab  = params[0];
    const uint hidden = params[1];
    const uint h = tid.x;
    const uint b = tid.y;
    if (h >= hidden) return;
    const uint id = ids[b];
    // Guard against out-of-range IDs — silently return 0 instead of OOB read.
    if (id >= vocab) {
        output[b * hidden + h] = 0.0f;
        return;
    }
    output[b * hidden + h] = embedding[id * hidden + h];
}

/// FP32 embedding backward: scatter-add of upstream gradient by token ID.
///
///   d_embedding[id, h] = Σ_{b: ids[b] == id} dy[b, h]
///
/// Each thread is assigned a unique (id, h) pair and scans over all batch
/// positions, accumulating contributions from any batch that points to its id.
/// No atomics needed since each thread writes a unique (id, h) cell.
///
/// Buffer layout:
///   buffer(0): dy           — float[batch, hidden]   (upstream gradient)
///   buffer(1): ids          — uint32[batch]          (forward token IDs)
///   buffer(2): d_embedding  — float[vocab, hidden]   (output; caller pre-zeros)
///   buffer(3): params       — uint[3]: (vocab, hidden, batch)
///
/// Grid: 2D (hidden, vocab); one thread per output cell.
kernel void embedding_scatter_add_f32(
    device const float    *dy           [[buffer(0)]],
    device const uint32_t *ids          [[buffer(1)]],
    device float          *d_embedding  [[buffer(2)]],
    device const uint     *params       [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint vocab  = params[0];
    const uint hidden = params[1];
    const uint batch  = params[2];
    const uint h  = tid.x;
    const uint id = tid.y;
    if (h >= hidden || id >= vocab) return;
    float acc = 0.0f;
    for (uint b = 0; b < batch; ++b) {
        if (ids[b] == id) {
            acc += dy[b * hidden + h];
        }
    }
    d_embedding[id * hidden + h] = acc;
}
