// TQ KV dequantize kernel — iter-20 Leg F ablation.
//
// Reads nibble-packed TurboQuant K or V cache at a given position and
// writes a dense F32 buffer of shape [num_kv_heads, head_dim].
//
// This enables the Leg F ablation: encode K/V as TQ, then dequantize back
// to F32 and dispatch the dense flash_attn_vec kernel. The output is in the
// FWHT-rotated domain (same as the TQ SDPA readpath), NOT the original F32.
// This isolates the SDPA kernel math from TQ representation noise.
//
// Dequant formula (matches flash_attn_vec_tq.metal inline dequant exactly):
//   D=256: value = CODEBOOK_4BIT[nibble] * inv_sqrt(head_dim) * norm
//   D=512: block0 (coords 0..255)  = CODEBOOK_4BIT[nibble] * inv_sqrt(head_dim) * norm0
//          block1 (coords 256..511) = CODEBOOK_4BIT[nibble] * inv_sqrt(head_dim) * norm1
//   where norm0, norm1 are the per-block norms (norms_per_pos=2).
//
// Packed layout: [num_kv_heads, cache_capacity, head_dim/2] u8 (nibble-packed)
// Norms layout:
//   D=256: [num_kv_heads, cache_capacity] f32 — 1 norm per position
//   D=512: [num_kv_heads, cache_capacity, 2] f32 — 2 per-block norms per position
// Output: [num_kv_heads, head_dim] f32 — dense dequantized values for ONE position

#include <metal_stdlib>
using namespace metal;

constant float CODEBOOK_4BIT_DQ[16] = {
    -2.7325896f, -2.0690172f, -1.6180464f, -1.2562312f,
    -0.9423405f, -0.6567591f, -0.3880483f, -0.1283950f,
     0.1283950f,  0.3880483f,  0.6567591f,  0.9423405f,
     1.2562312f,  1.6180464f,  2.0690172f,  2.7325896f,
};

struct TqDequantizeKvParams {
    uint head_dim;         // 256 or 512
    uint num_kv_heads;     // number of KV heads
    uint read_pos;         // cache position to read from (already wrapped for ring buffers)
    uint cache_capacity;   // KV cache capacity (stride in packed/norms buffers)
    uint norms_per_pos;    // 1 for D=256, 2 for D=512
    float scale_factor_d512; // iter-18 S2B: scale divisor for D=512 norm (1.0=bare)
};

// One threadgroup per KV head; head_dim threads per threadgroup.
// Each thread dequantizes head_dim/num_threads elements.
kernel void tq_dequantize_kv(
    device const uint8_t         *packed    [[buffer(0)]], // [nkv, capacity, hd/2] u8
    device const float           *norms     [[buffer(1)]], // [nkv, capacity, norms_per_pos] f32
    device       float           *dst       [[buffer(2)]], // [nkv, hd] f32 — OUTPUT
    constant TqDequantizeKvParams &params   [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],  // threadgroup = kv head index
    uint   tiitg [[thread_index_in_threadgroup]])   // thread index within threadgroup
{
    const uint kv_head  = tgpig[0];
    const uint hd       = params.head_dim;
    const uint cap      = params.cache_capacity;
    const uint pos      = params.read_pos;
    const float inv_sqrt_hd = rsqrt(float(hd));
    const float sf      = params.scale_factor_d512;

    if (kv_head >= params.num_kv_heads) return;

    // Packed base: [kv_head, pos, 0..hd/2]
    device const uint8_t *packed_pos = packed + kv_head * cap * (hd / 2) + pos * (hd / 2);

    // Norms base for this head+pos: at offset kv_head * cap * norms_per_pos + pos * norms_per_pos
    const uint npp = params.norms_per_pos;
    device const float *norms_pos = norms + kv_head * cap * npp + pos * npp;

    // Output base: [kv_head, 0..hd]
    device float *dst_head = dst + kv_head * hd;

    // Each thread processes one coordinate (tiitg = coord index).
    // Head_dim threads per threadgroup → every coord handled by exactly one thread.
    // For large head_dim we may need multiple passes; here threadgroups are sized to hd.
    for (uint coord = tiitg; coord < hd; coord += /* threads per TG */ hd) {
        // Determine which block this coord falls in (for D=512 per-block norms)
        uint block_idx = (hd > 256) ? (coord / 256) : 0;
        block_idx = min(block_idx, npp - 1);
        float norm = norms_pos[block_idx];
        // Apply scale divisor for D=512 ablation
        if (hd > 256) {
            norm = norm / sf;
        }
        float scale_norm = norm * inv_sqrt_hd;

        // Read nibble for this coord
        uint byte_idx = coord / 2;
        uint8_t packed_byte = packed_pos[byte_idx];
        uint nibble = (coord % 2 == 0) ? (packed_byte & 0xFu) : ((packed_byte >> 4u) & 0xFu);

        dst_head[coord] = CODEBOOK_4BIT_DQ[nibble] * scale_norm;
    }
}
