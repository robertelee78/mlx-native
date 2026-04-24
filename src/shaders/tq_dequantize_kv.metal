// TQ KV dequantize kernel — iter-21 Track A fix.
//
// Reads nibble-packed TurboQuant K or V cache at a given position and
// writes a dense F32 buffer of shape [num_kv_heads, head_dim].
//
// This enables the Leg F ablation: encode K/V as TQ, then dequantize back
// to F32 and dispatch the dense flash_attn_vec kernel. The output is in the
// FWHT-rotated domain (same as the TQ SDPA readpath), NOT the original F32.
// This isolates the SDPA kernel math from TQ representation noise.
//
// Dequant formula MUST match flash_attn_vec_tq.metal inline dequant exactly
// (see flash_attn_vec_tq.metal:305-348):
//   D=256: scale_norm = norm * inv_sqrt(256)     — single-norm convention (unchanged)
//   D=512: scale_norm = norm / scale_factor_d512 — per-block norm, NO inv_sqrt factor
//          (iter-16 fix: encoder stores raw blk_norm; decoder uses blk_norm directly)
//
// IMPORTANT (iter-21 Track A): iter-20 erroneously applied inv_sqrt_hd to BOTH
// D=256 and D=512. For D=512 this introduces a sqrt(512)≈22.6x scale error vs
// the TQ SDPA kernel, corrupting attention scores for global-attention layers
// even when the prefill shadow cache is correctly populated.
//
// Packed layout: [num_kv_heads, cache_capacity, head_dim/2] u8 (nibble-packed)
// Norms layout:
//   D=256: [num_kv_heads, cache_capacity] f32 — 1 norm per position
//   D=512: [num_kv_heads, cache_capacity, 2] f32 — 2 per-block norms per position
// Output: [num_kv_heads, head_dim] f32 — dense dequantized values for ONE position
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
    // inv_sqrt_hd is only used for D=256 (single-norm convention).
    // D=512 uses per-block raw norm (no inv_sqrt_dk factor) per iter-16 fix.
    const float inv_sqrt_hd = rsqrt(float(hd));
    const float sf      = params.scale_factor_d512;
    const bool is_d512  = (hd > 256);

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
        uint block_idx = is_d512 ? (coord / 256u) : 0u;
        block_idx = min(block_idx, npp - 1);
        float norm = norms_pos[block_idx];

        // scale_norm: must match flash_attn_vec_tq.metal decode convention exactly.
        //   D=256: norm * inv_sqrt(256)     — single-norm, same as before
        //   D=512: norm / scale_factor_d512 — per-block raw norm, NO inv_sqrt factor
        // (iter-21 Track A: iter-20 incorrectly applied inv_sqrt_hd to D=512 also,
        //  introducing a sqrt(512)≈22.6x scale error for global-attention layers.)
        float scale_norm;
        if (is_d512) {
            scale_norm = norm / sf;
        } else {
            scale_norm = norm * inv_sqrt_hd;
        }

        // Read nibble for this coord
        uint byte_idx = coord / 2;
        uint8_t packed_byte = packed_pos[byte_idx];
        uint nibble = (coord % 2 == 0) ? (packed_byte & 0xFu) : ((packed_byte >> 4u) & 0xFu);

        dst_head[coord] = CODEBOOK_4BIT_DQ[nibble] * scale_norm;
    }
}

// ============================================================================
// Track B (iter-21): higher-bit dequantize kernel.
// Reads byte-packed 5-bit or 6-bit indices from the higher-bit KV cache and
// writes dense F32 values in the FWHT-rotated domain (same as tq_dequantize_kv).
//
// Packed layout: [num_kv_heads, capacity, head_dim] u8 (byte-packed, 1 byte/elem)
// Norms layout: same as 4-bit (D=256: 1 norm/pos, D=512: 2 per-block norms/pos)
// ============================================================================

constant float CODEBOOK_5BIT_DQ[32] = {
    -3.2606790f, -2.6910589f, -2.3176743f, -2.0286608f,
    -1.7871646f, -1.5761599f, -1.3862739f, -1.2117410f,
    -1.0487242f, -0.8945114f, -0.7470884f, -0.6048936f,
    -0.4666676f, -0.3313550f, -0.1980377f, -0.0658849f,
     0.0658849f,  0.1980377f,  0.3313550f,  0.4666676f,
     0.6048936f,  0.7470884f,  0.8945114f,  1.0487242f,
     1.2117410f,  1.3862739f,  1.5761599f,  1.7871646f,
     2.0286608f,  2.3176743f,  2.6910589f,  3.2606790f,
};

constant float CODEBOOK_6BIT_DQ[64] = {
    -3.6996161f, -3.1907215f, -2.8640626f, -2.6161277f,
    -2.4129324f, -2.2388464f, -2.0853192f, -1.9471373f,
    -1.8208742f, -1.7041502f, -1.5952401f, -1.4928497f,
    -1.3959804f, -1.3038428f, -1.2157998f, -1.1313277f,
    -1.0499889f, -0.9714118f, -0.8952766f, -0.8213046f,
    -0.7492492f, -0.6788902f, -0.6100285f, -0.5424819f,
    -0.4760822f, -0.4106724f, -0.3461048f, -0.2822386f,
    -0.2189392f, -0.1560761f, -0.0935225f, -0.0311537f,
     0.0311537f,  0.0935225f,  0.1560761f,  0.2189392f,
     0.2822386f,  0.3461048f,  0.4106724f,  0.4760822f,
     0.5424819f,  0.6100285f,  0.6788902f,  0.7492492f,
     0.8213046f,  0.8952766f,  0.9714118f,  1.0499889f,
     1.1313277f,  1.2157998f,  1.3038428f,  1.3959804f,
     1.4928497f,  1.5952401f,  1.7041502f,  1.8208742f,
     1.9471373f,  2.0853192f,  2.2388464f,  2.4129324f,
     2.6161277f,  2.8640626f,  3.1907215f,  3.6996161f,
};

// ============================================================================
// 8-bit Lloyd-Max codebook for N(0,1): 256 reconstruction levels.
// Must match CODEBOOK_8BIT in hadamard_quantize_kv_fast.metal exactly.
// ============================================================================

constant float CODEBOOK_8BIT_DQ[256] = {
    -4.0354801f,  -3.5656248f,  -3.2681869f,  -3.0454753f,
    -2.8654909f,  -2.7135514f,  -2.5816441f,  -2.4648947f,
    -2.3601072f,  -2.2650656f,  -2.1781659f,  -2.0982056f,
    -2.0242566f,  -1.9555844f,  -1.8915951f,  -1.8317989f,
    -1.7757850f,  -1.7232032f,  -1.6737511f,  -1.6271638f,
    -1.5832071f,  -1.5416716f,  -1.5023685f,  -1.4651264f,
    -1.4297889f,  -1.3962121f,  -1.3642638f,  -1.3338215f,
    -1.3047719f,  -1.2770099f,  -1.2504379f,  -1.2249654f,
    -1.2005084f,  -1.1769889f,  -1.1543349f,  -1.1324796f,
    -1.1113613f,  -1.0909233f,  -1.0711132f,  -1.0518828f,
    -1.0331881f,  -1.0149885f,  -0.9972470f,  -0.9799298f,
    -0.9630061f,  -0.9464477f,  -0.9302290f,  -0.9143267f,
    -0.8987195f,  -0.8833881f,  -0.8683149f,  -0.8534840f,
    -0.8388808f,  -0.8244918f,  -0.8103051f,  -0.7963095f,
    -0.7824948f,  -0.7688517f,  -0.7553715f,  -0.7420464f,
    -0.7288690f,  -0.7158325f,  -0.7029306f,  -0.6901574f,
    -0.6775076f,  -0.6649758f,  -0.6525573f,  -0.6402476f,
    -0.6280424f,  -0.6159377f,  -0.6039295f,  -0.5920144f,
    -0.5801888f,  -0.5684495f,  -0.5567932f,  -0.5452171f,
    -0.5337183f,  -0.5222939f,  -0.5109414f,  -0.4996583f,
    -0.4884421f,  -0.4772904f,  -0.4662010f,  -0.4551717f,
    -0.4442004f,  -0.4332850f,  -0.4224237f,  -0.4116144f,
    -0.4008554f,  -0.3901447f,  -0.3794808f,  -0.3688618f,
    -0.3582862f,  -0.3477523f,  -0.3372586f,  -0.3268035f,
    -0.3163855f,  -0.3060032f,  -0.2956552f,  -0.2853401f,
    -0.2750566f,  -0.2648032f,  -0.2545787f,  -0.2443818f,
    -0.2342113f,  -0.2240659f,  -0.2139444f,  -0.2038456f,
    -0.1937683f,  -0.1837115f,  -0.1736739f,  -0.1636545f,
    -0.1536520f,  -0.1436655f,  -0.1336938f,  -0.1237359f,
    -0.1137907f,  -0.1038571f,  -0.0939341f,  -0.0840208f,
    -0.0741159f,  -0.0642186f,  -0.0543279f,  -0.0444426f,
    -0.0345618f,  -0.0246845f,  -0.0148097f,  -0.0049364f,
     0.0049364f,   0.0148097f,   0.0246845f,   0.0345618f,
     0.0444426f,   0.0543279f,   0.0642186f,   0.0741159f,
     0.0840208f,   0.0939341f,   0.1038571f,   0.1137907f,
     0.1237359f,   0.1336938f,   0.1436655f,   0.1536520f,
     0.1636545f,   0.1736739f,   0.1837115f,   0.1937683f,
     0.2038456f,   0.2139444f,   0.2240659f,   0.2342113f,
     0.2443818f,   0.2545787f,   0.2648032f,   0.2750566f,
     0.2853401f,   0.2956552f,   0.3060032f,   0.3163855f,
     0.3268035f,   0.3372586f,   0.3477523f,   0.3582862f,
     0.3688618f,   0.3794808f,   0.3901447f,   0.4008554f,
     0.4116144f,   0.4224237f,   0.4332850f,   0.4442004f,
     0.4551717f,   0.4662010f,   0.4772904f,   0.4884421f,
     0.4996583f,   0.5109414f,   0.5222939f,   0.5337183f,
     0.5452171f,   0.5567932f,   0.5684495f,   0.5801888f,
     0.5920144f,   0.6039295f,   0.6159377f,   0.6280424f,
     0.6402476f,   0.6525573f,   0.6649758f,   0.6775076f,
     0.6901574f,   0.7029306f,   0.7158325f,   0.7288690f,
     0.7420464f,   0.7553715f,   0.7688517f,   0.7824948f,
     0.7963095f,   0.8103051f,   0.8244918f,   0.8388808f,
     0.8534840f,   0.8683149f,   0.8833881f,   0.8987195f,
     0.9143267f,   0.9302290f,   0.9464477f,   0.9630061f,
     0.9799298f,   0.9972470f,   1.0149885f,   1.0331881f,
     1.0518828f,   1.0711132f,   1.0909233f,   1.1113613f,
     1.1324796f,   1.1543349f,   1.1769889f,   1.2005084f,
     1.2249654f,   1.2504379f,   1.2770099f,   1.3047719f,
     1.3338215f,   1.3642638f,   1.3962121f,   1.4297889f,
     1.4651264f,   1.5023685f,   1.5416716f,   1.5832071f,
     1.6271638f,   1.6737511f,   1.7232032f,   1.7757850f,
     1.8317989f,   1.8915951f,   1.9555844f,   2.0242566f,
     2.0982056f,   2.1781659f,   2.2650656f,   2.3601072f,
     2.4648947f,   2.5816441f,   2.7135514f,   2.8654909f,
     3.0454753f,   3.2681869f,   3.5656248f,   4.0354801f,
};

struct TqDequantizeHbKvParams {
    uint head_dim;
    uint num_kv_heads;
    uint read_pos;
    uint cache_capacity;
    uint norms_per_pos;
    float scale_factor_d512;
    uint codebook_bits;  // 5, 6, or 8
};

// ============================================================================
// Track B alternate approach: requantize-to-F32 kernel.
// Takes FWHT-rotated+normalized F32 K/V (from attn_k_normed after FWHT+norm)
// and snaps each value to the nearest 5-bit or 6-bit centroid, then writes
// the centroid value back as F32.  This simulates the quantize→dequantize
// round-trip without allocating a separate packed buffer.
//
// Input: [num_kv_heads, head_dim] f32 — post-FWHT normalized values
// Output: same shape — centroid-snapped F32 values
//
// Workflow for Track B:
//  1. Encode K/V to TQ 4-bit (existing hadamard_quantize_kv_fast path)
//  2. Dequantize K/V to F32 (tq_dequantize_kv) into attn_k_normed
//  3. Apply requantize_to_f32_hb to snap values to 5/6-bit centroids
//  4. Copy requantized F32 into leg_hb_kvs shadow cache
// ============================================================================
// Not used in the primary Track B ablation (see tq_dequantize_hb_kv below);
// this is here for future use if a cleaner 1-step path is needed.

kernel void tq_dequantize_hb_kv(
    device const uint8_t            *packed    [[buffer(0)]], // [nkv, capacity, hd] u8 byte-packed
    device const float              *norms     [[buffer(1)]],
    device       float              *dst       [[buffer(2)]], // [nkv, hd] f32
    constant TqDequantizeHbKvParams &params   [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    uint   tiitg [[thread_index_in_threadgroup]])
{
    const uint kv_head  = tgpig[0];
    const uint hd       = params.head_dim;
    const uint cap      = params.cache_capacity;
    const uint pos      = params.read_pos;
    const float inv_sqrt_hd = rsqrt(float(hd));
    const float sf      = params.scale_factor_d512;
    const bool is_d512  = (hd > 256);
    const bool use_5bit = (params.codebook_bits == 5u);

    if (kv_head >= params.num_kv_heads) return;

    // Packed base: [kv_head, pos, 0..hd] — byte-packed (1 byte per element)
    device const uint8_t *packed_pos = packed + kv_head * cap * hd + pos * hd;

    const uint npp = params.norms_per_pos;
    device const float *norms_pos = norms + kv_head * cap * npp + pos * npp;
    device float *dst_head = dst + kv_head * hd;

    for (uint coord = tiitg; coord < hd; coord += hd) {
        uint block_idx = is_d512 ? (coord / 256u) : 0u;
        block_idx = min(block_idx, npp - 1u);
        float norm = norms_pos[block_idx];

        // Same scale convention as tq_dequantize_kv (Track A fix)
        float scale_norm = is_d512 ? (norm / sf) : (norm * inv_sqrt_hd);

        uint idx = packed_pos[coord];  // byte-packed index
        float centroid;
        if (use_5bit) {
            centroid = CODEBOOK_5BIT_DQ[idx & 0x1Fu];
        } else if (params.codebook_bits == 6u) {
            centroid = CODEBOOK_6BIT_DQ[idx & 0x3Fu];
        } else {
            centroid = CODEBOOK_8BIT_DQ[idx];  // 8-bit: full byte is the index
        }
        dst_head[coord] = centroid * scale_norm;
    }
}
