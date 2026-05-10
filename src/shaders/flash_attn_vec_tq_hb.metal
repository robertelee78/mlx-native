// flash_attn_vec_tq_hb.metal — Native TQ SDPA for 5/6/8-bit byte-packed KV cache.
//
// Variant of flash_attn_vec_tq.metal that reads K/V from byte-packed (1 byte/element)
// higher-bit codebook indices instead of nibble-packed 4-bit indices.
//
// Bit-width is selected at compile time via template parameter CODEBOOK_BITS:
//   5  → 32 centroids  (Lloyd-Max N(0,1) optimal)
//   6  → 64 centroids
//   8  → 256 centroids
//
// Packed buffer layout: [num_kv_heads, capacity, head_dim] u8 (byte-packed)
//   One byte per element. For 5-bit only 5 LSBs are used (upper 3 zero).
//
// Dequant formula (same as tq_dequantize_hb_kv, which must match exactly):
//   D=256: scale_norm = norm * inv_sqrt(256)
//   D=512: scale_norm = norm / scale_factor_d512
//
// ADR-007 iter-24: measure Gate A/B/C at 5/6/8-bit to find shippable bit-width.

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32   // KV positions per simdgroup iteration
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

// Parameters — same layout as FlashAttnVecTqParams in flash_attn_vec_tq.metal.
struct FlashAttnVecTqHbParams {
    uint  n_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  kv_seq_len;
    uint  kv_capacity;
    float scale;
    uint  mask_type;
    uint  sliding_window;
    float softcap;
    uint  nwg;
    uint  ring_start;
    float scale_factor_d512;  // for D=512 norm dequant
    uint  codebook_bits;      // 5, 6, or 8 (runtime selector)
    uint  fuse_fwht_pre;      // ADR-028 iter-106: 0=caller-rotated Q, 1=kernel applies FWHT-pre
    uint  nsg;                // ADR-028 iter-127 Path D: simdgroups per workgroup (power-of-2 in [1, 32], practically capped at 4)
};

// Reduce params — shared with flash_attn_vec.
struct FlashAttnVecReduceParamsHb {
    uint nrows;
};

// ---------------------------------------------------------------------------
// 5-bit codebook (32 centroids, byte-packed — same as hadamard_quantize_kv_fast.metal)
// ---------------------------------------------------------------------------
constant float CODEBOOK_HB_5BIT[32] = {
    -3.2606790f, -2.6910589f, -2.3176743f, -2.0286608f,
    -1.7871646f, -1.5761599f, -1.3862739f, -1.2117410f,
    -1.0487242f, -0.8945114f, -0.7470884f, -0.6048936f,
    -0.4666676f, -0.3313550f, -0.1980377f, -0.0658849f,
     0.0658849f,  0.1980377f,  0.3313550f,  0.4666676f,
     0.6048936f,  0.7470884f,  0.8945114f,  1.0487242f,
     1.2117410f,  1.3862739f,  1.5761599f,  1.7871646f,
     2.0286608f,  2.3176743f,  2.6910589f,  3.2606790f,
};

// ---------------------------------------------------------------------------
// 6-bit codebook (64 centroids)
// ---------------------------------------------------------------------------
constant float CODEBOOK_HB_6BIT[64] = {
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

// ---------------------------------------------------------------------------
// 8-bit codebook (256 centroids, Lloyd-Max N(0,1), iter-24)
// Computed via Lloyd-Max iteration to convergence (tol=1e-12).
// Symmetry error: 3.41e-10. Range: [-5.0652659, +5.0652659].
// Must match CODEBOOK_8BIT in hadamard_quantize_kv_fast.metal exactly.
// ---------------------------------------------------------------------------
constant float CODEBOOK_HB_8BIT[256] = {
    -5.0652659f, -4.6836997f, -4.4467193f, -4.2715508f,
    -4.1311907f, -4.0132856f, -3.9111092f, -3.8205780f,
    -3.7390194f, -3.6645851f, -3.5959415f, -3.5320936f,
    -3.4722785f, -3.4158977f, -3.3624729f, -3.3116156f,
    -3.2630056f, -3.2163758f, -3.1715011f, -3.1281899f,
    -3.0862780f, -3.0456229f, -3.0061011f, -2.9676040f,
    -2.9300362f, -2.8933131f, -2.8573596f, -2.8221086f,
    -2.7874999f, -2.7534795f, -2.7199985f, -2.6870129f,
    -2.6544825f, -2.6223710f, -2.5906452f, -2.5592748f,
    -2.5282321f, -2.4974918f, -2.4670306f, -2.4368270f,
    -2.4068614f, -2.3771157f, -2.3475732f, -2.3182184f,
    -2.2890372f, -2.2600165f, -2.2311440f, -2.2024086f,
    -2.1737998f, -2.1453081f, -2.1169245f, -2.0886408f,
    -2.0604493f, -2.0323430f, -2.0043154f, -1.9763603f,
    -1.9484722f, -1.9206458f, -1.8928763f, -1.8651592f,
    -1.8374904f, -1.8098662f, -1.7822828f, -1.7547372f,
    -1.7272261f, -1.6997469f, -1.6722970f, -1.6448739f,
    -1.6174755f, -1.5900996f, -1.5627445f, -1.5354084f,
    -1.5080897f, -1.4807869f, -1.4534986f, -1.4262237f,
    -1.3989610f, -1.3717093f, -1.3444678f, -1.3172356f,
    -1.2900118f, -1.2627956f, -1.2355865f, -1.2083838f,
    -1.1811868f, -1.1539951f, -1.1268081f, -1.0996255f,
    -1.0724469f, -1.0452718f, -1.0180999f, -0.9909310f,
    -0.9637647f, -0.9366008f, -0.9094390f, -0.8822793f,
    -0.8551212f, -0.8279648f, -0.8008098f, -0.7736561f,
    -0.7465035f, -0.7193520f, -0.6922014f, -0.6650517f,
    -0.6379027f, -0.6107544f, -0.5836067f, -0.5564596f,
    -0.5293129f, -0.5021667f, -0.4750208f, -0.4478753f,
    -0.4207301f, -0.3935852f, -0.3664405f, -0.3392960f,
    -0.3121517f, -0.2850076f, -0.2578636f, -0.2307198f,
    -0.2035761f, -0.1764324f, -0.1492888f, -0.1221453f,
    -0.0950019f, -0.0678584f, -0.0407151f, -0.0135717f,
     0.0135717f,  0.0407151f,  0.0678584f,  0.0950019f,
     0.1221453f,  0.1492888f,  0.1764324f,  0.2035761f,
     0.2307198f,  0.2578636f,  0.2850076f,  0.3121517f,
     0.3392960f,  0.3664405f,  0.3935852f,  0.4207301f,
     0.4478753f,  0.4750208f,  0.5021667f,  0.5293129f,
     0.5564596f,  0.5836067f,  0.6107544f,  0.6379027f,
     0.6650517f,  0.6922014f,  0.7193520f,  0.7465035f,
     0.7736561f,  0.8008098f,  0.8279648f,  0.8551212f,
     0.8822793f,  0.9094390f,  0.9366008f,  0.9637647f,
     0.9909310f,  1.0180999f,  1.0452718f,  1.0724469f,
     1.0996255f,  1.1268081f,  1.1539951f,  1.1811868f,
     1.2083838f,  1.2355865f,  1.2627956f,  1.2900118f,
     1.3172356f,  1.3444678f,  1.3717093f,  1.3989610f,
     1.4262237f,  1.4534986f,  1.4807869f,  1.5080897f,
     1.5354084f,  1.5627445f,  1.5900996f,  1.6174755f,
     1.6448739f,  1.6722970f,  1.6997469f,  1.7272261f,
     1.7547372f,  1.7822828f,  1.8098662f,  1.8374904f,
     1.8651592f,  1.8928763f,  1.9206458f,  1.9484722f,
     1.9763603f,  2.0043154f,  2.0323430f,  2.0604493f,
     2.0886408f,  2.1169245f,  2.1453081f,  2.1737998f,
     2.2024086f,  2.2311440f,  2.2600165f,  2.2890372f,
     2.3182184f,  2.3475732f,  2.3771157f,  2.4068614f,
     2.4368270f,  2.4670306f,  2.4974918f,  2.5282321f,
     2.5592748f,  2.5906452f,  2.6223710f,  2.6544825f,
     2.6870129f,  2.7199985f,  2.7534795f,  2.7874999f,
     2.8221086f,  2.8573596f,  2.8933131f,  2.9300362f,
     2.9676040f,  3.0061011f,  3.0456229f,  3.0862780f,
     3.1281899f,  3.1715011f,  3.2163758f,  3.2630056f,
     3.3116156f,  3.3624729f,  3.4158977f,  3.4722785f,
     3.5320936f,  3.5959415f,  3.6645851f,  3.7390194f,
     3.8205780f,  3.9111092f,  4.0132856f,  4.1311907f,
     4.2715508f,  4.4467193f,  4.6836997f,  5.0652659f,
};

// ---------------------------------------------------------------------------
// Inline dequant: look up byte index in the selected codebook, scale by norm.
// CODEBOOK_BITS is a runtime value from params (not compile-time template),
// so we use if-else. The Metal compiler will constant-fold if the value is
// known constant per-dispatch via a push-constant variant, but runtime is fine
// for correctness.
//
// packed_base: pointer to start of this position's byte-packed data [head_dim bytes]
// coord:       coordinate index (0..head_dim-1)
// scale_norm:  pre-multiplied scale (norm * inv_sqrt_dk for D=256, norm/sf for D=512)
// cbits:       codebook_bits field from params (5, 6, or 8)
// ---------------------------------------------------------------------------
inline float dequant_hb_single(
    device const uint8_t *packed_pos,
    uint coord,
    float scale_norm,
    uint cbits
) {
    uint idx = (uint)packed_pos[coord];
    float centroid;
    if (cbits == 5u) {
        centroid = CODEBOOK_HB_5BIT[idx & 0x1Fu];
    } else if (cbits == 6u) {
        centroid = CODEBOOK_HB_6BIT[idx & 0x3Fu];
    } else {
        centroid = CODEBOOK_HB_8BIT[idx];  // 8-bit: full byte
    }
    return centroid * scale_norm;
}

// Reconstruct float4 from 4 consecutive byte-packed elements.
// coord_base must be a multiple of 4.
//
// ADR-028 iter-195: vectorized byte load.  Replaces 4 sequential
// `packed_pos[coord+i]` reads with 1 uint32 load + 4 bit-shift+mask
// extracts.  Apple Metal coalesces a single 4-byte aligned uint load
// better than 4 separate 1-byte reads.  Also hoists the cbits branch
// out of the per-element loop (one branch decides for all 4 indices).
//
// Alignment requirement: caller must pass coord_base divisible by 4.
// All call sites in this kernel pass coord_base = (anything)*4 — verified.
inline float4 dequant_hb_float4(
    device const uint8_t *packed_pos,
    uint coord_base,
    float scale_norm,
    uint cbits
) {
    // Vectorized 4-byte load.  packed_pos + coord_base is 4-byte aligned
    // because (a) packed_pos is from MlxBuffer (≥16-byte aligned) and
    // (b) coord_base is always a multiple of 4 at every call site.
    uint k_packed = ((device const uint *)(packed_pos + coord_base))[0];
    uint idx0 = (k_packed >>  0) & 0xFFu;
    uint idx1 = (k_packed >>  8) & 0xFFu;
    uint idx2 = (k_packed >> 16) & 0xFFu;
    uint idx3 = (k_packed >> 24) & 0xFFu;

    float c0, c1, c2, c3;
    if (cbits == 5u) {
        c0 = CODEBOOK_HB_5BIT[idx0 & 0x1Fu];
        c1 = CODEBOOK_HB_5BIT[idx1 & 0x1Fu];
        c2 = CODEBOOK_HB_5BIT[idx2 & 0x1Fu];
        c3 = CODEBOOK_HB_5BIT[idx3 & 0x1Fu];
    } else if (cbits == 6u) {
        c0 = CODEBOOK_HB_6BIT[idx0 & 0x3Fu];
        c1 = CODEBOOK_HB_6BIT[idx1 & 0x3Fu];
        c2 = CODEBOOK_HB_6BIT[idx2 & 0x3Fu];
        c3 = CODEBOOK_HB_6BIT[idx3 & 0x3Fu];
    } else {
        // 8-bit: full byte index, no mask (matches dequant_hb_single).
        c0 = CODEBOOK_HB_8BIT[idx0];
        c1 = CODEBOOK_HB_8BIT[idx1];
        c2 = CODEBOOK_HB_8BIT[idx2];
        c3 = CODEBOOK_HB_8BIT[idx3];
    }
    return float4(c0, c1, c2, c3) * scale_norm;
}

// ---------------------------------------------------------------------------
// ADR-028 iter-106 FWHT-pre fusion helpers (Prong A of item #19).
// Inlined from fwht_standalone.metal so the FA kernel can apply Q rotation
// in-kernel and eliminate the 30-call FWHT-pre dispatch + its forced
// memory_barrier per layer (~1.44 ms/token = 9% of decode).
// Tables MUST match fwht_standalone.metal byte-for-byte.
// ---------------------------------------------------------------------------
constant uint8_t TBQ_SIGNS_256_FA[32] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
};
constant uint8_t TBQ_SIGNS_512_FA[64] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
};

inline void butterfly_local_fa(thread float &a, thread float &b) {
    float sum = a + b;
    float diff = a - b;
    a = sum;
    b = diff;
}

template<ushort EPT>
inline void fwht_simd_fa(thread float *elems, uint lane) {
    for (ushort h = 1; h < EPT; h <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            ushort partner = i ^ h;
            if (partner > i) {
                butterfly_local_fa(elems[i], elems[partner]);
            }
        }
    }
    for (ushort delta = 1; delta < 32; delta <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            float partner_val = simd_shuffle_xor(elems[i], delta);
            if (lane & delta) {
                elems[i] = partner_val - elems[i];
            } else {
                elems[i] = elems[i] + partner_val;
            }
        }
    }
}

// Runtime selector via params field (added in iter-106): 0 = caller-rotated Q
// (production default, byte-identical to pre-iter-106), 1 = kernel applies
// FWHT-pre internally. Branch is uniform across the WG (all threads see the
// same params.fuse_fwht_pre); the Metal compiler hoists it out of the
// per-thread loop so cost is ~zero runtime overhead.

// ---------------------------------------------------------------------------
// Main kernel: native HB (higher-bit) TQ flash attention vector.
//
// Same structure as flash_attn_vec_tq_impl but reads from byte-packed K/V.
// 5/6/8-bit controlled by params.codebook_bits at runtime.
//
// Norms layout:
//   D=256: [num_kv_heads, capacity]    f32 — 1 norm per position
//   D=512: [num_kv_heads, capacity, 2] f32 — 2 per-block norms per position
// ---------------------------------------------------------------------------
template<short DK, short DV>
kernel void flash_attn_vec_tq_hb_impl(
    constant FlashAttnVecTqHbParams  &params      [[buffer(0)]],
    device const float               *Q           [[buffer(1)]],
    device const uint8_t             *K_packed    [[buffer(2)]],  // byte-packed
    device const float               *K_norms     [[buffer(3)]],
    device const uint8_t             *V_packed    [[buffer(4)]],  // byte-packed
    device const float               *V_norms     [[buffer(5)]],
    device       float               *dst         [[buffer(6)]],
    threadgroup  half                *shmem       [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK / 4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV / 4;
    constexpr short SH  = 4 * C;  // 128 halfs = 64 floats

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");
    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    const uint NWG = params.nwg;
    const uint NSG = params.nsg;  // ADR-028 iter-127b Path D: simdgroups per workgroup
    const ushort iwg = tgpig[2] % NWG;
    const ushort iq2 = tgpig[1];  // head index
    const ushort iq1 = tgpig[0];  // query index (0 for decode)

    // GQA: map query head to KV head.
    const uint heads_per_kv = params.n_heads / params.n_kv_heads;
    const uint kv_head = iq2 / heads_per_kv;

    // Shared memory layout (ADR-028 iter-127b: NSG-aware banks).
    // Layout:
    //   [0, PK)                                                     — Q as half4 (shared by all simdgroups)
    //   [PK + sgitg*SH, PK + (sgitg+1)*SH)                          — per-simdgroup score scratch
    //   [PK + NSG*SH + sgitg*2*PV, PK + NSG*SH + (sgitg+1)*2*PV)    — per-simdgroup output accumulator
    //
    // At NSG=1, sgitg=0:
    //   ss = shmem + PK            (matches pre-iter-127 layout)
    //   so4 = shmem + PK + 1*SH    (matches pre-iter-127 layout)
    // — byte-identical to scaffold/pre-iter-127 dispatch.
    threadgroup half4  *sq4 = (threadgroup half4  *)(shmem);
    threadgroup float  *ss  = (threadgroup float  *)(shmem + PK + (uint)sgitg * SH);
    threadgroup float4 *so4 = (threadgroup float4 *)(shmem + PK + NSG * SH + (uint)sgitg * 2 * PV);

    // ADR-028 iter-106: Q-load split between two paths via FUSE_FWHT_PRE
    // function constant. Default path (caller-rotated) preserved unchanged;
    // fused path (kernel applies FWHT-pre internally) eliminates the
    // standalone fwht_sign_premult_f32 dispatch + its forced barrier.
    if (params.fuse_fwht_pre != 0u) {
        // Each thread loads EPT contiguous elements, applies sign-premult +
        // FWHT (simd-shuffle butterfly) + 1/sqrt(d) normalization, then
        // stores 2 half4 cells in the strided shared-memory layout the
        // K-loop expects. Matches fwht_sign_premult_fast<DK> byte-for-byte.
        constexpr ushort EPT = DK / 32;  // 8 for D=256, 16 for D=512
        const uint base = iq2 * DK + tiisg * EPT;
        float elems[EPT];
        for (ushort i = 0; i < EPT; i++) {
            elems[i] = Q[base + i];
        }
        // D1 sign pre-mult (BEFORE FWHT).
        for (ushort i = 0; i < EPT; i++) {
            ushort j = tiisg * EPT + i;
            uint8_t sign_byte = (DK == 256) ? TBQ_SIGNS_256_FA[j >> 3] : TBQ_SIGNS_512_FA[j >> 3];
            float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
            elems[i] *= sign_val;
        }
        // FWHT + normalize.
        fwht_simd_fa<EPT>(elems, (uint)tiisg);
        const float inv_sqrt_d = rsqrt(float(DK));
        for (ushort i = 0; i < EPT; i++) {
            elems[i] *= inv_sqrt_d;
        }
        // Store as half4 in strided layout: thread tiisg writes sq4 indices
        // [tiisg * (EPT/4), tiisg * (EPT/4) + 1, ...]. For EPT=8 that's 2
        // contiguous cells per thread covering sq4[0..63] for D=256.
        constexpr ushort SQ4_PER_THREAD = EPT / 4;
        for (ushort q = 0; q < SQ4_PER_THREAD; q++) {
            ushort sq_idx = tiisg * SQ4_PER_THREAD + q;
            sq4[sq_idx] = half4(elems[q*4 + 0], elems[q*4 + 1],
                                elems[q*4 + 2], elems[q*4 + 3]);
        }
        // Zero-pad if PK4 > DK4 (only for non-power-of-2 DK; not hit at
        // DK=256 or DK=512 today, but guard preserved for future shapes).
        for (ushort i = tiisg + DK4; i < PK4; i += NW) {
            sq4[i] = half4(0.0h);
        }
    } else {
        // Caller-rotated path (production default — Q already FWHT'd).
        for (ushort i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                float4 qval = *((device const float4 *)(Q + iq2 * DK + i * 4));
                sq4[i] = half4(qval);
            } else {
                sq4[i] = half4(0.0h);
            }
        }
    }

    // Zero output accumulator.
    so4 += tiisg;
    for (short i = 0; i < DV4 / NL; ++i) {
        so4[i * NL] = float4(0.0f);
    }

    // Zero scratch buffer.
    for (ushort i = tiisg; i < SH / 4; i += NW) {
        ((threadgroup float *)(shmem + PK))[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state.
    float S = 0.0f;
    float M = -FLT_MAX / 2;

    const ushort tx = tiisg;
    const uint kv_seq_len = params.kv_seq_len;
    const uint kv_capacity = params.kv_capacity;
    const uint ring_start = params.ring_start;
    const uint cbits = params.codebook_bits;
    const float sf_d512 = params.scale_factor_d512;
    const bool is_d512 = (DK > 256);

    uint window_start_logical = 0;
    if (params.mask_type == 2 && params.sliding_window > 0 && kv_seq_len > params.sliding_window) {
        window_start_logical = kv_seq_len - params.sliding_window;
    }

    threadgroup const half4 *pq4 = sq4 + tx;

    // Main loop over KV cache in chunks of C=32.
    // ADR-028 iter-127b: NSG-axis K-stride. Each simdgroup `sgitg` within
    // workgroup `iwg` strides through K with step `NWG*NSG`. Matches
    // llama.cpp's flash_attn_vec_ext at ggml-metal.metal:6782.
    // At NSG=1 (sgitg always 0): `for (ic0 = iwg; ; ic0 += NWG)` — identical
    // to pre-iter-127 behavior.
    for (uint ic0 = iwg * NSG + (uint)sgitg; ; ic0 += NWG * NSG) {
        uint ic = ic0 * C;
        if (ic >= kv_seq_len) break;

        // Compute mask for this chunk.
        {
            uint k_pos = ic + tx;
            float mask_val = 0.0f;
            if (k_pos >= kv_seq_len) {
                mask_val = -65504.0f;
            } else {
                uint logical_idx = (k_pos - ring_start + kv_capacity) % kv_capacity;
                if (logical_idx >= kv_seq_len || logical_idx < window_start_logical) {
                    mask_val = -65504.0f;
                }
            }
            ss[tx] = mask_val;
        }

        if (simd_max(ss[tiisg]) <= -65504.0f) continue;

        // ---- Q * K^T ----
        {
            float mqk[C];
            const float inv_sqrt_dk = rsqrt(float(DK));

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) {
                    mqk[cc] = 0.0f;
                    continue;
                }

                // Dequant scale for K.
                float k_sn;
                if (is_d512) {
                    // D=512: per-block norms; block 0 = coords 0..255, block 1 = 256..511
                    // For K*Q^T we need both blocks. The dot product spans all DK coords.
                    // We compute the block-0 portion and block-1 portion separately,
                    // each with their own scale_norm.
                    // norm_base points to: [kv_head, kv_pos, 0..2] f32
                    device const float *knorm = K_norms + (kv_head * kv_capacity + kv_pos) * 2u;
                    // k_sn unused in this branch — handled in the inner loop below
                    (void)k_sn;
                    (void)inv_sqrt_dk;

                    device const uint8_t *k_base =
                        K_packed + (kv_head * kv_capacity + kv_pos) * DK;

                    float partial = 0.0f;
                    // Block 0: coords 0..255
                    // Each thread tx covers elements (tx + ii*NL)*4 .. (tx + ii*NL)*4+3
                    // for ii in [0..(DK/2)/4/NL). This mirrors the D=256 striding pattern.
                    {
                        float sn0 = knorm[0] / sf_d512;
                        for (short ii = 0; ii < (DK/2) / 4 / NL; ++ii) {
                            uint coord = (uint)(tx + ii * NL) * 4u;
                            float4 k_val = dequant_hb_float4(k_base, coord, sn0, cbits);
                            partial += dot(k_val, float4(pq4[ii * NL]));
                        }
                    }
                    // Block 1: coords 256..511
                    {
                        float sn1 = knorm[1] / sf_d512;
                        const uint blk1_start = DK / 2;
                        for (short ii = 0; ii < (DK/2) / 4 / NL; ++ii) {
                            uint coord = blk1_start + (uint)(tx + ii * NL) * 4u;
                            float4 k_val = dequant_hb_float4(k_base, coord, sn1, cbits);
                            partial += dot(k_val, float4(pq4[(DK4/2/NL + ii) * NL]));
                        }
                    }
                    mqk[cc] = simd_sum(partial);
                } else {
                    // D=256: single norm per position.
                    float k_norm_val = K_norms[kv_head * kv_capacity + kv_pos];
                    k_sn = k_norm_val * inv_sqrt_dk;

                    device const uint8_t *k_base =
                        K_packed + (kv_head * kv_capacity + kv_pos) * DK + tx * 4u;

                    float partial = 0.0f;
                    for (short ii = 0; ii < DK4 / NL; ++ii) {
                        float4 k_val = dequant_hb_float4(k_base, (uint)(ii * NL) * 4u, k_sn, cbits);
                        partial += dot(k_val, float4(pq4[ii * NL]));
                    }
                    mqk[cc] = simd_sum(partial);
                }
            }

            ss[tx] = fma(mqk[tx], params.scale, ss[tx]);
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Online softmax ----
        {
            const float m_old = M;
            const float s_new = ss[tiisg];
            M = simd_max(max(M, s_new));
            const float ms = exp(m_old - M);
            const float vs = exp(s_new - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] *= ms;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O + softmax_weights * V ----
        {
            float4 lo[DV4 / NL];
            for (short ii = 0; ii < DV4 / NL; ++ii) lo[ii] = float4(0.0f);

            const float inv_sqrt_dv = rsqrt(float(DV));

            for (short cc = 0; cc < C; ++cc) {
                uint kv_pos = ic + cc;
                if (kv_pos >= kv_seq_len) continue;

                if (is_d512) {
                    device const float *vnorm = V_norms + (kv_head * kv_capacity + kv_pos) * 2u;
                    device const uint8_t *v_base =
                        V_packed + (kv_head * kv_capacity + kv_pos) * DV;
                    float w = ss[cc];

                    // Block 0: coords 0..255
                    // Same striding pattern as D=256 and K D=512 above.
                    float sn0 = vnorm[0] / sf_d512 * w;
                    for (short ii = 0; ii < (DV/2) / 4 / NL; ++ii) {
                        uint coord = (uint)(tx + ii * NL) * 4u;
                        lo[ii] += dequant_hb_float4(v_base, coord, sn0, cbits);
                    }
                    // Block 1: coords 256..511
                    float sn1 = vnorm[1] / sf_d512 * w;
                    for (short ii = 0; ii < (DV/2) / 4 / NL; ++ii) {
                        uint coord = (uint)(DV/2) + (uint)(tx + ii * NL) * 4u;
                        lo[DV4/2/NL + ii] += dequant_hb_float4(v_base, coord, sn1, cbits);
                    }
                } else {
                    float v_norm_val = V_norms[kv_head * kv_capacity + kv_pos];
                    float v_sw = v_norm_val * inv_sqrt_dv * ss[cc];
                    device const uint8_t *v_base =
                        V_packed + (kv_head * kv_capacity + kv_pos) * DV + tx * 4u;

                    for (short ii = 0; ii < DV4 / NL; ++ii) {
                        lo[ii] += dequant_hb_float4(v_base, (uint)(ii * NL) * 4u, v_sw, cbits);
                    }
                }
            }

            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] += lo[ii];
            }
        }
    }

    // Store M and S for the reduce kernel (each simdgroup writes to its own bank).
    if (tiisg == 0) {
        ss[0] = S;
        ss[1] = M;
    }

    so4 -= tiisg;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Cross-simdgroup online-softmax reduce (ADR-028 iter-127c Path D) ----
    //
    // At NSG=1: skipped — sgitg=0 has the only (S, M, so), write proceeds.
    // At NSG>1: simdgroup 0 reads all NSG banks of (S_j, M_j, so_j), computes
    //   M_global = max(M_j)
    //   ms_j     = exp(M_j - M_global)
    //   S_total  = Σ S_j * ms_j
    //   so_total = Σ so_j * ms_j
    // Then overwrites simdgroup 0's bank (S, M, so4) with the merged values.
    // Existing per-WG write below uses the merged values.
    //
    // NSG_MAX=4 to bound the per-thread `ms_arr` static array (matches
    // llama.cpp's policy `nsg ∈ {1, 2, 4}` capped at 4).
    if (NSG > 1u && sgitg == 0) {
        constexpr ushort NSG_MAX = 4;
        float ms_arr[NSG_MAX];
        float M_global = -FLT_MAX / 2;
        // Pass 1: compute M_global across NSG simdgroups.
        for (ushort j = 0; j < NSG; ++j) {
            threadgroup const float *ssj = (threadgroup const float *)(shmem + PK + (uint)j * SH);
            M_global = max(M_global, ssj[1]);
        }
        // Pass 2: compute per-simdgroup rescale + accumulate S_total.
        float S_total = 0.0f;
        for (ushort j = 0; j < NSG; ++j) {
            threadgroup const float *ssj = (threadgroup const float *)(shmem + PK + (uint)j * SH);
            const float M_j = ssj[1];
            const float S_j = ssj[0];
            ms_arr[j] = exp(M_j - M_global);
            S_total += S_j * ms_arr[j];
        }
        // Pass 3: accumulate so banks into simdgroup 0's so4. Each thread of
        // simdgroup 0 strides DV4 with step NW=32 (matches the existing write
        // loop pattern below).
        for (ushort i = tiisg; i < DV4; i += NW) {
            float4 acc = float4(0.0f);
            for (ushort j = 0; j < NSG; ++j) {
                threadgroup const float4 *so4_j = (threadgroup const float4 *)(shmem + PK + NSG * SH + (uint)j * 2u * PV);
                acc += so4_j[i] * ms_arr[j];
            }
            so4[i] = acc;
        }
        // Update local S, M scalars for the write logic below. Only thread 0
        // commits to ss[0..2]; sgitg==0 already gates this whole block.
        if (tiisg == 0) {
            ss[0] = S_total;
            ss[1] = M_global;
        }
        S = S_total;
        M = M_global;
        // No barrier needed — only simdgroup 0 reads so4 below.
    }

    // ---- Write output ----
    if (sgitg == 0) {
        const int64_t nrows = params.n_heads;
        const int64_t rid = iq2 + (int64_t)iq1 * params.n_heads;
        const uint NWG_val = params.nwg;
        const float inv_S = (NWG_val == 1) ? ((S == 0.0f) ? 0.0f : 1.0f / S) : 1.0f;

        device float4 *dst4 = (device float4 *)dst;
        for (ushort i = tiisg; i < DV4; i += NW) {
            dst4[rid * DV4 * NWG_val + NWG_val * i + iwg] = so4[i] * inv_S;
        }

        if (NWG_val > 1 && tiisg == 0) {
            device float *dst1 = (device float *)dst + nrows * DV * NWG_val;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 0] = S;
            dst1[rid * (2 * NWG_val) + 2 * iwg + 1] = M;
        }
    }
}

// --------------------------------------------------------------------------
// Kernel instantiations
// --------------------------------------------------------------------------

typedef decltype(flash_attn_vec_tq_hb_impl<256, 256>) flash_attn_vec_tq_hb_t;

template [[host_name("flash_attn_vec_tq_hb_dk256")]]
kernel flash_attn_vec_tq_hb_t flash_attn_vec_tq_hb_impl<256, 256>;

template [[host_name("flash_attn_vec_tq_hb_dk512")]]
kernel flash_attn_vec_tq_hb_t flash_attn_vec_tq_hb_impl<512, 512>;
