// fwht_standalone.metal — Standalone FWHT using SIMD shuffle (zero threadgroup barriers)
//
// Pre-rotates Q before TurboQuant SDPA and inverse-rotates the output.
// FWHT is self-inverse (H = H^{-1} for normalized Hadamard).
//
// Architecture: 1 simdgroup (32 threads) per head.
// Each thread holds head_dim/32 elements in registers.
// All butterfly stages use local ops or simd_shuffle_xor — zero barriers.
//
// D1 SRHT variants (ADR-007 iter-14):
// - fwht_sign_premult_f32_d256/d512: apply sign BEFORE FWHT (Q pre-rotation with D1).
// - fwht_sign_undo_f32_d256/d512: apply sign AFTER FWHT (output inverse FWHT + sign undo).
// Sign tables verbatim from AmesianX cpy-utils.cuh:158-163 (D=256) + :211-220 (D=512).

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// D1 sign table D=256 (32 bytes). sha256=3ef1038e6c232e9519101daa2d6efd637d4c6bfdb29f4ee7101625c39d0ddc89
// Bit j = (table[j>>3] >> (j&7)) & 1; bit=1 → sign=-1, bit=0 → sign=+1.
constant uint8_t TBQ_SIGNS_256[32] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
};

// D1 sign table D=512 (64 bytes). sha256=44f13ce9f6db1edac62f558ee054f9de29cd474fd051362cadcaa98a55745f17
constant uint8_t TBQ_SIGNS_512[64] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
};

struct FwhtParams {
    uint head_dim;
    uint num_heads;
};

// Butterfly operation on a local element pair.
inline void butterfly_local(thread float &a, thread float &b) {
    float sum = a + b;
    float diff = a - b;
    a = sum;
    b = diff;
}

// FWHT via SIMD shuffle — zero barriers.
template<ushort EPT>
inline void fwht_simd(thread float *elems, uint lane) {
    // Local stages (h < EPT)
    for (ushort h = 1; h < EPT; h <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            ushort partner = i ^ h;
            if (partner > i) {
                butterfly_local(elems[i], elems[partner]);
            }
        }
    }

    // Cross-thread stages via simd_shuffle_xor
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

template<ushort HEAD_DIM>
kernel void fwht_standalone_fast(
    device float           *data   [[buffer(0)]],
    constant FwhtParams    &params [[buffer(1)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgid;
    const uint lane = tiisg;

    if (head_idx >= params.num_heads) return;

    // Load elements into registers.
    const uint base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = data[base + i];
    }

    // FWHT via SIMD shuffle.
    fwht_simd<EPT>(elems, lane);

    // Normalize by 1/sqrt(head_dim) and write back.
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        data[base + i] = elems[i] * inv_sqrt_d;
    }
}

// Instantiations.
template [[host_name("fwht_standalone_f32_d256")]]
kernel void fwht_standalone_fast<256>(device float *, constant FwhtParams &, uint, uint);

template [[host_name("fwht_standalone_f32_d512")]]
kernel void fwht_standalone_fast<512>(device float *, constant FwhtParams &, uint, uint);

// ---------------------------------------------------------------------------
// D1 SRHT variant: apply sign BEFORE FWHT.
// Used for Q pre-rotation: sign * Q → FWHT → normalize.
// Mirrors AmesianX cpy-utils.cuh:180-183 sign application pattern.
// ---------------------------------------------------------------------------
template<ushort HEAD_DIM>
kernel void fwht_sign_premult_fast(
    device float           *data   [[buffer(0)]],
    constant FwhtParams    &params [[buffer(1)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgid;
    const uint lane = tiisg;

    if (head_idx >= params.num_heads) return;

    const uint base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = data[base + i];
    }

    // Apply D1 sign BEFORE FWHT (sign is self-inverse).
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = (HEAD_DIM == 256) ? TBQ_SIGNS_256[j >> 3] : TBQ_SIGNS_512[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    // FWHT + normalize.
    fwht_simd<EPT>(elems, lane);
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        data[base + i] = elems[i] * inv_sqrt_d;
    }
}

template [[host_name("fwht_sign_premult_f32_d256")]]
kernel void fwht_sign_premult_fast<256>(device float *, constant FwhtParams &, uint, uint);

template [[host_name("fwht_sign_premult_f32_d512")]]
kernel void fwht_sign_premult_fast<512>(device float *, constant FwhtParams &, uint, uint);

// ---------------------------------------------------------------------------
// D1 SRHT variant: apply sign AFTER FWHT (inverse rotation + sign undo).
// Used for output: FWHT (= IWHT for normalized H) → normalize → sign undo.
// After IWHT, output element j is sign_j * V_weighted_j; sign undo recovers V_weighted_j.
// ---------------------------------------------------------------------------
template<ushort HEAD_DIM>
kernel void fwht_sign_undo_fast(
    device float           *data   [[buffer(0)]],
    constant FwhtParams    &params [[buffer(1)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgid;
    const uint lane = tiisg;

    if (head_idx >= params.num_heads) return;

    const uint base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = data[base + i];
    }

    // FWHT (= IWHT for normalized H) + normalize.
    fwht_simd<EPT>(elems, lane);
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= inv_sqrt_d;
    }

    // Apply D1 sign AFTER FWHT to undo encode-time sign pre-mult.
    // sign is self-inverse: sign * (sign * V) = V.
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = (HEAD_DIM == 256) ? TBQ_SIGNS_256[j >> 3] : TBQ_SIGNS_512[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    for (ushort i = 0; i < EPT; i++) {
        data[base + i] = elems[i];
    }
}

template [[host_name("fwht_sign_undo_f32_d256")]]
kernel void fwht_sign_undo_fast<256>(device float *, constant FwhtParams &, uint, uint);

template [[host_name("fwht_sign_undo_f32_d512")]]
kernel void fwht_sign_undo_fast<512>(device float *, constant FwhtParams &, uint, uint);
