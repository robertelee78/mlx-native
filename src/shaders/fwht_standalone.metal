// fwht_standalone.metal — Standalone FWHT using SIMD shuffle (zero threadgroup barriers)
//
// Pre-rotates Q before TurboQuant SDPA and inverse-rotates the output.
// FWHT is self-inverse (H = H^{-1} for normalized Hadamard).
//
// Architecture: 1 simdgroup (32 threads) per head.
// Each thread holds head_dim/32 elements in registers.
// All butterfly stages use local ops or simd_shuffle_xor — zero barriers.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

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
