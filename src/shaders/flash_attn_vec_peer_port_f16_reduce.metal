// flash_attn_vec_peer_port_f16_reduce.metal — verbatim port of
// llama.cpp's kernel_flash_attn_ext_vec_reduce (ggml-metal.metal lines 7232-7275)
// for DV=256, NWG=32. Used after the NWG=32 vec kernel to combine partial
// results from each workgroup into the final dst.
//
// ADR-029 iter-134 (cfa/fa-peer-port-nwg32 follow-up).
//
// Hypothesis (refined from iter-127/132): peer's flash-attn-vec uses NWG=32 +
// reduce kernel by default (ggml-metal-ops.cpp:2944 — `if (false)` disables
// NWG=1 path). The original CFA port (iter-126) targeted the unused NWG=1 dead
// code path, which falsified at tg5000 (-25%). This file is part of the proper
// NWG=32 + reduce-kernel port.
//
// Surface adaptations only:
//   (a) args struct → FlashAttnVecPeerPortReduceParams (just nrows)
//   (b) DV/NWG baked as file-header #define matching peer's FC-bake
//   (c) Buffer slots: 0=params, 1=htmp (workgroup partials, float*), 2=dst (float*)
// Kernel body VERBATIM from peer 7235-7275.

#include <metal_stdlib>
using namespace metal;

// FC-bake constants (peer specializes via function constants; we bake symbolically per RULE-1 of cfa spec)
#define DV  256
#define NWG 32

// Params struct
struct FlashAttnVecPeerPortReduceParams {
    int32_t nrows;
};

kernel void flash_attn_vec_peer_port_f16_reduce_dv256_nwg32(
        constant FlashAttnVecPeerPortReduceParams & args [[buffer(0)]],
        device   const char *                       htmp [[buffer(1)]],
        device         char *                       dst  [[buffer(2)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }
}
