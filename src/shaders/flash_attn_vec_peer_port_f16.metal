// flash_attn_vec_peer_port_f16.metal — Verbatim port of llama.cpp kernel_flash_attn_ext_vec
// for f16-K / f16-V, DK=DV=256, NWG=1, NSG=1, NE=1.
//
// ADR-029 CFA cfa-20260512-fa-peer-port (iter-125 redo).
// Peer source: /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal lines 6666-7096.
// Hypothesis: verbatim peer source body produces peer-equivalent PSO from Apple compiler.
//
// Surface adaptations only (RULE-1):
//   (a) args struct → FlashAttnVecPeerPortParams; args.* field accesses → params.*
//       nb11/nb21 (byte strides) inlined as DK*2/DV*2; ne11→kv_seq_len;
//       ne01=1(decode); GQA via num_heads/num_kv_heads.
//   (b) pm[ic+tiisg] external mask load → inline ring-buffer sliding-window compute
//       writing to same slot sm[tiisg]; skip block verbatim.
//   (c) Buffer slots: 0=params, 1=Q(float*), 2=K_f16(half*), 3=V_f16(half*), 4=dst(float*)
//       k/v buffers typed as half* so byte arithmetic converted to element arithmetic.
//   (d) FC flags baked via file-header constexpr/defines (RULE-1: preserve symbolic names
//       in live expressions): NWG=1, NSG=1, NE=1, has_mask=1(inline), has_sinks=0,
//       has_bias=0, has_scap=0, has_kvpad=0. Unreachable branches physically deleted;
//       symbolic names kept in live expressions per standing rule.
// Kernel body VERBATIM otherwise: loop structure, FOR_UNROLL, simd_shuffle_down
// ladder, online-softmax, V-loop, store formula all unchanged.

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32
#define PAD2(x, n)  (((x) + (n) - 1) & ~((n) - 1))
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
// MAXHALF: Metal stdlib defines __HALF_MAX__ (= 65504.0h) in metal_types.h.
// Use it directly to avoid redefinition warning.
#define MAXHALF __HALF_MAX__

// FC-bake constants — preserve symbolic names in live expressions per RULE-1.
// MSL does not allow program-scope constexpr short; use #define macros so the body
// source remains lexically identical to peer's symbolic references.
#define NWG 1
#define NSG 1
#define nl_k 1      // peer FA_TYPES for f16/f16: nl_k template param = 1
#define nl_v 1      // peer FA_TYPES for f16/f16: nl_v template param = 1
#define FC_flash_attn_ext_vec_has_mask  1
#define FC_flash_attn_ext_vec_has_sinks 0
#define FC_flash_attn_ext_vec_has_bias  0
#define FC_flash_attn_ext_vec_has_scap  0
#define FC_flash_attn_ext_vec_has_kvpad 0

// Peer NS10/NS20 element-strides (peer FA_TYPES: NS10=nb11/nb10, NS20=nb21/nb20).
// For f16-K + f16-V at DK=DV=256: NS10 = DK, NS20 = DV. Preserved symbolically per RULE-1.
#define NS10 DK
#define NS20 DV

// No-op stub helpers for dead else-branches under is_same<kd4_t,k4_t> guards.
// The compiler DCEs these when the f16 fast-path branch is taken (is_same resolves true).
// Stubs keep the call sites lexically valid so the source remains structurally identical to peer.
template <typename T> inline void deq_k_t4(device const T*, short, thread half4& out) { out = half4(0); }
template <typename T> inline void deq_v_t4(device const T*, short, thread half4& out) { out = half4(0); }

// FA_TYPES expansion for f16/f16 (peer ggml-metal.metal line 7101-7107):
//   q_t=half4, k_t=half4, v_t=half4, qk_t=float, s_t=float, s4_t=float4, o4_t=float4.
// kd4_t=k4_t=half4, vd4_t=v4_t=half4 (peer: kd4_t is the dequant type, equal to k4_t
// for F16 → is_same<kd4_t,k4_t>::value is true at compile time).
typedef half4  q4_t;
typedef half4  k4_t;
typedef half4  kd4_t;
typedef half4  v4_t;
typedef half4  vd4_t;
typedef float  qk_t;
typedef float  s_t;
typedef float4 s4_t;
typedef float4 o4_t;

// is_same<T,U>::value — peer uses this construct verbatim (peer ggml-metal.metal uses
// its own definition; Metal stdlib provides metal::is_same via <metal_stdlib> +
// `using namespace metal`. We use the stdlib version directly — equivalent semantics,
// avoids ambiguity with the global-scope redeclaration that conflicts in Metal 32023+.)
// No local redefinition needed: metal::is_same<T,U>::value is already in scope.

// Params struct — GPU layout matches FlashAttnVecPeerPortParamsGpu in Rust dispatcher.
// 9 fields × 4 bytes = 36 bytes.
struct FlashAttnVecPeerPortParams {
    uint  num_heads;
    uint  num_kv_heads;
    uint  head_dim;
    uint  kv_seq_len;
    uint  kv_capacity;
    float scale;
    uint  mask_type;
    uint  sliding_window;
    uint  ring_start;
};

kernel void flash_attn_vec_peer_port_f16_dk256_dv256(
        constant FlashAttnVecPeerPortParams & params                [[buffer(0)]],
        device const float                  * q                    [[buffer(1)]],
        device const half                   * k                    [[buffer(2)]],
        device const half                   * v                    [[buffer(3)]],
        device       float                  * dst                  [[buffer(4)]],
        threadgroup  half                   * shmem_f16            [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {

    // DK=DV=256 baked (instantiation scope).
    constexpr short DK = 256;
    constexpr short DV = 256;

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

    // Peer line 6688: iwg = tgpig[2]%NWG. NWG constexpr=1 at file header.
    const short iwg = tgpig[2]%NWG;

    // Peer lines 6690-6692.
    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);   // = 256
    constexpr short PK4 = PK/4;            // = 64

    constexpr short PV  = PAD2(DV, 128);   // = 256
    constexpr short PV4 = PV/4;            // = 64

    constexpr short NW  = N_SIMDWIDTH;     // = 32
    constexpr short NE  = 1;               // baked (NE_FC at file header; NE local for body expressions)
    constexpr short NL  = NW/NE;           // = 32
    constexpr short SH  = 4*C;             // = 128 (shared memory per simdgroup)

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    // Shared memory layout — verbatim peer lines 6713-6717, with NSG constexpr=1.
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    // store the result for all queries in shared memory (the O matrix from the paper)
    // verbatim peer line 6720
    so4 += tiisg;

    {
        // Adaptation (a): advance base pointers for this head/batch position.
        // Peer lines 6722-6729 do byte-pointer arithmetic via nb* stride fields.
        // Our buffers are typed (float*, half*) so we do element-count arithmetic.
        //
        // Peer: q += iq1*nb01 + iq2*nb02 + iq3*nb03
        //   For decode: iq1=0, nb01=DK*4. iq2 stride=num_heads*DK*4 for batched,
        //   but our Q is [n_heads, head_dim] shaped → stride per head = DK.
        //   iq3=0 (batch=1). So: q offset = iq2*DK elements (floats).
        q += (uint)iq2 * DK;

        // Peer lines 6725-6726: ikv2 = iq2/(ne02/ne_12_2) = iq2/(num_heads/num_kv_heads)
        const short ikv2 = (short)iq2 / (short)(params.num_heads / params.num_kv_heads);
        // ikv3: iq3/(ne03/ne_12_3) = 0 for batch=1.

        // Peer lines 6728-6729: k += ikv2*nb12 + ikv3*nb13; v += ikv2*nb22 + ikv3*nb23
        // nb12 = kv_capacity*DK*sizeof(half) bytes; our k is half* so offset = ikv2*kv_capacity*DK halfs.
        k += (uint)ikv2 * params.kv_capacity * NS10;
        v += (uint)ikv2 * params.kv_capacity * NS20;
    }

    // load heads from Q to shared memory — verbatim peer lines 6733-6743.
    device const float4 * q4 = (device const float4 *) ((device const char *) q);

    if (iq1 < 1u) {   // args.ne01=1 for single-query decode
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                sq4[i] = (q4_t) q4[i];
            } else {
                sq4[i] = (q4_t) 0.0f;
            }
        }
    }

    // zero out so — verbatim peer lines 6746-6748
    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    // zero out shared memory SH — verbatim peer lines 6751-6753
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        // thread indices inside the simdgroup — verbatim peer lines 6762-6763
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // Peer line 6768: slope=1.0f (has_bias=0 baked).
        float slope = 1.0f;
        (void)slope;  // unused at has_bias=0; kept for structural parity

        // Sliding-window ring-buffer state for inline mask (adaptation b).
        // Mirrors flash_attn_vec_hybrid.metal:489-491.
        uint window_start_logical = 0u;
        if (params.mask_type == 2u && params.sliding_window > 0u &&
            params.kv_seq_len > params.sliding_window) {
            window_start_logical = params.kv_seq_len - params.sliding_window;
        }

        // loop over the KV cache — verbatim peer line 6782.
        // NWG/NSG/iwg/sgitg all resolve from file-header constexpr declarations.
        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= (int)params.kv_seq_len) {   // args.ne11 = kv_seq_len
                break;
            }

            // has_kvpad=0 baked: kvpad branch physically deleted.

            // Adaptation (b): inline mask writing to sm[tiisg].
            // Replaces peer lines 6814-6816 (has_mask=1 external load).
            // Sliding-window ring-buffer logic from flash_attn_vec_hybrid.metal:506-519.
            if (FC_flash_attn_ext_vec_has_mask) {
                uint k_pos = (uint)ic + (uint)tiisg;
                half mask_val = (half)0.0f;
                if (k_pos >= params.kv_seq_len) {
                    mask_val = -MAXHALF;
                } else {
                    uint logical_idx = (k_pos - params.ring_start + params.kv_capacity)
                                       % params.kv_capacity;
                    if (logical_idx >= params.kv_seq_len ||
                        logical_idx < window_start_logical) {
                        mask_val = -MAXHALF;
                    }
                }
                sm[tiisg] = mask_val;
            }

            // skip -INF blocks — verbatim peer line 6819
            if (simd_max(sm[tiisg]) <= -MAXHALF) {
                continue;
            }

            {
                device      const k4_t * pk4 = (device const k4_t *) (k + (uint)ic*NS10);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [0 ... C/NE - 1] = 0.0f };

                // each simdgroup processes 1 query and NE (NW/NL = 1) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 + ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((uint)(ic + NE*cc + ty)*NS10));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        // simdgroup reduce
                        // [ 0 ..  7] -> [ 0]
                        // [ 8 .. 15] -> [ 8]
                        // [16 .. 23] -> [16]
                        // [24 .. 31] -> [24]
                        if (NE <= 1) {
                            mqk[cc] += simd_shuffle_down(mqk[cc], 16);
                        }
                        if (NE <= 2) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
                        }
                        if (NE <= 4) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
                        }
                        if (NE <= 8) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
                        }
                        if (NE <= 16) {
                            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
                        }

                        // broadcast
                        mqk[cc] = simd_shuffle(mqk[cc], NL*ty);
                    }
                }

                if (FC_flash_attn_ext_vec_has_mask &&
                   !FC_flash_attn_ext_vec_has_scap &&
                   !FC_flash_attn_ext_vec_has_bias) {
                    ss[NE*tx + ty] = fma(mqk[tx], params.scale, (qk_t) sm[NE*tx + ty]);
                } else {
                    mqk[tx] *= params.scale;

                    if (FC_flash_attn_ext_vec_has_scap) {
                        mqk[tx] = params.scale*precise::tanh(mqk[tx]);
                    }

                    if (FC_flash_attn_ext_vec_has_bias) {
                        mqk[tx] += (qk_t) sm[NE*tx + ty]*slope;
                    } else {
                        mqk[tx] += (qk_t) sm[NE*tx + ty];
                    }

                    ss[NE*tx + ty] = mqk[tx];
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                // the P matrix from the paper (Q rows, C columns)
                ss[tiisg] = vs;

                // O = diag(ms)*O
                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + (uint)ic*NS20);

                    pv4 += ty*NS20/4 + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((uint)(ic + NE*cc + ty)*NS20));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            lo[ii] += o4_t(float4(mv)*float4(ss[NE*cc + ty]));
                        }
                    }
                }

                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0], 16);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1], 16);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2], 16);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3], 16);
                    }

                    if (NE > 2) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  8);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  8);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  8);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  8);
                    }

                    if (NE > 4) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  4);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  4);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  4);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  4);
                    }

                    if (NE > 8) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  2);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  2);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  2);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  2);
                    }

                    if (NE > 16) {
                        lo[ii][0] += simd_shuffle_down(lo[ii][0],  1);
                        lo[ii][1] += simd_shuffle_down(lo[ii][1],  1);
                        lo[ii][2] += simd_shuffle_down(lo[ii][2],  1);
                        lo[ii][3] += simd_shuffle_down(lo[ii][3],  1);
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        // has_sinks=0 baked: sinks block physically deleted.

        // these are needed for reducing the results from the simdgroups — verbatim peer lines 7028-7031
        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce block (peer lines 7039-7066) PHYSICALLY OMITTED per spec notes 1 + invariant 9:
    // NSG=1 makes the block dead code; spec requires physical deletion, not dead-code retention.
    // After deletion, kernel goes directly to the final rescale + store block.

    // final rescale with 1/S and store to global memory — verbatim peer lines 7069-7090
    if (sgitg == 0) {
        // nrows = ne3*ne2*ne1 = 1*num_heads*1 for decode.
        const int64_t nrows = params.num_heads;
        // rid = iq3*ne2*ne1 + iq2 + iq1*ne1 = iq2 for decode (iq1=0, iq3=0, ne1=1).
        const int64_t rid   = iq3*params.num_heads*1 + iq2 + iq1*1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        // verbatim peer line 7076: NWG constexpr=1 at file header; expression preserved.
        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

        // interleave the workgroup data — verbatim peer line 7080.
        // NWG/iwg constexpr at file header; expression preserved per spec store_lines.
        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i]*S;
        }

        // store S and M — verbatim peer lines 7084-7089; NWG constexpr=1 → compiler DCEs.
        if (NWG > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }
}
