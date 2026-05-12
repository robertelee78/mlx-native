// flash_attn_vec_peer_port_f16.metal — Verbatim port of llama.cpp kernel_flash_attn_ext_vec
// for f16-K / f16-V, DK=DV=256, NWG=1, NSG=1, NE=1.
//
// ADR-029 CFA cfa-20260512-fa-peer-port (iter-122).
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
//   (d) FC flags baked: NWG=1, NSG=1, NE=1, has_mask=1(inline), has_sinks=0,
//       has_bias=0, has_scap=0, has_kvpad=0 — unreachable branches physically deleted.
// Kernel body VERBATIM otherwise: loop structure, FOR_UNROLL, simd_shuffle_down
// ladder, online-softmax, V-loop, store formula all unchanged.

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32
#define PAD2(x, n)  (((x) + (n) - 1) & ~((n) - 1))
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define MAXHALF 65504.0h

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

// is_same<T,U>::value — compile-time type equality (used verbatim in peer body).
template<typename T, typename U> struct is_same       { static constexpr bool value = false; };
template<typename T>             struct is_same<T, T> { static constexpr bool value = true;  };

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

    // NWG=1, NSG=1, NE=1 baked in (FC flags).
    // Peer line 6688: iwg = tgpig[2]%NWG — at NWG=1, iwg=0.
    const short iwg = 0;

    // Peer lines 6690-6692.
    const ushort iq3 = tgpig[2]; // /NWG = /1
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);   // = 256
    constexpr short PK4 = PK/4;            // = 64

    constexpr short PV  = PAD2(DV, 128);   // = 256
    constexpr short PV4 = PV/4;            // = 64

    constexpr short NW  = N_SIMDWIDTH;     // = 32
    constexpr short NE  = 1;              // baked
    constexpr short NL  = NW/NE;          // = 32
    constexpr short SH  = 4*C;            // = 128 (shared memory per simdgroup)

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    // Shared memory layout — verbatim peer lines 6713-6717, with NSG=1 baked.
    // sq4: [0, PK)          — query as q4_t (half4)
    // ss:  [PK, PK+SH)      — score scratch (sgitg=0 only, NSG=1)
    // sm:  [PK+2*C, PK+SH)  — mask scratch (within ss, offset 2*C halfs)
    // so4: [PK+SH, PK+SH+2*PV) — output accumulator (sgitg=0 only, NSG=1)
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + 1*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + 1*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + 1*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + 1*PK + 1*SH);

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
        k += (uint)ikv2 * params.kv_capacity * DK;
        v += (uint)ikv2 * params.kv_capacity * DV;
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

        // Peer line 6766: pm pointer — replaced by inline mask (adaptation b).
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

        // NS10: strides-in-elements for K. Peer: NS10 = nb11/nb10 = (DK*2)/2 = DK.
        // NS20: strides-in-elements for V. Peer: NS20 = nb21/nb20 = (DV*2)/2 = DV.
        // These drive pk4/pv4 pointer arithmetic inside the loop (verbatim peer).

        // loop over the KV cache — verbatim peer line 6782, NWG=1/NSG=1 baked.
        // Peer: for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG)
        // At NWG=1, NSG=1: for (int ic0 = 0; ; ic0 += 1)
        for (int ic0 = 0; ; ic0 += 1) {
            int ic = ic0*C;
            if (ic >= (int)params.kv_seq_len) {   // args.ne11 = kv_seq_len
                break;
            }

            // has_kvpad=0 baked: kvpad branch physically deleted.

            // Adaptation (b): inline mask writing to sm[tiisg].
            // Replaces peer lines 6814-6816 (has_mask=1 external load).
            // Sliding-window ring-buffer logic from flash_attn_vec_hybrid.metal:506-519.
            {
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

            // Q*K^T — verbatim peer lines 6824-6900
            {
                // Peer: pk4 = (device const k4_t *) (k + ic*args.nb11)
                // k is half*, ic*nb11 bytes = ic*DK halfs → ic*DK/4 half4 elements.
                device      const k4_t * pk4 = (device const k4_t *) (k + (uint)ic*DK);
                threadgroup const q4_t * pq4 = sq4;

                // Peer: pk4 += ty*NS10/4 + tx; NS10=DK → ty*DK/4 + tx
                pk4 += ty*(DK/4) + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [0 ... C/NE - 1] = 0.0f };

                // each simdgroup processes 1 query and NE (NW/NL = 1) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            // Peer: pk4[cc*NE*NS10/4 + ii*NL]; NS10=DK, NE=1 → pk4[cc*DK4 + ii*NL]
                            mqk[cc] += dot((float4) pk4[cc*NE*(DK/4) + ii*NL], (float4) pq4[ii*NL]);
                        }
                    } else {
                        // Dead branch at compile time (kd4_t==k4_t==half4) — kept verbatim
                        // per RULE-1 (compiler dead-code-eliminates this path).
                        device const kd4_t * pk = (device const kd4_t *) (k + ((uint)(ic + NE*cc + ty)*DK));

                        k4_t mk;
                        const short nl_k = 1;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            (void)mk; (void)pk; (void)nl_k; (void)i;

                            mqk[cc] += dot((float4) mk, (float4) sq4[i]);
                        }
                    }

                    if (NE == 1) {
                        mqk[cc] = simd_sum(mqk[cc]);
                    } else {
                        // simdgroup reduce (NE=4) — dead at NE=1, kept verbatim
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

                // has_mask=1, has_scap=0, has_bias=0 → fast path — verbatim peer lines 6882-6885.
                if (true &&
                   true &&
                   true) {
                    ss[NE*tx + ty] = fma(mqk[tx], params.scale, (qk_t) sm[NE*tx + ty]);
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax — verbatim peer lines 6906-6926
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

            // O = O + (Q*K^T)*V — verbatim peer lines 6930-7006
            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                if (is_same<vd4_t, v4_t>::value) {
                    // Peer: pv4 = (device const v4_t *) (v + ic*args.nb21)
                    // v is half*, ic*nb21 bytes = ic*DV halfs → ic*DV/4 v4_t elements.
                    device const v4_t * pv4 = (device const v4_t *) (v + (uint)ic*DV);

                    // Peer: pv4 += ty*NS20/4 + tx; NS20=DV → ty*DV/4 + tx
                    pv4 += ty*(DV/4) + tx;

                    const auto sst = ss + ty;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            // Peer: pv4[cc*NE*NS20/4 + ii*NL]; NS20=DV, NE=1 → pv4[cc*DV4 + ii*NL]
                            lo[ii] += o4_t(float4(pv4[cc*NE*(DV/4) + ii*NL])*float4(sst[cc*NE]));
                        }
                    }
                } else {
                    // Dead branch at compile time (vd4_t==v4_t==half4) — kept verbatim.
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((uint)(ic + NE*cc + ty)*DV));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            const short nl_v = 1;

                            (void)mv; (void)pv4; (void)nl_v; (void)i;

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

    // parallel reduce — verbatim peer lines 7039-7066, NSG=1 baked.
    // At NSG=1: r starts at 0, loop body never executes (dead code).
    // Kept verbatim per RULE-1 (compiler eliminates at NSG=1).
    for (short r = 1/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final rescale with 1/S and store to global memory — verbatim peer lines 7069-7090
    if (sgitg == 0) {
        // nrows = ne3*ne2*ne1 = 1*num_heads*1 for decode.
        const int64_t nrows = params.num_heads;
        // rid = iq3*ne2*ne1 + iq2 + iq1*ne1 = iq2 for decode (iq1=0, iq3=0, ne1=1).
        const int64_t rid   = iq3*params.num_heads*1 + iq2 + iq1*1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*1; // NWG=1

        // NWG=1: 1/ss[0] (peer line 7076)
        const float S = (1 == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f);

        // interleave the workgroup data — verbatim peer line 7080; NWG=1, iwg=0 baked.
        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*1 + 1*i + 0] = (float4) so4[i]*S;
        }

        // store S and M — verbatim peer lines 7084-7089; NWG=1 → if (1>1) never taken.
        if (1 > 1) {
            if (tiisg == 0) {
                dst1[rid*(2*1) + 2*0 + 0] = ss[0];
                dst1[rid*(2*1) + 2*0 + 1] = ss[1];
            }
        }
    }
}
