// ADR-037 Phase E1.1 — Tree-attention Metal kernel (2026-05-22).
//
// Derived from flash_attn_vec.metal (this file's parent in this crate).
// Identical in every respect EXCEPT the mask source:
//
//   * flash_attn_vec.metal computes an *implicit* causal/SWA mask from
//     `abs_pos = kv_seq_len - qL + iq1` and `causal_max_k = abs_pos + 1`.
//
//   * tree_attention.metal reads an *explicit* mask buffer of shape
//     `[qL, kv_seq_len]` (row-major float, attended = 0.0f, masked =
//     -65504.0f = -MAXHALF). Tree topologies (chains, fixed trees,
//     dynamic asymmetric trees) all reduce to this representation.
//
// Byte-identity contract (ADR-037 §4 Phase E1.1 acceptance gate):
//   When the explicit mask matches the implicit causal mask that
//   flash_attn_vec.metal would compute (i.e. row iq1 has 0.0f for
//   k_pos in [window_start, abs_pos] and -65504.0f elsewhere), this
//   kernel's output is byte-identical to flash_attn_vec.metal's output
//   for the same Q/K/V/params. This is the "tree=1" parity test.
//
// We do NOT support softcap or implicit sliding-window flags here —
// the mask buffer encodes everything visibility-related, so those
// flags would be redundant. Softcap (post-FMA logit scaling) is not
// needed for Qwen 3.6 27B (softcap=0 in production).
//
// Architecture: identical to flash_attn_vec.metal:
//   - NWG workgroups per head, each processes a chunk of KV positions
//   - NSG=1 simdgroup per workgroup (32 threads)
//   - C=32 KV positions per simdgroup iteration
//   - Online softmax with running max M and running sum S
//   - Results written to temp buffer with interleaved layout
//   - Reduce kernel combines NWG partial results (reuses
//     flash_attn_vec_reduce — same output layout)
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32

// Pad x up to next multiple of n (n must be power of 2).
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

// Parameters passed via buffer binding.
// IMPORTANT: this struct must match Rust `TreeAttentionParamsGpu`
// byte-for-byte (asserted by `test_gpu_params_layout` in
// src/ops/tree_attention.rs).
struct TreeAttentionParams {
    uint  n_heads;       // number of query heads
    uint  n_kv_heads;    // number of key/value heads (GQA)
    uint  head_dim;      // dimension per head (256 or 512)
    uint  kv_seq_len;    // current number of valid KV positions
    uint  kv_capacity;   // allocated capacity (stride between KV heads)
    float scale;         // attention score scaling factor
    uint  nwg;           // number of workgroups
    uint  q_l;           // number of queries (tree-node count to verify)
    // tree_mask layout: row-major [q_l, mask_stride].
    // Row iq1 occupies cells [iq1 * mask_stride, iq1 * mask_stride + kv_seq_len).
    // `mask_stride >= kv_seq_len`; using a separate stride lets the
    // caller align rows (e.g. to C=32 boundary) for safe out-of-bounds
    // reads inside a partial chunk crossing kv_seq_len. The shader
    // bounds-checks against kv_seq_len, so unaligned stride is also
    // legal — the stride only affects index arithmetic.
    uint  mask_stride;
};

// Reduce pass reuses `flash_attn_vec_reduce_*` from flash_attn_vec.metal
// verbatim — same output layout. No separate Reduce params struct here.

// --------------------------------------------------------------------------
// Template for the main tree-attention kernel.
//
// DK = head dim for keys, DV = head dim for values (both must equal here).
// KV_T = float for F32 KV, half for F16 KV.
//
// Thread/grid mapping mirrors flash_attn_vec_impl exactly so that the
// reduce kernel output layout is identical.
// --------------------------------------------------------------------------
template<short DK, short DV, typename KV_T>
kernel void tree_attention_impl(
    constant TreeAttentionParams &params     [[buffer(0)]],
    device const float           *Q          [[buffer(1)]],
    device const KV_T            *K          [[buffer(2)]],
    device const KV_T            *V          [[buffer(3)]],
    device       float           *dst        [[buffer(4)]],
    device const float           *tree_mask  [[buffer(5)]],
    threadgroup  half            *shmem      [[threadgroup(0)]],
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
    constexpr short SH  = 4 * C;

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");
    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    const uint NWG = params.nwg;

    // Threadgroup grid: (qL, n_heads, NWG)
    const ushort iwg = tgpig[2] % NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    // GQA mapping.
    const uint heads_per_kv = params.n_heads / params.n_kv_heads;
    const uint kv_head = iq2 / heads_per_kv;

    threadgroup half4  *sq4 = (threadgroup half4  *)(shmem);
    threadgroup float  *ss  = (threadgroup float  *)(shmem + PK);
    threadgroup float4 *so4 = (threadgroup float4 *)(shmem + PK + SH);

    so4 += tiisg;

    // Q layout: [n_heads, qL, head_dim] — same as flash_attn_vec.
    device const float4 *q4 =
        (device const float4 *)(Q + (iq2 * params.q_l + iq1) * DK);

    device const KV_T *k_base = K + kv_head * params.kv_capacity * DK;
    device const KV_T *v_base = V + kv_head * params.kv_capacity * DV;

    // Load Q into shared memory as half4 (identical to flash_attn_vec).
    for (ushort i = tiisg; i < PK4; i += NW) {
        sq4[i] = (i < DK4) ? half4(q4[i]) : half4(0.0h);
    }

    FOR_UNROLL (short i = 0; i < DV4 / NL; ++i) {
        so4[i * NL] = float4(0.0f);
    }

    for (ushort i = tiisg; i < SH / 4; i += NW) {
        ((threadgroup float *)(shmem + PK))[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S = 0.0f;
    float M = -FLT_MAX / 2;

    const ushort tx = tiisg;

    const uint kv_seq_len  = params.kv_seq_len;
    const uint mask_stride = params.mask_stride;

    // Tree-mask base for this query row (iq1).
    device const float *mask_row = tree_mask + (uint)iq1 * mask_stride;

    using kv4_t = vec<KV_T, 4>;

    // Loop over KV cache in chunks of C=32. The iteration bound is
    // `kv_seq_len`, identical to flash_attn_vec's `causal_max_k` when
    // running at qL=1 with the implicit-causal mask filled into the
    // explicit mask. Partial chunks crossing `kv_seq_len` get masked
    // values (-65504.0f) from the bounds-check path below.
    for (uint ic0 = iwg; ; ic0 += NWG) {
        uint ic = ic0 * C;
        if (ic >= kv_seq_len) {
            break;
        }

        // Read mask cell for (iq1, ic + tx). Out-of-range positions
        // (tx > kv_seq_len within a partial trailing chunk) get -65504.0f.
        {
            uint k_pos = ic + tx;
            float mask_val;
            if (k_pos < kv_seq_len) {
                mask_val = mask_row[k_pos];
            } else {
                mask_val = -65504.0f;
            }
            ss[tx] = mask_val;
        }

        if (simd_max(ss[tiisg]) <= -65504.0f) {
            continue;
        }

        // ---- Q * K^T ----  identical to flash_attn_vec.
        {
            device const kv4_t *pk4 = (device const kv4_t *)(k_base + ic * DK) + tx;
            threadgroup const half4 *pq4 = sq4 + tx;

            float mqk[C];

            FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                float partial = 0.0f;
                FOR_UNROLL (short ii = 0; ii < DK4 / NL; ++ii) {
                    partial += dot(float4(pk4[cc * DK4 + ii * NL]),
                                   float4(pq4[ii * NL]));
                }
                mqk[cc] = simd_sum(partial);
            }

            ss[tx] = fma(mqk[tx], params.scale, ss[tx]);
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Online softmax ---- identical to flash_attn_vec.
        {
            const float m_old = M;
            const float s_new = ss[tiisg];

            M = simd_max(max(M, s_new));

            const float ms = exp(m_old - M);
            const float vs = exp(s_new - M);

            S = S * ms + simd_sum(vs);

            ss[tiisg] = vs;

            FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] *= ms;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O + softmax_weights * V ---- identical to flash_attn_vec.
        {
            float4 lo[DV4 / NL];
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                lo[ii] = float4(0.0f);
            }

            device const kv4_t *pv4 = (device const kv4_t *)(v_base + ic * DV) + tx;

            FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                float weight = ss[cc];
                FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                    lo[ii] += float4(pv4[cc * DV4 + ii * NL]) * weight;
                }
            }

            FOR_UNROLL (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] += lo[ii];
            }
        }
    }

    if (tiisg == 0) {
        ss[0] = S;
        ss[1] = M;
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Write results ---- output layout matches flash_attn_vec exactly,
    // so we can reuse `flash_attn_vec_reduce` for the second pass.
    if (sgitg == 0) {
        const int64_t nrows = (int64_t)params.n_heads * (int64_t)params.q_l;
        const int64_t rid = iq2 + (int64_t)iq1 * params.n_heads;

        device float4 *dst4 = (device float4 *)dst;
        device float  *dst1 = (device float *)dst + nrows * DV * NWG;

        const float inv_S = (NWG == 1) ? ((S == 0.0f) ? 0.0f : 1.0f / S) : 1.0f;

        for (ushort i = tiisg; i < DV4; i += NW) {
            dst4[rid * DV4 * NWG + NWG * i + iwg] = so4[i] * inv_S;
        }

        if (NWG > 1 && tiisg == 0) {
            dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
            dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
        }
    }
}


// --------------------------------------------------------------------------
// Kernel instantiations.
// --------------------------------------------------------------------------

typedef decltype(tree_attention_impl<256, 256, float>) tree_attention_f32kv_t;

template [[host_name("tree_attention_dk256")]]
kernel tree_attention_f32kv_t tree_attention_impl<256, 256, float>;

template [[host_name("tree_attention_dk512")]]
kernel tree_attention_f32kv_t tree_attention_impl<512, 512, float>;

typedef decltype(tree_attention_impl<256, 256, half>) tree_attention_f16kv_t;

template [[host_name("tree_attention_f16kv_dk256")]]
kernel tree_attention_f16kv_t tree_attention_impl<256, 256, half>;

template [[host_name("tree_attention_f16kv_dk512")]]
kernel tree_attention_f16kv_t tree_attention_impl<512, 512, half>;
