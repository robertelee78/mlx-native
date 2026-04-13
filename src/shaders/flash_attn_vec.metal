// Ported from llama.cpp ggml-metal.metal — flash_attn_ext_vec template
// (MIT licensed). SIMD-vectorized decode-path scaled dot product attention.
// Source: /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
//
// Copyright the llama.cpp Authors. See LICENSE-MIT-llamacpp.
//
// Simplified for F32 Q/K/V with NE=1 (Gemma 4 head dims 256 and 512).
// No quantized KV, no ALiBi, no attention sinks, no logit softcapping.
// Supports causal masking and sliding window via implicit mask computation.
//
// Architecture:
//   - NWG workgroups per head, each processes a chunk of KV positions
//   - NSG=1 simdgroup per workgroup (32 threads)
//   - C=32 KV positions per simdgroup iteration
//   - Online softmax with running max M and running sum S
//   - Results written to temp buffer with interleaved layout
//   - Reduce kernel combines NWG partial results using SIMD reduction

#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define C           32   // KV positions per simdgroup iteration

// Pad x up to next multiple of n (n must be power of 2).
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

// Parameters passed via buffer binding.
struct FlashAttnVecParams {
    uint  n_heads;         // number of query heads
    uint  n_kv_heads;      // number of key/value heads (GQA)
    uint  head_dim;        // dimension per head (256 or 512)
    uint  kv_seq_len;      // current number of valid KV positions
    uint  kv_capacity;     // allocated capacity (stride between KV heads)
    float scale;           // attention score scaling factor
    uint  mask_type;       // 0=none, 1=causal, 2=sliding_window
    uint  sliding_window;  // window size (mask_type==2 only)
    float softcap;         // logit softcapping (0 = disabled)
    uint  nwg;             // number of workgroups
};

// Parameters for the reduce kernel.
struct FlashAttnVecReduceParams {
    uint nrows;            // total output rows (n_heads * batch)
};


// --------------------------------------------------------------------------
// Template for the main flash attention vector kernel.
//
// DK = head dimension for keys, DV = head dimension for values.
// Both must be multiples of 32.
//
// Thread mapping (NE=1):
//   NL = N_SIMDWIDTH = 32 (lanes per thread contributing to each dot product)
//   tx = tiisg  (each thread occupies a unique SIMD lane)
//   ty = 0      (always)
//
// In each iteration, thread tx:
//   - Computes partial dot products of Q with K[ic+cc] for cc in [0, C)
//     using DK4/NL float4 elements per dot product (DK/128 elements)
//   - Uses simd_sum to reduce partial dot products to full results
//   - Reads V[ic+cc] and multiplies by the attention weight ss[cc]
//     using DV4/NL float4 elements, accumulated into local registers
//
// Shared memory layout (in half units):
//   [0, PK)                              — Q vector as half4 (PK4 values)
//   [PK, PK + SH)                        — scratch for attention scores (SH = 4*C)
//   [PK + SH, PK + SH + 2*PV)           — output accumulator as float4
// --------------------------------------------------------------------------

template<short DK, short DV>
kernel void flash_attn_vec(
    device const FlashAttnVecParams *params [[buffer(0)]],
    device const float              *Q      [[buffer(1)]],
    device const float              *K      [[buffer(2)]],
    device const float              *V      [[buffer(3)]],
    device       float              *dst    [[buffer(4)]],
    threadgroup  half               *shmem  [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Compile-time constants.
    constexpr short DK4 = DK / 4;
    constexpr short DV4 = DV / 4;
    constexpr short NW  = N_SIMDWIDTH;      // 32
    constexpr short NL  = NW;               // NE=1 -> NL=NW
    constexpr short PK  = PAD2(DK, 128);    // pad head dim to 128 boundary
    constexpr short PK4 = PK / 4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV / 4;
    constexpr short SH  = 4 * C;            // 128 halfs = 64 floats

    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");
    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    const uint NWG = params->nwg;

    // Threadgroup grid: (n_queries, n_heads, n_batches * NWG)
    const ushort iwg = tgpig[2] % NWG;    // workgroup index within this head
    const ushort iq2 = tgpig[1];           // head index
    const ushort iq1 = tgpig[0];           // query index (0 for decode)

    // GQA: map query head to KV head.
    const uint heads_per_kv = params->n_heads / params->n_kv_heads;
    const uint kv_head = iq2 / heads_per_kv;

    // Shared memory pointers.
    // Q stored as half4 for reduced memory (loaded from float4, cast to half4).
    threadgroup half4  *sq4 = (threadgroup half4  *)(shmem);
    threadgroup float  *ss  = (threadgroup float  *)(shmem + PK);
    threadgroup float4 *so4 = (threadgroup float4 *)(shmem + PK + SH);

    // Each thread owns its SIMD lane in the output accumulator.
    so4 += tiisg;

    // Compute device pointers.
    // Q layout: [n_heads, seq_len, head_dim] — for decode, seq_len=1.
    device const float4 *q4 = (device const float4 *)(Q + iq2 * DK);

    // K layout: [n_kv_heads, kv_capacity, head_dim]
    device const float *k_base = K + kv_head * params->kv_capacity * DK;

    // V layout: [n_kv_heads, kv_capacity, head_dim]
    device const float *v_base = V + kv_head * params->kv_capacity * DV;

    // Load Q into shared memory as half4.
    for (ushort i = tiisg; i < PK4; i += NW) {
        sq4[i] = (i < DK4) ? half4(q4[i]) : half4(0.0h);
    }

    // Zero the output accumulator.
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

    // Compute masking bounds.
    const uint kv_seq_len = params->kv_seq_len;
    // For decode: single query at position (kv_seq_len - 1).
    const uint abs_pos = kv_seq_len - 1;
    const uint causal_max_k = min(abs_pos + 1, kv_seq_len); // = kv_seq_len

    uint window_start = 0;
    if (params->mask_type == 2 && params->sliding_window > 0) {
        window_start = (abs_pos >= params->sliding_window)
            ? (abs_pos - params->sliding_window + 1) : 0;
    }

    // Main loop over KV cache in chunks of C=32.
    // Workgroup iwg handles chunks: iwg, iwg+NWG, iwg+2*NWG, ...
    for (uint ic0 = iwg; ; ic0 += NWG) {
        uint ic = ic0 * C;
        if (ic >= causal_max_k) {
            break;
        }

        // Compute implicit mask for this chunk.
        {
            uint k_pos = ic + tx;
            float mask_val = 0.0f;
            if (k_pos >= causal_max_k || k_pos < window_start) {
                mask_val = -65504.0f;  // -MAXHALF: effectively -inf for half precision
            }
            ss[tx] = mask_val;
        }

        // Skip all-masked chunks.
        if (simd_max(ss[tiisg]) <= -65504.0f) {
            continue;
        }

        // ---- Q * K^T ----
        // Each thread tx computes partial dot products for KV rows [ic..ic+C).
        // cc indexes the KV position within this chunk (0..C-1).
        // Each dot product is reduced via simd_sum across all 32 threads.
        {
            // pk4 points to K[ic, 0] as float4, then offset by tx.
            device const float4 *pk4 = (device const float4 *)(k_base + ic * DK) + tx;
            threadgroup const half4 *pq4 = sq4 + tx;

            // mqk[cc] will hold the full dot product for KV position (ic + cc).
            float mqk[C];

            for (short cc = 0; cc < C; ++cc) {
                float partial = 0.0f;
                for (short ii = 0; ii < DK4 / NL; ++ii) {
                    partial += dot(float4(pk4[cc * DK4 + ii * NL]),
                                   float4(pq4[ii * NL]));
                }
                mqk[cc] = simd_sum(partial);
            }

            // Combine with mask and scale, store to scratch.
            // ss[tx] already contains the mask value for position (ic + tx).
            ss[tx] = fma(mqk[tx], params->scale, ss[tx]);
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

            // Store the softmax weight for use in V accumulation.
            ss[tiisg] = vs;

            // Rescale previous output accumulation.
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] *= ms;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O + softmax_weights * V ----
        {
            // Local accumulator for this chunk's contribution.
            float4 lo[DV4 / NL];
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                lo[ii] = float4(0.0f);
            }

            // pv4 points to V[ic, 0] as float4, then offset by tx.
            device const float4 *pv4 = (device const float4 *)(v_base + ic * DV) + tx;

            for (short cc = 0; cc < C; ++cc) {
                float weight = ss[cc];  // softmax weight for KV pos (ic + cc)
                for (short ii = 0; ii < DV4 / NL; ++ii) {
                    lo[ii] += float4(pv4[cc * DV4 + ii * NL]) * weight;
                }
            }

            // No SIMD reduction needed for NE=1 — each thread owns distinct
            // output dimensions. Accumulate directly.
            for (short ii = 0; ii < DV4 / NL; ++ii) {
                so4[ii * NL] += lo[ii];
            }
        }
    }

    // Store M and S for the reduce kernel.
    if (tiisg == 0) {
        ss[0] = S;
        // Reuse ss[1] for M (cast through float pointer).
        ss[1] = M;
    }

    // Remove per-thread offset before cross-simdgroup access.
    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Write results to global memory ----
    // Layout in dst: interleaved by workgroup for the reduce kernel.
    //   dst[rid * DV4 * NWG + NWG * i + iwg] = output float4 at dim chunk i
    //   After the DV data: S and M values for each (row, workgroup).
    if (sgitg == 0) {
        const int64_t nrows = params->n_heads;  // For batch=1
        const int64_t rid = iq2 + (int64_t)iq1 * params->n_heads;

        device float4 *dst4 = (device float4 *)dst;
        device float  *dst1 = (device float *)dst + nrows * DV * NWG;

        // When NWG==1, normalize directly. Otherwise store raw for reduce.
        const float inv_S = (NWG == 1) ? ((S == 0.0f) ? 0.0f : 1.0f / S) : 1.0f;

        for (ushort i = tiisg; i < DV4; i += NW) {
            dst4[rid * DV4 * NWG + NWG * i + iwg] = so4[i] * inv_S;
        }

        // Store S and M for the reduce kernel.
        if (NWG > 1 && tiisg == 0) {
            dst1[rid * (2 * NWG) + 2 * iwg + 0] = S;
            dst1[rid * (2 * NWG) + 2 * iwg + 1] = M;
        }
    }
}


// --------------------------------------------------------------------------
// Kernel instantiations
// --------------------------------------------------------------------------

typedef decltype(flash_attn_vec<256, 256>) flash_attn_vec_t;

template [[host_name("flash_attn_vec_dk256")]]
kernel flash_attn_vec_t flash_attn_vec<256, 256>;

template [[host_name("flash_attn_vec_dk512")]]
kernel flash_attn_vec_t flash_attn_vec<512, 512>;


// --------------------------------------------------------------------------
// Reduce kernel — combines partial results from NWG workgroups.
//
// Grid: (nrows, 1, 1)   Threadgroup: (32 * NWG, 1, 1)
//
// But we hardcode to a single simdgroup of 32 threads (NWG <= 32).
// Each thread reads the S and M for one workgroup, then SIMD operations
// combine them.
// --------------------------------------------------------------------------

template<short DV>
kernel void flash_attn_vec_reduce(
    device const FlashAttnVecReduceParams *params [[buffer(0)]],
    device const float                    *htmp   [[buffer(1)]],
    device       float                    *dst    [[buffer(2)]],
    constant     uint                     &nwg_param [[buffer(3)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short DV4 = DV / 4;

    const uint NWG = nwg_param;
    const uint64_t rid = tgpig;  // row index
    const ushort iwg = tiisg;    // each thread handles one workgroup

    // S and M values are stored after all DV data.
    device const float *sm = htmp + (uint64_t)params->nrows * DV * NWG;

    // Load this workgroup's S and M.
    float S_wg = (iwg < NWG) ? sm[rid * (2 * NWG) + 2 * iwg + 0] : 0.0f;
    float M_wg = (iwg < NWG) ? sm[rid * (2 * NWG) + 2 * iwg + 1] : -FLT_MAX / 2;

    // Find global max across all workgroups.
    const float M_global = simd_max(M_wg);

    // Compute rescaling factor for each workgroup.
    const float ms = exp(M_wg - M_global);

    // Sum of all rescaled S values.
    float S_total = simd_sum(S_wg * ms);
    float inv_S = (S_total == 0.0f) ? 0.0f : 1.0f / S_total;

    // Pointers to interleaved partial results.
    device const float4 *htmp4 = (device const float4 *)htmp + rid * DV4 * NWG;
    device       float4 *dst4  = (device       float4 *)dst  + rid * DV4;

    // Reduce: for each output dimension chunk, sum the rescaled contributions
    // from all workgroups using SIMD operations.
    for (short i = sgitg; i < DV4; i += NWG) {
        float4 val = (iwg < NWG) ? htmp4[i * NWG + iwg] * ms : float4(0.0f);
        float4 reduced = float4(simd_sum(val[0]),
                                simd_sum(val[1]),
                                simd_sum(val[2]),
                                simd_sum(val[3]));
        if (iwg == 0) {
            dst4[i] = reduced * inv_S;
        }
    }
}

typedef decltype(flash_attn_vec_reduce<256>) flash_attn_vec_reduce_t;

template [[host_name("flash_attn_vec_reduce_dk256")]]
kernel flash_attn_vec_reduce_t flash_attn_vec_reduce<256>;

template [[host_name("flash_attn_vec_reduce_dk512")]]
kernel flash_attn_vec_reduce_t flash_attn_vec_reduce<512>;
