// flash_attn_prefill_tensor.metal — tensor-API (Metal 4 matmul2d) flash
// attention for prefill, F16 K/V, F32 Q in, F32 O out.
//
// Port of the peer's `kernel_flash_attn_ext_tensor` (fa_tensor.metal) for
// T = half, without ALiBi, sinks or logit softcap, batch 1, one additive F16
// mask shared by all heads:
//
// - Q queries per threadgroup (32 at DK = 256, 16 at DK = 512), C = 64 cache
//   rows per step, NSG = 8 simdgroups (256 threads).
// - Q·Kᵀ and P·V are `mpp::tensor_ops::matmul2d` (matrix units on Apple10+).
// - Online softmax with lazy rescaling: a row's running max moves only when
//   it grows by more than 8 (e^8 fits in half); P is stored as half and the
//   row sum uses the same rounded values as P·V.
// - `blk` (mlx-native `flash_attn_prefill_blk_f16` with (BQ, BK) = (Q, 64)):
//   0 skips the 64-row tile, 2 skips the mask add, 1 adds the mask.
//
// Layouts (strides in the params; nothing assumes contiguity beyond rows):
// - Q   F32 `[rows, n_heads, DK]`: row stride `q_row_stride`, head stride
//   `q_head_stride` (elements).
// - K/V F16 `[n_kv_heads, kv_pad, D]`-like: row stride `k_row_stride` /
//   `v_row_stride`, head stride `k_head_stride` / `v_head_stride` (elements).
// - mask F16 `[rows, kv_pad]`, row stride `mask_row_stride` (elements).
// - O   F32 `[rows, n_heads, DV]` contiguous.
//
// Host contract (checked by the dispatcher): kv_pad % 64 == 0 and every K/V
// row below kv_pad is finite (padding rows are zero and masked -inf), so a
// masked cell contributes exp(-inf) * finite = +0.
//
// Per-row consistency: each row's max, sum and rescale are row-local; a row
// whose max did not move is scaled by exactly 1.0; a tile skipped by blk is
// fully masked for every row of its query tile, which equals adding zeros.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define FA_TENSOR_FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant short FA_TENSOR_NW = 32;
constant short FA_TENSOR_C = 64;
constant short FA_TENSOR_NSG = 8;

struct FlashAttnPrefillTensorParams {
    int   rows;            // query rows (ne01)
    int   n_heads;         // query heads (ne02 = ne1)
    int   n_kv_heads;      // ne_12_2
    int   kv_pad;          // cache rows, multiple of 64 (ne11)
    int   q_row_stride;    // elements
    int   q_head_stride;   // elements
    int   k_row_stride;    // elements
    int   k_head_stride;   // elements
    int   v_row_stride;    // elements
    int   v_head_stride;   // elements
    int   mask_row_stride; // elements
    float scale;
};

template<short DK, short DV, short Q>
kernel void flash_attn_prefill_tensor_impl(
        constant FlashAttnPrefillTensorParams & args [[buffer(0)]],
        device const float * q    [[buffer(1)]],
        device const half  * k    [[buffer(2)]],
        device const half  * v    [[buffer(3)]],
        device const half  * mask [[buffer(4)]],
        device const char  * blk  [[buffer(5)]],
        device       float * dst  [[buffer(6)]],
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW  = FA_TENSOR_NW;
    constexpr short C   = FA_TENSOR_C;
    constexpr short NSG = FA_TENSOR_NSG;
    constexpr short NT  = NW*NSG;
    constexpr short NQ  = Q/NSG;
    constexpr short NC  = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup half  * sq = (threadgroup half  *) shmem;      // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup half  * sp = (threadgroup half  *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last step (ic0 + 1) that rescaled O

    q += (uint64_t) iq1*args.q_row_stride + (uint64_t) iq2*args.q_head_stride;

    {
        const int ikv2 = iq2/(args.n_heads/args.n_kv_heads);

        k += (uint64_t) ikv2*args.k_head_stride;
        v += (uint64_t) ikv2*args.v_head_stride;
    }

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);

        float4 q4 = 0.0f;
        if (iq1 + j < args.rows) {
            q4 = ((device const float4 *) (q + (uint64_t) j*args.q_row_stride))[i%(DK/4)];
        }

        ((threadgroup half4 *) sq)[i] = (half4) (q4*args.scale);
    }

    device const half * pm[NQ];

    FA_TENSOR_FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm[jj] = mask + (uint64_t) (iq1 + j)*args.mask_row_stride;
    }

    {
        const int nblk0 = (args.kv_pad + C - 1)/C;

        blk += (uint64_t) (iq1/Q)*nblk0;
    }

    float M[NQ];
    float S[NQ];

    FA_TENSOR_FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    const int sk = args.k_row_stride;
    const int sv = args.v_row_stride;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    matmul2d<
        matmul2d_descriptor(Q, C, DK, false, true, false, matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    matmul2d<
        matmul2d_descriptor(Q, DV, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device half *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    // the O matrix
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FA_TENSOR_FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the host guarantees kv_pad % C == 0
    for (int ic0 = 0, ic = 0; ic < args.kv_pad; ++ic0, ic += C) {
        const char blk_cur = blk[ic0];

        if (blk_cur == 0) {
            continue;
        }

        // Q*K^T
        {
            auto tk = tensor((device half *) (k + (uint64_t) ic*sk), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FA_TENSOR_FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FA_TENSOR_FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (blk_cur != 2 && iq1 + j < args.rows) {
                FA_TENSOR_FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] += (float) pm[jj][ic + ii*NW + tiisg];
                }
            }

            float m = M[jj];

            FA_TENSOR_FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FA_TENSOR_FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const half p = (half) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FA_TENSOR_FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device half *) (v + (uint64_t) ic*sv), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FA_TENSOR_FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : 1.0f/S[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FA_TENSOR_FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    // store to global memory - rows past `rows` are clipped by the tensor extents
    device float * pdst = dst + ((uint64_t) iq2 + (uint64_t) iq1*args.n_heads)*DV;

    auto td = tensor(pdst, dextents<int32_t, 2>(DV, args.rows - iq1), array<int, 2>({1, args.n_heads*DV}));

    co.store(td);
}

typedef decltype(flash_attn_prefill_tensor_impl<256, 256, 32>) flash_attn_prefill_tensor_t;

template [[host_name("flash_attn_prefill_tensor_f16_d256")]] kernel flash_attn_prefill_tensor_t flash_attn_prefill_tensor_impl<256, 256, 32>;
template [[host_name("flash_attn_prefill_tensor_f16_d512")]] kernel flash_attn_prefill_tensor_t flash_attn_prefill_tensor_impl<512, 512, 16>;
