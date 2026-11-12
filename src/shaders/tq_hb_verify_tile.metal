// tq_hb_verify_tile.metal — tiled verify attention over the TQ-HB KV cache.
//
// One joint verify step attends, per lane, a window of up to TQV_ROWS rows at
// consecutive positions of the lane's slot. The per-row decode kernel
// (flash_attn_vec_tq_hb) scans a slot's history once per row and query head;
// here a threadgroup owns one (lane, KV head, history split) and computes all
// of the window's rows and the KV head's query heads together, M = TQV_ROWS *
// G fused rows, with `mpp::tensor_ops::matmul2d` for Q·Kᵀ and P·V over pages
// of TQV_PAGE cache tokens. The structure follows Splash's paged verify
// attention (runtime/metal/kernels/common/paged_attention_tile.h): balanced
// history splits writing F32 partials and (max, sum) statistics, and a
// fixed-order reduce.
//
// The cache is hf2q's TQ-HB format, unchanged: 8-bit Lloyd-Max codes
// `[slot, kv_head, capacity, 256]` and one F32 norm per (slot, kv_head,
// token), in the sign-Hadamard rotated domain. A page's codes are dequantized
// to their half centroids in threadgroup memory (the codebook is not linear,
// so the codes cannot be a matmul operand). The key norms scale the F32
// scores; a probability times its token's value norm is the half operand of
// P·V (Splash carries the same product in bf16), saturated at the largest
// half, so a value norm above about 1e6 would clip. Both matmuls accumulate
// in F32.
//
// Kernels:
// - tq_hb_verify_tile_pack_g{6,8}: sign-Hadamard rotation of the window rows'
//   queries (the arithmetic of fwht_sign_premult_f32_d256) into KV-head-major
//   half tiles `[lane, kv_head, row, head in group, 256]`.
// - tq_hb_verify_tile_split_g{6,8}: the page loop of one (KV head, split,
//   lane).
// - tq_hb_verify_tile_reduce_g{6,8}: combines one window row's and query
//   head's splits in split order into `[lane, row, head, 256]` F32 (rotated
//   domain).
// - tq_hb_verify_tile_undo_gate: the arithmetic of
//   fwht_sign_undo_sigmoid_mul_f32_d256 for every packed row, reading the
//   rotated row its map names (padding rows repeat a window row).
//
// Row independence: a lane's split count and page partition depend only on
// its committed length; a fused row's scores, running maximum, sum and
// rescale are its own (a row whose maximum did not move is scaled by exactly
// 1.0); tokens at or after a row's causal end get probability exactly 0.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define TQV_ROWS 8u
#define TQV_PAGE 32u
#define TQV_DIM 256u
#define TQV_MAX_SPLITS 128u
#define TQV_HALF_MAX 65504.0f

// One lane of the step. Must match `TqHbVerifyLaneGpu`.
struct TqHbVerifyLane {
    uint slot;       // outer index of the lane's slot in the cache buffers
    uint committed;  // tokens before the window (position of its first row)
    uint rows;       // window rows, 1..TQV_ROWS
    uint first_row;  // packed row of the window's first row
    uint splits;     // history splits of this lane
};

// Must match `TqHbVerifyParamsGpu`.
struct TqHbVerifyParams {
    uint  n_heads;
    uint  n_kv_heads;
    uint  kv_capacity;
    uint  slot_splits;  // partial slots per (lane, KV head)
    float scale;
};

// 8-bit Lloyd-Max centroids of flash_attn_vec_tq_hb.metal (CODEBOOK_HB_8BIT),
// rounded to half.
constant half TQV_CODEBOOK_8BIT[256] = {
    half(-5.0652659f), half(-4.6836997f), half(-4.4467193f), half(-4.2715508f),
    half(-4.1311907f), half(-4.0132856f), half(-3.9111092f), half(-3.8205780f),
    half(-3.7390194f), half(-3.6645851f), half(-3.5959415f), half(-3.5320936f),
    half(-3.4722785f), half(-3.4158977f), half(-3.3624729f), half(-3.3116156f),
    half(-3.2630056f), half(-3.2163758f), half(-3.1715011f), half(-3.1281899f),
    half(-3.0862780f), half(-3.0456229f), half(-3.0061011f), half(-2.9676040f),
    half(-2.9300362f), half(-2.8933131f), half(-2.8573596f), half(-2.8221086f),
    half(-2.7874999f), half(-2.7534795f), half(-2.7199985f), half(-2.6870129f),
    half(-2.6544825f), half(-2.6223710f), half(-2.5906452f), half(-2.5592748f),
    half(-2.5282321f), half(-2.4974918f), half(-2.4670306f), half(-2.4368270f),
    half(-2.4068614f), half(-2.3771157f), half(-2.3475732f), half(-2.3182184f),
    half(-2.2890372f), half(-2.2600165f), half(-2.2311440f), half(-2.2024086f),
    half(-2.1737998f), half(-2.1453081f), half(-2.1169245f), half(-2.0886408f),
    half(-2.0604493f), half(-2.0323430f), half(-2.0043154f), half(-1.9763603f),
    half(-1.9484722f), half(-1.9206458f), half(-1.8928763f), half(-1.8651592f),
    half(-1.8374904f), half(-1.8098662f), half(-1.7822828f), half(-1.7547372f),
    half(-1.7272261f), half(-1.6997469f), half(-1.6722970f), half(-1.6448739f),
    half(-1.6174755f), half(-1.5900996f), half(-1.5627445f), half(-1.5354084f),
    half(-1.5080897f), half(-1.4807869f), half(-1.4534986f), half(-1.4262237f),
    half(-1.3989610f), half(-1.3717093f), half(-1.3444678f), half(-1.3172356f),
    half(-1.2900118f), half(-1.2627956f), half(-1.2355865f), half(-1.2083838f),
    half(-1.1811868f), half(-1.1539951f), half(-1.1268081f), half(-1.0996255f),
    half(-1.0724469f), half(-1.0452718f), half(-1.0180999f), half(-0.9909310f),
    half(-0.9637647f), half(-0.9366008f), half(-0.9094390f), half(-0.8822793f),
    half(-0.8551212f), half(-0.8279648f), half(-0.8008098f), half(-0.7736561f),
    half(-0.7465035f), half(-0.7193520f), half(-0.6922014f), half(-0.6650517f),
    half(-0.6379027f), half(-0.6107544f), half(-0.5836067f), half(-0.5564596f),
    half(-0.5293129f), half(-0.5021667f), half(-0.4750208f), half(-0.4478753f),
    half(-0.4207301f), half(-0.3935852f), half(-0.3664405f), half(-0.3392960f),
    half(-0.3121517f), half(-0.2850076f), half(-0.2578636f), half(-0.2307198f),
    half(-0.2035761f), half(-0.1764324f), half(-0.1492888f), half(-0.1221453f),
    half(-0.0950019f), half(-0.0678584f), half(-0.0407151f), half(-0.0135717f),
    half(0.0135717f), half(0.0407151f), half(0.0678584f), half(0.0950019f),
    half(0.1221453f), half(0.1492888f), half(0.1764324f), half(0.2035761f),
    half(0.2307198f), half(0.2578636f), half(0.2850076f), half(0.3121517f),
    half(0.3392960f), half(0.3664405f), half(0.3935852f), half(0.4207301f),
    half(0.4478753f), half(0.4750208f), half(0.5021667f), half(0.5293129f),
    half(0.5564596f), half(0.5836067f), half(0.6107544f), half(0.6379027f),
    half(0.6650517f), half(0.6922014f), half(0.7193520f), half(0.7465035f),
    half(0.7736561f), half(0.8008098f), half(0.8279648f), half(0.8551212f),
    half(0.8822793f), half(0.9094390f), half(0.9366008f), half(0.9637647f),
    half(0.9909310f), half(1.0180999f), half(1.0452718f), half(1.0724469f),
    half(1.0996255f), half(1.1268081f), half(1.1539951f), half(1.1811868f),
    half(1.2083838f), half(1.2355865f), half(1.2627956f), half(1.2900118f),
    half(1.3172356f), half(1.3444678f), half(1.3717093f), half(1.3989610f),
    half(1.4262237f), half(1.4534986f), half(1.4807869f), half(1.5080897f),
    half(1.5354084f), half(1.5627445f), half(1.5900996f), half(1.6174755f),
    half(1.6448739f), half(1.6722970f), half(1.6997469f), half(1.7272261f),
    half(1.7547372f), half(1.7822828f), half(1.8098662f), half(1.8374904f),
    half(1.8651592f), half(1.8928763f), half(1.9206458f), half(1.9484722f),
    half(1.9763603f), half(2.0043154f), half(2.0323430f), half(2.0604493f),
    half(2.0886408f), half(2.1169245f), half(2.1453081f), half(2.1737998f),
    half(2.2024086f), half(2.2311440f), half(2.2600165f), half(2.2890372f),
    half(2.3182184f), half(2.3475732f), half(2.3771157f), half(2.4068614f),
    half(2.4368270f), half(2.4670306f), half(2.4974918f), half(2.5282321f),
    half(2.5592748f), half(2.5906452f), half(2.6223710f), half(2.6544825f),
    half(2.6870129f), half(2.7199985f), half(2.7534795f), half(2.7874999f),
    half(2.8221086f), half(2.8573596f), half(2.8933131f), half(2.9300362f),
    half(2.9676040f), half(3.0061011f), half(3.0456229f), half(3.0862780f),
    half(3.1281899f), half(3.1715011f), half(3.2163758f), half(3.2630056f),
    half(3.3116156f), half(3.3624729f), half(3.4158977f), half(3.4722785f),
    half(3.5320936f), half(3.5959415f), half(3.6645851f), half(3.7390194f),
    half(3.8205780f), half(3.9111092f), half(4.0132856f), half(4.1311907f),
    half(4.2715508f), half(4.4467193f), half(4.6836997f), half(5.0652659f),
};

// D1 sign table of fwht_standalone.metal (TBQ_SIGNS_256).
constant uint8_t TQV_SIGNS_256[32] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
};

// The pages a lane's tiles read: those holding its committed tokens and a
// full window. The bound does not depend on the window's row count.
inline uint tqv_pages(TqHbVerifyLane lane, uint kv_capacity) {
    const uint bound = min(lane.committed + TQV_ROWS, kv_capacity);
    return (bound + TQV_PAGE - 1) / TQV_PAGE;
}

inline uint tqv_pages_per_split(uint pages, uint splits) {
    return (pages + splits - 1) / splits;
}

// ---------------------------------------------------------------------------
// Query pack: one simdgroup per (query head, window row, lane).
// ---------------------------------------------------------------------------
template <uint G>
inline void tqv_pack(
    constant TqHbVerifyParams &params,
    constant TqHbVerifyLane *lanes,
    device const float *q,
    device half *queries,
    uint3 group,
    uint lane)
{
    constexpr uint D = TQV_DIM;
    constexpr uint EPT = D / 32;
    const uint head = group.x, row = group.y;
    const TqHbVerifyLane l = lanes[group.z];
    const uint kv_head = head / G;
    const ulong tile = (ulong(group.z) * params.n_kv_heads + kv_head) * TQV_ROWS + row;
    device half *dst = queries + (tile * G + head % G) * D + lane * EPT;
    if (row >= l.rows) {
        for (uint i = 0; i < EPT; ++i) dst[i] = 0.0h;
        return;
    }
    device const float *src =
        q + (ulong(l.first_row + row) * params.n_heads + head) * D + lane * EPT;
    float elems[EPT];
    for (uint i = 0; i < EPT; ++i) {
        const uint j = lane * EPT + i;
        const float sign = ((TQV_SIGNS_256[j >> 3] >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] = src[i] * sign;
    }
    for (uint h = 1; h < EPT; h <<= 1) {
        for (uint i = 0; i < EPT; ++i) {
            const uint partner = i ^ h;
            if (partner > i) {
                const float sum = elems[i] + elems[partner];
                const float diff = elems[i] - elems[partner];
                elems[i] = sum;
                elems[partner] = diff;
            }
        }
    }
    for (ushort delta = 1; delta < 32; delta <<= 1) {
        for (uint i = 0; i < EPT; ++i) {
            const float partner = simd_shuffle_xor(elems[i], delta);
            elems[i] = (lane & delta) ? partner - elems[i] : elems[i] + partner;
        }
    }
    const float inv_sqrt_d = rsqrt(float(D));
    for (uint i = 0; i < EPT; ++i) dst[i] = half(elems[i] * inv_sqrt_d);
}

#define TQV_PACK(Name, G)                                                      \
  kernel void Name(                                                            \
      constant TqHbVerifyParams &params [[buffer(0)]],                         \
      constant TqHbVerifyLane *lanes [[buffer(1)]],                            \
      device const float *q [[buffer(2)]],                                     \
      device half *queries [[buffer(3)]],                                      \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    tqv_pack<G>(params, lanes, q, queries, group, lane);                       \
  }

TQV_PACK(tq_hb_verify_tile_pack_g6, 6)
TQV_PACK(tq_hb_verify_tile_pack_g8, 8)
#undef TQV_PACK

// ---------------------------------------------------------------------------
// Split: 256 threads per (KV head, split, lane).
// ---------------------------------------------------------------------------

// A page's codes of one KV head -> half centroids `[token, 256]` in
// threadgroup memory: thread t owns 32 consecutive codes of token t / 8.
inline void tqv_dequant_page(
    device const uchar *codes, threadgroup half *staged, uint thread_index)
{
    const uint offset = thread_index * 32u;
    device const uint *src = reinterpret_cast<device const uint *>(codes + offset);
    threadgroup half4 *dst = reinterpret_cast<threadgroup half4 *>(staged + offset);
#pragma unroll
    for (uint i = 0; i < 8; ++i) {
        const uint word = src[i];
        dst[i] = half4(TQV_CODEBOOK_8BIT[word & 0xffu],
                       TQV_CODEBOOK_8BIT[(word >> 8) & 0xffu],
                       TQV_CODEBOOK_8BIT[(word >> 16) & 0xffu],
                       TQV_CODEBOOK_8BIT[word >> 24]);
    }
}

// One page of one tile: key-norm-scaled scores become value-norm-scaled half
// probabilities and the row statistics advance. Four threads own one fused
// row, eight consecutive tokens each (the layout of Splash's
// splash_attention_page_softmax).
template <uint G>
inline void tqv_page_softmax(
    threadgroup const float *scores, threadgroup half *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_norms, device const float4 *value_norms,
    float score_scale, uint token_start, uint committed, uint rows,
    uint thread_index)
{
    constexpr uint N = TQV_PAGE;
    constexpr uint M = TQV_ROWS * G;
    if (thread_index >= 4 * M)
        return;
    const uint fused_row = thread_index / 4;
    const uint part = thread_index % 4;
    const uint column = part * 8;
    const uint query_row = fused_row / G;
    // The row attends the tokens before its causal end; a row past the
    // window's repeats the last one and is never read.
    const uint causal_end = committed + min(query_row, rows - 1) + 1;
    const uint token = token_start + column;
    threadgroup const float4 *cells =
        reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N + column);
    const float value_scale = rsqrt(float(TQV_DIM));
    float score[8];
    {
        const float4 low = cells[0] * key_norms[part * 2] * score_scale;
        const float4 high = cells[1] * key_norms[part * 2 + 1] * score_scale;
        score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
        score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
    }
    float local_max = -INFINITY;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
        score[j] = token + j < causal_end ? score[j] : -INFINITY;
        local_max = max(local_max, score[j]);
    }
    local_max = max(local_max, simd_shuffle_xor(local_max, 1));
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
    const float previous_max = row_max[fused_row];
    const float next_max = max(previous_max, local_max);
    float probability[8];
    float local_sum = 0.0f;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
        probability[j] = token + j < causal_end ? exp(score[j] - next_max) : 0.0f;
        local_sum += probability[j];
    }
    local_sum += simd_shuffle_xor(local_sum, 1);
    local_sum += simd_shuffle_xor(local_sum, 2);
    if (part == 0) {
        const float scale = next_max == -INFINITY || next_max == previous_max
                                ? 1.0f
                                : exp(previous_max - next_max);
        previous_scale[fused_row] = scale;
        row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
        row_max[fused_row] = next_max;
        if (scale != 1.0f)
            atomic_store_explicit(rescale, 1u, memory_order_relaxed);
    }
    // The P·V operand: probability times value norm, rounded to half and
    // saturated at the largest half. Masked tokens stay exactly zero whatever
    // their stored norm holds.
    const float4 low_norms = value_norms[part * 2] * value_scale;
    const float4 high_norms = value_norms[part * 2 + 1] * value_scale;
    const float4 low(token + 0 < causal_end ? probability[0] * low_norms.x : 0.0f,
                     token + 1 < causal_end ? probability[1] * low_norms.y : 0.0f,
                     token + 2 < causal_end ? probability[2] * low_norms.z : 0.0f,
                     token + 3 < causal_end ? probability[3] * low_norms.w : 0.0f);
    const float4 high(token + 4 < causal_end ? probability[4] * high_norms.x : 0.0f,
                      token + 5 < causal_end ? probability[5] * high_norms.y : 0.0f,
                      token + 6 < causal_end ? probability[6] * high_norms.z : 0.0f,
                      token + 7 < causal_end ? probability[7] * high_norms.w : 0.0f);
    threadgroup half4 *out =
        reinterpret_cast<threadgroup half4 *>(probabilities + fused_row * N + column);
    out[0] = half4(clamp(low, float4(-TQV_HALF_MAX), float4(TQV_HALF_MAX)));
    out[1] = half4(clamp(high, float4(-TQV_HALF_MAX), float4(TQV_HALF_MAX)));
}

template <uint G>
inline void tqv_split(
    constant TqHbVerifyParams &params,
    constant TqHbVerifyLane *lanes,
    device half *queries,
    device const uchar *k_codes, device const float *k_norms,
    device const uchar *v_codes, device const float *v_norms,
    device float *partials, device float *statistics,
    threadgroup half *staged, threadgroup float *scores,
    threadgroup half *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint3 group, uint thread_index)
{
    constexpr ushort M = TQV_ROWS * G;
    constexpr ushort N = TQV_PAGE;
    constexpr ushort D = TQV_DIM;
    const uint kv_head = group.x, split = group.y;
    const TqHbVerifyLane lane = lanes[group.z];
    if (split >= lane.splits)
        return;
    const uint pages = tqv_pages(lane, params.kv_capacity);
    const uint per_split = tqv_pages_per_split(pages, lane.splits);
    const uint page_begin = split * per_split;
    if (page_begin >= pages)
        return;
    const uint page_end = min(pages, page_begin + per_split);
    if (thread_index < M) {
        row_max[thread_index] = -INFINITY;
        row_sum[thread_index] = 0.0f;
    }

    const ulong tile = ulong(group.z) * params.n_kv_heads + kv_head;
    const ulong head_tokens =
        (ulong(lane.slot) * params.n_kv_heads + kv_head) * params.kv_capacity;
    const float score_scale = rsqrt(float(D)) * params.scale;

    // Q·Kᵀ: half queries (device) x half keys (staged) -> F32 scores.
    // P·V: half probabilities x half values (staged) -> F32 accumulation.
    auto qt = tensor(queries + tile * TQV_ROWS * G * D, dextents<int, 2>{D, M},
                     array<int, 2>{1, D});
    auto kt = tensor(staged, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto q0 = qt.slice<D, M>(0, 0);
    auto k0 = kt.slice<D, N>(0, 0);
    auto s0 = st.slice<N, M>(0, 0);
    auto p0 = pt.slice<N, M>(0, 0);
    constexpr auto qk_descriptor =
        matmul2d_descriptor(M, N, D, false, true, false,
                            matmul2d_descriptor::mode::multiply);
    constexpr auto pv_descriptor =
        matmul2d_descriptor(M, D, N, false, false, false,
                            matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
    matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(k0), float>();
    const bool running_full =
        uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
    for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (running_full || running.is_valid_element(index))
            running[index] = 0.0f;
    }

    // Four barriers per page order the staged keys, the score store, the
    // softmax with the staged values, and the probability reads of P·V.
    for (uint page = page_begin; page < page_end; ++page) {
        const uint token_start = page * N;
        const ulong first = head_tokens + token_start;
        tqv_dequant_page(k_codes + first * D, staged, thread_index);
        if (thread_index == 0)
            atomic_store_explicit(rescale, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto page_scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(k0), float>();
        qk.run(q0, k0, page_scores);
        page_scores.store(s0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        tqv_page_softmax<G>(
            scores, probabilities, row_max, row_sum, previous_scale, rescale,
            reinterpret_cast<device const float4 *>(k_norms + first),
            reinterpret_cast<device const float4 *>(v_norms + first),
            score_scale, token_start, lane.committed, lane.rows, thread_index);
        tqv_dequant_page(v_codes + first * D, staged, thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
            for (ushort index = 0; index < running.get_capacity(); ++index) {
                if (!running_full && !running.is_valid_element(index))
                    continue;
                auto coordinates = running.get_multidimensional_index(index);
                running[index] *= previous_scale[coordinates[1]];
            }
        }
        pv.run(p0, k0, running);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ulong slot = tile * params.slot_splits + split;
    auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                         array<int, 2>{1, D});
    running.store(target.slice<D, M>(0, 0));
    if (thread_index < M) {
        statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
        statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
    }
}

#define TQV_SPLIT(Name, G)                                                     \
  kernel void Name(                                                            \
      constant TqHbVerifyParams &params [[buffer(0)]],                         \
      constant TqHbVerifyLane *lanes [[buffer(1)]],                            \
      device half *queries [[buffer(2)]],                                      \
      device const uchar *k_codes [[buffer(3)]],                               \
      device const float *k_norms [[buffer(4)]],                               \
      device const uchar *v_codes [[buffer(5)]],                               \
      device const float *v_norms [[buffer(6)]],                               \
      device float *partials [[buffer(7)]],                                    \
      device float *statistics [[buffer(8)]],                                  \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    alignas(16) threadgroup half staged[TQV_PAGE * TQV_DIM];                   \
    alignas(16) threadgroup float scores[TQV_ROWS * G * TQV_PAGE];             \
    alignas(16) threadgroup half probabilities[TQV_ROWS * G * TQV_PAGE];       \
    threadgroup float row_max[TQV_ROWS * G];                                   \
    threadgroup float row_sum[TQV_ROWS * G];                                   \
    threadgroup float previous_scale[TQV_ROWS * G];                            \
    threadgroup atomic_uint rescale;                                           \
    tqv_split<G>(params, lanes, queries, k_codes, k_norms, v_codes, v_norms,   \
                 partials, statistics, staged, scores, probabilities, row_max, \
                 row_sum, previous_scale, &rescale, group, thread_index);      \
  }

TQV_SPLIT(tq_hb_verify_tile_split_g6, 6)
TQV_SPLIT(tq_hb_verify_tile_split_g8, 8)
#undef TQV_SPLIT

// ---------------------------------------------------------------------------
// Reduce: 256 threads (one per dimension) per (query head, window row, lane).
// ---------------------------------------------------------------------------
template <uint G>
inline void tqv_reduce(
    constant TqHbVerifyParams &params,
    constant TqHbVerifyLane *lanes,
    device const float *partials, device const float *statistics,
    device float *attention,
    threadgroup float *weights, threadgroup float *group_values,
    uint3 group, uint thread_index)
{
    constexpr uint M = TQV_ROWS * G;
    constexpr uint D = TQV_DIM;
    const uint head = group.x, row = group.y;
    const TqHbVerifyLane lane = lanes[group.z];
    if (row >= lane.rows)
        return;
    const uint fused_row = row * G + head % G;
    const uint pages = tqv_pages(lane, params.kv_capacity);
    const uint per_split = tqv_pages_per_split(pages, lane.splits);
    const uint written = (pages + per_split - 1) / per_split;
    const ulong head_slot =
        (ulong(group.z) * params.n_kv_heads + head / G) * params.slot_splits;

    const uint lane32 = thread_index % 32, sg = thread_index / 32;
    const float maximum = thread_index < written
        ? statistics[((head_slot + thread_index) * M + fused_row) * 2]
        : -INFINITY;
    const float group_maximum = simd_max(maximum);
    if (lane32 == 0) group_values[sg] = group_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_maximum = simd_max(lane32 < D / 32 ? group_values[lane32] : -INFINITY);
    if (thread_index < written)
        weights[thread_index] = exp(maximum - row_maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Both accumulations run in split order.
    float numerator = 0.0f, denominator = 0.0f;
    for (uint split = 0; split < written; ++split) {
        const float weight = weights[split];
        const ulong cell = (head_slot + split) * M + fused_row;
        numerator += weight * partials[cell * D + thread_index];
        denominator += weight * statistics[cell * 2 + 1];
    }
    attention[((ulong(group.z) * TQV_ROWS + row) * params.n_heads + head) * D + thread_index] =
        denominator > 0.0f ? numerator / denominator : 0.0f;
}

#define TQV_REDUCE(Name, G)                                                    \
  kernel void Name(                                                            \
      constant TqHbVerifyParams &params [[buffer(0)]],                         \
      constant TqHbVerifyLane *lanes [[buffer(1)]],                            \
      device const float *partials [[buffer(2)]],                              \
      device const float *statistics [[buffer(3)]],                            \
      device float *attention [[buffer(4)]],                                   \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[TQV_MAX_SPLITS];                                 \
    threadgroup float group_values[TQV_DIM / 32];                              \
    tqv_reduce<G>(params, lanes, partials, statistics, attention, weights,     \
                  group_values, group, thread_index);                          \
  }

TQV_REDUCE(tq_hb_verify_tile_reduce_g6, 6)
TQV_REDUCE(tq_hb_verify_tile_reduce_g8, 8)
#undef TQV_REDUCE

// ---------------------------------------------------------------------------
// Un-rotation + sigmoid gate: one simdgroup per (query head, packed row).
//
// The statements of fwht_sign_undo_sigmoid_mul_fast<256>
// (fwht_standalone.metal), so a row's gated output is bitwise that kernel's
// on the same rotated row; only the rotated row's address differs.
// ---------------------------------------------------------------------------
inline void tqv_butterfly_local(thread float &a, thread float &b) {
    float sum = a + b;
    float diff = a - b;
    a = sum;
    b = diff;
}

template<ushort EPT>
inline void tqv_fwht_simd(thread float *elems, uint lane) {
    for (ushort h = 1; h < EPT; h <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            ushort partner = i ^ h;
            if (partner > i) {
                tqv_butterfly_local(elems[i], elems[partner]);
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

kernel void tq_hb_verify_tile_undo_gate(
    constant TqHbVerifyParams &params [[buffer(0)]],
    constant uint             *row_map [[buffer(1)]],
    device const float        *attention [[buffer(2)]],
    device const float        *gate [[buffer(3)]],
    device float              *output [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort HEAD_DIM = TQV_DIM;
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint lane = tiisg;
    // A packed row's rotated row: lane * TQV_ROWS + window row.
    const ulong source = (ulong(row_map[group.y]) * params.n_heads + group.x) * HEAD_DIM + lane * EPT;
    const ulong base = (ulong(group.y) * params.n_heads + group.x) * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = attention[source + i];
    }

    tqv_fwht_simd<EPT>(elems, lane);
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= inv_sqrt_d;
    }
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = TQV_SIGNS_256[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    for (ushort i = 0; i < EPT; i++) {
        const float g = gate[base + i];
        const float s = 1.0f / (1.0f + metal::exp(-g));
        output[base + i] = elems[i] * s;
    }
}
