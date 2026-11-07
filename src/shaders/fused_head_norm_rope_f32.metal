#include <metal_stdlib>
using namespace metal;

/// Fused per-head RMS normalization + NeoX rotary position embedding (float32).
///
/// Replaces two separate kernel dispatches (rms_norm_f32 per-head + rope_neox_f32)
/// with a single kernel launch per Q or K projection.
///
/// Supports optional freq_factors for Gemma 4's global attention layers.
/// When freq_factors[pair_idx] is very large (e.g. 1e30), the effective
/// rotation angle approaches zero, producing identity (cos~1, sin~0).
///
/// Layout:
///   Input : [seq_len, n_heads, head_dim] f32 — Q or K after the projection
///   Output: [seq_len, n_heads, head_dim] f32 — normalized and rotated
///
/// Buffer layout:
///   buffer(0): input           — float [rows * head_dim], rows = seq_len * n_heads
///   buffer(1): output          — float [rows * head_dim]
///   buffer(2): norm_weight     — float [head_dim] (ignored when has_weight=0)
///   buffer(3): params          — FusedHeadNormRopeF32Params struct
///   buffer(4): positions       — uint [seq_len]
///   buffer(5): freq_factors    — float [half_rope_dim] (ignored when has_freq_factors=0)
///   buffer(6): output_bf16     — bfloat [rows * head_dim] (ignored when has_bf16_output=0)
///   buffer(7): output_f32_perm — float [rows * head_dim] (ignored when has_f32_perm_output=0)
///
/// ADR-011 Phase 3 Wave P3b.4 — optional bf16 co-write.  When the
/// `has_bf16_output` params flag is set, the kernel writes every f32
/// output element also to a bf16 buffer (at the natural offset, or at the
/// permuted [head, token, i] offset with `bf16_permuted`).  This fuses the
/// f32→bf16 cast and the pre-attention permute_021_bf16 dispatches.
///
/// Thread mapping (one simdgroup per row, the reference implementation's
/// norm / rope pattern):
/// row = threadgroup * simdgroups_per_threadgroup + simdgroup.  There is no
/// threadgroup memory and no barrier: the sum of squares reduces in
/// registers and through simdgroup shuffles, and each lane rotates the pairs
/// (i, i + head_dim/2) it loaded itself.
///   * `vec4` (head_dim % 256 == 0, 16-byte aligned buffers): lane l owns
///     elements 4l..4l+3 of every 128-element block — float4 loads and
///     stores, BF16 as 8-byte bfloat4 stores; a simdgroup instruction moves
///     512 contiguous bytes.
///   * otherwise lane l owns elements l, l+32, l+64, ...
///
/// Bitwise contract: the outputs equal those of the earlier kernel (one
/// threadgroup of T = min(256, next_pow2(head_dim)) threads per row,
/// threadgroup-memory tree, cooperative rotation) as this toolchain compiled
/// it.  The order-carrying parts of its f32 operation sequence are written
/// out explicitly, with contraction and reassociation off so no compiler can
/// re-fuse them:
///   * slot t of T accumulates fma(x, x, s) over x[t], x[t+T], ... from
///     s = 0 (the first term is the rounded square);
///   * a tree of plain adds halves the slots, stride T/2 .. 1;
///   * o0 = fma(y0, cos, -(y1 * sin)), o1 = fma(y0, sin, y1 * cos).
/// The remaining expressions — rsqrt(sum / head_dim + eps), y = x * rms_inv
/// * w and the angle pos / pow(theta, 2i / head_dim) / ff — are the earlier
/// kernel's source expressions under the same default (fast) math lowering;
/// a pragma there would change the division lowering for non-power-of-two
/// head_dim.
///
/// Threadgroup: (32 * rows_per_threadgroup, 1, 1)
/// Grid       : (ceil(seq_len * n_heads / rows_per_threadgroup), 1, 1)

struct FusedHeadNormRopeF32Params {
    uint  head_dim;
    uint  n_heads;
    uint  half_rope_dim;    // may be < head_dim/2 for partial rotary
    float eps;
    uint  has_weight;       // 0 = no weight scale (V-norm variant), nonzero = apply weight
    float theta;            // RoPE base frequency (e.g. 10000 or 1000000)
    uint  has_freq_factors; // nonzero if freq_factors buffer is valid
    uint  has_bf16_output;  // ADR-011 P3b.4 — nonzero to co-write bf16 output
    uint  bf16_permuted;    // P4.15 — if 1, bf16 output written at permuted
                            // [head, token, i] index instead of natural
                            // [token, head, i] index, fusing the
                            // permute_021_bf16 pre-FA dispatch.
    uint  seq_len;          // P4.15 — needed to compute permuted index
    uint  has_f32_perm_output; // D.1 — nonzero to co-write f32 output at the
                               // permuted [head, token, i] layout (same
                               // index math as bf16_permuted).  Used by
                               // HF2Q_NO_FA so the tensor-mm Q@K^T can
                               // consume Q directly without an intermediate
                               // bf16→f32 cast dispatch.
    uint  vec4;             // fused_head_norm_rope_f32: nonzero selects the
                            // float4 lane mapping (head_dim % 256 == 0 and
                            // every bound buffer 16-byte aligned).
};

/// Number of reduction slots of the earlier one-threadgroup-per-row kernel:
/// its threadgroup size min(256, next_pow2(dim)).
static inline uint row_reduction_slots(const uint dim) {
    const uint pow2 = dim <= 1u ? 1u : (1u << (32u - clz(dim - 1u)));
    return min(256u, pow2);
}

/// Sum of squares of x[0..dim) for one simdgroup, lane l owning slots
/// l + 32k; every lane returns the total.
static inline float row_sum_sq_tree(device const float* x, const uint dim, const ushort lane) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const uint slots = row_reduction_slots(dim);
    const uint slots_per_lane = slots >= 32u ? slots / 32u : 1u;
    float partial[8];
    for (uint k = 0; k < 8u; ++k) {
        float sum_sq = 0.0f;
        const uint slot = uint(lane) + 32u * k;
        if (k < slots_per_lane && slot < slots) {
            for (uint i = slot; i < dim; i += slots) {
                const float val = x[i];
                sum_sq = fma(val, val, sum_sq);
            }
        }
        partial[k] = sum_sq;
    }
    // Tree strides >= 32 stay inside the lane: slot l+32k pairs with l+32(k+s/32).
    if (slots_per_lane == 8u) {
        partial[0] += partial[4];
        partial[1] += partial[5];
        partial[2] += partial[6];
        partial[3] += partial[7];
    }
    if (slots_per_lane >= 4u) {
        partial[0] += partial[2];
        partial[1] += partial[3];
    }
    if (slots_per_lane >= 2u) {
        partial[0] += partial[1];
    }
    // Strides 16 .. 1 pair slot l with slot l+s of lane l+s.
    float sum_sq = partial[0];
    for (ushort stride = ushort(min(16u, slots / 2u)); stride > 0; stride >>= 1) {
        sum_sq += simd_shuffle_down(sum_sq, stride);
    }
    return simd_broadcast(sum_sq, 0);
}

/// float4 form for dim % 256 == 0 (T = 256 slots): lane l owns slots
/// 4l+c (block parity 0) and 128+4l+c (parity 1) of every 256-element span;
/// x4 holds dim / 128 blocks of 32 float4.
static inline float row_sum_sq_tree4(device const float4* x4, const uint blocks, const ushort lane) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    float4 even = float4(0.0f);
    float4 odd = float4(0.0f);
    for (uint block = 0; block < blocks; block += 2u) {
        const float4 v0 = x4[32u * block + lane];
        const float4 v1 = x4[32u * (block + 1u) + lane];
        even = fma(v0, v0, even);
        odd = fma(v1, v1, odd);
    }
    even += odd;                                  // stride 128
    even += simd_shuffle_down(even, ushort(16));  // 64
    even += simd_shuffle_down(even, ushort(8));   // 32
    even += simd_shuffle_down(even, ushort(4));   // 16
    even += simd_shuffle_down(even, ushort(2));   // 8
    even += simd_shuffle_down(even, ushort(1));   // 4
    const float lo = even.x + even.z;             // 2
    const float hi = even.y + even.w;
    return simd_broadcast(lo + hi, 0);            // 1
}

/// NeoX angle of pair i: pos / theta^(2i / head_dim) [/ freq_factors[i]],
/// head_dim as the denominator (mlx-lm ProportionalRoPE).
static inline float rope_angle(
    const uint pos, const uint i, const uint head_dim, const float theta,
    const bool has_ff, device const float* freq_factors
) {
    const float dim_ratio = float(2 * i) / float(head_dim);
    float freq = float(pos) / pow(theta, dim_ratio);
    if (has_ff) {
        freq /= freq_factors[i];
    }
    return freq;
}

/// Rotated pair (y0, y1) by the angle with this cos / sin.
static inline float2 rope_rotate(const float y0, const float y1, const float cos_a, const float sin_a) {
#pragma clang fp contract(off)
    return float2(fma(y0, cos_a, -(y1 * sin_a)), fma(y0, sin_a, y1 * cos_a));
}

kernel void fused_head_norm_rope_f32(
    device const float*                      input            [[buffer(0)]],
    device float*                            output           [[buffer(1)]],
    device const float*                      norm_weight      [[buffer(2)]],
    constant FusedHeadNormRopeF32Params&     params           [[buffer(3)]],
    device const uint*                       positions        [[buffer(4)]],
    device const float*                      freq_factors     [[buffer(5)]],
    device bfloat*                           output_bf16      [[buffer(6)]],
    device float*                            output_f32_perm  [[buffer(7)]],
    uint   tgid     [[threadgroup_position_in_grid]],
    ushort sgitg    [[simdgroup_index_in_threadgroup]],
    ushort n_sg     [[simdgroups_per_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]]
) {
    const uint n_heads = params.n_heads;
    const uint row = tgid * uint(n_sg) + uint(sgitg);
    if (row >= n_heads * params.seq_len) {
        return;
    }
    const bool has_bf16     = (params.has_bf16_output != 0u);
    const bool has_f32_perm = (params.has_f32_perm_output != 0u);
    const uint head_dim     = params.head_dim;
    const uint half_rope    = params.half_rope_dim;
    const uint half_dim     = head_dim / 2;
    const bool has_weight   = (params.has_weight != 0u);
    const float theta       = params.theta;
    const bool has_ff       = (params.has_freq_factors != 0u);
    const bool bf16_perm    = (params.bf16_permuted != 0u);

    // row = token * n_heads + head.
    const uint seq_idx = row / n_heads;
    const uint pos = positions[seq_idx];
    const uint base = row * head_dim;
    // P4.15 — the permuted outputs are [head, token, i]:
    //   head * seq_len * head_dim + token * head_dim + i.
    const uint head_within_token = row % n_heads;
    const uint base_bf16 = bf16_perm
        ? (head_within_token * params.seq_len * head_dim + seq_idx * head_dim)
        : base;

    // NeoX rotation on pairs (i, i + half_dim) for i < half_rope; the pairs
    // [half_rope, half_dim) pass through normalized.
    if (params.vec4 != 0u) {
        device const float4* x4 = (device const float4*)(input + base);
        const float rms_inv =
            rsqrt(row_sum_sq_tree4(x4, head_dim / 128u, lane) / float(head_dim) + params.eps);
        device const float4* w4 = (device const float4*)norm_weight;
        device float4* out4 = (device float4*)(output + base);
        device bfloat4* bf4 = (device bfloat4*)(output_bf16 + base_bf16);
        device float4* perm4 = (device float4*)(output_f32_perm + base_bf16);
        const uint half4 = half_dim / 4u;
        for (uint q0 = lane; q0 < half4; q0 += 32u) {
            const uint q1 = q0 + half4;
            float4 y0 = x4[q0] * rms_inv;
            float4 y1 = x4[q1] * rms_inv;
            if (has_weight) {
                y0 *= w4[q0];
                y1 *= w4[q1];
            }
            float4 o0 = y0;
            float4 o1 = y1;
            for (uint c = 0; c < 4u; ++c) {
                const uint i = 4u * q0 + c;
                if (i < half_rope) {
                    const float freq = rope_angle(pos, i, head_dim, theta, has_ff, freq_factors);
                    const float2 o = rope_rotate(y0[c], y1[c], cos(freq), sin(freq));
                    o0[c] = o.x;
                    o1[c] = o.y;
                }
            }
            out4[q0] = o0;
            out4[q1] = o1;
            if (has_bf16) {
                bf4[q0] = bfloat4(o0);
                bf4[q1] = bfloat4(o1);
            }
            if (has_f32_perm) {
                perm4[q0] = o0;
                perm4[q1] = o1;
            }
        }
        return;
    }

    device const float* x = input + base;
    const float rms_inv = rsqrt(row_sum_sq_tree(x, head_dim, lane) / float(head_dim) + params.eps);
    for (uint i = lane; i < half_dim; i += 32u) {
        float y0 = x[i] * rms_inv;
        float y1 = x[i + half_dim] * rms_inv;
        if (has_weight) {
            y0 *= norm_weight[i];
            y1 *= norm_weight[i + half_dim];
        }
        float o0 = y0;
        float o1 = y1;
        if (i < half_rope) {
            const float freq = rope_angle(pos, i, head_dim, theta, has_ff, freq_factors);
            const float2 o = rope_rotate(y0, y1, cos(freq), sin(freq));
            o0 = o.x;
            o1 = o.y;
        }
        output[base + i]            = o0;
        output[base + i + half_dim] = o1;
        if (has_bf16) {
            output_bf16[base_bf16 + i]            = bfloat(o0);
            output_bf16[base_bf16 + i + half_dim] = bfloat(o1);
        }
        if (has_f32_perm) {
            output_f32_perm[base_bf16 + i]            = o0;
            output_f32_perm[base_bf16 + i + half_dim] = o1;
        }
    }
}

// ---------------------------------------------------------------------------
// fused_head_norm_rope_f32_v2 (ADR-028 iter-337) — decode-path variant.
//
// One threadgroup of min(256, next_pow2(head_dim)) threads per head with:
//   1. float4 vector loads (4× memory throughput per thread)
//   2. simd_sum() in-simdgroup reduction (1 HW op, no barrier)
//   3. inter-simdgroup shuffle via shared memory (2 barriers total)
//
// Phases 2-4 (normalize+store_to_shared, NeoX RoPE rotation, pass-through
// non-rotated dims) evaluate the same expressions as
// `fused_head_norm_rope_f32`; the load-bearing race-fix barrier at the
// Phase 1→Phase 2 transition is preserved per
// docs/spike-batched-prefill-race-rootcause.md.
//
// REQUIREMENT: `head_dim % 4 == 0`.  All hf2q production shapes meet this
// (gemma4 head_dim=256, qwen3.6 head_dim=128).  Dispatcher must guard or
// fall back to scalar.
//
// Numerically equivalent to `fused_head_norm_rope_f32` within f32
// accumulation tolerance (~1e-4).  Not bit-exact due to simd_sum vs
// tree-reduce non-associativity.
kernel void fused_head_norm_rope_f32_v2(
    device const float*                      input            [[buffer(0)]],
    device float*                            output           [[buffer(1)]],
    device const float*                      norm_weight      [[buffer(2)]],
    constant FusedHeadNormRopeF32Params&     params           [[buffer(3)]],
    device const uint*                       positions        [[buffer(4)]],
    device const float*                      freq_factors     [[buffer(5)]],
    device bfloat*                           output_bf16      [[buffer(6)]],
    device float*                            output_f32_perm  [[buffer(7)]],
    uint head_id  [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    threadgroup float* shared             [[threadgroup(0)]]
) {
    const bool has_bf16 = (params.has_bf16_output != 0u);
    const bool has_f32_perm = (params.has_f32_perm_output != 0u);
    const uint head_dim     = params.head_dim;
    const uint half_rope    = params.half_rope_dim;
    const uint half_dim     = head_dim / 2;
    const float eps         = params.eps;
    const bool has_weight   = (params.has_weight != 0u);
    const float theta       = params.theta;
    const bool has_ff       = (params.has_freq_factors != 0u);
    const uint n_heads      = params.n_heads;
    const bool bf16_perm    = (params.bf16_permuted != 0u);
    const uint seq_len_v    = params.seq_len;

    const uint seq_idx = head_id / n_heads;
    const uint pos = positions[seq_idx];

    const uint base = head_id * head_dim;
    const uint head_within_token = head_id % n_heads;
    const uint base_bf16 = bf16_perm
        ? (head_within_token * seq_len_v * head_dim + seq_idx * head_dim)
        : base;

    // -------------------------------------------------------------------------
    // Phase 1 (V2): float4 sum-of-squares + simd_sum + 2-barrier inter-SG
    // -------------------------------------------------------------------------
    const uint head_dim4 = head_dim / 4u;
    device const float4* input4 = (device const float4*)(input + base);

    float partial_sq = 0.0f;
    for (uint i = tid; i < head_dim4; i += tg_size) {
        const float4 v = input4[i];
        partial_sq += dot(v, v);
    }

    // In-simdgroup reduction (1 HW op).
    float sumf = simd_sum(partial_sq);

    // Stage per-simdgroup partial sums via threadgroup memory.
    if (tiisg == 0) {
        shared[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across simdgroups in the first simdgroup.
    const uint n_sg = tg_size / 32u;
    if (sgitg == 0) {
        const float vsg = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        const float total = simd_sum(vsg);
        if (tiisg == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_inv = rsqrt(shared[0] / float(head_dim) + eps);

    // CRITICAL: load-bearing race-fix barrier (preserved from v1 line 148).
    // All threads must complete the broadcast-read of shared[0] for rms_inv
    // BEFORE any thread overwrites shared[0] in Phase 2's `shared[i] = val`
    // write.  Removing this causes ~80% broken at head_dim=256, n_tokens*
    // n_heads >= 2000 threadgroups on Apple Silicon (per v1 line 135-147).
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 2 (UNCHANGED from v1): normalize, optional weight, store in shared
    // -------------------------------------------------------------------------
    for (uint i = tid; i < head_dim; i += tg_size) {
        float val = input[base + i] * rms_inv;
        if (has_weight) {
            val *= norm_weight[i];
        }
        shared[i] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 3 (UNCHANGED from v1): NeoX rotation on first 2*half_rope dims
    // -------------------------------------------------------------------------
    for (uint i = tid; i < half_rope; i += tg_size) {
        const float x0 = shared[i];
        const float x1 = shared[i + half_dim];

        const float dim_ratio = float(2 * i) / float(head_dim);
        float freq = float(pos) / pow(theta, dim_ratio);

        if (has_ff) {
            freq /= freq_factors[i];
        }

        const float cos_a = cos(freq);
        const float sin_a = sin(freq);

        const float o0 = x0 * cos_a - x1 * sin_a;
        const float o1 = x1 * cos_a + x0 * sin_a;
        output[base + i]            = o0;
        output[base + i + half_dim] = o1;
        if (has_bf16) {
            output_bf16[base_bf16 + i]            = bfloat(o0);
            output_bf16[base_bf16 + i + half_dim] = bfloat(o1);
        }
        if (has_f32_perm) {
            output_f32_perm[base_bf16 + i]            = o0;
            output_f32_perm[base_bf16 + i + half_dim] = o1;
        }
    }

    // -------------------------------------------------------------------------
    // Phase 4 (UNCHANGED from v1): pass-through non-rotated dims
    // -------------------------------------------------------------------------
    for (uint i = tid; i < half_dim - half_rope; i += tg_size) {
        uint src = half_rope + i;
        const float o0 = shared[src];
        const float o1 = shared[src + half_dim];
        output[base + src]            = o0;
        output[base + src + half_dim] = o1;
        if (has_bf16) {
            output_bf16[base_bf16 + src]            = bfloat(o0);
            output_bf16[base_bf16 + src + half_dim] = bfloat(o1);
        }
        if (has_f32_perm) {
            output_f32_perm[base_bf16 + src]            = o0;
            output_f32_perm[base_bf16 + src + half_dim] = o1;
        }
    }
}
