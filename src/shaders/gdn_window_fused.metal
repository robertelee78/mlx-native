// Gated DeltaNet core of a joint verify step, every lane window in one pass.
//
// Portions of this file are derived from Splash
// (https://github.com/incoai/splash, commit 031a25a6), Apache-2.0 licensed.
// Source: runtime/metal/kernels/decode/gdn.metal (the fused verify kernel:
// one threadgroup per value head and lane, a prologue, a scan and a gate
// phase that hand each other their operands in threadgroup memory, one
// simdgroup per verify row in the prologue and the gate, the head's state
// rows strided over the simdgroups in the scan).
// Modifications: the arithmetic is that of the stock hf2q kernels, operation
// for operation (ssm_conv_forward_f32, l2_norm_f32 and scalar_mul_f32,
// compute_g_beta_f32, gated_delta_net_decode_f32_4, ssm_norm_gate_f32), with
// F32 operands, F32 state and their tree reductions, so every output bit is
// the stock chain's; hf2q state layout with per-slot ping-pong buffers; the
// step writes the state after its rows and a replay tape of its inputs
// instead of folding the previous step's rows; tiled key-head mapping;
// padding lanes.
//
// Copyright the Splash authors. See LICENSE-APACHE-splash at the repository
// root.
//
// Grid (value heads, lanes), 256 threads: eight simdgroups of 32.
//
// A lane is a window of 1-8 consecutive packed rows of one cache slot.
// Thread `lane` of simdgroup `sg` owns head dimensions `32 g + lane`,
// g = 0..3, of window row `sg` in the prologue and the gate, and state cells
// `4 lane + j`, j = 0..3, of state rows `8 b + sg`, b = 0..15, in the scan.
//
// Layouts (F32 unless noted, elements):
// - qkv    [width][qkv_stride]: q (n_k_heads x 128), k, v (n_v_heads x 128)
// - z      [width][z_stride]:   n_v_heads x 128
// - alpha, beta logits [width][stride]: n_v_heads
// - conv_w [channels][k_width]
// - conv state A / B  [slots][channels][k_width - 1]
// - recurrent A / B   [slots][n_v_heads][128 state rows][128]
// - gated  [width][out_stride] F32 or BF16: n_v_heads x 128
// - tape   one lane's retained rows at the lane's offsets: scaled q and
//   normalized k [rows][n_k_heads x 128], v [rows][n_v_heads x 128], raw qkv
//   [rows][channels], g and beta [rows][n_v_heads]
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kGdnwHeadDim = 128;
constant constexpr uint kGdnwRows = 8;
constant constexpr uint kGdnwGroups = kGdnwHeadDim / 32;
constant constexpr uint kGdnwCells = kGdnwHeadDim / 32;
constant constexpr uint kGdnwRetain = 1;
constant constexpr uint kGdnwPadding = 2;

struct GdnWindowParams {
    uint n_k_heads;
    uint n_v_heads;
    uint channels;
    uint k_width;
    uint qkv_stride;
    uint z_stride;
    uint alpha_stride;
    uint beta_stride;
    uint out_stride;
    float eps;
    float q_scale;
    float head_dim;
};

struct GdnWindowLane {
    uint first_row;
    uint rows;
    uint flags;
    uint conv_in;
    uint conv_offset;
    uint rec_in;
    uint rec_offset;
    uint tape_q;
    uint tape_k;
    uint tape_v;
    uint tape_conv;
    uint tape_g;
    uint tape_beta;
    uint reserved[3];
};

// Threadgroup operands of one value head of one lane.
struct GdnWindowShared {
    float q[kGdnwRows * kGdnwHeadDim];
    float k[kGdnwRows * kGdnwHeadDim];
    float v[kGdnwRows * kGdnwHeadDim];
    float y[kGdnwRows * kGdnwHeadDim];
    float g[kGdnwRows];
    float beta[kGdnwRows];
};

// ssm_conv_forward_f32 for channel `c` at window row `t`: `x` is the
// window's first row, `state` the slot's carried rows.
inline float gdnw_conv_silu(device const float *x, uint x_stride,
                            device const float *kernel_w, device const float *state,
                            uint k_width, uint c, uint t) {
    float sum = 0.0f;
    for (uint k = 0; k < k_width; ++k) {
        const uint t_ext = t + k;
        float val;
        if (t_ext < k_width - 1u) {
            val = state[c * (k_width - 1u) + t_ext];
        } else {
            const uint t_in = t_ext - (k_width - 1u);
            val = x[t_in * x_stride + c];
        }
        sum += kernel_w[c * k_width + k] * val;
    }
    return sum / (1.0f + metal::exp(-sum));
}

// ssm_conv_state_update_f32: carried row `i` after `n_tokens` window rows.
inline float gdnw_conv_carry(device const float *x, uint x_stride, device const float *state,
                             uint k_width, uint n_tokens, uint c, uint i) {
    const uint k_minus1 = k_width - 1u;
    const uint t_ext = n_tokens + i;
    if (t_ext < k_minus1) return state[c * k_minus1 + t_ext];
    return x[(t_ext - k_minus1) * x_stride + c];
}

// Sum of squares of a 128-element row in the order of the 128-thread tree of
// l2_norm_f32 and ssm_norm_gate_f32: element i with i + 64, then with i + 32,
// then the lanes pairwise at strides 16, 8, 4, 2, 1. The caller holds
// elements `lane + 32 g`; every lane of the simdgroup must call this.
inline float gdnw_tree_sum_sq(float x0, float x1, float x2, float x3) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    const float s0 = x0 * x0;
    const float s1 = x1 * x1;
    const float s2 = x2 * x2;
    const float s3 = x3 * x3;
    const float a = s0 + s2;
    const float b = s1 + s3;
    float t = a + b;
    for (ushort stride = 16; stride > 0; stride >>= 1) {
        // Lanes below `stride` add the lane `stride` above; the others' sums
        // are not read again.
        t = t + simd_shuffle_xor(t, stride);
    }
    return simd_broadcast_first(t);
}

// compute_g_beta_f32: (g, beta) of one (row, value head).
inline float2 gdnw_gates(float alpha_logit, float beta_logit, float dt_bias, float ssm_a) {
    const float a_logit = alpha_logit + dt_bias;
    float sp;
    if (a_logit > 20.0f) {
        sp = a_logit;
    } else if (a_logit < -20.0f) {
        sp = 0.0f;
    } else {
        sp = precise::log(1.0f + precise::exp(a_logit));
    }
    const float g = sp * (-ssm_a);
    const float beta = 1.0f / (1.0f + precise::exp(-beta_logit));
    return float2(g, beta);
}

// l2_norm_f32 then scalar_mul_f32: each product is rounded on its own, as the
// two kernels round it through device memory.
inline float gdnw_scale(float value, float inverse) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    return value * inverse;
}

template <typename Out, uint Flight>
inline void gdnw_impl(
    device const float *qkv,
    device const float *z,
    device const float *alpha_logit,
    device const float *beta_logit,
    device const float *conv_w,
    device float *conv_a,
    device float *conv_b,
    device float *rec_a,
    device float *rec_b,
    device const float *dt_bias,
    device const float *ssm_a,
    device const float *norm_w,
    device Out *gated,
    device float *tape,
    constant GdnWindowParams &p,
    constant GdnWindowLane *lanes,
    threadgroup GdnWindowShared &shared,
    uint2 group, uint lane, uint sg) {
    const uint v_head = group.x;
    const GdnWindowLane w = lanes[group.y];
    const uint n = w.rows;
    const uint value_width = p.n_v_heads * kGdnwHeadDim;

    if (w.flags & kGdnwPadding) {
        // Padding rows carry zeros: finite, never read by a real row.
        if (sg < n) {
            device Out *row = gated + ulong(w.first_row + sg) * p.out_stride + v_head * kGdnwHeadDim;
            for (uint g = 0; g < kGdnwGroups; ++g) row[32 * g + lane] = Out(0.0f);
        }
        return;
    }

    const uint k_head = v_head % p.n_k_heads;
    const uint key_width = p.n_k_heads * kGdnwHeadDim;
    // The first value head of a key head writes what the heads share.
    const bool shared_writer = v_head < p.n_k_heads;
    const bool retain = (w.flags & kGdnwRetain) != 0;
    const uint k_minus1 = p.k_width - 1u;

    device const float *x = qkv + ulong(w.first_row) * p.qkv_stride;
    device const float *conv_in = (w.conv_in ? conv_b : conv_a) + w.conv_offset;
    device float *conv_out = (w.conv_in ? conv_a : conv_b) + w.conv_offset;
    device const float *rec_in = (w.rec_in ? rec_b : rec_a) + w.rec_offset;
    device float *rec_out = (w.rec_in ? rec_a : rec_b) + w.rec_offset;

    // ---- Prologue: convolution, q/k norms and gates of window row `sg` ----
    if (sg < n) {
        const uint t = sg;
        float q[kGdnwGroups], k[kGdnwGroups], v[kGdnwGroups];
        for (uint g = 0; g < kGdnwGroups; ++g) {
            const uint dim = 32 * g + lane;
            q[g] = gdnw_conv_silu(x, p.qkv_stride, conv_w, conv_in, p.k_width,
                                  k_head * kGdnwHeadDim + dim, t);
            k[g] = gdnw_conv_silu(x, p.qkv_stride, conv_w, conv_in, p.k_width,
                                  key_width + k_head * kGdnwHeadDim + dim, t);
            v[g] = gdnw_conv_silu(x, p.qkv_stride, conv_w, conv_in, p.k_width,
                                  2 * key_width + v_head * kGdnwHeadDim + dim, t);
        }
        const float q_inv = rsqrt(gdnw_tree_sum_sq(q[0], q[1], q[2], q[3]) + p.eps);
        const float k_inv = rsqrt(gdnw_tree_sum_sq(k[0], k[1], k[2], k[3]) + p.eps);
        for (uint g = 0; g < kGdnwGroups; ++g) {
            const uint dim = 32 * g + lane;
            const float query = gdnw_scale(gdnw_scale(q[g], q_inv), p.q_scale);
            const float key = gdnw_scale(k[g], k_inv);
            shared.q[t * kGdnwHeadDim + dim] = query;
            shared.k[t * kGdnwHeadDim + dim] = key;
            shared.v[t * kGdnwHeadDim + dim] = v[g];
            if (retain) {
                const uint vc = 2 * key_width + v_head * kGdnwHeadDim + dim;
                tape[w.tape_v + t * value_width + v_head * kGdnwHeadDim + dim] = v[g];
                tape[w.tape_conv + t * p.channels + vc] = x[t * p.qkv_stride + vc];
                if (shared_writer) {
                    const uint qc = k_head * kGdnwHeadDim + dim;
                    const uint kc = key_width + qc;
                    tape[w.tape_q + t * key_width + qc] = query;
                    tape[w.tape_k + t * key_width + qc] = key;
                    tape[w.tape_conv + t * p.channels + qc] = x[t * p.qkv_stride + qc];
                    tape[w.tape_conv + t * p.channels + kc] = x[t * p.qkv_stride + kc];
                }
            }
        }
        if (lane == 0) {
            const uint row = w.first_row + t;
            const float2 gates = gdnw_gates(alpha_logit[ulong(row) * p.alpha_stride + v_head],
                                            beta_logit[ulong(row) * p.beta_stride + v_head],
                                            dt_bias[v_head], ssm_a[v_head]);
            shared.g[t] = gates.x;
            shared.beta[t] = gates.y;
            if (retain) {
                tape[w.tape_g + t * p.n_v_heads + v_head] = gates.x;
                tape[w.tape_beta + t * p.n_v_heads + v_head] = gates.y;
            }
        }
    }
    // The rows the slot carries into its next step.
    if (sg < k_minus1) {
        for (uint g = 0; g < kGdnwGroups; ++g) {
            const uint dim = 32 * g + lane;
            const uint vc = 2 * key_width + v_head * kGdnwHeadDim + dim;
            conv_out[vc * k_minus1 + sg] = gdnw_conv_carry(x, p.qkv_stride, conv_in, p.k_width, n, vc, sg);
            if (shared_writer) {
                const uint qc = k_head * kGdnwHeadDim + dim;
                const uint kc = key_width + qc;
                conv_out[qc * k_minus1 + sg] = gdnw_conv_carry(x, p.qkv_stride, conv_in, p.k_width, n, qc, sg);
                conv_out[kc * k_minus1 + sg] = gdnw_conv_carry(x, p.qkv_stride, conv_in, p.k_width, n, kc, sg);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Scan: gated_delta_net_decode_f32_4 over the head's state rows ----
    // `Flight` rows advance together so their reductions and arithmetic
    // overlap; each row keeps the stock kernel's operations and their order.
    // The next rows' state is loaded before these rows' state is stored.
    constexpr uint kBatches = kGdnwHeadDim / kGdnwRows;
    float ls[Flight][kGdnwCells];
    #pragma clang loop unroll(full)
    for (short r = 0; r < short(Flight); ++r) {
        device const float *sp = rec_in + (ulong(v_head) * kGdnwHeadDim + r * kGdnwRows + sg) * kGdnwHeadDim + lane * kGdnwCells;
        #pragma clang loop unroll(full)
        for (short j = 0; j < short(kGdnwCells); ++j) {
            ls[r][j] = sp[j];
        }
    }
    for (uint batch = 0; batch < kBatches; batch += Flight) {
        for (uint t = 0; t < n; ++t) {
            const float beta_val = shared.beta[t];
            const float g_val = shared.g[t];
            const float alpha = metal::exp(-g_val);
            threadgroup const float *kp = shared.k + t * kGdnwHeadDim + lane * kGdnwCells;
            threadgroup const float *qp = shared.q + t * kGdnwHeadDim + lane * kGdnwCells;

            float sk[Flight];
            #pragma clang loop unroll(full)
            for (short r = 0; r < short(Flight); ++r) {
                float partial_sk = 0.0f;
                #pragma clang loop unroll(full)
                for (short j = 0; j < short(kGdnwCells); ++j) {
                    ls[r][j] *= alpha;
                    partial_sk += ls[r][j] * kp[j];
                }
                sk[r] = metal::simd_sum(partial_sk);
            }
            #pragma clang loop unroll(full)
            for (short r = 0; r < short(Flight); ++r) {
                const uint i20 = (batch + r) * kGdnwRows + sg;
                const float delta = (shared.v[t * kGdnwHeadDim + i20] - sk[r]) * beta_val;

                float partial_y = 0.0f;
                #pragma clang loop unroll(full)
                for (short j = 0; j < short(kGdnwCells); ++j) {
                    ls[r][j] += delta * kp[j];
                    partial_y += ls[r][j] * qp[j];
                }
                const float y = metal::simd_sum(partial_y);
                if (lane == 0) {
                    shared.y[t * kGdnwHeadDim + i20] = y;
                }
            }
        }
        float upcoming[Flight][kGdnwCells];
        const bool more = batch + Flight < kBatches;
        #pragma clang loop unroll(full)
        for (short r = 0; r < short(Flight); ++r) {
            const uint next = more ? (batch + Flight + r) * kGdnwRows + sg : 0u;
            device const float *sp = rec_in + (ulong(v_head) * kGdnwHeadDim + next) * kGdnwHeadDim + lane * kGdnwCells;
            #pragma clang loop unroll(full)
            for (short j = 0; j < short(kGdnwCells); ++j) {
                upcoming[r][j] = more ? sp[j] : 0.0f;
            }
        }
        #pragma clang loop unroll(full)
        for (short r = 0; r < short(Flight); ++r) {
            device float *sp = rec_out + (ulong(v_head) * kGdnwHeadDim + (batch + r) * kGdnwRows + sg) * kGdnwHeadDim + lane * kGdnwCells;
            #pragma clang loop unroll(full)
            for (short j = 0; j < short(kGdnwCells); ++j) {
                sp[j] = ls[r][j];
                ls[r][j] = upcoming[r][j];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Gate: ssm_norm_gate_f32 of window row `sg` ----
    if (sg < n) {
        const uint t = sg;
        const ulong row = w.first_row + t;
        float value[kGdnwGroups];
        for (uint g = 0; g < kGdnwGroups; ++g) {
            value[g] = shared.y[t * kGdnwHeadDim + 32 * g + lane];
        }
        const float rms_inv =
            rsqrt(gdnw_tree_sum_sq(value[0], value[1], value[2], value[3]) / p.head_dim + p.eps);
        device const float *gate = z + row * p.z_stride + v_head * kGdnwHeadDim;
        device Out *out = gated + row * p.out_stride + v_head * kGdnwHeadDim;
        for (uint g = 0; g < kGdnwGroups; ++g) {
            const uint dim = 32 * g + lane;
            const float normed = value[g] * rms_inv * norm_w[dim];
            const float zv = gate[dim];
            const float sig = 1.0f / (1.0f + precise::exp(-zv));
            out[dim] = Out(normed * (zv * sig));
        }
    }
}

#define GDN_WINDOW_KERNEL(NAME, OUT, FLIGHT) \
    kernel void NAME( \
        device const float *qkv [[buffer(0)]], \
        device const float *z [[buffer(1)]], \
        device const float *alpha_logit [[buffer(2)]], \
        device const float *beta_logit [[buffer(3)]], \
        device const float *conv_w [[buffer(4)]], \
        device float *conv_a [[buffer(5)]], \
        device float *conv_b [[buffer(6)]], \
        device float *rec_a [[buffer(7)]], \
        device float *rec_b [[buffer(8)]], \
        device const float *dt_bias [[buffer(9)]], \
        device const float *ssm_a [[buffer(10)]], \
        device const float *norm_w [[buffer(11)]], \
        device OUT *gated [[buffer(12)]], \
        device float *tape [[buffer(13)]], \
        constant GdnWindowParams &p [[buffer(14)]], \
        constant GdnWindowLane *lanes [[buffer(15)]], \
        threadgroup GdnWindowShared *shared [[threadgroup(0)]], \
        uint2 group [[threadgroup_position_in_grid]], \
        uint lane [[thread_index_in_simdgroup]], \
        uint sg [[simdgroup_index_in_threadgroup]]) { \
        gdnw_impl<OUT, FLIGHT>(qkv, z, alpha_logit, beta_logit, conv_w, conv_a, conv_b, rec_a, rec_b, \
                               dt_bias, ssm_a, norm_w, gated, tape, p, lanes, *shared, group, lane, sg); \
    }

// Two rows in flight: 0.068 ms per Qwen3.8 layer at four lanes against 0.070
// with one and 0.069 with four (M5 Max, DRAM-cold state).
GDN_WINDOW_KERNEL(gdn_window_fused_f32, float, 2)
GDN_WINDOW_KERNEL(gdn_window_fused_bf16, bfloat, 2)
