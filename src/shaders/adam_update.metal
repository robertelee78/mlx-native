#include <metal_stdlib>
using namespace metal;

/// In-place Adam optimizer step.
///
/// Updates `param`, `m`, `v` in-place per the standard Adam algorithm:
///
///   m_new   = β1·m + (1−β1)·g
///   v_new   = β2·v + (1−β2)·g²
///   m_hat   = m_new / (1 − β1^t)
///   v_hat   = v_new / (1 − β2^t)
///   θ_new   = θ − lr · m_hat / (sqrt(v_hat) + ε)
///
/// Buffer layout:
///   buffer(0): param   — float[n] (in/out)
///   buffer(1): grad    — float[n]
///   buffer(2): m       — float[n] (in/out)
///   buffer(3): v       — float[n] (in/out)
///   buffer(4): params  — float[6]: (lr, beta1, beta2, eps,
///                                   one_minus_beta1_pow_t,
///                                   one_minus_beta2_pow_t)
///   buffer(5): meta    — uint[1]: (n_elements)
///
/// Caller pre-computes `1 − β1^t` and `1 − β2^t` for the bias-correction
/// denominators (faster than per-element pow on GPU; the host loop's
/// step counter `t` lives outside the kernel anyway).
///
/// Grid: 1D threads across n.
kernel void adam_update_f32(
    device float       *param  [[buffer(0)]],
    device const float *grad   [[buffer(1)]],
    device float       *m      [[buffer(2)]],
    device float       *v      [[buffer(3)]],
    device const float *params [[buffer(4)]],
    device const uint  *meta   [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint n = meta[0];
    if (tid >= n) return;

    const float lr      = params[0];
    const float beta1   = params[1];
    const float beta2   = params[2];
    const float eps     = params[3];
    const float omb1_t  = params[4]; // 1 − β1^t
    const float omb2_t  = params[5]; // 1 − β2^t

    const float g = grad[tid];
    const float m_new = beta1 * m[tid] + (1.0f - beta1) * g;
    const float v_new = beta2 * v[tid] + (1.0f - beta2) * g * g;
    m[tid] = m_new;
    v[tid] = v_new;

    const float m_hat = m_new / omb1_t;
    const float v_hat = v_new / omb2_t;
    param[tid] = param[tid] - lr * m_hat / (metal::sqrt(v_hat) + eps);
}
