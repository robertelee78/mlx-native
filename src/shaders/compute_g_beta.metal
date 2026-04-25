/// Compute g and beta from alpha_logit, beta_logit, dt_bias, ssm_a.
///
/// g[t, vh]    = softplus(alpha_logit[t, vh] + dt_bias[vh]) * (-ssm_a[vh])
/// beta[t, vh] = sigmoid(beta_logit[t, vh])
///
/// All buffers are f32. Shapes: alpha_logit/beta_logit [seq, nv], g/beta [seq, nv],
/// dt_bias/ssm_a [nv].
///
/// Grid: dispatch `seq * nv` threads. Each thread handles one (t, vh) element.

#include <metal_stdlib>
using namespace metal;

kernel void compute_g_beta_f32(
    device const float *alpha_logit [[buffer(0)]],  // [seq, nv]
    device const float *beta_logit  [[buffer(1)]],  // [seq, nv]
    device const float *dt_bias     [[buffer(2)]],  // [nv]
    device const float *ssm_a       [[buffer(3)]],  // [nv] — stores -exp(A_log)
    device       float *g_out       [[buffer(4)]],  // [seq, nv]
    device       float *beta_out    [[buffer(5)]],  // [seq, nv]
    device const uint  *params      [[buffer(6)]],  // [seq, nv] packed as two u32
    uint tid [[thread_position_in_grid]]
) {
    const uint nv  = params[0];
    const uint seq = params[1];
    const uint n   = seq * nv;
    if (tid >= n) return;

    const uint t  = tid / nv;
    const uint vh = tid % nv;

    // g = softplus(alpha_logit + dt_bias) * (-ssm_a)
    // softplus(x) = log(1 + exp(x))
    // ssm_a stores -exp(A_log), so -ssm_a = exp(A_log) > 0.
    // Use precise:: math to match CPU f32 results (avoids fast-math ULP drift).
    const float a_logit = alpha_logit[t * nv + vh] + dt_bias[vh];
    float sp;
    if (a_logit > 20.0f) {
        sp = a_logit;
    } else if (a_logit < -20.0f) {
        sp = 0.0f;
    } else {
        sp = precise::log(1.0f + precise::exp(a_logit));
    }
    g_out[tid] = sp * (-ssm_a[vh]);

    // beta = sigmoid(beta_logit)
    beta_out[tid] = 1.0f / (1.0f + precise::exp(-beta_logit[t * nv + vh]));
}
