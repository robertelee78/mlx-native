#include <metal_stdlib>
using namespace metal;

/// Temperature-scaled softmax + categorical sample, entirely on GPU.
///
/// Eliminates the need to transfer the full logits array (up to 1MB for
/// vocab_size=262144) to the CPU for stochastic sampling.
///
/// Algorithm (single threadgroup, 4 passes):
///
///   Pass 1 — find max (subtract for numerical stability after temp scaling):
///     Each thread computes local max over its chunk, tree-reduces to global max.
///
///   Pass 2 — compute exp(logit/T - max) and accumulate partition sum:
///     Each thread computes exp values, writes to a temporary scratch buffer
///     (reusing the input layout), accumulates a local sum, tree-reduces.
///
///   Pass 3 — normalize to probabilities (divide by partition sum):
///     Each thread divides its scratch values by the global sum.
///
///   Pass 4 — compute CDF prefix sum + binary-search sample:
///     We use a simple sequential scan by thread 0 over the scratch buffer.
///     For large vocab sizes (262144) this is ~1ms extra vs a full parallel
///     prefix sum, which is acceptable.  A parallel version can be added later.
///
/// Output:
///   out_token  — u32 sampled token index
///   out_logprob — f32 log-probability of the sampled token (ln(prob))
///
/// Buffer layout:
///   buffer(0): logits      — float [n_elements]  input logits (read-only)
///   buffer(1): scratch     — float [n_elements]  temp buffer (probabilities)
///   buffer(2): out_token   — uint  [1]           sampled token
///   buffer(3): out_logprob — float [1]           log-prob of sampled token
///   buffer(4): params      — float [3]           [n_elements_f, temperature, random_val]
///
/// params encoding:
///   params[0] — n_elements cast to float (recast to uint inside kernel)
///   params[1] — temperature  (> 0.0)
///   params[2] — uniform random value in [0, 1) for categorical sampling
///
/// Threadgroup: (min(1024, next_power_of_two(n_elements)), 1, 1)
/// Grid:        (1, 1, 1)
/// Shared mem index 0: tg_size * sizeof(float) for float reduction
/// Shared mem index 1: (unused — reserved for future index tracking)
///
/// NOTE: tg_size must be a power of 2.

kernel void softmax_sample_f32(
    device const float* logits      [[buffer(0)]],
    device float*       scratch     [[buffer(1)]],
    device uint*        out_token   [[buffer(2)]],
    device float*       out_logprob [[buffer(3)]],
    device const float* params      [[buffer(4)]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup(0)]]
) {
    const uint  n_elements  = uint(params[0]);
    const float temperature = params[1];
    const float random_val  = params[2];

    const float inv_temp = 1.0f / temperature;

    // -----------------------------------------------------------------------
    // Pass 1: find max of (logit / temperature) for numerical stability.
    // -----------------------------------------------------------------------
    float local_max = -INFINITY;
    for (uint i = tid; i < n_elements; i += tg_size) {
        local_max = max(local_max, logits[i] * inv_temp);
    }

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float global_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Pass 2: exp(logit/T - max), write to scratch, accumulate local sum.
    // -----------------------------------------------------------------------
    float local_sum = 0.0f;
    for (uint i = tid; i < n_elements; i += tg_size) {
        float e = exp(logits[i] * inv_temp - global_max);
        scratch[i] = e;
        local_sum += e;
    }

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float partition = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Pass 3: normalize scratch to probabilities.
    // -----------------------------------------------------------------------
    const float inv_partition = 1.0f / partition;
    for (uint i = tid; i < n_elements; i += tg_size) {
        scratch[i] *= inv_partition;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Pass 4: thread 0 scans the CDF and finds the sampled token.
    //
    // This is O(n_elements) sequential work done by a single thread.  For
    // vocab_size=262144 this is roughly 262K multiply-adds at ~1 cycle each
    // (~0.08ms on M-series at 3GHz), which is much cheaper than a GPU->CPU
    // transfer of the full 1MB logits array.
    //
    // A parallel prefix-sum + binary-search can halve this if needed later.
    // -----------------------------------------------------------------------
    if (tid == 0) {
        float cdf     = 0.0f;
        uint  tok     = n_elements - 1;  // fallback: last token
        for (uint i = 0; i < n_elements; i++) {
            cdf += scratch[i];
            if (cdf >= random_val) {
                tok = i;
                break;
            }
        }
        out_token[0]   = tok;
        out_logprob[0] = log(scratch[tok]);  // ln(prob)
    }
}
