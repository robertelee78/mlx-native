// Stateful DeepSeek-V4 0731 learned KV compressor, ending before RoPE.

#include <metal_stdlib>
using namespace metal;

struct DeepSeekCompressorParams {
    uint batch;
    uint seq_len;
    uint start_pos;
    uint ratio;
    uint head_dim;
    uint cache_len;
    float epsilon;
    uint write_cache;
};

constant uint COMP_THREADS = 256;

inline float safe_state_value(float value) {
    return isfinite(value) ? value : 0.0f;
}

kernel void deepseek_compressor_bf16(
        constant DeepSeekCompressorParams &p [[buffer(0)]],
        device const float *kv                [[buffer(1)]],
        device const float *score             [[buffer(2)]],
        device const float *ape               [[buffer(3)]],
        device const float *norm              [[buffer(4)]],
        device float *kv_state                [[buffer(5)]],
        device float *score_state             [[buffer(6)]],
        device bfloat *output                 [[buffer(7)]],
        device bfloat *cache                  [[buffer(8)]],
        uint3 group                           [[threadgroup_position_in_grid]],
        uint tid                              [[thread_index_in_threadgroup]]) {
    const bool overlap = p.ratio == 4;
    const uint coff = overlap ? 2 : 1;
    const uint projected = coff * p.head_dim;
    const uint append_output_count =
        (p.start_pos + p.seq_len) / p.ratio - p.start_pos / p.ratio;
    const uint output_slots = p.start_pos == 0
        ? max(1u, p.seq_len / p.ratio)
        : max(1u, append_output_count);
    const uint batch = p.start_pos == 0 ? group.x / output_slots : group.x;
    const uint block = p.start_pos == 0 ? group.x % output_slots : 0;
    const ulong state_batch = ulong(batch) * coff * p.ratio * projected;
    const ulong input_batch = ulong(batch) * p.seq_len * projected;
    const ulong output_base = (ulong(batch) * output_slots + block) * p.head_dim;

    threadgroup float sums[COMP_THREADS];
    threadgroup uint bad[COMP_THREADS];
    threadgroup float rms_scale;
    threadgroup uint row_invalid;

    // A nonzero append owns one threadgroup per batch. Advance recurrent
    // compressor state in token order inside that group and emit every block
    // boundary into a contiguous output slot. This is byte-equivalent to a
    // sequence of one-token dispatches without paying one Metal dispatch and
    // memory barrier per token.
    if (p.start_pos != 0) {
        for (uint slot = 0; slot < output_slots; ++slot) {
            const ulong base = (ulong(batch) * output_slots + slot) * p.head_dim;
            for (uint feature = tid; feature < p.head_dim; feature += COMP_THREADS) {
                output[base + feature] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint emitted = 0;
        for (uint row = 0; row < p.seq_len; ++row) {
            const uint absolute = p.start_pos + row;
            const uint token = absolute % p.ratio;
            const uint slot = (overlap ? p.ratio : 0) + token;
            for (uint feature = tid; feature < projected; feature += COMP_THREADS) {
                const ulong src = input_batch + ulong(row) * projected + feature;
                const float v = kv[src];
                const float s = score[src] + ape[token * projected + feature];
                const ulong dst = state_batch + ulong(slot) * projected + feature;
                kv_state[dst] = safe_state_value(v);
                score_state[dst] = isfinite(s) && isfinite(v) ? s : NAN;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if ((absolute + 1) % p.ratio == 0) {
                float compressed[2] = { 0.0f, 0.0f };
                uint local_bad = 0;
                for (uint part = 0; part < 2; ++part) {
                    const uint feature = tid + part * COMP_THREADS;
                    if (feature >= p.head_dim) continue;
                    float maximum = -INFINITY;
                    const uint window = overlap ? 2 * p.ratio : p.ratio;
                    for (uint item = 0; item < window; ++item) {
                        const uint source_feature =
                            overlap && item >= p.ratio ? p.head_dim + feature : feature;
                        const ulong src = state_batch + ulong(item) * projected + source_feature;
                        const float s = score_state[src];
                        local_bad |=
                            (isnan(s) || s == INFINITY || !isfinite(kv_state[src])) ? 1u : 0u;
                        maximum = max(maximum, s);
                    }
                    float denominator = 0.0f;
                    float numerator = 0.0f;
                    for (uint item = 0; item < window; ++item) {
                        const uint source_feature =
                            overlap && item >= p.ratio ? p.head_dim + feature : feature;
                        const ulong src = state_batch + ulong(item) * projected + source_feature;
                        const float s = score_state[src];
                        const float v = kv_state[src];
                        const float weight = exp(s - maximum);
                        denominator += weight;
                        numerator = fma(weight, v, numerator);
                    }
                    const float pooled = numerator / denominator;
                    compressed[part] = float(bfloat(isfinite(pooled) ? pooled : 0.0f));
                    local_bad |= (!isfinite(pooled) || !isfinite(norm[feature])) ? 1u : 0u;
                }

                bad[tid] = local_bad;
                sums[tid] = compressed[0] * compressed[0] + compressed[1] * compressed[1];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = COMP_THREADS / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        bad[tid] += bad[tid + stride];
                        sums[tid] += sums[tid + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0) {
                    row_invalid = bad[0] != 0 || !isfinite(sums[0]);
                    rms_scale = row_invalid == 0
                        ? rsqrt(sums[0] / float(p.head_dim) + p.epsilon)
                        : 0.0f;
                    if (!isfinite(rms_scale)) row_invalid = 1;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float normalized[2] = { 0.0f, 0.0f };
                for (uint part = 0; part < 2; ++part) {
                    const uint feature = tid + part * COMP_THREADS;
                    if (feature < p.head_dim) {
                        normalized[part] = compressed[part] * rms_scale * norm[feature];
                    }
                }
                bad[tid] = (!isfinite(normalized[0]) || !isfinite(normalized[1])) ? 1u : 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = COMP_THREADS / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) bad[tid] += bad[tid + stride];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (tid == 0 && bad[0] != 0) row_invalid = 1;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const ulong output_base =
                    (ulong(batch) * output_slots + emitted) * p.head_dim;
                const uint cache_slot = absolute / p.ratio;
                const ulong cache_base =
                    (ulong(batch) * p.cache_len + cache_slot) * p.head_dim;
                for (uint part = 0; part < 2; ++part) {
                    const uint feature = tid + part * COMP_THREADS;
                    if (feature < p.head_dim) {
                        const bfloat result =
                            bfloat(row_invalid == 0 ? normalized[part] : 0.0f);
                        output[output_base + feature] = result;
                        if (p.write_cache != 0) cache[cache_base + feature] = result;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (overlap) {
                    const uint copy_count = p.ratio * projected;
                    for (uint i = tid; i < copy_count; i += COMP_THREADS) {
                        kv_state[state_batch + i] =
                            kv_state[state_batch + ulong(p.ratio) * projected + i];
                        score_state[state_batch + i] =
                            score_state[state_batch + ulong(p.ratio) * projected + i];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                emitted += 1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    // Prefill owns state initialization/update in block zero. Other blocks
    // consume input directly, so no cross-threadgroup state dependency exists.
    if (p.start_pos == 0 && block == 0) {
        const uint state_count = coff * p.ratio * projected;
        for (uint i = tid; i < state_count; i += COMP_THREADS) {
            kv_state[state_batch + i] = 0.0f;
            score_state[state_batch + i] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint cutoff = p.seq_len - p.seq_len % p.ratio;
        const uint remainder = p.seq_len - cutoff;
        if (overlap && cutoff >= p.ratio) {
            const uint copy_count = p.ratio * projected;
            for (uint i = tid; i < copy_count; i += COMP_THREADS) {
                const uint token_in_block = i / projected;
                const uint feature = i % projected;
                const ulong src = input_batch + ulong(cutoff - p.ratio + token_in_block) * projected + feature;
                const float v = kv[src];
                const float s = score[src] + ape[token_in_block * projected + feature];
                kv_state[state_batch + i] = safe_state_value(v);
                score_state[state_batch + i] = isfinite(s) && isfinite(v) ? s : NAN;
            }
        }
        const uint offset = overlap ? p.ratio : 0;
        const uint copy_count = remainder * projected;
        for (uint i = tid; i < copy_count; i += COMP_THREADS) {
            const uint token = i / projected;
            const uint feature = i % projected;
            const ulong src = input_batch + ulong(cutoff + token) * projected + feature;
            const ulong dst = state_batch + ulong(offset + token) * projected + feature;
            const float v = kv[src];
            const float s = score[src] + ape[token * projected + feature];
            kv_state[dst] = safe_state_value(v);
            score_state[dst] = isfinite(s) && isfinite(v) ? s : NAN;
        }
    }

    // Incremental calls update exactly one current-window slot.
    if (p.start_pos != 0) {
        const uint token = p.start_pos % p.ratio;
        const uint slot = (overlap ? p.ratio : 0) + token;
        for (uint feature = tid; feature < projected; feature += COMP_THREADS) {
            const float v = kv[input_batch + feature];
            const float s = score[input_batch + feature] + ape[token * projected + feature];
            const ulong dst = state_batch + ulong(slot) * projected + feature;
            kv_state[dst] = safe_state_value(v);
            score_state[dst] = isfinite(s) && isfinite(v) ? s : NAN;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint output_count = p.start_pos == 0 ? p.seq_len / p.ratio
                                               : uint((p.start_pos + 1) % p.ratio == 0);
    if (block >= output_count) {
        for (uint feature = tid; feature < p.head_dim; feature += COMP_THREADS) {
            output[output_base + feature] = bfloat(0.0f);
        }
        return;
    }

    float compressed[2] = { 0.0f, 0.0f };
    uint local_bad = 0;
    for (uint part = 0; part < 2; ++part) {
        const uint feature = tid + part * COMP_THREADS;
        if (feature >= p.head_dim) continue;
        float maximum = -INFINITY;
        const uint window = overlap ? 2 * p.ratio : p.ratio;
        for (uint item = 0; item < window; ++item) {
            float s = -INFINITY;
            if (p.start_pos == 0) {
                if (!overlap || item >= p.ratio || block > 0) {
                    const uint source_block = overlap && item < p.ratio ? block - 1 : block;
                    const uint token = overlap ? item % p.ratio : item;
                    const uint source_feature = overlap && item >= p.ratio ? p.head_dim + feature : feature;
                    const ulong src = input_batch + ulong(source_block * p.ratio + token) * projected + source_feature;
                    s = score[src] + ape[token * projected + source_feature];
                    local_bad |= (!isfinite(s) || !isfinite(kv[src])) ? 1u : 0u;
                }
            } else {
                const uint slot = item;
                const uint source_feature = overlap && item >= p.ratio ? p.head_dim + feature : feature;
                const ulong src = state_batch + ulong(slot) * projected + source_feature;
                s = score_state[src];
                local_bad |= (isnan(s) || s == INFINITY || !isfinite(kv_state[src])) ? 1u : 0u;
            }
            maximum = max(maximum, s);
        }
        float denominator = 0.0f;
        float numerator = 0.0f;
        for (uint item = 0; item < window; ++item) {
            float s = -INFINITY;
            float v = 0.0f;
            if (p.start_pos == 0) {
                if (!overlap || item >= p.ratio || block > 0) {
                    const uint source_block = overlap && item < p.ratio ? block - 1 : block;
                    const uint token = overlap ? item % p.ratio : item;
                    const uint source_feature = overlap && item >= p.ratio ? p.head_dim + feature : feature;
                    const ulong src = input_batch + ulong(source_block * p.ratio + token) * projected + source_feature;
                    s = score[src] + ape[token * projected + source_feature];
                    v = kv[src];
                }
            } else {
                const uint source_feature = overlap && item >= p.ratio ? p.head_dim + feature : feature;
                const ulong src = state_batch + ulong(item) * projected + source_feature;
                s = score_state[src];
                v = kv_state[src];
            }
            const float weight = exp(s - maximum);
            denominator += weight;
            numerator = fma(weight, v, numerator);
        }
        const float pooled = numerator / denominator;
        compressed[part] = float(bfloat(isfinite(pooled) ? pooled : 0.0f));
        local_bad |= (!isfinite(pooled) || !isfinite(norm[feature])) ? 1u : 0u;
    }

    bad[tid] = local_bad;
    sums[tid] = compressed[0] * compressed[0] + compressed[1] * compressed[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = COMP_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            bad[tid] += bad[tid + stride];
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        row_invalid = bad[0] != 0 || !isfinite(sums[0]);
        rms_scale = row_invalid == 0 ? rsqrt(sums[0] / float(p.head_dim) + p.epsilon) : 0.0f;
        if (!isfinite(rms_scale)) row_invalid = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float normalized[2] = { 0.0f, 0.0f };
    for (uint part = 0; part < 2; ++part) {
        const uint feature = tid + part * COMP_THREADS;
        if (feature < p.head_dim) {
            normalized[part] = compressed[part] * rms_scale * norm[feature];
        }
    }
    bad[tid] = (!isfinite(normalized[0]) || !isfinite(normalized[1])) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = COMP_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) bad[tid] += bad[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && bad[0] != 0) row_invalid = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint cache_slot = p.start_pos == 0 ? block : p.start_pos / p.ratio;
    const ulong cache_base = (ulong(batch) * p.cache_len + cache_slot) * p.head_dim;
    for (uint part = 0; part < 2; ++part) {
        const uint feature = tid + part * COMP_THREADS;
        if (feature < p.head_dim) {
            const bfloat result = bfloat(row_invalid == 0 ? normalized[part] : 0.0f);
            output[output_base + feature] = result;
            if (p.write_cache != 0) cache[cache_base + feature] = result;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (p.start_pos != 0 && overlap) {
        const uint copy_count = p.ratio * projected;
        for (uint i = tid; i < copy_count; i += COMP_THREADS) {
            kv_state[state_batch + i] = kv_state[state_batch + ulong(p.ratio) * projected + i];
            score_state[state_batch + i] = score_state[state_batch + ulong(p.ratio) * projected + i];
        }
    }
}
