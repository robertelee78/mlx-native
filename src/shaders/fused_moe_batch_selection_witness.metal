// Optional diagnostics from actual batched producers; legacy shaders unchanged.
#include <metal_stdlib>
using namespace metal;
struct FusedMoeRoutingParams { uint num_experts; uint top_k; };

kernel void fused_moe_routing_batch_f32_selection_witness(
    device const float*               logits_all      [[buffer(0)]],
    device uint*                      expert_ids_all  [[buffer(1)]],
    device float*                     routing_weights_all [[buffer(2)]],
    device const float*               per_expert_scale [[buffer(3)]],
    constant FusedMoeRoutingParams&   params          [[buffer(4)]],
    device float* witness_scores [[buffer(5)]],
    device uint* witness_ids [[buffer(6)]],
    device uint* witness_status [[buffer(7)]],
    uint tok_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint num_experts = params.num_experts;
    const uint top_k       = params.top_k;

    device const float* logits          = logits_all       + tok_id * num_experts;
    device uint*        expert_ids      = expert_ids_all   + tok_id * top_k;
    device float*       routing_weights = routing_weights_all + tok_id * top_k;

    // Uniform per-TG validation before any collective; nonzero lanes/rows
    // are covered without a lane0-only SIMD predicate.
    const bool geometry_ok = num_experts >= 32 && num_experts <= 256
        && (num_experts & (num_experts-1u))==0u && top_k > 0 && top_k <= 64 && top_k <= num_experts;
    if (!geometry_ok) {
        if (tid==0) {
            witness_status[tok_id]=3u;
            witness_scores[2*tok_id]=NAN; witness_scores[2*tok_id+1]=NAN;
            witness_ids[2*tok_id]=0xffffffffu; witness_ids[2*tok_id+1]=0xffffffffu;
        }
        return;
    }
    bool witness_ok = true;
    for (uint e=0; e<num_experts; ++e)
        witness_ok &= isfinite(logits[e]) && isfinite(per_expert_scale[e]);
    if (!witness_ok) {
        if (tid==0) {
            witness_status[tok_id]=3u;
            witness_scores[2*tok_id]=NAN; witness_scores[2*tok_id+1]=NAN;
            witness_ids[2*tok_id]=0xffffffffu; witness_ids[2*tok_id+1]=0xffffffffu;
            for (uint k=0; k<top_k; ++k) { expert_ids[k]=0xffffffffu; routing_weights[k]=NAN; }
        }
        return;
    }

    // Step 1: find max for numerical stability (softmax)
    float local_max = -INFINITY;
    for (uint i = tid; i < num_experts; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_val = shared[0];

    // All threads must complete the broadcast-read of shared[0] for max_val
    // BEFORE any thread (notably tid==0) overwrites shared[0] with its
    // local_sum. Same race class as the fused_head_norm_rope Phase-1→Phase-2
    // boundary (see b31505d / hf2q docs/spike-batched-prefill-race-rootcause.md).
    // Without this barrier, simdgroups that race ahead of simdgroup 0 read a
    // clobbered shared[0] and compute a corrupt max_val — produces
    // nondeterministic routing decisions at scale.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < num_experts; i += tg_size) {
        const float e = exp(logits[i] - max_val);
        shared[num_experts + i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum_exp = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: compute softmax probabilities in shared[0..num_experts)
    for (uint i = tid; i < num_experts; i += tg_size) {
        shared[i] = shared[num_experts + i] / sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: find top-K (serial, only thread 0 — K is tiny, typically 2)
    if (tid == 0) {
        for (uint k = 0; k < top_k; k++) {
            float best_val = -1.0f;
            uint best_idx = 0;
            for (uint i = 0; i < num_experts; i++) {
                if (shared[i] > best_val) {
                    best_val = shared[i];
                    best_idx = i;
                }
            }
            expert_ids[k] = best_idx;
            routing_weights[k] = best_val;
            shared[best_idx] = -1.0f;
        }

        // Read actual producer scratch BEFORE top-K renorm/expert scale.
        float next_score=-INFINITY; uint next_id=0xffffffffu;
        for (uint e=0;e<num_experts;++e) {
            bool selected=false;
            for (uint k=0;k<top_k;++k) selected |= expert_ids[k]==e;
            const float value=shared[e];
            if (!selected && (value>next_score || (value==next_score && e<next_id))) {
                next_score=value; next_id=e;
            }
        }
        witness_scores[2*tok_id]=routing_weights[top_k-1];
        witness_scores[2*tok_id+1]=(top_k==num_experts)?NAN:next_score;
        witness_ids[2*tok_id]=expert_ids[top_k-1];
        witness_ids[2*tok_id+1]=(top_k==num_experts)?0xffffffffu:next_id;
        witness_status[tok_id]=(top_k==num_experts)?2u:1u;

        float topk_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            topk_sum += routing_weights[k];
        }
        if (topk_sum > 0.0f) {
            for (uint k = 0; k < top_k; k++) {
                const uint eid = expert_ids[k];
                routing_weights[k] = (routing_weights[k] / topk_sum) * per_expert_scale[eid];
            }
        } else {
            for (uint k = 0; k < top_k; k++) {
                routing_weights[k] = 0.0f;
            }
        }
    }
}

kernel void fused_moe_routing_batch_f32_v3_selection_witness(
    device const float*               logits_all      [[buffer(0)]],
    device uint*                      expert_ids_all  [[buffer(1)]],
    device float*                     routing_weights_all [[buffer(2)]],
    device const float*               per_expert_scale [[buffer(3)]],
    constant FusedMoeRoutingParams&   params          [[buffer(4)]],
    device float* witness_scores [[buffer(5)]],
    device uint* witness_ids [[buffer(6)]],
    device uint* witness_status [[buffer(7)]],
    uint tok_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint num_experts = params.num_experts;
    const uint top_k       = params.top_k;
    const uint n_sg        = tg_size / 32u;

    device const float* logits          = logits_all       + tok_id * num_experts;
    device uint*        expert_ids      = expert_ids_all   + tok_id * top_k;
    device float*       routing_weights = routing_weights_all + tok_id * top_k;

    // Uniform per-TG validation before any collective; nonzero lanes/rows
    // are covered without a lane0-only SIMD predicate.
    const bool geometry_ok = num_experts >= 32 && num_experts <= 256
        && (num_experts & (num_experts-1u))==0u && top_k > 0 && top_k <= 64 && top_k <= num_experts;
    if (!geometry_ok) {
        if (tid==0) {
            witness_status[tok_id]=3u;
            witness_scores[2*tok_id]=NAN; witness_scores[2*tok_id+1]=NAN;
            witness_ids[2*tok_id]=0xffffffffu; witness_ids[2*tok_id+1]=0xffffffffu;
        }
        return;
    }
    bool witness_ok = true;
    for (uint e=0; e<num_experts; ++e)
        witness_ok &= isfinite(logits[e]) && isfinite(per_expert_scale[e]);
    if (!witness_ok) {
        if (tid==0) {
            witness_status[tok_id]=3u;
            witness_scores[2*tok_id]=NAN; witness_scores[2*tok_id+1]=NAN;
            witness_ids[2*tok_id]=0xffffffffu; witness_ids[2*tok_id+1]=0xffffffffu;
            for (uint k=0; k<top_k; ++k) { expert_ids[k]=0xffffffffu; routing_weights[k]=NAN; }
        }
        return;
    }

    // ADR-029 iter-175 Step 1j.2: SOFTMAX REDUCTION ORDER FIX.
    //
    // Step 1j originally used simd_max/simd_sum (matching unbatched V3).
    // BUT the unbatched V3 falls back to V2 when V3 is off; V2 unbatched
    // ALSO uses simd-reduce.  So unbatched V3 ≡ V2 byte-identical.
    //
    // Batched is DIFFERENT: when V3 is off, the BATCHED dispatcher falls
    // back to V1 (`fused_moe_routing_batch_f32`) — there is no V2 batched.
    // V1 batched uses TREE-REDUCE (scalar shmem reductions over tg_size).
    //
    // Different reduction order → different f32 rounding → softmax probs
    // differ at ULP scale.  At top-K boundary, this swaps which experts
    // get picked → different routing → cascading divergent decode.
    //
    // Production diagnosis (this iteration): test rig with
    //   HF2Q_BATCHED_PREFILL=0 forces unbatched prefill (uses unbatched V3).
    //   Result: V3-default ≡ V2-default BYTE-IDENTICAL output.
    //   Confirms divergence is from batched kernel, NOT decode kernel.
    //
    // Fix: switch V3 batched softmax to tree-reduce, matching V1 batched.
    // V3 batched still wins via PARALLEL top-K (Step 4 unchanged).
    // V3 batched softmax+top-K ≡ V1 batched softmax+top-K at f32 level.

    // --- Step 1: find max via TREE-REDUCE (matches V1 batched) ---
    float local_max = -INFINITY;
    for (uint i = tid; i < num_experts; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_val = shared[0];
    // Additional barrier (matches V1 batched) — prevents subsequent
    // shared[tid] = local_sum from clobbering shared[0] before all
    // simdgroups read max_val.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 2: exp(x - max) and sum via TREE-REDUCE (matches V1 batched) ---
    float local_sum = 0.0f;
    for (uint i = tid; i < num_experts; i += tg_size) {
        const float e = exp(logits[i] - max_val);
        shared[num_experts + i] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum_exp = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 3: write softmax probabilities to shared[0..num_experts) ---
    for (uint i = tid; i < num_experts; i += tg_size) {
        shared[i] = shared[num_experts + i] / sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 4 V3: PARALLEL top-K selection via SG-tournament reduce ---
    // Lex-on-tie semantics matching unbatched V3 (Step 1i.1) — restores
    // V3 ≡ V2 byte-identical output for production decode.
    for (uint k = 0; k < top_k; k++) {
        float my_val = -1.0f;
        uint  my_idx = 0u;
        for (uint i = tid; i < num_experts; i += tg_size) {
            const float v = shared[i];
            if (v > my_val) {
                my_val = v;
                my_idx = i;
            }
        }
        // SG-level tournament: OOR-guarded lex-on-tie (Step 1i.2 fix).
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            const float other_v = simd_shuffle_down(my_val, offset);
            const uint  other_i = simd_shuffle_down(my_idx, offset);
            const bool valid = ((ushort)tiisg + offset) < 32u;
            if (valid && (other_v > my_val || (other_v == my_val && other_i < my_idx))) {
                my_val = other_v;
                my_idx = other_i;
            }
        }
        // Per-SG winners staged to shmem.
        if (tiisg == 0) {
            shared[2u * num_experts + sgitg] = my_val;
            shared[2u * num_experts + n_sg + sgitg] = as_type<float>(my_idx);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // SG0 cross-SG tournament with same OOR-guarded lex-on-tie semantics.
        if (sgitg == 0) {
            float v = (tiisg < n_sg) ? shared[2u * num_experts + tiisg] : -1.0f;
            uint  i = (tiisg < n_sg)
                ? as_type<uint>(shared[2u * num_experts + n_sg + tiisg])
                : 0xFFFFFFFFu;
            for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_v = simd_shuffle_down(v, offset);
                const uint  other_i = simd_shuffle_down(i, offset);
                const bool valid = ((ushort)tiisg + offset) < 32u;
                if (valid && (other_v > v || (other_v == v && other_i < i))) {
                    v = other_v;
                    i = other_i;
                }
            }
            if (tiisg == 0) {
                expert_ids[k] = i;
                routing_weights[k] = v;
                shared[i] = -1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Step 5: renorm + per_expert_scale (single-thread, cheap) ---
    if (tid == 0) {
        // Read actual producer scratch BEFORE top-K renorm/expert scale.
        float next_score=-INFINITY; uint next_id=0xffffffffu;
        for (uint e=0;e<num_experts;++e) {
            bool selected=false;
            for (uint k=0;k<top_k;++k) selected |= expert_ids[k]==e;
            const float value=shared[e];
            if (!selected && (value>next_score || (value==next_score && e<next_id))) {
                next_score=value; next_id=e;
            }
        }
        witness_scores[2*tok_id]=routing_weights[top_k-1];
        witness_scores[2*tok_id+1]=(top_k==num_experts)?NAN:next_score;
        witness_ids[2*tok_id]=expert_ids[top_k-1];
        witness_ids[2*tok_id+1]=(top_k==num_experts)?0xffffffffu:next_id;
        witness_status[tok_id]=(top_k==num_experts)?2u:1u;

        float topk_sum = 0.0f;
        for (uint k = 0; k < top_k; k++) {
            topk_sum += routing_weights[k];
        }
        if (topk_sum > 0.0f) {
            for (uint k = 0; k < top_k; k++) {
                const uint eid = expert_ids[k];
                routing_weights[k] = (routing_weights[k] / topk_sum) * per_expert_scale[eid];
            }
        } else {
            for (uint k = 0; k < top_k; k++) {
                routing_weights[k] = 0.0f;
            }
        }
    }
}
