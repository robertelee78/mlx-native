// Diagnostic copies preserve each legacy cached variant and arithmetic.
#include <metal_stdlib>
using namespace metal;
struct FusedMoeRoutingParams {
    uint num_experts;
    uint top_k;
};

kernel void fused_moe_routing_f32_selection_witness(
    device const float*               logits          [[buffer(0)]],
    device uint*                      expert_ids      [[buffer(1)]],
    device float*                     routing_weights [[buffer(2)]],
    device const float*               per_expert_scale [[buffer(3)]],
    constant FusedMoeRoutingParams&   params          [[buffer(4)]],
    device float* witness_scores [[buffer(5)]],
    device uint* witness_ids [[buffer(6)]],
    device uint* witness_status [[buffer(7)]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint num_experts = params.num_experts;
    const uint top_k       = params.top_k;

    bool witness_input_ok = num_experts >= 32 && num_experts <= 256 && top_k > 0 && top_k <= 64 && top_k <= num_experts;
    if (!witness_input_ok) {
        if (tid==0) { witness_status[0]=3u;
            witness_scores[2*0]=NAN; witness_scores[2*0+1]=NAN;
            witness_ids[2*0]=0xffffffffu; witness_ids[2*0+1]=0xffffffffu; }
        return;
    }
    for (uint e=0;e<num_experts;++e) witness_input_ok &= isfinite(logits[e]) && isfinite(per_expert_scale[e]);
    if (!witness_input_ok) {
        if (tid==0) {
            witness_status[0]=3u; witness_scores[0]=NAN; witness_scores[1]=NAN;
            witness_ids[0]=0xffffffffu; witness_ids[1]=0xffffffffu;
            for(uint k=0;k<top_k;++k) { expert_ids[k]=0xffffffffu; routing_weights[k]=NAN; }
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

    // Step 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < num_experts; i += tg_size) {
        const float e = exp(logits[i] - max_val);
        shared[num_experts + i] = e;  // store exp values after first num_experts slots
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
        // Simple selection sort for K elements
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
            // Match the legacy passing path:
            //   1. softmax over all experts
            //   2. renormalize the selected top-K slice
            //   3. apply per_expert_scale
            // Critically, do NOT renormalize after applying per_expert_scale.
            routing_weights[k] = best_val;
            shared[best_idx] = -1.0f;  // mark as used
        }

        // Renormalize the top-K weights before applying per_expert_scale.
        float next_score=-INFINITY; uint next_id=0xffffffffu;
        for(uint e=0;e<num_experts;++e) {
            bool selected=false;
            for(uint k=0;k<top_k;++k) selected |= expert_ids[k]==e;
            const float candidate=shared[e];
            if(!selected && (candidate>next_score || (candidate==next_score && e<next_id))) {
                next_score=candidate; next_id=e;
            }
        }
        witness_scores[0]=routing_weights[top_k-1];
        witness_scores[1]=(top_k==num_experts)?NAN:next_score;
        witness_ids[0]=expert_ids[top_k-1]; witness_ids[1]=(top_k==num_experts)?0xffffffffu:next_id;
        witness_status[0]=(top_k==num_experts)?2u:1u;

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

kernel void fused_moe_routing_f32_v2_selection_witness(
    device const float*               logits          [[buffer(0)]],
    device uint*                      expert_ids      [[buffer(1)]],
    device float*                     routing_weights [[buffer(2)]],
    device const float*               per_expert_scale [[buffer(3)]],
    constant FusedMoeRoutingParams&   params          [[buffer(4)]],
    device float* witness_scores [[buffer(5)]],
    device uint* witness_ids [[buffer(6)]],
    device uint* witness_status [[buffer(7)]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint num_experts = params.num_experts;
    const uint top_k       = params.top_k;
    const uint n_sg        = tg_size / 32u;

    bool witness_input_ok = num_experts >= 32 && num_experts <= 256 && top_k > 0 && top_k <= 64 && top_k <= num_experts;
    if (!witness_input_ok) {
        if (tid==0) { witness_status[0]=3u;
            witness_scores[2*0]=NAN; witness_scores[2*0+1]=NAN;
            witness_ids[2*0]=0xffffffffu; witness_ids[2*0+1]=0xffffffffu; }
        return;
    }
    for (uint e=0;e<num_experts;++e) witness_input_ok &= isfinite(logits[e]) && isfinite(per_expert_scale[e]);
    if (!witness_input_ok) {
        if (tid==0) {
            witness_status[0]=3u; witness_scores[0]=NAN; witness_scores[1]=NAN;
            witness_ids[0]=0xffffffffu; witness_ids[1]=0xffffffffu;
            for(uint k=0;k<top_k;++k) { expert_ids[k]=0xffffffffu; routing_weights[k]=NAN; }
        }
        return;
    }

    // --- Step 1: find max for numerical stability (softmax) ---
    float local_max = -INFINITY;
    for (uint i = tid; i < num_experts; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    // Cross-SG max via simd_max + shared scratch.
    local_max = simd_max(local_max);
    if (tiisg == 0) {
        shared[num_experts + sgitg] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // SG0 finishes the cross-SG reduction; broadcast result back via shared[num_experts + 0].
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[num_experts + tiisg] : -INFINITY;
        const float total = simd_max(v);
        if (tiisg == 0) {
            shared[num_experts + 0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max_val = shared[num_experts + 0];

    // --- Step 2: compute exp(x - max) and sum ---
    float local_sum = 0.0f;
    for (uint i = tid; i < num_experts; i += tg_size) {
        const float e = exp(logits[i] - max_val);
        shared[num_experts + i] = e;  // store exp values for step 3
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (tiisg == 0) {
        // Store per-SG partial sums in shmem[2*num_experts + sgitg] (after the exp buffer).
        shared[2u * num_experts + sgitg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[2u * num_experts + tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[2u * num_experts + 0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum_exp = shared[2u * num_experts + 0];

    // --- Step 3: write softmax probabilities to shared[0..num_experts) ---
    // (overwrites the original logits slot, leaving exp values in shared[num_experts..])
    for (uint i = tid; i < num_experts; i += tg_size) {
        shared[i] = shared[num_experts + i] / sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 4: top-K selection (single-thread serial, same as V1) ---
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

        float next_score=-INFINITY; uint next_id=0xffffffffu;
        for(uint e=0;e<num_experts;++e) {
            bool selected=false;
            for(uint k=0;k<top_k;++k) selected |= expert_ids[k]==e;
            const float candidate=shared[e];
            if(!selected && (candidate>next_score || (candidate==next_score && e<next_id))) {
                next_score=candidate; next_id=e;
            }
        }
        witness_scores[0]=routing_weights[top_k-1];
        witness_scores[1]=(top_k==num_experts)?NAN:next_score;
        witness_ids[0]=expert_ids[top_k-1]; witness_ids[1]=(top_k==num_experts)?0xffffffffu:next_id;
        witness_status[0]=(top_k==num_experts)?2u:1u;

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

kernel void fused_moe_routing_f32_v3_selection_witness(
    device const float*               logits          [[buffer(0)]],
    device uint*                      expert_ids      [[buffer(1)]],
    device float*                     routing_weights [[buffer(2)]],
    device const float*               per_expert_scale [[buffer(3)]],
    constant FusedMoeRoutingParams&   params          [[buffer(4)]],
    device float* witness_scores [[buffer(5)]],
    device uint* witness_ids [[buffer(6)]],
    device uint* witness_status [[buffer(7)]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]],
    ushort sgitg  [[simdgroup_index_in_threadgroup]],
    ushort tiisg  [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    const uint num_experts = params.num_experts;
    const uint top_k       = params.top_k;
    const uint n_sg        = tg_size / 32u;

    bool witness_input_ok = num_experts >= 32 && num_experts <= 256 && top_k > 0 && top_k <= 64 && top_k <= num_experts;
    if (!witness_input_ok) {
        if (tid==0) { witness_status[0]=3u;
            witness_scores[2*0]=NAN; witness_scores[2*0+1]=NAN;
            witness_ids[2*0]=0xffffffffu; witness_ids[2*0+1]=0xffffffffu; }
        return;
    }
    for (uint e=0;e<num_experts;++e) witness_input_ok &= isfinite(logits[e]) && isfinite(per_expert_scale[e]);
    if (!witness_input_ok) {
        if (tid==0) {
            witness_status[0]=3u; witness_scores[0]=NAN; witness_scores[1]=NAN;
            witness_ids[0]=0xffffffffu; witness_ids[1]=0xffffffffu;
            for(uint k=0;k<top_k;++k) { expert_ids[k]=0xffffffffu; routing_weights[k]=NAN; }
        }
        return;
    }

    // --- Step 1: find max for numerical stability (softmax) — same as V2 ---
    float local_max = -INFINITY;
    for (uint i = tid; i < num_experts; i += tg_size) {
        local_max = max(local_max, logits[i]);
    }
    local_max = simd_max(local_max);
    if (tiisg == 0) {
        shared[num_experts + sgitg] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[num_experts + tiisg] : -INFINITY;
        const float total = simd_max(v);
        if (tiisg == 0) {
            shared[num_experts + 0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max_val = shared[num_experts + 0];

    // --- Step 2: compute exp(x - max) and sum — same as V2 ---
    float local_sum = 0.0f;
    for (uint i = tid; i < num_experts; i += tg_size) {
        const float e = exp(logits[i] - max_val);
        shared[num_experts + i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (tiisg == 0) {
        shared[2u * num_experts + sgitg] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const float v = (tiisg < n_sg) ? shared[2u * num_experts + tiisg] : 0.0f;
        const float total = simd_sum(v);
        if (tiisg == 0) {
            shared[2u * num_experts + 0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float sum_exp = shared[2u * num_experts + 0];

    // --- Step 3: write softmax probabilities — same as V2 ---
    for (uint i = tid; i < num_experts; i += tg_size) {
        shared[i] = shared[num_experts + i] / sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 4 V3: PARALLEL top-K selection via SG-tournament reduce ---
    // For each k in [0, top_k):
    //   1. Each thread scans its strided slice of shared[0..num_experts) and
    //      tracks (best_val, best_idx).
    //   2. SG-level tournament via simd_shuffle_down (5 rounds for SG=32).
    //   3. Stage per-SG winners (val + idx) into shmem.
    //   4. SG0 does cross-SG tournament to find global winner.
    //   5. Lane 0 writes expert_ids[k] / routing_weights[k] and sets
    //      shared[winner] = -1.0 to exclude from next round.
    //   6. Threadgroup barrier before next k iteration.
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
        // SG-level tournament max-with-LOWEST-index-on-tie.
        //
        // Step 1i.2 (this commit): two fixes vs original V3
        //   (a) lex-on-tie: `if (other_v > my_val || (other_v == my_val &&
        //       other_i < my_idx))` — matches V2's "lowest-idx-on-tie"
        //       from serial scan from i=0.
        //   (b) OOR shuffle guard: `simd_shuffle_down(x, offset)` from lane
        //       L reads lane L+offset.  When L+offset >= 32 the result is
        //       UNDEFINED per Apple Metal spec.  Without guarding, garbage
        //       propagates through subsequent reduction steps (lane 16's
        //       OOR-garbage at offset=16 reaches lane 0 at offset=4 etc.).
        //       Explicit `valid = (tiisg + offset) < 32u` filter protects
        //       against this.
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            const float other_v = simd_shuffle_down(my_val, offset);
            const uint  other_i = simd_shuffle_down(my_idx, offset);
            const bool valid = ((ushort)tiisg + offset) < 32u;
            if (valid && (other_v > my_val || (other_v == my_val && other_i < my_idx))) {
                my_val = other_v;
                my_idx = other_i;
            }
        }
        // Lane 0 of each SG holds that SG's best. Stage to shmem.
        if (tiisg == 0) {
            shared[2u * num_experts + sgitg] = my_val;
            // Pack idx as bits-of-float in adjacent slot.
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
                shared[i] = -1.0f;  // mark used
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float next_score=-INFINITY; uint next_id=0xffffffffu;
        for(uint e=0;e<num_experts;++e) {
            bool selected=false;
            for(uint k=0;k<top_k;++k) selected |= expert_ids[k]==e;
            const float candidate=shared[e];
            if(!selected && (candidate>next_score || (candidate==next_score && e<next_id))) {
                next_score=candidate; next_id=e;
            }
        }
        witness_scores[0]=routing_weights[top_k-1];
        witness_scores[1]=(top_k==num_experts)?NAN:next_score;
        witness_ids[0]=expert_ids[top_k-1]; witness_ids[1]=(top_k==num_experts)?0xffffffffu:next_id;
        witness_status[0]=(top_k==num_experts)?2u:1u;
    }

    // --- Step 5: renorm + per_expert_scale — single-thread (cheap, k iters) ---
    if (tid == 0) {
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
