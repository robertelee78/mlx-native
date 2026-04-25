//! SDPA decode kernel — F32 Q/K/V, multi-simdgroup tiled, single query token.
//!
//! Architecture
//! ============
//!   Grid  : (n_heads, 1, 1)   — one threadgroup per query head
//!   TG    : (n_sg * 32, 1, 1) — n_sg simdgroups of 32 threads each
//!
//! Each simdgroup `sg` processes KV positions [sg*chunk, (sg+1)*chunk),
//! accumulating a local online-softmax triple (m, d, acc) where:
//!   m   = running max of QK scores
//!   d   = sum of exp(score - m)              (denominator, not yet normalized)
//!   acc[e] = sum_k exp(score_k - m) * V[k][e]  (unnormalized output)
//!
//! After the per-simdgroup pass, simdgroup 0 reads all N_SG partial results
//! from threadgroup memory and merges them with the log-sum-exp combination:
//!
//!   given (m0, d0, unorm0) and (m1, d1, unorm1):
//!     new_m    = max(m0, m1)
//!     c0       = exp(m0 - new_m)
//!     c1       = exp(m1 - new_m)
//!     new_d    = d0*c0 + d1*c1
//!     new_unorm = unorm0*c0 + unorm1*c1
//!   then normalize: O = new_unorm / new_d
//!
//! NOTE: local_acc stored in shmem is already normalized by local_d, i.e.
//!   shmem_acc[s][lane*EPL + e] = local_acc[e]  (normalized by local_d[s])
//! The merge undoes normalization: unorm_s = local_acc_s * d_s, then merges.
//!
//! Threadgroup memory (buffer index 0, shmem pointer):
//!   sg_max : [n_sg]               floats  at offset 0
//!   sg_sum : [n_sg]               floats  at offset n_sg
//!   sg_acc : [n_sg * head_dim]    floats  at offset 2*n_sg
//!
//! Total bytes = 4 * (2*n_sg + n_sg*head_dim) = 4*n_sg*(head_dim + 2)
//!   n_sg=4, head_dim=256: 4*4*258 = 4128 bytes
//!   n_sg=4, head_dim=512: 4*4*514 = 8224 bytes
//!   n_sg=2, head_dim=256: 4*2*258 = 2064 bytes
//!
//! Supports head_dim = 128, 256, 512.

#include <metal_stdlib>
using namespace metal;

struct SdpaDecodeParams {
    uint  n_heads;       // total Q heads
    uint  n_kv_heads;    // KV heads (GQA: n_kv_heads <= n_heads)
    uint  head_dim;      // must be 128, 256, or 512
    uint  kv_seq_len;    // valid KV positions in the cache
    uint  kv_capacity;   // stride between KV heads (>= kv_seq_len)
    float scale;         // attention scale (typically 1/sqrt(head_dim))
    uint  n_sg;          // simdgroups per threadgroup (2 or 4)
};

kernel void sdpa_decode(
    device const float          *Q      [[buffer(0)]],  // [n_heads, head_dim]
    device const float          *K      [[buffer(1)]],  // [n_kv_heads, kv_cap, head_dim]
    device const float          *V      [[buffer(2)]],  // [n_kv_heads, kv_cap, head_dim]
    device       float          *O      [[buffer(3)]],  // [n_heads, head_dim]
    constant SdpaDecodeParams   &p      [[buffer(4)]],
    threadgroup  float          *shmem  [[threadgroup(0)]],
    uint   head_idx  [[threadgroup_position_in_grid]],
    ushort tid       [[thread_index_in_threadgroup]]   // 0 .. n_sg*32 - 1
) {
    const uint  n_heads    = p.n_heads;
    const uint  n_kv_heads = p.n_kv_heads;
    const uint  head_dim   = p.head_dim;
    const uint  kv_seq_len = p.kv_seq_len;
    const uint  kv_cap     = p.kv_capacity;
    const float scale      = p.scale;
    const uint  n_sg       = p.n_sg;

    if (head_idx >= n_heads) return;

    // Identify this thread's simdgroup (sg) and lane within that simdgroup.
    const ushort sg   = tid / 32;
    const ushort lane = tid % 32;

    // GQA: map Q head → KV head.
    const uint kv_head = head_idx * n_kv_heads / n_heads;

    // Elements per lane: each lane owns EPL contiguous floats of Q/K/V per position.
    const uint EPL = head_dim / 32;   // 4 for hd=128, 8 for hd=256, 16 for hd=512

    const uint q_off  = head_idx * head_dim;
    const uint kv_off = kv_head  * kv_cap  * head_dim;

    // Load Q into registers.
    float q_reg[16];
    for (uint e = 0; e < EPL; e++) {
        q_reg[e] = Q[q_off + lane * EPL + e];
    }

    // ── Shared memory pointers ────────────────────────────────────────────
    threadgroup float *sg_max = shmem;               // [n_sg]
    threadgroup float *sg_sum = shmem + n_sg;         // [n_sg]
    threadgroup float *sg_acc = shmem + 2 * n_sg;     // [n_sg * head_dim]

    // ── Per-simdgroup KV scan ─────────────────────────────────────────────
    // chunk = ceil(kv_seq_len / n_sg); simdgroup sg handles [sg_start, sg_end).
    const uint chunk    = (kv_seq_len + n_sg - 1) / n_sg;
    const uint sg_start = sg * chunk;
    const uint sg_end   = min(sg_start + chunk, kv_seq_len);

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float local_acc[16];
    for (uint e = 0; e < EPL; e++) local_acc[e] = 0.0f;

    for (uint k_pos = sg_start; k_pos < sg_end; k_pos++) {
        const uint kb = kv_off + k_pos * head_dim + lane * EPL;

        // Partial QK dot for this lane's EPL elements.
        float partial = 0.0f;
        for (uint e = 0; e < EPL; e++) {
            partial += q_reg[e] * K[kb + e];
        }
        // Full dot product via warp reduction.
        float dot = simd_sum(partial) * scale;

        // Online softmax update.
        float old_max  = local_max;
        local_max      = max(local_max, dot);
        float corr     = exp(old_max - local_max);
        float w        = exp(dot - local_max);
        local_sum      = local_sum * corr + w;
        for (uint e = 0; e < EPL; e++) {
            local_acc[e] = local_acc[e] * corr
                         + w * V[kv_off + k_pos * head_dim + lane * EPL + e];
        }
    }

    // ── Store partial results to threadgroup memory ───────────────────────
    // local_acc is unnormalized (sum_k w_k * V[k], w_k = exp(score_k - local_max)).
    // Store as-is; the merge will handle re-scaling.
    if (lane == 0) {
        sg_max[sg] = local_max;
        sg_sum[sg] = local_sum;
    }
    const uint acc_base = sg * head_dim + lane * EPL;
    for (uint e = 0; e < EPL; e++) {
        sg_acc[acc_base + e] = local_acc[e];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Merge partial results — simdgroup 0 only ─────────────────────────
    // Combine all (m_s, d_s, unnorm_acc_s) into the final output.
    // The stored sg_acc[s] IS unnormalized (w = exp(score - m_s), no /d_s).
    if (sg == 0) {
        float m = sg_max[0];
        float d = sg_sum[0];
        float unorm[16];
        for (uint e = 0; e < EPL; e++) {
            unorm[e] = sg_acc[0 * head_dim + lane * EPL + e];
        }

        for (uint s = 1; s < n_sg; s++) {
            float m_s = sg_max[s];
            if (m_s == -INFINITY) continue;   // empty chunk
            float d_s = sg_sum[s];

            float new_m = max(m, m_s);
            float c0    = exp(m   - new_m);
            float c1    = exp(m_s - new_m);
            float new_d = d * c0 + d_s * c1;

            for (uint e = 0; e < EPL; e++) {
                unorm[e] = unorm[e] * c0
                         + sg_acc[s * head_dim + lane * EPL + e] * c1;
            }
            m = new_m;
            d = new_d;
        }

        // Normalize and write output.
        float inv_d = (d > 0.0f) ? (1.0f / d) : 0.0f;
        const uint o_off = head_idx * head_dim + lane * EPL;
        for (uint e = 0; e < EPL; e++) {
            O[o_off + e] = unorm[e] * inv_d;
        }
    }
}
