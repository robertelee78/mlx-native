#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------------
// moe_mm_id_map0 — MoE token-to-expert sort pre-pass for two-pass mm_id
//
// Adapted from llama.cpp's `kernel_mul_mm_id_map0` at
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9724-9788. This is
// iter A of the ADR-033 §Pi multi-iter arc to port llama.cpp's two-pass
// MoE Q4_0 mm_id kernel. See
// project_adr033_pi_next_iter_arc_moe_q4_0_kernel_port_2026_05_23.md.
//
// Purpose: given top-k routing decisions for N tokens (`src2` of shape
// [N, ne20] where ne20 = n_expert_used, e.g. 8 for Qwen3.6), produce
// per-expert token-slot lists that the iter B main mm_id kernel will
// consume. The main kernel then processes each expert's tokens as a
// coalesced batch (NR1=32 tile per expert) instead of scattering through
// the activation mask, which is the structural advantage over hf2q's
// current dispatch_mm_id_q4_0 path.
//
// Outputs:
//   htpe[ide]              — count of (token, slot) pairs assigned to
//                            expert `ide`. Used by iter B kernel to
//                            early-exit when r1 ≥ neh1.
//   hids[ide][0..htpe[ide]] — packed list of `(i21 * ne20 + slot)` for
//                            every (token, slot) where the top-k decision
//                            named expert `ide`. The slot encodes which
//                            of the top-k positions this expert occupied
//                            for token i21, so the iter B kernel can
//                            recover the per-token routing weight.
//
// Threadgroup contract:
//   - One thread per expert (tpitg = ide). ntg ≥ n_experts.
//   - Each thread iterates over all tokens in chunks of ntg, cooperatively
//     loading top-k arrays into shmem for parallel scan.
//   - Templated on ne20 (n_expert_used) so the inner unroll loops are
//     constant-bounded — specialized variants registered for ne20 ∈
//     {1, 2, 4, 5, 6, 8, 10, 16, 22} matching llama.cpp's set.
// --------------------------------------------------------------------------

struct MoeMmIdMap0Params {
    // Layout mirrors ggml_metal_kargs_mul_mm_id_map0 at
    // ggml-metal-impl.h:496-505 — keep field order identical so the Rust
    // dispatch wrapper can transcribe arg values directly from a
    // llama.cpp-style call site if needed for debugging parity.
    int32_t  ne02;    // n_experts_total
    int32_t  ne10;    // unused here; carried for layout parity
    int32_t  ne11;    // n_expert_used (broadcast — equals ne20 for top-k)
    uint64_t nb11;    // unused here; carried for layout parity
    uint64_t nb12;    // unused here; carried for layout parity
    int32_t  ne21;    // n_tokens
    int32_t  ne20;    // n_expert_used (runtime echo of template param)
    uint64_t nb21;    // byte stride between successive top-k rows in src2
};

template<short NE20>
static void moe_mm_id_map0_impl(
    constant MoeMmIdMap0Params & args,
    device  const char         * src2,
    device        char         * htpe,
    device        char         * hids,
    threadgroup   char         * shmem,
    ushort tpitg,
    ushort ntg)
{
    // One thread = one expert.
    const short ide = (short) tpitg;

    uint32_t n_all = 0;
    device int32_t * ids_i32 = (device int32_t *) hids + ide * args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        // Cooperative load: each thread loads ONE token's top-k expert
        // list into its private shmem slot. Slots are striped by tpitg so
        // every thread reads contiguous src2 bytes.
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 =
                (device const int32_t *) (src2 + (i21 + tpitg) * args.nb21);

            threadgroup uint16_t * sids =
                (threadgroup uint16_t *) shmem + tpitg * NE20;

            #pragma unroll(NE20)
            for (short i20 = 0; i20 < NE20; i20++) {
                sids[i20] = (uint16_t) src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Sequential scan: each expert-thread walks the ntg tokens just
        // loaded and accumulates matches into its own hids slice. Uses
        // arithmetic instead of branches (sel += equality * (i20+1)) so
        // the inner loop is fully unrolled with no warp divergence.
        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) break;

            threadgroup const uint16_t * sids =
                (threadgroup const uint16_t *) shmem + t * NE20;

            short sel = 0;
            #pragma unroll(NE20)
            for (short i20 = 0; i20 < NE20; i20++) {
                sel += (sids[i20] == ide) * (i20 + 1);
            }

            // sel encodes: 0 = expert not chosen by token (i21+t), or
            // (chosen_slot_index + 1). The packed index encodes both the
            // token offset (i21+t) and the slot, so iter B can recover
            // the per-token, per-slot routing weight without a second
            // lookup.
            ids_i32[n_all] = (i21 + t) * NE20 + sel - 1;
            n_all += (sel > 0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) htpe;
    tpe_u32[ide] = n_all;
}

// Specializations. ne20 = n_expert_used for the routing decision.
// Qwen3.6 35B-A3B uses ne20=8. Set matches llama.cpp's
// kernel_mul_mm_id_map0_ne20_{1,2,4,5,6,8,10,16,22} family at
// ggml-metal.metal:9780-9788.

#define MOE_MM_ID_MAP0_VARIANT(NE) \
    [[host_name("moe_mm_id_map0_ne20_" #NE)]] \
    kernel void moe_mm_id_map0_ne20_##NE( \
        constant MoeMmIdMap0Params & args   [[buffer(0)]], \
        device  const char         * src2   [[buffer(1)]], \
        device        char         * htpe   [[buffer(2)]], \
        device        char         * hids   [[buffer(3)]], \
        threadgroup   char         * shmem  [[threadgroup(0)]], \
        ushort tpitg                       [[thread_position_in_threadgroup]], \
        ushort ntg                         [[threads_per_threadgroup]]) { \
        moe_mm_id_map0_impl<NE>(args, src2, htpe, hids, shmem, tpitg, ntg); \
    }

MOE_MM_ID_MAP0_VARIANT(1)
MOE_MM_ID_MAP0_VARIANT(2)
MOE_MM_ID_MAP0_VARIANT(4)
MOE_MM_ID_MAP0_VARIANT(5)
MOE_MM_ID_MAP0_VARIANT(6)
MOE_MM_ID_MAP0_VARIANT(8)
MOE_MM_ID_MAP0_VARIANT(10)
MOE_MM_ID_MAP0_VARIANT(16)
MOE_MM_ID_MAP0_VARIANT(22)
