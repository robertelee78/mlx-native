// flash_attn_vec_reduce_tq_hb_undo.metal — ADR-028 §iter-485 H3 fusion prototype.
//
// Phase 7d Worker C: TQ-HB SDPA output + FWHT-sign-undo fusion.
//
// FUSION RATIONALE
// ================
// Current TQ-HB SDPA chain (per layer, decode):
//   1. fwht_sign_premult_f32 on Q                  (1 dispatch + barrier)
//   2. flash_attn_vec_tq_hb_dk256                  (1 dispatch + barrier)
//   3. flash_attn_vec_reduce_dk256                 (1 dispatch + barrier)  ← FUSE HERE
//   4. fwht_sign_undo_f32_d256 on output           (1 dispatch + barrier)  ← FOLD IN
//   5. o_proj
//
// H3 (this kernel): fuse 3 + 4 into a single kernel that does the cross-WG
// online-softmax reduce, divides by S_total, then in-place applies the
// inverse-FWHT-then-sign-undo on the final output row before the global store.
//
// Each threadgroup handles ONE output row (one query head). The reduce uses
// `32 * NWG` threads (one simdgroup of 32 with one thread per WG). After the
// reduce produces the final per-head output in registers/threadgroup-mem, we
// run a 32-thread simdgroup-butterfly FWHT (EPT=8 for D=256, EPT=16 for D=512)
// + sign-mask flip identical to fwht_sign_undo_fast.
//
// CHESTERTON'S FENCE
// ==================
// The original flash_attn_vec_reduce kernel + standalone fwht_sign_undo kernel
// are NOT modified. This is a new variant gated by env flag HF2Q_TQ_HB_OUT_FUSED=1.
// Default OFF until parity + delta clears the +3% gate.

#include <metal_stdlib>
using namespace metal;

// Reduce params (matches FlashAttnVecReduceParams in flash_attn_vec.metal).
struct FlashAttnVecReduceTqHbUndoParams {
    uint nrows;
};

// Sign-mask tables — MUST match TBQ_SIGNS_{256,512} in fwht_standalone.metal
// byte-for-byte. Copy-pasted here so the fused kernel is self-contained.
constant uint8_t TBQ_SIGNS_256_REDUCE[32] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
};
constant uint8_t TBQ_SIGNS_512_REDUCE[64] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
};

// SIMD butterfly — identical to fwht_simd in fwht_standalone.metal.
inline void butterfly_local_undo(thread float &a, thread float &b) {
    float sum = a + b;
    float diff = a - b;
    a = sum;
    b = diff;
}

template<ushort EPT>
inline void fwht_simd_undo(thread float *elems, uint lane) {
    for (ushort h = 1; h < EPT; h <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            ushort partner = i ^ h;
            if (partner > i) {
                butterfly_local_undo(elems[i], elems[partner]);
            }
        }
    }
    for (ushort delta = 1; delta < 32; delta <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            float partner_val = simd_shuffle_xor(elems[i], delta);
            if (lane & delta) {
                elems[i] = partner_val - elems[i];
            } else {
                elems[i] = elems[i] + partner_val;
            }
        }
    }
}

// Fused reduce + FWHT-sign-undo.
//
// Grid: (nrows, 1, 1)   Threadgroup: (32 * NWG, 1, 1)
//
// Layout matches flash_attn_vec_reduce:
//   - htmp holds per-WG partial outputs followed by S/M scalars.
//   - One thread per WG (tiisg) loads its WG's S, M, then SIMD-reduces.
//
// Difference from the unfused reduce: instead of writing the per-D4 chunk
// directly to dst, we first accumulate the final output for THIS HEAD into
// threadgroup memory (`out_tg[DV]`), then run the 32-thread butterfly+sign
// pass on it, then write the inverse-rotated result to dst.
//
// Note: this kernel requires the threadgroup width to be EXACTLY 32 (one
// simdgroup) for the butterfly pass. NWG is the per-WG count from the SDPA
// kernel — at NWG=16 or 32 the threadgroup is 32*NWG wide, but we only use
// the first 32 threads (one simdgroup) for the FWHT. Other simdgroups
// participate in the load+reduce, then write their per-D4 partials into
// `out_tg` indexed by their D4 chunk, before the first simdgroup runs the
// butterfly.
template<short DV>
kernel void flash_attn_vec_reduce_tq_hb_undo(
    constant FlashAttnVecReduceTqHbUndoParams  &params [[buffer(0)]],
    device const float                          *htmp   [[buffer(1)]],
    device       float                          *dst    [[buffer(2)]],
    constant     uint                           &nwg_param [[buffer(3)]],
    threadgroup  float                          *out_tg [[threadgroup(0)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short DV4 = DV / 4;
    constexpr ushort EPT = DV / 32;  // 8 for D=256, 16 for D=512

    const uint NWG = nwg_param;
    const uint64_t rid = tgpig;  // row index (one head per threadgroup)
    const ushort iwg = tiisg;    // each thread handles one workgroup

    // S and M values are stored after all DV data (same layout as
    // flash_attn_vec_reduce — htmp packing is dictated by the SDPA kernel).
    device const float *sm = htmp + (uint64_t)params.nrows * DV * NWG;

    // Load this workgroup's S and M.
    float S_wg = (iwg < NWG) ? sm[rid * (2 * NWG) + 2 * iwg + 0] : 0.0f;
    float M_wg = (iwg < NWG) ? sm[rid * (2 * NWG) + 2 * iwg + 1] : -FLT_MAX / 2;

    // Find global max across all workgroups (simdgroup reduce, lanes 0..NWG-1).
    const float M_global = simd_max(M_wg);
    const float ms = exp(M_wg - M_global);
    float S_total = simd_sum(S_wg * ms);
    float inv_S = (S_total == 0.0f) ? 0.0f : 1.0f / S_total;

    // Pointers to interleaved partial results.
    device const float4 *htmp4 = (device const float4 *)htmp + rid * DV4 * NWG;

    // Reduce: for each D4 chunk, sum the rescaled contributions from all WGs.
    // Each thread of simdgroup `sgitg` handles output chunks with stride `NWG`.
    // We write the reduced float4 (divided by S_total) into `out_tg`, which is
    // shared between all simdgroups in this threadgroup.
    threadgroup float4 *out_tg4 = (threadgroup float4 *)out_tg;
    for (short i = sgitg; i < DV4; i += NWG) {
        float4 val = (iwg < NWG) ? htmp4[i * NWG + iwg] * ms : float4(0.0f);
        float4 reduced = float4(simd_sum(val[0]),
                                simd_sum(val[1]),
                                simd_sum(val[2]),
                                simd_sum(val[3]));
        if (iwg == 0) {
            out_tg4[i] = reduced * inv_S;
        }
    }

    // Synchronize: all simdgroups must finish writing out_tg before
    // simdgroup 0 reads it for the FWHT.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- FWHT (= IWHT for normalized H) + sign-undo, by simdgroup 0 ---
    //
    // Mirrors fwht_sign_undo_fast: load EPT consecutive elements per lane,
    // run the simd butterfly, normalize by 1/sqrt(d), apply sign mask, store.
    if (sgitg == 0) {
        const uint lane = tiisg;
        const uint base = lane * EPT;
        float elems[EPT];
        for (ushort i = 0; i < EPT; i++) {
            elems[i] = out_tg[base + i];
        }

        // FWHT + normalize.
        fwht_simd_undo<EPT>(elems, lane);
        const float inv_sqrt_d = rsqrt(float(DV));
        for (ushort i = 0; i < EPT; i++) {
            elems[i] *= inv_sqrt_d;
        }

        // Sign undo (same table as fwht_sign_premult; sign is self-inverse).
        for (ushort i = 0; i < EPT; i++) {
            ushort j = lane * EPT + i;
            uint8_t sign_byte = (DV == 256) ? TBQ_SIGNS_256_REDUCE[j >> 3]
                                            : TBQ_SIGNS_512_REDUCE[j >> 3];
            float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
            elems[i] *= sign_val;
        }

        // Store final output for this row to global memory.
        device float *dst_row = dst + rid * DV;
        for (ushort i = 0; i < EPT; i++) {
            dst_row[base + i] = elems[i];
        }
    }
}

typedef decltype(flash_attn_vec_reduce_tq_hb_undo<256>) flash_attn_vec_reduce_tq_hb_undo_t;

template [[host_name("flash_attn_vec_reduce_tq_hb_undo_dk256")]]
kernel flash_attn_vec_reduce_tq_hb_undo_t flash_attn_vec_reduce_tq_hb_undo<256>;

template [[host_name("flash_attn_vec_reduce_tq_hb_undo_dk512")]]
kernel flash_attn_vec_reduce_tq_hb_undo_t flash_attn_vec_reduce_tq_hb_undo<512>;
