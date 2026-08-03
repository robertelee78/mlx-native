// hadamard_quantize_kv_fast.metal — FWHT + quantize using SIMD shuffle (zero threadgroup barriers)
//
// Replaces hadamard_quantize_kv.metal. Same algorithm, but the FWHT butterfly
// uses simd_shuffle_xor instead of shared memory + threadgroup barriers.
//
// Architecture: 1 simdgroup (32 threads) per KV head.
// Each thread holds head_dim/32 elements in registers.
// - head_dim=256: 8 elements/thread, 8 butterfly stages (3 local + 5 shuffle)
// - head_dim=512: 16 elements/thread, 9 butterfly stages (4 local + 5 shuffle)
//
// Reference: HadaCore (arxiv 2412.08832) SIMD butterfly pattern.
//
// D1 random sign pre-multiplication (SRHT) per ADR-007 iter-13 shipping-impl study + iter-14 hypothesis test.
// Sign table verbatim from AmesianX TurboQuant at ggml-cuda/cpy-utils.cuh:158-163 (D=256) + :211-220 (D=512).
// Without D1, plain-WHT fails to decorrelate structured K/V (real Gemma activations, not random Gaussian).
// Sign is applied BEFORE WHT in encode + AFTER IWHT in decode (self-inverse since sign*sign=1).

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// D1 sign table for D=256 (32 bytes, 256 bits).
// Verbatim from AmesianX cpy-utils.cuh:158-163. sha256=3ef1038e6c232e9519101daa2d6efd637d4c6bfdb29f4ee7101625c39d0ddc89
// Bit j = (table[j>>3] >> (j&7)) & 1; bit=1 → sign=-1, bit=0 → sign=+1 (LSB-first).
constant uint8_t TBQ_SIGNS_256[32] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
};

// D1 sign table for D=512 (64 bytes, 512 bits).
// Verbatim from AmesianX cpy-utils.cuh:211-220. sha256=44f13ce9f6db1edac62f558ee054f9de29cd474fd051362cadcaa98a55745f17
// Same convention: bit j → table_512[j>>3] >> (j&7); bit=1 → sign=-1, bit=0 → sign=+1.
constant uint8_t TBQ_SIGNS_512[64] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
};

constant float BOUNDARIES_4BIT[15] = {
    -2.4008034f, -1.8435318f, -1.4371388f, -1.0992859f,
    -0.7995498f, -0.5224037f, -0.2582217f,  0.0000000f,
     0.2582217f,  0.5224037f,  0.7995498f,  1.0992859f,
     1.4371388f,  1.8435318f,  2.4008034f,
};

struct HadamardQuantizeParams {
    uint head_dim;
    uint num_kv_heads;
    uint write_pos;
    uint cache_capacity;
    uint is_sliding;
    // iter-18 S2B: D=512 per-block scale factor ablation.
    // Passed from Rust at dispatch time via HF2Q_SCALE_FORMULA env var.
    // bare=1.0  (iter-16 control), sqrt256=16.0, sqrt512≈22.627.
    // ONLY applied to D=512 path. D=256 path is unchanged.
    float scale_factor_d512;
    // iter-19 A1: post-scale RMS probe flag (catalog #21 fix).
    // When non-zero, kernel writes ALL EPT post-scale values per lane to scratch buffer.
    // Layout: rms_scratch[head_idx * HEAD_DIM + lane * EPT + i].
    //   D=256: each lane writes EPT=8 elements → 32 * 8 = 256 samples per block per head.
    //   D=512: blk 0 (lanes 0..15) writes EPT=16 each → 256 samples; blk 1 (lanes 16..31) writes 256.
    //          Layout: rms_scratch[head_idx * 512 + lane * 16 + i] (contiguous; blk0=[0..255], blk1=[256..511]).
    // Host divisor: 256 per block (D=256: sum over all 256 samples; D=512 blk0: samples [0..255]).
    uint rms_probe_enabled;
};

// Butterfly operation on a local element pair.
inline void butterfly_local(thread float &a, thread float &b) {
    float sum = a + b;
    float diff = a - b;
    a = sum;
    b = diff;
}

// FWHT using SIMD shuffle — zero threadgroup barriers.
// EPT = elements per thread (head_dim / 32).
// Each thread holds EPT consecutive elements from the head vector.
template<ushort EPT>
inline void fwht_simd(thread float *elems, uint lane) {
    // Stage 1: local butterfly stages (h < EPT)
    // h=1: pairs (0,1), (2,3), ...
    // h=2: pairs (0,2), (1,3), ...
    // ... up to h=EPT/2
    for (ushort h = 1; h < EPT; h <<= 1) {
        for (ushort i = 0; i < EPT; i++) {
            ushort partner = i ^ h;
            if (partner > i) {
                butterfly_local(elems[i], elems[partner]);
            }
        }
    }

    // Stage 2: cross-thread butterfly stages (h >= EPT)
    // h=EPT: exchange with thread lane^1
    // h=2*EPT: exchange with thread lane^2
    // h=4*EPT: exchange with thread lane^4
    // ... up to h=16*EPT (lane^16 for 32-thread simd)
    for (ushort delta = 1; delta < 32; delta <<= 1) {
        // Each element i in this thread exchanges with element i in thread (lane ^ delta).
        // The butterfly pair is (global_idx, global_idx ^ (delta * EPT)).
        // global_idx = lane * EPT + i, partner_global = (lane ^ delta) * EPT + i.
        for (ushort i = 0; i < EPT; i++) {
            float partner_val = simd_shuffle_xor(elems[i], delta);
            // Determine if this thread does (a+b) or (a-b).
            // The lower-indexed thread gets the sum, the higher gets the difference.
            if (lane & delta) {
                elems[i] = partner_val - elems[i];
            } else {
                elems[i] = elems[i] + partner_val;
            }
        }
    }
}

// Quantize one KV head's vector: load → FWHT → normalize → quantize → pack nibbles.
// 1 simdgroup per head. 32 threads. Each thread handles EPT = head_dim/32 elements.
//
// D=256 path: single global L2 norm, stored at norms[head * capacity + pos].
//
// D=512 path (ADR-007 iter-15 per-256-block norm, per AmesianX cpy-utils.cuh:241-269):
//   After full 512-FWHT the vector is split into 2 contiguous 256-halves (block 0 = [0..255],
//   block 1 = [256..511]). Each half gets an independent RMS norm. The norms buffer is indexed
//   as norms[head * capacity * NORMS_PER_POS + pos * NORMS_PER_POS + blk] where
//   NORMS_PER_POS = 1 for D=256, NORMS_PER_POS = 2 for D=512.
//   Lane assignment: for EPT=16, lane 0..15 own elements 0..255 (block 0),
//                                lane 16..31 own elements 256..511 (block 1).
//   Cite: AmesianX cpy-utils.cuh:241-269 (queen-verified); ADR-007 iter-14 D1 SRHT + iter-15 per-block norm.
template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_fast(
    device const float              *src        [[buffer(0)]],
    device       uint8_t            *packed     [[buffer(1)]],
    device       float              *norms      [[buffer(2)]],
    constant HadamardQuantizeParams &params     [[buffer(3)]],
    device       float              *rms_scratch [[buffer(4)]],  // iter-18 S2A: post-scale probe (may be null/unused when rms_probe_enabled=0)
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgid;
    const uint lane = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    // 1. Load EPT elements into registers.
    const uint src_base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = src[src_base + i];
    }

    // 1b. D1 sign pre-multiplication (SRHT) — applied BEFORE FWHT.
    // Select table by HEAD_DIM at compile time (constexpr branch).
    // Element global index j = lane * EPT + i.
    // sign_bit = (table[j>>3] >> (j&7)) & 1; sign = bit ? -1.0f : +1.0f.
    // Mirror of AmesianX cpy-utils.cuh:181 (D=256) / :229 (D=512).
    {
        for (ushort i = 0; i < EPT; i++) {
            ushort j = lane * EPT + i;  // global element index within head
            uint8_t sign_byte;
            if (HEAD_DIM == 256) {
                sign_byte = TBQ_SIGNS_256[j >> 3];
            } else {
                sign_byte = TBQ_SIGNS_512[j >> 3];
            }
            float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
            elems[i] *= sign_val;
        }
    }

    // 2. FWHT via SIMD shuffle (ZERO threadgroup barriers).
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize by 1/sqrt(head_dim) (normalized WHT convention).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= inv_sqrt_d;
    }

    // 4. Compute norm(s) via SIMD reduction (ZERO threadgroup barriers).
    //
    // D=256: single global L2 norm over all 256 elements.
    // D=512: 2 per-block RMS norms per AmesianX cpy-utils.cuh:241-269.
    //   Block 0 = elements [0..255], owned by lanes 0..15 (EPT=16 → lane*16+i in [0..255] iff lane<16).
    //   Block 1 = elements [256..511], owned by lanes 16..31.
    //   RMS norm: blk_norm[b] = sqrt(sum_sq[b] / 256.0f) where sum_sq[b] includes inv_sqrt_d factor.
    //   This matches AmesianX decode convention when decode uses: scale = blk_norm[b] * inv_sqrt(512).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) {
        local_sq_sum += elems[i] * elems[i];
    }

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        // Single global L2 norm (unchanged D=256 path).
        float norm = sqrt(simd_sum(local_sq_sum));
        norm0 = norm;
        norm1 = 0.0f;  // unused for D=256
    } else {
        // D=512: per-block RMS norms.
        // Lane 0..15 (block 0): contribute to blk0_sq; lanes 16..31 zero out.
        // Lane 16..31 (block 1): contribute to blk1_sq; lanes 0..15 zero out.
        float blk0_contribution = (lane < 16u) ? local_sq_sum : 0.0f;
        float blk1_contribution = (lane >= 16u) ? local_sq_sum : 0.0f;
        float blk0_sq = simd_sum(blk0_contribution);  // sum over all 32 lanes (blk1 contributes 0)
        float blk1_sq = simd_sum(blk1_contribution);  // sum over all 32 lanes (blk0 contributes 0)
        // RMS norm per block (256 elements each).
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 5. Normalize each element: scale to N(0,1) using per-block norm.
    //    D=256: scale = inv_norm0 * sqrt(256) (unchanged — single-global-norm, algebraically
    //           equivalent to AmesianX for the single-norm case).
    //    D=512 (iter-16 fix): scale = inv_blk_norm only. FWHT is normalized (inv_sqrt_d applied
    //           in step 3), so blk_norm ≈ 1 after FWHT → stored element ≈ N(0,1) via 1/norm alone.
    //           AmesianX cpy-utils.cuh:241-269 works on UNNORMALIZED 512-WHT and uses
    //           val = blk_data * inv_norm — no sqrt factor. Our normalized FWHT + AmesianX's
    //           sqrt factor = double normalization → quantizer input RMS = sqrt(512) ≈ 22.6
    //           instead of ~1.0, grossly misfitting N(0,1) codebook. Fix: remove sqrt(HEAD_DIM).
    //    Decode recovers: CODEBOOK[idx] * blk_norm = FWHT_norm(sign*x)[j].
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) {
            elems[i] *= scale;
        }
    } else {
        // D=512: apply per-block scale with ablation factor.
        // iter-18 S2B: scale = inv_blk_norm * params.scale_factor_d512.
        //   bare (1.0):       iter-16 control — inv_blk_norm only.
        //   sqrt256 (16.0):   hypothesis — matches unnormalized FWHT convention.
        //   sqrt512 (≈22.627): iter-15 regression (known FAIL from iter 15/16).
        // Decoder MUST apply the reciprocal: blk_norm / scale_factor_d512.
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * params.scale_factor_d512;
        for (ushort i = 0; i < EPT; i++) {
            elems[i] *= scale;
        }
    }

    // 5b. iter-19 A1: post-scale RMS probe — ALL lanes write ALL EPT post-scale values (catalog #21 fix).
    //     FIXED from iter-18: iter-18 wrote only 8 of 16 EPT samples for D=256 (8 real + 8 zeros)
    //     and the host divided by 16 → reported RMS ≈ sqrt(0.5) × true_RMS ≈ 0.7039. Fix: all 32
    //     lanes each write EPT samples → 256 samples per block; host divides by 256.
    //
    //     Layout: rms_scratch[head_idx * HEAD_DIM + lane * EPT + i]
    //       D=256: HEAD_DIM=256, EPT=8, 32 lanes × 8 = 256 samples per head (1 block).
    //       D=512: HEAD_DIM=512, EPT=16, lanes 0..15 (blk0) at offsets 0..255; lanes 16..31 (blk1) at 256..511.
    //     Host reads: rms_scratch[head_idx*HEAD_DIM .. head_idx*HEAD_DIM+256] for blk 0,
    //                 rms_scratch[head_idx*HEAD_DIM+256 .. head_idx*HEAD_DIM+512] for blk 1 (D=512 only).
    //     Host divisor: 256 per block (not 16, not HEAD_DIM/2).
    if (params.rms_probe_enabled != 0u && rms_scratch != nullptr) {
        // Every lane writes its EPT elements; scratch is contiguous by [head_idx * HEAD_DIM + lane * EPT + i].
        uint scratch_base = head_idx * HEAD_DIM + lane * EPT;
        for (ushort i = 0; i < EPT; i++) {
            rms_scratch[scratch_base + i] = elems[i];
        }
    }

    // 6. Quantize each element: find nearest Lloyd-Max centroid.
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx = 0;
        // Unrolled binary search (4 comparisons for 16 centroids).
        idx = (v > BOUNDARIES_4BIT[7]) ? 8 : 0;
        idx += (v > BOUNDARIES_4BIT[idx + 3]) ? 4 : 0;
        idx += (v > BOUNDARIES_4BIT[idx + 1]) ? 2 : 0;
        idx += (v > BOUNDARIES_4BIT[idx]) ? 1 : 0;
        indices[i] = idx;
    }

    // 7. Pack nibbles and write.
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;

    const uint packed_row_stride = HEAD_DIM / 2;
    const uint packed_base = head_idx * params.cache_capacity * packed_row_stride
                           + actual_pos * packed_row_stride;

    // Each thread writes EPT/2 bytes (EPT elements → EPT/2 nibble pairs).
    const uint byte_base = packed_base + lane * (EPT / 2);
    for (ushort i = 0; i < EPT; i += 2) {
        uint8_t lo = indices[i] & 0xFu;
        uint8_t hi = (indices[i + 1] & 0xFu) << 4;
        packed[byte_base + i / 2] = lo | hi;
    }

    // 8. Store norm(s).
    //    D=256: 1 norm at norms[head * capacity + pos] (NORMS_PER_POS = 1).
    //    D=512: 2 norms at norms[head * capacity * 2 + pos * 2 + blk] (NORMS_PER_POS = 2).
    //    Per AmesianX cpy-utils.cuh:256 y[blk].d = __float2half(blk_norm).
    if (HEAD_DIM == 256) {
        if (lane == 0) {
            uint norm_idx = head_idx * params.cache_capacity + actual_pos;
            norms[norm_idx] = norm0;
        }
    } else {
        // D=512: lane 0 writes norm0 (block 0), lane 16 writes norm1 (block 1).
        if (lane == 0u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
    }
}

// Instantiations for Gemma 4 head dimensions.
template [[host_name("hadamard_quantize_kv_fast_d256")]]
kernel void hadamard_quantize_kv_fast<256>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeParams &, device float *, uint, uint);

template [[host_name("hadamard_quantize_kv_fast_d512")]]
kernel void hadamard_quantize_kv_fast<512>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeParams &, device float *, uint, uint);

// ============================================================================
// ADR-028 iter-485 (Phase 7d / H4): fused K+V single-position 4-bit encoder.
//
// Combines the two `hadamard_quantize_kv_fast` dispatches (lines 3116/3127 in
// forward_mlx.rs) that write the F32 shadow TQ-packed K/V caches at every
// gemma4 decode-layer-token boundary. Grid is `(num_kv_heads, 1, 2)` with
// `tgpig.z = 0` selecting the K stream and `tgpig.z = 1` selecting the V
// stream — exactly the same Z-dim pattern as `hadamard_quantize_kv_hb_dual`,
// but emits 4-bit nibble-packed output (head_dim/2 bytes/pos) and reads the
// single-norm `HadamardQuantizeParams` struct (no `codebook_bits` field).
//
// Saves ONE Apple Metal kernel-launch floor (~14 µs) per layer per decode
// token. At gemma4 30 layers that drops 60→30 KV-write dispatches/decode-
// token, ~0.4 ms/token (~3% theoretical).
//
// Result is byte-identical to two `hadamard_quantize_kv_fast` dispatches at
// identical params (verified by mlx-native unit test
// `test_hadamard_quantize_kv_fast_dual_byte_identity_d256`). The RMS scratch
// probe path is intentionally NOT carried into the fused variant — that
// debug-only probe (HF2Q_DEBUG_TQ_RMS) still routes through the single-
// stream kernel, which is unmodified.
template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_fast_dual(
    device const float              *src_k    [[buffer(0)]],
    device const float              *src_v    [[buffer(1)]],
    device       uint8_t            *packed_k [[buffer(2)]],
    device       uint8_t            *packed_v [[buffer(3)]],
    device       float              *norms_k  [[buffer(4)]],
    device       float              *norms_v  [[buffer(5)]],
    constant HadamardQuantizeParams &params   [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgpig.x;
    const uint kv_sel   = tgpig.z; // 0 = K stream, 1 = V stream
    const uint lane     = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    device const float   *src    = (kv_sel == 0u) ? src_k    : src_v;
    device       uint8_t *packed = (kv_sel == 0u) ? packed_k : packed_v;
    device       float   *norms  = (kv_sel == 0u) ? norms_k  : norms_v;

    // 1. Load elements.
    const uint src_base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) {
        elems[i] = src[src_base + i];
    }

    // 1b. D1 sign pre-multiplication (SRHT).
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = (HEAD_DIM == 256) ? TBQ_SIGNS_256[j >> 3] : TBQ_SIGNS_512[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    // 2. FWHT via SIMD shuffle.
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize by 1/sqrt(head_dim).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) {
        elems[i] *= inv_sqrt_d;
    }

    // 4. Compute norm(s).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) {
        local_sq_sum += elems[i] * elems[i];
    }

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        norm0 = sqrt(simd_sum(local_sq_sum));
        norm1 = 0.0f;
    } else {
        // D=512: per-block RMS norms. Mirror single-stream pattern at line 216-222
        // (mask-then-sum; broadcast not needed because `simd_sum` is already
        // uniform across the simdgroup).
        float blk0_contribution = (lane < 16u) ? local_sq_sum : 0.0f;
        float blk1_contribution = (lane >= 16u) ? local_sq_sum : 0.0f;
        float blk0_sq = simd_sum(blk0_contribution);
        float blk1_sq = simd_sum(blk1_contribution);
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 5. Scale to N(0,1) — mirrors `hadamard_quantize_kv_fast`.
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    } else {
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * params.scale_factor_d512;
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    }

    // 6. Quantize each element: 4-bit Lloyd-Max via unrolled binary search
    //    over BOUNDARIES_4BIT[15] (4 comparisons for 16 centroids).
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx = 0;
        idx = (v > BOUNDARIES_4BIT[7]) ? 8 : 0;
        idx += (v > BOUNDARIES_4BIT[idx + 3]) ? 4 : 0;
        idx += (v > BOUNDARIES_4BIT[idx + 1]) ? 2 : 0;
        idx += (v > BOUNDARIES_4BIT[idx]) ? 1 : 0;
        indices[i] = idx;
    }

    // 7. Pack nibbles and write.
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;
    const uint packed_row_stride = HEAD_DIM / 2;
    const uint packed_base = head_idx * params.cache_capacity * packed_row_stride
                           + actual_pos * packed_row_stride;
    const uint byte_base = packed_base + lane * (EPT / 2);
    for (ushort i = 0; i < EPT; i += 2) {
        uint8_t lo = indices[i] & 0xFu;
        uint8_t hi = (indices[i + 1] & 0xFu) << 4;
        packed[byte_base + i / 2] = lo | hi;
    }

    // 8. Store norm(s).
    if (HEAD_DIM == 256) {
        if (lane == 0u) {
            norms[head_idx * params.cache_capacity + actual_pos] = norm0;
        }
    } else {
        if (lane == 0u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
    }
}

template [[host_name("hadamard_quantize_kv_fast_dual_d256")]]
kernel void hadamard_quantize_kv_fast_dual<256>(
    device const float *, device const float *,
    device uint8_t *, device uint8_t *,
    device float *, device float *,
    constant HadamardQuantizeParams &, uint3, uint);

template [[host_name("hadamard_quantize_kv_fast_dual_d512")]]
kernel void hadamard_quantize_kv_fast_dual<512>(
    device const float *, device const float *,
    device uint8_t *, device uint8_t *,
    device float *, device float *,
    constant HadamardQuantizeParams &, uint3, uint);

// ============================================================================
// Track B (iter-21): higher-bit codebooks for ablation.
// 5-bit (32 centroids) and 6-bit (64 centroids) Lloyd-Max codebooks for N(0,1).
// Byte-packed: 1 byte per element (upper 3 or 2 bits zeroed).
// Packed buffer layout: [num_kv_heads, capacity, head_dim] u8 (one byte per index).
// ============================================================================

constant float BOUNDARIES_5BIT[31] = {
    -2.9758689f, -2.5043666f, -2.1731675f, -1.9079127f,
    -1.6816622f, -1.4812169f, -1.2990074f, -1.1302326f,
    -0.9716178f, -0.8207999f, -0.6759910f, -0.5357806f,
    -0.3990113f, -0.2646964f, -0.1319613f,  0.0000000f,
     0.1319613f,  0.2646964f,  0.3990113f,  0.5357806f,
     0.6759910f,  0.8207999f,  0.9716178f,  1.1302326f,
     1.2990074f,  1.4812169f,  1.6816622f,  1.9079127f,
     2.1731675f,  2.5043666f,  2.9758689f,
};

constant float BOUNDARIES_6BIT[63] = {
    -3.4451688f, -3.0273920f, -2.7400952f, -2.5145300f,
    -2.3258894f, -2.1620828f, -2.0162282f, -1.8840057f,
    -1.7625122f, -1.6496952f, -1.5440449f, -1.4444151f,
    -1.3499116f, -1.2598213f, -1.1735638f, -1.0906583f,
    -1.0107003f, -0.9333442f, -0.8582906f, -0.7852769f,
    -0.7140697f, -0.6444593f, -0.5762552f, -0.5092820f,
    -0.4433773f, -0.3783886f, -0.3141717f, -0.2505889f,
    -0.1875076f, -0.1247993f, -0.0623381f,  0.0000000f,
     0.0623381f,  0.1247993f,  0.1875076f,  0.2505889f,
     0.3141717f,  0.3783886f,  0.4433773f,  0.5092820f,
     0.5762552f,  0.6444593f,  0.7140697f,  0.7852769f,
     0.8582906f,  0.9333442f,  1.0107003f,  1.0906583f,
     1.1735638f,  1.2598213f,  1.3499116f,  1.4444151f,
     1.5440449f,  1.6496952f,  1.7625122f,  1.8840057f,
     2.0162282f,  2.1620828f,  2.3258894f,  2.5145300f,
     2.7400952f,  3.0273920f,  3.4451688f,
};

// iter-24: 8-bit boundaries (255 boundaries for 256 centroids), derived from
// the Lloyd-Max CODEBOOK_8BIT table in turboquant.rs (converged to tol=1e-12,
// symmetry error 3.41e-10 — the codebook's single source of truth).
// BOUNDARIES_8BIT[i] = midpoint(CODEBOOK_8BIT[i], CODEBOOK_8BIT[i+1]).
constant float BOUNDARIES_8BIT[255] = {
    -4.8744828f, -4.5652095f, -4.3591350f, -4.2013707f,
    -4.0722382f, -3.9621974f, -3.8658436f, -3.7797987f,
    -3.7018022f, -3.6302633f, -3.5640175f, -3.5021860f,
    -3.4440881f, -3.3891853f, -3.3370443f, -3.2873106f,
    -3.2396907f, -3.1939384f, -3.1498455f, -3.1072339f,
    -3.0659504f, -3.0258620f, -2.9868525f, -2.9488201f,
    -2.9116746f, -2.8753363f, -2.8397341f, -2.8048042f,
    -2.7704897f, -2.7367390f, -2.7035057f, -2.6707477f,
    -2.6384267f, -2.6065081f, -2.5749600f, -2.5437535f,
    -2.5128620f, -2.4822612f, -2.4519288f, -2.4218442f,
    -2.3919885f, -2.3623444f, -2.3328958f, -2.3036278f,
    -2.2745269f, -2.2455802f, -2.2167763f, -2.1881042f,
    -2.1595539f, -2.1311163f, -2.1027826f, -2.0745450f,
    -2.0463962f, -2.0183292f, -1.9903379f, -1.9624162f,
    -1.9345590f, -1.9067610f, -1.8790177f, -1.8513248f,
    -1.8236783f, -1.7960745f, -1.7685100f, -1.7409817f,
    -1.7134865f, -1.6860220f, -1.6585855f, -1.6311747f,
    -1.6037875f, -1.5764221f, -1.5490764f, -1.5217490f,
    -1.4944383f, -1.4671427f, -1.4398612f, -1.4125923f,
    -1.3853351f, -1.3580886f, -1.3308517f, -1.3036237f,
    -1.2764037f, -1.2491911f, -1.2219851f, -1.1947853f,
    -1.1675909f, -1.1404016f, -1.1132168f, -1.0860362f,
    -1.0588593f, -1.0316859f, -1.0045155f, -0.9773478f,
    -0.9501827f, -0.9230199f, -0.8958592f, -0.8687003f,
    -0.8415430f, -0.8143873f, -0.7872329f, -0.7600798f,
    -0.7329278f, -0.7057767f, -0.6786266f, -0.6514772f,
    -0.6243286f, -0.5971806f, -0.5700331f, -0.5428862f,
    -0.5157398f, -0.4885937f, -0.4614481f, -0.4343027f,
    -0.4071577f, -0.3800128f, -0.3528683f, -0.3257239f,
    -0.2985797f, -0.2714356f, -0.2442917f, -0.2171479f,
    -0.1900042f, -0.1628606f, -0.1357171f, -0.1085736f,
    -0.0814302f, -0.0542868f, -0.0271434f,  0.0000000f,
     0.0271434f,  0.0542868f,  0.0814302f,  0.1085736f,
     0.1357171f,  0.1628606f,  0.1900042f,  0.2171479f,
     0.2442917f,  0.2714356f,  0.2985797f,  0.3257239f,
     0.3528683f,  0.3800128f,  0.4071577f,  0.4343027f,
     0.4614481f,  0.4885937f,  0.5157398f,  0.5428862f,
     0.5700331f,  0.5971806f,  0.6243286f,  0.6514772f,
     0.6786266f,  0.7057767f,  0.7329278f,  0.7600798f,
     0.7872329f,  0.8143873f,  0.8415430f,  0.8687003f,
     0.8958592f,  0.9230199f,  0.9501827f,  0.9773478f,
     1.0045155f,  1.0316859f,  1.0588593f,  1.0860362f,
     1.1132168f,  1.1404016f,  1.1675909f,  1.1947853f,
     1.2219851f,  1.2491911f,  1.2764037f,  1.3036237f,
     1.3308517f,  1.3580886f,  1.3853351f,  1.4125923f,
     1.4398612f,  1.4671427f,  1.4944383f,  1.5217490f,
     1.5490764f,  1.5764221f,  1.6037875f,  1.6311747f,
     1.6585855f,  1.6860220f,  1.7134865f,  1.7409817f,
     1.7685100f,  1.7960745f,  1.8236783f,  1.8513248f,
     1.8790177f,  1.9067610f,  1.9345590f,  1.9624162f,
     1.9903379f,  2.0183292f,  2.0463962f,  2.0745450f,
     2.1027826f,  2.1311163f,  2.1595539f,  2.1881042f,
     2.2167763f,  2.2455802f,  2.2745269f,  2.3036278f,
     2.3328958f,  2.3623444f,  2.3919885f,  2.4218442f,
     2.4519288f,  2.4822612f,  2.5128620f,  2.5437535f,
     2.5749600f,  2.6065081f,  2.6384267f,  2.6707477f,
     2.7035057f,  2.7367390f,  2.7704897f,  2.8048042f,
     2.8397341f,  2.8753363f,  2.9116746f,  2.9488201f,
     2.9868525f,  3.0258620f,  3.0659504f,  3.1072339f,
     3.1498455f,  3.1939384f,  3.2396907f,  3.2873106f,
     3.3370443f,  3.3891853f,  3.4440881f,  3.5021860f,
     3.5640175f,  3.6302633f,  3.7018022f,  3.7797987f,
     3.8658436f,  3.9621974f,  4.0722382f,  4.2013707f,
     4.3591350f,  4.5652095f,  4.8744828f,
};

struct HadamardQuantizeHbParams {
    uint head_dim;
    uint num_kv_heads;
    uint write_pos;
    uint cache_capacity;
    uint is_sliding;
    float scale_factor_d512; // Same semantics as 4-bit path
    uint codebook_bits;      // 5, 6, or 8
};

// Higher-bit quantization kernel: same FWHT + norm as 4-bit, but quantizes to
// 5-bit (32 centroids) or 6-bit (64 centroids) and writes 1 byte per element.
// Packed buffer: [num_kv_heads, capacity, head_dim] u8 (byte-packed).
template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_hb(
    device const float                    *src    [[buffer(0)]],
    device       uint8_t                  *packed [[buffer(1)]],  // byte-packed (1 byte/elem)
    device       float                    *norms  [[buffer(2)]],
    constant HadamardQuantizeHbParams     &params [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgid;
    const uint lane = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    // 1. Load elements.
    const uint src_base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) elems[i] = src[src_base + i];

    // 1b. D1 sign pre-multiplication (SRHT).
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = (HEAD_DIM == 256) ? TBQ_SIGNS_256[j >> 3] : TBQ_SIGNS_512[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    // 2. FWHT.
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize 1/sqrt(d).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) elems[i] *= inv_sqrt_d;

    // 4. Compute norm(s).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) local_sq_sum += elems[i] * elems[i];

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        norm0 = sqrt(simd_sum(local_sq_sum));
        norm1 = 0.0f;
    } else {
        float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
        float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
        blk0_sq = simd_broadcast(blk0_sq, 0u);
        blk1_sq = simd_broadcast(blk1_sq, 16u);
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 5. Scale elements to N(0,1) range for quantization.
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    } else {
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * params.scale_factor_d512;
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    }

    // 6. Quantize with higher-bit codebook (5, 6, or 8-bit).
    const uint cbits = params.codebook_bits;
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx;
        if (cbits == 5u) {
            // 5-bit: 32 centroids, binary search with 5 levels
            idx = (v > BOUNDARIES_5BIT[15]) ? 16 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_5BIT[idx]) ? 1 : 0;
        } else if (cbits == 6u) {
            // 6-bit: 64 centroids, binary search with 6 levels
            idx = (v > BOUNDARIES_6BIT[31]) ? 32 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_6BIT[idx]) ? 1 : 0;
        } else {
            // 8-bit: 256 centroids, binary search with 8 levels
            idx = (v > BOUNDARIES_8BIT[127]) ? 128 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 63]) ? 64 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 31]) ? 32 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 7])  ?  8 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 3])  ?  4 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 1])  ?  2 : 0;
            idx += (v > BOUNDARIES_8BIT[idx])      ?  1 : 0;
        }
        indices[i] = idx;
    }

    // 7. Write byte-packed output (1 byte per element).
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;
    // Packed layout: [head_idx, actual_pos, 0..HEAD_DIM] u8 — byte-packed.
    const uint packed_base = head_idx * params.cache_capacity * HEAD_DIM
                           + actual_pos * HEAD_DIM;
    const uint elem_base = packed_base + lane * EPT;
    for (ushort i = 0; i < EPT; i++) {
        packed[elem_base + i] = indices[i];
    }

    // 8. Store norm(s) — same as 4-bit path.
    if (HEAD_DIM == 256) {
        if (lane == 0) {
            norms[head_idx * params.cache_capacity + actual_pos] = norm0;
        }
    } else {
        if (lane == 0u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
    }
}

template [[host_name("hadamard_quantize_kv_hb_d256")]]
kernel void hadamard_quantize_kv_hb<256>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint, uint);

template [[host_name("hadamard_quantize_kv_hb_d512")]]
kernel void hadamard_quantize_kv_hb<512>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint, uint);

// ============================================================================
// ADR-040 M4 — BATCHED multi-sequence FWHT-V quantize.
//
// Identical FWHT + Lloyd-Max codebook math to `hadamard_quantize_kv_hb<HEAD_DIM>`
// above; ONLY the src/packed/norms base addressing + write_pos become per-query,
// driven by slot_id_arr[iq] / seq_pos_arr[iq]. One dispatch over N queries
// (grid.y = N) replaces the N per-slot host-side dispatches. Per-query math is
// byte-identical to N single-slot calls ⇒ bit-identical by construction.
//
// Grid: 2D — x=head (num_kv_heads), y=query (N). 1 simdgroup (32 lanes) per (head,query).
template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_hb_batched(
    device const float                    *src        [[buffer(0)]],  // [N, nkv*HEAD_DIM] F32
    device       uint8_t                  *packed     [[buffer(1)]],  // [n_seqs, nkv, cap, HEAD_DIM] u8
    device       float                    *norms      [[buffer(2)]],  // [n_seqs, nkv, cap, npp] f32
    constant HadamardQuantizeHbParams     &params     [[buffer(3)]],
    device const uint                     *slot_id_arr[[buffer(4)]],  // [N]
    device const uint                     *seq_pos_arr[[buffer(5)]],  // [N] raw seq position
    uint2 tgid [[threadgroup_position_in_grid]],   // x=head, y=query
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    constexpr ushort NPP = (HEAD_DIM == 256) ? 1 : 2;
    const uint head_idx = tgid.x;
    const uint iq       = tgid.y;
    const uint lane = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    const uint slot = slot_id_arr[iq];
    const uint wpos = seq_pos_arr[iq];

    // 1. Load elements (per-query src row offset).
    const uint src_base = iq * (params.num_kv_heads * HEAD_DIM)
                        + head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) elems[i] = src[src_base + i];

    // 1b. D1 sign pre-multiplication (SRHT).
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = (HEAD_DIM == 256) ? TBQ_SIGNS_256[j >> 3] : TBQ_SIGNS_512[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    // 2. FWHT.
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize 1/sqrt(d).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) elems[i] *= inv_sqrt_d;

    // 4. Compute norm(s).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) local_sq_sum += elems[i] * elems[i];

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        norm0 = sqrt(simd_sum(local_sq_sum));
        norm1 = 0.0f;
    } else {
        float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
        float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
        blk0_sq = simd_broadcast(blk0_sq, 0u);
        blk1_sq = simd_broadcast(blk1_sq, 16u);
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 5. Scale elements to N(0,1) range for quantization.
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    } else {
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * params.scale_factor_d512;
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    }

    // 6. Quantize with higher-bit codebook (5, 6, or 8-bit).
    const uint cbits = params.codebook_bits;
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx;
        if (cbits == 5u) {
            idx = (v > BOUNDARIES_5BIT[15]) ? 16 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_5BIT[idx]) ? 1 : 0;
        } else if (cbits == 6u) {
            idx = (v > BOUNDARIES_6BIT[31]) ? 32 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_6BIT[idx]) ? 1 : 0;
        } else {
            idx = (v > BOUNDARIES_8BIT[127]) ? 128 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 63]) ? 64 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 31]) ? 32 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 7])  ?  8 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 3])  ?  4 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 1])  ?  2 : 0;
            idx += (v > BOUNDARIES_8BIT[idx])      ?  1 : 0;
        }
        indices[i] = idx;
    }

    // 7. Write byte-packed output (per-query slot region).
    uint actual_pos = (params.is_sliding != 0u)
        ? (wpos % params.cache_capacity)
        : wpos;
    const uint slot_packed_base = slot * (params.num_kv_heads * params.cache_capacity * HEAD_DIM);
    const uint packed_base = slot_packed_base
                           + head_idx * params.cache_capacity * HEAD_DIM
                           + actual_pos * HEAD_DIM;
    const uint elem_base = packed_base + lane * EPT;
    for (ushort i = 0; i < EPT; i++) {
        packed[elem_base + i] = indices[i];
    }

    // 8. Store norm(s) — per-query slot region.
    const uint slot_norm_base = slot * (params.num_kv_heads * params.cache_capacity * NPP);
    if (HEAD_DIM == 256) {
        if (lane == 0) {
            norms[slot_norm_base + head_idx * params.cache_capacity + actual_pos] = norm0;
        }
    } else {
        if (lane == 0u) {
            uint norm_base = slot_norm_base + head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = slot_norm_base + head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
    }
}

template [[host_name("hadamard_quantize_kv_hb_batched_d256")]]
kernel void hadamard_quantize_kv_hb_batched<256>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, device const uint *, device const uint *,
    uint2, uint);

template [[host_name("hadamard_quantize_kv_hb_batched_d512")]]
kernel void hadamard_quantize_kv_hb_batched<512>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, device const uint *, device const uint *,
    uint2, uint);

// ============================================================================
// ADR-028 Phase 10e.5 (iter-351): no-FWHT V quantize kernel for the hybrid path.
//
// Same byte-packed Lloyd-Max quantization as `hadamard_quantize_kv_hb` BUT:
//   * NO D1 sign pre-multiply
//   * NO FWHT
//   * NO 1/sqrt(d) post-FWHT normalize
//   * Norm = RMS(raw) computed directly: `sqrt(simd_sum(raw²) / D)`
//
// Result: dequant in SDPA recovers raw V values (not FWHT-rotated).  Combined
// with hybrid F16-K (raw), the entire FWHT chain in attention can be eliminated
// (saves 60 dispatches/decode-token at gemma4 30L: 30 FWHT-pre on Q + 30
// FWHT-undo on output).
//
// Parity hypothesis: V coming into this kernel is already RMS-normalized via
// the layer's pre-attention `dispatch_rms_norm_unit_perhead` → distribution is
// approximately N(0, 1) per head per position. The Lloyd-Max codebook (designed
// for N(0,1)) should achieve quantization NRMSE comparable to the FWHT path
// (~8e-3 at 8-bit), provided V is well-conditioned.  Falsifier: parity test in
// /opt/mlx-native/tests/test_kv_quantize_v_no_fwht.rs measures NRMSE vs raw V;
// if > 5e-2 at 8-bit, hypothesis is FALSIFIED and we must keep FWHT-undo path.
//
// Kernel is V-only (no K variant) because the hybrid path stores K as F16 dense
// — only V needs quantization.  Same byte-packed buffer layout + same norm
// storage layout as `hadamard_quantize_kv_hb` so the dequant side of the SDPA
// kernel doesn't need ANY changes.
// ============================================================================
template<ushort HEAD_DIM>
kernel void kv_quantize_v_no_fwht(
    device const float                    *src    [[buffer(0)]],
    device       uint8_t                  *packed [[buffer(1)]],
    device       float                    *norms  [[buffer(2)]],
    constant HadamardQuantizeHbParams     &params [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgid;
    const uint lane = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    // 1. Load raw elements.
    const uint src_base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) elems[i] = src[src_base + i];

    // 2. Compute L2 norm directly from raw elements (no FWHT, no /sqrt(d)).
    //
    //    Math contract with the SDPA dequant formula
    //    `recovered = centroid * (norm0 * 1/sqrt(d))`:
    //
    //    FWHT path uses `sqrt(simd_sum)` AFTER `/sqrt(d)` normalize → norm0 = 1
    //    for RMS-1 input → dequant scale = 1/sqrt(d) → recovers post-FWHT elem
    //    of magnitude 1/sqrt(d).
    //
    //    No-FWHT path uses `sqrt(simd_sum)` WITHOUT `/sqrt(d)` → norm0 = sqrt(d)
    //    for RMS-1 input → dequant scale = sqrt(d)/sqrt(d) = 1 → recovers raw
    //    elem of magnitude 1.  Same dequant formula in both paths ✓.
    //
    //    For D=512: per-block — block 0 = lanes 0..15, block 1 = lanes 16..31.
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) local_sq_sum += elems[i] * elems[i];

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        norm0 = sqrt(simd_sum(local_sq_sum));
        norm1 = 0.0f;
    } else {
        // D=512: split into two 256-element blocks (lanes 0..15 = blk0, lanes 16..31 = blk1).
        float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
        float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
        blk0_sq = simd_broadcast(blk0_sq, 0u);
        blk1_sq = simd_broadcast(blk1_sq, 16u);
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 3. Scale raw elements to N(0,1) range for quantization.
    //    Same formula as FWHT path step 5 — only the input differs (raw vs rotated).
    //    quant_value = raw * (sqrt(d) / norm) → unit-variance if raw ~ N(0, norm²/d).
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    } else {
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * params.scale_factor_d512;
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    }

    // 4. Quantize via Lloyd-Max codebook (5/6/8-bit).
    //    Identical binary-search logic to `hadamard_quantize_kv_hb` step 6.
    const uint cbits = params.codebook_bits;
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx;
        if (cbits == 5u) {
            idx = (v > BOUNDARIES_5BIT[15]) ? 16 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_5BIT[idx]) ? 1 : 0;
        } else if (cbits == 6u) {
            idx = (v > BOUNDARIES_6BIT[31]) ? 32 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_6BIT[idx]) ? 1 : 0;
        } else {
            // 8-bit
            idx = (v > BOUNDARIES_8BIT[127]) ? 128 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 63]) ? 64 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 31]) ? 32 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 7])  ?  8 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 3])  ?  4 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 1])  ?  2 : 0;
            idx += (v > BOUNDARIES_8BIT[idx])      ?  1 : 0;
        }
        indices[i] = idx;
    }

    // 5. Write byte-packed output (1 byte/elem) — same layout as FWHT path.
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;
    const uint packed_base = head_idx * params.cache_capacity * HEAD_DIM
                           + actual_pos * HEAD_DIM;
    const uint elem_base = packed_base + lane * EPT;
    for (ushort i = 0; i < EPT; i++) {
        packed[elem_base + i] = indices[i];
    }

    // 6. Store norm(s) — same layout as FWHT path so SDPA dequant is unchanged.
    if (HEAD_DIM == 256) {
        if (lane == 0) {
            norms[head_idx * params.cache_capacity + actual_pos] = norm0;
        }
    } else {
        if (lane == 0u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
    }
}

template [[host_name("kv_quantize_v_no_fwht_d256")]]
kernel void kv_quantize_v_no_fwht<256>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint, uint);

template [[host_name("kv_quantize_v_no_fwht_d512")]]
kernel void kv_quantize_v_no_fwht<512>(
    device const float *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint, uint);

// ADR-028 iter-148: fused K+V dual single-position HB encoder.
// Saves one kernel-launch floor (~14 µs/Apple GPU) per layer per
// decode token. Grid Z-dim selects K (z=0) or V (z=1); each
// threadgroup is 1 simdgroup processing one (head, K|V) pair with
// the same FWHT+quantize logic as the single-stream kernel.
// Result is byte-identical to two `hadamard_quantize_kv_hb`
// dispatches at identical params.
template<ushort HEAD_DIM>
kernel void hadamard_quantize_kv_hb_dual(
    device const float                    *src_k    [[buffer(0)]],
    device const float                    *src_v    [[buffer(1)]],
    device       uint8_t                  *packed_k [[buffer(2)]],
    device       uint8_t                  *packed_v [[buffer(3)]],
    device       float                    *norms_k  [[buffer(4)]],
    device       float                    *norms_v  [[buffer(5)]],
    constant HadamardQuantizeHbParams     &params   [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgpig.x;
    const uint kv_sel   = tgpig.z; // 0 = K stream, 1 = V stream
    const uint lane     = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    device const float   *src    = (kv_sel == 0u) ? src_k    : src_v;
    device       uint8_t *packed = (kv_sel == 0u) ? packed_k : packed_v;
    device       float   *norms  = (kv_sel == 0u) ? norms_k  : norms_v;

    // 1. Load elements.
    const uint src_base = head_idx * HEAD_DIM + lane * EPT;
    float elems[EPT];
    for (ushort i = 0; i < EPT; i++) elems[i] = src[src_base + i];

    // 1b. D1 sign pre-multiplication (SRHT).
    for (ushort i = 0; i < EPT; i++) {
        ushort j = lane * EPT + i;
        uint8_t sign_byte = (HEAD_DIM == 256) ? TBQ_SIGNS_256[j >> 3] : TBQ_SIGNS_512[j >> 3];
        float sign_val = ((sign_byte >> (j & 7)) & 1u) ? -1.0f : 1.0f;
        elems[i] *= sign_val;
    }

    // 2. FWHT.
    fwht_simd<EPT>(elems, lane);

    // 3. Normalize 1/sqrt(d).
    const float inv_sqrt_d = rsqrt(float(HEAD_DIM));
    for (ushort i = 0; i < EPT; i++) elems[i] *= inv_sqrt_d;

    // 4. Compute norm(s).
    float local_sq_sum = 0.0f;
    for (ushort i = 0; i < EPT; i++) local_sq_sum += elems[i] * elems[i];

    float norm0, norm1;
    if (HEAD_DIM == 256) {
        norm0 = sqrt(simd_sum(local_sq_sum));
        norm1 = 0.0f;
    } else {
        float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
        float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
        blk0_sq = simd_broadcast(blk0_sq, 0u);
        blk1_sq = simd_broadcast(blk1_sq, 16u);
        norm0 = sqrt(blk0_sq / 256.0f);
        norm1 = sqrt(blk1_sq / 256.0f);
    }

    // 5. Scale to N(0,1).
    if (HEAD_DIM == 256) {
        float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
        float scale = inv_norm * sqrt(float(HEAD_DIM));
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    } else {
        float blk_norm = (lane < 16u) ? norm0 : norm1;
        float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
        float scale = inv_blk_norm * params.scale_factor_d512;
        for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
    }

    // 6. Quantize with selected codebook.
    const uint cbits = params.codebook_bits;
    uint8_t indices[EPT];
    for (ushort i = 0; i < EPT; i++) {
        float v = elems[i];
        uint8_t idx;
        if (cbits == 5u) {
            idx = (v > BOUNDARIES_5BIT[15]) ? 16 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_5BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_5BIT[idx]) ? 1 : 0;
        } else if (cbits == 6u) {
            idx = (v > BOUNDARIES_6BIT[31]) ? 32 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 7]) ? 8 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 3]) ? 4 : 0;
            idx += (v > BOUNDARIES_6BIT[idx + 1]) ? 2 : 0;
            idx += (v > BOUNDARIES_6BIT[idx]) ? 1 : 0;
        } else {
            idx = (v > BOUNDARIES_8BIT[127]) ? 128 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 63]) ? 64 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 31]) ? 32 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 15]) ? 16 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 7])  ?  8 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 3])  ?  4 : 0;
            idx += (v > BOUNDARIES_8BIT[idx + 1])  ?  2 : 0;
            idx += (v > BOUNDARIES_8BIT[idx])      ?  1 : 0;
        }
        indices[i] = idx;
    }

    // 7. Write byte-packed output.
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;
    const uint packed_base = head_idx * params.cache_capacity * HEAD_DIM
                           + actual_pos * HEAD_DIM;
    const uint elem_base = packed_base + lane * EPT;
    for (ushort i = 0; i < EPT; i++) {
        packed[elem_base + i] = indices[i];
    }

    // 8. Store norm(s).
    if (HEAD_DIM == 256) {
        if (lane == 0) {
            norms[head_idx * params.cache_capacity + actual_pos] = norm0;
        }
    } else {
        if (lane == 0u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 0u] = norm0;
        } else if (lane == 16u) {
            uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
            norms[norm_base + 1u] = norm1;
        }
    }
}

template [[host_name("hadamard_quantize_kv_hb_dual_d256")]]
kernel void hadamard_quantize_kv_hb_dual<256>(
    device const float *, device const float *,
    device uint8_t *, device uint8_t *,
    device float *, device float *,
    constant HadamardQuantizeHbParams &, uint3, uint);

template [[host_name("hadamard_quantize_kv_hb_dual_d512")]]
kernel void hadamard_quantize_kv_hb_dual<512>(
    device const float *, device const float *,
    device uint8_t *, device uint8_t *,
    device float *, device float *,
    constant HadamardQuantizeHbParams &, uint3, uint);

// ============================================================================
// ADR-028 Phase 10c.5 (iter-354): fused F16-K-copy + V-no-FWHT-encode kernel.
//
// Combines the two hf2q hybrid-path decode dispatches into one:
//   * z=0 (K stream): F32 src_k → F16 cache write.  Same effect as
//     `kv_cache_copy_batch_f32_to_f16` from the hybrid encode site (Phase 10c).
//   * z=1 (V stream): F32 src_v → byte-packed Lloyd-Max codebook + L2 norm.
//     Same effect as `kv_quantize_v_no_fwht` from Phase 10e.5.
//
// Saves ONE Apple Metal kernel-launch floor (~14 µs) per layer per decode
// token.  At gemma4 30 layers, drops 60→30 KV-write dispatches/decode-token,
// saving ~0.4 ms/token ≈ ~3% theoretical (per iter-351 measured ~1/3 realizes
// → expected +1% decode).
//
// Result is byte-identical to:
//   `dispatch_kv_cache_copy_batch_f32_to_f16(src_k, k_f16)` +
//   `dispatch_kv_quantize_v_no_fwht(src_v, v_packed, v_norms)`
// at identical params.  Each Z-stream takes the SAME math path as its
// stand-alone counterpart (no fused-arithmetic shortcuts).
//
// Threadgroup geometry: (32, 1, 2) = 1 simdgroup × 2 streams = 64 threads.
// Grid: (num_kv_heads, 1, 2).  Same K and V code paths as their respective
// stand-alone kernels above.
// ============================================================================
template<ushort HEAD_DIM>
kernel void kv_copy_kf16_quantize_v_no_fwht(
    device const float                    *src_k    [[buffer(0)]],
    device const float                    *src_v    [[buffer(1)]],
    device       half                     *cache_k  [[buffer(2)]],
    device       uint8_t                  *packed_v [[buffer(3)]],
    device       float                    *norms_v  [[buffer(4)]],
    constant HadamardQuantizeHbParams     &params   [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    constexpr ushort EPT = HEAD_DIM / 32;
    const uint head_idx = tgpig.x;
    const uint kv_sel   = tgpig.z; // 0 = K (F16 copy), 1 = V (no-FWHT quantize)
    const uint lane     = tiisg;

    if (head_idx >= params.num_kv_heads) return;

    // Common: compute write position (same convention as kv_cache_copy +
    // kv_quantize_v_no_fwht).
    uint actual_pos = (params.is_sliding != 0u)
        ? (params.write_pos % params.cache_capacity)
        : params.write_pos;

    if (kv_sel == 0u) {
        // ============================================================
        // K stream — F32 → F16 dense copy.
        // Mirrors `kernel_cpy_f32_f16` semantics; layout matches what the
        // hybrid SDPA kernel reads (`device const half *K_f16` at offset
        // [head_idx, actual_pos, 0..HEAD_DIM]).
        // ============================================================
        const uint src_base   = head_idx * HEAD_DIM + lane * EPT;
        const uint cache_base = head_idx * params.cache_capacity * HEAD_DIM
                              + actual_pos * HEAD_DIM
                              + lane * EPT;
        for (ushort i = 0; i < EPT; i++) {
            cache_k[cache_base + i] = (half) src_k[src_base + i];
        }
    } else {
        // ============================================================
        // V stream — Lloyd-Max codebook quantize without Hadamard rotation.
        // Byte-identical math to `kv_quantize_v_no_fwht_d{256,512}` above.
        // ============================================================
        const uint src_base = head_idx * HEAD_DIM + lane * EPT;
        float elems[EPT];
        for (ushort i = 0; i < EPT; i++) elems[i] = src_v[src_base + i];

        // Compute norm — see kv_quantize_v_no_fwht for derivation.
        float local_sq_sum = 0.0f;
        for (ushort i = 0; i < EPT; i++) local_sq_sum += elems[i] * elems[i];

        float norm0, norm1;
        if (HEAD_DIM == 256) {
            norm0 = sqrt(simd_sum(local_sq_sum));
            norm1 = 0.0f;
        } else {
            float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
            float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
            blk0_sq = simd_broadcast(blk0_sq, 0u);
            blk1_sq = simd_broadcast(blk1_sq, 16u);
            norm0 = sqrt(blk0_sq / 256.0f);
            norm1 = sqrt(blk1_sq / 256.0f);
        }

        // Scale to N(0,1) range for codebook lookup.
        if (HEAD_DIM == 256) {
            float inv_norm = (norm0 > 1.0e-10f) ? (1.0f / norm0) : 0.0f;
            float scale = inv_norm * sqrt(float(HEAD_DIM));
            for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
        } else {
            float blk_norm = (lane < 16u) ? norm0 : norm1;
            float inv_blk_norm = (blk_norm > 1.0e-10f) ? (1.0f / blk_norm) : 0.0f;
            float scale = inv_blk_norm * params.scale_factor_d512;
            for (ushort i = 0; i < EPT; i++) elems[i] *= scale;
        }

        // Quantize.
        const uint cbits = params.codebook_bits;
        uint8_t indices[EPT];
        for (ushort i = 0; i < EPT; i++) {
            float v = elems[i];
            uint8_t idx;
            if (cbits == 5u) {
                idx = (v > BOUNDARIES_5BIT[15]) ? 16 : 0;
                idx += (v > BOUNDARIES_5BIT[idx + 7]) ? 8 : 0;
                idx += (v > BOUNDARIES_5BIT[idx + 3]) ? 4 : 0;
                idx += (v > BOUNDARIES_5BIT[idx + 1]) ? 2 : 0;
                idx += (v > BOUNDARIES_5BIT[idx]) ? 1 : 0;
            } else if (cbits == 6u) {
                idx = (v > BOUNDARIES_6BIT[31]) ? 32 : 0;
                idx += (v > BOUNDARIES_6BIT[idx + 15]) ? 16 : 0;
                idx += (v > BOUNDARIES_6BIT[idx + 7]) ? 8 : 0;
                idx += (v > BOUNDARIES_6BIT[idx + 3]) ? 4 : 0;
                idx += (v > BOUNDARIES_6BIT[idx + 1]) ? 2 : 0;
                idx += (v > BOUNDARIES_6BIT[idx]) ? 1 : 0;
            } else {
                idx = (v > BOUNDARIES_8BIT[127]) ? 128 : 0;
                idx += (v > BOUNDARIES_8BIT[idx + 63]) ? 64 : 0;
                idx += (v > BOUNDARIES_8BIT[idx + 31]) ? 32 : 0;
                idx += (v > BOUNDARIES_8BIT[idx + 15]) ? 16 : 0;
                idx += (v > BOUNDARIES_8BIT[idx + 7])  ?  8 : 0;
                idx += (v > BOUNDARIES_8BIT[idx + 3])  ?  4 : 0;
                idx += (v > BOUNDARIES_8BIT[idx + 1])  ?  2 : 0;
                idx += (v > BOUNDARIES_8BIT[idx])      ?  1 : 0;
            }
            indices[i] = idx;
        }

        // Write packed bytes.
        const uint packed_base = head_idx * params.cache_capacity * HEAD_DIM
                               + actual_pos * HEAD_DIM;
        const uint elem_base = packed_base + lane * EPT;
        for (ushort i = 0; i < EPT; i++) {
            packed_v[elem_base + i] = indices[i];
        }

        // Store norm.
        if (HEAD_DIM == 256) {
            if (lane == 0) {
                norms_v[head_idx * params.cache_capacity + actual_pos] = norm0;
            }
        } else {
            if (lane == 0u) {
                uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
                norms_v[norm_base + 0u] = norm0;
            } else if (lane == 16u) {
                uint norm_base = head_idx * params.cache_capacity * 2u + actual_pos * 2u;
                norms_v[norm_base + 1u] = norm1;
            }
        }
    }
}

template [[host_name("kv_copy_kf16_quantize_v_no_fwht_d256")]]
kernel void kv_copy_kf16_quantize_v_no_fwht<256>(
    device const float *, device const float *,
    device half *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint3, uint);

template [[host_name("kv_copy_kf16_quantize_v_no_fwht_d512")]]
kernel void kv_copy_kf16_quantize_v_no_fwht<512>(
    device const float *, device const float *,
    device half *, device uint8_t *, device float *,
    constant HadamardQuantizeHbParams &, uint3, uint);
