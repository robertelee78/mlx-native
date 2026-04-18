// flash_attn_prefill_mask — GPU fill kernel for the bf16 additive attention
// mask consumed by the flash_attn_prefill family of kernels.
//
// Ported from: llama.cpp's llm_graph_input_attn_no_cache::set_input mask-fill
// algorithm at /opt/llama.cpp/src/llama-graph.cpp:380-444.
//
// llama.cpp fills the mask CPU-side then relies on implicit upload.  We fill
// it GPU-side because (a) Apple Silicon has unified memory so there is no
// meaningful "upload", (b) GPU fill parallelises trivially over (qL, kL) and
// stays on-device to match the rest of the mlx-native dispatcher, and
// (c) we avoid the cache-invalidation overhead of a large host→device transfer
// per prefill when we build both the global and sliding masks.  The mask
// values written by this kernel are byte-identical to llama.cpp's post-cast
// bf16 mask (see ADR-011 phase 2 §6.1).
//
// Reference: the canonical in-kernel attended predicate (simplified for the
// batch=1, single-sequence, no-ALiBi, causal_attn=true case, per ADR-011
// phase 2 §1.5) is:
//
//   attended iff (k_pos <= q_abs)              // causal
//            AND (!swa_standard ||
//                 q_abs - k_pos < n_swa)        // SWA window
//
// Otherwise the cell is written as bfloat16_t(-INFINITY), matching
// llama-graph.cpp:421,436 and the flash_attn_prefill.metal mask-sentinel
// contract (masked = bf16 -inf = bit pattern 0xFF80; attended = +0.0 =
// bit pattern 0x0000).
//
// Grid geometry:
//   Threadgroups: (ceil(qL / 32), qL_rows, 1) — one threadgroup per row,
//   32 threads each sweeping the K dimension in strides of 32.
//
//   Concretely the host dispatches threadgroups=(1, qL, 1) with tgsize=
//   (min(kL, 256), 1, 1) so each threadgroup fills one row and each
//   thread writes `ceil(kL / tgsize.x)` cells.  This mirrors softmax.metal's
//   one-threadgroup-per-row layout (see ops/softmax.rs:93-106).
//
// Params layout (inline bytes at buffer(1)):
//   struct FlashAttnPrefillMaskParams {
//     uint  seq_len_k;     // K dimension (mask stride between rows)
//     uint  q_abs_offset;  // absolute offset of the first query row (ql_off)
//     int   n_swa;         // sliding window size; -1 means "no window"
//     uint  causal;        // 1 = apply causal (k_pos > q_abs → masked)
//   };
//
// The `n_swa < 0` convention (rather than a separate boolean) is a host
// convenience: a global mask is built with n_swa=-1 and SWA is skipped.
// `causal` is a uint (not bool) so the param struct layout is trivial to
// serialize from Rust via bytemuck::Pod.

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#else
// If bfloat is unavailable the flash_attn_prefill kernel family would not
// compile either, so this path is never exercised on the Apple Silicon
// targets we support.  Declared here for compilation completeness.
typedef half bfloat16_t;
#endif

struct FlashAttnPrefillMaskParams {
    uint seq_len_k;
    uint q_abs_offset;
    int  n_swa;
    uint causal;
};

// Single row-per-threadgroup kernel.  One threadgroup per q row, 32 or more
// threads sweeping K in stride loops.
//
// Correctness is exact (0.0 vs -inf are discrete bf16 values); the kernel
// writes one of two well-defined bit patterns per cell.  Both halves of the
// mask (global and sliding) can be built with different (n_swa, causal)
// settings on separate dispatches — the kernel is stateless.
//
// align_K handling: the inner loop covers the full seq_len_k with a stride
// loop, so unaligned kL trailing remainders write correctly.  No special
// pad handling is required because the output is exactly the bf16 mask
// with no trailing pad cells.
kernel void flash_attn_prefill_mask_fill_bf16(
    device bfloat16_t* mask                                  [[buffer(0)]],
    constant FlashAttnPrefillMaskParams& params              [[buffer(1)]],
    uint q_row                                               [[threadgroup_position_in_grid]],
    uint tid                                                 [[thread_index_in_threadgroup]],
    uint tg_size                                             [[threads_per_threadgroup]]
) {
    const uint seq_len_k  = params.seq_len_k;
    const int  q_abs      = int(q_row + params.q_abs_offset);
    const int  n_swa      = params.n_swa;           // -1 = disabled
    const bool causal     = params.causal != 0u;
    const uint row_offset = q_row * seq_len_k;

    // Stride loop across the K dimension: every thread writes cells at
    // indices tid, tid + tg_size, tid + 2*tg_size, ...
    for (uint k_pos = tid; k_pos < seq_len_k; k_pos += tg_size) {
        const int kp = int(k_pos);

        // Mirror llama_hparams::is_masked_swa (llama-hparams.h:316-328) +
        // causal: attended iff (kp <= q_abs) for causal, AND (q_abs - kp <
        // n_swa) for SWA_STANDARD.
        //
        // The llama.cpp loops ("if future ... continue" and "if masked_swa
        // ... continue") map to our boolean OR of "is_masked" gates below.
        bool is_masked = false;
        if (causal && kp > q_abs) {
            is_masked = true;
        }
        if (n_swa > 0 && (q_abs - kp) >= n_swa) {
            is_masked = true;
        }

        // bf16(-INFINITY) has bit pattern 0xFF80 (sign=1, exp=0xFF, mant=0);
        // bf16(0.0) has bit pattern 0x0000.  Both are exact.  The constructor
        // from `float(-INFINITY)` selects the preserved-infinity path in
        // _MLX_BFloat16 / bfloat (the non-NaN branch of float_to_bfloat_bits):
        // input bits 0xFF800000, round-to-nearest-even rounds the mantissa
        // half-even (zero + zero → zero), shift >> 16 → 0xFF80.
        const bfloat16_t masked_val   = bfloat16_t(-INFINITY);
        const bfloat16_t attended_val = bfloat16_t(0.0);

        mask[row_offset + k_pos] = is_masked ? masked_val : attended_val;
    }
}
