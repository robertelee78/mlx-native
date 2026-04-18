// flash_attn_prefill_blk — pre-pass tile-skip classifier for the
// flash_attn_prefill family of kernels.
//
// Ported from llama.cpp's `kernel_flash_attn_ext_blk`
// (/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719).
//
// ## What it does
//
// Walks the additive attention mask in tile-sized chunks matching the main
// kernel's (BQ, BK) geometry and emits a single byte per (qtile, ktile) pair:
//
//   0 — "skip"          : the entire tile is masked to -inf.  Main kernel
//                         does `continue` for this KV tile (no K-load, no
//                         Q·K^T, no V-load, no mask-add, no softmax update).
//   1 — "mixed"         : tile has at least one finite mask value AND at
//                         least one attended cell.  Main kernel does the
//                         normal mask-load + mask-add path.
//   2 — "all_attended"  : every mask cell in the tile is exactly 0.0.  Main
//                         kernel computes Q·K^T and softmax normally but
//                         skips the mask-add (save one mask load per tile).
//
// ## Why not re-use llama.cpp's (8, 64) tile shape
//
// The blk byte is indexed by the main kernel as `blk[qt][kt]` where
// `(qt, kt)` is the main kernel's outer KV-tile loop position.  Our
// flash_attn_prefill main kernels use a DIFFERENT geometry than llama.cpp:
//
//   D=256  : BQ=32, BK=16   (candle-derived per-warp-Q-stacking template)
//   D=512  : BQ= 8, BK= 8   (llama.cpp-derived per-simdgroup-Q-distributed)
//
// The pre-pass MUST use the same (BQ, BK) as the main kernel it feeds,
// otherwise the blk index arithmetic is wrong.  See ADR-011 phase 2 §5.1
// for the full analysis.
//
// ## Sentinel convention (differs from llama.cpp)
//
// llama.cpp uses f16 masks with `-MAXHALF` as the "fully masked" sentinel
// (`ggml-metal.metal:5704`).  We use bf16 masks with a true `-INFINITY`
// encoding (`ADR-011-phase2-port-sentinel.md §2`, Wave 2A + Wave 2D).  The
// classification threshold is `mmax > bfloat(-1.0e30)` — conservative wide
// threshold that matches both true `-inf` and any "very negative" finite
// sentinel a future caller might pass.  The `mmin == 0 && mmax == 0` check
// is exact because the mask builder writes bit-exact `bfloat(0.0)` (bit
// pattern 0x0000) for every attended cell.
//
// ## Grid geometry
//
//   Threadgroups: (NK, NQ, 1)   — one threadgroup per (Q-tile, K-tile) pair.
//   Threads/TG  : (32, 1, 1)    — one simdgroup (matches llama.cpp:5666).
//
// Each lane in the simdgroup reads elements from the tile, performs a
// simdgroup-wide min/max reduction, and lane 0 writes the classification
// byte.
//
// For our D=256 geometry (BQ=32, BK=16) the tile is 32 rows × 16 cols = 512
// elements.  With NW=32 lanes each lane sees C=BK=16 elements of one row
// via the inner loop; only lanes [0, BK) participate on the col axis (half
// the simdgroup contributes identity values).  This is Option (b) in
// ADR §5.1 — acceptable for the initial port because the pre-pass is
// already cheap vs the main kernel.  Revisit only if profiling shows the
// pre-pass on the critical path.
//
// For our D=512 geometry (BQ=8, BK=8) the tile is 8 × 8 = 64 elements;
// lanes [0, 8) participate, lanes [8, 32) carry identity.  Same structure.
//
// ## Function constants
//
// | Index | Name   | Purpose |
// |-------|--------|---------|
// | 400   | BQ_blk | Q-rows per tile (matches main kernel's BQ).  32 or 8. |
// | 401   | BK_blk | K-cols per tile (matches main kernel's BK).  16 or 8. |
//
// A single pipeline is compiled per (BQ, BK) combination; the dispatcher
// decides which values to set based on which main kernel is being fed.
//
// SPDX-License-Identifier: MIT

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#else
// bf16 is the only mask dtype we support here — mirrors the Wave-2D mask
// builder.  On an Apple Silicon target without bfloat support the
// flash_attn_prefill family would not compile either, so this fallback is
// never exercised; the typedef exists only so the preprocessor doesn't
// error on the bfloat symbol.
typedef half bfloat16_t;
#endif

// Function constants — the dispatcher specialises these at pipeline
// creation time.  Kernels compiled with different (BQ, BK) share the same
// entry point name; the cache key in KernelRegistry disambiguates.
constant int BQ_blk [[function_constant(400)]];
constant int BK_blk [[function_constant(401)]];

// Sentinel guards for function-constant presence.  The dispatcher always
// sets these, but Metal's function-constant machinery requires us to
// handle the "undefined" case defensively to avoid spurious compile
// warnings during preview compiles.
constant int BQ_def = is_function_constant_defined(BQ_blk) ? BQ_blk : 32;
constant int BK_def = is_function_constant_defined(BK_blk) ? BK_blk : 16;

// Shader-side parameter block.  Mirrors the Rust `BlkParamsGpu` struct in
// flash_attn_prefill_blk.rs byte-for-byte.  Total 16 bytes (4 × i32).
struct FlashAttnPrefillBlkParams {
    int seq_len_q;    // qL — number of Q rows in the mask
    int seq_len_k;    // kL — number of K cols in the mask
    int mask_row_stride;  // stride between consecutive Q rows of the mask, in ELEMENTS (bf16 units)
    int _pad;         // explicit padding to keep bytemuck::Pod layout obvious
};

// Single-simdgroup tile classifier.
//
// One threadgroup per (qtile, ktile) pair; within the threadgroup, the
// lane 0 of the simdgroup writes the output byte after a 32-wide reduction.
//
// The tile is read directly from device memory — we do NOT stage the mask
// through threadgroup memory, matching llama.cpp's design choice
// (`ggml-metal.metal:5685-5699`).  This is what makes the pre-pass
// asymptotically cheaper than inline classification in the main kernel:
// no K/V loads, no mask staging to shared memory, no cross-simdgroup
// synchronisation.
kernel void flash_attn_prefill_blk_bf16(
    device const bfloat16_t* mask                   [[buffer(0)]],
    device       char*       blk_out                [[buffer(1)]],
    constant FlashAttnPrefillBlkParams& params      [[buffer(2)]],
    uint3  tgpig                                    [[threadgroup_position_in_grid]],
    ushort tiisg                                    [[thread_index_in_simdgroup]]
) {
    const int BQ = BQ_def;          // Q-rows per tile
    const int BK = BK_def;          // K-cols per tile
    const int NW = 32;              // simd width (Apple GPU)

    const int qt = int(tgpig.y);    // Q-tile index
    const int kt = int(tgpig.x);    // K-tile index

    const int qL = params.seq_len_q;
    const int kL = params.seq_len_k;
    const int M_stride = params.mask_row_stride;  // elements between mask rows

    // Mirror llama.cpp ggml-metal.metal:5683 — partial trailing K-tiles
    // (tile straddles the kL right edge) default to `mixed` (1).  Classifying
    // a partial tile cleanly would require per-element bound checks inside
    // the main kernel's loop, which is exactly what the main kernel already
    // does for the last KV tile.  Giving it `1` lets the normal path handle
    // the remainder correctly.
    //
    // Ordering: we ALSO check whether the tile is entirely past kL (can
    // happen when kL < kt*BK); such a tile gets byte=0 so the main kernel
    // skips it.  Technically the main kernel never dispatches a qtile
    // beyond kL_aligned+1, but we emit a defensive 0 to avoid coupling
    // the two kernels' bound-check logic.
    const int tile_k_start = kt * BK;
    const int tile_k_end   = tile_k_start + BK;

    char res;
    if (tile_k_start >= kL) {
        // Entire tile past the mask's last valid K column — fully masked.
        // Main kernel's outer loop clamps to kL so it would never issue a
        // tile like this; emit 0 for completeness.
        res = 0;
    } else if (tile_k_end > kL) {
        // Partial right-edge tile — cannot be cleanly classified as skip
        // (edge rows contain valid mask values) or all-zero (trailing pad
        // bytes are undefined).  Fallback to mixed; the main kernel's
        // kL_rem branch handles per-element bound checks.
        res = 1;
    } else {
        // Fully-inside tile — classify by simdgroup reduction.
        res = 0;

        // Per-lane pointer to the tile start: mask row 0 of the Q-tile,
        // column (tile_k_start + tiisg).  When tiisg >= BK, the pointer is
        // out-of-tile — we gate reads on `tiisg < BK` so those lanes
        // contribute identity values (bfloat16_t has no sentinel better
        // than what we start mmin/mmax at, so skipping the read is the
        // correct behaviour).
        device const bfloat16_t* mask_src =
            mask + (qt * BQ) * M_stride + tile_k_start + tiisg;

        // Use f32 for reduction to avoid bf16 comparison subtleties — bf16
        // min/max are well-defined on Apple Silicon but f32 reductions are
        // universally safe and identical in semantics for finite values.
        // IEEE-754 min/max both propagate -inf correctly: min(x, -inf) = -inf,
        // max(x, -inf) = x for any finite x.
        float mmin =  INFINITY;
        float mmax = -INFINITY;

        // Compute the number of rows this Q-tile actually contains.  If the
        // Q-tile straddles the bottom edge (qt * BQ + BQ > qL) we only read
        // the rows that exist — reading past qL would fetch undefined bytes
        // (the mask buffer is exactly qL * kL * sizeof(bfloat) bytes).
        //
        // This is an internal defensive check — the main kernel's Q-tile
        // launcher already clamps dispatched qtiles to `ceil(qL/BQ)`, so in
        // practice `qt * BQ < qL` always holds.  But the pre-pass is
        // dispatched with the same ceil, so the LAST Q-tile can still have
        // `qt*BQ + BQ > qL`.  Without the clamp we'd read garbage for rows
        // >= qL, which would spuriously set mmin/mmax and flip the
        // classification of the last Q-tile.
        int q_rows = BQ;
        if (qt * BQ + BQ > qL) {
            q_rows = qL - qt * BQ;
            if (q_rows < 0) {
                q_rows = 0;
            }
        }

        // Walk the tile's rows.  Only lanes with tiisg < BK do useful work;
        // other lanes carry identity (mmin=+inf, mmax=-inf) through the
        // reduction which is the simd_min/simd_max identity.
        if (tiisg < BK) {
            for (int j = 0; j < q_rows; ++j) {
                float v = float(mask_src[j * M_stride]);
                mmin = min(mmin, v);
                mmax = max(mmax, v);
            }
        }

        // Simdgroup-wide reductions — fold the per-lane min/max into a
        // tile-wide value.  Metal `simd_min` / `simd_max` follow IEEE-754
        // min/max semantics: for finite operands they behave like std::min/
        // std::max; for -inf they return -inf (min) / finite (max).  The
        // max value of a tile where every cell is -inf is therefore -inf,
        // which we detect below via `!isfinite(mmax) && mmax < 0`.
        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        // Three-way classification.  See the llama.cpp reference at
        // ggml-metal.metal:5704-5710 for the equivalent f16 logic.
        //
        // Fully-masked: our mask builder writes bit-exact `-INFINITY` for
        // blocked cells.  `simd_max` of a tile of all -inf returns -inf.
        // The defensive `<= -1e30f` threshold also catches finite "very
        // negative" sentinels (e.g., if a future caller switches to
        // -FLT_MAX/2), without requiring exact-representation checks.
        //
        // All-attended: the mask builder writes bit-exact `bfloat(0.0)`
        // (bit pattern 0x0000) for attended cells.  A tile of all zeros
        // reduces to mmin=mmax=0.  Exact equality is safe — no rounding
        // paths produce subnormal zeros here.
        if (mmax <= -1.0e30f) {
            res = 0;  // fully masked
        } else if (mmin == 0.0f && mmax == 0.0f) {
            res = 2;  // all attended (mask is a no-op for this tile)
        } else {
            res = 1;  // mixed
        }
    }

    // Write the classification byte.  dst layout is [NQ, NK] row-major;
    // only lane 0 writes so we don't race 32 lanes against one byte.
    const int NK = (kL + BK - 1) / BK;
    if (tiisg == 0) {
        blk_out[qt * NK + kt] = res;
    }
}
