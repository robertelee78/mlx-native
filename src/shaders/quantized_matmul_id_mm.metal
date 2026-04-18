// quantized_matmul_id_mm.metal — MoE-routed GGML-quantized matrix-matrix kernels.
//
// Ports llama.cpp's `kernel_mul_mm_id_<qtype>_f32`
// (ggml/src/ggml-metal/ggml-metal.metal:9650) and its preprocessing helper
// `kernel_mul_mm_id_map0_ne20_<N>` (same file:9584) to mlx-native.
//
// Why mm_id?  The MoE variant of quantized_matmul needs to run the same
// expert weight tile against many tokens that routed to that expert.  The
// existing `kernel_mul_mv_id_<qtype>_f32` re-reads each expert's weight
// blocks once per routed (token, slot) pair.  The mm_id variant stages a
// 64x32 tile of the expert's weights into threadgroup shared memory once
// and reuses it across a 32-row block of the expert's routed tokens —
// the same win as the dense mm kernel, but per-expert.
//
// Two-stage dispatch (matches llama.cpp):
//
//   1. `kernel_mul_mm_id_map0_ne20_8` — scans the (token, slot) -> expert
//       id table, for each expert builds a contiguous list of the packed
//       `(token_idx * ne20 + slot_idx)` values that routed to it, plus a
//       per-expert count.  Scratch buffers: `hids` `[n_experts, n_tokens]`
//       (int32, row-major), `htpe` `[n_experts]` (uint32).
//
//   2. `kernel_mul_mm_id_<qtype>_f32` — dispatched with one Z-per-expert.
//       Each threadgroup owns one (N-tile, M-tile, expert_id) and short-
//       circuits when the expert's routed-token count is below its M-tile
//       start.  Inside a tile, ALL 32 M-rows belong to the same expert,
//       so the 64x32 weight tile loaded into shmem is valid for every row.
//
// The preprocessing step is the key to making mm profitable at MoE:
// without it, 32 consecutive output rows in a tile could route to 32
// different experts, defeating weight reuse.
//
// Port rules identical to the dense mm port (see quantized_matmul_mm.metal
// for detail).  Both kernels are bit-compatible with the llama.cpp source;
// output tolerance-level matches with the existing mv_id kernel.
//
// Portions derived from llama.cpp (MIT).  Copyright the llama.cpp Authors.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define QK4_0 32
#define QK8_0 32
#define QK_K  256

#define QK_NL 16

// ---- Params for map0 ----
//
// Maps llama.cpp's `ggml_metal_kargs_mul_mm_id_map0`
// (ggml-metal-impl.h:483).  We omit `ne02` (== n_experts, used for a
// threadgroup-size assert upstream — we pass it host-side and dispatch
// exactly `n_experts` threads per threadgroup to mirror the upstream
// `ntg = n_experts` launch).

struct GgmlMatmulIdMm_Map0Params {
    int32_t  ne10;    // unused, kept for struct symmetry
    int32_t  ne11;    // n_expert_used (bcast), always == ne20 in our case
    uint64_t nb11;    // unused
    uint64_t nb12;    // unused
    int32_t  ne21;    // n_tokens
    int32_t  ne20;    // n_expert_used (== top_k)
    uint64_t nb21;    // bytes per token in the source ids table
                      //  (= ne20 * sizeof(int32_t) for our layout)
};

// ---- Params for mul_mm_id ----
//
// Maps llama.cpp's `ggml_metal_kargs_mul_mm_id`
// (ggml-metal-impl.h:494).

struct GgmlMatmulIdMm_MmParams {
    int32_t  ne00;   // K
    int32_t  ne02;   // n_experts
    uint64_t nb01;   // bytes per weight row (within one expert)
    uint64_t nb02;   // bytes per expert weight slab (== nb01 * ne01)
    uint64_t nb03;   // unused
    int32_t  ne11;   // n_expert_used (bcast, == ne20)
    uint64_t nb10;   // = sizeof(float) = 4
    uint64_t nb11;   // bytes per input row = K * sizeof(float)
    uint64_t nb12;   // bytes per input batch (n_tokens * nb11)
    uint64_t nb13;   // unused
    int32_t  ne20;   // n_expert_used (== top_k)
    int32_t  ne21;   // n_tokens
    int32_t  ne0;    // N (per-expert output rows)
    int32_t  ne1;    // batch (always == ne21 * ne20 in our layout)
    int16_t  r2;     // 1
    int16_t  r3;     // 1
    int16_t  _pad0;
    int16_t  _pad1;
};

// ---- block_q struct layouts (byte-for-byte GGUF) ----

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    half   d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;

// ---- Dequantize helpers (identical to quantized_matmul_mm.metal) ----

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;

    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (qs[i + 16*il] * d);
    }
    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K * xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;

    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg_f[i][0] = dl0 *  ((half)(q & 0xFF))      - ml;
        reg_f[i][1] = dl1 * ((float)(q & 0xFF00))    - ml;
        reg_f[i][2] = dl2 * ((float)(q & 0xFF0000))  - ml;
        reg_f[i][3] = dl3 * ((float)(q & 0xFF000000))- ml;
    }
    reg = (type4x4) reg_f;
}

// ====================================================================
// map0 — per-expert routed-token list builder
// ====================================================================
//
// Input:  src2 — per-token expert ids `[n_tokens, n_expert_used]` int32.
// Output: htpe — per-expert routed count  `[n_experts]` uint32.
// Output: hids — per-expert routed-token list `[n_experts, n_tokens]` int32.
//         Each slot holds `(token_idx * ne20 + slot_idx)` packed.
//
// Dispatch geometry (matches llama.cpp:9584):
//   threadgroups = (1, 1, 1)
//   threads_per_threadgroup = (n_experts, 1, 1)     (ntg == ne02)
//
// One thread per expert id (ide = tpitg).  Threads scan the token axis
// in blocks of `ntg` tokens, stage each token's `ne20` expert-ids into
// threadgroup shmem, then each thread scans that block looking for its
// own expert-id and appends entries into its slot in `hids`.

template<short ne20>
kernel void hf2q_mul_mm_id_map0_impl(
        constant GgmlMatmulIdMm_Map0Params & args [[buffer(0)]],
        device const char * src2 [[buffer(1)]],
        device       char * htpe [[buffer(2)]],
        device       char * hids [[buffer(3)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg;    // expert id owned by this thread

    uint32_t n_all = 0;
    device int32_t * ids_i32 = (device int32_t *) hids + ide * args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 =
                (device const int32_t *) (src2 + (i21 + tpitg) * args.nb21);
            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg * ne20;
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) break;
            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t * ne20;

            short sel = 0;
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide) * (i20 + 1);
            }
            ids_i32[n_all] = (i21 + t) * ne20 + sel - 1;
            n_all += sel > 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

// Gemma 4 uses top_k = 8.  We only instantiate that value today; other
// values can be added when needed.
template [[host_name("kernel_mul_mm_id_map0_ne20_8")]]
kernel void hf2q_mul_mm_id_map0_impl<8>(
    constant GgmlMatmulIdMm_Map0Params &,
    device const char *, device char *, device char *,
    threadgroup char *, ushort, ushort);

// ====================================================================
// mul_mm_id — MoE matmul using the map0-produced expert token lists
// ====================================================================
//
// Dispatch geometry (matches llama.cpp:2389):
//   threadgroups = (ceil(neh1/32), ceil(N/64), n_experts)
//   threads_per_threadgroup = (128, 1, 1)
//
// where `neh1 = htpe[expert]` (count of tokens routed to this expert).
//
// Each threadgroup owns:
//   * im       = tgpig.z     — expert index
//   * r0-block = tgpig.y     — output-N tile index
//   * r1-block = tgpig.x     — input-M (per-expert) tile index
//
// The kernel short-circuits inactive (expert, r1) combinations via the
// `r1 >= neh1` check — some threadgroups may exit early but the grid
// launch is uniform.
//
// Per-iteration of the K loop: stage a 64x32 A-tile (from the expert's
// weight slab) + 32x32 B-tile (from input rows selected via hids) into
// threadgroup memory, then run 8 simdgroup MMA ops per simdgroup.  Four
// simdgroups cooperate on one 64x32 output tile.
//
// Write-back staging ALWAYS goes through shmem because consecutive M-rows
// in this tile belong to the same expert but may correspond to different
// (token, slot) output rows (determined by hids[im * ne21 + r1 + j]).

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void hf2q_mul_mm_id_impl(
        constant GgmlMatmulIdMm_MmParams & args [[buffer(0)]],
        device const char * src0 [[buffer(1)]],
        device const char * src1 [[buffer(2)]],
        device const char * htpe [[buffer(3)]],
        device const char * hids [[buffer(4)]],
        device       char * dst  [[buffer(5)]],
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shmem);
    threadgroup float * sb = (threadgroup float *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;  // 2
    constexpr int NL1 = NK/8;   // 4

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0;
    const int r1 = tgpig.x * NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    // Early exit: this expert has fewer routed tokens than our tile's
    // M-base.  Whole threadgroup returns.
    if (r1 >= neh1) return;

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    // (token_idx, slot_idx) of the row this thread owns.
    const int id = ids_i32[im * args.ne21 + r1 + lr1];
    const short i11 = (id % args.ne20) % args.ne11;   // slot index
    const short i12 = (id / args.ne20);               // token index
    const short i13 = 0;

    // Base of expert `im`'s weight slab.
    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x =
        (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ---- A tile dequantize + stage ----
        {
            half4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        // ---- B tile stage (per-element, bounds-checked for K tail) ----
        for (short i = 0; i < 8; ++i) {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short lx = i;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(sb + 64*ib + 8*ly + lx) =
                (loop_k + iy + i < args.ne00) ? *((device float *) y + i) : 0.f;
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA accumulate ----
        threadgroup const half  * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const float * lsmb = (sb + 2*64*(sgitg/2));

        for (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // ---- Write-back (always through shmem staging: see header note) ----
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem)
        + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup `sgitg` strides through the tile in steps of 4 rows.
    // For each owned row j (within the tile), look up its (token, slot)
    // via hids, compute the destination address in dst, and copy.
    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;  // slot index (output row-major)
        const short idt = id / args.ne20;  // token index (output batch)

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

// Template instantiations for the three quant types our GGUF uses.

template [[host_name("kernel_mul_mm_id_q4_0_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q4_0, 2, dequantize_q4_0>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q8_0_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q8_0, 2, dequantize_q8_0>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);

template [[host_name("kernel_mul_mm_id_q6_K_f32")]]
kernel void hf2q_mul_mm_id_impl<block_q6_K, QK_NL, dequantize_q6_K>(
    constant GgmlMatmulIdMm_MmParams &,
    device const char *, device const char *, device const char *, device const char *,
    device char *, threadgroup char *, uint3, ushort, ushort, ushort);
