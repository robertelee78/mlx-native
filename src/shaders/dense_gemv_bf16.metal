// dense_gemv_bf16.metal — Dense bf16 × f32 → f32 GEMV (matrix-vector multiply).
//
// Implements the `kernel_mul_mv_t_t_4` template-instantiation shape for
// `bfloat, bfloat4, float, float4` (i.e. `kernel_mul_mv_bf16_f32_4`).
//
// The scalar-row entry point serves single-token decode and small projections;
// the tiled entry point reuses each weight load across four adjacent rows.
// Larger row widths use the GEMM tensor-core kernel.
//
// Layout contract:
//   src0  [src0_batch, N, K]  bfloat, row-major  (weight matrix, transposed convention)
//   src1  [src1_batch, M, K]  float,  row-major  (input vectors)
//   dst   [src1_batch, M, N]  float,  row-major  (output vectors)
//
// Reduction strategy:
//   - Grid: (ceil(N/NR0), M, src1_batch).
//   - Each threadgroup handles NR0=2 output elements (weight rows) for one
//     input row.
//   - Each threadgroup has 32 × NSG threads (one simdgroup per "lane block").
//     NSG = min(4, (K + 127) / 128) — an empirically chosen split.
//   - Each simdgroup computes a partial dot product of its K-slice and reduces
//     via simd_sum, then stores to threadgroup memory for the final cross-group
//     reduction.

#include <metal_stdlib>
using namespace metal;

// ---- Host-facing params struct ---------------------------------------------
//
// Field layout is identical to `ggml_metal_kargs_mul_mv` in ggml-metal-impl.h
// (bytes 0-111).  Unused fields (nb00, ne10) are present for layout
// compatibility but ignored by this kernel.
struct DenseGemvBf16Params {
    int32_t  ne00;   // K  — contract dim
    int32_t  ne01;   // N  — number of weight rows (output dim)
    int32_t  ne02;   // src0_batch
    uint64_t nb00;   // src0 element stride (bytes) — unused (assumed 2)
    uint64_t nb01;   // src0 row stride (bytes)     = K * sizeof(bfloat)
    uint64_t nb02;   // src0 batch stride (bytes)   = N * K * sizeof(bfloat)
    uint64_t nb03;   // src0 super-batch stride      — unused
    int32_t  ne10;   // ne10 — unused (= K)
    int32_t  ne11;   // M  — number of input rows
    int32_t  ne12;   // src1_batch
    uint64_t nb10;   // src1 element stride (bytes) — unused (assumed 4)
    uint64_t nb11;   // src1 row stride (bytes)     = K * sizeof(float)
    uint64_t nb12;   // src1 batch stride (bytes)   = M * K * sizeof(float)
    uint64_t nb13;   // src1 super-batch stride      — unused
    int32_t  ne0;    // N (output cols, = ne01)
    int32_t  ne1;    // M (output rows, = ne11)
    int32_t  nr0;    // NR0 — weight rows per threadgroup (always 2)
    int16_t  r2;     // src1_batch / src0_batch (GQA broadcast factor)
    int16_t  r3;     // super-batch broadcast — unused (always 1)
};

// ---- Kernel ----------------------------------------------------------------
//
// Template parameters:
//   NR0  = weight rows per threadgroup (2, the reference default).
//   NSG  = at most four simdgroups per threadgroup. The scalar-row host path
//          may launch fewer groups for a small K; unpopulated partial slots
//          remain zero and are harmless in the final reduction.

kernel void hf2q_dense_gemv_bf16_f32_4(
        constant DenseGemvBf16Params & args,
        device const char * src0,           // bfloat  [ne02, N, K]
        device const char * src1,           // float   [ne12, M, K]
        device       char * dst,            // float   [ne12, M, N]
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    // NSG = number of simdgroups per threadgroup (hard-coded 4).
    // NW  = simdgroup width on Apple GPU = 32.
    constexpr short NSG = 4;
    constexpr short NW  = 32;
    // NB  = elements per "inner block" (32 bfloat4 = 128 scalars).
    // NF  = bfloat4 vector elements loaded per thread-iteration (4×4=16).
    constexpr short NB  = 32;   // inner blocks of NF=16 are strided by NW=32
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4; // 4 — bfloat4 vectors per thread-iteration

    const int nb = args.ne00 / NB;  // number of full NB-wide inner blocks in K

    // Threadgroup position:
    //   tgpig.x = output-row tile index  (covers NR0 = 2 output rows)
    //   tgpig.y = input-row  index       (one threadgroup per M row)
    //   tgpig.z = batch index
    const int r0 = (int)tgpig.x * 2;  // NR0 = 2
    const int r1 = (int)tgpig.y;
    const int im = (int)tgpig.z;

    const uint i12 = (uint)im % (uint)args.ne12;
    const uint i13 = (uint)im / (uint)args.ne12;

    // Input vector (src1) pointer for this input row and batch.
    const uint64_t offset1 = (uint64_t)r1   * args.nb11
                           + (uint64_t)i12  * args.nb12
                           + (uint64_t)i13  * args.nb13;

    device const float4 * y4 = (device const float4 *)(src1 + offset1);

    // Weight row pointers for the 2 output rows this threadgroup handles.
    device const bfloat4 * ax4[2];
    for (short row = 0; row < 2; ++row) {
        const int output_row = min(r0 + (int)row, args.ne01 - 1);
        const uint64_t offset0 = (uint64_t)output_row * args.nb01
                               + (uint64_t)(i12 / (uint)args.r2) * args.nb02
                               + (uint64_t)(i13 / (uint)args.r3) * args.nb03;
        ax4[row] = (device const bfloat4 *)((device const char *)src0 + offset0);
    }

    // Partial dot products per row.
    float sumf[2] = { 0.f, 0.f };

    // Each simdgroup handles a contiguous slice of inner blocks.
    // ix = which thread within the simdgroup's NF-wide sub-slice (0..NW/NF-1 = 0..1).
    // il = which NF4-aligned bfloat4 sub-slice (0..NW/NF-1 = 0..1).
    const short ix = tiisg / (NW / NF);   // 0..1
    const short il = tiisg % (NW / NF);   // 0..1

    // Starting inner block for this simdgroup + thread.
    const int ib0 = (int)sgitg * NF + ix;

    // bfloat4 vector cache for the current y slice.
    float4 yl4[NF4];

    // Pointer to the starting position in y4 for this thread.
    device const float4 * yb4 = y4 + (ib0 * NB + il * NF) / 4;

    // Main loop: stride by NSG*NF inner blocks across K.
    for (int ib = ib0; ib < nb; ib += NSG * NF) {
        // Load NF4 float4 vectors from the input.
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        // Accumulate dot product for each weight row.
        for (short row = 0; row < 2; ++row) {
            device const bfloat4 * xb4 = ax4[row] + (ib * NB + il * NF) / 4;
            float sumq = 0.f;
            for (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf[row] += sumq;
        }

        yb4 += NSG * NF * NW / 4;
    }

    // Tail loop for any remaining scalars past the last full NB block.
    // Use scalar float/bfloat access.
    device const float  * y_scalar  = (device const float  *)(src1 + offset1);
    for (int i = nb * NB + (int)sgitg * NW + (int)tiisg; i < args.ne00; i += NW * NSG) {
        for (short row = 0; row < 2; ++row) {
            device const bfloat * ax_scalar = (device const bfloat *)ax4[row];
            sumf[row] += (float)ax_scalar[i] * y_scalar[i];
        }
    }

    // ---- Threadgroup reduction (standard mv reduce-and-write pattern) ----
    //
    // Layout of threadgroup memory: [NR0][NW] floats = [2][32] floats = 256 bytes.
    threadgroup float * shmem_f32 = (threadgroup float *)shmem;

    // Phase 1: simd_sum within each simdgroup, store to shmem[row][sgitg].
    for (short row = 0; row < 2; ++row) {
        if (sgitg == 0) {
            shmem_f32[row * NW + tiisg] = 0.f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < 2; ++row) {
        if (tiisg == 0) {
            shmem_f32[row * NW + sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: simd_sum across the NSG partial sums stored in shmem.
    // Output pointer for this batch + input-row.
    device float * dst_f32 = (device float *)dst
        + (uint64_t)im * (uint64_t)args.ne0 * (uint64_t)args.ne1
        + (uint64_t)r1 * (uint64_t)args.ne0;

    for (short row = 0; row < 2 && r0 + row < args.ne01; ++row) {
        float tot = simd_sum(shmem_f32[row * NW + tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

// Width-four BF16 GEMV. Each threadgroup loads two weight rows once and
// accumulates four independent input rows with the scalar GEMV reduction
// order. This removes the row-wise kernel's fourfold weight reread without
// expanding four live rows into the tensor kernel's 32-row tile.
[[host_name("hf2q_dense_gemv_bf16_f32_r1_4")]]
kernel void hf2q_dense_gemv_bf16_f32_r1_4(
        constant DenseGemvBf16Params & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr short M_TILE = 4;
    constexpr short NR0 = 2;
    constexpr short NSG = 4;
    constexpr short NW = 32;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF / 4;

    const int r0 = (int)tgpig.x * NR0;
    const int r1 = (int)tgpig.y * M_TILE;
    const int im = (int)tgpig.z;
    const uint i12 = (uint)im % (uint)args.ne12;
    const uint i13 = (uint)im / (uint)args.ne12;
    const int nb = args.ne00 / NB;

    device const bfloat4 * ax4[NR0];
    for (short out = 0; out < NR0; ++out) {
        const int output_row = min(r0 + (int)out, args.ne01 - 1);
        const uint64_t offset0 = (uint64_t)output_row * args.nb01
            + (uint64_t)(i12 / (uint)args.r2) * args.nb02
            + (uint64_t)(i13 / (uint)args.r3) * args.nb03;
        ax4[out] = (device const bfloat4 *)(src0 + offset0);
    }

    device const float4 * yb4[M_TILE];
    device const float * y_scalar[M_TILE];
    const short ix = tiisg / (NW / NF);
    const short il = tiisg % (NW / NF);
    const int ib0 = (int)sgitg * NF + ix;
    for (short row = 0; row < M_TILE; ++row) {
        const int input_row = min(r1 + (int)row, args.ne11 - 1);
        const uint64_t offset1 = (uint64_t)input_row * args.nb11
            + (uint64_t)i12 * args.nb12
            + (uint64_t)i13 * args.nb13;
        y_scalar[row] = (device const float *)(src1 + offset1);
        yb4[row] = (device const float4 *)(src1 + offset1)
            + (ib0 * NB + il * NF) / 4;
    }

    float sums[NR0][M_TILE] = {
        { 0.f, 0.f, 0.f, 0.f },
        { 0.f, 0.f, 0.f, 0.f },
    };

    for (int ib = ib0; ib < nb; ib += NSG * NF) {
        device const bfloat4 * xb4[NR0];
        for (short out = 0; out < NR0; ++out) {
            xb4[out] = ax4[out] + (ib * NB + il * NF) / 4;
        }
        float4 weight_vectors[NR0][NF4];
        for (short i = 0; i < NF4; ++i) {
            for (short out = 0; out < NR0; ++out) {
                weight_vectors[out][i] = float4(xb4[out][i]);
            }
        }
        for (short out = 0; out < NR0; ++out) {
            for (short row = 0; row < M_TILE; ++row) {
                float block_sum = 0.f;
                for (short i = 0; i < NF4; ++i) {
                    block_sum += dot(weight_vectors[out][i], yb4[row][i]);
                }
                sums[out][row] += block_sum;
            }
        }
        for (short row = 0; row < M_TILE; ++row) {
            yb4[row] += NSG * NF * NW / 4;
        }
    }

    for (int i = nb * NB + (int)sgitg * NW + (int)tiisg;
         i < args.ne00;
         i += NW * NSG) {
        float weights[NR0];
        for (short out = 0; out < NR0; ++out) {
            weights[out] = ((device const bfloat *)ax4[out])[i];
        }
        for (short row = 0; row < M_TILE; ++row) {
            const float input = y_scalar[row][i];
            for (short out = 0; out < NR0; ++out) {
                sums[out][row] += weights[out] * input;
            }
        }
    }

    threadgroup float * partials = (threadgroup float *)shmem;
    for (short out = 0; out < NR0; ++out) {
        for (short row = 0; row < M_TILE; ++row) {
            if (sgitg == 0) {
                partials[(out * M_TILE + row) * NW + tiisg] = 0.f;
            }
            sums[out][row] = simd_sum(sums[out][row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        for (short out = 0; out < NR0; ++out) {
            for (short row = 0; row < M_TILE; ++row) {
                partials[(out * M_TILE + row) * NW + sgitg] = sums[out][row];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst_f32 = (device float *)dst
        + (uint64_t)im * (uint64_t)args.ne0 * (uint64_t)args.ne1;
    for (short out = 0; out < NR0 && r0 + out < args.ne01; ++out) {
        for (short row = 0; row < M_TILE && r1 + row < args.ne11; ++row) {
            const float total = simd_sum(partials[(out * M_TILE + row) * NW + tiisg]);
            if (tiisg == 0 && sgitg == 0) {
                dst_f32[(uint64_t)(r1 + row) * args.ne0 + (r0 + out)] = total;
            }
        }
    }
}
