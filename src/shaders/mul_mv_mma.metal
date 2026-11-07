// Few-row MMA mat-vec (`mul_mv_mma`) for GGUF block weights at decode
// widths 1..=8: a port of the reference implementation's
// `kernels/mul_mv_mma.metal` (generic and 32-weight block kinds).
//
// One threadgroup owns 8·NT weight rows × 8·RT input columns. Each weight is
// read (and, for the generic kind, dequantized to f32) once per threadgroup
// and contracted against all 8·RT columns with 8×8 simdgroup matrices, so
// the weight stream is paid once regardless of the batch width. The NSG
// simdgroups split K round-robin; their partial tiles are summed in a fixed
// simdgroup order through threadgroup memory.
//
// Kinds ported here (the ones the reference uses for these types at ≤ 8
// columns, i.e. RT = 1):
//   * GEN  (`kernel_mul_mv_mma_gen`) — Q4_K, Q6_K, Q5_0: 64-weight chunks,
//     f32 A from the 16-weight dequantizers, f32 B, f32 accumulate.
//   * BLK  (`kernel_mul_mv_mma_blk`) — Q8_0: 32-weight blocks, exact-integer
//     half A (the 1024 + q trick), f32 B, per-block `fma(d, Σq·x, acc)`.
//
// Function constants (declared `int` because mlx-native's registry
// specializes bool + i32; the reference declares nsg as `short`, used here
// through a `short` local — identical values):
//   FC_mul_mv_mma_nsg  (i32 @ 2300) — simdgroups per threadgroup (K split).
//   FC_mul_mv_mma_ne00 (i32 @ 2304) — compile-time K of the BLK kind
//                                      (0 for GEN, which reads args.ne00).
//
// Missing columns: the src1 row pointer is clamped to `ne11 - 1`, so padding
// columns of a short batch duplicate the last real column, and the store
// masks rows ≥ ne01 and columns ≥ ne11. Nothing is zero-filled.
//
// The reference's fused residual ADD (src2 + FC add) and int16 batch
// function constants are not ported: callers here have no residual and a
// single weight matrix; the batch offsets come from the args struct.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#define QK5_0 32
#define QK8_0 32
#define QK_K  256
#define K_SCALE_SIZE 12
#define QK_NL 16

// ------- block typedefs (byte-for-byte GGUF layout) -------

typedef struct {
    half    d;
    uint8_t qh[4];
    uint8_t qs[QK5_0 / 2];
} block_q5_0;

typedef struct {
    half   d;
    int8_t qs[QK8_0];
} block_q8_0;

typedef struct {
    half    d;
    half    dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half    d;
} block_q6_K;

// Must match `MulMvExtGpuArgs` in src/ops/mul_mv_ext.rs (same layout as the
// reference's `mul_mv_ext` kernel args, which its MMA kernels also take).
struct hf2q_mul_mv_mma_args {
    int      ne00;
    int      ne01;
    int      ne02;
    ulong    nb00;
    ulong    nb01;
    ulong    nb02;
    ulong    nb03;
    int      ne10;
    int      ne11;
    int      ne12;
    ulong    nb10;
    ulong    nb11;
    ulong    nb12;
    ulong    nb13;
    int      ne0;
    int      ne1;
    short    r2;
    short    r3;
};

constant int FC_mul_mv_mma_nsg  [[function_constant(2300)]];
constant int FC_mul_mv_mma_ne00 [[function_constant(2304)]];

// ------- 16-weight dequantizers (reference kernels/dequantize.h, float4x4) -------

// External linkage: a kernel specialization whose template argument has
// internal linkage is itself internal and is left out of the library.
inline void mma_dequantize_q5_0(device const block_q5_0 * xb, short il, thread float4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = reg_f;
}

static inline uchar2 mma_get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

inline void mma_dequantize_q4_K(device const block_q4_K * xb, short il, thread float4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = mma_get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

inline void mma_dequantize_q6_K(device const block_q6_K * xb, short il, thread float4x4 & reg) {
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
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// ------- common tile helpers -------

// a lane of a few-row MMA tile: A fragment row fm, B fragment columns fn and fn + 1
struct mul_mv_mma_tile {
    short    fm;
    short    fn;
    int      i01;
    int      i11;
    int      i1m;
    uint64_t offset0;
    uint64_t offset1;
};

// the A fragment row and the first B fragment column that lane l holds in an 8x8 simdgroup matrix
static inline short mul_mv_mma_lane_fm(ushort l) { return ((l/4) & 4) + ((l/2) % 4); }
static inline short mul_mv_mma_lane_fn(ushort l) { return ((l/4) & 2)*2 + (l % 2)*2; }

template<short NT, short RT>
static inline mul_mv_mma_tile mul_mv_mma_tile_init(constant hf2q_mul_mv_mma_args & args, uint3 tgpig, ushort tiisg) {
    mul_mv_mma_tile tile;
    tile.fm  = mul_mv_mma_lane_fm(tiisg);
    tile.fn  = mul_mv_mma_lane_fn(tiisg);
    tile.i01 = tgpig.x*(8*NT);
    tile.i11 = tgpig.y*(8*RT);
    tile.i1m = tgpig.z;

    const int i12 = tile.i1m%args.ne12;
    const int i13 = tile.i1m/args.ne12;

    tile.offset0 = (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    tile.offset1 = i12*args.nb12 + i13*args.nb13;

    return tile;
}

// the src0 row of A fragment row fm in 8-row tile t, clamped to the last row
static inline device const char * mul_mv_mma_src0_row(
        thread const mul_mv_mma_tile & tile, constant hf2q_mul_mv_mma_args & args, device const char * src0, short t) {
    const int r = min(tile.i01 + 8*t + tile.fm, args.ne01 - 1);
    return src0 + tile.offset0 + (uint64_t) r*args.nb01;
}

// the src1 row of B fragment column fn + e in 8-row tile rt, clamped to the last row
static inline device const float * mul_mv_mma_src1_row(
        thread const mul_mv_mma_tile & tile, constant hf2q_mul_mv_mma_args & args, device const char * src1, short rt, short e) {
    const int r = min(tile.i11 + 8*rt + tile.fn + e, args.ne11 - 1);
    return (device const float *) (src1 + tile.offset1 + (uint64_t) r*args.nb11);
}

constexpr constant static ushort mma_f16_1024_bits = 0x6400;
constexpr constant static half   mma_f16_1024      = 1024.0h;

// the halves 1024 + q for integers q < 1024: exact normal values, unlike the subnormal q*2^-24 that Metal may flush to zero
static inline half2 mul_mv_mma_1024_plus(ushort2 q) {
    return as_type<half2>(q | mma_f16_1024_bits);
}

// adds up the K slices of the NSG simdgroups for an 8*NT x 8*RT output tile and writes it
template<short NT, short RT>
static inline void mul_mv_mma_store(
        thread float (&acc)[RT][NT][2],
        constant hf2q_mul_mv_mma_args & args,
        device char * dst,
        threadgroup char * shmem,
        thread const mul_mv_mma_tile & tile, ushort tiisg, ushort sgitg) {
    const short NSG = (short) FC_mul_mv_mma_nsg;

    threadgroup float * red = (threadgroup float *) shmem;

    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 0] = acc[rt][t][0];
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 1] = acc[rt][t][1];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst_f32 = (device float *) dst + (uint64_t) tile.i1m*args.ne0*args.ne1;

    for (short idx = sgitg*32 + tiisg; idx < RT*NT*64; idx += NSG*32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg*(RT*NT*64) + idx];
        }

        const short rt = idx/(NT*64);
        const short t  = (idx/64) % NT;
        const short l  = (idx % 64)/2;
        const short e  = idx % 2;

        const int r0 = tile.i01 + 8*t  + mul_mv_mma_lane_fm(l);
        const int r1 = tile.i11 + 8*rt + mul_mv_mma_lane_fn(l) + e;

        if (r0 < args.ne01 && r1 < args.ne11) {
            const uint64_t i = (uint64_t) r1*args.ne0 + r0;
            dst_f32[i] = sum;
        }
    }
}

// ------- BLK kind: 32-weight block types -------

// the per-type parts of kernel_mul_mv_mma_blk for 32-weight blocks. a src1 block splits into halves b0 and b1,
// and MMA step s uses half b1 when b1_step(s) and the .y value of a pair when y_step(s)
// q8_0: A lane (m, j) holds qs bytes 4*j .. 4*j + 7 (j even); flipping the sign bit of a quant byte gives the unsigned q + 128.
// B lane k = fm holds src1 values b, b + 1 (b0) and b + 4, b + 5 (b1) of a block, b = 8*(k/2) + 2*(k%2)
struct mul_mv_mma_q8_0 {
    typedef block_q8_0 block;
    typedef ushort4    quants;

    // weights per block, and the float2 offset of b1 in a src1 block
    enum { qk = QK8_0, b1 = 2 };

    static short a_off(short fn) { return 1 + 2*fn; }
    static short b_off(short fm) { return 8*(fm/2) + 2*(fm%2); }

    static quants load(device const ushort * qs) { return ushort4(qs[0], qs[1], qs[2], qs[3]); }
    static quants prep(quants q) {
        constexpr ushort sign_bits = 0x8080;
        return q ^ ushort4(sign_bits);
    }
    static float2 prep_b1(float2 v) { return v; }

    static bool b1_step(short s) { return s >= 2; }
    static bool y_step (short s) { return s % 2 != 0; }

    static half2 frag(quants q, short s) {
        constexpr ushort byte_mask = 0x00FF;
        constexpr half   q_bias    = mma_f16_1024 + 128.0h;

        const ushort2 w  = s < 2 ? q.xy : q.zw;
        const ushort2 qq = s % 2 == 0 ? w : w >> 8;
        return mul_mv_mma_1024_plus(qq & byte_mask) - q_bias;
    }
};

template<typename Q, short NT>
static inline void load_mma_blk_a(device const ushort * const x[NT], int off, short fn, thread typename Q::quants * q, thread float * d) {
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        device const ushort * qs = x[t] + off;
        q[t] = Q::load(qs);
        d[t] = as_type<half>(*(qs - Q::a_off(fn)));
    }
}

template<typename Q, short RT>
static inline void load_mma_blk_b(device const float2 * const y[RT][2], int ib, thread float2 (*b0)[2], thread float2 (*b1)[2]) {
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            b0[rt][e] = y[rt][e][ib*(Q::qk/2)];
            b1[rt][e] = y[rt][e][ib*(Q::qk/2) + Q::b1];
        }
    }
}

// few-row mat-mat on 8x8 simdgroup matrices for 32-weight block types: a threadgroup reads each weight once
// for 8*NT src0 rows x 8*RT src1 rows, and its NSG simdgroups split K
template<short NT, short RT, typename Q>
kernel void kernel_mul_mv_mma_blk(
        constant hf2q_mul_mv_mma_args & args [[buffer(0)]],
        device const char * src0             [[buffer(1)]],
        device const char * src1             [[buffer(2)]],
        device       char * dst              [[buffer(3)]],
        threadgroup  char * shmem            [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = (short) FC_mul_mv_mma_nsg;

    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(args, tgpig, tiisg);

    device const ushort * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const ushort *) mul_mv_mma_src0_row(tile, args, src0, t) + Q::a_off(tile.fn);
    }

    device const float2 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float2 *) (mul_mv_mma_src1_row(tile, args, src1, rt, e) + Q::b_off(tile.fm));
        }
    }

    float acc[RT][NT][2] = {};

    const int nb = FC_mul_mv_mma_ne00/Q::qk;

    // a block is d, then the quants
    constexpr short us_blk = sizeof(typename Q::block)/2;

    typename Q::quants q[NT];
    float  d[NT];
    float2 b0[RT][2];
    float2 b1[RT][2];

    const int ib0 = min((int) sgitg, nb - 1);
    load_mma_blk_a<Q, NT>(x, ib0*us_blk, tile.fn, q, d);
    load_mma_blk_b<Q, RT>(y, ib0, b0, b1);

    for (int ib = sgitg; ib < nb; ib += NSG) {
        typename Q::quants qc[NT];
        float  dc[NT];
        float2 b0c[RT][2];
        float2 b1c[RT][2];

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            qc[t] = Q::prep(q[t]);
            dc[t] = d[t];
        }
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short e = 0; e < 2; ++e) {
                b0c[rt][e] = b0[rt][e];
                b1c[rt][e] = Q::prep_b1(b1[rt][e]);
            }
        }

        const int ibn = min(ib + NSG, nb - 1);
        load_mma_blk_a<Q, NT>(x, ibn*us_blk, tile.fn, q, d);
        load_mma_blk_b<Q, RT>(y, ibn, b0, b1);

        simdgroup_float8x8 mp[RT][NT];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short t = 0; t < NT; ++t) {
                mp[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
        }

        FOR_UNROLL (short s = 0; s < 4; ++s) {
            simdgroup_float8x8 mb[RT];
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                const float2 v0 = Q::b1_step(s) ? b1c[rt][0] : b0c[rt][0];
                const float2 v1 = Q::b1_step(s) ? b1c[rt][1] : b0c[rt][1];
                mb[rt].thread_elements()[0] = Q::y_step(s) ? v0.y : v0.x;
                mb[rt].thread_elements()[1] = Q::y_step(s) ? v1.y : v1.x;
            }

            FOR_UNROLL (short t = 0; t < NT; ++t) {
                const half2 h = Q::frag(qc[t], s);

                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = h.x;
                ma.thread_elements()[1] = h.y;

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mp[rt][t], ma, mb[rt], mp[rt][t]);
                }
            }
        }

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                acc[rt][t][0] = fma(dc[t], mp[rt][t].thread_elements()[0], acc[rt][t][0]);
                acc[rt][t][1] = fma(dc[t], mp[rt][t].thread_elements()[1], acc[rt][t][1]);
            }
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, dst, shmem, tile, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_mma_blk<4, 1, mul_mv_mma_q8_0>) mul_mv_mma_t;

template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<1, 1, mul_mv_mma_q8_0>;
template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<2, 1, mul_mv_mma_q8_0>;
template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<4, 1, mul_mv_mma_q8_0>;

// ------- GEN kind: any type with a 16-weight dequantizer -------

// a lane dequantizes 16 consecutive weights of a 64-weight chunk once for all src1 rows.
// MMA step s at MMA-k index j reads chunk weight 16*(j/2) + 8*(j%2) + s.
template<short NT, short RT, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_mul_mv_mma_gen(
        constant hf2q_mul_mv_mma_args & args [[buffer(0)]],
        device const char * src0             [[buffer(1)]],
        device const char * src1             [[buffer(2)]],
        device       char * dst              [[buffer(3)]],
        threadgroup  char * shmem            [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = (short) FC_mul_mv_mma_nsg;

    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(args, tgpig, tiisg);

    device const block_q * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const block_q *) mul_mv_mma_src0_row(tile, args, src0, t);
    }

    // B lane j = fm reads chunk values 16*(fm/2) + 8*(fm%2) .. +7
    device const float4 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float4 *) mul_mv_mma_src1_row(tile, args, src1, rt, e) + 4*(tile.fm/2) + 2*(tile.fm%2);
        }
    }

    simdgroup_float8x8 mc[RT][NT];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            mc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    const int nch = args.ne00/64;

    for (int g = sgitg; g < nch; g += NSG) {
        simdgroup_float8x8 mb[RT][8];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            const float4 a0 = y[rt][0][16*g + 0];
            const float4 a1 = y[rt][0][16*g + 1];
            const float4 b0 = y[rt][1][16*g + 0];
            const float4 b1 = y[rt][1][16*g + 1];
            FOR_UNROLL (short s = 0; s < 4; ++s) {
                mb[rt][s    ].thread_elements()[0] = a0[s];
                mb[rt][s    ].thread_elements()[1] = b0[s];
                mb[rt][s + 4].thread_elements()[0] = a1[s];
                mb[rt][s + 4].thread_elements()[1] = b1[s];
            }
        }

        const int ci = 4*g + tile.fn/2;

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            float4x4 w;
            dequantize_func(x[t] + ci/nl, ci%nl, w);

            FOR_UNROLL (short s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s/4    ][s%4];
                ma.thread_elements()[1] = w[s/4 + 2][s%4];

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mc[rt][t], ma, mb[rt][s], mc[rt][t]);
                }
            }
        }
    }

    float acc[RT][NT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            acc[rt][t][0] = mc[rt][t].thread_elements()[0];
            acc[rt][t][1] = mc[rt][t].thread_elements()[1];
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, dst, shmem, tile, tiisg, sgitg);
}

// Explicit instantiations: `host_name` does not take concatenated string
// literals (a macro-built name leaves the function out of the library).
template [[host_name("kernel_mul_mv_mma_q5_0_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 1, block_q5_0, 2, mma_dequantize_q5_0>;
template [[host_name("kernel_mul_mv_mma_q5_0_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 1, block_q5_0, 2, mma_dequantize_q5_0>;
template [[host_name("kernel_mul_mv_mma_q5_0_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 1, block_q5_0, 2, mma_dequantize_q5_0>;
template [[host_name("kernel_mul_mv_mma_q4_K_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 1, block_q4_K, QK_NL, mma_dequantize_q4_K>;
template [[host_name("kernel_mul_mv_mma_q4_K_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 1, block_q4_K, QK_NL, mma_dequantize_q4_K>;
template [[host_name("kernel_mul_mv_mma_q4_K_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 1, block_q4_K, QK_NL, mma_dequantize_q4_K>;
template [[host_name("kernel_mul_mv_mma_q6_K_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 1, block_q6_K, QK_NL, mma_dequantize_q6_K>;
template [[host_name("kernel_mul_mv_mma_q6_K_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 1, block_q6_K, QK_NL, mma_dequantize_q6_K>;
template [[host_name("kernel_mul_mv_mma_q6_K_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 1, block_q6_K, QK_NL, mma_dequantize_q6_K>;
