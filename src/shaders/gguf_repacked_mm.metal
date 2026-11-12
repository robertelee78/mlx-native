// Repacked Q4_K/Q6_K dense MM over 32-row activation tiles.
// Layout and dequantization are derived from Splash's Apache-2.0 MDGG0001
// representation (https://github.com/incoai/splash, commit 031a25a6).
#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint kTileRows = 64;
constant constexpr uint kFmtQ4K = 0;
constant constexpr uint kFmtQ6K = 1;

struct RepackParams {
    uint rows;
    uint input_size;
    uint format;
    uint plane1_offset;
    uint meta_offset;
    uint src_row_bytes;
};

inline ulong tiled_index(uint row, uint unit, uint units) {
    return (ulong(row / kTileRows) * units + unit) * kTileRows + row % kTileRows;
}
inline uint quant_slot(uint e) { return 8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3); }
inline uint pair_word(thread const uchar *slots, uint c) {
    uint word = 0;
    for (uint i = 0; i < 8; ++i) word |= uint(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1));
    return word;
}
inline uint bit_word(thread const uchar *slots, uint bits, uint word_index) {
    const uint per = 32 / bits;
    uint word = 0;
    for (uint i = 0; i < per; ++i) word |= uint(slots[word_index * per + i]) << (bits * i);
    return word;
}

kernel void hf2q_gguf_repack_q4k_q6k(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant RepackParams &p [[buffer(2)]],
    uint t [[thread_position_in_grid]]) {
    const uint groups = p.input_size / 32;
    if (t >= p.rows * groups) return;
    const uint row = t / groups, g = t % groups, block = g / 8, j = g % 8;
    device const uchar *b = src + ulong(row) * p.src_row_bytes + ulong(block) * (p.format == kFmtQ4K ? 144 : 210);
    device uchar *p0 = dst + tiled_index(row, g, groups) * 16;
    uchar lo[32], hi[32];
    if (p.format == kFmtQ4K) {
        for (uint e = 0; e < 32; ++e) lo[quant_slot(e)] = (b[16 + (j / 2) * 32 + e] >> (4 * (j % 2))) & 15;
        for (uint c = 0; c < 4; ++c) ((device uint *)p0)[c] = pair_word(lo, c);
        if (j == 0) {
            device uchar *meta = dst + p.meta_offset + tiled_index(row, block, groups / 8) * 16;
            for (uint i = 0; i < 16; ++i) meta[i] = b[i];
        }
    } else {
        const uint hb = j / 4, quarter = j % 4;
        for (uint e = 0; e < 32; ++e) {
            lo[quant_slot(e)] = (b[64 * hb + 32 * (quarter & 1) + e] >> (4 * (quarter >> 1))) & 15;
            hi[quant_slot(e)] = (b[128 + 32 * hb + e] >> (2 * quarter)) & 3;
        }
        for (uint c = 0; c < 4; ++c) ((device uint *)p0)[c] = pair_word(lo, c);
        device uchar *p1 = dst + p.plane1_offset + tiled_index(row, g, groups) * 8;
        ((device uint *)p1)[0] = bit_word(hi, 2, 0);
        ((device uint *)p1)[1] = bit_word(hi, 2, 1);
        if (j == 0) {
            device uchar *meta = dst + p.meta_offset + tiled_index(row, block, groups / 8) * 20;
            for (uint i = 0; i < 16; ++i) meta[i] = b[192 + i];
            meta[16] = b[208]; meta[17] = b[209]; meta[18] = 0; meta[19] = 0;
        }
    }
}

// `splits` fixes the summation order of every output element: K is cut into
// that many equal runs, each accumulated from zero, and the runs are added in
// order. It belongs to the prepared weight, never to the row count, so the
// bits of an output row do not depend on the batch it is computed in.
struct MmParams {
    uint k;
    uint n;
    uint m;
    uint splits;
};

// Activation rows are covered by 32-row tiles. An `Edge` kernel owns a partial
// tile and neither reads nor writes rows >= m; the others read 32 rows
// unchecked.
//
// `Serial` selects who adds the K runs, not what is added:
//   false: the bound buffers hold one row tile (m <= 32) and grid x enumerates
//          the runs; with more than one run the partial sums go to `partials`
//          for hf2q_repacked_mm_reduce_f32.
//   true:  the bound buffers hold all m rows and grid x enumerates the full
//          row tiles, the `Edge` kernel taking the last, partial one; a
//          threadgroup walks every run and adds each finished run into the
//          output, in the reduce kernel's order (fold_run).
template <bool Edge, bool Serial>
inline uint tile_first_row(constant MmParams &a, uint3 group) {
    return Serial ? 32u * (Edge ? a.m / 32u : group.x) : 0u;
}

// Add one finished run of a serial kernel into its output tile and clear the
// accumulator for the next run. `first` marks the first run: accumulated from
// +0.0 it is never -0.0, so storing it equals the reduce kernel's 0.0f + run.
//
// The running total lives in the output buffer. It cannot live in
// thread-private state: a pipeline built with Metal shader validation does
// not preserve that much of it across the next run's K loop (a second
// cooperative tensor and a plain thread array both came back with whole
// SIMD-group slots zeroed). Threadgroup memory of its own costs occupancy:
// 16 KB more per threadgroup ran 1.5 to 1.7 times slower at 720 rows. So the
// run is staged through `staging`, the SIMD group's two weight staging
// buffers, which hold 32x32 floats and are idle between runs. Thread `lane`
// then adds activation row `lane`, whose 32 output columns are contiguous.
template <typename Sum>
inline void fold_run(thread Sum &sum, threadgroup float *staging, device float *tile,
                     uint n, uint rows, uint lane, bool first) {
    auto staged=tensor<threadgroup float,dextents<int32_t,2>,tensor_inline>(
        staging,dextents<int32_t,2>{32,32},array<int32_t,2>{1,32});
    simdgroup_barrier(mem_flags::mem_threadgroup);
    sum.store(staged);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if(lane<rows){
        threadgroup packed_float4 *run=(threadgroup packed_float4 *)(staging+lane*32);
        device packed_float4 *total=(device packed_float4 *)(tile+ulong(lane)*n);
        for(ushort c=0;c<8;++c){
            if(first) total[c]=run[c]; else total[c]=float4(total[c])+float4(run[c]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for(ushort i=0;i<sum.get_capacity();++i){
        if(sum.is_valid_element(i)) sum[i]=0.0f;
    }
}

inline uint4 nibble_pairs(uint word) { return (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu; }
inline uint spread2(uint bits) { return (bits & 0x3333u) | ((bits & 0xCCCCu) << 14); }
inline uint scale_pair(packed_uint4 bytes, ushort j) {
    const uint word = j < 2 ? bytes.x : j < 4 ? bytes.y : j < 6 ? bytes.z : bytes.w;
    return word >> (16u * (j & 1));
}
inline float2 q4_coef(uint4 h, ushort j, bool minimum) {
    uint sc, mn;
    if (j < 4) {
        const uint sh = 8u * j;
        sc = (h.y >> sh) & 63u; mn = (h.z >> sh) & 63u;
    } else {
        const uint sh = 8u * (j - 4), w = h.w >> sh;
        sc = (w & 15u) | ((((h.y >> sh) >> 6) & 3u) << 4);
        mn = ((w >> 4) & 15u) | ((((h.z >> sh) >> 6) & 3u) << 4);
    }
    const half d = as_type<half>(ushort(h.x)), dm = as_type<half>(ushort(h.x >> 16));
    return float2(float(d) * float(sc), minimum ? -float(dm) * float(mn) : 0.0f);
}
inline half2 linear_pair(uint pair, float s, float m, ushort zero) {
    const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + zero)));
    return zero ? half2(code * s) : half2(fma(code, float2(s), float2(m)));
}

inline void decode_q4k_half(uint4 payload, uint4 h, ushort j, ushort half_index, threadgroup half *out) {
    const float2 k = q4_coef(h, j, true);
    for (ushort c = 0; c < 4; ++c) {
        const uint4 p = nibble_pairs(payload[c]);
        const uint2 selected = half_index ? p.zw : p.xy;
        *((threadgroup half2 *)(out + 4*c)) = linear_pair(selected.x, k.x, k.y, 0);
        *((threadgroup half2 *)(out + 4*c + 2)) = linear_pair(selected.y, k.x, k.y, 0);
    }
}
inline void decode_q6k_half(uint4 lo, uint2 hb, packed_uint4 scales, uint dh, ushort j, ushort half_index, threadgroup half *out) {
    const float dscale = float(as_type<half>(ushort(dh))); const uint sp = scale_pair(scales, j);
    const float2 s = float2(dscale * float(as_type<char>(uchar(sp))), dscale * float(as_type<char>(uchar(sp >> 8))));
    for (ushort c = 0; c < 4; ++c) {
        const uint high = spread2(hb[c >> 1] >> (16 * (c & 1)));
        const uint4 p = nibble_pairs(lo[c]) | (((uint4(high) >> uint4(0,4,8,12)) & 0x00030003u) << 4);
        const uint2 selected = half_index ? p.zw : p.xy; const float scale = half_index ? s.y : s.x;
        *((threadgroup half2 *)(out + 4*c)) = linear_pair(selected.x, scale, 0, 32);
        *((threadgroup half2 *)(out + 4*c + 2)) = linear_pair(selected.y, scale, 0, 32);
    }
}

template <bool Q6, bool Edge, bool Serial>
kernel void hf2q_repacked_mm_impl(
    constant MmParams &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint2 &offsets [[buffer(4)]], device float *partials [[buffer(5)]],
    threadgroup char *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int COLS=32, ROWS=32, STEP=32;
    threadgroup half *tile=(threadgroup half *)shared+(tid/32)*3*COLS*STEP;
    threadgroup half *sx=tile;
    threadgroup half *sw0=sx+ROWS*STEP;
    threadgroup half *sw1=sw0+COLS*STEP;
    const uint sg=tid/32, lane=tid%32, output_origin=group.y*64+sg*COLS;
    const uint groups=a.k/32, units=groups/8, per=groups/a.splits;
    const uint row0=tile_first_row<Edge,Serial>(a,group);
    const uint rows=Edge?a.m-row0:uint(ROWS);
    device const float *src=input+ulong(row0)*a.k;
    auto xt=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sx,dextents<int32_t,2>{STEP,ROWS},array<int32_t,2>{1,STEP});
    auto wt0=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw0,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    auto wt1=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw1,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    matmul2d<matmul2d_descriptor(ROWS,COLS,STEP,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<1>> mm;
    auto sum=mm.get_destination_cooperative_tensor<decltype(xt),decltype(wt0),float>();
    const bool fold=Serial&&a.splits>1;

    const uint weight_row=output_origin+lane;
    const uint split0=Serial?0u:group.x, split1=Serial?a.splits:group.x+1;
    for(uint split=split0;split<split1;++split){
        const uint first=split*per, last=first+per;
        ulong ix=tiled_index(weight_row,first,groups);
        uint4 lo=*((device const uint4 *)(w+ix*16));
        uint2 hi=uint2(0);
        uint4 h=uint4(0);
        packed_uint4 scales=packed_uint4(0);
        uint dh=0;
        device const uchar *meta=w+offsets.y+tiled_index(weight_row,first/8,units)*(Q6?20:16);
        if constexpr(Q6){
            hi=*((device const uint2 *)(w+offsets.x+ix*8));
            scales=*((device const packed_uint4 *)meta);
            dh=*((device const uint *)(meta+16));
        }else{
            h=*((device const uint4 *)meta);
        }

        for(uint g=first;g<last;++g){
            if constexpr(Edge){
                for(ushort row=0;row<rows;++row)
                    sx[row*STEP+lane]=half(src[ulong(row)*a.k+g*STEP+lane]);
                for(ushort row=rows;row<ROWS;++row) sx[row*STEP+lane]=half(0);
            }else{
                for(ushort row=0;row<ROWS;++row)
                    sx[row*STEP+lane]=half(src[ulong(row)*a.k+g*STEP+lane]);
            }
            const uint buffer=(g-first)&1;
            threadgroup half *sw=buffer?sw1:sw0;
            if constexpr(Q6){
                decode_q6k_half(lo,hi,scales,dh,g%8,0,sw+lane*32);
                decode_q6k_half(lo,hi,scales,dh,g%8,1,sw+lane*32+16);
            }else{
                decode_q4k_half(lo,h,g%8,0,sw+lane*32);
                decode_q4k_half(lo,h,g%8,1,sw+lane*32+16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            uint4 next_lo=lo;
            uint2 next_hi=hi;
            uint4 next_h=h;
            packed_uint4 next_scales=scales;
            uint next_dh=dh;
            if(g+1<last){
                const uint next=g+1;
                const ulong next_ix=tiled_index(weight_row,next,groups);
                next_lo=*((device const uint4 *)(w+next_ix*16));
                if constexpr(Q6) next_hi=*((device const uint2 *)(w+offsets.x+next_ix*8));
                if(next/8!=g/8){
                    device const uchar *next_meta=w+offsets.y+tiled_index(weight_row,next/8,units)*(Q6?20:16);
                    if constexpr(Q6){
                        next_scales=*((device const packed_uint4 *)next_meta);
                        next_dh=*((device const uint *)(next_meta+16));
                    }else{
                        next_h=*((device const uint4 *)next_meta);
                    }
                }
            }

            auto left=xt.slice(0,0);
            if(buffer){
                auto right=wt1.slice(0,0); mm.run(left,right,sum);
            }else{
                auto right=wt0.slice(0,0); mm.run(left,right,sum);
            }
            if(g+1<last){
                lo=next_lo; hi=next_hi; h=next_h; scales=next_scales; dh=next_dh;
            }
        }
        if(fold) fold_run(sum,(threadgroup float *)sw0,
            output+ulong(row0)*a.n+output_origin,a.n,rows,lane,split==0);
    }
    if(!fold){
        device float *base=output;
        if constexpr(!Serial) if(a.splits>1) base=partials+ulong(group.x)*a.m*a.n;
        auto out=tensor<device float,dextents<int32_t,2>,tensor_inline>(
            base+ulong(row0)*a.n+output_origin,
            dextents<int32_t,2>{int(a.n),int(rows)},array<int32_t,2>{1,int(a.n)});
        sum.store(out);
    }
}

template <bool Q6, bool Edge, bool Serial>
kernel void hf2q_repacked_mm_bf16_input_impl(
    constant MmParams &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device bfloat *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint2 &offsets [[buffer(4)]], device float *partials [[buffer(5)]],
    threadgroup char *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int COLS=32, ROWS=32, STEP=32;
    threadgroup half *tile=(threadgroup half *)shared+(tid/32)*2*COLS*STEP;
    threadgroup half *sw0=tile;
    threadgroup half *sw1=sw0+COLS*STEP;
    const uint sg=tid/32, lane=tid%32, output_origin=group.y*64+sg*COLS;
    const uint groups=a.k/32, units=groups/8, per=groups/a.splits;
    const uint row0=tile_first_row<Edge,Serial>(a,group);
    const uint rows=Edge?a.m-row0:uint(ROWS);
    auto xt=tensor<device bfloat,dextents<int32_t,2>,tensor_inline>(
        input+ulong(row0)*a.k,
        dextents<int32_t,2>{int(a.k),int(rows)},array<int32_t,2>{1,int(a.k)});
    auto wt0=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw0,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    auto wt1=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw1,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    matmul2d<matmul2d_descriptor(ROWS,COLS,STEP,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<1>> mm;
    auto sum=mm.get_destination_cooperative_tensor<
        decltype(xt.template slice<STEP,ROWS>(0,0)),decltype(wt0.template slice<STEP,COLS>(0,0)),float>();
    const bool fold=Serial&&a.splits>1;

    const uint weight_row=output_origin+lane;
    const uint rotation=uint(ulong((group.y*40503u)&0xFFFFu)*per>>16);
    const uint split0=Serial?0u:group.x, split1=Serial?a.splits:group.x+1;
    for(uint split=split0;split<split1;++split){
        const uint begin=split*per, end=begin+per;
        const uint first=begin+rotation;
        ulong ix=tiled_index(weight_row,first,groups);
        uint4 lo=*((device const uint4 *)(w+ix*16));
        uint2 hi=uint2(0);
        uint4 h=uint4(0);
        packed_uint4 scales=packed_uint4(0);
        uint dh=0;
        device const uchar *meta=w+offsets.y+tiled_index(weight_row,first/8,units)*(Q6?20:16);
        if constexpr(Q6){
            hi=*((device const uint2 *)(w+offsets.x+ix*8));
            scales=*((device const packed_uint4 *)meta);
            dh=*((device const uint *)(meta+16));
        }else{
            h=*((device const uint4 *)meta);
        }

        uint g=first;
        for(uint walked=0;walked<per;++walked){
            const uint buffer=walked&1;
            threadgroup half *sw=buffer?sw1:sw0;
            if constexpr(Q6){
                decode_q6k_half(lo,hi,scales,dh,g%8,0,sw+lane*32);
                decode_q6k_half(lo,hi,scales,dh,g%8,1,sw+lane*32+16);
            }else{
                decode_q4k_half(lo,h,g%8,0,sw+lane*32);
                decode_q4k_half(lo,h,g%8,1,sw+lane*32+16);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            uint4 next_lo=lo;
            uint2 next_hi=hi;
            uint4 next_h=h;
            packed_uint4 next_scales=scales;
            uint next_dh=dh;
            const uint next=g+1<end?g+1:begin;
            if(walked+1<per){
                const ulong next_ix=tiled_index(weight_row,next,groups);
                next_lo=*((device const uint4 *)(w+next_ix*16));
                if constexpr(Q6) next_hi=*((device const uint2 *)(w+offsets.x+next_ix*8));
                if(next/8!=g/8){
                    device const uchar *next_meta=w+offsets.y+tiled_index(weight_row,next/8,units)*(Q6?20:16);
                    if constexpr(Q6){
                        next_scales=*((device const packed_uint4 *)next_meta);
                        next_dh=*((device const uint *)(next_meta+16));
                    }else{
                        next_h=*((device const uint4 *)next_meta);
                    }
                }
            }

            if constexpr(Edge){
                // A dynamic-extent slice keeps the tile's extents, and matmul2d
                // "will do edge checking ... against extents of original tensor"
                // (MPPTensorOpsMatMul2d.h, after its first example), so rows >= m
                // are not read. The static slices of the full tiles skip the check.
                auto left=xt.slice(int(g*STEP),0);
                if(buffer){
                    auto right=wt1.template slice<STEP,COLS>(0,0); mm.run(left,right,sum);
                }else{
                    auto right=wt0.template slice<STEP,COLS>(0,0); mm.run(left,right,sum);
                }
            }else{
                auto left=xt.template slice<STEP,ROWS>(g*STEP,0);
                if(buffer){
                    auto right=wt1.template slice<STEP,COLS>(0,0); mm.run(left,right,sum);
                }else{
                    auto right=wt0.template slice<STEP,COLS>(0,0); mm.run(left,right,sum);
                }
            }
            if(walked+1<per){
                g=next; lo=next_lo; hi=next_hi; h=next_h; scales=next_scales; dh=next_dh;
            }
        }
        if(fold) fold_run(sum,(threadgroup float *)sw0,
            output+ulong(row0)*a.n+output_origin,a.n,rows,lane,split==0);
    }
    if(!fold){
        device float *base=output;
        if constexpr(!Serial) if(a.splits>1) base=partials+ulong(group.x)*a.m*a.n;
        auto out=tensor<device float,dextents<int32_t,2>,tensor_inline>(
            base+ulong(row0)*a.n+output_origin,
            dextents<int32_t,2>{int(a.n),int(rows)},array<int32_t,2>{1,int(a.n)});
        sum.store(out);
    }
}

#define HF2Q_REPACKED_MM(NAME, IMPL, INPUT, Q6, EDGE, SERIAL) \
    template [[host_name(NAME)]] kernel void IMPL<Q6, EDGE, SERIAL>( \
        constant MmParams &, device const uchar *, INPUT, device float *, constant uint2 &, \
        device float *, threadgroup char *, uint3, ushort);
#define HF2Q_REPACKED_MM_VARIANTS(NAME, IMPL, INPUT, Q6) \
    HF2Q_REPACKED_MM(NAME, IMPL, INPUT, Q6, false, false) \
    HF2Q_REPACKED_MM(NAME "_serial", IMPL, INPUT, Q6, false, true) \
    HF2Q_REPACKED_MM(NAME "_edge", IMPL, INPUT, Q6, true, false) \
    HF2Q_REPACKED_MM(NAME "_edge_serial", IMPL, INPUT, Q6, true, true)
HF2Q_REPACKED_MM_VARIANTS("hf2q_repacked_mm_q4k_bf16_input_f32", hf2q_repacked_mm_bf16_input_impl, device bfloat *, false)
HF2Q_REPACKED_MM_VARIANTS("hf2q_repacked_mm_q6k_bf16_input_f32", hf2q_repacked_mm_bf16_input_impl, device bfloat *, true)
HF2Q_REPACKED_MM_VARIANTS("hf2q_repacked_mm_q4k_f32", hf2q_repacked_mm_impl, device const float *, false)
HF2Q_REPACKED_MM_VARIANTS("hf2q_repacked_mm_q6k_f32", hf2q_repacked_mm_impl, device const float *, true)
kernel void hf2q_repacked_mm_reduce_f32(
    device const float *partials [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=dims.x) return;
    float sum=0.0f;
    for(uint split=0;split<dims.y;++split) sum+=partials[ulong(split)*dims.x+i];
    output[i]=sum;
}
