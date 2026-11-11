// Repacked Q4_K/Q6_K dense MM for fixed-width decode.
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

struct MmParams {
    int32_t ne00;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne12;
    uint32_t splits;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int16_t r2;
    int16_t r3;
};

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

template <bool Q6>
kernel void hf2q_repacked_mm_impl(
    constant MmParams &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const char *input [[buffer(2)]], device char *output [[buffer(3)]],
    constant uint2 &offsets [[buffer(4)]], device char *partials [[buffer(5)]],
    threadgroup char *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int COLS=32, ROWS=32, STEP=32;
    threadgroup half *tile=(threadgroup half *)shared+(tid/32)*3*COLS*STEP;
    threadgroup half *sx=tile;
    threadgroup half *sw0=sx+ROWS*STEP;
    threadgroup half *sw1=sw0+COLS*STEP;
    const uint sg=tid/32, lane=tid%32, output_origin=group.y*64+sg*COLS;
    const uint groups=a.ne00/32, units=groups/8, per=groups/a.splits;
    const int im=group.z, i12=im%a.ne12, i13=im/a.ne12;
    device float *src=(device float *)(input+a.nb13*i13+a.nb12*i12);
    auto xt=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sx,dextents<int32_t,2>{STEP,ROWS},array<int32_t,2>{1,STEP});
    auto wt0=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw0,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    auto wt1=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw1,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    matmul2d<matmul2d_descriptor(ROWS,COLS,STEP,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<1>> mm;
    auto sum=mm.get_destination_cooperative_tensor<decltype(xt),decltype(wt0),float>();

    const uint weight_row=output_origin+lane;
    const uint first=group.x*per, last=first+per;
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
        for(ushort row=0;row<ROWS;++row)
            sx[row*STEP+lane]=half(src[ulong(row)*(a.nb11/sizeof(float))+g*STEP+lane]);
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
    device float *base=a.splits==1?(device float *)output:
        (device float *)partials+group.x*a.ne1*a.ne0;
    device float *dst=base+output_origin+im*a.ne1*a.ne0;
    auto out=tensor<device float,dextents<int32_t,2>,tensor_inline>(
        dst,dextents<int32_t,2>{a.ne0,ROWS},array<int32_t,2>{1,a.ne0});
    sum.store(out);
}

template <bool Q6>
kernel void hf2q_repacked_mm_bf16_input_impl(
    constant MmParams &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device bfloat *input [[buffer(2)]], device char *output [[buffer(3)]],
    constant uint2 &offsets [[buffer(4)]], device char *partials [[buffer(5)]],
    threadgroup char *shared [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int COLS=32, ROWS=32, STEP=32;
    threadgroup half *tile=(threadgroup half *)shared+(tid/32)*2*COLS*STEP;
    threadgroup half *sw0=tile;
    threadgroup half *sw1=sw0+COLS*STEP;
    const uint sg=tid/32, lane=tid%32, output_origin=group.y*64+sg*COLS;
    const uint groups=a.ne00/32, units=groups/8, per=groups/a.splits;
    const int im=group.z, i12=im%a.ne12, i13=im/a.ne12;
    device bfloat *src=(device bfloat *)((device char *)input+a.nb13*i13+a.nb12*i12);
    auto xt=tensor<device bfloat,dextents<int32_t,2>,tensor_inline>(
        src,dextents<int32_t,2>{a.ne00,ROWS},array<int32_t,2>{1,a.ne00});
    auto wt0=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw0,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    auto wt1=tensor<threadgroup half,dextents<int32_t,2>,tensor_inline>(
        sw1,dextents<int32_t,2>{STEP,COLS},array<int32_t,2>{1,STEP});
    matmul2d<matmul2d_descriptor(ROWS,COLS,STEP,false,true,false,
        matmul2d_descriptor::mode::multiply_accumulate),execution_simdgroups<1>> mm;
    auto left0=xt.template slice<STEP,ROWS>(0,0);
    auto right0=wt0.template slice<STEP,COLS>(0,0);
    auto sum=mm.get_destination_cooperative_tensor<decltype(left0),decltype(right0),float>();

    const uint weight_row=output_origin+lane;
    const uint begin=group.x*per, end=begin+per;
    const uint first=begin+uint(ulong((group.y*40503u)&0xFFFFu)*per>>16);
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

        auto left=xt.template slice<STEP,ROWS>(g*STEP,0);
        if(buffer){
            auto right=wt1.template slice<STEP,COLS>(0,0); mm.run(left,right,sum);
        }else{
            auto right=wt0.template slice<STEP,COLS>(0,0); mm.run(left,right,sum);
        }
        if(walked+1<per){
            g=next; lo=next_lo; hi=next_hi; h=next_h; scales=next_scales; dh=next_dh;
        }
    }
    device float *base=a.splits==1?(device float *)output:
        (device float *)partials+group.x*a.ne1*a.ne0;
    device float *dst=base+output_origin+im*a.ne1*a.ne0;
    auto out=tensor<device float,dextents<int32_t,2>,tensor_inline>(
        dst,dextents<int32_t,2>{a.ne0,ROWS},array<int32_t,2>{1,a.ne0});
    sum.store(out);
}


template [[host_name("hf2q_repacked_mm_q4k_bf16_input_f32")]] kernel void hf2q_repacked_mm_bf16_input_impl<false>(
    constant MmParams &,device const uchar *,device bfloat *,device char *,constant uint2 &,device char *,threadgroup char *,uint3,ushort);
template [[host_name("hf2q_repacked_mm_q6k_bf16_input_f32")]] kernel void hf2q_repacked_mm_bf16_input_impl<true>(
    constant MmParams &,device const uchar *,device bfloat *,device char *,constant uint2 &,device char *,threadgroup char *,uint3,ushort);
template [[host_name("hf2q_repacked_mm_q4k_f32")]] kernel void hf2q_repacked_mm_impl<false>(
    constant MmParams &,device const uchar *,device const char *,device char *,constant uint2 &,device char *,threadgroup char *,uint3,ushort);
template [[host_name("hf2q_repacked_mm_q6k_f32")]] kernel void hf2q_repacked_mm_impl<true>(
    constant MmParams &,device const uchar *,device const char *,device char *,constant uint2 &,device char *,threadgroup char *,uint3,ushort);
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
