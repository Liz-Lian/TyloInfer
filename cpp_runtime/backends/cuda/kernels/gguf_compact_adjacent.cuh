#pragma once
#include "gguf_compact_decode.cuh"
namespace mfq::cuda::gguf::compact_detail {
// Two four-value fragments share one circular stream window. Sixteen lanes
// decode a full 128-value block; the other half warp can process the next one.
// Only the access layout changes. The normative FP16 codebook is unchanged.
template<int Bits,int Steps>
__device__ __forceinline__ std::uint64_t eight_window(const std::uint8_t* q,int lane) {
    static_assert(Bits>0);
    if constexpr(Bits==8 && Steps==32) {
        const int first=2*lane;
        return std::uint64_t(q[(first+31)%32]) | (std::uint64_t(q[first])<<8) |
            (std::uint64_t(q[first+1])<<16);
    } else {
        constexpr int Bytes=Bits*Steps/8,States=Steps/16,WindowBytes=Steps==32?4:6;
        const int offset=(Steps-States-States*lane)*Bits,byte=offset/8;
        std::uint64_t window=0;
        #pragma unroll
        for(int j=0;j<WindowBytes;++j) {
            const int at=byte+j<Bytes?byte+j:byte+j-Bytes;
            window|=std::uint64_t(q[at])<<(j*8);
        }
        return window>>(offset%8);
    }
}
template<int Bits,int Steps,int Part,bool Shared>
__device__ __forceinline__ float4 eight_part(std::uint64_t window,const std::uint32_t* book) {
    std::uint32_t hash0,hash1;
    if constexpr(Steps==32) {
        std::uint32_t state;
        if constexpr(Bits==8) {
            const auto word=std::uint32_t(window>>(Part*8));
            state=((word&255u)<<8)|((word>>8)&255u);
        } else state=std::uint32_t(window>>((1-Part)*Bits));
        hash0=(state&32767u)*0x9e3779b1u;hash1=hash0;
    } else {
        hash0=(std::uint32_t(window>>((3-2*Part)*Bits))&32767u)*0x9e3779b1u;
        hash1=(std::uint32_t(window>>((2-2*Part)*Bits))&32767u)*0x9e3779b1u;
    }
    const auto lookup=[&](std::uint32_t index) {
        if constexpr(Shared) {
            union { std::uint32_t bits; __half2 value; } entry;
            entry.bits=book[index];return __half22float2(entry.value);
        } else return point(index);
    };
    const auto a=lookup(hash0>>21),b=lookup(Steps==32?(hash1>>10)&2047u:hash1>>21);
    return make_float4(a.x,a.y,b.x,b.y);
}

struct SixteenWindow {std::uint64_t first,second;};
template<int Bits,int Steps>
__device__ __forceinline__ SixteenWindow sixteen_window(const std::uint8_t* q,int lane) {
    if constexpr(Steps==64) {
        return {eight_window<Bits,Steps>(q,2*lane),eight_window<Bits,Steps>(q,2*lane+1)};
    } else if constexpr(Bits==8) {
        const int first=4*lane;
        return {std::uint64_t(q[(first+31)%32]) | (std::uint64_t(q[first])<<8) |
            (std::uint64_t(q[first+1])<<16) | (std::uint64_t(q[first+2])<<24) |
            (std::uint64_t(q[first+3])<<32),0};
    } else {
        constexpr int Bytes=Bits*Steps/8;
        const int offset=(Steps-4-4*lane)*Bits,byte=offset/8;
        std::uint64_t window=0;
        #pragma unroll
        for(int j=0;j<6;++j)window|=std::uint64_t(q[byte+j<Bytes?byte+j:byte+j-Bytes])<<(j*8);
        return {window>>(offset%8),0};
    }
}
template<int Bits,int Steps,int Part,bool Shared>
__device__ __forceinline__ float4 sixteen_part(SixteenWindow window,const std::uint32_t* book) {
    if constexpr(Steps==64) {
        return eight_part<Bits,Steps,Part%2,Shared>(Part<2?window.first:window.second,book);
    } else {
        const auto state=[&] {
            if constexpr(Bits==8)return ((std::uint32_t(window.first>>(Part*8))&255u)<<8) |
                (std::uint32_t(window.first>>((Part+1)*8))&255u);
            else return std::uint32_t(window.first>>((3-Part)*Bits));
        }();
        const auto hash=(state&32767u)*0x9e3779b1u;
        const auto lookup=[&](std::uint32_t index) {
            if constexpr(Shared) {
                union {std::uint32_t bits;__half2 value;} entry;
                entry.bits=book[index];return __half22float2(entry.value);
            } else return point(index);
        };
        const auto a=lookup(hash>>21),b=lookup((hash>>10)&2047u);
        return make_float4(a.x,a.y,b.x,b.y);
    }
}
}
