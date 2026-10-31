#pragma once
// Adapted from Strata src/kernels/cuda/s26_tsum.cuh, Copyright (c) 2026
// Niko1221 and contributors. MIT: ../../../third_party/strata.LICENSE.
// Each butterfly level keeps half the outputs per lane. This preserves the
// original FP32 pairwise addition tree with fewer cross-lane instructions.
#include <cuda_runtime.h>

namespace mfq::cuda::warp_sums {
__host__ __device__ constexpr int capacity(int count) {
    return count<=1?1:count<=2?2:count<=4?4:count<=8?8:count<=16?16:32;
}
__host__ __device__ constexpr int log2(int value) {
    return value<=1?0:value==2?1:value==4?2:value==8?3:value==16?4:5;
}
template<int Offset,int Width>
__device__ __forceinline__ float plain(float value) {
    if constexpr(Offset>0)return plain<Offset/2,Width>(value+__shfl_xor_sync(0xffffffffu,value,Offset,Width));
    else return value;
}
template<int Count,int Width=32,int Offset=Width/2>
__device__ __forceinline__ float reduce(float* values,int lane) {
    static_assert(Count>0 && Count<=Width && (Count&(Count-1))==0);
    if constexpr(Count>1) {
        constexpr int half=Count/2;
        const bool high=(lane&Offset)!=0;
        #pragma unroll
        for(int i=0;i<half;++i) {
            const float send=high?values[i]:values[i+half];
            const float keep=high?values[i+half]:values[i];
            values[i]=keep+__shfl_xor_sync(0xffffffffu,send,Offset,Width);
        }
        return reduce<half,Width,Offset/2>(values,lane);
    } else return plain<Offset,Width>(values[0]);
}
template<int Count,int Width=32>
__device__ __forceinline__ int index(int lane) {
    constexpr int bits=log2(Count),top=log2(Width)-1;
    int result=0;
    #pragma unroll
    for(int i=0;i<bits;++i)result=(result<<1)|((lane>>(top-i))&1);
    return result;
}
template<int Count,int Width=32>
__device__ __forceinline__ int owner(int index) {
    constexpr int bits=log2(Count),top=log2(Width)-1;
    int lane=0;
    #pragma unroll
    for(int i=0;i<bits;++i)lane|=((index>>(bits-1-i))&1)<<(top-i);
    return lane;
}
}
