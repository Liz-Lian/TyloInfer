#pragma once
namespace mfq::cuda::dflash2 {
namespace {
__device__ __forceinline__ void store_planar_q8(float value,signed char* codes,float* scales,
        int row,int column,int padded) {
    float maximum=fabsf(value);
    for(int shift=16;shift;shift>>=1)
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,shift));
    const float inverse=maximum>0.f?127.f/maximum:0.f;
    codes[int64_t(row)*padded+column]=static_cast<signed char>(__float2int_rn(value*inverse));
    if(!(column&31))scales[int64_t(row)*(padded/32)+column/32]=maximum/127.f;
}
}
}
