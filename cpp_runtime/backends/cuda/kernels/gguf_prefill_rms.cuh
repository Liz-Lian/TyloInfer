#pragma once

// Preserve ordinary RMSNorm's reduction order and its FP32 result for parallel
// floating consumers. The same row also produces the scaled GEMM operand.
template<class Input,int Threads>
__global__ void rms_prefill_kernel(const Input* input,const float* weight,float* normalized,
        half* packed,int32_t* exponents,int width,float eps,float offset) {
    const int64_t row=blockIdx.x,base=row*width;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    extern __shared__ float values[];
    __shared__ float maxima[Threads/32];
    __shared__ int exponent;
    float squared=0;
    for(int i=threadIdx.x;i<width;i+=Threads) {
        const float x=float(input[base+i]);squared+=x*x;
    }
    squared=block_sum<Threads/32>(squared);
    const float inverse=rsqrtf(squared/float(width)+eps);
    float maximum=0;
    for(int i=threadIdx.x;i<width;i+=Threads) {
        const float y=float(input[base+i])*inverse*(weight[i]+offset);
        normalized[base+i]=y;values[i]=y;maximum=fmaxf(maximum,fabsf(y));
    }
    maximum=warp_max(maximum);
    if(!lane)maxima[warp]=maximum;
    __syncthreads();
    if(!warp) {
        maximum=lane<Threads/32?maxima[lane]:0;
        maximum=warp_max(maximum);
        if(!lane){exponent=prefill_unit_exponent(maximum,width);exponents[row]=exponent;}
    }
    __syncthreads();
    for(int i=threadIdx.x;i<width;i+=Threads)
        packed[base+i]=__float2half_rn(prefill_ldexp(values[i],-exponent));
}
