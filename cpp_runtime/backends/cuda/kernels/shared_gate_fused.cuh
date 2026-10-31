#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mfq::cuda::shared_gate_detail {
// Scalar dot/sigmoid fusion follows Strata's shared_expert.cu design.
// Input and dot are rounded to BF16 before the runtime's FP32 sigmoid.
template<int Threads,bool Pairs,bool Double,bool HalfGate=false>
__global__ void project_sigmoid(const __half* input,const __nv_bfloat16* weight,
        __nv_bfloat16* dot,std::conditional_t<HalfGate,__half,__nv_bfloat16>* gate,int64_t width) {
    using Acc=std::conditional_t<Double,double,float>;
    __shared__ Acc partial[Threads/32];
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const auto* x=input+static_cast<std::size_t>(blockIdx.x)*width;
    Acc sum=0;
    if constexpr(Pairs) {
        for(int64_t i=threadIdx.x*2;i<width;i+=Threads*2) {
            const float a=__bfloat162float(__float2bfloat16_rn(__half2float(x[i])));
            if constexpr(Double)sum+=Acc(a)*Acc(__bfloat162float(weight[i]));
            else sum=__fmaf_rn(__bfloat162float(weight[i]),a,sum);
            if(i+1<width) {
                const float b=__bfloat162float(__float2bfloat16_rn(__half2float(x[i+1])));
                if constexpr(Double)sum+=Acc(b)*Acc(__bfloat162float(weight[i+1]));
                else sum=__fmaf_rn(__bfloat162float(weight[i+1]),b,sum);
            }
        }
    } else {
        for(int64_t i=threadIdx.x;i<width;i+=Threads) {
            const float a=__bfloat162float(__float2bfloat16_rn(__half2float(x[i])));
            if constexpr(Double)sum+=Acc(a)*Acc(__bfloat162float(weight[i]));
            else sum=__fmaf_rn(__bfloat162float(weight[i]),a,sum);
        }
    }
#pragma unroll
    for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(warp==0) {
        sum=lane<Threads/32 ? partial[lane] : Acc(0);
#pragma unroll
        for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
        if(lane==0) {
            const auto rounded=__float2bfloat16_rn(float(sum));
            if(dot)dot[blockIdx.x]=rounded;
            if constexpr(HalfGate) {
                // dense_projection returns the BF16 dot through FP16 before
                // sigmoid. Preserve that boundary and its FP16 result.
                const float projected=__half2float(__float2half_rn(__bfloat162float(rounded)));
                gate[blockIdx.x]=__float2half_rn(1.f/(1.f+expf(-projected)));
            } else gate[blockIdx.x]=__float2bfloat16_rn(1.f/(1.f+expf(-__bfloat162float(rounded))));
        }
    }
}
}
