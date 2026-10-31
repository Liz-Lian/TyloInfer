#pragma once
namespace mfq::cuda::gguf {
namespace {
__device__ __forceinline__ void store_projection(float* output,std::int64_t at,float value) {output[at]=value;}
__device__ __forceinline__ void store_projection(half* output,std::int64_t at,float value) {output[at]=__float2half_rn(value);}
__device__ __forceinline__ void store_projection(void* output,std::int64_t at,float value,bool half_output) {
    if(half_output)store_projection(static_cast<half*>(output),at,value);
    else store_projection(static_cast<float*>(output),at,value);
}
__device__ float finite_half(float value) {
    return fabsf(value) > 65504.f ? copysignf(65504.f, value) : value;
}
__device__ __forceinline__ void store_q8(float value,std::int64_t i,block_q8_1* output) {
    float maximum = fabsf(value), sum = value;
    for (int offset = 16; offset; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    // Keep extreme activation blocks finite, as in Strata q8_1_finite.hpp.
    const float scale = finite_half(maximum / 127.f);
    const float q = maximum == 0 ? 0 : roundf(value / scale);
    output[i / 32].qs[i % 32] = std::int8_t(fminf(127.f, fmaxf(-127.f, q)));
    if (i % 32 == 0) output[i / 32].ds = __floats2half2_rn(scale, finite_half(sum));
}
template<bool Half>
__device__ __forceinline__ float swiglu_q8_value(float g,float u,float limit) {
    if constexpr(Half) {g=__half2float(__float2half_rn(g));u=__half2float(__float2half_rn(u));}
    float value;
    if(limit>0) {
        g=fminf(g,limit);u=fminf(limit,fmaxf(-limit,u));
        value=__fmul_rn(__fmul_rn(g,1.f/(1.f+expf(-g))),u);
        if constexpr(Half)value=__half2float(__float2half_rn(value));
    } else {
        float sigmoid=1.f/(1.f+expf(-g));
        if constexpr(Half)sigmoid=__half2float(__float2half_rn(sigmoid));
        float silu=__fmul_rn(g,sigmoid);
        if constexpr(Half)silu=__half2float(__float2half_rn(silu));
        value=__fmul_rn(silu,u);
        if constexpr(Half)value=__half2float(__float2half_rn(value));
    }
    return value;
}
// The same Q8 codes and half-rounded scales, arranged as
// [word][K block][padded token] followed by [K block][padded token] scales.
__device__ __forceinline__ void store_q8_position_major(float value,std::int64_t i,
        std::int64_t columns,int tokens,void* output,block_q8_1* native=nullptr) {
    float maximum=fabsf(value),sum=value;
    for(int offset=16;offset;offset>>=1) {
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
        if(native)sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
    }
    const float scale=finite_half(maximum/127.f);
    const float q=maximum==0?0:roundf(value/scale);
    const auto code=std::int8_t(fminf(127.f,fmaxf(-127.f,q)));
    const int capacity=tokens==2?2:4;
    const auto token=i/columns,k=i%columns,blocks=columns/32,block=k/32,lane=k%32;
    static_cast<std::int8_t*>(output)[((lane/4*blocks+block)*capacity+token)*4+lane%4]=code;
    if(tokens==3 && token==0)
        static_cast<std::int8_t*>(output)[((lane/4*blocks+block)*capacity+3)*4+lane%4]=0;
    if(!lane)reinterpret_cast<float*>(static_cast<std::uint8_t*>(output)+columns*capacity)[block*capacity+token]
        =__half2float(__float2half_rn(scale));
    if(tokens==3 && token==0 && !lane)
        reinterpret_cast<float*>(static_cast<std::uint8_t*>(output)+columns*capacity)[block*capacity+3]=0;
    if(native && token<tokens) {
        native[i/32].qs[lane]=code;
        if(!lane)native[i/32].ds=__floats2half2_rn(scale,finite_half(sum));
    }
}
}
}
