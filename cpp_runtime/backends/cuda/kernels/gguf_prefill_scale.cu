#include "ggml-cuda/common.cuh"
#include "tyloinfer/gguf_quant.h"
#include "gguf_decode.cuh"
#include <algorithm>
#include <stdexcept>

#include "gguf_prefill_scale.cuh"

namespace mfq::cuda::gguf { namespace {
template<int F>__global__ void row_exponents(const unsigned char* packed,std::uint64_t stride,
        std::int64_t columns,std::int32_t* exponents) {
    const auto row=std::int64_t(blockIdx.x);float maximum=0;
    for(auto k=std::int64_t(threadIdx.x);k<columns;k+=blockDim.x)
        maximum=fmaxf(maximum,fabsf(gguf_value<F>(packed+row*stride,k)));
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    __shared__ float maxima[8];const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(!lane)maxima[warp]=maximum;__syncthreads();
    if(!warp) {
        maximum=lane<8?maxima[lane]:0.f;
        for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
        if(!lane) {
            exponents[row]=prefill_unit_exponent(maximum,columns);
        }
    }
}
template<int F>__global__ void decode_scaled(const unsigned char* packed,std::uint64_t stride,
        std::int64_t rows,std::int64_t columns,const std::int32_t* exponents,half* output) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*columns;i+=std::int64_t(gridDim.x)*blockDim.x) {
        const auto row=i/columns,k=i%columns;
        output[i]=__float2half_rn(prefill_ldexp(gguf_value<F>(packed+row*stride,k),-exponents[row]));
    }
}
void checked(){const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
void prefill_row_exponents(ggml_type type,const void* packed,std::uint64_t stride,std::int64_t rows,
        std::int64_t columns,std::int32_t* output,cudaStream_t stream) {
    if(rows<1 || rows>INT_MAX || columns<1)throw std::invalid_argument("GGUF row exponent dimensions are invalid");
    switch(type) {
#define ROW_SCALE(F) case GGML_TYPE_##F:row_exponents<GGML_TYPE_##F><<<unsigned(rows),256,0,stream>>>(static_cast<const unsigned char*>(packed),stride,columns,output);break;
        TYLOINFER_GGUF_FORMATS(ROW_SCALE)
#undef ROW_SCALE
        default:throw std::invalid_argument("Unsupported GGUF row exponent format");
    }
    checked();
}
void dequantize_rows_scaled(ggml_type type,const void* packed,std::uint64_t stride,std::int64_t rows,
        std::int64_t columns,const std::int32_t* exponents,void* output,cudaStream_t stream) {
    if(rows<1 || columns<1 || rows>INT64_MAX/columns)throw std::invalid_argument("GGUF scaled decode dimensions are invalid");
    const unsigned blocks=unsigned(std::min(std::int64_t(65535),(rows*columns+255)/256));
    switch(type) {
#define SCALE_DECODE(F) case GGML_TYPE_##F:decode_scaled<GGML_TYPE_##F><<<blocks,256,0,stream>>>(static_cast<const unsigned char*>(packed),stride,rows,columns,exponents,static_cast<half*>(output));break;
        TYLOINFER_GGUF_FORMATS(SCALE_DECODE)
#undef SCALE_DECODE
        default:throw std::invalid_argument("Unsupported GGUF scaled decode format");
    }
    checked();
}
}
