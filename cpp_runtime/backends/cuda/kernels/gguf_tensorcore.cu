#include "gguf.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <climits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace mfq::cuda::gguf {
namespace {
template<typename Value>
__global__ void pack_rows(const float* input,Value* output,std::int32_t* exponents,
        std::int64_t columns) {
    const auto row=std::int64_t(blockIdx.x);
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ float maxima[8];
    float maximum=0;
    for(auto k=std::int64_t(threadIdx.x);k<columns;k+=blockDim.x)
        maximum=fmaxf(maximum,fabsf(input[row*columns+k]));
    for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
    if(!lane)maxima[warp]=maximum;
    __syncthreads();
    if(!warp) {
        maximum=lane<8?maxima[lane]:0.f;
        for(int offset=16;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
        if(!lane) {
            // Keep the largest operand between 32 and 64. FP16 underflow then
            // affects only values far below the row's largest contribution.
            const int exponent=maximum>0 && isfinite(maximum)?max(-126,min(122,ilogbf(maximum)-5)):0;
            exponents[row]=exponent;maxima[0]=ldexpf(1.f,-exponent);
        }
    }
    __syncthreads();
    const float inverse=maxima[0];
    for(auto k=std::int64_t(threadIdx.x);k<columns;k+=blockDim.x)
        output[row*columns+k]=Value(input[row*columns+k]*inverse);
}
__global__ void restore_scale(float* output,const std::int32_t* input_exponents,
        const std::int32_t* weight_exponents,std::int64_t tokens,std::int64_t rows) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<tokens*rows;i+=std::int64_t(gridDim.x)*blockDim.x)
        output[i]=ldexpf(output[i],input_exponents[i/rows]+(weight_exponents?weight_exponents[i%rows]:0));
}
void checked(cudaError_t status) {if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
}
void pack_scaled_f16(const float* input,void* output,std::int32_t* exponents,
        std::int64_t rows,std::int64_t columns,cudaStream_t stream) {
    if(!rows || !columns)return;
    if(rows>INT_MAX)throw std::invalid_argument("GGUF scaled prefill rows exceed CUDA grid capacity");
    pack_rows<<<unsigned(rows),256,0,stream>>>(input,static_cast<half*>(output),exponents,columns);
    checked(cudaGetLastError());
}
void gemm_scaled_f16(const void* input,const void* weight,const std::int32_t* input_exponents,
        const std::int32_t* weight_exponents,float* output,std::int64_t tokens,std::int64_t rows,
        std::int64_t columns,void* blas_handle,cudaStream_t stream) {
    if(!tokens || !rows || !columns)return;
    if(tokens>INT_MAX || rows>INT_MAX || columns>INT_MAX)
        throw std::invalid_argument("GGUF scaled prefill dimensions exceed cuBLAS capacity");
    const float alpha=1.f,beta=0.f;
    const auto status=cublasGemmEx(static_cast<cublasHandle_t>(blas_handle),CUBLAS_OP_T,CUBLAS_OP_N,
        int(rows),int(tokens),int(columns),&alpha,weight,CUDA_R_16F,int(columns),input,CUDA_R_16F,int(columns),
        &beta,output,CUDA_R_32F,int(rows),CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("GGUF scaled FP16 GEMM failed: "+std::to_string(int(status)));
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))
            std::fprintf(stderr,"native_gguf_prefill_scaled_f16 rows=%lld output_rows=%lld columns=%lld fp32_output=1 power_of_two_row_scaling=1\n",
                static_cast<long long>(tokens),static_cast<long long>(rows),static_cast<long long>(columns));
    }
    const auto blocks=unsigned(std::min(std::int64_t(65535),(tokens*rows+255)/256));
    restore_scale<<<blocks,256,0,stream>>>(output,input_exponents,weight_exponents,tokens,rows);
    checked(cudaGetLastError());
}
void pack_scaled_f32(const float* input,float* output,std::int32_t* exponents,
        std::int64_t rows,std::int64_t columns,cudaStream_t stream) {
    if(rows<=0 || columns<=0 || rows>INT_MAX)throw std::invalid_argument("GGUF scaled MMQ geometry differs");
    pack_rows<<<unsigned(rows),256,0,stream>>>(input,output,exponents,columns);checked(cudaGetLastError());
}
void rescale_f32(float* output,const std::int32_t* input_exponents,
        std::int64_t tokens,std::int64_t rows,cudaStream_t stream) {
    const auto blocks=unsigned(std::min(std::int64_t(65535),(tokens*rows+255)/256));
    restore_scale<<<blocks,256,0,stream>>>(output,input_exponents,nullptr,tokens,rows);checked(cudaGetLastError());
}
}
