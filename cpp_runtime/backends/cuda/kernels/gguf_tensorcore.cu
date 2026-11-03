#include "gguf.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <climits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <memory>
#include <tuple>
#include <unordered_map>


#include "gguf_prefill_scale.cuh"
#include "reduce.cuh"

namespace mfq::cuda::gguf {
namespace {
template<typename Value,int TargetExponent=5>
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
            int exponent=maximum>0 && isfinite(maximum)?max(-126,min(122,ilogbf(maximum)-TargetExponent)):0;
            if constexpr(TargetExponent<0) {
                // Bound an entire dot product below 2^14 even if all terms
                // have the same sign. Both operands use this normalization.
                exponent=prefill_unit_exponent(maximum,columns);
            }
            exponents[row]=exponent;maxima[0]=prefill_ldexp(1.f,-exponent);
        }
    }
    __syncthreads();
    const float inverse=maxima[0];
    for(auto k=std::int64_t(threadIdx.x);k<columns;k+=blockDim.x)
        if constexpr(TargetExponent<0)output[row*columns+k]=Value(prefill_ldexp(input[row*columns+k],-exponents[row]));
        else output[row*columns+k]=Value(input[row*columns+k]*inverse);
}
__global__ void restore_scale(float* output,const std::int32_t* input_exponents,
        const std::int32_t* weight_exponents,std::int64_t tokens,std::int64_t rows) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<tokens*rows;i+=std::int64_t(gridDim.x)*blockDim.x)
        output[i]=prefill_ldexp(output[i],input_exponents[i/rows]+(weight_exponents?weight_exponents[i%rows]:0));
}
__global__ void restore_scale_strided(float* output,const std::int32_t* input_exponents,
        const std::int32_t* weight_exponents,std::int64_t tokens,std::int64_t rows,std::int64_t stride) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<tokens*rows;i+=std::int64_t(gridDim.x)*blockDim.x) {
        auto& value=output[i/rows*stride+i%rows];
        value=prefill_ldexp(value,(input_exponents?input_exponents[i/rows]:0)+
            (weight_exponents?weight_exponents[i%rows]:0));
    }
}
void checked(cudaError_t status) {if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
#include "gguf_prefill_lt.cuh"
}
void pack_scaled_f16(const float* input,void* output,std::int32_t* exponents,
        std::int64_t rows,std::int64_t columns,cudaStream_t stream) {
    if(!rows || !columns)return;
    if(rows>INT_MAX)throw std::invalid_argument("GGUF scaled prefill rows exceed CUDA grid capacity");
    pack_rows<<<unsigned(rows),256,0,stream>>>(input,static_cast<half*>(output),exponents,columns);
    checked(cudaGetLastError());
}
void prefill_gemm_f16(const void* input,const void* weight,void* output,bool half_output,
        const std::int32_t* input_exponents,const std::int32_t* weight_exponents,
        std::int64_t tokens,std::int64_t rows,std::int64_t columns,std::int64_t output_stride,
        void* blas_handle,cudaStream_t stream) {
    if(!tokens || !rows || !columns)return;
    if(tokens<0 || rows<0 || columns<0 || tokens>INT_MAX || rows>INT_MAX ||
            columns>INT_MAX || output_stride<rows || output_stride>INT_MAX ||
            (half_output && (input_exponents || weight_exponents)))
        throw std::invalid_argument("GGUF prefill GEMM dimensions or output scaling are invalid");
    const float alpha=1.f,beta=0.f;
    const auto status=cublasGemmEx(static_cast<cublasHandle_t>(blas_handle),CUBLAS_OP_T,CUBLAS_OP_N,
        int(rows),int(tokens),int(columns),&alpha,weight,CUDA_R_16F,int(columns),input,CUDA_R_16F,int(columns),
        &beta,output,half_output?CUDA_R_16F:CUDA_R_32F,int(output_stride),CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("GGUF direct prefill GEMM failed: "+std::to_string(int(status)));
    if(input_exponents || weight_exponents) {
        const auto blocks=unsigned(std::min(std::int64_t(65535),(tokens*rows+255)/256));
        restore_scale_strided<<<blocks,256,0,stream>>>(static_cast<float*>(output),input_exponents,
            weight_exponents,tokens,rows,output_stride);
        checked(cudaGetLastError());
    }
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

namespace mfq::cuda::gguf {
namespace {
template<typename Value>
__global__ void restore_half_product(const half* source,Value* output,const std::int32_t* input_exponents,
        const std::int32_t* weight_exponents,std::int64_t tokens,std::int64_t rows,std::int64_t stride) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<tokens*rows;i+=std::int64_t(gridDim.x)*blockDim.x)
        output[i/rows*stride+i%rows]=Value(prefill_ldexp(__half2float(source[i]),input_exponents[i/rows]+weight_exponents[i%rows]));
}
}
namespace {
template<bool Pairs>
__global__ void acc_rms_product(const half* product,const half* residual,const float* norm,
    const int32_t* row_exponents,const int32_t* column_exponents,
    int columns,float eps,float offset,half* sum,half* output) {
    const int64_t row=blockIdx.x,base=row*columns;
    const int exponent=row_exponents[row];
    float squared=0;
    if constexpr(Pairs) {
        const int pairs=columns/2;
        const auto p=reinterpret_cast<const half2*>(product+base),r=reinterpret_cast<const half2*>(residual+base);
        auto s=reinterpret_cast<half2*>(sum+base);
        for(int i=threadIdx.x;i<pairs;i+=256) {
            const auto value=__half22float2(p[i]);
            const auto added=__floats2half2_rn(prefill_ldexp(value.x,exponent+column_exponents[2*i]),
                prefill_ldexp(value.y,exponent+column_exponents[2*i+1]));
            const half2 stored=__hadd2(r[i],added);s[i]=stored;
            const auto v=__half22float2(stored);squared+=v.x*v.x+v.y*v.y;
        }
    }else for(int i=threadIdx.x;i<columns;i+=256) {
        const half added=__float2half_rn(prefill_ldexp(__half2float(product[base+i]),exponent+column_exponents[i]));
        const half stored=__float2half_rn(__half2float(residual[base+i])+__half2float(added));
        sum[base+i]=stored;const float v=__half2float(stored);squared+=v*v;
    }
    squared=block_sum<8>(squared);
    const float inverse=rsqrtf(squared/columns+eps);
    if constexpr(Pairs) {
        const auto s=reinterpret_cast<const half2*>(sum+base);
        const auto w=reinterpret_cast<const float2*>(norm);
        auto result=reinterpret_cast<half2*>(output+base);
        for(int i=threadIdx.x;i<columns/2;i+=256) {
            const auto value=__half22float2(s[i]),scale=w[i];
            result[i]=__floats2half2_rn(value.x*inverse*(scale.x+offset),value.y*inverse*(scale.y+offset));
        }
    }else for(int i=threadIdx.x;i<columns;i+=256)
        output[base+i]=__float2half_rn(__half2float(sum[base+i])*inverse*(norm[i]+offset));
}
}
void restore_scaled_f16(const void* input,void* output,bool half_output,
    const int32_t* row_exponents,const int32_t* column_exponents,
    int64_t rows,int64_t columns,int64_t stride,cudaStream_t stream) {
    if(rows<1 || columns<1 || rows>INT64_MAX/columns || stride<columns || !input || !output ||
        !row_exponents || !column_exponents)throw std::invalid_argument("Invalid scaled prefill output geometry");
    const auto blocks=unsigned(std::min(int64_t(65535),(rows*columns+255)/256));
    if(half_output)restore_half_product<<<blocks,256,0,stream>>>(static_cast<const half*>(input),static_cast<half*>(output),row_exponents,column_exponents,rows,columns,stride);
    else restore_half_product<<<blocks,256,0,stream>>>(static_cast<const half*>(input),static_cast<float*>(output),row_exponents,column_exponents,rows,columns,stride);
    checked(cudaGetLastError());
}
void acc_rms_scaled_f16(const void* product,const void* residual,const float* norm,
    const int32_t* row_exponents,const int32_t* column_exponents,int64_t rows,int64_t columns,
    float eps,float offset,void* sum,void* output,cudaStream_t stream) {
    if(rows<1 || rows>INT_MAX || columns<1 || columns>INT_MAX || rows>INT64_MAX/columns ||
        !product || !residual || !norm || !row_exponents || !column_exponents || !sum || !output)
        throw std::invalid_argument("Invalid scaled prefill residual/RMS geometry");
    const auto aligned=[](const void* p,size_t n){return (reinterpret_cast<uintptr_t>(p)&(n-1))==0;};
    const bool pairs=columns>=2048 && columns%2==0 && aligned(product,4) && aligned(residual,4) &&
        aligned(norm,8) && aligned(sum,4) && aligned(output,4);
    const auto launch=[&]<bool Pairs>() {
        acc_rms_product<Pairs><<<unsigned(rows),256,0,stream>>>(static_cast<const half*>(product),static_cast<const half*>(residual),
            norm,row_exponents,column_exponents,int(columns),eps,offset,static_cast<half*>(sum),static_cast<half*>(output));
    };
    if(pairs)launch.template operator()<true>();else launch.template operator()<false>();
    checked(cudaGetLastError());
}

void pack_scaled_f16_unit(const float* input,void* output,std::int32_t* exponents,
        std::int64_t rows,std::int64_t columns,cudaStream_t stream) {
    if(!rows || !columns)return;
    if(rows<0 || columns<0 || rows>INT_MAX)throw std::invalid_argument("GGUF unit-scaled prefill dimensions are invalid");
    pack_rows<half,-1><<<unsigned(rows),256,0,stream>>>(input,static_cast<half*>(output),exponents,columns);
    checked(cudaGetLastError());
}
PrefillResidualGemmPlan prepare_prefill_residual_gemm(const void* input,const void* weight,
        const void* residual,void* output,int64_t tokens,int64_t rows,int64_t columns,int64_t output_stride,cudaStream_t stream) {
    if(!tokens || !rows || !columns)return {};
    if(!input || !weight || !residual || !output || output==residual || tokens<0 || rows<0 || columns<0 ||
            tokens>INT_MAX || rows>INT_MAX || columns>INT_MAX || output_stride<rows || output_stride>INT_MAX)
        throw std::invalid_argument("Invalid GGUF residual GEMM planning geometry");
    auto plan=prefill_lt_residual_plan(input,weight,residual,output,tokens,rows,columns,output_stride,stream);
    const auto bytes=plan?plan->workspace:0;
    return {std::move(plan),bytes};
}

void prefill_gemm_f16_residual(const void* input,const void* weight,const void* residual,
        void* output,int64_t tokens,int64_t rows,int64_t columns,int64_t output_stride,
        const PrefillResidualGemmPlan& prepared,void* workspace,size_t workspace_bytes,void* blas_handle,cudaStream_t stream) {
    if(!tokens || !rows || !columns)return;
    if(!input || !weight || !residual || !output || output==residual || !blas_handle || tokens<0 || rows<0 || columns<0 ||
            tokens>INT_MAX || rows>INT_MAX || columns>INT_MAX || output_stride<rows || output_stride>INT_MAX)
        throw std::invalid_argument("GGUF FP16 residual GEMM dimensions are invalid");
    if(prepared) {
        const auto plan=std::static_pointer_cast<PrefillLtResidual>(prepared.implementation);
        int device=0;checked(cudaGetDevice(&device));
        if(plan->owner->device!=device || plan->tokens!=tokens || plan->rows!=rows || plan->columns!=columns ||
                plan->stride!=output_stride || workspace_bytes<plan->workspace || (plan->workspace && !workspace))
            throw std::invalid_argument("GGUF residual GEMM plan or scratch differs from its operands");
        const half one=__float2half(1.f);
        prefill_lt_checked(cublasLtMatmul(plan->owner->handle,plan->operation,&one,weight,plan->a,input,plan->b,
            &one,residual,plan->c,output,plan->c,&plan->algorithm,workspace,workspace_bytes,stream));
    } else {
        prefill_gemm_f16_accum(input,weight,nullptr,output,true,nullptr,nullptr,tokens,rows,columns,output_stride,blas_handle,stream);
        const auto blocks=unsigned(std::min(int64_t(65535),(tokens*rows+255)/256));
        prefill_residual_add<<<blocks,256,0,stream>>>(static_cast<half*>(output),static_cast<const half*>(residual),tokens*rows,rows,output_stride);
        checked(cudaGetLastError());
    }
    // The library adds C before the final FP16 store. This removes one FP16
    // rounding boundary relative to a separate result and residual add.
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported[2]{};
        if(!reported[int(bool(prepared))].exchange(true))std::fprintf(stderr,"native_gguf_prefill_gemm_residual actual_match=1 M=%lld N=%lld K=%lld immutable_residual=1 lt=%d separate_C_D=1 scratch_bytes=%zu allocator_owned=1\n",
            static_cast<long long>(tokens),static_cast<long long>(rows),static_cast<long long>(columns),int(bool(prepared)),prepared.workspace_bytes);
    }
}

void prefill_gemm_f16_accum(const void* input,const void* weight,void* temporary,void* output,bool half_output,
        const std::int32_t* input_exponents,const std::int32_t* weight_exponents,
        std::int64_t tokens,std::int64_t rows,std::int64_t columns,std::int64_t output_stride,
        void* blas_handle,cudaStream_t stream) {
    if(!tokens || !rows || !columns)return;
    if(tokens<0 || rows<0 || columns<0 || tokens>INT_MAX || rows>INT_MAX || columns>INT_MAX ||
            output_stride<rows || output_stride>INT_MAX || ((input_exponents || weight_exponents) && (!input_exponents || !weight_exponents || !temporary)) || (!input_exponents && !half_output))
        throw std::invalid_argument("GGUF FP16 accumulation dimensions or scaling are invalid");
    const half alpha=__float2half(1.f),beta=__float2half(0.f);
    const auto status=cublasGemmEx(static_cast<cublasHandle_t>(blas_handle),CUBLAS_OP_T,CUBLAS_OP_N,
        int(rows),int(tokens),int(columns),&alpha,weight,CUDA_R_16F,int(columns),input,CUDA_R_16F,int(columns),
        &beta,input_exponents?temporary:output,CUDA_R_16F,int(input_exponents?rows:output_stride),CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("GGUF FP16 accumulation GEMM failed: "+std::to_string(int(status)));
    if(!input_exponents)return;
    const auto blocks=unsigned(std::min(std::int64_t(65535),(tokens*rows+255)/256));
    if(half_output)restore_half_product<<<blocks,256,0,stream>>>(static_cast<const half*>(temporary),static_cast<half*>(output),input_exponents,weight_exponents,tokens,rows,output_stride);
    else restore_half_product<<<blocks,256,0,stream>>>(static_cast<const half*>(temporary),static_cast<float*>(output),input_exponents,weight_exponents,tokens,rows,output_stride);
    checked(cudaGetLastError());
}
}
