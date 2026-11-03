#include "gguf_m1_residual.h"
#include "gguf.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "gguf_quantize.cuh"
#include "gguf_dense_rows.cuh"
#include "kernel_dispatch_trace.h"
#include <climits>
#include <stdexcept>

#define RESIDUAL_FORMATS(X) X(IQ1_S) X(IQ1_M) X(IQ2_XXS) X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S) X(Q2_K) X(Q4_K) X(Q5_K) X(Q6_K) X(IQ4_XS)
namespace mfq::cuda::gguf { namespace {
#include "gguf_dense_m1_aligned.cuh"
struct ResidualProjectionStore {
    const half* residual;
    __device__ __forceinline__ void operator()(void* output,int64_t row,float value,bool) const {
        static_cast<half*>(output)[row]=__hadd(residual[row],__float2half_rn(value));
    }
};
void check_residual(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
template<int F,int R,int W,bool Aligned,int BlockWarps=4,bool Cached=false>
__global__ void __launch_bounds__(32*BlockWarps) project_m1_residual(
        const uint8_t* packed,uint64_t stride,int rows,int columns,
        const void* input,const half* residual,half* output) {
    __shared__ float scratch[4*BlockWarps];
    const ResidualProjectionStore store{residual};
    using Dot=M1ProjectionDot<F,Cached>;
    if constexpr(Cached)Dot::stage(threadIdx.y*32+threadIdx.x,32*BlockWarps);
    if constexpr(Aligned)
        project_aligned_m1_body<F,R,W,BlockWarps,ResidualProjectionStore,Dot>(packed,stride,rows,columns,
            static_cast<const uint4*>(input),
            reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(input)+columns),
            output,true,blockIdx.x,scratch,store);
    else project_dense_reused_q8_body<F,R,W,BlockWarps,ResidualProjectionStore,Dot>(packed,stride,rows,columns,
        static_cast<const block_q8_1*>(input),output,true,blockIdx.x,scratch,store);
}
template<int F,bool Aligned>int residual_k_warps(int64_t rows,int64_t columns) {
    if constexpr(F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M)return 1;
    const auto parts=columns/M1PreparedDot<F>::qk*M1PreparedDot<F>::parts;
    if(parts<128)return 1;
    const int cooperative_warps=parts%(3*32)==0 && parts%(4*32)!=0?3:4;
    if constexpr(F==GGML_TYPE_IQ4_XS)return cooperative_warps;
    int device=0;check_residual(cudaGetDevice(&device));
    static thread_local int cached_device=-1,sms=0,active=0;
    if(cached_device!=device) {
        check_residual(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
        check_residual(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active,project_m1_residual<F,2,1,Aligned>,128,0));
        cached_device=device;
    }
    return active>0 && uint64_t((rows+7)/8)<uint64_t(sms)*unsigned(active)?cooperative_warps:1;
}
}}
namespace mfq::cuda::gguf {
bool supports_m1_residual(ggml_type type) noexcept {
    switch(type) {
#define CASE(F) case GGML_TYPE_##F:return true;
    RESIDUAL_FORMATS(CASE)
#undef CASE
    default:return false;
    }
}
void matvec_m1_residual(ggml_type type,const void* packed,uint64_t stride,
        int64_t rows,int64_t columns,const void* input,bool aligned,
        const void* residual,void* output,cudaStream_t stream) {
    if(!supports_m1_residual(type)||rows<0||rows>INT_MAX||columns<=0||columns>INT_MAX||
            columns%ggml_blck_size(type)||stride<ggml_row_size(type,columns)||
            (rows && stride>UINT64_MAX/uint64_t(rows)))
        throw std::invalid_argument("M=1 residual projection geometry/format differs");
    if(!rows)return;
    if(!packed||!input||!residual||!output||
            (reinterpret_cast<uintptr_t>(packed)%2)||
            (reinterpret_cast<uintptr_t>(input)%(aligned?16:4))||
            (reinterpret_cast<uintptr_t>(residual)%2)||
            (reinterpret_cast<uintptr_t>(output)%2))
        throw std::invalid_argument("M=1 residual projection storage differs");
    const auto launch=[&]<int F,int R,int W,bool Aligned,bool Cached=false>() {
        constexpr int BlockWarps=W==3?3:4;
        const auto fn=project_m1_residual<F,R,W,Aligned,BlockWarps,Cached>;
        constexpr unsigned Shared=[] {if constexpr(Cached)return SharedM1PreparedDot<F>::shared_bytes;else return 0u;}();
        constexpr int Outputs=BlockWarps/W*R;
        fn<<<unsigned((rows+Outputs-1)/Outputs),dim3(32,BlockWarps),Shared,stream>>>(
            static_cast<const uint8_t*>(packed),stride,int(rows),int(columns),input,
            static_cast<const half*>(residual),static_cast<half*>(output));
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_m1_residual",
            reinterpret_cast<const void*>(fn),"M=1 format="+std::to_string(F)+
            " k_warps="+std::to_string(W)+" aligned="+std::to_string(Aligned)+
            " shared_codebook="+std::to_string(Cached));
    };
    const auto format=[&]<int F,bool Aligned>() {
        if constexpr(F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M) {
            if(columns/M1PreparedDot<F>::qk*M1PreparedDot<F>::parts>=64) {
                launch.template operator()<F,4,2,Aligned,true>();return;
            }
        }
        const auto warps=residual_k_warps<F,Aligned>(rows,columns);
        if(warps==3)
            launch.template operator()<F,2,3,Aligned>();
        else if(warps==4)
            launch.template operator()<F,2,4,Aligned>();
        else launch.template operator()<F,2,1,Aligned>();
    };
    const auto select=[&]<int F>() {
        if(aligned)format.template operator()<F,true>();
        else format.template operator()<F,false>();
    };
    switch(type) {
#define CASE(F) case GGML_TYPE_##F:select.template operator()<GGML_TYPE_##F>();break;
    RESIDUAL_FORMATS(CASE)
#undef CASE
    default:throw std::invalid_argument("Unsupported M=1 residual projection");
    }
    check_residual(cudaGetLastError());
}
}
#undef RESIDUAL_FORMATS
