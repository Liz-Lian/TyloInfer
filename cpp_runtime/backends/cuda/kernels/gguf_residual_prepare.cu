#include "gguf_residual_prepare.h"
#include "ggml-cuda/common.cuh"
#include "gguf_quantize.cuh"
#include "kernel_dispatch_trace.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <type_traits>
namespace mfq::cuda::gguf {
namespace {
template<class Branch,class Injection,bool Post,int Threads>
__global__ void prepare_gr_input(const float* residual,const Branch* branch,const Injection* injection,
        const float* weight,float* updated,float* normalized,block_q8_1* q8,int hidden,int streams,float eps) {
    const auto base=std::int64_t(blockIdx.y)*hidden,sample=std::int64_t(blockIdx.y/streams);
    const int sid=blockIdx.y%streams;
    using Update=std::conditional_t<std::is_same_v<Branch,float> || std::is_same_v<Injection,float>,float,half>;
    const auto value_at=[&](std::int64_t column) {
        if constexpr(Post) {
            const Update product=Update(__fmul_rn(float(branch[sample*hidden+column]),float(injection[sample*streams+sid])));
            return __fadd_rn(residual[base+column],float(product));
        } else return residual[base+column];
    };
    float sum=0;
    for(std::int64_t column=threadIdx.x;column<hidden;column+=Threads) {
        const float value=value_at(column);sum=__fadd_rn(sum,__fmul_rn(value,value));
    }
    // Preserve the original GR reduction and F32 operations. Replica CTAs
    // only partition output coordinates; none write another CTA's region.
    __shared__ float partial[Threads];partial[threadIdx.x]=sum;__syncthreads();
    for(int stride=Threads/2;stride;stride>>=1) {
        if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    const float inverse=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(partial[0],float(hidden)),eps)));
    for(std::int64_t column=std::int64_t(blockIdx.x)*Threads+threadIdx.x;column<hidden;column+=std::int64_t(gridDim.x)*Threads) {
        const float raw=value_at(column);
        if constexpr(Post)updated[base+column]=raw;
        const float value=__fmul_rn(__fmul_rn(raw,inverse),__fadd_rn(weight[std::int64_t(sid)*hidden+column],1.f));
        normalized[base+column]=value;store_q8(value,base+column,q8);
    }
}
int residual_prepare_replicas(int groups,int hidden,int threads) {
    struct DeviceSm {int device=-1,sm=1;};thread_local DeviceSm cached;
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess)return 1;
    if(cached.device!=device) {
        int sm=1;if(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess)return 1;
        cached={device,std::max(sm,1)};
    }
    // Small GR batches otherwise leave almost every SM idle. Limit
    // duplicated reduction work while distributing complete output tiles.
    const int target=(cached.sm+1)/2,needed=(target+groups-1)/groups;
    const int tiles=(hidden-1)/threads+1;
    int replicas=1;
    while(replicas<needed && replicas<32 && replicas*2<=tiles)replicas*=2;
    return replicas;
}
template<class Branch,class Injection,bool Post,int Threads>
void submit_gr(const float* residual,const void* branch,const void* injection,const float* norm,
        int groups,int hidden,int streams,float eps,float* updated,float* normalized,void* q8,cudaStream_t stream) {
    const int replicas=residual_prepare_replicas(groups,hidden,Threads);
    const auto function=prepare_gr_input<Branch,Injection,Post,Threads>;
    function<<<dim3(replicas,groups),Threads,0,stream>>>(residual,static_cast<const Branch*>(branch),
        static_cast<const Injection*>(injection),norm,updated,normalized,static_cast<block_q8_1*>(q8),hidden,streams,eps);
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
        "gguf_gr_input_q8",reinterpret_cast<const void*>(function),"groups="+std::to_string(groups)+
        " hidden="+std::to_string(hidden)+" replicas="+std::to_string(replicas)+" post="+std::to_string(Post));
}
}
void prepare_residual_input(const float* residual,const void* branch,bool half_branch,
        const void* injection,bool half_injection,const float* norm,
        std::int64_t tokens,std::int64_t hidden,std::int64_t streams,float eps,
        float* updated,float* normalized,void* native_q8,cudaStream_t stream) {
    if(!residual || !norm || !normalized || !native_q8 || tokens<1 || streams<1 || streams>65535 ||
       tokens>65535/streams || hidden<32 || hidden>INT_MAX || hidden%32 ||
       bool(branch)!=bool(injection) || normalized==residual ||
       (branch && (!updated || updated==residual || updated==normalized)))
        throw std::invalid_argument("GGUF GR input preparation geometry/storage differs");
    const auto dispatch=[&]<class Branch,class Injection,bool Post>() {
        const auto launch=[&]<int Threads>(){submit_gr<Branch,Injection,Post,Threads>(residual,branch,injection,norm,
            int(tokens*streams),int(hidden),int(streams),eps,updated,normalized,native_q8,stream);};
        if(hidden<=32)launch.template operator()<32>();
        else if(hidden<=64)launch.template operator()<64>();
        else if(hidden<=128)launch.template operator()<128>();
        else launch.template operator()<256>();
    };
    if(!branch)dispatch.template operator()<float,float,false>();
    else if(half_branch && half_injection)dispatch.template operator()<half,half,true>();
    else if(half_branch)dispatch.template operator()<half,float,true>();
    else if(half_injection)dispatch.template operator()<float,half,true>();
    else dispatch.template operator()<float,float,true>();
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
}
