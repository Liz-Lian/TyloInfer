#include "gguf_residual.h"
#include "gguf.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "gguf_quantize.cuh"
#include "warp_sums.cuh"
#include "kernel_dispatch_trace.h"
#include <climits>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace mfq::cuda::gguf {
namespace {
__global__ void prepare_activation(const float* down,const float* injection,
        std::int64_t count,std::int64_t injection_count,float streams,
        block_q8_1* packed,float* injection_output) {
    const auto start=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto stride=std::int64_t(gridDim.x)*blockDim.x;
    // columns is a multiple of a Q8 warp. The final CTA therefore retains
    // complete 32-lane reductions, including widths shorter than one CTA.
    for(auto i=start;i<count;i+=stride) {
        const float low=__fdiv_rn(down[i],streams);
        const float gate=1.f/(1.f+expf(-low));
        store_q8(__fmul_rn(low,gate),i,packed);
    }
    for(auto i=start;i<injection_count;i+=stride) {
        const float low=__fdiv_rn(injection[i],streams);
        injection_output[i]=__fmul_rn(2.f,1.f/(1.f+expf(-low)));
    }
}

template<int Format,int M>
__global__ void project_residual_mix(const std::uint8_t* packed,std::uint64_t stride,
        int hidden,int columns,int streams,const block_q8_1* input,const float* normalized,float* output) {
    using F=PreparedDot<Format>;
    const auto row=std::int64_t(blockIdx.x)*4+threadIdx.y;
    if(row>=hidden)return;
    // Distribute up to four streams over one warp. A one/two-stream GR
    // retains 32/16 lanes per projection instead of leaving lanes idle.
    const int lanes=streams==1?32:streams==2?16:8,parts=32/lanes;
    const int lane=threadIdx.x,local=lane%lanes;
    constexpr int Capacity=mfq::cuda::warp_sums::capacity(M);
    int token,owner;
    if(lanes==32){token=mfq::cuda::warp_sums::index<Capacity,32>(local);owner=mfq::cuda::warp_sums::owner<Capacity,32>(token);}
    else if(lanes==16){token=mfq::cuda::warp_sums::index<Capacity,16>(local);owner=mfq::cuda::warp_sums::owner<Capacity,16>(token);}
    else {token=mfq::cuda::warp_sums::index<Capacity,8>(local);owner=mfq::cuda::warp_sums::owner<Capacity,8>(token);}
    const auto width=std::int64_t(hidden)*streams;
    float mixed=0;
    for(int first=0;first<streams;first+=parts) {
        const int selected=first+lane/lanes;
        float partial[4][M]{};
        #pragma unroll
        for(int p=0;p<4;++p)if(p<parts)
            for(auto i=std::int64_t(local+p*lanes);i<std::int64_t(columns/F::qk)*F::parts;i+=32) {
                const int block=int(i/F::parts),part=F::step*int(i%F::parts);
                if(selected<streams) {
                    const auto w=F::load(packed+(std::int64_t(selected)*hidden+row)*stride,block,part);
                    #pragma unroll
                    for(int m=0;m<M;++m)partial[p][m]+=F::apply(w,input+std::int64_t(m)*(columns/32)+block*(F::qk/32));
                }
            }
        float values[Capacity]{};
        #pragma unroll
        for(int m=0;m<M;++m)values[m]=parts==1?partial[0][m]:parts==2?partial[0][m]+partial[1][m]:
            (partial[0][m]+partial[2][m])+(partial[1][m]+partial[3][m]);
        float reduced;
        if(lanes==32)reduced=mfq::cuda::warp_sums::reduce<Capacity,32>(values,local);
        else if(lanes==16)reduced=mfq::cuda::warp_sums::reduce<Capacity,16>(values,local);
        else reduced=mfq::cuda::warp_sums::reduce<Capacity,8>(values,local);
        for(int group=0;group<parts && first+group<streams;++group) {
            const float projection=__shfl_sync(0xffffffffu,reduced,group*lanes+owner);
            if(lane==owner && token<M) {
                const auto at=std::int64_t(token)*width+std::int64_t(first+group)*hidden+row;
                const float gate=1.f/(1.f+expf(-projection));
                mixed=__fadd_rn(mixed,__fmul_rn(gate,normalized[at]));
            }
        }
    }
    if(lane==owner && token<M)output[std::int64_t(token)*hidden+row]=mixed/streams;
}
}
void prepare_residual_activation(const float* down,const float* injection,
        std::int64_t tokens,std::int64_t columns,std::int64_t streams,
        void* packed,float* injection_output,cudaStream_t stream) {
    if(tokens<1 || columns<1 || columns%32 || streams<1 || streams>INT_MAX ||
            tokens>INT64_MAX/sizeof(float)/columns || tokens>INT64_MAX/sizeof(float)/streams ||
            !down || !injection || !packed || !injection_output)
        throw std::invalid_argument("GGUF residual activation geometry/storage differs");
    const auto count=tokens*columns;
    const auto blocks=unsigned(std::min<std::int64_t>(65535,(count-1)/256+1));
    prepare_activation<<<blocks,256,0,stream>>>(down,injection,count,tokens*streams,float(streams),
        static_cast<block_q8_1*>(packed),injection_output);
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_residual_activation",
        reinterpret_cast<const void*>(prepare_activation),"M="+std::to_string(tokens)+
        " columns="+std::to_string(columns)+" streams="+std::to_string(streams)+
        " bottleneck_q8=1 injection=1 launches=1");
}

bool residual_mix_supported(ggml_type type) noexcept {
    switch(type) {
#define MIX_SUPPORTED(name) case GGML_TYPE_##name:return true;
    TYLOINFER_GGUF_LEGACY_FORMATS(MIX_SUPPORTED)
    MIX_SUPPORTED(PTQ1_0)
#undef MIX_SUPPORTED
    default:return false;
    }
}
void residual_mix(ggml_type type,const void* packed,std::uint64_t stride,
        std::int64_t rows,std::int64_t columns,const void* input,std::int64_t tokens,
        const float* normalized,std::int64_t streams,float* output,cudaStream_t stream) {
    if(!residual_mix_supported(type) || tokens<1 || tokens>4 || streams<1 || streams>INT_MAX ||
            rows<streams || rows%streams || rows>INT_MAX || columns<1 || columns>INT_MAX ||
            columns%ggml_blck_size(type) || stride<mfq::gguf::row_bytes(type,columns) ||
            stride>UINT64_MAX/std::uint64_t(rows) || !packed || !input || !normalized || !output)
        throw std::invalid_argument("GGUF residual mix geometry/storage differs");
    const auto launch=[&]<int Format,int M>() {
        const auto function=project_residual_mix<Format,M>;
        const auto hidden=rows/streams;
        function<<<unsigned((hidden+3)/4),dim3(32,4),0,stream>>>(static_cast<const std::uint8_t*>(packed),stride,
            int(hidden),int(columns),int(streams),static_cast<const block_q8_1*>(input),normalized,output);
        const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_residual_mix",reinterpret_cast<const void*>(function),
            "format="+std::to_string(Format)+" M="+std::to_string(M)+" streams="+std::to_string(streams)+
            " rows="+std::to_string(rows)+" columns="+std::to_string(columns)+" original_weights=1");
    };
    const auto dispatch=[&]<int Format>() {
        if(tokens==1)launch.template operator()<Format,1>();else if(tokens==2)launch.template operator()<Format,2>();
        else if(tokens==3)launch.template operator()<Format,3>();else launch.template operator()<Format,4>();
    };
    switch(type) {
#define MIX_DISPATCH(name) case GGML_TYPE_##name:dispatch.template operator()<GGML_TYPE_##name>();break;
    TYLOINFER_GGUF_LEGACY_FORMATS(MIX_DISPATCH)
    MIX_DISPATCH(PTQ1_0)
#undef MIX_DISPATCH
    default:break;
    }
}
}
