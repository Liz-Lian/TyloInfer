#include "gguf_input.h"
#include "gguf.h"
#include "ggml-cuda/common.cuh"
#include "gguf_quantize.cuh"
#include "kernel_dispatch_trace.h"
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace mfq::cuda::gguf {
namespace {
template<class T,bool Contiguous> struct ActivationView {
    const T* input;
    std::int64_t columns,row_stride,column_stride;
    __device__ float4 operator()(std::int64_t i) const {
        const auto at=Contiguous?i:(i/columns)*row_stride+(i%columns)*column_stride;
        const auto stride=Contiguous?1:column_stride;
        return make_float4(float(input[at]),float(input[at+stride]),float(input[at+2*stride]),float(input[at+3*stride]));
    }
};
template<bool Half> struct HiddenView {
    const float *gate,*up; float limit;
    __device__ float4 operator()(std::int64_t i) const {
        return make_float4(swiglu_q8_value<Half>(gate[i],up[i],limit),swiglu_q8_value<Half>(gate[i+1],up[i+1],limit),
            swiglu_q8_value<Half>(gate[i+2],up[i+2],limit),swiglu_q8_value<Half>(gate[i+3],up[i+3],limit));
    }
};
template<class Read,bool Half,bool PublishCpu,int Block>
__global__ void rotate_quantize(Read read,std::int64_t count,std::int64_t columns,
        const std::int8_t* signs,int block,block_q8_1* output,float* mapped_cpu) {
    const int lane=threadIdx.x%32;
    for(auto tile=std::int64_t(blockIdx.x)*4+threadIdx.x/32;tile<(count+127)/128;
            tile+=std::int64_t(gridDim.x)*4) {
        const auto start=tile*128+lane*4;
        float v[4]{};
        if(start<count) {
            const auto values=read(start);
            v[0]=values.x;v[1]=values.y;v[2]=values.z;v[3]=values.w;
            if(signs) {
                const auto column=start%columns;
                #pragma unroll
                for(int j=0;j<4;++j)v[j]*=signs[column+j];
            }
        }
        // Small blocks stay within a lane; larger ones share the same warp
        // butterfly. Blocks divide the row width, so no transform crosses rows.
        if constexpr(Block>=2) {const float a=v[0],b=v[2];v[0]=a+v[1];v[1]=a-v[1];v[2]=b+v[3];v[3]=b-v[3];}
        if constexpr(Block>=4) {const float a=v[0],b=v[1];v[0]=a+v[2];v[2]=a-v[2];v[1]=b+v[3];v[3]=b-v[3];}
        #pragma unroll
        for(int mask=1;mask<Block/4;mask*=2) {
            #pragma unroll
            for(int j=0;j<4;++j) {
                const float other=__shfl_xor_sync(0xffffffffu,v[j],mask);
                v[j]=(lane&mask)?other-v[j]:v[j]+other;
            }
        }
        const float normalization=rsqrtf(float(block));
        #pragma unroll
        for(int j=0;j<4;++j) {
            v[j]*=normalization;
            if constexpr(Half)v[j]=__half2float(__float2half_rn(v[j]));
            if constexpr(PublishCpu)if(start+j<count)mapped_cpu[start+j]=v[j];
        }
        float sums[4]={v[0],v[1],v[2],v[3]};
        float maximum=fmaxf(fmaxf(fabsf(v[0]),fabsf(v[1])),fmaxf(fabsf(v[2]),fabsf(v[3])));
        // Reconstruct the existing 32-value Q8 sum tree exactly. Four values
        // per lane need eight-lane reductions before the component sums.
        for(int offset=4;offset;offset>>=1) {
            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset,8));
            #pragma unroll
            for(int j=0;j<4;++j)sums[j]+=__shfl_xor_sync(0xffffffffu,sums[j],offset,8);
        }
        const float scale=finite_half(maximum/127.f);
        std::uint32_t codes=0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const float q=maximum==0?0:roundf(v[j]/scale);
            codes|=std::uint32_t(std::uint8_t(std::int8_t(fminf(127.f,fmaxf(-127.f,q)))))<<(8*j);
        }
        if(start<count) {
            auto& packed=output[start/32];
            *reinterpret_cast<std::uint32_t*>(packed.qs+(lane%8)*4)=codes;
            if(!(lane%8))packed.ds=__floats2half2_rn(scale,finite_half((sums[0]+sums[2])+(sums[1]+sums[3])));
        }
    }
}
template<bool Half>
__global__ void materialize_hidden(const float* gate,const float* up,std::int64_t count,float limit,void* output) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::int64_t(blockDim.x)*gridDim.x)
        store_projection(output,i,swiglu_q8_value<Half>(gate[i],up[i],limit),Half);
}
bool rotation_geometry(std::int64_t rows,std::int64_t columns,int block) {
    if(rows<0 || columns<=0 || columns%32 || rows>INT64_MAX/columns || block<1 || (block&(block-1)) || columns%block)
        throw std::invalid_argument("GGUF fused Hadamard/Q8 geometry");
    return block<=128;
}
template<class Read,bool Half>
void launch_rotation(Read read,std::int64_t rows,std::int64_t columns,const std::int8_t* signs,
        int block,void* packed,float* mapped_cpu,cudaStream_t stream) {
    if(!rows)return;
    const auto count=rows*columns;
    const auto blocks=unsigned(std::min<std::int64_t>(65535,(count+511)/512));
    const auto launch_block=[&]<int Block>() {
        const auto launch=[&]<bool Cpu>() {
            rotate_quantize<Read,Half,Cpu,Block><<<blocks,128,0,stream>>>(read,count,columns,signs,block,static_cast<block_q8_1*>(packed),mapped_cpu);
            if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_rotation_q8",
                reinterpret_cast<const void*>(rotate_quantize<Read,Half,Cpu,Block>),
                "block="+std::to_string(block)+" rows="+std::to_string(rows)+" cpu="+std::to_string(Cpu));
        };
        if(mapped_cpu)launch.template operator()<true>();else launch.template operator()<false>();
    };
    // The actual transform block bounds the dependency chain; compile its
    // butterfly stages directly, independently of model or device geometry.
    switch(block) {
#define TYLOINFER_ROTATION_BLOCK(B) case B:launch_block.template operator()<B>();break
        TYLOINFER_ROTATION_BLOCK(1);TYLOINFER_ROTATION_BLOCK(2);TYLOINFER_ROTATION_BLOCK(4);TYLOINFER_ROTATION_BLOCK(8);
        TYLOINFER_ROTATION_BLOCK(16);TYLOINFER_ROTATION_BLOCK(32);TYLOINFER_ROTATION_BLOCK(64);TYLOINFER_ROTATION_BLOCK(128);
#undef TYLOINFER_ROTATION_BLOCK
        default:throw std::invalid_argument("GGUF register Hadamard block");
    }
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void prepare_resident(const std::uint64_t* mapped,std::uint64_t* device,
    const std::int64_t* ids,std::int64_t slots,std::int64_t experts,int capacity,
    RouteGroup* groups,const std::uint32_t* admission_ready,std::uint64_t* start_clock) {
    if(admission_ready) {
        if(!threadIdx.x) {
            const auto* ready=reinterpret_cast<const volatile std::uint32_t*>(admission_ready);
            while(!*ready) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
            __threadfence_system();
        }
        __syncthreads();
    }
    const auto slot=std::int64_t(threadIdx.x);
    const auto expert=slot<slots?ids[slot]:-1;
    int previous=0;
    if(slot<slots)for(std::int64_t i=0;i<slot;++i)previous+=ids[i]==expert;
    if(!previous && expert>=0 && expert<experts)
        for(int role=0;role<3;++role)device[role*experts+expert]=mapped[role*experts+expert];
    __syncthreads();
    if(!threadIdx.x) {
        // Keep the CPU/GPU balance clock after the snapshot and before group
        // construction, as in the separate snapshot/stamp/group sequence.
        if(start_clock) {
            unsigned long long now=0;
            asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
            *reinterpret_cast<volatile std::uint64_t*>(start_clock)=now;
        }
        if(groups)groups[0].count=0;
    }
    if(!groups)return;
    __syncthreads();
    if(slot>=slots || previous%capacity)return;
    auto& group=groups[1+atomicAdd(&groups[0].count,1)];
    group.expert=expert;group.count=0;
    for(auto i=slot;i<slots && group.count<capacity;++i)
        if(ids[i]==expert)group.slots[group.count++]=int(i);
}

template<typename Input, bool PublishCpu, bool Contiguous>
__global__ void prepare_input(const Input* input, std::int64_t count,
    std::int64_t columns, std::int64_t row_stride, std::int64_t column_stride,
    block_q8_1* packed, float* mapped_cpu) {
    // Each loop iteration contains complete Q8 warps, including a short CTA tail.
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;
        i+=std::int64_t(blockDim.x)*gridDim.x) {
        const auto at=Contiguous?i:(i/columns)*row_stride+(i%columns)*column_stride;
        const float value=float(input[at]);
        store_q8(value,i,packed);
        if constexpr(PublishCpu)mapped_cpu[i]=value;
    }
    // The following stream-ordered route-publication kernel provides the
    // system fence and ready flag, matching the former copy_cpu_input path.
}
}

bool try_prepare_resident_routes(const std::uint64_t* mapped,std::uint64_t* device,
    const std::int64_t* ids,std::int64_t slots,std::int64_t experts,int capacity,
    RouteGroup* groups,const std::uint32_t* admission_ready,std::uint64_t* start_clock,
    cudaStream_t stream) {
    if(slots<0 || experts<0 || (groups && (capacity<=0 || capacity>int(sizeof(RouteGroup::slots)/sizeof(std::int32_t)))))
        throw std::invalid_argument("GGUF resident preparation geometry");
    cudaFuncAttributes attributes{};
    const auto status=cudaFuncGetAttributes(&attributes,prepare_resident);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(slots>attributes.maxThreadsPerBlock)return false;
    const auto threads=unsigned(std::max<std::int64_t>(32,((slots+31)/32)*32));
    if(threads>unsigned(attributes.maxThreadsPerBlock))return false;
    prepare_resident<<<1,threads,0,stream>>>(mapped,device,ids,slots,experts,capacity,groups,admission_ready,start_clock);
    const auto launched=cudaGetLastError();
    if(launched!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(launched));
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
        "gguf_resident_prepare",reinterpret_cast<const void*>(prepare_resident),
        "slots="+std::to_string(slots)+" groups="+std::to_string(groups!=nullptr)+
        " clock="+std::to_string(start_clock!=nullptr)+" layer_admission_guard="+std::to_string(admission_ready!=nullptr));
    return true;
}

void prepare_moe_input(const void* input, InputType type, std::int64_t tokens,
    std::int64_t columns, std::int64_t row_stride, std::int64_t column_stride,
    void* packed, float* mapped_cpu, cudaStream_t stream) {
    if(tokens<0 || columns<=0 || columns%32 || row_stride<0 || column_stride<0 ||
        tokens>INT64_MAX/columns)
        throw std::invalid_argument("GGUF MoE input view geometry");
    if(!tokens)return;
    const auto count=tokens*columns;
    constexpr int threads=256;
    const auto blocks=unsigned(std::min<std::int64_t>(65535,(count-1)/threads+1));
    const auto launch_view=[&]<bool Contiguous>(auto* source) {
        if(mapped_cpu)prepare_input<std::remove_pointer_t<decltype(source)>,true,Contiguous>
            <<<blocks,threads,0,stream>>>(source,count,columns,row_stride,column_stride,
                static_cast<block_q8_1*>(packed),mapped_cpu);
        else prepare_input<std::remove_pointer_t<decltype(source)>,false,Contiguous>
            <<<blocks,threads,0,stream>>>(source,count,columns,row_stride,column_stride,
                static_cast<block_q8_1*>(packed),nullptr);
    };
    const auto launch=[&](auto* source) {
        if(row_stride==columns && column_stride==1)launch_view.template operator()<true>(source);
        else launch_view.template operator()<false>(source);
    };
    switch(type) {
        case InputType::Float32: launch(static_cast<const float*>(input));break;
        case InputType::Float16: launch(static_cast<const half*>(input));break;
        case InputType::BFloat16: launch(static_cast<const __nv_bfloat16*>(input));break;
        default: throw std::invalid_argument("GGUF MoE input type");
    }
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
bool try_prepare_rotated_moe_input(const void* input,InputType type,std::int64_t rows,
        std::int64_t columns,std::int64_t row_stride,std::int64_t column_stride,
        const std::int8_t* signs,int block,void* packed,float* mapped_cpu,cudaStream_t stream) {
    if(!rotation_geometry(rows,columns,block))return false;
    if(row_stride<0 || column_stride<0)throw std::invalid_argument("GGUF rotated input strides");
    const auto launch_view=[&]<class T,bool Half>(const T* data) {
        if(row_stride==columns && column_stride==1)
            launch_rotation<ActivationView<T,true>,Half>({data,columns,row_stride,column_stride},rows,columns,signs,block,packed,mapped_cpu,stream);
        else launch_rotation<ActivationView<T,false>,Half>({data,columns,row_stride,column_stride},rows,columns,signs,block,packed,mapped_cpu,stream);
    };
    switch(type) {
        case InputType::Float16:launch_view.template operator()<half,true>(static_cast<const half*>(input));break;
        case InputType::Float32:launch_view.template operator()<float,false>(static_cast<const float*>(input));break;
        case InputType::BFloat16:launch_view.template operator()<__nv_bfloat16,false>(static_cast<const __nv_bfloat16*>(input));break;
        default:throw std::invalid_argument("GGUF rotated input type");
    }
    return true;
}
bool try_quantize_rotated_moe_hidden(const float* gate,const float* up,std::int64_t rows,
        std::int64_t columns,bool half,float limit,const std::int8_t* signs,int block,void* packed,cudaStream_t stream) {
    if(!rotation_geometry(rows,columns,block))return false;
    if(half)launch_rotation<HiddenView<true>,true>({gate,up,limit},rows,columns,signs,block,packed,nullptr,stream);
    else launch_rotation<HiddenView<false>,false>({gate,up,limit},rows,columns,signs,block,packed,nullptr,stream);
    return true;
}
void materialize_moe_hidden(const float* gate,const float* up,std::int64_t rows,
        std::int64_t columns,bool half,float limit,void* output,cudaStream_t stream) {
    if(rows<0 || columns<=0 || rows>INT64_MAX/columns)throw std::invalid_argument("GGUF hidden geometry");
    if(!rows)return;
    const auto count=rows*columns,blocks=std::min<std::int64_t>(65535,(count+255)/256);
    if(half)materialize_hidden<true><<<unsigned(blocks),256,0,stream>>>(gate,up,count,limit,output);
    else materialize_hidden<false><<<unsigned(blocks),256,0,stream>>>(gate,up,count,limit,output);
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
