#include "gguf.h"
#include "gguf_prefill_scale.cuh"
#include <unordered_map>
#include "ggml-cuda/common.cuh"
#include "gguf_quantize.cuh"
#include "gguf_head_gate.h"
#include "kernel_dispatch_trace.h"
#include <climits>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include "reduce.cuh"
#include <cuda_bf16.h>

namespace mfq::cuda::gguf {
namespace {
#include "gguf_prefill_rms.cuh"

struct RmsPrefillLimits {int processors=0,dynamic256=0,dynamic1024=0;};
const RmsPrefillLimits& rms_prefill_limits(int device) {
    static thread_local std::unordered_map<int,RmsPrefillLimits> limits;
    auto found=limits.find(device);
    if(found==limits.end()) {
        auto checked=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
        int shared=0;RmsPrefillLimits value;
        checked(cudaDeviceGetAttribute(&shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
        checked(cudaDeviceGetAttribute(&value.processors,cudaDevAttrMultiProcessorCount,device));
        const auto prepare=[&]<int Threads>() {
            int limit=shared;
            const auto typed=[&]<class Input>() {
                cudaFuncAttributes attr{};checked(cudaFuncGetAttributes(&attr,rms_prefill_kernel<Input,Threads>));
                const int dynamic=std::max(0,shared-int(attr.sharedSizeBytes));
                checked(cudaFuncSetAttribute(rms_prefill_kernel<Input,Threads>,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic));
                limit=std::min(limit,dynamic);
            };
            typed.template operator()<float>();typed.template operator()<half>();typed.template operator()<__nv_bfloat16>();
            return limit;
        };
        value.dynamic256=prepare.template operator()<256>();value.dynamic1024=prepare.template operator()<1024>();
        found=limits.emplace(device,value).first;
    }
    return found->second;
}

template<bool CacheValues,bool ScaledGate=false>
__global__ void rms_gated_prefill_kernel(const float* input,const void* gate,const float* norm,
        int64_t heads,int64_t tokens,int width,int64_t gb,int64_t gt,float eps,bool silu,half* output,int32_t* exponents,
        const int32_t* gate_rows=nullptr,const int32_t* gate_columns=nullptr) {
    const int64_t row=blockIdx.x,b=row/tokens,t=row%tokens,columns=heads*width;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    extern __shared__ float values[];
    __shared__ float maxima[8];__shared__ int exponent;
    float maximum=0;
    for(int64_t h=warp;h<heads;h+=8) {
        const auto* x=input+((b*heads+h)*tokens+t)*width;
        // Match the ordinary 256-thread RMS reduction, including its FMA
        // and warp ordering, without materializing the normalized tensor.
        float cache[8];
        if constexpr(CacheValues) {
            #pragma unroll
            for(int i=0;i<8;++i)cache[i]=i*32+lane<width?x[i*32+lane]:0.f;
        }
        float sum=0;
        #pragma unroll
        for(int original_warp=0;original_warp<8;++original_warp) {
            if(original_warp*32>=width)continue;
            float partial=0;
            if constexpr(CacheValues)partial=__fmul_rn(cache[original_warp],cache[original_warp]);
            else for(int d=original_warp*32+lane;d<width;d+=256)partial=fmaf(x[d],x[d],partial);
            for(int step=16;step;step>>=1)partial=__fadd_rn(partial,__shfl_xor_sync(0xffffffffu,partial,step));
            if(lane==original_warp)sum=partial;
        }
        for(int step=16;step;step>>=1)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,step));
        const float inverse=rsqrtf(sum/width+eps);
        const auto emit=[&](int d,float input_value) {
            const auto column=h*width+d;
            float g;
            if constexpr(ScaledGate)g=prefill_ldexp(__half2float(static_cast<const half*>(gate)[b*gb+t*gt+column]),
                gate_rows[row]+gate_columns[column]);
            else g=static_cast<const float*>(gate)[b*gb+t*gt+column];
            const float normalized=__fmul_rn(__fmul_rn(input_value,inverse),norm[d]);
            const float value=__fmul_rn(normalized,silu?g/(1.f+expf(-g)):1.f/(1.f+expf(-g)));
            values[column]=value;maximum=fmaxf(maximum,fabsf(value));
        };
        if constexpr(CacheValues) {
            #pragma unroll
            for(int i=0;i<8;++i)if(i*32+lane<width)emit(i*32+lane,cache[i]);
        }else for(int d=lane;d<width;d+=32)emit(d,x[d]);
    }
    for(int step=16;step;step>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,step));
    if(!lane)maxima[warp]=maximum;
    __syncthreads();
    if(!warp) {
        maximum=lane<8?maxima[lane]:0;
        for(int step=16;step;step>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,step));
        if(!lane){exponent=prefill_unit_exponent(maximum,columns);exponents[row]=exponent;}
    }
    __syncthreads();
    for(auto d=int64_t(threadIdx.x);d<columns;d+=blockDim.x)
        output[row*columns+d]=__float2half_rn(prefill_ldexp(values[d],-exponent));
}
template<class Input>
__global__ void materialized_gate(const Input* up,const Input* gate,Input* output,
        std::int64_t count,bool silu) {
    for(auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::int64_t(gridDim.x)*blockDim.x) {
        const float g=float(gate[i]),u=float(up[i]);
        float activation=silu?g/(1.f+expf(-g)):1.f/(1.f+expf(-g));
        if constexpr(std::is_same_v<Input,half>)activation=__half2float(__float2half_rn(activation));
        output[i]=Input(__fmul_rn(activation,u));
    }
}

template<class Input,bool Aligned=false,bool PositionMajor=false>
__global__ void gated_input_q8(const Input* up,const Input* gate,std::int64_t count,
        bool silu,block_q8_1* output,int width,int tokens=1) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const float g=float(gate[i]),u=float(up[i]);
    float activation=silu?g/(1.f+expf(-g)):1.f/(1.f+expf(-g));
    // Preserve the standalone activation and multiply's dtype boundaries.
    if constexpr(std::is_same_v<Input,half>)activation=__half2float(__float2half_rn(activation));
    float value=__fmul_rn(activation,u);
    if constexpr(std::is_same_v<Input,half>)value=__half2float(__float2half_rn(value));
    if constexpr(PositionMajor)store_q8_position_major(value,i,width,tokens,output);
    else if constexpr(Aligned)store_q8_m1_aligned(value,i,width,output);else store_q8(value,i,output);
}

template<class Input,bool Aligned,bool SingleRow,bool PositionMajor=false>
__global__ void gated_head_input_q8(const Input* input,const Input* gate,int64_t count,
        int64_t tokens,int heads,int width,int64_t batch_stride,
        int64_t token_stride,int64_t head_stride,bool silu,block_q8_1* output) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const int64_t row=SingleRow?0:i/(heads*width);
    const int column=SingleRow?int(i):int(i%(heads*width));
    const auto gate_index=(row/tokens)*batch_stride+(row%tokens)*token_stride+
        (column/width)*head_stride+column%width;
    const float g=float(gate[gate_index]);
    float activation=silu?g/(1.f+expf(-g)):1.f/(1.f+expf(-g));
    if constexpr(std::is_same_v<Input,half>)activation=__half2float(__float2half_rn(activation));
    float value=__fmul_rn(activation,float(input[i]));
    if constexpr(std::is_same_v<Input,half>)value=__half2float(__float2half_rn(value));
    if constexpr(PositionMajor)store_q8_position_major(value,i,int(heads*width),int(count/(heads*width)),output);
    else if constexpr(Aligned)store_q8_m1_aligned(value,i,int(heads*width),output);
    else store_q8(value,i,output);
}

__device__ __forceinline__ float rms_warp_sum(float value) {
    #pragma unroll
    for(int offset=16;offset;offset>>=1)value+=__shfl_xor_sync(0xffffffffu,value,offset);
    return value;
}

template<class Input,int Threads,bool Aligned=false,bool PositionMajor=false>
__global__ void rms_input_q8(const Input* __restrict__ input,const float* __restrict__ weight,
        int width,float eps,float offset,float* __restrict__ normalized,block_q8_1* __restrict__ output,
        int rows=1,block_q8_1* native=nullptr) {
    const auto base=std::int64_t(blockIdx.y)*width;
    float square_sum=0.f;
    for(int i=threadIdx.x;i<width;i+=Threads) {
        const float value=float(input[base+i]);square_sum+=value*value;
    }
    square_sum=block_sum<Threads/32>(square_sum);
    const float inverse=rsqrtf(square_sum/float(width)+eps);
    for(int i=int(blockIdx.x)*256+threadIdx.x;threadIdx.x<256 && i<width;i+=int(gridDim.x)*256) {
        const float value=float(input[base+i])*inverse*(weight[i]+offset);
        normalized[base+i]=value;
        if constexpr(PositionMajor)store_q8_position_major(value,base+i,width,rows,output,native);
        else if constexpr(Aligned)store_q8_m1_aligned(value,base+i,width,output);else store_q8(value,base+i,output);
    }
}

template<class Added,int Threads,bool Aligned=false,bool PositionMajor=false,bool PairReduction=true>
__global__ void acc_rms_input_q8(const half2* __restrict__ residual,const Added* __restrict__ projection,
        const float* __restrict__ weight,int width,float eps,float offset,half* __restrict__ summed,
        half* __restrict__ normalized,block_q8_1* __restrict__ output,int rows=1,block_q8_1* native=nullptr) {
    const auto base=std::int64_t(blockIdx.y)*width;
    const auto value_at=[&](int pair) {
        half2 b;
        if constexpr(std::is_same_v<Added,float2>) {
            const auto v=projection[base/2+pair];b=__floats2half2_rn(v.x,v.y);
        }else b=projection[base/2+pair];
        return __hadd2(residual[base/2+pair],b);
    };
    float square_sum=0.f;
    if constexpr(PairReduction) {
        for(int pair=threadIdx.x;pair<width/2;pair+=Threads) {
            const auto v=__half22float2(value_at(pair));square_sum+=v.x*v.x+v.y*v.y;
        }
    }else for(int i=threadIdx.x;i<width;i+=Threads) {
        const auto pair=value_at(i/2);const auto value=__half2float(i%2?__high2half(pair):__low2half(pair));
        square_sum+=value*value;
    }
    square_sum=block_sum<Threads/32>(square_sum);
    const float inverse=rsqrtf(square_sum/float(width)+eps);
    for(int i=int(blockIdx.x)*256+threadIdx.x;threadIdx.x<256 && i<width;i+=int(gridDim.x)*256) {
        const auto pair=value_at(i/2);const half sum=i%2?__high2half(pair):__low2half(pair);
        const half norm=__float2half_rn(__half2float(sum)*inverse*(weight[i]+offset));
        summed[base+i]=sum;normalized[base+i]=norm;
        if constexpr(PositionMajor)store_q8_position_major(__half2float(norm),base+i,width,rows,output,native);
        else if constexpr(Aligned)store_q8_m1_aligned(__half2float(norm),base+i,width,output);
        else store_q8(__half2float(norm),base+i,output);
    }
}

template<bool PositionMajor=false>
__global__ void rms_gated_input_q8(const float* input,const float* gate,const float* weight,
        int heads,int tokens,int width,std::int64_t batch_stride,std::int64_t token_stride,
        float eps,bool silu,block_q8_1* output,int batch_count=1) {
    const std::int64_t row=blockIdx.x,base=row*width;
    const auto batch=row/(std::int64_t(heads)*tokens),head=(row/tokens)%heads,token=row%tokens;
    const auto destination=((batch*tokens+token)*heads+head)*width;
    float square_sum=0.f;
    for(int i=threadIdx.x;i<width;i+=256) {
        const float value=input[base+i];square_sum+=value*value;
    }
    square_sum=rms_warp_sum(square_sum);
    __shared__ float partial[8];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(!lane)partial[warp]=square_sum;
    __syncthreads();
    if(!warp) {
        square_sum=lane<8?partial[lane]:0.f;
        square_sum=rms_warp_sum(square_sum);
        if(!lane)partial[0]=square_sum;
    }
    __syncthreads();
    const float inverse=rsqrtf(partial[0]/float(width)+eps);
    for(int i=threadIdx.x;i<width;i+=256) {
        const float normalized=__fmul_rn(__fmul_rn(input[base+i],inverse),weight[i]);
        const float z=gate[batch*batch_stride+token*token_stride+head*width+i];
        const float activation=silu?z/(1.f+expf(-z)):1.f/(1.f+expf(-z));
        if constexpr(PositionMajor)store_q8_position_major(__fmul_rn(normalized,activation),destination+i,
            int64_t(heads)*width,batch_count*tokens,output);
        else store_q8(__fmul_rn(normalized,activation),destination+i,output);
    }
}
}
template<bool Aligned>
static void quantize_gated_input_impl(const void* up,const void* gate,bool half,std::int64_t tokens,
        std::int64_t columns,bool silu,void* output,cudaStream_t stream) {
    if constexpr(Aligned)if(tokens!=1 || !output || (reinterpret_cast<std::uintptr_t>(output)&15))throw std::invalid_argument("Aligned Q8 requires one token and aligned output");
    q8_row_bytes(columns);
    if(tokens<0 || tokens>INT64_MAX/columns || std::uint64_t(tokens)*columns>std::uint64_t(INT_MAX)*256)
        throw std::invalid_argument("GGUF gated activation geometry exceeds the grid");
    if(!tokens)return;
    const auto blocks=unsigned((tokens*columns+255)/256);
    if(half)gated_input_q8<::half,Aligned><<<blocks,256,0,stream>>>(static_cast<const ::half*>(up),static_cast<const ::half*>(gate),tokens*columns,silu,static_cast<block_q8_1*>(output),int(columns));
    else gated_input_q8<float,Aligned><<<blocks,256,0,stream>>>(static_cast<const float*>(up),static_cast<const float*>(gate),tokens*columns,silu,static_cast<block_q8_1*>(output),int(columns));
    const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void quantize_gated_input(const void* up,const void* gate,bool half,std::int64_t tokens,
        std::int64_t columns,bool silu,void* output,cudaStream_t stream) {quantize_gated_input_impl<false>(up,gate,half,tokens,columns,silu,output,stream);}
void quantize_gated_input_m1_aligned(const void* up,const void* gate,bool half,std::int64_t tokens,
        std::int64_t columns,bool silu,void* output,cudaStream_t stream) {quantize_gated_input_impl<true>(up,gate,half,tokens,columns,silu,output,stream);}

void quantize_gated_input_interleaved(const void* up,const void* gate,bool half_input,int64_t tokens,
        int64_t columns,bool silu,void* output,cudaStream_t stream) {
    interleaved_activation_bytes(tokens,columns);
    if(!up || !gate || !output)throw std::invalid_argument("GGUF interleaved gated activation storage missing");
    const auto launch=[&]<class Input>() {
        const auto kernel=gated_input_q8<Input,false,true>;
        kernel<<<unsigned((tokens*columns+255)/256),256,0,stream>>>(static_cast<const Input*>(up),
            static_cast<const Input*>(gate),tokens*columns,silu,static_cast<block_q8_1*>(output),int(columns),int(tokens));
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_interleaved_gated_q8",
            reinterpret_cast<const void*>(kernel),"M="+std::to_string(tokens)+" materialized_activation=0");
    };
    if(half_input)launch.template operator()<half>();else launch.template operator()<float>();
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}

template<bool PositionMajor>
static void quantize_gated_head_input_impl(const void* input,const void* gate,bool half_input,
        int64_t batch,int64_t tokens,int64_t heads,int64_t width,
        int64_t batch_stride,int64_t token_stride,int64_t head_stride,
        bool silu,bool aligned_m1,void* output,cudaStream_t stream) {
    if(batch<1 || tokens<1 || heads<1 || width<1 || heads>INT_MAX/width ||
        tokens>INT64_MAX/batch || batch*tokens>INT64_MAX/(heads*width) ||
        uint64_t(batch*tokens*heads*width)>uint64_t(INT_MAX)*256 ||
        head_stride<width || token_stride<(heads-1)*head_stride+width ||
        batch_stride<(tokens-1)*token_stride+(heads-1)*head_stride+width ||
        !input || !gate || !output)
        throw std::invalid_argument("GGUF head-gated activation geometry/storage differs");
    q8_row_bytes(heads*width);
    if constexpr(PositionMajor)interleaved_activation_bytes(batch*tokens,heads*width);
    if(aligned_m1 && (batch*tokens!=1 || (reinterpret_cast<uintptr_t>(output)&15)))
        throw std::invalid_argument("Aligned head-gated activation requires one token and aligned output");
    const auto count=batch*tokens*heads*width;
    const auto launch=[&]<class Input,bool Aligned>() {
        const auto emit=[&]<bool SingleRow>() {
        const auto kernel=gated_head_input_q8<Input,Aligned,SingleRow,PositionMajor>;
        kernel<<<unsigned((count+255)/256),256,0,stream>>>(static_cast<const Input*>(input),
            static_cast<const Input*>(gate),count,tokens,int(heads),int(width),batch_stride,token_stride,head_stride,
            silu,static_cast<block_q8_1*>(output));
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_head_gated_q8",
            reinterpret_cast<const void*>(kernel),"strided heads, aligned_m1="+std::to_string(Aligned)+
                " position_major="+std::to_string(PositionMajor));
        };
        if constexpr(PositionMajor)emit.template operator()<false>();
        else if constexpr(Aligned)emit.template operator()<true>();
        else if(batch*tokens==1)emit.template operator()<true>();
        else emit.template operator()<false>();
    };
    if constexpr(PositionMajor) {
        if(half_input)launch.template operator()<half,false>();else launch.template operator()<float,false>();
    }else if(half_input) {
        if(aligned_m1)launch.template operator()<half,true>();
        else launch.template operator()<half,false>();
    }else {
        if(aligned_m1)launch.template operator()<float,true>();
        else launch.template operator()<float,false>();
    }
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}

void quantize_gated_head_input(const void* input,const void* gate,bool half_input,
        int64_t batch,int64_t tokens,int64_t heads,int64_t width,
        int64_t batch_stride,int64_t token_stride,int64_t head_stride,
        bool silu,bool aligned_m1,void* output,cudaStream_t stream) {
    quantize_gated_head_input_impl<false>(input,gate,half_input,batch,tokens,heads,width,
        batch_stride,token_stride,head_stride,silu,aligned_m1,output,stream);
}
void quantize_gated_head_input_interleaved(const void* input,const void* gate,bool half_input,
        int64_t batch,int64_t tokens,int64_t heads,int64_t width,
        int64_t batch_stride,int64_t token_stride,int64_t head_stride,
        bool silu,void* output,cudaStream_t stream) {
    quantize_gated_head_input_impl<true>(input,gate,half_input,batch,tokens,heads,width,
        batch_stride,token_stride,head_stride,silu,false,output,stream);
}

template<bool Aligned,bool PositionMajor=false>
static void quantize_rms_input_impl(const void* input,bool half_input,const float* weight,
        std::int64_t rows,std::int64_t width,float eps,float offset,float* normalized,
        void* output,cudaStream_t stream,void* native=nullptr) {
    if constexpr(Aligned)if(rows!=1 || !output || (reinterpret_cast<std::uintptr_t>(output)&15))throw std::invalid_argument("Aligned Q8 requires one token and aligned output");
    q8_row_bytes(width);
    if(rows<1 || rows>INT_MAX || width>INT_MAX || rows>INT64_MAX/width)
        throw std::invalid_argument("GGUF RMS activation geometry exceeds the grid");
    if constexpr(PositionMajor) {
        interleaved_activation_bytes(rows,width);
        if(!input || !weight || !normalized || !output || input==normalized ||
                normalized==output || (native && native==output))
            throw std::invalid_argument("GGUF interleaved RMS storage differs");
    }
    int device=0,processors=0;
    auto checked=[](cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));};
    checked(cudaGetDevice(&device));checked(cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,device));
    // Replicate the small input reduction across otherwise idle SMs. Each
    // replica writes disjoint Q8 blocks, reducing the serial packing loop.
    const int replicas=std::min<int64_t>((width+255)/256,std::max<int64_t>(1,processors/rows));
    const auto launch=[&]<class Input,int Threads>() {
        const auto kernel=rms_input_q8<Input,Threads,Aligned,PositionMajor>;
        kernel<<<dim3(replicas,unsigned(rows)),Threads,0,stream>>>(static_cast<const Input*>(input),weight,
            int(width),eps,offset,normalized,static_cast<block_q8_1*>(output),int(rows),static_cast<block_q8_1*>(native));
        if constexpr(PositionMajor)if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_interleaved_rms_q8",
            reinterpret_cast<const void*>(kernel),"M="+std::to_string(rows)+" native_packet="+std::to_string(native!=nullptr));
    };
    // The larger block shortens the row's serial chain; narrow rows keep a
    // small block. No weight layout or device-model specialization is needed.
    if(half_input) {
        if(width>=2*1024)launch.template operator()<half,1024>();
        else launch.template operator()<half,256>();
    }else {
        if(width>=2*1024)launch.template operator()<float,1024>();
        else launch.template operator()<float,256>();
    }
    const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void quantize_rms_input(const void* input,bool half_input,const float* weight,
        std::int64_t rows,std::int64_t width,float eps,float offset,float* normalized,
        void* output,cudaStream_t stream) {quantize_rms_input_impl<false>(input,half_input,weight,rows,width,eps,offset,normalized,output,stream);}
void quantize_rms_input_m1_aligned(const void* input,bool half_input,const float* weight,
        std::int64_t rows,std::int64_t width,float eps,float offset,float* normalized,
        void* output,cudaStream_t stream) {quantize_rms_input_impl<true>(input,half_input,weight,rows,width,eps,offset,normalized,output,stream);}
void quantize_rms_input_interleaved(const void* input,bool half_input,const float* weight,
        int64_t rows,int64_t width,float eps,float offset,float* normalized,
        void* output,void* native,cudaStream_t stream) {
    quantize_rms_input_impl<false,true>(input,half_input,weight,rows,width,eps,offset,normalized,output,stream,native);
}

template<bool Aligned,bool PositionMajor=false>
static void quantize_acc_rms_input_impl(const void* residual,const void* projection,bool projection_float,
        const float* weight,std::int64_t rows,std::int64_t width,float eps,float offset,
        void* sum,void* normalized,void* output,cudaStream_t stream,void* native=nullptr) {
    if constexpr(Aligned)if(rows!=1 || !output || (reinterpret_cast<std::uintptr_t>(output)&15))throw std::invalid_argument("Aligned Q8 requires one token and aligned output");
    q8_row_bytes(width);
    if(rows<1 || rows>INT_MAX || width>INT_MAX || rows>INT64_MAX/width)
        throw std::invalid_argument("GGUF residual RMS activation geometry exceeds the grid");
    if constexpr(PositionMajor) {
        interleaved_activation_bytes(rows,width);
        if(!residual || !projection || !weight || !sum || !normalized || !output ||
                residual==sum || projection==sum || residual==normalized || projection==normalized ||
                sum==normalized || sum==output || normalized==output || (native && native==output) ||
                (reinterpret_cast<uintptr_t>(residual)&3) ||
                (reinterpret_cast<uintptr_t>(projection)&(projection_float?7:3)))
            throw std::invalid_argument("GGUF interleaved residual RMS storage differs");
    }
    auto checked=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
    int device=0,processors=0;checked(cudaGetDevice(&device));
    checked(cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,device));
    const int replicas=std::min<int64_t>((width+255)/256,std::max<int64_t>(1,processors/rows));
    const auto launch=[&]<class Added,int Threads,bool PairReduction=true>() {
        const auto kernel=acc_rms_input_q8<Added,Threads,Aligned,PositionMajor,PairReduction>;
        kernel<<<dim3(replicas,unsigned(rows)),Threads,0,stream>>>(
            static_cast<const half2*>(residual),static_cast<const Added*>(projection),weight,
            int(width),eps,offset,static_cast<half*>(sum),static_cast<half*>(normalized),static_cast<block_q8_1*>(output),
            int(rows),static_cast<block_q8_1*>(native));
        if constexpr(PositionMajor)if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_interleaved_acc_rms_q8",
            reinterpret_cast<const void*>(kernel),"M="+std::to_string(rows)+" native_packet="+std::to_string(native!=nullptr));
    };
    const auto typed=[&]<class Added>() {
        // Packet layout and token count must preserve the M1 RMS addition
        // tree; changing it can change Q8 rounding and the selected token.
        if(width>=4*1024)launch.template operator()<Added,1024>();
        else launch.template operator()<Added,256>();
    };
    if(projection_float)typed.template operator()<float2>();else typed.template operator()<half2>();
    checked(cudaGetLastError());
}
void quantize_acc_rms_input(const void* residual,const void* projection,bool projection_float,
        const float* weight,std::int64_t rows,std::int64_t width,float eps,float offset,
        void* sum,void* normalized,void* output,cudaStream_t stream) {quantize_acc_rms_input_impl<false>(residual,projection,projection_float,weight,rows,width,eps,offset,sum,normalized,output,stream);}
void quantize_acc_rms_input_m1_aligned(const void* residual,const void* projection,bool projection_float,
        const float* weight,std::int64_t rows,std::int64_t width,float eps,float offset,
        void* sum,void* normalized,void* output,cudaStream_t stream) {quantize_acc_rms_input_impl<true>(residual,projection,projection_float,weight,rows,width,eps,offset,sum,normalized,output,stream);}
void quantize_acc_rms_input_interleaved(const void* residual,const void* projection,bool projection_float,
        const float* weight,int64_t rows,int64_t width,float eps,float offset,
        void* sum,void* normalized,void* output,void* native,cudaStream_t stream) {
    quantize_acc_rms_input_impl<false,true>(residual,projection,projection_float,weight,rows,width,eps,offset,
        sum,normalized,output,stream,native);
}

template<bool PositionMajor>
static void quantize_rms_gated_input_impl(const float* input,const float* gate,const float* weight,
        std::int64_t batch,std::int64_t heads,std::int64_t tokens,std::int64_t width,
        std::int64_t gate_batch_stride,std::int64_t gate_token_stride,float eps,bool silu,
        void* output,cudaStream_t stream) {
    q8_row_bytes(width);
    if(batch<1 || heads<1 || tokens<1 || width>INT_MAX || heads>INT_MAX || tokens>INT_MAX ||
            batch>INT_MAX/heads || batch*heads>INT_MAX/tokens ||
            heads>INT64_MAX/width || gate_token_stride<heads*width ||
            gate_token_stride>INT64_MAX/tokens || gate_batch_stride<tokens*gate_token_stride ||
            !input || !gate || !weight || !output)
        throw std::invalid_argument("GGUF RMS gated activation geometry exceeds the grid");
    if constexpr(PositionMajor)interleaved_activation_bytes(batch*tokens,heads*width);
    const auto kernel=rms_gated_input_q8<PositionMajor>;
    kernel<<<unsigned(batch*heads*tokens),256,0,stream>>>(input,gate,weight,
        int(heads),int(tokens),int(width),gate_batch_stride,gate_token_stride,eps,silu,
        static_cast<block_q8_1*>(output),int(batch));
    if constexpr(PositionMajor)if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_interleaved_rms_gated_q8",
        reinterpret_cast<const void*>(kernel),"M="+std::to_string(batch*tokens)+" materialized_normalized_activation=0");
    const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void quantize_rms_gated_input(const float* input,const float* gate,const float* weight,
        int64_t batch,int64_t heads,int64_t tokens,int64_t width,
        int64_t batch_stride,int64_t token_stride,float eps,bool silu,void* output,cudaStream_t stream) {
    quantize_rms_gated_input_impl<false>(input,gate,weight,batch,heads,tokens,width,batch_stride,token_stride,eps,silu,output,stream);
}
void quantize_rms_gated_input_interleaved(const float* input,const float* gate,const float* weight,
        int64_t batch,int64_t heads,int64_t tokens,int64_t width,
        int64_t batch_stride,int64_t token_stride,float eps,bool silu,void* output,cudaStream_t stream) {
    quantize_rms_gated_input_impl<true>(input,gate,weight,batch,heads,tokens,width,batch_stride,token_stride,eps,silu,output,stream);
}
void quantize_rms_gated_input(const float* input,const float* gate,const float* weight,
        int64_t batch,int64_t heads,int64_t tokens,int64_t width,int64_t batch_stride,int64_t token_stride,
        float eps,void* output,cudaStream_t stream) {
    quantize_rms_gated_input(input,gate,weight,batch,heads,tokens,width,batch_stride,token_stride,eps,true,output,stream);
}
void materialize_gated_input(const void* up,const void* gate,bool fp16,std::int64_t count,
        bool silu,void* output,cudaStream_t stream) {
    if(count<0 || (count && (!up || !gate || !output)))throw std::invalid_argument("Invalid materialized gate operands");
    if(!count)return;
    const auto blocks=unsigned(std::min(int64_t(65535),(count+255)/256));
    if(fp16)materialized_gate<<<blocks,256,0,stream>>>(static_cast<const half*>(up),static_cast<const half*>(gate),static_cast<half*>(output),count,silu);
    else materialized_gate<<<blocks,256,0,stream>>>(static_cast<const float*>(up),static_cast<const float*>(gate),static_cast<float*>(output),count,silu);
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(dispatch_trace::enabled()) {
        static std::once_flag reported;
        std::call_once(reported,[&]{std::fprintf(stderr,"native_gguf_prefill_gate_fused actual_match=1 count=%lld fp16=%d silu=%d dtype_rounding=1\n",static_cast<long long>(count),int(fp16),int(silu));});
    }
}

bool rms_gated_prefill_available(int64_t columns,int device) {
    static thread_local std::unordered_map<int,int> limits;
    auto i=limits.find(device);
    if(i==limits.end()) {
        int available=0;
        auto checked=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
        checked(cudaDeviceGetAttribute(&available,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
        cudaFuncAttributes kernel{};checked(cudaFuncGetAttributes(&kernel,rms_gated_prefill_kernel<false>));
        const int dynamic=std::max(0,available-int(kernel.sharedSizeBytes));
        checked(cudaFuncSetAttribute(rms_gated_prefill_kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic));
        checked(cudaFuncSetAttribute(rms_gated_prefill_kernel<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic));
        checked(cudaFuncSetAttribute(rms_gated_prefill_kernel<false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic));
        checked(cudaFuncSetAttribute(rms_gated_prefill_kernel<true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic));
        i=limits.emplace(device,dynamic).first;
    }
    return columns>0 && uint64_t(columns)<=uint64_t(i->second)/sizeof(float);
}
template<bool ScaledGate>
static void pack_rms_gated_prefill_impl(const float* input,const void* gate,const float* norm,
        int64_t batch,int64_t heads,int64_t tokens,int64_t width,int64_t gb,int64_t gt,
        float eps,bool silu,void* output,int32_t* exponents,cudaStream_t stream,
        const int32_t* gate_rows=nullptr,const int32_t* gate_columns=nullptr) {
    if(batch<1 || tokens<1 || heads<1 || width<1 || width>INT_MAX || heads>INT_MAX/width ||
            batch>INT_MAX/tokens || gt<heads*width || gt>INT64_MAX/tokens || gb<tokens*gt ||
            !input || !gate || !norm || !output || !exponents ||
            (ScaledGate && (!gate_rows || !gate_columns)))
        throw std::invalid_argument("Invalid fused prefill RMS/gate geometry");
    const auto launch=[&]<bool CacheValues>() {
        rms_gated_prefill_kernel<CacheValues,ScaledGate><<<unsigned(batch*tokens),256,size_t(heads*width)*sizeof(float),stream>>>(input,gate,norm,
            heads,tokens,int(width),gb,gt,eps,silu,static_cast<half*>(output),exponents,gate_rows,gate_columns);
    };
    // Eight cached lane values cover a complete 256-element head. Longer
    // heads retain the arbitrary-width traversal and the same RMS reduction.
    if(width<=256)launch.template operator()<true>();else launch.template operator()<false>();
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}

bool rms_prefill_available(int64_t rows,int64_t width,int device) {
    if(rows<1 || rows>INT_MAX || width<1 || width>INT_MAX)return false;
    const auto& limits=rms_prefill_limits(device);
    // This is the ordinary RMSNorm block choice; changing it can alter sums.
    const int dynamic=width>=2048 && rows<limits.processors?limits.dynamic1024:limits.dynamic256;
    return uint64_t(width)<=uint64_t(dynamic)/sizeof(float);
}
void pack_rms_prefill(const void* input,ggml_type type,const float* weight,int64_t rows,int64_t width,
        float eps,float offset,float* normalized,void* output,int32_t* exponents,cudaStream_t stream) {
    int device=0;const auto status=cudaGetDevice(&device);
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(!input || !weight || !normalized || !output || !exponents ||
            !rms_prefill_available(rows,width,device))throw std::invalid_argument("Invalid fused prefill RMS geometry or shared memory requirement");
    const auto launch=[&]<class Input>() {
        const auto launch=[&]<int Threads>() {
            rms_prefill_kernel<Input,Threads><<<unsigned(rows),Threads,size_t(width)*sizeof(float),stream>>>(
                static_cast<const Input*>(input),weight,normalized,static_cast<half*>(output),exponents,int(width),eps,offset);
        };
        if(width>=2048 && rows<rms_prefill_limits(device).processors)launch.template operator()<1024>();
        else launch.template operator()<256>();
    };
    if(type==GGML_TYPE_F32)launch.template operator()<float>();
    else if(type==GGML_TYPE_F16)launch.template operator()<half>();
    else if(type==GGML_TYPE_BF16)launch.template operator()<__nv_bfloat16>();
    else throw std::invalid_argument("Fused prefill RMS input must be F32, F16 or BF16");
    const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
void pack_rms_gated_prefill(const float* input,const float* gate,const float* norm,
        int64_t batch,int64_t heads,int64_t tokens,int64_t width,int64_t gb,int64_t gt,
        float eps,bool silu,void* output,int32_t* exponents,cudaStream_t stream) {
    pack_rms_gated_prefill_impl<false>(input,gate,norm,batch,heads,tokens,width,gb,gt,eps,silu,output,exponents,stream);
}
void pack_rms_scaled_gate_prefill(const float* input,const void* gate,const float* norm,
        const int32_t* gate_rows,const int32_t* gate_columns,
        int64_t batch,int64_t heads,int64_t tokens,int64_t width,int64_t gb,int64_t gt,
        float eps,bool silu,void* output,int32_t* exponents,cudaStream_t stream) {
    pack_rms_gated_prefill_impl<true>(input,gate,norm,batch,heads,tokens,width,gb,gt,eps,silu,output,exponents,stream,gate_rows,gate_columns);
}

}
