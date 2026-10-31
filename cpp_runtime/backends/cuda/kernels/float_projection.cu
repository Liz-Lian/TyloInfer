#include "../ops/common/include/float_projection.h"
#include "mfq_cuda_context.h"
#include <cublas_v2.h>
#include "../ops/mfq/include/nint.h"
#include "qwen4_exp.h"
#include "../ops/common/include/gated_residual.h"

#include <limits>
#include <climits>
#include <algorithm>
#include <map>
#include <mutex>
#include <type_traits>
#include <cstdlib>
#include <array>
#include <cstdio>
#include <cuda_bf16.h>
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "gr_prefill_tf32.cuh"
#include "gr_prefill_half2.cuh"
#include "gr_prefill_decode.cuh"
#include "gr_prefill_pipeline.cuh"
#include "gr_prefill_blas.h"
#include "gr_prefill_matmul.cuh"
#include "gr_prefill_mix.cuh"
#endif
#include "kernel_dispatch_trace.h"
#include "gr_bf16_prefill_mix.cuh"
#include "gr_bf16_prefill_read.cuh"

namespace {
template<class Input>
__device__ __forceinline__ float dense_bf16_input(Input value) {
    if constexpr(std::is_same_v<Input,uint16_t>)return __uint_as_float(uint32_t(value)<<16);
    else return __bfloat162float(__float2bfloat16_rn(float(value)));
}
// Weight-pair traversal and token reuse follow Strata native_bf16.cu (MIT,
// ../../../third_party/strata.LICENSE). Each token has the singleton's sum tree.
template<int Rows,class Input,class Output>
__global__ void __launch_bounds__(128) dense_bf16_projection_kernel(
        const Input* input,const uint16_t* weight,Output* output,int width,int outputs,int rows) {
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    float sums[Rows]{};
    const auto* weights=weight+size_t(row)*width;
    for(int pair=tid;pair<(width+1)/2;pair+=128) {
        const int column=pair*2;
        const float w0=__uint_as_float(uint32_t(weights[column])<<16);
        const float w1=column+1<width?__uint_as_float(uint32_t(weights[column+1])<<16):0.f;
        #pragma unroll
        for(int m=0;m<Rows;++m)if(m<rows) {
            const auto* source=input+size_t(m)*width+column;
            sums[m]=__fmaf_rn(w0,dense_bf16_input(source[0]),sums[m]);
            if(column+1<width)sums[m]=__fmaf_rn(w1,dense_bf16_input(source[1]),sums[m]);
        }
    }
    __shared__ float partial[Rows][4];
    #pragma unroll
    for(int m=0;m<Rows;++m)if(m<rows) {
        #pragma unroll
        for(int step=16;step;step>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],step);
        if(!lane)partial[m][warp]=sums[m];
    }
    __syncthreads();
    if(tid<rows) {
        const float sum=(partial[tid][0]+partial[tid][2])+(partial[tid][1]+partial[tid][3]);
        const float rounded=__bfloat162float(__float2bfloat16_rn(sum));
        if constexpr(std::is_same_v<Output,float>)output[size_t(tid)*outputs+row]=rounded;
        else output[size_t(tid)*outputs+row]=__float2half_rn(rounded);
    }
}

template<class Input>
__device__ __forceinline__ float4 gr_dense_input4(const Input* input) {
    if constexpr(std::is_same_v<Input,float>)return *reinterpret_cast<const float4*>(input);
    else {
        const uint2 x=*reinterpret_cast<const uint2*>(input);
        return make_float4(__half2float(__ushort_as_half(x.x)),__half2float(__ushort_as_half(x.x>>16)),
            __half2float(__ushort_as_half(x.y)),__half2float(__ushort_as_half(x.y>>16)));
    }
}
__device__ __forceinline__ float4 gr_dense_weight4(const uint16_t* weight) {
    const uint2 w=*reinterpret_cast<const uint2*>(weight);
    return make_float4(__uint_as_float(w.x<<16),__uint_as_float(w.x&0xffff0000u),
        __uint_as_float(w.y<<16),__uint_as_float(w.y&0xffff0000u));
}
__device__ __forceinline__ float gr_dense_dot4(float sum,float4 w,float4 x) {
    sum=fmaf(w.x,x.x,sum);sum=fmaf(w.y,x.y,sum);
    sum=fmaf(w.z,x.z,sum);return fmaf(w.w,x.w,sum);
}
template<int Rows,class Input>
__global__ void __launch_bounds__(Rows==1?128:256) gr_bf16_down_kernel(
        const Input* input,const uint16_t* weight,float* output,int width,int rank,int streams,
        const uint16_t* injection,float* injection_output) {
    const bool injecting=int(blockIdx.x)>=rank;
    const int row=injecting?int(blockIdx.x)-rank:int(blockIdx.x),tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const auto* matrix=injecting?injection:weight;
    // Preserve the singleton's accumulation order. Multiple independent rows
    // use more warps to shorten each thread's serial projection loop.
    constexpr int threads=Rows==1?128:256,warps=threads/32;
    float sums[Rows]{};
    if(width%4==0 && reinterpret_cast<std::uintptr_t>(input)%(4*sizeof(Input))==0 &&
            reinterpret_cast<std::uintptr_t>(matrix)%8==0)for(int column=tid*4;column<width;column+=threads*4) {
        const float4 w=gr_dense_weight4(matrix+size_t(row)*width+column);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=gr_dense_dot4(sums[m],w,
            gr_dense_input4(input+size_t(m)*width+column));
    }else for(int column=tid;column<width;column+=threads) {
        const float w=__uint_as_float(uint32_t(matrix[size_t(row)*width+column])<<16);
        #pragma unroll
        for(int m=0;m<Rows;++m)sums[m]=fmaf(w,float(input[size_t(m)*width+column]),sums[m]);
    }
    __shared__ float partial[Rows][warps];
    #pragma unroll
    for(int m=0;m<Rows;++m) {
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],offset);
        if(lane==0)partial[m][warp]=sums[m];
    }
    __syncthreads();
    if(tid<Rows) {
        float sum;
        if constexpr(warps==4)sum=(partial[tid][0]+partial[tid][2])+(partial[tid][1]+partial[tid][3]);
        else {
            const float s0=partial[tid][0]+partial[tid][4],s1=partial[tid][1]+partial[tid][5];
            const float s2=partial[tid][2]+partial[tid][6],s3=partial[tid][3]+partial[tid][7];
            sum=(s0+s2)+(s1+s3);
        }
        const float low=__fdiv_rn(sum,float(streams));
        const float gate=1.f/(1.f+expf(-low));
        if(injecting)injection_output[size_t(tid)*streams+row]=__fmul_rn(2.f,gate);
        else output[size_t(tid)*rank+row]=__fmul_rn(low,gate);
    }
}
template<int Rows,class Normalized>
__global__ void __launch_bounds__(256) gr_bf16_up_mix_kernel(
        const float* input,const Normalized* normalized,const uint16_t* weight,float* output,
        int rank,int hidden,int streams) {
    // One warp owns a coordinate, with four 8-lane groups evaluating four
    // streams concurrently. Eight BF16 values per lane give contiguous 128B
    // transactions within each group; every loaded weight serves all tokens.
    const int column=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
    if(column>=hidden)return;
    const int group=lane/8,part=lane%8;
    float result[Rows]{};
    for(int base=0;base<streams;base+=4) {
        const int stream=base+group;
        const int row=stream*hidden+column;
        float sums[Rows]{};
        if(stream<streams) {
            if(rank%8==0 && reinterpret_cast<std::uintptr_t>(input)%16==0 &&
                    reinterpret_cast<std::uintptr_t>(weight)%16==0)for(int begin=part*8;begin<rank;begin+=64) {
                const uint4 packed=*reinterpret_cast<const uint4*>(weight+size_t(row)*rank+begin);
                const float4 w0=make_float4(__uint_as_float(packed.x<<16),__uint_as_float(packed.x&0xffff0000u),
                    __uint_as_float(packed.y<<16),__uint_as_float(packed.y&0xffff0000u));
                const float4 w1=make_float4(__uint_as_float(packed.z<<16),__uint_as_float(packed.z&0xffff0000u),
                    __uint_as_float(packed.w<<16),__uint_as_float(packed.w&0xffff0000u));
                #pragma unroll
                for(int m=0;m<Rows;++m) {
                    sums[m]=gr_dense_dot4(sums[m],w0,gr_dense_input4(input+size_t(m)*rank+begin));
                    sums[m]=gr_dense_dot4(sums[m],w1,gr_dense_input4(input+size_t(m)*rank+begin+4));
                }
            }else for(int k=part;k<rank;k+=8) {
                const float w=__uint_as_float(uint32_t(weight[size_t(row)*rank+k])<<16);
                #pragma unroll
                for(int m=0;m<Rows;++m)sums[m]=fmaf(w,input[size_t(m)*rank+k],sums[m]);
            }
        }
        #pragma unroll
        for(int m=0;m<Rows;++m) {
            #pragma unroll
            for(int offset=4;offset>0;offset>>=1)sums[m]+=__shfl_down_sync(0xffffffffu,sums[m],offset,8);
            float product=0;
            if(part==0 && stream<streams)product=__fmul_rn(1.f/(1.f+expf(-sums[m])),
                float(normalized[size_t(m)*hidden*streams+row]));
            // Every lane participates in each shuffle, including inactive
            // stream groups. Add in stream order as the ordinary mix does.
            #pragma unroll
            for(int s=0;s<4;++s) {
                const float value=__shfl_sync(0xffffffffu,product,s*8);
                if(lane==0 && base+s<streams)result[m]=__fadd_rn(result[m],value);
            }
        }
    }
    if(lane==0) {
        #pragma unroll
        for(int m=0;m<Rows;++m)output[size_t(m)*hidden+column]=result[m]/float(streams);
    }
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
template<bool AlignedQ8=false,int FixedGroup=0,class Input>
__device__ __forceinline__ float affine_sum(
    const Input* input, const unsigned char* values,
    const unsigned char* row_bits, const int64_t* row_offsets,
    const unsigned char* scales, const unsigned char* minima,
    const float* row_scales, const float* row_minima, const __half* zero_scales,
    int64_t width, int64_t groups, int group_size, bool zero,
    int64_t neuron,int64_t sample,int64_t first,int64_t last,int lane) {
    const int bits = AlignedQ8 || zero ? 8 : row_bits[neuron];
    float sum = 0;
    for (int64_t group = first; group < last; ++group) {
        const int64_t meta = neuron * groups + group;
        const float scale = FixedGroup==0 && zero ? __half2float(zero_scales[meta])
            : row_scales[neuron] * float(scales[meta]);
        const float minimum = FixedGroup==0 && zero ? 0 : row_minima[neuron] * float(minima[meta]);
        if constexpr(FixedGroup>0) {
            const auto* packed=values+(row_offsets[neuron]>>3)+group*FixedGroup;
            const auto* activation=input+sample*width+group*FixedGroup;
            if((group+1)*FixedGroup<=width) {
#pragma unroll
                for(int j=lane;j<FixedGroup;j+=32) {
                    const float weight=__fsub_rn(__fmul_rn(scale,float(packed[j])),minimum);
                    sum=fmaf(weight,static_cast<float>(activation[j]),sum);
                }
            } else {
#pragma unroll
                for(int j=lane;j<FixedGroup;j+=32) {
                    if(group*FixedGroup+j>=width)continue;
                    const float weight=__fsub_rn(__fmul_rn(scale,float(packed[j])),minimum);
                    sum=fmaf(weight,static_cast<float>(activation[j]),sum);
                }
            }
        } else {
            for (int j = lane; j < group_size; j += 32) {
                const int64_t column = group * group_size + j;
                if (column >= width) continue;
                float q;
                if (zero) {
                    q = float(static_cast<signed char>(values[meta * group_size + j]));
                } else if constexpr(AlignedQ8) {
                    q=float(values[(row_offsets[neuron]>>3)+column]);
                } else {
                    const uint64_t bit = uint64_t(row_offsets[neuron]) + uint64_t(column) * bits;
                    const int shift = int(bit & 7);
                    unsigned packed = values[bit >> 3];
                    if (shift + bits > 8) packed |= unsigned(values[(bit >> 3) + 1]) << 8;
                    q = float((packed >> shift) & ((1u << bits) - 1));
                }
                // Separate affine rounding matches the canonical FP32 decoder.
                const float weight = __fsub_rn(__fmul_rn(scale, q), minimum);
                sum = fmaf(weight, static_cast<float>(input[sample * width + column]), sum);
            }
        }
    }
    for (int step = 16; step; step >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, step);
    return sum;
}

template<class Output,int Activation>
__device__ __forceinline__ Output projection_activation(float sum,float streams) {
    const Output projected=static_cast<Output>(sum);
    const Output low=static_cast<Output>(__fdiv_rn(static_cast<float>(projected),streams));
    const Output gate=static_cast<Output>(1.0f/(1.0f+expf(-static_cast<float>(low))));
    const float multiplier=Activation==2?2.0f:static_cast<float>(low);
    return static_cast<Output>(__fmul_rn(multiplier,static_cast<float>(gate)));
}

template<class Input,bool AlignedQ8=false,int FixedGroup=0,class Output=float,int Activation=0>
__global__ void affine_projection(
    const Input* input,const unsigned char* values,
    const unsigned char* row_bits,const int64_t* row_offsets,
    const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const __half* zero_scales,
    Output* output,int64_t width,int64_t outputs,int64_t groups,int group_size,bool zero,
    int splits,float streams) {
    const int lane=threadIdx.x&31;
    const int64_t neuron=int64_t(blockIdx.x)*4+(threadIdx.x>>5);
    if(neuron>=outputs)return;
    const int64_t sample=blockIdx.y;
    const float sum=affine_sum<AlignedQ8,FixedGroup>(input,values,row_bits,row_offsets,scales,minima,row_scales,
        row_minima,zero_scales,width,groups,group_size,zero,neuron,sample,
        groups*blockIdx.z/splits,groups*(blockIdx.z+1)/splits,lane);
    if (!lane) {
        if constexpr(Activation==0)output[(sample * outputs + neuron)*splits+blockIdx.z]=sum;
        else output[sample*outputs+neuron]=projection_activation<Output,Activation>(sum,streams);
    }
}

template<class Gate,class Value,bool AlignedQ8,class Input,int FixedGroup=0>
__global__ void affine_projection_mix(const Input* input,const unsigned char* values,
    const unsigned char* row_bits,const int64_t* row_offsets,
    const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const __half* zero_scales,
    const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>* output,
    int64_t width,int64_t hidden,int64_t groups,int group_size,bool zero,int streams) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int stream=warp;
    const int64_t hidden_column=blockIdx.x;
    const int64_t neuron=int64_t(stream)*hidden+hidden_column,sample=blockIdx.y;
    __shared__ float products[32];
    if(hidden_column<hidden) {
        const float sum=affine_sum<AlignedQ8,FixedGroup>(input,values,row_bits,row_offsets,scales,minima,row_scales,
            row_minima,zero_scales,width,groups,group_size,zero,neuron,sample,0,groups,lane);
        if(!lane) {
            const Gate projection=static_cast<Gate>(sum);
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection))));
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),
                static_cast<float>(normalized[sample*hidden*streams+neuron])));
            products[warp]=static_cast<float>(product);
        }
    }
    __syncthreads();
    if(!lane && !stream && hidden_column<hidden) {
        float mixed=0;
        for(int s=0;s<streams;++s)mixed=__fadd_rn(mixed,products[s]);
        output[sample*hidden+hidden_column]=static_cast<Output>(__fdiv_rn(mixed,float(streams)));
    }
}
template<class Gate,class Value,class Input,int FixedGroup,int Streams>
__global__ void affine_projection_warp_mix(const Input* input,const unsigned char* values,
    const int64_t* row_offsets,const unsigned char* scales,const unsigned char* minima,
    const float* row_scales,const float* row_minima,const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>* output,
    int64_t width,int64_t hidden,int64_t groups) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
    const int lane=threadIdx.x&31;
    const int64_t column=int64_t(blockIdx.x)*Streams+(threadIdx.x>>5),sample=blockIdx.y;
    if(column>=hidden)return;
    float sums[Streams]{};
    const unsigned char* packed[Streams];
    float row_scale[Streams],row_minimum[Streams];
#pragma unroll
    for(int s=0;s<Streams;++s) {
        const auto neuron=int64_t(s)*hidden+column;
        packed[s]=values+(row_offsets[neuron]>>3);
        row_scale[s]=row_scales[neuron];
        row_minimum[s]=row_minima[neuron];
    }
    for(int64_t group=0;group<groups;++group) {
        float group_scale[Streams],group_minimum[Streams];
#pragma unroll
        for(int s=0;s<Streams;++s) {
            const auto meta=(int64_t(s)*hidden+column)*groups+group;
            group_scale[s]=__fmul_rn(row_scale[s],float(scales[meta]));
            group_minimum[s]=__fmul_rn(row_minimum[s],float(minima[meta]));
        }
#pragma unroll
        for(int j=lane;j<FixedGroup;j+=32) {
            if(group*FixedGroup+j>=width)continue;
            const float activation=static_cast<float>(input[sample*width+group*FixedGroup+j]);
#pragma unroll
            for(int s=0;s<Streams;++s) {
                const float q=float(packed[s][group*FixedGroup+j]);
                const float weight=__fsub_rn(__fmul_rn(group_scale[s],q),group_minimum[s]);
                sums[s]=fmaf(weight,activation,sums[s]);
            }
        }
    }
    for(int step=16;step;step>>=1) {
#pragma unroll
        for(int s=0;s<Streams;++s)sums[s]+=__shfl_down_sync(0xffffffffu,sums[s],step);
    }
    if(!lane) {
        float mixed=0;
#pragma unroll
        for(int s=0;s<Streams;++s) {
            const Gate projection=static_cast<Gate>(sums[s]);
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection))));
            const auto neuron=int64_t(s)*hidden+column;
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),
                static_cast<float>(normalized[sample*hidden*Streams+neuron])));
            mixed=__fadd_rn(mixed,static_cast<float>(product));
        }
        output[sample*hidden+column]=static_cast<Output>(__fdiv_rn(mixed,float(Streams)));
    }
}
__global__ void reduce_projection_splits(const float* partial,float* output,int64_t count,int splits) {
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=int64_t(gridDim.x)*blockDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[row*splits+split];
        output[row]=sum;
    }
}
template<class Output,int Activation>
__global__ void reduce_projection_activation(const float* partial,Output* output,int64_t count,int splits,float streams) {
    for(int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<count;row+=int64_t(gridDim.x)*blockDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[row*splits+split];
        output[row]=projection_activation<Output,Activation>(sum,streams);
    }
}
int projection_resident_blocks(int device) {
    static std::mutex mutex;
    static std::map<int,int> devices;
    std::lock_guard lock(mutex);
    if(const auto found=devices.find(device);found!=devices.end())return found->second;
    cudaDeviceProp properties{};
    TYLOINFER_CUDA_CHECK(cudaGetDeviceProperties(&properties,device));
    int active=0;
    TYLOINFER_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,affine_projection<float,false>,128,0));
    const int blocks=active*properties.multiProcessorCount;
    devices.emplace(device,blocks);return blocks;
}
#endif

} // namespace

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {
mfq_tensor_backend::Tensor float_projection_impl(
        const NintWeight& weight, const mfq_tensor_backend::Tensor& input,bool parallel_groups,bool native_input,
        bool fixed_groups,int activation,int64_t streams) {
    namespace tb = mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && weight.q_packed.is_cuda() &&
        input.device() == weight.q_packed.device() && input.dim() >= 1 &&
        input.size(-1) == weight.neuron_len && weight.neuron_len > 0 && weight.out > 0 &&
        weight.gs > 0 && weight.gs <= 64 &&
        weight.ng == (weight.neuron_len + weight.gs - 1) / weight.gs,
        "packed float projection dimensions/device disagree");
    MfqCudaGuard guard(input.device());
    auto x = native_input && input.scalar_type()==tb::kFloat16
        ? input.contiguous() : input.to(tb::kFloat32).contiguous();
    auto shape = x.sizes().vec();
    const auto rows = x.numel() / weight.neuron_len;
    TYLOINFER_RUNTIME_CHECK(rows <= 65535 && (weight.out + 3) / 4 <= std::numeric_limits<int>::max(),
        "packed float projection exceeds CUDA launch geometry");
    shape.back() = weight.out;
    const auto output_dtype=activation && input.scalar_type()==tb::kFloat16?tb::kFloat16:tb::kFloat32;
    auto result = tb::empty(shape, x.options().dtype(output_dtype));
    if (!rows) return result;
    const int64_t blocks=((weight.out+3)/4)*rows;
    const int target=parallel_groups?projection_resident_blocks(x.get_device()):1;
    const int splits=int(std::min<int64_t>(weight.ng,std::max<int64_t>(1,(target+blocks-1)/blocks)));
    auto partial=splits>1?tb::empty({rows,weight.out,splits},result.options().dtype(tb::kFloat32)):result;
    const auto launch=[&](auto input_tag,auto aligned_tag,auto group_tag) {
        using Input=decltype(input_tag);constexpr bool aligned=decltype(aligned_tag)::value;
        constexpr int group=decltype(group_tag)::value;
        const auto invoke=[&](auto output_tag,auto activation_tag,const tb::Tensor& target) {
        using Output=decltype(output_tag);constexpr int action=decltype(activation_tag)::value;
        affine_projection<Input,aligned,group,Output,action><<<dim3(unsigned((weight.out + 3) / 4), unsigned(rows),unsigned(splits)), 128, 0,
        mfq_current_cuda_stream()>>>(x.data_ptr<Input>(), weight.q_packed.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bits.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.row_q_bit_offsets.data_ptr<int64_t>(),
        weight.q8_zero ? nullptr : weight.sub_scale.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.sub_min.data_ptr<uint8_t>(),
        weight.q8_zero ? nullptr : weight.neuron_scale.data_ptr<float>(),
        weight.q8_zero ? nullptr : weight.neuron_min.data_ptr<float>(),
        weight.q8_zero ? weight.q8_zero_scale.data_ptr<__half>() : nullptr,
        target.data_ptr<Output>(), weight.neuron_len, weight.out, weight.ng, int(weight.gs),
        weight.q8_zero,splits,float(streams));
        };
        if(!activation || splits>1)invoke(float{},std::integral_constant<int,0>{},partial);
        else {
            const auto activated=[&](auto output_tag) {
                if(activation==2)invoke(output_tag,std::integral_constant<int,2>{},result);
                else invoke(output_tag,std::integral_constant<int,1>{},result);
            };
            if(output_dtype==tb::kFloat16)activated(__half{});else activated(float{});
        }
    };
    const auto dispatch=[&](auto input_tag) {
        if(native_input && weight.aligned_q8) {
            if(fixed_groups && !weight.q8_zero) {
                switch(weight.gs) {
                    case 32:launch(input_tag,std::true_type{},std::integral_constant<int,32>{});return;
                    case 48:launch(input_tag,std::true_type{},std::integral_constant<int,48>{});return;
                    case 64:launch(input_tag,std::true_type{},std::integral_constant<int,64>{});return;
                }
            }
            launch(input_tag,std::true_type{},std::integral_constant<int,0>{});
        } else launch(input_tag,std::false_type{},std::integral_constant<int,0>{});
    };
    if(x.scalar_type()==tb::kFloat16)dispatch(__half{});else dispatch(float{});
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    if(splits>1) {
        const int64_t count=rows*weight.out;
        if(!activation)reduce_projection_splits<<<unsigned((count+255)/256),256,0,mfq_current_cuda_stream()>>>(
                partial.data_ptr<float>(),result.data_ptr<float>(),count,splits);
        else {
            const auto invoke=[&](auto output_tag,auto activation_tag) {
                using Output=decltype(output_tag);constexpr int action=decltype(activation_tag)::value;
                reduce_projection_activation<Output,action><<<unsigned((count+255)/256),256,0,mfq_current_cuda_stream()>>>(
                    partial.data_ptr<float>(),result.data_ptr<Output>(),count,splits,float(streams));
            };
            const auto activated=[&](auto output_tag) {
                if(activation==2)invoke(output_tag,std::integral_constant<int,2>{});
                else invoke(output_tag,std::integral_constant<int,1>{});
            };
            if(output_dtype==tb::kFloat16)activated(__half{});else activated(float{});
        }
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    return result;
}
} // namespace

#endif

mfq_tensor_backend::Tensor try_gr_bf16_prefill_projection_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight) {
    namespace tb=mfq_tensor_backend;
    const auto* setting=std::getenv("TYLOINFER_GR_PREFILL_BF16");
    if(setting && setting[0]=='0')return {};
    // BF16 MMA has a 16-column output tile; smaller projections stay on
    // the existing path rather than paying a tensor-core staging round trip.
    if(!input.defined() || !weight.defined() || !input.is_cuda() || !weight.is_cuda() ||
            input.device()!=weight.device() || weight.scalar_type()!=tb::kBFloat16 || weight.dim()!=2 ||
            input.dim()<1 || input.size(-1)<=0 || weight.size(1)!=input.size(-1) ||
            weight.size(0)<16 || weight.size(0)>INT_MAX || weight.size(1)>INT_MAX ||
            (input.scalar_type()!=tb::kFloat32 && input.scalar_type()!=tb::kBFloat16))return {};
    const auto rows=input.numel()/input.size(-1);
    if(rows<=32 || rows>INT_MAX)return {};
    const MfqCudaGuard guard(input.get_device());
    int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
    // Native BF16 tensor-core GEMM starts at compute capability 8.0.
    if(major<8)return {};
    auto x=input.to(tb::kBFloat16).contiguous(),w=weight.contiguous();
    // GEMM may select vectorized global loads; retain arbitrary input views.
    if(reinterpret_cast<std::uintptr_t>(x.data_ptr())%16)x=x.clone();
    if(reinterpret_cast<std::uintptr_t>(w.data_ptr())%16)w=w.clone();
    auto shape=input.sizes().vec();shape.back()=weight.size(0);
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    const auto stream=mfq_current_cuda_stream();
    for(const auto& tensor:{x,w,output})tensor.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    auto& blas=mfq::cuda::default_context(input.get_device())->blas();blas.set_stream(stream);
    const float alpha=1.f,beta=0.f;
    const auto status=cublasGemmEx(blas.get(),CUBLAS_OP_T,CUBLAS_OP_N,int(weight.size(0)),int(rows),int(weight.size(1)),
        &alpha,w.data_ptr(),CUDA_R_16BF,int(weight.size(1)),x.data_ptr(),CUDA_R_16BF,int(weight.size(1)),
        &beta,output.data_ptr<float>(),CUDA_R_32F,int(weight.size(0)),CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status==CUBLAS_STATUS_NOT_SUPPORTED || status==CUBLAS_STATUS_ARCH_MISMATCH)return {};
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("BF16 residual prefill GEMM failed");
    static std::once_flag traced;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')std::call_once(traced,[&] {
        std::fprintf(stderr,"native_gr_prefill_bf16 actual_match=1 tokens=%lld rows=%lld columns=%lld fp32_accumulation=1\n",
            static_cast<long long>(rows),static_cast<long long>(weight.size(0)),static_cast<long long>(weight.size(1)));
    });
    return output;
}

mfq_tensor_backend::Tensor try_dense_bf16_projection_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight) {
    namespace tb=mfq_tensor_backend;
    if(!input.defined() || !weight.defined() || !input.is_cuda() || !weight.is_cuda() ||
            input.device()!=weight.device() || !input.is_contiguous() || !weight.is_contiguous() ||
            weight.scalar_type()!=tb::kBFloat16 || weight.dim()!=2 || input.dim()<1 || input.size(-1)<1 ||
            input.size(-1)!=weight.size(1) || weight.size(0)<1 || weight.size(0)>INT_MAX || weight.size(1)>=INT_MAX ||
            (input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kFloat32 && input.scalar_type()!=tb::kBFloat16))return {};
    const auto rows=input.numel()/input.size(-1);
    // The common adapter already uses one full GEMM from seven rows onward;
    // keep that tensor-core path instead of replacing it with a scalar dot.
    if(rows<1 || rows>6)return {};
    auto shape=input.sizes().vec();shape.back()=weight.size(0);
    auto output=tb::empty(shape,input.options().dtype(input.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16));
    const MfqCudaGuard guard(input.get_device());
    const auto stream_handle=mfq_get_current_cuda_stream(input.get_device());
    const auto stream=stream_handle.stream();
    for(const auto& tensor:{input,weight,output})mfq_cuda_record_stream(tensor,stream_handle);
    const auto launch=[&](auto capacity,auto input_tag,auto output_tag) {
        using Input=decltype(input_tag);using Output=decltype(output_tag);
        dense_bf16_projection_kernel<decltype(capacity)::value,Input,Output><<<unsigned(weight.size(0)),128,0,stream>>>(
            static_cast<const Input*>(input.data_ptr()),static_cast<const uint16_t*>(weight.data_ptr()),
            static_cast<Output*>(output.data_ptr()),int(weight.size(1)),int(weight.size(0)),int(rows));
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"dense_bf16_projection",
            reinterpret_cast<const void*>(dense_bf16_projection_kernel<decltype(capacity)::value,Input,Output>),
            "M="+std::to_string(rows)+" N="+std::to_string(weight.size(0))+" K="+std::to_string(weight.size(1)));
    };
    const auto dispatch=[&](auto input_tag,auto output_tag) {
        if(rows==1)launch(std::integral_constant<int,1>{},input_tag,output_tag);
        else if(rows==2)launch(std::integral_constant<int,2>{},input_tag,output_tag);
        else if(rows==3)launch(std::integral_constant<int,3>{},input_tag,output_tag);
        else if(rows==4)launch(std::integral_constant<int,4>{},input_tag,output_tag);
        else launch(std::integral_constant<int,8>{},input_tag,output_tag);
    };
    if(input.scalar_type()==tb::kFloat32)dispatch(float{},float{});
    else if(input.scalar_type()==tb::kFloat16)dispatch(__half{},__half{});
    else dispatch(uint16_t{},__half{});
    TYLOINFER_CUDA_CHECK(cudaGetLastError());return output;
}

std::vector<mfq_tensor_backend::Tensor> try_gr_bf16_down_injection_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight,const mfq_tensor_backend::Tensor& injection,int64_t streams) {
    namespace tb=mfq_tensor_backend;
    if(streams<1 || streams>INT_MAX || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 || input.size(-1)<1 ||
            input.size(-1)>INT_MAX || input.numel()/input.size(-1)<1 || input.numel()/input.size(-1)>6 ||
            (input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kFloat32) ||
            !weight.is_cuda() || !weight.is_contiguous() || weight.device()!=input.device() ||
            weight.scalar_type()!=tb::kBFloat16 || weight.dim()!=2 || weight.size(0)<1 || weight.size(0)>INT_MAX ||
            weight.size(1)!=input.size(-1))return {};
    if(injection.defined() && (!injection.is_cuda() || !injection.is_contiguous() || injection.device()!=input.device() ||
            injection.scalar_type()!=tb::kBFloat16 || injection.dim()!=2 || injection.size(0)!=streams ||
            injection.size(1)!=input.size(-1) || weight.size(0)>INT_MAX-streams))return {};
    const MfqCudaGuard guard(input.device());
    const int width=int(weight.size(1)),rank=int(weight.size(0)),rows=int(input.numel()/width);
    auto shape=input.sizes().vec();shape.back()=rank;
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    tb::Tensor gates;
    if(injection.defined()) {shape.back()=streams;gates=tb::empty(shape,output.options());}
    const auto launch=[&](auto capacity,auto type) {
        using Input=decltype(type);
        constexpr int threads=decltype(capacity)::value==1?128:256;
        gr_bf16_down_kernel<decltype(capacity)::value,Input><<<rank+(gates.defined()?int(streams):0),threads,0,mfq_current_cuda_stream()>>>(
            input.data_ptr<Input>(),static_cast<const uint16_t*>(weight.data_ptr()),output.data_ptr<float>(),width,rank,int(streams),
            gates.defined()?static_cast<const uint16_t*>(injection.data_ptr()):nullptr,gates.defined()?gates.data_ptr<float>():nullptr);
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            gates.defined()?"gr_bf16_down_silu_injection":"gr_bf16_down_silu",
            reinterpret_cast<const void*>(gr_bf16_down_kernel<decltype(capacity)::value,Input>),
            "M="+std::to_string(rows)+" N="+std::to_string(rank)+" K="+std::to_string(width));
    };
    const auto types=[&](auto capacity) {
        if(input.scalar_type()==tb::kFloat16)launch(capacity,__half{});else launch(capacity,float{});
    };
    switch(rows) {
        case 1:types(std::integral_constant<int,1>{});break;
        case 2:types(std::integral_constant<int,2>{});break;
        case 3:types(std::integral_constant<int,3>{});break;
        case 4:types(std::integral_constant<int,4>{});break;
        case 5:types(std::integral_constant<int,5>{});break;
        case 6:types(std::integral_constant<int,6>{});break;
    }
    TYLOINFER_CUDA_CHECK(cudaGetLastError());return {output,gates};
}

mfq_tensor_backend::Tensor try_gr_bf16_down_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight,int64_t streams) {
    auto result=try_gr_bf16_down_injection_cuda(input,weight,{},streams);
    return result.empty()?mfq_tensor_backend::Tensor{}:result[0];
}

mfq_tensor_backend::Tensor try_gr_bf16_up_mix_cuda(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& normalized,const mfq_tensor_backend::Tensor& weight,int64_t streams) {
    if(auto output=try_gr_bf16_prefill_mix(input,normalized,weight,streams);output.defined())return output;
    namespace tb=mfq_tensor_backend;
    if(streams<1 || streams>INT_MAX || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 || input.size(-1)<1 ||
            input.size(-1)>INT_MAX || input.numel()/input.size(-1)<1 || input.numel()/input.size(-1)>6 ||
            input.scalar_type()!=tb::kFloat32 || !normalized.is_cuda() || !normalized.is_contiguous() ||
            normalized.device()!=input.device() ||
            (normalized.scalar_type()!=tb::kFloat16 && normalized.scalar_type()!=tb::kFloat32) ||
            !weight.is_cuda() || !weight.is_contiguous() || weight.device()!=input.device() ||
            weight.scalar_type()!=tb::kBFloat16 || weight.dim()!=2 || weight.size(0)<streams || weight.size(0)>INT_MAX ||
            weight.size(0)%streams || weight.size(1)!=input.size(-1))return {};
    const int rank=int(weight.size(1)),hidden=int(weight.size(0)/streams),rows=int(input.numel()/rank);
    auto expected=input.sizes().vec();expected.back()=weight.size(0);
    if(normalized.sizes().vec()!=expected)return {};
    const MfqCudaGuard guard(input.device());
    auto shape=input.sizes().vec();shape.back()=hidden;
    auto output=tb::empty(shape,input.options());
    const auto launch=[&](auto capacity,auto type) {
        using Normalized=decltype(type);
        gr_bf16_up_mix_kernel<decltype(capacity)::value,Normalized><<<(hidden+7)/8,256,0,mfq_current_cuda_stream()>>>(
            input.data_ptr<float>(),normalized.data_ptr<Normalized>(),static_cast<const uint16_t*>(weight.data_ptr()),output.data_ptr<float>(),rank,hidden,int(streams));
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(mfq_current_cuda_stream(),
            "gr_bf16_up_mix",reinterpret_cast<const void*>(gr_bf16_up_mix_kernel<decltype(capacity)::value,Normalized>),
            "M="+std::to_string(rows)+" N="+std::to_string(weight.size(0))+" K="+std::to_string(rank));
    };
    const auto types=[&](auto capacity) {
        if(normalized.scalar_type()==tb::kFloat16)launch(capacity,__half{});else launch(capacity,float{});
    };
    switch(rows) {
        case 1:types(std::integral_constant<int,1>{});break;
        case 2:types(std::integral_constant<int,2>{});break;
        case 3:types(std::integral_constant<int,3>{});break;
        case 4:types(std::integral_constant<int,4>{});break;
        case 5:types(std::integral_constant<int,5>{});break;
        case 6:types(std::integral_constant<int,6>{});break;
    }
    TYLOINFER_CUDA_CHECK(cudaGetLastError());return output;
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
mfq_tensor_backend::Tensor nint_float_projection_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,bool parallel_groups,bool native_input,bool fixed_groups) {
    if(gr_prefill_selected(input))return gr_prefill_matmul(weight,input);
    return float_projection_impl(weight,input,parallel_groups,native_input,fixed_groups,0,1);
}

int nint_float_projection_resident_blocks(int device) {
    return projection_resident_blocks(device);
}

mfq_tensor_backend::Tensor nint_float_projection_activation_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,int64_t streams,bool injection,
        bool parallel_groups,bool native_input,bool fixed_groups) {
    TYLOINFER_RUNTIME_CHECK(streams>0,"packed residual activation requires positive stream count");
    if(gr_prefill_selected(input)) {
        auto projected=gr_prefill_matmul(weight,input);
        if(input.scalar_type()==mfq_tensor_backend::kFloat16)projected=projected.to(mfq_tensor_backend::kFloat16);
        return injection ? mfq_qwen4_exp::gated_residual_injection(projected,streams)
                         : mfq_qwen4_exp::gated_residual_bottleneck(projected,streams);
    }
    return float_projection_impl(weight,input,parallel_groups,native_input,fixed_groups,injection?2:1,streams);
}

mfq_tensor_backend::Tensor nint_float_projection_mix_cuda(const NintWeight& weight,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized,
        int64_t streams,bool native_input,bool fixed_groups,bool compact) {
    namespace tb=mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(input.dim()>0 && streams>0 && weight.out>0 && weight.out%streams==0 &&
        normalized.dim()==input.dim(),"packed residual mixing dimensions disagree");
    auto expected=input.sizes().vec();expected.back()=weight.out;
    TYLOINFER_RUNTIME_CHECK(normalized.sizes().vec()==expected && input.is_cuda() && normalized.is_cuda() &&
        input.device()==normalized.device() && input.device()==weight.q_packed.device() &&
        input.size(-1)==weight.neuron_len && weight.ng>0 && weight.gs>0 && weight.gs<=64 &&
        weight.ng==(weight.neuron_len+weight.gs-1)/weight.gs,"packed residual mixing geometry/device disagree");
    const auto rows=input.numel()/weight.neuron_len,hidden=weight.out/streams;
    if(gr_prefill_selected(input)) {
        if(gr_prefill_blas_selected(weight,input) && normalized.is_contiguous() &&
                (normalized.scalar_type()==tb::kFloat16 || normalized.scalar_type()==tb::kFloat32))
            return gr_prefill_blas_mix(weight,input,normalized,streams);
        if(gr_prefill_mix_selected(weight,input,normalized,streams))return gr_prefill_fused_mix(weight,input,normalized);
        auto projected=gr_prefill_matmul(weight,input);
        if(input.scalar_type()==tb::kFloat16)projected=projected.to(tb::kFloat16);
        return mfq_qwen4_exp::gated_residual_mix(projected,normalized,streams);
    }
    const auto ordinary_blocks=((weight.out+3)/4)*rows;
    MfqCudaGuard guard(input.device());
    // Retain the original group partition whenever the ordinary projection splits.
    const bool unsplit=weight.ng==1 || !rows || ordinary_blocks>=projection_resident_blocks(input.get_device());
    const auto half_or_float=[](const tb::Tensor& value) {
        return value.scalar_type()==tb::kFloat16 || value.scalar_type()==tb::kFloat32;
    };
    if(!unsplit || streams>32 || !input.is_contiguous() || !normalized.is_contiguous() ||
        !half_or_float(input) || !half_or_float(normalized)) {
        auto projection=nint_float_projection_cuda(weight,input,true,native_input,fixed_groups);
        if(input.scalar_type()==tb::kFloat16)projection=projection.to(tb::kFloat16);
        return mfq_qwen4_exp::gated_residual_mix(projection,normalized,streams);
    }
    TYLOINFER_RUNTIME_CHECK(rows<=65535 && hidden<=std::numeric_limits<int>::max(),
        "packed residual mixing exceeds CUDA launch geometry");
    auto x=native_input && input.scalar_type()==tb::kFloat16?input:input.to(tb::kFloat32);
    auto shape=input.sizes().vec();shape.back()=hidden;
    const auto dtype=input.scalar_type()==tb::kFloat32 || normalized.scalar_type()==tb::kFloat32
        ?tb::kFloat32:tb::kFloat16;
    auto output=tb::empty(shape,input.options().dtype(dtype));
    if(!rows)return output;
    const auto launch=[&](auto gate_tag,auto value_tag,auto aligned_tag,auto input_tag,auto group_tag) {
        using Gate=decltype(gate_tag);using Value=decltype(value_tag);
        using Input=decltype(input_tag);
        using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,__half>;
        constexpr bool aligned=decltype(aligned_tag)::value;
        constexpr int group=decltype(group_tag)::value;
        if constexpr(aligned && group>0) {
            if(compact && (streams==3 || streams==4)) {
                const auto warp_launch=[&](auto stream_tag) {
                    constexpr int count=decltype(stream_tag)::value;
                    affine_projection_warp_mix<Gate,Value,Input,group,count><<<
                        dim3(unsigned((hidden+count-1)/count),unsigned(rows)),count*32,0,mfq_current_cuda_stream()>>>(
                        x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),weight.row_q_bit_offsets.data_ptr<int64_t>(),
                        weight.sub_scale.data_ptr<uint8_t>(),weight.sub_min.data_ptr<uint8_t>(),
                        weight.neuron_scale.data_ptr<float>(),weight.neuron_min.data_ptr<float>(),
                        normalized.data_ptr<Value>(),output.data_ptr<Output>(),weight.neuron_len,hidden,weight.ng);
                };
                if(streams==3)warp_launch(std::integral_constant<int,3>{});
                else warp_launch(std::integral_constant<int,4>{});
                return;
            }
        }
        affine_projection_mix<Gate,Value,aligned,Input,group><<<dim3(unsigned(hidden),unsigned(rows)),unsigned(streams*32),0,
            mfq_current_cuda_stream()>>>(x.data_ptr<Input>(),weight.q_packed.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.row_q_bits.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.row_q_bit_offsets.data_ptr<int64_t>(),
            weight.q8_zero?nullptr:weight.sub_scale.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.sub_min.data_ptr<uint8_t>(),
            weight.q8_zero?nullptr:weight.neuron_scale.data_ptr<float>(),
            weight.q8_zero?nullptr:weight.neuron_min.data_ptr<float>(),
            weight.q8_zero?weight.q8_zero_scale.data_ptr<__half>():nullptr,
            normalized.data_ptr<Value>(),output.data_ptr<Output>(),weight.neuron_len,hidden,weight.ng,
            int(weight.gs),weight.q8_zero,int(streams));
    };
    const auto dispatch=[&](auto gate_tag,auto value_tag) {
        const auto typed=[&](auto input_tag) {
            if(weight.aligned_q8) {
                if(fixed_groups && !weight.q8_zero) {
                    switch(weight.gs) {
                        case 32:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,32>{});return;
                        case 48:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,48>{});return;
                        case 64:launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,64>{});return;
                    }
                }
                launch(gate_tag,value_tag,std::true_type{},input_tag,std::integral_constant<int,0>{});
            } else launch(gate_tag,value_tag,std::false_type{},input_tag,std::integral_constant<int,0>{});
        };
        if(x.scalar_type()==tb::kFloat16)typed(__half{});else typed(float{});
    };
    if(input.scalar_type()==tb::kFloat16 && normalized.scalar_type()==tb::kFloat16)dispatch(__half{},__half{});
    else if(input.scalar_type()==tb::kFloat16)dispatch(__half{},float{});
    else if(normalized.scalar_type()==tb::kFloat16)dispatch(float{},__half{});
    else dispatch(float{},float{});
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return output;
}

#endif

mfq_tensor_backend::Tensor prepare_gr_dense_vector_right(const mfq_tensor_backend::Tensor& right) {
    namespace tb=mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(right.is_cuda() && right.dim()==2 && right.scalar_type()==tb::kFloat32 &&
        right.size(0)>0 && right.size(0)%4==0 && right.size(1)>0 && right.size(1)<=4,
        "vector residual injection requires F32 [4*k, 1..4] weights");
    return right.reshape({right.size(0)/4,4,right.size(1)}).transpose(1,2).contiguous();
}
