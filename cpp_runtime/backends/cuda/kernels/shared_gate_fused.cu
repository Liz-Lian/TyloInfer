#include "mfq_cuda_shared_gate.h"
#include "mfq_cuda_context.h"
#include "shared_gate_fused.cuh"
#include "kernel_dispatch_trace.h"
#include "../ops/mfq/include/nint.h"
#include <cstdlib>

namespace {
template<class Input,class Output>
__global__ void shared_gate_multiply_f32(const Input* input,const float* weight,const Output* expert,
        float* output,int64_t width,int64_t output_width) {
    __shared__ float partial[4];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const auto* x=input+int64_t(blockIdx.x)*width;
    float sum=0.f;
    for(int64_t i=threadIdx.x;i<width;i+=128)sum=__fmaf_rn(weight[i],float(x[i]),sum);
    for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
    if(!lane)partial[warp]=sum;
    __syncthreads();
    if(!warp) {
        sum=lane<4?partial[lane]:0.f;
        for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
        if(!lane)partial[0]=1.f/(1.f+expf(-sum));
    }
    __syncthreads();
    const auto first=int64_t(blockIdx.x)*output_width;
    for(int64_t i=threadIdx.x;i<output_width;i+=128)
        output[first+i]=__fmul_rn(partial[0],float(expert[first+i]));
}
}

std::optional<mfq_tensor_backend::Tensor> try_shared_gate_multiply_cuda(
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& weight,
        const mfq_tensor_backend::Tensor& expert_output) {
    namespace tb=mfq_tensor_backend;
    const auto supported=[](const auto& value) {
        return value.defined() && value.is_cuda() && value.is_contiguous() && value.dim()==2;
    };
    if(!supported(input) || !supported(weight) || !supported(expert_output) ||
            input.device()!=weight.device() || input.device()!=expert_output.device() ||
            weight.scalar_type()!=tb::kFloat32 || weight.size(0)!=1 || input.size(1)<1 ||
            weight.size(1)!=input.size(1) || input.size(0)<1 || input.size(0)>6 ||
            expert_output.size(0)!=input.size(0) || expert_output.size(1)<1 ||
            (input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kFloat32) ||
            (expert_output.scalar_type()!=tb::kFloat16 && expert_output.scalar_type()!=tb::kFloat32))return std::nullopt;
    // dense_projection executes independent FP32 dots for these short
    // windows. Only the FP32 addition tree changes; all values stay FP32.
    mfq::cuda::DeviceGuard guard(input.get_device());
    auto output=tb::empty(expert_output.sizes(),expert_output.options().dtype(tb::kFloat32));
    const auto stream_handle=mfq::cuda::current_stream(input.get_device());
    const auto stream=stream_handle.stream();
    for(const auto& tensor:{input,weight,expert_output,output})mfq_cuda_record_stream(tensor,stream_handle);
    const auto launch=[&](auto* source,auto* expert) {
        using Input=std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
        using Expert=std::remove_cv_t<std::remove_pointer_t<decltype(expert)>>;
        const auto function=shared_gate_multiply_f32<Input,Expert>;
        function<<<unsigned(input.size(0)),128,0,stream>>>(source,weight.data_ptr<float>(),expert,
            output.data_ptr<float>(),input.size(1),expert_output.size(1));
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"shared_gate_multiply_f32",
            reinterpret_cast<const void*>(function),"rows="+std::to_string(input.size(0))+" K="+
            std::to_string(input.size(1))+" output_width="+std::to_string(expert_output.size(1))+" output_float=1");
    };
    const auto choose_output=[&](auto* source) {
        if(expert_output.scalar_type()==tb::kFloat16)launch(source,static_cast<const __half*>(expert_output.data_ptr()));
        else launch(source,expert_output.data_ptr<float>());
    };
    if(input.scalar_type()==tb::kFloat16)choose_output(static_cast<const __half*>(input.data_ptr()));
    else choose_output(input.data_ptr<float>());
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());return output;
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {
template<bool Ordered>
__global__ void nint_gate_prepared(const int8_t* activation,const float* activation_scale,
        const uint8_t* packed,const int64_t* offset,const uint8_t* scales,const uint8_t* minima,
        const float* outer_scale,const float* outer_minimum,__half* output,int groups) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const auto* x=activation+size_t(blockIdx.x)*groups*48;
    const auto* xs=activation_scale+size_t(blockIdx.x)*groups;
    const auto* w=packed+(uint64_t(offset[0])>>3);
    __shared__ float terms[64][12],partial[4];
    float accumulator=0.f;
    if constexpr(Ordered) {
        const int relative=lane/12,chunk=lane%12;
        if(relative<2)for(int group=warp*2+relative;group<groups;group+=8) {
            const int column=group*48+chunk*4;
            const int q=*reinterpret_cast<const int*>(x+column);
            const int code=*reinterpret_cast<const int*>(w+column);
            const int sum=__dp4a(0x01010101,q,0);
            const int dot=__dp4a(code^int(0x80808080u),q,0)+128*sum;
            terms[group][chunk]=outer_scale[0]*float(scales[group])*float(dot)-
                outer_minimum[0]*float(minima[group])*float(sum);
        }
        __syncthreads();
        if(warp)return;
        if(relative<2)for(int group=relative;group<groups;group+=2)
            accumulator+=xs[group]*terms[group][chunk];
        #pragma unroll
        for(int step=16;step;step>>=1)accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,step);
    } else {
        const int member=lane&3;
        for(int base=warp*8;base<groups;base+=32) {
            const int group=base+(lane>>2);const bool active=group<groups;
            int dot=0,sum=0;
            #pragma unroll
            for(int chunk=0;chunk<3;++chunk) {
                const int column=group*48+(chunk*4+member)*4;
                const int code=active?*reinterpret_cast<const int*>(w+column):0;
                const int q=active?*reinterpret_cast<const int*>(x+column):0;
                asm("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(dot) : "r"(code),"r"(q));
                sum=__dp4a(0x01010101,q,sum);
            }
            dot+=__shfl_xor_sync(0xffffffffu,dot,2,4);dot+=__shfl_xor_sync(0xffffffffu,dot,1,4);
            sum+=__shfl_xor_sync(0xffffffffu,sum,2,4);sum+=__shfl_xor_sync(0xffffffffu,sum,1,4);
            if(active && member==0)accumulator+=xs[group]*(outer_scale[0]*float(scales[group])*float(dot)-
                outer_minimum[0]*float(minima[group])*float(sum));
        }
        #pragma unroll
        for(int step=16;step;step>>=1)accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,step);
        if(!lane)partial[warp]=accumulator;
        __syncthreads();
        if(warp)return;
        accumulator=lane<4?partial[lane]:0.f;
        #pragma unroll
        for(int step=16;step;step>>=1)accumulator+=__shfl_xor_sync(0xffffffffu,accumulator,step);
    }
    if(!lane)output[blockIdx.x]=__float2half_rn(1.f/(1.f+expf(-__half2float(__float2half_rn(accumulator)))));
}

// One packed GS48 NINT8 row. Keep the ordinary projection's activation
// quantization, affine/FMA order, Half dot boundary and Half sigmoid result.
template<bool Ordered>
__global__ void nint_gate_sigmoid(const __half* input,const uint8_t* packed,const int64_t* offset,
        const uint8_t* subgroup_scale,const uint8_t* subgroup_minimum,
        const float* outer_scale,const float* outer_minimum,__half* output,int width,int groups) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const auto* x=input+size_t(blockIdx.x)*width;
    const auto* weights=packed+(uint64_t(offset[0])>>3);
    __shared__ float terms[64][12],scales[64],partial[4];
    for(int group=warp;group<groups;group+=blockDim.x/32) {
        const int column=group*48+lane;
        const float a=column<width?__half2float(x[column]):0.f;
        const float b=lane<16 && column+32<width?__half2float(x[column+32]):0.f;
        float maximum=fmaxf(fabsf(a),fabsf(b));
        #pragma unroll
        for(int step=16;step;step>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,step));
        const float scale=maximum>0.f?maximum/127.f:1.f;
        const int qa=max(-127,min(127,int(roundf(a/scale))));
        const int qb=max(-127,min(127,int(roundf(b/scale))));
        // All lanes participate in every shuffle; twelve lanes consume one
        // original packed four-byte chunk each.
        uint32_t q=0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int index=(lane%12)*4+j;
            const int first=__shfl_sync(0xffffffffu,qa,index&31);
            const int second=__shfl_sync(0xffffffffu,qb,index&31);
            q|=uint32_t(uint8_t(index<32?first:second))<<(j*8);
        }
        int dot=0,sum=0;
        if(lane<12) {
            const int w=*reinterpret_cast<const int*>(weights+group*48+lane*4);
            sum=__dp4a(0x01010101,int(q),0);
            dot=__dp4a(w^int(0x80808080u),int(q),0)+128*sum;
        }
        const float factor=outer_scale[0]*float(subgroup_scale[group]);
        const float minimum=outer_minimum[0]*float(subgroup_minimum[group]);
        if constexpr(Ordered) {
            if(lane<12)terms[group][lane]=factor*float(dot)-minimum*float(sum);
        } else {
            #pragma unroll
            for(int step=16;step;step>>=1) {
                dot+=__shfl_xor_sync(0xffffffffu,dot,step);
                sum+=__shfl_xor_sync(0xffffffffu,sum,step);
            }
            if(!lane)terms[group][0]=factor*float(dot)-minimum*float(sum);
        }
        if(!lane)scales[group]=scale;
    }
    __syncthreads();
    float value=0.f;
    if constexpr(Ordered) {
        if(warp)return;
        if(lane/12<2)for(int group=lane/12;group<groups;group+=2)
            value+=scales[group]*terms[group][lane%12];
        #pragma unroll
        for(int step=16;step;step>>=1)value+=__shfl_xor_sync(0xffffffffu,value,step);
    } else {
        if(warp<4) {
            if((lane&3)==0)for(int group=warp*8+(lane>>2);group<groups;group+=32)
                value+=scales[group]*terms[group][0];
            #pragma unroll
            for(int step=16;step;step>>=1)value+=__shfl_xor_sync(0xffffffffu,value,step);
            if(!lane)partial[warp]=value;
        }
        __syncthreads();
        if(warp)return;
        value=lane<4?partial[lane]:0.f;
        #pragma unroll
        for(int step=16;step;step>>=1)value+=__shfl_xor_sync(0xffffffffu,value,step);
    }
    if(!lane) {
        const float dot=__half2float(__float2half_rn(value));
        output[blockIdx.x]=__float2half_rn(1.f/(1.f+expf(-dot)));
    }
}
}

std::optional<mfq_tensor_backend::Tensor> try_nint_shared_gate_sigmoid_cuda(
        const mfq_tensor_backend::Tensor& input,const NintWeight& w,
        const mfq_tensor_backend::Tensor& quantized,const mfq_tensor_backend::Tensor& scales) {
    namespace tb=mfq_tensor_backend;
    if(!input.defined() || !input.is_cuda() || !input.is_contiguous() || input.dim()!=2 ||
            input.scalar_type()!=tb::kFloat16 || input.size(0)<1 || input.size(0)>6 ||
            !w.aligned_q8 || w.q8_zero || w.out!=1 || w.gs!=48 || w.ng<1 || w.ng>64 ||
            input.size(1)!=w.neuron_len || w.neuron_len>w.ng*48 ||
            !w.q_packed.is_cuda() || w.q_packed.device()!=input.device())return std::nullopt;
    // Explicit changes to the reference arithmetic keep their original path.
    if(const auto* option=std::getenv("TYLOINFER_NINT_Q8_DECODE"))if(std::atoi(option)!=2)return std::nullopt;
    if(input.size(0)==1) {
        if(const auto* option=std::getenv("TYLOINFER_NINT_Q8_ORDERED"))if(option[0]=='0')return std::nullopt;
    } else if(const auto* option=std::getenv("TYLOINFER_NINT_Q8_SMALL_WARPS"))if(option[0]=='0')return std::nullopt;
    for(const auto* tensor:{&w.q_packed,&w.row_q_bit_offsets,&w.sub_scale,&w.sub_min,&w.neuron_scale,&w.neuron_min})
        if(!tensor->is_cuda() || !tensor->is_contiguous() || tensor->device()!=input.device())return std::nullopt;
    if(quantized.defined() && (!scales.defined() || quantized.scalar_type()!=tb::kInt8 ||
            scales.scalar_type()!=tb::kFloat32 || !quantized.is_contiguous() || !scales.is_contiguous() ||
            quantized.device()!=input.device() || scales.device()!=input.device() ||
            quantized.numel()!=input.size(0)*w.ng*48 || scales.numel()!=input.size(0)*w.ng))return std::nullopt;
    auto output=tb::empty({input.size(0),1},input.options());
    mfq::cuda::DeviceGuard guard(input.get_device());
    const auto launch=[&](auto ordered) {
        if(quantized.defined()) {
            nint_gate_prepared<decltype(ordered)::value><<<int(input.size(0)),128,0,mfq_current_cuda_stream()>>>(
                quantized.data_ptr<int8_t>(),scales.data_ptr<float>(),w.q_packed.data_ptr<uint8_t>(),
                w.row_q_bit_offsets.data_ptr<int64_t>(),w.sub_scale.data_ptr<uint8_t>(),w.sub_min.data_ptr<uint8_t>(),
                w.neuron_scale.data_ptr<float>(),w.neuron_min.data_ptr<float>(),static_cast<__half*>(output.data_ptr()),int(w.ng));
            return;
        }
        nint_gate_sigmoid<decltype(ordered)::value><<<int(input.size(0)),1024,0,mfq_current_cuda_stream()>>>(
            static_cast<const __half*>(input.data_ptr()),w.q_packed.data_ptr<uint8_t>(),
            w.row_q_bit_offsets.data_ptr<int64_t>(),w.sub_scale.data_ptr<uint8_t>(),w.sub_min.data_ptr<uint8_t>(),
            w.neuron_scale.data_ptr<float>(),w.neuron_min.data_ptr<float>(),
            static_cast<__half*>(output.data_ptr()),int(input.size(1)),int(w.ng));
    };
    if(input.size(0)==1)launch(std::true_type{});else launch(std::false_type{});
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());return output;
}

#endif

std::optional<mfq_tensor_backend::Tensor> try_shared_gate_sigmoid_cuda(
        const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& weight) {
    namespace tb=mfq_tensor_backend;
    if(!input.defined() || !weight.defined() || !input.is_cuda() || !weight.is_cuda() ||
        input.get_device()!=weight.get_device() || !input.is_contiguous() || !weight.is_contiguous() ||
        input.scalar_type()!=tb::kFloat16 || weight.scalar_type()!=tb::kBFloat16 ||
        input.dim()!=2 || weight.dim()!=2 || weight.size(0)!=1 ||
        input.size(0)<1 || input.size(1)<1 ||
        input.size(1)!=weight.size(1))return std::nullopt;
    mfq::cuda::DeviceGuard guard(input.get_device());
    int maximum_grid=0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&maximum_grid,cudaDevAttrMaxGridDimX,input.get_device()));
    if(input.size(0)>maximum_grid)return std::nullopt;
    auto output=tb::empty({input.size(0),1},input.options());
    const auto stream=mfq::cuda::current_stream(input.get_device()).stream();
    mfq::cuda::shared_gate_detail::project_sigmoid<128,true,false,true>
        <<<int(input.size(0)),128,0,stream>>>(
            static_cast<const __half*>(input.data_ptr()),
            static_cast<const __nv_bfloat16*>(weight.data_ptr()),nullptr,
            static_cast<__half*>(output.data_ptr()),input.size(1));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"shared_gate_sigmoid",
        reinterpret_cast<const void*>(mfq::cuda::shared_gate_detail::project_sigmoid<128,true,false,true>),
        "rows="+std::to_string(input.size(0))+" K="+std::to_string(input.size(1))+" output_half=1");
    return output;
}
