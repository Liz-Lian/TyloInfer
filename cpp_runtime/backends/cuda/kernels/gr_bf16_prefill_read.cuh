#pragma once

// Keep only the BF16 projection image and one FP32 RMS value per stream.
// Injection and the fused up/mix reconstruct the original FP32 normalized
// operands from the residual, preserving the norm's offset and rounding.
namespace {
template<int Threads,class Residual=float,class Branch=float,class Injection=float,bool After=false>
__global__ void gr_bf16_prefill_norm_kernel(const Residual* input,const float* weight,
        float* updated,uint16_t* projection,float* rms,int hidden,int streams,float eps,
        const Branch* branch,const Injection* injection) {
    const int64_t group=blockIdx.x,base=group*hidden;
    const auto* scale=weight+(group%streams)*hidden;
    float sum=0;
    for(int64_t c=threadIdx.x;c<hidden;c+=Threads) {
        float value=float(input[base+c]);
        if constexpr(After) {
            float product=__fmul_rn(float(branch[group/streams*hidden+c]),float(injection[group]));
            if constexpr(std::is_same_v<Branch,__half> && std::is_same_v<Injection,__half>)
                product=__half2float(__float2half_rn(product));
            value=__fadd_rn(value,product);updated[base+c]=value;
        }
        sum=__fadd_rn(sum,__fmul_rn(value,value));
    }
    __shared__ float partial[Threads];partial[threadIdx.x]=sum;__syncthreads();
    for(int stride=Threads/2;stride>0;stride/=2) {
        if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    const float inverse=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(partial[0],float(hidden)),eps)));
    if(!threadIdx.x)rms[group]=inverse;
    if(projection)for(int64_t c=threadIdx.x;c<hidden;c+=Threads) {
        const float value=__fmul_rn(__fmul_rn(updated[base+c],inverse),__fadd_rn(scale[c],1.f));
        projection[base+c]=__bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
}
template<bool Injection>
__global__ void gr_bf16_prefill_projection_kernel(const float* input,const float* norm,const float* rms,
        const uint16_t* weight,float* output,int hidden,int streams,int outputs) {
    // Gate varies fastest, so neighboring blocks reuse a residual row in L2.
    const int64_t row=blockIdx.x/outputs;const int gate=blockIdx.x%outputs;
    const int64_t width=int64_t(hidden)*streams;
    float sum=0;
    for(int stream=0;stream<streams;++stream) {
        const auto base=int64_t(stream)*hidden;
        const float inverse=rms[row*streams+stream];
        for(int64_t c=threadIdx.x;c<hidden;c+=256) {
            const float value=__fmul_rn(__fmul_rn(input[row*width+base+c],inverse),__fadd_rn(norm[base+c],1.f));
            const float w=__uint_as_float(uint32_t(weight[int64_t(gate)*width+base+c])<<16);
            sum=fmaf(value,w,sum);
        }
    }
    __shared__ float partial[256];partial[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x) {
        float value=partial[0];
        if constexpr(Injection)value=__fmul_rn(2.f,1.f/(1.f+expf(-__fdiv_rn(value,float(streams)))));
        output[row*outputs+gate]=value;
    }
}
}

std::vector<mfq_tensor_backend::Tensor> try_gated_residual_bf16_prefill_cuda(
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& norm,
        const mfq_tensor_backend::Tensor& down,const mfq_tensor_backend::Tensor& up,
        const mfq_tensor_backend::Tensor& injection,int64_t streams,double eps,
        const mfq_tensor_backend::Tensor& branch,const mfq_tensor_backend::Tensor& prior,
        const GatedResidualMarker& marker) {
    namespace tb=mfq_tensor_backend;
    const auto* setting=std::getenv("TYLOINFER_GR_PREFILL_BF16");
    if(setting && setting[0]=='0')return {};
    const auto value=[](const tb::Tensor& t){return t.defined() && t.is_cuda() &&
        (t.scalar_type()==tb::kFloat16 || t.scalar_type()==tb::kFloat32);};
    if(!value(input) || input.dim()<1 || streams<1 || streams>INT_MAX-gr_mix_streams || input.size(-1)<streams ||
            input.size(-1)>INT_MAX || input.size(-1)<16 || input.size(-1)%streams || !norm.defined() || !norm.is_cuda() ||
            norm.device()!=input.device() || norm.scalar_type()!=tb::kFloat32 || norm.dim()!=1 ||
            norm.numel()!=input.size(-1))return {};
    const auto rows=input.numel()/input.size(-1),hidden=input.size(-1)/streams,rank=down.defined()&&down.dim()==2?down.size(0):0;
    if(rows<=32 || rows>int64_t(65535)*gr_mix_bm || rows>INT_MAX/streams || rank<1 || rank>INT_MAX-gr_mix_kc)return {};
    for(const auto& w:{down,up,injection})if(!w.defined() || !w.is_cuda() || w.device()!=input.device() ||
        w.scalar_type()!=tb::kBFloat16 || w.dim()!=2)return {};
    if(down.size(1)!=input.size(-1) || up.size(0)!=input.size(-1) || up.size(1)!=rank ||
            injection.size(0)!=streams || injection.size(1)!=input.size(-1))return {};
    if(branch.defined()) {
        auto shape=input.sizes().vec();shape.back()=hidden;
        auto gates=input.sizes().vec();gates.back()=streams;
        if(!value(branch) || !value(prior) || branch.device()!=input.device() || prior.device()!=input.device() ||
                branch.sizes().vec()!=shape || prior.sizes().vec()!=gates)return {};
        if(input.scalar_type()!=tb::kFloat32 && branch.scalar_type()!=tb::kFloat32 && prior.scalar_type()!=tb::kFloat32)return {};
    } else if(prior.defined() || input.scalar_type()!=tb::kFloat32)return {};
    const MfqCudaGuard guard(input.get_device());
    int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
    if(major<8)return {}; // BF16 Tensor Core projections and cp.async in up/mix.
    const auto stream=mfq_current_cuda_stream();
    if(marker)marker("begin");
    auto source=input.contiguous();
    auto branch_values=branch.defined()?branch.contiguous():tb::Tensor{};
    auto prior_values=prior.defined()?prior.contiguous():tb::Tensor{};
    auto updated=branch.defined()?tb::empty(source.sizes(),source.options().dtype(tb::kFloat32)):source;
    auto scale=norm.contiguous();
    auto image=rank>=16 || streams>=16?tb::empty(updated.sizes(),updated.options().dtype(tb::kBFloat16)):tb::Tensor{};
    auto rms=tb::empty({rows,streams},updated.options());
    const auto typed_norm=[&](auto residual_type,auto branch_type,auto injection_type,auto after) {
        using Residual=decltype(residual_type);using Branch=decltype(branch_type);using Injection=decltype(injection_type);
        const auto launch=[&](auto threads) {
            constexpr int count=decltype(threads)::value;
            gr_bf16_prefill_norm_kernel<count,Residual,Branch,Injection,decltype(after)::value>
                <<<unsigned(rows*streams),count,0,stream>>>(source.data_ptr<Residual>(),scale.data_ptr<float>(),updated.data_ptr<float>(),
                image.defined()?static_cast<uint16_t*>(image.data_ptr()):nullptr,rms.data_ptr<float>(),int(hidden),int(streams),float(eps),
                branch_values.defined()?branch_values.data_ptr<Branch>():nullptr,prior_values.defined()?prior_values.data_ptr<Injection>():nullptr);
        };
        if(hidden<=32)launch(std::integral_constant<int,32>{});
        else if(hidden<=64)launch(std::integral_constant<int,64>{});
        else if(hidden<=128)launch(std::integral_constant<int,128>{});
        else launch(std::integral_constant<int,256>{});
    };
    if(!branch.defined())typed_norm(float{},float{},float{},std::false_type{});
    else {
        const auto residual_type=[&](auto b,auto i) {
            if(source.scalar_type()==tb::kFloat16)typed_norm(__half{},b,i,std::true_type{});
            else typed_norm(float{},b,i,std::true_type{});
        };
        const auto injection_type=[&](auto b) {
            if(prior_values.scalar_type()==tb::kFloat16)residual_type(b,__half{});
            else residual_type(b,float{});
        };
        if(branch_values.scalar_type()==tb::kFloat16)injection_type(__half{});else injection_type(float{});
    }
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    if(marker)marker("norm");
    tb::Tensor low;
    if(rank>=16) {
        low=try_gr_bf16_prefill_projection_cuda(image,down);
        if(!low.defined())return {};
    } else {
        // The existing narrow projection uses FP32 normalized operands.
        // Reconstruct those here instead of rounding them to BF16.
        auto dw=down.contiguous();
        auto shape=input.sizes().vec();shape.back()=rank;low=tb::empty(shape,input.options().dtype(tb::kFloat32));
        gr_bf16_prefill_projection_kernel<false><<<unsigned(rows*rank),256,0,stream>>>(updated.data_ptr<float>(),
            scale.data_ptr<float>(),rms.data_ptr<float>(),static_cast<const uint16_t*>(dw.data_ptr()),low.data_ptr<float>(),int(hidden),int(streams),int(rank));
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
        dw.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    }
    if(marker)marker("down");
    low=mfq_qwen4_exp::gated_residual_bottleneck(low,streams);
    if(marker)marker("bottleneck");
    auto mixed=try_gr_bf16_prefill_mix(low,updated,up,streams,scale,rms);
    if(!mixed.defined())throw std::runtime_error("BF16 GR compact prefill mix rejected validated geometry");
    if(marker)marker("up_mix");
    auto gate_shape=input.sizes().vec();gate_shape.back()=streams;
    tb::Tensor gates;
    auto iw=injection.contiguous();
    if(streams>=16) {
        auto projected=try_gr_bf16_prefill_projection_cuda(image,iw);
        if(!projected.defined())return {};
        gates=mfq_qwen4_exp::gated_residual_injection(projected,streams);
    } else {
        gates=tb::empty(gate_shape,input.options().dtype(tb::kFloat32));
        gr_bf16_prefill_projection_kernel<true><<<unsigned(rows*streams),256,0,stream>>>(updated.data_ptr<float>(),
            scale.data_ptr<float>(),rms.data_ptr<float>(),static_cast<const uint16_t*>(iw.data_ptr()),gates.data_ptr<float>(),int(hidden),int(streams),int(streams));
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    for(const auto& t:{source,branch_values,prior_values,updated,scale,image,rms,low,mixed,iw,gates})
        if(t.defined())t.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    if(marker){marker("inject");marker("end");}
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::once_flag reported;
        std::call_once(reported,[&] {std::fprintf(stderr,
            "native_gr_prefill_read actual_match=1 M=%lld hidden=%lld K=%lld streams=%lld normalized_f32=0 materialized_projection=0\n",
            static_cast<long long>(rows),static_cast<long long>(hidden),static_cast<long long>(rank),static_cast<long long>(streams));});
    }
    return {mixed,updated,gates};
}
