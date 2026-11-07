#include "gated.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/kernels/selected_attention.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/allocation.h"
#include "engine/backends/cuda/kernels/gated_residual_fused.h"
#include "engine/backends/cuda/kernels/float_projection.h"
#include "engine/backends/cuda/executor/options.h"

#include <algorithm>
#include <cstdio>

namespace tyloi::cuda {

GatedResidual::Projection GatedResidual::projection(
        std::shared_ptr<const ::tyloi::Tensor> weight, Projector project, CudaProfiler& profiler,
        const Options& options, Part part) {
    using Projection = GatedResidual::Projection;
    const tyloi::cuda::AllocationScope weight_memory;
    const auto* payload = weight.get();
    TYLOINFER_RUNTIME_CHECK(
        dynamic_cast<const DenseTensor*>(payload) != nullptr ||
            dynamic_cast<const tyloi::NintTensor*>(payload) != nullptr ||
            dynamic_cast<const tyloi::encoding::TqTensor*>(payload) != nullptr,
        "unsupported floating residual projection");
    if (dynamic_cast<const tyloi::encoding::TqTensor*>(payload) != nullptr) {
        Projection::Forward forward=[weight,project,&profiler](const DenseTensor& input) {
            // TQ reconstructs FP16 weights. A BF16/F32 activation therefore
            // promotes residual projection math to FP32, as the dense path.
            return profiler.measure("tq.residual",[&] {
                return project.run(*weight,
                    input.scalar_type()==::tyloi::kFloat16?input:input.to(::tyloi::kFloat32));
            });
        };
        return {std::move(forward),{},{},std::move(weight),{}};
    }
    {
        const auto& dense = payload_as<::tyloi::DenseTensor>(*weight);
        auto right=dense.transpose(-1,-2);
        auto promoted=options.prepared_dense?right.to(::tyloi::kFloat32):DenseTensor{};
        const bool prepare_injection=options.dense_injection &&
            dense.scalar_type()==::tyloi::kBFloat16 && dense.is_contiguous() &&
            dense.dim()==2 && dense.size(0)>0 && dense.size(0)<=4;
        auto prepared_right=prepare_injection?(promoted.defined()?promoted:right.to(::tyloi::kFloat32)):DenseTensor{};
        const auto vector_setting=::tyloi::cuda::options::gr_dense_vector();
        if(prepare_injection && (!vector_setting || *vector_setting) &&
                !promoted.defined() && right.size(0)%4==0)
            prepared_right=prepare_gr_dense_vector_right(prepared_right);
        Projection::Forward forward=[right=std::move(right),promoted=std::move(promoted),
                &options](const DenseTensor& input) {
            const auto dtype=input.scalar_type()==right.scalar_type()?input.scalar_type(): ::tyloi::kFloat32;
            if(options.prepared_dense && promoted.defined() && dtype==::tyloi::kFloat32)
                return ::tyloi::cuda::matmul(input.to(::tyloi::kFloat32),promoted);
            return mfq_selected_attention::promoted_matmul(input,right);
        };
        Projection::Activation activated;
        Projection::Mix mixed;
        if(dense.scalar_type()==::tyloi::kBFloat16 && dense.is_contiguous() &&
                part==Part::final_down)
            activated=[weight,forward](const DenseTensor& input,int64_t streams,bool injection) {
                const auto option=::tyloi::cuda::options::gr_dense_bf16_direct();
                if(!injection && (!option || *option))
                    if(auto output=try_gr_bf16_down_cuda(input,payload_as<::tyloi::DenseTensor>(*weight),streams);output.defined())return output;
                auto projected=forward(input);
                return injection?mfq_qwen4_exp::gated_residual_injection(projected,streams):
                    mfq_qwen4_exp::gated_residual_bottleneck(projected,streams);
            };
        if(dense.scalar_type()==::tyloi::kBFloat16 && dense.is_contiguous() &&
                part==Part::final_up)
            mixed=[weight,forward](const DenseTensor& input,const DenseTensor& normalized,int64_t streams) {
                const auto option=::tyloi::cuda::options::gr_dense_bf16_direct();
                if(!option || *option)
                    if(auto output=try_gr_bf16_up_mix_cuda(input,normalized,payload_as<::tyloi::DenseTensor>(*weight),streams);output.defined())return output;
                return mfq_qwen4_exp::gated_residual_mix(forward(input),normalized,streams);
            };
        return {std::move(forward),std::move(mixed),std::move(activated),std::move(weight),std::move(prepared_right)};
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT

    Projection::Forward forward=[weight,&profiler,&options](const DenseTensor& input) {
        auto output = profiler.measure("nint.float_projection", [&] {
            return nint_float_projection_cuda(payload_as<tyloi::NintTensor>(*weight), input,true,options.native_input,
                options.fixed_group);
        });
        // Match Metal's promotion of NINT's logical F16 weight dtype.
        return input.scalar_type() == ::tyloi::kFloat16 ? output.to(::tyloi::kFloat16) : output;
    };
    Projection::Mix mixed;
    if(dynamic_cast<const tyloi::NintTensor*>(weight.get()) != nullptr)mixed=[weight,&profiler,&options](const DenseTensor& input,
        const DenseTensor& normalized,int64_t streams) {
        return profiler.measure("nint.float_projection_mix",[&] {
            return nint_float_projection_mix_cuda(payload_as<tyloi::NintTensor>(*weight),input,normalized,streams,options.native_input,
                options.fixed_group,options.compact_mix);
        });
    };
    Projection::Activation activated;
    if(dynamic_cast<const tyloi::NintTensor*>(weight.get()) != nullptr)activated=[weight,&profiler,&options](const DenseTensor& input,
        int64_t streams,bool injection) {
        return profiler.measure("nint.float_projection_activation",[&] {
            return nint_float_projection_activation_cuda(payload_as<tyloi::NintTensor>(*weight),input,streams,injection,true,
                options.native_input,options.fixed_group);
        });
    };
    DenseTensor group_coefficients;
    const auto group_option=::tyloi::cuda::options::gr_up_group_coefficients();
    if(group_option && *group_option!='0' && payload_as<tyloi::NintTensor>(*weight).aligned_q8 && !payload_as<tyloi::NintTensor>(*weight).q8_zero &&
            payload_as<tyloi::NintTensor>(*weight).gs==48 && payload_as<tyloi::NintTensor>(*weight).out==10240 && payload_as<tyloi::NintTensor>(*weight).neuron_len==320)
        group_coefficients=*group_option=='2'?prepare_gr_up_packed_metadata(payload_as<tyloi::NintTensor>(*weight)):
            prepare_gr_up_group_coefficients(payload_as<tyloi::NintTensor>(*weight));
    return {std::move(forward),std::move(mixed),std::move(activated),std::move(weight),DenseTensor{},std::move(group_coefficients)};

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

int64_t GatedResidual::prefill_work_chunk(const DenseTensor& input)const {
    if(input.dim()!=3 || input.size(0)!=1)return 0;
    const auto option=::tyloi::cuda::options::gr_prefill_work_chunk();
#ifdef TYLOINFER_NATIVE_CUDA
    const int64_t chunk=option?*option:8192;
    // Chat framing just beyond 8k needs no large temporary-memory cut.
    // Avoid extra concatenation there; longer inputs use 8k work tiles.
    if(!option && input.size(1)<=9216)return 0;
#else
    const int64_t chunk=option?*option:0;
#endif
    return chunk>=256 && input.size(1)>chunk?chunk:0;
}
template<class Project>
GatedResidual::Result GatedResidual::bounded_pre(const DenseTensor& input,int64_t chunk,bool reuse_input,Project project)const {
    const auto tokens=input.size(1);
    Result output;
    if(reuse_input)output.residual=input;
    for(int64_t begin=0;begin<tokens;) {
        auto count=std::min(chunk,tokens-begin);
        const auto tail=tokens-begin-count;
        // Both pieces must retain the main GR prefill kernels; a tiny
        // remainder selects different projection arithmetic.
        if(tail>0 && tail<128)count-=128-tail;
        auto part=project(begin,count);
        const auto append=[&](DenseTensor& destination,const DenseTensor& source) {
            if(!source.defined())return;
            if(!destination.defined()) {
                auto shape=source.sizes().vec();shape[1]=tokens;
                destination=::tyloi::cuda::empty(shape,source.options());
            }
            destination.narrow(1,begin,count).copy_(source);
        };
        append(output.mixed,part.mixed);
        if(!reuse_input)append(output.residual,part.residual);
        append(output.gate,part.gate);
        begin+=count;
    }
    return output;
}
bool GatedResidual::can_fuse(const DenseTensor& input)const {
    if(!options_)return false;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto* down_nint = down.weight
        ? dynamic_cast<const tyloi::NintTensor*>(down.weight.get()) : nullptr;
    const auto* up_nint = up.weight
        ? dynamic_cast<const tyloi::NintTensor*>(up.weight.get()) : nullptr;
    if(!options_->two_stage || !options_->fused_projections ||
        !options_->native_input || down_nint == nullptr || up_nint == nullptr ||
        streams<1 || streams>4 ||
        input.dim()<1 || input.size(-1)!=hidden*streams || input.numel()<hidden*streams ||
        input.numel()/(hidden*streams)>32 || !input.is_cuda() || !input.is_contiguous() ||
        (input.scalar_type()!=::tyloi::kFloat16 && input.scalar_type()!=::tyloi::kFloat32) ||
        !norm.is_contiguous() || norm.scalar_type()!=::tyloi::kFloat32)return false;
    const auto* injection_nint = injection.weight
        ? dynamic_cast<const tyloi::NintTensor*>(injection.weight.get()) : nullptr;
    const auto* injection_dense = injection.weight
        ? dynamic_cast<const DenseTensor*>(injection.weight.get()) : nullptr;
    if(injection_nint == nullptr && !(options_->dense_injection && injection_dense &&
        injection_dense->is_contiguous() &&
        injection_dense->scalar_type()==::tyloi::kBFloat16))return false;
    const auto& w=*up_nint;
    const auto rows=input.numel()/(hidden*streams);
    const auto continuous=::tyloi::cuda::options::gr_continuous_two_stage();
    if(rows>8 && (!*cache_serial_ || (continuous && !*continuous)))
        return false;
    const auto batch=::tyloi::cuda::options::gr_batch_two_stage();
    if(rows>1 && batch && !*batch)return false;
    const bool selected=w.ng==1 || ((w.out+3)/4)*rows>=nint_float_projection_resident_blocks(input.get_device());
    const auto trace=::tyloi::cuda::options::trace_gr_dispatch();
    if(selected && trace && *trace) {
        static thread_local uint64_t seen[2]{};
        const int kind=input.scalar_type()==::tyloi::kFloat16?0:1;
        if(!(seen[kind]&(uint64_t(1)<<rows))) {
            seen[kind]|=uint64_t(1)<<rows;
            std::fprintf(stderr,"gr_two_stage rows=%lld half=%d continuous=%d\n",
                static_cast<long long>(rows),kind==0,int(*cache_serial_));
        }
    }
    return selected;
#else
    return false;
#endif
}
GatedResidual::Result GatedResidual::fused(const DenseTensor& input,const DenseTensor& residual,const DenseTensor& gate,bool after,
    const GatedResidualMarker& marker)const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto& injection_payload = *injection.weight;
    const auto* injection_nint = dynamic_cast<const tyloi::NintTensor*>(&injection_payload);
    const auto* injection_dense = dynamic_cast<const DenseTensor*>(&injection_payload);
    auto result=gated_residual_two_stage_cuda(input,residual,gate,norm,
        payload_as<tyloi::NintTensor>(*down.weight),
        payload_as<tyloi::NintTensor>(*up.weight),
        injection_nint,streams,eps,after,
        injection_dense ? *injection_dense : DenseTensor{},injection.prepared_right,marker,up.up_group_coefficients);
    return {std::move(result[0]),std::move(result[1]),std::move(result[2])};
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
GatedResidual::Result GatedResidual::pre(const DenseTensor &input,
    const GatedResidualMarker& marker) const {
    if(const auto chunk=prefill_work_chunk(input))
        return bounded_pre(input,chunk,true,[&](int64_t begin,int64_t count) {
            return pre(input.narrow(1,begin,count),marker);
        });
    if(can_fuse(input))return fused({},input,{},false,marker);
    return prepare(input,{},marker);
}
DenseTensor GatedResidual::post(const DenseTensor& input,const DenseTensor& residual,const DenseTensor& gate) const {
    return mfq_qwen4_exp::gated_residual_post(input,residual,gate,streams);
}
GatedResidual::Result GatedResidual::pre_after(const DenseTensor& input,
    const DenseTensor& residual,const DenseTensor& gate,const GatedResidualMarker& marker) const {
    if(const auto chunk=prefill_work_chunk(residual))
        return bounded_pre(residual,chunk,false,[&](int64_t begin,int64_t count) {
            return pre_after(input.narrow(1,begin,count),residual.narrow(1,begin,count),
                gate.narrow(1,begin,count),marker);
        });
    if(can_fuse(residual) && input.is_contiguous() && gate.is_contiguous() &&
        (input.scalar_type()==::tyloi::kFloat16 || input.scalar_type()==::tyloi::kFloat32) &&
        (gate.scalar_type()==::tyloi::kFloat16 || gate.scalar_type()==::tyloi::kFloat32))
        return fused(input,residual,gate,true,marker);
    TYLOINFER_RUNTIME_CHECK(options_,"gated-residual operator is not initialized");
    if(!options_->fused_projections || !options_->fused_post_norm) {
        return prepare(post(input,residual,gate),{},marker);
    }
    TYLOINFER_RUNTIME_CHECK(input.size(-1)==hidden,"gated-residual chained hidden width disagrees");
    if(marker)marker("begin");
    auto prepared=mfq_qwen4_exp::gated_residual_post_norm(input,residual,gate,norm,streams,eps);
    return prepare(prepared[0],prepared[1],marker);
}

GatedResidual::Result GatedResidual::prepare(
        const DenseTensor& input, const DenseTensor& prepared_norm, const GatedResidualMarker& marker) const {
    TYLOINFER_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 && options_ && down && up,
        "gated-residual projection input dimensions disagree");
    if(marker && !prepared_norm.defined())marker("begin");
    auto normalized = prepared_norm.defined()?prepared_norm:
        mfq_qwen4_exp::grouped_rms_norm(input, norm, hidden, eps);
    if(marker)marker("norm");
    const bool fused=options_->fused_projections;
    const bool epilogue=fused && options_->fused_activation;
    const bool down_activated=epilogue && bool(down.activated);
    auto low = down_activated?down.activated(normalized,streams,false):down.forward(normalized);
    if(marker)marker("down");
    TYLOINFER_RUNTIME_CHECK(low.dim() == input.dim() && low.size(-1) > 0,
        "gated-residual bottleneck rank disagrees");
    if(!down_activated) {
        if(fused)low=mfq_qwen4_exp::gated_residual_bottleneck(low,streams);
        else {low=low/streams;low=low*::tyloi::cuda::sigmoid(low);}
    }
    if(marker)marker("bottleneck");
    auto mixed = [&] {
        if(fused && up.mixed)return up.mixed(low,normalized,streams);
        auto mixing=up.forward(low);
        TYLOINFER_RUNTIME_CHECK(mixing.sizes()==input.sizes(),"gated-residual mixing shape disagrees");
        return mfq_qwen4_exp::gated_residual_mix(mixing,normalized,streams);
    }();
    if(marker)marker("up_mix");
    ::tyloi::DenseTensor gate;
    if (injection.forward) {
        if(epilogue && injection.activated) {
            gate=injection.activated(normalized,streams,true);
            if(marker)marker("inject");
        } else {
            auto projected=injection.forward(normalized);
            if(marker)marker("inject");
            gate=fused?mfq_qwen4_exp::gated_residual_injection(projected,streams):
                2.0*::tyloi::cuda::sigmoid(projected/streams);
        }
        auto expected = input.sizes().vec();
        expected.back() = streams;
        TYLOINFER_RUNTIME_CHECK(gate.sizes().vec() == expected,
            "gated-residual injection shape disagrees");
    }
    if(marker)marker("end");
    return {mixed, input, gate};
}

} // namespace tyloi::cuda
