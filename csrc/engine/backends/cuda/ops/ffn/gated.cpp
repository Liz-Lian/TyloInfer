#ifndef TYLOINFER_NATIVE_CUDA
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
#include "engine/backends/cuda/executor/executor.h"

#include <algorithm>
#include <cstdio>

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA
GatedResidual::Projection GatedResidual::projection(::tyloi::Tensor weight, Part part) {
    const tyloi::cuda::AllocationScope weight_memory;
    const auto* payload = weight.implementation().get();
    TYLOINFER_RUNTIME_CHECK(
        dynamic_cast<const DenseTensor*>(payload) != nullptr ||
            dynamic_cast<const tyloi::NintTensor*>(payload) != nullptr ||
            dynamic_cast<const tyloi::encoding::TqTensor*>(payload) != nullptr,
        "unsupported floating residual projection");
    Projection result;
    result.weight = std::move(weight);
    if (dynamic_cast<const tyloi::encoding::TqTensor*>(payload) != nullptr)
        return result;
    {
        const auto& options = Executor::current().options().residual;
        const auto& dense = payload_as<DenseTensor>(result.weight);
        result.right = dense.transpose(-1,-2);
        result.promoted = options.prepared_dense ? result.right.to(::tyloi::kFloat32) : DenseTensor{};
        const bool prepare_injection = options.dense_injection &&
            dense.scalar_type() == ::tyloi::kBFloat16 && dense.is_contiguous() &&
            dense.dim() == 2 && dense.size(0) > 0 && dense.size(0) <= 4;
        if (prepare_injection)
            result.prepared_right = result.promoted.defined() ? result.promoted : result.right.to(::tyloi::kFloat32);
        const auto vector_setting = ::tyloi::cuda::options::gr_dense_vector();
        if (prepare_injection && (!vector_setting || *vector_setting) &&
            !result.promoted.defined() && result.right.size(0) % 4 == 0)
            result.prepared_right = prepare_gr_dense_vector_right(result.prepared_right);
        const bool direct = dense.scalar_type() == ::tyloi::kBFloat16 && dense.is_contiguous();
        result.can_activate = direct && part == Part::final_down;
        result.can_mix = direct && part == Part::final_up;
        return result;
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    result.can_mix = result.can_activate = result.weight.representation<tyloi::NintTensor>() != nullptr;
    const auto group_option = ::tyloi::cuda::options::gr_up_group_coefficients();
    const auto& nint = payload_as<tyloi::NintTensor>(result.weight);
    if (group_option && *group_option != '0' && nint.aligned_q8 && !nint.q8_zero &&
        nint.gs == 48 && nint.out == 10240 && nint.neuron_len == 320)
        result.up_group_coefficients = *group_option == '2' ? prepare_gr_up_packed_metadata(nint)
                                                           : prepare_gr_up_group_coefficients(nint);
    return result;
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

GatedResidual::DenseTensor GatedResidual::Projection::forward(const DenseTensor& input) const {
    auto& executor = Executor::current();
    const auto& options = executor.options().residual;
    if (weight.representation<tyloi::encoding::TqTensor>()) {
        return executor.profiler().measure("tq.residual", [&] {
            return executor.projector(Executor::current_ffn()).run(weight,
                input.scalar_type() == ::tyloi::kFloat16 ? input : input.to(::tyloi::kFloat32));
        });
    }
    if (weight.representation<DenseTensor>()) {
        const auto dtype = input.scalar_type() == right.scalar_type() ? input.scalar_type() : ::tyloi::kFloat32;
        if (options.prepared_dense && promoted.defined() && dtype == ::tyloi::kFloat32)
            return ::tyloi::cuda::matmul(input.to(::tyloi::kFloat32), promoted);
        return mfq_selected_attention::promoted_matmul(input, right);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    auto output = executor.profiler().measure("nint.float_projection", [&] {
        return nint_float_projection_cuda(payload_as<tyloi::NintTensor>(weight), input, true,
                                           options.native_input, options.fixed_group);
    });
    return input.scalar_type() == ::tyloi::kFloat16 ? output.to(::tyloi::kFloat16) : output;
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

GatedResidual::DenseTensor GatedResidual::Projection::mix(
        const DenseTensor& input, const DenseTensor& normalized, int64_t streams) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* nint = weight.representation<tyloi::NintTensor>()) {
        auto& executor = Executor::current();
        const auto& options = executor.options().residual;
        return executor.profiler().measure("nint.float_projection_mix", [&] {
            return nint_float_projection_mix_cuda(*nint, input, normalized, streams,
                options.native_input, options.fixed_group, options.compact_mix);
        });
    }
#endif
    const auto option = ::tyloi::cuda::options::gr_dense_bf16_direct();
    if (!option || *option)
        if (auto output = try_gr_bf16_up_mix_cuda(input, normalized, payload_as<DenseTensor>(weight), streams); output.defined())
            return output;
    return mfq_qwen4_exp::gated_residual_mix(forward(input), normalized, streams);
}

GatedResidual::DenseTensor GatedResidual::Projection::activate(
        const DenseTensor& input, int64_t streams, bool injection) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (const auto* nint = weight.representation<tyloi::NintTensor>()) {
        auto& executor = Executor::current();
        const auto& options = executor.options().residual;
        return executor.profiler().measure("nint.float_projection_activation", [&] {
            return nint_float_projection_activation_cuda(*nint, input, streams, injection, true,
                options.native_input, options.fixed_group);
        });
    }
#endif
    const auto option = ::tyloi::cuda::options::gr_dense_bf16_direct();
    if (!injection && (!option || *option))
        if (auto output = try_gr_bf16_down_cuda(input, payload_as<DenseTensor>(weight), streams); output.defined())
            return output;
    auto projected = forward(input);
    return injection ? mfq_qwen4_exp::gated_residual_injection(projected, streams)
                     : mfq_qwen4_exp::gated_residual_bottleneck(projected, streams);
}
#else
GatedResidual::Projection GatedResidual::projection(
        ::tyloi::Tensor weight, Projector project, CudaProfiler& profiler,
        const ResidualOptions& options, Part part) {
    using Projection = GatedResidual::Projection;
    const tyloi::cuda::AllocationScope weight_memory;
    const auto* payload = weight.implementation().get();
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
                return project.run(weight,
                    input.scalar_type()==::tyloi::kFloat16?input:input.to(::tyloi::kFloat32));
            });
        };
        return {std::move(forward),{},{},std::move(weight),{}};
    }
    {
        const auto& dense = payload_as<::tyloi::DenseTensor>(weight);
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
                    if(auto output=try_gr_bf16_down_cuda(input,payload_as<::tyloi::DenseTensor>(weight),streams);output.defined())return output;
                auto projected=forward(input);
                return injection?mfq_qwen4_exp::gated_residual_injection(projected,streams):
                    mfq_qwen4_exp::gated_residual_bottleneck(projected,streams);
            };
        if(dense.scalar_type()==::tyloi::kBFloat16 && dense.is_contiguous() &&
                part==Part::final_up)
            mixed=[weight,forward](const DenseTensor& input,const DenseTensor& normalized,int64_t streams) {
                const auto option=::tyloi::cuda::options::gr_dense_bf16_direct();
                if(!option || *option)
                    if(auto output=try_gr_bf16_up_mix_cuda(input,normalized,payload_as<::tyloi::DenseTensor>(weight),streams);output.defined())return output;
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
    if(weight.representation<tyloi::NintTensor>() != nullptr)mixed=[weight,&profiler,&options](const DenseTensor& input,
        const DenseTensor& normalized,int64_t streams) {
        return profiler.measure("nint.float_projection_mix",[&] {
            return nint_float_projection_mix_cuda(payload_as<tyloi::NintTensor>(*weight),input,normalized,streams,options.native_input,
                options.fixed_group,options.compact_mix);
        });
    };
    Projection::Activation activated;
    if(weight.representation<tyloi::NintTensor>() != nullptr)activated=[weight,&profiler,&options](const DenseTensor& input,
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
#endif

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
#ifdef TYLOINFER_NATIVE_CUDA
    if(!initialized_)return false;
    const auto& options=Executor::current().options().residual;
#else
    if(!options_)return false;
    const auto& options=*options_;
#endif
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto* down_nint = down.weight
        ? down.weight.representation<tyloi::NintTensor>() : nullptr;
    const auto* up_nint = up.weight
        ? up.weight.representation<tyloi::NintTensor>() : nullptr;
    if(!options.two_stage || !options.fused_projections ||
        !options.native_input || down_nint == nullptr || up_nint == nullptr ||
        streams<1 || streams>4 ||
        input.dim()<1 || input.size(-1)!=hidden*streams || input.numel()<hidden*streams ||
        input.numel()/(hidden*streams)>32 || !input.is_cuda() || !input.is_contiguous() ||
        (input.scalar_type()!=::tyloi::kFloat16 && input.scalar_type()!=::tyloi::kFloat32) ||
#ifdef TYLOINFER_NATIVE_CUDA
        !norm->is_contiguous() || norm->scalar_type()!=::tyloi::kFloat32)return false;
#else
        !norm.is_contiguous() || norm.scalar_type()!=::tyloi::kFloat32)return false;
#endif
    const auto* injection_nint = injection.weight
        ? injection.weight.representation<tyloi::NintTensor>() : nullptr;
    const auto* injection_dense = injection.weight
        ? injection.weight.representation<DenseTensor>() : nullptr;
    if(injection_nint == nullptr && !(options.dense_injection && injection_dense &&
        injection_dense->is_contiguous() &&
        injection_dense->scalar_type()==::tyloi::kBFloat16))return false;
    const auto& w=*up_nint;
    const auto rows=input.numel()/(hidden*streams);
    const auto continuous=::tyloi::cuda::options::gr_continuous_two_stage();
#ifdef TYLOINFER_NATIVE_CUDA
    const bool cache_serial=Executor::current_ffn().batch_serial;
#else
    const bool cache_serial=*cache_serial_;
#endif
    if(rows>8 && (!cache_serial || (continuous && !*continuous)))
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
                static_cast<long long>(rows),kind==0,int(cache_serial));
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
    const auto* injection_nint = injection_payload.representation<tyloi::NintTensor>();
    const auto* injection_dense = injection_payload.representation<DenseTensor>();
#ifdef TYLOINFER_NATIVE_CUDA
    auto result=gated_residual_two_stage_cuda(input,residual,gate,payload_as<DenseTensor>(*norm),
#else
    auto result=gated_residual_two_stage_cuda(input,residual,gate,norm,
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(initialized_,"gated-residual operator is not initialized");
    const auto& options=Executor::current().options().residual;
#else
    TYLOINFER_RUNTIME_CHECK(options_,"gated-residual operator is not initialized");
    const auto& options=*options_;
#endif
    if(!options.fused_projections || !options.fused_post_norm) {
        return prepare(post(input,residual,gate),{},marker);
    }
    TYLOINFER_RUNTIME_CHECK(input.size(-1)==hidden,"gated-residual chained hidden width disagrees");
    if(marker)marker("begin");
#ifdef TYLOINFER_NATIVE_CUDA
    auto prepared=mfq_qwen4_exp::gated_residual_post_norm(input,residual,gate,payload_as<DenseTensor>(norm),streams,eps);
#else
    auto prepared=mfq_qwen4_exp::gated_residual_post_norm(input,residual,gate,norm,streams,eps);
#endif
    return prepare(prepared[0],prepared[1],marker);
}

GatedResidual::Result GatedResidual::prepare(
        const DenseTensor& input, const DenseTensor& prepared_norm, const GatedResidualMarker& marker) const {
#ifdef TYLOINFER_NATIVE_CUDA
    const bool available = initialized_;
#else
    const bool available = bool(options_);
#endif
    TYLOINFER_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 && available && down && up,
        "gated-residual projection input dimensions disagree");
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& options=Executor::current().options().residual;
#else
    const auto& options=*options_;
#endif
    if(marker && !prepared_norm.defined())marker("begin");
    auto normalized = prepared_norm.defined()?prepared_norm:
#ifdef TYLOINFER_NATIVE_CUDA
        mfq_qwen4_exp::grouped_rms_norm(input, payload_as<DenseTensor>(norm), hidden, eps);
#else
        mfq_qwen4_exp::grouped_rms_norm(input, norm, hidden, eps);
#endif
    if(marker)marker("norm");
    const bool fused=options.fused_projections;
    const bool epilogue=fused && options.fused_activation;
#ifdef TYLOINFER_NATIVE_CUDA
    const bool down_activated=epilogue && down.can_activate;
    auto low = down_activated?down.activate(normalized,streams,false):down.forward(normalized);
#else
    const bool down_activated=epilogue && bool(down.activated);
    auto low = down_activated?down.activated(normalized,streams,false):down.forward(normalized);
#endif
    if(marker)marker("down");
    TYLOINFER_RUNTIME_CHECK(low.dim() == input.dim() && low.size(-1) > 0,
        "gated-residual bottleneck rank disagrees");
    if(!down_activated) {
        if(fused)low=mfq_qwen4_exp::gated_residual_bottleneck(low,streams);
        else {low=low/streams;low=low*::tyloi::cuda::sigmoid(low);}
    }
    if(marker)marker("bottleneck");
    auto mixed = [&] {
#ifdef TYLOINFER_NATIVE_CUDA
        if(fused && up.can_mix)return up.mix(low,normalized,streams);
#else
        if(fused && up.mixed)return up.mixed(low,normalized,streams);
#endif
        auto mixing=up.forward(low);
        TYLOINFER_RUNTIME_CHECK(mixing.sizes()==input.sizes(),"gated-residual mixing shape disagrees");
        return mfq_qwen4_exp::gated_residual_mix(mixing,normalized,streams);
    }();
    if(marker)marker("up_mix");
    ::tyloi::DenseTensor gate;
    if (injection) {
#ifdef TYLOINFER_NATIVE_CUDA
        if(epilogue && injection.can_activate) {
            gate=injection.activate(normalized,streams,true);
#else
        if(epilogue && injection.activated) {
            gate=injection.activated(normalized,streams,true);
#endif
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

#endif
