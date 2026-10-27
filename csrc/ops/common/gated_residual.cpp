#include "gated_residual.h"
#include "qwen4_exp.h"
#include "quant_linear.h"
#include "gated_residual_fused.h"
#include "float_projection.h"
#include "engine/executor/options.h"

#include <algorithm>
#include <cstdio>

namespace mfq::cuda {
namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;

int64_t GatedResidual::prefill_work_chunk(const Tensor& input)const {
    if(input.dim()!=3 || input.size(0)!=1)return 0;
    const auto option=mfq::cuda::runtime_options::gr_prefill_work_chunk();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
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
GatedResidual::Result GatedResidual::bounded_pre(const Tensor& input,int64_t chunk,bool reuse_input,Project project)const {
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
        const auto append=[&](Tensor& destination,const Tensor& source) {
            if(!source.defined())return;
            if(!destination.defined()) {
                auto shape=source.sizes().vec();shape[1]=tokens;
                destination=tb::empty(shape,source.options());
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
bool GatedResidual::can_fuse(const Tensor& input)const {
    if(!options_)return false;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if(!options_->two_stage || !options_->fused_projections ||
        !options_->native_input || !down.weight || !up.weight || !injection.weight ||
        !down.weight->is_nint() || !up.weight->is_nint() || streams<1 || streams>4 ||
        input.dim()<1 || input.size(-1)!=hidden*streams || input.numel()<hidden*streams ||
        input.numel()/(hidden*streams)>32 || !input.is_cuda() || !input.is_contiguous() ||
        (input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kFloat32) ||
        !norm.is_contiguous() || norm.scalar_type()!=tb::kFloat32)return false;
    if(!injection.weight->is_nint() && !(options_->dense_injection &&
        injection.weight->is_dense() && injection.weight->dense.is_contiguous() &&
        injection.weight->dense.scalar_type()==tb::kBFloat16))return false;
    const auto& w=up.weight->nint;
    const auto rows=input.numel()/(hidden*streams);
    const auto continuous=mfq::cuda::runtime_options::gr_continuous_two_stage();
    if(rows>8 && (!*cache_serial_ || (continuous && !*continuous)))
        return false;
    const auto batch=mfq::cuda::runtime_options::gr_batch_two_stage();
    if(rows>1 && batch && !*batch)return false;
    const bool selected=w.ng==1 || ((w.out+3)/4)*rows>=nint_float_projection_resident_blocks(input.get_device());
    const auto trace=mfq::cuda::runtime_options::trace_gr_dispatch();
    if(selected && trace && *trace) {
        static thread_local uint64_t seen[2]{};
        const int kind=input.scalar_type()==tb::kFloat16?0:1;
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
GatedResidual::Result GatedResidual::fused(const Tensor& input,const Tensor& residual,const Tensor& gate,bool after,
    const GatedResidualMarker& marker)const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto& projection=*injection.weight;
    auto result=gated_residual_two_stage_cuda(input,residual,gate,norm,down.weight->nint,up.weight->nint,
        projection.is_nint()?&projection.nint:nullptr,streams,eps,after,
        projection.is_dense()?projection.dense:Tensor{},injection.prepared_right,marker,up.up_group_coefficients);
    return {std::move(result[0]),std::move(result[1]),std::move(result[2])};
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
GatedResidual::Result GatedResidual::pre(const Tensor &input,
    const GatedResidualMarker& marker) const {
    if(const auto chunk=prefill_work_chunk(input))
        return bounded_pre(input,chunk,true,[&](int64_t begin,int64_t count) {
            return pre(input.narrow(1,begin,count),marker);
        });
    if(can_fuse(input))return fused({},input,{},false,marker);
    return prepare(input,{},marker);
}
Tensor GatedResidual::post(const Tensor& input,const Tensor& residual,const Tensor& gate) const {
    return mfq_qwen4_exp::gated_residual_post(input,residual,gate,streams);
}
GatedResidual::Result GatedResidual::pre_after(const Tensor& input,
    const Tensor& residual,const Tensor& gate,const GatedResidualMarker& marker) const {
    if(const auto chunk=prefill_work_chunk(residual))
        return bounded_pre(residual,chunk,false,[&](int64_t begin,int64_t count) {
            return pre_after(input.narrow(1,begin,count),residual.narrow(1,begin,count),
                gate.narrow(1,begin,count),marker);
        });
    if(can_fuse(residual) && input.is_contiguous() && gate.is_contiguous() &&
        (input.scalar_type()==tb::kFloat16 || input.scalar_type()==tb::kFloat32) &&
        (gate.scalar_type()==tb::kFloat16 || gate.scalar_type()==tb::kFloat32))
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
        const Tensor& input, const Tensor& prepared_norm, const GatedResidualMarker& marker) const {
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
        else {low=low/streams;low=low*tb::sigmoid(low);}
    }
    if(marker)marker("bottleneck");
    auto mixed = [&] {
        if(fused && up.mixed)return up.mixed(low,normalized,streams);
        auto mixing=up.forward(low);
        TYLOINFER_RUNTIME_CHECK(mixing.sizes()==input.sizes(),"gated-residual mixing shape disagrees");
        return mfq_qwen4_exp::gated_residual_mix(mixing,normalized,streams);
    }();
    if(marker)marker("up_mix");
    tb::Tensor gate;
    if (injection.forward) {
        if(epilogue && injection.activated) {
            gate=injection.activated(normalized,streams,true);
            if(marker)marker("inject");
        } else {
            auto projected=injection.forward(normalized);
            if(marker)marker("inject");
            gate=fused?mfq_qwen4_exp::gated_residual_injection(projected,streams):
                2.0*tb::sigmoid(projected/streams);
        }
        auto expected = input.sizes().vec();
        expected.back() = streams;
        TYLOINFER_RUNTIME_CHECK(gate.sizes().vec() == expected,
            "gated-residual injection shape disagrees");
    }
    if(marker)marker("end");
    return {mixed, input, gate};
}

} // namespace mfq::cuda
