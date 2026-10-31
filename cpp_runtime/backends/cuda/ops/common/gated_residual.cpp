#include "gated_residual.h"
#include "qwen4_exp.h"
#include "cuda_execution.h"
#include "float_projection.h"
#include <climits>

std::vector<mfq_tensor_backend::Tensor> try_gated_residual_bf16_cuda(
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& norm,
        const mfq_tensor_backend::Tensor& down,const mfq_tensor_backend::Tensor& up,
        const mfq_tensor_backend::Tensor& injection,int64_t streams,double eps,
        const mfq_tensor_backend::Tensor& branch,const mfq_tensor_backend::Tensor& prior_injection,
        const GatedResidualMarker& marker) {
    namespace tb=mfq_tensor_backend;
    if(input.defined() && input.dim()>0 && input.size(-1)>0 && input.numel()/input.size(-1)>32)
        return try_gated_residual_bf16_prefill_cuda(input,norm,down,up,injection,streams,eps,branch,prior_injection,marker);
    const auto value=[](const tb::Tensor& x){return x.scalar_type()==tb::kFloat32 || x.scalar_type()==tb::kFloat16;};
    if(streams<1 || streams>INT_MAX || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 || !value(input) ||
            input.size(-1)<1 || input.size(-1)>INT_MAX || input.size(-1)%streams ||
            input.numel()/input.size(-1)<1 || input.numel()/input.size(-1)>6 ||
            !norm.is_cuda() || norm.device()!=input.device() || !norm.is_contiguous() || norm.scalar_type()!=tb::kFloat32 ||
            norm.numel()!=input.size(-1))return {};
    for(const auto& w:{down,up,injection})
        if(!w.defined() || !w.is_cuda() || !w.is_contiguous() || w.device()!=input.device() ||
                w.scalar_type()!=tb::kBFloat16 || w.dim()!=2)return {};
    const auto width=input.size(-1),rank=down.size(0),hidden=width/streams;
    if(rank<1 || rank>INT_MAX-streams || down.size(1)!=width || up.size(0)!=width || up.size(1)!=rank ||
            injection.size(0)!=streams || injection.size(1)!=width)return {};
    if(branch.defined()) {
        auto shape=input.sizes().vec();shape.back()=hidden;
        auto gates=input.sizes().vec();gates.back()=streams;
        if(!prior_injection.defined() || !branch.is_cuda() || branch.device()!=input.device() || !branch.is_contiguous() ||
                !value(branch) || branch.sizes().vec()!=shape || !prior_injection.is_cuda() ||
                prior_injection.device()!=input.device() || !prior_injection.is_contiguous() ||
                !value(prior_injection) || prior_injection.sizes().vec()!=gates)return {};
    } else if(prior_injection.defined())return {};
    const MfqCudaGuard guard(input.device());
    if(marker)marker("begin");
    tb::Tensor updated=input,normalized;
    if(branch.defined()) {
        auto prepared=mfq_qwen4_exp::gated_residual_post_norm(branch,input,prior_injection,norm,streams,eps);
        updated=prepared[0];normalized=prepared[1];
    } else normalized=mfq_qwen4_exp::grouped_rms_norm(input,norm,hidden,eps);
    if(marker)marker("norm");
    auto projected=try_gr_bf16_down_injection_cuda(normalized,down,injection,streams);
    TYLOINFER_RUNTIME_CHECK(!projected.empty(),"BF16 gated-residual combined projection rejected validated inputs");
    if(marker){marker("down");marker("bottleneck");marker("inject");}
    auto mixed=try_gr_bf16_up_mix_cuda(projected[0],normalized,up,streams);
    TYLOINFER_RUNTIME_CHECK(mixed.defined(),"BF16 gated-residual mixed projection rejected validated inputs");
    if(marker){marker("up_mix");marker("end");}
    return {mixed,updated,projected[1]};
}

namespace {
std::vector<mfq_tensor_backend::Tensor> residual_pre(
        CudaExecutionContext& execution, const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& norm, const GatedResidualProjection& down,
        const GatedResidualProjection& up, const GatedResidualProjection& inject,
        int64_t hidden, int64_t streams, double eps,
        const GatedResidualMixProjection& mixed_projection,const GatedResidualMarker& marker,
        const GatedResidualActivationProjection& activated_down,
        const GatedResidualActivationProjection& activated_injection,
        const mfq_tensor_backend::Tensor& prepared_norm = {}) {
    namespace tb = mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 && down && up,
        "gated-residual projection input dimensions disagree");
    if(marker && !prepared_norm.defined())marker("begin");
    auto normalized = prepared_norm.defined()?prepared_norm:
        mfq_qwen4_exp::grouped_rms_norm(input, norm, hidden, eps);
    if(marker)marker("norm");
    const bool fused=execution.config.gr_fused_projections;
    const bool epilogue=fused && execution.config.gr_fused_projection_activation;
    const bool down_activated=epilogue && bool(activated_down);
    auto low = down_activated?activated_down(execution,normalized,streams,false):down(execution, normalized);
    if(marker)marker("down");
    TYLOINFER_RUNTIME_CHECK(low.dim() == input.dim() && low.size(-1) > 0,
        "gated-residual bottleneck rank disagrees");
    if(!down_activated) {
        if(fused)low=mfq_qwen4_exp::gated_residual_bottleneck(low,streams);
        else {low=low/streams;low=low*tb::sigmoid(low);}
    }
    if(marker)marker("bottleneck");
    auto mixed = [&] {
        if(fused && mixed_projection)return mixed_projection(execution,low,normalized,streams);
        auto mixing=up(execution,low);
        TYLOINFER_RUNTIME_CHECK(mixing.sizes()==input.sizes(),"gated-residual mixing shape disagrees");
        return mfq_qwen4_exp::gated_residual_mix(mixing,normalized,streams);
    }();
    if(marker)marker("up_mix");
    tb::Tensor injection;
    if (inject) {
        if(epilogue && activated_injection) {
            injection=activated_injection(execution,normalized,streams,true);
            if(marker)marker("inject");
        } else {
            auto projected=inject(execution,normalized);
            if(marker)marker("inject");
            injection=fused?mfq_qwen4_exp::gated_residual_injection(projected,streams):
                2.0*tb::sigmoid(projected/streams);
        }
        auto expected = input.sizes().vec();
        expected.back() = streams;
        TYLOINFER_RUNTIME_CHECK(injection.sizes().vec() == expected,
            "gated-residual injection shape disagrees");
    }
    if(marker)marker("end");
    return {mixed, input, injection};
}
}

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_projected(
    CudaExecutionContext& execution,const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& norm,const GatedResidualProjection& down,
    const GatedResidualProjection& up,const GatedResidualProjection& inject,
    int64_t hidden,int64_t streams,double eps,const GatedResidualMixProjection& mixed,
    const GatedResidualMarker& marker,const GatedResidualActivationProjection& activated_down,
    const GatedResidualActivationProjection& activated_injection) {
    return residual_pre(execution,input,norm,down,up,inject,hidden,streams,eps,
        mixed,marker,activated_down,activated_injection);
}

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_after_projected(
    CudaExecutionContext& execution,const mfq_tensor_backend::Tensor& branch,
    const mfq_tensor_backend::Tensor& residual,const mfq_tensor_backend::Tensor& prior_injection,
    const mfq_tensor_backend::Tensor& norm,const GatedResidualProjection& down,
    const GatedResidualProjection& up,const GatedResidualProjection& inject,
    int64_t hidden,int64_t streams,double eps,const GatedResidualMixProjection& mixed,
    const GatedResidualMarker& marker,const GatedResidualActivationProjection& activated_down,
    const GatedResidualActivationProjection& activated_injection) {
    if(!execution.config.gr_fused_projections || !execution.config.gr_fused_post_norm) {
        auto input=mfq_qwen4_exp::gated_residual_post(branch,residual,prior_injection,streams);
        return residual_pre(execution,input,norm,down,up,inject,hidden,streams,eps,
            mixed,marker,activated_down,activated_injection);
    }
    TYLOINFER_RUNTIME_CHECK(branch.size(-1)==hidden,"gated-residual chained hidden width disagrees");
    if(marker)marker("begin");
    auto prepared=mfq_qwen4_exp::gated_residual_post_norm(branch,residual,prior_injection,norm,streams,eps);
    return residual_pre(execution,prepared[0],norm,down,up,inject,hidden,streams,eps,
        mixed,marker,activated_down,activated_injection,prepared[1]);
}
