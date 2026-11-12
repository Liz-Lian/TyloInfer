#include "qwen4_exp.h"
#include "selected_attention.h"
#include "qwen4_raw.h"
#include "norm_raw.h"
#include "env.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace mfq_qwen4_jit {
namespace {

using Type = tyloi::cuda::NumericType;
namespace raw = tyloi::cuda::qwen4_raw;
Type type(const torch::Tensor& x) { return x.scalar_type()==at::kHalf ? Type::f16 : Type::f32; }
bool supported(const torch::Tensor& x) { return x.scalar_type()==at::kHalf || x.scalar_type()==at::kFloat; }
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }
torch::Tensor activation(const torch::Tensor& x,int64_t streams,bool injection) {
    mfq_selected_jit::values({&x});
    TORCH_CHECK(streams>0,"gated-residual stream count must be positive");
    if(x.is_contiguous() && supported(x)) {
        c10::cuda::CUDAGuard guard(x.device());
        auto out=torch::empty_like(x);
        if(out.numel()) {
            raw::activation(x.data_ptr(),out.data_ptr(),out.numel(),float(streams),type(x),injection,stream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        return out;
    }
    auto low=x/streams;
    return injection ? 2.0*torch::sigmoid(low) : low*torch::sigmoid(low);
}
torch::Tensor mix(const torch::Tensor& projection,const torch::Tensor& normalized,int64_t streams) {
    mfq_selected_jit::values({&projection,&normalized});
    TORCH_CHECK(projection.sizes()==normalized.sizes() && streams>0 && normalized.dim()>0 && normalized.size(-1)>0 && normalized.size(-1)%streams==0,"gated-residual mix shape disagrees");
    auto shape=normalized.sizes().vec();int hidden=int(shape.back()/streams);shape.back()=hidden;
    if(projection.is_contiguous() && normalized.is_contiguous() && supported(projection) && supported(normalized)) {
        c10::cuda::CUDAGuard guard(normalized.device());
        auto dtype=projection.scalar_type()==at::kFloat || normalized.scalar_type()==at::kFloat ? at::kFloat : at::kHalf;
        auto out=torch::empty(shape,normalized.options().dtype(dtype));
        if(out.numel()) raw::mix(projection.data_ptr(),normalized.data_ptr(),out.data_ptr(),out.numel(),hidden,int(streams),type(projection),type(normalized),stream());
        return out;
    }
    auto grouped=shape;grouped.back()=streams;grouped.push_back(hidden);
    return (torch::sigmoid(projection).reshape(grouped)*normalized.reshape(grouped)).mean(-2);
}
}
torch::Tensor grouped_rms_norm(const torch::Tensor& value,const torch::Tensor& weight,int64_t group,double eps) {
    mfq_selected_jit::values({&value,&weight});
    TORCH_CHECK(value.dim()>=1 && group>0 && value.size(-1)>0 && value.size(-1)%group==0 && weight.dim()==1 && weight.size(0)==value.size(-1),"Qwen4 grouped RMSNorm dimensions disagree");
    c10::cuda::CUDAGuard guard(value.device());
    if(value.is_contiguous() && supported(value)) {
        auto w=weight.to(at::kFloat).contiguous();auto out=torch::empty_like(value);
        int rows=int(value.numel()/group);
        if(rows) tyloi::cuda::norm_raw::grouped(value.data_ptr(),w.data_ptr<float>(),out.data_ptr(),rows,int(group),int(value.size(-1)/group),float(eps),1.0f,type(value),stream());
        return out;
    }
    auto shape=value.sizes().vec();shape.back()/=group;shape.push_back(group);
    auto source=value.to(at::kFloat).reshape(shape);
    auto normalized=source*torch::rsqrt((source*source).mean(-1,true)+eps);
    return (normalized.reshape(value.sizes())*(1.0+weight.to(at::kFloat))).to(value.scalar_type());
}
std::vector<torch::Tensor> gated_residual_pre(const torch::Tensor& input,const torch::Tensor& norm,const torch::Tensor& down,const torch::Tensor& up,const std::optional<torch::Tensor>& inject,int64_t hidden,int64_t streams,double eps) {
    mfq_selected_jit::values({&input,&norm,&down,&up});
    TORCH_CHECK(input.dim()>=1 && hidden>0 && streams>0 && input.size(-1)/streams==hidden && input.size(-1)%streams==0 && down.dim()==2 && up.dim()==2 && down.size(1)==input.size(-1) && up.size(0)==input.size(-1) && up.size(1)==down.size(0),"Qwen4 gated-residual projection dimensions disagree");
    if(inject) { mfq_selected_jit::values({&input,&*inject});TORCH_CHECK(inject->dim()==2 && inject->size(0)==streams && inject->size(1)==input.size(-1),"Qwen4 injection projection dimensions disagree"); }
    c10::cuda::CUDAGuard guard(input.device());
    auto normalized=grouped_rms_norm(input,norm,hidden,eps);
    auto low=activation(mfq_selected_jit::promoted_matmul(normalized,down.transpose(-1,-2)),streams,false);
    auto mixing=mfq_selected_jit::promoted_matmul(low,up.transpose(-1,-2));
    auto mixed=mix(mixing,normalized,streams);
    auto injection=inject ? activation(mfq_selected_jit::promoted_matmul(normalized,inject->transpose(-1,-2)),streams,true) : torch::Tensor{};
    return {mixed,input,injection};
}
torch::Tensor gated_residual_post(const torch::Tensor& branch,const torch::Tensor& residual,const torch::Tensor& injection,int64_t streams) {
    mfq_selected_jit::values({&branch,&residual,&injection});
    TORCH_CHECK(branch.dim()>=1 && streams>0 && branch.dim()==residual.dim() && branch.dim()==injection.dim(),"Qwen4 residual ranks disagree");
    auto shape=branch.sizes().vec();shape.back()*=streams;
    TORCH_CHECK(residual.sizes().vec()==shape,"Qwen4 residual width disagrees");shape.back()=streams;
    TORCH_CHECK(injection.sizes().vec()==shape,"Qwen4 injection shape disagrees");
    c10::cuda::CUDAGuard guard(branch.device());
    if(branch.is_contiguous() && residual.is_contiguous() && injection.is_contiguous() && branch.scalar_type()==residual.scalar_type() && branch.scalar_type()==injection.scalar_type() && supported(branch)) {
        auto out=torch::empty_like(residual);
        if(out.numel()) raw::post(branch.data_ptr(),residual.data_ptr(),injection.data_ptr(),out.data_ptr(),out.numel(),int(branch.size(-1)),int(streams),type(branch),stream());
        return out;
    }
    return residual+(branch.unsqueeze(-2)*injection.unsqueeze(-1)).reshape(residual.sizes());
}
torch::Tensor block_scores(const torch::Tensor& query,const torch::Tensor& pooled) {
    mfq_selected_jit::values({&query,&pooled});c10::cuda::CUDAGuard guard(query.device());
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_QSA_PREFILL_SCORE_GEMM");
    if(query.size(1)>=32 && option && option[0]=='1') {
        auto q=query.to(at::kFloat).permute({0,2,1,3}).contiguous();
        auto keys=pooled.to(at::kFloat).transpose(-1,-2).unsqueeze(1);
        return torch::matmul(q,keys).permute({0,2,1,3}).clamp_min(0.0).sum(-2)/std::sqrt(double(query.size(-1)));
    }
    return mfq_selected_jit::dots(query,pooled).sum(-2)/std::sqrt(double(query.size(-1)));
}
std::vector<torch::Tensor> ple_dilated_conv_silu(const torch::Tensor& input,const torch::Tensor& weight,const std::optional<torch::Tensor>& state,int64_t dilation) {
    mfq_selected_jit::values({&input,&weight});
    TORCH_CHECK(input.dim()==3,"Qwen4 PLE input must have [B,T,C] shape");
    auto w=weight;
    if(w.dim()==3) { TORCH_CHECK(w.size(0)==input.size(2) && w.size(1)==1,"Qwen4 PLE packed weight dimensions disagree");w=w.select(1,0); }
    TORCH_CHECK(w.dim()==2 && w.size(0)==input.size(2) && w.size(1)>0 && dilation>0 && w.size(1)-1<=std::numeric_limits<int64_t>::max()/dilation,"Qwen4 PLE kernel/dilation dimensions disagree");
    auto length=(w.size(1)-1)*dilation;std::vector<int64_t> state_shape{input.size(0),length,input.size(2)};
    if(state) { mfq_selected_jit::values({&input,&*state});TORCH_CHECK(state->sizes().vec()==state_shape,"Qwen4 PLE state shape disagrees"); }
    c10::cuda::CUDAGuard guard(input.device());
    auto previous=state ? state->to(input.scalar_type()) : torch::zeros(state_shape,input.options());
    auto combined=torch::cat({previous,input},1);
    auto output=torch::zeros(input.sizes(),input.options().dtype(at::kFloat));
    for(int64_t tap=0;tap<w.size(1);++tap)
        output=output+combined.narrow(1,tap*dilation,input.size(1)).to(at::kFloat)*w.select(1,tap).to(at::kFloat).unsqueeze(0).unsqueeze(0);
    output=output*torch::sigmoid(output);
    auto next=combined.narrow(1,length ? combined.size(1)-length : 0,length).contiguous();
    return {output.to(input.scalar_type()),next};
}
torch::Tensor dense_gqa_attention(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,int64_t offset) {
    return mfq_selected_jit::dense(q,k,v,offset,1.0/std::sqrt(double(q.size(3))));
}
torch::Tensor sparse_gqa_attention(const torch::Tensor& q,const torch::Tensor& k,const torch::Tensor& v,const torch::Tensor& indices) {
    return mfq_selected_jit::sparse(q,k,v,indices,1.0/std::sqrt(double(q.size(3))),false);
}
} // namespace mfq_qwen4_jit
