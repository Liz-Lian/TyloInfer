#include "backend.h"
#include "ssm_conv_raw.h"
#include <cstdint>
#include <vector>

::tyloi::DenseTensor ssm_conv_silu_cuda(::tyloi::DenseTensor conv_input, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias, int64_t n_tokens)
{
    TYLOINFER_RUNTIME_CHECK(conv_input.is_cuda() && conv_input.is_contiguous() && conv_input.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu: conv_input must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(conv_input.dim() == 3, "ssm_conv_silu: conv_input must be [B,K-1+T,C]");
    int B = (int)conv_input.size(0);
    int C = (int)conv_input.size(2);
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu: bias must be [C]");
    int K = 0;
    int layout = 0; // 0: channel-major [C,1,K]/[C,K], 1: [K,C]
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1, "ssm_conv_silu: [C,1,K] weight mismatch");
        K = (int)weight.size(2);
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2, "ssm_conv_silu: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C) {
            K = (int)weight.size(1);
            layout = 0;
        } else {
            TYLOINFER_RUNTIME_CHECK(weight.size(1) == C, "ssm_conv_silu: 2D weight must be [C,K] or [K,C]");
            K = (int)weight.size(0);
            layout = 1;
        }
    }
    int T = (int)n_tokens;
    TYLOINFER_RUNTIME_CHECK(T > 0 && conv_input.size(1) == T + K - 1, "ssm_conv_silu: conv_input length must be K-1+T");
    auto out = ::tyloi::cuda::empty({B, T, C}, conv_input.options());
    tyloi::cuda::ssm_conv_raw::Args a;
    a.input=conv_input.data_ptr<float>(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>(); a.output=out.data_ptr<float>();
    a.batch=B; a.tokens=T; a.channels=C; a.kernel=K; a.layout=layout; a.has_bias=has_bias;
    tyloi::cuda::ssm_conv_raw::prefill(a, mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor ssm_conv_silu_decode_cuda(::tyloi::DenseTensor state, ::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: state must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: x must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "ssm_conv_silu_decode: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3, "ssm_conv_silu_decode: state must be [B,K-1,C]");
    TYLOINFER_RUNTIME_CHECK(x.dim() == 3 && x.size(1) == 1, "ssm_conv_silu_decode: x must be [B,1,C]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int C = (int)state.size(2);
    TYLOINFER_RUNTIME_CHECK(x.size(0) == B && x.size(2) == C, "ssm_conv_silu_decode: x shape mismatch");
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "ssm_conv_silu_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2,
                    "ssm_conv_silu_decode: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C && weight.size(1) == K) {
            layout = 0;
        } else {
            TYLOINFER_RUNTIME_CHECK(weight.size(0) == K && weight.size(1) == C,
                        "ssm_conv_silu_decode: 2D weight must be [C,K] or [K,C]");
            layout = 1;
        }
    }
    auto out = ::tyloi::cuda::empty({B, 1, C}, x.options());
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.input=x.data_ptr<float>(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>(); a.output=out.data_ptr<float>();
    a.batch=B; a.channels=C; a.kernel=K; a.layout=layout; a.has_bias=has_bias;
    tyloi::cuda::ssm_conv_raw::decode(a, mfq_current_cuda_stream());
    return out;
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_decode_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qk,::tyloi::DenseTensor v,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor dt_bias,::tyloi::DenseTensor a_log,
    int64_t nk,int64_t nv,int64_t dk,int64_t dv,double eps) {
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==::tyloi::kFloat32 &&
        state.dim()==3 && state.size(1)>0 && nk>0 && nv>0 && nv%nk==0 && dk>0 && dk<=256 && dv>0,
        "GDN combined decode state/geometry disagrees");
    const int B=int(state.size(0)),K=int(state.size(1))+1,C=int(2*nk*dk+nv*dv);
    const auto same_device=[&](const ::tyloi::DenseTensor& x){return x.is_cuda() && x.device()==state.device();};
    TYLOINFER_RUNTIME_CHECK(same_device(qk) && same_device(v) && qk.is_contiguous() && v.is_contiguous() &&
        qk.scalar_type()==::tyloi::kFloat16 && v.scalar_type()==::tyloi::kFloat16 &&
        qk.sizes().vec()==std::vector<int64_t>({B,1,2*nk*dk}) &&
        v.sizes().vec()==std::vector<int64_t>({B,1,nv*dv}) && state.size(2)==C,
        "GDN combined decode projection shape/dtype disagrees");
    TYLOINFER_RUNTIME_CHECK(same_device(alpha) && same_device(beta) && alpha.scalar_type()==beta.scalar_type() &&
        (alpha.scalar_type()==::tyloi::kFloat16 || alpha.scalar_type()==::tyloi::kFloat32) &&
        alpha.sizes().vec()==std::vector<int64_t>({B,1,nv}) && beta.sizes()==alpha.sizes(),
        "GDN combined decode gates disagree");
    TYLOINFER_RUNTIME_CHECK(same_device(weight) && same_device(dt_bias) && same_device(a_log) &&
        weight.is_contiguous() && dt_bias.is_contiguous() && a_log.is_contiguous() &&
        weight.scalar_type()==::tyloi::kFloat32 && dt_bias.scalar_type()==::tyloi::kFloat32 && a_log.scalar_type()==::tyloi::kFloat32 &&
        dt_bias.numel()==nv && a_log.numel()==nv,"GDN combined decode parameters disagree");
    const bool channels_first=weight.sizes().vec()==std::vector<int64_t>({C,1,K});
    TYLOINFER_RUNTIME_CHECK(channels_first || weight.sizes().vec()==std::vector<int64_t>({K,C}),
        "GDN combined decode convolution shape disagrees");
    MfqCudaGuard guard(state.device());auto options=state.options();
    auto q=::tyloi::cuda::empty({B,nk,1,dk},options),k=::tyloi::cuda::empty_like(q),output=::tyloi::cuda::empty({B,nv,1,dv},options);
    auto decay=::tyloi::cuda::empty({B,nv,1},options),beta_out=::tyloi::cuda::empty_like(decay);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.qk=qk.data_ptr(); a.v=v.data_ptr(); a.weight=weight.data_ptr<float>();
    a.alpha=alpha.data_ptr(); a.beta=beta.data_ptr(); a.dt_bias=dt_bias.data_ptr<float>(); a.a_log=a_log.data_ptr<float>();
    a.q=q.data_ptr<float>(); a.k=k.data_ptr<float>(); a.output=output.data_ptr<float>(); a.decay=decay.data_ptr<float>(); a.beta_out=beta_out.data_ptr<float>();
    a.batch=B; a.nk=int(nk); a.nv=int(nv); a.dk=int(dk); a.dv=int(dv); a.kernel=K; a.layout=channels_first?0:1; a.eps=float(eps);
    a.half_gate=alpha.scalar_type()==::tyloi::kFloat16; a.a_stride0=alpha.stride(0); a.a_stride1=alpha.stride(2); a.b_stride0=beta.stride(0); a.b_stride1=beta.stride(2);
    tyloi::cuda::ssm_conv_raw::qkv_gate_decode(a, mfq_current_cuda_stream());
    return {q,k,output,decay,beta_out};
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_decode_cuda(
    ::tyloi::DenseTensor state,
    ::tyloi::DenseTensor qk,
    ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor weight,
    ::tyloi::DenseTensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_decode: state must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == ::tyloi::kHalf,
                "linear_conv_qkv_decode: qk must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == ::tyloi::kHalf,
                "linear_conv_qkv_decode: v must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_decode: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_decode: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3, "linear_conv_qkv_decode: state must be [B,K-1,C]");
    TYLOINFER_RUNTIME_CHECK(qk.dim() == 3 && qk.size(1) == 1, "linear_conv_qkv_decode: qk must be [B,1,2*nk*dk]");
    TYLOINFER_RUNTIME_CHECK(v.dim() == 3 && v.size(1) == 1, "linear_conv_qkv_decode: v must be [B,1,nv*dv]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TYLOINFER_RUNTIME_CHECK(qk.size(0) == B && qk.size(2) == qkC, "linear_conv_qkv_decode: qk shape mismatch");
    TYLOINFER_RUNTIME_CHECK(v.size(0) == B && v.size(2) == vsz, "linear_conv_qkv_decode: v shape mismatch");
    TYLOINFER_RUNTIME_CHECK(state.size(2) == C, "linear_conv_qkv_decode: state width mismatch");
    TYLOINFER_RUNTIME_CHECK(nv % nk == 0, "linear_conv_qkv_decode: nv must be divisible by nk");
    TYLOINFER_RUNTIME_CHECK(dk <= 256, "linear_conv_qkv_decode: dk > 256 is unsupported");
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_decode: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    auto opts = state.options();
    auto q = ::tyloi::cuda::empty({B, (int)nk, 1, (int)dk}, opts);
    auto k = ::tyloi::cuda::empty({B, (int)nk, 1, (int)dk}, opts);
    auto vo = ::tyloi::cuda::empty({B, (int)nv, 1, (int)dv}, opts);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.qk=qk.data_ptr(); a.v=v.data_ptr(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>();
    a.q=q.data_ptr<float>(); a.k=k.data_ptr<float>(); a.output=vo.data_ptr<float>();
    a.batch=B; a.nk=int(nk); a.nv=int(nv); a.dk=int(dk); a.dv=int(dv); a.kernel=K; a.layout=layout; a.has_bias=has_bias; a.eps=float(eps);
    tyloi::cuda::ssm_conv_raw::qkv_decode(a, mfq_current_cuda_stream());
    return {q,k,vo};
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_prefill_cuda(
    ::tyloi::DenseTensor state,
    ::tyloi::DenseTensor qk,
    ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor weight,
    ::tyloi::DenseTensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_prefill: state must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == ::tyloi::kHalf,
                "linear_conv_qkv_prefill: qk must be contiguous CUDA f16");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == ::tyloi::kHalf,
                "linear_conv_qkv_prefill: v must be contiguous CUDA f16");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_prefill: weight must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == ::tyloi::kFloat32,
                "linear_conv_qkv_prefill: bias must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3 && qk.dim() == 3 && v.dim() == 3,
                "linear_conv_qkv_prefill: state/qk/v must be rank 3");
    int B = (int)qk.size(0);
    int T = (int)qk.size(1);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TYLOINFER_RUNTIME_CHECK(T > 1 && qk.size(2) == qkC && v.size(0) == B && v.size(1) == T && v.size(2) == vsz,
                "linear_conv_qkv_prefill: qk/v shape mismatch");
    TYLOINFER_RUNTIME_CHECK(state.size(0) == B && state.size(2) == C, "linear_conv_qkv_prefill: state shape mismatch");
    TYLOINFER_RUNTIME_CHECK(dk <= 256 && K <= 8, "linear_conv_qkv_prefill: requires dk<=256 and K<=8");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_prefill: [C,1,K] weight mismatch");
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_prefill: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_prefill: bias shape mismatch");

    auto opts = state.options();
    auto qo = ::tyloi::cuda::empty({B, (int)nk, T, (int)dk}, opts);
    auto ko = ::tyloi::cuda::empty_like(qo);
    auto vo = ::tyloi::cuda::empty({B, (int)nv, T, (int)dv}, opts);
    auto new_state = ::tyloi::cuda::empty_like(state);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.qk=qk.data_ptr(); a.v=v.data_ptr(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>();
    a.q=qo.data_ptr<float>(); a.k=ko.data_ptr<float>(); a.output=vo.data_ptr<float>(); a.next=new_state.data_ptr<float>();
    a.batch=B; a.tokens=T; a.channels=C; a.nk=int(nk); a.nv=int(nv); a.dk=int(dk); a.dv=int(dv); a.kernel=K; a.layout=layout; a.has_bias=has_bias; a.eps=float(eps);
    tyloi::cuda::ssm_conv_raw::qkv_prefill(a, mfq_current_cuda_stream());
    return {qo,ko,vo,new_state};
}

std::vector<::tyloi::DenseTensor> linear_conv_qkv_gate_short_cuda(
    ::tyloi::DenseTensor state,::tyloi::DenseTensor qkv,
    ::tyloi::DenseTensor weight,::tyloi::DenseTensor alpha,
    ::tyloi::DenseTensor beta,::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log,int64_t nk,int64_t nv,int64_t width,double eps)
{
    TYLOINFER_RUNTIME_CHECK(qkv.is_cuda() && qkv.scalar_type()==::tyloi::kFloat16 &&
        qkv.dim()==3 && qkv.size(1)>=2 && qkv.size(1)<=8 && nk>0 && nv>0 && nv%nk==0 &&
        width>0 && width<=256 && qkv.stride(2)==1 && qkv.stride(1)>=qkv.size(2) &&
        (qkv.size(0)==1 || qkv.stride(0)>=qkv.size(1)*qkv.stride(1)),
        "short GDN requires row-strided half QKV and 2-8 tokens");
    const int B=int(qkv.size(0)),T=int(qkv.size(1)),K=int(state.size(1))+1;
    const int qkC=int(2*nk*width),vsz=int(nv*width),C=qkC+vsz;
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==::tyloi::kFloat32 &&
        state.sizes().vec()==std::vector<int64_t>({B,K-1,C}) && K>=2 && K<=8 && qkv.size(2)==C,
        "short GDN convolution state geometry disagrees");
    for(const auto* tensor:{&weight,&dt_bias,&a_log})
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->is_contiguous() && tensor->scalar_type()==::tyloi::kFloat32 &&
            tensor->device()==qkv.device(),"short GDN parameters require contiguous float on the QKV device");
    TYLOINFER_RUNTIME_CHECK(alpha.is_cuda() && beta.is_cuda() &&
        alpha.device()==qkv.device() && beta.device()==qkv.device() && state.device()==qkv.device() &&
        alpha.sizes().vec()==std::vector<int64_t>({B,T,nv}) && beta.sizes()==alpha.sizes() &&
        alpha.stride(2)==1 && beta.stride(2)==1 && alpha.stride(1)>=nv && beta.stride(1)>=nv &&
        (B==1 || (alpha.stride(0)>=T*alpha.stride(1) && beta.stride(0)>=T*beta.stride(1))) &&
        beta.scalar_type()==alpha.scalar_type() &&
        (alpha.scalar_type()==::tyloi::kFloat16 || alpha.scalar_type()==::tyloi::kFloat32) &&
        dt_bias.numel()==nv && a_log.numel()==nv,"short GDN gate geometry disagrees");
    const int layout=weight.dim()==2?1:0;
    TYLOINFER_RUNTIME_CHECK(layout?weight.sizes().vec()==std::vector<int64_t>({K,C}):
        weight.sizes().vec()==std::vector<int64_t>({C,1,K}),"short GDN convolution weight geometry disagrees");
    auto options=state.options();
    auto q=::tyloi::cuda::empty({B,nk,T,width},options),k=::tyloi::cuda::empty_like(q),v=::tyloi::cuda::empty({B,nv,T,width},options);
    auto next=::tyloi::cuda::empty_like(state),decay=::tyloi::cuda::empty({B,nv,T},options),bt=::tyloi::cuda::empty_like(decay);
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.qk=qkv.data_ptr(); a.weight=weight.data_ptr<float>(); a.alpha=alpha.data_ptr(); a.beta=beta.data_ptr();
    a.dt_bias=dt_bias.data_ptr<float>(); a.a_log=a_log.data_ptr<float>(); a.q=q.data_ptr<float>(); a.k=k.data_ptr<float>(); a.output=v.data_ptr<float>(); a.next=next.data_ptr<float>(); a.decay=decay.data_ptr<float>(); a.beta_out=bt.data_ptr<float>();
    a.batch=B; a.tokens=T; a.channels=C; a.nk=int(nk); a.nv=int(nv); a.dk=int(width); a.dv=int(width); a.kernel=K; a.layout=layout; a.eps=float(eps);
    a.half_gate=alpha.scalar_type()==::tyloi::kFloat16; a.q_stride0=qkv.stride(0); a.q_stride1=qkv.stride(1); a.a_stride0=alpha.stride(0); a.a_stride1=alpha.stride(1); a.b_stride0=beta.stride(0); a.b_stride1=beta.stride(1);
    tyloi::cuda::ssm_conv_raw::qkv_gate_short(a, mfq_current_cuda_stream());
    return {q,k,v,next,decay,bt};
}
