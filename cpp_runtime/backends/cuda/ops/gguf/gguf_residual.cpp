#include "gguf_residual.h"
#include "gguf_weight.h"
#include "../../kernels/gguf.h"
#include "../../kernels/gguf_residual.h"
#include "../../kernels/gguf_residual_prepare.h"
#include <algorithm>
#include "mfq_cuda_context.h"
#include "qwen4_exp.h"
#include <atomic>
#include <cstdio>
#include <climits>
#include <cstdlib>

mfq_tensor_backend::Tensor try_gguf_residual_mix(const GgufWeight& weight,
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized,std::int64_t streams) {
    namespace tb=mfq_tensor_backend;
    if(const auto* setting=std::getenv("TYLOINFER_GGUF_GR_MIX");setting && setting[0]=='0')return {};
    if(weight.prefers_original_input() || !mfq::cuda::gguf::residual_mix_supported(weight.type) ||
            !weight.packed.is_cuda() || !input.is_cuda() || !normalized.is_cuda() ||
            input.device()!=weight.packed.device() || normalized.device()!=input.device() ||
            input.scalar_type()!=tb::kFloat32 || normalized.scalar_type()!=tb::kFloat32 ||
            !input.dim() || !normalized.dim() || input.size(-1)!=weight.neuron_len ||
            weight.neuron_len<1 || weight.neuron_len>INT_MAX || weight.out>INT_MAX || streams<1 ||
            streams>INT_MAX || weight.out<streams || weight.out%streams || normalized.size(-1)!=weight.out)return {};
    const auto tokens=input.numel()/weight.neuron_len;
    if(tokens<1 || tokens>4 || normalized.numel()!=tokens*weight.out)return {};
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.reshape({tokens,weight.neuron_len}).contiguous();
    const auto values=normalized.contiguous();
    auto shape=normalized.sizes().vec();shape.back()/=streams;
    auto output=tb::empty(shape,input.options());
    auto activation=tb::empty({std::int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(weight.neuron_len))},weight.packed.options());
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    for(const auto& tensor:{source,values,output,activation,weight.packed})mfq_cuda_record_stream(tensor,stream);
    mfq::cuda::gguf::quantize_input(source.data_ptr<float>(),tokens,weight.neuron_len,activation.data_ptr(),stream.stream());
    const auto image=weight.canonical_packed();mfq_cuda_record_stream(image,stream);
    mfq::cuda::gguf::residual_mix(weight.type,image.data_ptr(),weight.row_bytes,weight.out,weight.neuron_len,
        activation.data_ptr(),tokens,values.data_ptr<float>(),streams,output.data_ptr<float>(),stream.stream());
    return output;
}

std::vector<mfq_tensor_backend::Tensor> try_gguf_gated_residual_pre(
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& norm,
        const GgufWeight& down,const GgufWeight& up,const GgufWeight& injection,
        std::int64_t streams,double eps,const mfq_tensor_backend::Tensor& branch,
        const mfq_tensor_backend::Tensor& prior_injection,const GatedResidualMarker& marker) {
    namespace tb=mfq_tensor_backend;
    if(!input.defined() || !input.is_cuda() || !input.is_contiguous() || input.dim()<1 ||
            input.scalar_type()!=tb::kFloat32 || streams<1 || streams>INT_MAX ||
            input.size(-1)<1 || input.size(-1)>INT_MAX || input.size(-1)%streams)return {};
    const auto width=input.size(-1),tokens=input.numel()/width,hidden=width/streams;
    if(tokens<1 || tokens>4 || down.out<1 || down.neuron_len!=width || up.neuron_len!=down.out ||
            up.out!=width || injection.neuron_len!=width || injection.out!=streams ||
            !mfq::cuda::gguf::residual_mix_supported(up.type))return {};
    for(const auto* weight:{&down,&up,&injection})
        if(weight->prefers_original_input() || !weight->packed.is_cuda() ||
                weight->packed.device()!=input.device())return {};
    if(!norm.is_cuda() || norm.device()!=input.device() || norm.dim()!=1 || norm.numel()!=width)return {};
    if(branch.defined()) {
        auto shape=input.sizes().vec();shape.back()=hidden;
        auto gates=input.sizes().vec();gates.back()=streams;
        const auto floating=[](const auto& x){return x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32;};
        if(!prior_injection.defined() || !branch.is_cuda() || !prior_injection.is_cuda() ||
                branch.device()!=input.device() || prior_injection.device()!=input.device() ||
                !floating(branch) || !floating(prior_injection) || branch.sizes().vec()!=shape ||
                prior_injection.sizes().vec()!=gates)return {};
    } else if(prior_injection.defined())return {};
    const MfqCudaGuard guard(input.device());
    if(marker)marker("begin");
    tb::Tensor updated=input,normalized,prepared_activation;
    const std::vector<GgufWeight> pre_weights{down,injection};
    // This preparation emits the native Q8 packet. Existing aligned and
    // position-major groups retain their corresponding preparation layouts.
    const bool fused_input=hidden%32==0 && !prefers_gguf_m1_aligned(pre_weights,tokens) &&
        std::none_of(pre_weights.begin(),pre_weights.end(),[&](const auto& weight){
            return mfq::cuda::gguf::supports_interleaved_activation(weight.type,tokens);
        });
    if(fused_input) {
        normalized=tb::empty_like(input);
        if(branch.defined())updated=tb::empty_like(input);
        prepared_activation=tb::empty({std::int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(width))},
            input.options().dtype(tb::kUInt8));
        auto weight=norm.to(tb::kFloat32).contiguous();
        const auto stream=mfq_get_current_cuda_stream(input.get_device());
        for(const auto& tensor:{input,branch,prior_injection,weight,updated,normalized,prepared_activation})
            if(tensor.defined())mfq_cuda_record_stream(tensor,stream);
        mfq::cuda::gguf::prepare_residual_input(input.data_ptr<float>(),
            branch.defined()?branch.data_ptr():nullptr,branch.defined() && branch.scalar_type()==tb::kFloat16,
            prior_injection.defined()?prior_injection.data_ptr():nullptr,
            prior_injection.defined() && prior_injection.scalar_type()==tb::kFloat16,
            weight.data_ptr<float>(),tokens,hidden,streams,float(eps),
            branch.defined()?updated.data_ptr<float>():nullptr,normalized.data_ptr<float>(),
            prepared_activation.data_ptr(),stream.stream());
    } else if(branch.defined()) {
        auto prepared=mfq_qwen4_exp::gated_residual_post_norm(branch,input,prior_injection,norm,streams,eps);
        updated=prepared[0];normalized=prepared[1];
    } else normalized=mfq_qwen4_exp::grouped_rms_norm(input,norm,hidden,eps);
    if(marker)marker("norm");
    // Independent dense projections consume the shared packet directly.
    // Weights retain their standard formats outside routed expert caches.
    std::vector<tb::Tensor> projected;
    if(prepared_activation.defined()) {
        GgufProjectionInput prepared;
        prepared.native=prepared_activation;prepared.shape=normalized.sizes().vec();
        prepared.tokens=tokens;prepared.dtype=tb::kFloat32;
        projected=prepared.project_all(pre_weights);
    } else projected=project_gguf_group(pre_weights,normalized);
    if(marker)marker("down");
    tb::Tensor inject,mixed;
    const auto* mix_setting=std::getenv("TYLOINFER_GGUF_GR_MIX");
    const bool fused_activation=down.out<=INT_MAX && down.out%32==0 && (!mix_setting || mix_setting[0]!='0');
    if(fused_activation) {
        auto activation=tb::empty({std::int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(down.out))},up.packed.options());
        inject=tb::empty_like(projected[1]);
        auto shape=input.sizes().vec();shape.back()=hidden;
        mixed=tb::empty(shape,input.options());
        const auto stream=mfq_get_current_cuda_stream(input.get_device());
        for(const auto& tensor:{projected[0],projected[1],activation,inject,normalized,mixed,up.packed})
            mfq_cuda_record_stream(tensor,stream);
        mfq::cuda::gguf::prepare_residual_activation(projected[0].data_ptr<float>(),projected[1].data_ptr<float>(),
            tokens,down.out,streams,activation.data_ptr(),inject.data_ptr<float>(),stream.stream());
        if(marker){marker("bottleneck");marker("inject");}
        const auto image=up.canonical_packed();mfq_cuda_record_stream(image,stream);
        mfq::cuda::gguf::residual_mix(up.type,image.data_ptr(),up.row_bytes,up.out,up.neuron_len,
            activation.data_ptr(),tokens,normalized.data_ptr<float>(),streams,mixed.data_ptr<float>(),stream.stream());
    } else {
        auto low=mfq_qwen4_exp::gated_residual_bottleneck(projected[0],streams);
        if(marker)marker("bottleneck");
        inject=mfq_qwen4_exp::gated_residual_injection(projected[1],streams);
        if(marker)marker("inject");
        mixed=try_gguf_residual_mix(up,low,normalized,streams);
        if(!mixed.defined())mixed=mfq_qwen4_exp::gated_residual_mix(up.forward(low),normalized,streams);
    }
    if(marker){marker("up_mix");marker("end");}
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        cudaStreamCaptureStatus capture{};
        const auto stream=mfq_current_cuda_stream();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
        static std::atomic<unsigned> seen{};
        const auto bit=1u<<unsigned(tokens+(capture==cudaStreamCaptureStatusActive?8:0));
        if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "native_gguf_gr_prepare actual_match=1 M=%lld captured=%d shared_quant=1 matrices=2 original_weights=1 fused_activation=%d fused_input=%d\n",
            static_cast<long long>(tokens),int(capture==cudaStreamCaptureStatusActive),int(fused_activation),int(fused_input));
    }
    return {mixed,updated,inject};
}
