#include "env.h"
#include "qwen4_exp.h"
#include "mfq_cuda_norm_ops.h"

#include "selected_attention.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include "qwen4_raw.h"

namespace mfq_qwen4_exp {
namespace raw = ::tyloi::cuda::qwen4_raw;
static ::tyloi::cuda::NumericType raw_type(const DenseTensor& x) { return x.scalar_type()==::tyloi::kFloat16 ? ::tyloi::cuda::NumericType::f16 : ::tyloi::cuda::NumericType::f32; }

std::vector<DenseTensor> live_index_pool(const DenseTensor& index,const DenseTensor& positions,
        const DenseTensor& cache_positions,int64_t pool,int64_t maximum) {
    TYLOINFER_RUNTIME_CHECK(index.is_cuda() && index.dim()==3 && index.size(0)==1 &&
        (index.scalar_type()==::tyloi::kFloat16 || index.scalar_type()==::tyloi::kFloat32) &&
        positions.device()==index.device() && cache_positions.device()==index.device() &&
        positions.dim()>=1 && positions.dim()<=3 &&
        (positions.scalar_type()==::tyloi::kInt32 || positions.scalar_type()==::tyloi::kInt64) &&
        cache_positions.dim()==1 && cache_positions.numel()>0 && cache_positions.is_contiguous() &&
        cache_positions.scalar_type()==::tyloi::kInt64 && positions.size(-1)==cache_positions.numel() &&
        pool>0 && pool<=INT_MAX && maximum>0 && maximum<=index.size(1) &&
        index.size(2)>0 && index.size(2)<=INT_MAX && cache_positions.numel()<=INT_MAX,
        "live QSA pool has incompatible index/position geometry");
    const MfqCudaGuard guard(index.device());
    auto pos=positions.contiguous();
    auto pooled=::tyloi::cuda::empty({1,cache_positions.numel(),index.size(2)},index.options());
    auto blocks=::tyloi::cuda::empty_like(cache_positions),block_positions=::tyloi::cuda::empty_like(pos);
    raw::live_pool(index.data_ptr(),pos.data_ptr(),cache_positions.data_ptr<int64_t>(),pooled.data_ptr(),blocks.data_ptr<int64_t>(),block_positions.data_ptr(),int(cache_positions.numel()),int(index.size(2)),int(pool),maximum,pos.numel(),index.stride(1),index.stride(2),raw_type(index),pos.scalar_type()==::tyloi::kInt64,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return {pooled,blocks,block_positions};
}

void fill_decode_positions(const DenseTensor& cache,const DenseTensor& positions,int64_t start,int64_t delta) {
    TYLOINFER_RUNTIME_CHECK(cache.is_cuda() && positions.is_cuda() && cache.is_contiguous() &&
        positions.is_contiguous() && cache.scalar_type()==::tyloi::kInt64 && cache.numel()>0 &&
        positions.numel()%cache.numel()==0 && positions.numel()>=cache.numel(),
        "decode position buffers have incompatible geometry");
    TYLOINFER_RUNTIME_CHECK(positions.scalar_type()==::tyloi::kInt32 || positions.scalar_type()==::tyloi::kInt64,"decode positions must use int32 or int64");
    raw::fill_positions(cache.data_ptr<int64_t>(),positions.data_ptr(),cache.numel(),positions.numel(),start,delta,positions.scalar_type()==::tyloi::kInt64,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool Injection>
DenseTensor gated_residual_activation(const DenseTensor& input,int64_t streams) {
    mfq_selected_attention::values({&input});
    TYLOINFER_RUNTIME_CHECK(streams>0,"gated-residual stream count must be positive");
    if(input.is_contiguous() && (input.scalar_type()==::tyloi::kFloat16 || input.scalar_type()==::tyloi::kFloat32)) {
        MfqCudaGuard guard(input.device());
        auto output=::tyloi::cuda::empty_like(input);const auto count=input.numel();
        if(!count)return output;
        raw::activation(input.data_ptr(),output.data_ptr(),count,float(streams),raw_type(input),Injection,mfq_current_cuda_stream());
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
        return output;
    }
    auto low=input/streams;
    if constexpr(Injection)return 2.0*::tyloi::cuda::sigmoid(low);
    else return low*::tyloi::cuda::sigmoid(low);
}

DenseTensor gated_residual_bottleneck(const DenseTensor& input,int64_t streams) {
    return gated_residual_activation<false>(input,streams);
}
DenseTensor gated_residual_injection(const DenseTensor& input,int64_t streams) {
    return gated_residual_activation<true>(input,streams);
}

DenseTensor gated_residual_mix(const DenseTensor& projection,const DenseTensor& normalized,int64_t streams) {
    mfq_selected_attention::values({&projection,&normalized});
    TYLOINFER_RUNTIME_CHECK(projection.sizes()==normalized.sizes() && streams>0 && normalized.dim()>0 &&
        normalized.size(-1)>0 && normalized.size(-1)%streams==0,"gated-residual mix shape disagrees");
    auto shape=normalized.sizes().vec();const int hidden=int(shape.back()/streams);shape.back()=hidden;
    const auto half_or_float=[](const DenseTensor& x){return x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32;};
    if(projection.is_contiguous() && normalized.is_contiguous() && half_or_float(projection) && half_or_float(normalized)) {
        MfqCudaGuard guard(normalized.device());
        const auto dtype=projection.scalar_type()==::tyloi::kFloat32 || normalized.scalar_type()==::tyloi::kFloat32?::tyloi::kFloat32: ::tyloi::kFloat16;
        auto output=::tyloi::cuda::empty(shape,normalized.options().dtype(dtype));const auto count=output.numel();
        if(count==0)return output;
        raw::mix(projection.data_ptr(),normalized.data_ptr(),output.data_ptr(),count,hidden,int(streams),raw_type(projection),raw_type(normalized),mfq_current_cuda_stream());
        return output;
    }
    auto grouped=shape;grouped.back()=streams;grouped.push_back(hidden);
    return (::tyloi::cuda::sigmoid(projection).reshape(grouped)*normalized.reshape(grouped)).mean(-2);
}
using ::tyloi::DenseTensor;

DenseTensor grouped_rms_norm(const DenseTensor& value, const DenseTensor& weight,
                        int64_t group, double eps) {
    mfq_selected_attention::values({&value, &weight});
    TYLOINFER_RUNTIME_CHECK(value.dim() >= 1 && group > 0 && value.size(-1) > 0 &&
        value.size(-1) % group == 0 && weight.dim() == 1 && weight.size(0) == value.size(-1),
        "Qwen4 grouped RMSNorm dimensions disagree");
    MfqCudaGuard guard(value.device());
    if (value.is_contiguous() && (value.scalar_type() == ::tyloi::kFloat16 || value.scalar_type() == ::tyloi::kFloat32))
        return grouped_rms_norm_cuda(value, weight.to(::tyloi::kFloat32).contiguous(), group, eps, 1.0);
    auto shape = value.sizes().vec();
    shape.back() /= group;
    shape.push_back(group);
    auto source = value.to(::tyloi::kFloat32).reshape(shape);
    auto normalized = source * ::tyloi::cuda::rsqrt((source * source).mean(-1, true) + eps);
    return (normalized.reshape(value.sizes()) * (1.0 + weight.to(::tyloi::kFloat32)))
        .to(value.scalar_type());
}

std::vector<DenseTensor> gated_residual_pre(
    const DenseTensor& input, const DenseTensor& norm, const DenseTensor& down, const DenseTensor& up,
    const std::optional<DenseTensor>& inject, int64_t hidden, int64_t streams, double eps) {
    mfq_selected_attention::values({&input, &norm, &down, &up});
    TYLOINFER_RUNTIME_CHECK(input.dim() >= 1 && hidden > 0 && streams > 0 &&
        input.size(-1) / streams == hidden && input.size(-1) % streams == 0 &&
        down.dim() == 2 && up.dim() == 2 && down.size(1) == input.size(-1) &&
        up.size(0) == input.size(-1) && up.size(1) == down.size(0),
        "Qwen4 gated-residual projection dimensions disagree");
    if (inject) {
        mfq_selected_attention::values({&input, &*inject});
        TYLOINFER_RUNTIME_CHECK(inject->dim() == 2 && inject->size(0) == streams &&
            inject->size(1) == input.size(-1), "Qwen4 injection projection dimensions disagree");
    }
    MfqCudaGuard guard(input.device());
    auto normalized = grouped_rms_norm(input, norm, hidden, eps);
    auto low = gated_residual_bottleneck(
        mfq_selected_attention::promoted_matmul(normalized, down.transpose(-1, -2)),streams);
    auto mixing = mfq_selected_attention::promoted_matmul(low, up.transpose(-1, -2));
    auto mixed = gated_residual_mix(mixing,normalized,streams);
    auto injection = inject
        ? gated_residual_injection(mfq_selected_attention::promoted_matmul(normalized, inject->transpose(-1, -2)),streams)
        : DenseTensor{};
    return {mixed, input, injection};
}

DenseTensor gated_residual_post(const DenseTensor& branch, const DenseTensor& residual,
                           const DenseTensor& injection, int64_t streams) {
    mfq_selected_attention::values({&branch, &residual, &injection});
    TYLOINFER_RUNTIME_CHECK(branch.dim() >= 1 && streams > 0 && branch.dim() == residual.dim() &&
        branch.dim() == injection.dim(), "Qwen4 residual ranks disagree");
    auto shape = branch.sizes().vec();
    shape.back() *= streams;
    TYLOINFER_RUNTIME_CHECK(residual.sizes().vec() == shape, "Qwen4 residual width disagrees");
    shape.back() = streams;
    TYLOINFER_RUNTIME_CHECK(injection.sizes().vec() == shape, "Qwen4 injection shape disagrees");
    MfqCudaGuard guard(branch.device());
    if(branch.is_contiguous() && residual.is_contiguous() && injection.is_contiguous() &&
        branch.scalar_type()==residual.scalar_type() && branch.scalar_type()==injection.scalar_type() &&
        (branch.scalar_type()==::tyloi::kFloat16 || branch.scalar_type()==::tyloi::kFloat32)) {
        auto output=::tyloi::cuda::empty_like(residual);const auto count=output.numel();
        if(count==0)return output;
        raw::post(branch.data_ptr(),residual.data_ptr(),injection.data_ptr(),output.data_ptr(),count,int(branch.size(-1)),int(streams),raw_type(branch),mfq_current_cuda_stream());
        return output;
    }
    return residual + (branch.unsqueeze(-2) * injection.unsqueeze(-1)).reshape(residual.sizes());
}

std::vector<DenseTensor> gated_residual_post_norm(const DenseTensor& branch,const DenseTensor& residual,
    const DenseTensor& injection,const DenseTensor& norm,int64_t streams,double eps) {
    mfq_selected_attention::values({&branch,&residual,&injection,&norm});
    TYLOINFER_RUNTIME_CHECK(branch.dim()>=1 && streams>0 && branch.size(-1)>0 &&
        branch.dim()==residual.dim() && branch.dim()==injection.dim(),
        "Qwen4 residual/norm ranks disagree");
    auto shape=branch.sizes().vec();shape.back()*=streams;
    TYLOINFER_RUNTIME_CHECK(residual.sizes().vec()==shape && norm.dim()==1 && norm.size(0)==shape.back(),
        "Qwen4 residual/norm width disagrees");
    shape.back()=streams;
    TYLOINFER_RUNTIME_CHECK(injection.sizes().vec()==shape,"Qwen4 residual/norm injection shape disagrees");
    const auto supported=[](const DenseTensor& x){return x.is_contiguous() &&
        (x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32);};
    if(!supported(branch) || !supported(residual) || !supported(injection)) {
        auto updated=gated_residual_post(branch,residual,injection,streams);
        return {updated,grouped_rms_norm(updated,norm,branch.size(-1),eps)};
    }
    MfqCudaGuard guard(branch.device());
    const auto dtype=branch.scalar_type()==::tyloi::kFloat32 || residual.scalar_type()==::tyloi::kFloat32 ||
        injection.scalar_type()==::tyloi::kFloat32?::tyloi::kFloat32: ::tyloi::kFloat16;
    auto updated=::tyloi::cuda::empty(residual.sizes(),residual.options().dtype(dtype));
    auto normalized=::tyloi::cuda::empty_like(updated);
    if(updated.numel()) {
        auto weight=norm.to(::tyloi::kFloat32).contiguous();
        raw::post_norm(branch.data_ptr(),residual.data_ptr(),injection.data_ptr(),weight.data_ptr<float>(),updated.data_ptr(),normalized.data_ptr(),updated.numel(),int(branch.size(-1)),int(streams),float(eps),raw_type(branch),raw_type(residual),raw_type(injection),mfq_current_cuda_stream());
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    return {updated,normalized};
}

DenseTensor block_scores(const DenseTensor& query, const DenseTensor& pooled) {
    mfq_selected_attention::values({&query, &pooled});
    MfqCudaGuard guard(query.device());
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_QSA_PREFILL_SCORE_GEMM");
    if(query.size(1)>=32 && option && option[0]=='1') {
        // Each index head owns a token matrix. This replaces thousands of
        // four-row GEMMs with a few large GEMMs and retains head reduction.
        auto q=query.to(::tyloi::kFloat32).permute({0,2,1,3}).contiguous();
        auto keys=pooled.to(::tyloi::kFloat32).transpose(-1,-2).unsqueeze(1);
        auto products=::tyloi::cuda::matmul(q,keys).permute({0,2,1,3}).clamp_min(0.0);
        return products.sum(-2)/std::sqrt(double(query.size(-1)));
    }
    return mfq_selected_attention::dots(query, pooled).sum(-2) /
        std::sqrt(double(query.size(-1)));
}

DenseTensor live_block_scores(const DenseTensor& query,const DenseTensor& pooled,const DenseTensor& positions,int64_t pool) {
    mfq_selected_attention::values({&query,&pooled});
    TYLOINFER_RUNTIME_CHECK(query.dim()==4 && query.size(0)==1 && pooled.dim()==3 && pooled.size(0)==1 &&
        query.size(3)==pooled.size(2) && query.size(2)>0 && query.size(3)>0 && pool>0 &&
        positions.is_cuda() && positions.is_contiguous() && positions.scalar_type()==::tyloi::kInt64 &&
        positions.numel()==query.size(1),"invalid live QSA score geometry");
    const bool supported=(query.scalar_type()==::tyloi::kFloat16 || query.scalar_type()==::tyloi::kFloat32) &&
        (pooled.scalar_type()==::tyloi::kFloat16 || pooled.scalar_type()==::tyloi::kFloat32);
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_QSA_LIVE_SCORES");
    if(!supported || (option && option[0]=='0')) {
        auto scores=block_scores(query,pooled);
        auto ends=(::tyloi::cuda::arange(pooled.size(1),positions.options())*pool+pool-1).reshape({1,1,-1});
        return ::tyloi::cuda::where(ends<=positions.reshape({1,-1,1}),scores,::tyloi::cuda::full_like(scores,-1e30));
    }
    MfqCudaGuard guard(query.device());
    auto out=::tyloi::cuda::empty({1,query.size(1),pooled.size(1)},query.options().dtype(::tyloi::kFloat32));
    if(!out.numel())return out;
    raw::live_scores(query.data_ptr(),pooled.data_ptr(),positions.data_ptr<int64_t>(),out.data_ptr<float>(),int(query.size(1)),int(pooled.size(1)),int(query.size(2)),int(query.size(3)),int(pool),query.stride(1),query.stride(2),query.stride(3),pooled.stride(1),pooled.stride(2),raw_type(query),raw_type(pooled),mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

DenseTensor live_topk_blocks(const DenseTensor& scores,const DenseTensor& positions,int64_t pool,int64_t count) {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_QSA_LIVE_TOPK");
    if((option && option[0]=='0') || count>1024)
        return std::get<1>(::tyloi::cuda::topk(scores,count,-1,true,false));
    TYLOINFER_RUNTIME_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.dim()==3 && scores.size(0)==1 &&
        scores.scalar_type()==::tyloi::kFloat32 && positions.is_cuda() && positions.is_contiguous() &&
        positions.scalar_type()==::tyloi::kInt64 && positions.numel()==scores.size(1) && pool>0 &&
        count>0 && count<=scores.size(2),"invalid live QSA top-k geometry");
    MfqCudaGuard guard(scores.device());
    const int rows=int(scores.size(1)),pools=int(scores.size(2));
    const auto* compact_option=tyloi::cuda::env_lookup("TYLOINFER_QSA_TOPK_LIVE_TILE");
    const bool compact=!compact_option || compact_option[0]!='0';
    int columns=pools,stage=0;
    auto keys=scores;DenseTensor ids;
    TYLOINFER_CUDA_CHECK(raw::prepare_live_topk());
    do {
        const int tiles=(columns+4095)/4096;
        auto next=::tyloi::cuda::empty({1,rows,tiles*count},scores.options());
        auto next_ids=::tyloi::cuda::empty(next.sizes(),scores.options().dtype(::tyloi::kInt64));
        raw::live_topk(keys.data_ptr<float>(),ids.defined()?ids.data_ptr<int64_t>():nullptr,next.data_ptr<float>(),next_ids.data_ptr<int64_t>(),positions.data_ptr<int64_t>(),rows,columns,pools,int(pool),int(count),stage++,compact,mfq_current_cuda_stream());
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();keys=std::move(next);ids=std::move(next_ids);columns=tiles*int(count);
    } while(columns>count);
    return ids;
}

std::vector<DenseTensor> ple_dilated_conv_silu(const DenseTensor& input,
    const DenseTensor& weight, const std::optional<DenseTensor>& state, int64_t dilation) {
    mfq_selected_attention::values({&input, &weight});
    TYLOINFER_RUNTIME_CHECK(input.dim() == 3, "Qwen4 PLE input must have [B,T,C] shape");
    auto w = weight;
    if (w.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(w.size(0) == input.size(2) && w.size(1) == 1,
            "Qwen4 PLE packed weight dimensions disagree");
        w = w.select(1, 0);
    }
    TYLOINFER_RUNTIME_CHECK(w.dim() == 2 && w.size(0) == input.size(2) &&
        w.size(1) > 0 && dilation > 0 && w.size(1) - 1 <=
            std::numeric_limits<int64_t>::max() / dilation,
        "Qwen4 PLE kernel/dilation dimensions disagree");
    const auto length = (w.size(1) - 1) * dilation;
    const std::vector<int64_t> state_shape{input.size(0), length, input.size(2)};
    if (state) {
        mfq_selected_attention::values({&input, &*state});
        TYLOINFER_RUNTIME_CHECK(state->sizes().vec() == state_shape, "Qwen4 PLE state shape disagrees");
    }
    MfqCudaGuard guard(input.device());
    auto previous = state ? state->to(input.scalar_type()) : ::tyloi::cuda::zeros(state_shape, input.options());
    auto combined = ::tyloi::cuda::cat({previous, input}, 1);
    auto output = ::tyloi::cuda::zeros(input.sizes(), input.options().dtype(::tyloi::kFloat32));
    for (int64_t tap = 0; tap < w.size(1); ++tap) {
        output = output + combined.narrow(1, tap * dilation, input.size(1)).to(::tyloi::kFloat32) *
            w.select(1, tap).to(::tyloi::kFloat32).unsqueeze(0).unsqueeze(0);
    }
    output = output * ::tyloi::cuda::sigmoid(output);
    auto next = combined.narrow(1, length ? combined.size(1) - length : 0, length).contiguous();
    return {output.to(input.scalar_type()), next};
}

DenseTensor dense_gqa_attention(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v, int64_t offset,tyloi::cuda::HotKvView hot) {
    return mfq_selected_attention::dense(
        q, k, v, offset, 1.0 / std::sqrt(double(q.size(3))),hot);
}

DenseTensor sparse_gqa_attention(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v,
                            const DenseTensor& indices,tyloi::cuda::HotKvView hot) {
    return mfq_selected_attention::sparse(
        q, k, v, indices, 1.0 / std::sqrt(double(q.size(3))), false,hot);
}

DenseTensor attention_gate(const DenseTensor& attended, const DenseTensor& gate, bool half_output) {
    return mfq_selected_attention::gated(attended, gate, half_output);
}

DenseTensor sparse_gqa_attention_gate(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v,
    const DenseTensor& indices, const DenseTensor& gate, bool half_output,tyloi::cuda::HotKvView hot) {
    return mfq_selected_attention::sparse_gated(q, k, v, indices, gate,
        1.0 / std::sqrt(double(q.size(3))), half_output,hot);
}

DenseTensor qsa_selected_tokens(const DenseTensor& blocks, const DenseTensor& positions,
    int64_t pool, int64_t budget) {
    return mfq_selected_attention::block_indices_to_tokens(blocks, positions, pool, budget);
}

DenseTensor causal_gqa_attention_gate(const DenseTensor& q, const DenseTensor& k, const DenseTensor& v,
    const DenseTensor& positions, const DenseTensor& gate, int64_t columns, bool half_output,tyloi::cuda::HotKvView hot) {
    return mfq_selected_attention::causal_gated(q, k, v, positions, gate, columns,
        1.0 / std::sqrt(double(q.size(3))), half_output,true,true,hot);
}

} // namespace mfq_qwen4_exp
