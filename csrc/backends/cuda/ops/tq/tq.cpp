#include "env.h"
#include "tq.h"
#include "detail.h"
#include "native/cuda/blas.h"
#include "mfq_cuda_activation_ops.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tb=mfq_tensor_backend;

namespace {
bool short_q8(tb::ScalarType type,int64_t tokens) {
    return tokens>=1 && tokens<=4 &&
        (type==tb::kFloat32 || type==tb::kFloat16);
}

void project_tokens(const TqWeight& weight,mfq::cuda::BlasResources* gemm,
        const void* input,void* output,int tokens,
        int type,void* workspace=nullptr,std::uint64_t workspace_bytes=0) {
    if(workspace && (type==0 || type==2) && tokens<=4) {
        tq_detail::status(mfq_tq_q8_tokens(weight.view,input,output,tokens,1,type,workspace,workspace_bytes,mfq_current_cuda_stream()));
    }else if(workspace && tokens>=(type==0?32:128)) {
        if(!gemm)throw std::logic_error("TQ GEMM resource lease is missing");
        if(type==0)tq_detail::status(mfq_tq_gemm(weight.view,input,output,tokens,workspace,workspace_bytes,
            gemm->get(),mfq_current_cuda_stream()));
        else tq_detail::status(mfq_tq_float_gemm(weight.view,input,output,tokens,1,type,workspace,workspace_bytes,
            gemm->get(),mfq_current_cuda_stream()));
    }else tq_detail::status(mfq_tq_gemv_tokens(weight.view,input,output,tokens,type,mfq_current_cuda_stream()));
}

tb::Tensor projection_workspace(const TqWeight& weight,int tokens,tb::ScalarType type,int groups=1) {
    // A single-token projection fitting one output tile cannot amortize a
    // separate Q8 preparation launch. The direct TQ kernel also cooperates
    // across warps on these narrow contractions, instead of serializing K
    // within one warp. Parallel projection groups still share Q8 preparation.
    if(tokens==1 && weight.out<=TRELLIS_GROUP_ROWS)return {};
    if(short_q8(type,tokens) && groups<=65535) {
        const auto columns=(std::uint64_t(weight.neuron_len)+511)/512*512;
        const auto bytes=std::uint64_t(groups)*tokens*columns*3/2;
        return tb::empty({static_cast<std::int64_t>(bytes)},weight.image.options().dtype(tb::kUInt8));
    }
    if(tokens<tq_detail::gemm_min_tokens(type))return {};
    // Bound temporary decode storage independently of the matrix size and
    // prompt length; experts retain only their compressed images in caches.
    auto bytes=std::max<std::uint64_t>(std::uint64_t(8)<<20,
        type==tb::kFloat16?
            std::uint64_t(weight.neuron_len)*2+15+(weight.view.bias?std::uint64_t(tokens)*4:0):
            std::uint64_t(weight.neuron_len)*8+36);
    if(type==tb::kFloat16 && tokens>=256) {
        // A larger row tile amortizes decode/GEMM submissions and avoids
        // narrow GEMMs on long prefill. This remains temporary storage, bounded
        // by useful matrix rows, 64 MiB, and 1/16 of currently free VRAM.
        const auto row_bytes=std::uint64_t(weight.neuron_len)*2+(weight.view.bias?std::uint64_t(tokens)*4:0);
        const auto wanted=std::min<std::uint64_t>(std::uint64_t(64)<<20,
            std::min<std::uint64_t>(weight.out/groups,65535)*row_bytes+15);
        if(wanted>bytes) {
            std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));
            bytes=std::max<std::uint64_t>(bytes,std::min<std::uint64_t>(wanted,available/16));
        }
    }
    if(bytes>(std::uint64_t(8)<<20) && mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE"))
        std::fprintf(stderr,"tq_projection_workspace tokens=%d K=%lld N=%lld groups=%d bytes=%llu\n",
            tokens,static_cast<long long>(weight.neuron_len),static_cast<long long>(weight.out),groups,
            static_cast<unsigned long long>(bytes));
    return tb::empty({static_cast<std::int64_t>(bytes)},weight.image.options().dtype(tb::kUInt8));
}
}

namespace {
TqWeight promote_tq(const TqWeight& weight,const tb::Device& device) {
    TqWeight result=weight;
    result.image=weight.image.to(device,false,false).contiguous();
    const auto old=reinterpret_cast<std::uintptr_t>(weight.image.data_ptr());
    const auto target=reinterpret_cast<std::uintptr_t>(result.image.data_ptr());
    auto address=[&](const void* p){return p?target+(reinterpret_cast<std::uintptr_t>(p)-old):std::uintptr_t(0);};
    result.view.words=reinterpret_cast<const std::uint32_t*>(address(weight.view.words));
    result.view.descriptors=reinterpret_cast<const std::uint64_t*>(address(weight.view.descriptors));
    result.view.scales=reinterpret_cast<const void*>(address(weight.view.scales));
    result.view.relative=reinterpret_cast<const void*>(address(weight.view.relative));
    return result;
}
}

tb::Tensor TqWeight::forward(mfq::cuda::BlasResources* blas,tb::Tensor input)const {
    if(!image.is_cuda() && input.is_cuda())return promote_tq(*this,input.device()).forward(blas,std::move(input));
    if(input.size(-1)!=neuron_len || !input.is_cuda() || input.get_device()!=image.get_device())
        throw std::invalid_argument("TQ projection input shape or device differs");
    const MfqCudaGuard guard(image.get_device());
    const auto source=input.contiguous();auto shape=input.sizes().vec();shape.back()=out;
    auto result=tb::empty(shape,input.options());
    const auto tokens=source.numel()/neuron_len;
    const auto element=source.element_size();
    const auto workspace=projection_workspace(*this,static_cast<int>(std::min<std::int64_t>(tokens,65535)),source.scalar_type());
    auto* gemm = tokens >= tq_detail::gemm_min_tokens(source.scalar_type()) ? blas : nullptr;
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        project_tokens(*this,gemm,static_cast<const std::uint8_t*>(source.data_ptr())+start*neuron_len*element,
            static_cast<std::uint8_t*>(result.data_ptr())+start*out*element,count,tq_detail::dtype(input.scalar_type()),
            workspace.defined()?workspace.data_ptr():nullptr,workspace.defined()?workspace.nbytes():0);
    }
    return result;
}
tb::Tensor TqWeight::forward_groupwise(mfq::cuda::BlasResources* blas,tb::Tensor input,std::int64_t groups)const {
    if(!image.is_cuda())return promote_tq(*this,input.device()).forward_groupwise(blas,std::move(input),groups);
    if(!input.is_cuda() || input.get_device()!=image.get_device() || groups<1 || groups>INT_MAX ||
       out%groups || input.dim()<2 || input.size(-1)!=neuron_len || input.size(-2)!=groups ||
       neuron_len>INT_MAX/groups)
        throw std::invalid_argument("TQ grouped input must match the weight's group rows and width");
    const MfqCudaGuard guard(image.get_device());
    const auto source=input.contiguous();
    const auto tokens=source.numel()/(groups*neuron_len);
    const auto element=source.element_size();
    auto shape=input.sizes().vec();shape.pop_back();shape.back()=out;
    auto result=tb::empty(shape,input.options());
    const auto workspace=projection_workspace(*this,static_cast<int>(std::min<std::int64_t>(tokens,65535)),source.scalar_type(),static_cast<int>(groups));
    auto* gemm = tokens >= tq_detail::gemm_min_tokens(source.scalar_type()) ? blas : nullptr;
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        const auto* x=static_cast<const uint8_t*>(source.data_ptr())+start*groups*neuron_len*element;
        auto* y=static_cast<uint8_t*>(result.data_ptr())+start*out*element;
        if(workspace.defined() && short_q8(source.scalar_type(),count)) {
            tq_detail::status(mfq_tq_q8_tokens(view,x,y,count,static_cast<int>(groups),tq_detail::dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),mfq_current_cuda_stream()));
        }else if(workspace.defined() && count>=tq_detail::gemm_min_tokens(source.scalar_type())) {
            if(!gemm)throw std::logic_error("TQ grouped GEMM resource lease is missing");
            if(source.scalar_type()==tb::kFloat16)tq_detail::status(mfq_tq_grouped_gemm(view,x,y,count,static_cast<int>(groups),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
            else tq_detail::status(mfq_tq_float_gemm(view,x,y,count,static_cast<int>(groups),tq_detail::dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
        }
        else tq_detail::status(mfq_tq_grouped_tokens(view,x,y,count,static_cast<int>(groups),tq_detail::dtype(source.scalar_type()),mfq_current_cuda_stream()));
    }
    return result;
}

tb::Tensor TqWeight::embedding(tb::Tensor ids)const {
    if(!image.is_cuda())return promote_tq(*this,ids.is_cuda()?ids.device():tb::Device(tb::kCUDA,mfq_current_cuda_device())).embedding(std::move(ids));
    const MfqCudaGuard guard(image.get_device());
    auto source=ids.to(image.device()).to(tb::kInt64).contiguous();
    auto shape=ids.sizes().vec();shape.push_back(neuron_len);
    auto result=tb::empty(shape,image.options().dtype(tb::kFloat16));
    for(std::int64_t start=0;start<source.numel();start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,source.numel()-start));
        tq_detail::status(mfq_tq_embedding(view,source.data_ptr<std::int64_t>()+start,count,
            static_cast<std::uint8_t*>(result.data_ptr())+start*neuron_len*2,0,mfq_current_cuda_stream()));
    }
    return result;
}

TqProjectionGroup::TqProjectionGroup(std::vector<TqWeight> weights):weights_(std::move(weights)) {
    if(weights_.empty())throw std::invalid_argument("empty TQ projection group");
    input_=weights_[0].neuron_len;
    if(input_<1 || input_>INT_MAX)throw std::invalid_argument("invalid TQ group input width");
    const auto device=weights_[0].image.device();
    std::vector<TrellisProjectionJob> jobs;jobs.reserve(weights_.size());
    std::int64_t blocks=0;
    for(const auto& weight:weights_) {
        if(weight.out<1 || weight.out!=weight.view.rows || weight.view.input_columns!=input_ ||
           !weight.image.is_cuda() || weight.image.device()!=device || weight.neuron_len!=input_)
            throw std::invalid_argument("TQ group input widths and CUDA devices must match");
        blocks+=(weight.out+TRELLIS_GROUP_ROWS-1)/TRELLIS_GROUP_ROWS;
        if(blocks>INT_MAX || output_>INT_MAX-weight.out)
            throw std::overflow_error("TQ projection group exceeds CUDA index range");
        jobs.push_back({weight.view,output_,static_cast<int>(blocks)});
        output_+=weight.out;
    }
    const MfqCudaGuard guard(weights_[0].image.get_device());
    blocks_=static_cast<int>(blocks);
    const auto bytes=static_cast<std::int64_t>(jobs.size()*sizeof(TrellisProjectionJob));
    plan_=tb::from_blob(jobs.data(),{bytes},tb::TensorOptions().dtype(tb::kUInt8))
        .clone().to(device,false,false).contiguous();
    host_plan_=std::move(jobs);
}

void TqProjectionGroup::join_prefix(std::size_t count) {
    if(count<1 || count>weights_.size())throw std::invalid_argument("invalid TQ joined prefix");
    joined_prefix_=count;joined_rows_=0;
    for(std::size_t i=0;i<count;++i)joined_rows_+=weights_[i].out;
}

std::vector<tb::Tensor> TqProjectionGroup::forward(mfq::cuda::BlasResources* blas,tb::Tensor input)const {
    if(!defined() || !input.is_cuda() || input.device()!=plan_.device() || input.size(-1)!=input_)
        throw std::invalid_argument("TQ group input shape or device differs");
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.contiguous();
    const auto tokens=source.numel()/input_;
    const auto join=[&](std::vector<tb::Tensor> values) {
        if(joined_prefix_>1) {
            auto combined=tb::cat(std::vector<tb::Tensor>(values.begin(),values.begin()+joined_prefix_),-1);
            values.erase(values.begin(),values.begin()+joined_prefix_);
            values.insert(values.begin(),std::move(combined));
        }
        return values;
    };
    if(tokens>=tq_detail::gemm_min_tokens(source.scalar_type())) {
        std::vector<tb::Tensor> result;result.reserve(weights_.size());
        for(const auto& weight:weights_)result.push_back(weight.forward(blas,source));
        return join(std::move(result));
    }
    if(tokens>1 || short_q8(source.scalar_type(),tokens)) {
        // Projection-major storage gives every output a contiguous view,
        // while one static row plan fuses all parallel token-reuse kernels.
        auto output=tb::empty({tokens*output_},source.options());
        const auto* jobs=reinterpret_cast<const TrellisProjectionJob*>(plan_.data_ptr());
        const bool q8=short_q8(source.scalar_type(),tokens) && weights_.size()<65535;
        if(q8) {
            const auto columns=(std::uint64_t(input_)+511)/512*512;
            const bool column_scaled=std::any_of(host_plan_.begin(),host_plan_.end(),
                [](const auto& job){return job.weight.columns!=0;});
            const auto slots=column_scaled?weights_.size()+1:1;
            const auto bytes=slots*tokens*columns*3/2;
            auto workspace=tb::empty({static_cast<std::int64_t>(bytes)},source.options().dtype(tb::kUInt8));
            tq_detail::status(mfq_tq_q8_parallel_tokens(host_plan_.data(),jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),
                output.data_ptr(),static_cast<int>(tokens),static_cast<int>(input_),tq_detail::dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),joined_prefix_>1?static_cast<int>(joined_prefix_):0,
                static_cast<int>(joined_rows_),mfq_current_cuda_stream()));
        }else tq_detail::status(mfq_tq_parallel_tokens(jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),output.data_ptr(),
            static_cast<int>(tokens),static_cast<int>(input_),tq_detail::dtype(source.scalar_type()),mfq_current_cuda_stream()));
        std::vector<tb::Tensor> result;result.reserve(weights_.size());
        std::int64_t offset=0;
        for(std::size_t i=0;i<weights_.size();) {
            const auto count=q8 && i==0?joined_prefix_:1;
            std::int64_t rows=0;
            for(std::size_t j=0;j<count;++j)rows+=weights_[i+j].out;
            auto shape=input.sizes().vec();shape.back()=rows;
            result.push_back(output.narrow(0,offset*tokens,rows*tokens).reshape(shape));offset+=rows;i+=count;
        }
        return q8?result:join(std::move(result));
    }
    auto shape=input.sizes().vec();shape.back()=output_;
    auto output=tb::empty(shape,input.options());
    const auto element=source.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        tq_detail::status(mfq_tq_parallel(reinterpret_cast<const TrellisProjectionJob*>(plan_.data_ptr()),
            static_cast<int>(weights_.size()),blocks_,
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*output_*element,
            count,static_cast<int>(input_),static_cast<int>(output_),tq_detail::dtype(input.scalar_type()),mfq_current_cuda_stream()));
    }
    std::vector<tb::Tensor> result;result.reserve(weights_.size());
    std::int64_t offset=0;
    for(std::size_t i=0;i<weights_.size();) {
        const auto count=i==0?joined_prefix_:1;
        std::int64_t rows=0;
        for(std::size_t j=0;j<count;++j)rows+=weights_[i+j].out;
        result.push_back(output.narrow(-1,offset,rows));offset+=rows;i+=count;
    }
    return result;
}

tb::Tensor TqProjectionGroup::swiglu(mfq::cuda::BlasResources* blas,tb::Tensor input)const {
    if(weights_.size()!=2 || weights_[0].out!=weights_[1].out || !input.is_cuda() ||
       input.device()!=plan_.device() || input.size(-1)!=input_)
        throw std::invalid_argument("TQ SwiGLU group requires two matching projections");
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.contiguous();
    if(source.numel()/input_>=tq_detail::gemm_min_tokens(source.scalar_type())) {
        const auto gate=weights_[0].forward(blas,source),up=weights_[1].forward(blas,source);
        return silu_mul_cuda(gate,up);
    }
    const auto rows=weights_[0].out;
    auto shape=input.sizes().vec();shape.back()=rows;
    auto output=tb::empty(shape,input.options());
    const auto tokens=source.numel()/input_;
    const auto element=source.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        tq_detail::status(mfq_tq_swiglu_tokens(weights_[0].view,weights_[1].view,
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*rows*element,
            count,tq_detail::dtype(input.scalar_type()),mfq_current_cuda_stream()));
    }
    return output;
}
