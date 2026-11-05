#include "env.h"
#include "tq.h"
#include "tq_detail.h"
#include "engine/weights/tq/image.h"
#include "engine/backends/cuda/native/blas.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

TrellisProjectionView tyloi::cuda::tq::view(const ::tyloi::encoding::TqTensor& weight) {
    const auto& layout = weight.layout();
    const auto image = weight.image();
    const auto* data = image.data_ptr<std::uint8_t>();
    return {reinterpret_cast<const std::uint32_t*>(data + layout.words),
        reinterpret_cast<const std::uint64_t*>(data + 64), data + layout.scales,
        data + layout.relative, nullptr, static_cast<int>(weight.size(0)),
        static_cast<int>(layout.stored_columns), static_cast<int>(weight.size(1)),
        layout.gslog, int(layout.columns)};
}

namespace {
bool short_q8(::tyloi::ScalarType type,int64_t tokens) {
    return tokens>=1 && tokens<=4 &&
        (type==::tyloi::kFloat32 || type==::tyloi::kFloat16);
}

void project_tokens(const TrellisProjectionView& view,tyloi::cuda::BlasResources* gemm,
        const void* input,void* output,int tokens,
        int type,void* workspace=nullptr,std::uint64_t workspace_bytes=0) {
    if(workspace && (type==0 || type==2) && tokens<=4) {
        tq_detail::status(mfq_tq_q8_tokens(view,input,output,tokens,1,type,workspace,workspace_bytes,mfq_current_cuda_stream()));
    }else if(workspace && tokens>=(type==0?32:128)) {
        if(!gemm)throw std::logic_error("TQ GEMM resource lease is missing");
        if(type==0)tq_detail::status(mfq_tq_gemm(view,input,output,tokens,workspace,workspace_bytes,
            gemm->get(),mfq_current_cuda_stream()));
        else tq_detail::status(mfq_tq_float_gemm(view,input,output,tokens,1,type,workspace,workspace_bytes,
            gemm->get(),mfq_current_cuda_stream()));
    }else tq_detail::status(mfq_tq_gemv_tokens(view,input,output,tokens,type,mfq_current_cuda_stream()));
}

::tyloi::DenseTensor projection_workspace(const ::tyloi::encoding::TqTensor& weight,int tokens,::tyloi::ScalarType type,int groups=1) {
    // A single-token projection fitting one output tile cannot amortize a
    // separate Q8 preparation launch. The direct TQ kernel also cooperates
    // across warps on these narrow contractions, instead of serializing K
    // within one warp. Parallel projection groups still share Q8 preparation.
    if(tokens==1 && weight.size(0)<=TRELLIS_GROUP_ROWS)return {};
    if(short_q8(type,tokens) && groups<=65535) {
        const auto columns=(std::uint64_t(weight.size(1))+511)/512*512;
        const auto bytes=std::uint64_t(groups)*tokens*columns*3/2;
        return ::tyloi::cuda::empty({static_cast<std::int64_t>(bytes)},weight.options().dtype(::tyloi::kUInt8));
    }
    if(tokens<tq_detail::gemm_min_tokens(type))return {};
    // Bound temporary decode storage independently of the matrix size and
    // prompt length; experts retain only their compressed images in caches.
    auto bytes=std::max<std::uint64_t>(std::uint64_t(8)<<20,
        type==::tyloi::kFloat16?
            std::uint64_t(weight.size(1))*2+15:
            std::uint64_t(weight.size(1))*8+36);
    if(type==::tyloi::kFloat16 && tokens>=256) {
        // A larger row tile amortizes decode/GEMM submissions and avoids
        // narrow GEMMs on long prefill. This remains temporary storage, bounded
        // by useful matrix rows, 64 MiB, and 1/16 of currently free VRAM.
        const auto row_bytes=std::uint64_t(weight.size(1))*2;
        const auto wanted=std::min<std::uint64_t>(std::uint64_t(64)<<20,
            std::min<std::uint64_t>(weight.size(0)/groups,65535)*row_bytes+15);
        if(wanted>bytes) {
            std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));
            bytes=std::max<std::uint64_t>(bytes,std::min<std::uint64_t>(wanted,available/16));
        }
    }
    if(bytes>(std::uint64_t(8)<<20) && tyloi::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE"))
        std::fprintf(stderr,"tq_projection_workspace tokens=%d K=%lld N=%lld groups=%d bytes=%llu\n",
            tokens,static_cast<long long>(weight.size(1)),static_cast<long long>(weight.size(0)),groups,
            static_cast<unsigned long long>(bytes));
    return ::tyloi::cuda::empty({static_cast<std::int64_t>(bytes)},weight.options().dtype(::tyloi::kUInt8));
}
}

::tyloi::DenseTensor tyloi::cuda::tq::project(const ::tyloi::encoding::TqTensor& weight, BlasResources* blas,DenseTensor input) {
    input.require_dense();
    if(!weight.is_cuda() && input.is_cuda()) {
        const auto promoted=promote_tq(weight,input.device());
        return project(promoted,blas,std::move(input));
    }
    const auto out=weight.size(0),neuron_len=weight.size(1);
    if(input.size(-1)!=neuron_len || !input.is_cuda() || input.get_device()!=weight.get_device())
        throw std::invalid_argument("TQ projection input shape or device differs");
    const auto kernel_view=view(weight);
    const MfqCudaGuard guard(weight.get_device());
    const auto source=input.contiguous();auto shape=input.sizes().vec();shape.back()=out;
    auto result=::tyloi::cuda::empty(shape,input.options());
    const auto tokens=source.numel()/neuron_len;
    const auto element=source.element_size();
    const auto workspace=projection_workspace(weight,static_cast<int>(std::min<std::int64_t>(tokens,65535)),source.scalar_type());
    auto* gemm = tokens >= tq_detail::gemm_min_tokens(source.scalar_type()) ? blas : nullptr;
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        project_tokens(kernel_view,gemm,static_cast<const std::uint8_t*>(source.data_ptr())+start*neuron_len*element,
            static_cast<std::uint8_t*>(result.data_ptr())+start*out*element,count,tq_detail::dtype(input.scalar_type()),
            workspace.defined()?workspace.data_ptr():nullptr,workspace.defined()?workspace.nbytes():0);
    }
    return result;
}
::tyloi::DenseTensor tyloi::cuda::tq::groupwise(const ::tyloi::encoding::TqTensor& weight, BlasResources* blas,DenseTensor input,std::int64_t groups) {
    input.require_dense();
    if(!input.is_cuda())throw std::invalid_argument("TQ grouped input must match the weight's group rows and width");
    if(!weight.is_cuda()) {
        const auto promoted=promote_tq(weight,input.device());
        return groupwise(promoted,blas,std::move(input),groups);
    }
    const auto out=weight.size(0),neuron_len=weight.size(1);
    if(input.get_device()!=weight.get_device() || groups<1 || groups>INT_MAX ||
       out%groups || input.dim()<2 || input.size(-1)!=neuron_len || input.size(-2)!=groups ||
       neuron_len>INT_MAX/groups)
        throw std::invalid_argument("TQ grouped input must match the weight's group rows and width");
    const auto kernel_view=view(weight);
    const MfqCudaGuard guard(weight.get_device());
    const auto source=input.contiguous();
    const auto tokens=source.numel()/(groups*neuron_len);
    const auto element=source.element_size();
    auto shape=input.sizes().vec();shape.pop_back();shape.back()=out;
    auto result=::tyloi::cuda::empty(shape,input.options());
    const auto workspace=projection_workspace(weight,static_cast<int>(std::min<std::int64_t>(tokens,65535)),source.scalar_type(),static_cast<int>(groups));
    auto* gemm = tokens >= tq_detail::gemm_min_tokens(source.scalar_type()) ? blas : nullptr;
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        const auto* x=static_cast<const uint8_t*>(source.data_ptr())+start*groups*neuron_len*element;
        auto* y=static_cast<uint8_t*>(result.data_ptr())+start*out*element;
        if(workspace.defined() && short_q8(source.scalar_type(),count)) {
            tq_detail::status(mfq_tq_q8_tokens(kernel_view,x,y,count,static_cast<int>(groups),tq_detail::dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),mfq_current_cuda_stream()));
        }else if(workspace.defined() && count>=tq_detail::gemm_min_tokens(source.scalar_type())) {
            if(!gemm)throw std::logic_error("TQ grouped GEMM resource lease is missing");
            if(source.scalar_type()==::tyloi::kFloat16)tq_detail::status(mfq_tq_grouped_gemm(kernel_view,x,y,count,static_cast<int>(groups),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
            else tq_detail::status(mfq_tq_float_gemm(kernel_view,x,y,count,static_cast<int>(groups),tq_detail::dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
        }
        else tq_detail::status(mfq_tq_grouped_tokens(kernel_view,x,y,count,static_cast<int>(groups),tq_detail::dtype(source.scalar_type()),mfq_current_cuda_stream()));
    }
    return result;
}

::tyloi::DenseTensor tyloi::cuda::tq::embedding(const ::tyloi::encoding::TqTensor& weight, DenseTensor ids) {
    ids.require_dense();
    if(!weight.is_cuda()) {
        const auto promoted=promote_tq(weight,ids.is_cuda()?ids.device(): ::tyloi::Device(::tyloi::kCUDA,mfq_current_cuda_device()));
        return embedding(promoted,std::move(ids));
    }
    const auto neuron_len=weight.size(1);
    const auto kernel_view=view(weight);
    const MfqCudaGuard guard(weight.get_device());
    auto source=ids.to(weight.device()).to(::tyloi::kInt64).contiguous();
    auto shape=ids.sizes().vec();shape.push_back(neuron_len);
    auto result=::tyloi::cuda::empty(shape,weight.options().dtype(::tyloi::kFloat16));
    for(std::int64_t start=0;start<source.numel();start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,source.numel()-start));
        tq_detail::status(mfq_tq_embedding(kernel_view,source.data_ptr<std::int64_t>()+start,count,
            static_cast<std::uint8_t*>(result.data_ptr())+start*neuron_len*2,0,mfq_current_cuda_stream()));
    }
    return result;
}

void tyloi::cuda::tq::validate_group(const std::vector<::tyloi::encoding::TqTensor>& weights) {
    if(weights.empty())throw std::invalid_argument("empty TQ projection group");
    const auto input=weights[0].size(1);
    if(input<1 || input>INT_MAX)throw std::invalid_argument("invalid TQ group input width");
    const auto device=weights[0].device();
    std::int64_t blocks=0, output=0;
    for(const auto& weight:weights) {
        if(weight.size(0)<1 ||
           !weight.is_cuda() || weight.device()!=device || weight.size(1)!=input)
            throw std::invalid_argument("TQ group input widths and CUDA devices must match");
        blocks+=(weight.size(0)+TRELLIS_GROUP_ROWS-1)/TRELLIS_GROUP_ROWS;
        if(blocks>INT_MAX || output>INT_MAX-weight.size(0))
            throw std::overflow_error("TQ projection group exceeds CUDA index range");
        output+=weight.size(0);
    }
}

TqProjectionGroup::TqProjectionGroup(std::vector<::tyloi::encoding::TqTensor> weights):weights_(std::move(weights)) {
    tyloi::cuda::tq::validate_group(weights_);
    input_=weights_[0].size(1);
    const auto device=weights_[0].device();
    std::vector<TrellisProjectionJob> jobs;jobs.reserve(weights_.size());
    std::int64_t blocks=0;
    for(const auto& weight:weights_) {
        blocks+=(weight.size(0)+TRELLIS_GROUP_ROWS-1)/TRELLIS_GROUP_ROWS;
        jobs.push_back({tyloi::cuda::tq::view(weight),output_,static_cast<int>(blocks)});
        output_+=weight.size(0);
    }
    const MfqCudaGuard guard(weights_[0].get_device());
    blocks_=static_cast<int>(blocks);
    const auto bytes=static_cast<std::int64_t>(jobs.size()*sizeof(TrellisProjectionJob));
    plan_=::tyloi::cuda::from_blob(jobs.data(),{bytes},::tyloi::TensorOptions().dtype(::tyloi::kUInt8))
        .clone().to(device,false,false).contiguous();
    host_plan_=std::move(jobs);
}

bool TqProjectionGroup::matches(const std::vector<::tyloi::encoding::TqTensor>& weights) const {
    if (weights.size() != weights_.size()) return false;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const auto& a = weights_[i];
        const auto& b = weights[i];
        const auto& x = a.layout();
        const auto& y = b.layout();
        if (a.size(0) != b.size(0) || a.size(1) != b.size(1) || a.device() != b.device() ||
            a.storage().get() != b.storage().get() ||
            a.storage().owner_before(b.storage()) || b.storage().owner_before(a.storage()) ||
            a.image().data_ptr() != b.image().data_ptr() || a.byte_offset() != b.byte_offset() ||
            x.words != y.words || x.words_bytes != y.words_bytes || x.scales != y.scales ||
            x.scales_bytes != y.scales_bytes || x.relative != y.relative || x.bytes != y.bytes ||
            x.stored_columns != y.stored_columns || x.gslog != y.gslog || x.columns != y.columns)
            return false;
    }
    return true;
}

std::vector<::tyloi::DenseTensor> TqProjectionGroup::buffers() const {
    std::vector<::tyloi::DenseTensor> result;
    if (plan_.defined()) result.push_back(plan_);
    for (const auto& weight : weights_) result.push_back(weight.image());
    return result;
}

void TqProjectionGroup::join_prefix(std::size_t count) {
    if(count<1 || count>weights_.size())throw std::invalid_argument("invalid TQ joined prefix");
    joined_prefix_=count;joined_rows_=0;
    for(std::size_t i=0;i<count;++i)joined_rows_+=weights_[i].size(0);
}

std::vector<::tyloi::DenseTensor> TqProjectionGroup::forward(tyloi::cuda::BlasResources* blas,::tyloi::DenseTensor input)const {
    input.require_dense();
    if(!defined() || !input.is_cuda() || input.device()!=plan_.device() || input.size(-1)!=input_)
        throw std::invalid_argument("TQ group input shape or device differs");
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.contiguous();
    const auto tokens=source.numel()/input_;
    const auto join=[&](std::vector<::tyloi::DenseTensor> values) {
        if(joined_prefix_>1) {
            auto combined=::tyloi::cuda::cat(std::vector<::tyloi::DenseTensor>(values.begin(),values.begin()+joined_prefix_),-1);
            values.erase(values.begin(),values.begin()+joined_prefix_);
            values.insert(values.begin(),std::move(combined));
        }
        return values;
    };
    if(tokens>=tq_detail::gemm_min_tokens(source.scalar_type())) {
        std::vector<::tyloi::DenseTensor> result;result.reserve(weights_.size());
        for(const auto& weight:weights_)result.push_back(tyloi::cuda::tq::project(weight,blas,source));
        return join(std::move(result));
    }
    if(tokens>1 || short_q8(source.scalar_type(),tokens)) {
        // Projection-major storage gives every output a contiguous view,
        // while one static row plan fuses all parallel token-reuse kernels.
        auto output=::tyloi::cuda::empty({tokens*output_},source.options());
        const auto* jobs=reinterpret_cast<const TrellisProjectionJob*>(plan_.data_ptr());
        const bool q8=short_q8(source.scalar_type(),tokens) && weights_.size()<65535;
        if(q8) {
            const auto columns=(std::uint64_t(input_)+511)/512*512;
            const bool column_scaled=std::any_of(host_plan_.begin(),host_plan_.end(),
                [](const auto& job){return job.weight.columns!=0;});
            const auto slots=column_scaled?weights_.size()+1:1;
            const auto bytes=slots*tokens*columns*3/2;
            auto workspace=::tyloi::cuda::empty({static_cast<std::int64_t>(bytes)},source.options().dtype(::tyloi::kUInt8));
            tq_detail::status(mfq_tq_q8_parallel_tokens(host_plan_.data(),jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),
                output.data_ptr(),static_cast<int>(tokens),static_cast<int>(input_),tq_detail::dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),joined_prefix_>1?static_cast<int>(joined_prefix_):0,
                static_cast<int>(joined_rows_),mfq_current_cuda_stream()));
        }else tq_detail::status(mfq_tq_parallel_tokens(jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),output.data_ptr(),
            static_cast<int>(tokens),static_cast<int>(input_),tq_detail::dtype(source.scalar_type()),mfq_current_cuda_stream()));
        std::vector<::tyloi::DenseTensor> result;result.reserve(weights_.size());
        std::int64_t offset=0;
        for(std::size_t i=0;i<weights_.size();) {
            const auto count=q8 && i==0?joined_prefix_:1;
            std::int64_t rows=0;
            for(std::size_t j=0;j<count;++j)rows+=weights_[i+j].size(0);
            auto shape=input.sizes().vec();shape.back()=rows;
            result.push_back(output.narrow(0,offset*tokens,rows*tokens).reshape(shape));offset+=rows;i+=count;
        }
        return q8?result:join(std::move(result));
    }
    auto shape=input.sizes().vec();shape.back()=output_;
    auto output=::tyloi::cuda::empty(shape,input.options());
    const auto element=source.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        tq_detail::status(mfq_tq_parallel(reinterpret_cast<const TrellisProjectionJob*>(plan_.data_ptr()),
            static_cast<int>(weights_.size()),blocks_,
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*output_*element,
            count,static_cast<int>(input_),static_cast<int>(output_),tq_detail::dtype(input.scalar_type()),mfq_current_cuda_stream()));
    }
    std::vector<::tyloi::DenseTensor> result;result.reserve(weights_.size());
    std::int64_t offset=0;
    for(std::size_t i=0;i<weights_.size();) {
        const auto count=i==0?joined_prefix_:1;
        std::int64_t rows=0;
        for(std::size_t j=0;j<count;++j)rows+=weights_[i+j].size(0);
        result.push_back(output.narrow(-1,offset,rows));offset+=rows;i+=count;
    }
    return result;
}

::tyloi::DenseTensor TqProjectionGroup::swiglu(tyloi::cuda::BlasResources* blas,::tyloi::DenseTensor input)const {
    input.require_dense();
    if(weights_.size()!=2 || weights_[0].size(0)!=weights_[1].size(0) || !input.is_cuda() ||
       input.device()!=plan_.device() || input.size(-1)!=input_)
        throw std::invalid_argument("TQ SwiGLU group requires two matching projections");
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.contiguous();
    if(source.numel()/input_>=tq_detail::gemm_min_tokens(source.scalar_type())) {
        const auto gate=tyloi::cuda::tq::project(weights_[0],blas,source),up=tyloi::cuda::tq::project(weights_[1],blas,source);
        return silu_mul_cuda(gate,up);
    }
    const auto rows=weights_[0].size(0);
    auto shape=input.sizes().vec();shape.back()=rows;
    auto output=::tyloi::cuda::empty(shape,input.options());
    const auto tokens=source.numel()/input_;
    const auto element=source.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        tq_detail::status(mfq_tq_swiglu_tokens(tyloi::cuda::tq::view(weights_[0]),tyloi::cuda::tq::view(weights_[1]),
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*rows*element,
            count,tq_detail::dtype(input.scalar_type()),mfq_current_cuda_stream()));
    }
    return output;
}
