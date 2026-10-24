#include "env.h"
#include "tq.h"
#include "execution/tq.h"
#include "storage/weights/tq/load.h"
#include "weight_memory.h"
#include "storage/weights/moe/expert_cache.h"
#include "quant_linear.h"
#include "moe.h"
#include "context.h"
#include "mfq_cuda_activation_ops.h"
#include "kernels/moe/pipeline.h"
#include "execution/decode_window.h"
#include "execution/options.h"
#include "backends/cpu/native/host_memory.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <list>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <climits>
#include <unordered_map>
#include <unordered_set>

namespace tb=mfq_tensor_backend;

namespace {
int dtype(tb::ScalarType type) {
    if(type==tb::kFloat16)return 0;
    if(type==tb::kBFloat16)return 1;
    if(type==tb::kFloat32)return 2;
    throw std::invalid_argument("TQ activation dtype must be FP16, BF16 or FP32");
}
void status(int code) {
    if(code<0)throw std::invalid_argument("invalid TQ CUDA projection arguments");
    if(code>=1000)throw std::runtime_error("TQ cuBLAS status "+std::to_string(code-1000));
    TYLOINFER_CUDA_CHECK(static_cast<cudaError_t>(code));
}

int gemm_min_tokens(tb::ScalarType type) {
    // Token reuse handles short decode windows. Larger groups amortize the
    // transient weight decode and GEMM launch; no model/device restriction.
    return type==tb::kFloat16?32:128;
}

bool short_q8(tb::ScalarType type,int64_t tokens) {
    return tokens>=1 && tokens<=4 &&
        (type==tb::kFloat32 || type==tb::kFloat16);
}

void project_tokens(const TqWeight& weight,const std::shared_ptr<TqGemmResources>& gemm,
        const void* input,void* output,int tokens,
        int type,void* workspace=nullptr,std::uint64_t workspace_bytes=0) {
    if(workspace && (type==0 || type==2) && tokens<=4) {
        status(mfq_tq_q8_tokens(weight.view,input,output,tokens,1,type,workspace,workspace_bytes,mfq_current_cuda_stream()));
    }else if(workspace && tokens>=(type==0?32:128)) {
        if(!gemm)throw std::logic_error("TQ GEMM resource lease is missing");
        if(type==0)status(mfq_tq_gemm(weight.view,input,output,tokens,workspace,workspace_bytes,
            gemm->get(),mfq_current_cuda_stream()));
        else status(mfq_tq_float_gemm(weight.view,input,output,tokens,1,type,workspace,workspace_bytes,
            gemm->get(),mfq_current_cuda_stream()));
    }else status(mfq_tq_gemv_tokens(weight.view,input,output,tokens,type,mfq_current_cuda_stream()));
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
    if(tokens<gemm_min_tokens(type))return {};
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

tb::Tensor TqWeight::forward(TqExec& exec,tb::Tensor input)const {
    if(!image.is_cuda() && input.is_cuda())return promote_tq(*this,input.device()).forward(exec,std::move(input));
    if(input.size(-1)!=neuron_len || !input.is_cuda() || input.get_device()!=image.get_device())
        throw std::invalid_argument("TQ projection input shape or device differs");
    const MfqCudaGuard guard(image.get_device());
    const auto source=input.contiguous();auto shape=input.sizes().vec();shape.back()=out;
    auto result=tb::empty(shape,input.options());
    const auto tokens=source.numel()/neuron_len;
    const auto element=source.element_size();
    const auto workspace=projection_workspace(*this,static_cast<int>(std::min<std::int64_t>(tokens,65535)),source.scalar_type());
    const auto gemm = tokens >= gemm_min_tokens(source.scalar_type())
        ? exec.get(image.get_device()) : std::shared_ptr<TqGemmResources>{};
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        project_tokens(*this,gemm,static_cast<const std::uint8_t*>(source.data_ptr())+start*neuron_len*element,
            static_cast<std::uint8_t*>(result.data_ptr())+start*out*element,count,dtype(input.scalar_type()),
            workspace.defined()?workspace.data_ptr():nullptr,workspace.defined()?workspace.nbytes():0);
    }
    return result;
}
tb::Tensor TqWeight::forward_groupwise(TqExec& exec,tb::Tensor input,std::int64_t groups)const {
    if(!image.is_cuda())return promote_tq(*this,input.device()).forward_groupwise(exec,std::move(input),groups);
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
    const auto gemm = tokens >= gemm_min_tokens(source.scalar_type())
        ? exec.get(image.get_device()) : std::shared_ptr<TqGemmResources>{};
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        const auto* x=static_cast<const uint8_t*>(source.data_ptr())+start*groups*neuron_len*element;
        auto* y=static_cast<uint8_t*>(result.data_ptr())+start*out*element;
        if(workspace.defined() && short_q8(source.scalar_type(),count)) {
            status(mfq_tq_q8_tokens(view,x,y,count,static_cast<int>(groups),dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),mfq_current_cuda_stream()));
        }else if(workspace.defined() && count>=gemm_min_tokens(source.scalar_type())) {
            if(!gemm)throw std::logic_error("TQ grouped GEMM resource lease is missing");
            if(source.scalar_type()==tb::kFloat16)status(mfq_tq_grouped_gemm(view,x,y,count,static_cast<int>(groups),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
            else status(mfq_tq_float_gemm(view,x,y,count,static_cast<int>(groups),dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),gemm->get(),mfq_current_cuda_stream()));
        }
        else status(mfq_tq_grouped_tokens(view,x,y,count,static_cast<int>(groups),dtype(source.scalar_type()),mfq_current_cuda_stream()));
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
        status(mfq_tq_embedding(view,source.data_ptr<std::int64_t>()+start,count,
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

std::vector<tb::Tensor> TqProjectionGroup::forward(TqExec& exec,tb::Tensor input)const {
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
    if(tokens>=gemm_min_tokens(source.scalar_type())) {
        std::vector<tb::Tensor> result;result.reserve(weights_.size());
        for(const auto& weight:weights_)result.push_back(weight.forward(exec,source));
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
            status(mfq_tq_q8_parallel_tokens(host_plan_.data(),jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),
                output.data_ptr(),static_cast<int>(tokens),static_cast<int>(input_),dtype(source.scalar_type()),workspace.data_ptr(),
                workspace.nbytes(),joined_prefix_>1?static_cast<int>(joined_prefix_):0,
                static_cast<int>(joined_rows_),mfq_current_cuda_stream()));
        }else status(mfq_tq_parallel_tokens(jobs,static_cast<int>(weights_.size()),blocks_,source.data_ptr(),output.data_ptr(),
            static_cast<int>(tokens),static_cast<int>(input_),dtype(source.scalar_type()),mfq_current_cuda_stream()));
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
        status(mfq_tq_parallel(reinterpret_cast<const TrellisProjectionJob*>(plan_.data_ptr()),
            static_cast<int>(weights_.size()),blocks_,
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*output_*element,
            count,static_cast<int>(input_),static_cast<int>(output_),dtype(input.scalar_type()),mfq_current_cuda_stream()));
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

tb::Tensor TqProjectionGroup::swiglu(TqExec& exec,tb::Tensor input)const {
    if(weights_.size()!=2 || weights_[0].out!=weights_[1].out || !input.is_cuda() ||
       input.device()!=plan_.device() || input.size(-1)!=input_)
        throw std::invalid_argument("TQ SwiGLU group requires two matching projections");
    const MfqCudaGuard guard(input.get_device());
    const auto source=input.contiguous();
    if(source.numel()/input_>=gemm_min_tokens(source.scalar_type())) {
        const auto gate=weights_[0].forward(exec,source),up=weights_[1].forward(exec,source);
        return silu_mul_cuda(gate,up);
    }
    const auto rows=weights_[0].out;
    auto shape=input.sizes().vec();shape.back()=rows;
    auto output=tb::empty(shape,input.options());
    const auto tokens=source.numel()/input_;
    const auto element=source.element_size();
    for(std::int64_t start=0;start<tokens;start+=65535) {
        const int count=static_cast<int>(std::min<std::int64_t>(65535,tokens-start));
        status(mfq_tq_swiglu_tokens(weights_[0].view,weights_[1].view,
            static_cast<const std::uint8_t*>(source.data_ptr())+start*input_*element,
            static_cast<std::uint8_t*>(output.data_ptr())+start*rows*element,
            count,dtype(input.scalar_type()),mfq_current_cuda_stream()));
    }
    return output;
}

namespace {
std::size_t available_host_bytes(std::size_t process_limit) {
    try {
        const auto memory=mfq::sample_host_memory();
        auto available=memory.available_host_bytes();
        if(process_limit)available=std::min<std::uint64_t>(available,
            memory.resident_bytes>=process_limit?0:process_limit-memory.resident_bytes);
        return static_cast<std::size_t>(std::min<std::uint64_t>(available,SIZE_MAX));
    } catch(const std::runtime_error&) {return 0;}
}
}

// Ordinary TQ projection images share a byte-budget LRU. The storage and
// lifetime policy is independent of the row precision and does not use MFE.
class TqExpertCache {
    struct HostImage {
        mfq::TqLayout layout;
        std::vector<std::uint8_t> bytes;
        std::size_t source_bytes;
        bool pin_attempted=false,registered=false;
        int pin_device=0;
        mfq::cuda::HostRegistrationRetirement retirement;
        explicit HostImage(const std::vector<std::uint8_t>& wire)
            :layout(wire.data(),wire.size()),bytes(tq_image(wire,layout)),source_bytes(wire.size()) {
            retirement.owner=this;
            retirement.dispose=[](void* owner)noexcept{delete static_cast<HostImage*>(owner);};
        }
        HostImage(const HostImage&)=delete;
        HostImage& operator=(const HostImage&)=delete;
        static std::shared_ptr<HostImage> create(const std::vector<std::uint8_t>& wire) {
            return std::shared_ptr<HostImage>(new HostImage(wire),[](HostImage* image)noexcept {
                if(image->registered)mfq::cuda::retire_host_registration(&image->retirement);
                else delete image;
            });
        }
        // Pin only a reused host image. One-use SSD reads retain bounded
        // staging; repeated RAM transfers avoid another full CPU copy.
        void try_pin() {
            if(pin_attempted)return;
            pin_attempted=true;
            TYLOINFER_CUDA_CHECK(cudaGetDevice(&pin_device));
            const auto result=cudaHostRegister(bytes.data(),bytes.size(),cudaHostRegisterPortable);
            if(result==cudaSuccess){registered=true;return;}
            if(mfq::cuda::pinned_allocation_rejected(result) || result==cudaErrorNotSupported ||
               result==cudaErrorOperatingSystem || result==cudaErrorNotPermitted ||
               result==cudaErrorHostMemoryAlreadyRegistered) {
                (void)cudaGetLastError();return;
            }
            TYLOINFER_CUDA_CHECK(result);
        }
        ~HostImage()noexcept {
            if(registered)try {
                const mfq::cuda::DeviceGuard guard(pin_device);
                if(cudaHostUnregister(bytes.data())!=cudaSuccess)(void)cudaGetLastError();
            }catch(...) {}
        }
    };
    struct HostTransfer {
        std::shared_ptr<mfq::cuda::Event> ready;
        std::vector<std::shared_ptr<HostImage>> images;
        ~HostTransfer()noexcept {try{ready->synchronize();}catch(...) {}}
    };
    struct HostEntry {
        std::shared_ptr<HostImage> image;
        std::list<std::uint64_t>::iterator recent;
    };
    struct DeviceKey {
        std::uint64_t image;
        int device;
        bool operator==(const DeviceKey&)const=default;
    };
    struct DeviceKeyHash {
        std::size_t operator()(const DeviceKey& key)const noexcept {
            return std::hash<std::uint64_t>{}(key.image)^(std::hash<int>{}(key.device)<<1);
        }
    };
    struct DeviceEntry {
        TqWeight weight;
        std::list<DeviceKey>::iterator recent;
        std::shared_ptr<mfq::cuda::Event> uploaded;
        cudaStream_t producer=nullptr;
        std::size_t source_bytes=0;
    };
    struct ViewTable {
        int device;
        cudaStream_t stream;
        tb::Tensor views;
        std::vector<std::uintptr_t> images;
        std::vector<std::uint8_t> bytes;
    };
    struct RouteReadback {
        int device;
        MfqCudaStream stream;
        std::mutex mutex;
        std::unique_ptr<mfq::cuda::HostBuffer> host;
        void* mapped=nullptr;
        bool can_map=false;
        RouteReadback(int device,MfqCudaStream stream):device(device),stream(std::move(stream)) {
            int supported=0;
            TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrCanMapHostMemory,device));
            can_map=supported!=0;
        }
    };
    struct Stage {
        mfq::cuda::HostBuffer host;
        mfq::cuda::Event uploaded;
        std::size_t cursor=0;
        bool pending=false;
        cudaStream_t stream=nullptr;
        explicit Stage(std::size_t bytes):host(bytes),uploaded(cudaEventDisableTiming) {
            mfq::cuda::charge_tensor_host_bytes(bytes);
        }
        ~Stage(){
            if(pending)try{uploaded.synchronize();}catch(...) {}
            mfq::cuda::tensor_host_bytes.fetch_sub(host.size());
        }
        void reuse(){if(pending){uploaded.synchronize();pending=false;}cursor=0;}
    };
    struct DeviceStages {
        std::array<std::unique_ptr<Stage>,2> stages;
        unsigned index=0;
    };
    // Keep the execution pool alive until all cached images and their tensor
    // storage have been destroyed; native tensor stream fences may still
    // reference streams retained by this pool during member teardown.
    std::shared_ptr<TqExec> tq_;
    std::unordered_map<std::uint64_t,HostEntry> host_;
    std::unordered_map<DeviceKey,DeviceEntry,DeviceKeyHash> device_;
    std::unordered_map<std::uint64_t,std::vector<ViewTable>> view_tables_;
    std::vector<std::shared_ptr<RouteReadback>> route_readbacks_;
    std::list<std::uint64_t> host_recent_;
    std::list<DeviceKey> device_recent_;
    std::size_t host_used_=0,device_used_=0,host_limit_=0,device_limit_=0;
    bool retain_all_=false;
    std::size_t payload_bytes_=0;
    std::size_t process_host_limit_=0;
    std::uint64_t host_evictions_=0,device_evictions_=0,source_reads_=0;
    struct GraphPool {
        MfqCudaStream stream;
        std::shared_ptr<mfq::cuda::Context> context;
    };
    std::unordered_map<cudaStream_t,GraphPool> graph_pools_;
    std::uint64_t sources_=0;
    std::mutex mutex_;
    std::unordered_map<int,DeviceStages> stages_;
    std::list<HostTransfer> host_transfers_;
    std::uint64_t calls_=0,gpu_hits_=0,host_hits_=0,source_bytes_=0,transfer_bytes_=0;
    std::uint64_t direct_transfer_bytes_=0,pin_attempts_=0,pin_failures_=0;
    double fetch_ms_=0,upload_submit_ms_=0;
    const bool profile_=mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE")!=nullptr;
    std::size_t reclaim_host_locked(std::size_t bytes) {
        const auto before=host_used_;
        while(before-host_used_<bytes && !host_recent_.empty()) {
            const auto key=host_recent_.back();host_recent_.pop_back();
            host_used_-=host_.at(key).image->bytes.size();host_.erase(key);++host_evictions_;
        }
        return before-host_used_;
    }
    std::size_t reserve_host_locked(std::size_t bytes) {
        if(!bytes || host_.empty())return 0;
        const auto available=available_host_bytes(process_host_limit_);
        return available>=bytes?0:reclaim_host_locked(bytes-available);
    }
    void report()const {
        if(profile_)std::fprintf(stderr,"tq_cache calls=%llu gpu_hits=%llu host_hits=%llu source_bytes=%llu h2d_bytes=%llu fetch_ms=%.3f upload_submit_ms=%.3f gpu_bytes=%zu host_bytes=%zu host_limit=%zu direct_h2d_bytes=%llu pin_attempts=%llu pin_failures=%llu\n",
            static_cast<unsigned long long>(calls_),static_cast<unsigned long long>(gpu_hits_),
            static_cast<unsigned long long>(host_hits_),static_cast<unsigned long long>(source_bytes_),
            static_cast<unsigned long long>(transfer_bytes_),fetch_ms_,upload_submit_ms_,device_used_,host_used_,host_limit_,
            static_cast<unsigned long long>(direct_transfer_bytes_),static_cast<unsigned long long>(pin_attempts_),
            static_cast<unsigned long long>(pin_failures_));
    }
    void upload(const tb::Tensor& destination,const std::vector<std::uint8_t>& bytes) {
        const auto started=std::chrono::steady_clock::now();
        auto& staging=stages_[destination.get_device()];
        auto* stage=staging.stages[staging.index].get();
        if(!stage || bytes.size()>stage->host.size()-stage->cursor) {
            staging.index^=1;
            auto& selected=staging.stages[staging.index];
            if(selected)selected->reuse();
            // Bound normal staging, while allowing any valid projection larger
            // than the normal chunk. This is a host DMA arena, not a GPU limit.
            const auto capacity=std::max<std::size_t>(std::size_t(32)<<20,bytes.size());
            if(!selected || selected->host.size()<capacity)selected=std::make_unique<Stage>(capacity);
            stage=selected.get();
        }
        auto* source=static_cast<std::uint8_t*>(stage->host.data())+stage->cursor;
        std::memcpy(source,bytes.data(),bytes.size());
        const auto stream=mfq_current_cuda_stream();
        // A staging arena can serve different request streams. Chain its
        // copies so the final reuse fence covers every outstanding DMA.
        if(stage->pending && stage->stream!=stream)
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,stage->uploaded.get(),0));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(destination.data_ptr(),source,bytes.size(),cudaMemcpyHostToDevice,stream));
        stage->uploaded.record(stream);stage->pending=true;stage->stream=stream;
        stage->cursor+=bytes.size();transfer_bytes_+=bytes.size();
        upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
public:
    TqExpertCache(std::size_t budget,std::size_t host_budget,std::size_t process_limit,bool retain_all,
                  std::shared_ptr<TqExec> tq)
        :tq_(std::move(tq)),
         host_limit_(host_budget?host_budget:available_host_bytes(process_limit)/5*4),device_limit_(budget),
         retain_all_(retain_all),process_host_limit_(process_limit) {
        if(!tq_)throw std::invalid_argument("TQ expert cache requires its execution pool");
    }
    std::shared_ptr<TqExec> pool() const noexcept { return tq_; }
    ~TqExpertCache(){
        host_transfers_.clear();
        for(auto& [_,pool]:graph_pools_)pool.context->end_graph_pool(pool.stream.stream());
        report();
    }
    void prepare_graph_pool() {
        const auto stream=mfq_get_current_cuda_stream();
        std::lock_guard<std::mutex> guard(mutex_);
        if(graph_pools_.count(stream.stream()))return;
        const auto context=mfq::cuda::default_context(stream.device());
        context->begin_graph_pool(stream.stream());
        try {graph_pools_.emplace(stream.stream(),GraphPool{stream,context});}
        catch(...) {context->end_graph_pool(stream.stream());throw;}
    }
    std::uint64_t register_source(std::size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        if(bytes>SIZE_MAX-payload_bytes_)throw std::overflow_error("TQ source payload exceeds size limits");
        if(++sources_>UINT32_MAX)throw std::overflow_error("too many TQ projection sources");
        payload_bytes_+=bytes;
        return sources_;
    }
    std::size_t budget_bytes() {
        std::lock_guard<std::mutex> guard(mutex_);
        return retain_all_?std::max(payload_bytes_,device_used_):device_limit_;
    }
    void resize(std::size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        retain_all_=false;
        device_limit_=bytes;
        while(device_used_>bytes && !device_recent_.empty()) {
            const auto key=device_recent_.back();device_recent_.pop_back();
            device_used_-=device_.at(key).weight.image.nbytes();device_.erase(key);++device_evictions_;
        }
    }
    std::size_t reclaim_host(std::size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        return reclaim_host_locked(bytes);
    }
    std::size_t reserve_host(std::size_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);return reserve_host_locked(bytes);
    }
    std::vector<std::pair<std::string,double>> memory_metrics() {
        std::lock_guard<std::mutex> guard(mutex_);
        std::size_t covered=0,staging=0;
        std::unordered_set<std::uint64_t> covered_images;
        covered_images.reserve(device_.size()+host_.size());
        for(const auto& [key,entry]:device_)
            if(covered_images.insert(key.image).second)covered+=entry.source_bytes;
        for(const auto& [key,entry]:host_)
            if(covered_images.insert(key).second)covered+=entry.image->source_bytes;
        for(const auto& [_,arenas]:stages_)for(const auto& stage:arenas.stages)
            if(stage)staging+=stage->host.size();
        const auto cold=payload_bytes_>covered?payload_bytes_-covered:0;
        const auto misses=calls_-gpu_hits_;
        return {{"resident_expert_cache_bytes",double(device_used_)},
            {"expert_cache_budget_bytes",double(retain_all_?std::max(payload_bytes_,device_used_):device_limit_)},
            {"ram_expert_enabled",host_used_?1.0:0.0},
            {"ram_expert_payload_bytes",double(host_used_)},
            {"ram_expert_cache_budget_bytes",double(host_limit_)},
            {"ram_expert_pcie_read_bytes",double(transfer_bytes_)},
            {"ram_expert_direct_pcie_read_bytes",double(direct_transfer_bytes_)},
            {"ram_expert_cache_hits",double(host_hits_)},
            {"ram_expert_cache_misses",double(misses-host_hits_)},
            {"ram_expert_cache_evictions",double(host_evictions_)},
            {"ssd_expert_enabled",cold?1.0:0.0},
            {"ssd_expert_payload_bytes",double(cold)},
            {"ssd_expert_bytes_read",double(source_bytes_)},
            {"ssd_expert_read_count",double(source_reads_)},
            {"expert_logical_transfer_bytes",double(transfer_bytes_)},
            {"expert_host_fetch_ms",fetch_ms_},{"expert_host_copy_ms",upload_submit_ms_},
            {"expert_decode_transfer_bytes",double(staging)},
            {"expert_cache_hits",double(gpu_hits_)},
            {"expert_cache_misses",double(misses)},
            {"expert_cache_evictions",double(device_evictions_)},
            {"host_tensor_bytes",double(mfq::cuda::tensor_host_bytes.load())}};
    }
    std::vector<std::int64_t> read_routes(const tb::Tensor& ids) {
        const auto source=ids.contiguous();
        std::vector<std::int64_t> result(source.numel());
        if(!source.numel())return result;
        if(!source.is_cuda() || (source.scalar_type()!=tb::kInt32 && source.scalar_type()!=tb::kInt64)) {
            const auto host=source.cpu().to(tb::kInt64);
            std::copy_n(host.data_ptr<std::int64_t>(),result.size(),result.data());return result;
        }
        const MfqCudaGuard device_guard(source.get_device());
        const auto stream=mfq_get_current_cuda_stream(source.get_device());
        std::shared_ptr<RouteReadback> readback;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            for(const auto& candidate:route_readbacks_)
                if(candidate->device==source.get_device() && candidate->stream.stream()==stream.stream()) {
                    readback=candidate;break;
                }
            if(!readback) {
                readback=std::make_shared<RouteReadback>(source.get_device(),stream);
                route_readbacks_.push_back(readback);
            }
        }
        // Keep a stable DMA destination per execution stream. Pageable tensor
        // allocation and GPU integer conversion are unnecessary for this ID list.
        std::lock_guard<std::mutex> guard(readback->mutex);
        const auto required=source.nbytes()+64;
        if(!readback->host || readback->host->size()<required) {
            const auto bytes=readback->host && readback->host->size()<=SIZE_MAX/2?
                std::max(required,readback->host->size()*2):required;
            readback->host=std::make_unique<mfq::cuda::HostBuffer>(bytes,readback->can_map);
            readback->mapped=nullptr;
            if(readback->can_map)TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&readback->mapped,readback->host->data(),0));
        }
        auto* values=static_cast<std::uint8_t*>(readback->host->data())+64;
        if(readback->mapped && source.scalar_type()==tb::kInt32 && source.numel()<=INT_MAX) {
            auto* ready=static_cast<std::uint32_t*>(readback->host->data());
            *static_cast<volatile std::uint32_t*>(ready)=0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            // Reuse the format-independent route publication already used by
            // DecodeWindow. The GPU publishes IDs plus a system-visible flag;
            // no copy-engine transfer or stream-wide synchronization is needed.
            mfq::cuda::moe_publish_routes(nullptr,source.data_ptr<std::int32_t>(),nullptr,
                reinterpret_cast<std::int32_t*>(readback->mapped)+16,
                static_cast<std::uint32_t*>(readback->mapped),0,static_cast<int>(source.numel()),stream.stream());
            TYLOINFER_CUDA_CHECK(cudaGetLastError());
            mfq::cuda::wait_route_publication(ready,stream.stream());
        }else {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(values,source.data_ptr(),source.nbytes(),cudaMemcpyDeviceToHost,stream.stream()));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        }
        if(source.scalar_type()==tb::kInt32) {
            std::copy_n(reinterpret_cast<const std::int32_t*>(values),result.size(),result.data());
        }else std::copy_n(reinterpret_cast<const std::int64_t*>(values),result.size(),result.data());
        return result;
    }
    tb::Tensor projection_table(std::uint64_t source,std::int64_t experts,
            const std::int64_t* ids,const std::vector<TqWeight>& weights,int device) {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto stream=mfq_current_cuda_stream();
        auto& tables=view_tables_[source];
        auto found=std::find_if(tables.begin(),tables.end(),[&](const ViewTable& table) {
            return table.device==device && table.stream==stream;
        });
        if(found==tables.end()) {
            auto views=tb::empty({experts*static_cast<std::int64_t>(sizeof(TrellisProjectionView))},
                tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
            tables.push_back({device,stream,std::move(views),std::vector<std::uintptr_t>(experts),
                std::vector<std::uint8_t>(experts*sizeof(TrellisProjectionView))});
            found=std::prev(tables.end());
        }
        // Each stream has independent metadata, so a concurrent upload cannot
        // overwrite a view another request is reading. Tables own no expert
        // images; the caller's selected weights retain their normal lifetime.
        bool changed=false;
        for(std::size_t i=0;i<weights.size();++i) {
            const auto image=reinterpret_cast<std::uintptr_t>(weights[i].image.data_ptr());
            if(found->images[ids[i]]==image)continue;
            std::memcpy(found->bytes.data()+ids[i]*sizeof(TrellisProjectionView),&weights[i].view,sizeof(TrellisProjectionView));
            found->images[ids[i]]=image;changed=true;
        }
        // One small DMA for changed metadata, no per-expert transfers and no
        // host synchronization. Resident hits submit no metadata work at all.
        if(changed)try {upload(found->views,found->bytes);}
        catch(...) {std::fill(found->images.begin(),found->images.end(),0);throw;}
        return found->views;
    }
    std::vector<TqWeight> get_rows(std::uint64_t source,const std::int64_t* ids,std::size_t count,
            const mfq::TqTable& table,std::int64_t out,int device) {
        const MfqCudaGuard device_guard(device);
        std::lock_guard<std::mutex> guard(mutex_);
        const auto resources=tq_->get(device);
        std::vector<TqWeight> result;result.reserve(count);
        const auto stream=mfq_get_current_cuda_stream(device);
        std::shared_ptr<mfq::cuda::Event> uploaded;
        for(auto it=host_transfers_.begin();it!=host_transfers_.end();)
            if(it->ready->ready())it=host_transfers_.erase(it);else ++it;
        auto transfer=host_transfers_.end();
        std::unordered_set<cudaEvent_t> waited;
        try {
        for(std::size_t index=0;index<count;++index) {
        const auto expert=static_cast<std::uint32_t>(ids[index]);
        const auto key=(source<<32)|expert;
        const DeviceKey device_key{key,device};
        ++calls_;
        const auto resident=device_.find(device_key);
        if(resident!=device_.end()) {
            if(resident->second.producer!=stream.stream()) {
                const auto event=resident->second.uploaded->get();
                if(waited.insert(event).second)TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),event,0));
                resources->retain(stream);
                mfq_cuda_record_stream(resident->second.weight.image,stream);
            }
            device_recent_.splice(device_recent_.begin(),device_recent_,resident->second.recent);
            ++gpu_hits_;result.push_back(resident->second.weight);
            if((calls_&4095)==0)report();
            continue;
        }
        const auto fetch_started=std::chrono::steady_clock::now();
        std::shared_ptr<HostImage> image;
        const auto cached=host_.find(key);
        if(cached!=host_.end()) {
            host_recent_.splice(host_recent_.begin(),host_recent_,cached->second.recent);
            image=cached->second.image;++host_hits_;
            if(resources->host_register() && !image->pin_attempted) {
                if(auto registration=mfq::cuda::try_host_registration();registration.owns_lock()) {
                    image->try_pin();++pin_attempts_;
                    if(!image->registered)++pin_failures_;
                }
            }
        } else {
            auto wire=table.slice_rows(std::uint64_t(expert)*out,std::uint64_t(expert+1)*out);
            // Only source misses allocate new host images. Reclaim optional
            // images against current RAM/commit headroom before packing one;
            // GPU hits and ordinary RAM streaming do not poll process memory.
            reserve_host_locked(wire.size());
            const bool admit=available_host_bytes(process_host_limit_)>=wire.size();
            source_bytes_+=wire.size();++source_reads_;image=HostImage::create(wire);
            while(host_used_+image->bytes.size()>host_limit_ && !host_recent_.empty()) {
                const auto old=host_recent_.back();host_recent_.pop_back();
                host_used_-=host_.at(old).image->bytes.size();host_.erase(old);++host_evictions_;
            }
            if(admit && image->bytes.size()<=host_limit_) {
                host_recent_.push_front(key);host_.emplace(key,HostEntry{image,host_recent_.begin()});host_used_+=image->bytes.size();
            }
        }
        fetch_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fetch_started).count();
        const auto needed=image->bytes.size();
        while(!retain_all_ && device_used_+needed>device_limit_ && !device_recent_.empty()) {
            const auto old=device_recent_.back();device_recent_.pop_back();
            device_used_-=device_.at(old).weight.image.numel();device_.erase(old);++device_evictions_;
        }
        auto storage=tb::empty({static_cast<std::int64_t>(needed)},tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
        auto weight=tq_view(std::move(storage),image->layout,table.width());
        if(image->registered) {
            if(!uploaded)uploaded=std::make_shared<mfq::cuda::Event>(cudaEventDisableTiming);
            if(transfer==host_transfers_.end()) {
                host_transfers_.emplace_back();transfer=std::prev(host_transfers_.end());transfer->ready=uploaded;
            }
            // A host-cache eviction must not unpin a source still consumed
            // by asynchronous DMA. One event retains the whole call's images.
            transfer->images.push_back(image);
            const auto started=std::chrono::steady_clock::now();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(weight.image.data_ptr(),image->bytes.data(),image->bytes.size(),
                cudaMemcpyHostToDevice,stream.stream()));
            transfer_bytes_+=image->bytes.size();direct_transfer_bytes_+=image->bytes.size();
            upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        }else upload(weight.image,image->bytes);
        if(retain_all_ || weight.image.numel()<=device_limit_) {
            if(!uploaded)uploaded=std::make_shared<mfq::cuda::Event>(cudaEventDisableTiming);
            device_recent_.push_front(device_key);
            device_.emplace(device_key,DeviceEntry{weight,device_recent_.begin(),uploaded,stream.stream(),image->source_bytes});
            device_used_+=weight.image.numel();
        }
        result.push_back(std::move(weight));
        if((calls_&4095)==0)report();
        }
        }catch(...) {
            // Publish readiness even when a later expert read/allocation
            // fails, before another request can observe the inserted entries.
            if(uploaded)uploaded->record(stream.stream());
            throw;
        }
        if(uploaded)uploaded->record(stream.stream());
        return result;
    }
};

std::size_t tq_expert_cache_budget_bytes(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->budget_bytes():0;
}
void resize_tq_expert_cache(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    if(cache)cache->resize(bytes);
}
std::size_t reclaim_tq_expert_host(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    return cache?cache->reclaim_host(bytes):0;
}
std::size_t reserve_tq_expert_host(const std::shared_ptr<TqExpertCache>& cache,std::size_t bytes) {
    return cache?cache->reserve_host(bytes):0;
}
std::vector<std::pair<std::string,double>> tq_expert_memory_metrics(const std::shared_ptr<TqExpertCache>& cache) {
    return cache?cache->memory_metrics():std::vector<std::pair<std::string,double>>{};
}

std::shared_ptr<TqExpertCache> ensure_tq_expert_cache(CudaExecutionContext& execution) {
    if(!execution.stores.tq_expert_cache) {
        const auto budget = execution.config.moe_device_cache_bytes.value_or(
            moe_expert_cache_budget_bytes(execution.stores.moe_expert_cache));
        const bool explicit_budget = execution.config.moe_device_cache_bytes.has_value() ||
            bool(execution.stores.moe_expert_cache);
        execution.stores.tq_expert_cache=std::make_shared<TqExpertCache>(
            budget,execution.config.moe_host_cache_bytes,
            execution.config.moe_host_physical_bytes,!explicit_budget,execution.tq);
    }
    return execution.stores.tq_expert_cache;
}

std::uint64_t register_tq_expert_source(
        const std::shared_ptr<TqExpertCache>& cache,std::size_t storage_bytes) {
    if(!cache)throw std::invalid_argument("TQ expert cache is missing");
    return cache->register_source(storage_bytes);
}

TqRoutedWeight::TqRoutedWeight(std::shared_ptr<mfq::TqTable> rows,
        std::shared_ptr<TqExpertCache> expert_cache,std::uint64_t id,
        std::int64_t count,std::int64_t outputs,std::int64_t columns)
    :table(std::move(rows)),cache(std::move(expert_cache)),source_id(id),
      experts(count),out(outputs),input(columns) {
    if(experts<1 || experts>UINT32_MAX || out<1 || input<1 ||
       !table || !cache || table->rows()!=std::uint64_t(experts)*out || table->width()!=input)
        throw std::invalid_argument("TQ routed projection shape differs");
}
tb::Tensor TqRoutedWeight::forward(tb::Tensor input_value,tb::Tensor ids)const {
    if(ids.dim()!=2 || ids.size(1)<1 || ids.size(1)>65535 || input_value.size(-1)!=input ||
       !input_value.is_cuda() || (input_value.dim()!=2 && input_value.dim()!=3))
        throw std::invalid_argument("invalid TQ routed activation geometry");
    const auto tokens=ids.size(0),routes=ids.size(1);
    if(input_value.size(0)!=tokens || (input_value.dim()==3 && input_value.size(1)!=routes))
        throw std::invalid_argument("TQ routed token/route count differs");
    const auto source=input_value.contiguous();
    auto result=tb::empty({tokens,routes,out},source.options());
    if(!tokens)return result;
    const MfqCudaGuard guard(source.get_device());
    auto host=ids.contiguous().cpu().to(tb::kInt64);
    const auto* selected=host.data_ptr<std::int64_t>();
    std::vector<TrellisProjectionView> views;views.reserve(host.numel());
    for(std::int64_t i=0;i<host.numel();++i) {
        if(selected[i]<0 || selected[i]>=experts)throw std::out_of_range("TQ routed expert ID is out of bounds");
    }
    const auto owners=cache->get_rows(source_id,selected,host.numel(),*table,out,source.get_device());
    for(const auto& owner:owners)views.push_back(owner.view);
    const auto bytes=static_cast<std::int64_t>(views.size()*sizeof(TrellisProjectionView));
    const auto metadata=tb::from_blob(views.data(),{bytes},tb::TensorOptions().dtype(tb::kUInt8))
        .clone().to(source.device(),false,false).contiguous();
    const auto* data=reinterpret_cast<const TrellisProjectionView*>(metadata.data_ptr());
    const auto element=source.element_size();
    for(std::int64_t token=0;token<tokens;++token) {
        const auto input_offset=token*(source.dim()==3?routes:1)*input*element;
        const auto output_offset=token*routes*out*element;
        status(mfq_tq_gemv_batch(data+token*routes,static_cast<const std::uint8_t*>(source.data_ptr())+input_offset,
            static_cast<std::uint8_t*>(result.data_ptr())+output_offset,static_cast<int>(routes),static_cast<int>(out),
            source.dim()==3?static_cast<int>(input):0,static_cast<int>(out),dtype(source.scalar_type()),
            0,0,mfq_current_cuda_stream()));
    }
    return result;
}

bool prepare_tq_graph_pool(const std::shared_ptr<TqExpertCache>& cache) {
    if(!cache)return false;
    cache->prepare_graph_pool();return true;
}
bool tq_graph_supports_tokens(std::int64_t tokens)noexcept {
#if CUDART_VERSION >= 11060
    return tokens>0;
#else
    return tokens==1;
#endif
}

namespace {
using TqIndexedTiles=std::array<std::vector<TrellisIndexedProjectionJob>,4>;
using TqWideTiles=std::array<std::vector<TrellisIndexedWideProjectionJob>,4>;
struct TqIndexedJobs {
    TqIndexedTiles gu,down;
    TqWideTiles wide;
    std::vector<TrellisIndexedWidePairJob> wide_gu;
};
TqIndexedJobs tq_indexed_jobs(const std::vector<TqWeight>& gate,
        const std::vector<TqWeight>& up,const std::vector<TqWeight>& down,
        const std::vector<std::vector<std::int64_t>>& slots,std::size_t begin,
        std::int64_t routes,std::int64_t input,std::int64_t hidden,std::int64_t output,bool wide=false) {
    TqIndexedTiles gu_tiles,down_tiles;
    TqWideTiles wide_tiles;
    std::vector<TrellisIndexedWidePairJob> wide_gu;
    for(std::size_t i=0;i<gate.size();++i) {
        const auto& rows=slots[begin+i];
        const auto complete=wide?rows.size()/8*8:0;
        for(std::size_t start=0;start<complete;start+=8) {
            TrellisIndexedWidePairJob job{};job.weight=gate[i].view;job.second=up[i].view;job.count=8;
            for(int t=0;t<8;++t) {
                const auto slot=rows[start+t];job.input_offsets[t]=(slot/routes)*input;job.output_offsets[t]=slot*hidden;
            }
            wide_gu.push_back(job);
        }
        // Eight columns fill the MMA instruction. Keep incomplete groups in
        // the short SIMT kernel instead of repeating padded MMA work.
        auto chunks=(rows.size()-complete+3)/4;
        for(std::size_t start=complete;start<rows.size();--chunks) {
            const auto tile=(rows.size()-start+chunks-1)/chunks;
            TrellisIndexedProjectionJob gu{},dn{};
            gu.weight=gate[i].view;gu.second=up[i].view;dn.weight=down[i].view;
            gu.count=dn.count=int(tile);
            for(int t=0;t<gu.count;++t) {
                const auto slot=rows[start+t];
                gu.input_offsets[t]=(slot/routes)*input;gu.output_offsets[t]=slot*hidden;
                dn.input_offsets[t]=slot*hidden;dn.output_offsets[t]=slot*output;
            }
            gu_tiles[tile-1].push_back(gu);
            if(!wide)down_tiles[tile-1].push_back(dn);
            start+=tile;
        }
        if(wide) {
            auto down_chunks=(rows.size()+7)/8;
            for(std::size_t start=0;start<rows.size();--down_chunks) {
                const auto tile=(rows.size()-start+down_chunks-1)/down_chunks;
                TrellisIndexedWideProjectionJob job{};job.weight=down[i].view;job.count=int(tile);
                for(int t=0;t<job.count;++t) {
                    const auto slot=rows[start+t];job.input_offsets[t]=slot*hidden;job.output_offsets[t]=slot*output;
                }
                if(tile>4)wide_tiles[tile-5].push_back(job);
                else {
                    TrellisIndexedProjectionJob short_job{};short_job.weight=job.weight;short_job.count=job.count;
                    std::copy_n(job.input_offsets,job.count,short_job.input_offsets);
                    std::copy_n(job.output_offsets,job.count,short_job.output_offsets);
                    down_tiles[tile-1].push_back(short_job);
                }
                start+=tile;
            }
        }
    }
    return {std::move(gu_tiles),std::move(down_tiles),std::move(wide_tiles),std::move(wide_gu)};
}
struct TqGraphLease {std::array<std::vector<TqWeight>,3> weights;void clear(){for(auto& w:weights)w.clear();}};
const char tq_graph_lease_identity=0;
struct TqMoeGraphStage {
    std::array<std::shared_ptr<TqRoutedWeight>,3> banks;
    std::shared_ptr<TqGraphLease> lease;
    MfqCudaStream stream;
    std::unique_ptr<mfq::cuda::HostBuffer> host_ids;
    std::array<tb::Tensor,3> views;
    tb::Tensor input,ids,weights,hidden,pairs,result,plan,route_snapshot;
    tb::ScalarType route_dtype;
    void* mapped_routes=nullptr;
    std::vector<int64_t> widened_routes;
    int routes,type,tokens;
    std::size_t slots=0,route_bytes=0,plan_bytes=0;
    bool wide=false;
    struct Plane {std::size_t offset=0;unsigned capacity=0,count=0;int tile=0,kind=0;};
    std::array<Plane,13> planes{};
    struct KernelChunk {int plane;std::size_t first,segment;unsigned capacity;cudaGraphNode_t node;};
    std::vector<KernelChunk> kernels;
    TqMoeGraphStage(const TqMoeBanks& b,const tb::Tensor& source,int count,tb::ScalarType route_type,
                   std::shared_ptr<TqGraphLease> l)
        :banks{b.gate,b.up,b.down},lease(std::move(l)),stream(mfq_get_current_cuda_stream(source.get_device())),
         route_dtype(route_type),
         routes(count),type(dtype(source.scalar_type())),tokens(int(source.size(0))) {
        slots=std::size_t(tokens)*routes;
        if(slots>INT_MAX)throw std::overflow_error("TQ graph route count exceeds plan limits");
        route_bytes=slots*sizeof(int64_t);
        if(tokens>1) {
            const auto pool=b.gate->cache->pool();
            if(!pool)throw std::logic_error("TQ graph cache has no execution pool");
            wide=type==0 && pool->get(source.get_device())->indexed_mma();
            for(int i=0;i<int(planes.size());++i) {
                auto& plane=planes[i];
                // GU1..4, GU8, Down1..4, Down5..8, in execution order.
                plane.kind=i<4?0:i==4?1:i<9?2:3;
                plane.tile=i<4?i+1:i==4?8:i-4;
                plane.offset=plan_bytes;
                // Every job consumes tile distinct routing slots. This bound
                // reserves less metadata than giving every tile all slots.
                plane.capacity=(plane.kind==1 || plane.kind==3) && !wide?0:unsigned(slots/plane.tile);
                const auto bytes=plane.kind==1?sizeof(TrellisIndexedWidePairJob):
                    plane.kind==3?sizeof(TrellisIndexedWideProjectionJob):sizeof(TrellisIndexedProjectionJob);
                if(plane.capacity>(std::size_t(INT64_MAX)-route_bytes-plan_bytes)/bytes)
                    throw std::overflow_error("TQ graph plan size exceeds tensor limits");
                plan_bytes+=plane.capacity*bytes;
            }
            plan=tb::empty({static_cast<int64_t>(plan_bytes)},source.options().dtype(tb::kUInt8));
            kernels.reserve(planes.size()*((slots+65534)/65535));
        }
        int can_map=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,source.get_device()));
        const auto* mapped_option=mfq::cuda::env_lookup("TYLOINFER_TQ_GRAPH_MAPPED_ROUTES");
        const bool integral=route_dtype==tb::kInt32 || route_dtype==tb::kInt64;
        const auto words=slots*(route_dtype==tb::kInt64?2:1);
        const bool map=can_map && integral && words<=INT_MAX && (!mapped_option || mapped_option[0]!='0');
        // The first cache line is the publication flag. Int64 IDs are copied
        // as two int32 words, preserving their full range without narrowing.
        host_ids=std::make_unique<mfq::cuda::HostBuffer>(64+route_bytes+plan_bytes,map);
        if(map)TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_routes,host_ids->data(),0));
        if(mapped_routes) {
            route_snapshot=tb::empty({tokens,count},source.options().dtype(route_dtype));
            if(route_dtype==tb::kInt32)widened_routes.resize(slots);
        }
        input=tb::empty_like(source);
        ids=tb::empty({tokens,count},source.options().dtype(tb::kInt64));
        weights=tb::empty({tokens,count},source.options().dtype(tb::kFloat32));
        hidden=tb::empty({int64_t(slots),b.up->out},source.options());
        pairs=tb::empty({tokens,count,b.down->out},source.options());
        result=tb::empty({tokens,b.down->out},source.options());
        if(tokens==1)for(int role=0;role<3;++role)views[role]=banks[role]->cache->projection_table(
            banks[role]->source_id,banks[role]->experts,nullptr,{},source.get_device());
        mfq::cuda::charge_tensor_host_bytes(host_ids->size());
    }
    void reset_routes() {
        if(!mapped_routes)return;
        *static_cast<volatile uint32_t*>(host_ids->data())=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    void* route_data()const {return static_cast<uint8_t*>(host_ids->data())+64;}
    void publish_routes(const tb::Tensor& current,bool views,const int64_t* device_ids) {
        if(!mapped_routes) {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(route_data(),device_ids,route_bytes,cudaMemcpyDeviceToHost,stream.stream()));
            return;
        }
        reset_routes();
        const auto* source=&current;
        if(!views || !current.is_contiguous()){route_snapshot.copy_(current);source=&route_snapshot;}
        mfq::cuda::moe_publish_routes(nullptr,reinterpret_cast<const int32_t*>(source->data_ptr()),nullptr,
            reinterpret_cast<int32_t*>(static_cast<uint8_t*>(mapped_routes)+64),
            static_cast<uint32_t*>(mapped_routes),0,int(slots*(route_dtype==tb::kInt64?2:1)),stream.stream());
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    ~TqMoeGraphStage() {
        // The D2H destination belongs to this stage, rather than the tensor
        // allocator. A window destroys its graphs before releasing the stage.
        int previous=0;const bool restore=cudaGetDevice(&previous)==cudaSuccess;
        (void)cudaSetDevice(stream.device());(void)cudaStreamSynchronize(stream.stream());
        mfq::cuda::tensor_host_bytes.fetch_sub(host_ids->size());
        if(restore)(void)cudaSetDevice(previous);
    }
    void load(mfq::cuda::DecodeWindow* window=nullptr) {
        const MfqCudaStreamGuard guard(stream);
        const auto* selected=static_cast<const int64_t*>(route_data());
        if(mapped_routes && route_dtype==tb::kInt32) {
            std::copy_n(static_cast<const int32_t*>(route_data()),slots,widened_routes.data());
            selected=widened_routes.data();
        }
        for(std::size_t i=0;i<slots;++i)if(selected[i]<0 || selected[i]>=banks[0]->experts)
            throw std::out_of_range("TQ graph expert ID is out of bounds");
        std::vector<int64_t> unique;
        std::vector<std::vector<int64_t>> grouped;
        if(tokens==1)unique.assign(selected,selected+routes);
        else {
            std::unordered_map<int64_t,std::size_t> positions;
            for(std::size_t i=0;i<slots;++i) {
                const auto [entry,inserted]=positions.emplace(selected[i],unique.size());
                if(inserted){unique.push_back(selected[i]);grouped.emplace_back();}
                grouped[entry->second].push_back(int64_t(i));
            }
        }
        // The preceding GPU segment has consumed the prior layer's images.
        // Keep one live layer across all TQ sources, independently of cache
        // ownership, so graph capture cannot pin every layer's experts.
        lease->clear();
        for(int role=0;role<3;++role) {
            const auto& bank=*banks[role];
            lease->weights[role]=bank.cache->get_rows(bank.source_id,unique.data(),unique.size(),*bank.table,bank.out,stream.device());
            if(tokens==1) {
                const auto table=bank.cache->projection_table(bank.source_id,bank.experts,unique.data(),
                    lease->weights[role],stream.device());
                if(table.data_ptr()!=views[role].data_ptr())throw std::logic_error("TQ graph view table address changed");
            }
        }
        if(tokens==1)return;
        auto [gu,dn,mma,gu8]=tq_indexed_jobs(lease->weights[0],lease->weights[1],lease->weights[2],
            grouped,0,routes,banks[0]->input,banks[1]->out,banks[2]->out,wide);
        auto* packet=static_cast<std::uint8_t*>(route_data())+route_bytes;
        std::memset(packet,0,plan_bytes);
        for(int i=0;i<int(planes.size());++i) {
            auto& plane=planes[i];
            const auto fill=[&](const auto& jobs) {
                if(jobs.size()>plane.capacity)throw std::overflow_error("TQ graph job count exceeds prepared plan");
                plane.count=unsigned(jobs.size());
                if(!jobs.empty())std::memcpy(packet+plane.offset,jobs.data(),jobs.size()*sizeof(jobs[0]));
            };
            if(plane.kind==0)fill(gu[plane.tile-1]);
            else if(plane.kind==1)fill(gu8);
            else if(plane.kind==2)fill(dn[plane.tile-1]);
            else fill(mma[plane.tile-5]);
        }
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(plan.data_ptr(),packet,plan_bytes,cudaMemcpyHostToDevice,stream.stream()));
        if(window)for(const auto& kernel:kernels) {
            const auto count=kernel.first>=planes[kernel.plane].count?0:
                unsigned(std::min<std::size_t>(kernel.capacity,planes[kernel.plane].count-kernel.first));
            window->set_kernel_rows(kernel.segment,kernel.node,count);
        }
        if(window && wide && mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE")) {
            unsigned mask=0,jobs=0;
            for(int i=9;i<int(planes.size());++i)if(planes[i].count){mask|=1u<<planes[i].tile;jobs+=planes[i].count;}
            if(planes[4].count)mask|=1u<<16;
            static std::atomic<unsigned> seen{0};
            if(mask & ~seen.fetch_or(mask,std::memory_order_relaxed))
                std::fprintf(stderr,"tq_indexed_dispatch mode=graph dtype=%d tokens=%d wide_jobs=%u gu8_jobs=%u reuse_mask=%u plan_bytes=%zu\n",
                    type,tokens,jobs,planes[4].count,mask,plan_bytes);
        }
    }
    void indexed(float limit,mfq::cuda::DecodeWindow& window,bool recording,const void* input_values) {
#if CUDART_VERSION >= 11060
        if(recording)kernels.clear();
        for(int i=0;i<int(planes.size());++i) {
            const auto& plane=planes[i];
            for(std::size_t first=0;first<plane.capacity;first+=65535) {
                const auto capacity=unsigned(std::min<std::size_t>(65535,plane.capacity-first));
                const auto active=first>=plane.count?0:
                    unsigned(std::min<std::size_t>(capacity,plane.count-first));
                const auto count=recording?capacity:std::max(1u,active);
                const auto* address=plan.data_ptr<std::uint8_t>()+plane.offset;
                const auto* jobs=reinterpret_cast<const TrellisIndexedProjectionJob*>(address)+first;
                if(plane.kind==0)status(mfq_tq_indexed_swiglu(jobs,input_values,hidden.data_ptr(),int(count),
                    int(banks[1]->out),type,plane.tile,limit,stream.stream()));
                else if(plane.kind==1)status(mfq_tq_indexed_wide_swiglu(
                    reinterpret_cast<const TrellisIndexedWidePairJob*>(address)+first,
                    input_values,hidden.data_ptr(),int(count),int(banks[1]->out),type,plane.tile,limit,stream.stream()));
                else if(plane.kind==2)status(mfq_tq_indexed(jobs,hidden.data_ptr(),pairs.data_ptr(),int(count),
                    int(banks[2]->out),type,plane.tile,stream.stream()));
                else status(mfq_tq_indexed_wide(reinterpret_cast<const TrellisIndexedWideProjectionJob*>(address)+first,
                    hidden.data_ptr(),pairs.data_ptr(),int(count),int(banks[2]->out),type,plane.tile,stream.stream()));
                if(recording) {
                    cudaStreamCaptureStatus capture;const cudaGraphNode_t* dependencies=nullptr;std::size_t n=0;
                    // CUDA 13 folded the versioned capture queries into the unversioned API
                    // and added an edge-data output before the dependency count.
#if CUDART_VERSION >= 13000
                    TYLOINFER_CUDA_CHECK(cudaStreamGetCaptureInfo(stream.stream(),&capture,nullptr,nullptr,&dependencies,nullptr,&n));
#else
                    TYLOINFER_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream.stream(),&capture,nullptr,nullptr,&dependencies,&n));
#endif
                    if(capture!=cudaStreamCaptureStatusActive || n!=1)throw std::logic_error("TQ kernel capture has no unique node");
                    kernels.push_back({i,first,window.segments()-1,capacity,dependencies[0]});
                }
            }
            if(i==4)window.mark_current("gate_up_end");
        }
#else
        (void)limit;(void)window;(void)recording;
        throw std::runtime_error("multi-token TQ graph plan updates require CUDA11.6 or newer");
#endif
    }
};
tb::Tensor tq_graph_forward(const TqMoeBanks& banks,const tb::Tensor& source,const tb::Tensor& ids,
        const tb::Tensor& weights,float limit,mfq::cuda::DecodeWindow& window) {
    window.mark_current("expert_begin");
    auto lease=window.workspace<TqGraphLease>(&tq_graph_lease_identity,[]{return std::make_shared<TqGraphLease>();});
    auto stage=window.workspace<TqMoeGraphStage>(&banks,[&] {
        return std::make_shared<TqMoeGraphStage>(banks,source,int(ids.size(1)),ids.scalar_type(),lease);
    });
    if(stage->routes!=ids.size(1) || stage->input.sizes()!=source.sizes() || stage->input.scalar_type()!=source.scalar_type() ||
       stage->route_dtype!=ids.scalar_type())
        throw std::invalid_argument("TQ graph stage geometry changed");
    const bool views=mfq::cuda::runtime_options::window_input_views() && source.is_contiguous() &&
        ids.is_contiguous() && weights.is_contiguous() && weights.scalar_type()==tb::kFloat32;
    if(mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE")) {
        const unsigned mode=(stage->mapped_routes?(stage->route_dtype==tb::kInt32?1u:2u):4u)|(views?8u:0u);
        static std::atomic<unsigned> seen{0};const unsigned bit=1u<<mode;
        if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
            std::fprintf(stderr,"tq_graph_handoff mapped_routes=%d route_bits=%d input_views=%d\n",
                int(stage->mapped_routes!=nullptr),stage->route_dtype==tb::kInt32?32:64,int(views));
    }
    if(!views){stage->input.copy_(source);stage->weights.copy_(weights);}
    const int64_t* device_ids=nullptr;
    if(views && ids.scalar_type()==tb::kInt64)device_ids=ids.data_ptr<int64_t>();
    else if(stage->tokens==1 || !stage->mapped_routes || !views) {
        stage->ids.copy_(ids);device_ids=stage->ids.data_ptr<int64_t>();
    }
    stage->publish_routes(ids,views,device_ids);
    window.mark_current("cpu_wait_begin");
    cudaStreamCaptureStatus capture;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stage->stream.stream(),&capture));
    if(capture==cudaStreamCaptureStatusNone) {
        // DecodeWindow also calls the body during preparation/warmup. Execute
        // real transfers there; host_barrier itself only records a callback.
        if(stage->mapped_routes)mfq::cuda::wait_route_publication(static_cast<const uint32_t*>(stage->host_ids->data()),stage->stream.stream());
        else TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stage->stream.stream()));
        stage->load();
    }
    window.enroll({stage.get(),[stage]{stage->reset_routes();},[stage,&window] {
        if(!stage->mapped_routes)return;
        mfq::cuda::wait_route_publication(static_cast<const uint32_t*>(stage->host_ids->data()),stage->stream.stream());
        stage->load(&window);
    },[lease]{lease->clear();},[]{},[lease]{lease->clear();},{}});
    window.host_barrier([stage,&window]{if(!stage->mapped_routes)stage->load(&window);});
    window.mark_current("hot_begin");
    const auto* input_values=views?source.data_ptr():stage->input.data_ptr();
    if(stage->tokens>1)stage->indexed(limit,window,capture==cudaStreamCaptureStatusActive,input_values);
    else {
    status(mfq_tq_routed_swiglu(reinterpret_cast<const TrellisProjectionView*>(stage->views[0].data_ptr()),
        reinterpret_cast<const TrellisProjectionView*>(stage->views[1].data_ptr()),device_ids,
        input_values,stage->hidden.data_ptr(),stage->routes,int(banks.up->out),stage->type,limit,stage->stream.stream()));
    window.mark_current("gate_up_end");
    status(mfq_tq_routed_down(reinterpret_cast<const TrellisProjectionView*>(stage->views[2].data_ptr()),
        device_ids,stage->hidden.data_ptr(),stage->pairs.data_ptr(),stage->routes,
        int(banks.down->out),int(banks.up->out),stage->type,stage->stream.stream()));
    }
    window.mark_current("down_end");
    status(mfq_tq_weighted_reduce(stage->pairs.data_ptr(),views?weights.data_ptr<float>():stage->weights.data_ptr<float>(),stage->result.data_ptr(),
        stage->tokens,stage->routes,int(banks.down->out),stage->type,stage->stream.stream()));
    window.mark_current("hot_end");
    return stage->result;
}
}

tb::Tensor TqMoeBanks::forward(tb::Tensor input_value,tb::Tensor ids,
        tb::Tensor weights,float limit,CudaProfiler* profiler,std::int64_t expert_begin,std::int64_t expert_end)const {
    const auto phase=[&](const char* name,auto&& operation) {
        return profiler?profiler->measure(name,operation):operation();
    };
    if(!gate || !up || !down || !gate->cache || !up->cache || !down->cache ||
       gate->cache->pool()!=up->cache->pool() || gate->cache->pool()!=down->cache->pool() ||
       gate->experts!=up->experts || up->experts!=down->experts ||
       gate->input!=up->input || gate->out!=up->out || down->input!=up->out ||
       ids.dim()!=2 || ids.size(1)<1 || ids.size(1)>INT_MAX || input_value.dim()!=2 ||
       !input_value.is_cuda() || weights.device()!=input_value.device() ||
       input_value.size(0)!=ids.size(0) || input_value.size(1)!=up->input ||
       weights.sizes().vec()!=ids.sizes().vec() || !std::isfinite(limit) || limit<0)
        throw std::invalid_argument("invalid TQ three-bank MoE geometry");
    if(expert_end==-1)expert_end=gate->experts;
    if(expert_begin<0 || expert_end<expert_begin || expert_end>gate->experts)
        throw std::invalid_argument("invalid TQ expert ownership range");
    const bool partitioned=expert_begin!=0 || expert_end!=gate->experts;
    const auto tokens=ids.size(0),routes=ids.size(1),hidden=up->out,width=down->out;
    if(tokens>INT_MAX || hidden>INT_MAX || width>INT_MAX || up->input>INT_MAX)
        throw std::overflow_error("TQ MoE activation dimensions exceed CUDA limits");
    const auto source=input_value.contiguous();
    const MfqCudaGuard guard(source.get_device());
    if(auto* window=mfq::cuda::DecodeWindow::recording();!partitioned && window && tq_graph_supports_tokens(tokens) && ids.device()==source.device())
        return tq_graph_forward(*this,source,ids,weights,limit,*window);
    auto result=tb::empty({tokens,width},source.options());
    if(!tokens)return result;
    const auto type=dtype(source.scalar_type());
    const auto pool=gate->cache->pool();
    const auto coefficients=weights.to(tb::kFloat32).contiguous();
    // Exactly one route readback for all three banks, independent of the
    // number of selected tokens and their expert reuse pattern.
    // Router output is normally int32. Widen this tiny ID list on CPU after
    // the readback, avoiding a GPU allocation/conversion before every wait.
    const auto host=phase("tq.moe.routes",[&] {return gate->cache->read_routes(ids);});
    const auto* selected=host.data();
    std::vector<std::int64_t> unique;
    std::vector<std::vector<std::int64_t>> slots;
    if(static_cast<std::uint64_t>(up->experts)<=host.size()) {
        // Dense counters use no more memory than the route readback. Count
        // first to reserve each expert's slots once, then reuse counters as
        // indices. Sparse expert domains retain bounded hash lookup below.
        std::vector<std::size_t> positions(static_cast<std::size_t>(up->experts),0);
        for(const auto expert:host) {
            if(expert<0 || expert>=up->experts)throw std::out_of_range("TQ routed expert ID is out of bounds");
            if(expert<expert_begin || expert>=expert_end)continue;
            if(!positions[expert]++)unique.push_back(expert);
        }
        slots.resize(unique.size());
        for(std::size_t i=0;i<unique.size();++i) {
            auto& count=positions[unique[i]];slots[i].reserve(count);count=i;
        }
        for(std::size_t i=0;i<host.size();++i)
            if(selected[i]>=expert_begin && selected[i]<expert_end)slots[positions[selected[i]]].push_back(i);
    }else {
        std::unordered_map<std::int64_t,std::size_t> positions;
        positions.reserve(host.size());
        for(std::size_t i=0;i<host.size();++i) {
            const auto expert=selected[i];
            if(expert<0 || expert>=up->experts)throw std::out_of_range("TQ routed expert ID is out of bounds");
            if(expert<expert_begin || expert>=expert_end)continue;
            const auto [position,inserted]=positions.emplace(expert,unique.size());
            if(inserted){unique.push_back(expert);slots.emplace_back();}
            slots[position->second].push_back(i);
        }
    }
    if(unique.empty())return tb::zeros({tokens,width},source.options());
    // Choose GEMM by each expert's actual token count. A long prompt can
    // still have many sparsely selected experts that belong in indexed jobs.
    std::size_t long_groups=0;
    for(std::size_t i=0;i<unique.size();++i)if(slots[i].size()>=std::size_t(gemm_min_tokens(source.scalar_type()))) {
        std::swap(unique[i],unique[long_groups]);std::swap(slots[i],slots[long_groups]);++long_groups;
    }
    const auto resources=pool->get(source.get_device());
    std::size_t prefill_groups=0;
    if(type==0 && width==up->input && hidden<=INT_MAX/2) {
        // Batch up to two 128-token reuse tiles to amortize expert dispatch.
        // Larger groups retain GEMM, which decodes each weight tile once.
        for(std::size_t i=0;i<long_groups;++i)if(slots[i].size()<=256) {
            std::swap(unique[i],unique[prefill_groups]);std::swap(slots[i],slots[prefill_groups]);++prefill_groups;
        }
    }
    auto pairs=partitioned?tb::zeros({tokens,routes,width},source.options()):tb::empty({tokens,routes,width},source.options());
    auto flat_pairs=pairs.reshape({tokens*routes,width});
    const auto upload_plan=[&](const auto& values) {
        if(values.empty())return tb::Tensor{};
        const auto bytes=static_cast<std::int64_t>(values.size()*sizeof(values[0]));
        return tb::from_blob(const_cast<void*>(static_cast<const void*>(values.data())),{bytes},
            tb::TensorOptions().dtype(tb::kUInt8)).clone().to(source.device(),false,false).contiguous();
    };
    if(prefill_groups) {
        // Similar token counts share a compute tile. A frequently selected
        // expert must not force padding every sparse expert to its capacity.
        std::vector<std::pair<int64_t,std::vector<int64_t>>> sorted;
        sorted.reserve(prefill_groups);
        for(std::size_t i=0;i<prefill_groups;++i)sorted.emplace_back(unique[i],std::move(slots[i]));
        std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second.size()>b.second.size();});
        std::size_t selected_slots=0;
        for(std::size_t i=0;i<prefill_groups;++i) {
            unique[i]=sorted[i].first;slots[i]=std::move(sorted[i].second);selected_slots+=slots[i].size();
        }
        const auto stream=mfq_get_current_cuda_stream(source.get_device());
        resources->retain(stream);
        std::size_t gemm_groups=0;
        while(gemm_groups<prefill_groups && (!resources->prefill_mma() || slots[gemm_groups].size()>128))++gemm_groups;
        const auto job_bytes=(prefill_groups*sizeof(TrellisPrefillMoeJob)+15)&~std::size_t(15);
        TqPrefillPlanLease lease{resources->prefill_plan(job_bytes+selected_slots*sizeof(int64_t)),stream.stream()};
        auto* host_bytes=static_cast<std::uint8_t*>(lease.plan->host.data_ptr());
        auto* device_bytes=static_cast<std::uint8_t*>(lease.plan->device.data_ptr());
        auto* host_slots=reinterpret_cast<int64_t*>(host_bytes+job_bytes);
        std::size_t first=0;
        for(std::size_t i=0;i<prefill_groups;++i) {
            std::memcpy(host_slots+first,slots[i].data(),slots[i].size()*sizeof(int64_t));first+=slots[i].size();
        }
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(device_bytes+job_bytes,host_slots,selected_slots*sizeof(int64_t),
            cudaMemcpyHostToDevice,stream.stream()));
        const auto free_bytes=[&] {
            return phase("tq.moe.capacity",[&] {
                std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));return available;
            });
        };
        const auto available=free_bytes();
        const auto max_width=std::uint64_t(std::max(up->input,hidden));
        const auto output_width=std::uint64_t(std::max(up->input,2*hidden));
        const auto max_capacity=(slots[0].size()+15)&~std::size_t(15);
        const auto minimum_activations=max_capacity*(gemm_groups?2*max_width+4*output_width:4*max_width+2*hidden)+32;
        std::uint64_t useful_activations=gemm_groups?
            max_capacity*(2*max_width+4*output_width)*std::min<std::size_t>(16,gemm_groups)+32:0;
        if(gemm_groups<prefill_groups) {
            const auto short_capacity=(slots[gemm_groups].size()+15)&~std::size_t(15);
            const auto useful=short_capacity*(4*max_width+2*hidden)*(prefill_groups-gemm_groups)+32;
            useful_activations=std::max<std::uint64_t>(useful_activations,
                std::min<std::uint64_t>(std::uint64_t(16)<<20,useful));
        }
        const auto activation_budget=std::max<std::uint64_t>(minimum_activations,
            std::min<std::uint64_t>({std::uint64_t(gemm_groups?64:16)<<20,available/16,useful_activations}));
        auto activations=tb::empty({static_cast<int64_t>(activation_budget)},source.options().dtype(tb::kUInt8));
        tb::Tensor decoded;
        cublasHandle_t blas=nullptr;
        if(gemm_groups) {
            const auto useful=4*std::uint64_t(up->input)*hidden*std::min<std::size_t>(16,gemm_groups);
            const auto bytes=std::max<std::uint64_t>(max_width*2,
                std::min<std::uint64_t>({std::uint64_t(128)<<20,available/16,useful}));
            decoded=tb::empty({static_cast<int64_t>(bytes)},source.options().dtype(tb::kUInt8));
            blas=resources->get();
        }
        const auto image_bound=[](const TqRoutedWeight& w) {
            return std::uint64_t(w.out)*(((std::uint64_t(w.input)+255)&~std::uint64_t(255))*2+16)+512;
        };
        const auto image_per_expert=image_bound(*gate)+image_bound(*up)+image_bound(*down);
        // Image leases and compute scratch have independent live budgets.
        // Account for allocated scratch before admitting compressed misses.
        const auto image_batch=std::min<std::size_t>(prefill_groups,
            std::max<std::uint64_t>(1,(free_bytes()/2)/image_per_expert));
        if(mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE"))std::fprintf(stderr,
            "tq_prefill_moe jobs=%zu gemm_jobs=%zu scratch=%lld decoded=%lld image_batch=%zu\n",
            prefill_groups,gemm_groups,static_cast<long long>(activations.numel()),
            static_cast<long long>(decoded.defined()?decoded.numel():0),image_batch);
        first=0;
        for(std::size_t begin=0;begin<prefill_groups;begin+=image_batch) {
            const auto count=std::min(image_batch,prefill_groups-begin);
            const auto banks=phase("tq.moe.weights",[&] {
                return std::array<std::vector<TqWeight>,3>{
                    gate->cache->get_rows(gate->source_id,unique.data()+begin,count,*gate->table,gate->out,source.get_device()),
                    up->cache->get_rows(up->source_id,unique.data()+begin,count,*up->table,up->out,source.get_device()),
                    down->cache->get_rows(down->source_id,unique.data()+begin,count,*down->table,down->out,source.get_device())};
            });
            for(std::size_t i=0;i<count;++i) {
                const auto index=begin+i;
                const TrellisPrefillMoeJob job{banks[0][i].view,banks[1][i].view,banks[2][i].view,
                    static_cast<int64_t>(first),static_cast<int>(slots[index].size())};
                std::memcpy(host_bytes+index*sizeof(job),&job,sizeof(job));first+=slots[index].size();
            }
            const auto offset=begin*sizeof(TrellisPrefillMoeJob);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(device_bytes+offset,host_bytes+offset,count*sizeof(TrellisPrefillMoeJob),
                cudaMemcpyHostToDevice,stream.stream()));
            for(std::size_t tile=begin;tile<begin+count;) {
                const bool gemm=tile<gemm_groups;
                const auto capacity=(slots[tile].size()+15)&~std::size_t(15);
                const auto per_expert=capacity*(gemm?2*max_width+4*output_width:4*max_width+2*hidden);
                const auto act_limit=gemm?activation_budget:std::min<std::uint64_t>(activation_budget,std::uint64_t(16)<<20);
                auto tile_count=std::min<std::size_t>({begin+count-tile,65535,
                    std::max<std::uint64_t>(1,(act_limit-std::min<std::uint64_t>(32,act_limit))/per_expert)});
                if(gemm) {
                    // Bound independent GEMMs while keeping full weight tiles
                    // when possible; larger matrices still use row tiling.
                    const auto full_tile=4*std::uint64_t(up->input)*hidden;
                    tile_count=std::min<std::size_t>({tile_count,gemm_groups-tile,16,
                        std::max<std::uint64_t>(1,decoded.nbytes()/full_tile)});
                }
                const auto* plan=reinterpret_cast<const TrellisPrefillMoeJob*>(device_bytes+tile*sizeof(TrellisPrefillMoeJob));
                const auto* device_slots=reinterpret_cast<const int64_t*>(device_bytes+job_bytes);
                if(gemm)phase("tq.moe.prefill_gemm",[&] {status(mfq_tq_prefill_moe_gemm(plan,device_slots,
                    source.data_ptr(),pairs.data_ptr(),static_cast<int>(tile_count),static_cast<int>(capacity),
                    static_cast<int>(up->input),static_cast<int>(hidden),static_cast<int>(routes),limit,
                    decoded.data_ptr(),decoded.nbytes(),activations.data_ptr(),activations.nbytes(),blas,stream.stream()));return 0;});
                else phase("tq.moe.prefill",[&] {status(mfq_tq_prefill_moe(plan,device_slots,
                    source.data_ptr(),pairs.data_ptr(),static_cast<int>(tile_count),static_cast<int>(capacity),
                    static_cast<int>(up->input),static_cast<int>(hidden),static_cast<int>(routes),limit,
                    activations.data_ptr(),activations.nbytes(),stream.stream()));return 0;});
                tile+=tile_count;
            }
        }
    }
    if(prefill_groups<long_groups) {
        // Long prefill gathers one expert's rows and uses the ordinary
        // multi-token/GEMM kernels. Only one compressed expert and bounded
        // decoded tiles need to be live, regardless of the prompt length.
        for(std::size_t index=prefill_groups;index<long_groups;++index) {
            const auto expert=unique[index];
            const auto g=gate->cache->get_rows(gate->source_id,&expert,1,*gate->table,gate->out,source.get_device());
            const auto u=up->cache->get_rows(up->source_id,&expert,1,*up->table,up->out,source.get_device());
            const auto d=down->cache->get_rows(down->source_id,&expert,1,*down->table,down->out,source.get_device());
            std::vector<std::int64_t> token_ids;token_ids.reserve(slots[index].size());
            for(const auto slot:slots[index])token_ids.push_back(slot/routes);
            const auto token_index=tb::from_blob(token_ids.data(),{static_cast<std::int64_t>(token_ids.size())},
                tb::TensorOptions().dtype(tb::kInt64)).clone().to(source.device(),false,false);
            const auto slot_index=tb::from_blob(slots[index].data(),{static_cast<std::int64_t>(slots[index].size())},
                tb::TensorOptions().dtype(tb::kInt64)).clone().to(source.device(),false,false);
            const auto gathered=source.index_select(0,token_index);
            tb::Tensor activated;
            if(limit>0) {
                const auto gv=g[0].forward(*pool,gathered),uv=u[0].forward(*pool,gathered);
                if(type==0)activated=clipped_silu_mul_cuda(gv,uv,limit);
                else activated=(tb::silu(tb::clamp_max(gv.to(tb::kFloat32),limit))*
                    tb::clamp(uv.to(tb::kFloat32),-limit,limit)).to(source.scalar_type()).contiguous();
            }else activated=TqProjectionGroup({g[0],u[0]}).swiglu(*pool,gathered);
            flat_pairs.index_copy_(0,slot_index,d[0].forward(*pool,std::move(activated)));
        }
    }
    if(long_groups<unique.size()) {
        auto activated=tb::empty({tokens,routes,hidden},source.options());
        // The live batch is bounded by current free VRAM, not a device or
        // model whitelist. FP16-sized images conservatively bound every
        // supported 1..8-bit compressed row plus metadata and scale data.
        const auto image_bound=[](const TqRoutedWeight& w) {
            return std::uint64_t(w.out)*(((std::uint64_t(w.input)+255)&~std::uint64_t(255))*2+16)+512;
        };
        const auto per_expert=image_bound(*gate)+image_bound(*up)+image_bound(*down);
        for(std::size_t begin=long_groups;begin<unique.size();) {
            const auto free=phase("tq.moe.capacity",[&] {
                std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));return available;
            });
            const auto count=std::min<std::size_t>(std::min<std::size_t>(65535,unique.size()-begin),
                std::max<std::uint64_t>(1,(free/2)/per_expert));
            const auto banks=phase("tq.moe.weights",[&] {
                return std::array<std::vector<TqWeight>,3>{
                    gate->cache->get_rows(gate->source_id,unique.data()+begin,count,*gate->table,gate->out,source.get_device()),
                    up->cache->get_rows(up->source_id,unique.data()+begin,count,*up->table,up->out,source.get_device()),
                    down->cache->get_rows(down->source_id,unique.data()+begin,count,*down->table,down->out,source.get_device())};
            });
            const auto& g=banks[0];const auto& u=banks[1];const auto& d=banks[2];
            if(tokens==1 && unique.size()==static_cast<std::size_t>(routes)) {
                // Preserve the tuned single-vector geometry. Each selected
                // expert gets one slot, with no multi-token dummy work.
                const auto tables=phase("tq.moe.views",[&] {
                    return std::array<tb::Tensor,3>{
                        gate->cache->projection_table(gate->source_id,gate->experts,unique.data()+begin,g,source.get_device()),
                        up->cache->projection_table(up->source_id,up->experts,unique.data()+begin,u,source.get_device()),
                        down->cache->projection_table(down->source_id,down->experts,unique.data()+begin,d,source.get_device())};
                });
                const auto& gp=tables[0];const auto& upp=tables[1];const auto& dp=tables[2];
                const auto route_ids=ids.to(source.device()).to(tb::kInt64).contiguous();
                const auto element=source.element_size();
                auto* h=static_cast<std::uint8_t*>(activated.data_ptr())+begin*hidden*element;
                auto* p=static_cast<std::uint8_t*>(pairs.data_ptr())+begin*width*element;
                phase("tq.moe.gu",[&] {status(mfq_tq_routed_swiglu(reinterpret_cast<const TrellisProjectionView*>(gp.data_ptr()),
                    reinterpret_cast<const TrellisProjectionView*>(upp.data_ptr()),route_ids.data_ptr<int64_t>()+begin,
                    source.data_ptr(),h,static_cast<int>(count),static_cast<int>(hidden),type,limit,mfq_current_cuda_stream()));return 0;});
                phase("tq.moe.down",[&] {status(mfq_tq_routed_down(reinterpret_cast<const TrellisProjectionView*>(dp.data_ptr()),
                    route_ids.data_ptr<int64_t>()+begin,h,p,static_cast<int>(count),static_cast<int>(width),
                    static_cast<int>(hidden),type,mfq_current_cuda_stream()));return 0;});
            }else {
                auto [gu_tiles,down_tiles,wide_tiles,wide_gu]=tq_indexed_jobs(g,u,d,slots,begin,routes,up->input,hidden,width,
                    type==0 && resources->indexed_mma());
                std::vector<TrellisIndexedProjectionJob> gu_jobs,down_jobs;
                std::vector<TrellisIndexedWideProjectionJob> wide_jobs;
                for(int tile=0;tile<4;++tile) {
                    gu_jobs.insert(gu_jobs.end(),gu_tiles[tile].begin(),gu_tiles[tile].end());
                    down_jobs.insert(down_jobs.end(),down_tiles[tile].begin(),down_tiles[tile].end());
                    wide_jobs.insert(wide_jobs.end(),wide_tiles[tile].begin(),wide_tiles[tile].end());
                }
                if(gu_jobs.size()>INT_MAX)throw std::overflow_error("too many TQ indexed jobs");
                const auto gp=upload_plan(gu_jobs),dp=upload_plan(down_jobs),wp=upload_plan(wide_jobs),gwp=upload_plan(wide_gu);
                const auto* gate_plan=gp.defined()?reinterpret_cast<const TrellisIndexedProjectionJob*>(gp.data_ptr()):nullptr;
                const auto* down_plan=dp.defined()?reinterpret_cast<const TrellisIndexedProjectionJob*>(dp.data_ptr()):nullptr;
                const auto* wide_plan=wp.defined()?reinterpret_cast<const TrellisIndexedWideProjectionJob*>(wp.data_ptr()):nullptr;
                std::size_t first=0;
                for(int tile=0;tile<4;++tile) {
                    const auto jobs=static_cast<int>(gu_tiles[tile].size());
                    if(!jobs)continue;
                    status(mfq_tq_indexed_swiglu(gate_plan+first,source.data_ptr(),activated.data_ptr(),jobs,
                        static_cast<int>(hidden),type,tile+1,limit,mfq_current_cuda_stream()));
                    first+=jobs;
                }
                if(!wide_gu.empty())status(mfq_tq_indexed_wide_swiglu(
                    reinterpret_cast<const TrellisIndexedWidePairJob*>(gwp.data_ptr()),source.data_ptr(),activated.data_ptr(),
                    static_cast<int>(wide_gu.size()),static_cast<int>(hidden),type,8,limit,mfq_current_cuda_stream()));
                first=0;
                for(int tile=0;tile<4;++tile) {
                    const auto jobs=static_cast<int>(down_tiles[tile].size());if(!jobs)continue;
                    status(mfq_tq_indexed(down_plan+first,activated.data_ptr(),pairs.data_ptr(),jobs,
                        static_cast<int>(width),type,tile+1,mfq_current_cuda_stream()));first+=jobs;
                }
                first=0;
                for(int tile=0;tile<4;++tile) {
                    const auto jobs=static_cast<int>(wide_tiles[tile].size());if(!jobs)continue;
                    status(mfq_tq_indexed_wide(wide_plan+first,activated.data_ptr(),pairs.data_ptr(),jobs,
                        static_cast<int>(width),type,tile+5,mfq_current_cuda_stream()));first+=jobs;
                }
            }
            begin+=count;
        }
    }
    phase("tq.moe.reduce",[&] {status(mfq_tq_weighted_reduce(pairs.data_ptr(),coefficients.data_ptr<float>(),result.data_ptr(),
        static_cast<int>(tokens),static_cast<int>(routes),static_cast<int>(width),type,mfq_current_cuda_stream()));return 0;});
    return result;
}

tb::Tensor TqMoeBanks::forward_parallel(CudaExecutionContext& execution,tb::Tensor input,
        tb::Tensor ids,tb::Tensor weights,float limit)const {
    const auto& parallel=moe_parallel_config(execution);
    if(!parallel.enabled())return forward(std::move(input),std::move(ids),std::move(weights),limit,&execution.profiler);
    if(!gate)throw std::invalid_argument("TQ expert parallelism requires a Gate bank");
    const auto slices=plan_moe_expert_parallel_slices(parallel,gate->experts,"TQ experts");
    std::vector<tb::Tensor> partials(slices.size());
    for(std::size_t launch=0;launch<slices.size();++launch) {
        const auto index=model_parallel_launch_index(execution.config,launch,slices.size());
        const auto& slice=slices[index];
        const MfqCudaGuard guard(slice.device);
        auto local_input=tensor_to_cuda_device(execution.model_parallel_collectives,input,slice.device);
        auto local_ids=tensor_to_cuda_device(execution.model_parallel_collectives,ids,slice.device);
        auto local_weights=tensor_to_cuda_device(execution.model_parallel_collectives,weights,slice.device);
        partials[index]=forward(std::move(local_input),std::move(local_ids),std::move(local_weights),
            limit,&execution.profiler,slice.begin,slice.end);
    }
    return reduce_model_parallel_outputs(execution,std::move(partials));
}
