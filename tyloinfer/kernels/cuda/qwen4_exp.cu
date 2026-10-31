#include "qwen4_exp.h"
#include "mfq_cuda_norm_ops.h"
#include "cpp_runtime/backends/cuda/kernels/kernel_dispatch_trace.h"

#include "selected_attention.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <atomic>
#include <cstdio>
#include <cub/block/block_radix_sort.cuh>

namespace mfq_qwen4_exp {
namespace tb = mfq_tensor_backend;

namespace {
template<class Value,class Position>
__global__ void live_index_pool_kernel(const Value* index,const Position* positions,
        const int64_t* cache,Value* pooled,int64_t* blocks,Position* block_positions,
        int tokens,int width,int pool,int64_t maximum,int64_t position_entries,
        int64_t row_stride,int64_t column_stride) {
    const int token=blockIdx.x;
    const int64_t block=cache[token]/pool,start=block*pool;
    if(threadIdx.x==0)blocks[token]=block;
    for(int64_t i=threadIdx.x;i<position_entries/tokens;i+=blockDim.x) {
        const int64_t at=i*tokens+token;
        block_positions[at]=Position(int64_t(positions[at])-cache[token]+start);
    }
    for(int column=threadIdx.x;column<width;column+=blockDim.x) {
        float sum=0;
        for(int i=0;i<pool;++i) {
            const auto row=min(max(int64_t(0),start+i),maximum-1);
            sum=__fadd_rn(sum,float(index[row*row_stride+column*column_stride]));
        }
        pooled[int64_t(token)*width+column]=Value(__fdiv_rn(sum,float(pool)));
    }
}
}

std::vector<Tensor> live_index_pool(const Tensor& index,const Tensor& positions,
        const Tensor& cache_positions,int64_t pool,int64_t maximum) {
    TYLOINFER_RUNTIME_CHECK(index.is_cuda() && index.dim()==3 && index.size(0)==1 &&
        (index.scalar_type()==tb::kFloat16 || index.scalar_type()==tb::kFloat32) &&
        positions.device()==index.device() && cache_positions.device()==index.device() &&
        positions.dim()>=1 && positions.dim()<=3 &&
        (positions.scalar_type()==tb::kInt32 || positions.scalar_type()==tb::kInt64) &&
        cache_positions.dim()==1 && cache_positions.numel()>0 && cache_positions.is_contiguous() &&
        cache_positions.scalar_type()==tb::kInt64 && positions.size(-1)==cache_positions.numel() &&
        pool>0 && pool<=INT_MAX && maximum>0 && maximum<=index.size(1) &&
        index.size(2)>0 && index.size(2)<=INT_MAX && cache_positions.numel()<=INT_MAX,
        "live QSA pool has incompatible index/position geometry");
    const MfqCudaGuard guard(index.device());
    auto pos=positions.contiguous();
    auto pooled=tb::empty({1,cache_positions.numel(),index.size(2)},index.options());
    auto blocks=tb::empty_like(cache_positions),block_positions=tb::empty_like(pos);
    const auto launch=[&](auto* source,auto* output,auto* p,auto* bp) {
        live_index_pool_kernel<<<unsigned(cache_positions.numel()),128,0,mfq_current_cuda_stream()>>>(
            source,p,cache_positions.data_ptr<int64_t>(),output,blocks.data_ptr<int64_t>(),bp,
            int(cache_positions.numel()),int(index.size(2)),int(pool),maximum,pos.numel(),index.stride(1),index.stride(2));
    };
    const auto typed=[&](auto* source,auto* output) {
        if(pos.scalar_type()==tb::kInt64)launch(source,output,pos.data_ptr<int64_t>(),block_positions.data_ptr<int64_t>());
        else launch(source,output,pos.data_ptr<int32_t>(),block_positions.data_ptr<int32_t>());
    };
    if(index.scalar_type()==tb::kFloat16)typed(index.data_ptr<mfq_half>(),pooled.data_ptr<mfq_half>());
    else typed(index.data_ptr<float>(),pooled.data_ptr<float>());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return {pooled,blocks,block_positions};
}

namespace {
template<class Position>
__global__ void fill_decode_positions_kernel(int64_t* cache,Position* positions,
    int64_t tokens,int64_t entries,int64_t start,int64_t delta) {
    for(int64_t i=threadIdx.x;i<entries;i+=blockDim.x) {
        positions[i]=static_cast<Position>(start+i%tokens+delta);
        if(i<tokens)cache[i]=start+i;
    }
}
}

void fill_decode_positions(const Tensor& cache,const Tensor& positions,int64_t start,int64_t delta) {
    TYLOINFER_RUNTIME_CHECK(cache.is_cuda() && positions.is_cuda() && cache.is_contiguous() &&
        positions.is_contiguous() && cache.scalar_type()==tb::kInt64 && cache.numel()>0 &&
        positions.numel()%cache.numel()==0 && positions.numel()>=cache.numel(),
        "decode position buffers have incompatible geometry");
    const auto launch=[&](auto* output) {
        fill_decode_positions_kernel<<<1,32,0,mfq_current_cuda_stream()>>>(cache.data_ptr<int64_t>(),output,
            cache.numel(),positions.numel(),start,delta);
    };
    if(positions.scalar_type()==tb::kInt32)launch(positions.data_ptr<int32_t>());
    else {
        TYLOINFER_RUNTIME_CHECK(positions.scalar_type()==tb::kInt64,"decode positions must use int32 or int64");
        launch(positions.data_ptr<int64_t>());
    }
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
template<class Value, bool Injection>
__global__ void gated_residual_activation_kernel(const Value* input,Value* output,
    int64_t count,float streams) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const Value low=static_cast<Value>(__fdiv_rn(static_cast<float>(input[i]),streams));
        const Value gate=static_cast<Value>(1.0f/(1.0f+expf(-static_cast<float>(low))));
        const float multiplier=Injection?2.0f:static_cast<float>(low);
        output[i]=static_cast<Value>(__fmul_rn(multiplier,static_cast<float>(gate)));
    }
}

template<bool Injection>
Tensor gated_residual_activation(const Tensor& input,int64_t streams) {
    mfq_selected_attention::values({&input});
    TYLOINFER_RUNTIME_CHECK(streams>0,"gated-residual stream count must be positive");
    if(input.is_contiguous() && (input.scalar_type()==tb::kFloat16 || input.scalar_type()==tb::kFloat32)) {
        MfqCudaGuard guard(input.device());
        auto output=tb::empty_like(input);const auto count=input.numel();
        if(!count)return output;
        const int blocks=int((count+255)/256);
        if(input.scalar_type()==tb::kFloat16)
            gated_residual_activation_kernel<mfq_half,Injection><<<blocks,256,0,mfq_current_cuda_stream()>>>(
                input.data_ptr<mfq_half>(),output.data_ptr<mfq_half>(),count,float(streams));
        else
            gated_residual_activation_kernel<float,Injection><<<blocks,256,0,mfq_current_cuda_stream()>>>(
                input.data_ptr<float>(),output.data_ptr<float>(),count,float(streams));
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
        return output;
    }
    auto low=input/streams;
    if constexpr(Injection)return 2.0*tb::sigmoid(low);
    else return low*tb::sigmoid(low);
}

template<class Gate, class Value>
__global__ void gated_residual_mix_kernel(const Gate* projection,const Value* normalized,
    std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,mfq_half>* output,
    int64_t count,int hidden,int streams) {
    using Output=std::conditional_t<std::is_same_v<Gate,float> || std::is_same_v<Value,float>,float,mfq_half>;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t base=(i/hidden)*hidden*streams+i%hidden;
        float sum=0;
        for(int stream=0;stream<streams;++stream) {
            const int64_t at=base+int64_t(stream)*hidden;
            const Gate gate=static_cast<Gate>(1.0f/(1.0f+expf(-static_cast<float>(projection[at]))));
            const Output product=static_cast<Output>(__fmul_rn(static_cast<float>(gate),static_cast<float>(normalized[at])));
            sum=__fadd_rn(sum,static_cast<float>(product));
        }
        output[i]=static_cast<Output>(sum/streams);
    }
}
template<class Branch,class Residual,class Injection>
using ResidualUpdate=std::conditional_t<std::is_same_v<Branch,float> ||
    std::is_same_v<Injection,float>,float,mfq_half>;
template<class Branch,class Residual,class Injection>
using ResidualOutput=std::conditional_t<std::is_same_v<Residual,float> ||
    std::is_same_v<ResidualUpdate<Branch,Residual,Injection>,float>,float,mfq_half>;

template<class Branch,class Residual,class Injection>
__global__ void gated_residual_post_kernel(const Branch* branch,const Residual* residual,
    const Injection* injection,ResidualOutput<Branch,Residual,Injection>* output,int64_t count,int hidden,int streams) {
    using Update=ResidualUpdate<Branch,Residual,Injection>;
    using Output=ResidualOutput<Branch,Residual,Injection>;
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t sample=i/(int64_t(hidden)*streams);
        const int stream=int((i/hidden)%streams);
        const Update update=static_cast<Update>(__fmul_rn(static_cast<float>(branch[sample*hidden+i%hidden]),
            static_cast<float>(injection[sample*streams+stream])));
        output[i]=static_cast<Output>(__fadd_rn(static_cast<float>(residual[i]),static_cast<float>(update)));
    }
}

template<class Branch,class Residual,class Injection>
__global__ void gated_residual_post_norm_kernel(const Branch* branch,const Residual* residual,
    const Injection* injection,const float* weight,
    ResidualOutput<Branch,Residual,Injection>* updated,
    ResidualOutput<Branch,Residual,Injection>* normalized,int hidden,int streams,float eps) {
    using Update=ResidualUpdate<Branch,Residual,Injection>;
    using Output=ResidualOutput<Branch,Residual,Injection>;
    const int stream=blockIdx.x%streams;
    const int64_t sample=blockIdx.x/streams,base=int64_t(blockIdx.x)*hidden;
    const float gate=static_cast<float>(injection[sample*streams+stream]);
    float sum=0;
    for(int column=threadIdx.x;column<hidden;column+=blockDim.x) {
        const Update product=static_cast<Update>(__fmul_rn(
            static_cast<float>(branch[sample*hidden+column]),gate));
        const Output value=static_cast<Output>(__fadd_rn(
            static_cast<float>(residual[base+column]),static_cast<float>(product)));
        updated[base+column]=value;
        const float f=static_cast<float>(value);
        sum=__fadd_rn(sum,__fmul_rn(f,f));
    }
    __shared__ float partial[256];partial[threadIdx.x]=sum;__syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2) {
        if(threadIdx.x<stride)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    const float mean=__fdiv_rn(partial[0],static_cast<float>(hidden));
    const float inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(mean,eps)));
    for(int column=threadIdx.x;column<hidden;column+=blockDim.x) {
        const float value=__fmul_rn(static_cast<float>(updated[base+column]),inverse);
        normalized[base+column]=static_cast<Output>(__fmul_rn(
            value,__fadd_rn(weight[int64_t(stream)*hidden+column],1.0f)));
    }
}

template<class Branch,class Residual,class Injection>
void launch_residual_post_norm(const Tensor& branch,const Tensor& residual,const Tensor& injection,
    const Tensor& norm,Tensor& updated,Tensor& normalized,int hidden,int streams,double eps) {
    using Output=ResidualOutput<Branch,Residual,Injection>;
    const int threads=hidden<=32?32:hidden<=64?64:hidden<=128?128:256;
    gated_residual_post_norm_kernel<Branch,Residual,Injection><<<
        updated.numel()/hidden,threads,0,mfq_current_cuda_stream()>>>(
        branch.data_ptr<Branch>(),residual.data_ptr<Residual>(),injection.data_ptr<Injection>(),
        norm.data_ptr<float>(),updated.data_ptr<Output>(),normalized.data_ptr<Output>(),hidden,streams,float(eps));
}
}

Tensor gated_residual_bottleneck(const Tensor& input,int64_t streams) {
    return gated_residual_activation<false>(input,streams);
}
Tensor gated_residual_injection(const Tensor& input,int64_t streams) {
    return gated_residual_activation<true>(input,streams);
}

Tensor gated_residual_mix(const Tensor& projection,const Tensor& normalized,int64_t streams) {
    mfq_selected_attention::values({&projection,&normalized});
    TYLOINFER_RUNTIME_CHECK(projection.sizes()==normalized.sizes() && streams>0 && normalized.dim()>0 &&
        normalized.size(-1)>0 && normalized.size(-1)%streams==0,"gated-residual mix shape disagrees");
    auto shape=normalized.sizes().vec();const int hidden=int(shape.back()/streams);shape.back()=hidden;
    const auto half_or_float=[](const Tensor& x){return x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32;};
    if(projection.is_contiguous() && normalized.is_contiguous() && half_or_float(projection) && half_or_float(normalized)) {
        MfqCudaGuard guard(normalized.device());
        const auto dtype=projection.scalar_type()==tb::kFloat32 || normalized.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
        auto output=tb::empty(shape,normalized.options().dtype(dtype));const auto count=output.numel();
        if(count==0)return output;
        const int blocks=int((count+255)/256);
        if(dtype==tb::kFloat16)
            gated_residual_mix_kernel<<<blocks,256,0,mfq_current_cuda_stream()>>>(projection.data_ptr<mfq_half>(),normalized.data_ptr<mfq_half>(),output.data_ptr<mfq_half>(),count,hidden,int(streams));
        else if(projection.scalar_type()==tb::kFloat16)
            gated_residual_mix_kernel<<<blocks,256,0,mfq_current_cuda_stream()>>>(projection.data_ptr<mfq_half>(),normalized.data_ptr<float>(),output.data_ptr<float>(),count,hidden,int(streams));
        else if(normalized.scalar_type()==tb::kFloat16)
            gated_residual_mix_kernel<<<blocks,256,0,mfq_current_cuda_stream()>>>(projection.data_ptr<float>(),normalized.data_ptr<mfq_half>(),output.data_ptr<float>(),count,hidden,int(streams));
        else
            gated_residual_mix_kernel<<<blocks,256,0,mfq_current_cuda_stream()>>>(projection.data_ptr<float>(),normalized.data_ptr<float>(),output.data_ptr<float>(),count,hidden,int(streams));
        return output;
    }
    auto grouped=shape;grouped.back()=streams;grouped.push_back(hidden);
    return (tb::sigmoid(projection).reshape(grouped)*normalized.reshape(grouped)).mean(-2);
}
using tb::Tensor;

Tensor grouped_rms_norm(const Tensor& value, const Tensor& weight,
                        int64_t group, double eps) {
    mfq_selected_attention::values({&value, &weight});
    TYLOINFER_RUNTIME_CHECK(value.dim() >= 1 && group > 0 && value.size(-1) > 0 &&
        value.size(-1) % group == 0 && weight.dim() == 1 && weight.size(0) == value.size(-1),
        "Qwen4 grouped RMSNorm dimensions disagree");
    MfqCudaGuard guard(value.device());
    if (value.is_contiguous() && (value.scalar_type() == tb::kFloat16 || value.scalar_type() == tb::kFloat32))
        return grouped_rms_norm_cuda(value, weight.to(tb::kFloat32).contiguous(), group, eps, 1.0);
    auto shape = value.sizes().vec();
    shape.back() /= group;
    shape.push_back(group);
    auto source = value.to(tb::kFloat32).reshape(shape);
    auto normalized = source * tb::rsqrt((source * source).mean(-1, true) + eps);
    return (normalized.reshape(value.sizes()) * (1.0 + weight.to(tb::kFloat32)))
        .to(value.scalar_type());
}

std::vector<Tensor> gated_residual_pre(
    const Tensor& input, const Tensor& norm, const Tensor& down, const Tensor& up,
    const std::optional<Tensor>& inject, int64_t hidden, int64_t streams, double eps) {
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
        : Tensor{};
    return {mixed, input, injection};
}

bool try_gated_residual_post_out(const Tensor& branch,const Tensor& residual,
        const Tensor& injection,const Tensor& output,int64_t streams) {
    const auto supported=[](const Tensor& t){return t.defined() && t.is_cuda() && t.is_contiguous() &&
        (t.scalar_type()==tb::kFloat16 || t.scalar_type()==tb::kFloat32);};
    if(streams<1 || streams>INT_MAX || !supported(branch) || !supported(residual) || !supported(injection) ||
            !supported(output) || branch.device()!=residual.device() || injection.device()!=residual.device() ||
            output.device()!=residual.device() || branch.dim()<1 || branch.size(-1)<1 || branch.size(-1)>INT_MAX ||
            branch.dim()!=residual.dim() || branch.dim()!=injection.dim() || output.sizes()!=residual.sizes())return false;
    auto shape=branch.sizes().vec();shape.back()*=streams;
    if(shape!=residual.sizes().vec())return false;
    shape.back()=streams;if(shape!=injection.sizes().vec())return false;
    const int mode=(branch.scalar_type()==tb::kFloat32?4:0) |
        (residual.scalar_type()==tb::kFloat32?2:0) | (injection.scalar_type()==tb::kFloat32?1:0);
    if(output.scalar_type()!=(mode?tb::kFloat32:tb::kFloat16))return false;
    if(!output.numel())return true;
    const auto overlap=[](const Tensor& a,const Tensor& b) {
        const auto x=reinterpret_cast<std::uintptr_t>(a.data_ptr()),y=reinterpret_cast<std::uintptr_t>(b.data_ptr());
        return x<=y?y-x<a.nbytes():x-y<b.nbytes();
    };
    if(overlap(output,branch) || overlap(output,injection) ||
            (overlap(output,residual) && output.data_ptr()!=residual.data_ptr()))return false;
    const MfqCudaGuard guard(residual.device());
    const auto count=output.numel();const int blocks=int((count+255)/256),hidden=int(branch.size(-1));
    const auto stream=mfq_current_cuda_stream();
    const auto launch=[&](auto b,auto r,auto i) {
        using Branch=decltype(b);using Residual=decltype(r);using Injection=decltype(i);
        using Output=ResidualOutput<Branch,Residual,Injection>;
        gated_residual_post_kernel<Branch,Residual,Injection><<<blocks,256,0,stream>>>(branch.data_ptr<Branch>(),
            residual.data_ptr<Residual>(),injection.data_ptr<Injection>(),output.data_ptr<Output>(),count,hidden,int(streams));
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gr_residual_post",
            reinterpret_cast<const void*>(gated_residual_post_kernel<Branch,Residual,Injection>),
            "M="+std::to_string(branch.numel()/hidden)+" hidden="+std::to_string(hidden)+" streams="+std::to_string(streams)+
            " type_flags="+std::to_string(mode)+" inplace="+std::to_string(output.data_ptr()==residual.data_ptr()));
    };
    const auto residual_type=[&](auto b,auto i) {
        if(residual.scalar_type()==tb::kFloat16)launch(b,mfq_half{},i);else launch(b,float{},i);
    };
    const auto injection_type=[&](auto b) {
        if(injection.scalar_type()==tb::kFloat16)residual_type(b,mfq_half{});else residual_type(b,float{});
    };
    if(branch.scalar_type()==tb::kFloat16)injection_type(mfq_half{});else injection_type(float{});
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    for(const auto& t:{branch,residual,injection,output})t.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1' && branch.numel()/hidden>32) {
        static std::atomic<unsigned> reported{0};
        const bool inplace=output.data_ptr()==residual.data_ptr();const unsigned bit=inplace?1:2;
        if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "native_gr_prefill_post actual_match=1 M=%lld hidden=%d streams=%lld type_flags=%d inplace=%d\n",
            static_cast<long long>(branch.numel()/hidden),hidden,static_cast<long long>(streams),mode,int(inplace));
    }
    return true;
}

Tensor gated_residual_post(const Tensor& branch, const Tensor& residual,
                           const Tensor& injection, int64_t streams) {
    mfq_selected_attention::values({&branch, &residual, &injection});
    TYLOINFER_RUNTIME_CHECK(branch.dim() >= 1 && streams > 0 && branch.dim() == residual.dim() &&
        branch.dim() == injection.dim(), "Qwen4 residual ranks disagree");
    auto shape = branch.sizes().vec();
    shape.back() *= streams;
    TYLOINFER_RUNTIME_CHECK(residual.sizes().vec() == shape, "Qwen4 residual width disagrees");
    shape.back() = streams;
    TYLOINFER_RUNTIME_CHECK(injection.sizes().vec() == shape, "Qwen4 injection shape disagrees");
    MfqCudaGuard guard(branch.device());
    const auto value=[](const Tensor& t){return t.scalar_type()==tb::kFloat16 || t.scalar_type()==tb::kFloat32;};
    if(branch.is_contiguous() && residual.is_contiguous() && injection.is_contiguous() &&
            value(branch) && value(residual) && value(injection)) {
        const auto dtype=branch.scalar_type()==tb::kFloat32 || residual.scalar_type()==tb::kFloat32 ||
            injection.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
        auto output=tb::empty(residual.sizes(),residual.options().dtype(dtype));
        if(try_gated_residual_post_out(branch,residual,injection,output,streams))return output;
    }
    return residual + (branch.unsqueeze(-2) * injection.unsqueeze(-1)).reshape(residual.sizes());
}

std::vector<Tensor> gated_residual_post_norm(const Tensor& branch,const Tensor& residual,
    const Tensor& injection,const Tensor& norm,int64_t streams,double eps) {
    mfq_selected_attention::values({&branch,&residual,&injection,&norm});
    TYLOINFER_RUNTIME_CHECK(branch.dim()>=1 && streams>0 && branch.size(-1)>0 &&
        branch.dim()==residual.dim() && branch.dim()==injection.dim(),
        "Qwen4 residual/norm ranks disagree");
    auto shape=branch.sizes().vec();shape.back()*=streams;
    TYLOINFER_RUNTIME_CHECK(residual.sizes().vec()==shape && norm.dim()==1 && norm.size(0)==shape.back(),
        "Qwen4 residual/norm width disagrees");
    shape.back()=streams;
    TYLOINFER_RUNTIME_CHECK(injection.sizes().vec()==shape,"Qwen4 residual/norm injection shape disagrees");
    const auto supported=[](const Tensor& x){return x.is_contiguous() &&
        (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32);};
    if(!supported(branch) || !supported(residual) || !supported(injection)) {
        auto updated=gated_residual_post(branch,residual,injection,streams);
        return {updated,grouped_rms_norm(updated,norm,branch.size(-1),eps)};
    }
    MfqCudaGuard guard(branch.device());
    const auto dtype=branch.scalar_type()==tb::kFloat32 || residual.scalar_type()==tb::kFloat32 ||
        injection.scalar_type()==tb::kFloat32?tb::kFloat32:tb::kFloat16;
    auto updated=tb::empty(residual.sizes(),residual.options().dtype(dtype));
    auto normalized=tb::empty_like(updated);
    if(updated.numel()) {
        auto weight=norm.to(tb::kFloat32).contiguous();
        const int hidden=int(branch.size(-1)),count=int(streams);
        const int mode=(branch.scalar_type()==tb::kFloat32?4:0) |
            (residual.scalar_type()==tb::kFloat32?2:0) | (injection.scalar_type()==tb::kFloat32?1:0);
        switch(mode) {
        case 0:launch_residual_post_norm<mfq_half,mfq_half,mfq_half>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        case 1:launch_residual_post_norm<mfq_half,mfq_half,float>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        case 2:launch_residual_post_norm<mfq_half,float,mfq_half>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        case 3:launch_residual_post_norm<mfq_half,float,float>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        case 4:launch_residual_post_norm<float,mfq_half,mfq_half>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        case 5:launch_residual_post_norm<float,mfq_half,float>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        case 6:launch_residual_post_norm<float,float,mfq_half>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        case 7:launch_residual_post_norm<float,float,float>(branch,residual,injection,weight,updated,normalized,hidden,count,eps);break;
        }
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    return {updated,normalized};
}

Tensor block_scores(const Tensor& query, const Tensor& pooled) {
    mfq_selected_attention::values({&query, &pooled});
    MfqCudaGuard guard(query.device());
    const auto* option=std::getenv("TYLOINFER_QSA_PREFILL_SCORE_GEMM");
    if(query.size(1)>=32 && option && option[0]=='1') {
        // Each index head owns a token matrix. This replaces thousands of
        // four-row GEMMs with a few large GEMMs and retains head reduction.
        auto q=query.to(tb::kFloat32).permute({0,2,1,3}).contiguous();
        auto keys=pooled.to(tb::kFloat32).transpose(-1,-2).unsqueeze(1);
        auto products=tb::matmul(q,keys).permute({0,2,1,3}).clamp_min(0.0);
        return products.sum(-2)/std::sqrt(double(query.size(-1)));
    }
    return mfq_selected_attention::dots(query, pooled).sum(-2) /
        std::sqrt(double(query.size(-1)));
}

namespace {
template<class Query,class Key>
__global__ void live_block_scores_kernel(const Query* query,const Key* keys,const int64_t* positions,
        float* output,int pools,int heads,int width,int pool,int64_t qs1,int64_t qs2,int64_t qs3,
        int64_t ks1,int64_t ks2) {
    const int lane=threadIdx.x&31,row=int(blockIdx.x)*4+threadIdx.x/32,token=int(blockIdx.y);
    if(row>=pools)return;
    if(int64_t(row+1)*pool>positions[token]+1) {
        if(lane==0)output[int64_t(token)*pools+row]=-1e30f;
        return;
    }
    float score=0;
    for(int head=0;head<heads;++head) {
        float dot=0;
        for(int d=lane;d<width;d+=32)
            dot=fmaf(float(query[int64_t(token)*qs1+int64_t(head)*qs2+int64_t(d)*qs3]),
                float(keys[int64_t(row)*ks1+int64_t(d)*ks2]),dot);
        for(int delta=16;delta;delta/=2)dot+=__shfl_down_sync(0xffffffff,dot,delta);
        score+=fmaxf(0.f,dot);
    }
    if(lane==0)output[int64_t(token)*pools+row]=score/sqrtf(float(width));
}
}

Tensor live_block_scores(const Tensor& query,const Tensor& pooled,const Tensor& positions,int64_t pool) {
    mfq_selected_attention::values({&query,&pooled});
    TYLOINFER_RUNTIME_CHECK(query.dim()==4 && query.size(0)==1 && pooled.dim()==3 && pooled.size(0)==1 &&
        query.size(3)==pooled.size(2) && query.size(2)>0 && query.size(3)>0 && pool>0 &&
        positions.is_cuda() && positions.is_contiguous() && positions.scalar_type()==tb::kInt64 &&
        positions.numel()==query.size(1),"invalid live QSA score geometry");
    const bool supported=(query.scalar_type()==tb::kFloat16 || query.scalar_type()==tb::kFloat32) &&
        (pooled.scalar_type()==tb::kFloat16 || pooled.scalar_type()==tb::kFloat32);
    const auto* option=std::getenv("TYLOINFER_QSA_LIVE_SCORES");
    if(!supported || (option && option[0]=='0')) {
        auto scores=block_scores(query,pooled);
        auto ends=(tb::arange(pooled.size(1),positions.options())*pool+pool-1).reshape({1,1,-1});
        return tb::where(ends<=positions.reshape({1,-1,1}),scores,tb::full_like(scores,-1e30));
    }
    MfqCudaGuard guard(query.device());
    auto out=tb::empty({1,query.size(1),pooled.size(1)},query.options().dtype(tb::kFloat32));
    if(!out.numel())return out;
    const dim3 grid(unsigned((pooled.size(1)+3)/4),unsigned(query.size(1)));
    const auto launch=[&](auto* q,auto* k) {
        live_block_scores_kernel<<<grid,128,0,mfq_current_cuda_stream()>>>(q,k,positions.data_ptr<int64_t>(),
            out.data_ptr<float>(),int(pooled.size(1)),int(query.size(2)),int(query.size(3)),int(pool),
            query.stride(1),query.stride(2),query.stride(3),pooled.stride(1),pooled.stride(2));
    };
    if(query.scalar_type()==tb::kFloat16) {
        if(pooled.scalar_type()==tb::kFloat16)launch(query.data_ptr<mfq_half>(),pooled.data_ptr<mfq_half>());
        else launch(query.data_ptr<mfq_half>(),pooled.data_ptr<float>());
    } else {
        if(pooled.scalar_type()==tb::kFloat16)launch(query.data_ptr<float>(),pooled.data_ptr<mfq_half>());
        else launch(query.data_ptr<float>(),pooled.data_ptr<float>());
    }
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

namespace {
template<int Items>
__device__ void live_topk_sort(const float* source,const int64_t* ids,float* output,int64_t* out_ids,
        int row,int columns,int begin,int live,int pools,int count,int64_t out,unsigned char* temporary,bool emit_tokens) {
    using Sort=cub::BlockRadixSort<float,256,Items,int64_t>;
    float keys[Items];int64_t indices[Items];
#pragma unroll
    for(int i=0;i<Items;++i) {
        const int column=begin+threadIdx.x*Items+i;
        const auto at=int64_t(row)*columns+column;
        keys[i]=column<live?source[at]:-1e30f;
        indices[i]=column<live?(ids?ids[at]:column):pools;
    }
    Sort(*reinterpret_cast<typename Sort::TempStorage*>(temporary)).SortDescending(keys,indices);
    // CUB no longer needs its storage once every thread has its sorted
    // register values. Reuse it for the final token expansion instead of
    // writing and rereading an otherwise unused global candidate list.
    if(emit_tokens)__syncthreads();
    auto* selected_ids=emit_tokens?reinterpret_cast<int64_t*>(temporary):out_ids+out;
#pragma unroll
    for(int i=0;i<Items;++i) {
        const int rank=threadIdx.x*Items+i;
        if(rank<count){if(!emit_tokens)output[out+rank]=keys[i];selected_ids[rank]=indices[i];}
    }
}
__global__ void live_topk_blocks_kernel(const float* source,const int64_t* ids,
        float* output,int64_t* out_ids,const int64_t* positions,int columns,
        int pools,int pool,int count,int stage,bool compact,int32_t* selected,int budget) {
    constexpr int tile=4096;
    extern __shared__ __align__(16) unsigned char temporary[];
    const int row=blockIdx.y,begin=int(blockIdx.x)*tile,tiles=int((int64_t(columns)+tile-1)/tile);
    int live=int(min(int64_t(pools),max(int64_t(0),(positions[row]+1)/pool)));
    for(int s=0;s<stage;++s)live=live/tile*count+min(count,live%tile);
    const int available=min(tile,max(0,live-begin));
    const int64_t out=(int64_t(row)*tiles+blockIdx.x)*count;
    auto* selected_ids=selected?reinterpret_cast<int64_t*>(temporary):out_ids+out;
    // Entirely unused tiles and already reduced candidate lists need no sort.
    if(available<=count) {
        for(int i=threadIdx.x;i<count;i+=256) {
            const auto at=int64_t(row)*columns+begin+i;
            if(!selected)output[out+i]=i<available?source[at]:-1e30f;
            selected_ids[i]=i<available?(ids?ids[at]:begin+i):pools;
        }
    } else {
    // The graph reserves the maximum context, while each replay has a much
    // shorter live prefix. Keep the reduction tiles and output layout fixed,
    // but sort only the power-of-two range containing this tile's live rows.
    // CUB's stable ordering is unchanged because input column order is kept.
    if(compact && available<=256)
        live_topk_sort<1>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else if(compact && available<=512)
        live_topk_sort<2>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else if(compact && available<=1024)
        live_topk_sort<4>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else if(compact && available<=2048)
        live_topk_sort<8>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    else live_topk_sort<16>(source,ids,output,out_ids,row,columns,begin,live,pools,count,out,temporary,selected!=nullptr);
    }
    if(selected) {
        // Only the final reduction has one CTA per query. Its IDs are now
        // complete; publish them to this CTA before expanding whole pools
        // and the incomplete causal tail into the existing token order.
        __syncthreads();
        const int64_t position=positions[row];
        const int width=budget+pool-1;
        for(int slot=threadIdx.x;slot<width;slot+=blockDim.x) {
            int64_t token=-1;
            if(position+1<=budget) {
                if(slot<=position)token=slot;
            } else if(int64_t(slot)<int64_t(count)*pool) {
                const int64_t block=selected_ids[slot/pool];
                if(block*pool+pool-1<=position)token=block*pool+slot%pool;
            } else if(slot>=budget) {
                const int offset=slot-budget;
                const int64_t tail=(position+1)%pool;
                if(offset<tail)token=position+1-tail+offset;
            }
            selected[int64_t(row)*width+slot]=int32_t(token);
        }
    }
}
}

static Tensor live_topk_impl(const Tensor& scores,const Tensor& positions,int64_t pool,int64_t count,int64_t budget) {
    TYLOINFER_RUNTIME_CHECK(!budget || (pool>0 && pool<=INT_MAX && budget>0 &&
        budget<=INT_MAX-pool && count<=budget/pool),"invalid live QSA token selection geometry");
    const auto* option=std::getenv("TYLOINFER_QSA_LIVE_TOPK");
    if((option && option[0]=='0') || count>1024) {
        auto ids=std::get<1>(tb::topk(scores,count,-1,true,false));
        return budget?mfq_selected_attention::block_indices_to_tokens(ids,positions,pool,budget):ids;
    }
    TYLOINFER_RUNTIME_CHECK(scores.is_cuda() && scores.is_contiguous() && scores.dim()==3 && scores.size(0)==1 &&
        scores.scalar_type()==tb::kFloat32 && positions.is_cuda() && positions.is_contiguous() &&
        positions.scalar_type()==tb::kInt64 && positions.numel()==scores.size(1) && pool>0 &&
        count>0 && count<=scores.size(2) && scores.size(1)<=INT_MAX && scores.size(2)<=INT_MAX,
        "invalid live QSA top-k geometry");
    MfqCudaGuard guard(scores.device());
    const int rows=int(scores.size(1)),pools=int(scores.size(2));
    const auto* compact_option=std::getenv("TYLOINFER_QSA_TOPK_LIVE_TILE");
    const bool compact=!compact_option || compact_option[0]!='0';
    int columns=pools,stage=0;
    auto keys=scores;Tensor ids;
    Tensor selected;
    if(budget)selected=tb::empty({1,rows,budget+pool-1},scores.options().dtype(tb::kInt32));
    constexpr auto max_bytes=sizeof(cub::BlockRadixSort<float,256,16,int64_t>::TempStorage);
    if constexpr(max_bytes>49152)TYLOINFER_CUDA_CHECK(cudaFuncSetAttribute(live_topk_blocks_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,int(max_bytes)));
    do {
        const int tiles=int((int64_t(columns)+4095)/4096);
        // The maximum candidate count is known without reading live graph
        // positions. Smaller contexts do not need the largest CUB workspace.
        const auto bytes=compact && columns<=1024?sizeof(cub::BlockRadixSort<float,256,4,int64_t>::TempStorage):
            compact && columns<=2048?sizeof(cub::BlockRadixSort<float,256,8,int64_t>::TempStorage):
            sizeof(cub::BlockRadixSort<float,256,16,int64_t>::TempStorage);
        auto next=tb::empty({1,rows,tiles*count},scores.options());
        auto next_ids=tb::empty(next.sizes(),scores.options().dtype(tb::kInt64));
        live_topk_blocks_kernel<<<dim3(tiles,rows),256,bytes,mfq_current_cuda_stream()>>>(
            keys.data_ptr<float>(),ids.defined()?ids.data_ptr<int64_t>():nullptr,
            next.data_ptr<float>(),next_ids.data_ptr<int64_t>(),positions.data_ptr<int64_t>(),
            columns,pools,int(pool),int(count),stage++,compact,
            tiles==1 && selected.defined()?selected.data_ptr<int32_t>():nullptr,int(budget));
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(
            mfq_current_cuda_stream(),"qsa_live_topk",reinterpret_cast<const void*>(live_topk_blocks_kernel),
            "M="+std::to_string(rows)+" candidate_columns="+std::to_string(columns)+
            " count="+std::to_string(count)+" selection_fused="+std::to_string(tiles==1 && selected.defined())+
            " minimum_sort_items="+std::to_string(compact?1:16)+" shared_bytes="+std::to_string(bytes));
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();keys=std::move(next);ids=std::move(next_ids);columns=tiles*int(count);
    } while(columns>count);
    return budget?selected:ids;
}

Tensor live_topk_blocks(const Tensor& scores,const Tensor& positions,int64_t pool,int64_t count) {
    return live_topk_impl(scores,positions,pool,count,0);
}

Tensor live_topk_selected_tokens(const Tensor& scores,const Tensor& positions,int64_t pool,int64_t count,int64_t budget) {
    return live_topk_impl(scores,positions,pool,count,budget);
}

std::vector<Tensor> ple_dilated_conv_silu(const Tensor& input,
    const Tensor& weight, const std::optional<Tensor>& state, int64_t dilation) {
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
    auto previous = state ? state->to(input.scalar_type()) : tb::zeros(state_shape, input.options());
    auto combined = tb::cat({previous, input}, 1);
    auto output = tb::zeros(input.sizes(), input.options().dtype(tb::kFloat32));
    for (int64_t tap = 0; tap < w.size(1); ++tap) {
        output = output + combined.narrow(1, tap * dilation, input.size(1)).to(tb::kFloat32) *
            w.select(1, tap).to(tb::kFloat32).unsqueeze(0).unsqueeze(0);
    }
    output = output * tb::sigmoid(output);
    auto next = combined.narrow(1, length ? combined.size(1) - length : 0, length).contiguous();
    return {output.to(input.scalar_type()), next};
}

Tensor dense_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, int64_t offset,mfq::cuda::HotKvView hot) {
    return mfq_selected_attention::dense(
        q, k, v, offset, 1.0 / std::sqrt(double(q.size(3))),hot);
}

Tensor sparse_gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& indices,mfq::cuda::HotKvView hot) {
    return mfq_selected_attention::sparse(
        q, k, v, indices, 1.0 / std::sqrt(double(q.size(3))), false,hot);
}

Tensor attention_gate(const Tensor& attended, const Tensor& gate, bool half_output) {
    return mfq_selected_attention::gated(attended, gate, half_output);
}

Tensor sparse_gqa_attention_gate(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& indices, const Tensor& gate, bool half_output,mfq::cuda::HotKvView hot) {
    return mfq_selected_attention::sparse_gated(q, k, v, indices, gate,
        1.0 / std::sqrt(double(q.size(3))), half_output,hot);
}

Tensor qsa_selected_tokens(const Tensor& blocks, const Tensor& positions,
    int64_t pool, int64_t budget) {
    return mfq_selected_attention::block_indices_to_tokens(blocks, positions, pool, budget);
}

Tensor causal_gqa_attention_gate(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& positions, const Tensor& gate, int64_t columns, bool half_output,mfq::cuda::HotKvView hot) {
    return mfq_selected_attention::causal_gated(q, k, v, positions, gate, columns,
        1.0 / std::sqrt(double(q.size(3))), half_output,true,true,hot);
}

} // namespace mfq_qwen4_exp
