#include "../../kernels/gguf_window.h"
#include "gguf_weight.h"
#include "gguf_prefill_fusion.h"
#include "gguf_rotation.h"
#include "gguf_row_correction.h"
#include "quant_linear.h"
#include "mfq_weight_memory.h"
#include "tyloinfer/runtime.h"
#include "mfq_cuda_context.h"
#include "../../kernels/gguf.h"
#include "../../kernels/gguf_m1_residual.h"
#include "../../kernels/gguf_compact.h"
#include "../../kernels/gguf_mmq.h"
#include "../../kernels/gguf_quad_storage.h"
#include "../../kernels/gguf_quad_matvec.h"
#include "../../kernels/gguf_quad_small_m.h"
#include "../../kernels/gguf_quad_decode.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "gguf_dense_schedule.h"
#endif
#include "ggml-quants.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <atomic>
#include <cstdio>
#include <map>
#include <tuple>

#include <mutex>
namespace tb = mfq_tensor_backend;

namespace {
std::int64_t prefill_weight_rows(std::int64_t,std::uint64_t,std::int64_t,int,bool);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
// Only row exponents are retained. Weak ownership prevents either model
// retention or reuse after an allocator gives another weight the same address.
struct PrefillRowScale {
    std::weak_ptr<void> owner;const void* data;ggml_type type;int device;
    std::int64_t rows,columns,quad_rows,quad_begin;
    tb::Tensor exponents;mfq::cuda::Event ready;
};
tb::Tensor prefill_cached_exponents(const GgufWeight& w) {
    static std::mutex mutex;static std::vector<std::unique_ptr<PrefillRowScale>> cache;
    const auto owner=w.packed.weak_storage_owner();const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    std::lock_guard lock(mutex);
    for(auto i=cache.begin();i!=cache.end();) {
        if((*i)->owner.expired()){i=cache.erase(i);continue;}
        auto& entry=**i;++i;
        if(!entry.owner.owner_before(owner) && !owner.owner_before(entry.owner) &&
                entry.data==w.packed.data_ptr() && entry.type==w.type && entry.rows==w.out &&
                entry.columns==w.neuron_len && entry.quad_rows==w.quad_storage_rows && entry.quad_begin==w.quad_row_begin) {
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),entry.ready.get(),0));
            mfq_cuda_record_stream(entry.exponents,stream);return entry.exponents;
        }
    }
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone)return {};
    auto entry=std::make_unique<PrefillRowScale>();entry->owner=owner;entry->data=w.packed.data_ptr();entry->type=w.type;
    entry->device=w.packed.get_device();entry->rows=w.out;entry->columns=w.neuron_len;
    entry->quad_rows=w.quad_storage_rows;entry->quad_begin=w.quad_row_begin;
    entry->exponents=tb::empty({w.out},w.packed.options().dtype(tb::kInt32));
    mfq_cuda_record_stream(entry->exponents,stream);
    if(w.quad_storage_rows) {
        const auto rows=prefill_weight_rows(w.out,w.row_bytes,128,w.packed.get_device(),false);
        auto image=tb::empty({std::int64_t(rows*w.row_bytes)},w.packed.options());
        mfq_cuda_record_stream(image,stream);mfq_cuda_record_stream(w.packed,stream);
        for(std::int64_t begin=0;begin<w.out;begin+=rows) {
            const auto count=std::min(rows,w.out-begin);
            mfq::cuda::gguf::unpack_quad_storage(w.type,w.packed.data_ptr(),w.quad_storage_rows,w.neuron_len,
                w.quad_row_begin+begin,count,nullptr,count,image.data_ptr(),w.row_bytes,stream.stream());
            mfq::cuda::gguf::prefill_row_exponents(w.type,image.data_ptr(),w.row_bytes,count,w.neuron_len,
                entry->exponents.data_ptr<std::int32_t>()+begin,stream.stream());
        }
    } else {
        mfq_cuda_record_stream(w.packed,stream);
        mfq::cuda::gguf::prefill_row_exponents(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
            entry->exponents.data_ptr<std::int32_t>(),stream.stream());
    }
    entry->ready.record(stream.stream());auto result=entry->exponents;cache.push_back(std::move(entry));return result;
}
#else
tb::Tensor prefill_cached_exponents(const GgufWeight&) { return {}; }
#endif
std::int64_t prefill_weight_rows(std::int64_t rows,std::uint64_t bytes_per_row,
        std::int64_t alignment,int device,bool double_buffer=false) {
    const auto context=mfq::cuda::default_context(device);
    using Key=std::tuple<int,std::int64_t,std::uint64_t,std::int64_t,bool>;
    static thread_local std::map<Key,std::int64_t> prepared;
    const Key key{device,rows,bytes_per_row,alignment,double_buffer};
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone) {
        if(const auto found=prepared.find(key);found!=prepared.end())return found->second;
        return std::min(rows,alignment);
    }
    const auto physical=context->available_memory(),reusable=context->reusable_workspace_bytes();
    const auto available=physical>SIZE_MAX-reusable?SIZE_MAX:physical+reusable;
    const auto loan=context->workspace_available();
    const auto budget=available>SIZE_MAX-loan?SIZE_MAX:available+loan;
    // Leave room for the following activation and the BLAS library's scratch.
    // A complete matrix is preferred whenever it fits this live device budget;
    // the old fixed 16 MiB tile needlessly fragmented large prompt GEMMs.
    auto count=std::min<std::uint64_t>(rows,(budget/2)/bytes_per_row);
    if(count<std::uint64_t(rows) && double_buffer)count/=2;
    const auto selected=count>=std::uint64_t(rows)?rows:count>=std::uint64_t(alignment)?
        std::int64_t(count/alignment*alignment):std::max<std::int64_t>(1,std::int64_t(count));
    prepared[key]=selected;
    return selected;
}
}

struct GgufPrefillInputScope::Impl {
    static thread_local Impl* active;Impl* previous=nullptr;
    tb::Tensor source,half,exponents;mfq::cuda::Event ready;
};
thread_local GgufPrefillInputScope::Impl* GgufPrefillInputScope::Impl::active=nullptr;
GgufPrefillInputScope::GgufPrefillInputScope()=default;
std::unique_ptr<GgufPrefillInputScope> GgufPrefillInputScope::prepare_rms(
        const tb::Tensor& input,const tb::Tensor& weight,double eps,double offset,tb::Tensor& normalized) {
    normalized={};const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if(!input.is_cuda() || !input.is_contiguous() || input.dim()<1 || input.size(-1)<1 ||
            input.numel()/input.size(-1)<512 ||
            (input.scalar_type()!=tb::kFloat32 && input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kBFloat16) ||
            weight.device()!=input.device() || weight.scalar_type()!=tb::kFloat32 || !weight.is_contiguous() ||
            weight.numel()!=input.size(-1) || !std::isfinite(eps) || eps<=0 || !std::isfinite(offset) ||
            (setting && setting[0]!='1' && setting[0]!='a'))return {};
    const MfqCudaGuard guard(input.get_device());const auto rows=input.numel()/input.size(-1),width=input.size(-1);
    if(!mfq::cuda::gguf::grouped_f16_available(input.get_device()) ||
            !mfq::cuda::gguf::rms_prefill_available(rows,width,input.get_device()))return {};
    auto scope=std::unique_ptr<GgufPrefillInputScope>(new GgufPrefillInputScope);
    auto plan=std::make_unique<Impl>();
    normalized=tb::empty(input.sizes(),input.options().dtype(tb::kFloat32));plan->source=normalized;
    plan->half=tb::empty(input.sizes(),input.options().dtype(tb::kFloat16));
    plan->exponents=tb::empty({rows},input.options().dtype(tb::kInt32));
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    for(const auto& tensor:{input,weight,normalized,plan->half,plan->exponents})mfq_cuda_record_stream(tensor,stream);
    const auto type=input.scalar_type()==tb::kFloat32?GGML_TYPE_F32:input.scalar_type()==tb::kFloat16?GGML_TYPE_F16:GGML_TYPE_BF16;
    mfq::cuda::gguf::pack_rms_prefill(input.data_ptr(),type,weight.data_ptr<float>(),rows,width,float(eps),float(offset),
        normalized.data_ptr<float>(),plan->half.data_ptr(),plan->exponents.data_ptr<int32_t>(),stream.stream());
    plan->ready.record(stream.stream());plan->previous=Impl::active;Impl::active=plan.get();scope->impl_=std::move(plan);
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_gguf_prefill_rms_input actual_match=1 M=%lld K=%lld normalized_fp32=1 reuse_scaled_fp16=1\n",
            static_cast<long long>(rows),static_cast<long long>(width));
    }
    return scope;
}
GgufPrefillInputScope::GgufPrefillInputScope(const tb::Tensor& input) {
    const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if(!input.is_cuda() || input.scalar_type()!=tb::kFloat32 || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)<1 || input.numel()/input.size(-1)<512 ||
            (setting && setting[0]!='1' && setting[0]!='a'))return;
    const MfqCudaGuard guard(input.get_device());
    if(!mfq::cuda::gguf::grouped_f16_available(input.get_device()))return;
    if(Impl::active && input.data_ptr()==Impl::active->source.data_ptr() &&
            input.device()==Impl::active->source.device() && input.sizes()==Impl::active->source.sizes())return;
    auto plan=std::make_unique<Impl>();plan->source=input;
    plan->half=tb::empty(input.sizes(),input.options().dtype(tb::kFloat16));
    plan->exponents=tb::empty({input.numel()/input.size(-1)},input.options().dtype(tb::kInt32));
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    for(const auto& tensor:{input,plan->half,plan->exponents})mfq_cuda_record_stream(tensor,stream);
    mfq::cuda::gguf::pack_scaled_f16_unit(input.data_ptr<float>(),plan->half.data_ptr(),plan->exponents.data_ptr<int32_t>(),
        input.numel()/input.size(-1),input.size(-1),stream.stream());
    plan->ready.record(stream.stream());plan->previous=Impl::active;Impl::active=plan.get();impl_=std::move(plan);
}
GgufPrefillInputScope::~GgufPrefillInputScope(){if(impl_)Impl::active=impl_->previous;}
bool GgufPrefillInputScope::lookup(const tb::Tensor& input,tb::Tensor& half,tb::Tensor& exponents) {
    const auto* plan=Impl::active;
    if(!plan || !input.defined() || input.dim()<1 || !input.is_contiguous() ||
            input.scalar_type()!=tb::kFloat32 || input.device()!=plan->source.device() || input.data_ptr()!=plan->source.data_ptr() || input.numel()!=plan->source.numel() ||
            input.size(-1)!=plan->source.size(-1))return false;
    const MfqCudaGuard guard(input.get_device());
    const auto stream=mfq_get_current_cuda_stream(input.get_device());
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),plan->ready.get(),0));
    half=plan->half;exponents=plan->exponents;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "native_gguf_prefill_shared_input actual_match=1 M=%lld K=%lld reuse_scaled_fp16=1\n",
            static_cast<long long>(input.numel()/input.size(-1)),static_cast<long long>(input.size(-1)));
    }
    mfq_cuda_record_stream(half,stream);mfq_cuda_record_stream(exponents,stream);return true;
}
GgufPrefillProduct gguf_prefill_product(const GgufWeight& w,const tb::Tensor& input) {
    const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if((setting && setting[0]!='1' && setting[0]!='a') || w.prefers_original_input() ||
            !w.packed.is_cuda() || !input.is_cuda() || input.device()!=w.packed.device() ||
            input.scalar_type()!=tb::kFloat32 || !input.is_contiguous() || input.dim()<1 ||
            input.size(-1)!=w.neuron_len || w.neuron_len<1 || input.numel()/w.neuron_len<512)return {};
    const MfqCudaGuard guard(w.packed.get_device());
    if(!mfq::cuda::gguf::grouped_f16_available(w.packed.get_device()))return {};
    const auto tokens=input.numel()/w.neuron_len;
    if(prefill_weight_rows(w.out,uint64_t(w.neuron_len)*2+uint64_t(tokens)*2,32,w.packed.get_device())<w.out)return {};
    auto columns=prefill_cached_exponents(w);if(!columns.defined())return {};
    tb::Tensor packed,rows;
    const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    if(!GgufPrefillInputScope::lookup(input,packed,rows)) {
        packed=tb::empty({tokens,w.neuron_len},input.options().dtype(tb::kFloat16));
        rows=tb::empty({tokens},input.options().dtype(tb::kInt32));
        mfq_cuda_record_stream(input,stream);mfq_cuda_record_stream(packed,stream);mfq_cuda_record_stream(rows,stream);
        mfq::cuda::gguf::pack_scaled_f16_unit(input.data_ptr<float>(),packed.data_ptr(),rows.data_ptr<int32_t>(),
            tokens,w.neuron_len,stream.stream());
    }
    auto image=tb::empty({w.out,w.neuron_len},input.options().dtype(tb::kFloat16));
    auto product=tb::empty({tokens,w.out},input.options().dtype(tb::kFloat16));
    for(const auto& tensor:{w.packed,packed,rows,columns,image,product})mfq_cuda_record_stream(tensor,stream);
    if(w.quad_storage_rows)mfq::cuda::gguf::decode_quad_rows_scaled(w.type,w.packed.data_ptr(),w.quad_storage_rows,
        w.neuron_len,w.quad_row_begin,w.out,columns.data_ptr<int32_t>(),image.data_ptr(),stream.stream());
    else mfq::cuda::gguf::dequantize_rows_scaled(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
        columns.data_ptr<int32_t>(),image.data_ptr(),stream.stream());
    auto& blas=mfq::cuda::default_context(w.packed.get_device())->blas(stream.stream());
    mfq::cuda::gguf::prefill_gemm_f16_accum(packed.data_ptr(),image.data_ptr(),nullptr,product.data_ptr(),true,
        nullptr,nullptr,tokens,w.out,w.neuron_len,w.out,blas.get(),stream.stream());
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,
            "native_gguf_prefill_scaled_product actual_match=1 M=%lld N=%lld K=%lld restore_at_consumer=1\n",
            static_cast<long long>(tokens),static_cast<long long>(w.out),static_cast<long long>(w.neuron_len));
    }
    return {product,rows,columns};
}

static GgufPrefillProduct rms_gated_product_impl(const GgufWeight& w,const tb::Tensor& attended,
        const tb::Tensor& gate,const tb::Tensor& norm,double eps,bool silu,const GgufPrefillProduct* scaled_gate) {
    const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if((setting && setting[0]!='1' && setting[0]!='a') || w.prefers_original_input() || !w.packed.is_cuda() ||
            !attended.is_cuda() || attended.device()!=w.packed.device() || !attended.is_contiguous() ||
            attended.scalar_type()!=tb::kFloat32 || attended.dim()!=4 ||
            norm.device()!=attended.device() || norm.scalar_type()!=tb::kFloat32 || !norm.is_contiguous() ||
            gate.device()!=attended.device() || gate.scalar_type()!=(scaled_gate?tb::kFloat16:tb::kFloat32) || gate.dim()!=3)return {};
    if(!std::isfinite(eps) || eps<=0)return {};
    const MfqCudaGuard guard(w.packed.get_device());
    const auto batch=attended.size(0),heads=attended.size(1),tokens=attended.size(2),width=attended.size(3);
    const auto m=batch*tokens;
    if(m<512 || heads*width!=w.neuron_len || norm.numel()!=width || gate.size(0)!=batch || gate.size(1)!=tokens ||
            gate.size(2)!=heads*width || gate.stride(2)!=1 || gate.stride(1)<heads*width ||
            gate.stride(0)<tokens*gate.stride(1) || !mfq::cuda::gguf::grouped_f16_available(w.packed.get_device()) ||
            !mfq::cuda::gguf::rms_gated_prefill_available(w.neuron_len,w.packed.get_device()))return {};
    if(scaled_gate) {
        for(const auto* e:{&scaled_gate->row_exponents,&scaled_gate->column_exponents})
            if(!e->defined() || !e->is_contiguous() || e->device()!=attended.device() || e->scalar_type()!=tb::kInt32)return {};
        if(scaled_gate->row_exponents.numel()!=m || scaled_gate->column_exponents.numel()!=w.neuron_len)return {};
    }
    const auto row_bytes=uint64_t(w.neuron_len)*2+uint64_t(m)*2;
    if(prefill_weight_rows(w.out,row_bytes,32,w.packed.get_device(),false)<w.out)return {};
    auto exponents=prefill_cached_exponents(w);if(!exponents.defined())return {};
    auto image=tb::empty({w.out,w.neuron_len},attended.options().dtype(tb::kFloat16));
    auto packed=tb::empty({m,w.neuron_len},attended.options().dtype(tb::kFloat16));
    auto rows=tb::empty({m},attended.options().dtype(tb::kInt32));
    auto product=tb::empty({m,w.out},attended.options().dtype(tb::kFloat16));
    const auto stream=mfq_get_current_cuda_stream(w.packed.get_device());
    for(const auto& tensor:{w.packed,attended,gate,norm,exponents,image,packed,rows,product})mfq_cuda_record_stream(tensor,stream);
    if(scaled_gate) {
        mfq_cuda_record_stream(scaled_gate->row_exponents,stream);mfq_cuda_record_stream(scaled_gate->column_exponents,stream);
        mfq::cuda::gguf::pack_rms_scaled_gate_prefill(attended.data_ptr<float>(),gate.data_ptr(),norm.data_ptr<float>(),
            scaled_gate->row_exponents.data_ptr<int32_t>(),scaled_gate->column_exponents.data_ptr<int32_t>(),
            batch,heads,tokens,width,gate.stride(0),gate.stride(1),float(eps),silu,packed.data_ptr(),rows.data_ptr<int32_t>(),stream.stream());
    } else mfq::cuda::gguf::pack_rms_gated_prefill(attended.data_ptr<float>(),gate.data_ptr<float>(),norm.data_ptr<float>(),
            batch,heads,tokens,width,gate.stride(0),gate.stride(1),float(eps),silu,packed.data_ptr(),rows.data_ptr<int32_t>(),stream.stream());
    if(w.quad_storage_rows)mfq::cuda::gguf::decode_quad_rows_scaled(w.type,w.packed.data_ptr(),w.quad_storage_rows,
        w.neuron_len,w.quad_row_begin,w.out,exponents.data_ptr<int32_t>(),image.data_ptr(),stream.stream());
    else mfq::cuda::gguf::dequantize_rows_scaled(w.type,w.packed.data_ptr(),w.row_bytes,w.out,w.neuron_len,
        exponents.data_ptr<int32_t>(),image.data_ptr(),stream.stream());
    const auto context=mfq::cuda::default_context(w.packed.get_device());auto& blas=context->blas(stream.stream());
    mfq::cuda::gguf::prefill_gemm_f16_accum(packed.data_ptr(),image.data_ptr(),nullptr,product.data_ptr(),true,
        nullptr,nullptr,m,w.out,w.neuron_len,w.out,blas.get(),stream.stream());
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> printed{false};if(!printed.exchange(true))std::fprintf(stderr,
            "native_gguf_prefill_rms_gate_pack actual_match=1 M=%lld N=%lld K=%lld fused_rms=1 fused_layout=1 fused_gate=%s fused_scale=1\n",
            static_cast<long long>(m),static_cast<long long>(w.out),static_cast<long long>(w.neuron_len),silu?"silu":"sigmoid");
    }
    return {product,rows,exponents};
}

GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& w,const tb::Tensor& attended,
        const tb::Tensor& gate,const tb::Tensor& norm,double eps,bool silu) {
    return rms_gated_product_impl(w,attended,gate,norm,eps,silu,nullptr);
}
GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& w,const tb::Tensor& attended,
        const GgufPrefillProduct& gate,const tb::Tensor& norm,double eps,bool silu) {
    if(!gate.defined() || gate.values.dim()!=2 || !gate.values.is_contiguous() || attended.dim()!=4 ||
        gate.values.size(0)!=attended.size(0)*attended.size(2) ||
        gate.values.size(1)!=attended.size(1)*attended.size(3))return {};
    auto result=rms_gated_product_impl(w,attended,
        gate.values.reshape({attended.size(0),attended.size(2),gate.values.size(1)}),norm,eps,silu,&gate);
    if(result.defined())if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_gguf_prefill_scaled_gate actual_match=1 restore_at_consumer=1\n");
    }
    return result;
}
tb::Tensor gguf_prefill_materialize(const GgufPrefillProduct& product) {
    TYLOINFER_RUNTIME_CHECK(product.defined() && product.values.is_cuda() && product.values.is_contiguous() &&
        product.values.dim()==2 && product.values.scalar_type()==tb::kFloat16,
        "scaled prefill materialization requires a contiguous FP16 product");
    const auto rows=product.values.size(0),columns=product.values.size(1);
    for(const auto* e:{&product.row_exponents,&product.column_exponents})
        TYLOINFER_RUNTIME_CHECK(e->defined() && e->is_contiguous() && e->device()==product.values.device() && e->scalar_type()==tb::kInt32,
            "scaled prefill materialization exponents disagree");
    TYLOINFER_RUNTIME_CHECK(product.row_exponents.numel()==rows && product.column_exponents.numel()==columns,
        "scaled prefill materialization exponent sizes disagree");
    const MfqCudaGuard guard(product.values.get_device());
    auto output=tb::empty({rows,columns},product.values.options().dtype(tb::kFloat32));
    const auto stream=mfq_get_current_cuda_stream(output.get_device());
    for(const auto& t:{product.values,product.row_exponents,product.column_exponents,output})mfq_cuda_record_stream(t,stream);
    mfq::cuda::gguf::restore_scaled_f16(product.values.data_ptr(),output.data_ptr(),false,
        product.row_exponents.data_ptr<int32_t>(),product.column_exponents.data_ptr<int32_t>(),rows,columns,columns,stream.stream());
    return output;
}

tb::Tensor gguf_rms_gated_prefill(const GgufWeight& w,const tb::Tensor& attended,
        const tb::Tensor& gate,const tb::Tensor& norm,double eps,bool silu) {
    const auto product=gguf_rms_gated_product(w,attended,gate,norm,eps,silu);
    if(!product.defined())return {};
    auto output=tb::empty(product.values.sizes(),attended.options());
    const auto stream=mfq_get_current_cuda_stream(output.get_device());
    for(const auto& tensor:{product.values,product.row_exponents,product.column_exponents,output})mfq_cuda_record_stream(tensor,stream);
    mfq::cuda::gguf::restore_scaled_f16(product.values.data_ptr(),output.data_ptr(),false,
        product.row_exponents.data_ptr<int32_t>(),product.column_exponents.data_ptr<int32_t>(),
        output.size(0),output.size(1),output.size(1),stream.stream());
    return output.reshape({attended.size(0),attended.size(2),w.out});
}
std::vector<tb::Tensor> gguf_prefill_acc_rms(const GgufPrefillProduct& product,const tb::Tensor& residual,
        const tb::Tensor& norm,double eps,double offset) {
    TYLOINFER_RUNTIME_CHECK(product.defined() && residual.is_cuda() && residual.is_contiguous() &&
        residual.scalar_type()==tb::kFloat16 && residual.dim()==2 && product.values.is_contiguous() &&
        product.values.device()==residual.device() && product.values.sizes()==residual.sizes() &&
        product.values.scalar_type()==tb::kFloat16 && norm.is_contiguous() && norm.device()==residual.device() &&
        norm.scalar_type()==tb::kFloat32 && norm.numel()==residual.size(1),"scaled prefill residual/RMS operands disagree");
    for(const auto* exponents:{&product.row_exponents,&product.column_exponents})
        TYLOINFER_RUNTIME_CHECK(exponents->defined() && exponents->is_cuda() && exponents->is_contiguous() &&
            exponents->device()==residual.device() && exponents->scalar_type()==tb::kInt32,
            "scaled prefill residual/RMS exponents disagree");
    TYLOINFER_RUNTIME_CHECK(product.row_exponents.numel()==residual.size(0) &&
        product.column_exponents.numel()==residual.size(1) && std::isfinite(eps) && eps>0,
        "scaled prefill residual/RMS exponent geometry or epsilon disagrees");
    const MfqCudaGuard guard(residual.get_device());
    auto sum=tb::empty_like(residual),normalized=tb::empty_like(residual);
    const auto stream=mfq_get_current_cuda_stream(residual.get_device());
    for(const auto& tensor:{residual,norm,product.values,product.row_exponents,product.column_exponents,sum,normalized})
        mfq_cuda_record_stream(tensor,stream);
    mfq::cuda::gguf::acc_rms_scaled_f16(product.values.data_ptr(),residual.data_ptr(),norm.data_ptr<float>(),
        product.row_exponents.data_ptr<int32_t>(),product.column_exponents.data_ptr<int32_t>(),
        residual.size(0),residual.size(1),float(eps),float(offset),sum.data_ptr(),normalized.data_ptr(),stream.stream());
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,"native_gguf_prefill_acc_rms actual_match=1 fused_restore=1 fused_residual=1 fused_rms=1\n");
    }
    return {sum,normalized};
}

void GgufWeight::repack_inplace() {
    // Called during loading, before publishing aliased immutable views.
    namespace gguf=mfq::cuda::gguf;
    if(quad_storage_rows || !packed.is_cuda() || out<4 ||
            !gguf::supports_quad_storage(type) || base_prefers_original_input())return;
    const MfqCudaGuard guard(packed.get_device());
    const auto stream=mfq_get_current_cuda_stream(packed.get_device());
    cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&status));
    TYLOINFER_RUNTIME_CHECK(status==cudaStreamCaptureStatusNone,
        "GGUF storage preparation must precede graph capture");
    TYLOINFER_RUNTIME_CHECK(packed.is_contiguous() && packed.scalar_type()==tb::kUInt8 &&
        std::uint64_t(packed.numel())==gguf::quad_storage_bytes(type,out,neuron_len),
        "GGUF compute storage must preserve the compressed byte count");
    // One temporary matrix, released before preparing another weight. The
    // original allocation/address survives; graph and aliased weight views
    // never retain a second permanent image.
    auto source=tb::empty_like(packed);
    mfq_cuda_record_stream(source,stream);mfq_cuda_record_stream(packed,stream);
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(source.data_ptr(),packed.data_ptr(),packed.nbytes(),
        cudaMemcpyDeviceToDevice,stream.stream()));
    gguf::pack_quad_storage(type,source.data_ptr(),row_bytes,out,neuron_len,packed.data_ptr(),stream.stream());
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    quad_storage_rows=out;quad_row_begin=0;
}

tb::Tensor GgufWeight::canonical_packed() const {
    if(!quad_storage_rows)return packed;
    const MfqCudaGuard guard(packed.get_device());
    const auto stream=mfq_get_current_cuda_stream(packed.get_device());
    auto result=tb::empty({std::int64_t(std::uint64_t(out)*row_bytes)},packed.options());
    mfq_cuda_record_stream(packed,stream);mfq_cuda_record_stream(result,stream);
    mfq::cuda::gguf::unpack_quad_storage(type,packed.data_ptr(),quad_storage_rows,neuron_len,
        quad_row_begin,out,nullptr,out,result.data_ptr(),row_bytes,stream.stream());
    return result;
}

namespace {
mfq::cuda::gguf::QuadProjection quad_projection(const GgufWeight& w,void* output) {
    return {{w.packed.data_ptr(),w.row_bytes,w.out,output,w.type},
        w.quad_storage_rows,w.quad_row_begin,bool(w.quad_storage_rows)};
}
}

tb::Tensor GgufWeight::forward_gated_residual(
        const tb::Tensor& gate,const tb::Tensor& up,const tb::Tensor& residual) const {
    TYLOINFER_RUNTIME_CHECK(!prefers_original_input() &&
        mfq::cuda::gguf::supports_m1_residual(type) && gate.is_cuda() &&
        gate.device()==packed.device() && up.device()==gate.device() &&
        residual.device()==gate.device() && gate.scalar_type()==tb::kFloat16 &&
        up.scalar_type()==tb::kFloat16 && residual.scalar_type()==tb::kFloat16 &&
        gate.is_contiguous() && up.is_contiguous() && residual.is_contiguous() &&
        gate.sizes()==up.sizes() && up.numel()==neuron_len && residual.numel()==out,
        "fused GGUF SwiGLU/residual requires one contiguous FP16 row");
    const MfqCudaGuard guard(gate.get_device());
    const auto stream=mfq_get_current_cuda_stream(gate.get_device());
    auto activation=tb::empty({int64_t(mfq::cuda::gguf::q8_row_bytes(neuron_len))},
        gate.options().dtype(tb::kUInt8));
    auto output=tb::empty_like(residual);
    for(const auto& tensor:{gate,up,residual,packed,activation,output})mfq_cuda_record_stream(tensor,stream);
    const bool aligned=mfq::cuda::gguf::prefers_m1_aligned(type);
    const auto quantize=aligned?mfq::cuda::gguf::quantize_gated_input_m1_aligned:
        mfq::cuda::gguf::quantize_gated_input;
    quantize(up.data_ptr(),gate.data_ptr(),true,1,neuron_len,true,activation.data_ptr(),stream.stream());
    if(quad_projection_ready())mfq::cuda::gguf::matvec_quad_m1(quad_projection(*this,output.data_ptr()),
        neuron_len,activation.data_ptr(),aligned,true,residual.data_ptr(),stream.stream());
    else {
        const auto image=canonical_packed();mfq_cuda_record_stream(image,stream);
        mfq::cuda::gguf::matvec_m1_residual(type,image.data_ptr(),row_bytes,out,neuron_len,
            activation.data_ptr(),aligned,residual.data_ptr(),output.data_ptr(),stream.stream());
    }
    return output;
}

bool same_gguf_input_basis(const GgufWeight& a, const GgufWeight& b) {
    return (!a.rotation && !b.rotation) || (a.rotation && b.rotation && a.rotation->equivalent(*b.rotation));
}
namespace {
class JoinedProjectionCorrection final:public GgufProjectionCorrection {
    std::vector<std::shared_ptr<const GgufProjectionCorrection>> corrections_;
    std::vector<std::int64_t> rows_;
public:
    explicit JoinedProjectionCorrection(const std::vector<GgufWeight*>& weights){
        for(const auto* w:weights){corrections_.push_back(w->projection_correction);rows_.push_back(w->out);}
    }
    void prepare(const Tensor& input)const override{
        for(const auto& correction:corrections_)if(correction)correction->prepare(input);
    }
    Tensor apply(const Tensor& input,Tensor output)const override{
        std::int64_t offset=0;
        for(std::size_t i=0;i<rows_.size();++i){
            if(corrections_[i]){auto view=output.narrow(output.dim()-1,offset,rows_[i]);
                view.copy_(corrections_[i]->apply(input,view));}offset+=rows_[i];}
        return output;
    }
};
}

GgufWeight combine_gguf_rows(const std::vector<GgufWeight*>& weights) {
    if(weights.empty() || !weights.front())throw std::invalid_argument("GGUF row combination requires weights");
    const auto& first=*weights.front();
    if(first.neuron_len<1 || !first.row_bytes || mfq::gguf::row_bytes(first.type,first.neuron_len)!=first.row_bytes)
        throw std::invalid_argument("invalid GGUF combined row encoding");
    std::int64_t rows=0;
    std::vector<tb::Tensor> images;images.reserve(weights.size());
    for(const auto* weight:weights) {
        if(!weight || weight->correction || weight->type!=first.type || weight->neuron_len!=first.neuron_len || weight->row_bytes!=first.row_bytes ||
                !same_gguf_input_basis(first,*weight) ||
                weight->out<1 || !weight->packed.defined() || weight->packed.device()!=first.packed.device() ||
                weight->packed.scalar_type()!=tb::kUInt8 || !weight->packed.is_contiguous() ||
                weight->out>INT64_MAX-rows || weight->row_bytes>std::uint64_t(INT64_MAX)/weight->out ||
                weight->packed.numel()!=std::int64_t(weight->row_bytes*(weight->quad_storage_rows?weight->quad_storage_rows:weight->out)))
            throw std::invalid_argument("incompatible GGUF row images");
        rows+=weight->out;images.push_back(weight->canonical_packed().reshape({-1}));
    }
    if(first.row_bytes>std::uint64_t(INT64_MAX)/rows)throw std::overflow_error("GGUF joined row image is too large");
    const mfq::cuda::WeightAllocationScope weight_memory;
    const auto image=tb::cat(images,0);
    GgufWeight joined{image,first.type,rows,first.neuron_len,first.row_bytes};
    joined.rotation=first.rotation;
    joined.repack_inplace();
    if(std::any_of(weights.begin(),weights.end(),[](const GgufWeight* w){return bool(w->projection_correction);}))
        joined.projection_correction=std::make_shared<JoinedProjectionCorrection>(weights);
    std::int64_t offset=0;
    for(auto* weight:weights) {
        const auto bytes=std::int64_t(weight->out*weight->row_bytes);
        weight->packed=joined.quad_storage_rows?image:image.narrow(0,offset,bytes);
        weight->quad_storage_rows=joined.quad_storage_rows;
        weight->quad_row_begin=joined.quad_storage_rows?offset/std::int64_t(first.row_bytes):0;
        offset+=bytes;
    }
    return joined;
}

void decode_gguf_cpu(ggml_type type, const void* packed, float* output, std::int64_t columns) {
    if (type == GGML_TYPE_Q8_K)
        dequantize_row_q8_K(static_cast<const block_q8_K*>(packed), output, columns);
    else {
        const auto* traits = ggml_get_type_traits(type);
        if (!traits->to_float) throw std::invalid_argument("GGUF format has no CPU decoder");
        traits->to_float(packed, output, columns);
    }
}

GgufWeight load_gguf_weight(const mfq::ModelSource& source, const std::string& name,
    int device, bool cuda, std::int64_t row_begin, std::int64_t row_end,
    std::int64_t column_begin, std::int64_t column_end) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 2)
        throw std::invalid_argument("GGUF projection requires a matrix: " + name);
    GgufWeight result;
    result.type = mfq::gguf::quant_type(record->dtype);
    const auto rows = record->shape->at(0), columns = record->shape->at(1);
    const auto source_stride = mfq::gguf::row_bytes(result.type, columns);
    if (rows <= 0 || source_stride > std::uint64_t(INT64_MAX) / rows ||
            record->nbytes != source_stride * rows)
        throw std::invalid_argument("invalid GGUF matrix payload: " + name);
    if (row_end == -1) row_end = rows;
    if (column_end == -1) column_end = columns;
    const auto block = ggml_blck_size(result.type);
    if (row_begin < 0 || row_begin >= row_end || row_end > rows || column_begin < 0 ||
            column_begin >= column_end || column_end > columns || column_begin % block || column_end % block)
        throw std::invalid_argument("GGUF slices must contain complete quantization blocks: " + name);
    result.out = row_end - row_begin;
    result.neuron_len = column_end - column_begin;
    result.row_bytes = mfq::gguf::row_bytes(result.type, result.neuron_len);
    result.rotation = load_gguf_input_rotation(source, name, columns, column_begin, column_end, device, cuda);
    auto host = tb::empty({static_cast<std::int64_t>(result.row_bytes * result.out)},
        tb::TensorOptions().dtype(tb::kUInt8).device(tb::kCPU));
    const auto read = source.tensor_reader(name);
    auto* destination = reinterpret_cast<std::byte*>(host.data_ptr());
    if (!column_begin && column_end == columns)
        read(row_begin * source_stride, destination, host.nbytes());
    else for (std::int64_t row = 0; row < result.out; ++row)
        read((row_begin + row) * source_stride + column_begin / block * ggml_type_size(result.type),
            destination + row * result.row_bytes, result.row_bytes);
    if (auto residual=source.row_residual(name);residual && residual->payload_bytes()) {
        auto all=std::make_shared<GgufRowCorrection>(std::move(residual),cuda?device:-1);
        result.correction=all->slice(row_begin,row_end,column_begin,column_end);
    }
    if (cuda) {
        MfqCudaGuard guard(device);
        result.packed = host.to(tb::Device(tb::kCUDA, device));
    } else result.packed = std::move(host);
    return result;
}

tb::Tensor GgufWeight::dequantize_stored() const {
    auto output = tb::empty({out, neuron_len}, packed.options().dtype(tb::kFloat32));
    if (packed.is_cuda()) {
        MfqCudaGuard guard(packed.get_device());
        const auto stream=mfq_get_current_cuda_stream(packed.get_device());
        mfq_cuda_record_stream(packed,stream);mfq_cuda_record_stream(output,stream);
        if(quad_storage_rows)mfq::cuda::gguf::decode_quad_rows(type,packed.data_ptr(),quad_storage_rows,
            neuron_len,quad_row_begin,out,nullptr,out,output.data_ptr(),false,stream.stream());
        else mfq::cuda::gguf::dequantize_rows(type,packed.data_ptr(),row_bytes,out,neuron_len,
            output.data_ptr(),false,stream.stream());
    } else {
        for (std::int64_t row = 0; row < out; ++row)
            decode_gguf_cpu(type, packed.data_ptr<std::uint8_t>() + row * row_bytes,
                output.data_ptr<float>() + row * neuron_len, neuron_len);
    }
    return output;
}

tb::Tensor GgufWeight::dequantize() const {
    auto output = dequantize_stored();
    if (rotation) rotation->apply_inplace(output, true);
    if (correction) {
        auto delta=correction->dense(packed.is_cuda()?packed.get_device():-1);
        if(output.is_cuda())output=output+delta;
        else for(std::int64_t i=0;i<output.numel();++i)output.data_ptr<float>()[i]+=delta.data_ptr<float>()[i];
    }
    return output;
}

tb::Tensor GgufWeight::embedding(tb::Tensor ids) const {
    auto output=embedding_base(ids);
    return correction?correction->add_embedding(ids.to(output.device()),std::move(output)):output;
}

tb::Tensor GgufWeight::embedding_base(tb::Tensor ids) const {
    auto shape = ids.sizes().vec();
    shape.push_back(neuron_len);
    ids = ids.to(packed.device()).to(tb::kInt64).contiguous().reshape({-1});
    if (!packed.is_cuda()) {
        auto output = tb::zeros({ids.numel(), neuron_len}, packed.options().dtype(tb::kFloat32));
        for (std::int64_t row = 0; row < ids.numel(); ++row) {
            const auto id = ids.data_ptr<std::int64_t>()[row];
            if (id < 0 || id >= out) continue;
            decode_gguf_cpu(type, packed.data_ptr<std::uint8_t>() + id * row_bytes,
                output.data_ptr<float>() + row * neuron_len, neuron_len);
        }
        return output.reshape(shape);
    }
    MfqCudaGuard guard(packed.get_device());
    auto output = tb::empty({ids.numel(), neuron_len}, packed.options().dtype(tb::kFloat32));
    const auto stream=mfq_get_current_cuda_stream(packed.get_device());
    mfq_cuda_record_stream(packed,stream); mfq_cuda_record_stream(ids,stream); mfq_cuda_record_stream(output,stream);
    if(quad_storage_rows) {
        mfq::cuda::gguf::decode_quad_rows(type,packed.data_ptr(),quad_storage_rows,neuron_len,
            quad_row_begin,out,ids.data_ptr<std::int64_t>(),ids.numel(),output.data_ptr<float>(),false,stream.stream());
    } else mfq::cuda::gguf::embedding_rows(type, packed.data_ptr(), row_bytes, out, neuron_len,
        ids.data_ptr<std::int64_t>(), ids.numel(), output.data_ptr<float>(), mfq_current_cuda_stream());
    if(rotation)rotation->apply_inplace(output,true);
    return output.reshape(shape);
}

bool prefers_gguf_m1_aligned(const std::vector<GgufWeight>& weights,int64_t tokens) {
    return tokens==1 && !weights.empty() &&
        std::any_of(weights.begin(),weights.end(),[](const auto& w){return mfq::cuda::gguf::prefers_m1_aligned(w.type);}) &&
        std::all_of(weights.begin(),weights.end(),[](const auto& w){return !w.base_prefers_original_input() && mfq::cuda::gguf::supports_m1_aligned(w.type);});
}

GgufProjectionInput prepare_gguf_group_input(const std::vector<GgufWeight>& weights,tb::Tensor input) {
    if(weights.empty())return {};
    const auto columns=weights.front().neuron_len;
    TYLOINFER_RUNTIME_CHECK(input.dim()>=1 && columns>0 && input.size(-1)==columns && input.is_cuda(),
        "GGUF projection group input width/device mismatch");
    const auto tokens=input.numel()/columns;
    TYLOINFER_RUNTIME_CHECK(tokens<=32,"GGUF shared quantization requires a small token window");
    for(const auto& weight:weights)
        TYLOINFER_RUNTIME_CHECK(weight.neuron_len==columns && weight.packed.is_cuda() &&
            weight.packed.device()==input.device(),"GGUF projection group contains incompatible input dimensions/devices");
    const auto dtype=input.scalar_type()==tb::kBFloat16?tb::kFloat16:input.scalar_type();
    const MfqCudaGuard guard(input.get_device());
    for(const auto& weight:weights)if(weight.projection_correction)weight.projection_correction->prepare(input);
    const auto stream_handle=mfq_get_current_cuda_stream(input.get_device());
    const auto stream=stream_handle.stream();
    auto source=input.reshape({tokens,columns});
    auto original=source;
    if(source.scalar_type()!=tb::kFloat16)source=source.to(tb::kFloat32);
    source=source.contiguous();
    const auto supported=[&](const auto& weight){return !weight.base_prefers_original_input() && weight.prefers_position_major_q8(tokens);};
    const auto floating_codebook=[](const auto& weight){return weight.base_prefers_original_input();};
    const bool keep_original=std::any_of(weights.begin(),weights.end(),[](const auto& weight){return weight.prefers_original_input();});
    const bool interleaved=std::any_of(weights.begin(),weights.end(),supported);
    const bool native=std::any_of(weights.begin(),weights.end(),[&](const auto& weight){return !supported(weight) && !floating_codebook(weight);});
    const bool aligned_m1=prefers_gguf_m1_aligned(weights,tokens);
    auto activation=native?tb::empty({std::int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(columns))},
        input.options().dtype(tb::kUInt8)):tb::Tensor{};
    auto position_major=interleaved?tb::empty({std::int64_t(mfq::cuda::gguf::interleaved_activation_bytes(tokens,columns))},
        input.options().dtype(tb::kUInt8)):tb::Tensor{};
    for(const auto& tensor:{source,activation,position_major})if(tensor.defined())mfq_cuda_record_stream(tensor,stream_handle);
    if(aligned_m1)mfq::cuda::gguf::quantize_input_m1_aligned(source.data_ptr(),source.scalar_type()==tb::kFloat16,columns,activation.data_ptr(),stream);
    else if(interleaved)
        mfq::cuda::gguf::quantize_input_interleaved(source.data_ptr(),source.scalar_type()==tb::kFloat16,tokens,columns,
            position_major.data_ptr(),activation.defined()?activation.data_ptr():nullptr,stream);
    else if(native && source.scalar_type()==tb::kFloat16)
        mfq::cuda::gguf::quantize_input_f16(source.data_ptr(),tokens,columns,activation.data_ptr(),stream);
    else if(native)mfq::cuda::gguf::quantize_input(source.data_ptr<float>(),tokens,columns,activation.data_ptr(),stream);
    return {std::move(activation),std::move(position_major),keep_original?std::move(original):tb::Tensor{},input.sizes().vec(),tokens,dtype,aligned_m1};
}

tb::Tensor GgufProjectionInput::project(const GgufWeight& weight) const {
    TYLOINFER_RUNTIME_CHECK(!shape.empty(),"GGUF projection input was not prepared");
    TYLOINFER_RUNTIME_CHECK(!aligned_m1 || tokens==1,"Aligned GGUF projection requires one token");
    const bool use_interleaved=weight.prefers_position_major_q8(tokens);
    auto output_shape=shape;output_shape.back()=weight.out;
    if(!tokens)return tb::empty(output_shape,weight.packed.options().dtype(dtype));
    if(weight.base_prefers_original_input()) {
        TYLOINFER_RUNTIME_CHECK(original.defined(),"GGUF floating codebook projection requires the original activation");
        auto base=weight;base.projection_correction.reset();
        auto result=base.forward(original).to(dtype).reshape(output_shape);
        return weight.projection_correction?weight.projection_correction->apply(original,std::move(result)):result;
    }
    auto result=(aligned_m1?weight.forward_m1_aligned_base(native,dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32):weight.forward_q8_base(use_interleaved?interleaved:native,tokens,
        dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32,use_interleaved)).to(dtype).reshape(output_shape);
    if(weight.projection_correction){TYLOINFER_RUNTIME_CHECK(original.defined(),"GGUF neuron correction requires original activation");
        result=weight.projection_correction->apply(original,std::move(result));}
    return result;
}

std::vector<tb::Tensor> project_gguf_group(const std::vector<GgufWeight>& weights,tb::Tensor input) {
    if(weights.empty())return {};
    const auto prepared=prepare_gguf_group_input(weights,std::move(input));
    return prepared.project_all(weights);
}

std::vector<tb::Tensor> GgufProjectionInput::project_all(const std::vector<GgufWeight>& weights) const {
    std::vector<tb::Tensor> output;output.reserve(weights.size());

    const bool grouped_small_m=weights.size()>1 && tokens>=2 && tokens<=8 &&
        std::all_of(weights.begin(),weights.end(),[&](const auto& weight) {
            return weight.quad_projection_ready() && !weight.base_prefers_original_input() &&
                mfq::cuda::gguf::supports_quad_small_m(weight.type,tokens);
        });
    if(grouped_small_m) {
        const MfqCudaGuard guard(weights.front().packed.get_device());
        const auto handle=mfq_get_current_cuda_stream(weights.front().packed.get_device());
        const auto stream=handle.stream();
        for(const auto& packet:{native,interleaved,original})
            if(packet.defined())mfq_cuda_record_stream(packet,handle);
        const auto result_type=dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32;
        std::vector<mfq::cuda::gguf::QuadProjection> quad;
        std::vector<mfq::cuda::gguf::DenseProjection> descriptors;
        std::vector<std::uint8_t> layouts;
        quad.reserve(weights.size());descriptors.reserve(weights.size());layouts.reserve(weights.size());
        for(const auto& weight:weights) {
            auto result=tb::empty({tokens,weight.out},weight.packed.options().dtype(result_type));
            mfq_cuda_record_stream(weight.packed,handle);mfq_cuda_record_stream(result,handle);
            quad.push_back(quad_projection(weight,result.data_ptr()));
            descriptors.push_back(quad.back().projection);
            layouts.push_back(weight.prefers_position_major_q8(tokens));
            output.push_back(std::move(result));
        }
        const auto separate=[&] {
            for(std::size_t i=0;i<quad.size();++i)
                mfq::cuda::gguf::matvec_quad_small_m(quad[i],weights.front().neuron_len,
                    (layouts[i]?interleaved:native).data_ptr(),tokens,layouts[i]!=0,
                    result_type==tb::kFloat16,stream);
        };
        const auto batch=[&] {
            mfq::cuda::gguf::matvec_quad_group_small_m(quad.data(),quad.size(),weights.front().neuron_len,
                native.defined()?native.data_ptr():nullptr,interleaved.defined()?interleaved.data_ptr():nullptr,
                layouts.data(),tokens,result_type==tb::kFloat16,stream);
        };
        const auto pair=[&] {
            mfq::cuda::gguf::matvec_quad_pair_small_m(quad.data(),quad.size(),weights.front().neuron_len,
                native.defined()?native.data_ptr():nullptr,interleaved.defined()?interleaved.data_ptr():nullptr,
                layouts.data(),tokens,result_type==tb::kFloat16,stream);
        };
        std::size_t selected=0;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if(tokens>=3 && quad.size()==2)
            selected=mfq_gguf_dense_schedule::choose_group(descriptors,weights.front().packed.get_device(),
                weights.front().neuron_len,tokens,result_type==tb::kFloat16,stream,separate,batch,pair);
        else selected=mfq_gguf_dense_schedule::choose_batch(descriptors,weights.front().packed.get_device(),
                weights.front().neuron_len,tokens,result_type==tb::kFloat16,stream,separate,batch)?1:0;
#endif
        if(selected==2)pair();else if(selected)batch();else separate();
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
            TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream,&capture));
            if(capture==cudaStreamCaptureStatusActive)
                std::fprintf(stderr,"native_gguf_quad_group_use actual_match=1 M=%lld matrices=%zu selected=%s captured=1\n",
                    static_cast<long long>(tokens),weights.size(),selected==2?"pair":selected?"batch":"separate");
        }
        for(std::size_t i=0;i<weights.size();++i) {
            auto output_shape=shape;output_shape.back()=weights[i].out;
            output[i]=output[i].to(dtype).reshape(output_shape);
            if(weights[i].projection_correction) {
                TYLOINFER_RUNTIME_CHECK(original.defined(),"GGUF grouped neuron correction requires original activation");
                output[i]=weights[i].projection_correction->apply(original,std::move(output[i]));
            }
        }
        return output;
    }
    const bool direct_small_m=std::any_of(weights.begin(),weights.end(),[&](const auto& weight){
        return weight.quad_projection_ready() && mfq::cuda::gguf::supports_quad_small_m(weight.type,tokens);
    });
    if(weights.size()<2 || !tokens || tokens>4 || direct_small_m) {
        for(const auto& weight:weights)output.push_back(project(weight));
        return output;
    }
    TYLOINFER_RUNTIME_CHECK(!aligned_m1 || (tokens==1 && native.defined() && native.is_cuda() &&
        native.device()==weights.front().packed.device() && native.is_contiguous() &&
        native.scalar_type()==tb::kUInt8 && native.nbytes()==mfq::cuda::gguf::q8_row_bytes(weights.front().neuron_len)),
        "Aligned GGUF group activation geometry/device/type differs");
    const MfqCudaGuard guard(weights.front().packed.get_device());
    const auto stream=mfq_get_current_cuda_stream(weights.front().packed.get_device());
    for(const auto& packet:{native,interleaved,original})if(packet.defined())mfq_cuda_record_stream(packet,stream);
    std::vector<mfq::cuda::gguf::DenseProjection> projections;projections.reserve(weights.size());
    std::vector<mfq::cuda::gguf::QuadProjection> quad_projections;quad_projections.reserve(weights.size());
    std::vector<tb::Tensor> canonical_images;canonical_images.reserve(weights.size());
    bool has_quad=false;
    std::vector<std::size_t> projection_indices;projection_indices.reserve(weights.size());
    const auto result_type=dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32;
    for(const auto& weight:weights) {
        if(weight.correction || weight.rotation || (weight.type>=GGML_TYPE_PTQ1_0 && weight.type<=GGML_TYPE_TQK4)) {
            auto base=weight;base.projection_correction.reset();
            output.push_back(project(base));
            continue;
        }
        auto result=tb::empty({tokens,weight.out},weight.packed.options().dtype(result_type));
        mfq_cuda_record_stream(weight.packed,stream);mfq_cuda_record_stream(result,stream);
        auto p=quad_projection(weight,result.data_ptr());
        if(weight.quad_storage_rows && (tokens!=1 || !weight.quad_projection_ready())) {
            const auto image=weight.canonical_packed();mfq_cuda_record_stream(image,stream);
            p.projection.packed=image.data_ptr();p.quad=false;
            canonical_images.push_back(image);
        }
        projections.push_back(p.projection);quad_projections.push_back(p);has_quad|=p.quad;
        projection_indices.push_back(output.size());
        output.push_back(std::move(result));
    }
    const auto finish=[&] {
        for(std::size_t i=0;i<weights.size();++i)if(weights[i].projection_correction) {
            TYLOINFER_RUNTIME_CHECK(original.defined(),"GGUF grouped neuron correction requires original activation");
            output[i]=weights[i].projection_correction->apply(original,std::move(output[i]));
        }
        return output;
    };
    if(projections.empty())return finish();
    if(has_quad) {
        // The mixed kernel consumes 256-weight/eight-fragment formats, the
        // same set supported by aligned M=1. Legacy block sizes still use
        // their normal projection alongside direct interleaved projections.
        const bool compatible=std::all_of(projections.begin(),projections.end(),[](const auto& p){
            return mfq::cuda::gguf::supports_m1_aligned(p.type);
        });
        if(compatible)mfq::cuda::gguf::matvec_quad_group_m1(quad_projections.data(),quad_projections.size(),
            weights.front().neuron_len,native.data_ptr(),aligned_m1,result_type==tb::kFloat16,stream.stream());
        else for(const auto& p:quad_projections)mfq::cuda::gguf::matvec_quad_m1(p,weights.front().neuron_len,
            native.data_ptr(),aligned_m1,result_type==tb::kFloat16,nullptr,stream.stream());
        for(const auto i:projection_indices){auto output_shape=shape;output_shape.back()=weights[i].out;
            output[i]=output[i].to(dtype).reshape(output_shape);}
        return finish();
    }
    const auto separate=[&] {
        for(const auto& p:projections) {
            if(aligned_m1)mfq::cuda::gguf::matvec_m1_aligned(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                native.data_ptr(),p.output,result_type==tb::kFloat16,stream.stream());
            else if(mfq::cuda::gguf::supports_interleaved_activation(p.type,tokens))
                mfq::cuda::gguf::matvec_interleaved(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    interleaved.data_ptr(),tokens,p.output,result_type==tb::kFloat16,stream.stream());
            else if(result_type==tb::kFloat16)
                mfq::cuda::gguf::matvec_q8_f16(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    native.data_ptr(),tokens,p.output,stream.stream());
            else mfq::cuda::gguf::matvec_q8(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    native.data_ptr(),tokens,static_cast<float*>(p.output),stream.stream());
        }
    };
    const auto batch=[&] {
        if(aligned_m1)mfq::cuda::gguf::matvec_group_m1_aligned(projections.data(),projections.size(),weights.front().neuron_len,
            native.data_ptr(),result_type==tb::kFloat16,stream.stream());
        else mfq::cuda::gguf::matvec_group(projections.data(),projections.size(),weights.front().neuron_len,
            native.defined()?native.data_ptr():nullptr,interleaved.defined()?interleaved.data_ptr():nullptr,
            tokens,result_type==tb::kFloat16,stream.stream());
    };
    bool selected=false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    selected=mfq_gguf_dense_schedule::choose_batch(projections,weights.front().packed.get_device(),
        weights.front().neuron_len,tokens,result_type==tb::kFloat16,stream.stream(),separate,batch,aligned_m1);
#endif
    if(selected)batch();else separate();
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&status));
        if(status==cudaStreamCaptureStatusActive)
            std::fprintf(stderr,"native_gguf_dense_schedule_use actual_match=1 M=%lld matrices=%zu selected=%s captured=1\n",
                static_cast<long long>(tokens),weights.size(),selected?"batch":"separate");
    }
    for(const auto i:projection_indices) {
        auto output_shape=shape;output_shape.back()=weights[i].out;
        output[i]=output[i].to(dtype).reshape(output_shape);
    }
    return finish();
}

tb::Tensor GgufWeight::forward_m1_aligned_base(const tb::Tensor& activation,tb::ScalarType dtype) const {
    TYLOINFER_RUNTIME_CHECK(!base_prefers_original_input() && mfq::cuda::gguf::supports_m1_aligned(type) &&
        packed.is_cuda() && activation.is_cuda() && activation.device()==packed.device() &&
        activation.is_contiguous() && activation.scalar_type()==tb::kUInt8 &&
        activation.nbytes()==mfq::cuda::gguf::q8_row_bytes(neuron_len) &&
        (dtype==tb::kFloat16 || dtype==tb::kFloat32),"Aligned GGUF M=1 activation geometry/device/type differs");
    const MfqCudaGuard guard(packed.get_device());const auto stream=mfq_get_current_cuda_stream(packed.get_device());
    auto output=tb::empty({1,out},packed.options().dtype(dtype));
    for(const auto& tensor:{activation,packed,output})mfq_cuda_record_stream(tensor,stream);
    if(quad_projection_ready())mfq::cuda::gguf::matvec_quad_m1(quad_projection(*this,output.data_ptr()),
        neuron_len,activation.data_ptr(),true,dtype==tb::kFloat16,nullptr,stream.stream());
    else {
        const auto image=canonical_packed();mfq_cuda_record_stream(image,stream);
        mfq::cuda::gguf::matvec_m1_aligned(type,image.data_ptr(),row_bytes,out,neuron_len,
            activation.data_ptr(),output.data_ptr(),dtype==tb::kFloat16,stream.stream());
    }
    return output;
}

bool GgufWeight::prefers_position_major_q8(std::int64_t tokens) const noexcept {
    // Three/four queries share coordinate words through aligned vector loads.
    // The two-query native packet retains its smaller register working set.
    if(tokens>=3 && tokens<=8 && quad_projection_ready() &&
            (type==GGML_TYPE_IQ2_XXS || type==GGML_TYPE_IQ2_XS || type==GGML_TYPE_IQ2_S))return true;
    if(tokens>=5 && tokens<=8 && packed.defined() && packed.is_cuda() &&
            mfq::cuda::gguf::supports_dense_split_window_q8(type,packed.data_ptr(),row_bytes,out,neuron_len,tokens))return true;
    if(tokens>=5 && tokens<=8 && quad_projection_ready() &&
            (type==GGML_TYPE_IQ1_S || type==GGML_TYPE_IQ1_M || type==GGML_TYPE_Q2_K))return true;
    return mfq::cuda::gguf::supports_interleaved_activation(type,tokens) &&
        !(quad_projection_ready() && mfq::cuda::gguf::supports_quad_small_m(type,tokens));
}

tb::Tensor GgufWeight::forward_q8(const tb::Tensor& activation,std::int64_t tokens,tb::ScalarType dtype,bool position_major) const {
    TYLOINFER_RUNTIME_CHECK(!projection_correction,"GGUF neuron correction requires its original activation");
    return forward_q8_base(activation,tokens,dtype,position_major);
}
tb::Tensor GgufWeight::forward_q8_base(const tb::Tensor& activation,std::int64_t tokens,tb::ScalarType dtype,bool position_major) const {
    TYLOINFER_RUNTIME_CHECK(!rotation && !correction,"GGUF rotated/residual projection requires its original activation before Q8 quantization");
    const auto bytes_per_row=mfq::cuda::gguf::q8_row_bytes(neuron_len);
    const auto bytes=position_major?mfq::cuda::gguf::interleaved_activation_bytes(tokens,neuron_len)
        :std::uint64_t(tokens)*bytes_per_row;
    TYLOINFER_RUNTIME_CHECK(packed.is_cuda() && activation.is_cuda() && activation.device()==packed.device() &&
        activation.is_contiguous() && activation.scalar_type()==tb::kUInt8 && tokens>=1 &&
        std::uint64_t(tokens)<=std::numeric_limits<std::uint64_t>::max()/bytes_per_row &&
        (dtype==tb::kFloat16 || dtype==tb::kFloat32) &&
        (!position_major || mfq::cuda::gguf::supports_interleaved_activation(type,tokens) || prefers_position_major_q8(tokens)) &&
        activation.nbytes()==bytes,"GGUF prepared activation geometry/device/type differs");
    const MfqCudaGuard guard(packed.get_device());const auto stream_handle=mfq_get_current_cuda_stream(packed.get_device());
    const auto stream=stream_handle.stream();
    auto result=tb::empty({tokens,out},packed.options().dtype(dtype));
    for(const auto& tensor:{activation,packed,result})mfq_cuda_record_stream(tensor,stream_handle);
    if(tokens==1 && !position_major && quad_projection_ready()) {
        mfq::cuda::gguf::matvec_quad_m1(quad_projection(*this,result.data_ptr()),neuron_len,
            activation.data_ptr(),false,dtype==tb::kFloat16,nullptr,stream);return result;
    }
    if(quad_projection_ready() && mfq::cuda::gguf::supports_quad_small_m(type,tokens)) {
        mfq::cuda::gguf::matvec_quad_small_m(quad_projection(*this,result.data_ptr()),neuron_len,
            activation.data_ptr(),tokens,position_major,dtype==tb::kFloat16,stream);
        return result;
    }
    const auto image=canonical_packed();mfq_cuda_record_stream(image,stream_handle);
    if(position_major && mfq::cuda::gguf::project_dense_split_window_q8(type,image.data_ptr(),row_bytes,out,neuron_len,
            activation.data_ptr(),tokens,result.data_ptr(),dtype==tb::kFloat16,stream))return result;
    if(position_major)mfq::cuda::gguf::matvec_interleaved(type,image.data_ptr(),row_bytes,out,neuron_len,
        activation.data_ptr(),tokens,result.data_ptr(),dtype==tb::kFloat16,stream);
    else if(dtype==tb::kFloat16)mfq::cuda::gguf::matvec_q8_f16(type,image.data_ptr(),row_bytes,out,neuron_len,
        activation.data_ptr(),tokens,result.data_ptr(),stream);
    else mfq::cuda::gguf::matvec_q8(type,image.data_ptr(),row_bytes,out,neuron_len,
        activation.data_ptr(),tokens,result.data_ptr<float>(),stream);
    return result;
}
tb::Tensor GgufWeight::forward(tb::Tensor input,bool allow_mmq) const {
    if(projection_correction)projection_correction->prepare(input);
    auto output=forward_base(input,allow_mmq);
    if(correction)output=correction->add(input,std::move(output));
    return projection_correction?projection_correction->apply(input,std::move(output)):output;
}

bool GgufWeight::can_forward_prefill_residual(int64_t tokens,const tb::Tensor& residual) const {
    const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
    if(tokens<=32 || (setting?setting[0]!='1' && setting[0]!='a':false) ||
            ((!setting || setting[0]=='a') && tokens<512) ||
            correction || projection_correction || !packed.is_cuda() || out<1 || neuron_len<1 ||
            !residual.is_cuda() || residual.device()!=packed.device() || !residual.is_contiguous() ||
            residual.scalar_type()!=tb::kFloat16 || residual.dim()<1 || residual.size(-1)!=out ||
            residual.numel()/out!=tokens)return false;
    const MfqCudaGuard guard(packed.get_device());
    return mfq::cuda::gguf::grouped_f16_available(packed.get_device());
}

tb::Tensor GgufWeight::forward_base(tb::Tensor input,bool allow_mmq) const {
    return forward_base(std::move(input),allow_mmq,{});
}

tb::Tensor GgufWeight::forward_base(tb::Tensor input,bool allow_mmq,const tb::Tensor& prefill_residual) const {
    TYLOINFER_RUNTIME_CHECK(input.dim() >= 1 && input.size(-1) == neuron_len,
        "GGUF projection input width mismatch");
    auto shape = input.sizes().vec();
    shape.back() = out;
    // CUDA attention and sampling consume FP16/FP32 activations. BF16 norms
    // from a GGUF source must not propagate BF16 logits into the sampler.
    const auto dtype = input.scalar_type() == tb::kBFloat16 ? tb::kFloat16 : input.scalar_type();
    auto x = input.reshape({-1, neuron_len}).contiguous();
    if(rotation)x=rotation->apply(x);
    TYLOINFER_RUNTIME_CHECK(!prefill_residual.defined() || (allow_mmq && dtype==tb::kFloat16 &&
        can_forward_prefill_residual(x.size(0),prefill_residual)),"Invalid GGUF prefill residual projection");
    if (!packed.is_cuda())
        return tb::matmul(x.to(tb::kFloat32), dequantize_stored().transpose(0, 1)).to(dtype).reshape(shape);
    MfqCudaGuard guard(packed.get_device());
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.get_device() == packed.get_device(),
        "GGUF weight and input must share a device");
    const auto tokens = x.size(0);
    if(quad_storage_rows && tokens<=32) {
        if(tokens==1 || (quad_projection_ready() && mfq::cuda::gguf::supports_quad_small_m(type,tokens))) {
            auto base=*this;base.projection_correction.reset();
            const auto prepared=prepare_gguf_group_input({base},x);
            return prepared.project(base).to(dtype).reshape(shape);
        }
        auto base=*this;base.packed=canonical_packed();base.quad_storage_rows=0;base.quad_row_begin=0;
        base.rotation.reset();
        // x already has the input rotation. Preserve only the base projection;
        // the outer forward() owns residual and neuron corrections.
        return base.forward_base(x,allow_mmq).to(dtype).reshape(shape);
    }
    const auto output_dtype=dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32;
    auto output = tb::empty({tokens, out}, packed.options().dtype(output_dtype));
    const auto stream=mfq_get_current_cuda_stream(packed.get_device());
    mfq_cuda_record_stream(packed,stream); mfq_cuda_record_stream(output,stream);
    if(prefill_residual.defined())mfq_cuda_record_stream(prefill_residual,stream);
    // Trellis weights decode to floating-point codebook entries. Preserve the
    // original activation instead of requantizing it to Q8 for a float dot.
    // Large prefill batches retain the tensor-core path below.
    if (type >= GGML_TYPE_TQ2_T && type <= GGML_TYPE_TQK4 && tokens <= 32) {
        auto source = x.scalar_type() == tb::kFloat16 ? x : x.to(tb::kFloat32).contiguous();
        mfq_cuda_record_stream(source,stream);
        mfq::cuda::gguf::compact_project(mfq::cuda::gguf::CompactFormat(type), packed.data_ptr(), row_bytes,
            out, neuron_len, source.data_ptr(), source.scalar_type() == tb::kFloat16
                ? mfq::cuda::gguf::CompactInput::Float16 : mfq::cuda::gguf::CompactInput::Float32,
            tokens, output.data_ptr(), output_dtype == tb::kFloat16, mfq_current_cuda_stream());
        return output.to(dtype).reshape(shape);
    }
    if (tokens <= 32) {
        auto source=x.scalar_type()==tb::kFloat16?x:x.to(tb::kFloat32).contiguous();
        mfq_cuda_record_stream(source,stream);
        const bool interleaved=prefers_position_major_q8(tokens);
        const bool aligned_m1=tokens==1 && mfq::cuda::gguf::prefers_m1_aligned(type);
        auto activation = tb::empty({static_cast<std::int64_t>(interleaved
                ?mfq::cuda::gguf::interleaved_activation_bytes(tokens,neuron_len)
                :tokens * mfq::cuda::gguf::q8_row_bytes(neuron_len))},
            packed.options());
        mfq_cuda_record_stream(activation,stream);
        if(aligned_m1)mfq::cuda::gguf::quantize_input_m1_aligned(source.data_ptr(),source.scalar_type()==tb::kFloat16,
            neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        else if(interleaved)
            mfq::cuda::gguf::quantize_input_interleaved(source.data_ptr(),source.scalar_type()==tb::kFloat16,
                tokens,neuron_len,activation.data_ptr(),nullptr,mfq_current_cuda_stream());
        else if(source.scalar_type()==tb::kFloat16)
            mfq::cuda::gguf::quantize_input_f16(source.data_ptr(),tokens,neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        else mfq::cuda::gguf::quantize_input(source.data_ptr<float>(),tokens,neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        if(aligned_m1)mfq::cuda::gguf::matvec_m1_aligned(type,packed.data_ptr(),row_bytes,out,neuron_len,
            activation.data_ptr(),output.data_ptr(),output_dtype==tb::kFloat16,mfq_current_cuda_stream());
        else if(interleaved && mfq::cuda::gguf::project_dense_split_window_q8(type,packed.data_ptr(),row_bytes,out,neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),output_dtype==tb::kFloat16,mfq_current_cuda_stream())) {}
        else if(interleaved)
            mfq::cuda::gguf::matvec_interleaved(type,packed.data_ptr(),row_bytes,out,neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),output_dtype==tb::kFloat16,mfq_current_cuda_stream());
        else if(output_dtype==tb::kFloat16)
            mfq::cuda::gguf::matvec_q8_f16(type,packed.data_ptr(),row_bytes,out,neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),mfq_current_cuda_stream());
        else mfq::cuda::gguf::matvec_q8(type,packed.data_ptr(),row_bytes,out,neuron_len,
            activation.data_ptr(),tokens,output.data_ptr<float>(),mfq_current_cuda_stream());
    } else {
        namespace mmq=mfq::cuda::gguf::mmq;
        const auto* mmq_setting=std::getenv("TYLOINFER_GGUF_DENSE_MMQ");
        const auto* fast_setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
        // Large prompts amortize decoding over many tokens. Keep MMQ for short
        // matrices and explicit non-MMQ reference calls. An explicit 1/0
        // permits enabling/disabling the reduced-accumulation GEMM mode.
        const bool fast_prefill=((fast_setting && fast_setting[0]=='1') ||
            ((!fast_setting || fast_setting[0]=='a') && tokens>=512 && allow_mmq)) &&
            mfq::cuda::gguf::grouped_f16_available(packed.get_device());
        const auto mmq_rows=fast_prefill?0:prefill_weight_rows(out,row_bytes,128,packed.get_device());
        const auto final_rows=mmq_rows>0?out%mmq_rows:0;
        if(allow_mmq && (!mmq_setting || mmq_setting[0]!='0') && mmq_rows>0 && tokens<=INT_MAX && neuron_len<=INT_MAX &&
                (dtype==tb::kFloat16 || dtype==tb::kFloat32) && mmq::fits(type,mmq_rows,packed.get_device()) &&
                (!final_rows || mmq::fits(type,final_rows,packed.get_device()))) {
            auto input_float=x.to(tb::kFloat32).contiguous();
            tb::Tensor exponents,scaled;
            if(dtype==tb::kFloat32) {
                scaled=tb::empty_like(input_float);
                exponents=tb::empty({tokens},packed.options().dtype(tb::kInt32));
                mfq::cuda::gguf::pack_scaled_f32(input_float.data_ptr<float>(),scaled.data_ptr<float>(),
                    exponents.data_ptr<std::int32_t>(),tokens,neuron_len,mfq_current_cuda_stream());
                input_float=scaled;
            }
            auto activation=tb::empty({std::int64_t(mmq::activation_bytes(tokens,neuron_len))},packed.options());
            mmq::quantize(input_float.data_ptr<float>(),nullptr,activation.data_ptr(),int(type),neuron_len,tokens,mfq_current_cuda_stream());
            auto result=output_dtype==tb::kFloat32?output:tb::empty({tokens,out},packed.options().dtype(tb::kFloat32));
            const auto guard_bytes=std::size_t(128)*ggml_type_size(type);
            auto weight=tb::empty({std::int64_t(mmq_rows*row_bytes+guard_bytes)},packed.options());
            for(const auto& tensor:{input_float,exponents,activation,result,weight})if(tensor.defined())mfq_cuda_record_stream(tensor,stream);
            mmq::Context context;
            for(std::int64_t begin=0;begin<out;begin+=mmq_rows) {
                const auto count=std::min(mmq_rows,out-begin);
                const auto bytes=std::uint64_t(count)*row_bytes;
                if(quad_storage_rows)mfq::cuda::gguf::unpack_quad_storage(type,packed.data_ptr(),
                    quad_storage_rows,neuron_len,quad_row_begin+begin,count,nullptr,count,
                    weight.data_ptr(),row_bytes,mfq_current_cuda_stream());
                else TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(weight.data_ptr(),packed.data_ptr<std::uint8_t>()+begin*row_bytes,
                    bytes,cudaMemcpyDeviceToDevice,mfq_current_cuda_stream()));
                TYLOINFER_CUDA_CHECK(cudaMemsetAsync(weight.data_ptr<std::uint8_t>()+bytes,0,guard_bytes,mfq_current_cuda_stream()));
                context.run({weight.data_ptr(),int(type),count,neuron_len,std::size_t(bytes),1,activation.data_ptr(),nullptr,nullptr,
                    tokens,tokens,result.data_ptr<float>()+begin,out},mfq_current_cuda_stream());
            }
            if(exponents.defined())mfq::cuda::gguf::rescale_f32(result.data_ptr<float>(),exponents.data_ptr<std::int32_t>(),
                tokens,out,mfq_current_cuda_stream());
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported[GGML_TYPE_COUNT]{};
                if(!reported[int(type)].exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_gguf_dense_mmq actual_match=1 format=%d tokens=%lld rows=%lld columns=%lld scaled_fp32=%d tile_rows=%lld\n",
                    int(type),static_cast<long long>(tokens),static_cast<long long>(out),static_cast<long long>(neuron_len),int(exponents.defined()),static_cast<long long>(mmq_rows));
            }
            return result.to(dtype).reshape(shape);
        }
        // Prefill shares each decoded tile across every token. Temporary weight
        // expansion is bounded and never changes permanent compressed storage.
        const bool half = dtype == tb::kFloat16;
        const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
        const bool scaled_half=!half && dtype==tb::kFloat32 && (fast_prefill || !setting || setting[0]!='0') &&
            mfq::cuda::gguf::grouped_f16_available(packed.get_device());
        x = x.to(half ? tb::kFloat16 : tb::kFloat32).contiguous();
        const auto cached_exponents=fast_prefill && scaled_half?prefill_cached_exponents(*this):tb::Tensor{};
        const bool fused_scaled=cached_exponents.defined();
        // Include both decoded and optional scaled source images. Direct BLAS
        // output removes the tile result and its strided copy for tensor cores.
        const auto workspace_row_bytes=std::uint64_t(neuron_len)*(half || fused_scaled?2:4)+
            (scaled_half && !fused_scaled?std::uint64_t(neuron_len)*2:0)+(!half && !scaled_half?std::uint64_t(tokens)*4:0)+(fast_prefill?std::uint64_t(tokens)*2:0);
        const auto tile_rows=prefill_weight_rows(out,workspace_row_bytes,32,packed.get_device(),true);
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
        const bool pipelined=tile_rows<out && capture==cudaStreamCaptureStatusNone;
        const auto decoder=pipelined?mfq::cuda::stream_from_pool(false,packed.get_device()):stream;
        const int buffers=pipelined?2:1;
        tb::Tensor tile[2],scaled_weight[2],weight_exponents[2],scaled_x,x_exponents;
        auto tile_output=fast_prefill && !half?tb::empty({tokens,tile_rows},x.options().dtype(tb::kFloat16)):tb::Tensor{};
        if(tile_output.defined())mfq_cuda_record_stream(tile_output,stream);
        const auto pack_half=fast_prefill?mfq::cuda::gguf::pack_scaled_f16_unit:mfq::cuda::gguf::pack_scaled_f16;
        std::optional<mfq::cuda::Event> ready[2],consumed[2];
        for(int i=0;i<buffers;++i) {
            tile[i]=tb::empty({tile_rows,neuron_len},x.options().dtype(fused_scaled?tb::kFloat16:x.scalar_type()));
            mfq_cuda_record_stream(tile[i],stream);mfq_cuda_record_stream(tile[i],decoder);
            if(scaled_half && !fused_scaled) {
                scaled_weight[i]=tb::empty({tile_rows,neuron_len},x.options().dtype(tb::kFloat16));
                weight_exponents[i]=tb::empty({tile_rows},x.options().dtype(tb::kInt32));
                for(const auto& tensor:{scaled_weight[i],weight_exponents[i]}) {
                    mfq_cuda_record_stream(tensor,stream);mfq_cuda_record_stream(tensor,decoder);
                }
            }
            if(pipelined){ready[i].emplace();consumed[i].emplace();}
        }
        const auto context=mfq::cuda::default_context(packed.get_device());
        auto& blas=context->blas(mfq_current_cuda_stream());
        if(scaled_half && !(fast_prefill && GgufPrefillInputScope::lookup(x,scaled_x,x_exponents))) {
            scaled_x=tb::empty(x.sizes(),x.options().dtype(tb::kFloat16));
            x_exponents=tb::empty({tokens},x.options().dtype(tb::kInt32));
            for(const auto& tensor:{scaled_x,x_exponents})mfq_cuda_record_stream(tensor,stream);
            pack_half(x.data_ptr<float>(),scaled_x.data_ptr(),x_exponents.data_ptr<std::int32_t>(),
                tokens,neuron_len,mfq_current_cuda_stream());
        }
        if(pipelined) {
            mfq::cuda::Event source_ready;source_ready.record(mfq_current_cuda_stream());
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(decoder.stream(),source_ready.get(),0));
            mfq_cuda_record_stream(packed,decoder);
        }
        const auto prepare=[&](std::int64_t index) {
            const auto begin=index*tile_rows,count=std::min(tile_rows,out-begin);
            const int slot=int(index%buffers);
            if(fused_scaled) {
                const auto* e=cached_exponents.data_ptr<std::int32_t>()+begin;
                if(quad_storage_rows)mfq::cuda::gguf::decode_quad_rows_scaled(type,packed.data_ptr(),quad_storage_rows,
                    neuron_len,quad_row_begin+begin,count,e,tile[slot].data_ptr(),decoder.stream());
                else mfq::cuda::gguf::dequantize_rows_scaled(type,packed.data_ptr<std::uint8_t>()+begin*row_bytes,
                    row_bytes,count,neuron_len,e,tile[slot].data_ptr(),decoder.stream());
                mfq_cuda_record_stream(cached_exponents,decoder);
            } else if(quad_storage_rows)mfq::cuda::gguf::decode_quad_rows(type,packed.data_ptr(),quad_storage_rows,
                neuron_len,quad_row_begin+begin,count,nullptr,count,tile[slot].data_ptr(),half,decoder.stream());
            else mfq::cuda::gguf::dequantize_rows(type,packed.data_ptr<std::uint8_t>()+begin*row_bytes,row_bytes,
                count,neuron_len,tile[slot].data_ptr(),half,decoder.stream());
            if(scaled_half && !fused_scaled)pack_half(tile[slot].data_ptr<float>(),scaled_weight[slot].data_ptr(),
                weight_exponents[slot].data_ptr<std::int32_t>(),count,neuron_len,decoder.stream());
            if(pipelined)ready[slot]->record(decoder.stream());
        };
        const auto tiles=(out+tile_rows-1)/tile_rows;
        for(int i=0;i<buffers && i<tiles;++i)prepare(i);
        for(std::int64_t index=0;index<tiles;++index) {
            const auto begin=index*tile_rows,count=std::min(tile_rows,out-begin);
            const int slot=int(index%buffers);
            if(pipelined)TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(mfq_current_cuda_stream(),ready[slot]->get(),0));
            if(fast_prefill && half) {
                auto* destination=static_cast<std::uint8_t*>(output.data_ptr())+begin*output.element_size();
                if(prefill_residual.defined()) {
                    const auto* residual=static_cast<const std::uint8_t*>(prefill_residual.data_ptr())+begin*sizeof(std::uint16_t);
                    auto plan=mfq::cuda::gguf::prepare_prefill_residual_gemm(x.data_ptr(),tile[slot].data_ptr(),residual,
                        destination,tokens,count,neuron_len,out,mfq_current_cuda_stream());
                    tb::Tensor workspace;
                    if(plan && plan.workspace_bytes) {
                        try {
                            workspace=tb::empty({int64_t(plan.workspace_bytes)},x.options().dtype(tb::kUInt8));
                            mfq_cuda_record_stream(workspace,stream);
                        } catch(const MfqResourceExhausted&) {
                            (void)cudaGetLastError();
                            plan={}; // Keep the ordinary result when scratch does not fit.
                        }
                    }
                    mfq::cuda::gguf::prefill_gemm_f16_residual(x.data_ptr(),tile[slot].data_ptr(),residual,destination,
                        tokens,count,neuron_len,out,plan,workspace.defined()?workspace.data_ptr():nullptr,
                        workspace.defined()?workspace.nbytes():0,blas.get(),mfq_current_cuda_stream());
                }
                else mfq::cuda::gguf::prefill_gemm_f16_accum(x.data_ptr(),tile[slot].data_ptr(),nullptr,
                    destination,true,nullptr,nullptr,tokens,count,neuron_len,out,blas.get(),mfq_current_cuda_stream());
            } else if(fast_prefill) {
                auto* destination=static_cast<std::uint8_t*>(output.data_ptr())+begin*output.element_size();
                mfq::cuda::gguf::prefill_gemm_f16_accum(scaled_x.data_ptr(),fused_scaled?tile[slot].data_ptr():scaled_weight[slot].data_ptr(),
                    tile_output.data_ptr(),destination,output_dtype==tb::kFloat16,
                    x_exponents.data_ptr<std::int32_t>(),fused_scaled?cached_exponents.data_ptr<std::int32_t>()+begin:weight_exponents[slot].data_ptr<std::int32_t>(),
                    tokens,count,neuron_len,out,blas.get(),mfq_current_cuda_stream());
            } else if(half || scaled_half) {
                auto* destination=static_cast<std::uint8_t*>(output.data_ptr())+begin*output.element_size();
                mfq::cuda::gguf::prefill_gemm_f16(scaled_half?scaled_x.data_ptr():x.data_ptr(),
                    scaled_half?scaled_weight[slot].data_ptr():tile[slot].data_ptr(),destination,half,
                    scaled_half?x_exponents.data_ptr<std::int32_t>():nullptr,
                    scaled_half?weight_exponents[slot].data_ptr<std::int32_t>():nullptr,
                    tokens,count,neuron_len,out,blas.get(),mfq_current_cuda_stream());
            } else output.slice(1,begin,begin+count).copy_(dense_projection(x,tile[slot].narrow(0,0,count)).to(output_dtype));
            if(index+buffers<tiles) {
                if(pipelined) {
                    consumed[slot]->record(mfq_current_cuda_stream());
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(decoder.stream(),consumed[slot]->get(),0));
                }
                prepare(index+buffers);
            }
        }
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported[GGML_TYPE_COUNT][2]{};
            if(!reported[int(type)][int(pipelined)].exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "native_gguf_prefill_stream actual_match=1 format=%d M=%lld N=%lld K=%lld tile_rows=%lld buffers=%d direct_output=%d f16_accum=%d fused_scale_decode=%d\n",
                int(type),static_cast<long long>(tokens),static_cast<long long>(out),static_cast<long long>(neuron_len),
                static_cast<long long>(tile_rows),buffers,int(half || scaled_half),int(fast_prefill),int(fused_scaled));
        }
    }
    return output.to(dtype).reshape(shape);
}
