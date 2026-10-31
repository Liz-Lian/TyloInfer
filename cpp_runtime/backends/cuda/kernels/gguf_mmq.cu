// Contiguous-group MMQ adapted from Strata src/prefill/moe_mmq.cu (MIT,
// ../../../third_party/strata.LICENSE). The kernels/quantizer are llama.cpp's
// matching vendored sources. Scratch uses the runtime's budgeted allocator.
#include "gguf_mmq.h"
#include "gguf_mmq_tab.h"
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mfq_cuda_context.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {
struct NativePool final:ggml_cuda_pool {
    std::shared_ptr<mfq::cuda::Context> context;
    int device;
    explicit NativePool(int selected):context(mfq::cuda::default_context(selected)),device(selected) {}
    void* alloc(size_t bytes,size_t* actual) override {
        *actual=bytes;return context->allocate(bytes,mfq::cuda::current_stream(device).stream());
    }
    void free(void* p,size_t bytes) override {
        context->release(p,bytes,mfq::cuda::current_stream(device).stream());
    }
};
}
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device,int) {
    return std::make_unique<NativePool>(device);
}
ggml_backend_cuda_context::~ggml_backend_cuda_context() {}

namespace mfq::cuda::gguf::mmq {
namespace {
void check() {const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
bool supported(int type) {
    switch(type) {
        case GGML_TYPE_Q2_0:case GGML_TYPE_IQ2_XXS:case GGML_TYPE_IQ2_XS:case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:case GGML_TYPE_IQ3_S:case GGML_TYPE_IQ4_NL:case GGML_TYPE_IQ4_XS:case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:case GGML_TYPE_Q5_K:case GGML_TYPE_Q6_K:case GGML_TYPE_Q5_1:return true;
        default:return false;
    }
}
std::int64_t pad_columns(std::int64_t columns) {
    if(columns<=0 || columns>INT_MAX-511)throw std::invalid_argument("GGUF MMQ reduction dimension is invalid");
    return (columns+511)/512*512;
}
struct GatherRange {
    const std::uint8_t* source;
    std::uint8_t* destination;
    std::uint64_t bytes;
    std::uint32_t first,blocks;
};
struct GatherPacket {GatherRange ranges[3*group_capacity+3];int count=0;};
__global__ void gather_ranges(GatherPacket packet) {
    __shared__ int range;
    if(!threadIdx.x) {
        int low=0,high=packet.count;
        while(low<high) {
            const int middle=(low+high)/2;
            if(packet.ranges[middle].first<=blockIdx.x)low=middle+1;else high=middle;
        }
        range=low-1;
    }
    __syncthreads();
    const auto& job=packet.ranges[range];
    const auto i=std::uint64_t(blockIdx.x-job.first)*4096+threadIdx.x*16;
    if(i>=job.bytes)return;
    auto* out=job.destination+i;
    const auto* in=job.source?job.source+i:nullptr;
    if(job.bytes-i>=16 && ((reinterpret_cast<std::uintptr_t>(out)|reinterpret_cast<std::uintptr_t>(in))&15)==0)
        *reinterpret_cast<uint4*>(out)=in?*reinterpret_cast<const uint4*>(in):make_uint4(0,0,0,0);
    else for(int byte=0;byte<16 && i+byte<job.bytes;++byte)out[byte]=in?in[byte]:0;
}
__global__ void clear_activation_guard_kernel(std::uint32_t* guard) {
    constexpr auto words=128*sizeof(block_q8_1_mmq)/sizeof(std::uint32_t);
    static_assert(128*sizeof(block_q8_1_mmq)%sizeof(std::uint32_t)==0);
    for(auto i=std::size_t(threadIdx.x);i<words;i+=blockDim.x)guard[i]=0;
}
void clear_activation_guard(void* output,std::size_t bytes,cudaStream_t stream) {
    // Keep the small padded Q8 tail on the compute stream as a kernel,
    // independently of the runtime's memset/copy-engine scheduling.
    auto* guard=static_cast<char*>(output)+bytes-128*sizeof(block_q8_1_mmq);
    clear_activation_guard_kernel<<<1,256,0,stream>>>(reinterpret_cast<std::uint32_t*>(guard));
    check();
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "native_gguf_mmq_guard_clear actual_match=1 method=kernel bytes=%zu\n",128*sizeof(block_q8_1_mmq));
    }
}
__global__ void activate(const float* gu,float* hidden,std::int64_t count,std::int64_t width,float limit) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto at=i/width*2*width+i%width;
    float gate=gu[at],up=gu[at+width];
    if(limit>0.f){gate=fminf(gate,limit);up=fmaxf(-limit,fminf(up,limit));}
    hidden[i]=gate/(1.f+__expf(-gate))*up;
}
__global__ void scatter_rows(const float* source,half* pairs,const std::int32_t* ids,std::int64_t count,std::int64_t width) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)pairs[std::int64_t(ids[i/width])*width+i%width]=__float2half_rn(source[i]);
}
}
bool fits(int type,std::int64_t rows,int device) {
    if(!supported(type) || rows<=0 || rows>INT_MAX || device<0 || device>=ggml_cuda_info().device_count)return false;
    const auto& gpu=ggml_cuda_info().devices[device];
    for(int j=8;j<=128;j+=8) {
        const auto config=ggml_cuda_mmq_get_config(ggml_type(type),j,rows%128!=0,gpu.cc);
        if(config.type!=GGML_TYPE_COUNT && mmq_get_nbytes_shared(config,gpu.cc)<=gpu.smpbo)return true;
    }
    return false;
}
bool direct_fits(int type,std::int64_t rows,std::int64_t columns,int device) {
    if(!fits(type,rows,device) || rows%128 || columns<=0 || columns%ggml_blck_size(ggml_type(type)))return false;
    // The imported table scheduler covers complete row tiles. Only use raw
    // cache images when no config can read beyond the matrix's K tail;
    // otherwise the existing zero-guarded group remains necessary.
    const auto qk=ggml_blck_size(ggml_type(type)),blocks=columns/qk;
    const auto no_overread=[&](std::int64_t k){return k/qk>0 && blocks%(k/qk)==0;};
    if(!no_overread(MMQ_ITER_K))return false;
    const auto& gpu=ggml_cuda_info().devices[device];
    for(int j=8;j<=128;j+=8) {
        const auto config=ggml_cuda_mmq_get_config(ggml_type(type),j,false,gpu.cc);
        if(config.type!=GGML_TYPE_COUNT && mmq_get_nbytes_shared(config,gpu.cc)<=gpu.smpbo && !no_overread(config.K_vram))return false;
    }
    return true;
}
std::size_t activation_bytes(std::int64_t rows,std::int64_t columns) {
    const auto blocks=pad_columns(columns)/QK8_1_MMQ;
    if(rows<=0 || std::uint64_t(rows)>((SIZE_MAX/sizeof(block_q8_1_mmq))-128)/std::uint64_t(blocks))
        throw std::overflow_error("GGUF MMQ activation size overflows");
    return (std::size_t(rows)*std::size_t(blocks)+128)*sizeof(block_q8_1_mmq);
}
void quantize(const float* input,const std::int32_t* ids,void* output,int type,
        std::int64_t columns,std::int64_t rows,cudaStream_t stream) {
    if(!supported(type) || columns%ggml_blck_size(ggml_type(type)) || rows<=0 || rows>INT_MAX)
        throw std::invalid_argument("GGUF MMQ activation geometry differs");
    clear_activation_guard(output,activation_bytes(rows,columns),stream);
    quantize_mmq_q8_1_cuda(input,ids,output,ggml_type(type),columns,columns,rows*columns,rows*columns,
        pad_columns(columns),rows,1,1,stream);check();
}
void quantize_scatter(const float* input,const std::int32_t* inverse,void* output,int type,
        std::int64_t columns,std::int64_t tokens,int routes,cudaStream_t stream) {
    if(!supported(type) || columns%ggml_blck_size(ggml_type(type)) || tokens<=0 || routes<=0 ||
            tokens>INT_MAX/routes || !inverse)
        throw std::invalid_argument("GGUF MMQ scatter geometry differs");
    const auto rows=tokens*routes;
    clear_activation_guard(output,activation_bytes(rows,columns),stream);
    quantize_scatter_mmq_q8_1_cuda(input,inverse,output,ggml_type(type),columns,columns,
        pad_columns(columns),tokens,rows,routes,stream);check();
}
void swiglu_quantize(const float* gate_up,void* output,int type,
        std::int64_t columns,std::int64_t rows,float limit,cudaStream_t stream) {
    if(!supported(type) || columns%ggml_blck_size(ggml_type(type)) || rows<=0 || rows>INT_MAX)
        throw std::invalid_argument("GGUF MMQ SwiGLU activation geometry differs");
    (void)activation_bytes(rows,columns);
    quantize_swiglu_mmq_q8_1_cuda(gate_up,output,ggml_type(type),columns,pad_columns(columns),int(rows),limit,stream);
    check();
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true))std::fprintf(stderr,
            "native_gguf_mmq_swiglu_quantize actual_match=1 type=%d width=%lld rows=%lld materialized_hidden=0 guard_in_kernel=1\n",
            type,static_cast<long long>(columns),static_cast<long long>(rows));
    }
}
struct Context::Impl {ggml_backend_cuda_context context;Impl():context(ggml_cuda_get_device()) {}};
Context::Context():impl_(std::make_unique<Impl>()) {}
Context::~Context()=default;
void Context::run(const Product& p,cudaStream_t stream) {
    if(p.experts<=0 || p.max_tokens<=0)return;
    const auto type=ggml_type(p.type);
    if(!fits(p.type,p.rows,ggml_cuda_get_device()) || p.columns%ggml_blck_size(type) ||
            p.expert_bytes%ggml_type_size(type) || p.tokens<=0 || p.output_stride<p.rows)
        throw std::invalid_argument("GGUF MMQ product geometry differs");
    const auto activation_rows=p.activation_rows?p.activation_rows:p.tokens;
    if(activation_rows<p.tokens || p.activation_offset<0 || p.activation_offset>activation_rows-p.tokens)
        throw std::invalid_argument("GGUF MMQ activation view exceeds its rows");
    const auto* activation=reinterpret_cast<const int*>(static_cast<const block_q8_1_mmq*>(p.activation)+p.activation_offset);
    const mmq_args args={static_cast<const char*>(p.weights),type,activation,p.ids,p.bounds,p.output,nullptr,
        p.columns,p.rows,p.tokens,p.columns/ggml_blck_size(type),activation_rows,p.output_stride,
        p.experts,p.experts,std::int64_t(p.expert_bytes/ggml_type_size(type)),0,0,
        1,1,0,0,0,p.max_tokens,p.max_tokens};
    if(p.weight_addresses) {
        if(p.experts>group_capacity || !direct_fits(p.type,p.rows,p.columns,ggml_cuda_get_device()))
            throw std::invalid_argument("GGUF direct MMQ requires complete tiles without a K over-read");
        mmq_x_tab table{};
        for(int expert=0;expert<p.experts;++expert) {
            if(!p.weight_addresses[expert])throw std::invalid_argument("GGUF direct MMQ has a null matrix");
            table.p[expert]=static_cast<const char*>(p.weight_addresses[expert]);
        }
        switch(type) {
#define TYLOINFER_MMQ_TAB(Type) case GGML_TYPE_##Type:mul_mat_q_tab_case<GGML_TYPE_##Type>(impl_->context,args,table,stream);break
            TYLOINFER_MMQ_TAB(Q2_0);TYLOINFER_MMQ_TAB(IQ2_XXS);TYLOINFER_MMQ_TAB(IQ2_XS);TYLOINFER_MMQ_TAB(IQ2_S);
            TYLOINFER_MMQ_TAB(IQ3_XXS);TYLOINFER_MMQ_TAB(IQ3_S);TYLOINFER_MMQ_TAB(IQ4_NL);TYLOINFER_MMQ_TAB(IQ4_XS);TYLOINFER_MMQ_TAB(Q8_0);
            TYLOINFER_MMQ_TAB(Q4_K);TYLOINFER_MMQ_TAB(Q5_K);TYLOINFER_MMQ_TAB(Q6_K);TYLOINFER_MMQ_TAB(Q5_1);
#undef TYLOINFER_MMQ_TAB
            default:throw std::invalid_argument("unsupported direct MMQ format");
        }
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<std::uint64_t> reported{0};
            const auto bit=std::uint64_t(1)<<unsigned(type);
            if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))
                std::fprintf(stderr,"native_gguf_prefill_inplace_mmq actual_match=1 format=%d rows=%lld K=%lld\n",int(type),
                    static_cast<long long>(p.rows),static_cast<long long>(p.columns));
        }
        check();return;
    }
    switch(type) {
#define TYLOINFER_MMQ_CASE(Type) case GGML_TYPE_##Type:mul_mat_q_case<GGML_TYPE_##Type>(impl_->context,args,stream);break
        TYLOINFER_MMQ_CASE(Q2_0);TYLOINFER_MMQ_CASE(IQ2_XXS);TYLOINFER_MMQ_CASE(IQ2_XS);TYLOINFER_MMQ_CASE(IQ2_S);
        TYLOINFER_MMQ_CASE(IQ3_XXS);TYLOINFER_MMQ_CASE(IQ3_S);TYLOINFER_MMQ_CASE(IQ4_NL);TYLOINFER_MMQ_CASE(IQ4_XS);TYLOINFER_MMQ_CASE(Q8_0);
        TYLOINFER_MMQ_CASE(Q4_K);TYLOINFER_MMQ_CASE(Q5_K);TYLOINFER_MMQ_CASE(Q6_K);TYLOINFER_MMQ_CASE(Q5_1);
#undef TYLOINFER_MMQ_CASE
        default:throw std::invalid_argument("GGUF MMQ format is unsupported");
    }
    check();
}
void gather(const Copy (&copies)[3],int experts,cudaStream_t stream) {
    if(experts<1 || experts>group_capacity)throw std::invalid_argument("GGUF MMQ group is too large");
    // Cold weights already occupy their final group positions; direct MMQ
    // also skips resident copies. Do not launch thousands of empty CTAs for
    // those ranges. Guards are independent small zero-fill ranges.
    GatherPacket packet{};std::uint64_t blocks=0;
    const auto append=[&](const void* source,void* destination,std::uint64_t bytes) {
        if(!bytes || source==destination)return;
        const auto count=bytes/4096+(bytes%4096!=0);
        if(count>INT_MAX || blocks>INT_MAX-count)throw std::overflow_error("GGUF MMQ copy grid overflows");
        packet.ranges[packet.count++]={static_cast<const std::uint8_t*>(source),static_cast<std::uint8_t*>(destination),
            bytes,std::uint32_t(blocks),std::uint32_t(count)};
        blocks+=count;
    };
    for(int role=0;role<3;++role) {
        const auto& job=copies[role];
        for(int expert=0;expert<experts;++expert)append(job.source[expert],
            static_cast<std::uint8_t*>(job.destination)+expert*job.stride+job.offset,job.bytes);
        append(nullptr,static_cast<std::uint8_t*>(job.destination)+experts*job.stride,job.guard_bytes);
    }
    if(!packet.count)return;
    gather_ranges<<<unsigned(blocks),256,0,stream>>>(packet);
    check();
}
void swiglu(const float* gate_up,float* hidden,std::int64_t tokens,std::int64_t width,float limit,cudaStream_t stream) {
    const auto count=tokens*width;
    activate<<<unsigned((count+255)/256),256,0,stream>>>(gate_up,hidden,count,width,limit);check();
}
void scatter(const float* source,void* pairs,const std::int32_t* ids,std::int64_t tokens,std::int64_t width,cudaStream_t stream) {
    const auto count=tokens*width;
    scatter_rows<<<unsigned((count+255)/256),256,0,stream>>>(source,static_cast<half*>(pairs),ids,count,width);check();
}
}
