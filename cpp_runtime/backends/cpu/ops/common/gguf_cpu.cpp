#include "gguf_cpu.h"
#include "gguf_cpu_activation.h"
#include "gguf_cpu_activation_impl.h"
#include "gguf_cpu_simd.h"
#include "gguf_cpu_q2.h"
#include "ggml-quants.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <atomic>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#if defined(TYLOINFER_GGUF_CPU_AVX2) && !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

namespace mfq::cpu::gguf {
namespace simd {
void q2_project(const Projection&,const void* const*,int,float* const*,int,int,int);
}
bool available() noexcept {
    static const bool result=[] {
#if defined(TYLOINFER_GGUF_CPU_AVX2) && defined(_MSC_VER)
        int r[4];__cpuid(r,0);if(r[0]<7)return false;
        __cpuidex(r,1,0);
        constexpr int required=(1<<12)|(1<<27)|(1<<28)|(1<<29); // FMA, OSXSAVE, AVX, F16C.
        if((r[2]&required)!=required || (_xgetbv(0)&6)!=6)return false;
        __cpuidex(r,7,0);return (r[1]&(1<<5))!=0;
#elif defined(TYLOINFER_GGUF_CPU_AVX2) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        return bool(__builtin_cpu_supports("avx2")) && bool(__builtin_cpu_supports("fma")) &&
            bool(__builtin_cpu_supports("f16c"));
#else
        return false;
#endif
    }();return result;
}
bool avx512_available() noexcept {
#if defined(TYLOINFER_GGUF_CPU_AVX512)
    static const bool result=[] {
        if(!available())return false;
#if defined(_MSC_VER)
        // F/DQ/CD/BW/VL cover the TU's /arch:AVX512 instruction set.
        // XCR0 must preserve opmask, upper ZMM and high ZMM registers too.
        if((_xgetbv(0)&0xe6)!=0xe6)return false;
        int r[4];__cpuidex(r,7,0);
        constexpr unsigned required=(1u<<16)|(1u<<17)|(1u<<28)|(1u<<30)|(1u<<31);
        return (unsigned(r[1])&required)==required;
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        return bool(__builtin_cpu_supports("avx512f")) && bool(__builtin_cpu_supports("avx512bw")) &&
            bool(__builtin_cpu_supports("avx512dq")) && bool(__builtin_cpu_supports("avx512vl"));
#else
        return false;
#endif
    }();return result;
#else
    return false;
#endif
}
int simd::vnni_encoding() noexcept {
    static const int result=[] {
#if defined(TYLOINFER_GGUF_CPU_AVX2) && (defined(STRATA_AVXVNNI) || defined(TYLOINFER_GGUF_VNNI_EVEX))
        if(!available())return 0;
        const auto cpuid=[](int leaf,int sub,int* r) {
#if defined(_MSC_VER)
            __cpuidex(r,leaf,sub);
#else
            __cpuid_count(leaf,sub,r[0],r[1],r[2],r[3]);
#endif
        };
        int r[4];cpuid(7,0,r);
        const int maximum=r[0];
#if defined(TYLOINFER_GGUF_VNNI_EVEX)
        const bool evex=(std::uint32_t(r[1])&((1u<<16)|(1u<<31)))==((1u<<16)|(1u<<31)) &&
            (r[2]&(1<<11)); // AVX512F, VL and VNNI: Intel CPUID leaf 7.0.
#endif
#if defined(STRATA_AVXVNNI) && STRATA_AVXVNNI
        if(maximum>=1) {
            cpuid(7,1,r);
            if(r[0]&(1<<4))return 1; // AVX-VNNI uses VEX and needs only YMM state.
        }
#endif
#if defined(TYLOINFER_GGUF_VNNI_EVEX)
        if(evex) {
            std::uint64_t state=0;
#if defined(_MSC_VER)
            state=_xgetbv(0);
#else
            std::uint32_t low,high;
            asm volatile("xgetbv":"=a"(low),"=d"(high):"c"(0));
            state=std::uint64_t(low)|(std::uint64_t(high)<<32);
#endif
            // EVEX256 still requires OS preservation of opmask/ZMM state.
            if((state&0xe6)==0xe6)return 2;
        }
#endif
#endif
        return 0;
    }();return result;
}
bool input_supported(int t) noexcept {return t==16 || t==17 || t==18 || t==21 || t==22 || t==23;}
void swiglu(const float* gate,const float* up,float* output,std::size_t count,bool half,float limit) {
#if defined(TYLOINFER_GGUF_CPU_AVX2)
    // The gain comes from batching the repeated half conversions. FP32 has
    // none and retains its shorter scalar library-expression path.
    if(half && available()){simd::swiglu_fp16_avx2(gate,up,output,count,limit);return;}
#endif
    for(std::size_t i=0;i<count;++i)output[i]=detail::swiglu_scalar(gate[i],up[i],half,limit);
}
bool down_supported(int t) noexcept {return t==20 || t==42;}
std::size_t input_bytes(int columns) {
    if(columns<=0 || columns%256)throw std::invalid_argument("CPU IQ rows require complete 256-value blocks");
    return std::size_t(columns/256)*sizeof(block_q8_K);
}
std::size_t hidden_bytes(int columns,int type) {
    if(columns<=0 || columns%32)throw std::invalid_argument("CPU down rows require complete 32-value blocks");
    if(type==42) {
        if(columns%64)throw std::invalid_argument("CPU Q2_0 rows require complete 64-value blocks");
        return simd::q2_activation_offset(columns)+std::size_t((columns+127)/128)*sizeof(simd::Q2Activation);
    }
    if(type!=20)throw std::invalid_argument("unsupported CPU down activation format");
    return std::size_t(columns/32)*sizeof(block_q8_0);
}
void quantize_input(const float* x,void* y,int n) {
    (void)input_bytes(n);
#if defined(TYLOINFER_GGUF_CPU_AVX2)
    if(available()){simd::q8k_quant_avx2(x,y,n);return;}
#endif
    quantize_row_q8_K_ref(x,static_cast<block_q8_K*>(y),n);
}
void quantize_hidden(const float* x,void* y,int n,int type) {
    (void)hidden_bytes(n,type);
#if defined(TYLOINFER_GGUF_CPU_AVX2)
    if(available()){simd::hidden_quant_avx2(x,y,n,type);return;}
#endif
    auto* q=static_cast<block_q8_0*>(y);quantize_row_q8_0_ref(x,q,n);
    if(type!=42)return;
    auto* prepared=reinterpret_cast<simd::Q2Activation*>(static_cast<std::uint8_t*>(y)+simd::q2_activation_offset(n));
    for(int block=0;block<(n+127)/128;++block) {
        auto& a=prepared[block];
        for(int half=0;half<4;++half) {
            const int index=block*4+half;
            a.scale[half]=index<n/32?ggml_fp16_to_fp32(q[index].d):0.f;
            int sum=0;if(index<n/32)for(int i=0;i<32;++i)sum+=q[index].qs[i];
            a.sum[half]=float(sum)*a.scale[half];
        }
        for(int plane=0;plane<4;++plane)for(int i=0;i<32;++i) {
            const int at=block*128+(i/16)*64+(i%16)*4+plane;
            a.planes[plane*32+i]=at<n?q[at/32].qs[at%32]:0;
        }
    }
}
ProjectionDispatch projection_dispatch() {
    const auto* isa=std::getenv("TYLOINFER_GGUF_CPU_ISA");
    const bool wide=isa && std::strcmp(isa,"avx512")==0 && avx512_available();
    const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
    const auto* vnni=std::getenv("TYLOINFER_GGUF_CPU_VNNI");
    const bool use_vnni=(!isa || std::strcmp(isa,"avx2")!=0) && (!vnni || vnni[0]!='0') && simd::vnni_encoding()!=0;
    return {wide,trace && trace[0]=='1',use_vnni};
}
void project(const Projection& p,const void* const* input,int tokens,float* const* output,int begin,int end) {
    project(p,input,tokens,output,begin,end,projection_dispatch());
}
void project(const Projection& p,const void* const* input,int tokens,float* const* output,int begin,int end,ProjectionDispatch dispatch) {
    if(!available())throw std::runtime_error("CPU GGUF rows require AVX2, FMA and F16C");
    if(!p.data || !input || !output || begin<0 || end<begin || end>p.rows || tokens<1 || tokens>4)
        throw std::invalid_argument("CPU GGUF projection geometry");
    const int variant=dispatch.vnni && simd::vnni_encoding()?simd::kIq256Vnni:0;
    if(input_supported(p.type)) {
        (void)input_bytes(p.columns);
#if defined(TYLOINFER_GGUF_CPU_AVX2)
        // Wider vectors did not consistently win in the complete host lane.
        // Keep explicit ISA comparison available without changing its default.
        const bool wide=p.type!=23 && dispatch.avx512 && avx512_available();
        if(dispatch.trace) {
            static std::atomic<unsigned> seen{0};const unsigned bit=wide?2:variant?4:1;
            if(!(seen.load(std::memory_order_relaxed)&bit) && !(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
                std::fprintf(stderr,"native_gguf_cpu_rows actual_match=1 isa=%s\n",wide?"avx512":variant?
                    (simd::vnni_encoding()==1?"avx_vnni":"avx512_vnni_256"):"avx2");
        }
#if defined(TYLOINFER_GGUF_CPU_AVX512)
        if(wide){simd::iq512_rows(p.type,p.data,p.row_bytes,p.columns,input,tokens,output,begin,end);return;}
#endif
        simd::iq256_rows_v(variant,p.type,p.data,p.row_bytes,p.columns,input,tokens,output,begin,end);return;
#endif
    } else if(down_supported(p.type)) {
        (void)hidden_bytes(p.columns,p.type);
        if(p.type==42 && p.columns%64)throw std::invalid_argument("CPU Q2_0 rows require complete 64-value blocks");
#if defined(TYLOINFER_GGUF_CPU_AVX2)
        if(dispatch.trace) {
            static std::atomic<unsigned> seen{0};
            const unsigned bit=(p.type==42?4u:1u)*(variant?2u:1u);
            if(!(seen.load(std::memory_order_relaxed)&bit) && !(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
                std::fprintf(stderr,"native_gguf_cpu_down actual_match=1 format=%d isa=%s\n",p.type,variant?
                    (simd::vnni_encoding()==1?"avx_vnni":"avx512_vnni_256"):"avx2");
        }
        if(p.type==20)simd::iq4nl256_down_rows_v(variant,p.data,p.row_bytes,p.columns,input,tokens,output,begin,end);
        else simd::q2_project(p,input,tokens,output,begin,end,variant?simd::vnni_encoding():0);
        return;
#endif
    }
    throw std::invalid_argument("unsupported CPU GGUF expert format");
}
}
