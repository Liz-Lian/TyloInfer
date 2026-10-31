#include "gguf_cpu.h"
#include "gguf_cpu_simd.h"
#include "gguf_cpu_q2.h"
#include "ggml-quants.h"
#include <immintrin.h>
#include <cstring>
#include <stdexcept>
namespace mfq::cpu::gguf::simd {
static float half_value(const uint8_t* p) {
    uint16_t h; std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}
static float hsum(__m256 x) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x,1));
    a = _mm_add_ps(a, _mm_movehl_ps(a,a));
    return _mm_cvtss_f32(_mm_add_ss(a,_mm_movehdup_ps(a)));
}
// Q2_0 stores four consecutive codes in each byte, value=code-1. Permute
// the activation once, then decode the four planes without row-wise shuffles.

#if defined(_MSC_VER)
#define MFQ_Q2_NOINLINE __declspec(noinline)
#else
#define MFQ_Q2_NOINLINE __attribute__((noinline))
#endif
namespace plain {
#define MFQ_Q2_VNNI 0
#define MFQ_Q2_TARGET
#include "gguf_cpu_q2_rows.inl"
}
#if defined(STRATA_AVXVNNI) && STRATA_AVXVNNI
namespace vex {
#define MFQ_Q2_VNNI 1
#if defined(__GNUC__) || defined(__clang__)
#define MFQ_Q2_TARGET __attribute__((target("avxvnni")))
#else
#define MFQ_Q2_TARGET
#endif
#include "gguf_cpu_q2_rows.inl"
}
#endif
#if defined(TYLOINFER_GGUF_VNNI_EVEX)
namespace evex {
#define MFQ_Q2_VNNI 2
#if defined(__GNUC__) || defined(__clang__)
#define MFQ_Q2_TARGET __attribute__((target("avx512f,avx512vl,avx512vnni")))
#else
#define MFQ_Q2_TARGET
#endif
#include "gguf_cpu_q2_rows.inl"
}
#endif
#undef MFQ_Q2_NOINLINE
void q2_project(const Projection& b,const void* const* input,int tokens,float* const* output,int begin,int end,int encoding) {
#if defined(STRATA_AVXVNNI) && STRATA_AVXVNNI
    if(encoding==1 && vnni_encoding()==1){vex::project(b,input,tokens,output,begin,end);return;}
#endif
#if defined(TYLOINFER_GGUF_VNNI_EVEX)
    if(encoding==2 && vnni_encoding()==2){evex::project(b,input,tokens,output,begin,end);return;}
#endif
    plain::project(b,input,tokens,output,begin,end);
}
}
