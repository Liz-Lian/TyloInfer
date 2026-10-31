#include "gguf_cpu_simd.h"
#include "gguf_cpu_activation_impl.h"
#include "gguf_cpu_q2.h"
#include <immintrin.h>
#include <cmath>

namespace mfq::cpu::gguf::simd {
void hidden_quant_avx2(const float* input,void* packed,int columns,int type) {
    auto* q=static_cast<block_q8_0*>(packed);
    auto* prepared=type==42?reinterpret_cast<Q2Activation*>(static_cast<uint8_t*>(packed)+q2_activation_offset(columns)):nullptr;
    const auto abs_mask=_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const auto perm=_mm256_setr_epi32(0,4,1,5,2,6,3,7);
    const auto transpose=_mm256_setr_epi8(0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15,
        0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15);
    const auto round_away=[&](__m256 value) {
        const auto whole=_mm256_cvttps_epi32(value);
        const auto fraction=_mm256_sub_ps(value,_mm256_cvtepi32_ps(whole));
        const auto increment=_mm256_castps_si256(_mm256_cmp_ps(_mm256_and_ps(fraction,abs_mask),_mm256_set1_ps(.5f),_CMP_GE_OQ));
        const auto direction=_mm256_or_si256(_mm256_srai_epi32(_mm256_castps_si256(value),31),_mm256_set1_epi32(1));
        return _mm256_add_epi32(whole,_mm256_and_si256(increment,direction));
    };
    for(int group=0;group<(columns+127)/128;++group) {
        __m256i bytes[4];
        for(int part=0;part<4;++part) {
            const int block=group*4+part;
            bytes[part]=_mm256_setzero_si256();
            if(block<columns/32) {
                const auto* x=input+block*32;
                __m256 value[4],maximum=_mm256_setzero_ps();int finite=255;
                for(int i=0;i<4;++i) {
                    value[i]=_mm256_loadu_ps(x+8*i);
                    const auto magnitude=_mm256_and_ps(value[i],abs_mask);
                    finite&=_mm256_movemask_ps(_mm256_cmp_ps(magnitude,_mm256_set1_ps(INFINITY),_CMP_LT_OQ));
                    maximum=_mm256_max_ps(maximum,magnitude);
                }
                auto m=_mm_max_ps(_mm256_castps256_ps128(maximum),_mm256_extractf128_ps(maximum,1));
                m=_mm_max_ps(m,_mm_movehl_ps(m,m));m=_mm_max_ss(m,_mm_movehdup_ps(m));
                const float d=_mm_cvtss_f32(m)/127.f,id=d?1.f/d:0.f;
                if(finite!=255 || !std::isfinite(id)) {
                    quantize_row_q8_0_ref(x,q+block,32);
                    bytes[part]=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q[block].qs));
                } else {
                    q[block].d=ggml_fp16_t(_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(d),_MM_FROUND_TO_NEAREST_INT)));
                    __m256i integers[4];
                    for(int i=0;i<4;++i)integers[i]=round_away(_mm256_mul_ps(value[i],_mm256_set1_ps(id)));
                    const auto low=_mm256_packs_epi32(integers[0],integers[1]);
                    const auto high=_mm256_packs_epi32(integers[2],integers[3]);
                    bytes[part]=_mm256_permutevar8x32_epi32(_mm256_packs_epi16(low,high),perm);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(q[block].qs),bytes[part]);
                }
            }
            if(prepared) {
                const float scale=block<columns/32?_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(int(q[block].d)))):0.f;
                const auto pairs=_mm256_maddubs_epi16(_mm256_set1_epi8(1),bytes[part]);
                const auto sums=_mm256_madd_epi16(pairs,_mm256_set1_epi16(1));
                auto sum=_mm_add_epi32(_mm256_castsi256_si128(sums),_mm256_extracti128_si256(sums,1));
                sum=_mm_add_epi32(sum,_mm_srli_si128(sum,8));sum=_mm_add_epi32(sum,_mm_srli_si128(sum,4));
                prepared[group].scale[part]=scale;
                prepared[group].sum[part]=float(_mm_cvtsi128_si32(sum))*scale;
            }
        }
        if(prepared) {
            // Transpose four 32-byte Q8 blocks into four 32-byte planes. Each
            // plane retains exactly the old (i/16)*64+(i%16)*4+p indexing.
            for(auto& v:bytes)v=_mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v,transpose),perm);
            const auto lo01=_mm256_unpacklo_epi64(bytes[0],bytes[1]),lo23=_mm256_unpacklo_epi64(bytes[2],bytes[3]);
            const auto hi01=_mm256_unpackhi_epi64(bytes[0],bytes[1]),hi23=_mm256_unpackhi_epi64(bytes[2],bytes[3]);
            auto* out=reinterpret_cast<__m256i*>(prepared[group].planes);
            _mm256_storeu_si256(out,_mm256_permute2x128_si256(lo01,lo23,0x20));
            _mm256_storeu_si256(out+1,_mm256_permute2x128_si256(hi01,hi23,0x20));
            _mm256_storeu_si256(out+2,_mm256_permute2x128_si256(lo01,lo23,0x31));
            _mm256_storeu_si256(out+3,_mm256_permute2x128_si256(hi01,hi23,0x31));
        }
    }
}
void swiglu_fp16_avx2(const float* gate,const float* up,float* output,std::size_t count,float limit) {
    const auto round=[](__m256 value) {
        return _mm256_cvtph_ps(_mm256_cvtps_ph(value,_MM_FROUND_TO_NEAREST_INT));
    };
    const auto finite=[](__m256 value) {
        const auto magnitude=_mm256_andnot_ps(_mm256_set1_ps(-0.f),value);
        return _mm256_movemask_ps(_mm256_cmp_ps(magnitude,_mm256_set1_ps(INFINITY),_CMP_LT_OQ))==255;
    };
    const auto one=_mm256_set1_ps(1.f);
    std::size_t at=0;
    for(;at+8<=count;at+=8) {
        auto g=round(_mm256_loadu_ps(gate+at)),u=round(_mm256_loadu_ps(up+at));
        // Keep the scalar min/clamp semantics for exceptional values.
        // Ordinary rows do
        // eight half conversions per instruction instead of per value.
        if(!finite(g) || !finite(u)) {
            for(int i=0;i<8;++i)output[at+i]=detail::swiglu_scalar(gate[at+i],up[at+i],true,limit);
            continue;
        }
        if(limit>0) {
            // Put the original operand second to preserve signed-zero ties.
            g=_mm256_min_ps(_mm256_set1_ps(limit),g);
            u=_mm256_min_ps(_mm256_set1_ps(limit),_mm256_max_ps(_mm256_set1_ps(-limit),u));
        }
        alignas(32) float gates[8],exponents[8];
        _mm256_store_ps(gates,g);
        for(int i=0;i<8;++i)exponents[i]=std::exp(-gates[i]);
        auto sigmoid=_mm256_div_ps(one,_mm256_add_ps(one,_mm256_load_ps(exponents)));
        if(!(limit>0))sigmoid=round(sigmoid);
        auto value=_mm256_mul_ps(g,sigmoid);
        if(!(limit>0))value=round(value);
        value=_mm256_mul_ps(value,u);
        value=round(value);
        if(finite(value))_mm256_storeu_ps(output+at,value);
        else for(int i=0;i<8;++i)output[at+i]=detail::swiglu_scalar(gate[at+i],up[at+i],true,limit);
    }
    for(;at<count;++at)output[at]=detail::swiglu_scalar(gate[at],up[at],true,limit);
}
}
