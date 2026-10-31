// Four planes sum to at most 4*2*3*128 per int16 lane.
// VNNI int32 accumulation therefore preserves the nonsaturating AVX2 bits.
template<int M> MFQ_Q2_TARGET void q2_rows(const Projection& b, const void* const* acts,
                                    float* const* out, int begin, int end) {
    for(int r=begin;r<end;++r) {
        __m256 sums[M];float correction[M]{};
        const Q2Activation* a[M];
        for(int t=0;t<M;++t){sums[t]=_mm256_setzero_ps();a[t]=reinterpret_cast<const Q2Activation*>(
            static_cast<const std::uint8_t*>(acts[t])+q2_activation_offset(b.columns));}
        for(int block=0;block<(b.columns+127)/128;++block) {
            const auto* w=b.data+size_t(r)*b.row_bytes+block*36;
            const bool tail=block*128+64>=b.columns;
            const float d0=half_value(w),d1=tail?0.f:half_value(w+18);
            const auto lo=_mm_loadu_si128((const __m128i*)(w+2));
            const auto hi=tail?_mm_setzero_si128():_mm_loadu_si128((const __m128i*)(w+20));
            const auto codes=_mm256_set_m128i(hi,lo),mask=_mm256_set1_epi8(3);
            __m256i acc[M];for(int t=0;t<M;++t)acc[t]=_mm256_setzero_si256();
            for(int plane=0;plane<4;++plane) {
                const auto c=_mm256_and_si256(_mm256_srli_epi16(codes,2*plane),mask);
                for(int t=0;t<M;++t) {
                    const auto x=_mm256_loadu_si256((const __m256i*)(a[t][block].planes+plane*32));
#if MFQ_Q2_VNNI == 2
                    acc[t]=_mm256_dpbusd_epi32(acc[t],c,x);
#elif MFQ_Q2_VNNI == 1
                    acc[t]=_mm256_dpbusd_avx_epi32(acc[t],c,x);
#else
                    acc[t]=_mm256_add_epi16(acc[t],_mm256_maddubs_epi16(c,x));
#endif
                }
            }
            for(int t=0;t<M;++t) {
                const auto& q=a[t][block];
                const auto scale=_mm256_setr_ps(d0*q.scale[0],d0*q.scale[0],d0*q.scale[1],d0*q.scale[1],
                    d1*q.scale[2],d1*q.scale[2],d1*q.scale[3],d1*q.scale[3]);
#if MFQ_Q2_VNNI
                const auto dot=acc[t];
#else
                const auto dot=_mm256_madd_epi16(acc[t],_mm256_set1_epi16(1));
#endif
                sums[t]=_mm256_fmadd_ps(scale,_mm256_cvtepi32_ps(dot),sums[t]);
                correction[t]+=d0*(q.sum[0]+q.sum[1])+d1*(q.sum[2]+q.sum[3]);
            }
        }
        for(int t=0;t<M;++t)out[t][r]=hsum(sums[t])-correction[t];
    }
}

MFQ_Q2_TARGET MFQ_Q2_NOINLINE void project(const Projection& b,const void* const* input,int tokens,float* const* output,int begin,int end) {
    switch(tokens) {
    case 1:q2_rows<1>(b,input,output,begin,end);break;
    case 2:q2_rows<2>(b,input,output,begin,end);break;
    case 3:q2_rows<3>(b,input,output,begin,end);break;
    case 4:q2_rows<4>(b,input,output,begin,end);break;
    default:throw std::invalid_argument("CPU GGUF Q2_0 window exceeds four tokens");
    }
}
#undef MFQ_Q2_TARGET
#undef MFQ_Q2_VNNI
