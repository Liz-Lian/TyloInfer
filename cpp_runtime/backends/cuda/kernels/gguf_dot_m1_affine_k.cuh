#pragma once
// Cover each whole Q8 block before scaling; original GGUF bytes are unchanged.
// M=1 and high-M Q6 windows share this decoder. These loads read global weights.
template<int Format>struct M1PreparedAffineK {
    static constexpr int qk=256,parts=8,step=1;
    struct Weight {int codes[8];int scale0,scale1;float2 dm;};
    struct Activation {int codes[8];int lo_sum,hi_sum;float scale;};
    __device__ static Weight load(const void* data,int block,int part) {
        Weight w;
        if constexpr(Format==GGML_TYPE_Q6_K) {
            const auto* b=static_cast<const block_q6_K*>(data)+block;
            // The 210-byte Q6_K block preserves halfword, not u32 alignment.
            const auto* low=b->ql+64*(part/4)+32*(part&1);
            const auto* high=b->qh+32*(part/4);
            #pragma unroll
            for(int j=0;j<8;++j) {
                const unsigned code=((unsigned(iq_weight_word<true>(low,j))>>(4*((part/2)&1)))&0x0f0f0f0fu)|(((unsigned(iq_weight_word<true>(high,j))>>(2*(part&3)))<<4)&0x30303030u);
                w.codes[j]=int((code+0x60606060u)^0x80808080u);
            }
            w.scale0=__ldcs(b->scales+2*part);w.scale1=__ldcs(b->scales+2*part+1);
            w.dm=make_float2(__half2float(__ldcs(&b->d)),0.f);
        }else {
            using Block=std::conditional_t<Format==GGML_TYPE_Q4_K,block_q4_K,block_q5_K>;
            const auto* b=static_cast<const Block*>(data)+block;
            const auto* low=reinterpret_cast<const unsigned*>(b->qs+32*(part/2));
            #pragma unroll
            for(int j=0;j<8;++j) {
                unsigned code=(__ldcs(low+j)>>(4*(part&1)))&0x0f0f0f0fu;
                if constexpr(Format==GGML_TYPE_Q5_K)code|=((__ldcs(reinterpret_cast<const unsigned*>(b->qh)+j)>>part)<<4)&0x10101010u;
                w.codes[j]=int(code);
            }
            if(part<4){w.scale0=__ldcs(b->scales+part)&63;w.scale1=__ldcs(b->scales+part+4)&63;}
            else {w.scale0=(__ldcs(b->scales+part+4)&15)|((__ldcs(b->scales+part-4)>>6)<<4);
                  w.scale1=(__ldcs(b->scales+part+4)>>4)|((__ldcs(b->scales+part)>>6)<<4);}
            w.dm=__half22float2(__ldcs(&b->dm));
        }
        return w;
    }
    __device__ static Weight load_streaming(const void* data,int block,int part){return load(data,block,part);}
    __device__ static Activation load_activation(const block_q8_1* x,int part) {
        Activation a;const auto& v=x[part];a.scale=__low2float(v.ds);a.lo_sum=0;a.hi_sum=0;
        #pragma unroll
        for(int j=0;j<8;++j){a.codes[j]=get_int_b4(v.qs,j);if constexpr(Format!=GGML_TYPE_Q6_K){if(j<4)a.lo_sum=ggml_cuda_dp4a(0x01010101,a.codes[j],a.lo_sum);else a.hi_sum=ggml_cuda_dp4a(0x01010101,a.codes[j],a.hi_sum);}}
        return a;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        int lo=0,hi=0;
        #pragma unroll
        for(int j=0;j<8;++j){if(j<4)lo=ggml_cuda_dp4a(w.codes[j],a.codes[j],lo);else hi=ggml_cuda_dp4a(w.codes[j],a.codes[j],hi);}
        if constexpr(Format==GGML_TYPE_Q6_K)return w.dm.x*(a.scale*(lo*w.scale0+hi*w.scale1));
        else return w.dm.x*(a.scale*((lo+hi)*w.scale0))-w.dm.y*(a.scale*((a.lo_sum+a.hi_sum)*w.scale1));
    }
};
template<>struct M1PreparedDot<GGML_TYPE_Q4_K>:M1PreparedAffineK<GGML_TYPE_Q4_K>{};
template<>struct M1PreparedDot<GGML_TYPE_Q5_K>:M1PreparedAffineK<GGML_TYPE_Q5_K>{};
template<>struct M1PreparedDot<GGML_TYPE_Q6_K>:M1PreparedAffineK<GGML_TYPE_Q6_K>{};
