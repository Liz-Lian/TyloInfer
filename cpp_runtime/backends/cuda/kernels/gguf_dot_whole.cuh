#pragma once
// Full 32-value activation groups share integer dots and normalization
// across the token window. Weight storage remains canonical GGUF.
struct WholeQ8Activation {
    struct Activation {int codes[8];float scale;};
    __device__ static Activation load_activation(const block_q8_1* input,int part) {
        Activation a;a.scale=__low2float(input[part].ds);
        #pragma unroll
        for(int j=0;j<8;++j)a.codes[j]=get_int_b4(input[part].qs,j);
        return a;
    }
};
struct WholeQ8Prepared : WholeQ8Activation {
    static constexpr int qk=32,parts=1,step=1;
    struct Weight {int codes[8];float scale;};
    __device__ static Weight load(const void* data,int block,int) {
        const auto* b=static_cast<const block_q8_0*>(data)+block;
        Weight w;w.scale=__half2float(b->d);
        #pragma unroll
        for(int j=0;j<8;++j)w.codes[j]=get_int_b2(b->qs,j);
        return w;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        int sum=0;
        #pragma unroll
        for(int j=0;j<8;++j)sum=ggml_cuda_dp4a(w.codes[j],a.codes[j],sum);
        return (w.scale*a.scale)*sum;
    }
};
struct WholeQ3Prepared : WholeQ8Activation {
    static constexpr int qk=256,parts=8,step=1;
    struct Weight {int codes[8];int lo_scale,hi_scale;float scale;};
    __device__ static Weight load(const void* data,int block,int group) {
        const auto* b=static_cast<const block_q3_K*>(data)+block;
        Weight w;w.scale=__half2float(b->d);
        const auto* low=b->qs+32*(group/4);const auto* high=b->hmask;
        const unsigned skew=unsigned(reinterpret_cast<std::uintptr_t>(b)&2),shift=8*skew;
        const auto* low_words=reinterpret_cast<const unsigned*>(low+skew);
        const auto* high_words=reinterpret_cast<const unsigned*>(high+skew);
        unsigned previous_low=unsigned(__ldcs(reinterpret_cast<const std::uint16_t*>(low)))<<16;
        unsigned previous_high=unsigned(__ldcs(reinterpret_cast<const std::uint16_t*>(high)))<<16;
        // Forward aligned words end inside qs/scales in this same 110-byte
        // block, including the final block; external padding is unnecessary.
        #pragma unroll
        for(int j=0;j<8;++j) {
            const auto next_low=__ldcs(low_words+j),next_high=__ldcs(high_words+j);
            const unsigned l=(__funnelshift_l(previous_low,next_low,shift)>>(2*(group&3)))&0x03030303u;
            const unsigned h=(__funnelshift_l(previous_high,next_high,shift)>>group)&0x01010101u;
            previous_low=next_low;previous_high=next_high;
            w.codes[j]=int(((l|(h<<2))+0x7c7c7c7cu)^0x80808080u);
        }
        const unsigned low_scales=__ldg(reinterpret_cast<const std::uint16_t*>(b->scales)+(group&3))>>(4*(group/4));
        const unsigned high_scales=__ldg(reinterpret_cast<const std::uint16_t*>(b->scales)+4+(group&1))>>(2*(group/2));
        w.lo_scale=int((low_scales&15)|((high_scales&3)<<4))-32;
        w.hi_scale=int(((low_scales>>8)&15)|(((high_scales>>8)&3)<<4))-32;
        return w;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        int lo=0,hi=0;
        #pragma unroll
        for(int j=0;j<8;++j) {
            if(j<4)lo=ggml_cuda_dp4a(w.codes[j],a.codes[j],lo);
            else hi=ggml_cuda_dp4a(w.codes[j],a.codes[j],hi);
        }
        return (w.scale*a.scale)*float(lo*w.lo_scale+hi*w.hi_scale);
    }
};
struct WholeIq4NlPrepared : WholeQ8Activation {
    static constexpr int qk=32,parts=1,step=1;
    struct Weight {int2 codes[4];float scale;};
    __device__ static Weight load(const void* data,int block,int) {
        const auto* b=static_cast<const block_iq4_nl*>(data)+block;
        Weight w;w.scale=__half2float(__ldcs(&b->d));
        const auto* p=b->qs;
        const unsigned skew=unsigned(reinterpret_cast<std::uintptr_t>(p)&2),shift=8*skew;
        const auto* words=reinterpret_cast<const unsigned*>(p+skew);
        unsigned previous=unsigned(__ldcs(reinterpret_cast<const std::uint16_t*>(p)))<<16;
        #pragma unroll
        for(int j=0;j<4;++j) {
            unsigned next;
            // This last halfword keeps the unaligned reader within the
            // canonical 18-byte block, including an unpadded final row.
            if(j==3 && skew)next=__ldcs(reinterpret_cast<const std::uint16_t*>(p+14));
            else next=__ldcs(words+j);
            const auto code=__funnelshift_l(previous,next,shift);previous=next;
            w.codes[j]=get_int_from_table_16(code,kvalues_iq4nl);
        }
        return w;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        int sum=0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            sum=ggml_cuda_dp4a(w.codes[j].x,a.codes[j],sum);
            sum=ggml_cuda_dp4a(w.codes[j].y,a.codes[j+4],sum);
        }
        return (w.scale*a.scale)*sum;
    }
};
template<int Format>using WholeWindowDot=std::conditional_t<Format==GGML_TYPE_Q8_0,WholeQ8Prepared,
    std::conditional_t<Format==GGML_TYPE_Q3_K,WholeQ3Prepared,WholeIq4NlPrepared>>;
