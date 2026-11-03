#pragma once
// Retain nonlinear IQ4_XS codes and share one Q8 activation across M=1 rows.
struct M1PreparedIq4Xs:PreparedDot<GGML_TYPE_IQ4_XS> {
    struct Activation {int codes[8];float scale;};
    __device__ static Activation load_activation(const block_q8_1* x,int part) {
        Activation a;const auto& b=x[part/4];a.scale=__low2float(b.ds);
        #pragma unroll
        for(int i=0;i<8;++i)a.codes[i]=get_int_b4(b.qs,i);
        return a;
    }
    __device__ static Weight load_streaming(const void* data,int block,int part) {
        const auto* b=static_cast<const block_iq4_xs*>(data)+block;Weight w;w.block=part/4;
        w.scale=__half2float(__ushort_as_half(__ldcs(reinterpret_cast<const uint16_t*>(&b->d))));
        w.scale_code=(((__ldcs(b->scales_l+part/8)>>(part&4))&15)|(((__ldcs(&b->scales_h)>>(part/2))&3)<<4))-32;
        #pragma unroll
        for(int j=0;j<4;++j)w.codes[j]=get_int_from_table_16(__ldcs(reinterpret_cast<const int*>(b->qs)+part+j),kvalues_iq4nl);
        return w;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        int sum=0;
        #pragma unroll
        for(int j=0;j<4;++j){sum=ggml_cuda_dp4a(w.codes[j].x,a.codes[j],sum);sum=ggml_cuda_dp4a(w.codes[j].y,a.codes[j+4],sum);}
        sum*=w.scale_code;return (w.scale*a.scale)*sum;
    }
};
template<>struct M1PreparedDot<GGML_TYPE_IQ4_XS>:M1PreparedIq4Xs{};
