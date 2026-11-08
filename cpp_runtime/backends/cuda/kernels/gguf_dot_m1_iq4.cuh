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
        const auto* codes=b->qs+part*4;
        if((reinterpret_cast<std::uintptr_t>(codes)&7u)==0) {
            // Each 136-byte block preserves eight-byte alignment. Retire
            // one compressed pair before loading the next; scalar views
            // still accept the original four-byte alignment.
            const auto* pairs=reinterpret_cast<const uint2*>(codes);
            const auto first=__ldcs(pairs);
            w.codes[0]=get_int_from_table_16(first.x,kvalues_iq4nl);
            w.codes[1]=get_int_from_table_16(first.y,kvalues_iq4nl);
            const auto second=__ldcs(pairs+1);
            w.codes[2]=get_int_from_table_16(second.x,kvalues_iq4nl);
            w.codes[3]=get_int_from_table_16(second.y,kvalues_iq4nl);
        }else {
            #pragma unroll
            for(int j=0;j<4;++j)
                w.codes[j]=get_int_from_table_16(__ldcs(reinterpret_cast<const int*>(codes)+j),kvalues_iq4nl);
        }
        return w;
    }
    template<bool Streaming,class T>
    __device__ static T load_window_word(const T* address) {
        if constexpr(Streaming)return __ldcs(address);
        else return __ldg(address);
    }
    template<bool Streaming>
    __device__ static Weight load_aligned_window(const void* data,int block,int part) {
        // The caller checks both the base and row stride. The 136-byte
        // block and four-word fragment preserve eight-byte alignment.
        // Checking each fragment instead duplicates the nonlinear decoder
        // in the hot loop; ordinary external four-byte views keep load().
        const auto* b=static_cast<const block_iq4_xs*>(data)+block;Weight w;w.block=part/4;
        w.scale=__half2float(__ushort_as_half(load_window_word<Streaming>(reinterpret_cast<const uint16_t*>(&b->d))));
        w.scale_code=(((load_window_word<Streaming>(b->scales_l+part/8)>>(part&4))&15)|
            ((load_window_word<Streaming>(&b->scales_h)>>(part/2)&3)<<4))-32;
        const auto* pairs=reinterpret_cast<const uint2*>(b->qs+part*4);
        const auto first=load_window_word<Streaming>(pairs),second=load_window_word<Streaming>(pairs+1);
        w.codes[0]=get_int_from_table_16(first.x,kvalues_iq4nl);
        w.codes[1]=get_int_from_table_16(first.y,kvalues_iq4nl);
        w.codes[2]=get_int_from_table_16(second.x,kvalues_iq4nl);
        w.codes[3]=get_int_from_table_16(second.y,kvalues_iq4nl);
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
