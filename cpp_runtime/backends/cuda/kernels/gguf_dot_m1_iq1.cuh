#pragma once
// Ordinary M=1 IQ1 dots retain the original compressed weights and reduction
// tree. M>1 continues to use PreparedDot and its existing scheduling.
template<int Format>struct M1PreparedDot:PreparedDot<Format>{};

template<bool Medium,bool Shared=false>struct PreparedIq1M1:Dot<Medium?GGML_TYPE_IQ1_M:GGML_TYPE_IQ1_S> {
    // A standalone projection can stage one compact codebook for all of
    // its rows. This is CTA scratch, unrelated to model weight storage.
    static constexpr unsigned shared_bytes=Medium?8192:4096;
    __device__ static void stage(int tid,int threads) {
        extern __shared__ __align__(16) unsigned char iq1_dot_cache[];
        const auto* source=Medium?reinterpret_cast<const uint4*>(iq1_grid_selectors):
            reinterpret_cast<const uint4*>(iq1_grid_2bit);
        for(unsigned i=unsigned(tid);i<shared_bytes/sizeof(uint4);i+=unsigned(threads))
            reinterpret_cast<uint4*>(iq1_dot_cache)[i]=__ldg(source+i);
        __syncthreads();
    }
    struct Weight {unsigned codes,high,scales;float scale;};
    struct Activation {int codes[8];float2 ds;};
    template<bool Streaming,class T>
    __device__ __forceinline__ static T read(const T* p) {
        if constexpr(Streaming)return __ldcs(p);else return *p;
    }
    template<bool Streaming>
    __device__ __forceinline__ static Weight load_mode(const void* data,int block,int part) {
        Weight w;
        if constexpr(Medium) {
            const auto* b=static_cast<const block_iq1_m*>(data)+block;
            w.codes=read<Streaming>(reinterpret_cast<const std::uint32_t*>(b->qs)+part);
            w.high=read<Streaming>(reinterpret_cast<const std::uint16_t*>(b->qh)+part);
            // IQ1_M already reads its codes as aligned 32-bit words. The
            // 56-byte block and scale offset 48 preserve that alignment.
            const auto* scales=reinterpret_cast<const std::uint32_t*>(b->scales);
            const unsigned sc0=read<Streaming>(scales),sc1=read<Streaming>(scales+1);
            const unsigned bits=((sc0>>12)&15)|((sc0>>24)&0xf0)|((sc1>>4)&0xf00)|((sc1>>16)&0xf000);
            w.scale=__half2float(__ushort_as_half(bits));
            w.scales=((part&4)?sc1:sc0)>>((part&2)*8+6*(part&1));
        }else {
            const auto* b=static_cast<const block_iq1_s*>(data)+block;
            const auto* codes=reinterpret_cast<const std::uint16_t*>(b->qs)+2*part;
            w.codes=unsigned(read<Streaming>(codes))|(unsigned(read<Streaming>(codes+1))<<16);
            w.high=read<Streaming>(b->qh+part);w.scales=0;
            const auto bits=read<Streaming>(reinterpret_cast<const std::uint16_t*>(&b->d));
            w.scale=__half2float(__ushort_as_half(bits))*(((w.high>>11)&14)+1);
        }
        return w;
    }
    __device__ __forceinline__ static Weight load(const void* data,int block,int part) {
        return load_mode<false>(data,block,part);
    }
    __device__ __forceinline__ static Weight load_streaming(const void* data,int block,int part) {
        return load_mode<true>(data,block,part);
    }
    __device__ __forceinline__ static void prepare_activation(Activation& a) {
        if constexpr(!Medium) {
            // Reuse the even/odd coordinate permutation across output rows.
            // The original Q8 packet and compressed weight bytes stay intact.
            #pragma unroll
            for(int j=0;j<8;j+=2) {
                const int low=a.codes[j],high=a.codes[j+1];
                a.codes[j]=__byte_perm(low,high,0x6420);
                a.codes[j+1]=__byte_perm(low,high,0x7531);
            }
        }
    }
    __device__ __forceinline__ static Activation load_activation(const block_q8_1* x,int part) {
        Activation a;a.ds=__half22float2(x[part].ds);
        #pragma unroll
        for(int j=0;j<8;++j)a.codes[j]=get_int_b4(x[part].qs,j);
        prepare_activation(a);
        return a;
    }
    __device__ __forceinline__ static int2 decode_rank_planes(unsigned index) {
        // CUDA masks the neighboring rank's selector bit 3; identical source
        // words make selector bit 2 irrelevant. Raw PRMT cannot do that mask.
        extern __shared__ __align__(16) unsigned char iq1_dot_cache[];
        const unsigned packed=Shared?reinterpret_cast<const uint16_t*>(iq1_dot_cache)[index]:
            __ldg(iq1_grid_2bit+index);
        return make_int2(__byte_perm(0x00020100u,0x00020100u,packed),
            __byte_perm(0x00020100u,0x00020100u,packed>>2));
    }
    __device__ __forceinline__ static int2 decode_grid(unsigned index,bool negative) {
        if constexpr(Medium) {
            extern __shared__ __align__(16) unsigned char iq1_dot_cache[];
            unsigned selector=Shared?reinterpret_cast<const uint32_t*>(iq1_dot_cache)[index]:
                __ldg(iq1_grid_selectors+index);
            selector|=negative?0x44444444u:0u;
            return make_int2(iq_grid_byte_perm<true>(0x000901f9u,0x0007fff7u,selector),
                iq_grid_byte_perm<true>(0x000901f9u,0x0007fff7u,selector>>16));
        }else {
            const auto planes=decode_rank_planes(index);
            return make_int2(__byte_perm(planes.x,planes.y,0x5140),
                __byte_perm(planes.x,planes.y,0x7362));
        }
    }

    __device__ __forceinline__ static float apply_activation(const Weight& w,const Activation& a) {
        int sums[Medium?2:1]{};
        #pragma unroll
        for(int j=0;j<4;++j) {
            const unsigned high=w.high>>((Medium?4:3)*j);
            const unsigned index=((w.codes>>(8*j))&255)|((high&7)<<8);
            const auto codes=[&] {
                if constexpr(Medium)return decode_grid(index,(high&8)!=0);
                else return decode_rank_planes(index);
            }();const int slot=Medium?j/2:0;
            sums[slot]=ggml_cuda_dp4a(codes.x,a.codes[2*j],sums[slot]);
            sums[slot]=ggml_cuda_dp4a(codes.y,a.codes[2*j+1],sums[slot]);
        }
        if constexpr(Medium) {
            // The affine values are exactly {-9,-7,-1,1,7,9}/8. At 32
            // int8 coordinates their scaled sums remain exact in FP32.
            const int sum=sums[0]*(2*(w.scales&7)+1)+sums[1]*(2*((w.scales>>3)&7)+1);
            return (w.scale*a.ds.x)*(float(sum)*.125f);
        }else {
            // Preserve IQ1_S's independently FP16-rounded activation sum.
            const float delta=-1.f+IQ1S_DELTA-(w.high&0x8000)*(2.f*IQ1S_DELTA/0x8000);
            return w.scale*(a.ds.x*sums[0]+a.ds.y*delta);
        }
    }
};
template<>struct M1PreparedDot<GGML_TYPE_IQ1_S>:PreparedIq1M1<false>{};
template<>struct M1PreparedDot<GGML_TYPE_IQ1_M>:PreparedIq1M1<true>{};

// Other formats retain their existing prepared dot and zero shared scratch.
template<int Format>struct SharedM1PreparedDot:M1PreparedDot<Format>{};
template<>struct SharedM1PreparedDot<GGML_TYPE_IQ1_S>:PreparedIq1M1<false,true>{};
template<>struct SharedM1PreparedDot<GGML_TYPE_IQ1_M>:PreparedIq1M1<true,true>{};

// Explicit specializations also keep CUDA 12.4's dependent requires paths
// independent of conditional type aliases.
template<int Format,bool Shared>struct M1ProjectionDot:M1PreparedDot<Format>{};
template<int Format>struct M1ProjectionDot<Format,true>:SharedM1PreparedDot<Format>{};
