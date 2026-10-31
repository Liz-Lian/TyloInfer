#pragma once
// Included after Dot inside gguf_dot.cuh. Keep the bundled ggml dot arithmetic
// and split its weight decode from activation work, as in Strata's multi-row
// traits (MIT, third_party/strata.LICENSE).
template<int Format> struct PreparedDot : Dot<Format> {
    struct Weight {const void* data;int block,part;};
    __device__ static Weight load(const void* data,int block,int part){return {data,block,part};}
    __device__ static float apply(const Weight& w,const block_q8_1* x){return Dot<Format>::apply(w.data,x,w.block,w.part);}
};

#define TYLOINFER_COMPACT_PREPARED(Name) \
template<> struct PreparedDot<GGML_TYPE_##Name> : Dot<GGML_TYPE_##Name> { using DecodedWeight = void; };
TYLOINFER_GGUF_COMPACT_FORMATS(TYLOINFER_COMPACT_PREPARED)
#undef TYLOINFER_COMPACT_PREPARED

// Affine K blocks follow Strata's Q4K/Q5K traits. Materialize the integer
// codes and scales before the token loop; a pointer-only preparation would
// leave their unpacking inside every activation dot.
template<int Format> struct PreparedAffineK : Dot<Format> {
    using DecodedWeight = void;
    using Block=std::conditional_t<Format==GGML_TYPE_Q5_K,block_q5_K,block_q4_K>;
    struct Weight {int codes[4];std::uint16_t aux[2];float2 scale;int offset,part;};
    struct Activation {int lo[2],hi[2],ones[2];float scale[2];};
    __device__ static Activation load_activation(const block_q8_1* x,int part) {
        Activation result;const int offset=2*((part/2)/4),word=(part/2)%4;
        #pragma unroll
        for(int i=0;i<2;++i) {
            const auto& block=x[offset+i];
            result.lo[i]=get_int_b4(block.qs,word);result.hi[i]=get_int_b4(block.qs,word+4);
            result.ones[i]=ggml_cuda_dp4a(0x01010101,result.hi[i],ggml_cuda_dp4a(0x01010101,result.lo[i],0));
            result.scale[i]=__low2float(block.ds);
        }
        return result;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        const auto* scales=reinterpret_cast<const std::uint8_t*>(w.aux);
        float sum=0,minimum=0;
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int dot=ggml_cuda_dp4a(w.codes[2*i+1],a.hi[i],ggml_cuda_dp4a(w.codes[2*i],a.lo[i],0));
            sum+=a.scale[i]*(dot*scales[i]);minimum+=a.scale[i]*(a.ones[i]*scales[i+2]);
        }
        return w.scale.x*sum-w.scale.y*minimum;
    }
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const Block*>(data)+block;
        Weight w;w.offset=2*((part/2)/4);w.part=(part/2)%4;
        const auto* ql=reinterpret_cast<const int*>(b->qs+16*w.offset+4*w.part);
        const int lo0=ql[0],lo1=ql[4];
        int hi0=0,hi1=0;
        if constexpr(Format==GGML_TYPE_Q5_K) {
            const auto* qh=reinterpret_cast<const int*>(b->qh+4*w.part);
            hi0=qh[0]>>w.offset;hi1=qh[4]>>w.offset;
        }
        #pragma unroll
        for(int i=0;i<2;++i) {
            w.codes[2*i]=(lo0>>(4*i))&0x0f0f0f0f;
            w.codes[2*i+1]=(lo1>>(4*i))&0x0f0f0f0f;
            if constexpr(Format==GGML_TYPE_Q5_K) {
                w.codes[2*i]|=((hi0>>i)<<4)&0x10101010;
                w.codes[2*i+1]|=((hi1>>i)<<4)&0x10101010;
            }
        }
        const auto* scales=reinterpret_cast<const std::uint16_t*>(b->scales);
        const int j=w.offset/2;
        if(j<2){w.aux[0]=scales[j]&0x3f3f;w.aux[1]=scales[j+2]&0x3f3f;}
        else {
            w.aux[0]=((scales[j+2]>>0)&0x0f0f)|((scales[j-2]&0xc0c0)>>2);
            w.aux[1]=((scales[j+2]>>4)&0x0f0f)|((scales[j]&0xc0c0)>>2);
        }
        w.scale=__half22float2(b->dm);return w;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        const auto* scales=reinterpret_cast<const std::uint8_t*>(w.aux);
        float sum=0,minimum=0;
        #pragma unroll
        for(int i=0;i<2;++i) {
            const auto& a=x[w.offset+i];
            const int lo=get_int_b4(a.qs,w.part),hi=get_int_b4(a.qs,w.part+4);
            const int dot=ggml_cuda_dp4a(w.codes[2*i+1],hi,ggml_cuda_dp4a(w.codes[2*i],lo,0));
            const int ones=ggml_cuda_dp4a(0x01010101,hi,ggml_cuda_dp4a(0x01010101,lo,0));
            const float scale=__low2float(a.ds);
            sum+=scale*(dot*scales[i]);minimum+=scale*(ones*scales[i+2]);
        }
        return w.scale.x*sum-w.scale.y*minimum;
    }
};
template<> struct PreparedDot<GGML_TYPE_Q4_K> : PreparedAffineK<GGML_TYPE_Q4_K> {};
template<> struct PreparedDot<GGML_TYPE_Q5_K> : PreparedAffineK<GGML_TYPE_Q5_K> {};

template<> struct PreparedDot<GGML_TYPE_Q6_K> : Dot<GGML_TYPE_Q6_K> {
    using DecodedWeight = void;
    struct Weight {int codes[2],scales[2];float scale;int offset,part;};
    struct Activation {int codes[2];float scale[2];};
    __device__ static Activation load_activation(const block_q8_1* input,int part) {
        Activation a;const int offset=4*(part/16)+(part%16)/8;
        #pragma unroll
        for(int i=0;i<2;++i) {
            const auto& x=input[offset+2*i];
            a.codes[i]=get_int_b4(x.qs,part%8);a.scale[i]=__low2float(x.ds);
        }
        return a;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        float sum=0;
        #pragma unroll
        for(int i=0;i<2;++i)sum+=a.scale[i]*(ggml_cuda_dp4a(w.codes[i],a.codes[i],0)*w.scales[i]);
        return w.scale*sum;
    }
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const block_q6_K*>(data)+block;
        Weight w;w.offset=4*(part/16)+(part%16)/8;w.part=part%8;
        const int scale_offset=8*(part/16)+(part%16)/4;
        const int lo=get_int_b2(b->ql,part);
        const int hi=get_int_b2(b->qh,8*(part/16)+part%8)>>(2*((part%16)/8));
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int code=((lo>>(4*i))&0x0f0f0f0f)|(((hi>>(4*i))<<4)&0x30303030);
            // Codes are 0..63 in each byte. Adding 0x60 cannot carry
            // between bytes; xor 0x80 gives exactly signed (code - 32).
            // Strata native_mmvq.cu's equivalent form avoids the expanded
            // saturating subtract; unsigned arithmetic defines the wrap.
            w.codes[i]=int((unsigned(code)+0x60606060u)^0x80808080u);
            w.scales[i]=b->scales[scale_offset+4*i];
        }
        w.scale=__half2float(b->d);return w;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        float sum=0;
        #pragma unroll
        for(int i=0;i<2;++i) {
            const auto& a=x[w.offset+2*i];
            sum+=__low2float(a.ds)*(ggml_cuda_dp4a(w.codes[i],get_int_b4(a.qs,w.part),0)*w.scales[i]);
        }
        return w.scale*sum;
    }
};
template<> struct PreparedDot<GGML_TYPE_Q8_0> : Dot<GGML_TYPE_Q8_0> {
    using DecodedWeight = void;
    struct Weight {int codes[VDR_Q8_0_Q8_1_MMVQ];float scale;int part;};
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const block_q8_0*>(data)+block;
        Weight w;w.scale=__half2float(b->d);w.part=part;
        #pragma unroll
        for(int i=0;i<VDR_Q8_0_Q8_1_MMVQ;++i)w.codes[i]=get_int_b2(b->qs,part+i);
        return w;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        int codes[VDR_Q8_0_Q8_1_MMVQ];
        #pragma unroll
        for(int i=0;i<VDR_Q8_0_Q8_1_MMVQ;++i)codes[i]=get_int_b4(x->qs,w.part+i);
        return vec_dot_q8_0_q8_1_impl<float,VDR_Q8_0_Q8_1_MMVQ>(w.codes,codes,w.scale,__low2float(x->ds));
    }
};
template<> struct PreparedDot<GGML_TYPE_Q2_0> : Dot<GGML_TYPE_Q2_0> {
    using DecodedWeight = void;
    struct Weight {int codes[8];float scale;int part;};
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const Block*>(data)+block;
        Weight w;w.scale=__half2float(b->d);w.part=part;
        const auto* values=reinterpret_cast<const std::int16_t*>(b->qs)+part*4;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int even=__byte_perm(0x020100FF,0x020100FF,values[j]);
            const int odd=__byte_perm(0x020100FF,0x020100FF,values[j]>>2);
            w.codes[j*2]=__byte_perm(even,odd,0x5140);w.codes[j*2+1]=__byte_perm(even,odd,0x7362);
        }
        return w;
    }
    __device__ static int input_block(const Weight& w){return w.part;}
    __device__ static float apply_u(const Weight& w,const int (&codes)[8],float scale) {
        int sum=0;
        #pragma unroll
        for(int j=0;j<8;++j)sum=ggml_cuda_dp4a(codes[j],w.codes[j],sum);
        return w.scale*scale*sum;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        int codes[8];
        #pragma unroll
        for(int j=0;j<8;++j)codes[j]=get_int_b4(x[w.part].qs,j);
        return apply_u(w,codes,__low2float(x[w.part].ds));
    }
};
template<> struct PreparedDot<GGML_TYPE_IQ4_NL> : Dot<GGML_TYPE_IQ4_NL> {
    using DecodedWeight = void;
    struct Weight {int2 codes[VDR_Q4_0_Q8_1_MMVQ];float scale;int part;};
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const block_iq4_nl*>(data)+block;
        Weight w;w.scale=__half2float(b->d);w.part=part;
        #pragma unroll
        for(int j=0;j<VDR_Q4_0_Q8_1_MMVQ;++j)w.codes[j]=get_int_from_table_16(get_int_b2(b->qs,part+j),kvalues_iq4nl);
        return w;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        const auto* codes=reinterpret_cast<const int*>(x->qs)+w.part;
        int sum=0;
        #pragma unroll
        for(int j=0;j<VDR_Q4_0_Q8_1_MMVQ;++j) {
            sum=ggml_cuda_dp4a(w.codes[j].x,codes[j],sum);sum=ggml_cuda_dp4a(w.codes[j].y,codes[j+4],sum);
        }
        return (w.scale*__low2float(x->ds))*sum;
    }
};
template<> struct PreparedDot<GGML_TYPE_IQ4_XS> : Dot<GGML_TYPE_IQ4_XS> {
    using DecodedWeight = void;
    struct Weight {int2 codes[4];float scale;int block,scale_code;};
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const block_iq4_xs*>(data)+block;
        Weight w;w.scale=__half2float(b->d);w.block=part/4;
        w.scale_code=(((b->scales_l[part/8]>>(part&4))&15)|(((b->scales_h>>(part/2))&3)<<4))-32;
        #pragma unroll
        for(int j=0;j<4;++j)w.codes[j]=get_int_from_table_16(get_int_b4(b->qs,part+j),kvalues_iq4nl);
        return w;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        int sum=0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            sum=ggml_cuda_dp4a(w.codes[j].x,get_int_b4(x[w.block].qs,j),sum);
            sum=ggml_cuda_dp4a(w.codes[j].y,get_int_b4(x[w.block].qs,j+4),sum);
        }
        sum*=w.scale_code;return (w.scale*__low2float(x[w.block].ds))*sum;
    }
};
template<int Format> struct PreparedIq3 : Dot<Format> {
    using DecodedWeight = void;
    struct Weight {int codes[8];float scale;int block,scale_code;};
    __device__ static Weight load(const void* data,int block,int part) {
        return load_grid<false>(data,block,part,nullptr);
    }
    template<bool Staged>
    __device__ static Weight load_grid(const void* data,int block,int part,const uint32_t* grid) {
        Weight w;w.block=part/2;
        if constexpr(Format==GGML_TYPE_IQ3_XXS) {
            const auto* b=static_cast<const block_iq3_xxs*>(data)+block;
            const int2 packed=make_int2(get_int_b2(b->qs,part),get_int_b2(b->qs,part+1));
            const auto* codes=reinterpret_cast<const std::uint8_t*>(&packed);
            const auto aux=std::uint32_t(get_int_b2(b->qs,QK_K/16+part/2));
            w.scale=__half2float(b->d);w.scale_code=aux>>28;
            #pragma unroll
            for(int j=0;j<8;j+=2) {
                if constexpr(Staged) {
                    const auto sign=(aux>>(7*j/2))&127u;
                    const auto lo=grid[256+sign],hi=grid[384+sign];
                    w.codes[j]=(grid[codes[j]]^lo)+(lo&0x01010101u);
                    w.codes[j+1]=(grid[codes[j+1]]^hi)+(hi&0x01010101u);
                }else {
                    const auto signs=unpack_ksigns(aux>>(7*j/2));
                    const int lo=__vcmpne4(signs&0x08040201,0),hi=__vcmpne4(signs&0x80402010,0);
                    w.codes[j]=__vsub4(iq3xxs_grid[codes[j]]^lo,lo);
                    w.codes[j+1]=__vsub4(iq3xxs_grid[codes[j+1]]^hi,hi);
                }
            }
        } else {
            const auto* b=static_cast<const block_iq3_s*>(data)+block;
            const int2 packed=make_int2(get_int_b2(b->qs,part),get_int_b2(b->qs,part+1));
            const auto* codes=reinterpret_cast<const std::uint8_t*>(&packed);
            const int qh=b->qh[part/2],sign_word=get_int_b2(b->signs,part/2);
            const auto* signs=reinterpret_cast<const std::uint8_t*>(&sign_word);
            w.scale=__half2float(b->d);w.scale_code=1+2*((b->scales[part/4]>>((part<<1)&4))&15);
            #pragma unroll
            for(int j=0;j<8;j+=2) {
                const int low_code=codes[j]|((qh<<(8-j))&256),high_code=codes[j+1]|((qh<<(7-j))&256);
                if constexpr(Staged) {
                    const auto lo=grid[512+signs[j/2]],hi=grid[768+signs[j/2]];
                    w.codes[j]=(grid[low_code]^lo)+(lo&0x01010101u);
                    w.codes[j+1]=(grid[high_code]^hi)+(hi&0x01010101u);
                }else {
                    const int lo=__vcmpne4(((signs[j/2]&3)<<7)|((signs[j/2]&12)<<21),0);
                    const int hi=__vcmpne4(((signs[j/2]&48)<<3)|((signs[j/2]&192)<<17),0);
                    w.codes[j]=__vsub4(iq3s_grid[low_code]^lo,lo);
                    w.codes[j+1]=__vsub4(iq3s_grid[high_code]^hi,hi);
                }
            }
        }
        return w;
    }
    __device__ static int input_block(const Weight& w){return w.block;}
    __device__ static float apply_u(const Weight& w,const int (&codes)[8],float scale) {
        int sum=0;
        #pragma unroll
        for(int j=0;j<8;++j)sum=ggml_cuda_dp4a(w.codes[j],codes[j],sum);
        if constexpr(Format==GGML_TYPE_IQ3_XXS)sum=(w.scale_code*sum+sum/2)/2;
        else sum*=w.scale_code;
        return (w.scale*scale)*sum;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        int codes[8];
        #pragma unroll
        for(int j=0;j<8;++j)codes[j]=get_int_b4(x[w.block].qs,j);
        return apply_u(w,codes,__low2float(x[w.block].ds));
    }
};
template<> struct PreparedDot<GGML_TYPE_IQ3_XXS> : PreparedIq3<GGML_TYPE_IQ3_XXS> {};
template<> struct PreparedDot<GGML_TYPE_IQ3_S> : PreparedIq3<GGML_TYPE_IQ3_S> {};

template<int Format> struct PreparedIq2 : Dot<Format> {
    using DecodedWeight = void;
    struct Weight {int codes[8];float scale;int block,low_scale,high_scale;};
    __device__ static Weight load(const void* data,int block,int part) {
        return load_grid<false>(data,block,part,nullptr);
    }
    template<bool Staged>
    __device__ static Weight load_grid(const void* data,int block,int part,const uint32_t* table) {
        constexpr int GridWords=Format==GGML_TYPE_IQ2_XXS?512:Format==GGML_TYPE_IQ2_XS?1024:2048;
        constexpr int Masks=Format==GGML_TYPE_IQ2_S?256:128;
        const auto decode=[&](uint64_t grid,int sign) {
            int lo,hi;
            if constexpr(Staged) {lo=table[GridWords+sign];hi=table[GridWords+Masks+sign];}
            else {
                const unsigned bits=Format==GGML_TYPE_IQ2_S?unsigned(sign):unsigned(unpack_ksigns(sign));
                if constexpr(Format==GGML_TYPE_IQ2_S) {
                    lo=__vcmpne4(((bits&3)<<7)|((bits&12)<<21),0);
                    hi=__vcmpne4(((bits&48)<<3)|((bits&192)<<17),0);
                }else {lo=__vcmpne4(bits&0x08040201,0);hi=__vcmpne4(bits&0x80402010,0);}
            }
            return make_int2(__vsub4(unsigned(grid)^lo,lo),__vsub4(unsigned(grid>>32)^hi,hi));
        };
        Weight w;w.block=part/2;w.high_scale=0;
        if constexpr(Format==GGML_TYPE_IQ2_XXS) {
            const auto* b=static_cast<const block_iq2_xxs*>(data)+block;
            const int packed=get_int_b2(b->qs,part);
            const auto* codes=reinterpret_cast<const std::uint8_t*>(&packed);
            const auto aux=std::uint32_t(get_int_b2(b->qs,part+1));
            w.scale=__half2float(b->d);w.low_scale=(aux>>27)|1;
            #pragma unroll
            for(int j=0;j<8;j+=2) {
                const int code=codes[j/2],sign=int((aux>>(7*j/2))&127);
                const auto grid=Staged?reinterpret_cast<const uint64_t*>(table)[code]:iq2xxs_grid[code];
                const auto values=decode(grid,sign);w.codes[j]=values.x;w.codes[j+1]=values.y;
            }
        } else if constexpr(Format==GGML_TYPE_IQ2_XS) {
            const auto* b=static_cast<const block_iq2_xs*>(data)+block;
            const int2 packed=make_int2(get_int_b2(b->qs,part),get_int_b2(b->qs,part+1));
            const auto* codes=reinterpret_cast<const std::uint16_t*>(&packed);
            w.scale=__half2float(b->d);w.low_scale=b->scales[part/2]&15;w.high_scale=b->scales[part/2]>>4;
            #pragma unroll
            for(int j=0;j<8;j+=2) {
                const int code=codes[j/2]&511,sign=codes[j/2]>>9;
                const auto grid=Staged?reinterpret_cast<const uint64_t*>(table)[code]:iq2xs_grid[code];
                const auto values=decode(grid,sign);w.codes[j]=values.x;w.codes[j+1]=values.y;
            }
        } else {
            const auto* b=static_cast<const block_iq2_s*>(data)+block;
            const int packed=get_int_b2(b->qs,part/2),sign_word=get_int_b2(b->qs,QK_K/32+part/2),qh=b->qh[part/2];
            const auto* codes=reinterpret_cast<const std::uint8_t*>(&packed);
            const auto* signs=reinterpret_cast<const std::uint8_t*>(&sign_word);
            w.scale=__half2float(b->d);w.low_scale=b->scales[part/2]&15;w.high_scale=b->scales[part/2]>>4;
            #pragma unroll
            for(int j=0;j<8;j+=2) {
                const int code=codes[j/2]|((qh<<(8-j))&768),sign=signs[j/2];
                const auto grid=Staged?reinterpret_cast<const uint64_t*>(table)[code]:iq2s_grid[code];
                const auto values=decode(grid,sign);w.codes[j]=values.x;w.codes[j+1]=values.y;
            }
        }
        return w;
    }
    __device__ static int input_block(const Weight& w){return w.block;}
    __device__ static float apply_u(const Weight& w,const int (&codes)[8],float scale) {
        int lo=0,hi=0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            lo=ggml_cuda_dp4a(w.codes[j],codes[j],lo);
            hi=ggml_cuda_dp4a(w.codes[j+4],codes[j+4],hi);
        }
        int sum;
        if constexpr(Format==GGML_TYPE_IQ2_XXS)sum=(lo+hi)*w.low_scale/8;
        else sum=(lo*w.low_scale+hi*w.high_scale+(lo+hi)/2)/4;
        return (w.scale*scale)*sum;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x) {
        int codes[8];
        #pragma unroll
        for(int j=0;j<8;++j)codes[j]=get_int_b4(x[w.block].qs,j);
        return apply_u(w,codes,__low2float(x[w.block].ds));
    }
};
template<> struct PreparedDot<GGML_TYPE_IQ2_XXS> : PreparedIq2<GGML_TYPE_IQ2_XXS> {};
template<> struct PreparedDot<GGML_TYPE_IQ2_XS> : PreparedIq2<GGML_TYPE_IQ2_XS> {};
template<> struct PreparedDot<GGML_TYPE_IQ2_S> : PreparedIq2<GGML_TYPE_IQ2_S> {};

// Strata's staged IQ3 grid and sign masks replace random global lookups and
// repeated byte negation. The result keeps the original integer dot exactly.
template<int Format> struct StagedPreparedDot : PreparedDot<Format> {
    static constexpr int words=0;
    __device__ static void stage(uint32_t*,int,int) {}
    __device__ static auto load(const void* data,int block,int part,const uint32_t*) {
        return PreparedDot<Format>::load(data,block,part);
    }
};
template<int Format> struct StagedIq3 : PreparedDot<Format> {
    static constexpr bool Xxs=Format==GGML_TYPE_IQ3_XXS;
    static constexpr int grid_words=Xxs?256:512,mask_words=Xxs?128:256,words=grid_words+2*mask_words;
    __device__ static void stage(uint32_t* grid,int tid,int threads) {
        for(int i=tid;i<grid_words;i+=threads) {
            if constexpr(Xxs)grid[i]=iq3xxs_grid[i];else grid[i]=iq3s_grid[i];
        }
        for(int i=tid;i<mask_words;i+=threads) {
            const auto signs=Xxs?uint32_t(i)|((__popc(uint32_t(i))&1u)<<7):uint32_t(i);
            uint32_t low=0,high=0;
            #pragma unroll
            for(int j=0;j<4;++j) {
                low|=((signs>>j)&1u)*(0xffu<<(8*j));
                high|=((signs>>(j+4))&1u)*(0xffu<<(8*j));
            }
            grid[grid_words+i]=low;grid[grid_words+mask_words+i]=high;
        }
        __syncthreads();
    }
    __device__ static auto load(const void* data,int block,int part,const uint32_t* grid) {
        return PreparedIq3<Format>::template load_grid<true>(data,block,part,grid);
    }
};
template<> struct StagedPreparedDot<GGML_TYPE_IQ3_XXS> : StagedIq3<GGML_TYPE_IQ3_XXS> {};
template<> struct StagedPreparedDot<GGML_TYPE_IQ3_S> : StagedIq3<GGML_TYPE_IQ3_S> {};

// IQ2 uses the same per-CTA lookup staging as IQ3. Stage the grid and sign
// masks once, then reuse decoded integers across all tokens of an expert.
template<int Format> struct StagedIq2 : PreparedDot<Format> {
    static constexpr int grid_words=Format==GGML_TYPE_IQ2_XXS?512:Format==GGML_TYPE_IQ2_XS?1024:2048;
    static constexpr int masks=Format==GGML_TYPE_IQ2_S?256:128,words=grid_words+2*masks;
    __device__ static void stage(uint32_t* grid,int tid,int threads) {
        for(int i=tid;i<grid_words;i+=threads) {
            if constexpr(Format==GGML_TYPE_IQ2_XXS)grid[i]=reinterpret_cast<const uint32_t*>(iq2xxs_grid)[i];
            else if constexpr(Format==GGML_TYPE_IQ2_XS)grid[i]=reinterpret_cast<const uint32_t*>(iq2xs_grid)[i];
            else grid[i]=reinterpret_cast<const uint32_t*>(iq2s_grid)[i];
        }
        for(int i=tid;i<masks;i+=threads) {
            const unsigned bits=Format==GGML_TYPE_IQ2_S?unsigned(i):unsigned(unpack_ksigns(i));
            if constexpr(Format==GGML_TYPE_IQ2_S) {
                grid[grid_words+i]=__vcmpne4(((bits&3)<<7)|((bits&12)<<21),0);
                grid[grid_words+masks+i]=__vcmpne4(((bits&48)<<3)|((bits&192)<<17),0);
            }else {
                grid[grid_words+i]=__vcmpne4(bits&0x08040201,0);
                grid[grid_words+masks+i]=__vcmpne4(bits&0x80402010,0);
            }
        }
        __syncthreads();
    }
    __device__ static auto load(const void* data,int block,int part,const uint32_t* grid) {
        return PreparedIq2<Format>::template load_grid<true>(data,block,part,grid);
    }
};
template<> struct StagedPreparedDot<GGML_TYPE_IQ2_XXS> : StagedIq2<GGML_TYPE_IQ2_XXS> {};
template<> struct StagedPreparedDot<GGML_TYPE_IQ2_XS> : StagedIq2<GGML_TYPE_IQ2_XS> {};
template<> struct StagedPreparedDot<GGML_TYPE_IQ2_S> : StagedIq2<GGML_TYPE_IQ2_S> {};
