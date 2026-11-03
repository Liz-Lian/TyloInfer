#pragma once
// Codebook decoding for lossless four-row compute storage.
// Reuse the ordinary M=1 decoder, including IQ1_S's even/odd activation
// registers and compact shared rank table. Storage only changes the loads.
template<bool Medium,int Offset>struct QuadIq1Dot:PreparedIq1M1<Medium,true> {
    static_assert(Offset==0);
};

template<int F,int Offset>struct QuadIqDot {
    static constexpr int GridCount=F==GGML_TYPE_IQ2_XXS||F==GGML_TYPE_IQ3_XXS?256:F==GGML_TYPE_IQ2_XS||F==GGML_TYPE_IQ3_S?512:1024;
    __device__ static unsigned grid_at(unsigned i){extern __shared__ unsigned quad_codebook[];return quad_codebook[Offset+i];}
    // IQ2 magnitudes use selector nibbles 0..2; their sign adds bit 2.
    // IQ3 magnitude selectors are 0..7. Bit 3 is always clear, so raw
    // PRMT matches byte_perm without a redundant runtime selector mask.
    __device__ static unsigned magnitude(unsigned i){if constexpr(F==GGML_TYPE_IQ3_S)return iq_grid_byte_perm<true>(0x07050301u,0x0f0d0b09u,grid_at(i));else return iq_grid_byte_perm<true>(0x1c140c04u,0x3e342c24u,grid_at(i));}
    static constexpr bool Two=F==GGML_TYPE_IQ2_XXS || F==GGML_TYPE_IQ2_XS || F==GGML_TYPE_IQ2_S;
    struct Raw {uint2 codes;unsigned signs,meta;};
    template<int J>__device__ static int2 decode(const Raw& w) {
        if constexpr(F==GGML_TYPE_IQ2_XXS)return PreparedIq2<F,true>::decode_selectors(grid_at((w.codes.x>>(8*J))&255),(w.signs>>(7*J))&127);
        else if constexpr(F==GGML_TYPE_IQ2_XS){const unsigned code=(J<2?w.codes.x:w.codes.y)>>(16*(J%2));return PreparedIq2<F,true>::decode_selectors(grid_at(code&511),(code>>9)&127);}
        else if constexpr(F==GGML_TYPE_IQ2_S){const unsigned code=((w.codes.x>>(8*J))&255)|(((w.meta>>24)<<(8-2*J))&768);return PreparedIq2<F,true>::decode_selectors(grid_at(code),(w.signs>>(8*J))&255);}
        else {
            const unsigned codes=(J<2?w.codes.x:w.codes.y)>>(16*(J%2));unsigned a,b,sign;
            if constexpr(F==GGML_TYPE_IQ3_XXS){sign=(w.signs>>(7*J))&127;sign|=(__popc(sign)&1u)<<7;a=magnitude(codes&255);b=magnitude((codes>>8)&255);const auto lo=iq_nibble_sign_mask(sign&15),hi=iq_nibble_sign_mask(sign>>4);return make_int2((a^lo)+(lo&0x01010101u),(b^hi)+(hi&0x01010101u));}
            else {sign=(w.signs>>(8*J))&255;const unsigned high=w.meta>>24;a=magnitude((codes&255)|((high<<(8-2*J))&256));b=magnitude(((codes>>8)&255)|((high<<(7-2*J))&256));return make_int2(a^(iq_nibble_sign_mask(sign&15)&0xfefefefeu),b^(iq_nibble_sign_mask(sign>>4)&0xfefefefeu));}
        }
    }
    __device__ static float finish(const Raw& w,int lo,int hi,float activation) {
        int sum;
        if constexpr(F==GGML_TYPE_IQ2_XXS)sum=(lo+hi)*int((w.signs>>27)|1)/8;
        else if constexpr(Two)sum=(lo*int((w.meta>>16)&15)+hi*int((w.meta>>20)&15)+(lo+hi)/2)/4;
        else if constexpr(F==GGML_TYPE_IQ3_XXS)sum=(int(2*(w.signs>>28)+1)*lo)/4;
        else sum=lo*int((w.meta>>16)&255);
        const float scale=__half2float(__ushort_as_half(w.meta&65535u));return (scale*activation)*sum;
    }

};
