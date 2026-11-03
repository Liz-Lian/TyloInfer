#pragma once
// Keep Q2_K storage unchanged. One M=1 fragment covers a whole Q8_1 block:
// its two 16-coordinate subscales are combined in integer arithmetic before
// applying the common activation scale. This reduces scale conversions and
// floating-point work versus four strided 4-coordinate groups. Integer sums
// are exact; only FP32 reduction grouping changes. M>1 retains PreparedDot.
template<>struct M1PreparedDot<GGML_TYPE_Q2_K>:Dot<GGML_TYPE_Q2_K> {
    static constexpr int qk=256,parts=8,step=1;
    struct Weight {int codes[8];unsigned scales;float2 dm;int shift;};
    struct Activation {int codes[8];int lo_sum,hi_sum;float scale;};
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const block_q2_K*>(data)+block;Weight w;
        const auto* payload=reinterpret_cast<const int*>(b->qs)+8*(part/4);
        #pragma unroll
        for(int j=0;j<8;++j)w.codes[j]=__ldcs(payload+j);
        w.shift=2*(part%4);w.scales=__ldcs(reinterpret_cast<const uint16_t*>(b->scales)+part);
        w.dm=__half22float2(__ldcs(&b->dm));return w;
    }
    __device__ static Weight load_streaming(const void* data,int block,int part){return load(data,block,part);}
    __device__ static Activation load_activation(const block_q8_1* x,int part) {
        Activation a;const auto& v=x[part];a.scale=__low2float(v.ds);a.lo_sum=0;a.hi_sum=0;
        #pragma unroll
        for(int j=0;j<8;++j){a.codes[j]=get_int_b4(v.qs,j);if(j<4)a.lo_sum=ggml_cuda_dp4a(0x01010101,a.codes[j],a.lo_sum);else a.hi_sum=ggml_cuda_dp4a(0x01010101,a.codes[j],a.hi_sum);}
        return a;
    }
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        int lo=0,hi=0;
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int codes=(w.codes[j]>>w.shift)&0x03030303;
            if(j<4)lo=ggml_cuda_dp4a(codes,a.codes[j],lo);else hi=ggml_cuda_dp4a(codes,a.codes[j],hi);
        }
        const int dot=lo*(w.scales&15)+hi*((w.scales>>8)&15);
        const int minimum=a.lo_sum*((w.scales>>4)&15)+a.hi_sum*(w.scales>>12);
        return w.dm.x*(a.scale*dot)-w.dm.y*(a.scale*minimum);
    }
};
