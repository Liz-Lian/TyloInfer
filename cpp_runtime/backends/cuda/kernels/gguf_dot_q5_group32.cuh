#pragma once
// A lane handles one complete 32-value affine group. Eight neighboring
// lanes gather a block's low/high codes together, retaining canonical GGUF
// bytes and amortizing FP32 normalization over the complete integer dot.
namespace mfq::cuda::gguf { namespace {
struct Q5Group32Prepared {
    static constexpr int qk=256,parts=8,step=1;
    struct Weight {int codes[8];int scale,minimum,group;float2 d;};
    struct Activation {int codes[8],ones;float d;};
    __device__ static Weight load(const void* data,int block,int group) {
        const auto* b=static_cast<const block_q5_K*>(data)+block;
        const unsigned mask=__activemask();
        const auto own=__ldg(reinterpret_cast<const uint4*>(b->qs+32*(group/2)+16*(group&1)));
        const uint4 peer=make_uint4(__shfl_xor_sync(mask,own.x,1,8),__shfl_xor_sync(mask,own.y,1,8),
            __shfl_xor_sync(mask,own.z,1,8),__shfl_xor_sync(mask,own.w,1,8));
        const auto a=(group&1)?peer:own,c=(group&1)?own:peer;
        const unsigned low[8]={a.x,a.y,a.z,a.w,c.x,c.y,c.z,c.w};
        const unsigned high=__ldg(reinterpret_cast<const unsigned*>(b->qh)+group);
        Weight w;w.group=group;
        #pragma unroll
        for(int j=0;j<8;++j) {
            const auto upper=__shfl_sync(mask,high,j,8);
            w.codes[j]=int((low[j]>>((group&1)*4))&0x0f0f0f0fu)|int(((upper>>group)<<4)&0x10101010u);
        }
        if(group<4){w.scale=b->scales[group]&63;w.minimum=b->scales[group+4]&63;}
        else {w.scale=(b->scales[group+4]&15)|((b->scales[group-4]>>6)<<4);w.minimum=(b->scales[group+4]>>4)|((b->scales[group]>>6)<<4);}
        w.d=__half22float2(b->dm);return w;
    }
    __device__ static Activation load_activation(const block_q8_1* input,int group) {
        const auto& x=input[group];Activation a;a.ones=0;a.d=__low2float(x.ds);
        #pragma unroll
        for(int j=0;j<8;++j){a.codes[j]=get_int_b4(x.qs,j);a.ones=ggml_cuda_dp4a(0x01010101,a.codes[j],a.ones);}
        return a;
    }
    __device__ static float apply(const Weight& w,const block_q8_1* x){return apply_activation(w,load_activation(x,w.group));}
    __device__ static float apply_activation(const Weight& w,const Activation& a) {
        int dot=0;
        #pragma unroll
        for(int j=0;j<8;++j)dot=ggml_cuda_dp4a(w.codes[j],a.codes[j],dot);
        const float sum=a.d*float(dot*w.scale),minimum=a.d*float(a.ones*w.minimum);
        return w.d.x*sum-w.d.y*minimum;
    }
};
}}
