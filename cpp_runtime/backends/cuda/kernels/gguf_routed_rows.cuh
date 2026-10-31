#pragma once
// Sub-row assignment follows Strata's native expert kernels (MIT,
// ../../../third_party/strata.LICENSE). Keep runtime dimensions and reconstruct
// the bundled ggml 32-lane addition tree, including arbitrary short/tail rows.
namespace mfq::cuda::gguf {
namespace {
template<int Format,int Lanes>
__device__ __forceinline__ float q8_dot_subrow(const std::uint8_t* weight,
        const block_q8_1* input,int columns,int lane) {
    using F=PreparedDot<Format>;
    static_assert(Lanes==8 || Lanes==16 || Lanes==32);
    constexpr int Parts=32/Lanes;
    float partial[Parts]{};
    #pragma unroll
    for(int p=0;p<Parts;++p) {
        for(int i=lane+p*Lanes;i<columns/F::qk*F::parts;i+=32) {
            const int block=i/F::parts,part=F::step*(i%F::parts);
            const auto prepared=F::load(weight,block,part);
            partial[p]+=F::apply(prepared,input+block*(F::qk/32));
        }
    }
    float sum;
    if constexpr(Parts==1)sum=partial[0];
    else if constexpr(Parts==2)sum=partial[0]+partial[1];
    else sum=(partial[0]+partial[2])+(partial[1]+partial[3]);
    #pragma unroll
    for(int step=Lanes/2;step;step>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,step,Lanes);
    return sum;
}
template<int Format,int Lanes>
__global__ void __launch_bounds__(128) project_routed_subrow_q8(const std::uint64_t* addresses,
        std::uint64_t stride,int experts,int rows,int columns,const std::int64_t* ids,int slots,
        int routes,const block_q8_1* input,bool routed_input,float* output,bool preserve_missing) {
    const auto index=blockIdx.x*(128/Lanes)+threadIdx.x/Lanes;
    if(index>=unsigned(slots)*unsigned(rows))return;
    const int row=int(index)%rows,slot=int(index)/rows,lane=threadIdx.x%Lanes;
    const auto expert=ids[slot];
    const auto address=expert>=0 && expert<experts?addresses[expert]:0;
    float sum=0;
    if(address)sum=q8_dot_subrow<Format,Lanes>(reinterpret_cast<const std::uint8_t*>(address)+std::uint64_t(row)*stride,
        input+(routed_input?slot:slot/routes)*(columns/32),columns,lane);
    if(!lane && (address || !preserve_missing))output[index]=sum;
}
}
}
