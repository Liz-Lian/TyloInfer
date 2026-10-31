#pragma once
// Shared activations and prepared weight reuse follow Strata iq_kernels.cu
// (MIT, cpp_runtime/third_party/strata.LICENSE). Virtual sub-rows preserve the
// bundled ggml 32-lane addition tree, including arbitrary output tails.
namespace mfq::cuda::gguf {
namespace {

template<int Format,int Count,int Lanes>
__device__ __forceinline__ void q8_dot_shared_subrows(const std::uint8_t* weight,const block_q8_1* input,
        int columns,int lane,float (&out)[Count],bool active) {
    using F=PreparedDot<Format>;
    constexpr int Parts=32/Lanes;
    float sums[Count][Parts]{};
    const int items=columns/F::qk*F::parts,stride=columns/32;
    #pragma unroll
    for(int p=0;p<Parts;++p)for(int i=lane+p*Lanes;active && i<items;i+=32) {
        const int block=i/F::parts,part=F::step*(i%F::parts);
        const auto w=F::load(weight,block,part);
        #pragma unroll
        for(int token=0;token<Count;++token)sums[token][p]+=F::apply(w,input+token*stride+block*(F::qk/32));
    }
    #pragma unroll
    for(int token=0;token<Count;++token) {
        #pragma unroll
        for(int step=Parts/2;step;step>>=1) {
            #pragma unroll
            for(int p=0;p<step;++p)sums[token][p]+=sums[token][p+step];
        }
        float sum=sums[token][0];
        for(int step=Lanes/2;step;step>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,step,Lanes);
        out[token]=sum;
    }
}

template<int Format,int Capacity>
__global__ void project_routed_shared_q8(const std::uint64_t* addresses,std::uint64_t stride,
        int experts,int rows,int columns,int slots,const std::int64_t* ids,const RouteGroup* groups,int routes,
        const block_q8_1* input,bool routed_input,float* output,bool preserve_missing) {
    extern __shared__ std::uint32_t activation_words[];
    constexpr int Lanes=Dot<Format>::qk<=64?8:16,Rows=256/Lanes;
    const int tid=threadIdx.y*32+threadIdx.x,lane=tid%Lanes;
    const unsigned row=blockIdx.x*Rows+unsigned(tid/Lanes);
    const int words=columns/32*int(sizeof(block_q8_1)/4);
    const bool valid=row<unsigned(rows);
    auto* staged=reinterpret_cast<block_q8_1*>(activation_words);
    if(groups[0].count==slots) {
        const unsigned tiles=gridDim.x*gridDim.y,block=blockIdx.y*gridDim.x+blockIdx.x;
        const unsigned total=unsigned(slots)*unsigned(rows);
        for(unsigned base=block*Rows;base<total;base+=tiles*Rows) {
            const unsigned i=base+unsigned(tid/Lanes);const bool live=i<total;
            const unsigned group_index=live?i/unsigned(rows):0,row_index=live?i%unsigned(rows):0;
            const auto id=ids[group_index];
            const auto address=id>=0 && id<experts?addresses[id]:0;
            const auto slot=int(group_index),input_row=routed_input?slot:slot/routes;
            const auto* weight=live && address?reinterpret_cast<const std::uint8_t*>(address)+std::uint64_t(row_index)*stride:nullptr;
            float sum[1]{};
            q8_dot_shared_subrows<Format,1,Lanes>(weight,input+input_row*(columns/32),columns,lane,sum,live && address!=0);
            if(live && !lane && (address || !preserve_missing))output[i]=sum[0];
        }
        return;
    }

    for(int group_index=blockIdx.y;group_index<groups[0].count;group_index+=gridDim.y) {
        const auto* group=groups+1+group_index;
        const int count=group->count;
        const auto expert=group->expert;
        const auto address=expert>=0 && expert<experts?addresses[expert]:0;
        if(!address) {
            if(valid && !lane && !preserve_missing)for(int token=0;token<count;++token)
                output[std::int64_t(group->slots[token])*rows+row]=0.f;
            continue;
        }
        if(count>1) {
            for(int token=0;token<count;++token) {
                const int slot=group->slots[token],input_row=routed_input?slot:slot/routes;
                const auto* source=reinterpret_cast<const std::uint32_t*>(input)+input_row*words;
                for(int word=tid;word<words;word+=256)activation_words[token*words+word]=source[word];
            }
            __syncthreads();
        }
        const auto compute=[&]<int Count>() {
            float sums[Count]{};
            const auto* x=staged;
            if constexpr(Count==1) {
                const auto slot=group->slots[0],input_row=routed_input?slot:slot/routes;
                x=input+input_row*(columns/32);
            }
            const auto* weight=valid && address?reinterpret_cast<const std::uint8_t*>(address)+std::uint64_t(row)*stride:nullptr;
            q8_dot_shared_subrows<Format,Count,Lanes>(weight,
                x,columns,lane,sums,valid && address!=0);
            if(valid && !lane && (address || !preserve_missing))for(int token=0;token<Count;++token)
                output[std::int64_t(group->slots[token])*rows+row]=sums[token];
        };
        if(count==1)compute.template operator()<1>();
        else if constexpr(Capacity==2)compute.template operator()<2>();
        else if constexpr(Capacity==3) {
            if(count==2)compute.template operator()<2>();else compute.template operator()<3>();
        } else if constexpr(Capacity>=4) {
            if(count==2)compute.template operator()<2>();else if(count==3)compute.template operator()<3>();else compute.template operator()<4>();
        }
        if(count>1 && group_index+int(gridDim.y)<groups[0].count)__syncthreads();
    }
}
}
}
