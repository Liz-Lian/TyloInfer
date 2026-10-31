#pragma once
#include "warp_sums.cuh"
// Cooperative row layout follows Strata native_mmvq.cu (MIT,
// ../../../third_party/strata.LICENSE). Preserve each ggml block dot; only
// FP32 reduction grouping differs from the one-warp dense implementation.
namespace mfq::cuda::gguf {
namespace {
template<int Format,int Tokens,int OutputRows,int Warps=4>
__device__ __forceinline__ void project_dense_cooperative_q8_body(
        const std::uint8_t* packed,int stride,int rows,int columns,int tokens,
        const block_q8_1* input,void* output,bool half_output,unsigned tile,float* scratch) {
    using F=PreparedDot<Format>;
    const int row0=int(tile)*OutputRows,lane=int(threadIdx.x),warp=int(threadIdx.y);
    float sums[OutputRows][Tokens]{};
    for(int i=warp*32+lane;warp<Warps && i<columns/F::qk*F::parts;i+=32*Warps) {
        const int block=i/F::parts,part=F::step*(i%F::parts);
        if constexpr(OutputRows==2 && requires {typename F::Activation;}) {
            // Keep two decoded weight fragments and retire each token's
            // activation immediately. Keeping all token activations alive
            // across output rows raises register pressure at small M.
            // Clamp the partial final tile's second load to a valid row;
            // its accumulator and store remain masked.
            const auto first=F::load(packed+std::uint64_t(row0)*stride,block,part);
            const auto second=F::load(packed+std::uint64_t(min(row0+1,rows-1))*stride,block,part);
            #pragma unroll
            for(int t=0;t<Tokens;++t)if(t<tokens) {
                const auto activation=F::load_activation(input+t*(columns/32)+block*(F::qk/32),part);
                sums[0][t]+=F::apply_activation(first,activation);
                if(row0+1<rows)sums[1][t]+=F::apply_activation(second,activation);
            }
        }else if constexpr(OutputRows>1 && requires {typename F::Activation;}) {
            // Reuse activation words, scales and affine correction sums
            // across output rows. Keep each output's original dot/add tree.
            typename F::Activation activation[Tokens];
            #pragma unroll
            for(int t=0;t<Tokens;++t)if(t<tokens)
                activation[t]=F::load_activation(input+t*(columns/32)+block*(F::qk/32),part);
            #pragma unroll
            for(int r=0;r<OutputRows;++r)if(row0+r<rows) {
                const auto weight=F::load(packed+std::uint64_t(row0+r)*stride,block,part);
                #pragma unroll
                for(int t=0;t<Tokens;++t)if(t<tokens)sums[r][t]+=F::apply_activation(weight,activation[t]);
            }
        }else {
        #pragma unroll
        for(int r=0;r<OutputRows;++r)if(row0+r<rows) {
            const auto weight=F::load(packed+std::uint64_t(row0+r)*stride,block,part);
            #pragma unroll
            for(int t=0;t<Tokens;++t)if(t<tokens)
                sums[r][t]+=F::apply(weight,input+t*(columns/32)+block*(F::qk/32));
        }
        }
    }
    auto& partial=*reinterpret_cast<float(*)[Warps-1][OutputRows][Tokens][32]>(scratch);
    if(warp && warp<Warps) {
        #pragma unroll
        for(int r=0;r<OutputRows;++r)
            #pragma unroll
            for(int t=0;t<Tokens;++t)partial[warp-1][r][t][lane]=sums[r][t];
    }
    __syncthreads();
    if(warp)return;
    constexpr int Count=OutputRows*Tokens,Capacity=warp_sums::capacity(Count);
    float values[Capacity]{};
    #pragma unroll
    for(int r=0;r<OutputRows;++r) {
        #pragma unroll
        for(int t=0;t<Tokens;++t) {
            float sum=sums[r][t];
            #pragma unroll
            for(int w=0;w<Warps-1;++w)sum+=partial[w][r][t][lane];
            values[r*Tokens+t]=sum;
        }
    }
    const float sum=warp_sums::reduce<Capacity>(values,lane);
    const int index=warp_sums::index<Capacity>(lane),r=index/Tokens,t=index%Tokens;
    if(index<Count && row0+r<rows && t<tokens && lane==warp_sums::owner<Capacity>(index))
        store_projection(output,t*rows+row0+r,sum,half_output);
}
template<int Format,int Tokens,int OutputRows,int Warps=4>
__global__ void __launch_bounds__(32*Warps) project_dense_cooperative_q8(
        const std::uint8_t* packed,int stride,int rows,int columns,int tokens,
        const block_q8_1* input,void* output,bool half_output) {
    __shared__ float partial[(Warps-1)*OutputRows*Tokens*32];
    project_dense_cooperative_q8_body<Format,Tokens,OutputRows,Warps>(packed,stride,rows,columns,tokens,
        input,output,half_output,blockIdx.x,partial);
}
}
}
