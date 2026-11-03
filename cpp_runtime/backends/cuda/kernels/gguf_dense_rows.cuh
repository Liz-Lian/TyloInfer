#pragma once
#include "warp_sums.cuh"
// Cooperative row layout follows Strata native_mmvq.cu (MIT,
// ../../../third_party/strata.LICENSE). Preserve each ggml block dot; only
// FP32 reduction grouping differs from the one-warp dense implementation.
namespace mfq::cuda::gguf {
namespace {

template<class Dot>
__device__ __forceinline__ auto load_dense_m1_weight(const void* packed,int block,int part) {
    if constexpr(requires {Dot::load_streaming(packed,block,part);})
        return Dot::load_streaming(packed,block,part);
    else return Dot::load(packed,block,part);
}

struct M1ProjectionStore {
    __device__ __forceinline__ void operator()(void* output,int64_t row,float value,bool half_output) const {
        store_projection(output,row,value,half_output);
    }
};

// Keep a decoded activation fragment in registers across output rows.
// Cooperative variants join each warp's row partials through scratch.
template<int Format,int Rows,int Warps=1,int BlockWarps=4,class Store=M1ProjectionStore,
    class Prepared=M1PreparedDot<Format>>
__device__ __forceinline__ void project_dense_reused_q8_body(
        const std::uint8_t* packed,std::uint64_t stride,int rows,int columns,
        const block_q8_1* input,void* output,bool half_output,unsigned tile,float* scratch=nullptr,Store store={}) {
    using F=Prepared;
    const int warp=int(threadIdx.y),lane=int(threadIdx.x);
    static_assert(Warps>0 && Warps<=BlockWarps);
    constexpr int ActiveWarps=BlockWarps/Warps*Warps;
    const bool active=BlockWarps%Warps==0 || warp<ActiveWarps;
    const unsigned row0=(tile*(BlockWarps/Warps)+unsigned(warp/Warps))*Rows;
    if constexpr(Warps==1)if(row0>=unsigned(rows))return;
    // A wide row tile has several reuse sites. Clamp only its final tile's
    // read addresses once, allowing all rows to share an unpredicated inner
    // loop. Masked output stores still preserve every partial row boundary.
    const std::uint8_t* safe_rows[Rows];
    if constexpr(Rows>2) {
        #pragma unroll
        for(int r=0;r<Rows;++r)safe_rows[r]=packed+std::uint64_t(min(row0+unsigned(r),unsigned(rows-1)))*stride;
    }
    const auto row_weights=[&](int r) {
        if constexpr(Rows>2)return safe_rows[r];
        else return packed+std::uint64_t(row0+r)*stride;
    };
    float sums[Rows]{};const int count=columns/F::qk*F::parts;
    for(int i=(warp%Warps)*32+lane;active && row0<unsigned(rows) && i<count;i+=32*Warps) {
        const int block=i/F::parts,part=F::step*(i%F::parts);
        const auto* x=input+block*(F::qk/32);
        if constexpr(requires {typename F::Activation;}) {
            // Decode the common activation before any row weight. Retire
            // each row's metadata immediately instead of keeping the first
            // weight alive across activation preparation and later rows.
            const auto activation=F::load_activation(x,part);
            #pragma unroll
            for(int r=0;r<Rows;++r)if(Rows>2 || row0+r<rows) {
                const auto* row=packed+std::uint64_t(min(row0+unsigned(r),unsigned(rows-1)))*stride;
                const auto w=load_dense_m1_weight<F>(row,block,part);
                sums[r]+=F::apply_activation(w,activation);
            }
        } else if constexpr(requires {&F::apply_u;&F::input_block;}) {
            const auto first=load_dense_m1_weight<F>(row_weights(0),block,part);
            int codes[8];const auto* activation=x+F::input_block(first);
            #pragma unroll
            for(int j=0;j<8;++j)codes[j]=get_int_b4(activation->qs,j);
            const float scale=__low2float(activation->ds);
            sums[0]+=F::apply_u(first,codes,scale);
            #pragma unroll
            for(int r=1;r<Rows;++r)if(Rows>2 || row0+r<rows) {
                const auto w=load_dense_m1_weight<F>(row_weights(r),block,part);
                sums[r]+=F::apply_u(w,codes,scale);
            }
        } else {
            const auto first=load_dense_m1_weight<F>(row_weights(0),block,part);
            sums[0]+=F::apply(first,x);
            #pragma unroll
            for(int r=1;r<Rows;++r)if(Rows>2 || row0+r<rows) {
                const auto w=load_dense_m1_weight<F>(row_weights(r),block,part);
                sums[r]+=F::apply(w,x);
            }
        }
    }
    // Distribute output rows across warp subgroups instead of running an
    // independent shuffle tree per row. The pairwise FP32 tree is retained.
    constexpr int Capacity=warp_sums::capacity(Rows);
    float values[Capacity]{};
    #pragma unroll
    for(int r=0;r<Rows;++r)values[r]=sums[r];
    float sum=warp_sums::reduce<Capacity>(values,lane);
    const int index=warp_sums::index<Capacity>(lane);
    const bool owner=lane==warp_sums::owner<Capacity>(index);
    if constexpr(Warps==1) {
        if(owner && index<Rows && row0+index<unsigned(rows))store(output,row0+index,sum,half_output);
    } else {
        if(active && owner && index<Rows)scratch[warp*Rows+index]=sum;
        __syncthreads();
        if(active && owner && index<Rows && warp%Warps==0 && row0+index<unsigned(rows)) {
            sum=scratch[warp*Rows+index];
            #pragma unroll
            for(int w=1;w<Warps;++w)sum+=scratch[(warp+w)*Rows+index];
            store(output,row0+index,sum,half_output);
        }
    }
}

// Low-bit IQ1 rows amortize activation work over four outputs. Two warps
// divide K so the tile still produces eight rows and keeps enough CTAs.
// Short reductions retain the single-warp path instead of idle K warps.
template<int Format>
__device__ __forceinline__ void project_dense_single_q8_body(
        const std::uint8_t* packed,std::uint64_t stride,int rows,int columns,
        const block_q8_1* input,void* output,bool half_output,unsigned tile,float* scratch) {
    if constexpr(Format==GGML_TYPE_IQ1_S || Format==GGML_TYPE_IQ1_M) {
        if(columns/Dot<Format>::qk*Dot<Format>::parts>=64) {
            project_dense_reused_q8_body<Format,4,2>(packed,stride,rows,columns,input,output,half_output,tile,scratch);
            return;
        }
    }
    project_dense_reused_q8_body<Format,2>(packed,stride,rows,columns,input,output,half_output,tile);
}

template<int Format,int Rows,int Warps=1,int BlockWarps=4>
__global__ void __launch_bounds__(32*BlockWarps) project_dense_reused_q8(
        const std::uint8_t* packed,std::uint64_t stride,int rows,int columns,
        const block_q8_1* input,void* output,bool half_output) {
    __shared__ float scratch[(Rows>4?Rows:4)*BlockWarps];
    if constexpr(Warps>1 || BlockWarps!=4)project_dense_reused_q8_body<Format,Rows,Warps,BlockWarps>(packed,stride,rows,columns,input,output,half_output,blockIdx.x,scratch);
    else if constexpr(Rows==2)project_dense_single_q8_body<Format>(packed,stride,rows,columns,input,output,half_output,blockIdx.x,scratch);
    else project_dense_reused_q8_body<Format,Rows>(packed,stride,rows,columns,input,output,half_output,blockIdx.x);
}
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
