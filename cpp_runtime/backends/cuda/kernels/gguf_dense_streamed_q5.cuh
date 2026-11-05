#pragma once

namespace mfq::cuda::gguf {
namespace {
// Consume each compressed word chunk across the token window immediately.
// Integer partials retain weight reuse without keeping both expanded rows
// alive throughout all activation loads. The native Q8_1 packet is unchanged.
template<int Tokens,int Warps,int BlockWarps=Warps>
__device__ __forceinline__ void project_dense_streamed_q5_body(
        const uint8_t* weights,int stride,int rows,int columns,const block_q8_1* input,
        void* output,bool half_output,unsigned tile,float* partial) {
    static_assert(Tokens>=2 && Tokens<=4);
    static_assert(BlockWarps>=Warps);
    constexpr int Rows=2,Capacity=warp_sums::capacity(Rows*Tokens),Groups=BlockWarps/Warps;
    const int block_warp=threadIdx.y,lane=threadIdx.x;
    const bool active=block_warp<Groups*Warps;
    const int group=active?block_warp/Warps:0,warp=Groups>1?block_warp%Warps:block_warp;
    const int row0=(int(tile)*Groups+group)*Rows;
    if constexpr(Groups>1)partial+=group*Warps*Capacity;
    const auto* first=weights+uint64_t(Groups>1?min(row0,rows-1):row0)*stride;
    const auto* second=weights+uint64_t(min(row0+1,rows-1))*stride;
    float sums[Rows][Tokens]{};
    for(int i=warp*32+lane;active && (Groups==1 || row0<rows) && i<columns/32;i+=32*Warps) {
        const int block=i/8,group=i%8;
        const auto* a=reinterpret_cast<const block_q5_K*>(first)+block;
        const auto* b=reinterpret_cast<const block_q5_K*>(second)+block;
        int dots[Rows][Tokens]{},ones[Tokens]{};
        #pragma unroll
        for(int chunk=0;chunk<2;++chunk) {
            const auto low_a=__ldg(reinterpret_cast<const uint4*>(a->qs+32*(group/2)+16*chunk));
            const auto high_a=__ldg(reinterpret_cast<const uint4*>(a->qh+16*chunk));
            const auto low_b=__ldg(reinterpret_cast<const uint4*>(b->qs+32*(group/2)+16*chunk));
            const auto high_b=__ldg(reinterpret_cast<const uint4*>(b->qh+16*chunk));
            const auto* al=reinterpret_cast<const unsigned*>(&low_a);
            const auto* ah=reinterpret_cast<const unsigned*>(&high_a);
            const auto* bl=reinterpret_cast<const unsigned*>(&low_b);
            const auto* bh=reinterpret_cast<const unsigned*>(&high_b);
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int ca=int((al[j]>>(4*(group&1)))&0x0f0f0f0fu)|int(((ah[j]>>group)<<4)&0x10101010u);
                const int cb=int((bl[j]>>(4*(group&1)))&0x0f0f0f0fu)|int(((bh[j]>>group)<<4)&0x10101010u);
                #pragma unroll
                for(int t=0;t<Tokens;++t) {
                    const int code=get_int_b4(input[t*(columns/32)+i].qs,chunk*4+j);
                    dots[0][t]=ggml_cuda_dp4a(ca,code,dots[0][t]);
                    dots[1][t]=ggml_cuda_dp4a(cb,code,dots[1][t]);
                    ones[t]=ggml_cuda_dp4a(0x01010101,code,ones[t]);
                }
            }
        }
        #pragma unroll
        for(int r=0;r<Rows;++r) {
            const auto* w=r?b:a;int scale,minimum;
            if(group<4){scale=w->scales[group]&63;minimum=w->scales[group+4]&63;}
            else {
                scale=(w->scales[group+4]&15)|((w->scales[group-4]>>6)<<4);
                minimum=(w->scales[group+4]>>4)|((w->scales[group]>>6)<<4);
            }
            const auto dm=__half22float2(w->dm);
            #pragma unroll
            for(int t=0;t<Tokens;++t) {
                const float d=__low2float(input[t*(columns/32)+i].ds);
                const float dot=d*float(dots[r][t]*scale),correction=d*float(ones[t]*minimum);
                sums[r][t]+=dm.x*dot-dm.y*correction;
            }
        }
    }
    float values[Capacity]{};
    #pragma unroll
    for(int r=0;r<Rows;++r)
        #pragma unroll
        for(int t=0;t<Tokens;++t)values[r*Tokens+t]=sums[r][t];
    const float value=warp_sums::reduce<Capacity>(values,lane);
    const int index=warp_sums::index<Capacity>(lane);
    const bool owner=lane==warp_sums::owner<Capacity>(index);
    // A mixed group launches four warps even for a three-warp member.
    // Its extra warp participates in the barrier without loading weights.
    if(active && owner && index<Rows*Tokens)partial[warp*Capacity+index]=value;
    __syncthreads();
    if(!active || warp || !owner || index>=Rows*Tokens)return;
    float sum=partial[index];
    #pragma unroll
    for(int w=1;w<Warps;++w)sum+=partial[w*Capacity+index];
    const int row=row0+index/Tokens,token=index%Tokens;
    if(row<rows)store_projection(output,int64_t(token)*rows+row,sum,half_output);
}

template<int Tokens,int Warps>
__global__ void __launch_bounds__(32*Warps) project_dense_streamed_q5(
        const uint8_t* weights,int stride,int rows,int columns,int tokens,
        const block_q8_1* input,void* output,bool half_output) {
    __shared__ float partial[Warps*warp_sums::capacity(2*Tokens)];
    project_dense_streamed_q5_body<Tokens,Warps>(weights,stride,rows,columns,input,output,half_output,blockIdx.x,partial);
}
}
}
