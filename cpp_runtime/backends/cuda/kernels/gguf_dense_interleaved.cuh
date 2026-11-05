#pragma once
#include "warp_sums.cuh"
// Activation layout follows Strata q8_1_il.cuh (MIT; ../../../third_party/strata.LICENSE).
// Keep the original per-format dot expression and the current dense reduction tree.
namespace mfq::cuda::gguf { namespace {
template<class Input>
__global__ void quantize_position_major(const Input* input,std::int64_t columns,int tokens,
        void* output,block_q8_1* native) {
    const auto i=std::int64_t(blockIdx.x)*256+threadIdx.x;
    if(i<std::int64_t(tokens)*columns)store_q8_position_major(float(input[i]),i,columns,tokens,output,native);
}
template<int C> struct DenseInterleavedVector;
template<> struct DenseInterleavedVector<2> {using I=int2;using F=float2;__device__ static I integers(const int* p){return *reinterpret_cast<const I*>(p);}__device__ static F floats(const float* p){return *reinterpret_cast<const F*>(p);}};
template<> struct DenseInterleavedVector<4> {using I=int4;using F=float4;__device__ static I integers(const int* p){return *reinterpret_cast<const I*>(p);}__device__ static F floats(const float* p){return *reinterpret_cast<const F*>(p);}};
template<int M,int W>
__device__ __forceinline__ void project_dense_iq4_streamed_body(const uint8_t* weights,uint64_t stride,int rows,int k,const void* input,const float* scales,void* output,bool half_output,unsigned tile,float* partial){
    constexpr int Words=1;const auto* x=static_cast<const int*>(input);
    constexpr int R=2,C=M==2?2:4,Count=R*M,Capacity=warp_sums::capacity(Count);
    const int lane=threadIdx.x,warp=threadIdx.y,row=int(tile)*R,groups=k/32;
    const auto* a=reinterpret_cast<const block_iq4_xs*>(weights+uint64_t(row)*stride);
    const auto* b=reinterpret_cast<const block_iq4_xs*>(weights+uint64_t(min(row+1,rows-1))*stride);
    float sums[R][M]{};
    for(int i=warp*32+lane;warp<W && i<groups;i+=32*W){
        const int block=i/8,group=i%8;const auto* wa=a+block;const auto* wb=b+block;
        int dot[R][M]{};
        #pragma unroll
        for(int chunk=0;chunk<4/Words;++chunk){
            using V=typename std::conditional<Words==2,uint2,unsigned>::type;
            const V va=__ldg(reinterpret_cast<const V*>(wa->qs+group*16)+chunk),vb=__ldg(reinterpret_cast<const V*>(wb->qs+group*16)+chunk);
            const auto* ca=reinterpret_cast<const unsigned*>(&va);const auto* cb=reinterpret_cast<const unsigned*>(&vb);
            #pragma unroll
            for(int j=0;j<Words;++j){
                const auto da=get_int_from_table_16(ca[j],kvalues_iq4nl),db=get_int_from_table_16(cb[j],kvalues_iq4nl);
                const auto lo=DenseInterleavedVector<C>::integers(x+((chunk*Words+j)*groups+i)*C);
                const auto hi=DenseInterleavedVector<C>::integers(x+((chunk*Words+j+4)*groups+i)*C);
                const auto* l=reinterpret_cast<const int*>(&lo);const auto* h=reinterpret_cast<const int*>(&hi);
                #pragma unroll
                for(int t=0;t<M;++t){
                    dot[0][t]=ggml_cuda_dp4a(da.x,l[t],dot[0][t]);dot[0][t]=ggml_cuda_dp4a(da.y,h[t],dot[0][t]);
                    dot[1][t]=ggml_cuda_dp4a(db.x,l[t],dot[1][t]);dot[1][t]=ggml_cuda_dp4a(db.y,h[t],dot[1][t]);
                }
            }
        }
        const auto factors=DenseInterleavedVector<C>::floats(scales+i*C);const auto* f=reinterpret_cast<const float*>(&factors);
        #pragma unroll
        for(int r=0;r<R;++r){const auto* w=r?wb:wa;const int code=((w->scales_l[group/2]>>(4*(group&1)))&15)|(((w->scales_h>>(2*group))&3)<<4);const float scale=__half2float(w->d);
            #pragma unroll
            for(int t=0;t<M;++t)sums[r][t]+=(scale*f[t])*(dot[r][t]*(code-32));
        }
    }
    float values[Capacity]{};
    #pragma unroll
    for(int r=0;r<R;++r)
        #pragma unroll
        for(int t=0;t<M;++t)values[r*M+t]=sums[r][t];
    const float value=warp_sums::reduce<Capacity>(values,lane);const int index=warp_sums::index<Capacity>(lane);const bool owner=lane==warp_sums::owner<Capacity>(index);
    if(warp<W && owner && index<Count)partial[warp*Capacity+index]=value;
    __syncthreads();if(warp||!owner||index>=Count)return;float sum=partial[index];
    #pragma unroll
    for(int w=1;w<W;++w)sum+=partial[w*Capacity+index];
    if(row+index/M<rows)store_projection(output,int64_t(index%M)*rows+row+index/M,sum,half_output);
}

template<int Format,int M,bool Coop=false,int Warps=4>
__device__ __forceinline__ void project_dense_position_major_body(const uint8_t* weights,std::uint64_t row_bytes,std::int64_t rows,int columns,const void* input,const float* scales,void* output,bool half_output,unsigned tile,float* scratch) {
 if constexpr(Format==GGML_TYPE_IQ4_XS && Coop) {
  // Read one compressed word, consume it across every token, then retire it.
  // Host validation bounds row counts to INT_MAX; weight strides stay 64-bit.
  project_dense_iq4_streamed_body<M,Warps>(weights,row_bytes,int(rows),columns,input,scales,output,half_output,tile,scratch);
 } else {
 using D=PreparedDot<Format>;constexpr int C=M==2?2:4;
 constexpr int R=Coop?2:1;
 const int lane=threadIdx.x,warp=threadIdx.y;
 const std::int64_t row=Coop?std::int64_t(tile)*R:std::int64_t(tile)*4+warp;
 if constexpr(!Coop)if(row>=rows)return;
 float sums[R][M]{};
 {
  for(int i=(Coop?warp*32:0)+lane;(!Coop || warp<Warps) && i<columns/D::qk*D::parts;i+=Coop?32*Warps:32) {
   const int block=i/D::parts,part=D::step*(i%D::parts);
   typename D::Weight w[R]{};
   #pragma unroll
   for(int r=0;r<R;++r)if(row+r<rows)w[r]=D::load(weights+size_t(row+r)*row_bytes,block,part);
   const int xb=block*(D::qk/32)+(Format==GGML_TYPE_IQ4_XS?part/4:part/2);
   int dot[R][M]{};
   if constexpr(Format==GGML_TYPE_IQ4_XS) {
    #pragma unroll
    for(int j=0;j<4;++j) {
     const auto lo=DenseInterleavedVector<C>::integers(static_cast<const int*>(input)+(j*(columns/32)+xb)*C);
     const auto hi=DenseInterleavedVector<C>::integers(static_cast<const int*>(input)+((j+4)*(columns/32)+xb)*C);
     const int* l=reinterpret_cast<const int*>(&lo);const int* h=reinterpret_cast<const int*>(&hi);
     #pragma unroll
     for(int r=0;r<R;++r) {
      #pragma unroll
      for(int t=0;t<M;++t){dot[r][t]=ggml_cuda_dp4a(w[r].codes[j].x,l[t],dot[r][t]);dot[r][t]=ggml_cuda_dp4a(w[r].codes[j].y,h[t],dot[r][t]);}
     }
    }
   } else {
    #pragma unroll
    for(int j=0;j<8;++j) {
     const auto u=DenseInterleavedVector<C>::integers(static_cast<const int*>(input)+(j*(columns/32)+xb)*C);const int* a=reinterpret_cast<const int*>(&u);
     #pragma unroll
     for(int r=0;r<R;++r) {
      #pragma unroll
      for(int t=0;t<M;++t)dot[r][t]=ggml_cuda_dp4a(w[r].codes[j],a[t],dot[r][t]);
     }
    }
   }
   const auto factors=DenseInterleavedVector<C>::floats(scales+xb*C);const float* f=reinterpret_cast<const float*>(&factors);
   #pragma unroll
   for(int r=0;r<R;++r) {
    #pragma unroll
    for(int t=0;t<M;++t) {
     int value=dot[r][t];
     if constexpr(Format==GGML_TYPE_IQ3_XXS)value=(w[r].scale_code*value+value/2)/2;else value*=w[r].scale_code;
     sums[r][t]+=(w[r].scale*f[t])*value;
    }
   }
  }
  if constexpr(Coop && Format==GGML_TYPE_IQ4_XS) {
   // Match native M1: reduce lanes inside each K warp, then add the K
   // partials in warp order. Token/row ownership may change independently.
   constexpr int Count=R*M,Capacity=warp_sums::capacity(Count);
   float values[Capacity]{};
   #pragma unroll
   for(int r=0;r<R;++r)
    #pragma unroll
    for(int t=0;t<M;++t)values[r*M+t]=sums[r][t];
   const float value=warp_sums::reduce<Capacity>(values,lane);
   const int index=warp_sums::index<Capacity>(lane);
   const bool owner=lane==warp_sums::owner<Capacity>(index);
   if(warp<Warps && owner && index<Count)scratch[warp*Capacity+index]=value;
   __syncthreads();
   if(warp || !owner || index>=Count)return;
   float sum=scratch[index];
   #pragma unroll
   for(int w=1;w<Warps;++w)sum+=scratch[w*Capacity+index];
   const int r=index/M,t=index%M;
   if(row+r<rows)store_projection(output,t*rows+row+r,sum,half_output);
   return;
  }
  if constexpr(Coop) {
   auto& partial=*reinterpret_cast<float(*)[Warps-1][R][M][32]>(scratch);
   if(warp && warp<Warps)for(int r=0;r<R;++r)for(int t=0;t<M;++t)partial[warp-1][r][t][lane]=sums[r][t];
   __syncthreads();if(warp)return;
   #pragma unroll
   for(int r=0;r<R;++r) {
    #pragma unroll
    for(int t=0;t<M;++t)for(int w=0;w<Warps-1;++w)sums[r][t]+=partial[w][r][t][lane];
   }
  }
 }
 constexpr int Count=R*M,Capacity=warp_sums::capacity(Count);
 float values[Capacity]{};
 #pragma unroll
 for(int r=0;r<R;++r) {
  #pragma unroll
  for(int t=0;t<M;++t)values[r*M+t]=sums[r][t];
 }
 const float sum=warp_sums::reduce<Capacity>(values,lane);
 const int index=warp_sums::index<Capacity>(lane),r=index/M,t=index%M;
 if(index<Count && row+r<rows && lane==warp_sums::owner<Capacity>(index))
  store_projection(output,t*rows+row+r,sum,half_output);
 }
}
template<int Format,int M,bool Coop=false,int Warps=4>
__global__ void __launch_bounds__(128) project_dense_position_major(const uint8_t* weights,std::uint64_t row_bytes,std::int64_t rows,int columns,const void* input,const float* scales,void* output,bool half_output) {
 if constexpr(Coop) {
  __shared__ float partial[Format==GGML_TYPE_IQ4_XS?Warps*warp_sums::capacity(2*M):(Warps-1)*2*M*32];
  project_dense_position_major_body<Format,M,Coop,Warps>(weights,row_bytes,rows,columns,input,scales,output,half_output,blockIdx.x,partial);
 }else project_dense_position_major_body<Format,M,Coop,Warps>(weights,row_bytes,rows,columns,input,scales,output,half_output,blockIdx.x,nullptr);
}
}}
