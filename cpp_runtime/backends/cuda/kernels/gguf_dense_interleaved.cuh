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
template<int Format,int M,bool Coop=false,int Warps=4>
__device__ __forceinline__ void project_dense_position_major_body(const uint8_t* weights,std::uint64_t row_bytes,std::int64_t rows,int columns,const void* input,const float* scales,void* output,bool half_output,unsigned tile,float* scratch) {
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
template<int Format,int M,bool Coop=false,int Warps=4>
__global__ void project_dense_position_major(const uint8_t* weights,std::uint64_t row_bytes,std::int64_t rows,int columns,const void* input,const float* scales,void* output,bool half_output) {
 if constexpr(Coop) {
  __shared__ float partial[(Warps-1)*2*M*32];
  project_dense_position_major_body<Format,M,Coop,Warps>(weights,row_bytes,rows,columns,input,scales,output,half_output,blockIdx.x,partial);
 }else project_dense_position_major_body<Format,M,Coop,Warps>(weights,row_bytes,rows,columns,input,scales,output,half_output,blockIdx.x,nullptr);
}
}}
