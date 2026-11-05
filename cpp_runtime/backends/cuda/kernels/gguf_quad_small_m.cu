#include "gguf_quad_small_m.h"
#include "gguf_quad_storage.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "gguf_quantize.cuh"
#include "gguf_dense_rows.cuh"
#include "kernel_dispatch_trace.h"
#include <climits>
#include <stdexcept>

namespace mfq::cuda::gguf {
namespace {
#include "gguf_quad_dot.cuh"
// The same lossless four-row planes used by M=1. No canonical or expanded
// weight image is materialized. Decode an eight-coordinate fragment once;
// consume each token's two activation words before decoding the next one.
template<class Index>__device__ __forceinline__ uint4 load_four(const uint8_t* base,Index offset){
 if constexpr(sizeof(Index)==4){uint4 v;asm volatile("{ .reg .u64 a; mad.wide.u32 a,%5,1,%4; ld.global.cs.v4.u32 {%0,%1,%2,%3},[a]; }":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(base),"r"(offset));return v;}
 else return __ldcs(reinterpret_cast<const uint4*>(base+offset));
}
template<class Index>__device__ __forceinline__ uint2 load_two(const uint8_t* base,Index offset){
 if constexpr(sizeof(Index)==4){uint2 v;asm volatile("{ .reg .u64 a; mad.wide.u32 a,%3,1,%2; ld.global.cs.v2.u32 {%0,%1},[a]; }":"=r"(v.x),"=r"(v.y):"l"(base),"r"(offset));return v;}
 else return __ldcs(reinterpret_cast<const uint2*>(base+offset));
}
template<class Index>__device__ __forceinline__ unsigned load_one(const uint8_t* base,Index offset){
 if constexpr(sizeof(Index)==4){unsigned v;asm volatile("{ .reg .u64 a; mad.wide.u32 a,%2,1,%1; ld.global.cs.u32 %0,[a]; }":"=r"(v):"l"(base),"r"(offset));return v;}
 else return __ldcs(reinterpret_cast<const unsigned*>(base+offset));
}
template<int R>__device__ __forceinline__ unsigned field(uint4 v){if constexpr(R==0)return v.x;else if constexpr(R==1)return v.y;else if constexpr(R==2)return v.z;else return v.w;}
template<int F>struct PlaneOffsets{static constexpr unsigned Extra=F==GGML_TYPE_IQ2_XXS?0:F==GGML_TYPE_IQ2_XS?32:F==GGML_TYPE_IQ2_S?64:F==GGML_TYPE_IQ3_XXS?128:176;};
template<int F>constexpr unsigned GridCount=F==GGML_TYPE_IQ1_S?1024:F==GGML_TYPE_IQ1_M?2048:F==GGML_TYPE_IQ2_XXS||F==GGML_TYPE_IQ3_XXS?256:F==GGML_TYPE_IQ2_XS||F==GGML_TYPE_IQ3_S?512:F==GGML_TYPE_IQ2_S?1024:0;
// Transfer four shared codebook words per lane. IQ1 tables already expose
// 16-byte alignment; the other source representations retain their natural
// alignment and widen only their shared stores.
template<int F,int Offset,int Threads=128>__device__ __forceinline__ void stage_global_grid(int tid){
 static_assert(Offset%4==0 && GridCount<F>%4==0);
 extern __shared__ uint4 quad_codebook_vectors[];
 for(int i=tid*4;i<GridCount<F>;i+=Threads*4){
  uint4 value;
  if constexpr(F==GGML_TYPE_IQ1_S)value=__ldg(reinterpret_cast<const uint4*>(iq1_grid_2bit)+i/4);
  else if constexpr(F==GGML_TYPE_IQ1_M)value=__ldg(reinterpret_cast<const uint4*>(iq1_grid_selectors)+i/4);
  else {
   const auto at=[&](int index)->unsigned {
    if constexpr(F==GGML_TYPE_IQ2_XXS)return iq2xxs_grid_selectors[index];
    else if constexpr(F==GGML_TYPE_IQ2_XS)return iq2xs_grid_selectors[index];
    else if constexpr(F==GGML_TYPE_IQ2_S)return iq2s_grid_selectors[index];
    else if constexpr(F==GGML_TYPE_IQ3_XXS)return iq3xxs_grid_selectors[index];
    else if constexpr(F==GGML_TYPE_IQ3_S)return iq3s_grid_selectors[index];
    else return 0;
   };
   value=make_uint4(at(i),at(i+1),at(i+2),at(i+3));
  }
  quad_codebook_vectors[(Offset+i)/4]=value;
 }
}
template<int Tokens,class Index>__device__ __forceinline__ half2 row_packet_scales(const void* input,Index groups,int token,unsigned part){return __ldg(reinterpret_cast<const half2*>(static_cast<const uint8_t*>(input)+uint64_t(groups)*Tokens*32)+Index(token)*groups+part);}
struct PackedRowCodes {int value[8];};
template<int Tokens,class Index>__device__ __forceinline__ PackedRowCodes load_row_vector(const void* input,Index groups,int token,unsigned part) {
 // columns<=INT_MAX, groups=columns/32 and Tokens8 give
 // packet<2^29. Keep the GPUbase and resulting byteaddress64bit.
 const unsigned packet=unsigned(token)*unsigned(groups)+part;uint4 a,b;
 const auto* high=static_cast<const uint8_t*>(input)+uint64_t(groups)*Tokens*16;
 asm("{ .reg .u64 ptr; mad.wide.u32 ptr,%5,16,%4; ld.global.nc.v4.u32 {%0,%1,%2,%3},[ptr]; }":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(input),"r"(packet));
 asm("{ .reg .u64 ptr; mad.wide.u32 ptr,%5,16,%4; ld.global.nc.v4.u32 {%0,%1,%2,%3},[ptr]; }":"=r"(b.x),"=r"(b.y),"=r"(b.z),"=r"(b.w):"l"(high),"r"(packet));
 return {{int(a.x),int(a.y),int(a.z),int(a.w),int(b.x),int(b.y),int(b.z),int(b.w)}};
}


#include "gguf_quad_small_m_extra.cuh"

// Reuse the same sign expansion across rows, fragments and tokens. Keep
// the magnitude selectors compressed; the extra shared table is 512/1024 B.
template<int F>constexpr unsigned SignCount=F==GGML_TYPE_IQ2_XXS || F==GGML_TYPE_IQ2_XS?128:F==GGML_TYPE_IQ2_S?256:0;
template<int F>__device__ __forceinline__ void stage_sign_selectors(int tid) {
 extern __shared__ unsigned quad_codebook[];
 for(unsigned i=tid;i<SignCount<F>;i+=128) {
  unsigned signs=i;if constexpr(F!=GGML_TYPE_IQ2_S)signs|=(__popc(signs)&1u)<<7;
  unsigned pairs=(signs|(signs<<4))&0x0f0fu;
  pairs=(pairs|(pairs<<2))&0x3333u;
  quad_codebook[GridCount<F>+i]=iq_grid_byte_perm<true>(0x44400400u,0u,pairs);
 }
}

// The host selects this aligned specialization once. Each packet contains
// the same coordinate word for four adjacent queries.
template<class Index>__device__ __forceinline__ uint4 token_word4(const void* input,Index packet) {
 return __ldg(static_cast<const uint4*>(input)+packet);
}
template<int F,int Rows,int Tokens,bool PositionMajor,class Index,bool VectorTokens=false>
__device__ __forceinline__ void quad_small_m_body(QuadProjection p,int columns,const void* input,bool half_output,unsigned tile) {
 if constexpr(GridCount<F>!=0){
  stage_global_grid<F,0>(threadIdx.y*32+threadIdx.x);
  stage_sign_selectors<F>(threadIdx.y*32+threadIdx.x);__syncthreads();
 }
 constexpr int W=1,Offset=0,Capacity=Tokens==2?2:4;
 const auto* packed=static_cast<const uint8_t*>(p.projection.packed);const int rows=int(p.projection.rows);void* output=p.projection.output;

 using D=QuadIqDot<F,Offset,true>;const unsigned lane=threadIdx.x,warp=threadIdx.y;
 const unsigned row=(tile*(4/W)+warp/W)*Rows;
 const Index source_row=Index(row)+Index(p.row_begin),quad=source_row/4,pair=(source_row%4)/2,blocks=Index(columns/256),groups=blocks*8;
 const Index total=(Index(p.storage_rows)/4)*blocks;
 float sums[Tokens][Rows]{};
 for(unsigned i=(warp%W)*32+lane;warp<(4/W)*W&&row<unsigned(rows)&&i<groups;i+=32*W){
  const Index entry=quad*groups+i,qb=quad*blocks+i/8,part=i%8;uint4 c0{},c1{},signs{};uint2 ds{},meta{};unsigned scales=0,high=0,wide_scales=0;
  if constexpr(Rows==4){c0=load_four(packed,entry*32);c1=load_four(packed,entry*32+16);ds=load_two(packed,total*(256+PlaneOffsets<F>::Extra)+qb*8);}
  else{const auto a=load_two(packed,entry*32+pair*8),b=load_two(packed,entry*32+16+pair*8);c0=make_uint4(a.x,a.y,0,0);c1=make_uint4(b.x,b.y,0,0);ds=make_uint2(load_one(packed,total*(256+PlaneOffsets<F>::Extra)+qb*8+pair*4),0);}
  if constexpr(F==GGML_TYPE_IQ2_XS)scales=load_one(packed,total*256+entry*4)>>(Rows==2?pair*16:0);
  else if constexpr(F==GGML_TYPE_IQ2_S){if constexpr(Rows==4)meta=load_two(packed,total*256+entry*8);else meta=make_uint2(load_one(packed,total*256+entry*8+pair*4),0);}
  else if constexpr(F==GGML_TYPE_IQ3_XXS||F==GGML_TYPE_IQ3_S){if constexpr(Rows==4)signs=load_four(packed,total*256+entry*16);else{const auto s=load_two(packed,total*256+entry*16+pair*8);signs=make_uint4(s.x,s.y,0,0);}
   if constexpr(F==GGML_TYPE_IQ3_S){high=load_one(packed,total*384+entry*4)>>(Rows==2?pair*16:0);wide_scales=load_one(packed,total*416+(entry/2)*4)>>(Rows==2?pair*16:0);}}

  if constexpr(Tokens>=5 && (Tokens==8 || PositionMajor) && Rows==4 && D::Two) {
   // A complete dual-scale window shares each token's activation across
   // all four output rows. Retire its two integer accumulators immediately;
   // shorter native packets retain their original row-major schedule.
  const auto prepare=[&]<int R>(){typename D::Raw w{};w.meta=((R<2?ds.x:ds.y)>>(16*(R%2)))&65535;w.codes=make_uint2(field<R>(c0),field<R>(c1));
   if constexpr(F==GGML_TYPE_IQ2_XXS||F==GGML_TYPE_IQ2_S)w.signs=field<R>(c1);
   if constexpr(F==GGML_TYPE_IQ2_XS)w.meta|=((scales>>(8*R))&255)<<16;
   else if constexpr(F==GGML_TYPE_IQ2_S)w.meta|=(((R<2?meta.x:meta.y)>>(16*(R%2)))&65535)<<16;
   else if constexpr(F==GGML_TYPE_IQ3_XXS)w.signs=field<R>(signs);
   else if constexpr(F==GGML_TYPE_IQ3_S){w.signs=field<R>(signs);w.meta|=((1+2*((wide_scales>>(8*R+4*(part&1)))&15))<<16)|(((high>>(8*R))&255)<<24);}

   return w;
  };
  typename D::Raw weight[4]={prepare.template operator()<0>(),prepare.template operator()<1>(),
      prepare.template operator()<2>(),prepare.template operator()<3>()};
  int2 decoded[4][4];
  #pragma unroll
  for(int r=0;r<4;++r) {
   decoded[r][0]=D::template decode<0>(weight[r]);decoded[r][1]=D::template decode<1>(weight[r]);
   decoded[r][2]=D::template decode<2>(weight[r]);decoded[r][3]=D::template decode<3>(weight[r]);
  }
  // Two independent query chains hide integer-dot latency without the
  // larger activation working set required by a four-query instruction tile.
  constexpr int Pair=2;
  #pragma unroll
  for(int m=0;m<Tokens;m+=Pair) {
   PackedRowCodes activation[Pair];float scale[Pair];
   #pragma unroll
   for(int c=0;c<Pair;++c) if(m+c<Tokens) {
    if constexpr(PositionMajor) {
     activation[c]=load_row_vector<Tokens>(input,groups,m+c,i);
     scale[c]=__low2float(row_packet_scales<Tokens>(input,groups,m+c,i));
    } else {
     const auto* x=static_cast<const block_q8_1*>(input)+Index(m+c)*groups+i;
     #pragma unroll
     for(int j=0;j<8;++j)activation[c].value[j]=get_int_b4(x->qs,j);
     scale[c]=__low2float(x->ds);
    }
   }
   #pragma unroll
   for(int r=0;r<4;++r) {
    int lo[Pair]{},hi[Pair]{};
    const auto consume=[&]<int J>() {
     #pragma unroll
     for(int c=0;c<Pair;++c) if(m+c<Tokens) {
      const auto& a=activation[c].value;
      if constexpr(J<2)lo[c]=ggml_cuda_dp4a(decoded[r][J].x,a[2*J],lo[c]);
      else hi[c]=ggml_cuda_dp4a(decoded[r][J].x,a[2*J],hi[c]);
     }
     #pragma unroll
     for(int c=0;c<Pair;++c) if(m+c<Tokens) {
      const auto& a=activation[c].value;
      if constexpr(J<2)lo[c]=ggml_cuda_dp4a(decoded[r][J].y,a[2*J+1],lo[c]);
      else hi[c]=ggml_cuda_dp4a(decoded[r][J].y,a[2*J+1],hi[c]);
     }
    };
    consume.template operator()<0>();consume.template operator()<1>();
    consume.template operator()<2>();consume.template operator()<3>();
    #pragma unroll
    for(int c=0;c<Pair;++c)if(m+c<Tokens)sums[m+c][r]+=D::finish(weight[r],lo[c],hi[c],scale[c]);
   }
   asm volatile("" ::: "memory");
  }
  } else {
  float activation_scales[Tokens];
  if constexpr(VectorTokens) {
   static_assert(PositionMajor && (Tokens==3 || Tokens==4) && QuadIqDot<F,0,true>::Two);
   const auto v=__ldg(reinterpret_cast<const float4*>(static_cast<const uint8_t*>(input)+int64_t(columns)*Capacity)+i);
   const float values[4]={v.x,v.y,v.z,v.w};
   #pragma unroll
   for(int m=0;m<Tokens;++m)activation_scales[m]=values[m];
  } else {
   #pragma unroll
   for(int m=0;m<Tokens;++m) {
    if constexpr(PositionMajor)activation_scales[m]=__ldg(reinterpret_cast<const float*>(static_cast<const uint8_t*>(input)+int64_t(columns)*Capacity)+i*Capacity+m);
    else activation_scales[m]=__low2float((static_cast<const block_q8_1*>(input)+Index(m)*groups+i)->ds);
   }
  }
  const auto apply=[&]<int R>(){typename D::Raw w{};w.meta=((R<2?ds.x:ds.y)>>(16*(R%2)))&65535;w.codes=make_uint2(field<R>(c0),field<R>(c1));
   if constexpr(F==GGML_TYPE_IQ2_XXS||F==GGML_TYPE_IQ2_S)w.signs=field<R>(c1);
   if constexpr(F==GGML_TYPE_IQ2_XS)w.meta|=((scales>>(8*R))&255)<<16;
   else if constexpr(F==GGML_TYPE_IQ2_S)w.meta|=(((R<2?meta.x:meta.y)>>(16*(R%2)))&65535)<<16;
   else if constexpr(F==GGML_TYPE_IQ3_XXS)w.signs=field<R>(signs);
   else if constexpr(F==GGML_TYPE_IQ3_S){w.signs=field<R>(signs);w.meta|=((1+2*((wide_scales>>(8*R+4*(part&1)))&15))<<16)|(((high>>(8*R))&255)<<24);}

   int lo[Tokens]{},hi[Tokens]{};
   const auto dot=[&]<int J>() {
    const auto v=D::template decode<J>(w);
    int word0[Capacity]{},word1[Capacity]{};
    if constexpr(VectorTokens) {
     const auto a=token_word4(input,Index(2*J)*groups+i),b=token_word4(input,Index(2*J+1)*groups+i);
     word0[0]=int(a.x);word0[1]=int(a.y);word0[2]=int(a.z);word0[3]=int(a.w);
     word1[0]=int(b.x);word1[1]=int(b.y);word1[2]=int(b.z);word1[3]=int(b.w);
    }
    #pragma unroll
    for(int m=0;m<Tokens;++m) {
     int x0,x1;
     if constexpr(VectorTokens) {x0=word0[m];x1=word1[m];}
     else if constexpr(PositionMajor) {
      x0=__ldg(static_cast<const int*>(input)+(Index(2*J)*groups+i)*Capacity+m);
      x1=__ldg(static_cast<const int*>(input)+(Index(2*J+1)*groups+i)*Capacity+m);
     }else {
      const auto* x=static_cast<const block_q8_1*>(input)+Index(m)*groups+i;
      x0=get_int_b4(x->qs,2*J);x1=get_int_b4(x->qs,2*J+1);
     }
     if constexpr(D::Two && J>=2)hi[m]=ggml_cuda_dp4a(v.y,x1,ggml_cuda_dp4a(v.x,x0,hi[m]));
     else lo[m]=ggml_cuda_dp4a(v.y,x1,ggml_cuda_dp4a(v.x,x0,lo[m]));
    }
   };
   dot.template operator()<0>();dot.template operator()<1>();dot.template operator()<2>();dot.template operator()<3>();
   #pragma unroll
   for(int m=0;m<Tokens;++m)sums[m][R]+=D::finish(w,lo[m],hi[m],activation_scales[m]);
  };
  apply.template operator()<0>();apply.template operator()<1>();
  if constexpr(Rows==4){apply.template operator()<2>();apply.template operator()<3>();}
  }
 }
 #pragma unroll
 for(int m=0;m<Tokens;++m) {
  const float value=warp_sums::reduce<Rows>(sums[m],lane);
  const int r=warp_sums::index<Rows>(lane);
  if(lane==warp_sums::owner<Rows>(r) && row+r<unsigned(rows))
   store_projection(output,int64_t(m)*rows+row+r,value,half_output);
 }
}
template<int F,int Rows,int Tokens,bool PositionMajor,class Index,bool VectorTokens=false>
__global__ void __launch_bounds__(128) quad_small_m_kernel(QuadProjection p,int columns,const void* input,bool half_output) {
 quad_small_m_body<F,Rows,Tokens,PositionMajor,Index,VectorTokens>(p,columns,input,half_output,blockIdx.x);
}

// Preserve the ordinary launch contract for shorter and dual-accumulator
// windows. The full single-accumulator tile benefits from scheduling four
// CTAs within the compiler's target register budget, without extra loads.
template<int F,int Rows,int Tokens,bool PositionMajor,class Index>
__global__ void __launch_bounds__(128,4) quad_single_accumulator_window(QuadProjection p,int columns,const void* input,bool half_output) {
 static_assert(Tokens==8 && !QuadIqDot<F,0,true>::Two);
 quad_small_m_body<F,Rows,Tokens,PositionMajor,Index>(p,columns,input,half_output,blockIdx.x);
}
} // namespace


namespace {
struct SmallGroupEntry {
    const void* packed;
    void* output;
    const void* input;
    int64_t storage_rows,row_begin;
    int rows,type;
    unsigned begin,end,flags;
};
struct SmallGroupPacket { SmallGroupEntry entries[8]; int count; };


template<int F,int M> __device__ __forceinline__ void group_compute(
        const SmallGroupEntry& e,int columns,bool half_output,unsigned tile) {
    const QuadProjection projection{{e.packed,0,e.rows,e.output,static_cast<ggml_type>(F)},
        e.storage_rows,e.row_begin,true};
    if constexpr(F==GGML_TYPE_IQ1_S || F==GGML_TYPE_IQ1_M) {
        quad_small_m_iq1_body<F==GGML_TYPE_IQ1_M,M,uint64_t,(M>=5)>(projection,columns,e.input,half_output,tile);
    } else if constexpr(F==GGML_TYPE_Q2_K) {
        quad_small_m_q2_body<M,uint64_t,(M>=5)>(projection,columns,e.input,half_output,tile);
    } else {
        constexpr bool PositionMajor=M>=3 && QuadIqDot<F,0,true>::Two;
        constexpr int Rows=(M==2 && F==GGML_TYPE_IQ3_S)?2:4;
        if constexpr(PositionMajor && (M==3 || M==4)) {
            if(e.flags&2) {
                quad_small_m_body<F,Rows,M,true,uint64_t,true>(projection,columns,e.input,half_output,tile);
                return;
            }
        }
        if constexpr(M==8 && (F==GGML_TYPE_IQ3_XXS || F==GGML_TYPE_IQ3_S)) {
            if(e.flags&4) {
                quad_small_m_body<F,Rows,M,false,uint32_t>(projection,columns,e.input,half_output,tile);
                return;
            }
        }
        quad_small_m_body<F,Rows,M,PositionMajor,uint64_t>(projection,columns,e.input,half_output,tile);
    }
}

template<int M> __global__ void __launch_bounds__(128,4)
group_small_m_kernel(SmallGroupPacket packet,int columns,bool half_output) {
    int index=0;
    while(index+1<packet.count && blockIdx.x>=packet.entries[index].end)++index;
    const auto& entry=packet.entries[index];
    const unsigned tile=blockIdx.x-entry.begin;
    switch(entry.type) {
#define FORMAT(F) case GGML_TYPE_##F:group_compute<GGML_TYPE_##F,M>(entry,columns,half_output,tile);break;
        FORMAT(IQ1_S) FORMAT(IQ1_M) FORMAT(Q2_K)
        FORMAT(IQ2_XXS) FORMAT(IQ2_XS) FORMAT(IQ2_S) FORMAT(IQ3_XXS) FORMAT(IQ3_S)
#undef FORMAT
        default:break;
    }
}
}

namespace {
template<int F> __device__ __forceinline__ void pair_m2_compute(
        QuadProjection p,int columns,const void* input,bool half_output,unsigned tile) {
    if constexpr(F==GGML_TYPE_IQ1_S || F==GGML_TYPE_IQ1_M)
        quad_small_m_iq1_body<F==GGML_TYPE_IQ1_M,2,uint64_t>(p,columns,input,half_output,tile);
    else if constexpr(F==GGML_TYPE_Q2_K)
        quad_small_m_q2_body<2,uint64_t>(p,columns,input,half_output,tile);
    else quad_small_m_body<F,F==GGML_TYPE_IQ3_S?2:4,2,false,uint64_t>(p,columns,input,half_output,tile);
}
template<int A,int B> __global__ void __launch_bounds__(128,8)
pair_m2_kernel(QuadProjection a,QuadProjection b,int columns,const void* input,bool half_output,unsigned split) {
    if(blockIdx.x<split)pair_m2_compute<A>(a,columns,input,half_output,blockIdx.x);
    else pair_m2_compute<B>(b,columns,input,half_output,blockIdx.x-split);
}
}
bool supports_quad_small_m(ggml_type type,int64_t tokens) noexcept {
    if(tokens<2 || tokens>8)return false;
    switch(type) {
        case GGML_TYPE_IQ1_S:case GGML_TYPE_IQ1_M:case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_XXS:case GGML_TYPE_IQ2_XS:case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:case GGML_TYPE_IQ3_S:return true;
        default:return false;
    }
}
void matvec_quad_small_m(const QuadProjection& p,int64_t columns,const void* input,
        int64_t tokens,bool position_major,bool half_output,cudaStream_t stream) {
    if(!supports_quad_small_m(p.projection.type,tokens) || !p.quad ||
            (position_major && (tokens>=5
                ? !(p.projection.type==GGML_TYPE_IQ2_XXS || p.projection.type==GGML_TYPE_IQ2_XS || p.projection.type==GGML_TYPE_IQ2_S || p.projection.type==GGML_TYPE_IQ1_S || p.projection.type==GGML_TYPE_IQ1_M || p.projection.type==GGML_TYPE_Q2_K)
                : (p.projection.type==GGML_TYPE_IQ1_S || p.projection.type==GGML_TYPE_IQ1_M || p.projection.type==GGML_TYPE_Q2_K))) ||
            (position_major && tokens>=5 && (reinterpret_cast<uintptr_t>(input)&15)) ||
            !p.projection.packed || !p.projection.output || !input || p.row_begin<0 ||
            p.projection.rows<1 || p.projection.rows>INT_MAX || columns<1 || columns>INT_MAX || columns%256 ||
            p.row_begin%4 || p.projection.rows%4 || p.storage_rows<4 ||
            p.projection.rows>p.storage_rows/4*4 || p.row_begin>p.storage_rows/4*4-p.projection.rows)
        throw std::invalid_argument("Invalid direct quad small-M projection");
    const auto format=[&]<int F>() {
        const auto token=[&]<int M>() {
            // Four output rows keep activation and codebook traffic shared
            // across the full verification window, including M=5..8.
            // IQ3_S carries signs, high bits and scales across its row tile.
            // At two tokens, halving that tile shortens their live ranges
            // while preserving the same lossless four-row weight planes.
            constexpr int Rows=(M==2 && F==GGML_TYPE_IQ3_S)?2:4,Outputs=4*Rows;
            const auto layout=[&]<bool PositionMajor>() {
                const bool narrow_offsets=[&] {
                    if constexpr(M==8 && (F==GGML_TYPE_IQ3_XXS || F==GGML_TYPE_IQ3_S))
                        return quad_storage_bytes(p.projection.type,p.storage_rows,columns)<=UINT32_MAX;
                    else return false;
                }();
                const auto function=[&] {
                    if constexpr(F==GGML_TYPE_IQ1_S || F==GGML_TYPE_IQ1_M) {
                        if constexpr(M>=5)if(position_major)return quad_small_m_iq1_kernel<F==GGML_TYPE_IQ1_M,M,uint64_t,true>;
                        return quad_small_m_iq1_kernel<F==GGML_TYPE_IQ1_M,M,uint64_t>;
                    }
                    else if constexpr(F==GGML_TYPE_Q2_K) {
                        if constexpr(M>=5)if(position_major)return quad_split_q2_kernel<M,uint64_t>;
                        if constexpr(M>=6)return quad_wide_q2_kernel<M,uint64_t>;
                        else return quad_small_m_q2_kernel<M,uint64_t>;
                    }
                    else if constexpr(M==8 && !QuadIqDot<F,0,true>::Two) {
                        // Keep the device base address 64-bit. Only the byte
                        // offsets use MAD.WIDE.U32 when the entire storage
                        // image fits; large allocations retain 64-bit offsets.
                        if(narrow_offsets)
                            return quad_single_accumulator_window<F,Rows,M,PositionMajor,uint32_t>;
                        return quad_single_accumulator_window<F,Rows,M,PositionMajor,uint64_t>;
                    }
                    else if constexpr(PositionMajor && (M==3 || M==4) && QuadIqDot<F,0,true>::Two) {
                        // uint4/float4 loads require 16-byte input alignment.
                        // Offset views retain the scalar packet reader.
                        if((reinterpret_cast<uintptr_t>(input)&15)==0)
                            return quad_small_m_kernel<F,Rows,M,PositionMajor,uint64_t,true>;
                        return quad_small_m_kernel<F,Rows,M,PositionMajor,uint64_t>;
                    }
                    else return quad_small_m_kernel<F,Rows,M,PositionMajor,uint64_t>;
                }();
                const unsigned shared_bytes=(F==GGML_TYPE_IQ1_S
                    ? Iq1SSequentialGridCount : GridCount<F>+SignCount<F>)*4;
                function<<<unsigned((p.projection.rows+Outputs-1)/Outputs),dim3(32,4),shared_bytes,stream>>>(p,int(columns),input,half_output);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gguf_quad_small_m",
                    reinterpret_cast<const void*>(function),"format="+std::to_string(F)+" M="+std::to_string(M)+
                    " shared_weight_decode=1 canonical_restore=0 output_rows="+std::to_string(Rows)+
                    " position_major="+std::to_string(position_major)+
                    " vector_token_loads="+std::to_string(PositionMajor && (M==3 || M==4) && QuadIqDot<F,0,true>::Two && (reinterpret_cast<uintptr_t>(input)&15)==0)+
                    " query_instruction_tile="+std::to_string(M>=5 && (M==8 || position_major) && QuadIqDot<F,0,true>::Two?2:1)+
                    " split_q8_planes="+std::to_string(M>=5 && position_major)+
                    " packed_affine_sums="+std::to_string(F==GGML_TYPE_Q2_K && M>=5 && position_major)+
                    " preweighted_codes="+std::to_string(F==GGML_TYPE_Q2_K && (M>=6 || position_major))+
                    " byte_offset_bits="+std::to_string(narrow_offsets?32:64));
            };
            if constexpr(M>=5 && QuadIqDot<F,0,true>::Two) {
                if(position_major)layout.template operator()<true>();else layout.template operator()<false>();
            } else if constexpr(M>4)layout.template operator()<false>();
            else if(position_major)layout.template operator()<true>();else layout.template operator()<false>();
        };
        switch(tokens) {
            case 2:token.template operator()<2>();break;case 3:token.template operator()<3>();break;
            case 4:token.template operator()<4>();break;case 5:token.template operator()<5>();break;
            case 6:token.template operator()<6>();break;case 7:token.template operator()<7>();break;
            default:token.template operator()<8>();break;
        }
    };
    switch(p.projection.type) {
#define FORMAT(F) case GGML_TYPE_##F:format.template operator()<GGML_TYPE_##F>();break;
        FORMAT(IQ1_S) FORMAT(IQ1_M) FORMAT(Q2_K)
        FORMAT(IQ2_XXS) FORMAT(IQ2_XS) FORMAT(IQ2_S) FORMAT(IQ3_XXS) FORMAT(IQ3_S)
#undef FORMAT
        default:break;
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}

namespace {
template<int F,int M,bool Narrow> __device__ __forceinline__ void compute_pair_tile(
        QuadProjection projection,int columns,const void* native,const void* interleaved,
        bool half_output,unsigned tile) {
    if constexpr(F==GGML_TYPE_IQ1_S || F==GGML_TYPE_IQ1_M)
        quad_small_m_iq1_body<F==GGML_TYPE_IQ1_M,M,uint64_t,(M>=5)>(projection,columns,M>=5?interleaved:native,half_output,tile);
    else if constexpr(F==GGML_TYPE_Q2_K)
        quad_small_m_q2_body<M,uint64_t,(M>=5)>(projection,columns,M>=5?interleaved:native,half_output,tile);
    else {
        constexpr bool split=QuadIqDot<F,0,true>::Two;
        if constexpr(M==3 || M==4)
            quad_small_m_body<F,4,M,split,uint64_t,split>(projection,columns,split?interleaved:native,half_output,tile);
        else if constexpr(!split && Narrow)
            quad_small_m_body<F,4,M,false,uint32_t>(projection,columns,native,half_output,tile);
        else quad_small_m_body<F,4,M,split,uint64_t>(projection,columns,split?interleaved:native,half_output,tile);
    }
}
template<int A,int B,int M,bool Narrow> __global__ void __launch_bounds__(128,4)
typed_packet_pair(QuadProjection a,QuadProjection b,int columns,const void* native,
        const void* interleaved,bool half_output,unsigned split) {
    if(blockIdx.x<split)compute_pair_tile<A,M,Narrow>(a,columns,native,interleaved,half_output,blockIdx.x);
    else compute_pair_tile<B,M,Narrow>(b,columns,native,interleaved,half_output,blockIdx.x-split);
}
}

namespace {
void launch_small_m_pair(const QuadProjection& a,const QuadProjection& b,
        int columns,const void* input,bool half_output,cudaStream_t stream) {
    const auto launch=[&]<int A,int B>() {
        constexpr unsigned a_rows=A==GGML_TYPE_IQ3_S?8:16;
        constexpr unsigned b_rows=B==GGML_TYPE_IQ3_S?8:16;
        const unsigned split=unsigned((a.projection.rows+a_rows-1)/a_rows);
        const unsigned total=split+unsigned((b.projection.rows+b_rows-1)/b_rows);
        constexpr unsigned a_shared=A==GGML_TYPE_IQ1_S?Iq1SSequentialGridCount:GridCount<A>+SignCount<A>;
        constexpr unsigned b_shared=B==GGML_TYPE_IQ1_S?Iq1SSequentialGridCount:GridCount<B>+SignCount<B>;
        constexpr unsigned shared=4*(a_shared>b_shared?a_shared:b_shared);
        const auto fn=pair_m2_kernel<A,B>;
        fn<<<total,dim3(32,4),shared,stream>>>(a,b,columns,input,half_output,split);
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_quad_group_small_m",
            reinterpret_cast<const void*>(fn),"M=2 matrices=2 paired_formats="+
            std::to_string(A)+"/"+std::to_string(B)+" shared_weight_decode=1 canonical_restore=0");
    };
#define SMALL_M_FORMATS(X) X(IQ1_S) X(IQ1_M) X(Q2_K) X(IQ2_XXS) X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S)
    const auto second=[&]<int A>() {
        switch(b.projection.type) {
#define FORMAT(F) case GGML_TYPE_##F:launch.template operator()<A,GGML_TYPE_##F>();break;
            SMALL_M_FORMATS(FORMAT)
#undef FORMAT
            default:throw std::invalid_argument("Unsupported paired small-M format");
        }
    };
    switch(a.projection.type) {
#define FORMAT(F) case GGML_TYPE_##F:second.template operator()<GGML_TYPE_##F>();break;
        SMALL_M_FORMATS(FORMAT)
#undef FORMAT
        default:throw std::invalid_argument("Unsupported paired small-M format");
    }
#undef SMALL_M_FORMATS
}
}

void matvec_quad_group_small_m(const QuadProjection* projections,size_t count,int64_t columns,
        const void* native,const void* interleaved,const uint8_t* position_major,int64_t tokens,
        bool half_output,cudaStream_t stream) {
    if(!count)return;
    if(tokens<2 || tokens>8 || !projections || !position_major ||
            columns<256 || columns>INT_MAX || columns%256)
        throw std::invalid_argument("Invalid grouped small-M projection geometry");
    // Eight descriptors fit comfortably in the kernel parameter packet. More
    // projections use another packet; this is independent of model geometry.
    for(size_t first=0;first<count;first+=8) {
        SmallGroupPacket packet{};unsigned total=0;
        const size_t end=first+std::min(size_t(8),count-first);
        for(size_t index=first;index<end;++index) {
            const auto& q=projections[index];const auto& d=q.projection;
            if(!supports_quad_small_m(d.type,tokens) || !q.quad || !d.packed || !d.output ||
                    d.rows<1 || d.rows>INT_MAX || q.storage_rows<4 || q.row_begin<0 ||
                    q.row_begin%4 || d.rows%4 || q.row_begin>q.storage_rows/4*4-d.rows)
                throw std::invalid_argument("Invalid grouped small-M row view");
            const auto bytes=quad_storage_bytes(d.type,q.storage_rows,columns);
            const bool split=position_major[index]!=0;
            const bool iq2=d.type==GGML_TYPE_IQ2_XXS || d.type==GGML_TYPE_IQ2_XS || d.type==GGML_TYPE_IQ2_S;
            const bool offset=d.type==GGML_TYPE_IQ1_S || d.type==GGML_TYPE_IQ1_M || d.type==GGML_TYPE_Q2_K;
            // Consume the existing shared activation producer's native,
            // word-major or split-plane ABI, without another quantization.
            if(split != ((tokens>=3 && iq2) || (tokens>=5 && offset)))
                throw std::invalid_argument("Grouped small-M activation layout differs");
            const auto* input=split?interleaved:native;
            if(!input || (reinterpret_cast<uintptr_t>(input)&3) ||
                    (split && tokens>=5 && (reinterpret_cast<uintptr_t>(input)&15)))
                throw std::invalid_argument("Invalid grouped small-M activation alignment");
            const unsigned rows=(tokens==2 && d.type==GGML_TYPE_IQ3_S)?8:16;
            const unsigned next=total+unsigned((d.rows+rows-1)/rows);
            packet.entries[packet.count++]={d.packed,d.output,input,q.storage_rows,q.row_begin,
                int(d.rows),int(d.type),total,next,unsigned(split?1:0)|
                unsigned((reinterpret_cast<uintptr_t>(input)&15)==0?2:0)|unsigned(bytes<=UINT32_MAX?4:0)};
            total=next;
        }
        if(tokens==2) {
            for(size_t i=first;i<end;i+=2) {
                if(i+1<end)launch_small_m_pair(projections[i],projections[i+1],int(columns),native,half_output,stream);
                else matvec_quad_small_m(projections[i],columns,native,tokens,false,half_output,stream);
            }
            continue;
        }
        const auto launch=[&]<int M>() {
            const auto fn=group_small_m_kernel<M>;
            fn<<<total,dim3(32,4),8192,stream>>>(packet,int(columns),half_output);
            if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_quad_group_small_m",
                reinterpret_cast<const void*>(fn),"M="+std::to_string(M)+
                " matrices="+std::to_string(packet.count)+" shared_weight_decode=1 canonical_restore=0");
        };
        switch(tokens) {
            case 3:launch.template operator()<3>();break;
            case 4:launch.template operator()<4>();break;
            case 5:launch.template operator()<5>();break;
            case 6:launch.template operator()<6>();break;
            case 7:launch.template operator()<7>();break;
            default:launch.template operator()<8>();break;
        }
    }
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
void matvec_quad_pair_small_m(const QuadProjection* projections,size_t count,int64_t columns,
        const void* native,const void* interleaved,const uint8_t* position_major,int64_t tokens,
        bool half_output,cudaStream_t stream) {
    if(count!=2 || !projections || !position_major ||
            (tokens<3 || tokens>8) || columns<256 || columns>INT_MAX || columns%256)
        throw std::invalid_argument("Invalid typed packet pair");
    bool narrow=true;
    for(int i=0;i<2;++i) {
        const auto& q=projections[i];const auto& d=q.projection;
        if(!q.quad || !d.packed || !d.output || d.rows<1 || d.rows>INT_MAX ||
                q.row_begin<0 || q.row_begin%4 || d.rows%4 || q.row_begin>q.storage_rows/4*4-d.rows)
            throw std::invalid_argument("Invalid typed pair row view");
        const auto bytes=quad_storage_bytes(d.type,q.storage_rows,columns);
        const bool iq3=d.type==GGML_TYPE_IQ3_S || d.type==GGML_TYPE_IQ3_XXS;
        if(iq3 && bytes>UINT32_MAX)narrow=false;
        const bool iq2=d.type==GGML_TYPE_IQ2_XXS || d.type==GGML_TYPE_IQ2_XS || d.type==GGML_TYPE_IQ2_S;
        const bool offset=d.type==GGML_TYPE_IQ1_S || d.type==GGML_TYPE_IQ1_M || d.type==GGML_TYPE_Q2_K;
        const bool split=iq2 || (tokens>=5 && offset);
        const auto* input=split?interleaved:native;
        if(split!=position_major[i] || !input || (reinterpret_cast<uintptr_t>(input)&3) ||
                (split && tokens>=5 && (reinterpret_cast<uintptr_t>(input)&15)))
            throw std::invalid_argument("Typed pair activation layout/alignment differs");
    }
    if((tokens==3 || tokens==4) && interleaved && (reinterpret_cast<uintptr_t>(interleaved)&15)) {
        // The scalar packet reader preserves valid four-byte offset views.
        matvec_quad_group_small_m(projections,count,columns,native,interleaved,position_major,tokens,half_output,stream);
        return;
    }
    const unsigned split=unsigned((projections[0].projection.rows+15)/16);
    const unsigned total=split+unsigned((projections[1].projection.rows+15)/16);
    const auto launch=[&]<int A,int B,int M,bool Narrow>() {
        constexpr unsigned sa=A==GGML_TYPE_IQ1_S?Iq1SSequentialGridCount:GridCount<A>+SignCount<A>;
        constexpr unsigned sb=B==GGML_TYPE_IQ1_S?Iq1SSequentialGridCount:GridCount<B>+SignCount<B>;
        constexpr unsigned bytes=4*(sa>sb?sa:sb);
        const auto fn=typed_packet_pair<A,B,M,Narrow>;
        fn<<<total,dim3(32,4),bytes,stream>>>(projections[0],projections[1],int(columns),native,interleaved,half_output,split);
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_quad_pair_small_m",
            reinterpret_cast<const void*>(fn),"M="+std::to_string(M)+" paired_formats="+
            std::to_string(A)+"/"+std::to_string(B)+" iq3_offset_bits="+std::to_string(Narrow?32:64)+
            " shared_weight_decode=1 canonical_restore=0");
    };
#define PAIR_FORMATS(X) X(IQ1_S) X(IQ1_M) X(Q2_K) X(IQ2_XXS) X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S)
    const auto second=[&]<int A>() {
        const auto run=[&]<int B>() {
            if(tokens==3)launch.template operator()<A,B,3,false>();
            else if(tokens==4)launch.template operator()<A,B,4,false>();
            else if(tokens==5)launch.template operator()<A,B,5,false>();
            else if(tokens==6)launch.template operator()<A,B,6,false>();
            else if(tokens==7)launch.template operator()<A,B,7,false>();
            else if(narrow)launch.template operator()<A,B,8,true>();
            else launch.template operator()<A,B,8,false>();
        };
        switch(projections[1].projection.type) {
#define FORMAT(F) case GGML_TYPE_##F:run.template operator()<GGML_TYPE_##F>();break;
            PAIR_FORMATS(FORMAT)
#undef FORMAT
            default:throw std::invalid_argument("Unsupported typed packet pair");
        }
    };
    switch(projections[0].projection.type) {
#define FORMAT(F) case GGML_TYPE_##F:second.template operator()<GGML_TYPE_##F>();break;
        PAIR_FORMATS(FORMAT)
#undef FORMAT
        default:throw std::invalid_argument("Unsupported typed packet pair");
    }
#undef PAIR_FORMATS
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace mfq::cuda::gguf
