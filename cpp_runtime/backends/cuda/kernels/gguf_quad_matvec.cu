#include "gguf_quad_matvec.h"
#include "gguf_quad_storage.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "gguf_quantize.cuh"
#include "gguf_dense_rows.cuh"
#include "gguf_m1_residual.h"
#include "kernel_dispatch_trace.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
namespace mfq::cuda::gguf {namespace{
#include "gguf_dense_m1_aligned.cuh"
#include "gguf_quad_dot.cuh"

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
#include "gguf_quad_q2_m1.cuh"
template<bool M,int Rows,int W,int Offset,bool Aligned,bool Half,class Index,class Store=M1ProjectionStore>__device__ __forceinline__ void global_quad_body(QuadProjection p,int columns,const void* input,unsigned tile,Store store={}){
 const auto* packed=static_cast<const uint8_t*>(p.projection.packed);const int rows=int(p.projection.rows);void* output=p.projection.output;

 using F=QuadIq1Dot<M,Offset>;
 const int lane=threadIdx.x,warp=threadIdx.y,tid=warp*32+lane;

 const unsigned row=(tile*(4/W)+warp/W)*Rows;
 const Index source_row=Index(row)+Index(p.row_begin),quad=source_row/4,pair=(source_row%4)/2,blocks=Index(columns/256),groups=blocks*8;
 const Index quad_blocks=(Index(p.storage_rows)/4)*blocks;
 float sums[Rows]{};
 for(unsigned i=(warp%W)*32+lane;warp<(4/W)*W&&row<unsigned(rows)&&i<groups;i+=32*W){
  const Index qb=quad*blocks+i/8,entry=quad*groups+i,part=i%8;uint4 code{};uint2 high{},ds{};unsigned local=0;
  if constexpr(Rows==4){code=load_four(packed,entry*16);high=load_two(packed,quad_blocks*128+entry*8);
   if constexpr(M){
    const Index base=quad_blocks*192+qb*32,byte=3*part,aligned=byte&~Index(3);
    const auto lo=load_one(packed,base+aligned),hi=load_one(packed,base+aligned+4);
    local=__funnelshift_r(lo,hi,unsigned(8*(byte&3)))&0xffffffu;ds=load_two(packed,base+24);
   }else ds=load_two(packed,quad_blocks*192+qb*8);
  }else{const auto c=load_two(packed,entry*16+pair*8);code=make_uint4(c.x,c.y,0,0);high=make_uint2(load_one(packed,quad_blocks*128+entry*8+pair*4),0);
   if constexpr(M){
    const Index base=quad_blocks*192+qb*32,byte=3*part,aligned=byte&~Index(3);
    const auto lo=load_one(packed,base+aligned),hi=load_one(packed,base+aligned+4);
    local=(__funnelshift_r(lo,hi,unsigned(8*(byte&3)))>>(12*pair))&4095u;
    ds=make_uint2(load_one(packed,base+24+pair*4),0);
   }else ds=make_uint2(load_one(packed,quad_blocks*192+qb*8+pair*4),0);
  }
  typename F::Activation x;
  if constexpr(Aligned){const auto* codes=static_cast<const uint4*>(input);const auto a=__ldg(codes+2*i),b=__ldg(codes+2*i+1);const auto bits=__ldg(reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(input)+columns)+i);
   x.codes[0]=a.x;x.codes[1]=a.y;x.codes[2]=a.z;x.codes[3]=a.w;x.codes[4]=b.x;x.codes[5]=b.y;x.codes[6]=b.z;x.codes[7]=b.w;x.ds=make_float2(__half2float(__ushort_as_half(bits)),__half2float(__ushort_as_half(bits>>16)));
  }else{const auto* a=static_cast<const block_q8_1*>(input)+i;x.ds=__half22float2(a->ds);
   #pragma unroll
   for(int j=0;j<8;++j)x.codes[j]=get_int_b4(a->qs,j);
  }
  F::prepare_activation(x);
  const auto apply=[&]<int R>(){typename F::Weight w;w.codes=field<R>(code);w.high=(R<2?high.x:high.y)>>(16*(R%2));
   if constexpr(M){w.scale=__half2float(__ushort_as_half((R<2?ds.x:ds.y)>>(16*(R%2))));w.scales=(local>>(6*R))&63;}
   else{w.scale=__half2float(__ushort_as_half((R<2?ds.x:ds.y)>>(16*(R%2))))*(((w.high>>11)&14)+1);w.scales=0;}
   sums[R]+=F::apply_activation(w,x);
  };
  apply.template operator()<0>();apply.template operator()<1>();if constexpr(Rows==4){apply.template operator()<2>();apply.template operator()<3>();}
 }
 const float value=warp_sums::reduce<Rows>(sums,lane);const int r=warp_sums::index<Rows>(lane);const bool owner=lane==warp_sums::owner<Rows>(r);__shared__ float scratch[16];
 if constexpr(W==1){if(owner&&row+r<unsigned(rows))store(output,row+r,value,Half);}
 else{if(owner&&warp<(4/W)*W)scratch[warp*Rows+r]=value;__syncthreads();if(owner&&warp<(4/W)*W&&warp%W==0&&row+r<unsigned(rows)){float sum=scratch[warp*Rows+r];
  #pragma unroll
  for(int k=1;k<W;++k)sum+=scratch[(warp+k)*Rows+r];store(output,row+r,sum,Half);}}
}
template<int F,int Rows,int W,int Offset,bool Aligned,bool Half,class Index,class Store=M1ProjectionStore>__device__ __forceinline__ void global_iq_body(QuadProjection p,int columns,const void* input,unsigned tile,Store store={}){
 const auto* packed=static_cast<const uint8_t*>(p.projection.packed);const int rows=int(p.projection.rows);void* output=p.projection.output;

 using D=QuadIqDot<F,Offset>;const unsigned lane=threadIdx.x,warp=threadIdx.y;
 const unsigned row=(tile*(4/W)+warp/W)*Rows;
 const Index source_row=Index(row)+Index(p.row_begin),quad=source_row/4,pair=(source_row%4)/2,blocks=Index(columns/256),groups=blocks*8;
 const Index total=(Index(p.storage_rows)/4)*blocks;
 float sums[Rows]{};
 for(unsigned i=(warp%W)*32+lane;warp<(4/W)*W&&row<unsigned(rows)&&i<groups;i+=32*W){
  const Index entry=quad*groups+i,qb=quad*blocks+i/8,part=i%8;uint4 c0{},c1{},signs{};uint2 ds{},meta{};unsigned scales=0,high=0,wide_scales=0;
  if constexpr(Rows==4){c0=load_four(packed,entry*32);c1=load_four(packed,entry*32+16);ds=load_two(packed,total*(256+PlaneOffsets<F>::Extra)+qb*8);}
  else{const auto a=load_two(packed,entry*32+pair*8),b=load_two(packed,entry*32+16+pair*8);c0=make_uint4(a.x,a.y,0,0);c1=make_uint4(b.x,b.y,0,0);ds=make_uint2(load_one(packed,total*(256+PlaneOffsets<F>::Extra)+qb*8+pair*4),0);}
  if constexpr(F==GGML_TYPE_IQ2_XS)scales=load_one(packed,total*256+entry*4)>>(Rows==2?pair*16:0);
  else if constexpr(F==GGML_TYPE_IQ2_S){if constexpr(Rows==4)meta=load_two(packed,total*256+entry*8);else meta=make_uint2(load_one(packed,total*256+entry*8+pair*4),0);}
  else if constexpr(F==GGML_TYPE_IQ3_XXS||F==GGML_TYPE_IQ3_S){if constexpr(Rows==4)signs=load_four(packed,total*256+entry*16);else{const auto s=load_two(packed,total*256+entry*16+pair*8);signs=make_uint4(s.x,s.y,0,0);}
   if constexpr(F==GGML_TYPE_IQ3_S){high=load_one(packed,total*384+entry*4)>>(Rows==2?pair*16:0);wide_scales=load_one(packed,total*416+(entry/2)*4)>>(Rows==2?pair*16:0);}}
  int x[8];float scale;
  if constexpr(Aligned){const auto* a=static_cast<const uint4*>(input);const auto p=__ldg(a+2*i),q=__ldg(a+2*i+1);x[0]=p.x;x[1]=p.y;x[2]=p.z;x[3]=p.w;x[4]=q.x;x[5]=q.y;x[6]=q.z;x[7]=q.w;
   const auto bits=__ldg(reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(input)+columns)+i);scale=__half2float(__ushort_as_half(bits));
  }else{const auto* a=static_cast<const block_q8_1*>(input)+i;
   #pragma unroll
   for(int j=0;j<8;++j)x[j]=get_int_b4(a->qs,j);scale=__low2float(a->ds);
  }
  const auto apply=[&]<int R>(){typename D::Raw w{};w.meta=((R<2?ds.x:ds.y)>>(16*(R%2)))&65535;w.codes=make_uint2(field<R>(c0),field<R>(c1));
   if constexpr(F==GGML_TYPE_IQ2_XXS||F==GGML_TYPE_IQ2_S)w.signs=field<R>(c1);
   if constexpr(F==GGML_TYPE_IQ2_XS)w.meta|=((scales>>(8*R))&255)<<16;
   else if constexpr(F==GGML_TYPE_IQ2_S)w.meta|=(((R<2?meta.x:meta.y)>>(16*(R%2)))&65535)<<16;
   else if constexpr(F==GGML_TYPE_IQ3_XXS)w.signs=field<R>(signs);
   else if constexpr(F==GGML_TYPE_IQ3_S){w.signs=field<R>(signs);w.meta|=((1+2*((wide_scales>>(8*R+4*(part&1)))&15))<<16)|(((high>>(8*R))&255)<<24);}
   int lo=0,hi=0;
   const auto dot=[&]<int J>(){const auto v=D::template decode<J>(w);
    if constexpr(D::Two&&J>=2)hi=ggml_cuda_dp4a(v.y,x[2*J+1],ggml_cuda_dp4a(v.x,x[2*J],hi));else lo=ggml_cuda_dp4a(v.y,x[2*J+1],ggml_cuda_dp4a(v.x,x[2*J],lo));};
   dot.template operator()<0>();dot.template operator()<1>();dot.template operator()<2>();dot.template operator()<3>();
   sums[R]+=D::finish(w,lo,hi,scale);
  };
  apply.template operator()<0>();apply.template operator()<1>();if constexpr(Rows==4){apply.template operator()<2>();apply.template operator()<3>();}
 }
 const float value=warp_sums::reduce<Rows>(sums,lane);const int r=warp_sums::index<Rows>(lane);const bool owner=lane==warp_sums::owner<Rows>(r);__shared__ float scratch[16];
 if constexpr(W==1){if(owner&&row+r<unsigned(rows))store(output,row+r,value,Half);}
 else{if(owner&&warp<(4/W)*W)scratch[warp*Rows+r]=value;__syncthreads();if(owner&&warp<(4/W)*W&&warp%W==0&&row+r<unsigned(rows)){float sum=scratch[warp*Rows+r];
  #pragma unroll
  for(int k=1;k<W;++k)sum+=scratch[(warp+k)*Rows+r];store(output,row+r,sum,Half);}}
}

template<int F>constexpr bool IsIq1=F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M;
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
struct QuadResidualStore {
    const half* residual;
    __device__ void operator()(void* output,int64_t row,float value,bool)const {
        static_cast<half*>(output)[row]=__hadd(residual[row],__float2half_rn(value));
    }
};
template<int F,int R,int W,bool Aligned,bool Half,class Index,class Store>
__device__ void quad_entry(QuadProjection p,int columns,const void* input,unsigned tile,Store store){
    if constexpr(F==GGML_TYPE_Q2_K)quad_q2_m1_body<R,W,Aligned,Half,Index>(p,columns,input,tile,store);
    else if constexpr(IsIq1<F>)global_quad_body<F==GGML_TYPE_IQ1_M,R,W,0,Aligned,Half,Index>(p,columns,input,tile,store);
    else global_iq_body<F,R,W,0,Aligned,Half,Index>(p,columns,input,tile,store);
}
template<int F,int R,int W,bool Aligned,bool Half,bool Residual,class Index>
__global__ void __launch_bounds__((F==GGML_TYPE_Q2_K && W==3)?96:128) quad_single(QuadProjection p,int columns,const void* input,const half* residual){
    if constexpr(GridCount<F>!=0){stage_global_grid<F,0>(threadIdx.y*32+threadIdx.x);__syncthreads();}
    if constexpr(Residual)quad_entry<F,R,W,Aligned,Half,Index>(p,columns,input,blockIdx.x,QuadResidualStore{residual});
    else quad_entry<F,R,W,Aligned,Half,Index>(p,columns,input,blockIdx.x,M1ProjectionStore{});
}
template<int F,int R,int W,bool Aligned,bool Half,class Index>
__device__ void compute_entry(QuadProjection p,int columns,const void* input,unsigned tile){
    if constexpr(GridCount<F>!=0 || F==GGML_TYPE_Q2_K)if(p.quad){quad_entry<F,R,W,Aligned,Half,Index>(p,columns,input,tile,M1ProjectionStore{});return;}
    const auto& d=p.projection;__shared__ float scratch[16];
    if constexpr(Aligned)project_aligned_m1_body<F,R,W>(static_cast<const uint8_t*>(d.packed),d.stride,int(d.rows),columns,
        static_cast<const uint4*>(input),reinterpret_cast<const unsigned*>(static_cast<const uint8_t*>(input)+columns),d.output,Half,tile,scratch);
    else project_dense_reused_q8_body<F,R,W>(static_cast<const uint8_t*>(d.packed),d.stride,int(d.rows),columns,
        static_cast<const block_q8_1*>(input),d.output,Half,tile,scratch);
}
template<int F,bool Aligned,bool Half,class Index,bool Typed=false>
__device__ void pair_entry(QuadProjection p,int columns,const void* input,unsigned tile){
    if constexpr(IsIq1<F>){if(columns/32>=64){compute_entry<F,4,2,Aligned,Half,Index>(p,columns,input,tile);return;}}
    else if constexpr(Typed){if(columns/32>=128){compute_entry<F,2,2,Aligned,Half,Index>(p,columns,input,tile);return;}}
    compute_entry<F,2,1,Aligned,Half,Index>(p,columns,input,tile);
}

template<int F,bool Aligned,bool Half,class Index>
__device__ void quad_pair_entry(QuadProjection p,int columns,const void* input,unsigned tile){
    if constexpr(IsIq1<F>){if(columns/32>=64){quad_entry<F,4,2,Aligned,Half,Index>(p,columns,input,tile,M1ProjectionStore{});return;}}
    else if(columns/32>=128){quad_entry<F,2,2,Aligned,Half,Index>(p,columns,input,tile,M1ProjectionStore{});return;}
    quad_entry<F,2,1,Aligned,Half,Index>(p,columns,input,tile,M1ProjectionStore{});
}
template<int A,int B,bool Aligned,bool Half,class Index>
__global__ void __launch_bounds__(128) quad_pair(QuadProjection a,QuadProjection b,unsigned split,int columns,const void* input){
    const int tid=threadIdx.y*32+threadIdx.x;
    // The host selects this kernel only when both row views use quad storage.
    // Canonical views retain the generic grouped kernel and its dot geometry.
    if(blockIdx.x<split){
        if constexpr(GridCount<A>!=0){stage_global_grid<A,0>(tid);__syncthreads();}
        quad_pair_entry<A,Aligned,Half,Index>(a,columns,input,blockIdx.x);
    }else{
        if constexpr(GridCount<B>!=0){stage_global_grid<B,0>(tid);__syncthreads();}
        quad_pair_entry<B,Aligned,Half,Index>(b,columns,input,blockIdx.x-split);
    }
}
struct QuadGroupEntry{QuadProjection p;unsigned begin,end;int warps=1;};
struct QuadGroupPacket{QuadGroupEntry entries[8];int count;};
template<bool Aligned,bool Half,class Index>
__global__ void __launch_bounds__(128) quad_group(QuadGroupPacket packet,int columns,const void* input){
    int index=0;while(index+1<packet.count&&blockIdx.x>=packet.entries[index].end)++index;
    const auto& e=packet.entries[index];const int tid=threadIdx.y*32+threadIdx.x;
    switch(e.p.projection.type){
#define CASE(F) case GGML_TYPE_##F: { \
    constexpr int Format=GGML_TYPE_##F; \
    if constexpr(GridCount<Format>!=0)if(e.p.quad){stage_global_grid<Format,0>(tid);__syncthreads();} \
    if constexpr(!IsIq1<Format>){ \
        if(e.warps==2)compute_entry<Format,2,2,Aligned,Half,Index>(e.p,columns,input,blockIdx.x-e.begin); \
        else if(e.warps==3)compute_entry<Format,2,3,Aligned,Half,Index>(e.p,columns,input,blockIdx.x-e.begin); \
        else if(e.warps==4)compute_entry<Format,2,4,Aligned,Half,Index>(e.p,columns,input,blockIdx.x-e.begin); \
        else pair_entry<Format,Aligned,Half,Index>(e.p,columns,input,blockIdx.x-e.begin); \
    }else pair_entry<Format,Aligned,Half,Index>(e.p,columns,input,blockIdx.x-e.begin); \
    break;}
    CASE(IQ1_S) CASE(IQ1_M) CASE(IQ2_XXS) CASE(IQ2_XS) CASE(IQ2_S) CASE(IQ3_XXS) CASE(IQ3_S)
    CASE(Q2_K) CASE(Q4_K) CASE(Q5_K) CASE(Q6_K) CASE(IQ4_XS)
#undef CASE
    default:break;
    }
}
void checked_quad(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
void validate_quad(const QuadProjection& p,int64_t columns){
    const auto& d=p.projection;
    if(columns<1||columns>INT_MAX||columns%ggml_blck_size(d.type)||d.rows<0||d.rows>INT_MAX||
       (d.rows&&(!d.packed||!d.output||d.stride<ggml_row_size(d.type,columns))))throw std::invalid_argument("Invalid GGUF compute projection");
    if(p.quad){
        quad_storage_bytes(d.type,p.storage_rows,columns);
        if(p.row_begin<0||p.row_begin%4||d.rows%4||p.row_begin>p.storage_rows/4*4-d.rows||reinterpret_cast<uintptr_t>(d.packed)%16)throw std::invalid_argument("Invalid GGUF quad row view");
    }
}
unsigned grid_words(int type){switch(type){case GGML_TYPE_IQ1_S:return 1024;case GGML_TYPE_IQ1_M:return 2048;case GGML_TYPE_IQ2_XXS:case GGML_TYPE_IQ3_XXS:return 256;case GGML_TYPE_IQ2_XS:case GGML_TYPE_IQ3_S:return 512;case GGML_TYPE_IQ2_S:return 1024;default:return 0;}}
bool small_offsets(const QuadProjection& p,int64_t columns){return !p.quad||quad_storage_bytes(p.projection.type,p.storage_rows,columns)<=UINT32_MAX;}
template<int F>int original_k_cooperation(int64_t rows,int64_t columns,uint64_t group_blocks=0){
    if(columns/32<128)return 1;
    int device=0;checked_quad(cudaGetDevice(&device));
    static thread_local int previous=-1,sms=0,active=0;
    if(device!=previous){checked_quad(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));checked_quad(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,project_dense_reused_q8<F,2,1>,128,0));previous=device;}
    return active>0&&(group_blocks?group_blocks:uint64_t((rows+7)/8))<uint64_t(sms)*unsigned(active)?(columns/32%96==0&&columns/32%128!=0?3:4):1;
}
}
void matvec_quad_m1(const QuadProjection& p,int64_t columns,const void* input,bool aligned,bool half_output,const void* residual,cudaStream_t stream){
    validate_quad(p,columns);const auto& d=p.projection;if(!d.rows)return;
    if(!input||reinterpret_cast<uintptr_t>(input)%(aligned?16:4)||(residual&&(!half_output||reinterpret_cast<uintptr_t>(residual)%2)))throw std::invalid_argument("Invalid GGUF quad activation/residual");
    if(!p.quad){
        if(residual)matvec_m1_residual(d.type,d.packed,d.stride,d.rows,columns,input,aligned,residual,d.output,stream);
        else if(aligned)matvec_m1_aligned(d.type,d.packed,d.stride,d.rows,columns,input,d.output,half_output,stream);
        else if(half_output)matvec_q8_f16(d.type,d.packed,d.stride,d.rows,columns,input,1,d.output,stream);
        else matvec_q8(d.type,d.packed,d.stride,d.rows,columns,input,1,static_cast<float*>(d.output),stream);
        return;
    }
    const auto select=[&]<int F>(){
        const auto layout=[&]<bool Aligned>(){
            const auto geometry=[&]<int R,int W>(){
                const auto format=[&]<bool Half,bool Residual,class Index>(){
                    const auto fn=quad_single<F,R,W,Aligned,Half,Residual,Index>;
                    constexpr int outputs=4*R/W;
                    // Q2_K keeps its native three-warp launch and has no LUT.
                    constexpr int block_warps=F==GGML_TYPE_Q2_K && W==3?3:4;
                    fn<<<unsigned((d.rows+outputs-1)/outputs),dim3(32,block_warps),4*GridCount<F>,stream>>>(p,int(columns),input,static_cast<const half*>(residual));
                    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_quad_m1",reinterpret_cast<const void*>(fn),"format="+std::to_string(F)+" k_warps="+std::to_string(W));
                };
                const auto width=[&]<class Index>(){if(residual)format.template operator()<true,true,Index>();else if(half_output)format.template operator()<true,false,Index>();else format.template operator()<false,false,Index>();};
                if(small_offsets(p,columns))width.template operator()<uint32_t>();else width.template operator()<uint64_t>();
            };
            if constexpr(IsIq1<F>){if(columns/32>=64){geometry.template operator()<4,2>();return;}geometry.template operator()<2,1>();}
            else{const int warps=original_k_cooperation<F>(d.rows,columns);if(warps==3)geometry.template operator()<2,3>();else if(warps==4)geometry.template operator()<2,4>();else geometry.template operator()<2,1>();}
        };
        if(aligned)layout.template operator()<true>();else layout.template operator()<false>();
    };
    switch(d.type){
#define CASE(F) case GGML_TYPE_##F:select.template operator()<GGML_TYPE_##F>();break;
    CASE(IQ1_S) CASE(IQ1_M) CASE(IQ2_XXS) CASE(IQ2_XS) CASE(IQ2_S) CASE(IQ3_XXS) CASE(IQ3_S) CASE(Q2_K)
#undef CASE
    default:throw std::invalid_argument("Unsupported GGUF quad projection");
    }
    checked_quad(cudaGetLastError());
}
void matvec_quad_group_m1(const QuadProjection* projections,size_t count,int64_t columns,const void* input,bool aligned,bool half_output,cudaStream_t stream){
    if(count&&!projections)throw std::invalid_argument("Missing GGUF compute projections");
    std::vector<DenseProjection> native;std::vector<QuadProjection> encoded;
    for(size_t i=0;i<count;++i){validate_quad(projections[i],columns);if(!projections[i].projection.rows)continue;if(projections[i].quad)encoded.push_back(projections[i]);else native.push_back(projections[i].projection);}
    if(encoded.empty()&&native.empty())return;
    if(!input||reinterpret_cast<uintptr_t>(input)%(aligned?16:4))throw std::invalid_argument("Invalid GGUF quad grouped activation");
    const bool pair=count==2&&!encoded.empty()&&
        (projections[0].quad||projections[0].projection.type==GGML_TYPE_Q2_K)&&
        (projections[1].quad||projections[1].projection.type==GGML_TYPE_Q2_K);
    if(pair && projections[0].quad && projections[1].quad){
        const auto launch=[&]<int A,int B,bool Aligned,bool Half,class Index>(){
            const auto fn=quad_pair<A,B,Aligned,Half,Index>;
            constexpr unsigned ARows=IsIq1<A>?8:4,BRows=IsIq1<B>?8:4;
            const unsigned a_rows=columns/32>=128?ARows:8,b_rows=columns/32>=128?BRows:8;
            const unsigned split=unsigned((projections[0].projection.rows+a_rows-1)/a_rows),end=unsigned((projections[1].projection.rows+b_rows-1)/b_rows);
            const unsigned words=std::max(projections[0].quad?GridCount<A>:0u,projections[1].quad?GridCount<B>:0u);
            fn<<<split+end,dim3(32,4),words*4,stream>>>(projections[0],projections[1],split,int(columns),input);
            if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_quad_m1_pair",reinterpret_cast<const void*>(fn),"formats="+std::to_string(A)+"/"+std::to_string(B));
        };
        const auto layouts=[&]<int A,int B>(){const auto out=[&]<bool Aligned,bool Half>(){if(small_offsets(projections[0],columns)&&small_offsets(projections[1],columns))launch.template operator()<A,B,Aligned,Half,uint32_t>();else launch.template operator()<A,B,Aligned,Half,uint64_t>();};if(aligned){if(half_output)out.template operator()<true,true>();else out.template operator()<true,false>();}else{if(half_output)out.template operator()<false,true>();else out.template operator()<false,false>();}};
#define QUAD_PAIR_FORMATS(X) X(IQ1_S) X(IQ1_M) X(IQ2_XXS) X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S) X(Q2_K)
        const auto second=[&]<int A>(){switch(projections[1].projection.type){
#define CASE(F) case GGML_TYPE_##F:layouts.template operator()<A,GGML_TYPE_##F>();break;
        QUAD_PAIR_FORMATS(CASE)
#undef CASE
        default:break;}};
        switch(projections[0].projection.type){
#define CASE(F) case GGML_TYPE_##F:second.template operator()<GGML_TYPE_##F>();break;
        QUAD_PAIR_FORMATS(CASE)
#undef CASE
        default:break;}
        checked_quad(cudaGetLastError());return;
    }
    // Keep native and interleaved weights in the original projection group.
    // Partitioning it would turn a three-projection Q/K/V group into a typed
    // two-matrix pair, changing its reduction tree and rounding.
    if(encoded.empty()){
        if(aligned)matvec_group_m1_aligned(native.data(),native.size(),columns,input,half_output,stream);
        else matvec_group(native.data(),native.size(),columns,input,nullptr,1,half_output,stream);
        return;
    }
    if(count==1){matvec_quad_m1(projections[0],columns,input,aligned,half_output,nullptr,stream);return;}
    uint64_t group_blocks=0;for(size_t i=0;i<count;++i)group_blocks+=uint64_t((projections[i].projection.rows+7)/8);
    std::vector<int> warps(count,1);
    bool cooperative=false;
    for(size_t i=0;i<count;++i){const auto& p=projections[i];
        switch(p.projection.type){
#define PLAN(F) case GGML_TYPE_##F:warps[i]=original_k_cooperation<GGML_TYPE_##F>(p.projection.rows,columns,group_blocks);break;
        case GGML_TYPE_IQ1_S:case GGML_TYPE_IQ1_M:break;
        PLAN(IQ2_XXS) PLAN(IQ2_XS) PLAN(IQ2_S) PLAN(IQ3_XXS) PLAN(IQ3_S)
        PLAN(Q2_K) PLAN(Q4_K) PLAN(Q5_K) PLAN(Q6_K) PLAN(IQ4_XS)
#undef PLAN
        default:throw std::invalid_argument("Unsupported GGUF mixed compute-storage projection");}
        cooperative|=warps[i]>1;
    }
    if(pair){
        for(size_t i=0;i<count;++i)
            if(projections[i].projection.type!=GGML_TYPE_IQ1_S&&projections[i].projection.type!=GGML_TYPE_IQ1_M)
                warps[i]=columns/32>=128?2:1;
    }else if(count==2&&!cooperative&&columns/32>=128){
        for(size_t i=0;i<count;++i)
            if(projections[i].projection.type!=GGML_TYPE_IQ1_S&&projections[i].projection.type!=GGML_TYPE_IQ1_M)warps[i]=2;
    }
    for(size_t first=0;first<count;first+=8){
        QuadGroupPacket packet{};unsigned blocks=0,words=0;bool small=true;
        for(size_t i=first;i<std::min(first+8,count);++i){
            if(!projections[i].projection.rows)continue;
            auto& e=packet.entries[packet.count++];e.p=projections[i];e.warps=warps[i];e.begin=blocks;
            const unsigned outputs=e.warps>1?4/e.warps*2:8;
            blocks+=unsigned((e.p.projection.rows+outputs-1)/outputs);e.end=blocks;
            if(e.p.quad)words=std::max(words,grid_words(e.p.projection.type));small&=small_offsets(e.p,columns);
        }
        if(!blocks)continue;
        const auto launch=[&]<bool Aligned,bool Half,class Index>(){const auto fn=quad_group<Aligned,Half,Index>;fn<<<blocks,dim3(32,4),words*4,stream>>>(packet,int(columns),input);};
        const auto out=[&]<bool Aligned,bool Half>(){if(small)launch.template operator()<Aligned,Half,uint32_t>();else launch.template operator()<Aligned,Half,uint64_t>();};
        if(aligned){if(half_output)out.template operator()<true,true>();else out.template operator()<true,false>();}else{if(half_output)out.template operator()<false,true>();else out.template operator()<false,false>();}
    }
    checked_quad(cudaGetLastError());
}
}
