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
#include "gguf_quad_small_m_extra.cuh"

template<int F,int Rows,int Tokens,bool PositionMajor,class Index>
__global__ void __launch_bounds__(128) quad_small_m_kernel(QuadProjection p,int columns,const void* input,bool half_output) {
 if constexpr(GridCount<F>!=0){stage_global_grid<F,0>(threadIdx.y*32+threadIdx.x);__syncthreads();}
 constexpr int W=1,Offset=0,Capacity=Tokens==2?2:4;
 const unsigned tile=blockIdx.x;
 const auto* packed=static_cast<const uint8_t*>(p.projection.packed);const int rows=int(p.projection.rows);void* output=p.projection.output;

 using D=QuadIqDot<F,Offset>;const unsigned lane=threadIdx.x,warp=threadIdx.y;
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

  float activation_scales[Tokens];
  #pragma unroll
  for(int m=0;m<Tokens;++m) {
   if constexpr(PositionMajor)activation_scales[m]=__ldg(reinterpret_cast<const float*>(static_cast<const uint8_t*>(input)+int64_t(columns)*Capacity)+i*Capacity+m);
   else activation_scales[m]=__low2float((static_cast<const block_q8_1*>(input)+Index(m)*groups+i)->ds);
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
    #pragma unroll
    for(int m=0;m<Tokens;++m) {
     int x0,x1;
     if constexpr(PositionMajor) {
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
 #pragma unroll
 for(int m=0;m<Tokens;++m) {
  const float value=warp_sums::reduce<Rows>(sums[m],lane);
  const int r=warp_sums::index<Rows>(lane);
  if(lane==warp_sums::owner<Rows>(r) && row+r<unsigned(rows))
   store_projection(output,int64_t(m)*rows+row+r,value,half_output);
 }
}
} // namespace

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
            (position_major && (tokens>4 || p.projection.type==GGML_TYPE_IQ1_S || p.projection.type==GGML_TYPE_IQ1_M || p.projection.type==GGML_TYPE_Q2_K)) ||
            !p.projection.packed || !p.projection.output || !input || p.row_begin<0 ||
            p.projection.rows<1 || p.projection.rows>INT_MAX || columns<1 || columns>INT_MAX || columns%256 ||
            p.row_begin%4 || p.projection.rows%4 || p.storage_rows<4 ||
            p.projection.rows>p.storage_rows/4*4 || p.row_begin>p.storage_rows/4*4-p.projection.rows)
        throw std::invalid_argument("Invalid direct quad small-M projection");
    const auto format=[&]<int F>() {
        const auto token=[&]<int M>() {
            // Bound live token/row accumulators for deeper drafts. This
            // changes work tiling, never the persistent weight layout.
            constexpr int Rows=M<=4?4:2,Outputs=4*Rows;
            const auto layout=[&]<bool PositionMajor>() {
                const auto function=[&] {
                    if constexpr(F==GGML_TYPE_IQ1_S || F==GGML_TYPE_IQ1_M)return quad_small_m_iq1_kernel<F==GGML_TYPE_IQ1_M,M,uint64_t>;
                    else if constexpr(F==GGML_TYPE_Q2_K)return quad_small_m_q2_kernel<M,uint64_t>;
                    else return quad_small_m_kernel<F,Rows,M,PositionMajor,uint64_t>;
                }();
                function<<<unsigned((p.projection.rows+Outputs-1)/Outputs),dim3(32,4),GridCount<F>*4,stream>>>(p,int(columns),input,half_output);
                if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gguf_quad_small_m",
                    reinterpret_cast<const void*>(function),"format="+std::to_string(F)+" M="+std::to_string(M)+
                    " shared_weight_decode=1 canonical_restore=0 position_major="+std::to_string(PositionMajor));
            };
            if constexpr(M>4)layout.template operator()<false>();
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
} // namespace mfq::cuda::gguf
