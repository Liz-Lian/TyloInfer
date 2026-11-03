#include "gguf_quad_storage.h"
#include "ggml-cuda/common.cuh"
#include "gguf_quad_q2.cuh"
#include "gguf_quad_iq1m.cuh"
#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>

namespace mfq::cuda::gguf {
namespace {
#define QUAD_FORMATS(X) \
 X(IQ1_S,block_iq1_s) X(IQ1_M,block_iq1_m) \
 X(IQ2_XXS,block_iq2_xxs) X(IQ2_XS,block_iq2_xs) X(IQ2_S,block_iq2_s) \
 X(IQ3_XXS,block_iq3_xxs) X(IQ3_S,block_iq3_s) X(Q2_K,block_q2_K)
template<int F>struct NativeBlock;
#define BLOCK(F,T) template<>struct NativeBlock<GGML_TYPE_##F>{using type=T;};
QUAD_FORMATS(BLOCK)
#undef BLOCK
template<int F>constexpr bool Iq1=F==GGML_TYPE_IQ1_S||F==GGML_TYPE_IQ1_M;
template<int F>constexpr std::uint64_t Extra=F==GGML_TYPE_IQ2_XS?32:F==GGML_TYPE_IQ2_S?64:F==GGML_TYPE_IQ3_XXS?128:F==GGML_TYPE_IQ3_S?176:0;

__device__ unsigned word(const void* pointer,unsigned index){
 const auto* p=static_cast<const unsigned short*>(pointer)+2*index;
 return unsigned(p[0])|(unsigned(p[1])<<16);
}
__device__ void put_word(void* pointer,unsigned index,unsigned value){
 auto* p=static_cast<unsigned short*>(pointer)+2*index;
 p[0]=static_cast<unsigned short>(value);p[1]=static_cast<unsigned short>(value>>16);
}
__device__ unsigned read_word(const unsigned char* p,std::uint64_t offset){return *reinterpret_cast<const unsigned*>(p+offset);}
__device__ void write_word(unsigned char* p,std::uint64_t offset,unsigned value){*reinterpret_cast<unsigned*>(p+offset)=value;}

template<int F>__global__ void pack_quad(const unsigned char* native,std::uint64_t stride,
        std::int64_t rows,std::uint64_t blocks,unsigned char* packed){
 using B=typename NativeBlock<F>::type;
 const auto total=std::uint64_t(rows/4)*blocks;
 const auto work=total+std::uint64_t(rows%4)*blocks;
 for(auto qb=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;qb<work;qb+=std::uint64_t(gridDim.x)*blockDim.x){
  if(qb>=total){
   const auto at=qb-total,row=std::uint64_t(rows/4)*4+at/blocks;
   const auto* source=native+row*stride+(at%blocks)*sizeof(B);
   auto* destination=packed+total*4*sizeof(B)+at*sizeof(B);
   for(unsigned j=0;j<sizeof(B);++j)destination[j]=source[j];
   continue;
  }
  const auto quad=qb/blocks,block=qb%blocks;
  if constexpr(F==GGML_TYPE_IQ1_M)
   pack_quad_iq1m_metadata(native+quad*4*stride+block*sizeof(B),stride,packed+total*192+qb*32);
  for(unsigned r=0;r<4;++r){
   const auto candidate=quad*4+r,row=candidate<std::uint64_t(rows)?candidate:std::uint64_t(rows-1);
   const auto* b=reinterpret_cast<const B*>(native+row*stride)+block;
   if constexpr(Iq1<F>){
    for(unsigned part=0;part<8;++part){const auto entry=qb*8+part;
     write_word(packed,entry*16+r*4,word(b->qs,part));
     reinterpret_cast<unsigned short*>(packed+total*128)[entry*4+r]=reinterpret_cast<const unsigned short*>(b->qh)[part];
    }
    if constexpr(F==GGML_TYPE_IQ1_S)reinterpret_cast<half*>(packed+total*192)[qb*4+r]=b->d;
   }else if constexpr(F==GGML_TYPE_Q2_K){
    for(unsigned part=0;part<8;++part)pack_quad_q2_fragment(b,packed,total,qb,r,part);
   }else{
    reinterpret_cast<half*>(packed+total*(256+Extra<F>))[qb*4+r]=b->d;
    for(unsigned part=0;part<8;++part){const auto entry=qb*8+part;
     unsigned a,c;
     if constexpr(F==GGML_TYPE_IQ2_S){a=word(b->qs,part);c=word(b->qs,8+part);}
     else{a=word(b->qs,2*part);c=word(b->qs,2*part+1);}
     write_word(packed,entry*32+r*4,a);write_word(packed,entry*32+16+r*4,c);
     if constexpr(F==GGML_TYPE_IQ2_XS)packed[total*256+entry*4+r]=b->scales[part];
     else if constexpr(F==GGML_TYPE_IQ2_S){auto* meta=reinterpret_cast<unsigned short*>(packed+total*256);meta[entry*4+r]=unsigned(b->scales[part])|(unsigned(b->qh[part])<<8);}
     else if constexpr(F==GGML_TYPE_IQ3_XXS)write_word(packed,total*256+entry*16+r*4,word(b->qs,16+part));
     else if constexpr(F==GGML_TYPE_IQ3_S){
      write_word(packed,total*256+entry*16+r*4,word(b->signs,part));
      packed[total*384+entry*4+r]=b->qh[part];
      if(!(part&1))packed[total*416+(entry/2)*4+r]=b->scales[part/2];
     }
    }
   }
  }
 }
}

template<int F>__global__ void unpack_quad(const unsigned char* packed,std::int64_t storage_rows,
        std::uint64_t blocks,std::int64_t begin,std::int64_t rows,const std::int64_t* indices,
        std::int64_t count,unsigned char* native,std::uint64_t stride){
 using B=typename NativeBlock<F>::type;
 const auto total=std::uint64_t(storage_rows/4)*blocks,work=std::uint64_t(count)*blocks;
 for(auto at=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;at<work;at+=std::uint64_t(gridDim.x)*blockDim.x){
  const auto output_row=at/blocks,block=at%blocks;
  const auto selected=indices?indices[output_row]:std::int64_t(output_row);
  auto* b=reinterpret_cast<B*>(native+output_row*stride)+block;
  if(selected<0||selected>=rows){auto* bytes=reinterpret_cast<unsigned char*>(b);for(unsigned j=0;j<sizeof(B);++j)bytes[j]=0;continue;}
  const auto row=std::uint64_t(begin+selected),qb=(row/4)*blocks+block;const unsigned r=unsigned(row%4);
  if(row>=std::uint64_t(storage_rows/4)*4){
   const auto* source=packed+total*4*sizeof(B)+(row-std::uint64_t(storage_rows/4)*4)*blocks*sizeof(B)+block*sizeof(B);
   auto* destination=reinterpret_cast<unsigned char*>(b);
   for(unsigned j=0;j<sizeof(B);++j)destination[j]=source[j];
   continue;
  }
  if constexpr(Iq1<F>){
   for(unsigned part=0;part<8;++part){const auto entry=qb*8+part;
    put_word(b->qs,part,read_word(packed,entry*16+r*4));
    reinterpret_cast<unsigned short*>(b->qh)[part]=reinterpret_cast<const unsigned short*>(packed+total*128)[entry*4+r];
   }
   if constexpr(F==GGML_TYPE_IQ1_S)b->d=reinterpret_cast<const half*>(packed+total*192)[qb*4+r];
   else restore_quad_iq1m_metadata(packed+total*192+qb*32,r,b);
  }else if constexpr(F==GGML_TYPE_Q2_K){
   for(unsigned part=0;part<8;++part)restore_quad_q2_fragment(packed,total,qb,r,part,b);
  }else{
   b->d=reinterpret_cast<const half*>(packed+total*(256+Extra<F>))[qb*4+r];
   for(unsigned part=0;part<8;++part){const auto entry=qb*8+part;
    const unsigned a=read_word(packed,entry*32+r*4),c=read_word(packed,entry*32+16+r*4);
    if constexpr(F==GGML_TYPE_IQ2_S){put_word(b->qs,part,a);put_word(b->qs,8+part,c);}
    else{put_word(b->qs,2*part,a);put_word(b->qs,2*part+1,c);}
    if constexpr(F==GGML_TYPE_IQ2_XS)b->scales[part]=packed[total*256+entry*4+r];
    else if constexpr(F==GGML_TYPE_IQ2_S){const auto meta=reinterpret_cast<const unsigned short*>(packed+total*256)[entry*4+r];b->scales[part]=meta;b->qh[part]=meta>>8;}
    else if constexpr(F==GGML_TYPE_IQ3_XXS)put_word(b->qs,16+part,read_word(packed,total*256+entry*16+r*4));
    else if constexpr(F==GGML_TYPE_IQ3_S){
     put_word(b->signs,part,read_word(packed,total*256+entry*16+r*4));b->qh[part]=packed[total*384+entry*4+r];
     if(!(part&1))b->scales[part/2]=packed[total*416+(entry/2)*4+r];
    }
   }
  }
 }
}
void checked(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
unsigned grid_for(std::uint64_t work){return unsigned(std::min<std::uint64_t>((work+255)/256,65535));}
std::uint64_t row_bytes(ggml_type type,std::int64_t columns){
 if(!supports_quad_storage(type)||columns<1||columns%256)throw std::invalid_argument("Invalid GGUF quad format or columns");
 return std::uint64_t(columns/256)*ggml_type_size(type);
}
}
bool supports_quad_storage(ggml_type type)noexcept{switch(type){
#define SUPPORTED(F,T) case GGML_TYPE_##F:return true;
 QUAD_FORMATS(SUPPORTED)
#undef SUPPORTED
 default:return false;}}
std::uint64_t quad_storage_bytes(ggml_type type,std::int64_t rows,std::int64_t columns){
 const auto stride=row_bytes(type,columns);
 if(rows<1)throw std::invalid_argument("Invalid GGUF quad rows");
 if(std::uint64_t(rows)>std::numeric_limits<std::uint64_t>::max()/stride)throw std::invalid_argument("GGUF quad storage size overflows");
 return std::uint64_t(rows)*stride;
}
void pack_quad_storage(ggml_type type,const void* native,std::uint64_t stride,std::int64_t rows,std::int64_t columns,void* packed,cudaStream_t stream){
 const auto bytes=quad_storage_bytes(type,rows,columns);
 if(!native||!packed||(reinterpret_cast<uintptr_t>(native)&1)||(reinterpret_cast<uintptr_t>(packed)&15)||stride<row_bytes(type,columns)||(stride&1)||std::uint64_t(rows)>UINT64_MAX/stride||native==packed)throw std::invalid_argument("Invalid GGUF quad packing storage");
 (void)bytes;const auto blocks=std::uint64_t(columns/256),work=(std::uint64_t(rows/4)+std::uint64_t(rows%4))*blocks;
 switch(type){
#define PACK(F,T) case GGML_TYPE_##F:pack_quad<GGML_TYPE_##F><<<grid_for(work),256,0,stream>>>(static_cast<const unsigned char*>(native),stride,rows,blocks,static_cast<unsigned char*>(packed));break;
 QUAD_FORMATS(PACK)
#undef PACK
 default:break;}
 checked(cudaGetLastError());
}
void unpack_quad_storage(ggml_type type,const void* packed,std::int64_t storage_rows,std::int64_t columns,std::int64_t begin,std::int64_t rows,const std::int64_t* indices,std::int64_t count,void* native,std::uint64_t stride,cudaStream_t stream){
 quad_storage_bytes(type,storage_rows,columns);
 if(begin<0||rows<1||begin>storage_rows-rows||count<0||(!indices&&count!=rows)||stride<row_bytes(type,columns)||(stride&1)||std::uint64_t(count)>UINT64_MAX/stride)throw std::invalid_argument("Invalid GGUF quad row view");
 if(!count)return;
 if(!packed||!native||(reinterpret_cast<uintptr_t>(packed)&15)||(reinterpret_cast<uintptr_t>(native)&1)||(indices&&(reinterpret_cast<uintptr_t>(indices)&7))||packed==native)throw std::invalid_argument("Invalid GGUF quad unpacking storage");
 const auto blocks=std::uint64_t(columns/256),work=std::uint64_t(count)*blocks;
 switch(type){
#define UNPACK(F,T) case GGML_TYPE_##F:unpack_quad<GGML_TYPE_##F><<<grid_for(work),256,0,stream>>>(static_cast<const unsigned char*>(packed),storage_rows,blocks,begin,rows,indices,count,static_cast<unsigned char*>(native),stride);break;
 QUAD_FORMATS(UNPACK)
#undef UNPACK
 default:break;}
 checked(cudaGetLastError());
}
}
