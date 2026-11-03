#include "gguf_quad_decode.h"
#include "gguf_quad_storage.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_quad_q2.cuh"
#include "gguf_quad_iq1m.cuh"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>


#include "gguf_prefill_scale.cuh"

namespace mfq::cuda::gguf {namespace {
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
__device__ void put_word(void* p,unsigned i,unsigned v){
    auto* q=static_cast<unsigned short*>(p)+2*i;q[0]=v;q[1]=v>>16;
}
__device__ unsigned read_word(const unsigned char* p,std::uint64_t at){return *reinterpret_cast<const unsigned*>(p+at);}

template<int F>__device__ void restore_fragment(const unsigned char* packed,std::uint64_t total,
        std::uint64_t qb,unsigned r,unsigned part,typename NativeBlock<F>::type* b){
    const auto entry=qb*8+part;
    if constexpr(Iq1<F>){
        put_word(b->qs,part,read_word(packed,entry*16+r*4));
        reinterpret_cast<unsigned short*>(b->qh)[part]=reinterpret_cast<const unsigned short*>(packed+total*128)[entry*4+r];
        if(!part){
            if constexpr(F==GGML_TYPE_IQ1_S)b->d=reinterpret_cast<const half*>(packed+total*192)[qb*4+r];
            else restore_quad_iq1m_metadata(packed+total*192+qb*32,r,b);
        }
    }else if constexpr(F==GGML_TYPE_Q2_K){
        restore_quad_q2_fragment(packed,total,qb,r,part,b);
    }else{
        if(!part)b->d=reinterpret_cast<const half*>(packed+total*(256+Extra<F>))[qb*4+r];
        const auto a=read_word(packed,entry*32+r*4),c=read_word(packed,entry*32+16+r*4);
        if constexpr(F==GGML_TYPE_IQ2_S){put_word(b->qs,part,a);put_word(b->qs,8+part,c);}
        else{put_word(b->qs,2*part,a);put_word(b->qs,2*part+1,c);}
        if constexpr(F==GGML_TYPE_IQ2_XS)b->scales[part]=packed[total*256+entry*4+r];
        else if constexpr(F==GGML_TYPE_IQ2_S){const auto meta=reinterpret_cast<const unsigned short*>(packed+total*256)[entry*4+r];b->scales[part]=meta;b->qh[part]=meta>>8;}
        else if constexpr(F==GGML_TYPE_IQ3_XXS)put_word(b->qs,16+part,read_word(packed,total*256+entry*16+r*4));
        else if constexpr(F==GGML_TYPE_IQ3_S){
            put_word(b->signs,part,read_word(packed,total*256+entry*16+r*4));
            b->qh[part]=packed[total*384+entry*4+r];
            if(!(part&1))b->scales[part/2]=packed[total*416+(entry/2)*4+r];
        }
    }
}
template<int F,bool Half,bool Scaled=false>__global__ void decode_quad(const unsigned char* packed,
        std::int64_t storage_rows,std::uint64_t blocks,std::int64_t begin,std::int64_t rows,
        const std::int64_t* indices,std::uint64_t count,void* output,const std::int32_t* exponents=nullptr){
    using B=typename NativeBlock<F>::type;
    // A warp owns one quantization block, so reconstruction requires no
    // block-wide barriers. Four independent blocks share a CTA. Their local
    // images remain aligned even when the encoded block size is not.
    constexpr unsigned Warps=4,Stride=(sizeof(B)+15)&~15u;
    __shared__ __align__(16) unsigned char images[Warps*Stride];
    const unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    auto* image=images+warp*Stride;
    const auto total=std::uint64_t(storage_rows/4)*blocks;
    const bool vector_output=!(reinterpret_cast<uintptr_t>(output)%16);
    for(auto at=std::uint64_t(blockIdx.x)*Warps+warp;at<count*blocks;
            at+=std::uint64_t(gridDim.x)*Warps){
        const auto output_row=at/blocks,block=at%blocks;
        const auto selected=indices?indices[output_row]:std::int64_t(output_row);
        const bool valid=selected>=0 && selected<rows;
        const auto row=valid?std::uint64_t(begin)+std::uint64_t(selected):0;
        if(valid){
            if(row<std::uint64_t(storage_rows/4)*4){
                if(lane<8)restore_fragment<F>(packed,total,(row/4)*blocks+block,
                    unsigned(row%4),lane,reinterpret_cast<B*>(image));
            }else{
                const auto* source=packed+total*4*sizeof(B)+(row-std::uint64_t(storage_rows/4)*4)*blocks*sizeof(B)+block*sizeof(B);
                for(unsigned j=lane;j<sizeof(B);j+=32)image[j]=source[j];
            }
        }
        __syncwarp();
        const auto offset=at*256;
        if(vector_output){
            float values[8]={};
            if(valid)gguf_values8<F>(image,lane*8,values);
            if constexpr(Scaled){
                const auto exponent=-exponents[output_row];
                #pragma unroll
                for(int j=0;j<8;++j)values[j]=prefill_ldexp(values[j],exponent);
            }
            if constexpr(Half)gguf_store8(static_cast<half*>(output)+offset+lane*8,values);
            else gguf_store8(static_cast<float*>(output)+offset+lane*8,values);
        }else{
            // Element-aligned output views retain coalesced scalar stores.
            #pragma unroll
            for(unsigned j=0;j<8;++j){
                const auto k=lane+j*32;
                float value=valid?gguf_value<F>(image,k):0.f;
                if constexpr(Scaled)value=prefill_ldexp(value,-exponents[output_row]);
                if constexpr(Half)static_cast<half*>(output)[offset+k]=__float2half_rn(value);
                else static_cast<float*>(output)[offset+k]=value;
            }
        }
        __syncwarp();
    }
}
void trace_decode(bool scaled){
    const auto* setting=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
    if(!setting || setting[0]!='1')return;
    static std::atomic<unsigned> seen{0};
    const unsigned bit=scaled?2:1;
    if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
        std::fprintf(stderr,"native_gguf_quad_decode actual_match=1 warp_per_block=1 blocks_per_cta=4 shared_codebook=1 scaled=%d\n",int(scaled));
}
}
void decode_quad_rows(ggml_type type,const void* packed,std::int64_t storage_rows,
        std::int64_t columns,std::int64_t begin,std::int64_t rows,const std::int64_t* indices,
        std::int64_t count,void* output,bool half_output,cudaStream_t stream){
    quad_storage_bytes(type,storage_rows,columns);
    if(begin<0||rows<1||begin>storage_rows-rows||count<0||(!indices&&count!=rows)||
       std::uint64_t(count)>UINT64_MAX/std::uint64_t(columns)/(half_output?2:4))
        throw std::invalid_argument("Invalid GGUF compute-storage decode view");
    if(!count)return;
    if(!packed||!output||packed==output||reinterpret_cast<uintptr_t>(packed)%16||
       reinterpret_cast<uintptr_t>(output)%(half_output?2:4)||
       (indices&&reinterpret_cast<uintptr_t>(indices)%8))
        throw std::invalid_argument("Invalid GGUF compute-storage decode buffers");
    const auto blocks=std::uint64_t(columns/256),work=std::uint64_t(count)*blocks;
    trace_decode(false);
    const auto launch=[&]<int F,bool Half>(){
        decode_quad<F,Half><<<unsigned(std::min<std::uint64_t>((work+3)/4,65535)),128,0,stream>>>(
            static_cast<const unsigned char*>(packed),storage_rows,blocks,begin,rows,indices,count,output);
    };
    switch(type){
#define DECODE(F,T) case GGML_TYPE_##F:if(half_output)launch.template operator()<GGML_TYPE_##F,true>();else launch.template operator()<GGML_TYPE_##F,false>();break;
    QUAD_FORMATS(DECODE)
#undef DECODE
    default:break;
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
void decode_quad_rows_scaled(ggml_type type,const void* packed,std::int64_t storage_rows,std::int64_t columns,
        std::int64_t begin,std::int64_t rows,const std::int32_t* exponents,void* output,cudaStream_t stream) {
    quad_storage_bytes(type,storage_rows,columns);
    if(begin<0 || rows<1 || begin>storage_rows-rows || !exponents || !packed || !output)
        throw std::invalid_argument("Invalid GGUF scaled compute-storage view");
    const auto blocks=std::uint64_t(columns/256),work=std::uint64_t(rows)*blocks;
    trace_decode(true);
    switch(type) {
#define SCALED_QUAD(F,T) case GGML_TYPE_##F:decode_quad<GGML_TYPE_##F,true,true><<<unsigned(std::min<std::uint64_t>((work+3)/4,65535)),128,0,stream>>>(static_cast<const unsigned char*>(packed),storage_rows,blocks,begin,rows,nullptr,rows,output,exponents);break;
        QUAD_FORMATS(SCALED_QUAD)
#undef SCALED_QUAD
        default:throw std::invalid_argument("Unsupported GGUF scaled compute-storage format");
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
}
