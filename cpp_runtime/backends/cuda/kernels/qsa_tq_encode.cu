#include "qsa_tq_encode.h"
#include "qsa_tq.h"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace tyloinfer::qsa {
namespace {
namespace profile14 {
constexpr int STATE_BITS=14;
#include "qsa_tq_encode_impl.cuh"
}
namespace profile16 {
constexpr int STATE_BITS=16;
#include "qsa_tq_encode_impl.cuh"
}

template<int Bits,int K> int capacity(int rows) {
    int active=0,device=0;cudaError_t error;
    error=cudaGetDevice(&device);if(error)return -int(error);
    cudaDeviceProp properties;error=cudaGetDeviceProperties(&properties,device);if(error)return -int(error);
    cudaFuncAttributes attributes{};
    if constexpr(Bits==14)error=cudaFuncGetAttributes(&attributes,profile14::solve<K>);
    else error=cudaFuncGetAttributes(&attributes,profile16::solve<K>);
    if(error)return -int(error);
    constexpr std::size_t dynamic=Bits==14?profile14::SH<K>:profile16::SH<K>;
    const auto available=std::max(properties.sharedMemPerBlock,properties.sharedMemPerBlockOptin);
    if(dynamic+attributes.sharedSizeBytes>available)return -int(cudaErrorNotSupported);
    if constexpr(Bits==14) {
        error=cudaFuncSetAttribute(profile14::solve<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,profile14::SH<K>);
        if(error)return -int(error);
        error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,profile14::solve<K>,profile14::THREADS<K>,profile14::SH<K>);
    } else {
        error=cudaFuncSetAttribute(profile16::solve<K>,cudaFuncAttributeMaxDynamicSharedMemorySize,profile16::SH<K>);
        if(error)return -int(error);
        error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,profile16::solve<K>,profile16::THREADS<K>,profile16::SH<K>);
    }
    if(error)return -int(error);
    return active?std::min(rows,active*properties.multiProcessorCount):-int(cudaErrorNotSupported);
}
template<int Bits> int capacity(int rows,int k) {
#define CASE(K) case K:return capacity<Bits,K>(rows)
    switch(k){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
#undef CASE
    return -int(cudaErrorInvalidValue);
}
template<int Bits,int K> int launch(float* values,std::uint8_t* symbols,std::uint8_t* trace,
    const int* ids,const int* counts,int rows,int waves,cudaStream_t stream) {
    if constexpr(Bits==14)
        profile14::solve<K><<<waves,profile14::THREADS<K>,profile14::SH<K>,stream>>>(values,symbols,trace,ids,counts,rows);
    else profile16::solve<K><<<waves,profile16::THREADS<K>,profile16::SH<K>,stream>>>(values,symbols,trace,ids,counts,rows);
    return int(cudaGetLastError());
}
template<int Bits> int launch(int k,float* values,std::uint8_t* symbols,std::uint8_t* trace,
    const int* ids,const int* counts,int rows,int waves,cudaStream_t stream) {
#define CASE(K) case K:return launch<Bits,K>(values,symbols,trace,ids,counts,rows,waves,stream)
    switch(k){CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);}
#undef CASE
    return int(cudaErrorInvalidValue);
}

float book_rms(int width) {
    const float gain=__half2float(__ushort_as_half(0x1eee)),bias=__half2float(__ushort_as_half(0xc931));
    double energy=0;
    for(unsigned state=0;state<(1u<<width);++state) {
        const unsigned x=state*0x83dcd12du;
        const auto sum=(x&255)+((x>>8)&255)+((x>>16)&255)+(x>>24);
        const float q=__half2float(__float2half_rn(std::fmaf(float(sum+1024),gain,bias)));
        energy+=double(q)*q;
    }
    return float(std::sqrt(energy/double(1u<<width)));
}
__device__ float warp_sum(float x) {
    for(int delta=16;delta;delta/=2)x+=__shfl_xor_sync(0xffffffff,x,delta);return x;
}
__global__ void prepare(const float* input,float* values,std::uint8_t* bits,const int* budgets,
    int fallback,int rows,float rms,float factor,int* status) {
    const int block=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid/32;
    extern __shared__ float gain[];
    for(int row=warp;row<rows;row+=blockDim.x/32) {
        const auto at=(int64_t(block)*rows+row)*256;
        float energy=0;
        for(int d=lane;d<256;d+=32){const float x=input[at+d];energy+=x*x;}
        energy=warp_sum(energy);
        const float scale=fmaxf(sqrtf(energy/256)/rms*factor,0x1p-24f);
        for(int d=lane;d<256;d+=32)values[at+d]=input[at+d]/scale;
        if(!lane){gain[row]=energy*.1875f;bits[int64_t(block)*rows+row]=1;}
    }
    __syncthreads();
    if(!tid) {
        const int target=budgets?budgets[block]:fallback;
        if(target<rows || target>8*rows){atomicExch(status,int(cudaErrorInvalidValue));return;}
        for(int used=rows;used<target;++used) {
            int best=-1;float value=-1;
            for(int row=0;row<rows;++row)
                if(bits[int64_t(block)*rows+row]<8 && gain[row]>value){best=row;value=gain[row];}
            ++bits[int64_t(block)*rows+best];gain[best]*=.25f;
        }
    }
}
__global__ void compact(const std::uint8_t* bits,int* ids,int* counts,int rows,const int* status) {
    const auto row=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=std::size_t(rows) || *status)return;
    const int k=bits[row]-1,index=atomicAdd(counts+k,1);ids[int64_t(k)*rows+index]=int(row);
}
__global__ void finalize(const float* input,const float* decoded,const std::uint8_t* symbols,
    const std::uint8_t* bits,std::uint8_t* image,float* sse,float relative,const int* status) {
    if(*status)return;
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid/32,k=bits[row];
    const auto at=int64_t(row)*256+tid;
    const float x=input[at],q=decoded[at];
    __shared__ float numerator[8],denominator[8],errors[8],scale;
    __shared__ half anchor;
    const float n=warp_sum(x*q),d=warp_sum(q*q);
    if(!lane){numerator[warp]=n;denominator[warp]=d;}__syncthreads();
    if(!tid) {
        float a=0,b=0;for(int w=0;w<8;++w){a+=numerator[w];b+=denominator[w];}
        anchor=__float2half_rn(fmaxf(a/fmaxf(b*relative,1e-30f),0.f));scale=__half2float(anchor)*relative;
        reinterpret_cast<half*>(image+int64_t(row)*258)[0]=anchor;
    }
    __syncthreads();
    const float diff=x-__half2float(__float2half_rn(q*scale)),loss=warp_sum(diff*diff);
    if(!lane)errors[warp]=loss;__syncthreads();
    if(!tid){float loss=0;for(int w=0;w<8;++w)loss+=errors[w];sse[row]=loss;}
    for(int byte=tid;byte<32*k;byte+=256) {
        const int first=byte*8/k,offset=byte*8%k;
        unsigned packed=unsigned(symbols[int64_t(row)*256+255-first])>>offset;
        for(int j=1;j<(8+k-1)/k+1;++j)
            if(j*k-offset<8 && first+j<256)packed|=unsigned(symbols[int64_t(row)*256+255-first-j])<<(j*k-offset);
        image[int64_t(row)*258+2+byte]=std::uint8_t(packed);
    }
}
__global__ void pack(const std::uint8_t* images,const std::uint8_t* bits,std::uint8_t* output,
    std::size_t stride,std::uint32_t* sizes,std::uint32_t* flags,int rows,int state_bits,int* status) {
    const int block=blockIdx.x,tid=threadIdx.x,base=block*rows;
    extern __shared__ int offsets[];
    __shared__ int meta,tag,used;
    if(!tid){used=*status?0:1;if(!used)sizes[block]=flags[block]=0;}
    __syncthreads();if(!used)return;
    if(!tid) {
        bool same=true;int total=0;
        for(int row=0;row<rows;++row){offsets[row]=total;total+=32*bits[base+row];same&=bits[base+row]==bits[base];}
        tag=same?bits[base]:255;
        meta=(2*rows+(rows+1)/2+(same?0:(3*rows+7)/8)+3)&~3;
        used=meta+total;
        if(std::size_t(used)>stride) {
            atomicExch(status,int(cudaErrorInvalidValue));used=0;sizes[block]=flags[block]=0;
        }else {sizes[block]=std::uint32_t(used);flags[block]=tq_precision_flags(std::uint8_t(tag),std::uint8_t(state_bits));}
    }
    __syncthreads();if(!used)return;
    auto* destination=output+std::size_t(block)*stride;
    for(int row=tid;row<rows;row+=blockDim.x) {
        destination[2*row]=images[int64_t(base+row)*258];destination[2*row+1]=images[int64_t(base+row)*258+1];
    }
    for(int i=tid;i<(rows+1)/2;i+=blockDim.x)destination[2*rows+i]=0x77;
    if(!tid) {
        for(int i=2*rows+(rows+1)/2;i<meta;++i)destination[i]=0;
        if(tag==255)for(int row=0;row<rows;++row) {
            const int bit=row*3,byte=bit/8,shift=bit&7;const unsigned code=bits[base+row]-1;
            destination[2*rows+(rows+1)/2+byte]|=std::uint8_t(code<<shift);
            if(shift>5)destination[2*rows+(rows+1)/2+byte+1]|=std::uint8_t(code>>(8-shift));
        }
    }
    for(int row=0;row<rows;++row)
        for(int byte=tid;byte<32*bits[base+row];byte+=blockDim.x)
            destination[meta+offsets[row]+byte]=images[int64_t(base+row)*258+2+byte];
}

template<class Kernel> int allow_shared_memory(Kernel kernel,std::size_t required) {
    cudaFuncAttributes attributes{};
    auto error=cudaFuncGetAttributes(&attributes,kernel);if(error)return int(error);
    if(required<=std::size_t(attributes.maxDynamicSharedSizeBytes))return 0;
    int device=0;error=cudaGetDevice(&device);if(error)return int(error);
    cudaDeviceProp properties;error=cudaGetDeviceProperties(&properties,device);if(error)return int(error);
    const auto maximum=std::max(properties.sharedMemPerBlock,properties.sharedMemPerBlockOptin)-attributes.sharedSizeBytes;
    if(required>maximum)return int(cudaErrorNotSupported);
    // This is a permission ceiling, not an allocation. Raising to the device
    // limit avoids lowering a limit used by another live plan or graph.
    return int(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(maximum)));
}
} // namespace

int tq_make_encode_plan(TqEncodePlan* output,int blocks,int rows,TqEncodeProfile profile,std::size_t budget) {
    if(!output || blocks<=0 || rows<=0 || rows%4 || rows>std::numeric_limits<int>::max()/8 || blocks>std::numeric_limits<int>::max()/rows ||
       (profile!=TqEncodeProfile::Fast && profile!=TqEncodeProfile::Quality))return int(cudaErrorInvalidValue);
    TqEncodePlan plan;plan.blocks=blocks;plan.rows_per_block=rows;plan.rows=blocks*rows;plan.state_bits=int(profile);
    plan.book_rms=book_rms(plan.state_bits);
    plan.relative=__half2float(__float2half_rn(std::exp2(-2.f+4.f*7/15.f)));
    std::size_t cursor=0;
    const auto allocate=[&](std::size_t n){const auto at=cursor;cursor=(cursor+n+255)&~std::size_t(255);return at;};
    plan.values=allocate(std::size_t(plan.rows)*256*sizeof(float));
    plan.symbols=allocate(std::size_t(plan.rows)*256);
    plan.bits=allocate(plan.rows);plan.row_images=allocate(std::size_t(plan.rows)*258);
    plan.row_sse=allocate(std::size_t(plan.rows)*sizeof(float));
    plan.ids=allocate(std::size_t(plan.rows)*8*sizeof(int));plan.counts=allocate(8*sizeof(int));
    plan.trace=cursor;
    std::size_t trace_size=0;
    for(int k=1;k<=8;++k) {
        const int maximum=plan.state_bits==14?capacity<14>(plan.rows,k):capacity<16>(plan.rows,k);
        if(maximum<0)return -maximum;
        const auto row_bytes=(std::size_t(256)<<(plan.state_bits-k))/(k==1?8:k<=4?2:1);
        if(budget && (budget<cursor || budget-cursor<row_bytes))return int(cudaErrorMemoryAllocation);
        plan.waves[k-1]=budget?int(std::min<std::size_t>(maximum,(budget-cursor)/row_bytes)):maximum;
        trace_size=std::max(trace_size,row_bytes*plan.waves[k-1]);
    }
    const int error=allow_shared_memory(prepare,rows*sizeof(float));if(error)return error;
    const int pack_error=allow_shared_memory(pack,rows*sizeof(int));if(pack_error)return pack_error;
    plan.workspace_bytes=cursor+trace_size;*output=plan;return 0;
}

int tq_limit_encode_waves(TqEncodePlan* plan,int active_rows) {
    if(!plan || active_rows<=0 || plan->blocks<=0 || plan->rows_per_block<=0 ||
        (plan->state_bits!=14 && plan->state_bits!=16))return int(cudaErrorInvalidValue);
    auto waves=plan->waves;std::size_t trace_bytes=0;
    for(int bits=1;bits<=8;++bits) {
        if(waves[bits-1]<=0)return int(cudaErrorInvalidValue);
        waves[bits-1]=std::min(waves[bits-1],active_rows);
        const auto row_bytes=(std::size_t(256)<<(plan->state_bits-bits))/(bits==1?8:bits<=4?2:1);
        if(std::size_t(waves[bits-1])>(std::numeric_limits<std::size_t>::max()-plan->trace)/row_bytes)
            return int(cudaErrorInvalidValue);
        trace_bytes=std::max(trace_bytes,row_bytes*waves[bits-1]);
    }
    plan->waves=waves;plan->workspace_bytes=plan->trace+trace_bytes;return int(cudaSuccess);
}

int tq_encode_modes(const TqEncodePlan& plan,const float* input,const std::int32_t* budgets,int fallback,
    float factor,std::uint8_t* images,std::size_t stride,std::uint32_t* sizes,std::uint32_t* flags,
    int* status,void* workspace,std::size_t workspace_bytes,cudaStream_t stream) {
    if(!input || !images || !sizes || !flags || !status || !workspace || !stride ||
       plan.blocks<=0 || plan.rows_per_block<=0 || stride>std::numeric_limits<std::size_t>::max()/plan.blocks ||
       (plan.state_bits!=14 && plan.state_bits!=16) || !std::isfinite(factor) || factor<=0 ||
       (!budgets && (fallback<plan.rows_per_block || fallback>8*plan.rows_per_block)))return int(cudaErrorInvalidValue);
    if(workspace_bytes<plan.workspace_bytes)return int(cudaErrorMemoryAllocation);
    auto* bytes=static_cast<std::uint8_t*>(workspace);
    auto* values=reinterpret_cast<float*>(bytes+plan.values);
    auto* symbols=bytes+plan.symbols;auto* bits=bytes+plan.bits;
    auto* row_images=bytes+plan.row_images;
    auto* sse=reinterpret_cast<float*>(bytes+plan.row_sse);
    auto* ids=reinterpret_cast<int*>(bytes+plan.ids);auto* counts=reinterpret_cast<int*>(bytes+plan.counts);
    auto error=cudaMemsetAsync(status,0,sizeof(int),stream);if(error)return int(error);
    error=cudaMemsetAsync(counts,0,8*sizeof(int),stream);if(error)return int(error);
    prepare<<<plan.blocks,256,plan.rows_per_block*sizeof(float),stream>>>(input,values,bits,budgets,fallback,
        plan.rows_per_block,plan.book_rms,factor,status);
    if((error=cudaGetLastError()))return int(error);
    compact<<<unsigned((std::size_t(plan.rows)+255)/256),256,0,stream>>>(bits,ids,counts,plan.rows,status);
    if((error=cudaGetLastError()))return int(error);
    for(int k=1;k<=8;++k) {
        const int result=plan.state_bits==14?launch<14>(k,values,symbols,bytes+plan.trace,ids,counts,plan.rows,plan.waves[k-1],stream):
            launch<16>(k,values,symbols,bytes+plan.trace,ids,counts,plan.rows,plan.waves[k-1],stream);
        if(result)return result;
    }
    finalize<<<plan.rows,256,0,stream>>>(input,values,symbols,bits,row_images,sse,plan.relative,status);
    if((error=cudaGetLastError()))return int(error);
    pack<<<plan.blocks,256,plan.rows_per_block*sizeof(int),stream>>>(row_images,bits,images,stride,sizes,flags,
        plan.rows_per_block,plan.state_bits,status);return int(cudaGetLastError());
}
} // namespace tyloinfer::qsa
