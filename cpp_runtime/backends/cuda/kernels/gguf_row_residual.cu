#include "gguf_row_residual.h"
#include "gguf_compact_decode.cuh"
#include <climits>
#include <stdexcept>

namespace mfq::cuda::gguf {
namespace {
using namespace compact_detail;
template<int Format>
__device__ float value(const std::uint8_t* row,std::int64_t column) {
    using L=Layout<Format>;
    const auto* block=row+(column/128)*L::bytes;
    const auto four=decode_four<L::bits,L::steps>(block+2,int(column%128)/4);
    const float values[4]={four.x,four.y,four.z,four.w};
    return scale_at(block)*values[column%4];
}
__device__ float decoded(const std::uint8_t* row,int type,std::int64_t column) {
    switch(type) {
#define ROW(F) case F:return value<F>(row,column)
        ROW(144);ROW(145);ROW(146);ROW(147);ROW(148);ROW(149);ROW(150);ROW(151);
#undef ROW
        default:return 0.f;
    }
}
__global__ void project(const std::uint8_t* packed,const std::uint64_t* offsets,const std::int32_t* types,
        std::int64_t first,std::int64_t rows,std::int64_t column_first,std::int64_t columns,
        const float* x,float* y,std::int64_t slots,const std::int64_t* ids,std::int64_t routes,
        std::int64_t experts,bool routed_input) {
    const auto row=std::int64_t(blockIdx.x)*4+threadIdx.x/32;
    const auto slot=std::int64_t(blockIdx.y)+std::int64_t(blockIdx.z)*65535;
    if(row>=rows || slot>=slots)return;
    const auto expert=ids?ids[slot]:0;
    if(expert<0 || expert>=experts)return;
    const auto at=first+expert*rows+row;
    const auto type=types[at];
    if(!type)return;
    const auto* q=packed+offsets[at];
    const auto* input=x+(ids && !routed_input?slot/routes:slot)*columns;
    float sum=0;
    for(auto column=std::int64_t(threadIdx.x%32);column<columns;column+=32)
        sum=fmaf(input[column],decoded(q,type,column_first+column),sum);
    for(int distance=16;distance;distance/=2)sum+=__shfl_down_sync(0xffffffffu,sum,distance);
    if(!(threadIdx.x%32))y[slot*rows+row]+=sum;
}
__global__ void embedding(const std::uint8_t* packed,const std::uint64_t* offsets,const std::int32_t* types,
        std::int64_t first,std::int64_t rows,std::int64_t column_first,std::int64_t columns,
        const std::int64_t* ids,float* output,std::int64_t count) {
    const auto slot=std::int64_t(blockIdx.y)+std::int64_t(blockIdx.z)*65535;
    const auto column=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(slot>=count || column>=columns)return;
    const auto row=ids?ids[slot]:slot;
    if(row<0 || row>=rows)return;
    const auto at=first+row,type=std::int64_t(types[at]);
    if(type)output[slot*columns+column]+=decoded(packed+offsets[at],int(type),column_first+column);
}
void check() {
    const auto e=cudaGetLastError();
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
dim3 grid(std::int64_t x,std::int64_t slots) {
    if(x<1 || x>INT_MAX || slots<1 || slots>std::int64_t(65535)*65535)
        throw std::invalid_argument("row residual grid exceeds CUDA launch limits");
    return dim3(unsigned(x),unsigned(slots<65535?slots:65535),unsigned((slots+65534)/65535));
}
}
void add_row_residual(const void* packed,const std::uint64_t* offsets,const std::int32_t* types,
        std::int64_t first,std::int64_t rows,std::int64_t column_first,std::int64_t columns,
        const float* x,float* y,std::int64_t tokens,const std::int64_t* ids,std::int64_t routes,
        std::int64_t experts,bool routed_input,cudaStream_t stream) {
    if(!tokens)return;
    if(!packed || !offsets || !types || !x || !y || first<0 || rows<1 || columns<1 || column_first<0 ||
            tokens<0 || routes<1 || experts<1 || tokens>INT64_MAX/routes)
        throw std::invalid_argument("invalid row residual projection");
    project<<<grid((rows+3)/4,tokens*routes),128,0,stream>>>(static_cast<const std::uint8_t*>(packed),offsets,
        types,first,rows,column_first,columns,x,y,tokens*routes,ids,routes,experts,routed_input);check();
}
void add_residual_embedding(const void* packed,const std::uint64_t* offsets,const std::int32_t* types,
        std::int64_t first,std::int64_t rows,std::int64_t column_first,std::int64_t columns,
        const std::int64_t* ids,float* y,std::int64_t count,cudaStream_t stream) {
    if(!count)return;
    if(!packed || !offsets || !types || !y || first<0 || rows<1 || columns<1 || column_first<0 || count<0)
        throw std::invalid_argument("invalid residual row lookup");
    embedding<<<grid((columns+127)/128,count),128,0,stream>>>(static_cast<const std::uint8_t*>(packed),offsets,
        types,first,rows,column_first,columns,ids,y,count);check();
}
}
