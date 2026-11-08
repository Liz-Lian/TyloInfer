#include "qsa_tq.h"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <climits>
#include <algorithm>
#include "qsa_tq_indexer.cuh"
#include "qsa_tq_indexer_integer.cuh"

namespace tyloinfer::qsa {
namespace {
__global__ void publish_blocks(uint64_t* table,const int32_t* ids,const uint64_t* records,
                              uint64_t* previous,int count) {
    const int at=blockIdx.x*blockDim.x+threadIdx.x;
    if(at>=count)return;
    // Classic device atomics preserve the publication order without the
    // Windows cuda::atomic header's otherwise unnecessary SM70 build floor.
    __threadfence();
    const auto old=atomicExch(reinterpret_cast<unsigned long long*>(table+ids[at]),static_cast<unsigned long long>(records[at]));
    __threadfence();
    if(previous)previous[at]=old;
}
__global__ void snapshot_blocks(const uint64_t* table,const SelectedBlock* selected,TqBlockView* out,int total) {
    const int at=blockIdx.x*blockDim.x+threadIdx.x;
    if(at>=total)return;
    const auto block=selected[at];
    if(block.id<0 || !block.token_mask){out[at]=TqBlockView{};return;}
    const auto address=atomicCAS(reinterpret_cast<unsigned long long*>(const_cast<uint64_t*>(table)+block.id),0ULL,0ULL);
    __threadfence();
    // Every selected logical block must have a recent or historical image.
    // A missing record is a violated residency invariant, never a zero KV row.
    if(!address){asm volatile("trap;");return;}
    out[at]=*reinterpret_cast<const TqBlockView*>(address);
}
__device__ float warp_sum(float value) {
    for (int d=16;d;d/=2) value+=__shfl_xor_sync(0xffffffff,value,d);
    return value;
}
__device__ float warp_min(float value) {
    for (int d=16;d;d/=2) value=fminf(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}
__device__ float warp_max(float value) {
    for (int d=16;d;d/=2) value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,d));
    return value;
}

template<int W> __global__ void restore_modes(const float* keys,const float* values,
    const float* ks,const float* vs,float* out_k,float* out_v,int heads,int width) {
    const int d=threadIdx.x,head=blockIdx.x,block=blockIdx.y,kind=blockIdx.z;
    const float* input=kind?values:keys;const float* signs=kind?vs:ks;
    float* output=kind?out_v:out_k;
    __shared__ float work[4*W];
    for(int m=0;m<4;++m)work[m*W+d]=input[((int64_t(block)*heads+head)*4+m)*W+d];
    __syncthreads();
    for(int s=1;s<W;s*=2) {
        float next[4];
        for(int m=0;m<4;++m) {
            const float a=work[m*W+(d&~s)],b=work[m*W+(d|s)];next[m]=(d&s)?a-b:a+b;
        }
        __syncthreads();for(int m=0;m<4;++m)work[m*W+d]=next[m];__syncthreads();
    }
    if(d<width)for(int t=0;t<4;++t) {
        float x=0;for(int m=0;m<4;++m)x+=((__popc(unsigned(t&m))&1)?-.5f:.5f)*work[m*W+d];
        output[((int64_t(block)*4+t)*heads+head)*width+d]=x*rsqrtf(float(W))*signs[d];
    }
}
__global__ void group_selected(const int32_t* tokens,SelectedBlock* blocks,int columns,int count,int total) {
    const int at=blockIdx.x*blockDim.x+threadIdx.x;if(at>=total)return;
    const int q=at/count,slot=at%count;SelectedBlock result;
    for(int j=0;j<4 && slot*4+j<columns;++j) {
        const int token=tokens[int64_t(q)*columns+slot*4+j];if(token<0)continue;
        const int id=token/4;
        if(result.id>=0 && result.id!=id){asm volatile("trap;");return;}
        result.id=id;result.token_mask|=1u<<(token&3);
    }
    blocks[at]=result;
}
__device__ float read_input(const void* x,bool fp16,int64_t at) {
    return fp16?__half2float(reinterpret_cast<const half*>(x)[at]):reinterpret_cast<const float*>(x)[at];
}
__global__ void stage_append(const void* keys,const void* values,const void* index,bool kh,bool vh,bool ih,
    const float* old_k,const float* old_v,const float* old_i,float* k,float* v,float* raw,
    int tokens,int prior,int heads,int width,int iw,int64_t kt,int64_t ks,int64_t vt,int64_t vs,int64_t it,
    const int64_t* position=nullptr,int staged=0) {
    if(position)prior=int(*position&3);
    const int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,channels=int64_t(heads)*width,cols=2*channels+iw;
    if(at>=int64_t(position?staged:tokens+prior)*cols)return;
    const auto row=at/cols,col=at%cols;
    if(row>=tokens+prior) {
        if(col<channels)k[row*channels+col]=0;
        else if(col<2*channels)v[row*channels+col-channels]=0;
        else raw[row*iw+col-2*channels]=0;
        return;
    }
    if(col<channels)k[row*channels+col]=row<prior?old_k[row*channels+col]:read_input(keys,kh,(row-prior)*kt+(col/width)*ks+col%width);
    else if(col<2*channels) {
        const auto d=col-channels;v[row*channels+d]=row<prior?old_v[row*channels+d]:read_input(values,vh,(row-prior)*vt+(d/width)*vs+d%width);
    }else {
        const auto d=col-2*channels;raw[row*iw+d]=row<prior?old_i[row*iw+d]:read_input(index,ih,(row-prior)*it+d);
    }
}
__global__ void commit_tail(const float* k,const float* v,const float* raw,float* out_k,float* out_v,float* out_i,
    int tokens,int remainder,int channels,int iw) {
    const int64_t at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,cols=int64_t(2)*channels+iw;
    if(at>=int64_t(remainder)*cols)return;
    const auto row=at/cols,col=at%cols,source=tokens-remainder+row;
    if(col<channels)out_k[row*channels+col]=k[source*channels+col];
    else if(col<2LL*channels)out_v[row*channels+col-channels]=v[source*channels+col-channels];
    else out_i[row*iw+col-2*channels]=raw[source*iw+col-2*channels];
}
__global__ void pack_archive(const uint8_t* k,const uint8_t* v,const float* raw,uint8_t* output,int64_t bytes,int64_t values) {
    const auto at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<bytes)output[at]=k[at];
    else if(at<2*bytes)output[at]=v[at-bytes];
    else if(at-2*bytes<values)reinterpret_cast<half*>(output+2*bytes)[at-2*bytes]=__float2half_rn(raw[at-2*bytes]);
}
__global__ void archive_indexer(const int16_t* codes,const uint16_t* scales,const int16_t* zero,
        uint16_t* output,int width,int64_t start,const int64_t* position) {
    const auto block=position?*position/4:start+blockIdx.x;
    auto* destination=output+int64_t(blockIdx.x)*(width+2);
    for(int d=threadIdx.x;d<width;d+=blockDim.x)destination[d]=uint16_t(codes[block*width+d]);
    if(!threadIdx.x){destination[width]=scales[block];destination[width+1]=uint16_t(zero[block]);}
}
__global__ void copy_pending_tail(PendingTail source,PendingTail destination,int64_t keys,int64_t rotated,int64_t index) {
    auto at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<keys){destination.key[at]=source.key[at];return;}at-=keys;
    if(at<keys){destination.value[at]=source.value[at];return;}at-=keys;
    if(at<rotated){destination.rotated_key[at]=source.rotated_key[at];return;}at-=rotated;
    if(at<rotated){destination.rotated_value[at]=source.rotated_value[at];return;}at-=rotated;
    if(at<index)destination.index[at]=source.index[at];
}
__global__ void publish_graph_batch(PendingBatchPublication b) {
    const auto at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto first=*b.position/4;const int staged=int(*b.position&3)+b.tokens,complete=staged/4,remainder=staged%4;
    const int64_t channels=int64_t(b.heads)*b.width,rotated=int64_t(b.heads)*b.padded_width;
    const int64_t record_bytes=2LL*b.image_bytes+8LL*b.index_width+(b.archive_index?2LL*(b.index_width+2):0);
    if(at<int64_t(complete)*record_bytes) {
        const auto block=at/record_bytes,offset=at%record_bytes;
        const auto destination=b.destinations[(first+block)%b.destination_slots];
        uint8_t value=0;
        if(offset<b.image_bytes) {
            value=b.keys[block*b.image_bytes+offset];const_cast<uint8_t*>(destination.key)[offset]=value;
        }else if(offset<2LL*b.image_bytes) {
            value=b.values[block*b.image_bytes+offset-b.image_bytes];const_cast<uint8_t*>(destination.value)[offset-b.image_bytes]=value;
        }else if(offset<2LL*b.image_bytes+8LL*b.index_width) {
            const auto d=offset-2LL*b.image_bytes;
            const auto bits=__half_as_ushort(__float2half_rn(b.raw_index[block*4*b.index_width+d/2]));
            value=uint8_t(bits>>((d&1)*8));
        }else {
            const auto d=offset-2LL*b.image_bytes-8LL*b.index_width;
            const auto id=first+block;
            if(d<2LL*b.index_width)value=reinterpret_cast<const uint8_t*>(b.index_codes+id*b.index_width)[d];
            else if(d<2LL*b.index_width+2)value=reinterpret_cast<const uint8_t*>(b.index_scales+id)[d-2LL*b.index_width];
            else value=reinterpret_cast<const uint8_t*>(b.index_zero+id)[d-2LL*b.index_width-2];
        }
        b.archive[at]=value;
    }
    if(at<complete)b.views[first+at]=b.destinations[(first+at)%b.destination_slots];
    if(!at && remainder) {
        const auto stride=4LL*b.heads*b.padded_width;
        b.views[first+complete]={reinterpret_cast<const uint8_t*>(b.fp16_keys+complete*stride),
            reinterpret_cast<const uint8_t*>(b.fp16_values+complete*stride),0,0};
    }
    const auto columns=2*channels+b.index_width;
    if(at<int64_t(remainder)*columns) {
        const auto row=at/columns,col=at%columns,source=staged-remainder+row;
        if(col<channels)b.tail.key[row*channels+col]=b.raw_keys[source*channels+col];
        else if(col<2*channels)b.tail.value[row*channels+col-channels]=b.raw_values[source*channels+col-channels];
        else b.tail.index[row*b.index_width+col-2*channels]=b.raw_index[source*b.index_width+col-2*channels];
    }
    if(at<int64_t(remainder)*rotated) {
        const auto source=int64_t(staged-remainder)*rotated+at;
        b.tail.rotated_key[at]=b.rotated_keys[source];b.tail.rotated_value[at]=b.rotated_values[source];
    }
}
template<int W,bool Query=false> __global__ void append_pending(const void* keys,const void* values,const void* index,bool key_half,bool value_half,bool index_half,
    const int64_t* position,float* tail_k,float* tail_v,float* tail_i,float* rotated_k,float* rotated_v,const float* ks,const float* vs,
    half* pending_k,half* pending_v,TqBlockView* views,int heads,int width,int iw,int64_t khs,int64_t vhs,float* km,float* vm,const void* query,float* rotated_query,int query_heads,int64_t qhs,bool query_half) {
    const int d=threadIdx.x;
    if constexpr(Query)if(blockIdx.x>=2*heads) {
        const int head=blockIdx.x-2*heads;
        __shared__ float query_work[W];
        float x=d<width?read_input(query,query_half,int64_t(head)*qhs+d)*ks[d]:0.f;
        for(int stride=1;stride<W;stride*=2) {
            if(stride<32){const auto other=__shfl_xor_sync(0xffffffffu,x,stride);x=(d&stride)?other-x:x+other;}
            else {query_work[d]=x;__syncthreads();const auto other=query_work[d^stride];x=(d&stride)?other-x:x+other;__syncthreads();}
        }
        rotated_query[int64_t(head)*W+d]=x*rsqrtf(float(W));
        return;
    }
    const int head=Query?int(blockIdx.x)%heads:int(blockIdx.x);
    const int kind=Query?int(blockIdx.x)/heads:int(blockIdx.y),prior=int(*position&3);
    auto* tail=kind?tail_v:tail_k;const auto* signs=kind?vs:ks;
    const auto value=d<width?read_input(kind?values:keys,kind?value_half:key_half,int64_t(head)*(kind?vhs:khs)+d):0.f;
    // Only the newly appended slot is mutated. Repeated warmup at the same
    // position leaves the committed prefix intact.
    if(d<width)tail[(int64_t(prior)*heads+head)*width+d]=value;
    if(!kind && !head)for(int i=d;i<iw;i+=W)tail_i[prior*iw+i]=read_input(index,index_half,i);
    // Rotate each token once. Earlier slots retain their RHT values across
    // replays; only the newly appended slot is written, so graph warmup is
    // idempotent and rollback can replace an incomplete block safely.
    __shared__ float work[W];
    float rotated=value*signs[d];
    for(int stride=1;stride<W;stride*=2) {
        if(stride<32) {
            const auto other=__shfl_xor_sync(0xffffffff,rotated,stride);
            rotated=(d&stride)?other-rotated:rotated+other;
        }else {
            work[d]=rotated;__syncthreads();
            const auto other=work[d^stride];rotated=(d&stride)?other-rotated:rotated+other;
            __syncthreads();
        }
    }
    rotated*=rsqrtf(float(W));
    auto* saved=kind?rotated_v:rotated_k;
    saved[(int64_t(prior)*heads+head)*W+d]=rotated;
    float x[4];
    for(int t=0;t<4;++t)x[t]=t<prior?saved[(int64_t(t)*heads+head)*W+d]:t==prior?rotated:0.f;
    auto* pending=kind?pending_v:pending_k;
    for(int m=0;m<4;++m) {
        float mode=0;
        for(int t=0;t<4;++t)mode+=((__popc(unsigned(t&m))&1)?-.5f:.5f)*x[t];
        const auto at=(int64_t(head)*4+m)*W+d;
        pending[at]=__float2half_rn(mode);if(km && vm)(kind?vm:km)[at]=mode;
    }
    if(!kind && !head && !d)views[*position/4]={reinterpret_cast<const uint8_t*>(pending_k),reinterpret_cast<const uint8_t*>(pending_v),0,0};
}

__global__ void finalize_pending(const float* km,const float* vm,const float* raw,const TqBlockView* destination,
    TqBlockView* views,const int64_t* position,uint8_t* archive,int rows,int iw,int slots) {
    const int row=blockIdx.x,kind=blockIdx.y,d=threadIdx.x,lane=d&31,warp=d/32;
    const auto view=destination[(*position/4)%slots];
    if((view.key_bits&255u)==0 && (view.value_bits&255u)==0) {
        const int bytes=rows*512;
        auto* image=reinterpret_cast<half*>(const_cast<uint8_t*>(kind?view.value:view.key));
        auto* saved=reinterpret_cast<half*>(archive+kind*bytes);
        const auto at=int64_t(row)*256+d;const auto x=__float2half_rn((kind?vm:km)[at]);image[at]=x;saved[at]=x;
        if(!row && !kind) {
            for(int i=d;i<4*iw;i+=256)reinterpret_cast<half*>(archive+2*bytes)[i]=__float2half_rn(raw[i]);
            if(!d)views[*position/4]=view;
        }
        return;
    }
    const int meta=(3*rows+3)&~3,bytes=meta+256*rows;
    auto* image=const_cast<uint8_t*>(kind?view.value:view.key);auto* saved=archive+kind*bytes;
    const float x=(kind?vm:km)[int64_t(row)*256+d];
    const auto lo=warp_min(fminf(x,0.f)),hi=warp_max(fmaxf(x,0.f));
    __shared__ float ranges[16];
    if(!lane){ranges[warp]=lo;ranges[8+warp]=hi;}__syncthreads();
    float low=0,high=0;for(int w=0;w<8;++w){low=fminf(low,ranges[w]);high=fmaxf(high,ranges[8+w]);}
    const float step=fmaxf((high-low)/255.f,0x1p-24f);auto scale=__float2half_rn(step);
    if(__half2float(scale)<step)scale=__ushort_as_half(__half_as_ushort(scale)+1);
    const float s=__half2float(scale);const int zero=max(-128,min(127,-128-int(nearbyintf(low/s))));
    const auto code=uint8_t(int8_t(max(-128,min(127,int(nearbyintf(x/s))+zero))));
    image[meta+row*256+d]=code;saved[meta+row*256+d]=code;
    if(!d) {
        reinterpret_cast<half*>(image)[row]=scale;reinterpret_cast<half*>(saved)[row]=scale;
        image[2*rows+row]=uint8_t(int8_t(zero));saved[2*rows+row]=uint8_t(int8_t(zero));
    }
    if(!row) {
        if(d< meta-3*rows){image[3*rows+d]=0;saved[3*rows+d]=0;}
        if(!kind) {
            for(int i=d;i<4*iw;i+=256)reinterpret_cast<half*>(archive+2*bytes)[i]=__float2half_rn(raw[i]);
            if(!d)views[*position/4]=view;
        }
    }
}
__global__ void initialize_destinations(uint8_t* payload,TqBlockView* output,int bytes,int slots,unsigned tag=254) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=slots)return;
    output[id]={payload+int64_t(2*id)*bytes,payload+int64_t(2*id+1)*bytes,tag,tag};
}
__global__ void publish_contiguous_views(TqBlockView base,TqBlockView* output,int bytes,int count) {
    const auto id=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    output[id]={base.key+id*bytes,base.value+id*bytes,base.key_bits,base.value_bits};
}

template<int W,bool F16> __global__ void prepare_modes(const void* keys,const void* values,
    const float* ks,const float* vs,float* km,float* vm,half* pending_k,half* pending_v,
    int tokens,int heads,int width,float* rotated_k,float* rotated_v) {
    const int d=threadIdx.x,head=blockIdx.x,block=blockIdx.y,kind=blockIdx.z;
    const auto* input=kind?values:keys;
    const auto* signs=kind?vs:ks;
    auto* output=kind?vm:km;
    auto* pending=kind?pending_v:pending_k;
    float x[4];
    for(int t=0;t<4;++t) {
        const auto token=block*4+t;
        const auto at=(int64_t(token)*heads+head)*width+d;
        x[t]=token<tokens && d<width?(F16?__half2float(reinterpret_cast<const half*>(input)[at]):
            reinterpret_cast<const float*>(input)[at])*signs[d]:0;
    }
    __shared__ float work[4*W];
    for(int mode=0;mode<4;++mode) {
        float value=0;
        for(int t=0;t<4;++t)value+=((__popc(unsigned(t&mode))&1)?-.5f:.5f)*x[t];
        work[mode*W+d]=value;
    }
    __syncthreads();
    for(int stride=1;stride<W;stride*=2) {
        float next[4];
        for(int mode=0;mode<4;++mode) {
            const float a=work[mode*W+(d&~stride)],b=work[mode*W+(d|stride)];
            next[mode]=(d&stride)?a-b:a+b;
        }
        __syncthreads();
        for(int mode=0;mode<4;++mode)work[mode*W+d]=next[mode];
        __syncthreads();
    }
    for(int mode=0;mode<4;++mode) {
        const auto at=((int64_t(block)*heads+head)*4+mode)*W+d;
        const auto value=work[mode*W+d]*rsqrtf(float(W));
        output[at]=value;if(pending)pending[at]=__float2half_rn(value);
    }
    if(auto* rotated=kind?rotated_v:rotated_k)for(int t=0;t<4;++t) {
        float value=0;
        for(int m=0;m<4;++m)value+=((__popc(unsigned(t&m))&1)?-.5f:.5f)*work[m*W+d];
        rotated[((int64_t(block)*4+t)*heads+head)*W+d]=value*rsqrtf(float(W));
    }
}

template<bool Records=false> __global__ void affine_encode(const float* input,int16_t* output,uint16_t* scales,
                            int16_t* zero,int width,const int64_t* position=nullptr,int capacity=INT_MAX) {
    const int input_row=blockIdx.x,lane=threadIdx.x;
    const int64_t row=position?*position/4+input_row:input_row;
    if(row>=capacity)return;
    float lo=0,hi=0;
    for (int d=lane;d<width;d+=32) {
        const auto x=input[int64_t(input_row)*width+d];lo=fminf(lo,x);hi=fmaxf(hi,x);
    }
    lo=warp_min(lo);hi=warp_max(hi);
    // Keep the range subtraction and reciprocal multiplication separately
    // rounded, including vectors exactly on an FP16 scale boundary.
    const float step=fmaxf(__fmul_rn(__fsub_rn(hi,lo),1.f/65535.f),0x1p-24f);
    auto scale=__float2half_rn(step);
    if (__half2float(scale)<step) scale=__ushort_as_half(__half_as_ushort(scale)+1);
    const float s=__half2float(scale);
    const int zp=max(-32768,min(32767,-32768-int(nearbyintf(lo/s))));
    if (!lane) {
        if constexpr(Records){output[row*(width+2)+width]=int16_t(__half_as_ushort(scale));output[row*(width+2)+width+1]=int16_t(zp);}
        else {scales[row]=__half_as_ushort(scale);zero[row]=int16_t(zp);}
    }
    for (int d=lane;d<width;d+=32) {
        const int code=max(-32768,min(32767,int(nearbyintf(input[int64_t(input_row)*width+d]/s))+zp));
        output[int64_t(row)*(width+(Records?2:0))+d]=int16_t(code);
    }
}

// A single-query warp shares its query across four keys and four heads.
// Aligned pairs reduce input loads; scalar loads handle arbitrary alignment.
// Multi-query warps retain two keys and one head at a time to bound register
// lifetimes. Both paths accumulate in FP32 before descending XOR reductions.
constexpr int affine_score_threads=256;
template<int QT,bool Records=false,bool Pairs=false> __global__ __launch_bounds__(affine_score_threads,QT==1?4:1)
void affine_scores(const float* query,const int16_t* keys,const uint16_t* scales,
    const int16_t* zero,const int64_t* positions,float* output,int queries,int heads,int width,int blocks,int pool=4) {
    constexpr int Rows=QT==1?4:2;
    const int lane=threadIdx.x&31,base=(blockIdx.x*(affine_score_threads/32)+threadIdx.x/32)*Rows,first=blockIdx.y*QT;
    if(base>=blocks)return;
    // Unpacked index rows each represent four tokens.
    const int pooling=Records?pool:4;
    bool active[QT][Rows];bool any=false;
    #pragma unroll
    for(int q=0;q<QT;++q){
        const auto limit=first+q<queries?(positions?(positions[first+q]+1)/pooling:blocks):0;
        #pragma unroll
        for(int r=0;r<Rows;++r){active[q][r]=first+q<queries && base+r<blocks && base+r<limit;any|=active[q][r];}
    }
    if(!any){if(!lane)for(int q=0;q<QT;++q)if(first+q<queries)
        for(int r=0;r<Rows;++r)if(base+r<blocks)output[int64_t(first+q)*blocks+base+r]=-CUDART_INF_F;return;}
    const int stride=width+(Records?2:0);
    float scale[Rows];int zp[Rows];
    #pragma unroll
    for(int r=0;r<Rows;++r){
        scale[r]=base+r<blocks?__half2float(__ushort_as_half(Records?uint16_t(keys[int64_t(base+r)*stride+width]):scales[base+r])):0.f;
        zp[r]=base+r<blocks?(Records?keys[int64_t(base+r)*stride+width+1]:zero[base+r]):0;
    }
    constexpr int HT=QT==1?4:1;float score[QT][Rows]={};
    for(int first_head=0;first_head<heads;first_head+=HT){
        float dot[QT][HT][Rows]={};
        if constexpr(Pairs) {
            for(int d=2*lane;d<width;d+=64) {
                float k0[Rows],k1[Rows];
                #pragma unroll
                for(int r=0;r<Rows;++r) {
                    short2 packed{};
                    if(base+r<blocks) {
                        const auto* pointer=keys+int64_t(base+r)*stride+d;
                        packed=*reinterpret_cast<const short2*>(pointer);
                    }
                    k0[r]=float(int(packed.x)-zp[r])*scale[r];
                    k1[r]=float(int(packed.y)-zp[r])*scale[r];
                }
                #pragma unroll
                for(int q=0;q<QT;++q)if(first+q<queries) {
                    #pragma unroll
                    for(int h=0;h<HT;++h)if(first_head+h<heads) {
                        const auto* pointer=query+(int64_t(first+q)*heads+first_head+h)*width+d;
                        const auto x=*reinterpret_cast<const float2*>(pointer);
                        #pragma unroll
                        for(int r=0;r<Rows;++r)if(QT==1 || active[q][r]) {
                            dot[q][h][r]=fmaf(x.x,k0[r],dot[q][h][r]);
                            dot[q][h][r]=fmaf(x.y,k1[r],dot[q][h][r]);
                        }
                    }
                }
            }
        }else {
            for(int d=lane;d<width;d+=32){
                float k[Rows];
                #pragma unroll
                for(int r=0;r<Rows;++r)k[r]=base+r<blocks?float(int(keys[int64_t(base+r)*stride+d])-zp[r])*scale[r]:0.f;
                #pragma unroll
                for(int q=0;q<QT;++q)if(first+q<queries){
                    #pragma unroll
                    for(int h=0;h<HT;++h)if(first_head+h<heads){
                        const float x=query[(int64_t(first+q)*heads+first_head+h)*width+d];
                        #pragma unroll
                        // Singleton masks are applied at publication. Keeping
                        // their predicates outside the FMA loop reduces live
                        // registers without changing any active score's order.
                        for(int r=0;r<Rows;++r)if(QT==1 || active[q][r])dot[q][h][r]=fmaf(x,k[r],dot[q][h][r]);
                    }
                }
            }
        }
        #pragma unroll
        for(int q=0;q<QT;++q){
            #pragma unroll
            for(int h=0;h<HT;++h)if(QT!=1 || first_head+h<heads){
                #pragma unroll
                for(int r=0;r<Rows;++r)score[q][r]+=fmaxf(warp_sum(dot[q][h][r]),0.f);
            }
        }
    }
    if(!lane){
        #pragma unroll
        for(int q=0;q<QT;++q)if(first+q<queries){
            #pragma unroll
            for(int r=0;r<Rows;++r)if(base+r<blocks)output[int64_t(first+q)*blocks+base+r]=
                active[q][r]?score[q][r]*(Records?rsqrtf(float(width)):1.f):-CUDART_INF_F;
        }
    }
}


template<bool Records=false> static int launch_affine_scores_pairs(const float* query,const int16_t* keys,
        const uint16_t* scales,const int16_t* zero,const int64_t* positions,float* output,
        int queries,int heads,int width,int blocks,cudaStream_t stream,int pool=4) {
    // Every key row and query head must preserve short2/float2 alignment.
    if(queries!=1 || (width&1) || (reinterpret_cast<std::uintptr_t>(keys)&3u) ||
            (reinterpret_cast<std::uintptr_t>(query)&7u))return int(cudaErrorNotSupported);
    affine_scores<1,Records,true><<<unsigned((int64_t(blocks)+31)/32),affine_score_threads,0,stream>>>(
        query,keys,scales,zero,positions,output,queries,heads,width,blocks,pool);
    const auto status=cudaGetLastError();
    if(status==cudaSuccess)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "native_indexer_affine16_pairs actual_match=1 queries=%d blocks=%d heads=%d width=%d records=%d vector_coordinates=2 fp32=1\n",
            queries,blocks,heads,width,int(Records));
    }
    return int(status);
}

__global__ void affine_record_decode(const int16_t* records,float* output,int rows,int width) {
    const auto at=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=int64_t(rows)*width)return;
    const auto row=at/width;const auto* record=records+row*(width+2);
    output[at]=float(int(record[at%width])-int(record[width+1]))*__half2float(__ushort_as_half(uint16_t(record[width])));
}

__global__ void online8(const float* input,const half* book,const uint8_t* candidates,
    int bins,float bin_min,float bin_step,float book_rms,float relative,
    half* anchors,uint8_t* payload,uint8_t* trace,float* sse) {
    const int tid=threadIdx.x,row=blockIdx.x,lane=tid&31,warp=tid/32;
    const float x=input[int64_t(row)*256+tid];
    __shared__ float old[256],sums[8],cross[8],targets[256],scale;
    __shared__ unsigned long long minimum[8];
    __shared__ int boundary;
    float energy=warp_sum(x*x);
    if(!lane)sums[warp]=energy;__syncthreads();
    if(!tid) {
        float total=0;for(int w=0;w<8;++w)total+=sums[w];
        const float anchor=fmaxf(sqrtf(total/256)/book_rms/relative,0x1p-24f);
        anchors[row]=__float2half_rn(anchor);scale=total>0?__half2float(anchors[row])*relative:1.f;
    }
    __syncthreads();
    targets[tid]=x/scale;__syncthreads();
    auto* history=trace+int64_t(row)*256*256;
    auto* symbols=payload+int64_t(row)*256;
    for(int pass=0;pass<2;++pass) {
        old[tid]=(pass && tid!=boundary)?CUDART_INF_F:0;
        __syncthreads();
        const int steps=pass?256:32;
        for(int t=0;t<steps;++t) {
            const float target=targets[pass?t:224+t];
            const int bin=max(0,min(bins-1,int(nearbyintf((target-bin_min)/bin_step))));
            const auto packed_choices=*reinterpret_cast<const unsigned long long*>(candidates+(int64_t(bin)*256+tid)*8);
            float best=CUDART_INF_F;int predecessor=0;
            #pragma unroll
            for(int c=0;c<8;++c) {
                const int previous=int((packed_choices>>(c*8))&255);
                const float delta=__half2float(book[(previous<<8)|tid])-target;
                const float cost=fmaf(delta,delta,old[previous]);
                if(cost<best){best=cost;predecessor=previous;}
            }
            __syncthreads();
            old[tid]=best;
            if(pass)history[t*256+tid]=uint8_t(predecessor);
            __syncthreads();
        }
        if(!pass) {
            auto key=(static_cast<unsigned long long>(__float_as_uint(old[tid]))<<32)|unsigned(tid);
            for(int d=16;d;d/=2)key=min(key,__shfl_down_sync(0xffffffff,key,d));
            if(!lane)minimum[warp]=key;__syncthreads();
            if(!tid) {auto best=minimum[0];for(int w=1;w<8;++w)best=min(best,minimum[w]);boundary=int(best&255);}
            __syncthreads();
        }
    }
    if(!tid) {
        // The endpoint equals the starting symbol, so all stored states are
        // closed inside this 256-element ring; no cross-block state is hidden.
        int state=boundary;
        for(int t=255;t>=0;--t) {symbols[255-t]=uint8_t(state);state=history[t*256+state];}
    }
    __syncthreads();
    const int low=symbols[255-tid],high=symbols[(256-tid)&255];
    const float y=__half2float(book[(high<<8)|low]);
    float numerator=warp_sum(x*y),denominator=warp_sum(y*y);
    if(!lane){cross[warp]=numerator;sums[warp]=denominator;}__syncthreads();
    if(!tid) {
        float a=0,b=0;for(int w=0;w<8;++w){a+=cross[w];b+=sums[w];}
        const auto anchor=__float2half_rn(a>0?fmaxf(a/fmaxf(b*relative,1e-30f),0x1p-24f):0.f);
        anchors[row]=anchor;scale=__half2float(anchor)*relative;
    }
    __syncthreads();
    const float error=x-__half2float(__float2half_rn(y*scale));
    const float loss=warp_sum(error*error);
    if(!lane)sums[warp]=loss;__syncthreads();
    if(!tid){float total=0;for(int w=0;w<8;++w)total+=sums[w];sse[row]=total;}
}

__device__ int row_bits(const uint8_t* data,int rows,int row,int uniform) {
    if (uniform!=255) return uniform;
    const auto* map=data+2*rows+(rows+1)/2;
    const int bit=3*row,byte=bit/8,shift=bit&7;
    unsigned code=map[byte]>>shift;
    if (shift>5) code|=unsigned(map[byte+1])<<(8-shift);
    return 1+(code&7);
}

template<int W,bool F16> __global__ void prepare_affine8(const void* keys,const void* values,
    const float* ks,const float* vs,uint8_t* key_images,uint8_t* value_images,
    int tokens,int heads,int width) {
    const int d=threadIdx.x,head=blockIdx.x,block=blockIdx.y,kind=blockIdx.z;
    const auto* input=kind?values:keys;
    const auto* signs=kind?vs:ks;
    float x[4],y[4];
    for(int t=0;t<4;++t) {
        const int token=block*4+t;
        const auto at=(int64_t(token)*heads+head)*width+d;
        x[t]=token<tokens && d<width?(F16?__half2float(reinterpret_cast<const half*>(input)[at]):
            reinterpret_cast<const float*>(input)[at])*signs[d]:0.f;
    }
    for(int m=0;m<4;++m) {
        y[m]=0;
        for(int t=0;t<4;++t)y[m]+=((__popc(unsigned(t&m))&1)?-.5f:.5f)*x[t];
    }
    __shared__ float work[4*W],ranges[8*32];
    for(int stride=1;stride<W;stride*=2) {
        if(stride<32)for(int m=0;m<4;++m) {
            const float other=__shfl_xor_sync(0xffffffff,y[m],stride);
            y[m]=(d&stride)?other-y[m]:y[m]+other;
        } else {
            for(int m=0;m<4;++m)work[m*W+d]=y[m];
            __syncthreads();
            for(int m=0;m<4;++m){const float other=work[m*W+(d^stride)];y[m]=(d&stride)?other-y[m]:y[m]+other;}
            __syncthreads();
        }
    }
    const int ring=d/256,lane=d&31,warp=d/32;
    for(int m=0;m<4;++m) {
        y[m]*=rsqrtf(float(W));
        const float lo=warp_min(fminf(y[m],0.f)),hi=warp_max(fmaxf(y[m],0.f));
        if(!lane){ranges[m*64+warp]=lo;ranges[m*64+32+warp]=hi;}
    }
    __syncthreads();
    const int rows=heads*4*(W/256),meta=(3*rows+3)&~3;
    auto* image=(kind?value_images:key_images)+int64_t(block)*(meta+rows*256);
    for(int m=0;m<4;++m) {
        float lo=0,hi=0;
        for(int w=ring*8;w<(ring+1)*8;++w){lo=fminf(lo,ranges[m*64+w]);hi=fmaxf(hi,ranges[m*64+32+w]);}
        const float step=fmaxf((hi-lo)/255.f,0x1p-24f);
        auto scale=__float2half_rn(step);
        if(__half2float(scale)<step)scale=__ushort_as_half(__half_as_ushort(scale)+1);
        const float s=__half2float(scale);
        const int zero=max(-128,min(127,-128-int(nearbyintf(lo/s))));
        const int row=(head*4+m)*(W/256)+ring;
        if(!(d&255)) {
            reinterpret_cast<half*>(image)[row]=scale;
            image[2*rows+row]=uint8_t(int8_t(zero));
        }
        image[meta+row*256+(d&255)]=uint8_t(int8_t(max(-128,min(127,int(nearbyintf(y[m]/s))+zero))));
    }
}

struct TqRowView {const uint32_t* words=nullptr;float scale=0;int bits=0;};
__device__ TqRowView tq_row(const uint8_t* data,int uniform,int row,int rows,const half* relative) {
    const unsigned state_bits=unsigned(uniform)>>8;
    uniform&=255;
    if (!uniform) return {reinterpret_cast<const uint32_t*>(data+row*512),0,0};
    if (uniform==254) {
        const int meta=(3*rows+3)&~3;
        const int zero=int(reinterpret_cast<const int8_t*>(data+2*rows)[row]);
        return {reinterpret_cast<const uint32_t*>(data+meta+row*256),
            __half2float(reinterpret_cast<const half*>(data)[row]),254|((zero+128)<<8)};
    }
    const int bits=row_bits(data,rows,row,uniform);
    int offset=0;
    if (uniform!=255) offset=row*8*bits;
    else for (int previous=0;previous<row;++previous)offset+=8*row_bits(data,rows,previous,uniform);
    int metadata=2*rows+(rows+1)/2+(uniform==255?(3*rows+7)/8:0);
    metadata=(metadata+3)&~3;
    const auto* words=reinterpret_cast<const uint32_t*>(data+metadata)+offset;
    const float anchor=__half2float(reinterpret_cast<const half*>(data)[row]);
    const auto selector=(data[2*rows+row/2]>>((row&1)*4))&15;
    if(state_bits>16){asm volatile("trap;");return {};}
    return {words,__fmul_rn(anchor,__half2float(relative[selector])),bits|int(state_bits<<16)};
}
__device__ __forceinline__ float tq_state_value(uint32_t state,float scale) {
    const uint32_t product=state*0x83dcd12du;
#if __CUDA_ARCH__ >= 610
    const unsigned sum=__dp4a(product,0x01010101u,0u);
#else
    const unsigned sum=(product&255)+((product>>8)&255)+((product>>16)&255)+(product>>24);
#endif
    const auto base=__float2half_rn(fmaf(float(sum+1024),
        __half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931))));
    return __half2float(__float2half_rn(__fmul_rn(__half2float(base),scale)));
}

__device__ __forceinline__ float tq_value(const TqRowView row,int col) {
    if (!row.bits) return __half2float(reinterpret_cast<const half*>(row.words)[col]);
    if ((row.bits&255)==254) return float(int(reinterpret_cast<const int8_t*>(row.words)[col])-(row.bits>>8)+128)*row.scale;
    const auto* words=row.words;
    const auto bits=row.bits&255;
    const int bit=(255-col)*bits,word=bit/32,shift=bit&31;
    // word is already in [0,8*bits); only its successor can wrap. Avoid
    // variable integer division in every decoded K/V coordinate.
    const int next=word+1;
    const auto lo=words[word],hi=words[next==8*bits?0:next];
    const unsigned profile=unsigned(row.bits)>>16;
    const uint32_t state=__funnelshift_r(lo,hi,shift)&((1u<<(profile?profile:16))-1);
    return tq_state_value(state,row.scale);
}

// Decode packed adjacent coordinates without expanding persistent KV.
__device__ __forceinline__ float4 tq_values4(const TqRowView row,int col) {
    if((row.bits&255)==254) {
        const uint32_t x=row.words[col/4];
        const int z=(row.bits>>8)-128;
        return make_float4(float(int(int8_t(x&255))-z)*row.scale,
            float(int(int8_t((x>>8)&255))-z)*row.scale,
            float(int(int8_t((x>>16)&255))-z)*row.scale,
            float(int(int8_t(x>>24))-z)*row.scale);
    }
    if(!row.bits) {
        const auto* x=reinterpret_cast<const half*>(row.words)+col;
        const half2 a=*reinterpret_cast<const half2*>(x),b=*reinterpret_cast<const half2*>(x+2);
        const auto p=__half22float2(a),q=__half22float2(b);
        return make_float4(p.x,p.y,q.x,q.y);
    }
    const unsigned bits=unsigned(row.bits)&255u,profile=unsigned(row.bits)>>16;
    const unsigned state_bits=profile?profile:16u;
    const unsigned first=(252u-unsigned(col))*bits,word=first/32u,shift=first&31u;
    // Four adjacent coordinates share a circular bitstream window. A rare
    // wide state/unaligned start spans three words and uses scalar reads.
    if(shift+3u*bits+state_bits>64u)
        return make_float4(tq_value(row,col),tq_value(row,col+1),tq_value(row,col+2),tq_value(row,col+3));
    const auto lo=row.words[word],hi=row.words[word+1u==8u*bits?0u:word+1u];
    const auto mask=(1u<<state_bits)-1u;
    float value[4];
    #pragma unroll
    for(unsigned i=0;i<4;++i) {
        const unsigned offset=shift+(3u-i)*bits;
        const auto state=(offset<32u?__funnelshift_r(lo,hi,offset):(hi>>(offset-32u)))&mask;
        value[i]=tq_state_value(state,row.scale);
    }
    return make_float4(value[0],value[1],value[2],value[3]);
}

__global__ void decode_modes(const TqBlockView* blocks,float* keys,float* values,const half* relative,int rows) {
    const int row=blockIdx.x%rows,block=blockIdx.x/rows,kind=blockIdx.y;
    const auto view=blocks[block];
    __shared__ TqRowView source;
    if(!threadIdx.x)source=tq_row(kind?view.value:view.key,kind?view.value_bits:view.key_bits,row,rows,relative);
    __syncthreads();
    (kind?values:keys)[(int64_t(block)*rows+row)*256+threadIdx.x]=tq_value(source,threadIdx.x);
}

__global__ void clear_requested_blocks(uint32_t* requested,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)requested[i]=0;
}
__global__ void mark_requested_blocks(const SelectedBlock* selected,uint32_t* requested,int64_t total,int count) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=total)return;
    const auto b=selected[i];if(b.id>=0 && b.id<count && b.token_mask)atomicExch(requested+b.id,1u);
}

// Decode once for a batch, with the same FP32 H4 sum and final FP16
// rounding as sparse_attention_mma8. Layout: [block,head,token,padded].
template<int W> __global__ void stage_attention_blocks(const TqBlockView* blocks,
    const half* relative,half* keys,half* values,int heads,const uint32_t* requested) {
    constexpr int Rings=W/256;
    const int block=blockIdx.x/heads,head=blockIdx.x%heads,kind=blockIdx.y;
    if(!requested[block])return;
    const auto view=blocks[block];const auto* image=kind?view.value:view.key;
    const auto flags=kind?view.value_bits:view.key_bits;
    __shared__ TqRowView source[4*Rings];
    if(threadIdx.x<4*Rings)source[threadIdx.x]=image
        ?tq_row(image,flags,head*4*Rings+threadIdx.x,heads*4*Rings,relative):TqRowView{};
    __syncthreads();
    for(int d=4*threadIdx.x;d<W;d+=4*blockDim.x) {
        float4 mode[4];
        #pragma unroll
        for(int m=0;m<4;++m)mode[m]=image?tq_values4(source[m*Rings+d/256],d&255):make_float4(0,0,0,0);
        #pragma unroll
        for(int t=0;t<4;++t) {
            float4 a=make_float4(0,0,0,0);
            #pragma unroll
            for(int m=0;m<4;++m) {
                const float sign=(__popc(unsigned(t&m))&1)?-.5f:.5f;
                a.x+=sign*mode[m].x;a.y+=sign*mode[m].y;a.z+=sign*mode[m].z;a.w+=sign*mode[m].w;
            }
            auto* out=(kind?values:keys)+((int64_t(block)*heads+head)*4+t)*W+d;
            *reinterpret_cast<half2*>(out)=__floats2half2_rn(a.x,a.y);
            *reinterpret_cast<half2*>(out+2)=__floats2half2_rn(a.z,a.w);
        }
    }
}

template<int W> __global__ void rotate_query(const void* input,const float* signs,float* output,
    int heads,int width,int64_t token_stride,int64_t head_stride,bool fp16) {
    const int d=threadIdx.x,row=blockIdx.x,token=row/heads,head=row%heads;
    __shared__ float work[W];
    float x=d<width?read_input(input,fp16,int64_t(token)*token_stride+int64_t(head)*head_stride+d)*signs[d]:0;
    for (int stride=1;stride<W;stride*=2) {
        if(stride<32){const auto other=__shfl_xor_sync(0xffffffff,x,stride);x=(d&stride)?other-x:x+other;}
        else {work[d]=x;__syncthreads();const auto other=work[d^stride];x=(d&stride)?other-x:x+other;__syncthreads();}
    }
    output[int64_t(row)*W+d]=x*rsqrtf(float(W));
}

template<int W,bool Snapshots,int SG> __device__ __forceinline__ void sparse_attention_impl(const float* query,const half* relative,
    const TqBlockView* blocks,const SelectedBlock* selected,const int64_t* positions,
    float* partial,float* stats,float* block_log_mass,
    int heads,int kv_heads,int width,int count,int parts,int head_offset=0) {
    constexpr int N=W/32;
    const int lane=threadIdx.x&31,group=threadIdx.x/32;
    const int part=blockIdx.x,kv_group=blockIdx.y,token=blockIdx.z;
    const int gqa=heads/kv_heads,groups=(gqa+SG-1)/SG;
    const int kh=kv_group/groups,local=(kv_group%groups)*SG+group+head_offset,head=kh*gqa+local;
    const bool active=local<gqa;
    const int64_t row=int64_t(token)*heads+head;
    const int span=(count+parts-1)/parts;
    float q[N],acc[N];
    for (int i=0;i<N;++i) {q[i]=active?query[row*W+lane+32*i]:0;acc[i]=0;}
    __shared__ __align__(16) float key[4*W],value[4*W];
    __shared__ TqRowView key_rows[4*(W/256)],value_rows[4*(W/256)];
    float maximum=-CUDART_INF_F,mass=0;
    const auto boundary=positions[token];
    const int begin=part*span,end=min(count,(part+1)*span);
    __shared__ SelectedBlock chosen[2];
    __shared__ TqBlockView images[2];
    if(!threadIdx.x && begin<end) {
        const auto item=selected[int64_t(token)*count+begin];chosen[0]=item;
        images[0]=item.id>=0 && item.token_mask?blocks[Snapshots?int64_t(token)*count+begin:item.id]:TqBlockView{};
    }
    __syncthreads();
    for (int slot=begin;slot<end;++slot) {
        const auto selected_block=chosen[(slot-begin)&1];
        const bool valid=selected_block.id>=0 && selected_block.token_mask;
        const auto view=images[(slot-begin)&1];
        const int rings=W/256,rows=kv_heads*4*rings;
        // Canonical Affine8 has a fixed row address. Construct each packed
        // row descriptor at the consumer, without the descriptor publication
        // barrier needed by variable-width TQ records.
        if(valid && view.key_bits==254 && view.value_bits==254) {
            const int meta=(3*rows+3)&~3;
            for(int at=4*threadIdx.x;at<4*W;at+=blockDim.x*4) {
                const int mode=at/W,d=at%W,r=kh*4*rings+mode*rings+d/256;
                const int kz=int(reinterpret_cast<const int8_t*>(view.key+2*rows)[r]);
                const int vz=int(reinterpret_cast<const int8_t*>(view.value+2*rows)[r]);
                const TqRowView k{reinterpret_cast<const uint32_t*>(view.key+meta+r*256),__half2float(reinterpret_cast<const half*>(view.key)[r]),254|((kz+128)<<8)};
                const TqRowView v{reinterpret_cast<const uint32_t*>(view.value+meta+r*256),__half2float(reinterpret_cast<const half*>(view.value)[r]),254|((vz+128)<<8)};
                reinterpret_cast<float4*>(key)[at/4]=tq_values4(k,d&255);
                reinterpret_cast<float4*>(value)[at/4]=tq_values4(v,d&255);
            }
        } else {
            if(threadIdx.x<4*rings) {
                const auto row=kh*4*rings+threadIdx.x;
                key_rows[threadIdx.x]=valid?tq_row(view.key,view.key_bits,row,rows,relative):TqRowView{};
                value_rows[threadIdx.x]=valid?tq_row(view.value,view.value_bits,row,rows,relative):TqRowView{};
            }
            __syncthreads();
            for (int at=4*threadIdx.x;at<4*W;at+=blockDim.x*4) {
                const int mode=at/W,d=at%W,r=mode*rings+d/256;
                reinterpret_cast<float4*>(key)[at/4]=valid?tq_values4(key_rows[r],d&255):make_float4(0,0,0,0);
                reinterpret_cast<float4*>(value)[at/4]=valid?tq_values4(value_rows[r],d&255):make_float4(0,0,0,0);
            }
        }
        __syncthreads();
        float mode_score[4],scores[4],weights[4];
        for (int m=0;m<4;++m) {
            float dot=0;
            for (int i=0;i<N;++i)dot=fmaf(q[i],key[m*W+lane+i*32],dot);
            mode_score[m]=warp_sum(dot)*rsqrtf(float(width));
        }
        float block_top=-CUDART_INF_F;
        for (int t=0;t<4;++t) {
            float dot=0;
            for (int m=0;m<4;++m)dot+=((__popc(unsigned(t&m))&1)?-.5f:.5f)*mode_score[m];
            const bool causal=valid && (selected_block.token_mask&(1u<<t)) && int64_t(selected_block.id)*4+t<=boundary;
            scores[t]=causal?dot:-CUDART_INF_F;block_top=fmaxf(block_top,scores[t]);
        }
        float block_mass=0;
        for (int t=0;t<4;++t){weights[t]=isfinite(scores[t])?expf(scores[t]-block_top):0;block_mass+=weights[t];}
        if(active && !lane) block_log_mass[row*count+slot]=block_mass>0?block_top+logf(block_mass):-CUDART_INF_F;
        const float top=fmaxf(maximum,block_top);
        const float factor=isfinite(maximum)?expf(maximum-top):0;
        mass*=factor;
        const float block_factor=isfinite(block_top)?expf(block_top-top):0;
        for (int t=0;t<4;++t) {weights[t]*=block_factor;mass+=weights[t];}
        maximum=top;
        float modes[4];
        for (int m=0;m<4;++m) {
            float sum=0;
            for (int t=0;t<4;++t)sum+=((__popc(unsigned(t&m))&1)?-.5f:.5f)*weights[t];
            modes[m]=sum;
        }
        for (int i=0;i<N;++i) {
            acc[i]*=factor;
            for (int m=0;m<4;++m)acc[i]=fmaf(modes[m],value[m*W+lane+i*32],acc[i]);
        }
        // The other descriptor slot is no longer consumed by any warp.
        // Publish the next selected image with the existing end-of-tile
        // barrier, keeping the format branch uniform across this CTA.
        if(!threadIdx.x && slot+1<end) {
            const int next=((slot-begin)&1)^1;
            const auto item=selected[int64_t(token)*count+slot+1];chosen[next]=item;
            images[next]=item.id>=0 && item.token_mask?blocks[Snapshots?int64_t(token)*count+slot+1:item.id]:TqBlockView{};
        }
        __syncthreads();
    }
    if (active) {
        const int64_t at=row*parts+part;
        for (int i=0;i<N;++i)partial[at*W+lane+i*32]=acc[i];
        if (!lane) {stats[at*2]=maximum;stats[at*2+1]=mass;}
    }
}

template<int W,bool Snapshots,int SG> __global__ void sparse_attention(const float* query,const half* relative,
    const TqBlockView* blocks,const SelectedBlock* selected,const int64_t* positions,
    float* partial,float* stats,float* block_log_mass,
    int heads,int kv_heads,int width,int count,int parts) {
    sparse_attention_impl<W,Snapshots,SG>(query,relative,blocks,selected,positions,partial,stats,
        block_log_mass,heads,kv_heads,width,count,parts);
}

// Sixteen GQA heads share the decoded sixteen-token tile. QK and PV use
// FP16 tensor-core operands with FP32 accumulation, as in FP16 attention.
template<int W,bool Snapshots> __global__ void sparse_attention_mma(const float* query,const half* relative,
    const TqBlockView* blocks,const SelectedBlock* selected,const int64_t* positions,
    float* partial,float* stats,float* block_log_mass,int heads,int kv_heads,int width,int count,int parts) {
#if __CUDA_ARCH__ >= 700
    namespace wmma=nvcuda::wmma;
    constexpr int N=W/32,rings=W/256,P=W+8;
    const int tid=threadIdx.x,lane=tid&31,warp=tid/32,part=blockIdx.x,token=blockIdx.z;
    const int gqa=heads/kv_heads,groups=(gqa+15)/16,kh=blockIdx.y/groups,base=(blockIdx.y%groups)*16;
    extern __shared__ __align__(32) unsigned char storage[];
    // Eight half elements skew successive rows across shared-memory banks.
    // WMMA still receives aligned bases and legal leading dimensions.
    auto* qh=reinterpret_cast<half*>(storage);auto* key=qh+16*P;auto* value=key+16*P;
    auto* temporary=reinterpret_cast<float*>(value+16*P);
    auto* scores=temporary+16*P;auto* probability=reinterpret_cast<half*>(scores+256);
    __shared__ TqRowView kr[16*rings],vr[16*rings];
    __shared__ SelectedBlock chosen[4];
    for(int at=tid;at<16*W;at+=128) {
        const int h=at/W,d=at%W,head=kh*gqa+base+h;
        qh[h*P+d]=__float2half_rn(base+h<gqa?query[(int64_t(token)*heads+head)*W+d]:0);
    }
    float accum[4][N],maximum[4],mass[4];
    for(int r=0;r<4;++r){maximum[r]=-CUDART_INF_F;mass[r]=0;for(int i=0;i<N;++i)accum[r][i]=0;}
    const int tiles=(count+3)/4,begin=4*(part*tiles/parts),end=min(count,4*((part+1)*tiles/parts));
    const auto boundary=positions[token];
    for(int first=begin;first<end;first+=4) {
        if(tid<4)chosen[tid]=first+tid<end?selected[int64_t(token)*count+first+tid]:SelectedBlock{};
        __syncthreads();
        if(tid<16*rings) {
            const int b=tid/(4*rings),r=tid%(4*rings),slot=first+b;
            const auto selection=chosen[b];const bool valid=selection.id>=0 && selection.token_mask;
            const auto view=valid?blocks[Snapshots?int64_t(token)*count+slot:selection.id]:TqBlockView{};
            kr[tid]=valid?tq_row(view.key,view.key_bits,kh*4*rings+r,kv_heads*4*rings,relative):TqRowView{};
            vr[tid]=valid?tq_row(view.value,view.value_bits,kh*4*rings+r,kv_heads*4*rings,relative):TqRowView{};
        }
        __syncthreads();
        for(int at=tid;at<4*W;at+=128) {
            const int b=at/W,d=at%W;const bool valid=chosen[b].id>=0 && chosen[b].token_mask;
            float k[4],v[4];
            for(int m=0;m<4;++m) {
                const int r=(b*4+m)*rings+d/256;
                k[m]=valid?tq_value(kr[r],d&255):0;v[m]=valid?tq_value(vr[r],d&255):0;
            }
            for(int t=0;t<4;++t) {
                float a=0,c=0;
                for(int m=0;m<4;++m){const float sign=(__popc(unsigned(t&m))&1)?-.5f:.5f;a+=sign*k[m];c+=sign*v[m];}
                key[(b*4+t)*P+d]=__float2half_rn(a);value[(b*4+t)*P+d]=__float2half_rn(c);
            }
        }
        __syncthreads();
        if(!warp) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;wmma::fill_fragment(c,0.f);
            for(int d=0;d<W;d+=16){wmma::load_matrix_sync(a,qh+d,P);wmma::load_matrix_sync(b,key+d,P);wmma::mma_sync(c,a,b,c);}
            wmma::store_matrix_sync(scores,c,16,wmma::mem_row_major);
        }
        __syncthreads();
        float factors[4];
        for(int r=0;r<4;++r) {
            const int local=warp+4*r,head=kh*gqa+base+local;
            const bool active=base+local<gqa;
            float score=-CUDART_INF_F;
            if(lane<16 && active) {
                const auto selection=chosen[lane/4];
                if(selection.id>=0 && (selection.token_mask&(1u<<(lane&3))) && int64_t(selection.id)*4+(lane&3)<=boundary)
                    score=scores[local*16+lane]*rsqrtf(float(width));
            }
            const float top=fmaxf(maximum[r],warp_max(score));
            factors[r]=isfinite(maximum[r])?expf(maximum[r]-top):0;
            const float weight=isfinite(score)?expf(score-top):0;
            mass[r]=mass[r]*factors[r]+warp_sum(weight);maximum[r]=top;
            if(lane<16)probability[local*16+lane]=__float2half_rn(weight);
            const float sum2=weight+__shfl_xor_sync(0xffffffff,weight,1);
            const float sum4=sum2+__shfl_xor_sync(0xffffffff,sum2,2);
            if(active && lane<16 && !(lane&3) && first+lane/4<end)
                block_log_mass[(int64_t(token)*heads+head)*count+first+lane/4]=sum4>0?top+logf(sum4):-CUDART_INF_F;
        }
        __syncthreads();
        for(int column=warp;column<W/16;column+=4) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
            wmma::fragment<wmma::accumulator,16,16,16,float> c;wmma::fill_fragment(c,0.f);
            wmma::load_matrix_sync(a,probability,16);wmma::load_matrix_sync(b,value+column*16,P);wmma::mma_sync(c,a,b,c);
            wmma::store_matrix_sync(temporary+column*16,c,P,wmma::mem_row_major);
        }
        __syncthreads();
        for(int r=0;r<4;++r)for(int i=0;i<N;++i)
            accum[r][i]=fmaf(accum[r][i],factors[r],temporary[(warp+4*r)*P+lane+i*32]);
        __syncthreads();
    }
    for(int r=0;r<4;++r) {
        const int local=base+warp+4*r,head=kh*gqa+local;
        if(local<gqa) {
            const auto at=(int64_t(token)*heads+head)*parts+part;
            for(int i=0;i<N;++i)partial[at*W+lane+i*32]=accum[r][i];
            if(!lane){stats[at*2]=maximum[r];stats[at*2+1]=mass[r];}
        }
    }
#endif
}

#include "qsa_tq_mma.cuh"

template<int W,bool Snapshots,bool Narrow=false,int HeadTiles=1>
bool mma_attention_available(int* resident=nullptr,int* sms=nullptr) {
    static thread_local int initialized=-1;
    static thread_local bool available=false;
    static thread_local int resident_blocks=0,multiprocessors=0;
    int device=0;cudaGetDevice(&device);
    if(initialized!=device) {
        resident_blocks=0;
        int major=0,limit=0;cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
        cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device);
        cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,device);
        constexpr int dynamic=Narrow?(W+8)*32+4352*HeadTiles:(W+8)*160+1536;
        // m16n8k16 requires SM80; the previous WMMA path requires SM70.
        const void* function=Narrow?reinterpret_cast<const void*>(sparse_attention_mma8<W,Snapshots,false,HeadTiles>):
            reinterpret_cast<const void*>(sparse_attention_mma<W,Snapshots>);
        cudaFuncAttributes attributes{};
        available=major>=(Narrow?8:7) && cudaFuncGetAttributes(&attributes,function)==cudaSuccess &&
            dynamic+attributes.sharedSizeBytes<=std::size_t(limit);
        // Two head tiles retain more accumulators. Keep the smaller tile
        // when the compiled kernel spills them into thread-local memory.
        if constexpr(Narrow && HeadTiles>1)available=available && !attributes.localSizeBytes;
        if(available) {
            available=cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic)==cudaSuccess;
            if constexpr(Narrow)if(available) {
                int blocks=0;
                // Independent CTAs cover compressed-KV decode/load latency.
                available=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,256,dynamic)==cudaSuccess && blocks>=2;
                if(available)resident_blocks=blocks;
            }
        }
        initialized=device;
    }
    if(resident)*resident=available?resident_blocks:0;
    if(sms)*sms=multiprocessors;
    return available;
}

template<int W,bool Snapshots>
int narrow_attention_head_tiles(int gqa,int* resident=nullptr,int* sms=nullptr) {
    // A second tile is useful only when a KV head serves more than eight
    // query heads. Resource checks use the actual compiled kernel/device.
    if(gqa>8 && mma_attention_available<W,Snapshots,true,2>(resident,sms))return 2;
    return mma_attention_available<W,Snapshots,true>(resident,sms)?1:0;
}

template<int W,bool Snapshots> int attention_partitions(int queries,int heads,int kv_heads,int count,bool fp16,bool fp16_math) {
    const int legacy=std::min(queries>8?4:64,count);
    if(queries>8 || !(fp16 || fp16_math) || heads/kv_heads<4)return legacy;
    int resident=0,sms=0;
    const int tiles_per_head=narrow_attention_head_tiles<W,Snapshots>(heads/kv_heads,&resident,&sms);
    if(!tiles_per_head || resident<=0 || sms<=0)return legacy;
    const int tile_heads=8*tiles_per_head;
    const auto groups=int64_t(queries)*kv_heads*((heads/kv_heads+tile_heads-1)/tile_heads);
    const auto capacity=int64_t(sms)*resident;
    const int tiles=int((int64_t(count)+3)/4);
    const int limit=int(std::min({int64_t(128),int64_t(tiles),std::max<int64_t>(1,capacity/groups)}));
    // Minimize serial tiles without introducing another occupancy wave.
    // Then use the fewest partitions that preserve that tile count,
    // reducing both partial writes and the final merge. This depends only
    // on available kernel resources and the requested attention workload.
    const int iterations=(tiles+limit-1)/limit;
    const int needed=(tiles+iterations-1)/iterations;
    return needed;

}

template<int W,bool Half> __global__ void merge_output(const float* partial,const float* stats,const float* signs,
    void* output,float* log_norm,int width,int parts,const void* gate,bool gate_half,int heads,int64_t gt,int64_t gh) {
    const int d=threadIdx.x,row=blockIdx.x;
    __shared__ float work[W],factors[128],denominator;
    if (d<32) {
        float maximum=-CUDART_INF_F;
        for(int p=d;p<parts;p+=32)maximum=fmaxf(maximum,stats[(int64_t(row)*parts+p)*2]);
        maximum=warp_max(maximum);
        float local_mass=0;
        for(int p=d;p<parts;p+=32) {
            const auto at=(int64_t(row)*parts+p)*2;
            const float factor=isfinite(stats[at])?expf(stats[at]-maximum):0;
            factors[p]=factor;local_mass+=factor*stats[at+1];
        }
        const float mass=warp_sum(local_mass);
        if(!d){denominator=mass;log_norm[row]=mass>0?maximum+logf(mass):-CUDART_INF_F;}
    }
    __syncthreads();
    float sum=0;
    for (int p=0;p<parts;++p)sum+=factors[p]*partial[(int64_t(row)*parts+p)*W+d];
    float x=denominator>0?sum/denominator:0;
    for (int stride=1;stride<W;stride*=2) {
        if(stride<32){const auto other=__shfl_xor_sync(0xffffffff,x,stride);x=(d&stride)?other-x:x+other;}
        else {work[d]=x;__syncthreads();const auto other=work[d^stride];x=(d&stride)?other-x:x+other;__syncthreads();}
    }
    if (d<width) {
        float value=x*signs[d]*rsqrtf(float(W));
        if(gate)value*=1.f/(1.f+expf(-read_input(gate,gate_half,int64_t(row/heads)*gt+int64_t(row%heads)*gh+d)));
        if constexpr(Half)reinterpret_cast<half*>(output)[int64_t(row)*width+d]=__float2half_rn(value);
        else reinterpret_cast<float*>(output)[int64_t(row)*width+d]=value;
    }
}

__global__ void observe_mass(const SelectedBlock* selected,const int64_t* positions,const float* block_log_mass,
    const float* log_norm,float* hits,float* mass,int heads,int count) {
    // Each warp reads adjacent blocks for one query head. Four warps share
    // the head reduction, keeping those reads coalesced while shortening
    // the serial head loop and exposing more independent block groups.
    constexpr int head_warps=4;
    const int lane=threadIdx.x,warp=threadIdx.y;
    const int slot=blockIdx.x*32+lane,token=blockIdx.y;
    SelectedBlock block{};
    bool active=slot<count;
    if(active) {
        block=selected[int64_t(token)*count+slot];
        const auto visible=positions[token]-int64_t(block.id)*4+1;
        const auto causal_mask=visible<=0?0u:visible>=4?15u:(1u<<int(visible))-1;
        active=block.id>=0 && (block.token_mask&causal_mask);
    }
    float value=0;
    for (int h=warp;active && h<heads;h+=head_warps) {
        const auto row=int64_t(token)*heads+h;
        const float a=block_log_mass[row*count+slot],b=log_norm[row];
        if (isfinite(a) && isfinite(b))value+=expf(a-b);
    }
    __shared__ float partial[head_warps][32];
    partial[warp][lane]=value;
    __syncthreads();
    if(!warp && active) {
        value=partial[0][lane]+partial[1][lane]+partial[2][lane]+partial[3][lane];
        atomicAdd(hits+block.id,1.f);atomicAdd(mass+block.id,value/heads);
    }
}

template<int W,bool Snapshots> int launch_attention(const void* query,const float* ks,const float* vs,
    const uint16_t* relative,const TqBlockView* blocks,const SelectedBlock* selected,const int64_t* positions,
    void* output,float* rotated,float* partial,float* stats,float* block_log_mass,float* log_norm,
    float* hits,float* mass,int queries,int heads,int kv_heads,int width,int count,int parts,
    int64_t token_stride,int64_t head_stride,cudaStream_t stream,const void* gate,bool gate_half,bool output_half,int64_t gt,int64_t gh,bool query_half,bool allow_fp16_math,bool query_pre_rotated,const uint16_t* staged_keys=nullptr,const uint16_t* staged_values=nullptr) {
    if(!query_pre_rotated)rotate_query<W><<<queries*heads,W,0,stream>>>(query,ks,rotated,heads,width,token_stride,head_stride,query_half);
    // Wider GQA groups reuse a decoded microblock across more query heads.
    // This depends on the attention geometry, independent of the GPU model.
#define ATTEND(SG) sparse_attention<W,Snapshots,SG><<<dim3(parts,kv_heads*((heads/kv_heads+SG-1)/SG),queries),min(heads/kv_heads,SG)*32,0,stream>>>(rotated, \
        reinterpret_cast<const half*>(relative),blocks,selected,positions,partial,stats,block_log_mass,heads,kv_heads,width,count,parts)
    // At least half of either 8/16-head tile must carry useful work. FP32 callers
    // opt into inference math explicitly; normalization often emits FP32.
    const int head_tiles=(query_half || allow_fp16_math) && heads/kv_heads>=4?
        narrow_attention_head_tiles<W,Snapshots>(heads/kv_heads):0;
    const bool use_narrow_mma=head_tiles!=0;
    const bool use_mma=use_narrow_mma || ((query_half || allow_fp16_math) && heads/kv_heads>=8 && mma_attention_available<W,Snapshots>());
    static thread_local unsigned reported=0;const unsigned category=queries==1?1:queries<=8?2:4;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1' && !(reported&category)) {
        reported|=category;
        std::fprintf(stderr,"native_qsa_tq_attention actual_match=1 tensor_core=%d query_fp16=%d queries=%d heads=%d kv_heads=%d padded=%d parts=%d\n",
            (use_narrow_mma?8*head_tiles:use_mma?16:0),int(query_half),queries,heads,kv_heads,W,parts);
    }
#define NARROW_ATTEND(TILES,STAGED) sparse_attention_mma8<W,Snapshots,STAGED,TILES><<< \
        dim3(parts,kv_heads*((heads/kv_heads+8*TILES-1)/(8*TILES)),queries),256,(W+8)*32+4352*TILES,stream>>>(rotated, \
        reinterpret_cast<const half*>(relative),blocks,selected,positions,partial,stats,block_log_mass,heads,kv_heads,width,count,parts, \
        reinterpret_cast<const half*>(staged_keys),reinterpret_cast<const half*>(staged_values))
    if(use_narrow_mma) {
        const bool staged=!Snapshots && staged_keys && staged_values;
        if(head_tiles==2){if(staged){NARROW_ATTEND(2,true);}else {NARROW_ATTEND(2,false);}}
        else {if(staged){NARROW_ATTEND(1,true);}else {NARROW_ATTEND(1,false);}}
    }
    else if(use_mma)
        sparse_attention_mma<W,Snapshots><<<dim3(parts,kv_heads*((heads/kv_heads+15)/16),queries),128,(W+8)*160+1536,stream>>>(rotated,
            reinterpret_cast<const half*>(relative),blocks,selected,positions,partial,stats,block_log_mass,heads,kv_heads,width,count,parts);
    else if(heads/kv_heads>4){ATTEND(8);}else {ATTEND(4);}
#undef NARROW_ATTEND
#undef ATTEND
    if(output_half)merge_output<W,true><<<queries*heads,W,0,stream>>>(partial,stats,vs,output,log_norm,width,parts,gate,gate_half,heads,gt,gh);
    else merge_output<W,false><<<queries*heads,W,0,stream>>>(partial,stats,vs,output,log_norm,width,parts,gate,gate_half,heads,gt,gh);
    if (hits && mass)observe_mass<<<dim3((count+31)/32,queries),dim3(32,4),0,stream>>>(selected,positions,block_log_mass,log_norm,hits,mass,heads,count);
    return int(cudaGetLastError());
}
} // namespace

int tq_attention_partitions(int queries,int heads,int kv_heads,int padded,int count,
    bool fp16,bool fp16_math,bool snapshots) {
    if(queries<=0 || heads<=0 || kv_heads<=0 || heads%kv_heads || count<=0)return 0;
#define PLAN(W) return snapshots?attention_partitions<W,true>(queries,heads,kv_heads,count,fp16,fp16_math):attention_partitions<W,false>(queries,heads,kv_heads,count,fp16,fp16_math)
    switch(padded){case 256:PLAN(256);case 512:PLAN(512);case 1024:PLAN(1024);default:return 0;}
#undef PLAN
}

int tq_publish_blocks(uint64_t* table,const int32_t* ids,const uint64_t* records,uint64_t* previous,
    int count,cudaStream_t stream) {
    if(!table || !ids || !records || count<=0)return int(cudaErrorInvalidValue);
    publish_blocks<<<(count+127)/128,128,0,stream>>>(table,ids,records,previous,count);
    return int(cudaGetLastError());
}
int tq_snapshot_blocks(const uint64_t* table,const SelectedBlock* selected,TqBlockView* output,
    int queries,int count,cudaStream_t stream) {
    if(!table || !selected || !output || queries<=0 || count<=0)return int(cudaErrorInvalidValue);
    snapshot_blocks<<<(queries*count+127)/128,128,0,stream>>>(table,selected,output,queries*count);
    return int(cudaGetLastError());
}

int tq_prepare_modes(const void* keys,const void* values,bool fp16,const float* ks,const float* vs,
    float* km,float* vm,uint16_t* pending_k,uint16_t* pending_v,
    int tokens,int heads,int width,int padded,cudaStream_t stream,float* rotated_k,float* rotated_v) {
    if(!keys || !values || !ks || !vs || !km || !vm || tokens<=0 || heads<=0 || width<=0 || width>padded)
        return int(cudaErrorInvalidValue);
#define PREPARE(W,F) prepare_modes<W,F><<<dim3(heads,(tokens+3)/4,2),W,0,stream>>>(keys,values,ks,vs,km,vm,reinterpret_cast<half*>(pending_k),reinterpret_cast<half*>(pending_v),tokens,heads,width,rotated_k,rotated_v)
    if(fp16) {switch(padded){case 256:PREPARE(256,true);break;case 512:PREPARE(512,true);break;case 1024:PREPARE(1024,true);break;default:return int(cudaErrorInvalidValue);}}
    else {switch(padded){case 256:PREPARE(256,false);break;case 512:PREPARE(512,false);break;case 1024:PREPARE(1024,false);break;default:return int(cudaErrorInvalidValue);}}
#undef PREPARE
    return int(cudaGetLastError());
}

int tq_prepare_affine8(const void* keys,const void* values,bool fp16,const float* ks,const float* vs,
    uint8_t* ki,uint8_t* vi,int tokens,int heads,int width,int padded,cudaStream_t stream) {
    if(!keys || !values || !ks || !vs || !ki || !vi || tokens<=0 || heads<=0 || width<=0 || width>padded)
        return int(cudaErrorInvalidValue);
#define AFFINE(W,F) prepare_affine8<W,F><<<dim3(heads,(tokens+3)/4,2),W,0,stream>>>(keys,values,ks,vs,ki,vi,tokens,heads,width)
    if(fp16){switch(padded){case 256:AFFINE(256,true);break;case 512:AFFINE(512,true);break;case 1024:AFFINE(1024,true);break;default:return int(cudaErrorInvalidValue);}}
    else {switch(padded){case 256:AFFINE(256,false);break;case 512:AFFINE(512,false);break;case 1024:AFFINE(1024,false);break;default:return int(cudaErrorInvalidValue);}}
#undef AFFINE
    return int(cudaGetLastError());
}

int tq_decode_modes(const TqBlockView* blocks,float* keys,float* values,const uint16_t* relative,
    int count,int heads,int padded,cudaStream_t stream) {
    if(!blocks || !keys || !values || !relative || count<=0 || heads<=0 ||
       padded<256 || padded%256)return int(cudaErrorInvalidValue);
    const auto rows=int64_t(heads)*4*(padded/256);
    if(rows>std::numeric_limits<int>::max() || count>std::numeric_limits<int>::max()/rows)
        return int(cudaErrorInvalidValue);
    decode_modes<<<dim3(unsigned(count*rows),2),256,0,stream>>>(blocks,keys,values,
        reinterpret_cast<const half*>(relative),int(rows));return int(cudaGetLastError());
}

int indexer_affine16_encode(const float* input,int16_t* output,uint16_t* scales,int16_t* zero,
    int rows,int width,cudaStream_t stream) {
    if (!input || !output || !scales || !zero || rows<=0 || width<=0)return int(cudaErrorInvalidValue);
    affine_encode<false><<<rows,32,0,stream>>>(input,output,scales,zero,width);return int(cudaGetLastError());
}
int indexer_affine16_append(const float* input,int16_t* output,uint16_t* scales,int16_t* zero,
    const int64_t* position,int width,cudaStream_t stream) {
    if(!input || !output || !scales || !zero || !position || width<=0)return int(cudaErrorInvalidValue);
    affine_encode<false><<<1,32,0,stream>>>(input,output,scales,zero,width,position);return int(cudaGetLastError());
}
int indexer_affine16_append_batch(const float* input,int16_t* output,uint16_t* scales,int16_t* zero,
    const int64_t* position,int rows,int width,int capacity,cudaStream_t stream) {
    if(!input || !output || !scales || !zero || !position || rows<=0 || width<=0 || capacity<=0)return int(cudaErrorInvalidValue);
    affine_encode<false><<<rows,32,0,stream>>>(input,output,scales,zero,width,position,capacity);return int(cudaGetLastError());
}
int indexer_affine16_scores(const float* query,const int16_t* keys,const uint16_t* scales,
    const int16_t* zero,const int64_t* positions,float* output,int queries,int heads,int width,int blocks,cudaStream_t stream) {
    if (!query || !keys || !scales || !zero || !positions || !output || queries<=0 || heads<=0 || width<=0 || blocks<=0)
        return int(cudaErrorInvalidValue);
    const int paired=launch_affine_scores_pairs(query,keys,scales,zero,positions,output,queries,heads,width,blocks,stream);
    if(paired!=int(cudaErrorNotSupported))return paired;
#define SCORES(QT) affine_scores<QT><<<dim3((blocks+(QT==1?31:15))/(QT==1?32:16),(queries+QT-1)/QT),affine_score_threads,0,stream>>>(query,keys,scales,zero,positions,output,queries,heads,width,blocks)
    const int tensor=launch_affine_scores_tensor(query,keys,scales,zero,positions,output,queries,heads,width,blocks,stream);
    if(tensor!=int(cudaErrorNotSupported))return tensor;
    const int tiled=launch_affine_scores_tiled(query,keys,scales,zero,positions,output,queries,heads,width,blocks,stream);
    if(tiled!=int(cudaErrorNotSupported))return tiled;
    if(queries==1){SCORES(1);}else if(queries==2){SCORES(2);}else {SCORES(4);}
#undef SCORES
    return int(cudaGetLastError());
}
int indexer_affine16_pack(const float* input,int16_t* output,int rows,int width,cudaStream_t stream) {
    if(!input || !output || rows<=0 || width<=0)return int(cudaErrorInvalidValue);
    affine_encode<true><<<rows,32,0,stream>>>(input,output,nullptr,nullptr,width);return int(cudaGetLastError());
}
int indexer_affine16_unpack(const int16_t* records,float* output,int rows,int width,cudaStream_t stream) {
    if(!records || !output || rows<=0 || width<=0)return int(cudaErrorInvalidValue);
    const auto n=int64_t(rows)*width;
    affine_record_decode<<<unsigned((n+255)/256),256,0,stream>>>(records,output,rows,width);return int(cudaGetLastError());
}
int indexer_affine16_record_scores(const float* query,const int16_t* records,const int64_t* positions,float* output,
        int queries,int heads,int width,int blocks,int pool,cudaStream_t stream) {
    if(!query || !records || !output || queries<=0 || heads<=0 || width<=0 || blocks<=0 || pool<=0)return int(cudaErrorInvalidValue);
    const int paired=launch_affine_scores_pairs<true>(query,records,nullptr,nullptr,positions,output,queries,heads,width,blocks,stream,pool);
    if(paired!=int(cudaErrorNotSupported))return paired;
    const int tensor=launch_affine_scores_tensor<true>(query,records,nullptr,nullptr,positions,output,queries,heads,width,blocks,stream,pool);
    if(tensor!=int(cudaErrorNotSupported))return tensor;
    const int tiled=launch_affine_scores_tiled<true>(query,records,nullptr,nullptr,positions,output,queries,heads,width,blocks,stream,pool);
    if(tiled!=int(cudaErrorNotSupported))return tiled;
#define RECORD_SCORES(QT) affine_scores<QT,true><<<dim3((blocks+(QT==1?31:15))/(QT==1?32:16),(queries+QT-1)/QT),affine_score_threads,0,stream>>>(query,records,nullptr,nullptr,positions,output,queries,heads,width,blocks,pool)
    if(queries==1){RECORD_SCORES(1);}else if(queries==2){RECORD_SCORES(2);}else {RECORD_SCORES(4);}
#undef RECORD_SCORES
    return int(cudaGetLastError());
}
int tq8_fit_online(const float* rows,const uint16_t* codebook,const uint8_t* candidates,
    int bins,float bin_min,float bin_step,float book_rms,float relative,uint16_t* anchors,
    uint8_t* payload,uint8_t* trace,float* sse,int count,cudaStream_t stream) {
    if(!rows || !codebook || !candidates || bins<2 || !(bin_step>0) || !(book_rms>0) || !(relative>0) ||
       !anchors || !payload || !trace || !sse || count<=0)return int(cudaErrorInvalidValue);
    online8<<<count,256,0,stream>>>(rows,reinterpret_cast<const half*>(codebook),candidates,bins,bin_min,bin_step,
        book_rms,relative,reinterpret_cast<half*>(anchors),payload,trace,sse);
    return int(cudaGetLastError());
}
int tq_restore_modes(const float* keys,const float* values,const float* ks,const float* vs,
    float* out_k,float* out_v,int blocks,int heads,int width,int padded,cudaStream_t stream) {
    if(!keys || !values || !ks || !vs || !out_k || !out_v || blocks<=0 || heads<=0 || width<=0 || width>padded)
        return int(cudaErrorInvalidValue);
#define RESTORE(W) restore_modes<W><<<dim3(heads,blocks,2),W,0,stream>>>(keys,values,ks,vs,out_k,out_v,heads,width)
    switch(padded){case 256:RESTORE(256);break;case 512:RESTORE(512);break;case 1024:RESTORE(1024);break;default:return int(cudaErrorInvalidValue);}
#undef RESTORE
    return int(cudaGetLastError());
}
int tq_group_selected(const int32_t* tokens,SelectedBlock* blocks,int queries,int columns,cudaStream_t stream) {
    if(!tokens || !blocks || queries<=0 || columns<=0)return int(cudaErrorInvalidValue);
    const int count=(columns+3)/4;
    if(queries>std::numeric_limits<int>::max()/count)return int(cudaErrorInvalidValue);
    const int total=queries*count;
    group_selected<<<unsigned((std::size_t(total)+255)/256),256,0,stream>>>(tokens,blocks,columns,count,total);
    return int(cudaGetLastError());
}
int tq_stage_append(const void* keys,const void* values,const void* index,bool kh,bool vh,bool ih,
    const float* old_k,const float* old_v,const float* old_i,float* k,float* v,float* raw,
    int tokens,int prior,int heads,int width,int iw,int64_t kt,int64_t ks,int64_t vt,int64_t vs,int64_t it,cudaStream_t stream) {
    if(!keys || !values || !index || !k || !v || !raw || tokens<=0 || prior<0 || prior>3 || heads<=0 || width<=0 || iw<=0 ||
       (prior && (!old_k || !old_v || !old_i)))return int(cudaErrorInvalidValue);
    const auto total=int64_t(tokens+prior)*(int64_t(2)*heads*width+iw);
    stage_append<<<unsigned((total+255)/256),256,0,stream>>>(keys,values,index,kh,vh,ih,old_k,old_v,old_i,k,v,raw,tokens,prior,heads,width,iw,kt,ks,vt,vs,it);
    return int(cudaGetLastError());
}
int tq_commit_tail(const float* k,const float* v,const float* raw,float* out_k,float* out_v,float* out_i,
    int tokens,int remainder,int heads,int width,int iw,cudaStream_t stream) {
    if(!k || !v || !raw || !out_k || !out_v || !out_i || tokens<=0 || remainder<0 || remainder>3 || remainder>tokens || heads<=0 || width<=0 || iw<=0)
        return int(cudaErrorInvalidValue);
    if(!remainder)return 0;
    const auto total=int64_t(remainder)*(int64_t(2)*heads*width+iw);
    commit_tail<<<unsigned((total+255)/256),256,0,stream>>>(k,v,raw,out_k,out_v,out_i,tokens,remainder,heads*width,iw);
    return int(cudaGetLastError());
}
int tq_pack_archive(const uint8_t* k,const uint8_t* v,const float* raw,uint8_t* output,int blocks,int image_bytes,int iw,cudaStream_t stream) {
    if(!k || !v || !raw || !output || blocks<=0 || image_bytes<=0 || image_bytes%2 || iw<=0)return int(cudaErrorInvalidValue);
    const auto bytes=int64_t(blocks)*image_bytes,values=int64_t(blocks)*4*iw,total=2*bytes+values;
    pack_archive<<<unsigned((total+255)/256),256,0,stream>>>(k,v,raw,output,bytes,values);
    return int(cudaGetLastError());
}
int tq_archive_indexer(const int16_t* codes,const uint16_t* scales,const int16_t* zero,
        uint8_t* archive,std::size_t offset,int blocks,int width,int64_t start,const int64_t* position,cudaStream_t stream) {
    if(!codes || !scales || !zero || !archive || (offset&1) || blocks<=0 || width<=0 ||
            (!position && (start<0 || start>INT64_MAX-blocks)) || (position && blocks!=1))return int(cudaErrorInvalidValue);
    archive_indexer<<<blocks,128,0,stream>>>(codes,scales,zero,reinterpret_cast<uint16_t*>(archive+offset),width,start,position);
    return int(cudaGetLastError());
}
int tq_copy_pending_tail(PendingTail source,PendingTail destination,int heads,int width,int padded,int iw,cudaStream_t stream) {
    if(!source.key || !source.value || !source.index || !source.rotated_key || !source.rotated_value ||
       !destination.key || !destination.value || !destination.index || !destination.rotated_key || !destination.rotated_value ||
       heads<=0 || width<=0 || padded<width || iw<=0)return int(cudaErrorInvalidValue);
    const auto keys=4LL*heads*width,rotated=4LL*heads*padded,index=4LL*iw;
    const auto total=2*keys+2*rotated+index;
    if((total+255)/256>INT_MAX)return int(cudaErrorInvalidValue);
    copy_pending_tail<<<unsigned((total+255)/256),256,0,stream>>>(source,destination,keys,rotated,index);
    return int(cudaGetLastError());
}
int tq_stage_graph_batch(const void* keys,const void* values,const void* index,bool kh,bool vh,bool ih,
    PendingTail previous,const int64_t* position,float* k,float* v,float* raw,int tokens,int staged,int heads,int width,int iw,
    int64_t kt,int64_t ks,int64_t vt,int64_t vs,int64_t it,cudaStream_t stream) {
    if(!keys || !values || !index || !position || !previous.key || !previous.value || !previous.index || !k || !v || !raw ||
       tokens<1 || tokens>8 || staged<(tokens+3+3)/4*4 || heads<=0 || width<=0 || iw<=0 || kt<=0 || ks<=0 || vt<=0 || vs<=0 || it<=0)
        return int(cudaErrorInvalidValue);
    const auto total=int64_t(staged)*(2LL*heads*width+iw);
    if((total+255)/256>INT_MAX)return int(cudaErrorInvalidValue);
    stage_append<<<unsigned((total+255)/256),256,0,stream>>>(keys,values,index,kh,vh,ih,previous.key,previous.value,previous.index,k,v,raw,
        tokens,0,heads,width,iw,kt,ks,vt,vs,it,position,staged);return int(cudaGetLastError());
}
int tq_publish_graph_batch(const PendingBatchPublication& b,cudaStream_t stream) {
    if(!b.position || !b.keys || !b.values || !b.fp16_keys || !b.fp16_values || !b.raw_keys || !b.raw_values || !b.raw_index ||
       !b.rotated_keys || !b.rotated_values || !b.index_codes || !b.index_scales || !b.index_zero || !b.destinations || !b.views || !b.archive ||
       !b.tail.key || !b.tail.value || !b.tail.index || !b.tail.rotated_key || !b.tail.rotated_value || b.tokens<1 || b.tokens>8 ||
       b.heads<=0 || b.width<=0 || b.padded_width<b.width || b.index_width<=0 || b.image_bytes<=0 || b.destination_slots<=0)
        return int(cudaErrorInvalidValue);
    const auto record=2LL*b.image_bytes+8LL*b.index_width+(b.archive_index?2LL*(b.index_width+2):0);
    const auto total=std::max<int64_t>({int64_t((b.tokens+3)/4)*record,3LL*(2LL*b.heads*b.width+b.index_width),3LL*b.heads*b.padded_width});
    if((total+255)/256>INT_MAX)return int(cudaErrorInvalidValue);
    publish_graph_batch<<<unsigned((total+255)/256),256,0,stream>>>(b);return int(cudaGetLastError());
}
int tq_append_pending(const void* key,const void* value,const void* index,bool kh,bool vh,bool ih,
    const int64_t* position,float* tail_k,float* tail_v,float* tail_i,float* rotated_k,float* rotated_v,const float* ks,const float* vs,
    uint16_t* pk,uint16_t* pv,TqBlockView* views,int heads,int width,int padded,int iw,int64_t khs,int64_t vhs,cudaStream_t stream,float* km,float* vm,const PendingQuery* query) {
    if(!key || !value || !index || !position || !tail_k || !tail_v || !tail_i || !rotated_k || !rotated_v || !ks || !vs || !pk || !pv || !views ||
       heads<=0 || width<=0 || width>padded || iw<=0 || khs<=0 || vhs<=0)return int(cudaErrorInvalidValue);
    if(query && (!query->input || !query->output || query->heads<=0 || query->head_stride<width ||
        heads>INT_MAX/2 || query->heads>INT_MAX-2*heads))return int(cudaErrorInvalidValue);
#define PENDING(W) do { \
    if(query)append_pending<W,true><<<2*heads+query->heads,W,0,stream>>>(key,value,index,kh,vh,ih,position,tail_k,tail_v,tail_i,rotated_k,rotated_v,ks,vs,reinterpret_cast<half*>(pk),reinterpret_cast<half*>(pv),views,heads,width,iw,khs,vhs,km,vm,query->input,query->output,query->heads,query->head_stride,query->fp16); \
    else append_pending<W,false><<<dim3(heads,2),W,0,stream>>>(key,value,index,kh,vh,ih,position,tail_k,tail_v,tail_i,rotated_k,rotated_v,ks,vs,reinterpret_cast<half*>(pk),reinterpret_cast<half*>(pv),views,heads,width,iw,khs,vhs,km,vm,nullptr,nullptr,0,0,false); \
} while(false)
    switch(padded){case 256:PENDING(256);break;case 512:PENDING(512);break;case 1024:PENDING(1024);break;default:return int(cudaErrorInvalidValue);}
#undef PENDING
    return int(cudaGetLastError());
}
int tq_finalize_pending(const float* km,const float* vm,const float* raw,const TqBlockView* destination,
    TqBlockView* views,const int64_t* position,uint8_t* archive,int rows,int iw,int slots,cudaStream_t stream) {
    if(!km || !vm || !raw || !destination || !views || !position || !archive || rows<=0 || iw<=0 || slots<=0)return int(cudaErrorInvalidValue);
    finalize_pending<<<dim3(rows,2),256,0,stream>>>(km,vm,raw,destination,views,position,archive,rows,iw,slots);
    return int(cudaGetLastError());
}
int tq_initialize_destinations(uint8_t* payload,TqBlockView* output,int bytes,int slots,cudaStream_t stream) {
    if(!payload || !output || bytes<=0 || slots<=0)return int(cudaErrorInvalidValue);
    initialize_destinations<<<(slots+127)/128,128,0,stream>>>(payload,output,bytes,slots);return int(cudaGetLastError());
}
int tq_initialize_fp16_destinations(uint8_t* payload,TqBlockView* output,int bytes,int slots,cudaStream_t stream) {
    if(!payload || !output || bytes<=0 || slots<=0)return int(cudaErrorInvalidValue);
    initialize_destinations<<<(slots+127)/128,128,0,stream>>>(payload,output,bytes,slots,0);return int(cudaGetLastError());
}
int tq_publish_contiguous_views(TqBlockView base,TqBlockView* output,int bytes,int count,cudaStream_t stream) {
    if(count<0 || bytes<=0 || !base.key || !base.value || !output)return int(cudaErrorInvalidValue);
    if(!count)return int(cudaSuccess);
    publish_contiguous_views<<<unsigned((int64_t(count)+127)/128),128,0,stream>>>(base,output,bytes,count);
    return int(cudaGetLastError());
}
int tq_sparse_attention_staged(const uint16_t* staged_keys,const uint16_t* staged_values,const void* query,const float* ks,const float* vs,const uint16_t* relative,
    const TqBlockView* blocks,const SelectedBlock* selected,const int64_t* positions,void* output,
    float* rotated,float* partial,float* stats,float* block_log_mass,float* log_norm,float* hits,float* mass,
    int queries,int heads,int kv_heads,int width,int padded,int count,int parts,int64_t ts,int64_t hs,cudaStream_t stream,bool snapshots,
    const void* gate,bool gate_half,bool output_half,int64_t gt,int64_t gh,bool query_half,bool allow_fp16_math,bool query_pre_rotated) {
    if (!query || !ks || !vs || !relative || !blocks || !selected || !positions || !output || !rotated ||
        !partial || !stats || !block_log_mass || !log_norm || queries<=0 || heads<=0 || kv_heads<=0 ||
        heads%kv_heads || width<=0 || width>padded || count<=0 || parts<=0 || parts>128 || ts<=0 || hs<=0 || (gate && (gt<=0 || gh<=0)))
        return int(cudaErrorInvalidValue);
#define CALL(W,S) launch_attention<W,S>(query,ks,vs,relative,blocks,selected,positions,output,rotated,partial,stats,block_log_mass,log_norm,hits,mass,queries,heads,kv_heads,width,count,parts,ts,hs,stream,gate,gate_half,output_half,gt,gh,query_half,allow_fp16_math,query_pre_rotated,staged_keys,staged_values)
    if(snapshots){switch(padded){case 256:return CALL(256,true);case 512:return CALL(512,true);case 1024:return CALL(1024,true);default:return int(cudaErrorInvalidValue);}}
    switch(padded) {case 256:return CALL(256,false);case 512:return CALL(512,false);case 1024:return CALL(1024,false);default:return int(cudaErrorInvalidValue);}
#undef CALL
}
int tq_sparse_attention(const void* query,const float* ks,const float* vs,const uint16_t* relative,
    const TqBlockView* blocks,const SelectedBlock* selected,const int64_t* positions,void* output,
    float* rotated,float* partial,float* stats,float* block_log_mass,float* log_norm,float* hits,float* mass,
    int queries,int heads,int kv_heads,int width,int padded,int count,int parts,int64_t ts,int64_t hs,cudaStream_t stream,bool snapshots,
    const void* gate,bool gate_half,bool output_half,int64_t gt,int64_t gh,bool query_half,bool allow_fp16_math,bool query_pre_rotated) {
    return tq_sparse_attention_staged(nullptr,nullptr,query,ks,vs,relative,blocks,selected,positions,output,
        rotated,partial,stats,block_log_mass,log_norm,hits,mass,queries,heads,kv_heads,width,padded,count,parts,
        ts,hs,stream,snapshots,gate,gate_half,output_half,gt,gh,query_half,allow_fp16_math,query_pre_rotated);
}

bool tq_prefill_staging_available(int heads,int kv_heads,int padded) {
    if(heads<=0 || kv_heads<=0 || heads%kv_heads || heads/kv_heads<4)return false;
    switch(padded){case 256:return mma_attention_available<256,false,true>();
        case 512:return mma_attention_available<512,false,true>();
        case 1024:return mma_attention_available<1024,false,true>();default:return false;}
}
int tq_stage_attention_blocks(const TqBlockView* blocks,const uint16_t* relative,
    uint16_t* keys,uint16_t* values,int count,int heads,int padded,cudaStream_t stream,
    const SelectedBlock* selected,int queries,int selected_count,uint32_t* requested) {
    if(!blocks || !relative || !keys || !values || !selected || !requested || queries<=0 || selected_count<=0 ||
       count<=0 || heads<=0 || count>INT_MAX/heads || (padded!=256 && padded!=512 && padded!=1024))
        return int(cudaErrorInvalidValue);
    clear_requested_blocks<<<unsigned((int64_t(count)+255)/256),256,0,stream>>>(requested,count);
    const auto total=int64_t(queries)*selected_count;
    mark_requested_blocks<<<unsigned((total+255)/256),256,0,stream>>>(selected,requested,total,count);
#define STAGE(W) stage_attention_blocks<W><<<dim3(count*heads,2),128,0,stream>>>(blocks,reinterpret_cast<const half*>(relative),reinterpret_cast<half*>(keys),reinterpret_cast<half*>(values),heads,requested)
    switch(padded){case 256:STAGE(256);break;case 512:STAGE(512);break;case 1024:STAGE(1024);break;default:return int(cudaErrorInvalidValue);}
#undef STAGE
    return int(cudaGetLastError());
}
} // namespace tyloinfer::qsa
