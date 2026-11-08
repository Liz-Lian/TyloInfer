#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <limits>
#include <climits>

namespace tyloinfer::qsa { namespace {
// Keys retain their INT16 affine representation. Three signed query digits
// supply near-FP32 precision; INT64 recombination preserves affine cancellation.
struct alignas(16) IndexerIntegerQuery { float scale; int visible; long long sum; };

__global__ void indexer_integer_prepare(const float* input, signed char* digits, IndexerIntegerQuery* info,
                             int rows, int width, int padded,const std::int64_t* positions,
                             int heads,int blocks,int pooling) {
    const int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    __shared__ float maxima[8];
    __shared__ long long sums[8];
    float amax=0.f;
    for(int d=tid;d<width;d+=256)amax=fmaxf(amax,fabsf(input[std::int64_t(row)*width+d]));
    for(int shift=16;shift;shift>>=1)amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,shift));
    if(!lane)maxima[warp]=amax;
    __syncthreads();
    amax=0.f;
    #pragma unroll
    for(int w=0;w<8;++w)amax=fmaxf(amax,maxima[w]);
    long long sum=0;
    for(int d=tid;d<padded;d+=256) {
        const float x=d<width?input[std::int64_t(row)*width+d]:0.f;
        const int value=amax>0.f?__float2int_rn((x/amax)*8355711.f):0;
        const int low=((value+128)&255)-128;
        const int upper=(value-low)/256;
        const int middle=((upper+128)&255)-128;
        const int high=(upper-middle)/256;
        const auto at=std::int64_t(row)*padded+d;
        digits[at]=static_cast<signed char>(high);
        digits[std::int64_t(rows)*padded+at]=static_cast<signed char>(middle);
        digits[std::int64_t(2)*rows*padded+at]=static_cast<signed char>(low);
        sum+=value;
    }
    for(int shift=16;shift;shift>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,shift);
    if(!lane)sums[warp]=sum;
    __syncthreads();
    if(!tid) {
        sum=0;for(int w=0;w<8;++w)sum+=sums[w];
        int visible=blocks;
        if(positions) {
            const auto position=positions[row/heads];
            if(position<0)visible=0;
            else {
                // Only the visible key count is needed. Unsigned addition
                // also keeps the final representable position well-defined.
                const auto count=(static_cast<unsigned long long>(position)+1)/unsigned(pooling);
                visible=count<unsigned(blocks)?int(count):blocks;
            }
        }
        info[row]={amax/8355711.f,visible,sum};
    }
}

template<bool Unsigned> __device__ __forceinline__ void indexer_integer_mma(int (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    if constexpr(Unsigned) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
#endif
}

template<bool Records,int Fragments> __global__ void indexer_integer_scores(
        const signed char* digits,const IndexerIntegerQuery* info,const std::int16_t* keys,
        const std::uint16_t* scales,const std::int16_t* zero,
        float* output,int queries,int heads,int width,int padded,int blocks,int per_fragment) {
#if __CUDA_ARCH__ >= 800
    const int lane=threadIdx.x&31,group=lane>>2,sub=lane&3;
    const int key_base=(blockIdx.x*(blockDim.x/32)+threadIdx.x/32)*16;
    if(key_base>=blocks)return;

    const int qbase=blockIdx.y*Fragments*per_fragment;
    const int rows=queries*heads,stride=width+(Records?2:0);
    int visible=0;
    for(int local=0;local<Fragments*per_fragment && qbase+local<queries;++local)
        visible=max(visible,info[(qbase+local)*heads].visible);
    if(key_base>=visible) {
        if(lane<16 && key_base+lane<blocks)
            for(int local=0;local<Fragments*per_fragment && qbase+local<queries;++local)
                output[std::int64_t(qbase+local)*blocks+key_base+lane]=-CUDART_INF_F;
        return;
    }
    float key_scale[2];
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int key=key_base+group+8*r;
        const auto offset=std::int64_t(key)*stride;
        key_scale[r]=key<blocks?__half2float(__ushort_as_half(Records?
            std::uint16_t(keys[offset+width]):scales[key])):0.f;
    }
    for(int hbase=0;hbase<heads;hbase+=8) {
        int dot[Fragments][3][2][4]={};
        for(int dbase=0;dbase<width;dbase+=32) {
            unsigned a[2][4]={};
            #pragma unroll
            for(int i=0;i<4;++i) {
                const int key=key_base+group+8*(i&1),d=dbase+sub*4+16*(i>>1);
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const unsigned value=key<blocks && d+j<width?
                        std::uint16_t(keys[std::int64_t(key)*stride+d+j]):0;
                    a[0][i]|=(value&255u)<<(8*j);
                    a[1][i]|=(value>>8)<<(8*j);
                }
            }
            #pragma unroll
            for(int f=0;f<Fragments;++f) {
                const int first=qbase+f*per_fragment;
                const int qr=first*heads+(heads<=8?group:hbase+group);
                const bool active=heads<=8?qr<rows:first<queries && hbase+group<heads;
                #pragma unroll
                for(int plane=0;plane<3;++plane) {
                    unsigned b[2]={};
                    if(active) {
                        const auto* source=digits+(std::int64_t(plane)*rows+qr)*padded+dbase+sub*4;
                        b[0]=*reinterpret_cast<const unsigned*>(source);
                        b[1]=*reinterpret_cast<const unsigned*>(source+16);
                    }
                    indexer_integer_mma<true>(dot[f][plane][0],a[0],b);
                    indexer_integer_mma<false>(dot[f][plane][1],a[1],b);
                }
            }
        }
        int key_zero[2];
        #pragma unroll
        for(int r=0;r<2;++r) {
            const int key=key_base+group+8*r;
            const auto offset=std::int64_t(key)*stride;
            key_zero[r]=key<blocks?(Records?keys[offset+width+1]:zero[key]):0;
        }
        #pragma unroll
        for(int f=0;f<Fragments;++f) {
            float values[2][2]={};
            #pragma unroll
            for(int i=0;i<4;++i) {
                const int col=2*sub+(i&1),r=i>>1;
                const int first=qbase+f*per_fragment;
                const int qr=first*heads+(heads<=8?col:hbase+col);
                const bool active=heads<=8?qr<rows:first<queries && hbase+col<heads;
                if(active) {
                    const auto param=info[qr];
                    long long total=0;
                    #pragma unroll
                    for(int plane=0;plane<3;++plane) {
                        const long long part=static_cast<long long>(dot[f][plane][1][i])*256+
                                             dot[f][plane][0][i];
                        total+=part*(plane==0?65536:plane==1?256:1);
                    }
                    total-=static_cast<long long>(key_zero[r])*param.sum;
                    values[r][i&1]=fmaxf(float(total)*param.scale*key_scale[r],0.f);
                }
            }
            for(int local=0;local<per_fragment && qbase+f*per_fragment+local<queries;++local) {
                float score[2]={};
                const int first=heads<=8?local*heads:0,count=heads<=8?heads:min(heads-hbase,8);
                for(int h=0;h<count;++h) {
                    const int col=first+h,source=group*4+col/2;
                    #pragma unroll
                    for(int r=0;r<2;++r) {
                        const float value=col&1?values[r][1]:values[r][0];
                        score[r]+=__shfl_sync(0xffffffffu,value,source);
                    }
                }
                if(!sub) {
                    const int q=qbase+f*per_fragment+local;
                    const auto limit=info[q*heads].visible;
                    const float factor=Records?rsqrtf(float(width)):1.f;
                    #pragma unroll
                    for(int r=0;r<2;++r) {
                        const int key=key_base+group+8*r;
                        if(key<blocks) {
                            const auto at=std::int64_t(q)*blocks+key;
                            output[at]=key<limit?(hbase?output[at]:0.f)+score[r]*factor:-CUDART_INF_F;
                        }
                    }
                }
            }
        }
    }
#endif
}
 } // namespace

std::size_t indexer_affine16_score_scratch(int queries,int heads,int width) {
    // Each low-byte dot is bounded by 255*128*width in signed INT32.
    // Wider rows use the ordinary scorer without integer accumulation.
    if(queries<=0 || heads<=0 || width<=0 || width>65536 ||
       std::int64_t(queries)*heads>INT_MAX)return 0;
    const std::size_t padded=(std::size_t(width)+31)/32*32,rows=std::size_t(queries)*heads;
    if(rows>(std::numeric_limits<std::size_t>::max()-15)/(3*padded+sizeof(IndexerIntegerQuery)))return 0;
    const auto payload=(3*rows*padded+15)/16*16;
    return payload+rows*sizeof(IndexerIntegerQuery);
}

namespace {
int launch_affine_scores_integer(const float* query,const std::int16_t* keys,
        const std::uint16_t* scales,const std::int16_t* zero,const std::int64_t* positions,
        float* output,int queries,int heads,int width,int blocks,int records,int pooling,
        void* workspace,std::size_t workspace_bytes,cudaStream_t stream) {
    if(!query || !keys || !output || queries<=0 || heads<=0 || width<=0 || blocks<=0 || pooling<=0 ||
       (!records && (!scales || !zero)))return int(cudaErrorInvalidValue);
    const auto needed=indexer_affine16_score_scratch(queries,heads,width);
    if(!needed || !workspace || workspace_bytes<needed || (reinterpret_cast<std::uintptr_t>(workspace)&15))
        return int(cudaErrorNotSupported);
    struct Configuration {int device=-1,max_grid_y=0;bool supported=false;};
    static thread_local Configuration config;
    int device=0;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return int(status);
    if(config.device!=device) {
        int major=0,max_grid_y=0;
        status=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
        if(status!=cudaSuccess)return int(status);
        status=cudaDeviceGetAttribute(&max_grid_y,cudaDevAttrMaxGridDimY,device);
        if(status!=cudaSuccess)return int(status);
        bool supported=false;
        // m16n8k32 integer MMA requires SM80 and a matching compiled body.
        if(major>=8) {
            cudaFuncAttributes image{};
            status=cudaFuncGetAttributes(&image,indexer_integer_scores<false,1>);
            if(status==cudaErrorNoKernelImageForDevice || status==cudaErrorInvalidDeviceFunction) {
                (void)cudaGetLastError();status=cudaSuccess;
            }else if(status!=cudaSuccess)return int(status);
            else supported=image.ptxVersion>=80 && image.maxThreadsPerBlock>=128;
        }
        config={device,max_grid_y,supported};
    }
    if(!config.supported)return int(cudaErrorNotSupported);
    const int per_fragment=heads<=8?8/heads:1;
    const int fragments=queries>2*per_fragment?4:queries>per_fragment?2:1;
    const auto grid_y=(std::int64_t(queries)+fragments*per_fragment-1)/(fragments*per_fragment);
    if(grid_y>config.max_grid_y)return int(cudaErrorNotSupported);
    const int padded=(width+31)/32*32,rows=queries*heads;
    const std::size_t payload=(std::size_t(3)*rows*padded+15)/16*16;
    auto* digits=static_cast<signed char*>(workspace);
    auto* info=reinterpret_cast<IndexerIntegerQuery*>(digits+payload);
    indexer_integer_prepare<<<rows,256,0,stream>>>(query,digits,info,rows,width,padded,positions,
        heads,blocks,records?pooling:4);
    const dim3 grid(unsigned((std::int64_t(blocks)+63)/64),unsigned(grid_y));
#define RUN(R,F) indexer_integer_scores<R,F><<<grid,128,0,stream>>>(digits,info,keys,scales,zero,output,queries,heads,width,padded,blocks,per_fragment)
    if(records){if(fragments==4){RUN(true,4);}else if(fragments==2){RUN(true,2);}else{RUN(true,1);}}
    else{if(fragments==4){RUN(false,4);}else if(fragments==2){RUN(false,2);}else{RUN(false,1);}}
#undef RUN
    status=cudaGetLastError();
    if(status==cudaSuccess)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<unsigned> reported{0};const auto bit=queries<=8?1u<<(queries-1):256u;
        if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "native_indexer_affine16_integer actual_match=1 queries=%d blocks=%d heads=%d width=%d records=%d query_bits=24 int64_affine_correction=1 workspace_bytes=%zu\n",
            queries,blocks,heads,width,records,needed);
    }
    return int(status);
}
} // namespace

int indexer_affine16_scores(const float* query,const std::int16_t* keys,
    const std::uint16_t* scales,const std::int16_t* zero,const std::int64_t* positions,
    float* output,int queries,int heads,int width,int blocks,cudaStream_t stream,
    void* workspace,std::size_t workspace_bytes) {
    if(!positions)return int(cudaErrorInvalidValue);
    const int status=launch_affine_scores_integer(query,keys,scales,zero,positions,output,
        queries,heads,width,blocks,0,4,workspace,workspace_bytes,stream);
    return status==int(cudaErrorNotSupported)?indexer_affine16_scores(query,keys,scales,zero,positions,
        output,queries,heads,width,blocks,stream):status;
}
int indexer_affine16_record_scores(const float* query,const std::int16_t* records,
    const std::int64_t* positions,float* output,int queries,int heads,int width,int blocks,
    int pooling,cudaStream_t stream,void* workspace,std::size_t workspace_bytes) {
    const int status=launch_affine_scores_integer(query,records,nullptr,nullptr,positions,output,
        queries,heads,width,blocks,1,pooling,workspace,workspace_bytes,stream);
    return status==int(cudaErrorNotSupported)?indexer_affine16_record_scores(query,records,positions,
        output,queries,heads,width,blocks,pooling,stream):status;
}
} // namespace tyloinfer::qsa
