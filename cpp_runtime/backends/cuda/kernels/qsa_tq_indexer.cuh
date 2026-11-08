#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace tyloinfer::qsa { namespace {
// Keep INT16 affine storage and reconstruct keys inside each warp. Two key
// fragments reuse the query decomposition; three TF32 products accumulate in
// FP32 without persistent expanded keys or shared-memory tiles.

__device__ __forceinline__ void indexer_tensor_mma(float (&c)[4],const float (&a)[4],const float (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(__float_as_uint(a[0])),"r"(__float_as_uint(a[1])),
          "r"(__float_as_uint(a[2])),"r"(__float_as_uint(a[3])),
          "r"(__float_as_uint(b[0])),"r"(__float_as_uint(b[1])));
#endif
}
__device__ __forceinline__ float indexer_tensor_round(float x) {
#if __CUDA_ARCH__ >= 800
    std::uint32_t bits;
    asm("cvt.rna.tf32.f32 %0,%1;" : "=r"(bits) : "f"(x));
    return __uint_as_float(bits);
#else
    return x;
#endif
}
template<bool Records,int Columns=2,int Rows=1> __global__ void affine_scores_tensor(const float* query,const std::int16_t* keys,
        const std::uint16_t* scales,const std::int16_t* zero,const std::int64_t* positions,
        float* output,int queries,int heads,int width,int blocks,int pooling) {
#if __CUDA_ARCH__ >= 800
    const int lane=threadIdx.x&31,group=lane>>2,sub=lane&3;
    const int key_base=(blockIdx.x*(blockDim.x/32)+threadIdx.x/32)*8*Columns;
    if(key_base>=blocks)return;
    const int per_fragment=heads<=16?16/heads:1,qtile=Rows*per_fragment,qbase=blockIdx.y*qtile;
    std::int64_t visible=positions?0:blocks;
    if(positions)for(int local=0;local<qtile && qbase+local<queries;++local)
        visible=max(visible,(positions[qbase+local]+1)/pooling);
    if(key_base>=visible) {
        if(lane<8)for(int col=0;col<Columns;++col)if(key_base+8*col+lane<blocks)
            for(int local=0;local<qtile && qbase+local<queries;++local)
                output[std::int64_t(qbase+local)*blocks+key_base+8*col+lane]=-CUDART_INF_F;
        return;
    }
    const int stride=width+(Records?2:0);
    float key_scales[Columns];int zeros[Columns];
    #pragma unroll
    for(int col=0;col<Columns;++col) {
        const int key=key_base+8*col+group;
        key_scales[col]=key<blocks?__half2float(__ushort_as_half(Records?
            std::uint16_t(keys[std::int64_t(key)*stride+width]):scales[key])):0.f;
        zeros[col]=key<blocks?(Records?keys[std::int64_t(key)*stride+width+1]:zero[key]):0;
    }
    for(int hbase=0;hbase<heads;hbase+=16) {
        float accum[Rows][Columns][4]={};
        for(int dbase=0;dbase<width;dbase+=8) {
            float bh[Columns][2],bl[Columns][2];
            #pragma unroll
            for(int col=0;col<Columns;++col) {
                const int key=key_base+8*col+group;
                #pragma unroll
                for(int i=0;i<2;++i) {
                    const int d=dbase+sub+4*i;
                    const float x=key<blocks && d<width?
                        float(int(keys[std::int64_t(key)*stride+d])-zeros[col])*key_scales[col]:0.f;
                    bh[col][i]=indexer_tensor_round(x);bl[col][i]=indexer_tensor_round(x-bh[col][i]);
                }
            }
            #pragma unroll
            for(int row_fragment=0;row_fragment<Rows;++row_fragment) {
                float ah[4],al[4];
                #pragma unroll
                for(int i=0;i<4;++i) {
                    const int row=group+(i&1)*8,d=dbase+sub+(i>>1)*4;
                    const int qfirst=qbase+row_fragment*per_fragment;
                    const auto qr=std::int64_t(qfirst)*heads+(heads<=16?row:hbase+row);
                    const bool active=heads<=16?qr<std::int64_t(queries)*heads:qfirst<queries && hbase+row<heads;
                    const float x=active && d<width?query[qr*width+d]:0.f;
                    ah[i]=indexer_tensor_round(x);al[i]=indexer_tensor_round(x-ah[i]);
                }
                #pragma unroll
                for(int col=0;col<Columns;++col) {
                    indexer_tensor_mma(accum[row_fragment][col],ah,bh[col]);
                    indexer_tensor_mma(accum[row_fragment][col],al,bh[col]);
                    indexer_tensor_mma(accum[row_fragment][col],ah,bl[col]);
                }
            }
        }
        #pragma unroll
        for(int row_fragment=0;row_fragment<Rows;++row_fragment) {
        #pragma unroll
        for(int col=0;col<Columns;++col) {
        const auto& c=accum[row_fragment][col];
        for(int local=0;local<per_fragment && qbase+row_fragment*per_fragment+local<queries;++local) {
            const int first=heads<=16?local*heads:0;
            const int count=heads<=16?heads:min(16,heads-hbase);
            float score0=0.f,score1=0.f;
            for(int h=0;h<count;++h) {
                const int row=first+h,source=(row&7)*4+sub;
                const float x=row<8?c[0]:c[2],y=row<8?c[1]:c[3];
                score0+=fmaxf(__shfl_sync(0xffffffffu,x,source),0.f);
                score1+=fmaxf(__shfl_sync(0xffffffffu,y,source),0.f);
            }
            if(!group) {
                const int q=qbase+row_fragment*per_fragment+local,k0=key_base+8*col+2*sub,k1=k0+1;
                const auto limit=positions?(positions[q]+1)/pooling:std::int64_t(blocks);
                const float factor=Records?rsqrtf(float(width)):1.f;
                if(k0<blocks) {
                    const auto at=std::int64_t(q)*blocks+k0;
                    output[at]=k0<limit?(hbase?output[at]:0.f)+score0*factor:-CUDART_INF_F;
                }
                if(k1<blocks) {
                    const auto at=std::int64_t(q)*blocks+k1;
                    output[at]=k1<limit?(hbase?output[at]:0.f)+score1*factor:-CUDART_INF_F;
                }
            }
        }
        }
        }
    }
#endif
}

template<bool Records=false> static int launch_affine_scores_tensor(const float* query,const int16_t* keys,
    const uint16_t* scales,const int16_t* zero,const int64_t* positions,float* output,
    int queries,int heads,int width,int blocks,cudaStream_t stream,int pool=4) {
    // A single query avoids padding with the existing FP32 warp reduction.
    if(queries<2)return int(cudaErrorNotSupported);
    struct Configuration {int device=-1,grid_y=0;bool supported=false;};
    static thread_local Configuration configurations[2];
    const int per_fragment=heads<=16?16/heads:1;
    const int rows=queries>per_fragment?2:1;
    auto& config=configurations[rows-1];
    int device=0;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return int(status);
    if(config.device!=device) {
        int major=0,grid_y=0;
        status=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
        if(status!=cudaSuccess)return int(status);
        status=cudaDeviceGetAttribute(&grid_y,cudaDevAttrMaxGridDimY,device);
        if(status!=cudaSuccess)return int(status);
        bool supported=false;
        // TF32 mma and cvt.rna.tf32 require SM80. Check the compiled virtual
        // target too: an older PTX image may have compiled out these bodies.
        if(major>=8) {
            cudaFuncAttributes image{};
            status=rows==2?cudaFuncGetAttributes(&image,affine_scores_tensor<Records,2,2>):
                cudaFuncGetAttributes(&image,affine_scores_tensor<Records,2,1>);
            if(status==cudaErrorNoKernelImageForDevice || status==cudaErrorInvalidDeviceFunction) {
                (void)cudaGetLastError();status=cudaSuccess;
            }else if(status!=cudaSuccess)return int(status);
            else supported=image.ptxVersion>=80 && image.maxThreadsPerBlock>=128;
        }
        config={device,grid_y,supported};
    }
    if(!config.supported)return int(cudaErrorNotSupported);
    const int query_tile=rows*per_fragment;
    const auto grid_y=(int64_t(queries)+query_tile-1)/query_tile;
    if(grid_y>config.grid_y)return int(cudaErrorNotSupported);
    const dim3 grid(unsigned((int64_t(blocks)+63)/64),unsigned(grid_y));
    if(rows==2)affine_scores_tensor<Records,2,2><<<grid,128,0,stream>>>(
        query,keys,scales,zero,positions,output,queries,heads,width,blocks,Records?pool:4);
    else affine_scores_tensor<Records,2,1><<<grid,128,0,stream>>>(
        query,keys,scales,zero,positions,output,queries,heads,width,blocks,Records?pool:4);
    status=cudaGetLastError();
    if(status==cudaSuccess)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<unsigned> reported{0};const unsigned bit=queries<=8?1u<<(queries-2):128u;
        if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "native_indexer_affine16_tensor actual_match=1 queries=%d blocks=%d heads=%d width=%d records=%d tile_keys=64 tile_query_heads=%d tf32_products=3 fp32_accum=1\n",
            queries,blocks,heads,width,int(Records),16*rows);
    }
    return int(status);
}

// 128 threads compute a 16-query x 32-key tile. Each thread retains four
// outputs. K is decoded once per tile, outside the query-head loop. Shared
// strides are odd to avoid bank conflicts during both upload and dot products.
template<bool Records=false> __global__ void affine_scores_tiled(const float* query,const int16_t* keys,const uint16_t* scales,
    const int16_t* zero,const int64_t* positions,float* output,int queries,int heads,int width,int blocks,int pool=4) {
    constexpr int QM=16,KN=32;const int tid=threadIdx.x;
    const int qbase=blockIdx.y*QM,kbase=blockIdx.x*KN;
    const int qr=tid/16,kr=tid%16;
    const int q0=qbase+qr,q1=q0+8,k0=kbase+kr,k1=k0+16;
    const bool vq0=q0<queries,vq1=q1<queries;
    const int64_t p0=vq0?(positions?(positions[q0]+1)/pool:blocks):0,p1=vq1?(positions?(positions[q1]+1)/pool:blocks):0;
    const bool a00=vq0 && k0<blocks && k0<p0,a01=vq0 && k1<blocks && k1<p0;
    const bool a10=vq1 && k0<blocks && k0<p1,a11=vq1 && k1<blocks && k1<p1;
    const bool any=__syncthreads_or(a00||a01||a10||a11);
    if(!any) {
        if(vq0 && k0<blocks)output[int64_t(q0)*blocks+k0]=-CUDART_INF_F;
        if(vq0 && k1<blocks)output[int64_t(q0)*blocks+k1]=-CUDART_INF_F;
        if(vq1 && k0<blocks)output[int64_t(q1)*blocks+k0]=-CUDART_INF_F;
        if(vq1 && k1<blocks)output[int64_t(q1)*blocks+k1]=-CUDART_INF_F;
        return;
    }
    extern __shared__ float scratch[];float* sq=scratch;float* sk=sq+QM*(width+1);
    for(int i=tid;i<KN*width;i+=128) {
        const int row=i/width,d=i%width,at=kbase+row;
        const auto stride=width+(Records?2:0);
        const int zp=at<blocks?(Records?keys[int64_t(at)*stride+width+1]:zero[at]):0;
        const auto scale=at<blocks?(Records?uint16_t(keys[int64_t(at)*stride+width]):scales[at]):uint16_t(0);
        sk[d*(KN+1)+row]=at<blocks?float(int(keys[int64_t(at)*stride+d])-zp)*__half2float(__ushort_as_half(scale)):0.f;
    }
    float s00=0,s01=0,s10=0,s11=0;
    for(int h=0;h<heads;++h) {
        for(int i=tid;i<QM*width;i+=128) {
            const int row=i/width,d=i%width,at=qbase+row;
            sq[row*(width+1)+d]=at<queries?query[(int64_t(at)*heads+h)*width+d]:0.f;
        }
        __syncthreads();
        float p00[32]={},p01[32]={},p10[32]={},p11[32]={};
        for(int base=0;base<width;base+=32) {
            #pragma unroll
            for(int lane=0;lane<32;++lane) {
                const int d=base+lane;
                if(d<width) {
                    const float x0=sq[qr*(width+1)+d],x1=sq[(qr+8)*(width+1)+d];
                    const float y0=sk[d*(KN+1)+kr],y1=sk[d*(KN+1)+kr+16];
                    p00[lane]=fmaf(x0,y0,p00[lane]);p01[lane]=fmaf(x0,y1,p01[lane]);
                    p10[lane]=fmaf(x1,y0,p10[lane]);p11[lane]=fmaf(x1,y1,p11[lane]);
                }
            }
        }
        // Match lane zero of the original descending XOR reduction, while
        // keeping each partial local to the owner of the output neuron.
        #pragma unroll
        for(int step=16;step;step/=2) {
            #pragma unroll
            for(int lane=0;lane<16;++lane)if(lane<step) {
                p00[lane]=__fadd_rn(p00[lane],p00[lane+step]);p01[lane]=__fadd_rn(p01[lane],p01[lane+step]);
                p10[lane]=__fadd_rn(p10[lane],p10[lane+step]);p11[lane]=__fadd_rn(p11[lane],p11[lane+step]);
            }
        }
        s00+=fmaxf(p00[0],0.f);s01+=fmaxf(p01[0],0.f);s10+=fmaxf(p10[0],0.f);s11+=fmaxf(p11[0],0.f);
        __syncthreads();
    }
    if(vq0 && k0<blocks)output[int64_t(q0)*blocks+k0]=a00?s00*(Records?rsqrtf(float(width)):1.f):-CUDART_INF_F;
    if(vq0 && k1<blocks)output[int64_t(q0)*blocks+k1]=a01?s01*(Records?rsqrtf(float(width)):1.f):-CUDART_INF_F;
    if(vq1 && k0<blocks)output[int64_t(q1)*blocks+k0]=a10?s10*(Records?rsqrtf(float(width)):1.f):-CUDART_INF_F;
    if(vq1 && k1<blocks)output[int64_t(q1)*blocks+k1]=a11?s11*(Records?rsqrtf(float(width)):1.f):-CUDART_INF_F;
}

// Geometry and actual per-device resources select this path. No model,
// compute-capability, or device-capacity whitelist is needed.
template<bool Records=false> static int launch_affine_scores_tiled(const float* query,const int16_t* keys,const uint16_t* scales,
    const int16_t* zero,const int64_t* positions,float* output,int queries,int heads,int width,int blocks,cudaStream_t stream,int pool=4) {
    if(queries<16 || blocks<32)return int(cudaErrorNotSupported);
    struct Configuration {int device=-1,shared=0,grid_y=0;};
    static thread_local Configuration config;
    int device=0;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return int(status);
    if(config.device!=device) {
        int ordinary=0,optin=0,grid_y=0;
        status=cudaDeviceGetAttribute(&ordinary,cudaDevAttrMaxSharedMemoryPerBlock,device);
        if(status!=cudaSuccess)return int(status);
        status=cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device);
        if(status!=cudaSuccess)return int(status);
        status=cudaDeviceGetAttribute(&grid_y,cudaDevAttrMaxGridDimY,device);
        if(status!=cudaSuccess)return int(status);
        cudaFuncAttributes image{};status=cudaFuncGetAttributes(&image,affine_scores_tiled<Records>);
        if(status!=cudaSuccess)return int(status);
        const int shared=int(std::max(ordinary,optin)-image.sharedSizeBytes);
        status=cudaFuncSetAttribute(affine_scores_tiled<Records>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
        if(status!=cudaSuccess)return int(status);
        config={device,shared,grid_y};
    }
    const auto shared=(int64_t(16)*(int64_t(width)+1)+int64_t(width)*33)*sizeof(float);
    const auto grid_y=(int64_t(queries)+15)/16;
    if(shared>std::size_t(config.shared) || grid_y>config.grid_y)return int(cudaErrorNotSupported);
    affine_scores_tiled<Records><<<dim3(unsigned((int64_t(blocks)+31)/32),unsigned(grid_y)),128,std::size_t(shared),stream>>>(
        query,keys,scales,zero,positions,output,queries,heads,width,blocks,pool);
    status=cudaGetLastError();
    if(status==cudaSuccess)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
            "native_indexer_affine16_tiled actual_match=1 queries=%d blocks=%d heads=%d width=%d tile_queries=16 tile_keys=32 shared_bytes=%zu fp32=1\n",
            queries,blocks,heads,width,std::size_t(shared));
    }
    return int(status);
}
} } // namespace tyloinfer::qsa
