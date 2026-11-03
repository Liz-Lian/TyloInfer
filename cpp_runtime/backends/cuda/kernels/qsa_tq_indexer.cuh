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
// 128 threads compute a 16-query x 32-key tile. Each thread retains four
// outputs. K is decoded once per tile, outside the query-head loop. Shared
// strides are odd to avoid bank conflicts during both upload and dot products.
__global__ void affine_scores_tiled(const float* query,const int16_t* keys,const uint16_t* scales,
    const int16_t* zero,const int64_t* positions,float* output,int queries,int heads,int width,int blocks) {
    constexpr int QM=16,KN=32;const int tid=threadIdx.x;
    const int qbase=blockIdx.y*QM,kbase=blockIdx.x*KN;
    const int qr=tid/16,kr=tid%16;
    const int q0=qbase+qr,q1=q0+8,k0=kbase+kr,k1=k0+16;
    const bool vq0=q0<queries,vq1=q1<queries;
    const int64_t p0=vq0?(positions[q0]+1)/4:0,p1=vq1?(positions[q1]+1)/4:0;
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
        sk[d*(KN+1)+row]=at<blocks?float(int(keys[int64_t(at)*width+d])-int(zero[at]))*__half2float(__ushort_as_half(scales[at])):0.f;
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
    if(vq0 && k0<blocks)output[int64_t(q0)*blocks+k0]=a00?s00:-CUDART_INF_F;
    if(vq0 && k1<blocks)output[int64_t(q0)*blocks+k1]=a01?s01:-CUDART_INF_F;
    if(vq1 && k0<blocks)output[int64_t(q1)*blocks+k0]=a10?s10:-CUDART_INF_F;
    if(vq1 && k1<blocks)output[int64_t(q1)*blocks+k1]=a11?s11:-CUDART_INF_F;
}

// Geometry and actual per-device resources select this path. No model,
// compute-capability, or device-capacity whitelist is needed.
static int launch_affine_scores_tiled(const float* query,const int16_t* keys,const uint16_t* scales,
    const int16_t* zero,const int64_t* positions,float* output,int queries,int heads,int width,int blocks,cudaStream_t stream) {
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
        cudaFuncAttributes image{};status=cudaFuncGetAttributes(&image,affine_scores_tiled);
        if(status!=cudaSuccess)return int(status);
        const int shared=int(std::max(ordinary,optin)-image.sharedSizeBytes);
        status=cudaFuncSetAttribute(affine_scores_tiled,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
        if(status!=cudaSuccess)return int(status);
        config={device,shared,grid_y};
    }
    const auto shared=(int64_t(16)*(int64_t(width)+1)+int64_t(width)*33)*sizeof(float);
    const auto grid_y=(int64_t(queries)+15)/16;
    if(shared>std::size_t(config.shared) || grid_y>config.grid_y)return int(cudaErrorNotSupported);
    affine_scores_tiled<<<dim3(unsigned((int64_t(blocks)+31)/32),unsigned(grid_y)),128,std::size_t(shared),stream>>>(
        query,keys,scales,zero,positions,output,queries,heads,width,blocks);
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
