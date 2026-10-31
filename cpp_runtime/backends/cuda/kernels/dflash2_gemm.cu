// Adapted from Strata src/kernels/cuda/dflash2.cu, Copyright (c) 2026
// Niko1221 and contributors, MIT: ../../../third_party/strata.LICENSE.
// Same 8-row/512-column tiles and warp GEMM; bounded padding supports tails.
#include "dflash2.h"
#include "kernel_dispatch_trace.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <climits>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include "dflash2_q8.cuh"

namespace mfq::cuda::dflash2 {
namespace {
void check() {
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
__device__ __forceinline__ int dot4(int a,int b,int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    // DP4A was introduced at compute capability 6.1.
    return __dp4a(a,b,c);
#else
    for(int i=0;i<4;++i)c+=int(static_cast<signed char>(unsigned(a)>>(8*i)))*
        int(static_cast<signed char>(unsigned(b)>>(8*i)));
    return c;
#endif
}
__device__ __forceinline__ int4 streaming_weight(const int4* pointer) {
    // Weight codes are consumed once per row tile and reused in registers.
    // Keep the globally shared activation codes/scales in L1 instead.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
    int4 result;
    asm("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" :
        "=r"(result.x),"=r"(result.y),"=r"(result.z),"=r"(result.w):"l"(pointer));
    return result;
#else
    return *pointer;
#endif
}
int padded_columns(int columns) {
    if(columns<=0 || columns%32 || columns>INT_MAX-511)
        throw std::invalid_argument("DFlash2 Q8_0 columns must contain whole blocks");
    return (columns+511)/512*512;
}
int padded_rows(int rows) {
    if(rows<=0 || rows>INT_MAX-31)throw std::invalid_argument("DFlash2 Q8_0 rows are invalid");
    return (rows+31)/32*32;
}
template<bool Reverse>
__global__ void tile(const unsigned char* source,unsigned char* destination,
        int rows,int columns,int pk,int pn) {
    const auto b=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int nb=columns/32,np=pk/32;
    const auto count=std::int64_t(Reverse?rows:pn)*(Reverse?nb:np);
    if(b>=count)return;
    const auto row=b/(Reverse?nb:np);
    const int blk=b%(Reverse?nb:np),r=row%8,st=blk/16,bi=blk%16;
    const auto scale=row/8*8*np+std::int64_t(st)*128+r*16+bi;
    const auto code=row/8*8*pk+std::int64_t(st)*4096+r*512+bi*32;
    if constexpr(Reverse) {
        const auto* scales=source+std::int64_t(pn)*pk;
        destination[b*34]=scales[scale*2];destination[b*34+1]=scales[scale*2+1];
        for(int i=0;i<32;++i)destination[b*34+2+i]=source[code+i];
    } else {
        auto* scales=destination+std::int64_t(pn)*pk;
        const bool valid=row<rows && blk<nb;
        const auto offset=(row*nb+blk)*34;
        scales[scale*2]=valid?source[offset]:0;
        scales[scale*2+1]=valid?source[offset+1]:0;
        for(int i=0;i<32;++i)destination[code+i]=valid?source[offset+2+i]:0;
    }
}
__global__ void quantize(const float* input,signed char* codes,float* scales,
        int tokens,int columns,int pk) {
    const int block=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
    if(block>=pk/32)return;
    const int column=block*32+lane,padded_tokens=(tokens+7)/8*8;
    for(std::int64_t row=blockIdx.y;row<padded_tokens;row+=gridDim.y) {
        const float value=row<tokens && column<columns?input[std::int64_t(row)*columns+column]:0.f;
        store_planar_q8(value,codes,scales,int(row),column,pk);
    }
}
__global__ void activate_float(const float* gate,const float* up,float* output,std::int64_t count) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)output[i]=__bfloat162float(__float2bfloat16_rn(gate[i]/(1.f+__expf(-gate[i]))*up[i]));
}
__global__ void activate_float_q8(const float* gate,const float* up,float* output,signed char* codes,
        float* scales,int rows,int width,int padded) {
    const int row=blockIdx.y,column=blockIdx.x*256+threadIdx.x;
    const bool live=row<rows && column<width;
    const auto i=int64_t(row)*width+column;
    const float value=live?__bfloat162float(__float2bfloat16_rn(gate[i]/(1.f+__expf(-gate[i]))*up[i])):0.f;
    if(live)output[i]=value;
    store_planar_q8(value,codes,scales,row,column,padded);
}
struct Df2Seg { const int8_t* q; const __half* d; float* y; int n_out; int row0; };
struct Df2Segs { Df2Seg s[3]; int n; };

template <int WARPS, bool COMPUTE, int Tokens=8, int UNR=2,int Rows=8>
__device__ __forceinline__ void gemm_tile(Df2Segs segs, const int8_t* __restrict__ xq,
        const float* __restrict__ xd, int n_in, int ncols,int group,int first) {
    constexpr int R=Rows;
    static_assert(R>0 && 8%R==0);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row_g = (group * WARPS + warp) * R;   // first row of this warp's group, over all the launch's matrices
    // Dynamic indexing materialized the whole 104-byte descriptor in local
    // memory per thread. Constant fields remain in kernel parameter storage.
    Df2Seg sg=segs.s[0];
    if(segs.n>1 && row_g>=segs.s[1].row0)sg=segs.s[1];
    if(segs.n>2 && row_g>=segs.s[2].row0)sg=segs.s[2];
    const int row0 = row_g - sg.row0;
    const int nb = n_in / 32, steps = nb / 16;
    // Preserve the eight-row weight image while a warp computes fewer rows.
    // Eight live input columns then need 32 accumulators rather than 64.
    const int4* wq = reinterpret_cast<const int4*>(sg.q + (size_t)(row0/8)*8*n_in+(row0%8)*512) + lane;
    const __half* wd = sg.d + (size_t)(row0/8)*8*nb+(row0%8)*16 + (lane >> 1);
    const int4* xv = reinterpret_cast<const int4*>(xq+size_t(first)*n_in) + lane;
    xd+=size_t(first)*nb;
    float acc[R][Tokens];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < Tokens; ++c) acc[r][c] = 0.0f;
#pragma unroll UNR
    for (int st = 0; st < steps; ++st) {
        int4 w[R];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            // The narrow-row/eight-column tile has more activation reuse
            // per weight row. Preserve its input working set in L1; the wide
            // short-batch tile retains its existing cached loads.
            if constexpr(Rows<8)w[r]=streaming_weight(wq+size_t(st)*256+r*32);
            else w[r]=wq[size_t(st)*256+r*32];
            dw[r] = __half2float(wd[(size_t) st * 128 + r * 16]);
        }
        if (!COMPUTE) {   // bench only: the same loads, almost no arithmetic
            for (int r = 0; r < R; ++r) acc[r][0] += (float) (w[r].x ^ w[r].y ^ w[r].z ^ w[r].w) * dw[r];
            continue;
        }
        float xs[Tokens];
#pragma unroll
        for (int c = 0; c < Tokens; ++c) xs[c] = xd[(size_t) c * nb + st * 16 + (lane >> 1)];
#pragma unroll
        for (int c = 0; c < Tokens; ++c) {
            const int4 x = xv[(size_t) c * (n_in / 16) + st * 32];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int isum = dot4(w[r].x, x.x, 0);
                isum = dot4(w[r].y, x.y, isum);
                isum = dot4(w[r].z, x.z, isum);
                isum = dot4(w[r].w, x.w, isum);
                acc[r][c] = fmaf(dw[r] * xs[c], (float) isum, acc[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < Tokens; ++c) {
            float v = acc[r][c];
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            if (lane == 0 && c < ncols && row0 + r < sg.n_out) sg.y[(size_t(first)+c)*sg.n_out+row0+r] = __bfloat162float(__float2bfloat16_rn(v));
        }
}

template<int WARPS,bool COMPUTE,int Tokens=8,int UNR=2,int Rows=8>
__global__ void __launch_bounds__(WARPS*32) gemm8_kernel(Df2Segs segs,const int8_t* xq,
        const float* xd,int n_in,int ncols) {
    gemm_tile<WARPS,COMPUTE,Tokens,UNR,Rows>(segs,xq,xd,n_in,ncols,blockIdx.x,0);
}
__global__ void __launch_bounds__(128) gemm_many_kernel(Df2Segs segs,const int8_t* xq,
        const float* xd,int n_in,int tokens,int groups) {
    const int tiles=(tokens+7)/8;
    // Nearby CTAs reuse a weight row group across input tiles in L2. A
    // bounded grid-stride launch also supports large token/output axes.
    for(std::int64_t work=blockIdx.x;work<std::int64_t(groups)*tiles;work+=gridDim.x) {
        const int first=int(work%tiles)*8,live=min(8,tokens-first);
        gemm_tile<4,true,8,2,4>(segs,xq,xd,n_in,live,int(work/tiles),first);
    }
}

}
void q8_tile(const void* blocks,void* tiles,int rows,int columns,cudaStream_t stream) {
    const int pk=padded_columns(columns),pn=padded_rows(rows);
    const auto count=std::int64_t(pn)*(pk/32);
    tile<false><<<unsigned((count+255)/256),256,0,stream>>>(static_cast<const unsigned char*>(blocks),
        static_cast<unsigned char*>(tiles),rows,columns,pk,pn);check();
}
void q8_untile(const void* tiles,void* blocks,int rows,int columns,cudaStream_t stream) {
    const int pk=padded_columns(columns),pn=padded_rows(rows);
    const auto count=std::int64_t(rows)*(columns/32);
    tile<true><<<unsigned((count+255)/256),256,0,stream>>>(static_cast<const unsigned char*>(tiles),
        static_cast<unsigned char*>(blocks),rows,columns,pk,pn);check();
}
void q8_quantize(const float* input,void* codes,float* scales,int tokens,int columns,cudaStream_t stream) {
    const int pk=padded_columns(columns);
    if(tokens<1 || tokens>INT_MAX-7)throw std::invalid_argument("DFlash2 padded token rows overflow");
    quantize<<<dim3((pk/32+7)/8,std::min((tokens+7)/8*8,65535)),256,0,stream>>>(input,static_cast<signed char*>(codes),scales,tokens,columns,pk);check();
}
void q8_gemm(const Q8TiledMatrix* matrices,int count,const void* codes,const float* scales,
        int tokens,int columns,cudaStream_t stream) {
    const int pk=padded_columns(columns);
    if(count<1 || count>3 || tokens<1 || tokens>INT_MAX-7)throw std::invalid_argument("DFlash2 tiled GEMM batch differs");
    Df2Segs segs{};segs.n=count;int rows=0;
    for(int i=0;i<count;++i) {
        const auto& m=matrices[i];
        if(m.padded_rows!=padded_rows(m.rows) || rows>INT_MAX-m.padded_rows)
            throw std::invalid_argument("DFlash2 tiled GEMM row padding differs");
        const auto* q=static_cast<const int8_t*>(m.weight);
        segs.s[i]={q,reinterpret_cast<const half*>(q+std::int64_t(m.padded_rows)*pk),m.output,m.rows,rows};
        rows+=m.padded_rows;
    }
    static std::once_flag traced;
    const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");
    if(trace && trace[0]!='0')std::call_once(traced,[&] {
        std::fprintf(stderr,"runtime_dflash2_q8_tiled actual_match=1 tokens=%d matrices=%d columns=%d padded_columns=%d rows=%d\n",
            tokens,count,columns,pk,rows);
    });
    if(tokens>8) {
        const int groups=rows/16;
        const auto blocks=std::min<std::int64_t>(std::int64_t(groups)*((tokens+7)/8),65535);
        gemm_many_kernel<<<unsigned(blocks),128,0,stream>>>(segs,static_cast<const int8_t*>(codes),scales,pk,tokens,groups);check();
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
            "dflash2_q8_projection_many",reinterpret_cast<const void*>(gemm_many_kernel),
            "M="+std::to_string(tokens)+" matrices="+std::to_string(count)+" K="+std::to_string(columns)+" token_tile=8");
        return;
    }
    // Reuse decoded weights across live activation columns. The tile's
    // capacity follows M; short committed-context updates do not execute
    // seven unused columns just because the proposal block holds eight.
    const auto launch=[&]<int Capacity>() {
        constexpr int Rows=Capacity>4?4:8;
        const auto kernel=gemm8_kernel<4,true,Capacity,2,Rows>;
        kernel<<<rows/(4*Rows),128,0,stream>>>(segs,static_cast<const int8_t*>(codes),scales,pk,tokens);check();
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
            "dflash2_q8_projection",reinterpret_cast<const void*>(kernel),
            "M="+std::to_string(tokens)+" columns="+std::to_string(Capacity)+
            " N="+std::to_string(rows)+" K="+std::to_string(columns)+" tile_rows="+std::to_string(Rows)+
            " weight_cache="+(Rows<8?"cg":"default"));
    };
    if(tokens==1)launch.template operator()<1>();
    else if(tokens==2)launch.template operator()<2>();
    else if(tokens<=4)launch.template operator()<4>();
    else launch.template operator()<8>();
}
void swiglu_float(const float* gate,const float* up,float* output,std::int64_t elements,cudaStream_t stream) {
    if(elements<=0)return;
    activate_float<<<unsigned((elements+255)/256),256,0,stream>>>(gate,up,output,elements);check();
}
void swiglu_float_q8(const float* gate,const float* up,float* output,void* codes,float* scales,
        int rows,int width,cudaStream_t stream) {
    if(rows<1 || rows>8)throw std::invalid_argument("DFlash2 fused SwiGLU/Q8 requires 1..8 rows");
    const int padded=padded_columns(width);
    activate_float_q8<<<dim3(padded/256,8),256,0,stream>>>(gate,up,output,static_cast<signed char*>(codes),scales,rows,width,padded);check();
    if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,
        "dflash2_swiglu_q8",reinterpret_cast<const void*>(activate_float_q8),"M="+std::to_string(rows)+" width="+std::to_string(width));
}
}
