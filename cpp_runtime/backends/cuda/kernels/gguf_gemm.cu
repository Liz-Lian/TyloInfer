#include "gguf.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include <mma.h>
#include <stdexcept>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace mfq::cuda::gguf {
namespace {
template<int Format>
__global__ void decode_group(const GemmJob* jobs,std::uint64_t stride,std::int64_t rows,std::int64_t columns,
        std::int64_t batch_stride,half* output) {
    const auto i=(std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x)*8;
    if(i>=rows*columns)return;
    const auto* packed=static_cast<const std::uint8_t*>(jobs[blockIdx.y].packed)+(i/columns)*stride;
    float values[8];
    gguf_values8<Format>(packed,i%columns,values);
    gguf_store8(output+std::int64_t(blockIdx.y)*batch_stride+i,values);
}
__global__ void gather_group(const GemmJob* jobs,std::int64_t tokens,std::int64_t columns,
    const half* input,std::int64_t routes,half* output) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=tokens*columns)return;
    const auto job=jobs[blockIdx.y];const auto row=i/columns;
    output[std::int64_t(blockIdx.y)*tokens*columns+i]=row<job.count?
        input[(job.slots[row]/routes)*columns+i%columns]:__float2half_rn(0.f);
}
__global__ void scatter_group(const GemmJob* jobs,std::int64_t tokens,std::int64_t columns,
    const half* input,half* output) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto job=jobs[blockIdx.y];
    if(i<job.count*columns)output[job.slots[i/columns]*columns+i%columns]=input[std::int64_t(blockIdx.y)*tokens*columns+i];
}
__global__ void scatter(const std::uint8_t* packet) {
    const auto chunk = reinterpret_cast<const UploadChunk*>(packet)[blockIdx.y];
    auto* output = reinterpret_cast<std::uint8_t*>(chunk.destination);
    const auto* input = packet + chunk.offset;
    const auto begin = std::uint64_t(blockIdx.x) * 65536;
    const auto end = min(chunk.bytes, begin + 65536);
    for (auto i = begin + std::uint64_t(threadIdx.x) * 16; i < end; i += std::uint64_t(blockDim.x) * 16) {
        if (i + 16 <= end) *reinterpret_cast<uint4*>(output+i) = *reinterpret_cast<const uint4*>(input+i);
        else for (auto tail=i;tail<end;++tail) output[tail]=input[tail];
    }
}
// Strata's grouped prompt path shares one decoded weight tile across token
// rows. Keep that organization with runtime dimensions and original GGUF
// blocks: four warps compute a 32x32 output tile, including incomplete experts.
template<int Format>
__global__ void grouped(const GemmJob* jobs, std::uint64_t stride, std::int64_t rows,
    std::int64_t columns, const half* input, std::int64_t routes, bool routed_input, half* output) {
#if __CUDA_ARCH__ >= 700
    const auto job = jobs[blockIdx.z];
    const auto token_begin = std::int64_t(blockIdx.y) * 32;
    const auto row_begin = std::int64_t(blockIdx.x) * 32;
    if (token_begin >= job.count) return;
    constexpr int KTile=128;
    __shared__ __align__(32) half x[32 * KTile], w[32 * KTile];
    __shared__ __align__(32) float result[32 * 32];
    using namespace nvcuda;
    const int warp = threadIdx.x / 32, token_tile = warp / 2 * 16, row_tile = warp % 2 * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.f);
    for (std::int64_t begin = 0; begin < columns; begin += KTile) {
        // Adjacent values share the GGUF codeword and scaling metadata. Decode
        // eight together, and amortize synchronization over eight MMA steps.
        for (int i = threadIdx.x * 8; i < 32 * KTile; i += blockDim.x * 8) {
            const auto row = row_begin + i / KTile, column = begin + i % KTile;
            const auto* packed = static_cast<const std::uint8_t*>(job.packed) + row * stride;
            uint4 decoded{};
            auto* words=reinterpret_cast<std::uint32_t*>(&decoded);
            #pragma unroll
            for(int pair=0;pair<4;++pair) {
                const auto k=column+pair*2;
                const auto a=__float2half_rn(row<rows && k<columns?gguf_value<Format>(packed,k):0.f);
                const auto b=__float2half_rn(row<rows && k+1<columns?gguf_value<Format>(packed,k+1):0.f);
                words[pair]=std::uint32_t(__half_as_ushort(a))|(std::uint32_t(__half_as_ushort(b))<<16);
            }
            *reinterpret_cast<uint4*>(w+i)=decoded;
            const auto token=token_begin+i/KTile;
            if(token<job.count && column+7<columns) {
                const auto slot=job.slots[token];
                *reinterpret_cast<uint4*>(x+i)=*reinterpret_cast<const uint4*>(
                    input+(routed_input?slot:slot/routes)*columns+column);
            } else {
                #pragma unroll
                for(int j=0;j<8;++j) x[i+j]=token<job.count && column+j<columns?
                    input[(routed_input?job.slots[token]:job.slots[token]/routes)*columns+column+j]:__float2half_rn(0.f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < KTile; k += 16) {
            wmma::load_matrix_sync(a, x + token_tile * KTile + k, KTile);
            wmma::load_matrix_sync(b, w + row_tile * KTile + k, KTile);
            wmma::mma_sync(c, a, b, c);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(result + token_tile * 32 + row_tile, c, 32, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        const auto token = token_begin + i / 32, row = row_begin + i % 32;
        if (token < job.count && row < rows) output[job.slots[token] * rows + row] = __float2half_rn(result[i]);
    }
#endif
}
__global__ void activate(const std::int64_t* slots, std::int64_t count, std::int64_t width,
    const half* gate, const half* up, half* hidden, float limit) {
    const auto i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count * width) return;
    const auto offset = slots[i / width] * width + i % width;
    float g = __half2float(gate[offset]), u = __half2float(up[offset]);
    if (limit > 0) {
        g = fminf(g, limit); u = fminf(limit, fmaxf(-limit, u));
        hidden[offset] = __float2half_rn(g / (1.f + expf(-g)) * u);
    } else {
        const auto sigmoid = __float2half_rn(1.f / (1.f + expf(-g)));
        const auto silu = __float2half_rn(g * __half2float(sigmoid));
        hidden[offset] = __float2half_rn(__half2float(silu) * u);
    }
}
__global__ void activate_gate_up(const half* gate_up,std::int64_t count,std::int64_t width,
        half* hidden,float limit) {
    const auto i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count*width)return;
    const auto at=i/width*(2*width)+i%width;
    float g=__half2float(gate_up[at]),u=__half2float(gate_up[at+width]);
    if(limit>0) {
        g=fminf(g,limit);u=fminf(limit,fmaxf(-limit,u));
        const float sigmoid=1.f/(1.f+expf(-g));
        hidden[i]=__float2half_rn(__fmul_rn(__fmul_rn(g,sigmoid),u));
    } else {
        const auto sigmoid=__float2half_rn(1.f/(1.f+expf(-g)));
        const auto silu=__float2half_rn(__fmul_rn(g,__half2float(sigmoid)));
        hidden[i]=__float2half_rn(__fmul_rn(__half2float(silu),u));
    }
}
void checked() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
bool grouped_f16_available(int device) {
    int major = 0;
    const auto status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    // WMMA FP16 instructions were introduced with compute capability 7.0.
    return major >= 7;
}
void grouped_decode_f16(ggml_type type,const GemmJob* jobs,int experts,std::uint64_t stride,
    std::int64_t rows,std::int64_t columns,void* output,cudaStream_t stream,std::int64_t batch_stride) {
    if(!experts || !rows || !columns)return;
    if(!batch_stride)batch_stride=rows*columns;
    if(batch_stride<rows*columns)throw std::invalid_argument("GGUF decoded batch stride overlaps matrices");
    const dim3 grid(unsigned((rows*columns+2047)/2048),unsigned(experts));
    switch(type) {
#define TYLOINFER_GGUF_DECODE_GROUP(name) case GGML_TYPE_##name: \
        decode_group<GGML_TYPE_##name><<<grid,256,0,stream>>>(jobs,stride,rows,columns,batch_stride,static_cast<half*>(output));break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_DECODE_GROUP)
#undef TYLOINFER_GGUF_DECODE_GROUP
        default:throw std::invalid_argument("unsupported CUDA GGUF grouped decode");
    }
    checked();
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported[GGML_TYPE_COUNT]{};
        if(!reported[type].exchange(true,std::memory_order_relaxed))
            std::fprintf(stderr,"native_gguf_prefill_decode format=%d values_per_thread=8 four_value_groups=%d rows=%lld columns=%lld\n",
                int(type),int(type>=GGML_TYPE_PTQ1_0 && type<=GGML_TYPE_TQK4?2:0),
                static_cast<long long>(rows),static_cast<long long>(columns));
    }
}
void gate_up_swiglu_f16(const void* gate_up,std::int64_t tokens,std::int64_t width,void* hidden,float limit,cudaStream_t stream) {
    if(!tokens || !width)return;
    activate_gate_up<<<unsigned((tokens*width+255)/256),256,0,stream>>>(
        static_cast<const half*>(gate_up),tokens,width,static_cast<half*>(hidden),limit);checked();
}
void grouped_gather_f16(const GemmJob* jobs,int experts,std::int64_t tokens,std::int64_t columns,
    const void* input,std::int64_t routes,void* output,cudaStream_t stream) {
    if(!experts || !tokens || !columns)return;
    gather_group<<<dim3(unsigned((tokens*columns+255)/256),unsigned(experts)),256,0,stream>>>(
        jobs,tokens,columns,static_cast<const half*>(input),routes,static_cast<half*>(output));checked();
}
void grouped_scatter_f16(const GemmJob* jobs,int experts,std::int64_t tokens,std::int64_t columns,
    const void* input,void* output,cudaStream_t stream) {
    if(!experts || !tokens || !columns)return;
    scatter_group<<<dim3(unsigned((tokens*columns+255)/256),unsigned(experts)),256,0,stream>>>(
        jobs,tokens,columns,static_cast<const half*>(input),static_cast<half*>(output));checked();
}
void scatter_upload(const void* packet, int chunks, std::uint64_t maximum_bytes, cudaStream_t stream) {
    if (!chunks) return;
    scatter<<<dim3(unsigned((maximum_bytes+65535)/65536),unsigned(chunks)),256,0,stream>>>(static_cast<const std::uint8_t*>(packet));
    checked();
}
void grouped_matmul_f16(ggml_type type, const GemmJob* jobs, int count, std::int64_t max_tokens,
    std::uint64_t stride, std::int64_t rows, std::int64_t columns,
    const void* input, std::int64_t routes, bool routed_input, void* output, cudaStream_t stream) {
    if (!count || !max_tokens || !rows) return;
    const dim3 grid(unsigned((rows + 31) / 32), unsigned((max_tokens + 31) / 32), unsigned(count));
    switch (type) {
#define TYLOINFER_GGUF_GEMM(name) case GGML_TYPE_##name: \
        grouped<GGML_TYPE_##name><<<grid, 128, 0, stream>>>(jobs, stride, rows, columns, \
            static_cast<const half*>(input), routes, routed_input, static_cast<half*>(output)); break;
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_GEMM)
#undef TYLOINFER_GGUF_GEMM
        default: throw std::invalid_argument("unsupported CUDA GGUF grouped GEMM");
    }
    checked();
}
void grouped_swiglu_f16(const std::int64_t* slots, std::int64_t count, std::int64_t width,
    const void* gate, const void* up, void* hidden, float limit, cudaStream_t stream) {
    if (!count) return;
    activate<<<unsigned((count * width + 255) / 256), 256, 0, stream>>>(slots, count, width,
        static_cast<const half*>(gate), static_cast<const half*>(up), static_cast<half*>(hidden), limit);
    checked();
}
}
