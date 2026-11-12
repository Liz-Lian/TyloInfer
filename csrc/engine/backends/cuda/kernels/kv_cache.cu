// KV cache writer for decode/prefill plumbing.
// Layout: k/v input [B, H, T, D], cache [B, H, max_seq, D].

#include <cuda_runtime.h>
#include "kv_cache_raw.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int KV_WARP_SIZE = 32;
constexpr int KV_WARPS_PER_BLOCK = 4;
constexpr int KV_BLOCK_SIZE = KV_WARP_SIZE * KV_WARPS_PER_BLOCK;
constexpr size_t KV_ROW_KERNEL_MIN_ROWS = 4096;

template <typename scalar_t, bool Vectorized>
__device__ __forceinline__ void copy_kv_row(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    size_t src,
    size_t dst,
    int D,
    int lane)
{
    if constexpr (Vectorized) {
        constexpr int elements_per_vector = sizeof(uint2) / sizeof(scalar_t);
        const int vectors = D / elements_per_vector;
        const auto* k_vector = reinterpret_cast<const uint2*>(k + src);
        const auto* v_vector = reinterpret_cast<const uint2*>(v + src);
        auto* k_cache_vector = reinterpret_cast<uint2*>(k_cache + dst);
        auto* v_cache_vector = reinterpret_cast<uint2*>(v_cache + dst);
        for (int index = lane; index < vectors; index += KV_WARP_SIZE) {
            k_cache_vector[index] = k_vector[index];
            v_cache_vector[index] = v_vector[index];
        }
    } else {
        for (int d = lane; d < D; d += KV_WARP_SIZE) {
            k_cache[dst + d] = k[src + d];
            v_cache[dst + d] = v[src + d];
        }
    }
}

}  // namespace

template <typename scalar_t>
__global__ void kv_cache_write_flat_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim)
{
    size_t n = (size_t)B * H * T * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        int d = (int)(idx % D);
        size_t t0 = idx / D;
        int t = (int)(t0 % T);
        size_t h0 = t0 / T;
        int h = (int)(h0 % H);
        int b = (int)(h0 / H);
        int64_t p = pos_dim == 1 ? positions[t] : positions[(size_t)b * T + t];
        if (p < 0 || p >= max_seq) {
            continue;
        }
        size_t src = (((size_t)b * H + h) * T + t) * D + d;
        size_t dst = (((size_t)b * H + h) * max_seq + (size_t)p) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}

template <typename scalar_t, bool Vectorized>
__global__ void kv_cache_write_rows_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B,
    int H,
    int T,
    int D,
    int max_seq,
    int pos_dim)
{
    const int warp = threadIdx.x / KV_WARP_SIZE;
    const int lane = threadIdx.x % KV_WARP_SIZE;
    const size_t row =
        static_cast<size_t>(blockIdx.x) * KV_WARPS_PER_BLOCK + warp;
    const size_t rows = static_cast<size_t>(B) * H * T;
    if (row >= rows) return;

    const int t = static_cast<int>(row % T);
    const size_t head_row = row / T;
    const int h = static_cast<int>(head_row % H);
    const int b = static_cast<int>(head_row / H);
    const int64_t p = pos_dim == 1
        ? positions[t]
        : positions[static_cast<size_t>(b) * T + t];
    if (p < 0 || p >= max_seq) return;

    const size_t src = row * D;
    const size_t dst =
        ((static_cast<size_t>(b) * H + h) * max_seq +
         static_cast<size_t>(p)) * D;
    copy_kv_row<scalar_t, Vectorized>(
        k, v, k_cache, v_cache, src, dst, D, lane);
}



template <typename scalar_t>
__global__ void kv_cache_write_ring_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    int64_t position_start,
    int B, int H, int T, int D, int capacity, int source_start)
{
    const int kept = T - source_start;
    const size_t n = (size_t)B * H * kept * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int d = (int)(idx % D);
        const size_t t0 = idx / D;
        const int local_t = (int)(t0 % kept);
        const size_t h0 = t0 / kept;
        const int h = (int)(h0 % H);
        const int b = (int)(h0 / H);
        const int t = source_start + local_t;
        const int slot = (int)((position_start + t) % capacity);
        const size_t src = (((size_t)b * H + h) * T + t) * D + d;
        const size_t dst = (((size_t)b * H + h) * capacity + slot) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}



template <typename scalar_t>
__global__ void kv_cache_write_ring_positions_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int64_t* __restrict__ positions,
    int B, int H, int T, int D, int capacity, int source_start)
{
    const int kept = T - source_start;
    const size_t n = (size_t)B * H * kept * D;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int d = (int)(idx % D);
        const size_t t0 = idx / D;
        const int local_t = (int)(t0 % kept);
        const int t = source_start + local_t;
        const size_t h0 = t0 / kept;
        const int h = (int)(h0 % H);
        const int b = (int)(h0 / H);
        const int slot = (int)(positions[t] % capacity);
        const size_t src = (((size_t)b * H + h) * T + t) * D + d;
        const size_t dst = (((size_t)b * H + h) * capacity + slot) * D + d;
        k_cache[dst] = k[src];
        v_cache[dst] = v[src];
    }
}



namespace tyloi::cuda::kv_cache_raw {
namespace {
void check() { const auto e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(std::string("kv cache launch: ") + cudaGetErrorString(e)); }
template<class T> void write_t(void* kc, void* vc, const void* k, const void* v, const int64_t* p, int B, int H, int N, int D, int cap, int rank, cudaStream_t stream) {
    const size_t rows = size_t(B)*H*N;
    if (rows >= KV_ROW_KERNEL_MIN_ROWS && sizeof(T) < sizeof(float)) {
        constexpr int per = sizeof(uint2)/sizeof(T);
        const auto addresses = reinterpret_cast<std::uintptr_t>(k) | reinterpret_cast<std::uintptr_t>(v) | reinterpret_cast<std::uintptr_t>(kc) | reinterpret_cast<std::uintptr_t>(vc);
        const bool aligned = (addresses & (alignof(uint2)-1)) == 0;
        const int grid = int((rows + KV_WARPS_PER_BLOCK - 1)/KV_WARPS_PER_BLOCK);
        if (D%per == 0 && aligned) kv_cache_write_rows_kernel<T,true><<<grid,KV_BLOCK_SIZE,0,stream>>>(static_cast<const T*>(k),static_cast<const T*>(v),static_cast<T*>(kc),static_cast<T*>(vc),p,B,H,N,D,cap,rank);
        else kv_cache_write_rows_kernel<T,false><<<grid,KV_BLOCK_SIZE,0,stream>>>(static_cast<const T*>(k),static_cast<const T*>(v),static_cast<T*>(kc),static_cast<T*>(vc),p,B,H,N,D,cap,rank);
    } else {
        const size_t elements = rows*D; const int grid = int(std::min<size_t>(4096,(elements+255)/256));
        kv_cache_write_flat_kernel<T><<<grid,256,0,stream>>>(static_cast<const T*>(k),static_cast<const T*>(v),static_cast<T*>(kc),static_cast<T*>(vc),p,B,H,N,D,cap,rank);
    }
}
template<class T> void ring_t(void* kc,void* vc,const void* k,const void* v,const int64_t* p,int64_t start,int B,int H,int N,int D,int cap,cudaStream_t stream) {
    const int first=std::max(0,N-cap); const size_t n=size_t(B)*H*(N-first)*D;
    const int grid=int(std::min<size_t>(4096,(n+255)/256));
    if(p) kv_cache_write_ring_positions_kernel<T><<<grid,256,0,stream>>>(static_cast<const T*>(k),static_cast<const T*>(v),static_cast<T*>(kc),static_cast<T*>(vc),p,B,H,N,D,cap,first);
    else kv_cache_write_ring_kernel<T><<<grid,256,0,stream>>>(static_cast<const T*>(k),static_cast<const T*>(v),static_cast<T*>(kc),static_cast<T*>(vc),start,B,H,N,D,cap,first);
}
}
void write(void* kc,void* vc,const void* k,const void* v,const int64_t* p,int B,int H,int N,int D,int cap,int rank,NumericType type,cudaStream_t stream) {
    if (type==NumericType::f32) write_t<uint32_t>(kc,vc,k,v,p,B,H,N,D,cap,rank,stream);
    else if (type==NumericType::f16 || type==NumericType::bf16) write_t<uint16_t>(kc,vc,k,v,p,B,H,N,D,cap,rank,stream);
    else throw std::invalid_argument("unsupported KV dtype"); check();
}
void ring(void* kc,void* vc,const void* k,const void* v,int64_t start,int B,int H,int N,int D,int cap,NumericType type,cudaStream_t stream) {
    if (type==NumericType::f32) ring_t<uint32_t>(kc,vc,k,v,nullptr,start,B,H,N,D,cap,stream);
    else if (type==NumericType::f16 || type==NumericType::bf16) ring_t<uint16_t>(kc,vc,k,v,nullptr,start,B,H,N,D,cap,stream);
    else throw std::invalid_argument("unsupported KV dtype"); check();
}
void ring_positions(void* kc,void* vc,const void* k,const void* v,const int64_t* p,int B,int H,int N,int D,int cap,NumericType type,cudaStream_t stream) {
    if (type==NumericType::f32) ring_t<uint32_t>(kc,vc,k,v,p,0,B,H,N,D,cap,stream);
    else if (type==NumericType::f16 || type==NumericType::bf16) ring_t<uint16_t>(kc,vc,k,v,p,0,B,H,N,D,cap,stream);
    else throw std::invalid_argument("unsupported KV dtype"); check();
}
}
