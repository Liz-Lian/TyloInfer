#include "mxfp4_sq_raw.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include "mxfp4_sq_device.cuh"
namespace {
using namespace tyloi::cuda::sq_device;
struct SqPackedDecoder {
    const std::uint8_t *blob, *row_q;
    const std::int32_t *offsets, *auxiliary;
    tyloi::sq::Layout q;
    __device__ void load8(int row, int column, float* destination, int outputs, int width) const {
        if (row >= outputs || column >= width) {
#pragma unroll
            for (int i = 0; i < 8; ++i) destination[i] = 0;
            return;
        }
        const int bits = row_q[row];
        const auto aux = static_cast<std::size_t>(auxiliary[row]);
        const auto* symbols = blob + q.symbols + offsets[row];
        const auto block = column / 32;
        unsigned exponent, palette = 0;
        if (bits == 4) exponent = blob[q.native_scales + aux * (q.width / 32) + block];
        else {
            const auto selector = aux * (q.width / 32) + block;
            const auto state = aux * 8 + block_tag(symbols, blob + q.selectors, block, selector, bits);
            exponent = q.base + read_bits(blob + q.scales, state, 2);
            palette = read_bits(blob + q.palettes, state, 5);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const auto symbol = read_bits(symbols, column + i, bits);
            destination[i] = bits == 4 ? decode_native(symbol, exponent) : decode_value(palette, symbol, exponent, bits);
        }
    }

};

// Decode each weight tile once for 32 activation rows. No dense weight buffer
// is allocated. TF32 keeps MXFP4's E8M0 exponent range (including weights
// outside FP16) using exact TF32 representations of its E2M1 mantissas.
__global__ void sq_packed_gemm(
    SqPackedDecoder decode, const __half* input, __half* output,
    int rows, int outputs, int width) {
    constexpr int tile_m = 32, tile_n = 64, tile_k = 32, stride = 40;
    constexpr int warps_n = tile_n / 16, warps = tile_m * tile_n / 256;
    constexpr int mma_k = 8;
    using Operand = nvcuda::wmma::precision::tf32;
    __shared__ __align__(32) float a[tile_m * stride];
    __shared__ __align__(32) float b[tile_n * stride];
    __shared__ __align__(32) float result[warps * 256];
    const int warp = threadIdx.x / 32;
    const int row_begin = blockIdx.y * tile_m;
    const int output_begin = blockIdx.x * tile_n;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, mma_k, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);
    for (int begin = 0; begin < width; begin += tile_k) {
        for (int i = threadIdx.x; i < tile_m * tile_k; i += blockDim.x) {
            const int m = i / tile_k, k = i % tile_k;
            const float value = row_begin + m < rows && begin + k < width
                ? __half2float(input[static_cast<std::size_t>(row_begin + m) * width + begin + k]) : 0.0f;
            a[m * stride + k] = value;
        }
        for (int i = threadIdx.x; i < tile_n * tile_k / 8; i += blockDim.x) {
            const int n = i / (tile_k / 8), k = (i % (tile_k / 8)) * 8;
            decode.load8(output_begin + n, begin + k, b + n * stride + k, outputs, width);
        }
        __syncthreads();
        for (int k = 0; k < tile_k; k += mma_k) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, mma_k, Operand,
                nvcuda::wmma::row_major> fa;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, mma_k, Operand,
                nvcuda::wmma::col_major> fb;
            nvcuda::wmma::load_matrix_sync(fa, a + (warp / warps_n) * 16 * stride + k, stride);
            nvcuda::wmma::load_matrix_sync(fb, b + (warp % warps_n) * 16 * stride + k, stride);
            nvcuda::wmma::mma_sync(acc, fa, fb, acc);
        }
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(result + warp * 256, acc, 16, nvcuda::wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < tile_m * tile_n; i += blockDim.x) {
        const int m = i / tile_n, n = i % tile_n;
        if (row_begin + m < rows && output_begin + n < outputs) {
            const int owner = (m / 16) * warps_n + n / 16;
            output[static_cast<std::size_t>(row_begin + m) * outputs + output_begin + n] =
                __float2half_rn(result[owner * 256 + (m % 16) * 16 + n % 16]);
        }
    }
}


template<typename T> __device__ __forceinline__ float as_float(T x) { return float(x); }
template<> __device__ __forceinline__ float as_float(__half x) { return __half2float(x); }
template<typename T> __device__ __forceinline__ T from_float(float x) { return T(x); }
template<> __device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }

template<typename T>
__global__ void sq_dequant(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        T* out,
        tyloi::sq::Layout q) {
    const auto* symbols = blob + q.symbols;
    const std::size_t count = std::size_t(q.outputs) * q.width;
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += std::size_t(gridDim.x) * blockDim.x) {
        const auto output = index / q.width;
        const auto column = index - output * q.width;
        const int bits = row_q[output];
        const auto auxiliary = std::size_t(row_auxiliary[output]);
        const auto* row_symbols =
            symbols + std::size_t(row_symbol_byte_offsets[output]);
        const auto block = column / 32;
        const auto symbol = read_bits(row_symbols, column, bits);
        if (bits == 4) {
            const auto exponent = blob[
                q.native_scales + auxiliary * (q.width / 32) + block];
            out[index] = from_float<T>(decode_native(symbol, exponent));
        } else {
            const auto selector_index =
                auxiliary * (q.width / 32) + block;
            const auto state = auxiliary * 8 + block_tag(
                row_symbols, blob + q.selectors, block, selector_index, bits);
            const auto exponent = q.base + read_bits(blob + q.scales, state, 2);
            out[index] = from_float<T>(decode_value(
                read_bits(blob + q.palettes, state, 5),
                symbol, exponent, bits));
        }
    }
}

// One warp per output, coalesced K loads, decode reuse across TILE_M rows.
// All batch sizes use packed weights; M=1..6 have exact tile specializations.
template<int TILE_M, typename T, bool ROUTED = false>
__global__ void sq_mmq(
                       const std::uint8_t* blob,
                       const std::uint8_t* row_q,
                       const std::int32_t* row_symbol_byte_offsets,
                       const std::int32_t* row_auxiliary,
                       const T* x, T* y,
                       tyloi::sq::Layout q, int rows,
                       const std::int32_t* expert_ids,
                       const std::int32_t* expert_local,
                       int n_experts,
                       int local_experts,
                       int out_per_expert,
                       int routes,
                       bool shared_input) {
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int logical_outputs = ROUTED ? out_per_expert : q.outputs;
    const std::int64_t output_tiles =
        (std::int64_t(logical_outputs) + 3) / 4;
    const std::int64_t row_tiles = (std::int64_t(rows) + TILE_M - 1) / TILE_M;
    const auto* symbols = blob + q.symbols;
    for (std::int64_t task = blockIdx.x; task < output_tiles * row_tiles; task += gridDim.x) {
        const auto logical_output = (task % output_tiles) * 4 + warp;
        if (logical_output >= logical_outputs) continue;
        const auto first_row = (task / output_tiles) * TILE_M;
        int local_expert = 0;
        if constexpr (ROUTED) {
            const int expert = expert_ids[first_row];
            if (expert < 0 || expert >= n_experts) continue;
            local_expert = expert_local[expert];
            if (local_expert < 0 || local_expert >= local_experts) continue;
        }
        const auto output = ROUTED
            ? std::int64_t(local_expert) * out_per_expert + logical_output
            : logical_output;
        const int bits = row_q[output];
        const auto auxiliary = std::size_t(row_auxiliary[output]);
        const auto* row_symbols =
            symbols + std::size_t(row_symbol_byte_offsets[output]);
        float accum[TILE_M] = {};
        for (int column = 0; column < q.width; column += 32) {
            const auto block = std::size_t(column / 32);
            unsigned state_value = 0;
            if (lane == 0) {
                if (bits == 4) {
                    state_value = blob[
                        q.native_scales + auxiliary * (q.width / 32) + block];
                } else {
                    const auto selector_index =
                        auxiliary * (q.width / 32) + block;
                    const auto state = auxiliary * 8 + block_tag(
                        row_symbols, blob + q.selectors,
                        block, selector_index, bits);
                    state_value = (read_bits(blob + q.palettes, state, 5) << 8) |
                                  (q.base + read_bits(blob + q.scales, state, 2));
                }
            }
            state_value = __shfl_sync(0xffffffffu, state_value, 0);
            const auto symbol = read_bits(
                row_symbols, std::size_t(column + lane), bits);
            const float w = bits == 4
                ? decode_native(symbol, state_value)
                : decode_value(
                    state_value >> 8,
                    symbol,
                    state_value & 255,
                    bits);
#pragma unroll
            for (int m = 0; m < TILE_M; ++m)
                if (first_row + m < rows)
                    accum[m] = fmaf(
                        w,
                        as_float(x[((ROUTED && shared_input)
                            ? (first_row + m) / routes
                            : first_row + m) * q.width + column + lane]),
                        accum[m]);
        }
#pragma unroll
        for (int m = 0; m < TILE_M; ++m) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1)
                accum[m] += __shfl_down_sync(0xffffffffu, accum[m], delta);
            if (lane == 0 && first_row + m < rows)
                y[(first_row + m) * logical_outputs + logical_output] =
                    from_float<T>(accum[m]);
        }
    }
}

template<int M, typename T>
void launch_mmq(
                const std::uint8_t* blob,
                const std::uint8_t* row_q,
                const std::int32_t* row_symbol_byte_offsets,
                const std::int32_t* row_auxiliary,
                const T* x, T* y,
                tyloi::sq::Layout q, int rows, cudaStream_t stream) {
    const auto tasks = ((std::int64_t(q.outputs) + 3) / 4) * ((std::int64_t(rows) + M - 1) / M);
    const int blocks = int(std::min<std::int64_t>(tasks, 65535));
    sq_mmq<M, T><<<blocks, 128, 0, stream>>>(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        x, y, q, rows,
        nullptr, nullptr, 0, 0, q.outputs, 1, false);
}

void launch_routed_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* x,
        __half* y,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        tyloi::sq::Layout q,
        int route_count,
        int routes,
        int n_experts,
        int local_experts,
        int out_per_expert,
        bool shared_input,
        cudaStream_t stream) {
    const auto tasks =
        std::int64_t(route_count) * ((std::int64_t(out_per_expert) + 3) / 4);
    const int blocks = int(std::min<std::int64_t>(tasks, 65535));
    sq_mmq<1, __half, true><<<blocks, 128, 0, stream>>>(
        blob,
        row_q,
        row_symbol_byte_offsets,
        row_auxiliary,
        x,
        y,
        q,
        route_count,
        expert_ids,
        expert_local,
        n_experts,
        local_experts,
        out_per_expert,
        routes,
        shared_input);
}

template<typename T>
void dispatch_mmq(
                  const std::uint8_t* blob,
                  const std::uint8_t* row_q,
                  const std::int32_t* row_symbol_byte_offsets,
                  const std::int32_t* row_auxiliary,
                  const T* x, T* y,
                  tyloi::sq::Layout q, int rows, cudaStream_t stream) {
#define TYLOINFER_SQ_M_CASE(M) case M: launch_mmq<M>(blob, row_q, row_symbol_byte_offsets, row_auxiliary, x, y, q, rows, stream); break
    switch (rows) {
        TYLOINFER_SQ_M_CASE(1); TYLOINFER_SQ_M_CASE(2); TYLOINFER_SQ_M_CASE(3);
        TYLOINFER_SQ_M_CASE(4); TYLOINFER_SQ_M_CASE(5); TYLOINFER_SQ_M_CASE(6);
        default: launch_mmq<8>(blob, row_q, row_symbol_byte_offsets, row_auxiliary, x, y, q, rows, stream); break;
    }
#undef TYLOINFER_SQ_M_CASE
}

}
namespace tyloi::cuda::mxfp4_sq_raw {
void dequant(const std::uint8_t* blob,const std::uint8_t* q,const std::int32_t* offsets,const std::int32_t* auxiliary,void* output,sq::Layout layout,bool fp32,cudaStream_t stream) {
    const int blocks=int(std::min<std::int64_t>((std::int64_t(layout.outputs)*layout.width+255)/256,65535));
    if(fp32)sq_dequant<float><<<blocks,256,0,stream>>>(blob,q,offsets,auxiliary,static_cast<float*>(output),layout);
    else sq_dequant<__half><<<blocks,256,0,stream>>>(blob,q,offsets,auxiliary,static_cast<__half*>(output),layout);
}
void matmul(const std::uint8_t* blob,const std::uint8_t* q,const std::int32_t* offsets,const std::int32_t* auxiliary,const void* input,void* output,sq::Layout layout,int rows,NumericType type,bool tensor_core,cudaStream_t stream) {
    if(type==NumericType::f32)dispatch_mmq(blob,q,offsets,auxiliary,static_cast<const float*>(input),static_cast<float*>(output),layout,rows,stream);
    else if(tensor_core)sq_packed_gemm<<<dim3((layout.outputs+63)/64,(rows+31)/32),256,0,stream>>>({blob,q,offsets,auxiliary,layout},static_cast<const __half*>(input),static_cast<__half*>(output),rows,layout.outputs,layout.width);
    else dispatch_mmq(blob,q,offsets,auxiliary,static_cast<const __half*>(input),static_cast<__half*>(output),layout,rows,stream);
}
void routed(const std::uint8_t* blob,const std::uint8_t* q,const std::int32_t* offsets,const std::int32_t* auxiliary,const void* input,void* output,const std::int32_t* ids,const std::int32_t* local,sq::Layout layout,int count,int routes,int experts,int local_experts,int out_per_expert,bool shared,cudaStream_t stream) {
    launch_routed_mmq(blob,q,offsets,auxiliary,static_cast<const __half*>(input),static_cast<__half*>(output),ids,local,layout,count,routes,experts,local_experts,out_per_expert,shared,stream);
}
}
