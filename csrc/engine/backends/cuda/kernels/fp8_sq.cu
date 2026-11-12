#include "fp8_sq_raw.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>
namespace {
using Layout=tyloi::cuda::fp8_sq_raw::Layout;
__device__ __forceinline__ unsigned read_bits(
        const std::uint8_t* data, std::size_t index, int bits) {
    const auto bit = index * static_cast<std::size_t>(bits);
    const unsigned shift = static_cast<unsigned>(bit & 7);
    unsigned value = data[bit >> 3];
    if (shift + static_cast<unsigned>(bits) > 8) {
        value |= static_cast<unsigned>(data[(bit >> 3) + 1]) << 8;
    }
    return (value >> shift) & ((1u << bits) - 1u);
}

__device__ __forceinline__ float decode_e4m3fn(std::uint8_t raw) {
    const unsigned magnitude = static_cast<unsigned>(raw & 0x7fu);
    const unsigned exponent = magnitude >> 3u;
    const unsigned mantissa = magnitude & 7u;
    const float value = exponent == 0u
        ? static_cast<float>(mantissa) * 0.001953125f
        : __uint_as_float(((exponent + 120u) << 23u) | (mantissa << 20u));
    return (raw & 0x80u) == 0u ? value : -value;
}

__device__ __forceinline__ float decode_e8m0(std::uint8_t raw) {
    return raw == 0u
        ? __uint_as_float(0x00400000u)
        : __uint_as_float(static_cast<unsigned>(raw) << 23u);
}

__device__ __forceinline__ std::uint16_t load_u16(
        const std::uint8_t* source) {
    return static_cast<std::uint16_t>(source[0]) |
        static_cast<std::uint16_t>(source[1]) << 8;
}

__device__ __forceinline__ std::uint32_t load_u32(
        const std::uint8_t* source) {
    return static_cast<std::uint32_t>(source[0]) |
        static_cast<std::uint32_t>(source[1]) << 8 |
        static_cast<std::uint32_t>(source[2]) << 16 |
        static_cast<std::uint32_t>(source[3]) << 24;
}

__device__ __forceinline__ float decode_fp8_128_scale(
        const std::uint8_t* source, int scale_kind) {
    if (scale_kind == 2) {
        return __uint_as_float(static_cast<unsigned>(load_u16(source)) << 16);
    }
    if (scale_kind == 3) {
        const auto raw = load_u16(source);
        const unsigned exponent = (raw >> 10u) & 31u;
        const unsigned mantissa = raw & 1023u;
        if (exponent == 0u) {
            return static_cast<float>(mantissa) * 0x1p-24f;
        }
        return (1.0f + static_cast<float>(mantissa) / 1024.0f) *
            __uint_as_float((exponent + 112u) << 23u);
    }
    return __uint_as_float(load_u32(source));
}

__device__ __forceinline__ std::uint8_t decode_code(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const Layout& layout,
        int output,
        int column) {
    const int bits = static_cast<int>(row_q[output]);
    const auto* row_symbols = blob + layout.symbols +
        static_cast<std::size_t>(row_symbol_byte_offsets[output]);
    if (bits == 8) {
        return row_symbols[column];
    }
    const auto symbol = read_bits(
        row_symbols, static_cast<std::size_t>(column), bits);
    return blob[layout.palettes + (std::size_t{1} << bits) - 2 + symbol];
}

template <bool MXFP8>
__device__ __forceinline__ float decode_scale(
        const std::uint8_t* blob,
        const Layout& layout,
        int output,
        int column) {
    const auto scale_index =
        static_cast<std::size_t>(output / layout.block_rows) *
            layout.scale_columns +
        column / layout.block_columns;
    if constexpr (MXFP8) {
        return decode_e8m0(blob[layout.scales + scale_index]);
    } else {
        const int itemsize = layout.scale_kind == 4 ? 4 : 2;
        return decode_fp8_128_scale(
            blob + layout.scales + scale_index * itemsize,
            layout.scale_kind);
    }
}

template <bool MXFP8>
__device__ __forceinline__ float decode_weight(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const Layout& layout,
        int output,
        int column) {
    return decode_e4m3fn(decode_code(
        blob, row_q, row_symbol_byte_offsets, layout, output, column)) *
        decode_scale<MXFP8>(blob, layout, output, column);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ float as_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float as_float(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
}

template <bool MXFP8, typename T>
__device__ __forceinline__ void dequant_body(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    const auto count = static_cast<std::size_t>(layout.outputs) * layout.width;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / layout.width);
        const int column = static_cast<int>(index % layout.width);
        output[index] = from_float<T>(decode_weight<MXFP8>(
            blob, row_q, row_symbol_byte_offsets, layout, row, column));
    }
}

template <typename T>
__global__ void mxfp8_sq_dequant_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    dequant_body<true>(blob, row_q, row_symbol_byte_offsets, output, layout);
}

template <typename T>
__global__ void fp8_128_sq_dequant_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        T* output,
        Layout layout) {
    dequant_body<false>(blob, row_q, row_symbol_byte_offsets, output, layout);
}

template <bool MXFP8, int TILE_M, typename T, bool ROUTED>
__device__ __forceinline__ void mmq_body(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int logical_outputs = ROUTED ? out_per_expert : layout.outputs;
    const auto output_tiles = (static_cast<std::int64_t>(logical_outputs) + 3) / 4;
    const auto row_tiles = (static_cast<std::int64_t>(rows) + TILE_M - 1) / TILE_M;
    for (std::int64_t task = blockIdx.x;
         task < output_tiles * row_tiles;
         task += gridDim.x) {
        const int logical_output = static_cast<int>(task % output_tiles) * 4 + warp;
        if (logical_output >= logical_outputs) continue;
        const int first_row = static_cast<int>(task / output_tiles) * TILE_M;
        int local_expert = 0;
        if constexpr (ROUTED) {
            const int expert = expert_ids[first_row];
            if (expert < 0 || expert >= global_experts) continue;
            local_expert = expert_local[expert];
            if (local_expert < 0 || local_expert >= local_experts) continue;
        }
        const int weight_row = ROUTED
            ? local_expert * out_per_expert + logical_output
            : logical_output;
        float accumulator[TILE_M] = {};
        for (int column = lane; column < layout.width; column += 32) {
            const float weight = decode_weight<MXFP8>(
                blob,
                row_q,
                row_symbol_byte_offsets,
                layout,
                weight_row,
                column);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                if (first_row + item >= rows) continue;
                const int source_row = ROUTED && shared_input
                    ? (first_row + item) / routes
                    : first_row + item;
                accumulator[item] = fmaf(
                    weight,
                    as_float(input[
                        static_cast<std::size_t>(source_row) * layout.width + column]),
                    accumulator[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const float value = warp_sum(accumulator[item]);
            if (lane == 0 && first_row + item < rows) {
                output[
                    static_cast<std::size_t>(first_row + item) * logical_outputs +
                    logical_output] = from_float<T>(value);
            }
        }
    }
}

template <int TILE_M, typename T, bool ROUTED = false>
__global__ void mxfp8_sq_mmq_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    mmq_body<true, TILE_M, T, ROUTED>(
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
        expert_ids, expert_local, global_experts, local_experts,
        out_per_expert, routes, shared_input);
}

template <int TILE_M, typename T, bool ROUTED = false>
__global__ void fp8_128_sq_mmq_kernel(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        int global_experts,
        int local_experts,
        int out_per_expert,
        int routes,
        bool shared_input) {
    mmq_body<false, TILE_M, T, ROUTED>(
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
        expert_ids, expert_local, global_experts, local_experts,
        out_per_expert, routes, shared_input);
}

template <bool MXFP8, int TILE_M, typename T>
void launch_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        cudaStream_t stream) {
    const auto tasks =
        (static_cast<std::int64_t>(layout.outputs) + 3) / 4 *
        ((static_cast<std::int64_t>(rows) + TILE_M - 1) / TILE_M);
    const int blocks = static_cast<int>(std::min<std::int64_t>(tasks, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_mmq_kernel<TILE_M><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
            nullptr, nullptr, 0, 0, layout.outputs, 1, false);
    } else {
        fp8_128_sq_mmq_kernel<TILE_M><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout, rows,
            nullptr, nullptr, 0, 0, layout.outputs, 1, false);
    }
}

template <bool MXFP8, typename T>
void dispatch_mmq(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const T* input,
        T* output,
        Layout layout,
        int rows,
        cudaStream_t stream) {
#define TYLOINFER_FP8_SQ_M_CASE(M) \
    case M: launch_mmq<MXFP8, M>( \
        blob, row_q, row_symbol_byte_offsets, input, output, layout, rows, stream); break
    switch (rows) {
        TYLOINFER_FP8_SQ_M_CASE(1);
        TYLOINFER_FP8_SQ_M_CASE(2);
        TYLOINFER_FP8_SQ_M_CASE(3);
        TYLOINFER_FP8_SQ_M_CASE(4);
        TYLOINFER_FP8_SQ_M_CASE(5);
        TYLOINFER_FP8_SQ_M_CASE(6);
        default:
            launch_mmq<MXFP8, 8>(
                blob, row_q, row_symbol_byte_offsets,
                input, output, layout, rows, stream);
            break;
    }
#undef TYLOINFER_FP8_SQ_M_CASE
}

template <bool MXFP8>
void launch_routed(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const __half* input,
        __half* output,
        const std::int32_t* expert_ids,
        const std::int32_t* expert_local,
        Layout layout,
        int route_count,
        int routes,
        int global_experts,
        int local_experts,
        int out_per_expert,
        bool shared_input,
        cudaStream_t stream) {
    const auto tasks = static_cast<std::int64_t>(route_count) *
        ((static_cast<std::int64_t>(out_per_expert) + 3) / 4);
    const int blocks = static_cast<int>(std::min<std::int64_t>(tasks, 65535));
    if constexpr (MXFP8) {
        mxfp8_sq_mmq_kernel<1, __half, true><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout,
            route_count, expert_ids, expert_local, global_experts,
            local_experts, out_per_expert, routes, shared_input);
    } else {
        fp8_128_sq_mmq_kernel<1, __half, true><<<blocks, 128, 0, stream>>>(
            blob, row_q, row_symbol_byte_offsets, input, output, layout,
            route_count, expert_ids, expert_local, global_experts,
            local_experts, out_per_expert, routes, shared_input);
    }
}

}
namespace tyloi::cuda::fp8_sq_raw {
template<bool MX, class T>
void decode_t(const std::uint8_t* blob,const std::uint8_t* q,const std::int32_t* offsets,void* output,Layout layout,cudaStream_t stream) {
    const int blocks=int(std::min<std::int64_t>((std::int64_t(layout.outputs)*layout.width+255)/256,65535));
    if constexpr(MX)mxfp8_sq_dequant_kernel<T><<<blocks,256,0,stream>>>(blob,q,offsets,static_cast<T*>(output),layout);
    else fp8_128_sq_dequant_kernel<T><<<blocks,256,0,stream>>>(blob,q,offsets,static_cast<T*>(output),layout);
}
void dequant(const std::uint8_t* blob,const std::uint8_t* q,const std::int32_t* offsets,void* output,Layout layout,bool mx,bool fp32,cudaStream_t stream) {
    if(mx) { if(fp32)decode_t<true,float>(blob,q,offsets,output,layout,stream);else decode_t<true,__half>(blob,q,offsets,output,layout,stream); }
    else { if(fp32)decode_t<false,float>(blob,q,offsets,output,layout,stream);else decode_t<false,__half>(blob,q,offsets,output,layout,stream); }
}
void matmul(const std::uint8_t* blob,const std::uint8_t* q,const std::int32_t* offsets,const void* input,void* output,Layout layout,int rows,bool mx,NumericType type,cudaStream_t stream) {
    if(mx) { if(type==NumericType::f32)dispatch_mmq<true>(blob,q,offsets,static_cast<const float*>(input),static_cast<float*>(output),layout,rows,stream); else dispatch_mmq<true>(blob,q,offsets,static_cast<const __half*>(input),static_cast<__half*>(output),layout,rows,stream); }
    else { if(type==NumericType::f32)dispatch_mmq<false>(blob,q,offsets,static_cast<const float*>(input),static_cast<float*>(output),layout,rows,stream); else dispatch_mmq<false>(blob,q,offsets,static_cast<const __half*>(input),static_cast<__half*>(output),layout,rows,stream); }
}
void routed(const std::uint8_t* blob,const std::uint8_t* q,const std::int32_t* offsets,const void* input,void* output,const std::int32_t* ids,const std::int32_t* local,Layout layout,int count,int routes,int experts,int local_experts,int out_per_expert,bool shared,bool mx,cudaStream_t stream) {
    if(mx)launch_routed<true>(blob,q,offsets,static_cast<const __half*>(input),static_cast<__half*>(output),ids,local,layout,count,routes,experts,local_experts,out_per_expert,shared,stream);
    else launch_routed<false>(blob,q,offsets,static_cast<const __half*>(input),static_cast<__half*>(output),ids,local,layout,count,routes,experts,local_experts,out_per_expert,shared,stream);
}
cublasStatus_t gemm(cublasHandle_t handle,const void* input,const void* weight,void* output,int rows,int width,int outputs,NumericType type,cudaStream_t stream) {
    (void)stream; // The adapter binds the handle before calling GEMM.
    if(type==NumericType::f16) {
        const __half alpha=__float2half(1.f),beta=__float2half(0.f);
        return cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,outputs,rows,width,&alpha,weight,CUDA_R_16F,width,input,CUDA_R_16F,width,&beta,output,CUDA_R_16F,outputs,CUBLAS_COMPUTE_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    const float alpha=1.f,beta=0.f;
    return cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,outputs,rows,width,&alpha,weight,CUDA_R_32F,width,input,CUDA_R_32F,width,&beta,output,CUDA_R_32F,outputs,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
}
}
