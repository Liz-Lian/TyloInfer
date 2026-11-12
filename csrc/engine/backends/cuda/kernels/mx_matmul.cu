#include "mx_raw.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include "mx_float_decode.cuh"
namespace {
using tyloi::cuda::mx::decode_e8m0;
using tyloi::cuda::mx::decode_e4m3fn;
using tyloi::cuda::mx::decode_mxfp4_e2m1;
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

template <
    int TILE_M,
    int OUTPUTS_PER_WARP,
    bool CACHE_ACTIVATION,
    typename Output>
__global__ void mxfp8_small_m_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ x,
        Output * __restrict__ y,
        int rows,
        int outputs,
        int width,
        int groups,
        int outputs_per_group,
        int scale_row_shift,int scale_column_shift) {
    extern __shared__ float shared_activation[];
    constexpr int kWarps = 8;
    const int lane = int(threadIdx.x) & 31;
    const int warp = int(threadIdx.x) >> 5;
    const int block_first_output =
        int(blockIdx.x) * kWarps * OUTPUTS_PER_WARP;
    const int first_output =
        block_first_output + warp * OUTPUTS_PER_WARP;
    const int first_row = int(blockIdx.y) * TILE_M;
    const int activation_group = block_first_output / outputs_per_group;
    const bool valid_output = first_output < outputs;
    if (first_row >= rows) {
        return;
    }
    if constexpr (CACHE_ACTIVATION) {
        for (int column = int(threadIdx.x); column < width;
             column += int(blockDim.x)) {
            shared_activation[column] = __half2float(x[
                std::size_t(activation_group) * width + column]);
        }
        __syncthreads();
    }
    if (!valid_output) {
        return;
    }

    float accum[TILE_M][OUTPUTS_PER_WARP];
#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            accum[row][output] = 0.0f;
        }
    }

    const int scale_columns = width >> scale_column_shift;
    const int block_columns=1<<scale_column_shift;
    for (int block_column = 0; block_column < width; block_column += block_columns) {
        float scale[OUTPUTS_PER_WARP];
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            const int global_output = first_output + output;
            scale[output] = global_output < outputs
                ? decode_e8m0(scales[
                    (global_output >> scale_row_shift) * scale_columns +
                    (block_column >> scale_column_shift)])
                : 0.0f;
        }
#pragma unroll
        for (int local_column = lane; local_column < block_columns;
             local_column += 32) {
            const int column = block_column + local_column;
            float activation[TILE_M];
#pragma unroll
            for (int row = 0; row < TILE_M; ++row) {
                const int global_row = first_row + row;
                if constexpr (CACHE_ACTIVATION) {
                    activation[row] = shared_activation[column];
                } else {
                    activation[row] = global_row < rows
                        ? __half2float(x[
                            (std::size_t(global_row) * groups + activation_group) *
                            width + column])
                        : 0.0f;
                }
            }
#pragma unroll
            for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
                const int global_output = first_output + output;
                if (global_output >= outputs) {
                    continue;
                }
                const float weight = decode_e4m3fn(
                    values[(std::size_t)global_output * width + column]) *
                    scale[output];
#pragma unroll
                for (int row = 0; row < TILE_M; ++row) {
                    accum[row][output] += activation[row] * weight;
                }
            }
        }
    }

#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
        const int global_row = first_row + row;
#pragma unroll
        for (int output = 0; output < OUTPUTS_PER_WARP; ++output) {
            const int global_output = first_output + output;
            const float total = warp_sum(accum[row][output]);
            if (lane == 0 && global_row < rows && global_output < outputs) {
                if constexpr (std::is_same_v<Output, float>) {
                    y[(std::size_t)global_row * outputs + global_output] =
                        total;
                } else {
                    y[(std::size_t)global_row * outputs + global_output] =
                        __float2half_rn(total);
                }
            }
        }
    }
}

inline constexpr int mxfp8_values_per_lane=8;

// Keep the same reduction groups for single-token decode and verification.
// Packed loads still share each decoded weight across every token in a tile.
template <int TILE_M, typename Output>
__global__ void mxfp8_packed_small_m_kernel(
        const std::uint8_t* __restrict__ values,
        const std::uint8_t* __restrict__ scales,
        const __half* __restrict__ x, Output* __restrict__ y,
        int rows,int outputs,int width,int groups,int outputs_per_group,
        int scale_row_shift,int scale_column_shift) {
    __shared__ float codebook[256];
    codebook[threadIdx.x]=decode_e4m3fn(std::uint8_t(threadIdx.x));
    __syncthreads();
    const int lane=int(threadIdx.x)&31;
    const int output=int(blockIdx.x)*8+(int(threadIdx.x)>>5);
    if(output>=outputs)return;
    const int group=output/outputs_per_group;
    const auto* weights=reinterpret_cast<const std::uint32_t*>(values+std::size_t(output)*width);
    const auto* scale_row=scales+std::size_t(output>>scale_row_shift)*(width>>scale_column_shift);
    float accum[TILE_M]{};
    constexpr int packed_values=mxfp8_values_per_lane;
    for(int column=lane*packed_values;column<width;column+=32*packed_values) {
        float decoded[packed_values];
#pragma unroll
        for(int word=0;word<packed_values/4;++word) {
            const auto packed=weights[column/4+word];
#pragma unroll
            for(int element=0;element<4;++element)
                decoded[word*4+element]=codebook[std::uint8_t(packed>>(element*8))];
        }
        const float scale=decode_e8m0(scale_row[column>>scale_column_shift]);
#pragma unroll
        for(int row=0;row<TILE_M;++row)if(row<rows) {
            const auto* activation=x+(std::size_t(row)*groups+group)*width+column;
            float partial=0;
#pragma unroll
            for(int pair=0;pair<packed_values/2;++pair) {
                const auto a=__half22float2(*reinterpret_cast<const __half2*>(activation+pair*2));
                partial=fmaf(a.y,decoded[pair*2+1],fmaf(a.x,decoded[pair*2],partial));
            }
            accum[row]=fmaf(partial,scale,accum[row]);
        }
    }
#pragma unroll
    for(int row=0;row<TILE_M;++row) {
        const auto total=warp_sum(accum[row]);
        if(lane==0 && row<rows) {
            if constexpr(std::is_same_v<Output,float>)y[std::size_t(row)*outputs+output]=total;
            else y[std::size_t(row)*outputs+output]=__float2half_rn(total);
        }
    }
}

__global__ void mxfp8_dequant_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        __half * __restrict__ dense,
        int outputs,
        int width,int scale_row_shift,int scale_column_shift) {
    const std::size_t index =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(outputs) * width;
    if (index >= count) {
        return;
    }
    const int output = int(index / width);
    const int column = int(index - std::size_t(output) * width);
    const int scale_columns = width >> scale_column_shift;
    const float scale = decode_e8m0(
        scales[(output >> scale_row_shift) * scale_columns + (column >> scale_column_shift)]);
    dense[index] = __float2half_rn(decode_e4m3fn(values[index]) * scale);
}

__global__ void mxfp8_embedding_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width,int scale_row_shift,int scale_column_shift) {
    const int64_t total = static_cast<int64_t>(count) * width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_index = static_cast<int>(linear / width);
        const int column = static_cast<int>(linear % width);
        const int64_t token = ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const int scale_columns = width >> scale_column_shift;
            const float scale = decode_e8m0(scales[
                (token >> scale_row_shift) * scale_columns + (column >> scale_column_shift)]);
            value = decode_e4m3fn(values[token * width + column]) * scale;
        }
        output[linear] = __float2half_rn(value);
    }
}

// Packed formats share one decoded Tensor Core tile. Padding rotates shared
// banks; output fragments reuse weight storage after the last reader barrier.
template<int Rows>
struct alignas(16) PackedPrefillTile {
    static_assert(Rows>=16 && Rows%16==0);
    static constexpr int k=64,stride=k+8;
    __align__(16) __half activation[Rows*stride];
    __align__(16) unsigned char weights[std::max(64*stride*sizeof(__half),Rows*64*sizeof(float))];
    __device__ __forceinline__ __half* decoded() {return reinterpret_cast<__half*>(weights);}
    __device__ __forceinline__ float* result() {return reinterpret_cast<float*>(weights);}
    template<class Accumulator>
    __device__ __forceinline__ void accumulate(Accumulator& acc,int warp) {
        using namespace nvcuda;
        __syncthreads();
        #pragma unroll
        for(int sub=0;sub<k;sub+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,activation+(warp/4)*16*stride+sub,stride);
            wmma::load_matrix_sync(bf,decoded()+(warp%4)*16*stride+sub,stride);
            wmma::mma_sync(acc,af,bf,acc);
        }
        __syncthreads();
    }
};
using PackedPrefillStorage=PackedPrefillTile<16>;
// Four token tiles reuse each decoded weight tile without expanding weights.
inline constexpr int mxfp8_prefill_rows=tyloi::cuda::mx_raw::prefill_rows;
inline constexpr int mxfp8_prefill_threads=tyloi::cuda::mx_raw::prefill_threads;
using Mxfp8PrefillStorage=PackedPrefillTile<mxfp8_prefill_rows>;

template <typename Output>
__global__ void __launch_bounds__(mxfp8_prefill_threads) mxfp8_matmul_kernel(
        const std::uint8_t* __restrict__ values,const std::uint8_t* __restrict__ scales,
        const __half* __restrict__ input,Output* __restrict__ output,
        int rows,int outputs,int width,int scale_row_shift,int scale_column_shift,
        int groups,int outputs_per_group) {
    using namespace nvcuda;
    __shared__ Mxfp8PrefillStorage tile;
    __shared__ float codebook[256];
    for(int i=threadIdx.x;i<256;i+=mxfp8_prefill_threads)
        codebook[i]=decode_e4m3fn(std::uint8_t(i));
    __syncthreads();
    constexpr int tile_k=Mxfp8PrefillStorage::k,stride=Mxfp8PrefillStorage::stride;
    auto* a=tile.activation;auto* b=tile.decoded();auto* c=tile.result();
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int blocks_per_group=(outputs_per_group-1)/64+1;
    const int group=int(blockIdx.x)/blocks_per_group;
    const int first_row=blockIdx.y*mxfp8_prefill_rows;
    const int first_output=group*outputs_per_group+(int(blockIdx.x)%blocks_per_group)*64;
    const int output_end=(group+1)*outputs_per_group;
    const int scale_columns=width>>scale_column_shift;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.0f);
    for(int k=0;k<width;k+=tile_k) {
        for(int i=threadIdx.x;i<mxfp8_prefill_rows*tile_k;i+=mxfp8_prefill_threads) {
            const int row=first_row+i/tile_k,column=k+i%tile_k;
            a[(i/tile_k)*stride+i%tile_k]=row<rows && column<width?
                input[(std::size_t(row)*groups+group)*width+column]:__float2half_rn(0);
        }
        for(int i=threadIdx.x;i<64*(tile_k/2);i+=mxfp8_prefill_threads) {
            const int neuron=first_output+i/(tile_k/2),column=k+2*(i%(tile_k/2));
            float lo=0,hi=0;
            if(neuron<output_end && column<width) {
                const auto offset=std::size_t(neuron)*width+column;
                const float scale=decode_e8m0(scales[std::size_t(neuron>>scale_row_shift)*scale_columns+(column>>scale_column_shift)]);
                lo=codebook[values[offset]]*scale;
                hi=codebook[values[offset+1]]*scale;
            }
            reinterpret_cast<__half2*>(b)[(i/(tile_k/2))*(stride/2)+i%(tile_k/2)]=__floats2half2_rn(lo,hi);
        }
        tile.accumulate(acc,warp);
    }
    wmma::store_matrix_sync(c+warp*16*16,acc,16,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<16*16;i+=32) {
        const int row=first_row+(warp/4)*16+i/16,neuron=first_output+(warp%4)*16+i%16;
        if(row<rows && neuron<output_end) {
            if constexpr(std::is_same_v<Output,float>)output[std::size_t(row)*outputs+neuron]=c[warp*16*16+i];
            else output[std::size_t(row)*outputs+neuron]=__float2half_rn(c[warp*16*16+i]);
        }
    }
}

__global__ void mxfp4_dequant_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        __half * __restrict__ dense,
        int outputs,
        int width) {
    const std::size_t index =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = std::size_t(outputs) * width;
    if (index >= count) return;
    const int output = int(index / width);
    const int column = int(index - std::size_t(output) * width);
    const std::uint8_t packed = values[
        std::size_t(output) * (width / 2) + column / 2];
    const std::uint8_t code = static_cast<std::uint8_t>(
        (packed >> ((column & 1) * 4)) & 15u);
    const float scale = decode_e8m0(
        scales[std::size_t(output) * (width / 32) + column / 32]);
    dense[index] = __float2half_rn(decode_mxfp4_e2m1(code) * scale);
}

__global__ void mxfp4_embedding_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const int64_t * __restrict__ ids,
        __half * __restrict__ output,
        int count,
        int vocab,
        int width) {
    const int64_t total = static_cast<int64_t>(count) * width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_index = static_cast<int>(linear / width);
        const int column = static_cast<int>(linear % width);
        const int64_t token = ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocab) {
            const std::uint8_t byte = values[
                token * (width / 2) + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (byte >> ((column & 1) * 4)) & 15u);
            value = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[token * (width / 32) + column / 32]);
        }
        output[linear] = __float2half_rn(value);
    }
}

__global__ void __launch_bounds__(256) mxfp4_matmul_f16_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        int rows,
        int outputs,
        int width,
        int row_tiles,
        int output_tiles) {
    constexpr int kRowsPerTile = 8;
    constexpr int kOutputsPerTile = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int64_t tasks = static_cast<int64_t>(row_tiles) * output_tiles;
    for (int64_t task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int output_tile = static_cast<int>(task % output_tiles);
        const int row_tile = static_cast<int>(task / output_tiles);
        const int neuron = output_tile * kOutputsPerTile + warp;
        if (neuron >= outputs) continue;
        const int first_row = row_tile * kRowsPerTile;
        float accumulators[kRowsPerTile];
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            accumulators[item] = 0.0f;
        }
        const int packed_columns = width / 2;
        const int scale_columns = width / 32;
        for (int column = lane; column < width; column += 32) {
            const std::uint8_t packed = values[
                static_cast<int64_t>(neuron) * packed_columns + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const float weight = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[static_cast<int64_t>(neuron) * scale_columns +
                       column / 32]);
#pragma unroll
            for (int item = 0; item < kRowsPerTile; ++item) {
                const int row = first_row + item;
                if (row < rows) {
                    accumulators[item] = fmaf(
                        __half2float(input[static_cast<int64_t>(row) * width + column]),
                        weight,
                        accumulators[item]);
                }
            }
        }
#pragma unroll
        for (int item = 0; item < kRowsPerTile; ++item) {
            const float value = warp_sum(accumulators[item]);
            const int row = first_row + item;
            if (lane == 0 && row < rows) {
                output[static_cast<int64_t>(row) * outputs + neuron] =
                    __float2half_rn(value);
            }
        }
    }
}

template <bool COMPACT_ROUTE>
__global__ void __launch_bounds__(256) mxfp4_moe_grouped_f16_kernel(
        const std::uint8_t * __restrict__ values,
        const std::uint8_t * __restrict__ scales,
        const __half * __restrict__ input,
        const std::int32_t * __restrict__ ids,
        const std::int32_t * __restrict__ expert_local,
        const std::int32_t * __restrict__ ids_dst,
        const std::int32_t * __restrict__ expert_bounds,
        const std::int32_t * __restrict__ tile_bounds,
        const std::int32_t * __restrict__ tile_experts,
        __half * __restrict__ output,
        int tokens,
        int routes,
        int global_experts,
        int pool_experts,
        int out_per_expert,
        int width,
        int max_tiles,
        int row_tiles,
        bool routed_input) {
    constexpr int kTileM = 8;
    constexpr int kRowsPerBlock = 8;
    const int warp = int(threadIdx.y);
    const int lane = int(threadIdx.x);
    const int pairs = tokens * routes;
    const int scale_columns = width / 32;
    const int packed_columns = width / 2;
    const int64_t max_tasks = COMPACT_ROUTE
        ? static_cast<int64_t>(max_tiles) * row_tiles
        : static_cast<int64_t>(pairs) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        int expert = -1;
        int first = 0;
        int last = 0;
        int single_pair = -1;
        if constexpr (COMPACT_ROUTE) {
            const int tile = static_cast<int>(task / row_tiles);
            if (tile >= tile_bounds[global_experts]) continue;
            expert = tile_experts[tile];
            const int local_tile = tile - tile_bounds[expert];
            first = expert_bounds[expert] + local_tile * kTileM;
            last = min(first + kTileM, expert_bounds[expert + 1]);
        } else {
            single_pair = static_cast<int>(task / row_tiles);
            expert = ids[single_pair];
            first = single_pair;
            last = single_pair + 1;
        }
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned>(local_expert) >=
                static_cast<unsigned>(pool_experts)) {
            continue;
        }
        const int local_row = row_tile * kRowsPerBlock + warp;
        if (local_row >= out_per_expert) continue;
        const int packed_row = local_expert * out_per_expert + local_row;
        float accum[kTileM];
#pragma unroll
        for (int item = 0; item < kTileM; ++item) accum[item] = 0.0f;
        for (int column = lane; column < width; column += 32) {
            const std::uint8_t packed = values[
                static_cast<int64_t>(packed_row) * packed_columns + column / 2];
            const std::uint8_t code = static_cast<std::uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const float weight = decode_mxfp4_e2m1(code) * decode_e8m0(
                scales[static_cast<int64_t>(packed_row) * scale_columns +
                       column / 32]);
#pragma unroll
            for (int item = 0; item < kTileM; ++item) {
                int pair = -1;
                if constexpr (COMPACT_ROUTE) {
                    const int compact = first + item;
                    if (compact >= last) continue;
                    pair = ids_dst[compact];
                } else {
                    if (item != 0) continue;
                    pair = single_pair;
                }
                const int source_row = routed_input ? pair : pair / routes;
                accum[item] = fmaf(
                    __half2float(input[static_cast<int64_t>(source_row) * width + column]),
                    weight,
                    accum[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < kTileM; ++item) {
            float value = warp_sum(accum[item]);
            if (lane != 0) continue;
            int pair = -1;
            if constexpr (COMPACT_ROUTE) {
                const int compact = first + item;
                if (compact >= last) continue;
                pair = ids_dst[compact];
            } else {
                if (item != 0) continue;
                pair = single_pair;
            }
            output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                __float2half_rn(value);
        }
    }
}

// Merge adjacent eight-row tiles from the same expert into one WMMA tile.
// Four warps share its activation and decode only 64x32 weights at a time.
// The remaining WMMA rows are zero, so partial expert groups need no padding
// in the caller and compressed weights never acquire a persistent expansion.
__global__ void __launch_bounds__(128) mxfp4_moe_grouped_tensor_kernel(
        const std::uint8_t* __restrict__ values,
        const std::uint8_t* __restrict__ scales,
        const __half* __restrict__ input,
        const std::int32_t* __restrict__ expert_local,
        const std::int32_t* __restrict__ ids_dst,
        const std::int32_t* __restrict__ expert_bounds,
        const std::int32_t* __restrict__ tile_bounds,
        const std::int32_t* __restrict__ tile_experts,
        __half* __restrict__ output,
        int routes,int global_experts,int pool_experts,
        int outputs,int width,int max_tiles,int output_tiles,bool routed_input) {
    using namespace nvcuda;
    __shared__ PackedPrefillStorage tile_storage;
    constexpr int tile_k=PackedPrefillStorage::k,stride=PackedPrefillStorage::stride;
    auto* a=tile_storage.activation;auto* b=tile_storage.decoded();auto* c=tile_storage.result();
    __shared__ int pair_ids[16];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int64_t tasks=int64_t(max_tiles)*output_tiles;
    for(int64_t task=blockIdx.x;task<tasks;task+=gridDim.x) {
        const int tile=int(task/output_tiles);
        if(tile>=tile_bounds[global_experts])continue;
        const int expert=tile_experts[tile],local=expert_local[expert];
        if(static_cast<unsigned>(local)>=static_cast<unsigned>(pool_experts))continue;
        const int local_tile=tile-tile_bounds[expert];
        if(local_tile&1)continue;
        const int first=expert_bounds[expert]+local_tile*8;
        const int last=min(first+16,expert_bounds[expert+1]);
        const int first_output=int(task%output_tiles)*64;
        if(threadIdx.x<16)pair_ids[threadIdx.x]=first+threadIdx.x<last?
            ids_dst[first+threadIdx.x]:-1;
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> acc;
        wmma::fill_fragment(acc,0.0f);
        for(int k=0;k<width;k+=tile_k) {
            for(int i=threadIdx.x;i<16*tile_k;i+=128) {
                const int pair=pair_ids[i/tile_k],column=k+i%tile_k;
                const int source_row=routed_input?pair:pair/routes;
                a[(i/tile_k)*stride+i%tile_k]=pair>=0 && column<width?
                    input[int64_t(source_row)*width+column]:__float2half_rn(0);
            }
            for(int i=threadIdx.x;i<64*(tile_k/2);i+=128) {
                const int neuron=first_output+i/(tile_k/2),column=k+2*(i%(tile_k/2));
                float lo=0,hi=0;
                if(neuron<outputs && column<width) {
                    const int64_t packed_row=int64_t(local)*outputs+neuron;
                    const auto packed=values[packed_row*(width/2)+column/2];
                    const float scale=decode_e8m0(scales[packed_row*(width/32)+column/32]);
                    lo=decode_mxfp4_e2m1(packed&15u)*scale;
                    hi=decode_mxfp4_e2m1(packed>>4u)*scale;
                }
                reinterpret_cast<__half2*>(b)[(i/(tile_k/2))*(stride/2)+i%(tile_k/2)]=__floats2half2_rn(lo,hi);
            }
            tile_storage.accumulate(acc,warp);
        }
        wmma::store_matrix_sync(c+warp*16*16,acc,16,wmma::mem_row_major);
        __syncwarp();
        for(int i=lane;i<16*16;i+=32) {
            const int pair=pair_ids[i/16],neuron=first_output+warp*16+i%16;
            if(pair>=0 && neuron<outputs)
                output[int64_t(pair)*outputs+neuron]=__float2half_rn(c[warp*16*16+i]);
        }
        __syncthreads();
    }
}

} // namespace
namespace tyloi::cuda::mx_raw {
namespace {
void record_small(const Args& a,int tile) {
    static std::atomic<unsigned> recorded{0};
    if(!(recorded.fetch_or(1u<<a.rows,std::memory_order_relaxed)&(1u<<a.rows)))
        std::fprintf(stderr,"mxfp8_packed_small_m scope=first_launch enabled=1 decode=shared_table values_per_lane=%d rows=%d tile_m=%d width=%d outputs=%d\n",mxfp8_values_per_lane,a.rows,tile,a.width,a.outputs);
}
template<int Tile,class Output>
void small_t(const Args& a,cudaStream_t stream) {
    constexpr int threads=256,warps=threads/32;
    const auto* x=static_cast<const __half*>(a.input);auto* y=static_cast<Output*>(a.output);
    if((reinterpret_cast<std::uintptr_t>(a.values)&3u)==0 && (reinterpret_cast<std::uintptr_t>(x)&3u)==0) {
        record_small(a,Tile);
        mxfp8_packed_small_m_kernel<Tile,Output><<<(a.outputs+warps-1)/warps,threads,0,stream>>>(a.values,a.scales,x,y,a.rows,a.outputs,a.width,a.groups,a.outputs_per_group,a.row_shift,a.column_shift);
        return;
    }
    const dim3 grid(unsigned((a.outputs+warps-1)/warps),1u,1u);
    if constexpr(Tile==1) {
        if(a.outputs>8192) {
            mxfp8_small_m_kernel<Tile,1,true,Output><<<grid,threads,size_t(a.width)*sizeof(float),stream>>>(a.values,a.scales,x,y,a.rows,a.outputs,a.width,a.groups,a.outputs_per_group,a.row_shift,a.column_shift);
            return;
        }
    }
    mxfp8_small_m_kernel<Tile,1,false,Output><<<grid,threads,0,stream>>>(a.values,a.scales,x,y,a.rows,a.outputs,a.width,a.groups,a.outputs_per_group,a.row_shift,a.column_shift);
}
template<class Output> void small_dispatch(const Args& a,cudaStream_t stream) {
    if(a.rows==1)small_t<1,Output>(a,stream);else if(a.rows<=2)small_t<2,Output>(a,stream);else if(a.rows<=4)small_t<4,Output>(a,stream);else small_t<8,Output>(a,stream);
}
template<class Output> void gemm8_t(const Args& a,cudaStream_t stream) {
    mxfp8_matmul_kernel<Output><<<dim3(unsigned(((a.outputs_per_group-1)/64+1)*a.groups),unsigned((a.rows+mxfp8_prefill_rows-1)/mxfp8_prefill_rows)),mxfp8_prefill_threads,0,stream>>>(a.values,a.scales,static_cast<const __half*>(a.input),static_cast<Output*>(a.output),a.rows,a.outputs,a.width,a.row_shift,a.column_shift,a.groups,a.outputs_per_group);
}
}
void dequant8(const Args& a,cudaStream_t s) {
    const std::size_t count=std::size_t(a.outputs)*a.width;
    mxfp8_dequant_kernel<<<int((count+255)/256),256,0,s>>>(a.values,a.scales,static_cast<__half*>(a.output),a.outputs,a.width,a.row_shift,a.column_shift);
}
void embed8(const Args& a,cudaStream_t s) {
    const int64_t total=int64_t(a.rows)*a.width;const int blocks=int(std::min<int64_t>((total+255)/256,4096));
    if(blocks>0)mxfp8_embedding_kernel<<<blocks,256,0,s>>>(a.values,a.scales,a.ids64,static_cast<__half*>(a.output),a.rows,a.outputs,a.width,a.row_shift,a.column_shift);
}
void small8(const Args& a,cudaStream_t s) { if(a.output_float)small_dispatch<float>(a,s);else small_dispatch<__half>(a,s); }
void gemm8(const Args& a,cudaStream_t s) { if(a.output_float)gemm8_t<float>(a,s);else gemm8_t<__half>(a,s); }
void dequant4(const Args& a,cudaStream_t s) {
    const std::size_t count=std::size_t(a.outputs)*a.width;
    mxfp4_dequant_kernel<<<int((count+255)/256),256,0,s>>>(a.values,a.scales,static_cast<__half*>(a.output),a.outputs,a.width);
}
void embed4(const Args& a,cudaStream_t s) {
    const int64_t total=int64_t(a.rows)*a.width;const int blocks=int(std::min<int64_t>((total+255)/256,4096));
    if(blocks>0)mxfp4_embedding_kernel<<<blocks,256,0,s>>>(a.values,a.scales,a.ids64,static_cast<__half*>(a.output),a.rows,a.outputs,a.width);
}
void gemm4(const Args& a,cudaStream_t s) {
    const int row_tiles=(a.rows+7)/8,output_tiles=(a.outputs+7)/8;
    const int blocks=int(std::min<int64_t>(int64_t(row_tiles)*output_tiles,4096));
    mxfp4_matmul_f16_kernel<<<blocks,dim3(32,8),0,s>>>(a.values,a.scales,static_cast<const __half*>(a.input),static_cast<__half*>(a.output),a.rows,a.outputs,a.width,row_tiles,output_tiles);
}
void routed4(const Args& a,cudaStream_t s) {
    const int row_tiles=(a.out_per_expert+7)/8,pairs=a.tokens*a.routes;
    if(a.compact && a.tensor_core) {
        const int output_tiles=(a.out_per_expert+63)/64;
        const int blocks=int(std::min<int64_t>(int64_t(a.max_tiles)*output_tiles,4096));
        mxfp4_moe_grouped_tensor_kernel<<<blocks,128,0,s>>>(a.values,a.scales,static_cast<const __half*>(a.input),a.expert_local,a.ids_dst,a.expert_bounds,a.tile_bounds,a.tile_experts,static_cast<__half*>(a.output),a.routes,a.global_experts,a.pool_experts,a.out_per_expert,a.width,a.max_tiles,output_tiles,a.routed_input);
    } else if(a.compact) {
        const int blocks=int(std::min<int64_t>(int64_t(a.max_tiles)*row_tiles,4096));
        mxfp4_moe_grouped_f16_kernel<true><<<blocks,dim3(32,8),0,s>>>(a.values,a.scales,static_cast<const __half*>(a.input),a.ids,a.expert_local,a.ids_dst,a.expert_bounds,a.tile_bounds,a.tile_experts,static_cast<__half*>(a.output),a.tokens,a.routes,a.global_experts,a.pool_experts,a.out_per_expert,a.width,a.max_tiles,row_tiles,a.routed_input);
    } else {
        const int blocks=int(std::min<int64_t>(int64_t(pairs)*row_tiles,4096));
        mxfp4_moe_grouped_f16_kernel<false><<<blocks,dim3(32,8),0,s>>>(a.values,a.scales,static_cast<const __half*>(a.input),a.ids,a.expert_local,nullptr,nullptr,nullptr,nullptr,static_cast<__half*>(a.output),a.tokens,a.routes,a.global_experts,a.pool_experts,a.out_per_expert,a.width,0,row_tiles,a.routed_input);
    }
}
}
