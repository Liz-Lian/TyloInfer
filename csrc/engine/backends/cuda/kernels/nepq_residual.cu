

#include <cuda.h>
#include "nepq_residual_raw.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float residual_dot(
    const __half * input,
    const __half * dictionary,
    int record,
    int position_bits,
    int block_vectors,
    int block,
    int width) {
    if (record < 0) return 0.0f;
    const int position = record & ((1 << position_bits) - 1);
    const int dictionary_id = record >> position_bits;
    const int vector = block * block_vectors + position;
    if (dictionary_id >= 1024 || vector >= width / 8) return 0.0f;
    const __half * source = input + static_cast<int64_t>(vector) * 8;
    const __half * code = dictionary + static_cast<int64_t>(dictionary_id) * 8;
    float value = 0.0f;
#pragma unroll
    for (int component = 0; component < 8; ++component) {
        value = fmaf(
            __half2float(source[component]),
            __half2float(code[component]),
            value);
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void nepq_sparse_residual_matmul_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    const __half * input,
    __half * output,
    int rows,
    int width,
    int input_rows,
    int blocks_per_row,
    int position_bits,
    int block_vectors) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int64_t logical =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const int64_t total = static_cast<int64_t>(input_rows) * rows;
    if (logical >= total) return;
    const int input_row = static_cast<int>(logical / rows);
    const int row = static_cast<int>(logical - static_cast<int64_t>(input_row) * rows);
    const __half * source = input + static_cast<int64_t>(input_row) * width;
    const int16_t * first_row = first + static_cast<int64_t>(row) * blocks_per_row;
    const int16_t * second_row = second + static_cast<int64_t>(row) * blocks_per_row;
    float value = 0.0f;
    for (int block = lane; block < blocks_per_row; block += kWarpSize) {
        value += residual_dot(
            source,
            dictionary,
            first_row[block],
            position_bits,
            block_vectors,
            block,
            width);
        const int record = second_row[block];
        if (record >= 0) {
            value += residual_dot(
                source,
                dictionary,
                record,
                position_bits,
                block_vectors,
                block,
                width);
        }
    }
    value = warp_sum(value);
    if (lane == 0) {
        __half * destination = output + logical;
        *destination = __float2half(__half2float(*destination) + value);
    }
}

__global__ void nepq_sparse_residual_dequant_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    __half * weight,
    int width,
    int blocks_per_row,
    int position_bits,
    int block_vectors,
    int64_t total_blocks) {
    const int64_t logical =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (logical >= total_blocks) return;
    const int row = static_cast<int>(logical / blocks_per_row);
    const int block = static_cast<int>(logical - static_cast<int64_t>(row) * blocks_per_row);
    const int records[2] = {first[logical], second[logical]};
#pragma unroll
    for (int stream = 0; stream < 2; ++stream) {
        const int record = records[stream];
        if (record < 0) continue;
        const int position = record & ((1 << position_bits) - 1);
        const int dictionary_id = record >> position_bits;
        const int column = (block * block_vectors + position) * 8;
        if (dictionary_id >= 1024 || column >= width) continue;
        __half * destination = weight + static_cast<int64_t>(row) * width + column;
        const __half * code = dictionary + static_cast<int64_t>(dictionary_id) * 8;
#pragma unroll
        for (int component = 0; component < 8; ++component) {
            destination[component] = __float2half(
                __half2float(destination[component])
                + __half2float(code[component]));
        }
    }
}

__global__ void nepq_sparse_residual_grouped_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    const __half * input,
    const int32_t * route_ids,
    const int32_t * expert_local,
    __half * output,
    int tokens,
    int routes,
    int out_per_expert,
    int width,
    int blocks_per_row,
    int position_bits,
    int block_vectors,
    bool routed_input,
    bool mapped_experts,
    int global_experts,
    int local_experts) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int64_t logical =
        static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const int64_t total =
        static_cast<int64_t>(tokens) * routes * out_per_expert;
    if (logical >= total) return;
    const int output_row = static_cast<int>(logical / out_per_expert);
    const int out = static_cast<int>(logical - static_cast<int64_t>(output_row) * out_per_expert);
    const int token = output_row / routes;
    const int route = output_row - token * routes;
    const int expert = route_ids[static_cast<int64_t>(token) * routes + route];
    if (expert < 0 ||
            (mapped_experts ? expert >= global_experts
                            : expert >= local_experts)) return;
    const int local = mapped_experts ? expert_local[expert] : expert;
    if (local < 0 || local >= local_experts) return;
    const int row = local * out_per_expert + out;
    const int input_row = routed_input ? output_row : token;
    const __half * source = input + static_cast<int64_t>(input_row) * width;
    const int16_t * first_row = first + static_cast<int64_t>(row) * blocks_per_row;
    const int16_t * second_row = second + static_cast<int64_t>(row) * blocks_per_row;
    float value = 0.0f;
    for (int block = lane; block < blocks_per_row; block += kWarpSize) {
        value += residual_dot(
            source,
            dictionary,
            first_row[block],
            position_bits,
            block_vectors,
            block,
            width);
        const int record = second_row[block];
        if (record >= 0) {
            value += residual_dot(
                source,
                dictionary,
                record,
                position_bits,
                block_vectors,
                block,
                width);
        }
    }
    value = warp_sum(value);
    if (lane == 0) {
        __half * destination = output + logical;
        *destination = __float2half(__half2float(*destination) + value);
    }
}

}  // namespace

namespace tyloi::cuda::nepq_residual_raw {

void matmul(const void* dictionary, const int16_t* first, const int16_t* second,
    const void* input, void* output, int rows, int width, int input_rows,
    int blocks_per_row, int position_bits, int block_vectors, int64_t total,
    cudaStream_t stream) {
    nepq_sparse_residual_matmul_kernel<<<(total + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kWarpSize * kWarpsPerBlock, 0, stream>>>(static_cast<const __half*>(dictionary),
        first, second, static_cast<const __half*>(input), static_cast<__half*>(output),
        rows, width, input_rows, blocks_per_row, position_bits, block_vectors);
}

void dequant(const void* dictionary, const int16_t* first, const int16_t* second,
    void* weight, int width, int blocks_per_row, int position_bits,
    int block_vectors, int64_t total, cudaStream_t stream) {
    nepq_sparse_residual_dequant_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const __half*>(dictionary), first, second, static_cast<__half*>(weight),
        width, blocks_per_row, position_bits, block_vectors, total);
}

void grouped(const void* dictionary, const int16_t* first, const int16_t* second,
    const void* input, const int32_t* route_ids, const int32_t* expert_local,
    void* output, int tokens, int routes, int out_per_expert, int width,
    int blocks_per_row, int position_bits, int block_vectors, bool routed_input,
    bool mapped_experts, int global_experts, int local_experts, int64_t total,
    cudaStream_t stream) {
    nepq_sparse_residual_grouped_kernel<<<(total + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kWarpSize * kWarpsPerBlock, 0, stream>>>(static_cast<const __half*>(dictionary),
        first, second, static_cast<const __half*>(input), route_ids, expert_local,
        static_cast<__half*>(output), tokens, routes, out_per_expert, width,
        blocks_per_row, position_bits, block_vectors, routed_input, mapped_experts,
        global_experts, local_experts);
}

}  // namespace tyloi::cuda::nepq_residual_raw
