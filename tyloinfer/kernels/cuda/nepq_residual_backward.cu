#include "nepq_residual_backward_raw.h"
#include <cuda_runtime.h>

namespace {
constexpr int kWarpSize = 32;

__global__ void nepq_sparse_residual_backward_input_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    const __half * output_gradient,
    __half * input_gradient,
    int input_rows,
    int rows,
    int width,
    int blocks_per_row,
    int position_bits,
    int block_vectors) {
    extern __shared__ float warp_gradients[];
    const int input_row = static_cast<int>(blockIdx.y);
    const int residual_block = static_cast<int>(blockIdx.x);
    const int values_per_block = block_vectors * 8;
    const int warp_count = static_cast<int>(blockDim.x) / kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int row_group = lane / 8;
    const int component = lane & 7;
    for (int index = static_cast<int>(threadIdx.x);
         index < warp_count * values_per_block;
         index += static_cast<int>(blockDim.x)) {
        warp_gradients[index] = 0.0f;
    }
    __syncthreads();

    const int position_mask = (1 << position_bits) - 1;
    for (int row_base = warp * 4;
         row_base < rows;
         row_base += warp_count * 4) {
        const int row = row_base + row_group;
        float gradient = 0.0f;
        int first_record = -1;
        int second_record = -1;
        if (component == 0 && row < rows) {
            gradient = __half2float(output_gradient[
                static_cast<int64_t>(input_row) * rows + row]);
            const int64_t record_index =
                static_cast<int64_t>(row) * blocks_per_row + residual_block;
            first_record = first[record_index];
            second_record = second[record_index];
        }
        const int source_lane = row_group * 8;
        gradient = __shfl_sync(0xffffffffu, gradient, source_lane);
        first_record = __shfl_sync(0xffffffffu, first_record, source_lane);
        second_record = __shfl_sync(0xffffffffu, second_record, source_lane);
        const int records[2] = {first_record, second_record};
#pragma unroll
        for (int stream = 0; stream < 2; ++stream) {
            const int record = records[stream];
            if (record < 0) continue;
            const int position = record & position_mask;
            const int dictionary_id = record >> position_bits;
            atomicAdd(
                warp_gradients + warp * values_per_block +
                    position * 8 + component,
                gradient * __half2float(dictionary[
                    static_cast<int64_t>(dictionary_id) * 8 + component]));
        }
    }
    __syncthreads();

    for (int index = static_cast<int>(threadIdx.x);
         index < values_per_block;
         index += static_cast<int>(blockDim.x)) {
        const int column = residual_block * values_per_block + index;
        if (column < width) {
            float value = 0.0f;
            for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
                value += warp_gradients[
                    source_warp * values_per_block + index];
            }
            const int64_t logical =
                static_cast<int64_t>(input_row) * width + column;
            input_gradient[logical] = __float2half(
                __half2float(input_gradient[logical]) + value);
        }
    }
}

__global__ void nepq_sparse_residual_backward_transpose_kernel(
    const __half * dictionary,
    const int32_t * transpose_offsets,
    const int32_t * transpose_rows,
    const int16_t * transpose_dictionary,
    const __half * output_gradient,
    const int8_t * rotation_signs,
    __half * input_gradient,
    int input_rows,
    int rows,
    int width,
    int vectors,
    bool fuse_rotation) {
    const int64_t thread =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t group = thread / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int component = static_cast<int>(thread & 7);
    const int record_split = lane / 8;
    const int64_t total_groups = static_cast<int64_t>(input_rows) * vectors;
    if (group >= total_groups) return;

    const int input_row = static_cast<int>(group / vectors);
    const int vector = static_cast<int>(group -
        static_cast<int64_t>(input_row) * vectors);
    const unsigned active = __activemask();
    int begin = lane == 0 ? transpose_offsets[vector] : 0;
    int end = lane == 0 ? transpose_offsets[vector + 1] : 0;
    begin = __shfl_sync(active, begin, 0);
    end = __shfl_sync(active, end, 0);

    float value = 0.0f;
    for (int record = begin + record_split;
         record < end;
         record += 4) {
        const unsigned record_active = __activemask();
        const int source_lane = record_split * 8;
        int output = component == 0 ? transpose_rows[record] : 0;
        int dictionary_id = component == 0
            ? static_cast<int>(transpose_dictionary[record])
            : 0;
        output = __shfl_sync(record_active, output, source_lane);
        dictionary_id = __shfl_sync(
            record_active, dictionary_id, source_lane);
        float gradient = component == 0
            ? __half2float(output_gradient[
                static_cast<int64_t>(input_row) * rows + output])
            : 0.0f;
        gradient = __shfl_sync(record_active, gradient, source_lane);
        value = fmaf(
            gradient,
            __half2float(dictionary[
                static_cast<int64_t>(dictionary_id) * 8 + component]),
            value);
    }

    const unsigned reduce_active = __activemask();
    const float split1 = __shfl_sync(
        reduce_active, value, component + 8);
    const float split2 = __shfl_sync(
        reduce_active, value, component + 16);
    const float split3 = __shfl_sync(
        reduce_active, value, component + 24);
    value += split1 + split2 + split3;
    if (record_split != 0) return;

    const int column = vector * 8 + component;
    const int64_t logical =
        static_cast<int64_t>(input_row) * width + column;
    value += __half2float(input_gradient[logical]);
    if (fuse_rotation) {
        const unsigned rotate_active = __activemask();
#pragma unroll
        for (int stride = 1; stride < 8; stride <<= 1) {
            const float other = __shfl_xor_sync(
                rotate_active, value, stride, 8);
            value = (component & stride) ? other - value : value + other;
        }
        value *= 0.3535533905932738f *
            static_cast<float>(rotation_signs[column]);
    }
    input_gradient[logical] = __float2half(value);
}

}  // namespace


namespace mfq_jit_nepq_residual {
void backward_input(const __half* dictionary, const std::int16_t* first,
    const std::int16_t* second, const __half* output_gradient, __half* input_gradient,
    std::int64_t input_rows, int rows, int width, std::int64_t blocks_per_row, int position_bits,
    int block_vectors, cudaStream_t stream) {
    const int threads = input_rows <= 8 ? 1024 : 256;
    const dim3 blocks(static_cast<unsigned>(blocks_per_row), static_cast<unsigned>(input_rows));
    const size_t shared_bytes = static_cast<size_t>(threads / kWarpSize) * block_vectors * 8 * sizeof(float);
    nepq_sparse_residual_backward_input_kernel<<<blocks, threads, shared_bytes, stream>>>(
        dictionary, first, second, output_gradient, input_gradient, static_cast<int>(input_rows),
        rows, width, static_cast<int>(blocks_per_row), position_bits, block_vectors);
}

void backward_transpose(const __half* dictionary, const std::int32_t* transpose_offsets,
    const std::int32_t* transpose_rows, const std::int16_t* transpose_dictionary,
    const __half* output_gradient, const std::int8_t* rotation_signs, __half* input_gradient,
    std::int64_t input_rows, int rows, int width, std::int64_t vectors, bool fuse_rotation,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int64_t total_threads = static_cast<int64_t>(input_rows) * vectors * kWarpSize;
    const int blocks = static_cast<int>((total_threads + threads - 1) / threads);
    nepq_sparse_residual_backward_transpose_kernel<<<blocks, threads, 0, stream>>>(
        dictionary, transpose_offsets, transpose_rows, transpose_dictionary, output_gradient,
        rotation_signs, input_gradient, static_cast<int>(input_rows), rows, width,
        static_cast<int>(vectors), fuse_rotation);
}
}
