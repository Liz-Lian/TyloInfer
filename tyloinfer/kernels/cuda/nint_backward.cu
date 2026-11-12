#include "nint_backward_raw.h"
#include "packed_nint.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
using namespace tyloi::cuda::packed_nint;
// One metadata-driven packed input-gradient kernel covers every NINT q/k
// allocation.  Each warp owns four adjacent input columns and reduces over
// output neurons; q is read from row metadata and k is already represented by
// the baked subgroup metadata.
__global__ void __launch_bounds__(128) nint_backward_input_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const __half * __restrict__ output_gradient,
        __half * __restrict__ input_gradient,
        int activation_rows,
        int output_rows,
        int groups,
        int width,
        int group_size) {
    constexpr int warps_per_block = 4;
    constexpr int columns_per_warp = 4;
    constexpr int rows_per_tile = 4;
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int chunks_per_group = (group_size + columns_per_warp - 1) /
        columns_per_warp;
    const int chunk_index = static_cast<int>(blockIdx.x) * warps_per_block +
        warp;
    const int group = chunk_index / chunks_per_group;
    const int group_element =
        (chunk_index - group * chunks_per_group) * columns_per_warp;
    const int column_base = group * group_size + group_element;
    const int first_activation_row =
        static_cast<int>(blockIdx.y) * rows_per_tile;
    if (group >= groups || column_base >= width ||
            first_activation_row >= activation_rows) {
        return;
    }

    float accumulators[rows_per_tile][columns_per_warp] = {};
    for (int output_row = lane; output_row < output_rows; output_row += 32) {
        const int bits = static_cast<int>(row_q_bits[output_row]);
        const uint64_t row_bit_offset = static_cast<uint64_t>(
            row_q_bit_offsets[output_row]);
        const size_t metadata_index =
            static_cast<size_t>(output_row) * groups + group;
        const float scale = neuron_scale[output_row] *
            static_cast<float>(subgroup_scale[metadata_index]);
        const float minimum = neuron_minimum[output_row] *
            static_cast<float>(subgroup_minimum[metadata_index]);
        const int valid_columns = min(
            columns_per_warp,
            min(group_size - group_element, width - column_base));
        uint32_t packed_codes = 0;
        if (valid_columns == columns_per_warp) {
            packed_codes = static_cast<uint32_t>(
                unpack_nint_codes4(
                    bitstream,
                    row_bit_offset +
                        static_cast<uint64_t>(column_base) * bits,
                    bits));
        } else {
#pragma unroll
            for (int component = 0; component < columns_per_warp; ++component) {
                if (component < valid_columns) {
                    packed_codes |= static_cast<uint32_t>(unpack_nint_code(
                        bitstream,
                        row_bit_offset,
                        column_base + component,
                        bits)) << (8 * component);
                }
            }
        }
        float weights[columns_per_warp];
#pragma unroll
        for (int component = 0; component < columns_per_warp; ++component) {
            const int code = (packed_codes >> (8 * component)) & 255;
            weights[component] = scale * static_cast<float>(code) - minimum;
        }
#pragma unroll
        for (int local_row = 0; local_row < rows_per_tile; ++local_row) {
            const int activation_row = first_activation_row + local_row;
            const float gradient = activation_row < activation_rows
                ? __half2float(output_gradient[
                      static_cast<size_t>(activation_row) * output_rows +
                      output_row])
                : 0.0f;
#pragma unroll
            for (int component = 0; component < columns_per_warp; ++component) {
                accumulators[local_row][component] = fmaf(
                    gradient,
                    weights[component],
                    accumulators[local_row][component]);
            }
        }
    }

#pragma unroll
    for (int local_row = 0; local_row < rows_per_tile; ++local_row) {
#pragma unroll
        for (int component = 0; component < columns_per_warp; ++component) {
            float total = accumulators[local_row][component];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffffu, total, offset);
            }
            const int activation_row = first_activation_row + local_row;
            const int column = column_base + component;
            if (lane == 0 && activation_row < activation_rows &&
                    component < group_size - group_element && column < width) {
                input_gradient[
                    static_cast<size_t>(activation_row) * width + column] =
                    __float2half_rn(total);
            }
        }
    }
}

} // namespace


namespace mfq_jit_nint {
void backward(const std::uint8_t* bits,const std::uint8_t* row_bits,const std::int64_t* offsets,
    const std::uint8_t* scales,const std::uint8_t* minimum,const float* neuron_scale,const float* neuron_minimum,
    const void* gradient,void* result,int rows,int outputs,int groups,int width,int group_size,cudaStream_t stream) {
    constexpr int warps=4, tile=4;
    const int chunks=groups*((group_size+3)/4);
    nint_backward_input_kernel<<<dim3((chunks+warps-1)/warps,(rows+tile-1)/tile),dim3(32,warps),0,stream>>>(
        bits,row_bits,offsets,scales,minimum,neuron_scale,neuron_minimum,
        static_cast<const __half*>(gradient),static_cast<__half*>(result),rows,outputs,groups,width,group_size);
}
}
