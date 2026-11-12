#include "mxfp4_sq_backward_raw.h"
#include "mxfp4_sq_device.cuh"

namespace {
using namespace tyloi::cuda::sq_device;
template<int Rows>
__global__ void __launch_bounds__(128) sq_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        tyloi::sq::Layout q,
        int rows,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int block = static_cast<int>(blockIdx.x) * 4 + warp;
    const int column = block * 32 + lane;
    if (column >= q.width) return;

    const auto* symbols = blob + q.symbols;
    const int split = static_cast<int>(blockIdx.y);
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, q.outputs);
    float accumulators[Rows] = {};
    for (int output = output0; output < output_end; ++output) {
        const int bits = row_q[output];
        const auto auxiliary = std::size_t(row_auxiliary[output]);
        const auto* row_symbols =
            symbols + std::size_t(row_symbol_byte_offsets[output]);
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
                state_value =
                    (read_bits(blob + q.palettes, state, 5) << 8) |
                    (q.base + read_bits(blob + q.scales, state, 2));
            }
        }
        state_value = __shfl_sync(0xffffffffu, state_value, 0);
        const auto symbol = read_bits(row_symbols, column, bits);
        const float weight = bits == 4
            ? decode_native(symbol, state_value)
            : decode_value(
                state_value >> 8,
                symbol,
                state_value & 255,
                bits);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row < rows
                ? __half2float(output_gradient[
                    std::size_t(row) * q.outputs + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            accumulators[row] = fmaf(gradient, weight, accumulators[row]);
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
        if (row < rows) {
            partials[(std::size_t(split) * rows + row) * q.width + column] =
                accumulators[row];
        }
    }
}
template<int Rows>
void launch_backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        tyloi::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((q.width + 127) / 128),
        static_cast<unsigned>((q.outputs + output_tile - 1) / output_tile));
    sq_backward_matrix<Rows><<<
        grid, 128, 0, stream>>>(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        output_gradient, partials, q, rows, output_tile);
}

}

void mfq_jit_sq4::backward_matrix(
        const std::uint8_t* blob,
        const std::uint8_t* row_q,
        const std::int32_t* row_symbol_byte_offsets,
        const std::int32_t* row_auxiliary,
        const __half* output_gradient,
        float* partials,
        tyloi::sq::Layout q,
        int rows,
        int output_tile,
        cudaStream_t stream) {
    if (rows == 1) {
        launch_backward_matrix<1>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 2) {
        launch_backward_matrix<2>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else if (rows <= 4) {
        launch_backward_matrix<4>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    } else {
        launch_backward_matrix<8>(
            blob, row_q, row_symbol_byte_offsets, row_auxiliary,
            output_gradient, partials, q, rows, output_tile, stream);
    }
}
