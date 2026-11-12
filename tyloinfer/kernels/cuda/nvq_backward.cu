#include "nvq_backward_raw.h"
#include "packed_nvq.cuh"
#include "nvq_format.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;
using namespace tyloi::cuda::packed_nvq;
using tyloi::cuda::nvq_format::launch_by_format;
using tyloi::cuda::nvq_format::launch_nepq_by_format;

namespace {
struct RawBuffer {
    void *data;
    int64_t size;
    template<class T> T *data_ptr() const { return reinterpret_cast<T *>(data); }
    int64_t numel() const { return size; }
};

template <int FORMAT, int Rows>
__global__ void __launch_bounds__(32) nvq_backward_quad_partial_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * sub_scale,
        int64_t sub_scale_nbytes,
        const float * neuron_scale,
        const int8_t * codebook,
        const __half * output_gradient,
        __half * partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x);
    const int chunk_linear = static_cast<int>(blockIdx.x) * 32 + lane;
    const int k0 = chunk_linear * 4;
    const bool valid = k0 < K;
    const int group = k0 / kGroupSize;
    const int chunk = (k0 - group * kGroupSize) / 4;
    const int split = static_cast<int>(blockIdx.y);
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, N);
    float accumulators[Rows][4];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int component = 0; component < 4; ++component) {
            accumulators[row][component] = 0.0f;
        }
    }
    for (int output = output0; output < output_end; ++output) {
        float anchor = lane == 0 ? neuron_scale[output] : 0.0f;
        anchor = __shfl_sync(0xffffffffu, anchor, 0);
        uint32_t state = 0;
        int packed = 0;
        float weight_scale = 0.0f;
        if (valid) {
            const int64_t state_index =
                static_cast<int64_t>(output) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    state_index,sub_bits,ng);
            packed = decode_chunk4<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                output, group, chunk, nvec, nsign, ng,
                sign_mode, state);
            weight_scale = format_scale<FORMAT>(anchor, state, codebook);
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row < M
                ? __half2float(output_gradient[
                    static_cast<int64_t>(row) * N + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            const float factor = gradient * weight_scale;
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                const int value = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * component)) & 0xff));
                accumulators[row][component] = fmaf(
                    factor, static_cast<float>(value),
                    accumulators[row][component]);
            }
        }
    }
    if (valid) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row < M) {
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    const int column = k0 + component;
                    if (column < K) {
                        partials[(static_cast<int64_t>(split) * M + row) *
                            K + column] = __float2half_rn(
                                accumulators[row][component]);
                    }
                }
            }
        }
    }
}

template <int FORMAT, int Rows>
void launch_nvq_backward_quad_small_m(
        const RawBuffer & indices,
        const RawBuffer & aux,
        const RawBuffer & sub_scale,
        const RawBuffer & neuron_scale,
        const RawBuffer & codebook,
        const RawBuffer & output_gradient,
        RawBuffer & partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((K + 127) / 128),
        static_cast<unsigned>((N + output_tile - 1) / output_tile));
    nvq_backward_quad_partial_kernel<FORMAT, Rows><<<
        grid, 32, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<__half>()),
            reinterpret_cast<__half *>(partials.data_ptr<__half>()),
            M, N, K, ng, nvec, nsign,
            sub_bits, sign_mode, output_tile);
}

template <int FORMAT, int Rows>
__global__ void __launch_bounds__(32) nvq_backward_vec8_partial_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * sub_scale,
        int64_t sub_scale_nbytes,
        const float * neuron_scale,
        const int8_t * codebook,
        const __half * output_gradient,
        __half * partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x);
    const int segment = static_cast<int>(blockIdx.x) * 32 + lane;
    const int k0 = segment * 8;
    const bool valid = k0 < K;
    const int group = k0 / kGroupSize;
    const int split = static_cast<int>(blockIdx.y);
    const int row0 = static_cast<int>(blockIdx.z) * Rows;
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, N);
    float accumulators[Rows][8];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int component = 0; component < 8; ++component) {
            accumulators[row][component] = 0.0f;
        }
    }
    for (int output = output0; output < output_end; ++output) {
        float anchor = lane == 0 ? neuron_scale[output] : 0.0f;
        anchor = __shfl_sync(0xffffffffu, anchor, 0);
        uint32_t state = 0;
        NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
        float weight_scale = 0.0f;
        if (valid) {
            const int64_t state_index =
                static_cast<int64_t>(output) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    state_index,sub_bits,ng);
            decoded = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                output, segment, group, ng, nvec, nsign,
                sign_mode, state);
            weight_scale = format_scale<FORMAT>(anchor, state, codebook);
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row0 + row < M
                ? __half2float(output_gradient[
                    static_cast<int64_t>(row0 + row) * N + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            const float factor = gradient * weight_scale;
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                accumulators[row][component] = fmaf(
                    factor, static_cast<float>(value),
                    accumulators[row][component]);
            }
        }
    }
    if (valid) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row0 + row < M) {
#pragma unroll
                for (int component = 0; component < 8; ++component) {
                    const int column = k0 + component;
                    if (column < K) {
                        partials[(static_cast<int64_t>(split) * M + row0 + row) *
                            K + column] = __float2half_rn(
                                accumulators[row][component]);
                    }
                }
            }
        }
    }
}

template <int FORMAT, int Rows>
void launch_nvq_backward_vec8_small_m(
        const RawBuffer & indices,
        const RawBuffer & aux,
        const RawBuffer & sub_scale,
        const RawBuffer & neuron_scale,
        const RawBuffer & codebook,
        const RawBuffer & output_gradient,
        RawBuffer & partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((K + 255) / 256),
        static_cast<unsigned>((N + output_tile - 1) / output_tile),
        static_cast<unsigned>((M + Rows - 1) / Rows));
    nvq_backward_vec8_partial_kernel<FORMAT, Rows><<<
        grid, 32, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<__half>()),
            reinterpret_cast<__half *>(partials.data_ptr<__half>()),
            M, N, K, ng, nvec, nsign,
            sub_bits, sign_mode, output_tile);
}

template <int FORMAT>
__global__ void __launch_bounds__(256) nvq_backward_mma_m8_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * sub_scale,
        int64_t sub_scale_nbytes,
        const float * neuron_scale,
        const int8_t * codebook,
        const __half * output_gradient,
        __half * input_gradient,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode) {
    constexpr int kWarps = 8;
    __shared__ __align__(16) __half gradient_tile[kWarps][16 * 16];
    __shared__ __align__(16) __half weight_tile[kWarps][16 * 16];
    __shared__ __align__(16) float result_tile[kWarps][16 * 16];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int column0 = static_cast<int>(blockIdx.x) * 16;
    for (int index = lane; index < 16 * 16; index += 32) {
        gradient_tile[warp][index] = __float2half_rn(0.0f);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int output_group = warp * 128;
         output_group < N;
         output_group += kWarps * 128) {
        for (int output0 = output_group;
             output0 < min(output_group + 128, N);
             output0 += 16) {
            for (int index = lane; index < M * 16; index += 32) {
                const int row = index >> 4;
                const int output = output0 + (index & 15);
                gradient_tile[warp][index] = output < N
                    ? output_gradient[static_cast<int64_t>(row) * N + output]
                    : __float2half_rn(0.0f);
            }
            const int output = output0 + (lane & 15);
            const int half = lane >> 4;
            const int k0 = column0 + half * 8;
            NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
            float weight_scale = 0.0f;
            if (output < N && k0 < K) {
                const int group = k0 / kGroupSize;
                const uint32_t state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    static_cast<int64_t>(output)*ng+group,sub_bits,ng);
                decoded = load_nvq_vec8<FORMAT>(
                    indices, indices_nbytes, aux, aux_nbytes, codebook,
                    output, k0 >> 3, group, ng, nvec, nsign,
                    sign_mode, state);
                weight_scale = format_scale<FORMAT>(
                    neuron_scale[output], state, codebook);
            }
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                const int column = k0 + component;
                weight_tile[warp][
                    (lane & 15) * 16 + half * 8 + component] =
                    decoded.valid && column < K
                    ? __float2half_rn(weight_scale * static_cast<float>(value))
                    : __float2half_rn(0.0f);
            }
            __syncwarp();
            wmma::fragment<
                wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
                    gradient_fragment;
            wmma::fragment<
                wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>
                    weight_fragment;
            wmma::load_matrix_sync(
                gradient_fragment, gradient_tile[warp], 16);
            wmma::load_matrix_sync(
                weight_fragment, weight_tile[warp], 16);
            wmma::mma_sync(
                accumulator, gradient_fragment, weight_fragment, accumulator);
            __syncwarp();
        }
    }
    wmma::store_matrix_sync(
        result_tile[warp], accumulator, 16, wmma::mem_row_major);
    __syncthreads();
    const int index = static_cast<int>(threadIdx.x);
    if (index < M * 16) {
        const int row = index >> 4;
        const int column = column0 + (index & 15);
        if (column < K) {
            float value = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
                value += result_tile[source_warp][index];
            }
            input_gradient[static_cast<int64_t>(row) * K + column] =
                __float2half_rn(value);
        }
    }
}

template <int FORMAT>
void launch_nvq_backward_mma_m8(
        const RawBuffer & indices,
        const RawBuffer & aux,
        const RawBuffer & sub_scale,
        const RawBuffer & neuron_scale,
        const RawBuffer & codebook,
        const RawBuffer & output_gradient,
        RawBuffer & result,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int sub_bits,
        int sign_mode,
        cudaStream_t stream) {
    nvq_backward_mma_m8_kernel<FORMAT><<<
        (K + 15) / 16, 256, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<__half>()),
            reinterpret_cast<__half *>(result.data_ptr<__half>()),
            M, N, K, ng, nvec, nsign, sub_bits, sign_mode);
}

template <int FORMAT>
__global__ void __launch_bounds__(256) nepq_backward_mma_m8_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * state_stream,
        int64_t state_nbytes,
        const float * neuron_scale,
        const int8_t * table_pool,
        const uint8_t * bank_ids,
        const __half * output_gradient,
        __half * input_gradient,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits) {
    constexpr int kWarps = 8;
    __shared__ __align__(16) __half gradient_tile[kWarps][16 * 16];
    __shared__ __align__(16) __half weight_tile[kWarps][16 * 16];
    __shared__ __align__(16) float result_tile[kWarps][16 * 16];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int column0 = static_cast<int>(blockIdx.x) * 16;
    for (int index = lane; index < 16 * 16; index += 32) {
        gradient_tile[warp][index] = __float2half_rn(0.0f);
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int output_group = warp * 128;
         output_group < N;
         output_group += kWarps * 128) {
        for (int output0 = output_group;
             output0 < min(output_group + 128, N);
             output0 += 16) {
            for (int index = lane; index < M * 16; index += 32) {
                const int row = index >> 4;
                const int output = output0 + (index & 15);
                gradient_tile[warp][index] = output < N
                    ? output_gradient[static_cast<int64_t>(row) * N + output]
                    : __float2half_rn(0.0f);
            }
            const int output = output0 + (lane & 15);
            const int half = lane >> 4;
            const int k0 = column0 + half * 8;
            NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
            float weight_scale = 0.0f;
            if (output < N && k0 < K) {
                const int group = k0 / kGroupSize;
                const uint32_t state = load_packed_bits(
                    state_stream,
                    (static_cast<int64_t>(output) * ng + group) * state_bits,
                    state_bits,
                    state_nbytes);
                const int8_t * table = nepq_active_table(
                    table_pool, bank_ids, output, group,
                    nsuper, table_stride);
                decoded = load_nepq_vec8<FORMAT>(
                    indices, indices_nbytes, aux, aux_nbytes, table,
                    output, k0 >> 3, group, ng, nvec, nsign, 0, state);
                weight_scale = format_scale<FORMAT>(
                    neuron_scale[output], state, table);
            }
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                const int column = k0 + component;
                weight_tile[warp][
                    (lane & 15) * 16 + half * 8 + component] =
                    decoded.valid && column < K
                    ? __float2half_rn(weight_scale * static_cast<float>(value))
                    : __float2half_rn(0.0f);
            }
            __syncwarp();
            wmma::fragment<
                wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>
                    gradient_fragment;
            wmma::fragment<
                wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>
                    weight_fragment;
            wmma::load_matrix_sync(
                gradient_fragment, gradient_tile[warp], 16);
            wmma::load_matrix_sync(
                weight_fragment, weight_tile[warp], 16);
            wmma::mma_sync(
                accumulator, gradient_fragment, weight_fragment, accumulator);
            __syncwarp();
        }
    }
    wmma::store_matrix_sync(
        result_tile[warp], accumulator, 16, wmma::mem_row_major);
    __syncthreads();
    const int index = static_cast<int>(threadIdx.x);
    if (index < M * 16) {
        const int row = index >> 4;
        const int column = column0 + (index & 15);
        if (column < K) {
            float value = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
                value += result_tile[source_warp][index];
            }
            input_gradient[static_cast<int64_t>(row) * K + column] =
                __float2half_rn(value);
        }
    }
}

template <int FORMAT>
void launch_nepq_backward_mma_m8(
        const RawBuffer & indices,
        const RawBuffer & aux,
        const RawBuffer & state_stream,
        const RawBuffer & neuron_scale,
        const RawBuffer & table_pool,
        const RawBuffer & bank_ids,
        const RawBuffer & output_gradient,
        RawBuffer & result,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits,
        cudaStream_t stream) {
    nepq_backward_mma_m8_kernel<FORMAT><<<
        (K + 15) / 16, 256, 0, stream>>>(
            indices.data_ptr<uint8_t>(), indices.numel(),
            aux.data_ptr<uint8_t>(), aux.numel(),
            state_stream.data_ptr<uint8_t>(), state_stream.numel(),
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),
            bank_ids.data_ptr<uint8_t>(),
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<__half>()),
            reinterpret_cast<__half *>(result.data_ptr<__half>()),
            M, N, K, ng, nvec, nsign, nsuper, table_stride, state_bits);
}

template <int FORMAT, int Rows>
__global__ void __launch_bounds__(32) nepq_backward_partial_kernel(
        const uint8_t * indices,
        int64_t indices_nbytes,
        const uint8_t * aux,
        int64_t aux_nbytes,
        const uint8_t * state_stream,
        int64_t state_nbytes,
        const float * neuron_scale,
        const int8_t * table_pool,
        const uint8_t * bank_ids,
        const __half * output_gradient,
        __half * partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits,
        int output_tile) {
    const int lane = static_cast<int>(threadIdx.x);
    const int segment = static_cast<int>(blockIdx.x) * 32 + lane;
    const int k0 = segment * 8;
    const bool valid = k0 < K;
    const int group = k0 / kGroupSize;
    const int split = static_cast<int>(blockIdx.y);
    const int row0 = static_cast<int>(blockIdx.z) * Rows;
    const int output0 = split * output_tile;
    const int output_end = min(output0 + output_tile, N);
    float accumulators[Rows][8];
#pragma unroll
    for (int row = 0; row < Rows; ++row) {
#pragma unroll
        for (int component = 0; component < 8; ++component) {
            accumulators[row][component] = 0.0f;
        }
    }
    for (int output = output0; output < output_end; ++output) {
        float anchor = lane == 0 ? neuron_scale[output] : 0.0f;
        anchor = __shfl_sync(0xffffffffu, anchor, 0);
        uint32_t state = 0;
        NvqVec8Values<FORMAT> decoded = {make_int2(0, 0), 0, false};
        float weight_scale = 0.0f;
        if (valid) {
            const int64_t state_index =
                static_cast<int64_t>(output) * ng + group;
            state = load_packed_bits(
                state_stream, state_index * state_bits,
                state_bits, state_nbytes);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, output, group,
                nsuper, table_stride);
            decoded = load_nepq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                output, segment, group, ng, nvec, nsign, 0, state);
            weight_scale = format_scale<FORMAT>(anchor, state, table);
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float gradient = lane == 0 && row0 + row < M
                ? __half2float(output_gradient[
                    static_cast<int64_t>(row0 + row) * N + output])
                : 0.0f;
            gradient = __shfl_sync(0xffffffffu, gradient, 0);
            const float factor = gradient * weight_scale;
#pragma unroll
            for (int component = 0; component < 8; ++component) {
                const int packed = component < 4
                    ? decoded.values.x : decoded.values.y;
                const int raw = static_cast<int>(static_cast<int8_t>(
                    (packed >> (8 * (component & 3))) & 0xff));
                int value = raw;
                if constexpr (FORMAT == kNvq1L) {
                    value = 8 * raw + decoded.delta;
                } else if constexpr (FORMAT == kNvq1S) {
                    value = 32 * raw + 5 * decoded.delta;
                }
                accumulators[row][component] = fmaf(
                    factor, static_cast<float>(value),
                    accumulators[row][component]);
            }
        }
    }
    if (valid) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            if (row0 + row < M) {
#pragma unroll
                for (int component = 0; component < 8; ++component) {
                    const int column = k0 + component;
                    if (column < K) {
                        partials[(static_cast<int64_t>(split) * M + row0 + row) *
                            K + column] = __float2half_rn(
                                accumulators[row][component]);
                    }
                }
            }
        }
    }
}

template <int FORMAT, int Rows>
void launch_nepq_backward_small_m(
        const RawBuffer & indices,
        const RawBuffer & aux,
        const RawBuffer & state_stream,
        const RawBuffer & neuron_scale,
        const RawBuffer & table_pool,
        const RawBuffer & bank_ids,
        const RawBuffer & output_gradient,
        RawBuffer & partials,
        int M,
        int N,
        int K,
        int ng,
        int nvec,
        int nsign,
        int nsuper,
        int table_stride,
        int state_bits,
        int output_tile,
        cudaStream_t stream) {
    const dim3 grid(
        static_cast<unsigned>((K + 255) / 256),
        static_cast<unsigned>((N + output_tile - 1) / output_tile),
        static_cast<unsigned>((M + Rows - 1) / Rows));
    nepq_backward_partial_kernel<FORMAT, Rows><<<grid, 32, 0, stream>>>(
        indices.data_ptr<uint8_t>(), indices.numel(),
        aux.data_ptr<uint8_t>(), aux.numel(),
        state_stream.data_ptr<uint8_t>(), state_stream.numel(),
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),
        bank_ids.data_ptr<uint8_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<__half>()),
        reinterpret_cast<__half *>(partials.data_ptr<__half>()),
        M, N, K, ng, nvec, nsign,
        nsuper, table_stride, state_bits, output_tile);
}

} // namespace

namespace mfq_jit_nvq_backward {
void nvq(const Args &a, bool mma, cudaStream_t stream) {
    RawBuffer indices{const_cast<void *>(a.indices), a.indices_size};
    RawBuffer aux{const_cast<void *>(a.aux), a.aux_size};
    RawBuffer state{const_cast<void *>(a.state), a.state_size};
    RawBuffer scale{const_cast<void *>(a.scale), a.outputs};
    RawBuffer table{const_cast<void *>(a.table), 0};
    RawBuffer gradient{const_cast<void *>(a.gradient), 0};
    RawBuffer output{a.output, 0};
    launch_by_format(a.format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (mma) {
            launch_nvq_backward_mma_m8<F>(indices, aux, state, scale, table, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.state_bits, a.sign_mode, stream);
        } else if (a.rows == 1) {
            launch_nvq_backward_vec8_small_m<F, 1>(indices, aux, state, scale, table, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.state_bits, a.sign_mode, a.output_tile, stream);
        } else if (a.rows <= 2) {
            launch_nvq_backward_vec8_small_m<F, 2>(indices, aux, state, scale, table, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.state_bits, a.sign_mode, a.output_tile, stream);
        } else if (a.rows <= 4) {
            launch_nvq_backward_vec8_small_m<F, 4>(indices, aux, state, scale, table, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.state_bits, a.sign_mode, a.output_tile, stream);
        } else if constexpr (is_d4_format(F)) {
            launch_nvq_backward_quad_small_m<F, 8>(indices, aux, state, scale, table, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.state_bits, a.sign_mode, a.output_tile, stream);
        } else {
            launch_nvq_backward_vec8_small_m<F, 4>(indices, aux, state, scale, table, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.state_bits, a.sign_mode, a.output_tile, stream);
        }
    });
}
void nepq(const Args &a, bool mma, cudaStream_t stream) {
    RawBuffer indices{const_cast<void *>(a.indices), a.indices_size};
    RawBuffer aux{const_cast<void *>(a.aux), a.aux_size};
    RawBuffer state{const_cast<void *>(a.state), a.state_size};
    RawBuffer scale{const_cast<void *>(a.scale), a.outputs};
    RawBuffer table{const_cast<void *>(a.table), 0};
    RawBuffer banks{const_cast<void *>(a.banks), 0};
    RawBuffer gradient{const_cast<void *>(a.gradient), 0};
    RawBuffer output{a.output, 0};
    launch_nepq_by_format(a.format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (mma) {
            launch_nepq_backward_mma_m8<F>(indices, aux, state, scale, table, banks, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.supergroups, a.table_stride, a.state_bits, stream);
        } else if (a.rows == 1) {
            launch_nepq_backward_small_m<F, 1>(indices, aux, state, scale, table, banks, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.supergroups, a.table_stride, a.state_bits, a.output_tile, stream);
        } else if (a.rows <= 2) {
            launch_nepq_backward_small_m<F, 2>(indices, aux, state, scale, table, banks, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.supergroups, a.table_stride, a.state_bits, a.output_tile, stream);
        } else {
            launch_nepq_backward_small_m<F, 4>(indices, aux, state, scale, table, banks, gradient, output,
                a.rows, a.outputs, a.width, a.groups, a.vectors, a.signs, a.supergroups, a.table_stride, a.state_bits, a.output_tile, stream);
        }
    });
}
} // namespace mfq_jit_nvq_backward
