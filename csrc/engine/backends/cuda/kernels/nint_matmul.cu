#include "env.h"
// Canonical CUDA execution for NINT.
//
// Every ordinary NINT tensor is normalized at the loader boundary to one
// row-major bitstream plus per-neuron q metadata.  Uniform presets and
// adaptive q/k allocations therefore execute through exactly the same
// kernels.  Small batched decode can stage decoded tiles in shared memory
// for Tensor Core projection; large prefill retains its GEMM dispatch.

// NINT8-0 is a separate symmetric format.  It retains one small-M packed
// kernel; larger batches use its row decoder followed by GEMM.


#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "async_copy.cuh"
#include "glu.cuh"

#include "packed_nint.cuh"
#include "nint_route_group.cuh"
#include "nint_short_decode.h"
#include "nint_raw.h"
#include "kernel_dispatch_trace.h"



namespace {


using namespace tyloi::cuda::packed_nint;


using NintGroupQuantizeArgs = tyloi::cuda::nint_raw::GroupQuantize;

__global__ void nint8_one_quantize_reconstruct_kernel(
        const __half * __restrict__ input,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scale_output,
        __half * __restrict__ sum_output,
        __half * __restrict__ reconstructed,
        int rows,
        int width,
        int groups) {
    const int group_index = static_cast<int>(blockIdx.x);
    if (group_index >= rows * groups) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int row = group_index / groups;
    const int group = group_index - row * groups;
    const int column = group * 32 + lane;
    const float value = column < width
        ? __half2float(input[static_cast<size_t>(row) * width + column])
        : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_xor_sync(0xffffffffu, maximum, offset));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    int code = static_cast<int>(roundf(value * inverse));
    code = max(-127, min(127, code));
    quantized[static_cast<size_t>(group_index) * 32 + lane] =
        static_cast<int8_t>(code);

    int sum = code;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    const __half stored_scale = __float2half_rn(scale);
    if (lane == 0) {
        scale_output[group_index] = stored_scale;
        sum_output[group_index] = __float2half_rn(
            static_cast<float>(sum) * scale);
    }
    if (column < width) {
        reconstructed[static_cast<size_t>(row) * width + column] =
            __float2half_rn(
                static_cast<float>(code) * __half2float(stored_scale));
    }
}


template <typename Output>
__device__ __forceinline__ Output nint_decode_store(float value) {
    if constexpr (std::is_same_v<Output, __half>) {
        return __float2half_rn(value);
    } else {
        return value;
    }
}


template <typename Output>
__global__ void nint_decode_rows_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        Output * __restrict__ output,
        int rows,
        int groups,
        int group_size,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int row = static_cast<int>(index / width);
        const int group = column / group_size;
        const size_t metadata = static_cast<size_t>(row) * groups + group;
        const int bits = static_cast<int>(row_q_bits[row]);
        const uint8_t code = unpack_nint_code(
            bitstream,
            static_cast<uint64_t>(row_q_bit_offsets[row]),
            column,
            bits);
        const float scale = neuron_scale[row] *
            static_cast<float>(subgroup_scale[metadata]);
        const float minimum = neuron_minimum[row] *
            static_cast<float>(subgroup_minimum[metadata]);
        output[index] = nint_decode_store<Output>(
            scale * static_cast<float>(code) - minimum);
    }
}


// Runtime group size deliberately avoids a kernel family per quantizer preset.
__device__ __forceinline__ float nint_activation_value(
        const void* input, size_t index, bool fp32) {
    // Match the existing fp16 activation contract without materializing a
    // converted tensor or launching a separate conversion kernel.
    return fp32 ? __half2float(__float2half_rn(static_cast<const float*>(input)[index]))
                : __half2float(static_cast<const __half*>(input)[index]);
}

template<bool CachedSum=false,int InterleavedColumns=0>
__global__ void __launch_bounds__(64) nint_quantize_activation_kernel(
        const void * __restrict__ input,
        const void * __restrict__ gate,
        int8_t * __restrict__ quantized,
        float * __restrict__ scale_output,
        int rows,
        int real_width,
        int padded_width,
        int groups,
        int group_size,
        int activation_mode,
        bool input_fp32,
        bool gate_fp32,
        int* sum_output=nullptr,
        int8_t* interleaved_output=nullptr) {
    const int row = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int column = group * group_size + lane;
    const bool valid = lane < group_size && column < real_width;
    float value = valid
        ? nint_activation_value(input, static_cast<size_t>(row) * real_width + column, input_fp32)
        : 0.0f;
    if (valid && gate != nullptr) {
        const float gate_value = nint_activation_value(
            gate, static_cast<size_t>(row) * real_width + column, gate_fp32);
        const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
        value *= activation_mode == 1 ? sigmoid : gate_value * sigmoid;
    }
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    if (blockDim.x == 32) {
        maximum = __shfl_sync(0xffffffffu, maximum, 0);
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        if (lane == 0) scale_output[static_cast<size_t>(row) * groups + group] = scale;
        const int code = valid ? max(-127, min(127, static_cast<int>(roundf(value / scale)))) : 0;
        if (lane < group_size) {
            if constexpr(InterleavedColumns>0) {
                if constexpr(InterleavedColumns==5 || InterleavedColumns==6) {
                    const size_t index=size_t(group)*7+lane/4;
                    const size_t at=row<4?index*4+row:
                        size_t(groups)*7*4+index*(InterleavedColumns-4)+(row-4);
                    interleaved_output[at*4+lane%4]=static_cast<int8_t>(code);
                }else {
                const size_t at=(size_t(group)*7+lane/4)*InterleavedColumns*4+lane%4;
                interleaved_output[at+row*4]=static_cast<int8_t>(code);
                if(row==rows-1) {
                    #pragma unroll
                    for(int padding=0;padding<InterleavedColumns;++padding)
                        if(padding>=rows)interleaved_output[at+padding*4]=0;
                }
                }
            }else {
            quantized[static_cast<size_t>(row) * padded_width + column] = static_cast<int8_t>(code);
            }
        }
        if constexpr(CachedSum) {
            int sum=code;
            #pragma unroll
            for(int offset=16;offset>0;offset>>=1)
                sum+=__shfl_down_sync(0xffffffffu,sum,offset);
            if(lane==0)sum_output[size_t(row)*groups+group]=sum;
        }
        return;
    }
    __shared__ float warp_maxima[2];
    if ((lane & 31) == 0) {
        warp_maxima[lane >> 5] = maximum;
    }
    __syncthreads();
    if (lane < 32) {
        const int active_warps = (blockDim.x + 31) / 32;
        maximum = lane < active_warps ? warp_maxima[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, offset));
        }
    }
    __shared__ float group_scale;
    if (lane == 0) {
        group_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scale_output[static_cast<size_t>(row) * groups + group] = group_scale;
    }
    __syncthreads();
    int code = 0;
    if (lane < group_size) {
        if (valid) {
            code = static_cast<int>(roundf(value / group_scale));
            code = max(-127, min(127, code));
        }
        quantized[static_cast<size_t>(row) * padded_width + column] =
            static_cast<int8_t>(code);
    }
    if constexpr(CachedSum) {
        int sum=code;
        #pragma unroll
        for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
        __shared__ int warp_sums[2];
        if((lane&31)==0)warp_sums[lane>>5]=sum;
        __syncthreads();
        if(lane==0)sum_output[size_t(row)*groups+group]=warp_sums[0]+warp_sums[1];
    }
}


// One warp per original group, including GS48's two values per lane.
// All formats keep the canonical fp16 rounding and per-group division.
__global__ void __launch_bounds__(32) nint_group_quantize_kernel(NintGroupQuantizeArgs a) {
    int selected=0;
    for(int i=1;i<a.count;++i)if(int(blockIdx.x)>=a.group[i].begin)selected=i;
    const auto g=a.group[selected];
    const int group=int(blockIdx.x)-g.begin,row=blockIdx.y,lane=threadIdx.x;
    const int column=group*g.gs+lane;
    const bool valid0=lane<g.gs && column<a.width;
    const bool valid1=lane+32<g.gs && column+32<a.width;
    const float v0=valid0?nint_activation_value(a.input,size_t(row)*a.width+column,a.fp32):0.f;
    const float v1=valid1?nint_activation_value(a.input,size_t(row)*a.width+column+32,a.fp32):0.f;
    float maximum=fmaxf(fabsf(v0),fabsf(v1));
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1)
        maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
    maximum=__shfl_sync(0xffffffffu,maximum,0);
    const float scale=maximum>0.f?maximum/127.f:1.f;
    if(lane==0)g.scale[size_t(row)*g.groups+group]=scale;
    const int c0=valid0?max(-127,min(127,int(roundf(v0/scale)))):0;
    const int c1=valid1?max(-127,min(127,int(roundf(v1/scale)))):0;
    if(lane<g.gs)g.codes[size_t(row)*g.groups*g.gs+column]=int8_t(c0);
    if(lane+32<g.gs)g.codes[size_t(row)*g.groups*g.gs+column+32]=int8_t(c1);
    if(g.interleaved && lane<g.gs) {
        const size_t index=size_t(group)*7+lane/4;
        if(g.columns==5 || g.columns==6) {
            const size_t at=row<4?index*4+row:size_t(g.groups)*7*4+index*(g.columns-4)+(row-4);
            g.interleaved[at*4+lane%4]=int8_t(c0);
        } else {
            const size_t at=index*g.columns*4+lane%4;
            g.interleaved[at+row*4]=int8_t(c0);
            if(row==a.rows-1)for(int padding=a.rows;padding<g.columns;++padding)
                g.interleaved[at+padding*4]=0;
        }
    }
    int sum=c0+c1;
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,offset);
    if(lane==0)g.sum[size_t(row)*g.groups+group]=sum;
}

constexpr int kNintPrefillTileN = tyloi::cuda::nint_raw::prefill_tile_n;
constexpr int kNintPrefillTileK = tyloi::cuda::nint_raw::prefill_tile_k;
constexpr int kNintPrefillStrideK = 72;
constexpr int kNintPrefillWarps = 8;


template <bool AsyncCopy, int TileM, bool Dense = false>
__device__ __forceinline__ void nint_prefill_load_activation_tile(
        const __half * __restrict__ input,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ activation_tile,
        int first,
        int last,
        int routes,
        int input_width,
        int column_base,
        bool routed_input,
        int thread) {
    constexpr int vector_width = 8;
    constexpr int vectors_per_row = kNintPrefillTileK / vector_width;
    constexpr int vectors = TileM * vectors_per_row;
    for (int index = thread; index < vectors;
         index += kNintPrefillWarps * 32) {
        const int matrix_row = index / vectors_per_row;
        const int vector = index - matrix_row * vectors_per_row;
        const int column = column_base + vector * vector_width;
        const int compact = first + matrix_row;
        int source_row = -1;
        if (compact < last) {
            const int pair = Dense ? compact : ids_dst[compact];
            source_row = Dense || routed_input ? pair : pair / routes;
        }
        __half * destination = activation_tile +
            matrix_row * kNintPrefillStrideK + vector * vector_width;
        if (source_row < 0 || column >= input_width) {
            *reinterpret_cast<int4 *>(destination) = make_int4(0, 0, 0, 0);
        } else if ((input_width & (vector_width - 1)) == 0 &&
                column + vector_width <= input_width) {
            const __half * source = input +
                static_cast<size_t>(source_row) * input_width + column;
            if constexpr (AsyncCopy) {
                tyloi::cuda_detail::copy_16_async(destination, source);
            } else {
                *reinterpret_cast<int4 *>(destination) =
                    *reinterpret_cast<const int4 *>(source);
            }
        } else {
#pragma unroll
            for (int element = 0; element < vector_width; ++element) {
                destination[element] = column + element < input_width
                    ? input[static_cast<size_t>(source_row) * input_width +
                        column + element]
                    : __float2half_rn(0.0f);
            }
        }
    }
}


__device__ __forceinline__ void nint_prefill_load_weight_tile(
        const uint8_t * __restrict__ expert_stream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        __half * __restrict__ weight_tile,
        int local_expert,
        int output_rows,
        int result_rows,
        int groups,
        int group_size,
        int input_width,
        int result_row_base,
        int column_base,
        int projection,
        int thread) {
    constexpr int values_per_vector = 8;
    constexpr int vectors_per_row =
        kNintPrefillTileK / values_per_vector;
    constexpr int vectors = kNintPrefillTileN * vectors_per_row;
    for (int index = thread; index < vectors;
         index += kNintPrefillWarps * 32) {
        const int tile_row = index / vectors_per_row;
        const int vector = index - tile_row * vectors_per_row;
        const int result_row = result_row_base + tile_row;
        const int output_row = result_row + projection * result_rows;
        const int column = column_base + vector * values_per_vector;
        __half * destination = weight_tile +
            tile_row * kNintPrefillStrideK + vector * values_per_vector;
        const bool valid_row = result_row < result_rows &&
            output_row < output_rows;
        const int weight_row = valid_row
            ? local_expert * output_rows + output_row
            : 0;
        const int row_lane = thread & 7;
        int bits = valid_row && row_lane == 0
            ? static_cast<int>(row_q_bits[weight_row])
            : 0;
        uint64_t row_bit_offset = valid_row && row_lane == 0
            ? static_cast<uint64_t>(row_q_bit_offsets[weight_row])
            : 0u;
        float outer_scale = valid_row && row_lane == 0
            ? neuron_scale[weight_row]
            : 0.0f;
        float outer_minimum = valid_row && row_lane == 0
            ? neuron_minimum[weight_row]
            : 0.0f;
        const uint32_t offset_low = __shfl_sync(
            0xffffffffu, static_cast<uint32_t>(row_bit_offset), 0, 8);
        const uint32_t offset_high = __shfl_sync(
            0xffffffffu, static_cast<uint32_t>(row_bit_offset >> 32), 0, 8);
        bits = __shfl_sync(0xffffffffu, bits, 0, 8);
        outer_scale = __shfl_sync(0xffffffffu, outer_scale, 0, 8);
        outer_minimum = __shfl_sync(0xffffffffu, outer_minimum, 0, 8);
        row_bit_offset = static_cast<uint64_t>(offset_low) |
            (static_cast<uint64_t>(offset_high) << 32);

        const int valid_values = valid_row && column < input_width
            ? min(values_per_vector, input_width - column)
            : 0;
        const uint64_t codes = valid_values > 0
            ? unpack_nint_codes8_packed(
                expert_stream,
                row_bit_offset + static_cast<uint64_t>(column) * bits,
                bits)
            : 0u;
        const uint32_t code_mask = bits > 0 ? (1u << bits) - 1u : 0u;
        const size_t metadata_base = static_cast<size_t>(weight_row) * groups;

        // The eight lanes serving one row cooperatively load every subgroup
        // touched by this 64-wide tile exactly once.  Standard NINT groups
        // are at least eight values, so at most eight groups are present.
        // Smaller experimental groups retain this kernel and load metadata
        // directly while decoding each value.
        const int physical_last_column = groups * group_size - 1;
        const int tile_first_group = column_base / group_size;
        const int tile_last_column = column_base + min(
            kNintPrefillTileK - 1,
            physical_last_column - column_base);
        const int tile_last_group = tile_last_column / group_size;
        const int first_group = min(column, physical_last_column) / group_size;
        const int last_group = valid_values > 0
            ? (column + valid_values - 1) / group_size
            : first_group;
        float first_scale = 0.0f;
        float first_minimum = 0.0f;
        float last_scale = 0.0f;
        float last_minimum = 0.0f;
        if (group_size >= values_per_vector) {
            const int source_group = tile_first_group + row_lane;
            float source_scale = 0.0f;
            float source_minimum = 0.0f;
            if (valid_row && source_group <= tile_last_group) {
                const size_t metadata = metadata_base + source_group;
                source_scale = outer_scale *
                    static_cast<float>(subgroup_scale[metadata]);
                source_minimum = outer_minimum *
                    static_cast<float>(subgroup_minimum[metadata]);
            }
            const int first_source_lane = first_group - tile_first_group;
            const int last_source_lane = last_group - tile_first_group;
            first_scale = __shfl_sync(
                0xffffffffu, source_scale, first_source_lane, 8);
            first_minimum = __shfl_sync(
                0xffffffffu, source_minimum, first_source_lane, 8);
            last_scale = __shfl_sync(
                0xffffffffu, source_scale, last_source_lane, 8);
            last_minimum = __shfl_sync(
                0xffffffffu, source_minimum, last_source_lane, 8);
        }

#pragma unroll
        for (int element = 0; element < values_per_vector; element += 2) {
            float decoded[2] = {0.0f, 0.0f};
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int local = element + half;
                if (local < valid_values) {
                    const int value_column = column + local;
                    const int group = value_column / group_size;
                    float scale = first_scale;
                    float minimum = first_minimum;
                    if (group_size < values_per_vector) {
                        const size_t metadata = metadata_base + group;
                        scale = outer_scale *
                            static_cast<float>(subgroup_scale[metadata]);
                        minimum = outer_minimum *
                            static_cast<float>(subgroup_minimum[metadata]);
                    } else if (group != first_group) {
                        scale = last_scale;
                        minimum = last_minimum;
                    }
                    const int code = static_cast<int>(
                        (codes >> (local * bits)) & code_mask);
                    decoded[half] =
                        scale * static_cast<float>(code) - minimum;
                }
            }
            reinterpret_cast<__half2 *>(destination)[element / 2] =
                __floats2half2_rn(decoded[0], decoded[1]);
        }
    }
}


// FusedGlu is an epilogue contract, not a q/k specialization.  Both variants
// consume the same heterogeneous row metadata and execute the same tiled
// decoder.  Keeping the plain projection separate prevents it from reserving
// a second accumulator set that it can never use.
template <int TileM, bool AsyncActivation, bool FusedGlu, bool Dense = false>
__global__ void __launch_bounds__(256, 1) nint_matmul_tiled_prefill_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const __half * __restrict__ input,
        __half * __restrict__ output,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        int routes,
        int experts,
        int output_rows,
        int groups,
        int group_size,
        int input_width,
        int q_expert_stride,
        int pool_phase,
        int epilogue_mode,
        bool routed_input,
        int dense_rows,
        int split_k,
        float* __restrict__ dense_partial) {
    static_assert(
        TileM == 16 || TileM == 32 || TileM == 64 || TileM == 128);
    constexpr int matrix_tiles_m = TileM / 16;
    constexpr int matrix_tiles_n = kNintPrefillTileN / 16;
    constexpr int accumulators_per_warp = (matrix_tiles_m + 1) / 2;
    constexpr int operand_halves =
        (kNintPrefillTileN + TileM) * kNintPrefillStrideK;
    constexpr int output_floats =
        2 * kNintPrefillWarps * 16 * 16;
    constexpr int operand_bytes = operand_halves * sizeof(__half);
    constexpr int output_bytes = output_floats * sizeof(float);
    constexpr int shared_bytes =
        operand_bytes > output_bytes ? operand_bytes : output_bytes;
    __shared__ __align__(16) uint8_t storage[shared_bytes];
    __half * weight_tile = reinterpret_cast<__half *>(storage);
    __half * activation_tile = weight_tile +
        kNintPrefillTileN * kNintPrefillStrideK;

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int thread = warp * 32 + lane;
    const int warp_m0 = warp / matrix_tiles_n;
    const int warp_n = warp - warp_m0 * matrix_tiles_n;
    const int result_rows = FusedGlu ? output_rows / 2 : output_rows;
    constexpr int projections = FusedGlu ? 2 : 1;
    const int output_tiles =
        (result_rows + kNintPrefillTileN - 1) / kNintPrefillTileN;
    const int total_tiles = Dense ? (dense_rows + TileM - 1) / TileM : tile_bounds[experts];
    const int64_t total_tasks =
        static_cast<int64_t>(total_tiles) * output_tiles;

    using FragmentA = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, 16, 16, 16,
        __half, nvcuda::wmma::row_major>;
    using FragmentB = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, 16, 16, 16,
        __half, nvcuda::wmma::col_major>;
    using FragmentC = nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, 16, 16, 16, float>;

    const int64_t task_shift = total_tasks == 0
        ? 0
        : (static_cast<int64_t>(pool_phase) * 17 * output_tiles) %
            total_tasks;
    for (int64_t logical_task = blockIdx.x;
         logical_task < total_tasks;
         logical_task += gridDim.x) {
        const int64_t task = (logical_task + task_shift) % total_tasks;
        const int tile = static_cast<int>(task / output_tiles);
        const int output_tile = static_cast<int>(
            task - static_cast<int64_t>(tile) * output_tiles);
        const int expert = Dense ? 0 : tile_experts[tile];
        const int local_expert = Dense ? 0 : expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int local_tile = Dense ? tile : tile - tile_bounds[expert];
        const int first = Dense ? local_tile * TileM : expert_bounds[expert] + local_tile * TileM;
        const int last = min(first + TileM, Dense ? dense_rows : expert_bounds[expert + 1]);
        const int result_row_base = output_tile * kNintPrefillTileN;
        const uint8_t * expert_stream = bitstream +
            static_cast<size_t>(local_expert) * q_expert_stride;

        FragmentC accumulators[projections][accumulators_per_warp];
#pragma unroll
        for (int projection = 0; projection < projections; ++projection) {
#pragma unroll
            for (int accumulator = 0;
                 accumulator < accumulators_per_warp; ++accumulator) {
                nvcuda::wmma::fill_fragment(
                    accumulators[projection][accumulator], 0.0f);
            }
        }

        const int split_columns = ((input_width + kNintPrefillTileK*split_k - 1) /
            (kNintPrefillTileK*split_k))*kNintPrefillTileK;
        const int column_begin = Dense ? int(blockIdx.y)*split_columns : 0;
        const int column_end = Dense ? min(input_width,column_begin+split_columns) : input_width;
        for (int column_base = column_begin; column_base < column_end;
             column_base += kNintPrefillTileK) {
            nint_prefill_load_activation_tile<AsyncActivation, TileM, Dense>(
                input, ids_dst, activation_tile, first, last, routes,
                input_width, column_base, routed_input, thread);
            if constexpr (AsyncActivation) {
                tyloi::cuda_detail::copy_async_commit();
            }
            nint_prefill_load_weight_tile(
                expert_stream, row_q_bits, row_q_bit_offsets,
                subgroup_scale, subgroup_minimum, neuron_scale,
                neuron_minimum, weight_tile, local_expert, output_rows,
                result_rows, groups, group_size, input_width,
                result_row_base, column_base, 0, thread);
            if constexpr (AsyncActivation) {
                tyloi::cuda_detail::copy_async_wait();
            }
            __syncthreads();

            const bool active_warp = warp_m0 < matrix_tiles_m;
#pragma unroll
            for (int k_local = 0; k_local < kNintPrefillTileK;
                 k_local += 16) {
                if (active_warp && column_base + k_local < input_width) {
                    FragmentB weight_fragment;
                    nvcuda::wmma::load_matrix_sync(
                        weight_fragment,
                        weight_tile + warp_n * 16 *
                            kNintPrefillStrideK + k_local,
                        kNintPrefillStrideK);
#pragma unroll
                    for (int accumulator = 0;
                         accumulator < accumulators_per_warp;
                         ++accumulator) {
                        const int matrix_tile_m =
                            warp_m0 + accumulator * 2;
                        if (matrix_tile_m < matrix_tiles_m) {
                            FragmentA activation_fragment;
                            nvcuda::wmma::load_matrix_sync(
                                activation_fragment,
                                activation_tile + matrix_tile_m * 16 *
                                    kNintPrefillStrideK + k_local,
                                kNintPrefillStrideK);
                            nvcuda::wmma::mma_sync(
                                accumulators[0][accumulator],
                                activation_fragment, weight_fragment,
                                accumulators[0][accumulator]);
                        }
                    }
                }
            }
            __syncthreads();

            if constexpr (FusedGlu) {
                nint_prefill_load_weight_tile(
                    expert_stream, row_q_bits, row_q_bit_offsets,
                    subgroup_scale, subgroup_minimum, neuron_scale,
                    neuron_minimum, weight_tile, local_expert, output_rows,
                    result_rows, groups, group_size, input_width,
                    result_row_base, column_base, 1, thread);
                __syncthreads();
#pragma unroll
                for (int k_local = 0; k_local < kNintPrefillTileK;
                     k_local += 16) {
                    if (active_warp && column_base + k_local < input_width) {
                        FragmentB weight_fragment;
                        nvcuda::wmma::load_matrix_sync(
                            weight_fragment,
                            weight_tile + warp_n * 16 *
                                kNintPrefillStrideK + k_local,
                            kNintPrefillStrideK);
#pragma unroll
                        for (int accumulator = 0;
                             accumulator < accumulators_per_warp;
                             ++accumulator) {
                            const int matrix_tile_m =
                                warp_m0 + accumulator * 2;
                            if (matrix_tile_m < matrix_tiles_m) {
                                FragmentA activation_fragment;
                                nvcuda::wmma::load_matrix_sync(
                                    activation_fragment,
                                    activation_tile + matrix_tile_m * 16 *
                                        kNintPrefillStrideK + k_local,
                                    kNintPrefillStrideK);
                                nvcuda::wmma::mma_sync(
                                    accumulators[1][accumulator],
                                    activation_fragment, weight_fragment,
                                    accumulators[1][accumulator]);
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }

        float * output_tile_storage = reinterpret_cast<float *>(storage);
#pragma unroll
        for (int accumulator = 0;
             accumulator < accumulators_per_warp; ++accumulator) {
            const int matrix_tile_m = warp_m0 + accumulator * 2;
            if (matrix_tile_m >= matrix_tiles_m) {
                continue;
            }
            float * first_projection = output_tile_storage + warp * 16 * 16;
            float * second_projection = first_projection +
                kNintPrefillWarps * 16 * 16;
            nvcuda::wmma::store_matrix_sync(
                first_projection, accumulators[0][accumulator], 16,
                nvcuda::wmma::mem_row_major);
            if constexpr (FusedGlu) {
                nvcuda::wmma::store_matrix_sync(
                    second_projection, accumulators[1][accumulator], 16,
                    nvcuda::wmma::mem_row_major);
            }
            __syncwarp();
#pragma unroll
            for (int element = lane; element < 16 * 16; element += 32) {
                const int row = element / 16;
                const int column = element - row * 16;
                const int compact = first + matrix_tile_m * 16 + row;
                const int result_row =
                    result_row_base + warp_n * 16 + column;
                if (compact < last && result_row < result_rows) {
                    const int pair = Dense ? compact : ids_dst[compact];
                    float value = first_projection[element];
                    if constexpr (FusedGlu) {
                        const float gate = __half2float(
                            __float2half_rn(value));
                        const float up = __half2float(
                            __float2half_rn(second_projection[element]));
                        value = mfq_glu_runtime(
                            gate, up, epilogue_mode == 2 ? 1 : 0);
                    }
                    if constexpr(Dense) {
                        if(split_k>1)dense_partial[(size_t(blockIdx.y)*dense_rows+pair)*result_rows+result_row]=value;
                        else output[static_cast<size_t>(pair)*result_rows+result_row]=__float2half_rn(value);
                    } else output[static_cast<size_t>(pair)*result_rows+result_row]=__float2half_rn(value);
                }
            }
            __syncwarp();
        }
        __syncthreads();
    }
}


template<int GroupLanes=0,int ResidentBlocks=1,int Rows=2>
__global__ void __launch_bounds__(128,ResidentBlocks) nint_mixed_routes_kernel(
        const int64_t* pointers,const int32_t* params,const int32_t* expert_pool,
        const int32_t* expert_local,const int32_t* ids,__half* output,
        int pairs,int routes,int experts,int output_rows,bool routed_input,bool multi_sum) {
    const int pair=blockIdx.y;
    const int output_row=int(blockIdx.x)*(4*Rows)+int(threadIdx.y)*Rows;
    if(pair>=pairs || output_row>=output_rows)return;
    const int expert=ids[pair];
    if(static_cast<unsigned>(expert)>=static_cast<unsigned>(experts))return;
    const int pool=expert_pool[expert],local=expert_local[expert];
    if(pool<0 || local<0)return;
    const auto* p=pointers+int64_t(pool)*9;
    const auto* geometry=params+int64_t(pool)*3;
    if constexpr(GroupLanes!=0) {
        if(geometry[1]==24 || geometry[1]==28) {
            float values[Rows];
            const int source=routed_input?pair:pair/routes;
#define TYLOINFER_NINT_ROUTED_GROUP(GS) tyloi::cuda::packed_nint::nint_grouped_routed_pair<GS,GroupLanes,Rows>( \
                reinterpret_cast<const uint8_t*>(p[0]),reinterpret_cast<const uint8_t*>(p[1]), \
                reinterpret_cast<const int64_t*>(p[2]),reinterpret_cast<const uint8_t*>(p[3]), \
                reinterpret_cast<const uint8_t*>(p[4]),reinterpret_cast<const float*>(p[5]), \
                reinterpret_cast<const float*>(p[6]), \
                reinterpret_cast<const int8_t*>(p[7])+int64_t(source)*geometry[0]*geometry[1], \
                reinterpret_cast<const float*>(p[8])+int64_t(source)*geometry[0], \
                local,output_row,output_rows,geometry[0],geometry[2],values,multi_sum)
            if(geometry[1]==24)TYLOINFER_NINT_ROUTED_GROUP(24);else TYLOINFER_NINT_ROUTED_GROUP(28);
#undef TYLOINFER_NINT_ROUTED_GROUP
            if(threadIdx.x==0) {
#pragma unroll
                for(int r=0;r<Rows;++r)if(output_row+r<output_rows)
                    output[int64_t(pair)*output_rows+output_row+r]=__float2half_rn(values[r]);
            }
            return;
        }
    }
    static_assert(Rows==2 || GroupLanes==32);
    if constexpr(Rows==1) {
        // The one-row variant is selected only for the grouped geometries.
        // Other group sizes use paired rows with the unchanged fallback.
        float values[2];
        nint_matmul_routed_pair<true>(reinterpret_cast<const uint8_t*>(p[0]),
            reinterpret_cast<const uint8_t*>(p[1]),reinterpret_cast<const int64_t*>(p[2]),
            reinterpret_cast<const uint8_t*>(p[3]),reinterpret_cast<const uint8_t*>(p[4]),
            reinterpret_cast<const float*>(p[5]),reinterpret_cast<const float*>(p[6]),
            reinterpret_cast<const int8_t*>(p[7]),reinterpret_cast<const float*>(p[8]),
            output,pair,routed_input?pair:pair/routes,local,output_row,output_rows,
            geometry[0],geometry[0]*geometry[1],geometry[1],geometry[2],0,values);
        if(threadIdx.x==0)output[int64_t(pair)*output_rows+output_row]=__float2half(values[0]);
        return;
    }
    nint_matmul_routed_pair(reinterpret_cast<const uint8_t*>(p[0]),
        reinterpret_cast<const uint8_t*>(p[1]),reinterpret_cast<const int64_t*>(p[2]),
        reinterpret_cast<const uint8_t*>(p[3]),reinterpret_cast<const uint8_t*>(p[4]),
        reinterpret_cast<const float*>(p[5]),reinterpret_cast<const float*>(p[6]),
        reinterpret_cast<const int8_t*>(p[7]),reinterpret_cast<const float*>(p[8]),
        output,pair,routed_input?pair:pair/routes,local,output_row,output_rows,
        geometry[0],geometry[0]*geometry[1],geometry[1],geometry[2],0);
}

template<int GroupSize,int Bits,int Warps=1,bool Single=false,bool LateScale=false,bool PackedRows=false>
__global__ void __launch_bounds__(128) nint_mixed_routes_hinted_kernel(
        const int64_t* pointers,const int32_t* params,const int32_t* expert_pool,
        const int32_t* expert_local,const int32_t* ids,__half* output,
        int pairs,int routes,int experts,int output_rows,bool routed_input,
        const NintSingleRouteWeight single,bool multi_sum) {
    static_assert(Warps==1 || Warps==4);
    static_assert(!PackedRows || Single);
    constexpr int Rows=Warps==4?4:2,rows_per_block=Warps==4?4:8;
    const int pair=blockIdx.y,first=int(blockIdx.x)*rows_per_block+(Warps==1?int(threadIdx.y)*Rows:0);
    if(pair>=pairs || first>=output_rows)return;
    const int expert=ids[pair];
    if(static_cast<unsigned>(expert)>=static_cast<unsigned>(experts))return;
    const int local=expert_local[expert];
    const int64_t* p;const int32_t* geometry;
    if constexpr(Single) {
        if(unsigned(local)>=unsigned(single.local_experts))return;
        p=single.pointers;geometry=single.geometry;
    } else {
        const int pool=expert_pool[expert];if(pool<0 || local<0)return;
        p=pointers+int64_t(pool)*9;geometry=params+int64_t(pool)*3;
    }
    const auto* bits=reinterpret_cast<const uint8_t*>(p[1]);
    const int neuron=local*output_rows+first,source=routed_input?pair:pair/routes;
    // Expert slots and row metadata may change after graph capture. A host
    // hint never substitutes for checking the current rows on the device.
    bool matching=geometry[1]==GroupSize && geometry[0]<INT_MAX/(GroupSize*Bits);
#pragma unroll
    for(int r=0;r<Rows;++r)if(first+r<output_rows)matching=matching && bits[neuron+r]==Bits;
    if(matching) {
        float values[Rows];
        tyloi::cuda::packed_nint::nint_fixed_group_rows<GroupSize,Bits,Warps,Rows,LateScale,PackedRows>(
            reinterpret_cast<const uint8_t*>(p[0]),reinterpret_cast<const int64_t*>(p[2]),
            reinterpret_cast<const uint8_t*>(p[3]),reinterpret_cast<const uint8_t*>(p[4]),
            reinterpret_cast<const float*>(p[5]),reinterpret_cast<const float*>(p[6]),
            reinterpret_cast<const int8_t*>(p[7])+int64_t(source)*geometry[0]*GroupSize,
            reinterpret_cast<const float*>(p[8])+int64_t(source)*geometry[0],
            local,first,output_rows,geometry[0],geometry[2],values,multi_sum,
            reinterpret_cast<const uint4*>(single.row_metadata));
        if constexpr(Warps==4) {
            __shared__ float partial[4][Rows];
            if(threadIdx.x==0) {
#pragma unroll
                for(int r=0;r<Rows;++r)partial[threadIdx.y][r]=values[r];
            }
            __syncthreads();
            if(threadIdx.x==0 && threadIdx.y==0) {
#pragma unroll
                for(int r=0;r<Rows;++r)if(first+r<output_rows)
                    output[int64_t(pair)*output_rows+first+r]=__float2half_rn(
                        (partial[0][r]+partial[1][r])+(partial[2][r]+partial[3][r]));
            }
        }else if(threadIdx.x==0) {
            output[int64_t(pair)*output_rows+first]=__float2half_rn(values[0]);
            if(first+1<output_rows)output[int64_t(pair)*output_rows+first+1]=__float2half_rn(values[1]);
        }
    } else if(Warps==1 || threadIdx.y==0) {
#pragma unroll
        for(int r=0;r<Rows;r+=2)nint_matmul_routed_pair(reinterpret_cast<const uint8_t*>(p[0]),bits,
            reinterpret_cast<const int64_t*>(p[2]),reinterpret_cast<const uint8_t*>(p[3]),
            reinterpret_cast<const uint8_t*>(p[4]),reinterpret_cast<const float*>(p[5]),
            reinterpret_cast<const float*>(p[6]),reinterpret_cast<const int8_t*>(p[7]),
            reinterpret_cast<const float*>(p[8]),output,pair,source,local,first+r,output_rows,
            geometry[0],geometry[0]*geometry[1],geometry[1],geometry[2],0);
    }
}

__device__ __forceinline__ int unpack_nint_codes4_words(
        const uint8_t* stream,uint64_t bit_offset,int bits,uint64_t bytes) {
    const uint64_t word_byte=(bit_offset>>3)&~uint64_t(3);
    // The selected stream base is aligned. Small unpadded tensors retain the
    // byte decoder at their end; two word loads never cross the storage bound.
    if(word_byte+8>bytes)return unpack_nint_codes4(stream,bit_offset,bits);
    const auto* words=reinterpret_cast<const uint32_t*>(stream+word_byte);
    const uint32_t packed=__funnelshift_r(words[0],words[1],int(bit_offset&31u));
    const uint32_t pairs=__byte_perm(packed,packed>>(2*bits),0x5410);
    const uint32_t codes=__byte_perm(pairs,pairs>>bits,0x6240);
    return static_cast<int>(codes * 1u & (((1u<<bits)-1u)*0x01010101u));

}

__device__ __forceinline__ int unpack_nint_codes4_unchecked(
        const uint8_t* stream,uint64_t bit_offset,int bits,uint64_t bytes) {
    const uint64_t word_byte=(bit_offset>>3)&~uint64_t(3);
    // The caller proves both aligned word loads fit the packed row storage.
    const auto* words=reinterpret_cast<const uint32_t*>(stream+word_byte);
    const uint32_t packed=__funnelshift_r(words[0],words[1],int(bit_offset&31u));
    const uint32_t pairs=__byte_perm(packed,packed>>(2*bits),0x5410);
    const uint32_t codes=__byte_perm(pairs,pairs>>bits,0x6240);
    return static_cast<int>(codes * 1u & (((1u<<bits)-1u)*0x01010101u));

}

// A lane accumulates one complete affine group with integer dot products.
// Row-specific bit widths remain dynamic, including adaptive NINTv2 and q8.
template<int GroupSize, int Rows, int Warps>
__global__ void __launch_bounds__(128) nint_matmul_dense_kernel(
        const uint8_t* __restrict__ bitstream,
        const uint8_t* __restrict__ row_q_bits,
        const int64_t* __restrict__ row_q_bit_offsets,
        const uint8_t* __restrict__ subgroup_scale,
        const uint8_t* __restrict__ subgroup_minimum,
        const float* __restrict__ neuron_scale,
        const float* __restrict__ neuron_minimum,
        const int8_t* __restrict__ activation,
        const float* __restrict__ activation_scale,
        half* __restrict__ output,
        int activation_rows, int output_rows, int groups, uint64_t packed_bytes) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int row = blockIdx.x * (4 / Warps) + warp / Warps;
    const int local_warp = warp % Warps;
    const bool valid_row = row < output_rows;
    const int bits = valid_row ? row_q_bits[row] : 4;
    const uint64_t row_bit = valid_row ? row_q_bit_offsets[row] : 0;
    const int padded_width = groups * GroupSize;
    const float outer_scale = valid_row ? neuron_scale[row] : 0.0f;
    const float outer_minimum = valid_row ? neuron_minimum[row] : 0.0f;
    const uint64_t last_word = ((row_bit + uint64_t(padded_width) * bits) >> 3) & ~uint64_t(3);
    const bool full_words = last_word <= packed_bytes && packed_bytes - last_word >= 8;
    float acc[Rows] = {};
    for (int group = valid_row ? local_warp * 32 + lane : groups;
         group < groups; group += Warps * 32) {
        int dots[Rows] = {}, sums[Rows] = {};
#pragma unroll
        for (int chunk = 0; chunk < GroupSize / 4; ++chunk) {
            const int column = group * GroupSize + chunk * 4;
            const uint64_t bit = row_bit + uint64_t(column) * bits;
            const int qw = full_words ? unpack_nint_codes4_unchecked(bitstream, bit, bits, packed_bytes)
                : unpack_nint_codes4_words(bitstream, bit, bits, packed_bytes);
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                if (m < activation_rows) {
                    const int qx = *reinterpret_cast<const int*>(activation + size_t(m) * padded_width + column);
                    sums[m] = __dp4a(0x01010101, qx, sums[m]);
                    dots[m] = __dp4a(bits == 8 ? qw ^ int(0x80808080u) : qw, qx, dots[m]);
                }
            }
        }
        const float scale = outer_scale * float(subgroup_scale[size_t(row) * groups + group]);
        const float minimum = outer_minimum * float(subgroup_minimum[size_t(row) * groups + group]);
#pragma unroll
        for (int m = 0; m < Rows; ++m) {
            if (m < activation_rows) {
                const int dot = dots[m] + (bits == 8 ? 128 * sums[m] : 0);
                acc[m] += activation_scale[size_t(m) * groups + group] *
                    (scale * float(dot) - minimum * float(sums[m]));
            }
        }
    }
#pragma unroll
    for (int m = 0; m < Rows; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], offset);
    }
    if constexpr (Warps == 1) {
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0 && valid_row && m < activation_rows)
                output[size_t(m) * output_rows + row] = __float2half_rn(acc[m]);
    } else {
        __shared__ float partial[Rows][4];
#pragma unroll
        for (int m = 0; m < Rows; ++m)
            if (lane == 0) partial[m][warp] = acc[m];
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int m = 0; m < Rows; ++m) {
                float value = lane < Warps ? partial[m][lane] : 0.0f;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_xor_sync(0xffffffffu, value, offset);
                if (lane == 0 && valid_row && m < activation_rows)
                    output[size_t(m) * output_rows + row] = __float2half_rn(value);
            }
        }
    }
}

#include "nint_whole_group.cuh"
#include "nint_whole_warp_rows.cuh"

// Decode four adjacent values once and store one aligned eight-byte vector.
template<int GroupSize>
__global__ void nint_decode_vec4_kernel(
        const uint8_t* bitstream, const uint8_t* row_q_bits,
        const int64_t* row_q_bit_offsets, const uint8_t* subgroup_scale,
        const uint8_t* subgroup_minimum, const float* neuron_scale,
        const float* neuron_minimum, half* output,
        int rows, int groups, int width, uint64_t packed_bytes) {
    const int column = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (column >= width) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        const int bits = row_q_bits[row];
        const int codes = unpack_nint_codes4_words(bitstream,
            uint64_t(row_q_bit_offsets[row]) + uint64_t(column) * bits, bits, packed_bytes);
        const size_t metadata = size_t(row) * groups + column / GroupSize;
        const float scale = neuron_scale[row] * float(subgroup_scale[metadata]);
        const float minimum = neuron_minimum[row] * float(subgroup_minimum[metadata]);
        half values[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            values[i] = __float2half_rn(scale * float((uint32_t(codes) >> (8 * i)) & 255u) - minimum);
        reinterpret_cast<uint2*>(output + size_t(row) * width)[column / 4] = make_uint2(
            (uint32_t(__half_as_ushort(values[0])) | (uint32_t(__half_as_ushort(values[1])) << 16)),
            (uint32_t(__half_as_ushort(values[2])) | (uint32_t(__half_as_ushort(values[3])) << 16)));
    }
}

// One NINT compute kernel covers every q, k, group size, M<=8, and routed MFE
// projection. q is row metadata; k has already been baked into the subgroup
// metadata values. Routed execution changes only the indexing contract, not
// the packed-weight compute kernel.
// Accumulate a complete quantization group in integers before its affine.
// The caller selects this experimental math only for one activation row.
__device__ __forceinline__ float nint_group_partial(
        const uint8_t* bitstream,int bits,uint64_t row_bit_offset,
        const uint8_t* scale_row,const uint8_t* minimum_row,
        float outer_scale,float outer_minimum,const int8_t* activation,
        const float* activation_scale,int groups,int padded_width,int group_size,
        uint64_t packed_bytes,int lane) {
    float partial=0.0f;
    const uint64_t end_word=((row_bit_offset+uint64_t(padded_width)*uint64_t(bits))>>3)&~uint64_t(3);
    const uint32_t row_bit_remainder=uint32_t(row_bit_offset&31u);
    const bool relative_fits=uint64_t(padded_width)*uint64_t(bits)+row_bit_remainder<=0xffffffffu;
    const auto* row_stream=bitstream+((row_bit_offset>>3)&~uint64_t(3));
    const bool complete_words=relative_fits && end_word<=packed_bytes && packed_bytes-end_word>=8;
    for(int group=lane;group<groups;group+=32) {
        const int column0=group*group_size;
        int dot=0,activation_sum=0;
        int element=0;
        for(;element+3<group_size;element+=4) {
            const int column=column0+element;
            const uint32_t relative_bit=row_bit_remainder+uint32_t(column)*uint32_t(bits);
            const int weight_codes=complete_words
                ? unpack_nint_codes4_unchecked(row_stream,relative_bit,bits,packed_bytes)
                : unpack_nint_codes4(bitstream,row_bit_offset+uint64_t(column)*uint64_t(bits),bits);
            const int activation_codes=load_i8x4(activation+column);
            activation_sum=__dp4a(0x01010101,activation_codes,activation_sum);
            dot=__dp4a(bits==8 ? (weight_codes^int(0x80808080u)):weight_codes,activation_codes,dot);
        }
        if(bits==8)dot+=128*activation_sum;
        for(;element<group_size;++element) {
            const int column=column0+element;
            const int qx=int(activation[column]);
            const int qw=int(unpack_nint_code(bitstream,row_bit_offset,column,bits));
            activation_sum+=qx;dot+=qx*qw;
        }
        const float inner_scale=float(scale_row[group]);
        const float inner_minimum=float(minimum_row[group]);
        partial+=activation_scale[group]*(outer_scale*inner_scale*float(dot)-outer_minimum*inner_minimum*float(activation_sum));
    }

    return partial;
}

template<bool Single=false,bool Grouped=false,int RowCapacity=8>
__global__ void __launch_bounds__(128) nint_matmul_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width,
        int group_size,
        const int32_t * __restrict__ route_ids,
        const int32_t * __restrict__ expert_local,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        int routes,
        int experts,
        int q_expert_stride,
        int pool_phase,
        int epilogue_mode,
        bool routed_input,
        uint64_t packed_bytes=0) {
    constexpr int warps_per_block = 4;
    constexpr int maximum_activation_rows = Single ? 1 : RowCapacity;
    constexpr int routed_rows_per_warp = 2;
    const int chunks = (group_size + 3) / 4;
    const int groups_per_warp = 32 / chunks;
    const int lane = static_cast<int>(threadIdx.x);

    if constexpr(!Single) {
    if (ids_dst != nullptr) {
        constexpr int route_tile = 8;
        const int rows_per_task = epilogue_mode == 0
            ? routed_rows_per_warp
            : 1;
        const int result_rows = epilogue_mode == 0
            ? output_rows
            : output_rows / 2;
        const int row_blocks =
            (result_rows + rows_per_task - 1) / rows_per_task;
        const int total_tiles = tile_bounds[experts];
        const int64_t total_tasks =
            static_cast<int64_t>(total_tiles) * row_blocks;
        const int64_t task_shift = total_tasks == 0
            ? 0
            : (static_cast<int64_t>(pool_phase) * 17 * row_blocks) %
                total_tasks;
        for (int64_t logical_task = blockIdx.x;
             logical_task < total_tasks;
             logical_task += gridDim.x) {
            const int64_t task =
                (logical_task + task_shift) % total_tasks;
            const int output_row0 =
                static_cast<int>(task % row_blocks) *
                    rows_per_task;
            const int tile = static_cast<int>(task / row_blocks);
            const int expert = tile_experts[tile];
            const int local_expert = expert_local[expert];
            if (local_expert < 0) {
                continue;
            }
            const int local_tile = tile - tile_bounds[expert];
            const int first =
                expert_bounds[expert] + local_tile * route_tile;
            const int last = min(
                first + route_tile, expert_bounds[expert + 1]);
            for (int compact = first + static_cast<int>(threadIdx.y);
                 compact < last;
                 compact += warps_per_block) {
                const int pair = ids_dst[compact];
                const int source_row = routed_input
                    ? pair
                    : pair / routes;
                nint_matmul_routed_pair(
                    bitstream, row_q_bits, row_q_bit_offsets,
                    subgroup_scale, subgroup_minimum, neuron_scale,
                    neuron_minimum, activation, activation_scale, output,
                    pair, source_row, local_expert, output_row0,
                    output_rows, groups, padded_width, group_size,
                    q_expert_stride, epilogue_mode);
            }
        }
        return;
    }

    if (route_ids != nullptr) {
        const int token = static_cast<int>(blockIdx.z) * warps_per_block +
            static_cast<int>(threadIdx.y);
        const int route = static_cast<int>(blockIdx.y);
        const int rows_per_task = epilogue_mode == 0
            ? routed_rows_per_warp
            : 1;
        const int output_row0 =
            static_cast<int>(blockIdx.x) * rows_per_task;
        if (token >= activation_rows || route >= routes) {
            return;
        }
        const int pair = token * routes + route;
        const int expert = route_ids[pair];
        if (static_cast<unsigned int>(expert) >=
                static_cast<unsigned int>(experts)) {
            return;
        }
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            return;
        }
        const int source_row = routed_input ? pair : token;
        nint_matmul_routed_pair(
            bitstream, row_q_bits, row_q_bit_offsets, subgroup_scale,
            subgroup_minimum, neuron_scale, neuron_minimum, activation,
            activation_scale, output, pair, source_row, local_expert,
            output_row0, output_rows, groups, padded_width, group_size,
            q_expert_stride, epilogue_mode);
        return;
    }

    }
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }

    const int bits = static_cast<int>(row_q_bits[output_row]);
    const uint64_t row_bit_offset =
        static_cast<uint64_t>(row_q_bit_offsets[output_row]);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }

    if constexpr(Single && Grouped) {
        accumulators[0]=nint_group_partial(bitstream,bits,row_bit_offset,
            scale_row,minimum_row,outer_scale,outer_minimum,activation,
            activation_scale,groups,padded_width,group_size,packed_bytes,lane);
    } else {
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int element = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const bool full_chunk = active_lane && element + 3 < group_size;
    const bool tail_chunk =
        active_lane && element < group_size && !full_chunk;
    const uint64_t end_word=((row_bit_offset+uint64_t(padded_width)*uint64_t(bits))>>3)&~uint64_t(3);
    const uint32_t row_bit_remainder=uint32_t(row_bit_offset&31u);
    const bool relative_fits=uint64_t(padded_width)*uint64_t(bits)+row_bit_remainder<=0xffffffffu;
    const auto* row_stream=bitstream+((row_bit_offset>>3)&~uint64_t(3));
    const bool complete_words=Single && full_chunk && relative_fits && end_word<=packed_bytes && packed_bytes-end_word>=8 && (group_size&3)==0 && (reinterpret_cast<uintptr_t>(activation)&3u)==0 && bits<8;
    if(complete_words) {
    for (int group = active_lane ? relative_group : groups,
             column = relative_group * group_size + element;
         group < groups;
         group += groups_per_warp, column += groups_per_warp * group_size) {
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        if (full_chunk) {
            const uint32_t bit_offset=row_bit_remainder+uint32_t(column)*uint32_t(bits);
            const int weight_codes=unpack_nint_codes4_unchecked(row_stream,bit_offset,bits,packed_bytes);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = *reinterpret_cast<const int*>(activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    } else {
    for (int group = active_lane ? relative_group : groups,
             column = relative_group * group_size + element;
         group < groups;
         group += groups_per_warp, column += groups_per_warp * group_size) {
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        if (full_chunk) {
            const uint64_t bit_offset = row_bit_offset +
                static_cast<uint64_t>(column) * static_cast<uint64_t>(bits);
            const int weight_codes = Single
                ? unpack_nint_codes4_words(bitstream,bit_offset,bits,packed_bytes)
                : unpack_nint_codes4(bitstream, bit_offset, bits);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = load_i8x4(
                        activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = bits == 8
                        ? __dp4a(
                              weight_codes ^ static_cast<int>(0x80808080u),
                              activation_codes,
                              0) + 128 * activation_sum
                        : __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        } else if (tail_chunk) {
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    int dot = 0;
                    int activation_sum = 0;
#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        if (element + component < group_size) {
                            const int activation_code = static_cast<int>(
                                activation[
                                    static_cast<size_t>(row) * padded_width +
                                    column + component]);
                            const int weight_code = static_cast<int>(
                                unpack_nint_code(
                                    bitstream,
                                    row_bit_offset,
                                    column + component,
                                    bits));
                            dot += weight_code * activation_code;
                            activation_sum += activation_code;
                        }
                    }
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    }
    }

#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}


// Reuse prepared activations across output warps for the one-row path.
// Packed row metadata and the original FP32 accumulation remain unchanged.
template<bool Grouped=false>
__global__ void __launch_bounds__(1024) nint_matmul_activation_cache_kernel(
    const uint8_t* __restrict__ bitstream,
    const uint8_t* __restrict__ row_q_bits,
    const int64_t* __restrict__ row_q_bit_offsets,
    const uint8_t* __restrict__ subgroup_scale,
    const uint8_t* __restrict__ subgroup_minimum,
    const float* __restrict__ neuron_scale,
    const float* __restrict__ neuron_minimum,
    const int8_t* __restrict__ prepared_activation,
    const float* __restrict__ prepared_scale,
    half* __restrict__ output,
    int padded_width, int groups, int group_size, int output_rows,
    uint64_t packed_bytes) {
    const int warps_per_block = int(blockDim.y);
    const int lane = int(threadIdx.x), warp = int(threadIdx.y);
    const int thread = warp * 32 + lane;
    extern __shared__ unsigned char scratch[];
    int8_t* activation = reinterpret_cast<int8_t*>(scratch);
    float* activation_scale = reinterpret_cast<float*>(scratch + ((padded_width + 3) & ~3));
    for (int column = thread * 4; column < padded_width; column += warps_per_block * 128) {
        if (column + 4 <= padded_width) {
            *reinterpret_cast<int*>(activation + column) = load_i8x4(prepared_activation + column);
        } else {
            for (int tail = column; tail < padded_width; ++tail)
                activation[tail] = prepared_activation[tail];
        }
    }
    for (int group = thread; group < groups; group += warps_per_block * 32)
        activation_scale[group] = prepared_scale[group];
    __syncthreads();
    constexpr bool Single = true;
    constexpr int maximum_activation_rows = 1;
    constexpr int activation_rows = 1;
    const int chunks = (group_size + 3) / 4;
    const int groups_per_warp = 32 / chunks;
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }

    const int bits = static_cast<int>(row_q_bits[output_row]);
    const uint64_t row_bit_offset =
        static_cast<uint64_t>(row_q_bit_offsets[output_row]);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }

    if constexpr(Grouped) {
        accumulators[0]=nint_group_partial(bitstream,bits,row_bit_offset,
            scale_row,minimum_row,outer_scale,outer_minimum,activation,
            activation_scale,groups,padded_width,group_size,packed_bytes,lane);
    } else {
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int element = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const bool full_chunk = active_lane && element + 3 < group_size;
    const bool tail_chunk =
        active_lane && element < group_size && !full_chunk;
    const uint64_t end_word=((row_bit_offset+uint64_t(padded_width)*uint64_t(bits))>>3)&~uint64_t(3);
    const uint32_t row_bit_remainder=uint32_t(row_bit_offset&31u);
    const bool relative_fits=uint64_t(padded_width)*uint64_t(bits)+row_bit_remainder<=0xffffffffu;
    const auto* row_stream=bitstream+((row_bit_offset>>3)&~uint64_t(3));
    const bool complete_words=Single && full_chunk && relative_fits && end_word<=packed_bytes && packed_bytes-end_word>=8 && (group_size&3)==0 && (reinterpret_cast<uintptr_t>(activation)&3u)==0 && bits<8;
    if(complete_words) {
    for (int group_base = 0;
         group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        if (full_chunk) {
            const uint32_t bit_offset=row_bit_remainder+uint32_t(column)*uint32_t(bits);
            const int weight_codes=(bits == 8 && (bit_offset & 31u) == 0) ?
                *reinterpret_cast<const int*>(row_stream + (bit_offset >> 3)) :
                unpack_nint_codes4_unchecked(row_stream,bit_offset,bits,packed_bytes);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = *reinterpret_cast<const int*>(activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    } else {
    for (int group_base = 0;
         group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        if (full_chunk) {
            const uint64_t bit_offset = row_bit_offset +
                static_cast<uint64_t>(column) * static_cast<uint64_t>(bits);
            const int weight_codes = Single
                ? ((bits == 8 && (bit_offset & 31u) == 0 && (bit_offset >> 3) + 4 <= packed_bytes) ?
                    *reinterpret_cast<const int*>(bitstream + (bit_offset >> 3)) :
                    unpack_nint_codes4_words(bitstream,bit_offset,bits,packed_bytes))
                : unpack_nint_codes4(bitstream, bit_offset, bits);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = load_i8x4(
                        activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = bits == 8
                        ? __dp4a(
                              weight_codes ^ static_cast<int>(0x80808080u),
                              activation_codes,
                              0) + 128 * activation_sum
                        : __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        } else if (tail_chunk) {
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    int dot = 0;
                    int activation_sum = 0;
#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        if (element + component < group_size) {
                            const int activation_code = static_cast<int>(
                                activation[
                                    static_cast<size_t>(row) * padded_width +
                                    column + component]);
                            const int weight_code = static_cast<int>(
                                unpack_nint_code(
                                    bitstream,
                                    row_bit_offset,
                                    column + component,
                                    bits));
                            dot += weight_code * activation_code;
                            activation_sum += activation_code;
                        }
                    }
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }
    }
    }

#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}


// The caller has proven q8
// row widths, aligned offsets, four-element groups and aligned storage.
#include "nint_q8_ordered.cuh"

template<int maximum_activation_rows,int FixedGroup=0>
__global__ void __launch_bounds__(128) nint_matmul_aligned_q8_kernel(
        const uint8_t * __restrict__ bitstream,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows, int output_rows, int groups,
        int padded_width, int runtime_group_size) {
    const int group_size=FixedGroup?FixedGroup:runtime_group_size;
    const int output_row = static_cast<int>(blockIdx.x) * 4 +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }
    const int chunks = group_size / 4;
    const int groups_per_warp = 32 / chunks;
    const int lane = static_cast<int>(threadIdx.x);
    const int relative_group = lane / chunks;
    const int element = (lane - relative_group * chunks) * 4;
    const uint8_t * weight_row = bitstream +
        (static_cast<uint64_t>(row_q_bit_offsets[output_row]) >> 3);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }
    for (int group_base = 0; group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (relative_group >= groups_per_warp || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        const int weight_codes = static_cast<int>(
            *reinterpret_cast<const uint32_t *>(weight_row + column));
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (maximum_activation_rows == 1 || row < activation_rows) {
                const int activation_codes = load_i8x4(
                    activation + static_cast<size_t>(row) * padded_width +
                    column);
                const int activation_sum = __dp4a(
                    0x01010101, activation_codes, 0);
                const int dot = __dp4a(
                    weight_codes ^ static_cast<int>(0x80808080u),
                    activation_codes, 0) + 128 * activation_sum;
                const float input_scale = activation_scale[
                    static_cast<size_t>(row) * groups + group];
                accumulators[row] += input_scale * (
                    outer_scale * inner_scale * static_cast<float>(dot) -
                    outer_minimum * inner_minimum *
                        static_cast<float>(activation_sum));
            }
        }
    }
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (maximum_activation_rows == 1 || row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}





template <typename Output>
__global__ void nint8_zero_decode_rows_kernel(
        const int8_t * __restrict__ quantized,
        const __half * __restrict__ scale,
        Output * __restrict__ output,
        int rows,
        int groups,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / width);
        const int column = static_cast<int>(index % width);
        const int group = column / 32;
        const int lane = column & 31;
        const size_t block = static_cast<size_t>(row) * groups + group;
        output[index] = nint_decode_store<Output>(
            __half2float(scale[block]) * static_cast<float>(
                quantized[block * 32 + lane]));
    }
}


// NINT8-0 has one packed small-M compute kernel; M is a grid dimension rather
// than a template parameter, so it does not create one binary per M.
__global__ void __launch_bounds__(128) nint8_zero_matmul_kernel(
        const int8_t * __restrict__ weight,
        const __half * __restrict__ weight_scale,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width) {
    constexpr int warps_per_block = 4;
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    const int activation_row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (output_row >= output_rows || activation_row >= activation_rows) {
        return;
    }
    float accumulator = 0.0f;
    for (int base = lane * 4; base < padded_width; base += 32 * 4) {
        const int group = base / 32;
        const int offset = base & 31;
        const size_t block = static_cast<size_t>(output_row) * groups + group;
        const int weight_codes = *reinterpret_cast<const int *>(
            weight + block * 32 + offset);
        const int activation_codes = *reinterpret_cast<const int *>(
            activation + static_cast<size_t>(activation_row) * padded_width +
            base);
        accumulator += __half2float(weight_scale[block]) *
            activation_scale[
                static_cast<size_t>(activation_row) * groups + group] *
            static_cast<float>(__dp4a(weight_codes, activation_codes, 0));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(
            0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
        output[static_cast<size_t>(activation_row) * output_rows +
            output_row] = __float2half_rn(accumulator);
    }
}








}  // namespace

void tyloi::cuda::nint_raw::group_quantize(
        const GroupQuantize & args, int total, cudaStream_t stream) {
    nint_group_quantize_kernel<<<dim3(total,args.rows),32,0,stream>>>(args);
}

void tyloi::cuda::nint_raw::quantize(const Quantize & a, cudaStream_t stream) {
    const auto launch = [&](auto cached_sum, auto columns) {
        nint_quantize_activation_kernel<decltype(cached_sum)::value,decltype(columns)::value><<<
            dim3(a.rows,a.groups),a.group_size<=32?32:64,0,stream>>>(
                a.input,a.gate,a.codes,a.scale,a.rows,a.width,a.padded_width,
                a.groups,a.group_size,a.activation_mode,a.input_float,a.gate_float,
                a.sums,a.interleaved);
    };
    const auto select_columns = [&](auto cached_sum) {
        if (!a.interleave) launch(cached_sum,std::integral_constant<int,0>{});
        else if(a.columns==2) launch(cached_sum,std::integral_constant<int,2>{});
        else if(a.columns==4) launch(cached_sum,std::integral_constant<int,4>{});
        else if(a.columns==5) launch(cached_sum,std::integral_constant<int,5>{});
        else if(a.columns==6) launch(cached_sum,std::integral_constant<int,6>{});
        else launch(cached_sum,std::integral_constant<int,8>{});
    };
    if(a.cached_sums) select_columns(std::true_type{});
    else select_columns(std::false_type{});
}

void tyloi::cuda::nint_raw::trace_group_quantize(
        const GroupQuantize & args, cudaStream_t stream) {
    if(tyloi::cuda::dispatch_trace::enabled())tyloi::cuda::dispatch_trace::record(
        stream,"nint_group_quantize",reinterpret_cast<const void*>(nint_group_quantize_kernel),
        "M="+std::to_string(args.rows)+" width="+std::to_string(args.width)+" formats="+std::to_string(args.count));
}

void tyloi::cuda::nint_raw::zero_decode(
        const std::uint8_t * quantized, const void * scale, void * output,
        int rows, int groups, std::int64_t width, bool output_float, cudaStream_t stream) {
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    if (output_float) {
        nint8_zero_decode_rows_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const int8_t *>(quantized),
            static_cast<const __half *>(scale), static_cast<float *>(output),
            rows, groups, static_cast<int>(width));
    } else {
        nint8_zero_decode_rows_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const int8_t *>(quantized),
            static_cast<const __half *>(scale), static_cast<__half *>(output),
            rows, groups, static_cast<int>(width));
    }
}

void tyloi::cuda::nint_raw::zero_gemv(
        const std::uint8_t *quantized, const void *scale, const void *input,
        std::int8_t *workspace, float *input_scale, void *output,
        int rows, int outputs, int groups, int width, bool input_float,
        cudaStream_t stream) {
    const int padded_width = groups * 32;
    nint_quantize_activation_kernel<false><<<dim3(rows, groups), 32, 0, stream>>>(
        input, nullptr, workspace, input_scale, rows, width, padded_width,
        groups, 32, 0, input_float, false);
    nint8_zero_matmul_kernel<<<dim3((outputs + 3) / 4, rows), dim3(32, 4), 0, stream>>>(
        reinterpret_cast<const int8_t *>(quantized), static_cast<const __half *>(scale),
        workspace, input_scale, static_cast<__half *>(output), rows, outputs,
        groups, padded_width);
}

void tyloi::cuda::nint_raw::one_quantize(const void *input,
        std::int8_t *quantized, void *scale, void *sum, void *reconstructed,
        int rows, int width, int groups, cudaStream_t stream) {
    nint8_one_quantize_reconstruct_kernel<<<rows * groups, 32, 0, stream>>>(
        static_cast<const __half *>(input), quantized, static_cast<__half *>(scale),
        static_cast<__half *>(sum), static_cast<__half *>(reconstructed), rows, width, groups);
}

tyloi::cuda::nint_raw::GemmStatus tyloi::cuda::nint_raw::gemm_nt(
        cublasHandle_t handle, cudaStream_t stream, const void *input,
        const void *weight, void *output, int rows, int outputs, int width,
        bool float_accum, bool float_output) {
    const auto status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return {status, status};
    const __half half_alpha = __float2half(1.0f), half_beta = __float2half(0.0f);
    const float float_alpha = 1.0f, float_beta = 0.0f;
    const void *alpha = float_accum ? static_cast<const void *>(&float_alpha) : &half_alpha;
    const void *beta = float_accum ? static_cast<const void *>(&float_beta) : &half_beta;
    const auto result = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        outputs, rows, width, alpha, weight, CUDA_R_16F, width,
        input, CUDA_R_16F, width, beta, output,
        float_output ? CUDA_R_32F : CUDA_R_16F, outputs,
        float_accum ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return {status, result};
}

void tyloi::cuda::nint_raw::decode(const Decode & a, cudaStream_t stream) {
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(a.rows) * a.width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    if (!a.output_float && (a.group_size == 24 || a.group_size == 28) &&
            a.width % 4 == 0 &&
            (reinterpret_cast<std::uintptr_t>(a.bitstream) & 3u) == 0 &&
            !a.dense_reference) {
        const auto launch = [&](auto gs) {
            nint_decode_vec4_kernel<decltype(gs)::value><<<
                dim3((a.width / 4 + 127) / 128, std::min(a.rows, 65535)),
                128, 0, stream>>>(
                a.bitstream, a.row_q_bits, a.row_q_bit_offsets,
                a.subgroup_scale, a.subgroup_minimum, a.neuron_scale,
                a.neuron_minimum, static_cast<__half *>(a.output), a.rows,
                a.groups, int(a.width), a.bytes);
        };
        if (a.group_size == 24) launch(std::integral_constant<int, 24>{});
        else launch(std::integral_constant<int, 28>{});
        return;
    }
    if (a.output_float) {
        nint_decode_rows_kernel<float><<<blocks, threads, 0, stream>>>(
            a.bitstream, a.row_q_bits, a.row_q_bit_offsets, a.subgroup_scale,
            a.subgroup_minimum, a.neuron_scale, a.neuron_minimum,
            static_cast<float *>(a.output), a.rows, a.groups, a.group_size,
            static_cast<int>(a.width));
    } else {
        nint_decode_rows_kernel<__half><<<blocks, threads, 0, stream>>>(
            a.bitstream, a.row_q_bits, a.row_q_bit_offsets, a.subgroup_scale,
            a.subgroup_minimum, a.neuron_scale, a.neuron_minimum,
            static_cast<__half *>(a.output), a.rows, a.groups, a.group_size,
            static_cast<int>(a.width));
    }
}

__global__ void nint_online_reduce_kernel(const float* partial,__half* output,int elements,int splits) {
    for(int index=int(blockIdx.x)*blockDim.x+threadIdx.x;index<elements;index+=blockDim.x*gridDim.x) {
        float sum=0;
        for(int split=0;split<splits;++split)sum+=partial[size_t(split)*elements+index];
        output[index]=__float2half_rn(sum);
    }
}

void tyloi::cuda::nint_raw::online(const Decode & a, const void *input,
        void *output, float *partial, int rows, int split, cudaStream_t stream) {
    const int output_tiles=(a.rows+kNintPrefillTileN-1)/kNintPrefillTileN;
    const auto launch=[&](auto row_tile) {
        nint_matmul_tiled_prefill_kernel<decltype(row_tile)::value,true,false,true><<<
            dim3(output_tiles,split),dim3(32,kNintPrefillWarps),0,stream>>>(
                a.bitstream,a.row_q_bits,a.row_q_bit_offsets,a.subgroup_scale,
                a.subgroup_minimum,a.neuron_scale,a.neuron_minimum,
                static_cast<const __half*>(input),static_cast<__half*>(output),
                nullptr,nullptr,nullptr,nullptr,nullptr,
                1,1,a.rows,a.groups,a.group_size,int(a.width),0,0,0,true,rows,split,partial);
    };
    if(rows<=16)launch(std::integral_constant<int,16>{});
    else launch(std::integral_constant<int,32>{});
    if(split>1)nint_online_reduce_kernel<<<(rows*a.rows+255)/256,256,0,stream>>>(
        partial,static_cast<__half*>(output),rows*a.rows,split);
}

void tyloi::cuda::nint_raw::q8(const Q8 & a, cudaStream_t stream) {
    const auto & w=a.weight;
    if(a.grouped) {
        const auto launch_short=[&](auto rows) {
            if(a.q8_decode>=2) {
                const NintWholeArgs args{w.bitstream,w.row_q_bits,w.subgroup_scale,
                    w.subgroup_minimum,w.row_q_bit_offsets,w.neuron_scale,w.neuron_minimum,
                    a.input,a.input_scale,static_cast<__half*>(a.output),w.bytes,
                    w.rows,w.groups,a.rows,false,a.sums};
                nint_launch_grouped_q8(args,stream);
            } else {
                nint_matmul_aligned_q8_kernel<decltype(rows)::value,48><<<
                    (w.rows+3)/4,dim3(32,4),0,stream>>>(
                    w.bitstream,w.row_q_bit_offsets,w.subgroup_scale,w.subgroup_minimum,
                    w.neuron_scale,w.neuron_minimum,a.input,a.input_scale,
                    static_cast<__half*>(a.output),a.rows,w.rows,w.groups,a.padded_width,48);
            }
        };
        switch(a.rows) {
        case 1:launch_short(std::integral_constant<int,1>{});break;
        case 2:launch_short(std::integral_constant<int,2>{});break;
        case 3:launch_short(std::integral_constant<int,3>{});break;
        case 4:launch_short(std::integral_constant<int,4>{});break;
        case 5:launch_short(std::integral_constant<int,5>{});break;
        case 6:launch_short(std::integral_constant<int,6>{});break;
        default:launch_short(std::integral_constant<int,8>{});break;
        }
    } else if(a.rows==1 && a.ordered) {
        nint_q8_ordered_kernel<<<w.rows,dim3(32,4),0,stream>>>(
            w.bitstream,w.row_q_bit_offsets,w.subgroup_scale,w.subgroup_minimum,
            w.neuron_scale,w.neuron_minimum,a.input,a.input_scale,
            static_cast<__half*>(a.output),w.rows,w.groups);
    } else if(a.rows==1) {
        nint_matmul_aligned_q8_kernel<1><<<dim3((w.rows+3)/4),dim3(32,4),0,stream>>>(
            w.bitstream,w.row_q_bit_offsets,w.subgroup_scale,w.subgroup_minimum,
            w.neuron_scale,w.neuron_minimum,a.input,a.input_scale,
            static_cast<__half*>(a.output),a.rows,w.rows,w.groups,a.padded_width,w.group_size);
    } else {
        nint_matmul_aligned_q8_kernel<8><<<dim3((w.rows+3)/4),dim3(32,4),0,stream>>>(
            w.bitstream,w.row_q_bit_offsets,w.subgroup_scale,w.subgroup_minimum,
            w.neuron_scale,w.neuron_minimum,a.input,a.input_scale,
            static_cast<__half*>(a.output),a.rows,w.rows,w.groups,a.padded_width,w.group_size);
    }
}



bool nint_late_scale_enabled(int64_t input_width,int bits,int /*device*/) {
    if(const auto* setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_LATE_SCALE"))return setting[0]!='0';
    if(input_width<2048 || bits<5 || bits>6)return false;
    return true;
}

void tyloi::cuda::nint_raw::mixed(const Mixed & a, cudaStream_t stream) {
    const auto launch=[&](auto lanes,auto blocks) {
    nint_mixed_routes_kernel<decltype(lanes)::value,decltype(blocks)::value><<<dim3(unsigned((a.outputs+7)/8),unsigned(a.pairs)),dim3(32,4),0,
        stream>>>(a.pointers,a.params,
        a.expert_pool,a.expert_local,a.ids,
        static_cast<__half*>(a.output),int(a.pairs),int(a.routes),int(a.experts),
        int(a.outputs),a.routed_input,a.multi_sum);
    };
    const char* grouped=a.grouped;
    const char* lanes=a.lanes;
    const char* blocks=a.blocks;
    const char* rows=a.rows;
    const char* hint=a.hint;
    if((!grouped || grouped[0]!='0') && (!hint || hint[0]!='0') &&
            (a.group_hint==24 || a.group_hint==28) && a.bits_hint>=4 && a.bits_hint<=6) {
        const auto hinted_warps=[&](auto gs,auto bits,auto warps) {
            constexpr int rows_per_block=decltype(warps)::value==4?4:8;
            const auto launch_scale=[&](auto late) {
            nint_mixed_routes_hinted_kernel<decltype(gs)::value,decltype(bits)::value,decltype(warps)::value,false,decltype(late)::value>
                <<<dim3(unsigned((a.outputs+rows_per_block-1)/rows_per_block),unsigned(a.pairs)),dim3(32,4),0,stream>>>(
                    a.pointers,a.params,a.expert_pool,
                    a.expert_local,a.ids,static_cast<__half*>(a.output),
                    int(a.pairs),int(a.routes),int(a.experts),int(a.outputs),a.routed_input,{},a.multi_sum);
            };
            if(a.late_scale)
                launch_scale(std::true_type{});
            else launch_scale(std::false_type{});
        };
        const auto hinted=[&](auto gs,auto bits) {
            const char* warps=a.warps;
            if(warps && std::atoi(warps)==4)hinted_warps(gs,bits,std::integral_constant<int,4>{});
            else hinted_warps(gs,bits,std::integral_constant<int,1>{});
        };
        const auto select_bits=[&](auto gs) {
            if(a.bits_hint==4)hinted(gs,std::integral_constant<int,4>{});
            else if(a.bits_hint==5)hinted(gs,std::integral_constant<int,5>{});
            else hinted(gs,std::integral_constant<int,6>{});
        };
        if(a.group_hint==24)select_bits(std::integral_constant<int,24>{});
        else select_bits(std::integral_constant<int,28>{});
    }
    else if(rows && std::atoi(rows)==1 && (!grouped || grouped[0]!='0')) {
        nint_mixed_routes_kernel<32,1,1><<<dim3(unsigned((a.outputs+3)/4),unsigned(a.pairs)),dim3(32,4),0,
            stream>>>(a.pointers,a.params,
            a.expert_pool,a.expert_local,a.ids,
            static_cast<__half*>(a.output),int(a.pairs),int(a.routes),int(a.experts),
            int(a.outputs),a.routed_input,a.multi_sum);
    }
    else if(grouped && grouped[0]=='0')launch(std::integral_constant<int,0>{},std::integral_constant<int,1>{});
    else if(lanes && std::atoi(lanes)==16)launch(std::integral_constant<int,16>{},std::integral_constant<int,1>{});
    // Short reductions gain from more resident warps. Long reductions keep
    // more registers for independent packed loads and instruction overlap.
    else if((blocks && std::atoi(blocks)==8) || (!blocks && a.input_width<=768 && a.outputs>=1024))
        launch(std::integral_constant<int,32>{},std::integral_constant<int,8>{});
    else launch(std::integral_constant<int,32>{},std::integral_constant<int,1>{});

}



void tyloi::cuda::nint_raw::single(NintSingleRouteWeight weight,
        const std::int32_t *expert_local, const std::int32_t *ids, void *output,
        int pairs, int routes, int experts, int outputs, bool routed_input,
        int bits_hint, bool packed_metadata, bool multi_sum, bool late_scale,
        cudaStream_t stream) {    const auto launch=[&](auto gs,auto bits) {
        const auto launch_scale=[&](auto late) {
        const auto launch_metadata=[&](auto packed) {
        nint_mixed_routes_hinted_kernel<decltype(gs)::value,decltype(bits)::value,1,true,decltype(late)::value,decltype(packed)::value>
            <<<dim3(unsigned((outputs+7)/8),unsigned(pairs)),dim3(32,4),0,stream>>>(
                nullptr,nullptr,nullptr,expert_local,ids,
                static_cast<__half*>(output),int(pairs),int(routes),
                int(experts),int(outputs),routed_input,weight,multi_sum);
        };
        if(packed_metadata) launch_metadata(std::true_type{});
        else launch_metadata(std::false_type{});
        };
        if(late_scale)
            launch_scale(std::true_type{});
        else launch_scale(std::false_type{});
    };
    const auto select_bits=[&](auto gs) {
        if(bits_hint==4)launch(gs,std::integral_constant<int,4>{});
        else if(bits_hint==5)launch(gs,std::integral_constant<int,5>{});
        else launch(gs,std::integral_constant<int,6>{});
    };
    if(weight.geometry[1]==24)select_bits(std::integral_constant<int,24>{});
    else select_bits(std::integral_constant<int,28>{});


}


void tyloi::cuda::nint_raw::routed(const Routed & a, cudaStream_t stream) {
    const auto tokens = a.tokens;
    const auto routes = a.routes;
    const auto experts = a.experts;
    const auto output_rows = a.output_rows;
    const auto groups = a.groups;
    const auto padded_width = a.padded_width;
    const auto input_width = a.input_width;
    const auto group_size = a.group_size;
    const auto q_expert_stride = a.q_expert_stride;
    const auto route_tile_m = a.route_tile_m;
    const auto pool_phase = a.pool_phase;
    const auto epilogue_mode = a.epilogue_mode;
    const auto route_map_ready = a.route_map_ready;
    const auto masked_experts = a.masked_experts;
    const auto routed_input = a.routed_input;
    constexpr int warps_per_block = 4;
    constexpr int rows_per_warp = 2;
    const int rows_per_task = epilogue_mode == 0 ? rows_per_warp : 1;
    const int result_rows = epilogue_mode == 0
        ? output_rows
        : output_rows / 2;
    const int row_blocks =
        (result_rows + rows_per_task - 1) / rows_per_task;
    const bool use_compact = tokens > 8 && route_map_ready;
    const bool use_tiled_prefill = use_compact &&
        (route_tile_m == 16 || route_tile_m == 32 ||
         route_tile_m == 64 || route_tile_m == 128);
    if (use_tiled_prefill) {
        constexpr int output_tile = kNintPrefillTileN;
        const int output_tiles =
            (result_rows + output_tile - 1) / output_tile;
        const int pairs = tokens * routes;
        const int64_t maximum_tiles =
            (pairs + route_tile_m - 1) / route_tile_m + experts;
        const int64_t maximum_tasks = maximum_tiles * output_tiles;
        int block_cap = pairs >= 32768 ? 8192 : 4096;
        const int scaled_block_cap = static_cast<int>(
            std::min<int64_t>(INT_MAX, (maximum_tasks + 3) / 4));
        block_cap = std::max(block_cap, scaled_block_cap);
        if (masked_experts && maximum_tasks > block_cap) {
            const int routed_rows_per_expert = std::max(
                1, (pairs + experts - 1) / experts);
            const int task_period =
                ((routed_rows_per_expert + route_tile_m - 1) /
                    route_tile_m) * output_tiles;
            if (task_period > 0 && block_cap % task_period == 0) {
                block_cap += task_period;
            }
        }
        const int blocks = static_cast<int>(std::max<int64_t>(
            1, std::min<int64_t>(maximum_tasks, block_cap)));
#define TYLOINFER_LAUNCH_NINT_TILED_PREFILL(TILE_M, FUSED_GLU) \
        nint_matmul_tiled_prefill_kernel<TILE_M, true, FUSED_GLU><<< \
            blocks, dim3(32, kNintPrefillWarps), 0, stream>>>( \
                a.bitstream, \
                a.row_q_bits, \
                a.row_q_bit_offsets, \
                a.subgroup_scale, \
                a.subgroup_minimum, \
                a.neuron_scale, \
                a.neuron_minimum, \
                static_cast<const __half *>(a.input), \
                static_cast<__half *>(a.output), \
                a.expert_local, \
                a.ids_dst, \
                a.expert_bounds, \
                a.tile_bounds, \
                a.tile_experts, \
                routes, experts, output_rows, groups, group_size, \
                input_width, q_expert_stride, pool_phase, epilogue_mode, \
                routed_input, 0, 1, nullptr)
        if (route_tile_m == 128 && epilogue_mode == 0) {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(128, false);
        } else if (route_tile_m == 128) {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(128, true);
        } else if (route_tile_m == 32 && epilogue_mode == 0) {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(32, false);
        } else if (route_tile_m == 32) {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(32, true);
        } else if (route_tile_m == 16 && epilogue_mode == 0) {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(16, false);
        } else if (route_tile_m == 16) {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(16, true);
        } else if (epilogue_mode == 0) {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(64, false);
        } else {
            TYLOINFER_LAUNCH_NINT_TILED_PREFILL(64, true);
        }
#undef TYLOINFER_LAUNCH_NINT_TILED_PREFILL

        return;
    }
    const int token_blocks =
        (tokens + warps_per_block - 1) / warps_per_block;
    const int64_t maximum_tasks =
        static_cast<int64_t>(tokens) * routes * row_blocks;
    constexpr int persistent_blocks = 4096;
    const int compact_blocks = static_cast<int>(
        std::min<int64_t>(maximum_tasks, persistent_blocks));
    const dim3 grid = use_compact
        ? dim3(compact_blocks)
        : dim3(row_blocks, routes, token_blocks);
    nint_matmul_kernel<false><<<
        grid,
        dim3(32, warps_per_block), 0, stream>>>(
            a.bitstream,
            a.row_q_bits,
            a.row_q_bit_offsets,
            a.subgroup_scale,
            a.subgroup_minimum,
            a.neuron_scale,
            a.neuron_minimum,
            a.quantized_input,
            a.input_scale,
            static_cast<__half *>(a.output),
            tokens,
            output_rows,
            groups,
            padded_width,
            group_size,
            a.route_ids,
            a.expert_local,
            use_compact ? a.ids_dst : nullptr,
            use_compact ? a.expert_bounds : nullptr,
            use_compact ? a.tile_bounds : nullptr,
            use_compact ? a.tile_experts : nullptr,
            routes,
            experts,
            q_expert_stride,
            pool_phase,
            epilogue_mode,
            routed_input);

}










int tyloi::cuda::nint_raw::project(const Projection & a, cudaStream_t stream) {
    const auto & w=a.q8.weight;
    const auto aligned_q8=a.aligned_q8, whole_input=a.whole_input, cached_sums=a.cached_sums;
    const auto dense_group_setting=a.dense_group_setting, fixed_setting=a.fixed_setting;
    const int group_size=w.group_size, groups=w.groups, output_rows=w.rows;
    const int activation_rows=a.q8.rows, padded_width=a.q8.padded_width;
    const int real_width=a.real_width, uniform_bits=a.uniform_bits;
    if (aligned_q8 && group_size % 4 == 0 &&
        (reinterpret_cast<uintptr_t>(w.bitstream) & 3u) == 0) {
        auto args=a.q8;
        if(!args.grouped && activation_rows==1) {
            const char* ordered=tyloi::cuda::env_lookup("TYLOINFER_NINT_Q8_ORDERED");
            args.ordered=group_size==48 && groups<=64 && output_rows<=2048 && (!ordered || ordered[0]!='0');
        }
        q8(args,stream);    } else if ((group_size == 24 || group_size == 28) &&
            (reinterpret_cast<std::uintptr_t>(w.bitstream) & 3u) == 0 &&
            (!dense_group_setting || dense_group_setting[0] != '0') &&
            tyloi::cuda::env_lookup("TYLOINFER_NINT_DENSE_REFERENCE") == nullptr) {
        const char* whole_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_WHOLE_GROUP");
        const char* batch_whole_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_BATCH_WHOLE_GROUP");
        const auto* small_group_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_FIXED_SMALL_GROUPS");
        const bool fixed_small=(!small_group_setting || small_group_setting[0]!='0') && groups<=32 &&
            activation_rows<=6 && (!fixed_setting || fixed_setting[0]!='0') &&
            ((group_size==24 && uniform_bits==6) || (group_size==28 && uniform_bits==5));
        if((activation_rows==1 || (activation_rows<=8 && (!batch_whole_setting || batch_whole_setting[0]!='0'))) &&
                (groups>32 || fixed_small) && whole_input &&
                (!whole_setting || whole_setting[0]!='0')) {
            const auto* group_load_option=tyloi::cuda::env_lookup("TYLOINFER_NINT_GROUP_LOAD");
            const bool group_load=group_load_option && group_load_option[0]=='1';
            const NintWholeArgs args{w.bitstream,w.row_q_bits,
                w.subgroup_scale,w.subgroup_minimum,
                w.row_q_bit_offsets,w.neuron_scale,w.neuron_minimum,
                a.q8.input,a.q8.input_scale,
                static_cast<__half*>(a.q8.output),uint64_t(w.bytes),output_rows,groups,activation_rows,group_load,a.q8.sums,reinterpret_cast<const int*>(a.interleaved)};
            const auto launch_whole=[&](auto gs,auto rows) {
                // Preserve the original group-to-lane map and four-term XOR
                // tree. A projection with at most 96 groups has no work for
                // warp 3; its contribution is the original positive zero.
                const auto* short_warps=tyloi::cuda::env_lookup("TYLOINFER_NINT_SHORT_WARPS");
                const auto* staged=tyloi::cuda::env_lookup("TYLOINFER_NINT_PACKED_STAGE");
                const auto* warp_rows=tyloi::cuda::env_lookup("TYLOINFER_NINT_WARP_ROWS");
                // Two-token projections benefit from four rows per CTA. At
                // six tokens the extra live accumulators regress, so the
                // default remains the original three-warps-per-row there.
                const bool tracing=tyloi::cuda::dispatch_trace::enabled();
                const auto details=tracing?"M="+std::to_string(activation_rows)+" N="+std::to_string(output_rows)+
                    " gs="+std::to_string(group_size)+" groups="+std::to_string(groups)+" capacity="+std::to_string(decltype(rows)::value):std::string{};
                if(nint_launch_fixed_decode(args,int(group_size),uniform_bits,stream))return;
                const auto launch_cached=[&](auto sum_constant) {
                constexpr bool CachedSum=decltype(sum_constant)::value;
                if(groups<=96 && (warp_rows?warp_rows[0]=='1':activation_rows==2)) {
                    nint_whole_warp_rows_kernel<decltype(gs)::value,decltype(rows)::value,CachedSum><<<(output_rows+3)/4,dim3(32,4),0,stream>>>(args);
                    if(tracing)tyloi::cuda::dispatch_trace::record(stream,"nint_whole_warp_rows",
                        reinterpret_cast<const void*>(nint_whole_warp_rows_kernel<decltype(gs)::value,decltype(rows)::value,CachedSum>),details);
                }else if(groups<=96 && staged && staged[0]=='1' &&
                    (reinterpret_cast<uintptr_t>(w.bitstream)&15u)==0) {
                    nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,true,CachedSum><<<output_rows,dim3(32,3),0,stream>>>(args);
                    if(tracing)tyloi::cuda::dispatch_trace::record(stream,"nint_packed_stage",
                        reinterpret_cast<const void*>(nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,true,CachedSum>),details);
                }else if(groups<=96 && (!short_warps || short_warps[0]!='0')) {
                    nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,false,CachedSum><<<output_rows,dim3(32,3),0,stream>>>(args);
                    if(tracing)tyloi::cuda::dispatch_trace::record(stream,"nint_whole_group_3warp",
                        reinterpret_cast<const void*>(nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,3,false,CachedSum>),details);
                }else {
                    nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,4,false,CachedSum><<<output_rows,dim3(32,4),0,stream>>>(args);
                    if(tracing)tyloi::cuda::dispatch_trace::record(stream,"nint_whole_group_4warp",
                        reinterpret_cast<const void*>(nint_whole_group_kernel<decltype(gs)::value,decltype(rows)::value,4,false,CachedSum>),details);
                }
                };
                if(cached_sums)launch_cached(std::true_type{});else launch_cached(std::false_type{});
            };
            const auto select_whole_rows=[&](auto gs) {
                if(activation_rows==1)launch_whole(gs,std::integral_constant<int,1>{});
                else if(activation_rows<=2)launch_whole(gs,std::integral_constant<int,2>{});
                else if(activation_rows<=4)launch_whole(gs,std::integral_constant<int,4>{});
                else if(const auto* exact=tyloi::cuda::env_lookup("TYLOINFER_NINT_EXACT_BATCH");!exact || exact[0]!='0') {
                    if(activation_rows==5)launch_whole(gs,std::integral_constant<int,5>{});
                    else if(activation_rows==6)launch_whole(gs,std::integral_constant<int,6>{});
                    else launch_whole(gs,std::integral_constant<int,8>{});
                }
                else launch_whole(gs,std::integral_constant<int,8>{});
            };
            if(group_size==24)select_whole_rows(std::integral_constant<int,24>{});
            else select_whole_rows(std::integral_constant<int,28>{});
        } else {
        const auto launch_dense = [&](auto gs, auto rows, auto warps) {
            constexpr int rows_per_block = 4 / decltype(warps)::value;
            nint_matmul_dense_kernel<decltype(gs)::value, decltype(rows)::value, decltype(warps)::value><<<
                (output_rows + rows_per_block - 1) / rows_per_block, dim3(32, 4), 0, stream>>>(
                w.bitstream, w.row_q_bits,
                w.row_q_bit_offsets, w.subgroup_scale,
                w.subgroup_minimum, w.neuron_scale,
                w.neuron_minimum, a.q8.input,
                a.q8.input_scale, static_cast<__half*>(a.q8.output),
                activation_rows, output_rows, groups, uint64_t(w.bytes));
        };
        const auto select_warps = [&](auto gs, auto rows) {
            if (groups <= 32) launch_dense(gs, rows, std::integral_constant<int, 1>{});
            else launch_dense(gs, rows, std::integral_constant<int, 4>{});
        };
        const auto select_rows = [&](auto gs) {
            if (activation_rows == 1) select_warps(gs, std::integral_constant<int, 1>{});
            else if (activation_rows <= 2) select_warps(gs, std::integral_constant<int, 2>{});
            else if (activation_rows <= 4) select_warps(gs, std::integral_constant<int, 4>{});
            else select_warps(gs, std::integral_constant<int, 8>{});
        };
        if (group_size == 24) select_rows(std::integral_constant<int, 24>{});
        else select_rows(std::integral_constant<int, 28>{});
        }
    } else {
    const auto launch=[&](auto single,auto grouped) {
    const auto launch_rows=[&](auto rows) {
    nint_matmul_kernel<decltype(single)::value,decltype(grouped)::value,decltype(rows)::value><<<
        dim3((output_rows + 3) / 4), dim3(32, 4), 0, stream>>>(
            w.bitstream,
            w.row_q_bits,
            w.row_q_bit_offsets,
            w.subgroup_scale,
            w.subgroup_minimum,
            w.neuron_scale,
            w.neuron_minimum,
            a.q8.input,
            a.q8.input_scale,
            static_cast<__half *>(a.q8.output),
            activation_rows,
            output_rows,
            groups,
            padded_width,
            static_cast<int>(group_size),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0,
            0,
            0,
            0,
            false,static_cast<uint64_t>(w.bytes));
    };
    const auto* exact=tyloi::cuda::env_lookup("TYLOINFER_NINT_EXACT_BATCH");
    if constexpr(!decltype(single)::value) {
        if(!exact || exact[0]!='0') {
            if(activation_rows==1)launch_rows(std::integral_constant<int,1>{});
            else if(activation_rows<=2)launch_rows(std::integral_constant<int,2>{});
            else if(activation_rows<=4)launch_rows(std::integral_constant<int,4>{});
            else if(activation_rows==5)launch_rows(std::integral_constant<int,5>{});
            else if(activation_rows==6)launch_rows(std::integral_constant<int,6>{});
            else launch_rows(std::integral_constant<int,8>{});
            return;
        }
    }
    launch_rows(std::integral_constant<int,8>{});
    };
    const auto* single=tyloi::cuda::env_lookup("TYLOINFER_NINT_SINGLE_ROW");
    if (activation_rows == 1 && single && std::strcmp(single, "1") == 0 &&
            (reinterpret_cast<std::uintptr_t>(w.bitstream) & 3u) == 0) {
        const size_t activation_bytes =
            (static_cast<size_t>(padded_width) + 3u) & ~size_t(3u);
        const size_t shared_bytes = activation_bytes +
            static_cast<size_t>(groups) * sizeof(float);
        bool use_cache = false;
        int cache_warps = 1;
        if (real_width > output_rows) {
            // Query the actual device limit once per host thread/device.
            thread_local int cached_device = -1;
            thread_local int shared_limit = 0;
            thread_local int processor_count = 0;
            const int device = a.device;
            if (cached_device != device) {
                if(cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device) != cudaSuccess) return 1;
                if(cudaDeviceGetAttribute(&processor_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || processor_count <= 0) return 2;
                cached_device = device;
            }
            const int rows_per_processor = (output_rows - 1) / processor_count + 1;
            while (cache_warps < rows_per_processor && cache_warps < 32) {
                cache_warps *= 2;
            }
            use_cache = cache_warps > 1 && shared_bytes <= static_cast<size_t>(shared_limit);
        }
        // At least one warp of groups avoids underfilled lanes.
        const auto* group_flag=tyloi::cuda::env_lookup("TYLOINFER_NINT_GROUP_DOT");
        const bool use_group=(!group_flag || std::strcmp(group_flag,"1")==0) &&
            groups>=32 && group_size>=1 && group_size<=64;
        if (use_cache) {
            const auto launch_cache=[&](auto grouped) {
            nint_matmul_activation_cache_kernel<decltype(grouped)::value><<<
                dim3((output_rows + cache_warps - 1) / cache_warps),
                dim3(32, cache_warps), shared_bytes, stream>>>(
                    w.bitstream, w.row_q_bits,
                    w.row_q_bit_offsets,
                    w.subgroup_scale,
                    w.subgroup_minimum,
                    w.neuron_scale, w.neuron_minimum,
                    a.q8.input, a.q8.input_scale,
                    static_cast<__half *>(a.q8.output),
                    padded_width, groups, static_cast<int>(group_size), output_rows,
                    static_cast<uint64_t>(w.bytes));
            };
            if(use_group)launch_cache(std::true_type{});
            else launch_cache(std::false_type{});
        } else {
            if(use_group)launch(std::true_type{},std::true_type{});
            else launch(std::true_type{},std::false_type{});
        }
    } else {
        launch(std::false_type{},std::false_type{});
    }
    }

    return 0;
}
