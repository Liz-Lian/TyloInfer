// Dense and canonical packed embedding lookup helpers.

#include <algorithm>
#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "embedding_raw.h"


namespace {


#if TYLOINFER_ENABLE_MFQ_COMPAT
__device__ __forceinline__ uint8_t unpack_nint_code(
        const uint8_t * stream,
        uint64_t row_bit_offset,
        int element,
        int bits) {
    const uint64_t bit = row_bit_offset +
        static_cast<uint64_t>(element) * static_cast<uint64_t>(bits);
    const uint64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7u);
    uint32_t word = static_cast<uint32_t>(stream[byte]);
    if (shift + bits > 8) {
        word |= static_cast<uint32_t>(stream[byte + 1]) << 8;
    }
    return static_cast<uint8_t>(
        (word >> shift) & ((1u << bits) - 1u));
}


#endif

template <typename Scalar>
__global__ void dense_embedding_flat_kernel(
        const Scalar * __restrict__ weight,
        const int64_t * __restrict__ token_ids,
        Scalar * __restrict__ output,
        int token_count,
        int width,
        int vocabulary) {
    const size_t total = static_cast<size_t>(token_count) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int token_index = static_cast<int>(index / width);
        const int64_t token = token_ids[token_index];
        output[index] = token >= 0 && token < vocabulary
            ? weight[static_cast<size_t>(token) * width + column]
            : Scalar(0);
    }
}


template <typename Scalar>
__global__ void dense_embedding_vector_kernel(
        const Scalar * __restrict__ weight,
        const int64_t * __restrict__ token_ids,
        Scalar * __restrict__ output,
        int token_count,
        int width,
        int vocabulary) {
    constexpr int elements_per_vector = sizeof(uint4) / sizeof(Scalar);
    const int vectors_per_token = width / elements_per_vector;
    const int tiles_per_token =
        (vectors_per_token + blockDim.x - 1) / blockDim.x;
    const size_t total_tiles =
        static_cast<size_t>(token_count) * tiles_per_token;
    for (size_t tile_index = blockIdx.x;
         tile_index < total_tiles;
         tile_index += gridDim.x) {
        const int token_index = static_cast<int>(
            tile_index / tiles_per_token);
        const int tile = static_cast<int>(tile_index % tiles_per_token);
        const int vector_index = tile * blockDim.x + threadIdx.x;
        if (vector_index >= vectors_per_token) {
            continue;
        }
        const int64_t token = token_ids[token_index];
        auto* output_vectors = reinterpret_cast<uint4 *>(output);
        const auto* weight_vectors =
            reinterpret_cast<const uint4 *>(weight);
        const size_t output_index = static_cast<size_t>(token_index) *
            vectors_per_token + vector_index;
        output_vectors[output_index] = token >= 0 && token < vocabulary
            ? weight_vectors[static_cast<size_t>(token) * vectors_per_token +
                vector_index]
            : make_uint4(0, 0, 0, 0);
    }
}


#if TYLOINFER_ENABLE_MFQ_COMPAT
__global__ void nint_embedding_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int64_t * __restrict__ token_ids,
        __half * __restrict__ output,
        int token_count,
        int vocabulary,
        int groups,
        int group_size,
        int width) {
    const size_t total = static_cast<size_t>(token_count) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int token_index = static_cast<int>(index / width);
        const int64_t token = token_ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocabulary) {
            const int group = column / group_size;
            const size_t metadata = static_cast<size_t>(token) * groups + group;
            const int bits = static_cast<int>(row_q_bits[token]);
            const uint8_t code = unpack_nint_code(
                bitstream,
                static_cast<uint64_t>(row_q_bit_offsets[token]),
                column,
                bits);
            const float scale = neuron_scale[token] *
                static_cast<float>(subgroup_scale[metadata]);
            const float minimum = neuron_minimum[token] *
                static_cast<float>(subgroup_minimum[metadata]);
            value = scale * static_cast<float>(code) - minimum;
        }
        output[index] = __float2half_rn(value);
    }
}


__global__ void nint_selected_rows_kernel(
        const uint8_t* __restrict__ packed,
        const uint32_t* __restrict__ descriptors,
        const int64_t* __restrict__ inverse,
        __half* __restrict__ output, size_t total, int width) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const auto selected = inverse ? inverse[index / width] : index / width;
        const uint32_t* row = descriptors + selected * 6;
        const uint32_t layout = row[3];
        const int qbits = layout & 15u, kbits = (layout >> 4u) & 15u;
        const int group = column / row[4];
        const float q = unpack_nint_code(packed + row[0], (layout >> 8u) & 7u, column, qbits);
        const float s = unpack_nint_code(packed + row[1], (layout >> 12u) & 7u, group, kbits);
        const float m = unpack_nint_code(packed + row[2], (layout >> 16u) & 7u, group, kbits);
        const float ns = __half2float(__ushort_as_half(static_cast<unsigned short>(row[5])));
        const float nm = __half2float(__ushort_as_half(static_cast<unsigned short>(row[5] >> 16u)));
        // Match the shared/Metal reconstruction order, without contraction.
        output[index] = __float2half_rn(__fsub_rn(__fmul_rn(__fmul_rn(ns, s), q), __fmul_rn(nm, m)));
    }
}

__global__ void nint8_zero_embedding_kernel(
        const int8_t * __restrict__ quantized,
        const __half * __restrict__ scale,
        const int64_t * __restrict__ token_ids,
        __half * __restrict__ output,
        int token_count,
        int vocabulary,
        int groups,
        int width) {
    const size_t total = static_cast<size_t>(token_count) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int token_index = static_cast<int>(index / width);
        const int64_t token = token_ids[token_index];
        float value = 0.0f;
        if (token >= 0 && token < vocabulary) {
            const int group = column / 32;
            const int lane = column & 31;
            const size_t block = static_cast<size_t>(token) * groups + group;
            value = __half2float(scale[block]) * static_cast<float>(
                quantized[block * 32 + lane]);
        }
        output[index] = __float2half_rn(value);
    }
}


#endif

int launch_blocks(size_t elements) {
    constexpr int threads = 256;
    return static_cast<int>(std::min<size_t>(
        (elements + threads - 1) / threads, 4096));
}


}  // namespace

namespace tyloi::cuda::embedding_raw {
template<class T>
void dense_t(const void* weight, const int64_t* ids, void* output, int tokens, int width,
    int vocabulary, cudaStream_t stream) {
    constexpr int threads = 256;
    const auto addresses = reinterpret_cast<std::uintptr_t>(weight) | reinterpret_cast<std::uintptr_t>(output);
    if (tokens >= 64 && (addresses & (alignof(uint4) - 1)) == 0 &&
        (static_cast<size_t>(width) * sizeof(T)) % sizeof(uint4) == 0) {
        constexpr int elements = sizeof(uint4) / sizeof(T);
        const size_t vectors = static_cast<size_t>(width) / elements;
        const size_t tiles = (vectors + threads - 1) / threads;
        dense_embedding_vector_kernel<T><<<launch_blocks(size_t(tokens) * tiles * threads), threads, 0, stream>>>(
            static_cast<const T*>(weight), ids, static_cast<T*>(output), tokens, width, vocabulary);
    } else {
        dense_embedding_flat_kernel<T><<<launch_blocks(size_t(tokens) * width), threads, 0, stream>>>(
            static_cast<const T*>(weight), ids, static_cast<T*>(output), tokens, width, vocabulary);
    }
}
void dense(const void* weight, const int64_t* ids, void* output, int tokens, int width,
    int vocabulary, NumericType type, cudaStream_t stream) {
    if (type == NumericType::f16) dense_t<__half>(weight, ids, output, tokens, width, vocabulary, stream);
    else dense_t<float>(weight, ids, output, tokens, width, vocabulary, stream);
}
#if TYLOINFER_ENABLE_MFQ_COMPAT
void nint(const uint8_t* packed, const uint8_t* bits, const int64_t* offsets,
    const uint8_t* sub_scale, const uint8_t* sub_min, const float* scale, const float* minimum,
    const int64_t* ids, void* output, int tokens, int vocabulary, int groups, int group_size,
    int width, cudaStream_t stream) {
    nint_embedding_kernel<<<launch_blocks(size_t(tokens) * width), 256, 0, stream>>>(packed, bits,
        offsets, sub_scale, sub_min, scale, minimum, ids, static_cast<__half*>(output), tokens,
        vocabulary, groups, group_size, width);
}
void selected(const uint8_t* packed, const int32_t* descriptors, const int64_t* inverse,
    void* output, size_t count, int width, cudaStream_t stream) {
    nint_selected_rows_kernel<<<launch_blocks(count), 256, 0, stream>>>(packed,
        reinterpret_cast<const uint32_t*>(descriptors), inverse, static_cast<__half*>(output), count, width);
}
void nint8(const uint8_t* q, const void* scale, const int64_t* ids, void* output,
    int tokens, int vocabulary, int groups, int width, cudaStream_t stream) {
    nint8_zero_embedding_kernel<<<launch_blocks(size_t(tokens) * width), 256, 0, stream>>>(
        reinterpret_cast<const int8_t*>(q), static_cast<const __half*>(scale), ids,
        static_cast<__half*>(output), tokens, vocabulary, groups, width);
}
#endif
}
