

#include <cuda.h>
#include "mfq_tensor_backend.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "nepq.h"
#include "hadamard_resource.cuh"
#include <algorithm>
#include <limits>
#include <stdexcept>


namespace {

template<class T, int Block, bool Adjoint>
__global__ void hadamard_warp(const T* input, const int8_t* signs,
        T* output, int64_t count, int64_t width, int runtime_block) {
    const int lane = threadIdx.x % 32;
    for (auto group = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
            group < (count + 127) / 128; group += int64_t(gridDim.x) * 4) {
        const auto start = group * 128 + lane * 4;
        // Four adjacent values wrap at most once, including two-wide rows.
        const auto column = start % width;
        float values[4]{};
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (start + j < count) {
            values[j] = float(input[start + j]);
            if constexpr (!Adjoint)
                if (signs) values[j] *= float(signs[Block == 1 ? (column + j) % width : column + j < width ? column + j : column + j - width]);
        }
        if constexpr (Block > 1) {
        const float a = values[0] + values[1], b = values[0] - values[1];
        const float c = values[2] + values[3], d = values[2] - values[3];
        if constexpr (Block >= 4) {
            values[0] = a + c; values[1] = b + d;
            values[2] = a - c; values[3] = b - d;
        } else {
            values[0] = a; values[1] = b; values[2] = c; values[3] = d;
        }
        }
        #pragma unroll
        for (int mask = 1; mask < Block / 4; mask *= 2) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float other = __shfl_xor_sync(0xffffffffu, values[j], mask);
                values[j] = (lane & mask) ? other - values[j] : values[j] + other;
            }
        }
        // A compile-time rsqrt is folded differently by nvcc. Use the same
        // device approximation as the original dynamic-block transform so
        // borderline FP16 rounding stays bit-exact.
        const float normalization = rsqrtf(float(runtime_block));
        #pragma unroll
        for (int j = 0; j < 4; ++j) if (start + j < count) {
            float value = values[j] * normalization;
            if constexpr (Adjoint)
                if (signs) value *= float(signs[Block == 1 ? (column + j) % width : column + j < width ? column + j : column + j - width]);
            output[start + j] = T(value);
        }
    }
}

template<class T, bool Adjoint>
__global__ void hadamard_shared(const T* input, const int8_t* signs,
        T* output, int64_t groups, int64_t width, int block_size) {
    extern __shared__ float values[];
    for (auto group = int64_t(blockIdx.x); group < groups; group += gridDim.x) {
        const auto start = group * block_size;
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float value = float(input[start + i]);
            if constexpr (!Adjoint) if (signs) value *= float(signs[(start + i) % width]);
            values[i] = value;
        }
        __syncthreads();
        for (int half = 1; half < block_size; half *= 2) {
            for (int butterfly = threadIdx.x; butterfly < block_size / 2; butterfly += blockDim.x) {
                const int first = (butterfly / half) * (2 * half) + butterfly % half;
                const float a = values[first], b = values[first + half];
                values[first] = a + b; values[first + half] = a - b;
            }
            __syncthreads();
        }
        const float normalization = rsqrtf(float(block_size));
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float value = values[i] * normalization;
            if constexpr (Adjoint) if (signs) value *= float(signs[(start + i) % width]);
            output[start + i] = T(value);
        }
        __syncthreads(); // Complete stores before the next group's writes.
    }
}

template<class T, bool Adjoint>
void launch_hadamard(const T* input, const int8_t* signs, T* output,
        int64_t count, int64_t width, int block, cudaStream_t stream) {
    // CUDA guarantees this grid-x range; strides cover arbitrarily many rows.
    const unsigned grid = unsigned(std::min<int64_t>(65535, (count + 511) / 512));
    switch (block) {
#define HADAMARD_CASE(B) case B: hadamard_warp<T, B, Adjoint><<<grid, 128, 0, stream>>>(input, signs, output, count, width, block); break;
        HADAMARD_CASE(1) HADAMARD_CASE(2) HADAMARD_CASE(4) HADAMARD_CASE(8) HADAMARD_CASE(16)
        HADAMARD_CASE(32) HADAMARD_CASE(64) HADAMARD_CASE(128)
#undef HADAMARD_CASE
        default:
            mfq::cuda::hadamard::prepare_shared(reinterpret_cast<const void*>(hadamard_shared<T, Adjoint>), size_t(block) * sizeof(float));
            hadamard_shared<T, Adjoint><<<unsigned(std::min<int64_t>(65535, count / block)),
                256, size_t(block) * sizeof(float), stream>>>(input, signs, output, count / block, width, block);
    }
}

}  // namespace


namespace mfq::cuda::hadamard {
template<class T>
void apply(const T* input, const std::int8_t* signs, T* output,
        std::int64_t rows, std::int64_t width, int block, bool adjoint, cudaStream_t stream) {
    if (rows < 0 || width < 0 || block < 1 || (block & (block - 1)) || width % block ||
            (width && rows > (std::numeric_limits<int64_t>::max() - 511) / width))
        throw std::invalid_argument("Hadamard requires complete power-of-two blocks");
    const auto count = rows * width;
    if (std::uint64_t(count) > std::numeric_limits<size_t>::max() / sizeof(T))
        throw std::invalid_argument("Hadamard buffer size overflow");
    if (!count) return;
    if (!input || !output) throw std::invalid_argument("Hadamard input/output is null");
    if (adjoint) launch_hadamard<T, true>(input, signs, output, count, width, block, stream);
    else launch_hadamard<T, false>(input, signs, output, count, width, block, stream);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void apply_f16(const void* input, const std::int8_t* signs, void* output,
        std::int64_t rows, std::int64_t width, int block, bool adjoint, cudaStream_t stream) {
    apply(static_cast<const __half*>(input), signs, static_cast<__half*>(output), rows, width, block, adjoint, stream);
}
void apply_f32(const float* input, const std::int8_t* signs, float* output,
        std::int64_t rows, std::int64_t width, int block, bool adjoint, cudaStream_t stream) {
    apply(input, signs, output, rows, width, block, adjoint, stream);
}
}

namespace {
mfq_tensor_backend::Tensor transform_tensor(mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor signs, int64_t block, bool adjoint) {
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.is_contiguous() && input.dim() == 2, "NEPQ Hadamard input must be CUDA contiguous fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(signs.is_cuda() && signs.scalar_type() == mfq_tensor_backend::kInt8 &&
        signs.is_contiguous() && signs.dim() == 1, "NEPQ Hadamard signs must be CUDA contiguous int8 rank-1");
    TYLOINFER_RUNTIME_CHECK(input.device() == signs.device(), "NEPQ Hadamard tensors must share one CUDA device");
    TYLOINFER_RUNTIME_CHECK(signs.numel() == input.size(1), "NEPQ Hadamard sign width mismatch");
    TYLOINFER_RUNTIME_CHECK(block >= 2 && block <= 2048 && (block & (block - 1)) == 0,
        "NEPQ Hadamard block must be a power of two in [2,2048]");
    MfqCudaGuard guard(input.device());
    auto output = mfq_tensor_backend::empty_like(input);
    mfq::cuda::hadamard::apply_f16(input.data_ptr<mfq_half>(), signs.data_ptr<int8_t>(),
        output.data_ptr<mfq_half>(), input.size(0), input.size(1), int(block), adjoint, mfq_current_cuda_stream());
    return output;
}
}

mfq_tensor_backend::Tensor nepq_hadamard_input_cuda(mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor signs, int64_t block) {
    return transform_tensor(input, signs, block, false);
}
mfq_tensor_backend::Tensor nepq_hadamard_adjoint_cuda(mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor signs, int64_t block) {
    return transform_tensor(input, signs, block, true);
}
