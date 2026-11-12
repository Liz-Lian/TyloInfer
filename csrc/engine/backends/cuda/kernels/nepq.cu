

#include <cuda.h>
#include "nepq_raw.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>


namespace {

__global__ void nepq_hadamard_input_kernel(
    const __half * input,
    const int8_t * signs,
    __half * output,
    int rows,
    int width,
    int block_size) {
    extern __shared__ float values[];
    const int row = blockIdx.y;
    const int offset = blockIdx.x * block_size;
    if (row >= rows || offset >= width) return;

    for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
        const int column = offset + index;
        values[index] = __half2float(input[static_cast<int64_t>(row) * width + column])
            * static_cast<float>(signs[column]);
    }
    __syncthreads();

    for (int half = 1; half < block_size; half <<= 1) {
        const int span = half << 1;
        for (int butterfly = threadIdx.x;
             butterfly < block_size / 2;
             butterfly += blockDim.x) {
            const int group = butterfly / half;
            const int within = butterfly - group * half;
            const int first = group * span + within;
            const int second = first + half;
            const float a = values[first];
            const float b = values[second];
            values[first] = a + b;
            values[second] = a - b;
        }
        __syncthreads();
    }

    const float normalization = rsqrtf(static_cast<float>(block_size));
    for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
        const int column = offset + index;
        output[static_cast<int64_t>(row) * width + column] =
            __float2half(values[index] * normalization);
    }
}

__global__ void nepq_hadamard_adjoint_kernel(
    const __half * input,
    const int8_t * signs,
    __half * output,
    int rows,
    int width,
    int block_size) {
    extern __shared__ float values[];
    const int row = blockIdx.y;
    const int offset = blockIdx.x * block_size;
    if (row >= rows || offset >= width) return;

    for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
        const int column = offset + index;
        values[index] = __half2float(
            input[static_cast<int64_t>(row) * width + column]);
    }
    __syncthreads();

    for (int half = 1; half < block_size; half <<= 1) {
        const int span = half << 1;
        for (int butterfly = threadIdx.x;
             butterfly < block_size / 2;
             butterfly += blockDim.x) {
            const int group = butterfly / half;
            const int within = butterfly - group * half;
            const int first = group * span + within;
            const int second = first + half;
            const float a = values[first];
            const float b = values[second];
            values[first] = a + b;
            values[second] = a - b;
        }
        __syncthreads();
    }

    const float normalization = rsqrtf(static_cast<float>(block_size));
    for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
        const int column = offset + index;
        output[static_cast<int64_t>(row) * width + column] = __float2half(
            values[index] * normalization * static_cast<float>(signs[column]));
    }
}

}  // namespace


namespace tyloi::cuda::nepq_raw {

void input(const void* source, const int8_t* signs, void* output,
           int rows, int width, int block_size, cudaStream_t stream) {
    nepq_hadamard_input_kernel<<<dim3(width / block_size, rows), 256,
        static_cast<size_t>(block_size) * sizeof(float), stream>>>(
        static_cast<const __half*>(source), signs, static_cast<__half*>(output),
        rows, width, block_size);
}

void adjoint(const void* source, const int8_t* signs, void* output,
             int rows, int width, int block_size, cudaStream_t stream) {
    nepq_hadamard_adjoint_kernel<<<dim3(width / block_size, rows), 256,
        static_cast<size_t>(block_size) * sizeof(float), stream>>>(
        static_cast<const __half*>(source), signs, static_cast<__half*>(output),
        rows, width, block_size);
}

}  // namespace tyloi::cuda::nepq_raw
