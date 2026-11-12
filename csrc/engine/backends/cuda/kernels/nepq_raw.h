#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::nepq_raw {

/** Launch the signed block Hadamard input transform on the supplied stream. */
void input(const void* input, const int8_t* signs, void* output,
           int rows, int width, int block_size, cudaStream_t stream);

/** Launch the signed block Hadamard adjoint transform on the supplied stream. */
void adjoint(const void* input, const int8_t* signs, void* output,
             int rows, int width, int block_size, cudaStream_t stream);

}  // namespace tyloi::cuda::nepq_raw
