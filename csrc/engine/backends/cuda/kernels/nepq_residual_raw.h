#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::nepq_residual_raw {

/** Add a sparse residual matrix product to an existing fp16 output. */
void matmul(const void* dictionary, const int16_t* first, const int16_t* second,
    const void* input, void* output, int rows, int width, int input_rows,
    int blocks_per_row, int position_bits, int block_vectors, int64_t total,
    cudaStream_t stream);

/** Add sparse residual vectors to an existing fp16 weight matrix. */
void dequant(const void* dictionary, const int16_t* first, const int16_t* second,
    void* weight, int width, int blocks_per_row, int position_bits,
    int block_vectors, int64_t total, cudaStream_t stream);

/** Add the routed sparse residual product to an existing fp16 output. */
void grouped(const void* dictionary, const int16_t* first, const int16_t* second,
    const void* input, const int32_t* route_ids, const int32_t* expert_local,
    void* output, int tokens, int routes, int out_per_expert, int width,
    int blocks_per_row, int position_bits, int block_vectors, bool routed_input,
    bool mapped_experts, int global_experts, int local_experts, int64_t total,
    cudaStream_t stream);

}  // namespace tyloi::cuda::nepq_residual_raw
