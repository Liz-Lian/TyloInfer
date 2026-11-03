#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::gguf {
// Packed rows retain the standard Gyro encoding and a per-neuron type.
// All pointers and the launch geometry are stable during graph replay.
void add_row_residual(const void* packed,const std::uint64_t* offsets,const std::int32_t* types,
    std::int64_t first_row,std::int64_t rows,std::int64_t first_column,std::int64_t columns,
    const float* input,float* output,std::int64_t tokens,const std::int64_t* expert_ids,
    std::int64_t routes,std::int64_t experts,bool routed_input,cudaStream_t stream);
void add_residual_embedding(const void* packed,const std::uint64_t* offsets,const std::int32_t* types,
    std::int64_t first_row,std::int64_t rows,std::int64_t first_column,std::int64_t columns,
    const std::int64_t* ids,float* output,std::int64_t count,cudaStream_t stream);
}
