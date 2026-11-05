#pragma once
#include "tyloinfer/gguf_quant.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::gguf {
// Native Q8_1 activations, canonical weights, and no persistent workspace.
// Unsupported formats/windows retain the ordinary projection dispatcher.
bool project_dense_window_q8(ggml_type type, const void* packed, std::uint64_t stride,
    std::int64_t rows, std::int64_t columns, const void* input, std::int64_t tokens,
    void* output, bool half_output, cudaStream_t stream);
// Full-window canonical Q5/Q6 use the same split activation packet as IQ2.
// Eligibility follows actual block geometry/alignment, independently of GPU.
bool supports_dense_split_window_q8(ggml_type,const void*,std::uint64_t,std::int64_t,std::int64_t,std::int64_t) noexcept;
bool project_dense_split_window_q8(ggml_type,const void*,std::uint64_t,std::int64_t,std::int64_t,const void*,std::int64_t,void*,bool,cudaStream_t);

}
