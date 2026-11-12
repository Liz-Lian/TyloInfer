#pragma once

#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::activation_raw {

using Type = NumericType;

/** Launch elementwise SiLU gate multiplication. */
void silu_mul(const void* gate, const void* up, void* output,
    std::size_t elements, Type type, cudaStream_t stream);
/** Launch elementwise tanh-GELU gate multiplication. */
void gelu_mul(const void* gate, const void* up, void* output,
    std::size_t elements, Type type, cudaStream_t stream);
/** Launch clipped FP16 SiLU gate multiplication. */
void clipped_silu_mul(const void* gate, const void* up, void* output,
    std::int64_t elements, float limit, cudaStream_t stream);
/** Launch squared ReLU. */
void relu_squared(const void* input, void* output, std::size_t elements,
    Type type, cudaStream_t stream);
/** Launch strided linear-attention gate and beta calculation. */
void linear_gate_beta(const void* alpha, const void* beta, const float* dt_bias,
    const float* a_log, float* gate, float* beta_out, std::int64_t batch,
    std::int64_t tokens, std::int64_t width, const std::int64_t alpha_strides[3],
    const std::int64_t beta_strides[3], Type type, bool stable_softplus, cudaStream_t stream);

} // namespace tyloi::cuda::activation_raw
