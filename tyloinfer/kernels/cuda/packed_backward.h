#pragma once

#include <torch/extension.h>
#include "packed_backward_launch.h"

namespace mfq_jit_backward {
/** Run the original FP16 backward GEMM on the supplied stream. */
void launch_half_gemm_nn(const __half*, const __half*, __half*, int rows, int outputs, int width, cudaStream_t);
/** Run the original FP32 backward GEMM on the supplied stream. */
void launch_float_gemm_nn(const float*, const float*, float*, int rows, int outputs, int width, cudaStream_t);
/** Compute input gradients with the original dtype conversions and GEMM accumulation. */
void launch_dense_half_weight(const torch::Tensor&, const torch::Tensor&, torch::Tensor&,
                              int rows, int outputs, int width, cudaStream_t);
}
