#pragma once

#include "engine/tensor.h"

#include <cuda_runtime_api.h>

namespace mfq::cuda {
using mfq::Device;
using mfq::DeviceType;
using mfq::ScalarType;
using mfq::TensorView;
using mfq::make_contiguous_view;
}

namespace mfq::cuda::kernels {

void silu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream);

void gelu_mul(
    const TensorView& gate,
    const TensorView& up,
    const TensorView& output,
    cudaStream_t stream);

void linear_gate_beta(
    const TensorView& alpha,
    const TensorView& beta,
    const TensorView& dt_bias,
    const TensorView& a_log,
    const TensorView& gate_t,
    const TensorView& beta_t,
    cudaStream_t stream,
    bool stable_softplus = false);

}  // namespace mfq::cuda::kernels
