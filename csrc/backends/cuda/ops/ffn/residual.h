#pragma once

#include "ops/ffn/ffn.h"
#include "ops/attention/full.h"
#include "ops/attention/linear.h"

namespace mfq::cuda {

mfq_tensor_backend::Tensor full_ffn(
    CudaExecutionContext& execution, const FFN& ffn,
    const mfq_tensor_backend::Tensor& ffn_norm, const FullAttention& attention,
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor output);

mfq_tensor_backend::Tensor linear_ffn(
    CudaExecutionContext& execution, const FFN& ffn,
    const mfq_tensor_backend::Tensor& ffn_norm, const LinearAttention& attention,
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor output);

} // namespace mfq::cuda
