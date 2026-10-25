#pragma once

#include "engine/models/ffn.h"
#include "ops/attention/full.h"
#include "ops/attention/linear.h"

struct CudaExecutionContext;

namespace mfq::cuda {

bool uses_expert_cache(const Ffn& ffn);
bool expert_parallel(const Ffn& ffn);
void prefetch_ffn(CudaExecutionContext& execution, const Ffn& ffn,
                  const mfq_tensor_backend::Tensor& input,
                  MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt);

mfq_tensor_backend::Tensor run_ffn(
    CudaExecutionContext& execution, const Ffn& ffn, mfq_tensor_backend::Tensor input,
    MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt,
    MfqOptional<mfq_tensor_backend::Tensor> router_bias = mfq_nullopt);

mfq_tensor_backend::Tensor full_ffn(
    CudaExecutionContext& execution, const Ffn& ffn,
    const mfq_tensor_backend::Tensor& ffn_norm, const FullAttention& attention,
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor output);

mfq_tensor_backend::Tensor linear_ffn(
    CudaExecutionContext& execution, const Ffn& ffn,
    const mfq_tensor_backend::Tensor& ffn_norm, const LinearAttention& attention,
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor output);

} // namespace mfq::cuda
