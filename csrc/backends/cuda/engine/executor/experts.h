#pragma once

#include "engine/weights/moe/experts.h"
#include "ops/common/include/moe.h"

struct CudaExecutionContext;
struct CudaExecutionConfig;

namespace mfq::cuda {

MoeOptions moe_options(const CudaExecutionConfig& config, bool prefill_off, bool pool_only);
void prefetch_experts(const Experts& weight, const MoeRoutePlan& route, bool begin = false);

mfq_tensor_backend::Tensor run_experts(
    CudaExecutionContext& execution, const Experts& weight, mfq_tensor_backend::Tensor input,
    const MoeRoutePlan& route, bool prequantized = false, int epilogue = 0);
mfq_tensor_backend::Tensor glu_experts(
    CudaExecutionContext& execution, const Experts& weight, mfq_tensor_backend::Tensor input,
    const MoeRoutePlan& route, bool gelu);
mfq_tensor_backend::Tensor gated_experts(
    CudaExecutionContext& execution, const Experts& weight, mfq_tensor_backend::Tensor input,
    const MoeRoutePlan& route, bool gelu);
mfq_tensor_backend::Tensor clamped_experts(
    CudaExecutionContext& execution, const Experts& weight, mfq_tensor_backend::Tensor input,
    const MoeRoutePlan& route, double limit);

} // namespace mfq::cuda
