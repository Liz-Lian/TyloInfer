#pragma once

#include "ops/attention/full.h"

struct CudaExecutionContext;
struct KVCache;

namespace mfq::cuda {

std::array<mfq_tensor_backend::Tensor, 2> run_attention(
    CudaExecutionContext& execution, const FullAttention& attention,
    KVCache& cache, AttentionScratch& scratch, mfq_tensor_backend::Tensor input,
    const AttentionStep& step, const RopeCache& rope);

} // namespace mfq::cuda
