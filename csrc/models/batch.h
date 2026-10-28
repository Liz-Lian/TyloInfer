#pragma once

#include "cuda_runtime_config.h"
#include "engine/batch/batch.h"
#include "models/qwen35/batch.h"
#include "backends/cuda/models/common/causal_model_ops.h"

#include <cstdint>
#include <memory>

class MoeExpertCache;

namespace mfq::cuda {

std::unique_ptr<mfq::engine::Batch> make_batch(
    std::shared_ptr<const qwen35::Model> model, std::unique_ptr<Executor> executor,
    Collectives& copies, Executor::Mode mode, qwen35::Batch::Options options,
    std::shared_ptr<MoeExpertCache> cache, CudaContinuousBatchConfig config,
    std::int64_t chunk_size);

std::unique_ptr<mfq::engine::Batch> make_batch(
    Qwen4CausalLm& model, CudaExecutionContext& execution,
    const CudaContinuousBatchConfig& config, std::int64_t chunk_size);

} // namespace mfq::cuda
