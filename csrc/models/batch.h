#pragma once

#include "cuda_runtime_config.h"
#include "engine/batch/batch.h"
#include "backends/cuda/engine/executor/run.h"
#include "backends/cuda/models/common/causal_model_ops.h"

#include <cstdint>
#include <memory>

class MoeExpertCache;

namespace mfq::cuda::qwen35 {
struct Model;
class State;
}

namespace mfq::cuda {

// serial is startup-only: release idle preallocated KV; running independent serial state is omitted.
std::unique_ptr<mfq::engine::Batch> make_batch(
    std::shared_ptr<const qwen35::Model> model, qwen35::State* serial,
    std::unique_ptr<Executor> executor, Collectives& copies, Executor::Mode mode,
    std::shared_ptr<MoeExpertCache> cache, CudaContinuousBatchConfig config,
    std::int64_t chunk_size, bool preallocate, bool cpu_offload, int cached_slots);

std::unique_ptr<mfq::engine::Batch> make_batch(
    Qwen4CausalLm& model, CudaExecutionContext& execution,
    const CudaContinuousBatchConfig& config, std::int64_t chunk_size);

} // namespace mfq::cuda
