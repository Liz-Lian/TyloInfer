#pragma once

#include "engine/config.h"
#include "engine/generation/batch.h"
#include "engine/backends/cuda/executor/executor.h"

#include <cstdint>
#include <memory>

struct CudaExecutionContext;
namespace tyloi::engine { class Budget; }

namespace tyloi::cuda::qwen35 {
struct Model;
class State;
}

namespace tyloi::cuda::qwen4_exp { struct Model; }

namespace tyloi::cuda {

// serial is startup-only: release idle preallocated KV; running independent serial state is omitted.
std::unique_ptr<tyloi::engine::Batch> make_batch(
    std::shared_ptr<const qwen35::Model> model, qwen35::State* serial,
    std::unique_ptr<Executor> executor, Collectives& copies, Executor::Mode mode,
    tyloi::engine::Budget& budget, tyloi::engine::ContinuousBatchConfig config,
    std::int64_t chunk_size, bool preallocate, bool cpu_offload, int cached_slots);

std::unique_ptr<tyloi::engine::Batch> make_batch(
    qwen4_exp::Model& model, CudaExecutionContext& execution, tyloi::engine::Budget& budget,
    const tyloi::engine::ContinuousBatchConfig& config, std::int64_t chunk_size);

} // namespace tyloi::cuda
