#pragma once

#include "cuda_runtime_config.h"
#include "backends/cuda/engine/executor/run.h"
#include "backends/cuda/engine/requests/prompt.h"
#include "engine/executor/sequence.h"
#include "mfq_cuda_kv_quantization.h"
#include "models/qwen35/components.h"
#include "models/qwen35/model.h"
#include "models/qwen35/state.h"

#include <cstdint>
#include <memory>
#include <optional>

class MoeExpertCache;

namespace mfq::engine {
class Batch;
struct InferenceRequest;
}

namespace mfq::cuda::weight_loader {
struct Loader;
}

namespace mfq::cuda {
class Projector;
}

namespace mfq::cuda::qwen35 {

struct Runtime {
    using Model = qwen35::Model;
    using State = qwen35::State;
    using Parts = qwen35::Components;

    std::shared_ptr<Model> model;
    Parts components;

private:
    std::unique_ptr<State> owned_state_;

public:
    std::unique_ptr<Executor> executor;

    Runtime(Model loaded, const weight_loader::Loader& weights,
            std::unique_ptr<Executor> loaded_executor, Collectives& copies,
            const bool& transposed, const int64_t& kv_capacity,
            Trace* const& trace, const int& trace_scope, bool optional,
            bool mtp_allowed, bool preallocate);

    Model& language() { return *model; }
    State& state() { return *owned_state_; }

    // Startup can release idle serial KV before allocating the batch.
    std::unique_ptr<mfq::engine::Batch> batch(
        std::unique_ptr<Executor> batch_executor, Collectives& copies,
        Executor::Mode mode, std::shared_ptr<MoeExpertCache> cache,
        CudaContinuousBatchConfig config, std::int64_t chunk_size,
        bool preallocate, bool cpu_offload, int cached_slots);

    mfq::StepSequence<PreparedPrompt> prepare(
        mfq::engine::InferenceRequest& input, Projector projector,
        std::shared_ptr<MoeExpertCache> cache, Collectives& copies, int primary);

    std::optional<KvQuantization> kv_quantization() const;
};

} // namespace mfq::cuda::qwen35
