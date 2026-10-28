#pragma once

#include "backends/cuda/engine/executor/run.h"
#include "models/qwen35/components.h"
#include "models/qwen35/model.h"
#include "models/qwen35/state.h"

#include <cstdint>
#include <memory>

namespace mfq::cuda::weight_loader {
struct Loader;
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
};

} // namespace mfq::cuda::qwen35
