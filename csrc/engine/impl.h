#pragma once

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/engine.h"
#include "engine/config.h"
#include "engine/budget.h"
#include "engine/generation/decode.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/session/cache.h"
#include "engine/requests/processor.h"
#include "engine/weights/memory.h"
#include "engine/weights/routed_cache.h"

namespace tyloi::engine {

// Engine private storage keeps device-specific types out of engine.h.
struct Engine::Impl {
    RuntimeConfig runtime_;
    std::vector<Device> devices_;
    bool budget_supported_ = false;
    std::shared_ptr<const ModelSource> source_;
    std::shared_ptr<tyloi::weights::Memory> ledger_;
    std::unique_ptr<cuda::RoutedCache> weights_;
    std::unique_ptr<models::State> idle_state_;
    models::ForwardOptions forward_options_{};
    bool graph_allowed_ = false;
    models::Info model_info_{};
    std::optional<Decode> graph_;
    std::optional<cuda::internal::TextSessionCache> cache_;
    std::optional<cuda::Budget> budget_;
    std::unique_ptr<TextProcessor> text_;
    EngineInfo info_;
    bool ready_ = false;
};

// Everything the CUDA loader assembles before an Engine exists.
struct EngineParts {
    std::shared_ptr<models::CausalModel> model;
    std::unique_ptr<tyloi::Executor> executor;
    std::unique_ptr<Engine::Impl> impl;
};

} // namespace tyloi::engine
#endif
