#pragma once

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/engine.h"
#include "engine/load.h"
#include "engine/budget.h"
#include "engine/generation/decode.h"
#include "engine/generation/mtp/state.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/session/cache.h"
#include "engine/requests/processor.h"
#include "engine/weights/workspace.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/memory.h"

namespace tyloi::engine {

// Engine private storage keeps device-specific types out of engine.h.
struct Engine::Impl {
    cuda::CudaEngineOptions options_;
    cuda::CudaLoadConfig load_;
    cuda::CudaRuntimeConfig runtime_;
    std::shared_ptr<const ModelSource> source_;
    std::shared_ptr<cuda::WeightMemoryLedger> ledger_;
    std::unique_ptr<cuda::RoutedCache> weights_;
    std::shared_ptr<cuda::KvMemoryManager> kv_;
    std::shared_ptr<cuda::KvOffloadStore> offload_;
    std::unique_ptr<models::State> idle_state_;
    std::unique_ptr<tyloi::Executor> forward_;
    models::ForwardOptions forward_options_{};
    bool graph_allowed_ = false;
    models::Info model_info_{};
    mtp::State mtp_state_;
    std::optional<Decode> graph_;
    std::optional<cuda::internal::TextSessionCache> cache_;
    std::optional<cuda::Budget> budget_;
    std::optional<cuda::ExpertSpace> expert_space_;
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
