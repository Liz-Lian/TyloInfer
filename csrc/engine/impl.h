#pragma once

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/engine.h"
#include "engine/load.h"
#include "engine/budget.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/generation/mtp/state.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/session/cache.h"
#include "engine/requests/processor.h"
#include "engine/weights/workspace.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/memory.h"

#include <functional>

namespace mfq::engine {

// Engine private storage keeps device-specific types out of engine.h.
struct Engine::Impl {
    cuda::CudaLoadConfig load_;
    cuda::CudaRuntimeConfig runtime_;
    std::shared_ptr<const ModelSource> source_;
    std::shared_ptr<cuda::WeightMemoryLedger> ledger_;
    std::unique_ptr<cuda::RoutedCache> weights_;
    std::shared_ptr<cuda::KvMemoryManager> kv_;
    std::shared_ptr<cuda::KvOffloadStore> offload_;
    std::unique_ptr<models::State> idle_state_;
    std::unique_ptr<cuda::Executor> forward_;
    models::Info model_info_{};
    mtp::State mtp_state_;
    std::optional<DecodeGraphCache> graph_;
    std::optional<cuda::internal::TextSessionCache> cache_;
    std::optional<cuda::Budget> budget_;
    std::optional<cuda::ExpertSpace> expert_space_;
    std::function<void()> restore_;
    std::unique_ptr<TextProcessor> text_;
    EngineInfo info_;
    bool ready_ = false;
};

} // namespace mfq::engine
#endif
