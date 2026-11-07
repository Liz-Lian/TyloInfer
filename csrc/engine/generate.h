#pragma once
#include "engine/models/common/prompt.h"

#include "engine/generation/batch.h"
#include "engine/generation/mtp/state.h"
#include "engine/generation/step.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/info.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/models/common/mtp.h"
#include "engine/load.h"

#include <functional>
#include <optional>
#include <utility>

namespace tyloi::engine { class Budget; template <class Backend> class SessionCache; }
namespace tyloi::cuda::internal { struct CudaSessionOps; }

namespace tyloi::cuda {
class Collectives;
}

namespace tyloi::weights { class Cache; }
namespace tyloi { class ModelSource; }

struct CudaProfiler;

namespace tyloi::cuda::internal {

// Creation and every resume run inside the owner's CUDA device/context scope.
// The budget is borrowed until the sequence is destroyed.
template <typename Model, typename State, typename Predictor = tyloi::models::Mtp>
tyloi::engine::Generation generate(
    Model& model, State& state, tyloi::cuda::Executor* executor,
    tyloi::cuda::Collectives& collectives, tyloi::cuda::Executor::Mode mode,
    CudaProfiler& profiler, ::tyloi::weights::Cache& weights, tyloi::engine::Budget& budget,
    bool& serial_branches, bool report_memory, bool graph_allowed,
    std::function<void()> drain,
    const tyloi::models::Info& info, const tyloi::ModelSource* source,
    DecodeGraphCache& graph, tyloi::engine::SessionCache<CudaSessionOps>& cache,
    const tyloi::engine::RuntimeConfig& config, tyloi::engine::InferenceRequest& request,
    tyloi::engine::InferenceOutput& output, tyloi::engine::mtp::State& mtp_state,
    Predictor* mtp = nullptr,
    std::optional<PreparedPrompt> prepared = {},
    tyloi::engine::Batch* batching = nullptr,
    std::string request_id = {}, bool decode_ready = false);
} // namespace tyloi::cuda::internal

// Creation and every resume run inside the owner's CUDA device/context scope.
// A supplied budget is borrowed until the sequence is destroyed.
template <typename Model, typename State>
tyloi::engine::Generation run_mtp_generation(
    Model& model, State& state, tyloi::cuda::Executor* executor,
    tyloi::cuda::Collectives& collectives, tyloi::cuda::Executor::Mode mode,
    ::tyloi::weights::Cache& weights,
    const tyloi::models::Info& info, const tyloi::ModelSource* source,
    tyloi::models::Mtp& mtp, tyloi::engine::mtp::State& mtp_state,
    tyloi::engine::InferenceRequest& request,
    tyloi::engine::InferenceOutput& output, int64_t prefill_chunk_size = 2048,
    const tyloi::cuda::PreparedPrompt* prepared = nullptr, std::size_t reused_tokens = 0,
    std::shared_ptr<const tyloi::Tensor> restored_last_hidden = {},
    std::shared_ptr<tyloi::Tensor>* session_last_hidden = nullptr,
    std::size_t stable_prefix_tokens = 0,
    tyloi::engine::Budget* budget = nullptr);
