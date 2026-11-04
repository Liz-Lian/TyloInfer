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

namespace tyloi::cuda::internal {
class TextSessionCache;
}

namespace tyloi::cuda {
class Collectives;
struct RoutedCache;
}

namespace tyloi { class ModelSource; }

struct CudaProfiler;

namespace tyloi::cuda::internal {

// Creation and every resume run inside the owner's CUDA device/context scope.
template <typename Model, typename State, typename Predictor = MtpModule>
tyloi::engine::Generation generate(
    Model& model, State& state, tyloi::cuda::Executor* executor,
    tyloi::cuda::Collectives& collectives, tyloi::cuda::Executor::Mode mode,
    CudaProfiler& profiler, tyloi::cuda::RoutedCache& weights,
    bool& serial_branches, bool report_memory, bool graph_allowed,
    std::function<void()> drain,
    const tyloi::models::Info& info, const tyloi::ModelSource* source,
    DecodeGraphCache& graph, TextSessionCache& cache,
    const CudaRuntimeConfig& config, tyloi::engine::InferenceRequest& request,
    tyloi::engine::InferenceOutput& output, tyloi::engine::mtp::State& mtp_state,
    Predictor* mtp = nullptr,
    std::optional<PreparedPrompt> prepared = {},
    tyloi::engine::Batch* batching = nullptr,
    std::string request_id = {}, std::function<void()> restore = {});
} // namespace tyloi::cuda::internal

// Creation and every resume run inside the owner's CUDA device/context scope.
template <typename Model, typename State>
tyloi::engine::Generation run_mtp_generation(
    Model& model, State& state, tyloi::cuda::Executor* executor,
    tyloi::cuda::Collectives& collectives, tyloi::cuda::Executor::Mode mode,
    tyloi::cuda::RoutedCache& weights,
    const tyloi::models::Info& info, const tyloi::ModelSource* source,
    MtpModule& mtp, tyloi::engine::mtp::State& mtp_state,
    tyloi::engine::InferenceRequest& request,
    tyloi::engine::InferenceOutput& output, int64_t prefill_chunk_size = 2048,
    const tyloi::cuda::PreparedPrompt* prepared = nullptr, std::size_t reused_tokens = 0,
    ::tyloi::DenseTensor restored_last_hidden = {},
    ::tyloi::DenseTensor* session_last_hidden = nullptr,
    std::size_t stable_prefix_tokens = 0,
    std::function<void()> restore = {});
