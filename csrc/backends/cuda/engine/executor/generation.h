#pragma once
#include "backends/cuda/engine/requests/prompt.h"

#include "engine/batch/batch.h"
#include "engine/executor/step.h"
#include "engine/executor/run.h"
#include "engine/models/common/causal_metadata.h"
#include "backends/cuda/engine/executor/decode_graph.h"
#include "engine/executor/mtp.h"
#include "cuda_runtime_config.h"

#include <functional>
#include <optional>
#include <utility>

namespace mfq::cuda::internal {
class TextSessionCache;
}

namespace mfq::cuda {
class Collectives;
struct RoutedCache;
}

struct CudaProfiler;

namespace mfq::cuda::internal {

// Creation and every resume run inside the owner's CUDA device/context scope.
template <typename Model, typename State, typename Predictor = MtpModule>
mfq::engine::Generation generate(
    Model& model, State& state, mfq::cuda::Executor* executor,
    mfq::cuda::Collectives& collectives, mfq::cuda::Executor::Mode mode,
    CudaProfiler& profiler, mfq::cuda::RoutedCache& weights,
    bool& serial_branches, bool report_memory, bool graph_allowed,
    std::function<void()> drain,
    const mfq::models::CausalLmMetadata& metadata,
    DecodeGraphCache& graph, TextSessionCache& cache,
    const CudaRuntimeConfig& config, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, Predictor* mtp = nullptr,
    std::optional<PreparedPrompt> prepared = {},
    mfq::engine::Batch* batching = nullptr,
    std::string request_id = {}, std::function<void()> restore = {});
} // namespace mfq::cuda::internal

// Creation and every resume run inside the owner's CUDA device/context scope.
template <typename Model, typename State>
mfq::engine::Generation run_mtp_generation(
    Model& model, State& state, mfq::cuda::Executor* executor,
    mfq::cuda::Collectives& collectives, mfq::cuda::Executor::Mode mode,
    mfq::cuda::RoutedCache& weights,
    const mfq::models::CausalLmMetadata& metadata,
    MtpModule& mtp, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, int64_t prefill_chunk_size = 2048,
    const mfq::cuda::PreparedPrompt* prepared = nullptr, std::size_t reused_tokens = 0,
    mfq_tensor_backend::Tensor restored_last_hidden = {},
    mfq_tensor_backend::Tensor* session_last_hidden = nullptr,
    std::size_t stable_prefix_tokens = 0,
    std::function<void()> restore = {});
