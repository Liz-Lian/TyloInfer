#pragma once

#include "engine/executor/step.h"
#include "engine/models/common/causal_metadata.h"
#include "backends/cuda/engine/batch/batch.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/decode_graph.h"
#include "engine/executor/mtp.h"
#include "cuda_runtime_config.h"

#include <optional>
#include <utility>

namespace mfq::cuda::internal {
class TextSessionCache;

template <typename Model, typename State>
mfq::engine::Generation generate(
    Model& model, State& state, const mfq::models::CausalLmMetadata& metadata,
    DecodeGraphCache& graph, TextSessionCache& cache,
    const CudaRuntimeConfig& config, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, MtpModule* mtp = nullptr,
    std::optional<CudaPreparedPrompt> prepared = {},
    mfq::engine::ContinuousBatch<CudaBatchOperations<Model>>* batching = nullptr,
    std::string request_id = {});
} // namespace mfq::cuda::internal

template <typename Model, typename State>
mfq::engine::Generation run_mtp_generation(
    Model& model, State& state, const mfq::models::CausalLmMetadata& metadata,
    MtpModule& mtp, mfq::engine::InferenceRequest& request,
    mfq::engine::InferenceOutput& output, int64_t prefill_chunk_size = 2048,
    const CudaPreparedPrompt* prepared = nullptr, std::size_t reused_tokens = 0,
    mfq_tensor_backend::Tensor restored_last_hidden = {},
    mfq_tensor_backend::Tensor* session_last_hidden = nullptr,
    std::size_t stable_prefix_tokens = 0);
