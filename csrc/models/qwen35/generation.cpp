#include "models/qwen35/forward.h"
#include "backends/cuda/engine/executor/generation_impl.h"
#include "backends/cuda/engine/executor/mtp_impl.h"

namespace mfq::cuda::internal {
template Generation generate(
    mfq::cuda::qwen35::Model &, mfq::cuda::qwen35::State &, mfq::cuda::Executor*,
    mfq::cuda::Collectives&, mfq::cuda::Executor::Mode, CudaProfiler&,
    mfq::cuda::RoutedCache&, bool&, bool, bool, std::function<void()>,
    const mfq::models::CausalLmMetadata &,
    DecodeGraphCache &, TextSessionCache &, const CudaRuntimeConfig &,
    mfq::engine::InferenceRequest &, mfq::engine::InferenceOutput &, MtpModule *,
    std::optional<PreparedPrompt>, mfq::engine::Batch*, std::string,
    std::function<void()>);
} // namespace mfq::cuda::internal

using mfq::engine::Generation;
using namespace mfq::engine;

template Generation run_mtp_generation<mfq::cuda::qwen35::Model, mfq::cuda::qwen35::State>(
    mfq::cuda::qwen35::Model &, mfq::cuda::qwen35::State &,
    mfq::cuda::Executor*, mfq::cuda::Collectives&, mfq::cuda::Executor::Mode,
    mfq::cuda::RoutedCache&, const mfq::models::CausalLmMetadata &,
    MtpModule &, InferenceRequest &, InferenceOutput &, int64_t,
    const mfq::cuda::PreparedPrompt *, std::size_t, mfq_tensor_backend::Tensor,
    mfq_tensor_backend::Tensor *, std::size_t, std::function<void()>);
