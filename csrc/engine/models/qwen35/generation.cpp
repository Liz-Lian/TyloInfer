#include "engine/weights/cache.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/generate_impl.h"
#include "engine/mtp_impl.h"

namespace tyloi::cuda::internal {
#define TYLOINFER_INSTANTIATE_QWEN35_GENERATION(PREDICTOR) \
template Generation generate( \
    tyloi::cuda::qwen35::Model &, tyloi::cuda::qwen35::State &, tyloi::cuda::Executor*, \
    tyloi::cuda::Collectives&, tyloi::cuda::Executor::Mode, CudaProfiler&, \
    ::tyloi::weights::Cache&, tyloi::engine::Budget&, bool&, bool, bool, std::function<void()>, \
    const tyloi::models::Info &, const tyloi::ModelSource*, \
    DecodeGraphCache &, tyloi::engine::SessionCache<CudaSessionOps> &, const tyloi::engine::Config &, \
    tyloi::engine::InferenceRequest &, tyloi::engine::InferenceOutput &, \
    tyloi::engine::mtp::State &, PREDICTOR *, \
    std::optional<PreparedPrompt>, tyloi::engine::Batch*, std::string, \
    bool);
TYLOINFER_INSTANTIATE_QWEN35_GENERATION(tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_QWEN35_GENERATION(tyloi::cuda::qwen35::Mtp)
#undef TYLOINFER_INSTANTIATE_QWEN35_GENERATION
} // namespace tyloi::cuda::internal

using tyloi::engine::Generation;
using namespace tyloi::engine;

template Generation run_mtp_generation<tyloi::cuda::qwen35::Model, tyloi::cuda::qwen35::State>(
    tyloi::cuda::qwen35::Model &, tyloi::cuda::qwen35::State &,
    tyloi::cuda::Executor*, tyloi::cuda::Collectives&, tyloi::cuda::Executor::Mode,
    ::tyloi::weights::Cache&, const tyloi::models::Info &, const tyloi::ModelSource*,
    tyloi::models::Mtp &, tyloi::engine::mtp::State &, InferenceRequest &, InferenceOutput &, int64_t,
    const tyloi::cuda::PreparedPrompt *, std::size_t, std::shared_ptr<const tyloi::Tensor>,
    std::shared_ptr<tyloi::Tensor> *, std::size_t, tyloi::engine::Budget*);
