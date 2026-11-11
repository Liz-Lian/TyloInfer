#include "engine/weights/cache.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/mtp.h"

#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/models/glm_dsa/forward.h"
#include "engine/models/deepseek_v4/forward.h"
#include "engine/models/glm5_next/forward.h"
#include "engine/models/minicpmo45/text/forward.h"
#include "engine/models/minicpmo45/forward.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/qwen4_exp/generation.h"
#include "engine/models/deepseek_v41/generation.h"

#endif

#include "engine/generation/cuda_impl.h"

namespace tyloi::cuda::internal {
#define TYLOINFER_INSTANTIATE_FLOW(MODEL, STATE, PREDICTOR)                                                                \
    template Generation generate(MODEL &, STATE &, tyloi::cuda::Executor*, tyloi::cuda::Collectives&, \
        tyloi::cuda::Executor::Mode, CudaProfiler&, tyloi::weights::Cache&, tyloi::engine::Budget&, bool&, bool, bool, \
        std::function<void()>,                                                                      \
        const tyloi::models::Info &, const tyloi::ModelSource*,                                          \
        DecodeGraphCache &,                                                                        \
        tyloi::engine::SessionCache<tyloi::cuda::internal::CudaSessionOps> &,                       \
        const tyloi::engine::Config &,                                                      \
        tyloi::engine::InferenceRequest &,                                                           \
        tyloi::engine::InferenceOutput &,                                                            \
        tyloi::engine::mtp::State &,                                                                  \
        PREDICTOR *,                                                                      \
        std::optional<tyloi::models::Prepared>,                                                         \
        tyloi::engine::Batch*, std::string);
#ifndef TYLOINFER_NATIVE_CUDA
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::Model, tyloi::cuda::minicpmo45::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::text::Model, tyloi::cuda::minicpmo45::text::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm5_next::Model, tyloi::cuda::glm5_next::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::qwen4_exp::Model, tyloi::cuda::qwen4_exp::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::deepseek_v41::Model, tyloi::cuda::deepseek_v41::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::nemotron_h::Model, tyloi::cuda::nemotron_h::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm_dsa::Model, tyloi::cuda::glm_dsa::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::deepseek_v4::Model, tyloi::cuda::deepseek_v4::State, tyloi::models::Mtp)
#endif
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::qwen35::Model, tyloi::cuda::qwen35::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::qwen35::Model, tyloi::cuda::qwen35::State, tyloi::cuda::qwen35::Mtp)
#undef TYLOINFER_INSTANTIATE_FLOW
} // namespace tyloi::cuda::internal
