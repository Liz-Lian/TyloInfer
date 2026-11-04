#include "engine/models/deepseek_v4/forward.h"
#include "engine/models/gemma4/forward.h"
#include "engine/models/glm5_next/forward.h"
#include "engine/models/glm_dsa/forward.h"
#include "engine/models/minicpmo45/text/forward.h"
#include "engine/models/minicpmo45/forward.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/qwen4_exp/generation.h"
#include "engine/models/deepseek_v41/generation.h"

#include "engine/generate_impl.h"

namespace tyloi::cuda::internal {
#define TYLOINFER_INSTANTIATE_FLOW(MODEL, STATE)                                                                \
    template Generation generate(MODEL &, STATE &, tyloi::cuda::Executor*, tyloi::cuda::Collectives&, \
        tyloi::cuda::Executor::Mode, CudaProfiler&, tyloi::cuda::RoutedCache&, bool&, bool, bool,       \
        std::function<void()>,                                                                      \
        const tyloi::models::Info &, const tyloi::ModelSource*,                                          \
        DecodeGraphCache &,                                                                        \
        TextSessionCache &,                                                                        \
        const CudaRuntimeConfig &,                                                                 \
        tyloi::engine::InferenceRequest &,                                                           \
        tyloi::engine::InferenceOutput &,                                                            \
        tyloi::engine::mtp::State &,                                                                  \
        MtpModule *,                                                                               \
        std::optional<PreparedPrompt>,                                                         \
        tyloi::engine::Batch*, std::string,                    \
        std::function<void()>);
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::Model, tyloi::cuda::minicpmo45::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::text::Model, tyloi::cuda::minicpmo45::text::State)
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
TYLOINFER_INSTANTIATE_FLOW(tyloi::models::CausalModel, tyloi::models::State)
#else
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::gemma4::Model, tyloi::cuda::gemma4::State)
#endif
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm_dsa::Model, tyloi::cuda::glm_dsa::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm5_next::Model, tyloi::cuda::glm5_next::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::Qwen4CausalLm, tyloi::cuda::Qwen4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::deepseek_v4::Model, tyloi::cuda::deepseek_v4::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::DeepseekV41CausalLm, tyloi::cuda::DeepseekV41CausalLm)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::nemotron_h::Model, tyloi::cuda::nemotron_h::State)
#undef TYLOINFER_INSTANTIATE_FLOW
} // namespace tyloi::cuda::internal
