#include "engine/models/deepseek_v4/forward.h"
#include "engine/models/gemma4/forward.h"
#include "engine/models/glm5_next/forward.h"
#include "engine/models/glm_dsa/forward.h"
#include "engine/models/minicpmo45/ops.h"
#include "engine/models/minicpmo45/forward.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/qwen4_exp/generation.h"
#include "engine/models/deepseek_v41/generation.h"

#include "engine/generate_impl.h"

namespace mfq::cuda::internal {
#define TYLOINFER_INSTANTIATE_FLOW(MODEL, STATE)                                                                \
    template Generation generate(MODEL &, STATE &, mfq::cuda::Executor*, mfq::cuda::Collectives&, \
        mfq::cuda::Executor::Mode, CudaProfiler&, mfq::cuda::RoutedCache&, bool&, bool, bool,       \
        std::function<void()>,                                                                      \
        const mfq::models::CausalLmMetadata &,                                                       \
        DecodeGraphCache &,                                                                        \
        TextSessionCache &,                                                                        \
        const CudaRuntimeConfig &,                                                                 \
        mfq::engine::InferenceRequest &,                                                           \
        mfq::engine::InferenceOutput &,                                                            \
        mfq::engine::mtp::State &,                                                                  \
        MtpModule *,                                                                               \
        std::optional<PreparedPrompt>,                                                         \
        mfq::engine::Batch*, std::string,                    \
        std::function<void()>);
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::minicpmo45::Model, mfq::cuda::minicpmo45::State)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::MiniCPMO45CausalLm, mfq::cuda::MiniCPMO45CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::gemma4::Model, mfq::cuda::gemma4::State)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::glm_dsa::Model, mfq::cuda::glm_dsa::State)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::glm5_next::Model, mfq::cuda::glm5_next::State)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::Qwen4CausalLm, mfq::cuda::Qwen4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::deepseek_v4::Model, mfq::cuda::deepseek_v4::State)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::DeepseekV41CausalLm, mfq::cuda::DeepseekV41CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::nemotron_h::Model, mfq::cuda::nemotron_h::State)
#undef TYLOINFER_INSTANTIATE_FLOW
} // namespace mfq::cuda::internal
